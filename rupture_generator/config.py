"""A rupture described in a TOML file, and the inputs :func:`generate` takes built from it.

This module is an adapter over the library and nothing in the library imports it.
The default models of :class:`~rupture_generator.rupture.models.RuptureModel`, the
correlation relation and the jump model are the library's own classes. A subclass makes
each one strict about unknown keys, and the library still defines every default once. A
subclass restates a field only to read a depth profile where the library takes any
function of position. A small class here builds whatever the file describes
declaratively and the library takes as a function: a depth profile, or a medium.

The file configures the default models alone. Any other model comes in through the
library.

A ``[segments.<name>]`` table overrides the global settings on one segment. Its
``slip``, ``rise``, ``rake``, ``speed`` and ``perturbation`` tables each start from the
global table and replace the keys they give. A depth profile or a noise list, one key
of its table, replaces the global one whole. :func:`load` does this merge. A loaded config stores each segment's tables in
full.

:func:`load` reads and checks a file, and :func:`build` turns it into a
:class:`Scenario`. :func:`dump` writes a loaded config back out with every default
filled in, which records what a run used.
"""

import dataclasses
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from mashumaro.codecs import BasicDecoder
from mashumaro.codecs.toml import TOMLEncoder
from mashumaro.config import BaseConfig
from mashumaro.exceptions import (
    ExtraKeysError,
    InvalidFieldValue,
    MissingField,
    SuitableVariantNotFoundError,
)
from mashumaro.mixins.toml import TOMLDialect
from mashumaro.types import Discriminator

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.geometry import (
    NSHM_ALIASES,
    Geometry,
    geometry_from_geojson,
    projected_crs,
)
from rupture_generator.rupture.generator import check_seed, moment_rng
from rupture_generator.rupture.medium import (
    Layers,
    Medium,
    SpatialField,
    constant_field,
    interpolated_field,
    layered_medium,
    ramp_field,
)
from rupture_generator.rupture.models import (
    MODELS,
    CorrelatedPerturbation,
    CorrelatedRise,
    DrawnSlip,
    GaussianRake,
    NoiseFamily,
    PulseSettings,
    RuptureModel,
    ShearFractionSpeed,
)
from rupture_generator.rupture.propagator import JumpModel, likeliest_path, sample_path
from rupture_generator.rupture.realisation import Hypocentre, Realisation
from rupture_generator.rupture.source import (
    GENSLIP_MAI_BEROZA,
    GENSLIP_SUZUKI_2022,
    MOMENT_SIGMA_LN,
    CorrelationRelation,
    SegmentSource,
    component_weights,
    moment_from_magnitude,
    rigidity_weighted_areas,
    segment_source,
    split_moment,
)


class _Strict(BaseConfig):
    forbid_extra_keys = True


# ------------------------------------------- the library's own settings, made strict


@dataclasses.dataclass(frozen=True)
class Correlation(CorrelationRelation):
    """A noise family's ``relation`` table, genslip's Mai & Beroza (2002) when absent.

    The lengths are a von Karman correlation's. genslip's own lengths are ``2 pi``
    longer than the patches it draws. Its offsets read here grown by ``log10(2 pi)``.

    Parameters
    ----------
    strike_offset : float
        The along-strike length's offset, genslip's grown by ``log10(2 pi)``.
    dip_offset : float
        The down-dip length's offset, genslip's grown by ``log10(2 pi)``.
    """

    strike_offset: float = GENSLIP_MAI_BEROZA.strike_offset
    dip_offset: float = GENSLIP_MAI_BEROZA.dip_offset
    Config = _Strict


@dataclasses.dataclass(frozen=True)
class Jump(JumpModel):
    """A propagation's ``model`` table, Shaw & Dieterich (2007) when absent.

    When the file doesn't give a tree, the model picks one. It also sets the distance each
    jump can cover.
    """

    Config = _Strict


# ---------------------------------------------------------------- depth profiles


class _DepthProfile:
    """A profile the file describes, read at positions as a :data:`SpatialField`."""

    def field(self) -> SpatialField:
        """Build the profile as a function of position.

        Returns
        -------
        SpatialField
            The profile at each position.
        """
        raise NotImplementedError

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """Read the profile at positions, as a :data:`SpatialField` does.

        Parameters
        ----------
        positions : numpy.ndarray
            Positions ``(..., 3)``, depth last in kilometres.

        Returns
        -------
        numpy.ndarray
            The profile at each position ``(...)``.
        """
        return self.field()(positions)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Constant(_DepthProfile):
    """The same value at every depth.

    Parameters
    ----------
    type : {"constant"}
        The profile's kind.
    value : float
        The value everywhere.
    """

    type: Literal["constant"] = "constant"
    value: float
    Config = _Strict

    def field(self) -> SpatialField:
        """Build the profile as a function of position.

        Returns
        -------
        SpatialField
            The value at every position.
        """
        return constant_field(self.value)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Profile(_DepthProfile):
    """Values at depths, linear between them and flat beyond.

    Parameters
    ----------
    type : {"profile"}
        The profile's kind.
    depth_km : list of float
        The depths, in kilometres, increasing.
    values : list of float
        One value per depth.
    """

    type: Literal["profile"] = "profile"
    depth_km: tuple[float, ...]
    values: tuple[float, ...]
    Config = _Strict

    def field(self) -> SpatialField:
        """Build the profile as a function of position.

        Returns
        -------
        SpatialField
            The values interpolated at each position's depth.
        """
        return interpolated_field(np.array(self.depth_km), np.array(self.values))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Ramp(_DepthProfile):
    """One value at shallow depths and another at deep ones, linear between.

    Parameters
    ----------
    type : {"ramp"}
        The profile's kind.
    centre_km : float
        The depth of the ramp's midpoint, in kilometres.
    half_width_km : float
        Half the ramp's depth range, in kilometres. Positive.
    shallow : float
        The value shallower than ``centre_km - half_width_km``.
    deep : float
        The value below ``centre_km + half_width_km``.
    """

    type: Literal["ramp"] = "ramp"
    centre_km: float
    half_width_km: float
    shallow: float
    deep: float
    Config = _Strict

    def field(self) -> SpatialField:
        """Build the profile as a function of position.

        Returns
        -------
        SpatialField
            The ramp at each position's depth.
        """
        return ramp_field(self.centre_km, self.half_width_km, self.shallow, self.deep)


type DepthProfile = Annotated[
    Constant | Profile | Ramp, Discriminator(field="type", include_supertypes=True)
]

UNMODIFIED = Constant(value=1.0)
"""The profile that leaves what it scales unmodified."""


# ----------------------------------------------------------- the default models


@dataclasses.dataclass(frozen=True)
class Family(NoiseFamily):
    """``[[<model>.noise]]``: :class:`~rupture_generator.rupture.models.NoiseFamily`.

    Parameters
    ----------
    relation : Correlation
        ``relation``, the family's correlation lengths.
    weight : DepthProfile
        ``weight``, how much of the field the family is.
    """

    relation: Correlation = dataclasses.field(default_factory=Correlation)
    weight: DepthProfile = UNMODIFIED
    Config = _Strict


STATIONARY = (Family(),)
"""genslip's Mai & Beroza (2002) lengths everywhere: rake's and the perturbation's default."""

GP_HYBRID = (
    Family(
        relation=Correlation(**dataclasses.asdict(GENSLIP_SUZUKI_2022)),
        weight=Ramp(centre_km=6.5, half_width_km=1.5, shallow=1.0, deep=0.0),
    ),
    Family(weight=Ramp(centre_km=6.5, half_width_km=1.5, shallow=0.0, deep=1.0)),
)
"""Graves & Pitarka's hybrid as genslip v5.6.3 configures it: slip's default."""


@dataclasses.dataclass(frozen=True)
class Slip(DrawnSlip):
    """``[slip]``: :class:`~rupture_generator.rupture.models.DrawnSlip`.

    Parameters
    ----------
    noise : tuple of Family
        ``[[slip.noise]]``, slip's noise.
    """

    noise: tuple[Family, ...] = GP_HYBRID
    Config = _Strict


@dataclasses.dataclass(frozen=True)
class Rise(CorrelatedRise):
    """``[rise]``: :class:`~rupture_generator.rupture.models.CorrelatedRise`.

    Parameters
    ----------
    factor : DepthProfile
        ``[rise.factor]``, the relative rise time.
    slip_weight : DepthProfile
        ``[rise.slip_weight]``, how much of rise time's correlation with slip is the
        configured value.
    """

    factor: DepthProfile = UNMODIFIED
    slip_weight: DepthProfile = UNMODIFIED
    Config = _Strict


@dataclasses.dataclass(frozen=True)
class Rake(GaussianRake):
    """``[rake]``: :class:`~rupture_generator.rupture.models.GaussianRake`.

    Parameters
    ----------
    noise : tuple of Family
        ``[[rake.noise]]``, rake's noise.
    """

    noise: tuple[Family, ...] = STATIONARY
    Config = _Strict


@dataclasses.dataclass(frozen=True)
class Speed(ShearFractionSpeed):
    """``[speed]``: :class:`~rupture_generator.rupture.models.ShearFractionSpeed`.

    Parameters
    ----------
    factor : DepthProfile
        ``[speed.factor]``, the factor on the front's speed.
    """

    factor: DepthProfile = UNMODIFIED
    Config = _Strict


@dataclasses.dataclass(frozen=True)
class Perturbation(CorrelatedPerturbation):
    """``[perturbation]``: :class:`~rupture_generator.rupture.models.CorrelatedPerturbation`.

    Parameters
    ----------
    noise : tuple of Family, optional
        ``[[perturbation.noise]]``, the displacement's own noise. Stationary when
        absent.
    """

    noise: tuple[Family, ...] | None = STATIONARY
    Config = _Strict


# ----------------------------------------------------------------------- tables


@dataclasses.dataclass(frozen=True)
class GeometryConfig:
    """``[geometry]``: where the fault sections are, and how to grid them.

    Parameters
    ----------
    path : Path
        The GeoJSON file of fault sections. A relative path is relative to the config
        file.
    crs : str
        The projected frame to build the charts in, as an EPSG code or anything else
        :class:`pyproj.CRS` reads. It has no default. It must measure in metres and
        cover the faults' region, as their UTM zone does.
    spacing_km : float
        The subfault size in kilometres.
    aliases : {"nshm"}, optional
        Read the sections' properties under the New Zealand NSHM's names.
    """

    path: Path
    crs: str
    spacing_km: float
    aliases: Literal["nshm"] | None = None
    Config = _Strict


@dataclasses.dataclass(frozen=True, kw_only=True)
class HypocentreConfig:
    """``[hypocentre]``: a segment, and a position on it.

    Give the position as arc lengths or as fractions of the segment's extent, one
    complete pair of either.

    Parameters
    ----------
    segment : str
        The segment the rupture starts on.
    strike_km : float, optional
        The distance along strike from the segment's start, in kilometres.
    dip_km : float, optional
        The distance down dip from the segment's top edge, in kilometres.
    strike_fraction : float, optional
        The position along strike, as a fraction of the segment's length.
    dip_fraction : float, optional
        The position down dip, as a fraction of the segment's width.
    """

    segment: str
    strike_km: float | None = None
    dip_km: float | None = None
    strike_fraction: float | None = None
    dip_fraction: float | None = None
    Config = _Strict

    def __post_init__(self) -> None:
        """Refuse anything but exactly one complete pair."""
        lengths = (self.strike_km, self.dip_km)
        fractions = (self.strike_fraction, self.dip_fraction)
        given = [None not in pair for pair in (lengths, fractions)]
        if given.count(True) != 1 or any(
            (None in pair) and pair != (None, None) for pair in (lengths, fractions)
        ):
            raise RuptureGeneratorError(
                "a hypocentre is strike_km and dip_km, or strike_fraction and "
                "dip_fraction: one pair, complete"
            )


@dataclasses.dataclass(frozen=True)
class MediumConfig:
    """``[medium]``: the rock, as a 1-D model with one value per layer.

    Parameters
    ----------
    bottom_depth_km : list of float
        Each layer's lower boundary, in kilometres, increasing.
    shear_speed_km_s : list of float
        Each layer's shear speed, in kilometres per second.
    density_g_cm3 : list of float
        Each layer's density, in grams per cubic centimetre.
    """

    bottom_depth_km: list[float]
    shear_speed_km_s: list[float]
    density_g_cm3: list[float]
    Config = _Strict

    def medium(self) -> Medium:
        """Build the medium the model describes.

        Returns
        -------
        Medium
            Shear speed and density as functions of position.
        """
        return layered_medium(
            Layers(np.array(self.bottom_depth_km)),
            np.array(self.shear_speed_km_s),
            np.array(self.density_g_cm3),
        )


@dataclasses.dataclass(frozen=True)
class PulseConfig:
    """``[pulse]``: the slip-rate sample interval and the pulse shape.

    Parameters
    ----------
    dt_s : float
        The slip-rate sample interval, in seconds.
    beta : DepthProfile, optional
        The Liu-Archuleta-Hartzell rising fraction. Without it, every pulse is a
        single-sample impulse.
    """

    dt_s: float = 0.005
    beta: DepthProfile | None = None
    Config = _Strict

    def __post_init__(self) -> None:
        """Refuse a sample interval that samples nothing."""
        if not self.dt_s > 0.0:
            raise RuptureGeneratorError(f"a sample interval of {self.dt_s} s is none")


@dataclasses.dataclass(frozen=True, kw_only=True)
class PerFault:
    """A magnitude and a rake for every segment, as a hazard model states them.

    Parameters
    ----------
    type : {"per_fault"}
        The source's kind.
    magnitudes : dict of str to float
        Each segment's moment magnitude, by name.
    rakes : dict of str to float
        Each segment's mean rake in degrees, by name.
    """

    type: Literal["per_fault"] = "per_fault"
    magnitudes: dict[str, float]
    rakes: dict[str, float]
    Config = _Strict


@dataclasses.dataclass(frozen=True, kw_only=True)
class ReferenceApportionment:
    """Each component's moment its share by area and rigidity, the same every draw.

    Parameters
    ----------
    type : {"reference"}
        The apportionment's kind.
    """

    type: Literal["reference"] = "reference"
    Config = _Strict


@dataclasses.dataclass(frozen=True, kw_only=True)
class SampledApportionment:
    """Each component's moment drawn about its share, lognormally, from the seed.

    Parameters
    ----------
    type : {"sampled"}
        The apportionment's kind.
    sigma_ln : float
        The standard deviation of each component's moment about its share, in
        natural-log units. Bradley & Faulkner's calibration when absent.
    """

    type: Literal["sampled"] = "sampled"
    sigma_ln: float = MOMENT_SIGMA_LN
    Config = _Strict

    def __post_init__(self) -> None:
        """Refuse a standard deviation that isn't a non-negative number."""
        if not (np.isfinite(self.sigma_ln) and self.sigma_ln >= 0.0):
            raise RuptureGeneratorError(
                f"sigma_ln is a non-negative number, got {self.sigma_ln}"
            )


type ApportionmentConfig = Annotated[
    ReferenceApportionment | SampledApportionment,
    Discriminator(field="type", include_supertypes=True),
]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Finite:
    """One magnitude and rake for the whole event.

    :func:`~rupture_generator.rupture.source.split_moment` shares the event's moment
    between its segments.

    Parameters
    ----------
    type : {"finite"}
        The source's kind.
    magnitude : float
        The event's moment magnitude.
    rake_deg : float
        The mean rake in degrees, on every segment.
    apportionment : ReferenceApportionment or SampledApportionment
        ``[source.apportionment]``, how the moment shares between components.
        ``reference`` when absent.
    """

    type: Literal["finite"] = "finite"
    magnitude: float
    rake_deg: float
    apportionment: ApportionmentConfig = ReferenceApportionment()
    Config = _Strict


type SourceConfig = Annotated[
    PerFault | Finite, Discriminator(field="type", include_supertypes=True)
]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Predetermined:
    """Which segment triggers which, as a scenario states it.

    Parameters
    ----------
    type : {"predetermined"}
        The propagation's kind.
    parents : dict of str to str
        Each triggered segment's parent, by name. The segment holding the hypocentre
        has none.
    model : Jump
        How far each jump can cover.
    """

    type: Literal["predetermined"] = "predetermined"
    parents: dict[str, str]
    model: Jump = dataclasses.field(default_factory=Jump)
    Config = _Strict


@dataclasses.dataclass(frozen=True, kw_only=True)
class Sampled:
    """A tree drawn from the jump model.

    Parameters
    ----------
    type : {"sampled"}
        The propagation's kind.
    model : Jump
        The jump model to draw the tree from.
    """

    type: Literal["sampled"] = "sampled"
    model: Jump = dataclasses.field(default_factory=Jump)
    Config = _Strict


@dataclasses.dataclass(frozen=True, kw_only=True)
class Likeliest:
    """The jump model's likeliest tree.

    Parameters
    ----------
    type : {"likeliest"}
        The propagation's kind.
    model : Jump
        The jump model to find the tree in.
    """

    type: Literal["likeliest"] = "likeliest"
    model: Jump = dataclasses.field(default_factory=Jump)
    Config = _Strict


type PropagationConfig = Annotated[
    Predetermined | Sampled | Likeliest,
    Discriminator(field="type", include_supertypes=True),
]


@dataclasses.dataclass(frozen=True)
class SegmentConfig:
    """``[segments.<name>]``: one segment's settings, where they differ.

    A table left out is the global one. :func:`load` fills each table given in from
    the global one.

    Parameters
    ----------
    slip : Slip, optional
        ``[segments.<name>.slip]``.
    rise : Rise, optional
        ``[segments.<name>.rise]``.
    rake : Rake, optional
        ``[segments.<name>.rake]``.
    speed : Speed, optional
        ``[segments.<name>.speed]``.
    perturbation : Perturbation, optional
        ``[segments.<name>.perturbation]``.
    """

    slip: Slip | None = None
    rise: Rise | None = None
    rake: Rake | None = None
    speed: Speed | None = None
    perturbation: Perturbation | None = None
    Config = _Strict


@dataclasses.dataclass(frozen=True, kw_only=True)
class RuptureConfig:
    """A whole rupture file.

    Parameters
    ----------
    seed : int
        The seed every random draw derives from.
    geometry : GeometryConfig
        ``[geometry]``.
    hypocentre : HypocentreConfig
        ``[hypocentre]``.
    medium : MediumConfig
        ``[medium]``.
    source : PerFault or Finite
        ``[source]``.
    propagation : Predetermined or Sampled or Likeliest, optional
        ``[propagation]``, which a single-segment rupture can leave out.
    slip : Slip
        ``[slip]``.
    rise : Rise
        ``[rise]``.
    rake : Rake
        ``[rake]``.
    speed : Speed
        ``[speed]``.
    perturbation : Perturbation
        ``[perturbation]``.
    pulse : PulseConfig
        ``[pulse]``.
    segments : dict of str to SegmentConfig
        ``[segments.<name>]``, each segment's own settings by name.
    """

    seed: int = dataclasses.field(metadata={"deserialize": check_seed})
    geometry: GeometryConfig
    hypocentre: HypocentreConfig
    medium: MediumConfig
    source: SourceConfig
    propagation: PropagationConfig | None = None
    slip: Slip = dataclasses.field(default_factory=Slip)
    rise: Rise = dataclasses.field(default_factory=Rise)
    rake: Rake = dataclasses.field(default_factory=Rake)
    speed: Speed = dataclasses.field(default_factory=Speed)
    perturbation: Perturbation = dataclasses.field(default_factory=Perturbation)
    pulse: PulseConfig = dataclasses.field(default_factory=PulseConfig)
    segments: dict[str, SegmentConfig] = dataclasses.field(default_factory=dict)
    Config = _Strict


_DECODER = BasicDecoder(RuptureConfig, default_dialect=TOMLDialect)
_SEGMENT_DECODER = BasicDecoder(SegmentConfig, default_dialect=TOMLDialect)
_ENCODER = TOMLEncoder(RuptureConfig)


# ------------------------------------------------------------------ the boundary


def _describe(error: Exception, within: str | None = None) -> RuptureGeneratorError:
    """Build one error that states the key at fault, from mashumaro's chain of them.

    Mashumaro wraps each level of nesting in its own exception, so the key path is the
    chain's field names, and the reason is at its far end. ``within`` prefixes the
    path.
    """
    path: list[str] = [] if within is None else [within]
    reason = str(error)
    link: BaseException | None = error
    while link is not None:
        match link:
            case MissingField():
                path.append(link.field_name)
                reason = "is missing"
                break
            case ExtraKeysError():
                reason = f"has unknown keys {sorted(link.extra_keys)}"
                break
            case SuitableVariantNotFoundError():
                reason = (
                    f"has no variant with {link.discriminator_name} = "
                    f"{link.discriminator_value!r}"
                )
                break
            case InvalidFieldValue():
                path.append(link.field_name)
                reason = f"{link.field_value!r} is not a {getattr(link.field_type, '__name__', link.field_type)}"
            case RuptureGeneratorError() | ValueError() | TypeError():
                reason = str(link)
                break
        link = link.__cause__ or link.__context__
    return RuptureGeneratorError(f"{'.'.join(path) or 'the file'}: {reason}")


def _table(table: object) -> dict[str, Any]:
    return table if isinstance(table, dict) else {}


def _merged(document: dict[str, Any], overrides: object) -> dict[str, Any]:
    """Fill one ``[segments.<name>]`` table in from the global tables it overrides.

    Each of its tables replaces the keys it gives, so a depth profile, which is one key
    of its table, replaces the global profile whole.

    Raises
    ------
    RuptureGeneratorError
        If the segment's entry isn't a table.
    """
    if not isinstance(overrides, dict):
        raise RuptureGeneratorError(f"{overrides!r} is not a table")
    shared = {key: _table(document.get(key)) for key in overrides}
    return {
        key: {**shared[key], **table} if isinstance(table, dict) else table
        for key, table in overrides.items()
    }


def _segments(document: dict[str, Any]) -> dict[str, SegmentConfig]:
    """Decode the ``[segments]`` tables, each filled in from the global ones.

    Raises
    ------
    RuptureGeneratorError
        If a segment's table isn't one, or has a key or value its setting refuses.
    """
    tables = document.get("segments", {})
    if not isinstance(tables, dict):
        raise RuptureGeneratorError(f"segments: {tables!r} is not a table")
    segments = {}
    for name, overrides in tables.items():
        try:
            segments[name] = _SEGMENT_DECODER.decode(_merged(document, overrides))
        except (
            RuptureGeneratorError,
            InvalidFieldValue,
            MissingField,
            ExtraKeysError,
        ) as error:
            raise _describe(error, f"segments.{name}") from error
    return segments


def load(path: str | Path) -> RuptureConfig:
    """Read and check a rupture file.

    Parameters
    ----------
    path : str or Path
        The rupture file.

    Returns
    -------
    RuptureConfig
        The file's contents, with the geometry's path resolved against the file's
        directory.

    Raises
    ------
    RuptureGeneratorError
        If the file isn't TOML, or describes no rupture.
    OSError
        If reading the file fails.
    """
    path = Path(path)
    try:
        document = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise RuptureGeneratorError(f"{path} is not TOML: {error}") from None
    try:
        config = _DECODER.decode({**document, "segments": {}})
    except (InvalidFieldValue, MissingField, ExtraKeysError) as error:
        raise _describe(error) from None
    geometry = dataclasses.replace(
        config.geometry, path=(path.parent / config.geometry.path).resolve()
    )
    return dataclasses.replace(config, geometry=geometry, segments=_segments(document))


def dump(config: RuptureConfig) -> str:
    """Write a config as TOML, every default filled in.

    Parameters
    ----------
    config : RuptureConfig
        The config to write.

    Returns
    -------
    str
        The TOML text.
    """
    return _ENCODER.encode(config)


# --------------------------------------------------------------------- building


@dataclasses.dataclass(frozen=True, eq=False)
class Scenario:
    """Everything that :func:`~rupture_generator.rupture.generator.generate` takes.

    Parameters
    ----------
    realisation : Realisation of Geometry
        The fault system, its hypocentre and its triggering tree.
    medium : Medium
        The rock.
    sources : dict of str to SegmentSource
        Each segment's source, by name.
    model : RuptureModel or dict of str to RuptureModel
        How to draw each field: one for every segment, or one per segment by name
        when the config has ``[segments]`` tables.
    jump_model : JumpModel
        How far each jump can cover.
    seed : int
        The seed every random draw derives from.
    """

    realisation: Realisation[Geometry]
    medium: Medium
    sources: dict[str, SegmentSource]
    model: RuptureModel | dict[str, RuptureModel]
    jump_model: JumpModel
    seed: int


def _sources(
    source: PerFault | Finite,
    realisation: Realisation[Geometry],
    medium: Medium,
    seed: int,
) -> dict[str, SegmentSource]:
    names = set(realisation)
    match source:
        case PerFault():
            for what, given in (
                ("magnitude", source.magnitudes),
                ("rake", source.rakes),
            ):
                if set(given) != names:
                    raise RuptureGeneratorError(
                        f"source: every segment needs a {what}; missing "
                        f"{sorted(names - set(given))}, unknown {sorted(set(given) - names)}"
                    )
            return {
                name: segment_source(source.magnitudes[name], source.rakes[name])
                for name in realisation
            }
        case Finite():
            moment_nm = moment_from_magnitude(source.magnitude)
            segment_weights = rigidity_weighted_areas(realisation, medium)
            weights = component_weights(realisation, segment_weights)
            match source.apportionment:
                case ReferenceApportionment():
                    shares = split_moment(moment_nm, weights, segment_weights)
                case SampledApportionment():
                    shares = split_moment(
                        moment_nm,
                        weights,
                        segment_weights,
                        sigma_ln=source.apportionment.sigma_ln,
                        rng=moment_rng(seed),
                    )
            return {
                name: SegmentSource(share_nm, source.rake_deg)
                for name, share_nm in shares.items()
            }


def _propagated(
    propagation: Predetermined | Sampled | Likeliest | None,
    realisation: Realisation[Geometry],
    hypocentre: Hypocentre,
    seed: int,
) -> Realisation[Geometry]:
    match propagation:
        case None:
            if len(realisation) > 1:
                raise RuptureGeneratorError(
                    "propagation: several segments need a [propagation] table to say "
                    "which triggers which"
                )
            return realisation
        case Predetermined():
            tree = {name: propagation.parents.get(name) for name in realisation}
            unknown = sorted(set(propagation.parents) - set(realisation))
            if unknown:
                raise RuptureGeneratorError(
                    f"propagation.parents: {unknown} are not segments"
                )
            return realisation.propagated(tree, hypocentre)
        case Sampled():
            rng = np.random.default_rng(seed)
            return sample_path(
                realisation, hypocentre, rng=rng, model=propagation.model
            )
        case Likeliest():
            return likeliest_path(realisation, hypocentre, model=propagation.model)


def _models(
    config: RuptureConfig, realisation: Realisation[Geometry]
) -> RuptureModel | dict[str, RuptureModel]:
    unknown = sorted(set(config.segments) - set(realisation))
    if unknown:
        raise RuptureGeneratorError(f"segments: {unknown} are not segments")
    shared = RuptureModel(
        **{key: getattr(config, key) for key in MODELS},
        pulse=PulseSettings(dt_s=config.pulse.dt_s, beta=config.pulse.beta),
    )
    if not config.segments:
        return shared
    models = {}
    for name in realisation:
        segment = config.segments.get(name, SegmentConfig())
        given = {
            key: value for key in MODELS if (value := getattr(segment, key)) is not None
        }
        models[name] = dataclasses.replace(shared, **given)
    return models


def build(config: RuptureConfig) -> Scenario:
    """Read the geometry a config names, and build every input a rupture needs.

    Parameters
    ----------
    config : RuptureConfig
        A loaded rupture file.

    Returns
    -------
    Scenario
        The fault system, its rock and sources, and every setting the draw takes.

    Raises
    ------
    RuptureGeneratorError
        If the parts of the config are inconsistent, or the geometry describes no
        fault system.
    OSError
        If reading the geometry file fails.
    """
    try:
        crs = projected_crs(config.geometry.crs)
    except RuptureGeneratorError as error:
        raise RuptureGeneratorError(f"geometry.crs: {error}") from error
    aliases = NSHM_ALIASES if config.geometry.aliases == "nshm" else None
    with Path(config.geometry.path).open(encoding="utf-8") as handle:
        coarse = geometry_from_geojson(handle, crs, aliases=aliases)
    charts = {
        name: chart.subdivide(config.geometry.spacing_km)
        for name, chart in coarse.items()
    }

    spec = config.hypocentre
    if spec.segment not in charts:
        raise RuptureGeneratorError(
            f"hypocentre.segment: {spec.segment!r} is not one of {sorted(charts)}"
        )
    # `HypocentreConfig` refuses anything but exactly one complete pair.
    if spec.strike_km is not None and spec.dip_km is not None:
        hypocentre = Hypocentre(spec.segment, spec.strike_km, spec.dip_km)
    else:
        assert spec.strike_fraction is not None and spec.dip_fraction is not None
        hypocentre = Hypocentre.from_fractions(
            spec.segment, charts[spec.segment], spec.strike_fraction, spec.dip_fraction
        )
    realisation = Realisation(charts, crs, hypocentre=hypocentre)

    medium = config.medium.medium()

    return Scenario(
        realisation=_propagated(
            config.propagation, realisation, hypocentre, config.seed
        ),
        medium=medium,
        sources=_sources(config.source, realisation, medium, config.seed),
        model=_models(config, realisation),
        jump_model=(Jump() if config.propagation is None else config.propagation.model),
        seed=config.seed,
    )


__all__ = ["RuptureConfig", "Scenario", "build", "dump", "load"]
