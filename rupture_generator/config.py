"""A rupture described in a TOML file, and the inputs :func:`generate` takes built from it.

This module is an adapter over the library and nothing in the library imports it.
Settings that are already plain data, such as the parts of :class:`RuptureSettings`,
the correlation relation and the jump model, are the library's own classes. A subclass
that restates no field makes each one strict about unknown keys, so the library
defines every default once. A small class here builds whatever the file describes
declaratively and the library takes as a function: a depth profile, or a medium.

:func:`load` reads and checks a file, and :func:`build` turns it into a
:class:`Scenario`. :func:`dump` writes a loaded config back out with every default
filled in, which records what a run used.
"""

import dataclasses
import tomllib
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pyproj
from mashumaro.codecs.toml import TOMLDecoder, TOMLEncoder
from mashumaro.config import BaseConfig
from mashumaro.exceptions import (
    ExtraKeysError,
    InvalidFieldValue,
    MissingField,
    SuitableVariantNotFoundError,
)
from mashumaro.types import Discriminator

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.geometry import NSHM_ALIASES, CellArray, geometry_from_geojson
from rupture_generator.rupture.generator import (
    FaultProfiles,
    RakeSettings,
    RiseSettings,
    RuptureSettings,
    SlipSettings,
    TimingSettings,
)
from rupture_generator.rupture.medium import (
    Layers,
    Medium,
    SpatialField,
    constant_field,
    interpolated_field,
    layered_medium,
    ramp_field,
)
from rupture_generator.rupture.propagator import JumpModel, likeliest_path, sample_path
from rupture_generator.rupture.realisation import Hypocentre, Realisation
from rupture_generator.rupture.source import (
    CorrelationRelation,
    SegmentSource,
    magnitude_from_moment,
    moment_from_magnitude,
    segment_source,
    split_moment,
)


class _Strict(BaseConfig):
    forbid_extra_keys = True


# ------------------------------------------- the library's own settings, made strict


@dataclasses.dataclass(frozen=True)
class Slip(SlipSettings):
    """``[slip]``: :class:`~rupture_generator.rupture.generator.SlipSettings`."""

    Config = _Strict


@dataclasses.dataclass(frozen=True)
class Rise(RiseSettings):
    """``[rise]``: :class:`~rupture_generator.rupture.generator.RiseSettings`."""

    Config = _Strict


@dataclasses.dataclass(frozen=True)
class Rake(RakeSettings):
    """``[rake]``: :class:`~rupture_generator.rupture.generator.RakeSettings`."""

    Config = _Strict


@dataclasses.dataclass(frozen=True)
class Timing(TimingSettings):
    """``[timing]``: :class:`~rupture_generator.rupture.generator.TimingSettings`."""

    Config = _Strict


@dataclasses.dataclass(frozen=True)
class Correlation(CorrelationRelation):
    """A source's ``correlation`` table, Mai & Beroza (2002) when absent."""

    Config = _Strict


@dataclasses.dataclass(frozen=True)
class Jump(JumpModel):
    """A propagation's ``model`` table, Shaw & Dieterich (2007) when absent.

    When the file doesn't give a tree, the model picks one. It also sets the distance each
    jump can cover.
    """

    Config = _Strict


# ---------------------------------------------------------------- depth profiles


@dataclasses.dataclass(frozen=True, kw_only=True)
class Constant:
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
class Profile:
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
    depth_km: list[float]
    values: list[float]
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
class Ramp:
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
        The projected frame to build the charts in, such as ``"EPSG:2193"``.
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
class ProfilesConfig:
    """``[profiles]``: the fault profiles, each a depth profile.

    Each one maps to a field of
    :class:`~rupture_generator.rupture.generator.FaultProfiles`, and leaving it out
    leaves that setting unmodified.

    Parameters
    ----------
    rise_time_factor : DepthProfile, optional
        The relative rise time.
    rise_time_slip_weight : DepthProfile, optional
        How much of rise time's correlation with slip is the configured value.
    rupture_speed_factor : DepthProfile, optional
        The factor on the front's speed.
    """

    rise_time_factor: DepthProfile | None = None
    rise_time_slip_weight: DepthProfile | None = None
    rupture_speed_factor: DepthProfile | None = None
    Config = _Strict

    def profiles(self) -> FaultProfiles:
        """Build the fault profiles the table gives.

        Returns
        -------
        FaultProfiles
            The profiles given, as functions of position, and the rest unmodified.
        """
        return FaultProfiles(
            **{
                field.name: profile.field()
                for field in dataclasses.fields(self)
                if (profile := getattr(self, field.name)) is not None
            }
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
    correlation : Correlation
        How the slip correlation lengths follow each segment's magnitude.
    """

    type: Literal["per_fault"] = "per_fault"
    magnitudes: dict[str, float]
    rakes: dict[str, float]
    correlation: Correlation = dataclasses.field(default_factory=Correlation)
    Config = _Strict


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
    correlation : Correlation
        How the slip correlation lengths follow each segment's magnitude.
    """

    type: Literal["finite"] = "finite"
    magnitude: float
    rake_deg: float
    correlation: Correlation = dataclasses.field(default_factory=Correlation)
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
    profiles : ProfilesConfig
        ``[profiles]``.
    slip : Slip
        ``[slip]``.
    rise : Rise
        ``[rise]``.
    rake : Rake
        ``[rake]``.
    timing : Timing
        ``[timing]``.
    pulse : PulseConfig
        ``[pulse]``.
    """

    seed: int
    geometry: GeometryConfig
    hypocentre: HypocentreConfig
    medium: MediumConfig
    source: SourceConfig
    propagation: PropagationConfig | None = None
    profiles: ProfilesConfig = dataclasses.field(default_factory=ProfilesConfig)
    slip: Slip = dataclasses.field(default_factory=Slip)
    rise: Rise = dataclasses.field(default_factory=Rise)
    rake: Rake = dataclasses.field(default_factory=Rake)
    timing: Timing = dataclasses.field(default_factory=Timing)
    pulse: PulseConfig = dataclasses.field(default_factory=PulseConfig)
    Config = _Strict


_DECODER = TOMLDecoder(RuptureConfig)
_ENCODER = TOMLEncoder(RuptureConfig)


# ------------------------------------------------------------------ the boundary


def _describe(error: Exception) -> RuptureGeneratorError:
    """Build one error that states the key at fault, from mashumaro's chain of them.

    Mashumaro wraps each level of nesting in its own exception, so the key path is the
    chain's field names, and the reason is at its far end.
    """
    path: list[str] = []
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
        config = _DECODER.decode(path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise RuptureGeneratorError(f"{path} is not TOML: {error}") from None
    except (InvalidFieldValue, MissingField, ExtraKeysError) as error:
        raise _describe(error) from None
    geometry = dataclasses.replace(
        config.geometry, path=(path.parent / config.geometry.path).resolve()
    )
    return dataclasses.replace(config, geometry=geometry)


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

    The SRF writer's inputs too, all built from a config.

    Parameters
    ----------
    realisation : Realisation
        The fault system, its hypocentre and its triggering tree.
    medium : Medium
        The rock.
    sources : dict of str to SegmentSource
        Each segment's source, by name.
    settings : RuptureSettings
        What every segment shares.
    jump_model : JumpModel
        How far each jump can cover.
    seed : int
        The seed every random draw derives from.
    dt_s : float
        The slip-rate sample interval, in seconds.
    beta : dict of str to CellArray or None
        The pulse shape's rising fraction per cell by segment. ``None`` gives
        single-sample impulses.
    """

    realisation: Realisation
    medium: Medium
    sources: dict[str, SegmentSource]
    settings: RuptureSettings
    jump_model: JumpModel
    seed: int
    dt_s: float
    beta: dict[str, CellArray] | None


def _sources(
    source: PerFault | Finite,
    realisation: Realisation,
    medium: Medium,
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
                name: segment_source(
                    source.magnitudes[name], source.rakes[name], source.correlation
                )
                for name in realisation
            }
        case Finite():
            shares = split_moment(
                moment_from_magnitude(source.magnitude), realisation, medium
            )
            return {
                name: segment_source(
                    magnitude_from_moment(moment_nm),
                    source.rake_deg,
                    source.correlation,
                )
                for name, moment_nm in shares.items()
            }


def _propagated(
    propagation: Predetermined | Sampled | Likeliest | None,
    realisation: Realisation,
    hypocentre: Hypocentre,
    seed: int,
) -> Realisation:
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
    crs = pyproj.CRS(config.geometry.crs)
    aliases = NSHM_ALIASES if config.geometry.aliases == "nshm" else None
    with open(config.geometry.path, encoding="utf-8") as handle:
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
    beta = (
        None
        if config.pulse.beta is None
        else {
            name: config.pulse.beta.field()(chart.centres)
            for name, chart in charts.items()
        }
    )

    return Scenario(
        realisation=_propagated(
            config.propagation, realisation, hypocentre, config.seed
        ),
        medium=medium,
        sources=_sources(config.source, realisation, medium),
        settings=RuptureSettings(
            slip=config.slip,
            rise=config.rise,
            rake=config.rake,
            timing=config.timing,
            profiles=config.profiles.profiles(),
        ),
        jump_model=(Jump() if config.propagation is None else config.propagation.model),
        seed=config.seed,
        dt_s=config.pulse.dt_s,
        beta=beta,
    )


__all__ = ["RuptureConfig", "Scenario", "build", "dump", "load"]
