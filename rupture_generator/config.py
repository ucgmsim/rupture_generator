"""A rupture described in a TOML file, and the inputs :func:`generate` takes built from it.

This module is an adapter over the library and nothing in the library imports it.
Settings that are already plain data -- :class:`RuptureSettings`' four parts, the
correlation relation, the jump model -- are the library's own classes, made strict
about unknown keys by a subclass that restates no field, so every default lives in one
place. What the file describes declaratively and the library takes as a function --
a depth profile, a medium -- has a small class here that builds it.

:func:`load` reads and checks a file, :func:`build` turns it into a :class:`Scenario`,
and :func:`dump` writes a loaded config back out with every default filled in, which
is the record of what was run.
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
    """A source's ``correlation`` table; Mai & Beroza (2002) when absent."""

    Config = _Strict


@dataclasses.dataclass(frozen=True)
class Jump(JumpModel):
    """A propagation's ``model`` table: Shaw & Dieterich (2007) when absent. It picks
    the tree, when the tree is not given, and how far each jump reaches."""

    Config = _Strict


# ---------------------------------------------------------------- depth profiles


@dataclasses.dataclass(frozen=True, kw_only=True)
class Constant:
    """The same value at every depth."""

    type: Literal["constant"] = "constant"
    value: float
    Config = _Strict

    def field(self) -> SpatialField:
        """The profile as a function of position."""
        return constant_field(self.value)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Profile:
    """Values at depths, linear between them and flat beyond."""

    type: Literal["profile"] = "profile"
    depth_km: list[float]
    values: list[float]
    Config = _Strict

    def field(self) -> SpatialField:
        """The profile as a function of position."""
        return interpolated_field(np.array(self.depth_km), np.array(self.values))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Ramp:
    """``shallow`` above ``centre_km - half_width_km``, ``deep`` below
    ``centre_km + half_width_km``, linear between."""

    type: Literal["ramp"] = "ramp"
    centre_km: float
    half_width_km: float
    shallow: float
    deep: float
    Config = _Strict

    def field(self) -> SpatialField:
        """The profile as a function of position."""
        return ramp_field(self.centre_km, self.half_width_km, self.shallow, self.deep)


type DepthProfile = Annotated[
    Constant | Profile | Ramp, Discriminator(field="type", include_supertypes=True)
]


# ----------------------------------------------------------------------- tables


@dataclasses.dataclass(frozen=True)
class GeometryConfig:
    """``[geometry]``: the GeoJSON sections, the frame, and the subfault size.

    A relative ``path`` is relative to the config file.
    """

    path: Path
    crs: str
    spacing_km: float
    aliases: Literal["nshm"] | None = None
    Config = _Strict


@dataclasses.dataclass(frozen=True, kw_only=True)
class HypocentreConfig:
    """``[hypocentre]``: a segment, and either arc lengths or fractions of its extent."""

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
    """``[medium]``: the rock, as a 1-D model with one value per layer."""

    bottom_depth_km: list[float]
    shear_speed_km_s: list[float]
    density_g_cm3: list[float]
    Config = _Strict

    def medium(self) -> Medium:
        """Shear speed and density as functions of position."""
        return layered_medium(
            Layers(np.array(self.bottom_depth_km)),
            np.array(self.shear_speed_km_s),
            np.array(self.density_g_cm3),
        )


@dataclasses.dataclass(frozen=True)
class ProfilesConfig:
    """``[profiles]``: :class:`~rupture_generator.rupture.generator.FaultProfiles`,
    each a depth profile and unmodified when absent."""

    rise_time_factor: DepthProfile | None = None
    rise_time_slip_weight: DepthProfile | None = None
    rupture_speed_factor: DepthProfile | None = None
    Config = _Strict

    def profiles(self) -> FaultProfiles:
        """The profiles given, as functions of position."""
        return FaultProfiles(
            **{
                field.name: profile.field()
                for field in dataclasses.fields(self)
                if (profile := getattr(self, field.name)) is not None
            }
        )


@dataclasses.dataclass(frozen=True)
class PulseConfig:
    """``[pulse]``: the slip-rate sample interval, and the Liu-Archuleta-Hartzell
    rising fraction; without ``beta`` every pulse is a single-sample impulse."""

    dt_s: float = 0.005
    beta: DepthProfile | None = None
    Config = _Strict

    def __post_init__(self) -> None:
        """Refuse a sample interval that samples nothing."""
        if not self.dt_s > 0.0:
            raise RuptureGeneratorError(f"a sample interval of {self.dt_s} s is none")


@dataclasses.dataclass(frozen=True, kw_only=True)
class PerFault:
    """A magnitude and a rake for every segment, as a hazard model states them."""

    type: Literal["per_fault"] = "per_fault"
    magnitudes: dict[str, float]
    rakes: dict[str, float]
    correlation: Correlation = dataclasses.field(default_factory=Correlation)
    Config = _Strict


@dataclasses.dataclass(frozen=True, kw_only=True)
class Finite:
    """One magnitude and rake for the event, its moment shared between segments by
    :func:`~rupture_generator.rupture.source.split_moment`."""

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
    """Which segment triggers which, as a scenario states it."""

    type: Literal["predetermined"] = "predetermined"
    parents: dict[str, str]
    model: Jump = dataclasses.field(default_factory=Jump)
    Config = _Strict


@dataclasses.dataclass(frozen=True, kw_only=True)
class Sampled:
    """A tree drawn from the jump model."""

    type: Literal["sampled"] = "sampled"
    model: Jump = dataclasses.field(default_factory=Jump)
    Config = _Strict


@dataclasses.dataclass(frozen=True, kw_only=True)
class Likeliest:
    """The jump model's likeliest tree."""

    type: Literal["likeliest"] = "likeliest"
    model: Jump = dataclasses.field(default_factory=Jump)
    Config = _Strict


type PropagationConfig = Annotated[
    Predetermined | Sampled | Likeliest,
    Discriminator(field="type", include_supertypes=True),
]


@dataclasses.dataclass(frozen=True, kw_only=True)
class RuptureConfig:
    """A whole rupture file. ``[propagation]`` may be left out for one segment."""

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
    """One error naming the key at fault, from mashumaro's chain of them.

    Mashumaro wraps each level of nesting in its own exception, so the key path is the
    chain's field names and the reason is at its far end.
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
    """Read and check a rupture file. The geometry's path is resolved against it.

    Raises
    ------
    RuptureGeneratorError
        If the file is not TOML, or does not describe a rupture.
    OSError
        If the file cannot be read.
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
    """A config as TOML, every default written out."""
    return _ENCODER.encode(config)


# --------------------------------------------------------------------- building


@dataclasses.dataclass(frozen=True, eq=False)
class Scenario:
    """Everything :func:`~rupture_generator.rupture.generator.generate` and the SRF
    writer take, built from a config.

    ``beta`` is the pulse shape's rising fraction per segment, or ``None`` for
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
    """Read the geometry a config names, and build everything a rupture is drawn from.

    Raises
    ------
    RuptureGeneratorError
        If the parts of the config do not fit together, or the geometry does not
        describe a fault system.
    OSError
        If the geometry file cannot be read.
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
    hypocentre = (
        Hypocentre(spec.segment, spec.strike_km, spec.dip_km)
        if spec.strike_km is not None
        else Hypocentre.from_fractions(
            spec.segment, charts[spec.segment], spec.strike_fraction, spec.dip_fraction
        )
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
