"""What the earthquake is: the input to ``rupture-generator generate``."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from mashumaro.codecs.json import JSONDecoder
from mashumaro.codecs.toml import TOMLDecoder
from mashumaro.codecs.yaml import YAMLDecoder
from mashumaro.types import Discriminator

from rupture_generator.config.core import ConfigObject
from rupture_generator.config.validation import (
    DepthKm,
    DipDeg,
    Magnitude,
    NonEmptyStr,
    PositiveFloat,
    RakeDeg,
    Seconds,
    UnitInterval,
    VelocityFraction,
    non_empty,
)
from rupture_generator.pulses import from_stype
from rupture_generator.sampling import VonKarmanFilterParameters, correlation_lengths

if TYPE_CHECKING:
    from rupture_generator.mesh import RuptureMesh

REMOVED_CORNER_MODELS = ("somerville", "suzuki", "given")
"""Corner relations refused by name; ``custom`` states coefficients instead."""

CORNER_MODELS = ("mai", "custom")
"""The corner relations a source may name. ``mai`` is the published relation and takes
no coefficients; ``custom`` takes all four and no name."""

CORNER_COEFFICIENTS = ("strike_offset", "dip_offset", "strike_exponent", "dip_exponent")
"""An exponent and an offset per axis, in kilometres through
``lambda = 10 ** (exponent * Mw - offset)`` -- see `sampling.correlation_lengths`."""

REMOVED_SPECTRUM_SHAPES = ("somerville", "frankel")
"""Spectral falloffs refused by name. Von Karman (Hurst 0.75) is the one kept."""


@dataclasses.dataclass
class RampConfig(ConfigObject):
    """A linear ramp between two depths, in kilometres."""

    centre_km: DepthKm
    half_width_km: PositiveFloat


@dataclasses.dataclass
class HypocentreConfig(ConfigObject):
    """Where the rupture starts, in the fault's own coordinates."""

    strike_km: DepthKm
    dip_km: DepthKm
    fault: str | None = None


@dataclasses.dataclass
class VelocityModelConfig(ConfigObject):
    """A layered one-dimensional velocity model, ordered shallow to deep."""

    bottom_depth_km: list[float]
    shear_speed_km_s: list[float]
    density_g_cm3: list[float]

    def __post_init__(self) -> None:
        """Validate the fields, then the invariants between them."""
        super().__post_init__()
        lengths = {
            "bottom_depth_km": len(self.bottom_depth_km),
            "shear_speed_km_s": len(self.shear_speed_km_s),
            "density_g_cm3": len(self.density_g_cm3),
        }
        if len(set(lengths.values())) != 1:
            self.refuse(
                "bottom_depth_km",
                f"the three columns describe different numbers of layers: {lengths}",
            )
        non_empty(self.bottom_depth_km)
        for name in lengths:
            for value in getattr(self, name):
                if value <= 0.0:
                    self.refuse(
                        name, f"every layer's {name} must be positive, got {value}"
                    )


@dataclasses.dataclass
class SourceConfig(ConfigObject):
    """What the earthquake is: a finite fault, a point source, or per-fault values."""

    class Config(ConfigObject.Config):
        discriminator = Discriminator(field="type", include_subtypes=True)

    def check_segments(self, segments: list[str]) -> None:
        """Refuse a source that does not describe this rupture's faults."""

    def magnitude_of(self, segment: str) -> float:
        """The magnitude this segment carries."""
        return float(self.magnitude)  # ty: ignore[unresolved-attribute]

    def rake_of(self, segment: str) -> float:
        """This segment's mean rake, in degrees -- the mechanism, not the field."""
        return float(self.average_rake_deg)  # ty: ignore[unresolved-attribute]

    def dip_of(self, segment: str, mesh: RuptureMesh) -> float:
        """This segment's mean dip, in degrees, read off the chart if not stated."""
        return float(self.average_dip_deg)  # ty: ignore[unresolved-attribute]

    def base_rake_deg_of(self, segment: str, default_deg: float) -> float:
        """What this segment's rake *field* is centred on, in degrees.

        Not :meth:`rake_of`: here it is the ``[field]`` section's ``base_rake_deg``.
        """
        return default_deg

    def covariance_of(self, segment: str) -> VonKarmanFilterParameters:
        """The patch structure this segment's magnitude implies."""
        return correlation_lengths(self.magnitude_of(segment))


@dataclasses.dataclass
class FiniteSourceConfig(SourceConfig):
    """A finite fault.

    ``model`` is the corner relation only, not the spectral shape a `[slip]` section
    chooses as ``shape``. ``mai`` is Mai & Beroza (2002) and carries no coefficients;
    ``custom`` states all four here.
    """

    magnitude: Magnitude
    average_dip_deg: DipDeg
    average_rake_deg: RakeDeg
    model: str = "mai"
    strike_offset: float | None = None
    dip_offset: float | None = None
    strike_exponent: float | None = None
    dip_exponent: float | None = None
    rise_time_coefficient: PositiveFloat = 1.6
    type: Literal["finite"] = "finite"

    def __post_init__(self) -> None:
        """Validate the fields, then the vocabulary."""
        super().__post_init__()
        if self.model in REMOVED_CORNER_MODELS:
            self.refuse(
                "model",
                f"the corner relation {self.model!r} was removed in the pipeline "
                "rewrite: production selects 'mai', and a relation of your own is "
                f"model = 'custom' with {', '.join(CORNER_COEFFICIENTS)} stated",
            )
        if self.model not in CORNER_MODELS:
            self.refuse("model", f"no corner relation is spelled {self.model!r}")
        missing = [name for name in CORNER_COEFFICIENTS if getattr(self, name) is None]
        if self.model == "mai":
            for name in CORNER_COEFFICIENTS:
                if getattr(self, name) is not None:
                    self.refuse(
                        name,
                        f"{name} is part of Mai & Beroza's published fit, so a 'mai' "
                        "relation does not take one -- overriding it would leave the "
                        "file naming a relation it is no longer using. Coefficients "
                        "of your own are model = 'custom', which takes all four",
                    )
        elif missing:
            self.refuse(
                missing[0],
                "a custom corner relation states all four of its coefficients, so "
                "that which relation it is does not depend on this package's "
                f"defaults; missing: {', '.join(missing)}",
            )
        else:
            for name in CORNER_COEFFICIENTS:
                value = getattr(self, name)
                if not math.isfinite(value):
                    self.refuse(name, f"must be a finite number, got {value}")
            for name in ("strike_exponent", "dip_exponent"):
                if getattr(self, name) < 0.0:
                    self.refuse(
                        name,
                        f"must be 0 or more, got {getattr(self, name)}: a negative "
                        "exponent gives a larger earthquake smaller asperities. Zero "
                        "is allowed, and is a correlation length of 10 ** -offset km "
                        "at every magnitude",
                    )

    def corner_coefficients(self) -> dict[str, float]:
        """The coefficients `sampling.correlation_lengths` takes; empty for ``mai``."""
        if self.model != "custom":
            return {}
        return {name: getattr(self, name) for name in CORNER_COEFFICIENTS}

    def covariance_of(self, segment: str) -> VonKarmanFilterParameters:
        """One structure for the whole event: one magnitude, one corner relation."""
        return correlation_lengths(self.magnitude, **self.corner_coefficients())


@dataclasses.dataclass
class PointSourceConfig(SourceConfig):
    """A point source: no spectrum, so no corner relation.

    ``rise_time_s`` is given rather than derived, as the fault-wide average in seconds.
    """

    magnitude: Magnitude
    rise_time_s: PositiveFloat
    average_dip_deg: DipDeg
    average_rake_deg: RakeDeg
    type: Literal["point"] = "point"

    def covariance_of(self, segment: str) -> VonKarmanFilterParameters:
        """Any positive lengths will do: a point source draws no fields."""
        return VonKarmanFilterParameters(1.0, 1.0)


@dataclasses.dataclass
class PerFaultSourceConfig(SourceConfig):
    """A rupture whose faults each carry a magnitude and a rake of their own.

    The event's magnitude is whatever they sum to; dip is read from each segment's
    chart rather than stated.
    """

    magnitudes: dict[str, float] = dataclasses.field(default_factory=dict)
    rakes: dict[str, float] = dataclasses.field(default_factory=dict)
    rise_time_coefficient: PositiveFloat = 1.6
    type: Literal["per_fault"] = "per_fault"

    def __post_init__(self) -> None:
        """Validate the fields, then the invariants between them."""
        super().__post_init__()
        if not self.magnitudes:
            self.refuse(
                "magnitudes",
                "a per-fault source needs a magnitude for each fault; with one fault "
                "and one magnitude, use a finite source",
            )
        missing = set(self.magnitudes) - set(self.rakes)
        if missing:
            self.refuse(
                "rakes",
                f"{', '.join(sorted(missing))} has a magnitude but no rake, and a "
                "fault that slips has a direction",
            )
        for name, magnitude in self.magnitudes.items():
            if not 3.0 <= magnitude <= 10.0:
                self.refuse(
                    "magnitudes",
                    f"{name} has magnitude {magnitude}, outside the [3, 10] this "
                    "generator models",
                )

    @property
    def magnitude(self) -> float:
        """The event's magnitude: the parts summed in moment, not in magnitude."""
        total = sum(
            10.0 ** (1.5 * (value + 6.0333003)) for value in self.magnitudes.values()
        )
        return (math.log10(total) - 9.0499505) / 1.5

    def check_segments(self, segments: list[str]) -> None:
        """Refuse magnitudes that do not name this rupture's faults, both ways round.

        Raises
        ------
        ValueError
        """
        unknown = set(self.magnitudes) - set(segments)
        if unknown:
            self.refuse(
                "magnitudes",
                f"the source gives magnitudes for {', '.join(sorted(unknown))}, which "
                f"are not segments of this rupture ({', '.join(segments)})",
            )
        missing = set(segments) - set(self.magnitudes)
        if missing:
            self.refuse(
                "magnitudes",
                f"{', '.join(sorted(missing))} has no magnitude, and a fault that "
                "ruptures carries moment",
            )

    def magnitude_of(self, segment: str) -> float:
        """This fault's own stated magnitude."""
        return float(self.magnitudes[segment])

    def rake_of(self, segment: str) -> float:
        """This fault's own stated rake."""
        return float(self.rakes[segment])

    def dip_of(self, segment: str, mesh: RuptureMesh) -> float:
        """This fault's mean dip, in degrees, read off its chart."""
        return float(np.mean(mesh.strike_dip_deg()[1]))

    def base_rake_deg_of(self, segment: str, default_deg: float) -> float:
        """The fault's own rake. A system with two mechanisms centres two fields."""
        return self.rakes[segment]


def default_wavelength_band(strike_km: float, dip_km: float) -> tuple[float, float]:
    """The wavelength limits in kilometres to use when none is given, for this grid.

    The low end is ``2*sqrt(dstk*ddip)/0.8``, so the band-pass rolls off at 80% of the
    Nyquist wavenumber of a grid spaced at the geometric mean of the cell sizes. The
    high end has no real bound.
    """
    return 2.0 * math.sqrt(strike_km * dip_km) / 0.8, 1.0e15


@dataclasses.dataclass
class SlipConfig(ConfigObject):
    """How the slip and rake fields are shaped and trimmed.

    ``coefficient_of_variation`` is the slip field's spread and is dimensionless;
    ``rake_sigma_deg`` is the rake field's and is in degrees; the tapers are fractions
    of an edge. The wavelength limits default to `None` because the right value
    depends on the grid -- see `default_wavelength_band`.
    """

    shape: str = "von_karman"
    coefficient_of_variation: PositiveFloat = 0.75
    rake_sigma_deg: PositiveFloat = 15.0
    min_wavelength_km: PositiveFloat | None = None
    max_wavelength_km: PositiveFloat | None = None
    side_taper: UnitInterval = 0.02
    top_taper: UnitInterval = 0.0
    bottom_taper: UnitInterval = 0.0

    def __post_init__(self) -> None:
        """Validate the fields, then the invariants between them."""
        super().__post_init__()
        if self.shape in REMOVED_SPECTRUM_SHAPES:
            self.refuse(
                "shape",
                f"the spectral shape {self.shape!r} was removed in the pipeline "
                "rewrite: production's corner relation takes von Karman, and the "
                "others go until someone asks for one with a reason",
            )
        if self.shape != "von_karman":
            self.refuse("shape", f"no spectral shape is spelled {self.shape!r}")
        # Only checkable when both are given: a `None` is filled in from the grid when
        # the field is sampled, too late for this constructor to see.
        if (
            self.min_wavelength_km is not None
            and self.max_wavelength_km is not None
            and self.max_wavelength_km <= self.min_wavelength_km
        ):
            self.refuse(
                "max_wavelength_km",
                f"must be above min_wavelength_km ({self.min_wavelength_km}), "
                f"got {self.max_wavelength_km}",
            )


@dataclasses.dataclass
class TimingConfig(ConfigObject):
    """How rupture time and rise time relate to slip.

    ``shallow_ramp`` and ``deep_ramp`` stretch rise time; ``shallow_speed_ramp`` and
    ``deep_speed_ramp`` do the same for rupture speed and default to the rise-time
    pair. Ramp depths are in kilometres, times in seconds.
    """

    rupture_time_scale: float
    rise_time_blend: RampConfig
    shallow_ramp: RampConfig
    deep_ramp: RampConfig
    beta_shallow_ramp: RampConfig
    beta_mid_ramp: RampConfig
    rupture_time_correlation: float = 0.8
    rupture_time_sigma: PositiveFloat = 1.0
    rupture_delay_s: Seconds = 0.0
    rise_time_correlation: float = 0.9
    rise_time_sigma: PositiveFloat = 0.75
    slip_exponent: float = 0.5
    shallow_rise_factor: PositiveFloat = 2.0
    deep_rise_factor: PositiveFloat = 2.0
    shallow_speed_ramp: RampConfig | None = None
    deep_speed_ramp: RampConfig | None = None
    shallow_speed_factor: PositiveFloat = 0.6
    deep_speed_factor: PositiveFloat = 0.6
    # An `stype` spelling, parsed by `pulses.from_stype` -- including `ucsb-T`'s
    # numeric suffix, which is why this is a string and not a `Literal`.
    slip_rate_shape: str | None = None
    beta_shallow: PositiveFloat = 0.5
    beta_mid: PositiveFloat = 0.13
    beta_deep: PositiveFloat = 0.13
    sample_interval_s: PositiveFloat = 0.005

    def __post_init__(self) -> None:
        """Validate the fields, then parse ``slip_rate_shape``'s own vocabulary."""
        super().__post_init__()
        if self.slip_rate_shape is not None:
            try:
                from_stype(self.slip_rate_shape)
            except ValueError as error:
                self.refuse("slip_rate_shape", str(error))


@dataclasses.dataclass
class FieldConfig(ConfigObject):
    """The two per-subfault fields the geometry does not supply, as constants."""

    base_rake_deg: RakeDeg = 175.0
    velocity_fraction: VelocityFraction = 0.8


def _key(name: str) -> int:
    """A name as a stable integer, for a spawn key.

    blake2b, not :func:`hash`, which is randomised per process for strings.
    """
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "big")


@dataclasses.dataclass
class RandomConfig(ConfigObject):
    """Which stream of numbers, and where in it.

    ``realisation`` selects an independent stream from the same seed.
    """

    seed: int
    realisation: int

    def __post_init__(self) -> None:
        """Validate the fields, then the invariants between them."""
        super().__post_init__()
        if self.realisation < 0:
            self.refuse("realisation", f"must be 0 or more, got {self.realisation}")

    def stream(self, *args: str) -> np.random.Generator:
        """A generator of this event's own, keyed by the names in `args`.

        By name rather than position, so adding a fault does not redraw the others.
        """
        spawn_key = [self.realisation, *(_key(name) for name in args)]
        return np.random.default_rng(
            np.random.SeedSequence(entropy=self.seed, spawn_key=spawn_key)
        )


DECODERS = {
    "toml": TOMLDecoder,
    "yaml": YAMLDecoder,
    "yml": YAMLDecoder,
    "json": JSONDecoder,
}
"""Which decoder an extension means. TOML is the default for anything unrecognised."""


@dataclasses.dataclass
class PropagationConfig(ConfigObject):
    """How a rupture crosses between segments: a computed or a stated tree."""

    class Config(ConfigObject.Config):
        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclasses.dataclass
class ComputedPropagation(PropagationConfig):
    """Sample which segment triggers which from how far apart they are.

    The probability that a rupture jumps a gap follows Shaw & Dieterich (2007):
    certain within ``delta_km``, decaying with characteristic length ``d0_km``, zero
    beyond ``max_jump_km``. The tree is sampled from those probabilities, or taken as
    its most likely member with ``maximum_likelihood``.
    """

    strategy: Literal["sampled", "maximum_likelihood"] = "sampled"
    d0_km: PositiveFloat = 3.0
    delta_km: PositiveFloat = 1.0
    max_jump_km: PositiveFloat = 15.0
    type: Literal["computed"] = "computed"


@dataclasses.dataclass
class PredeterminedPropagation(PropagationConfig):
    """State which segment triggers which, rather than sampling it.

    ``parents`` maps each triggered segment to the one that triggered it. The segment
    that is nobody's child is the root, and must be the one :class:`HypocentreConfig`
    names.

    Examples
    --------
    TOML::

        [propagation]
        type = "predetermined"
        parents = { kelly = "hope", conway = "kelly" }
    """

    parents: dict[str, str] = dataclasses.field(default_factory=dict)
    type: Literal["predetermined"] = "predetermined"

    def __post_init__(self) -> None:
        """Validate the fields, then the shape of what they describe."""
        super().__post_init__()
        if not self.parents:
            self.refuse(
                "parents",
                "a predetermined propagation needs to say which segment triggers "
                "which; with one segment there is nothing to state, so use the "
                "computed form or omit the section",
            )
        for child, parent in self.parents.items():
            if child == parent:
                self.refuse("parents", f"{child!r} cannot trigger itself")


@dataclasses.dataclass
class RuptureConfig(ConfigObject):
    """A whole generate config.

    See ``examples/crustal.toml`` for a worked file, ``examples/alpine_hope.toml`` for
    a multi-fault one. The sections are :class:`HypocentreConfig`,
    :class:`VelocityModelConfig`, :class:`SourceConfig`, :class:`SlipConfig`,
    :class:`TimingConfig` and :class:`RandomConfig`, each documented on its own class.
    """

    hypocentre: HypocentreConfig
    velocity_model: VelocityModelConfig
    source: SourceConfig
    timing: TimingConfig
    # No default: a rupture that did not state its seed cannot be reproduced.
    random: RandomConfig
    slip: SlipConfig = dataclasses.field(default_factory=SlipConfig)
    field: FieldConfig = dataclasses.field(default_factory=FieldConfig)
    propagation: PropagationConfig = dataclasses.field(
        default_factory=ComputedPropagation
    )
    schema_version: int = 1
    title: NonEmptyStr | None = None

    def __post_init__(self) -> None:
        """Validate the fields, then the invariants between them."""
        super().__post_init__()
        if isinstance(self.source, PointSourceConfig) and self.slip != SlipConfig():
            self.refuse(
                "slip",
                "a point source draws no fields, so a [slip] section would be read "
                "and ignored -- remove it, or use a finite source",
            )


def read_config(path: Path | str, format: str | None = None) -> RuptureConfig:
    """Read a generate config, in whichever of the three spellings it is written.

    Parameters
    ----------
    path : Path or str
        The file.
    format : str, optional
        ``"toml"``, ``"yaml"`` or ``"json"``. Inferred from the extension when
        omitted, defaulting to TOML.

    Returns
    -------
    RuptureConfig

    Raises
    ------
    InvalidFieldValue, MissingField
        If the file parses but does not describe a rupture.
    tomllib.TOMLDecodeError, json.JSONDecodeError, yaml.YAMLError
        If it does not parse.
    """
    path = Path(path)
    chosen = format or path.suffix.lstrip(".").lower()
    return DECODERS.get(chosen, TOMLDecoder)(RuptureConfig).decode(path.read_text())


def read_geometry(path: Path | str, format: str | None = None):
    """Read a geometry config -- the counterpart of :func:`read_config`.

    Returns
    -------
    GeometryConfig
    """
    from rupture_generator.config.geometry import GeometryConfig

    path = Path(path)
    chosen = format or path.suffix.lstrip(".").lower()
    return DECODERS.get(chosen, TOMLDecoder)(GeometryConfig).decode(path.read_text())


__all__ = [
    "CORNER_COEFFICIENTS",
    "CORNER_MODELS",
    "DECODERS",
    "REMOVED_CORNER_MODELS",
    "REMOVED_SPECTRUM_SHAPES",
    "FieldConfig",
    "FiniteSourceConfig",
    "HypocentreConfig",
    "PerFaultSourceConfig",
    "PointSourceConfig",
    "RampConfig",
    "RandomConfig",
    "RuptureConfig",
    "SlipConfig",
    "SourceConfig",
    "TimingConfig",
    "VelocityModelConfig",
    "default_wavelength_band",
    "read_config",
    "read_geometry",
    "tomllib",
]
