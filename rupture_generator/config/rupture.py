"""What the earthquake is: the input to ``rupture-generator generate``.

# One copy

The port kept three descriptions of a rupture model -- these dataclasses, a Rust spec
mirror, and a stub -- policed by a completeness test whose only job was to keep them
agreeing. Every one of the four wrong numbers found in the reduction sweep was a
disagreement between copies. This file is now the **only** copy: stages take frozen
parameter objects built from these classes, kernels take scalars and arrays, and there
is nothing left to mirror.

The one place the "configuration is the pipeline's vocabulary" rule bends is the
**hypocentre**, which is in-fault arc lengths here and a cell index in the pipeline.
That conversion is the mesh's, it is `DEFECTS.md` 17's exact subject, and it happens at
one seam -- ``mesh.RuptureMesh.cell_index`` -- with the convention written above it.

# The physics vocabulary shrank

Production selects one corner relation (``mai``), one spectral shape (``von_karman``),
and one slip-rate family (``OliuP2``). The others were documented knobs, so they are
**refused by name with a message saying they were removed** -- a reader who wrote
``model = "somerville"`` deserves to know it was a decision, not a typo. The evidence
that the config, not the output, is what adjudicates this selection is `DEFECTS.md`
11: the Mai/Somerville crossover at M7.37 makes output comparison unable to tell them
apart below that magnitude.
"""

from __future__ import annotations

import dataclasses
import math
import tomllib
from pathlib import Path
from typing import Literal

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

REMOVED_CORNER_MODELS = ("somerville", "suzuki", "given")
"""Corner relations the rewrite removed. Production's `defaults.yaml` sets
``srf.kmodel: 2`` -- Mai -- with no override on any path, so the others go until
someone asks for one with a reason."""

REMOVED_SPECTRUM_SHAPES = ("somerville", "frankel")
"""Spectral falloffs the rewrite removed. Von Karman (Hurst 0.75) is Mai's own
falloff, the shape ``kmodel: 2`` takes; the hybrid weights that fed the others are
inert in production."""


@dataclasses.dataclass
class RampConfig(ConfigObject):
    """A linear ramp between two depths, in kilometres."""

    centre_km: DepthKm
    half_width_km: PositiveFloat


@dataclasses.dataclass
class HypocentreConfig(ConfigObject):
    """Where the rupture starts, in the fault's own coordinates.

    Arc lengths rather than indices, and rather than the SRF's ``shyp``:
    ``strike_km`` from the ``j = 0`` end of the fault and ``dip_km`` from its top edge.
    Both are in-fault distances, so they mean the same thing whatever the fault is cut
    into -- which an index does not.
    """

    strike_km: DepthKm
    dip_km: DepthKm


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
    """What the earthquake is. Tagged: a finite fault and a point source enter the
    pipeline differently -- one draws fields, the other is the constant case."""

    class Config(ConfigObject.Config):
        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclasses.dataclass
class FiniteSourceConfig(SourceConfig):
    """A finite fault.

    ``model`` is the corner relation only, not the spectral shape a `[slip]` section
    chooses independently as ``shape`` -- the two used to be one vocabulary, and
    nothing checked that a `[source]` and a `[slip]` section naming different
    relations agreed (`DEFECTS.md` 11).
    """

    magnitude: Magnitude
    average_dip_deg: DipDeg
    average_rake_deg: RakeDeg
    model: str = "mai"
    strike_offset: float = 2.50
    dip_offset: float = 1.50
    rise_time_coefficient: PositiveFloat = 1.6
    type: Literal["finite"] = "finite"

    def __post_init__(self) -> None:
        """Validate the fields, then the vocabulary."""
        super().__post_init__()
        if self.model in REMOVED_CORNER_MODELS:
            self.refuse(
                "model",
                f"the corner relation {self.model!r} was removed in the pipeline "
                "rewrite: production selects 'mai', and the others go until someone "
                "asks for one with a reason",
            )
        if self.model != "mai":
            self.refuse("model", f"no corner relation is spelled {self.model!r}")


@dataclasses.dataclass
class PointSourceConfig(SourceConfig):
    """A point source.

    Not a finite source with fields left blank. There is no spectrum, so no corner
    relation, and ``rise_time_s`` is given rather than derived from the moment -- as
    the **fault-wide average**, which the depth ramp redistributes around.
    """

    magnitude: Magnitude
    rise_time_s: PositiveFloat
    average_dip_deg: DipDeg
    average_rake_deg: RakeDeg
    type: Literal["point"] = "point"


def default_wavelength_band(strike_km: float, dip_km: float) -> tuple[float, float]:
    """The wavelength limits genslip picks when none is given, for this grid.

    No constant is right on two grids, which is why `SlipConfig` leaves both `None`
    rather than carrying a literal: genslip derives the low end from the grid itself,
    ``2*sqrt(dstk*ddip)/0.8`` -- 80% of the Nyquist wavelength of a grid whose spacing
    is the geometric mean of the strike and dip cell sizes, so the band-pass rolls off
    at 80% of the Nyquist *wavenumber* rather than at it exactly. The high end has no
    real bound: genslip's own is ``1.0e15``, assigned after the ``getpar`` that reads
    it, so nothing a config file says about it is ever seen.
    """
    return 2.0 * math.sqrt(strike_km * dip_km) / 0.8, 1.0e15


@dataclasses.dataclass
class SlipConfig(ConfigObject):
    """How the slip and rake fields are shaped and trimmed.

    ``coefficient_of_variation`` is the slip field's spread and is dimensionless;
    ``rake_sigma_deg`` is the rake field's and is in **degrees**. Handing one to the
    other is `DEFECTS.md` 14, which gave every rake a spread of 0.75 degrees where the
    original gives 15 -- a factor of twenty, on every fault. They are never both bare
    numbers in the same expression here, and their names carry the difference.

    ``min_wavelength_km`` and ``max_wavelength_km`` default to `None` rather than to a
    literal, because the right value depends on the grid the field is sampled on --
    see `default_wavelength_band`. Either can still be set explicitly, which is
    honoured over the derived value.
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
        # Only checkable when both are given -- a `None` is filled in from the grid
        # when the field is sampled, too late for this constructor to see, and always
        # self-consistent by construction (`default_wavelength_band`'s low end is
        # kilometres, its high end is 1e15).
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

    ``shallow_ramp`` and ``deep_ramp`` stretch **rise time**. Rupture speed has ramps
    of its own, which default to the rise-time ones because that is the case the
    original's four independent parameters share; ``shallow_speed_ramp`` and
    ``deep_speed_ramp`` override them when they do not. `DEFECTS.md` 13 was one pair
    reaching both.
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
    # genslip's own `stype` spelling, parsed by `pulses.from_stype` -- including
    # `ucsb-T`'s numeric suffix, which is why this is a string and not a `Literal`.
    # The production workflow's `defaults.yaml` advertises removed shapes as valid
    # `stype` values, so the parse distinguishes "removed" from "unknown".
    slip_rate_shape: str | None = None
    beta_shallow: PositiveFloat = 0.5
    beta_mid: PositiveFloat = 0.13
    beta_deep: PositiveFloat = 0.13
    sample_interval_s: PositiveFloat = 0.005

    def __post_init__(self) -> None:
        """Validate the fields, then the one with its own vocabulary.

        Parsed here so an unrecognised or removed ``stype`` is refused when the file
        is read, naming the field, rather than partway through a generation run. The
        C falls through to ``brune`` on a name it does not know and silently produces
        a different rupture.
        """
        super().__post_init__()
        if self.slip_rate_shape is not None:
            try:
                from_stype(self.slip_rate_shape)
            except ValueError as error:
                self.refuse("slip_rate_shape", str(error))


@dataclasses.dataclass
class FieldConfig(ConfigObject):
    """The two per-subfault fields the geometry does not supply.

    Both are constants here. The stages take them per subfault, because a mesh may
    vary them; a config that could say so per subfault would need a way to address
    subfaults, which is a bigger thing than this needs to be yet.
    """

    base_rake_deg: RakeDeg = 175.0
    velocity_fraction: VelocityFraction = 0.8


@dataclasses.dataclass
class RandomConfig(ConfigObject):
    """Which stream of numbers, and where in it.

    One event seed. ``numpy.random.SeedSequence(seed)`` spawns every (stage, segment)
    pair its own named substream, so draw order inside the pipeline does not matter
    and changing one stage's parameters cannot change another's noise.
    ``realisation`` selects an independent stream from the same seed, which is what
    makes a campaign restartable.
    """

    seed: int = 1234
    realisation: int = 0

    def __post_init__(self) -> None:
        """Validate the fields, then the invariants between them."""
        super().__post_init__()
        if self.realisation < 0:
            self.refuse("realisation", f"must be 0 or more, got {self.realisation}")


DECODERS = {
    "toml": TOMLDecoder,
    "yaml": YAMLDecoder,
    "yml": YAMLDecoder,
    "json": JSONDecoder,
}
"""Which decoder an extension means. TOML is the default for anything unrecognised."""


@dataclasses.dataclass
class RuptureConfig(ConfigObject):
    """A whole generate config.

    Examples
    --------
    TOML::

        schema_version = 1
        title = "Crustal M6.2"

        [hypocentre]
        strike_km = 5.0
        dip_km = 4.0

        [velocity_model]
        bottom_depth_km  = [1.0, 5.0, 12.0, 1000.0]
        shear_speed_km_s = [1.8, 3.2, 3.5, 4.6]
        density_g_cm3    = [2.1, 2.5, 2.7, 3.2]

        [source]
        type = "finite"
        magnitude = 6.2
        average_dip_deg = 60.0
        average_rake_deg = 175.0

        [timing]
        rupture_time_scale = -0.35
        rise_time_blend   = { centre_km = 2.0,  half_width_km = 1.0 }
        shallow_ramp      = { centre_km = 6.5,  half_width_km = 1.5 }
        deep_ramp         = { centre_km = 17.5, half_width_km = 2.5 }
        beta_shallow_ramp = { centre_km = 2.0,  half_width_km = 1.0 }
        beta_mid_ramp     = { centre_km = 6.5,  half_width_km = 1.5 }

        [random]
        seed = 1234
    """

    hypocentre: HypocentreConfig
    velocity_model: VelocityModelConfig
    source: SourceConfig
    timing: TimingConfig
    slip: SlipConfig = dataclasses.field(default_factory=SlipConfig)
    field: FieldConfig = dataclasses.field(default_factory=FieldConfig)
    random: RandomConfig = dataclasses.field(default_factory=RandomConfig)
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
        If it does not parse. The CLI renders these differently -- a syntax error
        wants a line number and the line, and a validation error wants a key.
    """
    path = Path(path)
    chosen = format or path.suffix.lstrip(".").lower()
    return DECODERS.get(chosen, TOMLDecoder)(RuptureConfig).decode(path.read_text())


def read_geometry(path: Path | str, format: str | None = None):
    """Read a geometry config. The counterpart of :func:`read_config`.

    Returns
    -------
    GeometryConfig
    """
    from rupture_generator.config.geometry import GeometryConfig

    path = Path(path)
    chosen = format or path.suffix.lstrip(".").lower()
    return DECODERS.get(chosen, TOMLDecoder)(GeometryConfig).decode(path.read_text())


__all__ = [
    "DECODERS",
    "REMOVED_CORNER_MODELS",
    "REMOVED_SPECTRUM_SHAPES",
    "FieldConfig",
    "FiniteSourceConfig",
    "HypocentreConfig",
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
