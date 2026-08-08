"""What the earthquake is: the input to ``rupture-generator generate``.

# The configuration *is* the compiled core's types

`README.md` and `tests/harness/README.md` both state the rule this file is written
under:

    The configuration **is** the compiled core's types. [...] the moment they appear in
    the library there are two descriptions of a rupture model and they start to drift.

So every field below carries the **same name and the same unit** as the ``_core``
constructor argument it feeds, and every ``to_core`` is a constructor call with no
arithmetic in it -- a reader can check it is a copy by looking. No aliases, no renaming
to something friendlier, no unit written differently because it reads better.

``tests/test_config_completeness.py`` enforces it mechanically, by comparing these
classes against the stub's signatures. A field added to the core and forgotten here goes
red; so does a field here that the core does not have.

The one place that rule does not reach is the **hypocentre**, which is in-fault arc
lengths here and cell indices in the core. That conversion is the mesh's, it is
`DEFECTS.md` 17's exact subject, and it happens at one seam with the convention written
above it -- see ``_core.RefinedMesh.cell_index``.
"""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path
from typing import Literal

from mashumaro.codecs.json import JSONDecoder
from mashumaro.codecs.toml import TOMLDecoder
from mashumaro.codecs.yaml import YAMLDecoder
from mashumaro.types import Discriminator

from rupture_generator import _core
from rupture_generator.config.core import ConfigObject
from rupture_generator.config.slip_rate import SlipRateShapeConfig
from rupture_generator.config.validation import (
    DepthKm,
    DipDeg,
    Magnitude,
    NonEmptyStr,
    PositiveFloat,
    PositiveInt,
    RakeDeg,
    UnitInterval,
    VelocityFraction,
    non_empty,
    positive,
)

SPECTRUM_MODELS = {
    "somerville": _core.SpectrumModel.Somerville,
    "mai": _core.SpectrumModel.Mai,
    "frankel": _core.SpectrumModel.Frankel,
    "mai_somerville": _core.SpectrumModel.MaiSomerville,
    "suzuki": _core.SpectrumModel.Suzuki,
    "input_corners": _core.SpectrumModel.InputCorners,
}
"""Spelled in the config's own lower-case, mapped once.

The core's enum is `CamelCase` because it is Rust. Writing `Mai` in a TOML file would be
carrying a language convention into a document that has no reason to know about it.
"""

SpectrumModelName = Literal[
    "somerville", "mai", "frankel", "mai_somerville", "suzuki", "input_corners"
]

RISE_TIME_WEIGHTINGS = {
    "uniform": _core.RiseTimeWeighting.Uniform,
    "by_slip": _core.RiseTimeWeighting.BySlip,
    "by_slip_and_rupture_speed": _core.RiseTimeWeighting.BySlipAndRuptureSpeed,
}

WeightingName = Literal["uniform", "by_slip", "by_slip_and_rupture_speed"]

RANDOM_ENGINES = {
    "genslip_lcg": _core.RandomEngine.GenslipLcg,
    "pcg": _core.RandomEngine.Pcg,
}
"""Spelled the config's way, mapped once -- as with the spectrum models."""

EngineName = Literal["genslip_lcg", "pcg"]


@dataclasses.dataclass
class RampConfig(ConfigObject):
    """A linear ramp between two depths, in kilometres."""

    centre_km: DepthKm
    half_width_km: PositiveFloat

    def to_core(self) -> _core.Ramp:
        """The compiled spec this transliterates, argument for argument."""
        return _core.Ramp(self.centre_km, self.half_width_km)


@dataclasses.dataclass
class HypocentreConfig(ConfigObject):
    """Where the rupture starts, in the fault's own coordinates.

    Arc lengths rather than indices, and rather than the SRF's ``shyp``:
    ``strike_km`` from the ``i = 0`` end of the plane and ``dip_km`` from its top edge.
    Both are in-fault distances, so they mean the same thing whatever the plane is cut
    into -- which an index does not.

    ``surface`` names which surface of the mesh, and may be omitted when the mesh holds
    exactly one.
    """

    strike_km: DepthKm
    dip_km: DepthKm
    surface: str | None = None


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

    def to_core(self) -> _core.VelocityModel1D:
        """The compiled spec this transliterates, argument for argument."""
        import numpy as np

        return _core.VelocityModel1D(
            np.asarray(self.bottom_depth_km, dtype=np.float64),
            np.asarray(self.shear_speed_km_s, dtype=np.float64),
            np.asarray(self.density_g_cm3, dtype=np.float64),
        )


@dataclasses.dataclass
class SourceConfig(ConfigObject):
    """What the earthquake is. Tagged: a finite fault and a point source are different
    calls into the core, taking different specs."""

    class Config(ConfigObject.Config):
        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclasses.dataclass
class FiniteSourceConfig(SourceConfig):
    """A finite fault: `_core.SourceSpec`, field for field."""

    magnitude: Magnitude
    average_dip_deg: DipDeg
    average_rake_deg: RakeDeg
    model: SpectrumModelName = "mai"
    strike_offset: float = 2.50
    dip_offset: float = 1.50
    use_moment_magnitude: bool = True
    modified_corners: bool = False
    circular_average: bool = False
    saturation_magnitude: Magnitude = 6.3
    strike_exponent: float = 0.5
    dip_exponent: float = 0.5
    rise_time_coefficient: PositiveFloat = 1.6
    type: Literal["finite"] = "finite"

    def to_core(self) -> _core.SourceSpec:
        """The compiled spec this transliterates, argument for argument."""
        return _core.SourceSpec(
            self.magnitude,
            SPECTRUM_MODELS[self.model],
            self.strike_offset,
            self.dip_offset,
            average_dip_deg=self.average_dip_deg,
            average_rake_deg=self.average_rake_deg,
            use_moment_magnitude=self.use_moment_magnitude,
            modified_corners=self.modified_corners,
            circular_average=self.circular_average,
            saturation_magnitude=self.saturation_magnitude,
            strike_exponent=self.strike_exponent,
            dip_exponent=self.dip_exponent,
            rise_time_coefficient=self.rise_time_coefficient,
        )


@dataclasses.dataclass
class PointSourceConfig(SourceConfig):
    """A point source: `_core.PointSourceSpec`, field for field.

    Not a finite source with fields left blank. There is no spectrum, so no corner
    relation, and ``rise_time_s`` is given rather than derived from the moment -- as the
    **fault-wide average**, which the depth ramp redistributes around.
    """

    magnitude: Magnitude
    rise_time_s: PositiveFloat
    average_dip_deg: DipDeg
    average_rake_deg: RakeDeg
    use_moment_magnitude: bool = True
    type: Literal["point"] = "point"

    def to_core(self) -> _core.PointSourceSpec:
        """The compiled spec this transliterates, argument for argument."""
        return _core.PointSourceSpec(
            self.magnitude,
            self.rise_time_s,
            average_dip_deg=self.average_dip_deg,
            average_rake_deg=self.average_rake_deg,
            use_moment_magnitude=self.use_moment_magnitude,
        )


@dataclasses.dataclass
class SlipConfig(ConfigObject):
    """How the slip and rake fields are shaped and trimmed: `_core.SlipSpec`.

    ``coefficient_of_variation`` is the slip field's spread and is dimensionless;
    ``rake_sigma_deg`` is the rake field's and is in **degrees**. Handing one to the
    other is `DEFECTS.md` 14, which gave every rake a spread of 0.75 degrees where the
    original gives 15 -- a factor of twenty, on every fault. They are never both bare
    numbers in the same expression here, and their names carry the difference.
    """

    model: SpectrumModelName = "mai"
    coefficient_of_variation: PositiveFloat = 0.75
    rake_sigma_deg: PositiveFloat = 15.0
    min_wavelength_km: PositiveFloat = 1.5
    max_wavelength_km: PositiveFloat = 80.0
    strike_shift: float = 0.0
    dip_shift: float = 0.0
    side_taper: UnitInterval = 0.02
    top_taper: UnitInterval = 0.0
    bottom_taper: UnitInterval = 0.0
    truncate_negative: bool = True
    water_level: float = 0.0

    def __post_init__(self) -> None:
        """Validate the fields, then the invariants between them."""
        super().__post_init__()
        if self.max_wavelength_km <= self.min_wavelength_km:
            self.refuse(
                "max_wavelength_km",
                f"must be above min_wavelength_km ({self.min_wavelength_km}), "
                f"got {self.max_wavelength_km}",
            )

    def to_core(self) -> _core.SlipSpec:
        """The compiled spec this transliterates, argument for argument."""
        return _core.SlipSpec(
            SPECTRUM_MODELS[self.model],
            coefficient_of_variation=self.coefficient_of_variation,
            rake_sigma_deg=self.rake_sigma_deg,
            min_wavelength_km=self.min_wavelength_km,
            max_wavelength_km=self.max_wavelength_km,
            strike_shift=self.strike_shift,
            dip_shift=self.dip_shift,
            side_taper=self.side_taper,
            top_taper=self.top_taper,
            bottom_taper=self.bottom_taper,
            truncate_negative=self.truncate_negative,
            water_level=self.water_level,
        )


@dataclasses.dataclass
class TimingConfig(ConfigObject):
    """How rupture time and rise time relate to slip: `_core.TimingSpec`.

    ``shallow_ramp`` and ``deep_ramp`` stretch **rise time**. Rupture speed has ramps of
    its own, which default to the rise-time ones because that is the case the original's
    four independent parameters share; ``shallow_speed_ramp`` and ``deep_speed_ramp``
    override them when they do not. `DEFECTS.md` 13 was one pair reaching both.
    """

    rupture_time_scale: float
    rise_time_blend: RampConfig
    shallow_ramp: RampConfig
    deep_ramp: RampConfig
    beta_shallow_ramp: RampConfig
    beta_mid_ramp: RampConfig
    rupture_time_correlation: float = 0.8
    rupture_time_sigma: PositiveFloat = 1.0
    rupture_delay_s: DepthKm = 0.0
    rise_time_correlation: float = 0.9
    rise_time_sigma: PositiveFloat = 0.75
    slip_exponent: float = 0.5
    shallow_rise_factor: PositiveFloat = 2.0
    deep_rise_factor: PositiveFloat = 2.0
    shallow_speed_ramp: RampConfig | None = None
    deep_speed_ramp: RampConfig | None = None
    shallow_speed_factor: PositiveFloat = 0.6
    deep_speed_factor: PositiveFloat = 0.6
    weighting: WeightingName = "by_slip_and_rupture_speed"
    beta_shallow: PositiveFloat = 0.5
    beta_mid: PositiveFloat = 0.13
    beta_deep: PositiveFloat = 0.13
    slip_rate_shape: SlipRateShapeConfig | None = None
    sample_interval_s: PositiveFloat = 0.005
    max_samples: PositiveInt = 100_000

    def to_core(self) -> _core.TimingSpec:
        """The compiled spec this transliterates, argument for argument."""

        def ramp(value: RampConfig | None) -> _core.Ramp | None:
            return None if value is None else value.to_core()

        return _core.TimingSpec(
            rupture_time_scale=self.rupture_time_scale,
            rise_time_blend=self.rise_time_blend.to_core(),
            shallow_ramp=self.shallow_ramp.to_core(),
            deep_ramp=self.deep_ramp.to_core(),
            beta_shallow_ramp=self.beta_shallow_ramp.to_core(),
            beta_mid_ramp=self.beta_mid_ramp.to_core(),
            rupture_time_correlation=self.rupture_time_correlation,
            rupture_time_sigma=self.rupture_time_sigma,
            rupture_delay_s=self.rupture_delay_s,
            rise_time_correlation=self.rise_time_correlation,
            rise_time_sigma=self.rise_time_sigma,
            slip_exponent=self.slip_exponent,
            shallow_rise_factor=self.shallow_rise_factor,
            deep_rise_factor=self.deep_rise_factor,
            shallow_speed_ramp=ramp(self.shallow_speed_ramp),
            deep_speed_ramp=ramp(self.deep_speed_ramp),
            shallow_speed_factor=self.shallow_speed_factor,
            deep_speed_factor=self.deep_speed_factor,
            weighting=RISE_TIME_WEIGHTINGS[self.weighting],
            beta_shallow=self.beta_shallow,
            beta_mid=self.beta_mid,
            beta_deep=self.beta_deep,
            slip_rate_shape=(
                None if self.slip_rate_shape is None else self.slip_rate_shape.to_core()
            ),
            sample_interval_s=self.sample_interval_s,
            max_samples=self.max_samples,
        )


@dataclasses.dataclass
class FieldConfig(ConfigObject):
    """The two per-subfault fields the geometry does not supply.

    Both are constants here. The core takes them per subfault, because a mesh may vary
    them; a config that could say so per subfault would need a way to address subfaults,
    which is a bigger thing than this needs to be yet.
    """

    base_rake_deg: RakeDeg = 175.0
    velocity_fraction: VelocityFraction = 0.8


@dataclasses.dataclass
class RandomConfig(ConfigObject):
    """Which stream of numbers, and where in it.

    ``engine`` selects the generator. ``genslip_lcg`` reproduces genslip v5.6.2 bit for
    bit -- a 31-bit truncated LCG whose normals are twelve summed uniforms -- and exists
    for comparison against it. ``pcg`` is PCG64-DXSM with a ziggurat, and is the one to
    use for anything that is not a comparison.
    """

    seed: int = 1234
    realisation: int = 0
    engine: EngineName = "genslip_lcg"

    def __post_init__(self) -> None:
        """Validate the fields, then the invariants between them."""
        super().__post_init__()
        if self.realisation < 0:
            self.refuse("realisation", f"must be 0 or more, got {self.realisation}")

    def to_core(self) -> _core.RandomEngine:
        """The compiled engine this names."""
        return RANDOM_ENGINES[self.engine]


@dataclasses.dataclass
class GridConfig(ConfigObject):
    """The wraparound margin the spectral generators need.

    Left alone unless something specifically wants otherwise. genslip rounds each padded
    extent up to even because the generators address the Nyquist row and column
    directly, and the default here is the single-plane collapse of its rule,
    ``even(int(1.10 * n))`` -- which `tests/test_boundary.py` and
    `tests/harness/mapping.py` already agree on.
    """

    padded_strike: PositiveInt | None = None
    padded_dip: PositiveInt | None = None

    @staticmethod
    def default_padding(count: int) -> int:
        """genslip's rule: ten percent more, rounded up to even."""
        padded = int(1.10 * count)
        return padded + 1 if padded % 2 else padded


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
        engine = "pcg"
        seed = 1234
    """

    hypocentre: HypocentreConfig
    velocity_model: VelocityModelConfig
    source: SourceConfig
    timing: TimingConfig
    slip: SlipConfig = dataclasses.field(default_factory=SlipConfig)
    field: FieldConfig = dataclasses.field(default_factory=FieldConfig)
    random: RandomConfig = dataclasses.field(default_factory=RandomConfig)
    grid: GridConfig = dataclasses.field(default_factory=GridConfig)
    schema_version: int = 1
    title: NonEmptyStr | None = None

    def __post_init__(self) -> None:
        """Validate the fields, then the invariants between them."""
        super().__post_init__()
        positive(self.timing.sample_interval_s)
        if isinstance(self.source, PointSourceConfig) and self.slip != SlipConfig():
            self.refuse(
                "slip",
                "a point source draws no fields, so a [slip] section would be read and "
                "ignored -- remove it, or use a finite source",
            )


def read_config(path: Path | str, format: str | None = None) -> RuptureConfig:
    """Read a generate config, in whichever of the three spellings it is written.

    Parameters
    ----------
    path : Path or str
        The file.
    format : str, optional
        ``"toml"``, ``"yaml"`` or ``"json"``. Inferred from the extension when omitted,
        defaulting to TOML.

    Returns
    -------
    RuptureConfig

    Raises
    ------
    InvalidFieldValue, MissingField
        If the file parses but does not describe a rupture.
    tomllib.TOMLDecodeError, json.JSONDecodeError, yaml.YAMLError
        If it does not parse. The CLI renders these differently -- a syntax error wants
        a line number and the line, and a validation error wants a key.
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
    "FieldConfig",
    "FiniteSourceConfig",
    "GridConfig",
    "HypocentreConfig",
    "PointSourceConfig",
    "RampConfig",
    "RandomConfig",
    "RuptureConfig",
    "SlipConfig",
    "SourceConfig",
    "TimingConfig",
    "VelocityModelConfig",
    "read_config",
    "read_geometry",
    "tomllib",
]
