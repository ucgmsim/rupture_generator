"""Type stubs for the compiled rupture generator.

Hand-written, and checked against the extension member by member by
`tests/test_boundary.py`. A stub that drifts from what it describes is worse than
none, because it reads as documentation.
"""

import enum

import numpy as np

FloatArray = np.ndarray[tuple[int], np.dtype[np.float32]]
IndexArray = np.ndarray[tuple[int], np.dtype[np.uint64]]
Grid = np.ndarray[tuple[int, int], np.dtype[np.float32]]

class SpectrumModel(enum.Enum):
    """Which relation maps magnitude onto the slip spectrum's corners."""

    Somerville = ...
    Mai = ...
    Frankel = ...
    MaiSomerville = ...
    Suzuki = ...
    InputCorners = ...

class RiseTimeWeighting(enum.Enum):
    """How the fault-wide rise-time constant is averaged."""

    Uniform = ...
    BySlip = ...
    BySlipAndRuptureSpeed = ...

class SlipRateShape:
    """Which slip-rate function every subfault gets.

    A class with named constructors rather than an enum, because four of the shapes
    carry a parameter and Python enums do not. Four of them -- `ucsb`, `ucsb2`,
    `ucsb_t` and `ucsb_var_t1` -- are the same three-piece sinusoid `oliu_p2` is,
    with the breakpoints moved.
    """

    @staticmethod
    def oliu_p2() -> SlipRateShape: ...
    @staticmethod
    def ucsb() -> SlipRateShape: ...
    @staticmethod
    def ucsb2() -> SlipRateShape: ...
    @staticmethod
    def ucsb_t(stretch: float) -> SlipRateShape: ...
    @staticmethod
    def ucsb_var_t1(tau1_ratio: float = 0.13) -> SlipRateShape: ...
    @staticmethod
    def brune() -> SlipRateShape: ...
    @staticmethod
    def urs() -> SlipRateShape: ...
    @staticmethod
    def esg2006() -> SlipRateShape: ...
    @staticmethod
    def cos() -> SlipRateShape: ...
    @staticmethod
    def seki() -> SlipRateShape: ...
    @staticmethod
    def delta() -> SlipRateShape: ...
    @staticmethod
    def from_stype(stype: str) -> SlipRateShape:
        """Parse `generic_slip2srf`'s `stype`, including `ucsb-T`'s numeric suffix.

        Raises `ValueError` on an unrecognised name, where the C falls through to
        `brune` and silently generates a different rupture.
        """

class Ramp:
    """A linear ramp between two depths, in kilometres."""

    centre_km: float
    half_width_km: float
    def __init__(self, centre_km: float, half_width_km: float) -> None: ...

class FaultGrid:
    """The discretised fault.

    `depth_km` is one value per dip row; `base_rake_deg` and `velocity_fraction` are
    one per subfault, along-strike index fastest.
    """

    subfault_count: int
    def __init__(
        self,
        fault_strike: int,
        fault_dip: int,
        padded_strike: int,
        padded_dip: int,
        strike_km: float,
        dip_km: float,
        depth_km: FloatArray,
        base_rake_deg: FloatArray,
        velocity_fraction: FloatArray,
    ) -> None: ...

class VelocityModel1D:
    """A layered one-dimensional velocity model, ordered shallow to deep.

    The three getters return the constructor's own arguments, so a model can be
    handed to something that needs the layers rather than the layers being carried
    alongside it. Each returns a fresh array; writing to one does not reach the model.
    """

    bottom_depth_km: FloatArray
    shear_speed_km_s: FloatArray
    density_g_cm3: FloatArray
    def __init__(
        self,
        bottom_depth_km: FloatArray,
        shear_speed_km_s: FloatArray,
        density_g_cm3: FloatArray,
    ) -> None: ...
    def __len__(self) -> int: ...

class SourceSpec:
    """What the earthquake is, before any field is drawn.

    `model` selects the spectral falloff shape *and* the corner relation, and the two
    do not partition the same way: Frankel has a falloff of its own while taking
    Mai's corners. `circular_average` reaches Somerville and Mai only -- the two
    relations whose branches test it.
    """

    def __init__(
        self,
        magnitude: float,
        model: SpectrumModel,
        strike_offset: float,
        dip_offset: float,
        *,
        average_dip_deg: float,
        average_rake_deg: float,
        use_moment_magnitude: bool = True,
        modified_corners: bool = False,
        circular_average: bool = False,
        saturation_magnitude: float = 6.3,
        strike_exponent: float = 0.5,
        dip_exponent: float = 0.5,
        rise_time_coefficient: float = 1.6,
    ) -> None: ...

class PointSourceSpec:
    """What a point source is, over and above its geometry.

    Not a `SourceSpec` with fields left blank: there is no spectrum, so no corner
    relation, and the rise time is given rather than derived from the moment.

    `rise_time_s` is the **fault-wide average**, which the depth ramp redistributes
    around. `generic_slip2srf` treats its `risetime` as the unstretched value
    instead, so its ramp only ever lengthens.
    """

    def __init__(
        self,
        magnitude: float,
        rise_time_s: float,
        *,
        average_dip_deg: float,
        average_rake_deg: float,
        use_moment_magnitude: bool = True,
    ) -> None: ...

class SlipSpec:
    """How the slip and rake fields are shaped and trimmed.

    `coefficient_of_variation` is the slip field's spread and is dimensionless;
    `rake_sigma_deg` is the rake field's and is in degrees. Both are spreads of a
    field drawn through the same spectrum, which is how one came to stand in for the
    other -- see `DEFECTS.md` 14.
    """

    model: SpectrumModel
    def __init__(
        self,
        model: SpectrumModel,
        *,
        coefficient_of_variation: float = 0.75,
        rake_sigma_deg: float = 15.0,
        min_wavelength_km: float = 1.5,
        max_wavelength_km: float = 80.0,
        strike_shift: float = 0.0,
        dip_shift: float = 0.0,
        side_taper: float = 0.02,
        top_taper: float = 0.0,
        bottom_taper: float = 0.0,
        truncate_negative: bool = True,
        water_level: float = 0.0,
    ) -> None: ...

class TimingSpec:
    """How rupture time and rise time relate to slip.

    `shallow_ramp` and `deep_ramp` stretch **rise time**. Rupture speed has ramps of
    its own; they default to the rise-time ones, which is the case the original's
    four independent parameters share, and `shallow_speed_ramp`/`deep_speed_ramp`
    override them when they do not.
    """

    def __init__(
        self,
        *,
        rupture_time_scale: float,
        rise_time_blend: Ramp,
        shallow_ramp: Ramp,
        deep_ramp: Ramp,
        beta_shallow_ramp: Ramp,
        beta_mid_ramp: Ramp,
        rupture_time_correlation: float = 0.8,
        rupture_time_sigma: float = 1.0,
        rupture_delay_s: float = 0.0,
        rise_time_correlation: float = 0.9,
        rise_time_sigma: float = 0.75,
        slip_exponent: float = 0.5,
        shallow_rise_factor: float = 2.0,
        deep_rise_factor: float = 2.0,
        shallow_speed_ramp: Ramp | None = None,
        deep_speed_ramp: Ramp | None = None,
        shallow_speed_factor: float = 0.6,
        deep_speed_factor: float = 0.6,
        weighting: RiseTimeWeighting = ...,
        beta_shallow: float = 0.5,
        beta_mid: float = 0.13,
        beta_deep: float = 0.13,
        slip_rate_shape: SlipRateShape | None = None,
        sample_interval_s: float = 0.005,
        max_samples: int = 100000,
    ) -> None: ...

class GeneratedRupture:
    """A generated rupture model.

    Every array is flat over subfaults, along-strike index fastest. The slip-rate
    functions are ragged, so they come back concatenated with offsets that index
    into them -- the layout `scipy.sparse.csr_array` wants.
    """

    slip_cm: FloatArray
    rake_deg: FloatArray
    onset_s: FloatArray
    rise_time_s: FloatArray
    slip_rate: FloatArray
    slip_rate_offsets: IndexArray
    sample_interval_s: float
    moment_dyne_cm: float
    alpha_t: float
    shape: tuple[int, int]
    def slip_grid(self) -> Grid: ...

def generate_rupture(
    grid: FaultGrid,
    velocity_model: VelocityModel1D,
    source: SourceSpec,
    slip: SlipSpec,
    timing: TimingSpec,
    *,
    seed: int,
    hypocentre_strike: int,
    hypocentre_dip: int,
    realisation: int = 0,
) -> GeneratedRupture:
    """Generate one rupture model. Releases the GIL for the generation itself."""

def generate_point_source(
    grid: FaultGrid,
    velocity_model: VelocityModel1D,
    point_source: PointSourceSpec,
    timing: TimingSpec,
    *,
    hypocentre_strike: int,
    hypocentre_dip: int,
) -> GeneratedRupture:
    """Generate a point source: the same rupture model, with nothing drawn.

    No seed and no realisation, because nothing here is random -- the same inputs
    give bit-identical output every time. Onset is solved for from the hypocentre
    rather than written as one number everywhere, which agrees with
    `generic_slip2srf` exactly at a single subfault and gives a rupture front across
    a discretised plane where the C has none.
    """
