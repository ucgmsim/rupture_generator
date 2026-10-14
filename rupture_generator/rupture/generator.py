"""Drawing a rupture: slip, rise time, rake and onset, on one segment or many.

:func:`generate_segment` is the whole per-segment pipeline. It takes a chart, the
:class:`~rupture_generator.rupture.medium.Medium`, the segment's
:class:`~rupture_generator.rupture.source.SegmentSource` and the shared
:class:`RuptureSettings`, and returns a :class:`SegmentRupture`. Everything stochastic
happens here, in a fixed draw order, from the one generator the caller passes in, so a
segment's fields are a pure function of ``(chart, medium, source, settings, rng
state)``. This module reads the medium and the fault profiles at the chart's cell
centres once, and the rupture keeps what it read.

Slip, rise time, rake and onset come out together because three of them share slip's
latent Gaussian. Rise time and the onset displacement both correlate against it, and
the correlation is only linear before the marginal transforms. The latent is a local
variable and never leaves this module.

Every quantity that follows the moment, such as the mean rise time and the onset
spread, comes from the segment's own target moment. A fault system with a magnitude
per segment gets a rise time and a spread per segment.

:func:`generate` walks a whole fault system. The caller has already chosen the tree
and the moment on each segment. The crossings are still open: :func:`first_arrival` finds
each child's seed from its parent's solved onsets, so drawing parents first is a real
dependency and not a reporting order.
"""

import dataclasses
import hashlib
from collections.abc import Callable, Mapping

import numpy as np

from rupture_generator._kernels import eikonal_solve
from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.geometry import CellArray, CellMask, Geometry
from rupture_generator.rupture.medium import (
    Medium,
    SpatialField,
    constant_field,
    rigidity_pa,
)
from rupture_generator.rupture.propagator import DEFAULT_JUMP_MODEL, JumpModel
from rupture_generator.rupture.realisation import Realisation
from rupture_generator.rupture.source import SegmentSource
from rupture_generator.sampling import (
    NORMAL,
    Covariance,
    Grid,
    Marginal,
    MarginalFamily,
    PreCorrected,
    latent_correlation,
    mix,
    sampler,
    standardise,
)

M2_PER_KM2 = 1.0e6

CUBE_ROOT_DYNE_CM = 1.0e-9 * (1.0e7 ** (1.0 / 3.0))
"""``1e-9`` per cube-root dyne-centimetre."""

ALPHA_COEFFICIENT = 0.1
"""How much the dip-and-rake correction can move things, proportion (0, 1)."""

DIP_PLATEAU_DEG = 45.0
"""Below this dip the correction is at full strength, falling to nothing at vertical."""

REVERSE_RAKE_DEG = 90.0
"""Pure reverse slip, where the correction is at full strength."""

MINIMUM_VELOCITY_FRACTION = 0.25
RAYLEIGH_VELOCITY_FRACTION = 0.9194
"""The sub-Rayleigh ceiling for a mode-II crack; between it and the shear speed lies
the forbidden zone."""
MAXIMUM_VELOCITY_FRACTION = np.sqrt(2.0)
"""Burridge-Andrews: the fastest a supershear front travels."""

OFF_FAULT_SLOWNESS_FACTOR = 10.0
"""What an unoccupied cell's slowness is multiplied by, so the front does not cross the
part of the rectangle that is not fault."""

CAUSAL_MARGIN = 1.05
"""How much room the causal clamp leaves past the weight that would tie with the seed.
A tie-break, not a modelling choice: it keeps the seed the strict earliest cell."""


@dataclasses.dataclass(frozen=True)
class JumpSeed:
    """Records the location and time a rupture jumps from a segment.

    Attributes
    ----------
    cell : tuple of (int, int)
        The cell, ``(i, j)`` with ``i`` down dip and ``j`` along strike.
    time_s : float
        When the front leaves it, in seconds: zero at a hypocentre, and the crossing
        time on a segment another segment triggered.
    """

    cell: tuple[int, int]
    time_s: float


def _correlation(value: float) -> None:
    if not -1.0 <= value <= 1.0:
        raise RuptureGeneratorError(f"a correlation lies in [-1, 1], got {value}")


def _positive_marginal(
    what: str, family: MarginalFamily, spread: float, maximum: float
) -> Marginal:
    if family == "normal":
        raise RuptureGeneratorError(
            f"{what} values are positive, so they cannot follow a normal marginal"
        )
    return Marginal(family, spread, maximum)


@dataclasses.dataclass(frozen=True)
class SlipSettings:
    """Slip field generation settings.

    Attributes
    ----------
    family : MarginalFamily
        The unit-mean distribution slip values follow.
    coefficient_of_variation : float
        The spread of ``family``.
    maximum : float
        The largest slip on the fault in mean slips, for a ``doubly_truncated_normal``
        alone. Infinite otherwise.
    side_taper : float
        The fraction of the fault's length from each end over which slip ramps to zero.
    top_taper : float
        The fraction of the fault's width from the top edge over which slip ramps to
        zero. At zero, slip at the top edge has full amplitude.
    bottom_taper : float
        The fraction of the fault's width from the bottom edge over which slip ramps to
        zero.
    """

    family: MarginalFamily = "truncated_exponential"
    coefficient_of_variation: float = 0.90
    maximum: float = np.inf
    side_taper: float = 0.02
    top_taper: float = 0.0
    bottom_taper: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a taper outside ``[0, 0.5]``, or a spread outside the family's range."""
        for name in ("side_taper", "top_taper", "bottom_taper"):
            if not 0.0 <= getattr(self, name) <= 0.5:
                raise RuptureGeneratorError(
                    f"{name} is a fraction of the fault's extent from one edge, so it "
                    f"lies in [0, 0.5]; got {getattr(self, name)}"
                )
        # _positive_marginal has validation logic, so invoking the property that
        # calls it triggers this validation.
        _ = self.marginal

    @property
    def marginal(self) -> Marginal:
        """Marginal: The unit-mean distribution that slip values follow."""
        return _positive_marginal(
            "slip", self.family, self.coefficient_of_variation, self.maximum
        )


@dataclasses.dataclass(frozen=True)
class RiseSettings:
    """How long each subfault slips for.

    The fault-wide mean is ``coefficient * M0^(1/3) * alpha_T``. Where rise time is
    longer or shorter, and where it follows slip more tightly, belongs to the
    :class:`FaultProfiles`.

    Attributes
    ----------
    coefficient : float
        The mean's coefficient in the published units, per cube-root dyne-centimetre
        at ``1e-9``. Graves & Pitarka use 1.6.
    correlation : float
        The correlation of rise time with slip.
    slip_exponent : float
        The power of slip that rise time follows. At 0.5, rise time goes as the square
        root of slip.
    family : MarginalFamily
        The unit-mean distribution rise time values follow, a gamma by default.
    coefficient_of_variation : float
        The spread of ``family``. Below unit spread the gamma's mode is away from zero.
        No subfault then has a zero rise time.
    maximum : float
        The longest rise time in mean rise times, for a ``doubly_truncated_normal``
        alone. Infinite otherwise.
    """

    coefficient: float = 1.6
    correlation: float = 0.9
    slip_exponent: float = 0.5
    family: MarginalFamily = "gamma"
    coefficient_of_variation: float = 0.75
    maximum: float = np.inf

    def __post_init__(self) -> None:
        """Refuse a rise time the power law can't produce."""
        if self.slip_exponent <= 0.1:
            raise RuptureGeneratorError(
                f"a slip exponent of {self.slip_exponent} abandons the correlated field "
                "for independent noise, which is a different model; use one above 0.1"
            )
        _correlation(self.correlation)
        _ = self.marginal

    @property
    def marginal(self) -> Marginal:
        """Marginal: The unit-mean distribution that rise time values follow."""
        return _positive_marginal(
            "rise time", self.family, self.coefficient_of_variation, self.maximum
        )


@dataclasses.dataclass(frozen=True)
class RakeSettings:
    """How far each subfault's rake strays from its segment's: ``sigma * Z`` degrees.

    By design, rake doesn't depend on slip. A patch with more slip has no reason to
    move in a different direction. The marginal is normal, so rake is the one field
    that keeps slip's correlation lengths exactly, with no pre-correction.

    Attributes
    ----------
    sigma_deg : float
        The standard deviation of the rake about the segment's, in degrees.
    """

    sigma_deg: float = 15.0

    def __post_init__(self) -> None:
        """Refuse a negative spread."""
        if self.sigma_deg < 0.0:
            raise RuptureGeneratorError(
                f"a spread has no sign, got {self.sigma_deg} degrees"
            )


@dataclasses.dataclass(frozen=True)
class TimingSettings:
    """Settings for how the generator solves and correlates a segment's time field.

    The front moves at ``velocity_fraction / alpha_T`` of the chart's shear speed, held
    to the sub-Rayleigh or supershear branch, times the chart's rupture speed factor.
    The displacement's spread follows the moment as ``offset_s + coefficient *
    M0^(1/3)`` in the published units. With both terms zero the front is coherent.

    Attributes
    ----------
    velocity_fraction : float
        The rupture speed as a fraction of the shear speed, before the geometric
        correction.
    offset_s : float
        The constant term of the onset spread, in seconds: genslip's ``tsfac_bzero``,
        read as a magnitude.
    coefficient : float
        The moment term's coefficient in the published units: genslip's
        ``tsfac_slope``, read as a magnitude.
    correlation : float
        The correlation of the displacement with slip. A positive value makes
        high-slip patches rupture early.
    blend_sigma : float
        The width of the zone over which the displacement grows in from the seed, in
        units of the onset spread.
    """

    velocity_fraction: float = 0.8
    offset_s: float = 0.1
    coefficient: float = 0.5
    correlation: float = 0.8
    blend_sigma: float = 4.0

    def __post_init__(self) -> None:
        """Refuse a speed outside the allowed band, or a negative spread."""
        if not 0.0 < self.velocity_fraction <= MAXIMUM_VELOCITY_FRACTION:
            raise RuptureGeneratorError(
                f"a velocity fraction lies in (0, {MAXIMUM_VELOCITY_FRACTION:.4f}], "
                f"got {self.velocity_fraction}"
            )
        if self.offset_s < 0.0 or self.coefficient < 0.0:
            raise RuptureGeneratorError(
                "the onset spread's offset and coefficient are magnitudes"
            )
        if self.blend_sigma <= 0.0:
            raise RuptureGeneratorError(
                f"the onset blend spans {self.blend_sigma} sigma, which is no width"
            )
        _correlation(self.correlation)


UNMODIFIED = constant_field(1.0)
"""Constant profile for unmodified spatial fields."""


@dataclasses.dataclass(frozen=True)
class FaultProfiles:
    """Spatial fields for rupture propagation and rise time.

    Attributes
    ----------
    rise_time_factor : SpatialField
        The relative rise time.
    rise_time_slip_weight : SpatialField
        How much of rise time's correlation with slip is the configured value (1)
        and how much is exact (0). At 0 the rise-time latent is slip's own, so the
        two fields share their rank order. This is the shallow treatment of Graves &
        Pitarka. There, in the velocity-strengthening crust, the pulse length tracks
        the slip.
    rupture_speed_factor : SpatialField
        The factor on the front's speed, for a slower front near the ground surface or
        at depth.
    """

    rise_time_factor: SpatialField = UNMODIFIED
    rise_time_slip_weight: SpatialField = UNMODIFIED
    rupture_speed_factor: SpatialField = UNMODIFIED


@dataclasses.dataclass(frozen=True)
class RuptureSettings:
    """Complete set of rupture settings.

    Attributes
    ----------
    slip : SlipSettings
        The slip field's settings.
    rise : RiseSettings
        The rise time field's settings.
    rake : RakeSettings
        The rake field's settings.
    timing : TimingSettings
        The onset field's settings.
    profiles : FaultProfiles
        The depth-varying modifiers on rise time and rupture speed.
    """

    slip: SlipSettings = dataclasses.field(default_factory=SlipSettings)
    rise: RiseSettings = dataclasses.field(default_factory=RiseSettings)
    rake: RakeSettings = dataclasses.field(default_factory=RakeSettings)
    timing: TimingSettings = dataclasses.field(default_factory=TimingSettings)
    profiles: FaultProfiles = dataclasses.field(default_factory=FaultProfiles)


@dataclasses.dataclass(frozen=True, eq=False)
class SegmentRupture:
    """Fields and statistics drawn for a rupture segment.

    Attributes
    ----------
    geometry : Geometry
        The segment's chart.
    slip_m : CellArray
        Slip, in metres.
    rise_time_s : CellArray
        Rise time, in seconds.
    rake_deg : CellArray
        Rake, in degrees.
    onset_s : CellArray
        When each subfault starts to move, in seconds from the hypocentre's onset.
    speed_km_s : CellArray
        The rupture speed of the front, in kilometres per second.
    shear_speed_km_s : CellArray
        The medium's shear speed at the cell centres, in kilometres per second.
    density_g_cm3 : CellArray
        The medium's density at the cell centres, in grams per cubic centimetre.
        With the shear speed, this sets the rigidity of the moment.
    alpha_t : float
        Graves & Pitarka's geometric correction for this segment.
    rise_time_mean_s : float
        The fault-wide mean rise time, in seconds.
    onset_scale_s : float
        The onset displacement's spread, in seconds.
    """

    geometry: Geometry
    slip_m: CellArray
    rise_time_s: CellArray
    rake_deg: CellArray
    onset_s: CellArray
    speed_km_s: CellArray
    shear_speed_km_s: CellArray
    density_g_cm3: CellArray
    alpha_t: float
    rise_time_mean_s: float
    onset_scale_s: float

    @property
    def rigidity_pa(self) -> CellArray:
        """CellArray: Rigidity in pascals, from the rock this segment read."""
        return rigidity_pa(self.shear_speed_km_s, self.density_g_cm3)


# ---------------------------------------------------------------- shared readings


def _alpha_t(average_dip_deg: float, average_rake_deg: float) -> float:
    """Graves & Pitarka's geometric correction, in ``[1/1.1, 1]``.

    It's exactly 1 for a vertical strike-slip fault, and it shortens the rise time and
    raises the rupture speed by the same factor. The averaged rake wraps into
    ``[-180, 180]`` first. The correction is for reverse geometries, so normal faulting
    doesn't get one.
    """
    if average_dip_deg <= DIP_PLATEAU_DEG:
        dip_factor = 1.0
    else:
        dip_factor = 1.0 - (average_dip_deg - DIP_PLATEAU_DEG) / (
            90.0 - DIP_PLATEAU_DEG
        )

    rake_deg = (average_rake_deg + 180.0) % 360.0 - 180.0
    if 0.0 <= rake_deg <= 180.0:
        rake_factor = 1.0 - abs(rake_deg - REVERSE_RAKE_DEG) / REVERSE_RAKE_DEG
    else:
        rake_factor = 0.0

    return 1.0 / (1.0 + dip_factor * rake_factor * ALPHA_COEFFICIENT)


def _cube_root_scaling(coefficient: float, moment_nm: float) -> float:
    """``coefficient * M0^(1/3)`` with the coefficient in published dyne-cm units."""
    return coefficient * CUBE_ROOT_DYNE_CM * float(np.cbrt(moment_nm))


# ------------------------------------------------------------------------ slip


def _reach(mask: CellMask, axis: int, *, reverse: bool) -> np.ndarray:
    """How many occupied cells run up to each cell along one axis, inclusive.

    This counts the natural generalisation of rectangular fault based reach tapering.
    """
    ordered = np.flip(mask, axis) if reverse else mask
    index = np.arange(ordered.shape[axis], dtype=np.int32)
    index = index[:, None] if axis == 0 else index
    last_gap = np.maximum.accumulate(np.where(ordered, np.int32(-1), index), axis=axis)
    counted = index - last_gap
    return np.flip(counted, axis) if reverse else counted


def _taper_edges(field: CellArray, occupied: CellMask, slip: SlipSettings) -> CellArray:
    """Ramp a field to zero at the fault's edges, and zero the unoccupied cells.

    The taper is separable, a product of one ramp per edge in whole cells. The edges are
    the fault's and not the chart's. Where the ``occupied`` mask is partial, the ramps
    follow the outline of the occupied cells rather than the chart's rectangle.
    """
    ramp = np.ones(field.shape, dtype=np.float64)
    for fraction, axis, reverse in (
        (slip.top_taper, 0, False),
        (slip.bottom_taper, 0, True),
        (slip.side_taper, 1, False),
        (slip.side_taper, 1, True),
    ):
        cells = int(fraction * field.shape[axis] + 0.5)
        if cells:
            ramp *= np.minimum(_reach(occupied, axis, reverse=reverse) / cells, 1.0)
    return field * ramp * occupied


def _scale_to_moment(
    pattern: CellArray, rigidity_pa: CellArray, areas_km2: CellArray, moment_nm: float
) -> CellArray:
    """Scale a slip pattern, in metres, so the segment's moment is ``moment_nm``."""
    total = float(np.sum(rigidity_pa * areas_km2 * M2_PER_KM2 * pattern))
    if not total > 0.0:
        raise RuptureGeneratorError(
            "the slip pattern carries no moment anywhere -- every subfault was tapered "
            "or masked to zero"
        )
    return (moment_nm / total) * pattern


# --------------------------------------------------------------------- the front


def _speed_field(
    shear_speed_km_s: CellArray,
    timing: TimingSettings,
    geometric_correction: float,
) -> CellArray:
    """Rupture speed along fault."""
    fraction = timing.velocity_fraction / geometric_correction
    if fraction > RAYLEIGH_VELOCITY_FRACTION:
        fraction += 1.0 - RAYLEIGH_VELOCITY_FRACTION
    fraction = min(max(fraction, MINIMUM_VELOCITY_FRACTION), MAXIMUM_VELOCITY_FRACTION)
    speed = fraction * shear_speed_km_s
    if not np.all(speed > 0.0):
        worst = np.unravel_index(int(np.argmin(speed)), speed.shape)
        raise RuptureGeneratorError(
            f"the rupture speed at subfault {tuple(int(k) for k in worst)} is "
            f"{float(speed[worst]):.4g} km/s; check the shear speed field"
        )
    return speed


def _travel_times(
    geometry: Geometry, speed_km_s: CellArray, seed: JumpSeed
) -> CellArray:
    """First arrivals on ``(i, j)`` in seconds: the coherent front from the seed.

    The solve is ``|grad T| = 1/v`` over the smooth speed field.
    """
    slowness = np.where(
        geometry.occupied, 1.0 / speed_km_s, OFF_FAULT_SLOWNESS_FACTOR / speed_km_s
    )
    strike_km, dip_km = geometry.spacing_km
    return eikonal_solve(
        np.ascontiguousarray(slowness),
        (dip_km, strike_km),
        [(seed.cell[0], seed.cell[1], seed.time_s)],
    )


def _blend_onset(
    travel_s: CellArray,
    displacement_s: CellArray,
    scale_s: float,
    seed: JumpSeed,
    blend_sigma: float,
) -> CellArray:
    """Blend in rupture propagation perturbation terms preserving causality.

    ``t = T + min(tau / (n sigma), tau / (c max(-delta, 0)), 1) * delta`` with ``tau``
    the time since the seed and ``delta`` the displacement with its seed value removed.
    The first term is the model, the width of the zone over which roughness accumulates.
    The second is arithmetic, per cell. It stops any subfault from rupturing before the
    front that seeded it, and stops one deep dip from holding back the whole fault
    behind it.
    """
    if scale_s == 0.0:
        return travel_s
    since_seed_s = travel_s - seed.time_s
    displacement = displacement_s - displacement_s[seed.cell]
    weight = np.minimum(1.0, since_seed_s / (blend_sigma * scale_s))

    # Where the dip is zero or the draw moves the cell later, the ratio is astronomical
    # and the minimum ignores it: such a cell is under no causal bound at all.
    dip = np.maximum(-displacement, 0.0)
    with np.errstate(divide="ignore", over="ignore"):
        causal = since_seed_s / (CAUSAL_MARGIN * np.maximum(dip, 1e-300))
    return travel_s + np.minimum(weight, causal) * displacement


def _read(
    geometry: Geometry,
    name: str,
    field: SpatialField,
    accept: Callable[[CellArray], np.ndarray],
    wanted: str,
) -> CellArray:
    """A field at a chart's cell centres, checked.

    Raises
    ------
    RuptureGeneratorError
        If the field gives the wrong shape, or a value ``accept`` refuses.
    """
    values = np.asarray(field(geometry.centres), dtype=np.float64)
    if values.shape != geometry.cells:
        raise RuptureGeneratorError(
            f"{name} is shaped {values.shape} and the chart has {geometry.cells} cells"
        )
    if not np.all(accept(values)):
        raise RuptureGeneratorError(
            f"{name} runs {float(values.min()):.3g} to {float(values.max()):.3g}; "
            f"it wants {wanted}"
        )
    return values


def _positive(values: CellArray) -> np.ndarray:
    return np.isfinite(values) & (values > 0.0)


def generate_segment(
    geometry: Geometry,
    medium: Medium,
    source: SegmentSource,
    settings: RuptureSettings,
    *,
    seed: JumpSeed,
    rng: np.random.Generator,
) -> SegmentRupture:
    """Draw one segment's slip, rise time, rake and onset.

    The draws from ``rng`` come in a fixed order: slip's latent with rise time's
    independent latent, then rake, then the onset displacement's independent latent.
    Slip, rise and the displacement share one embedding, pre-corrected for slip's
    marginal, because this function mixes them in slip's latent space. Rake has its
    own embedding, uncorrected, because its marginal is normal.

    Parameters
    ----------
    geometry : Geometry
        The segment's chart.
    medium : Medium
        The rock, read at the chart's cell centres.
    source : SegmentSource
        The segment's moment, rake and slip covariance.
    settings : RuptureSettings
        How to draw each field.
    seed : JumpSeed
        Where and when the front starts on this segment.
    rng : numpy.random.Generator
        The segment's own generator.

    Returns
    -------
    SegmentRupture
        The drawn fields, with the rock they used.

    Raises
    ------
    RuptureGeneratorError
        If the medium or a profile reads out of range on this chart, the pattern has
        zero moment, the speed field isn't positive, or the covariance doesn't embed on
        this chart.
    """
    slip, rise, timing = settings.slip, settings.rise, settings.timing
    profiles = settings.profiles
    shear_speed_km_s = _read(
        geometry, "shear speed", medium.shear_speed_km_s, _positive, "positive values"
    )
    density_g_cm3 = _read(
        geometry, "density", medium.density_g_cm3, _positive, "positive values"
    )
    rise_time_factor = _read(
        geometry,
        "the rise-time factor",
        profiles.rise_time_factor,
        _positive,
        "positive values",
    )
    rise_time_slip_weight = _read(
        geometry,
        "the rise-time slip weight",
        profiles.rise_time_slip_weight,
        lambda weight: (weight >= 0.0) & (weight <= 1.0),
        "weights in [0, 1]",
    )
    rupture_speed_factor = _read(
        geometry,
        "the rupture speed factor",
        profiles.rupture_speed_factor,
        _positive,
        "positive values, or the front never arrives",
    )

    correction = _alpha_t(
        float(np.mean(geometry.dip_deg[geometry.occupied])), source.rake_deg
    )
    moment_nm = source.moment_nm
    strike_km, dip_km = geometry.spacing_km
    grid = Grid(geometry.cells, (dip_km, strike_km))
    covariance = source.covariance
    latent = sampler(
        grid,
        Covariance(
            PreCorrected(covariance.correlation, slip.marginal), covariance.lengths_km
        ),
    )
    # Latent sampler draws a complex field and returns both independent
    # components instead of throwing away a field.
    slip_latent, rise_independent = latent.draw(rng)
    pattern = _taper_edges(slip.marginal.apply(slip_latent), geometry.occupied, slip)
    slip_m = _scale_to_moment(
        pattern,
        rigidity_pa(shear_speed_km_s, density_g_cm3),
        geometry.areas_km2,
        moment_nm,
    )

    mean_rise_s = _cube_root_scaling(rise.coefficient, moment_nm) * correction
    rho = float(
        latent_correlation(slip.marginal, rise.marginal, np.array(rise.correlation))
    )

    weight = rise_time_slip_weight
    rise_latent = mix(
        (weight * rho + (1.0 - weight), slip_latent),
        (weight * np.sqrt(1.0 - rho * rho), rise_independent),
    )
    rise_pattern = (
        rise.marginal.apply(rise_latent) ** rise.slip_exponent * rise_time_factor
    )
    rise_time_s = rise_pattern * (
        mean_rise_s / float(rise_pattern[geometry.occupied].mean())
    )

    rake_latent, _ = sampler(grid, covariance).draw(rng)
    rake_deg = source.rake_deg + settings.rake.sigma_deg * standardise(rake_latent)

    speed_km_s = (
        _speed_field(shear_speed_km_s, timing, correction) * rupture_speed_factor
    )
    travel_s = _travel_times(geometry, speed_km_s, seed)
    scale_s = timing.offset_s + _cube_root_scaling(timing.coefficient, moment_nm)
    onset_independent, _ = latent.draw(rng)
    rho = float(latent_correlation(slip.marginal, NORMAL, np.array(timing.correlation)))
    # genslip's tsfac is negative and this package reads it as a magnitude, so the
    # displacement subtracts: with a positive correlation, high slip ruptures early.
    displacement_s = -scale_s * standardise(
        mix((rho, slip_latent), (np.sqrt(1.0 - rho * rho), onset_independent))
    )

    return SegmentRupture(
        geometry=geometry,
        slip_m=slip_m,
        rise_time_s=rise_time_s,
        rake_deg=rake_deg,
        onset_s=_blend_onset(
            travel_s, displacement_s, scale_s, seed, timing.blend_sigma
        ),
        speed_km_s=speed_km_s,
        shear_speed_km_s=shear_speed_km_s,
        density_g_cm3=density_g_cm3,
        alpha_t=correction,
        rise_time_mean_s=mean_rise_s,
        onset_scale_s=scale_s,
    )


# --------------------------------------------------------------- the whole system


def _segment_rng(seed: int, segment: str) -> np.random.Generator:
    """The generator one segment draws from, keyed by its **name**.

    Keying by name and not by position makes a fault system's segments independent of
    each other's presence. Adding or dropping a segment, or reordering the file, leaves
    every other segment's fields bit-identical. Without that, two realisations of a
    system wouldn't be comparable at all.

    The key is a BLAKE2b hash of the name. Python's :func:`hash` salts string hashes
    per interpreter, so it reproduces nothing between runs.
    """
    key = int.from_bytes(
        hashlib.blake2b(segment.encode(), digest_size=8).digest(), "big"
    )
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(key,)))


type JumpRule = Callable[
    [SegmentRupture, Geometry, Medium, JumpModel, np.random.Generator], JumpSeed
]
"""Where and when the front crosses from a drawn parent onto a child.

Its arguments are the parent, the child's chart, the medium between them, the jump
model, and the jump's own generator."""


def first_arrival(
    parent: SegmentRupture,
    child: Geometry,
    medium: Medium,
    model: JumpModel,
    rng: np.random.Generator,
) -> JumpSeed:
    """Find where and when the front crosses from a drawn parent onto a child.

    Parameters
    ----------
    parent : SegmentRupture
        The drawn segment that triggers the child.
    child : Geometry
        The chart of the segment to start.
    medium : Medium
        The rock between the two, which sets the crossing time.
    model : JumpModel
        How far a jump can land.
    rng : numpy.random.Generator
        The jump's own generator.

    Returns
    -------
    JumpSeed
        The child's starting cell and time.

    Raises
    ------
    RuptureGeneratorError
        If either chart has no fault cells.
    """
    landings, departures, gap_km = child.nearest_cells_to(parent.geometry)
    reach_km = model.reach_km(float(gap_km.min()), rng)
    arrival_s = parent.onset_s[departures] + medium.crossing_time_s(
        parent.geometry.centres[departures], child.centres[landings]
    )
    arrival_s = np.where(gap_km <= reach_km, arrival_s, np.inf)
    first = int(np.argmin(arrival_s))
    return JumpSeed(
        cell=(int(landings[0][first]), int(landings[1][first])),
        time_s=float(arrival_s[first]),
    )


def generate(
    realisation: Realisation[Geometry],
    medium: Medium,
    sources: Mapping[str, SegmentSource],
    settings: RuptureSettings,
    *,
    seed: int,
    jump_rule: JumpRule = first_arrival,
    jump_propagation_model: JumpModel = DEFAULT_JUMP_MODEL,
) -> Realisation[SegmentRupture]:
    """Draw every segment of a fault system, parents before children.

    This function adds where and when each front starts. The root starts at the
    hypocentre, and ``jump_rule`` finds every other start from the parent drawn just
    before.

    Parameters
    ----------
    realisation : Realisation of Geometry
        The fault system, with its hypocentre and tree.
    medium : Medium
        The rock.
    sources : Mapping of str to SegmentSource
        Each segment's source, by name.
    settings : RuptureSettings
        How to draw each field.
    seed : int
        The seed every segment's generator derives from.
    jump_rule : JumpRule
        How to find a child's seed from its parent.
    jump_propagation_model : JumpModel
        How far a jump can land.

    Returns
    -------
    Realisation of SegmentRupture
        The same realisation with every segment drawn, in causal order.

    Raises
    ------
    RuptureGeneratorError
        If a segment has no source, or the realisation has no hypocentre or no tree
        to walk.
    """
    missing = sorted(set(realisation) - set(sources))
    if missing:
        raise RuptureGeneratorError(f"{missing} have no source")

    drawn: dict[str, SegmentRupture] = {}
    for name, parent, geometry in realisation.in_causal_order():
        if parent is None:
            hypocentre = realisation.hypocentre
            # `in_causal_order` refuses a realisation without one.
            assert hypocentre is not None
            start = JumpSeed(
                geometry.cell_at(hypocentre.strike_km, hypocentre.dip_km), 0.0
            )
        else:
            start = jump_rule(
                drawn[parent],
                geometry,
                medium,
                jump_propagation_model,
                _segment_rng(seed, f"jump:{name}"),
            )
        drawn[name] = generate_segment(
            geometry,
            medium,
            sources[name],
            settings,
            seed=start,
            rng=_segment_rng(seed, name),
        )
    return realisation.with_segments(drawn)


__all__ = [
    "CAUSAL_MARGIN",
    "UNMODIFIED",
    "FaultProfiles",
    "JumpRule",
    "JumpSeed",
    "RakeSettings",
    "RiseSettings",
    "RuptureSettings",
    "SegmentRupture",
    "SlipSettings",
    "TimingSettings",
    "first_arrival",
    "generate",
    "generate_segment",
]
