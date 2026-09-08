"""Drawing a rupture: slip, rise time, rake and onset, on one segment or many.

:func:`generate_segment` is the whole per-segment pipeline. It takes a chart that
already carries its materials and returns the chart with the four fields the SRF wants
attached: ``slip_m``, ``rise_time_s``, ``rake_deg`` and ``onset_s``. Everything
stochastic happens here, in a fixed draw order, from the one generator passed in, so
a segment's fields are a pure function of ``(chart, parameters, rng state)``.

The four fields are one batch because three of them share slip's latent Gaussian:
rise time and the onset displacement both correlate against it, and the correlation
is only linear before the marginals are applied. The latent is a local variable and
never leaves this module.

Every quantity that follows the moment -- the mean rise time, the onset spread -- is
read off the segment's own target moment, so a fault system with a magnitude per
segment gets a rise time and a spread per segment.

:func:`generate` walks a whole fault system. The tree, the moment on each segment and
the jumps between them are decided before it is called; what it does is turn that
structure into seed times, hand each segment its own generator, and visit them
parents-first. There is no stage machinery: the walk is a loop.
"""

import dataclasses
import hashlib
from collections.abc import Mapping
from enum import StrEnum

import numpy as np

from rupture_generator._kernels import eikonal_solve
from rupture_generator.geometry import CellArray, CellMask, Geometry
from rupture_generator.rupture import Realisation
from rupture_generator.sampling import (
    NORMAL,
    Covariance,
    Grid,
    Marginal,
    PreCorrected,
    Sampler,
    latent_correlation,
    mix,
    standardise,
)


class Field(StrEnum):
    """The names of the cell fields this module reads and writes.

    A member is its own name, so ``segment[Field.SLIP]`` and ``segment["slip_m"]`` are
    the same lookup and the enum can be dropped into a chart built by anything else.
    The first four are the caller's, sampled onto the chart before the draw; the last
    four are what the draw attaches.
    """

    SHEAR_SPEED = "shear_speed_km_s"
    """The material field the front's speed is a fraction of. Required."""

    RIGIDITY = "rigidity_pa"
    """The material field the moment is counted in. Required."""

    RISE_TIME_FACTOR = "rise_time_factor"
    """Optional: the relative rise time at each subfault, 1 where it is unmodified.
    Absent, every subfault is 1. Any depth dependence -- longer pulses in the shallow
    crust, say -- is prescribed here, per cell, rather than by a built-in profile; the
    same goes for the rupture speed, which is whatever the shear speed field says."""

    RISE_TIME_SLIP_WEIGHT = "rise_time_slip_weight"
    """Optional, in ``[0, 1]``: how much of rise time's correlation with slip is the
    configured value (1) rather than exact (0). Absent, every subfault is 1. At 0 the
    rise-time latent is slip's own, so the two fields share their rank order -- the
    shallow treatment of Graves & Pitarka, where the pulse length tracks the slip
    amount in the velocity-strengthening upper crust."""

    SLIP = "slip_m"
    """Slip in metres, the pattern scaled so the segment carries its moment."""

    RISE_TIME = "rise_time_s"
    """How long each subfault slips for, in seconds."""

    RAKE = "rake_deg"
    """Which way each subfault slips, in degrees."""

    ONSET = "onset_s"
    """When each subfault starts, in seconds."""


class Attr(StrEnum):
    """The names of the per-chart scalars the draw records."""

    ALPHA_T = "alpha_t"
    """The dip-and-rake correction this segment's mean geometry asked for."""

    RISE_TIME_MEAN = "rise_time_mean_s"
    """The fault-wide mean rise time the moment set."""

    ONSET_SCALE = "onset_scale_s"
    """The spread of the onset displacement, in seconds."""

    HYPOCENTRE_STRIKE = "hypocentre_strike_km"
    """Where the rupture nucleated, along the root segment's top edge. The caller's:
    :func:`generate` reads it, and two arc lengths rather than two indices is what
    keeps a hypocentre meaningful when the chart is recut."""

    HYPOCENTRE_DIP = "hypocentre_dip_km"
    """Where the rupture nucleated, down the root segment's near edge."""


M2_PER_KM2 = 1.0e6

CUBE_ROOT_DYNE_CM = 1.0e-9 * (1.0e7 ** (1.0 / 3.0))
"""``1e-9`` per cube-root dyne-centimetre, as the published coefficients are stated,
carried to newton-metres. About 215.44; the easy factor to get wrong."""

ALPHA_COEFFICIENT = 0.1
"""How much the dip-and-rake correction can move things: at most a tenth."""

DIP_PLATEAU_DEG = 45.0
"""Below this dip the correction is at full strength, falling to nothing at vertical."""

REVERSE_RAKE_DEG = 90.0
"""Pure reverse slip, where the correction is at full strength."""

MINIMUM_VELOCITY_FRACTION = 0.25
RAYLEIGH_VELOCITY_FRACTION = 0.9194
"""The sub-Rayleigh ceiling for a mode-II crack; between it and the shear speed lies
the forbidden zone."""
MAXIMUM_VELOCITY_FRACTION = 2.0**0.5
"""Burridge-Andrews: the fastest a supershear front travels."""

OFF_FAULT_SLOWNESS_FACTOR = 10.0
"""What an unoccupied cell's slowness is multiplied by, so the front does not cross the
part of the rectangle that is not fault. Arrivals are bit-identical from x10 upward on
both CFM subduction interfaces."""

CAUSAL_MARGIN = 1.05
"""How much room the causal clamp leaves past the weight that would tie with the seed.
A tie-break, not a modelling choice: it keeps the seed the strict earliest cell."""


SLIP_MARGINAL = Marginal("truncated_exponential", 0.90)
"""Thingbaijam & Mai (2016): the family fitted to SRCMOD slip, at the spread that puts
the largest slip at 4.3 mean slips."""

RISE_MARGINAL = Marginal("gamma", 0.75)
"""A gamma below unit spread has its mode away from zero, so no subfault slips in no
time at all."""


class ParameterError(ValueError):
    """A parameter set no rupture answers to."""


@dataclasses.dataclass(frozen=True)
class SlipParameters:
    """What shapes a slip field, and the moment that sizes it.

    ``covariance`` is the correlation the *pattern* carries, after the marginal; the
    sampler is asked for a pre-corrected one so that it does. ``marginal`` is the
    unit-mean distribution of the pattern's values: the truncated exponential
    Thingbaijam & Mai (2016) fitted to SRCMOD, whose coefficient of variation also
    fixes the largest slip (0.90 puts it at 4.3 mean slips). The tapers are fractions
    of the fault's extent along each axis; ``top_taper`` is zero so slip reaches the
    surface at full amplitude.
    """

    moment_nm: float
    covariance: Covariance
    marginal: Marginal = SLIP_MARGINAL
    side_taper: float = 0.02
    top_taper: float = 0.0
    bottom_taper: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a moment or taper that is not one."""
        if not self.moment_nm > 0.0:
            raise ParameterError(
                f"a segment's moment must be positive, got {self.moment_nm}"
            )
        for name in ("side_taper", "top_taper", "bottom_taper"):
            if not 0.0 <= getattr(self, name) <= 0.5:
                raise ParameterError(
                    f"{name} is a fraction of the fault's extent from one edge, so it "
                    f"lies in [0, 0.5]; got {getattr(self, name)}"
                )
        if not self.marginal.is_positive:
            raise ParameterError(
                f"slip needs a marginal on the positive half-line, not {self.marginal.family}"
            )


@dataclasses.dataclass(frozen=True)
class RiseParameters:
    """How long each subfault slips for.

    The fault-wide mean is ``coefficient * M0^(1/3) * alpha_T`` with the coefficient in
    the published units, per cube-root dyne-centimetre at ``1e-9``; 1.6 is Graves &
    Pitarka's. ``correlation`` is with slip; ``slip_exponent`` 0.5 is rise time as the
    square root of slip. The marginal is a gamma so that the mode is away from zero: a
    subfault slipping in no time is an unbounded slip rate. ``floor_s`` is the shortest
    representable pulse, the writer's sample interval.

    Where rise time should be longer or shorter is not a parameter: it is the chart's
    :attr:`Field.RISE_TIME_FACTOR` field, applied to the drawn pattern and renormalised,
    so
    the factor shapes the field and the moment still sets its mean. Likewise where it
    should follow slip more tightly than ``correlation`` says is the chart's
    :attr:`Field.RISE_TIME_SLIP_WEIGHT` field.
    """

    coefficient: float = 1.6
    correlation: float = 0.9
    marginal: Marginal = RISE_MARGINAL
    slip_exponent: float = 0.5
    floor_s: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a rise time the power law cannot produce."""
        if self.slip_exponent <= 0.1:
            raise ParameterError(
                f"a slip exponent of {self.slip_exponent} abandons the correlated field "
                "for independent noise, which is a different model; use one above 0.1"
            )
        if not self.marginal.is_positive:
            raise ParameterError(
                f"a {self.marginal.family} rise-time marginal takes negative values, "
                f"and they have no {self.slip_exponent} power"
            )
        if not -1.0 <= self.correlation <= 1.0:
            raise ParameterError(
                f"a correlation lies in [-1, 1], got {self.correlation}"
            )


@dataclasses.dataclass(frozen=True)
class RakeParameters:
    """Which way each subfault slips: ``mean + sigma * Z``, both in degrees.

    Independent of slip by design: a patch that slips more has no reason to slip in a
    different direction. The marginal is normal, so this is the one field that carries
    slip's correlation lengths exactly, with no pre-correction.
    """

    mean_deg: float
    sigma_deg: float = 15.0

    def __post_init__(self) -> None:
        """Refuse a negative spread."""
        if self.sigma_deg < 0.0:
            raise ParameterError(f"a spread has no sign, got {self.sigma_deg} degrees")


@dataclasses.dataclass(frozen=True)
class RuptureTimeParameters:
    """When each subfault starts: a coherent front, then a displacement blended in.

    ``seeds`` are ``(i, j, t0_s)`` triples the front leaves at known times: one at zero
    for a hypocentre, several for a segment triggered along an edge. The front travels
    at ``velocity_fraction / alpha_T`` of the chart's shear speed field, held to the
    sub-Rayleigh or supershear branch; any depth profile of rupture speed is the
    caller's, baked into that field.

    The displacement's spread follows the moment, ``offset_s + coefficient *
    M0^(1/3)`` in the published units: genslip's ``tsfac_bzero`` and ``tsfac_slope``,
    read as magnitudes. Both zero is a coherent front. ``correlation`` is with slip, so
    high-slip patches rupture early; ``blend_sigma`` is the width of the zone over which
    the displacement grows in from the seed, in units of the spread.
    """

    seeds: tuple[tuple[int, int, float], ...]
    velocity_fraction: float = 0.8
    offset_s: float = 0.1
    coefficient: float = 0.5
    correlation: float = 0.8
    blend_sigma: float = 4.0

    def __post_init__(self) -> None:
        """Refuse a front with nowhere to start or a band it cannot travel in."""
        if not self.seeds:
            raise ParameterError("the front needs at least one seed")
        if not 0.0 < self.velocity_fraction <= MAXIMUM_VELOCITY_FRACTION:
            raise ParameterError(
                f"a velocity fraction lies in (0, {MAXIMUM_VELOCITY_FRACTION:.4f}], "
                f"got {self.velocity_fraction}"
            )
        if self.offset_s < 0.0 or self.coefficient < 0.0:
            raise ParameterError(
                "the onset spread's offset and coefficient are magnitudes"
            )
        if self.blend_sigma <= 0.0:
            raise ParameterError(
                f"the onset blend spans {self.blend_sigma} sigma, which is no width"
            )
        if not -1.0 <= self.correlation <= 1.0:
            raise ParameterError(
                f"a correlation lies in [-1, 1], got {self.correlation}"
            )


# ---------------------------------------------------------------- shared readings


def alpha_t(average_dip_deg: float, average_rake_deg: float) -> float:
    """Graves & Pitarka's geometric correction, in ``[1/1.1, 1]``.

    Exactly 1 for a vertical strike-slip fault. Shortens the rise time and raises the
    rupture speed by the same factor. The rake is wrapped into ``[-180, 180]`` after
    averaging; normal faulting gets no correction, since it is for reverse geometries.
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


def _slip_weight(segment: Geometry) -> CellArray | float:
    """The rise-time slip weight, checked to lie in ``[0, 1]``; 1 when absent."""
    if Field.RISE_TIME_SLIP_WEIGHT not in segment:
        return 1.0
    weight = segment[Field.RISE_TIME_SLIP_WEIGHT]
    on_fault = weight[segment.occupied]
    if not np.all((on_fault >= 0.0) & (on_fault <= 1.0)):
        raise ParameterError(
            f"'{Field.RISE_TIME_SLIP_WEIGHT}' runs {float(on_fault.min()):.3g} to "
            f"{float(on_fault.max()):.3g} on the fault; a weight lies in [0, 1]"
        )
    return weight


# ------------------------------------------------------------------------ slip


def _reach(mask: CellMask, axis: int, *, reverse: bool) -> np.ndarray:
    """How many occupied cells run up to each cell along one direction, inclusive.

    One for a cell whose neighbour on that side is off the fault or off the grid, and
    counting up from there: what makes a taper follow a ragged outline.
    """
    ordered = np.flip(mask, axis=axis) if reverse else mask
    ordered = np.moveaxis(ordered, axis, 0)
    counted = np.zeros(ordered.shape, dtype=np.int64)
    running = np.zeros(ordered.shape[1], dtype=np.int64)
    for line in range(ordered.shape[0]):
        running = np.where(ordered[line], running + 1, 0)
        counted[line] = running
    counted = np.moveaxis(counted, 0, axis)
    return np.flip(counted, axis=axis) if reverse else counted


def taper_edges(
    field: CellArray, occupied: CellMask, params: SlipParameters
) -> CellArray:
    """Ramp a field to zero at the fault's edges; unoccupied cells come back zero.

    Separable: the product of four ramps, one per edge, in whole cells. The edge is the
    fault's, not the chart's, so an interface tapers into its own trench rather than
    into the corner of its bounding rectangle.
    """
    cells_i, cells_j = field.shape

    def width(fraction: float, extent: int) -> int:
        return max(0, int(fraction * extent + 0.5))

    ramp = np.ones(field.shape, dtype=np.float64)
    for cells, axis, reverse in (
        (width(params.top_taper, cells_i), 0, False),
        (width(params.bottom_taper, cells_i), 0, True),
        (width(params.side_taper, cells_j), 1, False),
        (width(params.side_taper, cells_j), 1, True),
    ):
        if cells > 0:
            ramp *= np.minimum(_reach(occupied, axis, reverse=reverse) / cells, 1.0)
    return field * ramp * occupied


def scale_to_moment(
    pattern: CellArray, rigidity_pa: CellArray, areas_km2: CellArray, moment_nm: float
) -> CellArray:
    """Slip in metres: the pattern scaled so the segment carries ``moment_nm``.

    Raises
    ------
    ParameterError
         If the pattern moment is zero.
    """
    total = float(np.sum(rigidity_pa * areas_km2 * M2_PER_KM2 * pattern))
    if not total > 0.0:
        raise ParameterError(
            "the slip pattern carries no moment anywhere -- every subfault was tapered "
            "or masked to zero"
        )
    return (moment_nm / total) * pattern


# --------------------------------------------------------------------- the front


def speed_field(
    shear_speed_km_s: CellArray,
    params: RuptureTimeParameters,
    geometric_correction: float,
) -> CellArray:
    """Rupture speed at every subfault, km/s: one fraction of the shear speed field.

    The correction goes on first and the band last, so the realised fraction never sits
    in the mode-II forbidden zone: a fraction the correction pushes past the Rayleigh
    ceiling is shifted onto the supershear branch rather than clipped into the gap.

    Raises
    ------
    ParameterError
        If any speed is not positive, which is a subfault the front can never reach.
    """
    fraction = params.velocity_fraction / geometric_correction
    if fraction > RAYLEIGH_VELOCITY_FRACTION:
        fraction += 1.0 - RAYLEIGH_VELOCITY_FRACTION
    fraction = min(max(fraction, MINIMUM_VELOCITY_FRACTION), MAXIMUM_VELOCITY_FRACTION)
    speed = fraction * shear_speed_km_s
    if not np.all(speed > 0.0):
        worst = np.unravel_index(int(np.argmin(speed)), speed.shape)
        raise ParameterError(
            f"the rupture speed at subfault {tuple(int(k) for k in worst)} is "
            f"{float(speed[worst]):.4g} km/s; check the shear speed field"
        )
    return speed


def travel_times(
    segment: Geometry, params: RuptureTimeParameters, geometric_correction: float
) -> CellArray:
    """First arrivals on ``(i, j)`` in seconds: the coherent front from the seeds.

    ``|grad T| = 1/v`` over the smooth speed field, so each seed's own time is the time
    it was seeded at and no subfault precedes the earliest seed. Unoccupied cells are
    walled off rather than removed, since the sweep wants a rectangle.
    """
    speed = speed_field(segment[Field.SHEAR_SPEED], params, geometric_correction)
    slowness = np.where(
        segment.occupied, 1.0 / speed, OFF_FAULT_SLOWNESS_FACTOR / speed
    )
    strike_km, dip_km = segment.spacing_km()
    return eikonal_solve(
        np.ascontiguousarray(slowness), (dip_km, strike_km), list(params.seeds)
    )


def blend_onset(
    travel_s: CellArray,
    displacement_s: CellArray,
    params: RuptureTimeParameters,
    scale_s: float,
    *,
    seed_cell: tuple[int, int],
    seed_time_s: float,
) -> CellArray:
    """Displace the solved front, blending in from smooth at the seed.

    ``t = T + min(tau / (n sigma), tau / (c max(-delta, 0)), 1) * delta`` with ``tau``
    the time since the seed and ``delta`` the displacement with its seed value removed.
    The first term is the model, the width of the zone over which roughness accumulates;
    the second is arithmetic, per cell, so no subfault ruptures before the front that
    seeded it and no single deep dip holds back the whole fault behind it.
    """
    since_seed_s = travel_s - seed_time_s
    displacement = displacement_s - displacement_s[seed_cell]

    blend_s = params.blend_sigma * scale_s
    weight = (
        np.minimum(1.0, since_seed_s / blend_s)
        if blend_s > 0.0
        else np.ones_like(travel_s)
    )

    # Where the dip is zero or the draw moves the cell later, the ratio is astronomical
    # and the minimum ignores it: such a cell is under no causal bound at all.
    dip = np.maximum(-displacement, 0.0)
    with np.errstate(divide="ignore", over="ignore"):
        causal = since_seed_s / (CAUSAL_MARGIN * np.maximum(dip, 1e-300))
    return travel_s + np.minimum(weight, causal) * displacement


def generate_segment(
    segment: Geometry,
    slip_parameters: SlipParameters,
    rise_parameters: RiseParameters,
    rake_parameters: RakeParameters,
    rupture_time_parameters: RuptureTimeParameters,
    *,
    rng: np.random.Generator,
) -> Geometry:
    """One segment's four fields, drawn and attached.

    The chart must already carry ``shear_speed_km_s`` and ``rigidity_pa``, and may
    carry ``rise_time_factor`` and ``rise_time_slip_weight``. Four draws
    are made from ``rng``, in this order: slip's latent, rise time's independent
    latent, rake, the onset displacement's independent latent. Slip, rise and the
    displacement share one sampler, pre-corrected for slip's marginal, because they are
    mixed in slip's latent space; rake has its own, uncorrected, because its marginal is
    normal.

    Returns the chart with ``slip_m``, ``rise_time_s``, ``rake_deg`` and ``onset_s``
    attached, and ``alpha_t``, ``rise_time_mean_s`` and ``onset_scale_s`` recorded in
    its attrs.

    Raises
    ------
    GeometryError
        If the chart lacks a material field.
    ParameterError
        If a seed is off the chart, or the pattern carries no moment.
    SamplingError
        If the covariance does not embed on this chart.
    """
    rigidity = segment[Field.RIGIDITY]
    cells = segment.cells
    for i, j, _ in rupture_time_parameters.seeds:
        if not (0 <= i < cells[0] and 0 <= j < cells[1]):
            raise ParameterError(f"seed ({i}, {j}) is off a chart of {cells} cells")

    correction = alpha_t(
        float(np.mean(segment.strike_dip_deg()[1][segment.occupied])),
        rake_parameters.mean_deg,
    )
    moment_nm = slip_parameters.moment_nm

    strike_km, dip_km = segment.spacing_km()
    grid = Grid(segment.cells, (dip_km, strike_km))

    covariance = slip_parameters.covariance
    slip_marginal = slip_parameters.marginal

    # -- slip: the latent everything else correlates against
    latent_sampler = Sampler(
        grid,
        Covariance(
            PreCorrected(covariance.correlation, slip_marginal), covariance.lengths_km
        ),
    )
    slip_latent = latent_sampler.draw(rng)
    pattern = taper_edges(
        slip_marginal.apply(slip_latent), segment.occupied, slip_parameters
    )
    slip_m = scale_to_moment(pattern, rigidity, segment.areas_km2(), moment_nm)

    mean_rise_s = (
        _cube_root_scaling(rise_parameters.coefficient, moment_nm) * correction
    )
    independent = latent_sampler.draw(rng)
    rho = float(
        latent_correlation(
            slip_marginal,
            rise_parameters.marginal,
            np.array(rise_parameters.correlation),
        )
    )
    # The blend as two loadings whose norm `mix` divides out, so the latent stays
    # standard normal at every cell, which the marginal transform assumes. genslip's
    # `w, 1 - w` weights sum to one but do not square to one, and the variance dips
    # wherever they meet. Interpolating loadings rather than correlations is also what
    # lets the weight reach exact slip: NORTA cannot invert a correlation of 1 between
    # two different marginals.
    weight = _slip_weight(segment)
    rise_latent = mix(
        (weight * rho + (1.0 - weight), slip_latent),
        (weight * np.sqrt(1.0 - rho * rho), independent),
    )
    rise_pattern = (
        rise_parameters.marginal.apply(rise_latent) ** rise_parameters.slip_exponent
    )
    # The prescribed factor shapes the field; the mean is the moment's, so renormalise
    # after applying it rather than before.
    if Field.RISE_TIME_FACTOR in segment:
        rise_pattern = rise_pattern * segment[Field.RISE_TIME_FACTOR]
    rise_pattern = rise_pattern / float(rise_pattern[segment.occupied].mean())
    rise_time_s = np.maximum(rise_pattern * mean_rise_s, rise_parameters.floor_s)

    # -- rake: independent of everything
    rake_sampler = Sampler(grid, covariance)
    rake_deg = rake_parameters.mean_deg + rake_parameters.sigma_deg * standardise(
        rake_sampler.draw(rng)
    )

    # -- onset: the coherent front, then the displacement blended in from the seed
    travel_s = travel_times(segment, rupture_time_parameters, correction)
    scale_s = rupture_time_parameters.offset_s + _cube_root_scaling(
        rupture_time_parameters.coefficient, moment_nm
    )
    independent = latent_sampler.draw(rng)
    rho = float(
        latent_correlation(
            slip_marginal, NORMAL, np.array(rupture_time_parameters.correlation)
        )
    )
    displacement_s = scale_s * standardise(
        mix((rho, slip_latent), (np.sqrt(1.0 - rho * rho), independent))
    )
    i, j, seed_time_s = min(rupture_time_parameters.seeds, key=lambda seed: seed[2])
    onset_s = blend_onset(
        travel_s,
        displacement_s,
        rupture_time_parameters,
        scale_s,
        seed_cell=(i, j),
        seed_time_s=seed_time_s,
    )

    return segment.with_fields(
        **{
            Field.SLIP: slip_m,
            Field.RISE_TIME: rise_time_s,
            Field.RAKE: rake_deg,
            Field.ONSET: onset_s,
        }
    ).with_attrs(
        **{
            Attr.ALPHA_T: correction,
            Attr.RISE_TIME_MEAN: mean_rise_s,
            Attr.ONSET_SCALE: scale_s,
        }
    )


# --------------------------------------------------------------- the whole system


@dataclasses.dataclass(frozen=True)
class SegmentParameters:
    """One segment's four parameter sets, as :func:`generate` wants them.

    Everything here is the caller's, including the moment on ``slip`` -- :func:`generate`
    splits nothing and folds nothing. The one field it overwrites is ``timing.seeds``,
    which follows from the rupture tree rather than from any per-segment choice.
    """

    slip: SlipParameters
    rise: RiseParameters
    rake: RakeParameters
    timing: RuptureTimeParameters


def segment_rng(seed: int, segment: str) -> np.random.Generator:
    """The generator one segment draws from, keyed by its **name**.

    Keying by name rather than by position is what makes a fault system's segments
    independent of each other's presence: adding a segment, dropping one or reordering
    the file leaves every other segment's fields bit-identical, which is what makes two
    realisations of a system comparable at all.

    The name is hashed with BLAKE2b rather than :func:`hash`, whose string hashing is
    salted per interpreter and so reproduces nothing between runs.
    """
    key = int.from_bytes(
        hashlib.blake2b(segment.encode(), digest_size=8).digest(), "big"
    )
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(key,)))


def _seeds_of(
    realisation: Realisation, name: str, parent: str | None
) -> tuple[tuple[int, int, float], ...]:
    """Where and when the front starts on one segment.

    A root starts at its own hypocentre at time zero; a triggered segment starts at the
    cell its jump landed on, at the time the jump arrived.
    """
    if parent is not None:
        jump = realisation.jumps[name]
        row, column = jump.child_cell
        return ((int(row), int(column), float(jump.arrival_s)),)

    chart = realisation[name]
    missing = [
        str(attribute)
        for attribute in (Attr.HYPOCENTRE_STRIKE, Attr.HYPOCENTRE_DIP)
        if attribute not in chart.attrs
    ]
    if missing:
        raise ParameterError(
            f"{name!r} is where the rupture starts, and its chart records no {missing}; "
            "a root segment carries its hypocentre as two arc lengths in its attrs"
        )
    row, column = chart.cell_at(
        chart.attrs[Attr.HYPOCENTRE_STRIKE], chart.attrs[Attr.HYPOCENTRE_DIP]
    )
    return ((row, column, 0.0),)


def generate(
    realisation: Realisation,
    parameters: Mapping[str, SegmentParameters],
    *,
    seed: int,
) -> Realisation:
    """Draw every segment of a fault system, parents before children.

    The structure is decided before this is called: ``realisation.tree`` says which
    segment triggered which, ``realisation.jumps`` says where and when each crossing
    landed, and each segment's own ``parameters`` carry its moment. What this adds is
    the seed times -- the hypocentre for the root, the jump's landing cell and arrival
    time for everything else -- and one generator per segment.

    Nothing flows between segments during the draw, since the jumps are already fixed,
    so the causal order is the order the result is *reported* in rather than a
    dependency. It is walked anyway: when a jump rule that reads a parent's solved
    onsets arrives, this loop is where it goes.

    Returns the realisation with every chart drawn on; the tree and the jumps come
    through untouched.

    Raises
    ------
    ParameterError
        If a segment has no parameters, if the system has no tree to walk, or if the
        root records no hypocentre.
    GeometryError
        If a chart lacks a material field.
    """
    unparameterised = sorted(set(realisation) - set(parameters))
    if unparameterised:
        raise ParameterError(
            f"{unparameterised} have no parameters; every segment needs its own, "
            "including its moment"
        )

    charts: dict[str, Geometry] = {}
    for name, parent, geometry in realisation.in_causal_order():
        segment = parameters[name]
        charts[name] = generate_segment(
            geometry,
            segment.slip,
            segment.rise,
            segment.rake,
            dataclasses.replace(
                segment.timing, seeds=_seeds_of(realisation, name, parent)
            ),
            rng=segment_rng(seed, name),
        )
    return realisation.replace(**charts)


__all__ = [
    "CAUSAL_MARGIN",
    "Attr",
    "Field",
    "ParameterError",
    "RakeParameters",
    "RiseParameters",
    "RuptureTimeParameters",
    "SegmentParameters",
    "SlipParameters",
    "alpha_t",
    "blend_onset",
    "generate",
    "generate_segment",
    "scale_to_moment",
    "segment_rng",
    "speed_field",
    "taper_edges",
    "travel_times",
]
