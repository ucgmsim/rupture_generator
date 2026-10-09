"""Drawing a rupture: slip, rise time, rake and onset, on one segment or many.

:func:`generate_segment` is the whole per-segment pipeline: a chart, its materials and
its parameters in, a :class:`SegmentRupture` out. Everything stochastic happens here,
in a fixed draw order, from the one generator passed in, so a segment's fields are a
pure function of ``(chart, materials, parameters, rng state)``.

The four fields are one batch because three of them share slip's latent Gaussian:
rise time and the onset displacement both correlate against it, and the correlation
is only linear before the marginals are applied. The latent is a local variable and
never leaves this module.

Every quantity that follows the moment -- the mean rise time, the onset spread -- is
read off the segment's own target moment, so a fault system with a magnitude per
segment gets a rise time and a spread per segment.

:func:`generate` walks a whole fault system. The tree and the moment on each segment
are decided before it is called. The crossings are not: each child's seed is found
from its parent's *solved* onsets by :func:`jump_seed`, so visiting parents-first is a
real dependency rather than a reporting order.
"""

import dataclasses
import hashlib
from collections.abc import Mapping

import numpy as np

from rupture_generator._kernels import eikonal_solve
from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.geometry import CellArray, CellMask, Geometry
from rupture_generator.rupture.materials import Materials
from rupture_generator.rupture.realisation import Realisation
from rupture_generator.sampling import (
    NORMAL,
    Covariance,
    Grid,
    Marginal,
    PreCorrected,
    latent_correlation,
    mix,
    sampler,
    standardise,
)

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
MAXIMUM_VELOCITY_FRACTION = np.sqrt(2.0)
"""Burridge-Andrews: the fastest a supershear front travels."""

OFF_FAULT_SLOWNESS_FACTOR = 10.0
"""What an unoccupied cell's slowness is multiplied by, so the front does not cross the
part of the rectangle that is not fault. Arrivals are bit-identical from x10 upward on
both CFM subduction interfaces."""

ONSET_BLEND_SIGMA = 4.0
"""The width of the zone over which the onset displacement grows in from the seed, in
units of the displacement's spread."""

CAUSAL_MARGIN = 1.05
"""How much room the causal clamp leaves past the weight that would tie with the seed.
A tie-break, not a modelling choice: it keeps the seed the strict earliest cell."""

SLIP_MARGINAL = Marginal("truncated_exponential", 0.90)
"""Thingbaijam & Mai (2016): the family fitted to SRCMOD slip, at the spread that puts
the largest slip at 4.3 mean slips."""

RISE_MARGINAL = Marginal("gamma", 0.75)
"""A gamma below unit spread has its mode away from zero, so no subfault slips in no
time at all, which would be an unbounded slip rate."""


@dataclasses.dataclass(frozen=True)
class Seed:
    """The cell the rupture front leaves from, ``(i, j)`` with ``i`` down dip, and when.

    Zero at a hypocentre, and the crossing time on a segment something else triggered.
    """

    cell: tuple[int, int]
    time_s: float


@dataclasses.dataclass(frozen=True)
class SlipParameters:
    """What shapes a slip field, and the moment that sizes it.

    ``covariance`` is the correlation the *pattern* carries, after the marginal; the
    sampler is asked for a pre-corrected one so that it does. ``side_taper`` is the
    fraction of the fault's length over which slip ramps to zero at each end; slip
    reaches the top and bottom edges at full amplitude.
    """

    moment_nm: float
    covariance: Covariance
    side_taper: float = 0.02

    def __post_init__(self) -> None:
        """Refuse a moment or taper that is not one."""
        if not self.moment_nm > 0.0:
            raise RuptureGeneratorError(
                f"a segment's moment must be positive, got {self.moment_nm}"
            )
        if not 0.0 <= self.side_taper <= 0.5:
            raise RuptureGeneratorError(
                "side_taper is a fraction of the fault's length from one end, so it "
                f"lies in [0, 0.5]; got {self.side_taper}"
            )


@dataclasses.dataclass(frozen=True)
class RiseParameters:
    """How long each subfault slips for.

    The fault-wide mean is ``coefficient * M0^(1/3) * alpha_T`` with the coefficient in
    the published units, per cube-root dyne-centimetre at ``1e-9``; 1.6 is Graves &
    Pitarka's. ``correlation`` is with slip; ``slip_exponent`` 0.5 is rise time as the
    square root of slip. Where rise time is longer or shorter, and where it follows slip
    more tightly, are the chart's :class:`Materials`.
    """

    coefficient: float = 1.6
    correlation: float = 0.9
    slip_exponent: float = 0.5

    def __post_init__(self) -> None:
        """Refuse a rise time the power law cannot produce."""
        if self.slip_exponent <= 0.1:
            raise RuptureGeneratorError(
                f"a slip exponent of {self.slip_exponent} abandons the correlated field "
                "for independent noise, which is a different model; use one above 0.1"
            )
        if not -1.0 <= self.correlation <= 1.0:
            raise RuptureGeneratorError(
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
            raise RuptureGeneratorError(
                f"a spread has no sign, got {self.sigma_deg} degrees"
            )


@dataclasses.dataclass(frozen=True)
class RuptureTimeParameters:
    """When each subfault starts: a coherent front, then a displacement blended in.

    The front travels at ``velocity_fraction / alpha_T`` of the chart's shear speed,
    held to the sub-Rayleigh or supershear branch; any depth profile of rupture speed
    is the caller's, baked into that field.

    The displacement's spread follows the moment, ``offset_s + coefficient * M0^(1/3)``
    in the published units: genslip's ``tsfac_bzero`` and ``tsfac_slope``, read as
    magnitudes. Both zero is a coherent front. ``correlation`` is with slip, so
    high-slip patches rupture early.
    """

    velocity_fraction: float = 0.8
    offset_s: float = 0.1
    coefficient: float = 0.5
    correlation: float = 0.8

    def __post_init__(self) -> None:
        """Refuse a band the front cannot travel in, or a spread that is not one."""
        if not 0.0 < self.velocity_fraction <= MAXIMUM_VELOCITY_FRACTION:
            raise RuptureGeneratorError(
                f"a velocity fraction lies in (0, {MAXIMUM_VELOCITY_FRACTION:.4f}], "
                f"got {self.velocity_fraction}"
            )
        if self.offset_s < 0.0 or self.coefficient < 0.0:
            raise RuptureGeneratorError(
                "the onset spread's offset and coefficient are magnitudes"
            )
        if not -1.0 <= self.correlation <= 1.0:
            raise RuptureGeneratorError(
                f"a correlation lies in [-1, 1], got {self.correlation}"
            )


@dataclasses.dataclass(frozen=True)
class SegmentParameters:
    """One segment's four parameter sets, including its moment on ``slip``."""

    slip: SlipParameters
    rise: RiseParameters
    rake: RakeParameters
    timing: RuptureTimeParameters


@dataclasses.dataclass(frozen=True, eq=False)
class SegmentRupture:
    """One segment, drawn: its four fields, and what the draw decided along the way.

    ``speed_km_s`` is the rupture speed the front travelled at, which is what a jump
    off this segment leaves with.
    """

    geometry: Geometry
    slip_m: CellArray
    rise_time_s: CellArray
    rake_deg: CellArray
    onset_s: CellArray
    speed_km_s: CellArray
    alpha_t: float
    rise_time_mean_s: float
    onset_scale_s: float


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


# ------------------------------------------------------------------------ slip


def _reach(mask: CellMask, *, reverse: bool) -> np.ndarray:
    """How many occupied cells run up to each cell along strike, inclusive.

    One for a cell whose neighbour on that side is off the fault or off the grid, and
    counting up from there: what makes a taper follow a ragged outline. Zero off the
    fault.
    """
    ordered = mask[:, ::-1] if reverse else mask
    index = np.arange(ordered.shape[1], dtype=np.int32)
    last_gap = np.maximum.accumulate(np.where(ordered, np.int32(-1), index), axis=1)
    counted = index - last_gap
    return counted[:, ::-1] if reverse else counted


def taper_edges(field: CellArray, occupied: CellMask, side_taper: float) -> CellArray:
    """Ramp a field to zero at both ends of the fault; unoccupied cells come back zero.

    The product of two ramps, in whole cells. The ends are the fault's, not the
    chart's, so an interface tapers into its own outline rather than into the corner
    of its bounding rectangle.
    """
    cells = int(side_taper * field.shape[1] + 0.5)
    if cells == 0:
        return field * occupied
    ramp = np.minimum(_reach(occupied, reverse=False) / cells, 1.0)
    ramp *= np.minimum(_reach(occupied, reverse=True) / cells, 1.0)
    return field * ramp * occupied


def scale_to_moment(
    pattern: CellArray, rigidity_pa: CellArray, areas_km2: CellArray, moment_nm: float
) -> CellArray:
    """Slip in metres: the pattern scaled so the segment carries ``moment_nm``.

    Raises
    ------
    RuptureGeneratorError
         If the pattern carries no moment.
    """
    total = float(np.sum(rigidity_pa * areas_km2 * M2_PER_KM2 * pattern))
    if not total > 0.0:
        raise RuptureGeneratorError(
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

    Raises
    ------
    RuptureGeneratorError
        If any speed is not positive, which is a subfault the front can never reach.
    """
    fraction = params.velocity_fraction / geometric_correction
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


def travel_times(geometry: Geometry, speed_km_s: CellArray, seed: Seed) -> CellArray:
    """First arrivals on ``(i, j)`` in seconds: the coherent front from the seed.

    ``|grad T| = 1/v`` over the smooth speed field, so the seed's own time is the time
    it was seeded at and no subfault precedes it. Unoccupied cells are walled off
    rather than removed, since the sweep wants a rectangle.
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


def blend_onset(
    travel_s: CellArray, displacement_s: CellArray, scale_s: float, seed: Seed
) -> CellArray:
    """Displace the solved front, blending in from smooth at the seed.

    ``t = T + min(tau / (n sigma), tau / (c max(-delta, 0)), 1) * delta`` with ``tau``
    the time since the seed and ``delta`` the displacement with its seed value removed.
    The first term is the model, the width of the zone over which roughness accumulates;
    the second is arithmetic, per cell, so no subfault ruptures before the front that
    seeded it and no single deep dip holds back the whole fault behind it.
    """
    if scale_s == 0.0:
        return travel_s
    since_seed_s = travel_s - seed.time_s
    displacement = displacement_s - displacement_s[seed.cell]
    weight = np.minimum(1.0, since_seed_s / (ONSET_BLEND_SIGMA * scale_s))

    # Where the dip is zero or the draw moves the cell later, the ratio is astronomical
    # and the minimum ignores it: such a cell is under no causal bound at all.
    dip = np.maximum(-displacement, 0.0)
    with np.errstate(divide="ignore", over="ignore"):
        causal = since_seed_s / (CAUSAL_MARGIN * np.maximum(dip, 1e-300))
    return travel_s + np.minimum(weight, causal) * displacement


def generate_segment(
    geometry: Geometry,
    materials: Materials,
    params: SegmentParameters,
    *,
    seed: Seed,
    rng: np.random.Generator,
) -> SegmentRupture:
    """One segment's four fields, drawn.

    Three transforms are drawn from ``rng``, in this order: slip's latent with rise
    time's independent latent, rake, then the onset displacement's independent latent.
    Slip, rise and the displacement share one embedding, pre-corrected for slip's
    marginal, because they are mixed in slip's latent space; rake has its own,
    uncorrected, because its marginal is normal.

    Raises
    ------
    RuptureGeneratorError
        If the pattern carries no moment, the speed field is not positive, or the
        covariance does not embed on this chart.
    """
    correction = alpha_t(
        float(np.mean(geometry.dip_deg[geometry.occupied])), params.rake.mean_deg
    )
    moment_nm = params.slip.moment_nm
    strike_km, dip_km = geometry.spacing_km
    grid = Grid(geometry.cells, (dip_km, strike_km))
    covariance = params.slip.covariance
    latent = sampler(
        grid,
        Covariance(
            PreCorrected(covariance.correlation, SLIP_MARGINAL), covariance.lengths_km
        ),
    )

    slip_latent, rise_independent = latent.draw(rng)
    pattern = taper_edges(
        SLIP_MARGINAL.apply(slip_latent), geometry.occupied, params.slip.side_taper
    )
    slip_m = scale_to_moment(
        pattern, materials.rigidity_pa, geometry.areas_km2, moment_nm
    )

    mean_rise_s = _cube_root_scaling(params.rise.coefficient, moment_nm) * correction
    rho = float(
        latent_correlation(
            SLIP_MARGINAL, RISE_MARGINAL, np.array(params.rise.correlation)
        )
    )
    # The blend as two loadings whose norm `mix` divides out, so the latent stays
    # standard normal at every cell, which the marginal transform assumes. Loadings
    # rather than correlations are also what let the weight reach exact slip: NORTA
    # cannot invert a correlation of 1 between two different marginals.
    weight = materials.rise_time_slip_weight
    rise_latent = mix(
        (weight * rho + (1.0 - weight), slip_latent),
        (weight * np.sqrt(1.0 - rho * rho), rise_independent),
    )
    rise_pattern = (
        RISE_MARGINAL.apply(rise_latent) ** params.rise.slip_exponent
        * materials.rise_time_factor
    )
    rise_time_s = rise_pattern * (
        mean_rise_s / float(rise_pattern[geometry.occupied].mean())
    )

    rake_latent, _ = sampler(grid, covariance).draw(rng)
    rake_deg = params.rake.mean_deg + params.rake.sigma_deg * standardise(rake_latent)

    speed_km_s = speed_field(materials.shear_speed_km_s, params.timing, correction)
    travel_s = travel_times(geometry, speed_km_s, seed)
    scale_s = params.timing.offset_s + _cube_root_scaling(
        params.timing.coefficient, moment_nm
    )
    onset_independent, _ = latent.draw(rng)
    rho = float(
        latent_correlation(SLIP_MARGINAL, NORMAL, np.array(params.timing.correlation))
    )
    displacement_s = scale_s * standardise(
        mix((rho, slip_latent), (np.sqrt(1.0 - rho * rho), onset_independent))
    )

    return SegmentRupture(
        geometry=geometry,
        slip_m=slip_m,
        rise_time_s=rise_time_s,
        rake_deg=rake_deg,
        onset_s=blend_onset(travel_s, displacement_s, scale_s, seed),
        speed_km_s=speed_km_s,
        alpha_t=correction,
        rise_time_mean_s=mean_rise_s,
        onset_scale_s=scale_s,
    )


# --------------------------------------------------------------- the whole system


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


def jump_seed(parent: SegmentRupture, child: Geometry) -> Seed:
    """Where and when the front crosses from a drawn parent onto a child.

    The crossing that arrives first. The geometry is the chart's own --
    :meth:`~rupture_generator.geometry.Geometry.nearest_cells_to` pairs each of the
    parent's edge cells with the closest cell of the child and measures between them --
    and what is left here is the timing argument: minimise ``onset(p) + d(p) / v(p)``,
    the parent's own solved onset plus that distance at the rupture speed the front had
    when it left. No free parameter, and no delay model beyond distance over velocity.

    Taking only the nearest child cell loses nothing, since the speed at a given
    departure is fixed and the objective is then increasing in distance.

    Two known properties, neither fixed here. The departure is an ``argmin`` over a
    field that already carries its onset displacement, so it is an order statistic and
    reads early by roughly one onset spread; restricting candidates to the fault's edge
    rather than the whole chart, and the displacement's own correlation length, are what
    bound it. And the straight line ignores whether the rock between the two faults is
    there to break. This is the rule under redesign, so it is one function.

    Raises
    ------
    RuptureGeneratorError
        If either chart has no fault cells.
    """
    departures, landings, distance_km = parent.geometry.nearest_cells_to(child)
    arrival_s = parent.onset_s[departures] + distance_km / parent.speed_km_s[departures]
    first = int(np.argmin(arrival_s))
    return Seed(
        cell=(int(landings[0][first]), int(landings[1][first])),
        time_s=float(arrival_s[first]),
    )


def generate(
    realisation: Realisation,
    materials: Mapping[str, Materials],
    parameters: Mapping[str, SegmentParameters],
    *,
    seed: int,
) -> dict[str, SegmentRupture]:
    """Draw every segment of a fault system, parents before children.

    What this adds to the realisation is where and when each front starts: the
    hypocentre on the root, and :func:`jump_seed` for everything else, read off the
    parent that has just been drawn. That makes the causal order a real dependency,
    which is why it is walked rather than iterated.

    Returns the drawn segments in the order they were drawn.

    Raises
    ------
    RuptureGeneratorError
        If a segment has no materials or no parameters, or the realisation has no
        hypocentre or no tree to walk.
    """
    for what, given in (("materials", materials), ("parameters", parameters)):
        missing = sorted(set(realisation) - set(given))
        if missing:
            raise RuptureGeneratorError(f"{missing} have no {what}")

    drawn: dict[str, SegmentRupture] = {}
    for name, parent, geometry in realisation.in_causal_order():
        if parent is None:
            hypocentre = realisation.hypocentre
            start = Seed(geometry.cell_at(hypocentre.strike_km, hypocentre.dip_km), 0.0)
        else:
            start = jump_seed(drawn[parent], geometry)
        drawn[name] = generate_segment(
            geometry,
            materials[name],
            parameters[name],
            seed=start,
            rng=segment_rng(seed, name),
        )
    return drawn


__all__ = [
    "CAUSAL_MARGIN",
    "RakeParameters",
    "RiseParameters",
    "RuptureTimeParameters",
    "Seed",
    "SegmentParameters",
    "SegmentRupture",
    "SlipParameters",
    "alpha_t",
    "blend_onset",
    "generate",
    "generate_segment",
    "jump_seed",
    "scale_to_moment",
    "segment_rng",
    "speed_field",
    "taper_edges",
    "travel_times",
]
