"""Drawing a rupture: slip, rise time, rake and onset, on one segment or many.

:func:`generate_segment` is the whole per-segment pipeline: a chart, its materials, its
:class:`~rupture_generator.rupture.source.SegmentSource` and the shared
:class:`RuptureSettings` in, a :class:`SegmentRupture` out. Everything stochastic
happens here, in a fixed draw order, from the one generator passed in, so a segment's
fields are a pure function of ``(chart, materials, source, settings, rng state)``.

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
from collections.abc import Callable, Mapping

import numpy as np

from rupture_generator._kernels import eikonal_solve
from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.geometry import CellArray, CellMask, Geometry
from rupture_generator.rupture.materials import Materials
from rupture_generator.rupture.realisation import Realisation
from rupture_generator.rupture.source import SegmentSource
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

CAUSAL_MARGIN = 1.05
"""How much room the causal clamp leaves past the weight that would tie with the seed.
A tie-break, not a modelling choice: it keeps the seed the strict earliest cell."""


@dataclasses.dataclass(frozen=True)
class Seed:
    """The cell the rupture front leaves from, ``(i, j)`` with ``i`` down dip, and when.

    Zero at a hypocentre, and the crossing time on a segment something else triggered.
    """

    cell: tuple[int, int]
    time_s: float


def _correlation(value: float) -> None:
    if not -1.0 <= value <= 1.0:
        raise RuptureGeneratorError(f"a correlation lies in [-1, 1], got {value}")


@dataclasses.dataclass(frozen=True)
class SlipSettings:
    """What shapes every segment's slip field.

    ``coefficient_of_variation`` is the spread of the truncated exponential Thingbaijam
    & Mai (2016) fitted to SRCMOD slip, and it also fixes the largest slip on the fault:
    0.90 puts it at 4.3 mean slips. The tapers are fractions of the fault's extent from
    each edge over which slip ramps to zero; with ``top_taper`` zero, slip reaches the
    surface at full amplitude.
    """

    coefficient_of_variation: float = 0.90
    side_taper: float = 0.02
    top_taper: float = 0.0
    bottom_taper: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a taper that is not one, or a spread the family cannot take."""
        for name in ("side_taper", "top_taper", "bottom_taper"):
            if not 0.0 <= getattr(self, name) <= 0.5:
                raise RuptureGeneratorError(
                    f"{name} is a fraction of the fault's extent from one edge, so it "
                    f"lies in [0, 0.5]; got {getattr(self, name)}"
                )
        _ = self.marginal

    @property
    def marginal(self) -> Marginal:
        """The unit-mean truncated exponential slip's values follow."""
        return Marginal("truncated_exponential", self.coefficient_of_variation)


@dataclasses.dataclass(frozen=True)
class RiseSettings:
    """How long each subfault slips for.

    The fault-wide mean is ``coefficient * M0^(1/3) * alpha_T`` with the coefficient in
    the published units, per cube-root dyne-centimetre at ``1e-9``; 1.6 is Graves &
    Pitarka's. ``correlation`` is with slip; ``slip_exponent`` 0.5 is rise time as the
    square root of slip. The values follow a unit-mean gamma at
    ``coefficient_of_variation``: below unit spread its mode is away from zero, so no
    subfault slips in no time at all. Where rise time is longer or shorter, and where it
    follows slip more tightly, are the chart's :class:`Materials`.
    """

    coefficient: float = 1.6
    correlation: float = 0.9
    slip_exponent: float = 0.5
    coefficient_of_variation: float = 0.75

    def __post_init__(self) -> None:
        """Refuse a rise time the power law cannot produce."""
        if self.slip_exponent <= 0.1:
            raise RuptureGeneratorError(
                f"a slip exponent of {self.slip_exponent} abandons the correlated field "
                "for independent noise, which is a different model; use one above 0.1"
            )
        _correlation(self.correlation)
        _ = self.marginal

    @property
    def marginal(self) -> Marginal:
        """The unit-mean gamma rise time's values follow."""
        return Marginal("gamma", self.coefficient_of_variation)


@dataclasses.dataclass(frozen=True)
class RakeSettings:
    """How far each subfault's rake strays from its segment's: ``sigma * Z`` degrees.

    Independent of slip by design: a patch that slips more has no reason to slip in a
    different direction. The marginal is normal, so this is the one field that carries
    slip's correlation lengths exactly, with no pre-correction.
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
    """When each subfault starts: a coherent front, then a displacement blended in.

    The front travels at ``velocity_fraction / alpha_T`` of the chart's shear speed,
    held to the sub-Rayleigh or supershear branch, times the chart's rupture speed
    factor.

    The displacement's spread follows the moment, ``offset_s + coefficient * M0^(1/3)``
    in the published units: genslip's ``tsfac_bzero`` and ``tsfac_slope``, read as
    magnitudes. Both zero is a coherent front. ``correlation`` is with slip, so
    high-slip patches rupture early. ``blend_sigma`` is the width of the zone over which
    the displacement grows in from the seed, in units of its spread.
    """

    velocity_fraction: float = 0.8
    offset_s: float = 0.1
    coefficient: float = 0.5
    correlation: float = 0.8
    blend_sigma: float = 4.0

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
        if self.blend_sigma <= 0.0:
            raise RuptureGeneratorError(
                f"the onset blend spans {self.blend_sigma} sigma, which is no width"
            )
        _correlation(self.correlation)


@dataclasses.dataclass(frozen=True)
class RuptureSettings:
    """Everything every segment shares: how its fields are drawn, not how big it is."""

    slip: SlipSettings = dataclasses.field(default_factory=SlipSettings)
    rise: RiseSettings = dataclasses.field(default_factory=RiseSettings)
    rake: RakeSettings = dataclasses.field(default_factory=RakeSettings)
    timing: TimingSettings = dataclasses.field(default_factory=TimingSettings)


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


def _reach(mask: CellMask, axis: int, *, reverse: bool) -> np.ndarray:
    """How many occupied cells run up to each cell along one axis, inclusive.

    One for a cell whose neighbour on that side is off the fault or off the grid, and
    counting up from there: what makes a taper follow a ragged outline. Zero off the
    fault.
    """
    ordered = np.flip(mask, axis) if reverse else mask
    index = np.arange(ordered.shape[axis], dtype=np.int32)
    index = index[:, None] if axis == 0 else index
    last_gap = np.maximum.accumulate(np.where(ordered, np.int32(-1), index), axis=axis)
    counted = index - last_gap
    return np.flip(counted, axis) if reverse else counted


def taper_edges(field: CellArray, occupied: CellMask, slip: SlipSettings) -> CellArray:
    """Ramp a field to zero at the fault's edges; unoccupied cells come back zero.

    Separable: the product of four ramps, one per edge, in whole cells. The edges are
    the fault's, not the chart's, so an interface tapers into its own outline rather
    than into the corner of its bounding rectangle.
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
    timing: TimingSettings,
    geometric_correction: float,
) -> CellArray:
    """Rupture speed at every subfault, km/s: one fraction of the shear speed field.

    Raises
    ------
    RuptureGeneratorError
        If any speed is not positive, which is a subfault the front can never reach.
    """
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
    travel_s: CellArray,
    displacement_s: CellArray,
    scale_s: float,
    seed: Seed,
    blend_sigma: float,
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
    weight = np.minimum(1.0, since_seed_s / (blend_sigma * scale_s))

    # Where the dip is zero or the draw moves the cell later, the ratio is astronomical
    # and the minimum ignores it: such a cell is under no causal bound at all.
    dip = np.maximum(-displacement, 0.0)
    with np.errstate(divide="ignore", over="ignore"):
        causal = since_seed_s / (CAUSAL_MARGIN * np.maximum(dip, 1e-300))
    return travel_s + np.minimum(weight, causal) * displacement


def generate_segment(
    geometry: Geometry,
    materials: Materials,
    source: SegmentSource,
    settings: RuptureSettings,
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
    slip, rise, timing = settings.slip, settings.rise, settings.timing
    correction = alpha_t(
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

    slip_latent, rise_independent = latent.draw(rng)
    pattern = taper_edges(slip.marginal.apply(slip_latent), geometry.occupied, slip)
    slip_m = scale_to_moment(
        pattern, materials.rigidity_pa, geometry.areas_km2, moment_nm
    )

    mean_rise_s = _cube_root_scaling(rise.coefficient, moment_nm) * correction
    rho = float(
        latent_correlation(slip.marginal, rise.marginal, np.array(rise.correlation))
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
        rise.marginal.apply(rise_latent) ** rise.slip_exponent
        * materials.rise_time_factor
    )
    rise_time_s = rise_pattern * (
        mean_rise_s / float(rise_pattern[geometry.occupied].mean())
    )

    rake_latent, _ = sampler(grid, covariance).draw(rng)
    rake_deg = source.rake_deg + settings.rake.sigma_deg * standardise(rake_latent)

    speed_km_s = (
        speed_field(materials.shear_speed_km_s, timing, correction)
        * materials.rupture_speed_factor
    )
    travel_s = travel_times(geometry, speed_km_s, seed)
    scale_s = timing.offset_s + _cube_root_scaling(timing.coefficient, moment_nm)
    onset_independent, _ = latent.draw(rng)
    rho = float(latent_correlation(slip.marginal, NORMAL, np.array(timing.correlation)))
    displacement_s = scale_s * standardise(
        mix((rho, slip_latent), (np.sqrt(1.0 - rho * rho), onset_independent))
    )

    return SegmentRupture(
        geometry=geometry,
        slip_m=slip_m,
        rise_time_s=rise_time_s,
        rake_deg=rake_deg,
        onset_s=blend_onset(
            travel_s, displacement_s, scale_s, seed, timing.blend_sigma
        ),
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


type JumpRule = Callable[[SegmentRupture, Geometry], Seed]
"""Where and when the front crosses from a drawn parent onto a child."""


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
    there to break. This is the rule under redesign, so it is one function, and
    :func:`generate` takes any other :data:`JumpRule` in its place.

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
    sources: Mapping[str, SegmentSource],
    settings: RuptureSettings,
    *,
    seed: int,
    jump: JumpRule = jump_seed,
) -> dict[str, SegmentRupture]:
    """Draw every segment of a fault system, parents before children.

    What this adds to the realisation is where and when each front starts: the
    hypocentre on the root, and ``jump`` for everything else, read off the parent that
    has just been drawn. That makes the causal order a real dependency, which is why it
    is walked rather than iterated.

    Returns the drawn segments in the order they were drawn.

    Raises
    ------
    RuptureGeneratorError
        If a segment has no materials or no source, or the realisation has no
        hypocentre or no tree to walk.
    """
    for what, given in (("materials", materials), ("source", sources)):
        missing = sorted(set(realisation) - set(given))
        if missing:
            raise RuptureGeneratorError(f"{missing} have no {what}")

    drawn: dict[str, SegmentRupture] = {}
    for name, parent, geometry in realisation.in_causal_order():
        if parent is None:
            hypocentre = realisation.hypocentre
            start = Seed(geometry.cell_at(hypocentre.strike_km, hypocentre.dip_km), 0.0)
        else:
            start = jump(drawn[parent], geometry)
        drawn[name] = generate_segment(
            geometry,
            materials[name],
            sources[name],
            settings,
            seed=start,
            rng=segment_rng(seed, name),
        )
    return drawn


__all__ = [
    "CAUSAL_MARGIN",
    "JumpRule",
    "RakeSettings",
    "RiseSettings",
    "RuptureSettings",
    "Seed",
    "SegmentRupture",
    "SlipSettings",
    "TimingSettings",
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
