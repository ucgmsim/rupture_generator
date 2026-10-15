"""Drawing a rupture: slip, rise time, rake and onset, on one segment or many.

:func:`generate_segment` is the whole per-segment pipeline. It takes a chart, the
:class:`~rupture_generator.rupture.medium.Medium`, the segment's
:class:`~rupture_generator.rupture.source.SegmentSource` and a
:class:`~rupture_generator.rupture.models.RuptureModel`, and returns a
:class:`SegmentRupture`. The models draw the fields.

The moment is the drawn slip's in this medium, and it has to lie within
:data:`MOMENT_MAGNITUDE_TOLERANCE` magnitude units of the source's. Rise time and the
onset spread follow that moment, so a fault system with a magnitude per segment gets a
rise time and a spread per segment.

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
from rupture_generator.geometry import CellArray, Geometry
from rupture_generator.rupture.medium import Medium, rigidity_pa
from rupture_generator.rupture.models import (
    MODELS,
    RuptureModel,
    SegmentContext,
    check_cells,
    positive,
    read_at_centres,
)
from rupture_generator.rupture.propagator import DEFAULT_JUMP_MODEL, JumpModel
from rupture_generator.rupture.realisation import Realisation
from rupture_generator.rupture.source import SegmentSource, magnitude_from_moment

OFF_FAULT_SLOWNESS_FACTOR = 10.0
"""What an unoccupied cell's slowness is multiplied by, so the front does not cross the
part of the rectangle that is not fault."""

CAUSAL_MARGIN = 1.05
"""How much room the causal clamp leaves past the weight that would tie with the seed.
A tie-break, not a modelling choice: it keeps the seed the strict earliest cell."""

MOMENT_MAGNITUDE_TOLERANCE = 0.05
"""How far a segment's slip may set its moment magnitude from its source's: the rounding
of a magnitude stated to one decimal place."""


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
    statistics : Mapping of str to float
        The numbers each model reported about its draw. Each key starts with the
        model's name, as in ``rise.mean_s``.
    dt_s : float
        The slip-rate sample interval, in seconds.
    beta : CellArray or None
        The pulse's rising fraction, or ``None`` for single-sample impulses.
    """

    geometry: Geometry
    slip_m: CellArray
    rise_time_s: CellArray
    rake_deg: CellArray
    onset_s: CellArray
    speed_km_s: CellArray
    shear_speed_km_s: CellArray
    density_g_cm3: CellArray
    dt_s: float
    beta: CellArray | None
    statistics: Mapping[str, float] = dataclasses.field(default_factory=dict)

    @property
    def rigidity_pa(self) -> CellArray:
        """CellArray: Rigidity in pascals, from the rock this segment read."""
        return rigidity_pa(self.shear_speed_km_s, self.density_g_cm3)


# --------------------------------------------------------------------- the front


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
    blend_s: float,
    seed: JumpSeed,
) -> CellArray:
    """Blend in rupture propagation perturbation terms preserving causality.

    ``t = T + min(tau / blend, tau / (c max(-delta, 0)), 1) * delta`` with ``tau``
    the time since the seed and ``delta`` the displacement with its seed value removed.
    The first term is the model, the width of the zone over which roughness accumulates.
    The second is arithmetic, per cell. It stops any subfault from rupturing before the
    front that seeded it, and stops one deep dip from holding back the whole fault
    behind it.
    """
    displacement = displacement_s - displacement_s[seed.cell]
    if not np.any(displacement):
        return travel_s
    since_seed_s = travel_s - seed.time_s
    weight = (
        np.ones_like(since_seed_s)
        if blend_s == 0.0
        else np.minimum(1.0, since_seed_s / blend_s)
    )

    # Where the dip is zero or the draw moves the cell later, the ratio is astronomical
    # and the minimum ignores it: such a cell is under no causal bound at all.
    dip = np.maximum(-displacement, 0.0)
    with np.errstate(divide="ignore", over="ignore"):
        causal = since_seed_s / (CAUSAL_MARGIN * np.maximum(dip, 1e-300))
    return travel_s + np.minimum(weight, causal) * displacement


# ------------------------------------------------------------------ one segment


def _moment(context: SegmentContext, slip_m: CellArray) -> tuple[CellArray, float]:
    """Check a slip against its chart and source, and return it with its moment.

    Raises
    ------
    RuptureGeneratorError
        If the slip doesn't fit the chart, or is negative or not finite on a fault
        cell. Also if its moment is zero, or its magnitude lies more than
        :data:`MOMENT_MAGNITUDE_TOLERANCE` from the source's.
    """
    geometry = context.geometry
    slip_m = check_cells(
        geometry,
        "slip",
        slip_m,
        lambda slip: ~geometry.occupied | (np.isfinite(slip) & (slip >= 0.0)),
        "finite, non-negative values on the fault",
    )
    slip_m = np.where(geometry.occupied, slip_m, 0.0)
    moment_nm = context.moment_nm(slip_m)
    if not moment_nm > 0.0:
        raise RuptureGeneratorError("slip has zero moment")
    drawn = magnitude_from_moment(moment_nm)
    stated = magnitude_from_moment(context.source.moment_nm)
    if abs(drawn - stated) > MOMENT_MAGNITUDE_TOLERANCE:
        raise RuptureGeneratorError(
            f"slip sets a moment magnitude of {drawn:.3f} and the source's is "
            f"{stated:.3f}; they differ by more than {MOMENT_MAGNITUDE_TOLERANCE}"
        )
    return slip_m, moment_nm


def generate_segment(
    geometry: Geometry,
    medium: Medium,
    source: SegmentSource,
    model: RuptureModel,
    *,
    seed: JumpSeed,
    rng: np.random.Generator,
) -> SegmentRupture:
    """Draw one segment's slip, rise time, rake and onset.

    The models come in a fixed order: slip, rise time, rake, speed, then the onset
    displacement. ``rng`` spawns one generator per model in that order, so a model's
    draws depend on ``rng`` alone and never on another model's draws. Replacing one
    model leaves every other field it doesn't read bit-identical.

    The moment that rise time and the displacement follow is the slip's in this
    medium, which has to lie within :data:`MOMENT_MAGNITUDE_TOLERANCE` magnitude
    units of ``source.moment_nm``. Off-fault slip comes out zero.

    Parameters
    ----------
    geometry : Geometry
        The segment's chart.
    medium : Medium
        The rock, read at the chart's cell centres.
    source : SegmentSource
        The segment's moment, rake and slip covariance.
    model : RuptureModel
        The models to draw each field with.
    seed : JumpSeed
        Where and when the front starts on this segment.
    rng : numpy.random.Generator
        The segment's own generator.

    Returns
    -------
    SegmentRupture
        The drawn fields and pulse, with the rock they used.

    Raises
    ------
    RuptureGeneratorError
        If the medium reads out of range on this chart, a model refuses the chart, or
        a model returns a field that isn't shaped like the chart or holds values that
        field can't take. Also if the slip's moment disagrees with the source's, or
        the blend time is negative.
    """
    context = SegmentContext(
        geometry=geometry,
        source=source,
        shear_speed_km_s=read_at_centres(
            geometry,
            "shear speed",
            medium.shear_speed_km_s,
            positive,
            "positive values",
        ),
        density_g_cm3=read_at_centres(
            geometry, "density", medium.density_g_cm3, positive, "positive values"
        ),
    )
    beta = (
        None
        if model.pulse.beta is None
        else read_at_centres(
            geometry,
            "beta",
            model.pulse.beta,
            lambda beta: (beta > 0.0) & (beta <= 0.5),
            "rising fractions in (0, 0.5]",
        )
    )
    streams = dict(zip(MODELS, rng.spawn(len(MODELS)), strict=True))

    slip = model.slip(context, streams["slip"])
    slip_m, moment_nm = _moment(context, slip.slip_m)
    check_cells(geometry, "slip's latent", slip.latent, np.isfinite, "finite values")
    # The other models read the slip the rupture keeps.
    slip = dataclasses.replace(slip, slip_m=slip_m)
    rise = model.rise(context, slip, moment_nm, streams["rise"])
    rise_time_s = check_cells(
        geometry,
        "rise time",
        rise.values,
        lambda rise_s: np.isfinite(rise_s) & (rise_s >= 0.0),
        "finite, non-negative values",
    )
    rake = model.rake(context, streams["rake"])
    rake_deg = check_cells(geometry, "rake", rake.values, np.isfinite, "finite values")
    speed = model.speed(context, slip, streams["speed"])
    speed_km_s = check_cells(
        geometry,
        "the rupture speed",
        speed.values,
        positive,
        "positive values; check the shear speed field",
    )
    perturbation = model.perturbation(context, slip, moment_nm, streams["perturbation"])
    displacement_s = check_cells(
        geometry,
        "the onset displacement",
        perturbation.displacement_s,
        np.isfinite,
        "finite values",
    )
    if not (np.isfinite(perturbation.blend_s) and perturbation.blend_s >= 0.0):
        raise RuptureGeneratorError(
            f"the onset displacement blends in over {perturbation.blend_s} s, which "
            "is not a duration"
        )

    travel_s = _travel_times(geometry, speed_km_s, seed)
    drawn = (slip, rise, rake, speed, perturbation)
    return SegmentRupture(
        geometry=geometry,
        slip_m=slip_m,
        rise_time_s=rise_time_s,
        rake_deg=rake_deg,
        onset_s=_blend_onset(travel_s, displacement_s, perturbation.blend_s, seed),
        speed_km_s=speed_km_s,
        shear_speed_km_s=context.shear_speed_km_s,
        density_g_cm3=context.density_g_cm3,
        dt_s=model.pulse.dt_s,
        beta=beta,
        statistics={
            f"{name}.{key}": float(value)
            for name, model in zip(MODELS, drawn, strict=True)
            for key, value in model.statistics.items()
        },
    )


# --------------------------------------------------------------- the whole system


def check_seed(seed: object) -> int:
    """Refuse a seed that isn't a non-negative integer.

    Parameters
    ----------
    seed : object
        The seed to check.

    Returns
    -------
    int
        The seed.

    Raises
    ------
    RuptureGeneratorError
        If ``seed`` is a boolean, a float or another non-integer, or negative.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise RuptureGeneratorError(f"a seed is a non-negative integer, got {seed!r}")
    return int(seed)


FIELDS_STREAM = 0
"""The spawn key's first element for a segment's fields."""

JUMP_STREAM = 1
"""The spawn key's first element for the jump onto a segment."""


def _segment_rng(seed: int, stream: int, segment: str) -> np.random.Generator:
    """The generator one segment draws ``stream`` from, keyed by the segment's **name**.

    Keying by name and not by position makes a fault system's segments independent of
    each other's presence. Adding or dropping a segment, or reordering the file, leaves
    every other segment's fields bit-identical. Without that, two realisations of a
    system wouldn't be comparable at all. The stream comes first in the spawn key, so
    no segment name can collide with another stream's key.

    The key is a BLAKE2b hash of the name. Python's :func:`hash` salts string hashes
    per interpreter, so it reproduces nothing between runs.
    """
    key = int.from_bytes(
        hashlib.blake2b(segment.encode(), digest_size=8).digest(), "big"
    )
    return np.random.default_rng(
        np.random.SeedSequence(entropy=seed, spawn_key=(stream, key))
    )


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
        How far a jump can land, and whether it pairs interior cells as well as edge
        cells.
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
    landings, departures, gap_km = child.nearest_cells_to(
        parent.geometry, interior=model.interior
    )
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
    model: RuptureModel | Mapping[str, RuptureModel],
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
    model : RuptureModel or Mapping of str to RuptureModel
        How to draw each field: one for every segment, or one per segment by name.
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
        If the seed isn't a non-negative integer, a segment has no source or model,
        the models cover a segment the realisation lacks, a segment's draw fails, or
        the realisation has no hypocentre or no tree to walk. A failed draw's
        message starts with the segment's name.
    """
    seed = check_seed(seed)
    missing = sorted(set(realisation) - set(sources))
    if missing:
        raise RuptureGeneratorError(f"{missing} have no source")
    if not isinstance(model, RuptureModel):
        missing = sorted(set(realisation) - set(model))
        if missing:
            raise RuptureGeneratorError(f"{missing} have no model")
        unknown = sorted(set(model) - set(realisation))
        if unknown:
            raise RuptureGeneratorError(
                f"models name {unknown}, which are not segments of the realisation"
            )

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
                _segment_rng(seed, JUMP_STREAM, name),
            )
        try:
            drawn[name] = generate_segment(
                geometry,
                medium,
                sources[name],
                model if isinstance(model, RuptureModel) else model[name],
                seed=start,
                rng=_segment_rng(seed, FIELDS_STREAM, name),
            )
        except RuptureGeneratorError as error:
            raise RuptureGeneratorError(f"segment {name!r}: {error}") from error
    return realisation.with_segments(drawn)


__all__ = [
    "CAUSAL_MARGIN",
    "MOMENT_MAGNITUDE_TOLERANCE",
    "JumpRule",
    "JumpSeed",
    "SegmentRupture",
    "check_seed",
    "first_arrival",
    "generate",
    "generate_segment",
]
