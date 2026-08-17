"""The pipeline on a triangulated segment: the same stages, on faces instead of cells.

`pipeline.py` runs the stage order for a quad lattice; this runs the same order for a
:class:`~rupture_generator.triangular.mesh.TriangleMesh`. Every stage that is
elementwise over subfaults is `stages.py`'s own, called here unchanged. What a
triangulation does differently is three things: the taper (:func:`taper_edges`), the
depths the slowness is sampled at (:func:`travel_times`) and the seed seam in
:func:`solve_onsets`. HYBRID.md and MESH.md carry the architecture.

S9's output does not fit at production resolution -- 2.45 billion samples, 19.6 GB of
float64, on the CFM Hikurangi interface cut at 400 m -- so :func:`generate` takes
``synthesise=False`` and the writers run S9 themselves a block of faces at a time.
:func:`write_rupture_mesh` writes the *native* format; the SRF seam is
:mod:`rupture_generator.triangular.assemble`.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rupture_generator import moment, propagation, pulses, stages, timing
from rupture_generator.config.rupture import (
    ComputedPropagation,
    FieldConfig,
    PerFaultSourceConfig,
    PointSourceConfig,
    RuptureConfig,
    SourceConfig,
    VelocityModelConfig,
)
from rupture_generator.pipeline import (
    jump_model,
    named,
    perturbation_model,
    propagate,
    pulse_model,
    rise_time_model,
    speed_model,
)
from rupture_generator.realisation import Realisation
from rupture_generator.triangular import lattice as lattice_module
from rupture_generator.triangular.mesh import TriangleMesh, build_surface

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from rupture_generator.config.geometry import GeometryConfig
    from rupture_generator.sampling import VonKarmanFilterParameters

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

BOUNDARY_SAMPLES_PER_EDGE = 8
"""How finely the labelled boundary is resampled before distances are taken to it.

:func:`taper_edges` takes each face's distance to the nearest point of the resampled
polyline, which overstates it by at most half the sample spacing. At a spacing of
``max(ramp width, boundary edge length) / 8`` that is eight times finer than the lattice
taper's own half-cell precision, and costs ``8 B`` points for ``B`` boundary edges.
"""


STREAM_BUDGET_BYTES = 1 << 30
"""How much memory one block of slip-rate pulses may take while it is written out.

A budget rather than a block size: :func:`face_blocks` derives the block from it and
from the rupture's own rise times, so a rupture with longer pulses gets fewer faces per
block rather than more memory. A gibibyte is comfortably under the 2 GB that is safe to
hold beside the mesh and the fields, and far enough above the HDF5 chunk size that
blocking bounds the peak rather than making the writes small.
"""

PULSE_SAMPLE_MARGIN = 3
"""Samples a pulse may carry beyond the duration its rise time covers.

A bound on each pulse's length, not the kernel's own rounding transcribed.
`crates/kernels/src/pulse.rs` gives a resolved pulse one more sample than the duration
covers and floors an unresolved one at ``SPIKE_SAMPLES = 3``, so ``rise_time / dt + 3``
is above both for every rise time and a block comes out at worst smaller than the budget
allows. Only ``SPIKE_SAMPLES`` growing could invalidate it.
"""


# ============================================================================
# The hoisted geometry
# ============================================================================


@dataclasses.dataclass(frozen=True)
class SegmentGeometry:
    """One segment's derived geometry, computed once and passed down.

    Every array here is a function of the mesh's nodes and faces, and the mesh
    recomputes each of them on every call by design: at the 2.36 M faces a 400 m cut on
    the CFM Hikurangi interface gives, deriving them costs 15 s against a 6 s build. So
    :func:`generate` builds one of these per segment and hands it to every stage. They
    are not cached on the mesh, which stores nothing derived.

    Attributes
    ----------
    vertices_km : FloatArray
        ``(V, 3)`` node positions, offsets from the surface origin.
    faces : IntArray
        ``(F, 3)`` vertex indices.
    parameters_km : FloatArray
        ``(V, 2)`` parameter coordinates.
    centres_km : FloatArray
        ``(F, 3)`` face centres.
    areas_km2 : FloatArray
        ``(F,)`` **true** surface areas -- the moment fold's, so there is no area
        approximation anywhere in this pipeline to correct for.
    lattice : lattice_module.ParameterLattice
        The regular grid over this segment's parameter rectangle that both the sampler
        and the wavefront run on. The two stages that use it are in different passes
        over the segments and would otherwise build it twice.
    """

    vertices_km: FloatArray
    faces: IntArray
    parameters_km: FloatArray
    centres_km: FloatArray
    areas_km2: FloatArray
    lattice: lattice_module.ParameterLattice

    @classmethod
    def of(cls, mesh: TriangleMesh) -> SegmentGeometry:
        """Derive everything once from a chart.

        Returns
        -------
        SegmentGeometry
        """
        parameters_km = mesh.parameters_km()
        faces = mesh.faces()
        return cls(
            vertices_km=mesh.vertices_km(),
            faces=faces,
            parameters_km=parameters_km,
            centres_km=mesh.centres(),
            areas_km2=mesh.areas_km2(),
            lattice=lattice_module.ParameterLattice.of(parameters_km, faces),
        )

    @property
    def depth_km(self) -> FloatArray:
        """``(F,)`` face centre depths, positive down -- what every ramp reads."""
        return self.centres_km[:, 2]

    @property
    def face_count(self) -> int:
        """How many subfaults this segment has."""
        return int(self.faces.shape[0])


# ============================================================================
# The segments
# ============================================================================


def segments_of(geometry: GeometryConfig) -> Realisation:
    """Every surface of a geometry file as triangulated segments.

    The counterpart of `pipeline.segments_of`. There is no ``validate_chart`` step:
    :func:`~rupture_generator.triangular.mesh.check_admissible` runs inside the builder
    and is what replaces that refusal.

    Returns
    -------
    Realisation
        Segments named the way the causality tree names them, in the file's CRS.
    """
    segments: dict[str, TriangleMesh] = {}
    for surface in geometry.surfaces:
        segments |= named(surface.name, build_surface(surface, geometry.crs))
    return Realisation(segments, geometry.crs)


def charts_for(geometry: GeometryConfig, surface_name: str | None) -> Realisation:
    """The triangulated segments of one surface.

    ``surface_name`` may be omitted only when the geometry file holds exactly one
    surface.

    Returns
    -------
    Realisation

    Raises
    ------
    ValueError
        If the geometry holds several surfaces and none was named, or names one that is
        not there. Picking the first would run silently on a fault nobody chose.
    """
    names = [surface.name for surface in geometry.surfaces]
    if surface_name is None:
        if len(names) != 1:
            raise ValueError(
                f"the geometry holds {len(names)} surfaces ({', '.join(names)}), so "
                "one has to be named"
            )
        surface_name = names[0]
    elif surface_name not in names:
        raise ValueError(
            f"no surface is called {surface_name!r}; the geometry holds "
            f"{', '.join(names)}"
        )

    surface = next(item for item in geometry.surfaces if item.name == surface_name)
    return Realisation(
        named(surface_name, build_surface(surface, geometry.crs)), geometry.crs
    )


# ============================================================================
# S4 -- the taper, in parameter space
# ============================================================================


def _taper_widths_km(
    params: stages.SlipParams, strike_km: float, dip_km: float
) -> tuple[float, float, float]:
    """The three ramp widths in kilometres, and the refusal when two of them overlap.

    `stages._taper_widths`' argument, one conversion earlier. There the fractions are of
    a *cell count* and round to whole cells; here they are of the segment's own
    parameter extent and give kilometres directly, so a taper means the same fraction of
    the fault however finely it was cut.

    Returns
    -------
    tuple of float
        Lateral, top and bottom widths, kilometres.

    Raises
    ------
    ValueError
        If the two lateral ramps overlap, or the up-dip and down-dip ones do.
    """
    side = max(0.0, params.side_taper * strike_km)
    top = max(0.0, params.top_taper * dip_km)
    bottom = max(0.0, params.bottom_taper * dip_km)
    if 2.0 * side > strike_km:
        raise ValueError(
            f"a side taper of {params.side_taper} reaches {side:.3g} km from each end "
            f"of a fault {strike_km:.3g} km long, so the two ramps overlap. A taper is "
            "a statement about the edges; past a half of the fault it is a statement "
            "about the middle"
        )
    if top + bottom > dip_km:
        raise ValueError(
            f"the up-dip and down-dip tapers reach {top:.3g} and {bottom:.3g} km of a "
            f"fault {dip_km:.3g} km wide, so they overlap"
        )
    return side, top, bottom


def _distance_to_boundary_km(
    points_uv: FloatArray, segments_uv: FloatArray, width_km: float
) -> FloatArray:
    """Distance from each point to a boundary polyline, in the parameter plane.

    The polyline is resampled at :data:`BOUNDARY_SAMPLES_PER_EDGE` points per edge and
    the query is one ``cKDTree`` lookup: ``O(F log B)`` in the faces and the boundary
    edges, against ``O(F B)`` for an exact point-to-segment test, which at 2.36 M faces
    and thousands of boundary edges is 10^10 operations.

    ``points_uv`` is ``(n, 2)``, ``segments_uv`` is ``(B, 2, 2)`` boundary edges as
    endpoint pairs, and ``width_km`` sets the resampling together with the edge lengths.

    Returns
    -------
    FloatArray
        ``(n,)`` kilometres.
    """
    from scipy.spatial import cKDTree

    starts, ends = segments_uv[:, 0], segments_uv[:, 1]
    lengths_km = np.linalg.norm(ends - starts, axis=1)
    spacing_km = max(width_km, float(np.median(lengths_km))) / BOUNDARY_SAMPLES_PER_EDGE

    # One straight run of samples per edge, endpoints included, built by the
    # ragged-arange identity rather than by a list of per-edge arrays.
    steps = np.maximum(1, np.ceil(lengths_km / spacing_km)).astype(np.int64)
    counts = steps + 1
    offsets = np.cumsum(counts) - counts
    within = np.arange(int(counts.sum())) - np.repeat(offsets, counts)
    edge_of = np.repeat(np.arange(len(steps)), counts)
    fraction = within / np.repeat(steps, counts)
    samples = starts[edge_of] + fraction[:, None] * (ends - starts)[edge_of]

    distance_km, _ = cKDTree(samples).query(points_uv, k=1, workers=-1)
    return np.asarray(distance_km, dtype=np.float64)


def taper_edges(
    mesh: TriangleMesh,
    geometry: SegmentGeometry,
    field: FloatArray,
    params: stages.SlipParams,
) -> FloatArray:
    """Ramp a ``(F,)`` field to zero at the fault's edges, on a triangulation.

    The triangular form of `stages.taper_edges`, and the same model. A lattice counts
    whole cells inward from an index end; a triangulation has no index ends, so this
    measures each face's distance in the **parameter plane** to the boundary the label
    :meth:`~rupture_generator.triangular.mesh.TriangleMesh.boundary_labels` gives it,
    with the widths in kilometres.

    Separable, as the lattice form is: the result is the product of independent ramps to
    the lateral, top and bottom boundaries. ``u`` and ``v`` are independent axes and a
    top edge's outward normal is a ``-v`` direction where a lateral edge's is a ``u``
    one, so the lateral factor is a function of ``u`` and the other two of ``v``. On a
    rectangular parameter domain the distances *are* ``u - u_min`` and ``v - v_min``, so
    the two forms agree to the resampling error.

    Raises
    ------
    ValueError
        If two ramps overlap -- see :func:`_taper_widths_km`.
    """
    parameters = geometry.parameters_km
    strike_km = float(np.ptp(parameters[:, 0]))
    dip_km = float(np.ptp(parameters[:, 1]))
    widths_km = _taper_widths_km(params, strike_km, dip_km)
    if not any(width_km > 0.0 for width_km in widths_km):
        # No taper at all is a config, not a corner case, and returning before the
        # boundary walk -- a sort of every half-edge -- keeps it free.
        return field

    centres_uv = parameters[geometry.faces].mean(axis=1)
    # One half-edge walk for the whole taper: the edges are found once and then *handed*
    # to the labeller, where letting it walk again would cost six passes over 6.5
    # million half-edges at a 400 m cut.
    edges = mesh.boundary_edges()
    labels = mesh.boundary_labels(edges)

    weight = np.ones(geometry.face_count)
    for label, width_km in zip(("lateral", "top", "bottom"), widths_km, strict=True):
        if width_km <= 0.0:
            continue
        chosen = edges[labels == label]
        if not len(chosen):
            continue
        distance_km = _distance_to_boundary_km(centres_uv, parameters[chosen], width_km)
        weight *= np.clip(distance_km / width_km, 0.0, 1.0)
    return field * weight


# ============================================================================
# S4-S8 -- the draw
# ============================================================================


def field_sampler(
    lattice: lattice_module.ParameterLattice,
) -> stages.FieldSampler:
    """This segment's draw, as the shared field stages take it.

    An adapter: the stages are written against
    :data:`~rupture_generator.stages.FieldSampler`, whose first argument is the chart
    and which the stage never inspects, and what this segment draws from is its
    parameter lattice instead. The closure exists to bind that lattice, so one embedding
    is built per segment and all four of its fields come out of it.

    Returns
    -------
    stages.FieldSampler
        Called as ``sampler(chart, covariance, rng)``, returning one value per face.
    """

    def draw(
        chart: object,
        covariance: VonKarmanFilterParameters,
        rng: np.random.Generator,
    ) -> FloatArray:
        """One field of this covariance, one value per face."""
        del chart
        return lattice_module.draw_field(lattice, covariance, rng)

    return draw


# ============================================================================
# S7 -- the wavefront
# ============================================================================


def travel_times(
    geometry: SegmentGeometry,
    shear_speed_km_s: FloatArray,
    params: timing.SpeedParams,
    seeds: Sequence[tuple[int, int, float]],
) -> FloatArray:
    """S7 on a triangulation: first-arrival times per face, in seconds.

    `timing.travel_times`' counterpart, and the same solver: a factored fast sweep over
    a regular grid. What differs is which grid and where the depths come from. The sweep
    runs on the segment's **parameter** lattice, and the slowness it reads is sampled at
    the curved mesh's own **true** centre depths --
    :func:`~rupture_generator.triangular.lattice.travel_times` is where the two meet.

    ``shear_speed_km_s`` is ``(F,)`` from the velocity model at each face's own centre
    depth, and ``seeds`` are ``(i, j, t0_s)`` lattice cells the front leaves and when.

    Returns
    -------
    FloatArray
        ``(F,)`` travel times in seconds.

    Raises
    ------
    ValueError
        For a speed the front cannot travel at, or a seed off the lattice.
    """
    return lattice_module.travel_times(
        geometry.lattice,
        geometry.depth_km,
        shear_speed_km_s,
        params,
        list(seeds),
    )


# ============================================================================
# The stages, in order
# ============================================================================


def attach_materials(
    realisation: Realisation,
    geometries: Mapping[str, SegmentGeometry],
    velocity_model: VelocityModelConfig,
) -> Realisation:
    """The rock each subfault is in: shear speed, rigidity and density.

    `pipeline.attach_materials` on faces, reading the depths off the hoisted geometry.
    Every later stage reads these off the chart rather than the model, so the moment
    fold, the wavefront and the SRF all describe the same rock. ``realisation`` is
    annotated in place and returned.
    """
    bottoms = np.asarray(velocity_model.bottom_depth_km)
    speeds = np.asarray(velocity_model.shear_speed_km_s)
    layer_densities = np.asarray(velocity_model.density_g_cm3)

    for name, mesh in list(realisation.items()):
        depth_km = geometries[name].depth_km
        shear_speed, rigidity = moment.sample_velocity_model(
            depth_km, bottoms, speeds, layer_densities
        )
        realisation[name] = mesh.with_fields(
            shear_speed_kms=shear_speed,
            rigidity_pa=rigidity,
            density_g_cm3=layer_densities[moment.layer_of(depth_km, bottoms)],
        )
    return realisation


def draw_fields(
    realisation: Realisation,
    geometries: Mapping[str, SegmentGeometry],
    config: RuptureConfig,
) -> Realisation:
    """The four drawn fields: slip pattern, rise time, rake, onset perturbation.

    `pipeline.draw_fields` with one substitution and no second: the taper is
    :func:`taper_edges` instead of the lattice one, because a triangulation has no index
    ends to count cells in from. One batch, and the only place anything is drawn -- rise
    time and the perturbation both correlate against slip's own Gaussian, so that
    Gaussian and the sampler are locals that never leave this function. ``realisation``
    is annotated in place and returned.

    Raises
    ------
    ValueError
        If a segment's embedding is past
        :data:`~rupture_generator.sampling.MAXIMUM_EMBEDDING_CELLS`.
    """
    source = config.source
    random = config.random

    for name, mesh in list(realisation.items()):
        geometry = geometries[name]
        covariance = source.covariance_of(name)
        sampler = field_sampler(geometry.lattice)
        slip_params = stages.SlipParams(
            covariance=covariance,
            coefficient_of_variation=config.slip.coefficient_of_variation,
            side_taper=config.slip.side_taper,
            top_taper=config.slip.top_taper,
            bottom_taper=config.slip.bottom_taper,
        )
        pattern, gaussian, slip_draw = stages.slip_pattern(
            mesh,
            slip_params,
            random.stream("slip", name),
            sampler=sampler,
            taper=functools.partial(taper_edges, mesh, geometry),
        )

        average_s = stages.average_rise_time_s(
            moment.seismic_moment_nm(source.magnitude_of(name)),
            source.rise_time_coefficient,
            timing.alpha_t(source.dip_of(name, mesh), source.rake_of(name)),
        )
        rise_time_s = stages.rise_time_field(
            mesh,
            gaussian,
            slip_draw,
            rise_time_model(config, average_s),
            random.stream("rise_time", name),
            covariance,
            sample_interval_s=config.timing.sample_interval_s,
            sampler=sampler,
        )

        rake_deg = stages.rake_field(
            mesh,
            stages.RakeParams(
                covariance=covariance,
                base_rake_deg=source.base_rake_deg_of(name, config.field.base_rake_deg),
                sigma_deg=config.slip.rake_sigma_deg,
            ),
            random.stream("rake", name),
            sampler=sampler,
        )

        perturbation = stages.onset_perturbation(
            mesh,
            slip_draw,
            perturbation_model(config),
            random.stream("onset", name),
            covariance,
            sampler=sampler,
        )

        realisation[name] = mesh.with_fields(
            slip_pattern=pattern,
            rise_time_s=rise_time_s,
            rake_deg=rake_deg,
            onset_perturbation=perturbation,
        ).with_attrs(
            truncated_fraction=stages.truncated_fraction(gaussian, slip_params)
        )
    return realisation


def constant_fields(
    realisation: Realisation,
    geometries: Mapping[str, SegmentGeometry],
    source: PointSourceConfig,
    field: FieldConfig,
) -> Realisation:
    """The same four fields for a point source, given rather than drawn.

    `pipeline.constant_fields` on faces. It provides exactly the names
    :func:`draw_fields` does, so no later stage asks which kind of source it has --
    including the perturbation, which is a field of zeros rather than a missing one.
    ``realisation`` is annotated in place and returned.
    """
    for name, mesh in list(realisation.items()):
        faces = geometries[name].face_count
        realisation[name] = mesh.with_fields(
            slip_pattern=np.ones(faces),
            rise_time_s=np.full(faces, source.rise_time_s),
            rake_deg=np.full(faces, field.base_rake_deg),
            onset_perturbation=np.zeros(faces),
        ).with_attrs(truncated_fraction=0.0)
    return realisation


def scale_moment(
    realisation: Realisation,
    geometries: Mapping[str, SegmentGeometry],
    source: SourceConfig,
) -> Realisation:
    """Size the slip pattern into slip, in metres. The one global fold.

    `pipeline.scale_moment` verbatim in its arithmetic, with the areas read off the
    hoisted geometry. Either one factor over every segment, so how the moment divides
    between faults is the fields' own, or a target per fault when the source states the
    division.

    The fold is over faces with **true** surface areas -- ``areas_km2`` is
    :meth:`~rupture_generator.triangular.mesh.TriangleMesh.areas_km2` on the lifted
    triangles and never a parameter-plane area -- so where a flat model delivers 0.9690
    of the moment it was asked for, this delivers 1.0 exactly. A discrepancy here is a
    bug rather than an approximation, and the tests assert it as such. ``realisation``
    is annotated in place and returned.
    """
    names = list(realisation)
    patterns = [realisation[name]["slip_pattern"] for name in names]
    rigidities = [realisation[name]["rigidity_pa"] for name in names]
    areas = [geometries[name].areas_km2 for name in names]

    if isinstance(source, PerFaultSourceConfig):
        scaled = moment.scale_each_to_moment(
            patterns,
            rigidities,
            areas,
            [moment.seismic_moment_nm(source.magnitude_of(name)) for name in names],
        )
    else:
        scaled = moment.scale_to_moment(
            patterns, rigidities, areas, moment.seismic_moment_nm(source.magnitude)
        )

    for name, slip in zip(names, scaled, strict=True):
        realisation[name] = (
            realisation[name].with_fields(slip_m=slip).without("slip_pattern")
        )
    return realisation


def solve_onsets(
    realisation: Realisation,
    geometries: Mapping[str, SegmentGeometry],
    config: RuptureConfig,
) -> Realisation:
    """S7 and S8: the wavefront, in causal order, and where the front crossed.

    `pipeline.solve_onsets` with three seams changed. **Draws nothing**: every random
    choice was made in :func:`draw_fields`, so the one order-dependent traversal is a
    pure function of its inputs.

    The seams. The hypocentre is one flat face index rather than an ``(i, j)`` pair,
    found from **true arc lengths** because "12 km along strike" means along the fault;
    it is turned into the one lattice cell its centre falls in, which is what the sweep
    seeds, and since projection is a gather the hypocentre face reads that cell's seeded
    ``t0`` exactly. `propagation.causal_jump` returns a flat face index for both cells.
    ``realisation`` is annotated in place and returned.

    Raises
    ------
    ValueError
        For a hypocentre off the fault, a rupture speed the front cannot travel at, or
        two segments too far apart for the front to cross between.
    """
    onset_params = perturbation_model(config)
    jump_delay = jump_model(config)
    max_jump_km = (
        config.propagation.max_jump_km
        if isinstance(config.propagation, ComputedPropagation)
        else propagation.MAX_JUMP_KM
    )

    root = realisation.root
    hypocentre = realisation[root].cell_index(
        config.hypocentre.strike_km, config.hypocentre.dip_km
    )

    solved: dict[str, TriangleMesh] = {}
    jumps: dict[str, propagation.Jump] = {}

    for name in realisation.in_causal_order():
        mesh = realisation[name]
        geometry = geometries[name]
        parent = realisation.tree[name]

        if parent is None:
            seeds = [(*geometry.lattice.cell_of(hypocentre), 0.0)]
            pinned: int | None = hypocentre
            delay_s = config.timing.rupture_delay_s
        else:
            # Chosen on the parent's wavefront and timed on its onset: an argmin over
            # perturbed values finds the perturbation's negative tail, not the front.
            jump = propagation.causal_jump(
                solved[parent],
                solved[parent]["wavefront_s"],
                mesh,
                jump_delay,
                parent_onset_s=solved[parent]["onset_s"],
                max_distance_km=max_jump_km,
            )
            jumps[name] = jump
            seeds = [
                (
                    *geometry.lattice.cell_of(int(jump.child_cell)),
                    float(jump.arrival_s),
                )
            ]
            # Triggered from elsewhere: no pin and no clamp, so this segment's onsets
            # stay absolute rather than restarting from zero.
            pinned = None
            delay_s = 0.0

        travel_time_s = travel_times(
            geometry,
            mesh["shear_speed_kms"],
            speed_model(config, name, mesh),
            seeds,
        )
        solved[name] = mesh.with_fields(
            wavefront_s=travel_time_s + delay_s,
            onset_s=stages.apply_perturbation(
                travel_time_s,
                mesh["onset_perturbation"],
                onset_params,
                hypocentre=pinned,
                delay_s=delay_s,
            ),
        )

    # The hypocentre, in the root's own arc lengths and under the names a file uses.
    # Clamped to `arc_profile`'s last knot rather than to any one vertex's arc length:
    # a triangulation's vertices are in no particular order, so there is no "last node".
    root_mesh = solved[root]
    solved[root] = root_mesh.with_attrs(
        hypocentre_strike_km=min(
            config.hypocentre.strike_km, float(root_mesh.arc_profile(0)[1][-1])
        ),
        hypocentre_dip_km=min(
            config.hypocentre.dip_km, float(root_mesh.arc_profile(1)[1][-1])
        ),
    )

    realisation.update(solved)
    realisation.jumps = jumps
    return realisation


def face_blocks(
    rise_time_s: FloatArray,
    params: pulses.PulseParams,
    budget_bytes: int = STREAM_BUDGET_BYTES,
    bytes_per_sample: int = 8,
) -> Iterator[slice]:
    """Cut a segment's faces into runs whose pulses fit a memory budget.

    Consecutive runs, never a permutation: a rupture file's subfaults are one ordered
    block per segment and the samples are every pulse concatenated in that same order,
    so a block is a slice or it is a different rupture.

    The cut is made on the **pulses' own lengths**, not on a face count, because that is
    what the memory is: at ``dt = 0.005 s`` a four-second rise time carries eight
    hundred samples where half a second carries a hundred. :data:`PULSE_SAMPLE_MARGIN`
    is what makes the length a bound rather than a guess. A single face whose pulse
    alone exceeds the budget still gets its own block, there being nothing smaller to
    cut.

    ``rise_time_s`` is ``(F,)`` in seconds and ``bytes_per_sample`` is what one live
    sample costs the caller: the kernel's own ``f64`` output is always 8, and a writer
    that narrows into a second buffer before writing adds that buffer's width, which is
    4 for the SRF's ``float32``.

    Yields
    ------
    slice
        Consecutive, covering ``range(F)`` exactly once, in order.

    Raises
    ------
    ValueError
        For a budget too small to hold a single sample.
    """
    budget_samples = budget_bytes // bytes_per_sample
    if budget_samples < 1:
        raise ValueError(
            f"a budget of {budget_bytes} bytes does not hold one slip-rate sample, "
            f"which costs {bytes_per_sample} bytes while it is written"
        )

    bound = (
        np.asarray(rise_time_s, dtype=np.float64)
        * params.shape.duration_scale
        / params.sample_interval_s
        + PULSE_SAMPLE_MARGIN
    )
    # Walk the cumulative bound and cut whenever the next face would cross the budget;
    # `searchsorted` on the running total does that without a Python loop over faces.
    running = np.cumsum(bound)
    start = 0
    while start < running.size:
        ceiling = (running[start - 1] if start else 0.0) + budget_samples
        stop = int(np.searchsorted(running, ceiling, side="right"))
        stop = max(stop, start + 1)
        yield slice(start, stop)
        start = stop


def synthesise_pulses(
    realisation: Realisation,
    geometries: Mapping[str, SegmentGeometry],
    config: RuptureConfig,
) -> Realisation:
    """S9: a slip-rate pulse per subfault, carrying that subfault's slip.

    `pulses.synthesise` unchanged: it takes a depth and a slip per subfault and knows
    nothing about the shape of the chart. The only difference from
    `pipeline.synthesise_pulses` is that the depths come off the hoisted geometry.
    ``realisation`` is annotated in place and returned.
    """
    params = pulse_model(config)

    for name, mesh in list(realisation.items()):
        offsets, samples = pulses.synthesise(
            geometries[name].depth_km,
            mesh["slip_m"],
            mesh["rise_time_s"],
            params,
        )
        realisation[name] = mesh.with_pulses(offsets, samples).with_attrs(
            sample_interval_s=config.timing.sample_interval_s
        )
    return realisation


def generate(
    config: RuptureConfig,
    geometry: Realisation,
    *,
    synthesise: bool = True,
) -> Realisation:
    """Run the pipeline over a triangulated fault system.

    The same stage order as `pipeline.generate`, written down the same way -- as this
    function's own body. The geometry is a system nothing has been drawn on; the result
    is the same segments with the rupture attached, which is why the two are one type.
    Every segment's derived geometry is computed **here, once**
    (:class:`SegmentGeometry`) and passed to every stage.

    Parameters
    ----------
    config : RuptureConfig
        What the earthquake is.
    geometry : Realisation
        The segments and the frame they are in, from :func:`segments_of` or
        :func:`charts_for`. **Annotated in place**: the segments, the tree and the jumps
        are written onto this object, which is also what is returned.
    synthesise : bool, optional
        Whether to run S9 and attach each segment's slip-rate pulses. ``False`` stops
        after S8 with everything the pulses are a function of -- slip, rise time, depth
        and the sample interval -- already attached, leaving the writers to synthesise
        them a block of faces at a time. A realisation generated this way carries no
        pulses, so :func:`~rupture_generator.triangular.assemble.to_srf_file` and
        :func:`write_rupture_mesh` will not write one without ``params``.

    Returns
    -------
    Realisation

    Raises
    ------
    ValueError
        For a hypocentre off the fault, a propagation that is not a tree, an
        unrepresentable rise time (naming the subfault), a rupture speed the front
        cannot travel at, or a segment whose embedding is past
        :data:`~rupture_generator.sampling.MAXIMUM_EMBEDDING_CELLS`.
    """
    source = config.source
    source.check_segments(list(geometry))

    geometries = {name: SegmentGeometry.of(mesh) for name, mesh in geometry.items()}

    realisation = propagate(
        geometry,
        config.propagation,
        config.hypocentre,
        config.random.stream("propagation"),
    )
    realisation = attach_materials(realisation, geometries, config.velocity_model)

    if isinstance(source, PointSourceConfig):
        realisation = constant_fields(realisation, geometries, source, config.field)
    else:
        realisation = draw_fields(realisation, geometries, config)

    realisation = scale_moment(realisation, geometries, source)
    realisation = solve_onsets(realisation, geometries, config)
    if not synthesise:
        return realisation
    return synthesise_pulses(realisation, geometries, config)


def write_rupture_mesh(
    realisation: Realisation,
    path: Path | str,
    params: pulses.PulseParams | None = None,
    *,
    budget_bytes: int = STREAM_BUDGET_BYTES,
) -> None:
    """Write a generated triangular rupture out as a version 3 mesh file.

    :func:`~rupture_generator.triangular.mesh.write_mesh` writes each segment's whole
    dataset, which after the pipeline includes every attached field and the CSR pulses.
    Each *segment* is written as its own surface holding one segment, so the file's
    group names are the names a config selects.

    **Two routes in, chosen by what the realisation carries.** A rupture whose pulses
    are attached is written whole. A rupture generated with ``synthesise=False`` is
    handed ``params`` instead, and then S9 runs *here*, a block of faces at a time,
    appended to the file and dropped; :data:`STREAM_BUDGET_BYTES` bounds the peak and
    :func:`face_blocks` is where a block comes from. The stored form is identical either
    way -- the same two CSR arrays under the same names -- so a reader cannot tell.

    ``path`` is ``.h5``, or ``.zarr`` for a rupture that carries its pulses: the
    streaming route writes netCDF only, appending to a Zarr store being a different API
    nothing needs yet. The realisation's own CRS is what is stored.

    Raises
    ------
    ValueError
        If ``params`` is omitted and a segment has no pulses, if it is given and a
        segment already has them -- the two would be written twice -- or if the
        streaming route is asked for anything but a netCDF file.
    """
    from rupture_generator.triangular.mesh import write_mesh

    path = Path(path)
    attached = [name for name, mesh in realisation.items() if mesh.pulses is not None]
    if params is None:
        missing = [name for name in realisation if name not in attached]
        if missing:
            raise ValueError(
                f"segments {', '.join(missing)} carry no slip-rate pulses, so this "
                "would write a rupture that does not say how anything slipped. Either "
                "run `generate` without `synthesise=False`, or pass the pulse model so "
                "that S9 can run here a block of faces at a time"
            )
        write_mesh(
            {name: [mesh] for name, mesh in realisation.items()},
            realisation.crs,
            path,
        )
        return

    if attached:
        raise ValueError(
            f"segments {', '.join(attached)} already carry their pulses, and a pulse "
            "model was passed as well, so S9 would run twice and the second run's "
            "samples would be appended to the first's. Pass one or the other"
        )
    if path.suffix not in {".h5", ".nc", ".hdf5", ".netcdf"}:
        raise ValueError(
            f"the streaming route writes netCDF and {path.suffix or path} is not one. "
            "A rupture whose pulses do not fit in memory is what this route is for, "
            "and appending to a Zarr store is a different API nothing needs yet"
        )

    write_mesh(
        {
            name: [mesh.with_attrs(sample_interval_s=params.sample_interval_s)]
            for name, mesh in realisation.items()
        },
        realisation.crs,
        path,
    )
    _append_pulses(realisation, path, params, budget_bytes)


def _append_pulses(
    realisation: Realisation,
    path: Path,
    params: pulses.PulseParams,
    budget_bytes: int,
) -> None:
    """Synthesise each segment's pulses and append them to a written mesh file.

    The two CSR arrays :meth:`~rupture_generator.triangular.mesh.TriangleMesh
    .with_pulses` would have stored, written into the same variables of the same group,
    but grown a block at a time so that only one block of samples is ever resident.

    Through `h5netcdf` rather than `h5py`: the file is netCDF underneath, so a variable
    needs its dimension scales, and a bare HDF5 dataset dropped alongside reads back as
    a phony dimension that `read_mesh` refuses. ``sample`` is unlimited because its
    length is not known until the last block has been synthesised. ``path`` is the file
    :func:`~rupture_generator.triangular.mesh.write_mesh` just wrote.

    Raises
    ------
    ValueError
        For a subfault whose rise time the sample interval cannot represent, naming the
        segment and the block as well as the subfault -- the kernel numbers subfaults
        within the block it was handed, and a block-local index reported as a global one
        would name the wrong triangle.
    """
    import h5netcdf

    with h5netcdf.File(path, "a") as handle:
        for name, mesh in realisation.items():
            group = handle[name]["segment_0"]
            group.dimensions["sample"] = None
            group.dimensions["cell_edge"] = mesh.face_count + 1
            samples = group.create_variable("slip_rate", ("sample",), dtype="f8")

            depth_km = mesh.centres()[:, 2]
            slip_m = mesh["slip_m"]
            rise_time_s = mesh["rise_time_s"]
            lengths = np.zeros(mesh.face_count, dtype=np.int64)
            at = 0
            for block in face_blocks(rise_time_s, params, budget_bytes):
                try:
                    offsets, block_samples = pulses.synthesise(
                        depth_km[block], slip_m[block], rise_time_s[block], params
                    )
                except ValueError as error:
                    raise ValueError(
                        f"synthesising the pulses of segment {name!r} over faces "
                        f"{block.start} to {block.stop}, where the kernel numbers "
                        f"subfaults from {block.start}: {error}"
                    ) from error
                lengths[block] = np.diff(offsets)
                group.resize_dimension("sample", at + block_samples.size)
                samples[at : at + block_samples.size] = block_samples
                at += block_samples.size

            # The offsets are rebuilt across blocks rather than concatenated: each
            # block's own start at zero, and a segment's samples are one run.
            group.create_variable("slip_rate_offset", ("cell_edge",), dtype="i8")[:] = (
                np.concatenate([[0], np.cumsum(lengths)])
            )


__all__ = [
    "BOUNDARY_SAMPLES_PER_EDGE",
    "PULSE_SAMPLE_MARGIN",
    "STREAM_BUDGET_BYTES",
    "Realisation",
    "SegmentGeometry",
    "attach_materials",
    "charts_for",
    "constant_fields",
    "draw_fields",
    "face_blocks",
    "field_sampler",
    "generate",
    "scale_moment",
    "segments_of",
    "solve_onsets",
    "synthesise_pulses",
    "taper_edges",
    "travel_times",
    "write_rupture_mesh",
]
