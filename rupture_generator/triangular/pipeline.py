"""The pipeline on a triangulated segment: the same stages, on faces instead of cells.

`pipeline.py` runs the stage order for a quad lattice; this runs the same order for a
:class:`~rupture_generator.triangular.mesh.TriangleMesh`. It exists as a second module
rather than as a branch in the first because `MESH.md`'s phase 4 is what deletes the
quad path, and until then the two have to be green at the same time.

**What is shared, and why that is most of it.** A stage that is elementwise over
subfaults does not care whether a subfault is a cell of a lattice or a triangle of a
surface: `moment.py` is arithmetic on flat arrays, `pulses.py` takes a depth and a slip
per subfault, and :func:`~rupture_generator.stages.rise_time_field`,
:func:`~rupture_generator.stages.rake_field`,
:func:`~rupture_generator.stages.onset_perturbation` and
:func:`~rupture_generator.stages.apply_perturbation` are the models this package is
*for*. Every one of those is called here, from `stages.py`, unchanged -- so there is
one rise-time model and one perturbation model in the package rather than two
transcriptions free to drift. Two things made that possible and both are small: those
stages take the **draw** as a parameter (:data:`~rupture_generator.stages.FieldSampler`)
because the sampler is the one thing a mesh's shape decides, and
:func:`~rupture_generator.propagation.causal_jump` asks the chart where its fault runs
out rather than deriving it from a lattice shape.

**What is genuinely different is written here**, and it is four things:

- :func:`taper_edges`. The lattice taper counts whole cells from an index end; there is
  no index end here, so the same fractions become distances to the *labelled* boundary
  in the parameter plane. Separability survives -- ``u`` and ``v`` are independent axes
  -- and so does the refusal when two ramps overlap.
- :func:`travel_times`. `timing.travel_times` reads a spacing and sweeps a Cartesian
  stencil; this hands the vertices and faces to
  :func:`~rupture_generator.triangular.fim.solve`, whose slowness is per face and whose
  answer is per vertex, and carries the answer back to faces through
  :func:`~rupture_generator.triangular.fim.face_arrivals`.
- :func:`draw_fields`, which assembles **one** :class:`MaternOperator` per segment and
  draws the four fields from it. The circulant sampler's cost is the embedding and this
  one's is the factorisation, so in both cases the geometry is paid for once.
- The seed seam. A lattice seeds a *cell*; the solver here seeds *vertices*, and a
  face's arrival is the mean of its three corners. Seeding one corner of the hypocentre
  face would leave that face arriving ``~0.4 h S`` late -- 0.2 s at a 1 km cut, in the
  one quantity `MESH.md` says the model's own perturbation gives no cover for -- so
  :func:`face_seeds` seeds all three corners of the subfault the hypocentre is in. That
  is the faithful port of "seed the cell": the structured solver's seed cell is at
  ``t = 0`` across the whole cell too.

**Derived geometry is hoisted, deliberately.** The container stores nothing derived --
`mesh.py`'s docstring argues that a derived quantity written down is a second
description of the geometry -- so ``areas_km2()``, ``centres()`` and the boundary walk
recompute on every call. Measured on the CFM Hikurangi interface refined to 400 m
subfaults (2.36 M faces), deriving the quantities the pipeline needs costs 15 s against
a 6 s build. So :class:`SegmentGeometry` computes them **once per segment** in
:func:`generate` and every stage takes it as an argument. They are deliberately *not*
cached on the mesh: that would reintroduce exactly the coupling the container excludes.

Nothing here writes a file, exactly as in `pipeline.py`; the SRF seam is
:mod:`rupture_generator.triangular.assemble`.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
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
from rupture_generator.triangular import fim, spde
from rupture_generator.triangular.mesh import TriangleMesh, build_surface

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from rupture_generator.config.geometry import GeometryConfig
    from rupture_generator.sampling import VonKarmanFilterParameters

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

BOUNDARY_SAMPLES_PER_EDGE = 8
"""How finely the labelled boundary is resampled before distances are taken to it.

:func:`taper_edges` measures each face's distance to the boundary as the distance to the
nearest point of a resampled boundary polyline, which is one nearest-neighbour query
rather than a point-to-segment test against every boundary edge. This is the resampling.

**Derived from the two lengths in play, not chosen.** A polyline sampled at spacing
``s`` reports a distance too large by at most ``s / 2`` -- the lateral offset to the
nearest sample -- so the spacing is set to ``max(ramp width, boundary edge length) / 8``
and the error is at most a sixteenth of whichever of those two is longer. Both branches
are needed: tying it to the ramp alone would resample a coarse mesh's boundary
thousands of times per edge when the ramp is narrower than a subfault, and tying it to
the edge alone would under-resolve a ramp much finer than the mesh, which is a taper
that spans no cells anyway.

**The yardstick is the lattice taper's own precision.** `stages.taper_edges` gives the
outermost cell the weight ``h / width`` where a true distance ramp gives ``h / (2
width)``, so the incumbent model is itself a half-cell -- ``h / 2`` -- away from a
distance ramp. Eight samples per edge is 8 times finer than that, and the cost is
``8 B`` points for ``B`` boundary edges however wide or narrow the taper is.
"""


class StandInField(UserWarning):
    """The field on this segment is not a von Karman field at all. See
    :func:`white_noise_stand_in`."""


# ============================================================================
# The hoisted geometry
# ============================================================================


@dataclasses.dataclass(frozen=True)
class SegmentGeometry:
    """One segment's derived geometry, computed once and passed down.

    Every array here is a function of the mesh's nodes and faces, and the mesh
    recomputes each of them on every call by design. At the resolutions this track
    exists for -- 2.36 M faces at a 400 m cut on the CFM Hikurangi interface -- that is
    seconds a call, and there are at least three consumers of the areas and four of the
    centres. So :func:`generate` builds one of these per segment and hands it to every
    stage.

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
        ``(F,)`` true surface areas.
    """

    vertices_km: FloatArray
    faces: IntArray
    parameters_km: FloatArray
    centres_km: FloatArray
    areas_km2: FloatArray

    @classmethod
    def of(cls, mesh: TriangleMesh) -> SegmentGeometry:
        """Derive everything once from a chart.

        Parameters
        ----------
        mesh : TriangleMesh
            The segment.

        Returns
        -------
        SegmentGeometry
        """
        return cls(
            vertices_km=mesh.vertices_km(),
            faces=mesh.faces(),
            parameters_km=mesh.parameters_km(),
            centres_km=mesh.centres(),
            areas_km2=mesh.areas_km2(),
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
    :func:`~rupture_generator.triangular.mesh.check_admissible` runs inside the builder,
    and it is what replaces that refusal -- it names the modelling assumption ("this
    surface is a graph over a plane") rather than a per-bend proxy for it.

    Parameters
    ----------
    geometry : GeometryConfig
        What the geometry file said.

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

    Parameters
    ----------
    geometry : GeometryConfig
        What the geometry file said.
    surface_name : str, optional
        Which surface. May be omitted only when the file holds exactly one.

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
    parameter extent and give kilometres directly, which is what MESH.md means by the
    parameter-space form being cleaner -- no rounding, and a taper that means the same
    fraction of the fault however finely it was cut.

    The overlap refusal is kept exactly, and for the reason `stages.taper_edges` gives:
    past half the fault a taper is a statement about the middle, and the separable and
    overwriting forms of it disagree there.

    Parameters
    ----------
    params : stages.SlipParams
        Carrying the three fractions.
    strike_km, dip_km : float
        The segment's parameter extents.

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
    the query is one ``cKDTree`` lookup, which is ``O(F log B)`` in the faces and the
    boundary edges -- against ``O(F B)`` for an exact point-to-segment test, which at
    2.36 M faces and thousands of boundary edges is 10^10 operations. That constant's
    docstring carries the error the resampling costs and the yardstick it is measured
    against.

    Parameters
    ----------
    points_uv : FloatArray
        ``(n, 2)`` query points.
    segments_uv : FloatArray
        ``(B, 2, 2)`` boundary edges as endpoint pairs.
    width_km : float
        The ramp width, which sets the resampling together with the edge lengths.

    Returns
    -------
    FloatArray
        ``(n,)`` kilometres.
    """
    from scipy.spatial import cKDTree

    starts, ends = segments_uv[:, 0], segments_uv[:, 1]
    lengths_km = np.linalg.norm(ends - starts, axis=1)
    spacing_km = max(width_km, float(np.median(lengths_km))) / BOUNDARY_SAMPLES_PER_EDGE

    # One straight run of samples for every edge, endpoints included, built by the
    # ragged-arange identity rather than by a list of per-edge arrays: at 2.36 M faces
    # the boundary is thousands of edges and a Python object each is what the container
    # measurements say not to do.
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
    """Ramp a field to zero at the fault's edges, on a triangulation.

    The triangular form of `stages.taper_edges`, and the same model: a rupture that
    slips right up to its boundary is unphysical, because the edges are where the fault
    stops. What changes is how "near an edge" is measured. A lattice counts whole cells
    inward from an index end; a triangulation has no index ends, so this measures each
    face's distance in the **parameter plane** to the boundary the label
    :meth:`~rupture_generator.triangular.mesh.TriangleMesh.boundary_labels` gives it,
    with the widths in kilometres.

    **Separable, for the reason the lattice form is.** The result is the product of
    independent ramps -- one to the lateral boundary, one to the top, one to the bottom
    -- so a face two ramps reach is damped by both, and overlapping ramps are refused
    outright rather than left to disagree. That the product still means what it meant is
    a statement about the axes: ``u`` and ``v`` are independent, a top edge's outward
    normal is a ``-v`` direction and a lateral edge's is a ``u`` one, so the lateral
    factor is a function of ``u`` and the other two of ``v``, exactly as before. On a
    rectangular parameter domain the distances *are* ``u - u_min`` and ``v - v_min``, so
    the two forms agree to the resampling error.

    **Not the same numbers as the lattice taper, and deliberately.** The fraction there
    rounds to whole cells and the outermost cell gets weight ``1 / side``, which is
    exactly 1 -- no taper at all -- whenever the fraction rounds to one cell. That is
    what the shipped 2% side taper does on a 56 km fault cut at 1 km. Here the same 2%
    is 1.1 km of genuine ramp. So a planar comparison against the quad path is run with
    the tapers off, and the difference the tapers make is reported rather than tuned
    away.

    Parameters
    ----------
    mesh : TriangleMesh
        The segment, for its boundary and its labels.
    geometry : SegmentGeometry
        Its hoisted geometry.
    field : FloatArray
        ``(F,)`` the field to taper.
    params : stages.SlipParams
        Carrying the three fractions.

    Returns
    -------
    FloatArray
        ``(F,)`` the field, damped near the edges.

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
        # No taper at all is a config, not a corner case -- ``top_taper`` is zero in
        # production. Returning before the boundary is walked keeps that free, and the
        # walk is a sort of every half-edge.
        return field

    centres_uv = parameters[geometry.faces].mean(axis=1)
    # Both of these walk every half-edge, so they are read once here rather than once
    # per label: `boundary_faces(label)` would repeat the walk three times over.
    edges = mesh.boundary_edges()
    labels = mesh.boundary_labels()

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
# S4-S8 -- the samplers
# ============================================================================


def matern_sampler(
    mesh: TriangleMesh,
    geometry: SegmentGeometry,
    covariance: VonKarmanFilterParameters,
) -> stages.FieldSampler:
    """The mesh-native sampler, as a draw the shared stages can take.

    One :class:`~rupture_generator.triangular.spde.MaternOperator` is assembled and
    factorised here and every draw from it is a handful of sparse solves -- which is
    the counterpart of the circulant sampler holding its embedding, and the reason this
    returns a closure rather than a function that rebuilds per field. A segment draws
    four fields.

    The returned draw ignores the covariance it is handed, because the operator was
    built from it: the four stages of one segment all use ``source.covariance_of(name)``,
    so there is one covariance per segment by construction, and a sampler that quietly
    rebuilt for a second one would be the expensive thing happening invisibly. It is
    checked rather than assumed.

    Parameters
    ----------
    mesh : TriangleMesh
        The segment.
    geometry : SegmentGeometry
        Its hoisted geometry.
    covariance : VonKarmanFilterParameters
        The patch structure this segment's magnitude implies.

    Returns
    -------
    stages.FieldSampler
        Called as ``sampler(mesh, covariance, rng)``, returning one value per face.
    """
    del mesh
    operator = spde.MaternOperator(
        geometry.vertices_km, geometry.faces, geometry.parameters_km, covariance
    )

    def draw(
        chart: object,
        asked: VonKarmanFilterParameters,
        rng: np.random.Generator,
    ) -> FloatArray:
        """One field of this segment's covariance, one value per face."""
        del chart
        if asked != covariance:
            raise ValueError(
                f"this sampler was built for correlation lengths "
                f"{covariance.correlation_length_strike_km:.4g} x "
                f"{covariance.correlation_length_dip_km:.4g} km and is being asked for "
                f"{asked.correlation_length_strike_km:.4g} x "
                f"{asked.correlation_length_dip_km:.4g} km. One operator serves one "
                "covariance; build a second sampler rather than reusing this one"
            )
        return spde.face_values(operator.draw(rng), geometry.faces)

    return draw


def white_noise_stand_in(
    mesh: TriangleMesh,
    geometry: SegmentGeometry,
    covariance: VonKarmanFilterParameters,
) -> stages.FieldSampler:
    """**A stand-in, not a model**: independent noise per face, with no covariance.

    Here for one purpose -- exercising the pipeline's plumbing at a resolution the SPDE
    sampler cannot yet reach, because its direct sparse factorisation costs 392 s and
    6.2 GB at 300 thousand vertices and the 400 m Hikurangi mesh has 1.19 million. It is
    a **loud, explicit** stand-in: it warns on every construction, it is never selected
    automatically, and :func:`generate` takes it only if a caller names it. A silent
    fallback here would produce a rupture that looks entirely plausible and has no
    asperities at all -- the slip field would be white noise wearing a slip
    distribution's units.

    Nothing about the taper, the moment, the wavefront or the SRF is affected by using
    it. What is destroyed is the spatial structure of every drawn field, which is most
    of what the model *is*.

    Parameters
    ----------
    mesh : TriangleMesh
        The segment, unread.
    geometry : SegmentGeometry
        Its hoisted geometry, for the face count.
    covariance : VonKarmanFilterParameters
        Unread, and that is the point.

    Returns
    -------
    stages.FieldSampler

    Warns
    -----
    StandInField
        Always.
    """
    del mesh, covariance
    warnings.warn(
        "drawing this segment's fields as white noise: this is a stand-in for the "
        "Matern sampler, not a coarser version of it. The slip field will have no "
        "correlation structure at all, so the rupture has no asperities and its "
        "wavelength content is the mesh's. Use it to exercise the pipeline at a "
        "resolution the SPDE cannot yet reach, and never for a rupture anyone runs",
        StandInField,
        stacklevel=2,
    )
    faces = geometry.face_count

    def draw(
        chart: object,
        asked: VonKarmanFilterParameters,
        rng: np.random.Generator,
    ) -> FloatArray:
        """One field of independent standard normals, one value per face."""
        del chart, asked
        return rng.standard_normal(faces)

    return draw


type SamplerFactory = Callable[
    [TriangleMesh, SegmentGeometry, "VonKarmanFilterParameters"], stages.FieldSampler
]
"""How :func:`draw_fields` gets a segment's draw: :func:`matern_sampler`, or a stand-in.

A parameter rather than an import so that the one legitimate reason to use anything
other than the SPDE -- reaching a resolution it cannot yet factorise -- is a decision
the caller makes and is visible in the call, rather than a fallback the pipeline takes
when something is slow.
"""


# ============================================================================
# S7 -- the wavefront
# ============================================================================


def face_seeds(geometry: SegmentGeometry, face: int, t0_s: float) -> list[fim.Seed]:
    """The seeds that start a front from one subfault: all three of its corners.

    **The seam between a cell-indexed pipeline and a vertex-based solver**, and the one
    place the two disagree about what a source is.
    :func:`~rupture_generator.triangular.fim.solve` seeds vertices, and
    :func:`~rupture_generator.triangular.fim.face_arrivals` gives a face the mean of its
    three corners because the solution is piecewise linear and the mean *is* the value
    at the centroid. Seeding one corner therefore leaves the hypocentre's own face
    arriving late by about a third of the sum of its corner distances -- of order
    ``0.4 h S``, which is 0.2 s at a 1 km cut and 2.6 km/s, in the one quantity
    `MESH.md` singles out as having no perturbation to hide behind.

    Seeding all three corners makes it exactly ``t0``: each corner is its own ball's
    centre, a fixed vertex keeps its seeded time, and the mean of three ``t0`` is
    ``t0``. That is also the faithful port of what the lattice solver does, where the
    seed is a *cell* and the whole cell is at ``t = 0``; it is not a smeared source. The
    three balls overlap and cost three solves, which is
    :func:`~rupture_generator.triangular.fim.solve`'s own multi-seed contract -- first
    arrival from several sources is the pointwise minimum over sources -- rather than a
    new convention.

    Parameters
    ----------
    geometry : SegmentGeometry
        The segment's hoisted geometry.
    face : int
        Which face the front starts from.
    t0_s : float
        When it leaves: zero for a configured hypocentre, the parent's arrival plus the
        jump delay for a triggered segment.

    Returns
    -------
    list of fim.Seed
        Three seeds, one per corner.
    """
    return [
        fim.Seed(vertex=int(vertex), t0_s=float(t0_s))
        for vertex in geometry.faces[face]
    ]


def travel_times(
    geometry: SegmentGeometry,
    shear_speed_km_s: FloatArray,
    params: timing.SpeedParams,
    seeds: Sequence[fim.Seed],
) -> tuple[FloatArray, tuple[fim.SeedReport, ...]]:
    """S7 on a triangulation: first-arrival times per face, in seconds.

    `timing.travel_times`' counterpart. The speed field is `timing.speed_field`
    unchanged -- it is elementwise in depth and shear speed, and a face has both -- and
    what changes is the solver: `fim.solve` walks the surface's own triangles instead of
    a Cartesian stencil, so no spacing is read and no index space exists.

    **Slowness is per face and the answer is per vertex**, which is
    :mod:`~rupture_generator.triangular.fim`'s own convention rather than a choice made
    here: Fu et al. (2011) assign a constant speed to each triangle, and P1 finite
    elements solve at vertices. The reduction back to faces is
    :func:`~rupture_generator.triangular.fim.face_arrivals`, which is the interpolated
    value at the centroid the moment tensor is placed at.

    Parameters
    ----------
    geometry : SegmentGeometry
        The segment's hoisted geometry.
    shear_speed_km_s : FloatArray
        ``(F,)`` from the velocity model at each face's own centre depth.
    params : timing.SpeedParams
        How fast the front travels here.
    seeds : Sequence of fim.Seed
        Where the front starts, and when -- from :func:`face_seeds`.

    Returns
    -------
    tuple
        ``(F,)`` travel times in seconds, and one
        :class:`~rupture_generator.triangular.fim.SeedReport` per seed. The reports are
        returned rather than logged because ``r0``'s two bounds are evidence: a derived
        quantity nobody measures is a configured one that has stopped being written
        down.

    Raises
    ------
    ValueError
        For a speed the front cannot travel at, or anything `fim.solve` refuses.
    """
    speed_km_s = timing.speed_field(geometry.depth_km, shear_speed_km_s, params)
    vertex_s, reports = fim.solve_with_report(
        geometry.vertices_km, geometry.faces, 1.0 / speed_km_s, seeds
    )
    return fim.face_arrivals(geometry.faces, vertex_s), reports


# ============================================================================
# The stages, in order
# ============================================================================


def attach_materials(
    realisation: Realisation,
    geometries: Mapping[str, SegmentGeometry],
    velocity_model: VelocityModelConfig,
) -> Realisation:
    """The rock each subfault is in: shear speed, rigidity and density.

    `pipeline.attach_materials` on faces, reading the depths off the hoisted geometry
    rather than asking the mesh again. Every later stage reads these off the chart
    rather than the model, so the moment fold, the wavefront and the SRF all describe
    the same rock.

    Parameters
    ----------
    realisation : Realisation
        Annotated in place, and returned.
    geometries : Mapping of str to SegmentGeometry
        One per segment.
    velocity_model : VelocityModelConfig
        The 1-D model.

    Returns
    -------
    Realisation
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
    *,
    sampler_of: SamplerFactory = matern_sampler,
) -> Realisation:
    """The four drawn fields: slip pattern, rise time, rake, onset perturbation.

    `pipeline.draw_fields` with two substitutions and no third: the draw comes from
    ``sampler_of`` instead of the circulant embedding, and the taper is
    :func:`taper_edges` instead of the lattice one. The rise-time model, the rake model
    and the perturbation model are `stages.py`'s own, called here, so there is one of
    each in the package.

    One batch, and the only place anything is drawn. Rise time and the perturbation both
    correlate against slip's own Gaussian, so the Gaussian and the sampler are locals
    that never leave this function -- and the sampler is a local for a second reason
    here: it holds a factorisation the size of the mesh.

    Parameters
    ----------
    realisation : Realisation
        Annotated in place, and returned.
    geometries : Mapping of str to SegmentGeometry
        One per segment.
    config : RuptureConfig
        What the earthquake is.
    sampler_of : SamplerFactory, optional
        How to build each segment's draw. See :data:`SamplerFactory`; the default is
        the SPDE sampler and the alternative is a stand-in that says so.

    Returns
    -------
    Realisation
    """
    source = config.source
    random = config.random

    for name, mesh in list(realisation.items()):
        geometry = geometries[name]
        covariance = source.covariance_of(name)
        sampler = sampler_of(mesh, geometry, covariance)
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

    `pipeline.constant_fields` on faces. A point source is this pipeline with constant
    fields, not a path of its own, so it provides exactly the names :func:`draw_fields`
    does and no later stage asks which kind of source it has -- including the
    perturbation, which is a field of zeros rather than a missing one.

    Parameters
    ----------
    realisation : Realisation
        Annotated in place, and returned.
    geometries : Mapping of str to SegmentGeometry
        One per segment.
    source : PointSourceConfig
        Carrying the given rise time.
    field : FieldConfig
        Carrying the base rake.

    Returns
    -------
    Realisation
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

    `pipeline.scale_moment` verbatim in its arithmetic -- `moment.py` is flat-array
    arithmetic and needs nothing for a triangulation -- with the areas read off the
    hoisted geometry. Either one factor over every segment, so how the moment divides
    between faults is the fields' own, or a target per fault when the source states the
    division.

    Parameters
    ----------
    realisation : Realisation
        Annotated in place, and returned.
    geometries : Mapping of str to SegmentGeometry
        One per segment.
    source : SourceConfig
        What the earthquake is.

    Returns
    -------
    Realisation
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

    `pipeline.solve_onsets` with three seams changed and its argument intact. **Draws
    nothing**: every random choice was made in :func:`draw_fields`, so the one
    order-dependent traversal is a pure function of its inputs.

    The seams. The hypocentre is one flat face index rather than an ``(i, j)`` pair, and
    it is found from **true arc lengths** because "12 km along strike" means along the
    fault. The seeds are the three corners of that face rather than one lattice cell --
    see :func:`face_seeds`, which is where the exactness of the pinned hypocentre onset
    comes from. And `propagation.causal_jump` returns a flat face index for both cells,
    because that is what a triangulated chart calls a subfault; nothing else about the
    jump search changes, including the argument that jumps leave from *arrested* tips.

    Each segment also records what its seed ball actually was --
    :data:`~rupture_generator.triangular.fim.SEED_RING_DEPTH` and
    :data:`~rupture_generator.triangular.fim.SEED_SLOWNESS_BUDGET_S` are bounds on a
    derived radius, and a bound nobody can see is a free parameter keeping quiet.

    Parameters
    ----------
    realisation : Realisation
        Annotated in place, and returned.
    geometries : Mapping of str to SegmentGeometry
        One per segment.
    config : RuptureConfig
        What the earthquake is.

    Returns
    -------
    Realisation

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
            seeds = face_seeds(geometry, hypocentre, 0.0)
            pinned: int | None = hypocentre
            delay_s = config.timing.rupture_delay_s
        else:
            # Chosen on the parent's wavefront and timed on its onset: an argmin over a
            # hundred thousand perturbed values is an order statistic that finds the
            # perturbation's negative tail rather than the shape of the front.
            jump = propagation.causal_jump(
                solved[parent],
                solved[parent]["wavefront_s"],
                mesh,
                jump_delay,
                parent_onset_s=solved[parent]["onset_s"],
                max_distance_km=max_jump_km,
            )
            jumps[name] = jump
            seeds = face_seeds(geometry, int(jump.child_cell), jump.arrival_s)
            # Triggered from elsewhere: no pin and no clamp, so this segment's onsets
            # stay absolute. That is what lets a rupture propagate between faults
            # rather than restarting on each.
            pinned = None
            delay_s = 0.0

        travel_time_s, reports = travel_times(
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
        ).with_attrs(
            seed_radius_km=max(report.radius_km for report in reports),
            seed_slowness_error_s=max(report.slowness_error_s for report in reports),
            unsplit_obtuse_wedges=max(
                report.unsplit_obtuse_wedges for report in reports
            ),
        )

    # The hypocentre, in the root's own arc lengths and under the names a file uses, so
    # the writer copies it and no segment needs a special case. Clamped to the fault's
    # extent, which is `arc_profile`'s last knot rather than any one vertex's arc
    # length: the vertices of a triangulation are in no particular order, so the
    # lattice's "last node" has no counterpart here.
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


def synthesise_pulses(
    realisation: Realisation,
    geometries: Mapping[str, SegmentGeometry],
    config: RuptureConfig,
) -> Realisation:
    """S9: a slip-rate pulse per subfault, carrying that subfault's slip.

    `pulses.synthesise` unchanged: it takes a depth and a slip per subfault and knows
    nothing about the shape of the chart, which is what MESH.md means by this stage
    needing zero work. The only difference from `pipeline.synthesise_pulses` is that the
    depths come off the hoisted geometry.

    Parameters
    ----------
    realisation : Realisation
        Annotated in place, and returned.
    geometries : Mapping of str to SegmentGeometry
        One per segment.
    config : RuptureConfig
        What the earthquake is.

    Returns
    -------
    Realisation
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
    sampler_of: SamplerFactory = matern_sampler,
) -> Realisation:
    """Run the pipeline over a triangulated fault system.

    The same stage order as `pipeline.generate`, written down the same way -- as this
    function's own body. The geometry is a system nothing has been drawn on; the result
    is the same segments with the rupture attached, which is why the two are one type.

    The derived geometry of every segment is computed **here, once**, and passed to
    every stage; see :class:`SegmentGeometry` for the measurement that makes that worth
    doing.

    Parameters
    ----------
    config : RuptureConfig
        What the earthquake is.
    geometry : Realisation
        The segments and the frame they are in, from :func:`segments_of` or
        :func:`charts_for`. **Annotated in place**: the segments, the tree and the jumps
        are written onto this object, which is also what is returned.
    sampler_of : SamplerFactory, optional
        How each segment's fields are drawn. The default is the SPDE sampler; passing
        :func:`white_noise_stand_in` instead is a deliberate, warned-about choice and
        never something this function makes on its own.

    Returns
    -------
    Realisation

    Raises
    ------
    ValueError
        For a hypocentre off the fault, a propagation that is not a tree, an
        unrepresentable rise time (naming the subfault), or a rupture speed the front
        cannot travel at.
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
        realisation = draw_fields(
            realisation, geometries, config, sampler_of=sampler_of
        )

    realisation = scale_moment(realisation, geometries, source)
    realisation = solve_onsets(realisation, geometries, config)
    return synthesise_pulses(realisation, geometries, config)


def write_rupture_mesh(realisation: Realisation, path: Path | str) -> None:
    """Write a generated triangular rupture out as a version 3 mesh file.

    A convenience, and an honest one: :func:`~rupture_generator.triangular.mesh
    .write_mesh` writes each segment's whole dataset, which after the pipeline includes
    every attached field and the CSR pulses. So the triangular track has a native round
    trip without a rupture-file format of its own -- what MESH.md's phase 3 would add is
    a *reader's* vocabulary (areas in square metres, node positions), not storage.

    Each *segment* is written as its own surface holding one segment, because after
    fusion a segment is what ruptures and its name is what the causality tree uses; the
    file's group names are then the names a config selects.

    Parameters
    ----------
    realisation : Realisation
        A rupture that has been through the pipeline. Its own CRS is what is stored.
    path : Path or str
        Where to write it: ``.h5`` or ``.zarr``.
    """
    from rupture_generator.triangular.mesh import write_mesh

    write_mesh(
        {name: [mesh] for name, mesh in realisation.items()},
        realisation.crs,
        Path(path),
    )


__all__ = [
    "BOUNDARY_SAMPLES_PER_EDGE",
    "Realisation",
    "SamplerFactory",
    "SegmentGeometry",
    "StandInField",
    "attach_materials",
    "charts_for",
    "constant_fields",
    "draw_fields",
    "face_seeds",
    "generate",
    "matern_sampler",
    "scale_moment",
    "segments_of",
    "solve_onsets",
    "synthesise_pulses",
    "taper_edges",
    "travel_times",
    "white_noise_stand_in",
    "write_rupture_mesh",
]
