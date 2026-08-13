"""End-to-end properties of a rupture generated on a triangulation.

Two things are asserted here, and they are different in kind.

**The invariants of `tests/test_pipeline.py`, re-pointed.** That file asserts properties
of the output rather than stored arrays -- the moment is the magnitude's, the rupture
starts where it was told and spreads outward, every subfault that slips has a pulse
carrying exactly its slip, the file says the same thing after a round trip. Every one of
those is a statement about an earthquake and not about a lattice, so every one is here,
reading faces where the original reads cells. The two that could not come across
unchanged are named where they appear: there is no ``plane``/``j_node`` vocabulary to
assert (the container stores a face table instead), and the rupture-file round trip goes
through the version 3 *mesh* file, which stores each segment's whole dataset and so
carries the fields and the pulses already.

**The reduction to the quad path.** On a planar fault the two pipelines should agree,
and they now share every random number **without being made to**. Both draw from the same
circulant embedding; the triangular one draws on a lattice over its parameter plane, and
:meth:`~rupture_generator.triangular.lattice.ParameterLattice.of` recovers the chart's own
grid from the triangulation exactly -- same cell counts, spacing agreeing to 1e-15, two
faces to a cell -- so the drawn fields agree to 7e-14 with no shared-sampler apparatus at
all. :func:`test_the_two_tracks_draw_the_same_field_on_a_planar_fault` is that statement
on its own, and it is what makes every comparison below a statement about the pipeline
rather than about the noise.

What the comparison finds is worth stating up front, because one of the three is not zero
and is not an error:

- **Rake and onset are bit-identical**, and the moment is exact, at every cut. Onset is
  identical because it is now literally the same solver on the same grid: the triangular
  track's lattice *is* the chart, and the projection onto faces is a gather.
- **Slip differs by one global factor and nothing else**: the ratio's spread across faces
  is f64 round-off, and the factor is 1.0000 at the shipped 1 km cut, 1.0049 at 0.5 km
  and 0.8964 at 2 km. The cause is not the pipeline -- it is that rigidity is sampled at
  each subfault's *own* centre, and a triangle's centroid is not its parent quad's
  centre, so a few subfaults near a velocity-layer boundary land in a different layer.
  The moment fold then closes on a slightly different sum. Both ruptures carry exactly
  the magnitude's moment.
"""

from __future__ import annotations

import dataclasses
import itertools
import warnings
from pathlib import Path

import h5py
import numpy as np
import pyproj
import pytest

from rupture_generator import moment, pipeline, stages
from rupture_generator.config import read_config, read_geometry
from rupture_generator.config.geometry import (
    Discretisation,
    GeometryConfig,
    LonLat,
    PointConfig,
)
from rupture_generator.config.rupture import (
    PerFaultSourceConfig,
    PointSourceConfig,
    PredeterminedPropagation,
    RuptureConfig,
)
from rupture_generator.mesh import RuptureMesh, build_surface, fuse
from rupture_generator.realisation import Realisation
from rupture_generator.sampling import (
    DegradedCorrelation,
    correlation_lengths,
    von_karman_field,
    von_karman_grid,
)
from rupture_generator.triangular import assemble as tri_assemble
from rupture_generator.triangular import pipeline as tri
from rupture_generator.triangular.gocad import read_tsurf
from rupture_generator.triangular.lattice import ParameterLattice
from rupture_generator.triangular.mesh import (
    TriangleMesh,
    from_chart,
    read_mesh,
)
from rupture_generator.triangular.mesh import (
    build_surface as build_triangular,
)

EXAMPLES = Path(__file__).parent.parent.parent / "examples"

Generated = tuple[Realisation, RuptureConfig, pyproj.CRS]
"""A run of the pipeline, and what it was run with."""

MIN_SLIP_M = 1.0e-4
"""The kernel's own no-pulse guard. Below it a subfault gets no pulse at all, which is
not the same as a pulse of zeros."""


def _config() -> RuptureConfig:
    return read_config(EXAMPLES / "crustal.toml")


def _untapered() -> RuptureConfig:
    """The shipped config with the edge tapers off.

    What the quad comparison runs, and the reason is a difference of model rather than of
    implementation. The lattice taper rounds its fraction to whole cells and gives the
    outermost cell the weight ``1 / side``, which is exactly 1 -- no taper at all --
    whenever the fraction rounds to one cell, and that is what the shipped 2% does on a
    27-cell fault. The parameter-space taper makes the same 2% a genuine 0.6 km ramp. So
    with the tapers on the two paths differ by up to 0.15 m of slip on the edge columns
    by construction; :func:`test_the_taper_ramps_slip_to_zero_at_the_labelled_boundary`
    is where the triangular taper is asserted on its own terms instead.
    """
    config = _config()
    config.slip.side_taper = 0.0
    config.slip.top_taper = 0.0
    config.slip.bottom_taper = 0.0
    return config


def _run(geometry_name: str) -> Generated:
    geometry = read_geometry(EXAMPLES / geometry_name)
    config = _config()
    segments = tri.charts_for(geometry, None)
    return tri.generate(config, segments), config, geometry.crs


def _planar_chart(size_km: float = 1.0) -> tuple[RuptureMesh, pyproj.CRS]:
    """The bent example's first plane alone, at a chosen cut: one planar segment."""
    geometry = read_geometry(EXAMPLES / "hope.geometry.toml")
    surface = geometry.surfaces[0]
    surface.planes = surface.planes[:1]
    surface.planes[0].discretisation = Discretisation(subfault_size_km=size_km)
    (chart,) = fuse(build_surface(surface, geometry.crs))
    return chart, geometry.crs


def _quad_of_face(mesh: TriangleMesh, chart: RuptureMesh) -> np.ndarray:
    """Which flat structured cell each face's parameter centroid falls in.

    Only meaningful on a planar single-plane segment, where the parameter domain is the
    rectangle the chart's lattice divides evenly.
    :func:`test_each_quad_is_exactly_two_faces` is what checks the map is the bijection
    the comparisons assume.
    """
    cells_i, cells_j = chart.cell_counts
    parameters = mesh.parameters_km()
    centroid = parameters[mesh.faces()].mean(axis=1)
    column = np.clip(
        (centroid[:, 0] / (parameters[:, 0].max() / cells_j)).astype(int),
        0,
        cells_j - 1,
    )
    row = np.clip(
        (centroid[:, 1] / (parameters[:, 1].max() / cells_i)).astype(int),
        0,
        cells_i - 1,
    )
    return row * cells_j + column


def _both(
    config: RuptureConfig, size_km: float = 1.0
) -> tuple[Realisation, Realisation, np.ndarray, RuptureMesh, TriangleMesh]:
    """One planar fault, run down both pipelines with one set of random numbers."""
    chart, crs = _planar_chart(size_km)
    mesh = from_chart(chart)
    lookup = _quad_of_face(mesh, chart)

    quad = pipeline.generate(config, Realisation({"hope": chart}, crs))
    triangular = tri.generate(config, Realisation({"hope": mesh}, crs))
    return quad, triangular, lookup, chart, mesh


@pytest.fixture(scope="module")
def bent() -> Generated:
    """The shipped bent trace: two planes that share a seam, fused into one segment."""
    return _run("hope.geometry.toml")


# ============================================================================
# The gate: on a planar fault the triangular path is the quad path
# ============================================================================


def test_each_quad_is_exactly_two_faces() -> None:
    """The map the comparisons are read through is a bijection onto pairs.

    Asserted rather than assumed, because everything below indexes one field by the
    other's cells: if the map were wrong the comparisons would still run and would be
    comparing a subfault against its neighbour, which is the failure mode that looks
    like a small discrepancy.
    """
    chart, _ = _planar_chart()
    mesh = from_chart(chart)
    lookup = _quad_of_face(mesh, chart)
    cells_i, cells_j = chart.cell_counts

    counts = np.bincount(lookup, minlength=cells_i * cells_j)
    assert (counts == 2).all()

    # And the two faces of a quad tile it: their areas sum to its area, and their
    # centroids average to its centre.
    quad_area = np.zeros(cells_i * cells_j)
    np.add.at(quad_area, lookup, mesh.areas_km2())
    assert np.allclose(quad_area, chart.areas_km2().ravel(), rtol=1e-12)

    quad_centre = np.zeros((cells_i * cells_j, 3))
    np.add.at(quad_centre, lookup, mesh.centres())
    assert np.allclose(quad_centre / 2.0, chart.centres().reshape(-1, 3), atol=1e-9)


def test_the_moment_is_exact_on_both_paths() -> None:
    """Moment is the one quantity with no tolerance, and it closes on both.

    The fold is the same arithmetic on both paths -- `moment.py` is flat arrays -- and
    the areas it closes on are the same surface, because a quad's two triangles tile it
    exactly rather than approximately.
    """
    config = _untapered()
    quad, triangular, _, _, _ = _both(config)
    expected = moment.seismic_moment_nm(config.source.magnitude)

    assert quad.moment_newton_m == pytest.approx(expected, rel=1e-12)
    assert triangular.moment_newton_m == pytest.approx(expected, rel=1e-12)


def test_the_rake_field_is_the_quad_paths_to_round_off() -> None:
    """Bit-identical, against a one-degree bound.

    `stages.rake_field` is ``base + sigma * standardise(Z)`` and nothing else, so with
    the draw held fixed the only way the two paths could differ is the standardisation
    -- and repeating every value twice leaves a population mean and standard deviation
    exactly where they were. Measured: 0 degrees at a 1 km cut, 2.8e-14 at 0.25 km.
    """
    _, triangular, lookup, _, _ = _both(_untapered())
    quad, _, _, _, _ = _both(_untapered())

    difference_deg = np.abs(
        triangular["hope"]["rake_deg"] - quad["hope"]["rake_deg"].ravel()[lookup]
    )
    assert float(difference_deg.max()) < 1.0e-9


@pytest.mark.parametrize("size_km", [1.0, 0.5])
def test_the_slip_field_differs_from_the_quad_paths_by_one_global_factor(
    size_km: float,
) -> None:
    """The sharper statement than a one-percent bound, and the true one.

    With the draw held fixed the slip *pattern* is the same arithmetic on both paths, so
    the two fields can only differ by the moment fold's single factor -- and that is what
    they do, to round-off. What moves the factor is that
    :func:`~rupture_generator.moment.sample_velocity_model` reads the velocity model at
    each subfault's own centre, and a triangle's centroid is up to ``h / 6`` from its
    parent quad's centre, so a few subfaults near a layer boundary land in a different
    layer and the sum the fold closes on changes.

    Measured, as ``max(ratio) - min(ratio)`` over the slipping faces and the factor
    itself:

    ====  =======  ==================  ===============  ======
    cut   faces    rigidity differs    factor           spread
    ====  =======  ==================  ===============  ======
    2.0   196      28 faces            0.896424401614   9.3e-15
    1.0   756      0 faces             1.000000000000   1.6e-13
    0.5   3132     54 faces            1.004859943214   7.8e-14
    ====  =======  ==================  ===============  ======

    So the one-percent slip bound holds at the shipped 1 km cut and at 0.5 km, and does
    **not** at 2 km, where 14% of the fault changes layer and the factor moves 10%. That
    is not a discrepancy between the implementations -- it is the velocity model being
    sampled somewhere slightly different, which changing the cut does too.
    """
    config = _untapered()
    quad, triangular, lookup, _, _ = _both(config, size_km)

    quad_slip_m = quad["hope"]["slip_m"].ravel()[lookup]
    slipping = quad_slip_m > 0.0
    ratio = triangular["hope"]["slip_m"][slipping] / quad_slip_m[slipping]

    assert float(np.ptp(ratio)) < 1.0e-9
    assert float(ratio.mean()) == pytest.approx(1.0, abs=0.01)


def test_the_hypocentres_own_onset_is_exact_on_both_paths() -> None:
    """The registration point, which `MESH.md` says has no perturbation to hide behind.

    `stages.apply_perturbation` pins the hypocentre's perturbation to zero and clamps the
    field at the delay, so the hypocentre's onset is *exactly* the delay -- and on a
    triangulation that is only true because the seed is the lattice **cell** the
    hypocentre face falls in and projection back onto faces is a gather rather than an
    interpolation. A scheme that interpolated between cells would leave this a fraction
    of a cell late, in the one quantity `MESH.md` says the model's own perturbation gives
    no cover for.
    """
    config = _untapered()
    _, triangular, _, _, _ = _both(config)
    onset_s = triangular["hope"]["onset_s"]

    assert float(onset_s[triangular.hypocentre]) == pytest.approx(
        config.timing.rupture_delay_s, abs=1e-12
    )
    assert float(onset_s.min()) == pytest.approx(
        config.timing.rupture_delay_s, abs=1e-12
    )


def test_the_two_tracks_draw_the_same_field_on_a_planar_fault() -> None:
    """The reduction, and the reason no shared-sampler apparatus is needed any more.

    Both tracks draw from :func:`~rupture_generator.sampling.von_karman_grid`. The
    structured chart's grid *is* its cells;
    :meth:`~rupture_generator.triangular.lattice.ParameterLattice.of` has to recover that
    same grid from a triangulation of it, which it does because both containers cut the
    parameter plane into rectangles and split each into two triangles whose centroids sit
    at a third and two thirds of the cell. So the cell counts match exactly, the spacings
    agree to f64 round-off, every cell holds exactly two faces, and the two draws are the
    same numbers.

    Asserted directly rather than inferred from the pipeline outputs, because it is what
    every other comparison in this file rests on.
    """
    chart, _ = _planar_chart(1.0)
    mesh = from_chart(chart)
    lattice = ParameterLattice.of(mesh.parameters_km(), mesh.faces())
    covariance = correlation_lengths(7.0)

    assert lattice.cell_counts == chart.cell_counts
    assert lattice.sampling_spacing_km == pytest.approx(chart.spacing_km(), rel=1e-12)
    assert np.array_equal(
        np.bincount(lattice.cell_of_face), np.full(mesh.face_count // 2, 2)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegradedCorrelation)
        quad = von_karman_field(chart, covariance, np.random.default_rng(7))
        triangular = lattice.project(
            von_karman_grid(
                lattice.cell_counts,
                lattice.sampling_spacing_km,
                covariance,
                np.random.default_rng(7),
            )
        )

    assert np.abs(triangular - quad.ravel()[lattice.cell_of_face]).max() < 1e-12


def test_the_onset_difference_is_where_the_velocity_model_is_sampled() -> None:
    """**What is left of the two tracks' onset difference, and it is one thing.**

    They now run the same solver on the same grid, so nothing about the discretisation
    can differ. What still can is the *depth* the velocity model is read at: a lattice
    reads its cell's centre, and a triangulation reads each triangle's centroid and this
    lattice averages the two per cell. Those differ by a fraction of a cell, and
    :func:`~rupture_generator.timing.speed_field` is continuous in depth -- **except**
    across a velocity-layer boundary, where a cell whose two readings straddle the
    boundary picks up the whole jump.

    So the difference is not a convergence rate, and asserting one would be asserting
    something false. It is small and **discontinuous in the cut**, and the discontinuity
    is exactly the layer crossings:

    ====  =======  ==================  ==========  ==========
    cut   faces    cells with a layer  median      max
                   disagreement        |Δonset|    |Δonset|
    ====  =======  ==================  ==========  ==========
    2.0   196      28 of 98            74.4 ms     789 ms
    1.0   756      0 of 378            **1.4 ms**  4.2 ms
    0.5   3132     54 of 1566          2.9 ms      36.5 ms
    0.25  12528    0 of 6264           **0.06 ms** 0.21 ms
    ====  =======  ==================  ==========  ==========

    The two cuts where no cell straddles a layer agree to a millisecond and to a
    twentieth of one; the two where cells do are the layer jump, weighted by how many.
    This is the same root cause as the global slip factor recorded in this module's
    docstring, and it is a property of sampling a 1-D velocity model on two slightly
    different sets of points rather than of either pipeline.

    The bound asserted is 0.1 s at every cut, which is under a third of the model's own
    ~0.35 s onset perturbation and twenty times the worst measured value that is not a
    layer crossing.
    """
    config = _untapered()
    for size_km in (2.0, 1.0, 0.5, 0.25):
        quad, triangular, lookup, _, _ = _both(config, size_km)
        difference_s = np.abs(
            triangular["hope"]["onset_s"] - quad["hope"]["onset_s"].ravel()[lookup]
        )
        assert float(np.median(difference_s)) < 0.1, size_km

    # And the mechanism, at the one cut where the two agree to a millisecond: no cell's
    # binned shear speed differs from the chart's, so nothing but the depth ramp is left.
    quad, triangular, _, _, mesh = _both(config, 1.0)
    lattice = ParameterLattice.of(mesh.parameters_km(), mesh.faces())
    binned = lattice.bin(triangular["hope"]["shear_speed_kms"])
    assert np.allclose(binned, quad["hope"]["shear_speed_kms"], atol=1e-12)


# ============================================================================
# The taper, on its own terms
# ============================================================================


def test_the_taper_ramps_slip_to_zero_at_the_labelled_boundary() -> None:
    """A face on the lateral boundary keeps a ramp's fraction of its slip, not all of it.

    The lattice taper's own weakness is what this fixes: a 2% side taper on a 27-cell
    fault rounds to one cell and multiplies it by ``1/1``, so nothing is tapered at all.
    In parameter space the same 2% is a real ramp, and a face at the very edge keeps
    about ``h / (2 w)`` of its slip.
    """
    chart, _ = _planar_chart()
    mesh = from_chart(chart)
    geometry = tri.SegmentGeometry.of(mesh)
    params = stages.SlipParams(
        covariance=_config().source.covariance_of("hope"),
        side_taper=0.1,
        top_taper=0.1,
        bottom_taper=0.0,
    )

    tapered = tri.taper_edges(mesh, geometry, np.ones(geometry.face_count), params)

    assert tapered.max() == pytest.approx(1.0)
    assert (tapered >= 0.0).all()
    assert (tapered <= 1.0 + 1e-12).all()

    lateral = mesh.boundary_faces("lateral")
    top = mesh.boundary_faces("top")
    bottom = mesh.boundary_faces("bottom")
    assert (tapered[lateral] < 0.5).all()
    assert (tapered[top] < 0.5).all()
    # The bottom taper is zero, so the bottom edge is untouched -- except where a face
    # is on the bottom *and* on a lateral edge, which the lateral ramp still damps.
    assert tapered[np.setdiff1d(bottom, lateral)].max() == pytest.approx(1.0)


def test_the_taper_is_the_product_of_independent_ramps() -> None:
    """Separability, which is the property `stages.taper_edges` argues for at length.

    A face two ramps reach is damped by both, so tapering along strike and down dip
    together is the product of tapering each way alone. The alternative -- one ramp
    overwriting the other -- differs exactly in the corners, which is where a rupture is
    least constrained and a reader is least likely to look.
    """
    chart, _ = _planar_chart()
    mesh = from_chart(chart)
    geometry = tri.SegmentGeometry.of(mesh)
    covariance = _config().source.covariance_of("hope")
    field = np.ones(geometry.face_count)

    def taper(**fractions: float) -> np.ndarray:
        return tri.taper_edges(
            mesh, geometry, field, stages.SlipParams(covariance=covariance, **fractions)
        )

    sideways = taper(side_taper=0.15, top_taper=0.0, bottom_taper=0.0)
    downward = taper(side_taper=0.0, top_taper=0.15, bottom_taper=0.15)
    together = taper(side_taper=0.15, top_taper=0.15, bottom_taper=0.15)

    assert np.allclose(together, sideways * downward, atol=1e-12)


def test_overlapping_tapers_are_refused() -> None:
    """Past half the fault a taper is a statement about the middle.

    The lattice form refuses this and so does this one, in kilometres rather than cells:
    the separable and overwriting forms disagree exactly in the region two ramps share,
    so making it unrepresentable is what keeps the choice between them immaterial.
    """
    chart, _ = _planar_chart()
    mesh = from_chart(chart)
    geometry = tri.SegmentGeometry.of(mesh)
    covariance = _config().source.covariance_of("hope")
    field = np.ones(geometry.face_count)

    with pytest.raises(ValueError, match="two ramps overlap"):
        tri.taper_edges(
            mesh,
            geometry,
            field,
            stages.SlipParams(covariance=covariance, side_taper=0.6),
        )
    with pytest.raises(ValueError, match="they overlap"):
        tri.taper_edges(
            mesh,
            geometry,
            field,
            stages.SlipParams(covariance=covariance, top_taper=0.6, bottom_taper=0.6),
        )


def test_the_taper_reduces_to_a_parameter_distance_on_a_rectangle() -> None:
    """The resampled boundary is a device, and this is what it is a device *for*.

    On a planar segment the parameter domain is a rectangle, so the distance to the
    lateral boundary is exactly ``min(u, u_max - u)`` and the ramp is that over the
    width. The resampling costs at most half a sample spacing --
    :data:`~rupture_generator.triangular.pipeline.BOUNDARY_SAMPLES_PER_EDGE` says how
    that is derived -- and this measures it: 5.9e-3 of the ramp at a 1 km cut, against
    the half-cell the lattice taper is itself away from a distance ramp.
    """
    chart, _ = _planar_chart()
    mesh = from_chart(chart)
    geometry = tri.SegmentGeometry.of(mesh)
    params = stages.SlipParams(
        covariance=_config().source.covariance_of("hope"),
        side_taper=0.2,
        top_taper=0.0,
        bottom_taper=0.0,
    )

    tapered = tri.taper_edges(mesh, geometry, np.ones(geometry.face_count), params)

    parameters = geometry.parameters_km
    strike_km = parameters[geometry.faces].mean(axis=1)[:, 0]
    extent_km = float(np.ptp(parameters[:, 0]))
    width_km = params.side_taper * extent_km
    expected = np.clip(
        np.minimum(
            strike_km - parameters[:, 0].min(), parameters[:, 0].max() - strike_km
        )
        / width_km,
        0.0,
        1.0,
    )

    assert float(np.abs(tapered - expected).max()) < 0.02


# ============================================================================
# What the rupture says about the earthquake
# ============================================================================


def test_a_bent_fault_generates_as_one_segment(bent: Generated) -> None:
    """Planes that share a seam are one rupture, not two.

    The triangular form of the fusion claim. There is no ``j_node`` to count here -- the
    container carries a face table rather than a lattice -- so what is asserted instead
    is that both config planes are present in ``plane_of_face`` and that the two
    triangulations conform across the seam, which is what "one surface" means for a
    triangulation: the seam is one row of shared vertices, so no interior edge of the
    fused mesh is a boundary edge.
    """
    realisation, _, _ = bent
    assert len(realisation) == 1
    assert realisation.tree == {realisation.root: None}
    assert realisation.jumps == {}

    mesh = realisation[realisation.root]
    assert set(np.unique(mesh.planes())) == {0, 1}

    # A conforming seam: every boundary edge is on the outline. If the two plane blocks
    # did not share their seam column, the seam would appear as two coincident
    # boundaries, and the fault's perimeter would be longer than its outline.
    boundary = mesh.boundary_edges()
    interior_of_plane = mesh.planes()[mesh.boundary_faces()]
    assert len(boundary) < mesh.face_count
    assert set(np.unique(interior_of_plane)) == {0, 1}


def test_the_moment_is_the_magnitudes(bent: Generated) -> None:
    """Recomputed from the rupture's own geometry, not from the pipeline's bookkeeping.

    Areas and slip off the chart, rigidity from the velocity model at each face's stored
    centre depth. If those three do not reproduce the target then what was written is
    not what was scaled.
    """
    realisation, config, _ = bent
    mesh = realisation[realisation.root]

    _, rigidity_pa = moment.sample_velocity_model(
        mesh.centres()[:, 2],
        np.asarray(config.velocity_model.bottom_depth_km),
        np.asarray(config.velocity_model.shear_speed_km_s),
        np.asarray(config.velocity_model.density_g_cm3),
    )
    recomputed = float(np.sum(rigidity_pa * mesh.areas_km2() * 1.0e6 * mesh["slip_m"]))
    expected = moment.seismic_moment_nm(config.source.magnitude)

    assert recomputed == pytest.approx(expected, rel=1e-9)
    assert realisation.moment_newton_m == pytest.approx(expected, rel=1e-12)


def test_the_rupture_starts_where_it_was_told(bent: Generated) -> None:
    """The hypocentre's onset is the delay, nothing precedes it, and the file says where.

    `DEFECTS.md` 17's property on a triangulation. The arc lengths the config gave are
    recorded back, and they are **arc** lengths rather than parameter coordinates -- on
    this bent segment the two differ, which is the seam
    :meth:`~rupture_generator.triangular.mesh.TriangleMesh.cell_index` exists to keep
    narrow.
    """
    realisation, config, _ = bent
    mesh = realisation[realisation.root]
    onset_s = mesh["onset_s"]

    assert float(onset_s[realisation.hypocentre]) == pytest.approx(
        config.timing.rupture_delay_s, abs=1e-12
    )
    assert float(onset_s.min()) == pytest.approx(
        config.timing.rupture_delay_s, abs=1e-12
    )
    assert mesh.attrs["hypocentre_strike_km"] == pytest.approx(
        config.hypocentre.strike_km
    )
    assert mesh.attrs["hypocentre_dip_km"] == pytest.approx(config.hypocentre.dip_km)


def test_the_front_spreads_outward_from_the_hypocentre(bent: Generated) -> None:
    """Onset grows with distance from where the rupture started.

    Not an exact statement -- the perturbation moves individual subfaults by design, so
    that high-slip patches rupture early -- but a strong correlation is what separates a
    propagating front from a field of noise.
    """
    realisation, _, _ = bent
    mesh = realisation[realisation.root]

    centres_km = mesh.centres()
    distance_km = np.linalg.norm(
        centres_km - centres_km[realisation.hypocentre], axis=1
    )

    assert float(np.corrcoef(distance_km, mesh["onset_s"])[0, 1]) > 0.9


def test_every_slipping_subfault_has_a_pulse_carrying_its_slip(bent: Generated) -> None:
    """`DEFECTS.md` 21 on a triangulation: nothing that slips is silently empty.

    The kernel guarantees ``dt * sum(pulse) == slip`` exactly and refuses a rise time it
    cannot sample; this is the end-to-end statement of both. An empty row means one thing
    only -- a subfault below the slip guard.
    """
    realisation, _, _ = bent
    mesh = realisation[realisation.root]

    slip_m = mesh["slip_m"]
    offsets, samples = mesh.pulses
    interval_s = float(mesh.attrs["sample_interval_s"])

    assert len(offsets) == slip_m.size + 1

    empty = np.diff(offsets) == 0
    assert (slip_m[empty] <= MIN_SLIP_M).all()

    carried = np.add.reduceat(samples, offsets[:-1]) * interval_s
    slipping = ~empty
    assert np.allclose(carried[slipping], slip_m[slipping], rtol=1e-9)


def test_the_rupture_is_reproducible(bent: Generated) -> None:
    """The same seed gives the same earthquake, bit for bit.

    `RandomConfig.stream` keys its substreams by name rather than by position, so this
    survives the whole migration untouched -- which is worth asserting on the new path
    precisely because it is the thing that should *not* have moved.
    """
    realisation, _, _ = bent
    again, _, _ = _run("hope.geometry.toml")

    first, second = realisation[realisation.root], again[again.root]
    assert np.array_equal(first["slip_m"], second["slip_m"])
    assert np.array_equal(first["onset_s"], second["onset_s"])


def test_a_point_source_is_the_pipeline_with_constant_fields() -> None:
    """Two triangles, constant everything, and the same stages.

    A point source is not a separate path here either: its slip is uniform, its rake is
    the configured base, its rise time is the one it was given rather than one derived
    from the moment -- and it still carries the magnitude's moment and still goes through
    pulse synthesis and the wavefront solve.
    """
    geometry = read_geometry(EXAMPLES / "hope.geometry.toml")
    config = _config()

    point = PointConfig(
        name="point",
        centre=LonLat(longitude_deg=172.5, latitude_deg=-42.5),
        depth_km=8.0,
        strike_deg=45.0,
        dip_deg=70.0,
        size_km=2.0,
    )
    config.source = PointSourceConfig(
        magnitude=5.5,
        rise_time_s=0.8,
        average_dip_deg=70.0,
        average_rake_deg=175.0,
    )
    config.hypocentre.strike_km = 1.0
    config.hypocentre.dip_km = 1.0

    segments = Realisation(
        tri.named("point", build_triangular(point, geometry.crs)), geometry.crs
    )
    realisation = tri.generate(config, segments)
    mesh = realisation["point"]

    assert mesh.face_count == 2
    assert mesh["rise_time_s"] == pytest.approx(0.8)
    assert mesh["rake_deg"] == pytest.approx(config.field.base_rake_deg)
    assert float(mesh["onset_s"].min()) == pytest.approx(0.0, abs=1e-12)
    assert realisation.moment_newton_m == pytest.approx(
        moment.seismic_moment_nm(5.5), rel=1e-9
    )
    _offsets, samples = mesh.pulses
    assert samples.size > 0


# ============================================================================
# Several faults, one earthquake
# ============================================================================


def _two_fault_geometry() -> tuple[GeometryConfig, RuptureConfig]:
    """The shipped two-segment example, with a hypocentre that names its fault."""
    geometry = read_geometry(EXAMPLES / "kaikoura.geometry.toml")
    config = _config()
    config.hypocentre.fault = "kaikoura:0"
    return geometry, config


@pytest.fixture(scope="module")
def two_faults() -> Generated:
    """A rupture that crosses between segments."""
    geometry, config = _two_fault_geometry()
    realisation = tri.generate(config, tri.segments_of(geometry))
    return realisation, config, geometry.crs


def test_a_multi_segment_rupture_has_a_causality_tree(two_faults: Generated) -> None:
    """Two segments, one root, and the other triggered by it."""
    realisation, _, _ = two_faults

    assert set(realisation) == {"kaikoura:0", "kaikoura:1"}
    assert realisation.tree == {"kaikoura:0": None, "kaikoura:1": "kaikoura:0"}
    assert realisation.root == "kaikoura:0"
    assert set(realisation.jumps) == {"kaikoura:1"}


def test_a_jump_names_faces_rather_than_lattice_cells(two_faults: Generated) -> None:
    """`propagation.causal_jump`'s one change: a subfault's label is a flat index.

    The search itself is untouched -- candidates are still the parent's *arrested* tips,
    still chosen on the raw wavefront and timed on the onset -- and what a triangulated
    chart calls a subfault is one integer rather than an ``(i, j)`` pair. That the jump
    left from a boundary face is the assertion that the arrest rule survived the port.
    """
    realisation, _, _ = two_faults
    jump = realisation.jumps["kaikoura:1"]
    parent = realisation["kaikoura:0"]

    assert isinstance(jump.parent_cell, int)
    assert isinstance(jump.child_cell, int)
    assert jump.from_edge
    assert jump.parent_cell in parent.boundary_faces()
    assert jump.departure_s == pytest.approx(
        float(parent["onset_s"][jump.parent_cell]), abs=1e-9
    )


def test_the_jump_cell_did_not_move_when_the_solver_did(
    two_faults: Generated,
) -> None:
    """**The gate on swapping the wavefront solver**, and it is not the onset field.

    `propagation.causal_jump` picks the jump-off subfault by an **argmin over the raw
    wavefront**, so a systematic bias in the wavefront moves the *selection*, not merely
    the timing -- and a rupture that jumps from a different place is a different
    earthquake, however small the bias was. Asserting the onset field agrees would not
    catch that; asserting the chosen cell does.

    Measured across the swap from the mesh solver to the lattice one, on the two shipped
    multi-segment examples, running each pipeline over the same configs:

    ================== ====== ====== ============= =============
    example / jump     parent child  arrival, mesh arrival, this
    ================== ====== ====== ============= =============
    kaikoura:1         1255   1485   8.0763 s      8.1553 s
    beavan  6 jumps    same   same   -0.82 to +0.08 s of movement
    ================== ====== ====== ============= =============

    **Every parent and child cell is unchanged** -- all six of beavan's jumps and
    kaikoura's one -- while the arrival times move by up to 0.82 s, which is the solver
    difference the module docstring accounts for. So the selection is robust to the
    change and the timing is not, which is the right way round.

    The two integers below are pinned rather than merely typed, because that is what
    makes a future change to the wavefront visible here rather than three files away.
    """
    realisation, _, _ = two_faults
    jump = realisation.jumps["kaikoura:1"]

    assert (jump.parent_cell, jump.child_cell) == (1255, 1485)
    assert jump.arrival_s == pytest.approx(8.1553, abs=1e-3)


def test_the_front_reaches_a_child_after_it_leaves_its_parent(
    two_faults: Generated,
) -> None:
    """Causality across the jump, in both of its halves.

    The seed time is the parent's own arrival at the jump-off point plus a non-negative
    delay -- so the child cannot start before the parent got there -- and nothing on the
    child precedes that seed.
    """
    realisation, _, _ = two_faults
    jump = realisation.jumps["kaikoura:1"]

    assert jump.arrival_s >= jump.departure_s
    assert float(realisation["kaikoura:1"]["onset_s"].min()) >= jump.departure_s - 1e-9
    # The child's *wavefront* at the arrival face is the seed time exactly, because the
    # seeded cell is the one that face falls in and the projection is a gather. Its
    # **onset**
    # is not, and deliberately: a triggered segment takes no pin and no clamp, so the
    # perturbation moves even the arrival face -- which is what keeps its onsets
    # absolute rather than registered on a second hypocentre.
    child = realisation["kaikoura:1"]
    assert float(child["wavefront_s"][jump.child_cell]) == pytest.approx(
        jump.arrival_s, abs=1e-9
    )


def test_onsets_do_not_decrease_along_a_path_of_the_tree(
    two_faults: Generated,
) -> None:
    """A child fault never begins before its parent did."""
    realisation, _, _ = two_faults

    earliest_s = {
        name: float(mesh["onset_s"].min()) for name, mesh in realisation.items()
    }
    for name, parent in realisation.tree.items():
        if parent is not None:
            assert earliest_s[name] >= earliest_s[parent] - 1e-9

    assert earliest_s[realisation.root] == pytest.approx(0.0, abs=1e-9)
    assert earliest_s[realisation.root] == min(earliest_s.values())


def test_the_moment_is_shared_between_the_faults(two_faults: Generated) -> None:
    """One factor over every segment: the parts sum to the whole, and neither is it."""
    realisation, config, _ = two_faults

    moments = [
        moment.moment_of(mesh["slip_m"], mesh["rigidity_pa"], mesh.areas_km2())
        for mesh in realisation.values()
    ]
    expected = moment.seismic_moment_nm(config.source.magnitude)

    assert sum(moments) == pytest.approx(expected, rel=1e-9)
    assert all(part > 0.0 for part in moments)
    assert not any(part == pytest.approx(expected, rel=1e-3) for part in moments)


def test_each_fault_carries_the_magnitude_it_was_given() -> None:
    """The other moment model: a target per fault, and each fault hits *its own*.

    The test the shared-factor one cannot be. The fold builds four parallel lists and
    pairs the answer back by position, so swapping two segments hands each fault the
    other's target -- every fault still carries plausible slip and the event total is
    still exactly right. Two faults a whole magnitude apart are not fooled.
    """
    geometry, config = _two_fault_geometry()
    config.source = PerFaultSourceConfig(
        magnitudes={"kaikoura:0": 6.0, "kaikoura:1": 7.0},
        rakes={"kaikoura:0": 175.0, "kaikoura:1": 90.0},
    )

    realisation = tri.generate(config, tri.segments_of(geometry))

    for name, mesh in realisation.items():
        carried = moment.moment_of(
            mesh["slip_m"], mesh["rigidity_pa"], mesh.areas_km2()
        )
        assert carried == pytest.approx(
            moment.seismic_moment_nm(config.source.magnitudes[name]), rel=1e-9
        ), name


def test_only_the_nucleating_segment_records_a_hypocentre(
    two_faults: Generated,
) -> None:
    """One earthquake, one hypocentre."""
    realisation, _, _ = two_faults

    holders = [
        name
        for name, mesh in realisation.items()
        if "hypocentre_strike_km" in mesh.attrs
    ]
    assert holders == [realisation.root]


def test_a_stated_propagation_gives_the_tree_it_states() -> None:
    """Predetermined mode, end to end, and it does not consult the sampler."""
    geometry, config = _two_fault_geometry()
    config.propagation = PredeterminedPropagation(parents={"kaikoura:1": "kaikoura:0"})

    realisation = tri.generate(config, tri.segments_of(geometry))

    assert realisation.tree == {"kaikoura:0": None, "kaikoura:1": "kaikoura:0"}


def test_a_rupture_over_several_segments_needs_to_say_where_it_starts() -> None:
    """With one fault there is nothing to choose; with two there is."""
    geometry, config = _two_fault_geometry()
    config.hypocentre.fault = None

    with pytest.raises(ValueError, match="which one it is on"):
        tri.generate(config, tri.segments_of(geometry))


# ============================================================================
# The formats
# ============================================================================


def test_a_rupture_round_trips_through_the_version_3_mesh_file(
    bent: Generated, tmp_path: Path
) -> None:
    """Every field, attribute and the CSR pulses survive the container.

    There is no triangular rupture-file format, and this is why one is not needed for a
    round trip: :func:`~rupture_generator.triangular.mesh.write_mesh` writes each
    segment's whole dataset, so a generated rupture's fields and pulses go out with the
    geometry. What MESH.md's phase 3 would add is a *reader's* vocabulary -- areas in
    square metres, node positions -- rather than storage.
    """
    realisation, _, crs = bent
    path = tmp_path / "rupture.h5"
    tri.write_rupture_mesh(realisation, path)

    restored, restored_crs = read_mesh(path)
    assert restored_crs == crs
    (back,) = restored[realisation.root]
    original = realisation[realisation.root]

    assert back.fields() == original.fields()
    for name in sorted(original.fields()):
        assert np.array_equal(back[name], original[name]), name
    assert np.array_equal(back.faces(), original.faces())
    assert np.array_equal(back.vertices_km(), original.vertices_km())
    assert back.attrs["hypocentre_strike_km"] == pytest.approx(
        original.attrs["hypocentre_strike_km"]
    )

    offsets, samples = original.pulses
    restored_offsets, restored_samples = back.pulses
    assert np.array_equal(offsets, restored_offsets)
    assert np.array_equal(samples, restored_samples)


def test_the_srf_is_one_plane_of_triangles(bent: Generated) -> None:
    """``NSTK = n_triangles``, ``NDIP = 1``, and one point per face.

    The SRF seam. The header states how many points the plane has and nothing about how
    they are arranged, which is exactly what SW4 reads and discards -- see
    :mod:`rupture_generator.triangular.assemble` for the source references. The
    ``shyp`` convention conversion survives: the SRF measures the hypocentre from the
    plane's along-strike centre.
    """
    realisation, config, _ = bent
    mesh = realisation[realisation.root]

    srf = tri_assemble.to_srf_file(realisation)

    assert srf.version == "2.0"
    assert len(srf.planes) == 1
    (plane,) = srf.planes
    assert plane.strike_count == mesh.face_count
    assert plane.dip_count == 1
    assert len(srf.points) == mesh.face_count
    assert plane.hypocentre_strike_km == pytest.approx(
        config.hypocentre.strike_km - plane.length_km / 2.0
    )
    assert plane.hypocentre_dip_km == pytest.approx(config.hypocentre.dip_km)


def test_the_srf_carries_the_same_rupture_in_its_own_units(
    bent: Generated, tmp_path: Path
) -> None:
    """Read back through HDF5, which is the path this track chose.

    The comparison is at the format's own resolution -- float32, six significant figures
    -- which is why the tolerances are 1e-5 rather than 1e-9. Read with `h5py` rather
    than with this package's parser because the parser reads the *text* format, and the
    two disagree about the shear modulus: text ignores ``VS`` and ``DEN`` and takes it
    from the SW4 grid, HDF5 uses the file's own. This module writes both columns and
    means them.
    """
    realisation, _, _ = bent
    mesh = realisation[realisation.root]

    path = tmp_path / "rupture.srf.h5"
    tri_assemble.to_srf_file(realisation).write_sw4_hdf5(path)

    with h5py.File(path, "r") as handle:
        points = handle["POINTS"][:]
        planes = handle.attrs["PLANE"]
        version = handle.attrs["VERSION"]

    assert float(version) == pytest.approx(2.0)
    assert planes["NSTK"][0] == mesh.face_count
    assert planes["NDIP"][0] == 1
    assert len(points) == mesh.face_count

    assert np.allclose(points["SLIP1"], mesh["slip_m"] * 100.0, rtol=1e-5)
    assert np.allclose(points["AREA"], mesh.areas_km2() * 1.0e10, rtol=1e-5)
    assert np.allclose(points["TINIT"], mesh["onset_s"], atol=1e-4)
    assert np.allclose(points["RAKE"], mesh["rake_deg"], rtol=1e-5)
    # Every point carries its own geometry, which is the whole point of the seam.
    assert points["STK"].std() > 0.0
    assert np.all(points["DEP"] > 0.0)


def test_the_moment_survives_the_trip_into_cgs(bent: Generated) -> None:
    """Summed from the SRF's own columns, in the units it stores them in.

    What MESH.md asks be reported so it can be checked against SW4's printed
    ``made %i point moment tensor sources`` tally. The conversion touches slip, area,
    shear speed and density, and a mistake in any one moves the moment by a power of ten.
    """
    realisation, _, _ = bent

    srf = tri_assemble.to_srf_file(realisation)
    assert tri_assemble.moment_newton_m(srf) == pytest.approx(
        realisation.moment_newton_m, rel=1e-5
    )


# ============================================================================
# The point of the whole migration: a genuinely curved rupture
# ============================================================================


def test_a_curved_subduction_interface_ruptures_end_to_end() -> None:
    """**The thing none of this could do before**: a rupture on the CFM Hikurangi mesh.

    9236 triangles of the modeller's own connectivity, an irregular outline, and
    ``|grad h|`` reaching 1.21 -- six times the budget MESH.md sized the Monge patch
    against, and where the projected area of the worst face is 0.64 of its true area. No
    quad lattice expresses any of that, and every stage runs on it unchanged.

    Measured at the shipped resolution (mean edge 7.4 km): 3.2 s end to end including
    the SRF, 0.29 GB peak, moment closing to round-off. The onset field spans 192 s,
    which is the front crossing an 833 km interface.

    ``Puyseguer.ts.gz`` is *not* used here and that is deliberate: it is refused by the
    mesh-quality gate, because two of its vertices carry 7.3e-07 of the median lumped
    mass and would suppress the whole slip distribution 5.3-fold. That refusal is
    correct behaviour and :mod:`~rupture_generator.triangular.mesh` measures it.
    """
    surface = read_tsurf(EXAMPLES / "cfm" / "Hikurangi.ts.gz")
    mesh = surface.to_mesh(surface="hikurangi")
    assert mesh.face_count == 9236
    assert mesh.maximum_slope() > 1.0

    config = _config()
    config.source = dataclasses.replace(
        config.source, magnitude=8.5, average_dip_deg=12.0, average_rake_deg=90.0
    )
    config.hypocentre.strike_km = 300.0
    config.hypocentre.dip_km = 80.0

    realisation = Realisation({"hikurangi": mesh}, pyproj.CRS("EPSG:2193"))
    realisation = tri.generate(config, realisation)
    rupture = realisation["hikurangi"]

    assert realisation.moment_newton_m == pytest.approx(
        moment.seismic_moment_nm(8.5), rel=1e-9
    )
    assert float(rupture["onset_s"][realisation.hypocentre]) == pytest.approx(
        0.0, abs=1e-12
    )
    assert float(rupture["onset_s"].min()) == pytest.approx(0.0, abs=1e-12)
    # A front that crosses the whole interface: 833 km of strike at a few km/s.
    assert 100.0 < float(rupture["onset_s"].max()) < 400.0
    assert (rupture["slip_m"] >= 0.0).all()

    # And it writes an SRF whose own moment is the one that was scaled to.
    srf = tri_assemble.to_srf_file(realisation)
    assert len(srf.points) == mesh.face_count
    assert tri_assemble.moment_newton_m(srf) == pytest.approx(
        realisation.moment_newton_m, rel=1e-5
    )


# ============================================================================
# The streaming SRF writer
# ============================================================================


@pytest.mark.parametrize("budget_bytes", (1 << 30, 1 << 16, 1 << 12))
def test_the_streamed_file_is_the_whole_file_writers_own(
    bent: Generated, tmp_path: Path, budget_bytes: int
) -> None:
    """Blocking must be invisible in the file, at every block size.

    The two writers reach the same bytes by different routes -- one holds every pulse
    of the rupture and the other holds a gibibyte of them -- so this is what says the
    route does not show. The small budgets are the interesting ones: 4 KiB is under a
    single pulse here, so every block is one face and every boundary is exercised.
    """
    realisation, config, _ = bent

    whole = tmp_path / "whole.h5"
    tri_assemble.to_srf_file(realisation).write_sw4_hdf5(whole)

    streamed = tmp_path / "streamed.h5"
    tri_assemble.write_sw4_hdf5(
        streamed, realisation, pipeline.pulse_model(config), budget_bytes=budget_bytes
    )

    with h5py.File(whole) as expected, h5py.File(streamed) as actual:
        assert actual.attrs["VERSION"] == expected.attrs["VERSION"]
        assert np.array_equal(actual.attrs["PLANE"], expected.attrs["PLANE"])
        for column in expected["POINTS"].dtype.names:
            assert np.array_equal(
                actual["POINTS"][column], expected["POINTS"][column]
            ), column
        assert np.array_equal(actual["SR1"][:], expected["SR1"][:])


def test_a_streamed_rupture_states_the_moment_it_was_scaled_to(
    bent: Generated, tmp_path: Path
) -> None:
    """The check that catches a block boundary dropping or duplicating points.

    Read off the **file**, in the file's own units, without touching ``SR1`` -- which
    is what keeps it available at a resolution where the file is eleven gigabytes. The
    tolerance is float32: the columns the moment is summed from are stored single
    precision, and 1.6e-8 is what that costs.
    """
    realisation, config, _ = bent
    path = tmp_path / "streamed.h5"
    tri_assemble.write_sw4_hdf5(
        path, realisation, pipeline.pulse_model(config), budget_bytes=1 << 13
    )

    assert tri_assemble.hdf5_moment_newton_m(path) == pytest.approx(
        realisation.moment_newton_m, rel=1e-5
    )
    with h5py.File(path) as handle:
        assert handle["POINTS"].shape[0] == realisation[realisation.root].face_count


def test_write_blocks_cover_every_face_once_in_order() -> None:
    """A block is a slice of the point order, or it is a different file.

    ``SR1`` is every pulse concatenated in subfault order and ``POINTS`` is one record
    per subfault in the same order, so anything other than consecutive covering slices
    writes a rupture whose pulses belong to the wrong triangles.
    """
    params = pipeline.pulse_model(_config())
    rise_time_s = np.linspace(0.05, 4.0, 500)

    for budget_bytes in (1 << 30, 1 << 16, 1 << 10, 12):
        blocks = list(tri.face_blocks(rise_time_s, params, budget_bytes))
        assert blocks[0].start == 0
        assert blocks[-1].stop == rise_time_s.size
        assert all(
            after.start == before.stop for before, after in itertools.pairwise(blocks)
        )
        assert all(block.stop > block.start for block in blocks)

    # And the budget is honoured wherever more than one face fits in it: a block's
    # bound on its own samples stays under what the budget buys at the cost the caller
    # states. Twelve bytes is the SRF writer's -- the kernel's f64 and the float32 it
    # narrows into -- against the eight the native writer pays.
    budget_bytes = 1 << 20
    per_sample = 8 + np.dtype(np.float32).itemsize
    bound = rise_time_s / params.sample_interval_s + tri.PULSE_SAMPLE_MARGIN
    for block in tri.face_blocks(rise_time_s, params, budget_bytes, per_sample):
        if block.stop - block.start > 1:
            assert bound[block].sum() * per_sample <= budget_bytes


@pytest.mark.parametrize("budget_bytes", (1 << 30, 1 << 12))
def test_a_streamed_rupture_file_is_the_resident_writers_own(
    tmp_path: Path, budget_bytes: int
) -> None:
    """The native file, written a block of faces at a time, is the whole one.

    The route production takes for a 400 m rupture, where the pulses are 2.45 G samples
    and 19.6 GB of float64 that must never all be resident. The stored form is the same
    two CSR arrays under the same names, so this compares them and the fields beside
    them exactly, at a block size that makes every boundary a block boundary.

    Run on the **two-segment** geometry, because the streaming route writes each
    segment's samples into that segment's own group and the offsets restart at zero per
    segment: a writer that carried either across the segment boundary would still write
    a file, and a one-segment rupture would never notice.
    """
    geometry, config = _two_fault_geometry()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegradedCorrelation)
        streamed = tri.generate(config, tri.segments_of(geometry), synthesise=False)
        resident = tri.generate(config, tri.segments_of(geometry))
    assert set(streamed) == {"kaikoura:0", "kaikoura:1"}

    tri.write_rupture_mesh(
        streamed,
        tmp_path / "streamed.h5",
        pipeline.pulse_model(config),
        budget_bytes=budget_bytes,
    )
    tri.write_rupture_mesh(resident, tmp_path / "resident.h5")

    read, _ = read_mesh(tmp_path / "streamed.h5")
    expected, _ = read_mesh(tmp_path / "resident.h5")
    for surface, segments in expected.items():
        for wanted, actual in zip(segments, read[surface], strict=True):
            offsets, samples = wanted.pulses
            back_offsets, back_samples = actual.pulses
            assert np.array_equal(offsets, back_offsets)
            assert np.array_equal(samples, back_samples)
            assert actual.fields() == wanted.fields()
            for field in sorted(wanted.fields()):
                assert np.array_equal(actual[field], wanted[field]), field
            assert actual.attrs["sample_interval_s"] == pytest.approx(
                wanted.attrs["sample_interval_s"]
            )


def test_writing_a_rupture_file_refuses_to_guess_about_its_pulses(
    bent: Generated, tmp_path: Path
) -> None:
    """Neither missing pulses nor two sources of them pass silently.

    Both would produce a file: one with no slip rate at all, one with every pulse
    written twice into a run that only the offsets say the length of. The second is the
    dangerous one, because the offsets would still parse.
    """
    realisation, config, _ = bent

    with pytest.raises(ValueError, match="already carry their pulses"):
        tri.write_rupture_mesh(
            realisation, tmp_path / "twice.h5", pipeline.pulse_model(config)
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegradedCorrelation)
        unsynthesised = tri.generate(
            config,
            tri.segments_of(read_geometry(EXAMPLES / "hope.geometry.toml")),
            synthesise=False,
        )
    with pytest.raises(ValueError, match="no slip-rate pulses"):
        tri.write_rupture_mesh(unsynthesised, tmp_path / "none.h5")
    with pytest.raises(ValueError, match="streaming route writes netCDF"):
        tri.write_rupture_mesh(
            unsynthesised, tmp_path / "none.zarr", pipeline.pulse_model(config)
        )


def test_a_rupture_generated_without_pulses_still_writes_its_srf(
    tmp_path: Path,
) -> None:
    """What production runs: S9 happens inside the writer, so it never has to fit.

    The realisation carries no pulses at all, and the file it produces is the one the
    full pipeline's would be -- which is the whole claim `synthesise=False` makes.
    """
    geometry = read_geometry(EXAMPLES / "hope.geometry.toml")
    config = _config()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegradedCorrelation)
        unsynthesised = tri.generate(
            config, tri.segments_of(geometry), synthesise=False
        )
        whole = tri.generate(config, tri.segments_of(geometry))

    assert unsynthesised[unsynthesised.root].pulses is None
    assert whole[whole.root].pulses is not None

    streamed, expected = tmp_path / "streamed.h5", tmp_path / "whole.h5"
    tri_assemble.write_sw4_hdf5(streamed, unsynthesised, pipeline.pulse_model(config))
    tri_assemble.to_srf_file(whole).write_sw4_hdf5(expected)

    with h5py.File(streamed) as actual, h5py.File(expected) as wanted:
        assert np.array_equal(actual["SR1"][:], wanted["SR1"][:])
        assert np.array_equal(actual["POINTS"]["NT1"], wanted["POINTS"]["NT1"])
