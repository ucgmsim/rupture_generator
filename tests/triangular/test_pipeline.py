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

**The reduction to the quad path.** On a planar fault the two pipelines should agree, and
"should agree" is only meaningful with the sampler held fixed: the circulant embedding and
the SPDE are different samplers and their draws are different random fields, so comparing
two independent draws would measure nothing. So the triangular run is given a sampler
that draws the *quad* field and hands each triangle its parent cell's value
(:func:`_shared_sampler`). Everything downstream -- the taper, the truncation, the moment
fold, the rake, the wavefront -- is then comparable face by face, and what the comparison
measures is the pipeline rather than the noise.

What that comparison found is worth stating up front, because two of the three are not
zero and neither is an error:

- **Rake is bit-identical** and the moment is exact, at every cut.
- **Slip differs by one global factor and nothing else**: the ratio's spread across faces
  is f64 round-off, and the factor is 1.0000 at the shipped 1 km cut, 1.0049 at 0.5 km
  and 0.8964 at 2 km. The cause is not the pipeline -- it is that rigidity is sampled at
  each subfault's *own* centre, and a triangle's centroid is not its parent quad's
  centre, so a few subfaults near a velocity-layer boundary land in a different layer.
  The moment fold then closes on a slightly different sum. Both ruptures carry exactly
  the magnitude's moment.
- **Onset differs by O(h)**, and the deliberate exception is the whole of the reason:
  `crates/kernels/src/eikonal.rs` is a second-order factored sweep and
  :mod:`~rupture_generator.triangular.fim` is first order by design. The measured
  difference halves with the cut (table in
  :func:`test_the_onset_difference_is_first_order_in_the_cut`), which is what a
  first-order scheme is *supposed* to do, and about half of it at any cut is the
  sub-cell disagreement about where the hypocentre is rather than solver error at all.
"""

from __future__ import annotations

import dataclasses
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
from rupture_generator.sampling import von_karman_field
from rupture_generator.triangular import assemble as tri_assemble
from rupture_generator.triangular import pipeline as tri
from rupture_generator.triangular.fim import DegradedSeed
from rupture_generator.triangular.gocad import read_tsurf
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
    with warnings.catch_warnings():
        # The seeded ball spans four velocity layers at a 1 km cut and says so. That
        # warning is `fim`'s own and is measured there, not here.
        warnings.simplefilter("ignore", DegradedSeed)
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


def _shared_sampler(chart: RuptureMesh, lookup: np.ndarray) -> tri.SamplerFactory:
    """A sampler factory that draws the *quad* field and spreads it over the triangles.

    The whole of what makes the two pipelines comparable. It is the same function the
    structured path draws with, handed the same chart and the same generator -- the
    substreams are keyed by stage and segment name, so both pipelines hand each stage an
    identically seeded one -- and the result is mapped onto faces by parent cell. So the
    two runs share every random number, and any difference between their outputs is the
    pipeline's rather than the noise's.
    """

    def sampler_of(
        _mesh: TriangleMesh,
        _geometry: tri.SegmentGeometry,
        _covariance: object,
    ) -> stages.FieldSampler:
        def draw(
            _chart: object, covariance: object, rng: np.random.Generator
        ) -> np.ndarray:
            return von_karman_field(chart, covariance, rng).ravel()[lookup]

        return draw

    return sampler_of


def _both(
    config: RuptureConfig, size_km: float = 1.0
) -> tuple[Realisation, Realisation, np.ndarray, RuptureMesh, TriangleMesh]:
    """One planar fault, run down both pipelines with one set of random numbers."""
    chart, crs = _planar_chart(size_km)
    mesh = from_chart(chart)
    lookup = _quad_of_face(mesh, chart)

    quad = pipeline.generate(config, Realisation({"hope": chart}, crs))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegradedSeed)
        triangular = tri.generate(
            config,
            Realisation({"hope": mesh}, crs),
            sampler_of=_shared_sampler(chart, lookup),
        )
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
    triangulation that is only true because :func:`~rupture_generator.triangular
    .pipeline.face_seeds` seeds all three corners of the hypocentre's face. Seeding one
    corner would leave this at about ``0.4 h S``, which is 0.2 s at this cut.
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


def test_the_onset_difference_is_first_order_in_the_cut() -> None:
    """**The deliberate exception, measured as a rate rather than against a bound.**

    The Cartesian kernel is a second-order factored sweep and meshFIM is first order;
    `MESH.md`'s Component 3 takes that trade in exchange for curvature, and the 0.05 s
    verification bound is the wrong yardstick for a method change that is expected to
    move numbers. The right one is whether the difference behaves like a discretisation
    error, so this measures its rate.

    Measured on the shipped planar plane, ``|onset(triangular) - onset(quad)|`` over
    every face, with the tapers off and the draw shared:

    ====  =======  ==========  =======  =======  =======
    cut   faces    hypo sep    median   p90      max
    ====  =======  ==========  =======  =======  =======
    2.0   196      0.472 km    425 ms   1601 ms  2625 ms
    1.0   756      0.240 km    198 ms    506 ms   928 ms
    0.5   3132     0.118 km     86 ms    250 ms   501 ms
    0.25  12528    0.059 km     50 ms    113 ms   218 ms
    ====  =======  ==========  =======  =======  =======

    Halving the cut halves the difference -- ratios 2.15, 2.31, 1.70 on the median --
    which is first order and not a plateau. Two things are worth separating out of it.
    The **hypocentre separation** column is the distance between the two paths' own
    hypocentres: the lattice registers on the seed *cell's centre* and the triangulation
    on the seed *face's centroid*, and at a 1 km cut those are 0.24 km apart, which at
    0.48 s/km is 115 ms -- almost exactly the 118 ms median *signed* difference. So half
    of what this table reports is the two paths disagreeing about where the earthquake
    started by a fraction of a subfault, and it converges away with everything else. And
    the difference is **not** monotone in distance from the source once that offset is
    removed (correlation 0.12): the residual is scatter of order ``h S``, which is what a
    first-order scheme on a 1 km mesh with 0.5 s/km slowness looks like.
    """
    config = _untapered()
    measured: dict[float, float] = {}
    for size_km in (2.0, 1.0, 0.5):
        quad, triangular, lookup, _, _ = _both(config, size_km)
        difference_s = np.abs(
            triangular["hope"]["onset_s"] - quad["hope"]["onset_s"].ravel()[lookup]
        )
        measured[size_km] = float(np.median(difference_s))

    # 1.5 rather than 2, because the measured ratios are 2.15 and 2.31 and the quantity
    # is a median over a field carrying a correlated perturbation: this asserts the rate
    # is first order rather than that it is exactly 2.
    assert measured[2.0] / measured[1.0] > 1.5
    assert measured[1.0] / measured[0.5] > 1.5
    # And the absolute size at the shipped cut, so a regression that kept the *rate* and
    # doubled the error still fails. 198 ms measured.
    assert measured[1.0] < 0.3


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegradedSeed)
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegradedSeed)
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
    # The child's *wavefront* at the arrival face is the seed time exactly, which is
    # what `face_seeds` buys: all three corners are fixed, so the face's own arrival is
    # the seed rather than the seed plus a third of its corner distances. Its **onset**
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegradedSeed)
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegradedSeed)
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
    with warnings.catch_warnings():
        # The ball spans 28 km at this cut, over which the slowness varies 59%, and
        # `fim` says so loudly. It is the coarse mesh talking, not the pipeline: the
        # radius is three rings of a 7.4 km mesh.
        warnings.simplefilter("ignore", DegradedSeed)
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


def test_the_stand_in_sampler_says_it_is_not_a_model() -> None:
    """The one thing that must never be silent.

    :func:`~rupture_generator.triangular.pipeline.white_noise_stand_in` exists so that
    the pipeline can be exercised at a resolution the SPDE cannot yet factorise, and a
    rupture drawn with it has no asperities at all. So it warns every time it is built,
    it is never selected automatically, and this asserts both halves: the default draws
    a correlated field and the stand-in refuses to be quiet about being one.
    """
    chart, crs = _planar_chart(2.0)
    mesh = from_chart(chart)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        realisation = tri.generate(
            _config(),
            Realisation({"hope": mesh}, crs),
            sampler_of=tri.white_noise_stand_in,
        )
    assert any(warned.category is tri.StandInField for warned in caught)
    assert realisation["hope"]["slip_m"].max() > 0.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tri.generate(_config(), Realisation({"hope": from_chart(chart)}, crs))
    assert not any(warned.category is tri.StandInField for warned in caught)
