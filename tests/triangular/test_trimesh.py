"""Geometric properties of the triangulated Monge patch.

Every assertion here is a mathematical or physical claim about a segment, with the
reason for its tolerance written beside it. The measurements ``mesh.py``'s docstrings
quote are *made* here rather than transcribed: the numbers in
:func:`~rupture_generator.triangular.mesh.check_admissible`'s table come out of
:func:`test_the_shipped_geometry_is_admissible_and_this_is_by_how_much`, and the
docstring is the record of what it printed.

The tolerance vocabulary follows ``tests/test_mesh.py``: **1e-9 relative** is "exact",
six orders above the f64 round-off floor at fault scale (~3e-15 with offset
coordinates) and seven below the one-percent slip bound, so a failure at 1e-9 is a real
error and never arithmetic.

The real-world cases are the three NZ CFM v1.0 subduction interfaces in
``examples/cfm``. They are what the triangular track exists for -- irregular outlines
and 9236 triangles of their own connectivity, neither of which a quad lattice expresses
-- and they are the only fixtures on which the fold check is a genuine test rather than
a restatement of how `scipy.spatial.Delaunay` orients its output.

Memory: the largest thing built here is ``colombia`` at 185k nodes. ``alpine_hope`` is
cut at 0.25 km rather than at its shipped 0.1 km, which takes its largest segment from
555k nodes to 89k and the whole example from 2.1 million to 330k. That is licensed, not
convenient: the quantity it is measured for, ``|grad h|``, is the same to round-off at
either cut, which
:func:`test_the_slope_does_not_depend_on_how_finely_the_planes_were_cut` asserts.
"""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path

import numpy as np
import pyproj
import pytest

from rupture_generator.config.geometry import (
    Discretisation,
    FaultConfig,
    GeometryConfig,
    LonLat,
    PlaneConfig,
    PointConfig,
)
from rupture_generator.mesh import RuptureMesh, fuse
from rupture_generator.mesh import build_fault as structured_fault
from rupture_generator.triangular.gocad import read_tsurf
from rupture_generator.triangular.mesh import (
    BOUNDARY_LABELS,
    DEGENERATE_MASS_FRACTION,
    SCHEMA_VERSION,
    MongeFrame,
    TriangleMesh,
    build_fault,
    build_point,
    build_surface,
    check_admissible,
    fold_margin,
    from_chart,
    implied_axes,
    read_mesh,
    stated_axes,
    write_mesh,
)

EXACT = 1.0e-9
"""Relative tolerance for a quantity that is an identity in the projected frame."""

NZTM = pyproj.CRS("EPSG:2193")

EXAMPLES = Path("examples")

PLANAR_EXAMPLES = ("beavan", "kaikoura")
"""Shipped geometries whose every segment is a single config plane, so the Monge patch
must collapse onto the structured chart exactly."""


# ============================================================================
# Fixtures and helpers
# ============================================================================


def _geometry(name: str) -> GeometryConfig:
    path = EXAMPLES / f"{name}.geometry.toml"
    if not path.exists():
        pytest.skip(f"{path} is not shipped in this checkout")
    return GeometryConfig.from_toml(path.read_text())


def _faults(name: str) -> list[tuple[FaultConfig, pyproj.CRS]]:
    config = _geometry(name)
    return [
        (surface, config.crs)
        for surface in config.surfaces
        if isinstance(surface, FaultConfig)
    ]


def _cut(fault: FaultConfig, size_km: float) -> FaultConfig:
    """The same fault at a different subfault size."""
    return dataclasses.replace(
        fault,
        planes=[
            dataclasses.replace(
                plane, discretisation=Discretisation(subfault_size_km=size_km)
            )
            for plane in fault.planes
        ],
    )


def _straight(
    *,
    dip_deg: float = 60.0,
    strike_count: int = 12,
    dip_count: int = 8,
    bottom_depth_km: float = 12.0,
    dip_direction: str = "right",
    reversed_trace: bool = False,
) -> FaultConfig:
    """One plane, from a point due south-west to a point due north-east."""
    near = LonLat(longitude_deg=172.0, latitude_deg=-43.0)
    far = LonLat(longitude_deg=172.2, latitude_deg=-42.9)
    origin, end = (far, near) if reversed_trace else (near, far)
    return FaultConfig(
        name="straight",
        origin=origin,
        planes=[
            PlaneConfig(
                end=end,
                dip_deg=dip_deg,
                bottom_depth_km=bottom_depth_km,
                dip_direction=dip_direction,
                discretisation=Discretisation(
                    strike_count=strike_count, dip_count=dip_count
                ),
            )
        ],
    )


def _cylinder(curvature: float, cells: int = 80) -> TriangleMesh:
    """A vertical fault whose trace is the parabola ``north = c * east^2``.

    Genuinely curved -- a tilted *plane* is still planar, so the patch would collapse
    and ``h`` would vanish -- and its along-strike arc length is analytic, which is what
    makes it a reference rather than a second transcription of the implementation.
    """
    east = np.linspace(-10.0, 10.0, cells + 1)
    depth = np.linspace(0.0, 40.0, cells // 2 + 1)
    grid_east, grid_depth = np.meshgrid(east, depth, indexing="ij")
    nodes = np.stack(
        [grid_east, curvature * grid_east**2, grid_depth], axis=-1
    ).transpose(1, 0, 2)
    return TriangleMesh.from_patches(
        [nodes],
        strike_deg=90.0,
        dip_deg=90.0,
        origin_east_km=0.0,
        origin_north_km=0.0,
        surface="cylinder",
    )


def _parabola_arc_km(curvature: float, east_km: np.ndarray) -> np.ndarray:
    """Exact length of ``(x, c x^2)`` from ``x = -10`` to each ``x``."""
    if curvature == 0.0:
        return east_km + 10.0
    slope = 2.0 * curvature

    def primitive(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * np.sqrt(1.0 + (slope * x) ** 2) + np.arcsinh(slope * x) / (
            2.0 * slope
        )

    return primitive(east_km) - primitive(np.array(-10.0))


def _quad_of_face(mesh: TriangleMesh, chart: RuptureMesh) -> np.ndarray:
    """Which structured cell each face's parameter centroid falls in, flat.

    Only meaningful on a planar single-plane segment, where the parameter domain is the
    rectangle the chart's lattice divides evenly.
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


# ============================================================================
# The gate: on a planar fault the Monge patch collapses
# ============================================================================


@pytest.mark.parametrize("example", PLANAR_EXAMPLES)
def test_a_planar_segments_frame_is_the_configs_own_strike_and_dip(
    example: str,
) -> None:
    """The best-fit plane *is* the fault plane, so the frame *is* strike and dip.

    MESH.md says "that identity is a test"; this is it. The only arithmetic between
    the config's numbers and the frame is one SVD, so the agreement is at the SVD's own
    round-off.

    The bound is 1e-10 degrees rather than 1e-12 because that is the *quad* mesh's own
    resolution, not this one's: ``RuptureMesh.strike_dip_deg`` builds a normal per cell,
    and the cells of a single ``beavan`` plane disagree with each other by up to
    3.5e-12 degrees. The frame's own drift against their mean is far smaller -- measured
    at 5.7e-14 degrees of strike and 9.9e-14 of dip, against the one-degree rake bound
    an SRF can express.
    """
    worst_strike = 0.0
    worst_dip = 0.0
    for fault, crs in _faults(example):
        charts = fuse(structured_fault(fault, crs))
        meshes = build_fault(fault, crs)
        for chart, mesh, plane in zip(charts, meshes, fault.planes, strict=True):
            assert len(chart.blocks()) == 1, "this example should be single-plane"
            chart_strike, chart_dip = chart.strike_dip_deg()
            # A plane's cells all report one strike, so their mean is that strike.
            assert np.ptp(chart_strike) < 1.0e-10
            assert np.ptp(chart_dip) < 1.0e-10

            worst_strike = max(
                worst_strike, abs(mesh.frame.strike_deg - float(chart_strike.mean()))
            )
            worst_dip = max(worst_dip, abs(mesh.frame.dip_deg - plane.dip_deg))

    assert worst_strike < 1.0e-10, f"strike drifted {worst_strike} degrees"
    assert worst_dip < 1.0e-10, f"dip drifted {worst_dip} degrees"


@pytest.mark.parametrize("example", PLANAR_EXAMPLES)
def test_a_planar_segment_reproduces_the_quad_meshs_areas(example: str) -> None:
    """Two triangles per quad, and their areas sum to the quad's exactly.

    ``RuptureMesh.areas_km2`` is already a two-triangle cross-product formula, so this
    is an identity rather than an approximation -- which is why the bound is
    :data:`EXACT` rather than something geometric.
    """
    for fault, crs in _faults(example):
        charts = fuse(structured_fault(fault, crs))
        for chart, mesh in zip(charts, build_fault(fault, crs), strict=True):
            of_quad = _quad_of_face(mesh, chart)
            assert set(np.bincount(of_quad).tolist()) == {2}

            per_quad = np.bincount(
                of_quad, weights=mesh.areas_km2(), minlength=chart.areas_km2().size
            ).reshape(chart.cell_counts)
            assert per_quad == pytest.approx(chart.areas_km2(), rel=EXACT)
            assert mesh.areas_km2().sum() == pytest.approx(
                chart.areas_km2().sum(), rel=EXACT
            )


@pytest.mark.parametrize("example", PLANAR_EXAMPLES)
def test_a_planar_segment_reproduces_the_quad_meshs_centres(example: str) -> None:
    """The area-weighted mean of a quad's two triangle centroids is its centre.

    True because every cell a planar chart builds is a *parallelogram*, where the mean
    of the four corners and the centroid of the lamina are the same point. Measured
    worst disagreement across ``beavan`` and ``kaikoura``: 5.3e-15 km, which is f64
    round-off at fault scale.
    """
    for fault, crs in _faults(example):
        charts = fuse(structured_fault(fault, crs))
        for chart, mesh in zip(charts, build_fault(fault, crs), strict=True):
            of_quad = _quad_of_face(mesh, chart)
            cells = chart.areas_km2().size
            area = mesh.areas_km2()
            centres = mesh.centres()
            weight = np.bincount(of_quad, weights=area, minlength=cells)
            recovered = np.stack(
                [
                    np.bincount(
                        of_quad, weights=area * centres[:, axis], minlength=cells
                    )
                    / weight
                    for axis in range(3)
                ],
                axis=-1,
            ).reshape(*chart.cell_counts, 3)
            assert recovered == pytest.approx(chart.centres(), abs=1.0e-12)


@pytest.mark.parametrize("example", PLANAR_EXAMPLES)
def test_a_planar_segments_arc_lengths_are_its_parameter_coordinates(
    example: str,
) -> None:
    """With ``h`` identically zero the metric factor is one, so ``S(u) = u``.

    The half of the arc-length pair that can be checked without a curved reference:
    if these two ever drift on a *flat* fault, the metric factor is being computed from
    something other than the slope.
    """
    for fault, crs in _faults(example):
        for mesh in build_fault(fault, crs):
            parameters = mesh.parameters_km()
            assert mesh.strike_arc_km() == pytest.approx(parameters[:, 0], abs=1.0e-12)
            assert mesh.dip_arc_km() == pytest.approx(parameters[:, 1], abs=1.0e-12)


# ============================================================================
# The frame splits its two jobs
# ============================================================================


def test_the_frame_is_orthonormal_and_right_handed() -> None:
    """``e_u x e_v = n``, which is what makes a positive parameter area mean no fold."""
    for dip_deg, strike_deg, dips_left in itertools.product(
        (5.0, 30.0, 60.0, 89.0, 90.0), (0.0, 73.0, 180.0, 301.0), (False, True)
    ):
        strike, down_dip = stated_axes(strike_deg, dip_deg, dips_left=dips_left)
        assert float(strike @ down_dip) == pytest.approx(0.0, abs=1.0e-15)

        fault = _straight(dip_deg=dip_deg)
        frame = build_fault(fault, NZTM)[0].frame
        basis = np.stack([frame.strike_axis, frame.dip_axis, frame.normal])
        assert basis @ basis.T == pytest.approx(np.eye(3), abs=1.0e-14)
        assert np.cross(frame.strike_axis, frame.dip_axis) == pytest.approx(
            frame.normal, abs=1.0e-14
        )


def test_the_in_plane_axes_do_not_come_from_the_svd() -> None:
    """On a square patch the SVD's in-plane axes are 90 degrees from strike and dip.

    The measurement behind the whole frame decision, and the reason MESH.md refuses to
    take ``e_u`` from the SVD. Cut a plane 12 by 12 with a width chosen to match its
    length: the point cloud's leading in-plane singular vector lands on the **dip**
    direction rather than strike -- bearing 89.32 against a config strike of 359.32 --
    because at aspect 0.999 which of the two the SVD picks is decided by round-off.
    Take strike from the config and it is exactly right whatever the aspect.
    """
    # A due-north trace 11.106 km long, dipping 60 into a width of 11.119 km, cut 12
    # by 12: aspect 0.9988, where the two in-plane singular values are a coin toss.
    square = FaultConfig(
        name="square",
        origin=LonLat(longitude_deg=172.0, latitude_deg=-43.0),
        planes=[
            PlaneConfig(
                end=LonLat(longitude_deg=172.0, latitude_deg=-42.9),
                dip_deg=60.0,
                bottom_depth_km=9.6293,
                discretisation=Discretisation(strike_count=12, dip_count=12),
            )
        ],
    )
    mesh = build_fault(square, NZTM)[0]
    chart = fuse(structured_fault(square, NZTM))[0]

    extent = mesh.parameters_km().max(axis=0)
    assert abs(extent[0] / extent[1] - 1.0) < 0.01, "this patch has to be near-square"

    vertices = mesh.vertices_km()
    _, _, rotation = np.linalg.svd(vertices - vertices.mean(axis=0))
    leading_bearing = float(
        np.degrees(np.arctan2(rotation[0][0], rotation[0][1])) % 360.0
    )
    config_strike = float(chart.strike_dip_deg()[0].ravel()[0])

    # The SVD's leading in-plane axis is the *dip* direction, a quarter turn away.
    assert abs((leading_bearing - config_strike - 90.0 + 180.0) % 360.0 - 180.0) < 1.0
    # The frame ignores it and reports the config's strike.
    assert mesh.frame.strike_deg == pytest.approx(config_strike, abs=1.0e-11)


def test_a_strike_perpendicular_to_the_geometry_is_refused() -> None:
    """A stated strike lying along the fitted normal names no direction in the patch.

    The geometry is a vertical patch striking due east, so its normal points north; a
    stated strike of due north is exactly along it. What survives projection is nothing,
    and a frame built on nothing would report a strike chosen by round-off.
    """
    corners = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, 8.0], [0.0, 0.0, 8.0]]
    )
    with pytest.raises(ValueError, match="names no direction in the patch"):
        MongeFrame.fit(corners, strike_deg=0.0, dip_deg=90.0)

    # A degenerate cloud is refused before any of that.
    with pytest.raises(ValueError, match="at least 3 points"):
        MongeFrame.fit(corners[:2], strike_deg=90.0, dip_deg=90.0)


# ============================================================================
# The strike sign, which stops being subtle once the frame is fixed
# ============================================================================


def test_reversing_the_trace_reverses_the_strike() -> None:
    """The same surface walked the other way reports the opposite strike.

    A reversed-strike SRF is plausible-looking and physically backwards, so this gets
    its own test. The two configs describe the *same* plane -- reversing the trace and
    swapping the dip side leaves the geometry alone -- and the only thing that differs
    is which way the config says the trace runs. Both the frame and every face have to
    follow the config rather than the geometry, because a triangle has no trace
    direction of its own to fall back on.
    """
    forward = build_fault(_straight(dip_direction="right"), NZTM)[0]
    backward = build_fault(_straight(dip_direction="left", reversed_trace=True), NZTM)[
        0
    ]

    # The same surface: same area, and the same vertices once each mesh's own origin
    # is added back, since the two configs put the origin at opposite ends of the trace.
    assert backward.areas_km2().sum() == pytest.approx(
        forward.areas_km2().sum(), rel=EXACT
    )

    def absolute(mesh: TriangleMesh) -> np.ndarray:
        east_km, north_km = mesh.origin_km
        return np.sort(mesh.vertices_km() + np.array([east_km, north_km, 0.0]), axis=0)

    assert absolute(backward) == pytest.approx(absolute(forward), abs=1.0e-9)

    turned = (backward.frame.strike_deg - forward.frame.strike_deg) % 360.0
    assert turned == pytest.approx(180.0, abs=1.0e-9)
    assert backward.frame.dip_deg == pytest.approx(forward.frame.dip_deg, abs=1.0e-11)

    forward_strike, forward_dip = forward.strike_dip_deg()
    backward_strike, backward_dip = backward.strike_dip_deg()
    assert np.ptp(forward_strike) == pytest.approx(0.0, abs=1.0e-11)
    assert (backward_strike - forward_strike) % 360.0 == pytest.approx(
        np.full(backward.face_count, 180.0), abs=1.0e-9
    )
    # Dip is a property of the plane, not of which way you walk it.
    assert backward_dip == pytest.approx(forward_dip, abs=1.0e-11)


def test_every_face_of_a_planar_segment_reports_the_frames_strike_and_dip() -> None:
    """The face-level strike comes from ``dX/du``, which is the frame's axis."""
    for dip_deg in (20.0, 55.0, 90.0):
        for dip_direction in ("left", "right"):
            mesh = build_fault(
                _straight(dip_deg=dip_deg, dip_direction=dip_direction), NZTM
            )[0]
            strike_deg, face_dip_deg = mesh.strike_dip_deg()
            assert strike_deg == pytest.approx(
                np.full(mesh.face_count, mesh.frame.strike_deg), abs=1.0e-9
            )
            assert face_dip_deg == pytest.approx(
                np.full(mesh.face_count, dip_deg), abs=1.0e-9
            )


def test_a_point_source_reports_the_strike_and_dip_it_was_given() -> None:
    """A point source is the only config that states a strike outright.

    Which makes it the cleanest test of the frame there is: no trace bearing to derive,
    no fusion, and the answer is the number in the file.
    """
    for strike_deg, dip_deg in itertools.product(
        (0.0, 45.0, 137.0, 359.0), (15.0, 60.0, 90.0)
    ):
        point = PointConfig(
            name="point",
            centre=LonLat(longitude_deg=172.0, latitude_deg=-43.0),
            depth_km=8.0,
            strike_deg=strike_deg,
            dip_deg=dip_deg,
            size_km=2.0,
        )
        mesh = build_point(point, NZTM)[0]
        assert mesh.face_count == 2
        # Bearings wrap, so compare the turn between them rather than the numbers.
        assert (mesh.frame.strike_deg - strike_deg + 180.0) % 360.0 - 180.0 == (
            pytest.approx(0.0, abs=1.0e-9)
        )
        # arcsin is ill-conditioned at 90 degrees: an error eps in the axis' vertical
        # component becomes sqrt(2 eps) in the angle, so f64 buys 8e-7 degrees there
        # and 1e-9 everywhere else. The axis itself is well conditioned either way.
        assert mesh.frame.dip_deg == pytest.approx(dip_deg, abs=1.0e-6)
        _, stated_down_dip = stated_axes(strike_deg, dip_deg)
        assert mesh.frame.dip_axis == pytest.approx(stated_down_dip, abs=1.0e-12)
        assert mesh.areas_km2().sum() == pytest.approx(4.0, rel=EXACT)
        assert build_surface(point, NZTM)[0].face_count == 2


# ============================================================================
# Admissibility, and the measurement its docstring records
# ============================================================================


def test_the_shipped_geometry_is_admissible_and_this_is_by_how_much() -> None:
    """Measure what ``check_admissible``'s docstring records. Printed, not just asserted.

    ``alpine_hope`` is cut at 0.25 km rather than its shipped 0.1: the shipped cut is
    2.1 million nodes and about 3 GB of triangulation, and ``|grad h|`` is identical at
    either (see the next test). Run with ``-s`` to read the table.
    """
    expected = {
        # example: (worst |grad h|, worst fold margin)
        "beavan": (5.0e-13, 1.0),
        "kaikoura": (5.0e-13, 1.0),
        "hope": (1.7746e-01, 0.984),
        "colombia": (5.0e-13, 1.0),
        "alpine_hope": (6.3064e-01, 0.5366),
    }
    for example, (want_slope, want_margin) in expected.items():
        worst_slope = 0.0
        worst_margin = 1.0
        for fault, crs in _faults(example):
            cut = _cut(fault, 0.25) if example == "alpine_hope" else fault
            for index, mesh in enumerate(build_fault(cut, crs)):
                check_admissible(mesh)
                slope = mesh.maximum_slope()
                margin = fold_margin(mesh)
                assert margin > 0.0, f"{fault.name} segment {index} folds"
                print(
                    f"{example:12s} {fault.name!r:36s} seg {index} "
                    f"planes={len(np.unique(mesh.planes())):3d} "
                    f"|grad h|={slope:.4e} margin={margin:.4f} "
                    f"V={mesh.node_count} F={mesh.face_count}"
                )
                worst_slope = max(worst_slope, slope)
                worst_margin = min(worst_margin, margin)
        print(f"  -> {example}: |grad h|={worst_slope:.4e} margin={worst_margin:.4f}")
        assert worst_slope == pytest.approx(want_slope, rel=0.02, abs=5.0e-13)
        assert worst_margin == pytest.approx(want_margin, rel=0.01)


def test_the_slope_does_not_depend_on_how_finely_the_planes_were_cut() -> None:
    """``|grad h|`` is a property of the planes' orientations, not the discretisation.

    Which is what licenses measuring ``alpine_hope`` at a coarser cut than it ships
    with. Not merely close: every face's slope is a combination of the same planar
    patches whatever the lattice on them, so the worst of them is the same number to
    round-off. Measured across a factor of four in cell size: 8e-15 relative, which is
    f64 arithmetic and nothing else.
    """
    fault, crs = _faults("alpine_hope")[0]
    slopes = [
        build_fault(_cut(fault, size), crs)[0].maximum_slope() for size in (1.0, 0.25)
    ]
    assert slopes[0] == pytest.approx(slopes[1], rel=1.0e-12)


def test_the_parameter_domain_is_tiled_exactly_once() -> None:
    """The faces cover the patches' parameter footprints, and cover them once.

    This is injectivity stated as an area. Delaunay triangulates the convex hull, so
    the faces that survive culling cover the *union* of the config planes' parameter
    quads; if two planes overlapped in the parameter plane -- which is what a fold is --
    the quads would sum to more than the faces do.
    """
    for example in ("kaikoura", "hope", "beavan"):
        for fault, crs in _faults(example):
            for chart, mesh in zip(
                fuse(structured_fault(fault, crs)),
                build_fault(fault, crs),
                strict=True,
            ):
                nodes = chart.nodes()
                frame = mesh.frame
                quads = 0.0
                for _plane, start, stop in chart.blocks():
                    patch = nodes[:, start : stop + 1]
                    corner = frame.project(
                        np.stack(
                            [patch[0, 0], patch[0, -1], patch[-1, -1], patch[-1, 0]]
                        )
                    )[:, :2]
                    quads += 0.5 * abs(
                        sum(
                            corner[k][0] * corner[(k + 1) % 4][1]
                            - corner[(k + 1) % 4][0] * corner[k][1]
                            for k in range(4)
                        )
                    )
                assert float(mesh.parameter_areas_km2().sum()) == pytest.approx(
                    quads, rel=1.0e-12
                )


def test_a_folded_patch_is_refused() -> None:
    """A lattice that doubles back is not a graph over any plane, and is caught.

    **This test could not be written while the builder used Delaunay.** A triangulation
    of the projected points is positively oriented by construction, so the fold had to
    be faked by corrupting the connectivity afterwards -- which tested the check against
    input the builder could not produce. Now the connectivity comes from the lattice,
    the fold reaches :func:`check_admissible` on its own, and the refusal is a real one.
    """
    east = np.array([0.0, 4.0, 8.0, 6.0, 10.0])
    depth = np.linspace(0.0, 6.0, 4)
    grid_east, grid_depth = np.meshgrid(east, depth, indexing="ij")
    folded = np.stack(
        [grid_east, np.zeros_like(grid_east), grid_depth], axis=-1
    ).transpose(1, 0, 2)

    with pytest.raises(ValueError, match="folded or collapsed"):
        TriangleMesh.from_patches(
            [folded],
            strike_deg=90.0,
            dip_deg=90.0,
            origin_east_km=0.0,
            origin_north_km=0.0,
            surface="doubled",
        )


def test_two_nodes_at_one_parameter_point_are_refused() -> None:
    """A projection that is not injective is not a chart.

    Two lattice columns at the same easting collapse their quads to zero parameter
    area, which is the degenerate half of the admissibility condition -- the
    parameterisation has stopped being one, rather than having turned over.
    """
    depth = np.linspace(0.0, 6.0, 4)
    east = np.array([0.0, 4.0, 4.0, 8.0])
    grid_east, grid_depth = np.meshgrid(east, depth, indexing="ij")
    doubled = np.stack(
        [grid_east, np.zeros_like(grid_east), grid_depth], axis=-1
    ).transpose(1, 0, 2)
    with pytest.raises(ValueError, match="folded or collapsed"):
        TriangleMesh.from_patches(
            [doubled],
            strike_deg=90.0,
            dip_deg=90.0,
            origin_east_km=0.0,
            origin_north_km=0.0,
            surface="doubled",
        )


# ============================================================================
# The curved lattice: what Delaunay of the projection got wrong
# ============================================================================


def _curved_interface(cells_strike: int, cells_dip: int) -> np.ndarray:
    """A trench curving in map view over a dip that steepens with depth.

    Curved in *both* parameters, which is what makes its parameter footprint concave --
    a cylinder curved in one is not enough, because its footprint stays a rectangle.
    Measured: the footprint is 6581.4 km2 against a convex hull of 7400.2, so it is
    concave by 12.4%, and Delaunay of the same points returns 38 more faces than the
    lattice has quads at 20x20 (62 at 40x24, 19 at 12x9).
    """
    along = np.linspace(0.0, 120.0, cells_strike + 1)
    down = np.linspace(0.0, 60.0, cells_dip + 1)
    grid_along, grid_down = np.meshgrid(along, down, indexing="ij")
    return np.stack(
        [
            grid_along + 0.10 * grid_down,
            0.0035 * grid_along**2 - 0.55 * grid_down - 0.0025 * grid_down**2,
            3.0
            + 0.22 * grid_down
            + 0.0035 * grid_down**2
            + 0.0009 * grid_along * grid_down,
        ],
        axis=-1,
    ).transpose(1, 0, 2)


@pytest.mark.parametrize(
    ("cells_strike", "cells_dip"), ((20, 20), (40, 24), (12, 9), (2, 2))
)
def test_a_curved_lattice_gets_exactly_two_triangles_per_quad(
    cells_strike: int, cells_dip: int
) -> None:
    """The regression. A quad lattice has ``2 * n_i * n_j`` triangles and no others.

    The defect this guards was found on the Williams et al. (2013) Hikurangi interface:
    its largest fully populated 21x21 block gave **866** faces where 800 is the only
    right answer, because `scipy.spatial.Delaunay` triangulates the convex *hull* of the
    projected points and a curved surface's footprint is concave. The 66 extra carried
    1997.4 km2 -- 6.6% of an area whose bound is exact, since moment's is -- and their
    slopes reached 18.35 against a true maximum of 0.196.

    Every shipped example passed throughout, because a planar fault's footprint is a
    convex quadrilateral and the hull is then the domain exactly. That is why the
    fixture here has to curve in both parameters.
    """
    mesh = TriangleMesh.from_patches(
        [_curved_interface(cells_strike, cells_dip)],
        strike_deg=90.0,
        dip_deg=20.0,
        origin_east_km=0.0,
        origin_north_km=0.0,
        surface="interface",
    )
    assert mesh.face_count == 2 * cells_strike * cells_dip
    assert len(np.unique(mesh.faces())) == mesh.node_count


def test_a_curved_lattice_has_a_concave_footprint_so_the_test_has_teeth() -> None:
    """Assert the fixture is actually the hard case, not accidentally convex.

    Without this, :func:`test_a_curved_lattice_gets_exactly_two_triangles_per_quad`
    could pass on a fixture that never exercised the bug.
    """
    from scipy.spatial import ConvexHull, Delaunay

    cells_strike, cells_dip = 20, 20
    mesh = TriangleMesh.from_patches(
        [_curved_interface(cells_strike, cells_dip)],
        strike_deg=90.0,
        dip_deg=20.0,
        origin_east_km=0.0,
        origin_north_km=0.0,
        surface="interface",
    )
    parameters = mesh.parameters_km()
    footprint = float(mesh.parameter_areas_km2().sum())
    hull = float(ConvexHull(parameters).volume)
    assert hull / footprint - 1.0 > 0.05, "the fixture has to be genuinely concave"
    assert len(Delaunay(parameters).simplices) > 2 * cells_strike * cells_dip


def test_a_curved_lattice_reproduces_the_quad_charts_area() -> None:
    """Two triangles per quad still sum to the quad, on a surface that is not planar.

    The planar version of this is an identity; here the quads are genuinely
    non-coplanar, so it says the diagonal split agrees with the one
    ``RuptureMesh.areas_km2`` uses -- which is the ``(0, 2)`` diagonal, the same one.
    """
    nodes = _curved_interface(24, 16)
    chart = RuptureMesh.from_nodes(
        nodes[..., 0],
        nodes[..., 1],
        nodes[..., 2],
        origin_east_km=0.0,
        origin_north_km=0.0,
        surface="interface",
    )
    mesh = from_chart(chart)
    assert mesh.face_count == 2 * 24 * 16
    assert mesh.areas_km2().sum() == pytest.approx(chart.areas_km2().sum(), rel=EXACT)

    of_quad = np.repeat(np.arange(chart.areas_km2().size), 2)
    per_quad = np.bincount(of_quad, weights=mesh.areas_km2()).reshape(chart.cell_counts)
    assert per_quad == pytest.approx(chart.areas_km2(), rel=EXACT)


# ============================================================================
# The real thing: NZ CFM v1.0 subduction interfaces
# ============================================================================

CFM = Path("examples/cfm")

CFM_INTERFACES = (
    # stem, V, F, area km2, best-fit dip, |grad h| median / p90 / max, accepted
    ("Hikurangi", 5218, 9236, 181069.0, 14.105, 0.158, 0.425, 1.2142, True),
    ("Puyseguer", 2597, 4090, 67921.5, 21.226, 0.101, 0.881, 2.1435, False),
    ("Puysegur_Fiordland", 2312, 4041, 77968.6, 22.772, 0.132, 0.771, 1.9688, True),
)
"""The three subduction interfaces, and what this container measures on them.

From the NZ Community Fault Model v1.0 GOCAD TSurf archive, unmodified and unclipped;
``Puyseguer`` is the CFM's own spelling. The last field is whether
:func:`~rupture_generator.triangular.mesh.check_admissible` accepts the mesh --
``Puyseguer`` carries two vertices with essentially no surface and is refused, which is
:func:`test_a_degenerate_real_mesh_is_refused_and_named`.

``|grad h|`` quantiles are area-weighted. They sit a little under the sampler agent's
independently measured 0.161 / 0.419 / 1.199 because the reference plane here is fitted
**area-weighted** rather than per-vertex, which on a mesh whose triangles vary this much
in size is a different plane; see :meth:`MongeFrame.fit`.
"""

ACCEPTED_INTERFACES = tuple(row for row in CFM_INTERFACES if row[-1])


def _tsurf(stem: str):
    path = CFM / f"{stem}.ts.gz"
    if not path.exists():
        pytest.skip(f"{path} is not staged in this checkout")
    return read_tsurf(path)


def _interface(stem: str) -> TriangleMesh:
    return _tsurf(stem).to_mesh()


@pytest.mark.parametrize("row", CFM_INTERFACES, ids=lambda row: row[0])
def test_a_cfm_interface_loads_with_its_own_connectivity(row: tuple) -> None:
    """The file's triangles are the surface's, and they arrive unchanged.

    Vertex and face counts against the archive. This reads the file rather than
    building the mesh, so it covers the one interface the mesh gate refuses too.
    """
    stem, vertices, faces = row[0], row[1], row[2]
    surface = _tsurf(stem)
    assert len(surface.vertices_km) == vertices
    assert sum(len(part) for part in surface.parts) == faces
    assert len(surface.parts) == 1
    assert surface.vertices_km[:, 2].min() > 0.0, "depth is positive down"


@pytest.mark.parametrize("row", ACCEPTED_INTERFACES, ids=lambda row: row[0])
def test_a_cfm_interface_is_one_monge_patch(row: tuple) -> None:
    """Zero inverted triangles, tested against the surface's own connectivity.

    **This is the assertion that could not be made before.** Orientation checked against
    a Delaunay triangulation of the projected points is a tautology: qhull orients every
    face positively by construction, so the test cannot fail whatever the surface does.
    Checked against the faces the CFM wrote, it is a real question -- and the answer is
    that all 9236 and 4041 triangles agree, so each interface genuinely is a graph over
    its own best-fit plane.

    The global winding is normalised first, and that is not a weakening: which way a
    modeller numbers a face says whether they looked at it from the hanging wall or the
    footwall. All three of these are wound opposite to the frame, and reading that as
    9236 folds would be reading a file convention as geology. What survives
    normalisation is whether the faces agree with *each other*, which is what
    injectivity means.
    """
    stem, _v, faces, area_km2, dip_deg, median, p90, maximum, _ok = row
    mesh = _interface(stem)

    check_admissible(mesh)
    signed_km2 = mesh.parameter_areas_km2()
    assert int((signed_km2 <= 0.0).sum()) == 0
    assert len(signed_km2) == faces
    assert fold_margin(mesh) > 0.0
    assert mesh.areas_km2().sum() == pytest.approx(area_km2, rel=1.0e-6)
    assert mesh.frame.dip_deg == pytest.approx(dip_deg, abs=1.0e-3)

    slope = np.linalg.norm(mesh.slope(), axis=-1)
    weight = mesh.areas_km2()
    order = np.argsort(slope)
    quantile = np.cumsum(weight[order]) / weight.sum()
    assert slope[order][np.searchsorted(quantile, 0.5)] == pytest.approx(
        median, abs=1.0e-3
    )
    assert slope[order][np.searchsorted(quantile, 0.9)] == pytest.approx(
        p90, abs=1.0e-3
    )
    assert slope.max() == pytest.approx(maximum, abs=1.0e-3)


def test_a_degenerate_real_mesh_is_refused_and_named() -> None:
    """``Puyseguer`` is a real file that must not load, and the message has to be usable.

    Two of its 2597 vertices carry 2.16e-05 and 1.19e-04 km2 against a median of 29.4 --
    ratios of 7.3e-07 and 4.0e-06. In the SPDE a vertex with no support is barely
    constrained, its marginal variance reaches 9.5e6, and ``sampling.standardise``
    divides the whole field by a standard deviation those two dominate: the healthy slip
    distribution comes out suppressed 5.3-fold, still looking exactly like a slip
    distribution. That is why this is refused rather than warned about.

    The other two interfaces have **no** vertex below the line, so this is a measured
    separation between real meshes rather than a threshold anyone chose.
    """
    surface = _tsurf("Puyseguer")
    with pytest.raises(ValueError) as raised:
        surface.to_mesh()

    message = str(raised.value)
    assert "2 of 2597 vertices carry almost no surface" in message
    assert "2.16e-05 km^2" in message
    assert "quality" in message
    assert "Remesh" in message
    # It is refused for degeneracy, not for folding: the geometry is fine.
    assert "folded" not in message


@pytest.mark.parametrize("row", ACCEPTED_INTERFACES, ids=lambda row: row[0])
def test_an_accepted_cfm_interface_has_no_starved_vertex(row: tuple) -> None:
    """The other side of the gate, on real data rather than a fixture."""
    mesh = _interface(row[0])
    mass_km2 = mesh.lumped_mass_km2()
    assert mass_km2.sum() == pytest.approx(mesh.areas_km2().sum(), rel=EXACT)
    ratio = mass_km2 / np.median(mass_km2)
    assert ratio.min() > DEGENERATE_MASS_FRACTION
    # Measured headroom: the worst accepted vertex sits at 2.4e-05, a factor of 2.4
    # above the line, which is the tightest margin in the module.
    assert ratio.min() == pytest.approx(2.4e-05, rel=0.05)


def test_a_lattice_mesh_is_four_orders_clear_of_the_degeneracy_gate() -> None:
    """The worst a lattice can do is a corner vertex, and that is exactly 1/6.

    Which is what makes the gate safe to apply to everything: no mesh this package
    builds can approach it.
    """
    for mesh in (
        build_fault(_straight(), NZTM)[0],
        TriangleMesh.from_patches(
            [_curved_interface(20, 20)],
            strike_deg=90.0,
            dip_deg=20.0,
            origin_east_km=0.0,
            origin_north_km=0.0,
            surface="interface",
        ),
    ):
        mass_km2 = mesh.lumped_mass_km2()
        assert (mass_km2 / np.median(mass_km2)).min() > 0.15


def test_the_real_interfaces_exceed_the_slope_mesh_md_sized_things_against() -> None:
    """MESH.md says ``|grad h| <~ 0.33``. Real subduction interfaces reach 1.2 to 2.1.

    Recorded because it is a plan-level number that the data contradicts, and because
    the Monge patch survives it: the metric factor ``sqrt(1 + |grad h|^2)`` reaches 2.21
    on the meshes that load, so parameter distance and true surface length differ by up
    to 121% *at a point* -- against the 5% MESH.md sized against -- and every one of
    these surfaces is still injective.

    The *aggregate* is far milder than the local maximum, which is the number that
    matters for the SRF header and the hypocentre seam: measured end to end, the strike
    extent exceeds its projection by 0.2 to 1.4% and the dip extent by 2.8 to 12.4%,
    because the steep places are localised and :meth:`arc_profile` weights by area.
    """
    worst = 0.0
    for stem, *_ in ACCEPTED_INTERFACES:
        mesh = _interface(stem)
        worst = max(worst, mesh.maximum_slope())

        strike_ratio = mesh.strike_arc_km().max() / mesh.parameters_km()[:, 0].max()
        dip_ratio = mesh.dip_arc_km().max() / mesh.parameters_km()[:, 1].max()
        assert 1.0 <= strike_ratio < 1.02
        assert 1.0 <= dip_ratio < 1.13

    assert worst > 1.0, "these are the surfaces that break the 0.33 budget"
    assert np.sqrt(1.0 + worst**2) == pytest.approx(2.21, abs=0.01)


def test_the_slope_is_read_off_the_face_normal() -> None:
    """``|grad h|`` is ``tan`` of the angle between the face normal and the frame's.

    Two independent routes to the same number, which is what makes this a test: the
    implementation projects the face normal onto the frame's axes, and this recomputes
    it from the angle between the two normals. They agree to round-off on a real
    interface with 9236 triangles of wildly varying shape.

    The formulation matters because the obvious alternative -- inverting the
    parameter-space edge matrix to get ``dX/d(u, v)`` -- is singular exactly when a face
    is a sliver in projection, and real meshes have those. On this interface the two
    routes happen to agree (1.2142 either way), so the sliver-fragility is not
    *demonstrated* here; it is avoided on principle, because the face-normal route has
    no matrix inverse in it at all and so cannot be conditioned badly by a thin
    triangle.
    """
    mesh = _interface("Hikurangi")
    assert mesh.maximum_slope() == pytest.approx(1.2142, abs=1.0e-3)

    angle = np.arccos(
        np.clip(np.abs(mesh.face_normals() @ mesh.frame.normal), 0.0, 1.0)
    )
    assert np.linalg.norm(mesh.slope(), axis=-1) == pytest.approx(
        np.tan(angle), rel=1.0e-9
    )


@pytest.mark.parametrize("row", ACCEPTED_INTERFACES, ids=lambda row: row[0])
def test_a_cfm_interface_has_a_closed_boundary_and_labelled_edges(row: tuple) -> None:
    """An irregular outline is the capability triangulation buys, so it has to close."""
    mesh = _interface(row[0])
    edges = mesh.boundary_edges()
    assert len(np.unique(edges[:, 0])) == len(edges)
    assert set(edges[:, 0].tolist()) == set(edges[:, 1].tolist())
    assert set(mesh.boundary_labels().tolist()) <= set(BOUNDARY_LABELS)
    # The top of a subduction interface is shallower than its bottom.
    depth_km = mesh.vertices_km()[:, 2]
    assert depth_km[mesh.boundary_edges("top")].mean() < (
        depth_km[mesh.boundary_edges("bottom")].mean()
    )


def test_a_cfm_interface_round_trips_through_the_file_format(tmp_path: Path) -> None:
    """A surface with its own connectivity is exactly what schema 3 exists to store."""
    mesh = _interface("Puysegur_Fiordland")
    path = tmp_path / "interface.h5"
    write_mesh({"Puysegur_Fiordland": [mesh]}, NZTM, path)
    back, _crs = read_mesh(path)
    other = back["Puysegur_Fiordland"][0]
    assert np.array_equal(other.faces(), mesh.faces())
    assert np.array_equal(other.vertices_km(), mesh.vertices_km())
    assert np.array_equal(other.parameters_km(), mesh.parameters_km())
    assert other.areas_km2().sum() == pytest.approx(mesh.areas_km2().sum(), rel=EXACT)


# ============================================================================
# The GOCAD reader, and a triangulation that arrives with its own faces
# ============================================================================


def test_a_tsurf_declaring_depth_rather_than_elevation_is_refused(
    tmp_path: Path,
) -> None:
    """Reading a Depth file as an Elevation one mirrors it through sea level silently."""
    path = tmp_path / "flipped.ts"
    path.write_text(
        "GOCAD TSurf 1\nZPOSITIVE Depth\nTFACE\n"
        "VRTX 1 0 0 -1000\nVRTX 2 1000 0 -1000\nVRTX 3 0 1000 -2000\nTRGL 1 2 3\nEND\n"
    )
    with pytest.raises(ValueError, match="ZPOSITIVE Depth"):
        read_tsurf(path)


def test_a_tsurf_with_no_triangles_is_refused(tmp_path: Path) -> None:
    """A point set is not a surface, and this reader will not invent connectivity."""
    path = tmp_path / "cloud.ts"
    path.write_text(
        "GOCAD TSurf 1\nZPOSITIVE Elevation\nTFACE\nVRTX 1 0 0 -1000\nEND\n"
    )
    with pytest.raises(ValueError, match="no TRGL records"):
        read_tsurf(path)


def test_gocad_vertex_indices_are_read_as_one_based(tmp_path: Path) -> None:
    """The one mistake this reader exists to make once instead of at every call site."""
    path = tmp_path / "one.ts"
    path.write_text(
        "GOCAD TSurf 1\nZPOSITIVE Elevation\nTFACE\n"
        "VRTX 1 0 0 -1000\nVRTX 2 10000 0 -1000\nVRTX 3 10000 10000 -5000\n"
        "VRTX 4 0 10000 -5000\nTRGL 1 2 3\nTRGL 1 3 4\nEND\n"
    )
    surface = read_tsurf(path)
    assert surface.vertices_km.shape == (4, 3)
    # Depth is -z, in kilometres.
    assert surface.vertices_km[:, 2] == pytest.approx([1.0, 1.0, 5.0, 5.0])
    assert surface.parts[0].min() == 0
    assert surface.parts[0].max() == 3


def test_from_triangulation_keeps_the_faces_it_was_given() -> None:
    """No retriangulation, no culling, no repair -- the faces are the input."""
    nodes = _curved_interface(8, 6)
    lattice = TriangleMesh.from_patches(
        [nodes],
        strike_deg=90.0,
        dip_deg=20.0,
        origin_east_km=0.0,
        origin_north_km=0.0,
        surface="interface",
    )
    rebuilt = TriangleMesh.from_triangulation(
        lattice.vertices_km(),
        lattice.faces(),
        strike_deg=90.0,
        dip_deg=20.0,
        surface="interface",
    )
    assert np.array_equal(rebuilt.faces(), lattice.faces())
    assert rebuilt.areas_km2().sum() == pytest.approx(
        lattice.areas_km2().sum(), rel=EXACT
    )


def test_from_triangulation_normalises_the_winding_but_not_a_fold() -> None:
    """Reversing every face is a convention; reversing one is a fold."""
    nodes = _curved_interface(8, 6)
    lattice = TriangleMesh.from_patches(
        [nodes],
        strike_deg=90.0,
        dip_deg=20.0,
        origin_east_km=0.0,
        origin_north_km=0.0,
        surface="interface",
    )
    faces = lattice.faces()

    flipped = TriangleMesh.from_triangulation(
        lattice.vertices_km(),
        faces[:, [0, 2, 1]],
        strike_deg=90.0,
        dip_deg=20.0,
        surface="interface",
    )
    assert (flipped.parameter_areas_km2() > 0.0).all()
    assert flipped.areas_km2().sum() == pytest.approx(
        lattice.areas_km2().sum(), rel=EXACT
    )

    one_reversed = faces.copy()
    one_reversed[5] = one_reversed[5][::-1]
    with pytest.raises(ValueError, match="folded or collapsed"):
        TriangleMesh.from_triangulation(
            lattice.vertices_km(),
            one_reversed,
            strike_deg=90.0,
            dip_deg=20.0,
            surface="interface",
        )


def test_from_triangulation_refuses_indices_it_cannot_use() -> None:
    """Including the one-based mistake, named so the caller knows which it made."""
    points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 5.0]])
    common = {"strike_deg": 90.0, "dip_deg": 30.0, "surface": "bad"}
    with pytest.raises(ValueError, match="GOCAD's are one-based"):
        TriangleMesh.from_triangulation(points, np.array([[1, 2, 3]]), **common)
    with pytest.raises(ValueError, match="they are triangles"):
        TriangleMesh.from_triangulation(points, np.zeros((0, 3), int), **common)
    with pytest.raises(ValueError, match="at least 3"):
        TriangleMesh.from_triangulation(points[:2], np.array([[0, 1, 0]]), **common)
    with pytest.raises(ValueError, match="non-finite"):
        TriangleMesh.from_triangulation(
            np.full((3, 3), np.nan), np.array([[0, 1, 2]]), **common
        )


def test_implied_axes_are_the_planes_own_strike_and_dip() -> None:
    """For data with no config: the geologist's definition, and a fixed point.

    Not the failure mode the module rejects -- that is taking the in-plane axes from the
    SVD's in-plane singular vectors. This uses only the fitted *normal* and the vertical,
    so it is exactly "which way does this plane dip", and feeding it back to
    :meth:`MongeFrame.fit` returns the same axes.
    """
    for strike_deg, dip_deg in itertools.product((0.0, 73.0, 200.0), (5.0, 40.0, 89.0)):
        mesh = build_fault(_straight(dip_deg=dip_deg), NZTM)[0]
        del strike_deg
        implied_strike, implied_dip, dips_left = implied_axes(mesh.vertices_km())
        assert dips_left is False
        assert implied_dip == pytest.approx(dip_deg, abs=1.0e-9)
        assert (implied_strike - mesh.frame.strike_deg + 180.0) % 360.0 - 180.0 == (
            pytest.approx(0.0, abs=1.0e-9)
        )

        # A fixed point: refitting through the implied axes gives them back.
        frame = MongeFrame.fit(
            mesh.vertices_km(),
            strike_deg=implied_strike,
            dip_deg=implied_dip,
            dips_left=dips_left,
        )
        assert frame.strike_deg == pytest.approx(implied_strike, abs=1.0e-9)
        assert frame.dip_deg == pytest.approx(implied_dip, abs=1.0e-9)


def test_implied_axes_refuse_a_horizontal_surface() -> None:
    """A horizontal plane has no strike, and guessing one would be inventing geology."""
    flat = np.array([[0.0, 0.0, 5.0], [10.0, 0.0, 5.0], [10.0, 10.0, 5.0]])
    with pytest.raises(ValueError, match="horizontal"):
        implied_axes(flat)


# ============================================================================
# Arc length: the derived half of the coordinate pair
# ============================================================================


@pytest.mark.parametrize("curvature", (0.005, 0.02, 0.05, 0.06, 0.10))
def test_the_arc_length_of_a_curved_patch_is_the_analytic_one(
    curvature: float,
) -> None:
    """Against a closed form, not against a second transcription of the code.

    A parabolic-cylinder fault has along-strike arc length
    ``int sqrt(1 + (2cx)^2) dx``, which integrates in closed form, and its dip lines are
    exactly vertical so the dip arc is exactly ``v``.

    Re-validated at the gradients the real CFM interfaces reach, which is where this had
    to be checked rather than assumed. Measured at 80 cells over 20 km, so a 0.25 km
    cell:

    ==========  ============  ==========
    ``|grad h|``  error, km   of a cell
    ==========  ============  ==========
    0.099       5.2e-06       0.002%
    0.395       7.7e-05       0.031%
    0.988       3.7e-04       0.147%
    1.185       4.8e-04       0.192%
    1.975       9.3e-04       0.373%
    ==========  ============  ==========

    It does **not** degrade: the error grows roughly linearly in ``|grad h|`` and stays
    three orders below a cell at Hikurangi's 1.199 and Puyseguer's 2.048. The one thing
    the cylinder cannot exercise is variation *across* dip -- its metric factor is
    constant along ``v``, so :meth:`arc_profile`'s area-weighted average over dip is
    exact there and only the rectangle-flattening error remains. On a real interface
    that average is the definition rather than an error, which is the whole reason
    :meth:`arc_profile` is separable.
    """
    mesh = _cylinder(curvature)
    east_km = mesh.vertices_km()[:, 0]
    error_km = np.abs(mesh.strike_arc_km() - _parabola_arc_km(curvature, east_km)).max()

    cell_km = 20.0 / 80.0
    assert error_km < 0.01 * cell_km, f"arc length off by {error_km} km"
    assert mesh.dip_arc_km() == pytest.approx(mesh.parameters_km()[:, 1], abs=1.0e-12)


def test_the_arc_length_converges_second_order_under_refinement() -> None:
    """The only approximation in the arc integral is flattening a face to a rectangle.

    That is a second-order error in the cell size, and saying so is only worth anything
    if it is measured: halving the cells quarters the error, 3.1e-04 -> 7.7e-05 ->
    1.9e-05 -> 4.8e-06 km at curvature 0.02.
    """
    curvature = 0.02
    errors = []
    for cells in (40, 80, 160, 320):
        mesh = _cylinder(curvature, cells)
        errors.append(
            float(
                np.abs(
                    mesh.strike_arc_km()
                    - _parabola_arc_km(curvature, mesh.vertices_km()[:, 0])
                ).max()
            )
        )
    ratios = [before / after for before, after in itertools.pairwise(errors)]
    assert all(3.5 < ratio < 4.5 for ratio in ratios), f"ratios {ratios}"


def test_the_arc_length_map_is_strictly_increasing() -> None:
    """Which is what lets ``cell_index`` invert it, and what a per-line length is not.

    The metric factor is at least one everywhere, so ``S`` increases at least as fast
    as the parameter does -- and never slower, which is the other half of "arc length is
    never shorter than its projection".
    """
    for mesh in (
        _cylinder(0.05),
        build_fault(_faults("hope")[0][0], _faults("hope")[0][1])[0],
    ):
        for axis in (0, 1):
            knots, arc_km = mesh.arc_profile(axis)
            steps = np.diff(arc_km)
            # Faces share endpoints, so two knots can be a round-off apart and the
            # step between them zero. What matters is that the map never goes
            # backwards -- that is what makes `np.interp` a sound inverse -- and that
            # arc length never falls below the projection it came from.
            assert (steps >= 0.0).all()
            assert (steps >= np.diff(knots) - 1.0e-12).all()
            assert arc_km[-1] > arc_km[0]


def test_a_curved_patch_is_longer_than_its_projection() -> None:
    """``hope`` bends, so its true along-strike extent exceeds its parameter extent."""
    fault, crs = _faults("hope")[0]
    mesh = build_fault(fault, crs)[0]
    parameters = mesh.parameters_km()
    assert mesh.strike_arc_km().max() > parameters[:, 0].max()
    assert mesh.strike_arc_km().max() / parameters[:, 0].max() == pytest.approx(
        1.014, rel=0.01
    )


# ============================================================================
# Boundary detection, shared by the taper, the jump search and the parameterisation
# ============================================================================


@pytest.mark.parametrize("example", ("kaikoura", "hope", "beavan"))
def test_the_boundary_is_one_closed_loop(example: str) -> None:
    """Every edge is incident to one face or two, and the boundary edges close up.

    The property the taper and the jump search both rely on, and the one that the
    degenerate hull facets `from_patches` drops would otherwise break: with them in,
    five vertices of ``hope`` had two outgoing boundary edges each.
    """
    for fault, crs in _faults(example):
        for mesh in build_fault(fault, crs):
            edges = mesh.boundary_edges()
            assert len(edges) > 0
            # A closed loop visits each vertex once as a start and once as an end.
            assert len(np.unique(edges[:, 0])) == len(edges)
            assert len(np.unique(edges[:, 1])) == len(edges)
            assert set(edges[:, 0].tolist()) == set(edges[:, 1].tolist())

            labelled = mesh.boundary_labels()
            assert len(labelled) == len(edges)
            assert set(labelled.tolist()) <= set(BOUNDARY_LABELS)
            assert len(mesh.boundary_faces()) <= mesh.face_count


def test_the_boundary_labels_count_the_lattice_they_came_from() -> None:
    """A rectangular patch has one top edge per strike cell and two lateral sides."""
    for strike_count, dip_count in ((12, 8), (5, 17), (3, 3)):
        fault = _straight(strike_count=strike_count, dip_count=dip_count)
        mesh = build_fault(fault, NZTM)[0]
        counts = {label: len(mesh.boundary_edges(label)) for label in BOUNDARY_LABELS}
        assert counts == {
            "top": strike_count,
            "bottom": strike_count,
            "lateral": 2 * dip_count,
        }


def test_the_top_boundary_is_the_shallow_one() -> None:
    """The labels come from the parameter coordinates; this checks them against depth.

    Two independent descriptions of the same edge -- one from ``(u, v)`` and one from
    the fault's actual geometry -- which is what makes it a test rather than a
    restatement.
    """
    for dip_deg in (25.0, 60.0, 90.0):
        for dip_direction in ("left", "right"):
            mesh = build_fault(
                _straight(dip_deg=dip_deg, dip_direction=dip_direction), NZTM
            )[0]
            depth_km = mesh.vertices_km()[:, 2]
            top = depth_km[mesh.boundary_edges("top")]
            bottom = depth_km[mesh.boundary_edges("bottom")]
            assert top.max() < bottom.min()
            assert top.max() == pytest.approx(0.0, abs=1.0e-12)
            assert bottom.min() == pytest.approx(12.0, rel=EXACT)


def test_a_label_that_is_not_a_label_is_refused() -> None:
    """Naming what the labels are, rather than returning an empty array."""
    mesh = build_fault(_straight(), NZTM)[0]
    for method in (mesh.boundary_edges, mesh.boundary_faces):
        with pytest.raises(ValueError, match="not a boundary label"):
            method("side")


# ============================================================================
# The hypocentre seam
# ============================================================================


def test_a_position_in_a_face_comes_back_as_that_face() -> None:
    """The seam's whole job, checked at every face rather than at one.

    On a planar patch the arc coordinates and the parameter coordinates are the same,
    so a face's own centroid is unambiguously inside it and nothing else.
    """
    mesh = build_fault(_straight(strike_count=9, dip_count=6), NZTM)[0]
    arc = np.stack([mesh.strike_arc_km(), mesh.dip_arc_km()], axis=-1)
    centroids = arc[mesh.faces()].mean(axis=1)
    for index, (strike_km, dip_km) in enumerate(centroids):
        assert mesh.cell_index(float(strike_km), float(dip_km)) == index


def test_a_position_in_a_face_of_a_curved_patch_comes_back_as_that_face() -> None:
    """The same, where the arc coordinates and the parameter ones genuinely differ.

    The inversion through ``arc_profile`` has to be exact for this to hold, which is
    what strict monotonicity buys.
    """
    fault, crs = _faults("hope")[0]
    mesh = build_fault(fault, crs)[0]
    knots_u, arc_u = mesh.arc_profile(0)
    knots_v, arc_v = mesh.arc_profile(1)
    centroids = mesh.parameters_km()[mesh.faces()].mean(axis=1)
    for index in range(0, mesh.face_count, 37):
        strike_km = float(np.interp(centroids[index, 0], knots_u, arc_u))
        dip_km = float(np.interp(centroids[index, 1], knots_v, arc_v))
        assert mesh.cell_index(strike_km, dip_km) == index


def test_a_position_off_the_fault_is_refused_naming_the_axis() -> None:
    """A hypocentre off the fault is a config mistake, and the message says which way."""
    mesh = build_fault(_straight(), NZTM)[0]
    extent_strike = float(mesh.strike_arc_km().max())
    extent_dip = float(mesh.dip_arc_km().max())

    with pytest.raises(ValueError, match="strike_km .* is off the fault"):
        mesh.cell_index(extent_strike + 1.0, 1.0)
    with pytest.raises(ValueError, match="dip_km .* is off the fault"):
        mesh.cell_index(1.0, extent_dip + 1.0)
    with pytest.raises(ValueError, match="strike_km .* is off the fault"):
        mesh.cell_index(-1.0, 1.0)


def test_the_far_corner_of_the_fault_belongs_to_a_face() -> None:
    """ "At the bottom of the fault" is a thing people write, so it has to land."""
    mesh = build_fault(_straight(), NZTM)[0]
    for strike_km, dip_km in itertools.product(
        (0.0, float(mesh.strike_arc_km().max())),
        (0.0, float(mesh.dip_arc_km().max())),
    ):
        assert 0 <= mesh.cell_index(strike_km, dip_km) < mesh.face_count


# ============================================================================
# The container: fields, attributes, and being functional
# ============================================================================


def test_a_chart_starts_with_no_fields_and_keeps_its_own() -> None:
    """Attaching returns a new chart; the one it came from is untouched."""
    mesh = build_fault(_straight(), NZTM)[0]
    assert mesh.fields() == frozenset()

    with_slip = mesh.with_fields(slip_m=np.ones(mesh.face_count))
    assert with_slip.fields() == {"slip_m"}
    assert mesh.fields() == frozenset()
    assert "slip_m" in with_slip
    assert with_slip["slip_m"] == pytest.approx(np.ones(mesh.face_count))
    assert with_slip.without("slip_m").fields() == frozenset()
    assert with_slip.without("nothing_like_it").fields() == {"slip_m"}


def test_a_field_is_handed_back_read_only() -> None:
    """A writeable view is a mutable chart."""
    mesh = build_fault(_straight(), NZTM)[0].with_fields(
        slip_m=np.ones(build_fault(_straight(), NZTM)[0].face_count)
    )
    with pytest.raises(ValueError):
        mesh["slip_m"][0] = 2.0


def test_a_field_the_chart_does_not_carry_says_what_it_does() -> None:
    """A stage asking for a field nobody drew is a pipeline in the wrong order."""
    mesh = build_fault(_straight(), NZTM)[0].with_fields(
        rake_deg=np.zeros(build_fault(_straight(), NZTM)[0].face_count)
    )
    with pytest.raises(KeyError, match="rake_deg"):
        mesh["slip_m"]


@pytest.mark.parametrize(
    "name", ("east_km", "north_km", "depth_km", "strike_km", "dip_km", "plane_of_face")
)
def test_a_field_may_not_be_given_the_geometrys_own_name(name: str) -> None:
    """The chart's own arrays are not a stage's to overwrite."""
    mesh = build_fault(_straight(), NZTM)[0]
    with pytest.raises(ValueError, match="is the chart's own"):
        mesh.with_fields(**{name: np.zeros(mesh.face_count)})


def test_a_misshapen_or_non_finite_field_is_refused() -> None:
    """A NaN drawn here would otherwise reach the SRF with nothing having raised."""
    mesh = build_fault(_straight(), NZTM)[0]
    with pytest.raises(ValueError, match="faces"):
        mesh.with_fields(slip_m=np.ones(mesh.face_count + 1))
    with pytest.raises(ValueError, match="non-finite"):
        mesh.with_fields(slip_m=np.full(mesh.face_count, np.nan))


def test_attrs_are_read_only_and_the_structural_ones_are_not_a_stages() -> None:
    """The frame and the origin say what the chart *is*."""
    mesh = build_fault(_straight(), NZTM)[0]
    with pytest.raises(TypeError):
        mesh.attrs["surface"] = "elsewhere"

    assert mesh.with_attrs(hypocentre_face=3).attrs["hypocentre_face"] == 3
    assert "hypocentre_face" not in mesh.attrs
    for reserved in ("surface", "origin_east_km", "normal", "strike_axis"):
        with pytest.raises(ValueError, match="says what this chart is"):
            mesh.with_attrs(**{reserved: 0.0})


def test_a_patch_that_is_not_a_lattice_is_refused() -> None:
    """Named, with the shape a patch is supposed to be."""
    common = {
        "strike_deg": 90.0,
        "dip_deg": 90.0,
        "origin_east_km": 0.0,
        "origin_north_km": 0.0,
        "surface": "bad",
    }
    with pytest.raises(ValueError, match="at least 2x2 nodes"):
        TriangleMesh.from_patches([np.zeros((1, 4, 3))], **common)
    with pytest.raises(ValueError, match="non-finite"):
        TriangleMesh.from_patches([np.full((3, 4, 3), np.nan)], **common)
    with pytest.raises(ValueError, match="at least one patch"):
        TriangleMesh.from_patches([], **common)


def test_the_cell_arity_is_read_from_the_connectivity() -> None:
    """Following ``TetrahedralMeshSchema``: the arity is data, not a literal 3."""
    mesh = build_fault(_straight(), NZTM)[0]
    assert mesh.corner_count == mesh.faces().shape[1] == 3
    assert repr(mesh).startswith("TriangleMesh('straight'")


# ============================================================================
# The file
# ============================================================================


def test_a_segment_round_trips_through_a_file(tmp_path: Path) -> None:
    """The nodes, the faces and the parameter coordinates are the geometry.

    So everything derived from them comes back identical rather than close -- which is
    why the assertions are on exact equality rather than on a tolerance.
    """
    fault, crs = _faults("hope")[0]
    meshes = build_fault(fault, crs)
    path = tmp_path / "segments.h5"
    write_mesh({fault.name: meshes}, crs, path, attrs={"title": "a bent fault"})

    back, back_crs = read_mesh(path)
    assert back_crs == crs
    assert list(back) == [fault.name]
    for before, after in zip(meshes, back[fault.name], strict=True):
        assert np.array_equal(after.vertices_km(), before.vertices_km())
        assert np.array_equal(after.parameters_km(), before.parameters_km())
        assert np.array_equal(after.faces(), before.faces())
        assert np.array_equal(after.planes(), before.planes())
        assert after.origin_km == before.origin_km
        assert after.surface == before.surface
        assert np.array_equal(after.frame.normal, before.frame.normal)
        assert np.array_equal(after.frame.strike_axis, before.frame.strike_axis)
        assert np.array_equal(after.areas_km2(), before.areas_km2())


def test_a_version_2_file_still_reads_and_gets_triangulated() -> None:
    """Version 2 stored a lattice and no connectivity, so the reader supplies it.

    The nodes are the geometry, so the surface that comes back is the same surface --
    the same total area and the same frame -- and only the connectivity is new.
    """
    path = EXAMPLES / "mesh.h5"
    if not path.exists():
        pytest.skip("examples/mesh.h5 is not shipped in this checkout")

    meshes, crs = read_mesh(path)
    assert crs.is_projected
    assert meshes
    for segments in meshes.values():
        for mesh in segments:
            check_admissible(mesh)
            assert mesh.face_count > 0
            assert mesh.faces().max() < mesh.node_count
            # A v2 file holds planar per-plane charts, so the patch collapses.
            assert mesh.maximum_slope() < 1.0e-9


def test_reading_a_version_2_chart_gives_the_frame_the_config_would_have() -> None:
    """``from_chart`` reads the stated strike and dip back out of the geometry."""
    for fault, crs in _faults("beavan"):
        for chart, direct in zip(
            fuse(structured_fault(fault, crs)), build_fault(fault, crs), strict=True
        ):
            triangulated = from_chart(chart)
            assert triangulated.frame.strike_deg == pytest.approx(
                direct.frame.strike_deg, abs=1.0e-11
            )
            assert triangulated.frame.dip_deg == pytest.approx(
                direct.frame.dip_deg, abs=1.0e-11
            )
            assert triangulated.areas_km2().sum() == pytest.approx(
                chart.areas_km2().sum(), rel=EXACT
            )


def test_the_schema_version_is_three_and_the_structured_one_is_still_two() -> None:
    """The two formats are versioned together, and neither reader guesses."""
    from rupture_generator.formats.mesh import SCHEMA_VERSION as STRUCTURED_VERSION

    assert SCHEMA_VERSION == 3
    assert STRUCTURED_VERSION == 2


def test_a_mesh_is_not_written_as_an_srf(tmp_path: Path) -> None:
    """An SRF holds a rupture, not a surface."""
    fault, crs = _faults("kaikoura")[0]
    meshes = build_fault(fault, crs)
    with pytest.raises(ValueError, match="holds a fault surface"):
        write_mesh({fault.name: meshes}, crs, tmp_path / "surface.srf")
    with pytest.raises(ValueError, match="is not read from"):
        read_mesh(tmp_path / "surface.srf")
