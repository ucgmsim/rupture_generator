"""Curved faults: the TSurf reader, and the charts unrolling a mesh gives."""

import gzip
import io
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rupture_generator import RuptureGeneratorError
from rupture_generator.geometry import flatten, read_tsurf, surface, surface_chart
from rupture_generator.geometry.surface import MAXIMUM_STRETCH

CFM = Path(__file__).parents[1] / "examples" / "cfm"


def _grid(rows, columns, keep=None):
    """A quad grid of integer points, each quad cut into two triangles.

    Returns the ``(n, 2)`` points and the ``(m, 3)`` triangles, all wound one way.
    ``keep(r, c)`` drops the quads it rejects.
    """
    r, c = np.mgrid[0 : rows + 1, 0 : columns + 1]
    points = np.column_stack([r.ravel(), c.ravel()]).astype(float)
    index = np.arange(points.shape[0]).reshape(rows + 1, columns + 1)
    triangles = []
    for i in range(rows):
        for j in range(columns):
            if keep is not None and not keep(i, j):
                continue
            a, b = index[i, j], index[i + 1, j]
            c, d = index[i + 1, j + 1], index[i, j + 1]
            triangles += [(a, b, c), (a, c, d)]
    return points, np.array(triangles)


def _aligned(flat, expected):
    """``flat`` moved by the rigid motion, reflections included, closest to ``expected``."""
    a, b = flat - flat.mean(axis=0), expected - expected.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    return a @ (u @ vt) + expected.mean(axis=0)


def _plane(strike_vec, dip_vec, points_km, origin_km=(500.0, 5000.0, 2.0)):
    """Points ``(along strike, down dip)`` placed on a plane in 3D."""
    return (
        np.asarray(origin_km)
        + points_km[:, :1] * np.asarray(strike_vec)
        + points_km[:, 1:] * np.asarray(dip_vec)
    )


def _axes(dip_direction_deg, dip_deg):
    """Unit strike and down-dip vectors for a fault dipping to the right of strike."""
    azimuth, dip = np.radians(dip_direction_deg), np.radians(dip_deg)
    down = np.array(
        [np.cos(dip) * np.sin(azimuth), np.cos(dip) * np.cos(azimuth), np.sin(dip)]
    )
    strike = np.array([-np.cos(azimuth), np.sin(azimuth), 0.0])
    return strike, down


def _chart_stretch(chart, spacing_km):
    """The area-weighted 0.95 quantile of stretch over occupied cells."""
    p = chart.nodes
    down = 0.5 * (p[1:, :-1] - p[:-1, :-1] + p[1:, 1:] - p[:-1, 1:]) / spacing_km
    along = 0.5 * (p[:-1, 1:] - p[:-1, :-1] + p[1:, 1:] - p[1:, :-1]) / spacing_km
    jacobian = np.stack([down, along], axis=-1)[chart.occupied]
    sigma = np.linalg.svd(jacobian, compute_uv=False)
    stretch = np.abs(np.log(sigma)).max(axis=1)
    order = np.argsort(stretch)
    cumulative = np.cumsum(chart.areas_km2[chart.occupied][order])
    return float(stretch[order][np.searchsorted(cumulative, 0.95 * cumulative[-1])])


# ----------------------------------------------------------------------- flatten


@settings(max_examples=20, deadline=None)
@given(st.floats(5.0, 120.0), st.floats(0.3, 3.0))
def test_a_hinge_unrolls_exactly(angle_deg, wing_km):
    points, triangles = _grid(6, 5)
    across, along = points[:, 0] * wing_km / 3.0, points[:, 1]
    beyond = np.maximum(across - wing_km, 0.0)
    folded = np.radians(angle_deg)
    vertices = np.column_stack(
        [
            np.minimum(across, wing_km) + beyond * np.cos(folded),
            along,
            beyond * np.sin(folded),
        ]
    )
    developed = np.column_stack([across, along])
    flat = flatten(vertices, triangles)
    np.testing.assert_allclose(_aligned(flat, developed), developed, atol=1e-9)


def test_a_cylinder_unrolls_to_its_developed_rectangle():
    radius_km, steps, sweep = 20.0, 24, np.radians(90.0)
    points, triangles = _grid(steps, 10)
    angle = points[:, 0] * sweep / steps
    vertices = np.column_stack(
        [radius_km * np.sin(angle), points[:, 1], radius_km * (1.0 - np.cos(angle))]
    )
    # Each facet is a flat rectangle one chord wide.
    chord_km = 2.0 * radius_km * np.sin(0.5 * sweep / steps)
    developed = np.column_stack([points[:, 0] * chord_km, points[:, 1]])
    flat = flatten(vertices, triangles)
    # A fixed number of passes leaves this quarter turn under a millimetre from its
    # development, across 36 km.
    np.testing.assert_allclose(_aligned(flat, developed), developed, atol=1e-5)


# -------------------------------------------------------------------- read_tsurf


def _tsurf(
    body="VRTX 1 0 0 -1000\nVRTX 2 1000 0 -1000\nVRTX 3 0 1000 -2000\nTRGL 1 2 3\n",
    name="sheet",
    units='AXIS_UNIT "m" "m" "m"',
    zpositive="ZPOSITIVE Elevation",
    end="END\n",
):
    header = f"HEADER {{\nname:{name}\n*visible:true\n}}\n" if name else ""
    return (
        f"GOCAD TSurf 1\n{header}GOCAD_ORIGINAL_COORDINATE_SYSTEM\nNAME Default\n"
        f"{units}\n{zpositive}\nEND_ORIGINAL_COORDINATE_SYSTEM\nTFACE\n{body}{end}"
    )


def _read(text):
    return read_tsurf(io.StringIO(text))


def test_a_tsurf_reads_in_kilometres_with_depth_down():
    body = (
        "PVRTX 10 0 0 -1000 7.5\nVRTX 20 2000 0 -1000\nVRTX 30 0 3000 -4000\n"
        "VRTX 40 9 9 9\nVRTX 50 2000 3000 -4000\n\nTRGL 10 20 30\nTFACE\nTRGL 20 50 30\n"
    )
    vertices_km, triangles = _read(_tsurf(body))["sheet"]
    np.testing.assert_allclose(
        vertices_km, [[0, 0, 1], [2, 0, 1], [0, 3, 4], [2, 3, 4]], atol=1e-12
    )
    np.testing.assert_array_equal(triangles, [[0, 1, 2], [1, 3, 2]])


def test_a_tsurf_in_kilometres_measuring_depth_is_kept_as_it_is():
    text = _tsurf(units='AXIS_UNIT "km" "km" "km"', zpositive="ZPOSITIVE Depth")
    vertices_km, _ = _read(text)["sheet"]
    np.testing.assert_array_equal(vertices_km[2], [0.0, 1000.0, -2000.0])


def test_every_surface_in_a_file_is_read_by_name():
    surfaces = _read(_tsurf(name="one") + _tsurf(name="two"))
    assert list(surfaces) == ["one", "two"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "holds no GOCAD TSurf"),
        ("GOCAD PLine 1\nEND\n", "is not a TSurf"),
        ("VRTX 1 0 0 0\n" + _tsurf(), "outside any GOCAD TSurf"),
        (_tsurf(end="") + _tsurf(), "before the last one's END"),
        (_tsurf(end=""), "ends inside sheet 'sheet'"),
        (_tsurf(name=""), "no name"),
        (_tsurf(units=""), "no AXIS_UNIT or no ZPOSITIVE"),
        (_tsurf(zpositive=""), "no AXIS_UNIT or no ZPOSITIVE"),
        (_tsurf(units='AXIS_UNIT "ft" "ft" "ft"'), "is not three of"),
        (_tsurf(units='AXIS_UNIT "m" "m"'), "is not three of"),
        (_tsurf(zpositive="ZPOSITIVE Up"), "neither Elevation nor Depth"),
        (_tsurf(zpositive="ZPOSITIVE"), "neither Elevation nor Depth"),
        (_tsurf("VRTX 1 0 zero 0\nTRGL 1 1 1\n"), "is not a VRTX record"),
        (_tsurf("VRTX 1 0 0 0\nTRGL 1 2\n"), "is not a TRGL record"),
        (_tsurf("VRTX 1.5 0 0 0\n"), "is not a VRTX record"),
        (_tsurf("VRTX 1 0 0 0\n"), "no TRGL records"),
        (_tsurf("TRGL 1 2 3\n"), "names vertex 1"),
        (
            _tsurf("VRTX 1 0 0 0\nVRTX 2 1 0 0\nVRTX 3 0 1 0\nTRGL 1 2 4\n"),
            "names vertex 4",
        ),
        (
            _tsurf("VRTX 1 0 0 0\nVRTX 1 1 0 0\nVRTX 3 0 1 0\nTRGL 1 1 3\n"),
            "share an id",
        ),
        (_tsurf() + _tsurf(), "two sheets are named 'sheet'"),
        (
            _tsurf("VRTX 1 0 0 0\nVRTX 2 1 0 0\nVRTX 3 0 1 0\nTRGL 1 2 2\n"),
            "names one vertex twice",
        ),
        (
            _tsurf(
                "VRTX 1 0 0 0\nVRTX 2 1 0 0\nVRTX 3 0 1 0\nVRTX 4 1 1 0\n"
                "TRGL 1 2 3\nTRGL 2 3 4\n"
            ),
            "not one consistently wound sheet",
        ),
        (
            _tsurf(
                "VRTX 1 0 0 0\nVRTX 2 1 0 0\nVRTX 3 0 1 0\nVRTX 4 5 0 0\n"
                "VRTX 5 6 0 0\nTRGL 1 2 3\nTRGL 3 4 5\n"
            ),
            "2 pieces",
        ),
    ],
)
def test_a_tsurf_it_cannot_interpret_is_refused(text, message):
    with pytest.raises(RuptureGeneratorError, match=message):
        _read(text)


# ------------------------------------------------------------------ surface_chart


@settings(max_examples=25, deadline=None)
@given(st.floats(0.0, 360.0), st.floats(5.0, 85.0))
def test_a_plane_charts_down_dip_along_i_and_dipping_right_of_j(direction_deg, dip_deg):
    strike, down = _axes(direction_deg, dip_deg)
    points, triangles = _grid(6, 4)
    vertices_km = _plane(strike, down, points * [3.0, 2.0])
    spacing_km = 1.0
    chart = surface_chart(vertices_km, triangles, spacing_km)
    assert chart.cells == (8, 18)
    assert chart.plane_cells == (18,)
    assert chart.occupied.all()
    nodes = chart.nodes
    steps = np.broadcast_to(spacing_km * down, (8, 19, 3))
    np.testing.assert_allclose(nodes[1:] - nodes[:-1], steps, atol=1e-9)
    np.testing.assert_allclose(
        nodes[:, 1:] - nodes[:, :-1],
        np.broadcast_to(spacing_km * strike, (9, 18, 3)),
        atol=1e-9,
    )


def test_cells_are_occupied_where_their_centre_is_on_the_surface():
    strike, down = _axes(120.0, 40.0)
    points, triangles = _grid(10, 6, keep=lambda i, j: i < 4 or j < 3)
    used, triangles = np.unique(triangles, return_inverse=True)
    points = points[used] * 0.9
    vertices_km = _plane(strike, down, points)
    spacing_km = 0.7
    chart = surface_chart(vertices_km, triangles, spacing_km)
    offset = chart.centres - _plane(strike, down, np.zeros((1, 2)))[0]
    along, across = offset @ strike, offset @ down
    inside = (
        (along >= 0)
        & (along <= 9.0)
        & (across >= 0)
        & (across <= 5.4)
        & ((along <= 3.6) | (across <= 2.7))
    )
    assert inside.sum() > 0
    np.testing.assert_array_equal(chart.occupied, inside)
    # The nodes off the mesh continue the plane.
    np.testing.assert_allclose(
        (chart.nodes - _plane(strike, down, np.zeros((1, 2)))[0])
        @ np.cross(strike, down),
        0.0,
        atol=1e-9,
    )


def test_cells_on_a_curved_surface_keep_the_requested_size():
    radius_km, steps, sweep = 30.0, 60, np.radians(70.0)
    points, triangles = _grid(steps, 20)
    angle = points[:, 0] * sweep / steps
    vertices_km = np.column_stack(
        [
            radius_km * np.sin(angle),
            2.0 * points[:, 1],
            1.0 + radius_km * (1.0 - np.cos(angle)),
        ]
    )
    spacing_km = 1.5
    chart = surface_chart(vertices_km, triangles, spacing_km)
    occupied = chart.occupied
    assert occupied.mean() > 0.8
    nodes = chart.nodes
    down = np.linalg.norm(nodes[1:] - nodes[:-1], axis=-1)
    along = np.linalg.norm(nodes[:, 1:] - nodes[:, :-1], axis=-1)
    np.testing.assert_allclose(down[:, :-1][occupied], spacing_km, rtol=2e-3)
    np.testing.assert_allclose(along[:-1][occupied], spacing_km, rtol=2e-3)
    # Depth grows down every column of the fault.
    depth = np.where(occupied, chart.centres[..., 2], np.nan)
    assert np.all(np.diff(depth, axis=0)[~np.isnan(np.diff(depth, axis=0))] > 0)


@pytest.mark.parametrize("spacing_km", [0.0, -1.0, np.nan, np.inf])
def test_a_spacing_that_is_not_a_length_is_refused(spacing_km):
    points, triangles = _grid(2, 2)
    vertices_km = _plane(*_axes(90.0, 45.0), points)
    with pytest.raises(RuptureGeneratorError, match="not a positive length"):
        surface_chart(vertices_km, triangles, spacing_km)


def test_a_level_surface_is_refused():
    points, triangles = _grid(3, 3)
    vertices_km = np.column_stack([points, np.full(len(points), 5.0)])
    with pytest.raises(RuptureGeneratorError, match="level"):
        surface_chart(vertices_km, triangles, 0.5)


def test_a_folded_flattening_is_refused(monkeypatch):
    points, triangles = _grid(4, 4)
    vertices_km = _plane(*_axes(90.0, 45.0), points)
    # Flattening folds no mesh this test can build. The test puts the fold in its
    # result instead: an interior vertex pushed past its neighbours turns triangles.
    folded = points.copy()
    folded[12] += 1.5
    monkeypatch.setattr(surface, "flatten", lambda vertices, triangles: folded)
    with pytest.raises(RuptureGeneratorError, match=r"folds \d+ of the mesh's 32"):
        surface_chart(vertices_km, triangles, 0.5)


def test_a_surface_that_will_not_unroll_is_refused():
    points, triangles = _grid(12, 12)
    # A spherical cap, 90 degrees across.
    direction = np.column_stack([points / 6.0 - 1.0, np.ones(len(points))])
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    vertices_km = 50.0 * direction + [0.0, 0.0, 10.0]
    with pytest.raises(RuptureGeneratorError, match="at the 95th percentile") as error:
        surface_chart(vertices_km, triangles, 1.0)
    assert f"{MAXIMUM_STRETCH:.0%}" in str(error.value)


def test_a_spacing_coarser_than_the_surface_is_refused():
    # A V whose notch holds the bounding box's centre.
    points = np.array([[0.0, 2.0], [1.0, 0.0], [1.0, 0.5], [2.0, 2.0]])
    triangles = np.array([[0, 1, 2], [2, 1, 3]])
    vertices_km = _plane(*_axes(90.0, 45.0), points[:, ::-1])
    with pytest.raises(RuptureGeneratorError, match="too coarse"):
        surface_chart(vertices_km, triangles, 10.0)


@pytest.mark.parametrize(
    ("file", "bound"),
    [
        # Measured at 5 km: 0.85% and 1.63%.
        ("Hikurangi", 0.015),
        ("Puyseguer", 0.025),
    ],
)
def test_the_community_fault_model_interfaces_chart(file, bound):
    spacing_km = 5.0
    with gzip.open(CFM / f"{file}.ts.gz", "rt") as handle:
        surfaces = read_tsurf(handle)
    assert list(surfaces) == [file]
    vertices_km, triangles = surfaces[file]
    chart = surface_chart(vertices_km, triangles, spacing_km)
    assert chart.plane_cells == (chart.cells[1],)
    assert _chart_stretch(chart, spacing_km) < bound
    depth = chart.centres[..., 2][chart.occupied]
    assert 0.0 < depth.min() and depth.max() < 80.0
