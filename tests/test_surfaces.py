"""Reading a modeller's surface, and what survives resampling it onto a chart.

Three kinds of test, and the middle one is the point of the module.

1. **The parser** against the format's own rules -- one-based indices, the elevation
   sign, a file with no triangles.
2. **The resampling**, against the surface's own measured area. This is the claim the
   whole hybrid rests on: that a curved interface put onto a structured grid is still
   the same surface. It is checked on the shipped CFM interfaces rather than on a
   synthetic patch, because the failure it guards is a real triangulation's -- an
   outline that is not convex, and triangles that a Delaunay of the projected vertices
   re-cuts differently from the file.
3. **The chart contract**: that a resampled chart is one the rest of the package
   already knows how to handle, which is the whole reason for doing this at build time.

An exactly-planar patch is used where the expected answer must be known in closed
form, since a plane resampled onto its own reference plane is the identity.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pyproj
import pytest

from rupture_generator.errors import FormatError, GeometryError
from rupture_generator.formats import read_mesh, write_mesh
from rupture_generator.mesh import validate_chart
from rupture_generator.surfaces import MongeFrame, implied_axes, read_tsurf

CFM = Path("examples/cfm")
NZTM = pyproj.CRS("EPSG:2193")

CFM_INTERFACES = (
    # stem, vertices, faces, area km2
    ("Hikurangi", 5218, 9236, 181069.0),
    ("Puyseguer", 2597, 4090, 67921.5),
    ("Puysegur_Fiordland", 2312, 4041, 77968.6),
)
"""The NZ CFM v1.0 subduction interfaces, and what the archive's own files measure.

Unmodified and unclipped; ``Puyseguer`` is the CFM's own spelling. These are properties
of the files, not of this package, so they are the right thing to check a reader
against -- they stay true whatever reads them.
"""

AREA_TOLERANCE = 2.0e-3
"""How far a resampled chart's area may sit from the file's own.

Measured, not chosen: at 400 m all three interfaces come back inside 0.01%, and at 2 km
inside 0.09%. Two parts in a thousand is an order above the worst of that and three
orders below the 5.9% that a Delaunay-based occupancy test lost, which is the failure
this bound exists to catch.
"""


def _tsurf(stem: str):
    path = CFM / f"{stem}.ts.gz"
    if not path.exists():
        pytest.skip(f"{path} is not staged in this checkout")
    return read_tsurf(path)


def _planar_tsurf(tmp_path: Path, rows: int = 6, columns: int = 9) -> Path:
    """A flat dipping sheet as a TSurf, where every answer is known in closed form."""
    east, north = np.meshgrid(
        np.arange(columns, dtype=float), np.arange(rows, dtype=float), indexing="xy"
    )
    depth = 1.0 + 0.5 * north  # dips uniformly, so the best-fit plane is exact
    lines = ["GOCAD TSurf 1", "HEADER {name:flat}", "ZPOSITIVE Elevation", "TFACE"]
    for index, (x, y, z) in enumerate(
        zip(east.ravel(), north.ravel(), depth.ravel(), strict=True), start=1
    ):
        lines.append(f"VRTX {index} {x * 1000.0} {y * 1000.0} {-z * 1000.0}")
    for i in range(rows - 1):
        for j in range(columns - 1):
            a = i * columns + j + 1
            lines += [
                f"TRGL {a} {a + 1} {a + columns}",
                f"TRGL {a + 1} {a + columns + 1} {a + columns}",
            ]
    lines.append("END")
    path = tmp_path / "flat.ts"
    path.write_text("\n".join(lines))
    return path


# The parser.


@pytest.mark.parametrize("row", CFM_INTERFACES, ids=lambda row: row[0])
def test_a_cfm_interface_arrives_with_the_counts_the_archive_states(row: tuple) -> None:
    """The file's vertices and triangles, unchanged, and its own surface area.

    Against the archive rather than against a previous run of this package: these
    numbers describe the files, so they outlive any container that reads them.
    """
    stem, vertices, faces, area_km2 = row
    surface = _tsurf(stem)

    assert len(surface.vertices_km) == vertices
    assert len(surface.parts[0]) == faces
    assert surface.area_km2() == pytest.approx(area_km2, abs=0.05)


def test_positions_arrive_in_kilometres_with_depth_positive_down(
    tmp_path: Path,
) -> None:
    """The file is metres and elevation; this package is kilometres and depth."""
    surface = read_tsurf(_planar_tsurf(tmp_path))

    depth_km = surface.vertices_km[:, 2]
    assert depth_km.min() == pytest.approx(1.0)
    assert depth_km.max() == pytest.approx(1.0 + 0.5 * 5)
    assert surface.vertices_km[:, 0].max() == pytest.approx(8.0)


def test_a_gzipped_tsurf_reads_as_the_plain_one_does(tmp_path: Path) -> None:
    """The CFM ships them gzipped, so the reader opens both without being told."""
    plain = _planar_tsurf(tmp_path)
    zipped = tmp_path / "flat.ts.gz"
    zipped.write_bytes(gzip.compress(plain.read_bytes()))

    assert np.array_equal(read_tsurf(zipped).vertices_km, read_tsurf(plain).vertices_km)


def test_a_depth_positive_file_is_refused_rather_than_mirrored(tmp_path: Path) -> None:
    """``ZPOSITIVE Depth`` differs from Elevation by a sign on every vertex.

    Guessing puts the fault above sea level and nothing downstream would notice, so
    the reader refuses a convention it does not implement instead.
    """
    path = tmp_path / "down.ts"
    path.write_text("GOCAD TSurf 1\nZPOSITIVE Depth\nVRTX 1 0 0 0\nEND\n")

    with pytest.raises(FormatError, match="ZPOSITIVE"):
        read_tsurf(path)


def test_a_triangle_naming_an_undefined_vertex_is_refused(tmp_path: Path) -> None:
    """GOCAD's indices are one-based, and a file may number its vertices with gaps."""
    path = tmp_path / "gap.ts"
    path.write_text(
        "GOCAD TSurf 1\nVRTX 1 0 0 0\nVRTX 2 1 0 0\nVRTX 3 0 1 0\nTRGL 1 2 9\nEND\n"
    )

    with pytest.raises(FormatError, match="which no VRTX defines"):
        read_tsurf(path)


def test_a_file_with_no_triangles_is_not_a_surface(tmp_path: Path) -> None:
    """A point cloud has no connectivity, and the outline is the reason to read one."""
    path = tmp_path / "cloud.ts"
    path.write_text("GOCAD TSurf 1\nVRTX 1 0 0 0\nVRTX 2 1 0 0\nEND\n")

    with pytest.raises(FormatError, match="holds no TRGL"):
        read_tsurf(path)


# The resampling.


@pytest.mark.parametrize("row", CFM_INTERFACES, ids=lambda row: row[0])
@pytest.mark.parametrize("spacing_km", [2.0, 0.8])
def test_a_resampled_interface_keeps_the_surfaces_own_area(
    row: tuple, spacing_km: float
) -> None:
    """**The claim the one-track design rests on.**

    A curved interface cut onto a structured grid is still the same surface, and the
    measure of that is its area: the occupied cells must sum to what the file's own
    triangles do.

    This is the test that caught locating cells against a Delaunay of the projected
    vertices. Only 83% of that triangulation's simplices are triangles the file wrote,
    so treating the other 17% as outside the outline silently lost 5.9% of Hikurangi
    -- an error that does not shrink with resolution, because it is not
    discretisation.
    """
    surface = _tsurf(row[0])

    chart = surface.to_chart(spacing_km)

    resampled = chart.areas_km2()[chart.occupied()].sum()
    assert resampled == pytest.approx(surface.area_km2(), rel=AREA_TOLERANCE)


def test_an_outline_does_not_fill_the_rectangle_it_spans() -> None:
    """The mask is load-bearing, not decoration.

    If a subduction interface did fill its own bounding box there would be nothing for
    `occupied` to say and the rectangular taper would already be right. It does not:
    about a third of Hikurangi's cells are outside its outline.
    """
    chart = _tsurf("Hikurangi").to_chart(2.0)

    occupied = chart.occupied()
    assert 0.5 < occupied.mean() < 0.8
    assert not occupied.all()


def test_a_flat_sheet_resamples_to_its_own_plane(tmp_path: Path) -> None:
    """A plane is a Monge patch with ``h = 0``, so resampling it is the identity.

    The closed-form case: every node must land exactly on the sheet, which pins the
    frame fit, the projection and the lift together. A sign error in any of them shows
    up here and nowhere else, because on a curved surface every answer is approximate.
    """
    surface = read_tsurf(_planar_tsurf(tmp_path))

    chart = surface.to_chart(0.5)

    nodes = chart.nodes()
    east, north, depth = nodes[..., 0], nodes[..., 1], nodes[..., 2]
    # The sheet is depth = 1 + 0.5 * north, in its own absolute coordinates.
    origin_north = chart.origin_km[1]
    assert depth == pytest.approx(1.0 + 0.5 * (north + origin_north), abs=1e-9)
    assert np.isfinite(east).all()


def test_a_surface_that_folds_is_not_a_chart(tmp_path: Path) -> None:
    """Two heights over one point, and a chart holds one.

    Folding it flat would keep whichever sheet the interpolation reached and lose the
    other without saying so.
    """
    path = tmp_path / "fold.ts"
    # Four vertices whose triangles wind opposite ways once projected: the second
    # doubles back over the first.
    lines = [
        "GOCAD TSurf 1",
        "VRTX 1 0 0 0",
        "VRTX 2 10000 0 -1000",
        "VRTX 3 0 10000 -2000",
        "VRTX 4 -10000 0 -3000",
        "TRGL 1 2 3",
        "TRGL 3 2 1",
        "END",
    ]
    path.write_text("\n".join(lines))
    surface = read_tsurf(path)

    with pytest.raises(GeometryError, match="folds back"):
        surface.to_chart(1.0)


def test_a_part_the_file_does_not_have_is_refused() -> None:
    """Two disconnected sheets are two parts; asking for a third is a mistake."""
    surface = _tsurf("Hikurangi")

    with pytest.raises(FormatError, match="there is no part"):
        surface.to_chart(2.0, part=99)


@pytest.mark.parametrize("spacing_km", [0.0, -1.0])
def test_a_non_positive_spacing_is_refused(spacing_km: float) -> None:
    """Zero would be an infinite grid, and negative is not a resolution at all."""
    surface = _tsurf("Puyseguer")

    with pytest.raises(GeometryError, match="spacing must be positive"):
        surface.to_chart(spacing_km)


# The chart contract: what the rest of the package may now assume.


def test_a_resampled_chart_is_one_the_sampler_accepts() -> None:
    """The point of resampling at build time.

    `validate_chart` asks for one regular grid, and a chart cut on a reference plane
    is one by construction -- in its parameters, which is where the sampler and the
    sweep both work. Its *lifted* node steps vary with the curvature, and that is the
    surface's own stretch rather than an uneven discretisation.
    """
    chart = _tsurf("Hikurangi").to_chart(4.0)

    validate_chart(chart)

    assert chart.parameter_spacing_km() == (4.0, 4.0)
    assert chart.spacing_km() == (4.0, 4.0)


def test_the_lifted_steps_do_vary_even_though_the_grid_is_regular() -> None:
    """Why the spacing is stated rather than measured off the nodes.

    Reading a spacing from a curved chart's node positions would report the stretch as
    a discretisation, and `validate_chart` would refuse the surface for being curved --
    which is the thing the design is for.
    """
    chart = _tsurf("Hikurangi").to_chart(4.0)

    strike_steps, dip_steps = chart.line_steps()
    assert strike_steps.max() > 1.05 * strike_steps.min()
    assert dip_steps.max() > 1.05 * dip_steps.min()
    assert chart.parameter_spacing_km() == (4.0, 4.0)


def test_a_curved_chart_round_trips_through_the_mesh_file(tmp_path: Path) -> None:
    """The mask and the stated spacing are geometry, so a file that drops them lies.

    A chart read back without its mask fills its rectangle, which would put slip on a
    third of Hikurangi that is not fault; without its spacing it is refused as
    unevenly cut.
    """
    chart = _tsurf("Puysegur_Fiordland").to_chart(4.0)
    path = tmp_path / "interface.h5"

    write_mesh({"puysegur": [chart]}, NZTM, path)
    read_back, crs = read_mesh(path)

    got = read_back["puysegur"][0]
    assert got.cell_counts == chart.cell_counts
    assert np.array_equal(got.occupied(), chart.occupied())
    assert got.parameter_spacing_km() == chart.parameter_spacing_km()
    assert np.array_equal(got.nodes(), chart.nodes())
    assert crs == NZTM
    validate_chart(got)


def test_the_implied_axes_of_an_interface_are_its_own_best_fit() -> None:
    """A TSurf states no strike or dip, so they are read off the surface.

    Checked against the interface's geometry rather than a stored number: Hikurangi
    dips shallowly westward under the North Island.
    """
    surface = _tsurf("Hikurangi")

    strike_deg, dip_deg = implied_axes(
        surface.vertices_km[surface.parts[0]].reshape(-1, 3)
    )

    assert 0.0 < dip_deg < 30.0
    assert 0.0 <= strike_deg < 360.0


def test_a_horizontal_surface_has_no_strike_to_read() -> None:
    """Strike is the direction of the level line, and a level plane is all of them."""
    flat = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])

    with pytest.raises(GeometryError, match="horizontal"):
        implied_axes(flat)


def test_a_frame_fits_a_plane_exactly() -> None:
    """The frame is a least-squares fit, and on a plane the residual is zero."""
    strike_deg, dip_deg = 30.0, 45.0
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [2.0, 1.0, 1.0]]
    )

    frame = MongeFrame.fit(points, strike_deg=strike_deg, dip_deg=dip_deg)

    # Every point lies in the plane, so every height is zero, and lifting is exact.
    uvh = frame.project(points)
    assert uvh[:, 2] == pytest.approx(0.0, abs=1e-12)
    assert frame.lift(uvh) == pytest.approx(points, abs=1e-12)


def test_a_frame_needs_three_points_to_be_a_plane() -> None:
    """Two points name a line, and a line has infinitely many planes through it."""
    with pytest.raises(GeometryError, match="at least 3 points"):
        MongeFrame.fit(np.zeros((2, 3)), strike_deg=0.0, dip_deg=45.0)
