"""The mesh boundary: what crosses, what it means, and what it refuses.

`crates/genslip/tests/mesh.rs` already asserts the geometry as identities in Rust. This
does not repeat it. What it covers is everything that only exists once an array has
crossed into Python:

* the **shapes and orientation** of what comes back -- `(dip, strike)` for cells,
  `(dip_node, strike_node)` for nodes, one more node than cell on each axis;
* the **offsets and the origin**, which is the boundary's one real trap: every position
  is measured from `mesh.origin` and adding it back is the caller's job;
* `from_positions`, which is how a mesh arrives from a *file* and so is the one path
  that can be handed something the builders could never produce;
* `to_subfault_geometry`, the seam `assemble.py` has been asking for since it was
  written.

Hypothesis drives the extents and the geometry, because the failures worth catching here
are off-by-ones and transpositions, and those hide from any single fixture: a 12x12 fault
satisfies a transposed reader exactly as well as a correct one. Every extent-sensitive
test therefore uses a **non-square** fault, and most are given a range of them.
"""

import numpy as np
import pyproj
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rupture_generator._core import (
    Cuts,
    Fault,
    Plane,
    PointSource,
    Projected,
    RefinedMesh,
    build_fault_mesh,
    build_point_mesh,
)
from rupture_generator.assemble import SubfaultGeometry
from rupture_generator.mesh import node_positions_wgs84, to_subfault_geometry
from rupture_generator.units import CM2_PER_KM2

NZTM = pyproj.CRS("EPSG:2193")

# Somewhere in the middle of the South Island, in NZTM kilometres. The particular place
# does not matter to anything here; that it is a realistic magnitude does, because it is
# what the offset design exists to cope with.
ORIGIN = Projected(1_500.0, 5_180.0)

SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

EXTENTS = {
    "strike_count": st.integers(min_value=1, max_value=24),
    "dip_count": st.integers(min_value=1, max_value=16),
}


def a_plane(
    *,
    bearing_deg: float = 55.0,
    length_km: float = 20.0,
    dip_deg: float = 60.0,
    top_depth_km: float = 0.0,
    bottom_depth_km: float = 12.0,
    dips_left: bool = False,
) -> Fault:
    """A one-plane fault, given by a grid bearing and a length."""
    bearing = np.deg2rad(bearing_deg)
    end = Projected(
        ORIGIN.easting_km + length_km * np.sin(bearing),
        ORIGIN.northing_km + length_km * np.cos(bearing),
    )
    return Fault(
        ORIGIN,
        [
            Plane(
                end,
                dip_deg=dip_deg,
                bottom_depth_km=bottom_depth_km,
                dips_left=dips_left,
            )
        ],
        top_depth_km=top_depth_km,
    )


def a_mesh(strike_count: int = 20, dip_count: int = 12, **plane) -> RefinedMesh:
    return build_fault_mesh(a_plane(**plane), [Cuts(strike_count, dip_count)])


class TestShapesAndOrientation:
    """Arrays come back the shape and the way round the core expects.

    `crates/genslip/src/grid.rs` fixes the convention: `(dip, strike)`, strike varying
    fastest -- *"the order the RNG consumes draws in, the order an SRF stores points in,
    and the layout the FFT expects"*. A transposition here is invisible to anything that
    only counts elements, and produces a rupture that is entirely plausible.
    """

    @given(**EXTENTS)
    @SETTINGS
    def test_cell_arrays_are_dip_by_strike(
        self, strike_count: int, dip_count: int
    ) -> None:
        mesh = a_mesh(strike_count, dip_count)
        assert mesh.cell_extents(0) == (strike_count, dip_count)
        for array in (mesh.areas_km2(0), mesh.strike_deg(0), mesh.dip_deg(0)):
            assert array.shape == (dip_count, strike_count)
        for array in mesh.cell_centres(0):
            assert array.shape == (dip_count, strike_count)

    @given(**EXTENTS)
    @SETTINGS
    def test_node_arrays_have_one_more_of_each(
        self, strike_count: int, dip_count: int
    ) -> None:
        mesh = a_mesh(strike_count, dip_count)
        for array in mesh.node_positions(0):
            assert array.shape == (dip_count + 1, strike_count + 1)
        assert mesh.strike_arc_km(0).shape == (strike_count + 1,)
        assert mesh.dip_arc_km(0).shape == (dip_count + 1,)

    @given(**EXTENTS)
    @SETTINGS
    def test_depth_increases_down_the_first_axis(
        self, strike_count: int, dip_count: int
    ) -> None:
        """Which axis is dip, asserted by the one field that can only go one way.

        The check a transposition cannot survive: depth is constant along strike and
        strictly increasing down dip, so a swapped pair of axes is immediately visible.
        """
        mesh = a_mesh(strike_count, dip_count)
        _, _, depth_km = mesh.node_positions(0)

        assert np.all(np.diff(depth_km, axis=0) > 0.0), (
            "depth does not increase down dip"
        )
        assert np.allclose(np.diff(depth_km, axis=1), 0.0), "depth varies along strike"


class TestPositionsAreOffsets:
    """The boundary's one real trap.

    Every position is measured from `mesh.origin`. A caller that forgets is not off by
    a little -- it is 5,000 km out, which is at least loud. A caller that adds it twice
    is 5,000 km out the other way. Neither is subtle; what would be subtle is the
    library quietly returning absolute coordinates and losing the precision the offsets
    exist to keep.
    """

    def test_the_origin_is_the_fault_origin(self) -> None:
        mesh = a_mesh()
        assert mesh.origin == ORIGIN

    def test_a_point_source_is_centred_on_its_origin(self) -> None:
        centre = Projected(1_500.0, 5_180.0)
        mesh = build_point_mesh(
            PointSource(
                centre, depth_km=8.0, strike_deg=55.0, dip_deg=60.0, size_km=0.5
            )
        )
        assert mesh.origin == centre
        east_km, north_km, depth_km = mesh.cell_centres(0)

        # Zero to round-off rather than exactly zero: the cell centre is the mean of
        # four corners, and the half-cell up-dip step that places the top edge is
        # undone by a half-cell down-dip step computed a different way -- `size *
        # cos(dip)` against `depth_span / tan(dip)`, which agree to the last bits and
        # not beyond. Measured at 5.6e-17 km, which is 0.06 nanometres.
        #
        # What matters is the *scale*: had positions been absolute, this would read
        # 1500 and carry the rounding of a CRS coordinate into every derived quantity.
        assert abs(east_km[0, 0]) < 1e-12
        assert abs(north_km[0, 0]) < 1e-12
        assert depth_km[0, 0] == pytest.approx(8.0, abs=1e-12)

    @given(**EXTENTS)
    @SETTINGS
    def test_offsets_stay_at_fault_scale(
        self, strike_count: int, dip_count: int
    ) -> None:
        """No returned position is anywhere near a projected coordinate's magnitude.

        The property the offsets buy. A 20 km fault reaching 12 km down cannot have a
        node further than ~30 km from its origin; a node at 5,180 would mean absolute
        coordinates had leaked through, and every derived quantity would silently carry
        `1.2e-12` relative error instead of `3e-15`.
        """
        mesh = a_mesh(strike_count, dip_count)
        for array in mesh.node_positions(0):
            assert np.all(np.abs(array) < 50.0)


class TestGeometryCrossesIntact:
    """A spot-check that the identities survive the boundary.

    Not a second copy of `mesh.rs`'s suite -- that would be a transcription, and
    `README.md`'s fourth trap is about exactly that. These are the few that could break
    *at* the boundary: a wrong array handed back, a transposed reshape, a unit dropped.
    """

    @given(**EXTENTS)
    @SETTINGS
    def test_the_areas_sum_to_length_times_width(
        self, strike_count: int, dip_count: int
    ) -> None:
        mesh = a_mesh(strike_count, dip_count)
        width_km = 12.0 / np.sin(np.deg2rad(60.0))
        assert mesh.areas_km2(0).sum() == pytest.approx(20.0 * width_km, rel=1e-12)

    @given(
        dip_deg=st.floats(min_value=5.0, max_value=90.0),
        bearing_deg=st.floats(min_value=0.0, max_value=359.9),
    )
    @SETTINGS
    def test_every_cell_reports_the_plane_it_was_built_with(
        self, dip_deg: float, bearing_deg: float
    ) -> None:
        mesh = a_mesh(8, 5, dip_deg=dip_deg, bearing_deg=bearing_deg)
        assert mesh.dip_deg(0) == pytest.approx(dip_deg, abs=1e-9)
        # Bearing comes back folded into [0, 360); compare on the circle.
        gap = np.abs(mesh.strike_deg(0) - bearing_deg % 360.0)
        assert np.all(np.minimum(gap, 360.0 - gap) < 1e-9)

    @given(**EXTENTS)
    @SETTINGS
    def test_the_spacing_is_the_length_over_the_count(
        self, strike_count: int, dip_count: int
    ) -> None:
        mesh = a_mesh(strike_count, dip_count)
        strike_km, dip_km = mesh.spacing(0)
        width_km = 12.0 / np.sin(np.deg2rad(60.0))
        assert strike_km == pytest.approx(20.0 / strike_count, rel=1e-12)
        assert dip_km == pytest.approx(width_km / dip_count, rel=1e-12)

    @given(**EXTENTS)
    @SETTINGS
    def test_a_position_in_a_cell_comes_back_as_that_cell(
        self, strike_count: int, dip_count: int
    ) -> None:
        """`cell_index` round-trips over **every** cell.

        `DEFECTS.md` 17 was this arithmetic off by one cell in each direction, producing
        a rupture that was smooth, started at zero and correlated 0.99+ with the right
        one. Every cell rather than a sample, because the defect was a constant offset
        and nobody checked even one.
        """
        mesh = a_mesh(strike_count, dip_count)
        strike_arc = mesh.strike_arc_km(0)
        dip_arc = mesh.dip_arc_km(0)

        for dip in range(dip_count):
            for strike in range(strike_count):
                found = mesh.cell_index(
                    0,
                    0.5 * (strike_arc[strike] + strike_arc[strike + 1]),
                    0.5 * (dip_arc[dip] + dip_arc[dip + 1]),
                )
                assert found == (strike, dip)

    @pytest.mark.parametrize("dips_left", [False, True])
    def test_dip_direction_is_honoured(self, dips_left: bool) -> None:
        """A north-striking fault dips east or west, and which one is a flag.

        The axis convention. Every distance and area assertion is blind to it: a fault
        dipping the wrong way has all the right sizes.
        """
        mesh = a_mesh(8, 5, bearing_deg=0.0, dips_left=dips_left)
        east_km, _, _ = mesh.node_positions(0)
        drift = east_km[-1, 0] - east_km[0, 0]
        assert (drift < 0.0) if dips_left else (drift > 0.0)


class TestFromPositions:
    """How a mesh arrives from a file, and what that path has to refuse.

    The builders cannot produce a dangling index or a patch too small to have a cell.
    This path can, because it is handed arrays that may have been written by something
    else or edited by hand -- so it is the one place those checks are reachable, and the
    one place they matter.
    """

    @given(**EXTENTS)
    @SETTINGS
    def test_a_mesh_round_trips_through_its_positions(
        self, strike_count: int, dip_count: int
    ) -> None:
        """Rebuilt from its own arrays, a mesh derives the same everything.

        What makes a mesh file lossless. The node positions are the geometry, so a
        rebuild has to reproduce every quantity derived from them -- and if any of those
        were secretly carried alongside rather than derived, this is where it shows.
        """
        original = a_mesh(strike_count, dip_count)
        rebuilt = RefinedMesh.from_positions(
            original.origin, [original.node_positions(0)]
        )

        assert rebuilt.origin == original.origin
        assert rebuilt.patch_count == original.patch_count
        assert rebuilt.cell_extents(0) == original.cell_extents(0)
        for rebuilt_array, original_array in zip(
            (
                rebuilt.areas_km2(0),
                rebuilt.strike_deg(0),
                rebuilt.dip_deg(0),
                rebuilt.strike_arc_km(0),
                rebuilt.dip_arc_km(0),
                *rebuilt.cell_centres(0),
                *rebuilt.node_positions(0),
            ),
            (
                original.areas_km2(0),
                original.strike_deg(0),
                original.dip_deg(0),
                original.strike_arc_km(0),
                original.dip_arc_km(0),
                *original.cell_centres(0),
                *original.node_positions(0),
            ),
            strict=True,
        ):
            assert np.array_equal(rebuilt_array, original_array)

    def test_several_patches_round_trip_in_order(self) -> None:
        end = Projected(ORIGIN.easting_km + 10.0, ORIGIN.northing_km + 10.0)
        far = Projected(end.easting_km + 8.0, end.northing_km + 2.0)
        fault = Fault(
            ORIGIN,
            [
                Plane(end, dip_deg=70.0, bottom_depth_km=14.0),
                Plane(far, dip_deg=50.0, bottom_depth_km=10.0),
            ],
        )
        original = build_fault_mesh(fault, [Cuts(10, 8), Cuts(6, 4)])
        rebuilt = RefinedMesh.from_positions(
            original.origin,
            [original.node_positions(0), original.node_positions(1)],
        )

        assert rebuilt.patch_count == 2
        assert rebuilt.cell_extents(0) == (10, 8)
        assert rebuilt.cell_extents(1) == (6, 4)
        # The two planes have different dips, which is the thing a single fused grid
        # could not represent. Check they did not get mixed up.
        assert rebuilt.dip_deg(0) == pytest.approx(70.0, abs=1e-9)
        assert rebuilt.dip_deg(1) == pytest.approx(50.0, abs=1e-9)

    def test_a_ragged_patch_is_refused(self) -> None:
        east = np.zeros((3, 4))
        assert_refused = pytest.raises(ValueError, match="same shape")
        with assert_refused:
            RefinedMesh.from_positions(ORIGIN, [(east, np.zeros((3, 5)), east)])

    @pytest.mark.parametrize("shape", [(1, 4), (4, 1), (1, 1)])
    def test_a_patch_with_no_cells_is_refused(self, shape: tuple[int, int]) -> None:
        """One row of nodes is a line, not a surface.

        Reachable only from a file: refinement always asks for at least one cell.
        """
        empty = np.zeros(shape)
        with pytest.raises(ValueError, match="need at least"):
            RefinedMesh.from_positions(ORIGIN, [(empty, empty, empty)])

    @pytest.mark.parametrize("patch", [0, 1, 7])
    def test_a_patch_that_does_not_exist_is_refused(self, patch: int) -> None:
        mesh = a_mesh()
        if patch < mesh.patch_count:
            pytest.skip("that patch exists")
        with pytest.raises(IndexError, match="patch"):
            mesh.areas_km2(patch)


class TestRefusesBadInput:
    @pytest.mark.parametrize("dip_deg", [0.0, -10.0, 90.5, 120.0])
    def test_a_dip_off_a_fault_plane_is_refused(self, dip_deg: float) -> None:
        with pytest.raises(ValueError, match="dip"):
            build_fault_mesh(a_plane(dip_deg=dip_deg), [Cuts(4, 2)])

    def test_a_fault_with_no_planes_is_refused(self) -> None:
        """The one invariant Python can break that Rust cannot.

        `genslip::mesh::Fault` holds its first plane separately from the rest, so an
        empty fault has no representation there. A Python list has no such shape, so
        this boundary is the only place the check can live.
        """
        with pytest.raises(ValueError, match="at least one plane"):
            Fault(ORIGIN, [])

    def test_an_inverted_depth_range_is_refused(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            build_fault_mesh(
                a_plane(top_depth_km=12.0, bottom_depth_km=4.0), [Cuts(4, 2)]
            )

    def test_a_fault_above_the_ground_is_refused(self) -> None:
        with pytest.raises(ValueError, match="above the ground"):
            build_fault_mesh(a_plane(top_depth_km=-1.0), [Cuts(4, 2)])

    def test_a_repeated_trace_point_is_refused(self) -> None:
        fault = Fault(ORIGIN, [Plane(ORIGIN, dip_deg=60.0, bottom_depth_km=12.0)])
        with pytest.raises(ValueError, match="strictly positive"):
            build_fault_mesh(fault, [Cuts(4, 2)])

    @pytest.mark.parametrize("cuts", [(0, 4), (4, 0)])
    def test_a_face_cut_into_nothing_is_refused(self, cuts: tuple[int, int]) -> None:
        with pytest.raises(ValueError, match="need at least"):
            build_fault_mesh(a_plane(), [Cuts(*cuts)])

    @pytest.mark.parametrize("count", [0, 2, 3])
    def test_the_cuts_must_match_the_planes(self, count: int) -> None:
        with pytest.raises(ValueError, match="cuts"):
            build_fault_mesh(a_plane(), [Cuts(4, 2)] * count)

    @pytest.mark.parametrize(
        ("strike_km", "dip_km"), [(-0.1, 1.0), (25.0, 1.0), (1.0, -0.1), (1.0, 500.0)]
    )
    def test_a_hypocentre_off_the_fault_is_refused(
        self, strike_km: float, dip_km: float
    ) -> None:
        with pytest.raises(ValueError, match="outside a fault"):
            a_mesh().cell_index(0, strike_km, dip_km)


class TestTheSeamAssembleAsksFor:
    """`to_subfault_geometry`, which is what this whole module was built to reach.

    `assemble.py` has said since it was written that the subfault coordinates *"arrive in
    `SubfaultGeometry`, from whoever discretised the fault, because that is the only
    place that knows how the mesh became a grid"*. Nothing supplied them. This does.
    """

    @given(**EXTENTS)
    @SETTINGS
    def test_it_produces_one_value_per_subfault(
        self, strike_count: int, dip_count: int
    ) -> None:
        mesh = a_mesh(strike_count, dip_count)
        geometry = to_subfault_geometry(mesh, 0, NZTM)

        assert isinstance(geometry, SubfaultGeometry)
        subfaults = strike_count * dip_count
        for name in (
            "longitude_deg",
            "latitude_deg",
            "depth_km",
            "strike_deg",
            "dip_deg",
            "area_cm2",
        ):
            assert len(getattr(geometry, name)) == subfaults, name

    @given(**EXTENTS)
    @SETTINGS
    def test_it_flattens_along_strike_fastest(
        self, strike_count: int, dip_count: int
    ) -> None:
        """The order every field in the core is produced in, and an SRF stores.

        Depth is the field that shows it: flattened strike-fastest, it repeats each dip
        row's depth `strike_count` times before changing. Flattened the other way it
        would cycle through every depth once per subfault along strike.
        """
        mesh = a_mesh(strike_count, dip_count)
        geometry = to_subfault_geometry(mesh, 0, NZTM)
        rows = geometry.depth_km.reshape(dip_count, strike_count)

        assert np.allclose(np.diff(rows, axis=1), 0.0), "depth varies along a row"
        if dip_count > 1:
            assert np.all(np.diff(rows, axis=0) > 0.0), "depth does not increase down"

    @given(**EXTENTS)
    @SETTINGS
    def test_area_arrives_in_the_square_centimetres_the_format_stores(
        self, strike_count: int, dip_count: int
    ) -> None:
        """The unit conversion, which is the kind of thing that is invisible until it is
        a factor of ten billion.

        `README.md` records the sibling of this going wrong the other way: *"the SRF
        stored `vs` in cm/s while the port wrote km/s"*, silently.
        """
        mesh = a_mesh(strike_count, dip_count)
        geometry = to_subfault_geometry(mesh, 0, NZTM)
        assert geometry.area_cm2 == pytest.approx(
            mesh.areas_km2(0).ravel() * CM2_PER_KM2, rel=1e-12
        )

    def test_the_nodes_convert_too(self) -> None:
        """The corners, which are what a renderer and a format want.

        `project_patch` gives cell centres; this gives the mesh itself. One more of each
        along each axis, and bracketing the centres.
        """
        mesh = a_mesh(8, 5)
        longitude_deg, latitude_deg, depth_km = node_positions_wgs84(mesh, 0, NZTM)

        assert longitude_deg.shape == (6, 9)
        assert latitude_deg.shape == (6, 9)
        assert depth_km.shape == (6, 9)

        geometry = to_subfault_geometry(mesh, 0, NZTM)
        assert longitude_deg.min() < geometry.longitude_deg.min()
        assert longitude_deg.max() > geometry.longitude_deg.max()
        assert depth_km.min() < geometry.depth_km.min()
        assert depth_km.max() > geometry.depth_km.max()


# Deliberately not asserted:
#
# - The geometry identities themselves. `crates/genslip/tests/mesh.rs` has them, as exact
#   statements in the frame they hold in; repeating them here would be a transcription
#   that agrees with the code for the same reason the code is wrong, if it is.
# - Anything about longitude and latitude beyond shape and ordering. That is
#   `tests/test_projection.py`, which checks it against `pyproj.Geod` rather than against
#   the projection that produced it.
