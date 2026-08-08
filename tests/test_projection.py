"""Leaving the projected frame, and the grid convergence that goes with it.

The library works in a Cartesian CRS and reports strike from the projection's *northing
axis*. Grid north is not true north, and away from a projection's central meridian the
two diverge by the convergence angle -- **5.04 degrees** in Fiordland on NZTM2000, five
times the one-degree rake bound in `ENGINEERING_RULES.md`. `rupture_generator.mesh` adds
it. This is what says it adds the right thing, with the right sign.

# The reference is `pyproj.Geod`, which knows nothing about the projection

`README.md`'s fourth trap: *"a reference side that re-implements the original is not a
reference"*. Checking `meridian_convergence` against a second call to
`meridian_convergence` would prove only that pyproj is deterministic.

So the reference here is the **geodesic** -- `Geod.fwd` and `Geod.inv` on the WGS84
ellipsoid, which never touch the projection. `test_a_short_fault_reports_its_true_azimuth`
is the whole point of the file: place a trace endpoint at a known *true* azimuth with
`Geod.fwd`, build a mesh through the CRS, and ask what strike comes back. Every step of
the projection, the convergence and the sign has to be right for that to close, and none
of them is checked against itself.

The fault is deliberately **short**. The first version of this used 20 km and failed by
0.06 degrees, and the reference was wrong rather than the code: a geodesic's azimuth
turns as it goes, so the far end of a long trace genuinely runs at a different true
bearing from the near end while its *grid* bearing stays constant. That is the
correction seen from the other side, and
`test_each_cell_reports_the_azimuth_of_the_trace_at_that_cell` pins it per cell.

# What a wrong sign costs

`true = grid - convergence` is a plausible-looking mistake and it is not subtle in its
consequences: it doubles the error rather than removing it, which is 10 degrees across
New Zealand and 1.4 degrees in Christchurch. On a map it looks fine. `the_sign_is_not_the_other_one`
fails on it explicitly, because a bound that merely *permits* the right answer would let
the wrong one through wherever the convergence happens to be small.
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
    Projected,
    RefinedMesh,
    build_fault_mesh,
)
from rupture_generator.mesh import (
    WGS84,
    grid_convergence_deg,
    project_patch,
    to_projected,
)

NZTM = pyproj.CRS("EPSG:2193")
"""NZTM2000, whose central meridian is 173 degrees east."""

NZTM_CENTRAL_MERIDIAN_DEG = 173.0

GEOD = pyproj.Geod(ellps="WGS84")

# Spread across New Zealand so the convergence spans its whole range, both signs and
# zero. Fiordland and East Cape are the extremes; Nelson sits on the central meridian.
PLACES = pytest.mark.parametrize(
    ("longitude_deg", "latitude_deg", "name"),
    [
        (166.0, -45.9, "fiordland"),
        (168.7, -45.0, "queenstown"),
        (170.5, -43.5, "aoraki"),
        (172.6, -43.5, "christchurch"),
        (173.0, -41.3, "on the central meridian"),
        (174.8, -36.9, "auckland"),
        (176.2, -37.7, "rotorua"),
        (178.5, -38.0, "east cape"),
    ],
)

# Hypothesis over NZTM's own domain of validity rather than the whole globe: a
# projection is a choice with a region attached, and asserting anything outside it would
# be asserting a property of an extrapolation.
IN_NEW_ZEALAND = {
    "longitude_deg": st.floats(min_value=166.0, max_value=179.0),
    "latitude_deg": st.floats(min_value=-47.5, max_value=-34.0),
}

SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def a_true_azimuth(
    longitude_deg: float, latitude_deg: float, azimuth_deg: float, distance_km: float
) -> tuple[float, float]:
    """Where you get to going `distance_km` along a **true** azimuth, geodetically."""
    longitude, latitude, _ = GEOD.fwd(
        longitude_deg, latitude_deg, azimuth_deg, distance_km * 1000.0
    )
    return longitude, latitude


def a_fault_along(
    longitude_deg: float,
    latitude_deg: float,
    azimuth_deg: float,
    length_km: float,
    cells: int = 8,
) -> RefinedMesh:
    """A one-plane fault whose trace runs along a true azimuth from a point."""
    end_lon, end_lat = a_true_azimuth(
        longitude_deg, latitude_deg, azimuth_deg, length_km
    )
    origin = Projected(*to_projected(NZTM, longitude_deg, latitude_deg))
    end = Projected(*to_projected(NZTM, end_lon, end_lat))
    fault = Fault(origin, [Plane(end, dip_deg=70.0, bottom_depth_km=12.0)])
    return build_fault_mesh(fault, [Cuts(cells, 4)])


class TestTheProjectionRoundTrips:
    """Before anything about angles, the positions have to survive the trip."""

    @PLACES
    def test_a_point_comes_back_where_it_started(
        self, longitude_deg: float, latitude_deg: float, name: str
    ) -> None:
        easting_km, northing_km = to_projected(NZTM, longitude_deg, latitude_deg)
        back = pyproj.Transformer.from_crs(NZTM, WGS84, always_xy=True).transform(
            easting_km * 1000.0, northing_km * 1000.0
        )
        # A metre is 1e-5 degrees; this is four orders inside that, so it is round-off
        # in the projection rather than a tolerance on the conversion.
        assert back[0] == pytest.approx(longitude_deg, abs=1e-9), name
        assert back[1] == pytest.approx(latitude_deg, abs=1e-9), name

    @given(**IN_NEW_ZEALAND)
    @SETTINGS
    def test_any_point_comes_back_where_it_started(
        self, longitude_deg: float, latitude_deg: float
    ) -> None:
        easting_km, northing_km = to_projected(NZTM, longitude_deg, latitude_deg)
        back = pyproj.Transformer.from_crs(NZTM, WGS84, always_xy=True).transform(
            easting_km * 1000.0, northing_km * 1000.0
        )
        assert back[0] == pytest.approx(longitude_deg, abs=1e-9)
        assert back[1] == pytest.approx(latitude_deg, abs=1e-9)

    @given(**IN_NEW_ZEALAND)
    @SETTINGS
    def test_the_projected_coordinate_is_in_kilometres(
        self, longitude_deg: float, latitude_deg: float
    ) -> None:
        """The unit conversion, which is invisible until something is a thousand times
        too big.

        NZTM eastings run 1,000-2,100 km and northings 4,700-6,200 km. In metres those
        would be six and seven digits longer, and the mesh would be built at a scale
        where a subfault is round-off.
        """
        easting_km, northing_km = to_projected(NZTM, longitude_deg, latitude_deg)
        assert 900.0 < easting_km < 2_500.0
        assert 4_600.0 < northing_km < 6_300.0


class TestTheConvergenceAngle:
    @given(**IN_NEW_ZEALAND)
    @SETTINGS
    def test_it_is_what_the_geodesic_says_grid_north_is(
        self, longitude_deg: float, latitude_deg: float
    ) -> None:
        """Step along grid north; the geodesic's azimuth for that step *is* the
        convergence.

        The definition, checked against a path that never touches the projection. Ten
        kilometres because the azimuth of a geodesic changes along it, so a longer step
        measures a different angle at the far end than at the near -- see the tolerance.
        """
        easting_km, northing_km = to_projected(NZTM, longitude_deg, latitude_deg)
        to_wgs84 = pyproj.Transformer.from_crs(NZTM, WGS84, always_xy=True)
        north_lon, north_lat = to_wgs84.transform(
            easting_km * 1000.0, (northing_km + 10.0) * 1000.0
        )
        geodetic_azimuth, _, _ = GEOD.inv(
            longitude_deg, latitude_deg, north_lon, north_lat
        )

        convergence = grid_convergence_deg(
            NZTM, np.array(longitude_deg), np.array(latitude_deg)
        )
        # The convergence is the angle at a *point*; the geodesic azimuth is measured
        # over a 10 km step, and the two differ by how much the azimuth turns along it.
        # Measured at 3.8e-3 degrees in the worst corner of the country, which is 260
        # times inside the 1 degree an SRF stores.
        assert float(convergence) == pytest.approx(geodetic_azimuth, abs=0.01)

    @pytest.mark.parametrize("latitude_deg", [-46.0, -43.0, -40.0, -37.0, -35.0])
    def test_it_vanishes_on_the_central_meridian(self, latitude_deg: float) -> None:
        """Zero where grid north *is* true north, at every latitude.

        The one place the correction provably does nothing, so it is the one place a
        spurious offset would show up unmixed with anything else.
        """
        convergence = grid_convergence_deg(
            NZTM,
            np.array(NZTM_CENTRAL_MERIDIAN_DEG),
            np.array(latitude_deg),
        )
        assert float(convergence) == pytest.approx(0.0, abs=1e-9)

    @given(
        longitude_deg=st.floats(min_value=166.0, max_value=172.5),
        latitude_deg=st.floats(min_value=-47.5, max_value=-34.0),
    )
    @SETTINGS
    def test_west_of_the_central_meridian_it_is_positive(
        self, longitude_deg: float, latitude_deg: float
    ) -> None:
        assert (
            float(
                grid_convergence_deg(
                    NZTM, np.array(longitude_deg), np.array(latitude_deg)
                )
            )
            > 0.0
        )

    @given(
        longitude_deg=st.floats(min_value=173.5, max_value=179.0),
        latitude_deg=st.floats(min_value=-47.5, max_value=-34.0),
    )
    @SETTINGS
    def test_east_of_the_central_meridian_it_is_negative(
        self, longitude_deg: float, latitude_deg: float
    ) -> None:
        assert (
            float(
                grid_convergence_deg(
                    NZTM, np.array(longitude_deg), np.array(latitude_deg)
                )
            )
            < 0.0
        )

    def test_it_reaches_five_degrees_in_fiordland(self) -> None:
        """The number the documentation quotes, pinned.

        `mesh.py` and `mesh.rs` both argue the correction is not optional by citing this
        against the one-degree rake bound. If it drifts, the argument drifts with it.
        """
        convergence = grid_convergence_deg(NZTM, np.array(166.0), np.array(-45.9))
        assert float(convergence) == pytest.approx(5.04, abs=0.01)


class TestStrikeCrossesCorrected:
    """The correction itself.

    # A long fault does not have one true strike, and that is not a defect

    The first version of this asserted that every cell of a 20 km fault laid along true
    azimuth `A` reports `A`. It fails by 0.06 degrees, and the reference was wrong rather
    than the code: **`A` is the azimuth at the *start* of the trace**. A geodesic's
    azimuth turns as it goes, by `dlon * sin(latitude)`, which over 20 km in New Zealand
    is about 0.13 degrees -- so the far end of the trace genuinely runs at a different
    true bearing from the near end, while its *grid* bearing is constant because a
    straight line in a projection is straight.

    That is the whole point of the correction, seen from the other side. So the tests
    split: a short fault pins the correction exactly, and a long one pins the variation
    against a per-cell geodetic reference.
    """

    @PLACES
    @pytest.mark.parametrize("azimuth_deg", [0.0, 45.0, 90.0, 137.0, 215.0, 300.0])
    def test_a_short_fault_reports_its_true_azimuth(
        self,
        longitude_deg: float,
        latitude_deg: float,
        name: str,
        azimuth_deg: float,
    ) -> None:
        """**The sharp instrument.**

        Two kilometres and one cell, so the geodesic's azimuth turns by under a
        thousandth of a degree and what is left is the projection and the convergence
        alone. The trace endpoint is placed by `Geod.fwd`; the mesh is built through
        NZTM; the correction comes from the projection's own factors. Nothing is checked
        against itself.

        Residual measured at 0.008 degrees, which is the projected chord against the
        geodesic over 2 km plus the convergence being evaluated half a cell down dip.
        Bounded at 0.02, and 50 times inside the whole degree an SRF stores.
        """
        mesh = a_fault_along(longitude_deg, latitude_deg, azimuth_deg, 2.0, cells=1)
        located = project_patch(mesh, 0, NZTM)

        reported = float(located.strike_deg[0, 0])
        gap = abs(reported - azimuth_deg) % 360.0
        assert min(gap, 360.0 - gap) < 0.02, f"{name}: {reported} vs {azimuth_deg}"

    @given(
        azimuth_deg=st.floats(min_value=0.0, max_value=359.9),
        **IN_NEW_ZEALAND,
    )
    @SETTINGS
    def test_any_short_fault_reports_its_true_azimuth(
        self, longitude_deg: float, latitude_deg: float, azimuth_deg: float
    ) -> None:
        mesh = a_fault_along(longitude_deg, latitude_deg, azimuth_deg, 2.0, cells=1)
        located = project_patch(mesh, 0, NZTM)
        reported = float(located.strike_deg[0, 0])
        gap = abs(reported - azimuth_deg) % 360.0
        assert min(gap, 360.0 - gap) < 0.02

    @PLACES
    @pytest.mark.parametrize("azimuth_deg", [45.0, 90.0, 300.0])
    def test_each_cell_reports_the_azimuth_of_the_trace_at_that_cell(
        self,
        longitude_deg: float,
        latitude_deg: float,
        name: str,
        azimuth_deg: float,
    ) -> None:
        """A long fault's cells each report their *own* true strike, and it is right.

        The reference is per cell and geodetic: a point on a geodesic has a forward
        azimuth toward the far end which *is* the geodesic's azimuth there, so
        `Geod.inv(cell, end)` gives the answer without the projection appearing anywhere.

        Top row only. Deeper cells sit down dip of the trace, so their convergence is
        evaluated a little to the side of it, and the reference above no longer describes
        where they are.
        """
        length_km = 20.0
        end_lon, end_lat = a_true_azimuth(
            longitude_deg, latitude_deg, azimuth_deg, length_km
        )
        mesh = a_fault_along(
            longitude_deg, latitude_deg, azimuth_deg, length_km, cells=8
        )
        located = project_patch(mesh, 0, NZTM)
        arc_km = mesh.strike_arc_km(0)

        for strike in range(mesh.cell_extents(0)[0]):
            along_km = 0.5 * (arc_km[strike] + arc_km[strike + 1])
            here_lon, here_lat = a_true_azimuth(
                longitude_deg, latitude_deg, azimuth_deg, along_km
            )
            expected, _, _ = GEOD.inv(here_lon, here_lat, end_lon, end_lat)

            reported = float(located.strike_deg[0, strike])
            gap = abs(reported - expected) % 360.0
            assert min(gap, 360.0 - gap) < 0.03, (
                f"{name} cell {strike}: {reported} vs {expected}"
            )

    @PLACES
    def test_the_sign_is_not_the_other_one(
        self, longitude_deg: float, latitude_deg: float, name: str
    ) -> None:
        """Subtracting the convergence instead of adding it must fail.

        A bound that merely permits the right answer would let the wrong sign through
        wherever the convergence is small. This asserts the wrong sign is *rejected*,
        which is only meaningful where the correction is big enough to see -- so the
        central-meridian case, where the two agree by construction, skips itself.
        """
        convergence = float(
            grid_convergence_deg(NZTM, np.array(longitude_deg), np.array(latitude_deg))
        )
        if abs(convergence) < 0.1:
            pytest.skip(f"{name}: convergence is {convergence:.3f}, too small to sign")

        mesh = a_fault_along(longitude_deg, latitude_deg, 30.0, 2.0, cells=1)
        located = project_patch(mesh, 0, NZTM)
        wrong = np.mod(mesh.strike_deg(0) - convergence, 360.0)

        assert abs(float(located.strike_deg[0, 0]) - 30.0) < 0.02
        assert abs(float(wrong[0, 0]) - 30.0) > 0.1, (
            f"{name}: the wrong sign is indistinguishable, so this proves nothing"
        )

    @given(**IN_NEW_ZEALAND)
    @SETTINGS
    def test_the_correction_is_exactly_the_convergence(
        self, longitude_deg: float, latitude_deg: float
    ) -> None:
        """True strike minus grid strike is the convergence at that subfault, to the bit.

        Separates *what* is added from *whether it is right*: the tests above say the
        convergence is the correct angle, and this says nothing else got added with it.
        """
        mesh = a_fault_along(longitude_deg, latitude_deg, 42.0, 10.0, cells=4)
        located = project_patch(mesh, 0, NZTM)
        expected = grid_convergence_deg(
            NZTM, located.longitude_deg, located.latitude_deg
        )
        difference = (
            np.mod(located.strike_deg - mesh.strike_deg(0) + 180.0, 360.0) - 180.0
        )
        assert difference == pytest.approx(expected, abs=1e-12)


class TestDipAndAreaCrossUnchanged:
    """The two the projection is *not* allowed to move, asserted rather than assumed."""

    @PLACES
    @pytest.mark.parametrize("dip_deg", [15.0, 45.0, 70.0, 90.0])
    def test_dip_is_untouched(
        self, longitude_deg: float, latitude_deg: float, name: str, dip_deg: float
    ) -> None:
        end_lon, end_lat = a_true_azimuth(longitude_deg, latitude_deg, 55.0, 20.0)
        fault = Fault(
            Projected(*to_projected(NZTM, longitude_deg, latitude_deg)),
            [
                Plane(
                    Projected(*to_projected(NZTM, end_lon, end_lat)),
                    dip_deg=dip_deg,
                    bottom_depth_km=12.0,
                )
            ],
        )
        mesh = build_fault_mesh(fault, [Cuts(8, 4)])
        located = project_patch(mesh, 0, NZTM)
        assert located.dip_deg == pytest.approx(mesh.dip_deg(0), abs=0.0)
        assert located.dip_deg.ravel() == pytest.approx(dip_deg, abs=1e-9), name

    @PLACES
    def test_area_is_untouched(
        self, longitude_deg: float, latitude_deg: float, name: str
    ) -> None:
        mesh = a_fault_along(longitude_deg, latitude_deg, 55.0, 20.0)
        located = project_patch(mesh, 0, NZTM)
        assert located.area_km2 == pytest.approx(mesh.areas_km2(0), abs=0.0), name


# Deliberately not asserted:
#
# - That NZTM's area distortion is corrected for. It is not, on purpose: the fault's
#   area is the one the modeller specified in the CRS they chose for that region, and
#   reinterpreting it here would be a second opinion about a number nobody asked to have
#   reinterpreted. `mesh.py:project_patch` says so at the site.
# - Anything outside NZTM's domain. A projection is a choice with a region attached, and
#   a property asserted at longitude 20 would be a property of an extrapolation.
