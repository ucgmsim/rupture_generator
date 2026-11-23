import dataclasses
import os
import threading

import numpy as np
import pyproj
import pytest
import srf as srf_format

from rupture_generator import (
    Geometry,
    Hypocentre,
    Realisation,
    RuptureGeneratorError,
    SegmentRupture,
)
from rupture_generator.formats.srf import (
    NO_HYPOCENTRE,
    _plane_slices,
    read_rupture,
    write_rupture,
)


@pytest.fixture(scope="module")
def srf(srf_path):
    return srf_path, srf_format.read(srf_path)


def column(ruptures, pick):
    return np.concatenate(
        [
            pick(r)[:, columns].ravel()
            for r in ruptures.values()
            for columns in _plane_slices(r.geometry)
        ]
    )


def test_is_version_2_with_a_points_block_per_plane(srf, ruptures):
    path, parsed = srf
    text = path.read_text()
    planes = sum(r.geometry.planes for r in ruptures.values())
    assert text.startswith("2.0\n")
    assert text.count("POINTS ") == planes == len(parsed.planes)


def test_only_the_hypocentre_plane_has_a_hypocentre(srf):
    _, parsed = srf
    located = [plane for plane in parsed.planes if plane.shyp != NO_HYPOCENTRE]
    assert len(located) == 1
    assert all(
        plane.dhyp == NO_HYPOCENTRE for plane in parsed.planes if plane not in located
    )


def test_points_round_trip(srf, ruptures):
    _, parsed = srf
    points = parsed.points
    assert points.vs is not None and points.density is not None
    slip_cm = column(ruptures, lambda r: r.slip_m) * 100.0
    # The SRF records a point whose pulse has no samples as not slipping, and only slip
    # too small to resolve leaves a pulse empty.
    empty = np.diff(parsed.slip_rate.offsets) == 0
    assert np.all(slip_cm[empty] < 0.1)
    np.testing.assert_allclose(points.slip1, np.where(empty, 0.0, slip_cm), rtol=1e-6)
    np.testing.assert_allclose(
        points.tinit, column(ruptures, lambda r: r.onset_s), atol=1e-5
    )
    np.testing.assert_allclose(
        points.vs,
        column(ruptures, lambda r: r.shear_speed_km_s) * 1e5,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        points.density,
        column(ruptures, lambda r: r.density_g_cm3),
        rtol=1e-6,
    )


def test_moment_from_the_file_alone(srf, scenario):
    _, parsed = srf
    points = parsed.points
    assert points.vs is not None and points.density is not None
    rigidity = (
        points.density.astype(float) * 1e3 * (points.vs.astype(float) * 1e-2) ** 2
    )
    moment = np.sum(rigidity * points.area * 1e-4 * points.slip1 * 1e-2)
    expected = sum(source.moment_nm for source in scenario.sources.values())
    assert moment == pytest.approx(expected, rel=1e-5)


def test_pulses_integrate_to_slip(srf, scenario):
    _, parsed = srf
    row_ptr = parsed.slip_rate.offsets.astype(np.int64)
    rows = np.diff(row_ptr)
    integral = (
        np.add.reduceat(parsed.slip_rate.values.astype(float), row_ptr[:-1][rows > 0])
        * scenario.model.pulse.dt_s
    )
    np.testing.assert_allclose(integral, parsed.points.slip1[rows > 0], rtol=1e-5)


def test_positions_are_the_charts(srf, scenario, ruptures):
    _, parsed = srf
    to_lon_lat = pyproj.Transformer.from_crs(
        scenario.realisation.crs, "EPSG:4326", always_xy=True
    )
    east = column(ruptures, lambda r: r.geometry.centres[..., 0])
    north = column(ruptures, lambda r: r.geometry.centres[..., 1])
    lon, lat = to_lon_lat.transform(east * 1000, north * 1000)
    np.testing.assert_allclose(parsed.points.lon, lon, atol=1e-4)
    np.testing.assert_allclose(parsed.points.lat, lat, atol=1e-4)


def test_the_reader_returns_si(srf_path, srf, scenario):
    _, parsed = srf
    rupture = read_rupture(srf_path)
    np.testing.assert_allclose(rupture.slip_m, parsed.points.slip1 / 100.0)
    np.testing.assert_allclose(rupture.area_m2, parsed.points.area * 1e-4)
    moment = np.sum(rupture.rigidity_pa * rupture.area_m2 * rupture.slip_m)
    expected = sum(source.moment_nm for source in scenario.sources.values())
    assert moment == pytest.approx(expected, rel=1e-5)


def test_the_reader_finds_the_hypocentre(srf_path, scenario):
    hypocentre = scenario.realisation.hypocentre
    chart = scenario.realisation[hypocentre.segment]
    i, j = chart.cell_at(hypocentre.strike_km, hypocentre.dip_km)
    to_lon_lat = pyproj.Transformer.from_crs(
        scenario.realisation.crs, "EPSG:4326", always_xy=True
    )
    east, north, depth_km = chart.centres[i, j]
    lon, lat = to_lon_lat.transform(east * 1000, north * 1000)
    found = read_rupture(srf_path).hypocentre
    assert found is not None
    _, _, distance_m = pyproj.Geod(ellps="WGS84").inv(lon, lat, found[0], found[1])
    spacing_km = max(chart.spacing_km)
    assert distance_m < 1000 * spacing_km
    assert abs(found[2] - depth_km) < spacing_km


def _steep(ruptures):
    """The ruptures with a rising fraction the kernel refuses."""
    return ruptures.with_segments(
        {
            name: dataclasses.replace(r, beta=np.full(r.geometry.cells, 0.7))
            for name, r in ruptures.items()
        }
    )


def test_a_pulse_the_kernel_refuses_is_a_rupture_error(tmp_path, ruptures):
    with pytest.raises(RuptureGeneratorError, match="beta"):
        write_rupture(str(tmp_path / "bad.srf"), _steep(ruptures))
    # The writer had already put the header out. The failure removed the partial file.
    assert not list(tmp_path.iterdir())


def test_a_failed_write_leaves_the_existing_file(tmp_path, ruptures):
    path = tmp_path / "kept.srf"
    path.write_text("an earlier rupture")
    with pytest.raises(RuptureGeneratorError):
        write_rupture(path, _steep(ruptures))
    assert path.read_text() == "an earlier rupture"
    assert list(tmp_path.iterdir()) == [path]


def test_points_the_writer_refuses_are_a_rupture_error(
    tmp_path, scenario, ruptures, monkeypatch
):
    def refuse(self, points, slip_rate):
        raise srf_format.SrfError("The points run past what the planes declare")

    monkeypatch.setattr(srf_format.Writer, "write", refuse)
    path = tmp_path / "refused.srf"
    with pytest.raises(RuptureGeneratorError, match=f"^{path}: The points"):
        write_rupture(path, ruptures)
    assert not list(tmp_path.iterdir())


def test_a_file_that_is_not_an_srf_is_a_rupture_error(tmp_path):
    path = tmp_path / "not.srf"
    path.write_text("garbage")
    with pytest.raises(RuptureGeneratorError) as caught:
        read_rupture(path)
    assert str(caught.value).count(str(path)) == 1


def test_the_reader_reports_pulse_duration_not_rise_time(srf_path):
    rupture = read_rupture(srf_path)
    np.testing.assert_allclose(
        rupture.pulse_duration_s, np.diff(rupture.pulse_offsets) * rupture.dt_s
    )
    assert not hasattr(rupture, "rise_time_s")


def test_the_reader_takes_a_pipe(srf_path, tmp_path):
    fifo = tmp_path / "rupture.fifo"
    os.mkfifo(fifo)
    contents = srf_path.read_bytes()
    feeder = threading.Thread(target=fifo.write_bytes, args=(contents,))
    feeder.start()
    try:
        piped = read_rupture(fifo)
    finally:
        feeder.join()
    np.testing.assert_array_equal(piped.slip_m, read_rupture(srf_path).slip_m)


def test_a_record_the_reader_refuses_names_its_file(srf_path, monkeypatch):
    # srf.read puts the path in a parse error, and leaves it out of an error from the
    # checks on the records it builds from the parse.
    def refuse(source):
        raise srf_format.SrfError("The planes declare 3 points, but there are 2")

    monkeypatch.setattr(srf_format, "read", refuse)
    with pytest.raises(RuptureGeneratorError, match=f"^{srf_path}: The planes"):
        read_rupture(srf_path)


def test_a_point_without_a_pulse_records_no_slip(tmp_path, scenario, ruptures):
    # The kernel leaves the pulse empty below its slip floor. slip1 must agree.
    faint = {}
    for name, rupture in ruptures.items():
        slip_m = rupture.slip_m.copy()
        slip_m[0, :] = 5e-5
        faint[name] = dataclasses.replace(rupture, slip_m=slip_m)
    path = tmp_path / "faint.srf"
    write_rupture(path, ruptures.with_segments(faint))
    parsed = srf_format.read(path)
    empty = np.diff(parsed.slip_rate.offsets) == 0
    assert empty.any()
    assert np.all(parsed.points.slip1[empty] == 0.0)
    assert np.all(parsed.points.slip1[~empty] > 0.0)


def test_a_rise_time_under_half_a_sample_slips_within_one_sample(tmp_path):
    from rupture_generator import config, generate

    from .conftest import EXAMPLE, SPACING_KM

    # Small, linear in an uncorrelated marginal with density at zero: some slipping
    # subfault draws a rise time that rounds to no samples at all.
    loaded = config.load(EXAMPLE)
    scenario = config.build(
        dataclasses.replace(
            loaded,
            seed=3,
            geometry=dataclasses.replace(loaded.geometry, spacing_km=SPACING_KM),
            source=dataclasses.replace(
                loaded.source, magnitudes={"bent": 4.5, "next": 4.3}
            ),
            rise=dataclasses.replace(
                loaded.rise,
                family="truncated_normal",
                coefficient_of_variation=0.5,
                slip_exponent=1.0,
                correlation=0.0,
            ),
        )
    )
    drawn = generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        scenario.model,
        seed=scenario.seed,
        jump_propagation_model=scenario.jump_model,
    )
    dt_s = loaded.pulse.dt_s
    slip_m = column(drawn, lambda r: r.slip_m)
    rise_s = column(drawn, lambda r: r.rise_time_s)
    # 1e-4 m is the kernel's `pulse::MIN_SLIP_M`, under which a point has no pulse.
    short = (slip_m > 1.0e-4) & (rise_s < 0.5 * dt_s)
    assert short.any()

    path = tmp_path / "short.srf"
    write_rupture(path, drawn)
    parsed = srf_format.read(path)
    offsets = parsed.slip_rate.offsets.astype(np.int64)
    rates = parsed.slip_rate.values.astype(float)
    for point in np.flatnonzero(short):
        pulse = rates[offsets[point] : offsets[point + 1]]
        # The one-sample spike [0, slip/dt, 0], which keeps all of the slip.
        assert pulse.size == 3 and pulse[0] == pulse[2] == 0.0
        assert pulse.sum() * dt_s == pytest.approx(parsed.points.slip1[point], rel=1e-6)
        assert parsed.points.slip1[point] == pytest.approx(
            slip_m[point] * 100.0, rel=1e-6
        )


# ------------------------------------------------------------ charts in part fault

NZTM = pyproj.CRS("EPSG:2193")
TO_LON_LAT = pyproj.Transformer.from_crs(NZTM, "EPSG:4326", always_xy=True)
ORIGIN_KM = np.array([1600.0, 5300.0, 0.0])


def _drawn(chart, cell=None):
    """A rupture slipping 1 m on every occupied cell, starting at ``cell``'s centre."""
    cells = chart.cells
    ones = np.ones(cells)
    rupture = SegmentRupture(
        geometry=chart,
        slip_m=np.where(chart.occupied, 1.0, 0.0),
        rise_time_s=ones,
        rake_deg=np.full(cells, 90.0),
        onset_s=np.zeros(cells),
        speed_km_s=2.8 * ones,
        shear_speed_km_s=3.5 * ones,
        density_g_cm3=2.7 * ones,
        dt_s=0.05,
        beta=None,
    )
    hypocentre = None
    if cell is not None:
        i, j = cell
        hypocentre = Hypocentre(
            "a",
            0.5 * float(chart.strike_arc_km[j] + chart.strike_arc_km[j + 1]),
            0.5 * float(chart.dip_arc_km[i] + chart.dip_arc_km[i + 1]),
        )
    return Realisation({"a": rupture}, NZTM, hypocentre=hypocentre)


def _north_dipping_east(occupied):
    """1 km cells along frame north, dipping 45 degrees east, ``occupied`` as given."""
    rows, columns = occupied.shape
    i, j = np.mgrid[0 : rows + 1, 0 : columns + 1].astype(float)
    step = np.sqrt(0.5)
    nodes = ORIGIN_KM + np.stack([step * i, j, step * i], axis=-1)
    return Geometry(nodes, occupied, (columns,))


def test_a_header_summarises_only_the_occupied_cells(tmp_path):
    occupied = np.array(
        [
            [False, False, False, False],
            [False, True, True, True],
            [True, True, True, False],
        ]
    )
    chart = _north_dipping_east(occupied)
    path = tmp_path / "holed.srf"
    write_rupture(path, _drawn(chart, cell=(1, 2)))
    parsed = srf_format.read(path)
    (plane,) = parsed.planes
    step = np.sqrt(0.5)
    assert plane.dtop == pytest.approx(step, abs=1e-6)
    assert plane.dip == pytest.approx(45.0, abs=1e-4)
    assert (plane.nstk, plane.ndip) == (4, 3)
    # The top outline runs 1 km along row 2, 0.71 km up to row 1 and 3 km along it,
    # so its middle lies on row 1.
    half_km = 0.5 * (4.0 + step)
    lon, lat = TO_LON_LAT.transform(
        (ORIGIN_KM[0] + step) * 1000.0, (ORIGIN_KM[1] + half_km - step) * 1000.0
    )
    assert (plane.elon, plane.elat) == pytest.approx((lon, lat), abs=1e-5)
    # Every cell strikes along frame north, which is true north turned by the
    # meridian convergence. The header's strike is their mean over the fault.
    convergence_deg = (
        pyproj.Proj(NZTM)
        .get_factors(parsed.points.lon, parsed.points.lat)
        .meridian_convergence
    )
    np.testing.assert_allclose(parsed.points.stk, convergence_deg % 360.0, atol=1e-3)
    on_fault = parsed.points.stk[occupied.ravel()].astype(float)
    assert plane.stk == pytest.approx(on_fault.mean(), abs=1e-4)
    # Only the occupied cells slip, and the hypocentre reads back at its cell.
    assert np.array_equal(parsed.points.slip1 > 0, occupied.ravel())
    east, north, depth_km = chart.centres[1, 2]
    found = read_rupture(path).hypocentre
    assert found is not None
    centre = TO_LON_LAT.transform(east * 1000.0, north * 1000.0)
    _, _, distance_m = pyproj.Geod(ellps="WGS84").inv(*centre, found[0], found[1])
    assert distance_m < 1.0
    assert found[2] == pytest.approx(depth_km, abs=1e-3)


def test_nodes_off_the_fault_leave_the_header_alone(tmp_path):
    occupied = np.ones((3, 4), bool)
    occupied[0, 0] = occupied[2, 3] = False
    chart = _north_dipping_east(occupied)
    # Corners (0, 0) and (3, 4) belong to no occupied cell.
    nodes = chart.nodes.copy()
    nodes[0, 0] += [-5.0, -5.0, -3.0]
    nodes[3, 4] += [5.0, 5.0, 30.0]
    moved = Geometry(nodes, occupied, chart.plane_cells)
    headers = []
    for name, geometry in [("still", chart), ("moved", moved)]:
        write_rupture(tmp_path / name, _drawn(geometry))
        headers.append(srf_format.read(tmp_path / name).planes[0])
    for field in ("elon", "elat", "dtop", "dip", "stk"):
        assert getattr(headers[0], field) == getattr(headers[1], field), field


def test_each_point_strikes_along_its_own_cell(tmp_path):
    # A chart hung from an arc of 10 km radius, turning through 60 degrees, with its
    # rows dipping 60 degrees in toward the arc's centre.
    turn_rad = np.radians(np.linspace(0.0, 60.0, 9))
    rim = np.stack([np.sin(turn_rad), np.cos(turn_rad), np.zeros_like(turn_rad)], -1)
    down = -np.tan(np.radians(30.0)) * rim + [0.0, 0.0, 1.0]
    nodes = ORIGIN_KM + 10.0 * rim + np.arange(3.0)[:, None, None] * down
    chart = Geometry(nodes, np.ones((2, 8), bool), (8,))
    path = tmp_path / "arc.srf"
    write_rupture(path, _drawn(chart))
    stk = srf_format.read(path).points.stk.reshape(2, 8)

    geod = pyproj.Geod(ellps="WGS84")
    lon, lat = TO_LON_LAT.transform(nodes[0, :, 0] * 1000.0, nodes[0, :, 1] * 1000.0)
    forward, back, _ = geod.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    # The geodesic's azimuth halfway along each cell's top edge.
    turn = (np.asarray(back) + 180.0 - forward + 180.0) % 360.0 - 180.0
    expected = (np.asarray(forward) + 0.5 * turn) % 360.0
    assert np.ptp(expected) > 50.0
    for row in stk:
        np.testing.assert_allclose(
            (row - expected + 180.0) % 360.0 - 180.0, 0.0, atol=0.01
        )


def test_a_plane_with_no_fault_is_refused(tmp_path):
    occupied = np.ones((1, 4), bool)
    occupied[:, 2:] = False
    whole = _north_dipping_east(occupied)
    chart = Geometry(whole.nodes, occupied, (2, 2))
    with pytest.raises(RuptureGeneratorError, match="^a: plane 1 has no occupied"):
        write_rupture(tmp_path / "empty.srf", _drawn(chart))
    assert not list(tmp_path.iterdir())


def test_a_chart_that_is_fault_everywhere_heads_each_plane_from_its_top_edge(
    srf, ruptures
):
    _, parsed = srf
    planes = iter(parsed.planes)
    for rupture in ruptures.values():
        geometry = rupture.geometry
        assert geometry.occupied.all()
        for columns, nodes in zip(
            _plane_slices(geometry), geometry.plane_nodes(), strict=True
        ):
            plane = next(planes)
            middle = 0.5 * (nodes[0, 0] + nodes[0, -1])
            lon, lat = TO_LON_LAT.transform(middle[0] * 1000.0, middle[1] * 1000.0)
            assert (plane.elon, plane.elat) == pytest.approx((lon, lat), abs=1e-5)
            assert plane.dtop == pytest.approx(nodes[0, 0, 2], abs=1e-5)
            assert plane.dip == pytest.approx(
                geometry.dip_deg[:, columns].mean(), abs=1e-4
            )
