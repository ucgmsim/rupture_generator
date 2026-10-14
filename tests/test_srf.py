import dataclasses
import os
import threading

import numpy as np
import pyproj
import pytest
import srf as srf_format

from rupture_generator import RuptureGeneratorError
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
    np.testing.assert_allclose(points.slip1, slip_cm, rtol=1e-6)
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
