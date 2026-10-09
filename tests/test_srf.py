import numpy as np
import pyproj
import pytest

from rupture_generator import RuptureGeneratorError
from rupture_generator.formats.srf import (
    NO_HYPOCENTRE,
    _plane_major,
    read_rupture,
    write_rupture,
)
from rupture_generator.srf_parser import parse_srf


@pytest.fixture(scope="module")
def srf(srf_path):
    return srf_path, parse_srf(srf_path.read_bytes())


def column(ruptures, pick):
    return np.concatenate(
        [_plane_major(r.geometry, pick(r)) for r in ruptures.values()]
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
    metadata = parsed.metadata
    slip_cm = column(ruptures, lambda r: r.slip_m) * 100.0
    np.testing.assert_allclose(metadata.slip1, slip_cm, rtol=1e-6)
    np.testing.assert_allclose(
        metadata.tinit, column(ruptures, lambda r: r.onset_s), atol=1e-5
    )
    np.testing.assert_allclose(
        metadata.vs,
        column(ruptures, lambda r: r.shear_speed_km_s) * 1e5,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metadata.density,
        column(ruptures, lambda r: r.density_g_cm3),
        rtol=1e-6,
    )


def test_moment_from_the_file_alone(srf, scenario):
    _, parsed = srf
    metadata = parsed.metadata
    rigidity = (
        metadata.density.astype(float) * 1e3 * (metadata.vs.astype(float) * 1e-2) ** 2
    )
    moment = np.sum(rigidity * metadata.area * 1e-4 * metadata.slip1 * 1e-2)
    expected = sum(source.moment_nm for source in scenario.sources.values())
    assert moment == pytest.approx(expected, rel=1e-5)


def test_pulses_integrate_to_slip(srf, scenario):
    _, parsed = srf
    row_ptr = parsed.slipt1.row_ptr.astype(np.int64)
    rows = np.diff(row_ptr)
    integral = (
        np.add.reduceat(parsed.slipt1.data.astype(float), row_ptr[:-1][rows > 0])
        * scenario.dt_s
    )
    np.testing.assert_allclose(integral, parsed.metadata.slip1[rows > 0], rtol=1e-5)


def test_positions_are_the_charts(srf, scenario, ruptures):
    _, parsed = srf
    to_lon_lat = pyproj.Transformer.from_crs(
        scenario.realisation.crs, "EPSG:4326", always_xy=True
    )
    east = column(ruptures, lambda r: r.geometry.centres[..., 0])
    north = column(ruptures, lambda r: r.geometry.centres[..., 1])
    lon, lat = to_lon_lat.transform(east * 1000, north * 1000)
    np.testing.assert_allclose(parsed.metadata.lon, lon, atol=1e-4)
    np.testing.assert_allclose(parsed.metadata.lat, lat, atol=1e-4)


def test_the_reader_returns_si(srf_path, srf, scenario):
    _, parsed = srf
    rupture = read_rupture(srf_path)
    np.testing.assert_allclose(rupture.slip_m, parsed.metadata.slip1 / 100.0)
    np.testing.assert_allclose(rupture.area_m2, parsed.metadata.area * 1e-4)
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


def test_a_pulse_the_kernel_refuses_is_a_rupture_error(tmp_path, scenario, ruptures):
    beta = {name: np.full(r.geometry.cells, 0.7) for name, r in ruptures.items()}
    with pytest.raises(RuptureGeneratorError, match="beta"):
        write_rupture(
            str(tmp_path / "bad.srf"),
            scenario.realisation,
            ruptures,
            dt_s=scenario.dt_s,
            beta=beta,
        )


def test_a_file_that_is_not_an_srf_is_a_rupture_error(tmp_path):
    path = tmp_path / "not.srf"
    path.write_text("garbage")
    with pytest.raises(RuptureGeneratorError):
        read_rupture(path)
