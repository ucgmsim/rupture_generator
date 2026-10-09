import numpy as np
import pyproj
import pytest

from rupture_generator.formats.srf import NO_HYPOCENTRE, _plane_major, write_rupture
from rupture_generator.srf_parser import parse_srf


@pytest.fixture(scope="module")
def srf(tmp_path_factory, scenario, ruptures):
    path = tmp_path_factory.mktemp("srf") / "rupture.srf"
    write_rupture(
        str(path),
        scenario.realisation,
        ruptures,
        scenario.materials,
        dt_s=scenario.dt_s,
        beta=scenario.beta,
    )
    return path, parse_srf(path.read_bytes())


def column(ruptures, materials, pick):
    return np.concatenate(
        [
            _plane_major(r.geometry, pick(r, materials[name]))
            for name, r in ruptures.items()
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


def test_points_round_trip(srf, scenario, ruptures):
    _, parsed = srf
    metadata = parsed.metadata
    materials = scenario.materials
    slip_cm = column(ruptures, materials, lambda r, m: r.slip_m) * 100.0
    np.testing.assert_allclose(metadata.slip1, slip_cm, rtol=1e-6)
    np.testing.assert_allclose(
        metadata.tinit, column(ruptures, materials, lambda r, m: r.onset_s), atol=1e-5
    )
    np.testing.assert_allclose(
        metadata.vs,
        column(ruptures, materials, lambda r, m: m.shear_speed_km_s) * 1e5,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        metadata.density,
        column(ruptures, materials, lambda r, m: m.density_g_cm3),
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
    east = column(ruptures, scenario.materials, lambda r, m: r.geometry.centres[..., 0])
    north = column(
        ruptures, scenario.materials, lambda r, m: r.geometry.centres[..., 1]
    )
    lon, lat = to_lon_lat.transform(east * 1000, north * 1000)
    np.testing.assert_allclose(parsed.metadata.lon, lon, atol=1e-4)
    np.testing.assert_allclose(parsed.metadata.lat, lat, atol=1e-4)
