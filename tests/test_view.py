import numpy as np
import pytest

from rupture_generator.formats.srf import read_rupture

pytest.importorskip("rerun")

from rupture_generator.view import (
    contour_levels,
    isochrones,
    main,
    positions_m,
    slip_by,
    slip_directions,
)


@pytest.fixture(scope="module")
def rupture(srf_path):
    return read_rupture(srf_path)


def test_quads_have_the_files_areas(rupture):
    corners, _ = positions_m(rupture)
    areas_m2 = 0.5 * np.linalg.norm(
        np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 1]), axis=-1
    )
    np.testing.assert_allclose(areas_m2, rupture.area_m2, rtol=1e-5)


def test_slip_accumulates_to_the_files(rupture):
    before, after = slip_by(rupture, [-1.0, 1e6])
    pulsed = np.diff(rupture.pulse_offsets) > 0
    assert np.all(before == 0.0)
    np.testing.assert_allclose(after[pulsed], rupture.slip_m[pulsed], rtol=1e-6)


def test_the_hypocentre_is_drawn(rupture):
    assert positions_m(rupture)[1] is not None


def test_a_recording_is_written(srf_path, tmp_path):
    recording = tmp_path / "rupture.rrd"
    assert main([str(srf_path), "--save", str(recording)]) == 0
    assert recording.stat().st_size > 0


def test_contours_are_round_and_skip_the_start():
    np.testing.assert_allclose(contour_levels(0.0, 31.0), [5, 10, 15, 20, 25, 30])
    assert contour_levels(3.0, 3.0).size == 0


def test_an_isochrone_lies_on_its_level():
    i, j = np.mgrid[0:5, 0:7].astype(float)
    positions = np.stack([j, i, np.zeros_like(i)], axis=-1)
    lines = isochrones(j + 0.1 * i, positions, 3.2)
    assert len(lines)
    np.testing.assert_allclose(lines[..., 0] + 0.1 * lines[..., 1], 3.2)


def test_slip_directions_are_unit_and_in_plane(rupture):
    corners, _ = positions_m(rupture)
    directions = slip_directions(rupture, corners)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0])
    np.testing.assert_allclose(np.linalg.norm(directions, axis=-1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(
        np.einsum(
            "ij,ij->i", directions, normals / np.linalg.norm(normals, axis=-1)[:, None]
        ),
        0.0,
        atol=1e-6,
    )
