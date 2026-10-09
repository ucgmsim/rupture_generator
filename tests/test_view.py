import numpy as np
import pytest

from rupture_generator.formats.srf import read_rupture

pytest.importorskip("rerun")

from rupture_generator.view import main, positions_m, slip_by


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
