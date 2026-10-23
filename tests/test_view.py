import dataclasses
import sys
import warnings

import numpy as np
import pytest

from rupture_generator.formats.srf import read_rupture, write_rupture

pytest.importorskip("rerun")

from rupture_generator.view import (
    HOT,
    VIRIDIS,
    _colours,
    _contour_levels,
    _isochrones,
    _onset_contours,
    _packed,
    _positions_m,
    _positive_seconds,
    _slip_by,
    _slip_directions,
    _statistics,
    main,
)


@pytest.fixture(scope="module")
def rupture(srf_path):
    return read_rupture(srf_path)


def test_quads_have_the_files_areas(rupture):
    corners, _ = _positions_m(rupture)
    areas_m2 = 0.5 * np.linalg.norm(
        np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 1]), axis=-1
    )
    np.testing.assert_allclose(areas_m2, rupture.area_m2, rtol=1e-5)


def test_slip_accumulates_to_the_files(rupture):
    before, after = _slip_by(rupture, [-1.0, 1e6])
    pulsed = np.diff(rupture.pulse_offsets) > 0
    assert np.all(before == 0.0)
    np.testing.assert_allclose(after[pulsed], rupture.slip_m[pulsed], rtol=1e-6)


def test_the_hypocentre_is_drawn(rupture):
    assert _positions_m(rupture)[1] is not None


def test_a_recording_is_written(srf_path, tmp_path):
    recording = tmp_path / "rupture.rrd"
    assert main([str(srf_path), "--save", str(recording)]) == 0
    assert recording.stat().st_size > 0


def test_contours_are_round_and_skip_the_start():
    np.testing.assert_allclose(_contour_levels(0.0, 31.0), [5, 10, 15, 20, 25, 30])
    assert _contour_levels(3.0, 3.0).size == 0


def test_an_isochrone_lies_on_its_level():
    i, j = np.mgrid[0:5, 0:7].astype(float)
    positions = np.stack([j, i, np.zeros_like(i)], axis=-1)
    lines = _isochrones(j + 0.1 * i, positions, 3.2)
    assert len(lines)
    np.testing.assert_allclose(lines[..., 0] + 0.1 * lines[..., 1], 3.2)


def test_slip_directions_are_unit_and_in_plane(rupture):
    corners, _ = _positions_m(rupture)
    directions = _slip_directions(rupture, corners)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0])
    np.testing.assert_allclose(np.linalg.norm(directions, axis=-1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(
        np.einsum(
            "ij,ij->i", directions, normals / np.linalg.norm(normals, axis=-1)[:, None]
        ),
        0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("step", ["0", "-1", "nan", "soon"])
def test_a_step_that_is_not_a_positive_time_is_refused(srf_path, step, capsys):
    with pytest.raises(SystemExit):
        main([str(srf_path), "--step", step])
    assert "--step" in capsys.readouterr().err


def test_a_rupture_without_slip_is_described_without_nan(tmp_path, ruptures):
    still = {
        name: dataclasses.replace(r, slip_m=np.zeros_like(r.slip_m))
        for name, r in ruptures.items()
    }
    path = tmp_path / "still.srf"
    write_rupture(path, ruptures.with_segments(still))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text = _statistics(read_rupture(path))
        assert main([str(path), "--save", str(tmp_path / "still.rrd")]) == 0
    assert "nan" not in text.lower()
    assert "No slip" in text


def test_channels_pack_as_rgba_with_opaque_alpha():
    packed = _packed(
        np.array([255.0, 1.4, 0.0]),
        np.array([0.0, 2.6, 128.0]),
        np.array([1.0, 3.0, 255.0]),
    )
    assert packed.dtype == np.uint32
    np.testing.assert_array_equal(packed, [0xFF0001FF, 0x010303FF, 0x0080FFFF])


def test_colour_maps_run_end_to_end():
    assert VIRIDIS.shape == HOT.shape == (256,)
    assert (VIRIDIS[0], VIRIDIS[-1]) == (0x440154FF, 0xFDE725FF)
    assert (HOT[0], HOT[-1]) == (0x000000FF, 0xFFFFFFFF)
    np.testing.assert_array_equal(
        _colours(np.array([-1.0, 0.0, 5.0, 10.0, 11.0]), 0.0, 10.0, HOT),
        HOT[[0, 0, 127, 255, 255]],
    )


def test_a_level_nothing_reaches_draws_no_isochrone():
    i, j = np.mgrid[0:3, 0:4].astype(float)
    positions = np.stack([j, i, np.zeros_like(i)], axis=-1)
    assert _isochrones(i + j, positions, 100.0).shape == (0, 2, 3)


def test_a_rupture_that_starts_everywhere_at_once_has_no_isochrones(rupture):
    sudden = dataclasses.replace(rupture, onset_s=np.full_like(rupture.onset_s, 2.0))
    corners, _ = _positions_m(sudden)
    lines, anchors, labels = _onset_contours(sudden, corners, 1.0)
    assert lines.shape == (0, 2, 3)
    assert anchors.shape == (0, 3)
    assert labels == []


def test_a_step_is_read_in_seconds():
    assert _positive_seconds("0.25") == 0.25
    assert _positive_seconds("1e-3") == 1e-3


def test_a_coarser_step_still_records(srf_path, tmp_path):
    recording = tmp_path / "coarse.rrd"
    assert main([str(srf_path), "--step", "5", "--save", str(recording)]) == 0
    assert recording.stat().st_size > 0


def test_an_unreadable_srf_exits_with_two(tmp_path, capsys):
    missing = tmp_path / "missing.srf"
    assert main([str(missing), "--save", str(tmp_path / "x.rrd")]) == 2
    assert capsys.readouterr().err.startswith("rupture-view: ")
    garbage = tmp_path / "garbage.srf"
    garbage.write_text("not an srf\n")
    assert main([str(garbage), "--save", str(tmp_path / "y.rrd")]) == 2
    assert capsys.readouterr().err.startswith("rupture-view: ")
    assert not (tmp_path / "y.rrd").exists()


def test_without_rerun_the_viewer_says_what_to_install(srf_path, monkeypatch, capsys):
    # A None entry in sys.modules makes `import rerun` raise ImportError.
    monkeypatch.setitem(sys.modules, "rerun", None)
    assert main([str(srf_path)]) == 2
    assert "rupture-generator[vis]" in capsys.readouterr().err
