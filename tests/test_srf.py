"""Tests for rupture_generator.srf.

The geometry tests that lived here are gone with the properties they covered:
`SrfFile.geometry` and a projected `Plane` type needed NZTM projection, and rupture
generation uses neither. `SrfFile.planes` is now the file's own `PLANE` header block,
which is a different thing. See the module note in `srf.py`.
"""

import dataclasses
import gzip
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy as sp

from rupture_generator import srf

SRF_DIR = Path(__file__).parent / "srfs"


def point(points: srf.Points, index: int) -> dict[str, float]:
    """One subfault's values, for comparison against a literal."""
    return {
        field.name: values[index]
        for field in dataclasses.fields(points)
        if (values := getattr(points, field.name)) is not None
    }


def assert_points_equal(actual: srf.Points, expected: srf.Points) -> None:
    """Every array of every field, exactly equal."""
    for field in dataclasses.fields(expected):
        theirs, ours = getattr(expected, field.name), getattr(actual, field.name)
        if theirs is None:
            assert ours is None, field.name
            continue
        assert np.array_equal(ours, theirs), field.name


def test_christchurch_srf():
    """Test that the SRF reader can parse the Christchurch SRF and validate basic properties."""
    christchurch_srf = srf.read_srf(SRF_DIR / "3468575.srf")
    assert christchurch_srf.version == "1.0"
    assert len(christchurch_srf.planes) == 1
    assert len(christchurch_srf.points) == 14400
    assert dataclasses.asdict(christchurch_srf.planes[0]) == pytest.approx(
        {
            "centre_longitude_deg": 172.6966,
            "centre_latitude_deg": -43.5446,
            "strike_count": 160,
            "dip_count": 90,
            "length_km": 16.00,
            "width_km": 9.00,
            "strike_deg": 59,
            "dip_deg": 69,
            "top_depth_km": 0.63,
            "hypocentre_strike_km": -2.00,
            "hypocentre_dip_km": 6.00,
        }
    )
    # local strike and dip should match the header
    assert (christchurch_srf.points.dip_deg == 69).all()
    assert (christchurch_srf.points.strike_deg == 59).all()
    assert christchurch_srf.points.onset_s.min() == 0.0
    # For the Christchurch event, the slip is only defined in the t1 component.
    assert christchurch_srf.slip_rate.shape[0] == len(christchurch_srf.points)
    # dt is constant for SRF
    assert christchurch_srf.dt == 2.5e-02
    # Check that the segments code correctly identifies one segment
    assert len(christchurch_srf.segments) == 1
    assert len(christchurch_srf.segments[0]) == len(christchurch_srf.points)
    # The longest pulse, not the rupture's duration. Those were the same number
    # while the column index was `floor(tinit/dt) + i`; they are not now that it is
    # `i`. This event's longest pulse is 206 samples and its rupture spans 361.
    assert christchurch_srf.nt == 206

    assert point(christchurch_srf.points, 0) == pytest.approx(
        {
            "longitude_deg": 172.6127,
            "latitude_deg": -43.5821,
            "depth_km": 0.6767,
            "strike_deg": 59,
            "dip_deg": 69,
            "area_cm2": 1.0e08,
            "onset_s": 5.7029,
            "sample_interval_s": 2.5e-02,
            "rake_deg": 102,
            "slip_cm": 17.49,
            "rise_time_s": 0.3,
        }
    )
    # Every pulse starts at column zero now, so finding one no longer means
    # computing `tinit // dt` -- which was the arithmetic that quantised the onset.
    # have to manually slice because the sparse arrays do not support slicing
    slip_window = [christchurch_srf.slip_rate[0, t] for t in range(12)]
    assert slip_window == [
        0.00000e00,
        2.07568e02,
        2.42313e02,
        5.90245e01,
        4.89368e01,
        4.26333e01,
        3.50411e01,
        2.68253e01,
        1.87057e01,
        1.13937e01,
        5.52983e00,
        1.62786e00,
    ]

    # Just to check that the last row is also parsed correctly
    last_index = len(christchurch_srf.points) - 1
    end_slip_window = [christchurch_srf.slip_rate[last_index, t] for t in range(7)]
    assert end_slip_window == [
        0.00000e00,
        3.97588e02,
        7.47954e01,
        6.18204e01,
        4.37692e01,
        2.48125e01,
        9.33055e00,
    ]


def test_darfield_srf():
    """Test that the SRF reader can parse the Darfield SRF and validate basic properties."""
    darfield_srf = srf.read_srf(SRF_DIR / "3366146.srf")
    expected_planes = [
        {
            "centre_longitude_deg": 172.133408,
            "centre_latitude_deg": -43.550999,
            "strike_count": 50,
            "dip_count": 90,
            "length_km": 10.0000,
            "width_km": 18.0000,
            "strike_deg": 40,
            "dip_deg": 75,
            "top_depth_km": 1.0000,
            "hypocentre_strike_km": 1.0000,
            "hypocentre_dip_km": 10.0000,
        },
        {
            "centre_longitude_deg": 172.003906,
            "centre_latitude_deg": -43.568298,
            "strike_count": 60,
            "dip_count": 90,
            "length_km": 12.0000,
            "width_km": 18.0000,
            "strike_deg": 121,
            "dip_deg": 105,
            "top_depth_km": 0.0000,
            "hypocentre_strike_km": 6.0000,
            "hypocentre_dip_km": 6.0000,
        },
        {
            "centre_longitude_deg": 172.194901,
            "centre_latitude_deg": -43.588299,
            "strike_count": 100,
            "dip_count": 90,
            "length_km": 20.0000,
            "width_km": 18.0000,
            "strike_deg": 87,
            "dip_deg": 85,
            "top_depth_km": 0.0000,
            "hypocentre_strike_km": -10.0000,
            "hypocentre_dip_km": 6.0000,
        },
        {
            "centre_longitude_deg": 172.379898,
            "centre_latitude_deg": -43.571301,
            "strike_count": 70,
            "dip_count": 90,
            "length_km": 14.0000,
            "width_km": 18.0000,
            "strike_deg": 87,
            "dip_deg": 85,
            "top_depth_km": 0.0000,
            "hypocentre_strike_km": -7.0000,
            "hypocentre_dip_km": 6.0000,
        },
        {
            "centre_longitude_deg": 171.944305,
            "centre_latitude_deg": -43.578400,
            "strike_count": 35,
            "dip_count": 90,
            "length_km": 7.0000,
            "width_km": 18.0000,
            "strike_deg": 216,
            "dip_deg": 50,
            "top_depth_km": 0.0000,
            "hypocentre_strike_km": 3.5000,
            "hypocentre_dip_km": 6.0000,
        },
        {
            "centre_longitude_deg": 172.309799,
            "centre_latitude_deg": -43.549900,
            "strike_count": 55,
            "dip_count": 90,
            "length_km": 11.0000,
            "width_km": 18.0000,
            "strike_deg": 40,
            "dip_deg": 80,
            "top_depth_km": 0.0000,
            "hypocentre_strike_km": -5.5000,
            "hypocentre_dip_km": 6.0000,
        },
        {
            "centre_longitude_deg": 172.182205,
            "centre_latitude_deg": -43.508999,
            "strike_count": 40,
            "dip_count": 90,
            "length_km": 8.0000,
            "width_km": 18.0000,
            "strike_deg": 150,
            "dip_deg": 54,
            "top_depth_km": 0.0000,
            "hypocentre_strike_km": 4.0000,
            "hypocentre_dip_km": 6.0000,
        },
    ]
    assert len(darfield_srf.planes) == len(expected_planes)
    for actual, expected in zip(darfield_srf.planes, expected_planes):
        assert dataclasses.asdict(actual) == pytest.approx(expected)
    # Will not test the basic properties again because that is tested
    # in the Christchurch case pretty thoroughly. Will, however, test
    # the segment iteration thoroughly
    assert len(darfield_srf.segments) == len(darfield_srf.planes)
    for index, segment in enumerate(darfield_srf.segments):
        plane = darfield_srf.planes[index]
        assert len(segment) == plane.strike_count * plane.dip_count
        assert (segment.dip_deg == plane.dip_deg).all()
        assert (segment.strike_deg == plane.strike_deg).all()


def test_the_segments_partition_the_points():
    """Every point belongs to exactly one segment, in file order."""
    darfield_srf = srf.read_srf(SRF_DIR / "3366146.srf")
    rejoined = np.concatenate(
        [segment.longitude_deg for segment in darfield_srf.segments]
    )
    assert np.array_equal(rejoined, darfield_srf.points.longitude_deg)


def test_junk_srfs():
    """Test that malformed SRFs raise srf parsing errors."""
    with pytest.raises(srf.ParseError):
        srf.read_srf(SRF_DIR / "empty.srf")

    with pytest.raises(srf.ParseError):
        srf.read_srf(SRF_DIR / "bad_int.srf")

    with pytest.raises(srf.ParseError):
        srf.read_srf(SRF_DIR / "bad_float.srf")

    with pytest.raises(srf.ParseError):
        srf.read_srf(SRF_DIR / "no_points.srf")

    with pytest.raises(srf.ParseError):
        srf.read_srf(SRF_DIR / "bad_plane.srf")


def test_writing_christchurch():
    """Check that writing a copy an SRF produces an SRF with the same values."""
    christchurch_srf = srf.read_srf(SRF_DIR / "3468575.srf")
    with tempfile.NamedTemporaryFile() as tmp_christchurch_srf_handle:
        srf.write_srf(tmp_christchurch_srf_handle.name, christchurch_srf)
        christchurch_srf_tmp = srf.read_srf(tmp_christchurch_srf_handle.name)
        assert christchurch_srf.planes == christchurch_srf_tmp.planes
        assert_points_equal(christchurch_srf_tmp.points, christchurch_srf.points)
        assert (christchurch_srf.slip_rate != christchurch_srf_tmp.slip_rate).nnz == 0


def test_sw4_hdf5_read_write(tmp_path: Path):
    """Test that write_sw4_hdf5 preserves header, points, and slip data.

    The field-by-field assertions restate the mapping rather than reading it out of
    `srf`, so a transposed pair of same-typed neighbours -- ELON/ELAT, SHYP/DHYP --
    fails here instead of being mirrored.
    """
    original_srf = srf.read_srf(SRF_DIR / "3468575.srf")

    output_path = tmp_path / "test.h5"
    original_srf.write_sw4_hdf5(output_path)

    with h5py.File(output_path, "r") as h5file:
        # VERSION
        assert h5file.attrs["VERSION"] == np.float32(1.0)

        plane_data = h5file.attrs["PLANE"]
        assert plane_data.shape == (len(original_srf.planes),)
        for row, plane in zip(plane_data, original_srf.planes):
            assert row["ELON"] == pytest.approx(plane.centre_longitude_deg, abs=1e-3)
            assert row["ELAT"] == pytest.approx(plane.centre_latitude_deg, abs=1e-3)
            assert row["NSTK"] == plane.strike_count
            assert row["NDIP"] == plane.dip_count
            assert row["LEN"] == pytest.approx(plane.length_km, abs=1e-3)
            assert row["WID"] == pytest.approx(plane.width_km, abs=1e-3)
            assert row["STK"] == pytest.approx(plane.strike_deg, abs=1e-3)
            assert row["DIP"] == pytest.approx(plane.dip_deg, abs=1e-3)
            assert row["DTOP"] == pytest.approx(plane.top_depth_km, abs=1e-3)
            assert row["SHYP"] == pytest.approx(plane.hypocentre_strike_km, abs=1e-3)
            assert row["DHYP"] == pytest.approx(plane.hypocentre_dip_km, abs=1e-3)

        points = original_srf.points
        written = h5file["POINTS"]
        assert written.shape == (len(points),)
        for sw4_field, expected in [
            ("LON", points.longitude_deg),
            ("LAT", points.latitude_deg),
            ("DEP", points.depth_km),
            ("STK", points.strike_deg),
            ("DIP", points.dip_deg),
            ("AREA", points.area_cm2),
            ("TINIT", points.onset_s),
            ("DT", points.sample_interval_s),
            ("RAKE", points.rake_deg),
            ("SLIP1", points.slip_cm),
        ]:
            assert written[sw4_field] == pytest.approx(expected, abs=1e-3), sw4_field

        # VS/DEN default to 0.0 for Version 1.0 SRF
        assert written["VS"] == pytest.approx(0.0)
        assert written["DEN"] == pytest.approx(0.0)

        # NT1 from the slip-rate matrix's row boundaries
        assert written["NT1"] == pytest.approx(np.diff(original_srf.slip_rate.indptr))

        # SR1 slip-time function data
        assert h5file["SR1"][...] == pytest.approx(
            original_srf.slip_rate.data.astype(np.float32)
        )

        # Unused slip components stay zero
        for zero_field in ("SLIP2", "NT2", "SLIP3", "NT3"):
            assert written[zero_field] == pytest.approx(0)


def test_read_srf_v2():
    """Read a minimal hand-written version 2.0 SRF and verify every parsed value.

    The 2-point source file is small enough that all the expected values here,
    including the complete slip-rate sparse-matrix structure, can be checked by
    eye against the file. See test_read_real_srf_v2 for the complementary test
    on a real (genslip-generated) version 2.0 SRF.
    """
    srf_v2 = srf.read_srf(SRF_DIR / "point_source_v2.srf")
    assert srf_v2.version == "2.0"
    assert len(srf_v2.points) == 2
    assert srf_v2.points.has_material_properties
    assert srf_v2.points.shear_speed_cm_s.tolist() == pytest.approx([3.5e5, 3.6e5])
    assert srf_v2.points.density_g_cm3.tolist() == pytest.approx([2.7, 2.8])
    assert point(srf_v2.points, 0) == pytest.approx(
        {
            "longitude_deg": 172.0,
            "latitude_deg": -43.0,
            "depth_km": 0.5,
            "strike_deg": 45,
            "dip_deg": 80,
            "area_cm2": 1.0e10,
            "onset_s": 0.0,
            "sample_interval_s": 0.1,
            "shear_speed_cm_s": 3.5e5,
            "density_g_cm3": 2.7,
            "rake_deg": 90,
            "slip_cm": 10.0,
            "rise_time_s": 0.2,
        }
    )
    # 2 points, and the matrix is as wide as the longest pulse -- not as wide as the
    # rupture. Column i is the ith sample OF THE PULSE; the onset lives in `onset_s`
    # as a float and is not folded into the column index.
    #
    # It used to be: the column was `floor(tinit / dt) + i`, which quantised every
    # onset to a sample boundary and made the matrix as wide as the whole rupture.
    # Point 1 here starts at tinit = 0.1 with dt = 0.1, so it used to occupy columns
    # 1 and 2 and the matrix was (2, 3).
    assert srf_v2.slip_rate.shape == (2, 2)
    # the stored values, row by row; each point starts with a slip-rate of 0.0.
    assert srf_v2.slip_rate.data.tolist() == pytest.approx([0.0, 5.0, 0.0, 6.0])
    # Both points fill columns 0 and 1, whatever their onsets.
    assert srf_v2.slip_rate.indices.tolist() == [0, 1, 0, 1]
    # row boundaries into data/indices: nt1 = 2 per point, so cuts at 0, 2, 4.
    assert srf_v2.slip_rate.indptr.tolist() == [0, 2, 4]


def test_read_real_srf_v2(tmp_path: Path):
    """Read a real genslip-generated version 2.0 SRF end to end.

    Complements the hand-verifiable test_read_srf_v2 by covering what only a
    real file exercises: the comment lines genslip writes after the version
    line, and a full-size (2601-point) rupture. The expected values below are
    spot checks transcribed from the first, middle and last point blocks of
    the file. Because the parser consumes the file as one sequential token
    stream, a correct last point implies it stayed aligned through every
    preceding block.
    """
    srf_ffp = tmp_path / "test_v2.srf"
    srf_ffp.write_bytes(
        gzip.decompress(
            (Path(__file__).parent / "srfs" / "test_v2.srf.gz").read_bytes()
        )
    )
    real_srf = srf.read_srf(srf_ffp)
    assert real_srf.version == "2.0"
    assert dataclasses.asdict(real_srf.planes[0]) == pytest.approx(
        {
            "centre_longitude_deg": 176.514603,
            "centre_latitude_deg": -38.006092,
            "strike_count": 51,
            "dip_count": 51,
            "length_km": 5.0699,
            "width_km": 5.0699,
            "strike_deg": 240,
            "dip_deg": 88,
            "top_depth_km": 0.0,
            "hypocentre_strike_km": 0.0,
            "hypocentre_dip_km": 2.5350,
        }
    )
    assert len(real_srf.points) == 2601
    assert point(real_srf.points, 0) == pytest.approx(
        {
            "longitude_deg": 176.539108,
            "latitude_deg": -37.994919,
            "depth_km": 4.96747e-02,
            "strike_deg": 240,
            "dip_deg": 88,
            "area_cm2": 9.88234e07,
            "onset_s": 5.815377,
            "sample_interval_s": 5.0e-03,
            "shear_speed_cm_s": 3.8e04,
            "density_g_cm3": 1.81,
            "rake_deg": -16,
            "slip_cm": 94.2758,
            "rise_time_s": 64 * 5.0e-03,
        }
    )
    assert point(real_srf.points, 1300) == pytest.approx(
        {
            "longitude_deg": 176.514099,
            "latitude_deg": -38.005402,
            "depth_km": 2.53341,
            "strike_deg": 240,
            "dip_deg": 88,
            "area_cm2": 9.88234e07,
            "onset_s": 9.254894e-02,
            "sample_interval_s": 5.0e-03,
            "shear_speed_cm_s": 2.28e05,
            "density_g_cm3": 2.40,
            "rake_deg": -15,
            "slip_cm": 49.4048,
            "rise_time_s": 59 * 5.0e-03,
        }
    )
    assert point(real_srf.points, 2600) == pytest.approx(
        {
            "longitude_deg": 176.489090,
            "latitude_deg": -38.015869,
            "depth_km": 5.01714,
            "strike_deg": 240,
            "dip_deg": 88,
            "area_cm2": 9.88234e07,
            "onset_s": 2.567641,
            "sample_interval_s": 5.0e-03,
            "shear_speed_cm_s": 3.6e05,
            "density_g_cm3": 2.72,
            "rake_deg": -7,
            "slip_cm": 15.8105,
            "rise_time_s": 3 * 5.0e-03,
        }
    )
    # the first point's slip-rate function holds nt1 = 64 samples
    assert np.diff(real_srf.slip_rate.indptr)[0] == 64
    assert real_srf.slip_rate.data[:3] == pytest.approx([0.0, 9.69786, 21.3934])
    # the last point's slip-rate function holds nt1 = 3 samples
    assert np.diff(real_srf.slip_rate.indptr)[-1] == 3
    assert real_srf.slip_rate.data[-3:] == pytest.approx([0.0, 3.16209e03, 0.0])


def test_read_srf_v1_has_no_vs_den():
    """Regression: version 1.0 SRFs must not gain material properties."""
    christchurch_srf = srf.read_srf(SRF_DIR / "3468575.srf")
    assert not christchurch_srf.points.has_material_properties
    assert christchurch_srf.points.shear_speed_cm_s is None
    assert christchurch_srf.points.density_g_cm3 is None


def test_unsupported_version_srf(tmp_path: Path):
    """An otherwise-valid SRF whose version is neither 1.0 nor 2.0 is rejected."""
    bad_srf = tmp_path / "v9.srf"
    bad_srf.write_text(
        "9.0\n"
        "PLANE 1\n"
        "  172.0  -43.0   1   1   1.0   1.0\n"
        "  45   80   0.0   0.0   0.5\n"
        "POINTS 1\n"
        "  172.0  -43.0   0.5   45   80   1.0e10   0.0   0.1\n"
        "  90   10.0   1   0.0   0   0.0   0\n"
        "  0.0\n"
    )
    with pytest.raises(srf.ParseError):
        srf.read_srf(bad_srf)


def test_write_read_srf_v2(tmp_path: Path):
    """Check that writing a version 2.0 SRF round-trips, including vs/den."""
    srf_v2 = srf.read_srf(SRF_DIR / "point_source_v2.srf")
    out = tmp_path / "roundtrip_v2.srf"
    srf.write_srf(out, srf_v2)
    reread = srf.read_srf(out)
    assert reread.version == "2.0"
    assert srf_v2.planes == reread.planes
    assert_points_equal(reread.points, srf_v2.points)
    assert (srf_v2.slip_rate != reread.slip_rate).nnz == 0


def test_sw4_hdf5_v2(tmp_path: Path):
    """Test that write_sw4_hdf5 writes vs/den for a version 2.0 SRF."""
    srf_v2 = srf.read_srf(SRF_DIR / "point_source_v2.srf")
    out = tmp_path / "v2.h5"
    srf_v2.write_sw4_hdf5(out)
    with h5py.File(out, "r") as h5file:
        assert h5file.attrs["VERSION"] == np.float32(2.0)
        points = h5file["POINTS"]
        assert points["VS"] == pytest.approx(srf_v2.points.shear_speed_cm_s)
        assert points["DEN"] == pytest.approx(srf_v2.points.density_g_cm3)


class TestTheModelRefusesToContradictItself:
    """What a DataFrame used to catch for free, and what it never caught at all."""

    def points(self, count: int = 3, **overrides) -> srf.Points:
        arrays = {
            field.name: np.zeros(count, dtype=np.float32)
            for field in dataclasses.fields(srf.Points)
        }
        return srf.Points(**(arrays | overrides))

    def test_ragged_point_arrays_are_refused(self):
        with pytest.raises(ValueError, match="disagree on length"):
            self.points(onset_s=np.zeros(2, dtype=np.float32))

    def test_one_material_property_without_the_other_is_refused(self):
        with pytest.raises(ValueError, match="go together"):
            self.points(density_g_cm3=None)

    def test_version_2_without_material_properties_is_refused(self):
        bare = self.points(shear_speed_cm_s=None, density_g_cm3=None)
        with pytest.raises(ValueError, match="version 2.0 carries"):
            srf.SrfFile("2.0", [], bare, sp.sparse.csr_array((3, 1), dtype=np.float32))

    def test_version_1_with_material_properties_is_refused(self):
        with pytest.raises(ValueError, match="nowhere to put"):
            srf.SrfFile(
                "1.0",
                [],
                self.points(),
                sp.sparse.csr_array((3, 1), dtype=np.float32),
            )

    def test_an_unsupported_version_is_refused(self):
        with pytest.raises(ValueError, match="unsupported SRF version"):
            srf.SrfFile(
                "3.0",
                [],
                self.points(),
                sp.sparse.csr_array((3, 1), dtype=np.float32),
            )


class TestSelectingSubfaults:
    def test_a_slice_keeps_every_field(self):
        srf_v2 = srf.read_srf(SRF_DIR / "point_source_v2.srf")
        first = srf_v2.points[0:1]
        assert len(first) == 1
        assert first.longitude_deg == pytest.approx([172.0])
        assert first.shear_speed_cm_s == pytest.approx([3.5e5])

    def test_a_boolean_mask_selects(self):
        srf_v2 = srf.read_srf(SRF_DIR / "point_source_v2.srf")
        slipped_more = srf_v2.points[srf_v2.points.slip_cm > 11.0]
        assert len(slipped_more) == 1
        assert slipped_more.slip_cm == pytest.approx([12.0])
        assert slipped_more.shear_speed_cm_s == pytest.approx([3.6e5])

    def test_a_single_index_is_refused(self):
        # It would otherwise produce a Points of rank-zero arrays, which fails
        # later and somewhere else.
        srf_v2 = srf.read_srf(SRF_DIR / "point_source_v2.srf")
        with pytest.raises(TypeError, match="slice or an array"):
            srf_v2.points[0]


# Deliberately not asserted:
#
# - Anything about `write_hdf5` / `from_hdf5` / `to_xarray`. They are gone; see
#   PRUNED.md.
