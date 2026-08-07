"""Tests for rupture_generator.srf.

The geometry tests that lived here are gone with the properties they covered:
`SrfFile.geometry` and `.planes` needed NZTM projection and a Plane type, and
rupture generation uses neither. See the module note in `srf.py`.
"""

import gzip
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy as sp

from rupture_generator import srf

SRF_DIR = Path(__file__).parent / "srfs"


def test_christchurch_srf():
    """Test that the SRF reader can parse the Christchurch SRF and validate basic properties."""
    christchurch_srf = srf.read_srf(SRF_DIR / "3468575.srf")
    assert christchurch_srf.version == "1.0"
    assert len(christchurch_srf.header) == 1
    assert len(christchurch_srf.points) == 14400
    assert christchurch_srf.header.iloc[0].to_dict() == pytest.approx(
        {
            "elon": 172.6966,
            "elat": -43.5446,
            "nstk": 160,
            "ndip": 90,
            "len": 16.00,
            "wid": 9.00,
            "stk": 59,
            "dip": 69,
            "dtop": 0.63,
            "shyp": -2.00,
            "dhyp": 6.00,
        }
    )
    # local strike and dip should match the header
    assert (christchurch_srf.points["dip"] == 69).all()
    assert (christchurch_srf.points["stk"] == 59).all()
    assert christchurch_srf.points["tinit"].min() == 0.0
    # For the Christchurch event, the slip is only defined in the t1 component.
    assert christchurch_srf.slipt1_array.shape[0] == len(christchurch_srf.points)
    # dt is constant for SRF
    assert christchurch_srf.dt == 2.5e-02
    # Check that the segments code correctly identifies one segment
    assert len(christchurch_srf.segments) == 1
    assert len(christchurch_srf.segments[0]) == len(christchurch_srf.points)
    # The longest pulse, not the rupture's duration. Those were the same number
    # while the column index was `floor(tinit/dt) + i`; they are not now that it is
    # `i`. This event's longest pulse is 206 samples and its rupture spans 361.
    assert christchurch_srf.nt == 206

    assert christchurch_srf.points.iloc[0].to_dict() == pytest.approx(
        {
            "lon": 172.6127,
            "lat": -43.5821,
            "dep": 0.6767,
            "stk": 59,
            "dip": 69,
            "area": 1.0e08,
            "tinit": 5.7029,
            "dt": 2.5e-02,
            "rake": 102,
            "slip": 17.49,
            "rise": 0.3,
        }
    )
    # Every pulse starts at column zero now, so finding one no longer means
    # computing `tinit // dt` -- which was the arithmetic that quantised the onset.
    # have to manually slice because the sparse arrays do not support slicing
    slip_window = [christchurch_srf.slipt1_array[0, t] for t in range(12)]
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
    end_slip_window = [
        christchurch_srf.slipt1_array[last_index, t] for t in range(7)
    ]
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
    expected_headers = [
        {
            "elon": 172.133408,
            "elat": -43.550999,
            "nstk": 50,
            "ndip": 90,
            "len": 10.0000,
            "wid": 18.0000,
            "stk": 40,
            "dip": 75,
            "dtop": 1.0000,
            "shyp": 1.0000,
            "dhyp": 10.0000,
        },
        {
            "elon": 172.003906,
            "elat": -43.568298,
            "nstk": 60,
            "ndip": 90,
            "len": 12.0000,
            "wid": 18.0000,
            "stk": 121,
            "dip": 105,
            "dtop": 0.0000,
            "shyp": 6.0000,
            "dhyp": 6.0000,
        },
        {
            "elon": 172.194901,
            "elat": -43.588299,
            "nstk": 100,
            "ndip": 90,
            "len": 20.0000,
            "wid": 18.0000,
            "stk": 87,
            "dip": 85,
            "dtop": 0.0000,
            "shyp": -10.0000,
            "dhyp": 6.0000,
        },
        {
            "elon": 172.379898,
            "elat": -43.571301,
            "nstk": 70,
            "ndip": 90,
            "len": 14.0000,
            "wid": 18.0000,
            "stk": 87,
            "dip": 85,
            "dtop": 0.0000,
            "shyp": -7.0000,
            "dhyp": 6.0000,
        },
        {
            "elon": 171.944305,
            "elat": -43.578400,
            "nstk": 35,
            "ndip": 90,
            "len": 7.0000,
            "wid": 18.0000,
            "stk": 216,
            "dip": 50,
            "dtop": 0.0000,
            "shyp": 3.5000,
            "dhyp": 6.0000,
        },
        {
            "elon": 172.309799,
            "elat": -43.549900,
            "nstk": 55,
            "ndip": 90,
            "len": 11.0000,
            "wid": 18.0000,
            "stk": 40,
            "dip": 80,
            "dtop": 0.0000,
            "shyp": -5.5000,
            "dhyp": 6.0000,
        },
        {
            "elon": 172.182205,
            "elat": -43.508999,
            "nstk": 40,
            "ndip": 90,
            "len": 8.0000,
            "wid": 18.0000,
            "stk": 150,
            "dip": 54,
            "dtop": 0.0000,
            "shyp": 4.0000,
            "dhyp": 6.0000,
        },
    ]
    actual_headers = darfield_srf.header.to_dict(orient="records")
    assert len(actual_headers) == len(expected_headers)
    for actual, expected in zip(actual_headers, expected_headers):
        assert actual == pytest.approx(expected)
    # Will not test the basic properties again because that is tested
    # in the Christchurch case pretty thoroughly. Will, however, test
    # the segment iteration thoroughly
    assert len(darfield_srf.segments) == len(darfield_srf.header)
    for i, segment in enumerate(darfield_srf.segments):
        segment_header = darfield_srf.header.iloc[i]
        segment = darfield_srf.segments[i]
        assert len(segment) == segment_header["nstk"] * segment_header["ndip"]
        assert (segment["dip"] == segment_header["dip"]).all()
        assert (segment["stk"] == segment_header["stk"]).all()
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
        assert christchurch_srf.header.equals(christchurch_srf_tmp.header)
        assert christchurch_srf.points.equals(christchurch_srf_tmp.points)
        assert (christchurch_srf.slip != christchurch_srf_tmp.slip).nnz == 0


def test_hdf5_read_write():
    """
    Test that writing an SRF to HDF5 and reading it back produces an SrfFile
    with identical properties (version, header, points, slipt1_array).
    """
    original_srf = srf.read_srf(SRF_DIR / "3468575.srf")

    with tempfile.NamedTemporaryFile(suffix=".hdf5") as tmp_hdf5_file:
        hdf5_ffp = Path(tmp_hdf5_file.name)

        original_srf.write_hdf5(hdf5_ffp)

        reconstructed_srf = srf.SrfFile.from_hdf5(hdf5_ffp)

        assert original_srf.version == reconstructed_srf.version, "Version mismatch"
        assert original_srf.header.equals(reconstructed_srf.header), "Header mismatch"
        assert original_srf.points.equals(reconstructed_srf.points), "Points mismatch"

        assert (
            original_srf.slipt1_array.shape == reconstructed_srf.slipt1_array.shape
        ), "slipt1_array shape mismatch"
        assert np.array_equal(
            original_srf.slipt1_array.data, reconstructed_srf.slipt1_array.data
        ), "slipt1_array data mismatch"
        assert np.array_equal(
            original_srf.slipt1_array.indices, reconstructed_srf.slipt1_array.indices
        ), "slipt1_array indices mismatch"
        assert np.array_equal(
            original_srf.slipt1_array.indptr, reconstructed_srf.slipt1_array.indptr
        ), "slipt1_array indptr mismatch"

        assert (original_srf.slipt1_array != reconstructed_srf.slipt1_array).nnz == 0, (  # ty: ignore[unresolved-attribute]
            "slipt1_array content mismatch"
        )


def test_sw4_hdf5_read_write(tmp_path: Path):
    """Test that write_sw4_hdf5 preserves header, points, and slip data."""

    original_srf = srf.read_srf(SRF_DIR / "3468575.srf")

    output_path = tmp_path / "test.h5"
    original_srf.write_sw4_hdf5(output_path)

    with h5py.File(output_path, "r") as h5file:
        # VERSION
        assert h5file.attrs["VERSION"] == np.float32(1.0)

        plane = h5file.attrs["PLANE"]
        assert plane.shape == (len(original_srf.header),)
        assert srf.SW4_PLANE_DTYPE.names is not None
        for idx, row in original_srf.header.iterrows():
            for field in srf.SW4_PLANE_DTYPE.names:
                assert plane[idx][field] == pytest.approx(row[field.lower()], abs=1e-3)

        points = h5file["POINTS"]
        assert points.shape == (len(original_srf.points),)
        for field in (
            "LON",
            "LAT",
            "DEP",
            "STK",
            "DIP",
            "AREA",
            "TINIT",
            "DT",
            "RAKE",
        ):
            assert points[field] == pytest.approx(
                original_srf.points[field.lower()].to_numpy(), abs=1e-3
            )
        assert points["SLIP1"] == pytest.approx(
            original_srf.points["slip"].to_numpy(), abs=1e-3
        )

        # VS/DEN default to 0.0 for Version 1.0 SRF
        assert points["VS"] == pytest.approx(0.0)
        assert points["DEN"] == pytest.approx(0.0)

        # NT1 from slipt1_array.indptr
        assert points["NT1"] == pytest.approx(
            np.diff(original_srf.slipt1_array.indptr).astype(np.int32)
        )

        # SR1 slip-time function data
        assert h5file["SR1"][...] == pytest.approx(
            original_srf.slipt1_array.data.astype(np.float32)
        )

        # Unused slip components stay zero
        for zero_field in ("SLIP2", "NT2", "SLIP3", "NT3"):
            assert points[zero_field] == pytest.approx(0)


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
    assert "vs" in srf_v2.points
    assert "den" in srf_v2.points
    assert srf_v2.points["vs"].tolist() == pytest.approx([3.5e5, 3.6e5])
    assert srf_v2.points["den"].tolist() == pytest.approx([2.7, 2.8])
    assert srf_v2.points.iloc[0].to_dict() == pytest.approx(
        {
            "lon": 172.0,
            "lat": -43.0,
            "dep": 0.5,
            "stk": 45,
            "dip": 80,
            "area": 1.0e10,
            "tinit": 0.0,
            "dt": 0.1,
            "vs": 3.5e5,
            "den": 2.7,
            "rake": 90,
            "slip": 10.0,
            "rise": 0.2,
        }
    )
    # 2 points, and the matrix is as wide as the longest pulse -- not as wide as the
    # rupture. Column i is the ith sample OF THE PULSE; the onset lives in `tinit`
    # as a float and is not folded into the column index.
    #
    # It used to be: the column was `floor(tinit / dt) + i`, which quantised every
    # onset to a sample boundary and made the matrix as wide as the whole rupture.
    # Point 1 here starts at tinit = 0.1 with dt = 0.1, so it used to occupy columns
    # 1 and 2 and the matrix was (2, 3).
    assert srf_v2.slipt1_array.shape == (2, 2)
    # the stored values, row by row; each point starts with a slip-rate of 0.0.
    assert srf_v2.slipt1_array.data.tolist() == pytest.approx([0.0, 5.0, 0.0, 6.0])
    # Both points fill columns 0 and 1, whatever their onsets.
    assert srf_v2.slipt1_array.indices.tolist() == [0, 1, 0, 1]
    # row boundaries into data/indices: nt1 = 2 per point, so cuts at 0, 2, 4.
    assert srf_v2.slipt1_array.indptr.tolist() == [0, 2, 4]


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
    assert real_srf.header.iloc[0].to_dict() == pytest.approx(
        {
            "elon": 176.514603,
            "elat": -38.006092,
            "nstk": 51,
            "ndip": 51,
            "len": 5.0699,
            "wid": 5.0699,
            "stk": 240,
            "dip": 88,
            "dtop": 0.0,
            "shyp": 0.0,
            "dhyp": 2.5350,
        }
    )
    assert len(real_srf.points) == 2601
    assert real_srf.points.iloc[0].to_dict() == pytest.approx(
        {
            "lon": 176.539108,
            "lat": -37.994919,
            "dep": 4.96747e-02,
            "stk": 240,
            "dip": 88,
            "area": 9.88234e07,
            "tinit": 5.815377,
            "dt": 5.0e-03,
            "vs": 3.8e04,
            "den": 1.81,
            "rake": -16,
            "slip": 94.2758,
            "rise": 64 * 5.0e-03,
        }
    )
    assert real_srf.points.iloc[1300].to_dict() == pytest.approx(
        {
            "lon": 176.514099,
            "lat": -38.005402,
            "dep": 2.53341,
            "stk": 240,
            "dip": 88,
            "area": 9.88234e07,
            "tinit": 9.254894e-02,
            "dt": 5.0e-03,
            "vs": 2.28e05,
            "den": 2.40,
            "rake": -15,
            "slip": 49.4048,
            "rise": 59 * 5.0e-03,
        }
    )
    assert real_srf.points.iloc[2600].to_dict() == pytest.approx(
        {
            "lon": 176.489090,
            "lat": -38.015869,
            "dep": 5.01714,
            "stk": 240,
            "dip": 88,
            "area": 9.88234e07,
            "tinit": 2.567641,
            "dt": 5.0e-03,
            "vs": 3.6e05,
            "den": 2.72,
            "rake": -7,
            "slip": 15.8105,
            "rise": 3 * 5.0e-03,
        }
    )
    # the first point's slip-rate function holds nt1 = 64 samples
    assert np.diff(real_srf.slipt1_array.indptr)[0] == 64
    assert real_srf.slipt1_array.data[:3] == pytest.approx([0.0, 9.69786, 21.3934])
    # the last point's slip-rate function holds nt1 = 3 samples
    assert np.diff(real_srf.slipt1_array.indptr)[-1] == 3
    assert real_srf.slipt1_array.data[-3:] == pytest.approx([0.0, 3.16209e03, 0.0])


def test_read_srf_v1_has_no_vs_den():
    """Regression: version 1.0 SRFs must not gain vs/den columns."""
    christchurch_srf = srf.read_srf(SRF_DIR / "3468575.srf")
    assert "vs" not in christchurch_srf.points
    assert "den" not in christchurch_srf.points
    assert christchurch_srf.points.shape[1] == 11


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
    assert srf_v2.header.equals(reread.header)
    assert srf_v2.points.equals(reread.points)
    assert (srf_v2.slip != reread.slip).nnz == 0


def test_sw4_hdf5_v2(tmp_path: Path):
    """Test that write_sw4_hdf5 writes vs/den for a version 2.0 SRF."""
    srf_v2 = srf.read_srf(SRF_DIR / "point_source_v2.srf")
    out = tmp_path / "v2.h5"
    srf_v2.write_sw4_hdf5(out)
    with h5py.File(out, "r") as h5file:
        assert h5file.attrs["VERSION"] == np.float32(2.0)
        points = h5file["POINTS"]
        assert points["VS"] == pytest.approx(srf_v2.points["vs"].to_numpy())
        assert points["DEN"] == pytest.approx(srf_v2.points["den"].to_numpy())
