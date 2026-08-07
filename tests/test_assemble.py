"""Tests for assembling an SRF from a generated rupture.

Two things can go wrong here and nowhere else: a column can be filled from the wrong
array, and the ragged slip-rate pulses can be indexed wrongly on the way into a
sparse matrix. Both survive a round trip through the file unless something checks.
"""

import numpy as np
import pytest

from rupture_generator import (
    FaultGrid,
    Ramp,
    SlipSpec,
    SourceSpec,
    SpectrumModel,
    TimingSpec,
    VelocityModel1D,
    generate_rupture,
)
from rupture_generator.assemble import PlaneHeader, SubfaultGeometry, to_srf_file
from rupture_generator.srf import read_srf, write_srf

STRIKE, DIP = 12, 8
SUBFAULTS = STRIKE * DIP


def rupture():
    """A small generated rupture."""
    grid = FaultGrid(
        STRIKE,
        DIP,
        16,
        10,
        1.0,
        1.0,
        depth_km=np.array([0.5 + i * 2.0 for i in range(DIP)], dtype=np.float32),
        base_rake_deg=np.full(SUBFAULTS, 175.0, dtype=np.float32),
        velocity_fraction=np.full(SUBFAULTS, 0.8, dtype=np.float32),
    )
    model = VelocityModel1D(
        np.array([1.0, 5.0, 12.0, 30.0], dtype=np.float32),
        np.array([1.8, 2.6, 3.2, 3.6], dtype=np.float32),
        np.array([2.1, 2.4, 2.6, 2.7], dtype=np.float32),
    )
    shallow, deep = Ramp(6.5, 1.5), Ramp(17.5, 2.5)
    return generate_rupture(
        grid,
        model,
        SourceSpec(
            6.5,
            SpectrumModel.Mai,
            2.50,
            1.50,
            average_dip_deg=60.0,
            average_rake_deg=175.0,
        ),
        SlipSpec(SpectrumModel.Mai),
        TimingSpec(
            rupture_time_scale=-0.35,
            rise_time_blend=Ramp(2.0, 1.0),
            shallow_ramp=shallow,
            deep_ramp=deep,
            beta_shallow_ramp=Ramp(2.0, 1.0),
            beta_mid_ramp=Ramp(6.5, 1.5),
        ),
        seed=20260807,
        hypocentre_strike=STRIKE // 2,
        hypocentre_dip=DIP // 2,
    )


def geometry() -> SubfaultGeometry:
    """Subfaults on a plane near Christchurch, one square kilometre each."""
    strikes, dips = np.meshgrid(np.arange(STRIKE), np.arange(DIP))
    return SubfaultGeometry(
        longitude_deg=(172.6 + strikes.ravel() * 0.01).astype(np.float32),
        latitude_deg=(-43.5 - dips.ravel() * 0.005).astype(np.float32),
        depth_km=(0.5 + dips.ravel() * 2.0).astype(np.float32),
        strike_deg=np.full(SUBFAULTS, 45.0, dtype=np.float32),
        dip_deg=np.full(SUBFAULTS, 60.0, dtype=np.float32),
        area_cm2=np.full(SUBFAULTS, 1.0e10, dtype=np.float32),
    )


def header() -> PlaneHeader:
    return PlaneHeader(
        centre_longitude_deg=172.66,
        centre_latitude_deg=-43.52,
        strike_count=STRIKE,
        dip_count=DIP,
        length_km=12.0,
        width_km=8.0,
        strike_deg=45.0,
        dip_deg=60.0,
        top_depth_km=0.5,
        hypocentre_strike_km=0.0,
        hypocentre_dip_km=4.0,
    )


def assemble():
    generated = rupture()
    return generated, to_srf_file(
        generated,
        geometry(),
        header(),
        shear_speed_km_s=np.full(SUBFAULTS, 3.2, dtype=np.float32),
        density_g_cm3=np.full(SUBFAULTS, 2.6, dtype=np.float32),
    )


class TestTheColumnsComeFromTheRightArrays:
    def test_the_physics_columns_are_the_rupture(self) -> None:
        generated, srf_file = assemble()
        assert srf_file.points["slip"].to_numpy() == pytest.approx(generated.slip_cm)
        assert srf_file.points["rake"].to_numpy() == pytest.approx(generated.rake_deg)
        assert srf_file.points["tinit"].to_numpy() == pytest.approx(generated.onset_s)
        assert srf_file.points["rise"].to_numpy() == pytest.approx(generated.rise_time_s)

    def test_the_location_columns_are_the_geometry(self) -> None:
        _, srf_file = assemble()
        where = geometry()
        assert srf_file.points["lon"].to_numpy() == pytest.approx(where.longitude_deg)
        assert srf_file.points["lat"].to_numpy() == pytest.approx(where.latitude_deg)
        assert srf_file.points["dep"].to_numpy() == pytest.approx(where.depth_km)

    def test_the_header_is_the_plane(self) -> None:
        _, srf_file = assemble()
        row = srf_file.header.iloc[0]
        assert row["nstk"] == STRIKE
        assert row["ndip"] == DIP
        assert row["dip"] == pytest.approx(60.0)
        assert row["dhyp"] == pytest.approx(4.0)


class TestTheSlipRatePulses:
    def test_each_row_is_its_subfault_pulse(self) -> None:
        generated, srf_file = assemble()
        offsets = np.asarray(generated.slip_rate_offsets, dtype=np.int64)
        dense = srf_file.slipt1_array.toarray()
        for index in range(0, SUBFAULTS, 5):
            start, end = int(offsets[index]), int(offsets[index + 1])
            expected = generated.slip_rate[start:end]
            assert dense[index, : len(expected)] == pytest.approx(expected)
            # Everything past the pulse is padding, not signal.
            assert dense[index, len(expected) :] == pytest.approx(0.0)

    def test_every_row_integrates_to_its_slip(self) -> None:
        generated, srf_file = assemble()
        integrals = srf_file.slipt1_array.sum(axis=1) * generated.sample_interval_s
        assert integrals == pytest.approx(generated.slip_cm, rel=1e-3)

    def test_the_matrix_is_only_as_wide_as_the_longest_pulse(self) -> None:
        generated, srf_file = assemble()
        offsets = np.asarray(generated.slip_rate_offsets, dtype=np.int64)
        assert srf_file.slipt1_array.shape == (SUBFAULTS, int(np.diff(offsets).max()))


class TestRoundTrip:
    def test_writing_and_reading_preserves_the_model(self, tmp_path) -> None:
        generated, srf_file = assemble()
        path = tmp_path / "rupture.srf"
        write_srf(path, srf_file)
        reloaded = read_srf(path)

        assert reloaded.version == "2.0"
        # The format stores these as text at fixed precision, so the comparison is
        # to that precision rather than exact.
        assert reloaded.points["slip"].to_numpy() == pytest.approx(
            generated.slip_cm, rel=1e-4
        )
        assert reloaded.points["tinit"].to_numpy() == pytest.approx(
            generated.onset_s, abs=1e-4
        )
        assert reloaded.points["vs"].to_numpy() == pytest.approx(3.2)

    def test_the_reloaded_pulses_still_integrate_to_the_slip(self, tmp_path) -> None:
        generated, srf_file = assemble()
        path = tmp_path / "rupture.srf"
        write_srf(path, srf_file)
        reloaded = read_srf(path)

        integrals = reloaded.slipt1_array.sum(axis=1) * generated.sample_interval_s
        assert integrals == pytest.approx(generated.slip_cm, rel=1e-3)


class TestRefusesMismatchedInput:
    def test_a_geometry_of_the_wrong_size_is_refused(self) -> None:
        generated = rupture()
        where = geometry()
        short = SubfaultGeometry(
            longitude_deg=where.longitude_deg[:-1],
            latitude_deg=where.latitude_deg[:-1],
            depth_km=where.depth_km[:-1],
            strike_deg=where.strike_deg[:-1],
            dip_deg=where.dip_deg[:-1],
            area_cm2=where.area_cm2[:-1],
        )
        with pytest.raises(ValueError, match="geometry describes"):
            to_srf_file(
                generated,
                short,
                header(),
                shear_speed_km_s=np.full(SUBFAULTS, 3.2, dtype=np.float32),
                density_g_cm3=np.full(SUBFAULTS, 2.6, dtype=np.float32),
            )

    def test_ragged_geometry_arrays_are_refused(self) -> None:
        where = geometry()
        with pytest.raises(ValueError, match="disagree on length"):
            SubfaultGeometry(
                longitude_deg=where.longitude_deg,
                latitude_deg=where.latitude_deg[:-1],
                depth_km=where.depth_km,
                strike_deg=where.strike_deg,
                dip_deg=where.dip_deg,
                area_cm2=where.area_cm2,
            )

    def test_mismatched_material_properties_are_refused(self) -> None:
        generated = rupture()
        with pytest.raises(ValueError, match="shear_speed_km_s"):
            to_srf_file(
                generated,
                geometry(),
                header(),
                shear_speed_km_s=np.full(3, 3.2, dtype=np.float32),
                density_g_cm3=np.full(SUBFAULTS, 2.6, dtype=np.float32),
            )


# Deliberately not asserted:
#
# - Where the plane centre is. This module does no geodesy; the caller supplies the
#   header because only it knows how the mesh was discretised. genslip recomputes it
#   from a width and a dip with a tangent-plane approximation that is off by a
#   kilometre at subduction scale -- see DEFECTS.md and geodesy.rs.
