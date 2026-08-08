"""Tests for assembling an SRF from a generated rupture.

Two things can go wrong here and nowhere else: a column can be filled from the wrong
array, and the ragged slip-rate pulses can be indexed wrongly on the way into a
sparse matrix. Both survive a round trip through the file unless something checks.
"""

from pathlib import Path

import numpy as np
import pytest

from rupture_generator import (
    FaultGrid,
    PointSourceSpec,
    Ramp,
    SlipRateShape,
    SlipSpec,
    SourceSpec,
    SpectrumModel,
    TimingSpec,
    VelocityModel1D,
    generate_point_source,
    generate_rupture,
)
from rupture_generator.assemble import SubfaultGeometry, to_srf_file
from rupture_generator.srf import PlaneHeader, read_srf, write_srf

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
        depth_km=np.array([0.5 + i * 2.0 for i in range(DIP)], dtype=np.float64),
        base_rake_deg=np.full(SUBFAULTS, 175.0, dtype=np.float64),
        velocity_fraction=np.full(SUBFAULTS, 0.8, dtype=np.float64),
    )
    model = VelocityModel1D(
        np.array([1.0, 5.0, 12.0, 30.0], dtype=np.float64),
        np.array([1.8, 2.6, 3.2, 3.6], dtype=np.float64),
        np.array([2.1, 2.4, 2.6, 2.7], dtype=np.float64),
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
        strike_deg=np.full(SUBFAULTS, 45.0, dtype=np.float64),
        dip_deg=np.full(SUBFAULTS, 60.0, dtype=np.float64),
        area_cm2=np.full(SUBFAULTS, 1.0e10, dtype=np.float64),
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
        shear_speed_km_s=np.full(SUBFAULTS, 3.2, dtype=np.float64),
        density_g_cm3=np.full(SUBFAULTS, 2.6, dtype=np.float64),
    )


class TestTheColumnsComeFromTheRightArrays:
    def test_the_physics_columns_are_the_rupture(self) -> None:
        generated, srf_file = assemble()
        assert srf_file.points.slip_cm == pytest.approx(generated.slip_cm)
        assert srf_file.points.rake_deg == pytest.approx(generated.rake_deg)
        assert srf_file.points.onset_s == pytest.approx(generated.onset_s)
        assert srf_file.points.rise_time_s == pytest.approx(generated.rise_time_s)

    def test_the_location_columns_are_the_geometry(self) -> None:
        _, srf_file = assemble()
        where = geometry()
        assert srf_file.points.longitude_deg == pytest.approx(where.longitude_deg)
        assert srf_file.points.latitude_deg == pytest.approx(where.latitude_deg)
        assert srf_file.points.depth_km == pytest.approx(where.depth_km)

    def test_the_shear_speed_is_converted_to_the_formats_unit(self) -> None:
        # The SRF stores vs in cm/s -- genslip writes `1.0e+05*vmod->vs[k]`
        # (gslip_srf_subs.c:1609) -- and a velocity model is in km/s. Writing the
        # km/s straight through understates it by a factor of 1e5, which nothing
        # downstream would flag: 3.2 is a plausible number for something.
        _, srf_file = assemble()
        assert srf_file.points.shear_speed_cm_s == pytest.approx(3.2e5)
        assert srf_file.points.density_g_cm3 == pytest.approx(2.6)

    def test_the_header_is_the_plane(self) -> None:
        _, srf_file = assemble()
        assert len(srf_file.planes) == 1
        plane = srf_file.planes[0]
        assert plane.strike_count == STRIKE
        assert plane.dip_count == DIP
        assert plane.dip_deg == pytest.approx(60.0)
        assert plane.hypocentre_dip_km == pytest.approx(4.0)


class TestTheSlipRatePulses:
    def test_each_row_is_its_subfault_pulse(self) -> None:
        generated, srf_file = assemble()
        offsets = np.asarray(generated.slip_rate_offsets, dtype=np.int64)
        dense = srf_file.slip_rate.toarray()
        for index in range(0, SUBFAULTS, 5):
            start, end = int(offsets[index]), int(offsets[index + 1])
            expected = generated.slip_rate[start:end]
            assert dense[index, : len(expected)] == pytest.approx(expected)
            # Everything past the pulse is padding, not signal.
            assert dense[index, len(expected) :] == pytest.approx(0.0)

    def test_every_row_integrates_to_its_slip(self) -> None:
        generated, srf_file = assemble()
        integrals = srf_file.slip_rate.sum(axis=1) * generated.sample_interval_s
        assert integrals == pytest.approx(generated.slip_cm, rel=1e-3)

    def test_the_matrix_is_only_as_wide_as_the_longest_pulse(self) -> None:
        generated, srf_file = assemble()
        offsets = np.asarray(generated.slip_rate_offsets, dtype=np.int64)
        assert srf_file.slip_rate.shape == (SUBFAULTS, int(np.diff(offsets).max()))


class TestRoundTrip:
    def test_writing_and_reading_preserves_the_model(self, tmp_path: Path) -> None:
        generated, srf_file = assemble()
        path = tmp_path / "rupture.srf"
        write_srf(path, srf_file)
        reloaded = read_srf(path)

        assert reloaded.version == "2.0"
        # The format stores these as text at fixed precision, so the comparison is
        # to that precision rather than exact.
        assert reloaded.points.slip_cm == pytest.approx(generated.slip_cm, rel=1e-4)
        assert reloaded.points.onset_s == pytest.approx(generated.onset_s, abs=1e-4)
        assert reloaded.points.shear_speed_cm_s == pytest.approx(3.2e5)

    def test_the_reloaded_pulses_still_integrate_to_the_slip(
        self, tmp_path: Path
    ) -> None:
        generated, srf_file = assemble()
        path = tmp_path / "rupture.srf"
        write_srf(path, srf_file)
        reloaded = read_srf(path)

        integrals = reloaded.slip_rate.sum(axis=1) * generated.sample_interval_s
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
                shear_speed_km_s=np.full(SUBFAULTS, 3.2, dtype=np.float64),
                density_g_cm3=np.full(SUBFAULTS, 2.6, dtype=np.float64),
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
                shear_speed_km_s=np.full(3, 3.2, dtype=np.float64),
                density_g_cm3=np.full(SUBFAULTS, 2.6, dtype=np.float64),
            )


class TestAPointSourceAssemblesTheSameWay:
    """The end of the argument that a point source is not a separate program.

    `generic_slip2srf` exists to turn a slip distribution into an SRF. So does this
    module. If a point source needed anything here that a finite fault does not, the
    reuse would stop at the generator and this file would need a second path.

    It does not: a `GeneratedRupture` is a `GeneratedRupture`, and `to_srf_file` is
    called with the same five arguments. That is what these assert.
    """

    @staticmethod
    def point_rupture(shape: SlipRateShape | None = None):
        grid = FaultGrid(
            STRIKE,
            DIP,
            16,
            10,
            1.0,
            1.0,
            depth_km=np.array([0.5 + i * 2.0 for i in range(DIP)], dtype=np.float64),
            base_rake_deg=np.full(SUBFAULTS, 175.0, dtype=np.float64),
            velocity_fraction=np.full(SUBFAULTS, 0.8, dtype=np.float64),
        )
        shallow, deep = Ramp(6.5, 1.5), Ramp(17.5, 2.5)
        return generate_point_source(
            grid,
            VelocityModel1D(
                np.array([1.0, 5.0, 12.0, 30.0], dtype=np.float64),
                np.array([1.8, 2.6, 3.2, 3.6], dtype=np.float64),
                np.array([2.1, 2.4, 2.6, 2.7], dtype=np.float64),
            ),
            PointSourceSpec(5.2, 0.35, average_dip_deg=60.0, average_rake_deg=175.0),
            TimingSpec(
                rupture_time_scale=-0.35,
                rise_time_blend=Ramp(2.0, 1.0),
                shallow_ramp=shallow,
                deep_ramp=deep,
                beta_shallow_ramp=Ramp(2.0, 1.0),
                beta_mid_ramp=Ramp(6.5, 1.5),
                slip_rate_shape=shape,
            ),
            hypocentre_strike=STRIKE // 2,
            hypocentre_dip=DIP // 2,
        )

    def assemble_point(self, shape: SlipRateShape | None = None):
        generated = self.point_rupture(shape)
        return generated, to_srf_file(
            generated,
            geometry(),
            header(),
            shear_speed_km_s=np.full(SUBFAULTS, 3.2, dtype=np.float64),
            density_g_cm3=np.full(SUBFAULTS, 2.6, dtype=np.float64),
        )

    def test_it_writes_one_point_per_subfault(self) -> None:
        _, srf = self.assemble_point()
        assert len(srf.points.slip_cm) == SUBFAULTS
        assert len(srf.planes) == 1

    def test_the_slip_is_the_same_everywhere(self) -> None:
        # A point source's slip is uniform, so the SRF's is too -- the assembler
        # copies rather than recomputing, and this is what says so.
        _, srf = self.assemble_point()
        assert np.all(srf.points.slip_cm == srf.points.slip_cm[0])
        assert srf.points.slip_cm[0] > 0.0

    def test_it_survives_a_round_trip_through_the_file(self, tmp_path: Path) -> None:
        _, srf = self.assemble_point()
        path = tmp_path / "point.srf"
        write_srf(path, srf)
        back = read_srf(path)
        assert np.allclose(back.points.slip_cm, srf.points.slip_cm, rtol=1e-5)
        assert np.allclose(back.points.onset_s, srf.points.onset_s, atol=1e-5)

    @pytest.mark.parametrize(
        "shape",
        [
            SlipRateShape.brune(),
            SlipRateShape.ucsb(),
            SlipRateShape.urs(),
            SlipRateShape.esg2006(),
            SlipRateShape.cos(),
            SlipRateShape.seki(),
            SlipRateShape.delta(),
        ],
    )
    def test_every_shape_reaches_the_file(
        self, shape: SlipRateShape, tmp_path: Path
    ) -> None:
        generated, srf = self.assemble_point(shape)
        path = tmp_path / "point.srf"
        write_srf(path, srf)
        back = read_srf(path)
        # The pulses are what the shape changes, so the round trip has to carry them.
        assert back.slip_rate.nnz > 0
        assert np.allclose(back.points.slip_cm, generated.slip_cm, rtol=1e-5)


# Deliberately not asserted:
#
# - Where the plane centre is. This module does no geodesy; the caller supplies the
#   header because only it knows how the mesh was discretised. genslip recomputes it
#   from a width and a dip with a tangent-plane approximation that is off by a
#   kilometre at subduction scale -- see DEFECTS.md and geodesy.rs.
# - That a point source and a finite fault produce the *same* SRF. They do not and
#   should not; what they share is the path from a rupture to a file.
