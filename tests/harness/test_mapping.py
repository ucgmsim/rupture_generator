"""The getpar-to-spec-group mapping, one group at a time.

Two kinds of check, and the difference matters:

- **Against the binary.** genslip reports what it derived on stderr -- `nstk2`,
  `ndip2`, `dstk`, `ddip`, `alphaT`, `rvfrac_avg`, `trise_avg` -- so for those the
  reference *is* the oracle and no transcription of the C is involved. These skip
  without `GENSLIP_BINARY`.
- **Against the C's expression.** For everything genslip does not print, the check
  transcribes the expression from the source with a line number and asserts the
  mapping evaluates the same thing. That is weaker, and it is marked as such by
  carrying the citation.

Neither kind checks the mapping against the port. That would be circular: the port's
correctness is what the mapping is being built to measure.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from rupture_generator import Ramp, RiseTimeWeighting, SpectrumModel
from tests.harness import mapping
from tests.harness.genslip_config import (
    CustomCorrelationCorners,
    FaultGeometryLimits,
    KModel,
    RiseTimeNormalisation,
    SpatialFiltering,
)
from tests.harness.genslip_reference import parse_diagnostics, write_velocity_model
from tests.harness.test_genslip_reference import (
    BOTTOM_DEPTH_KM,
    DENSITY_G_CM3,
    DIP,
    SHEAR_SPEED_KM_S,
    STRIKE,
    geometry,
    run,
)
from tests.harness.test_unroll import _make_minimal_params

MAGNITUDE = 6.2
HYPOCENTRE_STRIKE_KM = 0.0
HYPOCENTRE_DIP_KM = 3.0

genslip = pytest.mark.skipif(
    not os.environ.get("GENSLIP_BINARY"),
    reason="set GENSLIP_BINARY to a genslip v5.6.2 built with -std=gnu17",
)


def parameters(**overrides):
    """The fixture's getpar set: the same one `test_genslip_reference.run` drives."""
    return _make_minimal_params(
        read_gsf=True, read_erf=False, alpha_rough=0.0, **overrides
    )


def derived(**overrides):
    """`Derived` for the fixture geometry."""
    return mapping.derive(
        geometry(),
        parameters(**overrides),
        magnitude=MAGNITUDE,
        strike_count=STRIKE,
        dip_count=DIP,
    )


class TestTheDiagnosticParser:
    def test_the_median_magnitude_does_not_shadow_the_magnitude(self) -> None:
        # genslip prints both on one line and a diagnostic name cannot contain a
        # space, so `median mag= 5.78` parses as `mag` too. Last-wins would report
        # 5.78 as the magnitude: a plausible number, and the wrong one.
        parsed = parse_diagnostics(b"mag= 6.20 median mag= 5.78 nslip= 1 nhypo= 1\n")
        assert parsed["mag"] == 6.20
        assert parsed["nslip"] == 1.0

    def test_it_reads_the_shapes_genslip_prints(self) -> None:
        parsed = parse_diagnostics(
            b"nstk2= 22 ndip2= 14\n"
            b"trise_avg= 0.6702 rvfrac_avg= 0.8010 alphaT= 0.9988\n"
            b"mom=   2.48314e+25 avgslip= 218 maxslip= 637\n"
        )
        assert parsed["nstk2"] == 22.0
        assert parsed["alphaT"] == 0.9988
        assert parsed["mom"] == pytest.approx(2.48314e25)


class TestFaultGrid:
    """Spec group 1."""

    def test_the_padded_extents_are_the_ones_the_readme_names(self) -> None:
        # The rule is `(int)(flen_max*extend_fac/flen * nstk)` rounded up to even,
        # with extend_fac defaulting to 1.10 -- not the fault rounded up. 20 scales to
        # 22 and stays; 12 scales to 13.2, truncates to 13, and is bumped to 14.
        assert derived().padded_strike == 22
        assert derived().padded_dip == 14

    def test_a_bigger_parent_fault_widens_the_padding(self) -> None:
        # The whole reason extend_fac exists: this segment is part of a longer
        # rupture, so its spectra are evaluated over the parent's length.
        wider = derived(
            fault_geometry_limits=FaultGeometryLimits(along_strike_length=40.0)
        )
        assert wider.padded_strike == 88  # (int)(40*1.10/10 * 20) = 88
        assert wider.padded_dip == 14  # fwid_max still clamps up to fwid

    def test_the_limits_clamp_up_rather_than_down(self) -> None:
        # `if(flen_max < flen) flen_max = flen` (line 1230). A caller who passes a
        # smaller parent than the segment gets the segment, not a narrower spectrum.
        assert (
            derived(
                fault_geometry_limits=FaultGeometryLimits(along_strike_length=1.0)
            ).padded_strike
            == derived().padded_strike
        )

    def test_the_extents_are_even(self) -> None:
        # The generators address the Nyquist row and column directly, which is why
        # genslip rounds up and why FaultGrid refuses an odd extent.
        result = derived()
        assert result.padded_strike % 2 == 0
        assert result.padded_dip % 2 == 0

    def test_depth_is_one_row_at_a_time_from_the_first_subfault(self) -> None:
        subfaults = geometry()
        depths = subfaults.depth_by_row_km(STRIKE)
        assert len(depths) == DIP
        assert depths == pytest.approx(subfaults.depth_km[::STRIKE])
        # A plane dipping into the earth: every row deeper than the one above.
        assert np.all(np.diff(depths) > 0.0)

    def test_the_velocity_fraction_carries_the_alpha_t_division(self) -> None:
        # genslip divides both `rvfrac` and every `psrc[j].rvf` by alphaT (lines
        # 1443-1445); the port applies alphaT to rise time only. Without this the
        # rupture is uniformly ~0.1% too slow, which looks like nothing and is not.
        result = derived()
        grid = mapping.fault_grid(
            geometry(),
            parameters(),
            result,
            strike_count=STRIKE,
            dip_count=DIP,
        )
        assert grid.subfault_count == STRIKE * DIP
        expected = mapping.DEFAULT_VELOCITY_FRACTION / result.alpha_t
        assert expected == pytest.approx(0.80098765, abs=5e-8)

    def test_the_grid_is_the_shape_it_was_given(self) -> None:
        grid = mapping.fault_grid(
            geometry(), parameters(), derived(), strike_count=STRIKE, dip_count=DIP
        )
        assert grid.subfault_count == STRIKE * DIP

    @genslip
    def test_the_padded_extents_are_the_binary_s(self) -> None:
        reported = run().diagnostics
        result = derived()
        assert result.padded_strike == reported["nstk2"]
        assert result.padded_dip == reported["ndip2"]

    @genslip
    def test_the_subfault_spacing_is_the_binary_s(self) -> None:
        reported = run().diagnostics
        subfaults = geometry()
        assert subfaults.mean_along_strike_km == pytest.approx(reported["dstk"])
        assert subfaults.mean_down_dip_km == pytest.approx(reported["ddip"])

    @genslip
    def test_the_velocity_fraction_is_the_binary_s(self) -> None:
        # `rvfrac_avg` is printed after the alphaT division, so this is the direct
        # check that the mapping divides and does not merely pass 0.8 through.
        reported = run().diagnostics
        result = derived()
        fraction = mapping.DEFAULT_VELOCITY_FRACTION / result.alpha_t
        assert fraction == pytest.approx(reported["rvfrac_avg"], abs=5e-5)
        assert fraction != pytest.approx(mapping.DEFAULT_VELOCITY_FRACTION, abs=5e-5), (
            "the division by alphaT has to be visible at this precision, or the test proves nothing"
        )


class TestTheHypocentreConversion:
    """`shypo`/`dhypo` are kilometres; the port takes 0-based subfault indices.

    genslip's `ixs`/`iys` count from one, because their only use is the Fortran
    solver's argument list. Returning them unconverted is `DEFECTS.md` 17, and it cost
    the whole onset field: every subfault ruptured as though the hypocentre were one
    cell further along strike and one further down dip.
    """

    def test_the_centre_of_the_fault_is_the_middle_subfault(self) -> None:
        # shypo is measured from the CENTRE and is signed, so 0.0 is the middle --
        # not the first subfault, which is what a proportion or an index would give.
        # An even grid has no middle subfault, and genslip's rounding takes the lower
        # of the two: `ixs = 10` is 1-based, so subfault 9 of 0..19.
        assert mapping.hypocentre_indices(0.0, 3.0, geometry(), derived()) == (9, 5)

    def test_the_far_end_of_the_fault_is_its_last_subfault(self) -> None:
        # The check that the conversion is 0-based, and the one the old convention
        # could not pass: it returned `(20, 12)` for a 20x12 fault -- one past the
        # last subfault in both directions, on a grid whose valid indices stop at
        # (19, 11). Nothing downstream complained.
        assert mapping.hypocentre_indices(5.0, 6.0, geometry(), derived()) == (19, 11)

    def test_the_near_end_of_the_fault_is_refused(self) -> None:
        # The other end does not have an index. genslip's rounding gives `ixs = 0`,
        # which is not a Fortran index at all; its padding then carries the source a
        # cell outside the fault, and an unsigned subfault index cannot say that.
        # Refused rather than clamped, because clamping is a different rupture.
        with pytest.raises(ValueError, match="OFF the near edge"):
            mapping.hypocentre_indices(-5.0, 3.0, geometry(), derived())
        with pytest.raises(ValueError, match="OFF the near edge"):
            mapping.hypocentre_indices(0.0, 0.0, geometry(), derived())

    def test_it_is_not_a_proportion(self) -> None:
        # `genslip_config.Hypocentre` used to describe these as proportions. Under
        # that reading the fixture's dhypo=3.0 would be off the fault; under the
        # correct one it is halfway down a 6 km width.
        strike, dip = mapping.hypocentre_indices(0.0, 3.0, geometry(), derived())
        assert dip == DIP // 2 - 1
        assert strike == STRIKE // 2 - 1

    @pytest.mark.parametrize(
        ("strike_km", "dip_km"),
        [(-6.0, 3.0), (6.0, 3.0), (0.0, -0.5), (0.0, 7.0)],
    )
    def test_a_hypocentre_off_the_fault_is_refused(
        self, strike_km: float, dip_km: float
    ) -> None:
        # genslip checks the same bounds at line 3155 and writes no rupture. Refusing
        # here means the fixture fails where it was written, not three stages later.
        with pytest.raises(ValueError, match="outside"):
            mapping.hypocentre_indices(strike_km, dip_km, geometry(), derived())


class TestVelocityModel:
    """Spec group 2: the one group with no getpar names in it."""

    def test_it_is_the_arrays_the_reference_file_is_written_from(
        self, tmp_path: Path
    ) -> None:
        # Not a translation -- the assertion is that both sides get one model. The
        # file stores *thicknesses* and the port takes layer *bottoms*, so this reads
        # the file back and re-accumulates: the one place the two representations
        # could disagree without either side looking wrong.
        path = tmp_path / "velocity_model.1d"
        write_velocity_model(BOTTOM_DEPTH_KM, SHEAR_SPEED_KM_S, DENSITY_G_CM3, path)

        rows = [line.split() for line in path.read_text().splitlines()[1:]]
        bottoms = np.cumsum([float(row[0]) for row in rows])
        assert bottoms == pytest.approx(BOTTOM_DEPTH_KM)
        assert [float(row[2]) for row in rows] == pytest.approx(SHEAR_SPEED_KM_S)
        assert [float(row[3]) for row in rows] == pytest.approx(DENSITY_G_CM3)

        assert mapping.velocity_model(BOTTOM_DEPTH_KM, SHEAR_SPEED_KM_S, DENSITY_G_CM3)

    def test_mismatched_layer_counts_are_refused(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            mapping.velocity_model(
                BOTTOM_DEPTH_KM, SHEAR_SPEED_KM_S[:-1], DENSITY_G_CM3
            )


class TestSourceSpec:
    """Spec group 3."""

    def test_the_moment_is_the_c_s_expression(self) -> None:
        # (orig. genslip_v5.6.2.c:1250)
        expected = float(
            np.float32(math.exp(math.log(10.0) * 1.5 * (MAGNITUDE + 10.73)))
        )
        assert mapping.seismic_moment(MAGNITUDE, True) == expected

    def test_the_magnitude_scale_moves_the_moment_by_a_tenth(self) -> None:
        # use_Mw picks 10.73 over 10.7. That is 0.03 in a base-10 exponent scaled by
        # 1.5, so ~10% in moment -- the point of having both scales.
        moment = mapping.seismic_moment(MAGNITUDE, True)
        local = mapping.seismic_moment(MAGNITUDE, False)
        assert local / moment == pytest.approx(10.0 ** (-1.5 * 0.03), rel=1e-6)

    def test_the_corner_defaults_follow_the_kmodel(self) -> None:
        # Not one pair with one default: three pairs, set immediately before the
        # getpar that may override them (lines 994-1035).
        assert mapping.corner_offsets(parameters(kmodel=KModel.MAI)) == (2.50, 1.50)
        assert mapping.corner_offsets(parameters(kmodel=KModel.SUZUKI)) == (1.67, 1.69)

    def test_a_custom_corner_overrides_the_default(self) -> None:
        given = parameters(
            kmodel=KModel.MAI,
            custom_correlation_corners=CustomCorrelationCorners(
                along_strike_corner=2.9, downdip_corner=1.1
            ),
        )
        assert mapping.corner_offsets(given) == (2.9, 1.1)

    def test_the_hybrid_model_ignores_the_corners_it_reads(self) -> None:
        # Line 996 defaults kx_corner for MAI_SOMERVILLE and line 998 lets the user
        # change it -- and then lines 1341-1342 evaluate literal 2.50 and 1.50. A
        # custom corner with this kmodel changes genslip's output not at all, and the
        # mapping has to reproduce the *use* rather than honour the read.
        hybrid = parameters(
            kmodel=KModel.MAI_SOMERVILLE,
            custom_correlation_corners=CustomCorrelationCorners(
                along_strike_corner=9.9, downdip_corner=9.9
            ),
        )
        assert mapping.corner_offsets(hybrid) == (2.50, 1.50)
        # ...and the same custom corner under plain Mai *is* honoured, so this is the
        # kmodel's quirk and not the mapping ignoring the parameter everywhere.
        assert mapping.corner_offsets(
            parameters(
                kmodel=KModel.MAI,
                custom_correlation_corners=CustomCorrelationCorners(
                    along_strike_corner=9.9, downdip_corner=9.9
                ),
            )
        ) == (9.9, 9.9)

    def test_somerville_carries_its_offsets_in_the_port(self) -> None:
        # Somerville's 1.72 and 1.93 are inline double literals in the C, not
        # variables, so there is nothing for a kx_corner to override and the port's
        # CornerRelation::Somerville takes no offsets. Zeros say "unread".
        assert mapping.corner_offsets(parameters(kmodel=KModel.SOMERVILLE)) == (
            0.0,
            0.0,
        )

    def test_the_magnitude_exponents_default_to_a_half(self) -> None:
        assert mapping.magnitude_exponents(parameters()) == (0.5, 0.5)
        assert mapping.magnitude_exponents(
            parameters(
                custom_correlation_corners=CustomCorrelationCorners(
                    along_strike_exponent=0.4, downdip_exponent=0.6
                )
            )
        ) == (0.4, 0.6)

    def test_input_corners_are_mandatory(self) -> None:
        # genslip reads them with mstpar (lines 1027-1028) and exits without them.
        with pytest.raises(mapping.UnmappableConfigurationError, match="mstpar"):
            mapping.corner_offsets(parameters(kmodel=KModel.INPUT_CORNERS))

    def test_input_corners_are_used_when_given(self) -> None:
        given = parameters(
            kmodel=KModel.INPUT_CORNERS,
            custom_correlation_corners=CustomCorrelationCorners(
                along_strike_corner=2.1, downdip_corner=1.9
            ),
        )
        assert mapping.corner_offsets(given) == (2.1, 1.9)

    def test_the_spectrum_model_is_the_kmodel(self) -> None:
        for kmodel, model in [
            (KModel.SOMERVILLE, SpectrumModel.Somerville),
            (KModel.MAI, SpectrumModel.Mai),
            (KModel.SUZUKI, SpectrumModel.Suzuki),
        ]:
            assert mapping._KMODEL_TO_SPECTRUM[kmodel] == model

    def test_it_builds_for_the_fixture(self) -> None:
        assert mapping.source_spec(geometry(), parameters(), magnitude=MAGNITUDE)

    @genslip
    def test_the_kmodel_reaches_the_binary(self) -> None:
        assert run().diagnostics["kmodel"] == float(KModel.MAI)

    @genslip
    def test_alpha_t_is_the_binary_s(self) -> None:
        # alphaT is derived from avgdip and avgrak, both of which the mapping reads
        # off the GSF rather than being told -- so agreement here checks the GSF
        # averaging as well as the correction itself.
        reported = run().diagnostics
        assert derived().alpha_t == pytest.approx(reported["alphaT"], abs=5e-5)

    @genslip
    def test_the_average_rise_time_is_the_binary_s(self) -> None:
        # `trise_avg` is `risetime_coef * 1e-9 * mom^(1/3) * alphaT`, so this checks
        # the moment, the rise-time coefficient and alphaT in one number.
        reported = run().diagnostics
        result = derived()
        trise = 2.3 * (1.0e-9 * math.exp(math.log(result.moment_dyne_cm) / 3.0))
        assert trise * result.alpha_t == pytest.approx(reported["trise_avg"], abs=5e-5)


class TestTheBoundaryGapsThatWereFixed:
    """`DEFECTS.md` 11-13, each now expressible.

    These were found by building the mapping and are pinned here so they cannot come
    back quietly. Each was a *boundary* gap -- `crates/genslip` modelled all three
    correctly the whole time -- and each would otherwise have compared two different
    models and reported the difference as a port defect.

    What these can assert from Python is that the configuration is accepted. That the
    resulting corners are the right *numbers* is `source_parity.rs`'s job, since the
    spec groups expose no getters; `correlation_lengths_match_for_every_relation` and
    `circular_relations_give_equal_corners` are the tests that carry it.
    """

    def test_circular_average_is_expressible(self) -> None:
        # Not merely equal corners: under Somerville the original switches to a third
        # offset, 1.825 rather than 1.72 and 1.93.
        assert mapping.source_spec(
            geometry(), parameters(circular_average=True), magnitude=MAGNITUDE
        )
        assert mapping.source_spec(
            geometry(),
            parameters(kmodel=KModel.SOMERVILLE, circular_average=True),
            magnitude=MAGNITUDE,
        )

    def test_frankel_is_expressible(self) -> None:
        # genslip:1303 is `kmodel == MAI_FLAG || kmodel == FRANKEL_FLAG`, so Frankel
        # shares the *Mai* corner relation while keeping its own falloff shape.
        assert mapping.source_spec(
            geometry(), parameters(kmodel=KModel.FRANKEL), magnitude=MAGNITUDE
        )

    def test_frankel_defaults_to_mai_s_corners(self) -> None:
        # The tell that it takes Mai's relation and not Somerville's: it reads
        # kx_corner/ky_corner, and they default to Mai's pair.
        assert mapping.corner_offsets(parameters(kmodel=KModel.FRANKEL)) == (2.50, 1.50)
        assert mapping.corner_offsets(parameters(kmodel=KModel.SOMERVILLE)) == (
            0.0,
            0.0,
        )

    def test_the_speed_ramps_can_differ_from_the_rise_time_ramps(self) -> None:
        # The case that was unrepresentable: genslip's shal_vrup_dep stays at its own
        # default when risetimedep moves, and a shared ramp would move both.
        given = parameters()
        given.rise_time.shallow_center_depth = 10.0
        assert given.rupture_velocity.shallow_center_depth == 6.5
        assert mapping.timing_spec(
            geometry(), given, derived(), hypocentre_dip_km=HYPOCENTRE_DIP_KM
        )

    def test_the_two_deep_ramps_take_the_same_adjustment_with_their_own_widths(
        self,
    ) -> None:
        # Both are pushed down to the hypocentre depth, each using its own half-width
        # (lines 2378-2381 and 2974-2977). Equal centres and unequal widths give
        # unequal ramps -- which one shared `deep_ramp` could not represent.
        subfaults = geometry()
        rise = mapping.deep_ramp_centre_km(subfaults, 17.5, 2.5, hypocentre_dip_km=20.0)
        speed = mapping.deep_ramp_centre_km(
            subfaults, 17.5, 4.0, hypocentre_dip_km=20.0
        )
        assert speed > rise
        assert speed - rise == pytest.approx(1.5, abs=1e-4)

    def test_they_agree_when_their_widths_do(self) -> None:
        # Which is the configured case, and why this was latent rather than loud.
        subfaults = geometry()
        assert mapping.deep_ramp_centre_km(
            subfaults, 17.5, 2.5, hypocentre_dip_km=20.0
        ) == mapping.deep_ramp_centre_km(subfaults, 17.5, 2.5, hypocentre_dip_km=20.0)


class TestSlipSpec:
    """Spec group 4."""

    def test_the_minimum_wavelength_tracks_the_grid_not_a_constant(self) -> None:
        # `2*sqrt(dstk*ddip)/0.8` -- 80% of Nyquist (line 1237). Halving the subfault
        # size halves this. A hardcoded number would filter at the wrong scale on
        # every grid but the one it was read off.
        assert mapping.minimum_wavelength_km(geometry(), parameters()) == pytest.approx(
            1.25
        )

    def test_a_given_minimum_wavelength_wins(self) -> None:
        given = parameters(spatial_filtering=SpatialFiltering(rake_min_wavelength=3.0))
        assert mapping.minimum_wavelength_km(geometry(), given) == 3.0

    def test_the_maximum_wavelength_is_hardwired(self) -> None:
        # Line 1235 assigns 1.0e+15 after the getpar that reads it, so no user value
        # reaches the filter. The port's own default of 80 km would band-limit a
        # spectrum genslip leaves alone -- so this constant is a correction to that
        # default, not a restatement of it.
        assert mapping.HARDWIRED_MAX_WAVELENGTH_KM == 1.0e15
        assert mapping.HARDWIRED_MAX_WAVELENGTH_KM > 80.0

    def test_the_roughness_wavelengths_are_not_the_slip_ones(self) -> None:
        # lambda_min/lambda_max belong to the roughness field. Setting them must not
        # move the slip spectrum's band, or the two pairs have been conflated.
        roughened = parameters(
            spatial_filtering=SpatialFiltering(
                roughness_min_wavelength=0.02, roughness_max_wavelength=4.0
            )
        )
        assert mapping.minimum_wavelength_km(
            geometry(), roughened
        ) == mapping.minimum_wavelength_km(geometry(), parameters())

    def test_the_water_level_sentinel_survives(self) -> None:
        # genslip's default is -1 and every use is guarded by `> 0`; the port spells
        # disabled the same way, so the sentinel carries rather than needing a flag.
        assert mapping.slip_water_level(parameters()) == -1.0
        assert mapping.slip_water_level(parameters(slip_water_level=0.05)) == 0.05

    def test_it_builds_for_the_fixture(self) -> None:
        assert mapping.slip_spec(geometry(), parameters())


class TestTimingSpec:
    """Spec group 5."""

    def test_the_rupture_time_scale_is_the_moment_relation(self) -> None:
        # tsfac_main = tsfac_bzero + tsfac_slope * 1e-9 * mom^(1/3) (line 1257).
        result = derived()
        cube_root = float(
            np.float32(1.0e-9 * math.exp(math.log(result.moment_dyne_cm) / 3.0))
        )
        expected = float(np.float32(-0.1 + -0.5 * cube_root))
        assert mapping.rupture_time_scale(parameters(), result) == expected

    def test_the_rupture_time_scale_is_negative(self) -> None:
        # Which is what makes high-slip patches rupture early. A positive value here
        # would invert the correlation and still produce a plausible-looking rupture.
        assert mapping.rupture_time_scale(parameters(), derived()) < 0.0

    def test_a_given_tsfac_main_wins(self) -> None:
        given = parameters()
        given.rupture_time_perturbation.main_value = -0.4
        assert mapping.rupture_time_scale(given, derived()) == -0.4

    def test_the_deep_ramp_follows_a_deep_hypocentre(self) -> None:
        # `xhypo = dhypo*sin(avgdip*rperd) + dtop + deep_risetimedep_range`, and the
        # ramp centre is the larger of that and the configured depth (lines
        # 2378-2381). A hypocentre at 3 km leaves the 17.5 km default alone; one at
        # 20 km down dip pushes the ramp below itself.
        shallow = mapping.deep_ramp_centre_km(
            geometry(), 17.5, 2.5, hypocentre_dip_km=3.0
        )
        assert shallow == 17.5

        deep = mapping.deep_ramp_centre_km(
            geometry(), 17.5, 2.5, hypocentre_dip_km=20.0
        )
        assert deep > 17.5
        assert deep == pytest.approx(
            20.0 * math.sin(geometry().mean_dip_deg * 0.017453293)
            + geometry().top_depth_km
            + 2.5,
            abs=1e-4,
        )

    def test_the_rise_time_sigma_defaults_to_the_slip_sigma(self) -> None:
        # `rtime1_sigma = slip_sigma` (line 1058), assigned before the getpar that
        # may override it -- so a group-4 parameter silently sets a group-5 one.
        sigma, _, _ = mapping.rise_time_perturbation_defaults(
            parameters(slip_sigma=0.31)
        )
        assert sigma == 0.31

    def test_an_explicit_rise_time_sigma_wins(self) -> None:
        given = parameters(slip_sigma=0.31)
        given.rise_time_perturbation.level1_sigma = 0.9
        sigma, _, _ = mapping.rise_time_perturbation_defaults(given)
        assert sigma == 0.9

    def test_the_blend_ramp_defaults_to_the_beta_shallow_ramp(self) -> None:
        # `rtime1_depth = stfparams.beta_shal_depth` and likewise the range (lines
        # 1063-1064). Another cross-group default: moving the beta shallow ramp moves
        # the rise-time blend with it.
        given = parameters()
        _, depth, depth_range = mapping.rise_time_perturbation_defaults(given)
        assert depth == given.beta.shallow_depth == 2.0
        assert depth_range == given.beta.shallow_depth_range == 1.0

    def test_moving_the_beta_ramp_moves_the_blend(self) -> None:
        given = parameters()
        given.beta.shallow_depth = 4.0
        given.beta.shallow_depth_range = 2.0
        _, depth, depth_range = mapping.rise_time_perturbation_defaults(given)
        assert (depth, depth_range) == (4.0, 2.0)

    def test_the_weighting_is_the_svr_wt_mode(self) -> None:
        # svr_wt is an int mode selector, not a weight, despite the name.
        assert (
            mapping._SVR_WT_TO_WEIGHTING[RiseTimeNormalisation.UNWEIGHTED_MEAN]
            == RiseTimeWeighting.Uniform
        )
        assert (
            mapping._SVR_WT_TO_WEIGHTING[RiseTimeNormalisation.SLIP_WEIGHTED]
            == RiseTimeWeighting.BySlip
        )

    def test_the_sample_cap_is_ntmax(self) -> None:
        # genslip assigns NTMAX at line 680 and never reads it from getpar.
        assert mapping.MAX_SLIP_RATE_SAMPLES == 100_000

    def test_it_builds_for_the_fixture(self) -> None:
        assert mapping.timing_spec(
            geometry(), parameters(), derived(), hypocentre_dip_km=HYPOCENTRE_DIP_KM
        )


class TestEveryGroupTogether:
    def test_all_five_build_from_one_parameter_set(self) -> None:
        # The point of the whole module: one `Parameters` renders both as arguments
        # for the binary and as the five groups the library takes.
        subfaults = geometry()
        given = parameters()
        result = mapping.derive(
            subfaults, given, magnitude=MAGNITUDE, strike_count=STRIKE, dip_count=DIP
        )

        assert mapping.fault_grid(
            subfaults, given, result, strike_count=STRIKE, dip_count=DIP
        )
        assert mapping.velocity_model(BOTTOM_DEPTH_KM, SHEAR_SPEED_KM_S, DENSITY_G_CM3)
        assert mapping.source_spec(subfaults, given, magnitude=MAGNITUDE)
        assert mapping.slip_spec(subfaults, given)
        assert mapping.timing_spec(
            subfaults, given, result, hypocentre_dip_km=HYPOCENTRE_DIP_KM
        )
        assert mapping.hypocentre_indices(
            HYPOCENTRE_STRIKE_KM, HYPOCENTRE_DIP_KM, subfaults, result
        ) == (9, 5)

    def test_a_ramp_is_a_ramp(self) -> None:
        ramp = Ramp(6.5, 1.5)
        assert ramp.centre_km == pytest.approx(6.5)
        assert ramp.half_width_km == pytest.approx(1.5)
