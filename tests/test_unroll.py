import dataclasses

import pytest

from rupture_generator.config import KModel, Stype
from rupture_generator.config import (
    AseismicParameters,
    BetaParameters,
    CustomCorrelationCorners,
    FaultGeometryLimits,
    FiniteDifferenceRupture,
    HybridCorrelationLength,
    MagnitudeArea,
    OutputOptions,
    Parameters,
    RiseTimeParameters,
    RiseTimePerturbation,
    RuptureTimePerturbation,
    SegmentDelay,
    SpatialFiltering,
    Tapering,
)


def _make_minimal_params(**overrides: object) -> Parameters:
    """Build a minimal Parameters instance with sensible defaults."""
    defaults = dict(
        resolution=0.5,
        dt=0.01,
        alpha_rough=0.005,
        perturb_subfault_location=True,
        slip_sigma=0.75,
        rake_sigma=15.0,
        fractal_rake=False,
        von_karman_order=4,
        magnitude_clamp=6.3,
        kmodel=KModel.MAI,
        use_moment_magnitude=True,
        use_median_mag=False,
        circular_average=False,
        modified_corners=False,
        mai_weight=0.5,
        somerville_weight=0.5,
        truncate_zero_slip=True,
        rupture_delay=0.0,
        rvfmin=0.25,
        rvfmax=1.414,
        xshift=0.0,
        yshift=0.0,
        read_erf=True,
        read_gsf=False,
        asperity_taper_factor=0.05,
        svr_wt=0.0,
        tapering=Tapering(side=0.02, bottom=0.0, top=0.0),
        beta=BetaParameters(
            shallow=0.5, deep=0.13, mid=0.13, asperity=0.3, sub_event=0.1,
            shallow_depth=2.0, shallow_depth_range=1.0, mid_depth=6.5, mid_depth_range=1.5,
        ),
        rise_time=RiseTimeParameters(
            coefficient=2.3, shallow_factor=2.0, shallow_center_depth=6.5,
            shallow_half_width=1.5, perturbation_sigma_ln=0.0, slip_scaling_factor=1.0,
            deep_factor=2.0, deep_center_depth=17.5, deep_half_width=2.5,
        ),
        rise_time_perturbation=RiseTimePerturbation(
            level1_slip_correlation=0.9, level2_roughness_correlation=0.5,
            level2_slip_exponent=0.5,
        ),
        rupture_time_perturbation=RuptureTimePerturbation(
            coefficient=1.1, intercept=-0.1, slope=-0.5, level1_sigma=1.0,
            level1_slip_correlation=0.8, level2_sigma=1.0e-10,
            level2_roughness_correlation=0.5, level2_lambda_max=5.0,
        ),
        hybrid_correlation_length=HybridCorrelationLength(
            enabled=False, kmodel=KModel.SUZUKI, factor=2.0, center_depth=6.5,
            center_depth_range=1.5, side_taper=0.08, shallow_weight_start=1.0,
            shallow_weight_end=0.0, deep_weight_start=0.0, deep_weight_end=1.0,
        ),
        aseismic=AseismicParameters(enabled=False, smooth=False, depth=10.0),
        finite_difference_rupture=FiniteDifferenceRupture(
            enabled=True, scale_speed_with_slip=False,
        ),
        segment_delay=SegmentDelay(
            enabled=False, boundary_zone_width=[], boundary_velocity_factor=[],
        ),
        spatial_filtering=SpatialFiltering(),
        custom_correlation_corners=CustomCorrelationCorners(),
        magnitude_area=MagnitudeArea(),
        output=OutputOptions(
            write_srf=True, write_gsf=False, srf_version="1.0",
            print_command=True, print_seed=True, dump_last_seed=False,
        ),
        fault_geometry_limits=FaultGeometryLimits(),
    )
    defaults.update(overrides)
    return Parameters(**defaults)


class TestUnroll:
    def test_returns_dict(self):
        p = _make_minimal_params()
        cmd = p.to_cmd()
        assert isinstance(cmd, dict)

    def test_flat_no_nested_dicts(self):
        p = _make_minimal_params()
        cmd = p.to_cmd()
        for v in cmd.values():
            assert not dataclasses.is_dataclass(v)

    def test_aliases_used_instead_of_field_names(self):
        """Fields with an alias metadata key should emit that alias, not the Python name."""
        p = _make_minimal_params()
        cmd = p.to_cmd()
        # Tapering.side has alias='side_taper' — should appear as side_taper
        assert cmd["side_taper"] == 0.02
        # OutputOptions.write_srf has no alias — should appear as write_srf
        assert cmd["write_srf"] is True

    def test_nested_values_unrolled(self):
        """Nested dataclass values should be flattened into the top-level dict."""
        p = _make_minimal_params(
            tapering=Tapering(side=0.1, bottom=0.2, top=0.3),
        )
        cmd = p.to_cmd()
        assert cmd["side_taper"] == 0.1
        assert cmd["bot_taper"] == 0.2
        assert cmd["top_taper"] == 0.3

    def test_all_aliases_match_genslip_parameter_names(self):
        """Every exported key should be a recognized genslip getpar name (spot-check)."""
        p = _make_minimal_params()
        cmd = p.to_cmd()
        # Check a representative subset of aliased fields
        expected_keys = {
            "side_taper", "bot_taper", "top_taper",
            "beta_shal", "beta_deep", "beta_asp", "beta_subevt",
            "risetime_coef", "risetimefac", "risetimedep", "risetimedep_range",
            "rt_rand", "rt_scalefac",
            "deep_risetimefac", "deep_risetimedep", "deep_risetimedep_range",
            "rtime1_scor", "rtime2_scor", "rtime2slip_exp",
            "tsfac_coef", "tsfac_bzero", "tsfac_slope",
            "tsfac1_sigma", "tsfac1_scor", "tsfac2_sigma", "tsfac2_scor",
            "hyb_corlen_flag", "hyb_corlen_kmodel", "hyb_corlen_fac",
            "aseis_flag", "aseis_smooth", "aseis_dep",
            "fdrup_time", "fdrup_scale_slip",
            "seg_delay", "gwid", "rvfac_seg",
            "lambda_min", "wavelength_max",
            "kx_corner", "xmag_exponent",
            "mag_area_Acoef",
            "write_srf", "srf_version", "dump_last_seed",
            "flen_max", "fwid_max", "extend_fac",
        }
        assert expected_keys.issubset(cmd.keys()), \
            f"Missing keys: {expected_keys - cmd.keys()}"

    def test_optional_none_fields_still_emitted(self):
        """Optional None-valued fields are still emitted (so the caller sees them)."""
        cmd = _make_minimal_params().to_cmd()
        assert "point_source_params" in cmd
        assert cmd["point_source_params"] is None

    def test_number_of_keys(self):
        cmd = _make_minimal_params().to_cmd()
        assert len(cmd) >= 100

    def test_to_cmd_is_deterministic(self):
        p1 = _make_minimal_params()
        p2 = _make_minimal_params()
        assert p1.to_cmd() == p2.to_cmd()


class TestValidation:
    def test_positive_accepts_good(self):
        Tapering(side=0.1, bottom=0.2, top=0.3)

    def test_positive_rejects_zero(self):
        with pytest.raises(ValueError, match="must be positive"):
            RiseTimeParameters(
                coefficient=0.0, shallow_factor=2.0, shallow_center_depth=6.5,
                shallow_half_width=1.5, perturbation_sigma_ln=0.0,
                slip_scaling_factor=1.0, deep_factor=2.0,
                deep_center_depth=17.5, deep_half_width=2.5,
            )

    def test_positive_rejects_negative(self):
        with pytest.raises(ValueError, match="must be positive"):
            RiseTimeParameters(
                coefficient=1.0, shallow_factor=-1.0, shallow_center_depth=6.5,
                shallow_half_width=1.5, perturbation_sigma_ln=0.0,
                slip_scaling_factor=1.0, deep_factor=2.0,
                deep_center_depth=17.5, deep_half_width=2.5,
            )

    def test_non_negative_accepts_zero(self):
        Tapering(side=0.0, bottom=0.0, top=0.0)

    def test_positive_rejects_negative_on_depth_field(self):
        with pytest.raises(ValueError, match="must be positive"):
            AseismicParameters(enabled=False, smooth=False, depth=-1.0)

    def test_proportion_rejects_too_large(self):
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            Tapering(side=1.5, bottom=0.0, top=0.0)

    def test_proportion_rejects_negative(self):
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            Tapering(side=0.5, bottom=-0.1, top=0.0)

    def test_none_skips_validation(self):
        p = _make_minimal_params(
            slip_time_function=Stype.ucsb,
            slip_water_level=None,
        )
        assert p.slip_water_level is None

    def test_top_level_validation(self):
        with pytest.raises(ValueError, match="must be positive"):
            _make_minimal_params(resolution=-1.0)

    def test_nested_none_deep_in_tree(self):
        """Optional None fields inside nested classes are accepted."""
        r = RiseTimePerturbation(
            level1_slip_correlation=0.5,
            level2_roughness_correlation=0.5,
            level2_slip_exponent=0.5,
            level1_sigma=None,
        )
        assert r.level1_sigma is None


class TestValidatorMetadata:
    def test_validator_in_metadata(self):
        f = dataclasses.fields(Tapering)[0]
        assert f.metadata.get("validator") is not None

    def test_validation_result_used(self):
        """Validate that the validated value is actually stored (no silent passthrough)."""
        p = _make_minimal_params()
        assert p.resolution == 0.5
