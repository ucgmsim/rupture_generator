//! The setup arithmetic reproduces genslip's.
//!
//! These are scalar relations, not kernels: a magnitude becomes a moment, a moment
//! becomes an average rise time, a magnitude becomes a pair of wavenumber corners.
//! None of them is a function in the C — they are all inline in `main` — so as with
//! the pipelines the reference is transcribed from the source with line numbers, and
//! that transcription is exact because there is nothing here a library could do
//! differently.

// The reference functions below transcribe the C's expressions deliberately, so
// they carry its shapes: comparison chains rather than range tests, and exact float
// equality where the claim IS exactness.
#![expect(
    clippy::manual_range_contains,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    reason = "these mirror the original's arithmetic and assert it exactly"
)]

use genslip::source::{
    CornerRelation, Layer, MagnitudeScale, VelocityModel, average_rise_time, correlation_lengths,
    geometry_correction, seismic_moment,
};
use proptest::prelude::*;

const LN_10: f64 = std::f64::consts::LN_10;

/// `main:1250`.
#[expect(clippy::cast_possible_truncation, reason = "mirrors the port's seams")]
fn reference_moment(magnitude: f32, coefficient: f64) -> f32 {
    (LN_10 * 1.5 * (f64::from(magnitude) + coefficient)).exp() as f32
}

/// `main:1330` and its siblings.
#[expect(clippy::cast_possible_truncation, reason = "mirrors the port's seams")]
fn reference_power_law(magnitude: f32, exponent: f64, offset: f64) -> f32 {
    (LN_10 * (exponent * f64::from(magnitude) - offset)).exp() as f32
}

/// `main:1412`.
#[expect(clippy::cast_possible_truncation, reason = "mirrors the port's seams")]
fn reference_rise_time(moment: f32, coefficient: f32) -> f32 {
    (f64::from(coefficient) * (1.0e-09 * (f64::from(moment).ln() / 3.0).exp())) as f32
}

/// `main:1418-1441`.
#[expect(clippy::cast_possible_truncation, reason = "mirrors the port's seams")]
fn reference_alpha_t(dip: f32, rake: f32) -> (f32, f32, f32) {
    let fd = if dip <= 90.0 && dip > 45.0 {
        (1.0 - (f64::from(dip) - 45.0) / 45.0) as f32
    } else if dip <= 45.0 && dip >= 0.0 {
        1.0
    } else {
        0.0
    };

    let mut wrapped = rake;
    while wrapped < -180.0 {
        wrapped += 360.0;
    }
    while wrapped > 180.0 {
        wrapped -= 360.0;
    }

    let fr = if wrapped <= 180.0 && wrapped >= 0.0 {
        let offset = f64::from(wrapped) - 90.0;
        (1.0 - (offset * offset).sqrt() / 90.0) as f32
    } else {
        0.0
    };

    let alpha = (1.0 / (1.0 + f64::from(fd * fr * 0.1))) as f32;
    (alpha, fd, fr)
}

#[test]
fn moments_match_on_both_magnitude_scales() {
    for magnitude in [4.0_f32, 5.5, 6.3, 7.1, 8.2] {
        assert_eq!(
            seismic_moment(magnitude, MagnitudeScale::Moment).to_bits(),
            reference_moment(magnitude, 10.73).to_bits(),
            "Mw {magnitude}"
        );
        assert_eq!(
            seismic_moment(magnitude, MagnitudeScale::Local).to_bits(),
            reference_moment(magnitude, 10.7).to_bits(),
            "M {magnitude}"
        );
    }
}

#[test]
fn correlation_lengths_match_for_every_relation() {
    const TWO_PI_DECADES: f64 = 0.79818;

    for magnitude in [5.0_f32, 6.3, 7.5, 8.4] {
        let mai = correlation_lengths(
            magnitude,
            CornerRelation::Mai {
                strike_offset: 2.50,
                dip_offset: 1.50,
                circular: false,
            },
            false,
        );
        assert_eq!(
            mai.strike_km.to_bits(),
            reference_power_law(magnitude, 0.5, f64::from(2.50_f32)).to_bits()
        );
        assert_eq!(
            mai.dip_km.to_bits(),
            reference_power_law(magnitude, 0.3333, f64::from(1.50_f32)).to_bits()
        );

        // Two separate subtractions, as the original writes them: `(a - b) - c` is
        // not `a - (b + c)`.
        let reference_somerville = |offset: f64| {
            (LN_10 * (0.5 * f64::from(magnitude) - offset - TWO_PI_DECADES)).exp() as f32
        };
        let somerville = correlation_lengths(
            magnitude,
            CornerRelation::Somerville { circular: false },
            false,
        );
        assert_eq!(
            somerville.strike_km.to_bits(),
            reference_somerville(1.72).to_bits()
        );
        assert_eq!(
            somerville.dip_km.to_bits(),
            reference_somerville(1.93).to_bits()
        );

        // Suzuki clamps the down-dip corner above the saturation magnitude.
        let suzuki = correlation_lengths(
            magnitude,
            CornerRelation::Suzuki {
                strike_offset: 1.67,
                dip_offset: 1.69,
                saturation_magnitude: 6.3,
            },
            false,
        );
        let clamped = magnitude.min(6.3);
        assert_eq!(
            suzuki.dip_km.to_bits(),
            reference_power_law(clamped, 0.5, f64::from(1.69_f32)).to_bits(),
            "Suzuki dip corner at M{magnitude}"
        );

        // `modified_corners` overrides whatever relation was chosen.
        let overridden = correlation_lengths(
            magnitude,
            CornerRelation::Somerville { circular: false },
            true,
        );
        let expected = reference_power_law(magnitude, 0.5, 2.00);
        assert_eq!(overridden.strike_km.to_bits(), expected.to_bits());
        assert_eq!(overridden.dip_km.to_bits(), expected.to_bits());
    }
}

#[test]
fn frankel_takes_mai_s_corners_because_one_branch_serves_both() {
    // `if(kmodel == MAI_FLAG || kmodel == FRANKEL_FLAG)` (main:1303). Frankel has a
    // spectral falloff of its own (`slip.c:1651`) but no corner relation of its own,
    // so the two uses of `kmodel` do not partition the same way. The PyO3 boundary
    // read that as Somerville for a while (`DEFECTS.md` 11).
    //
    // What the misrouting actually cost, measured rather than assumed:
    //
    // - **Along strike, a constant 4.3%.** Both scale as 10^(0.5*M), so only the
    //   offsets differ -- Mai's 2.50 against Somerville's 1.72 + 0.79818 = 2.51818.
    //   The ratio is 10^0.01818 at every magnitude.
    // - **Down dip, anything from 3.6x to 0.65x**, because the exponents differ:
    //   10^(0.3333*M) against 10^(0.5*M). They **cross at M7.37**, where the wrong
    //   relation is briefly indistinguishable from the right one.
    //
    // That crossover is why this is worth a test rather than an eyeball. A fixture
    // near M7.4 would have shown the defect as a rounding difference.
    let relations = |magnitude| {
        (
            correlation_lengths(
                magnitude,
                CornerRelation::Mai {
                    strike_offset: 2.50,
                    dip_offset: 1.50,
                    circular: false,
                },
                false,
            ),
            correlation_lengths(
                magnitude,
                CornerRelation::Somerville { circular: false },
                false,
            ),
        )
    };

    for magnitude in [4.0_f32, 5.0, 6.3, 7.5, 8.5] {
        let (mai, somerville) = relations(magnitude);
        let strike_ratio = mai.strike_km / somerville.strike_km;
        assert!(
            (strike_ratio - 1.042_75).abs() < 1e-4,
            "M{magnitude}: strike ratio {strike_ratio}, expected the constant 10^0.01818"
        );
    }

    // Mai's down-dip corner is the larger below the crossover and the smaller above.
    let (below, below_somerville) = relations(6.0);
    assert!(below.dip_km > below_somerville.dip_km);
    let (above, above_somerville) = relations(8.0);
    assert!(above.dip_km < above_somerville.dip_km);

    // And at the crossover they agree to better than a percent, which is the case a
    // fixture would have failed to notice.
    let (at, at_somerville) = relations(7.3676);
    assert!(
        ((at.dip_km / at_somerville.dip_km) - 1.0).abs() < 0.01,
        "M7.3676: Mai {} vs Somerville {}",
        at.dip_km,
        at_somerville.dip_km
    );
}

#[test]
fn circular_relations_give_equal_corners() {
    for magnitude in [5.0_f32, 7.5] {
        let mai = correlation_lengths(
            magnitude,
            CornerRelation::Mai {
                strike_offset: 2.50,
                dip_offset: 1.50,
                circular: true,
            },
            false,
        );
        assert_eq!(mai.strike_km, mai.dip_km);

        let somerville = correlation_lengths(
            magnitude,
            CornerRelation::Somerville { circular: true },
            false,
        );
        assert_eq!(somerville.strike_km, somerville.dip_km);
    }
}

#[test]
fn rise_times_match() {
    for magnitude in [5.0_f32, 6.3, 7.5, 8.2] {
        let moment = seismic_moment(magnitude, MagnitudeScale::Moment);
        for coefficient in [1.6_f32, 2.3, 3.75] {
            assert_eq!(
                average_rise_time(moment, coefficient).to_bits(),
                reference_rise_time(moment, coefficient).to_bits(),
                "M{magnitude}, coefficient {coefficient}"
            );
        }
    }
}

#[test]
fn the_geometry_correction_matches() {
    for dip in [90.0_f32, 75.0, 60.0, 45.0, 30.0, 10.0] {
        for rake in [
            -180.0_f32, -90.0, 0.0, 45.0, 90.0, 135.0, 180.0, 270.0, -450.0,
        ] {
            let produced = geometry_correction(dip, rake);
            let (alpha, fd, fr) = reference_alpha_t(dip, rake);
            assert_eq!(
                produced.alpha_t.to_bits(),
                alpha.to_bits(),
                "dip {dip}, rake {rake}"
            );
            assert_eq!(produced.dip_factor.to_bits(), fd.to_bits());
            assert_eq!(produced.rake_factor.to_bits(), fr.to_bits());
        }
    }
}

#[test]
fn a_vertical_strike_slip_fault_is_the_calibration_point() {
    // alpha_t is 1 there, so rise time and rupture speed are unchanged -- which is
    // what "calibrated on strike-slip" means. Every other geometry shortens the
    // pulse and speeds the rupture.
    let vertical = geometry_correction(90.0, 0.0);
    assert_eq!(vertical.alpha_t, 1.0);
    assert_eq!(vertical.dip_factor, 0.0);

    let reverse = geometry_correction(30.0, 90.0);
    assert!(reverse.alpha_t < 1.0, "alpha_t is {}", reverse.alpha_t);
    assert!(reverse.alpha_t > 0.9, "alpha_t is {}", reverse.alpha_t);
}

#[test]
fn the_velocity_model_clamps_below_its_deepest_layer() {
    let model = VelocityModel::new(vec![
        Layer {
            bottom_depth_km: 1.0,
            shear_speed_km_s: 1.0,
            density_g_cm3: 2.0,
        },
        Layer {
            bottom_depth_km: 5.0,
            shear_speed_km_s: 2.5,
            density_g_cm3: 2.4,
        },
        Layer {
            bottom_depth_km: 20.0,
            shear_speed_km_s: 3.4,
            density_g_cm3: 2.7,
        },
    ]);

    // On a boundary, the layer above owns it -- the search advances only while the
    // depth is strictly greater.
    assert_eq!(model.layer_at(1.0).shear_speed_km_s, 1.0);
    assert_eq!(model.layer_at(1.001).shear_speed_km_s, 2.5);
    // Below everything, the deepest layer rather than an extrapolation.
    assert_eq!(model.layer_at(1000.0).shear_speed_km_s, 3.4);

    // Rigidity is rho*vs^2 in CMS units.
    let expected = (f64::from(3.4_f32 * 3.4 * 2.7) * 1.0e+10) as f32;
    assert_eq!(
        model.layer_at(20.0).rigidity().to_bits(),
        expected.to_bits()
    );
}

proptest! {
    #[test]
    fn moments_match_for_any_magnitude(magnitude in 3.0f32..9.5) {
        prop_assert_eq!(
            seismic_moment(magnitude, MagnitudeScale::Moment).to_bits(),
            reference_moment(magnitude, 10.73).to_bits()
        );
    }

    #[test]
    fn the_geometry_correction_matches_for_any_geometry(
        dip in -10.0f32..100.0,
        rake in -540.0f32..540.0,
    ) {
        let produced = geometry_correction(dip, rake);
        let (alpha, fd, fr) = reference_alpha_t(dip, rake);
        prop_assert_eq!(produced.alpha_t.to_bits(), alpha.to_bits());
        prop_assert_eq!(produced.dip_factor.to_bits(), fd.to_bits());
        prop_assert_eq!(produced.rake_factor.to_bits(), fr.to_bits());
    }

    #[test]
    fn alpha_t_stays_within_its_bounds(dip in 0.0f32..90.0, rake in 0.0f32..180.0) {
        // 1/(1 + f_D*f_R*c) with both factors in [0, 1] and c = 0.1.
        let alpha = geometry_correction(dip, rake).alpha_t;
        prop_assert!(alpha <= 1.0 && alpha >= 1.0 / 1.1 - 1e-6, "alpha_t = {}", alpha);
    }
}

// Deliberately not asserted:
//
// - That the two magnitude scales agree. They differ by 0.03 in the constant, which
//   is a 10% difference in moment, and that is the point of having both.
// - Anything about which corner relation is right. They come from different
//   inversions of different catalogues; the port reproduces all four and chooses
//   none.
