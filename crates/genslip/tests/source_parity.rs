//! The setup arithmetic reproduces genslip's.
//!
//! These are scalar relations, not kernels: a magnitude becomes a moment, a moment
//! becomes an average rise time, a magnitude becomes a pair of wavenumber corners.
//! None of them is a function in the C — they are all inline in `main` — so as with
//! the pipelines the reference is transcribed from the source with line numbers, and
//! that transcription is exact because there is nothing here a library could do
//! differently.

// The reference functions below transcribe the C's expressions deliberately, so
// they carry its shapes: comparison chains rather than range tests.
//
// `cast_possible_truncation` used to be here too and is gone: the crate computes in
// `f64` now, so there is no narrowing left in either the port or the transcription.
#![expect(
    clippy::manual_range_contains,
    clippy::float_cmp,
    reason = "these mirror the original's own comparison chains, and assert exactness \
              where exactness IS the claim -- a circular average makes the two corner \
              lengths the same number, not merely close"
)]

mod common;

use common::tolerance::{agree, transcendental_spelling};
use genslip::source::{
    CornerRelation, Layer, MagnitudeScale, VelocityModel, average_rise_time, correlation_lengths,
    geometry_correction, seismic_moment,
};
use proptest::prelude::*;

const LN_10: f64 = std::f64::consts::LN_10;

/// `main:1250`.
fn reference_moment(magnitude: f64, coefficient: f64) -> f64 {
    (LN_10 * 1.5 * (magnitude + coefficient)).exp()
}

/// `main:1330` and its siblings.
fn reference_power_law(magnitude: f64, exponent: f64, offset: f64) -> f64 {
    (LN_10 * (exponent * magnitude - offset)).exp()
}

/// `main:1412`.
fn reference_rise_time(moment: f64, coefficient: f64) -> f64 {
    coefficient * (1.0e-09 * (moment.ln() / 3.0).exp())
}

/// `main:1418-1441`.
fn reference_alpha_t(dip: f64, rake: f64) -> (f64, f64, f64) {
    let fd = if dip <= 90.0 && dip > 45.0 {
        1.0 - (dip - 45.0) / 45.0
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
        let offset = wrapped - 90.0;
        1.0 - (offset * offset).sqrt() / 90.0
    } else {
        0.0
    };

    let alpha = 1.0 / (1.0 + (fd * fr * 0.1));
    (alpha, fd, fr)
}

#[test]
fn moments_match_on_both_magnitude_scales() {
    for magnitude in [4.0_f64, 5.5, 6.3, 7.1, 8.2] {
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

    for magnitude in [5.0_f64, 6.3, 7.5, 8.4] {
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
            reference_power_law(magnitude, 0.5, 2.50_f64).to_bits()
        );
        assert_eq!(
            mai.dip_km.to_bits(),
            reference_power_law(magnitude, 0.3333, 1.50_f64).to_bits()
        );

        // Two separate subtractions, as the original writes them: `(a - b) - c` is
        // not `a - (b + c)`.
        let reference_somerville =
            |offset: f64| (LN_10 * (0.5 * magnitude - offset - TWO_PI_DECADES)).exp();
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
            reference_power_law(clamped, 0.5, 1.69_f64).to_bits(),
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

    for magnitude in [4.0_f64, 5.0, 6.3, 7.5, 8.5] {
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
    for magnitude in [5.0_f64, 7.5] {
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
    // The one relation here where the port and the transcription use *different
    // formulas*: `cbrt` against the original's `exp(ln(M0)/3)`. Same function, two
    // evaluations, so they agree to a few ulps rather than to the bit.
    //
    // They agreed to the bit until this crate moved to `f64`, and that was an
    // artefact: the store to `float` rounded the difference away. `float_identities`
    // has asserted `cbrt != exp(ln x / 3)` at full width the whole time, and the two
    // statements only met when the narrowing went.
    for magnitude in [5.0_f64, 6.3, 7.5, 8.2] {
        let moment = seismic_moment(magnitude, MagnitudeScale::Moment);
        for coefficient in [1.6_f64, 2.3, 3.75] {
            let produced = average_rise_time(moment, coefficient);
            let expected = reference_rise_time(moment, coefficient);
            assert!(
                agree(
                    produced,
                    expected,
                    transcendental_spelling(moment, 1.0 / 3.0)
                ),
                "M{magnitude}, coefficient {coefficient}: {produced} vs {expected}"
            );
        }
    }
}

#[test]
fn the_geometry_correction_matches() {
    for dip in [90.0_f64, 75.0, 60.0, 45.0, 30.0, 10.0] {
        for rake in [
            -180.0_f64, -90.0, 0.0, 45.0, 90.0, 135.0, 180.0, 270.0, -450.0,
        ] {
            let produced = geometry_correction(dip, rake).expect("dip is in range");
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
    let vertical = geometry_correction(90.0, 0.0).expect("dip is in range");
    assert_eq!(vertical.alpha_t, 1.0);
    assert_eq!(vertical.dip_factor, 0.0);

    let reverse = geometry_correction(30.0, 90.0).expect("dip is in range");
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
    let expected = (3.4_f64 * 3.4 * 2.7) * 1.0e+10;
    assert_eq!(
        model.layer_at(20.0).rigidity().to_bits(),
        expected.to_bits()
    );
}

proptest! {
    #[test]
    fn moments_match_for_any_magnitude(magnitude in 3.0f64..9.5) {
        prop_assert_eq!(
            seismic_moment(magnitude, MagnitudeScale::Moment).to_bits(),
            reference_moment(magnitude, 10.73).to_bits()
        );
    }

    /// On a real fault plane the two agree bit for bit, at any rake including the
    /// ones that need wrapping.
    ///
    /// The dip range is 0..=90 now, where it used to run -10..100. That is not the
    /// test being weakened -- it is the range being *stated*. Outside it the two no
    /// longer agree, deliberately, and
    /// `a_dip_that_is_not_a_fault_plane_is_refused_where_the_original_returns_zero`
    /// asserts the disagreement rather than leaving it to a shrunk counterexample.
    #[test]
    fn the_geometry_correction_matches_for_any_fault_plane(
        dip in 0.0f64..=90.0,
        rake in -540.0f64..540.0,
    ) {
        let produced = geometry_correction(dip, rake).expect("0..=90 is a fault plane");
        let (alpha, fd, fr) = reference_alpha_t(dip, rake);
        prop_assert_eq!(produced.alpha_t.to_bits(), alpha.to_bits());
        prop_assert_eq!(produced.dip_factor.to_bits(), fd.to_bits());
        prop_assert_eq!(produced.rake_factor.to_bits(), fr.to_bits());
    }

    /// Outside 0..=90 the original returns a factor of **zero**, and the port refuses.
    ///
    /// This is the whole of the behaviour change, asserted as a positive claim on
    /// both sides: the reference really does answer, it really does answer zero, and
    /// zero really is indistinguishable from a vertical fault -- `alpha_t` comes back
    /// at exactly 1.0, the calibration point, for a dip of 120 degrees.
    ///
    /// A rupture generated that way is not flagged, not NaN and not obviously wrong.
    /// It is a fault whose rise time and rupture speed were never corrected, and the
    /// only signal was a factor that happens to equal the one a vertical strike-slip
    /// fault gets. `error.rs` says why that is the shape worth removing.
    #[test]
    fn a_dip_that_is_not_a_fault_plane_is_refused_where_the_original_returns_zero(
        dip in prop_oneof![-90.0f64..-0.001, 90.001f64..270.0],
        rake in -180.0f64..180.0,
    ) {
        prop_assert_eq!(
            geometry_correction(dip, rake),
            Err(genslip::Error::DipOutOfRange { degrees: dip })
        );

        let (alpha, fd, _) = reference_alpha_t(dip, rake);
        prop_assert_eq!(fd.to_bits(), 0.0f64.to_bits(), "the original refused after all");
        prop_assert_eq!(alpha.to_bits(), 1.0f64.to_bits(), "which reads as a vertical fault");
    }

    #[test]
    fn alpha_t_stays_within_its_bounds(dip in 0.0f64..90.0, rake in 0.0f64..180.0) {
        // 1/(1 + f_D*f_R*c) with both factors in [0, 1] and c = 0.1.
        let alpha = geometry_correction(dip, rake).expect("dip is in range").alpha_t;
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
