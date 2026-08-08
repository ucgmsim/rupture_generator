//! The shared helpers, tested — because a wrong one weakens every test that uses it.
//!
//! `common/` is not neutral scaffolding. `decompose` is the statistic two real
//! defects were diagnosed with, `CountingSource` is the whole draw-order audit, and
//! `field_draw_count` is a closed form asserted against a kernel. A silent error in
//! any of the three turns a suite that looks green into one that is not looking.
//!
//! So each helper is checked against a case whose answer is known by construction,
//! and the two that make claims about the *library* — the draw count, and that
//! counting is transparent — are checked against the library itself rather than
//! against a second reading of it (`ENGINEERING_RULES.md` rule 5).

mod common;

use common::counting::{CountingSource, field_draw_count};
use common::stats::{decompose, lag_one_along_dip, lag_one_along_strike, mean, pearson};
use common::tolerance::{Z, f32_sum_relative, wilson_hilferty_band};
use genslip::field::{CorrelationLengths, Spectrum2D, WavelengthBand, WavenumberStep};
use genslip::grid::Spectrum;
use genslip::rng::{DrawSource, GenslipLcg, Pcg};

/// A field with structure in both directions, longer-correlated along strike.
fn anisotropic(strike_count: usize, dip_count: usize) -> Vec<f64> {
    (0..strike_count * dip_count)
        .map(|index| {
            #[expect(clippy::cast_precision_loss, reason = "small test indices")]
            let (strike, dip) = ((index % strike_count) as f64, (index / strike_count) as f64);
            // A quarter-wavelength along strike for every full wavelength down dip:
            // smoother along strike by construction.
            (strike * 0.1).sin() + (dip * 0.8).sin()
        })
        .collect()
}

mod the_decomposition {
    use super::*;

    #[test]
    fn separates_a_level_change_from_a_spread_change() {
        let reference: Vec<f64> = (0..200).map(|i| f64::from(i) * 0.01).collect();

        // Scaled about zero: both level and spread move together.
        let scaled: Vec<f64> = reference.iter().map(|v| v * 2.0).collect();
        let scaled = decompose(&scaled, &reference);
        assert!((scaled.mean_ratio - 2.0).abs() < 1e-12, "{scaled}");
        assert!((scaled.sigma_ratio - 2.0).abs() < 1e-12, "{scaled}");
        assert!((scaled.correlation - 1.0).abs() < 1e-12, "{scaled}");

        // Stretched about its own mean: spread moves, level does not. This is the
        // case a mean-and-max check cannot see, and it is `DEFECTS.md` 18's shape.
        let centre = mean(&reference);
        let stretched: Vec<f64> = reference
            .iter()
            .map(|v| (v - centre) * 1.63 + centre)
            .collect();
        let stretched = decompose(&stretched, &reference);
        assert!((stretched.mean_ratio - 1.0).abs() < 1e-12, "{stretched}");
        assert!((stretched.sigma_ratio - 1.63).abs() < 1e-12, "{stretched}");
        assert!((stretched.correlation - 1.0).abs() < 1e-12, "{stretched}");
    }

    #[test]
    fn separates_both_from_a_change_of_shape() {
        // Reversed: same mean, same spread, different field. Only the correlation
        // moves -- and it is the only one of the three that can see a reordering,
        // which is what a shifted index does.
        let reference: Vec<f64> = (0..200).map(|i| f64::from(i) * 0.01).collect();
        let mut reversed = reference.clone();
        reversed.reverse();

        let result = decompose(&reversed, &reference);
        assert!((result.mean_ratio - 1.0).abs() < 1e-12, "{result}");
        assert!((result.sigma_ratio - 1.0).abs() < 1e-12, "{result}");
        assert!(result.correlation < -0.99, "{result}");
    }

    #[test]
    fn the_worst_difference_is_relative_to_scale_and_not_to_each_element() {
        // A field crossing zero has cells where any absolute drift is an unbounded
        // relative error. Scale-relative reports 1e-6; element-relative would report
        // 1.0 and read as a catastrophe.
        let reference = vec![-1.0, -1e-12, 0.0, 1e-12, 1.0];
        let produced = vec![-1.0, 1e-6, 1e-6, 1e-6, 1.0];

        let result = decompose(&produced, &reference);
        assert!(
            result.worst_scale_relative < 2e-6,
            "scale-relative should stay small: {result}"
        );
    }

    #[test]
    fn a_constant_field_has_no_shape_to_disagree_about() {
        // Guards the degenerate branch: a zero-variance field must not produce a NaN
        // correlation that then passes every `>` comparison in the suite.
        let flat = vec![3.0; 50];
        let result = decompose(&flat, &flat);
        assert!(result.correlation.is_finite(), "{result}");
        assert!((result.correlation - 1.0).abs() < 1e-12, "{result}");
        assert!((pearson(&flat, &flat) - 1.0).abs() < 1e-12);
    }
}

mod the_axis_statistic {
    use super::*;

    #[test]
    fn says_which_direction_the_field_is_smoother_in() {
        // The property the fast-axis contract rests on. Constructed so the answer is
        // known: a quarter-wavelength per subfault along strike, a full one down dip.
        let (strike_count, dip_count) = (40, 20);
        let field = anisotropic(strike_count, dip_count);

        let along = lag_one_along_strike(&field, strike_count, dip_count);
        let down = lag_one_along_dip(&field, strike_count, dip_count);
        assert!(
            along > down + 0.1,
            "strike {along:.4} should exceed dip {down:.4} on a field built to be \
             smoother along strike"
        );
    }
}

mod the_draw_counter {
    use super::*;

    /// Counting must not perturb what it counts.
    fn transparent_for<S: DrawSource>(mut bare: S, mut counted: CountingSource<S>) {
        for _ in 0..500 {
            let (a, b) = (bare.uniform(), counted.uniform());
            assert!((a - b).abs() < f64::EPSILON, "counting changed a uniform");
            let (a, b) = (bare.gaussian(0.75, 0.0), counted.gaussian(0.75, 0.0));
            assert!((a - b).abs() < f64::EPSILON, "counting changed a gaussian");
        }
        assert_eq!(counted.uniforms(), 500);
        assert_eq!(counted.gaussians(), 500);
    }

    #[test]
    fn is_transparent_for_both_generators() {
        transparent_for(
            GenslipLcg::new(77),
            CountingSource::new(GenslipLcg::new(77)),
        );
        transparent_for(Pcg::new(77), CountingSource::new(Pcg::new(77)));
    }

    #[test]
    fn checkpoints_report_each_stage_rather_than_a_running_total() {
        // The distinction that matters: a running total makes every stage after a
        // change look changed, so a diff cannot point at the one that moved.
        let mut source = CountingSource::new(GenslipLcg::new(1));
        source.skip_gaussians(10);
        source.checkpoint("first");
        source.skip_gaussians(3);
        source.checkpoint("second");

        assert_eq!(source.stage_gaussians(), vec![("first", 10), ("second", 3)]);
        assert_eq!(source.gaussians(), 13);
    }

    #[test]
    fn the_closed_form_matches_what_a_spectral_field_actually_draws() {
        // `field_draw_count` is a formula asserted about the library. Check it against
        // the library, on shapes chosen so the `dip/2 + 1` term is exercised at both
        // parities of the loop bound.
        for (strike_count, dip_count) in [(2, 2), (28, 16), (44, 12), (8, 30)] {
            let mut source = CountingSource::new(GenslipLcg::new(9));
            let mut spectrum = Spectrum::zeros(strike_count, dip_count);
            genslip::field::correlated_field(
                &mut spectrum,
                &mut source,
                Spectrum2D::Mai,
                WavenumberStep {
                    strike: 0.05,
                    dip: 0.07,
                },
                CorrelationLengths {
                    strike: 12.0,
                    dip: 6.0,
                },
                WavelengthBand::new(1.5, 80.0),
                3.5,
            );
            assert_eq!(
                source.gaussians(),
                field_draw_count(strike_count, dip_count),
                "{strike_count}x{dip_count}"
            );
        }
    }
}

mod the_derived_tolerances {
    use super::*;

    #[test]
    fn the_f32_sum_bound_grows_as_the_square_root_of_the_count() {
        // Not linearly. The worst case does, but it assumes every rounding goes the
        // same way; the bound here models them as independent, which is why it is
        // usable at 1e5 terms where `(n-1)*u` is not.
        let hundred = f32_sum_relative(100);
        let ten_thousand = f32_sum_relative(10_000);
        assert!(
            (ten_thousand / hundred - 10.0).abs() < 1e-9,
            "a hundredfold in count should be tenfold in error"
        );

        // And the documented detection floor at 1e5 is the reason to widen the
        // accumulator: six missing subfaults in a hundred thousand is invisible.
        assert!(f32_sum_relative(100_000) > 5.0e-5);
    }

    #[test]
    fn the_power_band_is_centred_near_one_and_tightens_with_more_freedom() {
        let (narrow_low, narrow_high) = wilson_hilferty_band(9_000, Z);
        let (wide_low, wide_high) = wilson_hilferty_band(280, Z);

        assert!(
            narrow_low < 1.0 && narrow_high > 1.0,
            "must contain the truth"
        );
        assert!(wide_low < 1.0 && wide_high > 1.0);
        assert!(
            narrow_high - narrow_low < wide_high - wide_low,
            "more degrees of freedom must give a tighter band"
        );
        // On the cube-root scale, 9000 degrees of freedom is a few percent at z = 5.
        assert!(narrow_high - narrow_low < 0.12);
    }
}
