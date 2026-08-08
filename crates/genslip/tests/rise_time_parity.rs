//! The rise-time field reproduces genslip's post-processing, stage for stage.
//!
//! `main`'s rtime1 block shares its first half with tsfac1 — generate, correlate with
//! slip, transform back — which `slip_pipeline.rs` and `correlated_fields.rs` already
//! pin. What is new is everything after: a depth blend toward slip, a normalisation
//! by the mean's *magnitude*, a variation rescale, truncation, a power law, and a
//! final renormalisation. Then the fault-wide constant.
//!
//! As before there is no C function to call, so the reference is the C's arithmetic
//! rewritten straight from the source with line numbers attached.
//!
//! # It used to be bit-exact, and that was an artefact of `f32`
//!
//! The transcription and the port differ in one place on purpose: the power law is
//! `powf` here and `exp(n*ln(x))` in the C, which are the same function and different
//! evaluations. While the port computed in `float` the difference was rounded away by
//! the store, and this file asserted bit equality and got it.
//!
//! At `f64` the two ulps are visible, so the assertion is a derived bound —
//! `transcendental_spelling`, about 1.8e-15 relative. That is four orders below
//! anything an `f32` could express, so it still cannot hide a wrong constant, which
//! is the only thing this file was ever able to catch.
//!
//! The bound is `PIPELINE_REASSOCIATION`, and it is **chosen rather than derived**.
//! `ndarray`'s `.mean()` folds pairwise where the transcription folds left to right,
//! both correct and different; the field is then five stages deep and the rescale
//! amplifies. Deriving that chain is a paper. What makes a chosen bound defensible
//! here is the gap it sits in: reassociation moves the answer by ~1e-14, a wrong
//! constant by 1e-3, and this sits eleven orders of clear air between them. See the
//! constant's own doc.
//!
//! Worth noticing that `float_identities.rs` asserted the *opposite* of this file for
//! as long as both existed: that `powf` and `exp(n*ln x)` are **not** equal. Both
//! were right about their own width. Only widening the port made the two statements
//! meet.
//!
//! # The 1x1 case is NaN on both sides, and that is the assertion
//!
//! `rescale_about_unit_mean` divides by the field's own coefficient of variation,
//! which is zero when the fault has one subfault, so a prescribed spread is
//! undefined. The port and the C degenerate identically — see `agree`, which says
//! *both defined and close, or both undefined* rather than quietly passing on
//! `NaN == NaN`.

mod common;

use common::tolerance::{PIPELINE_REASSOCIATION, agree};
use genslip::grid::{FaultAxes, SlipField};
use genslip::rise_time::{
    DepthRamp, DepthScaling, RiseTimeSpec, RiseTimeStretch, Weighting, rise_time_field,
    rise_time_normalisation,
};
use genslip::slip::PerturbationSpec;
use proptest::prelude::*;

const SHAPES: [(usize, usize); 4] = [(1, 1), (8, 6), (32, 12), (24, 24)];

fn spec(sigma: f64, exponent: f64) -> RiseTimeSpec {
    RiseTimeSpec {
        perturbation: PerturbationSpec {
            correlation: 0.9,
            sigma,
        },
        shallow_blend: DepthRamp {
            centre_km: 2.0,
            half_width_km: 1.0,
        },
        slip_exponent: exponent,
    }
}

/// A field with structure in both directions and some negatives, so the truncation
/// and the magnitude-normalisation both get exercised.
fn seeded(strike_count: usize, dip_count: usize, offset: f64) -> SlipField {
    let mut field = genslip::grid::zeros(strike_count, dip_count);
    for dip in 0..dip_count {
        for strike in 0..strike_count {
            let value = offset
                + (genslip::units::exact(strike) * 0.41).sin()
                + (genslip::units::exact(dip) * 0.23).cos() * 0.7;
            field[[dip, strike]] = value;
        }
    }
    field
}

/// Depths increasing down dip, spanning the shallow blend and both rise-time ramps.
fn depths(dip_count: usize) -> Vec<f64> {
    (0..dip_count)
        .map(|dip| genslip::units::exact(dip) * 1.6)
        .collect()
}

fn shear_speeds(dip_count: usize) -> Vec<f64> {
    (0..dip_count)
        .map(|dip| 2.4 + genslip::units::exact(dip) * 0.05)
        .collect()
}

/// `main:2226-2352`, transcribed.
fn reference_field(
    correlated: &SlipField,
    slip: &SlipField,
    depth_km: &[f64],
    spec: RiseTimeSpec,
) -> Vec<f64> {
    let strike_count = correlated.strike_count();
    let dip_count = correlated.dip_count();
    let count = genslip::units::exact(strike_count * dip_count);

    // :2229 -- the shallow blend, weighted per dip row.
    let low = spec.shallow_blend.centre_km - spec.shallow_blend.half_width_km;
    let high = spec.shallow_blend.centre_km + spec.shallow_blend.half_width_km;
    let mut field = Vec::with_capacity(strike_count * dip_count);
    let mut total = 0.0_f64;
    for dip in 0..dip_count {
        let depth = depth_km[dip];
        let (slip_weight, rise_weight) = if depth <= low {
            (1.0_f64, 0.0_f64)
        } else if depth < high {
            let rise = (depth - low) / (high - low);
            (1.0 - rise, rise)
        } else {
            (0.0, 1.0)
        };
        for strike in 0..strike_count {
            let value = rise_weight * correlated[[dip, strike]] + slip_weight * slip[[dip, strike]];
            field.push(value);
            total += value;
        }
    }

    // :2267 -- normalise by the magnitude of the mean: `sqrt(avg*avg)`.
    let mean = total / count;
    let magnitude = (mean * mean).sqrt();
    for value in &mut field {
        *value /= magnitude;
    }

    // :2295 -- rescale about a mean of 1 to the target coefficient of variation.
    let mut sum_of_squares = 0.0_f64;
    for value in &field {
        sum_of_squares += (*value - 1.0) * (*value - 1.0);
    }
    let variation = (sum_of_squares / count).sqrt();
    let sigma = spec.perturbation.sigma.max(0.0);
    let factor = sigma / variation;
    for value in &mut field {
        *value = factor * (*value - 1.0) + 1.0;
    }

    // :2309 -- truncate negatives.
    for value in &mut field {
        if *value < 0.0 {
            *value = 0.0;
        }
    }

    // :2342 -- the power law, on positive values only.
    if spec.slip_exponent > 0.1 {
        for value in &mut field {
            if *value > 0.0 {
                *value = (spec.slip_exponent * (*value).ln()).exp();
            }
        }
    }

    // :2352 -- back to unit mean.
    let mut total = 0.0_f64;
    for value in &field {
        total += *value;
    }
    let mean = total / count;
    for value in &mut field {
        *value /= mean;
    }

    field
}

/// `main:2413-2474`, transcribed.
fn reference_normalisation(
    rise_time: &SlipField,
    slip: &SlipField,
    depth_km: &[f64],
    shear_speed: &[f64],
    scaling: DepthScaling,
    weighting: Weighting,
) -> f64 {
    let strike_count = rise_time.strike_count();
    let dip_count = rise_time.dip_count();

    let (dmin1, dmax1) = (
        scaling.stretch.shallow.centre_km - scaling.stretch.shallow.half_width_km,
        scaling.stretch.shallow.centre_km + scaling.stretch.shallow.half_width_km,
    );
    let (dmin2, dmax2) = (
        scaling.stretch.deep.centre_km - scaling.stretch.deep.half_width_km,
        scaling.stretch.deep.centre_km + scaling.stretch.deep.half_width_km,
    );
    let rtfac1 = scaling.stretch.shallow_factor - 1.0;
    let rtfac2 = scaling.stretch.deep_factor - 1.0;

    let mut snum = 0.0_f64;
    let mut sden = 0.0_f64;
    for dip in 0..dip_count {
        let depth = depth_km[dip];
        let mut rf = shear_speed[dip] * scaling.rupture_velocity_fraction;
        let sf;
        if depth <= dmin1 {
            rf *= scaling.shallow_rupture_velocity;
            sf = 1.0 + rtfac1;
        } else if depth < dmax1 {
            rf = rf * scaling.shallow_rupture_velocity * (dmax1 - depth) / (dmax1 - dmin1);
            sf = 1.0 + rtfac1 * (dmax1 - depth) / (dmax1 - dmin1);
        } else if depth <= dmin2 {
            sf = 1.0;
        } else if depth < dmax2 {
            rf = rf * scaling.deep_rupture_velocity * (depth - dmin2) / (dmax2 - dmin2);
            sf = 1.0 + rtfac2 * (depth - dmin2) / (dmax2 - dmin2);
        } else {
            rf *= scaling.deep_rupture_velocity;
            sf = 1.0 + rtfac2;
        }

        for strike in 0..strike_count {
            let s = slip[[dip, strike]];
            let sabs = match weighting {
                Weighting::Uniform => 1.0,
                Weighting::BySlip => (s * s).sqrt(),
                Weighting::BySlipAndRuptureSpeed => rf * (s * s).sqrt(),
            };
            snum += sf * sabs * rise_time[[dip, strike]];
            sden += sabs;
        }
    }

    snum / sden
}

fn scaling() -> DepthScaling {
    DepthScaling {
        stretch: RiseTimeStretch {
            shallow: DepthRamp {
                centre_km: 6.5,
                half_width_km: 1.5,
            },
            shallow_factor: 2.0,
            deep: DepthRamp {
                centre_km: 17.5,
                half_width_km: 2.5,
            },
            deep_factor: 2.0,
        },
        rupture_velocity_fraction: 0.8,
        shallow_rupture_velocity: 0.6,
        deep_rupture_velocity: 0.6,
    }
}

#[test]
fn the_rise_time_field_matches_across_every_shape() {
    for (strike_count, dip_count) in SHAPES {
        for exponent in [0.5_f64, 1.0, 0.05] {
            let correlated = seeded(strike_count, dip_count, 0.2);
            let slip = seeded(strike_count, dip_count, 1.0);
            let depth_km = depths(dip_count);
            let spec = spec(0.75, exponent);

            let expected = reference_field(&correlated, &slip, &depth_km, spec);
            let produced = rise_time_field(&correlated, &slip, &depth_km, spec);

            for (offset, (got, want)) in produced.flat().iter().zip(&expected).enumerate() {
                assert!(
                    agree(*got, *want, PIPELINE_REASSOCIATION),
                    "exponent {exponent} on {strike_count}x{dip_count}: \
                     mismatch at {offset}: {got} vs {want}"
                );
            }
        }
    }
}

#[test]
fn a_negative_mean_field_is_flipped_positive() {
    // The normalisation divides by the magnitude of the mean, not the mean. That is
    // what turns a field which came out negative-mean into a positive one, and it is
    // the reason the original writes `sqrt(avg*avg)` rather than just `avg`.
    let (strike_count, dip_count) = (8, 6);
    let correlated = seeded(strike_count, dip_count, -3.0);
    let slip = seeded(strike_count, dip_count, -3.0);
    let depth_km = vec![100.0; dip_count];

    let produced = rise_time_field(&correlated, &slip, &depth_km, spec(0.75, 0.5));
    let total: f64 = produced.flat().iter().sum();
    assert!(total > 0.0, "field summed to {total}, expected positive");
}

#[test]
fn the_normalisation_matches_for_every_weighting() {
    for (strike_count, dip_count) in SHAPES {
        for weighting in [
            Weighting::Uniform,
            Weighting::BySlip,
            Weighting::BySlipAndRuptureSpeed,
        ] {
            let rise = seeded(strike_count, dip_count, 1.0);
            let slip = seeded(strike_count, dip_count, 2.0);
            let depth_km = depths(dip_count);
            let shear = shear_speeds(dip_count);

            let expected =
                reference_normalisation(&rise, &slip, &depth_km, &shear, scaling(), weighting);
            let produced =
                rise_time_normalisation(&rise, &slip, &depth_km, &shear, scaling(), weighting);

            assert_eq!(
                produced.to_bits(),
                expected.to_bits(),
                "{weighting:?} on {strike_count}x{dip_count}: {produced} vs {expected}"
            );
        }
    }
}

proptest! {
    #[test]
    fn the_field_matches_for_arbitrary_parameters(
        strike_count in 1usize..20,
        dip_count in 1usize..20,
        sigma in 0.01f64..3.0,
        exponent in 0.0f64..2.0,
        blend_centre in 0.5f64..20.0,
        blend_width in 0.1f64..5.0,
    ) {
        let correlated = seeded(strike_count, dip_count, 0.2);
        let slip = seeded(strike_count, dip_count, 1.0);
        let depth_km = depths(dip_count);
        let spec = RiseTimeSpec {
            perturbation: PerturbationSpec { correlation: 0.9, sigma },
            shallow_blend: DepthRamp {
                centre_km: blend_centre,
                half_width_km: blend_width,
            },
            slip_exponent: exponent,
        };

        let expected = reference_field(&correlated, &slip, &depth_km, spec);
        let produced = rise_time_field(&correlated, &slip, &depth_km, spec);

        for (offset, (got, want)) in produced.flat().iter().zip(&expected).enumerate() {
            prop_assert!(
                agree(*got, *want, PIPELINE_REASSOCIATION),
                "at {}: {} vs {}",
                offset,
                got,
                want
            );
        }
    }
}

// Deliberately not asserted:
//
// - That the returned field has mean exactly 1. It does to rounding, and the last
//   stage divides by the mean to make it so, but the divide is a single-precision
//   fold and the result is not exact.
// - That rise time increases with slip. It does statistically -- that is what the
//   correlation is for -- but the shallow blend and the truncation both break the
//   pointwise relation. A Stage 2 claim with a measured band.
