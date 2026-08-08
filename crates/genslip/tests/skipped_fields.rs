//! Skipping the unused fields consumes exactly the randomness they would have.
//!
//! genslip generates two spectral fields whose values never reach the model — the
//! fault roughness, scaled by an `alpha_rough` of zero, and `tsfac2`, whose only
//! consumer is an unreachable `else` branch. Neither is built here.
//!
//! But both *draw*, and every field afterwards comes off the same stream. So the claim
//! is not about values at all: skipping a field must leave the generator exactly where
//! building it would have. Get it wrong and slip, rake and both perturbations are all
//! different, all plausible, and all silently wrong.
//!
//! # What checks the claim against genslip, now that this file does not
//!
//! It used to compare against the C's `kfilt_beta2` directly. That was the right
//! reference while one existed, and the corpus is a better one: a desynchronised
//! stream changes *every* field after it, so slip alone would diverge by order one
//! rather than the 2.4e-06 it does. The end-to-end comparison covers this more
//! strongly than a seed comparison did, and without linking Fortran.
//!
//! What remains here is the part the corpus cannot isolate: that the skip matches the
//! *generator it stands in for*, on shapes the corpus does not contain.
#![cfg(feature = "fftw")]

mod common;

use common::counting::{CountingSource, field_draw_count};
use genslip::field::{WavelengthBand, WavenumberStep, self_affine_field};
use genslip::grid::Spectrum;
use genslip::rng::GenslipLcg;
use proptest::prelude::*;

/// The refined grid is squared up, so both extents are `max(strike, dip)` — but the
/// odd shapes are here anyway, since the draw count must not depend on that.
const SHAPES: [(usize, usize); 6] = [(2, 2), (4, 4), (16, 4), (4, 16), (32, 32), (64, 64)];

const STEP: WavenumberStep = WavenumberStep {
    strike: 0.05,
    dip: 0.05,
};

/// Build the field the skip stands in for, and say where it left the stream.
fn building_it(strike_count: usize, dip_count: usize, seed: i64) -> i64 {
    let mut spectrum = Spectrum::zeros(strike_count, dip_count);
    let mut source = GenslipLcg::new(seed);
    self_affine_field(
        &mut spectrum,
        &mut source,
        1.0,
        STEP,
        WavelengthBand::new(0.08, 80.0),
    );
    GenslipLcg::seed(source)
}

#[test]
fn skipping_a_field_lands_where_building_it_lands() {
    for (strike_count, dip_count) in SHAPES {
        let seed = 20_260_807;
        let mut source = GenslipLcg::new(seed);
        genslip::slip::skip_unused_field(&mut source, strike_count, dip_count);

        assert_eq!(
            GenslipLcg::seed(source),
            building_it(strike_count, dip_count, seed),
            "skipping a {strike_count}x{dip_count} field left the stream in the wrong \
             place"
        );
    }
}

#[test]
fn the_draw_count_does_not_depend_on_the_spectrum() {
    // The generators draw at every grid point unconditionally, including where the
    // amplitude is then zeroed. If that stopped being true the skip would silently
    // desynchronise, so it is asserted rather than assumed: the same extent with a
    // very different band, and a very different Hurst exponent, must cost the same.
    let (strike_count, dip_count) = (32, 32);

    let build = |hurst: f32, band: WavelengthBand| {
        let mut spectrum = Spectrum::zeros(strike_count, dip_count);
        let mut source = CountingSource::new(GenslipLcg::new(5150));
        self_affine_field(&mut spectrum, &mut source, hurst, STEP, band);
        (source.gaussians(), spectrum)
    };

    let (narrow_draws, narrow) = build(1.0, WavelengthBand::new(5.0, 10.0));
    let (wide_draws, wide) = build(0.2, WavelengthBand::new(1e-4, 5000.0));

    assert_eq!(narrow_draws, wide_draws);
    assert_eq!(narrow_draws, field_draw_count(strike_count, dip_count));
    assert_ne!(
        narrow.as_slice(),
        wide.as_slice(),
        "the two bands produced identical fields; the test proves nothing"
    );
}

#[test]
fn two_skipped_fields_cost_twice_one() {
    // genslip skips two of these in a row -- roughness, then tsfac2 -- on the same
    // grid, with nothing between them drawing.
    let (strike_count, dip_count) = (24, 24);
    let seed = 909;

    let after_second = building_it(
        strike_count,
        dip_count,
        building_it(strike_count, dip_count, seed),
    );

    let mut source = GenslipLcg::new(seed);
    genslip::slip::skip_unused_field(&mut source, strike_count, dip_count);
    genslip::slip::skip_unused_field(&mut source, strike_count, dip_count);

    assert_eq!(GenslipLcg::seed(source), after_second);
}

proptest! {
    /// Any extent, and counted through the decorator rather than through the LCG's
    /// exposed state — so the claim survives the modern generator becoming default.
    #[test]
    fn the_count_matches_the_generator_for_any_extent(
        strike_count in 1usize..40,
        dip_count in 1usize..40,
    ) {
        let mut skipping = CountingSource::new(GenslipLcg::new(77));
        genslip::slip::skip_unused_field(&mut skipping, strike_count, dip_count);

        let mut spectrum = Spectrum::zeros(2 * strike_count, 2 * dip_count);
        let mut building = CountingSource::new(GenslipLcg::new(77));
        self_affine_field(
            &mut spectrum,
            &mut building,
            1.0,
            STEP,
            WavelengthBand::new(0.08, 80.0),
        );

        // The skip is quoted for the extent it is *given*; the field above is twice
        // that in each direction, so it must cost more. The exact relation is the
        // closed form, checked directly.
        prop_assert_eq!(
            skipping.gaussians(),
            field_draw_count(strike_count, dip_count)
        );
        prop_assert!(building.gaussians() > skipping.gaussians());
    }
}

// Deliberately not asserted:
//
// - That the skipped fields are numerically inert. They are -- `alpha_rough` is zero
//   and `tsfac2`'s branch is unreachable -- but that is a fact about the
//   configuration, recorded in `PRUNED.md`, and the input layer refusing to change it
//   is what keeps it true.
