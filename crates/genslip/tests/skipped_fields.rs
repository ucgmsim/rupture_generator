//! Skipping the unused fields consumes exactly the randomness they would have.
//!
//! genslip generates two spectral fields whose values never reach the model — the
//! fault roughness, which is scaled by an `alpha_rough` of zero, and `tsfac2`, whose
//! only consumer is an unreachable `else` branch. Neither is built here.
//!
//! But both *draw*, and every field afterwards comes off the same stream. So the
//! claim that has to hold is not about values at all: it is that skipping a field
//! leaves the generator exactly where building it would have.
//!
//! That is checked against the real `kfilt_beta2` rather than against the formula,
//! because the formula is the thing that could be wrong.
#![cfg(feature = "fftw")]

use genslip::rng::GenslipLcg;
use genslip::slip::{skip_unused_field, unused_field_draw_count};
use genslip_oracle::{Complex, field as oracle};
use proptest::prelude::*;

/// The refined grid is squared up, so both extents are `max(strike, dip)` — but the
/// odd shapes are included anyway, since the draw count must not depend on that.
const SHAPES: [(usize, usize); 6] = [(2, 2), (4, 4), (16, 4), (4, 16), (32, 32), (64, 64)];

/// Where the C's generator leaves the seed after building a field of this extent.
fn oracle_seed_after(strike_count: usize, dip_count: usize, seed: i64) -> i64 {
    let mut grid = vec![Complex::default(); strike_count * dip_count];
    let mut oracle_seed = seed;
    oracle::self_affine_field(
        &mut grid,
        strike_count,
        dip_count,
        0.05,
        0.05,
        1.0,
        80.0,
        0.08,
        &mut oracle_seed,
    );
    oracle_seed
}

#[test]
fn skipping_a_field_lands_where_building_it_lands() {
    for (strike_count, dip_count) in SHAPES {
        let seed = 20_260_807;
        let expected = oracle_seed_after(strike_count, dip_count, seed);

        let mut source = GenslipLcg::new(seed);
        skip_unused_field(&mut source, strike_count, dip_count);

        assert_eq!(
            source.seed(),
            expected,
            "skipping a {strike_count}x{dip_count} field left the stream in the wrong place"
        );
    }
}

#[test]
fn the_draw_count_does_not_depend_on_the_spectrum() {
    // The generators draw at every grid point unconditionally, including where the
    // amplitude is then zeroed. If that ever stopped being true the skip would
    // silently desynchronise, so it is asserted rather than assumed: the same extent
    // with a very different band must cost the same.
    let (strike_count, dip_count) = (32, 32);
    let mut narrow = vec![Complex::default(); strike_count * dip_count];
    let mut wide = vec![Complex::default(); strike_count * dip_count];

    let mut narrow_seed: i64 = 5150;
    let mut wide_seed: i64 = 5150;
    oracle::self_affine_field(
        &mut narrow,
        strike_count,
        dip_count,
        0.05,
        0.05,
        1.0,
        10.0,
        5.0,
        &mut narrow_seed,
    );
    oracle::self_affine_field(
        &mut wide,
        strike_count,
        dip_count,
        0.05,
        0.05,
        1.0,
        5000.0,
        1e-4,
        &mut wide_seed,
    );

    assert_eq!(narrow_seed, wide_seed);
    assert_ne!(
        narrow, wide,
        "the two bands produced identical fields; the test proves nothing"
    );
}

#[test]
fn two_skipped_fields_cost_twice_one() {
    // genslip skips two of these in a row -- roughness, then tsfac2 -- on the same
    // grid. Nothing between them draws.
    let (strike_count, dip_count) = (24, 24);
    let seed = 909;

    let after_first = oracle_seed_after(strike_count, dip_count, seed);
    let after_second = oracle_seed_after(strike_count, dip_count, after_first);

    let mut source = GenslipLcg::new(seed);
    skip_unused_field(&mut source, strike_count, dip_count);
    skip_unused_field(&mut source, strike_count, dip_count);

    assert_eq!(source.seed(), after_second);
}

proptest! {
    #[test]
    fn the_count_matches_the_oracle_for_any_extent(
        half_strike in 1usize..24,
        half_dip in 1usize..24,
        seed in any::<i32>(),
    ) {
        let (strike_count, dip_count) = (2 * half_strike, 2 * half_dip);
        let seed = i64::from(seed);

        let expected = oracle_seed_after(strike_count, dip_count, seed);
        let mut source = GenslipLcg::new(seed);
        skip_unused_field(&mut source, strike_count, dip_count);

        prop_assert_eq!(source.seed(), expected);
        // And the count itself, stated so a failure says which of the two is wrong.
        prop_assert_eq!(
            unused_field_draw_count(strike_count, dip_count),
            2 * strike_count * (dip_count / 2 + 1)
        );
    }
}

// Deliberately not asserted:
//
// - Anything about the skipped fields' values. There are none; that is the point.
// - That `skip_gaussians` on the production generator matches anything. It has no
//   original to match, and the draw-count contract is a property of the
//   compatibility path alone.
