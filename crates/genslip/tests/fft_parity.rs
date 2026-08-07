//! The FFTW engine reproduces genslip's transforms bit for bit.
//!
//! Achievable only because genslip plans with `FFTW_ESTIMATE`, which chooses from a
//! cost model rather than by timing candidates. Had it used `FFTW_MEASURE` the
//! original would not be reproducible against *itself*, and this file could not
//! exist.
//!
//! `fft_contract.rs` covers what is true of both engines, and measures how far apart
//! they are. This file covers only the one that has to match.
#![cfg(feature = "fftw")]

use genslip::fft::{Direction, FftwFft, scale, spacing_product, transform_2d};
use genslip::grid::Spectrum;
use genslip_oracle::{Complex, field as oracle};
use num_complex::Complex32;
use proptest::prelude::*;

/// Extents that reach different FFTW code paths. Radix-2 lengths take the fast
/// codelets; 24 and 12 are mixed-radix; 6 and 10 have a prime factor above 5, which
/// sends FFTW down a different algorithm entirely.
const SHAPES: [(usize, usize); 7] = [
    (2, 2),
    (4, 4),
    (8, 16),
    (12, 6),
    (24, 10),
    (32, 24),
    (64, 8),
];

fn seeded(strike_count: usize, dip_count: usize) -> Spectrum {
    let mut spectrum = Spectrum::zeros(strike_count, dip_count);
    for dip in 0..dip_count {
        for strike in 0..strike_count {
            #[expect(clippy::cast_precision_loss, reason = "small test indices")]
            let value = Complex32::new(
                (strike as f32 * 0.37).sin() + (dip as f32 * 0.11).cos(),
                (strike as f32 * 0.19).cos() - (dip as f32 * 0.43).sin(),
            );
            spectrum[(strike, dip)] = value;
        }
    }
    spectrum
}

fn as_oracle_grid(spectrum: &Spectrum) -> Vec<Complex> {
    spectrum
        .as_slice()
        .iter()
        .map(|value| Complex {
            re: value.re,
            im: value.im,
        })
        .collect()
}

/// Run both sides and compare bit for bit.
fn check(strike_count: usize, dip_count: usize, direction: Direction, d1: f32, d2: f32) {
    let mut ported = seeded(strike_count, dip_count);
    let mut expected = as_oracle_grid(&ported);

    let sign = match direction {
        Direction::Forward => -1,
        Direction::Inverse => 1,
    };
    oracle::transform_2d(&mut expected, strike_count, dip_count, sign, d1, d2);

    let mut fft = FftwFft::new();
    transform_2d(&mut ported, &mut fft, direction);
    scale(&mut ported, spacing_product(d1, d2));

    for (offset, (got, want)) in ported.as_slice().iter().zip(&expected).enumerate() {
        assert_eq!(
            (got.re.to_bits(), got.im.to_bits()),
            (want.re.to_bits(), want.im.to_bits()),
            "{direction:?} on {strike_count}x{dip_count}: mismatch at \
             (strike {}, dip {}): {got:?} vs ({}, {})",
            offset % strike_count,
            offset / strike_count,
            want.re,
            want.im,
        );
    }
}

#[test]
fn forward_transforms_match_across_every_shape() {
    for (strike_count, dip_count) in SHAPES {
        check(strike_count, dip_count, Direction::Forward, 2.0, 1.5);
    }
}

#[test]
fn inverse_transforms_match_across_every_shape() {
    for (strike_count, dip_count) in SHAPES {
        check(strike_count, dip_count, Direction::Inverse, 0.031, 0.052);
    }
}

#[test]
fn separating_the_scaling_from_the_transform_moves_nothing() {
    // The original fuses the normalisation into the second pass's writeback. The
    // port applies it afterwards. This is the claim that makes that legal: either
    // way it is one f32 multiply of the same two f32 operands.
    for (strike_count, dip_count) in SHAPES {
        check(strike_count, dip_count, Direction::Forward, 0.7, 0.3);
    }
}

#[test]
fn a_reused_engine_matches_a_fresh_one() {
    // Plans are cached and the scratch buffer is reused across calls of differing
    // lengths, which reallocates and invalidates them. A stale plan would read the
    // wrong memory, so this drives several sizes through one engine.
    let mut shared = FftwFft::new();
    for (strike_count, dip_count) in SHAPES {
        let mut reused = seeded(strike_count, dip_count);
        transform_2d(&mut reused, &mut shared, Direction::Forward);

        let mut fresh_engine = FftwFft::new();
        let mut fresh = seeded(strike_count, dip_count);
        transform_2d(&mut fresh, &mut fresh_engine, Direction::Forward);

        assert_eq!(
            reused.as_slice(),
            fresh.as_slice(),
            "reusing an engine changed the answer at {strike_count}x{dip_count}"
        );
    }
}

proptest! {
    #[test]
    fn transforms_match_for_arbitrary_extents_and_spacings(
        half_strike in 1usize..20,
        half_dip in 1usize..20,
        forward in any::<bool>(),
        d1 in 1e-3f32..10.0,
        d2 in 1e-3f32..10.0,
    ) {
        // Even extents only. genslip rounds every padded grid up to even at
        // genslip_v5.6.2.c:1471-1490, so an odd one never reaches a transform.
        let (strike_count, dip_count) = (2 * half_strike, 2 * half_dip);
        let direction = if forward { Direction::Forward } else { Direction::Inverse };
        let mut ported = seeded(strike_count, dip_count);
        let mut expected = as_oracle_grid(&ported);

        oracle::transform_2d(
            &mut expected, strike_count, dip_count,
            if forward { -1 } else { 1 }, d1, d2,
        );

        let mut fft = FftwFft::new();
        transform_2d(&mut ported, &mut fft, direction);
        scale(&mut ported, spacing_product(d1, d2));

        for (offset, (got, want)) in ported.as_slice().iter().zip(&expected).enumerate() {
            prop_assert_eq!(got.re.to_bits(), want.re.to_bits(), "re at {}", offset);
            prop_assert_eq!(got.im.to_bits(), want.im.to_bits(), "im at {}", offset);
        }
    }
}
