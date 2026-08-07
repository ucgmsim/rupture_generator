//! The RNG is bit-identical to genslip's, draw for draw.
//!
//! This is the first gate in the port and the one everything else rests on. Every
//! field genslip generates comes off one shared stream, so a generator that is
//! merely *statistically* equivalent is useless: it desynchronises every consumer
//! after the first divergence, and the output still looks like plausible noise.
//!
//! These are differential tests against the real C, linked directly. There is no
//! golden file to regenerate and no driver to keep in sync — `oracle::rng::uniform`
//! *is* `sfrand`.

use genslip::rng::{DrawSource as _, GenslipLcg};
use genslip_oracle::rng as oracle;
use proptest::prelude::*;

/// Long enough to outrun any short-period artefact, short enough to stay quick.
/// A 2827-subfault fault consumes on the order of 10^6 draws per field.
const LONG_RUN: usize = 2_000_000;

#[test]
fn a_long_stream_is_bit_identical() {
    let mut oracle_seed: i64 = 1;
    let mut ported = GenslipLcg::new(1);

    for draw in 0..LONG_RUN {
        let expected = oracle::uniform(&mut oracle_seed);
        let actual = ported.uniform();

        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "draw {draw}: {actual:?} ({:#018x}) != {expected:?} ({:#018x})",
            actual.to_bits(),
            expected.to_bits(),
        );
    }

    // The state, not just the values. A generator that returned the right numbers
    // while leaving the seed somewhere else would desynchronise the next consumer.
    assert_eq!(
        ported.seed(),
        oracle_seed,
        "stream position diverged after {LONG_RUN} draws"
    );
}

#[test]
fn a_gaussian_costs_exactly_twelve_uniforms() {
    // Values alone cannot catch a Gaussian that consumes the wrong number of
    // deviates -- the caller would still get a plausible number and every
    // subsequent field would be silently wrong. So assert the position.
    let mut after_gaussian: i64 = 12_345;
    oracle::gaussian(1.0, 0.0, &mut after_gaussian);

    let mut after_twelve_uniforms: i64 = 12_345;
    oracle::uniforms(&mut after_twelve_uniforms, 12);

    assert_eq!(after_gaussian, after_twelve_uniforms);

    let mut ported = GenslipLcg::new(12_345);
    ported.gaussian(1.0, 0.0);
    assert_eq!(ported.seed(), after_gaussian);
}

#[test]
fn discard_lands_where_the_drawn_stream_lands() {
    // The realisation loop reaches realisation `js` by burning 10*js draws rather
    // than generating the realisations before it, so `discard` has to be exactly
    // equivalent to drawing and throwing away.
    let mut discarded = GenslipLcg::new(7);
    discarded.discard(10 * 43);

    let mut drawn = GenslipLcg::new(7);
    for _ in 0..(10 * 43) {
        drawn.uniform();
    }

    assert_eq!(discarded.seed(), drawn.seed());
}

proptest! {
    /// Bit-identical from any starting seed, not just the ones we happened to pick.
    #[test]
    fn uniforms_match_from_any_seed(
        seed in any::<i32>(),
        count in 1usize..2_000,
    ) {
        let mut oracle_seed = i64::from(seed);
        let mut ported = GenslipLcg::new(i64::from(seed));

        for draw in 0..count {
            let expected = oracle::uniform(&mut oracle_seed);
            let actual = ported.uniform();
            prop_assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "seed {}, draw {}", seed, draw
            );
        }
        prop_assert_eq!(ported.seed(), oracle_seed);
    }

    /// Gaussians match, including the f32 sigma/mean narrowing seam.
    #[test]
    fn gaussians_match_from_any_seed(
        seed in any::<i32>(),
        sigma in 1e-6f32..1e3,
        mean in -1e3f32..1e3,
        count in 1usize..200,
    ) {
        let mut oracle_seed = i64::from(seed);
        let mut ported = GenslipLcg::new(i64::from(seed));

        for draw in 0..count {
            let expected = oracle::gaussian(sigma, mean, &mut oracle_seed);
            let actual = ported.gaussian(sigma, mean);
            prop_assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "seed {}, draw {}", seed, draw
            );
        }
        prop_assert_eq!(ported.seed(), oracle_seed);
    }
}

// Deliberately not asserted:
//
// - That the uniform lies in [-1, 1). It does not, quite: the state spans
//   [0, 2^31) and is divided by 2^30, so the result reaches +1.0 exactly when the
//   state is 2^31 - 1... which the mask makes reachable. Asserting a half-open
//   range would be asserting something false about the original.
// - Any distributional property of `gaussian`. Irwin-Hall with n=12 has unit
//   variance but is bounded on [-6, 6] and has the wrong tails; it is an
//   approximation the program is calibrated around, not a normal distribution.
//   Testing it against a normal would be testing the wrong claim.
