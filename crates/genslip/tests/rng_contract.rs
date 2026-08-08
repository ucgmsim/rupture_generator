//! What every draw source must guarantee, regardless of algorithm.
//!
//! `rng_parity.rs` pins [`GenslipLcg`] to the C bit for bit. Those tests are
//! excellent regression tests and useless as specifications: they say what the
//! numbers are, never what makes them right, and they say nothing at all about
//! [`Pcg`].
//!
//! These do. Everything here is asserted against *both* implementations, so they
//! are the acceptance criteria for choosing a generator rather than an obstacle to
//! it — and if a third source is ever added, this file is the bar it has to clear.

use genslip::rng::{DrawSource, GenslipLcg, Pcg, Realisations};
use proptest::prelude::*;

/// Enough draws to make a distributional claim without making the suite slow.
const SAMPLE: usize = 20_000;

fn draws<S: DrawSource>(source: &mut S, count: usize) -> Vec<f64> {
    (0..count).map(|_| source.uniform()).collect()
}

/// The same seed gives the same stream. Everything else rests on this.
fn reproducible_from_its_seed<S: Realisations>(build: impl Fn(u64) -> S, seed: u64) {
    let first = draws(&mut build(seed), 1_000);
    let second = draws(&mut build(seed), 1_000);
    assert_eq!(first, second);
}

/// `realisation` is a pure function of the seed and index, not of stream position.
///
/// This is the property that makes a campaign restartable: realisation 40 must be
/// reproducible without having generated 0 to 39, and must not depend on whether
/// the parent stream has been drawn from.
fn realisations_are_positional<S: Realisations>(build: impl Fn(u64) -> S, seed: u64) {
    let parent = build(seed);
    let fresh = draws(&mut parent.realisation(40), 200);

    let mut drawn_parent = build(seed);
    let _ = draws(&mut drawn_parent, 5_000);
    let after_drawing = draws(&mut drawn_parent.realisation(40), 200);

    assert_eq!(
        fresh, after_drawing,
        "realisation() depends on the parent's stream position"
    );

    // And twice from the same parent.
    assert_eq!(fresh, draws(&mut parent.realisation(40), 200));
}

/// Distinct realisations are distinct streams.
fn realisations_differ<S: Realisations>(build: impl Fn(u64) -> S, seed: u64) {
    let parent = build(seed);
    let a = draws(&mut parent.realisation(0), 200);
    let b = draws(&mut parent.realisation(1), 200);
    assert_ne!(a, b);
}

/// Uniforms lie in the advertised closed interval.
fn uniforms_are_in_range<S: DrawSource>(source: &mut S) {
    for value in draws(source, SAMPLE) {
        assert!(
            (-1.0..=1.0).contains(&value),
            "uniform out of contract: {value}"
        );
        assert!(value.is_finite());
    }
}

/// A zero standard deviation gives the mean exactly, for any algorithm.
///
/// Worth pinning because the fields use it: several perturbations are configured
/// with a sigma that is zero or all but zero, and they must degenerate cleanly
/// rather than to a value near the mean.
fn zero_sigma_is_exactly_the_mean<S: DrawSource>(source: &mut S, mean: f64) {
    for _ in 0..100 {
        #[expect(
            clippy::float_cmp,
            reason = "exactness is the claim: a zero sigma must not merely land near the mean"
        )]
        let exact = source.gaussian(0.0, mean) == mean;
        assert!(exact);
    }
}

/// Normals have the requested first two moments.
///
/// Deliberately loose. This is not a normality test — Irwin-Hall is not normal and
/// would fail one — it checks that both sources are calibrated to the sigma and
/// mean they are asked for, which is the property the physics actually consumes.
fn normals_have_the_requested_moments<S: DrawSource>(source: &mut S, sigma: f64, mean: f64) {
    let values: Vec<f64> = (0..SAMPLE).map(|_| source.gaussian(sigma, mean)).collect();

    let n = genslip::units::exact(SAMPLE);
    let sample_mean = values.iter().sum::<f64>() / n;
    let variance = values
        .iter()
        .map(|value| (value - sample_mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);

    // The standard error of the mean is sigma/sqrt(n); five of them is a wide berth.
    let tolerance = 5.0 * sigma / n.sqrt();
    assert!(
        (sample_mean - mean).abs() < tolerance,
        "sample mean {sample_mean} not within {tolerance} of {mean}"
    );
    assert!(
        (variance.sqrt() - sigma).abs() < 0.05 * sigma,
        "sample sd {} not within 5% of {sigma}",
        variance.sqrt()
    );
}

/// Run the whole contract against one implementation.
macro_rules! contract_for {
    ($name:ident, $build:expr) => {
        mod $name {
            use super::*;

            #[test]
            fn is_reproducible_from_its_seed() {
                reproducible_from_its_seed($build, 20_260_807);
            }

            #[test]
            fn realisations_are_a_function_of_the_index_alone() {
                realisations_are_positional($build, 20_260_807);
            }

            #[test]
            fn distinct_realisations_are_distinct_streams() {
                realisations_differ($build, 20_260_807);
            }

            #[test]
            fn uniforms_stay_in_range() {
                uniforms_are_in_range(&mut ($build)(20_260_807));
            }

            #[test]
            fn a_zero_sigma_collapses_to_the_mean() {
                zero_sigma_is_exactly_the_mean(&mut ($build)(20_260_807), 3.25);
            }

            #[test]
            fn normals_are_calibrated() {
                normals_have_the_requested_moments(&mut ($build)(20_260_807), 0.75, 0.0);
                normals_have_the_requested_moments(&mut ($build)(1), 15.0, -4.0);
            }

            proptest! {
                #[test]
                fn any_seed_is_reproducible(seed in any::<u32>()) {
                    reproducible_from_its_seed($build, u64::from(seed));
                }

                #[test]
                fn any_seed_gives_positional_realisations(seed in any::<u32>()) {
                    realisations_are_positional($build, u64::from(seed));
                }
            }
        }
    };
}

// The seed types differ -- genslip's is a C `long`, the modern one a u64 -- so each
// gets its own constructor rather than the trait carrying a seed type it does not
// need.
#[expect(
    clippy::cast_possible_wrap,
    reason = "the LCG masks to 31 bits; the sign of the seed is not meaningful"
)]
mod builders {
    use super::{GenslipLcg, Pcg};

    pub fn genslip(seed: u64) -> GenslipLcg {
        GenslipLcg::new(seed as i64)
    }

    pub fn pcg(seed: u64) -> Pcg {
        Pcg::new(seed)
    }
}

contract_for!(genslip_lcg, builders::genslip);
contract_for!(pcg, builders::pcg);

/// What the compatibility generator promises beyond the shared contract.
///
/// These are not properties every draw source has — they are properties `GenslipLcg`
/// has *because* it reproduces genslip's, and the pipeline depends on both. They
/// moved here when the bit-parity suite went, restated against the port rather than
/// against the C, because what the pipeline needs is the relationship rather than the
/// provenance.
mod the_compatibility_generator {
    use super::{DrawSource, GenslipLcg};

    /// One normal costs exactly twelve uniforms.
    ///
    /// genslip's `gaus_rand` is Irwin–Hall: twelve uniforms summed, minus six. The
    /// *count* is the part that matters downstream — a generator producing the right
    /// numbers while consuming the wrong quantity of stream desynchronises every
    /// field after it, and the output still looks like plausible noise. Values alone
    /// cannot catch that, so the position is asserted.
    #[test]
    fn a_gaussian_costs_exactly_twelve_uniforms() {
        let mut after_gaussian = GenslipLcg::new(12_345);
        after_gaussian.gaussian(1.0, 0.0);

        let mut after_twelve = GenslipLcg::new(12_345);
        for _ in 0..12 {
            after_twelve.uniform();
        }

        assert_eq!(
            GenslipLcg::seed(after_gaussian),
            GenslipLcg::seed(after_twelve),
            "a normal did not advance the stream by twelve uniforms"
        );
    }

    /// Skipping ahead lands exactly where drawing and discarding would.
    ///
    /// The realisation loop reaches realisation `n` by burning `10n` draws rather
    /// than generating the realisations before it, so the shortcut has to be
    /// indistinguishable from the long way. `Realisations::realisation` rests on
    /// this, and `realisations_are_positional` above rests on that.
    #[test]
    fn discarding_lands_where_drawing_lands() {
        let mut skipped = GenslipLcg::new(7);
        skipped.discard(10 * 43);

        let mut drawn = GenslipLcg::new(7);
        for _ in 0..(10 * 43) {
            drawn.uniform();
        }

        assert_eq!(GenslipLcg::seed(skipped), GenslipLcg::seed(drawn));
    }
}

// Deliberately not asserted:
//
// - Normality. Irwin-Hall with n=12 is bounded on [-6, 6] and has no tails beyond
//   that; a Kolmogorov-Smirnov or Anderson-Darling test would fail it correctly and
//   tell us nothing we do not already know. The moment check above is the honest
//   version of the claim.
// - That the two sources agree with each other. They must not -- that is the point
//   of having both.
// - Independence between realisations beyond "they differ". GenslipLcg's
//   realisations are consecutive segments of one LCG sequence and are *not*
//   independent in any strong sense. Asserting they were would be asserting
//   something false about the compatibility path.
