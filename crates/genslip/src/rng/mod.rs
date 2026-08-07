//! Where the randomness in a rupture model comes from.
//!
//! Every stochastic field genslip produces — slip, rake, the rupture-time and
//! rise-time perturbations, the roughness — is drawn from one generator. The
//! physics does not care *which* generator, but it cares enormously that the
//! answer is reproducible from a seed. So the draw source is a trait with two
//! implementations, and the choice is a deployment decision rather than a
//! property of the model:
//!
//! * [`GenslipLcg`] reproduces genslip v5.6.2 exactly — a 31-bit truncated linear
//!   congruential generator, with normals formed by summing twelve uniforms. It
//!   exists to compare against the original and to regenerate historical results.
//!   It is not a good generator and is not meant to be.
//!
//! * [`Pcg`] is what new work should use: PCG64-DXSM for the uniforms and the
//!   ziggurat algorithm for the normals. Better statistical quality, a period no
//!   run will exhaust, and roughly an order of magnitude fewer operations per
//!   normal deviate.
//!
//! # The two are not interchangeable mid-run, and that is the point
//!
//! Swapping the generator changes every field, because the fields share a stream.
//! It also changes the *number of draws* a normal costs — twelve versus a ziggurat
//! sample that usually costs one — so it is not merely a different sequence, it is
//! a different consumption pattern. Choose one per run and record which.
//!
//! What must hold for both, and what the tests assert, is the contract below:
//! the same seed gives the same model, and different realisations are independent.

mod compat;
mod pcg;

pub use compat::GenslipLcg;
pub use pcg::Pcg;

/// A source of the random deviates a rupture model is built from.
///
/// Implementations are free to use any algorithm. What they may not do is vary
/// their output for a given construction — a source is a deterministic function of
/// its seed, or nothing downstream is reproducible.
pub trait DrawSource {
    /// A uniform deviate on `[-1, 1]`, advancing the stream.
    ///
    /// The interval is closed at both ends rather than half-open: genslip's
    /// generator can return exactly `1.0`, and consumers are written to tolerate
    /// it. Do not narrow this without checking them.
    fn uniform(&mut self) -> f64;

    /// A normal deviate with the given standard deviation and mean, advancing the
    /// stream.
    ///
    /// `sigma` and `mean` are `f32` because that is what the fields carry; the
    /// deviate is `f64`.
    fn gaussian(&mut self, sigma: f32, mean: f32) -> f64;

    /// Advance the stream by `count` normal deviates, discarding them.
    ///
    /// This exists for fields that genslip generates and never uses. Their values
    /// are dead, but their *draws* are not: every field afterwards comes off the
    /// same stream, so skipping the generation without skipping the randomness
    /// changes the whole model.
    ///
    /// Consuming rather than computing is both simpler and much cheaper — the two
    /// such fields live on a grid nine times the fault's area, and building them
    /// costs four transforms on it.
    fn skip_gaussians(&mut self, count: usize) {
        for _ in 0..count {
            self.gaussian(1.0, 0.0);
        }
    }
}

/// A draw source that can produce independent streams for successive slip
/// realisations.
///
/// A rupture model is generated many times over from one seed, and realisation `n`
/// must be reproducible **without** generating realisations `0..n` first — that is
/// what makes a campaign restartable and parallelisable.
///
/// The two implementations achieve this differently, and the difference is
/// instructive. [`GenslipLcg`] rewinds to the starting seed and burns a fixed
/// number of draws, which is cheap only because the count is small and is not
/// really independence at all — the streams are consecutive segments of one
/// sequence. [`Pcg`] derives a genuinely distinct stream per realisation. Both
/// satisfy the contract; only one of them satisfies it well.
pub trait Realisations: DrawSource + Sized {
    /// An independent stream for slip realisation `index`.
    ///
    /// Must be a pure function of the receiver's original seed and `index`: calling
    /// it twice, or calling it after drawing, must give the same stream.
    #[must_use]
    fn realisation(&self, index: u64) -> Self;
}
