//! How much randomness a stage consumed, and in what order.
//!
//! # Why `GenslipLcg::seed()` is not enough
//!
//! The compatibility generator exposes its state, and because that state is a
//! bijection of the draw count, comparing it after a run audits the count exactly.
//! That is how Stage 1 did it, and it has two holes.
//!
//! **It is generator-specific.** [`Pcg`](genslip::rng::Pcg) has no equivalent, so the
//! audit dies the moment the modern generator becomes the default — which is a Stage 3
//! goal. This decorator wraps *any* [`DrawSource`], so the same test covers both.
//!
//! **Count is not order.** Two stages that draw the same number of deviates can be
//! swapped without changing the total, or the final seed, while changing every field
//! downstream — and the output still looks like plausible noise. That is the failure
//! mode `ENGINEERING_RULES.md` rule 6 exists for, and the only defence is a
//! *checkpoint per stage*, which [`CountingSource::checkpoint`] provides.

use genslip::rng::{DrawSource, Realisations};

/// Wraps a draw source and records what passes through it.
///
/// Transparent: it forwards every call unchanged, so a run through the counter
/// produces bit-identical output to a run without one.
#[derive(Clone, Debug)]
pub struct CountingSource<S> {
    inner: S,
    uniforms: usize,
    gaussians: usize,
    checkpoints: Vec<Checkpoint>,
}

/// What had been drawn at a named point in the sequence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Checkpoint {
    pub label: &'static str,
    /// Gaussians drawn *since the previous checkpoint*, not since the start.
    ///
    /// A per-stage figure rather than a running total, because that is what makes a
    /// diff point at the stage that changed rather than at every stage after it.
    pub gaussians: usize,
    pub uniforms: usize,
}

impl<S: DrawSource> CountingSource<S> {
    #[must_use]
    pub const fn new(inner: S) -> Self {
        Self {
            inner,
            uniforms: 0,
            gaussians: 0,
            checkpoints: Vec::new(),
        }
    }

    /// Record what the stage just finished consumed, and start the next one.
    pub fn checkpoint(&mut self, label: &'static str) {
        let previous: (usize, usize) = self.checkpoints.iter().fold((0, 0), |(g, u), point| {
            (g + point.gaussians, u + point.uniforms)
        });
        self.checkpoints.push(Checkpoint {
            label,
            gaussians: self.gaussians - previous.0,
            uniforms: self.uniforms - previous.1,
        });
    }

    /// Total gaussians drawn.
    #[must_use]
    pub const fn gaussians(&self) -> usize {
        self.gaussians
    }

    /// Total uniforms drawn **directly**.
    ///
    /// Not the uniforms a gaussian consumed internally: the compatibility generator
    /// spends twelve per normal and the modern one spends about one, so counting
    /// those would make the audit generator-specific again — the thing this type
    /// exists to avoid.
    #[must_use]
    pub const fn uniforms(&self) -> usize {
        self.uniforms
    }

    #[must_use]
    pub fn checkpoints(&self) -> &[Checkpoint] {
        &self.checkpoints
    }

    /// The per-stage gaussian counts, in order, for comparing against a expectation.
    #[must_use]
    pub fn stage_gaussians(&self) -> Vec<(&'static str, usize)> {
        self.checkpoints
            .iter()
            .map(|point| (point.label, point.gaussians))
            .collect()
    }

    /// Give the wrapped source back.
    #[must_use]
    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S: DrawSource> DrawSource for CountingSource<S> {
    fn uniform(&mut self) -> f64 {
        self.uniforms += 1;
        self.inner.uniform()
    }

    fn gaussian(&mut self, sigma: f64, mean: f64) -> f64 {
        self.gaussians += 1;
        self.inner.gaussian(sigma, mean)
    }

    fn skip_gaussians(&mut self, count: usize) {
        // Forwarded rather than looped, so a source with a cheap skip keeps it -- but
        // still counted as the draws it stands for, because that is what the audit is
        // about.
        self.gaussians += count;
        self.inner.skip_gaussians(count);
    }
}

impl<S: Realisations> Realisations for CountingSource<S> {
    fn realisation(&self, index: u64) -> Self {
        Self::new(self.inner.realisation(index))
    }
}

/// Gaussians one spectral field consumes on a grid of these extents.
///
/// `2 * strike * (dip/2 + 1)`: a real and an imaginary part at every point of the
/// non-negative-wavenumber half, which is what Hermitian symmetry leaves free. It
/// does not depend on the spectrum, the band or the correlation lengths — every
/// generator draws first and shapes afterwards, including at points it then zeroes.
#[must_use]
pub const fn field_draw_count(strike_count: usize, dip_count: usize) -> usize {
    2 * strike_count * (dip_count / 2 + 1)
}
