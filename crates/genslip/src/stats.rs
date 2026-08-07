//! Summary statistics of a generated field.
//!
//! Used to renormalise a field after it is transformed back from the wavenumber
//! domain: the generators control the *shape* of the spectrum but not the mean or
//! the variance of the result, so both are measured and divided out.

use crate::grid::Spectrum;

/// The mean and population standard deviation of a field's real part.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeanAndSigma {
    pub mean: f32,
    /// Population standard deviation — divided by the sample count, not by
    /// `count - 1`. The field is the whole population, not a sample of one.
    pub sigma: f32,
}

/// Measure the mean and standard deviation of a spectrum's real part.
///
/// Only the real part participates: this is called after the inverse transform,
/// where the imaginary part is zero to rounding and carries no signal.
///
/// # Panics
///
/// Never in practice — [`Spectrum`] cannot be empty — but the division by the point
/// count would be undefined if it could be.
///
/// (orig. `get_mean_sigma_c`, misc.c:233)
#[must_use]
pub fn mean_and_sigma(spectrum: &Spectrum) -> MeanAndSigma {
    let values = spectrum.as_slice();
    assert!(!values.is_empty(), "cannot summarise an empty spectrum");

    #[expect(
        clippy::cast_precision_loss,
        reason = "grid point counts are far below 2^24"
    )]
    let count = values.len() as f32;

    // SIMPLIFY: accumulate in f64, or sum pairwise. This is a single-precision sum
    // over every grid point -- on a large fault that is ~10^5 terms, and a
    // left-to-right f32 fold loses several significant digits to accumulated
    // rounding. Changing it would be strictly *more* accurate, which is why it is a
    // simplification worth making rather than merely a tidy-up. Written this way
    // because the original accumulates through a `float *`.
    let mut total = 0.0_f32;
    for value in values {
        total += value.re;
    }
    let mean = total / count;

    // The second pass is the numerically stable form and should stay two-pass; only
    // the accumulator precision is worth changing.
    let mut sum_of_squares = 0.0_f32;
    for value in values {
        sum_of_squares += (value.re - mean) * (value.re - mean);
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the narrowing seam: C's sqrt returns double and is stored to a float"
    )]
    let sigma = f64::from(sum_of_squares / count).sqrt() as f32;

    MeanAndSigma { mean, sigma }
}
