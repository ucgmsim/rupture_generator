//! Summary statistics of a generated field.
//!
//! Used to renormalise a field after it is transformed back from the wavenumber
//! domain: the generators control the *shape* of the spectrum but not the mean or
//! the variance of the result, so both are measured and divided out.

use crate::grid::{FaultAxes, Spectrum};
use crate::units;

/// The mean and population standard deviation of a field's real part.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeanAndSigma {
    pub mean: f64,
    /// Population standard deviation — divided by the sample count, not by
    /// `count - 1`. The field is the whole population, not a sample of one.
    pub sigma: f64,
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
    let values = spectrum.flat();
    assert!(!values.is_empty(), "cannot summarise an empty spectrum");

    let count = values.len();

    // Accumulated in `f64`. The original folds through a `float` over every grid
    // point -- on a large fault ~10^5 terms, where a single-precision left-to-right
    // sum loses several significant digits. Widening is strictly more accurate, and
    // both results narrow to `f64` at the end because that is what the field is.
    let total: f64 = values.iter().map(|value| value.re).sum();
    let mean = total / units::exact(count);

    // Two passes deliberately: it is the numerically stable form, and one-pass or
    // Welford would trade accuracy for a speed that nothing here needs.
    let sum_of_squares: f64 = values.iter().map(|value| (value.re - mean).powi(2)).sum();

    MeanAndSigma {
        mean,
        sigma: (sum_of_squares / units::exact(count)).sqrt(),
    }
}
