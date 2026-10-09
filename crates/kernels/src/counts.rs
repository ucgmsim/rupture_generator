//! Counts and indices as floats, named once each.
//!
//! A grid index and a sample count are integers that arithmetic needs as `f64`. The
//! two conversions below are the only casts the crate makes, and the only place the
//! kernels depend on what a number counts.

/// A grid index, a sample index, or a count of either, as a float.
///
/// `usize as f64` stops being exact past 2⁵³, and clippy rightly warns about it in
/// general. Every integer this crate converts is a subfault index, a sample index or
/// a count of one of those, and a fault of nine quadrillion subfaults would signal a
/// bug somewhere else entirely.
///
/// So the suppression appears **once**, here, with the bound stated.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "counts and indices in this crate are subfault- or sample-sized, which \
              is many orders below the 2^53 where this stops being exact"
)]
pub(crate) const fn exact(count: usize) -> f64 {
    count as f64
}

/// How many samples a duration covers, rounded to nearest and floored at zero.
///
/// In Rust the float-to-int cast saturates and NaN becomes zero, which is the right
/// answer in every degenerate case: a pulse with no samples, rather than a pulse of
/// garbage. Named so the saturation is a decision rather than a language detail
/// nobody checked.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Rust's float-to-int cast saturates and maps NaN to zero, which is the \
              behaviour this function exists to name"
)]
pub(crate) fn samples(seconds: f64, sample_interval_s: f64) -> usize {
    ((seconds / sample_interval_s + 0.5) as i64).max(0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `samples` saturates rather than wrapping, at both ends and on NaN.
    ///
    /// A negative or NaN duration must come out as an empty pulse. A naive cast would
    /// be a huge allocation.
    #[test]
    fn a_sample_count_saturates_rather_than_wrapping() {
        // The count is of intervals, not points (ten, not eleven). `+ 0.5` then
        // truncate rounds to nearest, and 10.5 truncates down.
        assert_eq!(samples(1.0, 0.1), 10);
        assert_eq!(samples(1.06, 0.1), 11);

        assert_eq!(samples(-1.0, 0.1), 0);
        assert_eq!(samples(f64::NAN, 0.1), 0);
        assert_eq!(samples(0.0, 0.1), 0);
        assert!(
            samples(f64::INFINITY, 0.1) > 0,
            "saturates rather than wrapping"
        );
    }
}
