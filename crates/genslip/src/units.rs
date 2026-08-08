//! Units and conversions, named once.
//!
//! Stage 4 of the refactor fills this with the conversion constants that are still
//! written inline in `moment.rs` and `source.rs`. It starts with the one thing the
//! `f64` conversion needed: a name for turning a count into a float.

/// A grid index, a sample index, or a count of either, as a float.
///
/// `usize as f64` stops being exact above 2⁵³, and clippy says so — correctly, in
/// general. Every integer this crate converts is a subfault index, a sample index or
/// a count of one of those: a fault of nine quadrillion subfaults is not a case to
/// handle, it is a bug somewhere else entirely.
///
/// So the suppression is written **once**, here, with the bound stated. Before this
/// existed the same reason was copied across twenty-five call sites, each of which
/// could drift from the others and none of which a reader could check.
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "counts and indices in this crate are subfault- or sample-sized, which \
              is many orders below the 2^53 where this stops being exact"
)]
pub const fn exact(count: usize) -> f64 {
    count as f64
}

/// How many samples a duration covers, rounded to nearest and floored at zero.
///
/// The original writes `(int)(seconds/dt + 0.5)` in eleven places. In C a duration
/// that is NaN, negative or larger than `INT_MAX` makes that undefined behaviour; in
/// Rust the cast saturates and NaN becomes zero, which is the answer we want in every
/// one of those cases — a pulse with no samples, rather than a pulse of garbage.
///
/// Named so the saturation is a decision rather than a language detail nobody
/// checked, and so the eleven call sites cannot each round differently.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Rust's float-to-int cast saturates and maps NaN to zero, which is the \
              behaviour this function exists to name"
)]
pub fn samples(seconds: f64, sample_interval_s: f64) -> usize {
    ((seconds / sample_interval_s + 0.5) as i64).max(0) as usize
}

/// A count from a float that is already a whole number, floored at zero.
///
/// `urs`'s shoulder index, `(2 - tail) * peak`, is the only caller: a fraction of a
/// sample count that the original truncates to an int.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "as `samples` above: saturating, and NaN to zero"
)]
pub fn truncated(value: f64) -> usize {
    (value as i64).max(0) as usize
}
