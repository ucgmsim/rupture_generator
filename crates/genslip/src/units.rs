//! Units and conversions, named once each.
//!
//! genslip works in **CGS**: centimetres, grams, seconds, and therefore dyne-cm for
//! moment and dyne/cm² for rigidity. Its *inputs* are in kilometres and km/s, because
//! that is how a fault and a velocity model are written down. Every conversion in the
//! program is a consequence of that one mismatch.
//!
//! Each was a bare literal at its call site — `1.0e+10` twice in two modules,
//! `1.0e-09` once, `1.0e5` in the Python package — and a bare `1e10` is unreadable in
//! exactly the way that matters: you cannot tell an area conversion from a rigidity
//! conversion, and they are the same number for different reasons.
//!
//! # The derived ones are derived
//!
//! [`CM2_PER_KM2`] is `CM_PER_KM²` and [`RIGIDITY_SCALE`] is the same number again for
//! a third reason. They are written as expressions rather than as literals, so the
//! relationship is in the code, and `units.rs`'s own test asserts each against its
//! decimal value — a `const` expression that is wrong is wrong at compile time, and
//! the test is what says the decimal was right in the first place.

/// Centimetres per kilometre.
///
/// The root of every length conversion here.
pub const CM_PER_KM: f64 = 1.0e5;

/// Square centimetres per square kilometre.
///
/// Subfault areas: the fault is diced in kilometres and the moment sum is in dyne-cm,
/// so `μ A s` needs the area in cm².
pub const CM2_PER_KM2: f64 = CM_PER_KM * CM_PER_KM;

/// `(km/s)² · g/cm³` → `dyne/cm²`.
///
/// Rigidity is `ρ vs²`. A velocity model gives `vs` in km/s and `ρ` in g/cm³, and the
/// moment wants dyne/cm², so the speed's two factors of `CM_PER_KM` come through:
/// `(1e5)² = 1e10`. The same number as [`CM2_PER_KM2`] and not the same conversion,
/// which is the reason both have names.
pub const RIGIDITY_SCALE: f64 = CM_PER_KM * CM_PER_KM;

/// The coefficient in `trise = c · this · M₀^(1/3)`, Graves & Pitarka (2010).
///
/// Carries the units rather than a decade: `M₀` is in dyne-cm and `trise` in seconds,
/// so this is what makes a magnitude-7 moment give a rise time of order a second
/// rather than of order 1e9.
pub const RISE_TIME_MOMENT_SCALE: f64 = 1.0e-9;

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
/// `urs`'s shoulder index, `(2 - tail) · peak`, and the taper's `fraction · extent`:
/// fractions of a count that the original truncates to an int.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "as `samples` above: saturating, and NaN to zero"
)]
pub fn truncated(value: f64) -> usize {
    (value as i64).max(0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derived constants are the decimals they replaced.
    ///
    /// Not circular. The `const` expressions say *why* each number is what it is —
    /// `CM_PER_KM²` rather than `1e10` — and this says the decimal they replaced was
    /// right. Getting the exponent wrong by one is the failure mode, and it is a
    /// factor of ten in the seismic moment: an entire magnitude unit of error, from a
    /// literal nobody would look at twice.
    #[test]
    fn the_derived_constants_are_the_literals_they_replaced() {
        assert!((CM_PER_KM - 1.0e5).abs() < f64::EPSILON);
        assert!((CM2_PER_KM2 - 1.0e10).abs() < 1.0);
        assert!((RIGIDITY_SCALE - 1.0e10).abs() < 1.0);
        assert!((RISE_TIME_MOMENT_SCALE - 1.0e-9).abs() < f64::EPSILON * 1.0e-9);
    }

    /// A rigidity a seismologist would recognise.
    ///
    /// Crustal rock: 3.2 km/s and 2.6 g/cm³ gives about 2.7e11 dyne/cm², which is
    /// 27 `GPa`. That is the number this conversion exists to produce, and it is the
    /// check a reader can make without trusting the arithmetic above.
    #[test]
    fn crustal_rigidity_is_about_thirty_gigapascals() {
        let rigidity = 2.6 * 3.2 * 3.2 * RIGIDITY_SCALE;
        assert!(
            (2.0e11..3.5e11).contains(&rigidity),
            "{rigidity:e} dyne/cm^2 is not crustal rock"
        );
    }

    /// A subfault a kilometre across is 1e10 cm².
    #[test]
    fn a_square_kilometre_is_ten_billion_square_centimetres() {
        assert!((1.0 * 1.0 * CM2_PER_KM2 - 1.0e10).abs() < 1.0);
    }

    /// `samples` saturates rather than wrapping, at both ends and on NaN.
    ///
    /// The property the name exists to promise. A negative or NaN duration gives no
    /// samples, which is a pulse that does not exist; the alternative in C is
    /// undefined behaviour and in a naive Rust cast would be a huge allocation.
    #[test]
    fn a_sample_count_saturates_rather_than_wrapping() {
        // Ten intervals, not eleven points: `+ 0.5` then truncate rounds to nearest,
        // and 10.5 truncates down. That is the original's rounding and the reason
        // this is one function rather than eleven spellings of it.
        assert_eq!(samples(1.0, 0.1), 10);
        assert_eq!(samples(1.06, 0.1), 11);

        // The three that are undefined behaviour in C.
        assert_eq!(samples(-1.0, 0.1), 0);
        assert_eq!(samples(f64::NAN, 0.1), 0);
        assert_eq!(samples(0.0, 0.1), 0);
        assert!(
            samples(f64::INFINITY, 0.1) > 0,
            "saturates rather than wrapping"
        );
    }

    #[test]
    fn a_truncated_count_floors_at_zero() {
        assert_eq!(truncated(3.9), 3);
        assert_eq!(truncated(-3.9), 0);
        assert_eq!(truncated(f64::NAN), 0);
    }
}
