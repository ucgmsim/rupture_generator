//! Which of the original's roundabout spellings are *exactly* their short form.
//!
//! `SIMPLIFICATIONS.md` splits the deferred rewrites into "provably free" and "moves
//! bits", and until this file existed that split was an assertion. It was also wrong:
//! three `sqrt(x*x)` sites were filed as bit-moving when they are exact.
//!
//! The distinction is not a curiosity. Under `ENGINEERING_RULES.md` a free rewrite
//! needs no adjudication at all — the arithmetic is identical, so no test can tell and
//! none is owed. A bit-moving one needs a measurement and a drift line in the commit.
//! Getting a site into the wrong column either wastes an argument or skips one that
//! was needed.
//!
//! So each claim here is executable, and the ones that are **false** are asserted too.
//! A file that only recorded the exact identities would read as though every
//! roundabout spelling were free.
//!
//! Every site these describe has now been taken. The tests stay because the *claims*
//! outlive the sites: they are what says the four free rewrites owed no argument and
//! the bit-moving ones did. The final tally was two more misfilings — see
//! `a_ten_digit_pi_is_pi_in_single_precision`, and `SIMPLIFICATIONS.md` for the ramp.

use proptest::prelude::*;

/// `sqrt(x*x) == |x|`, and why.
///
/// Under round-to-nearest, `fl(x*x)` is within half an ulp of `x²`, and a correctly
/// rounded `sqrt` maps that back to the nearest representable root — which is `|x|`
/// itself, because `|x|` *is* representable. The rounding cannot escape the interval
/// that contains only one representable value.
///
/// It fails at both ends of the range, where `x*x` leaves the normal numbers, and
/// `the_identity_fails_once_the_square_leaves_the_normal_range` pins that.
mod magnitude_is_a_square_root {
    use super::*;

    /// Largest `f32` whose square is still finite.
    const F32_SAFE_MAX: f32 = 1.8e19;
    /// Smallest `f32` whose square is still normal.
    const F32_SAFE_MIN: f32 = 1.1e-19;

    #[test]
    fn on_the_values_the_kernels_actually_see() {
        // `source.rs` takes it on a rake offset bounded by 90; `rise_time.rs` on a
        // field mean and on slip in centimetres. Nothing near either end.
        for magnitude in [0.0_f64, 1.0, 90.0, 1e-6, 1234.5, 1e6, 1e12] {
            for value in [magnitude, -magnitude] {
                assert_eq!((value * value).sqrt().to_bits(), value.abs().to_bits());
                #[expect(clippy::cast_possible_truncation, reason = "small exact test values")]
                let single = value as f32;
                assert_eq!(
                    (single * single).sqrt().to_bits(),
                    single.abs().to_bits(),
                    "f32 at {single}"
                );
            }
        }
    }

    #[test]
    fn the_identity_fails_once_the_square_leaves_the_normal_range() {
        // The caveat, asserted rather than trusted -- so that anyone applying this
        // rewrite somewhere new has to check the range first.
        let huge = 1.0e30_f32;
        assert!((huge * huge).sqrt().is_infinite());
        assert!(huge.abs().is_finite());

        let tiny = 1.0e-30_f32;
        assert_eq!(
            (tiny * tiny).sqrt().to_bits(),
            0.0_f32.to_bits(),
            "the square underflowed to zero"
        );
        assert!(tiny.abs() > 0.0);
    }

    proptest! {
        /// Every `f32` in the safe range, in both evaluation orders.
        ///
        /// Two orders because the C widens at the call — `sqrt` takes a `double`, so
        /// `sqrt(x*x)` computes the product in `float` and the root in `double`. The
        /// identity has to hold either way for the rewrite to be free at every site.
        #[test]
        fn for_any_f32_whose_square_stays_normal(
            magnitude in F32_SAFE_MIN..F32_SAFE_MAX,
            negative in any::<bool>(),
        ) {
            let value = if negative { -magnitude } else { magnitude };

            // Product and root both single.
            prop_assert_eq!((value * value).sqrt().to_bits(), value.abs().to_bits());

            // Product single, root widened -- the C's shape.
            #[expect(clippy::cast_possible_truncation, reason = "the narrowing seam under test")]
            let widened = f64::from(value * value).sqrt() as f32;
            prop_assert_eq!(widened.to_bits(), value.abs().to_bits());

            // Both double, which is what `source.rs` does.
            let doubled = f64::from(value);
            prop_assert_eq!((doubled * doubled).sqrt().to_bits(), doubled.abs().to_bits());
        }
    }
}

/// `4*atan(1) == PI`, exactly.
///
/// `atan(1)` is correctly rounded to the nearest double below π/4, and multiplying by
/// four only shifts the exponent. Both steps are exact, so the original's spelling and
/// the constant are the same bits.
#[test]
fn four_arctangents_of_one_are_pi() {
    assert_eq!(
        (4.0 * 1.0_f64.atan()).to_bits(),
        std::f64::consts::PI.to_bits()
    );
}

/// A ten-digit literal is the constant in `f32` and is not in `f64`.
///
/// `SIMPLIFICATIONS.md` had genslip's `3.141592654` in the bit-moving column on the
/// reasoning that a ten-digit decimal is not pi. True of the number and false of the
/// `f32`: they first differ at the tenth digit, an `f32` carries about seven, and both
/// round to `0x40490fdb`. So the substitution in the pulse generator, where the
/// constant is a `float`, is free — and the corpus agrees, no slip-rate sample moved.
///
/// **What decides it is the width, not the digit count.** The same test on the same
/// literals in `double` fails, which is why the second half of this is asserted: it is
/// the difference between the pulse's `pi` (free) and the geodesy's `rperd` (not, and
/// one of the four separate faults that got `set_ll` deleted).
#[expect(
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::unreadable_literal,
    reason = "the original's literals are the subject of the test"
)]
#[test]
fn a_ten_digit_constant_survives_f32_and_not_f64() {
    // `gslip_sliprate_subs.c`: `float pi = 3.141592654;`
    let pi_single: f32 = 3.141592654;
    assert_eq!(pi_single.to_bits(), std::f32::consts::PI.to_bits());

    // `misc.c`: `rperd = 0.017453293`, about 5e-10 short of pi/180.
    let degrees_single: f32 = 0.017453293;
    assert_eq!(
        degrees_single.to_bits(),
        (std::f32::consts::PI / 180.0).to_bits()
    );

    // Both of them, at double width, are a different number.
    let pi_double: f64 = 3.141592654;
    assert_ne!(pi_double.to_bits(), std::f64::consts::PI.to_bits());

    let degrees_double: f64 = 0.017453293;
    assert_ne!(
        degrees_double.to_bits(),
        (std::f64::consts::PI / 180.0).to_bits()
    );
    assert!(
        (degrees_double - std::f64::consts::PI / 180.0).abs() < 1e-9,
        "and the gap is small enough that only the width makes it visible"
    );
}

/// The other side of the taxonomy: rewrites that are **not** free.
///
/// Each of these was a `SIMPLIFY` site, and each was worth making — `powi` is three
/// multiplies where `exp(n·ln x)` is two transcendental calls, and `cbrt` is exact
/// where the pair is not. But they change the last bits, so each owed a measurement
/// and a recorded drift rather than a silent commit. All are taken; the assertions
/// remain as the record of why they were not free.
mod these_move_bits_and_owe_a_measurement {
    /// `x.powi(4)` against `exp(4·ln x)` — `field.rs`'s band-pass.
    #[test]
    fn an_integer_power_is_not_its_exp_log_spelling() {
        let disagreements = (1..20_000)
            .map(|step| f64::from(step) * 0.001)
            .filter(|x| (4.0 * x.ln()).exp().to_bits() != x.powi(4).to_bits())
            .count();
        assert!(
            disagreements > 0,
            "if these ever agree everywhere, the band-pass rewrite became free"
        );
    }

    /// `cbrt` against `exp(ln x / 3)` — `source.rs`'s rise-time scaling.
    #[test]
    fn a_cube_root_is_not_its_exp_log_spelling() {
        // And `cbrt` is the *more* accurate of the two: it lands exactly on cubes.
        assert_eq!(27.0_f64.cbrt().to_bits(), 3.0_f64.to_bits());
        assert_ne!((27.0_f64.ln() / 3.0).exp().to_bits(), 3.0_f64.to_bits());
    }

    /// `hypot` against `sqrt(re² + im²)` — `field.rs`'s DC magnitude.
    #[test]
    fn a_hypotenuse_is_not_the_naive_square_root() {
        // Where they differ most is where the naive form overflows and `hypot` does
        // not, which is the argument for the rewrite rather than a curiosity.
        let large = 1.0e200_f64;
        assert!((large * large + large * large).sqrt().is_infinite());
        assert!(large.hypot(large).is_finite());
    }
}
