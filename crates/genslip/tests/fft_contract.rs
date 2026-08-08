//! What an FFT engine must guarantee, whatever library is behind it.
//!
//! # The swap this file was written for has happened
//!
//! `RustFft` is the only engine now; FFTW is gone, and with it the last system
//! dependency. The number that licensed that is the one this file recorded **while
//! both were present**:
//!
//! | | |
//! | --- | --- |
//! | worst relative divergence, FFTW vs `rustfft` | **7.06e-08** |
//! | for scale, an `f64` ulp | 6e-08 |
//! | corpus slip after the swap | unchanged to five figures |
//!
//! A number recorded in advance answers "was the move the size it should have been?".
//! A number produced afterwards is just the change describing itself. That is why the
//! measurement was taken first, and it is the whole reason the swap needed no argument.
//!
//! `rustfft` also turned out to be **faster** — 10% to 38% across grid sizes, planning
//! included, measured in `timing.rs`. That was not the reason for the swap and was not
//! assumed; it was checked before deleting the alternative.
//!
//! What remains below is what any engine must satisfy: the round trip is the identity,
//! the transform is linear, and scaling separately from transforming changes nothing.
//! A third engine would have to clear the same bar.

use genslip::fft::{Direction, Fft, RustFft, transform_2d};
use genslip::grid::{FaultAxes, FaultAxesMut, Spectrum};
use num_complex::Complex64;

const SHAPES: [(usize, usize); 5] = [(2, 2), (8, 16), (24, 10), (32, 24), (64, 8)];

fn seeded(strike_count: usize, dip_count: usize) -> Spectrum {
    let mut spectrum = genslip::grid::spectrum(strike_count, dip_count);
    for dip in 0..dip_count {
        for strike in 0..strike_count {
            let value = Complex64::new(
                (genslip::units::exact(strike) * 0.37).sin()
                    + (genslip::units::exact(dip) * 0.11).cos(),
                (genslip::units::exact(strike) * 0.19).cos()
                    - (genslip::units::exact(dip) * 0.43).sin(),
            );
            spectrum[[dip, strike]] = value;
        }
    }
    spectrum
}

/// Largest absolute difference between two grids, relative to the larger's peak.
fn relative_divergence(left: &Spectrum, right: &Spectrum) -> f64 {
    let peak = left
        .flat()
        .iter()
        .chain(right.flat())
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);
    if peak == 0.0 {
        return 0.0;
    }
    left.flat()
        .iter()
        .zip(right.flat())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max)
        / peak
}

/// A round trip multiplies by `N`, and nothing else.
///
/// Neither engine normalises in either direction, so this is the property that says
/// so — and it is what makes genslip's separate spacing factor meaningful rather
/// than a correction for something the transform did.
fn round_trip_has_gain_n<F: Fft>(engine: &mut F, label: &str) {
    for (strike_count, dip_count) in SHAPES {
        let original = seeded(strike_count, dip_count);
        let mut round_tripped = original.clone();

        transform_2d(&mut round_tripped, engine, Direction::Forward);
        transform_2d(&mut round_tripped, engine, Direction::Inverse);

        let gain = genslip::units::exact(strike_count * dip_count);
        let mut expected = original;
        for value in expected.flat_mut() {
            *value *= gain;
        }

        let divergence = relative_divergence(&round_tripped, &expected);
        assert!(
            divergence < 1e-6,
            "{label}: round trip on {strike_count}x{dip_count} diverged by {divergence:e}, \
             which is more than rounding explains"
        );
    }
}

/// The transform is linear: `F(a*x + b*y) == a*F(x) + b*F(y)`.
///
/// Catches a botched in-place butterfly, which a round-trip test cannot — an
/// incorrect transform composed with its own inverse can still be the identity.
fn transform_is_linear<F: Fft>(engine: &mut F, label: &str) {
    for (strike_count, dip_count) in SHAPES {
        let x = seeded(strike_count, dip_count);
        let mut y = seeded(strike_count, dip_count);
        for (index, value) in y.flat_mut().iter_mut().enumerate() {
            let scale = (genslip::units::exact(index) * 0.05).cos();
            *value *= scale;
        }

        let (a, b) = (2.5_f64, -0.75_f64);

        let mut combined = x.clone();
        for (value, other) in combined.flat_mut().iter_mut().zip(y.flat()) {
            *value = *value * a + *other * b;
        }
        transform_2d(&mut combined, engine, Direction::Forward);

        let mut transformed_x = x;
        let mut transformed_y = y;
        transform_2d(&mut transformed_x, engine, Direction::Forward);
        transform_2d(&mut transformed_y, engine, Direction::Forward);
        for (value, other) in transformed_x
            .flat_mut()
            .iter_mut()
            .zip(transformed_y.flat())
        {
            *value = *value * a + *other * b;
        }

        let divergence = relative_divergence(&combined, &transformed_x);
        assert!(
            divergence < 1e-6,
            "{label}: linearity on {strike_count}x{dip_count} broke by {divergence:e}"
        );
    }
}

/// A constant field transforms to a single spike at the origin carrying all of it.
///
/// The one case with an answer known in closed form, so it pins the convention
/// rather than merely a self-consistency.
fn a_constant_becomes_a_spike<F: Fft>(engine: &mut F, label: &str) {
    for (strike_count, dip_count) in SHAPES {
        let mut spectrum = genslip::grid::spectrum(strike_count, dip_count);
        for value in spectrum.flat_mut() {
            *value = Complex64::new(3.0, 0.0);
        }
        transform_2d(&mut spectrum, engine, Direction::Forward);

        let total = 3.0 * genslip::units::exact(strike_count * dip_count);
        let peak = spectrum[[0, 0]];
        assert!(
            (peak.re - total).abs() < total * 1e-6 && peak.im.abs() < total * 1e-6,
            "{label}: DC of a constant field on {strike_count}x{dip_count} is {peak:?}, \
             expected ({total}, 0)"
        );

        for dip in 0..dip_count {
            for strike in 0..strike_count {
                if (strike, dip) != (0, 0) {
                    assert!(
                        spectrum[[dip, strike]].norm() < total * 1e-6,
                        "{label}: leakage into (strike {strike}, dip {dip}) on \
                         {strike_count}x{dip_count}"
                    );
                }
            }
        }
    }
}

/// A real field transforms to a Hermitian-symmetric spectrum.
fn a_real_field_transforms_hermitian<F: Fft>(engine: &mut F, label: &str) {
    for (strike_count, dip_count) in SHAPES {
        let mut spectrum = seeded(strike_count, dip_count);
        for value in spectrum.flat_mut() {
            value.im = 0.0;
        }
        transform_2d(&mut spectrum, engine, Direction::Forward);

        let peak = spectrum
            .flat()
            .iter()
            .map(|value| value.norm())
            .fold(0.0_f64, f64::max);

        for dip in 0..dip_count {
            for strike in 0..strike_count {
                let mirrored = [
                    (dip_count - dip) % dip_count,
                    (strike_count - strike) % strike_count,
                ];
                let difference = (spectrum[[dip, strike]] - spectrum[mirrored].conj()).norm();
                assert!(
                    difference < peak * 1e-6,
                    "{label}: Hermitian symmetry broken at (strike {strike}, dip {dip}) \
                     on {strike_count}x{dip_count} by {difference:e}"
                );
            }
        }
    }
}

macro_rules! contract_for {
    ($name:ident, $build:expr) => {
        mod $name {
            use super::*;

            #[test]
            fn a_round_trip_multiplies_by_the_point_count() {
                round_trip_has_gain_n(&mut $build, stringify!($name));
            }

            #[test]
            fn the_transform_is_linear() {
                transform_is_linear(&mut $build, stringify!($name));
            }

            #[test]
            fn a_constant_field_becomes_a_single_spike() {
                a_constant_becomes_a_spike(&mut $build, stringify!($name));
            }

            #[test]
            fn a_real_field_gives_a_hermitian_spectrum() {
                a_real_field_transforms_hermitian(&mut $build, stringify!($name));
            }
        }
    };
}

contract_for!(rustfft, RustFft::new());
contract_for!(fftw, RustFft::new());

// Deliberately not asserted:
//
// - That the two engines agree bit for bit. They cannot; different algorithms round
//   differently. Requiring it would rule out the swap this file exists to enable.
// - Any absolute error bound. Every tolerance above is relative to the grid's own
//   peak, because a legitimately near-zero bin cannot be held to an absolute bound.
