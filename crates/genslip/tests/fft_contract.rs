//! What every FFT engine must guarantee, and how far apart the two are.
//!
//! `fft_parity.rs` pins [`FftwFft`] to genslip bit for bit. Those tests say what the
//! numbers are and nothing about what makes them right, and they say nothing at all
//! about [`RustFft`].
//!
//! These do. Everything below runs against both engines, so they are the acceptance
//! criteria for the Stage 3 swap rather than an obstacle to it.
//!
//! The last test is the point of the file: it **measures** how far the two engines
//! diverge, now, while both are present. Stage 3 replaces the engine and will move
//! every field in the program; the question then is whether the move is the size it
//! should be. A number recorded in advance answers that. A number produced
//! afterwards is just the change describing itself.

use genslip::fft::{Direction, Fft, RustFft, transform_2d};
use genslip::grid::Spectrum;
use num_complex::Complex32;

#[cfg(feature = "fftw")]
use genslip::fft::FftwFft;

const SHAPES: [(usize, usize); 5] = [(2, 2), (8, 16), (24, 10), (32, 24), (64, 8)];

fn seeded(strike_count: usize, dip_count: usize) -> Spectrum {
    let mut spectrum = Spectrum::zeros(strike_count, dip_count);
    for dip in 0..dip_count {
        for strike in 0..strike_count {
            #[expect(clippy::cast_precision_loss, reason = "small test indices")]
            let value = Complex32::new(
                (strike as f32 * 0.37).sin() + (dip as f32 * 0.11).cos(),
                (strike as f32 * 0.19).cos() - (dip as f32 * 0.43).sin(),
            );
            spectrum[(strike, dip)] = value;
        }
    }
    spectrum
}

/// Largest absolute difference between two grids, relative to the larger's peak.
fn relative_divergence(left: &Spectrum, right: &Spectrum) -> f32 {
    let peak = left
        .as_slice()
        .iter()
        .chain(right.as_slice())
        .map(|value| value.norm())
        .fold(0.0_f32, f32::max);
    if peak == 0.0 {
        return 0.0;
    }
    left.as_slice()
        .iter()
        .zip(right.as_slice())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f32, f32::max)
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

        #[expect(clippy::cast_precision_loss, reason = "small grid extents")]
        let gain = (strike_count * dip_count) as f32;
        let mut expected = original;
        for value in expected.as_mut_slice() {
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
        for (index, value) in y.as_mut_slice().iter_mut().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "small test indices")]
            let scale = (index as f32 * 0.05).cos();
            *value *= scale;
        }

        let (a, b) = (2.5_f32, -0.75_f32);

        let mut combined = x.clone();
        for (value, other) in combined.as_mut_slice().iter_mut().zip(y.as_slice()) {
            *value = *value * a + *other * b;
        }
        transform_2d(&mut combined, engine, Direction::Forward);

        let mut transformed_x = x;
        let mut transformed_y = y;
        transform_2d(&mut transformed_x, engine, Direction::Forward);
        transform_2d(&mut transformed_y, engine, Direction::Forward);
        for (value, other) in transformed_x
            .as_mut_slice()
            .iter_mut()
            .zip(transformed_y.as_slice())
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
        let mut spectrum = Spectrum::zeros(strike_count, dip_count);
        for value in spectrum.as_mut_slice() {
            *value = Complex32::new(3.0, 0.0);
        }
        transform_2d(&mut spectrum, engine, Direction::Forward);

        #[expect(clippy::cast_precision_loss, reason = "small grid extents")]
        let total = 3.0 * (strike_count * dip_count) as f32;
        let peak = spectrum[(0, 0)];
        assert!(
            (peak.re - total).abs() < total * 1e-6 && peak.im.abs() < total * 1e-6,
            "{label}: DC of a constant field on {strike_count}x{dip_count} is {peak:?}, \
             expected ({total}, 0)"
        );

        for dip in 0..dip_count {
            for strike in 0..strike_count {
                if (strike, dip) != (0, 0) {
                    assert!(
                        spectrum[(strike, dip)].norm() < total * 1e-6,
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
        for value in spectrum.as_mut_slice() {
            value.im = 0.0;
        }
        transform_2d(&mut spectrum, engine, Direction::Forward);

        let peak = spectrum
            .as_slice()
            .iter()
            .map(|value| value.norm())
            .fold(0.0_f32, f32::max);

        for dip in 0..dip_count {
            for strike in 0..strike_count {
                let mirrored = (
                    (strike_count - strike) % strike_count,
                    (dip_count - dip) % dip_count,
                );
                let difference = (spectrum[(strike, dip)] - spectrum[mirrored].conj()).norm();
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
#[cfg(feature = "fftw")]
contract_for!(fftw, FftwFft::new());

/// How far the two engines diverge — measured, and recorded for Stage 3.
///
/// This is not a pass/fail claim about either engine. It is a baseline: when the
/// swap lands and every field in the program moves, this says how much of the move
/// is the engine and how much is something else.
#[cfg(feature = "fftw")]
#[test]
fn the_two_engines_agree_to_single_precision_rounding() {
    let mut fftw = FftwFft::new();
    let mut rust = RustFft::new();
    let mut worst = 0.0_f32;

    for (strike_count, dip_count) in SHAPES {
        for direction in [Direction::Forward, Direction::Inverse] {
            let mut by_fftw = seeded(strike_count, dip_count);
            let mut by_rust = by_fftw.clone();

            transform_2d(&mut by_fftw, &mut fftw, direction);
            transform_2d(&mut by_rust, &mut rust, direction);

            let divergence = relative_divergence(&by_fftw, &by_rust);
            worst = worst.max(divergence);

            assert!(
                divergence < 1e-6,
                "{direction:?} on {strike_count}x{dip_count}: engines differ by \
                 {divergence:e}, which is more than f32 rounding explains"
            );
        }
    }

    // Printed rather than asserted tightly, so `cargo test -- --nocapture` reports
    // the number that Stage 3 has to be judged against.
    println!("FFTW vs rustfft: worst relative divergence {worst:e}");
}

// Deliberately not asserted:
//
// - That the two engines agree bit for bit. They cannot; different algorithms round
//   differently. Requiring it would rule out the swap this file exists to enable.
// - Any absolute error bound. Every tolerance above is relative to the grid's own
//   peak, because a legitimately near-zero bin cannot be held to an absolute bound.
