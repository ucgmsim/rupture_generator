//! Two-dimensional transforms between the fault and the wavenumber domain.
//!
//! The fields are generated in the wavenumber domain and consumed on the fault, so
//! every one of them makes this round trip. It is also the single hottest thing the
//! program does.
//!
//! # The engine is swappable, and the decomposition is not
//!
//! There is no two-dimensional FFT here, in either implementation: a 2-D transform
//! is a pass of 1-D transforms along one axis followed by a pass along the other.
//! That decomposition is the same whichever library does the 1-D work, so it lives
//! in [`transform_2d`] and the trait is only [`Fft`], a 1-D engine.
//!
//! [`RustFft`] is the only engine. It replaced FFTW, which the port called because
//! genslip does; the two agreed to 7.06e-08 relative — about an `f64` ulp — and
//! `rustfft` measured 10% to 38% faster. Both numbers were recorded before the
//! alternative was deleted — the accuracy one in `tests/fft_contract.rs`, the speed
//! one here, because a test that times a deleted engine turns into a test that times
//! its replacement against itself, which is what happened.
//!
//! **They were never bit-identical and could not be** — different algorithms round
//! differently, so changing engines moved the last bits of every field. What made the
//! change decidable rather than arguable was having the size of that move written down
//! first: on the corpus it turned out to be invisible, slip unchanged to five figures.
//!
//! # Normalisation is not the transform's business
//!
//! Neither engine scales in either direction — an unnormalised forward transform
//! followed by an unnormalised inverse multiplies by `N`. genslip does not correct
//! for that. Instead it multiplies by the product of the *sample spacings* of the
//! domain it started in, which is a discrete approximation to the integral in a
//! continuous Fourier transform, and it applies that factor only on the second pass.
//!
//! So the round trip is not the identity: it has gain `N * d1 * d2` one way and
//! `d1 * d2` the other, and the callers rely on it. See [`scale`].
//!
//! (orig. `fft2d_fftw`, slip.c:873)

mod rust;

pub use rust::RustFft;

use num_complex::Complex64;

use crate::grid::{FaultAxes, FaultAxesMut, Spectrum};

/// Which way a transform goes.
///
/// The names are the physical ones. genslip spells them `-1` and `+1`, which are
/// FFTW's sign conventions for the exponent and are easy to read backwards.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction {
    /// Fault to wavenumber. FFTW's `FFTW_FORWARD`, exponent sign `-1`.
    Forward,
    /// Wavenumber to fault. FFTW's `FFTW_BACKWARD`, exponent sign `+1`.
    Inverse,
}

/// A one-dimensional complex FFT engine.
///
/// Implementations must be **unnormalised in both directions**: a forward transform
/// followed by an inverse must multiply by the length, not return the input. That is
/// what both FFTW and `rustfft` do by default, and what the callers here assume.
pub trait Fft {
    /// Transform `buffer` in place.
    fn transform(&mut self, buffer: &mut [Complex64], direction: Direction);
}

/// Transform a grid in place, along strike and then down dip.
///
/// Unnormalised — see [`scale`], which the callers apply afterwards.
///
/// The pass order matters for the last bits even though it does not matter
/// mathematically: each pass rounds, so doing dip first would give a different
/// answer. Strike first is what the original does.
///
/// # The dip pass gathers, and that was measured twice
///
/// Neither FFTW as genslip calls it nor `rustfft` takes a stride, so a column has to
/// be made contiguous somehow. This gathers each column into a scratch buffer,
/// transforms it, and scatters it back, which is what genslip does.
///
/// It **did** transpose the whole grid instead — one tiled transpose, contiguous
/// rows, transpose back — on a measurement of 1.7× to 2.2×. Two later changes erased
/// that, and both are worth knowing because neither was about the transform:
///
/// | | gathered | transposed | |
/// | --- | ---: | ---: | ---: |
/// | hand-rolled grid type, `f32` | 7.22 ms | 3.29 ms | **2.19×** |
/// | after `ndarray` | 2.96 ms | 3.21 ms | 0.92× |
/// | after `f64` | 3.89 ms | 4.26 ms | 0.91× |
///
/// (512×512, forward and inverse, best of nine, `--release`.)
///
/// The transposed column never moved. **`ndarray` made the gather 2.4× faster** —
/// its `Index` computes offsets from precomputed strides where the hand-rolled one
/// reloaded an extent field through a reference the loop also wrote through, which
/// the compiler could not hoist. The win the transpose existed for was a defect in
/// the type it was measured against.
///
/// `f64` was the second chance and did not take it: doubling the memory traffic is
/// exactly the condition a cache-blocked transpose should win under, and the gather
/// is still ahead — by 10% at 512×512 and 22% at 32×32, where the transpose's
/// full-grid allocation is most of the work. So the transpose, its 32-element tiling
/// constant and that allocation are gone.
///
/// Reverting it took the whole rupture from 1.538 ms to **1.298 ms**, which is more
/// than the 6.5% `f64` cost back.
///
/// `tests/timing.rs::fft_dip_pass` keeps both spellings and asserts they are
/// bit-identical before timing either, so the table above can be re-run rather than
/// trusted — which is the only reason any of it was decidable.
pub fn transform_2d<F: Fft + ?Sized>(spectrum: &mut Spectrum, fft: &mut F, direction: Direction) {
    let strike_count = spectrum.strike_count();
    let dip_count = spectrum.dip_count();

    // Along strike: each row is already contiguous.
    for dip in 0..dip_count {
        let start = dip * strike_count;
        fft.transform(
            &mut spectrum.flat_mut()[start..start + strike_count],
            direction,
        );
    }

    // Down dip: each column is strided, so it is gathered into scratch and scattered
    // back. One buffer of `dip_count`, reused, against the transpose's full grid.
    let mut column = vec![Complex64::default(); dip_count];
    for strike in 0..strike_count {
        for (dip, value) in column.iter_mut().enumerate() {
            *value = spectrum[[dip, strike]];
        }
        fft.transform(&mut column, direction);
        for (dip, value) in column.iter().enumerate() {
            spectrum[[dip, strike]] = *value;
        }
    }
}

/// Multiply every point by `factor`.
///
/// Separated from [`transform_2d`] deliberately, though the original fuses it into
/// the second pass's writeback. It costs nothing to separate: `factor * value`
/// rounded to `f64` is the same whether the value was stored first or not, because
/// either way it is one `f64` multiply of the same two `f64` operands. And it keeps
/// a genslip convention out of a function whose job is the transform.
///
/// The factor callers pass is the product of the sample spacings of the domain the
/// transform started in.
pub fn scale(spectrum: &mut Spectrum, factor: f64) {
    for value in spectrum.flat_mut() {
        *value *= factor;
    }
}

/// The normalisation genslip applies after a transform: the product of the source
/// domain's sample spacings.
///
/// Formed in single precision, as the original forms it.
#[must_use]
pub fn spacing_product(first: f64, second: f64) -> f64 {
    first * second
}
