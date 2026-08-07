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
//! * [`FftwFft`] calls FFTW, which is what genslip calls. It exists to be compared
//!   against the original and is the Stage 1 default.
//! * [`RustFft`] uses `rustfft`, needs no C library, and is where this is going.
//!
//! **The two are not bit-identical and cannot be.** Different algorithms round
//! differently, so swapping engines moves the last bits of every field. What the
//! tests establish is that both satisfy the same contract, and *how large* the
//! difference is — measured now, so the Stage 3 swap has a baseline to be judged
//! against rather than an argument to have afterwards.
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

#[cfg(feature = "fftw")]
mod fftw;
mod rust;

#[cfg(feature = "fftw")]
pub use fftw::FftwFft;
pub use rust::RustFft;

use num_complex::Complex32;

use crate::grid::Spectrum;

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
    fn transform(&mut self, buffer: &mut [Complex32], direction: Direction);
}

/// Transform a grid in place, along strike and then down dip.
///
/// Unnormalised — see [`scale`], which the callers apply afterwards.
///
/// The pass order matters for the last bits even though it does not matter
/// mathematically: each pass rounds, so doing dip first would give a different
/// answer. Strike first is what the original does.
pub fn transform_2d<F: Fft + ?Sized>(spectrum: &mut Spectrum, fft: &mut F, direction: Direction) {
    let strike_count = spectrum.strike_count();
    let dip_count = spectrum.dip_count();

    // Along strike: each row is already contiguous.
    for dip in 0..dip_count {
        let start = dip * strike_count;
        fft.transform(
            &mut spectrum.as_mut_slice()[start..start + strike_count],
            direction,
        );
    }

    // Down dip: each column is strided, so it is gathered into a scratch buffer and
    // scattered back. The original does the same.
    //
    // SIMPLIFY: `rustfft` has no strided API either, but the gather/scatter can be
    // replaced by transposing once, running contiguous rows, and transposing back --
    // which is cache-friendlier at realistic fault sizes. Measure before believing
    // that; the predecessor project has three separate records of a predicted win
    // measuring smaller, absent, or backwards.
    let mut column = vec![Complex32::default(); dip_count];
    for strike in 0..strike_count {
        for (dip, value) in column.iter_mut().enumerate() {
            *value = spectrum[(strike, dip)];
        }
        fft.transform(&mut column, direction);
        for (dip, value) in column.iter().enumerate() {
            spectrum[(strike, dip)] = *value;
        }
    }
}

/// Multiply every point by `factor`.
///
/// Separated from [`transform_2d`] deliberately, though the original fuses it into
/// the second pass's writeback. It costs nothing to separate: `factor * value`
/// rounded to `f32` is the same whether the value was stored first or not, because
/// either way it is one `f32` multiply of the same two `f32` operands. And it keeps
/// a genslip convention out of a function whose job is the transform.
///
/// The factor callers pass is the product of the sample spacings of the domain the
/// transform started in.
pub fn scale(spectrum: &mut Spectrum, factor: f32) {
    for value in spectrum.as_mut_slice() {
        *value *= factor;
    }
}

/// The normalisation genslip applies after a transform: the product of the source
/// domain's sample spacings.
///
/// Formed in single precision, as the original forms it.
#[must_use]
pub fn spacing_product(first: f32, second: f32) -> f32 {
    first * second
}
