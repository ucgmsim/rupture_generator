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
//! genslip does; the two agreed to 7.06e-08 relative — about an `f32` ulp — and
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
///
/// # Why the dip pass transposes
///
/// Neither FFTW as genslip calls it nor `rustfft` takes a stride, so a column has to
/// be made contiguous somehow. genslip gathers each column into scratch, transforms
/// it, and scatters it back — one strided read and one strided write per column,
/// every one of them a cache miss on a grid larger than L2.
///
/// This transposes the whole grid once, runs the columns as contiguous rows, and
/// transposes back. The tiled transpose touches each cache line once instead of once
/// per element, and it is **bit-identical**: every 1-D transform sees exactly the same
/// input sequence, so only the order of the memory traffic changes.
///
/// Measured, forward and inverse, best of nine, `--release`:
///
/// | grid | gathered | transposed | |
/// | ---: | ---: | ---: | ---: |
/// | 32×32 | 8.4 µs | 4.9 µs | 1.70× |
/// | 64×64 | 32.7 µs | 18.8 µs | 1.74× |
/// | 128×128 | 192 µs | 99 µs | 1.93× |
/// | 256×256 | 1.43 ms | 756 µs | 1.89× |
/// | 512×512 | 7.08 ms | 3.21 ms | 2.20× |
/// | 1024×256 | 7.07 ms | 3.20 ms | 2.21× |
/// | 256×1024 | 5.71 ms | 3.22 ms | 1.77× |
///
/// The win grows with the grid, which is what a cache effect looks like and what
/// distinguishes it from noise. `tests/timing.rs::fft_dip_pass` keeps both spellings
/// so the table can be re-run rather than trusted.
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

    // Down dip: transpose, run contiguous rows, transpose back.
    let mut scratch = vec![Complex32::default(); strike_count * dip_count];
    transpose_into(spectrum.as_slice(), &mut scratch, dip_count, strike_count);
    for strike in 0..strike_count {
        let start = strike * dip_count;
        fft.transform(&mut scratch[start..start + dip_count], direction);
    }
    transpose_into(&scratch, spectrum.as_mut_slice(), strike_count, dip_count);
}

/// Side of the square block the transpose works in, in elements.
///
/// Both the read and the write side of a block have to stay resident at once, so the
/// working set is `2 * TILE² * 8` bytes — 16 KiB at 32, comfortably inside a 32 KiB
/// L1. Larger tiles start evicting the block being read; smaller ones stop amortising
/// the cache line, which holds 4 `Complex32`.
const TILE: usize = 32;

/// Transpose `source` (`rows` × `columns`, row-major) into `target`, in blocks.
///
/// Blocked rather than a plain double loop: one of the two sides is strided whichever
/// way it is written, and walking a whole row of one against a whole column of the
/// other touches a fresh cache line on every element. Confining both to a tile means
/// each line is touched once and reused `TILE` times.
fn transpose_into(source: &[Complex32], target: &mut [Complex32], rows: usize, columns: usize) {
    debug_assert_eq!(source.len(), rows * columns);
    debug_assert_eq!(target.len(), rows * columns);

    for row_block in (0..rows).step_by(TILE) {
        for column_block in (0..columns).step_by(TILE) {
            for row in row_block..(row_block + TILE).min(rows) {
                for column in column_block..(column_block + TILE).min(columns) {
                    target[row + column * rows] = source[column + row * columns];
                }
            }
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
