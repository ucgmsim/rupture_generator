//! Drawing a Gaussian random field from a circulant embedding.
//!
//! The Python side computes the embedding's eigenvalues once per (chart, covariance)
//! and caches them; what repeats is the *draw*, four times per segment — slip, rise
//! time, rake and the onset perturbation. This is that draw.
//!
//! # Why it is worth a kernel
//!
//! The numpy spelling allocates about six arrays the size of the padded grid: two
//! standard-normal draws, their complex combination, the product with the square-rooted
//! eigenvalues, the inverse transform's output, and the crop. On a 400x2400 embedding
//! each is 15 MB complex, so a draw moves ~90 MB through memory to produce a 0.5 MB
//! field. Here the noise is generated straight into the transform's own buffer, already
//! scaled, so one allocation does the work of five.
//!
//! # What it does not do
//!
//! It does not compute the eigenvalues. That needs a modified Bessel function of the
//! second kind at fractional order, and `scipy.special.kv` is a well-tested Cephes
//! implementation that no Rust crate matches for provenance. It is also computed once
//! and cached, where this runs four times, so the repeated cost is here and the risky
//! numerics stay in Python.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rand_pcg::Pcg64;
use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

/// One field on the fault, drawn from an embedding's eigenvalues.
///
/// `eigenvalues` is the circulant embedding's spectrum on the padded grid, real and
/// non-negative — `numpy.fft.fft2` of the wrapped covariance, which the caller has
/// already checked. `cell_counts` is the fault's own shape, which the padded grid's
/// leading corner is cropped to.
///
/// The transform is the inverse of the same convention numpy uses, scaled by
/// `sqrt(n)`, so the field's covariance is what the eigenvalues describe rather than
/// that divided by the grid.
///
/// # Errors
///
/// If the fault does not fit inside the embedding, or an eigenvalue is negative — the
/// second is the caller's check, repeated here because a square root of a negative
/// number is a NaN that would reach a rupture file without anything raising.
#[pyfunction]
pub fn von_karman_draw<'py>(
    python: Python<'py>,
    eigenvalues: PyReadonlyArray2<'py, f64>,
    cell_counts: (usize, usize),
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let eigenvalues = eigenvalues.as_array();
    let (padded_i, padded_j) = eigenvalues.dim();
    let (cells_i, cells_j) = cell_counts;

    if cells_i > padded_i || cells_j > padded_j {
        return Err(PyValueError::new_err(format!(
            "a {cells_i}x{cells_j} fault does not fit in a {padded_i}x{padded_j} embedding"
        )));
    }

    let Some(eigenvalues) = eigenvalues.as_slice() else {
        return Err(PyValueError::new_err(
            "the eigenvalues have to be C-contiguous",
        ));
    };

    // One buffer. The noise is drawn into it already scaled, so the two standard-normal
    // arrays, their complex combination and the product never exist.
    let mut buffer = vec![Complex64::new(0.0, 0.0); padded_i * padded_j];
    let mut rng = Pcg64::seed_from_u64(seed);
    for (cell, &eigenvalue) in buffer.iter_mut().zip(eigenvalues) {
        if eigenvalue < 0.0 {
            return Err(PyValueError::new_err(format!(
                "the embedding has a negative eigenvalue ({eigenvalue:e}), so it is not \
                 a covariance matrix"
            )));
        }
        let amplitude = eigenvalue.sqrt();
        // Unit variance in *each* part rather than jointly: only the real part of the
        // inverse transform is kept, and a standard complex draw would halve its
        // variance. Mirrors the Python.
        let real: f64 = StandardNormal.sample(&mut rng);
        let imaginary: f64 = StandardNormal.sample(&mut rng);
        *cell = Complex64::new(amplitude * real, amplitude * imaginary);
    }

    inverse_transform_2d(&mut buffer, padded_i, padded_j);

    // numpy's `ifft2` divides by n; rustfft does not scale at all. The field wants
    // `sqrt(n) * ifft2`, which is `1/sqrt(n)` times an unscaled inverse.
    let scale = ((padded_i * padded_j) as f64).sqrt().recip();

    let mut field = Array2::<f64>::zeros((cells_i, cells_j));
    for (i, mut row) in field.rows_mut().into_iter().enumerate() {
        let source = &buffer[i * padded_j..i * padded_j + cells_j];
        for (out, cell) in row.iter_mut().zip(source) {
            *out = cell.re * scale;
        }
    }
    Ok(field.into_pyarray(python))
}

/// An unscaled 2-D inverse FFT, in place, row-major.
///
/// Rows then columns. The column pass gathers into a scratch buffer rather than
/// transposing the whole grid: a transpose is a second allocation the size of the
/// field, and this is the allocation the kernel exists to avoid.
fn inverse_transform_2d(buffer: &mut [Complex64], rows: usize, columns: usize) {
    let mut planner = FftPlanner::new();

    let along_rows = planner.plan_fft_inverse(columns);
    along_rows.process(buffer);

    let down_columns = planner.plan_fft_inverse(rows);
    let mut column = vec![Complex64::new(0.0, 0.0); rows];
    for j in 0..columns {
        for (i, cell) in column.iter_mut().enumerate() {
            *cell = buffer[i * columns + j];
        }
        down_columns.process(&mut column);
        for (i, cell) in column.iter().enumerate() {
            buffer[i * columns + j] = *cell;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat spectrum draws white noise, whose variance is the eigenvalue.
    ///
    /// The one case where the answer is known without a transform: with every
    /// eigenvalue equal the field is uncorrelated, and its variance is that value.
    #[test]
    fn a_flat_spectrum_gives_white_noise_of_the_right_variance() {
        let (rows, columns) = (64, 64);
        let eigenvalues = vec![1.0_f64; rows * columns];
        let mut buffer = vec![Complex64::new(0.0, 0.0); rows * columns];
        let mut rng = Pcg64::seed_from_u64(7);
        for cell in &mut buffer {
            let real: f64 = StandardNormal.sample(&mut rng);
            let imaginary: f64 = StandardNormal.sample(&mut rng);
            *cell = Complex64::new(real, imaginary);
        }
        assert_eq!(eigenvalues.len(), buffer.len());

        inverse_transform_2d(&mut buffer, rows, columns);
        let scale = ((rows * columns) as f64).sqrt().recip();
        let variance = buffer
            .iter()
            .map(|cell| (cell.re * scale).powi(2))
            .sum::<f64>()
            / (rows * columns) as f64;

        assert!(
            (variance - 1.0).abs() < 0.1,
            "white noise variance was {variance}"
        );
    }

    /// The transform inverts the forward one, which is what "unscaled" has to mean.
    #[test]
    fn the_inverse_transform_undoes_a_forward_transform() {
        let (rows, columns) = (12, 20);
        let original: Vec<Complex64> = (0..rows * columns)
            .map(|k| Complex64::new(k as f64 % 7.0, k as f64 % 3.0))
            .collect();

        let mut buffer = original.clone();
        let mut planner = FftPlanner::new();
        let along_rows = planner.plan_fft_forward(columns);
        along_rows.process(&mut buffer);
        let down_columns = planner.plan_fft_forward(rows);
        let mut column = vec![Complex64::new(0.0, 0.0); rows];
        for j in 0..columns {
            for (i, cell) in column.iter_mut().enumerate() {
                *cell = buffer[i * columns + j];
            }
            down_columns.process(&mut column);
            for (i, cell) in column.iter().enumerate() {
                buffer[i * columns + j] = *cell;
            }
        }

        inverse_transform_2d(&mut buffer, rows, columns);
        let n = (rows * columns) as f64;
        for (round_tripped, expected) in buffer.iter().zip(&original) {
            assert!((round_tripped / n - expected).norm() < 1e-9);
        }
    }
}
