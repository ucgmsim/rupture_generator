//! Drawing a Gaussian random field whose correlation length varies with depth.
//!
//! This is *the* field sampler. A field whose correlation structure is the same
//! everywhere is the special case where the depth profile is flat, and
//! [`field::von_karman_draw`](crate::field::von_karman_draw) is its exact
//! factorisation: a covariance that is stationary down dip is Toeplitz, a Toeplitz
//! operator embeds in a circulant one, and a circulant operator's eigenvectors are
//! the DFT. The two paths are one model, and Python picks between them on a property
//! of the covariance rather than on which sampler a caller asked for.
//!
//! When the profile is *not* flat there is no such shortcut down dip. The field is
//! still stationary **along strike** — the correlation lengths depend on depth only —
//! so the covariance is block-circulant in the strike index and the strike axis still
//! diagonalises by DFT. What is left at each wavenumber is a dense `ndip x ndip`
//! spectral density across depths, and that has to be factorised.
//!
//! The factorisation is a Cholesky where one exists and a clipped eigendecomposition
//! where it does not. Cholesky is the cheaper of the two by about a factor of three
//! and is what a covariance matrix admits; the fallback exists because the NORTA
//! pre-correction Python applies before calling here is monotone but not obliged to
//! preserve positive definiteness, and a field still has to come out.

use faer::linalg::solvers::{Llt, SelfAdjointEigen};
use faer::{Mat, Side};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rand_pcg::Pcg64;
use rayon::prelude::*;
use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// What a factorisation of one covariance came to.
pub struct Factorisation {
    /// The per-wavenumber square roots, `mx` blocks of `ndip x ndip`, row-major.
    pub factors: Vec<f64>,
    /// How much variance the factorisation had to drop, over how much it kept.
    ///
    /// A covariance matrix has no negative eigenvalues; one that does is not a
    /// covariance, and the negative directions cannot be sampled. This is the
    /// magnitude of what was dropped as a fraction of what remains, summed over every
    /// wavenumber — so it says how much less variance the field has than was asked
    /// for, and is zero exactly when the covariance was positive definite throughout.
    ///
    /// It is a ratio of *sums* rather than of extremes on purpose. The spectral
    /// density decays to numerically zero at high wavenumbers, so the worst
    /// eigenvalue over the largest **within one block** is round-off over round-off
    /// there, and reports an alarming number about a block that carries no variance.
    pub variance_deficit: f64,
}

/// Errors this module reports back to Python by name.
#[derive(Debug)]
pub enum Error {
    /// The covariance blocks are not square, or the draw's grid does not fit.
    Shape(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(message) => write!(formatter, "{message}"),
        }
    }
}

impl std::error::Error for Error {}

/// Transform a lag-domain covariance along strike, then factorise each wavenumber.
///
/// `covariance` is `C[lag, i, j]`, `wavenumbers` blocks of `depths x depths`,
/// row-major. It must be **even in the lag index** — `C[p] == C[mx - p]`, which is
/// what makes it a circulant embedding — and symmetric in `(i, j)`. Both hold by
/// construction for a covariance built on wrapped lags, and the transform is real
/// because of the first.
///
/// # Errors
///
/// If the blocks are not square, or the array is not `wavenumbers * depths * depths`.
pub fn factorise(
    covariance: &[f64],
    wavenumbers: usize,
    depths: usize,
) -> Result<Factorisation, Error> {
    if covariance.len() != wavenumbers * depths * depths {
        return Err(Error::Shape(format!(
            "a {wavenumbers} x {depths} x {depths} covariance is {} values, got {}",
            wavenumbers * depths * depths,
            covariance.len()
        )));
    }

    let mut spectra = transform_along_strike(covariance, wavenumbers, depths);
    let block = depths * depths;
    // Each wavenumber is independent of every other, and faer is left sequential
    // inside one block so the two levels of parallelism do not oversubscribe.
    let (dropped, kept) = spectra
        .par_chunks_mut(block)
        .map(|spectrum| factorise_block(spectrum, depths))
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

    Ok(Factorisation {
        factors: spectra,
        variance_deficit: if kept > 0.0 { dropped / kept } else { 0.0 },
    })
}

/// The DFT along the lag axis, giving one real spectral density matrix per
/// wavenumber.
///
/// The covariance is symmetric in `(i, j)`, so only the lower triangle is transformed
/// and the result mirrored: that is a little under half the transforms, and on a
/// large chart the transform stage is `depths^2` strided gathers rather than anything
/// cache-friendly.
fn transform_along_strike(covariance: &[f64], wavenumbers: usize, depths: usize) -> Vec<f64> {
    let mut planner = FftPlanner::new();
    let transform: Arc<dyn Fft<f64>> = planner.plan_fft_forward(wavenumbers);
    let block = depths * depths;

    let mut spectra = vec![0.0_f64; wavenumbers * block];
    // One (i, j) pair at a time, so the scratch buffer is `wavenumbers` long rather
    // than the whole array again.
    let pairs: Vec<(usize, usize)> = (0..depths)
        .flat_map(|i| (0..=i).map(move |j| (i, j)))
        .collect();

    let columns: Vec<(usize, usize, Vec<f64>)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mut column: Vec<Complex64> = (0..wavenumbers)
                .map(|lag| Complex64::new(covariance[lag * block + i * depths + j], 0.0))
                .collect();
            transform.process(&mut column);
            (i, j, column.into_iter().map(|value| value.re).collect())
        })
        .collect();

    for (i, j, real) in columns {
        for (lag, &value) in real.iter().enumerate() {
            spectra[lag * block + i * depths + j] = value;
            spectra[lag * block + j * depths + i] = value;
        }
    }
    spectra
}

/// Replace one symmetric block with a square root of it, in place.
///
/// Returns `(dropped, kept)`: the summed magnitude of the negative eigenvalues, which
/// cannot be sampled, and the summed positive ones, which can. A Cholesky succeeding
/// *is* the proof that nothing was dropped, and the eigenvalues of a positive
/// semi-definite block sum to its trace, so that path needs no eigendecomposition to
/// answer.
fn factorise_block(spectrum: &mut [f64], depths: usize) -> (f64, f64) {
    let trace: f64 = (0..depths).map(|i| spectrum[i * depths + i]).sum();
    let matrix = Mat::from_fn(depths, depths, |i, j| spectrum[i * depths + j]);

    if let Ok(cholesky) = Llt::new(matrix.as_ref(), Side::Lower) {
        let lower = cholesky.L();
        for i in 0..depths {
            for j in 0..depths {
                // `L` is lower triangular and faer leaves the upper part unspecified.
                spectrum[i * depths + j] = if j <= i { lower[(i, j)] } else { 0.0 };
            }
        }
        return (0.0, trace.max(0.0));
    }

    let Ok(eigen) = SelfAdjointEigen::new(matrix.as_ref(), Side::Lower) else {
        // An eigendecomposition that does not converge leaves no field to draw, so
        // the block becomes zero and every bit of its variance counts as dropped.
        spectrum.fill(0.0);
        return (trace.abs(), 0.0);
    };
    let values = eigen.S();
    let vectors = eigen.U();

    let mut dropped = 0.0;
    let mut kept = 0.0;
    for k in 0..depths {
        let value = values[k];
        if value < 0.0 {
            dropped -= value;
        } else {
            kept += value;
        }
    }

    // `U diag(sqrt(max(lambda, 0)))`, which is a square root of the covariance's
    // nearest positive-semidefinite neighbour in the Frobenius norm.
    for i in 0..depths {
        for k in 0..depths {
            spectrum[i * depths + k] = vectors[(i, k)] * values[k].max(0.0).sqrt();
        }
    }

    (dropped, kept)
}

/// One field on the fault, drawn from per-wavenumber square roots.
///
/// `factors` is what [`factorise`] returned. `strike_cells` is how much of the padded
/// strike axis the fault occupies; the rest is the margin the circulant embedding
/// needed and is dropped. Returns `depths x strike_cells`, row-major — `i` down dip
/// and `j` along strike, the order every field in this package is in.
///
/// # Errors
///
/// If the fault does not fit inside the embedding, or the factors are not
/// `wavenumbers * depths * depths`.
pub fn draw(
    factors: &[f64],
    wavenumbers: usize,
    depths: usize,
    strike_cells: usize,
    seed: u64,
) -> Result<Vec<f64>, Error> {
    if factors.len() != wavenumbers * depths * depths {
        return Err(Error::Shape(format!(
            "a {wavenumbers} x {depths} x {depths} factorisation is {} values, got {}",
            wavenumbers * depths * depths,
            factors.len()
        )));
    }
    if strike_cells > wavenumbers {
        return Err(Error::Shape(format!(
            "a fault {strike_cells} cells along strike does not fit in a {wavenumbers}-wide \
             embedding"
        )));
    }

    // Drawn serially and in one pass, so the field a seed gives does not depend on
    // how many threads happen to be free.
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut noise = vec![Complex64::new(0.0, 0.0); wavenumbers * depths];
    for cell in &mut noise {
        let real: f64 = StandardNormal.sample(&mut rng);
        let imaginary: f64 = StandardNormal.sample(&mut rng);
        *cell = Complex64::new(real, imaginary);
    }

    let block = depths * depths;
    noise
        .par_chunks_mut(depths)
        .enumerate()
        .for_each(|(wavenumber, slice)| {
            let factor = &factors[wavenumber * block..(wavenumber + 1) * block];
            let mut product = vec![Complex64::new(0.0, 0.0); depths];
            for i in 0..depths {
                let row = &factor[i * depths..(i + 1) * depths];
                let mut sum = Complex64::new(0.0, 0.0);
                for (k, &entry) in row.iter().enumerate() {
                    sum += slice[k] * entry;
                }
                product[i] = sum;
            }
            slice.copy_from_slice(&product);
        });

    let mut planner = FftPlanner::new();
    let transform: Arc<dyn Fft<f64>> = planner.plan_fft_forward(wavenumbers);
    // The cast is exact for any embedding that fits in memory: f64 carries integers
    // to 2^53, and the Python side caps the padded grid far below that.
    #[allow(clippy::cast_precision_loss, reason = "cell count is far under 2^53")]
    let scale = (wavenumbers as f64).sqrt().recip();

    let rows: Vec<Vec<f64>> = (0..depths)
        .into_par_iter()
        .map(|i| {
            let mut column: Vec<Complex64> = (0..wavenumbers)
                .map(|wavenumber| noise[wavenumber * depths + i])
                .collect();
            transform.process(&mut column);
            column[..strike_cells]
                .iter()
                .map(|value| value.re * scale)
                .collect()
        })
        .collect();

    let mut field = vec![0.0_f64; depths * strike_cells];
    for (i, row) in rows.iter().enumerate() {
        field[i * strike_cells..(i + 1) * strike_cells].copy_from_slice(row);
    }
    Ok(field)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent white noise per depth: the covariance is the identity at every
    /// lag-zero block and zero elsewhere, so every wavenumber's spectrum is `I`.
    #[test]
    fn a_white_covariance_factorises_to_the_identity() {
        let (wavenumbers, depths) = (8, 4);
        let mut covariance = vec![0.0_f64; wavenumbers * depths * depths];
        for i in 0..depths {
            covariance[i * depths + i] = 1.0;
        }

        let result = factorise(&covariance, wavenumbers, depths).unwrap();
        assert!((result.variance_deficit - 0.0).abs() < 1e-12);
        for wavenumber in 0..wavenumbers {
            for i in 0..depths {
                for j in 0..depths {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    let got = result.factors[wavenumber * depths * depths + i * depths + j];
                    assert!((got - expected).abs() < 1e-12, "at {wavenumber} {i} {j}: {got}");
                }
            }
        }
    }

    /// The factor is a square root: `B B^T` recovers the spectrum it came from.
    #[test]
    fn the_factor_multiplied_by_its_transpose_is_the_spectrum() {
        let (wavenumbers, depths) = (16, 6);
        // A covariance that decays in both lag and depth separation, so the blocks
        // are neither diagonal nor equal to each other.
        let mut covariance = vec![0.0_f64; wavenumbers * depths * depths];
        for lag in 0..wavenumbers {
            let wrapped = lag.min(wavenumbers - lag);
            #[allow(clippy::cast_precision_loss, reason = "small test indices")]
            for i in 0..depths {
                for j in 0..depths {
                    let dz = (i as f64 - j as f64).abs();
                    let dx = wrapped as f64;
                    covariance[lag * depths * depths + i * depths + j] =
                        (-(dx / 3.0) - (dz / 2.0)).exp();
                }
            }
        }

        let spectra = transform_along_strike(&covariance, wavenumbers, depths);
        let result = factorise(&covariance, wavenumbers, depths).unwrap();
        assert!(result.variance_deficit < 1e-10);

        let block = depths * depths;
        for wavenumber in 0..wavenumbers {
            let factor = &result.factors[wavenumber * block..(wavenumber + 1) * block];
            for i in 0..depths {
                for j in 0..depths {
                    let mut sum = 0.0;
                    for k in 0..depths {
                        sum += factor[i * depths + k] * factor[j * depths + k];
                    }
                    let expected = spectra[wavenumber * block + i * depths + j];
                    assert!(
                        (sum - expected).abs() < 1e-9,
                        "wavenumber {wavenumber} ({i},{j}): {sum} vs {expected}"
                    );
                }
            }
        }
    }

    /// A covariance that is not positive definite is reported, not silently drawn.
    #[test]
    fn an_indefinite_covariance_reports_the_variance_it_dropped() {
        let (wavenumbers, depths) = (4, 3);
        let mut covariance = vec![0.0_f64; wavenumbers * depths * depths];
        // Constant in lag, so every wavenumber but zero has a zero spectrum, and at
        // zero the block is this one, which has a negative eigenvalue.
        for i in 0..depths {
            for j in 0..depths {
                covariance[i * depths + j] = if i == j { 1.0 } else { -0.9 };
            }
        }
        for lag in 1..wavenumbers {
            for entry in 0..depths * depths {
                covariance[lag * depths * depths + entry] = covariance[entry];
            }
        }

        let result = factorise(&covariance, wavenumbers, depths).unwrap();
        assert!(
            result.variance_deficit > 1e-6,
            "deficit was {}",
            result.variance_deficit
        );
    }

    /// The draw is reproducible from its seed and has the shape it promised.
    #[test]
    fn a_draw_is_the_faults_shape_and_repeats_from_its_seed() {
        let (wavenumbers, depths, strike_cells) = (32, 5, 20);
        let mut covariance = vec![0.0_f64; wavenumbers * depths * depths];
        for lag in 0..wavenumbers {
            let wrapped = lag.min(wavenumbers - lag);
            #[allow(clippy::cast_precision_loss, reason = "small test indices")]
            for i in 0..depths {
                covariance[lag * depths * depths + i * depths + i] =
                    (-(wrapped as f64) / 4.0).exp();
            }
        }
        let result = factorise(&covariance, wavenumbers, depths).unwrap();

        let first = draw(&result.factors, wavenumbers, depths, strike_cells, 7).unwrap();
        let again = draw(&result.factors, wavenumbers, depths, strike_cells, 7).unwrap();
        let other = draw(&result.factors, wavenumbers, depths, strike_cells, 8).unwrap();

        assert_eq!(first.len(), depths * strike_cells);
        assert_eq!(first, again);
        assert_ne!(first, other);
    }

    /// A fault wider than its embedding is refused rather than wrapped.
    #[test]
    fn a_fault_wider_than_its_embedding_is_refused() {
        let (wavenumbers, depths) = (8, 2);
        let factors = vec![0.0_f64; wavenumbers * depths * depths];
        assert!(draw(&factors, wavenumbers, depths, wavenumbers + 1, 1).is_err());
    }
}
