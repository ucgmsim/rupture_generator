//! Drawing Gaussian random fields from a circulant embedding.
//!
//! Python computes the embedding's spectrum once per (chart, covariance), clamps it,
//! square-roots it and caches the result; this draws from it. One draw is complex
//! noise scaled by the amplitudes, an inverse 2-D FFT, and a crop to the fault. The
//! real and imaginary parts of the result are two independent fields with the
//! embedded covariance (Dietrich & Newsam 1997), so a draw yields two.
//!
//! > **Dietrich, C. R. & Newsam, G. N. (1997).** Fast and exact simulation of
//! > stationary Gaussian processes through circulant embedding of the covariance
//! > matrix. *SIAM Journal on Scientific Computing* **18**(4), 1088–1107.
//!
//! It is a kernel because the padded grid is large and the work is all memory
//! traffic: the noise is generated straight into the buffer the transform runs in,
//! with no intermediate arrays, rows and columns are transformed on every core, and
//! only the columns the crop keeps are transformed at all. The spectrum itself stays in Python: it needs
//! `scipy.special.kv` at fractional order, and it is computed once where this runs
//! several times per segment.

use std::sync::Arc;

use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rand_pcg::Pcg64;
use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};

use crate::counts::exact;

/// Rows of noise drawn from one generator.
///
/// Fixed rather than derived from the thread count, so the draw for a seed is the
/// same on any machine.
const ROWS_PER_CHUNK: usize = 32;

/// Columns gathered and transformed together in the column pass: enough contiguous
/// complex values per row (256 bytes) to use whole cache lines on the gather.
const COLUMNS_PER_BLOCK: usize = 16;

/// Error cases for the draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// The padded grid needs at least one cell on each axis.
    EmptyEmbedding { padded: (usize, usize) },
    /// The amplitudes must be the `(P_i/2 + 1, P_j/2 + 1)` quadrant of the padded grid.
    WrongQuadrant {
        padded: (usize, usize),
        got: (usize, usize),
    },
    /// The fault must fit inside the embedding.
    FaultTooLarge {
        cells: (usize, usize),
        padded: (usize, usize),
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::EmptyEmbedding { padded: (pi, pj) } => {
                write!(f, "a {pi}x{pj} embedding has no cells")
            }
            Self::WrongQuadrant {
                padded: (pi, pj),
                got: (gi, gj),
            } => write!(
                f,
                "a {pi}x{pj} embedding needs a {}x{} amplitude quadrant, not {gi}x{gj}",
                pi / 2 + 1,
                pj / 2 + 1
            ),
            Self::FaultTooLarge {
                cells: (ci, cj),
                padded: (pi, pj),
            } => write!(f, "a {ci}x{cj} fault does not fit in a {pi}x{pj} embedding"),
        }
    }
}

impl std::error::Error for Error {}

/// Two independent standard-normal fields from one embedding draw.
///
/// `amplitudes` is row-major over `quadrant`, the square-rooted, non-negative
/// eigenvalues for `i <= P_i/2, j <= P_j/2`; the full grid's amplitude at `(i, j)` is
/// the quadrant's at `(min(i, P_i - i), min(j, P_j - j))`. `cells` is the fault's
/// shape, which the padded grid's leading corner is cropped to.
///
/// The transform is an unscaled inverse FFT times `1/sqrt(n)`, so each field's
/// covariance is what the eigenvalues describe. Returns `(real, imaginary)`, each
/// row-major over `cells`. The same seed gives the same fields whatever the thread
/// count.
///
/// # Errors
///
/// [`Error`]: an empty embedding, a quadrant of the wrong shape, or a fault that does
/// not fit.
///
/// # Panics
///
/// If `amplitudes` is not `quadrant.0 * quadrant.1` long, or a worker thread panics.
pub fn draw(
    amplitudes: &[f64],
    quadrant: (usize, usize),
    padded: (usize, usize),
    cells: (usize, usize),
    seed: u64,
) -> Result<(Vec<f64>, Vec<f64>), Error> {
    let (padded_i, padded_j) = padded;
    if padded_i == 0 || padded_j == 0 {
        return Err(Error::EmptyEmbedding { padded });
    }
    if quadrant != (padded_i / 2 + 1, padded_j / 2 + 1) {
        return Err(Error::WrongQuadrant {
            padded,
            got: quadrant,
        });
    }
    if cells.0 > padded_i || cells.1 > padded_j {
        return Err(Error::FaultTooLarge { cells, padded });
    }
    assert_eq!(amplitudes.len(), quadrant.0 * quadrant.1);
    debug_assert!(amplitudes.iter().all(|&a| a >= 0.0));

    // One generator per chunk of rows, all drawn up front from the seed: the
    // assignment of chunks to threads then cannot change the noise.
    let chunks = padded_i.div_ceil(ROWS_PER_CHUNK);
    let mut master = Pcg64::seed_from_u64(seed);
    let generators: Vec<Pcg64> = (0..chunks).map(|_| Pcg64::from_rng(&mut master)).collect();

    let quadrant_j = quadrant.1;
    let fill = |chunk: usize, rows: &mut Vec<Complex64>| {
        let mut rng = generators[chunk].clone();
        let first = chunk * ROWS_PER_CHUNK;
        for i in first..(first + ROWS_PER_CHUNK).min(padded_i) {
            let mirrored_i = i.min(padded_i - i);
            let amplitude_row = &amplitudes[mirrored_i * quadrant_j..][..quadrant_j];
            rows.extend((0..padded_j).map(|j| {
                let amplitude = amplitude_row[j.min(padded_j - j)];
                // Unit variance in each part, so each part of the result is a
                // standard-normal field rather than half of one.
                let real: f64 = StandardNormal.sample(&mut rng);
                let imaginary: f64 = StandardNormal.sample(&mut rng);
                Complex64::new(amplitude * real, amplitude * imaginary)
            }));
        }
    };

    let cropped = transform(padded, cells, fill);

    let scale = exact(padded_i * padded_j).sqrt().recip();
    Ok(row_major_parts(&cropped, cells, scale))
}

/// Fill the padded grid chunk by chunk, inverse-transform it, and return the `cells`
/// corner, unscaled, in blocks of [`COLUMNS_PER_BLOCK`] column-major columns.
///
/// `fill(chunk, rows)` appends rows `chunk * ROWS_PER_CHUNK..` of the spectrum to an
/// empty buffer; the same thread transforms them straight after. Each chunk is its
/// own allocation, so no thread waits on zeroing or faulting in the whole grid. The
/// column pass then transforms only the `cells.1` columns the crop keeps.
fn transform(
    padded: (usize, usize),
    cells: (usize, usize),
    fill: impl Fn(usize, &mut Vec<Complex64>) + Sync,
) -> Vec<Vec<Complex64>> {
    let (padded_i, padded_j) = padded;
    let (cells_i, cells_j) = cells;
    let mut planner = FftPlanner::new();
    let along_rows: Arc<dyn Fft<f64>> = planner.plan_fft_inverse(padded_j);
    let down_columns: Arc<dyn Fft<f64>> = planner.plan_fft_inverse(padded_i);

    let mut row_chunks: Vec<Vec<Complex64>> = vec![Vec::new(); padded_i.div_ceil(ROWS_PER_CHUNK)];
    spread(
        row_chunks.iter_mut().enumerate().collect(),
        || vec![Complex64::default(); along_rows.get_inplace_scratch_len()],
        |scratch, (chunk, rows)| {
            let count = ROWS_PER_CHUNK.min(padded_i - chunk * ROWS_PER_CHUNK);
            rows.reserve_exact(count * padded_j);
            fill(chunk, rows);
            assert_eq!(
                rows.len(),
                count * padded_j,
                "fill wrote the wrong number of cells"
            );
            along_rows.process_with_scratch(rows, scratch);
        },
    );

    let row =
        |i: usize| &row_chunks[i / ROWS_PER_CHUNK][(i % ROWS_PER_CHUNK) * padded_j..][..padded_j];
    let mut blocks: Vec<Vec<Complex64>> = vec![Vec::new(); cells_j.div_ceil(COLUMNS_PER_BLOCK)];
    if cells_i == 0 {
        return blocks;
    }
    spread(
        blocks.iter_mut().enumerate().collect(),
        || {
            (
                vec![Complex64::default(); COLUMNS_PER_BLOCK * padded_i],
                vec![Complex64::default(); down_columns.get_inplace_scratch_len()],
            )
        },
        |(columns, scratch), (block, out)| {
            let first = block * COLUMNS_PER_BLOCK;
            let width = COLUMNS_PER_BLOCK.min(cells_j - first);
            let columns = &mut columns[..width * padded_i];
            for i in 0..padded_i {
                for (b, &value) in row(i)[first..first + width].iter().enumerate() {
                    columns[b * padded_i + i] = value;
                }
            }
            down_columns.process_with_scratch(columns, scratch);
            out.reserve_exact(width * cells_i);
            for column in columns.chunks_exact(padded_i) {
                out.extend_from_slice(&column[..cells_i]);
            }
        },
    );
    blocks
}

/// The crop's column blocks as two scaled, row-major real fields.
fn row_major_parts(
    blocks: &[Vec<Complex64>],
    cells: (usize, usize),
    scale: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (cells_i, cells_j) = cells;
    let mut real = vec![0.0; cells_i * cells_j];
    let mut imaginary = vec![0.0; cells_i * cells_j];
    if real.is_empty() {
        return (real, imaginary);
    }
    let rows: Vec<_> = real
        .chunks_mut(ROWS_PER_CHUNK * cells_j)
        .zip(imaginary.chunks_mut(ROWS_PER_CHUNK * cells_j))
        .enumerate()
        .collect();
    spread(
        rows,
        || (),
        |(), (chunk, (real, imaginary))| {
            let first = chunk * ROWS_PER_CHUNK;
            for (offset, (re_row, im_row)) in real
                .chunks_exact_mut(cells_j)
                .zip(imaginary.chunks_exact_mut(cells_j))
                .enumerate()
            {
                let i = first + offset;
                for (j, (re, im)) in re_row.iter_mut().zip(im_row).enumerate() {
                    let value =
                        blocks[j / COLUMNS_PER_BLOCK][(j % COLUMNS_PER_BLOCK) * cells_i + i];
                    *re = value.re * scale;
                    *im = value.im * scale;
                }
            }
        },
    );
    (real, imaginary)
}

/// Run `work` over `items` on every core, each thread with its own `init()` state.
///
/// Items are dealt round-robin; each is independent, so the result does not depend
/// on how many threads there are.
fn spread<T: Send, S>(items: Vec<T>, init: impl Fn() -> S + Sync, work: impl Fn(&mut S, T) + Sync) {
    let threads = std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .min(items.len());
    if threads <= 1 {
        let mut state = init();
        for item in items {
            work(&mut state, item);
        }
        return;
    }
    let mut shares: Vec<Vec<T>> = (0..threads).map(|_| Vec::new()).collect();
    for (index, item) in items.into_iter().enumerate() {
        shares[index % threads].push(item);
    }
    let (init, work) = (&init, &work);
    std::thread::scope(|scope| {
        for share in shares {
            scope.spawn(move || {
                let mut state = init();
                for item in share {
                    work(&mut state, item);
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(padded: (usize, usize)) -> (Vec<f64>, (usize, usize)) {
        let quadrant = (padded.0 / 2 + 1, padded.1 / 2 + 1);
        (vec![1.0; quadrant.0 * quadrant.1], quadrant)
    }

    /// A flat spectrum draws white noise: unit variance in both fields, and the two
    /// uncorrelated with each other.
    #[test]
    fn a_flat_spectrum_gives_two_uncorrelated_unit_variance_fields() {
        let padded = (96, 80);
        let cells = (64, 50);
        let (amplitudes, quadrant) = flat(padded);
        let (real, imaginary) = draw(&amplitudes, quadrant, padded, cells, 7).unwrap();
        let n = exact(real.len());

        let mean = |field: &[f64]| field.iter().sum::<f64>() / n;
        let (mean_re, mean_im) = (mean(&real), mean(&imaginary));
        let variance =
            |field: &[f64], mean: f64| field.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let (variance_re, variance_im) = (variance(&real, mean_re), variance(&imaginary, mean_im));
        let covariance = real
            .iter()
            .zip(&imaginary)
            .map(|(a, b)| (a - mean_re) * (b - mean_im))
            .sum::<f64>()
            / n;
        let correlation = covariance / (variance_re * variance_im).sqrt();

        assert!(
            (variance_re - 1.0).abs() < 0.1,
            "real variance {variance_re}"
        );
        assert!(
            (variance_im - 1.0).abs() < 0.1,
            "imaginary variance {variance_im}"
        );
        assert!(correlation.abs() < 0.05, "correlation {correlation}");
    }

    /// The same seed gives the same fields; a different one does not.
    #[test]
    fn a_draw_is_determined_by_its_seed() {
        let padded = (70, 45);
        let cells = (40, 30);
        let (amplitudes, quadrant) = flat(padded);
        let first = draw(&amplitudes, quadrant, padded, cells, 11).unwrap();
        let again = draw(&amplitudes, quadrant, padded, cells, 11).unwrap();
        let other = draw(&amplitudes, quadrant, padded, cells, 12).unwrap();
        assert_eq!(first, again);
        assert_ne!(first.0, other.0);
    }

    /// Shapes are checked against the padded grid.
    #[test]
    fn a_mismatched_quadrant_or_oversized_fault_is_refused() {
        let (amplitudes, quadrant) = flat((8, 8));
        assert!(matches!(
            draw(&amplitudes, quadrant, (8, 10), (4, 4), 0),
            Err(Error::WrongQuadrant { .. })
        ));
        assert!(matches!(
            draw(&amplitudes, quadrant, (8, 8), (9, 4), 0),
            Err(Error::FaultTooLarge { .. })
        ));
    }

    /// The transform inverts the forward one, which is what "unscaled" has to mean,
    /// and the crop keeps the leading corner.
    #[test]
    fn the_inverse_transform_undoes_a_forward_transform() {
        let (rows, columns) = (40, 37);
        let cells = (25, 20);
        let original: Vec<Complex64> = (0..rows * columns)
            .map(|k| Complex64::new(exact(k) % 7.0, exact(k) % 3.0))
            .collect();

        let mut spectrum = original.clone();
        let mut planner = FftPlanner::new();
        planner.plan_fft_forward(columns).process(&mut spectrum);
        let down_columns = planner.plan_fft_forward(rows);
        let mut column = vec![Complex64::default(); rows];
        for j in 0..columns {
            for (i, cell) in column.iter_mut().enumerate() {
                *cell = spectrum[i * columns + j];
            }
            down_columns.process(&mut column);
            for (i, cell) in column.iter().enumerate() {
                spectrum[i * columns + j] = *cell;
            }
        }

        let blocks = transform((rows, columns), cells, |chunk, chunk_rows| {
            let start = chunk * ROWS_PER_CHUNK * columns;
            let end = (start + ROWS_PER_CHUNK * columns).min(spectrum.len());
            chunk_rows.extend_from_slice(&spectrum[start..end]);
        });
        let n = exact(rows * columns);
        for i in 0..cells.0 {
            for j in 0..cells.1 {
                let round_tripped =
                    blocks[j / COLUMNS_PER_BLOCK][(j % COLUMNS_PER_BLOCK) * cells.0 + i] / n;
                assert!((round_tripped - original[i * columns + j]).norm() < 1e-9);
            }
        }
    }
}
