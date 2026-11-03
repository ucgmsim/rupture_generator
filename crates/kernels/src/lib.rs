//! The rupture kernels, stateless and array-in/array-out.
//!
//! `rupture_generator._kernels` exposes `eikonal_solve` ([`eikonal::solve`]),
//! `synthesise_pulses` ([`pulse::synthesise_pulses`]), `circulant_draw`
//! ([`field::draw`]), and the class `JumpSearch` ([`jump::Search`]) to Python. Parameters arrive as scalars and arrays, and Python
//! sets every default. The maths is in [`eikonal`], [`pulse`], [`field`] and [`jump`] over
//! plain slices, where `tests/` can generate inputs for it. This file only marshals
//! numpy arrays in and out, and releases the GIL around each computation.

// PyO3's `#[pyfunction]` extraction hands wrappers owned values, and the lint is
// right in general and wrong at every site in this file.
#![expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 argument extraction requires owned values"
)]
// Every numeric cast here is a grid index, sample index or count to `f64`, or a
// duration in samples back to `usize`. Indices and counts are many orders below the
// 2^53 where `as f64` stops being exact, and `f64 as usize` saturates with NaN to 0,
// which turns a degenerate duration into an empty pulse rather than garbage.
#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "indices and counts are far below 2^53, and float-to-int saturates"
)]

pub mod eikonal;
pub mod field;
pub mod jump;
pub mod pulse;

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

fn value_error<T, E: std::error::Error>(error: E) -> PyResult<T> {
    Err(PyValueError::new_err(error.to_string()))
}

/// The CSR pair `synthesise_pulses` returns: row offsets, then the flat samples.
type PyCsr<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>);

/// The pair of fields `circulant_draw` returns: real part, then imaginary.
type PyFieldPair<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>);

/// The pairs `JumpSearch.pairs_below` returns: parent cells, then child cells.
type PyPairs<'py> = (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>);

/// First-arrival times over a fault chart, by factored fast sweeping.
///
/// `slowness` is 2-D in s/km, `i` down-dip and `j` along-strike; `spacing_km` is
/// `(d_i, d_j)`; `seeds` is a list of `(i, j, t0_seconds)`, the points the front leaves
/// at known times: one triple for a hypocentre, and several for a fault triggered
/// along an edge. Returns travel times in seconds, same shape as
/// `slowness`. Exact on uniform media, first-order convergent on smooth ones;
/// `crates/kernels/src/eikonal.rs` has the papers.
#[pyfunction]
fn eikonal_solve<'py>(
    py: Python<'py>,
    slowness: PyReadonlyArray2<'py, f64>,
    spacing_km: (f64, f64),
    seeds: Vec<(usize, usize, f64)>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let extent = (slowness.shape()[0], slowness.shape()[1]);
    let cells = slowness.as_slice()?;
    let seeds: Vec<eikonal::Seed> = seeds
        .into_iter()
        .map(|(i, j, t0_s)| eikonal::Seed { i, j, t0_s })
        .collect();

    let times = py
        .detach(|| eikonal::solve(cells, extent, spacing_km, &seeds))
        .or_else(value_error)?;
    let times =
        Array2::from_shape_vec(extent, times).expect("the solver returns one time per input cell");
    Ok(times.into_pyarray(py))
}

/// Slip-rate pulses for every subfault, as CSR rows.
///
/// `slip_m` (metres) and `rise_time_s` are flat, one entry per subfault. `beta`, the
/// per-subfault rising fraction in `(0, 0.5]`, selects the shape: given, the
/// Liu-Archuleta-Hartzell piecewise sinusoid; absent, a single-sample impulse. Returns `(offsets, samples)`:
/// subfault `k`'s pulse is `samples[offsets[k]:offsets[k+1]]` in m/s, normalised so
/// `dt_s * samples.sum()` recovers the slip. An empty row is a subfault that doesn't
/// slip. A rise time under half a sample slips instantaneously, as the spike
/// `[0, slip/dt, 0]`, and keeps all of its slip; with `beta`, a slipping subfault's
/// negative or non-finite rise time is a `ValueError` naming it.
#[pyfunction]
#[pyo3(signature = (slip_m, rise_time_s, dt_s, beta=None))]
fn synthesise_pulses<'py>(
    py: Python<'py>,
    slip_m: PyReadonlyArray1<'py, f64>,
    rise_time_s: PyReadonlyArray1<'py, f64>,
    dt_s: f64,
    beta: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<PyCsr<'py>> {
    let slip = slip_m.as_slice()?;
    let rise = rise_time_s.as_slice()?;
    let shape = match &beta {
        Some(beta) => pulse::Shape::OliuP {
            beta: beta.as_slice()?,
        },
        None => pulse::Shape::Delta,
    };
    let pulses = py
        .detach(|| pulse::synthesise_pulses(slip, rise, shape, dt_s))
        .or_else(value_error)?;

    let offsets: Vec<i64> = pulses
        .offsets
        .iter()
        .map(|&offset| i64::try_from(offset).expect("sample counts fit in i64"))
        .collect();
    Ok((offsets.into_pyarray(py), pulses.samples.into_pyarray(py)))
}

/// A pair of independent standard-normal fields from one circulant-embedding draw.
///
/// `amplitudes` is the `(padded_i//2 + 1, padded_j//2 + 1)` quadrant of the
/// square-rooted, non-negative eigenvalues of the embedding on the `padded_shape`
/// grid, which the full grid mirrors. Returns `(real, imaginary)`, each cropped to
/// `cell_counts` and each a field with the embedded covariance, independent of the
/// other (Dietrich & Newsam 1997). The same `seed` gives the same pair.
///
/// The doc on [`field`] has the details.
#[pyfunction]
fn circulant_draw<'py>(
    py: Python<'py>,
    amplitudes: PyReadonlyArray2<'py, f64>,
    padded_shape: (usize, usize),
    cell_counts: (usize, usize),
    seed: u64,
) -> PyResult<PyFieldPair<'py>> {
    let quadrant = (amplitudes.shape()[0], amplitudes.shape()[1]);
    let amplitudes = amplitudes.as_slice()?;
    let (real, imaginary) = py
        .detach(|| field::draw(amplitudes, quadrant, padded_shape, cell_counts, seed))
        .or_else(value_error)?;
    let as_array = |values| {
        Array2::from_shape_vec(cell_counts, values)
            .expect("the draw returns one value per fault cell")
            .into_pyarray(py)
    };
    Ok((as_array(real), as_array(imaginary)))
}

/// The search between a jump's parent and child cells, built once per jump.
///
/// `parent_xyz` and `child_xyz` are `(n, 3)` cell centres in kilometres, `onset_s` the
/// parent cells' onsets. The bound on a pair's arrival is
/// `onset_s[p] + |p - c| / max_speed_km_s`, so `max_speed_km_s` must bound the shear
/// speed anywhere a crossing passes; infinity always does. [`jump`] has the method.
#[pyclass(frozen)]
struct JumpSearch(jump::Search);

/// `(n, 3)` positions as triples.
fn triples(xyz: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<[f64; 3]>> {
    if xyz.shape()[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "positions are shaped {:?}; they want (n, 3)",
            xyz.shape()
        )));
    }
    Ok(xyz.as_slice()?.as_chunks::<3>().0.to_vec())
}

#[pymethods]
impl JumpSearch {
    #[new]
    fn new(
        parent_xyz: PyReadonlyArray2<'_, f64>,
        onset_s: PyReadonlyArray1<'_, f64>,
        child_xyz: PyReadonlyArray2<'_, f64>,
        max_speed_km_s: f64,
    ) -> PyResult<Self> {
        let search = jump::Search::new(
            triples(&parent_xyz)?,
            onset_s.as_slice()?.to_vec(),
            triples(&child_xyz)?,
            max_speed_km_s,
        );
        search.map(Self).or_else(value_error)
    }

    /// The narrowest gap between the two charts, in kilometres.
    fn min_gap_km(&self, py: Python<'_>) -> f64 {
        py.detach(|| self.0.min_gap_km())
    }

    /// The pair `reach_km` or less apart with the least bound, as `(parent, child)`,
    /// or `None` if no pair is that close.
    fn least_bound(&self, py: Python<'_>, reach_km: f64) -> Option<(u32, u32)> {
        py.detach(|| self.0.least_bound(reach_km))
    }

    /// Every pair `reach_km` or less apart whose bound is at or below `threshold_s`, as
    /// parallel `(parents, children)` index arrays.
    fn pairs_below<'py>(&self, py: Python<'py>, reach_km: f64, threshold_s: f64) -> PyPairs<'py> {
        let (parents, children) = py.detach(|| self.0.pairs_below(reach_km, threshold_s));
        (parents.into_pyarray(py), children.into_pyarray(py))
    }
}

#[pymodule]
#[pyo3(name = "_kernels")]
fn kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eikonal_solve, m)?)?;
    m.add_function(wrap_pyfunction!(synthesise_pulses, m)?)?;
    m.add_function(wrap_pyfunction!(circulant_draw, m)?)?;
    m.add_class::<JumpSearch>()?;
    Ok(())
}
