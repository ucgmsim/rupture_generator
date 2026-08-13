//! The rupture kernels: stateless, array-in/array-out, and nothing else.
//!
//! Two functions cross the Python boundary — [`eikonal::solve`] and
//! [`pulse::synthesise_pulses`] — as `rupture_generator._kernels`. No RNG (noise is
//! drawn in numpy and arrives as arrays), no spec structs (parameters arrive as
//! scalars and arrays; the single copy of every default lives in Python), per
//! `PLAN.md` §3.5. The maths lives in [`eikonal`] and [`pulse`] over plain slices,
//! where `tests/` can generate inputs for it; this file only marshals numpy arrays
//! in and out.

// PyO3's `#[pyfunction]` extraction hands wrappers owned values, and the lint is
// right in general and wrong at every site in this file.
#![expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 argument extraction requires owned values"
)]

mod counts;
pub mod eikonal;
pub mod field;
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

/// First-arrival times over a fault chart, by factored fast sweeping.
///
/// `slowness` is 2-D in s/km, `i` down-dip and `j` along-strike; `spacing_km` is
/// `(d_i, d_j)`; `seeds` is a list of `(i, j, t0_seconds)` — points the front leaves
/// at known times, which is one triple for a hypocentre and several for a fault
/// triggered along an edge. Returns travel times in seconds, same shape as
/// `slowness`. Exact on uniform media, first-order convergent on smooth ones;
/// `crates/kernels/src/eikonal.rs` has the papers.
#[pyfunction]
fn eikonal_solve<'py>(
    py: Python<'py>,
    slowness: PyReadonlyArray2<'py, f64>,
    spacing_km: (f64, f64),
    seeds: Vec<(i64, i64, f64)>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let extent = (slowness.shape()[0], slowness.shape()[1]);
    let cells = slowness.as_slice()?;
    let seeds: Vec<eikonal::Seed> = seeds
        .iter()
        .enumerate()
        .map(|(index, &(i, j, t0_s))| {
            // A negative index is out of bounds, said in the solver's own words
            // rather than as an integer-conversion failure.
            match (usize::try_from(i), usize::try_from(j)) {
                (Ok(i), Ok(j)) => Ok(eikonal::Seed { i, j, t0_s }),
                _ => Err(PyValueError::new_err(format!(
                    "seed {index} at ({i}, {j}) is outside a {}x{} grid",
                    extent.0, extent.1
                ))),
            }
        })
        .collect::<PyResult<_>>()?;

    let times = py
        .detach(|| eikonal::solve(cells, extent, spacing_km, &seeds))
        .or_else(value_error)?;
    let times =
        Array2::from_shape_vec(extent, times).expect("the solver returns one time per input cell");
    Ok(times.into_pyarray(py))
}

/// Slip-rate pulses for every subfault, as CSR rows.
///
/// `slip_m` (metres) and `rise_time_s` are flat, one entry per subfault; `shape` is
/// an already-resolved kernel shape, `"oliu_p"` (which requires `beta`, the
/// per-subfault rising fraction in `(0, 0.5]`) or `"delta"`. Returns
/// `(offsets, samples)`: subfault `k`'s pulse is `samples[offsets[k]:offsets[k+1]]`
/// in m/s, normalised so `dt_s * samples.sum()` recovers the slip. An empty row is a
/// subfault that does not slip; a subfault that slips but whose rise time rounds to
/// zero samples at `dt_s` is a `ValueError` naming it — never a silent zero
/// (`DEFECTS.md` 21).
#[pyfunction]
#[pyo3(signature = (slip_m, rise_time_s, dt_s, shape, beta=None))]
fn synthesise_pulses<'py>(
    py: Python<'py>,
    slip_m: PyReadonlyArray1<'py, f64>,
    rise_time_s: PyReadonlyArray1<'py, f64>,
    dt_s: f64,
    shape: &str,
    beta: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<PyCsr<'py>> {
    let slip = slip_m.as_slice()?;
    let rise = rise_time_s.as_slice()?;
    let pulses = match (shape, &beta) {
        ("oliu_p", Some(beta)) => {
            let beta = beta.as_slice()?;
            py.detach(|| pulse::synthesise_pulses(slip, rise, pulse::Shape::OliuP { beta }, dt_s))
        }
        ("oliu_p", None) => {
            return Err(PyValueError::new_err(
                "oliu_p needs beta, the per-subfault rising fraction",
            ));
        }
        ("delta", None) => {
            py.detach(|| pulse::synthesise_pulses(slip, rise, pulse::Shape::Delta, dt_s))
        }
        ("delta", Some(_)) => {
            return Err(PyValueError::new_err("delta has no beta parameter"));
        }
        // The vocabulary seam — genslip's `stype` names, and the refusal of the
        // removed ones — lives in Python. This kernel only knows its own two shapes.
        (other, _) => {
            return Err(PyValueError::new_err(format!(
                "'{other}' is not a kernel shape; resolved shapes are 'oliu_p' and 'delta'"
            )));
        }
    }
    .or_else(value_error)?;

    let offsets: Vec<i64> = pulses
        .offsets
        .iter()
        .map(|&offset| i64::try_from(offset).expect("sample counts fit in i64"))
        .collect();
    Ok((offsets.into_pyarray(py), pulses.samples.into_pyarray(py)))
}

#[pymodule]
#[pyo3(name = "_kernels")]
fn kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eikonal_solve, m)?)?;
    m.add_function(wrap_pyfunction!(synthesise_pulses, m)?)?;
    m.add_function(wrap_pyfunction!(field::von_karman_draw, m)?)?;
    Ok(())
}
