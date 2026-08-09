// These arguments are taken by value because their callers require it: PyO3's
// `#[pyfunction]` will not accept a reference to a `PyBuffer` or a `Py<T>`, and
// `map_err` hands its closure an owned error. The lint is right in general and wrong
// at every site in this file.
#![expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 argument extraction and map_err both require owned values"
)]

pub mod pytypes;
mod scanner;
mod srf_parser;
mod srf_writer;
mod types;

use numpy::PyArrayMethods;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::error;
use std::fs::File;
use std::io::{BufWriter, Error, Write};

use crate::pytypes::{PyCsrMatrix, PySrfFile, PySrfMetadata};
use crate::types::{
    CsrMatrixView, SrfFileView, SrfMetadataV2View, SrfMetadataVersioned, SrfMetadataView, SrfPlane,
};

const WRITE_BUFFER_CAPACITY: usize = 1 << 20;

fn marshall_os_error<T>(e: Error) -> PyResult<T> {
    Err(PyErr::new::<PyOSError, _>(e.to_string()))
}

fn marshall_value_error<T, U: error::Error>(e: U) -> PyResult<T> {
    Err(PyErr::new::<PyValueError, _>(e.to_string()))
}

fn buffer_bytes(buf: &PyBuffer<u8>) -> &[u8] {
    // SAFETY: caller guarantees a live, C-contiguous, readable u8 export.
    // Lifetime is tied to `buf`, so the borrow checker forbids dropping the
    // PyBuffer while this slice is in use.
    unsafe { std::slice::from_raw_parts(buf.buf_ptr().cast(), buf.item_count()) }
}

#[pyfunction]
/// # Errors
///
/// If the buffer is not C-contiguous, or does not hold a valid SRF file.
pub fn parse_srf(py: Python<'_>, buffer: PyBuffer<u8>) -> PyResult<Py<PySrfFile>> {
    if !buffer.is_c_contiguous() {
        return Err(PyValueError::new_err("SRF buffer must be C-contiguous"));
    }
    let bytes = buffer_bytes(&buffer);
    if bytes.is_empty() {
        return Err(PyValueError::new_err("Cannot parse SRF from empty buffer"));
    }
    let srf_file = py.detach(|| {
        let mut scanner = scanner::Scanner::new(bytes);
        srf_parser::read_srf_struct(&mut scanner).or_else(marshall_value_error)
    })?;
    Ok(srf_file.into_pyobject(py)?.unbind())
}

#[pyfunction]
/// # Errors
///
/// If the file cannot be opened or written.
pub fn write_srf(py: Python<'_>, py_srf_file: Py<PySrfFile>, file_path: &str) -> PyResult<()> {
    let srf = py_srf_file.borrow(py);
    let metadata = srf.metadata.borrow(py);
    let slipt1 = srf.slipt1.borrow(py);

    let planes: Vec<SrfPlane> = srf.planes.iter().map(|plane| *plane.borrow(py)).collect();

    // Readonly borrows of the numpy buffers, so nothing large is copied between Python
    // and Rust. The guards have to outlive the slices taken from them, which is why
    // they are bound to named locals here rather than inside the struct literal.
    //
    // The view's name is an argument because a `let` introduced inside a macro is
    // hygienic and would not be visible here.
    macro_rules! borrow_columns {
        ($view:ident = $($column:ident),* $(,)?) => {
            $(let $column = metadata.$column.bind(py).readonly();)*
            let $view: SrfMetadataView = SrfMetadataView {
                $($column: $column.as_slice()?,)*
            };
        };
    }
    borrow_columns! {
        base = lon, lat, dep, stk, dip, area, tinit, dt, rake, slip1, rise
    }

    let vs = metadata.vs.as_ref().map(|arr| arr.bind(py).readonly());
    let density = metadata.density.as_ref().map(|arr| arr.bind(py).readonly());
    let row_ptr = slipt1.row_ptr.bind(py).readonly();
    let data = slipt1.data.bind(py).readonly();
    // Absent unless a caller went out of its way to supply it, and the writer does not
    // read it either way -- `srf_writer` walks `row_ptr` and `data`. See `PyCsrMatrix`.
    let indices = slipt1
        .indices
        .as_ref()
        .map(|array| array.bind(py).readonly());

    let metadata_view = match (&vs, &density) {
        (Some(vs), Some(density)) => SrfMetadataVersioned::V2(SrfMetadataV2View {
            base,
            vs: vs.as_slice()?,
            density: density.as_slice()?,
        }),
        (None, None) => SrfMetadataVersioned::V1(base),
        _ => {
            return Err(PyErr::new::<PyValueError, _>(
                "vs and density must both be set (SRF v2) or both be None (SRF v1)",
            ));
        }
    };

    let srf_view: SrfFileView = SrfFileView {
        planes,
        metadata: metadata_view,
        slipt1: CsrMatrixView {
            row_ptr: row_ptr.as_slice()?,
            data: data.as_slice()?,
            indices: match &indices {
                Some(indices) => indices.as_slice()?,
                None => &[],
            },
        },
    };

    // The view only borrows plain slices, so the whole write can run without
    // the GIL.
    py.detach(|| {
        let file = File::create(file_path).or_else(marshall_os_error)?;
        let mut writer = BufWriter::with_capacity(WRITE_BUFFER_CAPACITY, file);
        srf_writer::write_srf(&mut writer, &srf_view).or_else(marshall_os_error)?;
        writer.flush().or_else(marshall_os_error)
    })
}

#[pymodule]
#[pyo3(name = "srf_parser")]
fn srf_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SrfPlane>()?;
    m.add_class::<PyCsrMatrix>()?;
    m.add_class::<PySrfMetadata>()?;
    m.add_class::<PySrfFile>()?;
    m.add_function(wrap_pyfunction!(write_srf, m)?)?;
    m.add_function(wrap_pyfunction!(parse_srf, m)?)?;

    Ok(())
}
