// These functions take their arguments by value because their callers require it:
// PyO3's `#[pyfunction]` doesn't accept a reference to a `PyBuffer` or a `Py<T>`, and
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
use std::io::{BufWriter, Error, ErrorKind, Write};

use crate::pytypes::{PyCsrMatrix, PySrfFile, PySrfMetadata};
use crate::srf_writer::{PointStream, StreamError};
use crate::types::{
    CsrMatrixView, SrfFileView, SrfMetadataV2View, SrfMetadataVersioned, SrfMetadataView, SrfPlane,
};

const WRITE_BUFFER_CAPACITY: usize = 1 << 20;

fn marshall_os_error<T>(e: Error) -> PyResult<T> {
    // `InvalidInput` is the writer saying the points don't fit the planes, which is
    // the caller's mistake rather than the file system's.
    if e.kind() == ErrorKind::InvalidInput {
        return Err(PyErr::new::<PyValueError, _>(e.to_string()));
    }
    Err(PyErr::new::<PyOSError, _>(e.to_string()))
}

fn marshall_stream_error<T>(e: StreamError) -> PyResult<T> {
    marshall_os_error(e.into())
}

/// Readonly borrows of a `PySrfMetadata`'s base columns, as a `SrfMetadataView`.
macro_rules! borrow_columns {
    ($py:ident, $metadata:ident, $view:ident = $($column:ident),* $(,)?) => {
        $(let $column = $metadata.$column.bind($py).readonly();)*
        let $view: SrfMetadataView = SrfMetadataView {
            $($column: $column.as_slice()?,)*
        };
    };
}

fn marshall_value_error<T, U: error::Error>(e: U) -> PyResult<T> {
    Err(PyErr::new::<PyValueError, _>(e.to_string()))
}

fn buffer_bytes(buf: &PyBuffer<u8>) -> &[u8] {
    // SAFETY: the caller promises a live, C-contiguous, readable u8 export. The
    // slice's lifetime is that of `buf`, and dropping the PyBuffer while the slice
    // is in use is a compile error.
    unsafe { std::slice::from_raw_parts(buf.buf_ptr().cast(), buf.item_count()) }
}

#[pyfunction]
/// # Errors
///
/// If the buffer isn't C-contiguous, or isn't a valid SRF file.
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
/// If opening or writing the file fails.
pub fn write_srf(py: Python<'_>, py_srf_file: Py<PySrfFile>, file_path: &str) -> PyResult<()> {
    let srf = py_srf_file.borrow(py);
    let metadata = srf.metadata.borrow(py);
    let slipt1 = srf.slipt1.borrow(py);

    let planes: Vec<SrfPlane> = srf.planes.iter().map(|plane| *plane.borrow(py)).collect();

    borrow_columns! {
        py, metadata, base = lon, lat, dep, stk, dip, area, tinit, dt, rake, slip1, rise
    }

    let vs = metadata.vs.as_ref().map(|arr| arr.bind(py).readonly());
    let density = metadata.density.as_ref().map(|arr| arr.bind(py).readonly());
    let row_ptr = slipt1.row_ptr.bind(py).readonly();
    let data = slipt1.data.bind(py).readonly();
    // Absent unless a caller went out of its way to supply it, and the writer doesn't
    // read it either way: `srf_writer` walks `row_ptr` and `data`. See `PyCsrMatrix`.
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

/// Streaming SRF writer.
#[pyclass(name = "SrfWriter")]
pub struct PySrfWriter {
    writer: Option<BufWriter<File>>,
    stream: PointStream,
}

#[pymethods]
impl PySrfWriter {
    /// # Errors
    ///
    /// If creating or writing the file fails.
    #[new]
    fn new(py: Python<'_>, file_path: &str, planes: Vec<SrfPlane>) -> PyResult<Self> {
        let writer = py.detach(|| {
            let file = File::create(file_path).or_else(marshall_os_error)?;
            let mut writer = BufWriter::with_capacity(WRITE_BUFFER_CAPACITY, file);
            srf_writer::write_header_v2(&mut writer, &planes).or_else(marshall_os_error)?;
            Ok::<_, PyErr>(writer)
        })?;
        Ok(Self {
            writer: Some(writer),
            stream: PointStream::new(&planes),
        })
    }

    /// # Errors
    ///
    /// If the writer has closed, the metadata has no `vs` or `density`, the rows and
    /// the points disagree in number, the points run past what the planes declare,
    /// or writing fails.
    fn write(
        &mut self,
        py: Python<'_>,
        metadata: PyRef<'_, PySrfMetadata>,
        slipt1: PyRef<'_, PyCsrMatrix>,
    ) -> PyResult<()> {
        let Some(writer) = self.writer.as_mut() else {
            return Err(PyValueError::new_err("SRF writer is closed"));
        };
        borrow_columns! {
            py, metadata, base = lon, lat, dep, stk, dip, area, tinit, dt, rake, slip1, rise
        }
        let (Some(vs), Some(density)) = (&metadata.vs, &metadata.density) else {
            return Err(PyValueError::new_err(
                "Only SRF 2.0 is supported: vs and density are required",
            ));
        };
        let vs = vs.bind(py).readonly();
        let density = density.bind(py).readonly();
        let row_ptr = slipt1.row_ptr.bind(py).readonly();
        let data = slipt1.data.bind(py).readonly();
        let points = base.lon.len();
        if row_ptr.len()? != points + 1 {
            return Err(PyValueError::new_err(format!(
                "Points declared in sparse matrix ({}) do not match point metadata length ({points})",
                row_ptr.len()?.saturating_sub(1)
            )));
        }
        let metadata_view = SrfMetadataV2View {
            base,
            vs: vs.as_slice()?,
            density: density.as_slice()?,
        };
        let slipt1_view = CsrMatrixView {
            row_ptr: row_ptr.as_slice()?,
            data: data.as_slice()?,
            indices: &[],
        };
        let stream = &mut self.stream;
        py.detach(|| {
            stream
                .write(writer, &metadata_view, &slipt1_view)
                .or_else(marshall_stream_error)
        })
    }

    /// # Errors
    ///
    /// If the stream wrote fewer points than the planes declare, or flushing fails.
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(mut writer) = self.writer.take() else {
            return Ok(());
        };
        let stream = &mut self.stream;
        py.detach(|| {
            stream.finish(&mut writer).or_else(marshall_stream_error)?;
            writer.flush().or_else(marshall_os_error)
        })
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// # Errors
    ///
    /// As `close`, when the block ended without an exception.
    fn __exit__(
        &mut self,
        py: Python<'_>,
        exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        if exc_type.is_some() {
            self.writer = None;
        } else {
            self.close(py)?;
        }
        Ok(false)
    }
}

#[pymodule]
#[pyo3(name = "srf_parser")]
fn srf_utils(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SrfPlane>()?;
    m.add_class::<PyCsrMatrix>()?;
    m.add_class::<PySrfMetadata>()?;
    m.add_class::<PySrfFile>()?;
    m.add_class::<PySrfWriter>()?;
    m.add_function(wrap_pyfunction!(write_srf, m)?)?;
    m.add_function(wrap_pyfunction!(parse_srf, m)?)?;

    Ok(())
}
