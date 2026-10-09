//! The Python-facing mirrors of the SRF's records.
//!
//! Each class here is a `#[pyclass]` whose every field is readable and writable,
//! with a `#[new]` taking those fields in order. Written out, that shape
//! states each field list **three** times: once in the struct with a `#[pyo3(get,
//! set)]` on every line, once as the constructor's parameters, and once in the
//! struct literal it returns. `PySrfMetadata` alone was 68 lines for thirteen fields,
//! and adding a column to the SRF meant three edits per class with a compiler that
//! caught only some of the omissions.
//!
//! The Python API stays the same. The field names are the macro's arguments, and
//! `PySrfMetadata(lon=..., lat=...)` and `plane.dtop` work exactly as before.

use numpy::PyArray1;
use pyo3::prelude::*;

use crate::types::SrfPlane;

/// A `#[pyclass]` whose `#[new]` takes every field, in declaration order.
///
/// The optional `signature` arm exists for the one class with defaulted arguments:
/// `PyO3` doesn't infer `vs=None` from `Option<T>`, and that default belongs to the
/// Python API rather than to the Rust type.
///
/// `#[macro_export]` rather than textual scope, because `types.rs` builds `SrfPlane`
/// with it. `macro_rules!` items don't accept `pub`/`pub(crate)`. Exporting is the
/// only way a macro crosses a module boundary, and it puts the macro at the crate
/// root, reachable as `crate::py_record!`.
#[macro_export]
macro_rules! py_record {
    (
        $(#[$attr:meta])*
        $name:ident { $($field:ident: $type:ty),* $(,)? }
    ) => {
        py_record!($(#[$attr])* $name { $($field: $type),* } signature = ($($field),*));
    };
    (
        $(#[$attr:meta])*
        $name:ident { $($field:ident: $type:ty),* $(,)? }
        signature = ($($signature:tt)*)
    ) => {
        $(#[$attr])*
        pub struct $name {
            $(#[pyo3(get, set)] pub $field: $type,)*
        }

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = ($($signature)*))]
            #[allow(clippy::too_many_arguments)]
            #[must_use]
            pub fn new($($field: $type),*) -> Self {
                Self { $($field),* }
            }
        }
    };
}

py_record! {
    /// The slip-rate matrix, as compressed sparse rows.
    ///
    /// `indices` is optional, and the writer has no use for it. Every pulse in an SRF
    /// starts at column zero and runs contiguously. A sample's column is then
    /// `arange(n) - repeat(row_ptr[:-1], diff(row_ptr))`, a function of `row_ptr`
    /// alone with no information of its own. `srf_writer` walks `row_ptr` and `data`
    /// and never looks at it.
    ///
    /// It has one `usize` per *sample*. On a twenty-fault rupture, materialising it to
    /// hand over took 7.6 GB, and the widening cast that produced it was the largest
    /// allocation on the path to writing a file that doesn't contain it. Give the
    /// writer `None`. The parser still fills it in, for a caller that wants to hand
    /// the result to `scipy.sparse`.
    #[pyclass]
    #[derive(Debug)]
    PyCsrMatrix {
        row_ptr: Py<PyArray1<usize>>,
        data: Py<PyArray1<f32>>,
        indices: Option<Py<PyArray1<usize>>>,
    }
    signature = (row_ptr, data, indices = None)
}

py_record! {
    #[pyclass]
    #[derive(Debug)]
    PySrfMetadata {
        lon: Py<PyArray1<f32>>,
        lat: Py<PyArray1<f32>>,
        dep: Py<PyArray1<f32>>,
        stk: Py<PyArray1<f32>>,
        dip: Py<PyArray1<f32>>,
        area: Py<PyArray1<f32>>,
        tinit: Py<PyArray1<f32>>,
        dt: Py<PyArray1<f32>>,
        rake: Py<PyArray1<f32>>,
        slip1: Py<PyArray1<f32>>,
        rise: Py<PyArray1<f32>>,
        vs: Option<Py<PyArray1<f32>>>,
        density: Option<Py<PyArray1<f32>>>,
    }
    signature = (
        lon, lat, dep, stk, dip, area, tinit, dt, rake, slip1, rise,
        vs = None, density = None
    )
}

py_record! {
    #[pyclass]
    #[derive(Debug)]
    PySrfFile {
        planes: Vec<Py<SrfPlane>>,
        metadata: Py<PySrfMetadata>,
        slipt1: Py<PyCsrMatrix>,
    }
}
