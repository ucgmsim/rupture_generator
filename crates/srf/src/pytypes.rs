//! The Python-facing mirrors of the SRF's records.
//!
//! Every one of them is the same shape: a `#[pyclass]` whose every field is readable
//! and writable, and a `#[new]` taking those fields in order. Written out, that shape
//! states each field list **three** times -- once in the struct with a `#[pyo3(get,
//! set)]` above every line, once as the constructor's parameters, and once in the
//! struct literal it returns. `PySrfMetadata` alone was 68 lines for thirteen fields,
//! and adding a column to the SRF meant three edits per class with a compiler that
//! caught only some of the omissions.
//!
//! Python sees exactly what it saw before: the field names are the macro's arguments,
//! so `PySrfMetadata(lon=..., lat=...)` and `plane.dtop` are unchanged.

use numpy::PyArray1;
use pyo3::prelude::*;

use crate::types::SrfPlane;

/// A `#[pyclass]` whose `#[new]` takes every field, in declaration order.
///
/// The optional `signature` arm exists for the one class with defaulted arguments:
/// `PyO3` will not infer `vs=None` from `Option<T>`, and that default is part of the
/// Python API rather than of the Rust type.
///
/// `#[macro_export]` rather than textual scope: `SrfPlane` is built with it from
/// `types.rs`. `macro_rules!` items do not accept `pub`/`pub(crate)` -- exporting
/// is the only way one crosses a module boundary, and it lands the macro at the
/// crate root, reachable as `crate::py_record!`.
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
    #[pyclass]
    #[derive(Debug)]
    PyCsrMatrix {
        row_ptr: Py<PyArray1<usize>>,
        indices: Py<PyArray1<usize>>,
        data: Py<PyArray1<f32>>,
    }
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
