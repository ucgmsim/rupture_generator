use numpy::PyArray1;
use pyo3::prelude::*;

#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct PySrfPlane {
    #[pyo3(get, set)]
    pub elon: f32,
    #[pyo3(get, set)]
    pub elat: f32,
    #[pyo3(get, set)]
    pub nstk: usize,
    #[pyo3(get, set)]
    pub ndip: usize,
    #[pyo3(get, set)]
    pub len: f32,
    #[pyo3(get, set)]
    pub wid: f32,
    #[pyo3(get, set)]
    pub stk: f32,
    #[pyo3(get, set)]
    pub dip: f32,
    #[pyo3(get, set)]
    pub dtop: f32,
    #[pyo3(get, set)]
    pub shyp: f32,
    #[pyo3(get, set)]
    pub dhyp: f32,
}

#[pymethods]
impl PySrfPlane {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        elon: f32,
        elat: f32,
        nstk: usize,
        ndip: usize,
        len: f32,
        wid: f32,
        stk: f32,
        dip: f32,
        dtop: f32,
        shyp: f32,
        dhyp: f32,
    ) -> Self {
        PySrfPlane {
            elon,
            elat,
            nstk,
            ndip,
            len,
            wid,
            stk,
            dip,
            dtop,
            shyp,
            dhyp,
        }
    }
}

#[pyclass]
#[derive(Debug)]
pub struct PyCsrMatrix {
    #[pyo3(get, set)]
    pub row_ptr: Py<PyArray1<usize>>,
    #[pyo3(get, set)]
    pub indices: Py<PyArray1<usize>>,
    #[pyo3(get, set)]
    pub data: Py<PyArray1<f32>>,
}

#[pymethods]
impl PyCsrMatrix {
    #[new]
    #[must_use]
    pub fn new(
        row_ptr: Py<PyArray1<usize>>,
        indices: Py<PyArray1<usize>>,
        data: Py<PyArray1<f32>>,
    ) -> Self {
        PyCsrMatrix {
            row_ptr,
            indices,
            data,
        }
    }
}

#[pyclass]
#[derive(Debug)]
pub struct PySrfMetadata {
    #[pyo3(get, set)]
    pub lon: Py<PyArray1<f32>>,
    #[pyo3(get, set)]
    pub lat: Py<PyArray1<f32>>,
    #[pyo3(get, set)]
    pub dep: Py<PyArray1<f32>>,
    #[pyo3(get, set)]
    pub stk: Py<PyArray1<f32>>,
    #[pyo3(get, set)]
    pub dip: Py<PyArray1<f32>>,
    #[pyo3(get, set)]
    pub area: Py<PyArray1<f32>>,
    #[pyo3(get, set)]
    pub tinit: Py<PyArray1<f32>>,
    #[pyo3(get, set)]
    pub dt: Py<PyArray1<f32>>,
    #[pyo3(get, set)]
    pub rake: Py<PyArray1<f32>>,
    #[pyo3(get, set)]
    pub slip1: Py<PyArray1<f32>>,
    #[pyo3(get, set)]
    pub rise: Py<PyArray1<f32>>,
    #[pyo3(get, set)]
    pub vs: Option<Py<PyArray1<f32>>>,
    #[pyo3(get, set)]
    pub density: Option<Py<PyArray1<f32>>>,
}

#[pymethods]
impl PySrfMetadata {
    #[new]
    #[pyo3(signature = (lon, lat, dep, stk, dip, area, tinit, dt, rake, slip1, rise, vs=None, density=None))]
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
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
    ) -> Self {
        PySrfMetadata {
            lon,
            lat,
            dep,
            stk,
            dip,
            area,
            tinit,
            dt,
            rake,
            slip1,
            rise,
            vs,
            density,
        }
    }
}

#[pyclass]
#[derive(Debug)]
pub struct PySrfFile {
    #[pyo3(get, set)]
    pub planes: Vec<Py<PySrfPlane>>,
    #[pyo3(get, set)]
    pub metadata: Py<PySrfMetadata>,
    #[pyo3(get, set)]
    pub slipt1: Py<PyCsrMatrix>,
}

#[pymethods]
impl PySrfFile {
    #[new]
    #[must_use]
    pub fn new(
        planes: Vec<Py<PySrfPlane>>,
        metadata: Py<PySrfMetadata>,
        slipt1: Py<PyCsrMatrix>,
    ) -> Self {
        PySrfFile {
            planes,
            metadata,
            slipt1,
        }
    }
}
