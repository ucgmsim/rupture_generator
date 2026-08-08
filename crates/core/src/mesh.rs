//! The mesh boundary: a fault surface in, node positions out.
//!
//! Everything here is in the **projected** frame `genslip::mesh` works in — eastings and
//! northings in a Cartesian CRS the modeller named, in kilometres. Nothing on this side
//! knows about longitude, latitude or an ellipsoid; `rupture_generator.mesh` owns the
//! conversion out, and the grid convergence correction that has to go with it.
//!
//! # Positions are offsets, and the origin comes back separately
//!
//! Every array returned is an offset from [`RefinedMesh::origin`]. That is not an
//! inconvenience to be tidied away at the boundary: an NZTM northing is ~5,180 km against
//! a ~1 km subfault, so a mesh that held absolute coordinates would round every node at
//! CRS scale and hand back cell-scale quantities carrying `1.2e-12` relative error. The
//! offsets are exact to `3e-15`. Adding the origin back is one vectorised `+` in numpy,
//! at the same seam that does the projection, and it is the only place the large number
//! appears.
//!
//! # Why a patch index rather than a patch object
//!
//! A patch borrows its mesh's vertex list, and a `#[pyclass]` cannot hold a borrow. The
//! alternatives were a patch that copies its own vertices — which throws away the index
//! sharing that makes connectivity checkable — or an `Arc` and a lifetime dance. Passing
//! an index is neither, and it matches the flat style the rest of this boundary already
//! uses for the same reason (`FLAT_IS_THE_PYTHON_API`).

use genslip::mesh;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;

use crate::Refused;

/// Three `(dip, strike)` arrays: one axis each.
///
/// Named because it is returned three times and clippy is right that the tuple spelled
/// out is unreadable. What the three *are* differs by caller -- node offsets, cell
/// centres -- so the name says the shape rather than the meaning.
type Triple<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

/// One patch's node positions on the way *in*, from a file.
type ReadTriple<'py> = (
    PyReadonlyArray2<'py, f64>,
    PyReadonlyArray2<'py, f64>,
    PyReadonlyArray2<'py, f64>,
);

/// A horizontal position in the geometry's projected CRS, in kilometres.
#[pyclass(eq, frozen, from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projected {
    #[pyo3(get)]
    pub easting_km: f64,
    #[pyo3(get)]
    pub northing_km: f64,
}

#[pymethods]
impl Projected {
    #[new]
    const fn new(easting_km: f64, northing_km: f64) -> Self {
        Self {
            easting_km,
            northing_km,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Projected(easting_km={}, northing_km={})",
            self.easting_km, self.northing_km
        )
    }
}

impl From<Projected> for mesh::Projected {
    fn from(point: Projected) -> Self {
        Self {
            easting_km: point.easting_km,
            northing_km: point.northing_km,
        }
    }
}

impl From<mesh::Projected> for Projected {
    fn from(point: mesh::Projected) -> Self {
        Self {
            easting_km: point.easting_km,
            northing_km: point.northing_km,
        }
    }
}

/// One plane of a fault: where its top edge ends, and how it hangs from that edge.
///
/// Where the top edge *begins* is not here — it is the previous plane's `end`, or the
/// fault's `origin`. See [`Fault`].
#[pyclass(frozen, from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    inner: mesh::Plane,
}

#[pymethods]
impl Plane {
    #[new]
    #[pyo3(signature = (end, *, dip_deg, bottom_depth_km, dips_left = false))]
    fn new(end: Projected, dip_deg: f64, bottom_depth_km: f64, dips_left: bool) -> Self {
        Self {
            inner: mesh::Plane {
                end: end.into(),
                dip_deg,
                dip_direction: if dips_left {
                    mesh::DipDirection::Left
                } else {
                    mesh::DipDirection::Right
                },
                bottom_depth_km,
            },
        }
    }

    #[getter]
    fn end(&self) -> Projected {
        self.inner.end.into()
    }

    #[getter]
    const fn dip_deg(&self) -> f64 {
        self.inner.dip_deg
    }

    #[getter]
    const fn bottom_depth_km(&self) -> f64 {
        self.inner.bottom_depth_km
    }

    /// Whether the plane dips left of the trace direction rather than right.
    #[getter]
    fn dips_left(&self) -> bool {
        matches!(self.inner.dip_direction, mesh::DipDirection::Left)
    }
}

/// A fault: one or more planes, connected end to end.
///
/// Disconnection is unrepresentable rather than refused. A plane says only where its top
/// edge *ends*; where it begins is the previous plane's end, or this fault's `origin`, so
/// there is no second copy of a shared corner to disagree with the first.
///
/// The Rust type holds its first plane separately from the rest, which makes "a fault has
/// at least one plane" a property of the type. Python has no such shape, so the empty
/// list is refused **here** — the one place it can be.
#[pyclass(frozen, from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Debug)]
pub struct Fault {
    inner: mesh::Fault,
}

#[pymethods]
impl Fault {
    #[new]
    #[pyo3(signature = (origin, planes, *, top_depth_km = 0.0))]
    fn new(origin: Projected, planes: Vec<Plane>, top_depth_km: f64) -> PyResult<Self> {
        let mut planes = planes.into_iter().map(|plane| plane.inner);
        let first = planes
            .next()
            .ok_or_else(|| PyValueError::new_err("a fault needs at least one plane"))?;
        Ok(Self {
            inner: mesh::Fault {
                origin: origin.into(),
                top_depth_km,
                first,
                rest: planes.collect(),
            },
        })
    }

    #[getter]
    fn origin(&self) -> Projected {
        self.inner.origin.into()
    }

    #[getter]
    const fn top_depth_km(&self) -> f64 {
        self.inner.top_depth_km
    }

    /// How many planes there are. At least one, by construction.
    #[getter]
    fn plane_count(&self) -> usize {
        self.inner.plane_count()
    }
}

/// A point source: one cell, of a given size, centred where it is told.
#[pyclass(frozen, from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Copy, Debug)]
pub struct PointSource {
    inner: mesh::PointSpec,
}

#[pymethods]
impl PointSource {
    #[new]
    #[pyo3(signature = (centre, *, depth_km, strike_deg, dip_deg, size_km))]
    const fn new(
        centre: Projected,
        depth_km: f64,
        strike_deg: f64,
        dip_deg: f64,
        size_km: f64,
    ) -> Self {
        Self {
            inner: mesh::PointSpec {
                centre: mesh::Projected {
                    easting_km: centre.easting_km,
                    northing_km: centre.northing_km,
                },
                depth_km,
                strike_deg,
                dip_deg,
                size_km,
            },
        }
    }
}

/// How one face is cut up.
#[pyclass(eq, frozen, from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cuts {
    #[pyo3(get)]
    pub strike_count: usize,
    #[pyo3(get)]
    pub dip_count: usize,
}

#[pymethods]
impl Cuts {
    #[new]
    const fn new(strike_count: usize, dip_count: usize) -> Self {
        Self {
            strike_count,
            dip_count,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Cuts(strike_count={}, dip_count={})",
            self.strike_count, self.dip_count
        )
    }
}

/// A fault surface, cut into cells.
///
/// Built by [`build_mesh`], or rebuilt from stored arrays by
/// [`RefinedMesh::from_positions`].
#[pyclass(frozen, skip_from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Debug)]
pub struct RefinedMesh {
    inner: mesh::RefinedMesh,
}

#[pymethods]
impl RefinedMesh {
    /// Rebuild a mesh from node positions, one array triple per patch.
    ///
    /// What a *reader* needs: a mesh file stores patches as `(dip_node, strike_node)`
    /// grids of positions, so loading one is this rather than a build followed by a
    /// refine, and the geometry that comes back is whatever was written.
    ///
    /// Each patch becomes its own vertices, so no sharing survives a round trip through a
    /// file. Nothing needs it to: sharing is a claim about how the mesh was *built*, and
    /// the only consumer that walks a seam is a renderer, which duplicates anyway.
    #[staticmethod]
    #[pyo3(signature = (origin, patches))]
    fn from_positions(origin: Projected, patches: Vec<ReadTriple<'_>>) -> PyResult<Self> {
        let mut vertices = Vec::new();
        let mut nodes = Vec::with_capacity(patches.len());

        for (index, (east_km, north_km, depth_km)) in patches.iter().enumerate() {
            let east = east_km.as_array();
            let north = north_km.as_array();
            let depth = depth_km.as_array();
            if east.dim() != north.dim() || east.dim() != depth.dim() {
                return Err(PyValueError::new_err(format!(
                    "patch {index} has east {:?}, north {:?} and depth {:?}, which must \
                     be the same shape",
                    east.dim(),
                    north.dim(),
                    depth.dim()
                )));
            }

            let start = vertices.len();
            for (position, _) in east.indexed_iter() {
                vertices.push(mesh::Vertex {
                    east_km: east[position],
                    north_km: north[position],
                    depth_km: depth[position],
                });
            }
            nodes.push(Array2::from_shape_fn(east.raw_dim(), |(dip, strike)| {
                start + dip * east.ncols() + strike
            }));
        }

        Ok(Self {
            inner: mesh::RefinedMesh::from_parts(origin.into(), vertices, nodes)
                .map_err(Refused)?,
        })
    }

    /// The point every position is measured from.
    #[getter]
    fn origin(&self) -> Projected {
        self.inner.origin().into()
    }

    /// How many patches there are — one per plane of the fault.
    #[getter]
    fn patch_count(&self) -> usize {
        self.inner.patch_count()
    }

    /// Cells along strike and down dip, on one patch.
    fn cell_extents(&self, patch: usize) -> PyResult<(usize, usize)> {
        Ok(self.view(patch)?.cell_extents())
    }

    /// Node offsets on one patch: east, north and depth, each `(dip_node, strike_node)`.
    ///
    /// Offsets from [`RefinedMesh::origin`], in kilometres. Add it back to get
    /// coordinates in the CRS — see the module note for why that is not done here.
    fn node_positions<'py>(&self, py: Python<'py>, patch: usize) -> PyResult<Triple<'py>> {
        let positions = self.view(patch)?.positions();
        Ok((
            positions.east_km.into_pyarray(py),
            positions.north_km.into_pyarray(py),
            positions.depth_km.into_pyarray(py),
        ))
    }

    /// Cell-centre offsets on one patch, each `(dip, strike)`.
    fn cell_centres<'py>(&self, py: Python<'py>, patch: usize) -> PyResult<Triple<'py>> {
        let centres = self.view(patch)?.centres();
        Ok((
            centres.east_km.into_pyarray(py),
            centres.north_km.into_pyarray(py),
            centres.depth_km.into_pyarray(py),
        ))
    }

    /// The area of each cell, in square kilometres.
    fn areas_km2<'py>(&self, py: Python<'py>, patch: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.view(patch)?.areas_km2().into_pyarray(py))
    }

    /// The strike of each cell, in degrees clockwise from the projection's northing axis.
    ///
    /// **Grid north, not true north.** `rupture_generator.mesh` adds the convergence
    /// angle; in NZTM that reaches five degrees, which is five times the rake bound.
    fn strike_deg<'py>(
        &self,
        py: Python<'py>,
        patch: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.view(patch)?.strike_deg().into_pyarray(py))
    }

    /// The dip of each cell, in degrees below horizontal. Needs no correction.
    fn dip_deg<'py>(&self, py: Python<'py>, patch: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.view(patch)?.dip_deg().into_pyarray(py))
    }

    /// Distance along strike to each node, in kilometres, from the `i = 0` edge.
    fn strike_arc_km<'py>(
        &self,
        py: Python<'py>,
        patch: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.view(patch)?.strike_arc_km().into_pyarray(py))
    }

    /// Distance down dip to each node, in kilometres, from the top edge.
    fn dip_arc_km<'py>(
        &self,
        py: Python<'py>,
        patch: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.view(patch)?.dip_arc_km().into_pyarray(py))
    }

    /// The uniform cell spacing a patch has, as `(strike_km, dip_km)`.
    fn spacing(&self, patch: usize) -> PyResult<(f64, f64)> {
        let spacing = self.view(patch)?.spacing().map_err(Refused)?;
        Ok((spacing.strike_km, spacing.dip_km))
    }

    /// The cell containing a position given as two in-fault arc lengths.
    ///
    /// `strike_km` is measured from the `i = 0` end and `dip_km` from the top edge, and
    /// the result is a zero-based `(strike, dip)` cell index. Not the SRF's `shyp`, which
    /// is measured from the along-strike centre.
    fn cell_index(&self, patch: usize, strike_km: f64, dip_km: f64) -> PyResult<(usize, usize)> {
        let found = self
            .view(patch)?
            .cell_index(strike_km, dip_km)
            .map_err(Refused)?;
        Ok((found.strike, found.dip))
    }
}

impl RefinedMesh {
    /// One patch, or an `IndexError` naming what there was.
    fn view(&self, patch: usize) -> PyResult<mesh::PatchView<'_>> {
        if patch >= self.inner.patch_count() {
            return Err(PyIndexError::new_err(format!(
                "patch {patch} of a mesh with {} patches",
                self.inner.patch_count()
            )));
        }
        Ok(self.inner.patch(patch))
    }
}

/// Discretise a fault into a mesh.
///
/// One [`Cuts`] per plane, in order.
///
/// # Errors
///
/// Whatever `genslip::mesh` refuses: a dip off a fault plane, an inverted depth range, a
/// surface above the ground, a repeated trace point, a cut into no cells, or a number of
/// cuts that does not match the number of planes.
#[pyfunction]
#[pyo3(signature = (fault, cuts))]
pub fn build_fault_mesh(fault: &Fault, cuts: Vec<Cuts>) -> PyResult<RefinedMesh> {
    let coarse = mesh::build(&mesh::Geometry::Fault(fault.inner.clone())).map_err(Refused)?;
    refine(&coarse, cuts)
}

/// Discretise a point source into a one-cell mesh.
///
/// Takes no cuts: a point source *is* one subfault, and asking for it to be cut into more
/// would be asking for a finite fault.
///
/// # Errors
///
/// A dip off a fault plane, a non-positive size, or a point too shallow to hold its own
/// subfault without reaching above the ground.
#[pyfunction]
pub fn build_point_mesh(point: &PointSource) -> PyResult<RefinedMesh> {
    let coarse = mesh::build(&mesh::Geometry::Point(point.inner)).map_err(Refused)?;
    refine(
        &coarse,
        vec![Cuts {
            strike_count: 1,
            dip_count: 1,
        }],
    )
}

fn refine(coarse: &mesh::Mesh, cuts: Vec<Cuts>) -> PyResult<RefinedMesh> {
    let cuts: Vec<mesh::Cuts> = cuts
        .into_iter()
        .map(|cut| mesh::Cuts {
            strike_count: cut.strike_count,
            dip_count: cut.dip_count,
        })
        .collect();
    Ok(RefinedMesh {
        inner: coarse.refine(&cuts).map_err(Refused)?,
    })
}
