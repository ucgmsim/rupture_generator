use numpy::PyArray1;
use pyo3::prelude::*;

use crate::py_record;
use crate::pytypes::{PyCsrMatrix, PySrfFile, PySrfMetadata};

py_record! {
    // `Copy` is safe: every field is a scalar, and it is what lets `write_srf`
    // read a plane out of its `Py<SrfPlane>` with a dereference rather than a
    // clone or a field-by-field rebuild.
    #[pyclass(name = "PySrfPlane", from_py_object)]
    #[derive(Debug, Copy, Clone)]
    SrfPlane {
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
    }
}

impl SrfPlane {
    pub fn points(&self) -> usize {
        self.nstk * self.ndip
    }
}

/// CSR matrix over any storage: `Vec`s when parsing (the parser appends), or
/// borrowed slices when writing data that another allocator (e.g. numpy) owns.
///
/// `row_ptr` follows the scipy `indptr` convention: an n-row matrix has n+1
/// entries, `row_ptr[0] == 0`, `row_ptr[n] == data.len()`, and row i occupies
/// `data[row_ptr[i]..row_ptr[i + 1]]`. This holds after every `add_row`, so the
/// matrix can be handed to scipy or iterated at any point without a fixup pass.
#[derive(Debug)]
pub struct CsrMatrix<R = Vec<usize>, D = Vec<f32>> {
    pub indices: R,
    pub row_ptr: R,
    pub data: D,
}

pub type CsrMatrixView<'a> = CsrMatrix<&'a [usize], &'a [f32]>;

impl<'py> IntoPyObject<'py> for CsrMatrix {
    type Target = PyCsrMatrix;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(mut self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        self.finalise();
        Ok(Py::new(
            py,
            PyCsrMatrix {
                row_ptr: PyArray1::from_vec(py, self.row_ptr).unbind(),
                indices: PyArray1::from_vec(py, self.indices).unbind(),
                data: PyArray1::from_vec(py, self.data).unbind(),
            },
        )?
        .into_bound(py))
    }
}

impl CsrMatrix {
    pub fn new(row_capacity: usize, data_capacity: usize) -> Self {
        CsrMatrix {
            row_ptr: {
                let mut row_ptr = Vec::with_capacity(row_capacity + 1);
                row_ptr.push(0);
                row_ptr
            },
            indices: Vec::with_capacity(data_capacity),
            data: Vec::with_capacity(data_capacity),
        }
    }

    /// Append one point's slip-rate pulse.
    ///
    /// Column `i` is the `i`th sample **of the pulse**, not of the rupture. The
    /// onset time lives in `tinit`, as a float, and is not folded in here.
    ///
    /// It used to be. `add_row` took a `starting` column of `floor(tinit / dt)` and
    /// placed the samples there, which quantised every onset to a sample boundary --
    /// a pulse starting at 1.003 s with `dt = 0.005` was written at 1.000 s. It also
    /// made the matrix as wide as the whole rupture rather than as wide as its
    /// longest pulse, and needed a guard against a negative `tinit` casting to a huge
    /// index.
    ///
    /// Relative columns remove all three, and they are the layout the rupture
    /// generator already produces.
    pub fn add_row<I>(&mut self, values: I)
    where
        I: Iterator<Item = f32>,
    {
        for (index, value) in values.enumerate() {
            self.indices.push(index);
            self.data.push(value);
        }
        self.row_ptr.push(self.data.len());
    }

    pub fn finalise(&mut self) {
        self.row_ptr.shrink_to_fit();
        self.data.shrink_to_fit();
        self.indices.shrink_to_fit();
    }
}

impl<R: AsRef<[usize]>, D: AsRef<[f32]>> CsrMatrix<R, D> {
    pub fn rows(&self) -> CsrRowIter<'_> {
        CsrRowIter {
            row_ptr: self.row_ptr.as_ref(),
            data: self.data.as_ref(),
            index: 0,
        }
    }
}

pub struct CsrRowIter<'a> {
    row_ptr: &'a [usize],
    data: &'a [f32],
    index: usize,
}

impl<'a> Iterator for CsrRowIter<'a> {
    type Item = &'a [f32];

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.index;
        // n rows are described by n+1 row_ptr entries, so the last valid row
        // index is row_ptr.len() - 2.
        if i + 1 >= self.row_ptr.len() {
            return None;
        }
        self.index += 1;
        Some(&self.data[self.row_ptr[i]..self.row_ptr[i + 1]])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.row_ptr.len().saturating_sub(self.index + 1);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for CsrRowIter<'_> {}

#[cfg(test)]
mod csr_tests {
    use super::*;

    fn build(rows: &[&[f32]]) -> CsrMatrix {
        let mut matrix = CsrMatrix::new(rows.len(), rows.iter().map(|row| row.len()).sum());
        for row in rows {
            matrix.add_row(row.iter().copied());
        }
        matrix
    }

    // row_ptr must be a valid scipy indptr after every add_row, not just once
    // finalise has run.
    fn assert_indptr_invariant(matrix: &CsrMatrix, rows: usize) {
        assert_eq!(matrix.row_ptr.len(), rows + 1);
        assert_eq!(matrix.row_ptr[0], 0);
        assert_eq!(*matrix.row_ptr.last().unwrap(), matrix.data.len());
        assert!(matrix.row_ptr.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn empty_matrix_has_zero_rows() {
        let matrix = build(&[]);
        assert_indptr_invariant(&matrix, 0);
        assert_eq!(matrix.rows().count(), 0);
    }

    #[test]
    fn invariant_holds_after_every_add_row() {
        let mut matrix = CsrMatrix::new(3, 6);
        for (i, row) in [&[1.0f32, 2.0][..], &[][..], &[3.0][..]].iter().enumerate() {
            matrix.add_row(row.iter().copied());
            assert_indptr_invariant(&matrix, i + 1);
        }
    }

    /// Every row starts at column zero, whatever its onset time.
    ///
    /// This test used to assert the opposite: rows were placed at
    /// `floor(tinit / dt)` on a shared timeline, so two pulses with different onsets
    /// occupied different columns and the matrix was as wide as the whole rupture.
    /// The onset is a float in the metadata now, so nothing quantises it and the
    /// matrix is only as wide as the longest pulse.
    #[test]
    fn every_row_starts_at_column_zero() {
        let mut matrix = CsrMatrix::new(2, 4);
        matrix.add_row([1.0f32, 2.0].into_iter());
        matrix.add_row([3.0f32, 4.0].into_iter());
        assert_eq!(matrix.indices, vec![0, 1, 0, 1]);
        assert_eq!(matrix.row_ptr, vec![0, 2, 4]);
        assert_eq!(matrix.indices.iter().max().copied(), Some(1));
    }

    #[test]
    fn rows_yields_exactly_the_added_rows() {
        let matrix = build(&[&[1.0, 2.0], &[], &[3.0]]);
        let rows: Vec<&[f32]> = matrix.rows().collect();
        assert_eq!(rows, vec![&[1.0f32, 2.0][..], &[][..], &[3.0][..]]);
    }

    // The bug this guards: a trailing empty phantom row, previously produced by
    // iterating a finalised row_ptr and masked downstream by zip().
    #[test]
    fn finalise_does_not_change_the_rows() {
        let mut matrix = build(&[&[1.0, 2.0], &[3.0]]);
        let before: Vec<Vec<f32>> = matrix.rows().map(<[f32]>::to_vec).collect();
        let row_ptr_before = matrix.row_ptr.clone();
        matrix.finalise();
        let after: Vec<Vec<f32>> = matrix.rows().map(<[f32]>::to_vec).collect();
        assert_eq!(before, after);
        assert_eq!(matrix.row_ptr, row_ptr_before);
    }

    #[test]
    fn exact_size_hint_matches_rows_produced() {
        let matrix = build(&[&[1.0, 2.0], &[], &[3.0]]);
        let mut iter = matrix.rows();
        for expected in (0..=3).rev() {
            assert_eq!(iter.len(), expected);
            assert_eq!(iter.size_hint(), (expected, Some(expected)));
            if iter.next().is_none() {
                break;
            }
        }
    }
}

/// A point's eleven columns, and the six things built from that list.
///
/// The list is the SRF's own, in the SRF's own order, and it appeared **eight** times
/// in this file before this: the array-of-structs `Point`, the struct-of-arrays
/// `SrfMetadata`, `with_capacity`, `push`, `iter`, `PointIter`, its `next`, and the
/// conversion into `PySrfMetadata`. Nothing but care kept the eight in step, and
/// several of them would still compile with a field quietly reading the wrong column.
///
/// `$first` is taken separately because the iterator needs one column to measure its
/// own length against, and picking it out is what lets every other use be uniform.
macro_rules! point_columns {
    ($first:ident, $($rest:ident),* $(,)?) => {
        /// One point: a row of the SRF's point block.
        #[derive(Debug, Copy, Clone)]
        pub struct Point {
            pub $first: f32,
            $(pub $rest: f32,)*
        }

        /// Per-point metadata in struct-of-arrays layout. Generic over storage:
        /// `Vec<f32>` (the default) when the parser builds it, `&[f32]` when viewing
        /// numpy-owned arrays for writing.
        #[derive(Debug)]
        pub struct SrfMetadata<S = Vec<f32>> {
            pub $first: S,
            $(pub $rest: S,)*
        }

        impl SrfMetadata {
            pub fn with_capacity(n: usize) -> Self {
                SrfMetadata {
                    $first: Vec::with_capacity(n),
                    $($rest: Vec::with_capacity(n),)*
                }
            }

            pub fn push(&mut self, point: &Point) {
                self.$first.push(point.$first);
                $(self.$rest.push(point.$rest);)*
            }
        }

        impl<S: AsRef<[f32]>> SrfMetadata<S> {
            pub fn iter(&self) -> PointIter<'_> {
                PointIter {
                    $first: self.$first.as_ref(),
                    $($rest: self.$rest.as_ref(),)*
                    index: 0,
                }
            }
        }

        pub struct PointIter<'a> {
            $first: &'a [f32],
            $($rest: &'a [f32],)*
            index: usize,
        }

        impl Iterator for PointIter<'_> {
            type Item = Point;

            fn next(&mut self) -> Option<Self::Item> {
                let i = self.index;
                if i >= self.$first.len() {
                    return None;
                }
                self.index += 1;
                Some(Point {
                    $first: self.$first[i],
                    $($rest: self.$rest[i],)*
                })
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let remaining = self.$first.len() - self.index;
                (remaining, Some(remaining))
            }
        }

        impl<'py> IntoPyObject<'py> for SrfMetadata {
            type Target = PySrfMetadata;
            type Output = Bound<'py, Self::Target>;
            type Error = PyErr;

            fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
                Ok(Py::new(
                    py,
                    PySrfMetadata {
                        $first: PyArray1::from_vec(py, self.$first).unbind(),
                        $($rest: PyArray1::from_vec(py, self.$rest).unbind(),)*
                        // Not columns of the point block: an SRF carries them only
                        // when a velocity model was written alongside it.
                        vs: None,
                        density: None,
                    },
                )?
                .into_bound(py))
            }
        }
    };
}

point_columns!(lon, lat, dep, stk, dip, area, tinit, dt, rake, slip1, rise);

pub type SrfMetadataView<'a> = SrfMetadata<&'a [f32]>;

impl ExactSizeIterator for PointIter<'_> {}

#[derive(Debug, Copy, Clone)]
pub struct PointV2 {
    pub base: Point,
    pub vs: f32,
    pub density: f32,
}

#[derive(Debug)]
pub struct SrfMetadataV2<S = Vec<f32>> {
    pub base: SrfMetadata<S>,
    pub vs: S,
    pub density: S,
}

pub type SrfMetadataV2View<'a> = SrfMetadataV2<&'a [f32]>;

impl SrfMetadataV2 {
    pub fn with_capacity(n: usize) -> Self {
        SrfMetadataV2 {
            base: SrfMetadata::with_capacity(n),
            vs: Vec::with_capacity(n),
            density: Vec::with_capacity(n),
        }
    }

    pub fn push(&mut self, point: &PointV2) {
        self.base.push(&point.base);
        self.vs.push(point.vs);
        self.density.push(point.density);
    }
}

impl<S: AsRef<[f32]>> SrfMetadataV2<S> {
    pub fn iter(&self) -> PointV2Iter<'_> {
        PointV2Iter {
            points: self.base.iter(),
            vs: self.vs.as_ref(),
            density: self.density.as_ref(),
        }
    }
}

pub struct PointV2Iter<'a> {
    points: PointIter<'a>,
    vs: &'a [f32],
    density: &'a [f32],
}

impl Iterator for PointV2Iter<'_> {
    type Item = PointV2;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.points.index;
        let base = self.points.next()?;
        Some(PointV2 {
            base,
            vs: self.vs[i],
            density: self.density[i],
        })
    }
}

impl<'py> IntoPyObject<'py> for SrfMetadataV2 {
    type Target = PySrfMetadata;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let base = self.base.into_pyobject(py)?;
        {
            let mut base_ref = base.borrow_mut();
            base_ref.vs = Some(PyArray1::from_vec(py, self.vs).unbind());
            base_ref.density = Some(PyArray1::from_vec(py, self.density).unbind());
        }
        Ok(base)
    }
}

#[derive(Debug)]
pub enum SrfMetadataVersioned<S = Vec<f32>> {
    V1(SrfMetadata<S>),
    V2(SrfMetadataV2<S>),
}

impl<'py> IntoPyObject<'py> for SrfMetadataVersioned {
    type Target = PySrfMetadata;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Self::V1(metadata) => metadata.into_pyobject(py),
            Self::V2(metadata) => metadata.into_pyobject(py),
        }
    }
}

#[derive(Debug)]
pub struct SrfFile<S = Vec<f32>, R = Vec<usize>> {
    pub planes: Vec<SrfPlane>,
    pub metadata: SrfMetadataVersioned<S>,
    pub slipt1: CsrMatrix<R, S>,
}

pub type SrfFileView<'a> = SrfFile<&'a [f32], &'a [usize]>;

impl<'py> IntoPyObject<'py> for SrfFile {
    type Target = PySrfFile;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let mut planes: Vec<Py<SrfPlane>> = Vec::with_capacity(self.planes.len());
        for plane in self.planes {
            planes.push(plane.into_pyobject(py)?.unbind());
        }
        let metadata = self.metadata.into_pyobject(py)?.unbind();
        let slipt1 = self.slipt1.into_pyobject(py)?.unbind();
        Ok(Py::new(
            py,
            PySrfFile {
                planes,
                metadata,
                slipt1,
            },
        )?
        .into_bound(py))
    }
}
