use crate::types::{
    CsrMatrix, Point, SrfFile, SrfMetadata, SrfMetadataV2, SrfMetadataVersioned, SrfPlane,
};
use std::io::{self, Write};

use lexical_core::{BUFFER_SIZE, ToLexical};
use thiserror::Error;

/// Why an SRF wasn't written: the output failed, or the input isn't an SRF.
#[derive(Debug, Error)]
pub enum WriteError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("the point metadata columns disagree on how many points there are")]
    RaggedMetadata,
    #[error("PLANE headers expect {expected} total points but the metadata has {points}")]
    PlanePoints { expected: usize, points: usize },
    #[error("slipt1 has {rows} rows but the metadata has {points} points")]
    SlipRows { rows: usize, points: usize },
    #[error("slipt1 is not a CSR matrix: {0}")]
    SlipShape(&'static str),
    #[error("slipt1 row {point} has column indices that don't strictly increase")]
    SlipColumns { point: usize },
    #[error("Points written exceed the {expected} declared by the plane headers")]
    TooMany { expected: usize },
    #[error("Points written ({written}) do not match the {expected} declared by the plane headers")]
    TooFew { written: usize, expected: usize },
}

type Result<T> = std::result::Result<T, WriteError>;

/// Refuse a slip-rate matrix that doesn't have one well-formed row per point.
///
/// [`CsrMatrix::rows`] relies on this, so a malformed matrix from Python is a
/// `ValueError` and not a panic.
fn check_slip<R: AsRef<[usize]>, D: AsRef<[f32]>>(
    slipt1: &CsrMatrix<R, D>,
    points: usize,
) -> Result<()> {
    let (row_ptr, data, indices) = (
        slipt1.row_ptr.as_ref(),
        slipt1.data.as_ref(),
        slipt1.indices.as_ref(),
    );
    let rows = row_ptr.len().saturating_sub(1);
    if rows != points {
        return Err(WriteError::SlipRows { rows, points });
    }
    if row_ptr.first().is_some_and(|&first| first != 0)
        || !row_ptr.is_sorted()
        || row_ptr.last().copied().unwrap_or(0) != data.len()
    {
        return Err(WriteError::SlipShape(
            "row_ptr must start at 0, never decrease, and end at the number of values",
        ));
    }
    if !indices.is_empty() && indices.len() != data.len() {
        return Err(WriteError::SlipShape(
            "indices must be empty or hold one column per value",
        ));
    }
    Ok(())
}

/// The samples an SRF stores for one pulse: one per column from zero to its last
/// stored column, with the columns the row skips written as zeros.
///
/// An SRF records only where a pulse starts, so a skipped column must be written out
/// or every later sample shifts earlier on re-read. A row without columns is dense.
fn pulse<'a>(
    point: usize,
    values: &'a [f32],
    columns: Option<&'a [usize]>,
) -> Result<impl ExactSizeIterator<Item = f32> + 'a> {
    let column = move |stored: usize| columns.map_or(stored, |columns| columns[stored]);
    if columns.is_some_and(|columns| !columns.is_sorted_by(|a, b| a < b)) {
        return Err(WriteError::SlipColumns { point });
    }
    let length = values
        .len()
        .checked_sub(1)
        .map_or(0, |last| column(last) + 1);
    let mut stored = 0;
    Ok((0..length).map(move |at| {
        if stored < values.len() && column(stored) == at {
            stored += 1;
            values[stored - 1]
        } else {
            0.0
        }
    }))
}

fn lexical_write<W: Write, T: ToLexical>(
    writer: &mut W,
    value: T,
    buffer: &mut [u8],
) -> Result<()> {
    let slice = lexical_core::write(value, buffer);
    writer.write_all(slice)?;
    Ok(())
}

const POINTS: &[u8] = b"POINTS ";
const EMPTY_SLIP_TAIL: &[u8] = b" 0.0 0 0.0 0";

/// The columns of a point's first line, space-separated, in the SRF's order.
///
/// `rake`, `slip1` and `rise` aren't here: they open the *second* line, which
/// `write_slip_row` writes.
fn write_point<W: Write>(writer: &mut W, point: &Point, buffer: &mut [u8]) -> Result<()> {
    let columns = [
        point.lon,
        point.lat,
        point.dep,
        point.stk,
        point.dip,
        point.area,
        point.tinit,
        point.dt,
    ];
    for (index, value) in columns.into_iter().enumerate() {
        if index > 0 {
            writer.write_all(b" ")?;
        }
        lexical_write(writer, value, buffer)?;
    }
    Ok(())
}

// Writes "rake slip1 nt1 0.0 0 0.0 0" followed by the nt1 slipt1 values on
// the next line, matching the record layout read_slip_row expects. slip2/nt2
// and slip3/nt3 are always written empty.
fn write_slip_row<W: Write>(
    writer: &mut W,
    point: &Point,
    samples: impl ExactSizeIterator<Item = f32>,
    buffer: &mut [u8],
) -> Result<()> {
    lexical_write(writer, point.rake, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(writer, point.slip1, buffer)?;
    writer.write_all(b" ")?;
    lexical_write(writer, samples.len(), buffer)?;
    writer.write_all(EMPTY_SLIP_TAIL)?;

    for (i, value) in samples.enumerate() {
        writer.write_all(if i == 0 { b"\n" } else { b" " })?;
        lexical_write(writer, value, buffer)?;
    }
    Ok(())
}

fn write_srf_points_v1<W: Write, S: AsRef<[f32]>, R: AsRef<[usize]>, D: AsRef<[f32]>>(
    writer: &mut W,
    metadata: &SrfMetadata<S>,
    slipt1: &CsrMatrix<R, D>,
) -> Result<()> {
    let mut buffer = [0u8; BUFFER_SIZE];
    writer.write_all(POINTS)?;
    lexical_write(writer, metadata.iter().len(), &mut buffer)?;
    writer.write_all(b"\n")?;

    for (index, (point, (values, columns))) in metadata.iter().zip(slipt1.rows()).enumerate() {
        let samples = pulse(index, values, columns)?;
        write_point(writer, &point, &mut buffer)?;
        writer.write_all(b"\n")?;
        write_slip_row(writer, &point, samples, &mut buffer)?;
        writer.write_all(b"\n")?;
    }

    Ok(())
}

/// Writes the points of a version 2 SRF in chunks of any size.
pub struct PointStream {
    counts: Vec<usize>,
    plane: usize,
    left: usize,
    written: usize,
}

impl PointStream {
    #[must_use]
    pub fn new(planes: &[SrfPlane]) -> Self {
        Self {
            counts: planes.iter().map(SrfPlane::points).collect(),
            plane: 0,
            left: 0,
            written: 0,
        }
    }

    fn expected(&self) -> usize {
        self.counts.iter().sum()
    }

    /// Write the `POINTS` line of the next plane.
    fn open_plane<W: Write>(&mut self, writer: &mut W, buffer: &mut [u8]) -> Result<()> {
        let count = self.counts[self.plane];
        writer.write_all(POINTS)?;
        lexical_write(writer, count, buffer)?;
        writer.write_all(b"\n")?;
        self.plane += 1;
        self.left = count;
        Ok(())
    }

    /// Write the next points.
    ///
    /// # Errors
    ///
    /// If the metadata is ragged, the matrix doesn't have one well-formed row per
    /// point, the points exceed those the plane headers declare, or writing fails.
    pub fn write<W: Write, S: AsRef<[f32]>, R: AsRef<[usize]>, D: AsRef<[f32]>>(
        &mut self,
        writer: &mut W,
        metadata: &SrfMetadataV2<S>,
        slipt1: &CsrMatrix<R, D>,
    ) -> Result<()> {
        let points = metadata.points().ok_or(WriteError::RaggedMetadata)?;
        check_slip(slipt1, points)?;
        let mut buffer = [0u8; BUFFER_SIZE];
        for (point, (values, columns)) in metadata.iter().zip(slipt1.rows()) {
            let samples = pulse(self.written, values, columns)?;
            while self.left == 0 {
                if self.plane == self.counts.len() {
                    return Err(WriteError::TooMany {
                        expected: self.expected(),
                    });
                }
                self.open_plane(writer, &mut buffer)?;
            }
            write_point(writer, &point.base, &mut buffer)?;
            writer.write_all(b" ")?;
            lexical_write(writer, point.vs, &mut buffer)?;
            writer.write_all(b" ")?;
            lexical_write(writer, point.density, &mut buffer)?;
            writer.write_all(b"\n")?;
            write_slip_row(writer, &point.base, samples, &mut buffer)?;
            writer.write_all(b"\n")?;
            self.left -= 1;
            self.written += 1;
        }
        Ok(())
    }

    /// Check the stream wrote every declared point, then write the `POINTS` lines of
    /// any trailing planes with no points.
    ///
    /// # Errors
    ///
    /// If the stream wrote fewer points than the plane headers declare, or writing
    /// fails.
    pub fn finish<W: Write>(&mut self, writer: &mut W) -> Result<()> {
        let expected = self.expected();
        if self.written != expected {
            return Err(WriteError::TooFew {
                written: self.written,
                expected,
            });
        }
        let mut buffer = [0u8; BUFFER_SIZE];
        while self.plane < self.counts.len() {
            self.open_plane(writer, &mut buffer)?;
        }
        Ok(())
    }
}

fn write_srf_points_v2<W: Write, S: AsRef<[f32]>, R: AsRef<[usize]>, D: AsRef<[f32]>>(
    writer: &mut W,
    planes: &[SrfPlane],
    metadata: &SrfMetadataV2<S>,
    slipt1: &CsrMatrix<R, D>,
) -> Result<()> {
    let mut stream = PointStream::new(planes);
    stream.write(writer, metadata, slipt1)?;
    stream.finish(writer)?;
    Ok(())
}

/// The version line and the plane header of a version 2 SRF: everything that
/// comes before the first point.
///
/// # Errors
///
/// If writing fails.
pub fn write_header_v2<W: Write>(writer: &mut W, planes: &[SrfPlane]) -> io::Result<()> {
    writer.write_all(VERSION_2)?;
    write_plane_header(writer, planes)
}

fn write_plane_header<W: Write>(writer: &mut W, planes: &[SrfPlane]) -> io::Result<()> {
    writeln!(writer, "PLANE {}", planes.len())?;
    for plane in planes {
        writeln!(
            writer,
            "{} {} {} {} {} {}",
            plane.elon, plane.elat, plane.nstk, plane.ndip, plane.len, plane.wid,
        )?;
        writeln!(
            writer,
            "{} {} {} {} {}",
            plane.stk, plane.dip, plane.dtop, plane.shyp, plane.dhyp
        )?;
    }
    Ok(())
}

const VERSION_1: &[u8] = b"1.0\n";
const VERSION_2: &[u8] = b"2.0\n";

fn write_version<W: Write, S>(
    writer: &mut W,
    metadata: &SrfMetadataVersioned<S>,
) -> io::Result<()> {
    writer.write_all(match metadata {
        SrfMetadataVersioned::V1(_) => VERSION_1,
        SrfMetadataVersioned::V2(_) => VERSION_2,
    })
}

/// Write a whole SRF, refusing one that isn't consistent before writing a byte.
///
/// # Errors
///
/// If the metadata is ragged, the plane headers declare a different number of points,
/// the matrix doesn't have one well-formed row per point, or writing fails.
pub fn write_srf<W: Write, S: AsRef<[f32]>, R: AsRef<[usize]>>(
    writer: &mut W,
    srf_file: &SrfFile<S, R>,
) -> Result<()> {
    let points = match &srf_file.metadata {
        SrfMetadataVersioned::V1(metadata) => metadata.points(),
        SrfMetadataVersioned::V2(metadata) => metadata.points(),
    }
    .ok_or(WriteError::RaggedMetadata)?;
    let expected = srf_file.planes.iter().map(SrfPlane::points).sum();
    if expected != points {
        return Err(WriteError::PlanePoints { expected, points });
    }
    check_slip(&srf_file.slipt1, points)?;
    write_version(writer, &srf_file.metadata)?;
    write_plane_header(writer, &srf_file.planes)?;
    match &srf_file.metadata {
        SrfMetadataVersioned::V1(metadata) => {
            write_srf_points_v1(writer, metadata, &srf_file.slipt1)
        }
        SrfMetadataVersioned::V2(metadata) => {
            write_srf_points_v2(writer, &srf_file.planes, metadata, &srf_file.slipt1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::srf_parser::read_srf;
    use crate::types::{CsrMatrixView, SrfFileView, SrfMetadataVersioned};

    const SRF_V1: &[u8] = b"1.0\n\
PLANE 1\n\
0.0 0.0 2 1 4.0 2.0\n\
90.0 45.0 0.0 0.0 1.0\n\
POINTS 2\n\
0.1 -43.0 5.0 90.0 45.0 1.0e10 0.5 0.1\n\
30.0 1.5 3 0.0 0 0.0 0\n\
0.1 0.2 0.3\n\
0.2 -43.1 5.5 90.0 45.0 1.0e10 0.6 0.1\n\
45.0 2.0 2 0.0 0 0.0 0\n\
0.4 0.5\n";

    const SRF_V2_TWO_PLANES: &[u8] = b"2.0\n\
PLANE 2\n\
0.0 0.0 1 1 4.0 2.0\n\
90.0 45.0 0.0 0.0 1.0\n\
0.5 0.5 1 1 4.0 2.0\n\
90.0 45.0 0.0 0.0 1.0\n\
POINTS 1\n\
0.1 -43.0 5.0 90.0 45.0 1.0e10 0.5 0.1 3.5 2.7\n\
30.0 1.5 2 0.0 0 0.0 0\n\
0.1 0.2\n\
POINTS 1\n\
0.2 -43.1 5.5 90.0 45.0 1.0e10 0.6 0.1 3.6 2.8\n\
45.0 2.0 1 0.0 0 0.0 0\n\
0.4\n";

    fn parse(data: &[u8]) -> SrfFile {
        read_srf(data).unwrap()
    }

    fn write_to_vec<S: AsRef<[f32]>, R: AsRef<[usize]>>(srf: &SrfFile<S, R>) -> Vec<u8> {
        let mut out = Vec::new();
        write_srf(&mut out, srf).unwrap();
        out
    }

    #[test]
    fn v1_roundtrips_through_parser() {
        let srf = parse(SRF_V1);
        let reparsed = parse(&write_to_vec(&srf));

        assert_eq!(reparsed.planes.len(), 1);
        let (SrfMetadataVersioned::V1(metadata), SrfMetadataVersioned::V1(reparsed_metadata)) =
            (&srf.metadata, &reparsed.metadata)
        else {
            panic!("expected V1 metadata")
        };
        assert_eq!(metadata.lon, reparsed_metadata.lon);
        assert_eq!(metadata.lat, reparsed_metadata.lat);
        assert_eq!(metadata.dep, reparsed_metadata.dep);
        assert_eq!(metadata.area, reparsed_metadata.area);
        assert_eq!(metadata.tinit, reparsed_metadata.tinit);
        assert_eq!(metadata.rake, reparsed_metadata.rake);
        assert_eq!(metadata.slip1, reparsed_metadata.slip1);
        assert_eq!(metadata.rise, reparsed_metadata.rise);
        assert_eq!(srf.slipt1.row_ptr, reparsed.slipt1.row_ptr);
        assert_eq!(srf.slipt1.data, reparsed.slipt1.data);
    }

    #[test]
    fn v2_multi_plane_roundtrips_through_parser() {
        let srf = parse(SRF_V2_TWO_PLANES);
        let reparsed = parse(&write_to_vec(&srf));

        assert_eq!(reparsed.planes.len(), 2);
        let (SrfMetadataVersioned::V2(metadata), SrfMetadataVersioned::V2(reparsed_metadata)) =
            (&srf.metadata, &reparsed.metadata)
        else {
            panic!("expected V2 metadata")
        };
        assert_eq!(metadata.base.lon, reparsed_metadata.base.lon);
        assert_eq!(metadata.base.rake, reparsed_metadata.base.rake);
        assert_eq!(metadata.vs, reparsed_metadata.vs);
        assert_eq!(metadata.density, reparsed_metadata.density);
        assert_eq!(srf.slipt1.row_ptr, reparsed.slipt1.row_ptr);
        assert_eq!(srf.slipt1.data, reparsed.slipt1.data);
    }

    #[test]
    fn borrowed_view_writes_same_bytes_as_owned() {
        let srf = parse(SRF_V1);
        let metadata = match &srf.metadata {
            SrfMetadataVersioned::V1(metadata) => metadata,
            SrfMetadataVersioned::V2(_) => panic!("expected V1 metadata"),
        };

        let view: SrfFileView = SrfFile {
            planes: srf.planes.clone(),
            metadata: SrfMetadataVersioned::V1(SrfMetadata {
                lon: &metadata.lon,
                lat: &metadata.lat,
                dep: &metadata.dep,
                stk: &metadata.stk,
                dip: &metadata.dip,
                area: &metadata.area,
                tinit: &metadata.tinit,
                dt: &metadata.dt,
                rake: &metadata.rake,
                slip1: &metadata.slip1,
                rise: &metadata.rise,
            }),
            slipt1: CsrMatrixView {
                row_ptr: &srf.slipt1.row_ptr,
                indices: &srf.slipt1.indices,
                data: &srf.slipt1.data,
            },
        };

        assert_eq!(write_to_vec(&view), write_to_vec(&srf));
    }

    /// Borrowed columns for a run of points, with the run's rows re-based to zero.
    type Chunk<'a> = (SrfMetadataV2<&'a [f32]>, CsrMatrix<Vec<usize>, &'a [f32]>);

    /// The points of `SRF_V2_TWO_PLANES`, from `start` to `stop`, as borrowed views.
    fn chunk<'a>(
        metadata: &'a SrfMetadataV2,
        slipt1: &'a CsrMatrix,
        start: usize,
        stop: usize,
    ) -> Chunk<'a> {
        let base = &metadata.base;
        let rows = &slipt1.row_ptr[start..=stop];
        let view = SrfMetadataV2 {
            base: SrfMetadata {
                lon: &base.lon[start..stop],
                lat: &base.lat[start..stop],
                dep: &base.dep[start..stop],
                stk: &base.stk[start..stop],
                dip: &base.dip[start..stop],
                area: &base.area[start..stop],
                tinit: &base.tinit[start..stop],
                dt: &base.dt[start..stop],
                rake: &base.rake[start..stop],
                slip1: &base.slip1[start..stop],
                rise: &base.rise[start..stop],
            },
            vs: &metadata.vs[start..stop],
            density: &metadata.density[start..stop],
        };
        let csr = CsrMatrix {
            row_ptr: rows.iter().map(|offset| offset - rows[0]).collect(),
            indices: Vec::new(),
            data: &slipt1.data[rows[0]..rows[stop - start]],
        };
        (view, csr)
    }

    fn v2(srf: &SrfFile) -> &SrfMetadataV2 {
        match &srf.metadata {
            SrfMetadataVersioned::V2(metadata) => metadata,
            SrfMetadataVersioned::V1(_) => panic!("expected V2 metadata"),
        }
    }

    #[test]
    fn a_stream_in_chunks_writes_the_whole_file() {
        let srf = parse(SRF_V2_TWO_PLANES);
        let metadata = v2(&srf);
        let mut streamed = Vec::new();
        write_header_v2(&mut streamed, &srf.planes).unwrap();
        let mut stream = PointStream::new(&srf.planes);
        // An empty chunk, then one point, then the other: the plane boundary falls
        // between chunks rather than inside one.
        for (start, stop) in [(0, 0), (0, 1), (1, 2)] {
            let (view, csr) = chunk(metadata, &srf.slipt1, start, stop);
            stream.write(&mut streamed, &view, &csr).unwrap();
        }
        stream.finish(&mut streamed).unwrap();
        assert_eq!(streamed, write_to_vec(&srf));
    }

    #[test]
    fn a_stream_refuses_points_the_planes_do_not_declare() {
        let srf = parse(SRF_V2_TWO_PLANES);
        let metadata = v2(&srf);
        let (view, csr) = chunk(metadata, &srf.slipt1, 0, 2);
        let mut stream = PointStream::new(&srf.planes[..1]);
        let error = stream.write(&mut Vec::new(), &view, &csr).unwrap_err();
        assert!(matches!(error, WriteError::TooMany { expected: 1 }));
    }

    #[test]
    fn a_stream_refuses_to_finish_short() {
        let srf = parse(SRF_V2_TWO_PLANES);
        let metadata = v2(&srf);
        let (view, csr) = chunk(metadata, &srf.slipt1, 0, 1);
        let mut stream = PointStream::new(&srf.planes);
        stream.write(&mut Vec::new(), &view, &csr).unwrap();
        let mut tail = Vec::new();
        let error = stream.finish(&mut tail).unwrap_err();
        assert!(matches!(
            error,
            WriteError::TooFew {
                written: 1,
                expected: 2
            }
        ));
        assert_eq!(tail, Vec::<u8>::new());
    }

    /// A single point, whose pulse is whatever `indices` and `data` make it.
    fn one_point(indices: Vec<usize>, data: Vec<f32>) -> SrfFile {
        let mut srf = parse(
            b"1.0\n\
PLANE 1\n\
0.0 0.0 1 1 4.0 2.0\n\
90.0 45.0 0.0 0.0 1.0\n\
POINTS 1\n\
0.1 -43.0 5.0 90.0 45.0 1.0e10 0.5 0.1\n\
30.0 1.5 0 0.0 0 0.0 0\n",
        );
        srf.slipt1 = CsrMatrix {
            row_ptr: vec![0, data.len()],
            indices,
            data,
        };
        srf
    }

    /// The bug this guards, from `source_modelling`: the writer dropped the column
    /// indices, so a row without its zeros stored (what `scipy.sparse` leaves) came
    /// back shorter, with every later sample moved earlier.
    #[test]
    fn skipped_columns_are_written_as_zeros() {
        let srf = one_point(vec![1, 3], vec![1.0, 2.0]);
        let reparsed = parse(&write_to_vec(&srf));
        assert_eq!(reparsed.slipt1.data, vec![0.0, 1.0, 0.0, 2.0]);
        assert_eq!(reparsed.slipt1.indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn a_row_without_columns_is_dense() {
        let srf = one_point(Vec::new(), vec![1.0, 2.0]);
        let reparsed = parse(&write_to_vec(&srf));
        assert_eq!(reparsed.slipt1.data, vec![1.0, 2.0]);
    }

    #[test]
    fn unsorted_or_repeated_columns_are_refused() {
        for columns in [vec![3, 1], vec![2, 2]] {
            let error = write_srf(&mut Vec::new(), &one_point(columns, vec![1.0, 2.0]));
            assert!(matches!(error, Err(WriteError::SlipColumns { point: 0 })));
        }
    }

    #[test]
    fn a_malformed_matrix_is_refused_not_a_panic() {
        let mut srf = one_point(Vec::new(), vec![1.0, 2.0]);
        srf.slipt1.row_ptr = vec![0, 5];
        let error = write_srf(&mut Vec::new(), &srf);
        assert!(matches!(error, Err(WriteError::SlipShape(_))));
        let mut srf = one_point(vec![0], vec![1.0, 2.0]);
        srf.slipt1.indices = vec![0];
        let error = write_srf(&mut Vec::new(), &srf);
        assert!(matches!(error, Err(WriteError::SlipShape(_))));
    }

    #[test]
    fn inconsistent_metadata_is_refused_before_a_byte_is_written() {
        let mut srf = parse(SRF_V1);
        if let SrfMetadataVersioned::V1(metadata) = &mut srf.metadata {
            metadata.rake.pop();
        }
        let mut out = Vec::new();
        assert!(matches!(
            write_srf(&mut out, &srf),
            Err(WriteError::RaggedMetadata)
        ));
        assert!(out.is_empty());

        let mut srf = parse(SRF_V1);
        srf.planes[0].nstk = 3;
        assert!(matches!(
            write_srf(&mut Vec::new(), &srf),
            Err(WriteError::PlanePoints {
                expected: 3,
                points: 2
            })
        ));
    }

    #[test]
    fn empty_slip_row_roundtrips() {
        let data = b"1.0\n\
PLANE 1\n\
0.0 0.0 1 1 4.0 2.0\n\
90.0 45.0 0.0 0.0 1.0\n\
POINTS 1\n\
0.1 -43.0 5.0 90.0 45.0 1.0e10 0.5 0.1\n\
30.0 1.5 0 0.0 0 0.0 0\n";
        let srf = parse(data);
        let reparsed = parse(&write_to_vec(&srf));
        assert_eq!(srf.slipt1.row_ptr, reparsed.slipt1.row_ptr);
        assert_eq!(reparsed.slipt1.data, Vec::<f32>::new());
    }
}
