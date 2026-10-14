//! The SRF grammar, as nom parsers over the file's bytes.
//!
//! Numbers go through [`number`], which hands the stream straight to
//! `lexical_core::parse_partial`: it recognises and converts in one pass, where nom's
//! `float` recognises, checks UTF-8 and then parses again through `str::parse`. On
//! 6.9 million numbers that was 0.113 s against 0.268 s, and number conversion is most
//! of a parse.
//!
//! A pulse's samples stream from [`number`] straight into the CSR matrix, so a
//! multi-gigabyte file is never also held as per-pulse temporaries.

#![expect(
    clippy::similar_names,
    reason = "the vocabulary is the SRF format's -- slip/slipt1, srf/srf_file -- and \
              renaming for distinctness would move it away from the spec"
)]

use nom::bytes::complete::tag;
use nom::character::complete::{line_ending, multispace0, not_line_ending, space0};
use nom::error::{ContextError, ErrorKind, ParseError, context};
use nom::multi::{count, many0_count};
use nom::sequence::{preceded, terminated};
use nom::{IResult, Parser};
use thiserror::Error;

use crate::types::{
    CsrMatrix, Point, PointV2, SrfFile, SrfMetadata, SrfMetadataV2, SrfMetadataVersioned, SrfPlane,
};

#[derive(Debug, PartialEq, Eq)]
pub enum Version {
    V1,
    V2,
}

#[derive(Debug, Error)]
pub enum SrfParseError {
    #[error("line {line}:{column}: expected {expected}, found \"{found}\"")]
    Syntax {
        line: usize,
        column: usize,
        expected: &'static str,
        found: String,
    },
    #[error("unknown SRF version: {0}")]
    UnknownVersion(String),
    #[error("PLANE headers expect {expected} total points but POINTS declares {declared}")]
    PointCountMismatch { declared: usize, expected: usize },
    #[error("plane {plane} expects {expected} points but its POINTS block declares {declared}")]
    PlanePointCountMismatch {
        plane: usize,
        declared: usize,
        expected: usize,
    },
    #[error("SRF parser does not support slip{0} array ")]
    UnsupportedSlipArray(usize),
}

/// Why a parser stopped: the bytes it stopped at and what it wanted there, or a file
/// that is well formed but breaks the format's rules.
#[derive(Debug)]
enum Fault<'a> {
    Syntax {
        at: &'a [u8],
        expected: &'static str,
    },
    // Boxed so every number read, millions per file, returns a small `Result`.
    Srf(Box<SrfParseError>),
}

impl<'a> ParseError<&'a [u8]> for Fault<'a> {
    /// Every leaf parser names what it wanted through `context`, so this is only the
    /// fallback for one that doesn't.
    fn from_error_kind(at: &'a [u8], _: ErrorKind) -> Self {
        Fault::Syntax {
            at,
            expected: "valid SRF syntax",
        }
    }

    fn append(_: &'a [u8], _: ErrorKind, other: Self) -> Self {
        other
    }
}

impl<'a> ContextError<&'a [u8]> for Fault<'a> {
    fn add_context(_: &'a [u8], expected: &'static str, other: Self) -> Self {
        match other {
            Fault::Syntax { at, .. } => Fault::Syntax { at, expected },
            srf @ Fault::Srf(_) => srf,
        }
    }
}

impl Fault<'_> {
    /// The public error, with a syntax fault's position worked out from what's left.
    ///
    /// Counted here rather than tracked during the parse: this is the error path, which
    /// runs once.
    #[expect(
        clippy::naive_bytecount,
        reason = "the error path, once per parse; a bytecount dependency for that is \
                  more than it is worth"
    )]
    fn into_error(self, input: &[u8]) -> SrfParseError {
        match self {
            Fault::Srf(error) => *error,
            Fault::Syntax { at, expected } => {
                let consumed = &input[..input.len() - at.len()];
                let line = consumed.iter().filter(|&&c| c == b'\n').count() + 1;
                let column = consumed
                    .iter()
                    .rposition(|&c| c == b'\n')
                    .map_or(consumed.len() + 1, |newline| consumed.len() - newline);
                let token = at.split(u8::is_ascii_whitespace).next().unwrap_or_default();
                SrfParseError::Syntax {
                    line,
                    column,
                    expected,
                    found: if at.is_empty() {
                        "end of input".to_owned()
                    } else {
                        String::from_utf8_lossy(&token[..token.len().min(32)]).into_owned()
                    },
                }
            }
        }
    }
}

type Parsed<'a, T> = IResult<&'a [u8], T, Fault<'a>>;

/// Stop the parse on a rule the format breaks, which no alternative could rescue.
fn refuse<'a, T>(error: SrfParseError) -> Parsed<'a, T> {
    Err(nom::Err::Failure(Fault::Srf(Box::new(error))))
}

/// One number, after any whitespace, read by `lexical_core` straight off the stream.
#[expect(
    clippy::inline_always,
    reason = "measured: without it a 234 MiB file parses at 323 MiB/s, with it 346"
)]
#[inline(always)]
fn number<T: lexical_core::FromLexical>(input: &[u8]) -> Parsed<'_, T> {
    let input = input.trim_ascii_start();
    match lexical_core::parse_partial(input) {
        Ok((value, read)) => Ok((&input[read..], value)),
        Err(_) => Err(nom::Err::Error(Fault::Syntax {
            at: input,
            expected: "a number",
        })),
    }
}

/// The rest of a line, which must be blank, and its newline.
fn end_of_line(input: &[u8]) -> Parsed<'_, ()> {
    context("the end of the line", preceded(space0, line_ending))
        .map(|_| ())
        .parse(input)
}

/// An exact keyword, after any whitespace.
fn keyword<'a>(word: &'static str) -> impl Parser<&'a [u8], Output = (), Error = Fault<'a>> {
    context(word, preceded(multispace0, tag(word.as_bytes()))).map(|_| ())
}

fn version(input: &[u8]) -> Parsed<'_, Version> {
    let (rest, line) =
        context("a version line", terminated(not_line_ending, line_ending)).parse(input)?;
    match line.trim_ascii() {
        b"1.0" => Ok((rest, Version::V1)),
        b"2.0" => Ok((rest, Version::V2)),
        other => refuse(SrfParseError::UnknownVersion(
            String::from_utf8_lossy(other).into_owned(),
        )),
    }
}

/// Comment lines, which a version 2 file may have between its version and `PLANE`.
fn comments(input: &[u8]) -> Parsed<'_, ()> {
    many0_count(preceded(
        multispace0,
        (tag(&b"#"[..]), not_line_ending, line_ending),
    ))
    .map(|_| ())
    .parse(input)
}

fn plane(input: &[u8]) -> Parsed<'_, SrfPlane> {
    let (input, (elon, elat, nstk, ndip, len, wid)) =
        (number, number, number, number, number, number).parse(input)?;
    let (input, ()) = end_of_line(input)?;
    let (input, (stk, dip, dtop, shyp, dhyp)) =
        (number, number, number, number, number).parse(input)?;
    Ok((
        input,
        SrfPlane {
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
        },
    ))
}

fn planes(input: &[u8]) -> Parsed<'_, Vec<SrfPlane>> {
    let (input, ()) = keyword("PLANE").parse(input)?;
    let (input, planes) = number::<usize>(input)?;
    count(plane, planes).parse(input)
}

// Slip values dominate SRF files, and EMOD3D writes them fixed-width (%13.5e
// style: 11 chars + separator). This is especially true of large magnitude
// ruptures with large slip histories.
const APPROX_BYTES_PER_SLIP_VALUE: usize = 12;

// Tiny PointHeader used to construct actual points later.
#[derive(Debug)]
struct PointHeader {
    lon: f32,
    lat: f32,
    dep: f32,
    stk: f32,
    dip: f32,
    area: f32,
    tinit: f32,
    dt: f32,
}

impl PointHeader {
    /// The header's eight columns, plus the three the slip line opens with.
    ///
    /// Both versions of the point block read the same header and then complete it the
    /// same way. Written out twice, the two copies were eleven fields each and nothing
    /// but reading them side by side would have caught one drifting.
    fn into_point(self, rake: f32, slip1: f32, rise: f32) -> Point {
        Point {
            lon: self.lon,
            lat: self.lat,
            dep: self.dep,
            stk: self.stk,
            dip: self.dip,
            area: self.area,
            tinit: self.tinit,
            dt: self.dt,
            rake,
            slip1,
            rise,
        }
    }
}

// Common point header values for both V1 and V2.
fn point_header(input: &[u8]) -> Parsed<'_, PointHeader> {
    (
        number, number, number, number, number, number, number, number,
    )
        .map(|(lon, lat, dep, stk, dip, area, tinit, dt)| PointHeader {
            lon,
            lat,
            dep,
            stk,
            dip,
            area,
            tinit,
            dt,
        })
        .parse(input)
}

/// The slip line's opening `rake slip1 nt1`, and the rise time `nt1` samples make.
fn slip_line(input: &[u8], dt: f32) -> Parsed<'_, (f32, f32, usize, f32)> {
    let (input, (rake, slip1, nt)) = (number, number, number::<usize>).parse(input)?;
    #[expect(
        clippy::cast_precision_loss,
        reason = "sample counts are far below 2^24; the format stores them as text"
    )]
    let rise = (nt as f32) * dt;
    Ok((input, (rake, slip1, nt, rise)))
}

/// One point's slip-rate pulse, streamed into the matrix.
///
/// The columns are relative to the pulse, and the onset stays the float already read
/// into the metadata. See `CsrMatrix::add_row`.
fn slip_row<'a>(input: &'a [u8], slipt1: &mut CsrMatrix, nt: usize) -> Parsed<'a, ()> {
    let (input, (_slip2, nt2, _slip3, nt3)) = (
        number::<f32>,
        number::<usize>,
        number::<f32>,
        number::<usize>,
    )
        .parse(input)?;
    if nt2 != 0 {
        return refuse(SrfParseError::UnsupportedSlipArray(2));
    }
    if nt3 != 0 {
        return refuse(SrfParseError::UnsupportedSlipArray(3));
    }
    let (mut rest, mut failure) = (input, None);
    let samples = std::iter::from_fn(|| match number::<f32>(rest) {
        Ok((after, sample)) => {
            rest = after;
            Some(sample)
        }
        Err(error) => {
            failure = Some(error);
            None
        }
    });
    slipt1.add_row(samples.take(nt));
    match failure {
        Some(error) => Err(error),
        None => Ok((rest, ())),
    }
}

fn points_v1<'a>(input: &'a [u8], planes: &[SrfPlane]) -> Parsed<'a, (SrfMetadata, CsrMatrix)> {
    let (mut input, ()) = keyword("POINTS").parse(input)?;
    let expected = planes.iter().map(SrfPlane::points).sum();
    let declared;
    (input, declared) = number(input)?;
    if declared != expected {
        return refuse(SrfParseError::PointCountMismatch { declared, expected });
    }
    let mut metadata = SrfMetadata::with_capacity(declared);
    let mut slipt1 = CsrMatrix::new(declared, input.len() / APPROX_BYTES_PER_SLIP_VALUE);

    for _ in 0..declared {
        let header;
        (input, header) = point_header(input)?;
        // technically the read_srf routines in EMOD3D don't need to have a
        // newline here, but requiring one distinguishes a mislabelled
        // version 1.0 SRF and a version 2.0 SRF because EOL conveniently
        // follows the point header (where-as SRF V2.0 has two vs/density still
        // to go):
        //
        // - read_srf (in the srf_subs.c versions): reads the floats with scanf(), which tolerates newlines.
        // - write_srf (same files): always writes a newline after the header (header + vs + density in SRF 2.0).
        (input, ()) = end_of_line(input)?;
        let (rake, slip1, nt, rise);
        (input, (rake, slip1, nt, rise)) = slip_line(input, header.dt)?;
        metadata.push(&header.into_point(rake, slip1, rise));
        (input, ()) = slip_row(input, &mut slipt1, nt)?;
    }
    Ok((input, (metadata, slipt1)))
}

fn points_v2<'a>(
    mut input: &'a [u8],
    planes: &[SrfPlane],
) -> Parsed<'a, (SrfMetadataV2, CsrMatrix)> {
    let total = planes.iter().map(SrfPlane::points).sum();
    let mut metadata = SrfMetadataV2::with_capacity(total);
    let mut slipt1 = CsrMatrix::new(total, input.len() / APPROX_BYTES_PER_SLIP_VALUE);

    for (plane, header) in planes.iter().enumerate() {
        (input, ()) = keyword("POINTS").parse(input)?;
        let declared;
        (input, declared) = number(input)?;
        if declared != header.points() {
            return refuse(SrfParseError::PlanePointCountMismatch {
                plane,
                declared,
                expected: header.points(),
            });
        }
        for _ in 0..declared {
            let (point, vs, density);
            (input, point) = point_header(input)?;
            (input, (vs, density)) = (number, number).parse(input)?;
            // Like the V1 parser, expecting EOL here helps detect mangled SRFs
            (input, ()) = end_of_line(input)?;
            let (rake, slip1, nt, rise);
            (input, (rake, slip1, nt, rise)) = slip_line(input, point.dt)?;
            metadata.push(&PointV2 {
                base: point.into_point(rake, slip1, rise),
                vs,
                density,
            });
            (input, ()) = slip_row(input, &mut slipt1, nt)?;
        }
    }
    Ok((input, (metadata, slipt1)))
}

fn srf(input: &[u8]) -> Parsed<'_, SrfFile> {
    let (input, version) = version(input)?;
    // Comments can only occur between the version indicator and the PLANE token.
    let (input, ()) = match version {
        Version::V1 => (input, ()),
        Version::V2 => comments(input)?,
    };
    let (input, planes) = planes(input)?;
    let (input, (metadata, slipt1)) = match version {
        Version::V1 => {
            let (input, (metadata, slipt1)) = points_v1(input, &planes)?;
            (input, (SrfMetadataVersioned::V1(metadata), slipt1))
        }
        Version::V2 => {
            let (input, (metadata, slipt1)) = points_v2(input, &planes)?;
            (input, (SrfMetadataVersioned::V2(metadata), slipt1))
        }
    };
    Ok((
        input,
        SrfFile {
            planes,
            metadata,
            slipt1,
        },
    ))
}

/// Parse a whole SRF.
///
/// # Errors
///
/// [`SrfParseError`] for bytes that aren't an SRF this parser reads, with the line and
/// column of a syntax error.
pub fn read_srf(input: &[u8]) -> Result<SrfFile, SrfParseError> {
    match srf(input) {
        Ok((_, file)) => Ok(file),
        Err(nom::Err::Error(fault) | nom::Err::Failure(fault)) => Err(fault.into_error(input)),
        Err(nom::Err::Incomplete(_)) => unreachable!("every parser here is complete"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    const SRF_V2: &[u8] = b"2.0\n\
PLANE 1\n\
0.0 0.0 2 1 4.0 2.0\n\
90.0 45.0 0.0 0.0 1.0\n\
POINTS 2\n\
0.1 -43.0 5.0 90.0 45.0 1.0e10 0.5 0.1 3.5 2.7\n\
30.0 1.5 3 0.0 0 0.0 0\n\
0.1 0.2 0.3\n\
0.2 -43.1 5.5 90.0 45.0 1.0e10 0.6 0.1 3.6 2.8\n\
45.0 2.0 2 0.0 0 0.0 0\n\
0.4 0.5\n";

    #[test]
    fn parses_v1() {
        let srf = read_srf(SRF_V1).unwrap();
        assert_eq!(srf.planes.len(), 1);
        let metadata = match &srf.metadata {
            SrfMetadataVersioned::V1(metadata) => metadata,
            SrfMetadataVersioned::V2(_) => panic!("expected V1 metadata"),
        };
        assert_eq!(metadata.lon, vec![0.1, 0.2]);
        assert_eq!(metadata.rake, vec![30.0, 45.0]);
        assert_eq!(metadata.rise, vec![3.0 * 0.1f32, 2.0 * 0.1f32]);
        assert_eq!(srf.slipt1.row_ptr, vec![0, 3, 5]);
        assert_eq!(srf.slipt1.data, vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    }

    #[test]
    fn parses_v2() {
        let srf = read_srf(SRF_V2).unwrap();
        let metadata = match &srf.metadata {
            SrfMetadataVersioned::V2(metadata) => metadata,
            SrfMetadataVersioned::V1(_) => panic!("expected V2 metadata"),
        };
        assert_eq!(metadata.base.lon, vec![0.1, 0.2]);
        assert_eq!(metadata.base.rake, vec![30.0, 45.0]);
        assert_eq!(metadata.vs, vec![3.5, 3.6]);
        assert_eq!(metadata.density, vec![2.7, 2.8]);
        assert_eq!(srf.slipt1.data, vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    }

    // Two 1x1 planes, each with its own POINTS block.
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

    #[test]
    fn rejects_unknown_version() {
        let err = read_srf(b"3.0\nPLANE 1\n").unwrap_err();
        assert!(matches!(err, SrfParseError::UnknownVersion(version) if version == "3.0"));
    }

    #[test]
    fn parses_v2_with_multiple_planes() {
        let srf = read_srf(SRF_V2_TWO_PLANES).unwrap();
        assert_eq!(srf.planes.len(), 2);
        let metadata = match &srf.metadata {
            SrfMetadataVersioned::V2(metadata) => metadata,
            SrfMetadataVersioned::V1(_) => panic!("expected V2 metadata"),
        };
        assert_eq!(metadata.base.lon, vec![0.1, 0.2]);
        assert_eq!(metadata.vs, vec![3.5, 3.6]);
        assert_eq!(metadata.density, vec![2.7, 2.8]);
        assert_eq!(srf.slipt1.row_ptr, vec![0, 2, 3]);
        assert_eq!(srf.slipt1.data, vec![0.1, 0.2, 0.4]);
    }

    const SRF_V1_TWO_PLANES: &[u8] = b"1.0\n\
PLANE 2\n\
0.0 0.0 1 1 4.0 2.0\n\
90.0 45.0 0.0 0.0 1.0\n\
0.5 0.5 1 1 4.0 2.0\n\
90.0 45.0 0.0 0.0 1.0\n\
POINTS 2\n\
0.1 -43.0 5.0 90.0 45.0 1.0e10 0.5 0.1\n\
30.0 1.5 2 0.0 0 0.0 0\n\
0.1 0.2\n\
0.2 -43.1 5.5 90.0 45.0 1.0e10 0.6 0.1\n\
45.0 2.0 1 0.0 0 0.0 0\n\
0.4\n";

    #[test]
    fn parses_v1_with_multiple_planes() {
        let srf = read_srf(SRF_V1_TWO_PLANES).unwrap();
        assert_eq!(srf.planes.len(), 2);
        let metadata = match &srf.metadata {
            SrfMetadataVersioned::V1(metadata) => metadata,
            SrfMetadataVersioned::V2(_) => panic!("expected V1 metadata"),
        };
        assert_eq!(metadata.lon, vec![0.1, 0.2]);
        assert_eq!(srf.slipt1.row_ptr, vec![0, 2, 3]);
        assert_eq!(srf.slipt1.data, vec![0.1, 0.2, 0.4]);
    }

    #[test]
    fn rejects_v1_point_count_mismatch() {
        // Plane declares 2x1 points but POINTS declares 3.
        let data = replace_once(SRF_V1, b"POINTS 2", b"POINTS 3");
        let err = read_srf(&data).unwrap_err();
        assert!(matches!(
            err,
            SrfParseError::PointCountMismatch {
                declared: 3,
                expected: 2,
            }
        ));
    }

    #[test]
    fn rejects_v2_plane_point_count_mismatch() {
        // Second plane is 1x1 but its POINTS block declares 2; the first
        // plane must parse cleanly before the mismatch can fail it.
        let data = replace_once(SRF_V2_TWO_PLANES, b"POINTS 1\n0.2", b"POINTS 2\n0.2");
        let err = read_srf(&data).unwrap_err();
        assert!(matches!(
            err,
            SrfParseError::PlanePointCountMismatch {
                plane: 1,
                declared: 2,
                expected: 1,
            }
        ));
    }

    #[test]
    fn truncated_file_never_panics() {
        // Truncating mid-token can still parse ("0.5" cut to "0." is a valid
        // float), and not every prefix errors. None may panic, though, and a cut
        // mid-record must error.
        for len in 0..SRF_V1.len() {
            let _ = read_srf(&SRF_V1[..len]);
        }
        assert!(read_srf(&SRF_V1[..SRF_V1.len() / 2]).is_err());
    }

    #[test]
    fn rejects_v1_data_mislabeled_as_v2() {
        // read_srf_points_v2 reads vs/density right after the point header,
        // so on v1 data it swallows the v1 line's rake/slip1 tokens and the
        // expect_end_of_line() check right after doesn't find a newline.
        let data = replace_once(SRF_V1, b"1.0\n", b"2.0\n");
        let err = read_srf(&data).unwrap_err();
        assert!(matches!(
            err,
            SrfParseError::Syntax {
                expected: "the end of the line",
                ..
            }
        ));
    }

    #[test]
    fn rejects_v2_data_mislabeled_as_v1() {
        // read_srf_points_v1 checks for end-of-line right after the point
        // header, but v2 data still has vs/density left on that same line.
        let data = replace_once(SRF_V2, b"2.0\n", b"1.0\n");
        let err = read_srf(&data).unwrap_err();
        assert!(matches!(
            err,
            SrfParseError::Syntax {
                expected: "the end of the line",
                ..
            }
        ));
    }

    #[test]
    fn version_line_tolerates_surrounding_whitespace() {
        let data = replace_once(SRF_V1, b"1.0\n", b"1.0 \r\n");
        assert!(read_srf(&data).is_ok());
    }

    #[test]
    fn a_syntax_error_names_its_line_column_and_token() {
        // The second point's latitude, on line 9 after the 4 characters of "0.2 ".
        let data = replace_once(SRF_V1, b"0.2 -43.1", b"0.2 abc");
        let err = read_srf(&data).unwrap_err();
        assert!(
            matches!(
                &err,
                SrfParseError::Syntax {
                    line: 9,
                    column: 5,
                    expected: "a number",
                    found,
                } if found == "abc"
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_short_pulse_is_a_syntax_error() {
        // The first pulse promises three samples and the line holds two, so the third
        // read lands on the next point's header and the counts no longer line up.
        let data = replace_once(SRF_V1, b"0.1 0.2 0.3\n", b"0.1 0.2\n");
        assert!(matches!(
            read_srf(&data).unwrap_err(),
            SrfParseError::Syntax { .. }
        ));
    }

    const SRF_BAD_PLANE_HEADER: &[u8] = b"0.0 0.0 2 1 4.0 2.0 90.0 45.0 0.0 0.0 1.0";

    #[test]
    fn plane_header_rejects_no_newline() {
        let err = plane(SRF_BAD_PLANE_HEADER).unwrap_err();
        let nom::Err::Error(fault) = err else {
            panic!("expected a recoverable error, got {err:?}");
        };
        let err = fault.into_error(SRF_BAD_PLANE_HEADER);
        assert!(matches!(
            err,
            SrfParseError::Syntax {
                expected: "the end of the line",
                ..
            }
        ));
    }

    /// Parse throughput, not correctness. `#[ignore]`d so the gate stays a gate.
    ///
    /// Run it with `cargo test -p srf --release parse_throughput`, passing `--ignored`
    /// and `--nocapture` to the test binary after the `--` separator.
    ///
    /// The number that matters is MB/s: callers hand this parser multi-gigabyte
    /// files, and a scanner change should land only if it keeps the throughput. Override the input with `SRF_THROUGHPUT_FILE`.
    #[test]
    #[ignore = "measures time, not behaviour"]
    fn parse_throughput() {
        const DEFAULT: &str = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/srfs/rupture_1.srf"
        );
        const RUNS: u32 = 5;

        let path = std::env::var("SRF_THROUGHPUT_FILE").unwrap_or_else(|_| DEFAULT.to_owned());
        let data = std::fs::read(&path).expect("throughput fixture is readable");

        let mut best = std::time::Duration::MAX;
        let mut points = 0;
        for _ in 0..RUNS {
            let start = std::time::Instant::now();
            let srf = read_srf(&data).expect("fixture parses");
            best = best.min(start.elapsed());
            points = match &srf.metadata {
                SrfMetadataVersioned::V1(metadata) => metadata.lon.len(),
                SrfMetadataVersioned::V2(metadata) => metadata.base.lon.len(),
            };
        }

        #[expect(
            clippy::cast_precision_loss,
            reason = "reporting a rate to three significant figures"
        )]
        let megabytes = data.len() as f64 / (1024.0 * 1024.0);
        let seconds = best.as_secs_f64();
        println!(
            "{path}: {megabytes:.1} MiB, {points} points, best of {RUNS} {seconds:.3} s \
             => {:.1} MiB/s",
            megabytes / seconds
        );
    }

    fn replace_once(data: &[u8], from: &[u8], to: &[u8]) -> Vec<u8> {
        let start = data
            .windows(from.len())
            .position(|window| window == from)
            .expect("pattern not found in fixture");
        let mut result = data[..start].to_vec();
        result.extend_from_slice(to);
        result.extend_from_slice(&data[start + from.len()..]);
        result
    }
}
