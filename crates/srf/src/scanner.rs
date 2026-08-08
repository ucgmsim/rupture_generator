//! The lexical layer: nom for the grammar, `lexical_core` for the numbers.
//!
//! Three things here are deliberate.
//!
//! - **nom does the scanning.** What used to be `iter().position(...)` followed by
//!   `self.index += x + 1` is now a combinator that returns the unconsumed rest. The
//!   off-by-one has nowhere to live.
//! - **The numbers stay `lexical_core`.** nom's `float` recognises the number, checks
//!   the bytes are UTF-8, then parses them again through `str::parse`;
//!   `lexical_core::parse_partial` does it in one pass. On the 6,893,172 numbers in
//!   `tests/srfs/rupture_1.srf` that is **0.113 s against 0.268 s** -- and the whole
//!   file parses in 0.177 s, so the leaf is most of the work and nom's would roughly
//!   double it. They agree bit for bit, both rounding correctly, so this is a cost
//!   question and not an accuracy one. `float_leaf_is_cheaper_through_lexical_core`
//!   measures both halves of that claim. The crate is here regardless because
//!   `srf_writer` formats with it.
//! - **The API is still a cursor.** `srf_parser` streams each point's pulse straight
//!   into a CSR matrix through repeated `next()`. Expressed as combinators it would
//!   have to collect every pulse into a `Vec` first, which for a multi-gigabyte file
//!   is the whole file in temporaries.
//!
//! Position is *computed* on failure rather than tracked. The old scanner counted the
//! newlines in every whitespace run it skipped, past every token, to maintain a line
//! number it only ever used in an error message.

use nom::Parser;
use nom::bytes::complete::{tag, take_until, take_while};
use nom::sequence::preceded;
use thiserror::Error;

/// One parser's result over SRF bytes: the unconsumed rest, and what was read.
type Parsed<'a, T> = nom::IResult<&'a [u8], T>;

const NEWLINE: &[u8] = b"\n";

#[derive(Debug, Error)]
pub enum ScannerError {
    #[error("line {line}:{column}: invalid number \"{source}\"")]
    InvalidNumber {
        source: lexical_core::Error,
        line: usize,
        column: usize,
    },
    #[error("line {line}:{column}: invalid token, expected: {expected}, found: \"{found}\"")]
    InvalidToken {
        expected: String,
        found: String,
        line: usize,
        column: usize,
    },
    #[error("line {line}:{column}: could not find newline.")]
    NoNewlineFound { line: usize, column: usize },
    #[error("unexpected end of input")]
    UnexpectedEof,
}

/// Everything before a token that is not part of it.
fn spaces(input: &[u8]) -> &[u8] {
    input.trim_ascii_start()
}

/// Blank space up to but not including the newline that must follow it.
fn blank_then_newline(input: &[u8]) -> Parsed<'_, &[u8]> {
    preceded(
        take_while(|c: u8| c.is_ascii_whitespace() && c != b'\n'),
        tag(NEWLINE),
    )
    .parse(input)
}

/// An exact keyword, and nothing about what may precede it.
fn keyword<'a>(input: &'a [u8], token: &[u8]) -> Parsed<'a, &'a [u8]> {
    tag(token).parse(input)
}

/// Everything up to the next newline, and the newline.
fn through_newline(input: &[u8]) -> Parsed<'_, &[u8]> {
    let (rest, line) = take_until(NEWLINE).parse(input)?;
    let (rest, _) = tag(NEWLINE).parse(rest)?;
    Ok((rest, line))
}

pub struct Scanner<'a> {
    /// The whole input, kept only so a failure can say where it was.
    input: &'a [u8],
    /// What has not been consumed.
    rest: &'a [u8],
}

impl<'a> Scanner<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            input: data,
            rest: data,
        }
    }

    pub fn remaining(&self) -> usize {
        self.rest.len()
    }

    pub fn peek(&self) -> Result<u8, ScannerError> {
        self.rest
            .first()
            .copied()
            .ok_or(ScannerError::UnexpectedEof)
    }

    /// Read one number, skipping any whitespace before it.
    pub fn next<T: lexical_core::FromLexical>(&mut self) -> Result<T, ScannerError> {
        self.skip_spaces()?;
        let (value, read) = lexical_core::parse_partial(self.rest).map_err(|source| {
            let (line, column) = self.position();
            ScannerError::InvalidNumber {
                source,
                line,
                column,
            }
        })?;
        self.rest = &self.rest[read..];
        Ok(value)
    }

    pub fn skip_spaces(&mut self) -> Result<(), ScannerError> {
        self.rest = spaces(self.rest);
        if self.rest.is_empty() {
            return Err(ScannerError::UnexpectedEof);
        }
        Ok(())
    }

    /// Consume the rest of the line, which must hold nothing but blank space.
    pub fn expect_end_of_line(&mut self) -> Result<(), ScannerError> {
        let Ok((rest, _)) = blank_then_newline(self.rest) else {
            let (line, column) = self.position();
            return if self.rest.trim_ascii_start().is_empty() {
                Err(ScannerError::UnexpectedEof)
            } else {
                Err(ScannerError::NoNewlineFound { line, column })
            };
        };
        self.rest = rest;
        Ok(())
    }

    /// Take the whole of the next line, without its newline.
    pub fn line(&mut self) -> Result<&'a [u8], ScannerError> {
        let Ok((rest, text)) = through_newline(self.rest) else {
            let (line, column) = self.position();
            return Err(ScannerError::NoNewlineFound { line, column });
        };
        self.rest = rest;
        Ok(text)
    }

    /// Consume an exact keyword, skipping any whitespace before it.
    pub fn skip_token(&mut self, token: &[u8]) -> Result<(), ScannerError> {
        self.rest = spaces(self.rest);
        let Ok((rest, _)) = keyword(self.rest, token) else {
            let found = self
                .rest
                .get(..token.len())
                .ok_or(ScannerError::UnexpectedEof)?;
            let (line, column) = self.position();
            return Err(ScannerError::InvalidToken {
                expected: String::from_utf8_lossy(token).into_owned(),
                found: String::from_utf8_lossy(found).into_owned(),
                line,
                column,
            });
        };
        self.rest = rest;
        Ok(())
    }

    /// Where the cursor is, 1-indexed, for an error message.
    ///
    /// Counted here rather than maintained as the scan runs: this is the error path,
    /// which happens once and then the parse is over.
    #[expect(
        clippy::naive_bytecount,
        reason = "the error path, once per parse; a bytecount dependency for that is \
                  more than it is worth"
    )]
    fn position(&self) -> (usize, usize) {
        let consumed = &self.input[..self.input.len() - self.rest.len()];
        let line = consumed.iter().filter(|&&c| c == b'\n').count() + 1;
        let column = consumed
            .iter()
            .rposition(|&c| c == b'\n')
            .map_or(consumed.len() + 1, |newline| consumed.len() - newline);
        (line, column)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "these values are exactly representable and the claim is exactness"
    )]
    fn next_parses_numbers_across_whitespace() {
        let mut scanner = Scanner::new(b"1.5  -2 \n 3e2");
        assert_eq!(scanner.next::<f32>().unwrap(), 1.5);
        assert_eq!(scanner.next::<i32>().unwrap(), -2);
        assert_eq!(scanner.next::<f32>().unwrap(), 300.0);
    }

    #[test]
    fn next_error_carries_position_and_context() {
        let mut scanner = Scanner::new(b"1.0 abc");
        scanner.next::<f32>().unwrap();
        let err = scanner.next::<f32>().unwrap_err();
        match err {
            ScannerError::InvalidNumber { line, column, .. } => {
                assert_eq!(line, 1);
                assert_eq!(column, 5);
            }
            other => panic!("expected InvalidNumber, got {other:?}"),
        }
    }

    #[test]
    fn next_on_exhausted_input_is_eof() {
        let mut scanner = Scanner::new(b"  \n ");
        assert!(matches!(
            scanner.next::<f32>().unwrap_err(),
            ScannerError::UnexpectedEof
        ));
    }

    #[test]
    fn skip_token_matches_and_advances() {
        let mut scanner = Scanner::new(b"  POINTS 2");
        scanner.skip_token(b"POINTS").unwrap();
        assert_eq!(scanner.next::<usize>().unwrap(), 2);
    }

    #[test]
    fn skip_token_mismatch_reports_both_tokens() {
        let mut scanner = Scanner::new(b"PLANES");
        match scanner.skip_token(b"POINTS").unwrap_err() {
            ScannerError::InvalidToken {
                expected, found, ..
            } => {
                assert_eq!(expected, "POINTS");
                assert_eq!(found, "PLANES");
            }
            other => panic!("expected InvalidToken, got {other:?}"),
        }
    }

    #[test]
    fn skip_token_on_truncated_input_errors_without_panicking() {
        let mut scanner = Scanner::new(b"POIN");
        assert!(matches!(
            scanner.skip_token(b"POINTS").unwrap_err(),
            ScannerError::UnexpectedEof
        ));
    }

    #[test]
    fn line_reads_until_newline() {
        let mut scanner = Scanner::new(b"1.0\nrest");
        assert_eq!(scanner.line().unwrap(), b"1.0");
        assert_eq!(scanner.remaining(), 4);
    }

    #[test]
    fn line_without_newline_errors() {
        let mut scanner = Scanner::new(b"no newline here");
        match scanner.line().unwrap_err() {
            ScannerError::NoNewlineFound { line, column } => {
                assert_eq!(line, 1);
                assert_eq!(column, 1);
            }
            other => panic!("expected NoNewlineFound, got {other:?}"),
        }
    }

    /// The module docstring says the number leaf stays `lexical_core` because nom's
    /// `float` costs more for the same answer. This is that claim, measured, and the
    /// "same answer" half asserted.
    ///
    /// ```sh
    /// cargo test -p srf --release -- --ignored --nocapture float_leaf
    /// ```
    #[test]
    #[ignore = "measures time, not behaviour"]
    fn float_leaf_is_cheaper_through_lexical_core() {
        const DEFAULT: &str = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/srfs/rupture_1.srf"
        );

        let path = std::env::var("SRF_THROUGHPUT_FILE").unwrap_or_else(|_| DEFAULT.to_owned());
        let data = std::fs::read(&path).expect("throughput fixture is readable");

        // Split once, so the timings cover conversion and nothing else. Keep only the
        // tokens both parsers consume whole -- PLANE, POINTS and the version line are
        // not numbers.
        let numbers: Vec<&[u8]> = data
            .split(u8::is_ascii_whitespace)
            .filter(|token| {
                matches!(lexical_core::parse_partial::<f32>(token), Ok((_, read)) if read == token.len())
            })
            .collect();

        let mut through_lexical = Vec::with_capacity(numbers.len());
        let start = std::time::Instant::now();
        for token in &numbers {
            let (value, _) = lexical_core::parse_partial::<f32>(token).expect("filtered");
            through_lexical.push(value);
        }
        let lexical_elapsed = start.elapsed();

        let mut through_nom = Vec::with_capacity(numbers.len());
        let start = std::time::Instant::now();
        for token in &numbers {
            let (_, value) = nom::number::complete::float::<&[u8], nom::error::Error<&[u8]>>(token)
                .expect("a number lexical_core accepted");
            through_nom.push(value);
        }
        let nom_elapsed = start.elapsed();

        assert_eq!(
            through_lexical, through_nom,
            "the two float parsers disagree, so the choice between them is not free"
        );

        let count = numbers.len();
        println!(
            "{count} numbers from {path}:\n  \
             lexical_core::parse_partial {:.3} s\n  \
             nom::number::complete::float {:.3} s",
            lexical_elapsed.as_secs_f64(),
            nom_elapsed.as_secs_f64(),
        );
    }

    #[test]
    fn next_error_line_tracking() {
        let mut scanner = Scanner::new(b"1.0\nabc");
        scanner.next::<f32>().unwrap();
        let err = scanner.next::<f32>().unwrap_err();
        match err {
            ScannerError::InvalidNumber { line, column, .. } => {
                assert_eq!(line, 2);
                assert_eq!(column, 1);
            }
            other => panic!("expected InvalidNumber, got {other:?}"),
        }
    }

    #[test]
    fn end_of_line_accepts_trailing_blanks_and_a_carriage_return() {
        let mut scanner = Scanner::new(b"1.0  \t\r\nnext");
        scanner.next::<f32>().unwrap();
        scanner.expect_end_of_line().unwrap();
        assert_eq!(scanner.rest, b"next");
    }

    #[test]
    fn end_of_line_rejects_another_token_on_the_line() {
        let mut scanner = Scanner::new(b"1.0 2.0\n");
        scanner.next::<f32>().unwrap();
        assert!(matches!(
            scanner.expect_end_of_line().unwrap_err(),
            ScannerError::NoNewlineFound { line: 1, column: 4 }
        ));
    }

    #[test]
    fn end_of_line_at_the_end_of_input_is_eof() {
        let mut scanner = Scanner::new(b"1.0   ");
        scanner.next::<f32>().unwrap();
        assert!(matches!(
            scanner.expect_end_of_line().unwrap_err(),
            ScannerError::UnexpectedEof
        ));
    }
}
