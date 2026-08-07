#![expect(
    clippy::naive_bytecount,
    reason = "two call sites counting newlines; a bytecount dependency for that is \
              more than it is worth"
)]

use thiserror::Error;

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

pub struct Scanner<'a> {
    data: &'a [u8],
    index: usize,
    line: usize,
}

impl<'a> Scanner<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            index: 0,
            line: 1,
        }
    }

    pub fn remaining(&self) -> usize {
        self.data.len() - self.index
    }

    pub fn peek(&self) -> Result<u8, ScannerError> {
        self.data
            .get(self.index)
            .copied()
            .ok_or(ScannerError::UnexpectedEof)
    }

    pub fn next<T: lexical_core::FromLexical>(&mut self) -> Result<T, ScannerError> {
        self.skip_spaces()?;
        let (val, read) =
            lexical_core::parse_partial(&self.data[self.index..]).map_err(|source| {
                ScannerError::InvalidNumber {
                    source,
                    line: self.line,
                    column: self.column(),
                }
            })?;
        self.index += read;
        Ok(val)
    }

    pub fn skip_spaces(&mut self) -> Result<(), ScannerError> {
        let nonwhitespace = self.data[self.index..]
            .iter()
            .position(|&c| !c.is_ascii_whitespace());
        match nonwhitespace {
            Some(x) => {
                self.line += self.data[self.index..self.index + x]
                    .iter()
                    .filter(|&&c| c == b'\n')
                    .count();
                self.index += x;

                Ok(())
            }
            _ => Err(ScannerError::UnexpectedEof),
        }
    }

    pub fn expect_end_of_line(&mut self) -> Result<(), ScannerError> {
        let jump = self.data[self.index..]
            .iter()
            .enumerate()
            .find(|&(_, &c)| c == b'\n' || !c.is_ascii_whitespace());
        match jump {
            Some((i, b'\n')) => {
                self.index += i + 1; // Plus one to skip the newline itself
                self.line += 1;
                Ok(())
            }
            Some(_) => Err(ScannerError::NoNewlineFound {
                line: self.line,
                column: self.column(),
            }),
            _ => Err(ScannerError::UnexpectedEof),
        }
    }

    pub fn line(&mut self) -> Result<&[u8], ScannerError> {
        let newline_index = self.data[self.index..].iter().position(|&c| c == b'\n');
        match newline_index {
            Some(x) => {
                let res = Ok(&self.data[self.index..self.index + x]);
                self.line += 1;
                self.index += x + 1; // plus 1 to skip the newline itself.
                res
            }
            _ => Err(ScannerError::NoNewlineFound {
                line: self.line,
                column: self.column(),
            }),
        }
    }

    pub fn skip_token(&mut self, token: &[u8]) -> Result<(), ScannerError> {
        self.skip_spaces()?;
        let next = self
            .data
            .get(self.index..self.index + token.len())
            .ok_or(ScannerError::UnexpectedEof)?;
        if next == token {
            self.index += token.len();
            self.line += token.iter().filter(|&&c| c == b'\n').count();
            Ok(())
        } else {
            Err(ScannerError::InvalidToken {
                line: self.line,
                column: self.column(),
                expected: String::from_utf8_lossy(token).into_owned(),
                found: String::from_utf8_lossy(next).into_owned(),
            })
        }
    }

    fn column(&self) -> usize {
        // Columns are 1-indexed for readability
        self.data[..self.index]
            .iter()
            .rposition(|&c| c == b'\n')
            .map_or(self.index + 1, |nl| self.index - nl)
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
}
