//! Direct FFI access to the genslip v5.6.2 C library, for differential testing.
//!
//! This crate exists so the port has a referee. It is **not** a dependency of
//! `genslip` and nothing ships against it — it is linked by tests, which call a C
//! function and the Rust function that replaces it on the same inputs and assert
//! bit-equality.
//!
//! Keep it. The predecessor project deleted its oracle once the port was certified
//! and then resurrected it twice from git history, rediscovering the same build
//! drift each time. This is a `build.rs` and a header; deleting it saves nothing.
//!
//! # What is safe here
//!
//! Nothing, strictly. Every function below is `unsafe extern "C"`. The wrappers in
//! [`rng`] and elsewhere are safe only because their arguments are plain scalars
//! with no aliasing or length obligations; anything taking a buffer keeps its
//! `unsafe` and documents the length contract at the call site, because the C takes
//! assumed-size pointers whose true length lives in the caller.

#![allow(
    clippy::missing_panics_doc,
    reason = "test-only crate; panics are the failure mode we want"
)]

pub mod field;
pub mod rng;

/// genslip's complex type (`Genslip/v5.6.2/structure.h`). Single precision, and
/// laid out as two floats rather than as a C99 `_Complex`.
///
/// Layout-compatible with `num_complex::Complex32`, which is also `repr(C)` with
/// the real part first, so a Rust grid can be handed straight to the C.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}
