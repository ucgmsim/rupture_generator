//! Kinematic rupture model generation.
//!
//! Given a discretised fault, a magnitude and a seed, produce a slip distribution,
//! rupture onset times, rise times and per-subfault slip-rate functions.
//!
//! A port of genslip v5.6.2. The contract is **scientific agreement** with the C, not
//! bit-equality: `ENGINEERING_RULES.md` defines when two ruptures are the same rupture
//! and what a failing test in each class obliges you to do. Read it before changing a
//! kernel. `PORTING_RULES.md` describes the bit-parity regime that got the port here
//! and is now archaeology.
//!
//! # Failure
//!
//! The entry points return [`error::Result`]. Everything below them asserts, because
//! by then the values came from this crate rather than from a caller — see
//! [`error`] for where that line is and why it is drawn there.

pub mod error;
pub mod fft;
pub mod field;
pub mod geodesy;
pub mod grid;
pub mod moment;
pub mod realisation;
pub mod rise_time;
pub mod rng;
pub mod rupture;
pub mod slip;
pub mod slip_rate;
pub mod source;
pub mod stats;
pub mod taper;
pub mod units;

pub use error::{Error, Result};
