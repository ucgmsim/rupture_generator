//! Kinematic rupture model generation.
//!
//! Given a discretised fault, a magnitude and a seed, produce a slip distribution,
//! rupture onset times, rise times and per-subfault slip-rate functions.
//!
//! A port of genslip v5.6.2. During Stage 1 the contract is bit-equality with the C,
//! checked per function against `genslip-oracle`; see `PORTING_RULES.md`.

pub mod fft;
pub mod field;
pub mod geodesy;
pub mod grid;
pub mod moment;
pub mod rng;
pub mod rupture;
pub mod slip;
pub mod stats;
pub mod taper;
