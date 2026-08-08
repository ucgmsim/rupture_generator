//! What every test in this directory needs, in one place.
//!
//! Seventeen test files used to rebuild the same fault, velocity model and spectrum
//! spec by hand, which meant that reshaping any of those structs — the thing
//! `ENGINEERING_RULES.md` rule 7 invites — touched all seventeen. That cost is why
//! the reshaping did not happen.
//!
//! Four things live here, and they are four because each answers a different question:
//!
//! | module | question |
//! | --- | --- |
//! | [`fixture`] | what fault are we generating? |
//! | [`counting`] | how much randomness did that consume, and in what order? |
//! | [`stats`] | how do two fields differ — in level, in spread, or in shape? |
//! | [`tolerance`] | how close is close enough, and what can that see? |
//!
//! Every tolerance here is *derived* from sample size rather than chosen, per
//! rule 1, and each carries its detection floor in the doc comment. A test that
//! reaches for a literal instead should first check whether the right helper is
//! missing.

// Each integration-test binary compiles this module separately and uses a different
// subset, so unused items are the normal case rather than a smell.
#![allow(dead_code)]

pub mod counting;
pub mod fixture;
pub mod stats;
pub mod tolerance;
