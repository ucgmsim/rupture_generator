//! The production draw source: PCG64-DXSM uniforms, ziggurat normals.
//!
//! This is what new work should use. Both halves are chosen, not inherited:
//!
//! * **PCG64-DXSM** for the uniforms. A 2^128 period, passes the standard test
//!   batteries, and — unlike the LCG it replaces — has no short-period structure in
//!   its low bits, which matters because the field generators consume draws in a
//!   strided pattern across a wavenumber grid.
//!
//! * **The ziggurat algorithm** for the normals, via `rand_distr`. It accepts on
//!   the first try roughly 99% of the time, so a normal deviate costs about one
//!   uniform instead of the twelve an Irwin-Hall sum costs — and it is an exact
//!   normal rather than a bounded twelve-fold approximation with no tails beyond
//!   six standard deviations.
//!
//! The tails are the part that matters physically. Slip, rise-time and rupture-time
//! perturbations are all log-normal or multiplicative in the deviate, so the
//! largest draws set the extremes of the rupture model. Irwin-Hall cannot produce a
//! deviate beyond ±6σ at all; a real normal can, and does, about twice in a billion
//! draws — which a large fault reaches.

use rand::{RngExt as _, SeedableRng};
use rand_distr::StandardNormal;
use rand_pcg::Pcg64Dxsm;

use super::{DrawSource, Realisations};

/// The production draw source.
#[derive(Clone, Debug)]
pub struct Pcg {
    /// Kept so realisations can be derived without consuming this stream.
    seed: u64,
    generator: Pcg64Dxsm,
}

impl Pcg {
    /// Start a stream at `seed`.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            generator: Pcg64Dxsm::seed_from_u64(seed),
        }
    }
}

impl DrawSource for Pcg {
    fn uniform(&mut self) -> f64 {
        // `random` gives [0, 1); the contract is [-1, 1]. The upper end is
        // unreachable here, which is permitted -- the contract is what consumers
        // must tolerate, not what every source must produce.
        self.generator.random::<f64>().mul_add(2.0, -1.0)
    }

    fn gaussian(&mut self, sigma: f32, mean: f32) -> f64 {
        let deviate: f64 = self.generator.sample(StandardNormal);
        deviate.mul_add(f64::from(sigma), f64::from(mean))
    }
}

impl Realisations for Pcg {
    fn realisation(&self, index: u64) -> Self {
        // Derive a distinct stream rather than skipping ahead in this one, so
        // realisations are independent rather than adjacent. SplitMix64's finalising
        // mix avalanches the low bits of `index`, which matters because the indices
        // are 0, 1, 2, ... and seeding directly from them would give correlated
        // starting states.
        Self::new(split_mix_64(self.seed ^ split_mix_64(index)))
    }
}

/// `SplitMix64`'s finalising mix, used to derive independent seeds.
const fn split_mix_64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}
