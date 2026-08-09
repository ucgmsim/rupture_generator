//! Generative strategies for the kernel contracts, shared so the two contract files
//! quantify over the same input space rather than each inventing a narrower one.
//!
//! Ranges are physical: slowness 0.2–2.0 s/km is shear speed 0.5–5 km/s, spacing
//! 0.1–2 km brackets every production subfault size, slip up to 30 m covers a
//! magnitude-9 asperity, and `dt` 1–50 ms brackets the sample intervals SRF files
//! carry. Grids are kept small because the properties are resolution-independent —
//! the one that is not, convergence order, sets its own sizes.

// Each integration test compiles this module and uses its half.
#![allow(dead_code)]

use _kernels::eikonal::Seed;
use proptest::prelude::*;

/// A test index as a float. Small by construction, so the cast is exact.
#[must_use]
#[expect(clippy::cast_precision_loss, reason = "test indices are small")]
pub fn exact(count: usize) -> f64 {
    count as f64
}

/// A heterogeneous medium with one or more seeds: the eikonal kernel's whole input.
#[derive(Clone, Debug)]
pub struct Grid {
    pub extent: (usize, usize),
    pub spacing_km: (f64, f64),
    pub slowness: Vec<f64>,
    pub seeds: Vec<Seed>,
}

impl Grid {
    pub fn at(&self, i: usize, j: usize) -> usize {
        i * self.extent.1 + j
    }

    pub fn is_seed(&self, i: usize, j: usize) -> bool {
        self.seeds.iter().any(|seed| (seed.i, seed.j) == (i, j))
    }

    pub fn slowest(&self) -> f64 {
        self.slowness.iter().copied().fold(0.0_f64, f64::max)
    }

    pub fn fastest(&self) -> f64 {
        self.slowness.iter().copied().fold(f64::INFINITY, f64::min)
    }
}

fn spacing_km() -> impl Strategy<Value = f64> {
    0.1_f64..2.0
}

fn slowness_s_per_km() -> impl Strategy<Value = f64> {
    0.2_f64..2.0
}

/// Grids of every aspect ratio, media with no structure at all — rougher than any
/// rupture-speed field, which is the point: the properties hold pointwise, not
/// because the medium is smooth — and up to `max_seeds` seeds at distinct or
/// coincident positions, with start times up to five seconds apart.
pub fn grid(max_seeds: usize) -> impl Strategy<Value = Grid> {
    ((2_usize..=20), (2_usize..=20))
        .prop_flat_map(move |(ni, nj)| {
            (
                Just((ni, nj)),
                (spacing_km(), spacing_km()),
                prop::collection::vec(slowness_s_per_km(), ni * nj),
                prop::collection::vec((0..ni, 0..nj, 0.0_f64..5.0), 1..=max_seeds),
            )
        })
        .prop_map(|(extent, spacing_km, slowness, seeds)| Grid {
            extent,
            spacing_km,
            slowness,
            seeds: seeds
                .into_iter()
                .map(|(i, j, t0_s)| Seed { i, j, t0_s })
                .collect(),
        })
}

/// A uniform medium with a single seed — the case with an exact analytic answer.
#[derive(Clone, Debug)]
pub struct UniformGrid {
    pub extent: (usize, usize),
    pub spacing_km: (f64, f64),
    pub slowness: f64,
    pub seed: Seed,
}

pub fn uniform_grid() -> impl Strategy<Value = UniformGrid> {
    ((2_usize..=24), (2_usize..=24))
        .prop_flat_map(|(ni, nj)| {
            (
                Just((ni, nj)),
                (spacing_km(), spacing_km()),
                slowness_s_per_km(),
                (0..ni, 0..nj, 0.0_f64..5.0),
            )
        })
        .prop_map(|(extent, spacing_km, slowness, (i, j, t0_s))| UniformGrid {
            extent,
            spacing_km,
            slowness,
            seed: Seed { i, j, t0_s },
        })
}

/// Per-subfault inputs for pulse synthesis: slip, rise time, beta, and the interval.
#[derive(Clone, Debug)]
pub struct Subfaults {
    pub slip_m: Vec<f64>,
    pub rise_time_s: Vec<f64>,
    pub beta: Vec<f64>,
    pub dt_s: f64,
}

/// Slip in metres: mostly slipping subfaults, salted with exact zeros and
/// slips at or below the guard, because the empty-row contract is half the point.
fn slip_m() -> impl Strategy<Value = f64> {
    prop_oneof![
        2 => Just(0.0),
        1 => 0.0..=_kernels::pulse::MIN_SLIP_M,
        5 => 0.001_f64..30.0,
    ]
}

fn subfaults(
    rise_multiplier: impl Strategy<Value = f64> + Clone,
) -> impl Strategy<Value = Subfaults> {
    (1_usize..=32, 1.0e-3_f64..0.05).prop_flat_map(move |(count, dt_s)| {
        (
            Just(dt_s),
            prop::collection::vec(slip_m(), count),
            prop::collection::vec(rise_multiplier.clone(), count),
            prop::collection::vec(0.05_f64..=0.5, count),
        )
            .prop_map(|(dt_s, slip_m, multipliers, beta)| Subfaults {
                rise_time_s: multipliers.iter().map(|multiple| multiple * dt_s).collect(),
                slip_m,
                beta,
                dt_s,
            })
    })
}

/// Every rise time resolvable: at least 1.6 samples, so `oliu_p` always has a
/// computed shape rather than its spike or its refusal.
pub fn resolvable_subfaults() -> impl Strategy<Value = Subfaults> {
    subfaults(1.6_f64..400.0)
}

/// Rise times down to zero — *below* what the sample interval can represent — so
/// the refusal contract is exercised, not just the happy path.
pub fn adversarial_subfaults() -> impl Strategy<Value = Subfaults> {
    subfaults(0.0_f64..3.0)
}
