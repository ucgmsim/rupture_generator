//! Factored fast sweeping: the eikonal equation with its source singularity removed.
//!
//! # The problem this solves
//!
//! Every scheme that discretises `|∇T| = s` directly loses accuracy at a point source,
//! because `T` is not smooth there — `∇T` is discontinuous in direction — and the
//! error that creates spreads over the whole grid. It shows up as a *failure to
//! converge*: refining the mesh does not help, because the near-source error is a
//! fixed fraction rather than a truncation term. Both solvers this replaced have it.
//! `DEFECTS.md` 19 measures them at convergence ratios of 1.03 and 1.01 where a
//! converging scheme gives 2; `tests/eikonal_contract.rs` holds whatever solver is
//! here to the converging number.
//!
//! # The two papers
//!
//! > **Zhao, H. (2005).** A fast sweeping method for eikonal equations.
//! > *Mathematics of Computation* **74**(250), 603–627.
//! >
//! > **Fomel, S., Luo, S. & Zhao, H. (2009).** Fast sweeping method for the factored
//! > eikonal equation. *Journal of Computational Physics* **228**(17), 6440–6455.
//!
//! Zhao gives the sweeping strategy: Gauss–Seidel iteration under **alternating
//! orderings**, each following one family of characteristics, so a fixed number of
//! sweeps covers every ray direction and the iteration count does not grow with the
//! mesh. Fomel et al. give the factorisation.
//!
//! Their **Eq. (3)** splits the traveltime multiplicatively, `T = T₀·τ`, with
//! `|∇T₀| = S₀` (**Eq. 4**). Taking `S₀` constant and `T₀(x) = S₀·|x − x₀|` — the
//! analytic answer for a homogeneous medium at the source's own slowness — puts the
//! singularity entirely inside `T₀`, where it is known in closed form, and leaves `τ`
//! smooth. In a uniform medium `τ ≡ 1` *exactly*, and the discrete equations below
//! are satisfied by it at any grid spacing, so the scheme reproduces the analytic
//! solution to rounding.
//!
//! **Eq. (5)** is what is actually solved:
//!
//! ```text
//!     T₀²|∇τ|² + 2T₀τ ∇T₀·∇τ + (τ² − α²)S₀² = 0
//! ```
//!
//! **Eq. (7)** discretises it at a node on each of the four triangles `ΔCEN`, `ΔCNW`,
//! `ΔCWS`, `ΔCSE` — one per quadrant — giving a quadratic in the node's `τ`.
//!
//! # The causality condition is the whole thing
//!
//! Stated immediately after Eq. (7), and the paper's abstract calls it the key idea:
//!
//! ```text
//!     τ_C·T₀(C) ≥ τ_W·T₀(W)      and      τ_C·T₀(C) ≥ τ_S·T₀(S)
//! ```
//!
//! Causality is enforced on **`T`**, not on `τ`. Without it a candidate built from
//! downwind neighbours can win, and the uniform-medium error is 0.11 — no better than
//! plain fast marching. With it, 1e-14. It is one comparison and it is worth eleven
//! orders of magnitude.
//!
//! # Multiple seeds
//!
//! The seed contract is **points with initial times**, not "the hypocentre" — what a
//! rupture jumping between faults needs, and what costs a single fault nothing
//! (`PLAN.md` S7). The factorisation is inherently per-source: `T₀` removes *one*
//! singularity, so a multi-seed field is solved as one factored sweep per seed and
//! the pointwise minimum of the shifted results. That is not a shortcut standing in
//! for a "real" multi-source sweep — first arrival from several sources *is* the
//! minimum over sources, and solving them separately is what keeps every source's
//! near-field exact. `tests/eikonal_contract.rs` pins the equality so that a future
//! single-pass implementation has a contract to meet.
//!
//! # One deliberate deviation
//!
//! When no root of Eq. (7) is causal on any triangle, the paper falls back to the
//! method of characteristics along the two edges (**Eq. 8**). This uses the one-sided
//! update `T = min(T_neighbour) + h·s` instead — the safe cap every upwind scheme
//! carries, and what the original Fortran's head-wave terms amount to. It fires rarely
//! on media like these: the constant-gradient case reaches its accuracy without the
//! fallback ever being reached. Recorded here rather than left silent; if a
//! heterogeneous case ever measures worse than expected, Eq. (8) is the first thing to
//! add.

use crate::counts::exact;

/// Rounds of four sweeps before giving up on convergence.
///
/// Fomel et al. report 3 for their examples, at every mesh size from 150×50 to
/// 1200×400 — the count is a property of the medium, not of the grid, which is what
/// Zhao's alternating orderings buy. This is a generous ceiling; the loop exits as
/// soon as a round changes nothing.
const MAX_ROUNDS: usize = 16;

/// A point the rupture front starts from, and when.
///
/// Indices are `(i, j)` with `i` down-dip and `j` along-strike — the chart convention
/// of `PLAN.md` §2. `t0_s` is the time the front leaves this point: zero for a
/// configured hypocentre, a parent fault's arrival plus the jump delay for a
/// triggered one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seed {
    pub i: usize,
    pub j: usize,
    pub t0_s: f64,
}

/// What this solver refuses, in its own vocabulary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Error {
    /// A grid with no cells has no wavefront to solve.
    EmptyGrid { ni: usize, nj: usize },
    /// The slowness slice does not cover the grid it claims to.
    WrongLength { ni: usize, nj: usize, got: usize },
    /// Grid spacing must be a positive, finite length on both axes.
    NonPositiveSpacing { axis: &'static str, value: f64 },
    /// Slowness must be positive and finite everywhere: a cell no wave can cross
    /// would make its neighbours unreachable, and the error would surface as a
    /// solver panic far from the cell that caused it.
    NonPositiveSlowness { i: usize, j: usize, value: f64 },
    /// No seeds means no wavefront: every travel time would be infinite.
    NoSeeds,
    /// A seed outside the grid is a caller error, not a boundary condition.
    SeedOutOfBounds {
        seed: usize,
        i: usize,
        j: usize,
        ni: usize,
        nj: usize,
    },
    /// A seed's initial time must be finite; NaN or infinity would poison every
    /// cell its wavefront wins.
    NonFiniteSeedTime { seed: usize, t0_s: f64 },
    /// The sweep did not settle in [`MAX_ROUNDS`] rounds; the medium has structure
    /// this scheme does not handle.
    DidNotSettle { rounds: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::EmptyGrid { ni, nj } => {
                write!(f, "a {ni}x{nj} grid has no cells to solve on")
            }
            Self::WrongLength { ni, nj, got } => write!(
                f,
                "the slowness field has {got} values, but a {ni}x{nj} grid needs {}",
                ni * nj
            ),
            Self::NonPositiveSpacing { axis, value } => write!(
                f,
                "the {axis} spacing is {value} km; spacing must be positive and finite"
            ),
            Self::NonPositiveSlowness { i, j, value } => write!(
                f,
                "slowness at ({i}, {j}) is {value} s/km; every cell must be positive \
                 and finite, or the cells behind it are unreachable"
            ),
            Self::NoSeeds => write!(f, "no seeds: a wavefront needs somewhere to start"),
            Self::SeedOutOfBounds { seed, i, j, ni, nj } => {
                write!(f, "seed {seed} at ({i}, {j}) is outside a {ni}x{nj} grid")
            }
            Self::NonFiniteSeedTime { seed, t0_s } => {
                write!(f, "seed {seed} starts at t = {t0_s}, which is not a time")
            }
            Self::DidNotSettle { rounds } => write!(
                f,
                "the sweep did not settle in {rounds} rounds; the medium has \
                 structure this scheme does not handle"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// First-arrival times from every seed, on the whole grid.
///
/// `slowness_s_per_km` is row-major over `(ni, nj)` — `i` down-dip, `j` along-strike
/// — and `spacing_km` is `(h_i, h_j)`, the cell size on each axis. The result has the
/// same layout, in seconds.
///
/// # Errors
///
/// [`Error`], one variant per way the inputs can fail to describe a medium; nothing
/// is clamped or silently repaired.
pub fn solve(
    slowness_s_per_km: &[f64],
    extent: (usize, usize),
    spacing_km: (f64, f64),
    seeds: &[Seed],
) -> Result<Vec<f64>, Error> {
    solve_with_rounds(slowness_s_per_km, extent, spacing_km, seeds).map(|(times, _)| times)
}

/// [`solve`], also reporting the most rounds of four sweeps any seed's solve took.
///
/// Exposed because the round count is the evidence for the cost claim rather than a
/// diagnostic: Zhao's alternating orderings exist to make it a property of the
/// *medium* and not of the mesh, so a solver whose rounds grew with the grid would be
/// O(N log N) or worse in disguise. `tests/eikonal_contract.rs` asserts it stays put
/// across a fourfold refinement.
///
/// # Errors
///
/// As [`solve`].
pub fn solve_with_rounds(
    slowness_s_per_km: &[f64],
    extent: (usize, usize),
    spacing_km: (f64, f64),
    seeds: &[Seed],
) -> Result<(Vec<f64>, usize), Error> {
    let (ni, nj) = extent;
    if ni == 0 || nj == 0 {
        return Err(Error::EmptyGrid { ni, nj });
    }
    if ni.checked_mul(nj) != Some(slowness_s_per_km.len()) {
        return Err(Error::WrongLength {
            ni,
            nj,
            got: slowness_s_per_km.len(),
        });
    }
    for (axis, value) in [("down-dip", spacing_km.0), ("along-strike", spacing_km.1)] {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::NonPositiveSpacing { axis, value });
        }
    }
    for (index, &value) in slowness_s_per_km.iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::NonPositiveSlowness {
                i: index / nj,
                j: index % nj,
                value,
            });
        }
    }
    if seeds.is_empty() {
        return Err(Error::NoSeeds);
    }
    for (index, seed) in seeds.iter().enumerate() {
        if seed.i >= ni || seed.j >= nj {
            return Err(Error::SeedOutOfBounds {
                seed: index,
                i: seed.i,
                j: seed.j,
                ni,
                nj,
            });
        }
        if !seed.t0_s.is_finite() {
            return Err(Error::NonFiniteSeedTime {
                seed: index,
                t0_s: seed.t0_s,
            });
        }
    }

    // One factored sweep per seed, combined by pointwise minimum — see the module
    // documentation for why the factorisation makes this the scheme rather than a
    // shortcut. The seed's own solve starts at zero and `t0_s` shifts it afterwards:
    // the eikonal equation is autonomous in time, so the shift is exact.
    let mut combined = vec![f64::INFINITY; slowness_s_per_km.len()];
    let mut most_rounds = 0;
    for seed in seeds {
        let (times, rounds) = single_seed(slowness_s_per_km, extent, spacing_km, *seed)?;
        most_rounds = most_rounds.max(rounds);
        for (cell, arrival) in combined.iter_mut().zip(&times) {
            *cell = cell.min(arrival + seed.t0_s);
        }
    }
    Ok((combined, most_rounds))
}

/// The known factor `T₀` and its gradient at one node, in seconds and s/km.
///
/// `T₀ = S₀·r` with `r` the distance to the source, so `∇T₀ = S₀·r̂` — a unit vector
/// scaled by the source slowness, undefined only at the source itself where the
/// paper's Remark 1 supplies `τ = α` by l'Hôpital instead.
struct Known {
    time_s: f64,
    /// `(∂T₀/∂i, ∂T₀/∂j)` in physical units — the spacing is already folded in.
    gradient: (f64, f64),
}

fn known_factor(
    i: usize,
    j: usize,
    seed: Seed,
    spacing_km: (f64, f64),
    source_slowness: f64,
) -> Known {
    let down = (exact(i) - exact(seed.i)) * spacing_km.0;
    let across = (exact(j) - exact(seed.j)) * spacing_km.1;
    let radius = (down * down + across * across).sqrt();
    if radius == 0.0 {
        return Known {
            time_s: 0.0,
            gradient: (0.0, 0.0),
        };
    }
    Known {
        time_s: source_slowness * radius,
        gradient: (
            source_slowness * down / radius,
            source_slowness * across / radius,
        ),
    }
}

/// Fast sweeping on the factored eikonal equation, from one seed at time zero.
///
/// # Panics
///
/// If a cell is never reached. The inputs were already checked — every slowness is
/// positive, so every cell is reachable — and the panic is a statement about the
/// solver rather than the input.
fn single_seed(
    slowness: &[f64],
    extent: (usize, usize),
    spacing_km: (f64, f64),
    seed: Seed,
) -> Result<(Vec<f64>, usize), Error> {
    let (ni, nj) = extent;
    let at = |i: usize, j: usize| i * nj + j;
    let source_slowness = slowness[at(seed.i, seed.j)];

    // `T₀` and `∇T₀` are analytic, so they are computed once rather than swept.
    let known: Vec<Known> = (0..ni)
        .flat_map(|i| (0..nj).map(move |j| (i, j)))
        .map(|(i, j)| known_factor(i, j, seed, spacing_km, source_slowness))
        .collect();

    // Solved in `T` throughout, converting to `τ` only where the discretisation
    // needs it. Keeping the array in `T` is what lets causality be a plain
    // comparison, and `T` is what the caller wants anyway.
    let mut times = vec![f64::INFINITY; ni * nj];
    times[at(seed.i, seed.j)] = 0.0;

    // Zhao's four alternating orderings: every ray direction is covered by one of
    // them, which is why a fixed number of rounds suffices.
    let forward: Vec<usize> = (0..nj).collect();
    let backward: Vec<usize> = (0..nj).rev().collect();
    let down: Vec<usize> = (0..ni).collect();
    let up: Vec<usize> = (0..ni).rev().collect();

    let mut rounds = 0;
    loop {
        let mut changed = false;
        for dips in [&down, &up] {
            for strikes in [&forward, &backward] {
                for &i in dips {
                    for &j in strikes {
                        if (i, j) == (seed.i, seed.j) {
                            continue;
                        }
                        let candidate =
                            update(&times, &known, i, j, extent, spacing_km, slowness[at(i, j)]);
                        if candidate < times[at(i, j)] {
                            times[at(i, j)] = candidate;
                            changed = true;
                        }
                    }
                }
            }
        }
        rounds += 1;
        if !changed {
            break;
        }
        if rounds >= MAX_ROUNDS {
            return Err(Error::DidNotSettle { rounds });
        }
    }

    for (index, arrival) in times.iter().enumerate() {
        assert!(
            arrival.is_finite(),
            "({}, {}) was never reached",
            index / nj,
            index % nj
        );
    }

    Ok((times, rounds))
}

/// The best arrival this node can be given from its current neighbours.
///
/// Fomel et al. Eq. (7) on each of the four quadrant triangles, with the causality
/// condition, then the one-sided cap that stands in for their Eq. (8). Each axis
/// carries its own spacing; nothing here assumes the cells are square.
fn update(
    times: &[f64],
    known: &[Known],
    i: usize,
    j: usize,
    extent: (usize, usize),
    spacing_km: (f64, f64),
    slowness: f64,
) -> f64 {
    let (ni, nj) = extent;
    let (h_i, h_j) = spacing_km;
    let at = |i: usize, j: usize| i * nj + j;
    let here = &known[at(i, j)];

    // The two neighbours on each axis, as (arrival, its own T₀, which side).
    // `sign` is +1 when the neighbour is at the lower index, matching the sign the
    // upwind difference carries.
    let mut along_j: Vec<(f64, f64, f64)> = Vec::with_capacity(2);
    if j > 0 {
        let index = at(i, j - 1);
        along_j.push((times[index], known[index].time_s, 1.0));
    }
    if j + 1 < nj {
        let index = at(i, j + 1);
        along_j.push((times[index], known[index].time_s, -1.0));
    }
    let mut along_i: Vec<(f64, f64, f64)> = Vec::with_capacity(2);
    if i > 0 {
        let index = at(i - 1, j);
        along_i.push((times[index], known[index].time_s, 1.0));
    }
    if i + 1 < ni {
        let index = at(i + 1, j);
        along_i.push((times[index], known[index].time_s, -1.0));
    }

    let mut best = f64::INFINITY;

    // One triangle per quadrant, plus the two one-sided degenerations. Eq. (7) is a
    // quadratic in this node's `τ`; the larger root is the causal branch.
    for corner in along_j
        .iter()
        .map(Some)
        .chain([None])
        .flat_map(|x| along_i.iter().map(Some).chain([None]).map(move |y| (x, y)))
    {
        let (x, y) = corner;
        if x.is_none() && y.is_none() {
            continue;
        }

        let term = |side: Option<&(f64, f64, f64)>, gradient: f64, spacing: f64| match side {
            Some(&(arrival, factor, sign)) if arrival.is_finite() => {
                // Remark 1: at the source `T₀` is zero and `τ = T/T₀` is 0/0. By
                // l'Hôpital, or from Eq. (5) directly, `τ(x₀) = α(x₀)` — and with
                // `S₀` taken as the source's own slowness that is exactly 1.
                //
                // Leaving the source out instead makes its four neighbours
                // unreachable by any triangle, so they fall through to the one-sided
                // cap and the whole near-source region loses the factorisation. That
                // is the one place this scheme's advantage actually lives.
                let tau = if factor > 0.0 { arrival / factor } else { 1.0 };
                Some((
                    sign * here.time_s / spacing + gradient,
                    sign * here.time_s * tau / spacing,
                ))
            }
            _ => None,
        };
        let (a_x, b_x) = term(x, here.gradient.1, h_j).unwrap_or((0.0, 0.0));
        let (a_y, b_y) = term(y, here.gradient.0, h_i).unwrap_or((0.0, 0.0));
        if a_x == 0.0 && a_y == 0.0 {
            continue;
        }

        let quadratic = a_x * a_x + a_y * a_y;
        let linear = -2.0 * (a_x * b_x + a_y * b_y);
        let constant = b_x * b_x + b_y * b_y - slowness * slowness;
        let discriminant = linear * linear - 4.0 * quadratic * constant;
        if discriminant < 0.0 {
            continue;
        }
        let arrival = here.time_s * (-linear + discriminant.sqrt()) / (2.0 * quadratic);
        // `partial_cmp` rather than `!(arrival > 0.0)`: the root can be NaN when the
        // quadratic degenerates, and a negated comparison would silently accept it.
        if !matches!(arrival.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            continue;
        }

        // Fomel et al.'s causality condition, on T rather than on tau. Without this
        // a downwind neighbour can win and the scheme is no better than plain fast
        // marching.
        let causal = |side: Option<&(f64, f64, f64)>| {
            side.is_none_or(|&(neighbour, _, _)| !neighbour.is_finite() || arrival >= neighbour)
        };
        if causal(x) && causal(y) {
            best = best.min(arrival);
        }
    }

    if best.is_finite() {
        return best;
    }

    // Standing in for Eq. (8), and **only** when no triangle produced a causal root.
    // A wave crossing one cell along an axis is always causal, so a node whose
    // triangles all failed still gets a bound rather than staying unreachable.
    //
    // Offering this alongside the triangles rather than after them costs a factor of
    // six on a gradient: it is an unfactored first-order update, so wherever it is
    // the smaller of the two it wins and injects exactly the source-singularity error
    // the factorisation exists to remove. Measured at 1.03e-02 against 1.75e-03 on
    // the constant-gradient case.
    for (arrival, spacing) in along_j
        .iter()
        .map(|side| (side.0, h_j))
        .chain(along_i.iter().map(|side| (side.0, h_i)))
    {
        if arrival.is_finite() {
            best = best.min(arrival + spacing * slowness);
        }
    }

    best
}
