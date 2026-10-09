//! Factored fast sweeping: the eikonal equation with its source singularity removed.
//!
//! # Reference papers
//!
//! > **Zhao, H. (2005).** A fast sweeping method for eikonal equations.
//! > *Mathematics of Computation* **74**(250), 603-627.
//! >
//! > **Fomel, S., Luo, S. & Zhao, H. (2009).** Fast sweeping method for the factored
//! > eikonal equation. *Journal of Computational Physics* **228**(17), 6440-6455.
//!
//! Zhao gives the sweeping strategy and Fomel et al. give the factorisation.
//!
//! Their **Eq. (3)** splits the traveltime multiplicatively, `T = T₀·τ`, with
//! `|∇T₀| = S₀` (**Eq. 4**). Taking `S₀` constant and `T₀(x) = S₀·|x − x₀|` (the
//! analytic answer for a homogeneous medium at the source's own slowness) puts the
//! singularity entirely inside `T₀`, whose closed form handles it, and leaves `τ`
//! smooth.
//!
//! **Eq. (5)** is the equation the solver works on:
//!
//! ```text
//!     T₀²|∇τ|² + 2T₀τ ∇T₀·∇τ + (τ² − α²)S₀² = 0
//! ```

use crate::counts::exact;

/// Rounds of four sweeps before the solver refuses.
///
/// Termination is at grid convergence, so this is only an upper bound on compute.
/// Fomel et al. find three sweeps enough in general; rounds grow with the medium's
/// slowness contrast, not with the grid (`the_sweep_count_does_not_grow_with_the_mesh`).
/// Measured on the Wellington `Ohariu` segment: 9, 12, 13 and 14 rounds at contrasts
/// of 3.3x, 6.8x, 12.9x and 26x, roughly one more round per doubling.
///
/// The contrast a caller can present has the a priori bound
///
/// ```text
/// (max_fraction * Vs_max) / (min_fraction * Vs_min) * off_fault_factor
/// ```
///
/// which is 41.9x on Wellington (fully occupied, Vs 0.50 to 3.70) and 144.6x on the
/// Hikurangi interface, where 37% of the chart is off-fault and
/// `OFF_FAULT_SLOWNESS_FACTOR` multiplies the contrast by ten. The limit of 64 allows
/// for that 144.6x.
const MAX_ROUNDS: usize = 64;

/// How much a sweep must improve a cell's arrival for the sweep to count as unsettled,
/// as a fraction of the shortest single-cell traversal time on the grid.
///
/// Gauss-Seidel on the eikonal decreases towards its fixed point and never stops
/// improving in exact terms: without a tolerance the `changed` flag stays set while
/// sweeps shave femtoseconds off, and the round limit trips on round-off rather than on
/// anything about the medium. On the Wellington `Ohariu` segment at a 26x slowness
/// contrast, the field is physically settled by round 14, where the largest
/// improvement is 9.7e-9 s, and rounds 15 to 17 move cells by 2e-10, 3.6e-12 and
/// 2.7e-12 s. The same three-round tail appeared at every contrast from 3.3x to 26x:
/// the stopping rule causes it, not the problem.
///
/// The tolerance is a fraction of `min(spacing) * min(slowness)` rather than an absolute
/// time or a relative one. An absolute time would go wrong whenever the grid spacing
/// changed. A time relative to each cell's own arrival would degenerate near a seed at
/// time zero, and behave differently again for a segment seeded at an absolute jump time of twenty seconds,
/// which is exactly the case that first hit this.
///
/// The solver still *takes* an improvement below the tolerance (it's strictly better),
/// and the tolerance sets only when to stop, never what the answer is. This keeps the
/// error far smaller than the tolerance: against the exact fixed point on that same segment, where
/// the tolerance works out to 1.9e-8 s, the arrival times come out at most 2.4e-10 s late
/// and 1.7e-13 s late on average. Late and never early, since Gauss-Seidel decreases
/// towards the fixed point. The loose bound is the tolerance times the path length in
/// cells, around 1e-5 s over a three-hundred-cell path. The measurement is four orders
/// inside it, and either way it's far below the 0.005 s sample interval.
const CONVERGENCE_TOLERANCE: f64 = 1.0e-6;

/// Location and initiation time of hypocentre.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seed {
    pub i: usize,
    pub j: usize,
    pub t0_s: f64,
}

/// Error cases for the solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Error {
    /// A grid with no cells has no wavefront to solve.
    EmptyGrid { ni: usize, nj: usize },
    /// Slowness vs solver grid mismatch.
    WrongLength { ni: usize, nj: usize, got: usize },
    /// Grid spacing must be a positive, finite length on both axes.
    NonPositiveSpacing { axis: &'static str, value: f64 },
    /// Slowness must be positive and finite everywhere.
    NonPositiveSlowness { i: usize, j: usize, value: f64 },
    /// The solver needs at least one seed.
    NoSeeds,
    /// Every seed must lie inside the domain.
    SeedOutOfBounds {
        seed: usize,
        i: usize,
        j: usize,
        ni: usize,
        nj: usize,
    },
    /// A seed's initial time must be finite.
    NonFiniteSeedTime { seed: usize, t0_s: f64 },
    /// The sweep didn't settle in [`MAX_ROUNDS`] rounds.
    DidNotSettle { rounds: usize, contrast: f64 },
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
            Self::DidNotSettle { rounds, contrast } => write!(
                f,
                "the sweep did not settle in {rounds} rounds over a medium whose \
                 slowness spans {contrast:.1}x. Rounds needed grow with that \
                 contrast -- measured 9, 12, 13 and 14 at 3.3x, 6.8x, 12.9x and 26x \
                 -- so the first thing to check is the rupture velocity band: \
                 narrowing rupture_velocity_max_fraction or raising \
                 rupture_velocity_min_fraction lowers the contrast directly"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// First-arrival times from every seed, over the entire grid.
///
/// `slowness_s_per_km` is row-major over `(ni, nj)`. The convention is: `i` down-dip, `j` along-strike
/// and `spacing_km` is `(h_i, h_j)`, the cell size on each axis. The result has the
/// same layout, in seconds.
///
/// # Errors
///
/// See [`Error`].
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
/// diagnostic. Zhao's alternating orderings exist to make it a property of the
/// *medium* and not of the mesh. A solver whose rounds grew with the grid would cost
/// O(N log N) or worse. `tests/eikonal_contract.rs` asserts that a fourfold
/// refinement leaves the count where it was.
///
/// # Errors
///
/// See [`Error`].
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
#[derive(Clone, Copy)]
struct Known {
    time_s: f64,
    /// `(dT_0/di, dT_0/dj)` in physical units.
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
/// If a cell is never reached. This is a panic rather than an error because the
/// module's boundary checks the solver's inputs.
fn single_seed(
    slowness: &[f64],
    extent: (usize, usize),
    spacing_km: (f64, f64),
    seed: Seed,
) -> Result<(Vec<f64>, usize), Error> {
    let (ni, nj) = extent;
    let at = |i: usize, j: usize| i * nj + j;
    let source_slowness = slowness[at(seed.i, seed.j)];

    // The analytical solution on a homogeneous medium. The table stores only `T₀`,
    // and the update recomputes the gradient at its own node, which is cheaper than
    // loading it.
    let known: Vec<f64> = (0..ni)
        .flat_map(|i| (0..nj).map(move |j| (i, j)))
        .map(|(i, j)| known_factor(i, j, seed, spacing_km, source_slowness).time_s)
        .collect();

    // Solved in `T` throughout, converting to `τ` only where the discretisation
    // needs it. Keeping the array in `T` is what lets causality be a plain
    // comparison, and `T` is what the caller wants anyway.
    let mut times = vec![f64::INFINITY; ni * nj];
    times[at(seed.i, seed.j)] = 0.0;

    // Zhao's four alternating orderings: each ray direction falls in one of them, and
    // a fixed number of rounds suffices.
    let forward: Vec<usize> = (0..nj).collect();
    let backward: Vec<usize> = (0..nj).rev().collect();
    let down: Vec<usize> = (0..ni).collect();
    let up: Vec<usize> = (0..ni).rev().collect();

    // A fraction of the shortest single-cell traversal time: see
    // `CONVERGENCE_TOLERANCE`.
    let fastest_cell_s =
        spacing_km.0.min(spacing_km.1) * slowness.iter().copied().fold(f64::INFINITY, f64::min);
    let tolerance = CONVERGENCE_TOLERANCE * fastest_cell_s;

    let mut rounds = 0;
    for round in 1..=MAX_ROUNDS {
        let mut changed = false;
        for dips in [&down, &up] {
            for strikes in [&forward, &backward] {
                for &i in dips {
                    for &j in strikes {
                        if (i, j) == (seed.i, seed.j) {
                            continue;
                        }
                        let here = known_factor(i, j, seed, spacing_km, source_slowness);
                        let candidate = update(
                            &times,
                            &known,
                            here,
                            i,
                            j,
                            extent,
                            spacing_km,
                            slowness[at(i, j)],
                        );
                        if candidate < times[at(i, j)] {
                            // Taken either way. The tolerance sets only when to stop.
                            changed |= candidate < times[at(i, j)] - tolerance;
                            times[at(i, j)] = candidate;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
        rounds = round;
    }
    if rounds >= MAX_ROUNDS {
        let (lo, hi) = slowness
            .iter()
            .copied()
            .fold((f64::INFINITY, 0.0f64), |(lo, hi), s| {
                (lo.min(s), hi.max(s))
            });
        return Err(Error::DidNotSettle {
            rounds,
            contrast: if lo > 0.0 { hi / lo } else { f64::INFINITY },
        });
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

/// A reached neighbour on one axis: `(arrival, T₀, sign)`. `sign` is +1 when the
/// neighbour is at the lower index, matching the sign of the upwind difference.
type Side = (f64, f64, f64);

/// The earliest arrival this node's current neighbours allow.
///
/// Fomel et al. Eq. (7) on each of the four quadrant triangles, with the causality
/// condition, then the one-sided cap in place of their Eq. (8). Each axis has its
/// own spacing, and nothing here assumes the cells are square. `known` is
/// `T₀` on the grid and `here` is `T₀` with its gradient at `(i, j)`.
#[expect(
    clippy::too_many_arguments,
    reason = "the hot loop's inputs, passed flat so nothing is packed per cell"
)]
fn update(
    times: &[f64],
    known: &[f64],
    here: Known,
    i: usize,
    j: usize,
    extent: (usize, usize),
    spacing_km: (f64, f64),
    slowness: f64,
) -> f64 {
    let (ni, nj) = extent;
    let (h_i, h_j) = spacing_km;
    let at = |i: usize, j: usize| i * nj + j;

    // A neighbour off the grid or not yet reached contributes nothing to any
    // triangle and is causal against anything, exactly as an absent neighbour, and
    // `side` returns `None` for it.
    let side = |index: usize, sign: f64| {
        let arrival = times[index];
        arrival.is_finite().then_some((arrival, known[index], sign))
    };
    let along_j: [Option<Side>; 2] = [
        if j > 0 { side(at(i, j - 1), 1.0) } else { None },
        if j + 1 < nj {
            side(at(i, j + 1), -1.0)
        } else {
            None
        },
    ];
    let along_i: [Option<Side>; 2] = [
        if i > 0 { side(at(i - 1, j), 1.0) } else { None },
        if i + 1 < ni {
            side(at(i + 1, j), -1.0)
        } else {
            None
        },
    ];

    let term = |side: Option<Side>, gradient: f64, spacing: f64| match side {
        Some((arrival, factor, sign)) => {
            // From Fomel et al. Remark 1: at the source `T₀` is zero and `τ = T/T₀` is
            // 0/0. By l'Hôpital, or from Eq. (5) directly, `τ(x₀) = α(x₀)`, and with
            // the source's own slowness as `S₀`, that's exactly 1.
            let tau = if factor > 0.0 { arrival / factor } else { 1.0 };
            (
                sign * here.time_s / spacing + gradient,
                sign * here.time_s * tau / spacing,
            )
        }
        None => (0.0, 0.0),
    };

    let mut best = f64::INFINITY;

    // One triangle per quadrant, plus the two one-sided degenerations. Eq. (7) is a
    // quadratic in this node's `τ`, and the larger root is the causal branch.
    for x in along_j.into_iter().flatten().map(Some).chain([None]) {
        for y in along_i.into_iter().flatten().map(Some).chain([None]) {
            if x.is_none() && y.is_none() {
                continue;
            }
            let (a_x, b_x) = term(x, here.gradient.1, h_j);
            let (a_y, b_y) = term(y, here.gradient.0, h_i);
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

            // Fomel et al.'s causality condition.
            let causal =
                |side: Option<Side>| side.is_none_or(|(neighbour, _, _)| arrival >= neighbour);
            if causal(x) && causal(y) {
                best = best.min(arrival);
            }
        }
    }

    if best.is_finite() {
        return best;
    }

    // In place of Eq. (8), and **only** when no triangle produced a causal root. A
    // wave crossing one cell along an axis is always causal, so a node whose
    // triangles all failed still gets a bound rather than an infinite arrival.
    //
    // Offering this alongside the triangles rather than after them costs a factor of
    // six on a gradient. It's an unfactored first-order update, and wherever it's the
    // smaller of the two, it replaces the triangles' value and injects exactly the source-singularity error
    // the factorisation exists to remove. Measured at 1.03e-02 against 1.75e-03 on
    // the constant-gradient case.
    for (arrival, spacing) in along_j
        .into_iter()
        .flatten()
        .map(|side| (side.0, h_j))
        .chain(along_i.into_iter().flatten().map(|side| (side.0, h_i)))
    {
        best = best.min(arrival + spacing * slowness);
    }

    best
}
