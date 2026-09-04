//! Factored fast sweeping: the eikonal equation with its source singularity removed.
//!
//! # Reference papers
//!
//! > **Zhao, H. (2005).** A fast sweeping method for eikonal equations.
//! > *Mathematics of Computation* **74**(250), 603–627.
//! >
//! > **Fomel, S., Luo, S. & Zhao, H. (2009).** Fast sweeping method for the factored
//! > eikonal equation. *Journal of Computational Physics* **228**(17), 6440–6455.
//!
//! Zhao gives the sweeping strategy and Fomel et al. give the factorisation.
//!
//! Their **Eq. (3)** splits the traveltime multiplicatively, `T = T₀·τ`, with
//! `|∇T₀| = S₀` (**Eq. 4**). Taking `S₀` constant and `T₀(x) = S₀·|x − x₀|` — the
//! analytic answer for a homogeneous medium at the source's own slowness — puts the
//! singularity entirely inside `T₀`, where it is known in closed form, and leaves `τ`
//! smooth.
//!
//! **Eq. (5)** is what is actually solved:
//!
//! ```text
//!     T₀²|∇τ|² + 2T₀τ ∇T₀·∇τ + (τ² − α²)S₀² = 0
//! ```

use crate::counts::exact;

/// Rounds of four sweeps before giving up on convergence.
///
/// Fomel et al. show only 3 sweeps worked generally, this is a pessimistic
/// assumption and an upper bound on compute because termination occurs at grid
/// convergence.
/// How many alternating-ordering sweeps the solver will make before refusing.
///
/// Rounds grow with the medium's slowness contrast, not with the grid -- see
/// `the_sweep_count_does_not_grow_with_the_mesh`. Measured on the shipped Wellington
/// `Ohariu` segment: 9, 12, 13 and 14 rounds at contrasts of 3.3x, 6.8x, 12.9x and 26x,
/// so roughly one more round per doubling.
///
/// The contrast a caller can present is bounded a priori, which is what makes this
/// number checkable rather than a guess. It is
///
/// ```text
/// (max_fraction * Vs_max) / (min_fraction * Vs_min) * off_fault_factor
/// ```
///
/// which the velocity band caps whatever the depth profile does. For the shipped examples
/// that bound is 41.9x on Wellington (fully
/// occupied, Vs 0.50 to 3.70) and 144.6x on the Hikurangi interface, where 37% of the
/// chart is off-fault and `OFF_FAULT_SLOWNESS_FACTOR` multiplies the contrast by ten. The
/// wall dominates the band by a wide margin on a resampled interface.
///
/// 64 is set against that 144.6x rather than against the case that first failed. A limit
/// of 16 was calibrated when the band ceiling was the shear speed and three of every
/// run's rounds went on round-off; both of those changed.
const MAX_ROUNDS: usize = 64;

/// How much a sweep must improve a cell's arrival for the sweep to count as unsettled,
/// as a fraction of the fastest single-cell traversal on the grid.
///
/// Gauss-Seidel on the eikonal approaches its fixed point from above and never stops
/// improving in exact terms: without a tolerance the `changed` flag stays set while
/// sweeps shave femtoseconds off, and the round limit trips on round-off rather than on
/// anything about the medium. Measured on the shipped Wellington `Ohariu` segment at a
/// 26x slowness contrast: the field is physically settled by round 14, where the largest
/// improvement is 9.7e-9 s, and rounds 15 to 17 move cells by 2e-10, 3.6e-12 and
/// 2.7e-12 s. That three-round round-off tail was measured at every contrast from 3.3x
/// to 26x, so it is the stopping rule and not the problem.
///
/// The tolerance is a fraction of `min(spacing) * min(slowness)` rather than an absolute
/// time or a relative one. Absolute would not survive a change of grid spacing; relative
/// to each cell's own arrival would degenerate near a seed at time zero and behave
/// differently again for a segment seeded at an absolute jump time of twenty seconds,
/// which is exactly the case that first hit this.
///
/// An improvement below the tolerance is still *taken* -- it is strictly better -- so the
/// tolerance decides when to stop, never what the answer is. That is why the error is far
/// smaller than the tolerance: against the exact fixed point on that same segment, where
/// the tolerance works out to 1.9e-8 s, the arrival times come out at most 2.4e-10 s late
/// and 1.7e-13 s late on average. Late and never early, since Gauss-Seidel approaches the
/// fixed point from above. The loose bound is the tolerance times the path length in
/// cells, around 1e-5 s over a three-hundred-cell path; the measurement is four orders
/// inside it, and either way it is far below the 0.005 s sample interval.
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
    /// Solver must be supplied at least one seed.
    NoSeeds,
    /// Seeds must be located inside domain.
    SeedOutOfBounds {
        seed: usize,
        i: usize,
        j: usize,
        ni: usize,
        nj: usize,
    },
    /// A seed's initial time must be finite.
    NonFiniteSeedTime { seed: usize, t0_s: f64 },
    /// The sweep did not settle in [`MAX_ROUNDS`] rounds.
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

/// First-arrival times from every seed, on the whole grid.
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
/// diagnostic: Zhao's alternating orderings exist to make it a property of the
/// *medium* and not of the mesh, so a solver whose rounds grew with the grid would be
/// O(N log N) or worse in disguise. `tests/eikonal_contract.rs` asserts it stays put
/// across a fourfold refinement.
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
/// If a cell is never reached, which is considered a panic rather than an error
/// because the solver's inputs should be checked at the boundary of the module.
fn single_seed(
    slowness: &[f64],
    extent: (usize, usize),
    spacing_km: (f64, f64),
    seed: Seed,
) -> Result<(Vec<f64>, usize), Error> {
    let (ni, nj) = extent;
    let at = |i: usize, j: usize| i * nj + j;
    let source_slowness = slowness[at(seed.i, seed.j)];

    // Calculate the analytical solution on a homogeneous medium.
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

    // A fraction of the fastest single-cell traversal: see `CONVERGENCE_TOLERANCE`.
    let fastest_cell_s = spacing_km.0.min(spacing_km.1)
        * slowness
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
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
                        let candidate =
                            update(&times, &known, i, j, extent, spacing_km, slowness[at(i, j)]);
                        if candidate < times[at(i, j)] {
                            // Taken either way; the tolerance only decides when to stop.
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
            .fold((f64::INFINITY, 0.0f64), |(lo, hi), s| (lo.min(s), hi.max(s)));
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
                // From Fomel et al. Remark 1: at the source `T₀` is zero and `τ = T/T₀` is 0/0. By
                // l'Hôpital, or from Eq. (5) directly, `τ(x₀) = α(x₀)` — and with
                // `S₀` taken as the source's own slowness that is exactly 1.
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

        // Fomel et al.'s causality condition.
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
