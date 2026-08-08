//! Factored fast sweeping: the eikonal equation with its source singularity removed.
//!
//! # The problem this solves
//!
//! Every scheme that discretises `|∇T| = s` directly loses accuracy at a point source,
//! because `T` is not smooth there — `∇T` is discontinuous in direction — and the
//! error that creates spreads over the whole grid. It shows up as a *failure to
//! converge*: refining the mesh does not help, because the near-source error is a
//! fixed fraction rather than a truncation term. Both solvers this replaces have it.
//! `tests/eikonal_contract.rs` measures them at ratios of 1.03 and 1.01 where a
//! converging scheme gives 2.
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

use crate::grid::FaultAxes;
use crate::rupture::{EikonalSolver, Hypocentre, SpeedGrid, TravelTimes};
use crate::units;

/// Rounds of four sweeps before giving up on convergence.
///
/// Fomel et al. report 3 for their examples, at every mesh size from 150×50 to
/// 1200×400 — the count is a property of the medium, not of the grid, which is what
/// Zhao's alternating orderings buy. This is a generous ceiling; the loop exits as
/// soon as a round changes nothing.
const MAX_ROUNDS: usize = 16;

/// Fast sweeping on the factored eikonal equation.
#[derive(Clone, Copy, Debug, Default)]
pub struct FactoredSweep {
    rounds: usize,
}

impl FactoredSweep {
    #[must_use]
    pub const fn new() -> Self {
        Self { rounds: 0 }
    }

    /// Rounds of four sweeps the last solve took, including the one that settled.
    ///
    /// Exposed because it is the evidence for the cost claim rather than a
    /// diagnostic: Zhao's alternating orderings exist to make this a property of the
    /// *medium* and not of the mesh, so a solver whose round count grew with the grid
    /// would be O(N log N) or worse in disguise. `tests/eikonal_contract.rs` asserts
    /// it stays put across a fourfold refinement.
    #[must_use]
    pub const fn rounds(&self) -> usize {
        self.rounds
    }
}

/// The known factor `T₀` and its gradient at one node, in seconds and s/km.
///
/// `T₀ = S₀·r` with `r` the distance to the source, so `∇T₀ = S₀·r̂` — a unit vector
/// scaled by the source slowness, undefined only at the source itself where the
/// paper's Remark 1 supplies `τ = α` by l'Hôpital instead.
struct Known {
    time_s: f64,
    gradient: (f64, f64),
}

fn known_factor(
    strike: usize,
    dip: usize,
    hypocentre: Hypocentre,
    spacing_km: f64,
    source_slowness: f64,
) -> Known {
    let (across, down) = (
        units::exact(strike) - units::exact(hypocentre.strike),
        units::exact(dip) - units::exact(hypocentre.dip),
    );
    let radius = spacing_km * (across * across + down * down).sqrt();
    if radius == 0.0 {
        return Known {
            time_s: 0.0,
            gradient: (0.0, 0.0),
        };
    }
    let scale = source_slowness / (radius / spacing_km);
    Known {
        time_s: source_slowness * radius,
        gradient: (scale * across, scale * down),
    }
}

impl EikonalSolver for FactoredSweep {
    /// # Panics
    ///
    /// If the hypocentre is outside the fault, or if a subfault is never reached.
    /// `SpeedGrid` already refuses a non-positive speed, so every cell is reachable
    /// and the second panic is a statement about the solver rather than the input.
    fn solve(&mut self, speed: &SpeedGrid, hypocentre: Hypocentre, spacing_km: f64) -> TravelTimes {
        let (strike_count, dip_count) = (speed.strike_count(), speed.dip_count());
        assert!(
            hypocentre.strike < strike_count && hypocentre.dip < dip_count,
            "hypocentre ({}, {}) is outside a {strike_count}x{dip_count} fault",
            hypocentre.strike,
            hypocentre.dip
        );

        let slowness = |strike: usize, dip: usize| 1.0 / speed[[dip, strike]];
        let source_slowness = slowness(hypocentre.strike, hypocentre.dip);
        let at = |strike: usize, dip: usize| strike + dip * strike_count;

        // `T₀` and `∇T₀` are analytic, so they are computed once rather than swept.
        let known: Vec<Known> = (0..dip_count)
            .flat_map(|dip| (0..strike_count).map(move |strike| (strike, dip)))
            .map(|(strike, dip)| known_factor(strike, dip, hypocentre, spacing_km, source_slowness))
            .collect();

        // Solved in `T` throughout, converting to `τ` only where the discretisation
        // needs it. Keeping the array in `T` is what lets causality be a plain
        // comparison, and `T` is what the caller wants anyway.
        let mut times = vec![f64::INFINITY; strike_count * dip_count];
        times[at(hypocentre.strike, hypocentre.dip)] = 0.0;

        // Zhao's four alternating orderings: every ray direction is covered by one of
        // them, which is why a fixed number of rounds suffices.
        let forward: Vec<usize> = (0..strike_count).collect();
        let backward: Vec<usize> = (0..strike_count).rev().collect();
        let down: Vec<usize> = (0..dip_count).collect();
        let up: Vec<usize> = (0..dip_count).rev().collect();

        let mut rounds = 0;
        loop {
            let mut changed = false;
            for dips in [&down, &up] {
                for strikes in [&forward, &backward] {
                    for &dip in dips {
                        for &strike in strikes {
                            if (strike, dip) == (hypocentre.strike, hypocentre.dip) {
                                continue;
                            }
                            let candidate = update(
                                &times,
                                &known,
                                strike,
                                dip,
                                strike_count,
                                dip_count,
                                spacing_km,
                                slowness(strike, dip),
                            );
                            if candidate < times[at(strike, dip)] {
                                times[at(strike, dip)] = candidate;
                                changed = true;
                            }
                        }
                    }
                }
            }
            rounds += 1;
            if !changed {
                self.rounds = rounds;
                break;
            }
            assert!(
                rounds < MAX_ROUNDS,
                "the sweep did not settle in {MAX_ROUNDS} rounds; the medium has \
                 structure this scheme does not handle"
            );
        }

        for (index, arrival) in times.iter().enumerate() {
            assert!(
                arrival.is_finite(),
                "({}, {}) was never reached",
                index % strike_count,
                index / strike_count
            );
        }

        crate::grid::from_values(strike_count, dip_count, times)
    }
}

/// The best arrival this node can be given from its current neighbours.
///
/// Fomel et al. Eq. (7) on each of the four quadrant triangles, with the causality
/// condition, then the one-sided cap that stands in for their Eq. (8).
#[expect(
    clippy::too_many_arguments,
    reason = "a local solver takes the node, its grid and its medium; bundling them \
              would build a struct that exists for one call"
)]
fn update(
    times: &[f64],
    known: &[Known],
    strike: usize,
    dip: usize,
    strike_count: usize,
    dip_count: usize,
    spacing_km: f64,
    slowness: f64,
) -> f64 {
    let at = |strike: usize, dip: usize| strike + dip * strike_count;
    let here = &known[at(strike, dip)];

    // The two neighbours on each axis, as (arrival, its own T₀, which side).
    // `sign` is +1 when the neighbour is at the lower index, matching the sign the
    // upwind difference carries.
    let mut across: Vec<(f64, f64, f64)> = Vec::with_capacity(2);
    if strike > 0 {
        let index = at(strike - 1, dip);
        across.push((times[index], known[index].time_s, 1.0));
    }
    if strike + 1 < strike_count {
        let index = at(strike + 1, dip);
        across.push((times[index], known[index].time_s, -1.0));
    }
    let mut down: Vec<(f64, f64, f64)> = Vec::with_capacity(2);
    if dip > 0 {
        let index = at(strike, dip - 1);
        down.push((times[index], known[index].time_s, 1.0));
    }
    if dip + 1 < dip_count {
        let index = at(strike, dip + 1);
        down.push((times[index], known[index].time_s, -1.0));
    }

    let mut best = f64::INFINITY;

    // One triangle per quadrant, plus the two one-sided degenerations. Eq. (7) is a
    // quadratic in this node's `τ`; the larger root is the causal branch.
    for corner in across
        .iter()
        .map(Some)
        .chain([None])
        .flat_map(|x| down.iter().map(Some).chain([None]).map(move |y| (x, y)))
    {
        let (x, y) = corner;
        if x.is_none() && y.is_none() {
            continue;
        }

        let term = |side: Option<&(f64, f64, f64)>, gradient: f64| match side {
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
                    sign * here.time_s / spacing_km + gradient,
                    sign * here.time_s * tau / spacing_km,
                ))
            }
            _ => None,
        };
        let (a_x, b_x) = term(x, here.gradient.0).unwrap_or((0.0, 0.0));
        let (a_y, b_y) = term(y, here.gradient.1).unwrap_or((0.0, 0.0));
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
    for side in across.iter().chain(&down) {
        if side.0.is_finite() {
            best = best.min(side.0 + spacing_km * slowness);
        }
    }

    best
}
