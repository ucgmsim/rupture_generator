//! What any first-arrival solver must satisfy, whatever scheme it uses.
//!
//! This file exists to be pointed at a *replacement*. `Wavefront2d` is genslip's
//! expanding-square tracker reached through the original Fortran; the destination is
//! a fast-marching solver, and the question "is the new one acceptable?" has to have
//! an answer that is not "it agrees with the old one".
//!
//! # Why agreement with the old solver is the wrong test
//!
//! Every other Stage 3 swap changes the *arithmetic* and drifts at 1e-8: a different
//! FFT engine computes the same transform. A different eikonal scheme does not
//! compute the same travel times — it computes a different discretisation of the same
//! equation, and two first-order schemes differ by their own truncation error, of
//! order `h·log(1/h)`. On the corpus that is plausibly 0.1 to 0.2 s, past
//! `ENGINEERING_RULES.md`'s 0.05 s onset bound, and legitimately so.
//!
//! So the acceptance criterion is **closeness to analytic truth**, on problems where
//! truth is known, and it is recorded here for whatever solver is in place. A
//! replacement that is closer is an improvement even if it moves every onset in the
//! corpus.
//!
//! # What is asserted, and what is deliberately not
//!
//! Several plausible-sounding properties are false for a discrete solution and are
//! called out below rather than left for someone to add:
//!
//! - **"Travel time increases with distance from the source"** is false in a
//!   heterogeneous medium — a fast channel reaches a distant cell before a slow one
//!   nearby. The correct universal statement is *causality*: every cell has a
//!   neighbour that ruptured earlier.
//! - **The Lipschitz bound `|T(a) − T(b)| ≤ ‖a − b‖₂ / v_min`** holds for the
//!   viscosity solution and is violated by a first-order scheme by ~20% on the
//!   diagonal — legitimate discretisation error, not a bug. Only the *neighbour* form
//!   is sharp.
//! - **Mesh convergence on max relative error** does not converge: the worst cell
//!   sits at a fixed grid offset from the source, so its relative error is
//!   resolution-independent. Absolute error is what converges.

#![cfg(feature = "wavefront-compat")]

use genslip::rupture::{EikonalSolver, Hypocentre, SpeedGrid, TravelTimes};

/// A uniform medium, where the analytic solution is `distance / speed`.
fn uniform(strike_count: usize, dip_count: usize, speed_km_s: f32) -> SpeedGrid {
    SpeedGrid::new(
        strike_count,
        dip_count,
        vec![speed_km_s; strike_count * dip_count],
    )
}

/// A medium that varies in both directions, as a depth taper and a slip field give.
fn heterogeneous(strike_count: usize, dip_count: usize) -> SpeedGrid {
    let values = (0..strike_count * dip_count)
        .map(|index| {
            #[expect(clippy::cast_precision_loss, reason = "small test indices")]
            let (strike, dip) = ((index % strike_count) as f32, (index / strike_count) as f32);
            // 1.6 to about 3.4 km/s: the range a shallow rupture-speed taper spans.
            2.0 + 0.7 * (dip * 0.3).tanh() + 0.3 * (strike * 0.4).sin()
        })
        .collect();
    SpeedGrid::new(strike_count, dip_count, values)
}

/// Distance from the source in kilometres, straight line.
fn straight_line_km(strike: usize, dip: usize, hypocentre: Hypocentre, spacing_km: f64) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "small test indices")]
    let (across, down) = (
        strike as f64 - hypocentre.strike as f64,
        dip as f64 - hypocentre.dip as f64,
    );
    spacing_km * (across * across + down * down).sqrt()
}

/// Cells traversed from the source moving only along the axes.
fn lattice_cells(strike: usize, dip: usize, hypocentre: Hypocentre) -> usize {
    strike.abs_diff(hypocentre.strike) + dip.abs_diff(hypocentre.dip)
}

fn extremes(speed: &SpeedGrid) -> (f64, f64) {
    let mut slowest = f64::INFINITY;
    let mut fastest = 0.0_f64;
    for dip in 0..speed.dip_count() {
        for strike in 0..speed.strike_count() {
            let value = f64::from(speed.speed(strike, dip));
            slowest = slowest.min(value);
            fastest = fastest.max(value);
        }
    }
    (slowest, fastest)
}

// ---------------------------------------------------------------------------------
// Accuracy, against a solution that is known
// ---------------------------------------------------------------------------------

/// Along the axes, a first-arrival solver is exact.
///
/// Any upwind scheme reduces to the one-dimensional solution on a row or column
/// through the source, because the update there is one-sided and the medium is
/// uniform. There is no anisotropy error to make, so this is a genuine equality
/// rather than a bound — and a scheme that fails it is not upwind.
fn axis_rays_are_exact<E: EikonalSolver>(solver: &mut E) {
    let (strike_count, dip_count) = (65, 65);
    let (speed, spacing) = (2.5_f32, 0.5_f64);
    let hypocentre = Hypocentre {
        strike: strike_count / 2,
        dip: dip_count / 2,
    };
    let times = solver.solve(
        &uniform(strike_count, dip_count, speed),
        hypocentre,
        spacing,
    );

    // The solver is handed a speed in `f32` and inverts it there, so the slowness it
    // actually integrates is `f64::from(1.0f32 / v)` rather than `1.0 / f64::from(v)`.
    // Those differ in the eighth digit, which is larger than the exactness claimed
    // here -- so the reference is built from the same slowness, and the claim stays
    // about the *scheme* rather than about a unit conversion.
    let slowness = f64::from(1.0_f32 / speed);

    for offset in 1..strike_count / 2 {
        #[expect(clippy::cast_precision_loss, reason = "small test indices")]
        let exact = offset as f64 * spacing * slowness;
        for cell in [
            times.time(hypocentre.strike + offset, hypocentre.dip),
            times.time(hypocentre.strike - offset, hypocentre.dip),
            times.time(hypocentre.strike, hypocentre.dip + offset),
            times.time(hypocentre.strike, hypocentre.dip - offset),
        ] {
            assert!(
                (cell - exact).abs() < 1e-9 * exact,
                "on an axis at {offset} cells: {cell} against an exact {exact}"
            );
        }
    }
}

/// Off-axis error stays inside the envelope a first-order scheme guarantees.
///
/// A point source on a grid gives `O(h·log(1/h))` error, worst on the diagonal, and
/// it does **not** vanish with resolution in relative terms near the source. The
/// bound is `ln(2 + d/h) / (d/h)` — the classical form, with a single documented
/// constant of one in front.
///
/// Returns the worst relative error, so the swap has a number to be judged against.
fn point_source_error<E: EikonalSolver>(solver: &mut E) -> f64 {
    let (strike_count, dip_count) = (65, 65);
    let (speed, spacing) = (2.5_f32, 0.5_f64);
    let hypocentre = Hypocentre {
        strike: strike_count / 2,
        dip: dip_count / 2,
    };
    let times = solver.solve(
        &uniform(strike_count, dip_count, speed),
        hypocentre,
        spacing,
    );

    let mut worst = 0.0_f64;
    for dip in 0..dip_count {
        for strike in 0..strike_count {
            let distance = straight_line_km(strike, dip, hypocentre, spacing);
            if distance == 0.0 {
                continue;
            }
            let exact = distance / f64::from(speed);
            let relative = (times.time(strike, dip) - exact).abs() / exact;
            let cells = distance / spacing;
            let envelope = (2.0 + cells).ln() / cells;
            assert!(
                relative <= envelope,
                "({strike}, {dip}) is {relative:.5} off at {cells:.1} cells, past an \
                 envelope of {envelope:.5}"
            );
            worst = worst.max(relative);
        }
    }
    worst
}

/// Halving the grid step halves the error, near enough.
///
/// Stated on **absolute** error in seconds, not relative: the worst relative error
/// sits at a fixed offset in cells from the source, so it is the same at every
/// resolution and a convergence test written on it looks broken. A first-order scheme
/// gives a ratio around 0.6; anything better is better.
fn refining_the_grid_converges<E: EikonalSolver>(solver: &mut E) {
    let error_at = |solver: &mut E, cells: usize, spacing: f64| {
        let hypocentre = Hypocentre {
            strike: cells / 2,
            dip: cells / 2,
        };
        let times = solver.solve(&uniform(cells, cells, 2.5), hypocentre, spacing);
        let mut worst = 0.0_f64;
        for dip in 0..cells {
            for strike in 0..cells {
                let exact = straight_line_km(strike, dip, hypocentre, spacing) / 2.5;
                worst = worst.max((times.time(strike, dip) - exact).abs());
            }
        }
        worst
    };

    // The same physical fault, twice the resolution.
    let coarse = error_at(solver, 33, 1.0);
    let fine = error_at(solver, 65, 0.5);
    assert!(
        fine <= 0.8 * coarse,
        "refining halved the step and moved the absolute error from {coarse:.5} s to \
         {fine:.5} s, which is not convergence"
    );
}

// ---------------------------------------------------------------------------------
// Properties that need no analytic solution
// ---------------------------------------------------------------------------------

/// First arrival is sandwiched between the straight line and the lattice path.
///
/// The strongest check available on a medium with no closed-form solution, and the
/// one that catches the errors worth catching: slowness used where speed belongs, the
/// grid step wrong, the axes transposed, the source in the wrong cell.
///
/// Above: the axis-only path is *a* path, so first arrival cannot beat it. Below: no
/// path is shorter than the straight line and none is faster than the medium's
/// fastest cell, less one cell's slack for how the source cell itself is discretised.
fn arrival_is_between_the_straight_line_and_the_lattice_path<E: EikonalSolver>(solver: &mut E) {
    let (strike_count, dip_count) = (40, 24);
    let spacing = 0.75_f64;
    let speed = heterogeneous(strike_count, dip_count);
    let (slowest, fastest) = extremes(&speed);
    let hypocentre = Hypocentre { strike: 11, dip: 7 };
    let times = solver.solve(&speed, hypocentre, spacing);

    for dip in 0..dip_count {
        for strike in 0..strike_count {
            let arrival = times.time(strike, dip);

            #[expect(clippy::cast_precision_loss, reason = "small test indices")]
            let along_axes = lattice_cells(strike, dip, hypocentre) as f64 * spacing / slowest;
            assert!(
                arrival <= along_axes + 1e-9,
                "({strike}, {dip}) arrives at {arrival}, later than the {along_axes} \
                 an axis-only path would take"
            );

            let straight = straight_line_km(strike, dip, hypocentre, spacing) / fastest;
            assert!(
                arrival >= straight - spacing / slowest,
                "({strike}, {dip}) arrives at {arrival}, sooner than the {straight} a \
                 straight line at the medium's fastest speed would take"
            );
        }
    }
}

/// Every cell has a neighbour that ruptured before it.
///
/// The correct form of "the front expands outward". Distance monotonicity is *false*
/// here: in a heterogeneous medium a fast channel reaches a far cell before a slow
/// one nearby. Causality is not — a first arrival came from somewhere, and on a
/// four-connected grid that somewhere is a face neighbour.
fn every_cell_ruptures_after_a_neighbour<E: EikonalSolver>(solver: &mut E) {
    let (strike_count, dip_count) = (40, 24);
    let hypocentre = Hypocentre { strike: 11, dip: 7 };
    let times = solver.solve(&heterogeneous(strike_count, dip_count), hypocentre, 0.75);

    for dip in 0..dip_count {
        for strike in 0..strike_count {
            if (strike, dip) == (hypocentre.strike, hypocentre.dip) {
                continue;
            }
            let here = times.time(strike, dip);
            let earlier = neighbours(strike, dip, strike_count, dip_count)
                .into_iter()
                .any(|(s, d)| times.time(s, d) < here);
            assert!(
                earlier,
                "({strike}, {dip}) at {here} is a local minimum; nothing reached it"
            );
        }
    }
}

/// Adjacent cells differ by no more than one cell's travel at the slowest speed.
///
/// The *sharp* Lipschitz statement, and the only one a first-order scheme satisfies:
/// the update enforces it structurally between face neighbours. Its Euclidean cousin,
/// `|T(a) − T(b)| ≤ ‖a − b‖₂ / v_min`, is a property of the exact viscosity solution
/// and is violated on the diagonal by about 20% — which is discretisation error, not
/// a defect, and asserting it would be asserting something false about the scheme.
fn neighbouring_cells_are_lipschitz<E: EikonalSolver>(solver: &mut E) {
    let (strike_count, dip_count) = (40, 24);
    let spacing = 0.75_f64;
    let speed = heterogeneous(strike_count, dip_count);
    let (slowest, _) = extremes(&speed);
    let times = solver.solve(&speed, Hypocentre { strike: 11, dip: 7 }, spacing);

    let bound = spacing / slowest;
    for dip in 0..dip_count {
        for strike in 0..strike_count {
            let here = times.time(strike, dip);
            for (s, d) in neighbours(strike, dip, strike_count, dip_count) {
                let step = (times.time(s, d) - here).abs();
                assert!(
                    step <= bound * (1.0 + 1e-6),
                    "({strike}, {dip}) to ({s}, {d}) jumps {step} s, past the {bound} \
                     one cell at the slowest speed takes"
                );
            }
        }
    }
}

/// A faster medium never ruptures later.
///
/// Two solves, and no analytic solution needed. It catches the error the sandwich
/// bound is weakest against: a solver that inverts its input, using speed where
/// slowness belongs, still produces a plausible field but reverses this.
fn raising_the_speed_never_delays_anything<E: EikonalSolver>(solver: &mut E) {
    let (strike_count, dip_count) = (30, 18);
    let hypocentre = Hypocentre { strike: 9, dip: 5 };
    let base = heterogeneous(strike_count, dip_count);

    let faster = SpeedGrid::new(
        strike_count,
        dip_count,
        (0..strike_count * dip_count)
            .map(|index| base.speed(index % strike_count, index / strike_count) * 1.2)
            .collect(),
    );

    let slow = solver.solve(&base, hypocentre, 0.75);
    let quick = solver.solve(&faster, hypocentre, 0.75);

    for dip in 0..dip_count {
        for strike in 0..strike_count {
            assert!(
                quick.time(strike, dip) <= slow.time(strike, dip) + 1e-12,
                "({strike}, {dip}) got later when the medium got faster"
            );
        }
    }
}

/// The source is where it was asked for, and it is the only zero.
fn the_source_is_the_only_zero<E: EikonalSolver>(solver: &mut E) {
    let hypocentre = Hypocentre { strike: 11, dip: 7 };
    let times: TravelTimes = solver.solve(&heterogeneous(40, 24), hypocentre, 0.75);

    assert_eq!(
        times.time(hypocentre.strike, hypocentre.dip).to_bits(),
        0.0_f64.to_bits()
    );
    for dip in 0..times.dip_count() {
        for strike in 0..times.strike_count() {
            if (strike, dip) != (hypocentre.strike, hypocentre.dip) {
                assert!(
                    times.time(strike, dip) > 0.0,
                    "({strike}, {dip}) is not later"
                );
            }
        }
    }
}

fn neighbours(
    strike: usize,
    dip: usize,
    strike_count: usize,
    dip_count: usize,
) -> Vec<(usize, usize)> {
    let mut cells = Vec::with_capacity(4);
    if strike > 0 {
        cells.push((strike - 1, dip));
    }
    if strike + 1 < strike_count {
        cells.push((strike + 1, dip));
    }
    if dip > 0 {
        cells.push((strike, dip - 1));
    }
    if dip + 1 < dip_count {
        cells.push((strike, dip + 1));
    }
    cells
}

/// Run the whole contract against one solver.
macro_rules! contract_for {
    ($name:ident, $solver:expr) => {
        mod $name {
            #[test]
            fn axis_rays_are_exact() {
                super::axis_rays_are_exact(&mut $solver);
            }

            /// The measurement the eikonal swap is judged against.
            ///
            /// `Wavefront2d` measures **0.00881**. A replacement is acceptable if it
            /// is *closer to analytic truth*, not if it agrees with this solver --
            /// see the module docstring. The bound below is deliberately slack in
            /// one direction only: a better scheme must pass, a worse one must not.
            #[test]
            fn the_point_source_error_stays_inside_its_envelope() {
                let worst = super::point_source_error(&mut $solver);
                println!(
                    "worst relative error against the analytic point-source \
                     solution: {worst:.5}"
                );
                assert!(
                    worst < 0.02,
                    "{worst:.5} is worse than twice what the expanding-square \
                     tracker manages"
                );
            }

            #[test]
            fn refining_the_grid_converges() {
                super::refining_the_grid_converges(&mut $solver);
            }

            #[test]
            fn arrival_is_bounded_both_ways() {
                super::arrival_is_between_the_straight_line_and_the_lattice_path(&mut $solver);
            }

            #[test]
            fn causality_holds() {
                super::every_cell_ruptures_after_a_neighbour(&mut $solver);
            }

            #[test]
            fn neighbours_are_lipschitz() {
                super::neighbouring_cells_are_lipschitz(&mut $solver);
            }

            #[test]
            fn a_faster_medium_never_ruptures_later() {
                super::raising_the_speed_never_delays_anything(&mut $solver);
            }

            #[test]
            fn the_source_is_the_only_zero() {
                super::the_source_is_the_only_zero(&mut $solver);
            }
        }
    };
}

contract_for!(wavefront2d, genslip::rupture::Wavefront2d::new());
