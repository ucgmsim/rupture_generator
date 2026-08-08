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

/// Speed at the surface, and its increase per kilometre of depth.
///
/// A crustal-looking profile: 2.0 km/s at the top rising to 3.5 km/s at 25 km.
const GRADIENT_SURFACE_KM_S: f64 = 2.0;
const GRADIENT_PER_KM: f64 = 0.06;

/// A medium whose speed rises linearly with depth — and has a closed-form solution.
///
/// This is the *discriminating* accuracy test. A uniform medium says only that a
/// scheme is exact where the answer is trivial; a linear gradient bends the rays into
/// circular arcs and still has an analytic first-arrival time, so it separates schemes
/// that reproduce curvature from schemes that assume plane waves.
fn linear_gradient(strike_count: usize, dip_count: usize, spacing_km: f64) -> SpeedGrid {
    let values = (0..strike_count * dip_count)
        .map(|index| {
            #[expect(clippy::cast_precision_loss, reason = "small test indices")]
            let depth_km = (index / strike_count) as f64 * spacing_km;
            #[expect(clippy::cast_possible_truncation, reason = "the grid is f32")]
            let speed = (GRADIENT_SURFACE_KM_S + GRADIENT_PER_KM * depth_km) as f32;
            speed
        })
        .collect();
    SpeedGrid::new(strike_count, dip_count, values)
}

/// First-arrival time in a constant-gradient medium, exactly.
///
/// For `v(z) = v₀ + g·z` the rays are circular arcs and
///
/// ```text
///     T = (1/g) · arccosh(1 + g²·d² / (2·v₁·v₂))
/// ```
///
/// with `d` the straight-line distance and `v₁`, `v₂` the speeds at the two endpoints.
/// Note it depends on the endpoints' *depths* only through their speeds, not on the
/// path — which is what makes it usable as a per-cell reference.
fn gradient_arrival_s(strike: usize, dip: usize, hypocentre: Hypocentre, spacing_km: f64) -> f64 {
    let speed_at = |dip: usize| {
        #[expect(clippy::cast_precision_loss, reason = "small test indices")]
        let depth = dip as f64 * spacing_km;
        GRADIENT_SURFACE_KM_S + GRADIENT_PER_KM * depth
    };
    let distance = straight_line_km(strike, dip, hypocentre, spacing_km);
    let cosh = 1.0
        + GRADIENT_PER_KM.powi(2) * distance * distance
            / (2.0 * speed_at(hypocentre.dip) * speed_at(dip));
    cosh.acosh() / GRADIENT_PER_KM
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

    // Compared against true analytic slowness, at a tolerance that admits an `f32`
    // inversion. `Wavefront2d` inverts the speed in `f32` before widening, so its
    // slowness differs from `1/v` in the eighth digit; `FastMarching` builds its cost
    // field in `f64` and does not. Asserting bit-exactness would be asserting which
    // width a solver inverts in, which is not a property of the scheme.
    let slowness = 1.0 / f64::from(speed);

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
                (cell - exact).abs() < 1e-6 * exact,
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

/// Worst error against the analytic solution for a linear velocity gradient.
///
/// The measurement that separates the schemes. A uniform medium rewards any scheme
/// that is exact for plane waves; a gradient bends the rays and asks whether the
/// scheme reproduces curvature.
fn gradient_error<E: EikonalSolver>(solver: &mut E, cells: usize, spacing: f64) -> f64 {
    let hypocentre = Hypocentre {
        strike: cells / 2,
        dip: cells / 2,
    };
    let times = solver.solve(&linear_gradient(cells, cells, spacing), hypocentre, spacing);

    let mut worst = 0.0_f64;
    for dip in 0..cells {
        for strike in 0..cells {
            if (strike, dip) == (hypocentre.strike, hypocentre.dip) {
                continue;
            }
            let exact = gradient_arrival_s(strike, dip, hypocentre, spacing);
            worst = worst.max((times.time(strike, dip) - exact).abs() / exact);
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

            /// The measurement the solver choice is judged on. See `ACCURACY`.
            #[test]
            fn the_point_source_error_stays_inside_its_envelope() {
                let worst = super::point_source_error(&mut $solver);
                println!(
                    "{}: worst relative error against the analytic point-source \
                     solution is {worst:.5}",
                    stringify!($name)
                );
            }

            #[test]
            fn refining_the_grid_converges() {
                super::refining_the_grid_converges(&mut $solver);
            }

            /// The discriminating accuracy measurement, and the convergence order.
            ///
            /// A scheme that reproduces curvature converges at first order here —
            /// the error halves with `h`. One that does not converges more slowly,
            /// because the point-source singularity pollutes it. Reported rather
            /// than bounded, because this test exists to *compare* solvers.
            #[test]
            fn the_gradient_error_and_its_convergence() {
                let coarse = super::gradient_error(&mut $solver, 33, 1.0);
                let fine = super::gradient_error(&mut $solver, 65, 0.5);
                println!(
                    "{}: linear gradient, worst relative error {coarse:.5e} at h=1.0, \
                     {fine:.5e} at h=0.5, ratio {:.2}",
                    stringify!($name),
                    coarse / fine
                );
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

/// Every solver's error against a solution that is known, and how it converges.
///
/// **The table the solver choice turns on.** Two analytic problems: a uniform medium,
/// where any scheme exact for plane waves does well, and a linear velocity gradient,
/// which bends the rays into circular arcs and asks whether the scheme reproduces
/// curvature. The second is the discriminating one.
///
/// | solver | uniform | gradient, h=1.0 | gradient, h=0.5 | ratio |
/// | --- | --- | --- | --- | --- |
/// | `Wavefront2d` | 0.0088 | 2.44e-02 | 2.37e-02 | **1.03** |
/// | `FastMarching` | 0.207 | 2.12e-01 | 2.10e-01 | **1.01** |
///
/// The ratio column is the finding. **Neither solver converges.** Halving the grid
/// step barely moves either error, so neither achieves even first-order accuracy on a
/// heterogeneous medium — which is not a defect in the stencils but the point-source
/// singularity polluting the whole field. Fomel, Luo & Zhao (2009) say it plainly of
/// the unfactored equation: it *"cannot achieve first order accuracy due to the
/// singularity at the point source"*, and measure a ratio of 1.65 where the factored
/// form gives 2.00.
///
/// So the expanding-square tracker's advantage over plain fast marching is real but
/// bounded: it is a better stencil applied to the same badly-posed problem. On the
/// uniform medium it is 23x better; on a gradient, only 9x, and both are stuck.
///
/// A replacement is acceptable if it is closer to analytic truth on **both** problems
/// *and* converges. That is a higher bar than either of these clears.
const ACCURACY: [(&str, f64); 2] = [("Wavefront2d", 0.0088), ("FastMarching", 0.2072)];

/// Neither existing solver converges on a heterogeneous medium.
///
/// Recorded as an assertion rather than a remark, because it is the justification for
/// replacing both and it would be easy to forget. A scheme that reproduces the
/// point-source curvature gives a ratio near 2; these give 1.03 and 1.01.
#[cfg(feature = "wavefront-compat")]
#[test]
fn neither_existing_solver_converges_on_a_gradient() {
    for (name, coarse, fine) in [
        (
            "Wavefront2d",
            gradient_error(&mut genslip::rupture::Wavefront2d::new(), 33, 1.0),
            gradient_error(&mut genslip::rupture::Wavefront2d::new(), 65, 0.5),
        ),
        (
            "FastMarching",
            gradient_error(&mut genslip::rupture::FastMarching::new(), 33, 1.0),
            gradient_error(&mut genslip::rupture::FastMarching::new(), 65, 0.5),
        ),
    ] {
        let ratio = coarse / fine;
        assert!(
            ratio < 1.3,
            "{name} now converges at a ratio of {ratio:.2}; if a scheme here started \
             reproducing the source curvature, the comparison table is stale"
        );
    }
}

/// The accuracy gap, asserted so it cannot quietly close or widen.
///
/// Recorded rather than tolerated: if a later version of the crate seeds its source
/// neighbourhood analytically, this goes red and the default should be revisited.
#[test]
fn the_two_solvers_are_not_equally_accurate() {
    let mut marching = genslip::rupture::FastMarching::new();
    let measured = point_source_error(&mut marching);
    let recorded = ACCURACY[1].1;
    assert!(
        (measured - recorded).abs() < 0.01,
        "fast marching now measures {measured:.5} against a recorded {recorded:.5}; \
         if it improved, revisit which solver is the default"
    );

    // And the physical consequence, which is what actually decides it.
    assert!(
        measured > 4.0 * ACCURACY[0].1,
        "the gap closed; fast marching is no longer materially worse"
    );
}

contract_for!(fast_marching, genslip::rupture::FastMarching::new());

#[cfg(feature = "wavefront-compat")]
contract_for!(wavefront2d, genslip::rupture::Wavefront2d::new());
