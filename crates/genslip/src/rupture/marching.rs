//! Fast marching, through the `eikonal` crate. **Not the default, and here is why.**
//!
//! This was meant to replace `Wavefront2d` — genslip's expanding-square tracker,
//! reached through the original Fortran — and with it the last reason this crate needs
//! an `EMOD3D` build. It satisfies every contract in `tests/eikonal_contract.rs`. It
//! is also **fourteen times less accurate**, and that decided it:
//!
//! | | worst relative | worst absolute |
//! | --- | --- | --- |
//! | `Wavefront2d` | 0.0088 | 0.011 s |
//! | `FastMarching` | 0.207 | **0.234 s** |
//!
//! Neither is defective. Fast marching computes every cell with the same upwind
//! update; the expanding-square tracker computes an *analytic* solution within a few
//! cells of the source and finite-differences outside, and that is worth the factor.
//! The crate's 0.106 beyond five cells is the published figure for first-order Godunov
//! fast marching, so it is doing exactly what it claims.
//!
//! What rules it out is the physics rather than the ranking: **0.234 s of
//! discretisation error is 4.7 times `ENGINEERING_RULES.md`'s 0.05 s onset bound.**
//! A replacement has to be better by that standard, and a solver whose own error
//! exceeds what counts as the same rupture cannot be.
//!
//! It stays, unwired, because the measurement is worth more than the code: the next
//! attempt starts from a number rather than from scratch. What would make it viable is
//! seeding the source's neighbours analytically — precisely genslip's `nsring` trick —
//! which the crate's API does not expose. `the_two_solvers_are_not_equally_accurate`
//! goes red if a later version closes the gap.
//!
//! # What it would have bought: the padding disappears with the solver
//!
//! `wfront2d` computes an analytic solution within a few cells of the source and
//! finite-differences outside, so the source must sit clear of every edge. genslip
//! meets that by growing the grid and replicating edge values into it — and the
//! replication overwrites real data whenever it pads the low side (`DEFECTS.md` 1).
//!
//! Fast marching has no near-source region: every cell including the source's
//! neighbours is computed by the same upwind update. So there is no padding, no edge
//! replication and no defect to reproduce. [`crate::rupture::Wavefront2d`] said this
//! swap would remove all three together, and it does — for whatever solver eventually
//! replaces the tracker.
//!
//! # Spacing folds into the cost field
//!
//! The crate solves `|grad T| = C` on a **unit-spaced** grid. That is the same problem
//! as `|grad T| = 1/v` on a grid of step `h`, with `C = h/v` — the equation is
//! homogeneous of degree one in the step, so a length scale can live in either place.
//! Folding it into the cost is why no spacing argument is needed on the way in.
//!
//! # Its grid is transposed relative to this one
//!
//! `ndarray` is row-major and the crate indexes `(row, column)`; this crate is
//! strike-fastest and indexes `(strike, dip)`. So strike is the *column* and dip is
//! the *row*, and the source goes in as `(dip, strike)`. Getting that backwards
//! produces a perfectly plausible rupture on a square fault and a shape error on any
//! other, which is why `contracts.rs` has a non-square fast-axis test.

use eikonal::{CostField, solve};
use ndarray::Array2;

use crate::rupture::{EikonalSolver, Hypocentre, SpeedGrid, TravelTimes};

/// A first-order upwind fast-marching solver.
#[derive(Clone, Copy, Debug, Default)]
pub struct FastMarching;

impl FastMarching {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl EikonalSolver for FastMarching {
    /// # Panics
    ///
    /// If the hypocentre is outside the fault. `SpeedGrid` already refuses a
    /// non-positive speed, so the cost field cannot contain the zero the crate reads
    /// as an impassable cell — but that is asserted here too, because a silently
    /// impassable subfault would give an infinite arrival rather than a wrong one.
    fn solve(&mut self, speed: &SpeedGrid, hypocentre: Hypocentre, spacing_km: f64) -> TravelTimes {
        let (strike_count, dip_count) = (speed.strike_count(), speed.dip_count());
        assert!(
            hypocentre.strike < strike_count && hypocentre.dip < dip_count,
            "hypocentre ({}, {}) is outside a {strike_count}x{dip_count} fault",
            hypocentre.strike,
            hypocentre.dip
        );

        // Slowness times the grid step: see the module docstring. Row is dip, column
        // is strike.
        let mut cost = Array2::<f64>::zeros((dip_count, strike_count));
        for dip in 0..dip_count {
            for strike in 0..strike_count {
                let value = spacing_km / f64::from(speed.speed(strike, dip));
                assert!(
                    value.is_finite() && value > 0.0,
                    "subfault ({strike}, {dip}) has a cost of {value}, which the \
                     solver would read as impassable"
                );
                cost[[dip, strike]] = value;
            }
        }

        let field = CostField::from_array(cost).expect("every cost is finite and positive");
        let solved = solve(&field, (hypocentre.dip, hypocentre.strike))
            .expect("the hypocentre is inside the grid and not impassable");
        let distance = solved.distance();

        let mut times = Vec::with_capacity(strike_count * dip_count);
        for dip in 0..dip_count {
            for strike in 0..strike_count {
                let arrival = distance[[dip, strike]];
                assert!(
                    arrival.is_finite(),
                    "({strike}, {dip}) was never reached; the fault is connected and \
                     every subfault ruptures"
                );
                times.push(arrival);
            }
        }

        TravelTimes::new(strike_count, dip_count, times)
    }
}
