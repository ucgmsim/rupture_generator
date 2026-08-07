//! genslip's solver: an expanding-square wavefront tracker.
//!
//! Afnimar & Koketsu (2000). The solution is analytic within `ring_radius` cells of
//! the source and finite-differenced outside, expanding one square ring at a time in
//! four directional sweeps.
//!
//! **Stage 1 only.** Reached through the original Fortran, so the port has something
//! exact to compare against. A fast-marching solver is the destination; nothing
//! outside this file depends on which is in use.
//!
//! # The padding, which is this solver's problem alone
//!
//! The analytic near-source solution needs `ring_radius + 1` cells of room in every
//! direction, so a hypocentre near a fault edge does not fit. genslip grows the grid
//! and offsets the source to make room, filling the new cells by replicating the
//! nearest real value.
//!
//! That happens here rather than in the caller, so swapping in a solver without the
//! requirement removes the padding along with it.
//!
//! # A defect in the replication, reproduced
//!
//! The edge replication overwrites real data whenever it pads the *low* side. See
//! [`Wavefront2d::pad_slowness`] — it is reachable, and it is pinned by a test rather
//! than left to be rediscovered.

use crate::rupture::{EikonalSolver, Hypocentre, SpeedGrid, TravelTimes};

/// `nsring` in the original, fixed at 2 there.
const DEFAULT_RING_RADIUS: usize = 2;

unsafe extern "C" {
    /// `subroutine wfront2d(m, n, is, js, h, ns, ttime, slwns, ntot, ti, jm)`
    ///
    /// All arguments by reference, Fortran-style. `is`/`js` are **1-based**.
    /// `ti` and `jm` are caller-allocated scratch of length `m + n`.
    #[link_name = "wfront2d_"]
    fn wfront2d(
        m: *const core::ffi::c_int,
        n: *const core::ffi::c_int,
        is: *const core::ffi::c_int,
        js: *const core::ffi::c_int,
        h: *const core::ffi::c_double,
        ns: *const core::ffi::c_int,
        ttime: *mut core::ffi::c_double,
        slwns: *mut core::ffi::c_double,
        ntot: *const core::ffi::c_int,
        ti: *mut core::ffi::c_double,
        jm: *mut core::ffi::c_int,
    );
}

/// How a grid was grown to give the source room, and where the source ended up.
#[derive(Clone, Copy, Debug)]
struct Padding {
    /// Cells added before the fault's first index.
    offset: usize,
    /// Total extent after padding.
    extent: usize,
    /// Source index within the padded grid, 0-based.
    source: usize,
}

impl Padding {
    /// Grow `extent` so `source` sits at least `radius + 1` from either edge.
    ///
    /// The two branches are not symmetric, and that is the original's shape: a
    /// source too close to the *low* edge is moved by inserting cells before it,
    /// while one too close to the *high* edge is accommodated by extending past it
    /// and leaving the source where it is.
    fn for_source(source: usize, extent: usize, radius: usize) -> Self {
        if source < radius + 1 {
            let offset = radius + 1 - source;
            Self {
                offset,
                extent: extent + offset,
                source: radius + 1,
            }
        } else if source + radius + 2 > extent {
            Self {
                offset: 0,
                extent: source + radius + 2,
                source,
            }
        } else {
            Self {
                offset: 0,
                extent,
                source,
            }
        }
    }
}

/// genslip's wavefront solver.
#[derive(Clone, Copy, Debug)]
pub struct Wavefront2d {
    ring_radius: usize,
}

impl Wavefront2d {
    /// The solver as genslip configures it, with a ring radius of 2.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ring_radius: DEFAULT_RING_RADIUS,
        }
    }

    /// Slowness on the padded grid, with edge values replicated into the new cells.
    ///
    /// # The defect
    ///
    /// The two dip-direction passes copy from rows `dip_offset` and `dip_count - 1`
    /// of the *padded* grid. The first is right — it is the fault's first row. The
    /// second is wrong whenever `dip_offset > 0`: the fault's last row is at
    /// `dip_offset + dip_count - 1`, so row `dip_count - 1` is an interior row, and
    /// the loop that writes rows `dip_count..padded` is writing over real data.
    ///
    /// On a 10-deep fault padded by 3, the deepest three rows of slowness are
    /// replaced by the values from six rows shallower. The along-strike passes have
    /// the same defect.
    ///
    /// Reproduced rather than fixed, per `PORTING_RULES.md` §5. It is reachable: the
    /// low-side branch triggers when the hypocentre is within three subfaults of an
    /// edge, and the along-strike hypocentre distribution is tapered rather than
    /// truncated.
    fn pad_slowness(speed: &SpeedGrid, strike: Padding, dip: Padding) -> Vec<f64> {
        let mut slowness = vec![0.0_f64; strike.extent * dip.extent];
        let inverse = |value: f32| f64::from(1.0 / value);

        // The fault itself, at its offset position.
        for row in 0..speed.dip_count() {
            for column in 0..speed.strike_count() {
                slowness[(column + strike.offset) + (row + dip.offset) * strike.extent] =
                    inverse(speed.speed(column, row));
            }
        }

        // Low-side strike padding, from the fault's first column.
        for row in 0..speed.dip_count() {
            for column in 0..strike.offset {
                slowness[column + (row + dip.offset) * strike.extent] =
                    inverse(speed.speed(0, row));
            }
        }

        // High-side strike padding, from the fault's last column -- but indexed from
        // `strike_count` rather than from `strike_count + offset`. See the defect.
        for row in 0..speed.dip_count() {
            for column in speed.strike_count()..strike.extent {
                slowness[column + (row + dip.offset) * strike.extent] =
                    inverse(speed.speed(speed.strike_count() - 1, row));
            }
        }

        // Low-side dip padding, from the fault's first row.
        for row in 0..dip.offset {
            for column in 0..strike.extent {
                slowness[column + row * strike.extent] =
                    slowness[column + dip.offset * strike.extent];
            }
        }

        // High-side dip padding, with the same off-by-offset as the strike case.
        for row in speed.dip_count()..dip.extent {
            for column in 0..strike.extent {
                slowness[column + row * strike.extent] =
                    slowness[column + (speed.dip_count() - 1) * strike.extent];
            }
        }

        slowness
    }
}

impl Default for Wavefront2d {
    fn default() -> Self {
        Self::new()
    }
}

impl EikonalSolver for Wavefront2d {
    fn solve(&mut self, speed: &SpeedGrid, hypocentre: Hypocentre, spacing_km: f64) -> TravelTimes {
        assert!(
            hypocentre.strike < speed.strike_count() && hypocentre.dip < speed.dip_count(),
            "hypocentre ({}, {}) is outside a {}x{} fault",
            hypocentre.strike,
            hypocentre.dip,
            speed.strike_count(),
            speed.dip_count()
        );

        let strike = Padding::for_source(hypocentre.strike, speed.strike_count(), self.ring_radius);
        let dip = Padding::for_source(hypocentre.dip, speed.dip_count(), self.ring_radius);

        let mut slowness = Self::pad_slowness(speed, strike, dip);
        let points = strike.extent * dip.extent;
        let mut times = vec![0.0_f64; points];

        // The Fortran allocates neither; both are the caller's, of length m + n.
        let mut time_scratch = vec![0.0_f64; strike.extent + dip.extent];
        let mut index_scratch = vec![0_i32; strike.extent + dip.extent];

        let extent_strike = i32::try_from(strike.extent).expect("grid extent must fit in a C int");
        let extent_dip = i32::try_from(dip.extent).expect("grid extent must fit in a C int");
        let total = i32::try_from(points).expect("grid size must fit in a C int");
        let radius = i32::try_from(self.ring_radius).expect("ring radius is small");

        // Fortran indexes from 1.
        let source_strike =
            i32::try_from(strike.source + 1).expect("source index must fit in a C int");
        let source_dip = i32::try_from(dip.source + 1).expect("source index must fit in a C int");

        // SAFETY: `slowness` and `times` both hold exactly `points` elements, which
        // is what `ntot` tells the routine to expect; both scratch buffers are
        // `m + n` long as its header requires. It zeroes `times` itself.
        unsafe {
            wfront2d(
                &raw const extent_strike,
                &raw const extent_dip,
                &raw const source_strike,
                &raw const source_dip,
                &raw const spacing_km,
                &raw const radius,
                times.as_mut_ptr(),
                slowness.as_mut_ptr(),
                &raw const total,
                time_scratch.as_mut_ptr(),
                index_scratch.as_mut_ptr(),
            );
        }

        // Crop back to the fault.
        let mut cropped = Vec::with_capacity(speed.strike_count() * speed.dip_count());
        for row in 0..speed.dip_count() {
            for column in 0..speed.strike_count() {
                cropped.push(times[(column + strike.offset) + (row + dip.offset) * strike.extent]);
            }
        }

        TravelTimes::new(speed.strike_count(), speed.dip_count(), cropped)
    }
}
