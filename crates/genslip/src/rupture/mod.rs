//! When each subfault starts to slip.
//!
//! Rupture spreads outward from the hypocentre at a speed that varies across the
//! fault — slower near the surface, slower at depth, faster where slip is large. The
//! arrival time at every subfault is the solution of the eikonal equation
//!
//! ```text
//!   |grad T| = 1 / v
//! ```
//!
//! with `T = 0` at the hypocentre. That is a first-arrival problem, and how it is
//! solved is an implementation detail of the solver, not of the physics.
//!
//! # The solver is swappable
//!
//! [`EikonalSolver`] is the trait, for the same reason [`crate::rng::DrawSource`] and
//! [`crate::fft::Fft`] are traits: the choice is a deployment decision, not a
//! property of the model.
//!
//! `Wavefront2d` is genslip's — an expanding-square wavefront tracker (Afnimar &
//! Koketsu, 2000), reached through the original Fortran. It exists to be compared
//! against, and it is the Stage 1 default.
//!
//! # Its edge requirement is the solver's problem, not the caller's
//!
//! `wfront2d` computes an analytic solution within a few cells of the source and
//! switches to finite differences outside, so the source must sit at least
//! `ring_radius + 1` cells from every edge. genslip meets that by growing the grid
//! and offsetting the source — code smeared across `main` between the rupture-speed
//! calculation and the solve.
//!
//! Here it lives inside `Wavefront2d`, which pads on the way in and crops on the
//! way out. A fast-marching solver has no such requirement and would simply not do
//! it. Putting the padding in the caller would have made that swap a rewrite rather
//! than a line.
//!
//! (orig. `genslip_v5.6.2.c:2995-3045`, `ruptime.c:get_rslow_stretch`, `wafront2d.f`)

#[cfg(feature = "wavefront-compat")]
mod wavefront;

#[cfg(feature = "wavefront-compat")]
pub use wavefront::Wavefront2d;

/// Rupture speed across the fault, in km/s.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeedGrid {
    strike_count: usize,
    dip_count: usize,
    values: Vec<f32>,
}

impl SpeedGrid {
    /// Build from speeds in km/s, along-strike index fastest.
    ///
    /// # Panics
    ///
    /// If `values` does not hold one speed per subfault, or if any speed is not
    /// strictly positive — the solver divides by it.
    #[must_use]
    pub fn new(strike_count: usize, dip_count: usize, values: Vec<f32>) -> Self {
        assert_eq!(
            values.len(),
            strike_count * dip_count,
            "got {} speeds for a {strike_count}x{dip_count} fault",
            values.len()
        );
        assert!(
            values.iter().all(|speed| *speed > 0.0),
            "rupture speeds must be strictly positive; the solver inverts them"
        );
        Self {
            strike_count,
            dip_count,
            values,
        }
    }

    /// Number of subfaults along strike.
    #[must_use]
    pub const fn strike_count(&self) -> usize {
        self.strike_count
    }

    /// Number of subfaults down dip.
    #[must_use]
    pub const fn dip_count(&self) -> usize {
        self.dip_count
    }

    /// Speed at `(strike, dip)`, in km/s.
    #[must_use]
    pub fn speed(&self, strike: usize, dip: usize) -> f32 {
        self.values[strike + dip * self.strike_count]
    }
}

/// Rupture onset time at every subfault, in seconds from the hypocentre.
///
/// Double precision throughout: the solver accumulates arrival times across the
/// whole fault, and genslip's is `real*8` for that reason.
#[derive(Clone, Debug, PartialEq)]
pub struct TravelTimes {
    strike_count: usize,
    dip_count: usize,
    values: Vec<f64>,
}

impl TravelTimes {
    /// Build from times in seconds, along-strike index fastest.
    ///
    /// # Panics
    ///
    /// If `values` does not hold one time per subfault.
    #[must_use]
    pub fn new(strike_count: usize, dip_count: usize, values: Vec<f64>) -> Self {
        assert_eq!(
            values.len(),
            strike_count * dip_count,
            "got {} times for a {strike_count}x{dip_count} fault",
            values.len()
        );
        Self {
            strike_count,
            dip_count,
            values,
        }
    }

    /// Number of subfaults along strike.
    #[must_use]
    pub const fn strike_count(&self) -> usize {
        self.strike_count
    }

    /// Number of subfaults down dip.
    #[must_use]
    pub const fn dip_count(&self) -> usize {
        self.dip_count
    }

    /// Onset time at `(strike, dip)`, in seconds.
    #[must_use]
    pub fn time(&self, strike: usize, dip: usize) -> f64 {
        self.values[strike + dip * self.strike_count]
    }

    /// All times, along-strike index fastest.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }
}

/// Where the rupture starts, as a subfault index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Hypocentre {
    pub strike: usize,
    pub dip: usize,
}

/// Solves for first-arrival times on a uniform grid.
///
/// Implementations must agree on the physics — `T = 0` at the hypocentre, times
/// increasing outward, and the arrival at each subfault being the *first* one — but
/// are free to differ in method and therefore in the last bits.
pub trait EikonalSolver {
    /// First-arrival time at every subfault.
    ///
    /// `spacing_km` is the grid step, the same in both directions. genslip warns
    /// rather than refuses when the two subfault dimensions differ, and then uses
    /// the along-strike one for both; a caller wanting square cells must arrange
    /// them.
    fn solve(&mut self, speed: &SpeedGrid, hypocentre: Hypocentre, spacing_km: f64) -> TravelTimes;
}
