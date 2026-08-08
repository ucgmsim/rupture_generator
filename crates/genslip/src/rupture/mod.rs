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
//! [`crate::fft::Fft`] are traits: the choice is a deployment decision, not a property
//! of the model.
//!
//! [`FactoredSweep`] is the only implementation, and the third to hold that position.
//! genslip's own — an expanding-square wavefront tracker (Afnimar & Koketsu, 2000)
//! reached through the original Fortran — and the `eikonal` crate's fast marching both
//! preceded it and both were removed on measurement: neither converges on a
//! heterogeneous medium. `DEFECTS.md` 19 has the numbers, and
//! `tests/eikonal_contract.rs` is the bar a fourth would have to clear.
//!
//! # There is no padding, and that is the point
//!
//! The tracker computed an analytic solution within a few cells of the source and
//! finite-differenced outside, so the source had to sit clear of every edge. genslip
//! meets that by growing the grid and replicating edge values into it — and the
//! replication overwrites real data whenever it pads the low side, which was
//! `DEFECTS.md` 1. The port reproduced all of it.
//!
//! Factored sweeping has no near-source region: the singularity is removed
//! analytically instead, so every cell is computed by the same update. The padding,
//! the replication and the defect went together, exactly as this module's earlier
//! text predicted they would.
//!
//! (orig. `genslip_v5.6.2.c:2995-3045`, `ruptime.c:get_rslow_stretch`, `wafront2d.f`)

mod sweeping;
pub use sweeping::FactoredSweep;

use ndarray::Array2;

use crate::grid::{FaultAxes, FaultAxesMut, SlipField};
use crate::rise_time::DepthRamp;
use ndarray::{Array1, Axis};

/// Rupture speed across the fault, in km/s.
///
/// A [`SlipField`] by another name — same shape, same layout, different meaning. The
/// invariant that used to live in a constructor (every speed strictly positive,
/// because the solver inverts it) is now the solver's own
/// [`Error::Unreachable`](crate::Error::Unreachable): a zero speed makes an arrival
/// infinite rather than large, which is what that variant is for and where a caller
/// can act on it.
pub type SpeedGrid = SlipField;

/// Rupture onset time at every subfault, in seconds from the hypocentre.
///
/// Double precision, where the fields around it are single: the solver accumulates
/// arrival times across the whole fault and genslip's is `real*8` for that reason.
/// Stage 2b makes the distinction moot.
pub type TravelTimes = Array2<f64>;

/// Move every onset by a per-subfault amount, never below zero.
///
/// One caller: [`crate::slip_rate::SlipRateShape::Seki`] radiates before `t = 0`, so
/// the arrival is moved back to compensate. The clamp is the original's
/// (`generic_slip2srf.c:454`) and is the right shape anyway — a subfault cannot
/// rupture before the earthquake starts, and the hypocentre is already at zero.
///
/// A free function rather than a method, because [`TravelTimes`] is an
/// [`ndarray::Array2`] now and this is not something an array does.
///
/// # Panics
///
/// If the two grids do not cover the same fault.
pub fn shift_onsets(onset_s: &mut TravelTimes, shift_s: &SlipField) {
    assert_eq!(
        onset_s.extent(),
        shift_s.extent(),
        "the onset and shift grids must cover the same fault"
    );
    for (time, shift) in onset_s.flat_mut().iter_mut().zip(shift_s.flat()) {
        *time = (*time + *shift).max(0.0);
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

/// How rupture speed varies with depth, as a fraction of the shear-wave speed.
///
/// Rupture is slower near the surface, where the rock is weaker and the free
/// surface unloads it, and slower again at depth as the transition to stable
/// sliding begins. Between the two it runs at the full configured fraction.
#[derive(Clone, Copy, Debug)]
pub struct SpeedProfile {
    /// Ramp from `shallow_factor` up to 1 with increasing depth.
    pub shallow: DepthRamp,
    pub shallow_factor: f64,
    /// Ramp from 1 down to `deep_factor` with increasing depth.
    pub deep: DepthRamp,
    pub deep_factor: f64,
}

impl SpeedProfile {
    /// The depth factor at `depth_km`.
    ///
    /// # This is not the ramp `rise_time` uses, and the difference looks like a bug
    ///
    /// [`crate::rise_time::rise_time_normalisation`] computes its own rupture-speed
    /// factor for the slip-and-rupture-speed weighting, and writes the shallow zone
    /// as `rvfrac * shal_vrup * (dmax - depth)/(dmax - dmin)` — which falls to
    /// **zero** at the bottom of the transition, where this correctly rises to
    /// **one**. A rupture speed of zero is not a physical statement about anything.
    ///
    /// It is latent: that factor is only consulted when the weighting is
    /// [`crate::rise_time::Weighting::BySlipAndRuptureSpeed`], and the configured
    /// weighting is uniform. Both are reproduced as they stand, separately, rather
    /// than unified onto whichever is right.
    fn depth_factor(self, depth_km: f64) -> f64 {
        // `DepthRamp::scaled_from_deep` and `scaled_from_shallow`, like every other
        // ramp in the program. The original ran this one in `double` throughout — the
        // literal `1.0 - shal_vr` makes it so, and only the two depth differences are
        // single — rounding once at the store, where the helper works in `f64`
        // start to finish. That extra rounding is why it stayed written out under
        // bit-parity.
        //
        // Measured before unifying, over the whole corpus: 87 subfaults across two of
        // the six cases moved, worst 9.5e-07 s. The onset bound is 0.05 s, so the
        // change is five orders of magnitude inside it, and it is the only thing in
        // this commit that moved anything at all. Slip, rake, rise time and every
        // slip-rate sample are bit-identical.
        if depth_km <= self.shallow.shallow_km() {
            self.shallow_factor
        } else if depth_km < self.shallow.deep_km() {
            1.0 - self
                .shallow
                .scaled_from_deep(1.0 - self.shallow_factor, depth_km)
        } else if depth_km <= self.deep.shallow_km() {
            1.0
        } else if depth_km < self.deep.deep_km() {
            1.0 - self
                .deep
                .scaled_from_shallow(1.0 - self.deep_factor, depth_km)
        } else {
            self.deep_factor
        }
    }
}

/// Build the rupture-speed field the solver runs on.
///
/// `shear_speed_km_s` is one value per subfault, `velocity_fraction` the configured
/// rupture-speed-to-shear-speed ratio at each — a single value repeated unless the
/// geometry supplied per-subfault ratios. `depth_km` gives one depth per dip row.
///
/// # Panics
///
/// If the inputs disagree about the fault's extent, or if any resulting speed is not
/// strictly positive.
///
/// (orig. `get_rspeed_vsden2`, ruptime.c:817)
#[must_use]
pub fn speed_field(
    shear_speed_km_s: &SlipField,
    velocity_fraction: &SlipField,
    depth_km: &[f64],
    profile: SpeedProfile,
) -> SpeedGrid {
    let strike_count = shear_speed_km_s.strike_count();
    let dip_count = shear_speed_km_s.dip_count();
    assert_eq!(
        (
            velocity_fraction.strike_count(),
            velocity_fraction.dip_count()
        ),
        (strike_count, dip_count),
        "the shear-speed and velocity-fraction fields must cover the same fault"
    );
    assert_eq!(depth_km.len(), dip_count, "one depth per dip row");

    // The depth factor is one value per dip row, broadcast down it.
    let depth_factor: Array1<f64> = depth_km
        .iter()
        .map(|depth| profile.depth_factor(*depth))
        .collect();

    velocity_fraction * &depth_factor.insert_axis(Axis(1)) * shear_speed_km_s
}

/// Where the rupture front is when it reaches each subfault, after perturbation.
#[derive(Clone, Copy, Debug)]
pub struct OnsetAdjustment {
    /// Amplitude of the slip-correlated timing perturbation, in seconds.
    ///
    /// genslip's `tsfac_main`, resolved from the moment as
    /// `tsfac_bzero + tsfac_slope * 1e-9 * Mo^(1/3)`. Negative, so patches that slip
    /// more rupture *earlier*.
    pub perturbation_scale: f64,
    /// Constant offset added to every subfault, in seconds.
    pub delay_s: f64,
    /// Whether this segment contains the hypocentre.
    ///
    /// Only the segment that does gets shifted to start at zero. The others keep
    /// their offsets, which is what makes a multi-segment rupture propagate between
    /// them rather than restarting.
    pub contains_hypocentre: bool,
}

/// Combine solved travel times with the timing perturbation.
///
/// # Panics
///
/// If the travel times and the perturbation field disagree about the fault's extent.
///
/// (orig. `genslip_v5.6.2.c:3134-3160`)
#[must_use]
pub fn onset_times(
    travel: &TravelTimes,
    perturbation: &SlipField,
    adjustment: OnsetAdjustment,
) -> TravelTimes {
    let strike_count = travel.strike_count();
    let dip_count = travel.dip_count();
    assert_eq!(
        (perturbation.strike_count(), perturbation.dip_count()),
        (strike_count, dip_count),
        "the travel times and the perturbation must cover the same fault"
    );

    // The perturbation is applied where the original stores `psrc[].rupt`.
    let mut onset = travel + adjustment.perturbation_scale * perturbation;

    // A segment without the hypocentre keeps its absolute times; only the one that
    // has it is re-zeroed.
    let shift = if adjustment.contains_hypocentre {
        onset.fold(f64::INFINITY, |earliest, time| earliest.min(*time))
    } else {
        0.0
    };

    onset += adjustment.delay_s - shift;
    onset
}
