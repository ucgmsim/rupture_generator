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

use crate::rise_time::DepthRamp;
use crate::taper::SlipField;

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

/// How rupture speed varies with depth, as a fraction of the shear-wave speed.
///
/// Rupture is slower near the surface, where the rock is weaker and the free
/// surface unloads it, and slower again at depth as the transition to stable
/// sliding begins. Between the two it runs at the full configured fraction.
#[derive(Clone, Copy, Debug)]
pub struct SpeedProfile {
    /// Ramp from `shallow_factor` up to 1 with increasing depth.
    pub shallow: DepthRamp,
    pub shallow_factor: f32,
    /// Ramp from 1 down to `deep_factor` with increasing depth.
    pub deep: DepthRamp,
    pub deep_factor: f32,
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
    fn depth_factor(self, depth_km: f32) -> f32 {
        let shallow_min = self.shallow.centre_km - self.shallow.half_width_km;
        let shallow_max = self.shallow.centre_km + self.shallow.half_width_km;
        let deep_min = self.deep.centre_km - self.deep.half_width_km;
        let deep_max = self.deep.centre_km + self.deep.half_width_km;

        // SIMPLIFY: `DepthRamp::scaled_from_deep` and `scaled_from_shallow`, which
        // every other ramp in the program now uses. Not applicable here, because the
        // scale factor `1.0 - shal_vr` is a *double* — the literal makes it so — and
        // the helper's is single. The multiply therefore happens at a different
        // width, and the result differs in the last bit.
        //
        // Two things are being reproduced at once. The *form*: `(max - depth)/(max -
        // min)` rather than `1 - (depth - min)/(max - min)`. And the *precision*: the
        // two differences are single, everything above them is double, and it rounds
        // once at the store.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the narrowing seam: C stores rfdep into a float"
        )]
        let factor = if depth_km <= shallow_min {
            self.shallow_factor
        } else if depth_km < shallow_max {
            (1.0 - (1.0 - f64::from(self.shallow_factor)) * f64::from(shallow_max - depth_km)
                / f64::from(shallow_max - shallow_min)) as f32
        } else if depth_km <= deep_min {
            1.0
        } else if depth_km < deep_max {
            (1.0 - (1.0 - f64::from(self.deep_factor)) * f64::from(depth_km - deep_min)
                / f64::from(deep_max - deep_min)) as f32
        } else {
            self.deep_factor
        };
        factor
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
    depth_km: &[f32],
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

    let mut speeds = Vec::with_capacity(strike_count * dip_count);
    for dip in 0..dip_count {
        let depth_factor = profile.depth_factor(depth_km[dip]);
        for strike in 0..strike_count {
            speeds.push(
                velocity_fraction[(strike, dip)] * depth_factor * shear_speed_km_s[(strike, dip)],
            );
        }
    }

    SpeedGrid::new(strike_count, dip_count, speeds)
}

/// Where the rupture front is when it reaches each subfault, after perturbation.
#[derive(Clone, Copy, Debug)]
pub struct OnsetAdjustment {
    /// Amplitude of the slip-correlated timing perturbation, in seconds.
    ///
    /// genslip's `tsfac_main`, resolved from the moment as
    /// `tsfac_bzero + tsfac_slope * 1e-9 * Mo^(1/3)`. Negative, so patches that slip
    /// more rupture *earlier*.
    pub perturbation_scale: f32,
    /// Constant offset added to every subfault, in seconds.
    pub delay_s: f32,
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

    let mut onset = Vec::with_capacity(strike_count * dip_count);
    let mut earliest = f64::INFINITY;
    for dip in 0..dip_count {
        for strike in 0..strike_count {
            // The perturbation is applied in single precision, as the original
            // stores it into `psrc[].rupt`, a float.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the narrowing seam: C stores the sum into a float"
            )]
            let narrowed = travel.time(strike, dip) as f32;
            let time =
                f64::from(narrowed + adjustment.perturbation_scale * perturbation[(strike, dip)]);
            earliest = earliest.min(time);
            onset.push(time);
        }
    }

    // A segment without the hypocentre keeps its absolute times; only the one that
    // has it is re-zeroed.
    let shift = if adjustment.contains_hypocentre {
        earliest
    } else {
        0.0
    };

    for time in &mut onset {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the narrowing seam: C stores both into floats"
        )]
        let (narrowed, narrowed_shift) = (*time as f32, shift as f32);
        *time = f64::from(narrowed - narrowed_shift + adjustment.delay_s);
    }

    TravelTimes::new(strike_count, dip_count, onset)
}
