//! The slip-rate function: how fast each subfault slips, moment by moment.
//!
//! Everything before this produces *how much* each subfault slips and *when* it
//! starts. This produces the shape in between — a time series per subfault, scaled so
//! its integral is the subfault's slip. That series is what an SRF file carries and
//! what a wave-propagation code convolves with.
//!
//! The shape is not a free choice. It has to rise sharply (rupture arrives as a
//! stress step), peak early, and decay slowly, because that is what dynamic rupture
//! models and kinematic inversions both produce. genslip's default is the
//! `OliuP` form — a piecewise sinusoid after Liu, Archuleta & Hartzell (2006), with a
//! `beta` parameter setting how much of the duration is spent in the rising limb.
//!
//! `OliuP2` is the configured shape. It differs from `OliuP` only in where `beta`
//! comes from -- the per-subfault array rather than a ramp recomputed in the loop --
//! so one generator serves both.
//!
//! (orig. `gslip_sliprate_subs.c` and `load_slip_srf_dd5_vsden`)

use crate::rise_time::{DepthRamp, RiseTimeStretch};
use crate::taper::SlipField;

/// Shape parameter of the slip-rate function, varying with depth.
///
/// Larger `beta` means a longer rising limb and so a smoother, less impulsive pulse.
/// Shallow subfaults get the largest value: near-surface rupture is slower and less
/// abrupt than rupture at depth.
#[derive(Clone, Copy, Debug)]
pub struct BetaProfile {
    /// Ramp from `shallow` to `mid`.
    pub shallow_ramp: DepthRamp,
    pub shallow: f32,
    /// Ramp from `mid` to `deep`.
    pub mid_ramp: DepthRamp,
    pub mid: f32,
    pub deep: f32,
}

impl BetaProfile {
    /// Shape parameter at `depth_km`.
    ///
    /// SIMPLIFY: `DepthRamp::scaled_from_shallow`. This is the one ramp in the
    /// program whose grouping differs: it precomputes a gradient,
    /// `(v_far - v_near) / width`, and multiplies by the offset — `(a/c)*b` where
    /// every other site writes `(a*b)/c`. Same number, different `f32`, so the
    /// helper cannot be used here without moving every beta value.
    #[must_use]
    pub fn beta_at(self, depth_km: f32) -> f32 {
        let shallow_min = self.shallow_ramp.centre_km - self.shallow_ramp.half_width_km;
        let shallow_max = self.shallow_ramp.centre_km + self.shallow_ramp.half_width_km;
        let mid_min = self.mid_ramp.centre_km - self.mid_ramp.half_width_km;
        let mid_max = self.mid_ramp.centre_km + self.mid_ramp.half_width_km;

        let shallow_gradient = (self.mid - self.shallow) / (shallow_max - shallow_min);
        let mid_gradient = (self.deep - self.mid) / (mid_max - mid_min);

        if depth_km <= shallow_min {
            self.shallow
        } else if depth_km < shallow_max {
            self.shallow + shallow_gradient * (depth_km - shallow_min)
        } else if depth_km <= mid_min {
            self.mid
        } else if depth_km < mid_max {
            self.mid + mid_gradient * (depth_km - mid_min)
        } else {
            self.deep
        }
    }
}

/// Shape parameter for every subfault.
///
/// # Panics
///
/// If `depth_km` does not hold one depth per dip row.
///
/// (orig. `genslip_v5.6.2.c:2884-2904`)
#[must_use]
pub fn beta_field(strike_count: usize, depth_km: &[f32], profile: BetaProfile) -> SlipField {
    let dip_count = depth_km.len();
    let mut field = SlipField::zeros(strike_count, dip_count);
    for dip in 0..dip_count {
        let beta = profile.beta_at(depth_km[dip]);
        for strike in 0..strike_count {
            field[(strike, dip)] = beta;
        }
    }
    field
}

/// Rise time at every subfault, in seconds.
///
/// `normalised` is the unit-mean field from [`crate::rise_time::rise_time_field`] and
/// `normalisation` the constant from [`crate::rise_time::rise_time_normalisation`];
/// together they turn `average_s` into a per-subfault duration.
///
/// A subfault whose normalised rise time is zero — one whose slip was truncated
/// away — gets the floor rather than zero, because a slip-rate function needs at
/// least one sample to exist at all.
///
/// # Panics
///
/// If the inputs disagree about the fault's extent, or if `sample_interval_s` or
/// `average_s` is not strictly positive.
///
/// (orig. `gslip_srf_subs.c:1497-1526`)
#[must_use]
pub fn rise_times(
    normalised: &SlipField,
    depth_km: &[f32],
    stretch: RiseTimeStretch,
    average_s: f32,
    normalisation: f32,
    sample_interval_s: f32,
) -> SlipField {
    assert_eq!(
        depth_km.len(),
        normalised.dip_count(),
        "one depth per dip row"
    );
    assert!(
        sample_interval_s > 0.0 && average_s > 0.0,
        "the sample interval and average rise time must be positive"
    );

    // The floor is one sample: a rise time below `dt` cannot be represented.
    let floor = sample_interval_s / average_s;

    let strike_count = normalised.strike_count();
    let mut field = SlipField::zeros(strike_count, normalised.dip_count());
    for dip in 0..normalised.dip_count() {
        let depth_factor = stretch.factor_at(depth_km[dip]);
        for strike in 0..strike_count {
            let scaled = normalised[(strike, dip)];
            let mut factor = if normalisation > 0.0 && scaled > 0.0 {
                depth_factor * scaled / normalisation
            } else if scaled <= 0.0 {
                0.0
            } else {
                depth_factor
            };
            if factor < floor {
                factor = floor;
            }
            field[(strike, dip)] = factor * average_s;
        }
    }
    field
}

/// The original's pi, and not `f32::consts::PI`.
///
/// It is written `3.141592654` and held in a `float`, so it is the nearest `f32` to
/// a ten-digit decimal rather than to pi itself. That value sets the phase of every
/// sinusoid in the pulse, so substituting the true constant moves every sample.
///
/// SIMPLIFY: `std::f32::consts::PI`. Unlike the `4.0*atan(1.0)` in `shift_phase`,
/// this one is **not** free -- the two differ in the last bit.
#[expect(
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::unreadable_literal,
    reason = "reproducing the original's literal, which is not pi; see the doc comment"
)]
const PI: f32 = 3.141592654;

/// A slip-rate function: samples at a fixed interval, integrating to the slip.
#[derive(Clone, Debug, PartialEq)]
pub struct SlipRate {
    values: Vec<f32>,
}

/// The slip below which the original emits no slip-rate function at all.
///
/// `MINSLIP` (`defs.h:15`), in centimetres. The guard is `sabs > MINSLIP` on
/// `|slip|` in the SRF loader (`gslip_srf_subs.c:1496`), *outside* the pulse
/// generator — so a subfault that does not slip gets `nt1 = 0` and a null `stf1`
/// rather than a short pulse of nothing.
pub const MIN_SLIP_CM: f32 = 1.0e-02;

impl SlipRate {
    /// A subfault that does not slip: no samples at all.
    ///
    /// Not the same as a pulse whose samples happen to be zero. The format stores
    /// `nt1 = 0` and no samples, and anything counting rows sees the difference.
    #[must_use]
    pub const fn empty() -> Self {
        Self { values: Vec::new() }
    }

    /// The samples, in cm/s if slip was in cm.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Number of samples. Zero means the subfault does not slip.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the subfault slips at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// The `OliuP` slip-rate function: a piecewise sinusoid after Liu, Archuleta &
/// Hartzell (2006).
///
/// Three pieces, of which `beta` sets the first two's extent. Writing `tau1` for
/// `beta * duration`:
///
/// * `0 .. tau1` — the rising limb, a raised cosine plus a half-period sine that
///   makes the rise sharper than the fall;
/// * `tau1 .. 2*tau1` — the peak and the start of the decay;
/// * `2*tau1 .. duration` — the tail, a quarter cosine to zero.
///
/// The result is normalised so `dt * sum` is `slip`, which is what makes the moment
/// come out right regardless of the shape.
///
/// # Degenerate cases, both reproduced
///
/// A duration under half a sample gives **no** samples and the subfault contributes
/// nothing. A duration of about one sample gives a fixed three-point spike rather
/// than anything computed — the shape is meaningless at that resolution, so the
/// original substitutes a triangle and moves on.
///
/// # Panics
///
/// If `sample_interval_s` is not strictly positive.
///
/// (orig. `gen_OliuP_stf`, `gslip_sliprate_subs.c`)
#[must_use]
pub fn oliu_p(
    slip: f32,
    duration_s: f32,
    beta: f32,
    sample_interval_s: f32,
    max_samples: usize,
) -> SlipRate {
    assert!(
        sample_interval_s > 0.0,
        "the sample interval must be positive"
    );

    let rise_end = beta * duration_s;
    let peak_end = 2.0 * rise_end;
    let decay_span = duration_s - rise_end;

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "C truncates toward zero; the value is a small non-negative count"
    )]
    let mut count = ((duration_s / sample_interval_s + 0.5) as i32).max(0) as usize;
    count = count.min(max_samples);

    if count == 0 {
        return SlipRate { values: Vec::new() };
    }

    let mut values;
    if count == 1 {
        // Too short to resolve. A fixed spike, not a computed shape.
        values = vec![0.0, 1.0, 0.0];
        values.truncate(max_samples.max(1));
    } else {
        values = vec![0.0_f32; count];
        for (index, value) in values.iter_mut().enumerate().skip(1) {
            #[expect(
                clippy::cast_precision_loss,
                reason = "sample counts are far below 2^24"
            )]
            let time = index as f32 * sample_interval_s;

            // Each `alpha` is a double expression stored into a float, as the
            // original stores it -- `cos` and `sin` take and return double.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the narrowing seam: C stores alpha into a float"
            )]
            let amplitude = if time < rise_end {
                let arg = PI * time / rise_end;
                (0.7 - 0.7 * f64::from(arg).cos() + 0.6 * f64::from(0.5 * arg).sin()) as f32
            } else if time < peak_end {
                let rising = PI * time / rise_end;
                let decaying = PI * (time - rise_end) / decay_span;
                (1.0 - 0.7 * f64::from(rising).cos() + 0.3 * f64::from(decaying).cos()) as f32
            } else if time < duration_s {
                let decaying = PI * (time - rise_end) / decay_span;
                (0.3 + 0.3 * f64::from(decaying).cos()) as f32
            } else {
                0.0
            };

            *value = amplitude;
        }

        // One more sample, forced to zero, so the pulse closes.
        let closed = (count + 1).min(max_samples);
        values.resize(closed, 0.0);
        if let Some(last) = values.last_mut() {
            *last = 0.0;
        }
    }

    // Normalise so the integral is the slip. A non-positive integral means the shape
    // degenerated, and the subfault contributes nothing.
    let mut integral = 0.0_f32;
    for value in &values {
        integral += sample_interval_s * *value;
    }
    if integral <= 0.0 {
        return SlipRate { values: Vec::new() };
    }

    let scale = slip / integral;
    for value in &mut values {
        *value *= scale;
    }

    SlipRate { values }
}
