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
//! # One shape library, two programs
//!
//! [`SlipRateShape`] covers genslip's finite-fault `stype` and `generic_slip2srf`'s,
//! which are different vocabularies spelled the same way. That is not a compromise:
//! four of `generic_slip2srf`'s ten shapes turn out to be [`oliu_p`] with the
//! breakpoints moved, so the alternative was a second copy of the same sinusoid.
//! `slip_rate_contract.rs` asserts the identity sample by sample.
//!
//! (orig. `gslip_sliprate_subs.c` and `load_slip_srf_dd5_vsden`; the alias family is
//! `generic_slip2srf/slip.c`)

// The original writes `3.141592654` where this uses `PI`, and in `f32` the two are the
// same number: they first differ at the tenth digit, an `f32` carries about seven, and
// both round to `0x40490fdb`. The pulse generator's `pi` is a `float`, so the
// substitution is free — and the corpus agrees, no slip-rate sample moved.
//
// What makes it free is the *width*, not the digit count. The same literal at double
// width is 4.6e-10 away from pi and is a different `f64`; genslip's other truncated
// constant, `rperd = 0.017453293`, is likewise exact as an `f32` and wrong as an
// `f64`, which is where it did real damage. `float_identities.rs` asserts all four.
use std::f32::consts::PI;

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
    /// Uses [`DepthRamp::scaled_from_shallow`], like every other ramp in the program.
    ///
    /// The original writes this one differently: it precomputes a gradient
    /// `(v_far - v_near)/width` and multiplies by the offset, `(a/c)*b` where every
    /// other site writes `(a*b)/c`. Equal in exact arithmetic and not in `f32`, which
    /// is why the two spellings stayed separate under bit-parity.
    ///
    /// The size of the difference was measured before unifying, and it is zero here:
    /// the shipped shallow ramp is 2 km wide, so `/(deep - shallow)` is a division by
    /// a power of two and exact, and the mid ramp is flat — `beta_mid` and
    /// `beta_deep` are both 0.13. Sampled at 200 000 depths across each ramp, the two
    /// groupings agree on every bit. They stop agreeing as soon as the width is not a
    /// power of two: 15% of depths differ in the last bit at a width of 3 km, 20% at
    /// 2.2 km. So this is a correctness-neutral unification for the configuration and
    /// a last-bit one for any other, and the corpus confirms it — no slip-rate sample
    /// moved.
    #[must_use]
    pub fn beta_at(self, depth_km: f32) -> f32 {
        if depth_km <= self.shallow_ramp.shallow_km() {
            self.shallow
        } else if depth_km < self.shallow_ramp.deep_km() {
            self.shallow
                + self
                    .shallow_ramp
                    .scaled_from_shallow(self.mid - self.shallow, depth_km)
        } else if depth_km <= self.mid_ramp.shallow_km() {
            self.mid
        } else if depth_km < self.mid_ramp.deep_km() {
            self.mid
                + self
                    .mid_ramp
                    .scaled_from_shallow(self.deep - self.mid, depth_km)
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

/// Which slip-rate function a subfault gets.
///
/// One vocabulary covering both of genslip's programs. The finite-fault generator
/// offers one shape, [`OliuP2`](SlipRateShape::OliuP2); `generic_slip2srf` offers ten
/// under the name `stype`. Rather than a second pulse library for the point-source
/// path, they are one enum, because **four of `generic_slip2srf`'s ten are
/// [`oliu_p`] already** and only differ in what they pass it:
///
/// | `stype` | is | (`generic_slip2srf/slip.c`) |
/// | --- | --- | --- |
/// | `ucsb` | `oliu_p(slip, T, 0.13)` | `gen_ucsb_stf`, :114 |
/// | `ucsb-varT1` | `oliu_p(slip, T, tau1_ratio)` | `gen_ucsbvT_stf`, :170 |
/// | `ucsb2` | `oliu_p(slip, 2T, 0.065)` | `gen_ucsb2_stf`, :226 |
/// | `ucsb-T<b>` | `oliu_p(slip, bT, 0.13/b)` | `gen_ucsbT_stf`, :282 |
///
/// 220 lines of C, four times the same three-piece sinusoid with the breakpoints
/// moved. `ucsb2` writes `tau = 2*t0` and `tau1 = 0.5*0.13*tau` — the comment says
/// *"keep peak at same place"* — which is `beta = 0.065` on a doubled duration.
/// `ucsb-T` writes `tau = b*t0` and `tau1 = 0.13*tau/b`, which is `0.13*t0`, so
/// `beta = 0.13/b`.
///
/// # Two differences from the C, in every one of the four
///
/// Both come from [`oliu_p`] being genslip's `gen_OliuP_stf` rather than
/// `generic_slip2srf`'s, and both are deliberate:
///
/// * **One trailing zero.** `oliu_p` appends a forced-zero sample so the pulse
///   closes; the C's ucsb family stops at the last computed sample. The samples that
///   exist are bit-identical — a zero contributes nothing to the integral, so the
///   normalisation is unchanged — and a slip-rate function that returns to zero is
///   the better of the two. `an_alias_is_oliu_p_plus_a_closing_zero` asserts exactly
///   that, rather than an approximate agreement.
/// * **A one-sample pulse.** Below two samples `oliu_p` substitutes a fixed
///   three-point spike; the C computes `alpha(0) = 0`, finds a zero integral, and
///   emits nothing. Reproducing both would mean two functions again.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SlipRateShape {
    /// genslip's finite-fault default: [`oliu_p`] with `beta` from
    /// [`beta_field`], so the shape varies with depth.
    OliuP2,
    /// `stype=ucsb`. [`oliu_p`] at a fixed `beta` of 0.13.
    Ucsb,
    /// `stype=ucsb2`. Twice the duration, with the peak left where `ucsb` puts it.
    Ucsb2,
    /// `stype=ucsb-T<b>`. The duration stretched by `b`, with the peak left where
    /// `ucsb` puts it. The C parses `b` off the end of the option string.
    UcsbT {
        /// `b`. One reproduces `ucsb` exactly.
        stretch: f32,
    },
    /// `stype=ucsb-varT1`. `beta` per subfault, from the input file's thirteenth
    /// column. The C defaults it to 0.13 when the column is absent, which is `Ucsb`.
    UcsbVarT1 {
        /// `beta`, the fraction of the duration spent rising.
        tau1_ratio: f32,
    },
}

impl SlipRateShape {
    /// Build one subfault's slip-rate function.
    ///
    /// `duration_s` is the subfault's rise time and `beta` the value
    /// [`beta_field`] gave it, which only [`OliuP2`](Self::OliuP2) reads. Every
    /// shape normalises so `sample_interval_s * sum` is `slip_cm`.
    ///
    /// # Panics
    ///
    /// If `sample_interval_s` is not strictly positive.
    #[must_use]
    pub fn pulse(
        self,
        slip_cm: f32,
        duration_s: f32,
        beta: f32,
        sample_interval_s: f32,
        max_samples: usize,
    ) -> SlipRate {
        let (duration_s, beta) = match self {
            Self::OliuP2 => (duration_s, beta),
            Self::Ucsb => (duration_s, UCSB_BETA),
            Self::Ucsb2 => (2.0 * duration_s, 0.5 * UCSB_BETA),
            Self::UcsbT { stretch } => (stretch * duration_s, UCSB_BETA / stretch),
            Self::UcsbVarT1 { tau1_ratio } => (duration_s, tau1_ratio),
        };
        oliu_p(slip_cm, duration_s, beta, sample_interval_s, max_samples)
    }
}

/// The fraction of the duration the ucsb family spends rising.
///
/// Written `0.13` four times in `generic_slip2srf/slip.c`, and once more as
/// `0.5 * 0.13` in `ucsb2` where the duration doubles.
const UCSB_BETA: f32 = 0.13;
