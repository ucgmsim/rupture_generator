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

    normalise(values, slip, sample_interval_s)
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
    /// `stype=brune`. `(t/T)·exp(-t/T)`, the classic ω⁻² source pulse.
    ///
    /// The only shape whose duration is not simply the rise time in the original:
    /// `generic_slip2srf` sets `T = 0.1·e⁻¹·√slip/1.2` — a time constant derived
    /// from the subfault's *slip* — and then multiplies it by the depth factor. The
    /// port uses the rise time like every other shape. That is a deliberate
    /// difference and the reason is the whole point of this crate having a rise-time
    /// model: a duration that is a function of slip is not one the depth ramp,
    /// the moment scaling and the fault-wide average can all be about at once. A
    /// caller wanting a corner frequency `f` sets the average rise time to
    /// `1/(2πf)`, which is exactly the relation the C's `brune_corner_freq` branch
    /// encodes.
    ///
    /// (orig. `gen_brune_stf`, `slip.c:4`)
    Brune,
    /// `stype=urs`. Two triangles: a narrow spike, then a long low tail.
    ///
    /// The tail's height is the one shape parameter in `generic_slip2srf` that varies
    /// with depth — 0.5 above 4 km falling to 0.2 below 6 km — and it is a
    /// [`DepthRamp`] like every other ramp here. See [`shape_parameter_field`].
    ///
    /// (orig. `gen_2tri_stf`, `slip.c:37`)
    Urs,
    /// `stype=esg2006`. A Gaussian, centred at twice the rise time and truncated at
    /// four times it.
    ///
    /// (orig. `gen_esg2006_stf`, `slip.c:338`)
    Esg2006,
    /// `stype=cos`. A full raised cosine, `1 - cos(2πt/T)`.
    ///
    /// (orig. `gen_cos_stf`, `slip.c:382`)
    Cos,
    /// `stype=seki`. `sech²(2(2t/T − 1))`, peaking at `T/2`.
    ///
    /// **The one shape that does not start at rest**: it is at 7% of its peak at
    /// `t = 0`. The original compensates by moving the subfault's onset back a
    /// quarter of the duration — see [`SlipRateShape::onset_shift_s`], which is
    /// plumbed rather than dropped, because without it the pulse arrives early.
    ///
    /// (orig. `gen_seki_stf`, `slip.c:421`)
    Seki,
    /// `stype=delta`. A single-sample impulse: `[0, slip/dt, 0]`.
    ///
    /// Exactly what [`oliu_p`] substitutes for a pulse too short to resolve, so it is
    /// that branch under its own name rather than a second spelling —
    /// `a_delta_is_the_spike_oliu_p_falls_back_to` asserts the two are identical.
    ///
    /// (orig. `generic_slip2srf.c:456`)
    Delta,
}

impl SlipRateShape {
    /// Build one subfault's slip-rate function.
    ///
    /// `duration_s` is the subfault's rise time and `shape_parameter` the value
    /// [`shape_parameter_field`] gave it, which only [`OliuP2`](Self::OliuP2) and
    /// [`Urs`](Self::Urs) read. **Every shape normalises so `sample_interval_s * sum`
    /// is `slip_cm`**, which is what makes the moment come out right whichever is
    /// chosen, and is the one thing they all have in common.
    ///
    /// # Panics
    ///
    /// If `sample_interval_s` is not strictly positive.
    #[must_use]
    pub fn pulse(
        self,
        slip_cm: f32,
        duration_s: f32,
        shape_parameter: f32,
        sample_interval_s: f32,
        max_samples: usize,
    ) -> SlipRate {
        assert!(
            sample_interval_s > 0.0,
            "the sample interval must be positive"
        );

        // The four that are `oliu_p` with the breakpoints moved.
        let liu = |duration_s: f32, beta: f32| {
            oliu_p(slip_cm, duration_s, beta, sample_interval_s, max_samples)
        };
        match self {
            Self::OliuP2 => liu(duration_s, shape_parameter),
            Self::Ucsb => liu(duration_s, UCSB_BETA),
            Self::Ucsb2 => liu(2.0 * duration_s, 0.5 * UCSB_BETA),
            Self::UcsbT { stretch } => liu(stretch * duration_s, UCSB_BETA / stretch),
            Self::UcsbVarT1 { tau1_ratio } => liu(duration_s, tau1_ratio),

            // The rest are sampled from a closed form and then normalised. One helper
            // does the sampling, the truncation and the normalisation; each shape
            // supplies only its extent and its amplitude.
            Self::Brune => sampled(
                slip_cm,
                BRUNE_TAIL * duration_s,
                sample_interval_s,
                max_samples,
                |time| {
                    let scaled = time / duration_s;
                    scaled * (-scaled).exp()
                },
            ),
            Self::Esg2006 => sampled(
                slip_cm,
                4.0 * duration_s,
                sample_interval_s,
                max_samples,
                |time| {
                    let offset = 4.0 * (time - 2.0 * duration_s) / duration_s;
                    (-offset * offset).exp()
                },
            ),
            Self::Cos => sampled(
                slip_cm,
                duration_s,
                sample_interval_s,
                max_samples,
                |time| 1.0 - (2.0 * PI * time / duration_s).cos(),
            ),
            Self::Seki => sampled(
                slip_cm,
                SEKI_TAIL * duration_s,
                sample_interval_s,
                max_samples,
                |time| {
                    let argument = 2.0 * (2.0 * time / duration_s - 1.0);
                    let sech = argument.cosh().recip();
                    sech * sech
                },
            ),
            Self::Urs => two_triangles(
                slip_cm,
                duration_s,
                shape_parameter,
                sample_interval_s,
                max_samples,
            ),
            Self::Delta => impulse(slip_cm, sample_interval_s, max_samples),
        }
    }

    /// How far this shape moves the subfault's onset, in seconds.
    ///
    /// Zero for every shape but [`Seki`](Self::Seki), which is at 7% of its peak at
    /// `t = 0` — energy before the rupture arrives. The original shifts the onset back
    /// a quarter of the duration to compensate (`generic_slip2srf.c:452`) and clamps
    /// at zero, and so does this.
    ///
    /// Kept separate from [`pulse`](Self::pulse) because it is a fact about the
    /// *arrival*, not about the samples, and the two are stored in different places.
    #[must_use]
    pub fn onset_shift_s(self, duration_s: f32) -> f32 {
        match self {
            Self::Seki => -0.25 * duration_s,
            _ => 0.0,
        }
    }
}

/// The fraction of the duration the ucsb family spends rising.
///
/// Written `0.13` four times in `generic_slip2srf/slip.c`, and once more as
/// `0.5 * 0.13` in `ucsb2` where the duration doubles.
const UCSB_BETA: f32 = 0.13;

/// How many time constants of Brune's exponential tail are kept.
///
/// The original writes `3.0 * 1.745 * e`, which is `t95` — the time by which 95% of
/// the slip has happened — times three. What is discarded is `(1+x)·e⁻ˣ` of the
/// total at `x = 14.2`, about one part in 10⁵; normalising folds that back in, so
/// the truncation costs accuracy in the *shape*'s tail rather than in the moment.
const BRUNE_TAIL: f32 = 3.0 * 1.745 * std::f32::consts::E;

/// How much of `seki`'s `sech²` is kept, in units of the duration.
const SEKI_TAIL: f32 = 1.5;

/// Sample `amplitude` over `[0, extent_s)` and normalise so the integral is `slip_cm`.
///
/// The shared tail of every closed-form shape. It is one function rather than five
/// because the five differ only in two lines each — an extent and an expression —
/// where `generic_slip2srf` repeats the sampling, the cap, the fold and the rescale
/// verbatim in all of them.
///
/// A trailing zero closes the pulse, as it does in [`oliu_p`]. `cos` is the one shape
/// where the original also does this, by returning `nstf + 1` after computing `nstf`
/// samples.
fn sampled(
    slip_cm: f32,
    extent_s: f32,
    sample_interval_s: f32,
    max_samples: usize,
    amplitude: impl Fn(f32) -> f32,
) -> SlipRate {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a small non-negative sample count, truncated as the C truncates it"
    )]
    let count = (((extent_s / sample_interval_s + 0.5) as i32).max(0) as usize).min(max_samples);
    if count == 0 {
        return SlipRate::empty();
    }

    let mut values = vec![0.0_f32; (count + 1).min(max_samples).max(count)];
    for (index, value) in values.iter_mut().enumerate().take(count) {
        #[expect(
            clippy::cast_precision_loss,
            reason = "sample counts are far below 2^24"
        )]
        let time = index as f32 * sample_interval_s;
        *value = amplitude(time);
    }
    normalise(values, slip_cm, sample_interval_s)
}

/// `urs`: a narrow triangle to full amplitude, then a long one from `tail` to zero.
///
/// `tail` is the second triangle's height as a fraction of the first's — the value
/// [`shape_parameter_field`] computes for [`SlipRateShape::Urs`].
///
/// The sample counts are the original's, floors and all: the spike is a tenth of the
/// duration but at least two samples, and the whole pulse at least four. Those floors
/// are what keeps the second triangle inside the first — `(2 - tail)` is at most 1.8,
/// and `1.8 * 2 < 4`.
///
/// (orig. `gen_2tri_stf`, `slip.c:37-112`)
fn two_triangles(
    slip_cm: f32,
    duration_s: f32,
    tail: f32,
    sample_interval_s: f32,
    max_samples: usize,
) -> SlipRate {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "small non-negative sample counts, truncated as the C truncates them"
    )]
    let samples = |seconds: f32| ((seconds / sample_interval_s + 0.5) as i32).max(0) as usize;

    let peak = samples(0.1 * duration_s).max(2);
    let end = samples(duration_s).max(4);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "the C's `it2 = (2 - beta)*it0`, a float product truncated to int"
    )]
    let shoulder = (((2.0 - tail) * peak as f32) as i32).max(0) as usize;
    let shoulder = shoulder.clamp(peak, end);

    let length = (end + 1).min(max_samples);
    if length == 0 {
        return SlipRate::empty();
    }
    let mut values = vec![0.0_f32; length];

    #[expect(
        clippy::cast_precision_loss,
        reason = "sample counts are far below 2^24"
    )]
    let step = 1.0 / peak as f32;
    for (index, value) in values.iter_mut().enumerate().take(end.min(length)) {
        #[expect(
            clippy::cast_precision_loss,
            reason = "sample counts are far below 2^24"
        )]
        let position = index as f32;
        #[expect(
            clippy::cast_precision_loss,
            reason = "sample counts are far below 2^24"
        )]
        let (peak_at, shoulder_at) = (peak as f32, shoulder as f32);
        *value = if index < peak {
            // Rising, zero to one.
            position * step
        } else if index < shoulder {
            // Falling, one to `tail`.
            (2.0 * peak_at - position) * step
        } else {
            // The tail, `tail` to zero.
            #[expect(
                clippy::cast_precision_loss,
                reason = "sample counts are far below 2^24"
            )]
            let span = (end - shoulder) as f32;
            tail + (shoulder_at - position) * tail / span
        };
    }
    normalise(values, slip_cm, sample_interval_s)
}

/// `delta`: everything in one sample.
///
/// The same three values [`oliu_p`] falls back to when a pulse is too short to
/// resolve, which is what a delta function is.
fn impulse(slip_cm: f32, sample_interval_s: f32, max_samples: usize) -> SlipRate {
    let mut values = vec![0.0_f32, 1.0, 0.0];
    values.truncate(max_samples.max(1));
    normalise(values, slip_cm, sample_interval_s)
}

/// Scale so `sample_interval_s * sum` is `slip_cm`, or give up.
///
/// A non-positive integral means the shape degenerated at this resolution, and the
/// subfault contributes nothing rather than contributing garbage. Shared so that
/// "conserves slip" is one line of code rather than a property each shape has to
/// remember to have.
fn normalise(mut values: Vec<f32>, slip_cm: f32, sample_interval_s: f32) -> SlipRate {
    // Accumulated in `f64`, where the original folds through a `float`. The fold runs
    // over every sample of the pulse and the samples span orders of magnitude --
    // `brune`'s tail is 1e-6 of its peak -- so a single-precision accumulator loses
    // the tail entirely. Measured on a 20 s `brune` at 5 ms: the `f32` fold put the
    // integral 1.3e-04 off the slip, against a 5.8e-05 round-trip bound. In `f64` it
    // lands inside the bound for every shape at every duration tested.
    let mut integral = 0.0_f64;
    for value in &values {
        integral += f64::from(sample_interval_s) * f64::from(*value);
    }
    if integral <= 0.0 {
        return SlipRate::empty();
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the narrowing seam: the scale is applied to `f32` samples"
    )]
    let scale = (f64::from(slip_cm) / integral) as f32;
    for value in &mut values {
        *value *= scale;
    }
    SlipRate { values }
}

/// The per-subfault shape parameter, one value per dip row broadcast along strike.
///
/// Two of the eleven shapes vary with depth and they use different ramps, so the
/// dispatch is here rather than at the call site: [`SlipRateShape::OliuP2`] takes
/// genslip's `beta` profile, and [`SlipRateShape::Urs`] the tail height
/// `generic_slip2srf` hardcodes between 4 and 6 km. Everything else gets a value it
/// does not read.
///
/// # Panics
///
/// If `depth_km` does not hold one depth per dip row.
#[must_use]
pub fn shape_parameter_field(
    shape: SlipRateShape,
    strike_count: usize,
    depth_km: &[f32],
    profile: BetaProfile,
) -> SlipField {
    match shape {
        SlipRateShape::OliuP2 => beta_field(strike_count, depth_km, profile),
        SlipRateShape::Urs => {
            let mut field = SlipField::zeros(strike_count, depth_km.len());
            for (dip, depth) in depth_km.iter().enumerate() {
                // Branched, not just scaled. `DepthRamp` interpolates and does not
                // clamp -- every caller brackets it, and this one did not until the
                // reference comparison caught it: at 7 km, one kilometre below the
                // ramp, the unbracketed form gave 0.05 where the C gives 0.2, and
                // half the pulse was wrong. Same shape as `BetaProfile::beta_at`.
                let tail = if *depth <= URS_RAMP.shallow_km() {
                    URS_SHALLOW_TAIL
                } else if *depth < URS_RAMP.deep_km() {
                    URS_DEEP_TAIL + URS_RAMP.scaled_from_deep(URS_TAIL_RANGE, *depth)
                } else {
                    URS_DEEP_TAIL
                };
                for strike in 0..strike_count {
                    field[(strike, dip)] = tail;
                }
            }
            field
        }
        _ => SlipField::zeros(strike_count, depth_km.len()),
    }
}

/// Where `urs`'s tail height changes, `dmin = 4` to `dmax = 6` km (`slip.c:48-49`).
const URS_RAMP: DepthRamp = DepthRamp {
    centre_km: 5.0,
    half_width_km: 1.0,
};
/// `betadeep` (`slip.c:44`): the tail height below the ramp.
const URS_DEEP_TAIL: f32 = 0.2;
/// `betashal` (`slip.c:45`): the tail height above it. A shallow subfault releases
/// more of its slip late.
const URS_SHALLOW_TAIL: f32 = 0.5;
/// `betashal - betadeep` (`slip.c:44-45`): how much taller it is above the ramp.
///
/// The C spells this as a precomputed gradient, `(betadeep - betashal)/(dmax - dmin)`,
/// multiplied by `(dmax - z)`. That is [`DepthRamp::scaled_from_deep`] with the sign
/// folded into the scale, which is how every other ramp in this crate is written.
const URS_TAIL_RANGE: f32 = 0.3;
