//! How long each subfault slips for.
//!
//! Rise time is not drawn independently — it tracks slip. A patch that slips more
//! slips for longer, because the two are set by the same dynamics. genslip builds a
//! field correlated with slip, raises it to a power, and normalises it so the
//! fault-wide average rise time comes out at the value the moment implies.
//!
//! Two things are layered on top of the plain correlated field:
//!
//! * **A shallow blend toward slip itself.** Near the surface the correlated field
//!   can give a subfault appreciable slip and near-zero rise time, which means a
//!   physically absurd acceleration. Above a configured depth the field is replaced
//!   by slip, which cannot do that.
//!
//! * **A power law.** `rtime2slip_exp = 0.5` gives rise time scaling as the square
//!   root of slip (Graves & Pitarka 2010); `1.0` gives it scaling with slip, which
//!   is constant slip-rate and hence constant dynamic stress drop (Frankel 2009).
//!
//! (orig. `genslip_v5.6.2.c:2186-2477`)

use crate::slip::PerturbationSpec;
use crate::taper::SlipField;

/// A linear ramp between two depths.
///
/// Weight 0 at or above `centre - half_width`, 1 at or below `centre + half_width`,
/// linear between. The shape genslip uses for every depth-dependent quantity.
#[derive(Clone, Copy, Debug)]
pub struct DepthRamp {
    pub centre_km: f32,
    pub half_width_km: f32,
}

impl DepthRamp {
    /// The shallow end of the ramp.
    #[must_use]
    pub fn shallow_km(self) -> f32 {
        self.centre_km - self.half_width_km
    }

    /// The deep end of the ramp.
    #[must_use]
    pub fn deep_km(self) -> f32 {
        self.centre_km + self.half_width_km
    }

    /// Fraction of the way through the ramp at `depth_km`, clamped to `0..=1`.
    #[must_use]
    pub fn weight(self, depth_km: f32) -> f32 {
        if depth_km <= self.shallow_km() {
            0.0
        } else if depth_km < self.deep_km() {
            (depth_km - self.shallow_km()) / (self.deep_km() - self.shallow_km())
        } else {
            1.0
        }
    }

    /// `scale * (depth - shallow) / width`, **unclamped**.
    ///
    /// # The grouping is the contract
    ///
    /// This multiplies before it divides. `(scale * offset) / width` and
    /// `scale * (offset / width)` are the same number and not the same `f32`, and
    /// the original consistently writes the first — a chain of `*` and `/` at equal
    /// precedence, evaluated left to right. Callers that need the second must say so
    /// and take the difference.
    ///
    /// Unclamped because every caller guards the range with its own branch first;
    /// clamping here would be a second, redundant comparison.
    #[must_use]
    pub fn scaled_from_shallow(self, scale: f32, depth_km: f32) -> f32 {
        scale * (depth_km - self.shallow_km()) / (self.deep_km() - self.shallow_km())
    }

    /// `scale * (deep - depth) / width`, **unclamped**.
    ///
    /// The mirror of [`scaled_from_shallow`](Self::scaled_from_shallow), and not its
    /// complement: `scaled_from_deep(s, d) + scaled_from_shallow(s, d)` is `s` only
    /// in exact arithmetic.
    #[must_use]
    pub fn scaled_from_deep(self, scale: f32, depth_km: f32) -> f32 {
        scale * (self.deep_km() - depth_km) / (self.deep_km() - self.shallow_km())
    }
}

/// How rise time is stretched with depth: longer near the surface and at depth,
/// unstretched in between.
///
/// One definition, used in two places that the original writes out separately and
/// identically — once to compute the fault-wide normalisation constant
/// (`genslip_v5.6.2.c:2429-2453`) and once to set each subfault's actual duration
/// (`gslip_srf_subs.c:1498-1508`). Confirmed branch for branch before merging.
#[derive(Clone, Copy, Debug)]
pub struct RiseTimeStretch {
    pub shallow: DepthRamp,
    pub shallow_factor: f32,
    pub deep: DepthRamp,
    pub deep_factor: f32,
}

impl RiseTimeStretch {
    /// The stretch at `depth_km`.
    ///
    /// Note the asymmetry, which is the original's: the shallow branch measures from
    /// the ramp's *deep* end and the deep branch from its *shallow* end, so each
    /// returns `1 + excess` at the outer edge and `1` at the inner one.
    #[must_use]
    pub fn factor_at(self, depth_km: f32) -> f32 {
        let shallow_excess = self.shallow_factor - 1.0;
        let deep_excess = self.deep_factor - 1.0;

        if depth_km <= self.shallow.shallow_km() {
            1.0 + shallow_excess
        } else if depth_km < self.shallow.deep_km() {
            1.0 + self.shallow.scaled_from_deep(shallow_excess, depth_km)
        } else if depth_km <= self.deep.shallow_km() {
            1.0
        } else if depth_km < self.deep.deep_km() {
            1.0 + self.deep.scaled_from_shallow(deep_excess, depth_km)
        } else {
            1.0 + deep_excess
        }
    }
}

/// How rise time relates to slip.
#[derive(Clone, Copy, Debug)]
pub struct RiseTimeSpec {
    /// Correlation with slip, and the spread of the resulting field.
    pub perturbation: PerturbationSpec,
    /// Depth ramp from slip (shallow) to the correlated field (deep).
    pub shallow_blend: DepthRamp,
    /// Exponent linking rise time to slip. 0.5 is Graves & Pitarka; 1.0 is Frankel.
    ///
    /// Below 0.1 genslip abandons the correlated field entirely for independent
    /// lognormal noise. That path is not implemented — see the module note in
    /// `PRUNED.md`.
    pub slip_exponent: f32,
}

/// Build the dimensionless rise-time field, normalised to unit mean.
///
/// `correlated` is the slip-correlated field from
/// [`crate::slip::correlated_perturbation`], **before** its mean is removed —
/// genslip normalises this one by its average rather than centring it. `slip` is the
/// spatial slip field on the same grid, and `depth_km` gives one depth per dip row.
///
/// # Panics
///
/// If the three inputs disagree about the fault's extent.
///
/// (orig. `genslip_v5.6.2.c:2226-2352`)
#[must_use]
pub fn rise_time_field(
    correlated: &SlipField,
    slip: &SlipField,
    depth_km: &[f32],
    spec: RiseTimeSpec,
) -> SlipField {
    let strike_count = correlated.strike_count();
    let dip_count = correlated.dip_count();
    assert_eq!(
        (slip.strike_count(), slip.dip_count()),
        (strike_count, dip_count),
        "the slip and rise-time fields must cover the same fault"
    );
    assert_eq!(
        depth_km.len(),
        dip_count,
        "got {} depths for {dip_count} dip rows",
        depth_km.len()
    );

    #[expect(
        clippy::cast_precision_loss,
        reason = "subfault counts are far below 2^24"
    )]
    let subfault_count = (strike_count * dip_count) as f32;

    // The shallow blend. Depth is taken per dip row, not per subfault: genslip reads
    // it from the first subfault of each row, which is exact for a planar segment.
    //
    // `DepthRamp::weight` is used here rather than reproduced because the original
    // spells this one the same way: `(dep - r_dmin)/(r_dmax - r_dmin)`, with no
    // scale factor fused into the numerator.
    let mut field = SlipField::zeros(strike_count, dip_count);
    let mut total = 0.0_f32;
    for dip in 0..dip_count {
        let deep_weight = spec.shallow_blend.weight(depth_km[dip]);
        let shallow_weight = 1.0 - deep_weight;
        for strike in 0..strike_count {
            let value =
                deep_weight * correlated[(strike, dip)] + shallow_weight * slip[(strike, dip)];
            field[(strike, dip)] = value;
            total += value;
        }
    }

    // SIMPLIFY: the original writes this as `sqrt(x*x)`, which is `abs(x)` -- and
    // `f32::abs` is a sign-bit mask where `sqrt` is a real square root, so this is
    // both slower and less exact. Dividing by the magnitude rather than the mean is
    // deliberate, though: it flips a negative-mean field positive.
    let mean_magnitude = (total / subfault_count).abs();
    for value in field.as_mut_slice() {
        *value /= mean_magnitude;
    }

    rescale_about_unit_mean(&mut field, spec.perturbation.sigma.max(0.0));

    // Negative rise time is meaningless. Unlike slip, there is no configuration that
    // turns this off.
    for value in field.as_mut_slice() {
        if *value < 0.0 {
            *value = 0.0;
        }
    }

    apply_slip_exponent(&mut field, spec.slip_exponent);
    normalise_to_unit_mean(&mut field);
    field
}

/// Stretch a field about a mean of 1 so its coefficient of variation is `sigma`.
fn rescale_about_unit_mean(field: &mut SlipField, sigma: f32) {
    #[expect(
        clippy::cast_precision_loss,
        reason = "subfault counts are far below 2^24"
    )]
    let subfault_count = (field.strike_count() * field.dip_count()) as f32;

    let mut sum_of_squares = 0.0_f32;
    for value in field.as_slice() {
        sum_of_squares += (*value - 1.0) * (*value - 1.0);
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the narrowing seam: C's sqrt returns double and is stored to a float"
    )]
    let variation = f64::from(sum_of_squares / subfault_count).sqrt() as f32;

    let factor = sigma / variation;
    for value in field.as_mut_slice() {
        *value = factor * (*value - 1.0) + 1.0;
    }
}

/// Raise every positive value to `exponent`.
///
/// Zeros are left alone: they are the truncated negatives, and `0^0.5` is 0 anyway,
/// but the original guards the call rather than relying on that.
fn apply_slip_exponent(field: &mut SlipField, exponent: f32) {
    if exponent <= 0.1 {
        return;
    }
    for value in field.as_mut_slice() {
        if *value > 0.0 {
            // SIMPLIFY: `value.powf(exponent)`. The original spells it
            // `exp(e * log(x))`, a transcendental pair where one call would do.
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the narrowing seam: C stores the result into a float"
            )]
            let raised = (f64::from(exponent) * f64::from(*value).ln()).exp() as f32;
            *value = raised;
        }
    }
}

/// Divide through by the mean, so the field averages 1.
fn normalise_to_unit_mean(field: &mut SlipField) {
    #[expect(
        clippy::cast_precision_loss,
        reason = "subfault counts are far below 2^24"
    )]
    let subfault_count = (field.strike_count() * field.dip_count()) as f32;

    let mut total = 0.0_f32;
    for value in field.as_slice() {
        total += *value;
    }
    let mean = total / subfault_count;
    for value in field.as_mut_slice() {
        *value /= mean;
    }
}

/// How the fault-wide rise-time average is weighted.
///
/// genslip's `svr_wt`, an int selecting among three schemes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Weighting {
    /// Every subfault counts equally.
    ///
    /// The configured default. A 2023 change made this genuinely uniform: it used to
    /// skip zero-slip subfaults, which made it slip-weighted by accident and
    /// underestimated the normalisation by around 10%.
    Uniform,
    /// Weighted by slip.
    BySlip,
    /// Weighted by slip times local rupture speed.
    ///
    /// The two roughly cancel with depth — rise time grows where rupture slows.
    BySlipAndRuptureSpeed,
}

/// Depth-dependent factors applied to rise time and rupture speed.
#[derive(Clone, Copy, Debug)]
pub struct DepthScaling {
    /// How rise time stretches with depth.
    pub stretch: RiseTimeStretch,
    /// Rupture speed as a fraction of shear-wave speed.
    pub rupture_velocity_fraction: f32,
    /// Extra reduction of rupture speed in the shallow zone.
    pub shallow_rupture_velocity: f32,
    /// Extra reduction of rupture speed in the deep zone.
    pub deep_rupture_velocity: f32,
}

/// The rupture speed this weighting uses, which is **not** the one the solver uses.
///
/// See `DEFECTS.md` #2. Both transitions here run the multiplier down to *zero* at
/// the inner edge of the zone, where [`crate::rupture::SpeedProfile`] correctly runs
/// it up to one. Latent — only the slip-and-rupture-speed weighting consults it, and
/// the configured weighting is uniform — but reproduced as it stands.
fn weighting_rupture_speed(depth_km: f32, scaling: DepthScaling, shear_speed: f32) -> f32 {
    let base = shear_speed * scaling.rupture_velocity_fraction;
    let shallow = scaling.stretch.shallow;
    let deep = scaling.stretch.deep;

    if depth_km <= shallow.shallow_km() {
        base * scaling.shallow_rupture_velocity
    } else if depth_km < shallow.deep_km() {
        shallow.scaled_from_deep(base * scaling.shallow_rupture_velocity, depth_km)
    } else if depth_km <= deep.shallow_km() {
        base
    } else if depth_km < deep.deep_km() {
        deep.scaled_from_shallow(base * scaling.deep_rupture_velocity, depth_km)
    } else {
        base * scaling.deep_rupture_velocity
    }
}

/// The constant that puts the fault-wide average rise time where the moment wants it.
///
/// Rise time at a subfault is this scale factor times the depth factor times the
/// normalised field; the constant is chosen so their weighted average is 1.
///
/// `depth_km` and `shear_speed_km_s` give one value per dip row; `slip` and
/// `rise_time` cover the whole fault.
///
/// # Panics
///
/// If the inputs disagree about the fault's extent.
///
/// (orig. `genslip_v5.6.2.c:2413-2474`)
#[must_use]
pub fn rise_time_normalisation(
    rise_time: &SlipField,
    slip: &SlipField,
    depth_km: &[f32],
    shear_speed_km_s: &[f32],
    scaling: DepthScaling,
    weighting: Weighting,
) -> f32 {
    let strike_count = rise_time.strike_count();
    let dip_count = rise_time.dip_count();
    assert_eq!(
        (slip.strike_count(), slip.dip_count()),
        (strike_count, dip_count),
        "the slip and rise-time fields must cover the same fault"
    );
    assert_eq!(depth_km.len(), dip_count, "one depth per dip row");
    assert_eq!(
        shear_speed_km_s.len(),
        dip_count,
        "one shear speed per dip row"
    );

    let mut numerator = 0.0_f32;
    let mut denominator = 0.0_f32;

    for dip in 0..dip_count {
        let rise_factor = scaling.stretch.factor_at(depth_km[dip]);
        let rupture_speed = weighting_rupture_speed(depth_km[dip], scaling, shear_speed_km_s[dip]);

        for strike in 0..strike_count {
            // SIMPLIFY: the original writes each of these as `sqrt(s*s)`, which is
            // `abs(s)`. Slip has already been truncated non-negative by this point,
            // so it is also a no-op -- but reproduced, because proving that here
            // means reasoning about a caller.
            let weight = match weighting {
                Weighting::Uniform => 1.0,
                Weighting::BySlip => slip[(strike, dip)].abs(),
                Weighting::BySlipAndRuptureSpeed => rupture_speed * slip[(strike, dip)].abs(),
            };

            numerator += rise_factor * weight * rise_time[(strike, dip)];
            denominator += weight;
        }
    }

    numerator / denominator
}
