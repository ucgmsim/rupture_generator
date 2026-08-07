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
    /// Fraction of the way through the ramp at `depth_km`.
    #[must_use]
    pub fn weight(self, depth_km: f32) -> f32 {
        let shallow = self.centre_km - self.half_width_km;
        let deep = self.centre_km + self.half_width_km;
        if depth_km <= shallow {
            0.0
        } else if depth_km < deep {
            (depth_km - shallow) / (deep - shallow)
        } else {
            1.0
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
    /// Shallow ramp: rise time is multiplied by `shallow_factor` above it.
    pub shallow: DepthRamp,
    pub shallow_factor: f32,
    /// Deep ramp: rise time is multiplied by `deep_factor` below it.
    pub deep: DepthRamp,
    pub deep_factor: f32,
    /// Rupture speed as a fraction of shear-wave speed.
    pub rupture_velocity_fraction: f32,
    /// Extra reduction of rupture speed in the shallow zone.
    pub shallow_rupture_velocity: f32,
    /// Extra reduction of rupture speed in the deep zone.
    pub deep_rupture_velocity: f32,
}

/// Rise-time and rupture-speed factors at one depth.
///
/// The two are computed together because they share the same ramps and the
/// slip-and-rupture-speed weighting needs both.
fn factors_at(depth_km: f32, scaling: DepthScaling, shear_speed: f32) -> (f32, f32) {
    let shallow_min = scaling.shallow.centre_km - scaling.shallow.half_width_km;
    let shallow_max = scaling.shallow.centre_km + scaling.shallow.half_width_km;
    let deep_min = scaling.deep.centre_km - scaling.deep.half_width_km;
    let deep_max = scaling.deep.centre_km + scaling.deep.half_width_km;

    let shallow_excess = scaling.shallow_factor - 1.0;
    let deep_excess = scaling.deep_factor - 1.0;
    let base_speed = shear_speed * scaling.rupture_velocity_fraction;

    if depth_km <= shallow_min {
        (
            base_speed * scaling.shallow_rupture_velocity,
            1.0 + shallow_excess,
        )
    } else if depth_km < shallow_max {
        let fraction = (shallow_max - depth_km) / (shallow_max - shallow_min);
        (
            base_speed * scaling.shallow_rupture_velocity * fraction,
            1.0 + shallow_excess * fraction,
        )
    } else if depth_km <= deep_min {
        (base_speed, 1.0)
    } else if depth_km < deep_max {
        let fraction = (depth_km - deep_min) / (deep_max - deep_min);
        (
            base_speed * scaling.deep_rupture_velocity * fraction,
            1.0 + deep_excess * fraction,
        )
    } else {
        (
            base_speed * scaling.deep_rupture_velocity,
            1.0 + deep_excess,
        )
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
        let (rupture_speed, rise_factor) =
            factors_at(depth_km[dip], scaling, shear_speed_km_s[dip]);

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
