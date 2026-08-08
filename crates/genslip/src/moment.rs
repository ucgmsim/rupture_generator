//! Turning a dimensionless slip field into slip in centimetres.
//!
//! The spectral generators fix the *shape* of the slip distribution but say nothing
//! about its size — the field they produce is normalised to unit mean and has no
//! physical units. This scales it, either so the rupture carries a target seismic
//! moment, or so it has a target average slip.
//!
//! Moment matching is the usual path: `M0 = sum over subfaults of mu * area * slip`,
//! so a single factor `M0_target / M0_unscaled` gives the field its magnitude.

use crate::taper::SlipField;

/// Subfault dimensions, in kilometres.
#[derive(Clone, Copy, Debug)]
pub struct SubfaultSize {
    pub strike_km: f32,
    pub dip_km: f32,
}

impl SubfaultSize {
    /// Subfault area in cm², which is what the moment sum needs.
    ///
    /// The kilometre product is formed in single precision and only then widened for
    /// the unit conversion, matching where the original rounds.
    fn area_cm2(self) -> f32 {
        const CM2_PER_KM2: f64 = 1.0e+10;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the narrowing seam: C stores the product into a float"
        )]
        let area = (f64::from(self.strike_km * self.dip_km) * CM2_PER_KM2) as f32;
        area
    }
}

/// What the slip field is scaled to match.
#[derive(Clone, Copy, Debug)]
pub enum SlipScaling {
    /// Scale so the rupture carries this seismic moment, in dyne-cm.
    ///
    /// The usual path. genslip selects it with a negative `target_savg`, which is
    /// the configured default.
    Moment { dyne_cm: f32 },
    /// Scale so the mean slip is this many centimetres, and report the moment that
    /// results.
    AverageSlip { centimetres: f32 },
}

/// A slip field in physical units, with the summary statistics its callers need.
#[derive(Clone, Debug)]
pub struct ScaledSlip {
    /// Slip in centimetres, on the fault's own extent.
    pub slip: SlipField,
    /// Mean slip over the fault, in centimetres.
    pub average_cm: f32,
    /// Largest slip on the fault, in centimetres.
    ///
    /// Floored at zero: the running maximum starts there, so a field that is
    /// negative everywhere reports zero rather than its least-negative value. That
    /// is reproduced from the original, and matters only before negative slip is
    /// truncated away.
    pub maximum_cm: f32,
    /// Seismic moment of the scaled field, in dyne-cm.
    pub moment_dyne_cm: f32,
}

/// Scale a slip field to a target moment or average slip.
///
/// `field` is the padded grid the generators wrote; the fault occupies
/// `dip_count` rows starting at `dip_offset`, at the field's own along-strike
/// stride. `rigidity` is one value per subfault of the fault itself, in the same
/// along-strike-fastest order.
///
/// # Panics
///
/// If the requested fault block does not fit inside `field`, or if `rigidity` does
/// not hold exactly one value per subfault. The original indexes without checking.
///
/// (orig. `scale_slip_r_vsden`, slip.c:502)
#[must_use]
pub fn scale_slip(
    field: &SlipField,
    dip_offset: usize,
    dip_count: usize,
    rigidity: &[f32],
    subfault: SubfaultSize,
    scaling: SlipScaling,
) -> ScaledSlip {
    let strike_count = field.strike_count();
    assert!(
        dip_offset + dip_count <= field.dip_count(),
        "fault block rows {dip_offset}..{} does not fit in a {}-row field",
        dip_offset + dip_count,
        field.dip_count()
    );
    assert_eq!(
        rigidity.len(),
        strike_count * dip_count,
        "got {} rigidity values for {strike_count}x{dip_count} subfaults",
        rigidity.len()
    );

    let area = subfault.area_cm2();
    let unscaled = |strike: usize, dip: usize| field[(strike, dip + dip_offset)];

    // Accumulated in `f64`. The original folds through a `float` over every subfault,
    // and on a 10^5-subfault fault that costs enough precision to matter: the moment
    // the model asks for and the moment the scaled field carries then disagree by
    // about 6e-5 relative, which is six missing subfaults' worth. In `f64` a single
    // missing subfault is visible. `the_scaled_field_carries_the_moment_it_was_asked
    // _for` is the test that became possible.
    let moment_sum = || -> f64 {
        (0..dip_count)
            .flat_map(|dip| (0..strike_count).map(move |strike| (strike, dip)))
            .map(|(strike, dip)| {
                f64::from(area)
                    * f64::from(rigidity[strike + dip * strike_count])
                    * f64::from(unscaled(strike, dip))
            })
            .sum()
    };

    #[expect(
        clippy::cast_precision_loss,
        reason = "subfault counts are far below 2^24"
    )]
    let subfault_count = (strike_count * dip_count) as f64;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the scale factor is applied to an f32 field"
    )]
    let factor = match scaling {
        SlipScaling::Moment { dyne_cm } => (f64::from(dyne_cm) / moment_sum()) as f32,
        SlipScaling::AverageSlip { centimetres } => {
            let sum: f64 = (0..dip_count)
                .flat_map(|dip| (0..strike_count).map(move |strike| (strike, dip)))
                .map(|(strike, dip)| f64::from(unscaled(strike, dip)))
                .sum();
            (f64::from(centimetres) * subfault_count / sum) as f32
        }
    };

    let mut slip = SlipField::zeros(strike_count, dip_count);
    let mut total = 0.0_f64;
    let mut maximum = 0.0_f32;
    for dip in 0..dip_count {
        for strike in 0..strike_count {
            let scaled = factor * unscaled(strike, dip);
            slip[(strike, dip)] = scaled;
            total += f64::from(scaled);
            if scaled > maximum {
                maximum = scaled;
            }
        }
    }

    // In moment mode the moment is the target by construction. In average-slip mode
    // the original recomputes it from the *unscaled* field, which makes the reported
    // moment that of the field before scaling rather than after -- reproduced here,
    // and flagged because it looks like a defect.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the reported summaries are f32; the accumulation is not"
    )]
    let summary = ScaledSlip {
        slip,
        average_cm: (total / subfault_count) as f32,
        maximum_cm: maximum,
        moment_dyne_cm: match scaling {
            SlipScaling::Moment { dyne_cm } => dyne_cm,
            SlipScaling::AverageSlip { .. } => moment_sum() as f32,
        },
    };
    summary
}
