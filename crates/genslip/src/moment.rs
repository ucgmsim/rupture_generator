//! Turning a dimensionless slip field into slip in centimetres.
//!
//! The spectral generators fix the *shape* of the slip distribution but say nothing
//! about its size — the field they produce is normalised to unit mean and has no
//! physical units. This scales it, either so the rupture carries a target seismic
//! moment, or so it has a target average slip.
//!
//! Moment matching is the usual path: `M0 = sum over subfaults of mu * area * slip`,
//! so a single factor `M0_target / M0_unscaled` gives the field its magnitude.

use ndarray::ArrayView2;

use crate::grid::{FaultAxes, SlipField};
use crate::units;

/// Subfault dimensions, in kilometres.
#[derive(Clone, Copy, Debug)]
pub struct SubfaultSize {
    pub strike_km: f64,
    pub dip_km: f64,
}

impl SubfaultSize {
    /// Subfault area in cm², which is what the moment sum needs.
    fn area_cm2(self) -> f64 {
        self.strike_km * self.dip_km * units::CM2_PER_KM2
    }
}

/// What the slip field is scaled to match.
#[derive(Clone, Copy, Debug)]
pub enum SlipScaling {
    /// Scale so the rupture carries this seismic moment, in dyne-cm.
    ///
    /// The usual path. genslip selects it with a negative `target_savg`, which is
    /// the configured default.
    Moment { dyne_cm: f64 },
    /// Scale so the mean slip is this many centimetres, and report the moment that
    /// results.
    AverageSlip { centimetres: f64 },
}

/// A slip field in physical units, with the summary statistics its callers need.
#[derive(Clone, Debug)]
pub struct ScaledSlip {
    /// Slip in centimetres, on the fault's own extent.
    pub slip: SlipField,
    /// Mean slip over the fault, in centimetres.
    pub average_cm: f64,
    /// Largest slip on the fault, in centimetres.
    ///
    /// Floored at zero: the running maximum starts there, so a field that is
    /// negative everywhere reports zero rather than its least-negative value. That
    /// is reproduced from the original, and matters only before negative slip is
    /// truncated away.
    pub maximum_cm: f64,
    /// Seismic moment of the scaled field, in dyne-cm.
    pub moment_dyne_cm: f64,
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
#[expect(
    clippy::needless_pass_by_value,
    reason = "an `ArrayView` is a borrow already; `&ArrayView` is a second indirection \
              that reads worse and buys nothing"
)]
pub fn scale_slip(
    field: ArrayView2<'_, f64>,
    rigidity: ArrayView2<'_, f64>,
    subfault: SubfaultSize,
    scaling: SlipScaling,
) -> ScaledSlip {
    // Both grids cover the same fault. `dip_offset`/`dip_count` used to carve a block
    // out of a larger field and no caller ever passed anything but the whole thing;
    // a view does that job at the call site if it is ever wanted again.
    assert_eq!(
        field.extent(),
        rigidity.extent(),
        "the slip and rigidity grids must cover the same fault"
    );

    let area = subfault.area_cm2();

    // Accumulated in `f64`. The original folds through a `float` over every subfault,
    // and on a 10^5-subfault fault that costs enough precision to matter: the moment
    // the model asks for and the moment the scaled field carries then disagree by
    // about 6e-5 relative, which is six missing subfaults' worth. In `f64` a single
    // missing subfault is visible. `the_scaled_field_carries_the_moment_it_was_asked
    // _for` is the test that became possible.
    let moment_sum = || area * (&field * &rigidity).sum();

    let factor = match scaling {
        SlipScaling::Moment { dyne_cm } => dyne_cm / moment_sum(),
        SlipScaling::AverageSlip { centimetres } => centimetres / field.mean().unwrap_or(0.0),
    };

    let slip = factor * &field;

    // In moment mode the moment is the target by construction. In average-slip mode
    // the original recomputes it from the *unscaled* field, which makes the reported
    // moment that of the field before scaling rather than after -- reproduced here,
    // and flagged because it looks like a defect.

    ScaledSlip {
        average_cm: slip.mean().unwrap_or(0.0),
        maximum_cm: slip.fold(0.0_f64, |highest, value| highest.max(*value)),
        moment_dyne_cm: match scaling {
            SlipScaling::Moment { dyne_cm } => dyne_cm,
            SlipScaling::AverageSlip { .. } => moment_sum(),
        },
        slip,
    }
}
