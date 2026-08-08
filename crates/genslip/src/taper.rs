//! Tapering slip to zero at the fault edges.
//!
//! A slip distribution generated in the wavenumber domain has no reason to vanish at
//! the fault boundary, but a rupture that slips right up to its edge is
//! unphysical — the edges are where the fault stops, so slip has to go to zero
//! there. This applies a linear ramp over a configurable fraction of each edge.
//!
//! Applied to the real slip field after the inverse transform, not in the wavenumber
//! domain.

use ndarray::{Array1, azip};

use crate::grid::{FaultAxes, SlipField};
use crate::units;

/// How far each edge taper reaches, as a fraction of the fault's extent.
#[derive(Clone, Copy, Debug, Default)]
pub struct EdgeTapers {
    /// Both along-strike ends. One value, so the taper is symmetric by construction.
    pub sides: f64,
    /// The up-dip (shallow) edge.
    pub top: f64,
    /// The down-dip (deep) edge.
    pub bottom: f64,
}

/// Convert an edge fraction into a width in subfaults.
///
/// Rounds half away from zero, then floors at zero. There is deliberately no upper
/// bound: a fraction above 1/2 gives overlapping tapers and a fraction above 1 makes
/// the two side ramps cross, which the arithmetic below tolerates without any
/// special case. Whether that is *sensible* is the caller's problem, and the
/// configured values are 0.02 and 0.0.
fn taper_width(fraction: f64, extent: usize) -> usize {
    // `+ 0.5` then truncate is round-half-away-from-zero, which is the original's
    // rounding; `units::truncated` floors at zero and handles NaN, so the explicit
    // negative branch this used to carry is gone with it.
    units::truncated(fraction * units::exact(extent) + 0.5)
}

/// The along-strike damping factor at `strike`, for a side taper `width` wide.
///
/// The two ends look asymmetric — `strike < width` against `strike > count - width`,
/// with different numerators — but on a fault wide enough for the ramps to stay
/// apart they are mirror images. The left ramp runs `1/width .. width/width`, so its
/// innermost cell is exactly 1 and undamped; the right ramp runs
/// `(width-1)/width .. 1/width` over the cells strictly inside the boundary. Both
/// damp the same `width - 1` cells to the same values.
///
/// **When the ramps overlap, the right one wins.** The original tests the two
/// conditions with separate `if`s rather than `else if`, so in the overlap the
/// second assignment simply overwrites the first. That happens once
/// `2 * width > count` — a side fraction above about a half — which the configured
/// 0.02 never reaches, but which is reachable and so is reproduced rather than
/// tidied into an `else`. Writing it as `else if` is wrong, and only shows up on a
/// narrow fault with a wide taper.
fn side_damping(strike: usize, count: usize, width: usize) -> f64 {
    if width == 0 {
        return 1.0;
    }

    let ramp = |numerator: usize| units::exact(numerator) / units::exact(width);

    let mut damping = 1.0;
    if strike < width {
        damping = ramp(strike + 1);
    }
    if strike + width > count {
        damping = ramp(count - strike);
    }
    damping
}

/// Taper slip to zero at the fault edges, in place.
///
/// The three passes are the up-dip band, the down-dip band, and the middle. The
/// first two apply both the down-dip ramp and the along-strike ramp; the middle
/// applies only the along-strike one.
///
/// # Panics
///
/// If a taper fraction is wide enough that its ramp would reach past the opposite
/// edge. The original does not check, and writes outside the array when it happens;
/// there is no way to reproduce that in safe Rust and no reason to want to, so this
/// refuses instead. The configured fractions are 0.02 and 0.0, so the bound is
/// nowhere near.
///
/// (orig. `taper_slip_all_r`, slip.c:181)
pub fn taper_edges(slip: &mut SlipField, tapers: &EdgeTapers) {
    let strike_count = slip.strike_count();
    let dip_count = slip.dip_count();

    let side_width = taper_width(tapers.sides, strike_count);
    let top_width = taper_width(tapers.top, dip_count);
    let bottom_width = taper_width(tapers.bottom, dip_count);

    assert!(
        side_width <= strike_count,
        "side taper of {} reaches {side_width} subfaults across a {strike_count}-wide fault",
        tapers.sides
    );
    assert!(
        top_width <= dip_count && bottom_width <= dip_count,
        "dip tapers of {} and {} reach {top_width} and {bottom_width} subfaults \
         across a {dip_count}-deep fault",
        tapers.top,
        tapers.bottom
    );

    let dip_ramp = |index: usize, width: usize| units::exact(index + 1) / units::exact(width);

    // Each pass multiplies whole rows by a per-column profile, so the profiles are
    // built once as vectors and the passes are one `azip!` each. There are two of
    // them and they are *not* the same vector -- see below, it is the whole subtlety
    // of this routine.
    let banded: Array1<f64> = (0..strike_count)
        .map(|strike| side_damping(strike, strike_count, side_width))
        .collect();

    // The middle's profile. Built by accumulating from both ends rather than by
    // indexing, because a cell the two ramps both reach must be damped *twice* --
    // `*=`, not `=`.
    let mut middle: Array1<f64> = Array1::ones(strike_count);
    for strike in 0..side_width {
        let damping = dip_ramp(strike, side_width);
        middle[strike] *= damping;
        middle[strike_count - 1 - strike] *= damping;
    }

    // The up-dip band: both ramps.
    for (dip, mut row) in slip.rows_mut().into_iter().enumerate().take(top_width) {
        let damping = dip_ramp(dip, top_width);
        azip!((cell in &mut row, &side in &banded) *cell *= side * damping);
    }

    // The down-dip band: both ramps, counting inward from the deep edge.
    for (offset, mut row) in slip
        .rows_mut()
        .into_iter()
        .rev()
        .enumerate()
        .take(bottom_width)
    {
        let damping = dip_ramp(offset, bottom_width);
        azip!((cell in &mut row, &side in &banded) *cell *= side * damping);
    }

    // The middle: the along-strike ramp only, and a *different* one.
    //
    // NOT a simplification candidate, despite the two profiles looking like they
    // should be one. `banded` tests `strike < width` and `strike + width > count`,
    // where `middle` accumulates from both ends. The two agree while the ramps stay
    // apart and **disagree once they overlap**: `middle` damps an overlapped cell
    // twice, once from each end, where `banded`'s second condition merely overwrites
    // the first. On a 14-wide fault with an 8-wide taper, cell 6 comes out at 7/8
    // here and at 1 there.
    //
    // Neither is obviously right. Unifying them would silently pick one, so they
    // stay separate until the scientific suite can say which.
    for mut row in slip
        .rows_mut()
        .into_iter()
        .take(dip_count.saturating_sub(bottom_width))
        .skip(top_width)
    {
        azip!((cell in &mut row, &damping in &middle) *cell *= damping);
    }
}
