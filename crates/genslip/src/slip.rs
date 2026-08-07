//! Generating a slip distribution.
//!
//! This is the pipeline the earlier modules exist to serve. Every stage of it is
//! pinned individually against the C; what lives here is the *order*, which in the
//! original is a few hundred lines inline in `main`.
//!
//! ```text
//!   a padded grid of ones
//!     -> forward transform            (to wavenumber)
//!     -> correlated_field             (give it the target spectrum)
//!     -> shift_phase                  (optional: translate it on the fault)
//!     -> inverse transform            (back to the fault)
//!     -> take the fault's own corner   of the padded grid
//!     -> fix the polarity              so the mean is positive
//!     -> normalise                     to unit mean
//!     -> rescale                       to the target coefficient of variation
//! ```
//!
//! The result is dimensionless, mean 1. Truncating negative slip, tapering the edges
//! and scaling to a moment happen afterwards and are the caller's to sequence — see
//! [`crate::taper`] and [`crate::moment`].
//!
//! # Why the grid is padded, and why the answer is its corner
//!
//! A discrete transform is periodic, so a slip distribution generated on a grid the
//! size of the fault wraps: structure running off one end reappears at the other. The
//! grid is therefore ~10% larger in each direction and only the fault's own corner is
//! kept. The rest is what absorbs the wraparound.
//!
//! (orig. `genslip_v5.6.2.c:1697-1860`)

use num_complex::Complex32;

use crate::fft::{self, Direction, Fft};
use crate::field::{
    CorrelationLengths, Spectrum2D, WavelengthBand, WavenumberStep, correlated_field, shift_phase,
};
use crate::grid::Spectrum;
use crate::rng::DrawSource;
use crate::taper::SlipField;

/// Sample spacing of the fault grid, in kilometres.
#[derive(Clone, Copy, Debug)]
pub struct SubfaultSpacing {
    pub strike_km: f32,
    pub dip_km: f32,
}

/// How large the fault is, in subfaults, and how large the padded grid around it is.
///
/// The padded extents are the caller's to compute — genslip derives them from the
/// fault's share of a possibly multi-segment rupture — and both are always even.
#[derive(Clone, Copy, Debug)]
pub struct GridExtents {
    pub fault_strike: usize,
    pub fault_dip: usize,
    pub padded_strike: usize,
    pub padded_dip: usize,
}

impl GridExtents {
    /// Wavenumber step of the padded grid, in cycles per kilometre.
    #[must_use]
    fn wavenumber_step(self, spacing: SubfaultSpacing) -> WavenumberStep {
        #[expect(
            clippy::cast_precision_loss,
            reason = "subfault counts are far below 2^24"
        )]
        let step = WavenumberStep {
            strike: 1.0 / (self.padded_strike as f32 * spacing.strike_km),
            dip: 1.0 / (self.padded_dip as f32 * spacing.dip_km),
        };
        step
    }
}

/// The spectral shape a slip field is given.
#[derive(Clone, Copy, Debug)]
pub struct SpectrumSpec {
    pub shape: Spectrum2D,
    pub correlation: CorrelationLengths,
    pub band: WavelengthBand,
    /// Target coefficient of variation. Non-positive leaves the field's own.
    ///
    /// genslip forces this negative for the Frankel spectrum, which rescales the
    /// field by subtracting its minimum instead.
    pub coefficient_of_variation: f32,
    /// Translation of the field on the fault, in fault lengths. Usually zero.
    pub phase_shift: (f64, f64),
}

/// Generate a dimensionless slip field with unit mean.
///
/// # Panics
///
/// If the fault does not fit inside the padded grid, or if either padded extent is
/// odd — see [`Spectrum::zeros`].
#[must_use]
pub fn generate_normalised<S: DrawSource, F: Fft>(
    source: &mut S,
    fft: &mut F,
    extents: GridExtents,
    spacing: SubfaultSpacing,
    spectrum_spec: SpectrumSpec,
) -> SlipField {
    assert!(
        extents.fault_strike <= extents.padded_strike && extents.fault_dip <= extents.padded_dip,
        "a {}x{} fault does not fit in a {}x{} grid",
        extents.fault_strike,
        extents.fault_dip,
        extents.padded_strike,
        extents.padded_dip
    );

    let mut spectrum = Spectrum::zeros(extents.padded_strike, extents.padded_dip);
    let step = extents.wavenumber_step(spacing);

    // A field of ones, transformed, is a single spike at the origin carrying the
    // whole grid. Its height is what sets the generated field's overall amplitude:
    // `correlated_field` scales its spectrum by it. Writing it directly would be
    // clearer and is what the port's signature allows, but the round trip is what
    // the original does and it is the value the scale must match.
    for value in spectrum.as_mut_slice() {
        *value = Complex32::new(1.0, 0.0);
    }
    fft::transform_2d(&mut spectrum, fft, Direction::Forward);
    fft::scale(
        &mut spectrum,
        fft::spacing_product(spacing.strike_km, spacing.dip_km),
    );

    let amplitude_scale = spectrum[(0, 0)].norm();
    correlated_field(
        &mut spectrum,
        source,
        spectrum_spec.shape,
        step,
        spectrum_spec.correlation,
        spectrum_spec.band,
        amplitude_scale,
    );

    let (strike_shift, dip_shift) = spectrum_spec.phase_shift;
    if strike_shift != 0.0 || dip_shift != 0.0 {
        shift_phase(&mut spectrum, step, strike_shift, dip_shift);
    }

    fft::transform_2d(&mut spectrum, fft, Direction::Inverse);
    fft::scale(&mut spectrum, fft::spacing_product(step.strike, step.dip));

    let mut slip = SlipField::zeros(extents.fault_strike, extents.fault_dip);
    let mut total = 0.0_f32;
    for dip in 0..extents.fault_dip {
        for strike in 0..extents.fault_strike {
            let value = spectrum[(strike, dip)].re;
            slip[(strike, dip)] = value;
            total += value;
        }
    }

    // A generated field is as likely to come out negative-mean as positive, and
    // everything downstream -- the correlations with rise time and rupture time --
    // assumes slip is mostly positive. Flipping the whole field is legitimate
    // because its sign is not physically determined; only its structure is.
    if total < 0.0 {
        for value in slip.as_mut_slice() {
            *value = -*value;
        }
        total = -total;
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "subfault counts are far below 2^24"
    )]
    let subfault_count = (extents.fault_strike * extents.fault_dip) as f32;
    let mean = total / subfault_count;
    for value in slip.as_mut_slice() {
        *value /= mean;
    }

    rescale_variation(&mut slip, spectrum_spec.coefficient_of_variation);
    slip
}

/// Rescale a unit-mean field about its mean so its coefficient of variation matches.
///
/// A non-positive target leaves the field alone: the spectrum already determines the
/// variation, and this only stretches it to a configured value.
fn rescale_variation(slip: &mut SlipField, target: f32) {
    /// The mean is exactly 1 here by construction, so it is not recomputed.
    const MEAN: f32 = 1.0;

    if target <= 0.0 {
        return;
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "subfault counts are far below 2^24"
    )]
    let subfault_count = (slip.strike_count() * slip.dip_count()) as f32;

    // SIMPLIFY: a single-precision fold again, over every subfault.
    let mut sum_of_squares = 0.0_f32;
    for value in slip.as_slice() {
        sum_of_squares += (*value - MEAN) * (*value - MEAN);
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the narrowing seam: C's sqrt returns double and is stored to a float"
    )]
    let variation = f64::from(sum_of_squares / subfault_count).sqrt() as f32 / MEAN;

    let factor = target / variation;
    for value in slip.as_mut_slice() {
        *value = factor * (*value - MEAN) + MEAN;
    }
}

/// Clip negative slip to zero, returning the fraction of subfaults clipped.
///
/// The generated field is symmetric about its mean, so a fault with a large
/// coefficient of variation will have negative patches. They are not physical — this
/// is a model of slip, not of slip deficit — so they are clipped. The fraction is
/// reported because it says how much the clipping distorted the spectrum the
/// generator was asked for: a large value means the requested variation was not
/// really achievable.
pub fn truncate_negative_slip(slip: &mut SlipField) -> f32 {
    let mut clipped = 0.0_f32;
    for value in slip.as_mut_slice() {
        if *value < 0.0 {
            clipped += 1.0;
            *value = 0.0;
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "subfault counts are far below 2^24"
    )]
    let subfault_count = (slip.strike_count() * slip.dip_count()) as f32;
    clipped / subfault_count
}

/// Raise every subfault to at least `fraction` of the field's mean.
///
/// Fills in near-zero patches, which would otherwise contribute nothing to the
/// radiated field. Off by default.
pub fn apply_water_level(slip: &mut SlipField, mean: f32, fraction: f32) {
    if fraction <= 0.0 {
        return;
    }
    let floor = mean * fraction;
    for value in slip.as_mut_slice() {
        if *value < floor {
            *value = floor;
        }
    }
}
