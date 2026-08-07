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
    CorrelationLengths, Spectrum2D, WavelengthBand, WavenumberStep, correlate_with,
    correlated_field, shift_phase,
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

/// A slip field and the padded-grid statistics the correlations need.
///
/// The statistics are measured on the padded grid immediately after the inverse
/// transform, before the fault's corner is taken — so they describe the generated
/// field rather than the trimmed one. [`reload_for_correlation`] maps the processed
/// field back onto them, which is what keeps every correlated field on the same
/// scale as the slip it blends with.
#[derive(Clone, Debug)]
pub struct NormalisedSlip {
    pub field: SlipField,
    pub padded: crate::stats::MeanAndSigma,
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
) -> NormalisedSlip {
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

    // Measured here, on the whole padded grid, exactly where the original measures
    // it -- before the fault's corner is taken and before anything is normalised.
    let padded = crate::stats::mean_and_sigma(&spectrum);

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
    NormalisedSlip {
        field: slip,
        padded,
    }
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

/// How tightly a perturbation tracks slip, and how far it spreads.
///
/// The two together are what "a perturbation" means here, so they travel together.
#[derive(Clone, Copy, Debug)]
pub struct PerturbationSpec {
    /// Correlation with the slip spectrum, 0 (independent) to 1 (a scaled copy).
    pub correlation: f32,
    /// Standard deviation of the resulting zero-mean field.
    ///
    /// Negative is treated as zero, which is how genslip disables a perturbation
    /// without removing it from the draw sequence.
    pub sigma: f32,
}

/// Rebuild the wavenumber-domain slip field that the perturbations correlate against.
///
/// By this point the slip field on the fault has been truncated, tapered and had its
/// mean and variation fixed — it is no longer the field the generator produced. The
/// perturbations must correlate against *that* field rather than the raw one, so it
/// goes back onto a padded grid and is transformed again.
///
/// Two things about this are worth stating because neither is obvious:
///
/// **The padding is zero-filled first, and that changes the statistics.** The mean
/// and standard deviation are then measured over the *whole padded grid*, zeros
/// included, so they are not the fault's own. The affine map that follows uses them
/// anyway.
///
/// **The map restores the padded field's original mean and deviation**, measured
/// before any of the fault-domain processing. So the padding does not stay zero: it
/// becomes the original mean. That is what keeps the correlated fields on the same
/// scale as the slip field they are blended with.
///
/// (orig. `genslip_v5.6.2.c:1982-2008`)
#[must_use]
pub fn reload_for_correlation<F: Fft>(
    slip: &SlipField,
    fft: &mut F,
    extents: GridExtents,
    spacing: SubfaultSpacing,
    original: crate::stats::MeanAndSigma,
) -> Spectrum {
    let mut spectrum = Spectrum::zeros(extents.padded_strike, extents.padded_dip);
    for dip in 0..slip.dip_count() {
        for strike in 0..slip.strike_count() {
            spectrum[(strike, dip)] = Complex32::new(slip[(strike, dip)], 0.0);
        }
    }

    let padded = crate::stats::mean_and_sigma(&spectrum);
    let factor = original.sigma / padded.sigma;
    for value in spectrum.as_mut_slice() {
        *value = Complex32::new(factor * (value.re - padded.mean) + original.mean, value.im);
    }

    fft::transform_2d(&mut spectrum, fft, Direction::Forward);
    fft::scale(
        &mut spectrum,
        fft::spacing_product(spacing.strike_km, spacing.dip_km),
    );
    spectrum
}

/// A field correlated with the slip distribution, on the fault, un-normalised.
///
/// The shared half of the two perturbations: generate a field with the same spectrum
/// as slip, blend it with slip's own spectrum at the requested correlation, transform
/// back and take the fault's corner. What the callers do next differs — see
/// [`correlated_perturbation`] and [`crate::rise_time::rise_time_field`].
///
/// The correlation is what makes patches that slip more also rupture sooner and slip
/// for longer, without either being a deterministic function of slip.
///
/// (orig. `genslip_v5.6.2.c:2100-2126`)
#[must_use]
pub fn correlated_field_on_fault<S: DrawSource, F: Fft>(
    source: &mut S,
    fft: &mut F,
    reference: &Spectrum,
    extents: GridExtents,
    spacing: SubfaultSpacing,
    spectrum_spec: SpectrumSpec,
    correlation: f32,
) -> SlipField {
    let mut spectrum = unit_spectrum(fft, extents, spacing);
    let step = extents.wavenumber_step(spacing);

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

    correlate_with(&mut spectrum, reference, correlation);

    fft::transform_2d(&mut spectrum, fft, Direction::Inverse);
    fft::scale(&mut spectrum, fft::spacing_product(step.strike, step.dip));

    extract_corner(&spectrum, extents)
}

/// A zero-mean field correlated with slip, centred and scaled to a target spread.
///
/// [`correlated_field_on_fault`] and then centre-and-rescale. That last step is what
/// the rupture-time perturbation wants and the rise-time one does not: rise time
/// normalises by the *magnitude* of its mean instead, which is what flips a
/// negative-mean field positive rather than centring it on zero.
///
/// (orig. `genslip_v5.6.2.c:2128-2166`)
#[must_use]
pub fn correlated_perturbation<S: DrawSource, F: Fft>(
    source: &mut S,
    fft: &mut F,
    reference: &Spectrum,
    extents: GridExtents,
    spacing: SubfaultSpacing,
    spectrum_spec: SpectrumSpec,
    perturbation: PerturbationSpec,
) -> SlipField {
    let mut field = correlated_field_on_fault(
        source,
        fft,
        reference,
        extents,
        spacing,
        spectrum_spec,
        perturbation.correlation,
    );
    remove_mean(&mut field);
    rescale_to_sigma(&mut field, perturbation.sigma.max(0.0));
    field
}

/// A rake perturbation field, added to the fault's base rake.
///
/// Unlike the perturbations above this does **not** correlate with slip: rake is
/// generated independently, so a patch that slips more has no reason to slip in a
/// different direction. `base_rake` is one value per subfault, in degrees.
///
/// # Panics
///
/// If `base_rake` does not hold exactly one value per subfault.
///
/// (orig. `genslip_v5.6.2.c:2014-2094`)
#[must_use]
pub fn rake_field<S: DrawSource, F: Fft>(
    source: &mut S,
    fft: &mut F,
    extents: GridExtents,
    spacing: SubfaultSpacing,
    spectrum_spec: SpectrumSpec,
    base_rake: &[f32],
    sigma_degrees: f32,
) -> SlipField {
    assert_eq!(
        base_rake.len(),
        extents.fault_strike * extents.fault_dip,
        "got {} base rakes for a {}x{} fault",
        base_rake.len(),
        extents.fault_strike,
        extents.fault_dip
    );

    let mut spectrum = unit_spectrum(fft, extents, spacing);
    let step = extents.wavenumber_step(spacing);

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

    fft::transform_2d(&mut spectrum, fft, Direction::Inverse);
    fft::scale(&mut spectrum, fft::spacing_product(step.strike, step.dip));

    let mut field = extract_corner(&spectrum, extents);
    remove_mean(&mut field);

    let variation = population_sigma(&field);
    let factor = sigma_degrees / variation;
    for (value, base) in field.as_mut_slice().iter_mut().zip(base_rake) {
        *value = factor * *value + *base;
    }
    field
}

/// A padded grid of ones, transformed and scaled — the starting point of every field.
fn unit_spectrum<F: Fft>(fft: &mut F, extents: GridExtents, spacing: SubfaultSpacing) -> Spectrum {
    let mut spectrum = Spectrum::zeros(extents.padded_strike, extents.padded_dip);
    for value in spectrum.as_mut_slice() {
        *value = Complex32::new(1.0, 0.0);
    }
    fft::transform_2d(&mut spectrum, fft, Direction::Forward);
    fft::scale(
        &mut spectrum,
        fft::spacing_product(spacing.strike_km, spacing.dip_km),
    );
    spectrum
}

/// The fault's own corner of a padded grid, real parts only.
fn extract_corner(spectrum: &Spectrum, extents: GridExtents) -> SlipField {
    let mut field = SlipField::zeros(extents.fault_strike, extents.fault_dip);
    for dip in 0..extents.fault_dip {
        for strike in 0..extents.fault_strike {
            field[(strike, dip)] = spectrum[(strike, dip)].re;
        }
    }
    field
}

/// Subtract the field's mean, leaving it centred on zero.
fn remove_mean(field: &mut SlipField) {
    #[expect(
        clippy::cast_precision_loss,
        reason = "subfault counts are far below 2^24"
    )]
    let count = (field.strike_count() * field.dip_count()) as f32;

    let mut total = 0.0_f32;
    for value in field.as_slice() {
        total += *value;
    }
    let mean = total / count;
    for value in field.as_mut_slice() {
        *value -= mean;
    }
}

/// Population standard deviation of a zero-mean field.
fn population_sigma(field: &SlipField) -> f32 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "subfault counts are far below 2^24"
    )]
    let count = (field.strike_count() * field.dip_count()) as f32;

    // SIMPLIFY: single-precision fold again.
    let mut sum_of_squares = 0.0_f32;
    for value in field.as_slice() {
        sum_of_squares += *value * *value;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the narrowing seam: C's sqrt returns double and is stored to a float"
    )]
    let sigma = f64::from(sum_of_squares / count).sqrt() as f32;
    sigma
}

/// Scale a zero-mean field so its standard deviation is `sigma`.
fn rescale_to_sigma(field: &mut SlipField, sigma: f32) {
    let factor = sigma / population_sigma(field);
    for value in field.as_mut_slice() {
        *value *= factor;
    }
}

/// Consume the randomness a spectral field would have used, without building it.
///
/// # Why two of genslip's fields are skipped rather than ported
///
/// The **roughness** field perturbs each subfault's position, strike and dip by an
/// amount proportional to `alpha_rough`, which is configured to `0.0`. Every
/// perturbation is therefore exactly zero and the field has no effect on anything.
///
/// **`tsfac2`** is a rupture-time perturbation correlated with that roughness. Its
/// values reach the model through one branch at `genslip_v5.6.2.c:3143`, and that
/// branch is unreachable: it is the `else` of `if(tsfac_main > -1.0e+10)`, and
/// `tsfac_main` is resolved at line 1255 to `tsfac_bzero + tsfac_slope * Mo^(1/3)`
/// whenever it was left at its sentinel — a small negative number, never below
/// `-1e10`. So the `if` always wins and `tsfac2_r` is never read.
///
/// Neither field is *inert*, though, because both draw. They sit on the 3x refined
/// grid, which is squared up to `max(strike, dip)` in both directions, so together
/// they are the largest consumer of randomness in the program. Skipping the draws
/// as well would change every field generated afterwards while still producing
/// output that looks entirely plausible.
///
/// # What would bring them back
///
/// Setting `alpha_rough` above zero, or passing `tsfac_main` below `-1e10` to reach
/// the `tsfac2` branch. The input layer should refuse both rather than silently
/// producing a model that ignores them — see `PRUNED.md`.
///
/// (orig. `genslip_v5.6.2.c:2482-2789`)
pub fn skip_unused_field<S: DrawSource>(source: &mut S, strike_count: usize, dip_count: usize) {
    source.skip_gaussians(unused_field_draw_count(strike_count, dip_count));
}

/// Normal deviates a spectral field of this extent consumes.
///
/// The generators walk the non-negative dip half — `dip_count / 2 + 1` rows, the
/// midpoint included — over every strike index, drawing a complex deviate at each
/// point. Two normals per point, unconditionally, whatever the spectrum.
#[must_use]
pub const fn unused_field_draw_count(strike_count: usize, dip_count: usize) -> usize {
    2 * strike_count * (dip_count / 2 + 1)
}

/// The reloaded slip spectrum brought back to the fault, in space.
///
/// The original inverse-transforms `slip_c` **in place** at
/// `genslip_v5.6.2.c:2225` — after both correlations have consumed it in the
/// wavenumber domain — and the shallow rise-time blend then reads `slip_c[ip2].re`
/// from that spatial field (line 2255).
///
/// It is **not** the tapered slip field the reload was built from. The reload
/// renormalises the whole padded grid, zeros included, onto the *original*
/// generated field's mean and sigma, so the on-fault values come back through an
/// affine map that depends on how much of the padded grid is padding. Blending
/// against the un-renormalised field instead is a different weighting of slip
/// against the correlated field, and it moves every rise time — including on rows
/// far below the blend zone, because the normalisations that follow are global.
///
/// (orig. `genslip_v5.6.2.c:2225`)
#[must_use]
pub fn reference_on_fault<F: Fft>(
    reference: &Spectrum,
    fft: &mut F,
    extents: GridExtents,
    spacing: SubfaultSpacing,
) -> SlipField {
    let mut spectrum = reference.clone();
    let step = extents.wavenumber_step(spacing);
    fft::transform_2d(&mut spectrum, fft, Direction::Inverse);
    fft::scale(&mut spectrum, fft::spacing_product(step.strike, step.dip));
    extract_corner(&spectrum, extents)
}
