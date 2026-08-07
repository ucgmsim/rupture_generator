//! Generating a random field with a prescribed wavenumber spectrum.
//!
//! Both generators here follow the same recipe: walk the non-negative dip half of
//! the wavenumber grid, draw a complex Gaussian at each point, scale it by an
//! amplitude that depends only on the wavenumber, then reflect the half-grid into a
//! Hermitian-symmetric whole so the inverse transform is real. What differs is the
//! amplitude.
//!
//! * [`correlated_field`] gives slip its spatial correlation structure — a spectrum
//!   that is flat below a corner wavenumber set by the correlation lengths and falls
//!   off above it. Used for slip, rake, and the slip-correlated rupture-time and
//!   rise-time perturbations.
//!
//! * [`self_affine_field`] gives a pure power law with no corner, which is what
//!   fault roughness is (Shi & Day 2014).
//!
//! # Three invariants worth stating before reading the code
//!
//! **Both deviates are drawn at every grid point, unconditionally**, including where
//! the amplitude is then set to zero. The draw count is therefore a function of the
//! grid extent alone — not of the spectrum, the correlation lengths, or the
//! band-pass. Anything that changes it desynchronises every field generated after.
//!
//! **The band-pass never removes a draw.** It is applied to the amplitude, not to
//! the drawing, so narrowing it until the whole grid is zero still consumes exactly
//! the same randomness.
//!
//! **The amplitudes are computed in double precision and narrowed to single.** The
//! grid is `f32`, but every `exp`, `log` and `sqrt` below runs in `f64` and rounds
//! once on the way out. This is not incidental: it is what the original does, since
//! C's `exp`/`log`/`sqrt` take and return `double` and their results are narrowed
//! only where they are stored. Computing the same expressions in `f32` throughout
//! gives different last bits.
//!
//! (orig. slip.c:1482 and slip.c:1585)

use num_complex::Complex32;

use crate::grid::{Spectrum, impose_hermitian_symmetry};
use crate::rng::DrawSource;

/// Order of the band-pass roll-off at both ends.
///
/// A local constant in the original, not the configurable `kord` — that one belongs
/// to the separate band-pass applied to the roughness-correlated rupture-time field.
const BAND_PASS_ORDER: f64 = 4.0;

/// Splits unit variance between the real and imaginary parts.
///
/// Each part gets a standard normal scaled by `1/sqrt(2)`, so the complex deviate
/// has unit total variance. The four self-conjugate points are then divided back
/// out, because those must be real and so carry the whole variance in one component.
const QUADRATURE_NORM: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Which spectral shape a correlated field takes.
///
/// The discriminants are genslip's `kmodel` values (`defs.h:9-14`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum Spectrum2D {
    /// Somerville et al. (1999): amplitude falls as `1 / sqrt(1 + a^2)`.
    Somerville = 1,
    /// Mai & Beroza (2002) von Karman. The default.
    Mai = 2,
    /// Frankel (2009): flat below the corner, then a strict power law.
    Frankel = 3,
    /// A Mai/Somerville hybrid.
    ///
    /// Takes the von Karman shape. In v5.6.2 the blend looks unfinished: a
    /// Somerville correlation-length pair is computed alongside the Mai one and
    /// never consumed.
    MaiSomerville = 4,
    /// Suzuki: von Karman shape with a magnitude-clamped down-dip corner.
    Suzuki = 5,
    /// Corners supplied directly rather than derived from magnitude.
    ///
    /// Takes the Somerville shape.
    InputCorners = -1,
}

impl Spectrum2D {
    /// Hurst-derived exponent of the von Karman falloff, `H + 1` with `H = 0.75`.
    const VON_KARMAN_EXPONENT: f64 = 1.75;
    /// Falloff exponent for the strictly self-similar Frankel branch.
    const FRANKEL_EXPONENT: f64 = 2.0;

    /// Spectral amplitude at normalised squared wavenumber `a`, scaled by `scale`.
    ///
    /// `a` is `(kx * clen_strike)^2 + (ky * clen_dip)^2`: wavenumber measured in
    /// units of the corner set by the correlation lengths, so `a = 1` is the corner
    /// and the shapes differ only in how they roll off past it.
    ///
    /// `scale` is folded in here rather than applied by the caller because the
    /// original divides by the square root *after* multiplying, and the rounding
    /// differs if that order changes.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the narrowing seam: C stores each of these into a float"
    )]
    fn amplitude(self, a: f32, scale: f32) -> f32 {
        // `a` stays single precision until it meets a double. In C the square is
        // float*float and only *then* widens for the addition, which rounds once
        // more than squaring in double would. Widening first shifts the last bit.
        let a_squared = f64::from(a * a);
        let a = f64::from(a);
        let scale = f64::from(scale);
        match self {
            Self::Somerville | Self::InputCorners => (scale / (1.0 + a_squared).sqrt()) as f32,
            Self::Mai | Self::Suzuki | Self::MaiSomerville => {
                // Two narrowings, matching the original's two assignments: the
                // falloff is stored into a float before the division happens.
                //
                // SIMPLIFY: `(1.0 + a).powf(VON_KARMAN_EXPONENT / 2.0)` -- the
                // exponentiation and the square root fold into one call, which also
                // removes the intermediate narrowing. Written this way because the
                // original stores the falloff before taking its root.
                let falloff = (Self::VON_KARMAN_EXPONENT * (1.0 + a).ln()).exp() as f32;
                (scale / f64::from(falloff).sqrt()) as f32
            }
            Self::Frankel => {
                if a < 1.0 {
                    scale as f32
                } else {
                    // SIMPLIFY: `scale / a`. The exponent is exactly -1, since
                    // FRANKEL_EXPONENT is 2.0 and it is halved here, so the whole
                    // exp/log pair is a reciprocal written the long way.
                    (scale * (-0.5 * Self::FRANKEL_EXPONENT * a.ln()).exp()) as f32
                }
            }
        }
    }
}

/// A band-pass on squared wavenumber, expressed as wavelength limits.
///
/// Rolls off above `1 / min_wavelength` and below `1 / max_wavelength`, both at
/// [`BAND_PASS_ORDER`]. Keeps the field free of structure finer than the subfault
/// spacing can represent and longer than the fault itself.
#[derive(Clone, Copy, Debug)]
pub struct WavelengthBand {
    min_squared: f32,
    max_squared: f32,
}

impl WavelengthBand {
    /// Build a band from wavelength limits in kilometres.
    ///
    /// # Panics
    ///
    /// If either limit is not strictly positive. Both are logarithm arguments, so a
    /// zero or negative limit yields an infinite or NaN amplitude across the whole
    /// grid rather than an obviously wrong one.
    #[must_use]
    pub fn new(min_wavelength_km: f32, max_wavelength_km: f32) -> Self {
        assert!(
            min_wavelength_km > 0.0 && max_wavelength_km > 0.0,
            "wavelength limits must be positive, got {min_wavelength_km} and {max_wavelength_km}"
        );
        Self {
            min_squared: min_wavelength_km * min_wavelength_km,
            max_squared: max_wavelength_km * max_wavelength_km,
        }
    }

    /// The band-pass denominator at squared wavenumber `k2`.
    ///
    /// Returned as a `f64` divisor rather than a gain so the caller can do the
    /// single division the original does.
    ///
    /// At `k2 == 0` both logarithms diverge; callers handle the origin separately.
    fn divisor(self, k2: f32) -> f64 {
        self.divisor_at_order(k2, BAND_PASS_ORDER)
    }

    /// As [`divisor`](Self::divisor), but at a caller-chosen roll-off order.
    ///
    /// [`band_pass`] takes its order from configuration; the generators do not.
    fn divisor_at_order(self, k2: f32, order: f64) -> f64 {
        // Both products are single precision before they reach the logarithm: the
        // original multiplies two floats and the widening happens at the call. The
        // rounding differs if the operands are widened first.
        let high = f64::from(k2 * self.min_squared);
        let low = f64::from(k2 * self.max_squared);
        // SIMPLIFY: `high.powi(order)` and `1.0 / low.powi(order)`. The order is
        // always an integer -- 4 in the generators, `kord` in `band_pass`, itself an
        // int getpar -- so both exp/log pairs are integer powers written the long
        // way, a transcendental pair each where three multiplies would do. Written
        // this way because the original does, and because `powi` and `exp(n*ln(x))`
        // disagree in the last bit somewhere in the domain.
        let high_cut = 1.0 + (order * high.ln()).exp();
        let low_cut = 1.0 + (-order * low.ln()).exp();
        high_cut * low_cut
    }
}

/// Wavenumber sample spacing of a grid, in radians per kilometre.
#[derive(Clone, Copy, Debug)]
pub struct WavenumberStep {
    pub strike: f32,
    pub dip: f32,
}

/// Correlation lengths of the slip field, in kilometres.
#[derive(Clone, Copy, Debug)]
pub struct CorrelationLengths {
    pub strike: f32,
    pub dip: f32,
}

/// Signed wavenumber of grid index `index` in an extent of `count` samples.
///
/// Indices past the midpoint represent negative wavenumbers, as in any discrete
/// transform.
fn wavenumber(index: usize, count: usize, step: f32) -> f32 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "grid extents are far below 2^24"
    )]
    let signed = if index <= count / 2 {
        index as f32
    } else {
        index as f32 - count as f32
    };
    signed * step
}

/// Generate a field whose spectrum has a corner set by the correlation lengths.
///
/// `amplitude_scale` multiplies the whole spectrum. The original reads it out of the
/// grid's DC term on entry, which makes the result depend on a value the caller left
/// behind in the buffer; it is an explicit argument here.
///
/// (orig. `kfilt_gaus2`, slip.c:1585)
pub fn correlated_field<S: DrawSource>(
    spectrum: &mut Spectrum,
    source: &mut S,
    shape: Spectrum2D,
    step: WavenumberStep,
    correlation: CorrelationLengths,
    band: WavelengthBand,
    amplitude_scale: f32,
) {
    let strike_count = spectrum.strike_count();
    let dip_count = spectrum.dip_count();

    let strike_correlation_squared = correlation.strike * correlation.strike;
    let dip_correlation_squared = correlation.dip * correlation.dip;

    for dip in 0..=dip_count / 2 {
        let ky = wavenumber(dip, dip_count, step.dip);
        for strike in 0..strike_count {
            let kx = wavenumber(strike, strike_count, step.strike);

            let normalised =
                kx * kx * strike_correlation_squared + ky * ky * dip_correlation_squared;
            let mut amplitude = shape.amplitude(normalised, amplitude_scale);

            let k2 = kx * kx + ky * ky;
            if k2 > 0.0 {
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the narrowing seam: C stores the quotient into a float"
                )]
                let banded = (f64::from(amplitude) / band.divisor(k2)) as f32;
                amplitude = banded;
            }

            spectrum[(strike, dip)] = draw_unit_complex(source) * amplitude;
        }
    }

    // Restore the variance the quadrature split took from the four real points.
    make_corners_real(spectrum, |value| value / QUADRATURE_NORM);
    impose_hermitian_symmetry(spectrum);
}

/// Band-pass an existing field in place, at a caller-chosen order.
///
/// Unlike the band-pass folded into the generators above, this one is applied to a
/// field that already exists, and its order is configurable — it is genslip's
/// `kord`, used on the roughness-correlated rupture-time perturbation.
///
/// # The origin is zeroed by infinity arithmetic
///
/// There is no `k2 > 0` guard here, unlike [`correlated_field`]. At the origin
/// `ln(0)` is `-inf`, so the high-cut term is `1 + exp(-inf) = 1` and the low-cut is
/// `1 + exp(+inf) = inf`; their product is infinite and the gain is `1/inf = 0`.
/// The DC component is therefore removed — deterministically, and only because IEEE
/// arithmetic makes it so rather than because anything says it should.
///
/// (orig. `kfilter`, slip.c:1698)
pub fn band_pass(spectrum: &mut Spectrum, step: WavenumberStep, band: WavelengthBand, order: i32) {
    let strike_count = spectrum.strike_count();
    let dip_count = spectrum.dip_count();
    let order = f64::from(order);

    for dip in 0..=dip_count / 2 {
        let ky = wavenumber(dip, dip_count, step.dip);
        for strike in 0..strike_count {
            let kx = wavenumber(strike, strike_count, step.strike);
            let k2 = kx * kx + ky * ky;

            #[expect(
                clippy::cast_possible_truncation,
                reason = "the narrowing seam: C stores the gain into a float"
            )]
            let gain = (1.0 / band.divisor_at_order(k2, order)) as f32;
            spectrum[(strike, dip)] *= gain;
        }
    }

    // Only the imaginary parts are cleared here; the real parts keep the gain they
    // were just given, unlike the correlated generator which rescales them.
    make_corners_real(spectrum, |value| value);
    impose_hermitian_symmetry(spectrum);
}

/// Generate a self-affine field: a pure power law, no corner.
///
/// `hurst_exponent` is the Hurst exponent `H`; the spectral falloff is `k^-(H+1)`.
///
/// (orig. `kfilt_beta2`, slip.c:1482)
pub fn self_affine_field<S: DrawSource>(
    spectrum: &mut Spectrum,
    source: &mut S,
    hurst_exponent: f32,
    step: WavenumberStep,
    band: WavelengthBand,
) {
    let strike_count = spectrum.strike_count();
    let dip_count = spectrum.dip_count();

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the narrowing seam: C stores the sum into a float"
    )]
    let falloff = (f64::from(hurst_exponent) + 1.0) as f32;

    for dip in 0..=dip_count / 2 {
        let ky = wavenumber(dip, dip_count, step.dip);
        for strike in 0..strike_count {
            let kx = wavenumber(strike, strike_count, step.strike);

            // Drawn before the amplitude is known, and drawn at the origin too,
            // where the amplitude is zero. The draw count depends on the extent
            // alone.
            let deviate = draw_unit_complex(source);

            let amplitude = if strike == 0 && dip == 0 {
                0.0
            } else {
                let k2 = kx * kx + ky * ky;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the narrowing seam: C stores each of these into a float"
                )]
                let gain = (1.0 / band.divisor(k2)) as f32;
                // SIMPLIFY: `f64::from(k2).powf(-0.5 * falloff)`, with the exponent
                // hoisted out of both loops -- it depends only on `hurst_exponent`
                // and is recomputed at every grid point.
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the narrowing seam: C stores the product into a float"
                )]
                let amplitude = (f64::from(gain)
                    * (-0.5 * f64::from(falloff) * f64::from(k2).ln()).exp())
                    as f32;
                amplitude
            };

            spectrum[(strike, dip)] = deviate * amplitude;
        }
    }

    // The origin is zeroed outright rather than merely made real: a self-affine
    // field has no mean, and the power law diverges there. The other three points
    // keep their real part unscaled -- unlike the correlated field, this one does
    // not restore the quadrature variance.
    spectrum[(0, 0)] = Complex32::default();
    make_corners_real(spectrum, |value| value);
    impose_hermitian_symmetry(spectrum);
}

/// One complex deviate with unit total variance.
fn draw_unit_complex<S: DrawSource>(source: &mut S) -> Complex32 {
    // `gaussian` returns f64 and the product with the f32 norm happens in f64,
    // narrowing once on store -- the original's `fnorm*gaus_rand(...)` into a float.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the narrowing seam: C stores each deviate into a float"
    )]
    let real = (f64::from(QUADRATURE_NORM) * source.gaussian(1.0, 0.0)) as f32;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the narrowing seam: C stores each deviate into a float"
    )]
    let imaginary = (f64::from(QUADRATURE_NORM) * source.gaussian(1.0, 0.0)) as f32;
    Complex32::new(real, imaginary)
}

/// Force the four self-conjugate points to be real.
///
/// DC and the three Nyquist corners map to themselves under conjugation, so they
/// cannot carry an imaginary part in a real-valued field.
fn make_corners_real(spectrum: &mut Spectrum, rescale: impl Fn(f32) -> f32) {
    let strike_nyquist = spectrum.strike_count() / 2;
    let dip_nyquist = spectrum.dip_count() / 2;

    for corner in [
        (0, 0),
        (strike_nyquist, 0),
        (0, dip_nyquist),
        (strike_nyquist, dip_nyquist),
    ] {
        spectrum[corner] = Complex32::new(rescale(spectrum[corner].re), 0.0);
    }
}

/// Shift the field's phase, translating it on the fault without regenerating it.
///
/// A translation in space is a linear phase ramp in wavenumber, so the whole slip
/// distribution can be slid along strike and down dip by multiplying each grid point
/// by `exp(-2*pi*i*(shift_strike*kx + shift_dip*ky))`. The shifts are in the same
/// units as the reciprocal of the wavenumber step.
///
/// # The DC term is restored as a magnitude, which can flip its sign
///
/// The original saves `|grid[0]|` on entry and writes it back to the real part
/// afterwards. At the origin both wavenumbers are zero, so the phase factor is
/// exactly 1 and the term is already unchanged — except that the saved value is a
/// *magnitude*. If the DC term was negative going in, it comes out positive. That is
/// reproduced here; whether it is intended is a question for the scientific suite.
///
/// (orig. `shift_phase`, slip.c:1917)
pub fn shift_phase(
    spectrum: &mut Spectrum,
    step: WavenumberStep,
    strike_shift: f64,
    dip_shift: f64,
) {
    let strike_count = spectrum.strike_count();
    let dip_count = spectrum.dip_count();

    // SIMPLIFY: the original writes pi as `4.0*atan(1.0)`. That is exactly
    // `std::f64::consts::PI` -- atan(1) is correctly rounded to pi/4 and scaling by
    // a power of two is exact -- so this one is free rather than bit-moving.
    let strike_argument = 2.0 * std::f64::consts::PI * strike_shift;
    let dip_argument = 2.0 * std::f64::consts::PI * dip_shift;

    // SIMPLIFY: `spectrum[(0, 0)].norm()`, which is a hypot and cannot overflow the
    // way squaring both parts can.
    let dc_magnitude = (spectrum[(0, 0)].re * spectrum[(0, 0)].re
        + spectrum[(0, 0)].im * spectrum[(0, 0)].im)
        .sqrt();

    for dip in 0..=dip_count / 2 {
        let ky = wavenumber(dip, dip_count, step.dip);
        let dip_phase = phase_factor(dip_argument * f64::from(ky));
        for strike in 0..strike_count {
            let kx = wavenumber(strike, strike_count, step.strike);
            let strike_phase = phase_factor(strike_argument * f64::from(kx));

            // Two separate multiplications rather than one combined factor, so the
            // rounding matches: the original applies the along-strike rotation,
            // stores the result, then applies the down-dip one.
            let rotated = spectrum[(strike, dip)] * strike_phase;
            spectrum[(strike, dip)] = rotated * dip_phase;
        }
    }

    spectrum[(0, 0)] = Complex32::new(dc_magnitude, 0.0);
    // Unlike the generators, only the imaginary parts of the remaining three
    // self-conjugate points are cleared; their real parts keep whatever the rotation
    // gave them.
    let strike_nyquist = strike_count / 2;
    let dip_nyquist = dip_count / 2;
    for corner in [
        (strike_nyquist, 0),
        (0, dip_nyquist),
        (strike_nyquist, dip_nyquist),
    ] {
        spectrum[corner] = Complex32::new(spectrum[corner].re, 0.0);
    }

    impose_hermitian_symmetry(spectrum);
}

/// `exp(-i * argument)`, narrowed to single precision as the original stores it.
fn phase_factor(argument: f64) -> Complex32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the narrowing seam: C stores each part into a float"
    )]
    let factor = Complex32::new(argument.cos() as f32, -argument.sin() as f32);
    factor
}
