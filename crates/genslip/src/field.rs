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
//! grid is `f64`, but every `exp`, `log` and `sqrt` below runs in `f64` and rounds
//! once on the way out. This is not incidental: it is what the original does, since
//! C's `exp`/`log`/`sqrt` take and return `double` and their results are narrowed
//! only where they are stored. Computing the same expressions in `f64` throughout
//! gives different last bits.
//!
//! (orig. slip.c:1482 and slip.c:1585)

use crate::units;
use num_complex::Complex64;

use crate::grid::{FaultAxes, FaultAxesMut, Spectrum, impose_hermitian_symmetry};
use crate::rng::DrawSource;

/// Order of the band-pass roll-off at both ends.
///
/// A local constant in the original, not the configurable `kord` — that one belongs
/// to the separate band-pass applied to the roughness-correlated rupture-time field.
const BAND_PASS_ORDER: i32 = 4;

/// Splits unit variance between the real and imaginary parts.
///
/// Each part gets a standard normal scaled by `1/sqrt(2)`, so the complex deviate
/// has unit total variance. The four self-conjugate points are then divided back
/// out, because those must be real and so carry the whole variance in one component.
const QUADRATURE_NORM: f64 = std::f64::consts::FRAC_1_SQRT_2;

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

    /// Whether a generated field is made non-negative by *shifting* it.
    ///
    /// A field drawn through any of these spectra comes out with roughly zero mean
    /// and both signs. There are two ways to turn that into slip, and which one is
    /// used is a property of the spectrum:
    ///
    /// - **About the mean** (everything but Frankel). Normalise to unit mean, then
    ///   stretch about it until the coefficient of variation is the configured one.
    ///   The field keeps its negative values, and truncation removes them later.
    /// - **From the minimum** (Frankel). Subtract the field's own minimum, so the
    ///   least-slipping subfault becomes exactly zero and nothing is negative, then
    ///   normalise to unit mean. The spread is then whatever the spectrum gave and
    ///   the configured coefficient of variation is **ignored** — genslip says this
    ///   by assigning `slip_sigma = -1.0`, which its one `slip_sigma > 0` guard then
    ///   skips on.
    ///
    /// The two are not small variations on each other. Both are affine in the
    /// generated field, so they produce the same *pattern*, but a stretch and a shift
    /// give different spreads — and it is the spread that survives truncation to
    /// become a different rupture. Getting this wrong left slip correlated at 0.993
    /// with the original and 63% too variable. `DEFECTS.md` 18.
    ///
    /// (orig. `genslip_v5.6.2.c:1809-1825`)
    #[must_use]
    pub const fn normalises_from_its_minimum(self) -> bool {
        matches!(self, Self::Frankel)
    }

    /// Spectral amplitude at normalised squared wavenumber `a`, scaled by `scale`.
    ///
    /// `a` is `(kx * clen_strike)^2 + (ky * clen_dip)^2`: wavenumber measured in
    /// units of the corner set by the correlation lengths, so `a = 1` is the corner
    /// and the shapes differ only in how they roll off past it.
    ///
    /// `scale` is folded in here rather than applied by the caller because the
    /// original divides by the square root *after* multiplying, and the rounding
    /// differs if that order changes.
    fn amplitude(self, a: f64, scale: f64) -> f64 {
        // `a` stays single precision until it meets a double. In C the square is
        // float*float and only *then* widens for the addition, which rounds once
        // more than squaring in double would. Widening first shifts the last bit.
        let a_squared = a * a;
        match self {
            Self::Somerville | Self::InputCorners => scale / (1.0 + a_squared).sqrt(),
            // von Karman. The exponent is halved because the amplitude is the square
            // root of the power spectrum, and folding the two into one call also
            // removes a narrowing the original had -- it stored the falloff into a
            // float before taking its root.
            Self::Mai | Self::Suzuki | Self::MaiSomerville => {
                scale / (1.0 + a).powf(Self::VON_KARMAN_EXPONENT / 2.0)
            }
            Self::Frankel => {
                if a < 1.0 {
                    scale
                } else {
                    // A reciprocal. The original spells it `exp(-0.5*beta2*log(a))`
                    // with `beta2 = 2`, so the exponent is exactly -1.
                    scale / a.powf(Self::FRANKEL_EXPONENT / 2.0)
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
    min_squared: f64,
    max_squared: f64,
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
    pub fn new(min_wavelength_km: f64, max_wavelength_km: f64) -> Self {
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
    fn divisor(self, k2: f64) -> f64 {
        self.divisor_at_order(k2, BAND_PASS_ORDER)
    }

    /// As [`divisor`](Self::divisor), but at a caller-chosen roll-off order.
    ///
    /// [`band_pass`] takes its order from configuration; the generators do not.
    fn divisor_at_order(self, k2: f64, order: i32) -> f64 {
        // Both products are single precision before they reach the logarithm: the
        // original multiplies two floats and the widening happens at the call. The
        // rounding differs if the operands are widened first.
        let high = k2 * self.min_squared;
        let low = k2 * self.max_squared;
        // Integer powers: the order is 4 in the generators and `kord` -- an integer
        // getpar -- in `band_pass`. Three multiplies each rather than a pair of
        // transcendental calls.
        //
        // `powi` keeps the DC behaviour the original relies on. At the origin `low`
        // is zero, so `1/low.powi(order)` is infinity and the gain is `1/inf = 0`:
        // the band-pass removes the mean through IEEE arithmetic rather than a guard,
        // which is `DEFECTS.md` 4 and is pinned by `contracts.rs`.
        let high_cut = 1.0 + high.powi(order);
        let low_cut = 1.0 + 1.0 / low.powi(order);
        high_cut * low_cut
    }
}

/// Wavenumber sample spacing of a grid, in radians per kilometre.
#[derive(Clone, Copy, Debug)]
pub struct WavenumberStep {
    pub strike: f64,
    pub dip: f64,
}

/// Correlation lengths of the slip field, in kilometres.
#[derive(Clone, Copy, Debug)]
pub struct CorrelationLengths {
    pub strike: f64,
    pub dip: f64,
}

/// The half of a grid the generators fill, row by row, with each row's wavenumber.
///
/// Every kernel below walks `0..=dip_count/2` — the non-negative dip half — because
/// the rest is filled by [`crate::grid::impose_hermitian_symmetry`] afterwards. The
/// dip wavenumber is constant along a row and the original recomputes it at every
/// point; yielding it *with* the row is what hoists it, and what leaves each kernel's
/// body about its filter rather than about indexing.
fn filled_rows(
    spectrum: &mut Spectrum,
    step: WavenumberStep,
) -> impl Iterator<Item = (f64, ndarray::ArrayViewMut1<'_, Complex64>)> {
    let dip_count = spectrum.dip_count();
    spectrum
        .rows_mut()
        .into_iter()
        .enumerate()
        .take(dip_count / 2 + 1)
        .map(move |(dip, row)| (wavenumber(dip, dip_count, step.dip), row))
}

/// Signed wavenumber of grid index `index` in an extent of `count` samples.
///
/// Indices past the midpoint represent negative wavenumbers, as in any discrete
/// transform.
fn wavenumber(index: usize, count: usize, step: f64) -> f64 {
    let signed = if index <= count / 2 {
        units::exact(index)
    } else {
        units::exact(index) - units::exact(count)
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
    amplitude_scale: f64,
) {
    let strike_count = spectrum.strike_count();
    let _dip_count = spectrum.dip_count();

    let strike_correlation_squared = correlation.strike * correlation.strike;
    let dip_correlation_squared = correlation.dip * correlation.dip;

    for (ky, mut row) in filled_rows(spectrum, step) {
        for (strike, value) in row.iter_mut().enumerate() {
            let kx = wavenumber(strike, strike_count, step.strike);

            let normalised =
                kx * kx * strike_correlation_squared + ky * ky * dip_correlation_squared;
            let mut amplitude = shape.amplitude(normalised, amplitude_scale);

            let k2 = kx * kx + ky * ky;
            if k2 > 0.0 {
                amplitude /= band.divisor(k2);
            }

            *value = draw_unit_complex(source) * amplitude;
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
    let _dip_count = spectrum.dip_count();

    for (ky, mut row) in filled_rows(spectrum, step) {
        for (strike, value) in row.iter_mut().enumerate() {
            let kx = wavenumber(strike, strike_count, step.strike);
            let k2 = kx * kx + ky * ky;
            *value *= 1.0 / band.divisor_at_order(k2, order);
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
    hurst_exponent: f64,
    step: WavenumberStep,
    band: WavelengthBand,
) {
    let strike_count = spectrum.strike_count();
    let _dip_count = spectrum.dip_count();

    let falloff = hurst_exponent + 1.0;

    // Hoisted: it depends only on the Hurst exponent, and the original recomputed it
    // at every grid point.
    let exponent = -0.5 * falloff;

    for (dip, (ky, mut row)) in filled_rows(spectrum, step).enumerate() {
        for (strike, value) in row.iter_mut().enumerate() {
            let kx = wavenumber(strike, strike_count, step.strike);

            // Drawn before the amplitude is known, and drawn at the origin too,
            // where the amplitude is zero. The draw count depends on the extent
            // alone.
            let deviate = draw_unit_complex(source);

            let amplitude = if strike == 0 && dip == 0 {
                0.0
            } else {
                let k2 = kx * kx + ky * ky;
                k2.powf(exponent) / band.divisor(k2)
            };

            *value = deviate * amplitude;
        }
    }

    // The origin is zeroed outright rather than merely made real: a self-affine
    // field has no mean, and the power law diverges there. The other three points
    // keep their real part unscaled -- unlike the correlated field, this one does
    // not restore the quadrature variance.
    spectrum[[0, 0]] = Complex64::default();
    make_corners_real(spectrum, |value| value);
    impose_hermitian_symmetry(spectrum);
}

/// One complex deviate with unit total variance.
fn draw_unit_complex<S: DrawSource>(source: &mut S) -> Complex64 {
    // `gaussian` returns f64 and the product with the f64 norm happens in f64,
    // narrowing once on store -- the original's `fnorm*gaus_rand(...)` into a float.
    let real = QUADRATURE_NORM * source.gaussian(1.0, 0.0);
    let imaginary = QUADRATURE_NORM * source.gaussian(1.0, 0.0);
    Complex64::new(real, imaginary)
}

/// Force the four self-conjugate points to be real.
///
/// DC and the three Nyquist corners map to themselves under conjugation, so they
/// cannot carry an imaginary part in a real-valued field.
fn make_corners_real(spectrum: &mut Spectrum, rescale: impl Fn(f64) -> f64) {
    let strike_nyquist = spectrum.strike_count() / 2;
    let dip_nyquist = spectrum.dip_count() / 2;

    // `[dip, strike]`, per `grid.rs`. The four corners are the wavenumbers that are
    // their own conjugate, so they have to be real for the inverse transform to be.
    for corner in [
        [0, 0],
        [0, strike_nyquist],
        [dip_nyquist, 0],
        [dip_nyquist, strike_nyquist],
    ] {
        spectrum[corner] = Complex64::new(rescale(spectrum[corner].re), 0.0);
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

    // The original writes pi as `4.0*atan(1.0)`, which is exactly this constant:
    // `atan(1)` is correctly rounded to pi/4 and scaling by a power of two is exact.
    // See `tests/float_identities.rs`.
    let strike_argument = 2.0 * std::f64::consts::PI * strike_shift;
    let dip_argument = 2.0 * std::f64::consts::PI * dip_shift;

    // A hypotenuse, which cannot overflow the way squaring both parts can. The
    // magnitude is what makes a negative mean come back positive -- `DEFECTS.md` 3,
    // pinned in `contracts.rs` -- and that is unchanged.
    let dc_magnitude = spectrum[[0, 0]].norm();

    for (ky, mut row) in filled_rows(spectrum, step) {
        let dip_phase = phase_factor(dip_argument * ky);
        for (strike, value) in row.iter_mut().enumerate() {
            let strike_phase =
                phase_factor(strike_argument * wavenumber(strike, strike_count, step.strike));

            // Two separate multiplications rather than one combined factor, so the
            // rounding matches: the original applies the along-strike rotation,
            // stores the result, then applies the down-dip one.
            *value = *value * strike_phase * dip_phase;
        }
    }

    spectrum[[0, 0]] = Complex64::new(dc_magnitude, 0.0);
    // Unlike the generators, only the imaginary parts of the remaining three
    // self-conjugate points are cleared; their real parts keep whatever the rotation
    // gave them.
    let strike_nyquist = strike_count / 2;
    let dip_nyquist = dip_count / 2;
    for corner in [
        [0, strike_nyquist],
        [dip_nyquist, 0],
        [dip_nyquist, strike_nyquist],
    ] {
        spectrum[corner] = Complex64::new(spectrum[corner].re, 0.0);
    }

    impose_hermitian_symmetry(spectrum);
}

/// `exp(-i * argument)`, narrowed to single precision as the original stores it.
fn phase_factor(argument: f64) -> Complex64 {
    Complex64::new(argument.cos(), -argument.sin())
}

/// Blend a field with a reference at a target correlation, in the wavenumber domain.
///
/// `target` becomes `rho * reference + sqrt(1 - rho^2) * target`. Because both
/// inputs have unit variance by construction and the weights are `cos`/`sin` of the
/// same angle, the result does too — so this sets the correlation without disturbing
/// the amplitude, which is what makes it composable with the rescaling that follows.
///
/// Done in the wavenumber domain rather than on the fault because the two fields
/// share a spectrum there: blending after the inverse transform would correlate the
/// *values* while leaving the spatial structure of each untouched.
///
/// This is how slip, rupture time and rise time become statistically linked — the
/// mechanism behind `tsfac1_scor` and `rtime1_scor`.
///
/// # Panics
///
/// If the two grids have different extents.
///
/// (orig. `genslip_v5.6.2.c:2116-2121`)
pub fn correlate_with(target: &mut Spectrum, reference: &Spectrum, correlation: f64) {
    assert_eq!(
        (target.strike_count(), target.dip_count()),
        (reference.strike_count(), reference.dip_count()),
        "cannot correlate grids of different extents"
    );

    let independent = (1.0 - correlation * correlation).sqrt();

    for (value, other) in target.flat_mut().iter_mut().zip(reference.flat().iter()) {
        *value = *other * correlation + *value * independent;
    }
}
