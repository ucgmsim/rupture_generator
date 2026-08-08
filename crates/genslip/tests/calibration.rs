//! Does the drawn field have the spectrum it was asked for?
//!
//! A red here means **a parameter changed meaning** — not that a value moved, which is
//! `contracts.rs`'s business, and not that the science changed, which is the corpus's.
//! Someone reconfigured what "Mai" or "Frankel" denotes, or a factor went missing from
//! an amplitude, and the field is still a perfectly plausible random field of the
//! wrong kind.
//!
//! # The one test here that is genuinely independent
//!
//! `the_falloff_exponent_is_recovered_from_the_field` **regresses the exponent out of
//! the data** and checks it against the published model — 1.75 for Mai & Beroza's von
//! Kármán at H = 0.75, 2 for Frankel, 1 for Somerville. It is told the *shape* of the
//! model but never the exponent, so it cannot pass by agreeing with a second reading
//! of `field.rs`. That distinction is `ENGINEERING_RULES.md` rule 5, and it is why
//! this test is worth more than the normalisation one below it.
//!
//! # Why the tolerances are formulas
//!
//! Each Fourier coefficient of a Gaussian random field is an independent complex
//! Gaussian, so `|S(k)|²/A(k)²` is exactly `χ²₂/2` — an exponential of mean one — and
//! `ln` of it has variance `π²/6` regardless of wavenumber. Both facts are exact, so
//! both tolerances are computed from the sample rather than chosen: see
//! `common::tolerance`, where each also carries its detection floor.
//!
//! # The band-pass is avoided rather than modelled
//!
//! Every generator multiplies its amplitude by a band-pass, and reproducing that here
//! would be a second reading of the code under test. Instead the fit is restricted to
//! wavenumbers where the band-pass gain is within 1e-3 of one — outside the roll-off
//! at both ends — and `the_fitted_band_is_actually_flat` asserts that restriction is
//! doing its job. Fitting where a factor is negligible is not the same as pretending
//! it is absent.

mod common;

use common::fixture;
use common::stats;
use common::tolerance::{Z, log_spectrum_slope_error, wilson_hilferty_band};
use genslip::field::{
    CorrelationLengths, Spectrum2D, WavelengthBand, WavenumberStep, correlated_field,
};
use genslip::grid::{FaultAxes, Spectrum};
use genslip::rng::{DrawSource, GenslipLcg, Pcg};

/// Big enough that the flat part of the band holds thousands of coefficients.
const CELLS: usize = 128;
/// Kilometres. With `CELLS` this puts wavenumbers on `[1/128, 0.5]` cycles/km.
const SPACING_KM: f64 = 1.0;

const MIN_WAVELENGTH_KM: f32 = 1.5;
const MAX_WAVELENGTH_KM: f32 = 80.0;

/// Correlation lengths, deliberately unequal.
///
/// An isotropic pair would make `a` a function of `|k|` alone, and a fit against
/// `log k` would then look correct even though it is the wrong regressor on any real
/// fault. Keeping them apart is what makes the regressor choice matter.
const CORNER_STRIKE_KM: f32 = 12.0;
const CORNER_DIP_KM: f32 = 5.0;

/// Wavenumber increment, `1/(N·d)`, in cycles per kilometre.
fn increment() -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "grid extents are far below 2^52"
    )]
    let cells = CELLS as f64;
    1.0 / (cells * SPACING_KM)
}

fn step() -> WavenumberStep {
    #[expect(clippy::cast_possible_truncation, reason = "the API is f32")]
    let value = increment() as f32;
    WavenumberStep {
        strike: value,
        dip: value,
    }
}

/// The signed wavenumber at an index, in cycles per kilometre.
///
/// The standard FFT convention — indices past the midpoint are negative frequencies —
/// not a property of this crate.
fn wavenumber(index: usize, count: usize, step: f64) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "grid extents are small")]
    let signed = if index <= count / 2 {
        index as f64
    } else {
        index as f64 - count as f64
    };
    signed * step
}

/// One drawn spectrum.
fn drawn<S: DrawSource>(source: &mut S, shape: Spectrum2D) -> Spectrum {
    let mut spectrum = genslip::grid::spectrum(CELLS, CELLS);
    correlated_field(
        &mut spectrum,
        source,
        shape,
        step(),
        CorrelationLengths {
            strike: CORNER_STRIKE_KM,
            dip: CORNER_DIP_KM,
        },
        WavelengthBand::new(MIN_WAVELENGTH_KM, MAX_WAVELENGTH_KM),
        1.0,
    );
    spectrum
}

/// Squared wavenumber normalised by the corner: the argument every shape is a
/// function of.
fn corner_argument(strike: usize, dip: usize) -> f64 {
    let increment = increment();
    let kx = wavenumber(strike, CELLS, increment) * f64::from(CORNER_STRIKE_KM);
    let ky = wavenumber(dip, CELLS, increment) * f64::from(CORNER_DIP_KM);
    kx * kx + ky * ky
}

/// The band-pass gain at a wavenumber, order four at both ends.
///
/// Used **only** to decide which coefficients to fit, never to correct one. Getting
/// it slightly wrong would shrink or grow the fitted band and nothing else.
fn band_gain(strike: usize, dip: usize) -> f64 {
    let increment = increment();
    let kx = wavenumber(strike, CELLS, increment);
    let ky = wavenumber(dip, CELLS, increment);
    let squared = kx * kx + ky * ky;
    if squared == 0.0 {
        return 0.0;
    }
    let high = 1.0 + (squared * f64::from(MIN_WAVELENGTH_KM).powi(2)).powi(4);
    let low = 1.0 + (squared * f64::from(MAX_WAVELENGTH_KM).powi(2)).powi(-4);
    1.0 / (high * low)
}

/// Coefficients in the flat interior of the band, excluding the self-conjugate points.
///
/// The four self-conjugate points — DC and the three Nyquist corners — are forced real
/// by Hermitian symmetry, so they carry one degree of freedom rather than two and
/// pooling them with the rest would bias every statistic here. Excluded outright
/// rather than weighted, because there are four of them and thousands of the others.
fn fitted_coefficients() -> Vec<(usize, usize)> {
    (0..=CELLS / 2)
        .flat_map(|dip| (0..CELLS).map(move |strike| (strike, dip)))
        .filter(|&(strike, dip)| {
            let self_conjugate =
                (strike == 0 || strike == CELLS / 2) && (dip == 0 || dip == CELLS / 2);
            !self_conjugate && (band_gain(strike, dip) - 1.0).abs() < 1e-3
        })
        .collect()
}

#[test]
fn the_fitted_band_is_actually_flat() {
    // The restriction the two tests below rest on: inside this set the band-pass is a
    // factor of one to within 1e-3, so leaving it out of the model costs nothing. If
    // the band ever narrowed to a handful of coefficients the tests would silently
    // lose their power, so the count is asserted too.
    let fitted = fitted_coefficients();
    assert!(
        fitted.len() > 1000,
        "only {} coefficients survive the band restriction; the fits below would be \
         underpowered",
        fitted.len()
    );
    for &(strike, dip) in &fitted {
        assert!((band_gain(strike, dip) - 1.0).abs() < 1e-3);
    }
}

/// The falloff exponent, recovered from the field rather than assumed.
///
/// `|S(k)|² = A(k)²·R` with `R` exponential of mean one, so
/// `ln|S|² = 2 ln A + ln R`. For each shape `2 ln A` is a *linear* function of that
/// shape's own argument, and the slope is the model's exponent:
///
/// | shape | regressor | slope |
/// | --- | --- | --- |
/// | Mai, Suzuki (von Kármán) | `ln(1 + a)` | −1.75 = −(H+1), H = 0.75 |
/// | Somerville | `ln(1 + a²)` | −1 |
/// | Frankel, above the corner | `ln a` | −2 |
///
/// Regressed on the shape's **own** argument rather than on `log k`. With unequal
/// correlation lengths `a` depends on direction, so `log k` is a mis-specified
/// regressor: the anisotropy enters as structured residual, biasing the slope and
/// inflating its error. That mistake would still produce a plausible-looking fit.
///
/// `ln R` has variance `π²/6` at every wavenumber, so the fit is homoscedastic and
/// ordinary least squares is the right estimator — and the tolerance is that variance
/// over the design matrix's spread, computed rather than chosen.
#[test]
fn the_falloff_exponent_is_recovered_from_the_field() {
    for (shape, expected, regressor) in [
        (Spectrum2D::Mai, -1.75_f64, "ln(1+a)"),
        (Spectrum2D::Suzuki, -1.75, "ln(1+a)"),
        (Spectrum2D::Somerville, -1.0, "ln(1+a^2)"),
        (Spectrum2D::Frankel, -2.0, "ln a, above the corner"),
    ] {
        let mut source = GenslipLcg::new(20_260_807);
        let fitted = fitted_coefficients();

        let mut points: Vec<(f64, f64)> = Vec::new();
        for _ in 0..4 {
            let spectrum = drawn(&mut source, shape);
            for &(strike, dip) in &fitted {
                let a = corner_argument(strike, dip);
                let regress_on = match shape {
                    Spectrum2D::Somerville | Spectrum2D::InputCorners => (1.0 + a * a).ln(),
                    Spectrum2D::Frankel => {
                        // Below the corner the shape is flat by construction, which
                        // is a different claim -- `the_frankel_shape_is_flat_below_
                        // its_corner` makes it.
                        if a < 2.0 {
                            continue;
                        }
                        a.ln()
                    }
                    _ => (1.0 + a).ln(),
                };
                let power = f64::from(spectrum[[dip, strike]].norm_sqr());
                if power > 0.0 {
                    points.push((regress_on, power.ln()));
                }
            }
        }

        #[expect(clippy::cast_precision_loss, reason = "coefficient counts are small")]
        let count = points.len() as f64;
        let mean_x = points.iter().map(|(x, _)| x).sum::<f64>() / count;
        let mean_y = points.iter().map(|(_, y)| y).sum::<f64>() / count;
        let spread: f64 = points.iter().map(|(x, _)| (x - mean_x).powi(2)).sum();
        let covariance: f64 = points
            .iter()
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum();
        let slope = covariance / spread;

        let tolerance = Z * log_spectrum_slope_error(spread);
        println!(
            "{shape:?}: exponent {slope:.4} against {expected} (on {regressor}, \
             {} coefficients, tolerance {tolerance:.4})",
            points.len()
        );
        assert!(
            (slope - expected).abs() < tolerance,
            "{shape:?}: recovered exponent {slope:.4}, model says {expected}"
        );
    }
}

/// Frankel is flat below its corner, and that is a different claim from its slope.
///
/// The shape is `A = scale` for `a < 1` and `scale/a` above — a hard corner rather
/// than a smooth roll-off, and the only shape here with one. A fit that spanned the
/// corner would recover some average of the two and pass a loose bound while getting
/// both halves wrong.
#[test]
fn the_frankel_shape_is_flat_below_its_corner() {
    let mut source = GenslipLcg::new(4242);
    let mut below: Vec<f64> = Vec::new();

    for _ in 0..4 {
        let spectrum = drawn(&mut source, Spectrum2D::Frankel);
        for &(strike, dip) in &fitted_coefficients() {
            if corner_argument(strike, dip) < 0.5 {
                let power = f64::from(spectrum[[dip, strike]].norm_sqr());
                if power > 0.0 {
                    below.push(power);
                }
            }
        }
    }

    assert!(
        below.len() > 30,
        "only {} coefficients below the corner; nothing to conclude",
        below.len()
    );

    // Flat means the pooled power ratio against a constant amplitude is one. `scale`
    // is 1 here, so the expected power is 1 and the ratio is the power itself.
    let (low, high) = wilson_hilferty_band(2 * below.len(), Z);
    #[expect(clippy::cast_precision_loss, reason = "coefficient counts are small")]
    let mean = below.iter().sum::<f64>() / below.len() as f64;
    assert!(
        (low..=high).contains(&mean.cbrt()),
        "power below the corner is {mean:.4}, outside [{low:.4}, {high:.4}] cubed"
    );
}

/// The realised power is the amplitude the shape prescribes, not merely proportional.
///
/// `|S|²/A²` is exactly `χ²₂/2` per coefficient, so pooling `M` of them gives
/// `χ²_{2M}/2M`, and Wilson–Hilferty turns that into a band on the cube root. This is
/// the weaker of the two spectral tests — it needs `A` written out, so a shared
/// misreading of the amplitude would pass it — but it catches what the slope cannot: a
/// missing `1/√2`, a band-pass applied twice, an amplitude scale dropped.
///
/// Run against **both** draw sources. The compatibility generator's normals are
/// Irwin–Hall rather than Gaussian, bounded at ±6σ, so the ratio's variance is 0.952
/// rather than 1 — the band is about 2% conservative for it, which is the right
/// direction and worth knowing rather than correcting.
#[test]
fn the_realised_power_matches_the_prescribed_amplitude() {
    fn check<S: DrawSource>(source: &mut S, label: &str) {
        let fitted = fitted_coefficients();
        let mut ratios: Vec<f64> = Vec::new();

        for _ in 0..4 {
            let spectrum = drawn(source, Spectrum2D::Mai);
            for &(strike, dip) in &fitted {
                // von Kármán, from Mai & Beroza: A = scale / sqrt((1+a)^1.75).
                let a = corner_argument(strike, dip);
                let amplitude_squared = (1.0 + a).powf(-1.75);
                let power = f64::from(spectrum[[dip, strike]].norm_sqr());
                ratios.push(power / amplitude_squared);
            }
        }

        #[expect(clippy::cast_precision_loss, reason = "coefficient counts are small")]
        let count = ratios.len() as f64;
        let mean = ratios.iter().sum::<f64>() / count;

        // Two degrees of freedom per coefficient: a real and an imaginary Gaussian.
        let (low, high) = wilson_hilferty_band(2 * ratios.len(), Z);
        println!(
            "{label}: pooled power ratio {mean:.5} over {} coefficients, band \
             [{:.5}, {:.5}] on the cube root",
            ratios.len(),
            low,
            high
        );
        assert!(
            (low..=high).contains(&mean.cbrt()),
            "{label}: pooled ratio {mean:.5} is outside its band"
        );
    }

    check(&mut GenslipLcg::new(909), "GenslipLcg");
    check(&mut Pcg::new(909), "Pcg");
}

/// The fault's mean rise time is the one the magnitude implies.
///
/// This is what `rise_time_normalisation` exists for, and nothing else asserted it.
/// The chain: the moment gives an average rise time through Graves & Pitarka's
/// relation `trise = c · 1e-9 · M0^(1/3)`, corrected by the dip-and-rake factor
/// `alphaT`; the rise-time *field* is dimensionless with unit mean; and the
/// normalisation constant is chosen so that scaling the one by the other lands the
/// fault-wide weighted average on the first. If the constant were computed over the
/// wrong weights — or over the wrong field, which is `DEFECTS.md` 15 — every rise time
/// would be off by a single factor and every pulse would be the wrong duration, while
/// the field's *shape* stayed perfectly plausible.
///
/// Held loosely on purpose. `rise_times` floors each subfault at one sample interval,
/// so subfaults whose rise time would be shorter are raised and the realised mean sits
/// a little above the target. That floor is physics — a pulse shorter than a sample
/// cannot be represented — not an error to tighten away.
#[test]
fn the_mean_rise_time_is_what_the_moment_implies() {
    let model = genslip::realisation::generate(
        &mut GenslipLcg::new(fixture::SEED),
        &mut genslip::fft::RustFft::new(),
        &mut genslip::rupture::FactoredSweep::new(),
        &fixture::fault(),
        &fixture::velocity_model(),
        fixture::source_spec(),
        fixture::slip_spec(),
        fixture::timing_spec(),
        fixture::hypocentre(),
    )
    .expect("the fixture geometry is valid");

    // Computed from the published relation, NOT by calling `average_rise_time`.
    // Calling it made this test a tautology: scaling that function by 1.15 scaled the
    // expectation by 1.15 too, and the mutation passed. Graves & Pitarka give
    // `trise = c · 1e-9 · M0^(1/3)`, and writing it out here is the only way the test
    // can disagree with the code.
    let source = fixture::source_spec();
    let expected = f64::from(source.rise_time_coefficient)
        * 1.0e-9
        * f64::from(model.moment_dyne_cm).cbrt()
        * f64::from(model.alpha_t);

    let realised = stats::mean(&stats::widen(model.rise_time_s.flat()));
    println!("mean rise time {realised:.5} s against {expected:.5} s implied by M0");

    assert!(
        (realised - expected).abs() < 0.05 * expected,
        "mean rise time {realised:.5} s, magnitude implies {expected:.5} s"
    );

    // And the correction is actually applied: alphaT is below one for this geometry,
    // so an implementation that dropped it would land 0.4% high and still pass a
    // bound of 5%. Asserting it is in range is what stops that being invisible.
    assert!(
        (1.0 / 1.1..1.0).contains(&f64::from(model.alpha_t)),
        "alpha_t is {}, outside the range its definition allows",
        model.alpha_t
    );
}
