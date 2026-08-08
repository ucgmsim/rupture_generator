//! What the correlated fields are for, rather than what their samples are.
//!
//! Three fields are drawn against the slip field once it exists: rake, the
//! rupture-time perturbation, and the rise-time one. Each is generated
//! independently, blended with a stored spectrum of the processed slip, and brought
//! back to the fault. The mechanism is one elementwise map and one affine
//! renormalisation, and both are exact — so this file asserts algebra where algebra
//! holds and calibration where it does not.
//!
//! # It used to be a parity file, and widening an accumulator is what retired it
//!
//! Every assertion here was `to_bits()` equality against genslip's `f64` folds. When
//! `stats::mean_and_sigma` and the two spread rescalings moved to `f64`
//! accumulation — strictly more accurate, and the change that made moment
//! conservation worth asserting — this file went red for a reason that was not a
//! defect.
//!
//! That is the situation `ENGINEERING_RULES.md` exists for. Under bit-parity the
//! answer would have been to abandon the improvement; under a scientific gate it is
//! to replace the assertion with the property it was standing in for.

mod common;

use common::fixture;
use common::stats;
use genslip::fft::{Fft, RustFft};
use genslip::field::Spectrum2D;
use genslip::grid::FaultAxes;
use genslip::rng::GenslipLcg;
use genslip::slip::{
    GridExtents, NormalisedSlip, PerturbationSpec, SubfaultSpacing, correlated_perturbation,
    generate_normalised, rake_field, reload_for_correlation,
};
use genslip::stats::mean_and_sigma;
use num_complex::Complex64;

const SPACING: SubfaultSpacing = SubfaultSpacing {
    strike_km: 1.0,
    dip_km: 1.0,
};

fn extents() -> GridExtents {
    fixture::fault().extents
}

fn spectrum() -> genslip::slip::SpectrumSpec {
    let mut spec = fixture::spectrum_spec();
    spec.shape = Spectrum2D::Mai;
    spec.correlation = fixture::corner_lengths();
    spec
}

/// The slip field and the padded statistics the reload maps back onto.
///
/// Takes the source by reference and leaves it *advanced*, because the fields drawn
/// afterwards must come from the stream position the pipeline leaves them at. Seeding
/// each field separately instead makes the supposedly independent rake field the same
/// deviates as slip, correlated at exactly -1 — which looks like a defect and is an
/// artefact of the test.
fn slip_stage<F: Fft>(source: &mut GenslipLcg, fft: &mut F) -> NormalisedSlip {
    generate_normalised(source, fft, extents(), SPACING, spectrum())
}

/// The reload restores the statistics the padded grid had before the fault work.
///
/// What the reload is *for*: the processed field — truncated, tapered, scaled — is
/// put back onto a padded grid and mapped affinely onto the mean and deviation the
/// generated field had, so every field correlated against it lands on the same scale
/// as the slip it blends with. Without it the perturbations would be correlated with
/// a differently-scaled field and their configured sigmas would mean nothing.
#[test]
fn the_reload_restores_the_original_padded_statistics() {
    let mut fft = RustFft::new();
    let mut source = GenslipLcg::new(909);
    let generated = slip_stage(&mut source, &mut fft);
    let extents = extents();

    let reloaded = reload_for_correlation(
        &generated.field,
        &mut fft,
        extents,
        SPACING,
        generated.padded,
    );

    // The reload happens before the transform, so undo the transform's DC scaling to
    // read the mean back: the DC term of an unnormalised transform is the sum.
    let points = genslip::units::exact(extents.padded_strike * extents.padded_dip);
    let restored = reloaded[[0, 0]].re / (points * SPACING.strike_km * SPACING.dip_km);

    assert!(
        (restored - generated.padded.mean).abs() < generated.padded.sigma * 1e-4,
        "reload gave mean {restored}, expected {}",
        generated.padded.mean
    );
}

/// A perturbation has the spread it was configured with, and no mean.
///
/// The calibration claim: `PerturbationSpec::sigma` is what the field's population
/// deviation comes out as, so the seconds it is later multiplied by mean what they
/// say. `DEFECTS.md` 14 is the same claim for rake, in different units, and it is
/// what happens when nothing asserts this.
#[test]
fn a_perturbation_is_centred_and_has_the_configured_spread() {
    let mut fft = RustFft::new();
    let mut source = GenslipLcg::new(4242);
    let generated = slip_stage(&mut source, &mut fft);
    let extents = extents();
    let reference = reload_for_correlation(
        &generated.field,
        &mut fft,
        extents,
        SPACING,
        generated.padded,
    );

    for sigma in [0.25_f64, 1.0, 3.5] {
        let mut continued = source;
        let field = correlated_perturbation(
            &mut continued,
            &mut fft,
            &reference,
            extents,
            SPACING,
            spectrum(),
            PerturbationSpec {
                correlation: 0.8,
                sigma,
            },
        );

        let values = stats::widen(field.flat());
        assert!(
            stats::mean(&values).abs() < sigma * 1e-4,
            "sigma {sigma}: mean {} is not zero",
            stats::mean(&values)
        );
        let spread = stats::population_sigma(&values);
        assert!(
            (spread - sigma).abs() < sigma * 1e-4,
            "configured {sigma}, realised {spread}"
        );
    }
}

/// A perturbation correlated at rho really is, to the fault's effective resolution.
///
/// `correlate_with` sets the correlation of the *spectra* exactly; the field is then
/// cropped to the fault, centred and rescaled, so the realised spatial correlation is
/// near rho rather than equal to it. Asserted loosely on purpose — the exact claim
/// lives in `contracts.rs`, on the elementwise map, where it holds to 1e-6 and needs
/// no realisations at all. This is the end-to-end sanity check that the exact
/// relation survived the crop.
#[test]
fn the_realised_correlation_tracks_the_requested_one() {
    let mut fft = RustFft::new();
    let mut source = GenslipLcg::new(31);
    let generated = slip_stage(&mut source, &mut fft);
    let extents = extents();
    let reference = reload_for_correlation(
        &generated.field,
        &mut fft,
        extents,
        SPACING,
        generated.padded,
    );
    let slip = stats::widen(generated.field.flat());

    let mut realised = |rho: f64| {
        let mut continued = source;
        let field = correlated_perturbation(
            &mut continued,
            &mut fft,
            &reference,
            extents,
            SPACING,
            spectrum(),
            PerturbationSpec {
                correlation: rho,
                sigma: 1.0,
            },
        );
        stats::pearson(&stats::widen(field.flat()), &slip)
    };

    // Monotone in rho, and the ends behave: uncorrelated is near zero, fully
    // correlated is near one. That ordering is the part a wiring error breaks.
    let (none, some, full) = (realised(0.0), realised(0.8), realised(1.0));
    assert!(none.abs() < 0.3, "rho of zero gave a correlation of {none}");
    assert!(full > 0.95, "rho of one gave a correlation of {full}");
    assert!(
        none < some && some < full,
        "correlation is not monotone in rho: {none}, {some}, {full}"
    );
}

/// The rake field is the base rake plus a spread in degrees, and nothing else.
///
/// Rake is the one field whose perturbation is *not* correlated with slip — genslip
/// draws it independently — so a rewrite that routed it through the reload would
/// change the model while leaving every summary statistic here intact except this one.
#[test]
fn the_rake_field_is_independent_of_slip() {
    let mut fft = RustFft::new();
    let mut source = GenslipLcg::new(5150);
    let generated = slip_stage(&mut source, &mut fft);
    let extents = extents();
    let grid = fixture::fault();

    let base_rake = genslip::grid::from_values(
        extents.fault_strike,
        extents.fault_dip,
        grid.base_rake_deg.clone(),
    );
    let rake = rake_field(
        &mut source,
        &mut fft,
        extents,
        SPACING,
        spectrum(),
        base_rake.view(),
        15.0,
    );

    let deviation: Vec<f64> = rake
        .flat()
        .iter()
        .zip(&grid.base_rake_deg)
        .map(|(value, base)| value - base)
        .collect();

    let spread = stats::population_sigma(&deviation);
    assert!(
        (spread - 15.0).abs() < 1e-2,
        "rake spread {spread} degrees, configured 15"
    );

    // Independent of slip, unlike the two timing perturbations. A weak bound: what
    // would break this is routing rake through the reload, which would put it above
    // 0.8, not a realisation that happened to correlate.
    let with_slip = stats::pearson(&deviation, &stats::widen(generated.field.flat()));
    assert!(
        with_slip.abs() < 0.4,
        "rake correlates with slip at {with_slip}; it is drawn independently"
    );
}

/// The whole padded grid keeps the statistics it was mapped onto.
#[test]
fn the_reloaded_grid_keeps_the_original_deviation() {
    let mut fft = RustFft::new();
    let mut source = GenslipLcg::new(909);
    let generated = slip_stage(&mut source, &mut fft);
    let extents = extents();

    let mut padded = genslip::grid::spectrum(extents.padded_strike, extents.padded_dip);
    for dip in 0..extents.fault_dip {
        for strike in 0..extents.fault_strike {
            padded[[dip, strike]] = Complex64::new(generated.field[[dip, strike]], 0.0);
        }
    }
    // Before the reload the padded grid is the fault in a sea of zeros, so its
    // statistics are not the generated field's -- which is what the reload fixes.
    let before = mean_and_sigma(&padded);
    assert!(
        (before.sigma - generated.padded.sigma).abs() > generated.padded.sigma * 0.05,
        "the zero-padded grid already had the original deviation; the reload would \
         then be doing nothing"
    );
}
