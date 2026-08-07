//! The rake and perturbation fields reproduce genslip's, stage for stage.
//!
//! Same shape of test as `slip_pipeline.rs`, and for the same reason: these blocks
//! are inline in `main`, so the reference is built from oracle calls made in the
//! order `main` makes them.
//!
//! What is new here is the correlation step. `tsfac1` and `rtime1` are blended with
//! the slip spectrum before the inverse transform, which is what makes rupture time
//! and rise time vary with slip. That blend happens against a *reloaded* slip field —
//! one that has been truncated, tapered, put back on a padded grid and transformed
//! again — so getting the reference right means reproducing that reload too.
#![cfg(feature = "fftw")]

use genslip::fft::FftwFft;
use genslip::field::{CorrelationLengths, Spectrum2D, WavelengthBand};
use genslip::rng::GenslipLcg;
use genslip::slip::{
    GridExtents, PerturbationSpec, SpectrumSpec, SubfaultSpacing, correlated_perturbation,
    generate_normalised, rake_field, reload_for_correlation, truncate_negative_slip,
};
use genslip::stats::mean_and_sigma;
use genslip::taper::{EdgeTapers, SlipField, taper_edges};
use genslip_oracle::{Complex, field as oracle};
use proptest::prelude::*;

const CASES: [GridExtents; 4] = [
    GridExtents {
        fault_strike: 2,
        fault_dip: 2,
        padded_strike: 4,
        padded_dip: 4,
    },
    GridExtents {
        fault_strike: 10,
        fault_dip: 6,
        padded_strike: 12,
        padded_dip: 8,
    },
    GridExtents {
        fault_strike: 32,
        fault_dip: 12,
        padded_strike: 36,
        padded_dip: 14,
    },
    GridExtents {
        fault_strike: 24,
        fault_dip: 24,
        padded_strike: 28,
        padded_dip: 28,
    },
];

const SPACING: SubfaultSpacing = SubfaultSpacing {
    strike_km: 2.0,
    dip_km: 1.5,
};
const CORRELATION: CorrelationLengths = CorrelationLengths {
    strike: 12.0,
    dip: 6.0,
};
const MIN_WAVELENGTH: f32 = 1.5;
const MAX_WAVELENGTH: f32 = 80.0;

fn spec() -> SpectrumSpec {
    SpectrumSpec {
        shape: Spectrum2D::Mai,
        correlation: CORRELATION,
        band: WavelengthBand::new(MIN_WAVELENGTH, MAX_WAVELENGTH),
        coefficient_of_variation: 0.75,
        phase_shift: (0.0, 0.0),
    }
}

/// Steps shared by every field: a padded grid of ones, forward-transformed.
fn unit_grid(extents: GridExtents) -> Vec<Complex> {
    let mut grid = vec![Complex { re: 1.0, im: 0.0 }; extents.padded_strike * extents.padded_dip];
    oracle::transform_2d(
        &mut grid,
        extents.padded_strike,
        extents.padded_dip,
        -1,
        SPACING.strike_km,
        SPACING.dip_km,
    );
    grid
}

#[expect(
    clippy::cast_precision_loss,
    reason = "mirrors the port's casts exactly"
)]
fn steps(extents: GridExtents) -> (f32, f32) {
    (
        1.0 / (extents.padded_strike as f32 * SPACING.strike_km),
        1.0 / (extents.padded_dip as f32 * SPACING.dip_km),
    )
}

/// The fault's corner of a padded grid, real parts only.
fn corner(grid: &[Complex], extents: GridExtents) -> Vec<f32> {
    let mut field = Vec::with_capacity(extents.fault_strike * extents.fault_dip);
    for dip in 0..extents.fault_dip {
        for strike in 0..extents.fault_strike {
            field.push(grid[strike + dip * extents.padded_strike].re);
        }
    }
    field
}

#[expect(
    clippy::cast_precision_loss,
    reason = "mirrors the port's casts exactly"
)]
fn remove_mean(field: &mut [f32]) {
    let count = field.len() as f32;
    let mut total = 0.0_f32;
    for value in field.iter() {
        total += *value;
    }
    let mean = total / count;
    for value in field.iter_mut() {
        *value -= mean;
    }
}

#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "mirrors the port's casts and narrowing seams exactly"
)]
fn population_sigma(field: &[f32]) -> f32 {
    let count = field.len() as f32;
    let mut sum_of_squares = 0.0_f32;
    for value in field {
        sum_of_squares += *value * *value;
    }
    f64::from(sum_of_squares / count).sqrt() as f32
}

/// A normalised, truncated, tapered slip field plus the padded statistics the
/// reload needs — the state `main` has reached before it generates anything else.
fn slip_stage(extents: GridExtents, seed: i64) -> (SlipField, genslip::stats::MeanAndSigma) {
    let mut source = GenslipLcg::new(seed);
    let mut fft = FftwFft::new();

    // The port's own generator, already pinned by slip_pipeline.rs. Its padded
    // statistics are measured the way `main` measures them: on the padded grid
    // right after the inverse transform.
    let mut grid = unit_grid(extents);
    let (strike_step, dip_step) = steps(extents);
    let mut oracle_seed = seed;
    oracle::correlated_field(
        &mut grid,
        extents.padded_strike,
        extents.padded_dip,
        strike_step,
        dip_step,
        CORRELATION.strike,
        CORRELATION.dip,
        &mut oracle_seed,
        Spectrum2D::Mai as i32,
        MAX_WAVELENGTH,
        MIN_WAVELENGTH,
    );
    oracle::transform_2d(
        &mut grid,
        extents.padded_strike,
        extents.padded_dip,
        1,
        strike_step,
        dip_step,
    );
    let (mean, sigma) =
        oracle::mean_and_sigma(&mut grid, extents.padded_strike, extents.padded_dip);
    let original = genslip::stats::MeanAndSigma { mean, sigma };

    let mut slip = generate_normalised(&mut source, &mut fft, extents, SPACING, spec()).field;
    truncate_negative_slip(&mut slip);
    taper_edges(
        &mut slip,
        &EdgeTapers {
            sides: 0.02,
            top: 0.0,
            bottom: 0.0,
        },
    );

    (slip, original)
}

/// `main`'s tsfac1 block, rebuilt from oracle calls.
#[expect(
    clippy::cast_possible_truncation,
    reason = "mirrors the port's narrowing seams exactly; that is the point"
)]
fn reference_perturbation(
    extents: GridExtents,
    slip: &SlipField,
    original: genslip::stats::MeanAndSigma,
    seed: i64,
    correlation: f32,
    sigma: f32,
) -> Vec<f32> {
    let (strike_step, dip_step) = steps(extents);
    let points = extents.padded_strike * extents.padded_dip;

    // :1982 -- the reload. Zero-fill, drop the fault in, measure over the WHOLE
    // padded grid, map back onto the original mean and deviation, transform.
    let mut reference = vec![Complex::default(); points];
    for dip in 0..extents.fault_dip {
        for strike in 0..extents.fault_strike {
            reference[strike + dip * extents.padded_strike].re = slip[(strike, dip)];
        }
    }
    let (padded_mean, padded_sigma) =
        oracle::mean_and_sigma(&mut reference, extents.padded_strike, extents.padded_dip);
    let factor = original.sigma / padded_sigma;
    for value in &mut reference {
        value.re = factor * (value.re - padded_mean) + original.mean;
    }
    oracle::transform_2d(
        &mut reference,
        extents.padded_strike,
        extents.padded_dip,
        -1,
        SPACING.strike_km,
        SPACING.dip_km,
    );

    // :2100 -- the perturbation's own field.
    let mut grid = unit_grid(extents);
    let mut oracle_seed = seed;
    oracle::correlated_field(
        &mut grid,
        extents.padded_strike,
        extents.padded_dip,
        strike_step,
        dip_step,
        CORRELATION.strike,
        CORRELATION.dip,
        &mut oracle_seed,
        Spectrum2D::Mai as i32,
        MAX_WAVELENGTH,
        MIN_WAVELENGTH,
    );

    // :2116 -- the blend, in the wavenumber domain.
    let independent = (1.0 - f64::from(correlation * correlation)).sqrt() as f32;
    for (value, other) in grid.iter_mut().zip(&reference) {
        value.re = correlation * other.re + independent * value.re;
        value.im = correlation * other.im + independent * value.im;
    }

    // :2123 -- back to the fault, then centre and rescale.
    oracle::transform_2d(
        &mut grid,
        extents.padded_strike,
        extents.padded_dip,
        1,
        strike_step,
        dip_step,
    );
    let mut field = corner(&grid, extents);
    remove_mean(&mut field);
    let factor = sigma / population_sigma(&field);
    for value in &mut field {
        *value *= factor;
    }
    field
}

fn assert_fields_equal(produced: &SlipField, expected: &[f32], extents: GridExtents, label: &str) {
    for (offset, (got, want)) in produced.as_slice().iter().zip(expected).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{label}: mismatch at (strike {}, dip {}): {got} vs {want}",
            offset % extents.fault_strike,
            offset / extents.fault_strike,
        );
    }
}

#[test]
fn correlated_perturbations_match_across_every_shape() {
    for extents in CASES {
        for (correlation, sigma) in [(0.8_f32, 1.0_f32), (0.9, 0.75), (0.0, 2.0), (1.0, 0.5)] {
            let seed = 20_260_807;
            let (slip, original) = slip_stage(extents, seed);

            let expected =
                reference_perturbation(extents, &slip, original, seed + 1, correlation, sigma);

            let mut source = GenslipLcg::new(seed + 1);
            let mut fft = FftwFft::new();
            let reference_spectrum =
                reload_for_correlation(&slip, &mut fft, extents, SPACING, original);
            let produced = correlated_perturbation(
                &mut source,
                &mut fft,
                &reference_spectrum,
                extents,
                SPACING,
                spec(),
                PerturbationSpec { correlation, sigma },
            );

            assert_fields_equal(
                &produced,
                &expected,
                extents,
                &format!(
                    "perturbation rho={correlation} sigma={sigma} on {}x{}",
                    extents.fault_strike, extents.fault_dip
                ),
            );
        }
    }
}

/// `main`'s rake block, rebuilt from oracle calls.
fn reference_rake(
    extents: GridExtents,
    seed: i64,
    base_rake: &[f32],
    sigma_degrees: f32,
) -> Vec<f32> {
    let (strike_step, dip_step) = steps(extents);
    let mut grid = unit_grid(extents);
    let mut oracle_seed = seed;
    oracle::correlated_field(
        &mut grid,
        extents.padded_strike,
        extents.padded_dip,
        strike_step,
        dip_step,
        CORRELATION.strike,
        CORRELATION.dip,
        &mut oracle_seed,
        Spectrum2D::Mai as i32,
        MAX_WAVELENGTH,
        MIN_WAVELENGTH,
    );
    oracle::transform_2d(
        &mut grid,
        extents.padded_strike,
        extents.padded_dip,
        1,
        strike_step,
        dip_step,
    );

    let mut field = corner(&grid, extents);
    remove_mean(&mut field);
    let factor = sigma_degrees / population_sigma(&field);
    for (value, base) in field.iter_mut().zip(base_rake) {
        *value = factor * *value + *base;
    }
    field
}

#[test]
fn rake_fields_match_across_every_shape() {
    for extents in CASES {
        let count = extents.fault_strike * extents.fault_dip;
        // A base rake that varies, so adding it cannot be confused with a constant.
        #[expect(clippy::cast_precision_loss, reason = "small test indices")]
        let base: Vec<f32> = (0..count)
            .map(|i| 175.0 + (i as f32 * 0.05).sin())
            .collect();

        let expected = reference_rake(extents, 4242, &base, 15.0);

        let mut source = GenslipLcg::new(4242);
        let mut fft = FftwFft::new();
        let produced = rake_field(&mut source, &mut fft, extents, SPACING, spec(), &base, 15.0);

        assert_fields_equal(
            &produced,
            &expected,
            extents,
            &format!("rake {}x{}", extents.fault_strike, extents.fault_dip),
        );
    }
}

#[test]
fn the_reload_restores_the_original_padded_statistics() {
    // Not a parity claim: a statement about what the reload is FOR. It maps the
    // processed field back onto the mean and deviation the padded grid had before
    // any fault-domain processing, which is what keeps the correlated fields on the
    // same scale as the slip they blend with.
    let extents = CASES[2];
    let (slip, original) = slip_stage(extents, 909);

    let mut fft = FftwFft::new();
    let mut padded = genslip::grid::Spectrum::zeros(extents.padded_strike, extents.padded_dip);
    for dip in 0..extents.fault_dip {
        for strike in 0..extents.fault_strike {
            padded[(strike, dip)] = num_complex::Complex32::new(slip[(strike, dip)], 0.0);
        }
    }
    let before = mean_and_sigma(&padded);

    let reloaded = reload_for_correlation(&slip, &mut fft, extents, SPACING, original);

    // The reload happens before the transform, so undo the transform's DC scaling to
    // read the mean back: the DC term of an unnormalised transform is the sum.
    #[expect(clippy::cast_precision_loss, reason = "small grid extents")]
    let points = (extents.padded_strike * extents.padded_dip) as f32;
    let restored_mean = reloaded[(0, 0)].re / (points * SPACING.strike_km * SPACING.dip_km);

    assert!(
        (restored_mean - original.mean).abs() < original.sigma * 1e-4,
        "reload gave mean {restored_mean}, expected {} (padded field had {})",
        original.mean,
        before.mean
    );
}

proptest! {
    #[test]
    fn perturbations_match_for_arbitrary_correlations(
        seed in any::<i32>(),
        fault_strike in 1usize..20,
        fault_dip in 1usize..20,
        correlation in 0.0f32..1.0,
        sigma in 0.01f32..5.0,
    ) {
        let extents = GridExtents {
            fault_strike,
            fault_dip,
            padded_strike: 2 * (fault_strike.div_ceil(2) + 1),
            padded_dip: 2 * (fault_dip.div_ceil(2) + 1),
        };
        let seed = i64::from(seed);
        let (slip, original) = slip_stage(extents, seed);

        let expected =
            reference_perturbation(extents, &slip, original, seed + 1, correlation, sigma);

        let mut source = GenslipLcg::new(seed + 1);
        let mut fft = FftwFft::new();
        let reference_spectrum =
            reload_for_correlation(&slip, &mut fft, extents, SPACING, original);
        let produced = correlated_perturbation(
            &mut source, &mut fft, &reference_spectrum, extents, SPACING, spec(),
            PerturbationSpec { correlation, sigma },
        );

        for (offset, (got, want)) in produced.as_slice().iter().zip(&expected).enumerate() {
            prop_assert_eq!(got.to_bits(), want.to_bits(), "at {}", offset);
        }
    }
}

// Deliberately not asserted:
//
// - That the realised correlation between slip and the perturbation equals the
//   requested rho. It will not, exactly: rho sets the correlation of the SPECTRA on
//   the padded grid, and the field is then cropped to the fault, centred and
//   rescaled. That is a Stage 2 claim with a measured band -- and it is what
//   check_cor_r was written to answer.
// - That a rho of 1 makes the perturbation a scaled copy of slip. It nearly does,
//   but the crop and the re-centring break the exact relation.
