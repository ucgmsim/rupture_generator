//! The slip pipeline reproduces genslip's, stage for stage.
//!
//! # Why this test is shaped differently from the others
//!
//! Every other parity test calls one C function. This one cannot: the slip pipeline
//! is not a function in genslip, it is a few hundred lines inline in `main` between
//! two `fprintf`s. There is nothing to link against.
//!
//! Pinning `main`'s intermediate values instead would be the wrong move — those are
//! C-shaped seams, and the predecessor project deleted every golden it had of that
//! kind because they fail whenever the internals move even though nothing observable
//! changed.
//!
//! So the reference here is built from the **oracle's own functions, called in the
//! order `main` calls them**. Each is already pinned bit-for-bit individually, so
//! composing them is a faithful stand-in for the block — and what it actually tests
//! is the thing that could be wrong: the sequence. A missed stage, a swapped pair, a
//! wavenumber step passed where a subfault spacing belongs.
#![cfg(feature = "fftw")]

use genslip::fft::FftwFft;
use genslip::field::{CorrelationLengths, Spectrum2D, WavelengthBand};
use genslip::rng::GenslipLcg;
use genslip::slip::{GridExtents, SpectrumSpec, SubfaultSpacing, generate_normalised};
use genslip_oracle::{Complex, field as oracle};
use proptest::prelude::*;

/// Fault and padded extents, as genslip's ~10% padding produces.
const CASES: [GridExtents; 5] = [
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
        fault_strike: 64,
        fault_dip: 8,
        padded_strike: 72,
        padded_dip: 10,
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

fn spec(coefficient_of_variation: f32, phase_shift: (f64, f64)) -> SpectrumSpec {
    SpectrumSpec {
        shape: Spectrum2D::Mai,
        correlation: CorrelationLengths {
            strike: 12.0,
            dip: 6.0,
        },
        band: WavelengthBand::new(1.5, 80.0),
        coefficient_of_variation,
        phase_shift,
    }
}

/// `main`'s slip block, rebuilt from oracle calls in the order it makes them.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "mirrors the port's casts and narrowing seams exactly; that is the point"
)]
fn reference(
    extents: GridExtents,
    spacing: SubfaultSpacing,
    spectrum_spec: SpectrumSpec,
    seed: i64,
) -> Vec<f32> {
    let (padded_strike, padded_dip) = (extents.padded_strike, extents.padded_dip);
    let points = padded_strike * padded_dip;

    // genslip_v5.6.2.c:1699 -- a padded grid of ones.
    let mut grid = vec![Complex { re: 1.0, im: 0.0 }; points];

    // :1705 -- forward, scaled by the subfault spacings.
    oracle::transform_2d(
        &mut grid,
        padded_strike,
        padded_dip,
        -1,
        spacing.strike_km,
        spacing.dip_km,
    );

    let strike_step = 1.0 / (padded_strike as f32 * spacing.strike_km);
    let dip_step = 1.0 / (padded_dip as f32 * spacing.dip_km);

    // :1707 -- the spectrum. Its amplitude scale is read from the grid's DC term,
    // which the transform above has just set, so nothing needs seeding here.
    let mut oracle_seed = seed;
    oracle::correlated_field(
        &mut grid,
        padded_strike,
        padded_dip,
        strike_step,
        dip_step,
        spectrum_spec.correlation.strike,
        spectrum_spec.correlation.dip,
        &mut oracle_seed,
        spectrum_spec.shape as i32,
        80.0,
        1.5,
    );

    // :1714 -- the optional translation.
    let (strike_shift, dip_shift) = spectrum_spec.phase_shift;
    if strike_shift != 0.0 || dip_shift != 0.0 {
        oracle::shift_phase_of(
            &mut grid,
            padded_strike,
            padded_dip,
            strike_step,
            dip_step,
            strike_shift,
            dip_shift,
        );
    }

    // :1720 -- back to the fault, scaled by the wavenumber steps.
    oracle::transform_2d(
        &mut grid,
        padded_strike,
        padded_dip,
        1,
        strike_step,
        dip_step,
    );

    // :1783 -- the fault's own corner of the padded grid, and its total.
    let mut slip = Vec::with_capacity(extents.fault_strike * extents.fault_dip);
    let mut total = 0.0_f32;
    for dip in 0..extents.fault_dip {
        for strike in 0..extents.fault_strike {
            let value = grid[strike + dip * padded_strike].re;
            slip.push(value);
            total += value;
        }
    }

    // :1795 -- polarity.
    if total < 0.0 {
        for value in &mut slip {
            *value = -*value;
        }
        total = -total;
    }

    // :1829 -- unit mean.
    let subfault_count = (extents.fault_strike * extents.fault_dip) as f32;
    let mean = total / subfault_count;
    for value in &mut slip {
        *value /= mean;
    }

    // :1838 -- the target coefficient of variation.
    if spectrum_spec.coefficient_of_variation > 0.0 {
        let mut sum_of_squares = 0.0_f32;
        for value in &slip {
            sum_of_squares += (*value - 1.0) * (*value - 1.0);
        }
        let variation = f64::from(sum_of_squares / subfault_count).sqrt() as f32;
        let factor = spectrum_spec.coefficient_of_variation / variation;
        for value in &mut slip {
            *value = factor * (*value - 1.0) + 1.0;
        }
    }

    slip
}

fn check(extents: GridExtents, spectrum_spec: SpectrumSpec, seed: i64, label: &str) {
    let expected = reference(extents, SPACING, spectrum_spec, seed);

    let mut source = GenslipLcg::new(seed);
    let mut fft = FftwFft::new();
    let produced = generate_normalised(&mut source, &mut fft, extents, SPACING, spectrum_spec);

    for (offset, (got, want)) in produced.as_slice().iter().zip(&expected).enumerate() {
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
fn the_default_pipeline_matches() {
    for extents in CASES {
        check(
            extents,
            spec(0.75, (0.0, 0.0)),
            20_260_807,
            &format!("default {}x{}", extents.fault_strike, extents.fault_dip),
        );
    }
}

#[test]
fn the_pipeline_matches_with_a_phase_shift() {
    for extents in CASES {
        check(
            extents,
            spec(0.75, (0.25, -0.5)),
            77,
            &format!("shifted {}x{}", extents.fault_strike, extents.fault_dip),
        );
    }
}

#[test]
fn the_pipeline_matches_without_a_variation_target() {
    // A non-positive target leaves the field's own variation alone, which is the
    // path the Frankel spectrum forces.
    for extents in CASES {
        check(
            extents,
            spec(-1.0, (0.0, 0.0)),
            5150,
            &format!("unscaled {}x{}", extents.fault_strike, extents.fault_dip),
        );
    }
}

#[test]
fn every_spectral_shape_matches() {
    let shapes = [
        Spectrum2D::Somerville,
        Spectrum2D::Mai,
        Spectrum2D::Frankel,
        Spectrum2D::MaiSomerville,
        Spectrum2D::Suzuki,
        Spectrum2D::InputCorners,
    ];
    for shape in shapes {
        let mut spectrum_spec = spec(0.75, (0.0, 0.0));
        spectrum_spec.shape = shape;
        check(CASES[2], spectrum_spec, 909, &format!("{shape:?}"));
    }
}

proptest! {
    #[test]
    fn the_pipeline_matches_for_arbitrary_faults(
        seed in any::<i32>(),
        fault_strike in 1usize..24,
        fault_dip in 1usize..24,
        extra_strike in 0usize..3,
        extra_dip in 0usize..3,
        coefficient_of_variation in 0.05f32..2.0,
    ) {
        // The padded grid is the fault rounded up to even, plus a little.
        let padded_strike = 2 * (fault_strike.div_ceil(2) + extra_strike);
        let padded_dip = 2 * (fault_dip.div_ceil(2) + extra_dip);
        let extents = GridExtents {
            fault_strike, fault_dip, padded_strike, padded_dip,
        };
        let spectrum_spec = spec(coefficient_of_variation, (0.0, 0.0));

        let expected = reference(extents, SPACING, spectrum_spec, i64::from(seed));
        let mut source = GenslipLcg::new(i64::from(seed));
        let mut fft = FftwFft::new();
        let produced =
            generate_normalised(&mut source, &mut fft, extents, SPACING, spectrum_spec);

        for (offset, (got, want)) in produced.as_slice().iter().zip(&expected).enumerate() {
            prop_assert_eq!(got.to_bits(), want.to_bits(), "at {}", offset);
        }
    }
}

// Deliberately not asserted:
//
// - Anything about `main`'s intermediate values. They are C-shaped seams and pinning
//   them would make every future refactor fail for no observable reason.
// - That the produced field has mean 1 after the variation rescale. It does to
//   rounding, but the rescale is an affine map about a mean assumed to be exactly 1,
//   and the assumption is only true to the precision of the fold that produced it.
//   That is a Stage 2 claim with a measured tolerance.
