//! The spectral field generators are bit-identical to genslip's.
//!
//! These two kernels produce every stochastic field in a rupture model, so they are
//! the second thing pinned after the generator that feeds them. As with the RNG, the
//! test is differential against the real C linked directly — there is no fixture.
//!
//! Grid extents are deliberately stratified rather than sampled: the generators
//! address the Nyquist row and column directly and reflect about them, so square,
//! wide, tall, and minimal grids exercise different index arithmetic. Uniformly
//! random extents would almost never produce the degenerate 2xN case.

use genslip::field::{
    CorrelationLengths, Spectrum2D, WavelengthBand, WavenumberStep, correlated_field,
    self_affine_field,
};
use genslip::grid::{Spectrum, impose_hermitian_symmetry};
use genslip::rng::GenslipLcg;
use genslip_oracle::{Complex, field as oracle};
use num_complex::Complex32;
use proptest::prelude::*;

/// Grid shapes that reach different index arithmetic.
///
/// Both extents must be even: the kernels index `count / 2` as the Nyquist point
/// and reflect about it.
const SHAPES: [(usize, usize); 6] = [
    (2, 2),   // minimal: every loop body runs zero or one time
    (4, 4),   // square
    (16, 4),  // wide
    (4, 16),  // tall
    (32, 24), // realistic small fault
    (64, 8),  // long thin fault, as a strike-slip rupture is
];

fn as_oracle_grid(spectrum: &Spectrum) -> Vec<Complex> {
    spectrum
        .as_slice()
        .iter()
        .map(|value| Complex {
            re: value.re,
            im: value.im,
        })
        .collect()
}

/// Compare bit for bit, reporting the first mismatch with its grid position.
fn assert_grids_equal(actual: &Spectrum, expected: &[Complex], context: &str) {
    let strike_count = actual.strike_count();
    for (offset, (got, want)) in actual.as_slice().iter().zip(expected).enumerate() {
        assert!(
            got.re.to_bits() == want.re.to_bits() && got.im.to_bits() == want.im.to_bits(),
            "{context}: mismatch at (strike {}, dip {}): \
             got {got:?} ({:#010x}, {:#010x}), want ({}, {}) ({:#010x}, {:#010x})",
            offset % strike_count,
            offset / strike_count,
            got.re.to_bits(),
            got.im.to_bits(),
            want.re,
            want.im,
            want.re.to_bits(),
            want.im.to_bits(),
        );
    }
}

#[test]
fn hermitian_symmetry_matches_across_every_shape() {
    for (strike_count, dip_count) in SHAPES {
        // Fill with something asymmetric in both axes, so a reflection that
        // transposed, dropped a conjugation, or did nothing would all be visible.
        let mut ported = Spectrum::zeros(strike_count, dip_count);
        for dip in 0..dip_count {
            for strike in 0..strike_count {
                #[expect(clippy::cast_precision_loss, reason = "small test indices")]
                let value = Complex32::new(strike as f32 + 1.0, dip as f32 - 2.0);
                ported[(strike, dip)] = value;
            }
        }

        let mut expected = as_oracle_grid(&ported);
        oracle::impose_hermitian_symmetry(&mut expected, strike_count, dip_count);
        impose_hermitian_symmetry(&mut ported);

        assert_grids_equal(
            &ported,
            &expected,
            &format!("hermit {strike_count}x{dip_count}"),
        );
    }
}

/// Every spectral shape, on every grid shape.
#[test]
fn correlated_fields_match_across_every_shape_and_model() {
    let shapes = [
        Spectrum2D::Somerville,
        Spectrum2D::Mai,
        Spectrum2D::Frankel,
        Spectrum2D::MaiSomerville,
        Spectrum2D::Suzuki,
        Spectrum2D::InputCorners,
    ];

    for (strike_count, dip_count) in SHAPES {
        for shape in shapes {
            let scale = 3.5_f32;

            // The C reads its amplitude scale out of the grid's DC term, so seed it
            // there for the oracle and pass it explicitly to the port.
            let mut oracle_grid = vec![Complex::default(); strike_count * dip_count];
            oracle_grid[0] = Complex { re: scale, im: 0.0 };
            let mut oracle_seed: i64 = 4_242;
            oracle::correlated_field(
                &mut oracle_grid,
                strike_count,
                dip_count,
                0.05,
                0.07,
                12.0,
                6.0,
                &mut oracle_seed,
                shape as i32,
                80.0,
                1.5,
            );

            let mut ported = Spectrum::zeros(strike_count, dip_count);
            let mut source = GenslipLcg::new(4_242);
            correlated_field(
                &mut ported,
                &mut source,
                shape,
                WavenumberStep {
                    strike: 0.05,
                    dip: 0.07,
                },
                CorrelationLengths {
                    strike: 12.0,
                    dip: 6.0,
                },
                WavelengthBand::new(1.5, 80.0),
                scale,
            );

            assert_grids_equal(
                &ported,
                &oracle_grid,
                &format!("correlated {shape:?} {strike_count}x{dip_count}"),
            );
            assert_eq!(
                source.seed(),
                oracle_seed,
                "{shape:?} {strike_count}x{dip_count}: draw count diverged"
            );
        }
    }
}

#[test]
fn self_affine_fields_match_across_every_shape() {
    for (strike_count, dip_count) in SHAPES {
        let mut oracle_grid = vec![Complex::default(); strike_count * dip_count];
        let mut oracle_seed: i64 = 909;
        oracle::self_affine_field(
            &mut oracle_grid,
            strike_count,
            dip_count,
            0.05,
            0.07,
            1.0,
            80.0,
            0.08,
            &mut oracle_seed,
        );

        let mut ported = Spectrum::zeros(strike_count, dip_count);
        let mut source = GenslipLcg::new(909);
        self_affine_field(
            &mut ported,
            &mut source,
            1.0,
            WavenumberStep {
                strike: 0.05,
                dip: 0.07,
            },
            WavelengthBand::new(0.08, 80.0),
        );

        assert_grids_equal(
            &ported,
            &oracle_grid,
            &format!("self-affine {strike_count}x{dip_count}"),
        );
        assert_eq!(
            source.seed(),
            oracle_seed,
            "self-affine {strike_count}x{dip_count}: draw count diverged"
        );
    }
}

proptest! {
    /// Bit-identical across the parameter space, not just the values we picked.
    #[test]
    fn correlated_fields_match_for_arbitrary_parameters(
        seed in any::<i32>(),
        half_strike in 1usize..12,
        half_dip in 1usize..12,
        strike_step in 1e-3f32..1.0,
        dip_step in 1e-3f32..1.0,
        strike_correlation in 0.1f32..50.0,
        dip_correlation in 0.1f32..50.0,
        min_wavelength in 0.05f32..2.0,
        max_wavelength in 10.0f32..500.0,
        model in 0usize..6,
    ) {
        let shape = [
            Spectrum2D::Somerville, Spectrum2D::Mai, Spectrum2D::Frankel,
            Spectrum2D::MaiSomerville, Spectrum2D::Suzuki, Spectrum2D::InputCorners,
        ][model];
        let (strike_count, dip_count) = (2 * half_strike, 2 * half_dip);
        let scale = 1.0_f32;

        let mut oracle_grid = vec![Complex::default(); strike_count * dip_count];
        oracle_grid[0] = Complex { re: scale, im: 0.0 };
        let mut oracle_seed = i64::from(seed);
        oracle::correlated_field(
            &mut oracle_grid, strike_count, dip_count,
            strike_step, dip_step, strike_correlation, dip_correlation,
            &mut oracle_seed, shape as i32, max_wavelength, min_wavelength,
        );

        let mut ported = Spectrum::zeros(strike_count, dip_count);
        let mut source = GenslipLcg::new(i64::from(seed));
        correlated_field(
            &mut ported, &mut source, shape,
            WavenumberStep { strike: strike_step, dip: dip_step },
            CorrelationLengths { strike: strike_correlation, dip: dip_correlation },
            WavelengthBand::new(min_wavelength, max_wavelength),
            scale,
        );

        for (offset, (got, want)) in ported.as_slice().iter().zip(&oracle_grid).enumerate() {
            prop_assert_eq!(got.re.to_bits(), want.re.to_bits(), "re at {}", offset);
            prop_assert_eq!(got.im.to_bits(), want.im.to_bits(), "im at {}", offset);
        }
        prop_assert_eq!(source.seed(), oracle_seed);
    }

    #[test]
    fn self_affine_fields_match_for_arbitrary_parameters(
        seed in any::<i32>(),
        half_strike in 1usize..12,
        half_dip in 1usize..12,
        strike_step in 1e-3f32..1.0,
        dip_step in 1e-3f32..1.0,
        hurst in 0.1f32..1.5,
        min_wavelength in 0.05f32..2.0,
        max_wavelength in 10.0f32..500.0,
    ) {
        let (strike_count, dip_count) = (2 * half_strike, 2 * half_dip);

        let mut oracle_grid = vec![Complex::default(); strike_count * dip_count];
        let mut oracle_seed = i64::from(seed);
        oracle::self_affine_field(
            &mut oracle_grid, strike_count, dip_count,
            strike_step, dip_step, hurst, max_wavelength, min_wavelength,
            &mut oracle_seed,
        );

        let mut ported = Spectrum::zeros(strike_count, dip_count);
        let mut source = GenslipLcg::new(i64::from(seed));
        self_affine_field(
            &mut ported, &mut source, hurst,
            WavenumberStep { strike: strike_step, dip: dip_step },
            WavelengthBand::new(min_wavelength, max_wavelength),
        );

        for (offset, (got, want)) in ported.as_slice().iter().zip(&oracle_grid).enumerate() {
            prop_assert_eq!(got.re.to_bits(), want.re.to_bits(), "re at {}", offset);
            prop_assert_eq!(got.im.to_bits(), want.im.to_bits(), "im at {}", offset);
        }
        prop_assert_eq!(source.seed(), oracle_seed);
    }
}

// Deliberately not asserted:
//
// - That the drawn field has the spectrum it was asked for. That is a scientific
//   property, belongs in the Stage 2 suite, and would need many realisations to
//   state honestly. These tests answer only "is it what the C produces".
// - Anything about the amplitude scale being read from grid[0]. That is an
//   interface accident of the C which the port does not reproduce -- the scale is
//   an argument here -- so there is nothing to pin.
