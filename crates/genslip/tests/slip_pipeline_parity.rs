//! The slip pipeline's spatial operations are bit-identical to genslip's.
//!
//! `shift_phase` translates a field on the fault by rotating its phase;
//! `taper_slip_all_r` ramps the resulting slip to zero at the fault edges. Both are
//! pure array operations with no randomness, so unlike the generators these are
//! pinned on their values alone — there is no stream position to compare.

use genslip::field::{
    CorrelationLengths, Spectrum2D, WavelengthBand, WavenumberStep, correlated_field, shift_phase,
};
use genslip::grid::Spectrum;
use genslip::rng::GenslipLcg;
use genslip::taper::{EdgeTapers, SlipField, taper_edges};
use genslip_oracle::{Complex, field as oracle};
use proptest::prelude::*;

const SHAPES: [(usize, usize); 6] = [(2, 2), (4, 4), (16, 4), (4, 16), (32, 24), (64, 8)];

/// A non-trivial field to operate on. Shifting or tapering zeros is a no-op that
/// would pass whatever the implementation did.
fn seeded_spectrum(strike_count: usize, dip_count: usize, seed: i64) -> Spectrum {
    let mut spectrum = Spectrum::zeros(strike_count, dip_count);
    let mut source = GenslipLcg::new(seed);
    correlated_field(
        &mut spectrum,
        &mut source,
        Spectrum2D::Mai,
        WavenumberStep {
            strike: 0.05,
            dip: 0.07,
        },
        CorrelationLengths {
            strike: 12.0,
            dip: 6.0,
        },
        WavelengthBand::new(1.5, 80.0),
        3.5,
    );
    spectrum
}

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

#[test]
fn phase_shifts_match_across_every_shape() {
    // Zero is the configured value and must be a no-op; the others must not be.
    for (strike_shift, dip_shift) in [(0.0, 0.0), (0.25, 0.0), (0.0, -0.5), (1.75, 2.5)] {
        for (strike_count, dip_count) in SHAPES {
            let mut ported = seeded_spectrum(strike_count, dip_count, 5150);
            let mut expected = as_oracle_grid(&ported);

            oracle::shift_phase_of(
                &mut expected,
                strike_count,
                dip_count,
                0.05,
                0.07,
                strike_shift,
                dip_shift,
            );
            shift_phase(
                &mut ported,
                WavenumberStep {
                    strike: 0.05,
                    dip: 0.07,
                },
                strike_shift,
                dip_shift,
            );

            for (offset, (got, want)) in ported.as_slice().iter().zip(&expected).enumerate() {
                assert_eq!(
                    (got.re.to_bits(), got.im.to_bits()),
                    (want.re.to_bits(), want.im.to_bits()),
                    "shift ({strike_shift}, {dip_shift}) on {strike_count}x{dip_count}: \
                     mismatch at (strike {}, dip {})",
                    offset % strike_count,
                    offset / strike_count,
                );
            }
        }
    }
}

#[test]
fn a_negative_dc_term_is_made_positive_by_a_phase_shift() {
    // Not a parity claim but a statement about a surprise: the DC term is restored
    // from a saved MAGNITUDE, so a negative mean comes back positive even though the
    // phase factor at the origin is exactly 1. Pinned so a rewrite has to decide.
    let mut spectrum = Spectrum::zeros(4, 4);
    spectrum[(0, 0)] = num_complex::Complex32::new(-7.5, 0.0);

    shift_phase(
        &mut spectrum,
        WavenumberStep {
            strike: 0.05,
            dip: 0.07,
        },
        0.0,
        0.0,
    );

    #[expect(
        clippy::float_cmp,
        reason = "exact: the magnitude of -7.5 is 7.5, not something near it"
    )]
    let sign_was_flipped = spectrum[(0, 0)].re == 7.5;
    assert!(sign_was_flipped);
}

/// Slip values with structure in both axes, so an edge ramp is visible everywhere.
fn seeded_slip(strike_count: usize, dip_count: usize) -> SlipField {
    let mut slip = SlipField::zeros(strike_count, dip_count);
    for dip in 0..dip_count {
        for strike in 0..strike_count {
            #[expect(clippy::cast_precision_loss, reason = "small test indices")]
            let value = 1.0 + strike as f32 * 0.25 - dip as f32 * 0.125;
            slip[(strike, dip)] = value;
        }
    }
    slip
}

#[test]
fn edge_tapers_match_across_every_shape_and_fraction() {
    // 0.02 is the configured side taper and 0.0 the configured top and bottom, so
    // both extremes are the production case. The larger values reach the branches
    // where the ramps are wide enough to overlap.
    let fractions = [
        EdgeTapers::default(),
        EdgeTapers {
            sides: 0.02,
            top: 0.0,
            bottom: 0.0,
        },
        EdgeTapers {
            sides: 0.1,
            top: 0.1,
            bottom: 0.1,
        },
        EdgeTapers {
            sides: 0.4,
            top: 0.3,
            bottom: 0.3,
        },
        EdgeTapers {
            sides: 0.0,
            top: 0.5,
            bottom: 0.5,
        },
    ];

    for (strike_count, dip_count) in SHAPES {
        for tapers in fractions {
            let mut ported = seeded_slip(strike_count, dip_count);
            let mut expected = ported.as_slice().to_vec();

            oracle::taper_edges(
                &mut expected,
                strike_count,
                dip_count,
                tapers.sides,
                tapers.bottom,
                tapers.top,
            );
            taper_edges(&mut ported, &tapers);

            for (offset, (got, want)) in ported.as_slice().iter().zip(&expected).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "tapers {tapers:?} on {strike_count}x{dip_count}: \
                     mismatch at (strike {}, dip {}): {got} vs {want}",
                    offset % strike_count,
                    offset / strike_count,
                );
            }
        }
    }
}

proptest! {
    #[test]
    fn phase_shifts_match_for_arbitrary_shifts(
        seed in any::<i32>(),
        half_strike in 1usize..10,
        half_dip in 1usize..10,
        strike_shift in -5.0f64..5.0,
        dip_shift in -5.0f64..5.0,
    ) {
        let (strike_count, dip_count) = (2 * half_strike, 2 * half_dip);
        let mut ported = seeded_spectrum(strike_count, dip_count, i64::from(seed));
        let mut expected = as_oracle_grid(&ported);

        oracle::shift_phase_of(
            &mut expected, strike_count, dip_count, 0.05, 0.07, strike_shift, dip_shift,
        );
        shift_phase(
            &mut ported,
            WavenumberStep { strike: 0.05, dip: 0.07 },
            strike_shift,
            dip_shift,
        );

        for (offset, (got, want)) in ported.as_slice().iter().zip(&expected).enumerate() {
            prop_assert_eq!(got.re.to_bits(), want.re.to_bits(), "re at {}", offset);
            prop_assert_eq!(got.im.to_bits(), want.im.to_bits(), "im at {}", offset);
        }
    }

    #[test]
    fn edge_tapers_match_for_arbitrary_fractions(
        strike_count in 1usize..40,
        dip_count in 1usize..40,
        sides in 0.0f32..0.6,
        top in 0.0f32..0.6,
        bottom in 0.0f32..0.6,
    ) {
        let tapers = EdgeTapers { sides, top, bottom };
        let mut ported = seeded_slip(strike_count, dip_count);
        let mut expected = ported.as_slice().to_vec();

        oracle::taper_edges(
            &mut expected, strike_count, dip_count, sides, bottom, top,
        );
        taper_edges(&mut ported, &tapers);

        for (offset, (got, want)) in ported.as_slice().iter().zip(&expected).enumerate() {
            prop_assert_eq!(got.to_bits(), want.to_bits(), "at {}", offset);
        }
    }
}

// Deliberately not asserted:
//
// - That a taper actually reaches zero at the boundary. It does not: the outermost
//   cell is damped to 1/width, not to 0. Asserting otherwise would be asserting
//   something false about the original.
// - That a phase shift preserves the field's energy. It does not exactly, because
//   the DC and Nyquist points are forced real afterwards, which discards whatever
//   imaginary part the rotation gave them.
