//! Scaling slip to a target moment is bit-identical to genslip's.
//!
//! The C writes its answer into `pointsource[i].slip` and reports three summary
//! values through out-parameters. The port takes rigidity as a slice and returns a
//! value, so the comparison is field-by-field rather than struct-by-struct — which
//! is the point: nothing here needs a twenty-one-field record.

use genslip::moment::{ScaledSlip, SlipScaling, SubfaultSize, scale_slip};
use genslip::taper::SlipField;
use genslip_oracle::{PointSource, field as oracle};
use proptest::prelude::*;

const SHAPES: [(usize, usize); 5] = [(1, 1), (4, 4), (16, 4), (32, 24), (64, 8)];

/// A field with both signs present, so the zero-floored maximum and the
/// negative-slip handling are exercised rather than assumed away.
fn seeded_field(strike_count: usize, dip_count: usize, dip_offset: usize) -> SlipField {
    let mut field = SlipField::zeros(strike_count, dip_count + dip_offset);
    for dip in 0..(dip_count + dip_offset) {
        for strike in 0..strike_count {
            #[expect(clippy::cast_precision_loss, reason = "small test indices")]
            let value = 0.6 + (strike as f32 * 0.31).sin() + (dip as f32 * 0.17).cos() * 0.4;
            field[(strike, dip)] = value;
        }
    }
    field
}

/// Rigidity varying across the fault, as a real velocity model gives.
fn seeded_rigidity(strike_count: usize, dip_count: usize) -> Vec<f32> {
    (0..strike_count * dip_count)
        .map(|index| {
            #[expect(clippy::cast_precision_loss, reason = "small test indices")]
            let depth_row = (index / strike_count) as f32;
            3.0e+11 + depth_row * 2.5e+10
        })
        .collect()
}

fn run_oracle(
    field: &SlipField,
    rigidity: &[f32],
    strike_count: usize,
    dip_count: usize,
    dip_offset: usize,
    moment: f32,
    average: f32,
) -> (Vec<f32>, oracle::SlipScalingResult) {
    let mut subfaults: Vec<PointSource> = rigidity
        .iter()
        .map(|&mu| PointSource {
            mu,
            ..PointSource::default()
        })
        .collect();
    let mut buffer = field.as_slice().to_vec();

    let result = oracle::scale_slip(
        &mut subfaults,
        &mut buffer,
        strike_count,
        dip_count,
        dip_offset,
        2.0,
        1.5,
        45.0,
        moment,
        average,
    );

    (
        subfaults.iter().map(|subfault| subfault.slip).collect(),
        result,
    )
}

fn assert_matches(
    ported: &ScaledSlip,
    expected_slip: &[f32],
    expected: oracle::SlipScalingResult,
    context: &str,
) {
    for (offset, (got, want)) in ported.slip.as_slice().iter().zip(expected_slip).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{context}: slip mismatch at {offset}: {got} vs {want}"
        );
    }
    assert_eq!(
        ported.average_cm.to_bits(),
        expected.average.to_bits(),
        "{context}: average {} vs {}",
        ported.average_cm,
        expected.average
    );
    assert_eq!(
        ported.maximum_cm.to_bits(),
        expected.maximum.to_bits(),
        "{context}: maximum {} vs {}",
        ported.maximum_cm,
        expected.maximum
    );
    assert_eq!(
        ported.moment_dyne_cm.to_bits(),
        expected.moment.to_bits(),
        "{context}: moment {} vs {}",
        ported.moment_dyne_cm,
        expected.moment
    );
}

#[test]
fn moment_scaling_matches_across_every_shape() {
    let moment = 1.25e+26_f32;

    for (strike_count, dip_count) in SHAPES {
        for dip_offset in [0, 1, 3] {
            let field = seeded_field(strike_count, dip_count, dip_offset);
            let rigidity = seeded_rigidity(strike_count, dip_count);

            // A negative average is how the C selects moment mode.
            let (expected_slip, expected) = run_oracle(
                &field,
                &rigidity,
                strike_count,
                dip_count,
                dip_offset,
                moment,
                -1.0,
            );

            let ported = scale_slip(
                &field,
                dip_offset,
                dip_count,
                &rigidity,
                SubfaultSize {
                    strike_km: 2.0,
                    dip_km: 1.5,
                },
                SlipScaling::Moment { dyne_cm: moment },
            );

            assert_matches(
                &ported,
                &expected_slip,
                expected,
                &format!("moment {strike_count}x{dip_count} offset {dip_offset}"),
            );
        }
    }
}

#[test]
fn average_slip_scaling_matches_across_every_shape() {
    for (strike_count, dip_count) in SHAPES {
        for dip_offset in [0, 2] {
            let field = seeded_field(strike_count, dip_count, dip_offset);
            let rigidity = seeded_rigidity(strike_count, dip_count);

            let (expected_slip, expected) = run_oracle(
                &field,
                &rigidity,
                strike_count,
                dip_count,
                dip_offset,
                0.0,
                125.0,
            );

            let ported = scale_slip(
                &field,
                dip_offset,
                dip_count,
                &rigidity,
                SubfaultSize {
                    strike_km: 2.0,
                    dip_km: 1.5,
                },
                SlipScaling::AverageSlip { centimetres: 125.0 },
            );

            assert_matches(
                &ported,
                &expected_slip,
                expected,
                &format!("average {strike_count}x{dip_count} offset {dip_offset}"),
            );
        }
    }
}

proptest! {
    #[test]
    fn moment_scaling_matches_for_arbitrary_faults(
        strike_count in 1usize..30,
        dip_count in 1usize..30,
        dip_offset in 0usize..4,
        moment in 1.0e+22f32..1.0e+28,
        strike_km in 0.1f32..5.0,
        dip_km in 0.1f32..5.0,
    ) {
        let field = seeded_field(strike_count, dip_count, dip_offset);
        let rigidity = seeded_rigidity(strike_count, dip_count);

        let mut subfaults: Vec<PointSource> = rigidity
            .iter()
            .map(|&mu| PointSource { mu, ..PointSource::default() })
            .collect();
        let mut buffer = field.as_slice().to_vec();
        let expected = oracle::scale_slip(
            &mut subfaults, &mut buffer, strike_count, dip_count, dip_offset,
            strike_km, dip_km, 45.0, moment, -1.0,
        );

        let ported = scale_slip(
            &field, dip_offset, dip_count, &rigidity,
            SubfaultSize { strike_km, dip_km },
            SlipScaling::Moment { dyne_cm: moment },
        );

        for (offset, (got, want)) in
            ported.slip.as_slice().iter().zip(&subfaults).enumerate()
        {
            prop_assert_eq!(got.to_bits(), want.slip.to_bits(), "slip at {}", offset);
        }
        prop_assert_eq!(ported.average_cm.to_bits(), expected.average.to_bits());
        prop_assert_eq!(ported.maximum_cm.to_bits(), expected.maximum.to_bits());
    }
}

// Deliberately not asserted:
//
// - That the scaled field's moment equals the target. It does to rounding, but the
//   sum is a single-precision fold over every subfault, so on a large fault the
//   round-trip error is visible. That is a scientific claim for Stage 2 with a
//   measured tolerance, not a bit-level one.
// - That `maximum_cm` is the largest value in the field. It is the largest value
//   *or zero*, because the running maximum starts at zero. On an all-negative field
//   those differ, and the original reports zero.
