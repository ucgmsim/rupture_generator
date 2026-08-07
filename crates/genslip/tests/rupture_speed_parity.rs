//! The rupture-speed field reproduces genslip's, and the onset assembly with it.
#![cfg(feature = "wavefront-compat")]

use genslip::rise_time::DepthRamp;
use genslip::rupture::{
    EikonalSolver, Hypocentre, OnsetAdjustment, SpeedProfile, Wavefront2d, onset_times, speed_field,
};
use genslip::taper::SlipField;
use genslip_oracle::{PointSource, field as oracle};
use proptest::prelude::*;

const SHAPES: [(usize, usize); 4] = [(1, 1), (8, 6), (32, 12), (24, 24)];

fn profile() -> SpeedProfile {
    SpeedProfile {
        shallow: DepthRamp {
            centre_km: 6.5,
            half_width_km: 1.5,
        },
        shallow_factor: 0.6,
        deep: DepthRamp {
            centre_km: 17.5,
            half_width_km: 2.5,
        },
        deep_factor: 0.6,
    }
}

fn oracle_profile() -> oracle::SpeedProfileArgs {
    let p = profile();
    oracle::SpeedProfileArgs {
        shallow_factor: p.shallow_factor,
        shallow_min_km: p.shallow.centre_km - p.shallow.half_width_km,
        shallow_max_km: p.shallow.centre_km + p.shallow.half_width_km,
        deep_factor: p.deep_factor,
        deep_min_km: p.deep.centre_km - p.deep.half_width_km,
        deep_max_km: p.deep.centre_km + p.deep.half_width_km,
    }
}

/// Depths spanning both transitions, so every branch of the profile is reached.
#[expect(clippy::cast_precision_loss, reason = "small test indices")]
fn depths(dip_count: usize) -> Vec<f32> {
    (0..dip_count).map(|dip| dip as f32 * 1.7).collect()
}

fn field(strike_count: usize, dip_count: usize, base: f32, spread: f32) -> SlipField {
    let mut field = SlipField::zeros(strike_count, dip_count);
    for dip in 0..dip_count {
        for strike in 0..strike_count {
            #[expect(clippy::cast_precision_loss, reason = "small test indices")]
            let value = base + spread * (strike as f32 * 0.31 + dip as f32 * 0.17).sin();
            field[(strike, dip)] = value;
        }
    }
    field
}

#[test]
fn rupture_speeds_match_across_every_shape() {
    for (strike_count, dip_count) in SHAPES {
        let shear = field(strike_count, dip_count, 3.0, 0.4);
        let fraction = field(strike_count, dip_count, 0.8, 0.05);
        let depth_km = depths(dip_count);

        let mut subfaults: Vec<PointSource> = (0..strike_count * dip_count)
            .map(|index| PointSource {
                dep: depth_km[index / strike_count],
                vs: shear.as_slice()[index],
                rvf: fraction.as_slice()[index],
                slip: 1.0,
                ..PointSource::default()
            })
            .collect();

        let expected = oracle::rupture_speed(
            &mut subfaults,
            strike_count,
            dip_count,
            oracle_profile(),
            false,
        );
        let produced = speed_field(&shear, &fraction, &depth_km, profile());

        for (offset, want) in expected.iter().enumerate() {
            let got = produced.speed(offset % strike_count, offset / strike_count);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{strike_count}x{dip_count}: mismatch at {offset}: {got} vs {want}"
            );
        }
    }
}

proptest! {
    #[test]
    fn rupture_speeds_match_for_arbitrary_profiles(
        strike_count in 1usize..20,
        dip_count in 1usize..20,
        shallow_factor in 0.1f32..1.0,
        deep_factor in 0.1f32..1.0,
        shallow_centre in 1.0f32..12.0,
        deep_centre in 13.0f32..30.0,
    ) {
        let shear = field(strike_count, dip_count, 3.0, 0.4);
        let fraction = field(strike_count, dip_count, 0.8, 0.05);
        let depth_km = depths(dip_count);
        let profile = SpeedProfile {
            shallow: DepthRamp { centre_km: shallow_centre, half_width_km: 1.5 },
            shallow_factor,
            deep: DepthRamp { centre_km: deep_centre, half_width_km: 2.5 },
            deep_factor,
        };
        let oracle_args = oracle::SpeedProfileArgs {
            shallow_factor,
            shallow_min_km: shallow_centre - 1.5,
            shallow_max_km: shallow_centre + 1.5,
            deep_factor,
            deep_min_km: deep_centre - 2.5,
            deep_max_km: deep_centre + 2.5,
        };

        let mut subfaults: Vec<PointSource> = (0..strike_count * dip_count)
            .map(|index| PointSource {
                dep: depth_km[index / strike_count],
                vs: shear.as_slice()[index],
                rvf: fraction.as_slice()[index],
                slip: 1.0,
                ..PointSource::default()
            })
            .collect();

        let expected = oracle::rupture_speed(
            &mut subfaults, strike_count, dip_count, oracle_args, false,
        );
        let produced = speed_field(&shear, &fraction, &depth_km, profile);

        for (offset, want) in expected.iter().enumerate() {
            let got = produced.speed(offset % strike_count, offset / strike_count);
            prop_assert_eq!(got.to_bits(), want.to_bits(), "at {}", offset);
        }
    }
}

#[test]
fn the_hypocentre_segment_starts_at_zero_and_others_keep_their_offset() {
    let (strike_count, dip_count) = (16, 8);
    let shear = field(strike_count, dip_count, 3.0, 0.4);
    let fraction = field(strike_count, dip_count, 0.8, 0.0);
    let depth_km = depths(dip_count);
    let speed = speed_field(&shear, &fraction, &depth_km, profile());

    let travel = Wavefront2d::new().solve(
        &speed,
        Hypocentre {
            strike: strike_count / 2,
            dip: dip_count / 2,
        },
        1.0,
    );
    let perturbation = field(strike_count, dip_count, 0.0, 0.3);

    let hosting = onset_times(
        &travel,
        &perturbation,
        OnsetAdjustment {
            perturbation_scale: -0.4,
            delay_s: 0.0,
            contains_hypocentre: true,
        },
    );
    let earliest = hosting
        .as_slice()
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert!(
        earliest.abs() < 1e-6,
        "the hosting segment starts at {earliest}, expected 0"
    );

    // Without the hypocentre, the times keep whatever offset the solver gave them --
    // which is what lets a rupture propagate between segments instead of restarting
    // in each. So the two differ by a constant, and that constant is what the
    // hosting segment removed.
    let following = onset_times(
        &travel,
        &perturbation,
        OnsetAdjustment {
            perturbation_scale: -0.4,
            delay_s: 0.0,
            contains_hypocentre: false,
        },
    );

    let removed = following
        .as_slice()
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    assert!(
        removed > 0.0,
        "the unshifted segment starts at {removed}; it should carry a real offset"
    );

    for (unshifted, shifted) in following.as_slice().iter().zip(hosting.as_slice()) {
        assert!(
            (unshifted - removed - shifted).abs() < 1e-5,
            "the two segments differ by {} rather than a constant {removed}",
            unshifted - shifted
        );
    }
}

#[test]
fn the_delay_moves_every_subfault_equally() {
    let (strike_count, dip_count) = (8, 8);
    let shear = field(strike_count, dip_count, 3.0, 0.0);
    let fraction = field(strike_count, dip_count, 0.8, 0.0);
    let depth_km = depths(dip_count);
    let speed = speed_field(&shear, &fraction, &depth_km, profile());
    let travel = Wavefront2d::new().solve(&speed, Hypocentre { strike: 4, dip: 4 }, 1.0);
    let perturbation = SlipField::zeros(strike_count, dip_count);

    let base = onset_times(
        &travel,
        &perturbation,
        OnsetAdjustment {
            perturbation_scale: 0.0,
            delay_s: 0.0,
            contains_hypocentre: true,
        },
    );
    let delayed = onset_times(
        &travel,
        &perturbation,
        OnsetAdjustment {
            perturbation_scale: 0.0,
            delay_s: 2.5,
            contains_hypocentre: true,
        },
    );

    for (early, late) in base.as_slice().iter().zip(delayed.as_slice()) {
        assert!((late - early - 2.5).abs() < 1e-6);
    }
}

// Deliberately not asserted:
//
// - Anything about the onset assembly against the C. It is nine lines inline in
//   `main` with no function to call, and every one of them is a float add over
//   values the surrounding tests already pin. The properties above say what it is
//   for instead.
