//! Rupture times reproduce genslip's, including its padding defect.
//!
//! `Wavefront2d` calls the same Fortran genslip calls, so the *solver* cannot
//! differ. What this pins is everything around it: the grid growth that gives the
//! source room, the edge replication that fills the new cells, the 0-to-1-based index
//! conversion, and the crop back to the fault.
//!
//! That surround is where the bugs live. It is spread across `main` and
//! `get_rslow_stretch` in the original, and one of its passes overwrites real data.
#![cfg(all(feature = "wavefront-compat", feature = "fftw"))]

use genslip::rupture::{EikonalSolver, Hypocentre, SpeedGrid, Wavefront2d};
use genslip_oracle::field as oracle;
use proptest::prelude::*;

const RING_RADIUS: usize = 2;

/// Fault shapes, including ones small enough that padding is forced.
const SHAPES: [(usize, usize); 5] = [(8, 8), (20, 10), (40, 12), (6, 6), (64, 4)];

/// A rupture-speed field that varies in both directions, as a depth taper gives.
fn seeded_speed(strike_count: usize, dip_count: usize) -> SpeedGrid {
    let values = (0..strike_count * dip_count)
        .map(|index| {
            #[expect(clippy::cast_precision_loss, reason = "small test indices")]
            let (strike, dip) = ((index % strike_count) as f32, (index / strike_count) as f32);
            // 2.0 to ~3.2 km/s, the range a shallow taper produces.
            2.0 + 0.6 * (dip * 0.3).tanh() + 0.15 * (strike * 0.2).sin()
        })
        .collect();
    SpeedGrid::new(strike_count, dip_count, values)
}

/// A grid index from a fraction of the extent, clamped inside it.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the fraction is in [0, 1) and extents are small"
)]
fn index_from(fraction: f64, extent: usize) -> usize {
    ((fraction * extent as f64) as usize).min(extent - 1)
}

/// `main`'s padding arithmetic, reproduced for the reference side.
fn padding(source: usize, extent: usize) -> (usize, usize, usize) {
    if source < RING_RADIUS + 1 {
        let offset = RING_RADIUS + 1 - source;
        (offset, extent + offset, RING_RADIUS + 1)
    } else if source + RING_RADIUS + 2 > extent {
        (0, source + RING_RADIUS + 2, source)
    } else {
        (0, extent, source)
    }
}

/// The C's whole sequence: grow, replicate, solve, crop.
fn reference(speed: &SpeedGrid, hypocentre: Hypocentre, spacing_km: f64) -> Vec<f64> {
    let (strike_offset, padded_strike, source_strike) =
        padding(hypocentre.strike, speed.strike_count());
    let (dip_offset, padded_dip, source_dip) = padding(hypocentre.dip, speed.dip_count());

    let mut speeds: Vec<f32> = (0..speed.dip_count())
        .flat_map(|dip| (0..speed.strike_count()).map(move |strike| (strike, dip)))
        .map(|(strike, dip)| speed.speed(strike, dip))
        .collect();

    let mut rng_state: i64 = 1;
    let mut slowness = oracle::padded_slowness(
        &mut speeds,
        speed.strike_count(),
        speed.dip_count(),
        padded_strike,
        padded_dip,
        strike_offset,
        dip_offset,
        0.0,
        &mut rng_state,
    );
    assert_eq!(
        rng_state, 1,
        "get_rslow_stretch drew from the stream; rvel_rand is hardwired to zero so it must not"
    );

    let times = oracle::wavefront_times(
        &mut slowness,
        padded_strike,
        padded_dip,
        source_strike,
        source_dip,
        spacing_km,
        RING_RADIUS,
    );

    let mut cropped = Vec::with_capacity(speed.strike_count() * speed.dip_count());
    for dip in 0..speed.dip_count() {
        for strike in 0..speed.strike_count() {
            cropped.push(times[(strike + strike_offset) + (dip + dip_offset) * padded_strike]);
        }
    }
    cropped
}

fn check(speed: &SpeedGrid, hypocentre: Hypocentre, spacing_km: f64, label: &str) {
    let expected = reference(speed, hypocentre, spacing_km);
    let produced = Wavefront2d::new().solve(speed, hypocentre, spacing_km);

    for (offset, (got, want)) in produced.as_slice().iter().zip(&expected).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{label}: mismatch at (strike {}, dip {}): {got} vs {want}",
            offset % speed.strike_count(),
            offset / speed.strike_count(),
        );
    }
}

#[test]
fn rupture_times_match_with_an_interior_hypocentre() {
    for (strike_count, dip_count) in SHAPES {
        let speed = seeded_speed(strike_count, dip_count);
        let hypocentre = Hypocentre {
            strike: strike_count / 2,
            dip: dip_count / 2,
        };
        check(
            &speed,
            hypocentre,
            1.0,
            &format!("interior {strike_count}x{dip_count}"),
        );
    }
}

#[test]
fn rupture_times_match_with_the_hypocentre_against_every_edge() {
    // Each of these forces a different padding branch, and the corners force two.
    for (strike_count, dip_count) in SHAPES {
        let speed = seeded_speed(strike_count, dip_count);
        for (strike, dip) in [
            (0, dip_count / 2),
            (strike_count - 1, dip_count / 2),
            (strike_count / 2, 0),
            (strike_count / 2, dip_count - 1),
            (0, 0),
            (strike_count - 1, dip_count - 1),
        ] {
            check(
                &speed,
                Hypocentre { strike, dip },
                1.0,
                &format!("edge ({strike},{dip}) on {strike_count}x{dip_count}"),
            );
        }
    }
}

#[test]
fn the_padding_overwrites_real_slowness_and_that_is_reproduced() {
    // genslip's edge replication fills the HIGH side starting from index
    // `count`, not `count + offset`. When the LOW side was padded -- which is what
    // happens for a hypocentre near a fault edge -- the fault's own last rows and
    // columns sit at indices at or above `count`, so the replication writes over
    // them with values from further in.
    //
    // Demonstrated here against the C rather than argued from the source, so the
    // claim is measured. A future fix has to change this test deliberately.
    let (strike_count, dip_count) = (12, 12);
    let speed = seeded_speed(strike_count, dip_count);

    // A hypocentre at dip 0 forces a dip offset of RING_RADIUS + 1 = 3.
    let hypocentre = Hypocentre {
        strike: strike_count / 2,
        dip: 0,
    };
    let (dip_offset, padded_dip, _) = padding(hypocentre.dip, dip_count);
    assert_eq!((dip_offset, padded_dip), (3, 15));

    let mut speeds: Vec<f32> = (0..dip_count)
        .flat_map(|dip| (0..strike_count).map(move |strike| (strike, dip)))
        .map(|(strike, dip)| speed.speed(strike, dip))
        .collect();
    let mut rng_state: i64 = 1;
    let slowness = oracle::padded_slowness(
        &mut speeds,
        strike_count,
        dip_count,
        strike_count,
        padded_dip,
        0,
        dip_offset,
        0.0,
        &mut rng_state,
    );

    // The fault's deepest row lives at padded row `dip_offset + dip_count - 1` = 14.
    // Uncorrupted, it would hold 1/speed of the fault's row 11.
    let deepest = slowness[strike_count + 14 * strike_count - strike_count];
    let intended = f64::from(1.0 / speed.speed(0, dip_count - 1));

    // It does not. The replication overwrote rows 12..15 from padded row 11, which
    // is the fault's row 8.
    let clobbered_with = f64::from(1.0 / speed.speed(0, 11 - dip_offset));
    assert!(
        (deepest - clobbered_with).abs() < 1e-12,
        "expected the clobbered value {clobbered_with}, got {deepest}"
    );
    assert!(
        (deepest - intended).abs() > 1e-9,
        "the deepest row was NOT clobbered -- has get_rslow_stretch been fixed?"
    );
}

proptest! {
    #[test]
    fn rupture_times_match_for_arbitrary_faults_and_hypocentres(
        strike_count in 4usize..30,
        dip_count in 4usize..30,
        strike_fraction in 0.0f64..1.0,
        dip_fraction in 0.0f64..1.0,
        spacing_km in 0.2f64..3.0,
    ) {
        let speed = seeded_speed(strike_count, dip_count);
        let hypocentre = Hypocentre {
            strike: index_from(strike_fraction, strike_count),
            dip: index_from(dip_fraction, dip_count),
        };

        let expected = reference(&speed, hypocentre, spacing_km);
        let produced = Wavefront2d::new().solve(&speed, hypocentre, spacing_km);

        for (offset, (got, want)) in produced.as_slice().iter().zip(&expected).enumerate() {
            prop_assert_eq!(got.to_bits(), want.to_bits(), "at {}", offset);
        }
    }
}

// Deliberately not asserted:
//
// - That the time at the hypocentre is exactly zero. It is, but that is the
//   solver's property and belongs in a contract test alongside the fast-marching
//   implementation, where both have to satisfy it.
// - That times increase monotonically away from the hypocentre. Same reason -- and
//   it is a claim about the physics, which is what makes it a Stage 2 property
//   rather than a parity one.
