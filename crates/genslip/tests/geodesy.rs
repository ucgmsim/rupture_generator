//! The flat-earth approximation reproduces genslip's, and the geodesic does better.
//!
//! Two claims, deliberately separate.
//!
//! `LocalFlatEarth` is pinned bit-for-bit against `set_ll`, because it exists only to
//! reproduce it.
//!
//! `Wgs84Geodesic` is not, and could not be — it is a different calculation on a
//! different ellipsoid. What is asserted of it is that it is *correct*: it round
//! trips, it agrees with the flat-earth model where that model is valid, and it
//! diverges from it in the direction and by the magnitude that theory predicts.
//!
//! The last of those is the point. Replacing a wrong calculation with a right one
//! moves every SRF plane header, and the useful question afterwards is whether it
//! moved by the amount it should have. That number is measured here, before the swap.
#![cfg(feature = "fftw")]

use genslip::geodesy::{Geodesy, LocalFlatEarth, Offset, Point, Wgs84Geodesic};
use genslip_oracle::field as oracle;
use proptest::prelude::*;

/// Latitudes spanning New Zealand and beyond, including both hemispheres and the
/// equator — the original has a documented southern-hemisphere history.
const ORIGINS: [(f64, f64); 6] = [
    (174.76, -36.85), // Auckland
    (172.64, -43.53), // Christchurch
    (166.0, -45.9),   // Fiordland, the deep south
    (0.0, 0.0),       // equator, prime meridian
    (-118.24, 34.05), // northern hemisphere
    (179.9, -37.0),   // hard against the antimeridian
];

fn check_matches_oracle(origin: (f64, f64), offset: Offset, label: &str) {
    #[expect(clippy::cast_possible_truncation, reason = "the C takes floats")]
    let (want_longitude, want_latitude) = oracle::offset_point(
        origin.0 as f32,
        origin.1 as f32,
        offset.north_km as f32,
        offset.east_km as f32,
    );

    let produced = LocalFlatEarth.offset(
        Point {
            longitude_deg: origin.0,
            latitude_deg: origin.1,
        },
        offset,
    );

    #[expect(
        clippy::cast_possible_truncation,
        reason = "comparing at the C's width"
    )]
    let (got_longitude, got_latitude) =
        (produced.longitude_deg as f32, produced.latitude_deg as f32);

    assert_eq!(
        got_longitude.to_bits(),
        want_longitude.to_bits(),
        "{label}: longitude {got_longitude} vs {want_longitude}"
    );
    assert_eq!(
        got_latitude.to_bits(),
        want_latitude.to_bits(),
        "{label}: latitude {got_latitude} vs {want_latitude}"
    );
}

#[test]
fn the_flat_earth_model_matches_the_original() {
    // Offsets from a metre to half a fault width on a wide subduction interface.
    for offset in [
        Offset {
            north_km: 0.0,
            east_km: 0.0,
        },
        Offset {
            north_km: 0.001,
            east_km: 0.0,
        },
        Offset {
            north_km: 5.0,
            east_km: -3.0,
        },
        Offset {
            north_km: -40.0,
            east_km: 25.0,
        },
        Offset {
            north_km: 120.0,
            east_km: -80.0,
        },
    ] {
        for origin in ORIGINS {
            check_matches_oracle(origin, offset, &format!("{origin:?} + {offset:?}"));
        }
    }
}

proptest! {
    #[test]
    fn the_flat_earth_model_matches_for_arbitrary_offsets(
        longitude in -180.0f64..180.0,
        latitude in -85.0f64..85.0,
        north_km in -200.0f64..200.0,
        east_km in -200.0f64..200.0,
    ) {
        #[expect(clippy::cast_possible_truncation, reason = "the C takes floats")]
        let (want_longitude, want_latitude) = oracle::offset_point(
            longitude as f32, latitude as f32, north_km as f32, east_km as f32,
        );
        let produced = LocalFlatEarth.offset(
            Point { longitude_deg: longitude, latitude_deg: latitude },
            Offset { north_km, east_km },
        );
        #[expect(clippy::cast_possible_truncation, reason = "comparing at the C's width")]
        let (got_longitude, got_latitude) =
            (produced.longitude_deg as f32, produced.latitude_deg as f32);

        prop_assert_eq!(got_longitude.to_bits(), want_longitude.to_bits());
        prop_assert_eq!(got_latitude.to_bits(), want_latitude.to_bits());
    }
}

/// Geodesic distance, for checking the solver against what it was asked for.
fn separation_km(a: Point, b: Point) -> f64 {
    use geographiclib_rs::InverseGeodesic as _;

    let geodesic = geographiclib_rs::Geodesic::wgs84();
    let metres: f64 = geodesic.inverse(
        a.latitude_deg,
        a.longitude_deg,
        b.latitude_deg,
        b.longitude_deg,
    );
    metres / 1000.0
}

#[test]
fn the_geodesic_travels_the_distance_it_was_asked_for() {
    // The property the flat-earth model fails at range: the destination should be
    // exactly the requested distance away, measured on the ellipsoid.
    for origin in ORIGINS {
        let origin = Point {
            longitude_deg: origin.0,
            latitude_deg: origin.1,
        };
        for offset in [
            Offset {
                north_km: 5.0,
                east_km: -3.0,
            },
            Offset {
                north_km: -40.0,
                east_km: 25.0,
            },
            Offset {
                north_km: 120.0,
                east_km: -80.0,
            },
        ] {
            let requested = offset.north_km.hypot(offset.east_km);
            let reached = separation_km(origin, Wgs84Geodesic::new().offset(origin, offset));
            assert!(
                (reached - requested).abs() < 1e-6,
                "{origin:?} + {offset:?}: travelled {reached} km, asked for {requested}"
            );
        }
    }
}

#[test]
fn a_zero_offset_is_the_origin_for_both() {
    for origin in ORIGINS {
        let origin = Point {
            longitude_deg: origin.0,
            latitude_deg: origin.1,
        };
        let zero = Offset {
            north_km: 0.0,
            east_km: 0.0,
        };
        assert_eq!(Wgs84Geodesic::new().offset(origin, zero), origin);

        // The flat-earth model reaches the origin too, but only to f32 -- it stores
        // the result at that width.
        let flat = LocalFlatEarth.offset(origin, zero);
        assert!((flat.latitude_deg - origin.latitude_deg).abs() < 1e-5);
        assert!((flat.longitude_deg - origin.longitude_deg).abs() < 1e-5);
    }
}

/// How far the two models disagree — measured, and recorded for the swap.
///
/// Not a pass/fail claim about either. It is the baseline: replacing `set_ll` moves
/// every SRF plane header, and this says by how much, in metres, as a function of
/// how far the offset reaches.
#[test]
fn the_two_models_diverge_as_the_offset_grows() {
    let origin = Point {
        longitude_deg: 172.64,
        latitude_deg: -43.53,
    };
    let mut previous = 0.0_f64;

    println!("offset (km)   disagreement (m)");
    for distance_km in [1.0_f64, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0] {
        // Due north-east, so both components are exercised.
        let offset = Offset {
            north_km: distance_km / std::f64::consts::SQRT_2,
            east_km: distance_km / std::f64::consts::SQRT_2,
        };
        let flat = LocalFlatEarth.offset(origin, offset);
        let geodesic = Wgs84Geodesic::new().offset(origin, offset);
        let disagreement_m = separation_km(flat, geodesic) * 1000.0;

        println!("{distance_km:>9.1}   {disagreement_m:>16.2}");

        assert!(
            disagreement_m >= previous,
            "disagreement fell from {previous} m to {disagreement_m} m at {distance_km} km; \
             the error should grow with distance"
        );
        previous = disagreement_m;
    }

    // At a subfault-scale offset the two are interchangeable, which is why the
    // approximation survived: it is only wrong where the fault is wide.
    let small = Offset {
        north_km: 0.5,
        east_km: 0.5,
    };
    let close = separation_km(
        LocalFlatEarth.offset(origin, small),
        Wgs84Geodesic::new().offset(origin, small),
    ) * 1000.0;
    assert!(
        close < 5.0,
        "disagreement at 0.7 km is {close} m, expected under 5"
    );
}

// Deliberately not asserted:
//
// - That the two models agree to any fixed tolerance. They do not, and the whole
//   reason for the geodesic is that one of them is wrong.
// - Anything about the flat-earth model's accuracy. It is reproduced, not defended.
