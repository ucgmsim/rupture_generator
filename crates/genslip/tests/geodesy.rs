//! The geodesic is correct, and the flat earth it replaced was not.
//!
//! # Nothing calls this yet, and that is deliberate
//!
//! `geodesy` has no caller in the library, in `crates/core`, or in the Python package
//! — `assemble.py` says so outright: *"There is no geodesy here and no projection. The
//! subfault coordinates come from"* the caller. The module exists because
//! `rupture_generator/geometry.py` is a stub that will eventually need to place
//! subfaults from a fault definition, and this is what it will place them with.
//!
//! That made the "swap `Wgs84Geodesic` in for the flat earth" item a false premise:
//! there was nothing to swap, only a compatibility shim to delete. `LocalFlatEarth`
//! reproduced genslip's `set_ll` bit for bit, existed for no other reason, and is now
//! gone along with the last use of `genslip-oracle` outside its own crate.
//!
//! # What the flat earth was wrong by, recorded before deleting it
//!
//! Measured from Christchurch, offset due north-east so both components are
//! exercised, against the WGS84 geodesic:
//!
//! | offset | disagreement |
//! | --- | --- |
//! | 1 km | 0.93 m |
//! | 5 km | 7.8 m |
//! | 10 km | 19.9 m |
//! | 25 km | 79.7 m |
//! | 50 km | 264 m |
//! | **100 km** | **944 m** |
//! | 200 km | 3.54 km |
//!
//! Quadratic in distance, which is what a linearisation gives. It is also why the
//! approximation survived so long: at the half-kilometre scale of a subfault the two
//! models agree to under a metre, and it is only wrong where the fault is wide.
//!
//! genslip's own version of this error is still visible in the corpus, in the plane
//! header it recomputes rather than reads — 43 m on a crustal fault, 1.9 km at
//! subduction scale. `test_corpus.py::TestTheGeometryDivergence` records it. Four
//! separate faults compound there: the linearisation itself, a 1964 IAU ellipsoid
//! rather than WGS84, geodetic and geocentric latitude mixed, and `π/180` truncated to
//! `0.017453293`.

use genslip::geodesy::{Geodesy, Offset, Point, Wgs84Geodesic};
use proptest::prelude::*;

/// Latitudes spanning New Zealand and beyond, including both hemispheres, the equator
/// and the antimeridian.
const ORIGINS: [(f64, f64); 6] = [
    (174.76, -36.85), // Auckland
    (172.64, -43.53), // Christchurch
    (166.0, -45.9),   // Fiordland, the deep south
    (0.0, 0.0),       // equator, prime meridian
    (-118.24, 34.05), // northern hemisphere
    (179.9, -37.0),   // hard against the antimeridian
];

fn point(origin: (f64, f64)) -> Point {
    Point {
        longitude_deg: origin.0,
        latitude_deg: origin.1,
    }
}

/// Great-circle separation in kilometres, by the haversine formula.
///
/// An independent calculation rather than a second call into `geographiclib` — using
/// the library to check the library would prove nothing.
///
/// It is a **sphere**, so it disagrees with an ellipsoidal geodesic by however far the
/// local radius of curvature is from the mean: the meridional radius runs from 6335 km
/// at the equator to 6400 km at the poles, about ±0.5%. That sets the tolerance below,
/// and it is a property of the reference rather than of the code under test. A tighter
/// bound would be asserting that the Earth is round.
fn separation_km(first: Point, second: Point) -> f64 {
    const EARTH_KM: f64 = 6_371.008_771_4;
    let (lat1, lat2) = (
        first.latitude_deg.to_radians(),
        second.latitude_deg.to_radians(),
    );
    let delta_lat = lat2 - lat1;
    let delta_lon = (second.longitude_deg - first.longitude_deg).to_radians();
    let haversine =
        (delta_lat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (delta_lon / 2.0).sin().powi(2);
    2.0 * EARTH_KM * haversine.sqrt().asin()
}

/// The geodesic travels the distance it was asked to travel.
///
/// The defining property, and the one the flat earth fails: an offset of `d` km must
/// land `d` km away. One percent, because the reference is spherical — see
/// `separation_km`. Loose enough to be honest and tight enough to catch what would
/// actually go wrong here: a kilometre/metre confusion, a factor of two, an azimuth
/// measured from the wrong axis. Transposition is `the_axes_are_not_transposed`.
#[test]
fn the_geodesic_travels_the_distance_it_was_asked_for() {
    for origin in ORIGINS {
        for distance_km in [1.0_f64, 10.0, 100.0] {
            for (north, east) in [
                (1.0, 0.0),
                (0.0, 1.0),
                (
                    std::f64::consts::FRAC_1_SQRT_2,
                    std::f64::consts::FRAC_1_SQRT_2,
                ),
                (-0.6, 0.8),
            ] {
                let offset = Offset {
                    north_km: distance_km * north,
                    east_km: distance_km * east,
                };
                let travelled = separation_km(
                    point(origin),
                    Wgs84Geodesic::new().offset(point(origin), offset),
                );
                assert!(
                    (travelled - distance_km).abs() < 0.01 * distance_km,
                    "{origin:?} + {offset:?}: travelled {travelled} km, asked for \
                     {distance_km}"
                );
            }
        }
    }
}

/// A zero offset is the origin, exactly.
#[test]
fn a_zero_offset_is_the_origin() {
    for origin in ORIGINS {
        let produced = Wgs84Geodesic::new().offset(
            point(origin),
            Offset {
                north_km: 0.0,
                east_km: 0.0,
            },
        );
        assert!((produced.longitude_deg - origin.0).abs() < 1e-12);
        assert!((produced.latitude_deg - origin.1).abs() < 1e-12);
    }
}

/// Due north moves latitude only, and due east moves longitude only.
///
/// The axis convention, which is the thing a transposition would break — and which
/// every distance check above is blind to, because a swapped pair still travels the
/// right distance.
#[test]
fn the_axes_are_not_transposed() {
    let origin = point((172.64, -43.53));

    let north = Wgs84Geodesic::new().offset(
        origin,
        Offset {
            north_km: 50.0,
            east_km: 0.0,
        },
    );
    assert!(
        (north.longitude_deg - origin.longitude_deg).abs() < 1e-9,
        "a due-north offset moved longitude"
    );
    assert!(
        north.latitude_deg > origin.latitude_deg,
        "a due-north offset did not go north"
    );

    let east = Wgs84Geodesic::new().offset(
        origin,
        Offset {
            north_km: 0.0,
            east_km: 50.0,
        },
    );
    assert!(
        east.longitude_deg > origin.longitude_deg,
        "a due-east offset did not go east"
    );
    // Latitude moves a little on an east-west geodesic away from the equator -- the
    // path is not a parallel -- so this is bounded rather than zero.
    assert!(
        (east.latitude_deg - origin.latitude_deg).abs() < 0.05,
        "a due-east offset moved latitude by {} degrees",
        east.latitude_deg - origin.latitude_deg
    );
}

proptest! {
    /// Any origin, any offset: the distance travelled is the distance asked for.
    #[test]
    fn the_distance_holds_for_arbitrary_offsets(
        longitude in -179.0f64..179.0,
        latitude in -70.0f64..70.0,
        north_km in -150.0f64..150.0,
        east_km in -150.0f64..150.0,
    ) {
        let origin = Point { longitude_deg: longitude, latitude_deg: latitude };
        let offset = Offset { north_km, east_km };
        let asked = north_km.hypot(east_km);
        prop_assume!(asked > 0.1);

        let travelled = separation_km(origin, Wgs84Geodesic::new().offset(origin, offset));
        prop_assert!(
            (travelled - asked).abs() < 0.012 * asked,
            "travelled {} km, asked for {}", travelled, asked
        );
    }
}

// Deliberately not asserted:
//
// - Anything against `set_ll`. `LocalFlatEarth` existed only to reproduce it and is
//   gone; the numbers it was wrong by are in the module docstring above, which is the
//   part worth keeping.
