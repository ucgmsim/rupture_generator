//! Turning an offset in kilometres into a longitude and latitude.
//!
//! The fault's segments are described by their top-edge centre, but what the SRF
//! header wants is the centre of the *plane* — half a fault width down dip, projected
//! onto the surface. That is an offset in kilometres from a known point, and turning
//! it back into coordinates is the whole of the geodesy in this program.
//!
//! # Nothing calls this, and nothing is scheduled to
//!
//! Not the library, not `crates/core`, not the Python package — `assemble.py` states
//! it: *"There is no geodesy here and no projection."* The subfault coordinates arrive
//! from the caller.
//!
//! This used to be justified by `rupture_generator/geometry.py`, a stub that would one
//! day place subfaults from a fault definition. That stub is deleted — it was pre-port
//! scaffold with `pass` for a body and no importers — so the justification is now
//! narrower and worth stating plainly: **this module is unused**, it is the correct
//! replacement for a measurably wrong thing (`set_ll`, below), and `geographiclib-rs`
//! is a dependency held for it alone.
//!
//! Keep it while placing subfaults on an ellipsoid is a thing the package might do.
//! Delete it, and the dependency with it, the moment that stops being true.
//!
//! # What was here and is not
//!
//! `LocalFlatEarth` reproduced genslip's `set_ll` bit for bit, for comparison, and
//! nothing else. With the parity suite retired it had no purpose left, so it went —
//! taking the last use of `genslip-oracle` outside its own crate with it.
//!
//! It was wrong by 0.93 m at 1 km, 20 m at 10 km, 264 m at 50 km, **944 m at 100 km**
//! and 3.5 km at 200 km: quadratic in distance, as a linearisation is.
//! `tests/geodesy.rs` keeps the full table.
//!
//! [`Wgs84Geodesic`] solves the direct geodesic problem properly — the destination of
//! a given distance along a given azimuth on the WGS84 ellipsoid, by Karney's
//! algorithm — and is the only implementation left.
//!
//! # What was wrong with the original, beyond being a linearisation
//!
//! Three more things, still visible in the corpus: genslip recomputes each plane's
//! centre with `set_ll` rather than reading the positions it was given, and
//! `test_corpus.py::TestTheGeometryDivergence` records the result at 43 m on a
//! crustal fault and 1.9 km at subduction scale.
//!
//! 1. **The ellipsoid is not WGS84.** `a = 6378.139 km` and `1/f = 298.256`, roughly
//!    the 1964 IAU figure. WGS84 is `6378.137` and `298.257223563`. Everything else
//!    in the pipeline — the SRF the workflow consumes, the station coordinates —
//!    is WGS84.
//! 2. **Geodetic and geocentric latitude are mixed.** `lat0` is converted to
//!    geocentric by `atan((1-f)*tan(lat))`, and its cosine is then used as though it
//!    were the geodetic parallel radius.
//! 3. **The degrees-to-radians constant is truncated** to `0.017453293`, which is
//!    about 2e-9 short.
//!
//! (orig. `set_ll`, misc.c:106)

use geographiclib_rs::{DirectGeodesic, Geodesic};

/// A point on the surface, in degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub longitude_deg: f64,
    pub latitude_deg: f64,
}

/// An offset in kilometres, in the local horizontal plane.
#[derive(Clone, Copy, Debug)]
pub struct Offset {
    pub north_km: f64,
    pub east_km: f64,
}

/// Converts a kilometre offset from a point into a new point.
///
/// Implementations agree on the meaning — the destination reached by travelling
/// `north_km` north and `east_km` east of the origin — and differ in how faithfully
/// they honour the shape of the Earth.
pub trait Geodesy {
    /// The point `offset` away from `origin`.
    fn offset(&self, origin: Point, offset: Offset) -> Point;
}

/// The direct geodesic problem on WGS84, by Karney's algorithm.
///
/// Accurate to round-off at any distance, on the ellipsoid everything downstream
/// already uses.
#[derive(Clone, Debug)]
pub struct Wgs84Geodesic {
    geodesic: Geodesic,
}

impl Wgs84Geodesic {
    /// A solver on the WGS84 ellipsoid.
    #[must_use]
    pub fn new() -> Self {
        Self {
            geodesic: Geodesic::wgs84(),
        }
    }
}

impl Default for Wgs84Geodesic {
    fn default() -> Self {
        Self::new()
    }
}

impl Geodesy for Wgs84Geodesic {
    fn offset(&self, origin: Point, offset: Offset) -> Point {
        let distance_km = offset.north_km.hypot(offset.east_km);
        if distance_km == 0.0 {
            return origin;
        }

        // Azimuth is measured clockwise from north, which is `atan2(east, north)` --
        // not the `atan2(north, east)` of ordinary mathematical convention.
        let azimuth_deg = offset.east_km.atan2(offset.north_km).to_degrees();

        let (latitude_deg, longitude_deg) = self.geodesic.direct(
            origin.latitude_deg,
            origin.longitude_deg,
            azimuth_deg,
            distance_km * 1000.0,
        );

        Point {
            longitude_deg,
            latitude_deg,
        }
    }
}
