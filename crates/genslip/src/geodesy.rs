//! Turning an offset in kilometres into a longitude and latitude.
//!
//! The fault's segments are described by their top-edge centre, but what the SRF
//! header wants is the centre of the *plane* — half a fault width down dip, projected
//! onto the surface. That is an offset in kilometres from a known point, and turning
//! it back into coordinates is the whole of the geodesy in this program.
//!
//! # The implementation is swappable, and one of them is wrong
//!
//! [`Geodesy`] is the trait, matching [`crate::rng::DrawSource`], [`crate::fft::Fft`]
//! and [`crate::rupture::EikonalSolver`].
//!
//! [`LocalFlatEarth`] is genslip's, reproduced for comparison. It is a tangent-plane
//! linearisation: kilometres per degree are evaluated once at the origin's latitude
//! and the offset is then applied linearly. Exact at zero distance and increasingly
//! wrong away from it.
//!
//! [`Wgs84Geodesic`] solves the direct geodesic problem properly — the destination of
//! a given distance along a given azimuth on the WGS84 ellipsoid, by Karney's
//! algorithm. It is what new work should use.
//!
//! # What is actually wrong with the original
//!
//! Four things, in rough order of how much they matter:
//!
//! 1. **It is a linearisation.** The offsets here are half a fault width projected
//!    to the surface, which for a wide subduction interface is tens of kilometres.
//!    The error grows with the square of the distance.
//! 2. **The ellipsoid is not WGS84.** `a = 6378.139 km` and `1/f = 298.256`, roughly
//!    the 1964 IAU figure. WGS84 is `6378.137` and `298.257223563`. Everything else
//!    in the pipeline — the SRF the workflow consumes, the station coordinates —
//!    is WGS84.
//! 3. **Geodetic and geocentric latitude are mixed.** `lat0` is converted to
//!    geocentric by `atan((1-f)*tan(lat))`, and its cosine is then used as though it
//!    were the geodetic parallel radius.
//! 4. **The degrees-to-radians constant is truncated** to `0.017453293`, which is
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

/// genslip's tangent-plane approximation.
///
/// Kept to reproduce the original exactly. Prefer [`Wgs84Geodesic`].
#[derive(Clone, Copy, Debug, Default)]
pub struct LocalFlatEarth;

impl LocalFlatEarth {
    /// Truncated in the original, and reproduced. About 2e-9 short of the true value.
    const RADIANS_PER_DEGREE: f64 = 0.017_453_293;
    /// Equatorial radius in km. Roughly the 1964 IAU figure, not WGS84's 6378.137.
    const RADIUS_KM: f64 = 6378.139;
    /// Inverse flattening. WGS84's is 298.257223563.
    const INVERSE_FLATTENING: f64 = 298.256;

    /// Kilometres per degree of latitude and of longitude, at `latitude_deg`.
    ///
    /// Returned as `(north, east)`, in single precision: the original computes both
    /// in double and stores them into `float`s before dividing by them, so the
    /// narrowing happens here rather than at the result.
    fn kilometres_per_degree(latitude_deg: f32) -> (f32, f32) {
        let flattening = 1.0 / Self::INVERSE_FLATTENING;
        let eccentricity_squared = 2.0 * flattening - flattening * flattening;
        let second_eccentricity_squared =
            eccentricity_squared / ((1.0 - flattening) * (1.0 - flattening));

        // Geocentric latitude -- and then used as though it were geodetic, which is
        // one of the reasons this is only an approximation.
        let reduced = ((1.0 - flattening)
            * (f64::from(latitude_deg) * Self::RADIANS_PER_DEGREE).tan())
        .atan();

        // The original narrows these two to `float` before using them.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the narrowing seam: C declares cosA and sinA as float"
        )]
        let cosine = reduced.cos() as f32;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the narrowing seam: C declares cosA and sinA as float"
        )]
        let sine = reduced.sin() as f32;
        let (cosine, sine) = (f64::from(cosine), f64::from(sine));

        let denominator = (1.0 / (1.0 + second_eccentricity_squared * sine * sine)).sqrt();
        let east = Self::RADIANS_PER_DEGREE * Self::RADIUS_KM * cosine * denominator;
        let north = Self::RADIANS_PER_DEGREE
            * Self::RADIUS_KM
            * (1.0
                + second_eccentricity_squared * sine * sine * (2.0 + second_eccentricity_squared))
                .sqrt()
            * denominator
            * denominator
            * denominator;

        #[expect(
            clippy::cast_possible_truncation,
            reason = "the narrowing seam: C declares kperd_n and kperd_e as float"
        )]
        let scales = (north as f32, east as f32);
        scales
    }
}

impl Geodesy for LocalFlatEarth {
    fn offset(&self, origin: Point, offset: Offset) -> Point {
        // Every argument the original takes is a `float`, so anything finer than
        // that never reaches its arithmetic. Narrowing at the boundary is what makes
        // this agree with the C for inputs a caller happens to hold in `f64`.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the narrowing seam: the C's arguments are all float"
        )]
        let (north_km, east_km, origin_latitude, origin_longitude) = (
            offset.north_km as f32,
            offset.east_km as f32,
            origin.latitude_deg as f32,
            origin.longitude_deg as f32,
        );

        let (north_per_degree, east_per_degree) = Self::kilometres_per_degree(origin_latitude);

        Point {
            latitude_deg: f64::from(north_km / north_per_degree + origin_latitude),
            longitude_deg: f64::from(east_km / east_per_degree + origin_longitude),
        }
    }
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
