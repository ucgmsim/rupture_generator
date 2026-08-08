//! What a caller can get wrong, said out loud.
//!
//! # Why this exists
//!
//! Until now the crate had no error type. Bad input was met one of two ways, and both
//! were wrong in the same direction:
//!
//! * an `assert!`, which is a panic — fine for an invariant the crate is responsible
//!   for, and not fine for a number a caller chose;
//! * or, worse, a **silent fallback**. [`crate::source::geometry_correction`] met a
//!   dip outside 0–90° by returning a correction factor of zero, which is not an
//!   error value: it is a valid-looking rupture with the geometry correction switched
//!   off, and nothing downstream could tell. That is the shape this module exists to
//!   remove.
//!
//! # Where the line is
//!
//! **Validate at the boundary, assert inside.** The entry points —
//! [`crate::realisation::generate`], [`crate::realisation::point_source`], and the
//! constructors that take numbers a caller picked — return [`Result`]. Everything
//! below them keeps its assertions, because by then the values came from this crate
//! rather than from outside it, and a failure is a bug here rather than a mistake
//! there. Those assertions say so at the site.
//!
//! The distinction matters for what a message can promise. An `Error` names the input
//! and what was wrong with it, so a caller can fix it. An assertion names an internal
//! invariant, so *we* can fix it.

use thiserror::Error;

/// A rupture that could not be generated from the inputs given.
///
/// Every variant names the input at fault and the constraint it broke. None of them
/// is recoverable in the sense of "try again" — they all mean the description of the
/// earthquake is inconsistent, so the fix is upstream.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// A grid extent is zero, or a padded extent is odd.
    ///
    /// The spectral generators address the Nyquist row and column directly, at
    /// `strike_count / 2` and `dip_count / 2`, which only means anything for an even
    /// extent. This is not a restriction the port invents: genslip sizes all four of
    /// its padded grids and then rounds each up to even
    /// (`genslip_v5.6.2.c:1471-1490`), so an odd extent means the sizing is wrong
    /// rather than that the grid should cope with it.
    #[error("{what} extent is {strike_count}x{dip_count}, which must be {constraint}")]
    Extent {
        /// Which grid: `"fault"`, `"padded"`, `"spectrum"`.
        what: &'static str,
        strike_count: usize,
        dip_count: usize,
        /// The constraint in words, e.g. `"non-zero"` or `"even"`.
        constraint: &'static str,
    },

    /// The fault does not fit inside the padded grid it is generated on.
    #[error(
        "a {fault_strike}x{fault_dip} fault does not fit in a \
         {padded_strike}x{padded_dip} grid"
    )]
    FaultLargerThanPadding {
        fault_strike: usize,
        fault_dip: usize,
        padded_strike: usize,
        padded_dip: usize,
    },

    /// A per-subfault or per-row array is the wrong length.
    ///
    /// `what` is the argument's own name, so the message points at the thing to fix
    /// rather than at the check that caught it.
    #[error("{what} has {found} entries, expected {expected}")]
    Shape {
        what: &'static str,
        found: usize,
        expected: usize,
    },

    /// A quantity that must be strictly positive is not.
    ///
    /// Sample intervals, rise times, wavelengths and magnitudes. Zero is as wrong as
    /// negative for all of them, and both give a division rather than a diagnosis.
    #[error("{what} is {value}, which must be strictly positive")]
    NotPositive { what: &'static str, value: f64 },

    /// The average dip is outside the range a fault plane can have.
    ///
    /// **The variant the review asked for.** `geometry_correction` used to answer a
    /// dip of 120° with a correction factor of zero, which propagates as a rupture
    /// whose rise time and rupture speed are silently uncorrected. A dip outside
    /// 0–90° is not a fault this program describes, and saying so is the only honest
    /// answer.
    #[error("average dip is {degrees} degrees, which must be in 0..=90")]
    DipOutOfRange { degrees: f64 },

    /// A mesh has too few of something to be a surface at all.
    ///
    /// A face asked to be cut into no cells along one of its axes.
    ///
    /// Notably *not* an empty or one-point fault: [`crate::mesh::Fault`] holds its first
    /// plane separately from the rest, so a fault with no planes cannot be constructed
    /// and there is nothing here to check.
    #[error("{what}: found {found}, need at least {needed}")]
    MeshTooSmall {
        /// What was short: `"strike cells"`, `"dip cells"`.
        what: &'static str,
        found: usize,
        needed: usize,
    },

    /// A fault surface would reach above the ground.
    ///
    /// Depth is measured downwards, so a negative one is in the air. Reached by a top
    /// depth given as negative, or by a point source shallower than half its own
    /// subfault -- genslip's answer to the latter is to floor the top depth at zero,
    /// which silently shrinks the subfault.
    #[error("the surface reaches {depth_km} km, which is above the ground")]
    AboveSurface { depth_km: f64 },

    /// A patch indexes a vertex the mesh does not have.
    ///
    /// Only reachable from [`crate::mesh::RefinedMesh::from_parts`], which is how a mesh
    /// arrives from a *file*. Refinement builds its own indices and cannot produce one.
    #[error("a patch indexes vertex {index}, but the mesh has {vertices}")]
    DanglingVertex { index: usize, vertices: usize },

    /// A mesh's cells are not all the same size along one of its axes.
    ///
    /// The spectral generators transform on a uniform grid and the eikonal sweep steps
    /// on one, so a mesh that is not uniform cannot be generated on. Refused here
    /// rather than averaged, because an average would put every subfault somewhere
    /// nobody chose.
    ///
    /// Refining a plane cannot produce one -- bilinear subdivision of a parallelogram is
    /// uniform by construction. This guards a mesh that arrived from a *file*, which may
    /// have been hand-edited or written by an importer, and so has to be checked where
    /// the data enters rather than assumed from how it was usually made.
    #[error("{axis} spacing runs from {smallest_km} km to {largest_km} km, which is not uniform")]
    NonUniformMesh {
        /// `"strike"` or `"dip"`.
        axis: &'static str,
        smallest_km: f64,
        largest_km: f64,
    },

    /// A position given in in-fault coordinates is off the mesh.
    ///
    /// Both coordinates are arc lengths from an edge — see
    /// [`crate::mesh::PatchView::cell_index`] for which edge, which is the part that is
    /// worth reading before assuming.
    #[error("{axis} position {position_km} km is outside a fault {extent_km} km across")]
    PositionOffMesh {
        /// `"strike"` or `"dip"`.
        axis: &'static str,
        position_km: f64,
        extent_km: f64,
    },

    /// The hypocentre is not on the fault.
    #[error("hypocentre ({strike}, {dip}) is outside a {strike_count}x{dip_count} fault")]
    HypocentreOffFault {
        strike: usize,
        dip: usize,
        strike_count: usize,
        dip_count: usize,
    },

    /// An edge taper reaches past the opposite edge of the fault.
    ///
    /// `DEFECTS.md` 7: the original does not check, and writes outside its array when
    /// this happens. There is no way to reproduce an out-of-bounds write in safe Rust
    /// and no reason to want one.
    #[error(
        "{edge} taper of {fraction} reaches {width} subfaults across a \
         {extent}-subfault fault"
    )]
    TaperTooWide {
        edge: &'static str,
        fraction: f64,
        width: usize,
        extent: usize,
    },

    /// A velocity model with no layers.
    #[error("a velocity model needs at least one layer")]
    EmptyVelocityModel,

    /// The eikonal sweep did not converge.
    ///
    /// Fast sweeping settles in a fixed number of rounds independent of mesh size —
    /// one round in every case measured, and Fomel, Luo & Zhao (2009) report three
    /// for theirs. Exhausting the cap means the speed field has structure the scheme
    /// does not handle, which is a statement about the *medium* rather than about the
    /// solver, so it names the cap and stops.
    #[error("the rupture-speed sweep did not settle in {rounds} rounds")]
    SweepDidNotConverge { rounds: usize },

    /// A subfault the rupture never reaches.
    ///
    /// Almost always a rupture speed of zero somewhere — a zero shear speed in the
    /// velocity model, or a velocity fraction of zero — which makes the travel time
    /// infinite rather than large.
    #[error("subfault ({strike}, {dip}) is never reached; is the rupture speed zero?")]
    Unreachable { strike: usize, dip: usize },
}

impl Error {
    /// `Err(NotPositive)` unless `value` is strictly positive and finite.
    ///
    /// A helper because this is the most common check in the crate and writing it out
    /// eleven times invites eleven slightly different messages. NaN fails: it is not
    /// positive, and a NaN sample interval produces a pulse of NaNs rather than a
    /// complaint.
    ///
    /// # Errors
    ///
    /// [`Error::NotPositive`] if `value` is zero, negative, infinite or NaN.
    pub fn require_positive(what: &'static str, value: f64) -> Result<()> {
        if value > 0.0 && value.is_finite() {
            Ok(())
        } else {
            Err(Self::NotPositive { what, value })
        }
    }
}

/// The crate's result type.
pub type Result<T, E = Error> = std::result::Result<T, E>;
