//! Where the fault is: a surface mesh, and the grid it refines into.
//!
//! Everything else in this crate takes the fault as *already discretised* — arrays of
//! depth, rake and rupture speed on a `(dip, strike)` grid. This is the thing that builds
//! them, from a description of a fault surface a person would recognise.
//!
//! # The pipeline
//!
//! ```text
//! Geometry  ──build──▶  Mesh          ──refine──▶  RefinedMesh
//!                       vertices                   vertices
//!                       quad faces                 patches: (i, j) per face
//! ```
//!
//! The middle object is a mesh in the sense a graphics library means it: a list of
//! vertices and a list of faces indexing into them. That is the abstraction the rest of
//! this is built on, and the reason is what comes next rather than what is here now — a
//! fault surface that curves, or a mesh imported from somewhere else, is the same
//! structure with different numbers in it, and only [`build`] would need to know.
//!
//! Refinement is **bilinear on each face**, and for a fault plane that is exact. A
//! plane's top and bottom edges are parallel and equal in length, because both of its top
//! corners step down dip by the same vector — so the face is a *parallelogram*, and
//! bilinear subdivision puts nodes at exactly the evenly-spaced positions a direct
//! construction would. The abstraction costs nothing.
//!
//! # There is no geodesy here
//!
//! Coordinates are **projected**: eastings and northings in a Cartesian coordinate
//! reference system the modeller chose — NZTM2000, a UTM zone — and depth below the
//! surface. Everything in this module is then plain vector arithmetic in a flat space,
//! and every quantity it derives is an exact identity rather than an approximation: areas
//! sum to length times width, a plane's cells all report the plane's own dip, and a fault
//! cut into twenty pieces has twenty cells of exactly one size.
//!
//! That is not a simplification that costs accuracy — it moves the accuracy question to
//! the one place it can be answered properly. A projection is a stated, invertible choice
//! with known distortion over a known region, made by the person who knows which region
//! the fault is in; and the conversion back to WGS84 happens once, at the boundary that
//! writes an SRF. Doing geodesy *here* instead would mean every derived quantity carried
//! a curvature error, which is measurable — an earlier draft of this module worked on the
//! ellipsoid and its cell areas were wrong by `1.4e-2` on a 60 km subduction interface,
//! with a "uniform" down-dip step that varied by `6.5e-3` — for no gain, because the
//! answer still has to be projected somewhere.
//!
//! **Grid north is not true north.** The strike this module reports is measured from the
//! projection's northing axis, and converting it needs the grid convergence angle added
//! at the same boundary that converts the positions. In NZTM that reaches about two
//! degrees at the country's edges, which is twice `ENGINEERING_RULES.md`'s rake bound, so
//! it is not optional. Dip and rake need no correction: both are measured within the
//! plane, and the convergence cancels.
//!
//! The unit is **kilometres**, not the metres a projected CRS usually reports. This crate
//! works in kilometres everywhere else and depth is already in them; converting at the
//! boundary means the two are never mixed inside a single expression.
//!
//! # Nodes are the geometry; everything else is derived
//!
//! A [`Patch`] holds the *positions* of the grid's corners and nothing else. Cell
//! centres, areas, per-cell strike, per-cell dip and the in-fault arc lengths are all
//! functions of those corners, computed on demand and never stored.
//!
//! That is a deliberate choice against the obvious alternative, which is to store cell
//! centres — what an SRF carries, and what `genslip` is handed in a GSF. Centres are the
//! wrong primitive:
//!
//! * They do not say where the fault *ends*. The area of an edge cell, and the position
//!   of the fault's boundary, both have to be guessed by extrapolating half a cell.
//! * They force strike, dip and area to be stored alongside, because they cannot be
//!   derived — and a stored quantity that could have been derived is a second description
//!   free to drift from the first. This crate makes that argument about `FaultGrid`
//!   already.
//!
//! # The `(i, j)` lattice is the in-fault coordinate system
//!
//! `i` runs along strike and `j` down dip, and a patch is shaped
//! `(dip_count + 1, strike_count + 1)` — `j` first, matching `crate::grid`'s rule that
//! strike varies fastest in memory. [`PatchView::strike_arc_km`] and
//! [`PatchView::dip_arc_km`] give the distance to each node along its own axis, which is
//! what makes a position on the fault expressible as two lengths rather than two indices.
//! A hypocentre is specified that way, and [`PatchView::cell_index`] converts.

use ndarray::{Array1, Array2};

use crate::error::{Error, Result};
use crate::grid::SlipField;
use crate::rupture::Hypocentre;
use crate::slip::SubfaultSpacing;

/// How far a spacing may vary across a patch and still be called uniform.
///
/// Relative. The generators need a uniform grid — `crate::fft` transforms on one and the
/// eikonal sweep steps on one — so a patch whose cells are not all the same size is
/// refused rather than silently averaged.
///
/// **The floor is `f64` round-off**, around `1e-15` relative: bilinear refinement of a
/// parallelogram gives exactly equal steps, and the only spread is the last bits of a sum
/// of them. This is six orders above that.
///
/// **The ceiling is what a person can actually ask for.** A patch this module builds is
/// always uniform, so the only way to reach this check is a mesh that came from somewhere
/// else — a hand-edited file, or an importer written later. The smallest mistake worth
/// catching there is a factor of two.
const UNIFORM_SPACING_TOLERANCE: f64 = 1.0e-9;

/// A horizontal position in the geometry's projected coordinate reference system.
///
/// Kilometres, not the metres a projected CRS reports — see the module note. Which CRS is
/// not recorded here: it is a property of the whole geometry rather than of one point, so
/// it travels with the file.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Projected {
    pub easting_km: f64,
    pub northing_km: f64,
}

impl Projected {
    /// The point `distance_km` away along `bearing_deg`, measured clockwise from north.
    #[must_use]
    pub fn along(self, bearing_deg: f64, distance_km: f64) -> Self {
        let bearing = bearing_deg.to_radians();
        Self {
            easting_km: self.easting_km + distance_km * bearing.sin(),
            northing_km: self.northing_km + distance_km * bearing.cos(),
        }
    }

    /// The bearing from here to `other`, clockwise from north.
    ///
    /// `atan2(east, north)`, not the `atan2(north, east)` of ordinary mathematical
    /// convention: a bearing is measured from the north axis and turns the other way.
    #[must_use]
    pub fn bearing_deg(self, other: Self) -> f64 {
        (other.easting_km - self.easting_km)
            .atan2(other.northing_km - self.northing_km)
            .to_degrees()
    }

    /// How far it is to `other`, in kilometres.
    #[must_use]
    pub fn distance_km(self, other: Self) -> f64 {
        (other.easting_km - self.easting_km).hypot(other.northing_km - self.northing_km)
    }

    /// How far this point is from `origin`.
    ///
    /// The subtraction is exact — differencing two nearby `f64` introduces no error of
    /// its own — so this loses nothing the inputs did not already carry.
    ///
    /// **Done first, before any other arithmetic.** Every construction below then works
    /// on numbers at fault scale rather than at CRS scale, which is where the precision
    /// comes from: `along` on an absolute coordinate rounds its result at ~5,180 km, and
    /// subtracting afterwards keeps the error rather than avoiding it. See [`Vertex`].
    #[must_use]
    pub fn offset_from(self, origin: Self) -> Self {
        Self {
            easting_km: self.easting_km - origin.easting_km,
            northing_km: self.northing_km - origin.northing_km,
        }
    }

    /// This point read as an offset already, at a depth.
    #[must_use]
    pub const fn at_depth(self, depth_km: f64) -> Vertex {
        Vertex {
            east_km: self.easting_km,
            north_km: self.northing_km,
            depth_km,
        }
    }
}

/// A corner of the mesh: how far it is from the mesh's origin, and how deep.
///
/// **An offset, not a position**, and that is load-bearing rather than a convenience.
///
/// A projected coordinate is large and a subfault is small: an NZTM easting runs to
/// ~1,500 km and a northing to ~5,180 km, against a cell of about 1 km. A vertex stored
/// *absolutely* is therefore rounded at CRS scale — an absolute error of
/// `f64::EPSILON * 5180 ~= 5.7e-13 km` — and every cell-scale quantity derived by
/// differencing two of them inherits that as a **relative** error of about `1.2e-12`.
/// Measured, not predicted: it is what `tests/mesh.rs` failed by when this held absolute
/// coordinates.
///
/// Holding offsets keeps every number at fault scale, where the same rounding is
/// `f64::EPSILON * 20 ~= 4e-15 km` and the derived quantities are exact to a part in
/// `1e-15`. The origin is added back once, at the boundary that emits coordinates in the
/// CRS — the same boundary that converts them to WGS84.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex {
    pub east_km: f64,
    pub north_km: f64,
    pub depth_km: f64,
}

/// Which side of the trace the fault dips towards.
///
/// Named by the right-hand rule off the trace direction, which is the convention a fault
/// trace is normally digitised under: walking from the first trace point to the last,
/// [`DipDirection::Right`] dips away to your right.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DipDirection {
    Right,
    Left,
}

impl DipDirection {
    /// Degrees to add to a trace bearing to get the down-dip azimuth.
    const fn quarter_turn(self) -> f64 {
        match self {
            Self::Right => 90.0,
            Self::Left => -90.0,
        }
    }
}

/// One plane of a fault: where its top edge ends, and how it hangs from that edge.
///
/// Geometry only. How finely it is cut is not here — that is an argument to
/// [`Mesh::refine`], because a coarse mesh describes a *surface* and the same surface can
/// be cut at any resolution.
///
/// Where the top edge *begins* is not here either. It is the previous plane's `end`, or
/// the fault's `origin`. See [`Fault`].
#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub end: Projected,
    pub dip_deg: f64,
    pub dip_direction: DipDirection,
    pub bottom_depth_km: f64,
}

/// A fault: one or more planes, connected end to end.
///
/// # Disconnection is not refused here, it is unrepresentable
///
/// A plane carries only where its top edge *ends*. Where it begins is the previous
/// plane's end, or this fault's `origin` — so there is no second copy of a shared corner
/// to disagree with the first, and two planes that do not meet cannot be written down.
///
/// The same trick does the other two invariants. `first` is split from `rest` so that "a
/// fault has at least one plane" is a property of the type; and because each plane
/// contributes exactly one trace point, the plane count and the trace-segment count
/// cannot disagree. Three validations, three error variants and three tests, all of them
/// replaced by a shape.
///
/// `top_depth_km` is shared by every plane, because it is the depth of the trace they all
/// hang from. A segment that starts deeper than its neighbour does not touch it and is a
/// separate fault; making it per-plane would make the shared corner a lie.
#[derive(Clone, Debug)]
pub struct Fault {
    pub origin: Projected,
    pub top_depth_km: f64,
    pub first: Plane,
    pub rest: Vec<Plane>,
}

impl Fault {
    /// The planes, in order.
    fn planes(&self) -> impl Iterator<Item = &Plane> {
        std::iter::once(&self.first).chain(&self.rest)
    }

    /// How many planes there are. At least one, by construction.
    #[must_use]
    pub fn plane_count(&self) -> usize {
        1 + self.rest.len()
    }
}

/// A point source: one cell, of a given size, centred where it is told.
///
/// Not a degenerate fault in its inputs, because a point is described by where its
/// *centre* is rather than where its top edge runs — which is how a catalogue gives one.
#[derive(Clone, Copy, Debug)]
pub struct PointSpec {
    pub centre: Projected,
    pub depth_km: f64,
    pub strike_deg: f64,
    pub dip_deg: f64,
    /// The cell's extent, along strike and down dip alike, in km.
    pub size_km: f64,
}

/// A fault surface, described rather than discretised.
#[derive(Clone, Debug)]
pub enum Geometry {
    Point(PointSpec),
    Fault(Fault),
}

/// How one face is cut up.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cuts {
    pub strike_count: usize,
    pub dip_count: usize,
}

/// A quadrilateral surface mesh: vertices, and faces indexing into them.
///
/// Vertices are offsets from [`Mesh::origin`]. See [`Vertex`] for why.
#[derive(Clone, Debug)]
pub struct Mesh {
    origin: Projected,
    vertices: Vec<Vertex>,
    faces: Vec<[usize; 4]>,
}

/// A mesh cut into cells, with each face's grid recorded.
///
/// Vertices are offsets from [`RefinedMesh::origin`]. See [`Vertex`] for why.
#[derive(Clone, Debug)]
pub struct RefinedMesh {
    origin: Projected,
    vertices: Vec<Vertex>,
    patches: Vec<Patch>,
}

/// One face's refinement: a structured block of vertex indices.
#[derive(Clone, Debug)]
pub struct Patch {
    /// `(dip_count + 1, strike_count + 1)` indices into the mesh's vertices.
    nodes: Array2<usize>,
}

/// A patch and the vertices it indexes, which is what every derived quantity needs.
#[derive(Clone, Copy, Debug)]
pub struct PatchView<'a> {
    vertices: &'a [Vertex],
    nodes: &'a Array2<usize>,
}

/// Node offsets on a patch, one array per axis, shaped `(dip_node, strike_node)`.
///
/// Offsets from the mesh's origin, in kilometres — add it back to get coordinates in the
/// CRS. See [`Vertex`].
#[derive(Clone, Debug)]
pub struct Positions {
    pub east_km: Array2<f64>,
    pub north_km: Array2<f64>,
    pub depth_km: Array2<f64>,
}

/// Cell-centred offsets, one value per subfault, on the `(dip, strike)` grid.
///
/// Offsets from the mesh's origin, like [`Positions`].
#[derive(Clone, Debug)]
pub struct Centres {
    pub east_km: SlipField,
    pub north_km: SlipField,
    pub depth_km: SlipField,
}

// ---------------------------------------------------------------------------
// Vectors. The frame is Cartesian and global, so there is one of these rather
// than one per place.
// ---------------------------------------------------------------------------

/// A position or a direction as `[east, north, down]`, in kilometres.
type Vector = [f64; 3];

/// Straight down.
const DOWN: Vector = [0.0, 0.0, 1.0];

const fn vector(vertex: Vertex) -> Vector {
    [vertex.east_km, vertex.north_km, vertex.depth_km]
}

fn between(from: Vector, to: Vector) -> Vector {
    [to[0] - from[0], to[1] - from[1], to[2] - from[2]]
}

fn add(left: Vector, right: Vector) -> Vector {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

fn scale(vector: Vector, factor: f64) -> Vector {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

fn dot(left: Vector, right: Vector) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: Vector, right: Vector) -> Vector {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn length(vector: Vector) -> f64 {
    dot(vector, vector).sqrt()
}

/// The bearing of a vector's horizontal part, clockwise from north, in `[0, 360)`.
fn bearing_of(vector: Vector) -> f64 {
    normalise_bearing(vector[0].atan2(vector[1]).to_degrees())
}

/// A bearing folded into `[0, 360)`.
fn normalise_bearing(degrees: f64) -> f64 {
    let folded = degrees % 360.0;
    if folded < 0.0 { folded + 360.0 } else { folded }
}

// ---------------------------------------------------------------------------
// Building the coarse mesh
// ---------------------------------------------------------------------------

/// Turn a description of a fault surface into a mesh.
///
/// # Errors
///
/// [`Error::DipOutOfRange`] for a dip outside `(0, 90]`; [`Error::NotPositive`] for a
/// non-positive depth range, size, or trace segment length; [`Error::AboveSurface`] if
/// any part of the surface would end up above ground.
pub fn build(geometry: &Geometry) -> Result<Mesh> {
    match geometry {
        Geometry::Fault(fault) => build_fault(fault),
        Geometry::Point(point) => build_point(*point),
    }
}

/// The horizontal distance a plane reaches from its top edge.
///
/// `width * cos(dip)`, where `width` is `depth_span / sin(dip)` — so the two trigonometric
/// calls collapse into `depth_span / tan(dip)`. A vertical fault gives zero, which is
/// right: `tan(90°)` is enormous rather than infinite in `f64`, so this is a very small
/// number rather than exactly zero, and `a_vertical_fault_is_vertical` pins how small.
fn reach_km(depth_span_km: f64, dip_deg: f64) -> f64 {
    depth_span_km / dip_deg.to_radians().tan()
}

/// The sharpest bend a conforming trace may have, in degrees of deflection.
///
/// The shared column at a bend is stepped down the bisector and stretched by
/// `1 / cos(half the deflection)`, which runs away as the deflection approaches 180. At
/// 120 degrees the stretch is exactly 2, so the two cells flanking the bend would be
/// twice the size of every other cell -- past which calling the grid uniform stops being
/// defensible.
const SHARPEST_BEND_DEG: f64 = 120.0;

/// Whether two planes meeting at a vertex form one continuous surface.
///
/// They do when they hang the same way: the same dip, the same side, and the same depth
/// range. Then their surfaces intersect along a line below the shared vertex, and one
/// column of nodes can lie in both.
///
/// They do not when any of those differ -- which is a fault with a *segment boundary*
/// rather than a bend, and is two surfaces that happen to touch along their top edge.
#[expect(
    clippy::float_cmp,
    reason = "these are values a person wrote down, and the question is whether they \
              wrote the same one. A near miss is a typo, and reading it as a segment \
              boundary -- two surfaces that merely touch -- is the safe answer: it \
              places each plane where its own numbers say, rather than somewhere \
              between them."
)]
fn conforming(near: &Plane, far: &Plane) -> bool {
    near.dip_deg == far.dip_deg
        && near.dip_direction == far.dip_direction
        && near.bottom_depth_km == far.bottom_depth_km
}

/// The down-dip azimuth and step stretch at a trace vertex.
///
/// At the ends, and where two planes do not conform, this is just the owning plane's own
/// quarter turn. At a bend between two that *do*, it is the **bisector** of the two
/// bearings, stretched by `1 / cos(half the deflection)`.
///
/// That stretch is what makes the shared column real. The bisector's projection onto
/// either plane's own down-dip direction is `cos(half the deflection)`, so undoing it
/// puts the point in *both* planes at once -- which is what "one surface" means, and
/// what `tests/mesh.rs::a_bend_shares_its_column_exactly` asserts. Without it the two
/// planes diverge below the vertex: measured at **1.285 km** on the `hope` example, a
/// 20-degree bend on a 14 km-deep fault.
fn step_at_bend(incoming_deg: f64, outgoing_deg: f64, quarter_turn: f64) -> Result<(f64, f64)> {
    let deflection_deg = normalise_bearing(outgoing_deg - incoming_deg + 180.0) - 180.0;
    if deflection_deg.abs() >= SHARPEST_BEND_DEG {
        return Err(Error::TraceDoublesBack { deflection_deg });
    }
    let stretch = 1.0 / (0.5 * deflection_deg.to_radians()).cos();
    Ok((incoming_deg + 0.5 * deflection_deg + quarter_turn, stretch))
}

fn build_fault(fault: &Fault) -> Result<Mesh> {
    if fault.top_depth_km < 0.0 {
        return Err(Error::AboveSurface {
            depth_km: fault.top_depth_km,
        });
    }

    let planes: Vec<&Plane> = fault.planes().collect();

    // The trace: `origin`, then every plane's far end. One more point than there are
    // planes, which is what makes a plane's near end the previous plane's far end.
    //
    // Reduced to offsets from `origin` immediately, so that every `along` below runs at
    // fault scale. See `Projected::offset_from`.
    let origin = fault.origin;
    let trace: Vec<Projected> = std::iter::once(origin)
        .chain(planes.iter().map(|plane| plane.end))
        .map(|point| point.offset_from(origin))
        .collect();

    for (index, plane) in planes.iter().enumerate() {
        require_dip(plane.dip_deg)?;
        Error::require_positive(
            "bottom_depth_km - top_depth_km",
            plane.bottom_depth_km - fault.top_depth_km,
        )?;
        Error::require_positive(
            "trace segment length",
            trace[index].distance_km(trace[index + 1]),
        )?;
    }

    let bearings: Vec<f64> = (0..planes.len())
        .map(|index| trace[index].bearing_deg(trace[index + 1]))
        .collect();

    // Top vertices come first and are shared: the vertex between two planes is one
    // vertex, referred to twice, rather than two that agree to round-off.
    let mut vertices: Vec<Vertex> = trace
        .iter()
        .map(|point| point.at_depth(fault.top_depth_km))
        .collect();
    let mut faces = Vec::with_capacity(planes.len());

    // The bottom vertex at each trace point, for each plane that reaches it. A vertex
    // where two conforming planes meet is placed once and used twice; anywhere else the
    // two planes get their own, because they genuinely are in different places.
    let mut bottom: Vec<[Option<usize>; 2]> = vec![[None, None]; trace.len()];

    for (index, plane) in planes.iter().enumerate() {
        let quarter_turn = plane.dip_direction.quarter_turn();
        let reach = reach_km(plane.bottom_depth_km - fault.top_depth_km, plane.dip_deg);

        for (vertex, side) in [(index, 1_usize), (index + 1, 0_usize)] {
            if bottom[vertex][side].is_some() {
                continue;
            }
            // Does a conforming neighbour share this vertex?
            let neighbour = if vertex == index && index > 0 {
                Some(index - 1)
            } else if vertex == index + 1 && index + 1 < planes.len() {
                Some(index + 1)
            } else {
                None
            };
            let shared = neighbour.filter(|other| conforming(plane, planes[*other]));

            let (azimuth, stretch) = match shared {
                Some(other) => {
                    let (incoming, outgoing) = if other < index {
                        (bearings[other], bearings[index])
                    } else {
                        (bearings[index], bearings[other])
                    };
                    step_at_bend(incoming, outgoing, quarter_turn)?
                }
                None => (bearings[index] + quarter_turn, 1.0),
            };

            vertices.push(
                trace[vertex]
                    .along(azimuth, reach * stretch)
                    .at_depth(plane.bottom_depth_km),
            );
            let placed = vertices.len() - 1;
            bottom[vertex][side] = Some(placed);
            if shared.is_some() {
                // One vertex, both sides of the seam.
                bottom[vertex][1 - side] = Some(placed);
            }
        }

        faces.push([
            index,
            index + 1,
            bottom[index + 1][0].expect("placed above"),
            bottom[index][1].expect("placed above"),
        ]);
    }

    Ok(Mesh {
        origin,
        vertices,
        faces,
    })
}

/// A point source: one face, four vertices.
///
/// A point is given by its centre; a face by its corners. Walking from one to the other
/// is half a cell up dip and half a cell back along strike.
fn build_point(point: PointSpec) -> Result<Mesh> {
    Error::require_positive("size_km", point.size_km)?;
    require_dip(point.dip_deg)?;

    let half = 0.5 * point.size_km;
    let dip = point.dip_deg.to_radians();
    let top_depth_km = point.depth_km - half * dip.sin();
    if top_depth_km < 0.0 {
        return Err(Error::AboveSurface {
            depth_km: top_depth_km,
        });
    }
    let bottom_depth_km = top_depth_km + point.size_km * dip.sin();
    let down_dip = point.strike_deg + DipDirection::Right.quarter_turn();

    // The origin is the point itself, so the whole construction runs from zero and every
    // number in it is at subfault scale. See `Projected::offset_from`.
    let origin = point.centre;
    let centre = origin.offset_from(origin);

    // Up dip is the reverse of down dip.
    let top_centre = centre.along(down_dip, -half * dip.cos());
    let near = top_centre.along(point.strike_deg, -half);
    let far = top_centre.along(point.strike_deg, half);
    let reach = reach_km(bottom_depth_km - top_depth_km, point.dip_deg);

    Ok(Mesh {
        origin,
        vertices: vec![
            near.at_depth(top_depth_km),
            far.at_depth(top_depth_km),
            far.along(down_dip, reach).at_depth(bottom_depth_km),
            near.along(down_dip, reach).at_depth(bottom_depth_km),
        ],
        faces: vec![[0, 1, 2, 3]],
    })
}

fn require_dip(dip_deg: f64) -> Result<()> {
    if dip_deg > 0.0 && dip_deg <= 90.0 {
        Ok(())
    } else {
        Err(Error::DipOutOfRange { degrees: dip_deg })
    }
}

// ---------------------------------------------------------------------------
// Refinement
// ---------------------------------------------------------------------------

impl Mesh {
    /// The point every vertex is measured from.
    #[must_use]
    pub const fn origin(&self) -> Projected {
        self.origin
    }

    /// The mesh's vertices, as offsets from [`Mesh::origin`].
    #[must_use]
    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    /// The mesh's faces, as vertex indices.
    ///
    /// Each is four indices, anticlockwise from the shallow end of the `i` axis:
    /// `[top-near, top-far, bottom-far, bottom-near]`.
    #[must_use]
    pub fn faces(&self) -> &[[usize; 4]] {
        &self.faces
    }

    /// Subdivide each face into `strike_count` by `dip_count` cells.
    ///
    /// # What is shared, and what is not
    ///
    /// Refinement **preserves the coarse mesh's sharing and creates nothing new**. A
    /// patch's four corner nodes *are* the face's four vertex indices, so two faces that
    /// shared a corner produce two patches that share it, by index. Everything else is a
    /// fresh vertex.
    ///
    /// It is deliberately not more clever than that. Two planes with matching dip and
    /// depths have coincident bottom corners, and merging those would mean comparing
    /// floats for equality — a classic source of silent mis-topology, for no gain: the
    /// generator does not walk the topology across a seam, and the renderer does not
    /// care. Sharing here means *the same input value*, never *the same computed value*.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] if there is not one [`Cuts`] per face, and
    /// [`Error::MeshTooSmall`] if either count is zero.
    pub fn refine(&self, cuts: &[Cuts]) -> Result<RefinedMesh> {
        if cuts.len() != self.faces.len() {
            return Err(Error::Shape {
                what: "cuts",
                found: cuts.len(),
                expected: self.faces.len(),
            });
        }

        let mut vertices = self.vertices.clone();
        let mut patches = Vec::with_capacity(self.faces.len());

        for (face, cut) in self.faces.iter().zip(cuts) {
            for (what, count) in [
                ("strike cells", cut.strike_count),
                ("dip cells", cut.dip_count),
            ] {
                if count == 0 {
                    return Err(Error::MeshTooSmall {
                        what,
                        found: 0,
                        needed: 1,
                    });
                }
            }

            let corner = face.map(|index| vector(self.vertices[index]));
            let mut nodes = Array2::zeros((cut.dip_count + 1, cut.strike_count + 1));

            for dip in 0..=cut.dip_count {
                let down = crate::units::exact(dip) / crate::units::exact(cut.dip_count);
                for strike in 0..=cut.strike_count {
                    let along = crate::units::exact(strike) / crate::units::exact(cut.strike_count);
                    nodes[[dip, strike]] = match (dip, strike) {
                        // The corners are the face's own vertices, not recomputed ones.
                        (0, 0) => face[0],
                        (0, s) if s == cut.strike_count => face[1],
                        (d, s) if d == cut.dip_count && s == cut.strike_count => face[2],
                        (d, 0) if d == cut.dip_count => face[3],
                        _ => {
                            vertices.push(unvector(bilinear(&corner, along, down)));
                            vertices.len() - 1
                        }
                    };
                }
            }
            patches.push(Patch { nodes });
        }

        Ok(RefinedMesh {
            origin: self.origin,
            vertices,
            patches,
        })
    }
}

/// A point inside a quadrilateral, `along` and `down` each running `0..=1`.
///
/// Corner order is [`Mesh::faces`]'s: `[top-near, top-far, bottom-far, bottom-near]`.
fn bilinear(corner: &[Vector; 4], along: f64, down: f64) -> Vector {
    let top = add(scale(corner[0], 1.0 - along), scale(corner[1], along));
    let bottom = add(scale(corner[3], 1.0 - along), scale(corner[2], along));
    add(scale(top, 1.0 - down), scale(bottom, down))
}

const fn unvector(vector: Vector) -> Vertex {
    Vertex {
        east_km: vector[0],
        north_km: vector[1],
        depth_km: vector[2],
    }
}

impl RefinedMesh {
    /// Build one directly from vertices and per-patch node indices.
    ///
    /// What a *reader* needs. A mesh file stores patches — node positions on a
    /// `(dip_node, strike_node)` grid — so loading one is this rather than a build
    /// followed by a refine, and the geometry that comes back is whatever was written
    /// rather than whatever this module would have produced.
    ///
    /// That is also why it validates: a file may have been hand-edited or written by
    /// something else, so an index into nothing and a patch too small to have a cell are
    /// both reachable here in a way they are not from [`Mesh::refine`].
    ///
    /// # Errors
    ///
    /// [`Error::DanglingVertex`] if a patch indexes a vertex that does not exist, and
    /// [`Error::MeshTooSmall`] if a patch has no cells.
    pub fn from_parts(
        origin: Projected,
        vertices: Vec<Vertex>,
        patches: Vec<Array2<usize>>,
    ) -> Result<Self> {
        for nodes in &patches {
            let (dip_nodes, strike_nodes) = nodes.dim();
            for (what, count) in [("strike nodes", strike_nodes), ("dip nodes", dip_nodes)] {
                if count < 2 {
                    return Err(Error::MeshTooSmall {
                        what,
                        found: count,
                        needed: 2,
                    });
                }
            }
            if let Some(index) = nodes.iter().find(|index| **index >= vertices.len()) {
                return Err(Error::DanglingVertex {
                    index: *index,
                    vertices: vertices.len(),
                });
            }
        }
        Ok(Self {
            origin,
            vertices,
            patches: patches.into_iter().map(|nodes| Patch { nodes }).collect(),
        })
    }

    /// The point every vertex is measured from.
    #[must_use]
    pub const fn origin(&self) -> Projected {
        self.origin
    }

    /// The refined mesh's vertices, as offsets from [`RefinedMesh::origin`].
    #[must_use]
    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    /// How many patches there are — one per face of the mesh this came from.
    #[must_use]
    pub fn patch_count(&self) -> usize {
        self.patches.len()
    }

    /// One patch, with the vertices it indexes.
    ///
    /// # Panics
    ///
    /// If `index` is not a patch of this mesh.
    #[must_use]
    pub fn patch(&self, index: usize) -> PatchView<'_> {
        PatchView {
            vertices: &self.vertices,
            nodes: &self.patches[index].nodes,
        }
    }

    /// Every patch, in order.
    pub fn patches(&self) -> impl Iterator<Item = PatchView<'_>> {
        (0..self.patches.len()).map(|index| self.patch(index))
    }

    /// Vertices and triangle indices, for a renderer.
    ///
    /// Two triangles per cell, sharing vertices with their neighbours — the
    /// topologically honest mesh.
    ///
    /// **Not what a per-cell colour wants.** A renderer that colours by *vertex* will
    /// interpolate across a shared one, which for a piecewise-constant field draws values
    /// that were never computed. Such a caller needs four vertices per cell of its own;
    /// this is the mesh, not a display list.
    #[must_use]
    pub fn triangles(&self) -> (&[Vertex], Vec<[usize; 3]>) {
        let mut triangles = Vec::new();
        for patch in self.patches() {
            let (strike_count, dip_count) = patch.cell_extents();
            for dip in 0..dip_count {
                for strike in 0..strike_count {
                    let corner = quad(dip, strike).map(|index| patch.nodes[index]);
                    triangles.push([corner[0], corner[1], corner[2]]);
                    triangles.push([corner[0], corner[2], corner[3]]);
                }
            }
        }
        (&self.vertices, triangles)
    }
}

// ---------------------------------------------------------------------------
// Derived quantities
// ---------------------------------------------------------------------------

/// The four node indices of a cell, anticlockwise from its shallow `i` end.
///
/// Order matters: [`PatchView::along_strike`] and [`PatchView::down_dip`] read edges out
/// of it by position, and [`PatchView::areas_km2`] splits it across the `(0, 2)` diagonal.
const fn quad(dip: usize, strike: usize) -> [[usize; 2]; 4] {
    [
        [dip, strike],
        [dip, strike + 1],
        [dip + 1, strike + 1],
        [dip + 1, strike],
    ]
}

impl PatchView<'_> {
    /// Cells along strike and down dip — one fewer than the nodes in each direction.
    #[must_use]
    pub fn cell_extents(&self) -> (usize, usize) {
        let (dip_nodes, strike_nodes) = self.nodes.dim();
        (strike_nodes - 1, dip_nodes - 1)
    }

    /// Nodes along strike and down dip.
    #[must_use]
    pub fn node_extents(&self) -> (usize, usize) {
        let (dip_nodes, strike_nodes) = self.nodes.dim();
        (strike_nodes, dip_nodes)
    }

    /// Which vertex a node is.
    ///
    /// The index rather than the position, which is what a caller checking *sharing*
    /// needs: two patches meeting at a corner hold the same index there, and comparing
    /// positions would only show they agree numerically.
    ///
    /// # Panics
    ///
    /// If `index` is not a node of this patch.
    #[must_use]
    pub fn node_index(&self, index: [usize; 2]) -> usize {
        self.nodes[index]
    }

    /// One node's position.
    fn at(&self, index: [usize; 2]) -> Vector {
        vector(self.vertices[self.nodes[index]])
    }

    /// A cell's four corners, in [`quad`]'s order.
    fn corners(&self, dip: usize, strike: usize) -> [Vector; 4] {
        quad(dip, strike).map(|index| self.at(index))
    }

    /// A cell's along-strike direction: the sum of its two along-strike edges.
    ///
    /// The sum rather than the mean, because only the direction is read from it.
    fn along_strike(&self, dip: usize, strike: usize) -> Vector {
        let corner = self.corners(dip, strike);
        add(between(corner[0], corner[1]), between(corner[3], corner[2]))
    }

    /// A cell's down-dip direction: the sum of its two down-dip edges.
    fn down_dip(&self, dip: usize, strike: usize) -> Vector {
        let corner = self.corners(dip, strike);
        add(between(corner[0], corner[3]), between(corner[1], corner[2]))
    }

    /// Where every node is, one array per axis.
    #[must_use]
    pub fn positions(&self) -> Positions {
        let dim = self.nodes.raw_dim();
        let mut east_km = Array2::zeros(dim);
        let mut north_km = Array2::zeros(dim);
        let mut depth_km = Array2::zeros(dim);
        for (index, node) in self.nodes.indexed_iter() {
            let vertex = self.vertices[*node];
            east_km[index] = vertex.east_km;
            north_km[index] = vertex.north_km;
            depth_km[index] = vertex.depth_km;
        }
        Positions {
            east_km,
            north_km,
            depth_km,
        }
    }

    /// Where each cell's centre is: the mean of its four corners.
    #[must_use]
    pub fn centres(&self) -> Centres {
        let (strike_count, dip_count) = self.cell_extents();
        let mut east_km = crate::grid::zeros(strike_count, dip_count);
        let mut north_km = crate::grid::zeros(strike_count, dip_count);
        let mut depth_km = crate::grid::zeros(strike_count, dip_count);

        for dip in 0..dip_count {
            for strike in 0..strike_count {
                let corner = self.corners(dip, strike);
                let centre = scale(
                    add(add(corner[0], corner[1]), add(corner[2], corner[3])),
                    0.25,
                );
                east_km[[dip, strike]] = centre[0];
                north_km[[dip, strike]] = centre[1];
                depth_km[[dip, strike]] = centre[2];
            }
        }

        Centres {
            east_km,
            north_km,
            depth_km,
        }
    }

    /// The area of each cell, in square kilometres.
    ///
    /// Split into two triangles across the `(0, 2)` diagonal and summed, so a cell whose
    /// four corners are not coplanar still has a well-defined area. Every cell this module
    /// builds is planar, so the split does not matter and the sum is exact — but the
    /// surfaces this will be generalised to will not be, and the formula that copes costs
    /// nothing.
    #[must_use]
    pub fn areas_km2(&self) -> SlipField {
        let (strike_count, dip_count) = self.cell_extents();
        let mut areas = crate::grid::zeros(strike_count, dip_count);

        for dip in 0..dip_count {
            for strike in 0..strike_count {
                let corner = self.corners(dip, strike);
                areas[[dip, strike]] = triangle_area(corner[0], corner[1], corner[2])
                    + triangle_area(corner[0], corner[2], corner[3]);
            }
        }
        areas
    }

    /// The strike of each cell, in degrees clockwise from the projection's northing axis,
    /// in `[0, 360)`.
    ///
    /// **Grid north, not true north** — see the module note on the convergence angle.
    #[must_use]
    pub fn strike_deg(&self) -> SlipField {
        self.orientation(|orientation| orientation.0)
    }

    /// The dip of each cell, in degrees below horizontal, in `[0, 90]`.
    #[must_use]
    pub fn dip_deg(&self) -> SlipField {
        self.orientation(|orientation| orientation.1)
    }

    /// Strike and dip of every cell, one of the two selected.
    ///
    /// Both come from the cell's **normal**, not from its edges. On a plane the two agree.
    /// Taking the normal is what keeps them right on a surface that is not one — where the
    /// down-dip edge need not be the steepest descent through the cell — and it costs a
    /// cross product.
    fn orientation(&self, select: impl Fn((f64, f64)) -> f64) -> SlipField {
        let (strike_count, dip_count) = self.cell_extents();
        let mut field = crate::grid::zeros(strike_count, dip_count);

        for dip in 0..dip_count {
            for strike in 0..strike_count {
                field[[dip, strike]] = select(orient(
                    self.along_strike(dip, strike),
                    self.down_dip(dip, strike),
                ));
            }
        }
        field
    }

    /// Distance along strike to each node, in kilometres, measured along the top edge.
    ///
    /// Zero at `i = 0`, and the patch's length at `i = strike_count`. This is the in-fault
    /// coordinate a hypocentre is given in.
    #[must_use]
    pub fn strike_arc_km(&self) -> Array1<f64> {
        let (strike_nodes, _) = self.node_extents();
        self.arc(strike_nodes, |step| [0, step])
    }

    /// Distance down dip to each node, in kilometres, measured down the `i = 0` edge.
    ///
    /// Zero at the top edge. The counterpart of [`PatchView::strike_arc_km`], and the
    /// other half of a hypocentre's position.
    #[must_use]
    pub fn dip_arc_km(&self) -> Array1<f64> {
        let (_, dip_nodes) = self.node_extents();
        self.arc(dip_nodes, |step| [step, 0])
    }

    /// Cumulative distance along one edge of the patch.
    fn arc(&self, nodes: usize, index: impl Fn(usize) -> [usize; 2]) -> Array1<f64> {
        let mut arc = Array1::zeros(nodes);
        for step in 1..nodes {
            let edge = between(self.at(index(step - 1)), self.at(index(step)));
            arc[step] = arc[step - 1] + length(edge);
        }
        arc
    }

    /// The uniform cell spacing this patch has, if it has one.
    ///
    /// # Errors
    ///
    /// [`Error::NonUniformMesh`] if the cells are not all the same size along an axis, to
    /// within [`UNIFORM_SPACING_TOLERANCE`]. The generators need a uniform grid; this is
    /// where a mesh that is not one is refused, rather than three layers down inside a
    /// transform.
    pub fn spacing(&self) -> Result<SubfaultSpacing> {
        Ok(SubfaultSpacing {
            strike_km: uniform_step("strike", &self.strike_arc_km())?,
            dip_km: uniform_step("dip", &self.dip_arc_km())?,
        })
    }

    /// The cell containing a position given as two in-fault arc lengths.
    ///
    /// # The convention, written here because getting it wrong is not obvious
    ///
    /// `strike_km` is measured from the `i = 0` **end** of the patch and `dip_km` from the
    /// **top** edge, both as arc lengths along the patch's own axes, and the result is a
    /// **zero-based** cell index.
    ///
    /// Three things this is deliberately not. It is not the SRF's `shyp`, which is
    /// measured from the along-strike *centre*; converting is the SRF writer's job and
    /// happens at the one seam that writes that format. It is not genslip's `ixs`/`iys`,
    /// which count from one and exist to be handed to Fortran — reading those as subfault
    /// indices costs a whole cell in each direction and produces a rupture that is
    /// *plausible*, smooth and correlated 0.99+ with the right one, which is
    /// `DEFECTS.md` 17 and was found by nothing short of a whole-rupture comparison. And
    /// it is not a node index: there is one more node than cell along each axis.
    ///
    /// A position exactly on the far edge belongs to the last cell rather than to no cell.
    ///
    /// # Errors
    ///
    /// [`Error::PositionOffMesh`] if the position is outside the patch.
    pub fn cell_index(&self, strike_km: f64, dip_km: f64) -> Result<Hypocentre> {
        Ok(Hypocentre {
            strike: locate("strike", strike_km, &self.strike_arc_km())?,
            dip: locate("dip", dip_km, &self.dip_arc_km())?,
        })
    }
}

/// Half the magnitude of the cross product: the area of a triangle in space.
fn triangle_area(apex: Vector, first: Vector, second: Vector) -> f64 {
    0.5 * length(cross(between(apex, first), between(apex, second)))
}

/// Strike and dip of the plane spanned by two in-plane directions.
fn orient(along_strike: Vector, down_dip: Vector) -> (f64, f64) {
    let normal = cross(along_strike, down_dip);
    let magnitude = length(normal);
    if magnitude == 0.0 {
        // A degenerate cell has no plane and so no orientation. Report the along-strike
        // direction's own bearing and a dip of zero rather than a NaN, which would travel
        // silently into an SRF.
        return (bearing_of(along_strike), 0.0);
    }
    let unit = scale(normal, 1.0 / magnitude);

    // The normal is vertical for a horizontal plane and horizontal for a vertical one, so
    // the angle it makes with the vertical is the complement of the dip.
    let dip_deg = dot(unit, DOWN).abs().clamp(0.0, 1.0).acos().to_degrees();

    // Strike is the horizontal line lying in the plane. `down x normal` is horizontal
    // because it is perpendicular to `down`, and lies in the plane because it is
    // perpendicular to the normal; its sign is fixed by the cell's own edges.
    let horizontal = cross(DOWN, unit);
    let strike_deg = if length(horizontal) == 0.0 {
        // A horizontal cell has no strike. Fall back for the same reason as above.
        bearing_of(along_strike)
    } else if dot(horizontal, along_strike) < 0.0 {
        bearing_of(scale(horizontal, -1.0))
    } else {
        bearing_of(horizontal)
    };

    (strike_deg, dip_deg)
}

/// The common step of a cumulative arc, or a refusal.
///
/// The **mean** step rather than the first: the two agree to round-off on any patch this
/// accepts, and a mean does not depend on which end the arc was walked from.
fn uniform_step(axis: &'static str, arc: &Array1<f64>) -> Result<f64> {
    let steps: Vec<f64> = arc
        .windows(2)
        .into_iter()
        .map(|pair| pair[1] - pair[0])
        .collect();
    if steps.is_empty() {
        return Err(Error::MeshTooSmall {
            what: "cells along an axis",
            found: 0,
            needed: 1,
        });
    }
    let smallest = steps.iter().copied().fold(f64::INFINITY, f64::min);
    let largest = steps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = steps.iter().sum::<f64>() / crate::units::exact(steps.len());

    if largest - smallest > UNIFORM_SPACING_TOLERANCE * mean {
        return Err(Error::NonUniformMesh {
            axis,
            smallest_km: smallest,
            largest_km: largest,
        });
    }
    Ok(mean)
}

/// Which cell an arc-length position falls in.
fn locate(axis: &'static str, position_km: f64, arc: &Array1<f64>) -> Result<usize> {
    let extent_km = *arc.last().expect("a patch has at least one node");
    if position_km < 0.0 || position_km > extent_km {
        return Err(Error::PositionOffMesh {
            axis,
            position_km,
            extent_km,
        });
    }
    // The last cell owns its far edge, so search among the interior boundaries only.
    let cells = arc.len() - 1;
    Ok((1..cells)
        .find(|cell| position_km < arc[*cell])
        .map_or(cells - 1, |cell| cell - 1))
}
