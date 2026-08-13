//! The fast iterative method on a triangulated surface: first arrivals, in parallel.
//!
//! The surface-native counterpart to [`crate::eikonal`]. That module sweeps a Cartesian
//! lattice with a factored 5-point stencil; this one iterates an unstructured
//! triangulation, which is what a curved fault needs and what a lattice cannot express.
//!
//! # The two papers
//!
//! > **Fu, Z., Jeong, W.-K., Pan, Y., Kirby, R. M. & Whitaker, R. T. (2011).** A fast
//! > iterative method for solving the eikonal equation on triangulated surfaces.
//! > *SIAM Journal on Scientific Computing* **33**(5), 2468–2488.
//! >
//! > **Kimmel, R. & Sethian, J. A. (1998).** Computing geodesic paths on manifolds.
//! > *Proceedings of the National Academy of Sciences USA* **95**(15), 8431–8435.
//!
//! Fu et al. give the iteration — an **active list** swept until it empties, with no
//! heap and no global ordering — and the per-triangle constant speed. Kimmel & Sethian
//! give the local solver (their §4.1, equations 4 and 5) and the virtual-edge unfolding
//! for obtuse triangles (their §4.2), which Fu et al. adopt by reference. The full
//! argument, including which parts of each paper are used and why the local solver is
//! Kimmel & Sethian's rather than Fu et al.'s equation (2.2), is in
//! `rupture_generator/triangular/fim.py`; this module is the same method in Rust and
//! does not restate it.
//!
//! # Why this exists, in numbers
//!
//! Production resolution for slip heterogeneity is 100 m, which on the CFM's Hikurangi
//! interface is about 17.6 M vertices. The numpy implementation of this same method,
//! `rupture_generator/triangular/fim.py`, is **66x to 71x slower** on a mesh built at
//! that resolution's own quality — measured on Hikurangi meshes from
//! `rupture_generator.triangular.mesh.remesh`:
//!
//! ```text
//!   spacing        V        numpy    this (1 thread)  speedup   peak
//!    3200 m     17,204     1.317 s       0.019 s        69x    0.19 GB
//!    1600 m     68,704     5.651 s       0.085 s        66x    0.27 GB
//!     800 m    275,049    25.006 s       0.354 s        71x    0.57 GB
//!     400 m  1,100,240   141.187 s       1.535 s        92x    1.45 GB
//!     200 m  4,400,971          —        5.670 s         —     4.99 GB
//!     100 m ~17,600,000         —      ~23 s (projected)  —   ~20 GB
//! ```
//!
//! Throughput is flat at 0.72 to 0.89 M vertices per second across two and a half orders
//! of magnitude of mesh, which is what makes the last row a projection rather than a
//! guess: 100 m is reachable on one core, in about half a minute and twenty gigabytes.
//!
//! Both are close to linear in the vertex count on a **well-shaped** mesh, so the 70x is
//! interpreter and allocator overhead rather than a difference of algorithm. What
//! separates them is what happens when the mesh is *not* well shaped, and that is worth
//! stating because it was most of the apparent difference before the meshes were built
//! rather than subdivided: on the raw CFM triangulation refined 1-to-4 — area ratio
//! 4.3e4, minimum angle 0.018 degrees, invariant under refinement because subdivision
//! produces *similar* triangles — the numpy batched pass degrades to roughly `V^1.5`
//! while this stays linear.
//!
//! The reason is the ordering. A **batched** pass moves information one ring per sweep, so
//! the band of vertices it has to keep revisiting thickens with the mesh: measured at 3.1,
//! 5.1 and 9.4 ring-populations across three refinements of the raw CFM mesh, and 15.5,
//! 32.0 then 69.4 vertex updates per vertex. This module runs Fu et al.'s **Algorithm 2.1
//! as written** — Gauss–Seidel, in place, "each update is immediately transferred to the
//! solution to be used by subsequent updates" — and measures **6.3 to 6.5 visits per
//! vertex, flat**, on every mesh tried, well shaped or not. That is the `O(N)` the paper
//! claims.
//!
//! The ordering does not change what the iteration converges to, only how long it takes:
//! both are fixed-point iterations of the same monotone update. On built meshes the two
//! implementations agree to **4e-12 s and better**, which
//! `tests/triangular/test_fim.py` pins.
//!
//! # Where the parallelism is, and what each level is worth
//!
//! Two levels, measured on Hikurangi meshes on eight cores. Neither is compute bound: the
//! wedge table is 123 MB at 800 m resolution against 12 MB of L3, and the hot loop reads a
//! wedge and then chases two random indices into the arrival array, so what limits it is
//! memory *latency* rather than bandwidth. That is also why the win over numpy is 70x
//! rather than the several hundred a compute-bound loop would give, and why neither level
//! of threading below scales anywhere near linearly.
//!
//! **Across seeds: exact, and 2.5x on eight cores.** A multi-seed field is the pointwise
//! minimum of independent single-seed solves — the same contract [`crate::eikonal`]
//! carries, for the same reason: first arrival from several sources *is* the minimum over
//! sources. Each solve owns its own array, so there is nothing to synchronise and the
//! answer is **bit-identical at every thread count**, which `tests/fim_contract.rs`
//! asserts. Measured with eight seeds on 300 k vertices: 6.18 s at one thread, 2.53 s at
//! eight. Short of linear because eight solves stream one shared wedge table past one
//! memory controller.
//!
//! **Within one solve: 1.6x, and it costs reproducibility.** This is Fu et al.'s reason
//! for choosing the method — the active list has no heap and every vertex on it can be
//! updated independently, which is why their own convergence study ran on a GPU — but a
//! *batched* pass is the thing measured above to cost an order of magnitude more work.
//! So the parallel path keeps Gauss–Seidel's immediacy instead: the list is chunked
//! across threads and every thread reads and writes the shared solution as it goes,
//! taking whatever value is currently there. That is **asynchronous relaxation**. Writes
//! are a `fetch_min` over the bit patterns of non-negative floats, so there is no data
//! race and no lock; reads are relaxed and may be stale, and a stale read costs an extra
//! visit rather than a wrong answer.
//!
//! What it buys is small, because the active band is a few thousand vertices against a
//! mesh of a million and each pass ends in a join. Measured on a built mesh at 800 m,
//! 275 k vertices: 0.354 s at one thread, 0.271 s at four, 0.277 s at eight — 1.3x.
//!
//! What it costs is **bit-reproducibility**, and this is the part worth stating plainly.
//! Every answer returned is a fixed point of the update — [`inconsistent`] runs
//! afterwards and would say so otherwise — but the update has more than one fixed point
//! at the 1e-4 level, and which one an asynchronous iteration lands on depends on thread
//! timing. Measured spread across thread counts: **9.1e-4 s**, which is 55 times under
//! `ENGINEERING_RULES.md`'s 0.05 s onset bound and 390 times under the 0.35 s the model
//! displaces every onset by on purpose. It is nonetheless a real loss for a generator
//! that is meant to be reproducible from its event seed, so `threads = 1` is the setting
//! that gives one answer, and at 1.3x it is the one to prefer. **The parallelism worth
//! having here is across seeds**, which is exact; within one solve it is available,
//! measured, and not recommended.
//!
//! # What stays in Python
//!
//! The analytic geodesic-ball boundary condition, the derivation of its radius `r0`
//! from the mesh and the velocity model, and the reporting of the two bounds that pin
//! it. This kernel takes an **already-chosen Dirichlet boundary** — vertices and their
//! arrival times — for the same reason [`crate::pulse`] takes an already-resolved
//! shape: policy and defaults live in one place, and that place is Python
//! (`PLAN.md` §3.5).

use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// How little a vertex's arrival may move and still count as settled, in seconds.
///
/// Fu et al.'s `|p − q| < ε`, the test that takes a vertex off the active list. They
/// never give it a value. Bounded below by f64 round-off — a 10 s fault-scale
/// traveltime resolves to about 2e-15 s — and above by anything a traveltime is
/// compared at, the tightest being `ENGINEERING_RULES.md`'s 0.05 s. Ten orders above
/// the floor and ten below the ceiling. It bounds the tail of the iteration, not the
/// accuracy: the scheme's own error is first order in the edge length, eleven orders
/// larger.
///
/// Deliberately equal to `rupture_generator.triangular.fim.SETTLED_TOLERANCE_S`. The
/// two implementations are held to each other by `tests/fim_contract.rs` and by
/// `tests/triangular/test_fim.py`, and a different stopping rule would make that
/// comparison measure the tolerance instead of the method.
pub const SETTLED_TOLERANCE_S: f64 = 1.0e-12;

/// How many triangles one obtuse wedge may be unfolded through before it is abandoned.
///
/// Kimmel & Sethian equation (6) bounds the count by a constant independent of the
/// mesh size, which is what keeps the construction `O(M)`. It is not a *small* constant
/// in the worst case, so this is a ceiling rather than an estimate. Wedges that hit it,
/// or that reach a boundary edge with nothing beyond to unfold, keep the one-sided edge
/// update and are counted on [`Report::unsplit_obtuse`] — a silently degraded stencil
/// reads as a plausible answer.
pub const UNFOLD_LIMIT: usize = 64;

/// How many times the front's own ring count the active list may iterate.
///
/// Fu et al.'s list advances by at least one ring per pass, so a front crossing `n`
/// rings needs `n` passes; a vertex re-enters only when a faster path overtakes an
/// earlier one, which is a property of the medium. Four times the boundary's own
/// eccentricity, measured on the mesh handed in, is a ceiling on a bounded quantity
/// rather than a knob — the loop exits on an empty list. Under Gauss–Seidel the
/// measured ratio on the CFM interfaces is under 2.
pub const MAX_SWEEP_FACTOR: usize = 4;

/// How many vertices one thread takes off the active list at a time.
///
/// Small enough that a list of a few thousand still spreads over eight cores, large
/// enough that the per-chunk `Vec` and the work-stealing handshake are amortised over
/// real work. Rayon splits recursively, so this is a floor on the leaf size rather than
/// a partition count, and the leaves stay balanced when the list is irregular — which it
/// is, because a wavefront band is not a uniform shape.
const CHUNK: usize = 256;

/// The smallest `sin(theta)` a corner may have and still be a triangle.
///
/// Relative — `|e1 × e2| / (|e1||e2|)` — so it is scale free. Below this the corner is
/// a straight line, the quadratic's leading coefficient collapses and its root is
/// noise. Matches the numpy implementation's `_DEGENERATE_SINE`.
const DEGENERATE_SINE: f64 = 1.0e-12;

/// A Dirichlet boundary: which vertices are held, and at what times.
///
/// One of these per seed. The ball around a hypocentre is what Python builds; this is
/// what it builds *into*.
#[derive(Clone, Copy, Debug)]
pub struct Boundary<'a> {
    /// Vertex indices, which may repeat — the earliest time wins, which is the same
    /// pointwise minimum several seeds are combined by.
    pub vertices: &'a [i64],
    /// Their arrival times in seconds, one per entry of `vertices`.
    pub times_s: &'a [f64],
}

/// What this solver refuses, in its own vocabulary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Error {
    /// A mesh with no vertices or no faces has no surface to solve on.
    EmptyMesh { vertices: usize, faces: usize },
    /// The vertex array is not a flat `(V, 3)`.
    NotThreeDimensional { values: usize },
    /// The face array is not a flat `(F, 3)`.
    NotTriangular { values: usize },
    /// A vertex position must be finite; a NaN travels into every derived quantity.
    NonFiniteVertex { vertex: usize },
    /// A face naming a vertex the mesh does not have is a caller error.
    FaceOutOfBounds {
        face: usize,
        vertex: i64,
        vertices: usize,
    },
    /// The slowness slice does not cover the faces it claims to.
    WrongLength { faces: usize, got: usize },
    /// Slowness must be positive and finite everywhere: a face no wave can cross would
    /// make the faces behind it unreachable, and the error would surface far from the
    /// face that caused it.
    NonPositiveSlowness { face: usize, value: f64 },
    /// A corner so thin that no gradient can be read across it.
    DegenerateTriangle {
        face: usize,
        vertex: usize,
        sine: f64,
    },
    /// No boundary means no wavefront: every travel time would be infinite.
    NoBoundary,
    /// A boundary's vertex and time arrays must agree in length.
    MismatchedBoundary { vertices: usize, times: usize },
    /// A held vertex outside the mesh is a caller error, not a boundary condition.
    BoundaryOutOfBounds {
        boundary: usize,
        vertex: i64,
        vertices: usize,
    },
    /// A held time must be finite; NaN or infinity would poison every vertex it wins.
    NonFiniteBoundaryTime {
        boundary: usize,
        entry: usize,
        time_s: f64,
    },
    /// The active list did not empty in [`MAX_SWEEP_FACTOR`] times the ring count.
    DidNotSettle { passes: usize },
    /// A vertex no wavefront reaches: its component of the mesh holds no boundary.
    Unreachable { vertex: usize, count: usize },
    /// The worker pool could not be built at the size asked for.
    NoWorkers { workers: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::EmptyMesh { vertices, faces } => write!(
                f,
                "a mesh of {vertices} vertices and {faces} faces has no surface to solve on"
            ),
            Self::NotThreeDimensional { values } => write!(
                f,
                "the vertex array holds {values} values, which is not a whole number of \
                 (east, north, depth) triples"
            ),
            Self::NotTriangular { values } => write!(
                f,
                "the face array holds {values} values; this solver is triangular, so every \
                 face needs exactly 3 corners"
            ),
            Self::NonFiniteVertex { vertex } => {
                write!(f, "vertex {vertex} is not at a finite position")
            }
            Self::FaceOutOfBounds {
                face,
                vertex,
                vertices,
            } => write!(
                f,
                "face {face} names vertex {vertex}, which is outside a mesh of {vertices} \
                 vertices"
            ),
            Self::WrongLength { faces, got } => write!(
                f,
                "the slowness field has {got} values, but a mesh of {faces} faces needs one \
                 per face"
            ),
            Self::NonPositiveSlowness { face, value } => write!(
                f,
                "slowness on face {face} is {value} s/km; every face must be positive and \
                 finite, or the faces behind it are unreachable"
            ),
            Self::DegenerateTriangle { face, vertex, sine } => write!(
                f,
                "face {face} has a corner at vertex {vertex} whose angle has sine {sine}; a \
                 triangle that thin is a line segment, and no gradient can be read across it"
            ),
            Self::NoBoundary => {
                write!(f, "no boundary: a wavefront needs somewhere to start")
            }
            Self::MismatchedBoundary { vertices, times } => write!(
                f,
                "{vertices} boundary vertices carry {times} times; each held vertex needs \
                 exactly one arrival"
            ),
            Self::BoundaryOutOfBounds {
                boundary,
                vertex,
                vertices,
            } => write!(
                f,
                "boundary {boundary} holds vertex {vertex}, which is outside a mesh of \
                 {vertices} vertices"
            ),
            Self::NonFiniteBoundaryTime {
                boundary,
                entry,
                time_s,
            } => write!(
                f,
                "boundary {boundary} entry {entry} starts at t = {time_s}, which is not a time"
            ),
            Self::DidNotSettle { passes } => write!(
                f,
                "the active list did not empty in {passes} passes, which is \
                 {MAX_SWEEP_FACTOR} times this mesh's own ring count; the medium has \
                 structure this scheme does not handle, or the mesh has a fold the \
                 admissibility check did not catch"
            ),
            Self::NoWorkers { workers } => write!(
                f,
                "a pool of {workers} worker threads could not be started; ask for fewer, \
                 or 0 to take one per core"
            ),
            Self::Unreachable { vertex, count } => write!(
                f,
                "{count} vertices are never reached, the first being {vertex}; they lie in a \
                 component of the mesh that holds no boundary, so give that component a seed \
                 or drop it"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// What one solve did, as evidence rather than as diagnostics.
///
/// The visit count is the claim this module rests on — `O(N)` against the numpy
/// implementation's `O(N^1.5)` — so it is returned rather than logged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// Passes of the active list, summed over boundaries.
    pub passes: usize,
    /// Vertex updates evaluated, summed over boundaries. Divide by the vertex count to
    /// get the number this module exists to keep flat.
    ///
    /// **Not reproducible under threading.** Asynchronous relaxation reads whatever is
    /// currently there, so a stale read costs an extra visit; the answer is unaffected.
    pub vertex_updates: usize,
    /// Obtuse wedges the unfolding could not split — a boundary edge with nothing
    /// beyond it, or [`UNFOLD_LIMIT`] exhausted. Each keeps the one-sided edge update.
    pub unsplit_obtuse: usize,
}

// ================================================================================
// The wedge table: one record per (triangle, corner), plus the virtual wedges
// ================================================================================

/// One wedge a vertex can be updated across, in the layout the hot loop reads it in.
///
/// Array of structs, not struct of arrays: the local solver touches every field of one
/// wedge and none of the next, so this is one cache line per evaluation. 56 bytes.
///
/// `sine` and `chord_km2` are derivable from the other three lengths — a planar wedge
/// has `chord² = a² + b² − 2ab cos` and `sin = √(1 − cos²)` — and are stored anyway.
/// The mesh is built once and swept `~5.5` times per vertex, so two square roots per
/// evaluation cost more than 16 bytes per wedge, and the coordinator's brief is
/// explicit that memory is available and time is not.
#[derive(Clone, Copy, Debug)]
struct Wedge {
    left: u32,
    right: u32,
    left_km: f64,
    right_km: f64,
    cosine: f64,
    sine: f64,
    chord_km2: f64,
    slowness: f64,
}

/// Every wedge of the mesh, grouped by apex in compressed-row order.
struct Corners {
    wedges: Vec<Wedge>,
    /// Row boundaries into `wedges`, length `vertices + 1`.
    start: Vec<u32>,
    unsplit_obtuse: usize,
}

/// A vertex-to-vertex adjacency in compressed-row order.
struct Adjacency {
    start: Vec<u32>,
    index: Vec<u32>,
}

impl Adjacency {
    fn neighbours(&self, vertex: usize) -> &[u32] {
        let (from, to) = (
            self.start[vertex] as usize,
            self.start[vertex + 1] as usize,
        );
        &self.index[from..to]
    }
}

/// Sorted, de-duplicated vertex adjacency from the face table.
fn adjacency(faces: &[[u32; 3]], vertices: usize) -> Adjacency {
    let mut counts = vec![0u32; vertices + 1];
    for face in faces {
        for corner in 0..3 {
            counts[face[corner] as usize + 1] += 2;
        }
    }
    let mut start = counts;
    for index in 1..start.len() {
        start[index] += start[index - 1];
    }
    let mut fill = start.clone();
    let mut index = vec![0u32; start[vertices] as usize];
    for face in faces {
        for corner in 0..3 {
            let (here, next) = (face[corner], face[(corner + 1) % 3]);
            index[fill[here as usize] as usize] = next;
            fill[here as usize] += 1;
            index[fill[next as usize] as usize] = here;
            fill[next as usize] += 1;
        }
    }
    // De-duplicate each row in place, compacting as we go: the two triangles sharing an
    // edge each contribute it, and a vertex must not be offered an update twice.
    let mut written = 0u32;
    let mut compact = vec![0u32; vertices + 1];
    for (vertex, self_index) in (0..vertices).zip(0u32..) {
        let (from, to) = (start[vertex] as usize, start[vertex + 1] as usize);
        compact[vertex] = written;
        index[from..to].sort_unstable();
        let mut previous = u32::MAX;
        for position in from..to {
            let candidate = index[position];
            if candidate != previous && candidate != self_index {
                index[written as usize] = candidate;
                written += 1;
                previous = candidate;
            }
        }
    }
    compact[vertices] = written;
    index.truncate(written as usize);
    Adjacency {
        start: compact,
        index,
    }
}

/// Graph distance in edges from a set of vertices; `u32::MAX` where unreachable.
fn hops(adjacency: &Adjacency, sources: &[u32], vertices: usize) -> Vec<u32> {
    let mut depth = vec![u32::MAX; vertices];
    let mut frontier: Vec<u32> = Vec::new();
    for &source in sources {
        if depth[source as usize] == u32::MAX {
            depth[source as usize] = 0;
            frontier.push(source);
        }
    }
    let mut next = Vec::new();
    let mut level = 0u32;
    while !frontier.is_empty() {
        level += 1;
        next.clear();
        for &vertex in &frontier {
            for &other in adjacency.neighbours(vertex as usize) {
                if depth[other as usize] == u32::MAX {
                    depth[other as usize] = level;
                    next.push(other);
                }
            }
        }
        std::mem::swap(&mut frontier, &mut next);
    }
    depth
}

/// The scalar cross product of two planar vectors: positive turning left.
fn turn(first: [f64; 2], second: [f64; 2]) -> f64 {
    first[0] * second[1] - first[1] * second[0]
}

fn dot2(first: [f64; 2], second: [f64; 2]) -> f64 {
    first[0] * second[0] + first[1] * second[1]
}

fn norm2(vector: [f64; 2]) -> f64 {
    dot2(vector, vector).sqrt()
}

/// The sine of the angle from one planar vector to another, signed left-positive.
///
/// A cross product divided by both lengths, so it is an angle rather than an area and
/// comparing it against a tolerance means the same thing at every scale.
fn sine_between(first: [f64; 2], second: [f64; 2]) -> f64 {
    turn(first, second) / (norm2(first) * norm2(second))
}

/// Place a triangle's third vertex in the plane, folded away from the origin.
///
/// The two-circle construction: the unfolded vertex keeps its true distances to the two
/// ends of the edge it is unfolded across, and lands on the far side of that edge from
/// the apex — which is at the origin, because that is the frame the wedge is laid out
/// in. Unfolding onto the apex's own side would fold the strip back over the wedge it
/// is meant to extend.
fn unfold(anchor: [f64; 2], other: [f64; 2], to_anchor: f64, to_other: f64) -> [f64; 2] {
    let along = [other[0] - anchor[0], other[1] - anchor[1]];
    let length = norm2(along);
    let unit = [along[0] / length, along[1] / length];
    let normal = [-unit[1], unit[0]];
    let forward =
        (to_anchor * to_anchor - to_other * to_other + length * length) / (2.0 * length);
    let sideways = (to_anchor * to_anchor - forward * forward).max(0.0).sqrt();
    let apex_side = turn(along, [-anchor[0], -anchor[1]]);
    let mut candidate = [
        anchor[0] + forward * unit[0] + sideways * normal[0],
        anchor[1] + forward * unit[1] + sideways * normal[1],
    ];
    if turn(along, [candidate[0] - anchor[0], candidate[1] - anchor[1]]) * apex_side > 0.0 {
        candidate = [
            anchor[0] + forward * unit[0] - sideways * normal[0],
            anchor[1] + forward * unit[1] - sideways * normal[1],
        ];
    }
    candidate
}

/// A corner whose angle is past a right angle, and which triangle it belongs to.
#[derive(Clone, Copy)]
struct Obtuse {
    apex: u32,
    left: u32,
    right: u32,
    face: u32,
}

/// One wedge still to be resolved, mid-unfolding.
///
/// The bounding rays are what a finished wedge is emitted from; the far edge is what
/// the next triangle is unfolded across. They come apart when an unfolded vertex lands
/// *outside* the wedge: nothing is split, and the strip carries on past it.
#[derive(Clone, Copy)]
struct Pending {
    ray: [(u32, [f64; 2]); 2],
    edge: [(u32, [f64; 2]); 2],
    face: u32,
}

/// The mesh as the unfolding needs to see it: positions, connectivity, and what is
/// across each edge.
///
/// Bundled rather than passed as three arguments because the unfolding is a walk over
/// the surface and these three are one thing -- the surface -- for the whole of it.
struct Topology<'a> {
    vertices: &'a [[f64; 3]],
    faces: &'a [[u32; 3]],
    across: &'a EdgeFaces,
}

impl Topology<'_> {
    fn distance(&self, first: u32, second: u32) -> f64 {
        let (a, b) = (
            self.vertices[first as usize],
            self.vertices[second as usize],
        );
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// The edge vector from one vertex to another.
    fn edge(&self, from: u32, to: u32) -> [f64; 3] {
        edge_between(self.vertices, from, to)
    }
}

/// The vector from one vertex to another.
fn edge_between(vertices: &[[f64; 3]], from: u32, to: u32) -> [f64; 3] {
    let (a, b) = (vertices[from as usize], vertices[to as usize]);
    [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
}

fn dot3(first: [f64; 3], second: [f64; 3]) -> f64 {
    first[0] * second[0] + first[1] * second[1] + first[2] * second[2]
}

fn cross3(first: [f64; 3], second: [f64; 3]) -> [f64; 3] {
    [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]
}

fn norm3(vector: [f64; 3]) -> f64 {
    dot3(vector, vector).sqrt()
}

/// The lengths, cosine and sine of the angle between two edge vectors at a corner.
fn corner_geometry(to_left: [f64; 3], to_right: [f64; 3]) -> (f64, f64, f64, f64) {
    let left_km = norm3(to_left);
    let right_km = norm3(to_right);
    let scale = left_km * right_km;
    (
        left_km,
        right_km,
        dot3(to_left, to_right) / scale,
        norm3(cross3(to_left, to_right)) / scale,
    )
}

/// Kimmel & Sethian §4.2: split one obtuse wedge into acute virtual ones.
///
/// Lay the wedge out in its own plane with the apex at the origin, then unfold the
/// neighbouring triangles across the far edge until a vertex lands inside the wedge.
/// That vertex splits it in two; each half is checked again and split again if it is
/// still obtuse. Returns how many wedges could not be split at all.
///
/// # A vertex that unfolds *onto* a bounding ray is its own case
///
/// Neither paper says what to do with it, and it splits nothing: one half of the split
/// has zero area and the other is the sector we started with. Deciding it with a bare
/// `turn(...) > 0.0` test means deciding it on the sign of a quantity that has cancelled
/// to zero, which is a rounding artefact of order 1e-17.
///
/// That is not a rare input on a *badly shaped* mesh. Subdividing an obtuse triangle 1-to-4
/// puts unfolded vertices exactly there by construction: measured on the CFM's Hikurangi
/// interface, 309 zero-area virtual wedges at its shipped resolution and 42,358 after
/// three refinements, touching 1% to 4% of vertices. Two implementations of the same
/// arithmetic then walk different routes and emit different wedge chains — and the numpy
/// implementation in `rupture_generator/triangular/fim.py` is one of them, which is why it
/// and this module cannot agree to round-off there however carefully either is written. A
/// 1-ulp difference in a vector norm is enough to flip the branch.
///
/// **On a mesh built by `rupture_generator.triangular.mesh.remesh` the case does not
/// arise**: minimum angle 31 degrees, 11% of corners mildly obtuse, and **zero** wedges
/// left unsplit at every resolution from 3200 m to 400 m. So this is robustness against a
/// mesh production should not be using, not a hot path — which is also why the stricter of
/// the two published acceptance conditions costs nothing to keep.
///
/// Resolved here by **keeping everything**: the zero-area sub-wedge is emitted, because
/// an unfolded flank need not be a mesh neighbour and its one-sided edge bound is then a
/// bound nothing else offers, and the walk continues into the other half, which covers
/// the same sector. Deterministic, and strictly more coverage than either side of the
/// coin toss. [`candidate`] gives the zero-area wedge the answer exact arithmetic gives
/// it, so emitting it costs a bound and risks nothing.
/// One obtuse corner in its own plane, apex at the origin and the left ray along `+x`.
///
/// The frame the whole unfolding happens in: every vertex the strip reaches is placed in
/// these coordinates, so the angle at the apex is a plain cross product throughout.
fn laid_out(wedge: Obtuse, mesh: &Topology<'_>) -> Pending {
    let (left_km, right_km, cosine, sine) = corner_geometry(
        mesh.edge(wedge.apex, wedge.left),
        mesh.edge(wedge.apex, wedge.right),
    );
    let left_2d = [left_km, 0.0];
    let right_2d = [right_km * cosine, right_km * sine];
    Pending {
        ray: [(wedge.left, left_2d), (wedge.right, right_2d)],
        edge: [(wedge.left, left_2d), (wedge.right, right_2d)],
        face: wedge.face,
    }
}

fn split_obtuse(
    wedge: Obtuse,
    mesh: &Topology<'_>,
    out: &mut Vec<(u32, [f64; 2], u32, [f64; 2])>,
) -> usize {
    let mut pending = vec![laid_out(wedge, mesh)];
    let mut budget = UNFOLD_LIMIT;
    let mut unsplit = 0;

    while let Some(wedge) = pending.pop() {
        if dot2(wedge.ray[0].1, wedge.ray[1].1) >= 0.0 {
            out.push((
                wedge.ray[0].0,
                wedge.ray[0].1,
                wedge.ray[1].0,
                wedge.ray[1].1,
            ));
            continue;
        }
        if budget == 0 {
            unsplit += 1;
            continue;
        }
        budget -= 1;

        let Some(next_face) = mesh
            .across
            .other(wedge.edge[0].0, wedge.edge[1].0, wedge.face)
        else {
            // A boundary edge: nothing left to unfold, so this wedge keeps the
            // one-sided edge update its own triangle already offers.
            unsplit += 1;
            continue;
        };
        let far = mesh.faces[next_face as usize]
            .into_iter()
            .find(|&corner| corner != wedge.edge[0].0 && corner != wedge.edge[1].0)
            .expect("a triangle across an edge has a third corner");
        let far_2d = unfold(
            wedge.edge[0].1,
            wedge.edge[1].1,
            mesh.distance(far, wedge.edge[0].0),
            mesh.distance(far, wedge.edge[1].0),
        );

        // **Scale-free, and that is load-bearing.** The branch below decides where the
        // strip walks next, and deciding it on a raw cross product makes it depend on
        // the magnitudes of two unfolded coordinates rather than on the angle between
        // them. On a 1-to-4 subdivided mesh the unfolded vertex frequently lands
        // *exactly* collinear with a bounding ray, where the raw cross product is a
        // rounding artefact of order 1e-17 and its sign is arbitrary; the walk then
        // takes a different route for any two implementations of the same arithmetic,
        // and emits a different chain of wedges. Normalising makes the comparison an
        // angle, and the guard below stops the walk instead of following the noise.
        let left_sine = sine_between(wedge.ray[0].1, far_2d);
        let right_sine = sine_between(far_2d, wedge.ray[1].1);
        // The collinear case: see this function's own documentation for why it is named
        // rather than left to the sign of a cancelling cross product.
        if left_sine.abs() <= DEGENERATE_SINE {
            out.push((
                wedge.ray[0].0,
                wedge.ray[0].1,
                far,
                far_2d,
            ));
            pending.push(Pending {
                ray: [(far, far_2d), wedge.ray[1]],
                edge: [(far, far_2d), wedge.edge[1]],
                face: next_face,
            });
            continue;
        }
        if right_sine.abs() <= DEGENERATE_SINE {
            out.push((far, far_2d, wedge.ray[1].0, wedge.ray[1].1));
            pending.push(Pending {
                ray: [wedge.ray[0], (far, far_2d)],
                edge: [wedge.edge[0], (far, far_2d)],
                face: next_face,
            });
            continue;
        }
        let inside_left = left_sine > 0.0;
        let inside_right = right_sine > 0.0;
        if inside_left && inside_right {
            pending.push(Pending {
                ray: [wedge.ray[0], (far, far_2d)],
                edge: [wedge.edge[0], (far, far_2d)],
                face: next_face,
            });
            pending.push(Pending {
                ray: [(far, far_2d), wedge.ray[1]],
                edge: [(far, far_2d), wedge.edge[1]],
                face: next_face,
            });
        } else if inside_left {
            // Unfolded counter-clockwise of the right ray: the wedge's interior still
            // crosses the far vertex's other edge, so the strip advances there.
            pending.push(Pending {
                ray: wedge.ray,
                edge: [wedge.edge[0], (far, far_2d)],
                face: next_face,
            });
        } else {
            pending.push(Pending {
                ray: wedge.ray,
                edge: [(far, far_2d), wedge.edge[1]],
                face: next_face,
            });
        }
    }
    unsplit
}

/// Which faces meet along each edge, as a sorted table of `(low, high, face)`.
///
/// A sort rather than a hash map: at 37.8 M faces this is 113 M records, and a
/// `HashMap` of that many keys costs more in probes and allocation than one sort of a
/// flat vector costs in comparisons.
struct EdgeFaces {
    entries: Vec<(u32, u32, u32)>,
}

impl EdgeFaces {
    fn build(faces: &[[u32; 3]]) -> Self {
        let mut entries = Vec::with_capacity(faces.len() * 3);
        for (face, corners) in faces.iter().enumerate() {
            let face = u32::try_from(face).expect("face counts fit in u32");
            for corner in 0..3 {
                let (here, next) = (corners[corner], corners[(corner + 1) % 3]);
                entries.push((here.min(next), here.max(next), face));
            }
        }
        entries.sort_unstable();
        Self { entries }
    }

    /// The face on the other side of an edge, if the edge has one.
    fn other(&self, first: u32, second: u32, from: u32) -> Option<u32> {
        let key = (first.min(second), first.max(second));
        let at = self
            .entries
            .partition_point(|&(low, high, _)| (low, high) < key);
        self.entries[at..]
            .iter()
            .take_while(|&&(low, high, _)| (low, high) == key)
            .map(|&(_, _, face)| face)
            .find(|&face| face != from)
    }
}

/// One real triangle corner as a wedge, or `None` if the corner is a straight line.
///
/// `None` rather than an error so the caller names the face and corner; this function
/// sees a corner and not a mesh.
fn real_wedge(
    vertices: &[[f64; 3]],
    apex: u32,
    left: u32,
    right: u32,
    slowness: f64,
) -> Option<Wedge> {
    let (left_km, right_km, cosine, sine) = corner_geometry(
        edge_between(vertices, apex, left),
        edge_between(vertices, apex, right),
    );
    // `is_nan` is named rather than reached by negating a comparison: two coincident
    // vertices make the scale zero and the sine 0/0, and a negated `>=` would accept
    // that silently.
    if sine < DEGENERATE_SINE || sine.is_nan() {
        return None;
    }
    let chord = edge_between(vertices, left, right);
    Some(Wedge {
        left,
        right,
        left_km,
        right_km,
        cosine,
        sine,
        chord_km2: dot3(chord, chord),
        slowness,
    })
}

/// One unfolded wedge, from the planar positions Kimmel & Sethian's §4.2 put it at.
///
/// The slowness is the triangle's whose obtuse angle this splits, not the strip's it was
/// unfolded through: the wedge reads the front's direction inside *that* triangle, and
/// the strip is only how far away the reading was taken.
fn virtual_wedge(
    left: u32,
    left_2d: [f64; 2],
    right: u32,
    right_2d: [f64; 2],
    slowness: f64,
) -> Wedge {
    let left_km = norm2(left_2d);
    let right_km = norm2(right_2d);
    let scale = left_km * right_km;
    let gap = [left_2d[0] - right_2d[0], left_2d[1] - right_2d[1]];
    Wedge {
        left,
        right,
        left_km,
        right_km,
        cosine: dot2(left_2d, right_2d) / scale,
        sine: turn(left_2d, right_2d).abs() / scale,
        chord_km2: gap[0] * gap[0] + gap[1] * gap[1],
        slowness,
    }
}

/// Build every wedge of the mesh, real and virtual, grouped by apex.
fn corners(
    vertices: &[[f64; 3]],
    faces: &[[u32; 3]],
    slowness: &[f64],
) -> Result<Corners, Error> {
    let mut counts = vec![0u32; vertices.len() + 1];
    let mut obtuse: Vec<Obtuse> = Vec::new();

    // Pass one: the real wedges' geometry, and which of them are obtuse.
    let mut real: Vec<(u32, Wedge)> = Vec::with_capacity(faces.len() * 3);
    for (face, corners) in faces.iter().enumerate() {
        for corner in 0..3 {
            let apex = corners[corner];
            let left = corners[(corner + 1) % 3];
            let right = corners[(corner + 2) % 3];
            let wedge = real_wedge(vertices, apex, left, right, slowness[face]).ok_or(
                Error::DegenerateTriangle {
                    face,
                    vertex: apex as usize,
                    sine: corner_geometry(
                        edge_between(vertices, apex, left),
                        edge_between(vertices, apex, right),
                    )
                    .3,
                },
            )?;
            if wedge.cosine < 0.0 {
                obtuse.push(Obtuse {
                    apex,
                    left,
                    right,
                    face: u32::try_from(face).expect("face counts fit in u32"),
                });
            }
            counts[apex as usize + 1] += 1;
            real.push((apex, wedge));
        }
    }

    // Pass two: Kimmel & Sethian §4.2 on each obtuse wedge. The originals stay — the
    // one-sided edge update they still offer is a valid upper bound on a real path, and
    // an upper bound never wins a minimum against a good answer.
    let mut virtual_wedges: Vec<(u32, Wedge)> = Vec::new();
    let mut unsplit_obtuse = 0;
    if !obtuse.is_empty() {
        let across = EdgeFaces::build(faces);
        let mesh = Topology {
            vertices,
            faces,
            across: &across,
        };
        let mut produced = Vec::new();
        for &wedge in &obtuse {
            produced.clear();
            unsplit_obtuse += split_obtuse(wedge, &mesh, &mut produced);
            for &(left, left_2d, right, right_2d) in &produced {
                counts[wedge.apex as usize + 1] += 1;
                virtual_wedges.push((
                    wedge.apex,
                    virtual_wedge(left, left_2d, right, right_2d, slowness[wedge.face as usize]),
                ));
            }
        }
    }

    let mut start = counts;
    for index in 1..start.len() {
        start[index] += start[index - 1];
    }
    let mut fill = start.clone();
    let total = start[vertices.len()] as usize;
    let mut wedges = vec![
        Wedge {
            left: 0,
            right: 0,
            left_km: 0.0,
            right_km: 0.0,
            cosine: 0.0,
            sine: 0.0,
            chord_km2: 0.0,
            slowness: 0.0,
        };
        total
    ];
    for (apex, wedge) in real.into_iter().chain(virtual_wedges) {
        wedges[fill[apex as usize] as usize] = wedge;
        fill[apex as usize] += 1;
    }
    Ok(Corners {
        wedges,
        start,
        unsplit_obtuse,
    })
}

// ================================================================================
// The local solver: Kimmel & Sethian equations (4) and (5)
// ================================================================================

/// The arrival one wedge offers its apex, or infinity if neither flank is reached.
///
/// A transcription of `_candidates` in `rupture_generator/triangular/fim.py`, term for
/// term and branch for branch, because the two are held to each other to round-off.
#[inline]
fn candidate(wedge: &Wedge, at_left: f64, at_right: f64) -> f64 {
    // Kimmel & Sethian order the flanks so that T(A) <= T(B); which is which is a
    // property of the current solution, not of the mesh, so it is decided here.
    let (near_s, far_s, near_km, far_km) = if at_left <= at_right {
        (at_left, at_right, wedge.left_km, wedge.right_km)
    } else {
        (at_right, at_left, wedge.right_km, wedge.left_km)
    };

    // The safe cap: a straight run along a real edge at that triangle's own slowness.
    // Always a valid upper bound, so it can never beat a triangle root that is any good
    // — which is what makes offering it here the same thing as `eikonal.rs`'s "only
    // when no triangle produced a causal root".
    let one_sided_s = (near_s + near_km * wedge.slowness).min(far_s + far_km * wedge.slowness);
    if !far_s.is_finite() {
        return one_sided_s;
    }
    // **A wedge of zero area gets the one-sided bound, which is what exact arithmetic
    // gives it.** Put `sine = 0` into equations (4) and (5): the discriminant becomes
    // `4 near^2 gap^2 (far - near)^2 - 4 (far - near)^2 near^2 gap^2`, identically zero,
    // and the root is `-near gap / (far - near)`, which is negative and so fails
    // `root > gap`. In floating point that cancellation lands at +-1e-20 and the root is
    // whatever the rounding made it, so the branch is decided by noise rather than by
    // geometry. Named here so the degenerate case has one answer instead of two.
    //
    // These wedges are not hypothetical: 1-to-4 subdivision of the CFM's Hikurangi
    // interface produces 309 of them at the shipped resolution and 42,358 after three
    // refinements, because subdividing an obtuse triangle puts an unfolded vertex exactly
    // on a bounding ray. They are kept rather than dropped -- an unfolded flank need not
    // be a mesh neighbour, so its one-sided term is a bound nothing else offers.
    if wedge.sine < DEGENERATE_SINE {
        return one_sided_s;
    }

    let gap_s = far_s - near_s;
    let quadratic = wedge.chord_km2;
    let linear = 2.0 * near_km * gap_s * (far_km * wedge.cosine - near_km);
    let constant = near_km
        * near_km
        * (gap_s * gap_s
            - wedge.slowness * wedge.slowness * far_km * far_km * wedge.sine * wedge.sine);
    let discriminant = linear * linear - 4.0 * quadratic * constant;
    if discriminant < 0.0 {
        return one_sided_s;
    }
    let root = (-linear + discriminant.sqrt()) / (2.0 * quadratic);
    // The root can be NaN where the quadratic degenerates, and a negated comparison
    // would accept it. Named, as `eikonal.rs` names the same hazard.
    if root.is_nan() || root <= gap_s {
        return one_sided_s;
    }
    let foot_km = near_km * (root - gap_s) / root;
    // Equation (5), as a pair of products rather than a division by a cosine that a
    // right-angled triangle makes exactly zero. Both say the same thing: the foot of
    // the perpendicular from the apex meets the level line inside the segment.
    if far_km * wedge.cosine < foot_km && foot_km * wedge.cosine < far_km {
        near_s + root
    } else {
        one_sided_s
    }
}

/// The best arrival a vertex can be given from its own one-ring.
///
/// Generic in how a neighbour's arrival is read so that the sequential path and the
/// threaded one share a single local solver: one reads a plain slice, the other reads
/// relaxed atomics, and neither gets its own copy of Kimmel & Sethian's algebra.
#[inline]
fn update_with<Read: Fn(u32) -> f64>(corners: &Corners, vertex: usize, arrival: Read) -> f64 {
    let (from, to) = (
        corners.start[vertex] as usize,
        corners.start[vertex + 1] as usize,
    );
    let mut best = f64::INFINITY;
    for wedge in &corners.wedges[from..to] {
        let offer = candidate(wedge, arrival(wedge.left), arrival(wedge.right));
        if offer < best {
            best = offer;
        }
    }
    best
}

#[inline]
fn update(corners: &Corners, vertex: usize, times_s: &[f64]) -> f64 {
    update_with(corners, vertex, |at| times_s[at as usize])
}

/// The bit pattern of a non-negative float, ordered the same way the float is.
///
/// Arrivals here are non-negative — a boundary carries a negative `t0` only if the
/// caller invented one — so the IEEE-754 bit pattern of an arrival is monotone in the
/// arrival, and `fetch_min` on the bits is `fetch_min` on the times. Infinity's pattern
/// is larger than every finite one, which is what makes an unreached vertex lose to any
/// candidate. Named so the assumption is written down rather than implied by a cast.
#[inline]
fn ordered_bits(time_s: f64) -> u64 {
    time_s.to_bits()
}

#[inline]
fn relaxed(cell: &AtomicU64) -> f64 {
    f64::from_bits(cell.load(Ordering::Relaxed))
}

/// The best arrival a vertex can be given, reading the shared solution as it stands.
#[inline]
fn update_shared(corners: &Corners, vertex: usize, times: &[AtomicU64]) -> f64 {
    update_with(corners, vertex, |at| relaxed(&times[at as usize]))
}

// ================================================================================
// Fu et al.'s active list, Gauss-Seidel
// ================================================================================

/// Algorithm 2.1, in place and single threaded, from a given active list.
///
/// Each pass updates every vertex on the list **and writes the result immediately**, so
/// a later vertex in the same pass sees it; a vertex whose arrival stopped moving is
/// taken off and its own unheld neighbours are offered an update, joining the list if it
/// improves them.
///
/// The immediacy is the whole point. A batched pass moves information one ring per
/// sweep, and the band it maintains thickens with the mesh — measured at 3.1, 5.1 and
/// 9.4 ring-populations across three refinements of the CFM's Hikurangi interface. This
/// does not: 5.4 to 6.1 visits per vertex over the same sequence.
fn sweep(
    corners: &Corners,
    adjacency: &Adjacency,
    times_s: &mut [f64],
    held: &[bool],
    in_list: &mut [bool],
    active: &mut Vec<u32>,
    max_passes: usize,
) -> Result<(usize, usize), Error> {
    let mut next: Vec<u32> = Vec::with_capacity(active.len());
    let mut passes = 0;
    let mut visits = 0;
    while !active.is_empty() {
        passes += 1;
        if passes > max_passes {
            return Err(Error::DidNotSettle { passes });
        }
        next.clear();
        // Indexed rather than iterated: the body writes `times_s` and pushes to `next`,
        // and holding a borrow of `active` across that is what a `for` loop would do.
        let mut position = 0;
        while position < active.len() {
            let at = active[position];
            let vertex = at as usize;
            position += 1;
            let before_s = times_s[vertex];
            let offer_s = update(corners, vertex, times_s);
            visits += 1;
            if offer_s < before_s {
                times_s[vertex] = offer_s;
            }
            if before_s - times_s[vertex] > SETTLED_TOLERANCE_S {
                next.push(at);
                continue;
            }
            in_list[vertex] = false;
            for &neighbour in adjacency.neighbours(vertex) {
                let other = neighbour as usize;
                if held[other] || in_list[other] {
                    continue;
                }
                let offer_s = update(corners, other, times_s);
                visits += 1;
                if offer_s < times_s[other] - SETTLED_TOLERANCE_S {
                    times_s[other] = offer_s;
                    in_list[other] = true;
                    next.push(neighbour);
                }
            }
        }
        std::mem::swap(active, &mut next);
    }
    Ok((passes, visits))
}

/// Algorithm 2.1 again, with the active list spread across threads.
///
/// **Asynchronous relaxation, not a batched pass.** Every thread reads and writes the
/// shared solution as it goes, so a vertex updated by one thread is visible to the next
/// thread that reads it — which keeps most of the information flow that makes
/// Gauss–Seidel cheap, rather than throwing it away as a batched pass does. Writes are a
/// `fetch_min` on the bit patterns, so a value can only ever fall; reads are relaxed and
/// may be stale, and a stale read costs an extra visit rather than a wrong answer.
///
/// The list membership flag is claimed with a `compare_exchange`, so a vertex two threads
/// discover at the same moment is queued exactly once. And correctness does not rest on
/// any of this: [`inconsistent`] runs afterwards either way, so anything a race dropped
/// is found and swept again.
fn sweep_shared(
    corners: &Corners,
    adjacency: &Adjacency,
    times: &[AtomicU64],
    held: &[bool],
    in_list: &[AtomicBool],
    active: &mut Vec<u32>,
    max_passes: usize,
) -> Result<(usize, usize), Error> {
    let mut passes = 0;
    let mut visits = 0;
    while !active.is_empty() {
        passes += 1;
        if passes > max_passes {
            return Err(Error::DidNotSettle { passes });
        }
        let (found, cost): (Vec<Vec<u32>>, Vec<usize>) = active
            .par_chunks(CHUNK)
            .map(|slice| {
                let mut next = Vec::new();
                let mut visits = 0usize;
                for &at in slice {
                    let vertex = at as usize;
                    let before_s = relaxed(&times[vertex]);
                    let offer_s = update_shared(corners, vertex, times);
                    visits += 1;
                    if offer_s < before_s {
                        times[vertex].fetch_min(ordered_bits(offer_s), Ordering::Relaxed);
                    }
                    if before_s - relaxed(&times[vertex]) > SETTLED_TOLERANCE_S {
                        next.push(at);
                        continue;
                    }
                    in_list[vertex].store(false, Ordering::Relaxed);
                    for &neighbour in adjacency.neighbours(vertex) {
                        let other = neighbour as usize;
                        if held[other] || in_list[other].load(Ordering::Relaxed) {
                            continue;
                        }
                        let offer_s = update_shared(corners, other, times);
                        visits += 1;
                        if offer_s < relaxed(&times[other]) - SETTLED_TOLERANCE_S {
                            times[other].fetch_min(ordered_bits(offer_s), Ordering::Relaxed);
                            if in_list[other]
                                .compare_exchange(
                                    false,
                                    true,
                                    Ordering::Relaxed,
                                    Ordering::Relaxed,
                                )
                                .is_ok()
                            {
                                next.push(neighbour);
                            }
                        }
                    }
                }
                (next, visits)
            })
            .unzip();
        visits += cost.iter().sum::<usize>();
        active.clear();
        for chunk in found {
            active.extend_from_slice(&chunk);
        }
    }
    Ok((passes, visits))
}

/// [`inconsistent`], over the shared solution and in parallel.
fn inconsistent_shared(
    corners: &Corners,
    times: &[AtomicU64],
    held: &[bool],
    in_list: &[AtomicBool],
    active: &mut Vec<u32>,
) -> usize {
    let found: Vec<Vec<u32>> = (0..times.len())
        .into_par_iter()
        .chunks(CHUNK)
        .map(|slice| {
            let mut next = Vec::new();
            for vertex in slice {
                in_list[vertex].store(false, Ordering::Relaxed);
                if held[vertex] {
                    continue;
                }
                if update_shared(corners, vertex, times)
                    < relaxed(&times[vertex]) - SETTLED_TOLERANCE_S
                {
                    in_list[vertex].store(true, Ordering::Relaxed);
                    next.push(u32::try_from(vertex).expect("vertex counts fit in u32"));
                }
            }
            next
        })
        .collect();
    active.clear();
    for chunk in found {
        active.extend_from_slice(&chunk);
    }
    active.len()
}

/// Every unheld vertex an update would still lower, gathered into an active list.
///
/// **The correctness guarantee this solver rests on, and it is not free with
/// Algorithm 2.1 alone.** The paper's removal condition takes a vertex off the list when
/// its own value stops moving, and that is not the same as the vertex being *consistent*
/// with its neighbours: two adjacent vertices can both stop moving in the same visit
/// while each still owes the other an update, and once both are off the list nothing
/// puts them back. The answer is then not a fixed point of the update, and the error
/// propagates downstream from wherever it happened.
///
/// Measured on the CFM's Hikurangi interface, where the element quality is irregular
/// enough to trigger it: the numpy implementation of the same algorithm leaves up to
/// 4 vertices per 20,000 inconsistent by as much as 3.6e-4 s, which moves the field by
/// up to 1.5e-2 s once it has spread. It is not specific to that implementation — this
/// one leaves 12 per 300,000 before the scan below — and it is not visible in any
/// regular triangulation, which is why it survived a lattice-only test suite.
///
/// So the sweep is run again from whatever this finds, until it finds nothing. Each scan
/// is one `update` per vertex, about a sixth of one sweep's work, and it converts "the
/// active list emptied" into "no vertex would move".
fn inconsistent(
    corners: &Corners,
    times_s: &[f64],
    held: &[bool],
    in_list: &mut [bool],
    active: &mut Vec<u32>,
) -> usize {
    active.clear();
    for (vertex, &arrival_s) in times_s.iter().enumerate() {
        in_list[vertex] = false;
        if held[vertex] {
            continue;
        }
        if update(corners, vertex, times_s) < arrival_s - SETTLED_TOLERANCE_S {
            in_list[vertex] = true;
            active.push(u32::try_from(vertex).expect("vertex counts fit in u32"));
        }
    }
    active.len()
}

/// One boundary's solve, on an already-checked mesh: sweep, then verify, then repeat.
///
/// The loop around the sweep is what makes the result a fixed point rather than merely a
/// terminated iteration; [`inconsistent`] says why those are not the same thing.
fn single(
    corners: &Corners,
    adjacency: &Adjacency,
    vertices: usize,
    held_at: &[u32],
    held_s: &[f64],
    parallel: bool,
) -> Result<(Vec<f64>, usize, usize), Error> {
    let mut times_s = vec![f64::INFINITY; vertices];
    let mut held = vec![false; vertices];
    for (&vertex, &time_s) in held_at.iter().zip(held_s) {
        let vertex = vertex as usize;
        // Several entries may name the same vertex; the earliest wins, which is the
        // same pointwise minimum several boundaries are combined by.
        if time_s < times_s[vertex] {
            times_s[vertex] = time_s;
        }
        held[vertex] = true;
    }
    let rings = hops(adjacency, held_at, vertices)
        .iter()
        .filter(|&&depth| depth != u32::MAX)
        .copied()
        .max()
        .unwrap_or(0) as usize;
    let max_passes = MAX_SWEEP_FACTOR * (rings + 1);

    let mut active: Vec<u32> = Vec::new();
    let mut seen = vec![false; vertices];
    for (vertex, &is_held) in held.iter().enumerate() {
        if !is_held {
            continue;
        }
        for &other in adjacency.neighbours(vertex) {
            if !held[other as usize] && !seen[other as usize] {
                seen[other as usize] = true;
                active.push(other);
            }
        }
    }

    let mut passes = 0;
    let mut visits = 0;
    if parallel {
        let times: Vec<AtomicU64> = times_s
            .iter()
            .map(|&arrival| AtomicU64::new(ordered_bits(arrival)))
            .collect();
        let in_list: Vec<AtomicBool> = seen.iter().map(|&on| AtomicBool::new(on)).collect();
        loop {
            let (took, cost) = sweep_shared(
                corners,
                adjacency,
                &times,
                &held,
                &in_list,
                &mut active,
                max_passes,
            )?;
            passes += took;
            visits += cost + vertices;
            if inconsistent_shared(corners, &times, &held, &in_list, &mut active) == 0 {
                break;
            }
            if passes > max_passes {
                return Err(Error::DidNotSettle { passes });
            }
        }
        for (slot, cell) in times_s.iter_mut().zip(&times) {
            *slot = relaxed(cell);
        }
    } else {
        let mut in_list = seen;
        loop {
            let (took, cost) = sweep(
                corners,
                adjacency,
                &mut times_s,
                &held,
                &mut in_list,
                &mut active,
                max_passes,
            )?;
            passes += took;
            visits += cost + vertices;
            if inconsistent(corners, &times_s, &held, &mut in_list, &mut active) == 0 {
                break;
            }
            if passes > max_passes {
                return Err(Error::DidNotSettle { passes });
            }
        }
    }
    Ok((times_s, passes, visits))
}

// ================================================================================
// The public entry point
// ================================================================================

/// Every boundary's vertices as in-range indices, checked.
fn checked_boundaries(
    boundaries: &[Boundary<'_>],
    vertices: usize,
) -> Result<Vec<Vec<u32>>, Error> {
    if boundaries.is_empty() {
        return Err(Error::NoBoundary);
    }
    let limit = i64::try_from(vertices).expect("vertex counts fit in i64");
    let mut held: Vec<Vec<u32>> = Vec::with_capacity(boundaries.len());
    for (index, boundary) in boundaries.iter().enumerate() {
        if boundary.vertices.len() != boundary.times_s.len() {
            return Err(Error::MismatchedBoundary {
                vertices: boundary.vertices.len(),
                times: boundary.times_s.len(),
            });
        }
        if boundary.vertices.is_empty() {
            return Err(Error::NoBoundary);
        }
        let mut at = Vec::with_capacity(boundary.vertices.len());
        for (entry, (&vertex, &time_s)) in
            boundary.vertices.iter().zip(boundary.times_s).enumerate()
        {
            if vertex < 0 || vertex >= limit {
                return Err(Error::BoundaryOutOfBounds {
                    boundary: index,
                    vertex,
                    vertices,
                });
            }
            if !time_s.is_finite() {
                return Err(Error::NonFiniteBoundaryTime {
                    boundary: index,
                    entry,
                    time_s,
                });
            }
            at.push(u32::try_from(vertex).expect("checked against the vertex count above"));
        }
        held.push(at);
    }
    Ok(held)
}

/// How far the worst vertex is from satisfying its own update equation, in seconds.
///
/// Zero means the field is a fixed point of the local solver: no vertex would move if
/// asked again. Positive means the iteration stopped early somewhere, and the value is
/// how much it left on the table at the worst vertex.
///
/// Exposed because it is a property of an answer rather than a comparison between two
/// answers, and that makes it the right way to hold this solver to account: an oracle can
/// only say "the same as me", whereas this says "consistent with its own equations".
/// `tests/fim_contract.rs` asserts it is zero, and it is the measurement that found Fu et
/// al.'s Algorithm 2.1 does not give that for free — see [`inconsistent`].
///
/// # Errors
///
/// [`Error`], as [`solve`], plus [`Error::MismatchedBoundary`] if `times_s` is not one
/// value per vertex.
///
/// # Panics
///
/// If a vertex or face count does not fit in `u32`; see [`solve`].
pub fn residual(
    vertices_km: &[f64],
    faces: &[i64],
    slowness_s_per_km: &[f64],
    times_s: &[f64],
    boundaries: &[Boundary<'_>],
) -> Result<f64, Error> {
    let vertices = checked_vertices(vertices_km)?;
    let faces = checked_faces(faces, vertices.len())?;
    if slowness_s_per_km.len() != faces.len() {
        return Err(Error::WrongLength {
            faces: faces.len(),
            got: slowness_s_per_km.len(),
        });
    }
    if times_s.len() != vertices.len() {
        return Err(Error::MismatchedBoundary {
            vertices: vertices.len(),
            times: times_s.len(),
        });
    }
    let held_at = checked_boundaries(boundaries, vertices.len())?;
    let mut held = vec![false; vertices.len()];
    for at in &held_at {
        for &vertex in at {
            held[vertex as usize] = true;
        }
    }
    let corners = corners(&vertices, &faces, slowness_s_per_km)?;
    Ok(times_s
        .iter()
        .enumerate()
        .filter(|&(vertex, _)| !held[vertex])
        .map(|(vertex, &arrival_s)| arrival_s - update(&corners, vertex, times_s))
        .fold(0.0_f64, f64::max))
}

/// A worker pool of a stated size.
///
/// Built per call rather than taken from rayon's global pool, so that a caller asking
/// for one thread gets one thread — the sequential reference the threaded path is
/// measured against has to be reachable from the same entry point, or the comparison is
/// between two builds instead of two settings.
fn pool(workers: usize) -> Result<rayon::ThreadPool, Error> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|_| Error::NoWorkers { workers })
}

/// Vertices as triples, checked.
fn checked_vertices(vertices_km: &[f64]) -> Result<Vec<[f64; 3]>, Error> {
    if !vertices_km.len().is_multiple_of(3) {
        return Err(Error::NotThreeDimensional {
            values: vertices_km.len(),
        });
    }
    let mut out = Vec::with_capacity(vertices_km.len() / 3);
    for (vertex, triple) in vertices_km.chunks_exact(3).enumerate() {
        if !triple.iter().all(|value| value.is_finite()) {
            return Err(Error::NonFiniteVertex { vertex });
        }
        out.push([triple[0], triple[1], triple[2]]);
    }
    Ok(out)
}

/// Faces as triples of in-range indices, checked.
fn checked_faces(faces: &[i64], vertices: usize) -> Result<Vec<[u32; 3]>, Error> {
    if !faces.len().is_multiple_of(3) {
        return Err(Error::NotTriangular { values: faces.len() });
    }
    let limit = i64::try_from(vertices).expect("vertex counts fit in i64");
    let mut out = Vec::with_capacity(faces.len() / 3);
    for (face, triple) in faces.chunks_exact(3).enumerate() {
        let mut corners = [0u32; 3];
        for (slot, &vertex) in corners.iter_mut().zip(triple) {
            if vertex < 0 || vertex >= limit {
                return Err(Error::FaceOutOfBounds {
                    face,
                    vertex,
                    vertices,
                });
            }
            *slot = u32::try_from(vertex).map_err(|_| Error::FaceOutOfBounds {
                face,
                vertex,
                vertices,
            })?;
        }
        out.push(corners);
    }
    Ok(out)
}

/// First arrivals at every vertex, from every boundary, by the fast iterative method.
///
/// `vertices_km` is flat `(V, 3)` in a projected CRS with depth positive down;
/// `faces` is flat `(F, 3)`; `slowness_s_per_km` is one value per face. Each
/// [`Boundary`] is one seed's already-chosen Dirichlet condition, and the result is the
/// pointwise minimum over them — first arrival from several sources *is* the minimum
/// over sources, and solving them separately is what keeps each source's near field
/// exact.
///
/// `threads` is how many worker threads to use within and across the solves; `0` means
/// one per core. `1` is the sequential Gauss–Seidel path, and it is the reference the
/// threaded path is measured against.
///
/// # Errors
///
/// [`Error`], one variant per way the inputs can fail to describe a medium or a
/// boundary; nothing is clamped or silently repaired.
///
/// # Panics
///
/// If a vertex or face count does not fit in `u32`, which is 4.29 billion of either —
/// two orders past the 18.9 M vertices of the largest interface this is built for, and
/// a bug elsewhere rather than a case to handle.
pub fn solve(
    vertices_km: &[f64],
    faces: &[i64],
    slowness_s_per_km: &[f64],
    boundaries: &[Boundary<'_>],
    threads: usize,
) -> Result<(Vec<f64>, Report), Error> {
    let vertices = checked_vertices(vertices_km)?;
    let faces = checked_faces(faces, vertices.len())?;
    if vertices.is_empty() || faces.is_empty() {
        return Err(Error::EmptyMesh {
            vertices: vertices.len(),
            faces: faces.len(),
        });
    }
    if slowness_s_per_km.len() != faces.len() {
        return Err(Error::WrongLength {
            faces: faces.len(),
            got: slowness_s_per_km.len(),
        });
    }
    for (face, &value) in slowness_s_per_km.iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::NonPositiveSlowness { face, value });
        }
    }
    let held = checked_boundaries(boundaries, vertices.len())?;

    let corners = corners(&vertices, &faces, slowness_s_per_km)?;
    let adjacency = adjacency(&faces, vertices.len());
    let count = vertices.len();
    let workers = if threads == 0 {
        rayon::current_num_threads()
    } else {
        threads
    };

    // **Two levels of parallelism, and only one of them is worth having at a time.**
    // Solves across boundaries are independent — separate arrays, nothing shared — so
    // when there are several seeds they go one per worker and each solve runs
    // sequentially, which keeps Gauss-Seidel's immediacy intact. With a single seed
    // there is nothing to spread that way, so the threads go inside the solve instead
    // and take the asynchronous-relaxation path. Doing both at once would oversubscribe
    // the pool and lose the immediacy for no gain.
    let across_seeds = workers > 1 && boundaries.len() > 1;
    let within_solve = workers > 1 && !across_seeds;

    let solve_one = |(at, boundary): (&Vec<u32>, &Boundary<'_>)| {
        single(
            &corners,
            &adjacency,
            count,
            at,
            boundary.times_s,
            within_solve,
        )
    };
    let solved: Vec<(Vec<f64>, usize, usize)> = if across_seeds {
        let pool = pool(workers)?;
        pool.install(|| {
            held.par_iter()
                .zip(boundaries.par_iter())
                .map(solve_one)
                .collect::<Result<Vec<_>, Error>>()
        })?
    } else if within_solve {
        let pool = pool(workers)?;
        pool.install(|| {
            held.iter()
                .zip(boundaries)
                .map(solve_one)
                .collect::<Result<Vec<_>, Error>>()
        })?
    } else {
        held.iter()
            .zip(boundaries)
            .map(solve_one)
            .collect::<Result<Vec<_>, Error>>()?
    };

    let mut combined = vec![f64::INFINITY; count];
    let mut report = Report {
        unsplit_obtuse: corners.unsplit_obtuse,
        ..Report::default()
    };
    for (times_s, passes, visits) in solved {
        report.passes += passes;
        report.vertex_updates += visits;
        for (slot, &arrival) in combined.iter_mut().zip(&times_s) {
            if arrival < *slot {
                *slot = arrival;
            }
        }
    }

    if let Some(vertex) = combined.iter().position(|arrival| !arrival.is_finite()) {
        let stranded = combined.iter().filter(|a| !a.is_finite()).count();
        return Err(Error::Unreachable {
            vertex,
            count: stranded,
        });
    }
    Ok((combined, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counts::exact;

    /// A right-angled lattice: the mesh a fault is normally cut into.
    fn lattice(cells: usize, extent_km: f64) -> (Vec<f64>, Vec<i64>) {
        let step = extent_km / exact(cells - 1);
        let mut vertices = Vec::with_capacity(cells * cells * 3);
        for down in 0..cells {
            for across in 0..cells {
                vertices.extend_from_slice(&[exact(across) * step, 0.0, exact(down) * step]);
            }
        }
        let mut faces = Vec::new();
        let at = |down: usize, across: usize| i64::try_from(down * cells + across).unwrap();
        for down in 0..cells - 1 {
            for across in 0..cells - 1 {
                faces.extend_from_slice(&[
                    at(down, across),
                    at(down + 1, across),
                    at(down + 1, across + 1),
                ]);
                faces.extend_from_slice(&[
                    at(down, across),
                    at(down + 1, across + 1),
                    at(down, across + 1),
                ]);
            }
        }
        (vertices, faces)
    }

    /// A uniform medium seeded on a disc is solved to round-off inside the disc, and to
    /// first order outside it.
    #[test]
    fn a_uniform_medium_is_exact_on_its_own_boundary() {
        let cells = 33;
        let (vertices, faces) = lattice(cells, 16.0);
        let slowness = vec![0.4; faces.len() / 3];
        let centre = [8.0, 0.0, 8.0];
        let radius: Vec<(i64, f64)> = vertices
            .chunks_exact(3)
            .enumerate()
            .map(|(vertex, point)| {
                let span = ((point[0] - centre[0]).powi(2) + (point[2] - centre[2]).powi(2)).sqrt();
                (i64::try_from(vertex).unwrap(), span)
            })
            .collect();
        let held: Vec<i64> = radius
            .iter()
            .filter(|&&(_, span)| span <= 3.0)
            .map(|&(vertex, _)| vertex)
            .collect();
        let held_s: Vec<f64> = radius
            .iter()
            .filter(|&&(_, span)| span <= 3.0)
            .map(|&(_, span)| 0.4 * span)
            .collect();

        let (times, report) = solve(
            &vertices,
            &faces,
            &slowness,
            &[Boundary {
                vertices: &held,
                times_s: &held_s,
            }],
            1,
        )
        .expect("a uniform lattice is a valid input");

        for (&(vertex, span), &arrival) in radius.iter().zip(&times) {
            let vertex = usize::try_from(vertex).unwrap();
            let exact_s = 0.4 * span;
            if span <= 3.0 {
                assert!(
                    (arrival - exact_s).abs() < 1e-12,
                    "held vertex {vertex} is {arrival}, not {exact_s}"
                );
            } else {
                assert!(
                    arrival >= exact_s - 1e-12,
                    "vertex {vertex} arrives at {arrival}, before the straight line {exact_s}"
                );
                assert!(
                    arrival - exact_s < 0.1,
                    "vertex {vertex} is {} s late, past first order",
                    arrival - exact_s
                );
            }
        }
        // The claim this module rests on: a bounded number of visits per vertex.
        let per_vertex = exact(report.vertex_updates) / exact(times.len());
        assert!(
            per_vertex < 12.0,
            "{per_vertex:.1} vertex updates per vertex is not the O(N) Fu et al. claim"
        );
    }

    /// Bad inputs are named in this module's own vocabulary, not clamped.
    #[test]
    fn a_face_no_wave_can_cross_is_refused_by_name() {
        let (vertices, faces) = lattice(5, 4.0);
        let mut slowness = vec![0.4; faces.len() / 3];
        slowness[3] = 0.0;
        let error = solve(
            &vertices,
            &faces,
            &slowness,
            &[Boundary {
                vertices: &[0],
                times_s: &[0.0],
            }],
            1,
        )
        .expect_err("a face of zero slowness is not a medium");
        assert_eq!(
            error,
            Error::NonPositiveSlowness {
                face: 3,
                value: 0.0
            }
        );
        assert!(error.to_string().contains("unreachable"), "{error}");
    }

    #[test]
    fn an_out_of_bounds_boundary_is_refused_by_name() {
        let (vertices, faces) = lattice(5, 4.0);
        let slowness = vec![0.4; faces.len() / 3];
        let error = solve(
            &vertices,
            &faces,
            &slowness,
            &[Boundary {
                vertices: &[999],
                times_s: &[0.0],
            }],
            1,
        )
        .expect_err("vertex 999 is off a 25-vertex mesh");
        assert_eq!(
            error,
            Error::BoundaryOutOfBounds {
                boundary: 0,
                vertex: 999,
                vertices: 25
            }
        );
    }

    #[test]
    fn no_boundary_and_a_mismatched_one_are_refused() {
        let (vertices, faces) = lattice(5, 4.0);
        let slowness = vec![0.4; faces.len() / 3];
        assert_eq!(
            solve(&vertices, &faces, &slowness, &[], 1),
            Err(Error::NoBoundary)
        );
        assert_eq!(
            solve(
                &vertices,
                &faces,
                &slowness,
                &[Boundary {
                    vertices: &[0, 1],
                    times_s: &[0.0]
                }],
                1
            ),
            Err(Error::MismatchedBoundary {
                vertices: 2,
                times: 1
            })
        );
    }

    /// A multi-boundary solve is the pointwise minimum of its single-boundary solves.
    #[test]
    fn several_boundaries_are_the_min_of_their_separate_solves() {
        let (vertices, faces) = lattice(17, 8.0);
        let slowness: Vec<f64> = (0..faces.len() / 3)
            .map(|face| 0.3 + 0.2 * exact(face % 7) / 7.0)
            .collect();
        let first = Boundary {
            vertices: &[0],
            times_s: &[0.0],
        };
        let second = Boundary {
            vertices: &[288],
            times_s: &[0.75],
        };
        let (together, _) = solve(&vertices, &faces, &slowness, &[first, second], 1).unwrap();
        let (alone_a, _) = solve(&vertices, &faces, &slowness, &[first], 1).unwrap();
        let (alone_b, _) = solve(&vertices, &faces, &slowness, &[second], 1).unwrap();
        for (index, &arrival) in together.iter().enumerate() {
            assert_eq!(
                arrival.to_bits(),
                alone_a[index].min(alone_b[index]).to_bits(),
                "vertex {index}"
            );
        }
    }

    /// A component with no boundary in it says so rather than returning infinity.
    #[test]
    fn a_component_with_no_boundary_is_refused() {
        let (left, left_faces) = lattice(5, 4.0);
        let mut vertices = left.clone();
        for point in left.chunks_exact(3) {
            vertices.extend_from_slice(&[point[0] + 100.0, point[1], point[2]]);
        }
        let mut faces = left_faces.clone();
        let offset = i64::try_from(left.len() / 3).unwrap();
        faces.extend(left_faces.iter().map(|&vertex| vertex + offset));
        let slowness = vec![0.4; faces.len() / 3];
        let error = solve(
            &vertices,
            &faces,
            &slowness,
            &[Boundary {
                vertices: &[0],
                times_s: &[0.0],
            }],
            1,
        )
        .expect_err("the right-hand patch holds no boundary");
        assert!(matches!(error, Error::Unreachable { .. }), "{error}");
        assert!(error.to_string().contains("never reached"), "{error}");
    }
}
