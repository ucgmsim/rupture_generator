//! What the triangular first-arrival solver must satisfy, whatever ordering it uses.
//!
//! The counterpart to `eikonal_contract.rs`, and built the same way: every property is
//! quantified over generated meshes, slowness fields and boundaries, because a property
//! asserted at a single point is a smoke test rather than a contract. Accuracy is judged
//! against **analytic truth** — a uniform medium, where the answer is `s·d` — never
//! against another solver.
//!
//! # What is asserted, and what is deliberately not
//!
//! The false-sounding-but-true list is inherited from the Cartesian contract: travel time
//! does not increase with distance in a heterogeneous medium, the global Lipschitz bound
//! is violated by a first-order scheme, and the worst *relative* error does not converge.
//! Two more are specific to this solver:
//!
//! - **Causality holds over the extended one-ring, not the mesh one-ring.** Kimmel &
//!   Sethian's virtual edges reach past an obtuse wedge into the next triangle, and Fu et
//!   al. note they "are not considered part of the mesh; they are used only in the
//!   solver". A vertex whose wedge was unfolded can legitimately be reached from a vertex
//!   it shares no edge with.
//! - **The answer is not bit-reproducible across thread counts within one solve**, and
//!   that is asserted as a bound rather than as an equality. Across *seeds* it is
//!   bit-identical, and that is asserted as one. `crates/kernels/src/fim.rs` measures
//!   both.

// Every integer converted below is a vertex index, a face index or a cell count on a
// mesh a test builds -- the largest is 17x17. The crate makes this suppression once, in
// `src/counts.rs`, with the same bound stated; a test that generated its own `exact`
// wrapper would be a second copy of that argument.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "indices and counts here are test-mesh sized, orders below 2^53"
)]

use _kernels::fim::{self, Boundary, Error};
use proptest::prelude::*;

/// A right-angled lattice of `cells²` vertices over an `extent_km` square.
///
/// The mesh a planar fault is normally cut into, and the one case with no obtuse corner
/// in it: the two split triangles have angles 90, 45, 45, so the virtual-edge machinery
/// is not exercised and every property below is a statement about the local solver alone.
fn lattice(cells: usize, extent_km: f64) -> (Vec<f64>, Vec<i64>) {
    let step = extent_km / (cells - 1) as f64;
    let mut vertices = Vec::with_capacity(cells * cells * 3);
    for down in 0..cells {
        for across in 0..cells {
            vertices.extend_from_slice(&[across as f64 * step, 0.0, down as f64 * step]);
        }
    }
    let at = |down: usize, across: usize| (down * cells + across) as i64;
    let mut faces = Vec::new();
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

/// The same lattice, lifted off its own plane into a genuinely curved patch.
///
/// `h(u, v) = A sin(pi u / L) sin(pi v / L)`, which at `A = 1.6` on a 16 km patch reaches
/// `|grad h| ~ 0.31` — the `|grad h| <~ 0.33` `MESH.md` measures on the worst shipped
/// surface. It carries obtuse corners, so it is where the unfolding is exercised.
fn warped(cells: usize, extent_km: f64, amplitude_km: f64) -> (Vec<f64>, Vec<i64>) {
    let (mut vertices, faces) = lattice(cells, extent_km);
    for point in vertices.chunks_exact_mut(3) {
        point[1] = amplitude_km
            * (std::f64::consts::PI * point[0] / extent_km).sin()
            * (std::f64::consts::PI * point[2] / extent_km).sin();
    }
    (vertices, faces)
}

/// A mesh, a medium and one or more boundaries: everything one solve needs.
#[derive(Clone, Debug)]
struct Problem {
    vertices: Vec<f64>,
    faces: Vec<i64>,
    slowness: Vec<f64>,
    held: Vec<Vec<i64>>,
    held_s: Vec<Vec<f64>>,
}

impl Problem {
    fn count(&self) -> usize {
        self.vertices.len() / 3
    }

    fn boundaries(&self) -> Vec<Boundary<'_>> {
        self.held
            .iter()
            .zip(&self.held_s)
            .map(|(vertices, times_s)| Boundary { vertices, times_s })
            .collect()
    }

    fn solve(&self, threads: usize) -> Vec<f64> {
        fim::solve(
            &self.vertices,
            &self.faces,
            &self.slowness,
            &self.boundaries(),
            threads,
        )
        .expect("a generated problem is a valid input")
        .0
    }

    fn position(&self, vertex: usize) -> [f64; 3] {
        let at = vertex * 3;
        [
            self.vertices[at],
            self.vertices[at + 1],
            self.vertices[at + 2],
        ]
    }

    fn span_km(&self, first: usize, second: usize) -> f64 {
        let (a, b) = (self.position(first), self.position(second));
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// The slowest face carrying each edge, which is the sharp Lipschitz constant for it.
    fn edge_slowness(&self) -> std::collections::HashMap<(usize, usize), f64> {
        let mut slowest = std::collections::HashMap::new();
        for (face, corners) in self.faces.chunks_exact(3).enumerate() {
            for pair in 0..3 {
                let (a, b) = (
                    corners[pair] as usize,
                    corners[(pair + 1) % 3] as usize,
                );
                let key = (a.min(b), a.max(b));
                let entry = slowest.entry(key).or_insert(0.0_f64);
                *entry = entry.max(self.slowness[face]);
            }
        }
        slowest
    }
}

/// Meshes to quantify over: a flat lattice with no obtuse corner, and a warped one with.
fn mesh() -> impl Strategy<Value = (Vec<f64>, Vec<i64>)> {
    prop_oneof![
        (9usize..17).prop_map(|cells| lattice(cells, 16.0)),
        (9usize..17).prop_map(|cells| warped(cells, 16.0, 1.6)),
    ]
}

/// A problem with `seeds` single-vertex boundaries and a smoothly varying medium.
fn problem(seeds: usize) -> impl Strategy<Value = Problem> {
    (mesh(), 0.15f64..0.9, 0.0f64..0.6, any::<u8>()).prop_map(
        move |((vertices, faces), base, wobble, offset)| {
            let count = vertices.len() / 3;
            let slowness: Vec<f64> = faces
                .chunks_exact(3)
                .map(|corners| {
                    let depth: f64 = corners
                        .iter()
                        .map(|&corner| vertices[corner as usize * 3 + 2])
                        .sum::<f64>()
                        / 3.0;
                    base * (1.0 + wobble * (0.7 * depth).sin())
                })
                .collect();
            let mut held = Vec::new();
            let mut held_s = Vec::new();
            for seed in 0..seeds {
                let vertex = (usize::from(offset) * 7 + seed * 101) % count;
                held.push(vec![vertex as i64]);
                held_s.push(vec![seed as f64 * 0.8]);
            }
            Problem {
                vertices,
                faces,
                slowness,
                held,
                held_s,
            }
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// A multi-boundary solve is the pointwise minimum of its single-boundary solves.
    ///
    /// The property `MESH.md` says carries over from `eikonal.rs` unchanged, and the one
    /// the across-seed threading rests on. Asserted **to the bit**, deliberately: each
    /// boundary owns its own array, so there is nothing for a race to perturb, and a
    /// future single-pass multi-source scheme would have to loosen this equality here,
    /// with the tolerance argued in front of it.
    #[test]
    fn a_multi_seed_solve_is_the_min_of_its_single_seed_solves(problem in problem(3)) {
        let together = problem.solve(1);
        let mut best = vec![f64::INFINITY; problem.count()];
        for index in 0..problem.held.len() {
            let alone = Problem {
                held: vec![problem.held[index].clone()],
                held_s: vec![problem.held_s[index].clone()],
                ..problem.clone()
            }
            .solve(1);
            for (slot, &arrival) in best.iter_mut().zip(&alone) {
                *slot = slot.min(arrival);
            }
        }
        for (vertex, (&all, &min)) in together.iter().zip(&best).enumerate() {
            prop_assert_eq!(
                all.to_bits(), min.to_bits(),
                "vertex {}: {} from the joint solve, {} from the min of singles",
                vertex, all, min
            );
        }
    }

    /// Threading across seeds does not move a single bit.
    ///
    /// The counterpart to the measured 9.1e-4 s spread *within* one solve: across seeds
    /// there is no shared array, so there is nothing to be non-deterministic about, and
    /// the equality is the evidence that the two levels really are separate.
    #[test]
    fn threading_across_seeds_is_bit_identical(problem in problem(4)) {
        let sequential = problem.solve(1);
        for threads in [2, 4] {
            let threaded = problem.solve(threads);
            for (vertex, (&one, &many)) in sequential.iter().zip(&threaded).enumerate() {
                prop_assert_eq!(
                    one.to_bits(), many.to_bits(),
                    "vertex {} moved from {} to {} at {} threads",
                    vertex, one, many, threads
                );
            }
        }
    }

    /// A lone seed's vertex ruptures exactly when it was seeded, not merely close to it.
    ///
    /// The pinned hypocentre `MESH.md` calls out as having no perturbation to hide behind:
    /// `stages.apply_perturbation` sets its noise to zero, so its onset is exactly travel
    /// time plus delay and it is the registration point every diagnostic is measured
    /// from.
    #[test]
    fn a_seed_never_ruptures_after_its_own_start_time(problem in problem(2)) {
        let times = problem.solve(1);
        for (held, held_s) in problem.held.iter().zip(&problem.held_s) {
            prop_assert!(times[held[0] as usize] <= held_s[0] + 1e-12);
        }
        let alone = Problem {
            held: vec![problem.held[0].clone()],
            held_s: vec![problem.held_s[0].clone()],
            ..problem.clone()
        }
        .solve(1);
        prop_assert_eq!(
            alone[problem.held[0][0] as usize].to_bits(),
            problem.held_s[0][0].to_bits()
        );
    }

    /// No edge is crossed faster than the slowest face carrying it.
    ///
    /// The sharp Lipschitz statement, per edge, because a triangulation has no single
    /// spacing to state it in. Structural: the one-sided cap enforces
    /// `T(c) <= T(n) + |edge| s` by construction for every real edge. Quantified over
    /// real edges only — a virtual edge is a device for reading a direction, not a path.
    #[test]
    fn neighbouring_vertices_are_lipschitz(problem in problem(1)) {
        let times = problem.solve(1);
        for (&(first, second), &slowest) in &problem.edge_slowness() {
            let step_s = (times[first] - times[second]).abs();
            let bound_s = problem.span_km(first, second) * slowest;
            prop_assert!(
                step_s <= bound_s * (1.0 + 1e-9),
                "edge {}-{} jumps {} s, past the {} s crossing it at the slowest face takes",
                first, second, step_s, bound_s
            );
        }
    }

    /// A faster medium never ruptures later.
    ///
    /// Two solves and no analytic solution needed. It catches the error every bound is
    /// weakest against: a solver that inverts its input, using speed where slowness
    /// belongs, still produces a plausible field but reverses this.
    #[test]
    fn a_faster_medium_never_ruptures_later(problem in problem(1)) {
        let slow = problem.solve(1);
        let quick = Problem {
            slowness: problem.slowness.iter().map(|value| value * 0.8).collect(),
            ..problem.clone()
        }
        .solve(1);
        for (vertex, (&was, &now)) in slow.iter().zip(&quick).enumerate() {
            prop_assert!(
                now <= was + 1e-12,
                "vertex {} got later ({} from {}) when the medium got faster",
                vertex, now, was
            );
        }
    }

    /// Every vertex is consistent with its own one-ring when the solver returns.
    ///
    /// **The property Fu et al.'s Algorithm 2.1 does not give on its own**, and the
    /// reason `fim.rs` runs a consistency scan around the sweep. The paper's removal
    /// condition takes a vertex off the active list when its own value stops moving,
    /// which is not the same as the vertex being consistent with its neighbours: two
    /// adjacent vertices can both stop moving in the same visit while each still owes the
    /// other an update, and nothing puts them back. Asserted here as "no vertex would
    /// move if asked again", which is what a fixed point means.
    ///
    /// **One boundary, and that is not a weakening.** A multi-seed field is the pointwise
    /// minimum of per-seed solves, and the minimum of two fixed points of a monotone
    /// update is a *super*-solution rather than a fixed point: `F(min(a, b)) <= F(a) = a`
    /// and likewise for `b`, so the update can still lower it. That is a property of the
    /// per-seed-then-minimum contract `eikonal.rs` documents and this module inherits --
    /// solving separately is what keeps each source's near field exact -- and not of the
    /// iteration. Measured on a two-seed problem the residual reaches 1.8e-2 s, entirely
    /// from that effect.
    #[test]
    fn the_answer_is_a_fixed_point_of_its_own_update(problem in problem(1)) {
        let times = problem.solve(1);
        let residual = fim::residual(
            &problem.vertices,
            &problem.faces,
            &problem.slowness,
            &times,
            &problem.boundaries(),
        )
        .expect("a generated problem is a valid input");
        prop_assert!(
            residual <= 1e-9,
            "the worst vertex is {} s from satisfying its own update equation",
            residual
        );
    }
}

// ---------------------------------------------------------------------------------
// Accuracy on the medium where truth is known
// ---------------------------------------------------------------------------------

/// Worst absolute error against `s·d` on a uniform medium, seeded on a disc of radius 3.
///
/// The **circular** boundary condition of Fu et al. §3.3.2, which is the one their own
/// convergence study reports slope 1.0 for. Held at a fixed radius under refinement,
/// because a boundary that shrank with the mesh would be a different problem at every
/// resolution and the slope would measure the sequence rather than the convergence.
fn circular_error(cells: usize) -> f64 {
    let (extent_km, slowness_s_per_km) = (16.0, 0.4);
    let (vertices, faces) = lattice(cells, extent_km);
    let centre = [extent_km / 2.0, 0.0, extent_km / 2.0];
    let radius: Vec<f64> = vertices
        .chunks_exact(3)
        .map(|point| ((point[0] - centre[0]).powi(2) + (point[2] - centre[2]).powi(2)).sqrt())
        .collect();
    let held: Vec<i64> = radius
        .iter()
        .enumerate()
        .filter(|&(_, &span)| span <= 3.0)
        .map(|(vertex, _)| vertex as i64)
        .collect();
    let held_s: Vec<f64> = held
        .iter()
        .map(|&vertex| slowness_s_per_km * radius[vertex as usize])
        .collect();

    let (times, _) = fim::solve(
        &vertices,
        &faces,
        &vec![slowness_s_per_km; faces.len() / 3],
        &[Boundary {
            vertices: &held,
            times_s: &held_s,
        }],
        1,
    )
    .expect("a uniform lattice is a valid input");

    radius
        .iter()
        .zip(&times)
        .filter(|&(&span, _)| span > 3.0)
        .map(|(&span, &arrival)| (arrival - slowness_s_per_km * span).abs())
        .fold(0.0_f64, f64::max)
}

/// Halving the edge length at least halves the error: first order, as the paper claims.
///
/// Fu et al. §3.3.2 verbatim: "For the circular boundary conditions, the slope of this
/// graph is 1.0, which is consistent to our claim that meshFIM is first-order accurate."
/// The bound here is 1.7 against a measured ratio near 2.0, loose on purpose — the
/// discriminating comparison is with the *point* boundary condition, which the same paper
/// reports as not first-order accurate and which `tests/triangular/test_fim.py` measures
/// at slope 0.73 through the numpy implementation of this same method.
#[test]
fn the_circular_boundary_error_converges_at_first_order() {
    let coarse = circular_error(33);
    let fine = circular_error(65);
    let ratio = coarse / fine;
    assert!(
        ratio > 1.7,
        "error {coarse:.5e} at 33 cells and {fine:.5e} at 65 is a ratio of {ratio:.2}; \
         below first order"
    );
}

/// A uniform medium is reproduced exactly on the boundary, and to first order off it.
#[test]
fn a_uniform_medium_is_exact_on_its_own_boundary() {
    let error = circular_error(33);
    assert!(error < 0.1, "{error} is not first-order accuracy at h = 0.5 km");
}

// ---------------------------------------------------------------------------------
// Refusals: bad inputs are named, not solved around
// ---------------------------------------------------------------------------------

fn small() -> (Vec<f64>, Vec<i64>, Vec<f64>) {
    let (vertices, faces) = lattice(5, 4.0);
    let slowness = vec![0.4; faces.len() / 3];
    (vertices, faces, slowness)
}

fn seed_at(vertex: i64) -> (Vec<i64>, Vec<f64>) {
    (vec![vertex], vec![0.0])
}

#[test]
fn a_face_no_wave_can_cross_is_refused_by_name() {
    let (vertices, faces, mut slowness) = small();
    let (held, held_s) = seed_at(0);
    for bad in [0.0, -0.3, f64::NAN, f64::INFINITY] {
        slowness[7] = bad;
        let error = fim::solve(
            &vertices,
            &faces,
            &slowness,
            &[Boundary {
                vertices: &held,
                times_s: &held_s,
            }],
            1,
        )
        .expect_err("face 7 is not a medium");
        assert!(
            matches!(error, Error::NonPositiveSlowness { face: 7, .. }),
            "{bad} on face 7 gave {error}"
        );
        assert!(error.to_string().contains("unreachable"), "{error}");
    }
}

#[test]
fn an_out_of_bounds_boundary_is_refused_by_name() {
    let (vertices, faces, slowness) = small();
    for bad in [-1, 25, 9999] {
        let (held, held_s) = seed_at(bad);
        let error = fim::solve(
            &vertices,
            &faces,
            &slowness,
            &[Boundary {
                vertices: &held,
                times_s: &held_s,
            }],
            1,
        )
        .expect_err("vertex is off a 25-vertex mesh");
        assert_eq!(
            error,
            Error::BoundaryOutOfBounds {
                boundary: 0,
                vertex: bad,
                vertices: 25
            }
        );
    }
}

#[test]
fn a_boundary_with_no_time_is_refused() {
    let (vertices, faces, slowness) = small();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = fim::solve(
            &vertices,
            &faces,
            &slowness,
            &[Boundary {
                vertices: &[0],
                times_s: &[bad],
            }],
            1,
        )
        .expect_err("a held time must be finite");
        assert!(
            matches!(error, Error::NonFiniteBoundaryTime { boundary: 0, .. }),
            "t = {bad} gave {error}"
        );
    }
}

#[test]
fn no_boundary_no_mesh_and_wrong_lengths_are_refused() {
    let (vertices, faces, slowness) = small();
    assert_eq!(
        fim::solve(&vertices, &faces, &slowness, &[], 1),
        Err(Error::NoBoundary)
    );
    assert_eq!(
        fim::solve(
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
    assert_eq!(
        fim::solve(
            &vertices,
            &faces,
            &slowness[..3],
            &[Boundary {
                vertices: &[0],
                times_s: &[0.0]
            }],
            1
        ),
        Err(Error::WrongLength {
            faces: faces.len() / 3,
            got: 3
        })
    );
    assert_eq!(
        fim::solve(
            &[],
            &[],
            &[],
            &[Boundary {
                vertices: &[0],
                times_s: &[0.0]
            }],
            1
        ),
        Err(Error::EmptyMesh {
            vertices: 0,
            faces: 0
        })
    );
}

#[test]
fn a_face_naming_a_vertex_the_mesh_lacks_is_refused() {
    let (vertices, mut faces, slowness) = small();
    faces[2] = 999;
    let error = fim::solve(
        &vertices,
        &faces,
        &slowness,
        &[Boundary {
            vertices: &[0],
            times_s: &[0.0],
        }],
        1,
    )
    .expect_err("face 0 names a vertex off the mesh");
    assert_eq!(
        error,
        Error::FaceOutOfBounds {
            face: 0,
            vertex: 999,
            vertices: 25
        }
    );
}

#[test]
fn a_triangle_with_no_thickness_is_refused() {
    let vertices = vec![
        0.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, //
        2.0, 0.0, 0.0, //
        0.0, 0.0, 1.0,
    ];
    let faces = vec![0, 1, 2, 0, 1, 3];
    let error = fim::solve(
        &vertices,
        &faces,
        &[0.4, 0.4],
        &[Boundary {
            vertices: &[0],
            times_s: &[0.0],
        }],
        1,
    )
    .expect_err("face 0 is collinear");
    assert!(
        matches!(error, Error::DegenerateTriangle { face: 0, .. }),
        "{error}"
    );
    assert!(error.to_string().contains("line segment"), "{error}");
}

#[test]
fn a_component_with_no_boundary_is_refused() {
    let (left, left_faces, _) = small();
    let mut vertices = left.clone();
    for point in left.chunks_exact(3) {
        vertices.extend_from_slice(&[point[0] + 100.0, point[1], point[2]]);
    }
    let mut faces = left_faces.clone();
    let offset = (left.len() / 3) as i64;
    faces.extend(left_faces.iter().map(|&vertex| vertex + offset));
    let slowness = vec![0.4; faces.len() / 3];
    let error = fim::solve(
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
