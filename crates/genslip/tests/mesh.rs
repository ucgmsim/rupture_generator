//! The mesh puts subfaults where it says it does.
//!
//! **Contract**, in `ENGINEERING_RULES.md`'s sense: a red here is a bug, never an
//! argument.
//!
//! # The reference is geometry, not a second transcription
//!
//! There is no C function to call: genslip is *given* a GSF and does not build one. Nor
//! is `tests/harness/gsf.py:on_a_plane` a reference — it is a deliberate flat-earth
//! layout for fixture input, and its own docstring disclaims being geodesy.
//!
//! So the reference is what the geometry has to be true of regardless of how it was
//! computed: adjacent nodes are one cell apart, a due-north trace runs due north, areas
//! sum to length times width, a plane's cells all report the plane's own strike and dip.
//! `README.md`'s fourth trap is the reason to insist on that — a reference side that
//! re-implements the code it checks agrees with it bit for bit while both are wrong,
//! which is what `DEFECTS.md` 17 and 18 both were.
//!
//! # Why the bounds are tiny
//!
//! The frame is a projected Cartesian one, so there is no curvature and nothing here is
//! an approximation of anything. Every quantity below is an identity, and the only slack
//! is `f64` round-off.
//!
//! Getting it down to round-off took one design decision, which is worth recording
//! because the first version did not have it. **A projected coordinate is large and a
//! subfault is small**: an NZTM northing runs to about 5,180 km against a cell of about
//! 1 km. When the mesh held absolute coordinates, every node was rounded at CRS scale —
//! an absolute error of `f64::EPSILON * 5180 ~= 5.7e-13 km` — and every cell-scale
//! quantity derived by differencing two of them inherited a **relative** error of
//! `1.2e-12`. Measured: it is what `strike_adjacent_nodes_are_one_cell_apart` and
//! `a_plane_has_cells_of_one_size` failed by.
//!
//! The mesh holds *offsets from an origin* instead, and reduces its trace to offsets
//! before any other arithmetic, so every number in the construction is at fault scale.
//! The same assertions now sit at **3e-15**, a factor of 400 better, and `EXACT` is two
//! orders above that.
//!
//! One floor is left, and it is not the code's. A quantity compared against a *nominal
//! input* — "this fault is 20 km long" — is limited by the resolution of the coordinate
//! that input was written in, which at NZTM scale is `f64::EPSILON * 5180 / 20 ~= 6e-14`
//! relative. That is what a coordinate in that CRS is worth, and no arithmetic recovers
//! it. `EXACT` sits above it.
//!
//! Angles are unaffected — they come from normalised directions — so those stay at
//! `1e-9`, and the ceiling for them is the 1 degree the SRF stores.
//!
//! An earlier draft of this module worked on the WGS84 ellipsoid, and the same assertions
//! needed `5e-3` on area and `1e-2` degrees on angles, floored by a curvature effect that
//! had to be measured before it could be bounded. That is what the projection bought.

use approx::assert_relative_eq;
use genslip::error::Error;
use genslip::grid::FaultAxes;
use genslip::mesh::{
    self, Cuts, DipDirection, Fault, Geometry, PatchView, Plane, PointSpec, Projected, RefinedMesh,
    Vertex,
};
use genslip::rupture::Hypocentre;
use proptest::prelude::*;

/// Everything here is an identity, so this is round-off and nothing else. Observed floor
/// 3e-15; ceiling the 1e-2 slip bound. See the module note for the derivation.
const EXACT: f64 = 1.0e-13;

fn at(easting_km: f64, northing_km: f64) -> Projected {
    Projected {
        easting_km,
        northing_km,
    }
}

/// A fault of one plane, given by a bearing and a length so a test can say "20 km at 055"
/// and have that be true rather than true to however many decimals got typed.
fn one_plane(
    bearing_deg: f64,
    length_km: f64,
    dip_deg: f64,
    top_depth_km: f64,
    bottom_depth_km: f64,
) -> Fault {
    let origin = at(1_500.0, 5_180.0);
    Fault {
        origin,
        top_depth_km,
        first: Plane {
            end: origin.along(bearing_deg, length_km),
            dip_deg,
            dip_direction: DipDirection::Right,
            bottom_depth_km,
        },
        rest: Vec::new(),
    }
}

/// The default fixture: 20 km at 055 degrees, dipping 60, from the surface to 12 km.
fn a_plane() -> Fault {
    one_plane(55.0, 20.0, 60.0, 0.0, 12.0)
}

/// Its down-dip width: 12 km of depth at 60 degrees.
fn a_plane_width_km() -> f64 {
    12.0 / 60.0_f64.to_radians().sin()
}

const CUT: Cuts = Cuts {
    strike_count: 20,
    dip_count: 12,
};

fn refined(fault: &Fault, cuts: &[Cuts]) -> RefinedMesh {
    mesh::build(&Geometry::Fault(fault.clone()))
        .expect("the fixture is valid")
        .refine(cuts)
        .expect("the cuts match the faces")
}

/// The default fixture, built and cut.
fn a_patch_of(fault: &Fault, cuts: Cuts) -> RefinedMesh {
    refined(fault, &[cuts])
}

fn refusal(geometry: &Geometry) -> Error {
    mesh::build(geometry).expect_err("expected a refusal")
}

fn separation(patch: &PatchView<'_>, from: [usize; 2], to: [usize; 2]) -> f64 {
    let positions = patch.positions();
    let delta = |field: &ndarray::Array2<f64>| field[to] - field[from];
    let (east, north, down) = (
        delta(&positions.east_km),
        delta(&positions.north_km),
        delta(&positions.depth_km),
    );
    (east * east + north * north + down * down).sqrt()
}

// ---------------------------------------------------------------------------
// The coarse mesh
// ---------------------------------------------------------------------------

/// A fault of N planes is a mesh of N faces and 3N+1 vertices.
///
/// Three per plane rather than four, because the vertex between two planes is one vertex.
/// That count *is* the connectivity claim: four per plane would mean the shared corner had
/// been written down twice.
#[test]
fn planes_share_their_trace_vertex() {
    let origin = at(1_500.0, 5_180.0);
    let bend = origin.along(40.0, 10.0);
    let fault = Fault {
        origin,
        top_depth_km: 0.0,
        first: Plane {
            end: bend,
            dip_deg: 70.0,
            dip_direction: DipDirection::Right,
            bottom_depth_km: 14.0,
        },
        rest: vec![Plane {
            end: bend.along(80.0, 6.0),
            // Its own dip, which the fused single-grid design could not allow.
            dip_deg: 50.0,
            dip_direction: DipDirection::Right,
            bottom_depth_km: 10.0,
        }],
    };

    let mesh = mesh::build(&Geometry::Fault(fault)).expect("a two-plane fault is valid");
    assert_eq!(mesh.faces().len(), 2);
    assert_eq!(
        mesh.vertices().len(),
        7,
        "3 per plane plus the shared corner"
    );

    // The first plane's far top corner and the second's near top corner are the same
    // *index*, not two positions that agree.
    assert_eq!(
        mesh.faces()[0][1],
        mesh.faces()[1][0],
        "the planes do not share a vertex index"
    );
}

/// A single plane is four vertices and one face.
#[test]
fn one_plane_is_one_face() {
    let mesh = mesh::build(&Geometry::Fault(a_plane())).expect("valid");
    assert_eq!(mesh.faces().len(), 1);
    assert_eq!(mesh.vertices().len(), 4);
}

// ---------------------------------------------------------------------------
// Refinement
// ---------------------------------------------------------------------------

/// A patch has one more node than cell along each axis.
///
/// The off-by-one this file exists to make impossible. A grid of centres and a grid of
/// corners are the same shape to anything that only counts elements.
#[test]
fn there_is_one_more_node_than_cell_on_each_axis() {
    let mesh = a_patch_of(&a_plane(), CUT);
    let patch = mesh.patch(0);
    assert_eq!(patch.cell_extents(), (20, 12));
    assert_eq!(patch.node_extents(), (21, 13));
    assert_eq!(patch.positions().depth_km.dim(), (13, 21));
}

/// Refinement reuses the face's own corners and invents the rest.
///
/// The sharing rule, asserted by counting. A 20x12 patch has 21x13 nodes; four of them are
/// the coarse mesh's vertices, so refinement adds exactly `21*13 - 4`.
#[test]
fn refinement_reuses_the_faces_corners() {
    let coarse = mesh::build(&Geometry::Fault(a_plane())).expect("valid");
    let fine = coarse.refine(&[CUT]).expect("valid");
    assert_eq!(fine.vertices().len(), 21 * 13 - 4 + coarse.vertices().len());

    let patch = fine.patch(0);
    let face = coarse.faces()[0];
    assert_eq!(patch.node_index([0, 0]), face[0]);
    assert_eq!(patch.node_index([0, 20]), face[1]);
    assert_eq!(patch.node_index([12, 20]), face[2]);
    assert_eq!(patch.node_index([12, 0]), face[3]);
}

/// A one-by-one cut is the face itself: no vertex is created.
///
/// The degenerate case, and the one a point source takes.
#[test]
fn a_single_cut_creates_no_vertices() {
    let coarse = mesh::build(&Geometry::Fault(a_plane())).expect("valid");
    let fine = coarse
        .refine(&[Cuts {
            strike_count: 1,
            dip_count: 1,
        }])
        .expect("valid");
    assert_eq!(fine.vertices().len(), coarse.vertices().len());
}

/// Refined patches keep the coarse mesh's sharing, and add none of their own.
#[test]
fn refined_patches_share_the_corner_their_faces_shared() {
    let origin = at(1_500.0, 5_180.0);
    let bend = origin.along(40.0, 10.0);
    let fault = Fault {
        origin,
        top_depth_km: 0.0,
        first: Plane {
            end: bend,
            dip_deg: 70.0,
            dip_direction: DipDirection::Right,
            bottom_depth_km: 14.0,
        },
        rest: vec![Plane {
            end: bend.along(80.0, 6.0),
            dip_deg: 50.0,
            dip_direction: DipDirection::Right,
            bottom_depth_km: 10.0,
        }],
    };
    let fine = refined(
        &fault,
        &[
            Cuts {
                strike_count: 10,
                dip_count: 8,
            },
            Cuts {
                strike_count: 6,
                dip_count: 4,
            },
        ],
    );

    assert_eq!(fine.patch_count(), 2);
    assert_eq!(
        fine.patch(0).node_index([0, 10]),
        fine.patch(1).node_index([0, 0]),
        "the shared trace corner was duplicated by refinement"
    );
}

/// The wrong number of cuts is refused, and zero cells are refused.
#[test]
fn refinement_refuses_what_it_cannot_cut() {
    let coarse = mesh::build(&Geometry::Fault(a_plane())).expect("valid");

    assert!(matches!(
        coarse.refine(&[CUT, CUT]),
        Err(Error::Shape {
            what: "cuts",
            found: 2,
            expected: 1
        })
    ));
    assert!(matches!(coarse.refine(&[]), Err(Error::Shape { .. })));

    for cuts in [
        Cuts {
            strike_count: 0,
            dip_count: 12,
        },
        Cuts {
            strike_count: 20,
            dip_count: 0,
        },
    ] {
        assert!(matches!(
            coarse.refine(&[cuts]),
            Err(Error::MeshTooSmall { .. })
        ));
    }
}

/// Every cell becomes two triangles, and every index is a real vertex.
#[test]
fn triangles_cover_every_cell() {
    let fine = a_patch_of(&a_plane(), CUT);
    let (vertices, triangles) = fine.triangles();
    assert_eq!(triangles.len(), 2 * 20 * 12);
    assert!(
        triangles
            .iter()
            .flatten()
            .all(|index| *index < vertices.len()),
        "a triangle indexes a vertex that does not exist"
    );
}

// ---------------------------------------------------------------------------
// Node placement
// ---------------------------------------------------------------------------

/// Nodes adjacent along strike are exactly one cell apart, everywhere on the patch.
#[test]
fn strike_adjacent_nodes_are_one_cell_apart() {
    let fine = a_patch_of(&a_plane(), CUT);
    let patch = fine.patch(0);
    let expected_km = 20.0 / 20.0;
    let (strike_nodes, dip_nodes) = patch.node_extents();

    for dip in 0..dip_nodes {
        for strike in 0..strike_nodes - 1 {
            assert_relative_eq!(
                separation(&patch, [dip, strike], [dip, strike + 1]),
                expected_km,
                max_relative = EXACT
            );
        }
    }
}

/// Nodes adjacent down dip are one cell apart, measured in three dimensions.
///
/// The horizontal separation alone is `dw * cos(dip)`, so this is the assertion that
/// catches a missing depth component — which would place every subfault too shallow and
/// leave the map view looking perfect.
#[test]
fn dip_adjacent_nodes_are_one_cell_apart_in_three_dimensions() {
    let fine = a_patch_of(&a_plane(), CUT);
    let patch = fine.patch(0);
    let expected_km = a_plane_width_km() / 12.0;
    let (strike_nodes, dip_nodes) = patch.node_extents();

    for dip in 0..dip_nodes - 1 {
        for strike in 0..strike_nodes {
            assert_relative_eq!(
                separation(&patch, [dip, strike], [dip + 1, strike]),
                expected_km,
                max_relative = EXACT
            );
        }
    }
}

/// A due-north trace runs due north, and dips due east under the right-hand rule.
///
/// The axis convention. Every distance assertion above is blind to it: a mesh laid out
/// along the wrong axis has all the right separations.
#[test]
fn a_due_north_trace_runs_north_and_dips_east() {
    let fine = a_patch_of(&one_plane(0.0, 20.0, 60.0, 0.0, 12.0), CUT);
    let positions = fine.patch(0).positions();

    // Along the top edge: northing climbs, easting does not move.
    for strike in 1..=20 {
        assert_relative_eq!(
            positions.east_km[[0, strike]],
            positions.east_km[[0, 0]],
            epsilon = EXACT
        );
        assert!(
            positions.north_km[[0, strike]] > positions.north_km[[0, strike - 1]],
            "a due-north trace did not go north"
        );
    }

    // Down dip: easting climbs, because east is a quarter turn right of north.
    for dip in 1..=12 {
        assert!(
            positions.east_km[[dip, 0]] > positions.east_km[[dip - 1, 0]],
            "a right-dipping north-striking fault did not dip east"
        );
    }
}

/// Dipping left is dipping the other way, and nothing else changes.
#[test]
fn dipping_left_mirrors_the_surface() {
    let mut fault = one_plane(0.0, 20.0, 60.0, 0.0, 12.0);
    let right = a_patch_of(&fault, CUT).patch(0).positions();
    fault.first.dip_direction = DipDirection::Left;
    let left = a_patch_of(&fault, CUT).patch(0).positions();

    assert_eq!(right.depth_km, left.depth_km, "depth is not handed");
    for dip in 1..=12 {
        assert!(
            left.east_km[[dip, 0]] < left.east_km[[dip - 1, 0]],
            "a left-dipping north-striking fault did not dip west"
        );
    }
}

/// Depth steps evenly and does not vary along strike.
#[test]
fn depth_steps_evenly_and_is_flat_along_strike() {
    let fine = a_patch_of(&a_plane(), CUT);
    let depth_km = fine.patch(0).positions().depth_km;

    for dip in 0..=12 {
        let expected = f64::from(u32::try_from(dip).expect("small")) * (12.0 / 12.0);
        for strike in 0..=20 {
            assert_relative_eq!(depth_km[[dip, strike]], expected, epsilon = EXACT);
        }
    }
}

/// A fault that does not reach the surface starts where it was told to.
#[test]
fn a_buried_fault_starts_at_its_top_depth() {
    let fine = a_patch_of(&one_plane(55.0, 20.0, 60.0, 4.0, 16.0), CUT);
    let depth_km = fine.patch(0).positions().depth_km;
    assert_relative_eq!(depth_km[[0, 0]], 4.0, epsilon = EXACT);
    assert_relative_eq!(depth_km[[12, 0]], 16.0, epsilon = EXACT);
}

// ---------------------------------------------------------------------------
// Derived quantities
// ---------------------------------------------------------------------------

/// The areas sum to the fault's own length times its width, exactly.
#[test]
fn the_areas_sum_to_length_times_width() {
    let fine = a_patch_of(&a_plane(), CUT);
    let total: f64 = fine.patch(0).areas_km2().flat().iter().sum();
    assert_relative_eq!(total, 20.0 * a_plane_width_km(), max_relative = EXACT);
}

/// Every cell of a plane is the same size.
#[test]
fn a_plane_has_cells_of_one_size() {
    let fine = a_patch_of(&a_plane(), CUT);
    let areas = fine.patch(0).areas_km2();
    let first = areas.flat()[0];
    for area in areas.flat() {
        assert_relative_eq!(*area, first, max_relative = EXACT);
    }
}

/// Every cell of a plane reports the plane's own strike and dip.
///
/// The derived quantities have to give back what the constructor was told, or one of the
/// two is wrong and there is no way to tell which from inside.
#[test]
fn a_plane_reports_the_strike_and_dip_it_was_built_with() {
    let fine = a_patch_of(&a_plane(), CUT);
    let patch = fine.patch(0);

    for strike in patch.strike_deg().flat() {
        assert_relative_eq!(*strike, 55.0, epsilon = 1e-9);
    }
    for dip in patch.dip_deg().flat() {
        assert_relative_eq!(*dip, 60.0, epsilon = 1e-9);
    }
}

/// A vertical fault dips 90 degrees and its columns are plumb lines.
///
/// The boundary of the allowed range, where `cos(dip)` is zero. `tan(90°)` is enormous
/// rather than infinite in `f64`, so the horizontal reach is a very small number rather
/// than exactly zero — this is what pins how small.
#[test]
fn a_vertical_fault_is_vertical() {
    let fine = a_patch_of(&one_plane(55.0, 20.0, 90.0, 0.0, 12.0), CUT);
    let patch = fine.patch(0);

    for dip in patch.dip_deg().flat() {
        assert_relative_eq!(*dip, 90.0, epsilon = 1e-9);
    }

    let positions = patch.positions();
    for dip in 1..=12 {
        let drift = (positions.east_km[[dip, 0]] - positions.east_km[[0, 0]])
            .hypot(positions.north_km[[dip, 0]] - positions.north_km[[0, 0]]);
        assert!(drift < 1e-12, "a vertical fault moved {drift} km sideways");
    }
}

/// A cell's centre is the mean of its corners.
#[test]
fn centres_sit_in_the_middle_of_their_cells() {
    let fine = a_patch_of(&a_plane(), CUT);
    let patch = fine.patch(0);
    let centres = patch.centres();
    let positions = patch.positions();
    assert_eq!(centres.depth_km.extent(), (20, 12));

    for dip in 0..12 {
        for strike in 0..20 {
            for (centre, node) in [
                (&centres.east_km, &positions.east_km),
                (&centres.north_km, &positions.north_km),
                (&centres.depth_km, &positions.depth_km),
            ] {
                let expected = 0.25
                    * (node[[dip, strike]]
                        + node[[dip, strike + 1]]
                        + node[[dip + 1, strike + 1]]
                        + node[[dip + 1, strike]]);
                assert_relative_eq!(centre[[dip, strike]], expected, epsilon = EXACT);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// In-fault coordinates
// ---------------------------------------------------------------------------

/// The arc lengths start at zero and end at the patch's extent, stepping evenly.
#[test]
fn the_arcs_span_the_patch() {
    let fine = a_patch_of(&a_plane(), CUT);
    let patch = fine.patch(0);
    let strike_arc = patch.strike_arc_km();
    let dip_arc = patch.dip_arc_km();

    assert_relative_eq!(strike_arc[0], 0.0, epsilon = EXACT);
    assert_relative_eq!(dip_arc[0], 0.0, epsilon = EXACT);
    assert_relative_eq!(strike_arc[20], 20.0, max_relative = EXACT);
    assert_relative_eq!(dip_arc[12], a_plane_width_km(), max_relative = EXACT);
}

/// A plane's spacing is its length over its count, on both axes.
#[test]
fn a_plane_has_a_uniform_spacing() {
    let fine = a_patch_of(&a_plane(), CUT);
    let spacing = fine.patch(0).spacing().expect("a plane is uniform");

    assert_relative_eq!(spacing.strike_km, 1.0, max_relative = EXACT);
    assert_relative_eq!(
        spacing.dip_km,
        a_plane_width_km() / 12.0,
        max_relative = EXACT
    );
}

/// A position in the middle of a cell comes back as that cell's index.
///
/// The round trip that `DEFECTS.md` 17 failed. It is asserted over every cell rather than
/// a sampled few, because the defect it guards against was a *constant* offset — one cell
/// in each direction — and nobody checked even one.
#[test]
fn a_position_in_a_cell_comes_back_as_that_cell() {
    let fine = a_patch_of(&a_plane(), CUT);
    let patch = fine.patch(0);
    let strike_arc = patch.strike_arc_km();
    let dip_arc = patch.dip_arc_km();

    for dip in 0..12 {
        for strike in 0..20 {
            let found = patch
                .cell_index(
                    0.5 * (strike_arc[strike] + strike_arc[strike + 1]),
                    0.5 * (dip_arc[dip] + dip_arc[dip + 1]),
                )
                .expect("the middle of a cell is on the fault");
            assert_eq!(found, Hypocentre { strike, dip });
        }
    }
}

/// The two far edges belong to the last cell rather than to no cell.
///
/// "At the bottom of the fault" is a thing people write, and the alternative is refusing a
/// position that is on the fault.
#[test]
fn the_far_edges_belong_to_the_last_cell() {
    let fine = a_patch_of(&a_plane(), CUT);
    let patch = fine.patch(0);

    assert_eq!(
        patch
            .cell_index(patch.strike_arc_km()[20], patch.dip_arc_km()[12])
            .expect("the far corner is on the fault"),
        Hypocentre {
            strike: 19,
            dip: 11
        }
    );
    assert_eq!(
        patch.cell_index(0.0, 0.0).expect("the near corner is on"),
        Hypocentre { strike: 0, dip: 0 }
    );
}

/// A position off either end is refused, naming the axis.
#[test]
fn a_position_off_the_patch_is_refused() {
    let fine = a_patch_of(&a_plane(), CUT);
    let patch = fine.patch(0);
    for (strike_km, dip_km, axis) in [
        (-0.1, 1.0, "strike"),
        (25.0, 1.0, "strike"),
        (1.0, -0.1, "dip"),
        (1.0, 500.0, "dip"),
    ] {
        match patch.cell_index(strike_km, dip_km) {
            Err(Error::PositionOffMesh { axis: named, .. }) => assert_eq!(named, axis),
            other => panic!("{strike_km}, {dip_km} gave {other:?}"),
        }
    }
}

/// A hand-built mesh whose cells are not all one size has no spacing.
///
/// Refinement cannot produce one, so this is built directly. The variant is not dead
/// though: `generate` reads a mesh *file*, which may have been hand-edited or written by
/// an importer, and the transform's requirement has to be checked where the data enters.
#[test]
fn a_mesh_with_uneven_cells_has_no_spacing() {
    // Three nodes along strike at 0, 1 and 4 km: one cell of 1 km and one of 3 km.
    let nodes = vec![
        vertex(0.0, 0.0, 0.0),
        vertex(1.0, 0.0, 0.0),
        vertex(4.0, 0.0, 0.0),
        vertex(0.0, 0.0, 1.0),
        vertex(1.0, 0.0, 1.0),
        vertex(4.0, 0.0, 1.0),
    ];
    let mesh = RefinedMesh::from_parts(
        at(0.0, 0.0),
        nodes,
        vec![ndarray::array![[0_usize, 1, 2], [3, 4, 5]]],
    )
    .expect("the indices are real vertices");

    assert!(matches!(
        mesh.patch(0).spacing(),
        Err(Error::NonUniformMesh { axis: "strike", .. })
    ));
}

fn vertex(east_km: f64, north_km: f64, depth_km: f64) -> Vertex {
    Vertex {
        east_km,
        north_km,
        depth_km,
    }
}

// ---------------------------------------------------------------------------
// Point sources
// ---------------------------------------------------------------------------

/// A point source is one cell of the size it asked for, centred where it asked.
#[test]
fn a_point_source_is_one_cell_where_it_says() {
    let centre = at(1_500.0, 5_180.0);
    let coarse = mesh::build(&Geometry::Point(PointSpec {
        centre,
        depth_km: 8.0,
        strike_deg: 55.0,
        dip_deg: 60.0,
        size_km: 0.5,
    }))
    .expect("valid");
    let fine = coarse
        .refine(&[Cuts {
            strike_count: 1,
            dip_count: 1,
        }])
        .expect("valid");
    let patch = fine.patch(0);

    assert_eq!(patch.cell_extents(), (1, 1));
    assert_relative_eq!(patch.areas_km2()[[0, 0]], 0.25, max_relative = EXACT);
    assert_relative_eq!(patch.strike_deg()[[0, 0]], 55.0, epsilon = 1e-9);
    assert_relative_eq!(patch.dip_deg()[[0, 0]], 60.0, epsilon = 1e-9);

    let centres = patch.centres();
    assert_relative_eq!(centres.depth_km[[0, 0]], 8.0, epsilon = EXACT);
    // A point source's origin is its centre, so the cell's centre offset is zero —
    // exactly, not nearly. That is the whole point of storing offsets.
    assert_eq!(
        fine.origin(),
        centre,
        "a point source is centred on its origin"
    );
    assert_relative_eq!(centres.east_km[[0, 0]], 0.0, epsilon = EXACT);
    assert_relative_eq!(centres.north_km[[0, 0]], 0.0, epsilon = EXACT);
}

/// A point too shallow to hold its own subfault is refused.
///
/// A 1 km cell dipping 60 degrees reaches 0.43 km above a centre at 0.2 km, which is in
/// the air. genslip's answer is to floor the top depth at zero, which silently shrinks the
/// subfault; saying so is the better one.
#[test]
fn a_point_source_above_the_ground_is_refused() {
    assert!(matches!(
        refusal(&Geometry::Point(PointSpec {
            centre: at(1_500.0, 5_180.0),
            depth_km: 0.2,
            strike_deg: 55.0,
            dip_deg: 60.0,
            size_km: 1.0,
        })),
        Error::AboveSurface { .. }
    ));
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

/// What the module will not build.
///
/// Notably short, because the shapes in `mesh.rs` removed three of these: a fault with no
/// planes, a trace with one point, and a plane count that disagrees with the trace are all
/// unrepresentable rather than refused.
#[test]
fn the_impossible_is_refused() {
    for dip_deg in [0.0, -10.0, 90.5, 120.0] {
        let mut fault = a_plane();
        fault.first.dip_deg = dip_deg;
        assert!(
            matches!(
                refusal(&Geometry::Fault(fault)),
                Error::DipOutOfRange { .. }
            ),
            "{dip_deg} degrees was accepted"
        );
    }

    let mut inverted = a_plane();
    inverted.first.bottom_depth_km = -4.0;
    assert!(matches!(
        refusal(&Geometry::Fault(inverted)),
        Error::NotPositive { .. }
    ));

    let mut flat = a_plane();
    flat.first.bottom_depth_km = flat.top_depth_km;
    assert!(matches!(
        refusal(&Geometry::Fault(flat)),
        Error::NotPositive { .. }
    ));

    let mut airborne = a_plane();
    airborne.top_depth_km = -1.0;
    assert!(matches!(
        refusal(&Geometry::Fault(airborne)),
        Error::AboveSurface { .. }
    ));

    let mut repeated = a_plane();
    repeated.first.end = repeated.origin;
    assert!(matches!(
        refusal(&Geometry::Fault(repeated)),
        Error::NotPositive { .. }
    ));
}

proptest! {
    /// Any plane, anywhere, at any orientation: the areas sum to length times width.
    #[test]
    fn the_areas_sum_to_length_times_width_always(
        bearing_deg in 0.0f64..360.0,
        length_km in 1.0f64..200.0,
        dip_deg in 5.0f64..90.0,
        top_depth_km in 0.0f64..5.0,
        depth_span_km in 1.0f64..40.0,
        strike_count in 1usize..25,
        dip_count in 1usize..15,
    ) {
        let fault = one_plane(
            bearing_deg, length_km, dip_deg, top_depth_km, top_depth_km + depth_span_km,
        );
        let fine = refined(&fault, &[Cuts { strike_count, dip_count }]);

        let width_km = depth_span_km / dip_deg.to_radians().sin();
        let total: f64 = fine.patch(0).areas_km2().flat().iter().sum();
        prop_assert!(
            (total - length_km * width_km).abs() < 1e-9 * length_km * width_km,
            "{total} against {}", length_km * width_km
        );
    }

    /// Any plane reports the strike and dip it was built with.
    #[test]
    fn a_plane_always_reports_its_own_orientation(
        bearing_deg in 0.0f64..360.0,
        dip_deg in 5.0f64..90.0,
    ) {
        let fine = refined(
            &one_plane(bearing_deg, 30.0, dip_deg, 0.0, 15.0),
            &[Cuts { strike_count: 8, dip_count: 5 }],
        );
        let patch = fine.patch(0);

        for strike in patch.strike_deg().flat() {
            let gap = (strike - bearing_deg).abs();
            prop_assert!(
                gap.min(360.0 - gap) < 1e-9,
                "strike {strike} against {bearing_deg}"
            );
        }
        for dip in patch.dip_deg().flat() {
            prop_assert!((dip - dip_deg).abs() < 1e-9, "dip {dip} against {dip_deg}");
        }
    }

    /// Refining more finely does not move the surface: the area is the same.
    ///
    /// The property that says refinement is a *subdivision* rather than a construction of
    /// its own. It is what licenses building the coarse mesh once and cutting it later.
    #[test]
    fn refinement_does_not_move_the_surface(
        strike_count in 1usize..30,
        dip_count in 1usize..20,
    ) {
        let fault = a_plane();
        let coarse = mesh::build(&Geometry::Fault(fault)).expect("valid");
        let one = coarse
            .refine(&[Cuts { strike_count: 1, dip_count: 1 }])
            .expect("valid");
        let many = coarse.refine(&[Cuts { strike_count, dip_count }]).expect("valid");

        let area = |fine: &RefinedMesh| -> f64 {
            fine.patch(0).areas_km2().flat().iter().sum()
        };
        let (whole, cut) = (area(&one), area(&many));
        prop_assert!(
            (whole - cut).abs() < 1e-9 * whole,
            "one cell gives {whole}, {strike_count}x{dip_count} gives {cut}"
        );
    }
}

// Deliberately not asserted:
//
// - Anything against `tests/harness/gsf.py:on_a_plane`. That is a flat-earth fixture
//   layout and disclaims being geodesy; comparing against it would either assert the
//   flat-earth answer is right or assert a number that is a property of the
//   approximation. `tests/test_mesh_cli.py` does compare against the corpus GSFs, which
//   is a different and honest claim.
// - Anything about longitude, latitude or the WGS84 ellipsoid. This module works in a
//   projected frame and never leaves it; the conversion, and the grid convergence
//   correction that goes with it, are `rupture_generator/mesh.py`'s and are tested there
//   against `pyproj`.
