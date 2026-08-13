"""Geometric properties of the one mesh type.

Every assertion here is a mathematical or physical claim about a chart, quantified
over generated geometries, with the reason for its tolerance written beside it. None
of them pins a number the C produced.

The tolerance vocabulary: **1e-9 relative** is "exact" -- six orders above the
measured f64 round-off floor at fault scale (~3e-15 with offset coordinates) and
seven below the one-percent slip bound, so a failure at 1e-9 is a real error and never
arithmetic. Where a looser bound is used it is because the *geometry* is genuinely
looser, and the docstring says which geometry.
"""

from __future__ import annotations

import itertools

import numpy as np
import pyproj
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rupture_generator.config.geometry import (
    Discretisation,
    FaultConfig,
    LonLat,
    PlaneConfig,
    PointConfig,
)
from rupture_generator.formats.mesh import read_mesh, write_mesh
from rupture_generator.mesh import (
    PLANARITY_TOLERANCE_KM,
    RuptureMesh,
    build_fault,
    build_point,
    fuse,
    grid_convergence_deg,
    project_cells,
    seam_gap_km,
    validate_chart,
)
from tests.strategies import point_sources, straight_faults

EXACT = 1.0e-9
"""Relative tolerance for a quantity that is an identity in the projected frame."""

NZTM = pyproj.CRS("EPSG:2193")

SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
"""Forty examples: the geometry is deterministic and low-dimensional, so the
shrinking matters more than the count, and each example builds and validates a whole
mesh."""


def _plane(
    end: LonLat,
    *,
    dip_deg: float = 60.0,
    bottom_depth_km: float = 12.0,
    size_km: float = 1.0,
    dip_direction: str = "right",
) -> PlaneConfig:
    return PlaneConfig(
        end=end,
        dip_deg=dip_deg,
        bottom_depth_km=bottom_depth_km,
        discretisation=Discretisation(subfault_size_km=size_km),
        dip_direction=dip_direction,
    )


# ============================================================================
# Conservation and closure
# ============================================================================


@SETTINGS
@given(fault=straight_faults(planes=1))
def test_a_single_plane_area_is_its_length_times_its_width(fault: FaultConfig) -> None:
    """A straight plane's cell areas sum to its own length times its own width.

    The closure that says the subdivision is a subdivision: it neither creates nor
    loses surface. Exact rather than approximate because in the projected frame the
    plane is a parallelogram and the sum is an identity -- which is the whole
    argument for working in a projection rather than on the ellipsoid.
    """
    (chart,) = build_fault(fault, NZTM)
    length_km = chart.strike_arc_km()[-1]
    width_km = chart.dip_arc_km()[-1]
    assert chart.areas_km2().sum() == pytest.approx(length_km * width_km, rel=EXACT)


@SETTINGS
@given(fault=straight_faults(planes=1), cuts=st.integers(min_value=2, max_value=6))
def test_refining_further_does_not_move_the_surface(
    fault: FaultConfig, cuts: int
) -> None:
    """Cutting a plane finer changes the cell count, not the total area.

    The property that says refinement is a subdivision rather than a construction of
    its own -- it is what licenses building the coarse mesh once and cutting it later,
    and it is what would fail if the subdivision accumulated a step instead of
    interpolating.
    """
    coarse = build_fault(fault, NZTM)[0]

    finer = FaultConfig(
        name=fault.name,
        origin=fault.origin,
        top_depth_km=fault.top_depth_km,
        planes=[
            PlaneConfig(
                end=plane.end,
                dip_deg=plane.dip_deg,
                bottom_depth_km=plane.bottom_depth_km,
                dip_direction=plane.dip_direction,
                discretisation=Discretisation(
                    strike_count=coarse.cell_counts[1] * cuts,
                    dip_count=coarse.cell_counts[0] * cuts,
                ),
            )
            for plane in fault.planes
        ],
    )
    fine = build_fault(finer, NZTM)[0]
    assert fine.areas_km2().sum() == pytest.approx(coarse.areas_km2().sum(), rel=EXACT)


@SETTINGS
@given(fault=straight_faults(planes=3))
def test_fusing_conserves_area_and_counts_the_seam_once(fault: FaultConfig) -> None:
    """Fusing planes into a segment adds their areas and shares their seam columns.

    Two claims in one, because they are the same claim: the fused chart has one
    node column where two planes had two, and the surface is unchanged by saying so.
    A fused chart with an extra column would have extra area, and one that dropped a
    column would have less.
    """
    charts = build_fault(fault, NZTM)
    (segment,) = fuse(charts)

    assert segment.areas_km2().sum() == pytest.approx(
        sum(chart.areas_km2().sum() for chart in charts), rel=EXACT
    )
    assert segment.cell_counts[1] == sum(chart.cell_counts[1] for chart in charts)


@SETTINGS
@given(fault=straight_faults(planes=2))
def test_nodes_outnumber_cells_by_one_on_each_axis(fault: FaultConfig) -> None:
    """The off-by-one this type exists to make impossible.

    A grid of centres and a grid of corners are the same shape to anything that only
    counts elements, which is how a mesh comes out one cell short and looks fine.
    """
    for chart in [*build_fault(fault, NZTM), *fuse(build_fault(fault, NZTM))]:
        cells_i, cells_j = chart.cell_counts
        assert chart.nodes().shape == (cells_i + 1, cells_j + 1, 3)
        assert chart.centres().shape == (cells_i, cells_j, 3)
        assert chart.areas_km2().shape == (cells_i, cells_j)
        assert chart.strike_arc_km().shape == (cells_j + 1,)
        assert chart.dip_arc_km().shape == (cells_i + 1,)


# ============================================================================
# Strike, dip and the axis convention
# ============================================================================


@SETTINGS
@given(fault=straight_faults(planes=1))
def test_a_plane_reports_the_dip_it_was_built_with(fault: FaultConfig) -> None:
    """Every cell of a plane reports the plane's own dip.

    Exact because the dip is a property of the plane, not of the cut: a chart whose
    cells disagreed about their dip would be a chart whose nodes are not on one
    plane.
    """
    (chart,) = build_fault(fault, NZTM)
    _, dip_deg = chart.strike_dip_deg()
    assert dip_deg == pytest.approx(fault.planes[0].dip_deg, rel=EXACT)


@SETTINGS
@given(fault=straight_faults(planes=1))
def test_a_plane_reports_one_strike(fault: FaultConfig) -> None:
    """Every cell of a straight plane reports the same grid strike.

    The strike is derived from the cell normal rather than its edges, so this is the
    assertion that the derivation agrees with itself across a chart. Its *value*
    against the trace bearing is not asserted here: the bearing is a config input in
    longitude and latitude, and comparing them would need the projection, which is
    section 9's own test.
    """
    (chart,) = build_fault(fault, NZTM)
    strike_deg, _ = chart.strike_dip_deg()
    spread = strike_deg.max() - strike_deg.min()
    assert spread == pytest.approx(0.0, abs=1.0e-9)


def test_a_due_north_trace_dipping_right_dips_east() -> None:
    """The axis convention, which every distance assertion is blind to.

    A mesh laid out along the wrong axis has all the right separations: the areas
    sum, the steps are uniform, the dip is what was asked for. Only a statement
    about *which way* catches a transposed or mirrored surface, so this one is
    written by hand rather than generated.
    """
    origin = LonLat(longitude_deg=173.0, latitude_deg=-43.0)
    north = LonLat(longitude_deg=173.0, latitude_deg=-42.7)
    (chart,) = build_fault(
        FaultConfig(
            name="north",
            origin=origin,
            planes=[_plane(north, dip_deg=45.0)],
        ),
        NZTM,
    )
    nodes = chart.nodes()

    # Along strike (j): northing grows, easting does not.
    assert nodes[0, -1, 1] > nodes[0, 0, 1]
    assert nodes[0, -1, 0] == pytest.approx(nodes[0, 0, 0], abs=1.0e-6)
    # Down dip (i): easting grows -- the fault dips to the right of the walk, east.
    assert nodes[-1, 0, 0] > nodes[0, 0, 0]
    assert nodes[-1, 0, 2] > nodes[0, 0, 2]


def test_dipping_left_mirrors_the_surface_and_nothing_else() -> None:
    """Handedness changes the horizontal displacement and leaves the depths alone."""
    origin = LonLat(longitude_deg=173.0, latitude_deg=-43.0)
    end = LonLat(longitude_deg=173.3, latitude_deg=-42.8)
    right = build_fault(
        FaultConfig(name="r", origin=origin, planes=[_plane(end)]), NZTM
    )[0].nodes()
    left = build_fault(
        FaultConfig(
            name="l", origin=origin, planes=[_plane(end, dip_direction="left")]
        ),
        NZTM,
    )[0].nodes()

    assert np.allclose(right[..., 2], left[..., 2], atol=1.0e-12)
    # The down-dip displacement from the trace is negated; the trace itself is not.
    assert np.allclose(
        right[-1, :, :2] - right[0, :, :2],
        -(left[-1, :, :2] - left[0, :, :2]),
        atol=1.0e-9,
    )


@SETTINGS
@given(fault=straight_faults(planes=1))
def test_centres_are_the_mean_of_their_corners(fault: FaultConfig) -> None:
    """A cell centre is where its corners say it is, and inside the cell."""
    (chart,) = build_fault(fault, NZTM)
    nodes = chart.nodes()
    centres = chart.centres()

    corners = np.stack(
        [nodes[:-1, :-1], nodes[:-1, 1:], nodes[1:, 1:], nodes[1:, :-1]], axis=0
    )
    assert np.allclose(centres, corners.mean(axis=0), rtol=EXACT)
    assert (centres >= corners.min(axis=0) - 1.0e-9).all()
    assert (centres <= corners.max(axis=0) + 1.0e-9).all()


def test_a_vertical_fault_is_vertical() -> None:
    """A 90-degree dip drifts sideways by less than a nanometre over 12 km.

    ``tan(90 degrees)`` is enormous rather than infinite in f64, so the horizontal
    reach is a very small number rather than exactly zero. This pins how small: a
    picometre of drift is round-off, a metre is a formula that divides where it
    should not.
    """
    (chart,) = build_fault(
        FaultConfig(
            name="vertical",
            origin=LonLat(longitude_deg=173.0, latitude_deg=-43.0),
            planes=[
                _plane(
                    LonLat(longitude_deg=173.3, latitude_deg=-42.8),
                    dip_deg=90.0,
                )
            ],
        ),
        NZTM,
    )
    nodes = chart.nodes()
    drift_km = np.linalg.norm(nodes[-1, :, :2] - nodes[0, :, :2], axis=-1).max()
    assert drift_km < 1.0e-12

    _, dip_deg = chart.strike_dip_deg()
    assert dip_deg == pytest.approx(90.0, rel=EXACT)


# ============================================================================
# The hypocentre seam -- DEFECTS.md 17
# ============================================================================


@SETTINGS
@given(fault=straight_faults(planes=2))
def test_a_position_in_a_cell_comes_back_as_that_cell(fault: FaultConfig) -> None:
    """Every cell's midpoint locates to that cell. Every cell, not a sample.

    `DEFECTS.md` 17: the hypocentre was one cell off *in both directions*, and it
    correlated 0.99+ with the right answer while moving onsets by up to a second --
    every diagnostic that asked "is this the right shape" said yes. The defect was a
    constant offset, so a test that checked one cell would have caught it and nobody
    checked even one. This checks all of them.
    """
    (segment,) = fuse(build_fault(fault, NZTM))
    cells_i, cells_j = segment.cell_counts
    strike_arc = segment.strike_arc_km()
    dip_arc = segment.dip_arc_km()

    for i in range(cells_i):
        for j in range(cells_j):
            found = segment.cell_index(
                0.5 * (strike_arc[j] + strike_arc[j + 1]),
                0.5 * (dip_arc[i] + dip_arc[i + 1]),
            )
            assert found == (i, j)


@SETTINGS
@given(fault=straight_faults(planes=1))
def test_the_far_edge_belongs_to_the_last_cell(fault: FaultConfig) -> None:
    """A hypocentre at the very bottom of the fault is on the fault.

    "At the bottom of the fault" is a thing people write, and the alternative to the
    last cell owning its far edge is refusing a position that is on the surface.
    """
    (chart,) = build_fault(fault, NZTM)
    cells_i, cells_j = chart.cell_counts
    assert chart.cell_index(chart.strike_arc_km()[-1], chart.dip_arc_km()[-1]) == (
        cells_i - 1,
        cells_j - 1,
    )


@SETTINGS
@given(fault=straight_faults(planes=1))
def test_an_interior_boundary_belongs_to_the_cell_below_it(
    fault: FaultConfig,
) -> None:
    """A position exactly on a cell boundary belongs to the upper cell.

    A tie has to go somewhere and the choice has to be *stated*, because a config
    that names a round number lands on one exactly.
    """
    (chart,) = build_fault(fault, NZTM)
    strike_arc = chart.strike_arc_km()
    dip_arc = chart.dip_arc_km()
    for j in range(1, len(strike_arc) - 1):
        assert chart.cell_index(float(strike_arc[j]), 0.0)[1] == j
    for i in range(1, len(dip_arc) - 1):
        assert chart.cell_index(0.0, float(dip_arc[i]))[0] == i


@SETTINGS
@given(fault=straight_faults(planes=1))
def test_a_position_off_the_fault_is_refused_naming_the_axis(
    fault: FaultConfig,
) -> None:
    """Off the end is an error that says which end and how long the fault is."""
    (chart,) = build_fault(fault, NZTM)
    extent_km = float(chart.strike_arc_km()[-1])

    with pytest.raises(ValueError, match="strike_km"):
        chart.cell_index(extent_km + 1.0, 0.0)
    with pytest.raises(ValueError, match="dip_km"):
        chart.cell_index(0.0, float(chart.dip_arc_km()[-1]) + 1.0)


# ============================================================================
# Fusion and seams
# ============================================================================


@SETTINGS
@given(fault=straight_faults(planes=3))
def test_planes_that_hang_the_same_way_share_their_seam_exactly(
    fault: FaultConfig,
) -> None:
    """A conforming bend's shared column agrees to round-off, not to a tolerance.

    This is what the ``1/cos(deflection/2)`` stretch buys: the bisector's projection
    onto either plane's down-dip direction is ``cos(deflection/2)``, and dividing it
    out puts the placed node in *both* planes at once. Without the stretch the two
    planes diverge below the vertex by a measured 1.285 km on the shipped ``hope``
    example -- so this assertion is six orders tighter than the seam tolerance it
    licenses.
    """
    charts = build_fault(fault, NZTM)
    for near, far in itertools.pairwise(charts):
        assert seam_gap_km(near, far) < 1.0e-12

    assert len(fuse(charts)) == 1


def test_planes_that_hang_differently_are_two_segments() -> None:
    """A dip change is a segment boundary, and segments are returned, not refused.

    The old pipeline refused this ("multi-segment is not written"); this one hands
    back both charts, because whether a rupture propagates across the gap is the
    propagation stage's question and not the geometry's. The kaikoura example is the
    measured case: 70 and 55 degrees separate by kilometres at the deepest row.
    """
    origin = LonLat(longitude_deg=173.0, latitude_deg=-42.6)
    fault = FaultConfig(
        name="kaikoura",
        origin=origin,
        planes=[
            _plane(LonLat(longitude_deg=173.4, latitude_deg=-42.4), dip_deg=70.0),
            _plane(
                LonLat(longitude_deg=173.9, latitude_deg=-42.1),
                dip_deg=55.0,
                bottom_depth_km=12.0,
            ),
        ],
    )
    charts = build_fault(fault, NZTM)
    assert seam_gap_km(charts[0], charts[1]) > 1.0

    segments = fuse(charts)
    assert len(segments) == 2
    assert [segment.cell_counts for segment in segments] == [
        chart.cell_counts for chart in charts
    ]


def test_one_conforming_surface_cut_two_ways_is_refused() -> None:
    """Planes that meet but are cut into different dip rows are a config mistake.

    Distinguished from a segment boundary on purpose: the geometry conforms -- the
    corners coincide -- so this is one surface asked for at two resolutions, and
    silently generating it as two ruptures would be worse than saying so.
    """
    origin = LonLat(longitude_deg=173.0, latitude_deg=-43.0)
    fault = FaultConfig(
        name="mixed",
        origin=origin,
        planes=[
            PlaneConfig(
                end=LonLat(longitude_deg=173.3, latitude_deg=-42.8),
                dip_deg=60.0,
                bottom_depth_km=12.0,
                discretisation=Discretisation(strike_count=10, dip_count=8),
            ),
            PlaneConfig(
                end=LonLat(longitude_deg=173.6, latitude_deg=-42.6),
                dip_deg=60.0,
                bottom_depth_km=12.0,
                discretisation=Discretisation(strike_count=10, dip_count=5),
            ),
        ],
    )
    with pytest.raises(ValueError, match="rows down dip"):
        fuse(build_fault(fault, NZTM))


def test_a_trace_that_doubles_back_is_refused() -> None:
    """Past 120 degrees the bend stretch exceeds 2, and the grid stops being uniform."""
    origin = LonLat(longitude_deg=173.0, latitude_deg=-43.0)
    fault = FaultConfig(
        name="hairpin",
        origin=origin,
        planes=[
            _plane(LonLat(longitude_deg=173.3, latitude_deg=-43.0)),
            _plane(LonLat(longitude_deg=173.05, latitude_deg=-43.05)),
        ],
    )
    with pytest.raises(ValueError, match="doubles back"):
        build_fault(fault, NZTM)


@SETTINGS
@given(fault=straight_faults(planes=1))
def test_fusing_one_chart_is_the_identity(fault: FaultConfig) -> None:
    """A single plane fuses to itself, node for node."""
    (chart,) = build_fault(fault, NZTM)
    (fused,) = fuse([chart])
    assert np.array_equal(fused.nodes(), chart.nodes())


# ============================================================================
# S3 -- chart validation
# ============================================================================


@SETTINGS
@given(fault=straight_faults(planes=3))
def test_a_built_chart_passes_validation(fault: FaultConfig) -> None:
    """Everything this module builds satisfies the sampler's assumptions.

    Including a bend: the fused chart is *piecewise* planar with a kink at the seam,
    which S3 accepts by validating per constant-plane block, because the sampler and
    the eikonal solver only ever see the index space and one spacing.
    """
    for segment in fuse(build_fault(fault, NZTM)):
        validate_chart(segment)


@SETTINGS
@given(fault=straight_faults(planes=1))
def test_every_line_of_a_chart_is_evenly_divided(fault: FaultConfig) -> None:
    """Each row's steps agree with one another, and each column's do.

    Measured on every line rather than on one edge, because a chart is not
    necessarily a parallelogram -- a fused bend is a trapezoid -- so an edge that is
    uniform says nothing about the interior.
    """
    (chart,) = build_fault(fault, NZTM)
    strike_steps, dip_steps = chart.line_steps()

    within_row = (strike_steps.max(axis=1) - strike_steps.min(axis=1)).max()
    within_column = (dip_steps.max(axis=0) - dip_steps.min(axis=0)).max()
    assert within_row < EXACT * strike_steps.mean()
    assert within_column < EXACT * dip_steps.mean()


def test_a_warped_chart_is_refused_as_not_planar() -> None:
    """A node lifted out of the plane is caught, naming the plane.

    The seam where curvature enters: this refusal is the whole of what stops a
    curved chart reaching a sampler that assumes flatness.
    """
    east, north = np.meshgrid(np.arange(6.0), np.arange(5.0))
    depth = np.zeros_like(east)
    flat = RuptureMesh.from_nodes(
        east,
        north,
        depth,
        origin_east_km=1500.0,
        origin_north_km=5180.0,
        surface="flat",
    )
    validate_chart(flat)

    warped = depth.copy()
    warped[2, 3] += 10.0 * PLANARITY_TOLERANCE_KM
    with pytest.raises(ValueError, match="deviates"):
        validate_chart(
            RuptureMesh.from_nodes(
                east,
                north,
                warped,
                origin_east_km=1500.0,
                origin_north_km=5180.0,
                surface="warped",
            )
        )


def test_a_shallow_fault_that_turns_sharply_has_no_single_spacing() -> None:
    """A bend skews by the dip's horizontal reach, not just by its own stretch.

    Measured: a 4 km-deep fault dipping 5 degrees reaches 46 km horizontally, so
    rotating its down-dip direction by half a 40-degree bend swings the bottom edge
    further than the 27 km plane is long -- the top edge is 27.8 km where the bottom
    is 11.2 km. The chart is still planar and every line is still evenly divided;
    what it does not have is *one spacing*, and a sampler handed its mean would be
    sampling a grid the fault does not have.

    This is why the spread check is on the measured geometry rather than on the
    deflection: the same bend on a steep fault is fine, and `hope` (75 degrees, a
    20-degree bend) spreads by 2.4%.
    """
    fault = FaultConfig(
        name="shallow",
        origin=LonLat(longitude_deg=170.0, latitude_deg=-38.0),
        planes=[
            _plane(end, dip_deg=5.0, bottom_depth_km=4.0)
            for end in (
                LonLat(longitude_deg=170.0, latitude_deg=-37.75),
                LonLat(longitude_deg=170.2260576372834, latitude_deg=-37.56421379363),
                LonLat(longitude_deg=170.4521152745668, latitude_deg=-37.37842758726),
            )
        ],
    )
    (segment,) = fuse(build_fault(fault, NZTM))
    with pytest.raises(ValueError, match="varies by"):
        validate_chart(segment)


def test_validation_says_nothing_about_padding_or_even_extents() -> None:
    """An odd-sized chart is fine. Padding is the sampler's private business.

    `PLAN.md` section 5 deletes "padded-extent assertions outside SpectralSampler";
    this is the assertion that they stayed deleted.
    """
    east, north = np.meshgrid(np.arange(8.0), np.arange(6.0))
    odd = RuptureMesh.from_nodes(
        east,
        north,
        np.zeros_like(east),
        origin_east_km=0.0,
        origin_north_km=0.0,
        surface="odd",
    )
    assert odd.cell_counts == (5, 7)
    validate_chart(odd)


def test_a_chart_needs_two_nodes_on_each_axis() -> None:
    """One node on an axis is no cells, which is not a surface."""
    with pytest.raises(ValueError, match="at least 2 nodes"):
        RuptureMesh.from_nodes(
            np.zeros((1, 4)),
            np.zeros((1, 4)),
            np.zeros((1, 4)),
            origin_east_km=0.0,
            origin_north_km=0.0,
            surface="sliver",
        )


def test_a_non_finite_node_is_refused_at_construction() -> None:
    """A NaN would travel silently into every derived quantity and into an SRF."""
    east, north = np.meshgrid(np.arange(4.0), np.arange(3.0))
    depth = np.zeros_like(east)
    depth[1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        RuptureMesh.from_nodes(
            east,
            north,
            depth,
            origin_east_km=0.0,
            origin_north_km=0.0,
            surface="nan",
        )


# ============================================================================
# Point sources
# ============================================================================


@SETTINGS
@given(point=point_sources())
def test_a_point_source_is_one_cell_of_the_size_it_asked_for(
    point: PointConfig,
) -> None:
    """A point source is an ordinary one-cell chart, centred where it was told.

    Not a special type: `PLAN.md` section 5 makes a point source the pipeline with
    constant fields, so everything downstream must be able to treat it as a chart.
    """
    (chart,) = build_point(point, NZTM)

    assert chart.cell_counts == (1, 1)
    assert float(chart.areas_km2().sum()) == pytest.approx(point.size_km**2, rel=1.0e-6)

    _, dip_deg = chart.strike_dip_deg()
    assert float(dip_deg[0, 0]) == pytest.approx(point.dip_deg, rel=EXACT)

    centre = chart.centres()[0, 0]
    assert float(centre[2]) == pytest.approx(point.depth_km, abs=1.0e-9)
    assert float(np.hypot(centre[0], centre[1])) < 1.0e-9


def test_a_point_source_whose_top_edge_is_in_the_air_is_refused() -> None:
    """genslip floors the top depth at zero, silently shrinking the subfault.

    Saying so is the better answer: a 1 km cell dipping 60 degrees reaches 0.43 km
    above a centre at 0.2 km, which is in the air.
    """

    with pytest.raises(ValueError, match="above the surface"):
        build_point(
            PointConfig(
                name="shallow",
                centre=LonLat(longitude_deg=173.0, latitude_deg=-43.0),
                depth_km=0.2,
                strike_deg=45.0,
                dip_deg=60.0,
                size_km=1.0,
            ),
            NZTM,
        )


# ============================================================================
# The fields a stage attaches
# ============================================================================


def _chart() -> RuptureMesh:
    """One small planar chart, not square, for the field API's own assertions.

    Not square on purpose: a transposed field is only detectable on a chart whose two
    cell counts differ, and that is the check `with_fields` exists for.
    """
    (chart,) = build_fault(
        FaultConfig(
            name="hope",
            origin=LonLat(longitude_deg=172.0, latitude_deg=-42.0),
            planes=[_plane(LonLat(longitude_deg=172.3, latitude_deg=-42.0), size_km=2.0)],
        ),
        NZTM,
    )
    return chart


def test_a_chart_starts_with_no_fields() -> None:
    """Geometry is not a field. What `fields` reports is what a stage put there."""
    chart = _chart()

    assert chart.fields() == frozenset()
    assert "depth_km" not in chart
    assert "slip_m" not in chart


def test_attaching_a_field_leaves_the_chart_it_came_from_alone() -> None:
    """Functional, never in place -- what lets stages share geometry rather than copy it."""
    chart = _chart()
    slip = np.ones(chart.cell_counts)

    annotated = chart.with_fields(slip_m=slip)

    assert annotated.fields() == {"slip_m"}
    assert chart.fields() == frozenset()
    assert np.array_equal(annotated["slip_m"], slip)
    # The geometry is the same geometry, not a copy that could drift from it.
    assert np.array_equal(annotated.areas_km2(), chart.areas_km2())


def test_a_field_is_handed_back_read_only() -> None:
    """An array a caller could write through is a way around an immutable chart.

    Not hypothetical: every stage takes arrays and returns arrays, so one that
    modified its input in place would edit a chart some other stage is still reading.
    """
    chart = _chart().with_fields(slip_m=np.ones(_chart().cell_counts))

    with pytest.raises(ValueError, match="read-only"):
        chart["slip_m"][0, 0] = 5.0


def test_a_field_the_chart_does_not_carry_says_what_it_does() -> None:
    """The list of what *is* attached is most of the diagnosis.

    A stage asking for a field nobody drew is a pipeline written in the wrong order,
    and that is the message that says so.
    """
    chart = _chart().with_fields(slip_m=np.ones(_chart().cell_counts))

    with pytest.raises(KeyError, match="rise_time_s.*carries slip_m"):
        chart["rise_time_s"]


def test_a_transposed_field_is_refused() -> None:
    """The check that earns the method.

    xarray objects only when dimension *sizes* disagree, so on a square patch a
    transposed field is assigned without complaint and every quantity derived from it
    is quietly wrong -- an (i, j) / (j, i) mix-up that no downstream assertion sees.
    """
    chart = _chart()
    cells_i, cells_j = chart.cell_counts
    assert cells_i != cells_j, "this test needs a chart that is not square"

    with pytest.raises(ValueError, match="shaped"):
        chart.with_fields(slip_m=np.ones((cells_j, cells_i)))


def test_a_non_finite_field_is_refused() -> None:
    """`from_nodes`' argument, one stage later: a NaN here reaches the SRF silently."""
    chart = _chart()
    slip = np.ones(chart.cell_counts)
    slip[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        chart.with_fields(slip_m=slip)


@pytest.mark.parametrize("name", ["depth_km", "east_km", "plane", "slip_rate"])
def test_a_field_may_not_be_given_the_geometrys_own_name(name: str) -> None:
    """A field called ``depth_km`` would sit beside the geometry under its name.

    It would not even collide -- the nodes are on ``(i_node, j_node)`` and a field is
    on ``(i, j)`` -- so nothing would raise, and the next reader of ``depth_km`` would
    get whichever xarray handed back.
    """
    chart = _chart()

    with pytest.raises(ValueError, match="the chart's own"):
        chart.with_fields(**{name: np.ones(chart.cell_counts)})


def test_a_recomputed_field_replaces_its_earlier_value() -> None:
    """The moment fold sizes `slip_pattern` into `slip_m`: one quantity, twice stated."""
    chart = _chart()
    ones = np.ones(chart.cell_counts)

    twice = chart.with_fields(slip_m=ones).with_fields(slip_m=2.0 * ones)

    assert np.array_equal(twice["slip_m"], 2.0 * ones)
    assert twice.fields() == {"slip_m"}


def test_dropping_a_field_that_is_not_there_is_not_an_error() -> None:
    """Dropping states something about the result, not a claim about the history."""
    chart = _chart().with_fields(slip_m=np.ones(_chart().cell_counts))

    assert chart.without("slip_pattern").fields() == {"slip_m"}
    assert chart.without("slip_m").fields() == frozenset()


def test_attrs_are_read_only_and_the_chart_keeps_its_own() -> None:
    """A mutable attrs view is a mutable chart."""
    chart = _chart().with_attrs(truncated_fraction=0.09)

    assert chart.attrs["truncated_fraction"] == 0.09
    assert chart.attrs["surface"] == "hope"
    with pytest.raises(TypeError):
        chart.attrs["truncated_fraction"] = 0.5  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("name", ["surface", "origin_east_km", "origin_north_km"])
def test_an_attr_that_says_what_the_chart_is_may_not_be_rewritten(name: str) -> None:
    """Rewriting one would move the fault, and every derived quantity with it."""
    with pytest.raises(ValueError, match="says what this chart is"):
        _chart().with_attrs(**{name: "elsewhere"})


def test_the_node_dataset_carries_the_geometry_and_nothing_else() -> None:
    """The one way out, and what it hands out is the surface, not the rupture."""
    chart = _chart().with_fields(slip_m=np.ones(_chart().cell_counts))

    nodes = chart.node_dataset()

    assert set(nodes.data_vars) == {"east_km", "north_km", "depth_km"}
    assert "slip_m" not in nodes
    assert "plane" not in nodes.coords
    # The units are written down once, in `from_nodes`, and reach a file through here.
    assert nodes["depth_km"].attrs["units"] == "kilometres"


def test_pulses_have_to_be_one_per_subfault() -> None:
    """A CSR indptr that is not one makes some subfault's pulse another's.

    Which is a plausible-looking rupture: every subfault still has slip, still has a
    rise time, and radiates something. Nothing downstream could question it.
    """
    chart = _chart()
    cells = chart.cell_counts[0] * chart.cell_counts[1]
    samples = np.ones(3 * cells)

    good = np.arange(cells + 1) * 3
    assert chart.with_pulses(good, samples).pulses is not None

    with pytest.raises(ValueError, match="wants"):
        chart.with_pulses(good[:-1], samples)
    with pytest.raises(ValueError, match="decrease"):
        chart.with_pulses(good[::-1], samples)
    with pytest.raises(ValueError, match="samples"):
        chart.with_pulses(good, samples[:-1])


def test_a_chart_prints_as_its_shape_rather_than_its_arrays() -> None:
    """One failed assertion should not be a screenful of dataset."""
    chart = _chart().with_fields(slip_m=np.ones(_chart().cell_counts))
    cells_i, cells_j = chart.cell_counts

    # A count rather than a shape: the triangular track replaces `cell_counts` with a
    # single `int`, so a chart that prints `7x12` prints something a triangulation of
    # the same fault cannot.
    assert repr(chart) == (
        f"RuptureMesh('hope', {cells_i * cells_j} cells, fields: slip_m)"
    )


# ============================================================================
# Precision, projection and the file
# ============================================================================


@SETTINGS
@given(fault=straight_faults(planes=1))
def test_the_geometry_is_the_same_at_crs_scale_as_at_the_origin(
    fault: FaultConfig,
) -> None:
    """Derived quantities do not degrade when the surface is 5,000 km from zero.

    The property that would have caught the absolute-coordinate regression: an NZTM
    northing reaches 5,180 km against a ~1 km subfault, so an absolute vertex is
    rounded at CRS scale -- measured at 1.2e-12 relative against 3e-15 for offsets.
    Charts hold offsets, so moving the origin must change nothing at all.
    """
    (chart,) = build_fault(fault, NZTM)
    nodes = chart.nodes()
    moved = RuptureMesh.from_nodes(
        nodes[..., 0],
        nodes[..., 1],
        nodes[..., 2],
        origin_east_km=1500.0,
        origin_north_km=5180.0,
        surface=chart.surface,
    )
    assert np.array_equal(moved.areas_km2(), chart.areas_km2())
    assert np.array_equal(moved.strike_arc_km(), chart.strike_arc_km())
    assert moved.spacing_km() == chart.spacing_km()


@SETTINGS
@given(fault=straight_faults(planes=1))
def test_true_strike_is_grid_strike_plus_the_convergence(fault: FaultConfig) -> None:
    """The projection seam adds the convergence angle and nothing else.

    Grid north is not true north: away from the projection's central meridian the
    two diverge by up to 5.04 degrees in NZTM, which is five times the one-degree
    rake bound. Dip crosses unchanged, because it is an angle within the plane.
    """
    (chart,) = build_fault(fault, NZTM)
    grid_strike_deg, grid_dip_deg = chart.strike_dip_deg()
    located = project_cells(chart, NZTM)

    convergence = grid_convergence_deg(
        NZTM,
        located["centre_longitude_deg"].to_numpy(),
        located["centre_latitude_deg"].to_numpy(),
    )
    expected = np.mod(grid_strike_deg + convergence, 360.0)
    assert np.allclose(located["strike_deg"].to_numpy(), expected, atol=1.0e-12)
    assert np.allclose(located["dip_deg"].to_numpy(), grid_dip_deg, atol=1.0e-12)
    assert np.allclose(located["area_km2"].to_numpy(), chart.areas_km2(), atol=0.0)


@pytest.mark.parametrize("suffix", [".h5", ".zarr"])
@SETTINGS
@given(fault=straight_faults(planes=3))
def test_a_mesh_file_round_trips_losslessly(
    fault: FaultConfig, suffix: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """What comes back is bit-identical, so every derived quantity is too.

    One round trip per format, and no assertions about xarray or Zarr internals --
    except the one that is this package's own trap: **Zarr does not preserve group
    order**, so a reader that trusted iteration order works in HDF5 and silently
    permutes the fault in Zarr. Three planes, so a permutation would show.
    """
    charts = build_fault(fault, NZTM)
    path = tmp_path_factory.mktemp("mesh") / f"mesh{suffix}"
    write_mesh({fault.name: charts}, NZTM, path)

    back, crs = read_mesh(path)
    assert crs == NZTM
    assert list(back) == [fault.name]
    for original, restored in zip(back[fault.name], charts, strict=True):
        assert np.array_equal(original.nodes(), restored.nodes())
