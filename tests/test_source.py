"""Segment sources, genslip's correlation relations, and the sampling of a fault for
its separation from another."""

import dataclasses
import math

import numpy as np
import pytest

from rupture_generator import Geometry, RuptureGeneratorError
from rupture_generator.rupture.source import (
    CYCLES_OFFSET,
    GENSLIP_MAI_BEROZA,
    GENSLIP_SUZUKI_2022,
    MAI_BEROZA,
    SUZUKI_2022,
    CorrelationRelation,
    SegmentSource,
    _points_beneath,
    as_genslip,
    segment_source,
)

# ------------------------------------------------------------------ genslip lengths


@pytest.mark.parametrize(
    "relation",
    [
        MAI_BEROZA,
        SUZUKI_2022,
        CorrelationRelation(
            strike_exponent=0.4,
            strike_offset=-0.3,
            dip_exponent=0.2,
            dip_offset=0.7,
            strike_saturation=7.1,
            dip_saturation=6.0,
        ),
    ],
)
@pytest.mark.parametrize("magnitude", [5.5, 6.5, 8.0])
def test_genslip_lengths_are_two_pi_shorter(relation, magnitude):
    genslip = as_genslip(relation)
    assert genslip.strike_offset == pytest.approx(
        relation.strike_offset + CYCLES_OFFSET
    )
    assert genslip.dip_offset == pytest.approx(relation.dip_offset + CYCLES_OFFSET)
    # Every other field, the saturations included, is the relation's own.
    assert (
        dataclasses.replace(
            genslip,
            strike_offset=relation.strike_offset,
            dip_offset=relation.dip_offset,
        )
        == relation
    )
    dip_km, strike_km = genslip.lengths_km(magnitude)
    published_dip_km, published_strike_km = relation.lengths_km(magnitude)
    assert dip_km == pytest.approx(published_dip_km / (2.0 * math.pi))
    assert strike_km == pytest.approx(published_strike_km / (2.0 * math.pi))


def test_the_genslip_relations_are_the_published_ones_shortened():
    assert as_genslip(MAI_BEROZA) == GENSLIP_MAI_BEROZA
    assert as_genslip(SUZUKI_2022) == GENSLIP_SUZUKI_2022
    # Suzuki et al.'s down-dip length still saturates at Mw 6.3.
    assert GENSLIP_SUZUKI_2022.lengths_km(7.5)[0] == pytest.approx(
        10.0 ** (0.5 * 6.3 - 1.69) / (2.0 * math.pi)
    )


# ------------------------------------------------------------------ segment sources


def test_a_segment_source_round_trips_its_magnitude():
    source = segment_source(6.7, 90.0)
    assert source.moment_nm == pytest.approx(10.0 ** (1.5 * 6.7 + 9.05))
    assert source.rake_deg == 90.0
    assert source.magnitude == pytest.approx(6.7)


@pytest.mark.parametrize("moment_nm", [0.0, -1.0e18, math.nan])
def test_a_moment_that_is_not_positive_is_refused(moment_nm):
    with pytest.raises(RuptureGeneratorError, match="moment must be positive"):
        SegmentSource(moment_nm, 0.0)


# ------------------------------------------------------- sampling below a depth


def chart_with_a_middle_column() -> Geometry:
    """Build a chart of three columns of cells, two rows 4 km deep.

    Only the middle column is fault. The fault slants east and north with depth, so a point cut from an edge differs
    from its nodes on every axis, and it bends at 4 km, so a cut is only right
    if it's taken on the edge that crosses the depth.
    """
    depth = np.array([0.0, 4.0, 8.0])[:, None]
    column = np.arange(4.0)[None, :]
    east = 2.0 * column + np.array([0.0, 2.0, 3.0])[:, None]
    north = np.array([0.0, 1.0, 3.0])[:, None]
    nodes = np.stack(np.broadcast_arrays(east, north, depth), axis=-1).astype(float)
    occupied = np.array([[False, True, False], [False, True, False]])
    return Geometry(nodes, occupied, (3,))


def sorted_rows(points: np.ndarray) -> np.ndarray:
    return points[np.lexsort(points.T[::-1])]


def test_a_cut_off_between_rows_is_sampled_on_the_fault_edges_only():
    # 5 km is a quarter of the way from the 4 km row to the 8 km row. Only node columns
    # 1 and 2 bound a fault cell.
    points = _points_beneath(chart_with_a_middle_column(), 5.0)
    expected = np.array(
        [
            # Where the edges of columns 1 and 2 cross 5 km.
            [4.25, 1.5, 5.0],
            [6.25, 1.5, 5.0],
            # The fault's nodes at 8 km.
            [5.0, 3.0, 8.0],
            [7.0, 3.0, 8.0],
        ]
    )
    np.testing.assert_allclose(sorted_rows(points), sorted_rows(expected))


def test_a_cut_off_on_a_row_of_nodes_samples_each_node_once():
    # At 4 km the cut-off falls on a row of nodes. The sampler takes those nodes, and
    # no edge adds them a second time.
    points = _points_beneath(chart_with_a_middle_column(), 4.0)
    expected = np.array(
        [
            [4.0, 1.0, 4.0],
            [6.0, 1.0, 4.0],
            [5.0, 3.0, 8.0],
            [7.0, 3.0, 8.0],
        ]
    )
    np.testing.assert_allclose(sorted_rows(points), sorted_rows(expected))
