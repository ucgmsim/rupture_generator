"""What the viewer's panels put on their axes.

Drawing is not an assertion, so most of this module is untested on purpose. The three
axis helpers are not drawings, and each replaced something that was wrong in a way a
reader could not see: a histogram indexed by bin number reads as a distribution over
the quantity, a moment axis of ``6.000000e18`` reads as precision, and a rose with no
circle round it reads as whatever the reader assumed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rupture_generator.scripts import view


def test_a_cell_is_fanned_into_triangles_whatever_its_arity() -> None:
    """Three corners or four, and a quad still splits the way it always did.

    The mesh writer reads the arity off ``corners.shape`` rather than assuming four,
    and the risk in that is a fan that silently disagrees with the two-triangle split
    the viewer has always drawn.
    """
    rerun = pytest.importorskip("rerun")
    quad = np.arange(4 * 3, dtype=np.float64).reshape(1, 4, 3)
    triangle = np.arange(3 * 3, dtype=np.float64).reshape(1, 3, 3)
    colours = np.array([[255, 0, 0]], dtype=np.uint8)

    quads = view._mesh(rerun, quad, colours).triangle_indices.as_arrow_array()
    triangles = view._mesh(rerun, triangle, colours).triangle_indices.as_arrow_array()

    assert quads.to_pylist() == [[0, 1, 2], [0, 2, 3]]
    assert triangles.to_pylist() == [[0, 1, 2]]


class _Recorder:
    """Catches what the viewer logs, so a panel can be asserted without a window."""

    def __init__(self, rerun) -> None:  # noqa: ANN001
        self.rerun = rerun
        self.logged: dict[str, object] = {}

    def __getattr__(self, name: str) -> object:
        return getattr(self.rerun, name)

    def log(self, path: str, entity: object, *, static: bool = False) -> None:
        del static
        self.logged[path] = entity


def _draw_histogram(values: np.ndarray | None = None) -> _Recorder:
    """One histogram of known values, drawn into a recorder."""
    rerun = pytest.importorskip("rerun")
    if values is None:
        values = np.array([0.5, 1.5, 1.6, 2.5])
    recorder = _Recorder(rerun)
    view._histogram(
        recorder,
        "/histogram/slip",
        values,
        (0.0, 4.0),
        4,
        lambda values: view.hot(values, 0.0, 4.0),
        "slip (m)",
    )
    return recorder


def test_a_histogram_stands_on_the_quantitys_own_axis() -> None:
    """The labels carry the quantity's own values, not bin numbers.

    Counts against bin index read as a distribution over the quantity while meaning
    something that changes when the bin count does. Four bins over 0 to 4 metres put
    a tick at every metre, and the axis says so.
    """
    labels = (
        _draw_histogram()
        .logged["/histogram/slip/axis/labels"]
        .labels.as_arrow_array()
        .to_pylist()
    )

    assert "slip (m)" in labels
    for metres in ("1", "2", "3", "4"):
        assert metres in labels


def test_a_histograms_bars_carry_the_faults_own_colours() -> None:
    """The same map at the same limits, so the two panels are one instrument.

    A band of colour on the fault is then findable in the distribution by its colour,
    rather than by reading a number off one panel and hunting for it in the other.

    Asserted against `hot` rather than merely for being non-empty, and this is the
    test that would have caught the version before it: Rerun's `BarChart` takes *one*
    colour for the whole chart and silently draws only the first of an array, so the
    bars came out uniformly `hot(0)` -- black on black, vanished rather than saying
    they could not be coloured.
    """
    boxes = _draw_histogram().logged["/histogram/slip"]
    centres = np.array([0.5, 1.5, 2.5, 3.5])

    packed = np.array(boxes.colors.as_arrow_array().to_pylist(), dtype=np.uint32)
    expected = np.array(
        [
            (int(r) << 24) | (int(g) << 16) | (int(b) << 8) | 255
            for r, g, b in view.hot(centres, 0.0, 4.0)
        ],
        dtype=np.uint32,
    )
    assert np.array_equal(packed, expected)
    assert len(packed) == 4


def test_subfaults_at_rest_are_counted_rather_than_binned() -> None:
    """A fault that has not slipped is not part of the distribution of slip on it.

    Tens of thousands of untouched cells pile into the first bin and make a spike
    several times the height of the distribution, flattening everything worth looking
    at. Dropping them silently would misstate the sample size, so the count is on the
    panel -- and the first bin holds only the cells that really are in it.
    """
    values = np.concatenate([np.zeros(7), np.array([0.5, 1.5, 1.6, 2.5])])

    recorder = _draw_histogram(values)

    labels = recorder.logged["/histogram/slip/axis/labels"].labels
    assert "7 subfaults at rest, not shown" in labels.as_arrow_array().to_pylist()
    # Four bars, and the one covering zero holds the single 0.5 rather than eight.
    boxes = recorder.logged["/histogram/slip"]
    assert len(boxes.colors.as_arrow_array()) == 4


def test_counts_are_drawn_on_a_log_height() -> None:
    """The tail is a handful of cells against tens of thousands in the mode.

    On a linear axis every bin outside the mode is a line one pixel high -- present,
    unreadable, and easy to read as empty. A bin holding a single cell must still
    stand above the baseline, which is why the transform is ``log10(count + 1)``
    rather than ``log10(count)``.
    """
    values = np.concatenate([np.full(1000, 0.5), np.array([1.5])])

    boxes = _draw_histogram(values).logged["/histogram/slip"]

    heights = (
        -2.0
        * np.array(boxes.centers.as_arrow_array().to_pylist(), dtype=np.float64)[:, 1]
    )
    tall, lone = heights[0], heights[1]
    assert tall == pytest.approx(1.0)
    # Linear would put one against a thousand at 0.001; log puts it near a third.
    assert lone > 0.05
    assert lone == pytest.approx(math.log10(2.0) / math.log10(1001.0))


def test_a_histograms_bars_stand_on_the_baseline() -> None:
    """Bars grow from zero upwards, which in screen coordinates is downwards.

    A bar drawn from the wrong edge, or centred on the axis, reads as a distribution
    with the wrong shape rather than as a drawing mistake.
    """
    boxes = _draw_histogram().logged["/histogram/slip"]

    centers = np.array(boxes.centers.as_arrow_array().to_pylist(), dtype=np.float64)
    half = np.array(boxes.half_sizes.as_arrow_array().to_pylist(), dtype=np.float64)

    # Every bar's lower edge sits exactly on y = 0, and its body is above it.
    assert np.allclose(centers[:, 1] + half[:, 1], 0.0)
    assert (centers[:, 1] <= 0.0).all()
    # The tallest bar fills the box.
    assert np.isclose((-2.0 * centers[:, 1]).max(), 1.0)


def test_axis_labels_are_drawn_in_a_colour_that_can_be_seen() -> None:
    """Rerun draws a label in its entity's colour, so a transparent point is silent.

    The first attempt used a fully transparent point to hide the marker dot, and got
    eight label backgrounds with no glyphs in them -- the plot looked like it had no
    axis rather than like it had a bug.
    """
    labels = _draw_histogram().logged["/histogram/slip/axis/labels"]

    colours = labels.colors.as_arrow_array().to_pylist()
    assert colours, "the labels carry no colour, so they draw as invisible text"
    assert all(colour & 0xFF for colour in colours), "a label is transparent"


@pytest.mark.parametrize(
    ("peak", "scale", "plotted"),
    [
        (6.2e18, 1.0e18, 6.2),
        (5.0e19, 1.0e18, 50.0),
        (3.4e21, 1.0e21, 3.4),
    ],
)
def test_a_moment_axis_is_scaled_to_a_power_of_a_thousand(
    peak: float, scale: float, plotted: float
) -> None:
    """Rerun 0.35 has no tick formatter, so the number that goes in is the only lever.

    A power of a thousand rather than of ten, so the exponent is one an SI reader has
    a word for and so it stays put while the curve grows through an order of
    magnitude -- which is why 5e19 plots as 50 against ``1e18`` rather than as 5
    against ``1e19``.
    """
    assert view._engineering_scale(np.array([peak])) == pytest.approx(scale)
    assert peak / view._engineering_scale(np.array([peak])) == pytest.approx(plotted)


def test_a_moment_axis_survives_a_rupture_that_released_nothing() -> None:
    """Dividing by the scale must not divide by zero, whatever the pulses did."""
    assert view._engineering_scale(np.zeros(4)) == 1.0
    assert view._engineering_scale(np.array([])) == 1.0


def test_the_rake_rose_has_a_circle_and_degrees_round_it() -> None:
    """A rose without a reference is a plot with no axis at all.

    Nothing on screen says which way is zero or how far round a wedge sits, which is
    the angular form of the same complaint as a histogram indexed by bin number. The
    outer circle is at radius one because that is where `rose` puts its longest wedge.
    """
    guides, positions, labels = view.rose_axis()

    assert view.AXIS_TEXT[:3] != (0, 0, 0), "the rose labels would be invisible"
    assert labels == ["-180°", "-135°", "-90°", "-45°", "0°", "45°", "90°", "135°"]
    assert len(positions) == len(labels)
    # The second guide is the outer circle, which the longest wedge reaches.
    assert np.linalg.norm(guides[1], axis=-1) == pytest.approx(1.0)
    # Zero degrees points right, the way `rose` draws it.
    assert positions[labels.index("0°")][0] > 0.0
    assert positions[labels.index("0°")][1] == pytest.approx(0.0)


def test_a_zero_onset_is_a_time_rather_than_a_subfault_at_rest() -> None:
    """The cells the rupture nucleated on are the ones the clock is zeroed at.

    `_histogram` counts zeros out and says so, which is right for slip and rise time
    and wrong for onset: the nucleation cells would leave the distribution, taking the
    left edge of the propagation with them and reporting themselves as never having
    moved. On the shipped Wellington scenario that is 91 subfaults.
    """
    rerun = pytest.importorskip("rerun")
    onset = np.array([0.0, 0.0, 1.0, 2.0])
    recorder = _Recorder(rerun)

    view._histogram(
        recorder,
        "/histogram/onset",
        onset,
        (0.0, 2.0),
        2,
        lambda values: view.viridis(values, 0.0, 2.0),
        "onset time (s)",
        zero_is_rest=False,
    )

    labels = recorder.logged["/histogram/onset/axis/labels"].labels
    assert not [
        label for label in labels.as_arrow_array().to_pylist() if "at rest" in label
    ]
    # All four are binned: two at zero, and the mode is therefore the first bar.
    boxes = recorder.logged["/histogram/onset"]
    heights = (
        -2.0
        * np.array(boxes.centers.as_arrow_array().to_pylist(), dtype=np.float64)[:, 1]
    )
    assert heights[0] == pytest.approx(1.0)


def _segment(corners: np.ndarray, **fields: object) -> view.Segment:
    """A segment with only its geometry filled in, for the drawing helpers."""
    flat = np.zeros(len(corners))
    return view.Segment(
        **{
            "name": "segment",
            "cells": (1, len(corners)),
            "corners_m": corners,
            "centres_m": corners.mean(axis=1),
            "slip_m": flat,
            "rise_time_s": flat,
            "onset_s": flat,
            "rake_deg": flat,
            "area_m2": flat,
            "strike_deg": flat,
            "dip_deg": flat,
            "pulse_offsets": np.zeros(len(corners) + 1, dtype=np.int64),
            "pulse_samples": np.zeros(0),
            "sample_interval_s": 0.1,
            **fields,
        }
    )


def _lattice(rows: int, columns: int) -> np.ndarray:
    """Positions on a flat unit lattice, ``(rows, columns, 3)``."""
    along, down = np.meshgrid(
        np.arange(columns, dtype=float), np.arange(rows, dtype=float)
    )
    return np.stack([along, down, np.zeros_like(along)], axis=-1)


def test_an_isochrone_lies_where_the_front_actually_was() -> None:
    """Marching squares, checked against two fronts whose contours are known exactly.

    A front whose onset is the east coordinate has a straight isochrone at every level,
    and one spreading from a point has a circular one at its own radius. Both are
    asserted because they fail differently: a transposed lattice still draws straight
    lines, and only the circle says the two axes were not swapped.
    """
    positions = _lattice(40, 60)
    straight = view.isochrones(positions[..., 0].copy(), positions, 5.0)

    assert straight[..., 0].min() == pytest.approx(5.0)
    assert straight[..., 0].max() == pytest.approx(5.0)
    # It reaches both edges, rather than being a fragment that happens to be straight.
    assert straight[..., 1].min() == pytest.approx(0.0)
    assert straight[..., 1].max() == pytest.approx(39.0)

    centre = np.array([30.0, 20.0, 0.0])
    radial = view.isochrones(
        np.linalg.norm(positions - centre, axis=-1), positions, 10.0
    )
    radii = np.linalg.norm(radial.reshape(-1, 3) - centre, axis=-1)

    # Off only by where a straight chord cuts the corner of a curve on a unit lattice.
    assert radii.mean() == pytest.approx(10.0, abs=0.01)
    assert radii.std() < 0.05


def test_a_front_that_never_arrived_is_not_drawn_as_one_that_did() -> None:
    """An unruptured patch has no onset, and a contour through it is not a line.

    The interpolation divides by the difference between two onsets, so a patch left at
    infinity by an eikonal solve that never reached it would put NaN vertices into the
    line strip -- which Rerun draws as a line to nowhere rather than as nothing.
    """
    positions = _lattice(40, 60)
    onset = np.linalg.norm(positions - np.array([30.0, 20.0, 0.0]), axis=-1)
    onset[18:22, 28:32] = np.inf

    crossings = view.isochrones(onset, positions, 10.0)

    assert np.isfinite(crossings).all()
    assert len(crossings) < len(view.isochrones(onset, positions, 10.0)) + 1
    # A level nothing reaches draws nothing, rather than an empty strip.
    assert view.isochrones(onset, positions, 1.0e6).shape == (0, 2, 3)


@pytest.mark.parametrize(
    ("duration_s", "step_s"),
    [(2.0, 0.25), (8.0, 1.0), (38.5, 5.0), (120.0, 15.0), (400.0, 60.0)],
)
def test_isochrones_are_spaced_at_a_number_a_reader_counts_in(
    duration_s: float, step_s: float
) -> None:
    """The interval comes off a ladder of round numbers, coarsest that still fits.

    Dividing the range by a fixed count would space a 38.5 second rupture every 3.85
    seconds, which makes the reader do arithmetic to answer how long it took the front
    to cross from one line to the next.
    """
    levels = view.contour_levels(0.0, duration_s)

    assert len(levels) <= view.TARGET_CONTOURS
    assert np.allclose(np.diff(levels), step_s)
    # The first line is a whole step in, because a contour through the onset minimum is
    # the single subfault the rupture nucleated on.
    assert levels[0] == pytest.approx(step_s)
    assert levels[-1] < duration_s


def _draw_colourbar(low: float = 0.0, high: float = 38.46) -> _Recorder:
    """One colourbar for a field over `low` to `high`, drawn into a recorder."""
    rerun = pytest.importorskip("rerun")
    recorder = _Recorder(rerun)
    view._log_colourbar(
        recorder,
        "/fault/onset",
        (low, high),
        lambda values: view.viridis(values, low, high),
        "onset time (s)",
        (np.array([0.0, 0.0, -20_000.0]), np.array([90_000.0, 40_000.0, 0.0])),
    )
    return recorder


def test_a_colourbar_is_labelled_only_where_it_reaches() -> None:
    """A tick past the top of the bar states a value the field never took.

    `_ticks` overshoots on purpose, so that a tick landing exactly on the top survives
    a floating-point comparison. Drawn, that put a `40` at the top of a bar whose field
    stops at 38.46 -- and `_fraction` clamps, so it sat on the 38.46 end reading as it.
    """
    labels = (
        _draw_colourbar()
        .logged["/fault/onset/colourbar/labels"]
        .labels.as_arrow_array()
        .to_pylist()
    )

    assert "onset time (s)" in labels
    assert "30" in labels
    assert "40" not in labels


def test_a_colourbar_carries_the_same_map_as_the_fault_it_stands_beside() -> None:
    """Otherwise it is a decoration, and reading a value off the fault by eye fails."""
    mesh = _draw_colourbar().logged["/fault/onset/colourbar"]

    colours = np.array(mesh.vertex_colors.as_arrow_array().to_pylist(), dtype=np.uint32)
    ramp = view.viridis(np.linspace(0.0, 38.46, view.COLOURBAR_CELLS), 0.0, 38.46)
    expected = (
        (ramp[:, 0].astype(np.uint32) << 24)
        | (ramp[:, 1].astype(np.uint32) << 16)
        | (ramp[:, 2].astype(np.uint32) << 8)
        | 255
    )
    # Four vertices a quad, and the ramp is drawn twice: see the crossed strips below.
    assert np.array_equal(colours[::4][: view.COLOURBAR_CELLS], expected)


def test_a_colourbar_cannot_be_seen_edge_on() -> None:
    """A flat bar in a 3D view vanishes at the azimuth that looks along it.

    Rerun 0.35 has no screen-space overlay for a 3D view, so the bar is scene geometry
    and the camera can go anywhere. Two strips crossed at right angles mean that at
    every azimuth at least one of them is presenting a face.
    """
    mesh = _draw_colourbar().logged["/fault/onset/colourbar"]

    vertices = np.array(mesh.vertex_positions.as_arrow_array().to_pylist())
    quads = vertices.reshape(-1, 4, 3)
    assert len(quads) == 2 * view.COLOURBAR_CELLS
    # One strip has width along east and none along north, and the other the reverse.
    spans = np.ptp(quads, axis=1)
    first, second = spans[0], spans[view.COLOURBAR_CELLS]
    assert first[0] > 0.0 and first[1] == 0.0
    assert second[1] > 0.0 and second[0] == 0.0


def test_a_label_leaves_the_fault_along_its_normal_rather_than_upwards() -> None:
    """Up is inside a vertical fault, which is most of the crustal faults there are.

    A label logged at a position on the mesh z-fights it: the text flickers through the
    surface it is naming. Lifting it up the way a viewer's vertical axis points fixes a
    thrust and does nothing at all for a strike-slip fault, so the lift is along the
    surface's own normal.
    """
    # One vertical cell striking north: anticlockwise from the shallow near corner.
    vertical = np.array(
        [[[0.0, 0.0, 0.0], [0.0, 1e3, 0.0], [0.0, 1e3, -1e3], [0.0, 0.0, -1e3]]]
    )

    normal = view._outward(_segment(vertical))

    assert np.linalg.norm(normal) == pytest.approx(1.0)
    assert normal[2] == pytest.approx(0.0), "the lift would stay inside the fault"
    assert abs(normal[0]) == pytest.approx(1.0)


def test_the_hypocentre_marker_stays_put_and_only_its_label_moves() -> None:
    """The marker is a measurement and the label is furniture.

    Moving the marker off the surface would misstate where the rupture nucleated, so
    the offset is the label's alone and a leader line joins the two. The label is a
    child entity, which is what makes it -- and the leader under it -- switchable off
    in the blueprint without losing the marker.
    """
    rerun = pytest.importorskip("rerun")
    vertical = np.array(
        [[[0.0, 0.0, 0.0], [0.0, 1e3, 0.0], [0.0, 1e3, -1e3], [0.0, 0.0, -1e3]]]
    )
    hypocentre = np.array([0.0, 500.0, -500.0])
    segment = _segment(vertical, hypocentre_m=hypocentre)
    drawing = view.Drawing(
        {}, {}, (np.array([0.0, 0.0, -1e3]), np.array([1e3, 1e3, 0.0])), 40
    )
    recorder = _Recorder(rerun)

    view._log_hypocentre(recorder, [segment], drawing)

    marker = recorder.logged["/fault/hypocentre/segment"]
    label = recorder.logged["/fault/hypocentre/segment/label"]
    assert "/fault/hypocentre/segment/label/leader" in recorder.logged
    assert np.allclose(
        marker.positions.as_arrow_array().to_pylist()[0], hypocentre, atol=1e-6
    )
    at = np.array(label.positions.as_arrow_array().to_pylist())
    # One label, off the plane east = 0 that the fault lies in. Not one a side: Rerun
    # draws a label over whatever is in front of it, so a mirrored one is not hidden by
    # the fault, it is written on top of the first.
    assert len(at) == 1
    assert not np.isclose(at[0, 0], 0.0)
    assert at[0, 1:] == pytest.approx(hypocentre[1:], abs=1e-6)
    assert len(label.labels.as_arrow_array()) == 1


def test_isochrones_are_drawn_on_both_faces_of_the_fault() -> None:
    """A contour lifted off one face is behind the fault from half of the viewpoints.

    Lifting it clear of the surface is what stops it z-fighting the mesh, and a single
    side would be defensible for one fault. For a system of them each segment has its
    own normal, so no camera position has all the contours in front of their faults --
    turning it moves the problem to a different segment rather than solving it.
    """
    rerun = pytest.importorskip("rerun")
    # A vertical fault in the plane east = 0, with the front spreading along strike.
    lattice = _lattice(8, 12)
    corners = np.stack(
        [
            np.stack(
                [
                    np.zeros(4),
                    np.array([0.0, 1.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, -1.0, -1.0]),
                ],
                axis=-1,
            )
        ]
    )
    segment = _segment(
        corners,
        cells=(8, 12),
        centres_m=np.stack(
            [np.zeros(96), lattice[..., 0].ravel(), -lattice[..., 1].ravel()], axis=-1
        ),
        onset_s=lattice[..., 0].copy(),
    )
    drawing = view.Drawing(
        {}, {}, (np.array([0.0, 0.0, -8.0]), np.array([0.0, 12.0, 0.0])), 40
    )
    recorder = _Recorder(rerun)

    view._log_isochrones(recorder, [segment], drawing, np.array([5.0]))

    strips = np.array(
        recorder.logged["/fault/onset/isochrones/segment"]
        .strips.as_arrow_array()
        .to_pylist()
    )
    east = strips.reshape(-1, 3)[:, 0]
    assert (east > 0.0).any() and (east < 0.0).any()
    # The same line twice, mirrored: as much of it one side as the other. The label is
    # not, which the assertion below is about.
    assert np.count_nonzero(east > 0.0) == np.count_nonzero(east < 0.0)
    labels = (
        recorder.logged["/fault/onset/isochrones/segment/labels"]
        .labels.as_arrow_array()
        .to_pylist()
    )
    assert labels == ["5 s"], "the line is mirrored; its label is not"


def _quads(rows: int, columns: int) -> np.ndarray:
    """A vertical lattice of unit cells, cornered the way `load` corners a chart."""
    down, along = np.meshgrid(
        np.arange(rows + 1, dtype=float),
        np.arange(columns + 1, dtype=float),
        indexing="ij",
    )
    nodes = np.stack([np.zeros_like(along), along, -down], axis=-1)
    return np.stack(
        [nodes[:-1, :-1], nodes[:-1, 1:], nodes[1:, 1:], nodes[1:, :-1]], axis=2
    ).reshape(-1, 4, 3)


def test_a_plane_joint_is_drawn_down_the_column_the_file_records() -> None:
    """A fault is one chart over all its planes, and the mesh says nothing about them.

    Which matters most in the case that looks least like it matters: a workflow that
    cuts a straight trace into equal lengths writes several planes that are collinear,
    and a continuous mesh over them is indistinguishable from a fault with one. The
    joint is the only thing on screen that tells the two apart.
    """
    rerun = pytest.importorskip("rerun")
    segment = _segment(_quads(3, 4), cells=(3, 4), plane=np.array([0, 0, 1, 1]))
    drawing = view.Drawing({}, {}, (np.zeros(3), np.array([0.0, 4.0, 3.0])), 40)
    recorder = _Recorder(rerun)

    view._log_joints(recorder, [segment], drawing)

    strips = (
        recorder.logged["/fault/slip/joints/segment"]
        .strips.as_arrow_array()
        .to_pylist()
    )
    # One joint, on both faces, as for a contour and for the same reason.
    assert len(strips) == 2
    line = np.array(strips[0])
    # The whole way down dip, on the node column between cell columns 1 and 2.
    assert len(line) == 4
    assert line[:, 1] == pytest.approx(2.0)
    assert sorted(line[:, 2]) == pytest.approx([-3.0, -2.0, -1.0, 0.0])
    assert not np.isclose(line[0, 0], 0.0), "the joint is inside the mesh it marks"


def test_a_fault_with_nothing_to_join_draws_no_joints() -> None:
    """One plane has no interior boundary, and an SRF does not record planes at all.

    In an SRF a plane *is* a segment, so the question never arises -- and drawing a
    joint round the edge of every segment would answer it wrongly rather than not at all.
    """
    rerun = pytest.importorskip("rerun")
    corners = _quads(3, 4)
    drawing = view.Drawing({}, {}, (np.zeros(3), np.array([0.0, 4.0, 3.0])), 40)

    single = _Recorder(rerun)
    view._log_joints(
        single, [_segment(corners, cells=(3, 4), plane=np.zeros(4, dtype=int))], drawing
    )
    unrecorded = _Recorder(rerun)
    view._log_joints(unrecorded, [_segment(corners, cells=(3, 4))], drawing)

    assert single.logged == {}
    assert unrecorded.logged == {}


def test_the_joints_are_in_the_slip_view_and_start_switched_off() -> None:
    """They are a fact about how the input was written down, not about the earthquake.

    So they ship present and invisible: the reader who wants to know where the planes
    meet finds the switch in the blueprint panel, and everyone else never sees a line
    that is not a measurement.
    """
    blueprint = pytest.importorskip("rerun.blueprint")

    faults = view.layout(blueprint).root_container.contents[1]
    slip = faults.contents[0]

    assert slip.name == "slip"
    assert "/fault/slip/**" in slip.contents
    behaviour = slip.visualizer_overrides["/fault/slip/joints"]
    assert behaviour.visible.as_arrow_array().to_pylist() == [False]
