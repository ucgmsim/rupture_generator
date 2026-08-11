"""What the viewer's panels put on their axes.

The viewer is otherwise untested -- it draws, and a drawing is not an assertion. But
the three helpers here turn numbers into axes, and each replaced something that was
wrong in a way a reader could not see: a histogram indexed by bin number reads as a
distribution over the quantity, a moment axis of ``6.000000e18`` reads as precision,
and a rose with no circle round it reads as whatever the reader assumed. Those are
worth pinning even though the panels they feed are not.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rupture_generator.scripts import view


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
