"""What the viewer's panels put on their axes, and what its decimation preserves.

Drawing is not an assertion, so most of this module is untested on purpose. Two parts
are not drawings and are pinned here.

The three axis helpers turn numbers into axes, and each replaced something that was
wrong in a way a reader could not see: a histogram indexed by bin number reads as a
distribution over the quantity, a moment axis of ``6.000000e18`` reads as precision,
and a rose with no circle round it reads as whatever the reader assumed.

The **decimation** is arithmetic. A triangular rupture is redrawn on a coarser surface
of the viewer's own making, and the whole question is what survives that: the moment
must be the file's exactly, or the panel beside the picture states a magnitude the file
does not contain. The rest of the tests below are the other side of that -- that no
drawn cell shows a timing or an angle no subfault had.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pyproj
import pytest

from rupture_generator.moment import moment_of
from rupture_generator.scripts import view
from rupture_generator.triangular.mesh import TriangleMesh, write_mesh

NZTM = pyproj.CRS("EPSG:2193")

MOMENT_EXACT = 1.0e-12
"""How closely a decimated rupture's moment must match the file's.

An identity, not a tolerance: the drawn cell's area is its subfaults' summed area and
its slip is their area-weighted mean, so ``sum(mu A s)`` over drawn cells is the same
sum of the same products in a different order. What is left is float64 re-association
over 1,600 terms, which is ~1e-15 relative; 1e-12 is three orders above that and ten
orders below the one-percent slip bound, so a failure here is a real error and never
arithmetic. Measured on the fixture below: 0.0.
"""


def _rupture(strike: int = 40, dip: int = 20, interval_s: float = 0.1) -> TriangleMesh:
    """A planar triangulated segment with a rupture on it, and pulses that agree.

    Fields a reader can check an aggregate against: slip and rake vary cell to cell so
    an average is distinguishable from a member of the set, onset is a front travelling
    along strike at 2 km/s so a display cell spans a spread of arrival times, and
    rigidity varies with depth so an area-weighted mean of it is *not* the weighting
    that preserves moment. Each pulse is a half-sine normalised to carry exactly its
    own subfault's slip, which is what makes the moment-release panel checkable.
    """
    east = np.arange(strike + 1, dtype=np.float64)
    depth = np.arange(dip + 1, dtype=np.float64)
    grid_east, grid_depth = np.meshgrid(east, depth, indexing="xy")
    mesh = TriangleMesh.from_patches(
        [np.stack([grid_east, np.zeros_like(grid_east), grid_depth + 1.0], axis=-1)],
        strike_deg=90.0,
        dip_deg=90.0,
        origin_east_km=1600.0,
        origin_north_km=5400.0,
        surface="synthetic",
    )

    generator = np.random.default_rng(11)
    faces = mesh.face_count
    centres = mesh.centres()
    slip_m = 1.0 + generator.random(faces)
    rise_time_s = 0.5 + generator.random(faces)
    onset_s = centres[:, 0] / 2.0
    rake_deg = 170.0 + 20.0 * generator.random(faces)

    lengths = np.maximum(np.rint(rise_time_s / interval_s).astype(np.int64), 1)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    samples = np.zeros(int(offsets[-1]))
    for face in range(faces):
        shape = np.sin(np.linspace(0.0, np.pi, int(lengths[face])))
        samples[offsets[face] : offsets[face + 1]] = (
            shape * slip_m[face] / (shape.sum() * interval_s)
        )

    return (
        mesh.with_fields(
            slip_m=slip_m,
            rise_time_s=rise_time_s,
            onset_s=onset_s,
            rake_deg=rake_deg,
            rigidity_pa=2.0e10 + 1.0e10 * centres[:, 2] / centres[:, 2].max(),
        )
        .with_pulses(offsets, samples)
        .with_attrs(
            sample_interval_s=interval_s,
            hypocentre_strike_km=3.5,
            hypocentre_dip_km=4.5,
        )
    )


def _written(tmp_path: Path, mesh: TriangleMesh) -> Path:
    """That segment as a version 3 file on disk, which is what `load` reads."""
    path = tmp_path / "synthetic.rupture.h5"
    write_mesh({"synthetic": [mesh]}, NZTM, path)
    return path


def test_a_decimated_rupture_carries_the_files_moment(tmp_path: Path) -> None:
    """The picture is coarser than the file; the moment beside it is not.

    Both halves of the claim. The drawn cells' own ``sum(mu A s)`` is the file's, which
    is what makes the fault a reader is looking at the earthquake the file holds; and
    the moment released by the *aggregated pulses* comes to the same number, which is
    what makes the moment-release panel the same rupture as the surface above it.

    This is the assertion the whole decimation is arranged around -- area-weighted slip
    on summed area, and rigidity weighted by ``area x slip`` rather than by area,
    without which a rupture whose rigidity varies with depth loses moment cell by cell.
    """
    mesh = _rupture()
    truth_nm = moment_of(mesh["slip_m"], mesh["rigidity_pa"], mesh.areas_km2())

    segments, provenance = view.load(_written(tmp_path, mesh), max_cells=100)
    (segment,) = segments

    assert len(segment.corners_m) < mesh.face_count, "nothing was decimated"
    assert "1,600 faces redrawn on" in provenance
    drawn_nm = float(np.sum(segment.area_m2 * segment.rigidity_pa * segment.slip_m))
    assert drawn_nm == pytest.approx(truth_nm, rel=MOMENT_EXACT)

    # The pulses, integrated the way the panel integrates them. Read before
    # `CumulativeSlip` overwrites the rates with their running sum.
    times_s = np.arange(0.0, 30.0, 0.05)
    _, cumulative = view.moment_release(segments, times_s)
    assert float(cumulative[-1]) == pytest.approx(truth_nm, rel=MOMENT_EXACT)

    # And the statistics panel, which reads the file's own subfaults rather than the
    # drawn cells, so it states the file's magnitude and the file's subfault count.
    summary = view.statistics(segments)
    assert f"{truth_nm:.3e} N m" in summary
    assert "| subfaults | 1,600 |" in summary
    assert f"{float(mesh['slip_m'].max()):.2f} m" in summary


def test_a_display_cell_carries_its_own_subfaults_values(tmp_path: Path) -> None:
    """Nothing drawn is an average of a quantity an average is wrong for.

    Rake and rise time are the nearest subfault's *actual* values, so they are members
    of the file's own set -- the property the strided path had, that no drawn cell shows
    a value no subfault had. Onset is the earliest of the cell's subfaults, so the cell
    starts moving exactly when its first subfault does and never after. Slip is the one
    average, and it is bracketed by the subfaults it averages.
    """
    mesh = _rupture()
    segments, _ = view.load(_written(tmp_path, mesh), max_cells=100)
    (segment,) = segments
    subfaults = segment.population

    assert np.isin(segment.rake_deg, subfaults.rake_deg).all()
    assert np.isin(segment.rise_time_s, subfaults.rise_time_s).all()
    assert np.isin(segment.onset_s, subfaults.onset_s).all()
    assert segment.onset_s.min() == pytest.approx(float(subfaults.onset_s.min()))
    # Averaging the front's arrival would put the cell's onset later than its first
    # subfault's, which is the smear this rule exists to refuse.
    display, cell_of_face = view.decimate(mesh, 2 * 100)
    del display
    for cell in (0, len(segment.onset_s) // 2, len(segment.onset_s) - 1):
        mine = subfaults.onset_s[cell_of_face == cell]
        assert segment.onset_s[cell] == pytest.approx(float(mine.min()))
        assert segment.slip_m[cell] <= subfaults.slip_m[cell_of_face == cell].max()
        assert segment.slip_m[cell] >= subfaults.slip_m[cell_of_face == cell].min()


def test_every_subfault_lands_in_exactly_one_display_cell(tmp_path: Path) -> None:
    """The assignment is a partition, which is why the moment cannot leak.

    Nearest-centre assignment gives every subfault one home and no subfault two, so a
    sum over display cells is a sum over the file's subfaults. The areas are the visible
    consequence: they add up to the file's total, not to the *drawn surface's*, which
    differs from it by the boundary staircase remeshing leaves.
    """
    mesh = _rupture()
    segments, _ = view.load(_written(tmp_path, mesh), max_cells=100)
    (segment,) = segments

    assert float(np.sum(segment.area_m2)) == pytest.approx(
        float(np.sum(segment.population.area_m2)), rel=MOMENT_EXACT
    )
    _, cell_of_face = view.decimate(mesh, 2 * 100)
    assert cell_of_face.shape == (mesh.face_count,)
    assert set(np.unique(cell_of_face)) <= set(range(len(segment.corners_m)))


def test_a_small_rupture_is_drawn_exactly_as_it_was_stored(tmp_path: Path) -> None:
    """Inside the budget there is nothing to gain by redrawing anything.

    A rupture whose faces fit the budget is drawn face for face, with the file's own
    values on them and its own pulses behind them -- so the decimation is a thing that
    happens to large files rather than a thing that always happens.
    """
    mesh = _rupture(strike=8, dip=8)
    segments, provenance = view.load(_written(tmp_path, mesh), max_cells=1_000)
    (segment,) = segments

    assert "redrawn" not in provenance
    assert len(segment.corners_m) == mesh.face_count
    assert segment.subfaults is None
    assert np.allclose(segment.slip_m, mesh["slip_m"])
    assert np.allclose(segment.onset_s, mesh["onset_s"])
    assert segment.pulse_samples.size == mesh.pulses[1].size


def test_the_pulses_are_carried_across_a_block_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blocking the one pass over the pulses changes nothing but the peak memory.

    The rates are the only part of a rupture file that does not fit -- 19.6 GB at a 400
    m cut -- so they are read in blocks. A block boundary falls in the middle of a
    subfault's pulse only if the arithmetic is wrong, and the assertion that it is not
    is that a pathologically small block gives bit-identical aggregates to one big
    enough to hold the file whole.
    """
    path = _written(tmp_path, _rupture())
    whole, _ = view.load(path, max_cells=100)

    monkeypatch.setattr(view, "PULSE_BLOCK_SAMPLES", 7)
    blocked, _ = view.load(path, max_cells=100)

    assert np.array_equal(whole[0].pulse_offsets, blocked[0].pulse_offsets)
    assert np.array_equal(whole[0].pulse_samples, blocked[0].pulse_samples)


def test_a_block_holds_whole_pulses_and_every_face_once(tmp_path: Path) -> None:
    """The blocks tile the faces: no face read twice, none missed, none split.

    Cut on the *samples* rather than on a face count, because a subfault's pulse is as
    long as its own rise time and a tapered edge cell has none at all.
    """
    mesh = _rupture()
    offsets = mesh.pulse_offsets
    blocks = list(view._face_blocks(offsets, 500))

    assert blocks[0][0] == 0
    assert blocks[-1][1] == mesh.face_count
    for (_, ends), (starts, _) in itertools.pairwise(blocks):
        assert ends == starts
    # Every block is inside the budget, unless one subfault's own pulse exceeds it --
    # a pulse is the unit that can be placed, so it is read whole.
    for first, last in blocks:
        held = int(offsets[last] - offsets[first])
        assert held <= 500 or last == first + 1


def test_the_hypocentre_marker_sits_on_the_subfault_that_nucleated(
    tmp_path: Path,
) -> None:
    """At the file's resolution, not the display's.

    The file records the hypocentre as arc lengths, and the marker is the centre of the
    face `TriangleMesh.cell_index` returns for them -- the same question the pipeline
    asked when it seeded the wavefront. A display cell is kilometres across, so placing
    the marker on one would be the one drawn thing whose *position* is wrong rather than
    merely coarse.
    """
    mesh = _rupture()
    segments, _ = view.load(_written(tmp_path, mesh), max_cells=100)
    (segment,) = segments

    face = mesh.cell_index(3.5, 4.5)
    centre_km = mesh.vertices_km()[mesh.faces()[face]].mean(axis=0)
    # The viewer's frame: metres east, north and up about the rupture's own centroid.
    vertices_km = mesh.vertices_km()
    offset_m = 1000.0 * np.array(
        [
            centre_km[0] - vertices_km[:, 0].mean(),
            centre_km[1] - vertices_km[:, 1].mean(),
            -centre_km[2],
        ]
    )
    assert segment.hypocentre_m == pytest.approx(offset_m)


def test_a_cell_is_fanned_into_triangles_whatever_its_arity() -> None:
    """Three corners or four, and a quad still splits the way it always did.

    The mesh writer reads the arity off ``corners.shape`` so that neither track pays
    for the other, and the risk in that is a fan that silently disagrees with the
    two-triangle split the structured viewer drew for years.
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
