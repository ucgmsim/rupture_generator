"""``rupture-generator view``, tested where the logic actually is.

The Rerun calls are thin glue. What carries meaning is four pure functions -- the render
mesh, the colour map, the cumulative slip that makes the animation a rupture, and the
moment rate -- and those are testable properly, so they are tested properly.

The logging itself gets a smoke test: `--save` writes a recording and the command exits
cleanly. Rerun 0.35 has no read-back API in the Python SDK and an `.rrd` is compressed,
so asserting on its *contents* is not available; asserting that it is large and was
written is what is.

Skipped entirely when `rerun-sdk` is absent, because it is an optional extra -- a cluster
running a thousand realisations should not be installing a GUI stack.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from rupture_generator.formats.rupture import planes_in, read_rupture
from rupture_generator.scripts.cli import app
from rupture_generator.scripts.view import (
    FIELDS,
    cumulative_slip,
    moment_rate_of,
    quads,
    viridis,
)

rerun = pytest.importorskip("rerun", reason="the `vis` extra is not installed")

runner = CliRunner()
EXAMPLES = Path(__file__).parent.parent / "examples"

SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def run(*arguments: object) -> object:
    return runner.invoke(app, [str(argument) for argument in arguments])


@pytest.fixture(scope="module")
def rupture(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A bent fault, generated once. Two planes, so the multi-plane paths are covered."""
    directory = tmp_path_factory.mktemp("view")
    mesh = directory / "hope.h5"
    assert run("mesh", EXAMPLES / "hope.geometry.toml", mesh, "--quiet").exit_code == 0

    output = directory / "hope.rupture.h5"
    assert (
        run("generate", EXAMPLES / "crustal.toml", mesh, output, "--quiet").exit_code
        == 0
    )
    return output


@pytest.fixture
def plane(rupture: Path):
    with read_rupture(rupture) as tree:
        return planes_in(tree)[0][2].load()


class TestTheRenderMesh:
    """Four vertices per cell, deliberately unshared.

    Rerun colours by *vertex*, so a mesh whose cells shared corners would interpolate a
    piecewise-constant field across them -- drawing a slip of 40 cm blending into its
    neighbour's 300 cm, which is a value that was never computed. This is a display list
    rather than a mesh, and that is the difference.
    """

    def test_there_are_four_vertices_and_two_triangles_per_cell(self, plane: object) -> None:
        vertices, triangles = quads(plane)
        cells = plane["slip_cm"].size

        assert vertices.shape == (4 * cells, 3)
        assert triangles.shape == (2 * cells, 3)

    def test_the_vertices_are_not_shared(self, plane: object) -> None:
        """The property the whole approach rests on.

        A shared-vertex mesh of the same fault has one vertex per *node*, which is far
        fewer. If this ever stopped being true, the colours would start interpolating
        and nothing else would fail.
        """
        vertices, _ = quads(plane)
        nodes = plane["node_east_km"].size

        assert len(np.unique(vertices, axis=0)) == pytest.approx(nodes, rel=0.01)
        assert len(vertices) > 3 * nodes, "the vertices look shared"

    def test_every_triangle_indexes_a_real_vertex(self, plane: object) -> None:
        vertices, triangles = quads(plane)
        assert triangles.max() < len(vertices)
        assert triangles.min() >= 0

    def test_up_is_positive(self, plane: object) -> None:
        """The file measures depth downwards and a viewer's vertical axis points up.

        Getting this backwards draws the fault above the ground, mirrored -- which looks
        like a fault and is one nobody has.
        """
        vertices, _ = quads(plane)
        depth_km = plane["node_depth_km"].to_numpy()

        assert vertices[:, 2].max() == pytest.approx(-depth_km.min() * 1000.0, abs=1e-2)
        assert vertices[:, 2].min() == pytest.approx(-depth_km.max() * 1000.0, abs=1e-2)

    def test_the_positions_are_in_metres(self, plane: object) -> None:
        """A viewer's units are metres, and the file's are kilometres."""
        vertices, _ = quads(plane)
        span_m = vertices[:, 2].max() - vertices[:, 2].min()
        depth_span_km = float(
            plane["node_depth_km"].max() - plane["node_depth_km"].min()
        )
        assert span_m == pytest.approx(depth_span_km * 1000.0, rel=1e-6)

    def test_each_cell_gets_its_own_four_corners(self, plane: object) -> None:
        """And they are its corners, not a neighbour's.

        Checked on the first cell against the node array directly, because an off-by-one
        in the corner offsets produces a mesh that is the right size and one cell out of
        place -- which is invisible on a rectangle.
        """
        vertices, _ = quads(plane)
        east = plane["node_east_km"].to_numpy() * 1000.0
        north = plane["node_north_km"].to_numpy() * 1000.0

        # A tenth of a millimetre. The vertices are `float32`, which is what a GPU
        # takes, and float32 resolves about 1e-4 m at the ~1e3 m coordinates a fault
        # spans -- so this is the format's own precision rather than a slack bound.
        for index, (down, along) in enumerate([(0, 0), (0, 1), (1, 1), (1, 0)]):
            assert vertices[index][0] == pytest.approx(east[down, along], abs=1e-4)
            assert vertices[index][1] == pytest.approx(north[down, along], abs=1e-4)


class TestTheColourMap:
    @given(
        values=st.lists(
            st.floats(min_value=-1e4, max_value=1e4), min_size=1, max_size=50
        )
    )
    @SETTINGS
    def test_it_always_returns_bytes(self, values: list[float]) -> None:
        colours = viridis(np.array(values), -1e4, 1e4)
        assert colours.shape == (len(values), 3)
        assert colours.dtype == np.uint8

    def test_the_ends_are_the_ends(self) -> None:
        colours = viridis(np.array([0.0, 1.0]), 0.0, 1.0)
        assert tuple(colours[0]) == (68, 1, 84), "not viridis' dark end"
        assert tuple(colours[-1]) == (253, 231, 37), "not viridis' bright end"

    def test_it_is_monotone_in_brightness(self) -> None:
        """Which is what makes it readable as a scale rather than a decoration."""
        colours = viridis(np.linspace(0.0, 1.0, 64), 0.0, 1.0).astype(float)
        luminance = colours @ np.array([0.2126, 0.7152, 0.0722])
        assert np.all(np.diff(luminance) > 0.0)

    @given(value=st.floats(min_value=-1e3, max_value=1e3))
    @SETTINGS
    def test_a_degenerate_range_does_not_divide_by_zero(self, value: float) -> None:
        """A fault where every subfault slipped the same amount is a valid fault.

        It also has `high == low`, and the obvious implementation gives NaN colours --
        which Rerun renders as nothing at all.
        """
        colours = viridis(np.full(5, value), value, value)
        assert colours.dtype == np.uint8
        assert np.all(colours == colours[0])

    def test_values_outside_the_range_are_clamped(self) -> None:
        below = viridis(np.array([-100.0]), 0.0, 1.0)
        above = viridis(np.array([100.0]), 0.0, 1.0)
        assert tuple(below[0]) == (68, 1, 84)
        assert tuple(above[0]) == (253, 231, 37)


class TestCumulativeSlip:
    """What makes the animation a rupture rather than a slide show."""

    @pytest.fixture
    def times_s(self, plane: object) -> np.ndarray:
        last = float(plane["onset_s"].max() + plane["rise_time_s"].max())
        return np.arange(float(plane["onset_s"].min()), last + 0.05, 0.05)

    def test_it_never_goes_backwards(self, plane: object, times_s: np.ndarray) -> None:
        """Slip accumulates. A subfault does not un-slip."""
        slipped = cumulative_slip(plane, times_s)
        assert np.all(np.diff(slipped, axis=0) >= -1e-9)

    def test_it_ends_at_the_total_slip(self, plane: object, times_s: np.ndarray) -> None:
        """The identity: integrating a subfault's whole pulse gives its slip.

        Measured at 9.5e-16 relative, which is the pulse's own arithmetic rather than
        anything this adds.
        """
        final = cumulative_slip(plane, times_s)[-1]
        expected = plane["slip_cm"].to_numpy().ravel()
        assert final == pytest.approx(expected, rel=1e-9, abs=1e-9)

    def test_nothing_has_slipped_at_the_first_moment(
        self,
        plane: object,
        times_s: np.ndarray,
    ) -> None:
        """The animation starts with an unbroken fault, which is the point of it."""
        assert np.all(cumulative_slip(plane, times_s)[0] == 0.0)

    def test_a_subfault_stays_still_until_its_onset(
        self, plane: object, times_s: np.ndarray
    ) -> None:
        """The propagation itself: at any moment, the part that has moved is the part
        the front has reached.

        Asserted over every subfault rather than a sample, because a sign error on the
        onset offset makes the whole fault slip at once -- which still animates, and
        still ends in the right place.
        """
        slipped = cumulative_slip(plane, times_s)
        onset_s = plane["onset_s"].to_numpy().ravel()

        for cell, onset in enumerate(onset_s):
            before = times_s < onset
            assert np.all(slipped[before, cell] == 0.0), cell

    def test_the_front_spreads_rather_than_appearing(
        self,
        plane: object,
        times_s: np.ndarray,
    ) -> None:
        """The fraction of the fault that has moved increases, and starts small."""
        slipped = cumulative_slip(plane, times_s)
        moving = (slipped > 0.0).mean(axis=1)

        assert moving[0] == 0.0
        assert moving[len(moving) // 2] > 0.1
        assert moving[-1] > 0.5
        assert np.all(np.diff(moving) >= 0.0), "the front receded"


class TestTheMomentRatePanel:
    def test_it_is_non_negative_and_finite(self, rupture: Path) -> None:
        with read_rupture(rupture) as tree:
            planes = planes_in(tree)
            times_s = np.linspace(0.0, 25.0, 200)
            rate = moment_rate_of(planes, times_s)

        assert len(rate) == len(times_s)
        assert np.all(rate >= 0.0)
        assert np.all(np.isfinite(rate))

    def test_it_sums_across_planes(self, rupture: Path) -> None:
        """A bent fault is one earthquake, so its panel is one curve."""
        with read_rupture(rupture) as tree:
            planes = planes_in(tree)
            assert len(planes) == 2

            times_s = np.linspace(0.0, 25.0, 200)
            whole = moment_rate_of(planes, times_s)
            first = moment_rate_of(planes[:1], times_s)

        assert whole.sum() > first.sum()


class TestTheCommandRuns:
    @pytest.mark.parametrize("field", sorted(FIELDS))
    def test_every_field_records(
        self, tmp_path: Path, rupture: Path, field: str
    ) -> None:
        """Including the three that do not animate, which take a different path."""
        output = tmp_path / f"{field}.rrd"
        result = run("view", rupture, "--field", field, "--save", output)

        assert result.exit_code == 0, result.output
        assert output.stat().st_size > 10_000, "the recording looks empty"

    def test_an_unknown_field_is_refused_with_the_choices(
        self, tmp_path: Path, rupture: Path
    ) -> None:
        result = run(
            "view", rupture, "--field", "displacement", "--save", tmp_path / "x.rrd"
        )

        assert result.exit_code == 1
        assert "slip" in result.output, "the message does not say what is available"

    def test_a_coarser_step_records_less(self, tmp_path: Path, rupture: Path) -> None:
        """`--time-step` is doing something, which a smoke test alone would not say."""
        fine, coarse = tmp_path / "fine.rrd", tmp_path / "coarse.rrd"
        run("view", rupture, "--time-step", 0.05, "--save", fine)
        run("view", rupture, "--time-step", 0.5, "--save", coarse)

        assert coarse.stat().st_size < fine.stat().st_size

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        result = run("view", tmp_path / "absent.h5", "--save", tmp_path / "x.rrd")
        assert result.exit_code != 0

    def test_help_exits_cleanly(self) -> None:
        assert runner.invoke(app, ["view", "--help"]).exit_code == 0

    def test_the_help_says_which_fields_animate(self) -> None:
        """Because "scrub a static field" is a reasonable expectation and is not what
        happens."""
        output = runner.invoke(app, ["view", "--help"]).output
        assert "animate" in output
