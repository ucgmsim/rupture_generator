"""``rupture-generator view``: watch a rupture happen.

Opens a Rerun viewer showing the fault in three dimensions, coloured by a field, with a
time scrub that plays the rupture at real speed -- and beside it the moment release,
the distribution of whichever field is selected, and the numbers that say what kind of
earthquake this is.

# Why Rerun rather than a plot

The thing worth looking at is *propagation*, and a still image cannot show it. A viewer
that can is a time-aware one, and Rerun's timeline gives play, pause, loop and a rate
multiplier for free, synchronised across every panel, provided the time is logged as a
**duration in seconds** rather than a frame number.

# The render mesh has unshared vertices

Rerun colours by *vertex*, so a vertex shared between neighbouring cells would
interpolate a piecewise-constant field and draw values that were never computed -- a
slip of 0.4 m blended into its neighbour's 3 m across the seam between them. So this
draws four vertices per cell with one flat colour each. That is a display list rather
than a mesh, and the difference matters exactly here.

# Large ruptures are decimated for display, and say so

A 100 m rupture of a whole fault system is millions of subfaults and tens of millions
of vertices; Rerun cannot animate that, and neither can a screen resolve it. Above a
cell budget the *displayed* mesh is strided down and a banner says by how much.
Histograms and statistics are always computed from every subfault, so only the picture
is coarsened -- which is the honest way round, because a decimated statistic would be
wrong where a decimated picture is merely coarse.

# What animates

`slip` is **cumulative slip at t**, integrated from each subfault's own pulse, and it
is the propagation. Rise time and rake are properties of the finished rupture rather
than of a moment in it, so they are drawn once and the cursor drives the moment release
instead.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import xarray as xr

from rupture_generator.formats import Format, from_path
from rupture_generator.formats.rupture import read_rupture, segments_in
from rupture_generator.moment import cumulative_moment, moment_rate, rigidity_pa
from rupture_generator.scripts.errors import console

FIELDS = {
    "slip": ("slip_m", "metres", "hot"),
    "rise-time": ("rise_time_s", "seconds", "viridis"),
    "rake": ("rake_deg", "degrees", "quiver"),
}
"""Which variable each field view shows, its unit, and how it is drawn.

Slip gets **hot** because it is a magnitude with a meaningful zero: black where the
fault did not move, and brightening monotonically in luminance, so the asperities read
as bright patches at a glance. Rise time gets **viridis** because it has no meaningful
zero and wants a perceptually even ramp rather than an implied floor. Rake is an
*angle*, so no colour ramp is honest for it at all -- a cyclic quantity on a linear
scale puts 179 and -179 degrees at opposite ends -- and it is drawn as arrows instead.
"""

ANIMATED = "slip"
"""The only field that is a function of time. See the module note."""

DEFAULT_CELL_BUDGET = 50_000
"""How many cells to draw before striding the display down.

Chosen from what a viewer can usefully show rather than from what it can survive: at
50,000 cells a 1,600-pixel window has about six pixels per subfault, so striding
further would discard detail the screen could have resolved, and drawing more would
paint several subfaults into one pixel.
"""

MAX_ARROWS = 3_000
"""How many rake arrows to draw. Beyond this they overlap into a solid mass and stop
being readable as directions."""


@dataclasses.dataclass(frozen=True)
class Segment:
    """One fault segment, in the local metric frame the viewer draws in.

    Both input formats reduce to this, which is what lets everything below be written
    once. Positions are **metres east, north and up** from the rupture's own centroid:
    up rather than down because a viewer's vertical axis points up and both file
    formats measure depth downwards.

    Attributes
    ----------
    corners_m : np.ndarray
        ``(cells, 4, 3)``, anticlockwise from the shallow near corner. Unshared
        between cells on purpose -- see the module note.
    rigidity_pa : np.ndarray or None
        Present when the file carries the material properties the generator sampled.
        An SRF version 2.0 does; the native format does not, and the moment release is
        then drawn at a nominal rigidity with the panel saying so.
    """

    name: str
    cells: tuple[int, int]
    corners_m: np.ndarray
    centres_m: np.ndarray
    slip_m: np.ndarray
    rise_time_s: np.ndarray
    onset_s: np.ndarray
    rake_deg: np.ndarray
    area_m2: np.ndarray
    strike_deg: np.ndarray
    dip_deg: np.ndarray
    pulse_offsets: np.ndarray
    pulse_samples: np.ndarray
    sample_interval_s: float
    hypocentre_m: np.ndarray | None = None
    rigidity_pa: np.ndarray | None = None

    def values(self, variable: str) -> np.ndarray:
        """One field, flattened over subfaults."""
        return getattr(self, variable).ravel()


# ============================================================================
# Colour
# ============================================================================


def viridis(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Perceptually even, and the right choice for a field with no meaningful zero."""
    return _ramp(_VIRIDIS_16, values, low, high)


def hot(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Black through red and yellow to white.

    Built from the definition rather than a table: the red channel saturates over the
    first three eighths, green over the next three, and blue only in the last quarter.
    Luminance rises monotonically, which is what makes a zero read as absent rather
    than as merely low.
    """
    fraction = _fraction(values, low, high)
    red = np.clip(fraction / 0.365, 0.0, 1.0)
    green = np.clip((fraction - 0.365) / 0.381, 0.0, 1.0)
    blue = np.clip((fraction - 0.746) / 0.254, 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).round().astype(np.uint8)


def _fraction(values: np.ndarray, low: float, high: float) -> np.ndarray:
    span = high - low
    if span <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - low) / span, 0.0, 1.0)


def _ramp(
    anchors: np.ndarray, values: np.ndarray, low: float, high: float
) -> np.ndarray:
    position = _fraction(values, low, high) * (len(anchors) - 1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weight = (position - lower)[:, None]
    blended = anchors[lower] * (1.0 - weight) + anchors[upper] * weight
    return blended.round().astype(np.uint8)


# A 16-entry viridis, linearly interpolated. Vendored rather than importing matplotlib
# for a lookup table: it is the only thing that would have been used from it.
_VIRIDIS_16 = np.array(
    [
        (68, 1, 84),
        (71, 24, 106),
        (72, 45, 117),
        (69, 65, 125),
        (64, 84, 131),
        (57, 102, 135),
        (50, 119, 138),
        (44, 136, 139),
        (39, 152, 138),
        (39, 168, 133),
        (54, 183, 122),
        (85, 197, 104),
        (124, 208, 80),
        (169, 217, 51),
        (216, 222, 26),
        (253, 231, 37),
    ],
    dtype=np.float64,
)

COLOURMAPS = {"hot": hot, "viridis": viridis}


# ============================================================================
# Loading -- both formats reduce to Segment
# ============================================================================


def load(path: Path) -> tuple[list[Segment], str]:
    """Read a rupture, whatever it is written as.

    Returns
    -------
    tuple
        The segments, and a sentence saying what was read and how faithfully.

    Raises
    ------
    ValueError
        For a format this cannot read.
    """
    chosen = from_path(path)
    if chosen in (Format.NETCDF, Format.ZARR):
        return _from_rupture_file(path), "native rupture file"
    if chosen is Format.SRF:
        return _from_srf(path), "SRF, mesh reconstructed from subfault centres"
    raise ValueError(
        f"a rupture cannot be read from {chosen.value}: this reads the native format "
        "and text SRF"
    )


def _local_frame(
    positions: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """A common origin for every segment, and the stacked positions relative to it."""
    stacked = np.concatenate([p.reshape(-1, 3) for p in positions])
    return stacked.mean(axis=0), stacked


def _from_rupture_file(path: Path) -> list[Segment]:
    """The native format, where the nodes are stored and nothing is reconstructed."""
    segments: list[Segment] = []
    with read_rupture(path) as tree:
        found = segments_in(tree)
        if not found:
            raise ValueError(f"{path} holds no rupture")

        origin = None
        for name, dataset in found:
            east = dataset["node_east_km"].to_numpy() * 1000.0
            north = dataset["node_north_km"].to_numpy() * 1000.0
            up = -dataset["node_depth_km"].to_numpy() * 1000.0
            nodes = np.stack([east, north, up], axis=-1)
            if origin is None:
                origin = np.array([nodes[..., 0].mean(), nodes[..., 1].mean(), 0.0])

            corners = (
                np.stack(
                    [
                        nodes[:-1, :-1],
                        nodes[:-1, 1:],
                        nodes[1:, 1:],
                        nodes[1:, :-1],
                    ],
                    axis=2,
                ).reshape(-1, 4, 3)
                - origin
            )

            hypocentre = None
            if "hypocentre_strike_km" in dataset.attrs:
                hypocentre = _hypocentre_position(dataset, nodes, origin)

            segments.append(
                Segment(
                    name=name,
                    cells=(dataset.sizes["i"], dataset.sizes["j"]),
                    corners_m=corners,
                    centres_m=corners.mean(axis=1),
                    slip_m=dataset["slip_m"].to_numpy(),
                    rise_time_s=dataset["rise_time_s"].to_numpy(),
                    onset_s=dataset["onset_s"].to_numpy(),
                    rake_deg=dataset["rake_deg"].to_numpy(),
                    area_m2=dataset["area_m2"].to_numpy(),
                    strike_deg=dataset["strike_deg"].to_numpy(),
                    dip_deg=dataset["dip_deg"].to_numpy(),
                    pulse_offsets=dataset["slip_rate_offset"].to_numpy(),
                    pulse_samples=dataset["slip_rate"].to_numpy(),
                    sample_interval_s=float(dataset.attrs["sample_interval_s"]),
                    hypocentre_m=hypocentre,
                )
            )
    return segments


def _hypocentre_position(
    dataset: xr.Dataset, nodes: np.ndarray, origin: np.ndarray
) -> np.ndarray:
    """Where the rupture nucleated, from the arc lengths the file records.

    The file stores the hypocentre the way the config states it -- as distances along
    strike and down dip -- so this walks the same arc lengths back to a position,
    which is the one conversion that keeps the marker on the cell the pipeline
    actually seeded.
    """
    strike_arc = dataset["strike_km"].to_numpy()
    dip_arc = dataset["dip_km"].to_numpy()
    j = int(
        np.searchsorted(
            strike_arc[1:-1], dataset.attrs["hypocentre_strike_km"], "right"
        )
    )
    i = int(np.searchsorted(dip_arc[1:-1], dataset.attrs["hypocentre_dip_km"], "right"))
    cell = nodes[i : i + 2, j : j + 2].reshape(-1, 3).mean(axis=0)
    return cell - origin


def _from_srf(path: Path) -> list[Segment]:
    """An SRF, whose mesh is reconstructed from what the format does store.

    An SRF records subfault *centres*, not corners, so the quads here are built by
    stepping half a cell along strike and down dip from each centre, using the cell
    size the plane header implies. That is exact for the uniform planar grids an SRF
    describes and approximate for nothing else, which is the same assumption the
    format itself makes.
    """
    from rupture_generator.srf import read_srf

    srf = read_srf(path)
    points = srf.points

    latitude = np.asarray(points.latitude_deg, dtype=np.float64)
    longitude = np.asarray(points.longitude_deg, dtype=np.float64)
    depth_m = np.asarray(points.depth_km, dtype=np.float64) * 1000.0

    # A local tangent frame about the rupture's own centroid. Good to a few metres
    # over a fault system, which is far below what a viewer resolves.
    lat0, lon0 = float(latitude.mean()), float(longitude.mean())
    metres_per_degree = 111_320.0
    east = (longitude - lon0) * metres_per_degree * math.cos(math.radians(lat0))
    north = (latitude - lat0) * metres_per_degree
    centres = np.stack([east, north, -depth_m], axis=-1)

    segments: list[Segment] = []
    start = 0
    for index, plane in enumerate(srf.planes):
        count = plane.strike_count * plane.dip_count
        block = slice(start, start + count)
        start += count

        strike = np.radians(np.asarray(points.strike_deg[block], dtype=np.float64))
        dip = np.radians(np.asarray(points.dip_deg[block], dtype=np.float64))
        along = np.stack(
            [np.sin(strike), np.cos(strike), np.zeros_like(strike)], axis=-1
        )
        # Down dip, in east-north-up: the horizontal part turns ninety degrees off
        # strike and the vertical part goes down.
        down = np.stack(
            [
                np.sin(strike + np.pi / 2) * np.cos(dip),
                np.cos(strike + np.pi / 2) * np.cos(dip),
                -np.sin(dip),
            ],
            axis=-1,
        )
        half_length = plane.length_km * 1000.0 / plane.strike_count / 2.0
        half_width = plane.width_km * 1000.0 / plane.dip_count / 2.0

        middle = centres[block]
        corners = np.stack(
            [
                middle - along * half_length - down * half_width,
                middle + along * half_length - down * half_width,
                middle + along * half_length + down * half_width,
                middle - along * half_length + down * half_width,
            ],
            axis=1,
        )

        rise = np.asarray(points.rise_time_s[block], dtype=np.float64)
        rigidity = None
        if points.shear_speed_cm_s is not None and points.density_g_cm3 is not None:
            # cm/s back to km/s, and the density is already what rigidity_pa wants.
            rigidity = rigidity_pa(
                np.asarray(points.shear_speed_cm_s[block], dtype=np.float64) / 1.0e5,
                np.asarray(points.density_g_cm3[block], dtype=np.float64),
            )

        offsets = srf.slip_rate.indptr[block.start : block.stop + 1]
        samples = srf.slip_rate.data[offsets[0] : offsets[-1]] / 100.0

        segments.append(
            Segment(
                name=f"plane_{index}",
                cells=(plane.dip_count, plane.strike_count),
                corners_m=corners,
                centres_m=middle,
                slip_m=np.asarray(points.slip_cm[block], dtype=np.float64) / 100.0,
                rise_time_s=rise,
                onset_s=np.asarray(points.onset_s[block], dtype=np.float64),
                rake_deg=np.asarray(points.rake_deg[block], dtype=np.float64),
                area_m2=np.asarray(points.area_cm2[block], dtype=np.float64) / 1.0e4,
                strike_deg=np.asarray(points.strike_deg[block], dtype=np.float64),
                dip_deg=np.asarray(points.dip_deg[block], dtype=np.float64),
                pulse_offsets=offsets - offsets[0],
                pulse_samples=samples,
                sample_interval_s=float(points.sample_interval_s[block.start]),
                hypocentre_m=None,
                rigidity_pa=rigidity,
            )
        )
    return segments


# ============================================================================
# What the numbers say
# ============================================================================


NOMINAL_RIGIDITY_PA = 3.0e10
"""30 GPa, crustal rock. Used only where a file does not carry the rigidity the
generator sampled, and the panel says when that is."""


def statistics(segments: list[Segment]) -> tuple[str, np.ndarray, np.ndarray, bool]:
    """The numbers that say what kind of earthquake this is, and the moment release.

    Returns
    -------
    tuple
        A markdown summary, the total moment released up to each display time, the
        display times, and whether the rigidity was the file's own or nominal.
    """
    exact = all(segment.rigidity_pa is not None for segment in segments)

    moment_nm = 0.0
    area_m2 = 0.0
    for segment in segments:
        rigidity = (
            segment.rigidity_pa
            if segment.rigidity_pa is not None
            else np.full(segment.slip_m.size, NOMINAL_RIGIDITY_PA)
        )
        moment_nm += float(
            np.sum(rigidity.ravel() * segment.area_m2.ravel() * segment.slip_m.ravel())
        )
        area_m2 += float(np.sum(segment.area_m2))

    magnitude = (math.log10(moment_nm) - 9.0499505) / 1.5 if moment_nm > 0 else 0.0

    starts = [float(segment.onset_s.min()) for segment in segments]
    # When the last subfault *stops* slipping, not when the last one starts: the
    # duration of the earthquake rather than of the front's travel.
    ends = [
        float((segment.onset_s + segment.rise_time_s).max()) for segment in segments
    ]
    duration_s = max(ends) - min(starts)

    slip = np.concatenate([segment.slip_m.ravel() for segment in segments])
    lines = [
        "# Rupture",
        "",
        "| | |",
        "| --- | --- |",
        f"| moment magnitude | **{magnitude:.2f}** |",
        f"| seismic moment | {moment_nm:.3e} N m |",
        f"| fault area | {area_m2 / 1.0e6:,.0f} km² |",
        f"| rupture duration | {duration_s:.1f} s |",
        f"| segments | {len(segments)} |",
        f"| subfaults | {slip.size:,} |",
        f"| mean slip | {slip.mean():.2f} m |",
        f"| peak slip | {slip.max():.2f} m |",
        "",
        (
            "Moment from the file's own rigidity."
            if exact
            else f"Moment at a nominal {NOMINAL_RIGIDITY_PA / 1e9:.0f} GPa: this "
            "format does not carry the rigidity the generator sampled."
        ),
    ]
    return "\n".join(lines), np.array([]), np.array([]), exact


def moment_release(
    segments: list[Segment], times_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Moment rate and cumulative moment over the display times.

    Both come from the same pulses, and the second is the running integral of the
    first, so a viewer comparing them is looking at one quantity two ways rather than
    at two computations that might disagree.
    """
    total = np.zeros_like(times_s)
    for segment in segments:
        rigidity = (
            segment.rigidity_pa.ravel()
            if segment.rigidity_pa is not None
            else np.full(segment.slip_m.size, NOMINAL_RIGIDITY_PA)
        )
        times, rate = moment_rate(
            segment.pulse_offsets,
            segment.pulse_samples,
            segment.onset_s.ravel(),
            segment.area_m2.ravel(),
            rigidity,
            sample_interval_s=segment.sample_interval_s,
            duration_s=float(times_s[-1] - times_s[0]),
        )
        total += np.interp(times_s, times, rate, left=0.0, right=0.0)
    return total, cumulative_moment(times_s, total)


def cumulative_slip(segment: Segment, times_s: np.ndarray) -> np.ndarray:
    """How much each subfault has slipped by each time, in metres.

    Each subfault's pulse integrated up to `t`, placed at its own onset. This is what
    makes the animation a rupture rather than a slide show: at any moment, the part of
    the fault that has moved is exactly the part the front has reached.
    """
    offsets = segment.pulse_offsets
    samples = segment.pulse_samples
    onset_s = segment.onset_s.ravel()
    interval_s = segment.sample_interval_s

    slipped = np.zeros((len(times_s), onset_s.size), dtype=np.float64)
    for cell in range(onset_s.size):
        start, stop = int(offsets[cell]), int(offsets[cell + 1])
        if stop == start:
            continue
        running = np.cumsum(samples[start:stop]) * interval_s
        into = np.floor((times_s - onset_s[cell]) / interval_s).astype(np.int64)
        slipped[:, cell] = np.where(
            into < 0, 0.0, running[np.clip(into, 0, len(running) - 1)]
        )
    return slipped


# ============================================================================
# Drawing
# ============================================================================


def stride_for(segments: list[Segment], budget: int) -> int:
    """How many cells to skip so the drawn mesh fits the budget.

    A stride rather than a resampling: every drawn cell is a real subfault with its
    real value, and the ones between are simply not drawn. Averaging into
    super-cells would paint values no subfault has.
    """
    total = sum(segment.slip_m.size for segment in segments)
    if total <= budget:
        return 1
    return math.ceil(math.sqrt(total / budget))


def strided(segment: Segment, stride: int) -> np.ndarray:
    """The flat indices of the cells to draw."""
    cells_i, cells_j = segment.cells
    rows = np.arange(0, cells_i, stride)
    columns = np.arange(0, cells_j, stride)
    return (rows[:, None] * cells_j + columns[None, :]).ravel()


def slip_direction(segment: Segment, keep: np.ndarray) -> np.ndarray:
    """Unit vectors along which each subfault slipped, in east-north-up.

    Rake is measured within the fault plane from the strike direction towards up-dip,
    so the slip direction is ``cos(rake)`` along strike plus ``sin(rake)`` up dip.
    Drawing it needs the plane's own basis, which strike and dip give.
    """
    strike = np.radians(segment.strike_deg.ravel()[keep])
    dip = np.radians(segment.dip_deg.ravel()[keep])
    rake = np.radians(segment.rake_deg.ravel()[keep])

    along = np.stack([np.sin(strike), np.cos(strike), np.zeros_like(strike)], axis=-1)
    up_dip = np.stack(
        [
            -np.sin(strike + np.pi / 2) * np.cos(dip),
            -np.cos(strike + np.pi / 2) * np.cos(dip),
            np.sin(dip),
        ],
        axis=-1,
    )
    return along * np.cos(rake)[:, None] + up_dip * np.sin(rake)[:, None]


def rose(values_deg: np.ndarray, bins: int = 36) -> list[np.ndarray]:
    """A rake distribution as wedges on a circle.

    A rake is an *angle*, so a bar chart of it is misleading twice over: it puts -179
    and 179 degrees at opposite ends of the axis when they are a degree apart, and it
    invites reading the horizontal axis as a magnitude. A rose puts the wrap where it
    belongs, which is nowhere.

    Returns
    -------
    list of np.ndarray
        One closed polygon per bin, in a two-dimensional view's own coordinates.
    """
    counts, edges = np.histogram(
        np.mod(values_deg + 180.0, 360.0) - 180.0, bins=bins, range=(-180.0, 180.0)
    )
    longest = counts.max() or 1
    wedges = []
    for count, low, high in zip(counts, edges[:-1], edges[1:], strict=True):
        if count == 0:
            continue
        radius = count / longest
        angles = np.radians(np.linspace(low, high, 6))
        # Screen coordinates with y downwards, so a rake of zero points right and the
        # circle runs the way a compass does.
        arc = np.stack([radius * np.cos(angles), -radius * np.sin(angles)], axis=-1)
        wedges.append(np.vstack([[0.0, 0.0], arc, [0.0, 0.0]]))
    return wedges


def require_rerun():
    """Import Rerun, or say how to get it.

    Guarded rather than imported at the top because `rerun-sdk` pulls pyarrow and
    pillow, and neither `mesh` nor `generate` needs a display. A cluster running a
    thousand realisations should not be installing a GUI stack.
    """
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:  # pragma: no cover - exercised by not having it
        message = (
            "rerun-sdk is required for `view`. Install it with:\n"
            "    pip install 'rupture-generator[vis]'"
        )
        raise ImportError(message) from error
    return rr, rrb


def layout(blueprint):  # noqa: ANN001 - Rerun's types
    """The panels: the fault, and three ways of reading the same numbers.

    Each selector is a tab strip rather than a command-line flag, because the whole
    point of a viewer is to look at one thing and then another without re-running it.
    """
    return blueprint.Blueprint(
        blueprint.Horizontal(
            blueprint.Vertical(
                blueprint.TextDocumentView(origin="/statistics", name="statistics"),
                blueprint.Tabs(
                    blueprint.BarChartView(origin="/histogram/slip", name="slip"),
                    blueprint.BarChartView(
                        origin="/histogram/rise_time", name="rise time"
                    ),
                    blueprint.Spatial2DView(origin="/histogram/rake", name="rake"),
                    name="distribution",
                ),
                blueprint.Tabs(
                    blueprint.TimeSeriesView(origin="/moment/rate", name="moment rate"),
                    blueprint.TimeSeriesView(
                        origin="/moment/cumulative", name="cumulative moment"
                    ),
                    name="moment release",
                ),
                row_shares=[2, 3, 3],
            ),
            blueprint.Tabs(
                blueprint.Spatial3DView(
                    origin="/fault",
                    name="slip",
                    contents=["/fault/slip/**", "/fault/hypocentre/**"],
                ),
                blueprint.Spatial3DView(
                    origin="/fault",
                    name="rise time",
                    contents=["/fault/rise_time/**", "/fault/hypocentre/**"],
                ),
                blueprint.Spatial3DView(
                    origin="/fault",
                    name="rake",
                    contents=["/fault/rake/**", "/fault/hypocentre/**"],
                ),
            ),
            column_shares=[1, 3],
        ),
        collapse_panels=True,
    )


def view(
    rupture: Annotated[
        Path,
        typer.Argument(
            help="Rupture file: the native format, or a text SRF.",
            exists=True,
            readable=True,
        ),
    ],
    time_step: Annotated[
        float, typer.Option(help="Seconds between animation frames.")
    ] = 0.25,
    bins: Annotated[int, typer.Option(help="Histogram bins.")] = 40,
    max_cells: Annotated[
        int,
        typer.Option(help="Cells to draw before the display is strided down."),
    ] = DEFAULT_CELL_BUDGET,
    save: Annotated[
        Path | None,
        typer.Option(help="Write a .rrd recording instead of opening a window."),
    ] = None,
) -> None:
    """Show a rupture propagating, with its moment release and field distributions."""
    try:
        rerun, blueprint = require_rerun()
    except ImportError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    try:
        segments, provenance = load(rupture)
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    rerun.init("rupture-generator", spawn=save is None)
    if save is not None:
        rerun.save(save)

    rerun.send_blueprint(layout(blueprint))
    log_rupture(rerun, segments, provenance, time_step, bins, max_cells)

    if save is not None:
        console.print(f"[green]wrote[/green] {save}")


def log_rupture(
    rerun,  # noqa: ANN001 - Rerun's module
    segments: list[Segment],
    provenance: str,
    time_step: float,
    bins: int,
    max_cells: int,
) -> None:
    """Log every panel, over the rupture's own timeline."""
    stride = stride_for(segments, max_cells)
    kept = {segment.name: strided(segment, stride) for segment in segments}

    summary, _, _, _ = statistics(segments)
    drawn = sum(len(indices) for indices in kept.values())
    total = sum(segment.slip_m.size for segment in segments)
    note = f"\n\nRead from a {provenance}." + (
        f"\n\n**Displayed at 1 cell in {stride}²** — {drawn:,} of {total:,} "
        "subfaults drawn. Every statistic and histogram above uses all of them."
        if stride > 1
        else ""
    )
    rerun.log(
        "/statistics",
        rerun.TextDocument(summary + note, media_type="text/markdown"),
        static=True,
    )

    # The whole rupture on one clock: the earliest onset to the last pulse's end.
    starts = [float(segment.onset_s.min()) for segment in segments]
    ends = [
        float((segment.onset_s + segment.rise_time_s).max()) for segment in segments
    ]
    times_s = np.arange(min(starts), max(ends) + time_step, time_step)

    _log_static_fields(rerun, segments, kept, bins)
    _log_hypocentre(rerun, segments)

    rate, cumulative = moment_release(segments, times_s)
    slipped = {segment.name: cumulative_slip(segment, times_s) for segment in segments}
    peak = max(float(values.max()) for values in slipped.values()) or 1.0

    for step, moment_s in enumerate(times_s):
        rerun.set_time("rupture", duration=float(moment_s))
        rerun.log("/moment/rate", rerun.Scalars(float(rate[step])))
        rerun.log("/moment/cumulative", rerun.Scalars(float(cumulative[step])))

        frame = []
        for segment in segments:
            indices = kept[segment.name]
            current = slipped[segment.name][step][indices]
            rerun.log(
                f"/fault/slip/{segment.name}",
                _mesh(rerun, segment.corners_m[indices], hot(current, 0.0, peak)),
            )
            frame.append(slipped[segment.name][step])
        rerun.log(
            "/histogram/slip",
            rerun.BarChart(
                np.histogram(np.concatenate(frame), bins=bins, range=(0.0, peak))[0]
            ),
        )


def _log_static_fields(
    rerun,  # noqa: ANN001
    segments: list[Segment],
    kept: dict[str, np.ndarray],
    bins: int,
) -> None:
    """Rise time and rake, which are properties of the finished rupture."""
    rise = np.concatenate([segment.rise_time_s.ravel() for segment in segments])
    rake = np.concatenate([segment.rake_deg.ravel() for segment in segments])
    low, high = float(rise.min()), float(rise.max())

    for segment in segments:
        indices = kept[segment.name]
        rerun.log(
            f"/fault/rise_time/{segment.name}",
            _mesh(
                rerun,
                segment.corners_m[indices],
                viridis(segment.rise_time_s.ravel()[indices], low, high),
            ),
            static=True,
        )

    # Rake, as arrows along the direction each subfault slipped. Thinned further than
    # the mesh: arrows overlap into a solid mass long before cells do.
    for segment in segments:
        indices = kept[segment.name]
        budget = max(1, MAX_ARROWS // len(segments))
        if len(indices) > budget:
            indices = indices[:: math.ceil(len(indices) / budget)]
        slip = segment.slip_m.ravel()[indices]
        scale = float(np.linalg.norm(segment.corners_m[0, 1] - segment.corners_m[0, 0]))
        lengths = scale * 6.0 * slip / (slip.max() or 1.0)
        rerun.log(
            f"/fault/rake/{segment.name}",
            rerun.Arrows3D(
                origins=segment.centres_m[indices],
                vectors=slip_direction(segment, indices) * lengths[:, None],
                colors=hot(slip, 0.0, float(slip.max()) or 1.0),
            ),
            static=True,
        )

    rerun.log(
        "/histogram/rise_time",
        rerun.BarChart(np.histogram(rise, bins=bins, range=(low, high))[0]),
        static=True,
    )
    rerun.log(
        "/histogram/rake",
        rerun.LineStrips2D(rose(rake), colors=[(216, 222, 26)]),
        static=True,
    )


def _log_hypocentre(rerun, segments: list[Segment]) -> None:  # noqa: ANN001
    """Where the rupture nucleated, on whichever segment holds it."""
    for segment in segments:
        if segment.hypocentre_m is None:
            continue
        rerun.log(
            "/fault/hypocentre",
            rerun.Points3D(
                positions=[segment.hypocentre_m],
                radii=[max(400.0, float(np.abs(segment.corners_m).max()) / 120.0)],
                colors=[(255, 64, 64)],
                labels=[f"hypocentre ({segment.name})"],
            ),
            static=True,
        )


def _mesh(rerun, corners: np.ndarray, colours: np.ndarray):  # noqa: ANN001
    """A `Mesh3D` of flat-coloured quads, two triangles each."""
    cells = len(corners)
    vertices = corners.reshape(-1, 3).astype(np.float32)
    base = np.arange(cells) * 4
    triangles = np.empty((cells * 2, 3), dtype=np.uint32)
    triangles[0::2] = np.stack([base, base + 1, base + 2], axis=-1)
    triangles[1::2] = np.stack([base, base + 2, base + 3], axis=-1)
    return rerun.Mesh3D(
        vertex_positions=vertices,
        triangle_indices=triangles,
        vertex_colors=np.repeat(colours, 4, axis=0),
    )


__all__ = [
    "FIELDS",
    "Segment",
    "cumulative_slip",
    "hot",
    "load",
    "moment_release",
    "rose",
    "statistics",
    "stride_for",
    "view",
    "viridis",
]
