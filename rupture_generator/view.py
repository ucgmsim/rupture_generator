"""``rupture-view RUPTURE.srf``: watch a rupture happen, in Rerun.

Reads the version 2 SRF ``rupture-generator`` writes, through
:func:`~rupture_generator.formats.srf.read_rupture`. Each point is drawn as its own
quadrilateral, sized from its plane's cells and turned by its own strike and dip, in
metres east, north and **up** from the rupture's centre -- up because a viewer's
vertical axis points up.

Slip is shown accumulating on ``hot``, each point's pulse integrated from its own
onset; onset and rise time are shown whole on viridis, and rake as arrows along each
point's slip, coloured by slip. Moment is counted from the same integration, and each
field's distribution is drawn beside the fault, slip's growing with it.
"""

import argparse
import math
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyproj

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.formats.srf import M_PER_KM, SrfRupture, read_rupture
from rupture_generator.rupture.source import magnitude_from_moment

if TYPE_CHECKING:
    import rerun as rr


def _packed(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    r, g, b = (np.round(channel).astype(np.uint32) for channel in (red, green, blue))
    return (r << 24) | (g << 16) | (b << 8) | 0xFF


_SHADE = np.linspace(0.0, 1.0, 256)

# Matplotlib's viridis at 16 evenly spaced anchors, rounded to 8-bit channels.
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
    ]
)

VIRIDIS = _packed(
    *(
        np.interp(_SHADE, np.linspace(0.0, 1.0, len(_VIRIDIS_16)), _VIRIDIS_16[:, k])
        for k in range(3)
    )
)
"""Viridis in 256 shades, packed RGBA as Rerun takes colours: for onset and rise
time, which have no meaningful zero."""

HOT = _packed(
    *(
        255.0 * np.clip((_SHADE - start) / span, 0.0, 1.0)
        for start, span in ((0.0, 0.365), (0.365, 0.381), (0.746, 0.254))
    )
)
"""Matplotlib's hot, black through red and yellow to white, for slip: red saturates
over the first three eighths, green over the next three, blue over the last quarter."""

MAX_ARROWS = 3_000
"""How many rake arrows to draw; beyond this they overlap into a solid mass."""

CONTOUR_LINE = 0xECF0F6FF
"""Near-white, because an isochrone has to be followed across the whole of viridis."""

CONTOUR_STEPS_S = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0, 120.0, 300.0)
"""The isochrone spacings worth offering, coarsest wins: round numbers a reader counts
in, so the gap between two lines needs no arithmetic."""

TARGET_CONTOURS = 10
"""How many isochrones a fault can carry before they stop being separable."""

UI_POINTS = -1.0
"""Rerun reads a negative radius as a width in screen points rather than metres, so a
contour stays visible however far the camera is zoomed out."""

TABS = {"slip": "slip", "onset": "onset", "rise_time": "rise time", "rake": "rake"}

BINS = 40
"""Histogram bins per field."""


def colours(
    values: np.ndarray, low: float, high: float, cmap: np.ndarray = VIRIDIS
) -> np.ndarray:
    """A packed colour from ``cmap`` for each value between ``low`` and ``high``."""
    shade = (values - low) * (255.0 / ((high - low) or 1.0))
    return cmap[np.clip(shade, 0.0, 255.0).astype(np.uint8)]


def histogram(values: np.ndarray, low: float, high: float) -> rr.BarChart:
    """``values`` counted in :data:`BINS` bins from ``low`` to ``high``, as a bar chart."""
    import rerun as rr

    counts, edges = np.histogram(values, bins=BINS, range=(low, high))
    return rr.BarChart(
        counts,
        abscissa=0.5 * (edges[:-1] + edges[1:]),
        widths=np.full(BINS, edges[1] - edges[0]),
    )


def positions_m(rupture: SrfRupture) -> tuple[np.ndarray, np.ndarray | None]:
    """Each point's cell, ``(points, 4, 3)``, and the hypocentre, in the viewer's frame."""
    frame = pyproj.Transformer.from_crs(
        "EPSG:4326",
        pyproj.CRS.from_proj4(
            f"+proj=aeqd +lat_0={rupture.lat_deg.mean()} "
            f"+lon_0={rupture.lon_deg.mean()} +datum=WGS84"
        ),
        always_xy=True,
    )

    def local(lon: np.ndarray, lat: np.ndarray, depth_km: np.ndarray) -> np.ndarray:
        east, north = frame.transform(lon, lat)
        return np.stack([east, north, -M_PER_KM * np.asarray(depth_km)], axis=-1)

    planes = rupture.planes
    counts = [plane.nstk * plane.ndip for plane in planes]
    half_m = 0.5 * M_PER_KM
    half_length = np.repeat([half_m * p.len / p.nstk for p in planes], counts)[:, None]
    half_width = np.repeat([half_m * p.wid / p.ndip for p in planes], counts)[:, None]
    centres = local(rupture.lon_deg, rupture.lat_deg, rupture.depth_km)
    # The file's strike is from true north, and the frame's north is true only at its
    # centre: across a long rupture the meridians lean by degrees, and every cell would
    # turn with them.
    leaning = local(rupture.lon_deg, rupture.lat_deg + 1e-4, rupture.depth_km)
    convergence = np.arctan2(*(leaning - centres)[:, :2].T)
    strike = np.radians(rupture.strike_deg) + convergence
    dip = np.radians(rupture.dip_deg)
    along = np.stack([np.sin(strike), np.cos(strike), np.zeros_like(strike)], axis=-1)
    down = np.stack(
        [np.cos(strike) * np.cos(dip), -np.sin(strike) * np.cos(dip), -np.sin(dip)],
        axis=-1,
    )
    along, down = along * half_length, down * half_width
    corners = np.stack(
        [
            centres - along - down,
            centres + along - down,
            centres + along + down,
            centres - along + down,
        ],
        axis=1,
    )
    hypocentre = None if rupture.hypocentre is None else local(*rupture.hypocentre)
    return corners, hypocentre


def contour_levels(
    low: float, high: float, target: int = TARGET_CONTOURS
) -> np.ndarray:
    """Round times to draw isochrones at: the coarsest step on
    :data:`CONTOUR_STEPS_S` that fits ``target`` lines between ``low`` and ``high``.

    The lowest onset is skipped: a contour through the front's start is a point.
    """
    span = high - low
    if not math.isfinite(span) or span <= 0.0:
        return np.array([])
    step = next(
        (step for step in CONTOUR_STEPS_S if span / step <= target), CONTOUR_STEPS_S[-1]
    )
    first = math.ceil(low / step) * step
    return np.arange(first if first > low else first + step, high, step)


def isochrones(values: np.ndarray, positions: np.ndarray, level: float) -> np.ndarray:
    """Where a lattice of values crosses ``level``, as ``(segments, 2, 3)`` lines.

    Marching squares on the ``(i, j)`` lattice of ``values``, mapped onto the
    ``(i, j, 3)`` ``positions`` it is laid over, so a contour follows a curved fault.
    Non-finite values take no part. Built from the crossed edges rather than from
    every cell, since a contour touches the square root of the cells, not all of them.
    """
    finite = np.isfinite(values)
    above = np.where(finite, values, -np.inf) >= level
    down = (above[:, :-1] != above[:, 1:]) & finite[:, :-1] & finite[:, 1:]
    across = (above[:-1, :] != above[1:, :]) & finite[:-1, :] & finite[1:, :]

    # A cell's four edges, anticlockwise from the one along its low i side.
    cell = np.stack(
        [down[:-1, :], across[:, 1:], down[1:, :], across[:, :-1]], axis=-1
    ).reshape(-1, 4)
    rows, edges = np.nonzero(cell)
    if not rows.size:
        return np.empty((0, 2, 3))

    # Every cell crosses an even number of its edges, so `nonzero` -- ascending within
    # each row -- already pairs them; a saddle's four resolve one of its two ways.
    columns = values.shape[1] - 1
    i, j = rows // columns, rows % columns
    starts = (i + np.array([0, 0, 1, 0])[edges], j + np.array([0, 1, 0, 0])[edges])
    ends = (i + np.array([0, 1, 1, 1])[edges], j + np.array([1, 1, 1, 0])[edges])

    first, second = values[starts], values[ends]
    fraction = ((level - first) / (second - first))[:, None]
    crossings = positions[starts] + fraction * (positions[ends] - positions[starts])
    return crossings.reshape(-1, 2, 3)


def onset_contours(
    rupture: SrfRupture, corners: np.ndarray, lift_m: float
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Isochrones of onset, plane by plane, and one labelled anchor per line.

    Each line is drawn on both faces of its plane, ``lift_m`` clear of it: on the
    surface it z-fights the mesh, and on one face it vanishes from half the views.
    """
    levels = contour_levels(float(rupture.onset_s.min()), float(rupture.onset_s.max()))
    centres = corners.mean(axis=1)
    lines, anchors, labels = [], [], []
    start = 0
    for plane in rupture.planes:
        stop = start + plane.nstk * plane.ndip
        shape = (plane.ndip, plane.nstk)
        onset, positions = rupture.onset_s[start:stop], centres[start:stop]
        first = corners[start]
        normal = np.cross(first[1] - first[0], first[3] - first[0])
        lift = lift_m * normal / (np.linalg.norm(normal) or 1.0)
        start = stop
        for level in levels:
            crossings = isochrones(
                onset.reshape(shape), positions.reshape(*shape, 3), float(level)
            )
            if not len(crossings):
                continue
            lines.extend([crossings + lift, crossings - lift])
            # At the middle crossing, inside the fault rather than on its edge.
            anchors.append(crossings[len(crossings) // 2, 0] + lift)
            labels.append(f"{level:g} s")
    if not lines:
        return np.empty((0, 2, 3)), np.empty((0, 3)), []
    return np.concatenate(lines), np.array(anchors), labels


def slip_directions(rupture: SrfRupture, corners: np.ndarray) -> np.ndarray:
    """Unit vectors each point slipped along: ``cos(rake)`` along strike plus
    ``sin(rake)`` up dip, read off the cells :func:`positions_m` draws."""
    along = corners[:, 1] - corners[:, 0]
    up_dip = corners[:, 0] - corners[:, 3]
    along /= np.linalg.norm(along, axis=-1, keepdims=True)
    up_dip /= np.linalg.norm(up_dip, axis=-1, keepdims=True)
    rake = np.radians(rupture.rake_deg)[:, None]
    return along * np.cos(rake) + up_dip * np.sin(rake)


def slip_by(rupture: SrfRupture, times_s: Iterable[float]) -> Iterator[np.ndarray]:
    """Slip so far at each time, metres, one value per point.

    One running sum over every pulse laid end to end, in double precision: a point's
    slip by ``t`` is the difference across its own row up to ``t``.
    """
    offsets = rupture.pulse_offsets
    starts, lengths = offsets[:-1], np.diff(offsets)
    integral = np.empty(offsets[-1] + 1)
    integral[0] = 0.0
    np.cumsum(rupture.pulses_m_s, dtype=np.float64, out=integral[1:])
    before = integral[starts]
    for time_s in times_s:
        samples = np.floor((time_s - rupture.onset_s) / rupture.dt_s) + 1.0
        taken = np.clip(samples, 0, lengths).astype(np.int64)
        yield (integral[starts + taken] - before) * rupture.dt_s


def statistics(rupture: SrfRupture) -> str:
    """The rupture in a few lines of Markdown."""
    moment = float(np.sum(rupture.rigidity_pa * rupture.area_m2 * rupture.slip_m))
    slipping = rupture.slip_m > 0
    return "\n".join(
        [
            f"**Mw {magnitude_from_moment(moment):.2f}** ({moment:.3e} N m)",
            "",
            f"- {rupture.slip_m.size:,} points on {len(rupture.planes)} planes",
            (
                f"- slip: mean {rupture.slip_m[slipping].mean():.2f} m, "
                f"max {rupture.slip_m.max():.2f} m"
            ),
            (
                f"- rise time: mean {rupture.rise_time_s[slipping].mean():.2f} s, "
                f"max {rupture.rise_time_s.max():.2f} s"
            ),
            f"- onset: last subfault at {rupture.onset_s.max():.1f} s",
            f"- rake: {rupture.rake_deg.mean():.0f} ± {rupture.rake_deg.std():.0f} deg",
        ]
    )


def layout() -> object:
    """The fault in one tab per field; the numbers and moment release beside it."""
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.TextDocumentView(origin="/statistics", name="statistics"),
                rrb.Tabs(
                    *(
                        rrb.BarChartView(origin=f"/distribution/{key}", name=name)
                        for key, name in TABS.items()
                    )
                ),
                rrb.Tabs(
                    rrb.TimeSeriesView(origin="/moment/rate", name="moment rate"),
                    rrb.TimeSeriesView(
                        origin="/moment/cumulative", name="cumulative moment"
                    ),
                ),
                row_shares=[2, 3, 3],
            ),
            rrb.Tabs(
                *(
                    rrb.Spatial3DView(
                        origin="/fault",
                        name=name,
                        contents=[f"/fault/{key}/**", "/fault/hypocentre"],
                    )
                    for key, name in TABS.items()
                )
            ),
            column_shares=[1, 3],
        ),
        rrb.BlueprintPanel(state="hidden"),
        rrb.SelectionPanel(state="hidden"),
        rrb.TimePanel(state="collapsed"),
    )


def log(rupture: SrfRupture, step_s: float) -> None:
    """Log the static fields once, then slip and moment on the rupture's timeline."""
    import rerun as rr

    corners, hypocentre = positions_m(rupture)
    count = corners.shape[0]
    positions = corners.reshape(-1, 3).astype(np.float32)
    first = 4 * np.arange(count, dtype=np.uint32)[:, None]
    triangles = np.concatenate([first + [0, 1, 2], first + [0, 2, 3]]).astype(np.uint32)

    def mesh(path: str, packed: np.ndarray | None) -> None:
        rr.log(
            path,
            rr.Mesh3D(
                vertex_positions=positions,
                triangle_indices=triangles,
                vertex_colors=None if packed is None else np.repeat(packed, 4),
            ),
            static=True,
        )

    rr.log(
        "/statistics",
        rr.TextDocument(statistics(rupture), media_type="text/markdown"),
        static=True,
    )
    for key, values in (
        ("onset", rupture.onset_s),
        ("rise_time", rupture.rise_time_s),
        ("rake", rupture.rake_deg),
    ):
        low, high = float(values.min()), float(values.max())
        rr.log(f"/distribution/{key}", histogram(values, low, high), static=True)
        if key != "rake":
            mesh(f"/fault/{key}", colours(values, low, high))

    reach = float(np.ptp(positions, axis=0).max())
    lines, anchors, labels = onset_contours(rupture, corners, 0.01 * reach)
    if labels:
        rr.log(
            "/fault/onset/isochrones",
            rr.LineStrips3D(lines, colors=[CONTOUR_LINE], radii=[UI_POINTS]),
            static=True,
        )
        rr.log(
            "/fault/onset/isochrones/labels",
            rr.Points3D(
                anchors,
                colors=[CONTOUR_LINE],
                labels=labels,
                show_labels=True,
                radii=[UI_POINTS],
            ),
            static=True,
        )

    # Rake as arrows along each point's slip, thinned, sized and coloured by slip on
    # the same map the slip view uses.
    peak = float(rupture.slip_m.max())
    shown = np.arange(0, count, -(-count // MAX_ARROWS))
    cell_m = np.linalg.norm(corners[shown, 1] - corners[shown, 0], axis=-1)
    slip = rupture.slip_m[shown]
    rr.log(
        "/fault/rake",
        rr.Arrows3D(
            origins=corners[shown].mean(axis=1),
            vectors=slip_directions(rupture, corners)[shown]
            * (6.0 * cell_m * slip / (peak or 1.0))[:, None],
            colors=colours(slip, 0.0, peak, HOT),
        ),
        static=True,
    )
    if hypocentre is not None:
        rr.log(
            "/fault/hypocentre",
            rr.Points3D([hypocentre], radii=[0.005 * reach]),
            static=True,
        )

    # Only the shape is static: Rerun lets a static component shadow every temporal
    # one on the same entity, so static colours would freeze the slip at zero.
    mesh("/fault/slip", None)
    finish_s = rupture.onset_s + np.diff(rupture.pulse_offsets) * rupture.dt_s
    times_s = np.arange(rupture.onset_s.min(), finish_s.max() + step_s, step_s)
    moment_per_slip = rupture.rigidity_pa * rupture.area_m2
    cumulative = np.empty(times_s.size)
    for frame, slipped in enumerate(slip_by(rupture, times_s)):
        rr.set_time("rupture", duration=float(times_s[frame]))
        packed = colours(slipped, 0.0, peak, HOT)
        rr.log("/fault/slip", rr.Mesh3D.from_fields(vertex_colors=np.repeat(packed, 4)))
        # Points at rest would swamp the first bin, so the histogram is of those moving.
        rr.log("/distribution/slip", histogram(slipped[slipped > 0], 0.0, peak))
        cumulative[frame] = moment_per_slip @ slipped

    timeline = [rr.TimeColumn("rupture", duration=times_s)]
    rate = np.diff(cumulative, prepend=0.0) / step_s
    rr.send_columns(
        "/moment/rate", indexes=timeline, columns=rr.Scalars.columns(scalars=rate)
    )
    rr.send_columns(
        "/moment/cumulative",
        indexes=timeline,
        columns=rr.Scalars.columns(scalars=cumulative),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the viewer, returning the exit status."""
    parser = argparse.ArgumentParser(
        prog="rupture-view", description="Watch a rupture from an SRF, in Rerun."
    )
    parser.add_argument("srf", type=Path, help="a version 2 SRF")
    parser.add_argument(
        "--step", type=float, default=0.5, help="seconds between frames (default 0.5)"
    )
    parser.add_argument(
        "--save", type=Path, help="write a .rrd recording instead of opening a window"
    )
    args = parser.parse_args(argv)
    try:
        import rerun as rr
    except ImportError:
        print(
            "rupture-view: needs rerun-sdk; install rupture-generator[vis]",
            file=sys.stderr,
        )
        return 2
    try:
        rupture = read_rupture(args.srf)
    except (RuptureGeneratorError, OSError) as error:
        print(f"rupture-view: {error}", file=sys.stderr)
        return 2

    rr.init("rupture-view", spawn=args.save is None)
    if args.save is not None:
        rr.save(args.save)
    rr.send_blueprint(layout())
    log(rupture, args.step)
    return 0


if __name__ == "__main__":
    sys.exit(main())
