"""``rupture-view RUPTURE.srf``: watch a rupture happen, in Rerun.

Reads the version 2 SRF ``rupture-generator`` writes, through
:func:`~rupture_generator.formats.srf.read_rupture`. Each point is drawn as its own
quadrilateral, sized from its plane's cells and turned by its own strike and dip, in
metres east, north and **up** from the rupture's centre -- up because a viewer's
vertical axis points up.

Slip is shown accumulating, each point's pulse integrated from its own onset; onset,
rise time and rake are shown whole. Moment is counted from the same integration.
"""

import argparse
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pyproj

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.formats.srf import M_PER_KM, SrfRupture, read_rupture
from rupture_generator.rupture.source import magnitude_from_moment


def _viridis() -> np.ndarray:
    stops = np.array(
        [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]]
    )
    shade = np.linspace(0.0, 1.0, 256)
    r, g, b = (
        np.interp(shade, np.linspace(0.0, 1.0, len(stops)), stops[:, k]).astype(
            np.uint32
        )
        for k in range(3)
    )
    return (r << 24) | (g << 16) | (b << 8) | 0xFF


VIRIDIS = _viridis()
"""Viridis in 256 shades from five stops, packed RGBA as Rerun takes colours."""

TABS = {"slip": "slip", "onset": "onset", "rise_time": "rise time", "rake": "rake"}


def colours(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """A packed viridis colour for each value between ``low`` and ``high``."""
    shade = (values - low) * (255.0 / ((high - low) or 1.0))
    return VIRIDIS[np.clip(shade, 0.0, 255.0).astype(np.uint8)]


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
    strike, dip = np.radians(rupture.strike_deg), np.radians(rupture.dip_deg)
    along = np.stack([np.sin(strike), np.cos(strike), np.zeros_like(strike)], axis=-1)
    down = np.stack(
        [np.cos(strike) * np.cos(dip), -np.sin(strike) * np.cos(dip), -np.sin(dip)],
        axis=-1,
    )
    along, down = along * half_length, down * half_width
    centres = local(rupture.lon_deg, rupture.lat_deg, rupture.depth_km)
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
                    rrb.TimeSeriesView(origin="/moment/rate", name="moment rate"),
                    rrb.TimeSeriesView(
                        origin="/moment/cumulative", name="cumulative moment"
                    ),
                ),
                row_shares=[1, 2],
            ),
            rrb.Tabs(
                *(
                    rrb.Spatial3DView(
                        origin="/fault",
                        name=name,
                        contents=[f"/fault/{key}", "/fault/hypocentre"],
                    )
                    for key, name in TABS.items()
                )
            ),
            column_shares=[1, 3],
        ),
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

    def mesh(path: str, packed: np.ndarray) -> None:
        rr.log(
            path,
            rr.Mesh3D(
                vertex_positions=positions,
                triangle_indices=triangles,
                vertex_colors=np.repeat(packed, 4),
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
        mesh(f"/fault/{key}", colours(values, values.min(), values.max()))
    if hypocentre is not None:
        reach = float(np.ptp(positions, axis=0).max())
        rr.log(
            "/fault/hypocentre",
            rr.Points3D([hypocentre], radii=[0.005 * reach]),
            static=True,
        )

    peak = float(rupture.slip_m.max())
    mesh("/fault/slip", colours(np.zeros(count), 0.0, peak))
    finish_s = rupture.onset_s + np.diff(rupture.pulse_offsets) * rupture.dt_s
    times_s = np.arange(rupture.onset_s.min(), finish_s.max() + step_s, step_s)
    moment_per_slip = rupture.rigidity_pa * rupture.area_m2
    cumulative = np.empty(times_s.size)
    for frame, slipped in enumerate(slip_by(rupture, times_s)):
        rr.set_time("rupture", duration=float(times_s[frame]))
        packed = colours(slipped, 0.0, peak)
        rr.log("/fault/slip", rr.Mesh3D.from_fields(vertex_colors=np.repeat(packed, 4)))
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
