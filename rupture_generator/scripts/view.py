"""``rupture-generator view``: watch a rupture happen.

Opens a Rerun viewer showing the fault in three dimensions, coloured by a field, with a
time scrub that plays the rupture at real speed -- and, beside it, the moment rate
function and a histogram of the field, both moving with the cursor.

# Why Rerun rather than a plot

The thing worth looking at is *propagation*, and a still image cannot show it. A viewer
that can is a time-aware one, and Rerun's timeline gives play, pause, loop and a rate
multiplier for free, synchronised across every panel, provided the time is logged as a
**duration in seconds** rather than a frame number. That is the whole reason this is a
few hundred lines instead of a few thousand.

# The render mesh is not `RefinedMesh::triangles`

That returns the topologically honest mesh, with vertices shared between neighbouring
cells. Rerun colours by *vertex*, so a shared one would interpolate a piecewise-constant
field and draw values that were never computed -- a slip of 0.4 m blended into its
neighbour's 3 m across the seam between them.

So this duplicates: four vertices per cell, one flat colour. That is a display list
rather than a mesh, and the difference matters exactly here.

# What animates

`slip` is **cumulative slip at t**, integrated from each subfault's own pulse, and it is
the propagation. The other fields -- rise time, onset, rake -- are properties of the
finished rupture rather than of a moment in it, so they are drawn once and the cursor
drives the moment rate and the rupture front instead. `--help` says so, because "scrub a
static field" is a reasonable thing to expect and is not what happens.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer

from rupture_generator.formats.rupture import read_rupture, segments_in
from rupture_generator.moment import moment_rate, rigidity_pa
from rupture_generator.scripts.errors import console

if TYPE_CHECKING:
    import xarray as xr

FIELDS = {
    "slip": ("slip_m", "metres"),
    "rise-time": ("rise_time_s", "seconds"),
    "onset": ("onset_s", "seconds"),
    "rake": ("rake_deg", "degrees"),
}
"""Which variable each `--field` shows, and what it is measured in."""

ANIMATED = "slip"
"""The only field that is a function of time. See the module note."""


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


# A 16-entry viridis, linearly interpolated to 256. Vendored rather than importing
# matplotlib for a lookup table: it is the only thing that would have been used from it,
# and it is a heavy dependency to carry into a viewer that already has one.
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


def viridis(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Colours for `values`, scaled between `low` and `high`.

    Returns
    -------
    np.ndarray
        `(n, 3)` of uint8.
    """
    span = high - low
    fraction = (
        np.zeros_like(values, dtype=np.float64)
        if span <= 0.0
        else (np.clip((values - low) / span, 0.0, 1.0))
    )
    position = fraction * (len(_VIRIDIS_16) - 1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(_VIRIDIS_16) - 1)
    weight = (position - lower)[:, None]
    blended = _VIRIDIS_16[lower] * (1.0 - weight) + _VIRIDIS_16[upper] * weight
    return blended.round().astype(np.uint8)


def quads(plane: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Four vertices and two triangles per cell, in a local east-north-up frame.

    Unshared on purpose -- see the module note. Positions are metres from the mesh
    origin, with **up positive**, because a viewer's vertical axis points up and the
    file's depth points down.

    Returns
    -------
    tuple
        Vertex positions `(4 * cells, 3)` and triangle indices `(2 * cells, 3)`.
    """
    east = plane["node_east_km"].to_numpy() * 1000.0
    north = plane["node_north_km"].to_numpy() * 1000.0
    up = -plane["node_depth_km"].to_numpy() * 1000.0
    dip_count, strike_count = (dimension - 1 for dimension in east.shape)

    corners = [(0, 0), (0, 1), (1, 1), (1, 0)]
    vertices = np.empty((dip_count * strike_count * 4, 3), dtype=np.float32)
    for index, (down, along) in enumerate(corners):
        rows = slice(down, down + dip_count)
        columns = slice(along, along + strike_count)
        block = np.stack(
            [east[rows, columns], north[rows, columns], up[rows, columns]], axis=-1
        )
        vertices[index::4] = block.reshape(-1, 3)

    base = np.arange(dip_count * strike_count) * 4
    triangles = np.empty((dip_count * strike_count * 2, 3), dtype=np.uint32)
    triangles[0::2] = np.stack([base, base + 1, base + 2], axis=-1)
    triangles[1::2] = np.stack([base, base + 2, base + 3], axis=-1)
    return vertices, triangles


def cumulative_slip(plane: xr.Dataset, times_s: np.ndarray) -> np.ndarray:
    """How much each subfault has slipped by each time, in metres.

    Each subfault's pulse integrated up to `t`, placed at its own onset. This is what
    makes the animation a rupture rather than a slide show: at any moment, the part of
    the fault that has moved is exactly the part the front has reached.

    Returns
    -------
    np.ndarray
        `(len(times_s), cells)`.
    """
    offsets = plane["slip_rate_offset"].to_numpy()
    samples = plane["slip_rate"].to_numpy()
    onset_s = plane["onset_s"].to_numpy().ravel()
    interval_s = float(plane.attrs["sample_interval_s"])
    cells = onset_s.size

    slipped = np.zeros((len(times_s), cells), dtype=np.float64)
    for cell in range(cells):
        start, stop = int(offsets[cell]), int(offsets[cell + 1])
        if stop == start:
            continue
        running = np.cumsum(samples[start:stop]) * interval_s
        # Where each display time falls in this pulse, in samples from its onset.
        into = np.floor((times_s - onset_s[cell]) / interval_s).astype(np.int64)
        slipped[:, cell] = np.where(
            into < 0, 0.0, running[np.clip(into, 0, len(running) - 1)]
        )
    return slipped


def view(
    rupture: Annotated[
        Path,
        typer.Argument(
            help="Rupture file from `rupture-generator generate`.",
            exists=True,
            readable=True,
        ),
    ],
    field: Annotated[
        str,
        typer.Option(
            help=(
                "Which field to colour by. Only `slip` animates -- the others are "
                "properties of the finished rupture, so they are drawn once while the "
                "cursor drives the moment rate and the rupture front."
            )
        ),
    ] = ANIMATED,
    time_step: Annotated[
        float, typer.Option(help="Seconds between animation frames.")
    ] = 0.05,
    bins: Annotated[int, typer.Option(help="Histogram bins.")] = 40,
    save: Annotated[
        Path | None,
        typer.Option(help="Write a .rrd recording instead of opening a window."),
    ] = None,
) -> None:
    """Show a rupture propagating, with its moment rate and slip distribution."""
    if field not in FIELDS:
        console.print(
            f"[red]no field {field!r}; choose one of {', '.join(FIELDS)}[/red]"
        )
        raise typer.Exit(1)

    try:
        rerun, blueprint = require_rerun()
    except ImportError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    with read_rupture(rupture) as tree:
        planes = segments_in(tree)
        if not planes:
            console.print(f"[red]{rupture} holds no rupture[/red]")
            raise typer.Exit(1)

        rerun.init("rupture-generator", spawn=save is None)
        if save is not None:
            rerun.save(save)

        rerun.send_blueprint(layout(blueprint, field))
        log_rupture(rerun, planes, field, time_step, bins)

    if save is not None:
        console.print(f"[green]wrote[/green] {save}")


def layout(blueprint, field: str):  # noqa: ANN001 - Rerun's types
    """The panels, side by side: the fault, and two views of the same numbers.

    The 3D view is given three quarters of the width because it is the one thing a still
    image cannot replace. The other two are stacked beside it and share the time cursor,
    which Rerun draws across every time-series view without being asked.
    """
    return blueprint.Blueprint(
        blueprint.Horizontal(
            blueprint.Vertical(
                blueprint.BarChartView(
                    origin="/histogram", name=f"{field} distribution"
                ),
                blueprint.TimeSeriesView(origin="/moment_rate", name="moment rate"),
            ),
            blueprint.Spatial3DView(origin="/fault", name="fault"),
            column_shares=[1, 3],
        ),
        collapse_panels=True,
    )


def log_rupture(
    rerun,  # noqa: ANN001 - Rerun's module
    planes: list[tuple[str, int, xr.Dataset]],
    field: str,
    time_step: float,
    bins: int,
) -> None:
    """Log every panel, over the rupture's own timeline."""
    variable, unit = FIELDS[field]
    animated = field == ANIMATED
    rerun.log(
        "/histogram",
        rerun.TextDocument(f"{field}, in {unit}"),
        static=True,
    )

    geometry = {
        f"{surface}/plane_{index}": quads(plane) for surface, index, plane in planes
    }
    values = {
        f"{surface}/plane_{index}": plane[variable].to_numpy().ravel()
        for surface, index, plane in planes
    }
    low = min(float(v.min()) for v in values.values())
    high = max(float(v.max()) for v in values.values())

    # The whole rupture, on one clock: the earliest onset to the last pulse's end.
    starts = [float(plane["onset_s"].min()) for _, _, plane in planes]
    ends = [
        float(plane["onset_s"].max() + plane["rise_time_s"].max())
        for _, _, plane in planes
    ]
    times_s = np.arange(min(starts), max(ends) + time_step, time_step)

    rerun.log(
        "/fault",
        rerun.AnnotationContext([]),
        static=True,
    )

    if not animated:
        # Static fields: draw once, and let the cursor drive the front instead.
        for path, (vertices, triangles) in geometry.items():
            rerun.log(
                f"/fault/{path}",
                mesh(rerun, vertices, triangles, values[path], low, high),
                static=True,
            )
        rerun.log(
            "/histogram",
            rerun.BarChart(
                np.histogram(
                    np.concatenate(list(values.values())), bins=bins, range=(low, high)
                )[0]
            ),
            static=True,
        )

    slipped = {
        f"{surface}/plane_{index}": cumulative_slip(plane, times_s)
        for surface, index, plane in planes
    }
    rate = moment_rate_of(planes, times_s)

    for step, moment_s in enumerate(times_s):
        rerun.set_time("rupture", duration=float(moment_s))
        rerun.log("/moment_rate", rerun.Scalars(float(rate[step])))

        if not animated:
            continue

        frame = []
        for path, (vertices, triangles) in geometry.items():
            current = slipped[path][step]
            rerun.log(
                f"/fault/{path}", mesh(rerun, vertices, triangles, current, low, high)
            )
            frame.append(current)
        rerun.log(
            "/histogram",
            rerun.BarChart(
                np.histogram(np.concatenate(frame), bins=bins, range=(low, high))[0]
            ),
        )


def mesh(rerun, vertices, triangles, values, low, high):  # noqa: ANN001
    """A `Mesh3D` whose cells are flat-coloured.

    Each value is repeated across its cell's four vertices, which is what makes the
    colour piecewise constant -- see the module note on why the vertices are not shared.
    """
    return rerun.Mesh3D(
        vertex_positions=vertices,
        triangle_indices=triangles,
        vertex_colors=np.repeat(viridis(values, low, high), 4, axis=0),
    )


def moment_rate_of(
    planes: list[tuple[str, int, xr.Dataset]], times_s: np.ndarray
) -> np.ndarray:
    """The whole rupture's moment rate, sampled at the display times.

    Summed across planes, because a bent fault is one earthquake. Rigidity comes from
    the shear speed and density the generator sampled, which the file does not carry --
    so this uses the slip and area it does, with a constant rigidity, and the curve's
    *shape* is what the panel is for. The absolute scale is on the axis and in the
    file's `moment_newton_m`.
    """
    total = np.zeros_like(times_s)
    for _, _, plane in planes:
        area_m2 = plane["area_m2"].to_numpy().ravel()
        # 30 GPa, which is crustal rock.
        rigidity = np.full_like(
            area_m2, float(rigidity_pa(np.array([3.2]), np.array([2.6]))[0])
        )

        plane_times, plane_rate = moment_rate(
            plane["slip_rate_offset"].to_numpy(),
            plane["slip_rate"].to_numpy(),
            plane["onset_s"].to_numpy().ravel(),
            area_m2,
            rigidity,
            sample_interval_s=float(plane.attrs["sample_interval_s"]),
            duration_s=float(times_s[-1] - times_s[0]),
        )
        total += np.interp(times_s, plane_times, plane_rate, left=0.0, right=0.0)
    return total


__all__ = ["FIELDS", "cumulative_slip", "quads", "view", "viridis"]
