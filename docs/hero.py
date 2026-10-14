"""Draw the README's picture: slip on a drawn rupture, with its rupture-time contours.

Slip is in bands of a reversed ``hot`` ramp, pale yellow through red to black, and
heavy black lines join the subfaults that start together, each labelled with its
time in a white box along the line. The faults stand at true scale, seen from the
south. ``just hero`` draws ``examples/beavan.toml`` and runs this on its fields file::

    uv run --with pyvista --with matplotlib python docs/hero.py rupture.h5 out.png
"""

import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from PIL import Image
from vtkmodules.vtkRenderingCore import vtkMapper

from rupture_generator import read_fields

BANDS = 8
"""About how many slip bands to draw."""
RAMP_START = 0.08
"""Where on ``hot_r`` the lowest band sits: pale yellow rather than white."""
CONTOUR_S = 5.0
"""The rupture time between contours, in seconds."""
CONTOUR_WIDTH_PX = 7.0
"""How heavy the contours are."""
LABEL_FONT_PT = 19
"""The size of a contour's label."""
LABEL_REACH_KM = 1.5
"""How far along its contour a label looks to decide which way the line runs."""
MINIMUM_LABEL_POINTS = 20
"""The fewest points a contour needs on a plane to carry its label there."""
ELEVATION_DEG = 20.0
"""How far above the horizon the camera looks down from the south."""
WINDOW_PX = (2400, 1100)
"""The render's size before cropping."""
MARGIN_PX = 30
"""The white border left around the faults."""


def _planes(path: str) -> list[pv.StructuredGrid]:
    """Each plane of each segment as a grid, with slip and onset on its cells."""
    grids = []
    for rupture in read_fields(path).values():
        geometry = rupture.geometry
        edges = np.cumsum([0, *geometry.plane_cells])
        for nodes, start, stop in zip(
            geometry.plane_nodes(), edges[:-1], edges[1:], strict=True
        ):
            grid = pv.StructuredGrid(nodes[..., 0], nodes[..., 1], -nodes[..., 2])
            # A structured grid numbers its cells with the first index varying first.
            grid.cell_data["slip"] = rupture.slip_m[:, start:stop].ravel(order="F")
            grid.cell_data["onset"] = rupture.onset_s[:, start:stop].ravel(order="F")
            grids.append(grid)
    return grids


def _slip_bands(slips: np.ndarray) -> tuple[ListedColormap, tuple[float, float]]:
    """The banded ramp and the slip range its bands cover.

    The bands cover zero to the 0.97 quantile on round boundaries, so the darker
    reds go to slip a few times the median. Slip beyond the last band takes the
    ramp's black end.
    """
    levels = MaxNLocator(BANDS).tick_values(0.0, float(np.quantile(slips, 0.97)))
    ramp = plt.get_cmap("hot_r")
    cmap = ListedColormap(ramp(np.linspace(RAMP_START, 0.95, len(levels) - 1)))
    cmap.set_over(ramp(1.0))
    return cmap, (float(levels[0]), float(levels[-1]))


def _label_places(
    contours: list[pv.PolyData], levels: np.ndarray
) -> list[tuple[float, np.ndarray]]:
    """Each rupture time, and the points of its contour on the nearest plane.

    The camera looks north, so the plane with the least northing among those the
    contour crosses is the one in front.
    """
    places = []
    for level in levels:
        crossings = [
            on
            for lines in contours
            if len(on := lines.points[np.isclose(lines.point_data["onset"], level)])
            >= MINIMUM_LABEL_POINTS
        ]
        if crossings:
            places.append((level, min(crossings, key=lambda on: on[:, 1].mean())))
    return places


def _on_screen(plotter: pv.Plotter, points: np.ndarray) -> np.ndarray:
    """Points' pixel columns, rows from the top, and depths in the z-buffer's range."""
    renderer = plotter.renderer
    shown = []
    for x, y, z in points:
        renderer.SetWorldPoint(x, y, z, 1.0)
        renderer.WorldToDisplay()
        shown.append(renderer.GetDisplayPoint())
    shown = np.array(shown)
    shown[:, 1] = plotter.window_size[1] - shown[:, 1]
    return shown


def _labels(
    plotter: pv.Plotter, places: list[tuple[float, np.ndarray]]
) -> list[tuple[str, float, float, float]]:
    """Each visible label's text, pixel column and row, and angle along its line.

    A label goes at the point of its contour nearest the contour's middle, turned to
    run along the line. A fault in front hiding that point drops the label.
    """
    labels = []
    for level, points in places:
        middle = points[np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1))]
        near = points[np.linalg.norm(points - middle, axis=1) < LABEL_REACH_KM]
        column, row, depth = _on_screen(plotter, middle[None])[0]
        shown = plotter.ren_win.GetZbufferDataAtPoint(
            int(column), int(plotter.window_size[1] - row)
        )
        if depth > shown + 1e-4:
            continue
        along = _on_screen(plotter, near)[:, :2]
        along -= along.mean(axis=0)
        direction = np.linalg.svd(along, full_matrices=False)[2][0]
        # Screen rows run down, which flips the angle's sign. Text stays upright.
        angle = float(np.degrees(np.arctan2(-direction[1], direction[0])))
        angle = (angle + 90.0) % 180.0 - 90.0
        labels.append((f"{level:g} s", float(column), float(row), angle))
    return labels


def _annotated(image: np.ndarray, labels: list[tuple[str, float, float, float]]):
    """The render with the contour labels drawn over it, as an RGB array."""
    height, width = image.shape[:2]
    dpi = 100
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.imshow(image)
    axes.set_axis_off()
    for text, column, row, angle in labels:
        axes.text(
            column,
            row,
            text,
            rotation=angle,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=LABEL_FONT_PT,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.35",
                "linewidth": 1.2,
            },
        )
    figure.canvas.draw()
    annotated = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return annotated


def _cropped(image: np.ndarray) -> Image.Image:
    """The picture cut down to the faults and a margin of white."""
    ink = np.nonzero((image < 250).any(axis=2))
    top, left = (max(int(i.min()) - MARGIN_PX, 0) for i in ink)
    bottom, right = (int(i.max()) + MARGIN_PX for i in ink)
    return Image.fromarray(image[top:bottom, left:right])


def main(source: str, output: str) -> None:
    """Draw the fields file ``source`` and save the picture as ``output``.

    Parameters
    ----------
    source : str
        A fields file, as ``rupture-generator`` writes for an ``.h5`` output.
    output : str
        Where to save the PNG.
    """
    grids = _planes(source)
    slips = np.concatenate([grid.cell_data["slip"] for grid in grids])
    onsets = np.concatenate([grid.cell_data["onset"] for grid in grids])
    levels = np.arange(CONTOUR_S, onsets.max(), CONTOUR_S)
    cmap, clim = _slip_bands(slips)

    # Lines lying on a face would flicker in and out of it without an offset.
    vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_PX)
    plotter.set_background("white")
    drawn = []
    for grid in grids:
        plotter.add_mesh(
            grid,
            scalars="slip",
            cmap=cmap,
            clim=clim,
            above_color=tuple(cmap.get_over()),
            lighting=False,
            show_scalar_bar=False,
        )
        contours = grid.cell_data_to_point_data().contour(levels, scalars="onset")
        if contours.n_points:
            plotter.add_mesh(
                contours, color="black", line_width=CONTOUR_WIDTH_PX, lighting=False
            )
            drawn.append(contours)
    plotter.view_xz()
    plotter.camera.elevation = ELEVATION_DEG
    plotter.reset_camera()
    plotter.camera.zoom(1.8)
    image = plotter.screenshot(return_img=True)
    labels = _labels(plotter, _label_places(drawn, levels))
    _cropped(_annotated(image, labels)).save(output, optimize=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
