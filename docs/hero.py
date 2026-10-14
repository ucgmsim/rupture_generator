"""Draw the README's picture: slip on a drawn rupture, with its rupture-time contours.

The picture bands slip on ``magma_r``, darker for more, and black lines join the
subfaults that start together. The faults stand at true scale, seen from the south.
``just hero`` draws ``examples/beavan.toml`` and runs this on its fields file::

    uv run --with pyvista python docs/hero.py rupture.h5 docs/images/beavan_slip.png
"""

import sys

import numpy as np
import pyvista as pv
from PIL import Image
from vtkmodules.vtkRenderingCore import vtkMapper

from rupture_generator import read_fields

BANDS = 10
"""The number of slip bands."""
CONTOUR_S = 2.0
"""The rupture time between contours, in seconds."""
ELEVATION_DEG = 20.0
"""How far above the horizon the camera looks down from the south."""
MARGIN_PX = 30
"""The white border left around the faults."""
MINIMUM_LABEL_POINTS = 20
"""The fewest points a contour needs on a plane to carry its label there."""


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


def _cropped(image: np.ndarray) -> Image.Image:
    """The screenshot cut down to the faults and a margin of white."""
    ink = np.nonzero((image < 250).any(axis=2))
    top, left = (max(int(i.min()) - MARGIN_PX, 0) for i in ink)
    bottom, right = (int(i.max()) + MARGIN_PX for i in ink)
    return Image.fromarray(image[top:bottom, left:right])


def _label_places(
    contours: list[pv.PolyData], levels: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """One place per rupture time to write it, on the nearest plane that has it.

    The camera looks north, so the plane with the least northing among those the
    contour crosses is the one in front. The label sits midway along it there.
    """
    places, labels = [], []
    for level in levels:
        crossings = [
            on
            for lines in contours
            if len(on := lines.points[np.isclose(lines.point_data["onset"], level)])
            >= MINIMUM_LABEL_POINTS
        ]
        if not crossings:
            continue
        front = min(crossings, key=lambda on: on[:, 1].mean())
        middle = front[np.argmin(np.linalg.norm(front - front.mean(axis=0), axis=1))]
        places.append(middle)
        labels.append(f"{level:g} s")
    return np.array(places), labels


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

    # Lines lying on a face would flicker in and out of it without an offset.
    vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
    plotter = pv.Plotter(off_screen=True, window_size=(2400, 1100))
    plotter.set_background("white")
    drawn = []
    for grid in grids:
        plotter.add_mesh(
            grid,
            scalars="slip",
            cmap="magma_r",
            n_colors=BANDS,
            clim=(0.0, float(np.quantile(slips, 0.99))),
            lighting=False,
            show_scalar_bar=False,
        )
        contours = grid.cell_data_to_point_data().contour(levels, scalars="onset")
        if contours.n_points:
            plotter.add_mesh(contours, color="black", line_width=4.0, lighting=False)
            drawn.append(contours)
    places, labels = _label_places(drawn, levels)
    plotter.add_point_labels(
        places,
        labels,
        font_size=28,
        text_color="black",
        shape_color="white",
        shape_opacity=0.85,
        margin=4,
        show_points=False,
        always_visible=False,
    )
    plotter.view_xz()
    plotter.camera.elevation = ELEVATION_DEG
    plotter.reset_camera()
    plotter.camera.zoom(1.8)
    _cropped(plotter.screenshot(return_img=True)).save(output, optimize=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
