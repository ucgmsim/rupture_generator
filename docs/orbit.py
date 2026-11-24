"""Draw the README's turntable: slip on a drawn rupture, the camera circling it.

Slip is in :mod:`hero`'s bands, the heavy black lines are the rupture front every
:data:`CONTOUR_S` seconds, and depth is stretched :data:`DEPTH_EXAGGERATION` times so a
shallow-dipping interface shows its shape. Only the fault's own cells are drawn: a
curved surface's chart leaves the rest of its rectangle unoccupied. An optional
coastline shapefile, such as Natural Earth's (public domain,
https://www.naturalearthdata.com), is drawn at sea level within
:data:`COAST_MARGIN_KM` of the fault. ``just orbit`` draws ``examples/maule.toml``
and runs this on its fields file::

    uv run --with pyvista --with matplotlib --with imageio --with pyshp \\
        python docs/orbit.py rupture.h5 out.gif [coastline.zip]
"""

import sys

import numpy as np
import pyproj
import pyvista as pv
import shapefile
from hero import _slip_bands
from vtkmodules.vtkRenderingCore import vtkMapper

from rupture_generator import read_fields

COAST_MARGIN_KM = 150.0
"""How far past the fault's plan extent the coastline runs."""
COAST_WIDTH_PX = 3.0
"""How heavy the coastline is."""
CONTOUR_S = 10.0
"""The rupture time between contours, in seconds."""
CONTOUR_WIDTH_PX = 4.0
"""How heavy the contours are."""
DEPTH_EXAGGERATION = 3.0
"""How many times depth is stretched against the horizontal."""
ELEVATION_DEG = 45.0
"""How far above the horizon the camera looks down."""
FRAMES = 180
"""How many frames make one turn."""
FPS = 12
"""The turntable's frame rate: a turn takes ``FRAMES / FPS`` seconds."""
WINDOW_PX = (960, 640)
"""The animation's size."""


def _faults(path: str) -> tuple[list[pv.UnstructuredGrid], np.ndarray, pyproj.CRS]:
    """Each segment's occupied cells, with slip and onset, centred and stretched.

    Also the centre taken off, in the projected CRS's kilometres, and that CRS.
    """
    realisation = read_fields(path)
    meshes = []
    for rupture in realisation.values():
        geometry = rupture.geometry
        nodes = geometry.nodes
        grid = pv.StructuredGrid(
            nodes[..., 0], nodes[..., 1], -DEPTH_EXAGGERATION * nodes[..., 2]
        )
        # A structured grid numbers its cells with the first index varying first.
        grid.cell_data["slip"] = rupture.slip_m.ravel(order="F")
        grid.cell_data["onset"] = rupture.onset_s.ravel(order="F")
        meshes.append(
            grid.extract_cells(np.flatnonzero(geometry.occupied.ravel(order="F")))
        )
    points = np.concatenate([mesh.points for mesh in meshes])
    centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
    for mesh in meshes:
        mesh.points -= centre
    return meshes, centre, realisation.crs


def _coast(
    path: str, crs: pyproj.CRS, centre: np.ndarray, meshes: list[pv.UnstructuredGrid]
) -> pv.PolyData:
    """The coastline near the fault at sea level, in the meshes' centred frame."""
    points = np.concatenate([mesh.points for mesh in meshes])[:, :2]
    low = points.min(axis=0) - COAST_MARGIN_KM
    high = points.max(axis=0) + COAST_MARGIN_KM
    to_crs = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lines = pv.PolyData()
    for shape in shapefile.Reader(path).shapes():
        xy = np.asarray(shape.points, dtype=float)
        if not len(xy):
            continue
        for part in np.split(xy, shape.parts[1:]):
            east, north = to_crs.transform(part[:, 0], part[:, 1])
            plan = np.column_stack([east, north]) / 1000.0 - centre[:2]
            inside = np.all((plan >= low) & (plan <= high), axis=1)
            # Each run of points inside the box is one line.
            breaks = np.flatnonzero(np.diff(inside.astype(int))) + 1
            for run, keep in zip(
                np.split(plan, breaks), np.split(inside, breaks), strict=True
            ):
                if keep[0] and len(run) > 1:
                    line = np.column_stack([run, np.full(len(run), -centre[2])])
                    lines += pv.lines_from_points(line)
    return lines


def main(source: str, output: str, coast: str | None = None) -> None:
    """Draw the fields file ``source`` turning, and save the animation as ``output``.

    Parameters
    ----------
    source : str
        A fields file, as ``rupture-generator`` writes for an ``.h5`` output.
    output : str
        Where to save the GIF.
    coast : str, optional
        A coastline shapefile in longitude and latitude, or a zip holding one.
    """
    meshes, centre, crs = _faults(source)
    slips = np.concatenate([mesh.cell_data["slip"] for mesh in meshes])
    onsets = np.concatenate([mesh.cell_data["onset"] for mesh in meshes])
    levels = np.arange(CONTOUR_S, onsets.max(), CONTOUR_S)
    cmap, clim = _slip_bands(slips)

    # Lines lying on a face would flicker in and out of it without an offset.
    vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_PX)
    plotter.set_background("white")
    for number, mesh in enumerate(meshes):
        plotter.add_mesh(
            mesh,
            scalars="slip",
            cmap=cmap,
            clim=clim,
            above_color=tuple(cmap.get_over()),
            ambient=0.55,
            diffuse=0.5,
            specular=0.0,
            show_scalar_bar=number == 0,
            scalar_bar_args={
                "title": "Slip (m)",
                "color": "black",
                "vertical": True,
                "position_x": 0.86,
                "position_y": 0.2,
                "width": 0.07,
                "height": 0.6,
                "n_labels": cmap.N + 1,
                "fmt": "%.0f",
                "above_label": "",
                "title_font_size": 18,
                "label_font_size": 16,
            },
        )
        contours = mesh.cell_data_to_point_data().contour(levels, scalars="onset")
        if contours.n_points:
            plotter.add_mesh(
                contours, color="black", line_width=CONTOUR_WIDTH_PX, lighting=False
            )

    if coast is not None:
        plotter.add_mesh(
            _coast(coast, crs, centre, meshes),
            color="dimgray",
            line_width=COAST_WIDTH_PX,
            lighting=False,
        )
    elevation = np.radians(ELEVATION_DEG)
    reach = float(
        np.ptp(np.concatenate([mesh.points for mesh in meshes]), axis=0).max()
    )
    plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.open_gif(output, fps=FPS)
    for azimuth in np.linspace(0.0, 2.0 * np.pi, FRAMES, endpoint=False):
        # From the south first, then round anticlockwise seen from above.
        # The meshes are centred on their bounding box, which the camera circles.
        # Only the direction matters: reset_camera backs off to fit the fault.
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.camera.position = tuple(
            10.0
            * reach
            * np.array(
                [
                    np.cos(elevation) * np.sin(azimuth),
                    -np.cos(elevation) * np.cos(azimuth),
                    np.sin(elevation),
                ]
            )
        )
        plotter.reset_camera()
        plotter.camera.zoom(1.15)
        plotter.write_frame()
    plotter.close()


if __name__ == "__main__":
    main(*sys.argv[1:4])
