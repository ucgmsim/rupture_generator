#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["h5py", "numpy", "pyproj", "shapely", "xarray", "h5netcdf"]
# ///
"""Build the 2010 Maule interface as a GOCAD TSurf from Slab2, cut to Hayes's outline.

The surface is the Slab2 South America depth grid:

    Hayes, G. P. et al. (2018). Slab2, a comprehensive subduction zone geometry model.
    Science 362(6410), 58-61. https://doi.org/10.1126/science.aat4723
    Data: https://doi.org/10.5066/F7PV6JNV (sam_slab2_dep_02.23.18.grd)

cut to the map-view outline of the five planes of the SRCMOD model s2010MAULEC02HAYE
(Hayes 2017), which steps its dip from 10 to 23.5 degrees down the interface and its
strike from 18.7 to 8.7 degrees along it, approximating the same slab in planes. The
outline is the union of the planes' footprints, its gaps closed by a 15 km
dilate-erode, so Slab2 sets the shape and Hayes the extent.

Run it as::

    examples/maule_surface.py s2010MAULEC02HAYE.h5 sam_slab2_dep_02.23.18.grd examples/slab2/Maule.ts.gz
"""

import gzip
import sys
from pathlib import Path

import h5py
import numpy as np
import shapely
import xarray as xr
from pyproj import Transformer

CRS = "EPSG:32718"
"""UTM zone 18S, which holds most of the rupture."""

CLOSE_M = 15_000.0
"""The dilate-erode radius that closes the gaps between Hayes's planes."""

TO_UTM = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)


def footprint(lon: np.ndarray, lat: np.ndarray) -> shapely.Polygon:
    """The map footprint of one plane, from its subfault centres.

    The centres sit half a cell in from each edge, so the corners step half a cell
    out along both grid directions from the corner centres.
    """
    x, y = TO_UTM.transform(lon, lat)
    p = np.stack([x, y], axis=-1)
    d_dip = (p[-1, 0] - p[0, 0]) / max(p.shape[0] - 1, 1)
    d_strike = (p[0, -1] - p[0, 0]) / max(p.shape[1] - 1, 1)
    corners = [
        p[0, 0] - (d_dip + d_strike) / 2,
        p[0, -1] - (d_dip - d_strike) / 2,
        p[-1, -1] + (d_dip + d_strike) / 2,
        p[-1, 0] + (d_dip - d_strike) / 2,
    ]
    return shapely.Polygon(corners)


def outline(model: Path) -> shapely.Polygon:
    """The union of the model's plane footprints, gaps closed."""
    with h5py.File(model) as h:
        feet = [
            footprint(h[s]["lon"][()], h[s]["lat"][()])
            for s in h
            if s.startswith("segment")
        ]
    merged = shapely.union_all(feet).buffer(CLOSE_M).buffer(-CLOSE_M)
    if not isinstance(merged, shapely.Polygon):
        raise TypeError(f"the planes of {model} do not close into one outline")
    return merged


def surface(grid: Path, bound: shapely.Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Slab2's nodes inside the outline, and the triangles of the cells they fill.

    Each grid cell whose four nodes are all inside becomes two triangles. Slab2 gives
    depth in km, negative down, which TSurf takes as elevation in metres.
    """
    with xr.open_dataset(grid, engine="h5netcdf") as ds:
        lon = ds["x"].to_numpy()
        lat = ds["y"].to_numpy()
        table = ds["z"].to_numpy()
    lon = np.where(lon > 180, lon - 360, lon)
    ring = np.asarray(bound.exterior.coords)
    ring_lon, ring_lat = Transformer.from_crs(
        CRS, "EPSG:4326", always_xy=True
    ).transform(ring[:, 0], ring[:, 1])
    minx, maxx = ring_lon.min(), ring_lon.max()
    miny, maxy = ring_lat.min(), ring_lat.max()
    keep_lon = (lon >= minx - 0.1) & (lon <= maxx + 0.1)
    keep_lat = (lat >= miny - 0.1) & (lat <= maxy + 0.1)
    lon, lat = lon[keep_lon], lat[keep_lat]
    depth = table[np.ix_(keep_lat, keep_lon)]
    glon, glat = np.meshgrid(lon, lat)
    x, y = TO_UTM.transform(glon, glat)
    inside = shapely.contains_xy(bound, x, y) & np.isfinite(depth)
    index = np.full(inside.shape, -1)
    index[inside] = np.arange(inside.sum())
    a, b = index[:-1, :-1], index[:-1, 1:]
    c, d = index[1:, :-1], index[1:, 1:]
    full = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    triangles = np.concatenate(
        [np.stack([a, b, d], -1)[full], np.stack([a, d, c], -1)[full]]
    )
    vertices = np.stack([x[inside], y[inside], depth[inside] * 1000.0], -1)
    return vertices, triangles


def write_tsurf(path: Path, name: str, vertices: np.ndarray, triangles: np.ndarray):
    """Write one named TSurf object, gzipped when the path ends in .gz."""
    lines = [
        "GOCAD TSurf 1",
        "HEADER {",
        f"name:{name}",
        "}",
        "GOCAD_ORIGINAL_COORDINATE_SYSTEM",
        "NAME Default",
        'AXIS_NAME "X" "Y" "Z"',
        'AXIS_UNIT "m" "m" "m"',
        "ZPOSITIVE Elevation",
        "END_ORIGINAL_COORDINATE_SYSTEM",
        "TFACE",
    ]
    lines += [
        f"VRTX {i + 1} {x:.2f} {y:.2f} {z:.2f}" for i, (x, y, z) in enumerate(vertices)
    ]
    lines += [f"TRGL {a + 1} {b + 1} {c + 1}" for a, b, c in triangles]
    lines.append("END")
    text = "\n".join(lines) + "\n"
    opener = gzip.open if path.suffix == ".gz" else open
    path.parent.mkdir(parents=True, exist_ok=True)
    with opener(path, "wt") as f:
        f.write(text)


def main(model: str, grid: str, out: str) -> None:
    """Build the surface from the model and grid paths, and write it to out."""
    bound = outline(Path(model))
    vertices, triangles = surface(Path(grid), bound)
    write_tsurf(Path(out), "Maule", vertices, triangles)
    print(
        f"{len(vertices)} vertices, {len(triangles)} triangles, "
        f"outline {bound.area / 1e6:,.0f} km2, depth "
        f"{-vertices[:, 2].max() / 1000:.1f}-{-vertices[:, 2].min() / 1000:.1f} km"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
