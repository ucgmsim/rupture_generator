# The mesh format

What `rupture-generator mesh` writes and `generate` reads: where a fault is, discretised.

```sh
rupture-generator mesh geometry.toml mesh.h5     # or mesh.zarr
```

An `xr.DataTree`, so `xarray.open_datatree` reads it and nothing here is needed to look
inside one.

```
/                          attrs: schema_version, created, title,
                                  crs        = "EPSG:2193"
                                  origins    = {"kaikoura": [1519.16, 5183.17]}
                                  geometry_config  <- the input, verbatim
/kaikoura/plane_0          dims (dip_node, strike_node)
    east_km   (dip_node, strike_node)   offset from the surface's origin
    north_km  (dip_node, strike_node)
    depth_km  (dip_node, strike_node)   positive downwards
  coords:
    strike_km (strike_node)   distance along the top edge from the i = 0 end
    dip_km    (dip_node)      distance down dip from the top edge
  attrs: surface, plane, strike_count, dip_count
/kaikoura/plane_1          ...
```

Every variable carries a `units` and a `long_name`. `README.md` argues this discipline is
what stopped shear speed being written in km/s where the SRF wants cm/s.

## Four decisions, and why

### Nodes, not cell centres

A group holds the **corners** of the grid. Cell centres, areas, per-cell strike and dip
are all functions of them, computed on read and never stored.

Storing centres instead — what an SRF carries, what a GSF carries — is the obvious
alternative and it is worse twice over. Centres do not say where the fault *ends*, so an
edge cell's area and the fault's boundary both have to be guessed by extrapolating half a
cell. And they force strike, dip and area to be stored alongside, because those cannot be
recovered from centres alone — which is a second description of the geometry, free to
drift from the first.

There is no face list because there does not need to be one: a structured grid's
connectivity *is* its shape, and `(dip_node, strike_node)` says it.

### Projected coordinates, not longitude and latitude

`crs` in the root attributes names a **projected** Cartesian frame — NZTM2000, a UTM
zone — and every position is in it. That is what makes the library's derived quantities
exact identities rather than approximations carrying a curvature error: areas sum to
length times width, a plane's cells all report the plane's own dip, twenty cells are the
same size.

The alternative was measured before being abandoned. On the WGS84 ellipsoid, a fault of
constant dip converges as it descends — depth means depth *below the surface* — so on a
60 km subduction interface the cell areas came out **1.4e-2** low and a mesh nobody asked
to be uneven had a down-dip step varying by **6.5e-3**. The first of those is larger than
`ENGINEERING_RULES.md`'s slip bound.

Longitude and latitude are derived at the one seam that needs them, in
`rupture_generator/mesh.py`, with `pyproj`.

> **Grid north is not true north.** A strike read straight out of this file is measured
> from the projection's northing axis. Converting it needs the grid convergence angle
> added, which in NZTM reaches **5.04°** — five times `ENGINEERING_RULES.md`'s one-degree
> rake bound. `project_patch` does it; anything reading the file by hand must too.

### Offsets, not absolute positions

`east_km` and `north_km` are measured from the surface's origin, which is in the root
`origins` attribute. Add it back for a coordinate in the CRS.

This is not tidiness. An NZTM northing is ~5,180 km against a ~1 km subfault, so absolute
positions round every node at CRS scale — an absolute error of `2.2e-16 × 5180 ≈ 5.7e-13`
km, which a cell-scale difference inherits as **1.2e-12 relative**. Measured: it is what
the mesh tests failed by when the library held absolute coordinates. Offsets put the same
quantities at **3e-15**, a factor of 400.

### One group per plane

A fault is a sequence of planes, each with its own dip, depth range and discretisation,
and each becomes its own group. They are connected because the config cannot express them
otherwise — a plane says only where its top edge *ends*, and where it begins is the
previous plane's end.

Planes are keyed by their stored `plane` attribute rather than by the order groups come
back in, because **Zarr does not preserve order**. Written `plane_0` through `plane_10`
and read back, it gives:

```
plane_10, plane_8, plane_5, plane_7, plane_9, plane_6, plane_4, plane_3, plane_2, plane_0, plane_1
```

which is neither insertion nor lexicographic. HDF5 *does* preserve insertion, so a reader
that trusted iteration order would work in one container and silently permute the fault
in the other — and since the order varies between runs, with two planes it comes out
wrong about half the time. A reordering is invisible in the file and produces a rupture
on the right surfaces with the wrong dips.

Anything reading these files by hand should sort on the `plane` attribute too.

## Reading one without this package

```python
import xarray as xr, numpy as np, json, pyproj

tree = xr.open_datatree("mesh.h5", engine="h5netcdf")
crs = pyproj.CRS(tree.attrs["crs"])
origin = json.loads(tree.attrs["origins"])["kaikoura"]

plane = tree["kaikoura/plane_0"].dataset
easting_km = origin[0] + plane["east_km"].to_numpy()
northing_km = origin[1] + plane["north_km"].to_numpy()

longitude, latitude = pyproj.Transformer.from_crs(
    crs, "EPSG:4326", always_xy=True
).transform(easting_km * 1000.0, northing_km * 1000.0)
```

Cell centres are the mean of each cell's four corners; areas are the two triangles a
corner quad splits into. If you want strike, remember the convergence.

## What is not in it

**No rupture.** Slip, rake, onset and the slip-rate pulses are `generate`'s output and
live in the rupture file, which embeds a copy of the mesh so a viewer needs only one
file. See `FORMAT.md`.

**No velocity model.** It is a property of the region rather than of the surface, it
changes independently, and it belongs to the source config.

**Nothing derived.** Listed above, and worth repeating because the temptation to cache is
real: a derived quantity written into the file is a second description that a later
change can leave behind.
