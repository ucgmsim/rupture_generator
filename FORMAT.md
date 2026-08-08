# The rupture format

What `rupture-generator generate` writes: a kinematic rupture model, and the mesh it was
generated on.

```sh
rupture-generator generate config.toml mesh.h5 rupture.h5      # native, one file
rupture-generator generate config.toml mesh.h5 rupture.zarr    # native, a store
rupture-generator generate config.toml mesh.h5 rupture.srf     # text SRF
rupture-generator generate config.toml mesh.h5 rupture.srf.h5  # SW4's SRF-HDF5
```

The first two are this package's own layout and are what the rest of this describes. The
last two are other people's formats, written unchanged: `rupture_generator/srf.py` owns
them, and the SRF's own limits — six significant figures, whole-degree rake, no
provenance — are why there is a native one at all.

## The layout

An `xr.DataTree`, so `xarray.open_datatree` reads it.

```
/                             attrs: schema_version, created, title, crs, origins,
                                     config          <- the input, verbatim
                                     surface, seed, realisation, rng_engine,
                                     moment_dyne_cm, alpha_t, sample_interval_s
/alpine/plane_0
  node variables      (dip_node, strike_node)
      node_east_km  node_north_km  node_depth_km      projected offsets
  cell variables      (dip, strike)
      centre_longitude_deg  centre_latitude_deg  centre_depth_km    WGS84
      strike_deg  dip_deg  area_cm2                                 true north
      slip_cm  rake_deg  onset_s  rise_time_s
  slip-rate pulses    (sample,) and (cell_edge,)
      slip_rate  slip_rate_column  slip_rate_offset                 CSR
  coords: strike_km (strike_node), dip_km (dip_node)
  attrs:  plane, strike_count, dip_count, sample_interval_s,
          moment_dyne_cm, alpha_t, hypocentre_strike_km, hypocentre_dip_km
```

Every variable carries a `units` and a `long_name`.

## Four decisions, and why

### One group per plane, fused or not

A fault whose trace bends is **one** rupture: `generate` concatenates its planes into a
single grid whose strike varies along it, so the rupture front crosses the bend. The
output is still one group per plane, with the fields split back — the file's shape
does not depend on how the generator happened to run.

A fault whose dip, dip direction or width changes between planes is *not* one surface,
and is refused by name. The test is geometric: the planes' shared column of nodes either
coincides or it does not.

### Self-contained

Each group carries the node positions as well as the fields, so a rupture file is
everything a viewer or a consumer needs. A rupture that pointed at a mesh file would be a
pair that can be separated, and a slip field without its geometry is a grid of numbers.

The cost is a copy of the mesh per rupture. It is small — nodes are `(n+1)×(m+1)` floats
against a slip-rate array that is typically thousands of samples per subfault — and a
study generating a hundred realisations on one mesh pays it a hundred times, knowingly.

### Two coordinate systems, and neither is redundant

The **nodes** are projected offsets in the CRS, exactly as the mesh file holds them: they
are the geometry, they are what a renderer draws, and they are exact — see `MESH.md` for
why offsets rather than absolute positions.

The **cell** variables are WGS84, because they are what an SRF is written from and what
every consumer downstream expects.

That is not the two-descriptions-of-one-thing this repo forbids elsewhere: one is input
and the other is derived output. `tests/test_generate_cli.py` asserts the two agree after
a round trip through the file, and against the SRF written from the same run.

> `strike_deg` here is **true** north. The mesh file's is **grid** north. They differ by
> the convergence angle, up to 5.04° in NZTM, and the variable's `long_name` says which
> is which in capitals for that reason.

### The pulses are CSR

Each subfault's pulse has its own length — `nt1` is what the slip-rate generator
*returned*, not `rise_time / dt`, which is `README.md`'s first trap. So they are stored
concatenated with an offset array:

- `slip_rate` — every sample, end to end
- `slip_rate_offset` — where each subfault's pulse starts; `len(cells) + 1` entries
- `slip_rate_column` — each sample's position **within its own pulse**

That is `scipy.sparse.csr_array`'s layout exactly, and the core's own, so nothing is
translated on either side:

```python
import scipy.sparse as sp
pulses = sp.csr_array(
    (ds["slip_rate"], ds["slip_rate_column"], ds["slip_rate_offset"]),
    shape=(strike_count * dip_count, longest_pulse),
)
```

A subfault with **no** pulse has zero length, and that is not the same as a pulse of
zeros: it is what the format says for a subfault that did not slip. On a tapered fault it
is every edge subfault.

### Planes are keyed on their attribute

`planes_in` sorts on the stored `plane`, not on iteration order, because **Zarr does not
preserve order**. `MESH.md` has the measurement.

## Reading one without this package

```python
import xarray as xr, numpy as np, scipy.sparse as sp

tree = xr.open_datatree("rupture.h5", engine="h5netcdf")
plane = tree["alpine/plane_0"].dataset

slip_cm = plane["slip_cm"].to_numpy()             # (dip, strike)
longitude = plane["centre_longitude_deg"].to_numpy()

offsets = plane["slip_rate_offset"].to_numpy()
pulses = sp.csr_array(
    (plane["slip_rate"].to_numpy(), plane["slip_rate_column"].to_numpy(), offsets),
    shape=(slip_cm.size, int(np.diff(offsets).max())),
)
dt = plane.attrs["sample_interval_s"]

# Subfault k's pulse, in centimetres per second, starting at its own onset:
pulse = pulses[[k]].toarray().ravel()[: offsets[k + 1] - offsets[k]]
```

Arrays are `(dip, strike)` and flatten **along strike fastest**, which is the order the
core produces every field in and an SRF stores points in.

## What provenance it carries

The whole config, verbatim, in the root `config` attribute — plus the seed, the
realisation and which random engine ran. An SRF has never had any of this: a file arrives
and there is no way to tell what produced it.

`moment_dyne_cm` is a target the generator was scaled to hit rather than something it
accumulated, so it is exact and is worth checking against: the integral of the moment
rate should return it. `rupture_generator.moment` does that, and
`tests/test_moment.py` is what says it holds.
