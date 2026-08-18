# rupture-generator

Kinematic rupture models for ground-motion simulation: given a fault geometry and a
magnitude, draw a slip distribution, work out when each subfault ruptures and how it
slips, and write the result as an SRF.

A port and rework of `genslip` v5.6.2.

## Install

```sh
uv sync            # builds the two Rust extensions as part of the install
uv sync --extra vis   # and the 3-D viewer
```

Requires a Rust toolchain; `setuptools-rust` drives `cargo` from `pyproject.toml`.

## Use

Two steps, because the geometry is worth keeping and reusing across realisations.

```sh
# A fault system, discretised.
rupture-generator mesh examples/alpine_hope.geometry.toml alpine_hope.h5

# A rupture on it.
rupture-generator generate examples/crustal.toml alpine_hope.h5 rupture.srf

# Watch it happen (needs the `vis` extra).
rupture-generator view rupture.srf
```

`generate` writes `.srf` or the package's own `.h5`, chosen from the extension. The
native format keeps the mesh, the material properties and the causality tree alongside
the fields; an SRF keeps what the format has columns for.

From Python, when a curved subduction interface is the geometry:

```python
import pyproj
from rupture_generator import pipeline
from rupture_generator.config import read_config
from rupture_generator.realisation import Realisation
from rupture_generator.surfaces import read_tsurf

chart = read_tsurf("examples/cfm/Hikurangi.ts.gz").to_chart(spacing_km=0.4)
rupture = pipeline.generate(
    read_config("hikurangi.toml"),
    Realisation({"hikurangi": chart}, pyproj.CRS("EPSG:2193")),
)
```

## How it is put together

**One geometry type.** A fault is a `RuptureMesh`: a structured `(i, j)` grid of node
positions, `i` down dip and `j` along strike. A curved interface is not a second kind
of thing — `surfaces.py` fits a reference plane to a modeller's triangulation and
resamples it onto such a grid once, at build time, and everything downstream sees a
chart. What curvature costs is measured, not hidden: resampling recovers the CFM
Hikurangi interface's own surface area to +0.00% at 400 m.

**One fixed pipeline.** `generate()` runs the stages in one order, and there is no
mechanism for composing a different one. Stages are plain functions of
`(mesh, params, rng)`; each takes its own named substream of the event seed, so
changing one stage's parameters cannot move another stage's noise.

**Geometry is computed, never stored.** The nodes are the state. Centres, areas,
strike, dip and arc lengths are methods, so nothing can go stale and a file that
round-trips the nodes round-trips everything.

**SI throughout.** Slip in metres, moment in newton-metres, area in square metres. The
centimetres the SRF format wants appear in `srf.py` and nowhere else.

**Rust for the three hot kernels and the SRF codec, Python for everything else.**
`_kernels` exposes an eikonal solve, a pulse synthesiser and a von Kármán draw —
stateless, array in, array out, no RNG state and no spec structs, so the single copy of
every default stays in Python. Orchestration, configuration, geometry and I/O are
Python because that is where they are legible.

**Errors say which kind of wrong.** Everything this package refuses on purpose derives
from `RuptureGeneratorError`; the CLI catches that and renders it as a card, and lets
anything else traceback. A bug in the generator is not reported as a mistake in your
file.

**Configuration is the only description of a rupture.** Stages take frozen parameter
objects built from the config classes, and kernels take arrays, so there is no second
copy to disagree with. TOML, YAML or JSON, validated on construction, with the failing
key named.

```
rupture_generator/
  mesh.py           RuptureMesh: the chart, and building one from a config
  surfaces.py       GOCAD TSurf -> chart. Where curved geometry stops being special
  realisation.py    A fault system mid-flight: charts, CRS, causality, jumps
  pipeline.py       generate(): the stage order, and the config -> params seam
  stages.py         Slip, taper, rise time, rake, onset perturbation
  sampling.py       von Karman fields by circulant embedding
  timing.py         Rupture speed, and the eikonal solve over it
  pulses.py         Slip-rate pulses
  moment.py         Magnitude, moment, rigidity, the velocity model
  propagation.py    Which fault triggers which, and where the jump lands
  assemble.py       Realisation -> SRF. The one SI -> CGS boundary
  srf.py            The SRF format, over the Rust codec
  formats.py        The native HDF5/Zarr layouts, and extension dispatch
  errors.py         What this package raises, and what each type means
  config/           The TOML schema, and its validation
  scripts/          The three CLI commands, and the viewer
crates/
  kernels/          eikonal_solve, synthesise_pulses, von_karman_draw
  srf/              SRF read and write
```

## Development

```sh
just test     # pytest, doctests, and the Rust suites
just lint     # ty, ruff, clippy
```

Tests are property-first where a property is expressible: Hypothesis on the Python
side, `proptest` on the Rust side, with each test module's docstring stating its
strategy. Numbers quoted in docstrings are measurements, and the ones that constrain
behaviour are asserted somewhere.

## Known limits

- The sampler refuses a grid whose circulant embedding exceeds
  `sampling.MAXIMUM_EMBEDDING_CELLS`. Hikurangi at 100 m needs 111.8 M cells against a
  67.1 M cap, so the production resolution needs a tiled sampler that is not written.
- On a curved chart the wavefront sweeps in the parameter plane, so paths are short by
  the surface's own stretch — a median of -0.14 to +0.03 s on the CFM interfaces,
  against ruptures 143 to 255 s long.
- SRF reading and the SW4 SRF-HDF5 stream are not wired into the CLI yet.
