# rupture-generator

Kinematic rupture models for ground-motion simulation. Given a fault system and its
magnitudes, the generator draws a slip distribution and works out when each subfault
starts and how long it slips for. The result is an SRF.

A port and rework of `genslip` v5.6.2.

## Install

```sh
uv sync                 # builds the two Rust extensions as part of the install
uv sync --extra vis     # adds the 3-D viewer
```

The build needs a Rust toolchain. `setuptools-rust` drives `cargo` from
`pyproject.toml`.

## Use

A rupture is one TOML file next to a GeoJSON fault system.

```sh
# A rupture file in, an SRF out. OUTPUT.toml records the run, every default included.
rupture-generator examples/alpine_hope.toml alpine_hope.srf

# Watch it happen (needs the `vis` extra), or save a Rerun recording instead.
rupture-view alpine_hope.srf
rupture-view alpine_hope.srf --save alpine_hope.rrd
```

The examples under `examples/` include the Landers, Northridge, and Colombia
earthquakes, the Beavan and Alpine-Hope workflow realisations, and two small systems
for testing. All run at 0.1 km except Alpine-Hope, which runs at 0.25 km to fit in
about 3.5 GB of memory. `examples/from_realisation.py` converts a workflow
`realisation.json` into the same pair of files.

From Python, the usual run is five calls:

```python
charts = {
    name: chart.subdivide(0.1)
    for name, chart in geometry_from_geojson(handle, crs).items()
}
realisation = sample_path(Realisation(charts, crs), hypocentre, rng=rng)
medium = layered_medium(Layers(bottom_depth_km), shear_speed_km_s, density_g_cm3)
ruptures = generate(realisation, medium, sources, RuptureSettings(), seed=seed)
write_rupture(path, realisation, ruptures, dt_s=0.005)
```

`rupture_generator.config` builds the same inputs from a TOML file.

## Design

**One geometry type.** A fault segment is a `Geometry`: a structured `(i, j)` grid of
node positions in a projected CRS, `i` down dip and `j` along strike. A segment hung
from a bent trace is several planes side by side. Each plane has a separate column of
nodes at its edge, so two planes meet along the trace and part below it, as a fault
does at a kink. Centres,
areas, dips and arc lengths derive from the nodes, so nothing can go stale.

**Fault systems come from GeoJSON.** Each `Feature` is a section: a trace, one dip,
a dip direction, and a depth range. That's the `simpleFaultSource` data model of
OpenQuake and of the New Zealand, GEM and USGS hazard models, so a hazard model's
file loads directly and opens in QGIS.

**The rock is a function of position.** A `Medium` holds shear speed and density as
`SpatialField`s: deterministic functions from positions `(..., 3)` to values `(...)`.
A segment reads the medium at its cell centres, and a jump reads it along the line it
crosses. A 1-D velocity model is `layered_medium`, and a caller can pass any other
field, a 3D model or a random medium realised up front. The settings the rupture
itself varies with depth, such as rise-time and rupture-speed factors, are separate
`FaultProfiles` inside `RuptureSettings`. In TOML these are the `[medium]` and
`[profiles]` tables.

**One field sampler.** Slip, rise time, rake and the onset displacement all come from
one circulant-embedding sampler over a von Kármán covariance, as in Mai & Beroza 2002.
NORTA fits each field's *marginal* exactly, rather than rescaling a Gaussian, and
pre-corrects the covariance so that the configured correlation length applies to the
written field rather than to the latent one. genslip's `1 + cov * Z` clipped at zero
is a normal with a point mass at zero, and matches neither the mean nor the spread its
configuration asks for. Rise time and the onset displacement correlate with slip because
they share its latent field.

**A coherent front first, roughness after.** A factored fast-sweeping
eikonal solve gives first arrivals over the rupture speed field, `|grad T| = 1/v`,
with its minimum at the seed. A slip-correlated displacement in seconds is then
added. Seconds are the unit because the spread of rupture-time heterogeneity is the
quantity calibrated against recorded ground motion. That spread follows each
segment's own moment:

    sigma = offset_s + coefficient * 1e-9 * M0^(1/3)

with `M0` in dyne-centimetres, the units the published coefficient uses. The defaults
are the magnitudes of genslip's `tsfac_bzero` and `tsfac_slope` as the production
workflow sets them.

**The displacement blends in from the seed.** genslip adds it raw. Its earliest
subfault is then whichever patch drew the deepest dip rather than the hypocentre, and
subfaults start before the front arrives at them. Here the displacement grows in with
time since the seed:

    tau   = travel - seed_time
    blend = min(1, tau / (n * sigma))                  # n = blend_sigma, 4
    clamp = min(1, tau / (c * max(-displacement, 0)))  # per cell, c = 1.05
    onset = travel + min(blend, clamp) * displacement

The blend is the model, a zone of `n` sigma over which roughness accumulates. The
clamp is arithmetic: per cell, no subfault starts before the front that seeded it, and
one deep dip doesn't delay the rest of the fault. The seed keeps its time exactly,
because both terms are zero there.

**The rupture speed has two branches.** In-plane rupture has no steady solution
between the Rayleigh speed and the shear speed. The speed fraction runs
sub-Rayleigh up to `0.9194 Vs`, or supershear from `Vs` to `sqrt(2) Vs`, with the
forbidden zone skipped by a shift rather than clipped into. genslip clips onto the
whole interval `[0.25, 1.414]`, forbidden zone included.

**Faults trigger each other through a tree.** Every segment except the first has one
triggering parent. A rupture file states the tree, or the generator draws one or
takes the likeliest under the Shaw & Dieterich 2007 jump model, where the chance of
crossing a gap decays with its width. Drawing uses Wilson's algorithm, and the likeliest tree is
a maximum spanning tree.

**A jump crosses as an S wave.** Each of the child's edge cells pairs with its
nearest parent cell. Its trigger time is the front's arrival at that parent cell plus
the shear-wave travel time across the gap, integrated through the medium. Pairing from
the parent's side instead would let a wave from far up the parent outrun the front,
since the shear speed is faster than the rupture speed along any straight line. The
Shaw & Dieterich model, read as a survival function, sets the jump's range: one draw
per jump from its own random stream, conditioned on the tree having crossed the
nearest gap. The earliest trigger time within that range is the child's seed.

**Each segment draws from its own stream.** A segment's random generator derives from
the event seed and the segment's name. Adding, removing, or reordering segments leaves
every other segment's fields bit-identical, and so does changing how fronts jump.

**SI throughout.** The package works in metres for slip, newton-metres for moment
and square metres for area. The SRF format's centimetres appear only in
`formats/srf.py`.

**Rust for the hot kernels and the SRF codec.** `_kernels` exposes the eikonal solve,
the slip-rate pulse synthesis and the circulant-embedding draw. They're stateless
functions of arrays, and the defaults they use come from Python. Orchestration,
configuration, geometry, and file handling are Python too.

**Errors say which kind of wrong.** Everything this package refuses on purpose raises
`RuptureGeneratorError`. The command line reports that as a message and lets anything
else raise, so a bug in the generator never looks like a mistake in your file.

```
rupture_generator/
  cli.py              rupture-generator: a rupture file in, an SRF out
  config.py           The TOML schema, and building the library's inputs from it
  errors.py           What this package raises
  view.py             rupture-view: the rupture in Rerun
  geometry/
    geometry.py       Geometry: the chart, its cells and distances between charts
    geojson.py        GeoJSON sections into charts
  rupture/
    generator.py      Slip, rise time, rake, onset, and the jump rule
    medium.py         Medium, SpatialField and the 1-D velocity model
    propagator.py     The jump model, and which fault triggers which
    realisation.py    A fault system: charts, CRS, hypocentre and tree
    source.py         Magnitude, moment, rake and correlation lengths per segment
  sampling/
    field.py          Gaussian fields by circulant embedding
    norta.py          Exact marginals, and the latent correlation behind them
    von_karman.py     The von Kármán correlation function
  formats/
    srf.py            SRF 2.0 written and read, over the Rust codec
crates/
  kernels/            eikonal_solve, synthesise_pulses, circulant_draw
  srf/                SRF read and write
```

## Development

```sh
just test     # pytest, doctests, and the Rust suites
just lint     # ty, ruff, clippy, numpydoc
```

Tests state properties wherever one exists to state, with Hypothesis on the Python
side and `proptest` on the Rust side. Numbers quoted in docstrings are measurements,
and tests assert the ones that constrain behaviour.

## Known limits

- The sampler refuses a grid whose circulant embedding exceeds
  `sampling.field.MAXIMUM_EMBEDDING_CELLS`, 134 million padded cells.
- Jumps cross along a straight line. A shear wave in a layered medium bends, and
  can arrive sooner along a fast layer than along the chord.
- The wavefront sweeps the chart's index grid, so on a curved or tapered chart the
  travel paths follow the grid's own metric rather than the fault's real geometry.
