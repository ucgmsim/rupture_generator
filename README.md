# rupture-generator

Kinematic rupture model generation: a pipeline in Python over two Rust kernels, with a
command line, a self-describing output format, and a viewer.

```sh
uv sync --extra test --extra vis --group dev

rupture-generator mesh     examples/hope.geometry.toml  mesh.h5
rupture-generator generate examples/crustal.toml        mesh.h5  rupture.h5
rupture-generator view     rupture.h5
```

Three steps, and the boundary between them is a file, because their inputs have
different lifetimes. A **geometry** is digitised once and reused across every
realisation run on it. A **source config** is what varies. A **rupture** is the output.

`rupture.h5` can equally be `.zarr`, `.srf`, or `.srf.h5` for SW4 — the format is
inferred from the extension. The native one is an `xr.DataTree` that carries the mesh it
was generated on, the config that produced it, and the causality tree, which an SRF has
never been able to do.

## The pipeline

A rupture realisation is a composition of nine stages. Each is a pure function; a
stage's output is the input to the next, and nothing else flows between them.

| | | |
| --- | --- | --- |
| **S1** | geometry → coarse mesh | a digitised trace becomes a quad mesh |
| **S2** | coarse mesh → chart | subdivided to the target subfault size |
| **S3** | chart → validated chart | assert what the spectral sampler assumes |
| **S4** | chart, M<sub>w</sub>, rng → slip | a correlated random field, scaled to the moment |
| **S5** | chart, slip, rng → rise time | correlated with slip, stretched by depth |
| **S6** | chart, rng → rake | an independent field of the same covariance |
| **S7** | chart, seeds → travel time | the eikonal equation, by factored fast sweeping |
| **S8** | travel time, rng → onset | perturbed so high-slip patches rupture early |
| **S9** | slip, rise, onset → pulses | a slip-rate function per subfault |

`pipeline.py` is the only file where that order is written down, and it is a
**convention rather than a contract**: every stage draws from its own substream, named
for the stage and the segment, so reordering them or changing one stage's parameters
cannot change another's noise.

```python
from rupture_generator.config import read_config, read_geometry
from rupture_generator.pipeline import generate, segments_of

geometry = read_geometry("examples/hope.geometry.toml")
config = read_config("examples/crustal.toml")

realisation = generate(
    config,
    segments_of(geometry),
    geometry.crs,
    propagation_config=geometry.propagation,
)
realisation.segments["hope"]["slip_m"]   # an xarray.Dataset per segment
```

## Several faults, one earthquake

A rupture may cross between faults. Which one triggers which is a **tree**, fixed
before any field is drawn, and it comes one of two ways:

```toml
[propagation]              # sample it from how far apart the faults are
type = "computed"
strategy = "sampled"       # or "maximum_likelihood"
d0_km = 3.0                # Shaw & Dieterich (2007)
delta_km = 1.0
max_jump_km = 15.0
```

```toml
[propagation]              # or state it
type = "predetermined"
parents = { kelly = "hope", conway = "kelly" }
```

The division of labour is exact: **the tree decides who triggers whom; the wavefront
decides where and when.** For each edge of the tree the jump point is found by
minimising arrival time over pairs — so a front that reaches the far end of a fault
early jumps from there, rather than from wherever the two faults happen to be closest.
Closest approach is a fact about geometry, and only arrival knows which way the front
was travelling.

The jump delay is a model rather than a formula: `Instantaneous`, `DistanceOverVelocity`,
and room for a stochastic one.

## Units

**MKS**: slip in metres, slip rate in metres per second, rigidity in pascals, moment in
newton-metres. Geometry is the one deliberate exception — a fault is written down and
meshed in kilometres, because that is the scale a fault has.

The centimetres and dyne-centimetres the SRF format wants appear in `assemble.py`, at
the writer, and nowhere else.

## Where the fault is

The mesh is built in a **projected Cartesian CRS the modeller names**, and never leaves
it. In that frame every derived quantity is an exact identity: areas sum to length times
width, every cell of a plane reports the plane's own dip, twenty cells are the same size.
On the WGS84 ellipsoid the same quantities came out with cell areas 1.4 × 10⁻² low on a
60 km subduction interface — larger than the slip bound.

There is exactly one seam out, in `mesh.py`, and it does two things rather than one:
positions are transformed, and **strike gains the grid convergence angle**, because grid
north is not true north. In NZTM2000 that reaches 5.04° — five times the one-degree rake
bound, and about the width of the difference between a reverse and an oblique-reverse
mechanism.

## Layout

```
crates/
  kernels/            eikonal solve + pulse synthesis, PyO3, array-in/array-out
  srf/                the SRF reader and writer
rupture_generator/
  config/             the single copy of every parameter
  mesh.py             RuptureMesh: the one mesh type, S1–S3, fusion,
                      the hypocentre seam, projected→WGS84
  sampling.py         FieldSampler protocol, CovarianceSpec, SpectralSampler
  moment.py           magnitude→moment, rigidity, the joint slip scaling
  stages.py           S4–S6, S8
  timing.py           S7: speed field, the geometric correction, the eikonal call
  propagation.py      the causality tree, causal jumps, JumpDelay
  pulses.py           S9, and the slip-rate vocabulary
  pipeline.py         generate(): the one statement of stage order
  formats/            mesh + rupture files
  assemble.py, srf.py the SRF output path
  scripts/            mesh / generate / view
```

Rust is confined to the two things it earns: the factored fast-sweeping eikonal solver,
and pulse synthesis. Both are stateless functions over numpy arrays, with no RNG and no
spec structs crossing the boundary. If profiling ever shows a Python stage to be the
bottleneck it drops into `crates/kernels` behind the same signature — that is the
performance escape hatch, and it is one-directional by design.

## Testing

**Generative property testing** — `proptest` in Rust, `hypothesis` in Python — over
encapsulated stages. The gate is plain `cargo test` and `pytest`.

```sh
cargo test && uv run pytest
```

Three rules, and they are what the suite is for:

- **Properties, not values.** No test pins a number the C produced. A test asserts a
  mathematical or physical property, with the reason for its tolerance written at the
  assertion.
- **Quantify over inputs.** A property holds for *generated* grids, seeds, magnitudes,
  hypocentres and trees. A property asserted at a single point is a smoke test.
- **A reference that re-implements the subject is not a reference.** Expected values
  come from analytic solutions, published formulas, conservation laws or statistical
  estimators — never from a second transcription of the code under test.

Three tolerance vocabularies appear, and which one a test uses is a claim about what
kind of statement it is making. An **identity** is asserted as an identity: the
correlation blend either is `ρ·a + √(1−ρ²)·b` or is not, and asserting it that way
catches a ρ of 0.8 written as 0.5 by a factor of a million where a sample correlation
would miss it by well under one standard error. A **construction** gets arithmetic
slack only. A **statistical** claim carries its estimator's error, derived at the
assertion — for a random field, from its effective patch count rather than its subfault
count.

## What the rewrite changed, and why

The port was correct and measured, but its structure was the C's: the fault surface was
represented six times between config and output, every spec type existed three times
with a test whose only job was to police the copies, and the stage order was a
load-bearing RNG contract that two dead fields were drawn and discarded to preserve.

- **One mesh type.** `RuptureMesh` is an `xarray.Dataset` of node positions with methods
  for everything derived from them. A derived quantity written down is a second
  description free to drift from the first.
- **One copy of every parameter.** The Rust spec mirror and the completeness gate that
  policed it are gone, because the class of bug they caught — disagreeing copies of one
  default — became unrepresentable. Every one of the four wrong numbers found in the
  reduction sweep was a disagreement between copies.
- **Randomness by name, not by order.** One event seed; `SeedSequence` spawns a stream
  per (stage, segment). Draw order stopped mattering, so the machinery that preserved it
  was deleted rather than ported.
- **The physics vocabulary shrank to what production selects** — one corner relation, one
  spectral shape, one slip-rate family. Each removed name is refused *by name, saying it
  was removed*, because the workflow's defaults file advertises them.
- **Segments, not refusals.** Planes that do not share a seam used to be an error
  ("multi-segment is not written"); they are now two segments, and whether a rupture
  crosses between them is the propagation stage's question.

`PLAN.md` is the argument for the whole shape, and `git log` carries the measurements —
each commit message records what a change cost or caught.
