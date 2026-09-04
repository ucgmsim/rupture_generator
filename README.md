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

# Or record it, through the same viewer run headless (needs ffmpeg).
python examples/record_rupture.py rupture.srf rupture.mp4
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

**One field sampler.** Slip, rise time, rake and the onset displacement all
come from `von_karman_field`, whatever their distribution and whether or not their
correlation length varies with depth. A field's *marginal* is fitted rather than approached by
rescaling — genslip's `1 + cov * Z` clipped at zero is a normal with a point mass at
zero and neither the mean nor the spread it was configured with — and NORTA
pre-corrects the covariance so the fitted correlation length lands on the field that
gets written out rather than on a latent one that does not. A field whose structure
does *not* vary with depth is the special case, and takes the cheap path by an
identity rather than by a second implementation: a covariance that is stationary down
dip is Toeplitz, a Toeplitz operator embeds in a circulant one, and a circulant
operator's eigenvectors are the DFT.

**The front is solved coherent, and roughened afterwards.** The eikonal solve runs over
a rupture speed that is a function of depth alone, so what it returns is a first-arrival
field: `|grad T| = 1/v` with `v` inside the configured band, its minimum exactly at its
seed. A slip-correlated displacement is then added to that field. Seconds is the
currency deliberately — the spread of rupture-time heterogeneity is the one quantity in
this model fitted against recorded ground motion, and it is not a form a perturbed
slowness field can consume, since the onset scatter such a field produces depends on the
fault's size, its dip and rake and the subfault size.

**That spread follows the moment**, and is read off each segment's own magnitude:

    sigma = rupture_time_offset_s + rupture_time_coefficient * 1e-9 * M0^(1/3)

with `M0` in dyne-centimetres, the published units the coefficient is quoted in — the
same convention `rise_time_coefficient` already uses. This is Graves & Pitarka's
relation, the Somerville et al. (1999) self-similar duration scaling with an offset that
flattens its magnitude dependence, and the two defaults are the magnitudes of genslip's
`tsfac_bzero` and `tsfac_slope` as the production workflow sets them
(`default_parameters/root/defaults.yaml`, where `tsfac_main` is left null so genslip
derives it per fault). It runs 0.14 s at Mw 5, 0.45 s at Mw 7 and 3.6 s at Mw 9. A
per-fault source therefore gets a different spread on every fault, which is how the
workflow invokes genslip — once per fault, with that fault's `mag=`. Both defaults zero
is a coherent front, which draws nothing.

**What makes that safe is the blend.** genslip adds the displacement raw
(`psrc[ip].rupt = rt + tsfac_main*tsfac1_r[ip]`, `genslip_v5.6.2.c:3135`) and pays for
it twice: the field's minimum lands on whichever high-slip patch drew the largest
negative displacement rather than on the hypocentre, so the whole field is shifted by
that minimum and the hypocentre in the SRF header no longer matches the one in the
times; and subfaults rupture earlier than any front could reach them. Here the
displacement is admitted with distance from the seed, over a zone whose width is stated
in sigmas of the displacement itself:

    tau   = solved - seed_time                      # travel since the front arrived
    blend = min(1, tau / (n * sigma))               # n = rupture_time_blend_sigma, 4
    clamp = min(1, tau / (c * max(-displacement, 0)))   # per cell; c = 1.05
    onset = solved + min(blend, clamp) * displacement

The rupture therefore leaves its hypocentre smooth and roughens as it goes —
heterogeneity is something a front accumulates, not something it starts with — and the
seed keeps its registration exactly, because the blend is zero there whatever the draw
did. The two terms answer different questions and that is why there are two. The blend
is the model, and its width is a number you choose. The clamp is arithmetic, and it
binds per cell on the draws that dip deeper than the blend admits.

Neither term could do the other's job. A single blend of `n * sigma` is not safe on
every draw: swept over 768 cases across four magnitudes, two resolutions, two fault
styles and four seed positions, one draw needed 8.04 sigma, and the depth of the worst
dip is a property of the realisation rather than of the parameters. Reading the width
off that worst dip instead — the earlier form here — makes the blend length something
the draw decides and holds the whole fault back behind a single deep cell. Splitting
them gives a configured width, a per-cell correction, and no fault-wide coupling. The
ramps are linear because admissibility is `sup w(s)/s <= 1`, which the linear ramp
attains and every sub-linear ramp violates near zero however wide it is made.

At the shipped defaults the blend zone is four sigma — 1.8 s on a Mw 7 fault, 14.6 s on
the Mw 9 Hikurangi interface — against ruptures tens to hundreds of seconds long, so it
costs a fraction of a per cent of the target spread. Because the width is stated in
sigmas rather than seconds it tracks the amplitude relation without a second
magnitude-dependent number. What it does not do is make the onsets causal *away* from
the seed: the field's gradient there is the drawn field's, bounded by nothing, and that
is the trade this mechanism makes for being expressible in the units the calibration is
in.

There is one timing field, `onset_s`, and not a second holding the front before the
displacement. That second field existed so a jump's departure could be chosen off a
smooth wavefront while its clock was read off a displaced one; two fields is a claim
that the front and the times disagree about where the rupture is. `causal_jump` chooses
on `onset_s`, over the parent's boundary arrests rather than its whole chart, and
minimises `onset + delay(distance)` — a delay that varies by seconds across a boundary
where the displacement's spread is that segment's sigma. Its docstring says what that
costs.

**The velocity band is two branches, not an interval.** What bounds the solved front's
`|grad T|` is a bound on the rupture speed, and the speeds a front can actually hold are
not an interval: for in-plane rupture there is no steady solution between the Rayleigh
speed and the shear speed, so the band spans sub-shear up to `cR = 0.9194 Vs` and
supershear from `Vs` to `sqrt(2) Vs` — the Burridge–Andrews speed — with the mode-II
forbidden zone between them left empty, skipped by a monotone shift rather than clipped
into. The shipped model never reaches either wall: `SpeedParams.__post_init__` holds the
background sub-Rayleigh, checking the fraction *after* the geometric correction —
`velocity_fraction` alone does not bound it, since dividing by `alpha_T` takes a
configured 1.0 on a reverse fault to 1.1 of the shear speed — and the depth profile only
scales that down. The branches are kept because the band is a statement about what a
front can hold, and a depth factor above 1 is a configuration that reaches for one.

genslip instead hard-clips onto the continuous interval `[rvfmin, rvfmax]` =
`[0.25, 1.414]` (`genslip_v5.6.2.c:658`, `ruptime.c:883`) and allows every value between,
the forbidden zone included; it counts the supershear subfaults and prints the fraction
(`isupsh`) but nothing acts on the count. Its clip is also applied to the fraction
*before* the depth profile (`rspd = rfslip*rfdep*vs`), which cannot keep the realised
speed out of the zone — a fraction of 1.0 scaled by a depth factor of 0.95 lands at 0.95,
inside it. Here the profile goes on first and the band is the last word, which is why
`MINIMUM_VELOCITY_FRACTION` has to sit at genslip's 0.25 rather than higher: the shallow
background is already `0.8 * 0.6 = 0.48`, so a floor of 0.4 would leave the shallow weak
zone almost no range to work in.

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
  stages.py         Slip, taper, rise time, rake, onset displacement and its blend
  sampling.py       von Karman fields, NORTA marginals, depth-varying lengths
  timing.py         Rupture speed, the causal band, and the eikonal solve over it
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
  kernels/          eikonal_solve, synthesise_pulses, von_karman_draw,
                    factorise_covariance, nonstationary_draw
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
- A depth-varying profile costs the *square* of the dip resolution in memory and its
  cube in time, where a stationary one costs the logarithm: the covariance across
  depths is dense at every wavenumber and has to be factorised. Wellington's five
  segments take 72 s at 100 m; the Hikurangi interface at 1 km is 1.85 GB of factors,
  and at 100 m it is out of reach. `[source.hybrid]` is opt-in for that reason among
  others.
- `[source.hybrid]` puts Suzuki et al. (2022) above the ramp and Mai & Beroza (2002)
  below it. Suzuki's down-dip length saturates at M6.3 and Mai's does not, so past
  about M8.6 the shallow branch is the *shorter* of the two down dip and the hybrid
  inverts. It is a crustal model; the interface examples do not turn it on.
- On a curved chart the wavefront sweeps in the parameter plane, so paths are short by
  the surface's own stretch — a median of -0.14 to +0.03 s on the CFM interfaces,
  against ruptures 143 to 255 s long.
- SRF reading and the SW4 SRF-HDF5 stream are not wired into the CLI yet.
