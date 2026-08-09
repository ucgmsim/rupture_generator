# PLAN — the pipeline rewrite

This document is the basis for a ground-up restructure of this repository. It is written
to be executed in phases by agents; each phase has a deliverable and acceptance criteria.
Nothing in it is implemented yet.

**Design decisions this plan encodes** (settled with the author, 2026-08-09):

0. **Units are MKS.** Slip in metres, rigidity in pascals, moment in newton-metres —
   the CGS the C worked in survives only inside the SRF writer, where the format
   demands centimetres. Eq. 7 is therefore its SI form,
   `log₁₀ M₀[N·m] = 1.5·M_w + 9.05`. Geometry stays in kilometres (the mesh file
   format is kept); the km²→m² conversion happens once, in the moment closure.

1. **Python orchestrates, Rust provides kernels.** The pipeline is composed in Python;
   Rust is reduced to stateless array-in/array-out kernels behind narrow protocols.
2. **The gate is properties, not parity.** Output is free to move relative to the current
   port. A change is acceptable when the property suite passes, full stop.
3. **Multi-segment is designed in now; curved geometry arrives later through seams.**
   The rupture-causality tree is sampled upfront (Shaw–Dieterich, as in
   `source_modelling.rupture_propagation`), but hypocentres and onsets on child faults
   are computed **causally from the wavefront**, not fitted by closest distance. Each
   fault has a single triggering parent. The jump delay is a pluggable model.
4. **The front end survives; the physics vocabulary shrinks.** `mesh` / `generate` /
   `view`, both file formats, and the viewer are kept. The slip-rate and spectrum menus
   are cut to what production selects.

---

## 1. Why

The port is correct and measured, but its structure is the C's. The evidence, from a
survey of the tree as it stands:

- **The fault surface is represented six times** between config and output:
  `GeometryConfig` → `genslip::mesh::Mesh` → `RefinedMesh`/`PatchView` →
  `generate_cli.Fused` (Python-only multi-plane concatenation) →
  `realisation::FaultGrid` → `Located`/`SubfaultGeometry`. Four of those exist only
  because no single type can carry both geometry and annotations.
- **Every spec type exists three times** — `genslip::realisation::{SourceSpec, SlipSpec,
  TimingSpec}` are mirrored in `crates/core` (PyO3 wrappers) and again in
  `config/rupture.py` (dataclasses), with a test (`test_config_completeness.py`) whose
  only job is to police the mirrors. Three files change for one new knob. Every one of
  the four wrong numbers found in the reduction sweep was a disagreement between copies.
- **Stage order is fused and unmovable.** `realisation::generate` draws slip, rake,
  rupture perturbation, and rise-time noise off one RNG stream in a fixed order; two
  dead genslip fields are still drawn-and-discarded (`slip::skip_unused_field`) purely
  to preserve that order. No stage can be reordered, retried, or tested in isolation.
- **The C's discretisation leaks everywhere**: padded even-extent rules asserted in
  three places, strike-fastest index conventions enforced by comment, a flat error enum
  named after C failure modes, and a `TimingSpec` bundling eleven concerns because
  that is how `getpar` state was shaped.

The inspiration is `~/src/nzcvm`: frozen dataclasses with `from_config` constructors, a
pipeline of composable stages behind one protocol, `xarray` as the interchange type, and
Rust confined to the kernels that earn it.

---

## 2. The pipeline as mathematics

A rupture realisation is the following composition. Each numbered stage is a function; a
stage's output is the input to the next; nothing else flows between them.

Notation: a **segment** is a chart `X: (i, j) → R³` — a structured grid of nodes in a
projected Cartesian CRS, `i` down-dip, `j` along-strike. For now every chart is planar
per fused surface (**the temporary flatness constraint**); the interfaces below never
assume it, only `X` does.

**S1. Mesh generation.** `geometry → coarse mesh`. A digitised geometry (trace, dips,
depths, in a named projected CRS) becomes a coarse quad mesh. Bent traces fuse into one
chart with strike varying along it; discontinuous dip/width is two segments.

**S2. Mesh subdivision.** `coarse mesh × resolution → mesh`. Each chart is subdivided to
the target subfault size, giving nodes `X_ij`, and derived quantities — cell centres,
areas `A_ij`, local strike/dip — **computed from nodes on demand, never stored in a
parallel struct**.

**S3. Chart validation** *(the temporary stage)*. Assert each chart is planar and
uniformly spaced, i.e. that the FFT sampler's assumptions hold. This stage is the seam
where curvature later enters: it is the only code allowed to know the sampler needs
flatness, and deleting it (plus swapping the sampler) is the whole curvature migration.

**S4. Slip field.** `mesh × (M_w, spectrum params, rng) → mesh + slip`. Sample a
zero-mean, unit-variance Gaussian random field `Z_s` with the configured covariance
(spectral shape + correlation lengths), transform to non-negative slip (mean-shift,
truncate, edge-taper), then scale **once, jointly over all segments** so that

    Σ_segments Σ_ij  μ_ij · A_ij · s_ij  =  M₀(M_w),   log₁₀ M₀ = 1.5·M_w + 16.05

(Hanks & Kanamori eq. 7 — the **one** magnitude convention; the constant in the port's
dyne-cm form is `10.699967`). Rigidity `μ` comes from the 1-D velocity model sampled at
subfault depth.

**S5. Rise-time field.** `mesh × slip × (ρ, α, ramps, rng) → mesh + rise_time`. Sample
`Z_τ` correlated with the slip field's Gaussian at level ρ (jointly:
`Z_τ = ρ·Z_s + √(1−ρ²)·Z_indep`), blend toward slip near the surface by a depth ramp,
apply the power law `τ ∝ s^α` (α = 0.5, Graves & Pitarka), normalise to unit mean, scale
by the moment-derived average rise time, and stretch by the shallow/deep depth ramps.

**S6. Rake field.** `mesh × (σ_rake, rng) → mesh + rake`. `rake_ij = base_rake_ij +
σ·Z_r`, with `Z_r` an independent GRF of the same covariance family. Not correlated with
slip (as today).

**S7. Wavefront propagation.** `mesh × hypocentre × speed params → mesh + travel_time`.
Rupture speed `v_ij = f·β(z_ij)·ramp(z_ij)` (velocity fraction × shear speed × depth
ramp, with the geometric correction applied to `f` **inside the stage**, not by the
caller). Solve the eikonal equation `|∇T| = 1/v` on the chart by factored fast sweeping
from seed point(s) with initial time(s). *The seed contract — points with times, not
"the hypocentre" — is what multi-segment needs and costs single-segment nothing.*

**S8. Onset perturbation.** `mesh × travel_time × slip × (c, rng) → mesh + onset`.
`t_ij = T_ij + c·σ_T·Z_p`, where `Z_p` is correlated with slip so high-slip patches
rupture systematically earlier (c < 0 today), and the hypocentre's onset is pinned to
its travel time.

**S9. Pulse synthesis and output.** `mesh × slip × rise_time × onset → annotated mesh`.
Per subfault, synthesise the slip-rate pulse (shape family, `∫ ṡ dt = s_ij` exactly; a
rise time unrepresentable at the sample interval is a **refusal naming the subfault**,
never a silent zero). The result is the input mesh with `slip`, `rake`, `rise_time`,
`onset`, and pulses attached — written to the existing rupture format, or through the
one projected→WGS84 seam to an SRF.

### Multi-segment (stages S7–S8 generalise; S4 already stated the joint scaling)

Given segments `{F_k}` and their sources:

1. **Tree sampling** (upfront, reusing `source_modelling.rupture_propagation`): distance
   graph → Shaw–Dieterich probability graph → sampled (or maximum-likelihood) spanning
   tree, rooted at the initial fault. This is where the stochasticity of *which* faults
   participate and *who triggers whom* lives.
2. **Causal jump points** (new — replaces closest-distance jump fitting). The division
   of labour is exact: **the sampled tree decides who triggers whom; the wavefront
   decides where and when.** Walk the tree in topological order. The root's wavefront
   is solved from the configured hypocentre. For each edge `P → C`, the jump pair is
   chosen **causally**:

       (p*, c*) = argmin_{p ∈ ∂P-region, c ∈ C}  [ t_P(p) + delay(‖X_P(p) − X_C(c)‖) ]

   The child's hypocentre is `c*`; its seed time is `t_P(p*) + delay(·)`; its wavefront
   is then solved with that seed. The hypocentre on a subsequent fault is where
   causality says the rupture arrives, not where the faults happen to be closest.
3. **Jump delay** is a protocol, not a formula: `JumpDelay(distance_km, context) →
   seconds`, with `Instantaneous`, `DistanceOverVelocity(v)`, and room for stochastic
   models. Configured, defaulted to `DistanceOverVelocity` with the local shear speed.
4. Each fault has exactly one triggering parent (the rupture is a tree). Fields (S4–S6)
   are sampled per segment with independent substreams; only the moment scaling in S4
   is global.

The minimisation domain in step 2 may be restricted to the near-approach region of each
pair (the closest-point machinery in `source_modelling.sources` bounds it); exhaustive
`N_P × N_C` search is an implementation choice, not part of the contract.

---

## 3. Architecture

### 3.1 One mesh type

The interchange type is an **`xarray.Dataset` per segment** — node coordinates
(`x`, `y`, `z` in the projected CRS, dims `(i, j)`) plus whatever annotation variables
the pipeline has attached so far — and an **`xarray.DataTree` for the event** (one child
per segment, the causality tree and the config in attrs). This is already the on-disk
rupture format, so "the pipeline returns an annotated mesh" and "the pipeline's output
is the file" become the same statement. A thin frozen wrapper class (`RuptureMesh`) may
carry derived-quantity methods (`areas()`, `centres()`, `spacing()`, `cell_index(hypo)`)
— methods, not stored copies.

`Fused`, `FaultGrid`, `Located`, and `SubfaultGeometry` all die. Multi-plane fusion
becomes part of S1/S2 (a property of the mesh type, not CLI glue). The projected→WGS84
conversion remains exactly one function, used by both output paths.

The **hypocentre keeps its one narrow conversion seam** (arc-length config → cell
index): `DEFECTS.md` 17 is the record of what widening it costs.

### 3.2 Stages and composition

A stage is a plain function

    stage(mesh: Dataset, params: <frozen dataclass>, rng: np.random.Generator) -> Dataset

that returns the mesh with new variables attached (functionally — no mutation). The
pipeline is one short module, `pipeline.py`, that composes S1–S9 in order; **it is the
only file where the order is written down**, and unlike `realisation::generate` the
order is a convention, not a load-bearing RNG contract.

### 3.3 Randomness

One event seed. `np.random.SeedSequence(seed).spawn(...)` gives every (stage × segment)
its own named, independent substream. Consequences:

- Draw order within the pipeline stops mattering. `skip_unused_field` and the
  "draw both Gaussians even when the amplitude is zero" invariants are deleted, not
  ported.
- Rust needs **no RNG at all**: noise is drawn in Python (numpy) and passed to kernels
  as arrays. `genslip::rng` (Pcg, ziggurat) is deleted.
- Reproducibility is per-stage: the same seed reproduces the same rupture; changing
  stage k's parameters cannot change stage j≠k's noise.

### 3.4 Field sampling — the Matérn seam

    class FieldSampler(Protocol):
        def sample(self, mesh, covariance: CovarianceSpec, rng) -> np.ndarray: ...
        def correlated_pair(self, mesh, covariance, rho, rng) -> tuple[ndarray, ndarray]: ...

- **`SpectralSampler`** (now): pad the grid, shape white noise by the spectral amplitude
  (flat below corner wavenumbers from the correlation lengths; von Kármán falloff above
  — the surviving shape, see §5; band limits **derived from the grid** as
  `2·√(dx·dy)/0.8`, never a constant), impose Hermitian symmetry, inverse FFT, take the
  fault's corner.
  This is FFTs and elementwise arithmetic — **pure numpy**, no Rust. Padding and
  even-extent rules become private details of this class; nothing outside it may know
  they exist.
- **`KernelSampler`** (later): Matérn/covariance sampling on arbitrary point sets —
  the curved-geometry path. `CovarianceSpec`'s correlation lengths map directly onto
  Matérn length scales, so the *configuration* is sampler-independent; only the
  mechanism swaps.

Correlation between fields happens **inside the sampler** (`correlated_pair`), because
the wavenumber-blend trick is spectral-specific and the GP-conditional construction is
kernel-specific; the pipeline only ever says "give me a field at correlation ρ with
that one."

### 3.5 Rust kernels

Three, each a stateless function over numpy arrays, in **one crate** (`crates/kernels`,
PyO3 directly — no separate boundary crate, no spec structs crossing the boundary):

| kernel | contract | source of the port to carry |
| --- | --- | --- |
| `eikonal_solve(slowness, spacing, seeds: [(i, j, t0)]) → travel_times` | factored fast sweeping; exact on uniform media; first-order convergent; multiple seeds | `crates/genslip/src/rupture/sweeping.rs` (a *result*, per `DEFECTS.md` 19 — genslip's own solver does not converge) |
| `synthesise_pulses(slip, rise_time, onset, shape, dt) → CSR pulses` | `∫ṡ = slip` exactly; refusal on unrepresentable rise time | `crates/genslip/src/slip_rate.rs`, shrunk (§5) |
| the `srf` crate | unchanged; 405 MiB/s parser stays measured | `crates/srf` as-is |

`crates/genslip` and `crates/core` are **deleted at the end** (Phase 6), their kernels
having been extracted. If profiling ever shows the numpy sampler or a Python stage to
be the bottleneck, it drops into `crates/kernels` behind the same signature — that is
the performance escape hatch, and it is one-directional by design.

### 3.6 Configuration

One copy: **Python dataclasses only** (mashumaro, `forbid_extra_keys` spelled
`Config` — the misspelling trap is already known). The Rust spec mirror and the
stub-vs-dataclass completeness gate die *because the class of bug they police —
disagreeing copies of the same default — becomes unrepresentable*. Kernel functions
take scalars and arrays; there is nothing left to mirror. Geometry config (lon/lat +
CRS) is unchanged from today's front end.

---

## 4. Package layout

```
crates/
  kernels/            eikonal + pulse synthesis, PyO3, array-in/array-out
  srf/                unchanged
rupture_generator/
  config/             the single copy of every parameter (geometry + rupture)
  mesh.py             RuptureMesh: the one mesh type, derived quantities, fusion,
                      the arc-length→cell hypocentre seam, projected→WGS84 (one function)
  sampling.py         FieldSampler protocol, CovarianceSpec, SpectralSampler
  moment.py           magnitude→moment (eq. 7), rigidity sampling, joint slip scaling
  stages.py           S4–S6, S8: slip, rise time, rake, onset perturbation
  timing.py           S7: speed field, geometric correction, eikonal call
  propagation.py      multi-segment: tree sampling (via source_modelling), causal
                      jump timing, JumpDelay protocol
  pulses.py           S9 driver over crates/kernels
  pipeline.py         generate(): the one statement of stage order
  formats/            mesh + rupture files (kept)
  assemble.py, srf.py SRF output path (kept, simplified against the one mesh type)
  scripts/            mesh / generate / view CLIs (kept, thinned: fuse() and
                      fault_grid() logic moves into mesh.py)
```

Target size: the survey counts ~26k lines today; this layout has a defensible budget of
roughly half that, most of the saving being the deleted mirrors, the deleted
representations, and the test rebalance in §6.

---

## 5. Kept, shrunk, deleted

**Kept (verbatim or nearly):** the three CLIs; both file formats and the viewer; the
`srf` crate; the factored-fast-sweeping solver; the geometry config and the
projected-CRS decision (`MESH.md`'s argument stands); the bounds table; `DEFECTS.md`.

**Shrunk — the physics vocabulary** (principle: production's selection survives;
everything else returns when someone asks for it, and deleting a *documented* knob is
done by refusing it loudly, not by crashing):

- **Corner relations:** `Mai` survives; `Somerville`, `Suzuki`, `Given` go. Verified
  against production config, not output (`DEFECTS.md` 11 records why output can't
  adjudicate this — the Mai/Somerville crossover at M7.37): the workflow's
  `defaults.yaml` sets `srf.kmodel: 2`, which is `MAI_FLAG` in genslip's `defs.h`,
  with `modified_corners: false` and `circular_average: false`, so no override applies.
- **Spectral shapes:** **von Kármán** (Hurst 0.75) survives — it is Mai's own falloff,
  the shape `kmodel: 2` takes. `Somerville` and `Frankel` go. The `mai_wt` /
  `somerville_wt` keys in the defaults are inert at `kmodel: 2`; they feed only the
  hybrid (`kmodel: 4`), which nothing selects.
- **Slip-rate shapes:** `OliuP2` (finite-fault production) and `delta` survive; the four
  proven aliases of `oliu_p` collapse into it as parametrisations; the rest of the
  eleven go. `from_stype` remains as the vocabulary seam and **refuses removed names
  with a message saying they were removed**, because `defaults.yaml` advertises them.
- **Point sources:** kept as what they were discovered to be — the pipeline with
  constant fields (S4–S6 replaced by constants, S7–S9 unchanged). No separate path.

**Deleted:** `crates/genslip` and `crates/core` (after kernel extraction); the spec
triplication and its completeness gate; `Fused`/`FaultGrid`/`Located`/
`SubfaultGeometry`; all RNG-stream-order machinery (`skip_unused_field`, draw-count
invariants, `Pcg`); padded-extent assertions outside `SpectralSampler`; the flat
C-failure-mode error enum (each stage refuses in its own vocabulary); `alpha_t` as a
caller-visible knob (the geometric correction is computed and applied inside S7 — the
`velocity_fraction / alpha_t` division at the config boundary was one of the four wrong
numbers).

---

## 6. Testing strategy

**The strategy is generative property testing — `proptest` in Rust, `hypothesis` in
Python — over encapsulated stages.** This is what the architecture in §3 buys: a stage
is a pure function of `(mesh, params, rng)`, so its whole input space is *generatable*,
and its contract can be asserted in aggregate over that space rather than checked on
one fixture and defended by a mutation harness. There is no `gate.sh` and no
`teeth.sh`: the gate is plain `cargo test` + `pytest` (with clippy/fmt and the
debug-vs-release agreement check as ordinary CI matrix entries, not a bespoke script),
and defect protection comes from quantified properties, not from mutating the source.

Rules, inherited and restated:

- **Properties, not values.** No test pins a number the C produced. A test asserts a
  mathematical or physical property with the reason for its tolerance written at the
  assertion.
- **Quantify over inputs.** A property holds for *generated* grids, seeds, magnitudes,
  hypocentres, and trees — shrinkable strategies, shared in one module per language
  (`proptest` strategies for kernel inputs; `hypothesis` strategies for meshes, configs,
  and segment trees). A property asserted at a single point is a smoke test, not a
  contract.
- **Rule 5 stands: a reference that re-implements the subject is not a reference.**
  Expected values come from analytic solutions, published formulas, conservation laws,
  or statistical estimators — never from a second transcription of the code under test.
- **No testing of other people's code.** No tests of xarray/zarr/pyproj/PyO3 mechanics
  beyond one round-trip per file format.
- **No property test lands without the refactor it licenses, in the same commit.**

Three tiers:

1. **Kernel contracts** (Rust, `proptest`, carried from today's best tests and made
   generative): eikonal — causality, Lipschitz bound, exactness on uniform media,
   convergence order on a gradient, multi-seed = min of single-seeds, all over
   generated grid shapes, spacings, slowness fields and seed sets; pulses — integral
   equals slip exactly, support equals rise time, refusal on the floor case, over
   generated (slip, rise time, dt) triples.
2. **Stage properties** (Python, `hypothesis`): sampler recovers its
   spectrum/correlation lengths and requested ρ (statistical, with stated estimator
   error); moment closes on the target exactly by construction and the assertion says
   so; rise–slip sample correlation ≈ ρ within estimator error; onset ≡ travel time +
   perturbation (identity); hypocentre onset pinned; **multi-segment causality: every
   child's seed time ≥ its parent's arrival at the jump-off point, onsets
   non-decreasing along every tree path, for generated trees and geometries**; the
   joint scaling makes segment moments sum to M₀. Determinism is itself a property:
   same seed ⇒ same output; different stage's params ⇒ this stage's noise unchanged.
3. **End-to-end**: one fixed-seed fixture per class (single plane, bent trace,
   two-segment jump, point source) asserting *invariants* of the output file — moment,
   causality, field statistics, format round-trip — never stored arrays.

**The defects register is covered by properties, not mutations.** Each `DEFECTS.md`
entry the rewrite could plausibly reintroduce (11, 17, 18, 21 at minimum, plus the four
wrong numbers) maps to a named generative property that would catch it — e.g. #17 is
"the hypocentre cell's onset is the minimum of the onset field" over generated
hypocentre positions; #21 is "every subfault with slip above the guard has a non-empty
pulse" over generated (rise time, dt) pairs. The property lands in the same phase as
the stage that could reintroduce the defect, and the test's docstring cites the entry.
Encapsulation is the other half of the protection: where a wrong construction can be
made unrepresentable by a type or a narrowed interface (as `Fault` did for
disconnection, and the `CornerModel`/`SpectrumShape` split did for #11), that is
preferred to any test.

---

## 7. Knowledge carried across (do not re-learn this)

From `HANDOVER.md` — none of it is C-inherited, all of it was expensive:

1. **The bounds table** (`ENGINEERING_RULES.md`): slip 1%, onset 0.05 s, rake 1°, rise
   time 1%, moment exact — each with its physical reason. It survives as the vocabulary
   for tolerances in tier-2 tests.
2. **`DEFECTS.md` as a register.** A rewrite from first principles will reproduce
   genslip's own defects by accident unless it knows them. The measured costs are the
   point: #11's crossover at M7.37; #17's off-by-one hypocentre that correlated 0.99+
   with the right answer; #18's stretched-not-shifted Frankel field with 1.63× the
   spread; #21's silently dropped one-sample subfaults carrying 0.63% of the moment.
3. **The four wrong numbers** (uncommitted on `port/stage-0-oracle-and-rng` — the fixes
   must be read from that working tree, not from this branch's base): one magnitude
   constant (eq. 7, `10.699967`); wavelength band derived from the grid, upper bound
   unbounded; the `velocity_fraction / alpha_t` division owned by the library;
   unrepresentable rise time is an error naming the subfault. All four are
   correct-by-construction in §2/§3 above, and each gets a tier-2 test.
4. **The eikonal result** (`DEFECTS.md` 19): genslip's tracker does not converge;
   factored fast sweeping is exact on uniform media and 29× closer on a gradient. The
   solver is a keeper, not a port detail.
5. **Line estimates for consolidations run over; only whole-file deletions hit their
   estimates.** Budget phases accordingly.

---

## 8. Phases

Each phase is a PR-sized unit an agent can execute; each lands green under plain
`cargo test` and `pytest` before the next starts (clippy `-D warnings`, fmt, and the
debug-vs-release agreement check are CI matrix entries, not a script). The old pipeline
keeps working until Phase 5 — the rewrite grows beside it, not through it.

**Phase 0 — scaffold and mesh type.**
New modules per §4 (empty pipelines allowed); `RuptureMesh` over `xr.Dataset` with
derived-quantity methods, multi-plane fusion, and the hypocentre seam, absorbing
`generate_cli.fuse`/`Fused`; S1–S3 (geometry → subdivided chart) working against the
existing geometry config and mesh file format.
*Accept:* mesh CLI produces today's mesh files through the new path; geometric property
tests (area sums, spacing, strike/dip, fusion seam rules) pass; `Fused` is gone.

**Phase 1 — kernels crate.**
`crates/kernels` with `eikonal_solve` (multi-seed contract) and `synthesise_pulses`
extracted from `sweeping.rs`/`slip_rate.rs`, shrunk per §5; tier-1 contracts moved over
and extended (multi-seed = min of single-seed solves).
*Accept:* contracts green in debug and release; no RNG, no spec structs, no `genslip`
dependency in the new crate.

**Phase 2 — sampling and fields.**
`sampling.py` (protocol, `CovarianceSpec`, `SpectralSampler` in numpy) and `stages.py`
S4–S6 + S8, `moment.py`, `timing.py`; randomness per §3.3.
*Accept:* tier-2 statistical tests (spectrum recovery, ρ recovery, moment closure,
rise–slip correlation, onset identity) pass; the four wrong numbers each have a test;
no code outside `SpectralSampler` mentions padding or even extents.

**Phase 3 — single-segment pipeline end to end.**
`pipeline.py` composing S1–S9; point source as constant fields; `assemble.py`/`srf.py`
recut against `RuptureMesh`; `generate` CLI switched to the new pipeline behind its
existing interface.
*Accept:* end-to-end fixtures (single plane, bent, point) pass; rupture files and SRFs
round-trip; `view` renders the output; the old Rust path is now unreachable from the
CLI.

**Phase 4 — multi-segment.**
`propagation.py`: tree sampling via `source_modelling.rupture_propagation`, causal jump
timing per §2, `JumpDelay` protocol, joint moment scaling across segments; config and
event-level `DataTree` output grown to carry multiple segments and the causality tree.
*Accept:* two- and three-segment fixtures pass the causality properties; a
single-segment config produces byte-identical behaviour to Phase 3 (the tree is
trivial); jump minimisation cost is measured before any restriction heuristic lands.

**Phase 5 — deletion.**
`crates/genslip`, `crates/core`, the config mirrors and their gates, the retired tests.
Docs recut: `README.md` describes the pipeline; `DEFECTS.md` dispositions updated;
`PORTING_RULES.md` and `SIMPLIFICATIONS.md` archived.
*Accept:* the workspace builds two crates; no import of `_core` remains; line count
reported against the §4 budget.

**Later — explicitly out of scope here:** `KernelSampler` (Matérn) and deleting S3;
curved charts in S1/S2; stochastic jump gating (the tree sampler already owns
stochasticity, so this slots into `propagation.py` without touching stages).

---

## 9. What this plan is not

It is not a port. Where this document and genslip disagree, this document wins; where
this document and a measured property disagree, the property wins and the document gets
corrected. The C is a historical reference for *what defects exist*, not for what the
code should look like — and a second reading of it is still not a reference.
