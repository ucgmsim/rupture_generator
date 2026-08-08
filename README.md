# rupture-generator

Kinematic rupture model generation: a Rust port of `genslip v5.6.2`, with a command
line, a self-describing output format, and a viewer.

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
was generated on and the config that produced it, which an SRF has never been able to do.

The library is still a library:

```python
import numpy as np
from rupture_generator import (
    FaultGrid, Ramp, SlipSpec, SourceSpec, SpectrumModel, TimingSpec,
    VelocityModel1D, generate_rupture,
)

rupture = generate_rupture(
    FaultGrid(24, 14, 28, 16, 1.0, 1.0,
              depth_km=depths, base_rake_deg=rakes, velocity_fraction=fractions),
    VelocityModel1D(bottom_depth_km, shear_speed_km_s, density_g_cm3),
    SourceSpec(6.8, SpectrumModel.Mai, 2.50, 1.50,
               average_dip_deg=60.0, average_rake_deg=175.0),
    SlipSpec(SpectrumModel.Mai),
    TimingSpec(rupture_time_scale=-0.35, rise_time_blend=Ramp(2.0, 1.0),
               shallow_ramp=Ramp(6.5, 1.5), deep_ramp=Ramp(17.5, 2.5),
               beta_shallow_ramp=Ramp(2.0, 1.0), beta_mid_ramp=Ramp(6.5, 1.5)),
    seed=1234, hypocentre_strike=12, hypocentre_dip=8,
)
```

## Where the fault is

The generator has never known. `assemble.py` has said since it was written that subfault
coordinates *"arrive from whoever discretised the fault, because that is the only place
that knows how the mesh became a grid"* — and nothing supplied them. `mesh` does.

**Geometry is specified in a projected Cartesian CRS** the modeller names, and the
library never leaves it. That is what makes every quantity it derives an exact identity:
areas sum to length times width, a plane's cells all report the plane's own dip. Working
on the ellipsoid instead was tried and measured — cell areas **1.4e-2** low on a 60 km
subduction interface, and a "uniform" mesh whose down-dip step varied by **6.5e-3**, the
first of those larger than the slip bound. `genslip::geodesy` is deleted; see `PRUNED.md`.

The conversion out happens once, in `rupture_generator/mesh.py`, with `pyproj` — and it
adds the **grid convergence angle** to strike, because grid north is not true north. In
NZTM that reaches **5.04°**, five times the one-degree rake bound.

A fault whose trace *bends* is one rupture: its planes are fused into a single grid whose
strike varies along it, which is genslip's `bent` case. A fault whose dip, dip direction
or width changes between planes is two surfaces that touch, and is refused by name. The
test is geometric — the planes' shared column of nodes either coincides or it does not.

`MESH.md` and `FORMAT.md` document the two file formats, including how to read them with
plain `xarray` and `pyproj`.

**Status: Stage 1 complete; Stage 2 in progress.** Against a stored corpus of six
whole ruptures, **slip, rake, onset and the slip-rate pulses all agree three orders of
magnitude inside what would matter** — nothing the corpus checks diverges.

**The point-source path is done, and it is not a port.** `generic_slip2srf` turned out
not to be a generator: it reads slip, rake and onset from a file, stretches a rise time
with depth, makes a pulse, and writes an SRF. The workflow that calls it hands it the
*same value in every row*. So a point source is the finite-fault generator with its
random fields made constant, and `generate_point_source` is exactly that — the same
assembler, the same depth ramp, the same eikonal solve, the same SRF writer, fed four
constant fields instead of four drawn ones. Four of the C's ten slip-rate shapes turned
out to be `oliu_p` with the breakpoints moved.

The gate is now **scientific agreement** rather than bit-equality with the C.
`ENGINEERING_RULES.md` says what that means: what makes two ruptures the same rupture,
and what a failing test in each class obliges you to do. The per-function parity suite
that got the port here has been retired — it had done its job, and leaving it in place
made every cleanup an argument.

The end-to-end check earned its keep immediately: it found five defects the
per-function suite structurally could not (`DEFECTS.md` 14-18). The last two are the
ones worth reading, because they are the same mistake twice. In each, the per-function
test's *reference* side re-implemented a piece of `main` — the hypocentre index
arithmetic in one case, the slip normalisation block in the other — and re-implemented
it exactly as wrongly as the port did. Both sides agreed bit for bit while both were
wrong. **A second reading of the source by the same reader is not a reference.**

## Layout

```
crates/genslip/          the port: physics and geometry, no I/O, no PyO3
crates/srf/              the Standard Rupture Format, reader and writer
crates/core/             the PyO3 boundary
rupture_generator/       the Python package
  config/                what a fault and an earthquake look like written down
  formats/               the mesh and rupture files, HDF5 or Zarr
  scripts/               the three subcommands
  mesh.py                the one seam that leaves the projected frame
  moment.py              the moment rate function
  srf.py, assemble.py    the SRF, and turning a rupture into one
tests/harness/           genslip's getpar vocabulary, for driving the reference
examples/                two geometries and a config, all of which a test runs
```

The configuration **is** the compiled core's types. Nothing in the library speaks
genslip's `getpar` names; that vocabulary lives only in `tests/harness`, which drives
the binary the port is compared against. See `tests/harness/README.md`.

An `SrfFile` is dataclasses of arrays, not tables: `points.onset_s` is one float32 per
subfault and delaying a rupture is `points.onset_s += 1`. The field names are
`GeneratedRupture`'s and `SubfaultGeometry`'s, so `assemble.to_srf_file` is a copy
rather than a translation, and every one of them carries its unit — which is how the
format's centimetres-per-second shear speed stopped being written in the velocity
model's kilometres per second.

## Gates

```sh
uv sync --extra test --extra vis --group dev   # builds the compiled extensions
./gate.sh
```

Runs clippy at `-D warnings` across all five crates, `cargo fmt --check`, the Rust
suite in **debug and release** (they must agree with each other — a disagreement
means the port depends on optimisation-level float behaviour), and pytest.

**Both extras are needed.** A bare `uv sync` drops the `test` extra and the `dev`
group, taking pytest with them, and `gate.sh`'s last stage then fails on a missing
module rather than on anything real.

`rupture_generator/*.so` is **not** tracked, so a fresh clone has to run the sync
before pytest will import anything. A committed binary goes stale silently: it
disagrees with `_core.pyi` and with the Rust it claims to be, and no gate can tell,
because pytest imports the `.so` while clippy checks source that never reached it.
Anything that changes `crates/core` needs a re-sync before pytest means anything.

There are no cargo features left. FFTW and the Fortran eikonal solver were the two
compatibility backends, both are gone, and the crate has **no system dependencies** —
`rustfft` and a fast-sweeping solver do the same jobs in pure Rust, more accurately in
the solver's case and slightly faster in the transform's.

**Nothing links EMOD3D any more.** `genslip-oracle` — a dev-only FFI to
`libgenrandv5.6.a`, used by the retired per-function parity suite — is deleted, and
with it the last `build.rs` in the workspace. It had been unwired for some time and
was kept for one stated reason: that `generic_slip2srf` was 1,450 lines of unwritten
port and per-function parity was how it would be built. That premise was wrong (see
below), so the crate had no remaining purpose.

What survives of the parity suite is in `contracts.rs`, `slip_rate_contract.rs`,
`rng_contract.rs` and `skipped_fields.rs`, restated as properties of the port rather
than as bit-equality with the C.

**Two binaries are still useful, and neither is linked.** Both are found by
environment variable, both are for *regenerating* or *re-measuring* rather than for
the gate, and the whole suite is green without either. Build them without fast-math
and without FP contraction:

```sh
cmake -B build-oracle \
  -DCMAKE_C_FLAGS="-O0 -fno-fast-math -ffp-contract=off -std=gnu17 \
                   -Wno-implicit-function-declaration -Wno-implicit-int" \
  -DCMAKE_Fortran_FLAGS="-O0 -fno-fast-math -ffp-contract=off"
```

- **`-std=gnu17` is required**, not a preference. EMOD3D declares `FILE *fopfile();`
  with an empty parameter list, which C23 reads as `(void)`, so gcc 15 refuses to
  compile `StandRupFormat/srf_subs.c` at all.
- **The Fortran `-O0` does not take.** `CMakeLists.txt` appends its own `-O2` after
  `CMAKE_Fortran_FLAGS`, and the later flag wins. `-fno-fast-math` and
  `-ffp-contract=off` do take, which are the two that decide float results.

`GENSLIP_BINARY` rebuilds the corpus; thirteen tests skip without it. And
`GENERIC_SLIP2SRF` re-runs the point-source reference comparison:

```sh
GENERIC_SLIP2SRF=/path/to/generic_slip2srf \
  .venv/bin/python -m pytest tests/harness/test_point_source_reference.py -q -s
```

That one asserts almost nothing on purpose — the port and `generic_slip2srf` differ
by design on onset and on what `risetime` means, so it prints the sizes instead. Nine
of the ten slip-rate shapes agree to **1e-6 relative**, which is the resolution of the
SRF text format rather than a tolerance; `brune` differs by the ratio of two time
constants, and that ratio is asserted so the choice is hard to undo by accident.

211 Rust tests and 784 Python tests pass with no EMOD3D build present, and
27 Python tests skip -- the thirteen that want `GENSLIP_BINARY`, the twelve that want
`GENERIC_SLIP2SRF`, and two that want a real model directory.

Two timing tests are `#[ignore]`d, because the gate answers questions about behaviour
and these answer one about cost. The SRF parser is handed multi-gigabyte files, so
before changing it, measure it:

```sh
cargo test -p srf --release -- --ignored --nocapture parse_throughput  # MiB/s, end to end
cargo test -p srf --release -- --ignored --nocapture float_leaf        # the number leaf alone
```

Currently **405 MiB/s** on `tests/srfs/rupture_1.srf` (69 MiB, 190,546 points).
`SRF_THROUGHPUT_FILE` overrides the input.

The eikonal solver and the whole pipeline are measured the same way:

```sh
cargo test -p genslip --release -- --ignored --nocapture solver_scaling
cargo test -p genslip --release -- --ignored --nocapture whole_rupture
```

The solver costs **372 to 393 ns per subfault per sweep round** from a 32x32 grid to a
1024x1024 one — flat across a 1024-fold range, which is what O(N) looks like. genslip's
solver was O(N^1.5), because it ordered each expanding ring with a selection sort; it
also declared that sort's scratch array `DIMENSION TI(400)`, so a fault more than 400
subfaults across would have read past the end of it. A whole rupture on the 24x14
fixture is 1.4 ms, of which the solve is 16%.

Thirteen tests drive the real `genslip_v5.6.2` and skip without it. Point
`GENSLIP_BINARY` at one to run them. **The corpus comparison is not among them** --
`tests/corpus/` is committed, so `test_corpus.py` runs anywhere. The binary is only
needed to *rebuild* the corpus.

## The documents, and which one answers your question

| | |
| --- | --- |
| `ENGINEERING_RULES.md` | **How the crate is written, and what makes a change acceptable.** Start here. Carries the definition of "the same rupture", the tolerance policy, and what a failing test in each class obliges you to do |
| `PORTING_RULES.md` | **Expired.** How the port *was* written under bit-parity. Archaeology: read it to understand a strange expression, not to decide whether to change one |
| `DEFECTS.md` | Eighteen defects with a disposition each: ten in the original, three in this port's PyO3 boundary, five in its call sites. The last five were found by the corpus and are the argument for having one. Every "live, and reproduced" entry is now an **open decision** rather than a settled one |
| `PRUNED.md` | What was deleted and why it was safe. Including two fields whose *draws* are consumed but whose values are not |
| `SIMPLIFICATIONS.md` | Expressions reproduced the long way, split into provably-free and bit-moving |
| `MESH.md` | **The mesh format.** Why nodes rather than cell centres, why a projected CRS rather than the ellipsoid, and how to read one with plain `xarray` |
| `FORMAT.md` | **The rupture format.** What it carries beyond an SRF, and how the ragged pulses are stored |

The two rules that cost the most to learn:

- **Precision is per expression.** C's `exp`/`log`/`sqrt` compute in `double` and
  narrow once at the store, and a `float*float` product widens at the *call*, not at
  its operands. Three separate kernels failed by one ulp on this.
- **Port the physics, not the data structures.** genslip threads a 21-field
  `pointsource` through everything; a function that needs rigidity takes rigidity.

## What is left

Ordered, and each is its own commit. **Start at item 1.** Before touching a kernel,
read `ENGINEERING_RULES.md` — in particular what makes a change acceptable and what a
failing test in each class obliges you to do. Before changing the SRF parser, measure
it.

Two habits this list assumes, both learned expensively:

- **Measure before you change, not after.** A number taken after a swap has nothing to
  compare against. Every performance and divergence figure below was recorded first.
- **When the original is silent, make it loud.** genslip exits 0 having written nothing
  if you forget `ns=1 nh=1`; the SRF stored `vs` in cm/s while the port wrote km/s.
  Both were invisible, and both are now pinned by a test that says why.

### Where this was left

**The front end is built.** `mesh`, `generate` and `view` exist, with two documented file
formats, a config schema checked against the core's own types, and 784 Python tests. What
that work turned up, because each is a thing to know rather than a thing done:

- **`DEFECTS.md` 20 is open.** `cos` silently drops any subfault whose rise time is the
  one-sample floor — three subfaults slipping 318, 269 and 209 cm on the fixture,
  carrying 1% of the moment, written to an SRF with `nt1 = 0` and indistinguishable from
  subfaults that never slipped. Found by the moment-rate integral, which every other
  shape closes to 1e-3. The fix is a choice between refusing an unrepresentable duration
  and inventing a longer pulse, and both change output.
- **Zarr does not preserve group order.** Eleven planes written `plane_0` to `plane_10`
  come back as `plane_10, plane_8, plane_5, …`, and it varies between runs. HDF5 does
  preserve insertion, so a reader trusting iteration order is green in one container and
  silently permutes the fault in the other. Both file readers key on a stored index.
- **`nzcvm`'s `forbid_extra_keys` has never worked.** mashumaro reads an inner class
  called `Config`; that codebase spells it `Meta`, which is ignored in silence. This one
  spells it `Config`, and `TestMisspellingsAreErrors` is parametrised over sections
  because a subclass adding a `Discriminator` has to inherit the settings rather than
  replace them.
- **`README.md`'s first trap still catches people.** A test comparing `rise_time_s`
  between an SRF and the native file failed by a factor of two. The SRF has no rise
  column — `srf_parser.rs:178` derives one as `nt1 * dt` — which is exactly what the trap
  says. Compare pulse lengths.

**Stage 1 is done for the finite-fault path.** The Stage 0 fixture corpus, rupture
onset and the Frankel spectrum all closed in the last three commits, and nothing the
corpus checks diverges any more. `tests/harness/mapping.py` maps every `getpar` name
onto the port's five spec groups, `tests/corpus/` holds six reference ruptures, and
`tests/harness/test_corpus.py` compares the port against them without needing a genslip
binary.

To get going:

```sh
uv sync --extra test --extra vis --group dev   # a bare `uv sync` drops pytest
export EMOD3D_BUILD_DIR=...           # an EMOD3D build, flags under "Gates" above
export GENSLIP_BINARY=...             # only needed to REBUILD the corpus
./gate.sh
```

What the comparison says today, on all six cases — measured against the bound each is
asserted at, so the headroom is visible:

| | measured | asserted at | why that bound |
| --- | --- | --- | --- |
| slip | **2.6e-06** | 1e-02 relative | below a broadband simulation's sensitivity |
| onset | **5.3e-05 s** | 5e-02 s | ~18 degrees of phase at 1 Hz |
| rake | **100% exact** after rounding | the format's own quantum | the SRF stores whole degrees |
| slip-rate pulse lengths | **100%** exact on three cases, 99.83% on `subduction` | | |
| slip-rate samples | 4.2e-05 relative where the lengths match | | |
| plane centre | genslip's flat-earth error, 43 m crustal to 1.9 km subduction | recorded, not asserted | not ours: it recomputes what the port is given |

Three orders of headroom, and the drift ledger keeps it honest: a refactor either
leaves those measurements where they are or records in its commit message what it
moved. `./teeth.sh` is the evidence the headroom did not cost anything — it puts each
of `DEFECTS.md` 14, 16, 17 and 18 back into the library and checks both suites go red.
All four are caught, narrowest margin 14x.

**Four traps that cost real time. Do not re-learn them.**

- **`nt1` is not `rise_time / dt`.** It is what the slip-rate generator *returned*.
  Comparing the port's rise time against `nt1 * dt` compares two different quantities
  and produces a bounded, systematic-looking offset in `[-2, -0.5]` samples that reads
  exactly like an off-by-one. It is not one. Compare pulse lengths.
- **`segno` is not inert when `seg_delay=0`.** genslip emits one `PLANE` per segment
  and writes points grouped by segment, so the SRF's order is not the GSF's — on the
  `bent` case by 0.18 degrees of position. Comparing in file order compares one
  subfault against another and reports it as a port defect. `corpus.segment_order` is
  the permutation, and it is the identity on the four single-plane cases, so only
  `bent` can catch its absence.
- **genslip's `ixs`/`iys` count from one, and its subfault indices count from zero.**
  They are not subfault indices at all — they exist to be handed to `wfront2d`, which
  is Fortran. Reading them as 0-based costs a whole cell in each direction, and the
  rupture that results is *plausible*: smooth, starting at zero, correlated 0.99+ with
  the right one. Nothing short of a whole-rupture comparison saw it. `DEFECTS.md` 17.
- **A reference side that re-implements the original is not a reference.** Both 17 and
  18 hid inside a parity test whose expected value was a second transcription of
  `main`, written by the same reader who wrote the port — and which therefore made the
  same mistake. They agreed bit for bit and were both wrong. Where a test cannot call
  the C directly, assert something about the **output** that no shared misreading can
  satisfy, and keep the transcription in the original's own variables so the
  conversion is visible at one seam.

### The technique that closed the last two

Both were found by **splitting a divergence into independent parts and removing one
exactly**, rather than by reading code until something looked wrong.

Onset is `travel_time + tsfac_main*perturbation`. Setting `tsfac_main = 0` on *both*
sides — zero is honoured, the sentinel being `-1.0e+15` against a `> -1.0e+10` guard —
deleted the second term and left travel times facing travel times. That showed the
perturbations already agreed bit for bit, and turned a search over the whole timing
path into a search over one function's arguments.

Frankel slip fell to a cheaper version: **look at mean, spread and correlation
separately before looking at the field.** "0.39 relative" says only "wrong somewhere".
Mean 0.98, correlation 0.993, spread **1.63** says one affine transform where another
belongs — and the slip block has exactly two.

Reach for both again in Stage 2 and Stage 3, where the gate can no longer be
bit-parity and every divergence will need decomposing before it can be argued about.

1. **Stage 2: the scientific suite that replaces bit-parity as the gate.** In progress.
   `ENGINEERING_RULES.md` is written and governs from now on; the four test classes it
   defines are being built. **Nothing in Stage 3 may land before this exists**, because
   each of those swaps changes the last bits on purpose and needs something other than
   bit-parity to adjudicate it.

   One rule from that document is worth repeating here, because it is what stops the
   suite becoming its own backlog: **no property test lands without the refactor it
   licenses, in the same commit.**

2. **Stage 3 is done.** `rustfft` replaced FFTW (measured divergence **7.06e-8**,
   recorded before the swap), `Wgs84Geodesic` replaced the flat-earth approximation
   (measured disagreement **944 m at 100 km**), and all nineteen `SIMPLIFY` sites are
   closed — `rg 'SIMPLIFY:' crates/genslip/src` now returns nothing.

   Six of the nineteen turned out not to be work at all, which is the part worth
   repeating: three `sqrt(x*x)` sites are exactly `|x|`, `3.141592654f` is exactly
   `f32::consts::PI`, and one regrouped ramp divides by a power of two. Every one had
   a note claiming it moved bits. `SIMPLIFICATIONS.md` has the tally and
   `float_identities.rs` asserts the classification so it cannot drift again.

   The largest single win was not arithmetic: `transform_2d`'s dip pass transposes
   rather than gathering each strided column, which is **1.7x to 2.2x faster** on the
   transform and bit-identical.

   **The eikonal solver is done**, and it is the one Stage 3 item that turned out to
   be a correction rather than a swap. genslip's expanding-square tracker does not
   converge: its worst error on a linear velocity gradient is 2.44e-02 at 1 km
   spacing and 2.37e-02 at 0.5 km, so refining the fault does not improve its onset
   times. Neither does the `eikonal` crate's fast marching, which was tried and
   rejected on the same measurement. Both are now replaced by factored fast sweeping
   (Zhao 2005; Fomel, Luo & Zhao 2009), which is *exact* on a uniform medium, 29x
   closer on a gradient, and converges at the expected first order.

   `DEFECTS.md` 19 carries the numbers and what the correction costs against
   genslip's own output: onset moves 1.2% to 3.6%, correlations stay above 0.9989,
   and slip and rake do not move at all.

3. **The point-source path is done**, and the premise it was scheduled under was
   wrong. It was carried here as *"~1,450 lines, untouched"* and as the one place
   `genslip-oracle` would get rewired, because that much new port needs per-function
   parity to be built at all.

   Neither half of that held. It was never 1,450 lines of port, and no per-function
   parity was needed — so `genslip-oracle` had no remaining reason to exist and is
   deleted, taking the workspace's last `build.rs` with it.

   `generic_slip2srf` does not generate anything: it
   reads `lon lat dep ds dw stk dip rake slip tinit segno` from a text file and turns
   each row into a pulse. Of its four `.c` files, 950 lines are an SRF writer this
   repo already has and a plane-header reconstruction `assemble.py` deliberately
   refuses to do. And the workflow that calls it (`realisation_to_srf.py:706-757`)
   passes **one slip, one rake and one onset**, repeated over every subfault.

   So a point source is the finite-fault generator with its random fields made
   constant. `realisation::point_source` builds four constant fields and calls the
   same `assemble` a finite fault reaches; the rise-time ramp, the fault-wide
   normalisation, the moment scaling, the eikonal solve and the SRF writer are all the
   code that was already there. Four of the ten `stype` shapes are `oliu_p` with the
   breakpoints moved, and `delta` is the spike `oliu_p` already falls back to.

   Two deliberate differences, both stated at the site: onset is **solved for** rather
   than written as one number everywhere, and `risetime` is the fault-wide **average**
   rather than an unstretched floor. At a single subfault both collapse to the C's
   answer exactly, which `point_source.rs` asserts as identities.

### Smaller things, found and done

Each was a paragraph of work, written down because the discovery cost more than the
fix. All four are closed; what they turned into is worth keeping.

- **`VelocityModel1D` was write-only from Python.** PyO3 exposed its constructor and
  no getters, so `write_velocity_model` in the harness took the three arrays and the
  caller had to remember to pass the same three to `VelocityModel1D`. It now takes the
  model, which makes *"both sides are given one model"* true by construction instead
  of by discipline. The stub-consistency test was the reason this went unnoticed for
  so long: it compared top-level names only, so a class could gain a getter in Rust
  and keep a stub that did not describe it. It now compares **members**, for every
  class, with a test asserting the class list is complete.
- **`rupture_generator/geometry.py` is deleted.** Pre-port scaffold from the initial
  commit: a `Geometry` class and a `closest_point_pair` whose bodies were `pass`, no
  importers anywhere in the repo, and not exported from the package. `assemble.py`
  named it as the source of subfault coordinates; the real contract is
  `SubfaultGeometry`, supplied by the caller, and it now says so. A mesh discretiser
  is a piece of work rather than a stub to fill in.
- **`tests/test_boundary.py`'s padded extents are genslip's now.** The helper used
  `2 * ((strike + 4) // 2 + 1)`, which gives 26 for a 20-subfault fault where genslip
  reports `nstk2 = 22`. It uses the single-plane collapse of the real rule,
  `even(int(1.10 * n))`, and `test_mapping.py` asserts that agrees with
  `mapping.padded_extents` — the general form, checked against the binary — at every
  extent from 2 to 40. One rule, one simplification, and an assertion tying them.
- **The reference binary is found by `GENSLIP_BINARY`.** This entry used to describe a
  gitignored `tools/` directory holding locally-built copies of `genslip_v5.6.2` and
  `fault_seg2gsf_dipdir`. Nothing in the repo ever referenced that path, the directory
  is not in `.gitignore`, and it does not exist here. What is true: the harness reads
  `GENSLIP_BINARY`, the thirteen tests that need it skip without it, and the corpus
  fixtures are committed — so a fresh clone runs the whole gate green with no genslip
  at all. The binary is needed only to *regenerate* the corpus.
