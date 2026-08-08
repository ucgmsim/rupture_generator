# rupture-generator

Kinematic rupture model generation: a Rust port of `genslip v5.6.2`, exposed to
Python.

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

**Status: Stage 1 complete; Stage 2 in progress.** Against a stored corpus of six
whole ruptures, **slip, rake, onset and the slip-rate pulses all agree three orders of
magnitude inside what would matter** — nothing the corpus checks diverges. The
point-source path (`generic_slip2srf`) is not started.

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
crates/genslip/          the port: physics, no I/O, no PyO3
crates/genslip-oracle/   dev-only FFI to libgenrandv5.6.a -- unwired; see below
crates/srf/              the Standard Rupture Format, reader and writer
crates/core/             the PyO3 boundary
rupture_generator/       the Python package: srf.py, assemble.py, geometry.py
tests/harness/           genslip's getpar vocabulary, for driving the reference
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
uv sync --extra test --group dev   # builds the compiled extensions
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

`genslip-oracle` needs an EMOD3D build. Point `EMOD3D_BUILD_DIR` at one, built
without fast-math and without FP contraction:

```sh
cmake -B build-oracle \
  -DCMAKE_C_FLAGS="-O0 -fno-fast-math -ffp-contract=off -std=gnu17 \
                   -Wno-implicit-function-declaration -Wno-implicit-int" \
  -DCMAKE_Fortran_FLAGS="-O0 -fno-fast-math -ffp-contract=off"
```

Two things that are not obvious:

- **`-std=gnu17` is required**, not a preference. EMOD3D declares `FILE *fopfile();`
  with an empty parameter list, which C23 reads as `(void)`, so gcc 15 refuses to
  compile `StandRupFormat/srf_subs.c` at all. This bites the `genslip_v5.6.2` binary
  the Stage 0 corpus needs; the `genrand` library the oracle links happens not to.
- **The Fortran `-O0` does not take.** `CMakeLists.txt` appends its own `-O2` after
  `CMAKE_Fortran_FLAGS`, and the later flag wins. `-fno-fast-math` and
  `-ffp-contract=off` do take, which are the two that decide float results.

**`genslip-oracle` is not linked by anything.** Not the library, not a test, not the
gate. The parity suite it existed for is retired, and what survives of it is in
`contracts.rs`, `slip_rate_contract.rs`, `rng_contract.rs` and `skipped_fields.rs` —
restated as properties of the port rather than as bit-equality with the C.

The crate stays, unwired, for one reason: `generic_slip2srf` is ~1,450 lines of port
that has not been written, and per-function parity is how it will be built.

**`EMOD3D_BUILD_DIR` is therefore not needed at all** to build the library, run the
gate, or generate a rupture. It is needed only if you rewire the oracle, or to *rebuild*
the corpus with `GENSLIP_BINARY`.

147 Rust tests and 258 Python tests pass with no EMOD3D build present.

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

**Stage 1 is done for the finite-fault path.** The Stage 0 fixture corpus, rupture
onset and the Frankel spectrum all closed in the last three commits, and nothing the
corpus checks diverges any more. `tests/harness/mapping.py` maps every `getpar` name
onto the port's five spec groups, `tests/corpus/` holds six reference ruptures, and
`tests/harness/test_corpus.py` compares the port against them without needing a genslip
binary.

To get going:

```sh
uv sync --extra test --group dev      # both extras: a bare `uv sync` drops pytest
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

3. **The point-source path**, via `generic_slip2srf` (~1,450 lines, untouched). Last —
   and the one place `genslip-oracle` gets rewired, because 1,450 lines of new port
   needs per-function parity to be built at all.

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
