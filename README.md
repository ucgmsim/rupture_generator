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
means the port depends on optimisation-level float behaviour), the suite again with
`--no-default-features`, and pytest.

**Both extras are needed.** A bare `uv sync` drops the `test` extra and the `dev`
group, taking pytest with them, and `gate.sh`'s last stage then fails on a missing
module rather than on anything real.

`rupture_generator/*.so` is **not** tracked, so a fresh clone has to run the sync
before pytest will import anything. A committed binary goes stale silently: it
disagrees with `_core.pyi` and with the Rust it claims to be, and no gate can tell,
because pytest imports the `.so` while clippy checks source that never reached it.
Anything that changes `crates/core` needs a re-sync before pytest means anything.

`--no-default-features` builds without FFTW and without the Fortran eikonal solver.
It is the Stage 3 endpoint, checked continuously so it cannot rot.

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

**`genslip-oracle` is no longer wired into anything the gate runs.** The eight
per-function parity files that used it are gone; what survives of them is in
`contracts.rs`, `slip_rate_contract.rs` and `rng_contract.rs`, restated as properties
of the port rather than as bit-equality with the C. Four files still link it
(`slip_pipeline`, `correlated_fields`, `skipped_fields`, `geodesy`) and are next.

The crate itself stays, unwired, for one reason: `generic_slip2srf` is ~1,450 lines of
port that has not been written, and per-function parity is how it will be built.

`EMOD3D_BUILD_DIR` is still needed regardless, because `wavefront-compat` gates the
only `EikonalSolver`. That is what makes replacing the solver the first Stage 3 item
rather than the last.

144 Rust tests and 258 Python tests pass in this configuration.

Two timing tests are `#[ignore]`d, because the gate answers questions about behaviour
and these answer one about cost. The SRF parser is handed multi-gigabyte files, so
before changing it, measure it:

```sh
cargo test -p srf --release -- --ignored --nocapture parse_throughput  # MiB/s, end to end
cargo test -p srf --release -- --ignored --nocapture float_leaf        # the number leaf alone
```

Currently **405 MiB/s** on `tests/srfs/rupture_1.srf` (69 MiB, 190,546 points).
`SRF_THROUGHPUT_FILE` overrides the input.

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

2. **Stage 3**, once the scientific suite is the gate: the `eikonal` crate for the
   Fortran solver **first**, because `wavefront-compat` gates the only `EikonalSolver`
   and until it is replaced `--no-default-features` compiles but cannot generate a
   rupture at all; then `rustfft` for FFTW (measured divergence **7.06e-8**, recorded
   before the swap), `Wgs84Geodesic` for the flat-earth approximation (measured
   disagreement **944 m at 100 km**), and the **fifteen** remaining `SIMPLIFY` sites.
   Four of the original nineteen were never work: three were mis-filed as bit-moving
   when `sqrt(x*x)` is provably exactly `abs(x)`, and one had been taken and never
   un-marked. `SIMPLIFICATIONS.md` has the audit and
   `crates/genslip/tests/float_identities.rs` makes it executable.

   Both those numbers were measured *before* the change they adjudicate, which is the
   only reason they mean anything. The eikonal swap is the exception to the whole
   scheme: it changes the *discretisation* rather than the arithmetic, so it is judged
   against analytic truth on a problem where truth is known, not against the solver it
   replaces. `ENGINEERING_RULES.md` says why.

3. **The point-source path**, via `generic_slip2srf` (~1,450 lines, untouched). Last —
   and the one place `genslip-oracle` gets rewired, because 1,450 lines of new port
   needs per-function parity to be built at all.

### Smaller things, found and not done

Each is a paragraph of work, and each is written down because the discovery cost more
than the fix will.

- **`VelocityModel1D` is write-only from Python.** PyO3 exposes its constructor and no
  getters, so `write_velocity_model` in the harness takes three arrays rather than the
  model a caller already built. Add `#[pyo3(get)]`, the stub entries, and let the
  stub-consistency test in `tests/test_boundary.py` check them.
- **`rupture_generator/geometry.py` is a stub.** `Geometry.discretise` and
  `closest_point_pair` are `pass`, and `DiscretisedGeometry` now has no callers — the
  harness grew its own `GsfSubfaults` because `DiscretisedGeometry` has no longitude or
  latitude to write. Decide what that module is for before something depends on it.
- **`tests/test_boundary.py`'s padded extents are not genslip's.** Its helper uses
  `2 * ((strike + 4) // 2 + 1)`, which gives 26 for a 20-subfault fault where genslip
  reports `nstk2 = 22`. The tests pass because `FaultGrid` takes the padding as an
  argument, so this misleads a reader rather than breaking anything. genslip's actual
  rule is now written down and checked against the binary —
  `mapping.padded_extents` — so this is a two-line change to use it.
- **`tools/` holds locally-built binaries.** `genslip_v5.6.2` and
  `fault_seg2gsf_dipdir` there were rebuilt from `build-oracle` with the flags above.
  The directory is gitignored, so on another machine they will not exist.
