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

**Status: Stage 1 complete for the finite-fault path, and now checked end to end.**
Every kernel is bit-identical to the C per function, verified against the real library
linked through `genslip-oracle`. Against a stored corpus of six whole ruptures, slip,
rake, onset and the slip-rate pulses all agree to the SRF's own precision. **One
divergence is left**: slip under `kmodel=Frankel`. The point-source path
(`generic_slip2srf`) is not started.

The end-to-end check earned its keep immediately: it found four defects the
per-function suite structurally could not (`DEFECTS.md` 14-17), each a correct,
C-verified function called wrongly. The fourth is the one worth reading — the
per-function test's *reference* side re-implemented genslip's index arithmetic and
made the same mistake the port did, so the two agreed exactly while both were a cell
off from genslip.

## Layout

```
crates/genslip/          the port: physics, no I/O, no PyO3
crates/genslip-oracle/   dev-only FFI to libgenrandv5.6.a -- the referee
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

Parity is checked against exactly this: 137 Rust tests pass with contraction off,
and 245 Python tests alongside them.

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
| `PORTING_RULES.md` | How the port is written. **Read rule 1 and rule 2 before touching a kernel.** Expires at the Stage 1/2 boundary |
| `DEFECTS.md` | Seventeen defects, each with a disposition and the test that pins it: ten in the original, three in this port's PyO3 boundary, four in its call sites. The last four were found by the corpus and are the argument for having one |
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
read `PORTING_RULES.md` rules 1 and 2; before changing the SRF parser, measure it.

Two habits this list assumes, both learned expensively:

- **Measure before you change, not after.** A number taken after a swap has nothing to
  compare against. Every performance and divergence figure below was recorded first.
- **When the original is silent, make it loud.** genslip exits 0 having written nothing
  if you forget `ns=1 nh=1`; the SRF stored `vs` in cm/s while the port wrote km/s.
  Both were invisible, and both are now pinned by a test that says why.

### Where this was left

The Stage 0 fixture corpus is **done**, and so is rupture onset — the last field that
did not agree. `tests/harness/mapping.py` maps every `getpar` name onto the port's five
spec groups, `tests/corpus/` holds six reference ruptures, and
`tests/harness/test_corpus.py` compares the port against them without needing a genslip
binary.

To get going:

```sh
uv sync --extra test --group dev      # both extras: a bare `uv sync` drops pytest
export EMOD3D_BUILD_DIR=...           # an EMOD3D build, flags under "Gates" above
export GENSLIP_BINARY=...             # only needed to REBUILD the corpus
./gate.sh
```

What the comparison says today:

| | |
| --- | --- |
| slip | 2.6e-06 relative, correlation 1.0000000 — except under `kmodel=Frankel` |
| rake | **100% of subfaults exact**, worst deviation 0.4999 deg (the SRF stores whole degrees, so the format is the floor) |
| onset | worst difference **5.3e-05 s** against a `%10.4f` field — half a quantum, the format's floor again. Except on `frankel_corners`, and there only by way of its slip |
| slip-rate pulse lengths | **100%** exact on three cases, 99.83% on `subduction` |
| slip-rate samples | 4.2e-05 relative where the lengths match |
| plane centre | genslip's flat-earth error, 43 m crustal to 1.9 km subduction. Not ours: it recomputes what the port is given |

Everything the corpus checks now agrees to the SRF's own precision except slip under
`kmodel=Frankel`, and the one field that trails it.

**Three traps that cost real time. Do not re-learn them.**

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

1. **Slip under `kmodel=Frankel`.** 0.39 relative on `frankel_corners`, correlation
   0.993 — the only case where slip diverges at all. Not the falloff exponent
   (`kfilt_gaus2` hardwires `beta2 = 2.0` at `slip.c:1610`, and so does the port) and
   not the corner relation (`DEFECTS.md` 11, fixed). Unexplained.

   It carries onset with it: the rupture-time perturbation is drawn correlated with
   slip at `tsfac1_scor = 0.8`, so `frankel_corners` is also the one case whose onset
   is not exact, by a spread of 0.041 s. That is a symptom rather than a second
   problem, and the `frankel_no_perturbation` twin is what proves it — the same fault
   with `tsfac_main = 0`, whose onset *is* exact.

   **Reuse the technique that closed onset.** Onset was `travel_time +
   tsfac_main*perturbation`: two unknowns summing to something plausible, and no
   diagnostic on the sum could separate them. Setting `tsfac_main = 0` on *both sides*
   removed one term exactly — zero is honoured rather than read as "unset", the
   sentinel being `-1.0e+15` against a `> -1.0e+10` guard — and left the other alone
   to be compared against its counterpart. It turned a search over the whole timing
   path into a search over one function's arguments. Look for the same seam here
   before reasoning about the whole slip pipeline at once.

2. **The point-source path**, via `generic_slip2srf` (~1,450 lines, untouched).
3. **Stage 2: the scientific suite that replaces bit-parity as the gate.** Designed in
   the plan, not started. **Nothing in Stage 3 may land before this exists**, because
   each of those swaps changes the last bits on purpose and needs something other than
   bit-parity to adjudicate it. `PORTING_RULES.md` expires at this boundary.
4. **Stage 3**, once the scientific suite is the gate: `rustfft` for FFTW (measured
   divergence **7.06e-8**, recorded before the swap), a fast-marching eikonal solver
   for the Fortran, `Wgs84Geodesic` for the flat-earth approximation (measured
   disagreement **944 m at 100 km**), and the eleven `SIMPLIFY` sites. Both those
   numbers were measured *before* the change they adjudicate, which is the only reason
   they mean anything — do the same for the eikonal solver before swapping it.

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
