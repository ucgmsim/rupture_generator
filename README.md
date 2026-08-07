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

**Status: Stage 1 complete for the finite-fault path.** Every kernel is
bit-identical to the C, verified per function against the real library linked
through `genslip-oracle`. The point-source path (`generic_slip2srf`) is not started.

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

Parity is checked against exactly this: 136 Rust tests pass with contraction off.

Two timing tests are `#[ignore]`d, because the gate answers questions about behaviour
and these answer one about cost. The SRF parser is handed multi-gigabyte files, so
before changing it, measure it:

```sh
cargo test -p srf --release -- --ignored --nocapture parse_throughput  # MiB/s, end to end
cargo test -p srf --release -- --ignored --nocapture float_leaf        # the number leaf alone
```

Currently **405 MiB/s** on `tests/srfs/rupture_1.srf` (69 MiB, 190,546 points).
`SRF_THROUGHPUT_FILE` overrides the input.

Seven tests drive the real `genslip_v5.6.2` and skip without it. Point
`GENSLIP_BINARY` at one to run them.

## The documents, and which one answers your question

| | |
| --- | --- |
| `PORTING_RULES.md` | How the port is written. **Read rule 1 and rule 2 before touching a kernel.** Expires at the Stage 1/2 boundary |
| `DEFECTS.md` | Nine defects in the original, each with a disposition and the test that pins it |
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

1. **Finish the Stage 0 fixture corpus.** The reference path works and is pinned:
   `tests/harness/gsf.py` writes the geometry file, `genslip_reference.py` renders the
   arguments, and seven tests check that the binary accepts them and that a seed
   reproduces a rupture. What is left is the half that compares:

   - ~~**Map genslip's `getpar` names onto the port's five spec groups.**~~ **Done**:
     `tests/harness/mapping.py`, pinned by `test_mapping.py`. One `Parameters` now
     renders both as the binary's command line and as the library's five groups.

     The lever that made it tractable was noticing genslip **reports what it
     derived** on stderr — `nstk2`, `ndip2`, `dstk`, `ddip`, `alphaT`, `trise_avg`,
     `rvfrac_avg` — so for those quantities the binary is the oracle and reading the
     C is not the evidence. `parse_diagnostics` reads them back.

     Four correspondences are not name-to-name, and each would have produced a
     plausible wrong rupture. Two were known; two were not:

     | | |
     | --- | --- |
     | `shypo`/`dhypo` are **km**, signed from the fault's centre and from its top edge | the port takes subfault *indices* |
     | padded extents are `nstk2`/`ndip2` = fault scaled by `extend_fac` (default **1.10**) then rounded up to even | 20x12 pads to 22x14 |
     | the slip spectrum's band is `wavelength_min`/`wavelength_max`, **not** `lambda_*` (which is roughness) | and `wavelength_max` is hardwired to `1.0e+15`, so the port's 80 km default is wrong |
     | `velocity_fraction` must carry the `alphaT` division | genslip divides both `rvfrac` and every `psrc[].rvf` by it; the port applies `alphaT` to rise time only |

     It also found three configurations the **PyO3 boundary could not spell** while
     the core could — `kmodel=Frankel` routed to the Somerville corner relation,
     `circular_average` absent entirely, and the rise-time and rupture-speed depth
     ramps collapsed into one pair. All three are `DEFECTS.md` 11-13 and **all three
     are now fixed**, each pinned by a test. The Frankel one was worth measuring: the
     two relations differ by a constant 4.3% along strike, by up to 3.6x down dip, and
     **cross at M7.37** — so a fixture near M7.4 would have shown the defect as a
     rounding difference and been believed.
   - ~~**Widen the spread**~~ **Done**: five cases under `tests/corpus/`, 2.4 MB
     gzipped, each with its GSF, its argument list and the bytes genslip wrote. See
     `tests/corpus/README.md` for what each case makes non-constant.
   - ~~**Compare the physics, and measure the geometry.**~~ **Done**:
     `tests/harness/test_corpus.py`, and it needs no binary.

     **Slip agrees** to the format's own precision — 2.6e-06 relative at worst,
     correlation 1.0000000 — on four of five cases. That is every draw, the spectrum,
     the taper and the moment scaling, in one number.

     **`rake_sigma` reaches nothing.** The rake field is normalised to the *slip*
     field's coefficient of variation, so every rake has a spread of 0.750 degrees
     where genslip gives 15.0 — exactly `slip_sigma`, on all five cases. That is
     `DEFECTS.md` 14 and it is the corpus's first find. The per-function parity tests
     could not have caught it: `rake_field` is correct and is tested with whatever
     sigma it is handed; the defect is in the **call**. Fixing it needs a boundary
     argument as well, like 11-13.

     Rise time (means 0.989-1.018), onset (correlations 0.92-0.996) and Frankel's
     slip (0.39 relative, the only case where slip diverges) are recorded and
     unexplained. Each is pinned with the number as measured, so it fails when it
     changes rather than sitting silent.

     **The geometry divergence is in the header, not the points** — which is not what
     this list predicted. genslip copies point positions straight out of the GSF, so
     they do not diverge at all; what it *derives* is each plane's top-edge centre,
     on a flat earth. That is **43 m** on a 10 km crustal plane and **1.9 km** at
     subduction scale, recorded per case.

     One trap, and only the bent case shows it: `segno` is **not** inert with
     `seg_delay=0`. genslip emits one `PLANE` per segment and writes points grouped
     by segment, so the SRF's order is not the GSF's — by up to 0.18 degrees of
     position. Comparing in file order compares one subfault against another and
     reports it as a port defect. `corpus.segment_order` is the permutation.
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
