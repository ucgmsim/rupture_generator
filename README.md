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

## Gates

```sh
./gate.sh
```

Runs clippy at `-D warnings` across all five crates, `cargo fmt --check`, the Rust
suite in **debug and release** (they must agree with each other — a disagreement
means the port depends on optimisation-level float behaviour), the suite again with
`--no-default-features`, and pytest.

`--no-default-features` builds without FFTW and without the Fortran eikonal solver.
It is the Stage 3 endpoint, checked continuously so it cannot rot.

`genslip-oracle` needs an EMOD3D build. Point `EMOD3D_BUILD_DIR` at one, built
without fast-math and without FP contraction:

```sh
cmake -B build -DCMAKE_C_FLAGS="-O0 -fno-fast-math -ffp-contract=off" \
               -DCMAKE_Fortran_FLAGS="-O0 -fno-fast-math -ffp-contract=off"
```

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

Ordered, and each is its own commit.

1. **Replace the SRF data model with dataclasses of arrays.** `SrfFile` holds pandas
   DataFrames; the Rust side already produces arrays and `GeneratedRupture` already
   looks like the target. Decide what happens to `write_hdf5` / `from_hdf5` /
   `write_sw4_hdf5` / `to_xarray`, the only pandas-dependent parts.
2. **Replace the hand-rolled scanner with `nom`.** `crates/srf/src/scanner.rs` is 242
   lines plus `lexical-core`. Semantics are not preserved for their own sake — the
   SRF suite passing is the contract. Measure: this is the hot path for a
   multi-gigabyte file.
3. **Build the Stage 0 fixture corpus.** Realisations spanning single-plane /
   multi-segment, crustal / subduction dip, small / large `nstk x ndip`, each with a
   fixed seed, storing the GSF and the argument list rather than just the SRF. Drive
   the real binary and compare parsed SRF fields column by column. This is the
   acceptance gate the per-function parity tests roll up into.
4. **The point-source path**, via `generic_slip2srf` (~1,450 lines, untouched).
5. **Stage 3**, once the scientific suite is the gate: `rustfft` for FFTW (measured
   divergence **7.06e-8**, recorded before the swap), a fast-marching eikonal solver
   for the Fortran, `Wgs84Geodesic` for the flat-earth approximation (measured
   disagreement **944 m at 100 km**), and the eleven `SIMPLIFY` sites.

Stage 2 — the scientific suite that replaces bit-parity as the gate — is designed in
the plan and not started. It is what has to exist before any of Stage 3 can land.
