# The genslip harness

**Nothing here is part of `rupture_generator`.** It exists to drive genslip
v5.6.2 — the C program the port reproduces — so the two can be compared.

The port speaks its own vocabulary: five spec groups that mirror the compiled core's
own decomposition. genslip speaks ~190 flat `getpar` names. Those names live here and
only here, because the moment they appear in the library there are two descriptions
of a rupture model and they start to drift.

| File | What it is |
| --- | --- |
| `genslip_config/` | The `getpar` parameter model, with each genslip name as an `alias` |
| `serialise.py` | Rendering that model as `name=value` arguments |
| `gsf.py` | The geometry file the binary reads, written and read back |
| `genslip_reference.py` | Running the binary, reading the SRF back, and parsing its diagnostics |
| `mapping.py` | Rendering the *same* model as the port's five spec groups |

`serialise.py` and `mapping.py` are the two halves of the comparison: one
`Parameters` becomes both the binary's command line and the library's arguments, so a
divergence between the two runs is a divergence in the physics rather than in what
each side was asked to compute. `mapping.py`'s own docstring lists the four
correspondences that are not name-to-name; `DEFECTS.md` entries 11-13 list the three
genslip configurations the PyO3 boundary cannot yet spell.

**The binary is an oracle for its own inputs.** genslip reports what it derived on
stderr before generating anything — `nstk2`, `ndip2`, `dstk`, `ddip`, `alphaT`,
`trise_avg`, `rvfrac_avg`. Those are exactly the quantities `mapping.py` has to
reconstruct, so `parse_diagnostics` turns the reference into a check on the mapping
and no transcription of the C is needed for them. It is the difference between "this
is what I read the source to mean" and "this is what the program says it did".

One catch, pinned by a test: genslip prints `mag= 6.20 median mag= 5.78` on one line,
and a diagnostic name cannot contain a space, so the second pair parses as `mag` too.
First occurrence wins, or the median magnitude silently becomes the magnitude.

The GSF reader and writer live here because only the binary reads geometry from a
file; the library takes arrays. `PRUNED.md` records the reader as deliberately not
ported. `gsf.py` also computes the four quantities genslip *derives* from a GSF --
`dstk`, `ddip`, the average dip and `dtop` -- because a caller has to pass some of
them back on the command line and they have to agree with what the binary works out
for itself.

Two things about driving the binary that cost time to find, both now pinned by
`test_genslip_reference.py`:

- **`ns=1 nh=1` are mandatory on the GSF path.** They default to -1, everything that
  computes them sits inside `if(read_erf == 1)`, and the two loops that write the SRF
  are bounded by them. Without them genslip runs the whole model, exits **0**, and
  writes a **zero-byte** file.
- **`mag`, `nstk` and `ndip` are `mstpar`**, and `nstk`/`ndip` are not in the GSF: it
  is a flat list of subfaults and does not say what shape the grid is.

Delete all of it when the comparison stops being useful — which is the same moment
`genslip-oracle` and the `fftw` feature go. (`wavefront-compat` already has.)
