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
| `genslip_reference.py` | Running the binary and reading the SRF back |

A GSF *writer* belongs here too when the Stage 0 corpus needs one: the binary reads
its geometry from a file, and the library does not.

Delete all of it when the comparison stops being useful — which is the same moment
`genslip-oracle`, the `fftw` feature and the `wavefront-compat` feature go.
