# The Stage 0 fixture corpus

What the real `genslip v5.6.2` produced for six ruptures over five faults, stored so
the port can be compared against it on a machine with no genslip binary and no EMOD3D
build.

Three files per case:

| | |
| --- | --- |
| `<name>.gsf` | the geometry genslip was given |
| `<name>.args` | every `name=value` it was invoked with, sorted |
| `<name>.srf.gz` | **exactly the bytes it wrote**, gzipped |

The SRF is genslip's own output, not a re-serialisation of it. A round trip through
this package's parser and writer would launder a defect in either into the reference,
and the reference is supposed to be genslip's answer rather than ours.

`<name>.args` records the invocation in a form a person can read and `diff`. It is
descriptive: `tests/harness/corpus.py` holds the case definitions, and rebuilding
reads that rather than these files, so there is one source of truth and this is its
receipt.

## The cases, and what each is for

A fixture whose purpose is not written down gets deleted by the next person to look
at the directory. Each case exists to stop some quantity being trivially constant,
because a mapping that reads subfault zero where it should average, or hardcodes a
number that should track the grid, passes on a single uniform plane.

| case | subfaults | what it makes non-constant |
| --- | --- | --- |
| `crustal_small` | 20x12 @ 0.5 km, M6.2 | nothing — the anchor, and the same fault `test_genslip_reference.py` drives |
| `crustal_large` | 48x20 @ 1.0 km, M7.1 | the grid: different padding, and `wavelength_min` tracks the subfault size |
| `subduction` | 48x24 @ 4.0 km, M8.2 | shallow dip, deep top edge, `magC` below the magnitude so Suzuki's down-dip corner *saturates*, and `dt = 0.05` |
| `bent` | 32x14 @ 1.0 km, M6.9 | `strike_deg` and `rake_deg` **within** the grid, so `avgrak` and `alphaT` stop being every subfault's value — and the SRF's point order stops being the GSF's |
| `frankel_corners` | 24x16 @ 1.0 km, M6.6 | the corner relation, which `DEFECTS.md` 11 got wrong |

And one that is deliberately **not** a different fault:

| case | | |
| --- | --- | --- |
| `frankel_no_perturbation` | `frankel_corners` with `tsfac_main = 0` | onset with the perturbation term removed, so it is the eikonal solve alone |

Onset is a travel time plus a slip-correlated perturbation, and two wrong terms
summing to a plausible field are indistinguishable in it. Zeroing `tsfac_main` removes
the second exactly, which is how `DEFECTS.md` 17 was localised: the whole onset
divergence turned out to be in the travel times and none of it in the perturbation.

It then did the job a second time. `frankel_corners`' onset was still off after 17 was
fixed, and the twin — same fault, no perturbation, onset already exact — said that the
remaining error was arriving *through* the perturbation's correlation with slip rather
than from the solver. That is what made `DEFECTS.md` 18 a slip problem before anything
was known about it.

Both are fixed and every case now agrees, so the twin's job from here is to keep the
two terms separable: a future regression in the eikonal solve and one in the slip
field would otherwise show up in the same number. It is built with
`dataclasses.replace` from the case it twins, so "identical but for one parameter" is a
fact rather than an intention, and `test_corpus.py` checks the stored argument lists
differ in `tsfac_main` and nothing else.

`subduction`'s magnitude and area agree with genslip's own median relation. An earlier
draft asked for M8.1 on a 72x48 km fault — four magnitude units of slip on a small
plane — and every rise time came out absurd. A fixture has to be a rupture that could
happen, or the numbers it pins are of nothing.

## Rebuilding

```sh
GENSLIP_BINARY=... .venv/bin/python -m tests.harness.corpus            # everything
GENSLIP_BINARY=... .venv/bin/python -m tests.harness.corpus bent       # named cases
```

The no-argument form overwrites everything here. **Do it only when the reference
genslip changes, and say so in the commit**, because it invalidates every comparison at
once: a stored reference is a claim about that binary, built with those flags, on the
day it was recorded. The gzip member time is pinned to zero, so rebuilding an unchanged
case produces an identical file rather than a diff that is only a timestamp.

That last property makes the no-argument form a useful *check* rather than only a
rewrite: rebuild everything, and a clean `git status` says the binary in hand is the
one the stored references came from. Adding a case should still name it, so the others
are not touched at all.

The binary must be built with `-std=gnu17` — the root `README.md` says why, and it is
not a preference.
