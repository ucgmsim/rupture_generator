# Engineering rules

These govern the crate now that the gate is **scientific agreement** rather than
bit-equality with `genslip v5.6.2`. They replace `PORTING_RULES.md`, which was written
for Stage 1 and said so; it is archaeology now — an explanation of why the C looks the
way it does, and why some of this code still carries its shape.

The rule that document was most worried about is the one to keep in mind here: a
rulebook that outlives its purpose calcifies. Rule 0 is that these expire too, and the
sign of it will be an argument you cannot settle by measuring something.

---

## What "the same rupture" means

Every tolerance in the suite hangs off this. Without it, "is this change acceptable?"
has no answer and every refactor becomes a matter of taste — which is the failure mode
Stage 1 ended in.

**Two ruptures are the same rupture when the fields the SRF carries agree within
bounds that make no difference to a broadband ground-motion simulation.**

| field | bound | why that number |
| --- | --- | --- |
| slip | **1%** relative | Slip enters the moment linearly and the radiated amplitude close to linearly. Realisation-to-realisation spread in a slip model is a factor of ~2; 1% is two orders inside it |
| onset | **0.05 s** | At 1 Hz — the top of the deterministic band these ruptures are used in — 0.05 s is 18° of phase. Below that, waveforms do not reorganise. It is also the coarsest `dt` in the corpus |
| rake | **1°** | The SRF stores whole degrees. The format is the floor |
| rise time | **1%** | Sets pulse duration and hence corner frequency. The scatter in the rise-time relation itself is tens of percent |
| moment | **exact**, to the f64 fold | Not an emergent quantity — it is a target the generator scales to hit. If it drifts, something is wrong rather than imprecise |

**The gap is what makes these workable.** They sit about four orders of magnitude
above numerical drift and one to two below the smallest defect ever found here:

| | slip | onset |
| --- | --- | --- |
| measured FFT-engine drift | 7e-08 | — |
| **the bound** | **1e-02** | **5e-02 s** |
| `DEFECTS.md` 18 as found | 3.9e-01 (39×) | — |
| `DEFECTS.md` 17 as found | — | 1.0 s (20×) |

So there is room to work in, and the room is measured rather than hoped for.

**One thing these bounds cannot adjudicate**: a change of *discretisation* rather than
of arithmetic. Swapping the eikonal solver replaces one first-order scheme with
another, and the two differ by their own truncation error — of order `h·log(1/h)`,
which is past the onset bound and legitimately so. Judge such a change against
**analytic truth on a problem where truth is known**, not against the code it
replaces. Say so in the commit.

---

## What a red test means

Four classes. The class tells you what to do when it fails, which is the only thing a
taxonomy is for.

| class | a red means | adjudication |
| --- | --- | --- |
| **Contract** | a bug, always | Fix the code. There is nothing to discuss — these are exact, and they are about registration, structure and guards rather than values |
| **Calibration** | you changed what a parameter *means* | Either the code is wrong or the parameter's definition moved. Both need a decision and a commit message |
| **Reference agreement** | you changed the science | Make the argument, against the bounds above. If the argument is good, re-record and say why in the commit |
| **Cost** | you made it slower | Not in `gate.sh`. Run deliberately; the gate answers questions about behaviour |

A test that does not know which class it is in is not finished.

---

## 1. Every tolerance is derived, and ships with its detection floor

A tolerance nobody can reconstruct is a tolerance nobody can defend, and it rots the
moment a grid size changes. So: compute it in the test from sample size, and write
down what it can *see*.

```rust
// Standard error of the slope is sqrt((pi^2/6) / sum((X - Xbar)^2)); z = 5 by the
// policy below. Detection floor: this resolves a change in H of about 0.05.
let tolerance = Z * ((PI * PI / 6.0) / sum_squared_deviation).sqrt();
```

`z = 5` throughout, from Bonferroni over the suite's assertion count: at ~300
statistical assertions and a family-wise false-failure rate of 1e-3, the per-assertion
two-tailed `z` is 4.65. Rounded up, and stated once here rather than argued per test.

A tolerance without a stated floor is half a specification: it says what passes and
not what would fail, so it cannot be told apart from a vacuous one.

## 2. Prefer an exact identity to a statistical test

Where a relation holds algebraically, assert the algebra. `correlate_with` sets
`t = ρ·r + √(1−ρ²)·t` elementwise, and the inverse transform is linear, so the
relation survives to the fault pointwise and exactly. Testing it with a correlation
coefficient costs several realisations and has roughly a millionth of the power.

The corollary is the one that bites: **conservation identities the code establishes by
construction are tautologies, not tests.** `Σ μ·A·s == M0` after dividing by exactly
that sum passes for any area and any rigidity. `dt·Σ pulse == slip` passes for a
boxcar. What has teeth is **registration** — `onset[hypocentre] == delay`,
`|slip| ≤ MINSLIP ⟺ no pulse`, rake measured in degrees — because those relate two
things the code does not derive from each other.

## 3. No property test lands without the refactor it licenses

Same commit. This is the mechanism that stops the suite becoming its own backlog: a
test written for a cleanup that never happens is cost with no return, and a suite
built entirely up front is the Stage 1 failure in new clothes.

## 4. Measure before you change, and record the drift

A number taken after a swap has nothing to compare against. Take it first.

Then every commit that can move a field records what it actually moved, in the commit
message, in the units of the bounds table:

```
drift: slip 7.1e-08, onset 0.0 s, rake 0 deg
```

That ledger is what turns the bounds from an assertion into evidence. If a year of
commits all sit at 1e-8, the bounds are loose enough to work in; if one sits at 8e-3,
it was worth an argument and the message says who won.

## 5. A reference that re-implements the original is not a reference

`DEFECTS.md` 17 and 18 are the same mistake twice: a parity test whose expected value
was a second transcription of `main`, written by the same reader who wrote the port,
which therefore made the same error. Both sides agreed bit for bit and both were
wrong.

Where a test cannot call the original directly, assert something about the **output**
that no shared misreading can satisfy — a field has a subfault at exactly zero, a
spread is not the configured one, an index is the argmin. Keep any transcription in
the original's own variables, and convert at one visible seam.

## 6. Draw order and draw count are still the contract

This survives Stage 1 unchanged, because it is about reproducibility rather than bit
parity. One generator feeds every stochastic field, so a routine consuming the wrong
quantity of randomness desynchronises everything after it — and the output still looks
like plausible noise.

Two things the Stage 1 version got half right:

- **Count is not order.** Swapping two stages that draw the same number of deviates
  preserves the count *and* the final seed, and changes every field downstream. Check
  per-stage, not just at the end.
- **The audit must not depend on the generator.** `GenslipLcg::seed()` works because
  the LCG's state is a bijection of its draw count; `Pcg` has no equivalent. Count
  through a decorator that wraps any `DrawSource`.

## 7. The shape is yours

`PORTING_RULES.md` rule 1 said port the physics, not the data structures, and it was
right — but it was stated defensively, as permission. State it positively now: the
crate's interfaces owe the C nothing. A function that needs rigidity takes rigidity.
Out-parameters are return values. An argument the original ignores does not exist.

This extends to the seams. `mean(field) == 1` is only observable because
`generate_normalised` and `taper_edges` are separate public functions; if fusing them
is faster and clearer, fuse them and move the test. **Do not let a test freeze a
decomposition it was only ever a witness to.**

## 8. Precision is a claim to be measured, not a shape to be preserved

Stage 1 placed every `f32` narrowing where the C stored to a `float`, and marked each
with `#[expect(clippy::cast_possible_truncation, reason = "the narrowing seam")]`.
Those seams are no longer load-bearing. Where one costs accuracy — a single-precision
fold over 10⁵ subfaults — widen it, and say what the widening bought:

> At N = 1e5, an f32 fold cannot detect fewer than about six missing subfaults. In f64
> a single missing subfault is a 3e8σ failure.

The comments must go with the constraint. A `reason = "the narrowing seam"` that no
test enforces is a lie in the source.

Widening is not free of consequence and is not automatically right: reproduce the
narrow form where the *physics* wants it, and delete the rest.

## 9. Control flow is arithmetic

Kept verbatim from Stage 1, because it is a hazard rather than a policy. Two
sequential `if`s are not `if`/`else if`:

```c
if(ix < xb)     xdamp = (ix+1)/xb;
if(ix > nx-xb)  xdamp = (nx-ix)/xb;
```

The second is *meant* to overwrite the first when both hold. Writing `else if` is
correct until the two ramps overlap and wrong after. A property test found this; a
fixed-shape test never reached it. Read every branch for whether it can be entered
twice, and every loop for whether its bounds can cross.

## 10. Decide about the reproduced defects

Stage 1 reproduced genslip's quirks and pinned each with a test, so that a later
rewrite would have to *decide* about them rather than discover them. This is that
later rewrite. `DEFECTS.md`'s "live, and reproduced" entries are now open questions,
and each one closes with an argument and a measurement:

- fix it, with the physical case and the measured effect on the corpus; or
- keep it, with the reason a user depends on it.

"It is what the C did" has stopped being a reason on its own.

## 11. Say what it computes, cite what it implements

Unchanged. A doc comment explains the physics and names what breaks if the code
changes. At most one `(orig. slip.c:NNN)` provenance line as a hook into the original
— nothing else about the C survives. Cite a published model with an equation number,
and only after reading that equation; an unchecked citation is worse than none,
because it reads as authority.
