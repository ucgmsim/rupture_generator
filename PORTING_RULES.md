# Porting rules

These govern Stage 1, where the contract is bit-equality with genslip v5.6.2 checked
per function against `genslip-oracle`. They **expire**. When the scientific suite in
Stage 2 takes over as the gate, this document becomes archaeology — an explanation of
why the C looks the way it does — and `ENGINEERING_RULES.md` governs the crate.

That expiry is written in from the start deliberately. The predecessor project's
equivalent rulebook calcified: one rule mandated Fortran-shaped arrays "for
fidelity", and undoing it later cost thirteen commits and produced three off-by-ones,
one of which no gate caught.

---

## 1. Port the physics, not the data structures

**The most important rule here, and the one most likely to be violated by accident.**

genslip passes a `struct pointsource` — twenty-one fields covering geometry, slip,
rupture timing, material properties, asperity masks and rupture-velocity factors —
into almost every routine. Any given routine reads two or three of them.
`scale_slip_r_vsden` reads `mu` and writes `slip`. That is a god object, and porting
it as-is would import the coupling along with the arithmetic.

So:

- A function that needs rigidity takes **rigidity**, not a subfault record.
- A function that produces slip **returns** slip, rather than writing into a field
  of a shared struct.
- Out-parameters become return values. `scale_slip_r_vsden`'s `mom`/`savg`/`smax`
  trio is a struct with three named fields, not three `&mut f32`.
- Where the C's argument is meaningless — `scale_slip_r_vsden` takes a `dtop` it
  never reads — it does not appear in the port at all.

The one place a faithful mirror is required is `genslip-oracle`, where
`PointSource` exists so that `ps[i].mu` reaches the right bytes. That is a layout
requirement, not a design. Nothing in `genslip` may depend on it.

**This does not weaken the bit-equality contract.** Reshaping the *interface* is
free; only reshaping the *arithmetic* costs anything. Every function reorganised this
way is still compared against the C value for value.

## 2. Precision is per expression

The single largest source of one-ulp failures, and the rule most worth reading
carefully before writing any new kernel.

**C's `exp`, `log`, `sqrt`, `sin` and `cos` take and return `double`.** A `float`
argument is widened at the call, the whole expression runs in double, and the result
narrows to single exactly once — where it is stored. So

```c
float fac, amp;
fac = amp0/sqrt(1.0 + amp*amp);
```

is: widen, compute in double, round once. Writing the same expression in `f32`
throughout rounds at every step and gives different last bits. Every such chain in
the port is `f64` with the narrowing placed where the C assigns to a `float`, and
each narrowing carries an `#[expect(clippy::cast_possible_truncation, reason = ...)]`
naming it as a seam.

**Widen late, not early.** In `exp(ord*log(k2*lmin2))`, the product `k2*lmin2` is
`float*float` and only widens at the call. Widening the operands first — the natural
Rust spelling — rounds one time fewer and shifts the result. Same for `amp*amp` in a
`1.0 + amp*amp`. Both cost a round of test failures to find.

**Count the narrowings.** Where the C stores an intermediate into a `float` before
using it again, that is a rounding the port must also perform. The von Kármán branch
narrows twice for exactly this reason.

**Accumulators keep their precision.** `get_mean_sigma_c` and `scale_slip_r_vsden`
sum through a `float`, over ~10⁵ terms on a large fault. Reproduce it, mark it
`SIMPLIFY`, and change it under the gate — where the *more accurate* answer can be
shown to be the one that moved.

## 3. Control flow is arithmetic

Two sequential `if`s are not `if`/`else if`. genslip's edge taper reads

```c
if(ix < xb)     xdamp = (ix+1)/xb;
if(ix > nx-xb)  xdamp = (nx-ix)/xb;
```

where the second is *meant* to overwrite the first when both hold. Writing `else if`
is correct until the two ramps overlap and wrong after. A property test found this;
a fixed-shape test never reached it.

Read every branch for whether it can be entered twice, and every loop for whether its
bounds can cross.

## 4. Draw order and draw count are the contract

One generator feeds every stochastic field, so a routine that produces the right
numbers while consuming the wrong quantity of randomness desynchronises everything
after it — and the output still looks like plausible noise.

- Every differential test asserts the **final seed** alongside the values.
- Deviates are drawn where the C draws them, including at grid points where the
  amplitude is then zeroed.
- Nothing is pruned without checking what it draws. `dump_last_seed` exists for
  exactly this; see `PRUNED.md`.

## 5. Reproduce defects; refuse undefined behaviour

Known quirks are reproduced and pinned by a test of their own, so a later rewrite has
to decide about them rather than discover them. Live examples: `kfilter` removing the
DC component via IEEE infinity arithmetic; `shift_phase` restoring the DC term from a
*magnitude*, so a negative mean comes back positive; `scale_slip_r_vsden` reporting a
maximum floored at zero.

The exception is memory unsafety. `taper_slip_all_r` writes outside its array for a
taper fraction above about one; there is no way to reproduce that in safe Rust and no
reason to want to, so the port asserts instead. Record each such divergence.

## 6. Say what it computes, cite what it implements

A doc comment explains the physics and names what breaks if the code changes. It gets
at most one `(orig. slip.c:NNN)` provenance line as a hook into the original —
nothing else about the C survives.

Where the code implements a published model, cite it with an equation number, and
only after reading that equation. An unchecked citation is worse than none, because
it reads as authority.

## 7. Note what to simplify; do not simplify yet

`exp(4*ln(x))` and `x.powi(4)` agree to an ulp and disagree somewhere. Under the
exact gate, converting one is a five-minute commit with an unambiguous verdict; after
the gate is gone it is an argument. So reproduce the roundabout form, leave

```rust
// SIMPLIFY: <what it should become> -- <why it is written this way now>
```

and let `SIMPLIFICATIONS.md` collect them. That document also separates the ones that
are provably free — dead stores, dead branches, hoisting — from the ones that move
bits.
