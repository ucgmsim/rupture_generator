# Simplifications waiting on the gate

Stage 1 transliterates. Where an expression in the original is written in a
roundabout way — `exp(n*log(x))` for a power, `sqrt(a*a)` for an absolute value, a
constant recomputed per grid point — the port **reproduces it** and leaves a note.

That is not deference. It is that these rewrites are exactly the ones that look free
and are not: `exp(4*log(x))` and `x.powi(4)` agree to within an ulp or two and
disagree *somewhere*, and a change that moves the last bit of every grid point is
indistinguishable from a real defect once the exact gate is gone. Doing them under
bit-parity, one at a time, is cheap; doing them afterwards is an argument.

## The convention

A site marked in the code with

```rust
// SIMPLIFY: <what it should become> -- <why it is written this way now>
```

is a candidate for the first refactor round. Collect them with:

```sh
rg 'SIMPLIFY:' crates/
```

Each one is its own commit, each states whether it stayed bit-identical, and each
that did not says by how much and why the new value is the better one.

## How to tell the two kinds apart

Some of these are **free** — provably identical, so they can land under bit-parity
with no adjudication at all:

* dead stores (a constant assigned three times, only the last read);
* dead branches (`if (j <= ny0/2)` inside a loop bounded by `j <= ny0/2`);
* dead computations whose result is never used;
* hoisting a loop-invariant out of a loop, *if* the arithmetic is unchanged.

The rest **move bits** and need the gate to say by how much:

* `exp(n*log(x))` → `powi`/`powf`;
* `sqrt(a*a + b*b)` → `hypot`;
* changing an accumulation from single to double precision, or reassociating it;
* folding two roundings into one.

Do the free ones first. They shrink the code without spending any of the
adjudication budget.

---

## Found so far

### `field.rs` — the spectral kernels

| Site | Now | Should be | Kind |
| --- | --- | --- | --- |
| `WavelengthBand::divisor` | `exp(4*ln(k2*l²))` and `exp(-4*ln(k2*l²))` | `x.powi(4)` and `1.0/x.powi(4)`, or better `x²` squared once and reused — the two terms share `k2` | moves bits |
| `Spectrum2D::amplitude`, von Karman | `exp(1.75*ln(1+a)).sqrt()` | `(1.0+a).powf(0.875)` — the `sqrt` and the exponent fold together | moves bits |
| `Spectrum2D::amplitude`, Frankel | `exp(-0.5*2.0*ln(a))` | `1.0/a`, since the exponent is exactly `-1` | moves bits |
| `self_affine_field` | `exp(-0.5*(H+1)*ln(k2))` | `k2.powf(-0.5*(H+1))`, with the exponent hoisted — it is loop-invariant | moves bits |
| `WavelengthBand::divisor` | recomputed per grid point | `min_squared`/`max_squared` are loop-invariant and already hoisted; the two `ln` calls are not, but `k2` varies so they cannot be | — |

Not carried over from the C at all, and recorded here so nobody re-adds them:

* `kfilt_gaus2` assigns `hcoef` three times (2.00, 1.80, 1.75) and `beta2` twice
  (`hcoef`, then 2.0). Only the last of each is read. The port has one constant each.
* `kfilt_beta2` computes `amp0 = exp(0.5*beta2*log(lmax2))` — "amplitude when
  `k2 = 1/lmax2`" — and never uses it.
* Both kernels contain `if (j <= ny0/2) ... else ...` inside a loop declared
  `for (j = 0; j <= ny0/2; j++)`. The `else` is unreachable. The port keeps the
  general wavenumber-wrapping helper because the *strike* loop genuinely needs it.
* `kfilt_gaus2` computes the Somerville amplitude unconditionally and then
  overwrites it for four of the six models. The port computes one.

### `misc.c` — statistics

| Site | Now | Should be | Kind |
| --- | --- | --- | --- |
| `get_mean_sigma_c` | mean and variance accumulated in `float` over `ns*nd` points | accumulate in `f64`, or use a pairwise sum. On a 2827-subfault fault a single-precision sum of ~10⁵ terms loses several digits | moves bits, and is **more accurate** |
| `get_mean_sigma_c` | two passes over the grid | one pass, or Welford. The two-pass form is the numerically stable one, so this is a *speed* change only and may not be worth it | moves bits |
| `check_cor_r` | three passes, each recomputing the same `nn` | `nn` is the same in all three; compute once | free |

### Structural, not arithmetic

* `check_cor_r` computes a correlation coefficient and then only `fprintf`s it. It
  is the natural home for the Stage 2 property "the realised correlation matches the
  requested `scor`", so it should return its statistics rather than print them. Not
  ported yet for that reason.
* `kfilter` has no `k2 > 0` guard, unlike `kfilt_gaus2`. At the origin `log(0)` is
  `-inf`, so the high-cut term is `1 + exp(-inf) = 1` and the low-cut is
  `1 + exp(+inf) = inf`, giving `fac = 1/inf = 0`. It zeroes DC by relying on IEEE
  infinity arithmetic rather than by saying so. Deterministic — and now pinned by
  `band_pass_removes_the_dc_component`, so a rewrite with an explicit guard has to
  decide deliberately instead of by accident.
* `kfilt_gaus2` and `kfilter` share their band-pass expression exactly, differing
  only in whether the order is a local constant or an argument. They are one
  function in the port (`WavelengthBand::divisor_at_order`), which is worth noting
  because the same expression appears a third time inside `kfilt_beta2`.
