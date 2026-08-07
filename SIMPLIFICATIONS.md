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

### `slip.c` — the spatial pipeline

| Site | Now | Should be | Kind |
| --- | --- | --- | --- |
| `shift_phase` | `pi = 4.0*atan(1.0)` | `std::f64::consts::PI`. `atan(1)` is correctly rounded to π/4 and scaling by a power of two is exact, so this one is provably identical | **free** |
| `shift_phase` | `sqrt(re*re + im*im)` | `.norm()`, which is a `hypot` and does not overflow when squaring both parts would | moves bits |
| `shift_phase` | two successive complex multiplies, one per axis | one combined factor `exp(-i(kx·sx + ky·sy))` — half the trigonometry and one rounding instead of two | moves bits |

**Looks like a simplification and is not.** `taper_slip_all_r`'s middle pass writes
`ix` and `nx-1-ix` explicitly while its first two passes test `ix < xb` and
`ix > nx-xb`. The two spellings agree while the ramps stay apart and **disagree once
they overlap**: the middle pass multiplies an overlapped cell twice, once from each
end, while the other two merely let the second condition overwrite the first. On a
14-wide fault with an 8-wide taper, cell 6 comes out at 7/8 one way and 1 the other.
Unifying them would silently pick one. Left separate until the scientific suite can
say which is right.

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

### `slip.c` — moment scaling

| Site | Now | Should be | Kind |
| --- | --- | --- | --- |
| `scale_slip_r_vsden` | `sinD = sin(dip*rperd)` computed at the top, recomputed in the `else` branch, and **never read** | delete both, and the truncated `rperd = 0.017453293` with them — it has no other use | **free** |
| `scale_slip_r_vsden` | `area` computed at the top and recomputed identically in the `else` branch | one computation | **free** |
| `scale_slip_r_vsden` | the two branches share their scale-and-summarise loop verbatim | one loop, a factor chosen before it — already so in the port | **free** |
| `scale_slip_r_vsden` | moment and average summed in `float` over every subfault | accumulate in `f64` or sum pairwise | moves bits, and is **more accurate** |
| `scale_slip_r_vsden` | takes `dtop` and never reads it | drop the argument — already dropped in the port | **free** |

**Looks like a defect.** In average-slip mode the reported moment is recomputed from
the **unscaled** field, so it describes the field before scaling rather than after.
In moment mode the answer is the target by construction, so this only shows in the
`target_savg > 0` path, which the defaults do not take. Reproduced and flagged rather
than fixed; Stage 2 should decide.

### `slip.c` — the transform

| Site | Now | Should be | Kind |
| --- | --- | --- | --- |
| `fft2d_fftw` | plans **on every call** — two `fftwf_plan_dft_1d` per 2-D transform, and the first is then leaked | plan once per (length, direction) and cache — already so in the port | **free** (and a real speed win) |
| `fft2d_fftw` | `check_realloc`s a `check_malloc`ed pointer and frees it with `fftwf_free` | one allocator throughout — already so in the port | **free** |
| `transform_2d`, dip pass | gathers each column into scratch and scatters it back | transpose once, run contiguous rows, transpose back. Cache-friendlier at realistic fault sizes — *measure before believing it* | moves nothing; speed only |
| the engine | FFTW | `rustfft`. Measured divergence between them is **7.06e-8** relative, about half an `f32` ulp | moves bits — this is the Stage 3 swap |

The engine swap is the one entry here with its adjudication baseline already
recorded: `fft_contract.rs` measures FFTW against `rustfft` now, while both are
present, so when the swap moves every field in the program there is a number saying
how much of that is the engine.

### `misc.c` / `geoproj_subs.c` — geodesy

`gcproj`, `gen_matrices`, `set_g2`, `geocen` and `latlon2km` — the whole gnomonic
projection — are called from **one place**, `genslip_v5.6.2.c:2572-2573`, inside the
roughness block. Roughness is not implemented, so none of them is ported. That is
175 lines of `geoproj_subs.c` plus the pieces of `misc.c` around them, gone with a
single upstream decision.

What survives is `set_ll`, reproduced as `geodesy::LocalFlatEarth` and replaced by
`geodesy::Wgs84Geodesic`. It is a tangent-plane linearisation with four separate
problems, worst first:

| Problem | Detail |
| --- | --- |
| It is a linearisation | Kilometres per degree are evaluated once at the origin and applied linearly. Error grows with the square of the distance. |
| The ellipsoid is not WGS84 | `a = 6378.139`, `1/f = 298.256` — roughly the 1964 IAU figure. WGS84 is `6378.137` and `298.257223563`, which is what everything downstream uses. |
| Latitude conventions are mixed | `lat0` is converted to *geocentric*, and its cosine is then used as though it were the geodetic parallel radius. |
| The constant is truncated | `rperd = 0.017453293`, about 2e-9 short. |

**Measured disagreement**, from `geodesy.rs`, at Christchurch, offset north-east:

| offset | disagreement |
| ---: | ---: |
| 1 km | 0.9 m |
| 10 km | 20 m |
| 50 km | 264 m |
| 100 km | 944 m |
| 200 km | 3.5 km |

At subfault scale the two are interchangeable, which is why the approximation
survived. At half the width of a subduction interface it is off by a kilometre. The
number is recorded before the swap so the change can be judged against it.
