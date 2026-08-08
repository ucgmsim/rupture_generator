# Simplifications, and what each one costs to make

Stage 1 transliterated. Where an expression in the original is written in a roundabout
way — `exp(n*log(x))` for a power, `sqrt(a*a)` for an absolute value, a constant
recomputed per grid point — the port **reproduced it** and left a note, because under
bit-parity a rewrite that moves the last bit of every grid point is indistinguishable
from a defect.

**That reservation is over.** `ENGINEERING_RULES.md` governs now, the gate is
scientific agreement, and rule 7 of the old rulebook — note what to simplify, do not
simplify yet — is reversed. What remains is the accounting: which of these are free,
which owe a measurement, and how many there actually are.

## Status

| | |
| --- | --- |
| markers in the code | **0** |
| closed | 19 |

`README.md` said eleven for a while. It was never eleven — the audit found nineteen,
four of which turned out not to be work at all, and two more of which turned out to
be free once measured.

The collection command now returns nothing, and that is the check:

```sh
rg 'SIMPLIFY:' crates/genslip/src
```

This document stops being a backlog here and becomes a record. What it is for now is
the *classification* — which kinds of rewrite are free and which owe a measurement —
because that outlives the sites, and `crates/genslip/tests/float_identities.rs`
asserts it.

### What they cost, measured

Fifteen closed in three commits, each one re-run against the corpus:

| | worst corpus drift |
| --- | --- |
| the seven transcendental rewrites — `exp(n·log x)` → `powi`/`powf`/`cbrt`, `sqrt(a²+b²)` → `hypot`, two complex multiplies → one | slip 2.3141e-06 → **2.2370e-06**; the snapshot moved one line in its eighth figure |
| the two accumulations single → double | folded into the same measurement |
| the three that "could not be unified without moving bits" | onset moved in **87 subfaults of two cases, worst 9.5e-07 s**, against a 0.05 s bound. Slip, rake, rise time and every slip-rate sample bit-identical |
| the two that were free after all — `sqrt(x*x)`, `4·atan(1)` | nothing, by construction |
| the dip pass, gather/scatter → tiled transpose | nothing, and asserted rather than measured — every 1-D transform sees the same input either way. **1.7× to 2.2× faster**, growing with the grid |

The third row deserves its detail, because two of its three sites turned out to move
nothing at all and the notes claiming otherwise were wrong:

* **`BetaProfile::beta_at`** regrouped `(a/c)*b` to `(a*b)/c`. The shipped shallow
  ramp is 2 km wide, so the division is by a power of two and exact; the mid ramp is
  flat. 200 000 sampled depths, not one bit different. At a width of 3 km, 15% of
  depths would differ.
* **`3.141592654f`** and `f32::consts::PI` are **the same `f32`** — `0x40490fdb` —
  because they first differ at the tenth digit and an `f32` carries about seven. The
  note said they differed in the last bit. They do not, and the marker was never work.
  The width is what decides: at double width the same literal is a different number,
  and so is `rperd` below, which is exact as an `f32` and wrong as an `f64`.
* **`SpeedProfile::depth_factor`** was the real one: the original runs it in `double`
  and rounds once at the store, the helper works in `f32` throughout. That extra
  rounding is the whole of the 9.5e-07 s above.

Two of three wrong is the argument for rule 4. A note that says *this moves bits* is
a prediction, and predictions get measured before they get believed.

## How to tell the two kinds apart

**Free** — provably identical, so no test can tell and none is owed:

* dead stores (a constant assigned three times, only the last read);
* dead branches (`if (j <= ny0/2)` inside a loop bounded by `j <= ny0/2`);
* dead computations whose result is never used;
* hoisting a loop-invariant out of a loop, *if* the arithmetic is unchanged;
* **`sqrt(x*x)` → `abs(x)`**, and **`4*atan(1)` → `PI``** — see below.

**Moves bits** — needs a measurement and a drift line in the commit:

* `exp(n*log(x))` → `powi`/`powf`/`cbrt`;
* `sqrt(a*a + b*b)` → `hypot`;
* changing an accumulation from single to double precision, or reassociating it;
* folding two roundings into one;
* regrouping `(a/c)*b` as `(a*b)/c`, or a `double` scale factor as a `float`.

### The classification was wrong, and being wrong is the reason to make it executable

`sqrt(x*x)` was filed under "moves bits". It does not. Under round-to-nearest with a
correctly rounded `sqrt`, `sqrt(fl(x*x)) == |x|` **exactly**, wherever `x*x` neither
overflows nor underflows: `fl(x*x)` lands within half an ulp of `x²`, and the root of
that interval contains exactly one representable value, which is `|x|` — and `|x|` is
representable because `x` is. Verified over 800k values in every evaluation order.

Three sites were mis-filed on that basis, and a fourth (`4*atan(1)`) was filed
correctly but never un-marked after being taken. So four of the nineteen were not work
at all.

The lesson is not "check harder". It is that a taxonomy nobody can execute drifts:
`crates/genslip/tests/float_identities.rs` now asserts each claim in both columns,
including the **false** ones, so a site cannot be moved between them by assertion.

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
| `shift_phase` | `pi = 4.0*atan(1.0)` | `std::f64::consts::PI`. `atan(1)` is correctly rounded to π/4 and scaling by a power of two is exact | **free — taken** |
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
| `transform_2d`, dip pass | gathered each column into scratch and scattered it back | transpose once, run contiguous rows, transpose back | moves nothing; speed only — **taken, 1.7x to 2.2x** |
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
| The constant is truncated | `rperd = 0.017453293`, 4.8e-10 **over** pi/180. Invisible in single precision — it is the nearest `f32` — and a different number in the `double` the projection actually runs in. |

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

### `genslip_v5.6.2.c` — rise time

| Site | Now | Should be | Kind |
| --- | --- | --- | --- |
| `main:2268` | `rt1_avg = sqrt(rt1_avg*rt1_avg)` | `rt1_avg.abs()`. A sign-bit mask instead of a real square root. The *behaviour* is deliberate and stays: dividing by the magnitude is what flips a negative-mean field positive | **free — taken** |
| `main:2461,2463` | `sabs = sqrt(slip*slip)` | `slip.abs()`, twice. Additionally a no-op, since slip arrives already truncated non-negative — but that is a fact about the caller, so the `abs` stays | **free — taken** |
| `main:2345` | `exp(rtime2slip_exp*log(x))` | `x.powf(rtime2slip_exp)` | moves bits |
| `main:2226,2262` | `slip_c` is inverse-transformed to get its spatial form and then forward-transformed back | the spatial field is what `reload_for_correlation` transformed in the first place; keeping it costs nothing and saves **two transforms on the padded grid**. The round trip is the identity in exact arithmetic — `dks2*dkd2*dstk*ddip` is exactly `1/N`, which cancels the unnormalised gain — so this differs only by FFT rounding | moves bits, and is a real speed win |

That last one is worth reading twice. The asymmetric normalisation flagged under
*the transform* exists precisely so a round trip is unity, and `main` uses that to
recover a value it already had.

### `genslip_v5.6.2.c` — setup

| Site | Now | Should be | Kind |
| --- | --- | --- | --- |
| `main:1440` | `sqrt((rake-90)*(rake-90))/90` | `(rake - 90.0).abs()/90.0`. The third `sqrt(a*a)` in the program, and the only one still written the long way when the classification was audited | **free — taken** |
| `main:1412` | `exp(log(Mo)/3.0)` | `Mo.cbrt()`, which is exact where a cube root lands and the exp/log pair is not | moves bits |
| `main:1249` | `bigM = log(10.0)` at run time | `f64::consts::LN_10`. Already done — the C's `log` is correctly rounded for an exact argument, so the two agree to the last bit | **free**, taken |

**Two things that look foldable and are not.** Somerville's corners are written
`0.5*mag - 1.72 - 0.79818`, two separate subtractions; `(a - b) - c` is not
`a - (b + c)`. And the offsets differ in *width* between relations — Somerville's are
inline `double` literals while Mai's and Suzuki's come from `float` getpar variables
and widen at the call. Both cost a round of test failures to find, and both are now
stated at the site.
