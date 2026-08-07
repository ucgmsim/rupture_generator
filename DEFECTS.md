# Defects in genslip v5.6.2

`PORTING_RULES.md` §5 says reproduce defects and pin each with a test, so a later
rewrite has to *decide* about them rather than discover them. This is that register.

Scattered doc comments were doing the job until there were seven of them. Collect the
live ones with:

```sh
rg 'PORTING_RULES.md' crates/genslip/src --files-with-matches
```

**Disposition** is one of:

* **reproduced** — the port does the same thing, and a test asserts it does. Fixing it
  is a Stage 3 commit with the physical argument and the measured effect written down.
* **refused** — the port raises instead. Reserved for memory unsafety, which cannot be
  reproduced in safe Rust and should not be wanted.
* **avoided** — the surrounding code was not ported, so the defect is unreachable.

---

## Live, and reproduced

| # | Where | What | Pinned by |
| --- | --- | --- | --- |
| 1 | `get_rslow_stretch`, `ruptime.c` | The high-side edge replication starts at index `count` rather than `count + offset`. Whenever the **low** side was padded — which is what happens for a hypocentre near a fault edge — the fault's own last rows and columns sit at or above `count` and are **overwritten** with values from further in. On a 12×12 fault with the hypocentre at the top edge, the deepest three rows of slowness are replaced by values three rows shallower. Reachable: the branch triggers within three subfaults of an edge, and the along-strike hypocentre distribution is tapered, not truncated. | `the_padding_overwrites_real_slowness_and_that_is_reproduced` |
| 2 | `main:2429` vs `get_rspeed_vsden2` | **Two rupture-speed depth ramps that disagree.** The solver's runs from `shal_vrup` up to 1.0 across the shallow zone. `rt_scalefac`'s writes the same transition as `rf*rvfrac*shal_vrup*(dmax1-dep)/(dmax1-dmin1)`, which runs from `shal_vrup` down to **zero** — a rupture speed of zero partway down the shallow zone, which is not a statement about anything. Latent: only consulted under the slip-and-rupture-speed weighting, and the configured weighting is uniform. Both reproduced separately rather than unified onto whichever is right. | `SpeedProfile::depth_factor` doc; the two are separately parity-tested |
| 3 | `shift_phase`, `slip.c:1964` | The DC term is saved as `\|grid[0]\|` and written back afterwards. At the origin both wavenumbers are zero so the phase factor is exactly 1 and the term is already unchanged — but the saved value is a **magnitude**, so a negative DC comes back positive. | `a_negative_dc_term_is_made_positive_by_a_phase_shift` |
| 4 | `kfilter`, `slip.c:1723` | No `k2 > 0` guard, unlike `kfilt_gaus2`. At the origin `ln(0)` is `-inf`, so the high cut is `1 + exp(-inf) = 1`, the low cut is `1 + exp(+inf) = inf`, and the gain is `1/inf = 0`. It removes the DC component by relying on IEEE infinity arithmetic rather than by saying so. Deterministic. | `band_pass_removes_the_dc_component` |
| 5 | `scale_slip_r_vsden`, `slip.c:571` | In average-slip mode the reported moment is recomputed from the **unscaled** field, so it describes the field before scaling rather than after. Invisible on the default path, which matches moment instead. | `average_slip_scaling_matches_across_every_shape` |
| 6 | `scale_slip_r_vsden`, `slip.c:525` | The running maximum starts at zero, so a field that is negative everywhere reports a maximum of zero rather than its least-negative value. Matters only before negative slip is truncated. | noted in `ScaledSlip::maximum_cm`; "deliberately not asserted" in `moment_parity` |

## Live, and refused

| # | Where | What | Instead |
| --- | --- | --- | --- |
| 7 | `taper_slip_all_r`, `slip.c:181` | For a taper fraction above about 1 the ramp reaches past the opposite edge and the routine writes **outside its array**. | `taper_edges` asserts. There is no way to reproduce an out-of-bounds write in safe Rust and no reason to want one. The configured fractions are 0.02 and 0.0, nowhere near the bound. |

## Benign, recorded so nobody "fixes" them into something worse

| # | Where | What |
| --- | --- | --- |
| 8 | `genslip_v5.6.2.c:737` | An unterminated comment swallows `tsfac_bzero = -0.1;` and `tsfac_slope = -0.5;`. Harmless **only** because the declaration-site initialisers at 562–563 are the same values. Deleting the comment would change nothing; deleting the initialisers would change everything. |
| 9 | `genslip_v5.6.2.c:1236, 1982` | `wavelength_max` and `reload_slip` are read by `getpar` and then overwritten unconditionally. A user value for either is silently discarded. Both are absent from the port's config; see `PRUNED.md`. |

## Not genslip's, but in the same blast radius

| # | Where | What |
| --- | --- | --- |
| 10 | `workflow/.../realisation_to_srf.py:805` | The default binary path is `genslip_v5.4.2`, but the `srf:` defaults are written for v5.6.2. `getpar` never asks for names it does not recognise, so `beta_asp`, `beta_subevt`, `beta_*_depth`, `hyb_corlen_*` and `rtime2slip_exp` have been passed and **silently dropped** in production. Same class as the HF port's `calpha = -99.0`. |

## Ours: gaps in the PyO3 boundary, found by building the getpar mapping — **fixed**

Each of these was a genslip configuration `crates/genslip` modelled correctly and
`crates/core` could not spell. They were **boundary** defects, not core ones: the
physics was ported, the constructor could not express the input. None was visible
before there was a mapping, because nothing had tried to drive the port from a full
getpar set.

All three are fixed. They stay in this register because the register's job is to make
a later rewrite *decide* rather than discover, and because each is now pinned by a
test that says what it cost.

| # | Where | What | Fixed by |
| --- | --- | --- | --- |
| 11 | `crates/core/src/lib.rs`, `SourceSpec::new` | `kmodel=FRANKEL` was routed to `CornerRelation::Somerville`. genslip shares the **Mai** relation with Frankel — the branch is `if(kmodel == MAI_FLAG \|\| kmodel == FRANKEL_FLAG)` (`genslip_v5.6.2.c:1303`) — and defaults `kx_corner`/`ky_corner` for it at 994-999. Frankel's *spectral falloff* was separately correct: `kflag` and the corner relation are two distinct uses of `kmodel` (`slip.c:1651`), and they do not partition the same way. | Frankel now takes `CornerRelation::Mai`. `frankel_takes_mai_s_corners_because_one_branch_serves_both` |
| 12 | `crates/core/src/lib.rs`, `SourceSpec::new` | `circular_average` had no parameter at all; `circular` was hardwired `false` in every arm. Under Somerville it is not merely equal corners — the original switches to a third offset, 1.825 rather than 1.72 and 1.93. | A `circular_average` argument, reaching Somerville and Mai only — the two relations whose branches test it. `test_mapping.py::test_circular_average_is_expressible` |
| 13 | `crates/core/src/lib.rs`, `TimingSpec::new` | One `shallow_ramp` and one `deep_ramp` went to **both** the rise-time stretch and the rupture-speed profile. genslip reads four independent pairs: `risetimedep`/`risetimedep_range` and `deep_risetimedep`/`deep_risetimedep_range` against `shal_vrup_dep`/`shal_vrup_deprange` and `deep_vrup_dep`/`deep_vrup_deprange`. They share defaults (6.5/1.5 and 17.5/2.5), which is why every fixture passed. | Optional `shallow_speed_ramp`/`deep_speed_ramp`, falling back to the rise-time ramps so the shared-default case is unchanged. `genslip_config.RuptureVelocity` carries the getpar names, which did not exist before |

### What #11 actually cost, measured

The two relations are closer than "different power law" suggests, and that is the
point:

| | |
| --- | --- |
| along strike | a **constant 4.3%** at every magnitude. Both scale as `10^(0.5*M)`, so only the offsets differ — Mai's 2.50 against Somerville's `1.72 + 0.79818 = 2.51818`, a ratio of `10^0.01818` |
| down dip | anything from **3.6x at M4 to 0.65x at M8.5**, because the exponents differ: `10^(0.3333*M)` against `10^(0.5*M)` |
| down dip, at **M7.37** | they **cross**, and the wrong relation is indistinguishable from the right one |

That crossover is why this needed a test rather than an eyeball. A fixture near M7.4
would have shown the defect as a rounding difference and been believed.

### 14: `rake_sigma` reached nothing

**Fixed.** Found by the corpus comparison, on the first run of it.

genslip normalises the rake field to a standard deviation of `rake_sigma` **degrees**
and adds it to each subfault's base rake:

```c
sigfac = rake_sigma/rk_sig;                           /* genslip_v5.6.2.c:2068 */
rake_r[ip] = sigfac*rake_r[ip] + psrc_rake[ip];
```

`rake_sigma` defaults to 15.0 and the fixtures configure 15.0. The port calls
`slip::rake_field(..., spectrum.coefficient_of_variation)` — the **slip** field's
coefficient of variation, 0.75 and dimensionless — where a spread in degrees belongs.
The parameter is named `sigma_degrees` at the definition, so the call site is reading
it as something it is not.

| | |
| --- | --- |
| reference rake spread | 14.96 to 15.02 degrees, on all five cases |
| port rake spread | **0.750 degrees, exactly, on all five** |
| ratio | 20, which is `rake_sigma / slip_sigma` |

Exactly `slip_sigma` on every case, regardless of geometry or magnitude, is what makes
this a wiring error rather than a numerical one.

There is a second half: **no spec group carries `rake_sigma` at all**, so this cannot
be fixed in `crates/genslip` alone. Like `DEFECTS.md` 11-13 it needs a boundary
argument, and `genslip_config` already has the getpar name waiting.

**Why nothing caught it earlier.** `slip::rake_field` is correct, and its parity test
passes whatever sigma it likes. The defect is in the *call*, and a suite that checks
every function against the C one function at a time cannot see a caller handing the
right function the wrong argument. That is the seam an end-to-end corpus closes, and
it is the argument for the corpus existing.

**The fix** adds `rake_sigma_deg` to `SlipSpec` — the core's and the boundary's — since
rake is a field drawn through the same spectrum as slip and differs from it only in
normalisation. Rake now matches the reference on **100% of subfaults across all five
cases**, with the largest deviation 0.4999 degrees: the SRF stores rake as whole
degrees, so the format is the floor. Pinned by
`test_corpus.py::TestRakeAgrees`.

### 15: the shallow rise-time blend read the wrong slip field

**Fixed.** Same commit, same class.

Near the surface the original blends the rise-time field toward slip, to avoid pairing
a near-zero rise time with appreciable slip. What it blends against is `slip_c` — and
`slip_c` at that point is the **reloaded** spectrum, inverse-transformed back to space
in place at `genslip_v5.6.2.c:2225`, *after* both correlations have consumed it in the
wavenumber domain.

That is not the tapered slip field the reload was built from. The reload renormalises
the whole padded grid, zeros included, onto the original generated field's mean and
sigma, so the on-fault values come back through an affine map whose coefficients
depend on how much of the padded grid is padding. The port was blending against the
un-renormalised field.

It moves rise times on rows far below the blend zone too, because every normalisation
after the blend is global. Rise-time field correlation went from 0.89–0.95 to
0.965–0.993 on the corpus; the rest closed with #16.

### 16: a subfault that does not slip was still given a pulse

**Fixed.** Same commit.

```c
sabs = sqrt(ps[ip0].slip*ps[ip0].slip);
if(sabs > MINSLIP)  { ... generate the STF ... }   /* gslip_srf_subs.c:1496 */
else
   apval_ptr[ip].nt1 = 0;
```

`MINSLIP` is `1.0e-02` cm (`defs.h:15`). The guard is in the **loader**, outside
`gen_OliuP2_stf`, so porting the generator faithfully does not reproduce it — and
`oliu_p` was faithful. Subfaults with no slip were getting a three-sample spike where
the original writes `nt1 = 0` and a null `stf1`.

On a tapered fault that is every edge subfault: 21 of 240 on the smallest corpus case,
108 of 1152 on the largest. With the guard in place the pulse lengths match the
reference on **100%** of subfaults on three cases and 99.83% on `subduction`, and
where the lengths match the samples agree to **4.2e-05** relative — the SRF's text
precision for slip-rate rows.

### Still open, and only measured

| what | measured | what has been ruled out |
| --- | --- | --- |
| onset | correlations 0.92 to 0.997; differences with spread 0.33 s to 1.05 s, on onsets of 4 to 47 s | **not** the perturbation's amplitude — generating with `rupture_time_scale = 0` and differencing gives a perturbation whose spread is `\|tsfac_main\| * tsfac1_sigma` exactly. **Not** a desynchronised stream — the rise-time field is drawn after this one and now agrees exactly. **Not wholly** the perturbation field either — the error correlates with the port's own perturbation at only −0.43 to +0.21, and on three cases its spread is *smaller* than the perturbation's. What is left is the travel times: the eikonal solve or the speed field it runs on |
| Frankel slip | 0.39 relative on `frankel_corners`, correlation 0.993; the only case where slip diverges at all | not the falloff exponent (`kfilt_gaus2` hardwires `beta2 = 2.0` at `slip.c:1610`, and so does the port) and not the corners (fixed in 11) |

A note on how the onset trail went cold, because it cost time: the rise-time
divergence looked like a fourth defect for a while. It was not. `nt1` is what the
slip-rate generator *returned*, not `rise_time / dt`, so comparing the port's rise
time against `nt1 * dt` compares two different quantities — and produces a bounded,
systematic-looking offset in `[-2, -0.5]` samples that reads exactly like a real
defect. The comparison is now against the pulse lengths themselves.

### Related, and not a defect

Both deep ramps are additionally pushed down to the hypocentre depth per realisation
(`genslip_v5.6.2.c:2378-2381` and `:2974-2977`), each using its **own** half-width in
the adjustment. So #13 also meant the two could diverge at a deep hypocentre even with
equal configured centres, provided their half-widths differed.
`mapping.deep_ramp_centre_km` takes the centre and half-width as arguments for exactly
that reason, and `test_mapping.py` pins both cases.

---

## The pattern worth noticing

Six of the nine live entries are a value computed one way in one place and a different
way in another — two rupture-speed ramps, two taper spellings, a moment measured
before scaling and reported after, a magnitude standing in for a value. None of them
is a typo. They are what happens when the same physical idea is written out by hand
each time it is needed.

### What was done about it

`DepthRamp` is now one type with three named operations, and the linear
interpolations that agree with it use it:

| Site | Uses |
| --- | --- |
| the rise-time shallow blend | `weight` |
| `RiseTimeStretch::factor_at` | `scaled_from_deep` / `scaled_from_shallow` |
| the weighting rupture speed | `scaled_from_deep` / `scaled_from_shallow` |

`RiseTimeStretch` itself replaced **two** identical hand-written copies — one
computing the fault-wide normalisation constant (`genslip_v5.6.2.c:2429-2453`), one
setting each subfault's duration (`gslip_srf_subs.c:1498-1508`) — confirmed branch for
branch before merging, and bit-parity held.

Two sites keep their own arithmetic, each marked `SIMPLIFY` at the site with the
reason:

| Site | Why the helper does not fit |
| --- | --- |
| `SpeedProfile::depth_factor` | Its scale factor `1.0 - shal_vr` is a **double**, because the literal makes it one. The helper's is single, so the multiply happens at a different width. |
| `BetaProfile::beta_at` | It precomputes a gradient, `(v_far - v_near)/width`, and multiplies by the offset — `(a/c)*b` where every other site writes `(a*b)/c`. |

The grouping is the reason there are only two exceptions rather than six. `*` and `/`
have equal precedence in C and associate left to right, so `scale * (far - depth) /
width` multiplies *before* it divides — and the helper does the same. Writing the
mathematically identical `scale * ((far - depth) / width)` would have matched nothing.
