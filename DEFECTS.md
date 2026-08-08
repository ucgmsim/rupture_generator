# Defects in genslip v5.6.2

`PORTING_RULES.md` §5 said reproduce defects and pin each with a test, so a later
rewrite would have to *decide* about them rather than discover them. This is that
register.

**That later rewrite is now.** Under `ENGINEERING_RULES.md` rule 10 every "live, and
reproduced" entry below is an open question, and each closes with an argument and a
measurement — fixed, with the physical case and the effect on the corpus; or kept,
with the reason a user depends on it. *"It is what the C did"* has stopped being a
reason on its own.

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
| 1 | `get_rslow_stretch`, `ruptime.c` | The high-side edge replication starts at index `count` rather than `count + offset`. Whenever the **low** side was padded — which is what happens for a hypocentre near a fault edge — the fault's own last rows and columns sit at or above `count` and are **overwritten** with values from further in. On a 12×12 fault with the hypocentre at the top edge, the deepest three rows of slowness are replaced by values three rows shallower. Reachable: the branch triggers within three subfaults of an edge, and the along-strike hypocentre distribution is tapered, not truncated. | **Unpinned since the parity suite went.** It was pinned against the C's `get_rslow_stretch`, which the port no longer links. Not re-pinned deliberately: fast marching needs no near-source analytic region, so replacing the solver deletes the padding, the replication and this defect together — see `crates/genslip/src/rupture/wavefront.rs`. Resolves by deletion rather than by argument |
| 2 | `main:2429` vs `get_rspeed_vsden2` | **Two rupture-speed depth ramps that disagree.** The solver's runs from `shal_vrup` up to 1.0 across the shallow zone. `rt_scalefac`'s writes the same transition as `rf*rvfrac*shal_vrup*(dmax1-dep)/(dmax1-dmin1)`, which runs from `shal_vrup` down to **zero** — a rupture speed of zero partway down the shallow zone, which is not a statement about anything. Latent: only consulted under the slip-and-rupture-speed weighting, and the configured weighting is uniform. Both reproduced separately rather than unified onto whichever is right. | `SpeedProfile::depth_factor` doc. Latent under the configured uniform weighting; open under rule 10 |
| 3 | `shift_phase`, `slip.c:1964` | The DC term is saved as `\|grid[0]\|` and written back afterwards. At the origin both wavenumbers are zero so the phase factor is exactly 1 and the term is already unchanged — but the saved value is a **magnitude**, so a negative DC comes back positive. | `a_negative_dc_term_is_made_positive_by_a_phase_shift` |
| 4 | `kfilter`, `slip.c:1723` | No `k2 > 0` guard, unlike `kfilt_gaus2`. At the origin `ln(0)` is `-inf`, so the high cut is `1 + exp(-inf) = 1`, the low cut is `1 + exp(+inf) = inf`, and the gain is `1/inf = 0`. It removes the DC component by relying on IEEE infinity arithmetic rather than by saying so. Deterministic. | `band_pass_removes_the_dc_component` |
| 5 | `scale_slip_r_vsden`, `slip.c:571` | In average-slip mode the reported moment is recomputed from the **unscaled** field, so it describes the field before scaling rather than after. Invisible on the default path, which matches moment instead. | **Unpinned since the parity suite went**, and reachable only on the `target_savg > 0` path that no configuration takes. Open under rule 10: either fix it and say what it changes, or record who depends on the reported number describing the unscaled field |
| 6 | `scale_slip_r_vsden`, `slip.c:525` | The running maximum starts at zero, so a field that is negative everywhere reports a maximum of zero rather than its least-negative value. Matters only before negative slip is truncated. | noted in `ScaledSlip::maximum_cm`. Reachable only before truncation, which no configured path skips. Open under rule 10 |

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

---

## Ours: call sites, found by the corpus comparison — **fixed**

A different class from 11-13. Nothing here is a function that computes the wrong
thing; each is a correct, C-verified function handed the wrong argument, or a guard
the original applies somewhere the port did not look. A suite that checks one function
at a time cannot see any of them, which is the argument for having an end-to-end
corpus at all.

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

### 17: the hypocentre was a cell off, in both directions

**Fixed.** Found by the corpus comparison; it is the whole of the onset divergence.

genslip's `ixs` and `iys` exist for one purpose — they are handed to `wfront2d`, which
is Fortran and indexes from **1**:

```fortran
ttime(IS + m*(JS-1)) = 0.0            ! wafront2d.f:31
```

```c
ixs = (int)((shypo+0.5*flen)/dstk + 0.5);   /* genslip_v5.6.2.c:3001 */
...
wfront2d_(&nstk_fd,&ndip_fd,&ixs,&iys,&dh,&nsring,fdrt,rslw,&ntot,fspace,ispace);
```

`mapping.hypocentre_indices` reproduced that formula and returned the result as the
port's `Hypocentre` — which is a 0-based subfault index, and which correctly adds one
on its way back to the Fortran. The one was therefore added twice. Every subfault
ruptured as though the hypocentre were one cell further along strike and one further
down dip.

The same confusion sat in `Padding::for_source`, which read genslip's conditions
(`ixs < nsring+1`, `ixs > nstk-(nsring+2)`) with a 0-based source. Both are now in
0-based terms with the conversion at the single seam where the Fortran is called.

| | measured |
| --- | --- |
| before | onset correlations 0.92 to 0.997, difference spreads 0.33 s to 1.05 s |
| after | **worst difference 5.3e-05 s** across the whole corpus, against a `tinit` field the SRF writes as `%10.4f` — half a quantum, which is the format's floor |

**Why it looked like physics.** Moving a hypocentre one cell does not make a rupture
implausible: the front still expands smoothly, onset still starts at zero, and the
error is a smooth function of position, so it correlates at 0.99+ with the right
answer while differing by up to a second. Every diagnostic that asked "is this the
right *shape*" said yes. It was ruled out as a stream desynchronisation for the same
reason — the stream was fine.

**Why the per-function suite could not see it.** `rupture_parity.rs` had a `padding()`
helper that transcribed the C's arithmetic, and transcribed it with the same 0-vs-1
error; it then passed the result to `oracle::wavefront_times`, which adds one, exactly
as the port does. Both sides were shifted together and matched bit for bit. That
helper is now written in genslip's own 1-based variables with a separate wrapper doing
the conversion, so the two readings can no longer drift together.

The general form: **a reference side that re-implements the original's logic is not a
reference.** It is a second reading of the same source by the same reader. Only
genslip's actual output could adjudicate this, which is what the corpus is.

### 18: a Frankel field was stretched where it should have been shifted

**Fixed.** Found by the corpus comparison; it is the whole of the Frankel slip
divergence, and the last field that did not agree.

A field drawn through any of these spectra comes out with roughly zero mean and both
signs. genslip turns that into slip in one of two ways, and **which one is a property
of the spectrum**:

```c
if(kmodel == FRANKEL_FLAG)                     /* genslip_v5.6.2.c:1809 */
   {
   slip_sigma = -1.0;
   slp_min = 1.0e+20;
   for(ip=0;ip<nstk*ndip;ip++)
      if(slip_r[ip] < slp_min) slp_min = slip_r[ip];

   slp_avg = 0.0;
   for(ip=0;ip<nstk*ndip;ip++)
      { slip_r[ip] = slip_r[ip] - slp_min; slp_avg = slp_avg + slip_r[ip]; }
   }
```

Everything else is normalised to unit mean and then **stretched** about it until the
coefficient of variation is the configured `slip_sigma` of 0.75. Frankel is **shifted**
to its own minimum instead — the least-slipping subfault becomes exactly zero, nothing
is negative — and the configured spread is then ignored. The original says that by
assigning `slip_sigma = -1.0`, which its single `if(slip_sigma > 0.0)` guard skips on
twenty lines later.

The port had no branch at all. A Frankel field was stretched like every other.

| | measured on `frankel_corners` |
| --- | --- |
| before | slip 0.39 relative, correlation 0.9934, **spread 1.63x** the reference's, means within 2% |
| after | slip **1.1e-06** relative, correlation 1.0000000 |

**Why it looked like physics.** Both operations are affine in the same generated field,
so they produce the same *pattern* — hence the 0.993 correlation and the matching mean,
which is forced by the moment scaling anyway. Only the spread differed, and a slip
field with the right pattern and too much contrast is a perfectly plausible earthquake.
The truncation of negative slip is what broke the affine relation and kept the
correlation off 1.0; a stretched field arrives at truncation with negative subfaults
and a shifted one cannot.

**Why the per-function suite could not see it, again.** `slip_pipeline.rs` builds its
reference by calling the oracle's functions in the order `main` calls them — and the
transcription omitted this stage on *both* sides. Same shape as 17, and the second time
in two commits: **a reference side that re-implements the original's logic is not a
reference.** Both tests now have the transcription written in the original's own terms
with the conversion at a visible seam, and `slip_pipeline.rs` additionally asserts
things about its *output* (`a_frankel_field_is_shifted_to_zero_and_keeps_the_spectrum_s_own_spread`)
that no shared misreading can satisfy.

This also closed the last of onset. `frankel_corners` was the one case whose onset
still diverged, by a spread of 0.041 s, because the rupture-time perturbation is drawn
correlated with slip at `tsfac1_scor = 0.8`. It was a symptom, and the
`frankel_no_perturbation` twin is what said so before the cause was known: same fault,
`tsfac_main = 0`, onset already exact.

### Nothing open

Every field the corpus checks — slip, rake, onset, the slip-rate pulses — agrees to the
SRF's own text precision on all six cases. The only divergence still recorded is
genslip's flat-earth error in the plane header, which is not the port's: it recomputes
a centre the port is given.

A note on how the onset trail went cold before 17 was found, because it cost time: the
rise-time divergence looked like a fourth defect for a while. It was not. `nt1` is what
the slip-rate generator *returned*, not `rise_time / dt`, so comparing the port's rise
time against `nt1 * dt` compares two different quantities — and produces a bounded,
systematic-looking offset in `[-2, -0.5]` samples that reads exactly like a real
defect. The comparison is now against the pulse lengths themselves.

The measurement that finally worked was subtraction rather than inspection. Onset is
`travel_time + tsfac_main*perturbation`; setting `tsfac_main = 0` on **both** sides
removes the second term exactly, leaving travel times against travel times. That
showed the perturbation fields already agreed bit for bit and the entire divergence was
upstream of them — which turned a search over the whole timing path into a search over
`Wavefront2d`'s inputs. Zero is honoured rather than read as "unset": the sentinel is
`-1.0e+15` and the guard is `tsfac_main > -1.0e+10`.

18 fell to a cheaper version of the same idea: **look at the summary statistics
separately before looking at the field.** A max relative difference of 0.39 says only
"wrong somewhere". Splitting it into mean, spread and correlation said mean 0.98,
correlation 0.993, **spread 1.63** — which is not a description of a broken pipeline,
it is a description of one affine transform where another belongs, and there are only
two of those in the slip block. The whole search was three numbers wide.

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
