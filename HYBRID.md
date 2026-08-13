# Curved geometry, planar solvers

## What this proposes

Keep the Monge-patch mesh built in `MESH.md`'s Component 1. Delete the two solvers built
for it — meshFIM and the Whittle–Matérn SPDE — and restore the factored fast sweep
(`crates/kernels/src/eikonal.rs`) and circulant embedding (`sampling.py`,
`crates/kernels/src/field.rs`), both of which run on a regular lattice over the
**parameter plane**. Project their results onto the curved mesh's faces.

The geometry's job changes. It stops being the domain the solvers run on and becomes the
thing that supplies **true areas, true depths, true positions and the true outline** —
which is where, on the measurements below, almost all of the value turned out to be.

```
                      keeps                         restores
    X(u,v) = O + u e_u + v e_v + h n     |    factored sweep on (i, j)
    true areas -> moment                 |    circulant embedding on (i, j)
    true depths -> rigidity, speed       |
    outline, taper, hypocentre seam      |    projection back onto faces
```

- **delete** ~9,700 lines: `triangular/fim.py`, `triangular/spde.py`, `crates/kernels/src/fim.rs`, and their tests
- **restore** ~1,300 lines already written and tested: `sampling.py`, `field.rs`, `eikonal.rs`
- **keep** ~5,500 lines: `triangular/mesh.py`, `triangular/gocad.py`, `triangular/pipeline.py`

## What the curvature study settles

`curvature/` measured a **flat** model — plane geometry throughout — against a curved one,
sharing mesh topology and white noise so every difference was geometry. This proposal is
**not** that flat model. It keeps areas and depths curved, which removes the two largest
error terms the study found before they arise.

| term the study measured | flat model | this proposal |
| --- | --- | --- |
| onset, median (Hikurangi) | +7.53 s | **−0.14 to −0.17 s** |
| onset, p05 / worst | — | **−0.95 to −1.9 s / −6.0 s** |
| moment early by more than 0.5 s | — | **15–33%** |
| moment delivered / target | 0.9690 | **1.0** — true areas |
| rigidity contribution | 0.9384 | **1.0** — true depths |
| correlation length, along strike | +8.7% | +8.7%, unresolved (below) |

This is the study's own true-depth counterfactual — flat paths, true depths, true areas —
independently reproduced to four decimals by the outline study (medians −0.1377 / −0.1743
/ −0.1559 s against the study's −0.1373 / −0.1736 / −0.1556 s).

**The median understates it.** The constant-velocity control's −0.075 s is a median over a
distribution with a long tail: the fifth percentile runs −0.95 to −1.9 s, the worst cell
−6.0 s, and **15–33% of the moment arrives more than half a second early** at every
hypocentre on both interfaces. Against a ~0.35 s deliberate onset perturbation and a
190 s rupture that is defensible, but it is the number this decision turns on, and quoting
the median alone makes the projection look three to four times better than it is.

**The restored solver is second-order where the one it replaces is first.** `fim.py` is
first-order by deliberate choice (`MESH.md` Component 3), measured against the factored
sweep at a systematic reaching **+116 ms** at 500 m and growing with distance from the
source. The projection costs 75 ms. So the hybrid is not obviously less accurate in
timing; it may be more so, and the two errors have opposite signs.

**Magnitude does not enter.** Every travel-time statistic is bit-identical between Mw 8.5
and magnitudes taken from each interface's own area (9.11 Hikurangi, 8.67 Puysegur), on
all three surfaces, with zero mismatches. The eikonal has no channel through which
magnitude reaches it, so a timing argument made at one magnitude holds at any.

## The one question the evidence does not settle

Circulant embedding delivers correlation lengths measured in **planar** distance; the SPDE
delivers them in **surface** distance. On Hikurangi that is 8.7% along strike and 4.0%
down dip, of which only 0.5–0.9% is the projection stretch — the rest is that the
Laplace–Beltrami operator on the real surface delivers a shorter length than the same
operator on its shadow.

Which is *right* depends on what Mai & Beroza's relations mean, and the data does not say:

- **For the plane.** Mai & Beroza (2002) fitted 44 **finite-source inversions on planar
  faults**. "Along strike" in equations (4) and (5) is a planar strike, and the two
  definitions coincide on the sources the relations were regressed from. Reading them with
  a surface-intrinsic metric applies the numbers outside the geometry they were measured
  in.
- **For the surface.** The fitted frame's azimuth is a property of *how the patch is cut*,
  not of the surface: splitting Hikurangi in two moves it from 218.5° to 211.6° and
  223.7°. A segmented interface therefore gets a 12–17° discontinuity at a seam for no
  physical reason, worth 10–25% in delivered along-strike length.

The second argument bites only when one surface is split into segments. The first is a
statement about what the source model is entitled to claim. **This is a modelling
decision, not a numerical one, and it should be made explicitly rather than inherited from
whichever sampler is installed.**

## The outline problem: real, and fixed by a slowness wall

A lattice eikonal has no concept of "not fault". The factored sweep runs on a full
rectangular `(i, j)` grid, and a subduction interface's parameter footprint is not convex
— Hikurangi is 10.3% concave against its hull, Puyseguer 14.2%, and a rectangular lattice
wastes 37–47% of its cells. The front can therefore arrive on the far side of a concavity
earlier than one constrained to the surface.

Measured, at fault cells, isolating it exactly by differencing two solves that share
solver, grid and seed and differ only in whether off-fault cells are walled:

| | median | worst cell | moment early by >0.5 s |
| --- | --- | --- | --- |
| Hikurangi | 0.000 s | −1.31 s | **0.0007%** |
| Puyseguer | 0.000 s | −3.68 s | **4.04%** |

Hikurangi is unaffected — the 84 affected cells sit trench-ward at 6.1 km depth, inside
the slip taper, carrying essentially no moment. Puyseguer is not: 99.97% of its affected
cells lie in the outer fifth of strike, beyond a re-entrant bay that nearly severs a deep
south-eastern lobe, at a median depth of 27.9 km — **squarely in the seismogenic zone**,
costing 4.85% of seismogenic moment.

**Two corrections to the diagnosis above, both of which matter.**

*The mechanism is not short-cutting.* Refilling off-fault cells with the **slowest**
on-fault slowness — geometry untouched, no fast lane possible — bounds pure geometric
short-cutting at **0.76 s** on both interfaces, touching under 0.03% of cells. Around 80%
of the Puyseguer error is instead that a rectangular lattice **must invent a medium off
the fault**, and nearest-neighbour fill copies deep fast rock (0.307 s/km) into the bay
while the on-fault detour runs through shallow slow rock (up to 0.51 s/km). The front
routes through ground that is both absent *and* quick. The fill rule — an arbitrary
implementation choice — changes the answer by a factor of five, and concavity does not
predict it.

*Concavity is not the control variable either.* On a swept re-entrant notch cut into
Hikurangi's deep edge, deleting 2% of the area moves concavity 3.6 points and costs 22 s;
9.7% moves it 8.7 points and costs 119 s against a 264 s rupture. The real criterion is
**topological**: the error is negligible while the notch stays deeper than the
hypocentre's own dip line — the front simply passes above it — and explodes once the notch
crosses that line and forces a detour. Two outlines with identical concavity differ by two
orders of magnitude depending on where the hypocentre sits.

**The wall works, and the concern about the factorisation was unfounded.** Masking
off-fault cells with a slowness wall was measured directly rather than argued from
`eikonal.rs`'s fallback warning:

- **Interior accuracy is bit-identical with and without it.** On a uniform medium with the
  exact answer `s·r`, maximum error 1.3e−13 s, and identical at wall factors of 10, 10³,
  10⁵ and 10⁸. Fomel et al.'s `τ ≡ 1` survives the wall exactly; the fallback is not
  triggered.
- **It stops the front rather than slowing it.** On an L-shaped fault where the geodesic
  must round a reflex corner, the walled solve matches the closed-form
  around-the-corner distance to 0.099 s over a 20 s traverse, where the unwalled solve
  short-cuts by 0.71 s median and 2.23 s worst.
- **It saturates at ×10.** `max|M(×10) − M(×10⁵)| = 0.0 s` exactly, on both interfaces and
  at every notch penetration. No leakage, no need for a large factor.
- **It costs 1.0–1.5×** the open solve — 2.2–2.7 s against 1.8–2.2 s on Hikurangi's
  1,650 × 664 lattice, against ~10 s for one meshFIM solve on 1.39 M faces.

So the outline problem is a masking rule, not an architectural obstacle. But note what it
concedes: **a mask is a per-cell decision, so the lattice does now carry the outline**, and
the boundary bookkeeping has to be budgeted rather than assumed away.

### What this leaves as the binding constraint

After masking, what remains is the **metric error** — planar path lengths — at −0.14 to
−0.30 s median, −6 s worst, touching 15–33% of moment. That is three to four times larger
than anything the outline does on either real interface, no mask removes it, and it is
therefore the quantity the decision should turn on. The outline question was never the
binding constraint; it only looked like one.

For scale, the *floor* of this comparison is itself large: the walled lattice and the
outline-respecting mesh solver differ by a median +0.08 to +0.14 s and up to +0.94 s,
being first-order meshFIM with geodesic-ball seeding against second-order factored
sweeping with a point seed. On Hikurangi that floor is **twenty times larger than the
outline error it would be hiding**.

### One case not measured

All three study hypocentres sit at `v = 180 km` and see 99.4–99.7% of the fault in
straight line. A hypocentre placed in Hikurangi's own deep south-western spur — behind its
210 km interior gap — is the configuration the topological criterion above says would be
worst, and no measurement was made of it. It should be, before anyone relies on
Hikurangi's clean result generalising.

## What stops being load-bearing

Several things built during the migration exist because of the two solvers being removed,
and their justifications go with them:

- **Multigrid** (`spde.subdivided`, `_VCycle`, `_IterativeSolver`) — built because the SPDE
  needed sparse solves at 10⁶ vertices. Circulant embedding is an FFT.
- **The surface anisotropy frame** — built because `H` had to be expressed on the surface.
  A lattice has no such question.
- **Neumann padding** (`padded_builder`, `spde.padded_operator`) — the reflection is a
  property of solving a PDE on a bounded domain. Circulant embedding pads and crops by
  construction, so **the boundary problem retires entirely**, along with the finding that
  it makes crustal faults wrong (`hope` and `kaikoura` sit at 2.2–4 correlation lengths and
  currently warn).
- **The mesh-quality gate** (`DEGENERATE_MASS_FRACTION`) — a near-degenerate vertex is
  catastrophic for a FEM operator and merely ugly for a lattice sample. It should survive
  as a geometry check, but its measured consequence (a 5.3× suppression of the whole slip
  field) is specific to the SPDE.
- **`remesh`'s conditioning argument** — element shape cost the SPDE 26× in iterations and
  meshFIM 2.2×. Neither applies. `remesh` remains valuable for *area and depth fidelity*
  and for the viewer's decimation, but the 36× argument evaporates.

None of these should be deleted reflexively. Each needs its docstring rewritten to the
reason that survives, or removing, and a citation left pointing at a deleted rationale is
worse than no citation.

## What must survive

Everything the geometry supplies, which is now the whole of its job:

`TriangleMesh` and its file format; `remesh`; `gocad`; the Monge frame and
`check_admissible`; `centres`, `areas_km2`, `strike_dip_deg`; the parameter and
arc-length coordinates; `cell_index` (the hypocentre seam); `boundary_edges`,
`boundary_faces` and the top/bottom/lateral labelling that the taper reads; and
`triangular/pipeline.py` with its stage seams.

From `sampling.py`, the model survives whichever sampler wins:
`VonKarmanFilterParameters`, `correlation_lengths`, `von_karman_correlation`,
`standardise`, `correlate_fields`, `HURST`, `MAI_MAXIMUM_RATIO`.

## Phases

**Phase A — done.** The outline question is answered: a ×10 slowness wall on off-fault
cells removes the problem at ~30% extra solve cost with no measurable damage to the
factorisation. The remaining decision is whether the metric error (−0.14 to −0.30 s
median, −6 s worst, 15–33% of moment) is acceptable, which is a modelling judgement rather
than a measurement. Outstanding: the hypocentre-behind-a-spur case named above.

**Phase B — the eikonal, behind the existing seam.** `triangular/pipeline.py` already
injects the solver. Add a lattice path: sample slowness on the parameter lattice at
**true** depths interpolated from the mesh, **wall the off-fault cells at ×10**, sweep,
project onsets onto faces. Keep `fim.solve` alive and compare, using the curvature study as
the harness — it already has the flat/curved apparatus, shared white noise and three
hypocentres.

The fill rule needs stating at the site rather than being left to whatever is convenient:
the wall is what makes it irrelevant, and without the wall the choice of fill is worth a
factor of five in the error.

Two things need care rather than translation. `causal_jump` chooses the jump cell by
argmin over the **raw wavefront**, so a systematic bias moves the selection; the jump cell
and arrival time must be asserted stable on `beavan` and `kaikoura`, as `MESH.md`
Component 3 required of the original swap. And `apply_perturbation` pins the hypocentre's
onset exactly, so the projection must not move the one cell that carries no noise.

**Phase C — the sampler.** Same shape, through `FieldSampler`. The decision above
determines whether the delivered correlation length is measured in planar or surface
distance; record which was chosen and why, at the site. Fix the
`MAXIMUM_EMBEDDING_CELLS` bypass first — `_candidate_extents` breaks on the first over-cap
candidate, returns an empty list, and `candidates or [smallest]` then hands back an
embedding that is itself over the cap; at 100 m that is 111.8 M cells against a 67.1 M
limit, silently.

**Phase D — delete.** meshFIM, the SPDE, and the machinery listed above, with every
docstring citing them rewritten to stand alone.

## Verification

The curvature study is the regression harness and needs no new apparatus: same mesh
topology, same white noise, three hypocentres, constant and standard velocity, and a
`results.json` of 4,678 measured leaves to compare against.

1. **Onset against the surface solver**, on all three interfaces. The gate is not the
   0.05 s bound — that is a verification tolerance between implementations, and the model's
   own perturbation is ~0.35 s. It is that the error stays well inside that perturbation,
   that the jump cell is stable, and that the pinned hypocentre onset is exact.
2. **Moment exact.** True areas mean there is no area term to absorb; a discrepancy here
   is a bug rather than an approximation.
3. **Delivered covariance** against `von_karman_correlation`, stated in whichever metric
   the decision above selects.
4. **The shipped crustal examples**, which are the case the projection is *most* accurate
   on — `sin(dip) ≈ 1`, so the frame ambiguity vanishes — and therefore the case where a
   regression would be unambiguous.

## What is given up

Honestly, and not much of it was used:

- **Adaptive refinement.** A triangulation admits it; a lattice does not. Nothing in the
  pipeline refines adaptively today.
- **Irregular boundaries respected by the solvers.** The mesh keeps its outline for areas,
  taper and output; the solvers stop seeing it. This is the same fact as the risk above.
- **The claim that the field is intrinsically Matérn on the surface.** It becomes Matérn on
  the plane, projected. Whether that is a loss is the unresolved question, not a given.
