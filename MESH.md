# Triangular-mesh rupture generation

## Context

The current generator models a fault as a structured chart `X: (i, j) → R³` — a quad
lattice. Three things in the package exist only to prop that up: `validate_chart`
refuses non-planar geometry because the spectral sampler needs it, `spacing_km` collapses
a whole segment to one `(strike, dip)` pair because the sampler and the eikonal kernel
both index in lattice space, and `build_fault` spends ~120 lines on bend stretch and seam
sharing to keep neighbouring planes on one conforming grid. `mesh.py`'s own docstring
calls `validate_chart` "the temporary stage" and says deleting it plus swapping the
sampler "is the whole curvature migration".

This plan does that migration. A segment becomes a **triangulated parametric chart** —
still `X(u, v) → R³`, but with the lattice replaced by a triangulation of the parameter
domain. That admits genuinely curved surfaces (subduction interfaces), adaptive
refinement, and irregular boundaries.

Three components have to change together, because each is the reason the others cannot:

1. the correlation sampler, which is a DFT method and cannot leave a regular lattice;
2. the eikonal solver, which is a 5-point stencil on a Cartesian lattice;
3. the geometry container, which is `(i, j)` from top to bottom.

**Decisions taken** (from discussion):

- **Genuinely curved surfaces**, modelled as a Monge patch — a best-fit reference plane
  plus a normal displacement `h(u, v)`, with `u, v` playing strike and dip.
- **Matérn smoothness kept at Mai's ν = 0.75** via rational approximation, not rounded to
  an integer.
- **Published methods only in the first implementation.** The second-order route needs an
  extension of FSMCT that is not in the literature; it is documented with the test that
  would validate it, and deliberately not built yet.
- Built as a **full parallel track** beside the working structured pipeline, switched over
  once it reproduces the quad results.

---

## What the literature settles

**The eikonal gap is real.** No published method is both surface-native and factored:

|                                          | surface (2-manifold) | obtuse triangles       | factored           | order |
| ---------------------------------------- | -------------------- | ---------------------- | ------------------ | ----- |
| FSMCT — Chen et al. 2026                 | ✗ 2D (x,z)           | ✓ no subdivision       | ✓ mult. + additive | —     |
| meshFIM — Fu et al. 2011                 | ✓                    | virtual-edge unfolding | ✗                  | 1st   |
| FMM on manifolds — Kimmel & Sethian 1998 | ✓                    | unfolding              | ✗                  | 1st   |
| FSM triangular — Qian et al. 2007        | ✗ 2D                 | needs subdivision      | ✗                  | —     |

The singularity cannot simply be ignored: `crates/kernels/src/eikonal.rs` records that the
unfactored solver does not converge at the point source, and Fu et al. state plainly that
meshFIM "is not first-order accurate for nonsmooth boundaries".

**But factorisation is not the only way to remove it.** Fu et al.'s own study reports
clean first-order convergence for _circular_ boundary conditions — the failure is
specifically the nonsmooth point boundary. Seeding a geodesic ball with the analytic
homogeneous solution makes the boundary smooth, which puts the published method back in
the regime where its own paper demonstrates it converges. That keeps the first
implementation entirely within published work, at the cost of dropping from second-order
to first-order. Component 3 takes that trade and documents the second-order route as an
upgrade path.

**The Matérn smoothness forces a rational approximation.** The von Kármán ACF with Hurst
`H` _is_ the Matérn correlation of smoothness ν = H — `sampling.py:44-47` already says so.
Mai's H = 0.75 on a 2-surface gives SPDE exponent α = ν + d/2 = 1.75, non-integer, which
the plain sparse FEM of Lindgren et al. cannot represent. Bolin & Kirchner's rational SPDE
handles any α and, critically for the "report the model error" requirement, derives an
**explicit mean-square convergence rate**; they report the rational error is small
compared to the FEM error.

**On a curved surface the SPDE is not a convenience — it is what makes the model
well-posed.** Gneiting (2013) shows the Matérn coupled with geodesic distance is positive
definite on a sphere only for ν ≤ 1/2. Mai's ν = 0.75 is outside that range, so the
obvious alternative — compute geodesic distances between face centres and evaluate
`von_karman_correlation` — is not guaranteed to produce a valid random field at all. The
SPDE solution always is, by construction, because it is defined by an operator rather than
by a covariance function.

That also settles what would otherwise be a modelling choice. Sampling in the parameter
domain with the Euclidean metric is exactly "drape flat noise over curved geometry": two
points 1 km apart in projection but 1.05 km apart on the fault would receive the 1 km
correlation. The SPDE avoids it _and_ avoids computing any distances — assemble the FEM
matrices from the **lifted** triangles (true 3D areas, true edge cotangents) and the field
is intrinsically Matérn on the surface. No arc-length integration appears anywhere.

Parameter length versus true surface arc length therefore affects only two **scalar
labels**: the hypocentre spec (`strike_km`, `dip_km`) and the SRF header extents. Both are
cheap (`sqrt(1 + |∇X|²)` per element), so store both — `(u, v)` as the parameter
coordinates the covariance and the mesh are built on, and true arc lengths as derived
per-vertex fields for the hypocentre seam and the SRF.

**SW4 does not care about the SRF plane header — verified in source.** In both readers
`NSTK`/`NDIP` are `sscanf`'d, printed, and discarded; the text reader's plane variables
are block-scoped and die at the closing brace (`sw4/src/parseInputFile.C:6244-6259`), and
the HDF5 reader `free`s the metadata before opening the POINTS dataset
(`sw4/src/readhdf5.C:1050-1066`). Point count comes from the `POINTS n` line alone. There
is no `nstk*ndip == npts` check anywhere. **One PLANE with `NSTK = n_triangles, NDIP = 1`
is geometrically inert to SW4**, exactly as you supposed.

---

## Component 1 — the geometry container

A segment is a **Monge patch**: a reference plane, and a normal displacement over it.

```
X(u, v) = O + u·ê_u + v·ê_v + h(u, v)·n̂
```

**The frame splits its two jobs**, because neither choice alone is right:

- `n̂` — the **best-fit plane normal**, from the smallest singular vector of the centred
  corner cloud. This minimises `|∇h|` by construction, which maximises the margin before
  the projection folds.
- `ê_u`, `ê_v` — from the **config's stated strike and dip**, not from SVD. SVD's in-plane
  axes are principal axes of the point cloud: they coincide with strike and dip only when
  length > width, they are degenerate (and can land 45° out) on a square patch, and their
  sign is arbitrary — which is precisely the reversed-strike failure that looks plausible
  and is physically backwards. Taking them from the config keeps the anisotropy tensor `H`
  diagonal so Mai's two correlation lengths do not mix.

  ```
  ê_u = normalise(strike direction projected into the plane)
  ê_v = n̂ × ê_u
  ```

In the planar case this collapses exactly: the best-fit plane _is_ the fault plane, and
`(ê_u, ê_v)` _are_ strike and dip. That identity is a test.

**Admissibility replaces `validate_chart`.** The patch is valid iff the projection is
injective, which is checkable as "every triangle is positively oriented in the parameter
plane" — no folds. Measured on the shipped examples: `kaikoura` turns 4.6° in total,
`hope` 19.7°, and the worst surface in `alpine_hope` 36.5° across 7 planes, giving
`|∇h| ≲ 0.33`. Comfortable. But `SHARPEST_BEND_DEG = 120` _permits_ geometry that would
fold, so either that constant tightens or the refusal moves to this check — and the check
is the better home, because it names the modelling assumption directly.

Storage follows the pattern from
`nzcvm/models/mesh.py:43-82` (`TetrahedralMeshSchema`): flat per-vertex arrays plus a
connectivity table, with cell arity read from `connectivity.shape` rather than hard-coded.

**Do not reuse nzcvm's `i`/`j`/`k` dim names.** In nzcvm those mean structured axes in
`GridSchema` and (vertex, cell, corner) in `TetrahedralMeshSchema` simultaneously, and
anything that aligns across the two silently broadcasts garbage. Use `node` / `face` /
`corner`.

```
node   dim:  east_km, north_km, depth_km        (V,)   vertex positions, offsets from origin
             strike_km, dip_km                  (V,)   parameter coordinates (u, v)
             strike_arc_km, dip_arc_km          (V,)   TRUE surface arc lengths, derived
face   dim:  faces                              (F, 3) vertex indices
             plane_of_face                      (F,)   config-plane provenance
             <attached fields>                  (F,)   slip_m, rake_deg, onset_s, ...
attrs:       reference frame: origin, ê_u, ê_v, n̂
```

**Storing `(u, v)` per vertex is the single highest-leverage decision in this plan.** It
is what makes the following fall out rather than needing geodesic machinery:

- `strike_arc_km` / `dip_arc_km` — read off, not solved for.
- `cell_index(strike_km, dip_km)` — a point-in-triangle query in the parameter plane,
  returning one flat face index. The narrow seam stays narrow. Note this consumer wants
  **true arc length**, since "the hypocentre is 12 km along strike" means along the fault.
- `taper_edges` — distance to boundary in parameter space, with widths in kilometres.
- the SPDE anisotropy tensor — Mai's correlation lengths are defined along strike and down
  dip, which _are_ the parameter axes, so `H` is diagonal.
- meshing — triangulate the 2D parameter domain (Delaunay, always valid) and lift to 3D.
  No 3D surface mesher, no self-intersection, no orientation ambiguity.

**Boundary detection** is shared infrastructure, needed by the taper, the jump search and
the parameterisation: an edge incident to exactly one face is a boundary edge. Build all
`3F` edges, canonicalise as sorted pairs, `np.unique(..., return_counts=True)`, take
count 1. One `boundary_edges()` / `boundary_faces()` method on the mesh, not three
implementations. The taper additionally needs the boundary **labelled** top / bottom /
lateral, which the parameter coordinates give directly.

Carries over unchanged from the current `RuptureMesh`: `centres()` (mean of 3 corners
instead of 4), `areas_km2()` (drop the second triangle — it is already a two-triangle
cross-product formula), dip from `strike_dip_deg()` (already computed from the normal).

**The strike _sign_ stops being subtle** once the frame is fixed as above. Today it comes
from the cell's own along-strike edges, which ties it to the trace direction; a triangle
has no trace direction. Take it from the frame instead: `along_strike = ∂X/∂u`, which is
oriented by `ê_u` and therefore by the config. Still worth its own test — a reversed-strike
SRF is plausible-looking and physically backwards.

Not `uxarray` or `xugrid`: both are built for UGRID climate data on 2D horizontal meshes,
and neither models a parameterised surface embedded in 3D with per-vertex parameter
coordinates. Reuse nzcvm's generic `encode`/`decode` traversal (`nzcvm/xarray.py:12-40`)
instead — it is domain-agnostic and useful verbatim.

## Component 2 — the sampler

Replace circulant embedding entirely. It is definitionally a lattice method: the
covariance is block-circulant only because lag is index-difference × spacing
(`sampling.py:239-246`), and the whole module — including `crates/kernels/src/field.rs` —
is built on that.

Solve the Whittle–Matérn SPDE on the triangulation instead:

```
(κ² − ∇·H∇)^{α/2} u = W,     α = ν + d/2 = 1.75,  ν = 0.75 (Mai's H)
```

- `H` is the anisotropy tensor in the `(ê_u, ê_v)` frame, giving Mai's two correlation
  lengths. Because that frame is strike and dip by construction, `H` is **diagonal** —
  the general varying-local-anisotropy machinery of Fuglstad et al. is available but not
  needed at the outset.
- On a surface the operator is Laplace–Beltrami. The P1 FEM stiffness matrix for it on a
  triangulation **is** the cotangent Laplacian — assemble per triangle in its own tangent
  frame with the metric `H`, and scatter-add. ~40 lines of numpy; `scikit-fem` is an
  option but its embedded-surface support is not something to depend on unverified.
  Assembling from the **lifted** triangles is what makes the field Matérn on the surface
  rather than on its projection; it is the whole of the difference, and it costs nothing.
- Non-integer α: rational approximation of `L^{-β}`, β = α/2 = 0.875 (Bolin & Kirchner).
  `m = 1..3` rational terms is typically enough.
- Sample by sparse Cholesky of the precision `Q`: `x = L^{-T} z`, `z ~ N(0, I)`. CHOLMOD
  via `scikit-sparse`, or `scipy.sparse.linalg` if the dependency is unwelcome.

**Error reporting** — this replaces `_warn_if_degraded`, and is strictly better than what
it replaces. Bolin & Kirchner give an explicit rate in both the mesh size `h` and the
rational order `m`, so the sampler can report a bound rather than the current
after-the-fact measurement of delivered correlation lengths. Keep the existing
`MAI_MAXIMUM_RATIO = 0.6` check (Mai & Beroza fig. 13) — that is a statement about the
model's validity, not about the numerics, and it survives the change of method.

Carries over unchanged: `VonKarmanFilterParameters`, `correlation_lengths`,
`von_karman_correlation`, `standardise`, `correlate_fields`. The ACF is a function of a
scalar dimensionless distance and is deliberately 1-D.

## Component 3 — the eikonal solver

**meshFIM (Fu et al. 2011) with analytic geodesic-ball seeding.** Entirely published, and
surface-native by construction. Factorisation is deliberately _not_ used in the first
implementation; the singularity is removed by fixing the boundary condition instead.

The justification is in Fu et al.'s own convergence study. They report slope 1.0 — clean
first-order — for **circular** boundary conditions, and explicitly note the method "is not
first-order accurate for nonsmooth boundaries". A point source is the nonsmooth boundary.
So rather than repairing the solution afterwards with a factorisation, make the boundary
condition smooth in the first place:

1. Take the geodesic ball of radius `r₀` around the hypocentre.
2. Fix every node in it to the analytic homogeneous solution `T = S₀·d(x, x₀)`.
3. Run meshFIM from that ring outward, with those nodes held.

`r₀` is chosen as a few times the mean edge length — large enough that the seeded boundary
is resolved as a smooth curve rather than a jagged one, small enough that the slowness is
constant across it to a stated tolerance. Both conditions are checkable at run time and
should be _reported_, not assumed.

This also disposes of the geodesic-distance problem rather than solving it. A factored
`T₀ = S₀·r` has to be valid over the **whole domain**, so `r` must be a true geodesic
distance and curvature error accumulates everywhere. Seeding needs `d(x, x₀)` only
**inside the ball**, where chordal and geodesic distance agree to `O(κ²r₀²)` — with `r₀` a
few cells, that is negligible and boundable from the mesh's own discrete curvature. No
heat method, no global distance field.

Obtuse triangles go through Fu et al.'s virtual-edge unfolding, which is part of the
published method. The Monge patch helps here: we _choose_ the triangulation (Delaunay in
the parameter domain, which maximises the minimum angle), so badly obtuse elements are
rare and mild for the `|∇h| ≲ 0.33` the shipped geometry exhibits.

Multi-seed pointwise `min(t + t0)` carries over unchanged, as does the existing
`eikonal_contract.rs` convergence test — port it first, it is the safety net.

**What this costs, and why it is affordable.** The current
`crates/kernels/src/eikonal.rs` is a _second-order_ factored fast sweep; meshFIM is
_first-order_. That is a real regression on a planar fault — but it should be measured
against the right yardstick. `draw_fields` deliberately perturbs the wavefront by
`c·σ·Z_p` with `rupture_time_scale = -0.35` s in `crustal.toml`, so the _intended_
stochastic displacement of every onset is ~0.35 s: about seven times the 0.05 s bound in
`ENGINEERING_RULES.md`. A discretisation error an order of magnitude below the model's own
deliberate noise is not what limits the answer.

The 0.05 s bound is a _verification_ tolerance — "does this implementation agree with that
one" — not a claim about physical distinguishability, so it is the wrong test to apply to
a method change that is expected to move numbers.

**Two places the perturbation does not provide cover**, and these are the real gate:

- **Jump selection.** `solve_onsets` chooses the jump cell by argmin over the **raw
  wavefront**, not the perturbed onset — deliberately, because an argmin over a hundred
  thousand perturbed values is an order statistic that finds the perturbation's negative
  tail. So the wavefront's _shape_ selects where a multi-segment rupture crosses, and a
  systematic first-order bias moves that selection. Assert the jump cell and arrival time
  are stable between solvers on the shipped `beavan` and `kaikoura` geometries.
- **The pinned hypocentre.** `apply_perturbation` sets the hypocentre's perturbation to
  zero so its onset is exactly travel time plus delay. That one cell carries no noise to
  hide behind, and it is the registration point every diagnostic is measured from.

A first-order bias is also not random noise: it does not average out across realisations
and it grows with distance from the source. Report it as a systematic, not an RMS.

### Upgrade path, documented but not built: FSMCT in tangent frames

Worth recording because it is a small step from published work and would recover
second-order accuracy with factorisation intact — but it is **our** step, not the
literature's, so it is not what ships first.

FSMCT's local solver maps a physical triangle to a canonical reference triangle through a
2×2 Jacobian `J` (Chen et al. eqs. 2–4) and discretises on the reference triangle. Every
triangle of a surface mesh is planar, so expressing its three vertices in **its own
orthonormal tangent frame** should make the surface metric vanish into `J`, leaving the
local solve identical to the published planar one — including the multiplicative and
additive factorisations (eqs. 17–21), both acceptance conditions (angle: θ ≥ 0°,
θ + α ≤ 90°; causality: τ_C > τ_A, τ_C > τ_B), and the characteristic-method fallback.
It would also inherit FSMCT's avoidance of obtuse-triangle subdivision.

Two things would need `T₀` to be a genuine geodesic distance, which is where the heat
method (Crane et al. 2013) would come in — two sparse solves against the same cotangent
Laplacian the sampler already assembles, so the two components would share one matrix.
Note the asymmetry with the sampler if this is ever pursued: geodesic distance is _wrong_
for the covariance (Gneiting) but exactly right for `T₀`, which is a traveltime and
carries no positive-definiteness requirement.

**The validation that would settle it** is a constant-speed solve on a sphere, where the
geodesic distance is analytic (`R·arccos(...)`). A sphere is the one curved surface with
an exact answer, so it separates "the tangent-frame reduction is valid" from "it happens
to look plausible". Until that test exists and passes, this stays documented.

## The SRF seam

Emit **one PLANE with `NSTK = n_triangles`, `NDIP = 1`**, dummy header geometry, and one
POINT per triangle carrying its own lon/lat/depth/strike/dip/rake/area/slip. Verified
inert against SW4. Three caveats, all real:

- **This is SW4-specific.** Consumers that reconstruct geometry from the header get
  nonsense — including `scripts/view.py:298-392`, which builds quads from
  `strike_count`/`dip_count`/`length_km`/`width_km`. The viewer must read the native
  rupture file for triangular models, not round-trip through SRF.
- **The low-slip filter will silently eat tapered edges.** SW4 drops any point whose
  `dt·Σṡ` is under `1e-4` m in a single-precision build, warning only ten times
  (`parseInputFile.C:6353-6365`, `readhdf5.C:1234-1245`). Edge tapers drive slip to zero
  by construction, so a fine mesh loses moment quietly. Check the generated file's own
  moment against SW4's printed `made %i point moment tensor sources` tally.
- **Text and HDF5 SRF disagree on μ.** Text ignores `VS`/`DEN` and takes the shear modulus
  from the SW4 grid; HDF5 uses the file's own. Same rupture, two moments. Pick one path
  and say which.

---

## Phases

Built as a parallel track: `mesh.py` and friends stay untouched and green throughout.

**Phase 0 — validated solver spikes, no pipeline.** Standalone implementations checked
against analytic solutions before any integration.

- meshFIM on a _planar_ triangulation, point-seeded → must reproduce Fu et al.'s reported
  behaviour, including the degraded slope for a point boundary. Reproducing the _failure_
  first is what shows the implementation is faithful.
- The same solver, geodesic-ball seeded → must recover slope 1.0, and must agree with the
  existing Rust solver on a regular mesh. **The gate is not the 0.05 s bound** (see
  Component 3): it is that the wavefront error stays well under the ~0.35 s deliberate
  perturbation, _and_ that the jump cell and arrival time are unchanged on `beavan` and
  `kaikoura`, _and_ that the pinned hypocentre onset is right. Report the error as a
  systematic against distance from the source, not as an RMS.
- A sensitivity sweep on `r₀` — the seeded radius has to be reported as a bound, not
  chosen by taste. Show onset error against `r₀` and against the slowness variation across
  the ball.
- SPDE sampler on a planar triangulation → empirical covariance must match
  `von_karman_correlation` at Mai's correlation lengths, and must match the existing
  circulant sampler's delivered covariance on a regular mesh.
- SPDE sampler on a **warped** patch → the empirical correlation must track _surface_
  separation, not projected separation. Concretely: at `|∇h| ≈ 0.33` the two differ by
  ~5%, which is above the 1% slip bound, so the test can tell "Matérn on the surface"
  apart from "flat noise draped over it". This is the assertion that the lifted assembly
  is doing what it claims.

**Phase 1 — the container.** `TriangleMesh` beside `RuptureMesh`: the Monge frame
(SVD normal, config in-plane axes), parameter _and_ arc-length coordinates, boundary
detection, the fold-admissibility check, and the 2D-Delaunay-and-lift builder. File format
`SCHEMA_VERSION` 3, with the v2 reader retained so existing meshes still load
(triangulating on the fly). Gate: on a planar fault the frame must reproduce the config's
strike and dip exactly, and the lifted mesh must reproduce `RuptureMesh.areas_km2()` and
`centres()` to round-off.

**Phase 2 — the stages.** Most of this is free. Per the inventory: `pulses.py` needs zero
work (already flat per-subfault); `propagation.py` needs only `_edge_cells` →
`boundary_faces()` and `(i,j)` → `int`; `stages.py` is elementwise apart from
`taper_edges`, which becomes distance-to-labelled-boundary in parameter space; `moment.py`
is pure `.ravel()`. `CELL_DIMS = ("i","j")` → `("face",)` and `cell_counts` → `int` are
wide but mechanical — one `ast-grep` pass.

**Phase 3 — output and viewer.** The SRF seam above; `view.py` drops `strided_corners`
and draws faces directly (Rerun handles 10⁵–10⁶ triangles).

**Phase 4 — switch over and delete.** Once the triangular path reproduces the quad
results, delete `validate_chart`, `line_steps`, `_block_cut_sizes`, `spacing_km`,
`_subdivide`, the bend-stretch machinery in `build_fault`, and
`crates/kernels/src/field.rs`.

---

## Verification

Each phase gates on the one before; nothing merges without the analytic check.

1. **Solver accuracy against analytic solutions** — constant-speed traveltime on a planar
   mesh for the eikonal, and the empirical covariance for the sampler. These are
   references, not second transcriptions of the implementation. (The sphere test belongs
   to the documented upgrade path, not to this implementation.)
2. **Reduction to the existing answers.** On a regular triangulation of a planar fault,
   the new eikonal must agree with `crates/kernels/src/eikonal.rs` and the new sampler
   with the circulant one, to the tolerances in `ENGINEERING_RULES.md`'s bounds table
   (slip 1%, onset 0.05 s, rake 1°, moment exact).
3. **The existing invariant tests, re-pointed.** `tests/test_pipeline.py` asserts
   invariants of the output rather than stored arrays — moment closure, hypocentre onset,
   outward spreading, pulse-carries-its-slip, round trips. Almost all should pass against
   a triangular realisation unchanged, and the ones that do not will be the ones worth
   arguing about.
4. **Model-error reporting.** Assert the sampler's reported bound is consistent with the
   measured empirical covariance error under mesh refinement — i.e. the bound is a bound.
5. **End to end through SW4.** Generate a curved-interface rupture, write the single-plane
   SRF, run SW4, and confirm `made %i point moment tensor sources` equals the triangle
   count and the summed moment matches. This is the only check that catches the low-slip
   filter.

---

## Risks

- **First-order accuracy is a deliberate trade, not an oversight.** It is affordable
  because the model's own onset perturbation is an order of magnitude larger, but the two
  places that cover does not reach — jump selection off the raw wavefront, and the pinned
  hypocentre — are exactly the two that are hardest to notice when wrong. `DEFECTS.md` 17
  is the precedent: a hypocentre one cell out correlated 0.99+ with the truth while moving
  onsets by up to a second.
- **`r₀` is a new free parameter**, and this codebase has a stated aversion to those. Two
  bounds constrain it from opposite sides (resolve the seeded boundary as a smooth curve;
  keep slowness constant across the ball), so it should be _derived_ from the mesh and the
  velocity model and reported, in the manner of `sampling.py`'s `_predicted_extents`,
  rather than configured.
- **Rational SPDE cost.** `m` extra sparse solves per draw, times four drawn fields, times
  segments. Probably fine at fault scale; worth profiling in phase 0 before committing.
- **A fault that turns too far cannot be one Monge patch.** The shipped examples have
  `|∇h| ≲ 0.33` with wide margin, but the config permits 120° bends, which fold. The
  admissibility check catches it honestly; what it cannot do is _generate_ such a fault.
  If one is ever needed, the reference surface has to become ruled rather than planar —
  which is what `build_fault`'s existing bend-stretch construction already builds, so the
  escape hatch is to keep that as the reference surface and displace normal to it. Worth
  knowing the escape exists; not worth building until something needs it.
- **`RandomConfig.stream` keys substreams by name, not position** — that survives the
  migration untouched and is worth noting as one thing that will _not_ move numbers.

## Papers I could not reach

Available and read: Chen et al. 2026 (in `papers/`, open access), Fu et al. 2011 (PMC
open access), Bolin & Kirchner (arXiv 1711.04333), Mai & Beroza 2002 (in `papers/`).

Behind paywalls, and I would like them before implementing the parts they govern:

- **Lindgren, Rue & Lindström (2011)**, _JRSS-B_ 73(4), 423–498 — the foundational SPDE
  paper. Section 3.1 covers manifolds explicitly, which is the part this plan leans on.
  **The one I most want**, since the sampler is now the least-published component.
- **Kimmel & Sethian (1998)**, _PNAS_ 95(15), 8431–8435 — the virtual-edge unfolding for
  obtuse triangles that Fu et al. adopt by reference rather than restate. Needed to
  implement meshFIM's local solver correctly. PNAS may be open.

Not needed for the first implementation, only for the documented upgrade path:

- **Qian, Zhang & Zhao (2007)**, _SIAM J. Numer. Anal._ 45(1), 83–107 — the triangular FSM
  FSMCT builds on.
- **Zhang, Ma & Nie (2021)**, _Geophysics_ 86(3), U49–U61 — anisotropic eikonal on
  unstructured triangular meshes; the fallback if the tangent-frame reduction fails.
- **Fomel, Luo & Zhao (2009)**, _J. Comput. Phys._ 228(17), 6440–6455 — the factored
  eikonal. `eikonal.rs` already implements it, so you may already have this.
- **Gneiting (2013)**, _Bernoulli_ 19(4), 1327–1349, "Strictly and non-strictly positive
  definite functions on spheres" — the ν ≤ 1/2 restriction that rules out geodesic-distance
  Matérn at Mai's ν = 0.75. An arXiv preprint exists (1111.7077), so this one may not need
  chasing; worth confirming the bound as published before it goes in a docstring.
