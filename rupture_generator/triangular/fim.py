"""First arrivals on a triangulated surface, by the fast iterative method.

The eikonal solver of the triangular track. It answers the same question as
`crates/kernels/src/eikonal.rs` -- when does the front reach each point of a fault --
but on a triangulation of a curved surface rather than on a Cartesian lattice, which
is what a Monge patch needs and what a 5-point stencil cannot give.

The papers

    **Fu, Z., Jeong, W.-K., Pan, Y., Kirby, R. M. & Whitaker, R. T. (2011).** A fast
    iterative method for solving the eikonal equation on triangulated surfaces.
    *SIAM Journal on Scientific Computing* **33**(5), 2468-2488.

    **Kimmel, R. & Sethian, J. A. (1998).** Computing geodesic paths on manifolds.
    *Proceedings of the National Academy of Sciences USA* **95**(15), 8431-8435.

Fu et al. supply the *iteration* -- an active list swept until it empties, with no heap
and no global ordering -- and the per-triangle constant speed ("We assign a constant
speed ``f`` to each triangle", section 2.1), which is why the slowness here is per face.

**The local solver below is Kimmel & Sethian's, and that is a deliberate choice rather
than an assumption.** Fu et al. have a local solver of their own: their equation (2.2)
parametrises the characteristic by where it crosses the base edge,
``Phi_3 = lambda Phi_12 + Phi_1 + f||e_13 - lambda e_12||``, and accepts it when
``lambda`` lands in ``[0, 1]``. But they never print the quadratic -- "assigning zero to
the derivative (with respect to lambda) of (2.2) gives a quadratic equation from which
we solve for lambda" is the whole of it -- and squaring that derivative introduces a
spurious root the paper does not discuss. Kimmel & Sethian section 4.1 is the same
linear-element Godunov update in a different parametrisation, written out in closed form
with its acceptance condition, so it is the one that can be implemented from the
literature rather than from a re-derivation.

The two agree, and `tests/triangular/test_fim.py` measures where. **On acute wedges they
are the same rule**: over 400 random acute triangles, Kimmel & Sethian's equation (5)
accepts exactly when the minimiser of Fu et al.'s equation (2.2) is interior, and where
both accept they return the same arrival to round-off. **On obtuse wedges they differ in
half of cases**, and the difference is one condition: equation (5) additionally requires
``u < t``, which is "the later flank is still upwind of the answer". Fu et al.'s stated
``lambda in [0, 1]`` does not say that, and adding it restores exact agreement over
400 random obtuse triangles. This module keeps the stricter published rule. **The two
fallbacks are the same two numbers**: Fu et al. clamp to ``lambda = 0`` and
``lambda = 1`` and take the smaller, which is ``min(Phi_1 + f b, Phi_2 + f a)`` --
Kimmel & Sethian's else-branch exactly.

What Fu et al. *do* adopt by reference is the obtuse-triangle repair: "we adopt the
method used in [14]", [14] being Kimmel & Sethian. Section 4.2 below is that.

The local solver -- Kimmel & Sethian section 4.1

Update vertex ``C`` from a triangle ``ABC`` whose other two vertices already carry
arrivals, ordered so that ``T(A) <= T(B)``. Write ``b = |CA|``, ``a = |CB|``, ``theta``
for the angle at ``C``, ``u = T(B) - T(A)``, and look for ``t = T(C) - T(A)``. The
level set through ``B`` meets ``CA`` at a point ``D`` with ``CD = b(t - u)/t``, and
``t - u = F h`` with ``h`` the distance from ``C`` to the line ``BD``. Their equation
(3) gives ``h`` by the sine and cosine rules, and eliminating it gives their equation
(4), a quadratic in ``t``:

.. math::

    (a^2 + b^2 - 2ab\\cos\\theta)\\,t^2 + 2bu(a\\cos\\theta - b)\\,t
    + b^2(u^2 - F^2a^2\\sin^2\\theta) = 0

whose leading coefficient is just ``|AB|^2``. The larger root is the causal branch.
Their equation (5) is when it may be used:

.. math::  u < t, \\qquad a\\cos\\theta < \\frac{b(t-u)}{t} < \\frac{a}{\\cos\\theta}

and otherwise the update falls back to the one-sided edge form
``T(C) = min(T(A) + bF, T(B) + aF)`` -- the same safe cap `eikonal.rs` carries in place
of Fomel et al.'s equation (8).

Equation (5)'s two-sided inequality is written here as ``a cos(theta) < d`` and
``d cos(theta) < a`` rather than as a division by the cosine. The two say the same
thing -- that the foot of the perpendicular from ``C`` to the level line ``BD`` falls
*inside* the segment ``BD``, which is one acute angle at each end -- and the product
form does not divide by a cosine that a right-angled triangle makes exactly zero. A
regular triangulation of a fault is made of right triangles, so that is the common case
rather than a corner. Where the two forms would part company, at an obtuse ``theta``,
the division form rejects everything and ``u < t`` is doing the work anyway.

Obtuse triangles -- Kimmel & Sethian section 4.2

``u < t`` is where an obtuse angle bites, and the geometry says exactly how hard. A
plane front arriving at ``C`` from inside the wedge can be used only if *both* ``A``
and ``B`` are upwind of ``C``, and the set of arrival directions for which that holds
subtends ``pi - theta``, not ``theta``. The whole wedge is usable iff
``theta <= pi/2``: that is what "acute triangulation" means and why it is required.
Kimmel & Sethian's figure 7 labels the same angle ``alpha = pi - theta_max``, and they
give the cost of getting it wrong -- "The accuracy of the first order scheme for acute
triangles is of ``O(h_max) ~ O(e_max)``. The accuracy for the obtuse case with the
above construction becomes ``O(l_max) = O(e_max/(pi - theta_max))``".

Their repair is the **virtual edge**: unfold the neighbouring triangles across the far
edge, in the plane, until an unfolded vertex ``B`` lands inside the wedge; connect
``C`` to it by a virtual edge whose length is the unfolded distance; the obtuse wedge
becomes two acute ones. Their equation (6) bounds the number of triangles that have to
be unfolded by a constant, so the construction costs ``O(M)`` and the method stays
optimal. :data:`UNFOLD_LIMIT` is that constant made explicit.

Unfolding is done once, as a preprocessing pass, and its virtual wedges are added to
the same corner table the real triangles fill. The slowness charged to a virtual wedge
is the slowness of the triangle whose obtuse angle it splits, not of the strip it was
unfolded through: the wedge is a device for reading the front's direction inside *that*
triangle, and the strip is only how far away the reading was taken.

Why a geodesic ball, and not factorisation

Fu et al.'s section 3.3.2, "Error analysis", is the justification for everything below.
Seven regularly triangulated 16x16 squares from 256 to 1,048,576 vertices, solved twice:
once from a pair of isolated points, once from a pair of circles of radius 3. Verbatim:

    "For the circular boundary conditions, the slope of this graph is 1.0, which is
    consistent to our claim that meshFIM is first-order accurate. For the point boundary
    conditions, the slope is less--showing the method is not first-order accurate for
    nonsmooth boundaries, which are inconsistent with the governing equations."

A point source is that nonsmooth boundary, and it is the same failure `eikonal.rs`
records for an unfactored Cartesian sweep: refining the mesh does not help, because the
near-source error is a fixed fraction rather than a truncation term. Note that the paper
commits to no number for the degraded slope -- only "less" -- and that its figure 3.3(b)
is the whole of the evidence, so `tests/triangular/test_fim.py` measures the slope here
rather than asserting the paper's.

`MESH.md`'s Component 3 takes the published way out rather than a new one. Instead of
repairing the solution afterwards with a factorisation -- which has no surface-native
publication -- make the boundary condition smooth in the first place:

1. take the ball of radius ``r0`` around the source;
2. fix every vertex in it to the analytic homogeneous solution ``T = S0 d(x, x0)``;
3. iterate outward with those held.

That puts the method back in the regime its own paper demonstrates it converges in, at
the cost of first order rather than the second order the Cartesian solver reaches. It
also disposes of the geodesic-distance problem rather than solving it: a factored
``T0 = S0 r`` has to be valid over the whole domain, so ``r`` would have to be a true
geodesic distance and its curvature error would accumulate everywhere, while seeding
needs ``d(x, x0)`` only *inside* the ball, where the chordal distance used here agrees
with the geodesic one to ``O(kappa^2 r0^2)``.

:data:`SEED_RING_DEPTH` and :data:`SEED_SLOWNESS_BUDGET_S` are the two bounds that pin
``r0`` from opposite sides. ``r0`` is derived from the mesh and the velocity model and
is never configured; both bounds are measured on the mesh actually handed in and
reported on :class:`SeedReport`, because `MESH.md`'s Risks section is explicit that a
bound assumed is a free parameter wearing a hat.

Multiple seeds

Carried over from `eikonal.rs` unchanged, and for the same reason. The seed contract is
**points with initial times**, not "the hypocentre" -- what a rupture jumping between
faults needs. The boundary condition is inherently per-source: one ball removes *one*
singularity, so a multi-seed field is one ball-seeded solve per seed and the pointwise
minimum of the shifted results. That is not a shortcut standing in for a "real"
multi-source solve; first arrival from several sources *is* the minimum over sources,
and solving them separately is what keeps every source's near field exact.

What this costs

The Cartesian solver is a second-order factored fast sweep; this is first order. The
regression is real and is affordable for one reason only: `draw_fields` deliberately
perturbs every onset by ``c sigma Z_p`` with ``rupture_time_scale = -0.35`` s in
`examples/crustal.toml`, so the *intended* stochastic displacement is about seven times
the 0.05 s verification bound. A discretisation error an order of magnitude below the
model's own deliberate noise is not what limits the answer.

Two places that cover does not reach, and they are the gate rather than the RMS:

- **Jump selection.** `propagation.causal_jump` chooses the jump cell by argmin over
  the *raw* wavefront, deliberately, because an argmin over a hundred thousand
  perturbed values is an order statistic that finds the perturbation's negative tail.
  So the wavefront's shape selects where a multi-segment rupture crosses, and a
  systematic first-order bias moves that selection.
- **The pinned hypocentre.** `stages.apply_perturbation` sets the hypocentre's
  perturbation to zero, so that one cell carries no noise to hide behind, and it is the
  registration point every diagnostic is measured from.

A first-order bias is not random noise: it does not average out across realisations and
it grows with distance from the source. `tests/triangular/test_fim.py` reports it as a
systematic against source distance for that reason, and measures the jump cell directly.

Upgrade path, documented and deliberately not built: FSMCT in tangent frames

    **Chen, Y., et al. (2026).** A fast sweeping method for the eikonal equation on
    triangular meshes with factorisation (FSMCT). `papers/1-s2.0-S199582262600049X-main.pdf`.

FSMCT is second order *and* factored, and would recover everything given up above, but
it is written for a 2-D ``(x, z)`` domain. Its local solver maps a physical triangle to
a canonical reference triangle through a 2x2 Jacobian ``J`` (equations 2-4) and
discretises there. Every triangle of a surface mesh is planar, so expressing its three
vertices in **its own orthonormal tangent frame** should make the surface metric vanish
into ``J``, leaving the local solve identical to the published planar one -- including
the multiplicative and additive factorisations (equations 17-21), both acceptance
conditions (angle, and causality ``tau_C > tau_A``, ``tau_C > tau_B``), and the
characteristic-method fallback. It would also inherit FSMCT's avoidance of
obtuse-triangle subdivision, which is the whole of section 4.2 above.

That reduction is **our** step, not the literature's, which is why it is not what ships
first. Two of its terms would need ``T0`` to be a genuine geodesic distance, which is
where the heat method (Crane et al. 2013) would come in -- two sparse solves against
the same cotangent Laplacian the SPDE sampler already assembles.

**The test that would settle it does not exist yet.** It is a constant-speed solve on a
sphere, where the geodesic distance is analytic, ``R arccos(...)``. A sphere is the one
curved surface with an exact answer, so it separates "the tangent-frame reduction is
valid" from "it happens to look plausible". Until that test is written and passes, this
section is documentation and nothing here implements it.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

SEED_RING_DEPTH = 3
"""How many rings of the mesh graph the analytic ball must be deep, at least.

The lower bound on ``r0``, and it is a count of rings rather than a multiple of an
edge length so that a graded mesh cannot make it mean two different things in two
places.

**One ring is the floor**, because a vertex updated across a triangle containing the
source vertex reads the singularity itself, which is the whole thing the ball exists to
remove; seeding the source's own one-ring puts every remaining stencil clear of it.
**Three is what makes the seeded boundary a curve.** On a triangulation whose interior
vertices have six neighbours, ring ``k`` carries about ``6k`` vertices, so the seeded
boundary is a polygon of about 18 sides at three rings against 6 at one. Its departure
from the circle is the sagitta ``1 - cos(pi/n)``: 13.4% of the radius at 6 sides, 1.5%
at 18, 0.4% at 36.

**Measured, in `tests/triangular/test_fim.py`'s sweep**, and this is what settles the
number rather than the sagitta argument. Worst-vertex error against the analytic
constant-gradient solution, in milliseconds, by ring depth:

    ==================  ====  ====  ====  ====  ====  ====
    medium              1     2     3     4     6     8
    ==================  ====  ====  ====  ====  ====  ====
    1.8+0.15z, h=1 km   207   180   191   216   506   1050
    1.8+0.15z, h=0.25    83    73    66    63    65     73
    3.0+0.06z, h=1 km   196   167   152   147   166    303
    ==================  ====  ====  ====  ====  ====  ====

Two to four rings is flat -- the error varies by 1.20x, 1.16x and 1.13x across it --
and three is the middle. Outside the window ``r0`` starts to matter in both directions:
at one ring the boundary is a hexagon rather than a circle, and by six the
constant-slowness error of :data:`SEED_SLOWNESS_BUDGET_S` has taken over. The point of
deriving ``r0`` rather than configuring it is not that the choice does not matter, but
that both edges of the window it sits in are visible.

**A ring count is a physical length only if the mesh has one**, and that is a real
condition rather than a formality. Measured over 200 seed positions on Hikurangi:

    ================  ============  =====================  ==================
    mesh              edge spread   ``r0`` across seeds    over the slowness
                                                            budget
    ================  ============  =====================  ==================
    CFM as shipped    4608x         **9.3x** (4.4-41 km)   **164 of 200**
    built at 800 m    1.9x          1.5x (2.9-4.3 km)      7 of 200
    built at 400 m    1.9x          1.3x (1.7-2.2 km)      **0 of 200**
    ================  ============  =====================  ==================

On a mesh built by :func:`~rupture_generator.triangular.mesh.remesh` the derivation holds
and ``r0`` is a fixed physical radius to within 30%. On the raw CFM triangulation it is
not: three rings is 4.4 km in one place and 41 km in another, and the constant-slowness
assumption fails at 82% of seed positions. That is a statement about which mesh to solve
on, not an argument for making ``r0`` configurable -- and it is *detected* rather than
assumed, because :class:`SeedReport` measures it and :class:`DegradedSeed` says so.
"""

SEED_SLOWNESS_BUDGET_S = 0.05
"""The traveltime a constant-slowness ball may cost, in seconds.

The upper bound on ``r0``, from the velocity model rather than from the mesh. The
analytic seed asserts one slowness ``S0`` across the whole ball, so at the ring it is
wrong by at most ``r0 max|S - S0|`` -- the quantity :class:`SeedReport` carries -- and
that error is then carried outward into every arrival the ring feeds.

`ENGINEERING_RULES.md`'s 0.05 s onset bound, and no margin under it, because this is the
one place `MESH.md` says the model's own 0.35 s perturbation gives no cover. Past 0.05 s
the seeding is spending the entire verification budget by itself, before the scheme has
discretised anything.

**Measured, in `tests/triangular/test_fim.py`'s sweep.** ``r0 max|S - S0|`` is a genuine
bound: it exceeds the seeding error actually committed at every radius of every medium
swept, by a factor between 2.1 and 3.6. It is therefore conservative by about three, and
never optimistic -- which is what a bound has to be, and why the budget is compared
against it rather than against an estimate.

**And it does not always hold.** On a 1 km subfault mesh in a ``v = 1.8 + 0.15 z`` km/s
crust the three-ring ball spans 4.2 km and the bound reaches 0.28 s, six times the
budget. That is honest rather than fixable: at that spacing the scheme's own
discretisation error is the same size (`MESH.md`'s expected first-order regression), so
shrinking ``r0`` to meet the budget would trade one error for a larger one -- which is
exactly what the sweep shows. Exceeding it is **reported and warned about**, not
refused. A hypocentre a kilometre from a layer boundary is an ordinary earthquake, and a
solver that declined to run there would be answering a different question.
"""

UNFOLD_LIMIT = 64
"""How many triangles one obtuse wedge may be unfolded through before it is abandoned.

Kimmel & Sethian equation (6) bounds the count by
``m = e_max^2 / (theta_min h_min^2 alpha^3)`` with ``alpha = pi - theta_max``, and the
point of the bound is that it is a **constant** -- independent of the mesh size -- which
is what keeps the construction ``O(M)`` and the method optimal. It is not a small
constant in the worst case, so this is a ceiling rather than an estimate: a wedge that
needs more than 64 triangles has ``alpha`` small enough that equation (6)'s
``O(e_max/alpha)`` accuracy is worse than the edge fallback it would be replacing.

Wedges that hit the ceiling, or that run into a boundary edge with nothing left to
unfold, keep the one-sided edge update and are **counted** on :class:`SeedReport` --
a silently degraded stencil is exactly the kind of thing that reads as a plausible
answer.
"""

MAX_SWEEP_FACTOR = 4
"""How many times the front's own ring count the active list may iterate.

Fu et al.'s active list advances by at least one ring of vertices per pass, so a front
that has to cross ``n`` rings needs ``n`` passes; a vertex re-enters the list only when
a later, faster path overtakes an earlier one, which is a property of the medium rather
than of the mesh. Four times the seeded set's own eccentricity, measured on the mesh
handed in, is therefore a ceiling on a bounded quantity and not a tuning knob -- the
loop exits as soon as the list is empty rather than on the count.

Measured in `tests/triangular/test_fim.py`, passes over rings: 1.05 on a regular
lattice and on a warped one, 1.29 on a Delaunay mesh, and 1.60 on a lattice with
alternating diagonals, which gives the front two ways round every quad and so the most
re-entry. Two and a half times the worst headroom used.
"""

SETTLED_TOLERANCE_S = 1.0e-12
"""How little a vertex's arrival may move and still count as settled, in seconds.

Fu et al.'s ``|p - q| < epsilon`` test, the one that takes a vertex off the active list.
**They never give it a value**, in section 2.4 or anywhere else, so this is derived
rather than transcribed. Bounded below by f64 round-off -- a 10 s fault-scale traveltime
resolves to about 2e-15 s -- and above by anything a traveltime is ever compared at, the
tightest being `ENGINEERING_RULES.md`'s 0.05 s. Ten orders above the floor and ten below
the ceiling, and it bounds the tail of the iteration rather than the accuracy of the
answer: the scheme's own error is first order in the edge length, eleven orders larger.
"""

_DEGENERATE_SINE = 1.0e-12
"""The smallest ``sin(theta)`` a triangle corner may have and still be a triangle.

A relative measure -- ``|e1 x e2| / (|e1||e2|)`` -- so it is scale free. Below this the
corner is a straight line, the quadratic's leading coefficient collapses, and the root
is noise; a mesh carrying one is refused by name rather than solved around.
"""


NUMPY = "numpy"
"""The reference implementation: the batched pass in this module.

The default everywhere, and the oracle `crates/kernels/tests/fim_contract.rs` and
:func:`test_the_kernel_reproduces_the_reference` hold the Rust kernel to. Slower by 45x
to 70x, and kept for exactly that reason -- a reference that is the same code as the
thing it checks is not a reference, which `DEFECTS.md` 17 and 18 both are.
"""

KERNEL = "kernel"
"""`crates/kernels/src/fim.rs`: the same method, Gauss-Seidel, optionally threaded.

**66x to 92x faster, and agreeing with this module to 4e-12 s.** Measured on Hikurangi
meshes built by :func:`~rupture_generator.triangular.mesh.remesh`, which is what
production uses:

    =========  ==========  ==========  ==========  ==========
    spacing    vertices    reference   kernel      agreement
    =========  ==========  ==========  ==========  ==========
    3200 m         17,204     1.317 s     0.019 s   3.8e-12 s
    1600 m         68,704     5.651 s     0.085 s   9.1e-12 s
    800 m         275,049    25.006 s     0.354 s   1.2e-11 s
    400 m       1,100,240   141.187 s     1.535 s   2.4e-11 s
    200 m       4,400,971           --     5.670 s          --
    =========  ==========  ==========  ==========  ==========

Two differences of substance, both measured and both documented in the Rust module:

- It runs Fu et al.'s Algorithm 2.1 as written -- in place, each update visible to the
  next vertex -- where this module runs a batched pass. Same fixed point; on a
  well-shaped mesh both are near-linear in the vertex count, and the 70x is interpreter
  and allocator overhead. On a *badly* shaped one the batched pass degrades to about
  ``V**1.5`` and the kernel does not.
- It sweeps again until no vertex would move, which Algorithm 2.1 does not give on its
  own: the paper's removal condition takes a vertex off the list when its own value stops
  moving, and two adjacent vertices can both stop moving in one visit while each still
  owes the other an update. **This module inherits that defect** -- see the note on
  :func:`solve_from_boundary` -- and the kernel does not.

Reach for it above about 100,000 vertices, which is where this module starts to cost
more than the answer is worth.
"""

BACKENDS = (NUMPY, KERNEL)
"""The two implementations, by name."""


class DegradedSeed(UserWarning):
    """The analytic ball's assumptions do not hold as well as they should."""


@dataclasses.dataclass(frozen=True)
class Seed:
    """A vertex the rupture front starts from, and when.

    Mirrors `eikonal.rs`'s ``Seed`` with the lattice index replaced by a vertex index.

    Attributes
    ----------
    vertex : int
        Which vertex of the mesh, indexing ``vertices_km``.
    t0_s : float
        The time the front leaves this point: zero for a configured hypocentre, a
        parent fault's arrival plus the jump delay for a triggered one.
    """

    vertex: int
    t0_s: float = 0.0


@dataclasses.dataclass(frozen=True)
class SeedReport:
    """What one seed's analytic ball actually was, and how well it was justified.

    Both of ``r0``'s bounds are checkable on the mesh handed in, and this is where they
    are checked. Returned by :func:`solve_with_report` rather than logged, because a
    bound nobody can see is a free parameter that has learned to keep quiet.

    Attributes
    ----------
    vertex : int
        The seed's vertex.
    radius_km : float
        ``r0``: the radius of the ball fixed to the analytic solution.
    source_slowness_s_per_km : float
        ``S0``, the area-weighted mean slowness of the faces meeting the seed vertex.
        A vertex sits between faces and has no slowness of its own; the area weighting
        is the same P0-to-P1 reduction :func:`face_arrivals` inverts.
    seeded_vertices : int
        How many vertices the ball fixed.
    boundary_vertices : int
        How many unseeded vertices touch it -- the polygon the front actually leaves
        from.
    boundary_radius_spread : float
        ``(max r - min r) / r0`` over those boundary vertices.

        **A weak discriminator, and that is worth knowing rather than discovering.** The
        first unseeded ring spans ``r0`` to ``r0 + h``, so on a mesh of uniform spacing
        this is about ``h / r0``, which :data:`SEED_RING_DEPTH` fixes at roughly ``1/3``
        by construction -- measured 0.28 to 0.33 on regular lattices *and* 0.33 on the
        raw CFM Hikurangi interface, whose edges span a factor of 4608. It reports that
        the ring exists and is one cell thick; it does not report whether the mesh is
        uniform enough for the ring to be a circle. What does is
        :attr:`slowness_error_s`, which on the same CFM mesh exceeds its budget for 164 of
        200 seed positions, and the spread of :attr:`radius_km` itself across seeds --
        9.3x on the raw CFM mesh against a uniform value on one built by
        :func:`~rupture_generator.triangular.mesh.remesh`.
    slowness_spread : float
        ``max|S - S0| / S0`` over the faces the ball covers.
    slowness_error_s : float
        ``r0 max|S - S0|``: what the constant-slowness assumption costs at the ring, in
        seconds. The *upper* bound on ``r0`` in measured form; compared against
        :data:`SEED_SLOWNESS_BUDGET_S`.
    unsplit_obtuse_wedges : int
        Obtuse corners the unfolding could not turn into acute ones -- a boundary edge
        with nothing beyond it, or :data:`UNFOLD_LIMIT` exhausted. Each keeps the
        one-sided edge update and its ``O(e_max / (pi - theta_max))`` accuracy.
    sweeps : int
        Passes of the active list before it emptied.
    """

    vertex: int
    radius_km: float
    source_slowness_s_per_km: float
    seeded_vertices: int
    boundary_vertices: int
    boundary_radius_spread: float
    slowness_spread: float
    slowness_error_s: float
    unsplit_obtuse_wedges: int
    sweeps: int


# ============================================================================
# The corner table: one record per (triangle, vertex), plus the virtual wedges
# ============================================================================


@dataclasses.dataclass(frozen=True)
class _Corners:
    """Every wedge a vertex can be updated across, in compressed-row order.

    One record per triangle corner, plus one per virtual wedge from the unfolding. The
    two flanking vertices are stored unordered -- which of them is Kimmel & Sethian's
    ``A`` depends on which carries the smaller arrival, and that is not known until the
    solve runs.
    """

    apex: IntArray
    left: IntArray
    right: IntArray
    left_km: FloatArray
    right_km: FloatArray
    cosine: FloatArray
    sine: FloatArray
    chord_km2: FloatArray
    slowness: FloatArray
    start: IntArray
    order: IntArray
    unsplit_obtuse: int


def _corner_geometry(
    to_left: FloatArray, to_right: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Lengths, cosine and sine of the angle between two edge vectors at a corner.

    Parameters
    ----------
    to_left, to_right : FloatArray
        ``(n, 3)`` edge vectors from the apex to each flanking vertex.

    Returns
    -------
    tuple of FloatArray
        ``(|to_left|, |to_right|, cos theta, sin theta)``.
    """
    left_km = np.linalg.norm(to_left, axis=1)
    right_km = np.linalg.norm(to_right, axis=1)
    scale = left_km * right_km
    cosine = np.einsum("ij,ij->i", to_left, to_right) / scale
    sine = np.linalg.norm(np.cross(to_left, to_right), axis=1) / scale
    return left_km, right_km, cosine, sine


def _edge_neighbours(faces: IntArray) -> dict[tuple[int, int], list[int]]:
    """Which faces meet along each edge, keyed by the sorted vertex pair."""
    shared: dict[tuple[int, int], list[int]] = {}
    for face, corners in enumerate(faces):
        for first, second in ((0, 1), (1, 2), (2, 0)):
            key = (
                int(min(corners[first], corners[second])),
                int(max(corners[first], corners[second])),
            )
            shared.setdefault(key, []).append(face)
    return shared


def _turn(first_km: FloatArray, second_km: FloatArray) -> float:
    """The scalar cross product of two planar vectors: positive turning left.

    `numpy.cross` deprecated its 2-D form in numpy 2, and the replacement it suggests
    -- padding to three components -- allocates a copy of every wedge to read one
    number off it. This is that number.
    """
    return float(first_km[0] * second_km[1] - first_km[1] * second_km[0])


def _unfold(
    anchor_km: FloatArray, other_km: FloatArray, to_anchor: float, to_other: float
) -> FloatArray:
    """Place a triangle's third vertex in the plane, folded away from the origin.

    The two-circle construction: the unfolded vertex keeps its true distances to the
    two vertices of the edge it is unfolded across, and lands on the far side of that
    edge from the apex -- which is at the origin, since that is the frame the wedge is
    laid out in.

    Parameters
    ----------
    anchor_km, other_km : FloatArray
        ``(2,)`` positions of the shared edge's two vertices, already unfolded.
    to_anchor, to_other : float
        True lengths from the vertex being placed to each of them, in kilometres.

    Returns
    -------
    FloatArray
        ``(2,)`` -- the unfolded position.
    """
    along = other_km - anchor_km
    length = float(np.linalg.norm(along))
    unit = along / length
    normal = np.array([-unit[1], unit[0]])
    forward = (to_anchor**2 - to_other**2 + length**2) / (2.0 * length)
    sideways = np.sqrt(max(to_anchor**2 - forward**2, 0.0))
    # The apex is at the origin, so "away from the apex" is the side of the edge the
    # origin is not on. Unfolding onto the apex's own side would fold the strip back
    # over the wedge it is meant to be extending.
    apex_side = _turn(along, -anchor_km)
    candidate = anchor_km + forward * unit + sideways * normal
    if _turn(along, candidate - anchor_km) * apex_side > 0.0:
        candidate = anchor_km + forward * unit - sideways * normal
    return candidate


def _split_obtuse(
    apex: int,
    left: int,
    right: int,
    face: int,
    positions_km: FloatArray,
    faces: IntArray,
    shared: dict[tuple[int, int], list[int]],
) -> tuple[list[tuple[int, int, FloatArray, FloatArray]], int]:
    """Kimmel & Sethian section 4.2: split one obtuse wedge into acute virtual ones.

    Lay the wedge out in its own plane with the apex at the origin, then unfold the
    neighbouring triangles across the far edge until a vertex lands inside the wedge.
    That vertex splits it in two; each half is checked again and split again if it is
    still obtuse. A vertex that unfolds *outside* the wedge does not split anything, so
    the strip carries on past it -- the far edge advances to the one the wedge's
    interior still crosses.

    Parameters
    ----------
    apex : int
        The vertex whose angle is obtuse.
    left, right : int
        The wedge's two other vertices, in the face.
    face : int
        The face whose corner this is; the first edge unfolded across is its far one.
    positions_km : FloatArray
        ``(V, 3)`` vertex positions.
    faces : IntArray
        ``(F, 3)`` connectivity.
    shared : dict
        Edge-to-faces map from :func:`_edge_neighbours`.

    Returns
    -------
    tuple
        The virtual wedges as ``(left vertex, right vertex, left 2-D, right 2-D)``, and
        how many wedges could not be split at all.
    """
    origin = positions_km[apex]
    to_left = positions_km[left] - origin
    to_right = positions_km[right] - origin
    left_km = float(np.linalg.norm(to_left))
    right_km = float(np.linalg.norm(to_right))
    cosine = float(np.dot(to_left, to_right)) / (left_km * right_km)
    sine = float(np.linalg.norm(np.cross(to_left, to_right))) / (left_km * right_km)
    # The wedge in its own plane: the left ray along +x, the right ray at +theta.
    left_2d = np.array([left_km, 0.0])
    right_2d = right_km * np.array([cosine, sine])

    wedges: list[tuple[int, int, FloatArray, FloatArray]] = []
    unsplit = 0
    # Each entry is a wedge still to be resolved: its two bounding rays, the far edge
    # currently being unfolded across, and the face on this side of that edge.
    pending = [(left, right, left_2d, right_2d, left, right, left_2d, right_2d, face)]
    budget = UNFOLD_LIMIT
    while pending:
        (
            ray_left,
            ray_right,
            ray_left_2d,
            ray_right_2d,
            edge_left,
            edge_right,
            edge_left_2d,
            edge_right_2d,
            here,
        ) = pending.pop()
        if float(np.dot(ray_left_2d, ray_right_2d)) >= 0.0:
            wedges.append((ray_left, ray_right, ray_left_2d, ray_right_2d))
            continue
        if budget <= 0:
            unsplit += 1
            continue
        budget -= 1

        key = (min(edge_left, edge_right), max(edge_left, edge_right))
        beyond = [other for other in shared.get(key, []) if other != here]
        if not beyond:
            # A boundary edge: there is nothing left to unfold, so this wedge keeps the
            # one-sided edge update its own triangle already offers.
            unsplit += 1
            continue
        next_face = beyond[0]
        far = next(
            int(vertex)
            for vertex in faces[next_face]
            if vertex not in (edge_left, edge_right)
        )
        far_2d = _unfold(
            edge_left_2d,
            edge_right_2d,
            float(np.linalg.norm(positions_km[far] - positions_km[edge_left])),
            float(np.linalg.norm(positions_km[far] - positions_km[edge_right])),
        )

        inside_left = _turn(ray_left_2d, far_2d) > 0.0
        inside_right = _turn(far_2d, ray_right_2d) > 0.0
        if inside_left and inside_right:
            pending.append(
                (
                    ray_left,
                    far,
                    ray_left_2d,
                    far_2d,
                    edge_left,
                    far,
                    edge_left_2d,
                    far_2d,
                    next_face,
                )
            )
            pending.append(
                (
                    far,
                    ray_right,
                    far_2d,
                    ray_right_2d,
                    far,
                    edge_right,
                    far_2d,
                    edge_right_2d,
                    next_face,
                )
            )
        elif not inside_left:
            # Unfolded clockwise of the left ray: the wedge's interior now crosses the
            # far vertex's *other* edge, so the strip advances there.
            pending.append(
                (
                    ray_left,
                    ray_right,
                    ray_left_2d,
                    ray_right_2d,
                    far,
                    edge_right,
                    far_2d,
                    edge_right_2d,
                    next_face,
                )
            )
        else:
            pending.append(
                (
                    ray_left,
                    ray_right,
                    ray_left_2d,
                    ray_right_2d,
                    edge_left,
                    far,
                    edge_left_2d,
                    far_2d,
                    next_face,
                )
            )
    return wedges, unsplit


def _corners(
    vertices_km: FloatArray, faces: IntArray, slowness_s_per_km: FloatArray
) -> _Corners:
    """Build every wedge of the mesh, real and virtual, in compressed-row order."""
    positions = vertices_km
    corners = np.arange(3)
    apex = faces[:, corners].reshape(-1)
    left = faces[:, (corners + 1) % 3].reshape(-1)
    right = faces[:, (corners + 2) % 3].reshape(-1)
    of_face = np.repeat(np.arange(faces.shape[0]), 3)

    left_km, right_km, cosine, sine = _corner_geometry(
        positions[left] - positions[apex], positions[right] - positions[apex]
    )
    if (sine < _DEGENERATE_SINE).any():
        worst = int(np.argmin(sine))
        raise ValueError(
            f"face {int(of_face[worst])} has a corner at vertex {int(apex[worst])} "
            f"whose angle has sine {float(sine[worst]):.3g}; a triangle that thin is a "
            "line segment, and no gradient can be read across it"
        )
    chord_km2 = np.sum((positions[left] - positions[right]) ** 2, axis=1)
    slowness = slowness_s_per_km[of_face]

    obtuse = np.flatnonzero(cosine < 0.0)
    unsplit = 0
    virtual: list[tuple[int, int, int, float, float, float, float, float, float]] = []
    if obtuse.size:
        shared = _edge_neighbours(faces)
        for record in obtuse:
            wedges, missed = _split_obtuse(
                int(apex[record]),
                int(left[record]),
                int(right[record]),
                int(of_face[record]),
                positions,
                faces,
                shared,
            )
            unsplit += missed
            for wedge_left, wedge_right, left_2d, right_2d in wedges:
                span_left = float(np.linalg.norm(left_2d))
                span_right = float(np.linalg.norm(right_2d))
                scale = span_left * span_right
                virtual.append(
                    (
                        int(apex[record]),
                        wedge_left,
                        wedge_right,
                        span_left,
                        span_right,
                        float(np.dot(left_2d, right_2d)) / scale,
                        abs(_turn(left_2d, right_2d)) / scale,
                        float(np.sum((left_2d - right_2d) ** 2)),
                        float(slowness[record]),
                    )
                )

    if virtual:
        extra = np.array(virtual, dtype=np.float64)
        apex = np.concatenate([apex, extra[:, 0].astype(np.int64)])
        left = np.concatenate([left, extra[:, 1].astype(np.int64)])
        right = np.concatenate([right, extra[:, 2].astype(np.int64)])
        left_km = np.concatenate([left_km, extra[:, 3]])
        right_km = np.concatenate([right_km, extra[:, 4]])
        cosine = np.concatenate([cosine, extra[:, 5]])
        sine = np.concatenate([sine, extra[:, 6]])
        chord_km2 = np.concatenate([chord_km2, extra[:, 7]])
        slowness = np.concatenate([slowness, extra[:, 8]])

    order = np.argsort(apex, kind="stable")
    start = np.searchsorted(apex[order], np.arange(positions.shape[0] + 1))
    return _Corners(
        apex=apex[order],
        left=left[order],
        right=right[order],
        left_km=left_km[order],
        right_km=right_km[order],
        cosine=cosine[order],
        sine=sine[order],
        chord_km2=chord_km2[order],
        slowness=slowness[order],
        start=start.astype(np.int64),
        order=order,
        unsplit_obtuse=unsplit,
    )


def _adjacency(faces: IntArray, vertex_count: int) -> tuple[IntArray, IntArray]:
    """Vertex-to-vertex neighbours in compressed-row order, from the face table."""
    pairs = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    both = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    both = np.unique(both, axis=0)
    counts = np.bincount(both[:, 0], minlength=vertex_count)
    start = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return start, np.ascontiguousarray(both[:, 1])


def _ragged(start: IntArray, index: IntArray, rows: IntArray) -> IntArray:
    """Gather several compressed rows at once, concatenated in the order given.

    Parameters
    ----------
    start : IntArray
        Row boundaries, length ``n + 1``.
    index : IntArray
        The flat entries.
    rows : IntArray
        Which rows to gather.

    Returns
    -------
    IntArray
        The gathered entries. Where each row begins within them is
        ``cumsum(counts) - counts``, which the one caller that needs it recomputes
        rather than every caller being handed an array most of them discard.
    """
    counts = start[rows + 1] - start[rows]
    within = np.arange(int(counts.sum())) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    return index[np.repeat(start[rows], counts) + within]


def _neighbours_of(start: IntArray, index: IntArray, rows: IntArray) -> IntArray:
    """The distinct vertices adjacent to any of ``rows``."""
    if rows.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(_ragged(start, index, rows))


def _hops(start: IntArray, index: IntArray, sources: IntArray, count: int) -> IntArray:
    """Graph distance in edges from a set of vertices; ``-1`` where unreachable."""
    hops = np.full(count, -1, dtype=np.int64)
    hops[sources] = 0
    frontier = np.asarray(sources, dtype=np.int64)
    depth = 0
    while frontier.size:
        depth += 1
        found = _neighbours_of(start, index, frontier)
        frontier = found[hops[found] < 0]
        hops[frontier] = depth
    return hops


# ============================================================================
# The local solver
# ============================================================================


def _candidates(corners: _Corners, picked: IntArray, times_s: FloatArray) -> FloatArray:
    """Kimmel & Sethian equations (4) and (5) on a batch of wedges.

    Parameters
    ----------
    corners : _Corners
        The wedge table.
    picked : IntArray
        Which records of it to solve.
    times_s : FloatArray
        Current arrivals at every vertex.

    Returns
    -------
    FloatArray
        One candidate arrival per record, infinite where neither flank has been reached.
    """
    at_left = times_s[corners.left[picked]]
    at_right = times_s[corners.right[picked]]
    # Kimmel & Sethian order the flanks so that T(A) <= T(B); which is which is a
    # property of the current solution, not of the mesh, so it is decided here.
    left_first = at_left <= at_right
    near_s = np.where(left_first, at_left, at_right)
    far_s = np.where(left_first, at_right, at_left)
    near_km = np.where(left_first, corners.left_km[picked], corners.right_km[picked])
    far_km = np.where(left_first, corners.right_km[picked], corners.left_km[picked])
    cosine = corners.cosine[picked]
    sine = corners.sine[picked]
    slowness = corners.slowness[picked]

    # The safe cap: a straight run along a real edge of the triangle, at that
    # triangle's own slowness. Always a valid upper bound, so it can never beat a
    # triangle root that is any good -- which is what makes offering it here the same
    # thing as `eikonal.rs`'s "only when no triangle produced a causal root".
    one_sided_s = np.minimum(near_s + near_km * slowness, far_s + far_km * slowness)

    both = np.isfinite(far_s)
    if not both.any():
        return one_sided_s

    # Both flanks are zeroed where the far one has not been reached: `inf` minus `inf`
    # is a NaN that would have to be masked out of every term below, and a wedge with
    # one flank is answered by `one_sided_s` regardless.
    gap_s = np.where(both, far_s, 0.0) - np.where(both, near_s, 0.0)
    quadratic = corners.chord_km2[picked]
    linear = 2.0 * near_km * gap_s * (far_km * cosine - near_km)
    constant = near_km**2 * (gap_s**2 - slowness**2 * far_km**2 * sine**2)
    discriminant = linear**2 - 4.0 * quadratic * constant
    root = np.where(
        discriminant >= 0.0,
        (-linear + np.sqrt(np.maximum(discriminant, 0.0))) / (2.0 * quadratic),
        np.nan,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        foot_km = near_km * (root - gap_s) / root
        # Equation (5), as a pair of products rather than a division by a cosine that
        # a right-angled triangle makes exactly zero. Both say the same thing: the
        # perpendicular from the apex meets the level line inside the segment.
        usable = (
            both
            & (discriminant >= 0.0)
            & (root > gap_s)
            & (far_km * cosine < foot_km)
            & (foot_km * cosine < far_km)
        )
    return np.where(usable, near_s + root, one_sided_s)


def _update(corners: _Corners, vertices: IntArray, times_s: FloatArray) -> FloatArray:
    """The best arrival each of ``vertices`` can be given from its own one-ring."""
    if vertices.size == 0:
        return np.empty(0, dtype=np.float64)
    picked = _ragged(corners.start, np.arange(corners.apex.size), vertices)
    candidate = _candidates(corners, picked, times_s)
    counts = corners.start[vertices + 1] - corners.start[vertices]
    best = np.full(vertices.size, np.inf)
    # A vertex belonging to no face has no wedge to be updated across and keeps its
    # infinity; `_reachable` is what turns that into a refusal, once, at the end.
    filled = counts > 0
    if picked.size:
        best[filled] = np.fmin.reduceat(candidate, (np.cumsum(counts) - counts)[filled])
    return best


def _sweep(
    corners: _Corners,
    times_s: FloatArray,
    fixed: np.ndarray,
    start: IntArray,
    index: IntArray,
    max_sweeps: int,
) -> int:
    """Fu et al.'s active list, run until it empties.

    Algorithm 2.1. The list starts as the unfixed neighbours of the boundary. Each pass
    updates every vertex on it; a vertex whose arrival stopped moving to within
    :data:`SETTLED_TOLERANCE_S` is taken off and its own unfixed neighbours are offered
    an update, joining the list if it improves them. That is the whole method: no heap,
    no global ordering, and a vertex may re-enter the list when a faster path overtakes
    an earlier one.

    **The pass is batched rather than Gauss-Seidel**, which is a deviation from the
    paper's prose ("each update is immediately transferred to the solution to be used by
    subsequent updates") and not from its method: Fu et al.'s third design premise is
    that "the algorithm is able to simultaneously update multiple points", and their own
    multithreaded variant updates arbitrary sublists of the active list at once. Their
    three correctness conditions -- everything inconsistent is on the list, nothing
    leaves while inconsistent, termination only on an empty list -- are what the proof
    rests on, and a batched pass satisfies all three. It changes how many passes the
    list takes, not what it converges to.

    Returns
    -------
    int
        How many passes it took.
    """
    seeded = np.flatnonzero(fixed)
    active = _neighbours_of(start, index, seeded)
    active = active[~fixed[active]]

    sweeps = 0
    while active.size:
        sweeps += 1
        if sweeps > max_sweeps:
            raise ValueError(
                f"the active list did not empty in {max_sweeps} passes, which is "
                f"{MAX_SWEEP_FACTOR} times this mesh's own ring count; the medium has "
                "structure this scheme does not handle, or the mesh has a fold the "
                "admissibility check did not catch"
            )
        before_s = times_s[active]
        candidate_s = _update(corners, active, times_s)
        times_s[active] = np.minimum(before_s, candidate_s)
        settled = (before_s - times_s[active]) <= SETTLED_TOLERANCE_S

        staying = active[~settled]
        leaving = active[settled]
        if leaving.size:
            nearby = _neighbours_of(start, index, leaving)
            nearby = nearby[~fixed[nearby]]
            nearby = nearby[~np.isin(nearby, staying)]
            if nearby.size:
                offered_s = _update(corners, nearby, times_s)
                improved = offered_s < times_s[nearby] - SETTLED_TOLERANCE_S
                times_s[nearby[improved]] = offered_s[improved]
                staying = np.concatenate([staying, nearby[improved]])
        active = staying
    return sweeps


# ============================================================================
# The public solve
# ============================================================================


def _checked(
    vertices_km: FloatArray, faces: IntArray, slowness_s_per_km: FloatArray
) -> tuple[FloatArray, IntArray, FloatArray]:
    """Refuse anything that does not describe a medium, in `eikonal.rs`'s vocabulary."""
    positions = np.asarray(vertices_km, dtype=np.float64)
    connectivity = np.asarray(faces, dtype=np.int64)
    slowness = np.asarray(slowness_s_per_km, dtype=np.float64).reshape(-1)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"vertices_km has shape {positions.shape}; a surface in space needs "
            "(V, 3) positions"
        )
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(
            f"faces has shape {connectivity.shape}; this solver is triangular, so "
            "every face needs exactly 3 corners"
        )
    if connectivity.shape[0] == 0 or positions.shape[0] == 0:
        raise ValueError(
            f"a mesh of {positions.shape[0]} vertices and {connectivity.shape[0]} "
            "faces has no surface to solve on"
        )
    if not np.isfinite(positions).all():
        raise ValueError("vertices_km carries a non-finite position")
    out_of_range = (connectivity < 0) | (connectivity >= positions.shape[0])
    if out_of_range.any():
        face = int(np.flatnonzero(out_of_range.any(axis=1))[0])
        raise ValueError(
            f"face {face} is {connectivity[face].tolist()}, which is outside a mesh of "
            f"{positions.shape[0]} vertices"
        )
    if slowness.size != connectivity.shape[0]:
        raise ValueError(
            f"the slowness field has {slowness.size} values, but a mesh of "
            f"{connectivity.shape[0]} faces needs one per face"
        )
    bad = ~np.isfinite(slowness) | (slowness <= 0.0)
    if bad.any():
        face = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"slowness on face {face} is {float(slowness[face])} s/km; every face must "
            "be positive and finite, or the faces behind it are unreachable"
        )
    return positions, connectivity, slowness


def _reachable(times_s: FloatArray) -> None:
    """Refuse a mesh whose components do not all hold a seed."""
    stranded = ~np.isfinite(times_s)
    if stranded.any():
        raise ValueError(
            f"{int(stranded.sum())} vertices are never reached, the first being "
            f"{int(np.flatnonzero(stranded)[0])}; they lie in a component of the mesh "
            "that holds no seed, so give that component a seed or drop it"
        )


def solve_from_boundary(
    vertices_km: FloatArray,
    faces: IntArray,
    slowness_s_per_km: FloatArray,
    boundary_vertices: IntArray,
    boundary_times_s: FloatArray,
    *,
    backend: str = NUMPY,
    threads: int = 1,
) -> FloatArray:
    """meshFIM proper: first arrivals from an arbitrary Dirichlet boundary condition.

    Fu et al. (2011) with Kimmel & Sethian's local solver, and nothing else -- no ball,
    no analytic seed, no choice of ``r0``. :func:`solve` is this function with the
    boundary chosen as the analytic geodesic ball; handing it a single vertex instead
    reproduces the **point-source** boundary condition that Fu et al.'s own convergence
    study reports as not first-order accurate.

    That is why it is public. `tests/triangular/test_fim.py` measures both boundary
    conditions through this one entry point, so the degraded slope and the recovered
    slope are the same code reading two boundaries rather than two code paths.

    Parameters
    ----------
    vertices_km : FloatArray
        ``(V, 3)`` positions in the projected CRS, kilometres, depth positive down.
    faces : IntArray
        ``(F, 3)`` vertex indices.
    slowness_s_per_km : FloatArray
        ``(F,)`` piecewise constant per triangle.
    boundary_vertices : IntArray
        Which vertices are held fixed.
    boundary_times_s : FloatArray
        Their arrivals, in seconds. Same length as ``boundary_vertices``.
    backend : str, optional
        :data:`NUMPY`, the reference, or :data:`KERNEL`, the Rust one. Defaults to the
        reference so that nothing that already calls this function changes answer.
    threads : int, optional
        Workers, for :data:`KERNEL` only. ``1`` is the sequential Gauss-Seidel path and
        the only one whose answer is bit-reproducible; ``0`` takes one per core. Within a
        single solve threading buys 1.6x on eight cores and costs about 1e-3 s of
        reproducibility, so the default is one.

    Returns
    -------
    FloatArray
        ``(V,)`` first arrivals in seconds.

    Raises
    ------
    ValueError
        If the mesh, the slowness or the boundary does not describe a solvable problem,
        or if the active list does not empty; one message per way in, each naming what
        failed.
    """
    positions, connectivity, slowness = _checked(vertices_km, faces, slowness_s_per_km)
    held = np.asarray(boundary_vertices, dtype=np.int64).reshape(-1)
    held_s = np.asarray(boundary_times_s, dtype=np.float64).reshape(-1)
    if held.size == 0:
        raise ValueError("no boundary vertices: a wavefront needs somewhere to start")
    if held_s.size != held.size:
        raise ValueError(
            f"{held.size} boundary vertices carry {held_s.size} times; each held "
            "vertex needs exactly one arrival"
        )
    out_of_range = (held < 0) | (held >= positions.shape[0])
    if out_of_range.any():
        first = int(held[np.flatnonzero(out_of_range)[0]])
        raise ValueError(
            f"boundary vertex {first} is outside a mesh of {positions.shape[0]} "
            "vertices"
        )
    if not np.isfinite(held_s).all():
        first = int(np.flatnonzero(~np.isfinite(held_s))[0])
        raise ValueError(
            f"boundary vertex {int(held[first])} starts at t = {float(held_s[first])}, "
            "which is not a time"
        )

    if backend not in BACKENDS:
        raise ValueError(
            f"{backend!r} is not a backend; the two are {NUMPY!r} and {KERNEL!r}"
        )
    if backend == KERNEL:
        return _kernel_solve(
            positions, connectivity, slowness, [(held, held_s)], threads
        )
    times_s, _ = _solve_held(positions, connectivity, slowness, held, held_s)
    return times_s


def _kernel_solve(
    positions: FloatArray,
    connectivity: IntArray,
    slowness: FloatArray,
    boundaries: list[tuple[IntArray, FloatArray]],
    threads: int,
) -> FloatArray:
    """Hand one or more boundaries to `crates/kernels/src/fim.rs`.

    The one seam. Arrays are made contiguous and typed here rather than at each call
    site, because `numpy`'s ``as_slice`` refuses a strided view and the failure would
    surface as a type error a long way from the transpose that caused it.
    """
    from rupture_generator import _kernels

    times_s, _passes, _visits, _unsplit = _kernels.fim_solve(
        np.ascontiguousarray(positions, dtype=np.float64),
        np.ascontiguousarray(connectivity, dtype=np.int64),
        np.ascontiguousarray(slowness, dtype=np.float64),
        [
            (
                np.ascontiguousarray(held, dtype=np.int64),
                np.ascontiguousarray(held_s, dtype=np.float64),
            )
            for held, held_s in boundaries
        ],
        threads,
    )
    return times_s


def _solve_held(
    positions: FloatArray,
    connectivity: IntArray,
    slowness: FloatArray,
    held: IntArray,
    held_s: FloatArray,
) -> tuple[FloatArray, tuple[int, int]]:
    """The solve itself, on already-checked inputs; also the sweep and wedge counts."""
    corners = _corners(positions, connectivity, slowness)
    start, index = _adjacency(connectivity, positions.shape[0])

    times_s = np.full(positions.shape[0], np.inf)
    # Several boundary vertices may name the same vertex; the earliest wins, which is
    # the same pointwise minimum multiple seeds are combined by.
    np.fmin.at(times_s, held, held_s)
    fixed = np.zeros(positions.shape[0], dtype=bool)
    fixed[held] = True

    rings = _hops(start, index, held, positions.shape[0])
    max_sweeps = MAX_SWEEP_FACTOR * (int(rings.max()) + 1)
    sweeps = _sweep(corners, times_s, fixed, start, index, max_sweeps)
    _reachable(times_s)
    return times_s, (sweeps, corners.unsplit_obtuse)


def _ball(
    positions: FloatArray,
    connectivity: IntArray,
    slowness: FloatArray,
    start: IntArray,
    index: IntArray,
    seed: Seed,
) -> tuple[IntArray, FloatArray, float, float, float, float, float, int]:
    """The analytic geodesic ball around one seed, and the two bounds that pin it.

    ``r0`` is the radius that reaches :data:`SEED_RING_DEPTH` rings of the mesh graph.
    Chordal distance is what the seeded times use: inside the ball it agrees with the
    geodesic distance to ``O(kappa^2 r0^2)``, and with ``r0`` a few cells across that is
    negligible and boundable from the mesh's own discrete curvature -- which is the
    whole reason the ball is small.
    """
    reach_km = np.linalg.norm(positions - positions[seed.vertex], axis=1)
    rings = _hops(start, index, np.array([seed.vertex]), positions.shape[0])
    within = (rings >= 0) & (rings <= SEED_RING_DEPTH)
    radius_km = float(reach_km[within].max())

    # A vertex has no slowness of its own, so S0 is the area-weighted mean over the
    # faces meeting it -- the P0-to-P1 reduction `face_arrivals` inverts.
    touching = np.flatnonzero((connectivity == seed.vertex).any(axis=1))
    corners_km = positions[connectivity[touching]]
    areas_km2 = 0.5 * np.linalg.norm(
        np.cross(
            corners_km[:, 1] - corners_km[:, 0], corners_km[:, 2] - corners_km[:, 0]
        ),
        axis=1,
    )
    source_slowness = float(
        np.average(slowness[touching], weights=areas_km2)
        if areas_km2.sum() > 0.0
        else slowness[touching].mean()
    )

    seeded = reach_km <= radius_km
    held = np.flatnonzero(seeded)
    held_s = source_slowness * reach_km[held] + seed.t0_s

    # How far the seeded boundary is from the circle it stands for: the first ring of
    # vertices the front actually leaves from.
    ring = _neighbours_of(start, index, held)
    ring = ring[~seeded[ring]]
    spread = (
        float(reach_km[ring].max() - reach_km[ring].min()) / radius_km
        if ring.size
        else 0.0
    )

    # How far from constant the slowness the seed asserts actually is, over every face
    # the ball touches.
    covered = seeded[connectivity].any(axis=1)
    departure = float(np.abs(slowness[covered] - source_slowness).max())
    return (
        held,
        held_s,
        radius_km,
        source_slowness,
        spread,
        departure / source_slowness,
        radius_km * departure,
        int(ring.size),
    )


def solve(
    vertices_km: FloatArray,
    faces: IntArray,
    slowness_s_per_km: FloatArray,
    seeds: Sequence[Seed],
    *,
    backend: str = NUMPY,
    threads: int = 1,
) -> FloatArray:
    """First-arrival times at every vertex, from every seed.

    meshFIM with the analytic geodesic-ball boundary condition of `MESH.md`'s
    Component 3: each seed's ball is fixed to ``T = S0 d(x, x0) + t0``, the front is
    iterated outward from there, and the seeds are combined by pointwise minimum.

    Parameters
    ----------
    vertices_km : FloatArray
        ``(V, 3)`` positions in the projected CRS, kilometres, depth positive down.
    faces : IntArray
        ``(F, 3)`` vertex indices.
    slowness_s_per_km : FloatArray
        ``(F,)`` piecewise constant per triangle.
    seeds : Sequence of Seed
        Where the front starts, and when.
    backend : str, optional
        :data:`NUMPY` or :data:`KERNEL`; see :func:`solve_from_boundary`.
    threads : int, optional
        Workers, for :data:`KERNEL` only. Across seeds the answer is bit-identical at any
        thread count, so a multi-segment rupture can take all the cores it likes.

    Returns
    -------
    FloatArray
        ``(V,)`` first arrivals in seconds.

    Raises
    ------
    ValueError
        As :func:`solve_from_boundary`, plus a seed outside the mesh or with a
        non-finite start time.

    Warns
    -----
    DegradedSeed
        If a ball's constant-slowness assumption costs more than
        :data:`SEED_SLOWNESS_BUDGET_S`.
    """
    return solve_with_report(
        vertices_km, faces, slowness_s_per_km, seeds, backend=backend, threads=threads
    )[0]


def solve_with_report(
    vertices_km: FloatArray,
    faces: IntArray,
    slowness_s_per_km: FloatArray,
    seeds: Sequence[Seed],
    *,
    backend: str = NUMPY,
    threads: int = 1,
) -> tuple[FloatArray, tuple[SeedReport, ...]]:
    """:func:`solve`, also reporting what each seed's ball was and how well it held.

    Exposed because ``r0``'s two bounds are evidence rather than diagnostics: `MESH.md`
    derives ``r0`` from the mesh and the velocity model precisely so that it is not a
    configured parameter, and a derived quantity nobody measures is a configured one
    that has stopped being written down.

    Parameters
    ----------
    vertices_km : FloatArray
        ``(V, 3)`` positions in the projected CRS, kilometres, depth positive down.
    faces : IntArray
        ``(F, 3)`` vertex indices.
    slowness_s_per_km : FloatArray
        ``(F,)`` piecewise constant per triangle.
    seeds : Sequence of Seed
        Where the front starts, and when.
    backend : str, optional
        :data:`NUMPY` or :data:`KERNEL`; see :func:`solve_from_boundary`.
    threads : int, optional
        Workers, for :data:`KERNEL` only.

    Returns
    -------
    tuple
        ``(V,)`` first arrivals in seconds, and one :class:`SeedReport` per seed in the
        order they were given.

    Raises
    ------
    ValueError
        As :func:`solve`.

    Warns
    -----
    DegradedSeed
        As :func:`solve`.
    """
    positions, connectivity, slowness = _checked(vertices_km, faces, slowness_s_per_km)
    if not seeds:
        raise ValueError("no seeds: a wavefront needs somewhere to start")
    for order, seed in enumerate(seeds):
        if not 0 <= seed.vertex < positions.shape[0]:
            raise ValueError(
                f"seed {order} at vertex {seed.vertex} is outside a mesh of "
                f"{positions.shape[0]} vertices"
            )
        if not np.isfinite(seed.t0_s):
            raise ValueError(
                f"seed {order} starts at t = {seed.t0_s}, which is not a time"
            )

    if backend not in BACKENDS:
        raise ValueError(
            f"{backend!r} is not a backend; the two are {NUMPY!r} and {KERNEL!r}"
        )

    start, index = _adjacency(connectivity, positions.shape[0])
    combined_s = np.full(positions.shape[0], np.inf)
    reports: list[SeedReport] = []
    # **The ball is derived here whichever backend solves.** `r0`, its two bounds and the
    # warning are policy, and `MESH.md` puts policy in Python; the kernel takes the
    # boundary that falls out. So the two backends are compared on one boundary condition
    # rather than on two derivations of it.
    balls = [
        _ball(positions, connectivity, slowness, start, index, seed) for seed in seeds
    ]
    kernel_s = (
        _kernel_solve(
            positions,
            connectivity,
            slowness,
            [(ball[0], ball[1]) for ball in balls],
            threads,
        )
        if backend == KERNEL
        else None
    )
    for seed, ball in zip(seeds, balls, strict=True):
        (
            held,
            held_s,
            radius_km,
            source_slowness,
            spread,
            slowness_spread,
            slowness_error_s,
            ring_size,
        ) = ball
        if kernel_s is None:
            times_s, (sweeps, unsplit) = _solve_held(
                positions, connectivity, slowness, held, held_s
            )
            np.fmin(combined_s, times_s, out=combined_s)
        else:
            # One kernel call already combined every seed, so the counts are the whole
            # solve's rather than this seed's and are reported as zero rather than as a
            # number that would mean something different from the numpy path's.
            combined_s = kernel_s
            sweeps, unsplit = 0, 0
        reports.append(
            SeedReport(
                vertex=seed.vertex,
                radius_km=radius_km,
                source_slowness_s_per_km=source_slowness,
                seeded_vertices=int(held.size),
                boundary_vertices=ring_size,
                boundary_radius_spread=spread,
                slowness_spread=slowness_spread,
                slowness_error_s=slowness_error_s,
                unsplit_obtuse_wedges=unsplit,
                sweeps=sweeps,
            )
        )
        if slowness_error_s > SEED_SLOWNESS_BUDGET_S:
            warnings.warn(
                f"the analytic ball around vertex {seed.vertex} spans "
                f"{radius_km:.3g} km, over which the slowness varies by "
                f"{slowness_spread * 100:.1f}% -- so fixing it at "
                f"{source_slowness:.4g} s/km costs up to "
                f"{slowness_error_s * 1e3:.1f} ms at the ring, past the "
                f"{SEED_SLOWNESS_BUDGET_S * 1e3:.0f} ms budget. The hypocentre's own "
                "onset carries no perturbation to hide that in. Refine the mesh near "
                "the seed, or accept a seeded onset biased by that much",
                DegradedSeed,
                stacklevel=3,
            )
    _reachable(combined_s)
    return combined_s, tuple(reports)


def face_arrivals(faces: IntArray, vertex_times_s: FloatArray) -> FloatArray:
    """Carry vertex arrivals to faces: the mean of each triangle's three corners.

    The solver works at vertices, because that is what P1 finite elements and meshFIM
    both do; the pipeline attaches fields per face, because a subfault is a patch that
    slips. This is the seam.

    **The mean, and not the minimum.** The solution is piecewise linear over each
    triangle by construction -- that is what the local solver builds -- so the mean of
    the three corners is *exactly* the interpolated arrival at the centroid, which is
    the point the face's moment tensor is placed at. The minimum would report when the
    first corner broke rather than when the subfault did, and would shift the whole
    field early by about ``h S / 3``: a bias, uniform across the fault, in the one
    quantity `MESH.md` warns does not average out and has no perturbation to hide
    behind. The maximum has the mirror problem. The mean is the only one of the three
    that is a statement about the discretisation rather than about the mesh.

    Parameters
    ----------
    faces : IntArray
        ``(F, 3)`` vertex indices.
    vertex_times_s : FloatArray
        ``(V,)`` arrivals from :func:`solve`.

    Returns
    -------
    FloatArray
        ``(F,)`` arrivals in seconds.
    """
    connectivity = np.asarray(faces, dtype=np.int64)
    times_s = np.asarray(vertex_times_s, dtype=np.float64).reshape(-1)
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise ValueError(
            f"faces has shape {connectivity.shape}; this solver is triangular, so "
            "every face needs exactly 3 corners"
        )
    out_of_range = (connectivity < 0) | (connectivity >= times_s.size)
    if out_of_range.any():
        face = int(np.flatnonzero(out_of_range.any(axis=1))[0])
        raise ValueError(
            f"face {face} is {connectivity[face].tolist()}, which is outside a field "
            f"of {times_s.size} vertex arrivals"
        )
    return times_s[connectivity].mean(axis=1)


__all__ = [
    "BACKENDS",
    "KERNEL",
    "MAX_SWEEP_FACTOR",
    "NUMPY",
    "SEED_RING_DEPTH",
    "SEED_SLOWNESS_BUDGET_S",
    "SETTLED_TOLERANCE_S",
    "UNFOLD_LIMIT",
    "DegradedSeed",
    "Seed",
    "SeedReport",
    "face_arrivals",
    "solve",
    "solve_from_boundary",
    "solve_with_report",
]
