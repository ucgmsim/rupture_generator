"""Von Karman fields on a triangulated surface, by the Whittle-Matern SPDE.

Sampled by solving

.. math:: (\\kappa^2 - \\nabla \\cdot H \\nabla)^{\\alpha/2} u = W,
          \\qquad \\alpha = \\nu + d/2

with piecewise-linear finite elements on the triangulation, following Lindgren,
Rue & Lindstrom (2011) for the operator and Bolin & Kirchner (2020) for the
non-integer power. It replaces `sampling.py`'s circulant embedding, which is
definitionally a lattice method: its covariance is block-circulant only because
lag is index-difference times spacing.

**Why an SPDE rather than a geodesic-distance Matern.** The obvious alternative
-- compute distances between points *on the surface* and evaluate
:func:`~rupture_generator.sampling.von_karman_correlation` -- is not guaranteed
to produce a valid random field at all. Gneiting (2013), *Bernoulli* 19(4),
example 2, settles the Matern family on spheres: the correlation is strictly
positive definite on spheres of every dimension when ``nu <= 1/2``, and for
``nu > 1/2`` it "does not belong to the class Psi_1, and thus neither to any of
the classes Psi_d" -- it is not positive definite on any sphere. Mai's
``nu = H = 0.75`` is outside the permitted range, so a geodesic-distance von
Karman ACF on a curved fault has no theorem behind it. Lindgren et al. (2011)
section 3.1 makes the same argument (via Gneiting 1998, theorem 2) and draws the
same conclusion: "we can still use its origin, the SPDE ... the solution is still
what we mean by a Matern field, but defined directly for the given manifold". The
SPDE solution is positive definite by construction, because it is defined by an
operator rather than by a covariance function.

**Why the lifted triangles.** The stiffness matrix is assembled from the
triangles' true three-dimensional geometry -- true areas, true edge cotangents --
so the discrete operator is the Laplace-Beltrami operator *of the fault surface*.
Assembling from the projected ``(u, v)`` triangles instead would give the flat
Laplacian of the parameter domain, which is exactly "drape flat noise over curved
geometry": two points 1 km apart in projection but 1.05 km apart on the fault
would receive the 1 km correlation. The difference is the whole of what makes
this intrinsically Matern on the surface, and it costs nothing -- no arc-length
integration and no distance computation appears anywhere in this module.

The model
---------

Mai & Beroza (2002) equation (1) writes the von Karman ACF at the dimensionless
distance ``r = sqrt(x^2/a_x^2 + z^2/a_z^2)``. That is the isotropic Matern of
smoothness ``H`` in coordinates scaled by the two correlation lengths, so the
SPDE that delivers it is the one above with

- ``kappa = 1``, and
- ``H = a_s^2 e_s e_s^T + a_d^2 e_d e_d^T``, the anisotropy tensor whose
  eigenvalues are the *squared* correlation lengths along strike and down dip.

Folding the correlation lengths into ``H`` rather than into ``kappa`` is what
makes the whole operator dimensionless: the symbol is ``1 + a_s^2 k_s^2 +
a_d^2 k_d^2``, whose reciprocal is Mai's spectrum, and the smallest eigenvalue of
the discrete operator is ``>= 1`` -- which is the normalisation Bolin & Kirchner
section 3.5 asks for anyway ("we will rescale the operator L so that its
eigenvalues are bounded from below by one").

``e_s`` and ``e_d`` are the **surface's own** strike and dip: ``e_s`` is the
horizontal direction in the tangent plane, ``e_d`` the steepest descent. They are
computed from the face normal alone (:func:`_surface_frames`, matching
`mesh.RuptureMesh.strike_dip_deg`), so ``H`` is diagonal in that frame by
construction and none of Fuglstad et al.'s varying-local-anisotropy machinery is
needed. On a planar fault they collapse onto the Monge frame's ``(e_u, e_v)``
exactly, which `test_planar_frame_is_the_planes_strike_and_dip` asserts.

Where the surface goes horizontal the strike direction ceases to exist, and the
anisotropy is faded to isotropic rather than read off noise --
:data:`DEGENERATE_DIP_SINE` and :func:`_anisotropy` say how, and why the fade
preserves the marginal variance exactly. This is not hypothetical: 1.55% of the
shipped Hikurangi interface is flat enough to need it.

**Nothing in the operator reads the parameter coordinates.** Strike and dip are
properties of the surface, not of how it was parameterised, so the same geometry
meshed through a different chart yields the same field -- which is what makes
this sampler mesh-native, and is asserted by `test_the_operator_is_mesh_native`.
Taking the frame from the parameterisation instead (from ``dX/du``) was tried and
rejected: on the shipped CFM interfaces the parameter frame is up to 41 degrees
from orthogonal on the surface, and choosing which of the two parameter axes to
orthogonalise from moved the delivered correlation length by up to 29%.

What this costs
---------------

**The solve, and how far it reaches.** `scikit-sparse` is not a dependency, so
the sparse factorisations come from `scipy.sparse.linalg.splu` (SuperLU) in its
symmetric mode -- `permc_spec="MMD_AT_PLUS_A"`, `diag_pivot_thresh=0`,
`SymmetricMode` -- which is the documented way to get Cholesky-like fill without
one. Every matrix factorised here is symmetric positive definite (see
:class:`MaternOperator`), so a true Cholesky would still roughly halve the
factor storage and the flops; SuperLU computes ``L`` *and* ``U``.

Measured on the CFM Hikurangi interface refined 1-to-4, at Mw 8.5:

=====  =========  =========  ==========  =========  ========
level  edge (km)  vertices   setup       draw       peak
=====  =========  =========  ==========  =========  ========
1      3.70       19,671     0.15 s      0.008 s    0.13 GB
2      1.79       76,285     2.96 s      0.048 s    0.48 GB
3      0.87       300,345    392.6 s     0.511 s    6.16 GB
=====  =========  =========  ==========  =========  ========

Setup grows by 130x per 4x vertices -- that is fill-in, and it puts 400 m
(1.19 M vertices) out of reach by this route: extrapolating the same exponent
gives roughly 14 hours and 50 GB.

**The iterative alternative, and what actually limits it.** A draw needs *solves*
with ``P_l``, not a factorisation of it, and every shifted operator is SPD, so
conjugate gradients is admissible and its memory is linear. What it costs is set
by the mesh's element quality, not by the equation. Measured at Mw 8.5, Jacobi
preconditioned, per shifted factor:

===============================  =========  ==============  =================
mesh                             ``h``      area max/min    CG iterations
===============================  =========  ==============  =================
regular triangulation, 66 k      0.088      1.0             540, 108, 19
CFM Hikurangi refined, 76 k      0.187      4.3e4           3214, 1961, 940
===============================  =========  ==============  =================

-- an order of magnitude more iterations on the CFM triangulation at *half* the
resolution, because its worst elements set the largest eigenvalue. On a
well-shaped mesh the count grows as ``1/h``, which is the textbook rate and is
affordable; on a mesh with a four-decade area spread it is not. `MESH.md`
Component 1 triangulates the parameter domain with Delaunay, which produces the
well-shaped case, so this is an argument for building the mesh rather than
refining the supplied one. An incomplete-Cholesky preconditioner was tried and
abandoned: `spilu` inherits the same fill-in and did not finish 300 k vertices in
nine minutes.

**But the field does not need a fine mesh.** The correlation length at Mw 8.5 is
21.5 km down dip, and the delivered covariance is already within ~1% of the ACF
at ``h = 0.18`` correlation lengths -- level 2 above, 1.79 km edges, three
seconds and half a gigabyte. Solving at 100 m would be oversampling the field by
a factor of twenty in each direction. The finite element solution *is* a
piecewise-linear function, so evaluating it on a finer mesh is exact rather than
approximate: sample on a mesh sized by the correlation length and interpolate
onto whatever mesh the eikonal and the output want. That dissolves the scaling
problem rather than solving it, and it is the recommended route.

The factorisations are built once per :class:`MaternOperator` and reused across
draws, which is what makes the four fields a segment needs affordable: on the
185150-vertex `colombia` mesh, 0.23 s of draws against a 2.5 s setup, of which
0.5 s is numpy assembly and 1.5 s SuperLU.

**Boundary folding, which is the largest caveat in this module.** The SPDE is
solved with the natural (Neumann) boundary condition, which does not give a
stationary field: Lindgren et al. (2011) appendix A.4, theorem 1 shows the
resulting covariance is the Matern one *folded* at the boundary,
``cov(u, v) = sum_k [r_M(u, v - 2kL) + r_M(u, 2kL - v)]``. The same appendix
states its reach -- the folded covariance "is nearly indistinguishable from the
stationary Matern covariance at distances greater than twice the range away from
the borders of the domain" -- which on a large domain makes it an edge effect.

**Faults are not large domains.** Mai & Beroza figure 13 puts the correlation
length between 0.25 and 0.6 of the source dimension, so a fault is between 1.7
and 4 correlation lengths across *by construction of the model*, and the shipped
`colombia` example is 17.4 km wide against a 9.26 km down-dip correlation length
-- 1.9 lengths, entirely inside the fold. Measured there (`_warn_if_folded`), the
marginal variance is 2.6 times the model's and the correlation at one correlation
length is 0.878 against 0.5005. The circulant sampler delivers 0.5005 at every
one of those sizes, because it pads the embedding and crops. So this is not a
discretisation difference between the two samplers, it is a different field, and
it is the one thing that must be settled before the triangular path can reproduce
the quad results. The fix is the circulant sampler's own -- triangulate a
parameter domain extended by :data:`BOUNDARY_FOLDING_LENGTHS` correlation lengths
on each side, solve there, keep the vertices inside the fault -- and it is a
change to *what mesh is built*, so it belongs to the mesh builder and not here.
Until it exists, :func:`_warn_if_folded` refuses to be quiet about it.

References
----------
Lindgren, F., Rue, H. & Lindstrom, J. (2011). An explicit link between Gaussian
fields and Gaussian Markov random fields: the stochastic partial differential
equation approach. *JRSS-B* 73(4), 423-498. Sections 2-3.1; appendix A.2 for the
element matrices, A.4 for the boundary folding.

Bolin, D. & Kirchner, K. (2020). The rational SPDE approach for Gaussian random
fields with general smoothness. *JCGS* 29(2), 274-285 (arXiv:1711.04333v4).
Sections 3.3-3.5 and appendix A for the rational approximation; theorem 3.3 for
the error bound.

Mai, P. M. & Beroza, G. C. (2002). A spatial random field model to characterize
complexity in earthquake slip. *JGR* 107(B11), ESE 10. Equation (1), figures 11
and 13.

Gneiting, T. (2013). Strictly and non-strictly positive definite functions on
spheres. *Bernoulli* 19(4), 1327-1349. Example 2.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from rupture_generator.sampling import (
    MAI_MAXIMUM_RATIO,
    DegradedCorrelation,
    VonKarmanFilterParameters,
)

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

MANIFOLD_DIMENSION = 2
"""The ``d`` of the SPDE: a fault surface is a 2-manifold, whatever it is embedded in.

It is the dimension of the *domain*, not of the space the domain sits in, and it
enters twice: through ``alpha = nu + d/2`` (Lindgren et al. 2011, equation 2) and
through Bolin & Kirchner's convergence rate, which is ``min(2*beta - d/2, 2)``.
Sampling a curved fault as a surface rather than as a solid is the whole reason
``d = 2`` and not ``3``.
"""

RATIONAL_ORDER = 2
"""How many terms the rational approximation of ``L^-beta`` gets, Bolin & Kirchner's ``m``.

Bounded from below by accuracy and from above by cost, and both bounds are
measured, in `test_rational_order_is_the_cheapest_that_converges`.

Below: the delivered covariance error on a mesh at ``h = 0.177`` correlation
lengths is 1.77e-2 at ``m = 1`` against 1.08e-2 at ``m = 2`` -- 64% worse, which
is the rational term of theorem 3.3 overtaking the finite element term, exactly
as the bound says it must (at that ``h`` the two terms are 0.375 and 0.273 for
``m = 1``, and 0.098 and 0.273 for ``m = 2``).

Above: ``m = 3`` measures 1.13e-2 at the same mesh -- no better than ``m = 2``,
because the error there is the finite element discretisation and not the rational
approximation -- while costing 31% more setup and 39% more per draw at fault
scale (3.25 s and 79 ms against 2.48 s and 57 ms on the 185k-vertex `colombia`
mesh). Bolin & Kirchner remark 4.2 also warns the precision "can be
ill-conditioned for m > 1 if a FEM approximation with piecewise linear basis
functions is used", which is the basis used here, and the fit itself starts to
degrade past ``m = 4``: its supremum error stops falling (1.6e-3 at ``m = 4``,
1.9e-3 at ``m = 5``).
"""

BOUNDARY_FOLDING_LENGTHS = 2.0
"""How many correlation lengths from the boundary the Neumann folding reaches.

Lindgren et al. (2011) appendix A.4: the folded covariance "is nearly
indistinguishable from the stationary Matern covariance at distances greater than
twice the range away from the borders of the domain". Not a tolerance -- a
statement of where the sampler's field stops being the stationary one, reported
by :attr:`ModelError.boundary_reach` and warned about by :func:`_warn_if_folded`.
"""

LAWSON_ITERATIONS = 128
"""How many reweightings the rational fit takes before it is called minimax.

Lawson's algorithm converges to the equioscillating (supremum-optimal) rational
approximation that Bolin & Kirchner's theorem 3.3 assumes. Bolin & Kirchner reach
the same target with Remez, which they note "is often unstable in computations",
or with Clenshaw-Lord Chebyshev-Pade, which needs Chebfun.

Measured at ``beta = 0.875`` against a 256-iteration reference, worst over orders
1-4: 128 iterations is within 0.41%, 64 within 0.87%, 32 within 3.7% and 16
within 15%. The approach is **from below** -- each round moves the fit from
least-squares towards minimax, so its supremum error rises -- which means
under-iterating does not merely blur the fit, it makes
:attr:`RationalApproximation.supremum_error`, and therefore the reported model
error, optimistic. That is the direction worth paying for, and the whole fit is
a handful of SVDs of a 1024-by-11 matrix, cached per ``(beta, order)``.
"""

RATIONAL_SAMPLE_COUNT = 1024
"""How many points the rational fit is evaluated on.

The fit is a supremum-norm problem on a continuum, discretised. Chebyshev-spaced
points cluster where the extrema of the error go, and 1024 of them resolve the
``2m + 3 <= 11`` equioscillation points of every order this module admits by two
orders of magnitude. Measured: the supremum error moves by under 6e-6 between
1024 and 4096 points, against fitted errors of 4e-3 to 4e-2; at 512 points it
still moves by 7e-5, which is 2% of the ``m = 3`` fit.
"""

MODEL_ERROR_CONSTANT = 0.05
"""The ``C`` of Bolin & Kirchner theorem 3.3, measured rather than derived.

The theorem bounds the error by ``C * (h^min(2b-d/2,2) + h^min(2(b-1),0)-d/2 *
e_m)`` with ``C`` explicitly "independent of h, m" but otherwise unknown -- it
carries the domain, the operator and the norm. A bound with an unknown constant
cannot be reported as a number, so the constant is measured once and pinned here.

Measured by `test_bound_is_a_bound` as the ratio of the delivered covariance
error to the bracketed quantity, over a four-step refinement from ``h = 1.414``
to ``h = 0.177`` at ``m = 2``: the ratios are 0.0137, 0.0353, 0.0416 and 0.0291.
Across orders 1-3 as well the largest observed is 0.0458, so 0.05 is a bound on
all of them and is not loose by an order of magnitude.

**Flat is the whole content of the measurement.** A constant that drifted with
``h`` would mean the rate was wrong and the reported number was a fit rather than
a bound; these vary by a factor of three while ``h`` varies by a factor of eight
and the bracketed quantity by a factor of four, so the rate is right. That is
what `test_bound_is_a_bound` asserts, by requiring the ratio to stay within
``[0.25, 1.0]`` times this constant at every refinement level.
"""

DEGENERATE_DIP_SINE = np.radians(1.0) / np.radians(45.0)
"""The ``sin(dip)`` below which a patch is too flat to carry an anisotropy.

Derived, not chosen. The strike direction is ``cross(down, n)``, whose length is
``sin(dip)``, so a perturbation of ``eps`` in the normal's orientation swings the
strike direction by roughly ``eps / sin(dip)``. `ENGINEERING_RULES.md` bounds this
package's angular agreement at **one degree**; a swing of **45 degrees** is the
point at which strike and dip have effectively exchanged places, since the frame
is a right angle and half of it is the worst a rotation can do. Setting
``1 deg / sin(dip) = 45 deg`` gives ``sin(dip) = 1/45 = 0.0222``, a dip of 1.27
degrees.

Below that, a normal this package would call correct admits a frame in which the
two correlation lengths have swapped, so the anisotropy carries no information.
:func:`_anisotropy` fades it out rather than reading noise.

Measured on the shipped CFM interfaces: **Hikurangi has 1.55% of its area below
this dip** (its shallow trench-ward portion reaches 2.80 km depth and 0.024
degrees), Puyseguer 0% (its shallowest face dips 7.36 degrees) and
Puysegur-Fiordland 0% (2.10 degrees). So the fade is exercised by real geometry
and is not a hypothetical.
"""

MINIMUM_LUMPED_MASS_RATIO = 5.0e-6
"""How little of the median area a vertex may be left with before the mesh is refused.

A vertex's lumped mass is its share of the surface, and ``C~^-1`` sits in the
middle of the precision, so a vertex starved of area is barely constrained and
its variance runs away. Because the stages standardise a field by its sample
spread, one such vertex divides the whole segment -- see
:func:`_refuse_starved_vertices`.

**Placed by measurement, between the last ratio that works and the first that
does not.** The defect was reproduced from the one the CFM data actually carries
(a near-duplicate vertex 2.19 m from its twin on a mesh whose median edge is
7.24 km, carrying a single needle triangle) at controlled size on a healthy
45-degree dipping mesh, reading the standardised spread of the *healthy* part of
the field against the 1% slip bound:

==========  ============================
mass ratio  healthy field after standardise
==========  ============================
1.47e-4     1.0000
1.47e-5     0.9996
1.47e-6     **0.0487**
1.47e-7     0.0000
==========  ============================

The transition is a cliff, not a slope, and the shipped CFM interfaces fall
either side of it exactly as that predicts: Hikurangi at 2.36e-5 and
Puysegur-Fiordland at 2.35e-5 both deliver a healthy field (0.996, 1.000), and
Puyseguer at 7.35e-7 delivers 0.187. So the floor is set at the geometric middle
of the measured cliff -- 3.4 times above the worst ratio that failed, 2.9 below
the best that passed, and admitting both real meshes that work while refusing the
one that does not.

**It is a proxy, and a coarse one.** What actually fails is conditioning, which
depends on the starved vertex's neighbourhood as well as its own area, so a mesh
could in principle fail above this floor or survive below it. The floor is a
backstop for a defect whose real fix is upstream: a mesh with a duplicate vertex
should not be built, and the mesh builder's admissibility check is where that
belongs.
"""

_DOWN = np.array([0.0, 0.0, 1.0])
"""Depth positive down, matching `mesh.py`'s ``_DOWN`` and the whole package."""

_EAST = np.array([1.0, 0.0, 0.0])
"""An arbitrary axis, used only to fill the frame on an exactly horizontal patch.

Which axis it is cannot matter: :func:`_anisotropy` gives a horizontal patch two
equal eigenvalues, so the frame it is expressed in drops out. It exists so the
arithmetic never divides by zero.
"""

_LEADING_COEFFICIENT_FLOOR = 1.0e-12
"""Below this the fitted numerator or denominator has no leading term to divide by.

The rational fit is normalised to unit coefficient vector, so a leading
coefficient this small means the fit collapsed to a lower degree and its roots
are not the ``m`` and ``m + 1`` the construction needs.
"""


def matern_exponent(hurst: float) -> float:
    """The SPDE exponent ``alpha`` for a von Karman roughness, on a surface.

    Lindgren et al. (2011) equation (2): ``alpha = nu + d/2``. The von Karman
    Hurst exponent *is* the Matern smoothness ``nu``, so Mai's ``H = 0.75`` on a
    2-manifold gives ``alpha = 1.75`` -- non-integer, which is exactly why the
    plain sparse finite element method of Lindgren et al. cannot represent it and
    Bolin & Kirchner's rational approximation is needed.

    Parameters
    ----------
    hurst : float
        The von Karman roughness exponent, in ``(0, 1)``.

    Returns
    -------
    float
        ``alpha``.
    """
    return hurst + MANIFOLD_DIMENSION / 2.0


def _rational_shift(beta: float) -> int:
    """Bolin & Kirchner's ``m_beta = max(1, floor(beta))``, equation (3.8)."""
    return max(1, int(np.floor(beta)))


def _interval_floor(order: int) -> float:
    """The ``delta`` of the fitting interval ``[delta, 1]``, Bolin & Kirchner section 3.5.

    Their choice verbatim: ``delta = 10^-(5+m)/2``, which they report "gives
    acceptable results for all values of beta". The interval has to start above
    zero because the function being approximated, ``x^(beta - m_beta)``, has a
    negative exponent here and is unbounded at the origin.

    **Fixed, deliberately, and not widened to cover the mesh's own spectrum.**
    The same section says ``delta`` "should ideally be chosen such that J_h is
    contained in J* for all considered mesh sizes h", and the meshes this package
    builds do not satisfy that: the shipped `colombia` example is cut at 0.1 km
    against a 9.3 km down-dip correlation length, which puts the bottom of its
    spectrum at 5e-6 against a ``delta`` of 3.2e-4 at ``m = 2``, so the fit is
    extrapolated over nearly two decades. Widening ``delta`` to cover it was
    tried and is **worse**, measured on a mesh at ``h = 0.044`` correlation
    lengths (`test_fixed_fitting_interval_beats_the_mesh_spectrum`):

    ==================  =================  ==========================
    ``delta``           supremum error     delivered covariance error
    ==================  =================  ==========================
    3.2e-4 (the default) 1.13e-2           8.3e-3
    1e-5 (this mesh)     5.25e-2           1.62e-2
    1e-6 (`colombia`)    1.08e-1           4.35e-2
    ==================  =================  ==========================

    -- so covering the spectrum costs a factor of two, and covering a real
    fault's spectrum a factor of five, in the quantity that matters.

    The reason is that the supremum norm weights the whole spectrum equally and
    the field does not. The variance in the mode at eigenvalue ``lambda`` falls
    as ``lambda^-2beta``, so the ``lambda ~ 1e5`` end of a fine mesh's spectrum
    carries ~1e-9 of the variance of the ``lambda ~ 1`` end: buying accuracy
    there, at fixed order, spends it where all the variance is. So the
    extrapolation is left in place and reported instead --
    :attr:`ModelError.spectrum_floor` against
    :attr:`RationalApproximation.interval_floor` says how far it reaches.
    """
    return 10.0 ** (-(5.0 + order) / 2.0)


@dataclasses.dataclass(frozen=True)
class RationalApproximation:
    """A rational approximation of ``x^-beta``, in the factored form the solver wants.

    Bolin & Kirchner (2020) equation (3.9) and appendix A, which factor the two
    polynomials through their roots so that the operator becomes a *product of
    shifted operators*. That is what makes the sparse solve a sequence of
    well-conditioned shifted problems rather than one badly-conditioned product,
    and it is the form their appendix A iteration is written in:

    .. math::

        x^{-\\beta} \\approx
        \\frac{c_m \\prod_{i=1}^{m} (1 - r_{1i} x)}
             {b_{m+1} \\prod_{j=1}^{m+1} (1 - r_{2j} x)}

    Attributes
    ----------
    numerator_roots : FloatArray
        The ``m`` roots ``r_1i``. All real and negative -- see
        :func:`rational_approximation`.
    denominator_roots : FloatArray
        The ``m + 1`` roots ``r_2j``, likewise.
    numerator_leading, denominator_leading : float
        ``c_m`` and ``b_{m+1}``.
    supremum_error : float
        The measured ``sup |x^(beta - m_beta) - q1(x)/q2(x)|`` over the fitting
        interval. This is the quantity Bolin & Kirchner's appendix B bounds by
        ``C * exp(-2 pi sqrt(|beta - m_beta| m))`` (Stahl 2003, theorem 1) and
        then feeds into theorem 3.3; measuring it rather than using the
        asymptotic bound makes the reported model error tighter and honest about
        which fit was actually computed.
    interval_floor : float
        The ``delta`` the fit was computed on. The approximation is only
        controlled on ``[delta, 1]``; :attr:`ModelError.spectrum_floor` reports
        whether the discrete operator stayed inside it.
    """

    numerator_roots: FloatArray
    denominator_roots: FloatArray
    numerator_leading: float
    denominator_leading: float
    supremum_error: float
    interval_floor: float

    @property
    def order(self) -> int:
        """Bolin & Kirchner's ``m``: the numerator degree."""
        return int(self.numerator_roots.size)


@functools.lru_cache(maxsize=32)
def rational_approximation(
    beta: float, order: int, floor: float
) -> RationalApproximation:
    """Fit ``x^(beta - m_beta) ~ q1/q2`` on ``[delta, 1]``, in the supremum norm.

    Bolin & Kirchner section 3.3 decompose ``f(x) = x^beta`` as
    ``f(x) = x^(beta - m_beta) * x^m_beta`` and approximate only the first factor
    by a rational function of degrees ``(m, m + 1)``; the second is carried
    exactly by the operator. Section 3.5 computes the fit once on a fixed
    interval rather than per mesh, which is what lets one approximation serve
    every fault.

    The fit is by **Lawson's algorithm**: a sequence of weighted linear
    least-squares problems on the linearised residual ``f * q2 - q1``, with the
    weights reweighted by the realised error each round, which converges to the
    equioscillating supremum-optimal fit that theorem 3.3 assumes. Bolin &
    Kirchner use Remez (which they call "often unstable in computations") or
    Clenshaw-Lord Chebyshev-Pade via Chebfun; Lawson needs neither and is a
    dozen lines of numpy. The polynomials are carried in the Chebyshev basis of
    the fitting interval, because a monomial Vandermonde over ``[1e-4, 1]`` at
    degree 4 is conditioned at ``1e12`` and loses the small roots entirely.

    Parameters
    ----------
    beta : float
        Half the SPDE exponent, ``alpha / 2``.
    order : int
        Bolin & Kirchner's ``m``.
    floor : float
        The ``delta`` of the fitting interval ``[delta, 1]``, from
        :func:`_interval_floor`.

    Returns
    -------
    RationalApproximation

    Raises
    ------
    ValueError
        If the fit's roots are not all real and negative. The solver turns each
        root into a shifted matrix ``C - r L``; a negative real root makes that
        ``C + |r| L``, which is symmetric positive definite and cheap to
        factorise, and a complex or positive root would make it indefinite or
        complex. Stahl (2003) puts the poles and zeros of the best rational
        approximation of ``x^s`` on the negative real axis, so this failing means
        the fit did not converge rather than that the model is unrepresentable --
        the caller can drop ``order``.
    """
    if order < 1:
        raise ValueError(f"the rational order must be at least 1, got {order}")
    if not (0.0 < floor < 1.0):
        raise ValueError(f"the fitting interval floor must be in (0, 1), got {floor}")

    exponent = beta - _rational_shift(beta)

    # Chebyshev points of the second kind on [floor, 1]: they cluster at the ends,
    # which is where the equioscillation extrema of a fit to a power function go.
    angles = np.linspace(0.0, np.pi, RATIONAL_SAMPLE_COUNT)
    unit = np.cos(angles)
    sample = 0.5 * (1.0 - floor) * (unit + 1.0) + floor
    target = sample**exponent

    numerator_basis = np.polynomial.chebyshev.chebvander(unit, order)
    denominator_basis = np.polynomial.chebyshev.chebvander(unit, order + 1)
    # The linearised residual f * q2 - q1, as one matrix acting on the stacked
    # coefficients. Solved by smallest singular vector, which imposes the unit
    # normalisation the homogeneous problem needs.
    design = np.hstack([-numerator_basis, target[:, None] * denominator_basis])

    weight = np.ones(RATIONAL_SAMPLE_COUNT)
    coefficients = np.zeros(design.shape[1])
    error = np.zeros(RATIONAL_SAMPLE_COUNT)
    for _ in range(LAWSON_ITERATIONS):
        _, _, rotation = np.linalg.svd(
            np.sqrt(weight)[:, None] * design, full_matrices=False
        )
        coefficients = rotation[-1]
        denominator = denominator_basis @ coefficients[order + 1 :]
        # The linearised residual divided by q2 is the error actually made; Lawson
        # reweights by it so the next round pushes hardest where the fit is worst.
        with np.errstate(divide="ignore", invalid="ignore"):
            error = np.where(
                denominator != 0.0,
                (design @ coefficients) / denominator,
                np.inf,
            )
        magnitude = np.abs(error)
        if not np.isfinite(magnitude).all() or magnitude.max() == 0.0:
            break
        weight = weight * magnitude
        weight /= weight.sum()

    numerator_chebyshev = coefficients[: order + 1]
    denominator_chebyshev = coefficients[order + 1 :]
    supremum = float(np.abs(error).max())

    # Chebyshev coefficients in the mapped variable, back to monomials in x, so
    # that the roots are roots in x and the leading coefficients really are c_m
    # and b_{m+1}. The target domain and window are given explicitly: converting
    # onto the *fitting* domain instead leaves the coefficients expressed in the
    # mapped variable, which scales the two leading coefficients by different
    # powers of the map and puts a spurious constant factor on q1/q2. The roots
    # survive that mistake, which is what makes it worth naming -- it shows up
    # only as a scale, and only in the field's variance.
    window = np.array([floor, 1.0])
    unit_window = np.array([-1.0, 1.0])
    numerator = np.polynomial.chebyshev.Chebyshev(
        numerator_chebyshev, domain=window
    ).convert(kind=np.polynomial.Polynomial, domain=unit_window, window=unit_window)
    denominator = np.polynomial.chebyshev.Chebyshev(
        denominator_chebyshev, domain=window
    ).convert(kind=np.polynomial.Polynomial, domain=unit_window, window=unit_window)

    numerator_leading = float(numerator.coef[-1])
    denominator_leading = float(denominator.coef[-1])
    if (
        abs(numerator_leading) < _LEADING_COEFFICIENT_FLOOR
        or abs(denominator_leading) < _LEADING_COEFFICIENT_FLOOR
    ):
        raise ValueError(
            f"the rational fit of order {order} for beta {beta:.4g} collapsed to a "
            f"lower degree -- its leading coefficients are {numerator_leading:.3g} "
            f"and {denominator_leading:.3g} against a unit-norm coefficient vector. "
            "Use a smaller rational order"
        )

    roots = []
    for name, polynomial, degree in (
        ("numerator", numerator, order),
        ("denominator", denominator, order + 1),
    ):
        found = polynomial.roots()
        if found.size != degree:
            raise ValueError(
                f"the rational fit's {name} has {found.size} roots and the "
                f"construction needs {degree}; use a smaller rational order"
            )
        if np.iscomplexobj(found) and np.abs(found.imag).max() > 0.0:
            raise ValueError(
                f"the rational fit of order {order} for beta {beta:.4g} has complex "
                f"{name} roots, so the shifted operators would not be symmetric "
                "positive definite. Use a smaller rational order"
            )
        real = np.asarray(found.real, dtype=np.float64)
        if real.max() >= 0.0:
            raise ValueError(
                f"the rational fit of order {order} for beta {beta:.4g} has a "
                f"non-negative {name} root ({real.max():.3g}), so the shifted "
                "operator would be indefinite. Use a smaller rational order"
            )
        roots.append(real)

    # Re-measure the error through the factored form the solver will actually use,
    # rather than trusting the fit's own residual. This is the same quantity --
    # Bolin & Kirchner's ||f - r_h||_C(J), equation (B.1), since
    # |x^b - rhat(x) x^m_b| at x = 1/lambda is |lambda^-b - p_r/p_l| -- but computed
    # from the roots and leading coefficients rather than from the Chebyshev
    # coefficients, so any error in extracting them shows up here as a number
    # instead of silently as a scale on the field.
    eigenvalue = 1.0 / sample
    solver_form = (
        numerator_leading * np.prod([1.0 - r * eigenvalue for r in roots[0]], axis=0)
    ) / (
        denominator_leading * np.prod([1.0 - r * eigenvalue for r in roots[1]], axis=0)
    )
    measured = float(np.abs(eigenvalue**-beta - solver_form).max())
    if not np.isfinite(measured) or measured > 10.0 * max(supremum, 1.0e-12):
        raise ValueError(
            f"the rational fit of order {order} for beta {beta:.4g} reached a "
            f"residual of {supremum:.3g} but its factored form is off by "
            f"{measured:.3g}, so the roots do not reproduce the fit. This is a bug "
            "in the sampler, not a property of the model"
        )

    return RationalApproximation(
        numerator_roots=roots[0],
        denominator_roots=roots[1],
        numerator_leading=numerator_leading,
        denominator_leading=denominator_leading,
        supremum_error=measured,
        interval_floor=floor,
    )


@dataclasses.dataclass(frozen=True)
class ModelError:
    """What the sampler knows about its own error, before drawing anything.

    This replaces `sampling.py`'s `_warn_if_degraded`, and is a different kind of
    statement: that one measured the correlation lengths the embedding *actually*
    delivered and reported the discrepancy after the fact, whereas Bolin &
    Kirchner theorem 3.3 gives an a-priori bound in the mesh width and the
    rational order, so this is computable from the mesh alone.

    Attributes
    ----------
    mesh_width : float
        Bolin & Kirchner's ``h``, in **correlation lengths**: the longest edge of
        the lifted triangulation measured in the ``H`` metric,
        ``sqrt(e^T H^-1 e)``. That is the only nondimensionalisation the operator
        admits -- after folding the correlation lengths into ``H`` the SPDE reads
        ``(1 - Delta_y)^(alpha/2)``, whose unit of length is one correlation
        length -- and it is exactly Mai's dimensionless ``r`` applied to an edge.
    rational_order : int
        Bolin & Kirchner's ``m``.
    finite_element_term, rational_term : float
        The two summands of theorem 3.3, before the constant:
        ``h^min(2b - d/2, 2)`` and ``h^(min(2(b-1), 0) - d/2) * e_m``, with
        ``e_m`` the *measured* :attr:`RationalApproximation.supremum_error`
        rather than its asymptotic bound.
    bound : float
        :data:`MODEL_ERROR_CONSTANT` times the two terms. The number to quote.
    rational_supremum_error : float
        ``e_m``, carried through so a caller can see which of the two terms
        dominates without re-deriving it.
    spectrum_floor : float
        A lower bound on ``1 / lambda_max`` of the discrete operator, from
        Gershgorin. The rational approximation is only controlled on
        ``[delta, 1]``, so this being below
        :attr:`RationalApproximation.interval_floor` means the fit is being
        extrapolated over the top of the spectrum. That is the normal case at
        fault resolution -- `colombia` reaches 5e-6 against a ``delta`` of 3.2e-4
        -- and is deliberately not corrected; :func:`_interval_floor` says why,
        and why correcting it is measurably worse.
    boundary_reach : float
        How far the Neumann folding reaches, in the parameter domain's own units:
        :data:`BOUNDARY_FOLDING_LENGTHS` times the larger correlation length.
        Compare it against the segment's extent to see how much of the fault
        carries the folded covariance rather than the stationary one.
    """

    mesh_width: float
    rational_order: int
    finite_element_term: float
    rational_term: float
    bound: float
    rational_supremum_error: float
    spectrum_floor: float
    boundary_reach: float

    def __str__(self) -> str:
        """One line, in the units the bound is quoted in."""
        return (
            f"mesh width {self.mesh_width:.3g} correlation lengths, rational order "
            f"{self.rational_order}: error bound {self.bound:.3g} "
            f"(finite element {self.finite_element_term:.3g}, rational "
            f"{self.rational_term:.3g})"
        )


def _model_error(
    mesh_width: float,
    beta: float,
    approximation: RationalApproximation,
    spectrum_floor: float,
    boundary_reach: float,
) -> ModelError:
    """Instantiate Bolin & Kirchner theorem 3.3 for one mesh.

    .. math::

        \\|u - u^R_{h,m}\\| \\le C \\left( h^{\\min(2\\beta - d/2,\\, 2)}
        + \\mathbb{1}_{\\beta \\notin \\mathbb{N}}\\,
          h^{\\min(2(\\beta-1),\\, 0) - d/2}\\, e_m \\right)

    At Mai's ``beta = 0.875`` and ``d = 2`` the two exponents are ``0.75`` and
    ``-1.25``: the finite element term converges and the rational term *diverges*
    in ``h``, which is theorem 3.3's honest statement that refining the mesh
    without raising ``m`` eventually stops helping. Remark 3.4 is the
    counterpart: choosing ``m`` so the rational error tracks ``h^(2 max(beta, 1))``
    makes the overall rate ``min(2 beta - d/2, 2) = 0.75``.
    """
    shift = _rational_shift(beta)
    half_dimension = MANIFOLD_DIMENSION / 2.0
    finite_element = mesh_width ** min(2.0 * beta - half_dimension, 2.0)
    rational = 0.0
    if beta != float(int(beta)):
        rational = (
            mesh_width ** (min(2.0 * (beta - 1.0), 0.0) - half_dimension)
            * approximation.supremum_error
        )
    del shift
    return ModelError(
        mesh_width=mesh_width,
        rational_order=approximation.order,
        finite_element_term=finite_element,
        rational_term=rational,
        bound=MODEL_ERROR_CONSTANT * (finite_element + rational),
        rational_supremum_error=approximation.supremum_error,
        spectrum_floor=spectrum_floor,
        boundary_reach=boundary_reach,
    )


def _surface_frames(
    vertices_km: FloatArray, faces: IntArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Per-face geometry: areas, gradients, the strike and dip directions, and dip.

    Everything the assembly needs, computed **from the lifted triangles alone**.
    No parameter coordinate enters, which is what makes the operator mesh-native:
    the same surface sampled through a different parameterisation gets the same
    field.

    The gradients are the standard piecewise-linear ones written directly in three
    dimensions, ``grad phi_i = n x e_i / (2|T|)`` with ``e_i`` the edge opposite
    corner ``i`` and ``n`` the unit normal -- no tangent basis has to be
    constructed, and the result lies in the triangle's plane by construction.

    The frame is the **surface's own strike and dip**, matching
    `mesh.RuptureMesh.strike_dip_deg`: ``cross(down, n)`` is perpendicular to both
    the vertical and the normal, hence horizontal and in the tangent plane, which
    is the strike direction; the dip direction is ``n x strike``, the steepest
    descent, perpendicular by construction. These are what "along strike" and
    "down dip" physically mean, and they are the directions Mai & Beroza measured
    their two correlation lengths along.

    Unlike `strike_dip_deg` this does **not** disambiguate the strike's sign.
    That function reports a bearing, where the sign is the whole difference
    between a fault and its reverse; here the frame enters only through the outer
    products ``e e^T`` of :func:`_anisotropy`, which are even in the sign. Nothing
    downstream can see it.

    Returns
    -------
    tuple of FloatArray
        Areas ``(F,)``, gradients ``(F, 3, 3)`` indexed corner then component,
        the unit strike and dip directions each ``(F, 3)``, and ``sin(dip)``
        ``(F,)`` -- which is the length of the unnormalised strike vector, and so
        measures how well determined the frame is.

    Raises
    ------
    ValueError
        For a triangle with no area: it has no tangent plane, hence no
        Laplace-Beltrami operator and no frame.
    """
    corners = vertices_km[faces]
    first = corners[:, 1] - corners[:, 0]
    second = corners[:, 2] - corners[:, 0]
    normal = np.cross(first, second)
    twice_area = np.linalg.norm(normal, axis=-1)
    if not (twice_area > 0.0).all():
        bad = int(np.flatnonzero(twice_area <= 0.0)[0])
        raise ValueError(
            f"face {bad} of the lifted mesh has no area, so it has no tangent plane "
            "and no Laplace-Beltrami operator. Drop the degenerate faces or remesh"
        )
    unit_normal = normal / twice_area[:, None]
    area = 0.5 * twice_area

    # Edge opposite each corner, in the corner's own cyclic order.
    opposite = np.stack(
        [
            corners[:, 2] - corners[:, 1],
            corners[:, 0] - corners[:, 2],
            corners[:, 1] - corners[:, 0],
        ],
        axis=1,
    )
    gradient = np.cross(unit_normal[:, None, :], opposite) / twice_area[:, None, None]

    horizontal = np.cross(_DOWN, unit_normal)
    sine_dip = np.linalg.norm(horizontal, axis=-1)
    # Where the patch is horizontal the strike direction genuinely does not exist
    # -- the limit depends on the direction of approach -- so the frame is filled
    # with an arbitrary tangent vector and :func:`_anisotropy` makes the choice
    # irrelevant by turning the anisotropy off there.
    flat = sine_dip == 0.0
    substitute = np.cross(unit_normal, _EAST)
    substitute_length = np.linalg.norm(substitute, axis=-1)
    horizontal = np.where(flat[:, None], substitute, horizontal)
    length = np.where(flat, substitute_length, sine_dip)
    strike_direction = horizontal / length[:, None]
    dip_direction = np.cross(unit_normal, strike_direction)

    return area, gradient, strike_direction, dip_direction, sine_dip


def _anisotropy(
    sine_dip: FloatArray, covariance: VonKarmanFilterParameters
) -> tuple[FloatArray, FloatArray]:
    """The two eigenvalues of ``H``, faded to isotropic where the frame dies.

    ``H``'s eigenvalues are the *squared* correlation lengths along strike and
    down dip. Where the patch is too flat for its own strike direction to be
    determined (:data:`DEGENERATE_DIP_SINE`) there is no along-strike and no
    down-dip, so an anisotropy there is a claim about a distinction the geometry
    does not make. It is faded out instead:

    .. math::

        \\lambda_s = a_s^{1+w} a_d^{1-w}, \\qquad
        \\lambda_d = a_d^{1+w} a_s^{1-w}, \\qquad
        w = \\min(1, \\sin(\\mathrm{dip}) / s_0)

    At ``w = 1`` these are ``a_s^2`` and ``a_d^2``, the model unchanged. At
    ``w = 0`` both are ``a_s a_d``: isotropic, with the **geometric mean** as the
    correlation length.

    **The geometric mean is forced, not chosen.** Interpolating the eigenvalues
    geometrically keeps ``lambda_s lambda_d = a_s^2 a_d^2`` for every ``w``, and
    ``det H`` is exactly what sets the marginal variance --
    :func:`_marginal_variance` is ``1 / (4 pi (alpha - 1) sqrt(det H))``. So the
    field's variance is untouched right through the transition, and the fade
    moves structure between the two axes without creating or destroying any. An
    arithmetic interpolation, or a fallback to either length alone, would put a
    variance step at the degeneracy -- which on Hikurangi would trace the shallow
    contour as a visible seam in the slip.

    The fade is linear in ``sin(dip)`` and therefore only continuous, not smooth.
    That is enough: ``H`` is piecewise constant per face regardless, since the
    finite element assembly evaluates it once per triangle.

    Parameters
    ----------
    sine_dip : FloatArray
        ``(F,)`` from :func:`_surface_frames`.
    covariance : VonKarmanFilterParameters
        The two correlation lengths.

    Returns
    -------
    tuple of FloatArray
        ``(F,)`` eigenvalues along strike and down dip, each a squared length.
    """
    weight = np.minimum(sine_dip / DEGENERATE_DIP_SINE, 1.0)
    strike = covariance.correlation_length_strike_km
    dip = covariance.correlation_length_dip_km
    return (
        strike ** (1.0 + weight) * dip ** (1.0 - weight),
        dip ** (1.0 + weight) * strike ** (1.0 - weight),
    )


def _assemble(
    vertices_km: FloatArray,
    faces: IntArray,
    covariance: VonKarmanFilterParameters,
) -> tuple[FloatArray, sparse.csc_matrix, float]:
    """The lumped mass vector, the anisotropic stiffness matrix, and the mesh width.

    Lindgren et al. (2011) appendix A.2, assembled per triangle and scatter-added.
    Two departures from the isotropic formulas there, both deliberate:

    **The metric.** The stiffness entry is
    ``|T| (H grad phi_i) . (grad phi_j)`` with
    ``H = lambda_s e_s e_s^T + lambda_d e_d e_d^T``, the frame from
    :func:`_surface_frames` and the eigenvalues from :func:`_anisotropy` -- so
    ``H`` varies per face, both because the surface turns and because the
    anisotropy fades where the surface flattens. Lindgren's anisotropic form (equation
    20) writes the same quantity as ``e_i^T adj(H) e_j / (4|T|)``; the two are
    identical, because ``grad phi_i`` is ``n x e_i / (2|T|)`` and rotating both
    arguments of a 2x2 quadratic form by a right angle turns ``H`` into its
    adjugate. The gradient form is used here because it is the form
    ``-div(H grad)`` is written in (Bolin & Kirchner equation 3.1), so the
    eigenvalues of ``H`` are the squared correlation lengths without an adjugate
    to invert first. With ``a_u = a_v`` it collapses to ``a^2`` times
    ``e_i . e_j / (4|T|)``, the cotangent Laplacian.

    **The lumping.** The mass matrix is returned already lumped to its row sums,
    ``|T|/3`` per corner. Lindgren et al. appendix A.3 is explicit that "all the C
    should be replaced by C-tilde to obtain a Markov model", and Bolin & Kirchner
    appendix A lump for the same reason: ``C^-1`` appears in the precision, and a
    dense inverse there would destroy the sparsity the whole method exists for.

    Returns
    -------
    tuple
        The lumped mass diagonal ``(V,)``, the stiffness matrix ``(V, V)`` in CSC,
        and the mesh width in correlation lengths.
    """
    area, gradient, strike_direction, dip_direction, sine_dip = _surface_frames(
        vertices_km, faces
    )
    strike_squared, dip_squared = _anisotropy(sine_dip, covariance)

    # Components of each corner's gradient along the two principal directions of H,
    # which is all H does to them: (F, 3).
    strike_component = np.einsum("fcx,fx->fc", gradient, strike_direction)
    dip_component = np.einsum("fcx,fx->fc", gradient, dip_direction)
    element = area[:, None, None] * (
        strike_squared[:, None, None]
        * strike_component[:, :, None]
        * strike_component[:, None, :]
        + dip_squared[:, None, None]
        * dip_component[:, :, None]
        * dip_component[:, None, :]
    )

    vertex_count = vertices_km.shape[0]
    rows = np.broadcast_to(faces[:, :, None], element.shape)
    columns = np.broadcast_to(faces[:, None, :], element.shape)
    stiffness = sparse.coo_matrix(
        (element.ravel(), (rows.ravel(), columns.ravel())),
        shape=(vertex_count, vertex_count),
    ).tocsc()

    lumped_mass = np.bincount(
        faces.ravel(),
        weights=np.repeat(area / 3.0, 3),
        minlength=vertex_count,
    )

    # The mesh width in the H metric: sqrt(e^T H^-1 e) over every edge, which is
    # Mai's dimensionless r applied to an edge. H^-1 is diagonal in (e1, e2), so
    # this is the same projection the stiffness uses, with reciprocal lengths.
    corners = vertices_km[faces]
    edges = np.stack(
        [
            corners[:, 2] - corners[:, 1],
            corners[:, 0] - corners[:, 2],
            corners[:, 1] - corners[:, 0],
        ],
        axis=1,
    )
    scaled = (
        np.einsum("fcx,fx->fc", edges, strike_direction) ** 2 / strike_squared[:, None]
        + np.einsum("fcx,fx->fc", edges, dip_direction) ** 2 / dip_squared[:, None]
    )
    mesh_width = float(np.sqrt(scaled.max()))

    return lumped_mass, stiffness, mesh_width


def _marginal_variance(alpha: float, covariance: VonKarmanFilterParameters) -> float:
    """The stationary marginal variance of the SPDE solution, on the whole plane.

    .. math:: \\sigma^2 = \\frac{\\Gamma(\\nu)}
        {\\Gamma(\\alpha)\\,(4\\pi)^{d/2}\\,\\sqrt{\\det H}}

    Lindgren et al. (2011) equation (1) with ``kappa = 1``; the ``det H`` is the
    Jacobian of the coordinate scaling that turns the anisotropic operator into
    the isotropic one, so it is ``a_u a_v`` here. At ``d = 2`` the gamma ratio is
    ``1/(alpha - 1)`` and the whole thing is ``1 / (4 pi (alpha - 1) a_u a_v)``.

    Dividing by its square root is what makes the draw a *standard*-normal
    marginal, so that the field is directly comparable with the circulant
    sampler's, which delivers ``C(0) = 1`` by construction. It is the continuum
    value: the discrete field differs from it by the finite element error, and
    near the boundary by the Neumann folding of Lindgren et al. appendix A.4,
    which the module docstring covers. Downstream, `sampling.standardise` fixes
    the sample statistics anyway; this only has to be right enough that
    `sampling.correlate_fields` sees two fields of the same scale.
    """
    return 1.0 / (
        4.0
        * np.pi
        * (alpha - 1.0)
        * covariance.correlation_length_strike_km
        * covariance.correlation_length_dip_km
    )


def _warn_if_oversized(
    parameters_uv: FloatArray, covariance: VonKarmanFilterParameters
) -> None:
    """Mai & Beroza figure 13's bound on correlation length against source dimension.

    Carried over from `sampling._warn_if_degraded` unchanged in meaning, because
    it is a statement about the *model's* validity and not about the numerics:
    the SPDE will happily deliver a correlation length longer than the fault, and
    it will be just as far outside the data Mai & Beroza fitted as the circulant
    embedding's was. What has gone is the second half of that function -- the
    after-the-fact measurement of delivered correlation lengths -- which
    :class:`ModelError` replaces with an a-priori bound.

    The extents are read off the parameter coordinates, which are strike and dip
    by construction, so this is the same ratio the structured path takes from
    ``cell_counts`` times ``spacing_km``.
    """
    extents = np.ptp(parameters_uv, axis=0)
    if not (extents > 0.0).all():
        return
    ratios = (
        covariance.correlation_length_strike_km / float(extents[0]),
        covariance.correlation_length_dip_km / float(extents[1]),
    )
    if max(ratios) > MAI_MAXIMUM_RATIO:
        warnings.warn(
            f"a {extents[0]:.3g} x {extents[1]:.3g} km segment cannot carry "
            f"correlation lengths of "
            f"{covariance.correlation_length_strike_km:.3g} km along strike and "
            f"{covariance.correlation_length_dip_km:.3g} km down dip -- they are "
            f"{ratios[0]:.2g} and {ratios[1]:.2g} of the segment, where Mai & Beroza "
            f"(2002) figure 13 puts every model they fitted between 0.25 and "
            f"{MAI_MAXIMUM_RATIO}. The field it gets varies little across the "
            "segment. Slip, moment and timing are unaffected; what is degraded is "
            "how the slip is distributed",
            DegradedCorrelation,
            stacklevel=3,
        )


def _warn_if_folded(
    parameters_uv: FloatArray, covariance: VonKarmanFilterParameters
) -> None:
    """Refuse to be quiet about a segment small enough that the folding is the field.

    The Neumann boundary condition reflects the covariance (Lindgren et al. 2011
    appendix A.4, theorem 1), and on a domain only a few correlation lengths
    across the reflections do not stay near the edge -- they are the field.
    Measured on a square domain of ``L`` correlation lengths, at
    ``h = 0.088``, reading the marginal variance and the correlation at one
    correlation length off the centre (`test_boundary_folding_is_lindgren_a4`):

    ====  ========  ===================
    L     variance  correlation at r=1
    ====  ========  ===================
    2     2.586     0.878
    3     1.473     0.703
    4     1.168     0.584
    6     1.039     0.507
    8     1.024     0.496
    16    1.022     0.495
    ====  ========  ===================

    against a target of 0.5005, which the circulant sampler delivers to 5e-4 at
    *every* one of those sizes because it pads the embedding and crops. So this
    is not a discretisation difference between the two samplers, it is a
    different field, and it appears exactly where real faults live: the shipped
    `colombia` example is 17.4 km wide against a 9.26 km down-dip correlation
    length, which is the ``L = 2`` row.

    **The remedy is the circulant sampler's own.** `sampling.MINIMUM_EMBEDDING`
    and `sampling.DECAY_LENGTHS` exist to put the wrap where the covariance has
    faded; the same trick works here -- triangulate a parameter domain extended
    by :data:`BOUNDARY_FOLDING_LENGTHS` correlation lengths on each side, solve
    on that, and keep the vertices inside the fault. That is a change to what
    mesh is built rather than to how it is sampled, so it belongs to the mesh
    builder and not here, and until it exists this warns.
    """
    extents = np.ptp(parameters_uv, axis=0)
    if not (extents > 0.0).all():
        return
    lengths = (
        covariance.correlation_length_strike_km,
        covariance.correlation_length_dip_km,
    )
    spans = tuple(
        float(extent) / length for extent, length in zip(extents, lengths, strict=True)
    )
    if min(spans) < 2.0 * BOUNDARY_FOLDING_LENGTHS:
        axis = "along strike" if spans[0] < spans[1] else "down dip"
        warnings.warn(
            f"this segment is {min(spans):.2g} correlation lengths across {axis}, and "
            "the SPDE is solved with the natural (Neumann) boundary condition, whose "
            "covariance is the Matern one reflected at the boundary (Lindgren et al. "
            f"2011 appendix A.4). Below {2.0 * BOUNDARY_FOLDING_LENGTHS:.0f} "
            "correlation lengths the reflections are the field rather than an edge "
            "effect: at 2 lengths the marginal variance is 2.6 times the model's and "
            "the correlation at one correlation length is 0.88 against 0.50. The "
            "circulant sampler does not do this, because it pads the embedding and "
            "crops -- the equivalent here is to triangulate a parameter domain "
            f"extended by {BOUNDARY_FOLDING_LENGTHS:.0f} correlation lengths on each "
            "side and keep the vertices inside the fault. Slip, moment and timing are "
            "unaffected; what is degraded is how the slip is distributed",
            DegradedCorrelation,
            stacklevel=3,
        )


def _refuse_starved_vertices(lumped_mass: FloatArray) -> None:
    """Refuse a mesh with a vertex whose sliver leaves it almost no area.

    The lumped mass ``C~_ii`` is a vertex's share of the surface, and ``C~^-1``
    sits in the middle of the precision, so a vertex left with almost none is
    almost unconstrained by the operator and its marginal variance explodes.

    **This is not a cosmetic defect.** The stages standardise a field by its
    sample spread (`sampling.standardise`), so one runaway vertex divides the
    whole segment. Measured on the shipped CFM interfaces at Mw 8.5, taking the
    ratio of each mesh's smallest lumped mass to its median:

    ======================  ==========  ==============  ===============
    surface                 worst ratio max variance    healthy field
    ======================  ==========  ==============  ===============
    Puysegur-Fiordland      2.35e-5     8.9             std 1.000
    Hikurangi               2.36e-5     18.1            std 0.996
    Puyseguer               7.35e-7     9.5e6           **std 0.187**
    ======================  ==========  ==============  ===============

    -- so two vertices of Puyseguer's 2597, from six faces of quality below
    1e-3, suppress the entire slip distribution by a factor of 5.3 while leaving
    a field that still looks like a field. That is the failure mode `DEFECTS.md`
    17 is the precedent for, and it is worth a refusal rather than a warning.

    :data:`MINIMUM_LUMPED_MASS_RATIO` is where the threshold sits and how it was
    placed. The real fix is upstream -- a mesh this shape should not be built,
    and the mesh builder's admissibility check is where that belongs -- but the
    damage appears here, so here is where it is caught.

    Parameters
    ----------
    lumped_mass : FloatArray
        ``(V,)`` from :func:`_assemble`.

    Raises
    ------
    ValueError
        Naming the worst vertex, its area, the median, and what to do.
    """
    median = float(np.median(lumped_mass))
    if median <= 0.0:
        raise ValueError("every vertex of this mesh has zero area")
    ratio = lumped_mass / median
    starved = np.flatnonzero(ratio < MINIMUM_LUMPED_MASS_RATIO)
    if starved.size == 0:
        return
    worst = int(starved[np.argmin(ratio[starved])])
    raise ValueError(
        f"vertex {worst} is left {lumped_mass[worst]:.3g} km^2 of surface against a "
        f"median of {median:.3g} km^2, a ratio of {ratio[worst]:.2g} where "
        f"{MINIMUM_LUMPED_MASS_RATIO:.0e} is the floor ({starved.size} vertices are "
        "below it). A vertex starved of area is barely constrained by the operator, "
        "so its variance runs away and standardising the field by its sample spread "
        "shrinks the whole segment -- measured at a factor of 5.3 on the CFM "
        "Puyseguer interface. Collapse the sliver triangles around that vertex, or "
        "remesh; the fault's geometry is fine, its triangulation is not"
    )


class MaternOperator:
    """One assembled, factorised SPDE operator: a mesh, a covariance, and its solvers.

    The counterpart of `sampling.Embedding`, and held for the same reason: the
    expensive part does not depend on the noise, so it is done once and reused
    across draws. A segment needs four correlated fields, which is four to eight
    draws against one operator.

    The construction, following Bolin & Kirchner (2020) appendix A with the mass
    matrix lumped throughout. Write ``M`` for the lumped mass diagonal, ``G`` for
    the anisotropic stiffness and ``K = M + G`` for the discrete operator's
    weak form. With ``r_2j`` the denominator roots of the rational approximation,

    .. math::

        P_\\ell = b_{m+1}\\, M \\prod_{j=1}^{m+1}(I - r_{2j} M^{-1} K),
        \\qquad
        P_r = c_m \\prod_{i=1}^{m}(I - r_{1i} M^{-1} K)

    and the field's weights are ``u = P_r x`` with ``x ~ N(0, Q^-1)``,
    ``Q = P_l^T M^-1 P_l``.

    **The precision is never factorised.** ``Q = P_l^T M^-1 P_l`` is already a
    factorisation: ``Q^-1 = P_l^-1 M P_l^-T``, so ``x = P_l^-1 M^(1/2) z`` has
    exactly the right covariance for ``z ~ N(0, I)``, and ``M`` is diagonal
    because it is lumped. That removes the sparse Cholesky of ``Q`` the plan
    called for -- which is fortunate, since CHOLMOD is not available -- and
    replaces it with ``m + 1`` solves against the *individual* shifted factors.
    Solving the factors separately rather than forming ``P_l`` is also what keeps
    the conditioning tolerable at ``m > 1``, which is Bolin & Kirchner's
    remark 4.2.

    Each factor is solved in the symmetric form
    ``(M - r_2j K) y = M w``, and since every ``r_2j`` is real and negative
    (:func:`rational_approximation` refuses anything else) that matrix is
    ``M + |r_2j| K``: symmetric positive definite, because ``M`` is positive
    diagonal and ``K = M + G`` is positive definite.

    Parameters
    ----------
    vertices_km : FloatArray
        ``(V, 3)`` lifted vertex positions in the projected CRS, kilometres.
    faces : IntArray
        ``(F, 3)`` vertex indices.
    parameters_uv : FloatArray, optional
        ``(V, 2)`` parameter coordinates, strike then dip. **Nothing in the
        operator reads these.** The frame ``H`` is diagonal in comes from the
        surface's own normal (:func:`_surface_frames`), so the field is a
        function of the geometry alone -- reparameterise the same surface and the
        draw is unchanged, which `test_the_operator_is_mesh_native` asserts by
        drawing with and without them. They survive only so that
        :func:`_warn_if_oversized` and :func:`_warn_if_folded` can say how large
        the segment is along strike and down dip; omit them and those two checks
        are skipped.
    covariance : VonKarmanFilterParameters
        The two correlation lengths and the Hurst exponent.
    order : int, optional
        The rational order ``m``. Defaults to :data:`RATIONAL_ORDER`.

    Raises
    ------
    ValueError
        For arrays that disagree in shape or carry non-finite values, for a face
        index off the end of the vertex array, for a face with no area, or for a
        vertex starved of area by a sliver (:func:`_refuse_starved_vertices`). A
        non-finite vertex would travel silently into every matrix entry and come
        back out as a field of NaN.

    Warns
    -----
    DegradedCorrelation
        For a correlation length past Mai & Beroza figure 13's range, or a
        segment small enough that the Neumann folding is the field.
    """

    def __init__(
        self,
        vertices_km: FloatArray,
        faces: IntArray,
        parameters_uv: FloatArray | None = None,
        covariance: VonKarmanFilterParameters | None = None,
        order: int = RATIONAL_ORDER,
    ) -> None:
        """Assemble and factorise. See the class docstring."""
        if covariance is None:
            raise ValueError("a covariance is required")
        vertices_km = np.asarray(vertices_km, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)

        if vertices_km.ndim != 2 or vertices_km.shape[1] != 3:
            raise ValueError(
                f"vertices_km is shaped {vertices_km.shape}, and the sampler wants "
                "(V, 3) lifted positions"
            )
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(
                f"faces is shaped {faces.shape}, and the sampler wants (F, 3) triangles"
            )
        if faces.size and (faces.min() < 0 or faces.max() >= vertices_km.shape[0]):
            raise ValueError(
                f"faces indexes vertices {faces.min()}..{faces.max()}, and there are "
                f"{vertices_km.shape[0]} vertices"
            )
        if not np.isfinite(vertices_km).all():
            raise ValueError("vertices_km carries a non-finite value")

        if parameters_uv is not None:
            parameters_uv = np.asarray(parameters_uv, dtype=np.float64)
            if parameters_uv.shape != (vertices_km.shape[0], 2):
                raise ValueError(
                    f"parameters_uv is shaped {parameters_uv.shape} against "
                    f"{vertices_km.shape[0]} vertices, so it is not one (u, v) pair "
                    "per vertex"
                )
            if not np.isfinite(parameters_uv).all():
                raise ValueError("parameters_uv carries a non-finite value")
            _warn_if_oversized(parameters_uv, covariance)
            _warn_if_folded(parameters_uv, covariance)

        alpha = matern_exponent(covariance.hurst)
        beta = alpha / 2.0

        lumped_mass, stiffness, mesh_width = _assemble(vertices_km, faces, covariance)
        _refuse_starved_vertices(lumped_mass)
        vertex_count = lumped_mass.size
        mass = sparse.diags(lumped_mass, format="csc")
        weak_operator = (mass + stiffness).tocsc()

        # Gershgorin on M^-1 K bounds the largest eigenvalue, hence the smallest
        # point of the spectrum of the inverse -- the interval the rational fit has
        # to cover. Computed before the fit, because the fit's interval follows it.
        scaled_rows = np.abs(weak_operator) @ np.ones(vertex_count) / lumped_mass
        spectrum_floor = float(1.0 / max(scaled_rows.max(), 1.0))
        approximation = rational_approximation(beta, order, _interval_floor(order))

        self._lumped_mass = lumped_mass
        self._root_mass = np.sqrt(lumped_mass)
        self._approximation = approximation
        self._solvers = [
            sparse_linalg.splu(
                (mass - root * weak_operator).tocsc(),
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0,
                options={"SymmetricMode": True},
            )
            for root in approximation.denominator_roots
        ]
        # P_r is only ever applied, never solved, so it is kept as the shifted
        # matrices themselves.
        self._numerator_factors = [
            sparse.identity(vertex_count, format="csc")
            - root * sparse.diags(1.0 / lumped_mass, format="csc") @ weak_operator
            for root in approximation.numerator_roots
        ]
        self._scale = 1.0 / np.sqrt(_marginal_variance(alpha, covariance))
        self._error = _model_error(
            mesh_width=mesh_width,
            beta=beta,
            approximation=approximation,
            spectrum_floor=spectrum_floor,
            boundary_reach=BOUNDARY_FOLDING_LENGTHS
            * max(
                covariance.correlation_length_strike_km,
                covariance.correlation_length_dip_km,
            ),
        )

    @property
    def error(self) -> ModelError:
        """Bolin & Kirchner theorem 3.3, instantiated for this mesh."""
        return self._error

    @property
    def vertex_count(self) -> int:
        """How many vertices the field will have."""
        return int(self._lumped_mass.size)

    def _forward(self, noise: FloatArray) -> FloatArray:
        """The map ``A`` from unit white noise to the field, ``A = s P_r P_l^-1 M^(1/2)``.

        ``m + 1`` sparse solves and ``m`` sparse matrix-vector products, against
        factorisations built once in the constructor. The field's covariance is
        ``A A^T``, which is what :meth:`covariance_column` exploits.
        """
        # P_l x = M^(1/2) z, peeled one shifted factor at a time. The leading
        # b_{m+1} and the outer M divide out here.
        state = noise / (self._root_mass * self._approximation.denominator_leading)
        for solver in self._solvers:
            state = solver.solve(self._lumped_mass * state)
        for factor in self._numerator_factors:
            state = factor @ state
        return self._scale * self._approximation.numerator_leading * state

    def _adjoint(self, weights: FloatArray) -> FloatArray:
        """``A^T``, the transpose of :meth:`_forward`.

        Every factor is transposed and the order reversed. The shifted matrices
        ``M - r K`` are symmetric, so the *same* factorisations serve: the
        transpose of ``F_j = I - r_j M^-1 K`` is ``(M - r_j K) M^-1``, whose
        inverse is ``M (M - r_j K)^-1``.
        """
        state = self._scale * self._approximation.numerator_leading * weights
        for factor in reversed(self._numerator_factors):
            state = factor.T @ state
        for solver in reversed(self._solvers):
            state = self._lumped_mass * solver.solve(state)
        return state / (self._root_mass * self._approximation.denominator_leading)

    def draw(self, rng: np.random.Generator) -> FloatArray:
        """One field, ``(V,)``, with standard-normal marginals.

        Parameters
        ----------
        rng : np.random.Generator
            The noise source. One standard normal per vertex is consumed.

        Returns
        -------
        FloatArray
            ``(V,)`` vertex values.
        """
        return self._forward(rng.standard_normal(self.vertex_count))

    def covariance_column(self, vertex: int) -> FloatArray:
        """The **exact** covariance between one vertex and every other, ``(V,)``.

        The discrete field's covariance is ``A A^T`` for the ``A`` of
        :meth:`_forward`, so one column of it is ``A (A^T e_v)`` -- twice the work
        of a draw, and no Monte Carlo error at all.

        This is the counterpart of `sampling._delivered_lengths`, which reads the
        covariance the circulant embedding *actually* delivers off the embedding
        rather than off a sample, and it is here for the same reason: verifying a
        sampler against an analytic covariance through draws costs a Monte Carlo
        error that is larger than the discretisation error being measured. Every
        covariance number this module's tests quote comes from here.

        Parameters
        ----------
        vertex : int
            Which vertex to take the covariance against.

        Returns
        -------
        FloatArray
            ``(V,)``. Entry ``vertex`` is that vertex's marginal variance, which
            is 1 only in the limit -- see the module docstring on the Neumann
            folding and :func:`_marginal_variance` on the continuum value.

        Raises
        ------
        ValueError
            For a vertex index off the end of the mesh.
        """
        if not (0 <= vertex < self.vertex_count):
            raise ValueError(
                f"vertex {vertex} is off a mesh of {self.vertex_count} vertices"
            )
        indicator = np.zeros(self.vertex_count)
        indicator[vertex] = 1.0
        return self._forward(self._adjoint(indicator))


def matern_field(
    vertices_km: FloatArray,
    faces: IntArray,
    parameters_uv: FloatArray | None,
    covariance: VonKarmanFilterParameters,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw one von Karman field on a triangulated surface, per vertex.

    The counterpart of `sampling.von_karman_field`, taking plain arrays rather
    than a mesh so that it can be validated before the mesh container exists.

    **Assemble once for repeated draws.** This builds a whole
    :class:`MaternOperator` -- assembly plus ``m + 1`` sparse factorisations --
    and throws it away. A segment wants four correlated fields off one geometry;
    hold a :class:`MaternOperator` and call :meth:`MaternOperator.draw` for those,
    and the per-draw cost falls to the solves alone.

    Parameters
    ----------
    vertices_km : FloatArray
        ``(V, 3)`` lifted positions in the projected CRS, kilometres.
    faces : IntArray
        ``(F, 3)`` vertex indices.
    parameters_uv : FloatArray, optional
        ``(V, 2)`` parameter coordinates, strike then dip. The operator does not
        read them; they only let the two segment-size checks report. See
        :class:`MaternOperator`.
    covariance : VonKarmanFilterParameters
        The correlation lengths and roughness.
    rng : np.random.Generator
        The noise source.

    Returns
    -------
    FloatArray
        ``(V,)``, standard-normal marginals away from the boundary.
    """
    return MaternOperator(vertices_km, faces, parameters_uv, covariance).draw(rng)


def face_values(vertex_values: FloatArray, faces: IntArray) -> FloatArray:
    """Carry a vertex field to the faces the pipeline attaches fields to.

    **The mean of the three corners**, which is not a choice of reduction so much
    as an evaluation: the field is a piecewise-linear finite element function
    ``u(x) = sum_j u_j phi_j(x)``, and the barycentric basis functions are each
    exactly ``1/3`` at the centroid, so ``(u_0 + u_1 + u_2)/3`` *is* ``u`` at the
    face centroid. Nothing is approximated, and it is the same point
    `RuptureMesh.centres` reports for a cell, so a field and the position it is
    attached to agree.

    The alternatives are worse for reasons worth recording. Taking one corner's
    value picks an arbitrary vertex and breaks the mesh's symmetry. An
    area-weighted or nodal-quadrature average is the same number here, since a
    triangle's three corners carry equal weight. A minimum or maximum would not
    be linear, and the pipeline's stages are written as ``1 + cov * Z``.

    What it does cost, and why it converges away. For a linear function the
    centroid value equals the mean over the triangle, so the face field is also
    the *element average* of the finite element field -- and a piecewise-linear
    field is not stationary within an element: its variance is largest at the
    nodes and smallest at the centroid. That is a discretisation artefact of the
    P1 basis, not a property of the model, and it vanishes with the mesh.
    Measured exactly (`test_face_values_smoothing_converges`), as the face
    field's variance over the vertex field's and the face field's correlation at
    one correlation length:

    =======  =================  ==================
    ``h``    variance ratio     correlation at r=1
    =======  =================  ==================
    0.707    0.783              0.590
    0.354    0.897              0.538
    0.177    0.957              0.516
    0.088    0.983              0.509
    0.044    0.995              --
    0.022    0.999              --
    =======  =================  ==================

    against a target of 0.5005. So at the ``h ~ 0.35`` of a coarse study mesh the
    face field's correlation is 3.7e-2 high -- well past the 1% slip bound -- and
    at the ``h = 0.0125`` the shipped `colombia` example is actually cut at it is
    below 1e-3. The reduction is safe at the resolutions faults are meshed at,
    and it is *not* safe on a mesh cut at the correlation length; the last two
    rows are blank because at those refinements the domain had to shrink to stay
    affordable and the boundary folding, not the reduction, dominates.

    Parameters
    ----------
    vertex_values : FloatArray
        ``(V,)`` a vertex field, such as :func:`matern_field` returns.
    faces : IntArray
        ``(F, 3)`` vertex indices.

    Returns
    -------
    FloatArray
        ``(F,)`` the field at each face centroid.

    Raises
    ------
    ValueError
        If the field is not one value per vertex, or a face indexes past its end.
    """
    vertex_values = np.asarray(vertex_values, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertex_values.ndim != 1:
        raise ValueError(
            f"vertex_values is shaped {vertex_values.shape}, and this wants one value "
            "per vertex"
        )
    if faces.size and faces.max() >= vertex_values.size:
        raise ValueError(
            f"faces indexes vertex {faces.max()} and the field has "
            f"{vertex_values.size} values"
        )
    return vertex_values[faces].mean(axis=1)


__all__ = [
    "BOUNDARY_FOLDING_LENGTHS",
    "MANIFOLD_DIMENSION",
    "MODEL_ERROR_CONSTANT",
    "RATIONAL_ORDER",
    "MaternOperator",
    "ModelError",
    "RationalApproximation",
    "face_values",
    "matern_exponent",
    "matern_field",
    "rational_approximation",
]
