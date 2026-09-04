"""Random fields with von Karman correlations, and the distributions they carry.

Sampled by circulant embedding (Dietrich & Newsam 1993; Wood & Chan 1994), and given
their distribution by NORTA -- NORmal-To-Anything, Cario & Nelson (1997). A field is
drawn Gaussian and pushed through :math:`F^{-1}(\\Phi(\\cdot))`, which gives every cell
the requested marginal exactly; because that map is nonlinear it also destroys some
correlation, so the sampler is asked for a covariance the field will *not* have, one
pre-corrected by :func:`latent_correlation` to land on the target afterwards.

That is what the module exists to get right. genslip reaches a non-negative slip field
by truncating ``1 + cov * Z`` at zero, which leaves a distribution with neither the
mean nor the spread it was configured with, and leaves the fitted correlation length
on a latent field that is never written out -- at the production spread the slip that
*is* written out is 4.4% shorter-correlated than Mai & Beroza (2002) asked for. A
truncated normal has both moments and, pre-corrected, both correlation lengths.

The Gaussian case is the identity throughout, so a field whose marginal really is
normal -- rake, the onset perturbation -- pays nothing for any of this.

Mai & Beroza (2002) equation (1) gives the von Karman autocorrelation directly:

.. math::

    C(r) = \\frac{G_H(r)}{G_H(0)}, \\qquad G_H(r) = r^H K_H(r), \\qquad
    r = \\sqrt{\\frac{x^2}{a_x^2} + \\frac{z^2}{a_z^2}}

``r`` is a **dimensionless** distance -- the separation measured in correlation
lengths, one per axis -- and ``C`` is the standard Matern correlation of smoothness
``H``.

References
----------
Cario, M. C., & Nelson, B. L. (1997). Modeling and generating random vectors with
arbitrary marginal distributions and correlation matrix. Technical report, Department
of Industrial Engineering and Management Sciences, Northwestern University.

Dietrich, C. R., & Newsam, G. N. (1993). A fast and exact method for multidimensional
Gaussian stochastic simulation. *Water Resources Research*, 29(8), 2861-2869.

Mai, P. M., & Beroza, G. C. (2002). A spatial random field model to characterize
complexity in earthquake slip. *Journal of Geophysical Research*, 107(B11), 2308.

Wood, A. T. A., & Chan, G. (1994). Simulation of stationary Gaussian processes in
[0,1]^d. *Journal of Computational and Graphical Statistics*, 3(4), 409-432.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.fft import next_fast_len
from scipy.optimize import brentq
from scipy.special import gamma, kv
from scipy.stats import gamma as gamma_distribution
from scipy.stats import norm, truncexpon, truncnorm

from rupture_generator import _kernels
from rupture_generator.errors import CapacityError, ConfigError

if TYPE_CHECKING:
    from rupture_generator.mesh import RuptureMesh

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

HURST = 0.75
"""The von Karman roughness exponent, and the Matern smoothness ``nu``.

Mai & Beroza (2002) figure 11: the median over their 44 finite-source models is 0.75
for the circular average, with 0.71 along strike and 0.77 down dip. One number for
both axes is theirs, not a simplification made here.
"""

MINIMUM_EMBEDDING = 2
"""How many times the fault each padded axis is, before any enlargement.

A Toeplitz matrix of ``n`` lags embeds in a circulant of ``2n - 2``: the first row
carries ``c(0) ... c(n-1)`` and then the mirror ``c(n-2) ... c(1)``. Anything smaller
cannot hold the covariance.
"""

MAXIMUM_DOUBLINGS = 3
"""How many times the margin may be doubled before the embedding is refused.

Three doublings from :data:`DECAY_LENGTHS` reaches a margin of 24 correlation lengths,
past which the covariance is not one this fault can carry.
"""

DECAY_LENGTHS = 3.0
"""How many correlation lengths of margin to try first, chosen so that the first
attempt usually embeds and no doubling is allocated."""

CORRELATION_LENGTH_TOLERANCE = 0.02
"""How far the delivered correlation length may sit from the one asked for, as a
fraction per axis: 0.02 is within two percent on each axis."""

MAXIMUM_VARIANCE_DEFICIT = 1.0e-10
"""How much of a covariance's variance may sit in unsamplable directions and still be
round-off, as a fraction of what is kept.

An embedding is sampled through the square roots of its eigenvalues, so a negative one
has no square root and is clipped to zero -- which drops the variance it carried. At
this level that is the transform's own error and the field is the one that was asked
for. Past it the covariance genuinely is not positive definite on this grid, and the
draw comes back with less variance at the shortest wavelengths than the covariance
specifies; the sampler says so rather than failing, since the field is still usable
and a larger margin is the usual cure.

The marginal transform is what makes this reachable at all: it is monotone, so it
preserves the *ordering* of a field, but it does not have to preserve positive
definiteness of a covariance.
"""

MAI_MAXIMUM_RATIO = 0.6
"""The largest correlation length, as a fraction of the source dimension, that the
model was fitted on.

Mai & Beroza (2002) figure 13: across all 44 finite-source models the ratio sits
between 0.25 and 0.6 on each axis.
"""

MAXIMUM_EMBEDDING_CELLS = 1 << 26
"""The largest padded grid to transform, in cells; past it the draw is refused.

A memory bound, and the number is the transform's peak rather than the grid's:
``_attempt`` evaluates the covariance on the padded grid as ``float64`` and then takes
its ``complex128`` transform, so at 2\\ :sup:`26` = 67.1 M cells that is 537 MB for the
covariance and 1.07 GB for the transform, live at the same time and before the
caller's own mesh and fields. On a 30 GB machine with a 12 GB address-space limit that
is the largest single allocation leaving room for the rest of a run.

For scale: the CFM Hikurangi interface cut at 400 m embeds to 6.9 M cells, a tenth of
the cap; the same interface cut at 100 m embeds to 111.8 M, 1.67 times it.
"""


@dataclasses.dataclass(frozen=True)
class VonKarmanFilterParameters:
    """How far a field's structure reaches, and how rough it is.

    The correlation lengths are the patch size along strike and down dip, in km: the
    corner of the spectrum is the ellipse through their reciprocals.
    """

    correlation_length_strike_km: float
    correlation_length_dip_km: float
    hurst: float = HURST

    def __post_init__(self) -> None:
        """Refuse a spec that cannot describe a field."""
        for name in (
            "correlation_length_strike_km",
            "correlation_length_dip_km",
        ):
            value = getattr(self, name)
            if not (value > 0.0) or not np.isfinite(value):
                raise ConfigError(f"{name} must be a positive length, got {value}")
        if not (0.0 < self.hurst < 1.0):
            raise ConfigError(f"hurst must be in (0, 1), got {self.hurst}")


SUZUKI_COEFFICIENTS = {
    "strike_offset": 1.67 + np.log10(2 * np.pi),
    "dip_offset": 1.69 + np.log10(2 * np.pi),
    "strike_exponent": 0.5,
    "dip_exponent": 0.5,
    "dip_saturation_magnitude": 6.3,
}
"""Suzuki et al. (2022)'s corner relation, as :func:`correlation_lengths` takes it.

The shallow branch of the hybrid profile -- see
:class:`~rupture_generator.config.rupture.HybridConfig` for why there is one. Its
down-dip length **saturates** above magnitude 6.3, which is genslip's crustal default
for the parameter and is not read from a file when Suzuki is only the shallow branch,
so the number is here rather than in the config schema. At M7.2 this gives 13.55 km
along strike and 4.59 km down dip, against Mai & Beroza's 2.00 and 1.26 -- a ratio of
6.8 on one axis and 3.6 on the other, which is why the profile cannot be stated as one
factor on the deep lengths.
"""


def correlation_lengths(
    magnitude: float,
    *,
    strike_offset: float = 2.50 + np.log10(2 * np.pi),
    dip_offset: float = 1.50 + np.log10(2 * np.pi),
    strike_exponent: float = 0.5,
    dip_exponent: float = 1.0 / 3.0,
    dip_saturation_magnitude: float | None = None,
) -> VonKarmanFilterParameters:
    """A corner relation's correlation lengths for a magnitude, in kilometres.

    .. math::

        \\lambda_{strike} = 10^{c_{s} M_w - a}, \\qquad
        \\lambda_{dip}    = 10^{c_{d} \\min(M_w, M_c) - b}

    The defaults are Mai & Beroza (2002), which have no :math:`M_c`. The dip exponent
    is **a third, not 0.3333**: their equation (5) reads ``log(a_z) ~ -1.5 + (1/3)
    Mw``.

    ``dip_saturation_magnitude`` is :math:`M_c`, above which the down-dip length stops
    growing. Mai & Beroza's does not saturate and this is ``None`` for them;
    :data:`SUZUKI_COEFFICIENTS` carries one, because a fault only has so much width
    and a relation fitted across it cannot keep extrapolating.
    """
    dip_magnitude = (
        magnitude
        if dip_saturation_magnitude is None
        else min(magnitude, dip_saturation_magnitude)
    )
    return VonKarmanFilterParameters(
        correlation_length_strike_km=10.0
        ** (strike_exponent * magnitude - strike_offset),
        correlation_length_dip_km=10.0 ** (dip_exponent * dip_magnitude - dip_offset),
    )


# NORTA: what distribution a field's values follow, and what it costs the correlation


NORTA_ORDER = 20
"""How many Hermite terms a marginal transform is expanded in.

The series :math:`\\sum_k b_k` is exactly 1 for a converged expansion, which
:func:`_hermite_coefficients` asserts: at 20 terms both production marginals reach
1.000000, so the truncation is not what limits the accuracy here.
"""

NORTA_QUADRATURE_POINTS = 160
"""Gauss-Hermite nodes used to project a marginal onto the Hermite basis.

Enough that the highest term, :math:`He_{20}`, is integrated exactly -- Gauss-Hermite
on ``n`` nodes is exact to degree ``2n - 1`` -- with the margin going to the quantile
function, which is not a polynomial.
"""

NORTA_TAIL = 1.0e-12
"""How far into a marginal's tails the quantile function is evaluated.

``Phi(z)`` underflows to 0 and rounds to 1 at the outermost quadrature nodes
(``|z| = 24.3``), and a marginal with unbounded support answers those with an
infinity. Clipping the probability puts the node at a finite, extreme quantile
instead.
"""

NORTA_INVERSE_POINTS = 20001
"""Points on ``[-1, 1]`` the correlation series is inverted through.

:math:`g` has no closed-form inverse, so it is tabulated and read backwards. The
spacing is 1e-4 in the latent correlation and the inversion is linear between
samples; the round trip through :func:`latent_correlation` and
:func:`transformed_correlation` measures 2e-10, seven orders below
:data:`CORRELATION_LENGTH_TOLERANCE`.
"""

NORTA_CORRELATION_SLACK = 1.0e-9
"""How far past an attainable correlation a target may sit and still be round-off.

The auto series sums to 1 to within 2e-16 and a covariance reaches exactly 1 at zero
lag, so the two ends miss each other by round-off that is not a modelling error.
Above this the target is one the marginals genuinely cannot share.
"""

TRUNCATED_NORMAL_MAXIMUM_COV = 1.0
"""The coefficient of variation a unit-mean truncated normal cannot reach.

The family is parameterised by where zero falls in standard deviations: pushing that
point to ``-inf`` leaves the whole half-line and a spread approaching the mean, so the
coefficient of variation rises to 1 without attaining it. Above this a truncated
normal is the wrong marginal, not a badly conditioned one.
"""

TRUNCATED_EXPONENTIAL_MINIMUM_COV = 1.0 / math.sqrt(3.0)
"""The coefficient of variation a unit-mean truncated exponential cannot fall to.

The family is parameterised by where the cut sits in decay lengths: pulling it down
towards zero leaves the exponential no room to decay over, so the shape flattens into
a uniform on ``[0, 2]`` and the spread stops falling at :math:`1/\\sqrt{3} = 0.5774`.
Below this a truncated exponential is the wrong marginal, not a badly conditioned one
-- a truncated normal reaches any spread down to zero, and so does a gamma.
"""

TRUNCATED_EXPONENTIAL_MAXIMUM_COV = 1.0
"""The coefficient of variation a unit-mean truncated exponential cannot reach.

Pushing the cut the other way, to infinity, leaves the whole exponential, whose
spread equals its mean; so the coefficient of variation rises to 1 without attaining
it. The family's window is therefore the **open** interval ``(0.5774, 1)``, narrower
at both ends than either of the others, and a spread outside it names a distribution
this family does not contain.
"""

type MarginalFamily = Literal[
    "normal", "truncated_normal", "truncated_exponential", "gamma"
]
"""The distributions a field's values may follow."""


@dataclasses.dataclass(frozen=True)
class Marginal:
    """The distribution one field's values follow, one draw per subfault.

    This is the *marginal* of the NORTA construction (Cario & Nelson 1997): a field is
    drawn as a Gaussian and then pushed through :math:`F^{-1}(\\Phi(\\cdot))`, which
    gives every cell the requested distribution exactly rather than approximately.
    What that costs is correlation -- a monotone nonlinear map always destroys some --
    and :func:`latent_correlation` buys it back by asking the sampler for more
    correlation than the field is meant to end up with.

    ``normal`` is the standard normal and is the identity: a field whose marginal is
    already Gaussian needs no transform and no pre-correction, so every function here
    reduces to what this module did before NORTA. The other three families are
    **unit-mean**, because the fields that use them are patterns whose size is set
    later -- by :func:`~rupture_generator.moment.scale_to_moment` for slip and by the
    requested average for rise time -- and ``coefficient_of_variation`` is their
    dimensionless spread.

    ``truncated_normal`` is what replaces genslip's clip: ``1 + cov * Z`` truncated at
    zero is a normal with a point mass at zero and neither the mean nor the spread it
    was asked for, where a truncated normal is a distribution with both. ``gamma``
    is the same idea for rise time, and is preferred there because its mode sits away
    from zero -- a near-zero rise time carrying finite slip is an unbounded slip rate.

    ``truncated_exponential`` is the marginal Thingbaijam & Mai (2016) fitted to the
    190 finite-source models of SRCMOD and found the best of the six they tried, and
    the one Castro-Cruz & Mai (2025) equation (3) rescale their von Karman field onto
    -- the same NORTA construction as this class, arrived at independently. Its
    parameter is where the distribution is cut off, so it fixes the largest slip on
    the fault as well as the spread: the two are one number here (see
    :func:`_distribution`), and the cut is what makes the tail finite rather than
    something a taper has to bring back down. It is the most skewed of the three,
    which is what it costs -- a monotone map's damage to correlation grows with how
    far it departs from linear, so :func:`attainable_correlation` against a Gaussian
    or a gamma field is lower here than for a truncated normal of the same spread,
    and `[timing]`'s slip correlations are checked against it rather than assumed.

    References
    ----------
    Cario, M. C., & Nelson, B. L. (1997). Modeling and generating random vectors with
    arbitrary marginal distributions and correlation matrix. Technical report,
    Northwestern University.

    Castro-Cruz, D., & Mai, P. M. (2025). A new kinematic rupture generation
    technique. *Geophysical Journal International*, 243(1).

    Liu, P.-L., & Der Kiureghian, A. (1986). Multivariate distribution models with
    prescribed marginals and covariances. *Probabilistic Engineering Mechanics*,
    1(2), 105-112.

    Thingbaijam, K. K. S., & Mai, P. M. (2016). Evidence for truncated exponential
    probability distribution of earthquake slip. *Bulletin of the Seismological
    Society of America*, 106(4), 1802-1816.
    """

    family: MarginalFamily = "normal"
    coefficient_of_variation: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a marginal no distribution answers to."""
        if self.family == "normal":
            return
        value = self.coefficient_of_variation
        if not (value > 0.0) or not np.isfinite(value):
            raise ConfigError(
                f"a {self.family} marginal needs a positive coefficient of variation, "
                f"got {value}"
            )
        if self.family == "truncated_normal" and value >= TRUNCATED_NORMAL_MAXIMUM_COV:
            raise ConfigError(
                f"a unit-mean truncated normal cannot have a coefficient of variation "
                f"of {value}: the family reaches 1 only in the limit where zero is "
                "infinitely far below the mean. Use a gamma marginal, which has no "
                "such bound"
            )
        if self.family == "truncated_exponential" and not (
            TRUNCATED_EXPONENTIAL_MINIMUM_COV
            < value
            < TRUNCATED_EXPONENTIAL_MAXIMUM_COV
        ):
            raise ConfigError(
                f"a unit-mean truncated exponential cannot have a coefficient of "
                f"variation of {value}: the family runs from "
                f"{TRUNCATED_EXPONENTIAL_MINIMUM_COV:.4f} in the limit where the cut "
                "reaches the mean and the shape flattens into a uniform, up to 1 in "
                "the limit where the cut goes to infinity and nothing is truncated, "
                "and attains neither end. Use a truncated normal below that window, "
                "or a gamma on either side of it"
            )

    @property
    def is_normal(self) -> bool:
        """Whether this marginal is the identity transform."""
        return self.family == "normal"

    @property
    def is_positive(self) -> bool:
        """Whether this marginal's support excludes negative values.

        What a stage asks before raising a field to a power or dividing by it.
        """
        return self.family in ("truncated_normal", "truncated_exponential", "gamma")

    def apply(self, latent: FloatArray) -> FloatArray:
        """Give a standard normal field this marginal, cell by cell.

        :math:`F^{-1}(\\Phi(u))`, the monotone map that carries a standard normal onto
        this distribution. Monotone, so it moves no cell past another: the field's
        *shape* -- which patch is the large one -- is the latent's, and only the values
        change. The latent must be an **unstandardised** draw, since standardising
        divides by a sample spread and the transform is a statement about the
        population.
        """
        if self.is_normal:
            return np.asarray(latent, dtype=np.float64)
        probability = np.clip(norm.cdf(latent), NORTA_TAIL, 1.0 - NORTA_TAIL)
        return np.asarray(_distribution(self).ppf(probability), dtype=np.float64)


NORMAL = Marginal()
"""The standard normal: the marginal that leaves the sampler exactly as it was.

Every transform and pre-correction in this module is the identity here, so a field
that is meant to be Gaussian -- rake, the onset perturbation -- costs nothing for
NORTA being available.
"""


def _truncated_exponential_spread(cut: float) -> float:
    """The coefficient of variation of a unit exponential cut off at ``cut`` decays.

    .. math:: \\frac{\\sigma}{\\mu} = \\frac{\\sqrt{u^2 - b^2 u - b^2}}{u - b},
              \\qquad u = e^b - 1

    Written out rather than read off ``truncexpon(b).std() / .mean()``, which is what
    the truncated normal above can afford to do. A barely-truncated exponential is
    nearly a uniform, and SciPy forms its moments as differences that cancel to
    nothing there: below a cut of about 0.01 its coefficient of variation is noise in
    the fourth digit -- it is not even monotone -- and below 1e-5 the variance comes
    out negative and the answer is a NaN. Over the common denominator the cancellation
    is confined to the numerator, which holds twelve digits down to a cut of 5e-4 and
    lets the bracket in :func:`_distribution` reach the family's floor rather than
    stopping a percent above it. The scale cancels, which is why one argument suffices.
    """
    grown = math.expm1(cut)
    return math.sqrt(grown * grown - cut * cut * grown - cut * cut) / (grown - cut)


@functools.lru_cache(maxsize=16)
def _distribution(marginal: Marginal) -> Any:
    """The frozen SciPy distribution a :class:`Marginal` names.

    The three unit-mean families are fitted here rather than stated, because what the
    model carries is a mean and a spread and none of them is parameterised by those.
    A gamma inverts in closed form -- its coefficient of variation is
    :math:`1/\\sqrt{a}` and its mean is :math:`a\\theta`. The two truncated families do
    not, but they separate: truncating :math:`N(\\alpha, 1)` at zero gives a
    coefficient of variation that depends on ``alpha`` **alone**, and cutting a unit
    exponential off at ``b`` decay lengths gives one that depends on ``b`` alone,
    since in both cases mean and spread scale together with the width. So one
    bracketed root-find fixes the shape and a division fixes the mean, rather than a
    two-dimensional solve for both at once.

    For the truncated exponential that one shape parameter also fixes the fault's
    largest slip. The support ends at the cut, and the division that makes the mean 1
    divides that too, so the cut in decay lengths and the coefficient of variation and
    Thingbaijam & Mai (2016)'s ``u_max / u_bar`` are three names for one number: the
    shipped slip spread of 0.75 is a cut of 1.830 and a maximum of 2.81 mean slips,
    where their equation (4) regression on SRCMOD puts that ratio at 3.9 to 4.4 -- a
    spread nearer 0.90.
    """
    if marginal.is_normal:
        return norm(0.0, 1.0)

    spread = marginal.coefficient_of_variation
    if marginal.family == "gamma":
        shape = 1.0 / (spread * spread)
        return gamma_distribution(a=shape, scale=1.0 / shape)

    if marginal.family == "truncated_exponential":
        # Monotone *increasing* in the cut -- the opposite sense to the truncated
        # normal below -- from 1/sqrt(3) as the cut goes to zero to 1 as it goes to
        # infinity, so any spread `__post_init__` admits is bracketed. The ends are
        # where double precision runs out rather than where the family does: a cut of
        # 5e-4 is a coefficient of variation of 0.5773984, five parts in a hundred
        # thousand above the family's own floor, and a cut of 36 is 1 - 1.4e-13.
        cut = brentq(
            lambda value: _truncated_exponential_spread(value) - spread,
            5.0e-4,
            36.0,
            xtol=1.0e-13,
        )
        mean = float(truncexpon(b=cut, loc=0.0, scale=1.0).mean())
        return truncexpon(b=cut, loc=0.0, scale=1.0 / mean)

    def spread_at(alpha: float) -> float:
        """The coefficient of variation when zero sits ``alpha`` widths below."""
        shifted = truncnorm(a=-alpha, b=np.inf, loc=alpha, scale=1.0)
        return float(shifted.std() / shifted.mean())

    # Monotone decreasing in alpha, from 1 as alpha -> -inf to 0 as alpha -> +inf, so
    # any spread `__post_init__` admits is bracketed. The bracket is wide rather than
    # tight: alpha = -40 is a coefficient of variation of 0.9989 and alpha = 60 is
    # 0.0167, which covers everything the family can be asked for.
    alpha = brentq(lambda value: spread_at(value) - spread, -40.0, 60.0, xtol=1.0e-13)
    mean = float(truncnorm(a=-alpha, b=np.inf, loc=alpha, scale=1.0).mean())
    return truncnorm(a=-alpha, b=np.inf, loc=alpha / mean, scale=1.0 / mean)


@functools.lru_cache(maxsize=16)
def _hermite_coefficients(marginal: Marginal) -> tuple[FloatArray, float]:
    """A marginal's Hermite coefficients, and the spread they imply.

    Writes :math:`h(z) = F^{-1}(\\Phi(z))` in the probabilists' Hermite basis,

    .. math:: h(z) = \\sum_k a_k He_k(z), \\qquad
              a_k = \\frac{1}{k!} \\mathbb{E}[h(Z) He_k(Z)]

    by Gauss-Hermite quadrature. The expectation is what makes the correlation series
    below a polynomial: for jointly standard normal :math:`(U, V)` correlated at
    :math:`\\rho`, :math:`\\mathbb{E}[He_j(U) He_k(V)] = \\delta_{jk} k! \\rho^k`, so
    the covariance of two transformed fields is a power series in the latent
    correlation with no cross terms.

    Returns the coefficients and :math:`\\sigma = \\sqrt{\\sum_{k\\ge1} a_k^2 k!}`, the
    transformed field's standard deviation, which normalises that series.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(NORTA_QUADRATURE_POINTS)
    # hermegauss integrates against exp(-x^2/2); the standard normal density is that
    # over sqrt(2 pi), so this turns the quadrature into an expectation.
    weights = weights / np.sqrt(2.0 * np.pi)
    transformed = marginal.apply(nodes)

    orders = np.arange(NORTA_ORDER + 1)
    factorials = np.array([float(math.factorial(k)) for k in orders])
    basis = np.polynomial.hermite_e.hermevander(nodes, NORTA_ORDER)
    coefficients = (weights * transformed) @ basis / factorials
    return coefficients, float(np.sqrt(np.sum(coefficients[1:] ** 2 * factorials[1:])))


@functools.lru_cache(maxsize=64)
def _correlation_series(first: Marginal, second: Marginal) -> FloatArray:
    """Coefficients of :math:`g_{fg}`, the map from latent to delivered correlation.

    .. math:: g_{fg}(\\rho) = \\sum_{k\\ge1}
              \\frac{a^f_k a^g_k k!}{\\sigma_f \\sigma_g} \\rho^k

    With ``first is second`` this is the auto series, whose coefficients are
    non-negative; it is increasing and sums to exactly 1, so :math:`g(1) = 1` and a
    field can be perfectly correlated with itself. The **lower** end is not
    symmetric: alternating signs make :math:`g(-1) = \\sum_k (-1)^k b_k`, which for a
    skewed marginal is well above ``-1`` -- a truncated normal at the production
    spread bottoms out at -0.861, because no Gaussian copula can make a distribution
    with a long right tail the mirror of itself. Between different marginals the
    series sums to :math:`g_{fg}(1) \\le 1`; see :func:`attainable_correlation`.
    """
    left, left_spread = _hermite_coefficients(first)
    right, right_spread = _hermite_coefficients(second)
    factorials = np.array([float(math.factorial(k)) for k in range(1, NORTA_ORDER + 1)])
    return left[1:] * right[1:] * factorials / (left_spread * right_spread)


def _evaluate_series(coefficients: FloatArray, latent: FloatArray) -> FloatArray:
    """A correlation series at a latent correlation, by Horner from the top down."""
    latent = np.asarray(latent, dtype=np.float64)
    delivered = np.zeros_like(latent)
    for coefficient in coefficients[::-1]:
        delivered = (delivered + coefficient) * latent
    return delivered


def transformed_correlation(
    first: Marginal, second: Marginal, latent: FloatArray
) -> FloatArray:
    """What correlation two NORTA fields have, given their latents' correlation.

    :math:`g_{fg}`, applied pointwise. The identity when both marginals are normal.
    """
    if first.is_normal and second.is_normal:
        return np.asarray(latent, dtype=np.float64)
    return _evaluate_series(_correlation_series(first, second), latent)


def latent_correlation(
    first: Marginal, second: Marginal, target: FloatArray
) -> FloatArray:
    """What correlation to ask the sampler for, to deliver ``target`` on the fields.

    :math:`g_{fg}^{-1}`, the NORTA pre-correction. Without it the correlation the
    model is parameterised on lands on a latent field that is never written out, and
    the field that *is* written out is measurably less correlated -- which is what
    genslip's clip-at-zero silently gives up.

    :math:`g` is increasing, so the inverse exists between the two correlations the
    marginals can actually reach; it is tabulated on :data:`NORTA_INVERSE_POINTS` and
    read backwards.

    Raises
    ------
    ConfigError
        If the target sits outside that range by more than
        :data:`NORTA_CORRELATION_SLACK` -- no latent delivers it, and silently
        returning the nearest one that exists would answer a different question than
        the caller asked.
    """
    if first.is_normal and second.is_normal:
        return np.asarray(target, dtype=np.float64)

    coefficients = _correlation_series(first, second)
    grid = np.linspace(-1.0, 1.0, NORTA_INVERSE_POINTS)
    delivered = _evaluate_series(coefficients, grid)
    # np.interp needs an increasing table and says nothing when it does not have one.
    if not np.all(np.diff(delivered) > 0.0):
        raise ConfigError(
            f"the correlation map between a {first.family} and a {second.family} "
            "marginal is not increasing, so it cannot be inverted -- the Hermite "
            "expansion has not converged for one of them"
        )

    target = np.asarray(target, dtype=np.float64)
    floor, ceiling = float(delivered[0]), float(delivered[-1])
    worst = max(float(target.max()) - ceiling, floor - float(target.min()))
    if worst > NORTA_CORRELATION_SLACK:
        raise ConfigError(
            f"a {first.family} and a {second.family} marginal can be correlated "
            f"between {floor:.4f} and {ceiling:.4f} under a Gaussian copula, and "
            f"{float(target.min()):.4f} to {float(target.max()):.4f} was asked for. "
            "Two fields pushed through different monotone maps cannot be made more "
            "alike than the maps are"
        )
    return np.interp(np.clip(target, floor, ceiling), delivered, grid)


def attainable_correlation(first: Marginal, second: Marginal) -> float:
    """The largest correlation two marginals can share, :math:`g_{fg}(1)`.

    The Gaussian-copula ceiling. Two fields with different marginals cannot be
    perfectly correlated even when their latents are: pushing one standard normal
    through two different monotone maps gives two different fields. At the production
    spread of 0.75 a truncated-normal slip reaches 0.9934 with a gamma rise time and
    0.9638 with a Gaussian field, both above what `[timing]` asks for.
    """
    return float(np.sum(_correlation_series(first, second)))


def von_karman_correlation(
    normalised_distance: FloatArray, hurst: float = HURST
) -> FloatArray:
    """Mai & Beroza (2002) equation (1)'s von Karman ACF, at a dimensionless distance.

    .. math:: C(r) = \\frac{G_H(r)}{G_H(0)} = \\frac{2^{1-H}}{\\Gamma(H)} r^H K_H(r)

    ``K_H`` is the modified Bessel function of the second kind. The argument is a
    **distance in correlation lengths**, not kilometres; the anisotropy lives in how
    that distance is formed (see :func:`_wrapped_distance`). ``C(1) = 0.5005`` at
    ``H = 0.75`` -- one correlation length is where half the field is forgotten.
    """
    distance = np.asarray(normalised_distance, dtype=np.float64)
    correlation = np.ones_like(distance)
    # The limit at r = 0 is 1 and the expression there is 0 * inf, so it is taken
    # rather than evaluated.
    away = distance > 0.0
    scaled = distance[away]
    correlation[away] = (
        2.0 ** (1.0 - hurst) / gamma(hurst) * scaled**hurst * kv(hurst, scaled)
    )
    return correlation


def _wrapped_distance(
    extents: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
) -> FloatArray:
    """Distance to every cell of the periodic grid, in correlation lengths.

    **Wrapped**, which is what makes the embedding circulant: on a periodic grid of
    ``m`` cells the lag to cell ``p`` is ``min(p, m - p)``, and unwrapped lags build a
    Toeplitz matrix whose DFT is not its eigenvalues. Each axis is divided by its own
    correlation length first, so the result is Mai & Beroza (2002) equation (1)'s
    ``r``.
    """
    return _quadrant_distance(extents, spacing_km, parameters)[
        np.ix_(*(_wrapped_lag_index(extent) for extent in extents))
    ]


def _wrapped_lag_index(extent: int) -> np.ndarray:
    """Which distinct lag each cell of a periodic axis is at: ``min(p, m - p)``."""
    positions = np.arange(extent)
    return np.minimum(positions, extent - positions)


def _quadrant_distance(
    extents: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
) -> FloatArray:
    """The distinct wrapped distances only: one quadrant of the padded grid.

    ``min(p, m - p)`` takes each value twice on each axis, and the expensive step is a
    modified Bessel function per point: on a 400x2400 embedding, 241 thousand
    evaluations rather than 960 thousand.
    """
    padded_i, padded_j = extents
    strike_km, dip_km = spacing_km

    dip_lag = (
        np.arange(padded_i // 2 + 1) * dip_km / parameters.correlation_length_dip_km
    )
    strike_lag = (
        np.arange(padded_j // 2 + 1)
        * strike_km
        / parameters.correlation_length_strike_km
    )
    return np.sqrt(dip_lag[:, None] ** 2 + strike_lag[None, :] ** 2)


class DegradedCorrelation(UserWarning):
    """A warning provided if the correlation fails to converge."""


@dataclasses.dataclass(frozen=True)
class Embedding:
    """A circulant embedding of one covariance on one grid.

    ``extents`` is the padded grid ``(i, j)``, ``correlation_lengths`` the ones the
    field will actually have, ``(strike, dip)``, and ``correlation_length_error`` how
    far those sit from what was asked, as a fraction, worst axis.
    ``variance_deficit`` is how much variance the embedding had to drop over how much
    it kept, which is round-off for a covariance and a real failure for a
    pre-corrected one -- :func:`latent_correlation` is monotone but not
    positive-definiteness preserving.
    """

    extents: tuple[int, int]
    eigenvalues: FloatArray
    correlation_lengths: tuple[float, float]
    correlation_length_error: float
    variance_deficit: float


@functools.lru_cache(maxsize=32)
def _embed(
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
    marginal: Marginal = NORMAL,
) -> Embedding:
    """Embed this covariance on this grid, as closely as the grid allows.

    ``marginal`` is the distribution the field's values will be given, and the
    embedding is of the **latent** covariance that delivers this one afterwards -- see
    :func:`latent_correlation`. It is part of the cache key because it changes the
    eigenvalues, not just what is done with them.

    Warns
    -----
    DegradedCorrelation
        If the embedding fails to reproduce the correlation structure precisely, or
        no candidate is positive definite.
    """
    best: Embedding | None = None
    fallback: Embedding | None = None
    for extents in _candidate_extents(cell_counts, spacing_km, parameters):
        candidate = _attempt(extents, cell_counts, spacing_km, parameters, marginal)
        # Kept whatever its eigenvalues, so a grid that cannot carry this covariance at
        # any margin still gets a field.
        if (
            fallback is None
            or candidate.correlation_length_error < fallback.correlation_length_error
        ):
            fallback = candidate
        # A candidate that is not positive definite is not an embedding of anything, so
        # it does not compete on correlation length. A larger margin is the usual cure.
        if candidate.variance_deficit > MAXIMUM_VARIANCE_DEFICIT:
            continue
        # Not assumed monotone in the margin, though it is in practice.
        if (
            best is None
            or candidate.correlation_length_error < best.correlation_length_error
        ):
            best = candidate
        if candidate.correlation_length_error <= CORRELATION_LENGTH_TOLERANCE:
            break

    assert fallback is not None, "_candidate_extents never returns an empty list"
    chosen = best if best is not None else fallback
    _warn_if_degraded(cell_counts, spacing_km, parameters, chosen, marginal)
    return chosen


_WARN_STACKLEVEL = 5
"""How far up to point a :class:`DegradedCorrelation`, counted rather than guessed.

``_warn_if_degraded`` -> ``_embed`` -> :func:`von_karman_grid` ->
:func:`von_karman_field` -> the stage, which is five frames.
"""


def _warn_if_degraded(
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
    best: Embedding,
    marginal: Marginal = NORMAL,
) -> None:
    strike_km, dip_km = spacing_km
    length_km = cell_counts[1] * strike_km
    width_km = cell_counts[0] * dip_km
    ratios = (
        parameters.correlation_length_strike_km / length_km,
        parameters.correlation_length_dip_km / width_km,
    )

    if max(ratios) > MAI_MAXIMUM_RATIO:
        warnings.warn(
            f"a {length_km:.3g} x {width_km:.3g} km segment cannot carry correlation "
            f"lengths of {parameters.correlation_length_strike_km:.3g} km along strike "
            f"and {parameters.correlation_length_dip_km:.3g} km down dip -- they are "
            f"{ratios[0]:.2g} and {ratios[1]:.2g} of the segment, where Mai & Beroza "
            f"(2002) figure 13 puts every model they fitted between 0.25 and "
            f"{MAI_MAXIMUM_RATIO}. The field it gets varies little across the segment. "
            "Slip, moment and timing are unaffected; what is degraded is how the slip "
            "is distributed",
            DegradedCorrelation,
            stacklevel=_WARN_STACKLEVEL,
        )
    elif best.correlation_length_error > CORRELATION_LENGTH_TOLERANCE:
        warnings.warn(
            _degraded_message(cell_counts, spacing_km, parameters, best),
            DegradedCorrelation,
            stacklevel=_WARN_STACKLEVEL,
        )
    elif best.variance_deficit > MAXIMUM_VARIANCE_DEFICIT:
        warnings.warn(
            f"no embedding of this covariance under a {marginal.family} marginal is a "
            f"covariance matrix: {best.variance_deficit:.2%} of its variance sits in "
            "directions that cannot be sampled, past the "
            f"{MAXIMUM_VARIANCE_DEFICIT:.0e} that is round-off. They are dropped, which "
            "delivers a field with that much less variance at the shortest wavelengths "
            "than was asked for. The marginal transform is monotone but does not have "
            "to preserve positive definiteness",
            DegradedCorrelation,
            stacklevel=_WARN_STACKLEVEL,
        )


def _attempt(
    extents: tuple[int, int],
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
    marginal: Marginal,
) -> Embedding:
    """Embed one covariance at one padded size, measured in the field's own space.

    Both correlation lengths are read off **transformed** correlations, not latent
    ones. Under NORTA the sampler is deliberately asked for a covariance the field
    will not have -- that is the whole of the pre-correction -- so comparing the
    latent it delivers against the latent it was asked for would report on a field
    nobody writes out, and would call a correct embedding wrong by the 4.4% that
    separates the two at the production spread.
    """
    quadrant = von_karman_correlation(
        _quadrant_distance(extents, spacing_km, parameters), parameters.hurst
    )
    # The pre-correction is pointwise, so it commutes with the mirroring that turns a
    # quadrant into the padded grid: applying it here is a quarter of the work, and
    # leaves the padded grid the only full-size array this function holds.
    latent = latent_correlation(marginal, marginal, quadrant)[
        np.ix_(*(_wrapped_lag_index(extent) for extent in extents))
    ]

    spectrum = np.fft.fft2(latent).real
    eigenvalues = np.maximum(spectrum, 0.0)
    # The clipped array is already here, so the two sums cost no further allocation:
    # what it keeps, and what it keeps less the signed total, which is what it dropped.
    kept = float(eigenvalues.sum())
    deficit = (kept - float(spectrum.sum())) / kept if kept > 0.0 else 0.0
    delivered = np.fft.ifft2(eigenvalues).real

    lengths = _delivered_lengths(
        delivered, cell_counts, spacing_km, parameters, marginal
    )
    # The quadrant is the target in the field's own space already, and its first row
    # and column are the same two profiles the padded grid would give -- so this needs
    # no second full-size array.
    wanted = _delivered_lengths(quadrant, cell_counts, spacing_km, parameters, NORMAL)

    return Embedding(
        extents=extents,
        eigenvalues=eigenvalues,
        correlation_lengths=lengths,
        correlation_length_error=_relative_error(lengths, wanted),
        variance_deficit=deficit,
    )


def _candidate_extents(
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
) -> list[tuple[int, int]]:
    """Progressively larger embeddings to try, smallest first, never empty.

    The smallest entry is the smallest embedding a Toeplitz matrix of this many lags
    admits at all, so a covariance the fault is too small to carry still gets a field,
    degraded and warned about rather than refused; the cap binds on that one too.
    ``cell_counts`` is ``(dip, strike)``, ``spacing_km`` the other way round.

    Raises
    ------
    CapacityError
        If even the minimum embedding is past :data:`MAXIMUM_EMBEDDING_CELLS`.
    """
    smallest = (
        int(next_fast_len(MINIMUM_EMBEDDING * cell_counts[0])),
        int(next_fast_len(MINIMUM_EMBEDDING * cell_counts[1])),
    )
    if smallest[0] * smallest[1] > MAXIMUM_EMBEDDING_CELLS:
        raise CapacityError(
            f"a {cell_counts[0]} x {cell_counts[1]} grid embeds in no less than "
            f"{smallest[0]} x {smallest[1]} = {smallest[0] * smallest[1]:,} cells, past "
            f"the {MAXIMUM_EMBEDDING_CELLS:,} this machine can transform -- a circulant "
            "embedding is at least twice the grid on each axis and there is nothing "
            "smaller to try. Either cut this segment into larger subfaults, or raise "
            "`sampling.MAXIMUM_EMBEDDING_CELLS`, whose docstring says what one cell "
            "costs while the transform is live"
        )

    candidates: list[tuple[int, int]] = []
    for doubling in range(MAXIMUM_DOUBLINGS + 1):
        extents = _predicted_extents(
            cell_counts, spacing_km, parameters, DECAY_LENGTHS * 2**doubling
        )
        # Monotone in the margin, so the first over-cap candidate ends the search
        # rather than skipping one.
        if extents[0] * extents[1] > MAXIMUM_EMBEDDING_CELLS:
            break
        if extents not in candidates:
            candidates.append(extents)
    return candidates or [smallest]


def _degraded_message(
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
    best: Embedding,
) -> str:
    """What the field got instead, in the units the model is parameterised in."""
    strike_km, dip_km = spacing_km
    length_km = cell_counts[1] * strike_km
    width_km = cell_counts[0] * dip_km
    return (
        f"a {length_km:.3g} x {width_km:.3g} km segment cannot carry correlation "
        f"lengths of {parameters.correlation_length_strike_km:.3g} km along strike and "
        f"{parameters.correlation_length_dip_km:.3g} km down dip -- they are "
        f"{parameters.correlation_length_strike_km / length_km:.2g} and "
        f"{parameters.correlation_length_dip_km / width_km:.2g} of the segment, where "
        "Mai & Beroza (2002) figure 13 puts every model they fitted between 0.25 and "
        f"0.6. The field it gets has {best.correlation_lengths[0]:.3g} x "
        f"{best.correlation_lengths[1]:.3g} km instead, off by "
        f"{best.correlation_length_error * 100:.0f}%: the largest patches this grid can "
        "carry. Slip, moment and timing are unaffected; what is degraded is how the "
        "slip is distributed"
    )


def _delivered_lengths(
    covariance: FloatArray,
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
    marginal: Marginal = NORMAL,
) -> tuple[float, float]:
    """The correlation lengths a covariance array actually has, ``(strike, dip)``.

    ``marginal`` says what space ``covariance`` is in: :data:`NORMAL` for one already
    in the field's space, and a field's own marginal for a latent one, which is then
    mapped through :func:`transformed_correlation` first. Only the two profiles
    through the origin are read, so the array may be the padded grid or the quadrant
    it was mirrored from.
    """
    strike_km, dip_km = spacing_km
    half = float(von_karman_correlation(np.array([1.0]), parameters.hurst)[0])
    profiles = (covariance[0, : cell_counts[1]], covariance[: cell_counts[0], 0])
    along_strike, down_dip = (
        transformed_correlation(marginal, marginal, profile) for profile in profiles
    )
    return (
        _crossing_km(along_strike, half, strike_km),
        _crossing_km(down_dip, half, dip_km),
    )


def _relative_error(
    delivered: tuple[float, float], wanted: tuple[float, float]
) -> float:
    """How far the delivered correlation lengths are from the target's, worst axis.

    Both are measured **by the same estimator on the same lags**, so the grid's own
    discretisation cancels: a fault cut at 1 km carrying a 1.8 km correlation length
    has under two samples per length, and comparing an interpolated crossing against
    the exact ``a`` would report coarseness as error.
    """
    errors = [
        abs(got - want) / want
        for got, want in zip(delivered, wanted, strict=True)
        if np.isfinite(want) and want > 0.0
    ]
    # Where the target never decorrelates but the delivered field does, the delivered
    # length is the whole of what is wrong and there is no ratio to take.
    if not errors:
        return 0.0 if all(not np.isfinite(got) for got in delivered) else np.inf
    return max(errors)


def _crossing_km(profile: FloatArray, level: float, spacing_km: float) -> float:
    """Where a covariance profile first falls to ``level``, interpolated, in km."""
    below = np.flatnonzero(profile <= level)
    if below.size == 0 or below[0] == 0:
        return np.inf
    crossed = int(below[0])
    above, under = profile[crossed - 1], profile[crossed]
    return spacing_km * (crossed - 1 + (above - level) / (above - under))


def _predicted_extents(
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
    margin: float,
) -> tuple[int, int]:
    """How large the embedding has to be, from the covariance's own decay length.

    The fault, plus ``margin`` correlation lengths so the wrap lands where the
    covariance has faded, never less than :data:`MINIMUM_EMBEDDING` times the fault.
    """
    strike_km, dip_km = spacing_km
    wanted = (
        max(
            MINIMUM_EMBEDDING * cell_counts[0],
            cell_counts[0]
            + int(np.ceil(margin * parameters.correlation_length_dip_km / dip_km)),
        ),
        max(
            MINIMUM_EMBEDDING * cell_counts[1],
            cell_counts[1]
            + int(
                np.ceil(margin * parameters.correlation_length_strike_km / strike_km)
            ),
        ),
    )
    return (int(next_fast_len(wanted[0])), int(next_fast_len(wanted[1])))


# Correlation lengths that change with depth, and the one sampler that covers both


HYBRID_DECAY_LENGTHS = 6.0
"""How many correlation lengths of margin the nonstationary path pads by.

Twice :data:`DECAY_LENGTHS`, because this path does not retry: a second attempt is a
second factorisation, which is the expensive half of the sampler, so the margin is
chosen once and generously. Measured on the shipped crustal example -- whose shallow
branch reaches 36% of the fault, the hardest case a wrap faces -- the variance the
embedding cannot sample falls 3.0e-4, 9.2e-5, 3.1e-5, 9.2e-7 at margins of 3, 4.5, 6
and 9 lengths, so 6 is where it drops below :data:`MAXIMUM_VARIANCE_DEFICIT` without
paying for the last two.
"""

HYBRID_CACHE_ENTRIES = 4
"""How many factorisations to keep.

Each is ``wavenumbers x cells_i x cells_i`` of float64 -- 1.85 GB on the Hikurangi
interface cut at 1 km, against 537 MB for a stationary embedding of the same chart.
Four is two charts' worth at the two marginals a segment draws (slip's, for every
field correlated with slip, and :data:`NORMAL` for rake), so a multi-segment rupture
re-uses within a segment and pays again at the next one rather than holding all of
them.
"""

MAXIMUM_FACTORISATION_CELLS = 1 << 31
"""The largest factorisation to attempt, in cells; past it the draw is refused.

The covariance and its factors are both ``wavenumbers x cells_i x cells_i`` float64
and both live at once, so 2\\ :sup:`31` cells is 17.2 GB for the pair. Far above
:data:`MAXIMUM_EMBEDDING_CELLS`, because this is the sampler you reach for when the
stationary one cannot express the model, and the alternative to spending the memory
is not sampling at all.
"""


@dataclasses.dataclass(frozen=True)
class HybridFilterParameters:
    """Correlation lengths that change with depth, as one nonstationary field.

    Shallow slip is patchier than deep slip: genslip carries that as
    ``hyb_corlen_flag``, two stationary fields drawn at different correlation lengths
    and summed under a depth-dependent weight. This is the same model stated as one
    field whose correlation length varies, which is not the same thing and is the
    thing that was meant.

    The difference is not the marginal variance -- that is a notch a weighting fixes.
    It is that a **blend of two fields is not a field with an intermediate correlation
    length**. Measured on the shipped crustal profile, the best two-field blend
    delivers a correlation length 33% away from the target in the middle of the
    transition, and no choice of weights does better because an intermediate Matern is
    not in the cone spanned by the two endpoints. Eight fields blended with fitted
    per-depth weights close that to 0.7% -- but they reach it by making neighbouring
    depths depend on *different* fields, so the correlation **across** the transition
    collapses: 0.00 where this model gives 0.28, which is to say the shallow and deep
    halves of the fault come out statistically independent. The transition is the only
    place the hybrid model does anything, so a blend gets everything right except the
    part that matters.

    ``shallow`` and ``deep`` are the two ends, ``transition_depth_km`` the middle of
    the ramp and ``transition_half_width_km`` its half-width, both in kilometres and
    both about **depth**, not down-dip distance. Setting the two ends equal is the
    stationary field, and is dispatched to the cheap sampler rather than computed the
    expensive way.

    References
    ----------
    Paciorek, C. J., & Schervish, M. J. (2006). Spatial modelling using a new class of
    nonstationary covariance functions. *Environmetrics*, 17(5), 483-506.

    Stein, M. L. (2005). Nonstationary spatial covariance functions. Technical report,
    University of Chicago.
    """

    shallow: VonKarmanFilterParameters
    deep: VonKarmanFilterParameters
    transition_depth_km: float
    transition_half_width_km: float

    def __post_init__(self) -> None:
        """Refuse a ramp that is not one."""
        if not (self.transition_half_width_km > 0.0) or not np.isfinite(
            self.transition_half_width_km
        ):
            raise ConfigError(
                "transition_half_width_km must be a positive length, got "
                f"{self.transition_half_width_km}; a transition with no width is a "
                "step, and the field either side of it would be two fields"
            )
        if not np.isfinite(self.transition_depth_km):
            raise ConfigError(
                f"transition_depth_km must be finite, got {self.transition_depth_km}"
            )

    @property
    def is_stationary(self) -> bool:
        """Whether both ends are the same, so the field's structure does not vary."""
        return self.shallow == self.deep

    def deep_weight(self, depth_km: FloatArray) -> FloatArray:
        """0 above the ramp, 1 below it, linear across -- clamped at both ends."""
        low = self.transition_depth_km - self.transition_half_width_km
        high = self.transition_depth_km + self.transition_half_width_km
        return np.clip(
            (np.asarray(depth_km, dtype=np.float64) - low) / (high - low), 0.0, 1.0
        )

    def profile(
        self, depth_km: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``(strike, dip, hurst)`` at each depth, one value per dip row.

        Log-linear in the lengths and linear in the Hurst exponent, which is genslip's
        own interpolation: a correlation length is a scale, so the halfway point of a
        transition between 2 km and 14 km is 5.3 km rather than 8 km.
        """
        weight = self.deep_weight(depth_km)

        def between(shallow: float, deep: float) -> FloatArray:
            """One length interpolated across the ramp, geometrically."""
            return np.exp((1.0 - weight) * np.log(shallow) + weight * np.log(deep))

        return (
            between(
                self.shallow.correlation_length_strike_km,
                self.deep.correlation_length_strike_km,
            ),
            between(
                self.shallow.correlation_length_dip_km,
                self.deep.correlation_length_dip_km,
            ),
            (1.0 - weight) * self.shallow.hurst + weight * self.deep.hurst,
        )


type FilterParameters = VonKarmanFilterParameters | HybridFilterParameters
"""What a field's correlation structure may be: the same everywhere, or depth-varying.

One sampler takes either. :class:`HybridFilterParameters` whose two ends agree is the
stationary case and is dispatched as one, so nothing pays for the general path unless
its covariance needs it.
"""


def _matern_correlation(
    normalised_distance: FloatArray, smoothness: FloatArray
) -> FloatArray:
    """:func:`von_karman_correlation` with a smoothness that varies point by point.

    The same function, evaluated in logs because a varying order means the
    :math:`2^{1-H}/\\Gamma(H)` normalisation cannot be lifted out of the array.
    """
    distance = np.asarray(normalised_distance, dtype=np.float64)
    smoothness = np.broadcast_to(
        np.asarray(smoothness, dtype=np.float64), distance.shape
    )
    correlation = np.ones_like(distance)
    away = distance > 0.0
    order, scaled = smoothness[away], distance[away]
    with np.errstate(divide="ignore"):
        correlation[away] = np.exp(
            (1.0 - order) * np.log(2.0)
            - np.log(gamma(order))
            + order * np.log(scaled)
            + np.log(kv(order, scaled))
        )
    return correlation


def _nonstationary_covariance(
    lag_strike_km: FloatArray,
    downdip_km: FloatArray,
    strike_km: FloatArray,
    dip_km: FloatArray,
    hurst: FloatArray,
) -> FloatArray:
    """The covariance between every pair of depths, at every strike lag.

    Paciorek & Schervish (2006) theorem 1 for a length scale that varies, Stein (2005)
    for a smoothness that does. The quadratic form uses the **averaged** length
    tensor, which is what makes the construction positive definite for any smooth
    profile and what makes the marginal variance exactly 1 at every depth -- so the
    NORTA marginal below is a statement about the same field at the surface as at the
    bottom.

    ``lag_strike_km`` is the wrapped lag, ``downdip_km`` the distance down the fault
    (not the depth: correlation is a property of the surface, while the *profile* that
    picks the lengths is a property of depth). Returns ``(lags, rows, rows)``.
    """
    strike_km = np.asarray(strike_km, dtype=np.float64)
    dip_km = np.asarray(dip_km, dtype=np.float64)
    hurst = np.asarray(hurst, dtype=np.float64)

    separation = downdip_km[:, None] - downdip_km[None, :]
    averaged_strike = 0.5 * (strike_km[:, None] ** 2 + strike_km[None, :] ** 2)
    averaged_dip = 0.5 * (dip_km[:, None] ** 2 + dip_km[None, :] ** 2)
    averaged_hurst = 0.5 * (hurst[:, None] + hurst[None, :])

    # The determinant ratio that normalises the averaged tensor back to unit variance.
    scale = np.sqrt(
        np.sqrt(
            (strike_km[:, None] * dip_km[:, None])
            * (strike_km[None, :] * dip_km[None, :])
        )
        ** 2
        / (averaged_strike * averaged_dip)
    )
    if np.ptp(hurst) > 0.0:
        scale = (
            scale
            * gamma(averaged_hurst)
            / np.sqrt(gamma(hurst[:, None]) * gamma(hurst[None, :]))
        )

    distance = np.sqrt(
        lag_strike_km[:, None, None] ** 2 / averaged_strike[None]
        + separation[None] ** 2 / averaged_dip[None]
    )
    if np.ptp(hurst) > 0.0:
        return scale[None] * _matern_correlation(distance, averaged_hurst[None])
    return scale[None] * von_karman_correlation(distance, float(hurst.flat[0]))


@dataclasses.dataclass(frozen=True)
class Factorisation:
    """A nonstationary covariance factorised into one square root per wavenumber."""

    factors: FloatArray
    strike_cells: int
    variance_deficit: float
    correlation_lengths: tuple[float, float]
    correlation_length_error: float


@functools.lru_cache(maxsize=HYBRID_CACHE_ENTRIES)
def _factorise(
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: HybridFilterParameters,
    marginal: Marginal,
    depths_km: tuple[float, ...],
) -> Factorisation:
    """Build this covariance on this grid and factorise it, once per chart.

    ``depths_km`` is one depth per dip row, and is part of the key because the profile
    is a function of it. It is a tuple because a cache key has to be hashable, and its
    length is the number of dip rows.

    Raises
    ------
    ConfigError
        If the factorisation this grid needs is past
        :data:`MAXIMUM_FACTORISATION_CELLS`.

    Warns
    -----
    DegradedCorrelation
        If the covariance is not positive definite, or the strike axis cannot carry
        the correlation lengths asked of it.
    """
    cells_i, cells_j = cell_counts
    strike_spacing_km, dip_spacing_km = spacing_km
    depths = np.asarray(depths_km, dtype=np.float64)
    strike, dip, hurst = parameters.profile(depths)

    wavenumbers = _strike_extent(cells_j, strike_spacing_km, float(strike.max()))
    if wavenumbers * cells_i * cells_i > MAXIMUM_FACTORISATION_CELLS:
        raise CapacityError(
            f"a {cells_i} x {cells_j} grid needs a {wavenumbers} x {cells_i} x "
            f"{cells_i} = {wavenumbers * cells_i * cells_i:,}-cell factorisation, past "
            f"the {MAXIMUM_FACTORISATION_CELLS:,} this machine can hold -- the "
            "covariance across depths is dense, so this sampler costs the square of "
            "the dip resolution where the stationary one costs its logarithm. Either "
            "cut this segment more coarsely down dip, or raise "
            "`sampling.MAXIMUM_FACTORISATION_CELLS`"
        )

    # The covariance is even in the lag, so only the distinct lags are evaluated and
    # the modified Bessel function -- much the most expensive part of the build -- is
    # called half as often.
    distinct = np.arange(wavenumbers // 2 + 1, dtype=np.float64) * strike_spacing_km
    downdip_km = (np.arange(cells_i, dtype=np.float64) + 0.5) * dip_spacing_km
    target = _nonstationary_covariance(distinct, downdip_km, strike, dip, hurst)
    covariance = np.ascontiguousarray(
        latent_correlation(marginal, marginal, target)[_wrapped_lag_index(wavenumbers)]
    )
    del target

    factors, deficit = _kernels.factorise_covariance(covariance)
    del covariance

    delivered, error = _delivered_profile(
        factors, spacing_km, parameters, marginal, strike, hurst
    )
    factorisation = Factorisation(
        factors=factors,
        strike_cells=cells_j,
        variance_deficit=deficit,
        correlation_lengths=delivered,
        correlation_length_error=error,
    )
    _warn_if_factorisation_degraded(cell_counts, spacing_km, factorisation, marginal)
    return factorisation


def _strike_extent(cells_j: int, strike_spacing_km: float, longest_km: float) -> int:
    """How wide the padded strike axis has to be, from the profile's longest length.

    The margin comes off the **longest** correlation length on the profile, which is
    the shallow end: the wrap has to land where the covariance has faded everywhere,
    and a margin that suits the deep end would not. Unlike the stationary embedding
    there is no retry at a larger margin -- a second attempt is a second
    factorisation, which is the expensive half of this sampler -- so the margin is
    :data:`HYBRID_DECAY_LENGTHS` rather than :data:`DECAY_LENGTHS`, chosen once and
    wide enough that the retry is not wanted.
    """
    wanted = max(
        MINIMUM_EMBEDDING * cells_j,
        cells_j + int(np.ceil(HYBRID_DECAY_LENGTHS * longest_km / strike_spacing_km)),
    )
    return int(next_fast_len(wanted))


def _delivered_profile(
    factors: FloatArray,
    spacing_km: tuple[float, float],
    parameters: HybridFilterParameters,
    marginal: Marginal,
    strike: FloatArray,
    hurst: FloatArray,
) -> tuple[tuple[float, float], float]:
    """The along-strike correlation length the field really has, at each depth.

    Only the strike axis can degrade: down dip the factorisation is exact, with no
    wrap to land badly. The diagonal of each wavenumber's spectrum is
    ``sum_j B[k, i, j]^2`` and needs no second pass over the covariance, so this
    measures what was delivered rather than what was asked for. Returns the delivered
    lengths at the shallowest and deepest rows, and the worst relative error over
    every row.
    """
    strike_spacing_km, _ = spacing_km
    power = np.einsum("kij,kij->ki", factors, factors)
    latent = np.fft.ifft(power, axis=0).real
    rows = latent.shape[1]

    errors: list[float] = []
    lengths: list[float] = []
    for row in range(rows):
        profile = latent[:, row]
        if profile[0] <= 0.0:
            errors.append(np.inf)
            lengths.append(np.inf)
            continue
        field = transformed_correlation(marginal, marginal, profile / profile[0])
        half = float(von_karman_correlation(np.array([1.0]), float(hurst[row]))[0])
        delivered = _crossing_km(field, half, strike_spacing_km)
        lengths.append(delivered)
        wanted = float(strike[row])
        if np.isfinite(delivered) and wanted > 0.0:
            errors.append(abs(delivered - wanted) / wanted)
        else:
            errors.append(np.inf)

    return (lengths[0], lengths[-1]), max(errors)


def _warn_if_factorisation_degraded(
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    factorisation: Factorisation,
    marginal: Marginal,
) -> None:
    """Say what the field got instead, when it is not what was asked for."""
    if factorisation.variance_deficit > MAXIMUM_VARIANCE_DEFICIT:
        warnings.warn(
            f"this covariance under a {marginal.family} marginal is not positive "
            f"definite: {factorisation.variance_deficit:.2%} of its variance sits in "
            f"directions that cannot be sampled, past the {MAXIMUM_VARIANCE_DEFICIT:.0e} "
            "that is round-off. They are dropped, which delivers a field with that "
            "much less variance at the shortest wavelengths than was asked for",
            DegradedCorrelation,
            stacklevel=_WARN_STACKLEVEL,
        )
    if factorisation.correlation_length_error > CORRELATION_LENGTH_TOLERANCE:
        length_km = cell_counts[1] * spacing_km[0]
        warnings.warn(
            f"a segment {length_km:.3g} km along strike cannot carry this depth "
            "profile's correlation lengths: the worst row is off by "
            f"{factorisation.correlation_length_error * 100:.0f}%, and the shallowest "
            f"and deepest rows got {factorisation.correlation_lengths[0]:.3g} and "
            f"{factorisation.correlation_lengths[1]:.3g} km. The shallow end of a "
            "hybrid profile is the long one, and it is the one a short segment cannot "
            "hold. Slip, moment and timing are unaffected; what is degraded is how "
            "the slip is distributed",
            DegradedCorrelation,
            stacklevel=_WARN_STACKLEVEL,
        )


def standardise(field: FloatArray) -> FloatArray:
    """Zero mean, unit sample variance.

    The embedding already delivers ``C(0) = 1``; this is what makes the stages'
    ``1 + cov * Z`` exact on each realisation rather than on average.
    """
    spread = float(field.std())
    # A one-cell chart has a single sample and hence no variance; the mesh CLI produces
    # one for any plane shorter than half the requested subfault size.
    if spread == 0.0:
        return np.zeros_like(field)
    return (field - field.mean()) / spread


def von_karman_grid(
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    covariance: VonKarmanFilterParameters,
    rng: np.random.Generator,
    marginal: Marginal = NORMAL,
) -> FloatArray:
    """Draw the **latent** field behind this covariance on a regular grid.

    Always Gaussian, one standard-normal value per cell. ``marginal`` says what
    distribution the field is destined for, and the draw is pre-corrected for it so
    that ``marginal.apply`` of this returns a field with the covariance asked for --
    see :func:`latent_correlation`. It is the caller that applies the marginal,
    because two fields can only be mixed by :func:`correlate_fields` while they are
    still Gaussian.

    At :data:`NORMAL` the pre-correction is the identity and the field is the finished
    one. The embedding is cached on all four arguments. ``cell_counts`` is
    ``(dip, strike)``; ``spacing_km`` is in kilometres and ``(strike, dip)``, the
    opposite order and the one
    :meth:`~rupture_generator.mesh.RuptureMesh.spacing_km` returns.

    Raises
    ------
    ConfigError
        If the embedding this grid needs is past :data:`MAXIMUM_EMBEDDING_CELLS`, or
        the pre-corrected covariance does not embed.

    Warns
    -----
    DegradedCorrelation
        If the grid cannot carry the correlation lengths asked of it.
    """
    embedding = _embed(cell_counts, spacing_km, covariance, marginal)
    seed = int(rng.integers(1 << 63, dtype=np.int64))
    return _kernels.von_karman_draw(embedding.eigenvalues, cell_counts, seed)


def von_karman_field(
    mesh: RuptureMesh,
    covariance: FilterParameters,
    rng: np.random.Generator,
    marginal: Marginal = NORMAL,
) -> FloatArray:
    """Draw the latent field behind this covariance on this chart.

    The one entry point, for either kind of covariance. A
    :class:`HybridFilterParameters` whose two ends agree describes a field whose
    structure does not vary, and is dispatched to :func:`von_karman_grid` -- not as an
    approximation but as the exact factorisation of the general case: a covariance
    that is stationary down dip is Toeplitz, a Toeplitz operator embeds in a circulant
    one, and a circulant operator's eigenvectors are the DFT. The two paths are one
    model, and which one runs is a property of the covariance rather than something a
    caller chooses.

    A profile that really does vary needs the depth of every dip row, which is why
    this takes a chart where :func:`von_karman_grid` takes a grid. On a chart whose
    rows are not level -- a resampled interface -- the row's mean depth is the
    profile's argument, which is exact for a planar chart and the projection's own
    approximation for a curved one.
    """
    if isinstance(covariance, VonKarmanFilterParameters):
        return von_karman_grid(
            mesh.cell_counts, mesh.spacing_km(), covariance, rng, marginal
        )
    if covariance.is_stationary:
        return von_karman_grid(
            mesh.cell_counts, mesh.spacing_km(), covariance.deep, rng, marginal
        )

    depths_km = np.asarray(mesh.centres()[..., 2], dtype=np.float64).mean(axis=1)
    factorisation = _factorise(
        mesh.cell_counts,
        mesh.spacing_km(),
        covariance,
        marginal,
        tuple(float(depth) for depth in depths_km),
    )
    seed = int(rng.integers(1 << 63, dtype=np.int64))
    return _kernels.nonstationary_draw(
        factorisation.factors, factorisation.strike_cells, seed
    )


def correlate_fields(
    field_a: FloatArray, field_b: FloatArray, rho: float
) -> FloatArray:
    """A field correlated at ``rho`` with ``field_a``, on the fault.

    ``rho * A + sqrt(1 - rho^2) * B``: the weights are the cosine and sine of one
    angle, so the result keeps the shared covariance. Both fields must be
    **unstandardised** draws of it -- standardising first divides each by its own
    sample spread, perturbing the relation by the estimator's error.
    """
    if not (-1.0 <= rho <= 1.0):
        raise ConfigError(f"a correlation must be in [-1, 1], got {rho}")

    return rho * field_a + np.sqrt(1.0 - rho * rho) * field_b


__all__ = [
    "CORRELATION_LENGTH_TOLERANCE",
    "HURST",
    "MAI_MAXIMUM_RATIO",
    "MAXIMUM_DOUBLINGS",
    "MAXIMUM_EMBEDDING_CELLS",
    "MINIMUM_EMBEDDING",
    "NORMAL",
    "NORTA_ORDER",
    "SUZUKI_COEFFICIENTS",
    "DegradedCorrelation",
    "Embedding",
    "Factorisation",
    "FilterParameters",
    "HybridFilterParameters",
    "Marginal",
    "MarginalFamily",
    "VonKarmanFilterParameters",
    "attainable_correlation",
    "correlate_fields",
    "correlation_lengths",
    "latent_correlation",
    "standardise",
    "transformed_correlation",
    "von_karman_correlation",
    "von_karman_field",
    "von_karman_grid",
]
