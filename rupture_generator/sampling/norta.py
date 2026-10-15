"""Generate joint distributions with the normal to anything (NORTA) framework.

NORTA is a framework for generating random vectors from an arbitrary joint
distribution introduced in Cario & Nelson (1997). The general idea to generate a
random vector X = (X_1, X_2, ..., X_N), where X_i ~ F_i and Corr[X] = Sigma_X is
as follows:

1. Sample a multivariate random vector Z = (Z_1, Z_2, ..., Z_k) with correlation
matrix Sigma_Z. The vector Z is the "latent vector" and its distribution the
"latent space". The matrix Sigma_Z is the "latent correlation",
2. Apply the transform X_i = F_i^-1(Phi(Z_i)). The vector X_i is the "marginal
vector" and its distribution the "marginal distribution". The matrix Sigma_X is the "marginal correlation".

A theorem in statistics states that X_i then has the distribution given by the CDF
F_i. The map F = (F_i) doesn't preserve the correlation between the Z_i. The
algorithms in this module derive the Sigma_Z such that X = F(Z) has correlation
matrix Sigma_X.
"""

import dataclasses
import functools
from typing import Any, Literal, get_args

import numpy as np
from scipy.optimize import brentq
from scipy.special import factorial, log_ndtr
from scipy.stats import norm, truncexpon, truncnorm

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.sampling.field import Correlation

NORTA_ORDER = 20
"""Hermite terms in a marginal's expansion; for the default slip and rise-time
marginals the truncated series recovers the variance to six digits, so truncation is
not what limits the accuracy."""

NORTA_FACTORIALS = np.asarray(factorial(np.arange(NORTA_ORDER + 1)), np.float64)
"""``k!`` for ``k = 0, ..., NORTA_ORDER``, the squared norm of each ``He_k``."""

NORTA_QUADRATURE_POINTS = 160
"""Gauss-Hermite nodes, exact to degree 319 against a top term of degree 20."""

NORTA_TAIL = 1.0e-12
"""How far into the tails the quantile function is evaluated: ``Phi`` rounds to 0
and 1 at the outermost nodes and an unbounded marginal answers those with infinity."""

NORTA_TRANSFORM_POINTS = 20001
"""Points ``F^-1(Phi(z))`` is tabulated on between the tails; read back linearly,
every marginal agrees with the direct quantile to 2e-6, relative."""

NORTA_INVERSE_POINTS = 20001
"""Points ``g`` is tabulated on for inversion. The round trip measures 2e-10."""

NORTA_CORRELATION_SLACK = 1.0e-9
"""How far past an attainable correlation a target may sit and still be round-off."""

TRUNCATED_NORMAL_MAXIMUM_COV = 1.0
"""A unit-mean truncated normal approaches this spread as the cut recedes to
``-inf`` and never reaches it."""

TRUNCATED_EXPONENTIAL_MINIMUM_COV = 1.0 / np.sqrt(3.0)
"""A unit-mean truncated exponential flattens to a uniform on ``[0, 2]`` as the cut
goes to zero, and its spread stops falling at ``1/sqrt(3)``."""

TRUNCATED_EXPONENTIAL_MAXIMUM_COV = 1.0
"""A unit-mean truncated exponential becomes the whole exponential as the cut goes
to infinity, whose spread equals its mean."""

DOUBLY_TRUNCATED_NORMAL_TAIL = 40.0
"""How many parent standard deviations the nearer cut may sit from the parent mean.
SciPy's truncated-normal moments lose precision past about 80."""

DOUBLY_TRUNCATED_NORMAL_WIDEST = 20.0
"""The widest parent scale, in maxima. The spread converges as ``(maximum/scale)^2``
to its flat-parent limit: at a maximum of 2 it then sits 2e-5 below the uniform's
``1/sqrt(3)``."""

type MarginalFamily = Literal[
    "normal",
    "truncated_normal",
    "doubly_truncated_normal",
    "truncated_exponential",
]


@dataclasses.dataclass(frozen=True)
class Marginal:
    """The distribution one field's values follow.

    ``normal`` is the standard normal and the identity transform. The other families
    are unit-mean, with ``coefficient_of_variation`` their spread.
    ``truncated_normal`` lies on the positive half-line, and ``truncated_exponential`` is the
    family Thingbaijam & Mai (2016) fitted to SRCMOD slip, whose cut also fixes the
    largest value on the fault. ``doubly_truncated_normal`` lies on
    ``[0, maximum]``, so it takes its largest value separately from its spread.

    A ``truncated_normal`` also takes a lower cut, ``minimum``. The fit redoes the
    family on ``[minimum, inf)`` rather than clipping it. The cut family
    stays unit-mean with the stated spread, and no value falls under the cut. The
    quantile transform spreads the probability the parent had under the cut over the
    whole distribution, and no value piles up at the cut. A truncated normal cut at
    ``minimum`` is the zero-truncated normal of spread
    ``coefficient_of_variation / (1 - minimum)``, scaled by ``1 - minimum`` and
    shifted up by ``minimum``. It reaches any spread below ``1 - minimum``.

    Every expensive function in this module caches its results by marginal, so
    instances are immutable and hashable.

    Attributes
    ----------
    family : MarginalFamily
        Which distribution the values follow.
    coefficient_of_variation : float
        The spread of a unit-mean family, its standard deviation over its mean.
        Unused by ``normal``.
    maximum : float
        The largest value, in means, of a ``doubly_truncated_normal``. Every other
        family leaves it infinite.
    minimum : float
        The smallest value, in means, of a ``truncated_normal``, in ``[0, 1)``.
        Zero, the default, leaves the family uncut, and every other family takes zero
        alone.
    """

    family: MarginalFamily = "normal"
    coefficient_of_variation: float = 0.0
    maximum: float = np.inf
    minimum: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a family and spread that no distribution in the family attains."""
        families = get_args(MarginalFamily.__value__)
        if self.family not in families:
            raise RuptureGeneratorError(
                f"there is no {self.family!r} marginal; the families are "
                f"{', '.join(families)}"
            )
        if self.family != "doubly_truncated_normal" and self.maximum != np.inf:
            raise RuptureGeneratorError(
                f"only a doubly_truncated_normal marginal takes a maximum; a "
                f"{self.family} was given {self.maximum}"
            )
        if not 0.0 <= self.minimum < 1.0:
            raise RuptureGeneratorError(
                f"a minimum is a fraction of the unit mean, so it lies in [0, 1); got "
                f"{self.minimum}"
            )
        if self.minimum > 0.0 and self.family != "truncated_normal":
            raise RuptureGeneratorError(
                f"only a truncated_normal marginal takes a minimum; a "
                f"{self.family} was given {self.minimum}"
            )
        if self.family == "normal":
            return
        value = self.coefficient_of_variation
        if not (value > 0.0) or not np.isfinite(value):
            raise RuptureGeneratorError(
                f"a {self.family} marginal needs a positive coefficient of variation, "
                f"got {value}"
            )
        if self.family == "truncated_normal" and self.minimum > 0.0:
            ceiling = TRUNCATED_NORMAL_MAXIMUM_COV * (1.0 - self.minimum)
            if value >= ceiling:
                raise RuptureGeneratorError(
                    f"a unit-mean truncated normal cut at {self.minimum} has a "
                    f"coefficient of variation below {ceiling:.4f}, where it becomes "
                    f"an exponential above the cut; got {value}"
                )
        if self.family == "truncated_normal" and value >= TRUNCATED_NORMAL_MAXIMUM_COV:
            raise RuptureGeneratorError(
                f"a unit-mean truncated normal cannot have a coefficient of variation "
                f"of {value}: the family stays below "
                f"{TRUNCATED_NORMAL_MAXIMUM_COV}"
            )
        if self.family == "truncated_exponential" and not (
            TRUNCATED_EXPONENTIAL_MINIMUM_COV
            < value
            < TRUNCATED_EXPONENTIAL_MAXIMUM_COV
        ):
            raise RuptureGeneratorError(
                f"a unit-mean truncated exponential cannot have a coefficient of "
                f"variation of {value}: the family runs over "
                f"({TRUNCATED_EXPONENTIAL_MINIMUM_COV:.4f}, 1) and attains neither end"
            )
        if self.family == "doubly_truncated_normal":
            if not (1.0 < self.maximum < np.inf):
                raise RuptureGeneratorError(
                    f"a unit-mean doubly truncated normal needs a finite maximum above "
                    f"its mean of 1, got {self.maximum}"
                )
            ceiling = _doubly_truncated_normal_ceiling(self.maximum)
            if value >= ceiling:
                raise RuptureGeneratorError(
                    f"a unit-mean doubly truncated normal with a maximum of "
                    f"{self.maximum} has a coefficient of variation below "
                    f"{ceiling:.4f}, where it all but becomes the truncated "
                    f"exponential with that maximum; got {value}"
                )

    @property
    def is_normal(self) -> bool:
        """bool: Whether this marginal is the identity transform."""
        return self.family == "normal"

    def apply(self, latent: np.ndarray) -> np.ndarray:
        """Give a standard-normal field this marginal, cell by cell.

        The transform is ``F^-1(Phi(latent))``, read off a table rather than through
        the quantile function, which costs seconds per million cells.
        Beyond the tails the table clamps, which is the same as clipping ``Phi`` to
        :data:`NORTA_TAIL`.

        Parameters
        ----------
        latent : np.ndarray
            Standard-normal values, of any shape.

        Returns
        -------
        np.ndarray
            The values transformed to this marginal, shaped like ``latent``.
        """
        if self.is_normal:
            return np.asarray(latent, dtype=np.float64)
        latents, values = _transform_table(self)
        return np.interp(latent, latents, values)

    def power_mean(self, power: float) -> float:
        """The mean of a value raised to ``power``, ``E[X^power]``.

        Gauss-Hermite quadrature integrates the quantile function against the standard
        normal. It skips the caches :meth:`apply` fills, and a caller searching over
        many marginals leaves them to the marginals it draws with.

        Parameters
        ----------
        power : float
            The exponent.

        Returns
        -------
        float
            ``E[X^power]``, which is 1 at ``power`` 0 or 1.

        Raises
        ------
        RuptureGeneratorError
            If the marginal is the standard normal, whose values have no real
            fractional power.
        """
        if self.is_normal:
            raise RuptureGeneratorError(
                "a normal marginal's values can be negative, so they have no power mean"
            )
        quantiles, weights = _quadrature_quantiles()
        values = np.asarray(_fit(self).ppf(quantiles), dtype=np.float64)
        return float(np.sum(weights * values**power))


@functools.cache
def _quadrature_quantiles() -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes as probabilities inside the tails, and normalised weights."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(NORTA_QUADRATURE_POINTS)
    quantiles = np.clip(norm.cdf(nodes), NORTA_TAIL, 1.0 - NORTA_TAIL)
    return quantiles, weights / np.sqrt(2.0 * np.pi)


NORMAL = Marginal()
"""The standard normal: every transform and pre-correction here is the identity."""


def _truncated_exponential_spread(cut: float) -> float:
    """Coefficient of variation of a unit exponential cut at ``cut`` decays.

    Over a common denominator rather than ``truncexpon.std() / .mean()``, whose
    moments cancel to noise below a cut of 0.01 and to a NaN below 1e-5.
    """
    grown = np.expm1(cut)
    return np.sqrt(grown * grown - cut * cut * grown - cut * cut) / (grown - cut)


def _doubly_truncated_normal(location: float, scale: float, maximum: float) -> Any:
    """A normal of ``location`` and ``scale`` cut to ``[0, maximum]``."""
    return truncnorm(
        a=-location / scale,
        b=(maximum - location) / scale,
        loc=location,
        scale=scale,
    )


def _doubly_truncated_normal_location(scale: float, maximum: float) -> float:
    """The parent mean that puts the cut normal's mean at 1.

    The cut normal's mean rises with the parent's, so a bracket grown outward from 1
    finds it. Growing it, not guessing it wide, keeps SciPy out of the far tails.
    """

    def excess(location: float) -> float:
        with np.errstate(invalid="ignore"):
            mean = _doubly_truncated_normal(location, scale, maximum).mean()
        return float(mean) - 1.0

    lower, upper, step = 1.0, 1.0, scale
    while excess(lower) > 0.0:
        lower -= step
        step *= 2.0
    step = scale
    while excess(upper) < 0.0:
        upper += step
        step *= 2.0
    return brentq(excess, lower, upper, xtol=1.0e-14, rtol=1.0e-14)


def _doubly_truncated_normal_spread(scale: float, maximum: float) -> float:
    """Coefficient of variation of the unit-mean cut normal with parent ``scale``.

    It rises with ``scale``, from a narrow normal around 1 toward the truncated
    exponential on ``[0, maximum]`` as the parent flattens.
    """
    location = _doubly_truncated_normal_location(scale, maximum)
    # SciPy computes the skewness alongside and warns when it underflows.
    with np.errstate(invalid="ignore"):
        mean, variance = _doubly_truncated_normal(location, scale, maximum).stats("mv")
    return float(np.sqrt(variance) / mean)


@functools.lru_cache(maxsize=16)
def _doubly_truncated_normal_widest(maximum: float) -> float:
    """The widest parent scale the fit searches.

    Doubling from ``maximum``, the last scale within both
    :data:`DOUBLY_TRUNCATED_NORMAL_WIDEST` maxima and a nearer cut
    :data:`DOUBLY_TRUNCATED_NORMAL_TAIL` parent deviations from the parent mean. Near
    a maximum of 2 the parent mean stays near 1 however wide it grows, so the tail
    alone never stops the doubling.
    """
    scale = maximum
    while (wider := 2.0 * scale) <= DOUBLY_TRUNCATED_NORMAL_WIDEST * maximum:
        location = _doubly_truncated_normal_location(wider, maximum)
        if abs(location) / wider > DOUBLY_TRUNCATED_NORMAL_TAIL:
            break
        scale = wider
    return scale


@functools.lru_cache(maxsize=16)
def _doubly_truncated_normal_ceiling(maximum: float) -> float:
    """The largest spread the fit can give ``maximum``, at the widest parent."""
    return _doubly_truncated_normal_spread(
        _doubly_truncated_normal_widest(maximum), maximum
    )


def _zero_truncated_normal_moments(alpha: float) -> tuple[float, float]:
    """Mean and coefficient of variation of ``N(alpha, 1)`` cut below at zero.

    In closed form, with the inverse Mills ratio ``phi(alpha) / Phi(alpha)`` taken in
    logarithms so that it holds far into the lower tail.
    """
    ratio = float(np.exp(norm.logpdf(alpha) - log_ndtr(alpha)))
    mean = alpha + ratio
    return mean, float(np.sqrt(max(1.0 - ratio * mean, 0.0))) / mean


def _cut_truncated_normal(spread: float, minimum: float) -> Any:
    """The unit-mean normal cut at ``minimum`` with coefficient of variation ``spread``.

    Less the minimum, a normal cut at the minimum is a normal cut at zero, with mean
    ``1 - minimum`` and the same standard deviation, so spread
    ``spread / (1 - minimum)``. The fit finds that one as it does the uncut family,
    then scales it by ``1 - minimum`` and shifts it up by ``minimum``. The moments
    come in closed form rather than from SciPy's, which cost a hundred times as
    much. A search over cuts then stays quick.
    """
    relative = spread / (1.0 - minimum)
    alpha = brentq(
        lambda value: _zero_truncated_normal_moments(value)[1] - relative,
        -40.0,
        60.0,
        xtol=1.0e-13,
    )
    mean = _zero_truncated_normal_moments(alpha)[0]
    return truncnorm(
        a=-alpha,
        b=np.inf,
        loc=minimum + (1.0 - minimum) * alpha / mean,
        scale=(1.0 - minimum) / mean,
    )


def _fit(marginal: Marginal) -> Any:
    """Map from marginal distribution to scipy distribution class.

    Each distribution's shape gives the target coefficient of variation, and its
    support starts at the marginal's minimum. Uncached: :func:`_distribution` is the
    cached form.
    """
    spread = marginal.coefficient_of_variation
    # A marginal's parameters are its mean and coefficient of variation. The truncated
    # exponential and truncated normal invert by optimisation. A minimum takes its own
    # branch, and a truncated normal without one keeps the fit it always had.
    match marginal.family:
        case "normal":
            return norm(0.0, 1.0)  # TODO: Should this not be also a function of spread?
        case "truncated_exponential":
            cut = brentq(
                lambda value: _truncated_exponential_spread(value) - spread,
                5.0e-4,
                36.0,
                xtol=1.0e-13,
            )
            mean = float(truncexpon(b=cut, loc=0.0, scale=1.0).mean())
            return truncexpon(b=cut, loc=0.0, scale=1.0 / mean)
        case "truncated_normal" if marginal.minimum > 0.0:
            return _cut_truncated_normal(spread, marginal.minimum)
        case "truncated_normal":

            def spread_at(alpha: float) -> float:
                shifted = truncnorm(a=-alpha, b=np.inf, loc=alpha, scale=1.0)
                return float(shifted.std() / shifted.mean())

            alpha = brentq(
                lambda value: spread_at(value) - spread, -40.0, 60.0, xtol=1.0e-13
            )
            mean = float(truncnorm(a=-alpha, b=np.inf, loc=alpha, scale=1.0).mean())
            return truncnorm(a=-alpha, b=np.inf, loc=alpha / mean, scale=1.0 / mean)
        case "doubly_truncated_normal":
            # Cutting a normal only narrows it, so a parent scale of half the target
            # spread gives less than the target.
            maximum = marginal.maximum
            scale = brentq(
                lambda value: _doubly_truncated_normal_spread(value, maximum) - spread,
                0.5 * spread,
                _doubly_truncated_normal_widest(maximum),
                xtol=1.0e-13,
            )
            location = _doubly_truncated_normal_location(scale, maximum)
            return _doubly_truncated_normal(location, scale, maximum)
        case unknown:
            raise ValueError(f"Unimplementhed marginal distribution: {unknown}")


_distribution = functools.lru_cache(maxsize=16)(_fit)
"""The fitted distribution of a marginal, cached by marginal."""


@functools.lru_cache(maxsize=16)
def _transform_table(marginal: Marginal) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform table for z ~ N(0, 1.0) -> marginal distribution. Relies on inverse distribution sampling.
    """
    edge = float(norm.isf(NORTA_TAIL))
    latents = np.linspace(-edge, edge, NORTA_TRANSFORM_POINTS)
    values = _distribution(marginal).ppf(norm.cdf(latents))
    return latents, np.asarray(values, dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class HermiteExpansion:
    """A marginal's NORTA transform ``h`` in the probabilists' Hermite basis.

    ``h(z) = sum_k coefficients[k] He_k(z)``, truncated at :data:`NORTA_ORDER`.

    Attributes
    ----------
    coefficients : np.ndarray
        ``a_k = E[h(Z) He_k(Z)] / k!`` for ``k = 0, ..., NORTA_ORDER``. ``a_0`` is the
        mean of ``h(Z)``.
    spread : float
        ``sqrt(sum_{k>=1} a_k^2 k!)``, the standard deviation of ``h(Z)``
        (Parseval).
    """

    coefficients: np.ndarray
    spread: float


@functools.lru_cache(maxsize=16)
def _hermite_expansion(marginal: Marginal) -> HermiteExpansion:
    """Hermite coefficients of a marginal's NORTA transform, for correcting correlation.

    Repeated theme of this module: two jointly standard-normal latents Z_1, Z_2 with
    correlation rho map to marginal values h_1(Z_1), h_2(Z_2), where
    h(z) = F^{-1}(Phi(z)), whose correlation is some g(rho). Computing g(rho)
    directly is a 2D integral over the bivariate normal, and hitting a target
    correlation means a root-find over rho with that integral in the inner loop.

    Instead expand h in the probabilists' Hermite polynomials He_k, truncated at
    NORTA_ORDER terms. By Mehler's formula,

        E[He_j(Z_1) He_k(Z_2)] = delta_jk * factorial(k) * rho^k,

    so in cov(h_1(Z_1), h_2(Z_2)) the cross terms vanish, the k = 0 term (the
    product of the means) cancels, and g reduces to a power series in rho:

        g(rho) = sum_{k>=1} a_k b_k factorial(k) rho^k / (sigma_1 sigma_2),

    with a_k, b_k the coefficients of h_1, h_2 and sigma the standard deviations of
    h_1(Z_1), h_2(Z_2). See Grigoriu (1995), Applied Non-Gaussian Processes.

    This function computes one marginal's half, a_k = E[h(Z) He_k(Z)] / k!, by
    Gauss-Hermite quadrature.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(NORTA_QUADRATURE_POINTS)
    weights = weights / np.sqrt(2.0 * np.pi)
    transformed = marginal.apply(nodes)

    basis = np.polynomial.hermite_e.hermevander(nodes, NORTA_ORDER)
    coefficients = (weights * transformed) @ basis / NORTA_FACTORIALS
    spread = float(np.sqrt(np.sum(coefficients[1:] ** 2 * NORTA_FACTORIALS[1:])))
    return HermiteExpansion(coefficients, spread)


@functools.lru_cache(maxsize=64)
def _correlation_series(first: Marginal, second: Marginal) -> np.ndarray:
    """Power-series coefficients of g, the map from latent to marginal correlation.

    With h_1 the transform of ``first`` and h_2 of ``second``, and a_k, b_k and
    sigma_1, sigma_2 their :class:`HermiteExpansion` coefficients and spreads,

        g(rho) = sum_{k>=1} a_k b_k factorial(k) rho^k / (sigma_1 sigma_2).

    Entry k - 1 of the result is the coefficient of rho^k. The series has no constant
    term.
    """
    left = _hermite_expansion(first)
    right = _hermite_expansion(second)
    return (
        left.coefficients[1:]
        * right.coefficients[1:]
        * NORTA_FACTORIALS[1:]
        / (left.spread * right.spread)
    )


def latent_correlation(
    first: Marginal, second: Marginal, target: np.ndarray
) -> np.ndarray:
    """Compute the latent correlation that gives ``target`` as the marginal correlation.

    This is g^-1, for g the power series from :func:`_correlation_series`. The
    function tabulates g on :data:`NORTA_INVERSE_POINTS` latent correlations in
    [-1, 1] and reads the table backwards, so g must be increasing there.

    Parameters
    ----------
    first : Marginal
        The first field's marginal.
    second : Marginal
        The second field's marginal.
    target : np.ndarray
        The target marginal correlations between the two fields, of any shape.

    Returns
    -------
    np.ndarray
        The latent correlations rho with g(rho) = ``target``, shaped like it.

    Raises
    ------
    RuptureGeneratorError
        If g isn't increasing on [-1, 1], or if ``target`` lies outside
        [g(-1), g(1)], the marginal correlations the two marginals can share under a
        Gaussian copula, by more than :data:`NORTA_CORRELATION_SLACK`.
    """
    if first.is_normal and second.is_normal:
        return np.asarray(target, dtype=np.float64)

    coefficients = _correlation_series(first, second)
    grid = np.linspace(-1.0, 1.0, NORTA_INVERSE_POINTS)
    # g has no constant term, so a latent correlation of 0 gives a marginal one of 0.
    delivered = np.polynomial.polynomial.polyval(grid, np.r_[0.0, coefficients])

    # For one latent correlation alone to give the marginals ``target`` marginal
    # correlation, g must be one-to-one on [-1, 1], and so strictly monotonic.
    # Checking that it strictly increases suffices, because by Price's theorem g is
    # non-decreasing. np.interp needs it too.
    if not np.all(np.diff(delivered) > 0.0):
        raise RuptureGeneratorError(
            f"the correlation map between a {first.family} and a {second.family} "
            "marginal is not increasing, so it cannot be inverted"
        )

    target = np.asarray(target, dtype=np.float64)
    floor, ceiling = float(delivered[0]), float(delivered[-1])
    worst = max(float(target.max()) - ceiling, floor - float(target.min()))
    if worst > NORTA_CORRELATION_SLACK:
        raise RuptureGeneratorError(
            f"a {first.family} and a {second.family} marginal can be correlated "
            f"between {floor:.4f} and {ceiling:.4f} under a Gaussian copula, and "
            f"{float(target.min()):.4f} to {float(target.max()):.4f} was asked for"
        )
    return np.interp(np.clip(target, floor, ceiling), delivered, grid)


@dataclasses.dataclass(frozen=True)
class PreCorrected:
    """A correlation function pre-corrected so a marginal delivers ``target``.

    It applies ``g^-1`` pointwise. Ask the sampler for this and, after
    :meth:`Marginal.apply`, the field's correlation function is ``target``. It's a
    :class:`~rupture_generator.sampling.field.Correlation` like any other, so the
    sampler can't tell it from the uncorrected one. For ``NORMAL`` it's the identity.

    Attributes
    ----------
    target : Correlation
        The correlation function the transformed field should have.
    marginal : Marginal
        The marginal the field takes after the transform.
    """

    target: Correlation
    marginal: Marginal

    def __call__(self, lag: np.ndarray) -> np.ndarray:
        """Evaluate the latent correlation at a lag in correlation lengths.

        Parameters
        ----------
        lag : np.ndarray
            Lags in correlation lengths, of any shape.

        Returns
        -------
        np.ndarray
            The latent correlation at each lag.
        """
        return latent_correlation(self.marginal, self.marginal, self.target(lag))
