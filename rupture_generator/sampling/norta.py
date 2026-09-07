"""NORTA: giving a Gaussian field a marginal, and paying for it in correlation.

Cario & Nelson (1997). A field is drawn Gaussian and pushed through
``F^-1(Phi(.))``, which gives every cell the requested distribution exactly. The map
is monotone, so it preserves which patch is the large one, but it is nonlinear, so
it shrinks every correlation strictly between 0 and 1. :func:`latent_correlation`
inverts that shrinkage, and :class:`PreCorrected` applies the inverse to a whole
correlation function so the field comes out with the correlation lengths it was
asked for. The sampler never sees any of this: it embeds whatever correlation
function it is handed.
"""

import dataclasses
import functools
from typing import Any, Literal

import numpy as np
from scipy.optimize import brentq
from scipy.special import factorial
from scipy.stats import gamma as gamma_distribution
from scipy.stats import norm, truncexpon, truncnorm

from rupture_generator.sampling.field import Correlation, SamplingError

NORTA_ORDER = 20
"""Hermite terms in a marginal's expansion; both production marginals sum to
1.000000 here, so truncation is not what limits the accuracy."""

NORTA_QUADRATURE_POINTS = 160
"""Gauss-Hermite nodes, exact to degree 319 against a top term of degree 20."""

NORTA_TAIL = 1.0e-12
"""How far into the tails the quantile function is evaluated: ``Phi`` rounds to 0
and 1 at the outermost nodes and an unbounded marginal answers those with infinity."""

NORTA_INVERSE_POINTS = 20001
"""Points ``g`` is tabulated on for inversion; the round trip measures 2e-10."""

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

type MarginalFamily = Literal[
    "normal", "truncated_normal", "truncated_exponential", "gamma"
]


@dataclasses.dataclass(frozen=True)
class Marginal:
    """The distribution one field's values follow.

    ``normal`` is the standard normal and the identity transform. The other three are
    unit-mean, with ``coefficient_of_variation`` their spread: ``truncated_normal``
    and ``gamma`` on the positive half-line, and ``truncated_exponential`` the family
    Thingbaijam & Mai (2016) fitted to SRCMOD slip, whose cut also fixes the largest
    value on the fault. Frozen and hashable: every expensive function here is cached
    on it.
    """

    family: MarginalFamily = "normal"
    coefficient_of_variation: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a marginal no distribution answers to."""
        if self.family == "normal":
            return
        value = self.coefficient_of_variation
        if not (value > 0.0) or not np.isfinite(value):
            raise SamplingError(
                f"a {self.family} marginal needs a positive coefficient of variation, "
                f"got {value}"
            )
        if self.family == "truncated_normal" and value >= TRUNCATED_NORMAL_MAXIMUM_COV:
            raise SamplingError(
                f"a unit-mean truncated normal cannot have a coefficient of variation "
                f"of {value}; use a gamma marginal"
            )
        if self.family == "truncated_exponential" and not (
            TRUNCATED_EXPONENTIAL_MINIMUM_COV
            < value
            < TRUNCATED_EXPONENTIAL_MAXIMUM_COV
        ):
            raise SamplingError(
                f"a unit-mean truncated exponential cannot have a coefficient of "
                f"variation of {value}: the family runs over "
                f"({TRUNCATED_EXPONENTIAL_MINIMUM_COV:.4f}, 1) and attains neither end"
            )

    @property
    def is_normal(self) -> bool:
        """Whether this marginal is the identity transform."""
        return self.family == "normal"

    @property
    def is_positive(self) -> bool:
        """Whether the support excludes negative values."""
        return self.family in ("truncated_normal", "truncated_exponential", "gamma")

    def apply(self, latent: np.ndarray) -> np.ndarray:
        """``F^-1(Phi(latent))``: give a standard-normal field this marginal, cellwise."""
        if self.is_normal:
            return np.asarray(latent, dtype=np.float64)
        probability = np.clip(norm.cdf(latent), NORTA_TAIL, 1.0 - NORTA_TAIL)
        return np.asarray(_distribution(self).ppf(probability), dtype=np.float64)


NORMAL = Marginal()
"""The standard normal: every transform and pre-correction here is the identity."""


def _truncated_exponential_spread(cut: float) -> float:
    """Coefficient of variation of a unit exponential cut at ``cut`` decays.

    Over a common denominator rather than ``truncexpon.std() / .mean()``, whose
    moments cancel to noise below a cut of 0.01 and to a NaN below 1e-5.
    """
    grown = np.expm1(cut)
    return np.sqrt(grown * grown - cut * cut * grown - cut * cut) / (grown - cut)


@functools.lru_cache(maxsize=16)
def _distribution(marginal: Marginal) -> Any:
    """The frozen SciPy distribution a marginal names, fitted to unit mean.

    A gamma inverts in closed form. The two truncated families separate: the spread
    depends on one shape parameter alone, so a bracketed root-find fixes the shape
    and a division fixes the mean.
    """
    if marginal.is_normal:
        return norm(0.0, 1.0)

    spread = marginal.coefficient_of_variation
    if marginal.family == "gamma":
        shape = 1.0 / (spread * spread)
        return gamma_distribution(a=shape, scale=1.0 / shape)

    if marginal.family == "truncated_exponential":
        cut = brentq(
            lambda value: _truncated_exponential_spread(value) - spread,
            5.0e-4,
            36.0,
            xtol=1.0e-13,
        )
        mean = float(truncexpon(b=cut, loc=0.0, scale=1.0).mean())
        return truncexpon(b=cut, loc=0.0, scale=1.0 / mean)

    def spread_at(alpha: float) -> float:
        shifted = truncnorm(a=-alpha, b=np.inf, loc=alpha, scale=1.0)
        return float(shifted.std() / shifted.mean())

    alpha = brentq(lambda value: spread_at(value) - spread, -40.0, 60.0, xtol=1.0e-13)
    mean = float(truncnorm(a=-alpha, b=np.inf, loc=alpha, scale=1.0).mean())
    return truncnorm(a=-alpha, b=np.inf, loc=alpha / mean, scale=1.0 / mean)


@functools.lru_cache(maxsize=16)
def _hermite_coefficients(marginal: Marginal) -> tuple[np.ndarray, float]:
    """A marginal's probabilists' Hermite coefficients, and the spread they imply.

    ``h(z) = sum_k a_k He_k(z)`` with ``a_k = E[h(Z) He_k(Z)] / k!`` by Gauss-Hermite
    quadrature. Returns the coefficients and ``sqrt(sum_{k>=1} a_k^2 k!)``, the
    transformed field's standard deviation.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(NORTA_QUADRATURE_POINTS)
    weights = weights / np.sqrt(2.0 * np.pi)
    transformed = marginal.apply(nodes)

    orders = np.arange(NORTA_ORDER + 1)
    factorials = np.asarray(factorial(orders), np.float64)
    basis = np.polynomial.hermite_e.hermevander(nodes, NORTA_ORDER)
    coefficients = (weights * transformed) @ basis / factorials
    return coefficients, float(np.sqrt(np.sum(coefficients[1:] ** 2 * factorials[1:])))


@functools.lru_cache(maxsize=64)
def _correlation_series(first: Marginal, second: Marginal) -> np.ndarray:
    """Coefficients of ``g_fg``, the map from latent to delivered correlation.

    ``g_fg(rho) = sum_{k>=1} a^f_k a^g_k k! / (sigma_f sigma_g) rho^k``. The auto
    series is increasing and sums to exactly 1; between different marginals it sums
    to ``g_fg(1) <= 1``, the most they can be correlated.
    """
    left, left_spread = _hermite_coefficients(first)
    right, right_spread = _hermite_coefficients(second)
    orders = np.arange(1, NORTA_ORDER + 1)
    factorials = np.asarray(factorial(orders), np.float64)
    return left[1:] * right[1:] * factorials / (left_spread * right_spread)


def _evaluate_series(coefficients: np.ndarray, latent: np.ndarray) -> np.ndarray:
    latent = np.asarray(latent, dtype=np.float64)
    delivered = np.zeros_like(latent)
    for coefficient in coefficients[::-1]:
        delivered = (delivered + coefficient) * latent
    return delivered


def transformed_correlation(
    first: Marginal, second: Marginal, latent: np.ndarray
) -> np.ndarray:
    """What correlation two NORTA fields have, given their latents' correlation."""
    if first.is_normal and second.is_normal:
        return np.asarray(latent, dtype=np.float64)
    return _evaluate_series(_correlation_series(first, second), latent)


def latent_correlation(
    first: Marginal, second: Marginal, target: np.ndarray
) -> np.ndarray:
    """What correlation to ask the sampler for, to deliver ``target`` on the fields.

    ``g_fg^-1``, tabulated on :data:`NORTA_INVERSE_POINTS` and read backwards.

    Raises
    ------
    SamplingError
        If the target sits outside the correlations the two marginals can share
        under a Gaussian copula by more than :data:`NORTA_CORRELATION_SLACK`.
    """
    if first.is_normal and second.is_normal:
        return np.asarray(target, dtype=np.float64)

    coefficients = _correlation_series(first, second)
    grid = np.linspace(-1.0, 1.0, NORTA_INVERSE_POINTS)
    delivered = _evaluate_series(coefficients, grid)
    if not np.all(np.diff(delivered) > 0.0):
        raise SamplingError(
            f"the correlation map between a {first.family} and a {second.family} "
            "marginal is not increasing, so it cannot be inverted"
        )

    target = np.asarray(target, dtype=np.float64)
    floor, ceiling = float(delivered[0]), float(delivered[-1])
    worst = max(float(target.max()) - ceiling, floor - float(target.min()))
    if worst > NORTA_CORRELATION_SLACK:
        raise SamplingError(
            f"a {first.family} and a {second.family} marginal can be correlated "
            f"between {floor:.4f} and {ceiling:.4f} under a Gaussian copula, and "
            f"{float(target.min()):.4f} to {float(target.max()):.4f} was asked for"
        )
    return np.interp(np.clip(target, floor, ceiling), delivered, grid)


def attainable_correlation(first: Marginal, second: Marginal) -> float:
    """The largest correlation two marginals can share, ``g_fg(1)``."""
    return float(np.sum(_correlation_series(first, second)))


@dataclasses.dataclass(frozen=True)
class PreCorrected:
    """A correlation function pre-corrected so a marginal delivers ``target``.

    ``g^-1`` applied pointwise: ask the sampler for this and, after
    :meth:`Marginal.apply`, the field's correlation function is ``target``. A
    :class:`~rupture_generator.sampling.field.Correlation` like any other, so the
    sampler cannot tell it from the uncorrected one. The identity for ``NORMAL``.
    """

    target: Correlation
    marginal: Marginal

    def __call__(self, lag: np.ndarray) -> np.ndarray:
        """Evaluate the latent correlation at a lag in correlation lengths."""
        return latent_correlation(self.marginal, self.marginal, self.target(lag))
