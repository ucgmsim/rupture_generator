"""Correlated Gaussian random fields with von Karman correlations.

Sampled by **circulant embedding** (Dietrich & Newsam 1993; Wood & Chan 1994), which
is exact: on the fault's own grid the drawn field has precisely the target covariance,
to floating point, rather than an approximation of it.

# The model is a covariance, not a spectrum

Mai & Beroza (2002) equation (1) gives the von Karman autocorrelation directly:

.. math::

    C(r) = \\frac{G_H(r)}{G_H(0)}, \\qquad G_H(r) = r^H K_H(r), \\qquad
    r = \\sqrt{\\frac{x^2}{a_x^2} + \\frac{z^2}{a_z^2}}

``r`` is a **dimensionless** distance -- the separation measured in correlation
lengths, one per axis -- and ``C`` is the standard Matern correlation of smoothness
``H``. That expression is what this module implements, and implementing it is what
makes ``a_x`` and ``a_z`` unambiguous.

The paper also states the matching power spectrum,
``P(k) = a_x a_z / (1 + k^2)^{H+1}`` with ``k`` the dimensionless wavenumber
``sqrt(a_x^2 k_x^2 + a_z^2 k_z^2)``. Sampling *that* instead requires deciding whether
``k_x`` is angular or in cycles, and the two answers differ by a factor of ``2*pi`` in
the delivered correlation length -- an ambiguity no output can adjudicate, because both
give plausible-looking slip. Working from ``C(r)`` removes the question rather than
answering it.

# Why circulant embedding rather than shaping noise with a spectrum

Multiplying white noise by a sampled spectrum and inverse-transforming is cheaper, and
it is what this module used to do. Its field is only approximately the target: the
covariance it delivers is the target *wrapped* by the periodic grid and *aliased* by
the discrete spectrum, and nothing in the method reports how large either error is --
the padding that controls the first was a fraction of the fault, which is not a scale
the covariance knows about.

Circulant embedding instead writes the covariance down at the grid's own lags, embeds
it in a circulant matrix whose eigenvalues are that array's DFT, and draws by
multiplying their square roots into complex white noise. Exactness costs a padded grid
at least twice the fault on each axis, and carries one failure mode -- an embedding
whose eigenvalues are not all non-negative is not a covariance matrix, so it raises
rather than sampling something else.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.fft import next_fast_len
from scipy.special import gamma, kv

from rupture_generator import _kernels

if TYPE_CHECKING:
    from rupture_generator.mesh import RuptureMesh

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

HURST = 0.75
"""The von Karman roughness exponent, and the only correlation shape left.

Mai & Beroza (2002) figure 11: the median over their 44 finite-source models is 0.75
for the circular average, with 0.71 along strike and 0.77 down dip and no dependence
on magnitude or faulting style. One number for both axes is theirs, not a
simplification made here.

It is the Matern smoothness ``nu``. At ``H = 0.5`` the von Karman correlation is the
exponential one, which is why the paper finds the two fit its data comparably.
"""

MINIMUM_EMBEDDING = 2
"""How many times the fault each padded axis is, before any enlargement.

A Toeplitz matrix of ``n`` lags embeds in a circulant of ``2n - 2``: the first row
carries ``c(0) ... c(n-1)`` and then the mirror ``c(n-2) ... c(1)``. Anything smaller
cannot hold the covariance, whatever fraction it is dressed up as -- which is the
argument the old ten-percent margin did not have.
"""

MAXIMUM_DOUBLINGS = 3
"""How many times the margin may be doubled before the embedding is refused.

Dietrich & Newsam's recipe is to enlarge until the embedding is good enough. Starting
from :data:`DECAY_LENGTHS` means the first try is usually the answer, so this bounds a
correction rather than a search: three doublings reaches a margin of 24 correlation
lengths, past which the covariance is not one this fault can carry.
"""

DECAY_LENGTHS = 3.0
"""How many correlation lengths of margin to try first.

The wrap has to land where the covariance has faded, and that distance is set by the
**correlation length**, not by the fault. Three is where the measured error comes in
around ``5e-3`` on the grids this package meets; it is a starting point, not a bound,
because :func:`_embed` measures what it actually got and enlarges if that was
optimistic.
"""

CORRELATION_LENGTH_TOLERANCE = 0.02
"""How far the field's delivered correlation length may sit from the one asked for.

A fraction, per axis: 0.02 is "the fault you get has correlation lengths within two
percent of the ones the magnitude implies".

Stated this way because it is the quantity a seismologist can weigh. The natural
alternative -- the largest difference between the delivered and target correlation
functions -- is a number in correlation units that says nothing about how wrong the
resulting earthquake is. What the model is parameterised by is ``a``, so what a
tolerance should bound is ``a``.

Two percent is far inside what the relation itself knows: Mai & Beroza's scatter about
equation (5) is ``sigma`` of 0.18 in log-length, a factor of 1.5, and figure 4 puts the
error on an individual measured correlation length at 13% for long ones and 55% for
short. The sampler is not the uncertain step.
"""

MAI_MAXIMUM_RATIO = 0.6
"""The largest correlation length, as a fraction of the source dimension, the model was
fitted on.

Mai & Beroza (2002) figure 13: across all 44 finite-source models the ratio sits between
0.25 and 0.6 on each axis, with no dependence on magnitude. Past the upper end a segment
is being asked to carry structure longer than itself, which is not a rupture the
relations describe -- whether or not the grid can reproduce it.
"""

MAXIMUM_EMBEDDING_CELLS = 1 << 26
"""The largest padded grid worth transforming: 67 million cells, about a gigabyte.

A budget rather than a doubling count, because what makes an embedding impossible is
its absolute size and not how many times it was enlarged. Checked before any Bessel
function is evaluated, so a covariance that cannot fit is refused in arithmetic rather
than discovered after several transforms.
"""

EIGENVALUE_TOLERANCE = 1.0e-10
"""How negative an eigenvalue may be, relative to the largest, and still be round-off.

The DFT of a valid covariance is non-negative exactly; what comes back is that
arithmetic in floating point, so the smallest eigenvalues scatter about zero. Ten
orders below the largest is round-off; anything deeper is the embedding failing.
"""


@dataclasses.dataclass(frozen=True)
class VonKarmanFilterParameters:
    """How far a field's structure reaches, and how rough it is.

    Attributes
    ----------
    correlation_length_strike_km, correlation_length_dip_km : float
        The patch size along strike and down dip. The corner of the spectrum is the
        ellipse through the reciprocals of these, so structure larger than them is
        flat and structure smaller falls off.
    hurst : float
        The von Karman roughness exponent. 0.75 is the only value production
        selects, and the same number is a Matern smoothness for the sampler that
        replaces this one.
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
                raise ValueError(f"{name} must be a positive length, got {value}")
        if not (0.0 < self.hurst < 1.0):
            raise ValueError(f"hurst must be in (0, 1), got {self.hurst}")


def correlation_lengths(
    magnitude: float,
    *,
    strike_offset: float = 2.50,
    dip_offset: float = 1.50,
    strike_exponent: float = 0.5,
    dip_exponent: float = 1.0 / 3.0,
) -> VonKarmanFilterParameters:
    """A corner relation's correlation lengths for a magnitude, in kilometres.

    .. math::

        \\lambda_{strike} = 10^{c_{s} M_w - a}, \\qquad
        \\lambda_{dip}    = 10^{c_{d} M_w - b}

    A corner relation is a straight line in log-length against magnitude, so those
    four numbers are the whole of one: an exponent and an offset per axis. **The
    defaults are Mai & Beroza (2002) equation (5)**, which reads

    .. math::

        \\log(a_{s}) \\approx -2.5 + \\tfrac{1}{2} M_w, \\qquad
        \\log(a_{z}) \\approx -1.5 + \\tfrac{1}{3} M_w

    -- what a config naming ``mai`` takes, where stating four of your own is what a
    config naming ``custom`` does. The exponents are the paper's own ``1/2`` and
    ``1/3``, written as fractions because that is how equation (5) writes them; the
    measured regressions they simplify are in its table 3, at 0.53 and 0.37 for all
    mechanisms together.

    These are a *simplification* the paper offers, and it says so: the same table
    gives coefficients per faulting style, and the scatter about equation (5) is
    ``sigma`` of 0.19 and 0.18 in log-length -- a factor of 1.5. A ``custom`` relation
    is how a caller who wants the strike-slip or dip-slip row states it.

    The three named relations this replaced -- Somerville, Suzuki, Given -- are still
    refused by name in the config rather than re-spelled here as coefficients,
    because a name is a claim output cannot adjudicate: `DEFECTS.md` 11 records that
    Mai and Somerville cross over at M7.37, so a comparison below that magnitude says
    whichever one you started from is right. A reader who has the coefficients and a
    reason can still state them as a ``custom`` relation, where the file says the
    numbers rather than a name that stands in for them.
    """
    return VonKarmanFilterParameters(
        correlation_length_strike_km=10.0
        ** (strike_exponent * magnitude - strike_offset),
        correlation_length_dip_km=10.0 ** (dip_exponent * magnitude - dip_offset),
    )


def von_karman_correlation(
    normalised_distance: FloatArray, hurst: float = HURST
) -> FloatArray:
    """Mai & Beroza (2002) equation (1)'s von Karman ACF, at a dimensionless distance.

    .. math:: C(r) = \\frac{G_H(r)}{G_H(0)} = \\frac{2^{1-H}}{\\Gamma(H)} r^H K_H(r)

    ``G_H(r) = r^H K_H(r)`` with ``K_H`` the modified Bessel function of the second
    kind; the normalisation is its own limit at the origin, ``G_H(0) = 2^{H-1}
    \\Gamma(H)``, which is what makes ``C(0) = 1`` rather than infinite.

    The argument is a **distance in correlation lengths**, not in kilometres -- the
    anisotropy lives in how that distance is formed (see :func:`_wrapped_distance`),
    so this function is one-dimensional and has no axes to confuse.

    ``C(1) = 0.5005`` at ``H = 0.75``: a separation of one correlation length is
    where the field has forgotten about half of itself. That number is the whole
    meaning of ``a``, and it is checkable against this function without running the
    sampler at all.
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
    """Distance from the origin to every cell of the periodic grid, in correlation lengths.

    **Wrapped**, which is what makes the embedding circulant: on a periodic grid of
    ``m`` cells the lag to cell ``p`` is ``min(p, m - p)``, because the short way round
    is the distance. Feeding unwrapped lags here builds a Toeplitz matrix instead, whose
    DFT is not its eigenvalues, and the field that falls out has no particular
    covariance at all.

    Each axis is divided by its own correlation length before the two are combined, so
    the result is the ``r`` of Mai & Beroza equation (1) -- anisotropy enters here and
    nowhere else.
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

    ``min(p, m - p)`` takes each value twice on each axis, so the full grid holds every
    distance four times over. The covariance is a function of the distance alone, and
    evaluating it is this module's expensive step -- a modified Bessel function per
    point -- so it is evaluated on the quadrant and gathered out to the full grid.

    Worth the indirection: on a 400x2400 embedding that is 241 thousand Bessel
    evaluations rather than 960 thousand, and the gather is a memory copy.
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
    """The delivered correlation lengths are not the ones the model asked for.

    Raised as a warning rather than an error because a rupture generator has to
    generate: a fault whose structure this grid cannot carry still has to appear in the
    file, and a caller who wants the refusal can turn this into one with
    ``warnings.simplefilter("error", DegradedCorrelation)``.

    What it means, physically, is that the segment is outside the range Mai & Beroza
    fitted -- their 44 models all have a correlation length between 0.25 and 0.6 of the
    source dimension, and a segment shorter than its own correlation length is not one
    of them. The field it gets is the closest this grid can carry, which is a fault
    whose slip patches are as large as it can make them.
    """


@dataclasses.dataclass(frozen=True)
class Embedding:
    """A circulant embedding of one covariance on one grid, and what it cost.

    Attributes
    ----------
    extents : tuple of int
        The padded grid, ``(i, j)``.
    eigenvalues : FloatArray
        Non-negative, on that grid. The negatives a real embedding carries have been
        clipped -- see :attr:`correlation_length_error` for what that cost.
    delivered_km : tuple of float
        The correlation lengths the field will actually have, ``(strike, dip)``,
        measured off the covariance this embedding delivers.
    correlation_length_error : float
        How far :attr:`delivered_km` sits from what was asked for, as a fraction, worst
        of the two axes.

        **Measured, not bounded.** Clipping is the standard remedy when an embedding is
        not quite non-negative (Wood & Chan 1994; Stein 2002), and its cost is one
        inverse transform to find out rather than an argument to trust.
    """

    extents: tuple[int, int]
    eigenvalues: FloatArray
    delivered_km: tuple[float, float]
    correlation_length_error: float


@functools.lru_cache(maxsize=32)
def _embed(
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
) -> Embedding:
    """Embed this covariance on this grid, as closely as the grid allows.

    The covariance is written down at the padded grid's wrapped lags and its 2-D DFT is
    the circulant matrix's eigenvalues, real by construction because the covariance is
    symmetric. Where those are all non-negative the embedding is exact and sampling from
    it reproduces the covariance to floating point.

    Where they are not -- the ordinary case for a smooth covariance on a fine grid --
    **the negatives are clipped and the resulting error is measured**, in the delivered
    correlation length. That is the difference between demanding a positive-definite
    matrix and demanding an accurate covariance, and only the second is what a rupture
    needs: insisting on the first refuses grids that deliver the covariance to a part in
    a thousand.

    Larger embeddings are tried until the error is inside
    :data:`CORRELATION_LENGTH_TOLERANCE`. **It never refuses.** A generator has to
    generate, so a segment whose structure no affordable grid can carry gets the closest
    the grid allows and a :class:`DegradedCorrelation` warning saying what it got
    instead. Cached, so that warning is raised once per segment rather than once per
    field drawn on it.

    Returns
    -------
    Embedding
    """
    best: Embedding | None = None
    for extents in _candidate_extents(cell_counts, spacing_km, parameters):
        candidate = _attempt(extents, cell_counts, spacing_km, parameters)
        # Not assumed monotone in the margin, though it is in practice.
        if best is None or candidate.correlation_length_error < best.correlation_length_error:
            best = candidate
        if candidate.correlation_length_error <= CORRELATION_LENGTH_TOLERANCE:
            break

    assert best is not None, "_candidate_extents never returns an empty list"
    _warn_if_degraded(cell_counts, spacing_km, parameters, best)
    return best


def _warn_if_degraded(
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
    best: Embedding,
) -> None:
    """Say so when the field will not have the structure the model asked for.

    Two independent things can be wrong, and they are not the same thing:

    **The grid could not deliver it.** The embedding is as large as the budget allows
    and the correlation length still comes out wrong by more than the tolerance.

    **The model was never fitted here.** Mai & Beroza's 44 models all have a
    correlation length between 0.25 and 0.6 of the source dimension; past
    :data:`MAI_MAXIMUM_RATIO` a segment is shorter than the structure it is being asked
    to carry, and the field is close to constant across it. That can happen with *no*
    numerical error at all -- a covariance far longer than the fault is reproduced
    perfectly, as a fault that barely varies -- which is why it is checked separately
    rather than inferred from the error.
    """
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
            stacklevel=4,
        )
    elif best.correlation_length_error > CORRELATION_LENGTH_TOLERANCE:
        warnings.warn(
            _degraded_message(cell_counts, spacing_km, parameters, best),
            DegradedCorrelation,
            stacklevel=4,
        )


def _attempt(
    extents: tuple[int, int],
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
) -> Embedding:
    """One embedding at one size: eigenvalues, and what they deliver."""
    # Evaluated on the quadrant and gathered, which is the same array as
    # `von_karman_correlation(_wrapped_distance(...))` at a quarter of the Bessel
    # calls -- see `_quadrant_distance`.
    quadrant = von_karman_correlation(
        _quadrant_distance(extents, spacing_km, parameters), parameters.hurst
    )
    target = quadrant[np.ix_(*(_wrapped_lag_index(extent) for extent in extents))]

    eigenvalues = np.maximum(np.fft.fft2(target).real, 0.0)
    # What clipping actually delivered, read at the lags the fault spans. Longer lags
    # exist on the padded grid and never occur between two of its subfaults.
    delivered = np.fft.ifft2(eigenvalues).real
    lengths = _delivered_lengths(delivered, cell_counts, spacing_km, parameters)
    wanted = _delivered_lengths(target, cell_counts, spacing_km, parameters)

    return Embedding(
        extents=extents,
        eigenvalues=eigenvalues,
        delivered_km=lengths,
        correlation_length_error=_relative_error(lengths, wanted),
    )


def _candidate_extents(
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
) -> list[tuple[int, int]]:
    """Progressively larger embeddings to try, none past the memory budget.

    Always at least one, so a covariance no grid can carry still gets a field: the
    smallest embedding a Toeplitz matrix of this many lags admits at all.
    """
    smallest = (
        int(next_fast_len(MINIMUM_EMBEDDING * cell_counts[0])),
        int(next_fast_len(MINIMUM_EMBEDDING * cell_counts[1])),
    )
    candidates: list[tuple[int, int]] = []
    for doubling in range(MAXIMUM_DOUBLINGS + 1):
        extents = _predicted_extents(
            cell_counts, spacing_km, parameters, DECAY_LENGTHS * 2**doubling
        )
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
        f"0.6. The field it gets has {best.delivered_km[0]:.3g} x "
        f"{best.delivered_km[1]:.3g} km instead, off by "
        f"{best.correlation_length_error * 100:.0f}%: the largest patches this grid can "
        "carry. Slip, moment and timing are unaffected; what is degraded is how the "
        "slip is distributed"
    )


def _delivered_lengths(
    covariance: FloatArray,
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
) -> tuple[float, float]:
    """The correlation lengths a covariance array actually has, ``(strike, dip)``.

    Read the way a correlation length is defined: the separation at which the
    correlation falls to ``C(1)``, which at ``H = 0.75`` is 0.5005 -- the distance over
    which the field forgets half of itself. Infinite on an axis where it never falls
    that far, which is a field that stays correlated across the whole segment.
    """
    strike_km, dip_km = spacing_km
    half = float(von_karman_correlation(np.array([1.0]), parameters.hurst)[0])
    return (
        _crossing_km(covariance[0, : cell_counts[1]], half, strike_km),
        _crossing_km(covariance[: cell_counts[0], 0], half, dip_km),
    )


def _relative_error(
    delivered: tuple[float, float], wanted: tuple[float, float]
) -> float:
    """How far the delivered correlation lengths are from the target's, worst axis.

    Both are measured **by the same estimator on the same lags**, so the grid\'s own
    discretisation cancels. It has to: a fault cut at 1 km carrying a 1.8 km
    correlation length has under two samples per correlation length, and comparing an
    interpolated crossing against the exact ``a`` there would report the grid\'s
    coarseness as the sampler\'s error.

    An axis with fewer than two cells, or one the target itself never decorrelates
    over, contributes nothing -- there is no correlation length on it to get wrong.
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
    """Where a covariance profile first falls to ``level``, interpolated, in kilometres.

    Infinite when it never does: the field then has no correlation length this fault
    can express, because it stays correlated across the whole of it.
    """
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
    covariance has faded -- and never less than :data:`MINIMUM_EMBEDDING` times the
    fault, which is what a Toeplitz matrix of this many lags needs whatever its
    covariance. Rounded to a length the transform likes.

    The margin is in **correlation lengths**, which is the only scale the wraparound
    knows about. A fraction of the fault -- the rule this replaced -- is a statement
    about the wrong quantity: it gives a 240 km fault twenty times the margin of a
    12 km one carrying exactly the same structure.
    """
    strike_km, dip_km = spacing_km
    wanted = (
        max(
            MINIMUM_EMBEDDING * cell_counts[0],
            cell_counts[0]
            + math.ceil(margin * parameters.correlation_length_dip_km / dip_km),
        ),
        max(
            MINIMUM_EMBEDDING * cell_counts[1],
            cell_counts[1]
            + math.ceil(margin * parameters.correlation_length_strike_km / strike_km),
        ),
    )
    return (int(next_fast_len(wanted[0])), int(next_fast_len(wanted[1])))


def standardise(field: FloatArray) -> FloatArray:
    """Zero mean, unit sample variance.

    Not needed for the variance -- the embedding delivers ``C(0) = 1`` by construction
    -- but the stages are written as ``1 + cov * Z`` and mean the sample statistics, so
    this is what makes that arithmetic exact on each realisation rather than on average.
    """
    spread = float(field.std())
    # A one-cell chart has a single sample and hence no variance. The mesh CLI produces
    # one for any plane shorter than half the requested subfault size; dividing there
    # gave infinity, then infinity times zero, and an SRF of NaN that nothing refused.
    if spread == 0.0:
        return np.zeros_like(field)
    return (field - field.mean()) / spread


def von_karman_field(
    mesh: RuptureMesh,
    covariance: VonKarmanFilterParameters,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw a field with this covariance on this chart.

    Returned **unstandardised**, on the fault's own cells. The caller standardises when
    it wants sample statistics and correlates first when it wants a related field --
    :func:`correlate_fields` is exact on these, so nothing has to travel in the
    wavenumber domain.

    The draw itself is `_kernels.von_karman_draw`: the noise is generated straight into
    the transform's buffer already scaled by the square-rooted eigenvalues, so one
    allocation does what the numpy spelling needed six for. On a large embedding that is
    the difference between moving a gigabyte per field and moving a sixth of it.

    The seed is drawn from ``rng`` rather than passed through, because the kernel has a
    generator of its own. Reproducibility is unaffected: the seed is a pure function of
    the event seed, the realisation, the calculation and the segment, which is what
    `random.Streams` guarantees and all this pipeline ever relied on.
    """
    embedding = _embed(mesh.cell_counts, mesh.spacing_km(), covariance)
    seed = int(rng.integers(1 << 63, dtype=np.int64))
    return _kernels.von_karman_draw(
        embedding.eigenvalues, mesh.cell_counts, seed
    )


def correlate_fields(
    field_a: FloatArray, field_b: FloatArray, rho: float
) -> FloatArray:
    """A field correlated at ``rho`` with ``field_a``, on the fault.

    ``rho * A + sqrt(1 - rho^2) * B``. Both operands have the same covariance and the
    weights are the cosine and sine of one angle, so the result has it too: the
    correlation is set without disturbing the covariance, which is what makes the blend
    composable with whatever rescaling a stage applies afterwards.

    Done **on the fault** rather than in the wavenumber domain. The inverse transform is
    linear and the crop is a restriction, so the two are the same expression -- and this
    way a drawn field never has to exist as a padded spectrum, which is what lets every
    draw go straight to a cropped array.

    Both fields must be unstandardised draws of the same covariance. Standardising first
    divides each by its own sample spread, which perturbs the relation by the
    estimator's error.
    """
    if not (-1.0 <= rho <= 1.0):
        raise ValueError(f"a correlation must be in [-1, 1], got {rho}")

    return rho * field_a + np.sqrt(1.0 - rho * rho) * field_b


__all__ = [
    "CORRELATION_LENGTH_TOLERANCE",
    "HURST",
    "MAI_MAXIMUM_RATIO",
    "MAXIMUM_DOUBLINGS",
    "MINIMUM_EMBEDDING",
    "DegradedCorrelation",
    "Embedding",
    "VonKarmanFilterParameters",
    "correlate_fields",
    "correlation_lengths",
    "standardise",
    "von_karman_correlation",
    "von_karman_field",
]
