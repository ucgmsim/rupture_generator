"""Gaussian random fields with von Karman correlations.

Sampled by circulant embedding (Dietrich & Newsam 1993; Wood & Chan 1994). Mai &
Beroza (2002) equation (1) gives the von Karman autocorrelation directly:

.. math::

    C(r) = \\frac{G_H(r)}{G_H(0)}, \\qquad G_H(r) = r^H K_H(r), \\qquad
    r = \\sqrt{\\frac{x^2}{a_x^2} + \\frac{z^2}{a_z^2}}

``r`` is a **dimensionless** distance -- the separation measured in correlation
lengths, one per axis -- and ``C`` is the standard Matern correlation of smoothness
``H``.

References
----------
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

Dietrich & Newsam (1993) enlarge until the embedding is good enough; three doublings
from :data:`DECAY_LENGTHS` reaches a margin of 24 correlation lengths, past which the
covariance is not one this fault can carry.
"""

DECAY_LENGTHS = 3.0
"""How many correlation lengths of margin to try first, chosen so that the first
attempt usually embeds and no doubling is allocated."""

CORRELATION_LENGTH_TOLERANCE = 0.02
"""How far the delivered correlation length may sit from the one asked for: a
fraction, per axis, so 0.02 is within two percent on each axis."""

MAI_MAXIMUM_RATIO = 0.6
"""The largest correlation length, as a fraction of the source dimension, that the
model was fitted on.

Mai & Beroza (2002) figure 13: across all 44 finite-source models the ratio sits
between 0.25 and 0.6 on each axis. Past the upper end a segment is being asked to
carry structure longer than itself.
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

EIGENVALUE_TOLERANCE = 1.0e-10
"""How negative an eigenvalue may be, relative to the largest, and still be round-off.

Anything more negative rejects the embedding.
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

    The defaults are Mai & Beroza (2002). The dip exponent is **a third, not 0.3333**:
    their equation (5) reads ``log(a_z) ~ -1.5 + (1/3) Mw``.
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
    """

    extents: tuple[int, int]
    eigenvalues: FloatArray
    correlation_lengths: tuple[float, float]
    correlation_length_error: float


@functools.lru_cache(maxsize=32)
def _embed(
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
) -> Embedding:
    """Embed this covariance on this grid, as closely as the grid allows.

    Warns
    -----
    DegradedCorrelation
        If the embedding fails to reproduce the correlation structure precisely.
    """
    best: Embedding | None = None
    for extents in _candidate_extents(cell_counts, spacing_km, parameters):
        candidate = _attempt(extents, cell_counts, spacing_km, parameters)
        # Not assumed monotone in the margin, though it is in practice.
        if (
            best is None
            or candidate.correlation_length_error < best.correlation_length_error
        ):
            best = candidate
        if candidate.correlation_length_error <= CORRELATION_LENGTH_TOLERANCE:
            break

    assert best is not None, "_candidate_extents never returns an empty list"
    _warn_if_degraded(cell_counts, spacing_km, parameters, best)
    return best


_WARN_STACKLEVEL = 5
"""How far up to point a :class:`DegradedCorrelation`, counted rather than guessed.

``_warn_if_degraded`` -> ``_embed`` -> :func:`von_karman_grid` ->
:func:`von_karman_field` -> the stage, which is five frames. A caller reaching
:func:`von_karman_grid` directly is reported one frame further out than its own call.
"""


def _warn_if_degraded(
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
    best: Embedding,
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


def _attempt(
    extents: tuple[int, int],
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
) -> Embedding:
    quadrant = von_karman_correlation(
        _quadrant_distance(extents, spacing_km, parameters), parameters.hurst
    )
    target = quadrant[np.ix_(*(_wrapped_lag_index(extent) for extent in extents))]

    eigenvalues = np.maximum(np.fft.fft2(target).real, 0.0)
    delivered = np.fft.ifft2(eigenvalues).real
    lengths = _delivered_lengths(delivered, cell_counts, spacing_km, parameters)
    wanted = _delivered_lengths(target, cell_counts, spacing_km, parameters)

    return Embedding(
        extents=extents,
        eigenvalues=eigenvalues,
        correlation_lengths=lengths,
        correlation_length_error=_relative_error(lengths, wanted),
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
    ValueError
        If even the minimum embedding is past :data:`MAXIMUM_EMBEDDING_CELLS`.
    """
    smallest = (
        int(next_fast_len(MINIMUM_EMBEDDING * cell_counts[0])),
        int(next_fast_len(MINIMUM_EMBEDDING * cell_counts[1])),
    )
    if smallest[0] * smallest[1] > MAXIMUM_EMBEDDING_CELLS:
        raise ValueError(
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
) -> tuple[float, float]:
    """The correlation lengths a covariance array actually has, ``(strike, dip)``."""
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
            + int(np.ceil(margin * parameters.correlation_length_strike_km / strike_km)),
        ),
    )
    return (int(next_fast_len(wanted[0])), int(next_fast_len(wanted[1])))


def standardise(field: FloatArray) -> FloatArray:
    """Zero mean, unit sample variance.

    The embedding already delivers ``C(0) = 1``; this is what makes the stages'
    ``1 + cov * Z`` exact on each realisation rather than on average.
    """
    spread = float(field.std())
    # A one-cell chart has a single sample and hence no variance. The mesh CLI produces
    # one for any plane shorter than half the requested subfault size; dividing there
    # gave infinity, then infinity times zero, and an SRF of NaN that nothing refused.
    if spread == 0.0:
        return np.zeros_like(field)
    return (field - field.mean()) / spread


def von_karman_grid(
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    covariance: VonKarmanFilterParameters,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw a field with this covariance on a regular grid.

    The embedding is cached on these arguments. ``cell_counts`` is ``(dip, strike)``;
    ``spacing_km`` is in kilometres and ``(strike, dip)``, the opposite order and the
    one :meth:`~rupture_generator.mesh.RuptureMesh.spacing_km` returns. Returns one
    standard-normal-marginal value per cell.

    Raises
    ------
    ValueError
        If the embedding this grid needs is past :data:`MAXIMUM_EMBEDDING_CELLS`.

    Warns
    -----
    DegradedCorrelation
        If the grid cannot carry the correlation lengths asked of it.
    """
    embedding = _embed(cell_counts, spacing_km, covariance)
    seed = int(rng.integers(1 << 63, dtype=np.int64))
    return _kernels.von_karman_draw(embedding.eigenvalues, cell_counts, seed)


def von_karman_field(
    mesh: RuptureMesh,
    covariance: VonKarmanFilterParameters,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw a field with this covariance on this chart."""
    return von_karman_grid(mesh.cell_counts, mesh.spacing_km(), covariance, rng)


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
        raise ValueError(f"a correlation must be in [-1, 1], got {rho}")

    return rho * field_a + np.sqrt(1.0 - rho * rho) * field_b


__all__ = [
    "CORRELATION_LENGTH_TOLERANCE",
    "HURST",
    "MAI_MAXIMUM_RATIO",
    "MAXIMUM_DOUBLINGS",
    "MAXIMUM_EMBEDDING_CELLS",
    "MINIMUM_EMBEDDING",
    "DegradedCorrelation",
    "Embedding",
    "VonKarmanFilterParameters",
    "correlate_fields",
    "correlation_lengths",
    "standardise",
    "von_karman_correlation",
    "von_karman_field",
    "von_karman_grid",
]
