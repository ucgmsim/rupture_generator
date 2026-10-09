"""Gaussian random fields on a regular grid, by circulant embedding.

This module embeds any correlation function and doesn't assume anything about what
the caller does with the field. A :class:`Covariance` is a :class:`Correlation` and the
lengths that scale its argument. :func:`sampler` embeds one on a :class:`Grid` once
and draws from it as often as asked. Every tuple is ``(axis 0, axis 1)``.
"""

import dataclasses
import functools
import math
from collections.abc import Callable

import numpy as np
import scipy as sp

from rupture_generator._kernels import circulant_draw
from rupture_generator.errors import RuptureGeneratorError

type FieldArray = np.ndarray[tuple[int, int], np.dtype[np.float64]]

type Correlation = Callable[[np.ndarray], np.ndarray]
"""A stationary correlation function of a lag measured in correlation lengths."""

WRAP_TOLERANCE = 1.0e-2
"""The correlation two cells may pick up through the periodic boundary.

The embedding is periodic, so cells at opposite ends of the grid are also neighbours
across the wrap. The margin is sized so the correlation function has decayed to this
by the time it wraps, which is what the embedding delivers as spurious correlation
between the fault's far edges.

A percent, not round-off, because the correlation lengths it protects are an
empirical regression's, and they scatter far more than that. Measured on a 25 x 60
km fault at 0.1 km, the delivered covariance's own best-fit lengths stay within
0.01% of the target's at Mw 6.5 and 7.5, and the largest error at any lag is 1e-2.
The padding is what this buys back: at Mw 7.5 the embedding shrinks from 13.7
million cells to 1.2 million and takes a tenth of the time.
"""

MAXIMUM_VARIANCE_DEFICIT = 1.0e-2
"""How much of a covariance's variance may sit in unsamplable directions.

An embedding is sampled through the square roots of its eigenvalues. A negative one
is clipped to zero, and the variance it held is dropped. This is the ratio of what
was dropped to what was kept. Past it, the covariance is not positive definite
enough on this grid, and a larger margin is the cure.

It moves with :data:`WRAP_TOLERANCE`. Held at round-off, it forces a doubled margin
on nearly every embedding a looser wrap allows, and then costs more than the wrap
saves. At a percent each, the first margin passes, and the clipped variance is a
few parts in a thousand on the faults above.
"""

MAXIMUM_EMBEDDING_DOUBLINGS = 3
"""How many times the margin is doubled before the embedding is refused."""

MAXIMUM_EMBEDDING_CELLS = 1 << 27
"""The largest padded grid to transform.

A draw holds the padded grid as ``complex128``, 2 GB at 2^27 cells; the embedding
itself keeps a quarter of it as ``float64``.
"""

SAMPLER_CACHE_SIZE = 4
"""Embeddings kept between segments: a segment uses two, and a realisation of the
same chart asks for the same two again."""


@dataclasses.dataclass(frozen=True)
class Grid:
    """A regular grid of cells.

    Attributes
    ----------
    shape : tuple of int
        How many cells lie on each axis.
    resolution_km : tuple of float
        How far apart the cells lie on each axis, in kilometres.
    """

    shape: tuple[int, int]
    resolution_km: tuple[float, float]


@dataclasses.dataclass(frozen=True)
class Covariance:
    """A stationary covariance, as a correlation function and the metric it reads lags in.

    All the anisotropy is in ``lengths_km``. The covariance divides each axis of a lag
    by its own length before taking the Euclidean norm. The correlation's contours are
    then the ellipse through the two lengths.

    Attributes
    ----------
    correlation : Correlation
        The correlation as a function of a lag in correlation lengths.
    lengths_km : tuple of float
        The correlation length on each axis, in kilometres.
    """

    correlation: Correlation
    lengths_km: tuple[float, float]


def _decay_length(correlation: Correlation, tolerance: float) -> float:
    """The lag, in correlation lengths, past which the correlation is below ``tolerance``."""
    upper = 1.0
    while correlation(np.array([upper]))[0] > tolerance:
        upper *= 2.0
        if upper > 2.0**20:
            raise RuptureGeneratorError(
                f"the correlation function has not decayed below {tolerance:.0e} by "
                f"{upper:.0e} correlation lengths"
            )
    return float(
        sp.optimize.brentq(
            lambda lag: correlation(np.array([lag]))[0] - tolerance,
            upper / 2.0,
            upper,
        )
    )


def _padded_extent(
    extent: int, resolution_km: float, length_km: float, margin: float
) -> int:
    """The smallest fast, **even** transform length holding the grid and its margin.

    Even, because then the wrapped covariance's spectrum is exactly the DCT-I of its
    first half, which is what :func:`_embed` computes.
    """
    padded = sp.fft.next_fast_len(
        extent + math.ceil(margin * length_km / resolution_km)
    )
    while padded % 2:
        padded = sp.fft.next_fast_len(padded + 1)
    return int(padded)


def _mirror_weights(extent: int) -> np.ndarray:
    """How many times each entry of a half spectrum appears in the whole one."""
    weights = np.full(extent // 2 + 1, 2.0)
    weights[[0, -1]] = 1.0
    return weights


def _embed(grid: Grid, covariance: Covariance) -> tuple[FieldArray, tuple[int, int]]:
    """The square-rooted eigenvalues of a circulant embedding, and the padded shape.

    The embedding periodises the covariance onto a padded torus, whose eigenvalues are
    the transform of its first row. That row is real and even along both axes, and so
    is its transform. With an even padded length, the non-redundant quarter of the
    transform is the type-I DCT of the quarter of the row this function evaluates. The
    function keeps only that quarter, and :func:`circulant_draw` mirrors it.

    The margin starts at the correlation's decay length and doubles until the variance
    the clipped negative eigenvalues drop is within :data:`MAXIMUM_VARIANCE_DEFICIT`.
    """
    decay = _decay_length(covariance.correlation, WRAP_TOLERANCE)
    for doubling in range(MAXIMUM_EMBEDDING_DOUBLINGS):
        margin = decay * 2**doubling
        rows, columns = (
            _padded_extent(extent, resolution, length, margin)
            for extent, resolution, length in zip(
                grid.shape, grid.resolution_km, covariance.lengths_km, strict=True
            )
        )
        padded = (rows, columns)
        if padded[0] * padded[1] > MAXIMUM_EMBEDDING_CELLS:
            raise RuptureGeneratorError(
                f"a {grid.shape[0]}x{grid.shape[1]} grid with correlation lengths "
                f"{covariance.lengths_km} km embeds in {padded[0]}x{padded[1]} = "
                f"{padded[0] * padded[1]:,} cells, past the {MAXIMUM_EMBEDDING_CELLS:,} "
                "this machine can transform"
            )

        lags = [
            np.arange(extent // 2 + 1) * resolution / length
            for extent, resolution, length in zip(
                padded, grid.resolution_km, covariance.lengths_km, strict=True
            )
        ]
        quadrant = covariance.correlation(np.hypot(lags[0][:, None], lags[1][None, :]))
        spectrum = sp.fft.dctn(quadrant, type=1, workers=-1)

        weights = np.outer(*(_mirror_weights(extent) for extent in padded))
        total = float(np.sum(weights * spectrum))
        eigenvalues = np.maximum(spectrum, 0.0, out=spectrum)
        kept = float(np.sum(weights * eigenvalues))
        deficit = (kept - total) / kept if kept > 0.0 else 0.0
        if deficit <= MAXIMUM_VARIANCE_DEFICIT:
            amplitudes = np.sqrt(eigenvalues, out=eigenvalues)
            amplitudes.setflags(write=False)
            return amplitudes, padded

    raise RuptureGeneratorError(
        f"correlation lengths {covariance.lengths_km} km do not embed on a "
        f"{grid.shape[0]}x{grid.shape[1]} grid at {grid.resolution_km} km: "
        f"{deficit:.1e} of the variance is unsamplable even at a "
        f"{margin:.0f}-correlation-length margin"
    )


@dataclasses.dataclass(frozen=True, eq=False)
class Sampler:
    """One covariance embedded on one grid, the expensive part, done once.

    Attributes
    ----------
    shape : tuple of int
        The grid's cell counts, the shape every draw comes back in.
    amplitudes : FieldArray
        The square-rooted eigenvalues of the embedding, one quadrant of the padded
        grid.
    padded : tuple of int
        The padded grid's shape.
    """

    shape: tuple[int, int]
    amplitudes: FieldArray
    padded: tuple[int, int]

    def draw(self, rng: np.random.Generator) -> tuple[FieldArray, FieldArray]:
        """Draw a pair of independent fields on the grid, standard normal at every cell.

        One complex transform gives both. Its real and imaginary parts are independent
        fields with the embedded covariance (Dietrich & Newsam 1997).

        Parameters
        ----------
        rng : np.random.Generator
            The generator the draw's seed comes from.

        Returns
        -------
        tuple of FieldArray
            The real and imaginary fields, each shaped like the grid.
        """
        seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        return circulant_draw(self.amplitudes, self.padded, self.shape, seed)


@functools.lru_cache(maxsize=SAMPLER_CACHE_SIZE)
def sampler(grid: Grid, covariance: Covariance) -> Sampler:
    """Embed ``covariance`` on ``grid``, or return the embedding already made.

    Parameters
    ----------
    grid : Grid
        The grid to draw on.
    covariance : Covariance
        The covariance to embed.

    Returns
    -------
    Sampler
        The embedding, ready to draw from.

    Raises
    ------
    RuptureGeneratorError
        If the covariance doesn't embed within the cell cap, or isn't positive
        definite at any margin tried.
    """
    return Sampler(grid.shape, *_embed(grid, covariance))


def mix(*terms: tuple[float | FieldArray, FieldArray]) -> FieldArray:
    """Mix standard-normal fields by their loadings into one standard-normal field.

    The result is ``sum(a * z) / sqrt(sum(a^2))``, with scalar or per-cell loadings.
    The division keeps every cell standard normal, which is the precondition a
    marginal transform states. Loading a pair of independent fields at
    ``(rho, sqrt(1 - rho^2))`` gives a field correlated with the first at ``rho``.

    Parameters
    ----------
    *terms : tuple of (float or FieldArray, FieldArray)
        Each a loading and the standard-normal field it applies to.

    Returns
    -------
    FieldArray
        The mixed field, standard normal at every cell.
    """
    total = sum(loading * field for loading, field in terms)
    norm = np.sqrt(sum(np.square(loading) for loading, _ in terms))
    return np.asarray(total / norm, dtype=np.float64)


def standardise(field: FieldArray) -> FieldArray:
    """Shift and scale a field to zero mean and unit sample variance.

    Parameters
    ----------
    field : FieldArray
        The field to standardise.

    Returns
    -------
    FieldArray
        The standardised field, or zeros for a field with no spread.
    """
    spread = float(field.std())
    if spread == 0.0:
        return np.zeros_like(field)
    return (field - field.mean()) / spread
