"""Gaussian random fields on a regular grid, by circulant embedding.

This module embeds any correlation function and doesn't assume anything about what
the caller does with the field. A :class:`Covariance` is a :class:`Correlation` and the
lengths that scale its argument. :func:`sampler` embeds one on a :class:`Grid` once
and draws from it as often as asked. A :class:`Blend` crossfades several covariances
cell by cell, for a field whose correlation lengths change across the grid. Every tuple
is ``(axis 0, axis 1)``.
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
by the time it wraps. This prevents correlated slip on opposite edges of a segment.
"""

MAXIMUM_VARIANCE_DEFICIT = 1.0e-2
"""How much of a covariance's variance may sit in unsamplable directions. """

MAXIMUM_EMBEDDING_DOUBLINGS = 3
"""How many times the margin is doubled before the embedding is refused."""

# TODO: Make this a keyword argument of `sampler` if an HPC run hits it. It's a
# resource limit that depends on the machine, unlike the accuracy tolerances before
# it, which are constants so that every run's fields are equally accurate.
MAXIMUM_EMBEDDING_CELLS = 1 << 27
"""The largest padded grid to transform. """

SAMPLER_CACHE_SIZE = 128
"""How many embeddings :func:`sampler` keeps. Each segment takes two, and one more for
each family a :class:`Blend` adds, so a fault system of up to 64 segments, or 42 with
two families each, draws again without embedding anything."""


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

    def sampler(self, grid: Grid) -> "Sampler":
        """Embed the covariance on ``grid``, through :func:`sampler`'s cache.

        Parameters
        ----------
        grid : Grid
            The grid to draw on.

        Returns
        -------
        Sampler
            The embedding, ready to draw from.
        """
        return sampler(grid, self)


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

    The DCT optimisation needs an even transform length. In
    practice the extra padding is more than made up for by the DCT savings over
    naive FFT2.
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


def _embed(
    grid: Grid, covariance: Covariance, floor: tuple[int, int] = (0, 0)
) -> tuple[FieldArray, tuple[int, int]]:
    """The square-rooted eigenvalues of a circulant embedding, and the padded shape.

    The margin starts at the correlation's decay length and doubles until the variance
    the clipped negative eigenvalues drop is within :data:`MAXIMUM_VARIANCE_DEFICIT`.
    The padded shape is at least ``floor`` on each axis.
    """
    decay = _decay_length(covariance.correlation, WRAP_TOLERANCE)
    for doubling in range(MAXIMUM_EMBEDDING_DOUBLINGS):
        margin = decay * 2**doubling
        rows, columns = (
            max(_padded_extent(extent, resolution, length, margin), least)
            for extent, resolution, length, least in zip(
                grid.shape,
                grid.resolution_km,
                covariance.lengths_km,
                floor,
                strict=True,
            )
        )
        padded = (rows, columns)
        if padded[0] * padded[1] > MAXIMUM_EMBEDDING_CELLS:
            raise RuptureGeneratorError(
                f"a {grid.shape[0]}x{grid.shape[1]} grid with correlation lengths "
                f"{covariance.lengths_km} km embeds in {padded[0]}x{padded[1]} = "
                f"{padded[0] * padded[1]:,} cells, past the limit of "
                f"{MAXIMUM_EMBEDDING_CELLS:,}"
            )

        lags = [
            np.arange(extent // 2 + 1) * resolution / length
            for extent, resolution, length in zip(
                padded, grid.resolution_km, covariance.lengths_km, strict=True
            )
        ]
        quadrant = covariance.correlation(np.hypot(lags[0][:, None], lags[1][None, :]))
        spectrum = sp.fft.dctn(quadrant, type=1)

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
        return circulant_draw(self.amplitudes, self.padded, self.shape, _seed(rng))


def _seed(rng: np.random.Generator) -> int:
    """Take one draw's seed from ``rng``."""
    return int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))


@functools.lru_cache(maxsize=SAMPLER_CACHE_SIZE)
def sampler(
    grid: Grid, covariance: Covariance, floor: tuple[int, int] = (0, 0)
) -> Sampler:
    """Embed ``covariance`` on ``grid``, or return the embedding already made.

    Parameters
    ----------
    grid : Grid
        The grid to draw on.
    covariance : Covariance
        The covariance to embed.
    floor : tuple of int, optional
        The least padded shape, so covariances that share noise share a padded grid.

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
    return Sampler(grid.shape, *_embed(grid, covariance, floor))


@dataclasses.dataclass(frozen=True, eq=False)
class Blend:
    """Stationary covariances that filter one white noise, crossfaded cell by cell.

    It is the covariance of a field whose correlation lengths change across the grid,
    such as slip with long lengths near the surface and short ones at depth. Every
    component filters the same noise, so a feature changes shape through a crossfade
    rather than fading out. A cell's loadings are ``sin(pi w / 2)``, scaled to unit
    variance from the components' exact cross-correlations, so the field is standard
    normal at every cell however the weights overlap.

    Attributes
    ----------
    components : tuple of Covariance
        The stationary covariances.
    weights : tuple of FieldArray
        Each component's weight at each cell, in ``[0, 1]``. Every cell needs a
        positive weight.
    """

    components: tuple[Covariance, ...]
    weights: tuple[FieldArray, ...]

    def sampler(self, grid: Grid) -> "BlendedSampler":
        """Embed every component on ``grid`` at one padded shape, and load them.

        Parameters
        ----------
        grid : Grid
            The grid to draw on, shaped like the weights.

        Returns
        -------
        BlendedSampler
            The embeddings and each one's loading, ready to draw from.

        Raises
        ------
        RuptureGeneratorError
            If a cell has no positive weight, or a component doesn't embed.
        """
        gains = np.sin(0.5 * np.pi * np.asarray(self.weights))
        power = np.sum(np.square(gains), axis=0)
        if not np.all(power > 0.0):
            raise RuptureGeneratorError(
                f"{int(np.sum(~(power > 0.0)))} cells have no positive weight in the blend"
            )
        floor = (0, 0)
        while True:
            samplers = [
                sampler(grid, component, floor) for component in self.components
            ]
            widest = tuple(max(axis) for axis in zip(*(s.padded for s in samplers)))
            if all(s.padded == widest for s in samplers):
                break
            floor = widest
        weights = np.outer(*(_mirror_weights(extent) for extent in widest))
        amplitudes = np.array([s.amplitudes for s in samplers])
        cross = np.einsum("ikl,jkl,kl->ij", amplitudes, amplitudes, weights)
        cross /= widest[0] * widest[1]
        gains /= np.sqrt(power)
        variance = np.einsum("ikl,ij,jkl->kl", gains, cross, gains)
        return BlendedSampler(tuple(samplers), gains / np.sqrt(variance))


@dataclasses.dataclass(frozen=True, eq=False)
class BlendedSampler:
    """A :class:`Blend`'s components, embedded on one padded grid, and their loadings.

    Attributes
    ----------
    samplers : tuple of Sampler
        The components' embeddings, all at one padded shape.
    loadings : numpy.ndarray
        Each component's loading at each cell, shaped ``(components, *grid)``.
    """

    samplers: tuple[Sampler, ...]
    loadings: np.ndarray

    def draw(self, rng: np.random.Generator) -> tuple[FieldArray, FieldArray]:
        """Draw a pair of independent blended fields, standard normal at every cell.

        Every component draws from the one seed, so they filter the same noise. The
        imaginary parts are independent of every real part, because the amplitudes
        are even.

        Parameters
        ----------
        rng : np.random.Generator
            The generator the draw's seed comes from.

        Returns
        -------
        tuple of FieldArray
            The real and imaginary fields, each shaped like the grid.
        """
        seed = _seed(rng)
        real = np.zeros(self.loadings.shape[1:])
        imaginary = np.zeros(self.loadings.shape[1:])
        for loading, embedding in zip(self.loadings, self.samplers, strict=True):
            parts = circulant_draw(
                embedding.amplitudes, embedding.padded, embedding.shape, seed
            )
            real += loading * parts[0]
            imaginary += loading * parts[1]
        return real, imaginary


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
