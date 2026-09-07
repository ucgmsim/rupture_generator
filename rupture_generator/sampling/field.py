"""Gaussian random fields on a regular grid, by circulant embedding.

Nothing here knows what correlation function it is embedding or what will be done
with the field afterwards. A :class:`Covariance` is a :class:`Correlation` and the
lengths that scale its argument; a :class:`Sampler` embeds one on a :class:`Grid`
once and draws from it as often as asked. Every tuple is ``(axis 0, axis 1)``.
"""

import dataclasses
import math
from typing import Protocol

import numpy as np
import scipy as sp

from rupture_generator._kernels import circulant_draw

type FieldArray = np.ndarray[tuple[int, int], np.dtype[np.float64]]

WRAP_TOLERANCE = 1.0e-5
"""The correlation two cells may pick up through the periodic boundary.

The embedding is periodic, so cells at opposite ends of the grid are also neighbours
across the wrap. The margin is sized so the correlation function has decayed to this
by the time it wraps, which is what the embedding delivers as spurious correlation
between the fault's far edges.
"""

MAXIMUM_VARIANCE_DEFICIT = 1.0e-10
"""How much of a covariance's variance may sit in unsamplable directions.

An embedding is sampled through the square roots of its eigenvalues; a negative one
is clipped to zero and the variance it carried is dropped. This is the ratio of what
was dropped to what was kept. At this level it is round-off; above it the covariance
is not positive definite on this grid and a larger margin is the cure.
"""

MAXIMUM_EMBEDDING_DOUBLINGS = 3
"""How many times the margin is doubled before the embedding is refused."""

MAXIMUM_EMBEDDING_CELLS = 1 << 26
"""The largest padded grid to transform.

The transform holds the covariance as ``float64`` and its spectrum as ``complex128``
at once: at 2^26 cells that is 0.5 GB and 1.0 GB, before the caller's own fields.
"""


class Correlation(Protocol):
    """A stationary correlation function of a dimensionless lag."""

    def __call__(self, lag: np.ndarray) -> np.ndarray:
        """Evaluate at a lag measured in correlation lengths."""
        ...


@dataclasses.dataclass(frozen=True)
class Grid:
    """A regular grid: how many cells on each axis, and how far apart they are."""

    shape: tuple[int, int]
    resolution_km: tuple[float, float]


@dataclasses.dataclass(frozen=True)
class Covariance:
    """A stationary covariance: a correlation function and the metric it is read in.

    The anisotropy is entirely in ``lengths_km``. Each axis of a lag is divided by its
    own length before the Euclidean norm, so the correlation's contours are the
    ellipse through the two lengths.
    """

    correlation: Correlation
    lengths_km: tuple[float, float]


class SamplingError(ValueError):
    """A covariance that does not embed on the grid it was asked for."""


def _wrapped_lag_index(extent: int) -> np.ndarray:
    positions = np.arange(extent)
    return np.minimum(positions, extent - positions)


def _decay_length(correlation: Correlation, tolerance: float) -> float:
    """The lag, in correlation lengths, past which the correlation is below ``tolerance``."""
    upper = 1.0
    while correlation(np.array([upper]))[0] > tolerance:
        upper *= 2.0
        if upper > 2.0**20:
            raise SamplingError(
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
    return int(
        sp.fft.next_fast_len(extent + math.ceil(margin * length_km / resolution_km))
    )


def _embed(grid: Grid, covariance: Covariance) -> FieldArray:
    # The circulant embedding method samples a stationary random field by
    # embedding that field on a torus, and using fourier transforms to shape
    # noise. To do that in a stable fashion (i.e. to capture the correlation
    # lengths properly), one must pad the grid appropriately. The circulant
    # embedding method can measure the variance deficit *before sampling* which
    # is what we do when we check deficit against the maximum variance deficit.
    # A Sampler holds the result, so repeatedly sampling fields with the same
    # shape and correlation lengths (e.g., sampling slip, rise, rake independently)
    # only has to pay the cost of this search once.
    decay = _decay_length(covariance.correlation, WRAP_TOLERANCE)
    for doubling in range(MAXIMUM_EMBEDDING_DOUBLINGS):
        margin = decay * 2**doubling
        padded = tuple(
            _padded_extent(extent, resolution, length, margin)
            for extent, resolution, length in zip(
                grid.shape, grid.resolution_km, covariance.lengths_km, strict=True
            )
        )
        if padded[0] * padded[1] > MAXIMUM_EMBEDDING_CELLS:
            raise SamplingError(
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
        wrapped = quadrant[np.ix_(*(_wrapped_lag_index(extent) for extent in padded))]

        spectrum = np.fft.fft2(wrapped).real
        eigenvalues = np.maximum(spectrum, 0.0)
        kept = eigenvalues.sum().item()
        deficit = (kept - spectrum.sum().item()) / kept if kept > 0.0 else 0.0
        if deficit <= MAXIMUM_VARIANCE_DEFICIT:
            eigenvalues.setflags(write=False)
            return eigenvalues

    raise SamplingError(
        f"correlation lengths {covariance.lengths_km} km do not embed on a "
        f"{grid.shape[0]}x{grid.shape[1]} grid at {grid.resolution_km} km: "
        f"{deficit:.1e} of the variance is unsamplable even at a "
        f"{margin:.0f}-correlation-length margin"
    )


class Sampler:
    """One covariance embedded on one grid: the expensive part, done once.

    Every :meth:`draw` is a fresh standard-normal field with this covariance. Fields
    that must be mixed by :func:`mix` are drawn from the same sampler.
    """

    def __init__(self, grid: Grid, covariance: Covariance) -> None:
        """Embed ``covariance`` on ``grid``.

        Raises
        ------
        SamplingError
            If the covariance does not embed within the cell cap, or is not
            positive definite at any margin tried.
        """
        self.shape = grid.shape
        self.eigenvalues = _embed(grid, covariance)

    def draw(self, rng: np.random.Generator) -> FieldArray:
        """A field on the grid, standard normal at every cell."""
        seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        return circulant_draw(self.eigenvalues, self.shape, seed)


def mix(*terms: tuple[float | FieldArray, FieldArray]) -> FieldArray:
    """A standard-normal field from loadings on standard-normal fields.

    ``sum(a * z) / sqrt(sum(a^2))``. Scalar or per-cell loadings; the division is
    what keeps every cell standard normal, which is the precondition a marginal
    transform states. Two fields drawn from one :class:`Sampler` loaded at
    ``(rho, sqrt(1 - rho^2))`` are correlated at ``rho``.
    """
    if not terms:
        raise ValueError("mix needs at least one term")
    total = sum(loading * field for loading, field in terms)
    norm = np.sqrt(sum(np.square(loading) for loading, _ in terms))
    return np.asarray(total / norm, dtype=np.float64)


def standardise(field: FieldArray) -> FieldArray:
    """Zero mean, unit sample variance; zeros for a field with no spread."""
    spread = float(field.std())
    if spread == 0.0:
        return np.zeros_like(field)
    return (field - field.mean()) / spread
