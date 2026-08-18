"""The field stages: slip, rise time, rake, and the onset perturbation.

Each is a pure function of ``(mesh, params, rng)``. Every stage takes its own
generator, named from the event seed, so changing one stage's parameters cannot
change another stage's noise.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Protocol

import numpy as np

from rupture_generator.sampling import (
    VonKarmanFilterParameters,
    correlate_fields,
    standardise,
    von_karman_field,
)

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]
BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]


class Chart(Protocol):
    """The whole of what a field stage asks of a chart."""

    def centres(self) -> FloatArray:
        """Subfault centres, positions with depth last."""
        ...

    def occupied(self) -> BoolArray:
        """Which cells are really fault. All true unless the chart says otherwise."""
        ...


type FieldSampler = Callable[..., FloatArray]
"""Draws one field of a segment's covariance, one value per subfault.

Called as ``sampler(mesh, covariance, rng)``; the chart is passed through and otherwise
unread, so a stage can be given a fixed field by passing a closure that ignores it.
"""


@dataclasses.dataclass(frozen=True)
class DepthRamp:
    """A linear transition between two depths, in kilometres."""

    centre_km: float
    half_width_km: float

    @property
    def shallow_km(self) -> float:
        """Where the transition begins."""
        return self.centre_km - self.half_width_km

    @property
    def deep_km(self) -> float:
        """Where it finishes."""
        return self.centre_km + self.half_width_km

    def weight(self, depth_km: FloatArray) -> FloatArray:
        """0 above the ramp, 1 below it, linear across -- clamped at both ends."""
        return np.clip(
            (depth_km - self.shallow_km) / (self.deep_km - self.shallow_km), 0.0, 1.0
        )


# S4 -- slip


@dataclasses.dataclass(frozen=True)
class SlipParams:
    """What shapes a slip field, before the moment decides its size.

    ``coefficient_of_variation`` is **dimensionless**, unlike
    :attr:`RakeParams.sigma_deg`. The tapers are fractions of the fault's own extent;
    ``top_taper`` is zero in production, so slip reaches the free surface at full
    amplitude.
    """

    covariance: VonKarmanFilterParameters
    coefficient_of_variation: float = 0.75
    side_taper: float = 0.02
    top_taper: float = 0.0
    bottom_taper: float = 0.0


def _taper_widths(
    params: SlipParams, cells_i: int, cells_j: int
) -> tuple[int, int, int]:
    def width(fraction: float, extent: int) -> int:
        return max(0, int(fraction * extent + 0.5))

    side = width(params.side_taper, cells_j)
    top = width(params.top_taper, cells_i)
    bottom = width(params.bottom_taper, cells_i)
    if 2 * side > cells_j:
        raise ValueError(
            f"a side taper of {params.side_taper} reaches {side} cells from each end "
            f"of a fault {cells_j} cells long, so the two ramps overlap. A taper is a "
            "statement about the edges; past a half of the fault it is a statement "
            "about the middle"
        )
    if top + bottom > cells_i:
        raise ValueError(
            f"the up-dip and down-dip tapers reach {top} and {bottom} cells of a "
            f"fault {cells_i} cells wide, so they overlap"
        )
    return side, top, bottom


def _reach(mask: BoolArray, axis: int, *, reverse: bool) -> IntArray:
    """How many occupied cells run up to each cell along one direction, inclusive.

    One for a cell whose neighbour on that side is off the fault or off the grid, and
    counting up from there. This is what makes a taper follow a ragged outline: on a
    chart that is fault everywhere it is just the index, so the ramps below are the
    index ramps they always were.
    """
    ordered = np.flip(mask, axis=axis) if reverse else mask
    ordered = np.moveaxis(ordered, axis, 0)
    counted = np.zeros(ordered.shape, dtype=np.int64)
    running = np.zeros(ordered.shape[1], dtype=np.int64)
    for line in range(ordered.shape[0]):
        running = np.where(ordered[line], running + 1, 0)
        counted[line] = running
    counted = np.moveaxis(counted, 0, axis)
    return np.flip(counted, axis=axis) if reverse else counted


def taper_edges(
    field: FloatArray, params: SlipParams, occupied: BoolArray | None = None
) -> FloatArray:
    """Ramp a field to zero at the fault's edges.

    Applied after truncation and before the moment scaling. Separable: the product of
    four independent ramps, one per edge, so a cell two of them reach is damped by
    both. Widths are in whole cells.

    The edge is the **fault's**, not the chart's. On a chart that fills its rectangle
    those are the same thing and the ramps run from the index bounds. On one resampled
    from a modeller's outline they are not: the ramp starts wherever the fault does
    along that axis, so an interface tapers into its own trench and down-dip limit
    rather than into the corner of a bounding box, and unoccupied cells come back zero.

    Which edge is which still comes off the axes rather than off the outline: up-dip is
    ``i`` decreasing whatever shape the boundary is there. That is exact for an
    interface whose outline runs along strike and down dip, which is what a subduction
    model ships, and approximate for one with a boundary cutting across.
    """
    cells_i, cells_j = field.shape
    side, top, bottom = _taper_widths(params, cells_i, cells_j)
    mask = np.ones(field.shape, dtype=bool) if occupied is None else occupied

    ramp = np.ones(field.shape, dtype=np.float64)
    for width, axis, reverse in (
        (top, 0, False),
        (bottom, 0, True),
        (side, 1, False),
        (side, 1, True),
    ):
        if width > 0:
            ramp *= np.minimum(_reach(mask, axis, reverse=reverse) / width, 1.0)

    return field * ramp * mask


def slip_pattern(
    mesh: Chart,
    params: SlipParams,
    rng: np.random.Generator,
    *,
    sampler: FieldSampler = von_karman_field,
    taper: Callable[..., FloatArray] = taper_edges,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """S4, up to the moment: a non-negative, tapered, unit-ish slip pattern.

    ``1 + cov * Z`` on a standardised von Karman field, truncated at zero and tapered;
    the size is set later, by :func:`~rupture_generator.moment.scale_to_moment`.
    Returns the pattern, the standardised Gaussian, and that Gaussian **before
    standardising** -- the draw later stages correlate against, since truncation
    breaks the affine relation the correlation is a statement about.
    """
    drawn = sampler(mesh, params.covariance, rng)
    gaussian = standardise(drawn)
    pattern = 1.0 + params.coefficient_of_variation * gaussian
    pattern = np.maximum(pattern, 0.0)
    # A chart that fills its rectangle reports an all-true mask, so this is the
    # rectangular taper there and the outline-following one only where it differs.
    return taper(pattern, params, mesh.occupied()), gaussian, drawn


def truncated_fraction(gaussian: FloatArray, params: SlipParams) -> float:
    """What fraction the truncation clipped; about 9% at the production spread 0.75."""
    return float(np.mean(1.0 + params.coefficient_of_variation * gaussian < 0.0))


# S5 -- rise time


@dataclasses.dataclass(frozen=True)
class RiseTimeParams:
    """How long each subfault slips for.

    ``average_s`` is the fault-wide mean in seconds and ``sigma`` the field's
    coefficient of variation, dimensionless. ``slip_exponent`` is the power law: 0.5,
    Graves & Pitarka, is rise time as the square root of slip. Above ``shallow_blend``
    the field is replaced by slip itself.
    """

    average_s: float
    correlation: float = 0.9
    sigma: float = 0.75
    slip_exponent: float = 0.5
    shallow_blend: DepthRamp = DepthRamp(2.0, 1.0)
    shallow_stretch: DepthRamp = DepthRamp(6.5, 1.5)
    deep_stretch: DepthRamp = DepthRamp(17.5, 2.5)
    shallow_factor: float = 2.0
    deep_factor: float = 2.0

    def stretch_at(self, depth_km: FloatArray) -> FloatArray:
        """The depth stretch: ``1 + excess`` outside the ramps, exactly 1 between."""
        shallow_excess = self.shallow_factor - 1.0
        deep_excess = self.deep_factor - 1.0
        return (
            1.0
            + shallow_excess * (1.0 - self.shallow_stretch.weight(depth_km))
            + deep_excess * self.deep_stretch.weight(depth_km)
        )


def average_rise_time_s(
    moment_nm: float, coefficient: float, geometric_correction: float
) -> float:
    """The fault-wide mean rise time, from the moment.

    .. math:: \\bar\\tau = c \\, M_0^{1/3} \\, \\alpha_T

    The published scale ``1e-9`` is per cube-root dyne-centimetre; this is its
    newton-metre equivalent, ``1e-9 * (1e7)^(1/3)``.
    """
    scale = 1.0e-9 * (1.0e7 ** (1.0 / 3.0))
    return coefficient * scale * np.cbrt(moment_nm) * geometric_correction


def rise_time_field(
    mesh: Chart,
    slip_gaussian: FloatArray,
    slip_draw: FloatArray,
    params: RiseTimeParams,
    rng: np.random.Generator,
    covariance: VonKarmanFilterParameters,
    *,
    sample_interval_s: float,
    sampler: FieldSampler = von_karman_field,
) -> FloatArray:
    """S5: a rise time for every subfault, in seconds.

    The mean is the requested average by construction, except where the floor binds:
    a pulse shorter than one sample cannot be represented.
    """
    depth_km = mesh.centres()[..., 2]

    # Correlated with slip's Gaussian rather than with the slip itself: both are
    # standardised by construction, so the two fields are already on one scale.
    independent = sampler(mesh, covariance, rng)
    correlated = standardise(
        correlate_fields(slip_draw, independent, params.correlation)
    )

    # Near the surface, blend toward slip's own Gaussian.
    weight = params.shallow_blend.weight(depth_km)
    blended = weight * correlated + (1.0 - weight) * slip_gaussian

    spread = float(blended.std())
    if spread == 0.0:
        pattern = np.ones_like(blended)
    else:
        pattern = 1.0 + params.sigma * (blended - blended.mean()) / spread
    pattern = np.maximum(pattern, 0.0)

    if params.slip_exponent <= 0.1:
        raise ValueError(
            f"a slip exponent of {params.slip_exponent} abandons the correlated "
            "field for independent lognormal noise, which is a different model and "
            "is not implemented -- use an exponent above 0.1"
        )
    pattern = np.where(pattern > 0.0, pattern**params.slip_exponent, pattern)

    mean = float(pattern.mean())
    if mean == 0.0:
        raise ValueError("every subfault's rise-time pattern is zero")
    pattern = pattern / mean

    stretched = params.stretch_at(depth_km) * pattern
    normalisation = float(stretched.mean())
    floor = sample_interval_s / params.average_s
    return np.maximum(stretched / normalisation, floor) * params.average_s


# S6 -- rake


@dataclasses.dataclass(frozen=True)
class RakeParams:
    """Which way each subfault slips. ``sigma_deg`` is in **degrees**."""

    covariance: VonKarmanFilterParameters
    base_rake_deg: float = 175.0
    sigma_deg: float = 15.0


def rake_field(
    mesh: Chart,
    params: RakeParams,
    rng: np.random.Generator,
    *,
    sampler: FieldSampler = von_karman_field,
) -> FloatArray:
    """S6: a rake for every subfault, in degrees.

    ``base + sigma * Z``, with ``Z`` an independent field of the same covariance
    family -- not correlated with slip.
    """
    field = standardise(sampler(mesh, params.covariance, rng))
    return params.base_rake_deg + params.sigma_deg * field


# S8 -- onset perturbation


@dataclasses.dataclass(frozen=True)
class OnsetParams:
    """How the rupture front's arrival is perturbed away from the wavefront.

    ``scale_s`` is seconds per unit of perturbation and **negative** in production
    (-0.35): high-slip patches correlate positively with the field, so a negative
    scale makes them rupture systematically *earlier*. ``sigma`` is dimensionless;
    its product with ``scale_s`` is the perturbation's spread in seconds.
    """

    scale_s: float
    correlation: float = 0.8
    sigma: float = 1.0


def onset_perturbation(
    mesh: Chart,
    slip_draw: FloatArray,
    params: OnsetParams,
    rng: np.random.Generator,
    covariance: VonKarmanFilterParameters,
    *,
    sampler: FieldSampler = von_karman_field,
) -> FloatArray:
    """S8's draw: the shape of the onset perturbation, correlated with slip.

    Dimensionless and standardised, one value per subfault; ``slip_draw`` is slip's
    own unstandardised draw. The seconds, the pinning and the clamp belong to
    :func:`apply_perturbation`.
    """
    independent = sampler(mesh, covariance, rng)
    return standardise(correlate_fields(slip_draw, independent, params.correlation))


def apply_perturbation(
    travel_time_s: FloatArray,
    perturbation: FloatArray,
    params: OnsetParams,
    *,
    hypocentre: tuple[int, int] | int | None,
    delay_s: float,
) -> FloatArray:
    """S8: onset from travel time plus an already-drawn perturbation.

    .. math:: t_{ij} = T_{ij} + c\\,\\sigma\\,Z_{p,ij} + \\mathrm{delay}

    **The hypocentre's perturbation is pinned to zero**, so its onset is exactly its
    travel time plus the delay. ``hypocentre`` is the ``(i, j)`` cell of a lattice or
    the flat face index of a triangulation, or ``None`` for a segment triggered from
    elsewhere, whose onsets stay absolute; ``delay_s`` is the configured delay at the
    root and zero for a triggered segment.
    """
    spread = float(perturbation.std())
    scaled = (
        np.zeros_like(perturbation)
        if spread == 0.0
        else params.sigma * (perturbation - perturbation.mean()) / spread
    )

    if hypocentre is not None:
        scaled = scaled.copy()
        scaled[hypocentre] = 0.0

    onset = travel_time_s + params.scale_s * scaled + delay_s

    if hypocentre is None:
        return onset

    # Nothing ruptures before the earthquake starts. The perturbation is signed and
    # the scale is negative, so a high-slip patch close to the hypocentre can be
    # pulled earlier than the hypocentre itself -- measured at -0.04 s on the shipped
    # example.
    return np.maximum(onset, delay_s)


__all__ = [
    "Chart",
    "DepthRamp",
    "FieldSampler",
    "OnsetParams",
    "RakeParams",
    "RiseTimeParams",
    "SlipParams",
    "apply_perturbation",
    "average_rise_time_s",
    "onset_perturbation",
    "rake_field",
    "rise_time_field",
    "slip_pattern",
    "taper_edges",
    "truncated_fraction",
]
