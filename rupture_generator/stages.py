"""The field stages: slip, taper, rise time, rake, and the onset displacement.

Each is a pure function of ``(mesh, params, rng)``. Every stage takes its own
generator, named from the event seed, so changing one stage's parameters cannot
change another stage's noise.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Protocol

import numpy as np

from rupture_generator.errors import ConfigError
from rupture_generator.sampling import (
    NORMAL,
    FilterParameters,
    Marginal,
    MarginalFamily,
    correlate_fields,
    latent_correlation,
    standardise,
    von_karman_field,
)

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]
BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]


SLIP_MARGINAL_FAMILY: MarginalFamily = "truncated_exponential"
"""What distribution slip is drawn from.

Named here rather than left as a bare default, because the config layer refuses a spread
the family cannot hold and has to ask the same family this one uses.
"""


class Chart(Protocol):
    """The whole of what a field stage asks of a chart."""

    def centres(self) -> FloatArray:
        """Subfault centres, positions with depth last."""
        ...

    def occupied(self) -> BoolArray:
        """Which cells are really fault. All true unless the chart says otherwise."""
        ...

    def spacing_km(self) -> tuple[float, float]:
        """Subfault size, ``(strike, dip)``, in kilometres."""
        ...


type FieldSampler = Callable[..., FloatArray]
"""Draws one **latent** field of a segment's covariance, one value per subfault.

Called as ``sampler(mesh, covariance, rng, marginal=...)``; the chart is passed through
and otherwise unread, so a stage can be given a fixed field by passing a closure that
ignores it. What comes back is always Gaussian -- the marginal says what the draw is
pre-corrected for, and the stage applies it.
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

    ``marginal_family`` is what distribution the pattern's values follow. The default is
    the **truncated exponential**, which Thingbaijam & Mai (2016) fitted to 190 rupture
    models in SRCMOD and Castro-Cruz & Mai (2025) draw slip from; ``truncated_normal``
    is the previous default and remains available. Either is a distribution with the
    requested mean and spread and no negative values, which is what genslip's
    ``1 + cov * Z`` clipped at zero is not -- that is a normal with a point mass at zero
    and neither moment it was configured with.

    The spread and the largest slip are **one number** for the truncated exponential:
    the family is parameterised by where the cut falls, so a coefficient of variation of
    0.90 is a maximum of 4.30 mean slips, and Thingbaijam & Mai's own regression
    (``log10 u_max = 0.95 log10 u_bar + 0.62``) puts that ratio at 3.9 to 4.4 over the
    mean slips a crustal event produces. Hence the default of 0.90 rather than the 0.75
    inherited alongside genslip's clip. See
    :func:`~rupture_generator.sampling._distribution`.

    References
    ----------
    Thingbaijam, K. K. S., & Mai, P. M. (2016). Evidence for truncated exponential
    probability distribution of earthquake slip. *Bulletin of the Seismological Society
    of America*, 106(4), 1802-1816.
    """

    covariance: FilterParameters
    coefficient_of_variation: float = 0.90
    marginal_family: MarginalFamily = SLIP_MARGINAL_FAMILY
    side_taper: float = 0.02
    top_taper: float = 0.0
    bottom_taper: float = 0.0

    @property
    def marginal(self) -> Marginal:
        """The unit-mean distribution the pattern's values are drawn from."""
        return Marginal(self.marginal_family, self.coefficient_of_variation)


def _taper_widths(
    params: SlipParams, cells_i: int, cells_j: int
) -> tuple[int, int, int]:
    def width(fraction: float, extent: int) -> int:
        return max(0, int(fraction * extent + 0.5))

    side = width(params.side_taper, cells_j)
    top = width(params.top_taper, cells_i)
    bottom = width(params.bottom_taper, cells_i)
    if 2 * side > cells_j:
        raise ConfigError(
            f"a side taper of {params.side_taper} reaches {side} cells from each end "
            f"of a fault {cells_j} cells long, so the two ramps overlap. A taper is a "
            "statement about the edges; past a half of the fault it is a statement "
            "about the middle"
        )
    if top + bottom > cells_i:
        raise ConfigError(
            f"the up-dip and down-dip tapers reach {top} and {bottom} cells of a "
            f"fault {cells_i} cells wide, so they overlap"
        )
    return side, top, bottom


def _reach(mask: BoolArray, axis: int, *, reverse: bool) -> IntArray:
    """How many occupied cells run up to each cell along one direction, inclusive.

    One for a cell whose neighbour on that side is off the fault or off the grid, and
    counting up from there. This is what makes a taper follow a ragged outline; on a
    chart that is fault everywhere it is just the index.
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

    The edge is the **fault's**, not the chart's: the ramp starts wherever the fault
    does along that axis, so an interface tapers into its own trench and down-dip limit
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
) -> tuple[FloatArray, FloatArray]:
    """S4, up to the moment: a non-negative, tapered, unit-mean slip pattern.

    The latent von Karman draw pushed through :attr:`SlipParams.marginal` and tapered;
    the size is set later, by :func:`~rupture_generator.moment.scale_to_moment`. The
    mean and the coefficient of variation are the marginal's own, so no rescaling
    happens here and nothing is clipped: what the sampler was asked for is a
    correlation the *pattern* carries, not one a latent field carries and then loses.

    Returns the pattern and the latent draw behind it. Later stages correlate against
    the **latent**, because that is where the correlation is linear -- and they must
    take it unstandardised, since standardising divides by a sample spread the
    marginal transform is not a statement about.
    """
    latent = sampler(mesh, params.covariance, rng, marginal=params.marginal)
    pattern = params.marginal.apply(latent)
    return taper(pattern, params, mesh.occupied()), latent


# S5 -- rise time


@dataclasses.dataclass(frozen=True)
class RiseTimeParams:
    """How long each subfault slips for.

    ``average_s`` is the fault-wide mean in seconds and ``sigma`` the field's
    coefficient of variation, dimensionless. ``slip_exponent`` is the power law: 0.5,
    Graves & Pitarka, is rise time as the square root of slip. Above ``shallow_blend``
    the field is replaced by slip itself.

    ``marginal_family`` is a gamma rather than slip's truncated normal for one reason:
    at a spread near 1 a truncated normal has its mode **at** zero, and a subfault
    that slips a metre in no time at all is an unbounded slip rate. A gamma with a
    coefficient of variation below 1 has shape ``a > 1`` and a mode away from zero.
    """

    average_s: float
    correlation: float = 0.9
    sigma: float = 0.75
    marginal_family: MarginalFamily = "gamma"
    slip_exponent: float = 0.5
    shallow_blend: DepthRamp = DepthRamp(2.0, 1.0)
    shallow_stretch: DepthRamp = DepthRamp(6.5, 1.5)
    deep_stretch: DepthRamp = DepthRamp(17.5, 2.5)
    shallow_factor: float = 2.0
    deep_factor: float = 2.0

    @property
    def marginal(self) -> Marginal:
        """The unit-mean distribution the pattern's values are drawn from."""
        return Marginal(self.marginal_family, self.sigma)

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
    slip_latent: FloatArray,
    slip_marginal: Marginal,
    params: RiseTimeParams,
    rng: np.random.Generator,
    covariance: FilterParameters,
    *,
    sample_interval_s: float,
    sampler: FieldSampler = von_karman_field,
) -> FloatArray:
    """S5: a rise time for every subfault, in seconds.

    The mean is the requested average by construction, except where the floor binds:
    a pulse shorter than one sample cannot be represented.

    Both refusals are made before anything is drawn, so a config that cannot be
    honoured costs no sampling.

    Raises
    ------
    ConfigError
        If the slip exponent is below the floor, or the marginal is one the power law
        cannot be applied to.
    """
    if params.slip_exponent <= 0.1:
        raise ConfigError(
            f"a slip exponent of {params.slip_exponent} abandons the correlated "
            "field for independent lognormal noise, which is a different model and "
            "is not implemented -- use an exponent above 0.1"
        )
    if not params.marginal.is_positive:
        raise ConfigError(
            f"a {params.marginal.family} rise-time marginal takes negative values, "
            f"and they have no {params.slip_exponent} power -- rise time needs a "
            "marginal on the positive half-line"
        )
    depth_km = mesh.centres()[..., 2]

    # Everything mixed here is latent: the correlation is only linear before the
    # marginals are applied, so the independent draw is pre-corrected for *slip's*
    # marginal to put it in the same space as `slip_latent`.
    independent = sampler(mesh, covariance, rng, marginal=slip_marginal)
    correlation = float(
        latent_correlation(slip_marginal, params.marginal, np.array(params.correlation))
    )

    # The shallow blend is a correlation with slip that rises to one at the surface,
    # written as the two loadings it puts on slip and on the independent draw. Dividing
    # by their norm is what genslip's `w, 1 - w` weights get wrong: those two sum to
    # one but do not square to one, so the marginal variance notches at the ramp.
    # Here the latent is standard normal at every depth by construction, which is what
    # the marginal transform below is a statement about.
    weight = params.shallow_blend.weight(depth_km)
    on_slip = weight * correlation + (1.0 - weight)
    on_independent = weight * np.sqrt(1.0 - correlation * correlation)
    latent = (on_slip * slip_latent + on_independent * independent) / np.hypot(
        on_slip, on_independent
    )

    pattern = params.marginal.apply(latent) ** params.slip_exponent

    mean = float(pattern.mean())
    if mean == 0.0:
        raise ConfigError("every subfault's rise-time pattern is zero")
    pattern = pattern / mean

    stretched = params.stretch_at(depth_km) * pattern
    normalisation = float(stretched.mean())
    floor = sample_interval_s / params.average_s
    return np.maximum(stretched / normalisation, floor) * params.average_s


# S6 -- rake


@dataclasses.dataclass(frozen=True)
class RakeParams:
    """Which way each subfault slips. ``sigma_deg`` is in **degrees**."""

    covariance: FilterParameters
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

    The one field that needs no NORTA at all: its marginal *is* normal, so the
    transform and the pre-correction are both the identity, and being uncorrelated
    with slip means it does not have to share slip's latent space either. Rake is
    therefore the field that carries the requested correlation lengths exactly.
    """
    field = standardise(sampler(mesh, params.covariance, rng, marginal=NORMAL))
    return params.base_rake_deg + params.sigma_deg * field


# S8 -- the onset perturbation, blended in from the seed


CAUSAL_MARGIN = 1.05
"""How much room the causal clamp leaves past the weight that would tie with the seed.

The clamp admits :math:`w = \\tau / (-\\delta)` exactly, which puts that subfault's onset
*at* the seed time rather than after it; an argmin over the field then breaks the tie
arbitrarily and the hypocentre stops being the unique earliest subfault. Widening the
clamp by a twentieth costs a twentieth of the displacement on the cells it binds -- the
deep tail of the draw -- and buys back a hypocentre that is the strict minimum of the
field. Not configurable: it is a tie-break, not a modelling choice, and the modelling
choice is ``blend_sigma``.
"""


def onset_scale_s(moment_nm: float, offset_s: float, coefficient: float) -> float:
    """The spread of the onset displacement, in seconds, from the moment.

    .. math:: \\sigma = b_0 + c \\, M_0^{1/3}

    Graves & Pitarka's rupture-time perturbation amplitude, and the one quantity in
    this model fitted against recorded ground motion. The offset is what flattens the
    magnitude dependence of the Somerville et al. (1999) self-similar duration scaling
    that the cube-root term is; without it the spread falls to nothing at small
    magnitudes, where the recordings say it does not.

    ``coefficient`` is in the **published** units, per cube-root dyne-centimetre at a
    scale of ``1e-9``, exactly as :func:`average_rise_time_s` takes its own -- so the
    0.5 here is genslip's ``tsfac_slope`` and the 0.1 its ``tsfac_bzero``, read off the
    production defaults rather than re-derived. Both are stated there as negatives,
    because genslip subtracts its ``tsfac``; a spread has no sign, and this returns the
    magnitude.

    The SCEC BBP v19.4 release notes print the coefficient as ``-5.0e-8``, which is a
    factor of 100 off what genslip implements.

    References
    ----------
    Graves, R., & Pitarka, A. (2016). Kinematic ground-motion simulations on rough
    faults including effects of 3D stochastic velocity perturbations. *Bulletin of the
    Seismological Society of America*, 106(5), 2136-2153.

    Somerville, P., Irikura, K., Graves, R., et al. (1999). Characterizing crustal
    earthquake slip models for the prediction of strong ground motion. *Seismological
    Research Letters*, 70(1), 59-80.
    """
    scale = 1.0e-9 * (1.0e7 ** (1.0 / 3.0))
    return offset_s + coefficient * scale * float(np.cbrt(moment_nm))


@dataclasses.dataclass(frozen=True)
class OnsetParams:
    """How far the onset is displaced from the solved front, and how that is admitted.

    ``scale_s`` is the spread of the displacement in **seconds**, which
    :func:`onset_scale_s` reads off the segment's own moment -- the one quantity in this
    model calibrated against recorded ground motion, and the reason the mechanism that
    consumes seconds directly is the one this package ships. Zero is a coherent front
    and draws nothing. ``correlation`` is the coherence with slip: high-slip patches
    rupture earlier.

    ``blend_sigma`` is the width of the blend zone in units of ``scale_s``, so the
    default of 4 blends over the first ``4 * scale_s`` seconds of front travel. See
    :func:`taper_onset` for why the width is stated in sigmas and what the causal clamp
    does with the draws that overrun it.
    """

    scale_s: float
    correlation: float = 0.8
    blend_sigma: float = 4.0

    def __post_init__(self) -> None:
        """Refuse a spread that is not one, or a blend zone with no width."""
        if self.scale_s < 0.0:
            raise ConfigError(
                f"the onset displacement's scale is a spread, got {self.scale_s}"
            )
        if self.blend_sigma <= 0.0:
            raise ConfigError(
                f"the onset blend spans {self.blend_sigma} sigma, which is not a "
                "width the front can blend over; use a positive number of sigmas, or "
                "scale_s = 0 for a coherent front"
            )


def onset_perturbation(
    mesh: Chart,
    slip_latent: FloatArray,
    slip_marginal: Marginal,
    params: OnsetParams,
    rng: np.random.Generator,
    covariance: FilterParameters,
    *,
    sampler: FieldSampler = von_karman_field,
) -> FloatArray:
    """S8's draw: the displacement field, in seconds, correlated with slip.

    **Not** filtered. This field is added to the solved times rather than fed to the
    solve, so it is the spectrum of the onset displacement directly and there is no
    integration by the solver to undo first.

    A ``scale_s`` of zero returns zeros without drawing, so a coherent front costs no
    sample and takes the same code path as a perturbed one.
    """
    if params.scale_s == 0.0:
        return np.zeros(np.shape(slip_latent), dtype=np.float64)

    independent = sampler(mesh, covariance, rng, marginal=slip_marginal)
    correlation = float(
        latent_correlation(slip_marginal, NORMAL, np.array(params.correlation))
    )
    return params.scale_s * standardise(
        correlate_fields(slip_latent, independent, correlation)
    )


def taper_onset(
    travel_time_s: FloatArray,
    perturbation: FloatArray,
    params: OnsetParams,
    *,
    seed_cell: tuple[int, int] | int,
    seed_time_s: float = 0.0,
) -> FloatArray:
    """S8: displace the solved onsets, blending from a smooth front near the seed.

    .. math::
        t_{ij} = T_{ij} + w_{ij} \\, \\delta_{ij}, \\qquad
        w_{ij} = \\min \\! \\left( \\underbrace{\\frac{\\tau_{ij}}{n \\sigma}}_{\\text{blend}},
        \\; \\underbrace{\\frac{\\tau_{ij}}{c \\, \\max(-\\delta_{ij}, 0)}}_{\\text{clamp}},
        \\; 1 \\right)

    where :math:`\\tau` is the time since the front reached this segment,
    :math:`\\delta` is the displacement with its value at the seed removed,
    :math:`\\sigma` is ``scale_s``, :math:`n` is ``blend_sigma`` and :math:`c` is
    :data:`CAUSAL_MARGIN`.

    **Two terms, and they answer different questions.** The blend term is the model: the
    front leaves the seed smooth and reaches full roughness :math:`n \\sigma` of travel
    time later, which is a statement about where onset heterogeneity has had room to
    accumulate and is stated in sigmas because that is the only length the displacement
    field has. The clamp term is arithmetic: a subfault cannot rupture before the front
    that seeded it, which is exactly :math:`w (-\\delta) < \\tau`, and it binds
    **per cell** on the draws that dip deeper than the blend zone allows.

    Splitting them is what lets the blend width be a number you choose. A single term
    cannot: :math:`n \\sigma` is not safe on every draw -- swept over 768 cases across
    four magnitudes, two resolutions, two fault styles and four seed positions, one draw
    needed 8.04 sigma, and the depth of the worst dip is a property of the draw rather
    than of the parameters -- while reading the width off the draw's worst dip instead
    makes the blend length something the realisation decides. So the width is
    configured, the clamp catches what the width does not cover, and the clamp is
    per-cell rather than fault-wide so a single deep dip no longer holds back the whole
    fault behind it.

    Why the ramps are linear and not some other shape. Admissibility is
    :math:`\\sup_s w(s)/s \\le 1` in :math:`s = \\tau / (c e)`; the linear ramp attains
    that bound, which makes it the **largest** admissible taper -- nothing delivers more
    displacement at the same safety -- and any sub-linear ramp violates it near zero and
    cannot be rescued by widening the zone.

    The weight is not monotone in :math:`\\tau`, and that is the price of the per-cell
    clamp: the clamp's denominator is this cell's own dip, so two neighbours at the same
    travel time can be weighted differently. The blend term is monotone, so the
    *pattern* -- smooth near the seed, rough away from it -- is; what varies within it
    is which individual deep-dipping cells are held back, which is the intent.
    """
    since_seed_s = np.asarray(travel_time_s, dtype=np.float64) - seed_time_s
    displacement = np.asarray(perturbation, dtype=np.float64).copy()
    displacement -= displacement[seed_cell]

    blend_s = params.blend_sigma * params.scale_s
    if blend_s > 0.0:
        weight = np.minimum(1.0, since_seed_s / blend_s)
    else:
        weight = np.ones_like(since_seed_s)

    # The clamp, per cell. `1e-300` rather than a branch: where the dip is zero or the
    # displacement is positive the ratio comes out astronomically large and the minimum
    # ignores it, which is the right answer -- a subfault the draw moves *later* is
    # under no causal bound at all, however close to the seed it is.
    dip = np.maximum(-displacement, 0.0)
    with np.errstate(divide="ignore", over="ignore"):
        causal = since_seed_s / (CAUSAL_MARGIN * np.maximum(dip, 1e-300))
    weight = np.minimum(weight, causal)

    return np.asarray(travel_time_s, dtype=np.float64) + weight * displacement


__all__ = [
    "CAUSAL_MARGIN",
    "SLIP_MARGINAL_FAMILY",
    "Chart",
    "DepthRamp",
    "FieldSampler",
    "OnsetParams",
    "RakeParams",
    "RiseTimeParams",
    "SlipParams",
    "average_rise_time_s",
    "onset_perturbation",
    "onset_scale_s",
    "rake_field",
    "rise_time_field",
    "slip_pattern",
    "taper_edges",
    "taper_onset",
]
