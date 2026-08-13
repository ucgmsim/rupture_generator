"""The field stages: slip, rise time, rake, and the onset perturbation.

Each is a pure function of ``(mesh, params, rng)`` returning the chart with a new
variable attached. Nothing is mutated and nothing is shared between them except the
chart, which is why the order they run in is a convention rather than a contract --
`pipeline.py` writes it down once and no stage reads another's stream.

Every stage takes its own generator, named from the event seed, so changing one
stage's parameters cannot change another stage's noise. That deletes a whole class of
ordering machinery: fields drawn and discarded purely to keep a stream in step,
invariants counting deviates, and the rule that a band-pass must never remove a draw.

Three of these stages read a depth ramp, and all three read the same one:
:class:`DepthRamp`, a linear transition between two depths. Its asymmetry is
deliberate -- each branch measures from the ramp's *far* end, so the value is exactly
1 at the inner edge and ``1 + excess`` at the outer.
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


class Chart(Protocol):
    """The whole of what a field stage asks of a chart: where its subfaults are.

    A protocol rather than :class:`~rupture_generator.mesh.RuptureMesh` because three
    of these stages read nothing but the centre depths, and the depths of a
    triangulated segment's faces are the same quantity as the depths of a lattice's
    cells. Naming the concrete type instead would have made every stage that only
    reads depth look like it needed a lattice.
    """

    def centres(self) -> FloatArray:
        """Subfault centres, positions with depth last."""
        ...


type FieldSampler = Callable[..., FloatArray]
"""Draws one field of a segment's covariance, one value per subfault.

Called as ``sampler(mesh, covariance, rng)``, which is
:func:`~rupture_generator.sampling.von_karman_field`'s own signature -- so the default
is that function itself and there is no adapter to keep in step.

**There is one sampler in this package**: circulant embedding
(:mod:`rupture_generator.sampling`). What this parameter carries is not a *choice* of
sampler but the shape of the thing it draws on, which is the one thing the mesh
container decides. A structured chart's cells are already the grid, so it takes the
default; a triangulated segment draws on a lattice over its own parameter plane and
projects onto its faces, so it passes
:func:`~rupture_generator.triangular.pipeline.field_sampler`'s closure, which returns
one value per face instead of a ``(i, j)`` array. That difference in *return shape* is
why the two cannot be one call, and passing the draw in is what lets one rise-time
model, one rake model and one perturbation model serve both containers instead of two
transcriptions free to drift.

Deliberately spelled ``Callable[..., FloatArray]`` rather than a protocol: the first
argument is the chart, which the stage passes through and never inspects, and a
protocol would have to name a type for it that both mesh containers satisfy.
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


# ============================================================================
# S4 -- slip
# ============================================================================


@dataclasses.dataclass(frozen=True)
class SlipParams:
    """What shapes a slip field, before the moment decides its size.

    Attributes
    ----------
    covariance : VonKarmanFilterParameters
        The patch structure, from magnitude.
    coefficient_of_variation : float
        The field's spread as a fraction of its mean, **dimensionless**. Never to be
        confused with :attr:`RakeParams.sigma_deg`, which is in degrees: handing one
        to the other gave every rake a spread of 0.75 degrees where the model wants
        15, a factor of twenty on every fault, and the names are what carry the
        difference.
    side_taper, top_taper, bottom_taper : float
        Fractions of the fault's own extent over which slip ramps to zero at each
        edge. ``top_taper`` is **zero** in production: slip reaches the free surface
        at full amplitude, which is what a surface-rupturing event does.
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


def taper_edges(field: FloatArray, params: SlipParams) -> FloatArray:
    """Ramp a field to zero at the fault's edges.

    A rupture that slips right up to its boundary is unphysical: the edges are where
    the fault stops. Applied to the real field after truncation and before the moment
    scaling, so the moment closes on what the taper left.

    **Separable**, deliberately: the result is the product of two independent
    one-dimensional ramps, one along strike and one down dip, so a cell that two ramps
    reach is damped by both. The alternative -- one ramp overwriting the other where
    they overlap -- is 7/8 in one place and 1 in another on a fourteen-cell fault with
    an eight-cell taper. Overlapping tapers are refused outright, so the region where
    the two forms disagree is unrepresentable.

    A **lattice** taper: the widths are whole cells and the ramps run along the index
    axes, which is why :func:`slip_pattern` takes it as a parameter rather than calling
    it. A triangulated segment has no index axes, so
    :func:`rupture_generator.triangular.pipeline.taper_edges` measures the same
    fractions as distances to the labelled boundary in the parameter plane instead --
    the same product of two independent ramps, and the same refusal when they overlap.
    """
    cells_i, cells_j = field.shape
    side, top, bottom = _taper_widths(params, cells_i, cells_j)

    along_strike = np.ones(cells_j)
    for offset in range(side):
        ramp = (offset + 1) / side
        along_strike[offset] *= ramp
        along_strike[cells_j - 1 - offset] *= ramp

    down_dip = np.ones(cells_i)
    for offset in range(top):
        down_dip[offset] *= (offset + 1) / top
    for offset in range(bottom):
        down_dip[cells_i - 1 - offset] *= (offset + 1) / bottom

    return field * down_dip[:, None] * along_strike[None, :]


def slip_pattern(
    mesh: Chart,
    params: SlipParams,
    rng: np.random.Generator,
    *,
    sampler: FieldSampler = von_karman_field,
    taper: Callable[[FloatArray, SlipParams], FloatArray] = taper_edges,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """S4, up to the moment: a non-negative, tapered, unit-ish slip pattern.

    .. code-block:: text

        Z  = a standardised von Karman field  zero mean, unit variance
        f  = 1 + cov * Z                  mean shift and spread, in one step
        f  = max(f, 0)                    truncation
        f  = taper_edges(f)               the edges are where the fault stops

    The mean *shift* is the whole simplification: dividing by the field's own mean,
    rescaling the residual to a target variation, and flipping the sign when the mean
    came out negative is exactly ``1 + cov * Z`` once the field is standardised first.
    The sign flip existed only to make the intermediate well-defined, and a field's
    sign is not physically determined; its structure is.

    The size is not set here. :func:`~rupture_generator.moment.scale_to_moment` does
    that, once, jointly across every segment.

    Parameters
    ----------
    mesh : Chart
        The segment, passed through to ``sampler`` and otherwise unread.
    params : SlipParams
    rng : np.random.Generator
        This stage's own substream.
    sampler : FieldSampler, optional
        How to draw a field of this covariance on this chart. See
        :data:`FieldSampler`.
    taper : callable, optional
        ``taper(field, params)``, ramping the field to zero at the fault's edges.
        Defaults to :func:`taper_edges`, the lattice form; a triangulated segment
        passes its own, because a taper is the one part of this stage that has to know
        what shape the chart is. Everything else here -- the mean shift, the spread,
        the truncation and the order they happen in -- is the model, and the model is
        one implementation.

    Returns
    -------
    tuple
        The pattern, the standardised Gaussian it came from, and that Gaussian
        **before standardising**. Later stages correlate against the unstandardised
        draw rather than redrawing, which is what makes "rise time follows slip" a
        statement about the slip this rupture actually has -- and they take the
        unstandardised one because standardising divides by a sample spread, which
        perturbs the blend by the estimator's error.

        Later stages correlate against **the Gaussian**, not against the
        truncated tapered pattern, because truncation is what breaks the affine
        relation the correlation is a statement about -- and they correlate against
        *this* Gaussian rather than a freshly drawn one, which is what makes "rise
        time follows slip" a statement about the slip this rupture actually has.
    """
    drawn = sampler(mesh, params.covariance, rng)
    gaussian = standardise(drawn)
    pattern = 1.0 + params.coefficient_of_variation * gaussian
    pattern = np.maximum(pattern, 0.0)
    return taper(pattern, params), gaussian, drawn


def truncated_fraction(gaussian: FloatArray, params: SlipParams) -> float:
    """What fraction of the fault the truncation clipped.

    A diagnostic worth keeping: a large value says the requested variation was not
    achievable and the delivered spectrum is distorted. At the production spread of
    0.75 the expectation is about 9%.
    """
    return float(np.mean(1.0 + params.coefficient_of_variation * gaussian < 0.0))


# ============================================================================
# S5 -- rise time
# ============================================================================


@dataclasses.dataclass(frozen=True)
class RiseTimeParams:
    """How long each subfault slips for.

    Attributes
    ----------
    average_s : float
        The fault-wide mean rise time, in seconds -- from the moment, in
        :func:`average_rise_time_s`.
    correlation : float
        How strongly the rise-time field follows slip's own Gaussian. A patch that
        slips more slips for longer, because the two are set by the same dynamics.
    sigma : float
        The rise-time field's coefficient of variation, dimensionless.
    slip_exponent : float
        The power law: rise time goes as slip to this power. 0.5 is Graves & Pitarka
        -- rise time as the square root of slip. 1.0 would be constant slip rate.
    shallow_blend : DepthRamp
        Above this the field is replaced by slip itself. Near the surface a
        correlated field can hand a subfault appreciable slip and near-zero rise
        time, which is a physically absurd acceleration; slip cannot do that.
    shallow_stretch, deep_stretch : DepthRamp
    shallow_factor, deep_factor : float
        Rise time is longer near the surface and longer again at depth, unstretched
        between.
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
        """The depth stretch: ``1 + excess`` outside the ramps, exactly 1 between.

        Each branch measures from its ramp's far end, which is what makes the value
        exactly one at both inner edges. The asymmetry is the original's and is
        right.
        """
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

    Rise time grows with the cube root of moment -- linearly with fault dimension.
    The scale constant carries the units: the published ``1e-9`` is per cube-root
    dyne-centimetre, and this is its newton-metre equivalent, ``1e-9 * (1e7)^(1/3)``.

    ``geometric_correction`` is the same :func:`~rupture_generator.timing.alpha_t`
    the rupture speed uses. It shortens the pulse and raises the speed by the same
    factor, which is why the two stages must call one implementation.
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
    The mean is the requested average **by construction** -- the normalising constant
    is the mean of the stretched pattern, so dividing by it and multiplying by the
    average closes the identity exactly, except where the floor binds.

    That floor is physics, not slack: a pulse shorter than one sample cannot be
    represented, so the shortest subfaults are raised to one sample and the realised
    mean sits a little above the target.
    """
    depth_km = mesh.centres()[..., 2]

    # Correlated with slip's Gaussian rather than with the slip itself: both are
    # standardised by construction, so getting the two fields onto one scale is free.
    # The realised correlation is slightly higher than correlating against the
    # truncated, tapered, moment-scaled field would give, because it is no longer
    # being measured through the truncation.
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


# ============================================================================
# S6 -- rake
# ============================================================================


@dataclasses.dataclass(frozen=True)
class RakeParams:
    """Which way each subfault slips.

    ``sigma_deg`` is in **degrees**, unlike the slip field's dimensionless spread.
    """

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
    family. **Not correlated with slip**: a patch that slips more has no reason to
    slip in a different direction.

    Parameters
    ----------
    mesh : Chart
        The segment, passed through to ``sampler`` and otherwise unread.
    params : RakeParams
    rng : np.random.Generator
        This stage's own substream.
    sampler : FieldSampler, optional
        How to draw a field of this covariance on this chart.

    Returns
    -------
    FloatArray
        Degrees, one per subfault.
    """
    field = standardise(sampler(mesh, params.covariance, rng))
    return params.base_rake_deg + params.sigma_deg * field


# ============================================================================
# S8 -- onset perturbation
# ============================================================================


@dataclasses.dataclass(frozen=True)
class OnsetParams:
    """How the rupture front's arrival is perturbed away from the wavefront.

    Attributes
    ----------
    scale_s : float
        Seconds per unit of perturbation, and **negative** in production (-0.35).
        High-slip patches correlate positively with the perturbation field, so a
        negative scale makes them rupture systematically *earlier*. A sign error here
        produces a perfectly plausible rupture that is physically backwards.
    correlation : float
        How strongly the perturbation follows slip. Read by
        :func:`onset_perturbation`, which draws the field; the other two are read by
        :func:`apply_perturbation`, which spends it. One object because they are one
        model, split across two functions because the draw needs slip's own field
        application needs the travel times.
    sigma : float
        The perturbation field's standard deviation, dimensionless; its product with
        ``scale_s`` is the perturbation's spread in seconds.
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

    Dimensionless and standardised. The seconds, the pinning and the clamp belong to
    :func:`apply_perturbation`, which needs the travel times and the hypocentre; this
    needs slip's own draw. Splitting there is what lets every drawn field be drawn
    in one pass over the segments while the wavefront is solved later, in causal order,
    drawing nothing.

    Correlated against slip's own **draw**, rather than against the slip itself -- the
    same argument :func:`rise_time_field` makes, and the reason both take the field the
    Gaussian was standardised from rather than the tapered pattern.

    Parameters
    ----------
    mesh : Chart
        The segment, passed through to ``sampler`` and otherwise unread.
    slip_draw : FloatArray
        Slip's own unstandardised draw.
    params : OnsetParams
    rng : np.random.Generator
        This stage's own substream.
    covariance : VonKarmanFilterParameters
        The segment's patch structure.
    sampler : FieldSampler, optional
        How to draw a field of this covariance on this chart.

    Returns
    -------
    FloatArray
        Dimensionless, standardised, one value per subfault.
    """
    independent = sampler(mesh, covariance, rng)
    return standardise(
        correlate_fields(slip_draw, independent, params.correlation)
    )


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
    travel time plus the delay. Subtracting the perturbed field's global minimum
    instead does *not* pin it: with a non-zero scale some other cell can be earlier,
    and then the cell the rupture started from is not the earliest thing in the file.
    Pinning is what makes the registration assertable, and registration is easy to get
    wrong invisibly -- a hypocentre one cell off in each direction gives onset fields
    correlating 0.92 to 0.997 with the truth while differing by up to a second.

    Note what is *not* asserted: that the hypocentre is the global minimum of the
    perturbed field. It is not, and cannot be without clamping every other cell.
    Causality is a property of the travel times, which S7 owns.

    A pure function of its arguments: given the perturbation, nothing here draws. That
    is what makes the causal traversal deterministic, and why a point source -- whose
    perturbation is a field of zeros rather than a missing one -- needs no branch here.

    Parameters
    ----------
    hypocentre : tuple of int, int, or None
        The subfault the rupture starts from -- the ``(i, j)`` cell of a lattice or the
        flat face index of a triangulation, since either indexes its own field the same
        way -- or ``None`` for a segment triggered from elsewhere, whose onsets stay
        absolute, which is what lets a multi-segment rupture propagate rather than
        restart on every fault.
    delay_s : float
        A constant offset added to every subfault -- what the hypocentre's own onset
        becomes. Passed rather than carried on ``params`` because it is the *caller's*
        per-segment decision: the root takes the configured delay, and a segment
        triggered from elsewhere takes zero.
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
    # example, which is a subfault radiating before the event it belongs to.
    #
    # Clamping is also what makes the pinning assertable both ways: with it, the
    # hypocentre's onset is exactly the delay *and* the delay is the minimum of the
    # field. Without it only the first is true, and "the rupture starts at the
    # hypocentre" stops being a statement about the file.
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
