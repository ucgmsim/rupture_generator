"""The models a rupture draws its fields with, and what the generator gives them.

:func:`~rupture_generator.rupture.generator.generate_segment` asks five models for a
segment's fields, in a fixed order:

1. A :class:`SlipModel` gives slip, and the latent Gaussian field slip is monotone in.
2. A :class:`RiseModel` gives rise time, from slip's latent and the moment.
3. A :class:`RakeModel` gives rake.
4. A :class:`SpeedModel` gives the front's speed, from slip's latent.
5. A :class:`PerturbationModel` gives the onset displacement, from slip's latent and
   the moment, and how far from the seed it takes to grow in.

Slip's latent is the one dependency between models. A model that correlates with slip
mixes its own draw into :attr:`SlipField.latent`. Its draw comes from
:attr:`SlipField.sampler`. The mixed field then has the latent's covariance.

Every model that draws noise names it as a :data:`Noise`, which is one or more
:class:`NoiseFamily` crossfaded by position. :func:`embed` places it on a chart.
Graves & Pitarka's hybrid slip is two families, long correlation lengths near the
surface and short ones at depth.
Correlations with slip are between the latents: Gaussian-copula correlations, which
mean the same whatever slip's marginal is, a drawn slip's or a given one's.
:func:`gaussian_anamorphosis` gives any slip such a latent.

The defaults are Graves & Pitarka's model: :class:`DrawnSlip`, :class:`CorrelatedRise`,
:class:`GaussianRake`, :class:`ShearFractionSpeed` and :class:`CorrelatedPerturbation`.
"""

import dataclasses
import functools
from collections.abc import Callable, Mapping
from typing import Protocol

import numpy as np
import scipy as sp

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.geometry import CellArray, CellMask, Geometry
from rupture_generator.rupture.medium import (
    SpatialField,
    constant_field,
    ramp_field,
    rigidity_pa,
)
from rupture_generator.rupture.source import (
    MAI_BEROZA,
    SUZUKI_2022,
    CorrelationRelation,
    SegmentSource,
)
from rupture_generator.sampling import (
    HURST,
    NORMAL,
    Covariance,
    Grid,
    Marginal,
    MarginalFamily,
    PreCorrected,
    Sampler,
    VonKarman,
    mix,
    sampler,
    standardise,
)

M2_PER_KM2 = 1.0e6

CUBE_ROOT_DYNE_CM = 1.0e-9 * (1.0e7 ** (1.0 / 3.0))
"""``1e-9`` per cube-root dyne-centimetre."""

ALPHA_COEFFICIENT = 0.1
"""How much the dip-and-rake correction can move things, proportion (0, 1)."""

DIP_PLATEAU_DEG = 45.0
"""Below this dip the correction is at full strength, falling to nothing at vertical."""

REVERSE_RAKE_DEG = 90.0
"""Pure reverse slip, where the correction is at full strength."""

MINIMUM_VELOCITY_FRACTION = 0.25
RAYLEIGH_VELOCITY_FRACTION = 0.9194
"""The sub-Rayleigh ceiling for a mode-II crack; between it and the shear speed lies
the forbidden zone."""
MAXIMUM_VELOCITY_FRACTION = np.sqrt(2.0)
"""Burridge-Andrews: the fastest a supershear front travels."""

UNMODIFIED = constant_field(1.0)
"""Constant profile for unmodified spatial fields."""


# ------------------------------------------------------------ what models see


@dataclasses.dataclass(frozen=True, eq=False)
class SegmentContext:
    """What every model may read about the segment it draws on.

    Attributes
    ----------
    geometry : Geometry
        The segment's chart.
    source : SegmentSource
        The segment's moment and rake.
    shear_speed_km_s : CellArray
        The medium's shear speed at the cell centres, in kilometres per second.
    density_g_cm3 : CellArray
        The medium's density at the cell centres, in grams per cubic centimetre.
    """

    geometry: Geometry
    source: SegmentSource
    shear_speed_km_s: CellArray
    density_g_cm3: CellArray

    @property
    def rigidity_pa(self) -> CellArray:
        """CellArray: Rigidity in pascals at the cell centres."""
        return rigidity_pa(self.shear_speed_km_s, self.density_g_cm3)

    def moment_nm(self, slip_m: CellArray) -> float:
        """The moment of a slip on this segment's cells, in this rock.

        Parameters
        ----------
        slip_m : CellArray
            Slip, in metres.

        Returns
        -------
        float
            The moment, in newton-metres.
        """
        return float(
            np.sum(self.rigidity_pa * self.geometry.areas_km2 * M2_PER_KM2 * slip_m)
        )


@dataclasses.dataclass(frozen=True, eq=False)
class SlipField:
    """Slip, and the latent field the other models correlate with.

    Attributes
    ----------
    slip_m : CellArray
        Slip, in metres. The generator zeroes it off the fault.
    latent : CellArray
        A standard normal field that slip is monotone in.
    sampler : Sampler
        What the model drew ``latent`` from, or the nearest for a slip it didn't draw.
        A model that mixes its own draw into ``latent`` draws from this.
    statistics : Mapping of str to float
        Numbers the model reports about its draw.
    """

    slip_m: CellArray
    latent: CellArray
    sampler: Sampler
    statistics: Mapping[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, eq=False)
class Drawn:
    """One field a model drew, and the numbers it reports about it.

    Attributes
    ----------
    values : CellArray
        The field, on the chart's cells.
    statistics : Mapping of str to float
        Numbers the model reports about its draw, such as a mean it scaled to.
    """

    values: CellArray
    statistics: Mapping[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, eq=False)
class Perturbation:
    """The onset displacement, and the time it takes to grow in from the seed.

    Attributes
    ----------
    displacement_s : CellArray
        Each cell's shift off the coherent front, in seconds. Negative is early.
    blend_s : float
        The time since the seed over which the displacement grows to full strength,
        in seconds. At zero it applies in full from the seed onwards.
    statistics : Mapping of str to float
        Numbers the model reports about its draw.
    """

    displacement_s: CellArray
    blend_s: float
    statistics: Mapping[str, float] = dataclasses.field(default_factory=dict)


class SlipModel(Protocol):
    """Draws a segment's slip and its latent field."""

    def __call__(self, context: SegmentContext, rng: np.random.Generator) -> SlipField:
        """Draw the field on a segment.

        Parameters
        ----------
        context : SegmentContext
            The segment.
        rng : numpy.random.Generator
            The model's own generator.

        Returns
        -------
        SlipField
            Slip, in metres, and the latent field slip is monotone in.
        """
        ...


class RiseModel(Protocol):
    """Draws a segment's rise time, given slip and the moment it sets."""

    def __call__(
        self,
        context: SegmentContext,
        slip: SlipField,
        moment_nm: float,
        rng: np.random.Generator,
    ) -> Drawn:
        """Draw the field on a segment.

        Parameters
        ----------
        context : SegmentContext
            The segment.
        slip : SlipField
            The segment's slip, and its latent.
        moment_nm : float
            The moment the segment's slip sets, in newton-metres.
        rng : numpy.random.Generator
            The model's own generator.

        Returns
        -------
        Drawn
            Rise time in seconds.
        """
        ...


class RakeModel(Protocol):
    """Draws a segment's rake."""

    def __call__(self, context: SegmentContext, rng: np.random.Generator) -> Drawn:
        """Draw the field on a segment.

        Parameters
        ----------
        context : SegmentContext
            The segment.
        rng : numpy.random.Generator
            The model's own generator.

        Returns
        -------
        Drawn
            Rake in degrees.
        """
        ...


class SpeedModel(Protocol):
    """Draws the front's speed, given slip."""

    def __call__(
        self, context: SegmentContext, slip: SlipField, rng: np.random.Generator
    ) -> Drawn:
        """Draw the field on a segment.

        Parameters
        ----------
        context : SegmentContext
            The segment.
        slip : SlipField
            The segment's slip, and its latent.
        rng : numpy.random.Generator
            The model's own generator.

        Returns
        -------
        Drawn
            The speed in kilometres per second.
        """
        ...


class PerturbationModel(Protocol):
    """Draws the onset displacement, given slip and the moment it sets."""

    def __call__(
        self,
        context: SegmentContext,
        slip: SlipField,
        moment_nm: float,
        rng: np.random.Generator,
    ) -> Perturbation:
        """Draw the field on a segment.

        Parameters
        ----------
        context : SegmentContext
            The segment.
        slip : SlipField
            The segment's slip, and its latent.
        moment_nm : float
            The moment the segment's slip sets, in newton-metres.
        rng : numpy.random.Generator
            The model's own generator.

        Returns
        -------
        Perturbation
            The displacement in seconds, and the time it takes to grow in.
        """
        ...


# -------------------------------------------------------------------- helpers


def check_cells(
    geometry: Geometry,
    name: str,
    values: CellArray,
    accept: Callable[[CellArray], np.ndarray],
    wanted: str,
) -> CellArray:
    """Check that a field matches a chart's cells and that ``accept`` takes every value.

    Parameters
    ----------
    geometry : Geometry
        The chart.
    name : str
        What to call the field in an error.
    values : CellArray
        The field.
    accept : callable
        Whether each value is acceptable, cell by cell.
    wanted : str
        What ``accept`` wants, for an error.

    Returns
    -------
    CellArray
        The field, as double precision.

    Raises
    ------
    RuptureGeneratorError
        If the field has the wrong shape, or a value ``accept`` refuses.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape != geometry.cells:
        raise RuptureGeneratorError(
            f"{name} is shaped {values.shape} and the chart has {geometry.cells} cells"
        )
    if not np.all(accept(values)):
        raise RuptureGeneratorError(
            f"{name} runs {float(np.nanmin(values)):.3g} to "
            f"{float(np.nanmax(values)):.3g}; it wants {wanted}"
        )
    return values


def read_at_centres(
    geometry: Geometry,
    name: str,
    field: SpatialField,
    accept: Callable[[CellArray], np.ndarray],
    wanted: str,
) -> CellArray:
    """Read a spatial field at a chart's cell centres, and check it.

    Parameters
    ----------
    geometry : Geometry
        The chart.
    name : str
        What to call the field in an error.
    field : SpatialField
        The field, a function of position.
    accept : callable
        Whether each value is acceptable, cell by cell.
    wanted : str
        What ``accept`` wants, for an error.

    Returns
    -------
    CellArray
        The field at the cell centres.

    Raises
    ------
    RuptureGeneratorError
        If the field gives the wrong shape, or a value ``accept`` refuses.
    """
    return check_cells(geometry, name, field(geometry.centres), accept, wanted)


def unit_interval(values: CellArray) -> np.ndarray:
    """Whether each value lies in ``[0, 1]``.

    Parameters
    ----------
    values : CellArray
        The values.

    Returns
    -------
    numpy.ndarray
        One boolean per value.
    """
    return (values >= 0.0) & (values <= 1.0)


def positive(values: CellArray) -> np.ndarray:
    """Whether each value is finite and greater than zero.

    Parameters
    ----------
    values : CellArray
        The values.

    Returns
    -------
    numpy.ndarray
        One boolean per value.
    """
    return np.isfinite(values) & (values > 0.0)


def geometric_correction(geometry: Geometry, rake_deg: float) -> float:
    """Compute Graves & Pitarka's geometric correction ``alpha_T``, in ``[1/1.1, 1]``.

    It's exactly 1 for a vertical strike-slip fault, and it shortens the rise time and
    raises the rupture speed by the same factor. The dip is the mean over the fault
    cells. The rake wraps into ``[-180, 180]`` first. The correction is for reverse
    geometries, so normal faulting doesn't get one.

    Parameters
    ----------
    geometry : Geometry
        The segment's chart.
    rake_deg : float
        The segment's mean rake, in degrees.

    Returns
    -------
    float
        The correction.
    """
    dip_deg = float(np.mean(geometry.dip_deg[geometry.occupied]))
    if dip_deg <= DIP_PLATEAU_DEG:
        dip_factor = 1.0
    else:
        dip_factor = 1.0 - (dip_deg - DIP_PLATEAU_DEG) / (90.0 - DIP_PLATEAU_DEG)

    rake_deg = (rake_deg + 180.0) % 360.0 - 180.0
    if 0.0 <= rake_deg <= 180.0:
        rake_factor = 1.0 - abs(rake_deg - REVERSE_RAKE_DEG) / REVERSE_RAKE_DEG
    else:
        rake_factor = 0.0

    return 1.0 / (1.0 + dip_factor * rake_factor * ALPHA_COEFFICIENT)


def cube_root_scaling(coefficient: float, moment_nm: float) -> float:
    """``coefficient * M0^(1/3)``, with the coefficient in published dyne-cm units.

    Parameters
    ----------
    coefficient : float
        The coefficient, per cube-root dyne-centimetre at ``1e-9``.
    moment_nm : float
        The moment, in newton-metres.

    Returns
    -------
    float
        The scaled moment, in the coefficient's units.
    """
    return coefficient * CUBE_ROOT_DYNE_CM * float(np.cbrt(moment_nm))


def gaussian_anamorphosis(values: CellArray, occupied: CellMask) -> CellArray:
    """The standard normal scores of a field over the fault cells.

    Each fault cell's score is ``Phi^-1`` of its mid-rank, ``(rank - 1/2) / n``. Tied
    values share their mean rank, so a patch of equal values shares one score. The
    scores are monotone in the values, so a field correlated with them at a
    Gaussian-copula correlation has that correlation with the values themselves.

    Parameters
    ----------
    values : CellArray
        The field, such as a slip from elsewhere.
    occupied : CellMask
        The fault cells.

    Returns
    -------
    CellArray
        The scores on the fault cells, and zero off them.

    Examples
    --------
    >>> gaussian_anamorphosis(np.array([[1.0, 4.0, 2.0]]), np.ones((1, 3), bool))
    array([[-0.96742157,  0.96742157,  0.        ]])
    """
    ranks = sp.stats.rankdata(values[occupied], method="average")
    scores = np.zeros(np.shape(values), dtype=np.float64)
    scores[occupied] = sp.stats.norm.ppf((ranks - 0.5) / ranks.size)
    return scores


def _correlation(value: float) -> None:
    if not -1.0 <= value <= 1.0:
        raise RuptureGeneratorError(f"a correlation lies in [-1, 1], got {value}")


def _positive_marginal(
    what: str, family: MarginalFamily, spread: float, maximum: float
) -> Marginal:
    if family == "normal":
        raise RuptureGeneratorError(
            f"{what} values are positive, so they cannot follow a normal marginal"
        )
    return Marginal(family, spread, maximum)


# ----------------------------------------------------------------------- noise


@dataclasses.dataclass(frozen=True)
class NoiseFamily:
    """One stationary von Karman noise, and how much of a field it's where.

    Attributes
    ----------
    relation : CorrelationRelation
        The correlation lengths at the segment's magnitude.
    hurst : float
        The Hurst exponent.
    weight : SpatialField
        How much of the field this family is at each position, in ``[0, 1]``.
    """

    relation: CorrelationRelation = MAI_BEROZA
    hurst: float = HURST
    weight: SpatialField = UNMODIFIED


type Noise = tuple[NoiseFamily, ...]
"""Noise families crossfaded by position, each sharing one white noise."""

STATIONARY: Noise = (NoiseFamily(),)
"""Mai & Beroza (2002)'s lengths and Hurst exponent everywhere."""

GP_HYBRID: Noise = (
    NoiseFamily(SUZUKI_2022, weight=ramp_field(6.5, 1.5, 1.0, 0.0)),
    NoiseFamily(weight=ramp_field(6.5, 1.5, 0.0, 1.0)),
)
"""Graves & Pitarka's hybrid, as genslip v5.6.3 configures it by default.

Suzuki et al. (2022)'s lengths above 5 km and Mai & Beroza (2002)'s below 8 km, both
at the default Hurst exponent, crossfaded over ``6.5 +- 1.5`` km, genslip's
``hyb_corlen_dep`` and ``hyb_corlen_dep_range``.
"""


@functools.lru_cache(maxsize=128)
def embed(
    noise: Noise, geometry: Geometry, magnitude: float, marginal: Marginal = NORMAL
) -> Sampler:
    """Embed a noise on a chart, or return the embedding already made.

    The pre-correction for ``marginal`` gives each family its own correlation after the
    marginal. The cache keeps the last 128. Drawing the same fault system again doesn't
    embed. ``embed.cache_clear()`` releases the cache.

    Parameters
    ----------
    noise : Noise
        The families.
    geometry : Geometry
        The chart.
    magnitude : float
        The segment's moment magnitude, which sets each family's lengths.
    marginal : Marginal, optional
        The marginal the drawn field takes. Normal by default.

    Returns
    -------
    Sampler
        The embedding, ready to draw from.

    Raises
    ------
    RuptureGeneratorError
        If a weight reads out of ``[0, 1]``, or a family doesn't embed on the chart.
    """
    strike_km, dip_km = geometry.spacing_km
    return sampler(
        Grid(geometry.cells, (dip_km, strike_km)),
        [
            Covariance(
                PreCorrected(VonKarman(family.hurst), marginal),
                family.relation.lengths_km(magnitude),
            )
            for family in noise
        ],
        [
            read_at_centres(
                geometry, "a noise weight", family.weight, np.isfinite, "finite values"
            )
            for family in noise
        ],
    )


# ------------------------------------------------------------------------ slip


def _reach(mask: CellMask, axis: int, *, reverse: bool) -> np.ndarray:
    """How many occupied cells run up to each cell along one axis, inclusive.

    This counts the natural generalisation of rectangular fault based reach tapering.
    """
    ordered = np.flip(mask, axis) if reverse else mask
    index = np.arange(ordered.shape[axis], dtype=np.int32)
    index = index[:, None] if axis == 0 else index
    last_gap = np.maximum.accumulate(np.where(ordered, np.int32(-1), index), axis=axis)
    counted = index - last_gap
    return np.flip(counted, axis) if reverse else counted


@dataclasses.dataclass(frozen=True)
class DrawnSlip:
    """Slip drawn from a noise, with an exact marginal and tapered edges.

    The noise is pre-corrected for the marginal, so the slip itself has the noise's
    correlation lengths. The model scales slip to the source's moment.

    Attributes
    ----------
    family : MarginalFamily
        The unit-mean distribution slip values follow.
    coefficient_of_variation : float
        The spread of ``family``.
    maximum : float
        The largest slip on the fault in mean slips, for a ``doubly_truncated_normal``
        alone. Infinite otherwise.
    side_taper : float
        The fraction of the fault's length from each end over which slip ramps to zero.
    top_taper : float
        The fraction of the fault's width from the top edge over which slip ramps to
        zero. At zero, slip at the top edge has full amplitude.
    bottom_taper : float
        The fraction of the fault's width from the bottom edge over which slip ramps to
        zero.
    noise : Noise
        The noise slip's latent comes from, and rise time's independent part.
        Graves & Pitarka's hybrid, :data:`GP_HYBRID`, by default.
    """

    family: MarginalFamily = "truncated_exponential"
    coefficient_of_variation: float = 0.90
    maximum: float = np.inf
    side_taper: float = 0.02
    top_taper: float = 0.0
    bottom_taper: float = 0.0
    noise: Noise = GP_HYBRID

    def __post_init__(self) -> None:
        """Refuse a taper outside ``[0, 0.5]``, or a spread outside the family's range."""
        for name in ("side_taper", "top_taper", "bottom_taper"):
            if not 0.0 <= getattr(self, name) <= 0.5:
                raise RuptureGeneratorError(
                    f"{name} is a fraction of the fault's extent from one edge, so it "
                    f"lies in [0, 0.5]; got {getattr(self, name)}"
                )
        # _positive_marginal has validation logic, so invoking the property that
        # calls it triggers this validation.
        _ = self.marginal

    @property
    def marginal(self) -> Marginal:
        """Marginal: The unit-mean distribution that slip values follow."""
        return _positive_marginal(
            "slip", self.family, self.coefficient_of_variation, self.maximum
        )

    def _taper(self, field: CellArray, occupied: CellMask) -> CellArray:
        """Ramp a field to zero at the fault's edges, and zero the unoccupied cells.

        The taper is separable, a product of one ramp per edge in whole cells. The
        edges are the fault's and not the chart's. Where the ``occupied`` mask is
        partial, the ramps follow the outline of the occupied cells rather than the
        chart's rectangle.
        """
        ramp = np.ones(field.shape, dtype=np.float64)
        for fraction, axis, reverse in (
            (self.top_taper, 0, False),
            (self.bottom_taper, 0, True),
            (self.side_taper, 1, False),
            (self.side_taper, 1, True),
        ):
            cells = int(fraction * field.shape[axis] + 0.5)
            if cells:
                ramp *= np.minimum(_reach(occupied, axis, reverse=reverse) / cells, 1.0)
        return field * ramp * occupied

    def __call__(self, context: SegmentContext, rng: np.random.Generator) -> SlipField:
        """Draw slip on a segment.

        Parameters
        ----------
        context : SegmentContext
            The segment.
        rng : numpy.random.Generator
            The slip's own generator.

        Returns
        -------
        SlipField
            Slip scaled to the source's moment, and the latent it came from.

        Raises
        ------
        RuptureGeneratorError
            If the noise doesn't embed on this chart, or the taper removes all slip
            anywhere.
        """
        geometry = context.geometry
        noise = embed(self.noise, geometry, context.source.magnitude, self.marginal)
        latent, _ = noise.draw(rng)
        pattern = self._taper(self.marginal.apply(latent), geometry.occupied)
        total = context.moment_nm(pattern)
        if not total > 0.0:
            raise RuptureGeneratorError(
                "the slip pattern has no moment anywhere: every subfault was tapered "
                "or masked to zero"
            )
        return SlipField(
            slip_m=(context.source.moment_nm / total) * pattern,
            latent=latent,
            sampler=noise,
        )


# ------------------------------------------------------------------- rise time


@dataclasses.dataclass(frozen=True)
class CorrelatedRise:
    """Rise time correlated with slip, its mean following the moment.

    The fault-wide mean is ``coefficient * M0^(1/3) * alpha_T``, with ``alpha_T``
    from :func:`geometric_correction`. The pattern is a power of a field
    correlated with slip's latent, times a depth profile.

    Attributes
    ----------
    coefficient : float
        The mean's coefficient in the published units, per cube-root dyne-centimetre
        at ``1e-9``. Graves & Pitarka use 1.6.
    correlation : float
        The Gaussian-copula correlation of rise time with slip.
    slip_exponent : float
        The power of slip that rise time follows. At 0.5, rise time goes as the square
        root of slip.
    family : MarginalFamily
        The unit-mean distribution rise time values follow, a gamma by default.
    coefficient_of_variation : float
        The spread of ``family``. Below unit spread the gamma's mode is away from zero.
        No subfault then has a zero rise time.
    maximum : float
        The longest rise time in mean rise times, for a ``doubly_truncated_normal``
        alone. Infinite otherwise.
    factor : SpatialField
        The relative rise time.
    slip_weight : SpatialField
        How much of rise time's correlation with slip is ``correlation`` (1) and how
        much is exact (0). At 0 the rise-time latent is slip's own, and rise time then
        follows slip's rank order. This is the shallow treatment of Graves & Pitarka.
        There, in the velocity-strengthening crust, the pulse length tracks the slip.
    """

    coefficient: float = 1.6
    correlation: float = 0.9
    slip_exponent: float = 0.5
    family: MarginalFamily = "gamma"
    coefficient_of_variation: float = 0.75
    maximum: float = np.inf
    factor: SpatialField = UNMODIFIED
    slip_weight: SpatialField = UNMODIFIED

    def __post_init__(self) -> None:
        """Refuse a rise time the power law can't produce."""
        if self.slip_exponent <= 0.1:
            raise RuptureGeneratorError(
                f"a slip exponent of {self.slip_exponent} abandons the correlated field "
                "for independent noise, which is a different model; use one above 0.1"
            )
        _correlation(self.correlation)
        _ = self.marginal

    @property
    def marginal(self) -> Marginal:
        """Marginal: The unit-mean distribution that rise time values follow."""
        return _positive_marginal(
            "rise time", self.family, self.coefficient_of_variation, self.maximum
        )

    def __call__(
        self,
        context: SegmentContext,
        slip: SlipField,
        moment_nm: float,
        rng: np.random.Generator,
    ) -> Drawn:
        """Draw rise time on a segment.

        Parameters
        ----------
        context : SegmentContext
            The segment.
        slip : SlipField
            The segment's slip, and its latent.
        moment_nm : float
            The segment's moment, which sets the mean.
        rng : numpy.random.Generator
            The rise time's own generator.

        Returns
        -------
        Drawn
            Rise time in seconds, with the mean as ``mean_s`` and the correction as
            ``alpha_t``.

        Raises
        ------
        RuptureGeneratorError
            If a profile reads out of range on this chart.
        """
        geometry = context.geometry
        factor = read_at_centres(
            geometry, "the rise-time factor", self.factor, positive, "positive values"
        )
        slip_weight = read_at_centres(
            geometry,
            "the rise-time slip weight",
            self.slip_weight,
            unit_interval,
            "weights in [0, 1]",
        )
        correction = geometric_correction(geometry, context.source.rake_deg)
        mean_s = cube_root_scaling(self.coefficient, moment_nm) * correction
        rho = self.correlation
        latent = mix(
            (slip_weight * rho + (1.0 - slip_weight), slip.latent),
            (slip_weight * np.sqrt(1.0 - rho * rho), slip.sampler.draw(rng)[0]),
        )
        pattern = self.marginal.apply(latent) ** self.slip_exponent * factor
        return Drawn(
            pattern * (mean_s / float(pattern[geometry.occupied].mean())),
            {"mean_s": mean_s, "alpha_t": correction},
        )


# ------------------------------------------------------------------------ rake


@dataclasses.dataclass(frozen=True)
class GaussianRake:
    """Rake that strays from the source's by ``sigma * Z`` degrees.

    By design, rake doesn't depend on slip. A patch with more slip has no reason to
    move in a different direction. The marginal is normal, so rake keeps the noise's
    correlation lengths exactly, with no pre-correction.

    Attributes
    ----------
    sigma_deg : float
        The standard deviation of the rake about the segment's, in degrees.
    noise : Noise
        The noise rake strays by. Stationary by default, as in genslip.
    """

    sigma_deg: float = 15.0
    noise: Noise = STATIONARY

    def __post_init__(self) -> None:
        """Refuse a negative spread."""
        if self.sigma_deg < 0.0:
            raise RuptureGeneratorError(
                f"a spread has no sign, got {self.sigma_deg} degrees"
            )

    def __call__(self, context: SegmentContext, rng: np.random.Generator) -> Drawn:
        """Draw rake on a segment.

        Parameters
        ----------
        context : SegmentContext
            The segment.
        rng : numpy.random.Generator
            The rake's own generator.

        Returns
        -------
        Drawn
            Rake in degrees.
        """
        latent, _ = embed(self.noise, context.geometry, context.source.magnitude).draw(
            rng
        )
        return Drawn(context.source.rake_deg + self.sigma_deg * standardise(latent))


# ----------------------------------------------------------------------- speed


@dataclasses.dataclass(frozen=True)
class ShearFractionSpeed:
    """A front at a fraction of the shear speed, held off the forbidden zone.

    The front moves at ``velocity_fraction / alpha_T`` of the shear speed, held to the
    sub-Rayleigh or supershear branch, times a depth profile.

    Attributes
    ----------
    velocity_fraction : float
        The rupture speed as a fraction of the shear speed, before the geometric
        correction.
    factor : SpatialField
        The factor on the front's speed, for a slower front near the ground surface or
        at depth.
    """

    velocity_fraction: float = 0.8
    factor: SpatialField = UNMODIFIED

    def __post_init__(self) -> None:
        """Refuse a speed outside the allowed band."""
        if not 0.0 < self.velocity_fraction <= MAXIMUM_VELOCITY_FRACTION:
            raise RuptureGeneratorError(
                f"a velocity fraction lies in (0, {MAXIMUM_VELOCITY_FRACTION:.4f}], "
                f"got {self.velocity_fraction}"
            )

    def __call__(
        self, context: SegmentContext, slip: SlipField, rng: np.random.Generator
    ) -> Drawn:
        """Find the front's speed on a segment.

        Parameters
        ----------
        context : SegmentContext
            The segment.
        slip : SlipField
            The segment's slip, which this speed ignores.
        rng : numpy.random.Generator
            The speed's own generator, which this speed ignores.

        Returns
        -------
        Drawn
            The speed in kilometres per second, with the correction as ``alpha_t``.

        Raises
        ------
        RuptureGeneratorError
            If the speed factor reads out of range on this chart.
        """
        factor = read_at_centres(
            context.geometry,
            "the rupture speed factor",
            self.factor,
            positive,
            "positive values, or the front never arrives",
        )
        correction = geometric_correction(context.geometry, context.source.rake_deg)
        fraction = self.velocity_fraction / correction
        if fraction > RAYLEIGH_VELOCITY_FRACTION:
            fraction += 1.0 - RAYLEIGH_VELOCITY_FRACTION
        fraction = min(
            max(fraction, MINIMUM_VELOCITY_FRACTION), MAXIMUM_VELOCITY_FRACTION
        )
        return Drawn(
            fraction * context.shear_speed_km_s * factor, {"alpha_t": correction}
        )


# ----------------------------------------------------------------- perturbation


@dataclasses.dataclass(frozen=True)
class CorrelatedPerturbation:
    """An onset displacement correlated with slip, its spread following the moment.

    The spread is ``offset_s + coefficient * M0^(1/3)`` in the published units. With
    both terms zero the front is coherent.

    Attributes
    ----------
    offset_s : float
        The constant term of the spread, in seconds: genslip's ``tsfac_bzero``, read
        as a magnitude.
    coefficient : float
        The moment term's coefficient in the published units: genslip's
        ``tsfac_slope``, read as a magnitude.
    correlation : float
        The Gaussian-copula correlation of the displacement with slip. A positive
        value makes high-slip patches rupture early.
    blend_sigma : float
        The width of the zone over which the displacement grows in from the seed, in
        units of the spread.
    noise : Noise, optional
        The noise of the displacement's own draw. :data:`STATIONARY` by default, as
        genslip v5.6.3 draws it while slip's is hybrid. Slip's when ``None``.
    """

    offset_s: float = 0.1
    coefficient: float = 0.5
    correlation: float = 0.8
    blend_sigma: float = 4.0
    noise: Noise | None = STATIONARY

    def __post_init__(self) -> None:
        """Refuse a negative spread, or a blend of no width."""
        if self.offset_s < 0.0 or self.coefficient < 0.0:
            raise RuptureGeneratorError(
                "the onset spread's offset and coefficient are magnitudes"
            )
        if self.blend_sigma <= 0.0:
            raise RuptureGeneratorError(
                f"the onset blend spans {self.blend_sigma} sigma, which is no width"
            )
        _correlation(self.correlation)

    def __call__(
        self,
        context: SegmentContext,
        slip: SlipField,
        moment_nm: float,
        rng: np.random.Generator,
    ) -> Perturbation:
        """Draw the onset displacement on a segment.

        Parameters
        ----------
        context : SegmentContext
            The segment.
        slip : SlipField
            The segment's slip, and its latent.
        moment_nm : float
            The segment's moment, which sets the spread.
        rng : numpy.random.Generator
            The displacement's own generator.

        Returns
        -------
        Perturbation
            The displacement in seconds, with the spread as ``scale_s``.
        """
        scale_s = self.offset_s + cube_root_scaling(self.coefficient, moment_nm)
        # genslip's tsfac is negative and this package reads it as a magnitude, so the
        # displacement subtracts: with a positive correlation, high slip ruptures early.
        rho = self.correlation
        noise = slip.sampler
        if self.noise is not None:
            noise = embed(self.noise, context.geometry, context.source.magnitude)
        independent, _ = noise.draw(rng)
        displacement_s = -scale_s * standardise(
            mix((rho, slip.latent), (np.sqrt(1.0 - rho * rho), independent))
        )
        return Perturbation(
            displacement_s, self.blend_sigma * scale_s, {"scale_s": scale_s}
        )


# ----------------------------------------------------------------- the model


@dataclasses.dataclass(frozen=True)
class PulseSettings:
    """Settings for the slip-rate pulse each subfault's slip and rise time make.

    Attributes
    ----------
    dt_s : float
        The slip-rate sample interval, in seconds.
    beta : SpatialField, optional
        The Liu-Archuleta-Hartzell rising fraction, in ``(0, 0.5]``. Without it, every
        pulse is a single-sample impulse.
    """

    dt_s: float = 0.005
    beta: SpatialField | None = None

    def __post_init__(self) -> None:
        """Refuse a sample interval that samples nothing."""
        if not self.dt_s > 0.0:
            raise RuptureGeneratorError(f"a sample interval of {self.dt_s} s is none")


@dataclasses.dataclass(frozen=True)
class RuptureModel:
    """The models a segment's fields come from, one per field.

    Attributes
    ----------
    slip : SlipModel
        Slip, and the latent the others correlate with.
    rise : RiseModel
        Rise time.
    rake : RakeModel
        Rake.
    speed : SpeedModel
        The front's speed.
    perturbation : PerturbationModel
        The onset displacement off the coherent front.
    pulse : PulseSettings
        The slip-rate pulse's sampling and shape.
    """

    slip: SlipModel = dataclasses.field(default_factory=DrawnSlip)
    rise: RiseModel = dataclasses.field(default_factory=CorrelatedRise)
    rake: RakeModel = dataclasses.field(default_factory=GaussianRake)
    speed: SpeedModel = dataclasses.field(default_factory=ShearFractionSpeed)
    perturbation: PerturbationModel = dataclasses.field(
        default_factory=CorrelatedPerturbation
    )
    pulse: PulseSettings = dataclasses.field(default_factory=PulseSettings)


MODELS = tuple(
    field.name for field in dataclasses.fields(RuptureModel) if field.name != "pulse"
)
"""The models' names, in the order a segment draws them and spawns their generators.

The pulse shapes the slip rates and draws nothing, so it is no model.
"""
