"""Each segment's moment and rake, and the correlation lengths its magnitude implies.

A :class:`SegmentSource` is the part of a rupture that differs from segment to segment.
A :class:`~rupture_generator.rupture.models.RuptureModel` draws the fields from
it. The functions here derive sources from magnitudes, which is how a hazard model
states them. A :class:`CorrelationRelation` gives a model's noise its lengths at a
segment's magnitude. :func:`split_moment` shares one magnitude for a whole event between
its segments, by Bradley & Faulkner's apportionment.

References
----------
Bradley, B. A., & Faulkner, J. A. M. A kinematic rupture generator for multi-segment
fault geometries. Manuscript in preparation.

Hanks, T. C., & Kanamori, H. (1979). A moment magnitude scale. *Journal of Geophysical
Research*, 84(B5), 2348-2350.

Mai, P. M., & Beroza, G. C. (2002). A spatial random field model to characterize
complexity in earthquake slip. *Journal of Geophysical Research*, 107(B11), 2308.

Suzuki, F., Kato, K., & Watanabe, T. (2022). Spatial complexity characteristics of fault
slip from Japanese earthquakes based on self-similar models. *Bulletin of the
Seismological Society of America*, 112(4), 1853-1870.
"""

import dataclasses
import itertools
import math

import numpy as np
import scipy as sp

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.geometry import Geometry
from rupture_generator.rupture.medium import Medium
from rupture_generator.rupture.realisation import Realisation

DYNE_CM_PER_NM = 1.0e7


def moment_from_magnitude(magnitude: float) -> float:
    """Convert a moment magnitude to seismic moment, by Hanks & Kanamori (1979) Eq. 7.

    :math:`\\log_{10} M_0 = 1.5 \\mathbf{M} + 16.05` with :math:`M_0` in dyne-cm.

    Parameters
    ----------
    magnitude : float
        The moment magnitude.

    Returns
    -------
    float
        The seismic moment in newton-metres.
    """
    return 10.0 ** (1.5 * magnitude + 16.05) / DYNE_CM_PER_NM


def magnitude_from_moment(moment_nm: float) -> float:
    """Convert a seismic moment to moment magnitude: :func:`moment_from_magnitude` inverted.

    Parameters
    ----------
    moment_nm : float
        The seismic moment in newton-metres.

    Returns
    -------
    float
        The moment magnitude.
    """
    return (np.log10(moment_nm * DYNE_CM_PER_NM) - 16.05) / 1.5


@dataclasses.dataclass(frozen=True)
class CorrelationRelation:
    """Slip correlation lengths that grow with magnitude.

    ``length = 10 ** (exponent * min(Mw, saturation) - offset)`` kilometres, along strike
    and down dip. The defaults are Mai & Beroza (2002)'s fit for the von Karman
    correlation, which never saturates.

    Parameters
    ----------
    strike_exponent : float
        How fast the length along strike grows with magnitude.
    strike_offset : float
        The length along strike's offset, in decades of kilometres.
    dip_exponent : float
        How fast the length down dip grows with magnitude.
    dip_offset : float
        The length down dip's offset, in decades of kilometres.
    strike_saturation : float
        The magnitude past which the length along strike stops growing.
    dip_saturation : float
        The magnitude past which the length down dip stops growing, such as the
        seismogenic width bounding it.
    """

    strike_exponent: float = 0.5
    strike_offset: float = 2.5
    dip_exponent: float = 1.0 / 3.0
    dip_offset: float = 1.5
    strike_saturation: float = np.inf
    dip_saturation: float = np.inf

    def lengths_km(self, magnitude: float) -> tuple[float, float]:
        """Compute the correlation lengths for a magnitude.

        Parameters
        ----------
        magnitude : float
            The moment magnitude.

        Returns
        -------
        tuple of float
            ``(down dip, along strike)`` in kilometres, the chart's axis order.
        """
        dip_magnitude = min(magnitude, self.dip_saturation)
        strike_magnitude = min(magnitude, self.strike_saturation)
        return (
            10.0 ** (self.dip_exponent * dip_magnitude - self.dip_offset),
            10.0 ** (self.strike_exponent * strike_magnitude - self.strike_offset),
        )


MAI_BEROZA = CorrelationRelation()
"""Mai & Beroza (2002): ``10^(Mw/2 - 2.5)`` along strike, ``10^(Mw/3 - 1.5)`` down dip."""

SUZUKI_2022 = CorrelationRelation(
    strike_offset=1.67, dip_exponent=0.5, dip_offset=1.69, dip_saturation=6.3
)
"""Suzuki et al. (2022): ``10^(Mw/2 - 1.67)`` along strike, ``10^(min(Mw, 6.3)/2 - 1.69)``
down dip. Smoother than Mai & Beroza, the shallow slip of Graves & Pitarka's hybrid."""

CYCLES_OFFSET = math.log10(2.0 * math.pi)
"""How many decades genslip's lengths overstate its patches by.

genslip filters its spectrum with ``1 + k^2 a^2``, ``k`` in cycles per kilometre, where a
von Karman correlation ``C(r / a)`` has the spectrum ``1 + 4 pi^2 k^2 a^2``. genslip's
patches are then those of a length ``2 pi`` shorter than the one it states.
"""


def as_genslip(relation: CorrelationRelation) -> CorrelationRelation:
    """Shorten a relation's lengths to the patches genslip draws from it.

    Parameters
    ----------
    relation : CorrelationRelation
        The relation as published, its lengths those of a von Karman correlation.

    Returns
    -------
    CorrelationRelation
        The relation with both offsets grown by :data:`CYCLES_OFFSET`.
    """
    return dataclasses.replace(
        relation,
        strike_offset=relation.strike_offset + CYCLES_OFFSET,
        dip_offset=relation.dip_offset + CYCLES_OFFSET,
    )


GENSLIP_MAI_BEROZA = as_genslip(MAI_BEROZA)
"""Mai & Beroza (2002) as genslip applies it, ``2 pi`` shorter than published."""

GENSLIP_SUZUKI_2022 = as_genslip(SUZUKI_2022)
"""Suzuki et al. (2022) as genslip applies it, ``2 pi`` shorter than published."""


@dataclasses.dataclass(frozen=True)
class SegmentSource:
    """One segment's moment and mean rake.

    Parameters
    ----------
    moment_nm : float
        The segment's seismic moment in newton-metres. Positive.
    rake_deg : float
        The segment's mean rake in degrees.
    """

    moment_nm: float
    rake_deg: float

    def __post_init__(self) -> None:
        """Refuse a moment of zero or less."""
        if not self.moment_nm > 0.0:
            raise RuptureGeneratorError(
                f"a segment's moment must be positive, got {self.moment_nm}"
            )

    @property
    def magnitude(self) -> float:
        """float: The segment's moment magnitude."""
        return magnitude_from_moment(self.moment_nm)


def segment_source(magnitude: float, rake_deg: float) -> SegmentSource:
    """Build a segment's source from its magnitude.

    Parameters
    ----------
    magnitude : float
        The segment's moment magnitude.
    rake_deg : float
        The segment's mean rake in degrees.

    Returns
    -------
    SegmentSource
        The moment the magnitude gives, and the rake.
    """
    return SegmentSource(moment_from_magnitude(magnitude), rake_deg)


COMPONENT_SEPARATION_KM = 5.0
"""Two segments closer than this below :data:`COMPONENT_DEPTH_KM` may share a component."""

COMPONENT_DIP_DELTA_DEG = 20.0
"""Two segments whose dips differ by this much or more are never in one component."""

COMPONENT_DEPTH_KM = 5.0
"""The depth below which a component's separation is measured, in kilometres."""

MOMENT_SIGMA_LN = 0.85
"""The standard deviation of a component's moment about its share, natural-log units.

Bradley & Faulkner calibrate it from 238 components of 54 multi-component SRCMOD
ruptures, as the spread of ``ln(M_c / M_T)`` about the share of
:func:`component_weights`, each rupture's residuals centred. It shows no dependence on
the component's share of the area.
"""


def _occupied_nodes(chart: Geometry) -> np.ndarray:
    """Flag the nodes that are a corner of at least one fault cell, ``(n_i+1, n_j+1)``."""
    occupied = chart.occupied
    nodes = np.zeros((occupied.shape[0] + 1, occupied.shape[1] + 1), dtype=bool)
    for rows, columns in itertools.product((slice(None, -1), slice(1, None)), repeat=2):
        nodes[rows, columns] |= occupied
    return nodes


def _points_beneath(chart: Geometry, depth_km: float) -> np.ndarray:
    """Sample a chart's fault at and below a depth, ``(n, 3)`` positions in kilometres.

    The samples are the fault's nodes below ``depth_km``, and the point where each
    down-dip edge of a fault cell crosses ``depth_km``. The second set covers a cut-off
    that falls between two rows of nodes.
    """
    nodes = chart.nodes
    depth = nodes[..., 2]
    below = nodes[_occupied_nodes(chart) & (depth >= depth_km)]
    occupied = chart.occupied
    # A down-dip edge is a fault edge if a fault cell lies on either side of it.
    edge = np.zeros((occupied.shape[0], occupied.shape[1] + 1), dtype=bool)
    edge[:, :-1] |= occupied
    edge[:, 1:] |= occupied
    upper, lower = depth[:-1], depth[1:]
    crossing = edge & ((upper - depth_km) * (lower - depth_km) < 0.0)
    fraction = (depth_km - upper[crossing]) / (lower[crossing] - upper[crossing])
    cut = nodes[:-1][crossing] + fraction[:, None] * (
        nodes[1:][crossing] - nodes[:-1][crossing]
    )
    return np.concatenate([below, cut])


def _separation_km(
    a: Geometry, b: Geometry, depth_km: float, within_km: float
) -> float:
    """The distance between the closest points of two charts' faults below a depth.

    It caps the depth at 99% of the shallower chart's bottom, so a segment that stops
    short of ``depth_km`` still has its deepest part to measure from. A separation of
    ``within_km`` or more comes back as infinite, since a caller asks only whether the
    two are closer than that.
    """
    bottom_km = min(
        float(chart.nodes[..., 2][_occupied_nodes(chart)].max()) for chart in (a, b)
    )
    depth_km = min(depth_km, 0.99 * bottom_km)
    points_a, points_b = _points_beneath(a, depth_km), _points_beneath(b, depth_km)
    # Only points of a inside b's bounding box, grown by `within_km`, can lie that close
    # to b. On a long fault they're a small fraction of it.
    near = np.all(
        (points_a > points_b.min(axis=0) - within_km)
        & (points_a < points_b.max(axis=0) + within_km),
        axis=1,
    )
    if not near.any():
        return math.inf
    distance_km = sp.spatial.cKDTree(points_b).query(
        points_a[near], distance_upper_bound=within_km
    )[0]
    return float(distance_km.min())


def _mean_dip_deg(chart: Geometry) -> float:
    """The area-weighted mean dip of a chart's fault cells, in degrees."""
    areas = chart.areas_km2[chart.occupied]
    return float(np.sum(chart.dip_deg[chart.occupied] * areas) / np.sum(areas))


def moment_components(
    realisation: Realisation[Geometry],
    *,
    separation_km: float = COMPONENT_SEPARATION_KM,
    dip_delta_deg: float = COMPONENT_DIP_DELTA_DEG,
    depth_km: float = COMPONENT_DEPTH_KM,
) -> list[tuple[str, ...]]:
    """Group a fault system's segments into the components that share moment.

    Parameters
    ----------
    realisation : Realisation of Geometry
        The segments to group.
    separation_km : float
        The separation, in kilometres, below which two segments can join.
    dip_delta_deg : float
        The difference in dip, in degrees, below which two segments can join.
    depth_km : float
        The depth, in kilometres, below which the separation counts.

    Returns
    -------
    list of tuple of str
        Each component's segment names, in the realisation's order. The components are
        in order of their alphabetically first segment, so the list's order doesn't
        depend on the realisation's.
    """
    names = list(realisation)
    dips = {name: _mean_dip_deg(realisation[name]) for name in names}
    joined = sp.cluster.hierarchy.DisjointSet(names)
    for a, b in itertools.combinations(names, 2):
        # The dip test is cheap, so it goes first, and segments already joined skip both.
        if abs(dips[a] - dips[b]) >= dip_delta_deg or joined.connected(a, b):
            continue
        if (
            _separation_km(realisation[a], realisation[b], depth_km, separation_km)
            < separation_km
        ):
            joined.merge(a, b)
    components = [
        tuple(name for name in names if name in subset) for subset in joined.subsets()
    ]
    return sorted(components, key=min)


def rigidity_weighted_areas(
    realisation: Realisation[Geometry], medium: Medium
) -> dict[str, float]:
    """Sum ``mu * A`` over each segment's fault cells.

    Parameters
    ----------
    realisation : Realisation of Geometry
        The segments.
    medium : Medium
        The rock, which sets each cell's rigidity.

    Returns
    -------
    dict of str to float
        Each segment's rigidity-weighted area in pascal square kilometres, by name, in
        the realisation's order.
    """
    return {
        name: float(
            np.sum(
                (medium.rigidity_pa(chart.centres) * chart.areas_km2)[chart.occupied]
            )
        )
        for name, chart in realisation.items()
    }


def component_weights(
    realisation: Realisation[Geometry], segment_weights: dict[str, float]
) -> dict[tuple[str, ...], float]:
    """Weight each component by its segments' summed weight times its area to the half.

    With :func:`rigidity_weighted_areas` as ``segment_weights``, a component ``c`` weighs
    :math:`\\mu_c A_c^{3/2}`, Bradley & Faulkner's rule.

    Parameters
    ----------
    realisation : Realisation of Geometry
        The segments, grouped by :func:`moment_components`.
    segment_weights : dict of str to float
        Each segment's weight, by name.

    Returns
    -------
    dict of tuple of str to float
        Each component's weight, keyed by its segments' names, in the order of
        :func:`moment_components`.
    """
    return {
        names: sum(segment_weights[name] for name in names)
        * math.sqrt(
            sum(
                float(np.sum(realisation[name].areas_km2[realisation[name].occupied]))
                for name in names
            )
        )
        for names in moment_components(realisation)
    }


def split_moment(
    moment_nm: float,
    component_weights: dict[tuple[str, ...], float],
    segment_weights: dict[str, float],
    *,
    sigma_ln: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Split event moment between its segments, using Bradley & Faulkner (2026).

    Parameters
    ----------
    moment_nm : float
        The event's seismic moment in newton-metres.
    component_weights : dict of tuple of str to float
        Each component's weight, keyed by its segments' names, as
        :func:`component_weights` gives. The draws go to the components in this order.
    segment_weights : dict of str to float
        Each segment's weight within its component, by name, as
        :func:`rigidity_weighted_areas` gives.
    sigma_ln : float
        The standard deviation of each component's moment about its share, in
        natural-log units. Zero gives the rule's own shares.
    rng : numpy.random.Generator, optional
        The generator to draw each component's departure from. Needed for a positive
        ``sigma_ln``.

    Returns
    -------
    dict of str to float
        Each segment's moment in newton-metres, by name. The shares sum to
        ``moment_nm``.

    Raises
    ------
    RuptureGeneratorError
        If ``sigma_ln`` is negative or not finite, or positive with no ``rng``.
    """
    if not (math.isfinite(sigma_ln) and sigma_ln >= 0.0):
        raise RuptureGeneratorError(
            f"a moment's standard deviation is a non-negative number, got {sigma_ln}"
        )
    if sigma_ln > 0.0 and rng is None:
        raise RuptureGeneratorError(
            "a moment shared with a standard deviation needs a generator to draw from"
        )
    components = list(component_weights)
    if len(components) == 1:
        component_moments = [moment_nm]
    else:
        scaled = list(component_weights.values())
        if sigma_ln > 0.0:
            assert rng is not None
            departures = np.exp(rng.normal(0.0, sigma_ln, len(components)))
            scaled = [float(s * d) for s, d in zip(scaled, departures, strict=True)]
        total = sum(scaled)
        component_moments = [moment_nm * share / total for share in scaled]
    shares = {}
    for names, component_nm in zip(components, component_moments, strict=True):
        component_weight = sum(segment_weights[name] for name in names)
        for name in names:
            shares[name] = component_nm * segment_weights[name] / component_weight
    return {name: shares[name] for name in segment_weights}


__all__ = [
    "COMPONENT_DEPTH_KM",
    "COMPONENT_DIP_DELTA_DEG",
    "COMPONENT_SEPARATION_KM",
    "CYCLES_OFFSET",
    "GENSLIP_MAI_BEROZA",
    "GENSLIP_SUZUKI_2022",
    "MAI_BEROZA",
    "MOMENT_SIGMA_LN",
    "SUZUKI_2022",
    "CorrelationRelation",
    "SegmentSource",
    "as_genslip",
    "component_weights",
    "magnitude_from_moment",
    "moment_components",
    "moment_from_magnitude",
    "rigidity_weighted_areas",
    "segment_source",
    "split_moment",
]
