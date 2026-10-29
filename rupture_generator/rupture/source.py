"""Each segment's moment and rake, and the correlation lengths its magnitude implies.

A :class:`SegmentSource` is the part of a rupture that differs from segment to segment.
A :class:`~rupture_generator.rupture.models.RuptureModel` draws the fields from
it. The functions here derive sources from magnitudes, which is how a hazard model
states them. A :class:`CorrelationRelation` gives a model's noise its lengths at a
segment's magnitude.

References
----------
Hanks, T. C., & Kanamori, H. (1979). A moment magnitude scale. *Journal of Geophysical
Research*, 84(B5), 2348-2350.

Mai, P. M., & Beroza, G. C. (2002). A spatial random field model to characterize
complexity in earthquake slip. *Journal of Geophysical Research*, 107(B11), 2308.

Suzuki, F., Kato, K., & Watanabe, T. (2022). Spatial complexity characteristics of fault
slip from Japanese earthquakes based on self-similar models. *Bulletin of the
Seismological Society of America*, 112(4), 1853-1870.
"""

import dataclasses
import math

import numpy as np

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


def split_moment(
    moment_nm: float, realisation: Realisation[Geometry], medium: Medium
) -> dict[str, float]:
    """Share one event's moment between its segments.

    Each segment's share is in proportion to ``sum(mu * A)`` over its fault cells,
    which is what one mean slip across the whole event would give it.

    Parameters
    ----------
    moment_nm : float
        The event's seismic moment in newton-metres.
    realisation : Realisation of Geometry
        The segments to share it between.
    medium : Medium
        The rock, which sets each cell's rigidity.

    Returns
    -------
    dict of str to float
        Each segment's moment in newton-metres, by name. The shares sum to
        ``moment_nm``.
    """
    weights = {
        name: float(
            np.sum(
                (medium.rigidity_pa(chart.centres) * chart.areas_km2)[chart.occupied]
            )
        )
        for name, chart in realisation.items()
    }
    total = sum(weights.values())
    return {name: moment_nm * weight / total for name, weight in weights.items()}


__all__ = [
    "CYCLES_OFFSET",
    "GENSLIP_MAI_BEROZA",
    "GENSLIP_SUZUKI_2022",
    "MAI_BEROZA",
    "SUZUKI_2022",
    "CorrelationRelation",
    "SegmentSource",
    "as_genslip",
    "magnitude_from_moment",
    "moment_from_magnitude",
    "segment_source",
    "split_moment",
]
