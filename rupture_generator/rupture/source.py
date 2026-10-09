"""What each segment carries: its moment, its rake, and how big its slip patches are.

A :class:`SegmentSource` is the part of a rupture that differs from segment to segment;
everything they share is :class:`~rupture_generator.rupture.generator.RuptureSettings`.
The functions here derive sources from magnitudes, which is how a hazard model states
them.

References
----------
Hanks, T. C., & Kanamori, H. (1979). A moment magnitude scale. *Journal of Geophysical
Research*, 84(B5), 2348-2350.

Mai, P. M., & Beroza, G. C. (2002). A spatial random field model to characterize
complexity in earthquake slip. *Journal of Geophysical Research*, 107(B11), 2308.
"""

import dataclasses

import numpy as np

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.rupture.medium import Medium
from rupture_generator.rupture.realisation import Realisation
from rupture_generator.sampling import Correlation, Covariance, VonKarman

DYNE_CM_PER_NM = 1.0e7


def moment_from_magnitude(magnitude: float) -> float:
    """Seismic moment in newton-metres, from Hanks & Kanamori (1979) equation 7.

    :math:`\\log_{10} M_0 = 1.5 \\mathbf{M} + 16.05` with :math:`M_0` in dyne-cm.
    """
    return 10.0 ** (1.5 * magnitude + 16.05) / DYNE_CM_PER_NM


def magnitude_from_moment(moment_nm: float) -> float:
    """The inverse of :func:`moment_from_magnitude`."""
    return (np.log10(moment_nm * DYNE_CM_PER_NM) - 16.05) / 1.5


@dataclasses.dataclass(frozen=True)
class CorrelationRelation:
    """Slip correlation lengths that grow with magnitude.

    ``length = 10 ** (exponent * Mw - offset)`` kilometres, along strike and down dip.
    The defaults are Mai & Beroza (2002)'s fit for the von Karman correlation.
    """

    strike_exponent: float = 0.5
    strike_offset: float = 2.5
    dip_exponent: float = 1.0 / 3.0
    dip_offset: float = 1.5

    def lengths_km(self, magnitude: float) -> tuple[float, float]:
        """``(down dip, along strike)`` correlation lengths: the chart's axis order."""
        return (
            10.0 ** (self.dip_exponent * magnitude - self.dip_offset),
            10.0 ** (self.strike_exponent * magnitude - self.strike_offset),
        )


VON_KARMAN = VonKarman()
"""Mai & Beroza (2002)'s correlation function, at their median Hurst exponent."""

MAI_BEROZA = CorrelationRelation()
"""Mai & Beroza (2002): ``10^(Mw/2 - 2.5)`` along strike, ``10^(Mw/3 - 1.5)`` down dip."""


@dataclasses.dataclass(frozen=True)
class SegmentSource:
    """One segment's moment, mean rake, and slip covariance.

    ``covariance`` is the correlation the slip *pattern* carries, its lengths in the
    chart's axis order, ``(down dip, along strike)``.
    """

    moment_nm: float
    rake_deg: float
    covariance: Covariance

    def __post_init__(self) -> None:
        """Refuse a moment that is not one."""
        if not self.moment_nm > 0.0:
            raise RuptureGeneratorError(
                f"a segment's moment must be positive, got {self.moment_nm}"
            )


def segment_source(
    magnitude: float,
    rake_deg: float,
    relation: CorrelationRelation = MAI_BEROZA,
    correlation: Correlation = VON_KARMAN,
) -> SegmentSource:
    """A segment's source from its magnitude: the moment, and the lengths it implies."""
    return SegmentSource(
        moment_nm=moment_from_magnitude(magnitude),
        rake_deg=rake_deg,
        covariance=Covariance(correlation, relation.lengths_km(magnitude)),
    )


def split_moment(
    moment_nm: float, realisation: Realisation, medium: Medium
) -> dict[str, float]:
    """One event's moment shared between its segments.

    Each segment's share is in proportion to ``sum(mu * A)`` over its fault cells,
    which is what one mean slip across the whole event would give it.
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
    "MAI_BEROZA",
    "VON_KARMAN",
    "CorrelationRelation",
    "SegmentSource",
    "magnitude_from_moment",
    "moment_from_magnitude",
    "segment_source",
    "split_moment",
]
