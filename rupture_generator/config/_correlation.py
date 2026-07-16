from dataclasses import dataclass, field

from ._core import _ValidateMixin
from .enums import KModel
from .validation import is_non_negative, is_positive


@dataclass
class HybridCorrelationLength(_ValidateMixin):
    enabled: bool = field(
        metadata=dict(alias="hyb_corlen_flag"),
        doc="If enabled, incorporate hybrid correlation lengths, using two different correlation models for shallow and deep regions.",
    )
    kmodel: KModel = field(
        metadata=dict(alias="hyb_corlen_kmodel"),
        doc="Model selected for the shallow region in the hybrid slip approach. Default=5 is the Suzuki (2022) model.",
    )
    factor: float = field(
        metadata=dict(alias="hyb_corlen_fac", validator=is_positive),
        doc="When using the Mai and Beroza (2022) model for both the shallow and deep regions in the hybrid slip approach, hyb_corlen_fac is the multiplicative factor used for adjusting the correlation length in the shallow region",
    )
    center_depth: float = field(
        metadata=dict(alias="hyb_corlen_dep", validator=is_positive),
        doc="Center depth of the shallow-to-deep transition zone (km) used for the hybrid slip model",
    )
    center_depth_range: float = field(
        metadata=dict(alias="hyb_corlen_dep_range", validator=is_positive),
        doc="Half-width of the shallow-to-deep transition zone (km) used for the hybrid slip model",
    )
    side_taper: float = field(
        metadata=dict(alias="hyb_corlen_side_taper", validator=is_non_negative),
        doc="Side taper of hybrid correlation structure.",
    )
    shallow_weight_start: float = field(
        metadata=dict(alias="hyb_corlen_shal_wt_start"),
        doc="Parameter setting the weighting in the hybrid slip approach",
    )
    shallow_weight_end: float = field(
        metadata=dict(alias="hyb_corlen_shal_wt_end"),
        doc="Parameter setting the weighting in the hybrid slip approach",
    )
    deep_weight_start: float = field(
        metadata=dict(alias="hyb_corlen_deep_wt_start"),
        doc="Parameter setting the weighting in the hybrid slip approach",
    )
    deep_weight_end: float = field(
        metadata=dict(alias="hyb_corlen_deep_wt_end"),
        doc="Parameter setting the weighting in the hybrid slip approach",
    )


@dataclass
class CustomCorrelationCorners(_ValidateMixin):
    along_strike_corner: float | None = field(
        default=None,
        metadata=dict(alias="kx_corner"),
        doc="Corner wavenumber for along-strike correlation lengths.",
    )
    downdip_corner: float | None = field(
        default=None,
        metadata=dict(alias="ky_corner"),
        doc="Corner wavenumber for down-dip correlation lengths.",
    )
    along_strike_exponent: float | None = field(
        default=None,
        metadata=dict(alias="xmag_exponent"),
        doc="Sets correlation lengths according to custom relation: clen_s = exp(bigM*(xmag_exponent*mag - kx_corner))",
    )
    downdip_exponent: float | None = field(
        default=None,
        metadata=dict(alias="ymag_exponent"),
        doc="Sets correlation widths according to custom relation: clen_d = exp(bigM*(ymag_exponent*mag - ky_corner));",
    )
