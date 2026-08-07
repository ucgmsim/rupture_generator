from dataclasses import dataclass, field

from ._core import _ValidateMixin
from .validation import is_non_negative, is_positive


@dataclass
class BetaParameters(_ValidateMixin):
    shallow: float = field(
        metadata=dict(alias="beta_shal", validator=is_positive),
        doc="Fraction of the rise time used in the computation of the slip-rate function for shallow subfaults. Default value according to Eq. 5 in Graves and Pitarka (2022).",
    )
    deep: float = field(
        metadata=dict(alias="beta_deep", validator=is_positive),
        doc="Fraction of the rise time used in the computation of the slip-rate function used for deep subfaults. Default value according to Eq. 5 in Graves and Pitarka (2022).",
    )
    mid: float = field(
        metadata=dict(alias="beta_mid", validator=is_positive),
        doc="Fraction of the rise time used in the computation of the slip-rate function for mid-crust depths. Default value according to Eq. 5 in Graves and Pitarka (2022).",
    )
    asperity: float = field(
        metadata=dict(alias="beta_asp", validator=is_positive),
        doc="Minimum value of $\\beta$ applied to asperity patches. This functionality is not used (as asp_mask = 0).",
    )
    sub_event: float = field(
        metadata=dict(alias="beta_subevt", validator=is_positive),
        doc="Minimum value of $\\beta$ enforced on subevent patches. This functionality is not used (as subevt_mask = 0).",
    )
    shallow_depth: float = field(
        metadata=dict(alias="beta_shal_depth", validator=is_positive),
        doc="Center depth of the shallow transition zone (km).",
    )
    shallow_depth_range: float = field(
        metadata=dict(alias="beta_shal_depth_range", validator=is_positive),
        doc="Half-width of the shallow transition zone (km).",
    )
    mid_depth: float = field(
        metadata=dict(alias="beta_mid_depth", validator=is_positive),
        doc="Center depth of the mid-to-deep transition zone (km). Default value according to Eq. 5 in Graves and Pitarka (2022).",
    )
    mid_depth_range: float = field(
        metadata=dict(alias="beta_mid_depth_range", validator=is_positive),
        doc="Half-width of the mid-to-deep transition zone (km).",
    )


@dataclass
class RiseTimeParameters(_ValidateMixin):
    coefficient: float = field(
        metadata=dict(alias="risetime_coef", validator=is_positive),
        doc="Constant used to scale the average slip rise time with seismic moment.",
    )
    shallow_factor: float = field(
        metadata=dict(alias="risetimefac", validator=is_positive),
        doc="Rise time factor for generic_slip2srf",
    )
    shallow_center_depth: float = field(
        metadata=dict(alias="risetimedep", validator=is_positive),
        doc="Rise time depth dependency for generic_slip2srf",
    )
    shallow_half_width: float = field(
        metadata=dict(alias="risetimedep_range", validator=is_positive),
        doc="Half-width of the shallow transition zone (km) used for local rise time",
    )
    # `rt_rand` is deliberately absent. It is read into `stfparams` at
    # genslip_v5.6.2.c:868 but the only code that reads it back is
    # `load_slip_srf_dd2` (gslip_srf_subs.c:677), which nothing calls -- main uses
    # `load_slip_srf_dd5_vsden` (genslip_v5.6.2.c:2964). Rise-time perturbation is
    # supplied by the correlated `rtime1`/`rtime2` fields instead.
    slip_scaling_factor: float = field(
        metadata=dict(alias="rt_scalefac", validator=is_positive),
        doc="A value of 1 implies that local rise time is scaled with sqrt(slip), consistent with GP10 Eq.7.",
    )
    deep_factor: float = field(
        metadata=dict(alias="deep_risetimefac", validator=is_positive),
        doc="Risetime adjustment factor",
    )
    deep_center_depth: float = field(
        metadata=dict(alias="deep_risetimedep", validator=is_positive),
        doc="Sets the midpoint for the deep rise time adjustment zone.",
    )
    deep_half_width: float = field(
        metadata=dict(alias="deep_risetimedep_range", validator=is_positive),
        doc="Sets the range for the deep rise time adjustment zone.",
    )


@dataclass
class RiseTimePerturbation(_ValidateMixin):
    level1_slip_correlation: float = field(
        metadata=dict(alias="rtime1_scor"),
        doc="Correlation coefficient between slip and rise time. set between 0 (uncorrelated) and 1 (perfectly correlated)",
    )
    level2_roughness_correlation: float = field(
        metadata=dict(alias="rtime2_scor"),
        doc="Correlation coefficient for additional perturbation of rise time with roughness.",
    )
    level2_slip_exponent: float = field(
        metadata=dict(alias="rtime2slip_exp"),
        doc="Adjusts the power for the correlation between rise time and slip. Rise time is correlated with slip^p, defaulting to sqrt(slip).",
    )
    level1_sigma: float | None = field(
        default=None,
        metadata=dict(alias="rtime1_sigma"),
        doc="The coefficient of variation of the risetime distribution. The default value is slip_sigma.",
    )
    level1_depth: float | None = field(
        default=None,
        metadata=dict(alias="rtime1_depth"),
        doc="This value sets the midpoint of the transition range for risetime-slip scaling. The default value is set to beta_shal_depth.",
    )
    level1_depth_range: float | None = field(
        default=None,
        metadata=dict(alias="rtime1_depth_range"),
        doc="The default value is set to beta_shal_depth_range. See rtime1_depth for a description.",
    )
    roughness_correlation_enabled: float | None = field(
        default=None,
        metadata=dict(alias="rtime_rand"),
        doc="If set, correlate rise time perturbations with roughness (not a boolean despite the name).",
    )
