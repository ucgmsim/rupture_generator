from dataclasses import dataclass, field

from ._core import _ValidateMixin
from .validation import is_positive


@dataclass
class RuptureTimePerturbation(_ValidateMixin):
    coefficient: float = field(
        metadata=dict(alias="tsfac_coef"),
        doc="Coefficient equal to 1.1 given in Eq. A2 from GP16 used to scale rupture time perturbation with seismic moment. Not used anymore in the code, as now this scaling is performed with tsfac_bzero and tsfac_slop",
    )
    intercept: float = field(
        metadata=dict(alias="tsfac_bzero"),
        doc="Offset constant value used when scaling the rupture time perturbation with seismic moment",
    )
    slope: float = field(
        metadata=dict(alias="tsfac_slope"),
        doc="Coefficient used to scale rupture time perturbation with seismic moment, similar to Eq A2 from GP16 but now adding an offset value of tsfac_bzero",
    )
    level1_sigma: float = field(
        metadata=dict(alias="tsfac1_sigma", validator=is_positive),
        doc="Adjusts stddev of risetime perturbations given by gaussian random numbers (via tsfac1_r array) with zero mean and sigma set to tsfac1_sigma*tsfac.",
    )
    level1_slip_correlation: float = field(
        metadata=dict(alias="tsfac1_scor"),
        doc="Allow specification of correlation levels between slip and rupture time. Correlation can be between 0 (uncorrelated) and 1.0 (1:1 correlation).",
    )
    level2_sigma: float = field(
        metadata=dict(alias="tsfac2_sigma", validator=is_positive),
        doc="The standard deviation of the rupture time perturbations due to roughness.",
    )
    level2_roughness_correlation: float = field(
        metadata=dict(alias="tsfac2_scor"),
        doc="Allow specification of correlation levels between roughness and rupture time. Correlation can be between 0 (uncorrelated) and 1.0 (1:1 correlation). (Not used as alpha_rough=0)",
    )
    level2_lambda_max: float = field(
        metadata=dict(alias="tsfac2_lambda_max"),
        doc="Maximum wavelength considered when band-pass filtering the wavenumber spectra associated with the fault roughness model, as explained in GP16. Value in km. (Not used as alpha_rough=0)",
    )
    main_value: float | None = field(
        default=None,
        metadata=dict(alias="tsfac_main"),
        doc="Depends on tsfac_bzero, tsfac_slope and moment",
    )
    level2_lambda_min: float | None = field(
        default=None,
        metadata=dict(alias="tsfac2_lambda_min"),
        doc="Minimum wavelength considered when band-pass filtering the wavenumber spectra associated with the fault roughness model, as explained in GP16. Value in km. (Not used as alpha_rough=0)",
    )


@dataclass
class FiniteDifferenceRupture(_ValidateMixin):
    enabled: bool = field(
        metadata=dict(alias="fdrup_time"),
        doc="Calculate rupture initiation times with wavefront propagation",
    )
    scale_speed_with_slip: bool = field(
        metadata=dict(alias="fdrup_scale_slip"),
        doc="Scale rslow with slip prior to computing FD times",
    )


@dataclass
class SegmentDelay(_ValidateMixin):
    enabled: bool = field(
        metadata=dict(alias="seg_delay"),
        doc="If true, enable per segment delays",
    )
    boundary_zone_width: list[float] = field(
        metadata=dict(alias="gwid"),
        doc="Width of per-segment delay zone, see rvfac_seg",
    )
    boundary_velocity_factor: list[float] = field(
        metadata=dict(alias="rvfac_seg"),
        doc="Per-segment delays for boundaries of segments",
    )
