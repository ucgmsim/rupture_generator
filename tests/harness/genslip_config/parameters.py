from dataclasses import dataclass, field
from typing import Any

from ._core import _unroll_dataclass, _ValidateMixin
from ._correlation import CustomCorrelationCorners, HybridCorrelationLength
from ._geometry import FaultGeometryLimits, SpatialFiltering, Tapering
from ._output import AseismicParameters, MagnitudeArea, OutputOptions
from ._rupture import (
    FiniteDifferenceRupture,
    Hypocentre,
    RuptureTimePerturbation,
    RuptureVelocity,
    SegmentDelay,
)
from ._stf import BetaParameters, RiseTimeParameters, RiseTimePerturbation
from .enums import KModel, RiseTimeNormalisation, SlipRateFunction
from .types import PointSourceParams
from .validation import is_non_negative, is_positive, is_proportion


@dataclass
class Parameters(_ValidateMixin):
    resolution: float = field(
        metadata=dict(validator=is_positive),
        doc="The resolution of the SRF discretisation.",
    )
    dt: float = field(
        metadata=dict(validator=is_positive),
        doc="SRF temporal resolution (timestep).",
    )
    alpha_rough: float = field(
        metadata=dict(validator=is_proportion),
        doc="Scalar indicating fault roughness (0 = disabled)",
    )
    perturb_subfault_location: bool = field(
        doc="Shift subfault location according to roughness (in addition to perturbing strike and dip). Only used if alpha_rough > 0",
    )
    slip_sigma: float = field(
        metadata=dict(validator=is_non_negative),
        doc="Target SRF slip CoV",
    )
    rake_sigma: float = field(
        metadata=dict(validator=is_positive),
        doc="Target rake std deviation (absolute value, degrees)",
    )
    fractal_rake: bool = field(
        doc="If enabled, uses a von Karman filter for rake (producing self-similar fractal rake).",
    )
    von_karman_order: int = field(
        metadata=dict(alias="kord", validator=is_positive),
        doc="Order of the band-pass filter applied to the roughness-correlated rupture time field (tsfac2). Read as an int by genslip (genslip_v5.6.2.c:1098).",
    )
    magnitude_clamp: float = field(
        metadata=dict(alias="magC"),
        doc="magnitude clamping for down-dip correlation lengths. only used if the KModel is Suzuki (5).",
    )
    kmodel: KModel = field(
        doc="Correlation lengths relationship. Defaults to Mai 2002.",
    )
    use_moment_magnitude: bool = field(
        metadata=dict(alias="use_Mw"),
        doc="If true, use Hanks and Kanamori (1979) Eq 4 for magnitude ('Mw'). Otherwise use Eq 7 ('M').",
    )
    use_median_mag: bool = field(
        doc="If true set the magnitude of the rupture according to area",
    )
    circular_average: bool = field(
        doc="if set, correlation lengths are equal in both directions. only used if KModel is Mai or Sommerville.",
    )
    modified_corners: bool = field(
        doc="another correlation model of unknown origin. overrides the value of KModel",
    )
    mai_weight: float = field(
        metadata=dict(alias="mai_wt", validator=is_proportion),
        doc="The weighting for Mai correlation model. Only used if KModel is set to hybrid Mai-Sommerville",
    )
    somerville_weight: float = field(
        metadata=dict(alias="somerville_wt", validator=is_proportion),
        doc="Parameter only used if kmodel is set as MAI_SOMERVILLE_HYBRID_FLAG, which is not the default value. weight of the Sommerville model in the hybrid Sommerville and Mai correlation model",
    )
    truncate_zero_slip: bool = field(
        doc="If true, truncates negative slip. This can skew the spectral distribution of slip but removes non-physical values.",
    )
    rupture_delay: float = field(
        metadata=dict(validator=is_non_negative),
        doc="Scalar delay to all rupture initiation times (s)",
    )
    rvfmin: float = field(
        metadata=dict(validator=is_positive),
        doc="Lower bound on how much faster rupture velocity can be than the average rupture velocity. Rupture velocity is adjusted to be faster in regions of high slip.",
    )
    rvfmax: float = field(
        metadata=dict(validator=is_positive),
        doc="Upper bound on how much faster rupture velocity can be than the average rupture velocity. Rupture velocity is adjusted to be faster in regions of high slip.",
    )
    xshift: float = field(
        doc="Shift phase of slip field",
    )
    yshift: float = field(
        doc="Shift phase of slip field",
    )
    read_erf: bool = field(
        doc="If true, read an ERF file.",
    )
    read_gsf: bool = field(
        doc="If set, read a geometry input definition",
    )
    asperity_taper_factor: float = field(
        metadata=dict(alias="asp_taper_fac", validator=is_non_negative),
        doc="Size (proportion) of the asperity patch taper width.",
    )
    svr_wt: RiseTimeNormalisation = field(
        doc="How the fault-wide rise-time normalisation constant rt_scalefac is averaged: unweighted, slip-weighted, or slip x rupture-velocity weighted (genslip_v5.6.2.c:2461-2466).",
    )

    hypocentre: Hypocentre
    rupture_velocity: RuptureVelocity
    tapering: Tapering
    beta: BetaParameters
    rise_time: RiseTimeParameters
    rise_time_perturbation: RiseTimePerturbation
    rupture_time_perturbation: RuptureTimePerturbation
    hybrid_correlation_length: HybridCorrelationLength
    aseismic: AseismicParameters
    finite_difference_rupture: FiniteDifferenceRupture
    segment_delay: SegmentDelay
    spatial_filtering: SpatialFiltering
    custom_correlation_corners: CustomCorrelationCorners
    magnitude_area: MagnitudeArea
    output: OutputOptions
    fault_geometry_limits: FaultGeometryLimits

    slip_time_function: SlipRateFunction | None = field(
        default=None,
        metadata=dict(alias="stype"),
        doc="Slip-rate function shape for genslip. None means genslip's own default, OliuP2. NOT the same vocabulary as PointSourceParams.stype, which goes to generic_slip2srf.",
    )
    slip_water_level: float | None = field(
        default=None,
        doc="Minimum background slip level given as a percentage of the average slip amount (basically fills-in very low/zero slip patches with long rise time low slip.",
    )
    target_slip_average: float | None = field(
        default=None,
        metadata=dict(alias="target_savg"),
        doc="Target slip average",
    )
    set_rake: float | None = field(
        default=None,
        doc="If set, fix rake at every location",
    )
    moment_fraction: float | None = field(
        default=None,
        doc="Scales seismic moment by fraction given",
    )
    point_source_params: PointSourceParams | None = field(
        default=None,
        doc="Parameters for point source approximation, if applicable",
    )

    def to_cmd(self) -> dict[str, Any]:
        return dict(_unroll_dataclass(self))
