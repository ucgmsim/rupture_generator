from dataclasses import dataclass, field

from ._core import _ValidateMixin
from .validation import is_positive


@dataclass
class RuptureVelocity(_ValidateMixin):
    """How fast the rupture front travels, and how that varies with depth.

    These are **not** the rise-time depth ramps, though they share their defaults --
    6.5/1.5 shallow and 17.5/2.5 deep, so the two agree until someone moves one. That
    coincidence hid a boundary gap (`DEFECTS.md` 13) until this group existed to
    exercise it.

    `fraction` carries a further trap: genslip divides it, and every per-subfault
    copy, by `alphaT` at `genslip_v5.6.2.c:1443-1445`. The port applies `alphaT` to
    rise time internally and takes the fraction as given, so `mapping.fault_grid`
    does that division. See `mapping.DEFAULT_VELOCITY_FRACTION`.
    """

    fraction: float = field(
        default=0.8,
        metadata=dict(alias="rvfrac", validator=is_positive),
        doc="Rupture speed as a fraction of local shear-wave speed, before the depth"
        " ramps and before the alphaT division. genslip's DEFAULT_VR_TO_VS_FRAC.",
    )
    shallow_factor: float = field(
        default=0.6,
        metadata=dict(alias="shal_vrup", validator=is_positive),
        doc="Rupture speed multiplier at the shallow end of the shallow ramp.",
    )
    shallow_center_depth: float = field(
        default=6.5,
        metadata=dict(alias="shal_vrup_dep", validator=is_positive),
        doc="Centre depth of the shallow rupture-speed transition (km). Independent"
        " of risetimedep despite sharing its default.",
    )
    shallow_half_width: float = field(
        default=1.5,
        metadata=dict(alias="shal_vrup_deprange", validator=is_positive),
        doc="Half-width of the shallow rupture-speed transition (km).",
    )
    deep_factor: float = field(
        default=0.6,
        metadata=dict(alias="deep_vrup", validator=is_positive),
        doc="Rupture speed multiplier at the deep end of the deep ramp.",
    )
    deep_center_depth: float = field(
        default=17.5,
        metadata=dict(alias="deep_vrup_dep", validator=is_positive),
        doc="Centre depth of the deep rupture-speed transition (km). Pushed down to"
        " the hypocentre depth when that is deeper (genslip_v5.6.2.c:2974-2977),"
        " using deep_vrup_deprange -- its own half-width, not the rise time's.",
    )
    deep_half_width: float = field(
        default=2.5,
        metadata=dict(alias="deep_vrup_deprange", validator=is_positive),
        doc="Half-width of the deep rupture-speed transition (km).",
    )


@dataclass
class Hypocentre(_ValidateMixin):
    """Where the rupture starts, in **kilometres** -- not proportions, not indices.

    Both fields were named and documented as proportions here, and are neither.
    genslip converts them to subfault indices itself, with
    `ixs = (int)((shypo + 0.5*flen)/dstk + 0.5)` and `iys = (int)(dhypo/ddip + 0.5)`
    (`genslip_v5.6.2.c:3001` and `:3018`), and refuses to write a rupture if they fall
    outside the fault (line 3155).

    The distinction is a trap rather than a crash. Read as proportions, a hypocentre
    lands somewhere else on the same fault and produces a perfectly plausible rupture
    -- which is why `mapping.hypocentre_indices` owns the conversion and
    `test_mapping.py` pins it against genslip's own arithmetic.
    """

    along_strike_km: float = field(
        metadata=dict(alias="shypo"),
        doc="Hypocentre position along strike, in km from the fault's CENTRE. Signed:"
        " it runs from -flen/2 to +flen/2, so 0.0 is the middle of the fault.",
    )
    down_dip_km: float = field(
        metadata=dict(alias="dhypo"),
        doc="Hypocentre position down dip, in km from the fault's TOP EDGE. Runs from"
        " 0 to fwid. Unset, genslip uses dhypo_frac*fwid = 0.75*fwid"
        " (genslip_v5.6.2.c:1627-1628).",
    )


@dataclass
class RuptureTimePerturbation(_ValidateMixin):
    # `tsfac_coef` is deliberately absent. It is the pre-v5.4.1 scaling coefficient
    # (GP16 eq. A2); in v5.6.2 it is declared and parsed (genslip_v5.6.2.c:561, 973)
    # and never read again -- `tsfac_main` is built from `intercept` and `slope`
    # instead (line 1256).
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
