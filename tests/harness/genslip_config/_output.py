from dataclasses import dataclass, field

from ._core import _ValidateMixin
from .validation import is_positive


@dataclass
class AseismicParameters(_ValidateMixin):
    enabled: bool = field(
        metadata=dict(alias="aseis_flag"),
        doc="If true, enable aseismogenic creep adjustments.",
    )
    smooth: bool = field(
        metadata=dict(alias="aseis_smooth"),
        doc="If true, smooth aseismogenic factors with an averaging kernel of width 3.",
    )
    depth: float = field(
        metadata=dict(alias="aseis_dep", validator=is_positive),
        doc="Terminal depth for aseismic scaling region. slip above this depth is scaled down linearly, according to asei_fac. Below this depth no adjust occurs.",
    )
    factor: float | None = field(
        default=None,
        metadata=dict(alias="aseis_fac"),
        doc="Global setting for aseismic slip adjustment. Can be per subfault if read_aseis is set. only used if aseis_flag is true",
    )


@dataclass
class MagnitudeArea(_ValidateMixin):
    intercept: float | None = field(
        default=None,
        metadata=dict(alias="mag_area_Acoef"),
        doc="the 'A' in M = A + B log10(area). only used if use_median_mag is true",
    )
    slope: float | None = field(
        default=None,
        metadata=dict(alias="mag_area_Bcoef"),
        doc="the 'B' in M = A + B log10(area). only used if use_median_mag is true",
    )


@dataclass
class OutputOptions(_ValidateMixin):
    write_srf: bool = field(
        doc="Output SRF",
    )
    write_gsf: bool = field(
        doc="Output geometry definition",
    )
    srf_version: str = field(
        doc="Version of SRF to output (1.0 = basic, 2.0 = rho and Vs, 3.0 = Vp, rho, Vs and optional moment tensor)",
    )
    print_command: bool = field(
        doc="If set, output SRF command to file",
    )
    print_seed: bool = field(
        doc="If set, output SRF seed to file",
    )
    dump_last_seed: bool = field(
        doc="If true, write the final state of the SRF random seed generator to seedfile",
    )
