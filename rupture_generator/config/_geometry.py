from dataclasses import dataclass, field

from ._core import _ValidateMixin
from .validation import is_proportion


@dataclass
class Tapering(_ValidateMixin):
    side: float = field(
        metadata=dict(alias="side_taper", validator=is_proportion),
        doc="Fraction of along-strike length used to taper slip to zero at the lateral fault edges",
    )
    bottom: float = field(
        metadata=dict(alias="bot_taper", validator=is_proportion),
        doc="Fraction of down-dip width used to taper slip to zero at the bottom edge.",
    )
    top: float = field(
        metadata=dict(alias="top_taper", validator=is_proportion),
        doc="Fraction of down-dip width used to taper slip to zero at the top (shallow)",
    )


@dataclass
class SpatialFiltering(_ValidateMixin):
    roughness_min_wavelength: float | None = field(
        default=None,
        metadata=dict(alias="lambda_min"),
        doc="Minimum wavelength for slip correlation (null = no limit).",
    )
    roughness_max_wavelength: float | None = field(
        default=None,
        metadata=dict(alias="lambda_max"),
        doc="Maximum wavelength for slip correlation (null = no limit).",
    )
    rake_min_wavelength: float | None = field(
        default=None,
        metadata=dict(alias="wavelength_min"),
        doc="Minimum wavelength considered when band-pass filtering the rake spectral distribution (only if fractal_rake=0).",
    )
    # `wavelength_max` is deliberately absent. genslip v5.6.2 parses it
    # (genslip_v5.6.2.c:1092) and then overwrites it unconditionally with 1.0e+15 at
    # line 1236 ("hardwire for now 2016-10-21"), so no user value can ever reach the
    # filters. Exposing it would be a knob that silently does nothing.


@dataclass
class FaultGeometryLimits(_ValidateMixin):
    along_strike_length: float | None = field(
        default=None,
        metadata=dict(alias="flen_max"),
        doc="Seems to be a maximum along-strike length to generate SRF output",
    )
    downdip_width: float | None = field(
        default=None,
        metadata=dict(alias="fwid_max"),
        doc="Seems to be a maximum down-dip width to generate SRF output",
    )
    extension_factor: float | None = field(
        default=None,
        metadata=dict(alias="extend_fac"),
        doc="geometric scaling variable used during multi-segment fault simulations to adjust how spatial wavenumber spectra are evaluated."
        " It scales individual segment dimensions up to the full aggregate dimensions of the parent multi-segment fault system,"
        " ensuring long-wavelength features are not artificially truncated.",
    )
