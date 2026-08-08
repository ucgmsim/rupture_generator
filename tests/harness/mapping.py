"""genslip's `getpar` names, mapped onto the port's five spec groups.

**Nothing here is part of `rupture_generator`.** It is the other half of the
comparison: `genslip_reference.py` renders a `Parameters` as arguments for the binary,
and this renders the *same* `Parameters` as the five groups the library takes, so a
divergence between the two runs is a divergence in the physics rather than in what
each side was asked to compute.

# Why this is its own module, and its own commit

A wrong mapping and a wrong port are indistinguishable from the outside. Both produce
a reference SRF and a generated SRF that disagree, and neither says which side moved.
So every correspondence below is justified from the C by line number, and pinned by
`test_mapping.py` against the expression genslip actually evaluates -- not against
what the port happens to produce.

# The four correspondences that are not name-to-name

Each of these looked like a rename until the C said otherwise, and each would have
produced a plausible, wrong rupture:

- **`shypo` and `dhypo` are kilometres; the port takes subfault indices.** genslip
  measures `shypo` along strike *from the fault's centre* (so it is signed) and
  `dhypo` down dip *from the top edge*. `hypocentre_indices` is the conversion, and it
  is genslip's own, truncation and all (lines 3001, 3018).
- **The padded extents are `nstk2`/`ndip2`, and they are not the fault rounded up.**
  They are the fault scaled by `flen_max*extend_fac/flen` and then rounded up to even,
  where `extend_fac` defaults to **1.10** (line 1073). A 20x12 fault pads to 22x14.
- **The slip spectrum's wavelength band is `wavelength_min`/`wavelength_max`, not
  `lambda_min`/`lambda_max`.** The `lambda_*` pair belongs to the roughness field.
  Worse, `wavelength_max` is overwritten with `1.0e+15` at line 1235 regardless of
  what was passed, and `wavelength_min` defaults to a function of the subfault
  spacing rather than to a constant.
- **`velocity_fraction` carries genslip's `alphaT` division and the port's does
  not.** genslip applies `alphaT` to two things: the average rise time (line 1439)
  and the rupture-velocity fraction (lines 1443-1445). The port applies it to the
  first internally and takes the second as given, so the caller has to divide.

# What building this found in the boundary

Three genslip configurations had no spelling in the PyO3 boundary while
`crates/genslip` modelled all three: `kmodel=Frankel` routed to the Somerville corner
relation, `circular_average` absent entirely, and the rise-time and rupture-speed
depth ramps collapsed into one pair. They are `DEFECTS.md` 11-13, they are **fixed**,
and this module now maps all three -- which is the only reason there is a test that
would notice if they came back.

None of them was visible until something tried to drive the port from a full getpar
set. That is what a mapping is for.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from rupture_generator import (
    FaultGrid,
    Ramp,
    RiseTimeWeighting,
    SlipSpec,
    SourceSpec,
    SpectrumModel,
    TimingSpec,
    VelocityModel1D,
)
from tests.harness.genslip_config import KModel, Parameters, RiseTimeNormalisation
from tests.harness.gsf import RADIANS_PER_DEGREE, FloatArray, GsfSubfaults

# genslip's `extend_fac` default, applied when the parameter is negative
# (`genslip_v5.6.2.c:1073`). It is why a fault's padded extents exceed it by a tenth
# rather than matching it.
DEFAULT_EXTEND_FACTOR = 1.10

# `NTMAX` (`defs.h:8`), which genslip assigns to `stfparams.nt` at line 680 and never
# reads from getpar. The port's `max_samples` default is the same number.
MAX_SLIP_RATE_SAMPLES = 100_000

# `wavelength_max = 1.0e+15` (line 1235), assigned unconditionally after the getpar
# block that reads it. No user value reaches the filters, which is why
# `SpatialFiltering` does not expose one.
HARDWIRED_MAX_WAVELENGTH_KM = 1.0e15

# `DEFAULT_VR_TO_VS_FRAC` (`defs.h:32`). genslip has a `rvfrac` getpar; `Parameters`
# does not carry one, so every fixture runs at the default and this is that default.
DEFAULT_VELOCITY_FRACTION = 0.8

_KMODEL_TO_SPECTRUM = {
    KModel.SOMERVILLE: SpectrumModel.Somerville,
    KModel.MAI: SpectrumModel.Mai,
    KModel.FRANKEL: SpectrumModel.Frankel,
    KModel.MAI_SOMERVILLE: SpectrumModel.MaiSomerville,
    KModel.SUZUKI: SpectrumModel.Suzuki,
    KModel.INPUT_CORNERS: SpectrumModel.InputCorners,
}

_SVR_WT_TO_WEIGHTING = {
    RiseTimeNormalisation.UNWEIGHTED_MEAN: RiseTimeWeighting.Uniform,
    RiseTimeNormalisation.SLIP_WEIGHTED: RiseTimeWeighting.BySlip,
    RiseTimeNormalisation.SLIP_AND_RUPTURE_VELOCITY_WEIGHTED: (
        RiseTimeWeighting.BySlipAndRuptureSpeed
    ),
}


class UnmappableConfigurationError(ValueError):
    """A parameter set genslip accepts and the PyO3 boundary cannot express.

    Raised rather than approximated. An approximation here would make every
    downstream comparison meaningless in a way no assertion could detect.
    """


def _f32(value: float) -> float:
    """Round a Python float through float32, as storing it in a C `float` would."""
    return float(np.float32(value))


@dataclasses.dataclass(frozen=True)
class Derived:
    """The quantities genslip computes before any spec group can be built.

    genslip derives these in `main` between reading its parameters and generating
    anything, and several spec groups need the same ones -- so they are computed once
    here rather than three times with three chances to disagree.

    Attributes
    ----------
    moment_dyne_cm : float
        `mom`, from the magnitude and its scale coefficient (line 1250).
    alpha_t : float
        `alphaT`, the dip-and-rake correction (line 1438). The port recomputes this
        internally from `average_dip_deg` and `average_rake_deg`; it is here because
        `velocity_fraction` has to be divided by it *before* it crosses the boundary.
    length_km, width_km : float
        `flen` and `fwid`: the subfault spacing times the grid shape (lines 1227-1228).
        Not the GSF's extent -- genslip does not measure the fault, it multiplies.
    padded_strike, padded_dip : int
        `nstk2` and `ndip2` (lines 1471-1477).
    """

    moment_dyne_cm: float
    alpha_t: float
    length_km: float
    width_km: float
    padded_strike: int
    padded_dip: int


def seismic_moment(magnitude: float, use_moment_magnitude: bool) -> float:
    """`mom = exp(ln10 * 1.5 * (mag + mag_scale_coef))`.

    (orig. `genslip_v5.6.2.c:1250`, coefficient at 1077-1079)

    Parameters
    ----------
    magnitude : float
        `mag`.
    use_moment_magnitude : bool
        `use_Mw`. Selects 10.73 over 10.7 -- a 0.03 difference in the exponent's
        offset, which is 10% in moment.

    Returns
    -------
    float
        Seismic moment in dyne-cm.
    """
    coefficient = 10.73 if use_moment_magnitude else 10.7
    return _f32(math.exp(math.log(10.0) * 1.5 * (magnitude + coefficient)))


def alpha_t(average_dip_deg: float, average_rake_deg: float) -> float:
    """genslip's `alphaT`: the rise-time and rupture-speed geometry correction.

    Unity for a vertical strike-slip fault, which is the geometry GP2010 was
    calibrated on; below unity for everything else, shortening the pulse and speeding
    the rupture.

    This duplicates `genslip::source::geometry_correction`, which
    `crates/genslip/tests/source_parity.rs` pins bit-for-bit against the same
    expression. It is duplicated deliberately: using the port's own value to build the
    port's own input would make the comparison circular.

    (orig. `genslip_v5.6.2.c:1418-1438`)

    Parameters
    ----------
    average_dip_deg : float
        `avgdip`, as `GsfSubfaults.mean_dip_deg` derives it.
    average_rake_deg : float
        `avgrak`, as `GsfSubfaults.mean_rake_deg` derives it. Wrapped here.

    Returns
    -------
    float
        `alphaT`, in (0, 1].
    """
    dip_factor = 0.0
    if 45.0 < average_dip_deg <= 90.0:
        dip_factor = _f32(1.0 - (average_dip_deg - 45.0) / 45.0)
    elif 0.0 <= average_dip_deg <= 45.0:
        dip_factor = 1.0

    rake = average_rake_deg
    while rake < -180.0:
        rake += 360.0
    while rake > 180.0:
        rake -= 360.0

    rake_factor = 0.0
    if 0.0 <= rake <= 180.0:
        rake_factor = _f32(1.0 - math.sqrt((rake - 90.0) * (rake - 90.0)) / 90.0)

    return _f32(1.0 / (1.0 + _f32(dip_factor * rake_factor) * 0.1))


def padded_extents(
    strike_count: int,
    dip_count: int,
    length_km: float,
    width_km: float,
    parameters: Parameters,
) -> tuple[int, int]:
    """genslip's `nstk2` and `ndip2`: the wraparound extents the generators address.

    Not the fault rounded up to even. The fault is first *scaled* by
    `flen_max*extend_fac/flen`, where `extend_fac` defaults to 1.10, and the result is
    truncated to an int and then rounded up to even. For a 20x12 fault with no
    multi-segment limits that is 22x14, because `(int)(1.10*20)` is 22 and even while
    `(int)(1.10*12)` is 13 and odd.

    `flen_max` and `fwid_max` clamp up to `flen` and `fwid` (lines 1230-1233), so a
    single-plane fault always has a ratio of exactly one and the whole rule collapses
    to `extend_fac`. They are only larger when this segment is part of a bigger
    rupture, which is the case `extend_fac` exists to serve.

    The arithmetic runs in float32 because the C's does, and the truncation makes that
    visible: a ratio that computes as 21.999998 rather than 22.0 is a padded extent of
    22 rather than 24 once the round-to-even applies, which changes every wavenumber.

    (orig. `genslip_v5.6.2.c:1471-1477`)

    Parameters
    ----------
    strike_count, dip_count : int
        `nstk` and `ndip`.
    length_km, width_km : float
        `flen` and `fwid`.
    parameters : Parameters
        Read for `flen_max`, `fwid_max` and `extend_fac`.

    Returns
    -------
    tuple[int, int]
        `(nstk2, ndip2)`.
    """
    limits = parameters.fault_geometry_limits
    extend = limits.extension_factor
    if extend is None or extend < 0:
        extend = DEFAULT_EXTEND_FACTOR

    length_max = max(limits.along_strike_length or -1.0, length_km)
    width_max = max(limits.downdip_width or -1.0, width_km)

    def scaled(maximum: float, extent: float, count: int) -> int:
        """One axis of the rule, in float32 and truncating exactly where the C does."""
        ratio = np.float32(
            np.float32(np.float32(maximum) * np.float32(extend)) / np.float32(extent)
        )
        padded = int(np.float32(ratio * np.float32(count)))
        return padded + 1 if padded % 2 else padded

    return scaled(length_max, length_km, strike_count), scaled(
        width_max, width_km, dip_count
    )


def derive(
    geometry: GsfSubfaults,
    parameters: Parameters,
    *,
    magnitude: float,
    strike_count: int,
    dip_count: int,
) -> Derived:
    """Compute everything genslip derives before it builds a rupture.

    Parameters
    ----------
    geometry : GsfSubfaults
        The subfaults, for `dstk`, `ddip`, `avgdip` and `avgrak`.
    parameters : Parameters
        The getpar set.
    magnitude : float
        `mag`.
    strike_count, dip_count : int
        `nstk` and `ndip`.

    Returns
    -------
    Derived
        The shared quantities.
    """
    length_km = geometry.mean_along_strike_km * strike_count
    width_km = geometry.mean_down_dip_km * dip_count
    padded_strike, padded_dip = padded_extents(
        strike_count, dip_count, length_km, width_km, parameters
    )
    return Derived(
        moment_dyne_cm=seismic_moment(magnitude, parameters.use_moment_magnitude),
        alpha_t=alpha_t(geometry.mean_dip_deg, geometry.mean_rake_deg),
        length_km=length_km,
        width_km=width_km,
        padded_strike=padded_strike,
        padded_dip=padded_dip,
    )


def hypocentre_indices(
    hypocentre_strike_km: float,
    hypocentre_dip_km: float,
    geometry: GsfSubfaults,
    derived: Derived,
) -> tuple[int, int]:
    """Convert genslip's `shypo`/`dhypo` to the subfault indices the port takes.

    **`shypo` is signed and measured from the fault's centre**; `dhypo` is measured
    down dip from the top edge. Neither is a proportion and neither is an index, so
    passing either straight through puts the hypocentre somewhere else entirely --
    usually still inside the fault, which is why this is a trap rather than a crash.

    The `+ 0.5` and the truncation are genslip's own rounding, kept rather than
    replaced with `round`: they differ for negative arguments, and `shypo` is negative
    over half the fault.

    # genslip's `ixs` counts from one, and the port's index counts from zero

    `ixs` and `iys` exist in genslip for one purpose: they are handed to `wfront2d`,
    which is Fortran and indexes from 1 (`wafront2d.f:31`, `ttime(IS + m*(JS-1))`).
    The port's `Hypocentre` is a 0-based subfault index like every other index it
    holds, so **the conversion is here** -- and the whole of `DEFECTS.md` 17 is what
    happens when it is not: onset off by a cell in each direction, which reads as a
    rupture that is merely a bit early rather than as an index error.

    (orig. `genslip_v5.6.2.c:3001` and `:3018`)

    Parameters
    ----------
    hypocentre_strike_km : float
        `shypo`, in km from the centre along strike, in [-flen/2, +flen/2].
    hypocentre_dip_km : float
        `dhypo`, in km down dip from the top edge, in [0, fwid].
    geometry : GsfSubfaults
        For `dstk` and `ddip`.
    derived : Derived
        For `flen`.

    Returns
    -------
    tuple[int, int]
        `(hypocentre_strike, hypocentre_dip)` as 0-based subfault indices.

    Raises
    ------
    ValueError
        If the hypocentre is off the fault. genslip checks the same bounds at line
        3155 and refuses to write a rupture, so this is its check, moved earlier.

        Or if genslip's rounding puts the source *off the near edge of the grid* --
        `ixs = 0`, reached only at `shypo = -flen/2` or `dhypo = 0` exactly. genslip
        accepts that and its padding carries the source one cell outside the fault;
        the port's index is unsigned and cannot say "one cell before subfault zero",
        so this refuses rather than silently rounding into the fault, which would be
        a different rupture.
    """
    if not -0.5 * derived.length_km <= hypocentre_strike_km <= 0.5 * derived.length_km:
        raise ValueError(
            f"shypo={hypocentre_strike_km} km is outside a {derived.length_km} km "
            "fault, which is measured from its centre and so is signed"
        )
    if not 0.0 <= hypocentre_dip_km <= derived.width_km:
        raise ValueError(
            f"dhypo={hypocentre_dip_km} km is outside a {derived.width_km} km width"
        )

    strike = int(
        (hypocentre_strike_km + 0.5 * derived.length_km) / geometry.mean_along_strike_km
        + 0.5
    )
    dip = int(hypocentre_dip_km / geometry.mean_down_dip_km + 0.5)
    if strike == 0 or dip == 0:
        raise ValueError(
            f"shypo={hypocentre_strike_km} km, dhypo={hypocentre_dip_km} km round to "
            f"genslip's ixs={strike}, iys={dip}; a zero there is one cell OFF the "
            "near edge of the grid, which genslip allows and the port's unsigned "
            "subfault index cannot represent"
        )
    return strike - 1, dip - 1


def fault_grid(
    geometry: GsfSubfaults,
    parameters: Parameters,
    derived: Derived,
    *,
    strike_count: int,
    dip_count: int,
) -> FaultGrid:
    """Spec group 1: the discretised fault.

    Three of the six correspondences here are derived rather than named:

    - `strike_km`/`dip_km` are `dstk`/`ddip`, the *means* of the GSF's per-subfault
      dimensions, not a parameter. genslip averages them in double (`iofunc.c:645`).
    - `depth_km` is one value per dip row taken from that row's first subfault, which
      is how genslip indexes it -- see `GsfSubfaults.depth_by_row_km`.
    - `velocity_fraction` is `rvfrac / alphaT`, because genslip divides both the
      scalar and every `psrc[j].rvf` by `alphaT` (lines 1443-1445) and the port
      applies `alphaT` only to rise time.

    Parameters
    ----------
    geometry : GsfSubfaults
        The subfaults.
    parameters : Parameters
        The getpar set. Read for the padding limits only; the rest is geometry.
    derived : Derived
        For the padded extents and `alphaT`.
    strike_count, dip_count : int
        `nstk` and `ndip`.

    Returns
    -------
    FaultGrid
        The port's first spec group.
    """
    del parameters  # every field here comes from the geometry or from `derived`

    # The GSF reader stays float32 because that is what genslip reads the file into,
    # and both sides being handed the same numbers is the whole point of this module.
    # The port computes in float64, so widening here is exact and loses nothing --
    # the *values* are still the file's.

    fraction = _f32(DEFAULT_VELOCITY_FRACTION / derived.alpha_t)
    subfaults = strike_count * dip_count
    return FaultGrid(
        strike_count,
        dip_count,
        derived.padded_strike,
        derived.padded_dip,
        geometry.mean_along_strike_km,
        geometry.mean_down_dip_km,
        # The port takes a depth per subfault now. genslip does not: it reads
        # `psrc[j*nstk].dep`, the first subfault of each row, for every
        # depth-dependent quantity. Repeating that row depth along strike is what
        # hands the port the depths the binary actually used, which is the whole
        # point of this file -- the corpus cannot exercise a per-subfault depth
        # because the reference has no way to express one.
        depth_km=np.repeat(
            geometry.depth_by_row_km(strike_count).astype(np.float64), strike_count
        ),
        base_rake_deg=geometry.rake_deg.astype(np.float64),
        velocity_fraction=np.full(subfaults, fraction, dtype=np.float64),
    )


def velocity_model(
    bottom_depth_km: FloatArray,
    shear_speed_km_s: FloatArray,
    density_g_cm3: FloatArray,
) -> VelocityModel1D:
    """Spec group 2: the layered velocity model.

    The only group with no getpar names in it at all. genslip reads its layers from
    `velfile`, and `write_velocity_model` writes that file from these same three
    arrays -- so this is not a translation, it is the assertion that both sides are
    handed one model. The P speed the file carries is a placeholder nothing reads;
    see `genslip_reference.write_velocity_model`.

    Parameters
    ----------
    bottom_depth_km, shear_speed_km_s, density_g_cm3 : FloatArray
        Layer bottoms, S speeds and densities, shallow to deep.

    Returns
    -------
    VelocityModel1D
        The port's second spec group.
    """
    return VelocityModel1D(
        np.asarray(bottom_depth_km, dtype=np.float64),
        np.asarray(shear_speed_km_s, dtype=np.float64),
        np.asarray(density_g_cm3, dtype=np.float64),
    )


def corner_offsets(parameters: Parameters) -> tuple[float, float]:
    """The offsets `SourceSpec` needs, which are `kx_corner`/`ky_corner` only sometimes.

    Three things happen here that a rename would miss:

    - **The defaults are per `kmodel`,** set immediately before the getpar that may
      override them (lines 994-1035). Mai, Frankel and the hybrid default to
      (2.50, 1.50); Suzuki to (1.67, 1.69); Input Corners has no default at all
      because both are `mstpar`.
    - **The hybrid model ignores what it reads.** Line 996 defaults `kx_corner` for
      `MAI_SOMERVILLE` and line 998 lets the user change it, and then lines 1341-1342
      evaluate literal 2.50 and 1.50. A custom corner with that kmodel changes
      genslip's output not at all, so the literals are returned rather than the
      parameter.
    - **Somerville has no offset variables.** Its 1.72 and 1.93 are inline `double`
      literals the port carries itself, so anything passed here is unread; the zeros
      say so rather than pretending.

    (orig. `genslip_v5.6.2.c:994-1035` for the defaults, `:1303-1370` for the use)

    Parameters
    ----------
    parameters : Parameters
        Read for `kmodel` and the custom corners.

    Returns
    -------
    tuple[float, float]
        The strike and dip offsets to hand `SourceSpec`.

    Raises
    ------
    UnmappableConfigurationError
        If `kmodel=INPUT_CORNERS` and either corner is missing.
    """
    corners = parameters.custom_correlation_corners
    kmodel = parameters.kmodel

    # The hybrid reads both and then evaluates literals. Reproducing the *use*, not
    # the read, is what keeps the mapping faithful.
    if kmodel == KModel.MAI_SOMERVILLE:
        return 2.50, 1.50

    if kmodel in (KModel.MAI, KModel.FRANKEL):
        defaults = (2.50, 1.50)
    elif kmodel == KModel.SUZUKI:
        defaults = (1.67, 1.69)
    elif kmodel == KModel.INPUT_CORNERS:
        if corners.along_strike_corner is None or corners.downdip_corner is None:
            raise UnmappableConfigurationError(
                "kmodel=INPUT_CORNERS takes kx_corner and ky_corner as mstpar "
                "(genslip_v5.6.2.c:1027-1028); genslip would exit rather than default"
            )
        return corners.along_strike_corner, corners.downdip_corner
    else:
        defaults = (0.0, 0.0)

    strike = corners.along_strike_corner
    dip = corners.downdip_corner
    return (
        defaults[0] if strike is None else strike,
        defaults[1] if dip is None else dip,
    )


def magnitude_exponents(parameters: Parameters) -> tuple[float, float]:
    """`xmag_exponent`/`ymag_exponent`, which only `INPUT_CORNERS` reads.

    Both default to 0.5 (lines 1030-1031). Every other relation carries its exponents
    in the port's own `CornerRelation`, so these reach `SourceSpec` and are unread.

    (orig. `genslip_v5.6.2.c:1030-1033`)

    Parameters
    ----------
    parameters : Parameters
        Read for the custom exponents.

    Returns
    -------
    tuple[float, float]
        The strike and dip magnitude exponents.
    """
    corners = parameters.custom_correlation_corners
    strike = corners.along_strike_exponent
    dip = corners.downdip_exponent
    return (0.5 if strike is None else strike, 0.5 if dip is None else dip)


def slip_water_level(parameters: Parameters) -> float:
    """`slip_water_level`, whose "off" is a sentinel rather than a flag.

    genslip defaults it to -1 (line 649) and guards every use with `> 0`. The port
    spells disabled the same way -- any non-positive value -- so the sentinel carries
    across unchanged and no flag has to be invented for it.

    Parameters
    ----------
    parameters : Parameters
        Read for `slip_water_level`.

    Returns
    -------
    float
        The water level, or genslip's -1 sentinel.
    """
    level = parameters.slip_water_level
    return -1.0 if level is None else level


def source_spec(
    geometry: GsfSubfaults,
    parameters: Parameters,
    *,
    magnitude: float,
) -> SourceSpec:
    """Spec group 3: what the earthquake is, before any field is drawn.

    The corner offsets are the subtle part. `kx_corner`/`ky_corner` mean different
    things under different `kmodel`s, have different defaults, and under
    `MAI_SOMERVILLE` are read and then **ignored** -- that branch evaluates literal
    2.50 and 1.50 (lines 1341-1342) even though line 996 just defaulted the variables
    to the same numbers and line 998 let the user change them. Passing a custom
    `kx_corner` with the hybrid model therefore changes genslip's output not at all,
    and this reproduces that by feeding the literals.

    (orig. `genslip_v5.6.2.c:1303-1370` for the relations, `:1412` for rise time)

    Parameters
    ----------
    geometry : GsfSubfaults
        For `avgdip` and `avgrak`.
    parameters : Parameters
        The getpar set.
    magnitude : float
        `mag`.

    Returns
    -------
    SourceSpec
        The port's third spec group.

    Raises
    ------
    UnmappableConfigurationError
        If `kmodel=INPUT_CORNERS` is asked for without its mandatory corners.
    """
    strike_offset, dip_offset = corner_offsets(parameters)
    strike_exponent, dip_exponent = magnitude_exponents(parameters)

    return SourceSpec(
        magnitude,
        _KMODEL_TO_SPECTRUM[parameters.kmodel],
        strike_offset,
        dip_offset,
        use_moment_magnitude=parameters.use_moment_magnitude,
        modified_corners=parameters.modified_corners,
        circular_average=parameters.circular_average,
        saturation_magnitude=parameters.magnitude_clamp,
        strike_exponent=strike_exponent,
        dip_exponent=dip_exponent,
        rise_time_coefficient=parameters.rise_time.coefficient,
        average_dip_deg=geometry.mean_dip_deg,
        average_rake_deg=geometry.mean_rake_deg,
    )


def minimum_wavelength_km(geometry: GsfSubfaults, parameters: Parameters) -> float:
    """genslip's `wavelength_min`, including the default that is not a constant.

    When unset it is `2*sqrt(dstk*ddip)/0.8` -- 80% of the grid's Nyquist wavelength,
    so it tracks the discretisation rather than the fault. A fixture that halves its
    subfault size halves this, and a mapping that hardcoded a number would filter the
    slip spectrum at the wrong scale on every grid but one.

    (orig. `genslip_v5.6.2.c:1236-1237`)

    Parameters
    ----------
    geometry : GsfSubfaults
        For `dstk` and `ddip`.
    parameters : Parameters
        Read for `wavelength_min`.

    Returns
    -------
    float
        The band's lower edge, in km.
    """
    given = parameters.spatial_filtering.rake_min_wavelength
    if given is not None and given >= 0.0:
        return given
    spacing = geometry.mean_along_strike_km * geometry.mean_down_dip_km
    return _f32(2.0 * math.sqrt(spacing) / 0.8)


def slip_spec(geometry: GsfSubfaults, parameters: Parameters) -> SlipSpec:
    """Spec group 4: how the slip field is shaped and trimmed.

    The wavelength band is the trap. `SpatialFiltering.roughness_min_wavelength` and
    `roughness_max_wavelength` are `lambda_min`/`lambda_max`, which belong to the
    *roughness* field; the slip spectrum is filtered by `wavelength_min` and
    `wavelength_max` (line 1707, and the four sibling `kfilt_gaus2` calls). The two
    pairs interact -- line 1239 raises `lambda_min` to `wavelength_min` -- which makes
    them easy to mistake for each other.

    `max_wavelength_km` is `1.0e+15` and not a parameter: genslip assigns it at line
    1235, after the getpar that reads it. The port's own default of 80 km would band-
    limit a spectrum genslip leaves alone.

    (orig. `genslip_v5.6.2.c:1707`, `slip.c:1585` for the filter itself)

    Parameters
    ----------
    geometry : GsfSubfaults
        For the Nyquist-derived wavelength default.
    parameters : Parameters
        The getpar set.

    Returns
    -------
    SlipSpec
        The port's fourth spec group.
    """
    return SlipSpec(
        _KMODEL_TO_SPECTRUM[parameters.kmodel],
        coefficient_of_variation=parameters.slip_sigma,
        # `rake_sigma`, in degrees, and emphatically not `slip_sigma`. genslip
        # normalises the rake field to this spread about each subfault's base rake
        # (`sigfac = rake_sigma/rk_sig`, line 2068). The two sit next to each other
        # in the getpar list, mean different things and carry different units, and
        # the port took the wrong one for as long as nothing drove it end to end.
        rake_sigma_deg=parameters.rake_sigma,
        min_wavelength_km=minimum_wavelength_km(geometry, parameters),
        max_wavelength_km=HARDWIRED_MAX_WAVELENGTH_KM,
        strike_shift=parameters.xshift,
        dip_shift=parameters.yshift,
        side_taper=parameters.tapering.side,
        top_taper=parameters.tapering.top,
        bottom_taper=parameters.tapering.bottom,
        truncate_negative=parameters.truncate_zero_slip,
        water_level=slip_water_level(parameters),
    )


def rupture_time_scale(parameters: Parameters, derived: Derived) -> float:
    """genslip's `tsfac_main`, the amplitude of the rupture-time perturbation.

    Computed from the moment unless given outright:
    `tsfac_bzero + tsfac_slope * (1e-9 * mom^(1/3))`. Negative in practice, which is
    what makes high-slip patches rupture *early*.

    `tsfac_coef`, the pre-v5.4.1 spelling of the same idea, is parsed and never read;
    `RuptureTimePerturbation` records why it is absent.

    (orig. `genslip_v5.6.2.c:1256-1257`)

    Parameters
    ----------
    parameters : Parameters
        For `tsfac_main`, `tsfac_bzero` and `tsfac_slope`.
    derived : Derived
        For the moment.

    Returns
    -------
    float
        `tsfac_main`, in seconds.
    """
    perturbation = parameters.rupture_time_perturbation
    if perturbation.main_value is not None and perturbation.main_value >= -1.0e10:
        return perturbation.main_value
    cube_root = _f32(1.0e-09 * math.exp(math.log(derived.moment_dyne_cm) / 3.0))
    return _f32(perturbation.intercept + perturbation.slope * cube_root)


def deep_ramp_centre_km(
    geometry: GsfSubfaults,
    centre_km: float,
    half_width_km: float,
    *,
    hypocentre_dip_km: float,
) -> float:
    """A deep ramp's centre, pushed down to the hypocentre when that is deeper.

    **This is not a parameter read straight through.** genslip recomputes it per
    hypocentre:

    ```
    xhypo = dhypo*sin(avgdip*rperd) + dtop + deep_risetimedep_range
    if (xhypo > deep_risetimedep) deep_risetimedep = xhypo
    ```

    so a deep hypocentre moves the deep rise-time ramp below itself, keeping the
    hypocentre out of the stretched zone. `deep_vrup_dep` gets the identical treatment
    at line 2974 **with its own half-width**, which is why this takes the centre and
    range as arguments rather than reading one pair: the rise-time ramp and the
    rupture-speed ramp go through the same adjustment with different inputs, and can
    come out unequal even when their configured centres agree.

    `rperd` is genslip's truncated radians constant, not `math.radians`; `gsf.py`
    explains why that matters.

    (orig. `genslip_v5.6.2.c:2378-2381` and `:2974-2977`)

    Parameters
    ----------
    geometry : GsfSubfaults
        For `avgdip` and `dtop`.
    centre_km : float
        The configured centre depth: `deep_risetimedep` or `deep_vrup_dep`.
    half_width_km : float
        That ramp's own half-width: `deep_risetimedep_range` or `deep_vrup_deprange`.
    hypocentre_dip_km : float
        `dhypo`, in km down dip.

    Returns
    -------
    float
        The deep ramp's centre depth, in km.
    """
    hypocentre_depth = _f32(
        hypocentre_dip_km * math.sin(geometry.mean_dip_deg * RADIANS_PER_DEGREE)
        + geometry.top_depth_km
        + half_width_km
    )
    return max(centre_km, hypocentre_depth)


def rise_time_perturbation_defaults(
    parameters: Parameters,
) -> tuple[float, float, float]:
    """The three group-5 fields whose defaults are set by *other* groups.

    genslip assigns them from parameters that belong elsewhere, immediately before the
    getpar that may override them:

    - `rtime1_sigma = slip_sigma` (line 1058) -- from the slip group.
    - `rtime1_depth = stfparams.beta_shal_depth` (line 1063) -- from the beta ramp.
    - `rtime1_depth_range = stfparams.beta_shal_depth_range` (line 1064).

    So changing the slip CoV silently changes the rise-time CoV, and moving the beta
    shallow ramp silently moves the rise-time blend. Both are invisible in a flat
    parameter list, and both would look like port defects.

    (orig. `genslip_v5.6.2.c:1058-1064`)

    Parameters
    ----------
    parameters : Parameters
        The getpar set.

    Returns
    -------
    tuple[float, float, float]
        `(rtime1_sigma, rtime1_depth, rtime1_depth_range)`.
    """
    perturbation = parameters.rise_time_perturbation
    beta = parameters.beta

    sigma = perturbation.level1_sigma
    if sigma is None:
        sigma = parameters.slip_sigma
    depth = perturbation.level1_depth
    if depth is None:
        depth = beta.shallow_depth
    depth_range = perturbation.level1_depth_range
    if depth_range is None:
        depth_range = beta.shallow_depth_range

    return sigma, depth, depth_range


def timing_spec(
    geometry: GsfSubfaults,
    parameters: Parameters,
    derived: Derived,
    *,
    hypocentre_dip_km: float,
) -> TimingSpec:
    """Spec group 5: how rupture time and rise time relate to slip.

    The largest group and the one with the most defaulting-from-elsewhere:
    `rtime1_sigma` defaults to `slip_sigma` (line 1058) and `rtime1_depth` /
    `rtime1_depth_range` default to the *beta* shallow ramp (lines 1063-1064), so
    three of these fields silently follow parameters in other groups.

    The deep ramps additionally depend on the hypocentre -- see `deep_ramp_centre_km`
    -- which is why this takes `dhypo` and the other four groups do not.

    **The rupture-speed ramps are not the rise-time ramps.** They come from
    `RuptureVelocity`, they are passed explicitly rather than left to the port's
    fallback, and that matters as soon as anything moves one pair and not the other:
    genslip's `shal_vrup_dep` stays at 6.5 when `risetimedep` is set to 10, and a
    fallback would silently move it too.

    (orig. `genslip_v5.6.2.c:860-890` for the getpar block, `:2378` and `:2974` for
    the deep ramps)

    Parameters
    ----------
    geometry : GsfSubfaults
        For the deep ramp's hypocentre adjustment.
    parameters : Parameters
        The getpar set.
    derived : Derived
        For the moment `tsfac_main` needs.
    hypocentre_dip_km : float
        `dhypo`, in km down dip.

    Returns
    -------
    TimingSpec
        The port's fifth spec group.
    """
    rise_time = parameters.rise_time
    rise_perturbation = parameters.rise_time_perturbation
    rupture_perturbation = parameters.rupture_time_perturbation
    beta = parameters.beta
    velocity = parameters.rupture_velocity

    level1_sigma, blend_depth, blend_range = rise_time_perturbation_defaults(parameters)

    return TimingSpec(
        rupture_time_correlation=rupture_perturbation.level1_slip_correlation,
        rupture_time_sigma=rupture_perturbation.level1_sigma,
        rupture_time_scale=rupture_time_scale(parameters, derived),
        rupture_delay_s=parameters.rupture_delay,
        rise_time_correlation=rise_perturbation.level1_slip_correlation,
        rise_time_sigma=level1_sigma,
        rise_time_blend=Ramp(blend_depth, blend_range),
        slip_exponent=rise_perturbation.level2_slip_exponent,
        shallow_ramp=Ramp(rise_time.shallow_center_depth, rise_time.shallow_half_width),
        shallow_rise_factor=rise_time.shallow_factor,
        deep_ramp=Ramp(
            deep_ramp_centre_km(
                geometry,
                rise_time.deep_center_depth,
                rise_time.deep_half_width,
                hypocentre_dip_km=hypocentre_dip_km,
            ),
            rise_time.deep_half_width,
        ),
        deep_rise_factor=rise_time.deep_factor,
        # Passed rather than left to the port's fallback: these are `shal_vrup_dep`
        # and `deep_vrup_dep`, which share the rise time's defaults and are not the
        # same parameters. The deep one takes the same hypocentre adjustment with its
        # own half-width, so the two can diverge without either being reconfigured.
        shallow_speed_ramp=Ramp(
            velocity.shallow_center_depth, velocity.shallow_half_width
        ),
        deep_speed_ramp=Ramp(
            deep_ramp_centre_km(
                geometry,
                velocity.deep_center_depth,
                velocity.deep_half_width,
                hypocentre_dip_km=hypocentre_dip_km,
            ),
            velocity.deep_half_width,
        ),
        shallow_speed_factor=velocity.shallow_factor,
        deep_speed_factor=velocity.deep_factor,
        weighting=_SVR_WT_TO_WEIGHTING[RiseTimeNormalisation(int(parameters.svr_wt))],
        beta_shallow_ramp=Ramp(beta.shallow_depth, beta.shallow_depth_range),
        beta_shallow=beta.shallow,
        beta_mid_ramp=Ramp(beta.mid_depth, beta.mid_depth_range),
        beta_mid=beta.mid,
        beta_deep=beta.deep,
        sample_interval_s=parameters.dt,
        max_samples=MAX_SLIP_RATE_SAMPLES,
    )
