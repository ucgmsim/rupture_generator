"""Magnitude, moment, rigidity, and the one scaling that closes them.

Everything here is SI: moment in newton-metres, rigidity in pascals, slip in metres.
The one conversion, square kilometres to square metres, happens in
:func:`scale_to_moment`.

.. math:: \\log_{10} M_0 [\\mathrm{N\\,m}] = 1.5 (M_w + 6.0333003)

Hanks & Kanamori (1979) equation 7, in the SI form the paper itself publishes; their
equation 4 is a different relation and reads 1.109 times too much moment.

References
----------
Hanks, T. C., & Kanamori, H. (1979). A moment magnitude scale.
*Journal of Geophysical Research*, 84(B5), 2348-2350.
"""

from __future__ import annotations

import numpy as np

from rupture_generator.errors import ConfigError
from rupture_generator.units import M2_PER_KM2

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

MAGNITUDE_COEFFICIENT = 10.699967 - 7.0 / 1.5
"""The constant in Hanks & Kanamori (1979) equation 7's SI form, at full precision.

The literature writes 10.699967 for the dyne-centimetre form; newton-metres are 1e7
larger and the slope is 1.5, so the SI constant is exactly that much smaller -- about
6.0333003, the 6.03 the paper rounds to. Written as the derivation rather than its
decimal expansion because rounding at the seventh figure moves the moment by 1.2e-7
relative, and two forms of one constant that disagree make a later comparison
ambiguous.
"""


def seismic_moment_nm(magnitude: float) -> float:
    """Moment in newton-metres, from moment magnitude. An M6 is about 1.1e18."""
    return float(10.0 ** (1.5 * (magnitude + MAGNITUDE_COEFFICIENT)))


def rigidity_pa(shear_speed_km_s: FloatArray, density_g_cm3: FloatArray) -> FloatArray:
    """Rigidity in pascals, from shear speed and density.

    :math:`\\mu = \\rho v_s^2`, with the velocity model's own units -- kilometres per
    second and grams per cubic centimetre -- carried into SI by a single factor of
    ``1e9``: ``(1e3 m/s)^2 x (1e3 kg/m^3)``. Crustal rock is about 3e10 Pa.
    """
    return np.asarray(density_g_cm3) * np.asarray(shear_speed_km_s) ** 2 * 1.0e9


def layer_of(depth_km: FloatArray, bottom_depth_km: FloatArray) -> IntArray:
    """Which layer of a 1-D velocity model each depth falls in.

    A depth exactly on a layer boundary belongs to the layer **above** it, which is
    what ``side="left"`` gives; a depth below the deepest layer clamps to it rather
    than extrapolating. Returns layer indices shaped like ``depth_km``.
    """
    bottoms = np.asarray(bottom_depth_km, dtype=np.float64)
    return np.minimum(
        np.searchsorted(bottoms, np.asarray(depth_km), side="left"), len(bottoms) - 1
    )


def sample_velocity_model(
    depth_km: FloatArray,
    bottom_depth_km: FloatArray,
    shear_speed_km_s: FloatArray,
    density_g_cm3: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Shear speed in km/s and rigidity in pascals at each subfault's own depth.

    Sampled **per subfault**: one lookup per dip row broadcast along strike would be
    exact for a plane and for nothing else.
    """
    layer = layer_of(depth_km, bottom_depth_km)
    shear_speed = np.asarray(shear_speed_km_s, dtype=np.float64)[layer]
    density = np.asarray(density_g_cm3, dtype=np.float64)[layer]
    return shear_speed, rigidity_pa(shear_speed, density)


def scale_to_moment(
    fields: list[FloatArray],
    rigidities_pa: list[FloatArray],
    areas_km2: list[FloatArray],
    target_moment_nm: float,
) -> list[FloatArray]:
    """Scale unit-mean slip patterns so that together they carry the target moment.

    .. math::

        \\gamma = \\frac{M_0}{\\sum_k \\sum_{ij} \\mu_{ij} A_{ij} f_{ij}},
        \\qquad s_{ij} = \\gamma f_{ij}

    One factor, shared across every segment: only the total is a target. The
    accumulation is in float64 -- single precision on a hundred thousand subfaults
    costs about 6e-5 relative. Returns slip in **metres**, one array per segment.

    Raises
    ------
    ConfigError
        If every field is zero everywhere, which carries no moment.
    """
    total = 0.0
    for field, rigidity, area in zip(fields, rigidities_pa, areas_km2, strict=True):
        total += float(np.sum(rigidity * area * M2_PER_KM2 * field))

    if not (total > 0.0):
        raise ConfigError(
            "the slip pattern carries no moment anywhere, so there is no factor that "
            "makes it carry the target -- every subfault was truncated to zero"
        )

    factor = target_moment_nm / total
    return [factor * field for field in fields]


def scale_each_to_moment(
    fields: list[FloatArray],
    rigidities_pa: list[FloatArray],
    areas_km2: list[FloatArray],
    target_moments_nm: list[float],
) -> list[FloatArray]:
    """Scale each segment to a target of its own, returning slip in metres.

    The counterpart of :func:`scale_to_moment`: each segment's moment is exact and
    the event's total is whatever the parts sum to.

    Raises
    ------
    ConfigError
        If a segment's pattern carries no moment anywhere, naming its position.
    """
    scaled = []
    for index, (field, rigidity, area, target) in enumerate(
        zip(fields, rigidities_pa, areas_km2, target_moments_nm, strict=True)
    ):
        total = float(np.sum(rigidity * area * M2_PER_KM2 * field))
        if not (total > 0.0):
            raise ConfigError(
                f"segment {index}'s slip pattern carries no moment anywhere, so no "
                "factor makes it carry its target -- every subfault was truncated"
            )
        scaled.append((target / total) * field)
    return scaled


def moment_of(
    slip_m: FloatArray, rigidity_pa: FloatArray, area_km2: FloatArray
) -> float:
    """One segment's seismic moment, newton-metres."""
    return float(np.sum(rigidity_pa * area_km2 * M2_PER_KM2 * slip_m))


def moment_rate(
    pulse_offsets: np.ndarray,
    pulse_samples: FloatArray,
    onset_s: FloatArray,
    area_m2: FloatArray,
    rigidity_pa: FloatArray,
    *,
    sample_interval_s: float,
    duration_s: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """The moment rate function, sampled at the rupture's own interval.

    .. math:: \\dot{M}(t) = \\sum_i \\mu_i A_i \\dot{s}_i(t - t_i)

    Each subfault's pulse starts at its own onset, so this places each at its own
    offset into a shared timeline rather than summing aligned arrays. Onsets are
    quantised to the sample interval, an error under half a sample -- 0.0025 s at the
    default; interpolating would smear each pulse across two samples and change the
    peak, which is the number people read off this.

    ``pulse_offsets`` and ``pulse_samples`` are the CSR pulses in m/s; ``onset_s``,
    ``area_m2`` and ``rigidity_pa`` are one value per subfault, flattened along strike
    fastest. Returns times in seconds from the first onset, and rate in N m/s.
    """
    offsets = np.asarray(pulse_offsets, dtype=np.int64)
    samples = np.asarray(pulse_samples, dtype=np.float64)
    lengths = np.diff(offsets)
    subfaults = len(lengths)

    onset_s = np.asarray(onset_s, dtype=np.float64).ravel()
    first_s = float(onset_s.min()) if subfaults else 0.0
    starts = np.rint((onset_s - first_s) / sample_interval_s).astype(np.int64)

    if duration_s is None:
        finish = int((starts + lengths).max()) + 1 if subfaults else 1
    else:
        finish = int(np.ceil(duration_s / sample_interval_s)) + 1

    rate = np.zeros(finish, dtype=np.float64)
    weight = (
        np.asarray(area_m2, dtype=np.float64).ravel()
        * np.asarray(rigidity_pa, dtype=np.float64).ravel()
    )

    for subfault in range(subfaults):
        length = int(lengths[subfault])
        if length == 0:
            # A subfault that did not slip has no pulse at all -- no samples, which
            # is not the same as a pulse of zeros. On a tapered fault that is every
            # edge subfault.
            continue
        start = int(starts[subfault])
        stop = min(start + length, finish)
        if stop <= start:
            continue
        pulse = samples[offsets[subfault] : offsets[subfault] + (stop - start)]
        rate[start:stop] += weight[subfault] * pulse

    return np.arange(finish, dtype=np.float64) * sample_interval_s + first_s, rate


def cumulative_moment(times_s: FloatArray, rate_newton_m_s: FloatArray) -> FloatArray:
    """Moment released up to each time, the running integral of the rate, in N m.

    Its last value is the rupture's total moment -- the moment-rate test's identity.
    """
    if len(times_s) < 2:
        return np.zeros_like(rate_newton_m_s)
    interval_s = float(times_s[1] - times_s[0])
    return np.cumsum(rate_newton_m_s) * interval_s


__all__ = [
    "MAGNITUDE_COEFFICIENT",
    "cumulative_moment",
    "layer_of",
    "moment_of",
    "moment_rate",
    "rigidity_pa",
    "sample_velocity_model",
    "scale_each_to_moment",
    "scale_to_moment",
    "seismic_moment_nm",
]
