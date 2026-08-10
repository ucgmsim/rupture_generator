"""Magnitude, moment, rigidity, and the one scaling that closes them.

Everything here is SI: moment in newton-metres, rigidity in pascals, slip in metres.
The mesh works in kilometres, so the one conversion -- square kilometres to square
metres -- happens here, once, in :func:`scale_to_moment`.

# One magnitude convention

.. math::

    \\log_{10} M_0 [\\mathrm{N\\,m}] = 1.5 (M_w + 6.0333003)

Hanks & Kanamori (1979) **equation 7**, in the SI form the paper itself publishes.
Equation 4 is a different relation with a different constant, and defaulting to it
read 1.109 times too much moment and mean slip against every config that leaves the
production default alone -- one of the four wrong numbers, and worth naming because
the error is a clean multiplicative factor that no diagnostic about *shape* can see.
"""

from __future__ import annotations

import numpy as np

from rupture_generator.units import M2_PER_KM2

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

MAGNITUDE_COEFFICIENT = 10.699967 - 7.0 / 1.5
"""The constant in eq. 7's SI form, at full precision.

The seismological literature and genslip both write **10.699967** for the
dyne-centimetre form. Newton-metres are ``1e7`` larger, and the relation's slope is
1.5, so the SI constant is exactly that much smaller -- about 6.0333003, which is the
6.03 the paper rounds to.

Written as the derivation rather than as its decimal expansion because rounding it at
the seventh figure moves the moment by 1.2e-7 relative: harmless, and needless, and
the kind of gratuitous disagreement between two forms of one constant that makes a
later comparison ambiguous.
"""


def seismic_moment_nm(magnitude: float) -> float:
    """Moment in newton-metres, from moment magnitude.

    Parameters
    ----------
    magnitude : float
        Moment magnitude.

    Returns
    -------
    float
        Seismic moment, newton-metres. An M6 is about 1.1e18.
    """
    return float(10.0 ** (1.5 * (magnitude + MAGNITUDE_COEFFICIENT)))


def rigidity_pa(shear_speed_km_s: FloatArray, density_g_cm3: FloatArray) -> FloatArray:
    """Rigidity in pascals, from shear speed and density.

    :math:`\\mu = \\rho v_s^2`, with the velocity model's own units -- kilometres per
    second and grams per cubic centimetre -- carried into SI by a single factor of
    ``1e9``: ``(1e3 m/s)^2 x (1e3 kg/m^3)``.

    Returns
    -------
    FloatArray
        Pascals. Crustal rock is about 3e10, which is 30 GPa.
    """
    return np.asarray(density_g_cm3) * np.asarray(shear_speed_km_s) ** 2 * 1.0e9


def layer_of(depth_km: FloatArray, bottom_depth_km: FloatArray) -> IntArray:
    """Which layer of a 1-D velocity model each depth falls in.

    Two conventions that are choices rather than consequences, both kept:

    A depth **exactly on a layer boundary belongs to the layer above it**, which is
    what ``side="left"`` gives; the alternative makes a fault whose top edge sits on
    a boundary sample the layer it is not in.

    A depth **below the deepest layer clamps** to that layer rather than
    extrapolating. A subfault below the model is a modelling error, not a reason to
    invent properties for it.

    Both conventions live here rather than at each lookup, because a second copy is a
    second thing to disagree with -- and the callers are in different modules: the
    materials of a subfault, and the shear speed of the rock a jump crosses.

    Returns
    -------
    IntArray
        Layer indices, shaped like ``depth_km``.
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
    """Shear speed and rigidity at each subfault's depth.

    Sampled **per subfault**, not per row: one lookup per dip row broadcast along
    strike is exact for a plane and for nothing else, and a bent chart has a
    different depth at every subfault in a row.

    Returns
    -------
    tuple of FloatArray
        Shear speed in km/s and rigidity in pascals, shaped like ``depth_km``.
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

    **One factor, shared across every segment.** That is the whole content of the
    joint scaling: a segment's own moment is whatever the shared factor and its own
    pattern give it, and only the total is a target. A per-segment scaling would
    make the moment right and the *partition between faults* an artefact of how the
    patterns happened to normalise.

    That the sum then equals the target is a tautology -- it is divided by exactly
    that sum. What the assertion is worth is the **registration**: that the sum runs
    over all segments, that the areas are the mesh's own rather than a nominal
    product of spacings, and that the accumulation is in float64. The C folds through
    single precision, which on a hundred thousand subfaults costs about 6e-5
    relative -- six missing subfaults' worth, where in float64 one missing subfault
    is visible.

    Parameters
    ----------
    fields : list of FloatArray
        Per-segment slip patterns, dimensionless and non-negative.
    rigidities_pa, areas_km2 : list of FloatArray
        Per-segment rigidity in pascals and cell area in square kilometres.
    target_moment_nm : float
        The moment the whole event must carry, newton-metres.

    Returns
    -------
    list of FloatArray
        Slip in **metres**, one array per segment.

    Raises
    ------
    ValueError
        If every field is zero everywhere, which carries no moment and cannot be
        scaled to carry any.
    """
    total = 0.0
    for field, rigidity, area in zip(fields, rigidities_pa, areas_km2, strict=True):
        total += float(np.sum(rigidity * area * M2_PER_KM2 * field))

    if not (total > 0.0):
        raise ValueError(
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
    """Scale each segment to a target of its own.

    The counterpart of :func:`scale_to_moment`, for a source that states how the
    moment divides between faults rather than letting the fields decide. A hazard
    model that derived each fault's magnitude from its own area has already made that
    decision; re-deriving it would discard what the model said.

    The two are genuinely different: here each segment's moment is exact and the
    event's total is whatever the parts sum to, where jointly the total is exact and
    the parts are whatever the fields give.

    Returns
    -------
    list of FloatArray
        Slip in metres, one array per segment.

    Raises
    ------
    ValueError
        If a segment's pattern carries no moment anywhere, naming its position --
        which for a per-fault source means that fault cannot reach its target at all.
    """
    scaled = []
    for index, (field, rigidity, area, target) in enumerate(
        zip(fields, rigidities_pa, areas_km2, target_moments_nm, strict=True)
    ):
        total = float(np.sum(rigidity * area * M2_PER_KM2 * field))
        if not (total > 0.0):
            raise ValueError(
                f"segment {index}'s slip pattern carries no moment anywhere, so no "
                "factor makes it carry its target -- every subfault was truncated"
            )
        scaled.append((target / total) * field)
    return scaled


def moment_of(
    slip_m: FloatArray, rigidity_pa: FloatArray, area_km2: FloatArray
) -> float:
    """One segment's seismic moment, newton-metres.

    The inverse reading of :func:`scale_to_moment`, for reporting and for the test
    that the parts sum to the whole.
    """
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

    It is the first thing anyone looks at to judge whether a generated rupture is
    plausible: a source time function that is ragged, or that peaks at the very
    start, or whose integral misses the target moment, says something is wrong before
    any map does. That makes it a library quantity rather than a viewer's, and it has
    a test a viewer could not give it -- the integral is the moment the generator was
    scaled to hit.

    Each subfault's pulse has its own length and starts at its own onset, so this
    places each at its own offset into a shared timeline rather than summing aligned
    arrays. Onsets are quantised to the sample interval: a pulse starts at the sample
    nearest its onset, an error under half a sample -- 0.0025 s at the default
    interval, a twentieth of the onset bound. Interpolating instead would smear each
    pulse across two samples and change the peak, which is the number people read off
    this.

    Parameters
    ----------
    pulse_offsets, pulse_samples : np.ndarray
        The CSR pulses, in metres per second.
    onset_s, area_m2, rigidity_pa : FloatArray
        One value per subfault, flattened along strike fastest.
    sample_interval_s : float
    duration_s : float, optional
        How long a timeline to build. Defaults to just past the last pulse's last
        sample, which is the shortest one that loses nothing.

    Returns
    -------
    tuple of FloatArray
        Times in seconds from the first onset, and moment rate in newton-metres per
        second.
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
    """Moment released up to each time, in newton-metres.

    The running integral of the rate. Its last value is the rupture's total moment,
    which is the identity the moment-rate test rests on.
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
