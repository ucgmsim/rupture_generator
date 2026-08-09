"""The moment rate function: how fast the earthquake is releasing moment.

.. math::

    \\dot{M}(t) = \\sum_i \\mu_i A_i \\dot{s}_i(t - t_i)

summed over subfaults, where :math:`\\mu` is rigidity, :math:`A` is area,
:math:`\\dot{s}` is the subfault's slip-rate pulse and :math:`t_i` its onset. In
dyne-centimetres per second, because the core works in CGS.

# Why it is here rather than in the viewer

It is the first thing anyone looks at to judge whether a generated rupture is plausible:
a source time function that is ragged, or that peaks at the very start, or whose
integral misses the target moment, says something is wrong before any map does. That
makes it a library quantity, and it has a test the viewer could not give it -- the
integral must equal the moment the generator was *scaled to hit*, which
`ENGINEERING_RULES.md` classes as exact to the f64 fold.

# The pulses are ragged, and the sum is a scatter-add

Each subfault's pulse has its own length -- `nt1` is what the slip-rate generator
returned, not `rise_time / dt`, which is `README.md`'s first trap -- and starts at its
own onset. So this places each pulse at its own offset into a shared timeline rather
than summing aligned arrays: a sliced ``+=`` per subfault, which accumulates where two
of them overlap in time.
"""

from __future__ import annotations

import numpy as np

from rupture_generator._core import GeneratedRupture
from rupture_generator.units import CM2_PER_KM2

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


def rigidity_dyne_cm2(
    shear_speed_km_s: FloatArray, density_g_cm3: FloatArray
) -> FloatArray:
    """Rigidity from shear speed and density, in dyne per square centimetre.

    :math:`\\mu = \\rho v_s^2`, with the kilometres-per-second squared turned into
    centimetres-per-second squared -- which is the same factor as
    :data:`~rupture_generator.units.CM2_PER_KM2` and is a different quantity with the
    same number. `crates/genslip/src/units.rs` names both, and says why having two names
    for `1e10` is the point.

    Parameters
    ----------
    shear_speed_km_s, density_g_cm3 : FloatArray
        One value per subfault, in the units a velocity model is written in.

    Returns
    -------
    FloatArray
        Dyne per square centimetre. Crustal rock is about 3e11, which is 30 GPa.
    """
    return density_g_cm3 * shear_speed_km_s * shear_speed_km_s * CM2_PER_KM2


def moment_rate(
    rupture: GeneratedRupture,
    area_cm2: FloatArray,
    rigidity_dyne_cm2: FloatArray,
    *,
    duration_s: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """The moment rate function, sampled at the rupture's own interval.

    Parameters
    ----------
    rupture : GeneratedRupture
        With its ragged slip-rate pulses and their offsets.
    area_cm2, rigidity_dyne_cm2 : FloatArray
        One value per subfault, along-strike fastest -- the order every field in the
        core is produced in.
    duration_s : float, optional
        How long a timeline to build. Defaults to just past the last pulse's last
        sample, which is the shortest one that loses nothing.

    Returns
    -------
    tuple of FloatArray
        Times in seconds from the first onset, and moment rate in dyne-cm per second.

    Raises
    ------
    ValueError
        If the material arrays do not describe the rupture's subfaults.

    Notes
    -----
    Onsets are quantised to the sample interval, so a pulse starts at the sample nearest
    its onset rather than at the onset exactly. The error is under half a sample --
    0.0025 s at the default 0.005 s interval, which is a twentieth of
    `ENGINEERING_RULES.md`'s 0.05 s onset bound. Interpolating instead would smear each
    pulse across two samples and change the peak, which is the number people read off
    this.
    """
    strike_count, dip_count = rupture.shape
    subfaults = strike_count * dip_count
    for name, values in (
        ("area_cm2", area_cm2),
        ("rigidity_dyne_cm2", rigidity_dyne_cm2),
    ):
        if len(values) != subfaults:
            raise ValueError(
                f"{name} has {len(values)} entries for {subfaults} subfaults"
            )

    interval_s = rupture.sample_interval_s
    offsets = np.asarray(rupture.slip_rate_offsets, dtype=np.int64)
    samples = np.asarray(rupture.slip_rate, dtype=np.float64)
    lengths = np.diff(offsets)

    # Each subfault's pulse starts at the sample nearest its onset. Measured from the
    # earliest onset rather than from zero, so a rupture with a delay does not carry a
    # run of leading zeros nobody asked for.
    onset_s = np.asarray(rupture.onset_s, dtype=np.float64)
    first_s = float(onset_s.min()) if subfaults else 0.0
    starts = np.rint((onset_s - first_s) / interval_s).astype(np.int64)

    if duration_s is None:
        finish = int((starts + lengths).max()) + 1 if subfaults else 1
    else:
        finish = int(np.ceil(duration_s / interval_s)) + 1

    rate = np.zeros(finish, dtype=np.float64)
    weight = np.asarray(area_cm2, dtype=np.float64) * np.asarray(
        rigidity_dyne_cm2, dtype=np.float64
    )

    for subfault in range(subfaults):
        length = int(lengths[subfault])
        if length == 0:
            # A subfault that did not slip has no pulse at all -- `nt1 = 0` and no
            # samples, which is not the same as a pulse of zeros. On a tapered fault
            # that is every edge subfault.
            continue
        start = int(starts[subfault])
        stop = min(start + length, finish)
        if stop <= start:
            continue
        pulse = samples[offsets[subfault] : offsets[subfault] + (stop - start)]
        rate[start:stop] += weight[subfault] * pulse

    return np.arange(finish, dtype=np.float64) * interval_s + first_s, rate


def cumulative_moment(times_s: FloatArray, rate_dyne_cm_s: FloatArray) -> FloatArray:
    """Moment released up to each time, in dyne-centimetres.

    The running integral of the rate. Its last value is the rupture's total moment,
    which is the identity `tests/test_moment.py` rests on.
    """
    if len(times_s) < 2:
        return np.zeros_like(rate_dyne_cm_s)
    interval_s = float(times_s[1] - times_s[0])
    return np.cumsum(rate_dyne_cm_s) * interval_s


__all__ = ["cumulative_moment", "moment_rate", "rigidity_dyne_cm2"]
