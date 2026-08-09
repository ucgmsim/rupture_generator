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


def sample_velocity_model(
    depth_km: FloatArray,
    bottom_depth_km: FloatArray,
    shear_speed_km_s: FloatArray,
    density_g_cm3: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Shear speed and rigidity at each subfault's depth.

    Two conventions that are choices rather than consequences, both kept:

    A depth **exactly on a layer boundary belongs to the layer above it**, which is
    what ``side="left"`` gives; the alternative makes a fault whose top edge sits on
    a boundary sample the layer it is not in.

    A depth **below the deepest layer clamps** to that layer rather than
    extrapolating. A subfault below the model is a modelling error, not a reason to
    invent properties for it.

    Sampled **per subfault**, not per row: one lookup per dip row broadcast along
    strike is exact for a plane and for nothing else, and a bent chart has a
    different depth at every subfault in a row.

    Returns
    -------
    tuple of FloatArray
        Shear speed in km/s and rigidity in pascals, shaped like ``depth_km``.
    """
    bottoms = np.asarray(bottom_depth_km, dtype=np.float64)
    layer = np.minimum(
        np.searchsorted(bottoms, np.asarray(depth_km), side="left"), len(bottoms) - 1
    )
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


def moment_of(
    slip_m: FloatArray, rigidity_pa: FloatArray, area_km2: FloatArray
) -> float:
    """One segment's seismic moment, newton-metres.

    The inverse reading of :func:`scale_to_moment`, for reporting and for the test
    that the parts sum to the whole.
    """
    return float(np.sum(rigidity_pa * area_km2 * M2_PER_KM2 * slip_m))


__all__ = [
    "MAGNITUDE_COEFFICIENT",
    "moment_of",
    "rigidity_pa",
    "sample_velocity_model",
    "scale_to_moment",
    "seismic_moment_nm",
]
