"""S7: how fast the rupture front travels, and when it reaches each subfault.

A rupture speed field, and the eikonal solve over it. The speed is a fraction of the
local shear speed, reduced near the surface and at depth; the arrival times are the
solution of :math:`|\\nabla T| = 1/v` from the seed points, computed by the factored
fast sweeping kernel.

:func:`alpha_t` is Graves & Pitarka's dip-and-rake correction, and it lives here
rather than in the config. Their model was calibrated on strike-slip events; a
shallow-dipping reverse fault has the free surface closer to the whole fault plane, so
slip is faster and the pulse shorter. The correction shortens the rise time and raises
the rupture speed **by the same factor**, which is why one function serves both stages.

Applying it at the *config boundary* instead is how a dip-45 reverse fault came to
rupture up to 10% slow: a correction applied by two callers can be applied once, or
twice, or with the wrong sign.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from rupture_generator import _kernels
from rupture_generator.mesh import RuptureMesh
from rupture_generator.stages import DepthRamp

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

ALPHA_COEFFICIENT = 0.1
"""How much the geometry correction can move things: at most a tenth.

A literal, and it stays one. The last thing to make it configurable used ``-99.0`` as
its "use the default" sentinel, which once the deck reader was gone went through
literally and gave every non-strike-slip fault a negative corner frequency.
"""

DIP_PLATEAU_DEG = 45.0
"""Below this dip the correction is at full strength; it falls to nothing at vertical."""

REVERSE_RAKE_DEG = 90.0
"""Pure reverse slip, where the correction is at full strength."""


def alpha_t(average_dip_deg: float, average_rake_deg: float) -> float:
    """Graves & Pitarka's geometric correction, in ``[1/1.1, 1]``.

    Exactly 1 for a vertical strike-slip fault, which is the calibration point, so a
    strike-slip rupture is unaffected by this whole apparatus.

    Parameters
    ----------
    average_dip_deg : float
        The fault's mean dip, in ``[0, 90]``.
    average_rake_deg : float
        The fault's mean rake. Wrapped into ``[-180, 180]`` **after** averaging, so a
        fault straddling the wrap gives a mean that is not the mean of its angles.
        Reproduced deliberately: the alternative is a circular mean, which is a
        different model of what "the fault's rake" means.

    Raises
    ------
    ValueError
        For a dip outside ``[0, 90]``, rather than the factor of *zero* that reads as
        a rupture with the correction silently switched off.
    """
    if not (0.0 <= average_dip_deg <= 90.0):
        raise ValueError(
            f"a dip of {average_dip_deg} degrees is not a fault plane, and the "
            "geometric correction has no meaning outside [0, 90]"
        )

    if average_dip_deg <= DIP_PLATEAU_DEG:
        dip_factor = 1.0
    else:
        dip_factor = 1.0 - (average_dip_deg - DIP_PLATEAU_DEG) / (
            90.0 - DIP_PLATEAU_DEG
        )

    rake_deg = (average_rake_deg + 180.0) % 360.0 - 180.0
    if 0.0 <= rake_deg <= 180.0:
        rake_factor = 1.0 - abs(rake_deg - REVERSE_RAKE_DEG) / REVERSE_RAKE_DEG
    else:
        # Normal faulting: the correction is for reverse-slip geometries and means it.
        rake_factor = 0.0

    return 1.0 / (1.0 + dip_factor * rake_factor * ALPHA_COEFFICIENT)


@dataclasses.dataclass(frozen=True)
class SpeedParams:
    """How fast the front travels.

    Attributes
    ----------
    velocity_fraction : float
        The **raw** configured fraction of the shear speed. The geometric correction
        is applied to it inside :func:`speed_field`, never by the caller.
    average_dip_deg, average_rake_deg : float
        The only inputs to that correction.
    shallow, deep : DepthRamp
        Where the speed reduction begins and ends at each end of the depth range.
        They default to the rise-time stretch ramps, which is the case the four
        independent parameters share.
    shallow_factor, deep_factor : float
        The speed multiplier outside each ramp. Exactly 1 in between.
    """

    velocity_fraction: float
    average_dip_deg: float
    average_rake_deg: float
    # `DepthRamp` is a frozen dataclass, so these defaults cannot be mutated through
    # an instance; ruff flags the call only because the class is imported rather than
    # declared here, where it makes the same judgement itself.
    shallow: DepthRamp = DepthRamp(6.5, 1.5)  # noqa: RUF009
    deep: DepthRamp = DepthRamp(17.5, 2.5)  # noqa: RUF009
    shallow_factor: float = 0.6
    deep_factor: float = 0.6

    def depth_factor(self, depth_km: FloatArray) -> FloatArray:
        """The speed multiplier at each depth: reduced at both ends, 1 between.

        Each branch measures from its ramp's far end, which is what makes the value
        exactly one at both inner edges rather than nearly one.
        """
        return (
            1.0
            - (1.0 - self.shallow_factor) * (1.0 - self.shallow.weight(depth_km))
            - (1.0 - self.deep_factor) * self.deep.weight(depth_km)
        )


def speed_field(
    depth_km: FloatArray, shear_speed_km_s: FloatArray, params: SpeedParams
) -> FloatArray:
    """The rupture speed at every subfault, in kilometres per second.

    .. math:: v_{ij} = \\frac{f}{\\alpha_T} \\, \\beta(z_{ij}) \\, r(z_{ij})

    The division by the geometric correction happens **here**, inside the stage, and
    nowhere else -- the same correction that shortens the rise time, from the same
    function, so the two cannot drift apart.

    Raises
    ------
    ValueError
        If any speed is not strictly positive. The solver inverts it, so a
        non-positive speed is a subfault the front can never reach.
    """
    corrected_fraction = params.velocity_fraction / alpha_t(
        params.average_dip_deg, params.average_rake_deg
    )
    speed = corrected_fraction * params.depth_factor(depth_km) * shear_speed_km_s

    if not np.all(speed > 0.0):
        worst = np.unravel_index(int(np.argmin(speed)), speed.shape)
        # Plain ints: numpy's own scalars repr as `np.int64(2)`, which turns a
        # subfault's coordinates into something nobody wants to read in an error.
        located = tuple(int(index) for index in worst)
        raise ValueError(
            f"the rupture speed at subfault {located} is {float(speed[worst]):.4g} "
            "km/s, which the front can never travel at. Check the velocity model's "
            "shear speeds and the depth ramps"
        )
    return speed


def travel_times(
    mesh: RuptureMesh,
    shear_speed_km_s: FloatArray,
    params: SpeedParams,
    seeds: list[tuple[int, int, float]],
) -> FloatArray:
    """S7: first-arrival times over the chart, in seconds.

    Parameters
    ----------
    mesh : RuptureMesh
        The chart. Its spacing is what the solver steps on.
    shear_speed_km_s : FloatArray
        Per subfault, from the velocity model at each subfault's own depth.
    params : SpeedParams
    seeds : list of tuple
        ``(i, j, t0_seconds)`` -- points the front leaves at known times. One triple
        for a hypocentre; several for a fault triggered along an edge by another
        segment. Seeds-with-times leaves no "the hypocentre" special case to get off
        by one.

    Returns
    -------
    FloatArray
        Travel times on ``(i, j)``, seconds.
    """
    speed = speed_field(mesh.centres()[..., 2], shear_speed_km_s, params)
    strike_km, dip_km = mesh.spacing_km()
    # The solver steps in index space: `i` is down dip, `j` along strike, so the
    # spacings go in that order.
    return _kernels.eikonal_solve(1.0 / speed, (dip_km, strike_km), seeds)


__all__ = [
    "ALPHA_COEFFICIENT",
    "DIP_PLATEAU_DEG",
    "REVERSE_RAKE_DEG",
    "SpeedParams",
    "alpha_t",
    "speed_field",
    "travel_times",
]
