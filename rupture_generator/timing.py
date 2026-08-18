"""S7: how fast the rupture front travels, and when it reaches each subfault.

A rupture speed field, and the eikonal solve over it. The speed is a fraction of the
local shear speed, reduced near the surface and at depth; the arrival times are the
solution of :math:`|\\nabla T| = 1/v` from the seed points, computed by the factored
fast sweeping kernel.

:func:`alpha_t` is Graves & Pitarka's dip-and-rake correction. It shortens the rise
time and raises the rupture speed **by the same factor**, so one function has to serve
both stages; applied at the config boundary instead, a dip-45 reverse fault ruptured
up to 10% slow.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from rupture_generator import _kernels
from rupture_generator.errors import ConfigError
from rupture_generator.mesh import RuptureMesh
from rupture_generator.stages import DepthRamp

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

ALPHA_COEFFICIENT = 0.1
"""How much the geometry correction can move things: at most a tenth. A literal, and
it stays one -- the last attempt to make it configurable used ``-99.0`` as its "use
the default" sentinel and gave every non-strike-slip fault a negative corner
frequency."""

DIP_PLATEAU_DEG = 45.0
"""Below this dip the correction is at full strength, falling to nothing at
vertical."""

REVERSE_RAKE_DEG = 90.0
"""Pure reverse slip, where the correction is at full strength."""

OFF_FAULT_SLOWNESS_FACTOR = 10.0
"""What the slowness of an unoccupied cell is multiplied by.

A chart resampled from a modeller's outline is a rectangle and the fault inside it is
not, so the sweep would otherwise run the front through cells that are not fault and
arrive around the outline's concavities early.

Swept over x10 to x10\\ :sup:`5` on both CFM subduction interfaces: the arrival field is
bit-identical from x10 upward, so there is nothing above it left to buy. The wall is
finite because the kernel refuses non-finite slowness. It stops the front rather than
slowing it, and does not disturb Fomel et al.'s multiplicative factorisation -- on a
uniform medium inside a rectangular fault the maximum error is 1.3e-13 s with the wall
and without it, so ``tau = 1`` survives it exactly.
"""


def alpha_t(average_dip_deg: float, average_rake_deg: float) -> float:
    """Graves & Pitarka's geometric correction, in ``[1/1.1, 1]``.

    Exactly 1 for a vertical strike-slip fault, the calibration point. The rake is
    wrapped into ``[-180, 180]`` **after** averaging, so a fault straddling the wrap
    gives a mean that is not the mean of its angles.

    Raises
    ------
    ConfigError
        For a dip outside ``[0, 90]``, rather than the factor of *zero* that reads as
        a rupture with the correction silently switched off.
    """
    if not (0.0 <= average_dip_deg <= 90.0):
        raise ConfigError(
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

    ``velocity_fraction`` is the **raw** configured fraction of the shear speed: the
    correction is applied to it inside :func:`speed_field`, never by the caller. The
    factors are the speed multiplier outside each ramp, exactly 1 in between.
    """

    velocity_fraction: float
    average_dip_deg: float
    average_rake_deg: float
    # `DepthRamp` is frozen, so these defaults cannot be mutated; ruff flags the call
    # only because the class is imported rather than declared here.
    shallow: DepthRamp = DepthRamp(6.5, 1.5)  # noqa: RUF009
    deep: DepthRamp = DepthRamp(17.5, 2.5)  # noqa: RUF009
    shallow_factor: float = 0.6
    deep_factor: float = 0.6

    def depth_factor(self, depth_km: FloatArray) -> FloatArray:
        """The speed multiplier at each depth: reduced at both ends, 1 between.

        Each branch measures from its ramp's *far* end -- ``1 - shallow.weight`` and
        ``deep.weight`` -- which is what makes the value exactly one at both inner
        edges rather than nearly one, and ``factor`` at each outer edge. The same
        algebra runs in :meth:`~rupture_generator.stages.RiseTimeParams.stretch_at`
        and :meth:`~rupture_generator.pulses.PulseParams.beta_at`.
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
    nowhere else.

    Raises
    ------
    ConfigError
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
        raise ConfigError(
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
    """S7: first-arrival times on ``(i, j)``, in seconds.

    ``shear_speed_km_s`` is per subfault. ``seeds`` are ``(i, j, t0_seconds)``
    triples -- points the front leaves at known times, one for a hypocentre and
    several for a fault triggered along an edge, which leaves no "the hypocentre"
    special case to get off by one.

    Cells the chart marks unoccupied are walled off by
    :data:`OFF_FAULT_SLOWNESS_FACTOR` rather than removed, since the sweep wants a
    rectangle. They need no invented medium first: an unoccupied cell has real corners
    and so a real depth, and the velocity model answers there like anywhere else.

    The metric error is what no wall removes: the sweep measures ``|d(u, v)|`` where
    the front travels ``|dX|``, so on a curved surface paths are short by its own
    stretch. On the two CFM subduction interfaces that is a median of -0.14 to +0.03 s
    against ruptures 143 to 255 s long.
    """
    speed = speed_field(mesh.centres()[..., 2], shear_speed_km_s, params)
    slowness = 1.0 / speed
    occupied = mesh.occupied()
    if not occupied.all():
        slowness = np.where(occupied, slowness, slowness * OFF_FAULT_SLOWNESS_FACTOR)
    strike_km, dip_km = mesh.spacing_km()
    # The solver steps in index space: `i` is down dip, `j` along strike, so the
    # spacings go in that order.
    return _kernels.eikonal_solve(
        np.ascontiguousarray(slowness), (dip_km, strike_km), seeds
    )


__all__ = [
    "ALPHA_COEFFICIENT",
    "DIP_PLATEAU_DEG",
    "REVERSE_RAKE_DEG",
    "SpeedParams",
    "alpha_t",
    "speed_field",
    "travel_times",
]
