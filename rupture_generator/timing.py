"""S7: how fast the rupture front travels, and when it reaches each subfault.

A rupture speed field, and the eikonal solve over it. The speed is a fraction of the
local shear speed, reduced near the surface and at depth; the arrival times solve
:math:`|\\nabla T| = 1/v` from the seed points, by the factored fast sweeping kernel.

:func:`alpha_t` is Graves & Pitarka's dip-and-rake correction. It shortens the rise
time and raises the rupture speed **by the same factor**, so one function serves both
stages.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from rupture_generator import _kernels
from rupture_generator.config.validation import (
    MAXIMUM_VELOCITY_FRACTION,
    MINIMUM_VELOCITY_FRACTION,
    RAYLEIGH_VELOCITY_FRACTION,
)
from rupture_generator.errors import ConfigError
from rupture_generator.mesh import RuptureMesh
from rupture_generator.stages import DepthRamp

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

ALPHA_COEFFICIENT = 0.1
"""How much the geometry correction can move things: at most a tenth. Not
configurable."""

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
bit-identical from x10 upward. The wall is finite because the kernel refuses non-finite
slowness, and does not disturb Fomel et al.'s multiplicative factorisation -- on a
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
        For a dip outside ``[0, 90]``, rather than a factor of *zero* that reads as the
        correction silently switched off.
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
    correction is applied inside :func:`speed_field`, never by the caller. The factors
    are the speed multiplier outside each ramp, exactly 1 in between.
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
    minimum_fraction: float = MINIMUM_VELOCITY_FRACTION
    maximum_fraction: float = MAXIMUM_VELOCITY_FRACTION

    def __post_init__(self) -> None:
        """Refuse a band the front cannot travel in, or a background it cannot hold.

        Raises
        ------
        ConfigError
            If the band is empty, its ceiling is past Burridge-Andrews or inside the
            forbidden zone, its floor is supershear, or the **corrected** background is
            not a speed a steady front can hold.
        """
        if not (0.0 < self.minimum_fraction <= self.maximum_fraction):
            raise ConfigError(
                f"the rupture velocity band [{self.minimum_fraction}, "
                f"{self.maximum_fraction}] of the shear speed is not a band the front "
                "can travel in"
            )
        if self.maximum_fraction > MAXIMUM_VELOCITY_FRACTION:
            raise ConfigError(
                f"the rupture velocity band reaches {self.maximum_fraction} of the "
                f"shear speed, past the {MAXIMUM_VELOCITY_FRACTION:.4f} "
                "(Burridge-Andrews) this model holds supershear rupture at"
            )
        if RAYLEIGH_VELOCITY_FRACTION < self.maximum_fraction < 1.0:
            raise ConfigError(
                f"the rupture velocity band's ceiling of {self.maximum_fraction} sits "
                f"in the mode-II forbidden zone "
                f"({RAYLEIGH_VELOCITY_FRACTION:.4f} to 1.0 of the shear speed), where "
                "in-plane rupture has no steady speed to hold. Use "
                f"{RAYLEIGH_VELOCITY_FRACTION:.4f} for a sub-shear model or "
                f"{MAXIMUM_VELOCITY_FRACTION:.4f} to allow supershear"
            )
        if self.minimum_fraction >= 1.0:
            raise ConfigError(
                f"the rupture velocity band starts at {self.minimum_fraction} of the "
                "shear speed, so every subfault is supershear and the background "
                "rupture has nowhere sub-shear to sit"
            )

        # `VelocityFraction` bounds the *configured* fraction at 1, but the geometric
        # correction divides by `alpha_t <= 1`, so the corrected background can exceed
        # it: 1.0 configured on a reverse fault is 1.1 of the shear speed before any
        # zone this class skips meaningless. The background is the speed the front
        # travels at, so it has to be a speed a steady front can hold -- at or below
        # the Rayleigh speed.
        corrected = self.velocity_fraction / alpha_t(
            self.average_dip_deg, self.average_rake_deg
        )
        if corrected > RAYLEIGH_VELOCITY_FRACTION:
            raise ConfigError(
                f"a velocity fraction of {self.velocity_fraction} on a fault dipping "
                f"{self.average_dip_deg} degrees with rake {self.average_rake_deg} "
                f"corrects to {corrected:.4f} of the shear speed, past the Rayleigh "
                f"speed at {RAYLEIGH_VELOCITY_FRACTION:.4f}. The geometric correction "
                "divides by alpha_t, so a fraction under 1 does not stay under it -- "
                f"the largest that corrects safely here is "
                f"{RAYLEIGH_VELOCITY_FRACTION * alpha_t(self.average_dip_deg, self.average_rake_deg):.4f}"
            )

    def depth_factor(self, depth_km: FloatArray) -> FloatArray:
        """The speed multiplier at each depth: reduced at both ends, 1 between.

        Each branch measures from its ramp's *far* end -- ``1 - shallow.weight`` and
        ``deep.weight`` -- which makes the value exactly one at both inner edges rather
        than nearly one, and ``factor`` at each outer edge.
        """
        return (
            1.0
            - (1.0 - self.shallow_factor) * (1.0 - self.shallow.weight(depth_km))
            - (1.0 - self.deep_factor) * self.deep.weight(depth_km)
        )


def speed_field(
    depth_km: FloatArray,
    shear_speed_km_s: FloatArray,
    params: SpeedParams,
) -> FloatArray:
    """The rupture speed at every subfault, in kilometres per second.

    .. math::

        f_{ij} = \\mathrm{clip}\\!\\left(\\frac{f}{\\alpha_T} \\, r(z_{ij}),
        \\, f_{\\min}, \\, f_{\\max}\\right), \\qquad
        v_{ij} = f_{ij} \\, \\beta(z_{ij})

    The division by the geometric correction happens **here** and nowhere else.

    The field is a function of depth alone: rupture-time heterogeneity is
    :func:`~rupture_generator.stages.taper_onset`'s, applied to the solved times, and
    the solve itself runs over a smooth speed. So the front this produces is the
    coherent one the blend starts from.

    **The band is the last word.** genslip clips the fraction first and then multiplies
    by its depth factor (``get_rspeed_rvfslip``), and that order cannot keep the
    realised speed out of the mode-II forbidden zone: a subfault at 1.0 of the shear
    speed scaled by a depth factor of 0.95 lands at 0.95, which is inside it. Here the
    profile goes on first and the band is applied to the fraction the front *ends up*
    travelling at.

    That band spans two branches --
    :data:`~rupture_generator.config.validation.MINIMUM_VELOCITY_FRACTION` up to
    :data:`~rupture_generator.config.validation.RAYLEIGH_VELOCITY_FRACTION`, and the
    shear speed up to
    :data:`~rupture_generator.config.validation.MAXIMUM_VELOCITY_FRACTION` -- with the
    forbidden zone between them left empty, and the supershear branch reached by
    shifting rather than by clipping into the zone. :meth:`SpeedParams.__post_init__`
    holds the corrected background sub-Rayleigh, so nothing gets there on the shipped
    depth profile; a ``shallow_factor`` or ``deep_factor`` above 1 can, and this is what
    happens to it when it does.

    Raises
    ------
    ConfigError
        If any speed is not strictly positive. The solver inverts it, so a
        non-positive speed is a subfault the front can never reach.
    """
    corrected_fraction = params.velocity_fraction / alpha_t(
        params.average_dip_deg, params.average_rake_deg
    )
    fraction = np.full_like(np.asarray(depth_km, dtype=np.float64), corrected_fraction)
    fraction = fraction * params.depth_factor(depth_km)

    gap = 1.0 - RAYLEIGH_VELOCITY_FRACTION
    fraction = np.where(fraction > RAYLEIGH_VELOCITY_FRACTION, fraction + gap, fraction)
    fraction = np.clip(fraction, params.minimum_fraction, params.maximum_fraction)

    speed = fraction * shear_speed_km_s

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

    The **coherent** front: ``|grad T| = 1/v`` over the smooth speed field, so a seed's
    own time is exactly the time it was seeded at and no subfault precedes it -- a
    first-arrival field has its minimum at its seed. Heterogeneity is added afterwards,
    by :func:`~rupture_generator.stages.taper_onset`, which blends it in from zero at
    the seed and clamps it per cell so that both of those properties survive.

    ``shear_speed_km_s`` is per subfault. ``seeds`` are ``(i, j, t0_seconds)`` triples:
    points the front leaves at known times, one for a hypocentre and several for a
    fault triggered along an edge, so there is no "the hypocentre" special case.

    Cells the chart marks unoccupied are walled off by
    :data:`OFF_FAULT_SLOWNESS_FACTOR` rather than removed, since the sweep wants a
    rectangle. They need no invented medium: an unoccupied cell has real corners and so
    a real depth, and the velocity model answers there like anywhere else.

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
    "MAXIMUM_VELOCITY_FRACTION",
    "MINIMUM_VELOCITY_FRACTION",
    "OFF_FAULT_SLOWNESS_FACTOR",
    "RAYLEIGH_VELOCITY_FRACTION",
    "REVERSE_RAKE_DEG",
    "SpeedParams",
    "alpha_t",
    "speed_field",
    "travel_times",
]
