"""Constraints, as functions and as type aliases.

A validator takes a value and either returns (accepted, optionally coerced) or raises
``ValueError`` with a message saying what was wanted. ``ConfigObject.__post_init__``
runs them; nothing else calls them directly.

Every message follows one template -- ``"must be ..., got ..."`` -- so a panel can show
the constraint and the reader's own value side by side.
"""

from collections.abc import Callable
from typing import Annotated, Any

# Comparisons


def positive(value: float | None) -> float | None:
    """Strictly greater than zero. Zero is as wrong as negative for a size."""
    if value is not None and value <= 0.0:
        raise ValueError(f"must be greater than 0, got {value}")
    return value


def non_negative(value: float | None) -> float | None:
    """Zero or more."""
    if value is not None and value < 0.0:
        raise ValueError(f"must be 0 or more, got {value}")
    return value


def in_range(
    low: float, high: float, *, open_low: bool = False
) -> Callable[[Any], Any]:
    """Between `low` and `high`, inclusive unless `open_low`.

    `open_low` is for ranges whose lower end is a degenerate case rather than a value.
    """
    low_bracket = "(" if open_low else "["

    def check(value: float | None) -> float | None:
        if value is None:
            return value
        too_low = value <= low if open_low else value < low
        if too_low or value > high:
            raise ValueError(f"must be in {low_bracket}{low}, {high}], got {value}")
        return value

    return check


def non_empty[T](value: list[T] | str | None) -> list[T] | str | None:
    """At least one of whatever it is."""
    if value is not None and len(value) == 0:
        raise ValueError("must not be empty")
    return value


# Generic aliases

PositiveFloat = Annotated[float, positive]
PositiveInt = Annotated[int, positive]
NonNegativeFloat = Annotated[float, non_negative]
UnitInterval = Annotated[float, in_range(0.0, 1.0)]
NonEmptyStr = Annotated[str, non_empty]

# Where things are

Longitude = Annotated[float, in_range(-180.0, 180.0)]
Latitude = Annotated[float, in_range(-90.0, 90.0)]

DepthKm = Annotated[float, non_negative]
"""Depth below the surface. Downwards is positive, so a negative one is in the air."""

# How long things take

Seconds = Annotated[float, non_negative]
"""A duration or a delay. Time runs one way here, so negative is not a value."""

# What a fault is

Magnitude = Annotated[float, in_range(3.0, 10.0)]
"""Moment magnitude, in ``[3, 10]``.

Below 3 the scaling relations are extrapolations, and 10 is larger than any earthquake
ever recorded.
"""

StrikeDeg = Annotated[float, in_range(0.0, 360.0)]
"""Clockwise from north. 360 is allowed because it is a spelling of 0."""

DipDeg = Annotated[float, in_range(0.0, 90.0, open_low=True)]
"""Below horizontal, in ``(0, 90]``.

Open at zero: a horizontal fault has no down-dip direction, and placing its nodes would
divide by a vanishing ``tan(dip)``. 90 is a vertical fault and entirely ordinary.
"""

RakeDeg = Annotated[float, in_range(-180.0, 180.0)]
"""Slip direction within the plane, from the strike direction."""

VelocityFraction = Annotated[float, in_range(0.0, 1.0, open_low=True)]
"""Rupture speed as a fraction of the shear-wave speed, in ``(0, 1]``.

The **effective** ceiling is lower and depends on the fault's geometry, so it cannot
live in this type: the usable range is
``(0, RAYLEIGH_VELOCITY_FRACTION * alpha_T(dip, rake)]`` -- 0.9194 for strike-slip and
0.8358 for reverse -- and
:meth:`~rupture_generator.timing.SpeedParams.__post_init__` is what enforces it, naming
the largest value that corrects safely for the segment in hand.

Bounded above at 1: this is the rupture speed the front actually travels at, and the
depth profile only scales it down. :data:`MAXIMUM_VELOCITY_FRACTION` reaches past 1
because it is the wall of the *band*, which has a supershear branch; nothing in the
shipped model reaches it.
"""


MINIMUM_VELOCITY_FRACTION = 0.25
"""The slowest the front may travel, as a fraction of the local shear speed.

The lower wall of the band, and genslip's value (``rvfmin=0.25``,
``genslip_v5.6.2.c:658``).

It has to sit this low because the band is applied to the **realised** fraction, after
the depth profile -- see :func:`~rupture_generator.timing.speed_field`. The shallow
background is already ``0.8 * 0.6 = 0.48``, so a floor of 0.4 would be the floor
overriding the shallow weak zone rather than bounding anything: it leaves the profile
only a fifth of its own range to work in.
"""

RAYLEIGH_VELOCITY_FRACTION = 0.9194
"""The Rayleigh speed over the shear speed, and the top of the sub-shear branch.

For a Poisson solid (``nu = 0.25``), ``cR / Vs = 0.9194``. In-plane rupture has **no
steady solution** between ``cR`` and ``Vs``: the energy release rate there does not
admit one, which is the mode-II forbidden zone. A band whose ceiling sits inside that
zone is asking the front to travel at a speed it cannot hold, so the zone is skipped
rather than clipped into -- see
:func:`~rupture_generator.timing.speed_field`.

Fixed at ``nu = 0.25`` rather than computed from the velocity model's own Poisson
ratio. Over the range a crustal model spans, ``nu`` in ``[0.20, 0.30]``, ``cR / Vs``
moves between 0.9110 and 0.9274 -- under a percent, and well inside the width of the
zone being avoided. Mode III has no forbidden zone at all and a real rupture is
mixed-mode, so the zone is a mode-II steady-state statement rather than a prohibition;
this treats it as one anyway, because a speed no in-plane front can sustain is not a
speed to hand a solver.
"""

MAXIMUM_VELOCITY_FRACTION = 2.0**0.5
"""The fastest the front may travel, as a fraction of the local shear speed.

``sqrt(2) Vs``, the Burridge-Andrews speed: the stable supershear speed for in-plane
rupture, and what genslip allows through ``rvfmax=1.414``. Above the forbidden zone, so
a front placed here can hold its speed.

Note what this does **not** say. The shipped model never gets here: the rupture is
sub-Rayleigh everywhere, and what holds it there is
:meth:`~rupture_generator.timing.SpeedParams.__post_init__`, **not**
:data:`VelocityFraction`. The geometric correction divides the configured fraction by
``alpha_t <= 1``, so a configured value under 1 does not stay under 1, and 1.0 on a
reverse fault corrects to 1.1 of the shear speed -- which that check refuses. The
branch is kept because the band is a statement about what speeds a front can hold, and
a depth factor above 1 is a configuration that can reach for one.

``sqrt(2) Vs`` is below ``Vp`` for any Poisson solid (``Vp / Vs = sqrt(3)`` at
``nu = 0.25``, and 1.41 only at ``nu = 0``), so this stays inside the physical
supershear window without needing ``Vp``.
"""
PerturbedVelocityFraction = Annotated[
    float, in_range(0.0, MAXIMUM_VELOCITY_FRACTION, open_low=True)
]
"""A wall of the band the realised rupture speed is held inside.

Reaches past 1, unlike :data:`VelocityFraction`, because the band has a supershear
branch above the forbidden zone as well as a sub-shear one below it.
"""
