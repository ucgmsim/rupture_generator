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

Bounded above at 1: the depth profile only scales this down, and a supershear rupture
is a different model rather than a number above one here.
"""
