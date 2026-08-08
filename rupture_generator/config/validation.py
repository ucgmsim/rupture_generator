"""Constraints, as functions and as type aliases.

A validator takes a value and either returns (accepted, optionally coerced) or raises
``ValueError`` with a message saying what was wanted. ``ConfigObject.__post_init__``
runs them; nothing else calls them directly.

Every message follows one template -- ``"must be ..., got ..."`` -- because it ends up
in a panel next to the key that broke, where the reader wants the *constraint* and their
own value side by side and nothing else.

# The domain aliases are the point

The generic ones (``PositiveFloat``, ``Latitude``) could come from anywhere. The ones
below them could not: ``DipDeg`` is ``(0, 90]`` rather than ``[0, 90]`` because a fault
that does not dip is not a fault this program describes, and `DEFECTS.md` records
``geometry_correction`` answering a dip of 120 degrees with a correction factor of
*zero* -- a valid-looking rupture with the geometry correction silently switched off.
Writing the range down where the field is declared is how that stops being a thing
anyone has to remember.
"""

from collections.abc import Callable
from typing import Annotated, Any

# ============================================================================
# Comparisons
# ============================================================================


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


def at_least(limit: float) -> Callable[[Any], Any]:
    """Greater than or equal to `limit`."""

    def check(value: float | None) -> float | None:
        if value is not None and value < limit:
            raise ValueError(f"must be {limit} or more, got {value}")
        return value

    return check


def at_most(limit: float) -> Callable[[Any], Any]:
    """Less than or equal to `limit`."""

    def check(value: float | None) -> float | None:
        if value is not None and value > limit:
            raise ValueError(f"must be {limit} or less, got {value}")
        return value

    return check


def in_range(
    low: float, high: float, *, open_low: bool = False
) -> Callable[[Any], Any]:
    """Between `low` and `high`, inclusive unless `open_low`.

    `open_low` exists for the ranges whose lower end is a degenerate case rather than a
    value: a dip of zero is a horizontal fault and a magnitude of zero is not an
    earthquake.
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


# ============================================================================
# Generic aliases
# ============================================================================

PositiveFloat = Annotated[float, positive]
NonNegativeFloat = Annotated[float, non_negative]
PositiveInt = Annotated[int, positive]
UnitInterval = Annotated[float, in_range(0.0, 1.0)]
NonEmptyStr = Annotated[str, non_empty]

# ============================================================================
# Where things are
# ============================================================================

Longitude = Annotated[float, in_range(-180.0, 180.0)]
Latitude = Annotated[float, in_range(-90.0, 90.0)]

DepthKm = Annotated[float, non_negative]
"""Depth below the surface. Downwards is positive, so a negative one is in the air."""

# ============================================================================
# What a fault is
# ============================================================================

Magnitude = Annotated[float, in_range(3.0, 10.0)]
"""Moment magnitude.

Bounded because both ends are typing mistakes rather than earthquakes: below 3 the
generator's scaling relations are extrapolations of relations fitted to nothing that
small, and 10 is larger than any earthquake ever recorded.
"""

StrikeDeg = Annotated[float, in_range(0.0, 360.0)]
"""Clockwise from north. 360 is allowed because it is a spelling of 0."""

DipDeg = Annotated[float, in_range(0.0, 90.0, open_low=True)]
"""Below horizontal, in `(0, 90]`.

Open at zero: a horizontal fault has no down-dip direction and no strike, and the
generator would divide by a vanishing `tan(dip)` to place its nodes. Closed at 90, which
is a vertical fault and entirely ordinary.
"""

RakeDeg = Annotated[float, in_range(-180.0, 180.0)]
"""Slip direction within the plane, from the strike direction."""

VelocityFraction = Annotated[float, in_range(0.0, 1.0, open_low=True)]
"""Rupture speed as a fraction of the shear-wave speed.

Bounded above at 1: this is the *configured* fraction, and the depth profile scales it
down rather than up. A supershear rupture is a different model, not a number above one
here.
"""
