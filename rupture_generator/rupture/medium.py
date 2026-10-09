"""The rock a rupture happens in, as functions of position.

The whole interface is a :data:`SpatialField`, which takes positions and returns one
value per position. A field takes no fault geometry, so it depends only on the
positions passed to it and a test can call it on its own. Callers read it at a chart's
cell centres, along the straight line a jump crosses, and could equally read it on a 3D
grid. A :class:`Medium` holds the two fields the rock needs, shear speed and density.

Every field is **deterministic**: the same position gives the same value on every call.
A random medium is still a medium, but whatever constructs it draws the randomness once,
from its own generator, before anyone calls it. A fresh random value per call would
give a segment and the jump onto it different rock.

Units follow the field names. Shear speed is in kilometres per second and density in
grams per cubic centimetre, the units a 1-D velocity model uses. :func:`rigidity_pa`
derives rigidity in pascals from them, and its factor of ``1e9`` is the whole
conversion.
"""

import dataclasses
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from rupture_generator.errors import RuptureGeneratorError

type SpatialField = Callable[[np.ndarray], np.ndarray]
"""Given positions, ``(..., 3)``, one value per position, ``(...)``.

Positions are a chart's own: east, north and depth in kilometres in the realisation's
projected CRS, depth positive down. A field that depends only on depth reads
``positions_km[..., 2]``.
"""

DEPTH = 2
"""Which column of a position is depth."""

PA_PER_KM_S_SQUARED_G_CM3 = 1.0e9
"""``(1e3 m/s)^2 x (1e3 kg/m^3)``, the factor from a velocity model's units to SI."""

CROSSING_SAMPLES = 32
"""How many points along a crossing the midpoint rule reads the medium at."""


def rigidity_pa(shear_speed_km_s: np.ndarray, density_g_cm3: np.ndarray) -> np.ndarray:
    """Compute rigidity, :math:`\\mu = \\rho v_s^2`, from a velocity model's units.

    Parameters
    ----------
    shear_speed_km_s : np.ndarray
        Shear speed in kilometres per second.
    density_g_cm3 : np.ndarray
        Density in grams per cubic centimetre, broadcastable against the shear speed.

    Returns
    -------
    np.ndarray
        Rigidity in pascals.
    """
    return density_g_cm3 * shear_speed_km_s**2 * PA_PER_KM_S_SQUARED_G_CM3


@dataclasses.dataclass(frozen=True)
class Medium:
    """Shear speed and density of the rock, defined everywhere.

    Attributes
    ----------
    shear_speed_km_s : SpatialField
        Shear speed in kilometres per second. The rupture front travels at a fraction
        of it, and a shear wave crossing between faults travels at it.
    density_g_cm3 : SpatialField
        Density in grams per cubic centimetre. With the shear speed it gives the
        rigidity that converts slip to moment.
    """

    shear_speed_km_s: SpatialField
    density_g_cm3: SpatialField

    def rigidity_pa(self, positions_km: np.ndarray) -> np.ndarray:
        """Compute the rigidity, :math:`\\mu = \\rho v_s^2`, at each position.

        Parameters
        ----------
        positions_km : np.ndarray
            Positions, ``(..., 3)``: east, north and depth in kilometres.

        Returns
        -------
        np.ndarray
            Rigidity in pascals, ``(...)``.
        """
        return rigidity_pa(
            self.shear_speed_km_s(positions_km), self.density_g_cm3(positions_km)
        )

    def crossing_time_s(
        self,
        start_km: np.ndarray,
        end_km: np.ndarray,
        samples: int = CROSSING_SAMPLES,
    ) -> np.ndarray:
        """Time a shear wave along the straight line from each start to its end.

        The time is :math:`\\int ds / v_s`, by the midpoint rule over ``samples``
        points. That equals the length over the *harmonic* mean of the shear speed
        along the line, which is the mean a travel time needs. The arithmetic mean
        would let a fast layer hide a slow one.

        Parameters
        ----------
        start_km : np.ndarray
            Where each crossing starts, ``(..., 3)``: east, north and depth in
            kilometres.
        end_km : np.ndarray
            Where each crossing ends, shaped like ``start_km``.
        samples : int
            How many points along each line to read the shear speed at.

        Returns
        -------
        np.ndarray
            Travel time of each crossing in seconds, ``(...)``.
        """
        start = np.asarray(start_km, dtype=np.float64)
        step = np.asarray(end_km, dtype=np.float64) - start
        fractions = (np.arange(samples) + 0.5) / samples
        points = start[..., None, :] + fractions[:, None] * step[..., None, :]
        slowness = 1.0 / self.shear_speed_km_s(points)
        return np.linalg.norm(step, axis=-1) * slowness.mean(axis=-1)


def constant_field(value: float) -> SpatialField:
    """Build a field with the same value everywhere.

    Parameters
    ----------
    value : float
        The value at every position.

    Returns
    -------
    SpatialField
        The constant field.
    """

    def field(positions_km: np.ndarray) -> np.ndarray:
        return np.full(np.shape(positions_km)[:-1], float(value), dtype=np.float64)

    return field


def _increasing_depths(depth_km: np.ndarray, what: str) -> np.ndarray:
    """Depths as a non-empty, strictly increasing 1-D array.

    Raises
    ------
    RuptureGeneratorError
        Otherwise: which interval a depth falls in, or what lies between two points, is
        ambiguous in any other order.
    """
    depths = np.asarray(depth_km, dtype=np.float64)
    if depths.ndim != 1 or depths.size == 0:
        raise RuptureGeneratorError(
            f"the {what} are shaped {depths.shape}; they want a non-empty list of depths"
        )
    if np.any(np.diff(depths) <= 0.0):
        raise RuptureGeneratorError(f"the {what} {depths.tolist()} do not increase")
    return depths


@dataclasses.dataclass(frozen=True)
class Layers:
    """A 1-D earth model, as the depth intervals its layers occupy.

    The deepest layer extends to any depth. A subfault deeper than the model gets the
    deepest layer rather than an error, because a velocity model describes the crust
    and doesn't limit the fault's depth.

    A depth exactly on a boundary belongs to the shallower layer, so each layer
    includes its own bottom.

    Attributes
    ----------
    bottom_depth_km : np.ndarray
        Each layer's lower boundary in kilometres, strictly increasing and below the
        surface.
    """

    bottom_depth_km: np.ndarray

    def __post_init__(self) -> None:
        """Refuse boundaries that fail to form a stack of layers."""
        bottoms = _increasing_depths(self.bottom_depth_km, "layer boundaries")
        if bottoms[0] <= 0.0:
            raise RuptureGeneratorError(
                f"the shallowest layer reaches {bottoms[0]} km, which is not below the "
                "surface"
            )

    def __len__(self) -> int:
        """How many layers."""
        return len(self.bottom_depth_km)

    def _layer_for(self, depth_km: np.ndarray) -> np.ndarray:
        """Which layer each depth falls in, shaped like ``depth_km``."""
        return np.minimum(
            np.searchsorted(self.bottom_depth_km, depth_km, side="left"), len(self) - 1
        )

    def _values(self, name: str, per_layer: ArrayLike) -> np.ndarray:
        """One value per layer, checked against the boundaries.

        Raises
        ------
        RuptureGeneratorError
            Unless there is exactly one positive, finite value per layer.
        """
        values = np.asarray(per_layer, dtype=np.float64)
        if values.shape != (len(self),):
            raise RuptureGeneratorError(
                f"{name} has {values.size} values for {len(self)} layers"
            )
        if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
            raise RuptureGeneratorError(
                f"{name} runs {values.min()} to {values.max()}; every layer needs a "
                "positive, finite value"
            )
        return values


def layered_field(layers: Layers, per_layer: ArrayLike, *, name: str) -> SpatialField:
    """Build a field that depends on depth alone, from a 1-D model.

    The field reads each position's **own** depth rather than one depth per dip row.
    One lookup broadcast along strike would be exact for a plane and for nothing else.

    Parameters
    ----------
    layers : Layers
        The depth intervals.
    per_layer : ArrayLike
        One positive value per layer.
    name : str
        The property's name, for error messages.

    Returns
    -------
    SpatialField
        The value of the layer each position lies in.

    Raises
    ------
    RuptureGeneratorError
        Unless there is exactly one positive, finite value per layer.
    """
    values = layers._values(name, per_layer)

    def field(positions_km: np.ndarray) -> np.ndarray:
        return values[layers._layer_for(positions_km[..., DEPTH])]

    return field


def interpolated_field(depth_km: np.ndarray, values: np.ndarray) -> SpatialField:
    """Build a depth profile from points, linear between them.

    A thin wrapper on :func:`numpy.interp`, which holds the end values outside the
    given depths, so the profile is flat shallower than the first point and deeper
    than the last. That clamping lets one call describe a whole profile, rather
    than one ramp per transition:

    - A ramp, such as the weight tying shallow rise time to slip, takes two points,
      ``[1, 3]`` against ``[0, 1]``.
    - The two-sided rise-time stretch takes four, ``[5, 8, 15, 20]`` against
      ``[2, 1, 1, 2]``. Pulses lengthen near the ground surface and again at depth, and
      the factor stays at exactly 1 through the middle of the fault.
    - A measured profile takes one point per measurement.

    Parameters
    ----------
    depth_km : np.ndarray
        Depths of the points in kilometres, strictly increasing.
    values : np.ndarray
        The profile's value at each depth.

    Returns
    -------
    SpatialField
        The profile at each position's depth.

    Raises
    ------
    RuptureGeneratorError
        If the depths fail to increase, or the values fail to give one finite number
        per depth. :func:`numpy.interp` reads unsorted points as a different profile
        rather than as a mistake, so the ordering check is the one that matters.
    """
    depths = _increasing_depths(depth_km, "profile depths")
    heights = np.asarray(values, dtype=np.float64)
    if heights.shape != depths.shape:
        raise RuptureGeneratorError(
            f"the profile has {heights.size} values for {depths.size} depths"
        )
    if not np.all(np.isfinite(heights)):
        raise RuptureGeneratorError("a profile value is not finite")

    def field(positions_km: np.ndarray) -> np.ndarray:
        return np.interp(positions_km[..., DEPTH], depths, heights)

    return field


def ramp_field(
    centre_km: float, half_width_km: float, shallow: float, deep: float
) -> SpatialField:
    """Build a depth profile that ramps linearly from one value to another.

    Parameters
    ----------
    centre_km : float
        Depth of the middle of the ramp in kilometres.
    half_width_km : float
        Half the ramp's depth extent in kilometres. The ramp spans
        ``centre_km +- half_width_km``.
    shallow : float
        The value shallower than the ramp.
    deep : float
        The value deeper than the ramp.

    Returns
    -------
    SpatialField
        The ramp at each position's depth.

    Raises
    ------
    RuptureGeneratorError
        If the ramp has no width.
    """
    if not half_width_km > 0.0:
        raise RuptureGeneratorError(f"a ramp {half_width_km} km wide is a step")
    return interpolated_field(
        np.array([centre_km - half_width_km, centre_km + half_width_km]),
        np.array([shallow, deep]),
    )


def layered_medium(
    layers: Layers, shear_speed_km_s: ArrayLike, density_g_cm3: ArrayLike
) -> Medium:
    """Build a medium from a 1-D velocity model.

    Parameters
    ----------
    layers : Layers
        The depth intervals.
    shear_speed_km_s : ArrayLike
        One shear speed per layer, in kilometres per second.
    density_g_cm3 : ArrayLike
        One density per layer, in grams per cubic centimetre.

    Returns
    -------
    Medium
        The layered medium.

    Raises
    ------
    RuptureGeneratorError
        Unless each property has exactly one positive, finite value per layer.
    """
    return Medium(
        shear_speed_km_s=layered_field(
            layers, shear_speed_km_s, name="shear_speed_km_s"
        ),
        density_g_cm3=layered_field(layers, density_g_cm3, name="density_g_cm3"),
    )


__all__ = [
    "CROSSING_SAMPLES",
    "DEPTH",
    "Layers",
    "Medium",
    "SpatialField",
    "constant_field",
    "interpolated_field",
    "layered_field",
    "layered_medium",
    "ramp_field",
    "rigidity_pa",
]
