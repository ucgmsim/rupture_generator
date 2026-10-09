"""The rock a rupture happens in, as functions of position.

A :data:`SpatialField` is the whole interface: given positions, return one value per
position. Nothing here knows what a fault is, so a field is a pure function of where
it is asked about and testable on its own, and it can be asked anywhere -- at a chart's
cell centres, along the straight line a jump crosses, or on a 3-D grid. A
:class:`Medium` is the two fields the rock needs, shear speed and density.

A field is **deterministic**: the same position gives the same value, every time it is
asked. A random medium is still a medium, but its randomness is realised when it is
built -- by whatever constructs it, from its own generator -- and never when it is
called, or a segment and the jump onto it would see different rock.

Units follow the field names. Shear speed is kilometres per second and density grams
per cubic centimetre, which is how a 1-D velocity model is written down; rigidity is
derived from them in pascals, and the single ``1e9`` in :func:`rigidity_pa` is the
whole conversion.
"""

import dataclasses
from collections.abc import Callable

import numpy as np

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
"""``(1e3 m/s)^2 x (1e3 kg/m^3)``: what carries a velocity model's own units to SI."""

CROSSING_SAMPLES = 32
"""Points along a crossing at which the medium is read, by the midpoint rule."""


def rigidity_pa(shear_speed_km_s: np.ndarray, density_g_cm3: np.ndarray) -> np.ndarray:
    """Rigidity in pascals, :math:`\\mu = \\rho v_s^2`, from a velocity model's units."""
    return density_g_cm3 * shear_speed_km_s**2 * PA_PER_KM_S_SQUARED_G_CM3


@dataclasses.dataclass(frozen=True)
class Medium:
    """The rock: shear speed and density, everywhere.

    Attributes
    ----------
    shear_speed_km_s : SpatialField
        What the front's speed is a fraction of, and what a jump crosses at.
    density_g_cm3 : SpatialField
        With the shear speed, what the rigidity the moment is counted in comes from.
    """

    shear_speed_km_s: SpatialField
    density_g_cm3: SpatialField

    def rigidity_pa(self, positions_km: np.ndarray) -> np.ndarray:
        """Rigidity at each position, :math:`\\mu = \\rho v_s^2`, in pascals."""
        return rigidity_pa(
            self.shear_speed_km_s(positions_km), self.density_g_cm3(positions_km)
        )

    def crossing_time_s(
        self,
        start_km: np.ndarray,
        end_km: np.ndarray,
        samples: int = CROSSING_SAMPLES,
    ) -> np.ndarray:
        """How long a shear wave takes along the straight line from each start to its
        end: :math:`\\int ds / v_s`, by the midpoint rule over ``samples`` points.

        That is the length over the *harmonic* mean of the shear speed along the line,
        which is what a travel time is; the arithmetic mean would let a fast layer hide
        a slow one. ``start_km`` and ``end_km`` are ``(..., 3)``; the result is
        ``(...)``.
        """
        start = np.asarray(start_km, dtype=np.float64)
        step = np.asarray(end_km, dtype=np.float64) - start
        fractions = (np.arange(samples) + 0.5) / samples
        points = start[..., None, :] + fractions[:, None] * step[..., None, :]
        slowness = 1.0 / self.shear_speed_km_s(points)
        return np.linalg.norm(step, axis=-1) * slowness.mean(axis=-1)


def constant_field(value: float) -> SpatialField:
    """The same value everywhere."""

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
    """A 1-D earth model: what the rock is like in each depth interval.

    ``bottom_depth_km`` is each layer's lower boundary, increasing, and the deepest one
    is the floor everything below clamps to -- a subfault deeper than the model gets
    the deepest layer rather than an error, because a velocity model is a description
    of the crust and not a bound on the fault.

    A depth exactly on a boundary belongs to the layer **above** it, so a layer owns
    its own bottom.
    """

    bottom_depth_km: np.ndarray

    def __post_init__(self) -> None:
        """Refuse boundaries that are not a stack of layers."""
        bottoms = _increasing_depths(self.bottom_depth_km, "layer boundaries")
        if bottoms[0] <= 0.0:
            raise RuptureGeneratorError(
                f"the shallowest layer reaches {bottoms[0]} km, which is not below the "
                "surface"
            )

    def __len__(self) -> int:
        """How many layers."""
        return len(self.bottom_depth_km)

    def layer_for(self, depth_km: np.ndarray) -> np.ndarray:
        """Which layer each depth falls in, shaped like ``depth_km``."""
        return np.minimum(
            np.searchsorted(self.bottom_depth_km, depth_km, side="left"), len(self) - 1
        )

    def values(self, name: str, per_layer: np.ndarray) -> np.ndarray:
        """One value per layer, checked against the boundaries.

        Raises
        ------
        RuptureGeneratorError
            If there is not exactly one value per layer, or one of them is not positive.
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


def layered_field(layers: Layers, per_layer: np.ndarray, *, name: str) -> SpatialField:
    """A property that depends on depth alone, read off a 1-D model.

    Read at each position's **own** depth rather than once per dip row: one lookup
    broadcast along strike is exact for a plane and for nothing else.
    """
    values = layers.values(name, per_layer)

    def field(positions_km: np.ndarray) -> np.ndarray:
        return values[layers.layer_for(positions_km[..., DEPTH])]

    return field


def interpolated_field(depth_km: np.ndarray, values: np.ndarray) -> SpatialField:
    """A depth profile, given as points and read linearly between them.

    A thin wrapper on :func:`numpy.interp`, which holds the end values outside the
    given depths, so the profile is flat above the first point and below the last. That
    clamping is what makes a whole profile one call rather than a ramp per transition:

    - a single ramp, such as the weight tying shallow rise time to slip, is two points,
      ``[1, 3]`` against ``[0, 1]``;
    - the two-sided rise-time stretch is four, ``[5, 8, 15, 20]`` against
      ``[2, 1, 1, 2]`` -- longer pulses near the surface, longer again at depth, and
      exactly 1 through the middle of the fault where the profile has nothing to say;
    - a measured profile is however many points were measured.

    Raises
    ------
    RuptureGeneratorError
        If the depths do not increase, or there is not one finite value per depth.
        :func:`numpy.interp` reads unsorted points as a different profile rather than
        as a mistake, so this is the check that matters.
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
    """``shallow`` above the ramp, ``deep`` below it, and linear across
    ``centre_km +- half_width_km``.

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
    layers: Layers, shear_speed_km_s: np.ndarray, density_g_cm3: np.ndarray
) -> Medium:
    """A 1-D velocity model as a medium: one shear speed and density per layer."""
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
