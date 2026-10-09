"""What the rock is like at each subfault, and how it gets onto a chart.

No depth profile is compiled into anything downstream, so *any* depth dependence -- a
slower front near the surface, longer pulses in the shallow crust -- is prescribed
here, per cell, where it can be inspected and plotted before a rupture is drawn.

A :data:`CellSampler` is the whole interface: given the positions of a chart's cell
centres, return one value per cell. Nothing here knows what a fault is, so a sampler is
a pure function of position and testable on its own. The functions below build one by
closing over a model, and :func:`sample_materials` runs them over a chart.

Units follow the field names. Shear speed is kilometres per second and density grams
per cubic centimetre, which is how a 1-D velocity model is written down; rigidity is
derived from them in pascals, and the single ``1e9`` in :attr:`Materials.rigidity_pa`
is the whole conversion.
"""

import dataclasses
import functools
from collections.abc import Callable

import numpy as np

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.geometry import CellArray, Geometry, NodeArray

type CellSampler = Callable[[NodeArray], CellArray]
"""Given cell centres, ``(n_i, n_j, 3)``, one value per cell, ``(n_i, n_j)``.

Positions are the chart's own: east, north and depth in kilometres, depth positive
down. A sampler that depends only on depth reads ``centres_km[..., 2]``.
"""

DEPTH = 2
"""Which column of a position is depth."""

PA_PER_KM_S_SQUARED_G_CM3 = 1.0e9
"""``(1e3 m/s)^2 x (1e3 kg/m^3)``: what carries a velocity model's own units to SI."""


@dataclasses.dataclass(frozen=True, eq=False)
class Materials:
    """What one chart's rupture reads off the rock, one value per cell.

    Attributes
    ----------
    shear_speed_km_s : CellArray
        What the front's speed is a fraction of.
    density_g_cm3 : CellArray
        With the shear speed, what the rigidity the moment is counted in comes from.
    rise_time_factor : CellArray or float
        The relative rise time, 1 where it is unmodified: any depth dependence of the
        pulse length is prescribed here.
    rise_time_slip_weight : CellArray or float
        How much of rise time's correlation with slip is the configured value (1)
        rather than exact (0). At 0 the rise-time latent is slip's own, so the two
        fields share their rank order -- the shallow treatment of Graves & Pitarka,
        where the pulse length tracks the slip in the velocity-strengthening crust.
    """

    shear_speed_km_s: CellArray
    density_g_cm3: CellArray
    rise_time_factor: CellArray | float = 1.0
    rise_time_slip_weight: CellArray | float = 1.0

    def __post_init__(self) -> None:
        """Refuse a slip weight outside ``[0, 1]``."""
        weight = np.asarray(self.rise_time_slip_weight)
        if not np.all((weight >= 0.0) & (weight <= 1.0)):
            raise RuptureGeneratorError(
                f"the rise-time slip weight runs {float(weight.min()):.3g} to "
                f"{float(weight.max()):.3g}; a weight lies in [0, 1]"
            )

    @functools.cached_property
    def rigidity_pa(self) -> CellArray:
        """Rigidity in pascals, :math:`\\mu = \\rho v_s^2`.

        In the velocity model's own units of km/s and g/cm^3, carried to SI by a single
        factor. Crustal rock is about 3e10 Pa.
        """
        return self.density_g_cm3 * self.shear_speed_km_s**2 * PA_PER_KM_S_SQUARED_G_CM3


def sample_materials(
    geometry: Geometry,
    *,
    shear_speed_km_s: CellSampler,
    density_g_cm3: CellSampler,
    rise_time_factor: CellSampler | None = None,
    rise_time_slip_weight: CellSampler | None = None,
) -> Materials:
    """Run each sampler over a chart's cell centres.

    Raises
    ------
    RuptureGeneratorError
        If a sampler returns the wrong shape for the chart.
    """
    centres_km = geometry.centres

    def sample(name: str, cell_sampler: CellSampler) -> CellArray:
        values = cell_sampler(centres_km)
        if values.shape != geometry.cells:
            raise RuptureGeneratorError(
                f"{name} is shaped {values.shape} and the chart has {geometry.cells} cells"
            )
        return values

    optional = {
        name: sample(name, cell_sampler)
        for name, cell_sampler in (
            ("rise_time_factor", rise_time_factor),
            ("rise_time_slip_weight", rise_time_slip_weight),
        )
        if cell_sampler is not None
    }
    return Materials(
        shear_speed_km_s=sample("shear_speed_km_s", shear_speed_km_s),
        density_g_cm3=sample("density_g_cm3", density_g_cm3),
        **optional,
    )


def constant_sampler(value: float) -> CellSampler:
    """The same value at every subfault."""

    def sample(centres_km: NodeArray) -> CellArray:
        return np.full(np.shape(centres_km)[:-1], float(value), dtype=np.float64)

    return sample


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


def layered_1d_sampler(
    layers: Layers, per_layer: np.ndarray, *, name: str
) -> CellSampler:
    """A property that depends on depth alone, read off a 1-D model.

    Sampled at each subfault's **own** depth rather than once per dip row: one lookup
    broadcast along strike is exact for a plane and for nothing else.
    """
    values = layers.values(name, per_layer)

    def sample(centres_km: NodeArray) -> CellArray:
        return values[layers.layer_for(centres_km[..., DEPTH])]

    return sample


def interpolated_sampler(depth_km: np.ndarray, values: np.ndarray) -> CellSampler:
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

    def sample(centres_km: NodeArray) -> CellArray:
        return np.interp(centres_km[..., DEPTH], depths, heights)

    return sample


def velocity_model(
    layers: Layers, shear_speed_km_s: np.ndarray, density_g_cm3: np.ndarray
) -> tuple[CellSampler, CellSampler]:
    """Shear speed and density from one 1-D model, in that order::

    speed, density = velocity_model(layers, shear_speed_km_s, density_g_cm3)
    materials = sample_materials(
        chart, shear_speed_km_s=speed, density_g_cm3=density
    )
    """
    return (
        layered_1d_sampler(layers, shear_speed_km_s, name="shear_speed_km_s"),
        layered_1d_sampler(layers, density_g_cm3, name="density_g_cm3"),
    )


__all__ = [
    "DEPTH",
    "CellSampler",
    "Layers",
    "Materials",
    "constant_sampler",
    "interpolated_sampler",
    "layered_1d_sampler",
    "sample_materials",
    "velocity_model",
]
