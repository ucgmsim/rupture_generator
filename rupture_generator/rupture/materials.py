"""What the rock is like at each subfault, and how it gets onto a chart.

Sampling materials is a stage: it takes a fault system and puts fields on its charts.
What reads them afterwards is not this module's concern, and no depth profile is
compiled into anything downstream, so *any* depth dependence -- a slower front near the
surface, longer pulses in the shallow crust -- is prescribed here, per cell, where it
can be inspected and plotted before a rupture is drawn.

A :data:`Sampler` is the whole interface: given the positions of a chart's cell
centres, return one value per cell. Nothing here knows what a fault is, so a sampler is
a pure function of position and testable on its own. The functions below build one by
closing over a model -- a partial application, spelled as a closure -- and
:func:`sample_materials` is what runs them over a fault system.

Nothing here knows what a field will be called, or which of them some later stage
insists on. A sampler produces values and the caller names them, so this module sits on
its own: the field vocabulary stays with the code that reads it, and the order of the
stages lives in the command line rather than in an import.

Units follow the field names. Shear speed is kilometres per second and density grams
per cubic centimetre, which is how a 1-D velocity model is written down; rigidity comes
out in pascals, and the single ``1e9`` in :func:`rigidity_pa` is the whole conversion.
"""

import dataclasses
from collections.abc import Callable, Mapping

import numpy as np

from rupture_generator.geometry import CellArray, Geometry, NodeArray
from rupture_generator.rupture.realisation import Realisation

type Sampler = Callable[[NodeArray], CellArray]
"""Given cell centres, ``(n_i, n_j, 3)``, one value per cell, ``(n_i, n_j)``.

Positions are the chart's own: east, north and depth in kilometres, depth positive
down. A sampler that depends only on depth reads ``centres_km[..., 2]``.
"""

DEPTH = 2
"""Which column of a position is depth."""

PA_PER_KM_S_SQUARED_G_CM3 = 1.0e9
"""``(1e3 m/s)^2 x (1e3 kg/m^3)``: what carries a velocity model's own units to SI."""


class MaterialError(ValueError):
    """A material model no rock answers to."""


def rigidity_pa(
    shear_speed_km_s: float | np.ndarray, density_g_cm3: float | np.ndarray
) -> np.ndarray:
    """Rigidity in pascals, from shear speed and density.

    :math:`\\mu = \\rho v_s^2`, in the velocity model's own units of km/s and g/cm^3,
    carried to SI by a single factor. Crustal rock is about 3e10 Pa.
    """
    return (
        np.asarray(density_g_cm3, dtype=np.float64)
        * np.asarray(shear_speed_km_s, dtype=np.float64) ** 2
        * PA_PER_KM_S_SQUARED_G_CM3
    )


def constant_sampler(value: float) -> Sampler:
    """The same value at every subfault.

    A uniform half-space, and the thing to reach for when a field is a placeholder or
    a study is holding one property fixed on purpose.
    """

    def sample(centres_km: NodeArray) -> CellArray:
        return np.full(np.shape(centres_km)[:-1], float(value), dtype=np.float64)

    return sample


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
        bottoms = np.asarray(self.bottom_depth_km, dtype=np.float64)
        if bottoms.ndim != 1 or bottoms.size == 0:
            raise MaterialError(
                f"the layer boundaries are shaped {bottoms.shape}; a 1-D model wants a "
                "non-empty list of depths"
            )
        if np.any(np.diff(bottoms) <= 0.0):
            raise MaterialError(
                f"the layer boundaries {bottoms.tolist()} do not increase, so it is "
                "ambiguous which layer a depth falls in"
            )
        if bottoms[0] <= 0.0:
            raise MaterialError(
                f"the shallowest layer reaches {bottoms[0]} km, which is not below the "
                "surface"
            )
        object.__setattr__(self, "bottom_depth_km", bottoms)

    def __len__(self) -> int:
        """How many layers."""
        return len(self.bottom_depth_km)

    def layer_for(self, depth_km: np.ndarray) -> np.ndarray:
        """Which layer each depth falls in, shaped like ``depth_km``."""
        return np.minimum(
            np.searchsorted(
                self.bottom_depth_km,
                np.asarray(depth_km, dtype=np.float64),
                side="left",
            ),
            len(self) - 1,
        )

    def values(self, name: str, per_layer: np.ndarray) -> np.ndarray:
        """One value per layer, checked against the boundaries.

        Raises
        ------
        MaterialError
            If there is not exactly one value per layer, or one of them is not positive.
        """
        values = np.asarray(per_layer, dtype=np.float64)
        if values.shape != (len(self),):
            raise MaterialError(
                f"{name} has {values.size} values for {len(self)} layers"
            )
        if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
            raise MaterialError(
                f"{name} runs {values.min()} to {values.max()}; every layer needs a "
                "positive, finite value"
            )
        return values


def layered_1d_sampler(layers: Layers, per_layer: np.ndarray, *, name: str) -> Sampler:
    """A property that depends on depth alone, read off a 1-D model.

    Sampled at each subfault's **own** depth rather than once per dip row: one lookup
    broadcast along strike is exact for a plane and for nothing else, and a curved
    interface is the case this package exists to handle.
    """
    values = layers.values(name, per_layer)

    def sample(centres_km: NodeArray) -> CellArray:
        return values[layers.layer_for(np.asarray(centres_km)[..., DEPTH])]

    return sample


def rigidity_sampler(
    layers: Layers, shear_speed_km_s: np.ndarray, density_g_cm3: np.ndarray
) -> Sampler:
    """Rigidity from a 1-D model, in pascals.

    The one material a rupture needs that a velocity model does not list directly. It
    is here rather than left to the caller because :math:`\\rho v_s^2` and its unit
    factor are what a hand-rolled version gets wrong.
    """
    speed = layers.values("shear_speed_km_s", shear_speed_km_s)
    density = layers.values("density_g_cm3", density_g_cm3)
    rigidity = rigidity_pa(speed, density)

    def sample(centres_km: NodeArray) -> CellArray:
        return rigidity[layers.layer_for(np.asarray(centres_km)[..., DEPTH])]

    return sample


def interpolated_sampler(depth_km: np.ndarray, values: np.ndarray) -> Sampler:
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

    One point is a constant, though :func:`constant_sampler` says that more plainly.

    Raises
    ------
    MaterialError
        If the depths do not increase, or there is not one value per depth.
        :func:`numpy.interp` reads unsorted points as a different profile rather than
        as a mistake, so this is the check that matters.
    """
    depths = np.asarray(depth_km, dtype=np.float64)
    heights = np.asarray(values, dtype=np.float64)
    if depths.ndim != 1 or depths.size == 0:
        raise MaterialError(
            f"the profile depths are shaped {depths.shape}; a profile wants a "
            "non-empty list of depths"
        )
    if heights.shape != depths.shape:
        raise MaterialError(
            f"the profile has {heights.size} values for {depths.size} depths"
        )
    if np.any(np.diff(depths) <= 0.0):
        raise MaterialError(
            f"the profile depths {depths.tolist()} do not increase, and interpolating "
            "between them in that order describes a different profile"
        )
    if not np.all(np.isfinite(heights)):
        raise MaterialError("a profile value is not finite")

    def sample(centres_km: NodeArray) -> CellArray:
        return np.interp(np.asarray(centres_km)[..., DEPTH], depths, heights)

    return sample


def velocity_model(
    layers: Layers, shear_speed_km_s: np.ndarray, density_g_cm3: np.ndarray
) -> tuple[Sampler, Sampler]:
    """Shear speed and rigidity from one 1-D model, in that order.

    A convenience over :func:`layered_1d_sampler` and :func:`rigidity_sampler` so that
    the common case is one call and the two cannot disagree about which model they came
    from. Returned unnamed, because what a chart calls them is the caller's business::

        speed, rigidity = velocity_model(layers, shear_speed_km_s, density_g_cm3)
        realisation = sample_materials(
            realisation, {Field.SHEAR_SPEED: speed, Field.RIGIDITY: rigidity}
        )
    """
    return (
        layered_1d_sampler(layers, shear_speed_km_s, name="shear_speed_km_s"),
        rigidity_sampler(layers, shear_speed_km_s, density_g_cm3),
    )


def sample_materials(
    realisation: Realisation, samplers: Mapping[str, Sampler]
) -> Realisation:
    """Run every sampler over every segment and attach what comes back.

    One set of samplers for the whole system, since a velocity model is regional and a
    fault system sits inside it. A segment that needs its own takes a second call, or
    :meth:`~rupture_generator.rupture.realisation.Realisation.replace` by hand.

    Whatever the samplers are keyed by becomes a field name, and nothing here checks
    that against what a rupture will later want: the module that reads a field is the
    one that should say it is missing.

    Returns the realisation with the sampled fields on every chart. Existing fields of
    the same name are replaced, so re-sampling is how a model is corrected.

    Raises
    ------
    GeometryError
        If a sampler returns the wrong shape for the chart it was given.
    """
    sampled: dict[str, Geometry] = {}
    for name, chart in realisation.items():
        # Once per chart, not once per field: on a production-resolution interface the
        # centres are the expensive part and every sampler reads the same ones.
        centres_km = chart.centres()
        sampled[name] = chart.with_fields(
            **{field: sampler(centres_km) for field, sampler in samplers.items()}
        )
    return realisation.replace(**sampled)


__all__ = [
    "DEPTH",
    "Layers",
    "MaterialError",
    "Sampler",
    "constant_sampler",
    "interpolated_sampler",
    "layered_1d_sampler",
    "rigidity_pa",
    "rigidity_sampler",
    "sample_materials",
    "velocity_model",
]
