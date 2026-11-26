"""Covariates on a chart, and the profiles that read a field off one.

A profile, such as a noise family's weight or the rise-time factor, is a function of
the fault. The plain form is a :data:`~rupture_generator.rupture.medium.SpatialField`,
a function of position that a model reads at the cell centres, and most of those read
depth alone. A **covariate** is any other number a cell has on its chart, in physical
units, and a profile can be a function of one in place of depth. On a curved
interface, where the fault's top and bottom wander in depth along strike, the
distance down dip from the up-dip edge or the fraction of the way to the down-dip edge
places a transition where depth cannot.

:data:`COVARIATES` names the built-in ones, and a covariate from elsewhere, such as an
interface-coupling raster sampled at :attr:`~rupture_generator.geometry.Geometry.centres`,
is any function of the chart. :func:`covariate_profile` and :func:`covariate_ramp`
build a profile on either kind, as :func:`~rupture_generator.rupture.medium.interpolated_field`
and :func:`~rupture_generator.rupture.medium.ramp_field` build one on depth. A profile
that needs more of the chart than one covariate is a :class:`ChartField`, a spatial
field with an ``on_chart`` method, which :func:`chart_field` makes of a function.
Every model reads a profile by its ``on_chart`` where it has one, and at the cell
centres otherwise, so a plain function of position reads as it always has.

Every covariate but depth is a distance or fraction **on the fault**, a property of
the chart: :attr:`~rupture_generator.geometry.Geometry.dip_km` and the rest, which say
exactly how each is measured. They are measured on each chart, so on a rupture of
several faults each fault's own distances start from its own edges. A chart's row 0
is up dip and its column 0 the start of its trace, as every chart is built. The
distances are across the chart at its spacing, so on a curved chart they carry the
flattening's stretch, about 1%. A cell off the fault takes the value of the fault cell
nearest it on the chart, which keeps every covariate finite on the whole chart, as a
noise weight needs. Depth is defined everywhere and reads the cell's own.
"""

import dataclasses
import functools
import operator
import types
from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.geometry import CellArray, Geometry
from rupture_generator.rupture.medium import DEPTH, SpatialField, checked_profile

type Covariate = Callable[[Geometry], ArrayLike]
"""A covariate from elsewhere: given a chart, one finite value per cell, ``(n_i, n_j)``."""

DEPTH_KM = "depth_km"
"""The covariate a profile reads when it names none, as every profile did before
covariates."""


# ------------------------------------------------------------- the built-in ones


def _depth_km(geometry: Geometry) -> CellArray:
    return geometry.centres[..., DEPTH]


COVARIATES: Mapping[str, Covariate] = types.MappingProxyType(
    {
        DEPTH_KM: _depth_km,
        **{
            name: operator.attrgetter(name)
            for name in (
                "dip_km",
                "dip_fraction",
                "strike_km",
                "strike_fraction",
                "edge_distance_km",
            )
        },
    }
)
"""The built-in covariates by name, the names a rupture file's profile may give.

``depth_km`` is each cell centre's depth, and every other name is the chart's property
of that name: :attr:`~rupture_generator.geometry.Geometry.dip_km` and the rest."""


RANGES: Mapping[str, tuple[float, float]] = types.MappingProxyType(
    {
        "dip_km": (0.0, np.inf),
        "dip_fraction": (0.0, 1.0),
        "strike_km": (0.0, np.inf),
        "strike_fraction": (0.0, 1.0),
        "edge_distance_km": (0.0, np.inf),
    }
)
"""The values each built-in covariate but depth takes on the fault, bounds included.
A profile whose points all lie outside them is one value over the whole fault."""


def check_covariate(name: str) -> str:
    """Check that a name is a built-in covariate's.

    Parameters
    ----------
    name : str
        The name.

    Returns
    -------
    str
        The name.

    Raises
    ------
    RuptureGeneratorError
        If no built-in covariate has the name.
    """
    if name not in COVARIATES:
        raise RuptureGeneratorError(
            f"{name!r} is not a covariate; the covariates are {', '.join(COVARIATES)}"
        )
    return name


def read_covariate(geometry: Geometry, covariate: str | Covariate) -> CellArray:
    """Read a covariate on a chart's cells.

    Parameters
    ----------
    geometry : Geometry
        The chart.
    covariate : str or callable
        A name in :data:`COVARIATES`, or a function of the chart.

    Returns
    -------
    CellArray
        The covariate on every cell, ``(n_i, n_j)``. A built-in's is the chart's own
        cached array, as :attr:`~rupture_generator.geometry.Geometry.centres` is, so
        leave it unchanged.

    Raises
    ------
    RuptureGeneratorError
        If the name is no covariate's, or the covariate isn't one finite value per
        cell.
    """
    if isinstance(covariate, str):
        covariate = COVARIATES[check_covariate(covariate)]
    values = np.asarray(covariate(geometry), dtype=np.float64)
    if values.shape != geometry.cells:
        raise RuptureGeneratorError(
            f"a covariate is shaped {values.shape} and the chart has {geometry.cells} "
            "cells"
        )
    if not np.all(np.isfinite(values)):
        raise RuptureGeneratorError("a covariate is not finite on every cell")
    return values


# ----------------------------------------------------------------------- profiles


@runtime_checkable
class ChartField(Protocol):
    """A :data:`SpatialField` that a model reads on its chart rather than at positions.

    Wherever a model takes a profile, :func:`read_profile` reads one of these by its
    ``on_chart``, so it may read covariates, or anything else of the chart. Called at
    positions, it reads them as it can, or refuses them.
    """

    def __call__(self, positions_km: np.ndarray, /) -> np.ndarray:
        """Read the profile at positions, where it can.

        Parameters
        ----------
        positions_km : numpy.ndarray
            Positions ``(..., 3)``, depth last in kilometres.

        Returns
        -------
        numpy.ndarray
            The profile at each position ``(...)``.
        """
        ...

    def on_chart(self, geometry: Geometry) -> CellArray:
        """Read the profile on a chart's cells.

        Parameters
        ----------
        geometry : Geometry
            The chart.

        Returns
        -------
        CellArray
            One value per cell, ``(n_i, n_j)``.
        """
        ...


def read_profile(profile: SpatialField, geometry: Geometry) -> np.ndarray:
    """Read a profile on a chart's cells, by its ``on_chart`` or at the cell centres.

    Parameters
    ----------
    profile : SpatialField
        The profile: a :class:`ChartField`, read by its ``on_chart``, or any other
        function of position, read at the cell centres.
    geometry : Geometry
        The chart.

    Returns
    -------
    numpy.ndarray
        What the profile gives, unchecked.
    """
    if isinstance(profile, ChartField):
        return profile.on_chart(geometry)
    return profile(geometry.centres)


def _on_chart_only(what: str) -> RuptureGeneratorError:
    return RuptureGeneratorError(
        f"{what} is read on a chart, by on_chart, not at positions"
    )


@dataclasses.dataclass(frozen=True)
class _OnChart:
    """A function of the chart, as :func:`chart_field` makes it a profile."""

    function: Callable[[Geometry], ArrayLike]

    def __call__(self, positions_km: np.ndarray, /) -> np.ndarray:
        """Refuse positions, which hold no chart."""
        raise _on_chart_only("a profile of the chart")

    def on_chart(self, geometry: Geometry) -> CellArray:
        """Read the profile on a chart's cells.

        Parameters
        ----------
        geometry : Geometry
            The chart.

        Returns
        -------
        CellArray
            What the function gives on the chart.
        """
        return np.asarray(self.function(geometry), dtype=np.float64)


def chart_field(function: Callable[[Geometry], ArrayLike]) -> ChartField:
    """Make a function of the chart into a profile.

    Parameters
    ----------
    function : callable
        Given a chart, one value per cell. It can read :func:`read_covariate`, the
        chart's centres, dips and areas, or anything else.

    Returns
    -------
    ChartField
        The profile, which refuses to be read at positions.

    Examples
    --------
    >>> deeper_down_dip = chart_field(lambda geometry: geometry.dip_fraction**2)
    """
    return _OnChart(function)


@dataclasses.dataclass(frozen=True)
class CovariateField:
    """A profile that is a function of one covariate.

    Attributes
    ----------
    covariate : str or callable
        A name in :data:`COVARIATES`, or a function of the chart.
    function : callable
        The profile's value at each covariate value, elementwise.
    """

    covariate: str | Covariate
    function: Callable[[np.ndarray], np.ndarray]

    def __post_init__(self) -> None:
        """Refuse a name that no built-in covariate has."""
        if isinstance(self.covariate, str):
            check_covariate(self.covariate)

    def on_chart(self, geometry: Geometry) -> CellArray:
        """Read the profile on a chart's cells.

        Parameters
        ----------
        geometry : Geometry
            The chart.

        Returns
        -------
        CellArray
            The profile at each cell's covariate.
        """
        return self.function(read_covariate(geometry, self.covariate))

    def __call__(self, positions_km: np.ndarray) -> np.ndarray:
        """Read a depth profile at positions, as a :data:`SpatialField` does.

        Parameters
        ----------
        positions_km : numpy.ndarray
            Positions ``(..., 3)``, depth last in kilometres.

        Returns
        -------
        numpy.ndarray
            The profile at each position's depth ``(...)``.

        Raises
        ------
        RuptureGeneratorError
            For a profile of any covariate but depth, which a position alone doesn't
            give.
        """
        if self.covariate != DEPTH_KM:
            raise _on_chart_only(f"a profile of {self.covariate!r}")
        return self.function(np.asarray(positions_km)[..., DEPTH])


def covariate_profile(
    covariate: str | Covariate, points: ArrayLike, values: ArrayLike
) -> CovariateField:
    """Build a profile of a covariate from points, linear between them, flat beyond.

    :func:`~rupture_generator.rupture.medium.interpolated_field` on a covariate. On
    ``"depth_km"`` it gives the same values as that field does.

    Parameters
    ----------
    covariate : str or callable
        A name in :data:`COVARIATES`, or a function of the chart.
    points : ArrayLike
        The covariate's values at the points, strictly increasing.
    values : ArrayLike
        The profile's value at each point.

    Returns
    -------
    CovariateField
        The profile.

    Raises
    ------
    RuptureGeneratorError
        If the points fail to increase, the values fail to give one finite number per
        point, the covariate's name is no built-in's, or the points lie wholly outside
        the built-in's :data:`RANGES`.
    """
    at, heights = checked_profile(points, values, "points")
    if isinstance(covariate, str) and covariate in RANGES:
        low, high = RANGES[covariate]
        if at[-1] < low or at[0] > high:
            raise RuptureGeneratorError(
                f"a profile from {at[0]:g} to {at[-1]:g} lies wholly outside "
                f"{covariate}'s range, {low:g} to {high:g}, so it is one value over "
                "the whole fault"
            )
    return CovariateField(covariate, functools.partial(_interpolate, at, heights))


def _interpolate(at: np.ndarray, heights: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.interp(x, at, heights)


def covariate_ramp(
    covariate: str | Covariate,
    centre: float,
    half_width: float,
    below: float,
    above: float,
) -> CovariateField:
    """Build a profile that ramps linearly from one value to another along a covariate.

    :func:`~rupture_generator.rupture.medium.ramp_field` on a covariate, its
    ``shallow`` and ``deep`` now the values below and above the ramp.

    Parameters
    ----------
    covariate : str or callable
        A name in :data:`COVARIATES`, or a function of the chart.
    centre : float
        The covariate's value at the middle of the ramp.
    half_width : float
        Half the ramp's extent, in the covariate's units. The ramp spans
        ``centre +- half_width``.
    below : float
        The value where the covariate is below the ramp.
    above : float
        The value where the covariate is above the ramp.

    Returns
    -------
    CovariateField
        The ramp.

    Raises
    ------
    RuptureGeneratorError
        If the ramp has no width, its centre or width isn't finite, or it lies wholly
        outside a built-in covariate's :data:`RANGES`.
    """
    if not half_width > 0.0:
        raise RuptureGeneratorError(f"a ramp {half_width} wide is a step")
    return covariate_profile(
        covariate,
        np.array([centre - half_width, centre + half_width]),
        np.array([below, above]),
    )


__all__ = [
    "COVARIATES",
    "DEPTH_KM",
    "RANGES",
    "ChartField",
    "Covariate",
    "CovariateField",
    "chart_field",
    "check_covariate",
    "covariate_profile",
    "covariate_ramp",
    "read_covariate",
    "read_profile",
]
