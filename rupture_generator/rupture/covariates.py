"""Covariates on a chart, and the profiles that read a field off one.

A profile, such as a noise family's weight or the rise-time factor, is a function of
the fault. The plain form is a :data:`~rupture_generator.rupture.medium.SpatialField`,
a function of position that a model reads at the cell centres, and most of those read
depth alone. A **covariate** is any other number a cell has on its chart, in physical
units, and a profile can be a function of one in place of depth. On a curved
interface, where the fault's top and bottom wander in depth along strike, the
distance down dip from the top edge or the fraction of the way to the bottom edge
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

Every covariate but depth is a distance or fraction **on the fault**, so it is
measured over the chart's fault cells, the ``occupied`` ones. A cell off the fault
takes the value of the fault cell nearest it on the chart, which keeps every covariate
finite on the whole chart, as a noise weight needs. Depth is defined everywhere and
reads the cell's own.
"""

import dataclasses
import functools
import types
from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.geometry import CellArray, Geometry
from rupture_generator.rupture.medium import DEPTH, SpatialField

type Covariate = Callable[[Geometry], ArrayLike]
"""A covariate from elsewhere: given a chart, one finite value per cell, ``(n_i, n_j)``."""

DEPTH_KM = "depth_km"
"""The covariate a profile reads when it names none, as every profile did before
covariates."""


# ------------------------------------------------------------- the built-in ones


def _off_fault_from_nearest(geometry: Geometry, values: np.ndarray) -> CellArray:
    """Give each cell off the fault the value of the fault cell nearest it on the chart.

    Raises
    ------
    RuptureGeneratorError
        For a chart with no fault cell, which has nothing to measure from.
    """
    occupied = geometry.occupied
    if not occupied.any():
        raise RuptureGeneratorError(
            "a chart with no fault cell has no covariate measured on the fault"
        )
    if occupied.all():
        return values
    strike_km, dip_km = geometry.spacing_km
    _, (i, j) = sp.ndimage.distance_transform_edt(
        ~occupied, sampling=(dip_km, strike_km), return_indices=True
    )
    return values[i, j]


def _cell_steps(geometry: Geometry, axis: int) -> CellArray:
    """Each cell's length along one chart axis, in kilometres, ``(n_i, n_j)``.

    The mean of the two node steps on the cell's sides along that axis, so the length
    of the line through the cell's centre.
    """
    steps = np.linalg.norm(np.diff(geometry.nodes, axis=axis), axis=-1)
    return (
        0.5 * (steps[:, :-1] + steps[:, 1:])
        if axis == 0
        else 0.5 * (steps[:-1] + steps[1:])
    )


@functools.lru_cache(maxsize=32)
def _arcs(geometry: Geometry, axis: int) -> tuple[CellArray, CellArray]:
    """Distance and fraction along one chart axis from the fault's first edge.

    Along each column (``axis = 0``) or row (``axis = 1``) of cells, the distance runs
    from the near side of its first fault cell, through each cell's centre, and the
    fraction divides it by the distance to the far side of its last fault cell. Cells
    off the fault between the two count towards both.
    """
    occupied = geometry.occupied
    steps = _cell_steps(geometry, axis)
    to_centre = np.cumsum(steps, axis=axis) - 0.5 * steps
    count = occupied.shape[axis]
    first = np.argmax(occupied, axis=axis)
    last = count - 1 - np.argmax(np.flip(occupied, axis), axis=axis)
    if axis == 0:
        lines = np.arange(occupied.shape[1])
        start = to_centre[first, lines] - 0.5 * steps[first, lines]
        end = to_centre[last, lines] + 0.5 * steps[last, lines]
        start, end = start[None, :], end[None, :]
    else:
        lines = np.arange(occupied.shape[0])
        start = to_centre[lines, first] - 0.5 * steps[lines, first]
        end = to_centre[lines, last] + 0.5 * steps[lines, last]
        start, end = start[:, None], end[:, None]
    distance = to_centre - start
    fraction = distance / (end - start)
    return (
        _frozen(_off_fault_from_nearest(geometry, distance)),
        _frozen(_off_fault_from_nearest(geometry, fraction)),
    )


@functools.lru_cache(maxsize=32)
def _edge_distance(geometry: Geometry) -> CellArray:
    """Distance from each cell centre to the fault's outline, on the chart.

    The outline is every cell side between a fault cell and a cell off the fault or
    off the chart. On a lattice at half the spacing, cell centres and the midpoints and
    ends of cell sides are all lattice points, and the point of a side nearest a
    centre is one of those three, so a Euclidean distance transform on that lattice
    is exact at the chart's spacing.
    """
    occupied = geometry.occupied
    n_i, n_j = occupied.shape
    padded = np.pad(occupied, 1)
    outline = np.zeros((2 * n_i + 1, 2 * n_j + 1), dtype=bool)
    # Sides along strike, on node row a between cell rows a - 1 and a.
    across = padded[:-1, 1:-1] != padded[1:, 1:-1]
    for offset in range(3):
        outline[::2, offset : offset + 2 * n_j : 2] |= across
    # Sides down dip, on node column b between cell columns b - 1 and b.
    along = padded[1:-1, :-1] != padded[1:-1, 1:]
    for offset in range(3):
        outline[offset : offset + 2 * n_i : 2, ::2] |= along
    strike_km, dip_km = geometry.spacing_km
    distance = sp.ndimage.distance_transform_edt(
        ~outline, sampling=(0.5 * dip_km, 0.5 * strike_km)
    )[1::2, 1::2]
    return _frozen(_off_fault_from_nearest(geometry, distance))


def _frozen(values: np.ndarray) -> CellArray:
    """The values as a read-only array, since a cache hands the same one out again."""
    values = np.array(values, dtype=np.float64)
    values.flags.writeable = False
    return values


def depth_km(geometry: Geometry) -> CellArray:
    """Depth of each cell centre, in kilometres below sea level.

    Parameters
    ----------
    geometry : Geometry
        The chart.

    Returns
    -------
    CellArray
        The centres' depths, on every cell.
    """
    return geometry.centres[..., DEPTH]


def dip_km(geometry: Geometry) -> CellArray:
    """Distance down dip from the fault's top edge, in kilometres.

    Measured down each column of cells, through the cell centres, from the top of the
    column's shallowest fault cell. On a curved chart a column follows the unrolled
    sheet's down-dip direction, and where the trench wanders, the top edge does too.

    Parameters
    ----------
    geometry : Geometry
        The chart.

    Returns
    -------
    CellArray
        The distance at each cell centre.
    """
    return _arcs(geometry, 0)[0]


def dip_fraction(geometry: Geometry) -> CellArray:
    """How far from the fault's top edge to its bottom edge each cell lies, in ``[0, 1]``.

    :func:`dip_km` over the length of the cell's column, from the top of its
    shallowest fault cell to the bottom of its deepest. 0 is the top edge and 1 the
    bottom edge, wherever along strike they lie.

    Parameters
    ----------
    geometry : Geometry
        The chart.

    Returns
    -------
    CellArray
        The fraction at each cell centre.
    """
    return _arcs(geometry, 0)[1]


def strike_km(geometry: Geometry) -> CellArray:
    """Distance along strike from the fault's start, in kilometres.

    Measured along each row of cells, through the cell centres, from the start of the
    row's first fault cell.

    Parameters
    ----------
    geometry : Geometry
        The chart.

    Returns
    -------
    CellArray
        The distance at each cell centre.
    """
    return _arcs(geometry, 1)[0]


def strike_fraction(geometry: Geometry) -> CellArray:
    """How far from the fault's start to its end each cell lies along strike, ``[0, 1]``.

    :func:`strike_km` over the length of the cell's row, from the start of its first
    fault cell to the end of its last.

    Parameters
    ----------
    geometry : Geometry
        The chart.

    Returns
    -------
    CellArray
        The fraction at each cell centre.
    """
    return _arcs(geometry, 1)[1]


def edge_distance_km(geometry: Geometry) -> CellArray:
    """Distance to the nearest edge of the fault, in kilometres.

    The edge is the fault's whole outline, its top, bottom and ends and any hole,
    and the distance is the straight line across the chart from the cell centre to the
    outline at the chart's :attr:`~rupture_generator.geometry.Geometry.spacing_km`.
    On a curved chart that is the distance across the unrolled sheet, which the
    flattening keeps to within its stretch, about 1% on the NZ Community Fault Model's
    interfaces.

    Parameters
    ----------
    geometry : Geometry
        The chart.

    Returns
    -------
    CellArray
        The distance at each cell centre: half a cell for a fault cell on the edge.
    """
    return _edge_distance(geometry)


COVARIATES: Mapping[str, Covariate] = types.MappingProxyType(
    {
        DEPTH_KM: depth_km,
        "dip_km": dip_km,
        "dip_fraction": dip_fraction,
        "strike_km": strike_km,
        "strike_fraction": strike_fraction,
        "edge_distance_km": edge_distance_km,
    }
)
"""The built-in covariates by name, the names a rupture file's profile may give."""


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
        The covariate on every cell, ``(n_i, n_j)``.

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
    >>> deeper_down_dip = chart_field(lambda geometry: dip_fraction(geometry) ** 2)
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


def _increasing(points: ArrayLike) -> np.ndarray:
    """Points as a non-empty, finite, strictly increasing 1-D array.

    Raises
    ------
    RuptureGeneratorError
        Otherwise: :func:`numpy.interp` reads unsorted points as a different profile
        rather than as a mistake.
    """
    at = np.asarray(points, dtype=np.float64)
    if at.ndim != 1 or at.size == 0:
        raise RuptureGeneratorError(
            f"the profile's points are shaped {at.shape}; they want a non-empty list"
        )
    if not np.all(np.isfinite(at)):
        raise RuptureGeneratorError(
            f"the profile's points {at.tolist()} are not all finite"
        )
    if np.any(np.diff(at) <= 0.0):
        raise RuptureGeneratorError(
            f"the profile's points {at.tolist()} do not increase"
        )
    return at


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
        point, or the covariate's name is no built-in's.
    """
    at = _increasing(points)
    heights = np.asarray(values, dtype=np.float64)
    if heights.shape != at.shape:
        raise RuptureGeneratorError(
            f"the profile has {heights.size} values for {at.size} points"
        )
    if not np.all(np.isfinite(heights)):
        raise RuptureGeneratorError("a profile value is not finite")
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
        If the ramp has no width, or its centre or width isn't finite.
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
    "ChartField",
    "Covariate",
    "CovariateField",
    "chart_field",
    "check_covariate",
    "covariate_profile",
    "covariate_ramp",
    "depth_km",
    "dip_fraction",
    "dip_km",
    "edge_distance_km",
    "read_covariate",
    "read_profile",
    "strike_fraction",
    "strike_km",
]
