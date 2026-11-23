"""A chart, which describes one fault as a grid of quadrilateral cells.

Positions are east, north and depth in kilometres in the realisation's projected CRS,
depth positive down. ``i`` runs down dip and ``j`` along strike. A chart is one or more
planes laid side by side along strike, each sharing its junction column of nodes with
the next, so the chart is one continuous grid. Its ``occupied`` mask says which cells
are fault. The spacing and the cell a position names come from occupied cells only,
so nodes off the fault need only be finite.
"""

import dataclasses
import functools
import itertools

import numpy as np
import scipy as sp

from rupture_generator.errors import RuptureGeneratorError

type NodeArray = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
"""Node positions, ``(n_i+1, n_j+1, 3)``."""

type CellArray = np.ndarray[tuple[int, int], np.dtype[np.float64]]
"""One value per cell, ``(n_i, n_j)``."""

type CellMask = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
"""One flag per cell, ``(n_i, n_j)``."""

type CellSelection = tuple[np.ndarray, np.ndarray]
"""Some cells named as two index arrays, ``(i, j)``: what indexing a field wants."""

MAXIMUM_HYPOCENTRE_OFFSET_KM = 1.0
"""How far in plan, past half a cell's diagonal, a hypocentre given as a position may
lie from the centre of the fault cell :meth:`Geometry.cell_near` moves it to.

A position on the fault lies within half a diagonal of its own cell's centre, so the
half diagonal admits every point of the fault at any spacing, and the kilometre past it
admits a position just beyond the fault's outline."""

VERTICAL_TOLERANCE = 1e-8
"""How near vertical, as ``|cos(dip)|``, or horizontal, as ``sin(dip)``, a cell's
normal leaves its strike to the cell's ``j`` edge.

Projected positions reach 1e4 km, where double precision rounds to 2e-12 km, so a 10 m
cell's normal can tilt by about 2e-10 from rounding alone. The tolerance is fifty times
that, 6e-7 degrees."""


def bordering(mask: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """Flag the nodes or node steps that border a flagged cell.

    Each axis in ``axes`` grows the mask by one: an entry is flagged when the cell on
    either side of it along that axis is. ``(0, 1)`` gives the nodes cornering a
    flagged cell, ``(n_i+1, n_j+1)``; ``(0,)`` the steps along strike that border one,
    ``(n_i+1, n_j)``; and ``(1,)`` the steps down dip, ``(n_i, n_j+1)``.

    Parameters
    ----------
    mask : np.ndarray
        One flag per cell, ``(n_i, n_j)``.
    axes : tuple of int
        The axes to grow along.

    Returns
    -------
    np.ndarray
        The grown mask.
    """
    for axis in axes:
        before, after = [(0, 0), (0, 0)], [(0, 0), (0, 0)]
        before[axis], after[axis] = (1, 0), (0, 1)
        mask = np.pad(mask, before) | np.pad(mask, after)
    return mask


def _position_to_cell_index(
    position_km: float, arc_km: np.ndarray, *, axis: str
) -> int:
    """The cell containing a position along an arc of node distances.

    A position on an interior boundary belongs to the cell after it, and one on the far
    edge belongs to the last cell.
    """
    extent = float(arc_km[-1])
    if not 0.0 <= position_km <= extent:
        raise RuptureGeneratorError(
            f"{position_km} km is off the fault along {axis}, which runs 0 to "
            f"{extent:.3f} km"
        )
    index = int(np.searchsorted(arc_km, position_km, side="right")) - 1
    return min(index, len(arc_km) - 2)


def _resample(plane: NodeArray, rows: int, columns: int) -> NodeArray:
    """One plane's nodes bilinearly resampled onto ``rows x columns`` cells."""
    interpolate = sp.interpolate.RegularGridInterpolator(
        (np.arange(plane.shape[0]), np.arange(plane.shape[1])), plane
    )
    down = np.linspace(0.0, plane.shape[0] - 1, rows + 1)
    along = np.linspace(0.0, plane.shape[1] - 1, columns + 1)
    return interpolate(np.stack(np.meshgrid(down, along, indexing="ij"), axis=-1))


@dataclasses.dataclass(frozen=True, eq=False)
class Geometry:
    """A planar geometric description of a discretised fault.

    A chart caches each derived quantity on first use, which is safe because a chart
    never changes after construction.

    Attributes
    ----------
    nodes : NodeArray
        Node positions, ``(n_i+1, n_j+1, 3)``, kilometres in the projected CRS.
    occupied : CellMask
        Which cells are fault. All true for a chart built from planes; a chart built
        directly may leave part of its rectangle unoccupied.
    plane_cells : tuple of int
        How many cell columns each plane has, in trace order. Each plane writes as
        its own SRF plane.
    """

    nodes: NodeArray
    occupied: CellMask
    plane_cells: tuple[int, ...]

    def __post_init__(self) -> None:
        """Check the shapes agree."""
        nodes = self.nodes
        if nodes.ndim != 3 or nodes.shape[-1] != 3 or nodes.shape[0] < 2:
            raise RuptureGeneratorError(
                f"nodes are shaped {nodes.shape}; a chart wants (n_i+1, n_j+1, 3) "
                "with at least one row of cells"
            )
        if not self.plane_cells or min(self.plane_cells) < 1:
            raise RuptureGeneratorError(
                f"every plane needs at least one cell column, got {self.plane_cells}"
            )
        columns = sum(self.plane_cells) + 1
        if nodes.shape[1] != columns:
            raise RuptureGeneratorError(
                f"{len(self.plane_cells)} planes of {self.plane_cells} cells want "
                f"{columns} node columns, and the nodes have {nodes.shape[1]}; "
                "neighbouring planes share their junction column"
            )
        if not np.all(np.isfinite(nodes)):
            raise RuptureGeneratorError("nodes contain a NaN or infinity")
        if self.occupied.shape != self.cells:
            raise RuptureGeneratorError(
                f"the occupied mask is shaped {self.occupied.shape} and the chart has "
                f"{self.cells} cells"
            )

    def __repr__(self) -> str:
        """The shape, not the arrays."""
        n_i, n_j = self.cells
        planes = "" if self.planes == 1 else f" over {self.planes} planes"
        return f"{type(self).__name__}({n_i}x{n_j} cells{planes})"

    @property
    def cells(self) -> tuple[int, int]:
        """tuple[int, int]: Cell counts down dip and along strike, ``(n_i, n_j)``."""
        return (self.nodes.shape[0] - 1, sum(self.plane_cells))

    @property
    def planes(self) -> int:
        """int: How many planes make up the chart."""
        return len(self.plane_cells)

    # ---------------------------------------------------------- one plane at a time

    def plane_nodes(self) -> list[NodeArray]:
        """List each plane's node grid, as a view rather than a copy.

        Neighbouring views overlap in the junction column they share.

        Returns
        -------
        list of NodeArray
            One ``(n_i+1, columns+1, 3)`` view per plane, in trace order.
        """
        edges = np.cumsum([0, *self.plane_cells])
        return [self.nodes[:, a : b + 1] for a, b in itertools.pairwise(edges)]

    @functools.cached_property
    def centres(self) -> NodeArray:
        """NodeArray: Cell centres, ``(n_i, n_j, 3)``, the mean of the corners."""
        p = self.nodes
        return 0.25 * (p[:-1, :-1] + p[:-1, 1:] + p[1:, 1:] + p[1:, :-1])

    @functools.cached_property
    def _diagonal_normals(self) -> NodeArray:
        """NodeArray: ``AC x BD`` per cell, along its normal, twice its area long."""
        p = self.nodes
        return np.cross(p[1:, 1:] - p[:-1, :-1], p[1:, :-1] - p[:-1, 1:])

    @functools.cached_property
    def areas_km2(self) -> CellArray:
        """CellArray: Cell areas in square kilometres, ``(n_i, n_j)``.

        Each area is half the cross product of the cell's diagonals, exact for a planar
        quadrilateral. For a warped one it gives the area projected onto the plane the
        two diagonals span.
        """
        return 0.5 * np.linalg.norm(self._diagonal_normals, axis=-1)

    @functools.cached_property
    def dip_deg(self) -> CellArray:
        """CellArray: Cell dips from horizontal in degrees, 0 for a degenerate cell."""
        normal = self._diagonal_normals
        length = np.linalg.norm(normal, axis=-1)
        cosine = np.divide(
            np.abs(normal[..., 2]), length, out=np.ones_like(length), where=length > 0
        )
        return np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))

    @functools.cached_property
    def strike(self) -> np.ndarray:
        """np.ndarray: Unit horizontal vectors, ``(n_i, n_j, 2)``, along each strike.

        Each is east and north in the projected CRS. It's the horizontal direction in
        the plane the cell's diagonals span, oriented so the fault dips to its right.
        Where that plane lies within :data:`VERTICAL_TOLERANCE` of vertical,
        which dips to neither side, or of horizontal, which has no strike line, the
        vector follows the horizontal part of the cell's ``j`` edges instead. A cell
        with neither reads as east, ``(1, 0)``, as :attr:`dip_deg` reads 0 for a
        degenerate cell.
        """
        normal = self._diagonal_normals
        tolerance = VERTICAL_TOLERANCE * 2.0 * self.areas_km2
        # With depth positive down, an upward normal's horizontal part points down dip;
        # a quarter turn anticlockwise of the dip direction is the strike.
        direction = np.sign(normal[..., 2:]) * np.stack(
            [normal[..., 1], -normal[..., 0]], -1
        )
        undecided = (np.abs(normal[..., 2]) <= tolerance) | (
            np.hypot(direction[..., 0], direction[..., 1]) <= tolerance
        )
        i, j = np.nonzero(undecided)
        p = self.nodes[..., :2]
        direction[i, j] = p[i, j + 1] - p[i, j] + p[i + 1, j + 1] - p[i + 1, j]
        size = np.linalg.norm(direction, axis=-1, keepdims=True)
        east = np.broadcast_to([1.0, 0.0], direction.shape)
        return np.divide(direction, size, out=east.copy(), where=size > 0)

    @functools.cached_property
    def spacing_km(self) -> tuple[float, float]:
        """tuple[float, float]: The chart's ``(strike, dip)`` spacing in kilometres.

        The sampler and the eikonal solve both take this one spacing. It averages
        every along-strike and every down-dip node step that borders an occupied cell,
        rather than reading one edge, because a plane hung from a bent trace is a
        trapezoid, whose steps vary from one edge to the other.

        Raises
        ------
        RuptureGeneratorError
            For a chart with no occupied cell, which has no step on the fault.
        """
        occupied = self.occupied
        if not occupied.any():
            raise RuptureGeneratorError(
                "a chart with no occupied cell has no spacing on the fault"
            )
        strike = np.linalg.norm(np.diff(self.nodes, axis=1), axis=-1)
        dip = np.linalg.norm(np.diff(self.nodes, axis=0), axis=-1)
        return (
            float(np.mean(strike, where=bordering(occupied, (0,)))),
            float(np.mean(dip, where=bordering(occupied, (1,)))),
        )

    @functools.cached_property
    def strike_arc_km(self) -> np.ndarray:
        """np.ndarray: Distance along strike of each top-edge cell boundary, in km.

        Shaped ``(n_j+1,)``. With :attr:`dip_arc_km` it lets a hypocentre be two
        lengths rather than two indices.
        """
        steps = np.linalg.norm(np.diff(self.nodes[0], axis=0), axis=-1)
        return np.concatenate([[0.0], np.cumsum(steps)])

    @functools.cached_property
    def dip_arc_km(self) -> np.ndarray:
        """np.ndarray: Distance down dip of each ``j = 0`` edge node row, in km."""
        steps = np.linalg.norm(np.diff(self.nodes[:, 0], axis=0), axis=-1)
        return np.concatenate([[0.0], np.cumsum(steps)])

    def cell_at(self, strike_km: float, dip_km: float) -> tuple[int, int]:
        """Find the cell containing a position on the fault.

        The distances run along the chart's top row and first column, which lie off
        the fault where the chart leaves those cells unoccupied.

        Parameters
        ----------
        strike_km : float
            Distance along strike from the chart's first top corner, in kilometres.
        dip_km : float
            Distance down dip from the top edge, in kilometres.

        Returns
        -------
        tuple[int, int]
            The cell's 0-based ``(i, j)``.

        Raises
        ------
        RuptureGeneratorError
            For a position off the chart, naming the axis and the chart's extent, or
            one in a cell off the fault.
        """
        cell = (
            _position_to_cell_index(dip_km, self.dip_arc_km, axis="dip"),
            _position_to_cell_index(strike_km, self.strike_arc_km, axis="strike"),
        )
        if not self.occupied[cell]:
            raise RuptureGeneratorError(
                f"{strike_km} km along strike and {dip_km} km down dip is cell {cell}, "
                "which is off the fault"
            )
        return cell

    def centre_arc_km(self, i: int, j: int) -> tuple[float, float]:
        """The distances that name a cell's centre, the inverse of :meth:`cell_at`.

        Parameters
        ----------
        i, j : int
            The cell's 0-based row and column.

        Returns
        -------
        tuple[float, float]
            Its centre's distance along strike and down dip, in kilometres, measured
            as :meth:`cell_at` measures them.
        """
        strike, dip = self.strike_arc_km, self.dip_arc_km
        return 0.5 * float(strike[j] + strike[j + 1]), 0.5 * float(dip[i] + dip[i + 1])

    def cell_near(self, east_km: float, north_km: float) -> tuple[int, int]:
        """Find the occupied cell whose centre is nearest a position in plan.

        Parameters
        ----------
        east_km : float
            Easting in the projected CRS, in kilometres.
        north_km : float
            Northing in the projected CRS, in kilometres.

        Returns
        -------
        tuple[int, int]
            The cell's 0-based ``(i, j)``.

        Raises
        ------
        RuptureGeneratorError
            For a chart with no occupied cell, which has no spacing, or a position
            farther in plan from every occupied cell's centre than half the diagonal
            of a cell of :attr:`spacing_km` and :data:`MAXIMUM_HYPOCENTRE_OFFSET_KM`
            together, naming the distance.
        """
        # The spacing refuses a chart with no occupied cell.
        reach_km = MAXIMUM_HYPOCENTRE_OFFSET_KM + 0.5 * float(
            np.hypot(*self.spacing_km)
        )
        centres = self.centres
        distance_km = np.hypot(centres[..., 0] - east_km, centres[..., 1] - north_km)
        distance_km[~self.occupied] = np.inf
        nearest = np.unravel_index(np.argmin(distance_km), distance_km.shape)
        if not distance_km[nearest] <= reach_km:
            raise RuptureGeneratorError(
                f"({east_km}, {north_km}) km lies {distance_km[nearest]:.3f} km in plan "
                f"from the nearest fault cell's centre, past the {reach_km:.3f} km a "
                "position may be moved onto it: half a cell's diagonal and "
                f"{MAXIMUM_HYPOCENTRE_OFFSET_KM} km more"
            )
        return int(nearest[0]), int(nearest[1])

    def subdivide(self, spacing_km: float) -> "Geometry":
        """Recut this chart into cells of about ``spacing_km`` on a side.

        Bilinear interpolation over each plane's own nodes gives the new nodes, with
        the cell count along each plane rounded and at least one. The planes of one
        chart share a dip extent, so all of them take the row count the ``j = 0``
        edge implies, and a junction column resamples the same from either side.

        Parameters
        ----------
        spacing_km : float
            The target cell size in kilometres.

        Returns
        -------
        Geometry
            The recut chart.

        Raises
        ------
        RuptureGeneratorError
            For a spacing that fails to be a positive length, or a chart with any cell
            off the fault, since a partial outline gives nothing to resample it by.
        """
        if not spacing_km > 0.0 or not np.isfinite(spacing_km):
            raise RuptureGeneratorError(
                f"a subfault size of {spacing_km} km is not a positive length"
            )
        if not self.occupied.all():
            raise RuptureGeneratorError(
                "only a chart that is fault everywhere subdivides"
            )
        rows = max(1, round(float(self.dip_arc_km[-1]) / spacing_km))
        planes, columns = [], []
        for plane in self.plane_nodes():
            length_km = float(np.linalg.norm(np.diff(plane[0], axis=0), axis=-1).sum())
            columns.append(max(1, round(length_km / spacing_km)))
            resampled = _resample(plane, rows, columns[-1])
            # Every plane after the first starts on the junction its neighbour ended on.
            planes.append(resampled if not planes else resampled[:, 1:])
        return Geometry(
            nodes=np.concatenate(planes, axis=1),
            occupied=np.ones((rows, sum(columns)), dtype=bool),
            plane_cells=tuple(columns),
        )

    # -------------------------------------------------------- between two charts

    @functools.cached_property
    def _edge_cells(self) -> tuple[CellSelection, np.ndarray]:
        """tuple[CellSelection, np.ndarray]: Edge cells and their centres.

        An edge cell is a fault cell with a neighbour off the fault or off the grid,
        where a front can leave this chart or arrive from another. The code averages
        their centres from the nodes rather than reading :attr:`centres`, because on a
        large chart the edge is a small fraction of the cells.
        """
        i, j = np.nonzero(
            self.occupied & ~sp.ndimage.binary_erosion(self.occupied, border_value=0)
        )
        nodes = self.nodes
        centres = 0.25 * (
            nodes[i, j] + nodes[i, j + 1] + nodes[i + 1, j + 1] + nodes[i + 1, j]
        )
        return (i, j), centres

    @functools.cached_property
    def _edge(self) -> tuple[CellSelection, sp.spatial.cKDTree]:
        """tuple[CellSelection, sp.spatial.cKDTree]: Edge cells and a tree over them."""
        cells, centres = self._edge_cells
        return cells, sp.spatial.cKDTree(centres)

    @functools.cached_property
    def _fault(self) -> tuple[CellSelection, sp.spatial.cKDTree]:
        """tuple[CellSelection, sp.spatial.cKDTree]: Fault cells, a tree over them."""
        i, j = np.nonzero(self.occupied)
        return (i, j), sp.spatial.cKDTree(self.centres[i, j])

    def jump_cells(self, *, interior: bool = False) -> tuple[CellSelection, np.ndarray]:
        """The cells a jump can cross from or onto, and their centres.

        Parameters
        ----------
        interior : bool
            Every fault cell, not only edge cells.

        Returns
        -------
        tuple[CellSelection, np.ndarray]
            The cells, and their centres, ``(n, 3)``, in kilometres.
        """
        if not interior:
            return self._edge_cells
        i, j = np.nonzero(self.occupied)
        return (i, j), self.centres[i, j]

    def nearest_cells_to(
        self, other: "Geometry", *, interior: bool = False
    ) -> tuple[CellSelection, CellSelection, np.ndarray]:
        """Pair each of this chart's edge cells with the closest edge cell of another.

        Parameters
        ----------
        other : Geometry
            The chart to search.
        interior : bool
            Pair every fault cell on both charts, not only edge cells. Intersecting
            faults meet away from their edges.

        Returns
        -------
        tuple[CellSelection, CellSelection, np.ndarray]
            This chart's edge (or fault) cells, the cell of ``other`` nearest each, and
            the straight-line distance between each pair in kilometres, as parallel
            arrays.

        Raises
        ------
        RuptureGeneratorError
            If either chart has no fault cells.
        """
        (here, mine), (there, theirs) = (
            (self._fault, other._fault) if interior else (self._edge, other._edge)
        )
        if here[0].size == 0 or there[0].size == 0:
            raise RuptureGeneratorError(
                "a distance between charts needs fault on both sides, and one of these "
                "charts is entirely unoccupied"
            )
        distance_km, nearest = theirs.query(mine.data)
        return here, (there[0][nearest], there[1][nearest]), distance_km
