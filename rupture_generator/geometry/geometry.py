"""One chart: a fault surface as a grid of quadrilateral cells.

Positions are east, north and depth in kilometres in the realisation's projected CRS,
depth positive down. ``i`` runs down dip and ``j`` along strike. A chart is one or more
planes laid side by side along strike, and each keeps its **own** seam column of nodes,
so two planes hung from a bent trace meet along the trace and are free to part below
it, which is what a kinked fault actually does.

Cell arrays are two-dimensional across the whole chart, ``(n_i, n_j)``, because the
eikonal solve and the slip spectrum are both index-grid calculations on a uniform
``(strike, dip)`` metric. Node positions carry the real geometry: the areas, the dips,
and the distances between charts.
"""

import dataclasses
import functools
import itertools
from collections.abc import Callable

import numpy as np
import scipy as sp

from rupture_generator.errors import RuptureGeneratorError

type NodeArray = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
"""Node positions, ``(n_i+1, n_j+n_k, 3)`` for a chart of ``n_k`` planes."""

type CellArray = np.ndarray[tuple[int, int], np.dtype[np.float64]]
"""One value per cell, ``(n_i, n_j)``."""

type CellMask = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
"""One flag per cell, ``(n_i, n_j)``."""

type CellSelection = tuple[np.ndarray, np.ndarray]
"""Some cells named as two index arrays, ``(i, j)``: what indexing a field wants."""


def _locate(position_km: float, arc_km: np.ndarray, *, axis: str) -> int:
    """The cell containing a position along an arc of node distances.

    A position on an interior boundary belongs to the cell after it; one on the far
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
    """One plane's nodes bilinearly resampled onto ``rows x columns`` cells.

    Exact for a parallelogram, which is every plane hung from a straight trace segment.
    """
    interpolate = sp.interpolate.RegularGridInterpolator(
        (np.arange(plane.shape[0]), np.arange(plane.shape[1])), plane
    )
    down = np.linspace(0.0, plane.shape[0] - 1, rows + 1)
    along = np.linspace(0.0, plane.shape[1] - 1, columns + 1)
    return interpolate(np.stack(np.meshgrid(down, along, indexing="ij"), axis=-1))


@dataclasses.dataclass(frozen=True, eq=False)
class Geometry:
    """One chart. See the module docstring for frames and the plane layout.

    Derived quantities are computed once and cached, which is safe because nothing
    here changes after construction.

    Attributes
    ----------
    nodes : NodeArray
        Node positions, ``(n_i+1, n_j+n_k, 3)``, kilometres in the projected CRS.
    occupied : CellMask
        Which cells are fault. All true for a chart built from planes; a resampled
        curved interface fills only part of its parameter rectangle.
    plane_cells : tuple of int
        How many cell columns each plane has, in trace order.
    """

    nodes: NodeArray
    occupied: CellMask
    plane_cells: tuple[int, ...]

    def __post_init__(self) -> None:
        """Check the shapes agree."""
        nodes = self.nodes
        if nodes.ndim != 3 or nodes.shape[-1] != 3 or nodes.shape[0] < 2:
            raise RuptureGeneratorError(
                f"nodes are shaped {nodes.shape}; a chart wants (n_i+1, n_j+n_k, 3) "
                "with at least one row of cells"
            )
        if not self.plane_cells or min(self.plane_cells) < 1:
            raise RuptureGeneratorError(
                f"every plane needs at least one cell column, got {self.plane_cells}"
            )
        columns = sum(self.plane_cells) + len(self.plane_cells)
        if nodes.shape[1] != columns:
            raise RuptureGeneratorError(
                f"{len(self.plane_cells)} planes of {self.plane_cells} cells want "
                f"{columns} node columns, and the nodes have {nodes.shape[1]}; every "
                "plane carries its own seam column"
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
        """``(n_i, n_j)``: cells down dip, cells along strike."""
        return (self.nodes.shape[0] - 1, sum(self.plane_cells))

    @property
    def planes(self) -> int:
        """How many planes the chart is made of."""
        return len(self.plane_cells)

    # ---------------------------------------------------------- one plane at a time

    def plane_nodes(self) -> list[NodeArray]:
        """Each plane's own node grid: a contiguous slice, never a copy."""
        starts = np.cumsum([0, *(cells + 1 for cells in self.plane_cells)])
        return [self.nodes[:, start:stop] for start, stop in itertools.pairwise(starts)]

    def _per_plane(self, cellwise: Callable[[NodeArray], np.ndarray]) -> np.ndarray:
        """A cell quantity computed plane by plane and joined along strike."""
        parts = [cellwise(plane) for plane in self.plane_nodes()]
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

    @functools.cached_property
    def centres(self) -> NodeArray:
        """Cell centres, ``(n_i, n_j, 3)``: the mean of the four corners."""
        return self._per_plane(
            lambda p: 0.25 * (p[:-1, :-1] + p[:-1, 1:] + p[1:, 1:] + p[1:, :-1])
        )

    @functools.cached_property
    def _diagonal_normals(self) -> NodeArray:
        """``AC x BD`` for every cell: along its normal, at twice its area."""
        return self._per_plane(
            lambda p: np.cross(p[1:, 1:] - p[:-1, :-1], p[1:, :-1] - p[:-1, 1:])
        )

    @functools.cached_property
    def areas_km2(self) -> CellArray:
        """Cell areas, ``(n_i, n_j)``: half the cross product of the diagonals.

        Exact for a planar quadrilateral; for a warped one it is the area projected
        onto the plane the two diagonals span.
        """
        return 0.5 * np.linalg.norm(self._diagonal_normals, axis=-1)

    @functools.cached_property
    def dip_deg(self) -> CellArray:
        """Cell dips from horizontal, ``(n_i, n_j)``; zero for a degenerate cell."""
        normal = self._diagonal_normals
        length = np.linalg.norm(normal, axis=-1)
        cosine = np.divide(
            np.abs(normal[..., 2]), length, out=np.ones_like(length), where=length > 0
        )
        return np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))

    @functools.cached_property
    def spacing_km(self) -> tuple[float, float]:
        """One ``(strike, dip)`` spacing for the chart: what the sampler and the
        eikonal get.

        The mean of every along-strike and every down-dip step, not one edge's: a plane
        hung from a bent trace is a trapezoid, and a resampled interface is not uniform
        anywhere.
        """
        strike, dip = [], []
        for plane in self.plane_nodes():
            strike.append(np.linalg.norm(np.diff(plane, axis=1), axis=-1).ravel())
            dip.append(np.linalg.norm(np.diff(plane, axis=0), axis=-1).ravel())
        return float(np.concatenate(strike).mean()), float(np.concatenate(dip).mean())

    @functools.cached_property
    def strike_arc_km(self) -> np.ndarray:
        """Distance along strike of each cell boundary on the top edge, ``(n_j+1,)``.

        With the dip arc, what makes a hypocentre two lengths rather than two indices.
        """
        steps = [
            np.linalg.norm(np.diff(plane[0], axis=0), axis=-1)
            for plane in self.plane_nodes()
        ]
        return np.concatenate([[0.0], np.cumsum(np.concatenate(steps))])

    @functools.cached_property
    def dip_arc_km(self) -> np.ndarray:
        """Distance down dip of each node row on the ``j = 0`` edge, ``(n_i+1,)``."""
        steps = np.linalg.norm(np.diff(self.nodes[:, 0], axis=0), axis=-1)
        return np.concatenate([[0.0], np.cumsum(steps)])

    def cell_at(self, strike_km: float, dip_km: float) -> tuple[int, int]:
        """The cell containing an in-fault position, as 0-based ``(i, j)``.

        Raises
        ------
        RuptureGeneratorError
            For a position off the fault, naming the axis and the fault's extent.
        """
        return (
            _locate(dip_km, self.dip_arc_km, axis="dip"),
            _locate(strike_km, self.strike_arc_km, axis="strike"),
        )

    def seam_divergence_km(self) -> np.ndarray:
        """How far apart adjacent planes' shared edges run, one value per seam.

        Two planes hung from a bent trace meet along the trace and part below it, by
        1.285 km at the deepest row of the Hope example. The largest separation down
        each seam, ``(n_k-1,)``, empty for one plane.
        """
        return np.array(
            [
                float(np.linalg.norm(near[:, -1] - far[:, 0], axis=-1).max())
                for near, far in itertools.pairwise(self.plane_nodes())
            ]
        )

    def subdivide(self, spacing_km: float) -> Geometry:
        """This chart recut into cells of about ``spacing_km`` on a side.

        Each plane is resampled bilinearly over its own nodes, its cell count rounded
        and floored at one. Every plane takes the rows the ``j = 0`` edge asks for,
        since the planes of one chart share a dip extent.

        Raises
        ------
        RuptureGeneratorError
            For a spacing that is not a positive length, or a chart that is not fault
            everywhere: a partial outline has nothing to resample it by.
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
            planes.append(_resample(plane, rows, columns[-1]))
        return Geometry(
            nodes=np.concatenate(planes, axis=1),
            occupied=np.ones((rows, sum(columns)), dtype=bool),
            plane_cells=tuple(columns),
        )

    # -------------------------------------------------------- between two charts

    @functools.cached_property
    def _edge(self) -> tuple[CellSelection, sp.spatial.cKDTree]:
        """The fault's edge cells, and a tree over their centres.

        An edge cell is occupied with a neighbour off the fault or off the grid: where
        a front can leave this chart or land on it. Their centres are gathered from the
        nodes rather than read off :attr:`centres`, since the edge of a production
        interface is a few thousand cells against twenty million.
        """
        cells = np.nonzero(
            self.occupied & ~sp.ndimage.binary_erosion(self.occupied, border_value=0)
        )
        i, j = cells
        left = j + np.repeat(np.arange(self.planes), self.plane_cells)[j]
        nodes = self.nodes
        centres = 0.25 * (
            nodes[i, left]
            + nodes[i, left + 1]
            + nodes[i + 1, left + 1]
            + nodes[i + 1, left]
        )
        return cells, sp.spatial.cKDTree(centres)

    def nearest_cells_to(
        self, other: Geometry
    ) -> tuple[CellSelection, CellSelection, np.ndarray]:
        """For each of this chart's edge cells, the closest edge cell of ``other``.

        Returns three parallel arrays: this chart's edge cells, the cell on ``other``
        each one is closest to, and the straight-line distance between them in
        kilometres.

        Raises
        ------
        RuptureGeneratorError
            If either chart is entirely unoccupied.
        """
        (here, mine), (there, theirs) = self._edge, other._edge
        if here[0].size == 0 or there[0].size == 0:
            raise RuptureGeneratorError(
                "a distance between charts needs fault on both sides, and one of these "
                "charts is entirely unoccupied"
            )
        distance_km, nearest = theirs.query(mine.data)
        return here, (there[0][nearest], there[1][nearest]), distance_km
