import dataclasses
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

type NodeArray = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
"""Node positions, ``(n_i+1, n_j+1, 3)``: east, north, depth, kilometres."""

type CellArray = np.ndarray[tuple[int, int], np.dtype[np.float64]]
"""One value per cell, ``(n_i, n_j)``."""

type CellMask = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
"""One flag per cell, ``(n_i, n_j)``."""

_DOWN = np.array([0.0, 0.0, 1.0])


class GeometryError(ValueError):
    """A chart, field or realisation that does not fit together."""


def _bearing_deg(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """Compass bearing of a horizontal direction, degrees clockwise from grid north."""
    return np.degrees(np.arctan2(east, north)) % 360.0


def _locate(position_km: float, arc_km: np.ndarray, *, axis: str) -> int:
    """The cell containing a position along an arc of node distances.

    A position on an interior boundary belongs to the cell after it; one on the far
    edge belongs to the last cell.
    """
    extent = float(arc_km[-1])
    if not 0.0 <= position_km <= extent:
        raise GeometryError(
            f"{position_km} km is off the fault along {axis}, which runs 0 to "
            f"{extent:.3f} km"
        )
    index = int(np.searchsorted(arc_km, position_km, side="right")) - 1
    return min(index, len(arc_km) - 2)


@dataclasses.dataclass(frozen=True, eq=False)
class Geometry:
    """One chart with what has been drawn on it. See the module docstring for frames.

    Attributes
    ----------
    nodes : NodeArray
        Node positions, ``(n_i+1, n_j+1, 3)`` -- east, north, depth in kilometres,
        offsets from ``origin_km``.
    origin_km : tuple of float
        The surface's ``(east, north)`` origin in the realisation's CRS, kilometres.
    occupied : CellMask
        Which cells are fault. All true for a chart built from planes; a resampled
        curved interface fills only part of its parameter rectangle. Fields are defined
        everywhere and meaningful where this is true.
    fields : Mapping of str to CellArray
        Cell fields, each ``(n_i, n_j)``, names unit-suffixed.
    attrs : Mapping of str to float
        Per-chart scalars: ``hypocentre_strike_km``, ``hypocentre_dip_km``, diagnostics.
    """

    nodes: NodeArray
    origin_km: tuple[float, float]
    occupied: CellMask
    fields: Mapping[str, CellArray] = dataclasses.field(default_factory=dict)
    attrs: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the shapes agree, and make the mappings read-only."""
        nodes = np.asarray(self.nodes, dtype=np.float64)
        if nodes.ndim != 3 or nodes.shape[-1] != 3 or min(nodes.shape[:2]) < 2:
            raise GeometryError(
                f"nodes are shaped {nodes.shape}; a chart wants (n_i+1, n_j+1, 3) with "
                "at least one cell"
            )
        if not np.all(np.isfinite(nodes)):
            raise GeometryError("nodes contain a NaN or infinity")
        object.__setattr__(self, "nodes", nodes)

        cells = self.cells
        occupied = np.asarray(self.occupied, dtype=bool)
        if occupied.shape != cells:
            raise GeometryError(
                f"the occupied mask is shaped {occupied.shape} and the chart has "
                f"{cells} cells"
            )
        object.__setattr__(self, "occupied", occupied)

        fields = {
            name: self._as_cell_array(name, array)
            for name, array in self.fields.items()
        }
        object.__setattr__(self, "fields", MappingProxyType(fields))
        object.__setattr__(
            self,
            "attrs",
            MappingProxyType({k: float(v) for k, v in self.attrs.items()}),
        )

    def _as_cell_array(self, name: str, array: object) -> CellArray:
        cells = self.cells
        out = np.asarray(array, dtype=np.float64)
        if out.shape != cells:
            raise GeometryError(
                f"field {name!r} is shaped {out.shape} and the chart has {cells} cells"
            )
        return out

    def __repr__(self) -> str:
        """The shape and what is drawn on it, not the arrays."""
        n_i, n_j = self.cells
        names = ", ".join(sorted(self.fields)) or "none"
        return f"{type(self).__name__}({n_i}x{n_j} cells, fields: {names})"

    # ---------------------------------------------------------------- the fields

    @property
    def cells(self) -> tuple[int, int]:
        """``(n_i, n_j)``: cells down dip, cells along strike."""
        return (self.nodes.shape[0] - 1, self.nodes.shape[1] - 1)

    def __getitem__(self, name: str) -> CellArray:
        """A field by name, shaped :attr:`cells`."""
        try:
            return self.fields[name]
        except KeyError:
            have = ", ".join(sorted(self.fields)) or "none"
            raise GeometryError(
                f"this chart has no field {name!r}; it has {have}"
            ) from None

    def __contains__(self, name: object) -> bool:
        """Whether a field of this name is attached."""
        return name in self.fields

    def with_fields(self, **arrays: CellArray) -> Geometry:
        """This chart with fields added or replaced. Shapes are checked."""
        return dataclasses.replace(self, fields={**self.fields, **arrays})

    def without(self, *names: str) -> Geometry:
        """This chart with the named fields dropped; missing names are not an error."""
        kept = {name: f for name, f in self.fields.items() if name not in names}
        return dataclasses.replace(self, fields=kept)

    def with_attrs(self, **values: float) -> Geometry:
        """This chart with scalars added or replaced."""
        return dataclasses.replace(self, attrs={**self.attrs, **values})

    def on_fault(self, name: str) -> np.ndarray:
        """The occupied cells' values of a field, flat, in ``(i, j)`` row-major order.

        The one place the mask meets a field, so the moment fold, the writer and the
        diagnostics do not each mask for themselves.
        """
        return self[name][self.occupied]

    def _corners(self) -> tuple[NodeArray, NodeArray, NodeArray, NodeArray]:
        """Cell corners, each ``(n_i, n_j, 3)``, anticlockwise from the shallow near
        end -- the order the area's diagonal split and the strike sign rely on."""
        nodes = self.nodes
        return nodes[:-1, :-1], nodes[:-1, 1:], nodes[1:, 1:], nodes[1:, :-1]

    def centres(self) -> NodeArray:
        """Cell centres, ``(n_i, n_j, 3)``: the mean of the four corners."""
        c0, c1, c2, c3 = self._corners()
        return 0.25 * (c0 + c1 + c2 + c3)

    def areas_km2(self) -> CellArray:
        """Cell areas, ``(n_i, n_j)``, as two triangles split on the (0, 2) diagonal.

        The split costs nothing and copes with non-coplanar corners.
        """
        c0, c1, c2, c3 = self._corners()

        def triangle(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
            return 0.5 * np.linalg.norm(np.cross(q - p, r - p), axis=-1)

        return triangle(c0, c1, c2) + triangle(c0, c2, c3)

    def _directions(self) -> tuple[NodeArray, NodeArray]:
        """Per-cell along-strike and down-dip vectors: edge sums, since only the
        direction is read."""
        c0, c1, c2, c3 = self._corners()
        return (c1 - c0) + (c2 - c3), (c3 - c0) + (c2 - c1)

    def normals(self) -> NodeArray:
        """Unit cell normals, ``(n_i, n_j, 3)``, oriented by the ``(strike, dip)``
        right-hand rule. A degenerate cell reports the zero vector, never NaN."""
        along_strike, down_dip = self._directions()
        normal = np.cross(along_strike, down_dip)
        magnitude = np.linalg.norm(normal, axis=-1, keepdims=True)
        return np.divide(
            normal, magnitude, out=np.zeros_like(normal), where=magnitude > 0
        )

    def strike_dip_deg(self) -> tuple[CellArray, CellArray]:
        along_strike, _ = self._directions()
        unit = self.normals()
        degenerate = np.all(unit == 0.0, axis=-1)

        dip_deg = np.degrees(np.arccos(np.clip(np.abs(unit[..., 2]), 0.0, 1.0)))
        dip_deg = np.where(degenerate, 0.0, dip_deg)

        # cross(DOWN, n) is horizontal and in the plane: the strike direction up to sign.
        horizontal = np.cross(np.broadcast_to(_DOWN, unit.shape), unit)
        flat = np.linalg.norm(horizontal, axis=-1) == 0.0
        sign = np.where(np.sum(horizontal * along_strike, axis=-1) < 0.0, -1.0, 1.0)
        oriented = horizontal * sign[..., None]

        fallback = _bearing_deg(along_strike[..., 0], along_strike[..., 1])
        strike_deg = np.where(
            degenerate | flat,
            fallback,
            _bearing_deg(oriented[..., 0], oriented[..., 1]),
        )
        return strike_deg, dip_deg

    def strike_arc_km(self) -> np.ndarray:
        """Distance along strike of each node column on the top edge, ``(n_j+1,)``.

        With the dip arc, what makes a hypocentre two lengths rather than two indices.
        """
        steps = np.linalg.norm(np.diff(self.nodes[0], axis=0), axis=-1)
        return np.concatenate([[0.0], np.cumsum(steps)])

    def dip_arc_km(self) -> np.ndarray:
        """Distance down dip of each node row on the ``j = 0`` edge, ``(n_i+1,)``."""
        steps = np.linalg.norm(np.diff(self.nodes[:, 0], axis=0), axis=-1)
        return np.concatenate([[0.0], np.cumsum(steps)])

    def spacing_km(self) -> tuple[float, float]:
        """One ``(strike, dip)`` spacing for the chart: what the sampler and the
        eikonal get.

        The mean of every along-strike and every down-dip step, not one edge's: a fused
        bend is a trapezoid, and a resampled interface is not uniform anywhere.
        """
        nodes = self.nodes
        strike_steps = np.linalg.norm(np.diff(nodes, axis=1), axis=-1)
        dip_steps = np.linalg.norm(np.diff(nodes, axis=0), axis=-1)
        return float(strike_steps.mean()), float(dip_steps.mean())

    def cell_at(self, strike_km: float, dip_km: float) -> tuple[int, int]:
        """The cell containing an in-fault position, as 0-based ``(i, j)``.

        Not the SRF's ``shyp``, which is measured from the along-strike centre, and not
        a node index.

        Raises
        ------
        GeometryError
            For a position off the fault, naming the axis and the fault's extent.
        """
        return (
            _locate(dip_km, self.dip_arc_km(), axis="dip"),
            _locate(strike_km, self.strike_arc_km(), axis="strike"),
        )
