"""The one mesh type: a fault surface as nodes, and everything else derived.

A segment is a chart ``X: (i, j) -> R^3``: node positions in a projected Cartesian CRS,
``i`` down-dip, ``j`` along-strike, depth positive down, kilometres. Arithmetic is in
the projection, not on the ellipsoid, where a 60 km subduction interface came out with
cell areas 1.4e-2 low; strike here is therefore **grid north**, and dip and rake cross
unchanged.

Positions are offsets from a per-surface origin, taken first: an NZTM northing reaches
~5,180 km against a ~1 km subfault, so an absolute vertex is rounded at 1.2e-12 relative
against 3e-15 for an offset. The origin is added back only at the projection seam.

Planes that hang the same way -- equal dip, dip direction and bottom depth -- share
their seam column exactly and fuse into one chart; planes that do not are two segments,
not an error.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pyproj
import xarray as xr

from rupture_generator.errors import GeometryError
from rupture_generator.units import M_PER_KM

if TYPE_CHECKING:
    from rupture_generator.config.geometry import (
        Discretisation,
        FaultConfig,
        PointConfig,
    )

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]
BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]

WGS84 = pyproj.CRS("EPSG:4326")
"""What an SRF's coordinates are, and what everything downstream of one expects."""

SHARPEST_BEND_DEG = 120.0
"""The steepest trace bend a fused surface accepts.

A bend column is stretched by ``1 / cos(deflection / 2)``; at 120 degrees that is 2.
"""

SEAM_TOLERANCE_KM = 1.0e-6
"""How far apart two planes' shared node columns may be and still be one surface.

A millimetre: six orders above the round-off floor (~1e-13 km at fault scale) and six
below anything real -- the kaikoura example, at 70 and 55 degrees, separates by 3.5 km
at its deepest row.
"""

SPACING_SPREAD = 0.10
"""The widest relative spread of per-plane spacings a fused surface accepts.

What rounding one requested size can produce: a plane of length L cut at size s gets
cells within s/2L of s -- under 2% on a 27 km plane at 1 km, reaching 10% only at about
five cells.
"""

UNIFORM_SPACING_TOLERANCE = 1.0e-9
"""Relative spread above which an edge's steps are not one spacing.

Six orders from f64 round-off (~1e-15 relative, since bilinear refinement of a
parallelogram gives exactly equal steps) and six from a factor of two.
"""

PLANARITY_TOLERANCE_KM = 1.0e-6
"""The worst out-of-plane node distance a planar chart may have.

A chart this module builds is planar to round-off (~1e-13 km at fault scale); any real
non-planarity is kilometres.
"""

_MAX_BEND_SPREAD = 1.0 / np.cos(np.radians(SHARPEST_BEND_DEG / 2.0)) - 1.0
"""How far a block's line spacings may spread around their mean.

What the sharpest accepted bend contributes through the stretch alone, derived from
:data:`SHARPEST_BEND_DEG` so the two cannot drift. A bend also swings the block's bottom
edge by ``depth_span / tan(dip)`` -- 46 km on a 4 km-deep fault dipping 5 degrees
against a 27 km plane -- so the check is on measured spread, not on the deflection.
"""

_DOWN = np.array([0.0, 0.0, 1.0])


# Primitives. Bearings are degrees clockwise from grid north, hence atan2(east, north)
# rather than the mathematical atan2(y, x).


def _normalise_bearing(degrees: float) -> float:
    folded = degrees % 360.0
    return folded + 360.0 if folded < 0.0 else folded


def _bearing_deg(from_point: FloatArray, to_point: FloatArray) -> float:
    """The bearing between two horizontal points, in atan2's own (-180, 180]."""
    return float(
        np.degrees(np.arctan2(to_point[0] - from_point[0], to_point[1] - from_point[1]))
    )


def _along(point: FloatArray, bearing_deg: float, distance_km: float) -> FloatArray:
    """A horizontal step along a bearing."""
    radians = np.radians(bearing_deg)
    return np.array(
        [
            point[0] + distance_km * np.sin(radians),
            point[1] + distance_km * np.cos(radians),
        ]
    )


def _bearing_of(east: FloatArray, north: FloatArray) -> FloatArray:
    """Bearings of direction vectors, normalised to [0, 360)."""
    return np.degrees(np.arctan2(east, north)) % 360.0


def to_projected(
    crs: pyproj.CRS, longitude_deg: float, latitude_deg: float
) -> tuple[float, float]:
    """A longitude and latitude as an easting and northing, in **kilometres**."""
    easting_m, northing_m = pyproj.Transformer.from_crs(
        WGS84, crs, always_xy=True
    ).transform(longitude_deg, latitude_deg)
    return easting_m / M_PER_KM, northing_m / M_PER_KM


def grid_convergence_deg(
    crs: pyproj.CRS, longitude_deg: FloatArray, latitude_deg: FloatArray
) -> FloatArray:
    """The angle from true north to grid north, in degrees, at each point.

    Add it to a grid azimuth to get a true one. In NZTM2000 it runs from -3.4 degrees at
    East Cape to +5.0 in Fiordland, five times the one-degree rake bound, and it varies
    across a fault -- hence per subfault.
    """
    factors = pyproj.Proj(crs).get_factors(longitude_deg, latitude_deg)
    return np.asarray(factors.meridian_convergence, dtype=np.float64)


# The mesh type

NODE_VARIABLES = ("east_km", "north_km", "depth_km")
"""The chart's own geometry, on ``(i_node, j_node)``. What a mesh file stores."""

CELL_DIMS = ("i", "j")
"""The dims a stage's field lives on: ``i`` down dip, ``j`` along strike."""

RESERVED_FIELDS = frozenset(
    {*NODE_VARIABLES, "plane", "occupied", "slip_rate", "slip_rate_offset"}
)
"""Names a stage may not attach a field under."""

RESERVED_ATTRS = frozenset({"surface", "origin_east_km", "origin_north_km"})


@dataclasses.dataclass(frozen=True, eq=False)
class RuptureMesh:
    """One rupture geometry expressed as a mesh."""

    _dataset: xr.Dataset

    def __repr__(self) -> str:
        """The chart's name, shape and fields -- not the dataset behind it."""
        cells_i, cells_j = self.cell_counts
        fields = ", ".join(sorted(self.fields())) or "none"
        return (
            f"{type(self).__name__}({self.surface!r}, "
            f"{cells_i * cells_j} cells, fields: {fields})"
        )

    def _with(self, dataset: xr.Dataset) -> RuptureMesh:
        return type(self)(dataset)

    @classmethod
    def from_nodes(
        cls,
        east_km: FloatArray,
        north_km: FloatArray,
        depth_km: FloatArray,
        *,
        origin_east_km: float,
        origin_north_km: float,
        surface: str,
        plane_of_column: FloatArray | None = None,
        occupied: BoolArray | None = None,
        parameter_spacing_km: tuple[float, float] | None = None,
    ) -> RuptureMesh:
        """Build a chart from node position arrays of shape ``(n_i+1, n_j+1)``.

        Parameters
        ----------
        east_km, north_km, depth_km : FloatArray
            Node offsets from the origin, ``i`` down-dip and ``j`` along-strike, km.
        origin_east_km, origin_north_km : float
            The surface origin, in the CRS, km. Added back only at the projection seam.
        surface : str
            The surface's name, which becomes the group name in files.
        plane_of_column : FloatArray, optional
            Which config plane each *cell* column came from, length ``n_j``. Default
            all zeros, a single-plane chart.
        occupied : BoolArray, optional
            Which cells are really fault, ``(n_i, n_j)``. See :attr:`occupied`.
        parameter_spacing_km : tuple of float, optional
            ``(strike, dip)`` cell size in the chart's own parameters, km. See
            :meth:`parameter_spacing_km`.

        Raises
        ------
        GeometryError
            If the arrays disagree in shape, are smaller than 2x2 nodes, carry anything
            non-finite, or the mask is the wrong shape or empty.
        """
        east_km = np.asarray(east_km, dtype=np.float64)
        north_km = np.asarray(north_km, dtype=np.float64)
        depth_km = np.asarray(depth_km, dtype=np.float64)
        if not (east_km.shape == north_km.shape == depth_km.shape):
            raise GeometryError(
                f"the node arrays disagree in shape: east {east_km.shape}, "
                f"north {north_km.shape}, depth {depth_km.shape}"
            )
        if east_km.ndim != 2 or min(east_km.shape) < 2:
            raise GeometryError(
                f"a chart needs at least 2 nodes on each axis, got {east_km.shape}"
            )
        for name, values in (
            ("east_km", east_km),
            ("north_km", north_km),
            ("depth_km", depth_km),
        ):
            if not np.isfinite(values).all():
                raise GeometryError(f"{name} carries a non-finite node position")

        cells_j = east_km.shape[1] - 1
        if plane_of_column is None:
            plane_of_column = np.zeros(cells_j, dtype=np.int64)
        plane_of_column = np.asarray(plane_of_column, dtype=np.int64)
        if plane_of_column.shape != (cells_j,):
            raise GeometryError(
                f"plane_of_column has {plane_of_column.shape[0]} entries for "
                f"{cells_j} cell columns"
            )

        cells = (east_km.shape[0] - 1, cells_j)
        coords: dict[str, Any] = {"plane": ("j", plane_of_column)}
        if occupied is not None:
            mask = np.asarray(occupied, dtype=bool)
            if mask.shape != cells:
                raise GeometryError(
                    f"the occupancy mask is shaped {mask.shape} for a chart of "
                    f"{cells} cells"
                )
            if not mask.any():
                raise GeometryError(
                    f"{surface!r} has no occupied cells, so the surface does not "
                    "reach its own parameter rectangle anywhere"
                )
            coords["occupied"] = (CELL_DIMS, mask)

        extra: dict[str, Any] = {}
        if parameter_spacing_km is not None:
            strike_km, dip_km = (float(value) for value in parameter_spacing_km)
            if not (strike_km > 0.0 and dip_km > 0.0):
                raise GeometryError(
                    f"the parameter spacing must be positive, got "
                    f"({strike_km}, {dip_km})"
                )
            extra["parameter_spacing_km"] = np.array([strike_km, dip_km])

        dataset = xr.Dataset(
            data_vars={
                "east_km": (
                    ("i_node", "j_node"),
                    east_km,
                    {
                        "units": "kilometres",
                        "long_name": "Easting offset from the mesh origin",
                    },
                ),
                "north_km": (
                    ("i_node", "j_node"),
                    north_km,
                    {
                        "units": "kilometres",
                        "long_name": "Northing offset from the mesh origin",
                    },
                ),
                "depth_km": (
                    ("i_node", "j_node"),
                    depth_km,
                    {
                        "units": "kilometres",
                        "long_name": "Depth below the surface, positive downwards",
                    },
                ),
            },
            coords=coords,
            attrs={
                "surface": surface,
                "origin_east_km": float(origin_east_km),
                "origin_north_km": float(origin_north_km),
                **extra,
            },
        )
        return cls(dataset)

    # ------------------------------------------------------------------ shape

    @property
    def surface(self) -> str:
        """The surface this chart belongs to."""
        return str(self._dataset.attrs["surface"])

    @property
    def origin_km(self) -> tuple[float, float]:
        """The surface origin (easting, northing), in the CRS, kilometres."""
        return (
            float(self._dataset.attrs["origin_east_km"]),
            float(self._dataset.attrs["origin_north_km"]),
        )

    @property
    def cell_counts(self) -> tuple[int, int]:
        """Cells ``(n_i, n_j)`` one fewer than nodes on each axis."""
        return (
            self._dataset.sizes["i_node"] - 1,
            self._dataset.sizes["j_node"] - 1,
        )

    def nodes(self) -> FloatArray:
        """Node positions, ``(n_i+1, n_j+1, 3)``, components (east, north, depth)."""
        return np.stack(
            [
                self._dataset["east_km"].to_numpy(),
                self._dataset["north_km"].to_numpy(),
                self._dataset["depth_km"].to_numpy(),
            ],
            axis=-1,
        )

    def planes(self) -> FloatArray:
        """Which config plane each cell column came from, length ``n_j``."""
        return self._dataset["plane"].to_numpy()

    def parameter_spacing_km(self) -> tuple[float, float] | None:
        """``(strike, dip)`` cell size in the chart's own parameters, or ``None``.

        Set for a resampled curved surface, which is uniform *in its parameters* while
        its lifted node spacing varies with the curvature -- measuring a spacing from
        those nodes would read the surface's own stretch as a discretisation. ``None``
        for a chart built from a config, whose spacing is measured from nodes.
        """
        stored = self._dataset.attrs.get("parameter_spacing_km")
        if stored is None:
            return None
        strike_km, dip_km = (float(value) for value in np.asarray(stored).reshape(2))
        return (strike_km, dip_km)

    def occupied(self) -> BoolArray:
        """Which cells are really fault, shaped :attr:`cell_counts`.

        All true for a fault built from a config, which fills its parameter rectangle
        exactly. A resampled surface does not: the CFM Hikurangi interface fills about
        two thirds of the rectangle its outline spans. Unoccupied cells carry positions,
        because the grid needs corners, but no fault -- the wavefront walls them off,
        the moment does not count them, and the SRF does not write them.
        """
        if "occupied" not in self._dataset.coords:
            return np.ones(self.cell_counts, dtype=bool)
        return np.asarray(self._dataset["occupied"].to_numpy(), dtype=bool)

    def blocks(self) -> list[tuple[int, int, int]]:
        """Contiguous constant-plane runs, as ``(plane, start, stop)`` cell columns.

        The unit of planarity and spacing: a bent fault is only piecewise planar.
        """
        plane = self.planes()
        boundaries = np.flatnonzero(np.diff(plane)) + 1
        starts = np.concatenate([[0], boundaries])
        stops = np.concatenate([boundaries, [len(plane)]])
        return [
            (int(plane[start]), int(start), int(stop))
            for start, stop in zip(starts, stops, strict=True)
        ]

    # -------------------------------------------------------------- the fields

    def fields(self) -> frozenset[str]:
        """Attached fields' names: the variables whose dims are the cell dims.

        Geometry is not in here; geometry is computed.
        """
        return frozenset(
            str(name)
            for name, variable in self._dataset.data_vars.items()
            if variable.dims == CELL_DIMS
        )

    def __contains__(self, name: object) -> bool:
        """Whether a field of that name has been attached."""
        return isinstance(name, str) and name in self.fields()

    def __getitem__(self, name: str) -> FloatArray:
        """A field a stage attached, shaped :attr:`cell_counts`, read-only.

        Raises
        ------
        KeyError
            Naming the field and listing what this chart does carry.
        """
        if name not in self.fields():
            attached = ", ".join(sorted(self.fields())) or "nothing"
            raise KeyError(
                f"{self.surface!r} carries no field called {name!r}; it carries "
                f"{attached}"
            )
        values: FloatArray = self._dataset[name].to_numpy()
        view = values.view()
        view.flags.writeable = False
        return view

    # An xarray-like API, without handing out the underlying dataset.

    def with_fields(self, **arrays: FloatArray) -> RuptureMesh:
        """This chart with more cell fields on it. Functional, never in place.

        Raises
        ------
        GeometryError
            For an array that is not the chart's shape, one carrying a non-finite value,
            or a name in :data:`RESERVED_FIELDS`. The shape check is this module's own:
            xarray objects only when dimension *sizes* disagree, so a transposed field
            on a square patch would go in unremarked.
        """
        cell_counts = self.cell_counts
        prepared = {}
        for name, values in arrays.items():
            if name in RESERVED_FIELDS:
                raise GeometryError(
                    f"{name!r} is the chart's own, not a field to attach; "
                    f"reserved names are {', '.join(sorted(RESERVED_FIELDS))}"
                )
            array = np.asarray(values, dtype=np.float64)
            if array.shape != cell_counts:
                raise GeometryError(
                    f"{name} is shaped {array.shape}, and this chart has "
                    f"{cell_counts} cells (i down dip, j along strike)"
                )
            if not np.isfinite(array).all():
                raise GeometryError(f"{name} carries a non-finite value")
            prepared[name] = (CELL_DIMS, array)

        return self._with(self._dataset.assign(prepared))

    def without(self, *names: str) -> RuptureMesh:
        """This chart with those fields dropped. A name that is not there is fine."""
        return self._with(self._dataset.drop_vars(names, errors="ignore"))

    @property
    def attrs(self) -> Mapping[str, Any]:
        """What this chart records about itself, read-only.

        What a stage learns that is not one value per subfault: the truncation
        diagnostic, and where the rupture nucleated.
        """
        return types.MappingProxyType(dict(self._dataset.attrs))

    def with_attrs(self, **values: Any) -> RuptureMesh:
        """This chart with the attributes given in **values.

        Scalars by convention: these go straight into a file's group attributes.

        Raises
        ------
        GeometryError
            For a name in :data:`RESERVED_ATTRS`.
        """
        reserved = RESERVED_ATTRS & set(values)
        if reserved:
            raise GeometryError(
                f"{', '.join(sorted(reserved))} says what this chart is, and is not "
                "a stage's to rewrite"
            )
        return self._with(self._dataset.assign_attrs(**values))

    def with_pulses(self, offsets: np.ndarray, samples: FloatArray) -> RuptureMesh:
        """This chart with its slip-rate pulses attached, as CSR.

        ``offsets`` is where each subfault's pulse starts, length ``n_i * n_j + 1``,
        strike-fastest; ``samples`` is every pulse concatenated.

        Raises
        ------
        GeometryError
            For an indptr that is not one: wrong length, decreasing, or not ending at
            ``samples.size``.
        """
        cells_i, cells_j = self.cell_counts
        offsets = np.asarray(offsets, dtype=np.int64)
        samples = np.asarray(samples, dtype=np.float64)

        if offsets.shape != (cells_i * cells_j + 1,):
            raise GeometryError(
                f"the pulse offsets are shaped {offsets.shape}, and this chart has "
                f"{cells_i * cells_j} subfaults, so it wants "
                f"{cells_i * cells_j + 1} (one per subfault, plus the end)"
            )
        if np.any(np.diff(offsets) < 0):
            raise GeometryError("the pulse offsets decrease, so some subfault has none")
        if offsets[0] != 0 or offsets[-1] != samples.size:
            raise GeometryError(
                f"the pulse offsets run {offsets[0]} to {offsets[-1]}, and there are "
                f"{samples.size} samples"
            )

        return self._with(
            self._dataset.assign(
                {
                    "slip_rate": ("sample", samples),
                    "slip_rate_offset": ("cell_edge", offsets),
                }
            )
        )

    @property
    def pulses(self) -> tuple[np.ndarray, FloatArray] | None:
        """The slip-rate pulses as ``(offsets, samples)``, or ``None`` if unset."""
        if "slip_rate" not in self._dataset:
            return None
        return (
            self._dataset["slip_rate_offset"].to_numpy(),
            self._dataset["slip_rate"].to_numpy(),
        )

    def node_dataset(self) -> xr.Dataset:
        """The node positions with their units, and nothing else."""
        return self._dataset[list(NODE_VARIABLES)].drop_vars("plane", errors="ignore")

    # ------------------------------------------------------- derived quantities

    def _corners(self) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Cell corners, each ``(n_i, n_j, 3)``, anticlockwise from the shallow near
        end: the order the area's diagonal split and the strike sign rely on."""
        nodes = self.nodes()
        return (
            nodes[:-1, :-1],
            nodes[:-1, 1:],
            nodes[1:, 1:],
            nodes[1:, :-1],
        )

    def centres(self) -> FloatArray:
        """Cell centres, shape ``(n_i, n_j, 3)`` -- the mean of the four corners."""
        c0, c1, c2, c3 = self._corners()
        return 0.25 * (c0 + c1 + c2 + c3)

    def areas_km2(self) -> FloatArray:
        """Cell areas, ``(n_i, n_j)``, as two triangles split on the (0, 2) diagonal.

        The split costs nothing and copes with non-coplanar corners.
        """
        c0, c1, c2, c3 = self._corners()

        def triangle(p: FloatArray, q: FloatArray, r: FloatArray) -> FloatArray:
            return 0.5 * np.linalg.norm(np.cross(q - p, r - p), axis=-1)

        return triangle(c0, c1, c2) + triangle(c0, c2, c3)

    def _direction_vectors(self) -> tuple[FloatArray, FloatArray]:
        """Per-cell along-strike and down-dip vectors -- edge sums, since only the
        direction is read."""
        c0, c1, c2, c3 = self._corners()
        return (c1 - c0) + (c2 - c3), (c3 - c0) + (c2 - c1)

    def strike_dip_deg(self) -> tuple[FloatArray, FloatArray]:
        """Per-cell strike (grid north, [0, 360)) and dip ([0, 90]), from the normal.

        From the normal rather than the edges, which keeps them right on a surface that
        is not planar. Dip is independent of the normal's sign; strike's sign comes from
        the cell's along-strike edges, tying it to the trace direction. A degenerate cell
        reports dip 0 and the strike of its along-strike edge, never NaN.
        """
        along_strike, down_dip = self._direction_vectors()
        normal = np.cross(along_strike, down_dip)
        magnitude = np.linalg.norm(normal, axis=-1)
        degenerate = magnitude == 0.0
        safe = np.where(degenerate, 1.0, magnitude)

        unit = normal / safe[..., None]
        dip_deg = np.degrees(np.arccos(np.clip(np.abs(unit[..., 2]), 0.0, 1.0)))
        dip_deg = np.where(degenerate, 0.0, dip_deg)

        # cross(DOWN, n) is perpendicular to down (horizontal) and to the normal (in
        # the plane): the strike direction, up to sign.
        horizontal = np.cross(np.broadcast_to(_DOWN, unit.shape), unit)
        horizontal_norm = np.linalg.norm(horizontal, axis=-1)
        flat = horizontal_norm == 0.0
        sign = np.where(np.sum(horizontal * along_strike, axis=-1) < 0.0, -1.0, 1.0)
        oriented = horizontal * sign[..., None]

        fallback = _bearing_of(along_strike[..., 0], along_strike[..., 1])
        strike_deg = np.where(
            degenerate | flat,
            fallback,
            _bearing_of(oriented[..., 0], oriented[..., 1]),
        )
        return strike_deg, dip_deg

    def strike_arc_km(self) -> FloatArray:
        """Distance along strike of each node column, on the top edge, ``(n_j+1,)``.

        With the dip arc, what makes a hypocentre two lengths rather than two indices.
        """
        top = self.nodes()[0]
        steps = np.linalg.norm(np.diff(top, axis=0), axis=-1)
        return np.concatenate([[0.0], np.cumsum(steps)])

    def dip_arc_km(self) -> FloatArray:
        """Distance down dip of each node row on the ``j = 0`` edge, ``(n_i+1,)``."""
        near = self.nodes()[:, 0]
        steps = np.linalg.norm(np.diff(near, axis=0), axis=-1)
        return np.concatenate([[0.0], np.cumsum(steps)])

    def line_steps(self) -> tuple[FloatArray, FloatArray]:
        """Every along-strike and every down-dip step, as ``(strike, dip)`` arrays.

        Shapes ``(n_i+1, n_j)`` and ``(n_i, n_j+1)``. Every line, not one edge: a uniform
        edge says nothing about the interior of a non-parallelogram.
        """
        nodes = self.nodes()
        return (
            np.linalg.norm(np.diff(nodes, axis=1), axis=-1),
            np.linalg.norm(np.diff(nodes, axis=0), axis=-1),
        )

    def _block_cut_sizes(self) -> tuple[list[float], list[float], list[int]]:
        """Each block's realised subfault size, unstretched, and its cell count.

        Unstretched, so comparing blocks asks about discretisation rather than bends: on
        a trapezoidal fused bend the two unstretched lines are the trace and the
        shortest column.
        """
        nodes = self.nodes()
        cells_i = nodes.shape[0] - 1
        trace_steps = np.linalg.norm(np.diff(nodes[0], axis=0), axis=-1)
        column_lengths = np.linalg.norm(nodes[-1] - nodes[0], axis=-1)

        strike_sizes: list[float] = []
        dip_sizes: list[float] = []
        weights: list[int] = []
        for _plane, start, stop in self.blocks():
            strike_sizes.append(float(trace_steps[start:stop].mean()))
            dip_sizes.append(float(column_lengths[start : stop + 1].min()) / cells_i)
            weights.append(stop - start)
        return strike_sizes, dip_sizes, weights

    def spacing_km(self) -> tuple[float, float]:
        """One ``(strike, dip)`` spacing for the chart -- what the sampler gets.

        A chart with a stated :meth:`parameter_spacing_km` returns it. Otherwise a
        block's spacing is the mean of its steps and the chart's is the
        cell-count-weighted mean of its blocks', a mean rather than one line's step
        because a fused bend is a trapezoid: rows differ by up to 2.4% on the shipped
        ``hope`` example.

        Raises
        ------
        GeometryError
            If the blocks were cut at resolutions too far apart to average, judged on
            their unstretched sizes so a bend is never read as a mismatch.
        """
        stated = self.parameter_spacing_km()
        if stated is not None:
            return stated

        strike_sizes, dip_sizes, weights = self._block_cut_sizes()
        cells_i = self.cell_counts[0]
        _refuse_mixed_resolution(strike_sizes, weights, axis="strike")
        _refuse_mixed_resolution(dip_sizes, [cells_i] * len(dip_sizes), axis="dip")

        strike_steps, dip_steps = self.line_steps()
        strike_means: list[float] = []
        dip_means: list[float] = []
        for _plane, start, stop in self.blocks():
            strike_means.append(float(strike_steps[:, start:stop].mean()))
            dip_means.append(float(dip_steps[:, start : stop + 1].mean()))

        return (
            float(np.average(strike_means, weights=weights)),
            float(np.average(dip_means, weights=weights)),
        )

    # ------------------------------------------------------- the hypocentre seam

    def cell_index(self, strike_km: float, dip_km: float) -> tuple[int, int]:
        """The cell containing an in-fault position, as 0-based ``(i, j)``.

        Not the SRF's ``shyp``, which is measured from the along-strike centre, and not a
        node index. A position on an interior boundary belongs to the upper cell; one on
        the far edge belongs to the last cell.

        Raises
        ------
        GeometryError
            For a position off the fault, naming the axis and the fault's extent.
        """
        return (
            _locate(dip_km, self.dip_arc_km(), axis="dip"),
            _locate(strike_km, self.strike_arc_km(), axis="strike"),
        )

    def boundary_faces(self) -> IntArray:
        """Flat indices of the chart's perimeter cells, ascending, strike-fastest.

        All four edges, including the surface trace: a front reaching the trace is a real
        arrest, and excluding it would be a minimum jump depth under another name.
        """
        rows, columns = self.cell_counts
        on_edge = np.zeros((rows, columns), dtype=bool)
        on_edge[(0, rows - 1), :] = True
        on_edge[:, (0, columns - 1)] = True
        return np.flatnonzero(on_edge.reshape(-1))

    def cell_key(self, flat_index: int) -> tuple[int, int]:
        """The ``(i, j)`` a flat strike-fastest index names -- this chart's own label.

        What a :class:`~rupture_generator.propagation.Jump` records, so that
        ``field[jump.parent_cell]`` indexes a field of this chart's shape.
        """
        return tuple(
            int(index) for index in np.unravel_index(flat_index, self.cell_counts)
        )


def _refuse_mixed_resolution(
    sizes: list[float], counts: list[int], *, axis: str
) -> None:
    """Refuse blocks cut at resolutions too far apart to average into one grid.

    The bound scales with how short the shortest block is: a plane cut into ``n`` cells
    has a realised size within ``1/(2n)`` of the size requested, so a five-cell plane can
    be a legitimate 20% from its neighbour. The shipped Alpine-Hope traces at 100 m have
    planes of five cells that a flat 10% bound refused.
    """
    from_rounding = 1.0 / min(counts)
    permitted = max(SPACING_SPREAD, from_rounding)

    spread = (max(sizes) - min(sizes)) / min(sizes)
    if spread > permitted:
        raise GeometryError(
            f"the planes were cut into {axis} subfaults of "
            f"{[f'{size:.3g}' for size in sizes]} km, a {spread:.0%} spread against "
            f"the {permitted:.0%} that rounding onto their cell counts could produce. "
            "The generator runs on one grid with one spacing -- give the planes the "
            "same subfault size"
        )


def _locate(position_km: float, arc_km: FloatArray, *, axis: str) -> int:
    """Which cell an arc-length position lands in.

    0-based; ties go up; the far edge belongs to the last cell.
    """
    extent_km = float(arc_km[-1])
    if position_km < 0.0 or position_km > extent_km:
        raise GeometryError(
            f"hypocentre: {axis}_km {position_km} is off the fault, whose {axis} "
            f"extent is {extent_km:.2f} km"
        )
    return int(np.searchsorted(arc_km[1:-1], position_km, side="right"))


# S1 + S2: geometry config to charts


def cell_counts(
    discretisation: Discretisation, length_km: float, width_km: float
) -> tuple[int, int]:
    """How many cells a plane gets, from a size or from explicit counts.

    Returns ``(strike_count, dip_count)``. A size is a request: the plane is cut into
    whole cells, rounded to nearest and floored at one, so the size actually used is the
    plane's own length over the count.
    """
    if discretisation.subfault_size_km is not None:
        size = discretisation.subfault_size_km
        return (
            max(1, round(length_km / size)),
            max(1, round(width_km / size)),
        )
    assert discretisation.strike_count and discretisation.dip_count
    return discretisation.strike_count, discretisation.dip_count


def _subdivide(
    corners: list[FloatArray], strike_cells: int, dip_cells: int
) -> FloatArray:
    """Bilinear subdivision of a quad into ``(n_i+1, n_j+1, 3)`` nodes.

    Exact for a parallelogram: both top corners step down dip by the same vector.
    ``np.arange(n+1) / n`` rather than an accumulated step, which loses the endpoint.
    """
    c0, c1, c2, c3 = corners
    a = (np.arange(strike_cells + 1) / strike_cells)[:, None]
    top = (1.0 - a) * c0 + a * c1
    bottom = (1.0 - a) * c3 + a * c2
    d = (np.arange(dip_cells + 1) / dip_cells)[:, None, None]
    return (1.0 - d) * top[None, :, :] + d * bottom[None, :, :]


def _conforming(near: object, far: object) -> bool:
    """Whether two adjacent planes hang the same way.

    Exact float equality: these are values a person wrote down, and the question is
    whether they wrote the same one.
    """
    return (
        near.dip_deg == far.dip_deg
        and near.dip_direction == far.dip_direction
        and near.bottom_depth_km == far.bottom_depth_km
    )


def build_fault(fault: FaultConfig, crs: pyproj.CRS) -> list[RuptureMesh]:
    """S1 + S2 for a fault: trace to planar charts, one per config plane.

    One chart per plane, in trace order, all sharing the surface origin. Fusing
    conforming neighbours is :func:`fuse`'s job.

    Raises
    ------
    GeometryError
        For a repeated trace point or a bend of 120 degrees or more.
    """
    origin_e, origin_n = to_projected(
        crs, fault.origin.longitude_deg, fault.origin.latitude_deg
    )

    # The trace as offsets, before any other arithmetic, so every number stays at fault
    # scale rather than CRS scale.
    trace = [np.array([0.0, 0.0])]
    for plane in fault.planes:
        east, north = to_projected(crs, plane.end.longitude_deg, plane.end.latitude_deg)
        trace.append(np.array([east - origin_e, north - origin_n]))

    count = len(fault.planes)
    lengths = [float(np.hypot(*(trace[k + 1] - trace[k]))) for k in range(count)]
    for k, length in enumerate(lengths):
        if not (length > 0.0):
            raise GeometryError(
                f"{fault.name}: trace points {k} and {k + 1} coincide -- a trace "
                "segment needs a positive length"
            )
    bearings = [_bearing_deg(trace[k], trace[k + 1]) for k in range(count)]

    deflections = [
        _normalise_bearing(bearings[v] - bearings[v - 1] + 180.0) - 180.0
        for v in range(1, count)
    ]
    for v, deflection in enumerate(deflections, start=1):
        if abs(deflection) >= SHARPEST_BEND_DEG:
            raise GeometryError(
                f"{fault.name}: the trace turns {abs(deflection):.1f} degrees at "
                f"point {v}, which doubles back -- the bend stretch would exceed 2"
            )

    reaches = [
        (plane.bottom_depth_km - fault.top_depth_km) / np.tan(np.radians(plane.dip_deg))
        for plane in fault.planes
    ]
    quarter_turns = [-90.0 if plane.dips_left else 90.0 for plane in fault.planes]

    # Bottom corners per trace vertex. At a conforming junction the shared corner is
    # placed once, along the bisector, stretched by 1/cos(deflection/2) so it lies in
    # both planes; without the stretch the planes diverge below the vertex by a measured
    # 1.285 km on the hope example.
    bottom_near: list[FloatArray] = [None] * count  # plane k's corner at vertex k
    bottom_far: list[FloatArray] = [None] * count  # plane k's corner at vertex k+1

    for vertex in range(count + 1):
        arriving = vertex - 1 if vertex > 0 else None
        leaving = vertex if vertex < count else None

        if (
            arriving is not None
            and leaving is not None
            and _conforming(fault.planes[arriving], fault.planes[leaving])
        ):
            deflection = deflections[vertex - 1]
            azimuth = bearings[arriving] + deflection / 2.0 + quarter_turns[arriving]
            stretch = 1.0 / np.cos(np.radians(deflection / 2.0))
            shared = _along(trace[vertex], azimuth, reaches[arriving] * stretch)
            bottom_far[arriving] = shared
            bottom_near[leaving] = shared
            continue

        if arriving is not None:
            bottom_far[arriving] = _along(
                trace[vertex],
                bearings[arriving] + quarter_turns[arriving],
                reaches[arriving],
            )
        if leaving is not None:
            bottom_near[leaving] = _along(
                trace[vertex],
                bearings[leaving] + quarter_turns[leaving],
                reaches[leaving],
            )

    charts: list[RuptureMesh] = []
    for k, plane in enumerate(fault.planes):
        width_km = (plane.bottom_depth_km - fault.top_depth_km) / np.sin(
            np.radians(plane.dip_deg)
        )
        strike_cells, dip_cells = cell_counts(
            plane.discretisation, lengths[k], float(width_km)
        )
        corners = [
            np.array([*trace[k], fault.top_depth_km]),
            np.array([*trace[k + 1], fault.top_depth_km]),
            np.array([*bottom_far[k], plane.bottom_depth_km]),
            np.array([*bottom_near[k], plane.bottom_depth_km]),
        ]
        nodes = _subdivide(corners, strike_cells, dip_cells)
        charts.append(
            RuptureMesh.from_nodes(
                nodes[..., 0],
                nodes[..., 1],
                nodes[..., 2],
                origin_east_km=origin_e,
                origin_north_km=origin_n,
                surface=fault.name,
                plane_of_column=np.full(strike_cells, k, dtype=np.int64),
            )
        )
    return charts


def build_point(point: PointConfig, crs: pyproj.CRS) -> list[RuptureMesh]:
    """S1 + S2 for a point source: an ordinary one-cell chart.

    A point is given by its centre; a chart by its corners.

    Raises
    ------
    GeometryError
        If the cell's top edge would be above the ground -- a 1 km cell dipping 60
        degrees reaches 0.43 km above a centre at 0.2 km.
    """
    origin_e, origin_n = to_projected(
        crs, point.centre.longitude_deg, point.centre.latitude_deg
    )

    half = 0.5 * point.size_km
    dip_radians = np.radians(point.dip_deg)
    top_depth_km = point.depth_km - half * np.sin(dip_radians)
    if top_depth_km < 0.0:
        raise GeometryError(
            f"{point.name}: a {point.size_km} km cell dipping {point.dip_deg} "
            f"degrees centred at {point.depth_km} km reaches {-top_depth_km:.2f} km "
            "above the surface -- deepen the centre or shrink the cell"
        )
    bottom_depth_km = top_depth_km + point.size_km * np.sin(dip_radians)
    down_dip_azimuth = point.strike_deg + 90.0
    reach_km = (bottom_depth_km - top_depth_km) / np.tan(dip_radians)

    centre = np.array([0.0, 0.0])
    top_centre = _along(centre, down_dip_azimuth, -half * np.cos(dip_radians))
    near = _along(top_centre, point.strike_deg, -half)
    far = _along(top_centre, point.strike_deg, half)

    corners = [
        np.array([*near, top_depth_km]),
        np.array([*far, top_depth_km]),
        np.array([*_along(far, down_dip_azimuth, reach_km), bottom_depth_km]),
        np.array([*_along(near, down_dip_azimuth, reach_km), bottom_depth_km]),
    ]
    nodes = _subdivide(corners, 1, 1)
    return [
        RuptureMesh.from_nodes(
            nodes[..., 0],
            nodes[..., 1],
            nodes[..., 2],
            origin_east_km=origin_e,
            origin_north_km=origin_n,
            surface=point.name,
        )
    ]


def build_surface(
    surface: FaultConfig | PointConfig, crs: pyproj.CRS
) -> list[RuptureMesh]:
    """Discretise one surface: the dispatch the mesh CLI calls."""
    # A local import, so `mesh` can be imported without the config package.
    from rupture_generator.config.geometry import PointConfig as _PointConfig

    if isinstance(surface, _PointConfig):
        return build_point(surface, crs)
    return build_fault(surface, crs)


# Fusion: per-plane charts to segments


def seam_gap_km(near: RuptureMesh, far: RuptureMesh) -> float:
    """How far apart two charts' shared node columns are, in kilometres.

    The maximum over the column, not the first node: planes sharing a trace vertex agree
    there whatever their dips, so the disagreement grows with depth. Infinity when the
    columns have different node counts, which is not one grid either way.
    """
    last = near.nodes()[:, -1]
    first = far.nodes()[:, 0]
    if last.shape != first.shape:
        return float("inf")
    return float(np.linalg.norm(last - first, axis=-1).max())


def fuse(charts: list[RuptureMesh]) -> list[RuptureMesh]:
    """Concatenate per-plane charts into segments along strike.

    Adjacent charts whose seam columns coincide (within :data:`SEAM_TOLERANCE_KM`) are
    one surface and fuse, the shared column stored once. Charts that do not are a segment
    boundary: two segments, not an error.

    Raises
    ------
    GeometryError
        When two charts of one surface coincide at the seam but are cut into different
        dip rows, or when a fused segment's spacings are too far apart to average.
    """
    if not charts:
        return []

    segments: list[list[RuptureMesh]] = [[charts[0]]]
    for chart in charts[1:]:
        near = segments[-1][-1]
        gap = seam_gap_km(near, chart)
        if gap <= SEAM_TOLERANCE_KM:
            segments[-1].append(chart)
            continue
        # Different geometry, or same geometry differently cut: the seam column's corner
        # nodes are shared vertices when the geometry conforms, whatever the cut.
        near_corner_top = near.nodes()[0, -1]
        near_corner_bottom = near.nodes()[-1, -1]
        far_corner_top = chart.nodes()[0, 0]
        far_corner_bottom = chart.nodes()[-1, 0]
        corners_agree = (
            np.linalg.norm(near_corner_top - far_corner_top) <= SEAM_TOLERANCE_KM
            and np.linalg.norm(near_corner_bottom - far_corner_bottom)
            <= SEAM_TOLERANCE_KM
        )
        if corners_agree:
            raise GeometryError(
                f"{chart.surface}: two conforming planes are cut into "
                f"{near.cell_counts[0]} and {chart.cell_counts[0]} rows down dip, "
                "so they are not one grid. Give them the same dip discretisation"
            )
        segments.append([chart])

    fused: list[RuptureMesh] = []
    for parts in segments:
        if len(parts) == 1:
            fused.append(parts[0])
        else:
            nodes = np.concatenate(
                [parts[0].nodes()] + [part.nodes()[:, 1:] for part in parts[1:]],
                axis=1,
            )
            plane_of_column = np.concatenate([part.planes() for part in parts])
            masks = [part.occupied() for part in parts]
            merged = RuptureMesh.from_nodes(
                nodes[..., 0],
                nodes[..., 1],
                nodes[..., 2],
                origin_east_km=parts[0].origin_km[0],
                origin_north_km=parts[0].origin_km[1],
                surface=parts[0].surface,
                plane_of_column=plane_of_column,
                # Omitted when every part fills its own rectangle, so a fault built
                # from a config stores no mask at all.
                occupied=(
                    None
                    if all(mask.all() for mask in masks)
                    else np.concatenate(masks, axis=1)
                ),
            )
            fused.append(merged)
        # Eager, so a spacing spread past the bound refuses here rather than later.
        fused[-1].spacing_km()
    return fused


# S3: chart validation


def validate_chart(mesh: RuptureMesh) -> None:
    """Assert a chart satisfies the spectral sampler's assumptions.

    The only code that knows what the sampler needs, which is one regular grid, not a
    flat surface. A chart cut on a reference plane already is one and returns here
    immediately.

    A chart built from a config is checked for three claims, each with its own
    tolerance. Every line is evenly divided (to :data:`UNIFORM_SPACING_TOLERANCE`,
    against a measured round-off floor of ~1e-14); lines agree with each other to within
    the bend stretch, which reaches 2.4% on the shipped ``hope`` example; and each
    constant-plane block is planar, because a fused bent fault has a kink at every seam.

    Raises
    ------
    GeometryError
        Naming what failed and which plane. Non-finite nodes are refused at
        construction.
    """
    if mesh.parameter_spacing_km() is not None:
        return

    strike_steps, dip_steps = mesh.line_steps()
    nodes = mesh.nodes()

    for plane, start, stop in mesh.blocks():
        rows = strike_steps[:, start:stop]
        columns = dip_steps[:, start : stop + 1]

        for axis, steps, lines in (
            ("strike", rows, rows.mean(axis=1)),
            ("dip", columns, columns.mean(axis=0)),
        ):
            within = float(steps.max(axis=None) - steps.min(axis=None))
            per_line = float(
                (
                    steps.max(axis=1 if axis == "strike" else 0)
                    - steps.min(axis=1 if axis == "strike" else 0)
                ).max()
            )
            mean = float(steps.mean())
            if per_line > UNIFORM_SPACING_TOLERANCE * mean:
                raise GeometryError(
                    f"{mesh.surface}: plane {plane} has a line whose {axis} steps "
                    f"are not one spacing -- they spread {per_line:.3g} km within a "
                    "single line. A chart this package builds divides every line "
                    "evenly, so this mesh came from somewhere else and cannot be "
                    "sampled on one grid"
                )
            spread = (lines.max() - lines.min()) / mean
            if spread > _MAX_BEND_SPREAD:
                raise GeometryError(
                    f"{mesh.surface}: plane {plane}'s {axis} spacing varies by "
                    f"{spread:.1%} across the plane, which is too far from one grid "
                    "to sample on one. A bend stretches the surface by the horizontal "
                    "reach of the dip, so a shallow-dipping fault turning sharply "
                    "skews far more than a steep one -- dip more steeply, turn less, "
                    "or give the planes different dips so they become two segments"
                )
            del within

        block = nodes[:, start : stop + 1].reshape(-1, 3)
        centred = block - block.mean(axis=0)
        # The least-variance direction is the block's normal; the worst node's distance
        # along it is the planarity residual, in kilometres.
        _, _, rotation = np.linalg.svd(centred, full_matrices=False)
        residual = float(np.abs(centred @ rotation[2]).max())
        if residual > PLANARITY_TOLERANCE_KM:
            raise GeometryError(
                f"{mesh.surface}: plane {plane} deviates {residual:.3g} km from "
                "flat, and the spectral sampler assumes a planar chart. A curved "
                "surface needs the kernel sampler, which is not written"
            )


# The projection seam, used by both output paths


def project_cells(mesh: RuptureMesh, crs: pyproj.CRS) -> xr.Dataset:
    """Cell-centred WGS84 positions and true-north angles, on dims ``(i, j)``.

    The one place the origin is added back, and the one place strike crosses from grid
    north to true north. Dip and area cross unchanged.
    """
    centres = mesh.centres()
    origin_e, origin_n = mesh.origin_km
    to_wgs84 = pyproj.Transformer.from_crs(crs, WGS84, always_xy=True)
    longitude_deg, latitude_deg = to_wgs84.transform(
        (origin_e + centres[..., 0]) * M_PER_KM,
        (origin_n + centres[..., 1]) * M_PER_KM,
    )
    longitude_deg = np.asarray(longitude_deg, dtype=np.float64)
    latitude_deg = np.asarray(latitude_deg, dtype=np.float64)

    grid_strike_deg, dip_deg = mesh.strike_dip_deg()
    true_strike_deg = np.mod(
        grid_strike_deg + grid_convergence_deg(crs, longitude_deg, latitude_deg),
        360.0,
    )

    return xr.Dataset(
        {
            "centre_longitude_deg": (("i", "j"), longitude_deg),
            "centre_latitude_deg": (("i", "j"), latitude_deg),
            "centre_depth_km": (("i", "j"), centres[..., 2]),
            "strike_deg": (("i", "j"), true_strike_deg),
            "dip_deg": (("i", "j"), dip_deg),
            "area_km2": (("i", "j"), mesh.areas_km2()),
        }
    )


__all__ = [
    "PLANARITY_TOLERANCE_KM",
    "SEAM_TOLERANCE_KM",
    "SHARPEST_BEND_DEG",
    "SPACING_SPREAD",
    "UNIFORM_SPACING_TOLERANCE",
    "WGS84",
    "RuptureMesh",
    "build_fault",
    "build_point",
    "build_surface",
    "cell_counts",
    "fuse",
    "grid_convergence_deg",
    "project_cells",
    "seam_gap_km",
    "to_projected",
    "validate_chart",
]
