"""The one mesh type: a fault surface as nodes, and everything else derived.

A segment is a chart ``X: (i, j) -> R^3`` -- a structured grid of node positions in a
projected Cartesian CRS the modeller named, ``i`` down-dip, ``j`` along-strike, depth
positive down. :class:`RuptureMesh` wraps that chart as an `xarray.Dataset` and carries
**methods, not stored copies**, for every derived quantity: cell centres, areas, local
strike and dip, arc lengths, spacing. A derived quantity written down is a second
description of the geometry, free to drift from the first.

This module is S1 (geometry to coarse mesh), S2 (subdivision), S3 (chart validation),
fusion, the hypocentre's arc-length-to-cell seam, and the one projected-to-WGS84 seam.
It is also the only module in the package that imports `pyproj`.

# The frame

Everything here is plain vector arithmetic in the projected CRS, in kilometres. The
measurement that settled that is in the git history's ``MESH.md``: on the WGS84
ellipsoid a 60 km subduction interface came out with cell areas 1.4e-2 low -- larger
than the slip bound -- where in the projection the same quantities are exact
identities. Two consequences: strike computed here is **grid north** (section on the
projection seam below), and dip and rake cross to WGS84 unchanged.

Positions are **offsets from a per-surface origin**, never absolute CRS coordinates.
An NZTM northing reaches ~5,180 km against a ~1 km subfault, so an absolute vertex is
rounded at CRS scale -- 1.2e-12 relative against 3e-15 for offsets, a measured factor
of ~400. The offset is taken *before any other arithmetic*, and the origin is added
back at exactly one place: the projection seam.

# Fusion produces segments, not refusals

A fault whose trace bends is one continuous surface cut into one grid whose strike
varies along it. Planes that hang the same way -- equal dip, dip direction and bottom
depth, written down as the same numbers -- share their seam column exactly and fuse
into one chart. Planes that do not are **two segments**: where the old pipeline
refused ("multi-segment is not written"), this one returns both charts and lets the
caller decide what it can propagate across. That is the multi-segment seam PLAN
designs in now.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pyproj
import xarray as xr

from rupture_generator.units import M_PER_KM

if TYPE_CHECKING:
    from rupture_generator.config.geometry import (
        Discretisation,
        FaultConfig,
        PointConfig,
    )

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

WGS84 = pyproj.CRS("EPSG:4326")
"""What an SRF's coordinates are, and what everything downstream of one expects."""

SHARPEST_BEND_DEG = 120.0
"""The steepest trace bend a fused surface accepts.

The bend column is stretched by ``1 / cos(deflection / 2)`` to lie in both planes at
once; at 120 degrees the stretch is exactly 2, so the cells flanking the bend would be
twice the size of every other cell -- past which calling the grid uniform stops being
defensible.
"""

SEAM_TOLERANCE_KM = 1.0e-6
"""How far apart two planes' shared node columns may be and still be one surface.

A millimetre. Planes that genuinely share a column are built from the same trace
vertex and the same dip, so they agree to round-off -- around 1e-13 km at fault scale.
Planes that differ in dip, dip direction or width diverge by *kilometres* below the
seam: the kaikoura example, at 70 and 55 degrees, separates by 3.5 km at its deepest
row. Six orders above the floor and six below anything real, which is the widest gap
available and means the check never has to be argued about.
"""

SPACING_SPREAD = 0.10
"""The widest relative spread of per-plane spacings a fused surface accepts.

The bound is what rounding one requested size can produce: a plane of length L cut at
size s gets cells within s/2L of s -- under 2% on a 27 km plane at 1 km, reaching 10%
only at about five cells. More than that means the planes were cut at genuinely
different resolutions, which is a request rather than a rounding, and averaging it
would silently split the difference.
"""

UNIFORM_SPACING_TOLERANCE = 1.0e-9
"""Relative spread above which an edge's steps are not one spacing.

Both ends argued: the floor is f64 round-off, about 1e-15 relative, since bilinear
refinement of a parallelogram gives exactly equal steps; the ceiling is what a person
can ask for -- a chart this module builds is always uniform, so the check can only be
reached by a hand-edited file or an importer, and the smallest mistake worth catching
there is a factor of two. 1e-9 is six orders from each.
"""

PLANARITY_TOLERANCE_KM = 1.0e-6
"""The worst out-of-plane node distance a planar chart may have.

Same construction as the seam tolerance: a chart this module builds is planar to
round-off (~1e-13 km at fault scale), and any real non-planarity -- a curved import --
is kilometres.
"""

_MAX_BEND_SPREAD = 1.0 / np.cos(np.radians(SHARPEST_BEND_DEG / 2.0)) - 1.0
"""How far a block's line spacings may spread around their mean.

Set to what the sharpest accepted bend contributes through the stretch alone --
derived from :data:`SHARPEST_BEND_DEG` so the two cannot drift; at 120 degrees the
stretch is 2, so the bound is 1.

The stretch is not the only contribution. Rotating the down-dip direction by half the
deflection swings the block's *bottom* edge by the horizontal reach of the dip, which
is ``depth_span / tan(dip)`` -- 46 km on a 4 km-deep fault dipping 5 degrees, against
a 27 km plane. A shallow fault that turns sharply therefore skews far past this bound
while a steep one at the same bend does not, which is why the check is on the measured
spread rather than on the deflection.
"""

_DOWN = np.array([0.0, 0.0, 1.0])


# ============================================================================
# Primitives. Bearings are degrees clockwise from grid north, which is why the
# arguments to atan2 are (east, north) rather than the mathematical (y, x).
# ============================================================================


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
    """A longitude and latitude as an easting and northing, in **kilometres**.

    The way in. A trace is digitised in longitude and latitude, and the mesh is built
    in the CRS, so this runs once per trace point.

    Returns
    -------
    tuple of float
        Easting and northing, in kilometres.
    """
    easting_m, northing_m = pyproj.Transformer.from_crs(
        WGS84, crs, always_xy=True
    ).transform(longitude_deg, latitude_deg)
    return easting_m / M_PER_KM, northing_m / M_PER_KM


def grid_convergence_deg(
    crs: pyproj.CRS, longitude_deg: FloatArray, latitude_deg: FloatArray
) -> FloatArray:
    """The angle from true north to grid north, in degrees, at each point.

    Add it to a grid azimuth to get a true one. Zero on the projection's central
    meridian and growing away from it -- in NZTM2000, whose central meridian is 173
    degrees east, it runs from about -3.4 degrees at East Cape to +5.0 in Fiordland:
    five times the one-degree rake bound, and about the width of the whole difference
    between a reverse and an oblique-reverse mechanism. A strike written without it is
    wrong by more than the SRF can express.

    Parameters
    ----------
    crs : pyproj.CRS
        The projected CRS the grid azimuth was measured in.
    longitude_deg, latitude_deg : FloatArray
        Where to evaluate it. The convergence varies across a fault, so this is per
        subfault rather than one number for the mesh.

    Returns
    -------
    FloatArray
        Degrees, the same shape as the inputs.
    """
    factors = pyproj.Proj(crs).get_factors(longitude_deg, latitude_deg)
    return np.asarray(factors.meridian_convergence, dtype=np.float64)


# ============================================================================
# The mesh type
# ============================================================================


NODE_VARIABLES = ("east_km", "north_km", "depth_km")
"""The chart's own geometry, on ``(i_node, j_node)``. What :meth:`RuptureMesh.node_dataset`
hands out, and what a mesh file stores."""

CELL_DIMS = ("i", "j")
"""The dims a stage's field lives on: ``i`` down dip, ``j`` along strike."""

RESERVED_FIELDS = frozenset({*NODE_VARIABLES, "plane", "slip_rate", "slip_rate_offset"})
"""Names a stage may not attach a field under.

Not tidiness. The node variables live on ``(i_node, j_node)`` and a field lives on
``(i, j)``, so ``with_fields(depth_km=...)`` would not *collide* -- it would sit beside
the geometry under the geometry's own name, and the next reader of ``depth_km`` would
get whichever xarray handed back.
"""

RESERVED_ATTRS = frozenset({"surface", "origin_east_km", "origin_north_km"})
"""What the chart *is*, as against what a stage found out about it.

A stage that rewrote one of these would move the fault, and every derived quantity
after it would describe somewhere else.
"""


@dataclasses.dataclass(frozen=True, eq=False)
class RuptureMesh:
    """One chart of a fault surface, and whatever has been attached to it.

    The dataset holds ``east_km``, ``north_km`` and ``depth_km`` on dims
    ``(i_node, j_node)`` -- offsets from the surface origin in ``attrs`` -- plus a
    ``plane`` coordinate on the cell dim ``j`` recording which config plane each cell
    column came from.

    # Two vocabularies, one chart

    Geometry is **derived**: nodes in, methods out, and nothing stored that could be
    computed. Fields are **given**: a stage draws an ``(i, j)`` array, hands it back
    under a name, and no later stage can tell which stage put it there. The mapping
    protocol is over the fields alone -- ``mesh["slip_m"]`` is something a stage
    remembered, ``mesh.areas_km2()`` is something this class computes, and they are
    spelled differently because they are different kinds of thing.

    # The dataset is private

    Its dims, coordinate names and variable attributes are the file formats' business
    and nobody else's. A stage reaching through it could write on the wrong dims, or
    attach a field under a geometry variable's name and quietly replace the fault --
    which is what a bare ``dataset.assign`` allows. :meth:`node_dataset` is the one way
    out, and what it hands out is the geometry.

    Equality is identity (``eq=False``). The generated ``__eq__`` would compare two
    datasets with ``==``, which returns a dataset rather than a bool, so every
    ``mesh_a == mesh_b`` raised "truth value of an array is ambiguous". Use
    :meth:`equals` for a structural comparison.
    """

    _dataset: xr.Dataset

    def __repr__(self) -> str:
        """The chart's name, shape and fields -- not the dataset behind it.

        The dataclass default prints every array, which turns one failed assertion
        into a screenful.
        """
        cells_i, cells_j = self.cell_counts
        fields = ", ".join(sorted(self.fields())) or "none"
        return (
            f"{type(self).__name__}({self.surface!r}, "
            f"{cells_i}x{cells_j} cells, fields: {fields})"
        )

    def equals(self, other: RuptureMesh) -> bool:
        """Whether two charts hold the same geometry, fields and attrs."""
        return self._dataset.equals(other._dataset)

    def _with(self, dataset: xr.Dataset) -> RuptureMesh:
        """This chart, behind a different dataset.

        The single place a chart is made from another, so "functional, never in place"
        is one line rather than a rule every method has to remember.
        """
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
    ) -> RuptureMesh:
        """Build a chart from node position arrays of shape ``(n_i+1, n_j+1)``.

        Parameters
        ----------
        east_km, north_km, depth_km : FloatArray
            Node positions, offsets from the origin, ``i`` down-dip and ``j``
            along-strike.
        origin_east_km, origin_north_km : float
            The surface origin, in the CRS, kilometres. Held in attrs and added back
            only at the projection seam.
        surface : str
            The surface's name, which becomes the group name in files.
        plane_of_column : FloatArray, optional
            Which config plane each *cell* column came from, length ``n_j``.
            Defaults to all zeros -- a single-plane chart.

        Raises
        ------
        ValueError
            If the arrays disagree in shape, are smaller than 2x2 nodes, or carry
            anything non-finite. A chart with one node on an axis has no cells, and a
            NaN travels silently into every derived quantity.
        """
        east_km = np.asarray(east_km, dtype=np.float64)
        north_km = np.asarray(north_km, dtype=np.float64)
        depth_km = np.asarray(depth_km, dtype=np.float64)
        if not (east_km.shape == north_km.shape == depth_km.shape):
            raise ValueError(
                f"the node arrays disagree in shape: east {east_km.shape}, "
                f"north {north_km.shape}, depth {depth_km.shape}"
            )
        if east_km.ndim != 2 or min(east_km.shape) < 2:
            raise ValueError(
                f"a chart needs at least 2 nodes on each axis, got {east_km.shape}"
            )
        for name, values in (
            ("east_km", east_km),
            ("north_km", north_km),
            ("depth_km", depth_km),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} carries a non-finite node position")

        cells_j = east_km.shape[1] - 1
        if plane_of_column is None:
            plane_of_column = np.zeros(cells_j, dtype=np.int64)
        plane_of_column = np.asarray(plane_of_column, dtype=np.int64)
        if plane_of_column.shape != (cells_j,):
            raise ValueError(
                f"plane_of_column has {plane_of_column.shape[0]} entries for "
                f"{cells_j} cell columns"
            )

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
            coords={"plane": ("j", plane_of_column)},
            attrs={
                "surface": surface,
                "origin_east_km": float(origin_east_km),
                "origin_north_km": float(origin_north_km),
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
        """Cells ``(n_i, n_j)`` -- one fewer than nodes on each axis."""
        return (
            self._dataset.sizes["i_node"] - 1,
            self._dataset.sizes["j_node"] - 1,
        )

    def nodes(self) -> FloatArray:
        """Node positions, shape ``(n_i+1, n_j+1, 3)``, components (east, north, depth)."""
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

    def blocks(self) -> list[tuple[int, int, int]]:
        """Contiguous constant-plane runs, as ``(plane, start, stop)`` cell columns.

        The unit of planarity and spacing on a fused chart: a bent fault is
        *piecewise* planar, one plane per block, with the seam column shared.
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
        """Every attached field's name.

        Defined as the variables whose dims are exactly the cell dims, so the dims are
        the discriminator and no second list of names has to be kept in step. Geometry
        is not in here; geometry is computed.
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
        """A field a stage attached, shaped :attr:`cell_counts`.

        Returned **read-only**: the chart is immutable, and an array that could be
        written through is a way around that which nothing would report. A caller who
        wants to change one wants a new field, and spells it ``np.array(mesh[name])``.

        Raises
        ------
        KeyError
            Naming the field and listing what this chart does carry. A stage asking
            for a field nobody drew is a pipeline written in the wrong order, and the
            list of what *is* there is most of the diagnosis.
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

    def with_fields(self, **arrays: FloatArray) -> RuptureMesh:
        """This chart with more cell fields on it. Functional, never in place.

        Keyword arguments because a field name is an identifier, so the call site
        reads as the assignment it is::

            mesh.with_fields(slip_m=slip, rake_deg=rake)

        Never in place is what lets stages *share* geometry rather than copy it: the
        returned chart holds the same node arrays as this one, and the only new objects
        are the fields themselves.

        A field of the same name is replaced, because a stage that recomputes one --
        the moment fold, sizing ``slip_pattern`` into ``slip_m`` -- is making a second
        statement about one quantity rather than adding a second quantity.

        Raises
        ------
        ValueError
            For an array that is not the chart's shape, one carrying a non-finite
            value, or a name in :data:`RESERVED_FIELDS`.

            The shape check is the one that earns its keep. xarray objects only when
            dimension *sizes* disagree, so a transposed field on a square patch is
            assigned without complaint and every quantity derived from it is quietly
            wrong. The finiteness check is `from_nodes`' argument one stage later: a
            NaN drawn here reaches the SRF with nothing having raised.
        """
        cell_counts = self.cell_counts
        prepared = {}
        for name, values in arrays.items():
            if name in RESERVED_FIELDS:
                raise ValueError(
                    f"{name!r} is the chart's own, not a field to attach; "
                    f"reserved names are {', '.join(sorted(RESERVED_FIELDS))}"
                )
            array = np.asarray(values, dtype=np.float64)
            if array.shape != cell_counts:
                raise ValueError(
                    f"{name} is shaped {array.shape}, and this chart has "
                    f"{cell_counts} cells (i down dip, j along strike)"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{name} carries a non-finite value")
            prepared[name] = (CELL_DIMS, array)

        return self._with(self._dataset.assign(prepared))

    def without(self, *names: str) -> RuptureMesh:
        """This chart with those fields dropped. Functional, never in place.

        For a working field a stage is finished with -- the unit-mean slip pattern,
        once the moment has sized it. A name that is not there is not an error:
        dropping is a statement about the result, not a claim about the history.
        """
        return self._with(self._dataset.drop_vars(names, errors="ignore"))

    @property
    def attrs(self) -> Mapping[str, Any]:
        """What this chart records about itself, read-only.

        What a stage learns that is not one value per subfault: the fraction of the
        slip field truncation clipped, and -- on the one segment that holds it -- where
        the rupture nucleated, in this chart's own arc lengths. Read-only, because a
        mutable view is a mutable chart.
        """
        return types.MappingProxyType(dict(self._dataset.attrs))

    def with_attrs(self, **values: Any) -> RuptureMesh:
        """This chart with more recorded about it. Functional, never in place.

        Scalars by convention: these are written straight into a file's group
        attributes, and an attribute is a scalar or an array, never a mapping.

        Raises
        ------
        ValueError
            For a name in :data:`RESERVED_ATTRS`, which say what the chart *is*.
        """
        reserved = RESERVED_ATTRS & set(values)
        if reserved:
            raise ValueError(
                f"{', '.join(sorted(reserved))} says what this chart is, and is not "
                "a stage's to rewrite"
            )
        return self._with(self._dataset.assign_attrs(**values))

    def with_pulses(self, offsets: np.ndarray, samples: FloatArray) -> RuptureMesh:
        """This chart with its slip-rate pulses attached. Functional, never in place.

        The one thing a stage produces that is not a cell field: a pulse per subfault,
        each its own length, so the array is ragged and no ``(i, j)`` shape describes
        it. It gets a method of its own rather than a widened :meth:`with_fields`
        because there is exactly one such quantity, it has an exact meaning, and the
        checks it wants are checks nothing else wants.

        Stored as CSR under the names the rupture file uses, so the writer copies
        rather than translates. The CSR *indices* are not stored: a pulse is
        contiguous, so they are ``arange`` and the file says so.

        Parameters
        ----------
        offsets : np.ndarray
            Where each subfault's pulse starts, length ``n_i * n_j + 1``, strike
            fastest -- the CSR indptr, so the last entry is the sample count.
        samples : FloatArray
            Every pulse, concatenated.

        Raises
        ------
        ValueError
            For an indptr that is not one: the wrong length for this chart, decreasing
            anywhere, or not ending at ``samples.size``. Each would make some
            subfault's pulse another's, which is a plausible-looking rupture nothing
            downstream could question.
        """
        cells_i, cells_j = self.cell_counts
        offsets = np.asarray(offsets, dtype=np.int64)
        samples = np.asarray(samples, dtype=np.float64)

        if offsets.shape != (cells_i * cells_j + 1,):
            raise ValueError(
                f"the pulse offsets are shaped {offsets.shape}, and this chart has "
                f"{cells_i * cells_j} subfaults, so it wants "
                f"{cells_i * cells_j + 1} (one per subfault, plus the end)"
            )
        if np.any(np.diff(offsets) < 0):
            raise ValueError("the pulse offsets decrease, so some subfault has none")
        if offsets[0] != 0 or offsets[-1] != samples.size:
            raise ValueError(
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
        """The node positions with their units, and nothing else.

        The **one** place the private dataset is handed out, and what it hands out is
        the geometry: three variables on ``(i_node, j_node)``, no attached fields, no
        pulses, no ``plane``. It exists for the mesh file, which stores a *surface*
        rather than a rupture and renames the dims to its own spelling at that seam.
        The units and long names are written down once, in :meth:`from_nodes`; this is
        how they reach a file without being written down a second time.
        """
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
        """Cell areas, shape ``(n_i, n_j)``.

        Split across the (0, 2) diagonal into two triangles and summed. Every cell
        this module builds is planar, so the split does not matter -- but the formula
        that copes with non-coplanar corners costs nothing and is what the
        curved-geometry migration needs.
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

        Both come from the cell's normal rather than its edges: on a plane the two
        agree, and the normal is what keeps them right on a surface that is not one.
        The absolute value on the normal's vertical component makes the dip
        independent of the normal's sign; the strike's sign is fixed by the cell's
        own along-strike edges, which ties it to the trace direction rather than its
        reverse. A degenerate cell reports dip 0 and the strike of its along-strike
        edge -- never NaN, which would travel silently into an SRF.
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
        """Distance along strike of each node column, measured on the top edge.

        Shape ``(n_j+1,)``, starting at zero. With the dip arc, this is what makes a
        position on the fault expressible as two lengths rather than two indices --
        which is how a hypocentre is specified.
        """
        top = self.nodes()[0]
        steps = np.linalg.norm(np.diff(top, axis=0), axis=-1)
        return np.concatenate([[0.0], np.cumsum(steps)])

    def dip_arc_km(self) -> FloatArray:
        """Distance down dip of each node row, measured on the ``j = 0`` edge.

        Shape ``(n_i+1,)``, starting at zero.
        """
        near = self.nodes()[:, 0]
        steps = np.linalg.norm(np.diff(near, axis=0), axis=-1)
        return np.concatenate([[0.0], np.cumsum(steps)])

    def line_steps(self) -> tuple[FloatArray, FloatArray]:
        """Every along-strike and every down-dip step, as ``(strike, dip)`` arrays.

        Shapes ``(n_i+1, n_j)`` and ``(n_i, n_j+1)``: the distance between adjacent
        nodes along each row and down each column. Measuring every line rather than
        one edge is what makes the uniformity assertion an actual claim about the
        chart -- an edge that is uniform says nothing about the interior of a
        surface that is not a parallelogram.
        """
        nodes = self.nodes()
        return (
            np.linalg.norm(np.diff(nodes, axis=1), axis=-1),
            np.linalg.norm(np.diff(nodes, axis=0), axis=-1),
        )

    def _block_cut_sizes(self) -> tuple[list[float], list[float], list[int]]:
        """Each block's realised subfault size, unstretched, and its cell count.

        The **unstretched reference**, which is what makes comparing two blocks a
        question about their discretisation rather than about their bends. A fused
        bend is a trapezoid, so its rows and columns differ in length; the two lines
        that carry no stretch are the *trace* (the top edge, which is the config's
        own trace and is divided evenly) and the *shortest column* (the one furthest
        from the bend). Reading the size off those separates "these planes were cut
        at different resolutions", which is a request, from "this surface bends",
        which is geometry.
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

        Each line of the chart is evenly divided (`validate_chart` asserts it), so a
        block's spacing is the mean of its steps, and the chart's is the
        cell-count-weighted mean of its blocks'. That last step is the same thing
        genslip does when it averages a GSF's per-subfault ``ds`` and ``dw`` into the
        single ``dstk`` and ``ddip`` it uses everywhere.

        The mean rather than any one line's step, because a fused bend is a
        *trapezoid*: its bottom edge is longer than its top by the bend stretch, so
        rows differ from one another by up to 2.4% on the shipped ``hope`` example.
        One grid needs one number and a mean does not depend on which line it was
        read from; `validate_chart` is where the spread around it is bounded.

        Returns
        -------
        tuple of float
            ``(strike_km, dip_km)``.

        Raises
        ------
        ValueError
            If the blocks were cut at resolutions too far apart to average -- judged
            on their unstretched sizes, so a bend is never mistaken for a
            discretisation mismatch.
        """
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

        **The one narrow conversion seam** between the config's arc lengths and the
        pipeline's indices -- `DEFECTS.md` 17 is the record of what widening it
        costs: a hypocentre one cell off in both directions correlated 0.99+ with the
        right answer while moving onsets by up to a second.

        This is not the SRF's ``shyp`` (measured from the along-strike centre; the
        SRF writer converts), not genslip's 1-based ``ixs``/``iys``, and not a node
        index. Positions exactly on an interior boundary belong to the upper cell; a
        position exactly on the far edge belongs to the last cell, because "at the
        bottom of the fault" is a thing people write.

        Raises
        ------
        ValueError
            For a position off the fault, naming the axis and the fault's extent.
        """
        return (
            _locate(dip_km, self.dip_arc_km(), axis="dip"),
            _locate(strike_km, self.strike_arc_km(), axis="strike"),
        )


def _refuse_mixed_resolution(
    sizes: list[float], counts: list[int], *, axis: str
) -> None:
    """Refuse blocks cut at resolutions too far apart to average into one grid.

    The bound scales with how *short* the shortest block is, because rounding alone
    produces more spread on a short plane than a long one. A plane cut into ``n``
    cells has a realised size within ``1/(2n)`` of the size requested -- the request
    lands anywhere in ``[n - 1/2, n + 1/2]`` cells -- so two planes can differ by
    ``1/(2n_a) + 1/(2n_b)`` through rounding and nothing else. A five-cell plane can
    therefore be a legitimate 20% from its neighbour.

    Measured: the shipped Alpine-Hope traces at 100 m have planes of five cells, and
    a flat 10% bound refused them for a spread rounding had produced. The check can
    only be as sharp as the geometry allows, and saying so is better than a constant
    that is right at one resolution.
    """
    from_rounding = 1.0 / min(counts)
    permitted = max(SPACING_SPREAD, from_rounding)

    spread = (max(sizes) - min(sizes)) / min(sizes)
    if spread > permitted:
        raise ValueError(
            f"the planes were cut into {axis} subfaults of "
            f"{[f'{size:.3g}' for size in sizes]} km, a {spread:.0%} spread against "
            f"the {permitted:.0%} that rounding onto their cell counts could produce. "
            "The generator runs on one grid with one spacing -- give the planes the "
            "same subfault size"
        )


def _locate(position_km: float, arc_km: FloatArray, *, axis: str) -> int:
    """Which cell an arc-length position lands in. 0-based; ties go up; the far edge
    belongs to the last cell."""
    extent_km = float(arc_km[-1])
    if position_km < 0.0 or position_km > extent_km:
        raise ValueError(
            f"hypocentre: {axis}_km {position_km} is off the fault, whose {axis} "
            f"extent is {extent_km:.2f} km"
        )
    return int(np.searchsorted(arc_km[1:-1], position_km, side="right"))


# ============================================================================
# S1 + S2: geometry config to charts
# ============================================================================


def cell_counts(
    discretisation: Discretisation, length_km: float, width_km: float
) -> tuple[int, int]:
    """How many cells a plane gets, from a size or from explicit counts.

    A size is a *request*: the plane is cut into whole cells, so the size actually
    used is the plane's own length over the count. Rounded to nearest rather than
    down, and floored at one -- a plane shorter than the size asked for is still a
    plane, and zero cells is not a surface.

    Parameters
    ----------
    discretisation : Discretisation
        What the config asked for.
    length_km, width_km : float
        The plane's own dimensions, which is why this cannot happen at parse time.

    Returns
    -------
    tuple of int
        ``(strike_count, dip_count)`` -- cells along strike and down dip.
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

    For the parallelograms S1 builds this is exact: both top corners step down dip by
    the same vector, so bilinear interpolation puts nodes at exactly the
    evenly-spaced positions a direct construction would. ``np.arange(n+1) / n``
    rather than an accumulated step, which loses the exact endpoint.
    """
    c0, c1, c2, c3 = corners
    a = (np.arange(strike_cells + 1) / strike_cells)[:, None]
    top = (1.0 - a) * c0 + a * c1
    bottom = (1.0 - a) * c3 + a * c2
    d = (np.arange(dip_cells + 1) / dip_cells)[:, None, None]
    return (1.0 - d) * top[None, :, :] + d * bottom[None, :, :]


def _conforming(near: object, far: object) -> bool:
    """Whether two adjacent planes hang the same way -- exact float equality,
    deliberately: these are values a person wrote down, and the question is whether
    they wrote the same one. A near miss is a typo, and reading it as a segment
    boundary places each plane where its own numbers say rather than somewhere
    between them."""
    return (
        near.dip_deg == far.dip_deg
        and near.dip_direction == far.dip_direction
        and near.bottom_depth_km == far.bottom_depth_km
    )


def build_fault(fault: FaultConfig, crs: pyproj.CRS) -> list[RuptureMesh]:
    """S1 + S2 for a fault: trace to planar charts, one per config plane.

    Parameters
    ----------
    fault : FaultConfig
        The digitised geometry: an origin, planes each giving where its top edge
        ends, a shared top depth.
    crs : pyproj.CRS
        The projected CRS to build in.

    Returns
    -------
    list of RuptureMesh
        One chart per plane, in trace order, all sharing the surface origin. Fusing
        conforming neighbours into segments is :func:`fuse`'s job, so that a mesh
        file (one group per plane) and a pipeline segment (one chart per fused run)
        are both spellings of this output.

    Raises
    ------
    ValueError
        For a repeated trace point or a bend of 120 degrees or more. Everything a
        single field could catch is already refused by the config's own validators.
    """
    origin_e, origin_n = to_projected(
        crs, fault.origin.longitude_deg, fault.origin.latitude_deg
    )

    # The trace, as offsets from the origin -- before any other arithmetic, so every
    # number stays at fault scale rather than CRS scale.
    trace = [np.array([0.0, 0.0])]
    for plane in fault.planes:
        east, north = to_projected(crs, plane.end.longitude_deg, plane.end.latitude_deg)
        trace.append(np.array([east - origin_e, north - origin_n]))

    count = len(fault.planes)
    lengths = [float(np.hypot(*(trace[k + 1] - trace[k]))) for k in range(count)]
    for k, length in enumerate(lengths):
        if not (length > 0.0):
            raise ValueError(
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
            raise ValueError(
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
    # both planes at once -- without the stretch the planes diverge below the vertex
    # by a measured 1.285 km on the hope example. Placed once means both planes read
    # the same value, which is what "one surface" means.
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
    """S1 + S2 for a point source: one cell, centred where it was told.

    A point is given by its **centre**; a chart by its corners. Walking from one to
    the other is half a cell up dip and half a cell back along strike. The result is
    an ordinary one-cell chart, because a point source is the pipeline with constant
    fields, not a special type.

    Raises
    ------
    ValueError
        If the cell's top edge would be above the ground. genslip floors the top
        depth at zero, which silently shrinks the subfault; saying so is better. A
        1 km cell dipping 60 degrees reaches 0.43 km above a centre at 0.2 km, which
        is in the air.
    """
    origin_e, origin_n = to_projected(
        crs, point.centre.longitude_deg, point.centre.latitude_deg
    )

    half = 0.5 * point.size_km
    dip_radians = np.radians(point.dip_deg)
    top_depth_km = point.depth_km - half * np.sin(dip_radians)
    if top_depth_km < 0.0:
        raise ValueError(
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
    # A local import: config.geometry imports nothing from here, but keeping the
    # dependency one-way at module load lets `mesh` be imported without the config
    # package and vice versa.
    from rupture_generator.config.geometry import PointConfig as _PointConfig

    if isinstance(surface, _PointConfig):
        return build_point(surface, crs)
    return build_fault(surface, crs)


# ============================================================================
# Fusion: per-plane charts to segments
# ============================================================================


def seam_gap_km(near: RuptureMesh, far: RuptureMesh) -> float:
    """How far apart two charts' shared node columns are, in kilometres.

    The maximum over the column, not the first node: planes sharing a trace vertex
    agree there whatever their dips, so the disagreement shows below the top edge
    and grows with depth. Geometric rather than a list of scalar comparisons because
    a dip, dip-direction or width change all show up the same way -- kilometres of
    separation on the fault -- and one test covers all three.

    Returns infinity when the columns have different node counts: planes cut into
    different dip rows are not one grid whatever their positions.
    """
    last = near.nodes()[:, -1]
    first = far.nodes()[:, 0]
    if last.shape != first.shape:
        return float("inf")
    return float(np.linalg.norm(last - first, axis=-1).max())


def fuse(charts: list[RuptureMesh]) -> list[RuptureMesh]:
    """Concatenate per-plane charts into segments along strike.

    Adjacent charts whose seam columns coincide (within :data:`SEAM_TOLERANCE_KM`)
    are one surface and fuse into one chart, the shared column stored once. Charts
    that do not are a segment boundary -- **two segments, not an error**: whether a
    rupture can propagate across the gap is the propagation stage's question, not
    the geometry's.

    Raises
    ------
    ValueError
        When two charts of one surface coincide at the seam but are cut into
        different dip rows -- one surface at two resolutions is a config mistake,
        not a segment boundary -- or when a fused segment's per-plane spacings are
        too far apart to average (the check runs inside ``spacing_km``, but fusing
        performs it eagerly so the refusal happens at fusion time, naming the
        surface).
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
        # Distinguish "different geometry" from "same geometry, different cuts":
        # the corner nodes of the seam column are shared trace/bottom vertices when
        # the geometry conforms, whatever the discretisation.
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
            raise ValueError(
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
            merged = RuptureMesh.from_nodes(
                nodes[..., 0],
                nodes[..., 1],
                nodes[..., 2],
                origin_east_km=parts[0].origin_km[0],
                origin_north_km=parts[0].origin_km[1],
                surface=parts[0].surface,
                plane_of_column=plane_of_column,
            )
            fused.append(merged)
        # Eager: a spacing spread past the bound should refuse at fusion, not at
        # whichever later stage happens to ask first.
        fused[-1].spacing_km()
    return fused


# ============================================================================
# S3: chart validation -- the temporary stage
# ============================================================================


def validate_chart(mesh: RuptureMesh) -> None:
    """Assert a chart satisfies the spectral sampler's assumptions.

    **The temporary stage.** This is the only code allowed to know the sampler needs
    flatness; deleting it (plus swapping the sampler) is the whole curvature
    migration. Nothing here mentions padding, evenness or Nyquist: those rules are
    the sampler's own, and a chart with odd extents passes unremarked.

    Three checks, each with a different tolerance because each is a different claim.

    **Every line is evenly divided** -- each row's along-strike steps agree with one
    another, and each column's down-dip steps do. Tight (:data:`UNIFORM_SPACING_TOLERANCE`,
    against a measured round-off floor of ~1e-14), because bilinear subdivision
    divides a line exactly and anything else came from somewhere this package did
    not build.

    **Lines agree with each other to within the bend stretch.** A straight plane is
    exact to round-off, but a fused bend is a *trapezoid*: its shared column is
    stretched by ``1/cos(deflection/2)`` so it lies in both planes at once, and the
    subdivision spreads that along the block. Rows therefore differ by up to the
    stretch -- 2.4% on the shipped ``hope`` example, bounded by 2 at the 120-degree
    refusal in :func:`build_fault`. This is the accepted cost of generating a bent
    fault on one grid, and it is invisible to what consumes the chart: the sampler
    and the eikonal solver see an index space and one spacing, not the positions.

    **Each block is planar.** Per constant-plane block, not per chart, because a
    fused bent fault has a kink at every seam by construction.

    Raises
    ------
    ValueError
        Naming what failed and which plane. Non-finite nodes are refused earlier, at
        construction.
    """
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
                raise ValueError(
                    f"{mesh.surface}: plane {plane} has a line whose {axis} steps "
                    f"are not one spacing -- they spread {per_line:.3g} km within a "
                    "single line. A chart this package builds divides every line "
                    "evenly, so this mesh came from somewhere else and cannot be "
                    "sampled on one grid"
                )
            spread = (lines.max() - lines.min()) / mean
            if spread > _MAX_BEND_SPREAD:
                raise ValueError(
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
        # The least-variance direction is the block's normal; the worst node's
        # distance along it is the planarity residual, in kilometres.
        _, _, rotation = np.linalg.svd(centred, full_matrices=False)
        residual = float(np.abs(centred @ rotation[2]).max())
        if residual > PLANARITY_TOLERANCE_KM:
            raise ValueError(
                f"{mesh.surface}: plane {plane} deviates {residual:.3g} km from "
                "flat, and the spectral sampler assumes a planar chart. A curved "
                "surface needs the kernel sampler, which is not written"
            )


# ============================================================================
# The projection seam -- one function out, used by both output paths
# ============================================================================


def project_cells(mesh: RuptureMesh, crs: pyproj.CRS) -> xr.Dataset:
    """Cell-centred WGS84 positions and true-north angles, on dims ``(i, j)``.

    The one place the origin is added back, and the one place strike crosses from
    grid north to true north (plus the grid convergence, evaluated per subfault).
    Dip and area cross unchanged: dip is an angle within the plane, and the fault's
    true area is the one the modeller specified in the CRS they chose -- correcting
    it here would be a second opinion about a number nobody asked to be
    reinterpreted.
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


def project_nodes(
    mesh: RuptureMesh, crs: pyproj.CRS
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The chart's *corners* in longitude, latitude and depth.

    The counterpart of :func:`project_cells` for consumers that want the mesh
    itself -- a renderer, or a format that stores geometry rather than samples.

    Returns
    -------
    tuple of FloatArray
        Longitude, latitude and depth, each shaped ``(n_i+1, n_j+1)``.
    """
    nodes = mesh.nodes()
    origin_e, origin_n = mesh.origin_km
    to_wgs84 = pyproj.Transformer.from_crs(crs, WGS84, always_xy=True)
    longitude_deg, latitude_deg = to_wgs84.transform(
        (origin_e + nodes[..., 0]) * M_PER_KM,
        (origin_n + nodes[..., 1]) * M_PER_KM,
    )
    return (
        np.asarray(longitude_deg, dtype=np.float64),
        np.asarray(latitude_deg, dtype=np.float64),
        nodes[..., 2],
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
    "project_nodes",
    "seam_gap_km",
    "to_projected",
    "validate_chart",
]
