"""A modeller's curved surface, resampled onto a chart.

The NZ Community Fault Model distributes its subduction interfaces as GOCAD TSurf
files: a triangulation with its own connectivity and an irregular outline. Nothing
downstream of this module knows that. A surface is read, a reference plane is fitted to
it, and it is **resampled once onto a structured chart** whose nodes carry the
curvature and whose :meth:`~rupture_generator.mesh.RuptureMesh.occupied` mask carries
the outline. From there a curved interface is a `RuptureMesh` like any other, and the
sampler, the wavefront, the stages and the writers need to know nothing about it.

Resampling at build time rather than solving on the triangulation is what removes the
second geometry track. The solvers always did run on a regular grid over the parameter
plane -- a triangulated interface was binned onto one and gathered back on every call.
Doing it once, and keeping the grid, is the same arithmetic with one representation
instead of two.

The chart is a rectangle in ``(u, v)`` and an outline is not, so about a third of the
CFM Hikurangi interface's cells fall outside it. Those cells carry corner positions,
because a grid needs corners, and are marked unoccupied.

A surface must be a **Monge patch** over its reference plane -- a height ``h(u, v)``,
one sheet, no folds. A surface that folds has two heights over one ``(u, v)`` and
cannot be a chart at all; :meth:`TSurf.to_chart` refuses rather than silently keeping
whichever sheet it interpolated.
"""

from __future__ import annotations

import dataclasses
import gzip
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import NearestNDInterpolator

from rupture_generator.errors import FormatError, GeometryError
from rupture_generator.mesh import RuptureMesh
from rupture_generator.units import M_PER_KM

if TYPE_CHECKING:
    from collections.abc import Iterator

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]
BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]

_DOWN = np.array([0.0, 0.0, 1.0])

_MINIMUM_IN_PLANE_LENGTH = 1.0e-9
"""Below this a direction projected into the plane is round-off, not a direction."""

_NAME = re.compile(r"^name\s*:\s*(.+)$", re.IGNORECASE)


# The reference plane.


def implied_axes(points_km: FloatArray) -> tuple[float, float]:
    """The strike and dip a surface implies when no config states them.

    Fit the plane through the ``(n, 3)`` positions and read the strike and dip off its
    steepest descent direction -- the geologist's definition, unique up to the
    convention that a plane dips to the right of its strike. It uses only the fitted
    normal crossed with the vertical, never the SVD's in-plane axes, so it does not
    depend on the sampling.

    Raises
    ------
    GeometryError
        If the fitted plane is horizontal, where a strike is not defined at all.
    """
    points = np.asarray(points_km, dtype=np.float64)
    _, _, rotation = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    normal = rotation[2]

    # Steepest descent in the plane: straight down, less its out-of-plane part.
    down_dip = _DOWN - float(_DOWN @ normal) * normal
    length = float(np.linalg.norm(down_dip))
    if length <= _MINIMUM_IN_PLANE_LENGTH:
        raise GeometryError(
            "the best-fit plane of these points is horizontal, so it has no strike "
            "and no dip direction. State them, or check the depths are not constant"
        )
    down_dip = down_dip / length
    return (
        float(np.degrees(np.arctan2(down_dip[0], down_dip[1])) % 360.0 - 90.0) % 360.0,
        float(np.degrees(np.arcsin(np.clip(down_dip[2], -1.0, 1.0)))),
    )


def stated_axes(strike_deg: float, dip_deg: float) -> tuple[FloatArray, FloatArray]:
    """The strike and down-dip unit vectors a config's numbers name.

    Components are ``(east, north, depth)`` with depth positive down, and bearings are
    degrees clockwise from grid north -- which is why the horizontal parts are
    ``(sin, cos)``. The two ``(3,)`` unit vectors are orthogonal at every dip.
    """
    strike = np.radians(strike_deg)
    dip = np.radians(dip_deg)
    down_dip_azimuth = np.radians(strike_deg + 90.0)
    return (
        np.array([np.sin(strike), np.cos(strike), 0.0]),
        np.array(
            [
                np.cos(dip) * np.sin(down_dip_azimuth),
                np.cos(dip) * np.cos(down_dip_azimuth),
                np.sin(dip),
            ]
        ),
    )


@dataclasses.dataclass(frozen=True, eq=False)
class MongeFrame:
    """The reference plane a surface is a height over.

    Four vectors in the projected CRS, kilometres, components ``(east, north, depth)``:
    an origin on the plane and an orthonormal triple with ``e_u x e_v = n``. ``u`` runs
    along strike and ``v`` down dip, which is the parameter order; a chart's grid is
    ``(i, j) = (v, u)``, which is the solvers' order.
    """

    origin_km: FloatArray
    strike_axis: FloatArray
    dip_axis: FloatArray
    normal: FloatArray

    @classmethod
    def fit(
        cls,
        points_km: FloatArray,
        *,
        strike_deg: float,
        dip_deg: float,
        weights: FloatArray | None = None,
    ) -> MongeFrame:
        """Fit a frame to a point cloud, with the in-plane axes taken from the config.

        The normal is the smallest right singular vector of the centred ``(n, 3)``
        cloud, its sign flipped to agree with the stated ``cross(strike, dip)``; the
        in-plane axes come from the stated strike alone. ``weights`` are the square
        kilometres each point carries, which makes this the exact continuous
        least-squares plane -- a fit to bare vertices drifts by 4e-4 relative across a
        factor of four in triangle size, because a triangulation samples its own
        surface unevenly.

        Raises
        ------
        GeometryError
            If fewer than three points are given, or the stated strike is
            perpendicular to the fitted plane and so names no direction in it.
        """
        points = np.asarray(points_km, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
            raise GeometryError(
                f"a reference plane needs at least 3 points shaped (n, 3), got "
                f"{points.shape}"
            )

        share = (
            np.ones(len(points))
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        centroid = (share[:, None] * points).sum(axis=0) / share.sum()
        # sqrt on the rows turns an unweighted SVD into a weighted least squares.
        _, _, rotation = np.linalg.svd(
            np.sqrt(share)[:, None] * (points - centroid), full_matrices=False
        )
        normal = rotation[2]

        stated_strike, stated_dip = stated_axes(strike_deg, dip_deg)
        if float(normal @ np.cross(stated_strike, stated_dip)) < 0.0:
            normal = -normal

        strike_axis = stated_strike - float(stated_strike @ normal) * normal
        length = float(np.linalg.norm(strike_axis))
        if length <= _MINIMUM_IN_PLANE_LENGTH:
            raise GeometryError(
                f"the stated strike of {strike_deg} degrees lies within "
                f"{np.degrees(np.arcsin(length)):.3g} degrees of the normal to the "
                "best-fit plane of the geometry, so it names no direction in the "
                "surface. The stated strike and dip and the node positions are "
                "describing different surfaces"
            )
        strike_axis = strike_axis / length
        return cls(
            origin_km=centroid,
            strike_axis=strike_axis,
            dip_axis=np.cross(normal, strike_axis),
            normal=normal,
        )

    def project(self, points_km: FloatArray) -> FloatArray:
        """Positions as ``(u, v, h)``: along strike, down dip, and out of the plane."""
        offset = np.asarray(points_km, dtype=np.float64) - self.origin_km
        return np.stack(
            [offset @ self.strike_axis, offset @ self.dip_axis, offset @ self.normal],
            axis=-1,
        )

    def lift(self, parameters_km: FloatArray) -> FloatArray:
        """``(u, v, h)`` back to ``(east, north, depth)``. The inverse of `project`."""
        uvh = np.asarray(parameters_km, dtype=np.float64)
        return (
            self.origin_km
            + uvh[..., 0, None] * self.strike_axis
            + uvh[..., 1, None] * self.dip_axis
            + uvh[..., 2, None] * self.normal
        )


# The file.


def _lines(path: Path) -> Iterator[str]:
    """Every line of a TSurf, transparently through gzip."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        yield from handle


@dataclasses.dataclass(frozen=True, eq=False)
class TSurf:
    """One TSurf file's vertices, its parts, and the name it calls itself.

    Not a chart: the file's contents, in kilometres and depth-positive-down, with no
    frame fitted and no admissibility claimed. :meth:`to_chart` does that, and is
    separate because it needs a resolution the file does not carry.

    ``vertices_km`` is ``(V, 3)`` positions ``(east, north, depth)``, **absolute** in
    the file's CRS; ``parts`` is one ``(F, 3)`` zero-based face table per ``TFACE``.
    """

    name: str
    vertices_km: FloatArray
    parts: list[IntArray]

    def __repr__(self) -> str:
        """The surface's name and shape, not its arrays."""
        faces = ", ".join(str(len(part)) for part in self.parts)
        return (
            f"TSurf({self.name!r}, {len(self.vertices_km)} vertices, "
            f"parts of {faces} faces)"
        )

    def area_km2(self, part: int = 0) -> float:
        """The part's own surface area, summed over the triangles the file wrote.

        What a chart resampled from it is checked against: the resampling is faithful
        exactly insofar as it recovers this.
        """
        corners = self.vertices_km[self.parts[part]]
        cross = np.cross(
            corners[:, 1] - corners[:, 0],
            corners[:, 2] - corners[:, 0],
        )
        return float(0.5 * np.linalg.norm(cross, axis=1).sum())

    def to_chart(
        self,
        spacing_km: float,
        part: int = 0,
        *,
        strike_deg: float | None = None,
        dip_deg: float | None = None,
        surface: str | None = None,
    ) -> RuptureMesh:
        """Resample one part onto a structured chart at a stated resolution.

        Positions become **offsets from the part's own origin**, the minimum easting
        and northing over its vertices: an NZTM northing reaches ~5,180 km against a
        ~9 km triangle, which costs a factor of ~400 in rounding.

        A TSurf carries no strike or dip, so they default to what :func:`implied_axes`
        reads off the part's own best-fit plane.

        Parameters
        ----------
        spacing_km : float
            The cell size to cut at, on both axes. The chart is square-celled because
            the sampler's correlation lengths and the wavefront's sweep both assume it.
        part : int, optional
            Which ``TFACE`` to take. Two disconnected sheets are two parts and two
            charts: one plane fitted through both describes neither.

        Returns
        -------
        RuptureMesh
            Curved, and carrying an occupancy mask for the file's outline.

        Raises
        ------
        FormatError
            For a part index the file does not have.
        GeometryError
            For a spacing that is not positive, or a surface that folds over its own
            reference plane and so is not a chart.
        """
        if not 0 <= part < len(self.parts):
            raise FormatError(
                f"{self.name!r} has {len(self.parts)} part(s), so there is no part "
                f"{part}"
            )
        if not spacing_km > 0.0:
            raise GeometryError(f"the cell spacing must be positive, got {spacing_km}")

        faces = self.parts[part]
        used = np.unique(faces)
        renumber = np.full(len(self.vertices_km), -1, dtype=np.int64)
        renumber[used] = np.arange(len(used))
        points = self.vertices_km[used].copy()
        faces = renumber[faces]

        origin_east_km = float(points[:, 0].min())
        origin_north_km = float(points[:, 1].min())
        points[:, 0] -= origin_east_km
        points[:, 1] -= origin_north_km

        if strike_deg is None or dip_deg is None:
            strike_deg, dip_deg = implied_axes(points)

        frame = MongeFrame.fit(
            points,
            strike_deg=strike_deg,
            dip_deg=dip_deg,
            weights=_vertex_areas_km2(points, faces),
        )
        chart = _resample(
            frame,
            points,
            faces,
            spacing_km,
            surface=surface
            or (self.name if len(self.parts) == 1 else f"{self.name}_{part}"),
            origin_east_km=origin_east_km,
            origin_north_km=origin_north_km,
        )
        return chart


def read_tsurf(path: Path | str) -> TSurf:
    """Parse a GOCAD TSurf, gzipped or not.

    As much of the format as this reads:

    - ``VRTX id x y z`` (or ``PVRTX``, which adds per-vertex properties this ignores)
      -- positions in the CRS named by the file's own header, in **metres**.
    - ``ZPOSITIVE Elevation`` -- so ``z`` is height and depth is ``-z``. ``ZPOSITIVE
      Depth`` is refused rather than guessed: the two differ by a sign on every vertex.
    - ``TRGL i j k`` -- **one-based** vertex indices.
    - ``TFACE`` -- a part boundary. One file may hold several connected surfaces
      sharing one vertex numbering, and each becomes its own part.

    No CRS is read or checked: a TSurf's header names its coordinate system in a
    vocabulary that is not EPSG, and the CFM's files say ``NAME Default``. The caller
    states the CRS.

    Raises
    ------
    FormatError
        If the file declares ``ZPOSITIVE Depth``, holds no triangles, or has a ``TRGL``
        naming a vertex it never defined.
    """
    path = Path(path)
    name = path.name.split(".")[0]
    positions: list[tuple[float, float, float]] = []
    numbering: dict[int, int] = {}
    parts: list[list[list[int]]] = []

    for line in _lines(path):
        token = line.split()
        if not token:
            continue
        head = token[0].upper()
        if head in ("VRTX", "PVRTX"):
            numbering[int(token[1])] = len(positions)
            positions.append((float(token[2]), float(token[3]), float(token[4])))
        elif head == "TRGL":
            if not parts:
                parts.append([])
            try:
                parts[-1].append([numbering[int(index)] for index in token[1:4]])
            except KeyError as error:
                raise FormatError(
                    f"{path.name}: a TRGL names vertex {error.args[0]}, which no VRTX "
                    "defines. GOCAD's indices are one-based and this file's numbering "
                    "has a gap"
                ) from error
        elif head == "TFACE":
            parts.append([])
        elif head == "ZPOSITIVE" and token[1].upper() != "ELEVATION":
            raise FormatError(
                f"{path.name} declares ZPOSITIVE {token[1]}; this reader assumes "
                "Elevation and negates it to get depth. Reading a Depth file as an "
                "Elevation one mirrors the surface through sea level, which nothing "
                "downstream would notice"
            )
        else:
            match = _NAME.match(line.strip())
            if match and not positions:
                name = match.group(1).strip()

    populated = [np.array(part, dtype=np.int64) for part in parts if part]
    if not populated:
        raise FormatError(f"{path.name} holds no TRGL records, so it is not a surface")

    vertices_km = np.array(positions, dtype=np.float64)
    vertices_km[:, :2] /= M_PER_KM
    # ZPOSITIVE Elevation: z is height above sea level, and this package's third
    # component is depth below it.
    vertices_km[:, 2] /= -M_PER_KM
    return TSurf(name, vertices_km, populated)


# The resampling.


def _vertex_areas_km2(points_km: FloatArray, faces: IntArray) -> FloatArray:
    """How much surface each vertex carries: a third of each triangle it touches.

    The quadrature weight :meth:`MongeFrame.fit` needs to be a continuous least-squares
    fit rather than one biased towards wherever the triangulation is dense.
    """
    corners = points_km[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    carried = np.zeros(len(points_km), dtype=np.float64)
    np.add.at(carried, faces, areas[:, None] / 3.0)
    return carried


def _resample(
    frame: MongeFrame,
    points_km: FloatArray,
    faces: IntArray,
    spacing_km: float,
    *,
    surface: str,
    origin_east_km: float,
    origin_north_km: float,
) -> RuptureMesh:
    """Cut the frame's parameter rectangle into cells and lift the height onto it.

    Three steps, in the parameter plane:

    1. the grid, from the surface's own ``(u, v)`` bounding box at ``spacing_km``;
    2. the node heights, read off the file's own triangles by barycentric weight, and
       continued to the nearest known value outside the outline -- a rectangle's
       corners reach past an irregular outline and a chart still needs positions there;
    3. the occupancy, per cell, from whether its centre lands in a triangle at all.
    """
    uvh = frame.project(points_km)
    uv = uvh[:, :2]
    _refuse_folds(uv, faces)

    low = uv.min(axis=0)
    high = uv.max(axis=0)
    cells_u, cells_v = (
        max(int(np.ceil((high[axis] - low[axis]) / spacing_km)), 1) for axis in (0, 1)
    )
    # Centred: the grid is a whole number of cells and the surface is not, so the
    # half-cell of slack is split rather than hung off one edge.
    low = low - 0.5 * (np.array([cells_u, cells_v]) * spacing_km - (high - low))

    node_u = low[0] + spacing_km * np.arange(cells_u + 1)
    node_v = low[1] + spacing_km * np.arange(cells_v + 1)
    # (i, j) = (v, u): a chart indexes down dip first.
    grid_v, grid_u = np.meshgrid(node_v, node_u, indexing="ij")
    nodes_uv = np.stack([grid_u, grid_v], axis=-1)

    corners_uv = uv[faces]
    triangle_of, weights = _rasterise(
        corners_uv, low, spacing_km, (cells_v + 1, cells_u + 1)
    )
    located = triangle_of >= 0
    height = np.empty(triangle_of.shape, dtype=np.float64)
    height[located] = np.einsum(
        "nc,nc->n", weights[located], uvh[:, 2][faces[triangle_of[located]]]
    )
    if not located.all():
        height[~located] = NearestNDInterpolator(uv, uvh[:, 2])(nodes_uv[~located])

    nodes = frame.lift(np.concatenate([nodes_uv, height[..., None]], axis=-1))

    centres, _ = _rasterise(
        corners_uv, low + 0.5 * spacing_km, spacing_km, (cells_v, cells_u)
    )
    occupied = centres >= 0

    return RuptureMesh.from_nodes(
        nodes[..., 0],
        nodes[..., 1],
        nodes[..., 2],
        origin_east_km=origin_east_km,
        origin_north_km=origin_north_km,
        surface=surface,
        occupied=occupied,
        parameter_spacing_km=(spacing_km, spacing_km),
    )


def _rasterise(
    corners_uv: FloatArray,
    origin_uv: FloatArray,
    spacing_km: float,
    shape: tuple[int, int],
) -> tuple[IntArray, FloatArray]:
    """Locate a regular grid of points in a triangulation, by scanning the triangles.

    Point location against **the triangles the file wrote**, rather than against a
    Delaunay of the same vertices. The two are not the same triangulation: on the CFM
    Hikurangi interface a Delaunay of the projected vertices reproduces only 83% of the
    file's faces, and taking the other 17% for "outside" loses 5.9% of the surface.

    The queries are a regular grid, so there is no search structure: each triangle
    converts its own parameter bounding box straight into grid indices and tests the
    points inside it. Overlaps cannot happen on a surface that does not fold, so the
    last triangle to claim a point agrees with the first.

    Parameters
    ----------
    corners_uv : FloatArray
        ``(F, 3, 2)`` the triangles' parameter coordinates.
    origin_uv : FloatArray
        ``(2,)`` where grid point ``(0, 0)`` sits.
    shape : tuple of int
        ``(i, j)`` grid point counts, ``i`` down dip and ``j`` along strike.

    Returns
    -------
    tuple
        ``(F,)``-valued triangle index per grid point, ``-1`` outside, shaped
        ``shape``; and the ``shape + (3,)`` barycentric weights of each point in its
        triangle, zero where there is none.
    """
    rows, columns = shape
    triangle_of = np.full(rows * columns, -1, dtype=np.int64)
    weights = np.zeros((rows * columns, 3), dtype=np.float64)

    a, b, c = corners_uv[:, 0], corners_uv[:, 1], corners_uv[:, 2]
    determinant = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (
        b[:, 1] - a[:, 1]
    )
    low = corners_uv.min(axis=1)
    high = corners_uv.max(axis=1)
    first = np.ceil((low - origin_uv) / spacing_km).astype(np.int64)
    last = np.floor((high - origin_uv) / spacing_km).astype(np.int64)
    np.clip(first, 0, [columns - 1, rows - 1], out=first)
    np.clip(last, 0, [columns - 1, rows - 1], out=last)

    for triangle in np.flatnonzero(determinant != 0.0):
        js = np.arange(first[triangle, 0], last[triangle, 0] + 1)
        is_ = np.arange(first[triangle, 1], last[triangle, 1] + 1)
        if not js.size or not is_.size:
            continue
        u = origin_uv[0] + spacing_km * js
        v = origin_uv[1] + spacing_km * is_
        du = u[None, :] - a[triangle, 0]
        dv = v[:, None] - a[triangle, 1]

        # Barycentric against the (a, b, c) corners, all three in [0, 1] iff inside.
        beta = (
            du * (c[triangle, 1] - a[triangle, 1])
            - dv * (c[triangle, 0] - a[triangle, 0])
        ) / determinant[triangle]
        gamma = (
            dv * (b[triangle, 0] - a[triangle, 0])
            - du * (b[triangle, 1] - a[triangle, 1])
        ) / determinant[triangle]
        alpha = 1.0 - beta - gamma
        inside = (alpha >= 0.0) & (beta >= 0.0) & (gamma >= 0.0)
        if not inside.any():
            continue

        rows_in, columns_in = np.nonzero(inside)
        flat = is_[rows_in] * columns + js[columns_in]
        triangle_of[flat] = triangle
        weights[flat, 0] = alpha[rows_in, columns_in]
        weights[flat, 1] = beta[rows_in, columns_in]
        weights[flat, 2] = gamma[rows_in, columns_in]

    return triangle_of.reshape(shape), weights.reshape((*shape, 3))


def _refuse_folds(uv: FloatArray, faces: IntArray) -> None:
    """Refuse a surface that is not one sheet over its reference plane.

    A fold puts two heights over one ``(u, v)``, and a chart can hold one. It shows up
    as a projected triangle with no area or reversed orientation: the file's triangles
    are consistently wound on the surface, so a projection that preserves the sheet
    preserves their sign.

    Raises
    ------
    GeometryError
        Naming how much of the surface folded.
    """
    corners = uv[faces]
    signed = 0.5 * (
        (corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
        - (corners[:, 2, 0] - corners[:, 0, 0]) * (corners[:, 1, 1] - corners[:, 0, 1])
    )
    positive = float(signed[signed > 0.0].sum())
    negative = float(-signed[signed < 0.0].sum())
    folded = min(positive, negative)
    total = positive + negative
    if total > 0.0 and folded / total > _FOLD_TOLERANCE:
        raise GeometryError(
            f"{folded / total:.1%} of this surface folds back over its own reference "
            "plane, so it has two heights over one point and is not a chart. Cut it "
            "into parts that are each one sheet, or state a strike and dip whose "
            "plane the whole surface is a height over"
        )


_FOLD_TOLERANCE = 1.0e-3
"""How much of a surface may project backwards before it is not one sheet.

Not zero: a triangulation of a curved surface has slivers, and one whose projected area
comes out microscopically negative is round-off rather than geology. A tenth of a
percent is four orders above that and three below any real fold.
"""


__all__ = [
    "MongeFrame",
    "TSurf",
    "implied_axes",
    "read_tsurf",
    "stated_axes",
]
