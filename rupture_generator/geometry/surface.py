"""Curved faults as triangle meshes, unrolled into charts.

A curved fault is a triangulated sheet. :func:`read_tsurf` reads the sheets in a
GOCAD TSurf file, the format of the NZ Community Fault Model's subduction
interfaces. :func:`flatten` unrolls a sheet onto a plane with as little stretch as
it can, and :func:`surface_chart` lays a regular grid on the unrolled sheet and maps
each node back onto the mesh. The result is an ordinary :class:`Geometry`, whose
``occupied`` mask records the mesh's outline.
"""

from typing import TextIO

import numpy as np
import scipy as sp

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.geometry.geometry import Geometry

PASSES = 50
"""Local/global passes :func:`flatten` runs.

On the NZ Community Fault Model's Hikurangi and Puysegur interfaces meshed at 2 km,
50 passes reach a 95th-percentile stretch of 0.78% and 1.40%, against 0.78% and
1.33% at convergence after 135 and 455 passes. Each pass costs about 0.06 s on the
43,421-vertex Hikurangi mesh.
"""

MAXIMUM_STRETCH = 0.05
"""The largest 95th-percentile stretch a flattened mesh may have and be charted.

Stretch is a triangle's worst-direction length error, ``|log(sigma)|`` over the
singular values of its flattening, and the percentile weights each triangle by its
area. Across the 27 Slab2 subduction interfaces cut at 60 km depth the worst is 2.5%
(Hindu Kush), and 5% is twice that.
"""

_LEVEL_GRADIENT = 1e-9
"""A mean depth gradient below this, a millimetre per thousand kilometres, is a level
sheet, which has no down-dip direction to orient a chart by."""

_ROUNDING = 1e-9
"""Round-off allowed, relative to the spacing or in barycentric coordinates, when
counting cells and testing which triangle covers a lattice point. It keeps a sheet an
exact number of cells long at that many cells, and a point on an edge two triangles
share covered by one of them."""

_MINIMUM_COTANGENT = 1e-3
"""The least weight an edge has in :func:`flatten`.

An obtuse angle has a negative cotangent, which can make the system singular or
indefinite. Clamping keeps the matrix positive definite once a vertex is pinned.
"""

_RASTER_BLOCK = 1 << 22
"""Candidate lattice points tested per block while rasterising, which bounds the
temporary arrays at a few hundred megabytes."""

_EDGES = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
"""A triangle's edges as ``(start, end, opposite)`` corner indices."""


# ------------------------------------------------------------------------ reading


def _numbers(words: list[str], kinds: tuple[type, ...], line: int) -> list:
    """Parse the values after a record's keyword, one per kind, refusing bad ones."""
    try:
        if len(words) <= len(kinds):
            raise ValueError
        return [kind(word) for kind, word in zip(kinds, words[1:], strict=False)]
    except ValueError:
        raise RuptureGeneratorError(
            f"line {line}: {' '.join(words)!r} is not a {words[0]} record of "
            f"{len(kinds)} numbers"
        ) from None


def _check_sheet(name: str, triangles: np.ndarray, vertices: int) -> None:
    """Refuse triangles that fail to make one connected, consistently wound sheet.

    A sheet uses each directed edge at most once: two triangles that share an edge
    traverse it in opposite directions, and no edge borders more than two triangles.
    """
    if np.any(
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 2] == triangles[:, 0])
    ):
        raise RuptureGeneratorError(
            f"{name}: a triangle names one vertex twice, so it has no area"
        )
    starts = np.concatenate([triangles[:, a] for a, _, _ in _EDGES])
    ends = np.concatenate([triangles[:, b] for _, b, _ in _EDGES])
    if np.unique(starts * vertices + ends).size < starts.size:
        raise RuptureGeneratorError(
            f"{name}: an edge borders two triangles wound the same way, or more than "
            "two triangles, so the mesh is not one consistently wound sheet"
        )
    owner = np.tile(np.arange(len(triangles)), 3)
    key = np.minimum(starts, ends) * vertices + np.maximum(starts, ends)
    order = np.argsort(key, kind="stable")
    shared = key[order][1:] == key[order][:-1]
    left, right = owner[order][:-1][shared], owner[order][1:][shared]
    adjacency = sp.sparse.coo_array(
        (np.ones(left.size), (left, right)), shape=(len(triangles),) * 2
    )
    pieces, _ = sp.sparse.csgraph.connected_components(adjacency, directed=False)
    if pieces > 1:
        raise RuptureGeneratorError(
            f"{name}: the triangles form {pieces} pieces joined by no shared edge, "
            "and a fault is one connected sheet"
        )


def _surface(
    name: str | None,
    units_km: list[float] | None,
    depth_sign: float | None,
    ids: list[int],
    positions: list[list[float]],
    corners: list[list[int]],
) -> tuple[str, tuple[np.ndarray, np.ndarray]]:
    """Turn one TSurf object's records into its name, vertices and triangles."""
    if name is None:
        raise RuptureGeneratorError("a TSurf object has no name in its HEADER")
    if units_km is None or depth_sign is None:
        raise RuptureGeneratorError(
            f"{name}: the file states no AXIS_UNIT or no ZPOSITIVE, so its units or "
            "the sign of its third coordinate would be a guess"
        )
    if not corners:
        raise RuptureGeneratorError(f"{name}: no TRGL records, so it is not a mesh")
    order = np.argsort(ids, kind="stable")
    sorted_ids = np.asarray(ids, dtype=np.int64)[order]
    if np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise RuptureGeneratorError(f"{name}: two VRTX records share an id")
    named = np.asarray(corners, dtype=np.int64)
    found = np.minimum(np.searchsorted(sorted_ids, named), len(sorted_ids) - 1)
    undefined = sorted_ids[found] != named if len(ids) else np.ones_like(named, bool)
    if undefined.any():
        raise RuptureGeneratorError(
            f"{name}: a TRGL names vertex {named[undefined][0]}, which no VRTX defines"
        )
    triangles = order[found]
    used, triangles = np.unique(triangles, return_inverse=True)
    triangles = triangles.reshape(-1, 3)
    _check_sheet(name, triangles, len(used))
    scale = np.array([*units_km[:2], depth_sign * units_km[2]])
    vertices_km = np.asarray(positions, dtype=np.float64)[used] * scale
    return name, (vertices_km, triangles)


def read_tsurf(handle: TextIO) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read every triangulated sheet in a GOCAD TSurf file.

    The reader takes from each ``GOCAD TSurf`` object, up to its ``END``:

    - ``name:`` in the ``HEADER``, which keys the sheet.
    - ``AXIS_UNIT``, each axis in ``"m"`` or ``"km"``.
    - ``ZPOSITIVE``, ``Elevation`` (the third coordinate is height, and depth its
      negation) or ``Depth``.
    - ``VRTX id x y z`` and ``PVRTX id x y z ...``, a vertex and its id.
    - ``TRGL a b c``, a triangle by its corners' vertex ids.

    It skips every other record, ``TFACE`` included, so all the triangles of one
    object make one sheet. Coordinates stay in the file's own frame, which the
    reader neither reads nor checks.

    Parameters
    ----------
    handle : TextIO
        The open TSurf file, as text.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Each sheet's vertices, ``(n, 3)``, east, north and depth in kilometres with
        depth positive down, and its triangles, ``(m, 3)``, as 0-based indices into
        those vertices, keyed by name in file order. The vertices leave out any that no
        triangle uses.

    Raises
    ------
    RuptureGeneratorError
        For an object other than a TSurf, a record outside an object, an object
        without a name, units or ``ZPOSITIVE``, a unit or ``ZPOSITIVE`` other than
        those listed, a malformed record, a vertex id defined twice or never, a file
        that ends inside an object, two sheets with one name, a file with no
        sheet, or triangles that fail to make one connected, consistently wound
        sheet.

    Examples
    --------
    >>> import io
    >>> text = '''GOCAD TSurf 1
    ... HEADER {
    ... name:wedge
    ... }
    ... GOCAD_ORIGINAL_COORDINATE_SYSTEM
    ... AXIS_UNIT "m" "m" "m"
    ... ZPOSITIVE Elevation
    ... END_ORIGINAL_COORDINATE_SYSTEM
    ... TFACE
    ... VRTX 1 0 0 -1000
    ... VRTX 2 2000 0 -1000
    ... VRTX 3 0 3000 -4000
    ... TRGL 1 2 3
    ... END
    ... '''
    >>> vertices_km, triangles = read_tsurf(io.StringIO(text))["wedge"]
    >>> vertices_km[2].tolist(), triangles.tolist()
    ([0.0, 3.0, 4.0], [[0, 1, 2]])
    """
    surfaces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    current: dict | None = None
    for line, text in enumerate(handle, start=1):
        words = text.split()
        if not words:
            continue
        keyword = words[0]
        if keyword == "GOCAD":
            if current is not None:
                raise RuptureGeneratorError(
                    f"line {line}: a GOCAD object starts before the last one's END"
                )
            if words[1:2] != ["TSurf"]:
                raise RuptureGeneratorError(
                    f"line {line}: {text.strip()!r} is not a TSurf, the one GOCAD "
                    "object this reads"
                )
            current = {
                "name": None,
                "units_km": None,
                "depth_sign": None,
                "ids": [],
                "positions": [],
                "corners": [],
            }
            continue
        if current is None:
            raise RuptureGeneratorError(
                f"line {line}: {text.strip()!r} lies outside any GOCAD TSurf object"
            )
        if keyword == "END":
            name, surface = _surface(**current)
            if name in surfaces:
                raise RuptureGeneratorError(f"two sheets are named {name!r}")
            surfaces[name] = surface
            current = None
        elif keyword.startswith("name:") and current["name"] is None:
            current["name"] = text.strip().removeprefix("name:").strip()
        elif keyword == "AXIS_UNIT":
            units = [word.strip('"') for word in words[1:]]
            scales = {"m": 1e-3, "km": 1.0}
            if len(units) != 3 or not all(unit in scales for unit in units):
                raise RuptureGeneratorError(
                    f"line {line}: AXIS_UNIT {' '.join(words[1:])} is not three of "
                    f"{sorted(scales)}"
                )
            current["units_km"] = [scales[unit] for unit in units]
        elif keyword == "ZPOSITIVE":
            signs = {"Elevation": -1.0, "Depth": 1.0}
            if words[1:2] == [] or words[1] not in signs:
                raise RuptureGeneratorError(
                    f"line {line}: ZPOSITIVE {' '.join(words[1:])} is neither "
                    "Elevation nor Depth"
                )
            current["depth_sign"] = signs[words[1]]
        elif keyword in ("VRTX", "PVRTX"):
            identifier, *position = _numbers(words, (int, float, float, float), line)
            current["ids"].append(identifier)
            current["positions"].append(position)
        elif keyword == "TRGL":
            current["corners"].append(_numbers(words, (int, int, int), line))
    if current is not None:
        raise RuptureGeneratorError(
            f"the file ends inside sheet {current['name']!r}, before its END"
        )
    if not surfaces:
        raise RuptureGeneratorError("the file holds no GOCAD TSurf sheet")
    return surfaces


# --------------------------------------------------------------------- flattening


def _local_frames(
    vertices: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each triangle laid flat on its own, ``(m, 3, 2)``, and its area, ``(m,)``.

    The first corner is at the origin and the second on the positive first axis, so
    every frame winds anticlockwise.
    """
    first = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    second = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    length = np.linalg.norm(first, axis=1)
    along = np.einsum("ij,ij->i", second, first) / length
    across = np.sqrt(np.maximum(np.einsum("ij,ij->i", second, second) - along**2, 0.0))
    frames = np.zeros((len(triangles), 3, 2))
    frames[:, 1, 0] = length
    frames[:, 2, 0] = along
    frames[:, 2, 1] = across
    return frames, 0.5 * length * across


def _jacobians(
    frames: np.ndarray, flat: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """Each triangle's flattening as a linear map from its own frame, ``(m, 2, 2)``."""
    source = np.stack([frames[:, 1] - frames[:, 0], frames[:, 2] - frames[:, 0]], -1)
    corners = flat[triangles]
    target = np.stack(
        [corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]], -1
    )
    return target @ np.linalg.inv(source)


def flatten(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Unroll a triangulated sheet onto a plane, stretching it as little as possible.

    The method is the local/global as-rigid-as-possible parameterisation of Liu, L.,
    Zhang, L., Xu, Y., Gotsman, C. & Gortler, S. J. (2008), "A local/global approach
    to mesh parameterization", *Computer Graphics Forum* 27(5), 1495-1504. It lays
    each triangle flat on its own. Each pass then finds, per triangle and in closed
    form, the least-squares rotation of its flat copy onto the current layout
    (local), and moves every vertex in one sparse solve so each edge matches its
    rotated copy (global), with edges weighted by the cotangent of the angle opposite.
    It factorises the matrix once. The start is the projection onto the vertices'
    least-squares plane, and :data:`PASSES` passes follow.

    A developable sheet, one that unrolls without stretch, unrolls exactly.

    Parameters
    ----------
    vertices : np.ndarray
        Vertex positions, ``(n, 3)``.
    triangles : np.ndarray
        Corner indices, ``(m, 3)``, of one connected sheet wound consistently.

    Returns
    -------
    np.ndarray
        Flat coordinates, ``(n, 2)``, in the units of ``vertices``, facing the way
        that makes the triangles' summed signed area positive, so a sheet that
        unrolls without folding winds anticlockwise.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    n = len(vertices)
    frames, _ = _local_frames(vertices, triangles)

    centred = vertices - vertices.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    flat = centred @ axes[:2].T
    if np.linalg.det(_jacobians(frames, flat, triangles)).sum() < 0.0:
        flat[:, 1] *= -1.0

    cotangents = np.empty((len(triangles), 3))
    for k, (a, b, opposite) in enumerate(_EDGES):
        p = frames[:, a] - frames[:, opposite]
        q = frames[:, b] - frames[:, opposite]
        cotangents[:, k] = np.einsum("ij,ij->i", p, q) / np.abs(
            p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]
        )
    cotangents = np.maximum(cotangents, _MINIMUM_COTANGENT)

    rows, columns, values = [], [], []
    for k, (a, b, _) in enumerate(_EDGES):
        i, j, c = triangles[:, a], triangles[:, b], cotangents[:, k]
        rows += [i, j, i, j]
        columns += [i, j, j, i]
        values += [c, c, -c, -c]
    laplacian = sp.sparse.csc_array(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(n, n),
    )
    # Pinning vertex 0 where the start put it removes the translation the energy
    # leaves free, and makes the rest of the system positive definite.
    solve = sp.sparse.linalg.factorized(laplacian[1:, 1:])
    pinned = laplacian[1:, [0]].toarray()

    for _ in range(PASSES):
        covariance = np.zeros((len(triangles), 2, 2))
        for k, (a, b, _) in enumerate(_EDGES):
            covariance += cotangents[:, k, None, None] * np.einsum(
                "ti,tj->tij",
                flat[triangles[:, a]] - flat[triangles[:, b]],
                frames[:, a] - frames[:, b],
            )
        angle = np.arctan2(
            covariance[:, 1, 0] - covariance[:, 0, 1],
            covariance[:, 0, 0] + covariance[:, 1, 1],
        )
        cos, sin = np.cos(angle), np.sin(angle)
        rotations = np.stack([np.stack([cos, -sin], -1), np.stack([sin, cos], -1)], 1)
        load = np.zeros((n, 2))
        for k, (a, b, _) in enumerate(_EDGES):
            target = cotangents[:, k, None] * np.einsum(
                "tij,tj->ti", rotations, frames[:, a] - frames[:, b]
            )
            for axis in (0, 1):
                load[:, axis] += np.bincount(triangles[:, a], target[:, axis], n)
                load[:, axis] -= np.bincount(triangles[:, b], target[:, axis], n)
        load = load[1:] - pinned * flat[0]
        flat[1:] = np.column_stack([solve(load[:, 0]), solve(load[:, 1])])
    return flat


# ----------------------------------------------------------------------- charting


def _rasterise(
    flat: np.ndarray,
    triangles: np.ndarray,
    origin: np.ndarray,
    spacing_km: float,
    shape: tuple[int, int],
) -> np.ndarray:
    """The triangle covering each point of a lattice, and -1 where none does.

    Point ``(i, j)`` of the lattice is at ``origin + spacing_km * (i, j)``. Each
    triangle tests the points inside its bounding box, in blocks of
    :data:`_RASTER_BLOCK` candidates.
    """
    corners = flat[triangles]
    edges = np.stack([corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]], -1)
    inverse = np.linalg.inv(edges)
    low = np.ceil((corners.min(axis=1) - origin) / spacing_km - _ROUNDING).astype(
        np.int64
    )
    high = np.floor((corners.max(axis=1) - origin) / spacing_km + _ROUNDING).astype(
        np.int64
    )
    low = np.maximum(low, 0)
    high = np.minimum(high, np.array(shape) - 1)
    extent = np.maximum(high - low + 1, 0)
    counts = extent.prod(axis=1)
    cuts = np.searchsorted(
        np.cumsum(counts),
        np.arange(1, -(-counts.sum() // _RASTER_BLOCK)) * _RASTER_BLOCK,
    )

    covering = np.full(shape, -1, dtype=np.int64)
    for block in np.split(np.arange(len(triangles)), cuts):
        owner = np.repeat(block, counts[block])
        offset = np.arange(owner.size) - np.repeat(
            np.cumsum(counts[block]) - counts[block], counts[block]
        )
        i = low[owner, 0] + offset // extent[owner, 1]
        j = low[owner, 1] + offset % extent[owner, 1]
        point = origin + spacing_km * np.column_stack([i, j]) - corners[owner, 0]
        weights = np.einsum("tab,tb->ta", inverse[owner], point)
        inside = (weights >= -_ROUNDING).all(axis=1) & (
            weights.sum(axis=1) <= 1.0 + _ROUNDING
        )
        covering[i[inside], j[inside]] = owner[inside]
    return covering


def surface_chart(
    vertices_km: np.ndarray, triangles: np.ndarray, spacing_km: float
) -> Geometry:
    """Chart a triangulated fault on a regular grid laid over the unrolled mesh.

    :func:`flatten` unrolls the mesh, then:

    - *Check.* It refuses a folded triangle, or a stretch beyond
      :data:`MAXIMUM_STRETCH` over the most stretched twentieth of the area.
    - *Orient.* The flat sheet turns so that depth increases along axis 0, by the
      least-squares gradient of depth over the flat coordinates, and reflects if
      needed so that the fault dips to the right of axis 1. Row ``i`` then runs down
      dip and column ``j`` along strike, as on every other chart.
    - *Grid.* A lattice of ``spacing_km`` covers the flat sheet's bounding box,
      centred on it. Each node takes the triangle it falls in, or the triangle of the
      nearest node that falls in one, and maps back onto the mesh by that
      triangle's affine map, extended past the triangle's edges where the node lies
      outside it. A cell is ``occupied`` when its centre falls on the mesh.

    Parameters
    ----------
    vertices_km : np.ndarray
        Vertex positions, ``(n, 3)``, east, north and depth in kilometres with depth
        positive down.
    triangles : np.ndarray
        Corner indices, ``(m, 3)``, of one connected sheet wound consistently, as
        :func:`read_tsurf` returns.
    spacing_km : float
        The cell size along both axes of the unrolled sheet, in kilometres.

    Returns
    -------
    Geometry
        The chart, one SRF plane, ``plane_cells == (n_j,)``. A cell off the mesh
        has ``occupied`` false, and every node is finite.

    Raises
    ------
    RuptureGeneratorError
        For a spacing that fails to be a positive length, a level mesh, a flattening
        that folds a triangle or stretches more than :data:`MAXIMUM_STRETCH`, or a
        spacing so coarse that no node or no cell centre falls on the mesh.
    """
    if not spacing_km > 0.0 or not np.isfinite(spacing_km):
        raise RuptureGeneratorError(
            f"a cell size of {spacing_km} km is not a positive length"
        )
    vertices_km = np.asarray(vertices_km, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    flat = flatten(vertices_km, triangles)

    # Check, on the sheet as flatten returns it, wound anticlockwise.
    frames, areas_km2 = _local_frames(vertices_km, triangles)
    jacobians = _jacobians(frames, flat, triangles)
    folded = int(np.count_nonzero(np.linalg.det(jacobians) <= 0.0))
    if folded:
        raise RuptureGeneratorError(
            f"flattening folds {folded} of the mesh's {len(triangles)} triangles "
            "over the rest"
        )
    stretch = np.abs(np.log(np.linalg.svd(jacobians, compute_uv=False))).max(axis=1)
    order = np.argsort(stretch)
    cumulative = np.cumsum(areas_km2[order])
    percentile = float(
        stretch[order][np.searchsorted(cumulative, 0.95 * cumulative[-1])]
    )
    if percentile > MAXIMUM_STRETCH:
        raise RuptureGeneratorError(
            f"flattening stretches the mesh by {percentile:.1%} at the 95th "
            f"percentile of area, beyond the {MAXIMUM_STRETCH:.0%} a chart may have"
        )

    # Orient: the rows of ``slope`` are the 3D rates of change along each flat axis.
    centred = flat - flat.mean(axis=0)
    design = np.column_stack([np.ones(len(flat)), centred])
    slope = np.linalg.lstsq(design, vertices_km, rcond=None)[0][1:]
    gradient = slope[:, 2]
    if np.hypot(*gradient) < _LEVEL_GRADIENT:
        raise RuptureGeneratorError(
            "the mesh is level, so it has no down-dip direction to chart it by"
        )
    down = gradient / np.hypot(*gradient)
    along = np.array([-down[1], down[0]])
    dip, strike = down @ slope, along @ slope
    if strike[0] * dip[1] - strike[1] * dip[0] > 0.0:
        along = -along
    flat = centred @ np.column_stack([down, along])

    # Grid.
    low, high = flat.min(axis=0), flat.max(axis=0)
    cells = np.maximum(
        np.ceil((high - low) / spacing_km - _ROUNDING).astype(np.int64), 1
    )
    origin = 0.5 * (low + high) - 0.5 * spacing_km * cells
    n_i, n_j = (int(count) for count in cells)
    covering = _rasterise(flat, triangles, origin, spacing_km, (n_i + 1, n_j + 1))
    occupied = (
        _rasterise(flat, triangles, origin + 0.5 * spacing_km, spacing_km, (n_i, n_j))
        >= 0
    )
    if not (covering >= 0).any() or not occupied.any():
        raise RuptureGeneratorError(
            f"no node or no cell centre of a {spacing_km} km grid falls on the "
            "mesh, so the spacing is too coarse for it"
        )
    nearest = sp.ndimage.distance_transform_edt(
        covering < 0, return_distances=False, return_indices=True
    )
    owner = covering[tuple(nearest)]

    # Each triangle's affine map from the flat sheet back onto the mesh.
    corners = flat[triangles]
    edges = np.stack([corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]], -1)
    spatial = vertices_km[triangles]
    spans = np.stack([spatial[:, 1] - spatial[:, 0], spatial[:, 2] - spatial[:, 0]], -1)
    affine = spans @ np.linalg.inv(edges)
    offset = spatial[:, 0] - np.einsum("tcd,td->tc", affine, corners[:, 0])
    u = (origin[0] + spacing_km * np.arange(n_i + 1))[:, None]
    v = (origin[1] + spacing_km * np.arange(n_j + 1))[None, :]
    positions = np.stack(
        [
            offset[owner, c] + affine[owner, c, 0] * u + affine[owner, c, 1] * v
            for c in range(3)
        ],
        axis=-1,
    )
    return Geometry(nodes=positions, occupied=occupied, plane_cells=(n_j,))
