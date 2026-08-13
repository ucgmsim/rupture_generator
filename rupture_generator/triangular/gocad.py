"""Read a GOCAD TSurf: the format a 3-D fault model actually ships in.

The NZ Community Fault Model v1.0 distributes its subduction interfaces as GOCAD TSurf
files, and they are the reason :class:`~rupture_generator.triangular.mesh.TriangleMesh`
exists: a Hikurangi interface is 5218 vertices and 9236 triangles with an irregular
outline, which no quad lattice expresses and which
:meth:`~rupture_generator.triangular.mesh.TriangleMesh.from_patches` cannot represent.

**These files carry their own connectivity**, and that is the whole point of reading
them rather than their vertices. A triangulation derived from the projected points --
what `scipy.spatial.Delaunay` gives -- is positively oriented by construction, so
testing it for folds is a tautology; the surface's own triangles are a real test with a
real answer.

The format, as much of it as this reads:

- ``VRTX id x y z`` (or ``PVRTX``, which adds per-vertex properties this ignores) --
  positions in the CRS named by the file's own header, in **metres**.
- ``ZPOSITIVE Elevation`` -- so ``z`` is height and depth is ``-z``. The alternative,
  ``ZPOSITIVE Depth``, is refused rather than guessed, because the two differ by a sign
  on every vertex and nothing downstream would notice.
- ``TRGL i j k`` -- **one-based** vertex indices, which is the mistake this module
  exists to make once rather than at every call site.
- ``TFACE`` -- a part boundary. One file may hold several connected surfaces sharing one
  vertex numbering; each becomes its own patch, and
  :func:`read_surfaces` hands them back separately because two disconnected sheets are
  two Monge patches, not one.

What it deliberately does not do: no CRS is read or checked. A TSurf's header names its
coordinate system in a vocabulary that is not EPSG, and the CFM's files say
``NAME Default``. The caller states the CRS, the same way a geometry config does.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rupture_generator.triangular.mesh import TriangleMesh, implied_axes
from rupture_generator.units import M_PER_KM

if TYPE_CHECKING:
    from collections.abc import Iterator

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

_NAME = re.compile(r"^name\s*:\s*(.+)$", re.IGNORECASE)


def _lines(path: Path) -> Iterator[str]:
    """Every line of a TSurf, transparently through gzip.

    The CFM archive is 356 KB gzipped against 1.4 MB plain, and the format is line
    oriented, so there is no reason to hold either in memory twice.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        yield from handle


class TSurf:
    """One TSurf file's vertices, its parts, and the name it calls itself.

    Not a mesh: this is the file's contents, in kilometres and depth-positive-down, with
    no frame fitted and no admissibility claimed. :meth:`to_mesh` is the step that turns
    a part into a Monge patch, and it is separate because that step needs a stated
    strike and dip the file does not carry.
    """

    def __init__(
        self, name: str, vertices_km: FloatArray, parts: list[IntArray]
    ) -> None:
        """Hold what :func:`read_tsurf` parsed.

        Parameters
        ----------
        name : str
            What the file's header calls the surface.
        vertices_km : FloatArray
            ``(V, 3)`` positions, components ``(east, north, depth)``, depth positive
            down, kilometres, **absolute** in the file's CRS.
        parts : list of IntArray
            One ``(F, 3)`` zero-based face table per ``TFACE``.
        """
        self.name = name
        self.vertices_km = vertices_km
        self.parts = parts

    def __repr__(self) -> str:
        """The surface's name and shape, not its arrays."""
        faces = ", ".join(str(len(part)) for part in self.parts)
        return (
            f"TSurf({self.name!r}, {len(self.vertices_km)} vertices, "
            f"parts of {faces} faces)"
        )

    def to_mesh(
        self,
        part: int = 0,
        *,
        strike_deg: float | None = None,
        dip_deg: float | None = None,
        dips_left: bool = False,
        surface: str | None = None,
    ) -> TriangleMesh:
        """One part as a Monge patch, with its faces kept exactly as the file wrote them.

        Positions become **offsets from the part's own origin**, taken as the minimum
        easting and northing over its vertices. That is not cosmetic: an NZTM northing
        reaches ~5,180 km against a ~9 km triangle, and
        :mod:`rupture_generator.mesh` measures the rounding that costs at a factor of
        ~400.

        Parameters
        ----------
        part : int, optional
            Which ``TFACE`` to take.
        strike_deg, dip_deg : float, optional
            The stated geometry. A TSurf carries none, so both default to what
            :func:`~rupture_generator.triangular.mesh.implied_axes` reads off the part's
            own best-fit plane -- the geologist's strike and dip of that plane, from the
            fitted normal and the vertical, never from the SVD's in-plane axes.
        dips_left : bool, optional
            Whether the surface dips left of the strike direction. Ignored unless
            ``strike_deg`` is given, since the implied axes fix the sign by convention.
        surface : str, optional
            A name for the mesh. Defaults to the file's own, with the part appended when
            there is more than one.

        Returns
        -------
        TriangleMesh
            Admissible, or the call raises.

        Raises
        ------
        ValueError
            For a part index the file does not have, or a surface that folds.
        """
        if not 0 <= part < len(self.parts):
            raise ValueError(
                f"{self.name!r} has {len(self.parts)} part(s), so there is no part "
                f"{part}"
            )
        faces = self.parts[part]
        used = np.unique(faces)
        renumber = np.full(len(self.vertices_km), -1, dtype=np.int64)
        renumber[used] = np.arange(len(used))
        points = self.vertices_km[used].copy()

        origin_east_km = float(points[:, 0].min())
        origin_north_km = float(points[:, 1].min())
        points[:, 0] -= origin_east_km
        points[:, 1] -= origin_north_km

        if strike_deg is None or dip_deg is None:
            strike_deg, dip_deg, dips_left = implied_axes(points)

        if surface is None:
            surface = self.name if len(self.parts) == 1 else f"{self.name}_{part}"
        return TriangleMesh.from_triangulation(
            points,
            renumber[faces],
            strike_deg=strike_deg,
            dip_deg=dip_deg,
            dips_left=dips_left,
            origin_east_km=origin_east_km,
            origin_north_km=origin_north_km,
            surface=surface,
        )


def read_tsurf(path: Path | str) -> TSurf:
    """Parse a GOCAD TSurf, gzipped or not.

    Parameters
    ----------
    path : Path or str
        The ``.ts`` or ``.ts.gz`` file.

    Returns
    -------
    TSurf
        Vertices in kilometres with depth positive down, and one face table per
        ``TFACE``.

    Raises
    ------
    ValueError
        If the file declares ``ZPOSITIVE Depth`` -- this reader assumes elevation and
        negates, and getting that backwards mirrors the surface through sea level
        without changing anything a shape check would notice -- if it holds no
        triangles, or if a ``TRGL`` names a vertex the file never defined.
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
                raise ValueError(
                    f"{path.name}: a TRGL names vertex {error.args[0]}, which no VRTX "
                    "defines. GOCAD's indices are one-based and this file's numbering "
                    "has a gap"
                ) from error
        elif head == "TFACE":
            parts.append([])
        elif head == "ZPOSITIVE" and token[1].upper() != "ELEVATION":
            raise ValueError(
                f"{path.name} declares ZPOSITIVE {token[1]}; this reader assumes "
                "Elevation and negates it to get depth. Reading a Depth file as an "
                "Elevation one mirrors the surface through sea level, which nothing "
                "downstream would notice"
            )
        elif head == "NAME" and len(token) > 1 and not positions:
            pass
        else:
            match = _NAME.match(line.strip())
            if match and not positions:
                name = match.group(1).strip()

    populated = [np.array(part, dtype=np.int64) for part in parts if part]
    if not populated:
        raise ValueError(f"{path.name} holds no TRGL records, so it is not a surface")

    vertices_km = np.array(positions, dtype=np.float64)
    vertices_km[:, :2] /= M_PER_KM
    # ZPOSITIVE Elevation: z is height above sea level, and this package's third
    # component is depth below it.
    vertices_km[:, 2] /= -M_PER_KM
    return TSurf(name, vertices_km, populated)


def read_surfaces(
    path: Path | str,
    *,
    strike_deg: float | None = None,
    dip_deg: float | None = None,
    dips_left: bool = False,
) -> list[TriangleMesh]:
    """Every part of a TSurf as its own Monge patch.

    Two disconnected sheets are two patches, not one: a single best-fit plane through
    both would describe neither, and their parameter domains would overlap, which is
    the one thing :func:`~rupture_generator.triangular.mesh.check_admissible` cannot
    see.

    Parameters
    ----------
    path : Path or str
        The ``.ts`` or ``.ts.gz`` file.
    strike_deg, dip_deg : float, optional
        The stated geometry, applied to every part. Left unset, each part reads its own
        off its own best-fit plane, which is usually what is wanted for parts that are
        separate surfaces.
    dips_left : bool, optional
        Whether the surfaces dip left of the strike direction.

    Returns
    -------
    list of TriangleMesh
        One per ``TFACE``, in file order.
    """
    surface = read_tsurf(path)
    return [
        surface.to_mesh(
            part, strike_deg=strike_deg, dip_deg=dip_deg, dips_left=dips_left
        )
        for part in range(len(surface.parts))
    ]


__all__ = ["TSurf", "read_surfaces", "read_tsurf"]
