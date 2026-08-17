"""The triangular mesh type: a fault segment as a Monge patch, triangulated.

A segment is a chart ``X(u, v) = O + u e_u + v e_v + h(u, v) n`` -- a reference plane
and a normal displacement over it -- with the parameter domain triangulated and the
triangulation lifted to R^3. :class:`TriangleMesh` wraps that as an `xarray.Dataset`
and carries methods, not stored copies, for cell centres, areas, local strike and dip,
arc lengths and boundaries. Everything is plain vector arithmetic in the projected CRS,
in kilometres, with positions as offsets from a per-surface origin; the one crossing to
WGS84 belongs to :mod:`rupture_generator.mesh`.

The frame splits its two jobs. ``n`` is the best-fit plane normal, which minimises
``|grad h|`` -- the quantity bounding both the margin before the projection folds and
the metric error the parameter-plane solvers carry. ``e_u`` and ``e_v`` come from the
config's stated strike and dip, never from the SVD, whose in-plane singular vectors are
degenerate on a square patch and arbitrary in sign; they also make the covariance
separable in ``(u, v)``, Mai & Beroza's two correlation lengths being defined along
strike and down dip. :func:`implied_axes` supplies them when no config does.

The patch is a patch at all only if ``X -> (u, v)`` is injective, which on a
triangulation is exactly *every triangle is positively oriented in the parameter
plane*: :func:`check_admissible` is the refusal, :func:`fold_margin` the diagnostic.
Connectivity always comes from the surface, never from the projection.

Storing ``(u, v)`` per vertex is what makes the rest fall out: the arc lengths are the
metric factor ``sqrt(1 + |grad h|^2)`` integrated, the hypocentre seam is a
point-in-triangle query, and the lattice the solvers run on is a grid in ``(u, v)``.
True surface arc lengths sit beside them, the SRF header wanting arc length where the
covariance wants ``(u, v)``. This module does not sample, solve or taper.

References
----------
Mai, P. M., & Beroza, G. C. (2002). A spatial random field model to characterize
complexity in earthquake slip. *Journal of Geophysical Research*, 107(B11), 2308.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyproj
import xarray as xr

from rupture_generator.formats import Format, resolve
from rupture_generator.mesh import (
    SEAM_TOLERANCE_KM,
    RuptureMesh,
    fuse,
)
from rupture_generator.mesh import (
    build_fault as build_structured_fault,
)
from rupture_generator.mesh import (
    build_point as build_structured_point,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from rupture_generator.config.geometry import FaultConfig, PointConfig

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

SCHEMA_VERSION = 3
"""Bumped when a reader of an older file would get a wrong answer, not an error.

Version 3 stores a triangulation; versions 1 and 2 stored a structured
``(dip_node, strike_node)`` lattice whose connectivity was its shape.
"""

NODE_DIM = "node"
"""The dim a vertex quantity lives on."""

FACE_DIMS = ("face",)
"""The dims a stage's field lives on: one value per triangle, flat."""

CORNER_DIM = "corner"
"""The dim a face's vertex indices lie along; its size is the cell arity."""

NODE_VARIABLES = ("east_km", "north_km", "depth_km", "strike_km", "dip_km")
"""Per vertex: three positions and the two parameter coordinates. All a file stores."""

FACE_VARIABLES = ("faces", "plane_of_face")
"""The chart's own topology and provenance, per face."""

RESERVED_FIELDS = frozenset({*NODE_VARIABLES, *FACE_VARIABLES})
"""Names a stage may not attach a field under."""

RESERVED_ATTRS = frozenset(
    {
        "surface",
        "origin_east_km",
        "origin_north_km",
        "frame_origin_km",
        "strike_axis",
        "dip_axis",
        "normal",
    }
)
"""Attribute names that say what the chart *is*, rather than what a stage learned."""

DEGENERATE_MASS_FRACTION = 5.0e-6
"""How little surface a vertex may carry, against the median, before refusal.

A vertex with almost no lumped mass sits at the tip of a sliver. 5e-6 is the geometric
middle of the decade a sweep found a cliff across, and it separates on real data: over
the three NZ CFM v1.0 interfaces the worst ratio is 7.3e-07 against 2.4e-05, where
every lattice mesh built here sits at 1/6.
"""

BOUNDARY_LABELS = ("top", "bottom", "lateral")
"""What a boundary edge can be, which is what the taper needs to tell apart."""

_MINIMUM_IN_PLANE_LENGTH = float(np.sqrt(np.finfo(np.float64).eps))
"""How much of the stated strike must survive projection into the fitted plane.

An error bound, not a modelling choice: a residual of length ``L`` carries a direction
error of about ``eps / L``, so at ``L = sqrt(eps) = 1.5e-8`` the recovered strike is
uncertain by 8.5e-7 degrees, six orders inside the one-degree rake bound.
"""

_DOWN = np.array([0.0, 0.0, 1.0])


# ============================================================================
# The reference frame
# ============================================================================


def stated_axes(
    strike_deg: float, dip_deg: float, *, dips_left: bool = False
) -> tuple[FloatArray, FloatArray]:
    """The strike and down-dip unit vectors a config's numbers name.

    Components are ``(east, north, depth)`` with depth positive down, and bearings are
    degrees clockwise from grid north -- which is why the horizontal parts are
    ``(sin, cos)``. ``dip_deg`` is in ``(0, 90]``, and the two ``(3,)`` unit vectors
    returned are orthogonal at every dip.
    """
    strike = np.radians(strike_deg)
    dip = np.radians(dip_deg)
    down_dip_azimuth = np.radians(strike_deg + (-90.0 if dips_left else 90.0))
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


def implied_axes(points_km: FloatArray) -> tuple[float, float, bool]:
    """The strike and dip a surface implies when no config states them.

    Fit the plane through the ``(n, 3)`` positions and read the strike and dip off its
    steepest descent direction -- the geologist's definition, unique up to the
    convention that a plane dips to the right of its strike, so the returned
    ``dips_left`` is always false. It uses only the fitted normal crossed with the
    vertical, never the SVD's in-plane axes, so it does not depend on the sampling.

    Raises
    ------
    ValueError
        If the fitted plane is horizontal, where a strike is not defined at all.
    """
    points = np.asarray(points_km, dtype=np.float64)
    _, _, rotation = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    normal = rotation[2]

    # Steepest descent in the plane: straight down, less its out-of-plane part.
    down_dip = _DOWN - float(_DOWN @ normal) * normal
    length = float(np.linalg.norm(down_dip))
    if length <= _MINIMUM_IN_PLANE_LENGTH:
        raise ValueError(
            "the best-fit plane of these points is horizontal, so it has no strike "
            "and no dip direction. State them, or check the depths are not constant"
        )
    down_dip = down_dip / length
    return (
        float(np.degrees(np.arctan2(down_dip[0], down_dip[1])) % 360.0 - 90.0) % 360.0,
        float(np.degrees(np.arcsin(np.clip(down_dip[2], -1.0, 1.0)))),
        False,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class MongeFrame:
    """The reference plane a segment is a displacement over.

    Four vectors in the projected CRS, kilometres, components ``(east, north, depth)``:
    an origin on the plane and an orthonormal triple with ``e_u x e_v = n``, so a
    positively oriented triangle in ``(u, v)`` is one the projection has not folded.
    """

    origin_km: FloatArray
    """Where ``(u, v, h) = (0, 0, 0)`` sits, as an offset from the surface origin."""

    strike_axis: FloatArray
    """``e_u``. The config's stated strike, projected into the best-fit plane."""

    dip_axis: FloatArray
    """``e_v``. ``n x e_u``, which is the config's stated dip direction on a plane."""

    normal: FloatArray
    """``n``. The best-fit plane normal, signed to agree with the config's."""

    @classmethod
    def fit(
        cls,
        corners_km: FloatArray,
        *,
        strike_deg: float,
        dip_deg: float,
        dips_left: bool = False,
        weights: FloatArray | None = None,
    ) -> MongeFrame:
        """Fit a frame to a point cloud, with the axes taken from the config.

        The normal is the smallest right singular vector of the centred ``(n, 3)``
        cloud, its sign flipped to agree with the config's ``cross(strike, dip)``; the
        in-plane axes come from the config alone, and ``origin_km`` comes back at the
        weighted centroid. Callers pass a quadrature of the surface
        (:func:`_surface_moment`) rather than its nodes, ``weights`` being the square
        kilometres each point carries, which makes this the exact continuous
        least-squares plane; a fit to the nodes drifts by 4e-4 relative across a factor
        of four in cell size.

        Raises
        ------
        ValueError
            If fewer than three points are given, if the weights are the wrong shape or
            sum to nothing, or if the stated strike is perpendicular to the fitted plane
            and so names no direction in the patch.
        """
        corners = np.asarray(corners_km, dtype=np.float64)
        if corners.ndim != 2 or corners.shape[1] != 3 or corners.shape[0] < 3:
            raise ValueError(
                f"a reference plane needs at least 3 points shaped (n, 3), got "
                f"{corners.shape}"
            )

        if weights is None:
            share = np.ones(len(corners), dtype=np.float64)
        else:
            share = np.asarray(weights, dtype=np.float64)
            if share.shape != (len(corners),):
                raise ValueError(
                    f"the weights are shaped {share.shape} for {len(corners)} points"
                )
            if not np.isfinite(share).all() or share.min() < 0.0 or share.sum() <= 0.0:
                raise ValueError(
                    "the weights must be finite, non-negative and sum to more than "
                    "zero; they are the surface area each node carries"
                )

        centroid = (share[:, None] * corners).sum(axis=0) / share.sum()
        # sqrt on the rows turns an unweighted SVD into a weighted least squares.
        _, _, rotation = np.linalg.svd(
            np.sqrt(share)[:, None] * (corners - centroid), full_matrices=False
        )
        normal = rotation[2]

        stated_strike, stated_dip = stated_axes(
            strike_deg, dip_deg, dips_left=dips_left
        )
        if float(normal @ np.cross(stated_strike, stated_dip)) < 0.0:
            normal = -normal

        strike_axis = stated_strike - float(stated_strike @ normal) * normal
        length = float(np.linalg.norm(strike_axis))
        if length <= _MINIMUM_IN_PLANE_LENGTH:
            raise ValueError(
                f"the stated strike of {strike_deg} degrees lies within "
                f"{np.degrees(np.arcsin(length)):.3g} degrees of the normal to the "
                "best-fit plane of the geometry, so it names no direction in the "
                "patch. The stated strike and dip and the node positions are "
                "describing different surfaces"
            )
        strike_axis = strike_axis / length
        return cls(
            origin_km=centroid,
            strike_axis=strike_axis,
            dip_axis=np.cross(normal, strike_axis),
            normal=normal,
        )

    def translated(self, strike_km: float, dip_km: float) -> MongeFrame:
        """The same plane with its origin moved within it, by ``(strike_km, dip_km)``.

        Moving it within the plane leaves ``h`` untouched, which is what lets the
        builder put ``(u, v) = (0, 0)`` at the patch's shallow near corner.
        """
        return dataclasses.replace(
            self,
            origin_km=self.origin_km
            + strike_km * self.strike_axis
            + dip_km * self.dip_axis,
        )

    def project(self, points_km: FloatArray) -> FloatArray:
        """Points as ``(u, v, h)``: components along ``e_u``, ``e_v`` and ``n``."""
        basis = np.stack([self.strike_axis, self.dip_axis, self.normal], axis=1)
        return (np.asarray(points_km, dtype=np.float64) - self.origin_km) @ basis

    def lift(self, parameters_km: FloatArray) -> FloatArray:
        """The inverse of :meth:`project`: ``(u, v, h)`` back to a position."""
        basis = np.stack([self.strike_axis, self.dip_axis, self.normal], axis=0)
        return self.origin_km + np.asarray(parameters_km, dtype=np.float64) @ basis

    @property
    def strike_deg(self) -> float:
        """The frame's own strike: ``e_u``'s grid-north bearing, in ``[0, 360)``."""
        return float(
            np.degrees(np.arctan2(self.strike_axis[0], self.strike_axis[1])) % 360.0
        )

    @property
    def dip_deg(self) -> float:
        """The frame's own dip: how far ``e_v`` plunges below horizontal, in degrees."""
        return float(np.degrees(np.arcsin(np.clip(self.dip_axis[2], -1.0, 1.0))))

    def to_attrs(self) -> dict[str, FloatArray]:
        """The frame as four ``(3,)`` arrays, keyed by :data:`RESERVED_ATTRS` names."""
        return {
            "frame_origin_km": np.asarray(self.origin_km, dtype=np.float64),
            "strike_axis": np.asarray(self.strike_axis, dtype=np.float64),
            "dip_axis": np.asarray(self.dip_axis, dtype=np.float64),
            "normal": np.asarray(self.normal, dtype=np.float64),
        }

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any]) -> MongeFrame:
        """Read a frame back out of what :meth:`to_attrs` wrote.

        Never refitted: a refit would move the chart under its own coordinates.
        """
        return cls(
            origin_km=np.asarray(attrs["frame_origin_km"], dtype=np.float64),
            strike_axis=np.asarray(attrs["strike_axis"], dtype=np.float64),
            dip_axis=np.asarray(attrs["dip_axis"], dtype=np.float64),
            normal=np.asarray(attrs["normal"], dtype=np.float64),
        )


# ============================================================================
# The mesh type
# ============================================================================


class TriangleMesh:
    """One rupture geometry expressed as a triangulated Monge patch."""

    _dataset: xr.Dataset

    def __init__(self, dataset: xr.Dataset) -> None:
        """Wrap a dataset carrying :data:`NODE_VARIABLES`, :data:`FACE_VARIABLES` and
        the frame. Use :meth:`from_patches` instead, or read a file."""
        self._dataset = dataset

    def __repr__(self) -> str:
        """The chart's name, size and fields -- not the dataset behind it."""
        fields = ", ".join(sorted(self.fields())) or "none"
        return (
            f"{type(self).__name__}({self.surface!r}, {self.face_count} faces, "
            f"{self.node_count} nodes, fields: {fields})"
        )

    def _with(self, dataset: xr.Dataset) -> TriangleMesh:
        return type(self)(dataset)

    # ------------------------------------------------------------ construction

    @classmethod
    def from_patches(
        cls,
        patches: Sequence[FloatArray],
        *,
        strike_deg: float,
        dip_deg: float,
        dips_left: bool = False,
        origin_east_km: float,
        origin_north_km: float,
        surface: str,
    ) -> TriangleMesh:
        """The lattice builder: fit a frame, split every quad, lift.

        ``patches`` is one ``(n_i+1, n_j+1, 3)`` node lattice per config plane, ``i``
        down dip and ``j`` along strike, positions offset from the surface origin. The
        connectivity is read straight off the lattice, two triangles per quad split on
        the ``(i, j) -> (i+1, j+1)`` diagonal, and patches sharing nodes exactly are
        welded. Not Delaunay of the projected points: that triangulates the convex
        **hull**, and a curved surface's footprint is about 1% concave, which on the
        Williams et al. (2013) Hikurangi interface filled the notches with 66 extra
        faces carrying 6.6% of an area whose bound is exact.

        Raises
        ------
        ValueError
            For a patch that is not a lattice of at least 2x2 nodes, one carrying a
            non-finite position, or a patch that folds -- see :func:`check_admissible`.
        """
        prepared: list[FloatArray] = []
        for index, patch in enumerate(patches):
            nodes = np.asarray(patch, dtype=np.float64)
            if nodes.ndim != 3 or nodes.shape[2] != 3 or min(nodes.shape[:2]) < 2:
                raise ValueError(
                    f"patch {index} is shaped {nodes.shape}; a patch is a lattice of "
                    "at least 2x2 nodes, shaped (n_i+1, n_j+1, 3)"
                )
            if not np.isfinite(nodes).all():
                raise ValueError(f"patch {index} carries a non-finite node position")
            prepared.append(nodes)
        if not prepared:
            raise ValueError("a segment needs at least one patch")

        stacked = np.concatenate([patch.reshape(-1, 3) for patch in prepared])
        # Exact equality: a shared seam column is the same arithmetic run twice, so
        # duplicates are bitwise identical. `np.unique` sorts, so vertex order is *not*
        # lattice order -- `inverse` carries one into the other.
        vertices, inverse = np.unique(stacked, axis=0, return_inverse=True)
        welded = np.asarray(inverse).ravel()

        faces: list[IntArray] = []
        plane_of_face: list[IntArray] = []
        start = 0
        for index, patch in enumerate(prepared):
            rows, columns = patch.shape[:2]
            lattice = welded[start : start + rows * columns].reshape(rows, columns)
            start += rows * columns
            # Anticlockwise in (u, v): +u along a row, +v down a column, so the quad
            # near -> far -> opposite -> beside turns positively, and so do both its
            # triangles.
            near, far = lattice[:-1, :-1], lattice[:-1, 1:]
            opposite, beside = lattice[1:, 1:], lattice[1:, :-1]
            split = np.stack(
                [
                    np.stack([near, far, opposite], axis=-1),
                    np.stack([near, opposite, beside], axis=-1),
                ],
                axis=2,
            ).reshape(-1, 3)
            faces.append(split)
            plane_of_face.append(np.full(len(split), index, dtype=np.int64))

        connectivity = np.concatenate(faces)
        sample, share = _surface_moment(vertices, connectivity)
        frame = MongeFrame.fit(
            sample,
            strike_deg=strike_deg,
            dip_deg=dip_deg,
            dips_left=dips_left,
            weights=share,
        )
        parameters = frame.project(vertices)
        frame = frame.translated(
            float(parameters[:, 0].min()), float(parameters[:, 1].min())
        )

        mesh = cls._from_frame(
            vertices_km=vertices,
            faces=connectivity,
            plane_of_face=np.concatenate(plane_of_face),
            frame=frame,
            origin_east_km=origin_east_km,
            origin_north_km=origin_north_km,
            surface=surface,
        )
        check_admissible(mesh)
        return mesh

    @classmethod
    def from_triangulation(
        cls,
        vertices_km: FloatArray,
        faces: IntArray,
        *,
        strike_deg: float,
        dip_deg: float,
        dips_left: bool = False,
        origin_east_km: float = 0.0,
        origin_north_km: float = 0.0,
        surface: str,
        plane_of_face: IntArray | None = None,
    ) -> TriangleMesh:
        """The builder for a surface that arrives with its own faces.

        What a 3-D fault model gives, and what
        :mod:`rupture_generator.triangular.gocad` reads GOCAD TSurf files into.
        ``vertices_km`` is ``(V, 3)`` offsets from the surface origin and ``faces`` is
        ``(F, 3)`` zero-based indices, wound anticlockwise seen from the
        ``strike_deg``/``dip_deg`` side. The connectivity is kept exactly as given; only
        the global winding is normalised, being a convention rather than a fact, with
        faces disagreeing with the majority refused as folds. The frame is fitted to
        every vertex, a triangulated interface having no corners.

        Raises
        ------
        ValueError
            For vertices or faces of the wrong shape, a non-finite position, a vertex
            index out of range, or a surface that folds.
        """
        points, connectivity, frame = _fit_surface(
            vertices_km,
            faces,
            strike_deg=strike_deg,
            dip_deg=dip_deg,
            dips_left=dips_left,
        )

        if plane_of_face is None:
            plane_of_face = np.zeros(len(connectivity), dtype=np.int64)
        plane_of_face = np.asarray(plane_of_face, dtype=np.int64)
        if plane_of_face.shape != (len(connectivity),):
            raise ValueError(
                f"plane_of_face has {plane_of_face.shape[0]} entries for "
                f"{len(connectivity)} faces"
            )

        mesh = cls._from_frame(
            vertices_km=points,
            faces=connectivity,
            plane_of_face=plane_of_face,
            frame=frame,
            origin_east_km=origin_east_km,
            origin_north_km=origin_north_km,
            surface=surface,
        )
        check_admissible(mesh)
        return mesh

    @classmethod
    def _from_frame(
        cls,
        *,
        vertices_km: FloatArray,
        faces: IntArray,
        plane_of_face: IntArray,
        frame: MongeFrame,
        origin_east_km: float,
        origin_north_km: float,
        surface: str,
    ) -> TriangleMesh:
        """Project through a fitted frame; the caller runs :func:`check_admissible`."""
        return cls._assemble(
            vertices_km=vertices_km,
            parameters_km=frame.project(vertices_km)[:, :2],
            faces=faces,
            plane_of_face=plane_of_face,
            frame=frame,
            origin_east_km=origin_east_km,
            origin_north_km=origin_north_km,
            surface=surface,
        )

    @classmethod
    def _assemble(
        cls,
        *,
        vertices_km: FloatArray,
        parameters_km: FloatArray,
        faces: IntArray,
        plane_of_face: IntArray,
        frame: MongeFrame,
        origin_east_km: float,
        origin_north_km: float,
        surface: str,
    ) -> TriangleMesh:
        """Lay the arrays out as the dataset, with no geometry decided here."""
        dataset = xr.Dataset(
            data_vars={
                "east_km": (
                    NODE_DIM,
                    vertices_km[:, 0],
                    {
                        "units": "kilometres",
                        "long_name": "Easting offset from the mesh origin",
                    },
                ),
                "north_km": (
                    NODE_DIM,
                    vertices_km[:, 1],
                    {
                        "units": "kilometres",
                        "long_name": "Northing offset from the mesh origin",
                    },
                ),
                "depth_km": (
                    NODE_DIM,
                    vertices_km[:, 2],
                    {
                        "units": "kilometres",
                        "long_name": "Depth below the surface, positive downwards",
                    },
                ),
                "strike_km": (
                    NODE_DIM,
                    parameters_km[:, 0],
                    {
                        "units": "kilometres",
                        "long_name": "Parameter coordinate u, along the frame's strike",
                    },
                ),
                "dip_km": (
                    NODE_DIM,
                    parameters_km[:, 1],
                    {
                        "units": "kilometres",
                        "long_name": "Parameter coordinate v, down the frame's dip",
                    },
                ),
                "faces": (
                    (FACE_DIMS[0], CORNER_DIM),
                    faces,
                    {"long_name": "Vertex indices of each triangle"},
                ),
                "plane_of_face": (
                    FACE_DIMS,
                    plane_of_face,
                    {"long_name": "Which config plane each face came from"},
                ),
            },
            attrs={
                "surface": surface,
                "origin_east_km": float(origin_east_km),
                "origin_north_km": float(origin_north_km),
                **frame.to_attrs(),
            },
        )
        return cls(dataset)

    def with_triangulation(
        self, vertices_km: FloatArray, faces: IntArray, plane_of_face: IntArray
    ) -> TriangleMesh:
        """The same surface in the same frame, laid on a different triangulation.

        The frame, the origin and the surface name are kept exactly, so the parameter
        coordinates of every surviving vertex do not move -- the hypocentre seam, the
        taper and the solvers' lattice are all read off ``(u, v)``. No field and no
        attribute comes across. ``vertices_km`` is ``(V, 3)`` offsets from this mesh's
        origin, ``faces`` ``(F, 3)`` indices, ``plane_of_face`` ``(F,)`` provenance.

        Raises
        ------
        ValueError
            For arrays of the wrong shape, or a triangulation that folds in this frame.
        """
        vertices_km = np.asarray(vertices_km, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int64)
        plane_of_face = np.asarray(plane_of_face, dtype=np.int64)
        if vertices_km.ndim != 2 or vertices_km.shape[1] != 3:
            raise ValueError(
                f"vertices_km is shaped {vertices_km.shape}, and this wants (V, 3)"
            )
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces is shaped {faces.shape}, and this wants (F, 3)")
        if plane_of_face.shape != (faces.shape[0],):
            raise ValueError(
                f"plane_of_face has {plane_of_face.shape} entries for "
                f"{faces.shape[0]} faces"
            )

        origin_east_km, origin_north_km = self.origin_km
        mesh = TriangleMesh._from_frame(
            vertices_km=vertices_km,
            faces=faces,
            plane_of_face=plane_of_face,
            frame=self.frame,
            origin_east_km=origin_east_km,
            origin_north_km=origin_north_km,
            surface=self.surface,
        )
        check_admissible(mesh)
        return mesh

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
    def frame(self) -> MongeFrame:
        """The reference plane this chart is a normal displacement over."""
        return MongeFrame.from_attrs(self._dataset.attrs)

    @property
    def node_count(self) -> int:
        """How many vertices the triangulation has."""
        return int(self._dataset.sizes[NODE_DIM])

    @property
    def face_count(self) -> int:
        """How many triangles the triangulation has."""
        return int(self._dataset.sizes[FACE_DIMS[0]])

    @property
    def corner_count(self) -> int:
        """The cell arity, read from ``faces.shape`` rather than assumed to be three."""
        return int(self._dataset.sizes[CORNER_DIM])

    def vertices_km(self) -> FloatArray:
        """Node positions, ``(V, 3)``, (east, north, depth) km from the origin."""
        return np.stack(
            [
                self._dataset["east_km"].to_numpy(),
                self._dataset["north_km"].to_numpy(),
                self._dataset["depth_km"].to_numpy(),
            ],
            axis=-1,
        )

    def faces(self) -> IntArray:
        """Triangles as vertex indices, ``(F, 3)``, positively oriented in ``(u,v)``."""
        return self._dataset["faces"].to_numpy()

    def parameters_km(self) -> FloatArray:
        """Per-vertex ``(u, v)``, ``(V, 2)``, km, zero at the shallow near corner.

        :meth:`strike_arc_km` and :meth:`dip_arc_km` are the true surface lengths.
        """
        return np.stack(
            [
                self._dataset["strike_km"].to_numpy(),
                self._dataset["dip_km"].to_numpy(),
            ],
            axis=-1,
        )

    def planes(self) -> IntArray:
        """Which config plane each face came from, ``(F,)``, in trace order."""
        return self._dataset["plane_of_face"].to_numpy()

    # -------------------------------------------------------------- the fields

    def fields(self) -> frozenset[str]:
        """Every attached field's name: the :data:`FACE_DIMS` variables that are not
        the chart's own. Geometry is not in here; geometry is computed."""
        return frozenset(
            str(name)
            for name, variable in self._dataset.data_vars.items()
            if variable.dims == FACE_DIMS and name not in RESERVED_FIELDS
        )

    def __contains__(self, name: object) -> bool:
        """Whether a field of that name has been attached."""
        return isinstance(name, str) and name in self.fields()

    def __getitem__(self, name: str) -> FloatArray:
        """A field a stage attached, shaped ``(F,)``.

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

    def with_fields(self, **arrays: FloatArray) -> TriangleMesh:
        """This chart with more face fields on it. Functional, never in place.

        Raises
        ------
        ValueError
            For an array that is not one value per face, one carrying a non-finite
            value, or a name in :data:`RESERVED_FIELDS`. A NaN drawn here would
            otherwise reach the SRF with nothing having raised.
        """
        prepared = {}
        for name, values in arrays.items():
            if name in RESERVED_FIELDS:
                raise ValueError(
                    f"{name!r} is the chart's own, not a field to attach; "
                    f"reserved names are {', '.join(sorted(RESERVED_FIELDS))}"
                )
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (self.face_count,):
                raise ValueError(
                    f"{name} is shaped {array.shape}, and this chart has "
                    f"{self.face_count} faces, so it wants ({self.face_count},)"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{name} carries a non-finite value")
            prepared[name] = (FACE_DIMS, array)

        return self._with(self._dataset.assign(prepared))

    def without(self, *names: str) -> TriangleMesh:
        """This chart with those fields dropped; a name that is not there is fine."""
        return self._with(self._dataset.drop_vars(names, errors="ignore"))

    @property
    def attrs(self) -> Mapping[str, Any]:
        """The frame, the origin and whatever a stage recorded, as a read-only proxy."""
        return types.MappingProxyType(dict(self._dataset.attrs))

    def with_attrs(self, **values: Any) -> TriangleMesh:
        """This chart with the attributes in ``values``, scalars by convention.

        They are written straight into a file's group attributes.

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

    def with_pulses(self, offsets: IntArray, samples: FloatArray) -> TriangleMesh:
        """This chart with its slip-rate pulses attached. Functional, never in place.

        A pulse per face, each its own length, carried as CSR with the same dim names
        and checks as :meth:`~rupture_generator.mesh.RuptureMesh.with_pulses`.
        ``offsets`` is where each face's pulse starts, length ``face_count + 1``, and
        ``samples`` is every pulse concatenated.

        Raises
        ------
        ValueError
            For an indptr that is not one: the wrong length for this chart, decreasing
            anywhere, or not ending at ``samples.size``.
        """
        offsets = np.asarray(offsets, dtype=np.int64)
        samples = np.asarray(samples, dtype=np.float64)

        if offsets.shape != (self.face_count + 1,):
            raise ValueError(
                f"the pulse offsets are shaped {offsets.shape}, and this chart has "
                f"{self.face_count} faces, so it wants {self.face_count + 1} (one per "
                "face, plus the end)"
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
    def pulses(self) -> tuple[IntArray, FloatArray] | None:
        """The slip-rate pulses as ``(offsets, samples)``, or ``None`` if unset."""
        if "slip_rate" not in self._dataset:
            return None
        return (
            self._dataset["slip_rate_offset"].to_numpy(),
            self._dataset["slip_rate"].to_numpy(),
        )

    @property
    def pulse_offsets(self) -> IntArray | None:
        """Where each face's pulse starts (CSR indptr), or ``None`` if unset.

        One number per face rather than one per sample, and therefore always
        affordable: 11 MB at 1.39 M faces against the 3.2 GB of rates it indexes into.
        """
        if "slip_rate_offset" not in self._dataset:
            return None
        offsets: IntArray = self._dataset["slip_rate_offset"].to_numpy()
        return offsets

    @property
    def pulse_rates(self) -> xr.DataArray | None:
        """Every pulse concatenated, **not read**, or ``None`` if unset.

        For a reader that cannot afford them whole: at a 400 m cut one segment's rates
        are 2.45 G samples and 19.6 GB of ``f64``. Backed by whatever the dataset is
        backed by, so after :func:`read_mesh` a slice is what triggers the read.
        """
        if "slip_rate" not in self._dataset:
            return None
        return self._dataset["slip_rate"]

    # ------------------------------------------------------- derived quantities

    def centres(self) -> FloatArray:
        """Face centres, ``(F, 3)``: the mean of the three corners."""
        return self.vertices_km()[self.faces()].mean(axis=1)

    def areas_km2(self) -> FloatArray:
        """Face areas, ``(F,)``: half the cross product of two edges.

        Exactly one of the two terms
        :meth:`~rupture_generator.mesh.RuptureMesh.areas_km2` sums.
        """
        corners = self.vertices_km()[self.faces()]
        return 0.5 * np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=-1,
        )

    def parameter_areas_km2(self) -> FloatArray:
        """**Signed** face areas in the parameter plane, ``(F,)``.

        A non-positive entry is a fold, which is the whole of the admissibility test.
        """
        corners = self.parameters_km()[self.faces()]
        first = corners[:, 1] - corners[:, 0]
        second = corners[:, 2] - corners[:, 0]
        return 0.5 * (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])

    def lumped_mass_km2(self) -> FloatArray:
        """Surface area carried by each vertex, ``(V,)``, in square kilometres.

        The barycentric dual area, which sums to the total area exactly.
        :data:`DEGENERATE_MASS_FRACTION` says how small a share is too small.
        """
        return _vertex_area_km2(self.vertices_km(), self.faces())

    def face_quality(self) -> FloatArray:
        """Shape quality of each face, ``(F,)``, in ``[0, 1]``, dimensionless.

        ``4 sqrt(3) A / (a^2 + b^2 + c^2)``: one for an equilateral triangle, zero for a
        degenerate one, and 0.866 for a lattice split on its diagonal, where every mesh
        this package builds sits.
        """
        corners = self.vertices_km()[self.faces()]
        squared = sum(
            np.sum((corners[:, (index + 1) % 3] - corners[:, index]) ** 2, axis=-1)
            for index in range(3)
        )
        return np.where(
            squared > 0.0, 4.0 * np.sqrt(3.0) * self.areas_km2() / squared, 0.0
        )

    def face_normals(self) -> FloatArray:
        """Per-face unit normals, ``(F, 3)``, in the projected CRS.

        ``cross(X1 - X0, X2 - X0)`` normalised, so it agrees with the frame's own normal
        on every positively oriented face. A zero-area face gets the frame's, not a NaN.
        """
        corners = self.vertices_km()[self.faces()]
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        magnitude = np.linalg.norm(normal, axis=-1)
        degenerate = magnitude == 0.0
        unit = normal / np.where(degenerate, 1.0, magnitude)[:, None]
        return np.where(degenerate[:, None], self.frame.normal, unit)

    def slope(self) -> FloatArray:
        """Per-face ``grad h = (dh/du, dh/dv)``, ``(F, 2)``, dimensionless.

        Computed from the face normal, not from the affine map ``dX/d(u, v)``: a Monge
        patch's normal is proportional to ``(-h_u, -h_v, 1)`` in the ``(e_u, e_v, n)``
        basis, so this needs no matrix inverse, where inverting the parameter-space edge
        matrix is near-singular wherever a face is a sliver in projection. ``|grad h|``
        is ``tan`` of the angle between the face's normal and the frame's, so it is both
        the margin before the projection folds and the factor ``sqrt(1 + |grad h|^2)``
        by which true surface length exceeds parameter length.
        """
        frame = self.frame
        normal = self.face_normals()
        out_of_plane = normal @ frame.normal
        return (
            -np.stack([normal @ frame.strike_axis, normal @ frame.dip_axis], axis=-1)
            / np.where(out_of_plane == 0.0, np.finfo(np.float64).tiny, out_of_plane)[
                :, None
            ]
        )

    def maximum_slope(self) -> float:
        """The worst ``|grad h|`` on the patch. Zero on a planar patch, to round-off."""
        return float(np.linalg.norm(self.slope(), axis=-1).max(initial=0.0))

    def strike_dip_deg(self) -> tuple[FloatArray, FloatArray]:
        """Per-face strike (grid north, ``[0, 360)``) and dip (``[0, 90]``), in degrees.

        Each ``(F,)``. The surface's own strike and dip,
        ``e_strike = normalise(z x n)`` and ``e_dip = n x e_strike``, which is what the
        SRF header and the rake convention read per subfault. Both come from the face's
        normal rather than its edges -- :meth:`slope` says why the parameter-space
        affine map is not safe to read -- and a degenerate face reports dip 0 and the
        frame's strike. The strike's sign is fixed by the frame's ``e_u``, a triangle
        having no along-strike edges to orient it the way a quad does.
        """
        unit = self.face_normals()
        dip_deg = np.degrees(np.arccos(np.clip(np.abs(unit[..., 2]), 0.0, 1.0)))

        # cross(DOWN, n) is horizontal and in the face's plane: strike, up to sign.
        horizontal = np.cross(np.broadcast_to(_DOWN, unit.shape), unit)
        flat = np.linalg.norm(horizontal, axis=-1) == 0.0
        strike_axis = self.frame.strike_axis
        sign = np.where(horizontal @ strike_axis < 0.0, -1.0, 1.0)
        oriented = horizontal * sign[..., None]

        strike_deg = np.where(
            flat,
            _bearing_of(strike_axis[0], strike_axis[1]),
            _bearing_of(oriented[..., 0], oriented[..., 1]),
        )
        return strike_deg, np.where(flat, 0.0, dip_deg)

    def arc_profile(self, axis: int) -> tuple[FloatArray, FloatArray]:
        """The parameter-to-arc-length map along one axis, as a knotted polyline.

        ``S(u) = integral of M(u') du'``, where ``M`` is the area-weighted mean of
        ``sqrt(1 + h_u^2)`` over the faces the patch has at ``u'``. Averaging across dip
        rather than following one line ``v = const`` is what the consumers want -- the
        fault's extent is one number per ``u`` -- and it makes the map strictly
        increasing, so :meth:`cell_index` can invert it. ``axis`` is 0 for strike and 1
        for dip; the knots come back ascending with the arc length at each, the two
        arrays `numpy.interp` takes.
        """
        parameters = self.parameters_km()[self.faces()][..., axis]
        low, high = parameters.min(axis=1), parameters.max(axis=1)
        knots = np.unique(np.concatenate([low, high]))
        if knots.size < 2:
            return knots, np.zeros_like(knots)

        # A triangle is a tent in `axis`; flattening it to a rectangle of the same area
        # keeps both integrals below exact.
        density = self.parameter_areas_km2() / (high - low)
        metric = np.sqrt(1.0 + self.slope()[:, axis] ** 2)
        opens = np.searchsorted(knots, low)
        closes = np.searchsorted(knots, high)

        def active(weights: FloatArray) -> FloatArray:
            """Total weight of the faces spanning each interval between knots.

            A face opens at its low knot and closes at its high one, so a running sum
            gives every interval's total in one pass.
            """
            size = knots.size + 1
            events = np.asarray(
                np.bincount(opens, weights=weights, minlength=size)
                - np.bincount(closes, weights=weights, minlength=size),
                dtype=np.float64,
            )
            return np.cumsum(events)[: knots.size - 1]

        extent = active(density)
        lifted = active(density * metric)
        mean_metric = np.where(
            extent > 0.0, lifted / np.where(extent > 0.0, extent, 1.0), 1.0
        )
        return knots, np.concatenate([[0.0], np.cumsum(np.diff(knots) * mean_metric)])

    def strike_arc_km(self) -> FloatArray:
        """True surface distance along strike, per vertex, ``(V,)``.

        Kilometres, zero at the patch's near end. Not ``u``, the projected length: the
        two differ by ``sqrt(1 + h_u^2)``, which reaches 2.37 on a real interface.
        """
        knots, arc_km = self.arc_profile(0)
        return np.interp(self.parameters_km()[:, 0], knots, arc_km)

    def dip_arc_km(self) -> FloatArray:
        """True surface distance down dip, ``(V,)`` km, zero at the patch's top edge."""
        knots, arc_km = self.arc_profile(1)
        return np.interp(self.parameters_km()[:, 1], knots, arc_km)

    # ------------------------------------------------------------- the boundary

    def _edge_keys(self) -> tuple[IntArray, IntArray]:
        """Every directed edge, and the one integer that identifies it undirected.

        The canonical form is ``min * V + max``, a single int64, so deduplicating is a
        plain sort rather than ``np.unique(..., axis=0)`` over ``(3F, 2)`` rows: 15.6
        times faster on the CFM Hikurangi interface at 400 m. The largest key is
        ``(V - 1) V``, inside int64 up to three billion vertices. The edges come back
        ``(3F, 2)`` in face order, corner 0-1 of every face then 1-2 then 2-0.
        """
        faces = self.faces()
        directed = np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
        )
        low = directed.min(axis=1)
        high = directed.max(axis=1)
        return directed, low * self.node_count + high

    def _half_edges(self) -> tuple[IntArray, IntArray, IntArray]:
        """Every directed edge, its face, and how many faces share it undirected.

        ``(3F, 2)`` edges in face order, ``(3F,)`` face indices, ``(3F,)`` counts.
        """
        directed, keys = self._edge_keys()
        of_face = np.tile(np.arange(self.face_count, dtype=np.int64), 3)
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        return directed, of_face, counts[inverse.ravel()]

    def edges(self) -> IntArray:
        """Every undirected edge once, ``(E, 2)``, keyed by :meth:`_edge_keys`."""
        _, keys = self._edge_keys()
        unique = np.unique(keys)
        return np.stack([unique // self.node_count, unique % self.node_count], axis=-1)

    def boundary_edges(self, label: str | None = None) -> IntArray:
        """The edges incident to exactly one face, ``(B, 2)`` vertex index pairs.

        Directed rather than sorted: each is returned in the order its own face names
        it, so the interior lies to its left, which is what :meth:`boundary_labels`
        reads. ``label`` takes only that part of the boundary.

        Raises
        ------
        ValueError
            For a label that is not one of :data:`BOUNDARY_LABELS`.
        """
        directed, _, counts = self._half_edges()
        edges = directed[counts == 1]
        if label is None:
            return edges
        if label not in BOUNDARY_LABELS:
            raise ValueError(
                f"{label!r} is not a boundary label; they are "
                f"{', '.join(BOUNDARY_LABELS)}"
            )
        return edges[self.boundary_labels() == label]

    def boundary_faces(self, label: str | None = None) -> IntArray:
        """The faces with at least one boundary edge, ``(b,)`` indices, ascending.

        ``label``, one of :data:`BOUNDARY_LABELS`, takes only that part of the boundary.

        Raises
        ------
        ValueError
            For a label that is not one of :data:`BOUNDARY_LABELS`.
        """
        _, of_face, counts = self._half_edges()
        on_boundary = of_face[counts == 1]
        if label is None:
            return np.unique(on_boundary)
        if label not in BOUNDARY_LABELS:
            raise ValueError(
                f"{label!r} is not a boundary label; they are "
                f"{', '.join(BOUNDARY_LABELS)}"
            )
        return np.unique(on_boundary[self.boundary_labels() == label])

    def boundary_labels(self, edges: IntArray | None = None) -> np.ndarray:
        """Each boundary edge as ``top``, ``bottom`` or ``lateral``, ``(B,)`` strings.

        Read straight off the parameter coordinates: a boundary edge runs with the
        interior on its left, so its outward normal is that direction turned a right
        angle clockwise, and whichever component dominates says which boundary it is.
        Dominance rather than an angle threshold, so there is no tolerance to justify.
        ``edges`` defaults to walking :meth:`boundary_edges`, which is the whole
        half-edge pass again, and nothing checks that what is handed over is a
        boundary.
        """
        if edges is None:
            edges = self.boundary_edges()
        parameters = self.parameters_km()
        direction = parameters[edges[:, 1]] - parameters[edges[:, 0]]
        # A right angle clockwise: interior on the left means this points outward.
        outward_u, outward_v = direction[:, 1], -direction[:, 0]

        labels = np.full(len(edges), "lateral", dtype="<U7")
        along_dip = np.abs(outward_v) > np.abs(outward_u)
        labels[along_dip & (outward_v < 0.0)] = "top"
        labels[along_dip & (outward_v > 0.0)] = "bottom"
        return labels

    # ------------------------------------------------------- the hypocentre seam

    def cell_index(self, strike_km: float, dip_km: float) -> int:
        """The face containing an in-fault position, as one flat index in ``[0, F)``.

        Both arguments are true surface arc lengths from the patch's shallow near
        corner, not parameter coordinates, because that is what "12 km along strike"
        means. Each is inverted through :meth:`arc_profile`, which is strictly
        increasing, and the query is then a point-in-triangle test in the parameter
        plane. This is not the SRF's ``shyp``, measured from the along-strike centre.

        Raises
        ------
        ValueError
            For a position off the fault, naming the axis and the fault's extent. A
            position outside every face but inside the extents is accepted only if it
            misses by less than :data:`~rupture_generator.mesh.SEAM_TOLERANCE_KM`,
            which is round-off; anything further is refused rather than snapped,
            because snapping is how a hypocentre lands one cell out.
        """
        given_km = (float(strike_km), float(dip_km))
        query = np.zeros(2)
        for axis, name in ((0, "strike"), (1, "dip")):
            knots, arc_km = self.arc_profile(axis)
            extent_km = float(arc_km[-1])
            if given_km[axis] < 0.0 or given_km[axis] > extent_km:
                raise ValueError(
                    f"hypocentre: {name}_km {given_km[axis]} is off the fault, whose "
                    f"{name} extent is {extent_km:.2f} km"
                )
            # `arc_profile` is strictly increasing, so this inverts it exactly.
            query[axis] = np.interp(given_km[axis], arc_km, knots)

        corners = self.parameters_km()[self.faces()]
        first = corners[:, 1] - corners[:, 0]
        second = corners[:, 2] - corners[:, 0]
        offset = query - corners[:, 0]
        determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        alpha = (
            offset[:, 0] * second[:, 1] - offset[:, 1] * second[:, 0]
        ) / determinant
        beta = (first[:, 0] * offset[:, 1] - first[:, 1] * offset[:, 0]) / determinant
        worst = np.minimum(np.minimum(alpha, beta), 1.0 - alpha - beta)

        best = int(np.argmax(worst))
        if worst[best] >= 0.0:
            return best

        # Outside every face. Only round-off is forgiven, and in kilometres so that the
        # forgiveness is a length rather than a barycentric fraction.
        diameter_km = float(
            np.linalg.norm(
                corners[best] - np.roll(corners[best], 1, axis=0), axis=-1
            ).max()
        )
        if -float(worst[best]) * diameter_km <= SEAM_TOLERANCE_KM:
            return best
        raise ValueError(
            f"hypocentre: ({strike_km}, {dip_km}) km is inside the fault's extents but "
            f"outside every subfault, by {-float(worst[best]) * diameter_km:.4g} km. "
            "The fault is not a rectangle -- a bent segment's parameter domain has "
            "corners cut off it -- so give a position on the surface"
        )

    def cell_key(self, flat_index: int) -> int:
        """The label this chart puts on the subfault at a flat index: the index itself.

        The counterpart of :meth:`~rupture_generator.mesh.RuptureMesh.cell_key`, whose
        answer is an ``(i, j)`` pair, so that
        :func:`~rupture_generator.propagation.causal_jump` need not know which chart it
        has.
        """
        return int(flat_index)


# ============================================================================
# Admissibility
# ============================================================================


def check_admissible(mesh: TriangleMesh) -> None:
    """Assert a segment is a Monge patch, and a mesh anything can be solved on.

    Two claims. First, ``X -> (u, v)`` must be injective, or the chart describes two
    pieces of fault at one parameter point; on a triangulation that is exactly *every
    triangle is positively oriented in the parameter plane*, tested with no tolerance on
    the sign of a determinant. Second, no vertex may carry less than
    :data:`DEGENERATE_MASS_FRACTION` of the mesh's median lumped mass. Degenerate faces
    are refused rather than dropped, arriving as they do in the input file.

    Measured by ``tests/triangular/test_trimesh.py``: the shipped geometry reaches
    ``|grad h| = 0.63`` at a fold margin of 0.844, and the three NZ CFM v1.0 subduction
    interfaces reach 1.21 to 2.14 with zero inverted triangles between them. It catches
    neither patches that overlap each other in the parameter plane nor a frame that has
    stopped *meaning* strike and dip, whose signal is ``|grad h|``, reported by
    :meth:`TriangleMesh.maximum_slope` rather than refused on.

    Raises
    ------
    ValueError
        For a fold, naming the worst face and its signed parameter area; or for a
        near-degenerate mesh, naming the starved vertices and the faces around them.
    """
    signed_km2 = mesh.parameter_areas_km2()
    if signed_km2.size == 0:
        raise ValueError(f"{mesh.surface!r} has no faces, so it covers no fault")

    worst = int(np.argmin(signed_km2))
    if signed_km2[worst] <= 0.0:
        folded = int((signed_km2 <= 0.0).sum())
        raise ValueError(
            f"{mesh.surface!r}: {folded} of {signed_km2.size} faces are folded or "
            f"collapsed in the parameter plane -- face {worst} has signed area "
            f"{signed_km2[worst]:.3g} km^2, which is not positive. The surface turns "
            "far enough that it is no longer a graph over its own best-fit plane, so a "
            "parameter point names two pieces of fault. Split it into segments that "
            "each face one way, or cut the bend that doubles back"
        )

    mass_km2 = mesh.lumped_mass_km2()
    median_km2 = float(np.median(mass_km2))
    if median_km2 <= 0.0:
        raise ValueError(
            f"{mesh.surface!r}: more than half its vertices carry no area at all, so "
            "this is not a mesh anything can be solved on"
        )

    floor_km2 = DEGENERATE_MASS_FRACTION * median_km2
    starved = np.flatnonzero(mass_km2 < floor_km2)
    if not starved.size:
        return

    quality = mesh.face_quality()
    touching = np.isin(mesh.faces(), starved).any(axis=1)
    thin = np.flatnonzero(touching)[np.argsort(quality[touching])][:4]
    listed = starved[np.argsort(mass_km2[starved])][:4]

    raise ValueError(
        f"{mesh.surface!r}: {starved.size} of {mass_km2.size} vertices carry almost no "
        f"surface -- "
        + ", ".join(
            f"vertex {index} has {mass_km2[index]:.3g} km^2"
            f" ({mass_km2[index] / median_km2:.2g} of the median {median_km2:.3g})"
            for index in listed
        )
        + f" -- from {int(touching.sum())} near-degenerate faces, worst "
        + ", ".join(f"face {index} at quality {quality[index]:.3g}" for index in thin)
        + ". A vertex with no support is barely constrained by the sampler's operator, "
        "so its variance explodes and standardising the field then suppresses every "
        "healthy subfault with it -- measured at 5.3x on one real interface. Remesh the "
        "surface, or drop those faces deliberately; they are refused rather than "
        "dropped here because they came from the file and are not this package's to "
        "delete quietly"
    )


def fold_margin(mesh: TriangleMesh) -> float:
    """How much orientation the worst face has left, as a fraction of the mean.

    ``min(signed area) / mean(signed area)``: one on a patch whose triangles are all the
    same size and orientation, zero at the fold, negative past it. A margin rather than
    a pass/fail.
    """
    signed_km2 = mesh.parameter_areas_km2()
    mean = float(signed_km2.mean())
    return float(signed_km2.min()) / mean if mean != 0.0 else 0.0


# ============================================================================
# Primitives
# ============================================================================


def _bearing_of(east: FloatArray, north: FloatArray) -> FloatArray:
    """Bearings of direction vectors, normalised to ``[0, 360)``."""
    return np.degrees(np.arctan2(east, north)) % 360.0


def remesh(
    vertices_km: FloatArray,
    faces: IntArray,
    spacing_km: float,
    *,
    strike_deg: float,
    dip_deg: float,
    dips_left: bool = False,
    origin_east_km: float = 0.0,
    origin_north_km: float = 0.0,
    surface: str,
) -> TriangleMesh:
    """**Build** a well-shaped mesh on a supplied surface at a target resolution.

    A replacement for the supplied ``(V, 3)`` vertices and ``(F, 3)`` faces, not a
    refinement of them: the parameter domain is sampled on a regular lattice at
    ``spacing_km``, every node is lifted onto the source by piecewise-linear
    interpolation, and the connectivity comes from the lattice. A node is kept exactly
    when it lies inside some source face, which is both the outline test and the
    precondition for interpolating there, so the boundary is data rather than an
    alpha-shape guess, resolved to a staircase one cell deep. The source need not pass
    :func:`check_admissible` but must be a graph over its own best-fit plane, and
    ``spacing_km`` is a *request*: the extents are cut into whole cells.

    Measured on full Hikurangi: parameter area max/min is ``1.000000000011`` at 100 m,
    the minimum angle holds at 31 degrees against the CFM source's 0.018, and the area
    deficit halves with the spacing, -1.49% at 2 km to -0.075% at 100 m. Cost is 35.7 s
    and 10.8 GB at 100 m for 17.6 M vertices. Remeshing repairs ``Puyseguer``, whose
    lumped-mass ratio goes from the 7.3e-07 that
    :data:`DEGENERATE_MASS_FRACTION` refuses to 1/6.

    Raises
    ------
    ValueError
        For a source this cannot read (see :func:`_fit_surface`), a non-positive
        spacing, or a spacing so coarse that no lattice cell falls entirely inside the
        surface, which names the extents.
    """
    if not spacing_km > 0.0:
        raise ValueError(
            f"the target spacing is {spacing_km} km, which is not positive"
        )

    points, connectivity, frame = _fit_surface(
        vertices_km, faces, strike_deg=strike_deg, dip_deg=dip_deg, dips_left=dips_left
    )
    projected = frame.project(points)
    source_uv = projected[:, :2]
    source_h = projected[:, 2]

    extent_km = source_uv.max(axis=0) - source_uv.min(axis=0)
    counts = np.maximum(np.round(extent_km / spacing_km).astype(np.int64), 1)
    low = source_uv.min(axis=0)
    step = extent_km / counts
    grid_u = low[0] + np.arange(counts[0] + 1) * step[0]
    grid_v = low[1] + np.arange(counts[1] + 1) * step[1]

    height_km = np.zeros((counts[1] + 1, counts[0] + 1), dtype=np.float64)
    located = np.zeros_like(height_km, dtype=bool)

    corners = source_uv[connectivity]
    edge_one = corners[:, 1] - corners[:, 0]
    edge_two = corners[:, 2] - corners[:, 0]
    determinant = edge_one[:, 0] * edge_two[:, 1] - edge_one[:, 1] * edge_two[:, 0]
    diameter_km = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=-1).max(
        axis=1
    )

    # One pass per source face, scattering into the lattice rather than searching it
    # per face: F is thousands where the lattice is millions.
    for index in range(len(connectivity)):
        if determinant[index] == 0.0:
            continue
        # `floor` and `ceil` the wrong way round on purpose, widening the candidate
        # block by one node each side: a source face's extent is very often exactly a
        # lattice coordinate, and round-off in the tight bound drops the boundary ring.
        first_u = int(np.floor((corners[index, :, 0].min() - low[0]) / step[0]))
        last_u = int(np.ceil((corners[index, :, 0].max() - low[0]) / step[0]))
        first_v = int(np.floor((corners[index, :, 1].min() - low[1]) / step[1]))
        last_v = int(np.ceil((corners[index, :, 1].max() - low[1]) / step[1]))
        first_u, last_u = max(first_u, 0), min(last_u, counts[0])
        first_v, last_v = max(first_v, 0), min(last_v, counts[1])
        if first_u > last_u or first_v > last_v:
            continue

        block_u = grid_u[first_u : last_u + 1]
        block_v = grid_v[first_v : last_v + 1]
        offset_u = block_u[None, :] - corners[index, 0, 0]
        offset_v = block_v[:, None] - corners[index, 0, 1]
        alpha = (
            offset_u * edge_two[index, 1] - offset_v * edge_two[index, 0]
        ) / determinant[index]
        beta = (
            edge_one[index, 0] * offset_v - edge_one[index, 1] * offset_u
        ) / determinant[index]
        weight = np.stack([1.0 - alpha - beta, alpha, beta])
        # A node on a shared edge is inside both faces at the same height, so
        # overwriting is harmless; the tolerance is a length, not a fraction.
        inside = weight.min(axis=0) * diameter_km[index] >= -SEAM_TOLERANCE_KM

        target = (slice(first_v, last_v + 1), slice(first_u, last_u + 1))
        heights = np.einsum("cvu,c->vu", weight, source_h[connectivity[index]])
        height_km[target] = np.where(inside, heights, height_km[target])
        located[target] |= inside

    keep = located[:-1, :-1] & located[:-1, 1:] & located[1:, 1:] & located[1:, :-1]
    if not keep.any():
        raise ValueError(
            f"{surface!r}: no lattice cell of {step[0]:.4g} x {step[1]:.4g} km falls "
            f"entirely inside the surface, whose parameter extent is "
            f"{extent_km[0]:.4g} x {extent_km[1]:.4g} km. Ask for a finer spacing"
        )

    # Number only the nodes some kept cell uses, so the vertex table has no orphans.
    used = np.zeros_like(located)
    for row_shift, column_shift in ((0, 0), (0, 1), (1, 1), (1, 0)):
        used[
            row_shift : row_shift + keep.shape[0],
            column_shift : column_shift + keep.shape[1],
        ] |= keep
    numbering = np.full(used.shape, -1, dtype=np.int64)
    numbering[used] = np.arange(int(used.sum()))

    mesh_v, mesh_u = np.nonzero(used)
    lifted = frame.lift(
        np.stack([grid_u[mesh_u], grid_v[mesh_v], height_km[used]], axis=-1)
    )

    near = numbering[:-1, :-1][keep]
    far = numbering[:-1, 1:][keep]
    opposite = numbering[1:, 1:][keep]
    beside = numbering[1:, :-1][keep]
    connectivity = np.stack(
        [
            np.stack([near, far, opposite], axis=-1),
            np.stack([near, opposite, beside], axis=-1),
        ],
        axis=1,
    ).reshape(-1, 3)

    parameters = frame.project(lifted)
    frame = frame.translated(
        float(parameters[:, 0].min()), float(parameters[:, 1].min())
    )
    mesh = TriangleMesh._from_frame(
        vertices_km=lifted,
        faces=connectivity,
        plane_of_face=np.zeros(len(connectivity), dtype=np.int64),
        frame=frame,
        origin_east_km=origin_east_km,
        origin_north_km=origin_north_km,
        surface=surface,
    )
    check_admissible(mesh)
    return mesh


def _fit_surface(
    vertices_km: FloatArray,
    faces: IntArray,
    *,
    strike_deg: float,
    dip_deg: float,
    dips_left: bool = False,
) -> tuple[FloatArray, IntArray, MongeFrame]:
    """Validate a supplied triangulation, fit its frame, and settle its winding.

    Shared by :meth:`TriangleMesh.from_triangulation` and :func:`remesh`, which needs
    it on surfaces :func:`check_admissible` refuses. Returns the vertices as float64,
    the faces wound to agree with the frame, and the frame with its origin already at
    the parameter domain's low corner.

    Raises
    ------
    ValueError
        For vertices or faces of the wrong shape, a non-finite position, or a vertex
        index out of range.
    """
    points = np.asarray(vertices_km, dtype=np.float64)
    connectivity = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError(
            f"the vertices are shaped {points.shape}; a surface needs at least 3, "
            "shaped (V, 3)"
        )
    if not np.isfinite(points).all():
        raise ValueError("the vertices carry a non-finite position")
    if connectivity.ndim != 2 or connectivity.shape[1] != 3 or not len(connectivity):
        raise ValueError(
            f"the faces are shaped {connectivity.shape}; they are triangles, shaped "
            "(F, 3), and there has to be at least one"
        )
    if connectivity.min() < 0 or connectivity.max() >= len(points):
        raise ValueError(
            f"a face names vertex {connectivity.max()} of {len(points)}; the indices "
            "are zero-based, and GOCAD's are one-based"
        )

    sample, share = _surface_moment(points, connectivity)
    frame = MongeFrame.fit(
        sample,
        strike_deg=strike_deg,
        dip_deg=dip_deg,
        dips_left=dips_left,
        weights=share,
    )
    parameters = frame.project(points)
    frame = frame.translated(
        float(parameters[:, 0].min()), float(parameters[:, 1].min())
    )

    # Global winding is a file-format convention, not geology; whether the faces agree
    # with each other is not, so the total signed area fixes the convention and
    # `check_admissible` refuses every face that disagrees.
    corners = frame.project(points)[connectivity, :2]
    first = corners[:, 1] - corners[:, 0]
    second = corners[:, 2] - corners[:, 0]
    if float((first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]).sum()) < 0.0:
        connectivity = connectivity[:, [0, 2, 1]]
    return points, connectivity, frame


def _surface_moment(
    vertices_km: FloatArray, faces: IntArray
) -> tuple[FloatArray, FloatArray]:
    """A quadrature of the surface, for fitting a plane to it rather than to its nodes.

    The three edge midpoints of every face, weighted by a third of its area. That rule
    integrates any quadratic over a triangle exactly, and the second moment a
    least-squares fit needs is quadratic, so the fit is the exact continuous
    ``integral (x - c)(x - c)^T dA``: identical to 1e-15 across a factor of four in cell
    size, where a vertex fit moves ``|grad h|`` by 4e-4 relative.
    """
    corners = vertices_km[faces]
    area_km2 = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=-1
    )
    midpoints = np.concatenate(
        [0.5 * (corners[:, index] + corners[:, (index + 1) % 3]) for index in range(3)]
    )
    return midpoints, np.tile(area_km2 / 3.0, 3)


def _vertex_area_km2(vertices_km: FloatArray, faces: IntArray) -> FloatArray:
    """How much surface each vertex carries: the barycentric-dual area, ``(V,)``."""
    corners = vertices_km[faces]
    area_km2 = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=-1
    )
    return np.asarray(
        np.bincount(
            faces.ravel(),
            weights=np.repeat(area_km2 / 3.0, faces.shape[1]),
            minlength=len(vertices_km),
        ),
        dtype=np.float64,
    )


# ============================================================================
# Geometry config to segments
# ============================================================================


def _stated_geometry(nodes: FloatArray) -> tuple[float, float, bool]:
    """A structured chart's own stated strike, dip and dip side.

    Read back out of the geometry the config produced: the trace bearing is the top
    edge's chord and the dip is the near column's plunge, exact for a chart this package
    built.
    """
    chord = nodes[0, -1] - nodes[0, 0]
    column = nodes[-1, 0] - nodes[0, 0]
    strike_deg = float(np.degrees(np.arctan2(chord[0], chord[1])) % 360.0)
    dip_deg = float(
        np.degrees(np.arctan2(column[2], float(np.hypot(column[0], column[1]))))
    )
    dips_left = bool(chord[0] * column[1] - chord[1] * column[0] > 0.0)
    return strike_deg, dip_deg, dips_left


def from_chart(chart: RuptureMesh) -> TriangleMesh:
    """Triangulate a structured chart, one config plane at a time.

    The compatibility path for a version 2 mesh file, which holds a
    ``(dip_node, strike_node)`` lattice and nothing else. A fused chart is split back
    into its constant-plane blocks first, so ``plane_of_face`` survives the round trip,
    and the frame is the one :func:`_stated_geometry` reads off the nodes.
    """
    nodes = chart.nodes()
    strike_deg, dip_deg, dips_left = _stated_geometry(nodes)
    return TriangleMesh.from_patches(
        [nodes[:, start : stop + 1] for _plane, start, stop in chart.blocks()],
        strike_deg=strike_deg,
        dip_deg=dip_deg,
        dips_left=dips_left,
        origin_east_km=chart.origin_km[0],
        origin_north_km=chart.origin_km[1],
        surface=chart.surface,
    )


def build_fault(fault: FaultConfig, crs: pyproj.CRS) -> list[TriangleMesh]:
    """A fault's geometry as triangulated segments, one Monge patch each.

    The same geometry :func:`rupture_generator.mesh.build_fault` and
    :func:`rupture_generator.mesh.fuse` produce, retriangulated, one mesh per fused
    segment in trace order. The frame's stated strike is the segment's top-edge chord,
    first trace point to last, so on a single-plane segment it is the plane's own
    bearing exactly and on a bent one every departure from it is carried by ``h``. Dip
    and dip direction come from the segment's first plane.

    Raises
    ------
    ValueError
        For anything :func:`rupture_generator.mesh.build_fault` refuses, and for a
        segment that folds -- see :func:`check_admissible`.
    """
    segments = fuse(build_structured_fault(fault, crs))
    meshes: list[TriangleMesh] = []
    for segment in segments:
        nodes = segment.nodes()
        first_plane = fault.planes[int(segment.planes()[0])]
        chord = nodes[0, -1] - nodes[0, 0]
        meshes.append(
            TriangleMesh.from_patches(
                [
                    nodes[:, start : stop + 1]
                    for _plane, start, stop in segment.blocks()
                ],
                strike_deg=float(np.degrees(np.arctan2(chord[0], chord[1])) % 360.0),
                dip_deg=first_plane.dip_deg,
                dips_left=first_plane.dips_left,
                origin_east_km=segment.origin_km[0],
                origin_north_km=segment.origin_km[1],
                surface=segment.surface,
            )
        )
    return meshes


def build_point(point: PointConfig, crs: pyproj.CRS) -> list[TriangleMesh]:
    """A point source as two triangles: an ordinary one-quad patch."""
    chart = build_structured_point(point, crs)[0]
    return [
        TriangleMesh.from_patches(
            [chart.nodes()],
            strike_deg=point.strike_deg,
            dip_deg=point.dip_deg,
            origin_east_km=chart.origin_km[0],
            origin_north_km=chart.origin_km[1],
            surface=chart.surface,
        )
    ]


def build_surface(
    surface: FaultConfig | PointConfig, crs: pyproj.CRS
) -> list[TriangleMesh]:
    """Discretise one surface into segments: the dispatch a mesh CLI would call."""
    from rupture_generator.config.geometry import PointConfig as _PointConfig

    if isinstance(surface, _PointConfig):
        return build_point(surface, crs)
    return build_fault(surface, crs)


# ============================================================================
# The file format
# ============================================================================


def to_datatree(
    meshes: Mapping[str, list[TriangleMesh]],
    crs: pyproj.CRS,
    *,
    attrs: Mapping[str, Any] | None = None,
) -> xr.DataTree:
    """Lay segments out as a tree of ``/<surface>/segment_<n>`` groups.

    Only the geometry is stored -- vertices, faces, the parameter coordinates and the
    frame -- everything else being a function of those. ``meshes`` maps a surface name
    to its segments in trace order, and the CRS is stored once in the root.
    """
    groups: dict[str, xr.Dataset] = {}
    origins: dict[str, list[float]] = {}

    for name, segments in meshes.items():
        origins[name] = list(segments[0].origin_km)
        for index, segment in enumerate(segments):
            dataset = segment._dataset.copy()
            dataset.attrs = {**dataset.attrs, "surface": name, "segment": index}
            groups[f"{name}/segment_{index}"] = dataset

    tree = xr.DataTree.from_dict(groups)
    tree.attrs = {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "crs": crs.to_string(),
        # One origin per surface, as JSON: an attribute is a scalar or an array.
        "origins": json.dumps(origins),
        **dict(attrs or {}),
    }
    return tree


def from_datatree(
    tree: xr.DataTree,
) -> tuple[dict[str, list[TriangleMesh]], pyproj.CRS]:
    """Rebuild segments from a tree, of either schema.

    A version 3 file is read as it was written, connectivity and all. A version 1 or 2
    file holds a structured lattice per config plane, so it is read through
    :func:`rupture_generator.formats.mesh.from_datatree` and triangulated by
    :func:`from_chart` on the way in. Returns the surfaces mapped to their segments,
    and the CRS.

    Raises
    ------
    ValueError
        If the tree carries no CRS, a surface has no recorded origin, or a surface's
        segments are numbered with a gap, which means one is missing.
    """
    version = int(tree.attrs.get("schema_version", 1))
    if version < SCHEMA_VERSION:
        # Local import so `triangular.mesh` can be imported without the older format.
        from rupture_generator.formats.mesh import from_datatree as read_structured

        charts, crs = read_structured(tree)
        return {
            name: [from_chart(chart) for chart in surface_charts]
            for name, surface_charts in charts.items()
        }, crs

    crs_name = tree.attrs.get("crs")
    if crs_name is None:
        raise ValueError("the file has no crs attribute, so its positions mean nothing")
    origins = json.loads(tree.attrs.get("origins", "{}"))

    # Keyed by the *stored* segment index, not by the order the groups come back in:
    # Zarr does not preserve order and HDF5 does.
    by_surface: dict[str, dict[int, xr.Dataset]] = {}
    for path, node in tree.subtree_with_keys:
        if not node.has_data or "faces" not in node.dataset:
            continue
        dataset = node.dataset
        surface = node.attrs.get("surface") or Path(path).parent.name
        index = int(node.attrs["segment"])
        segments = by_surface.setdefault(surface, {})
        if index in segments:
            raise ValueError(f"{surface!r} has two segments numbered {index}")
        segments[index] = dataset

    meshes: dict[str, list[TriangleMesh]] = {}
    for surface, segments in by_surface.items():
        if surface not in origins:
            raise ValueError(f"{surface!r} has no origin, so its offsets mean nothing")
        expected = set(range(len(segments)))
        if set(segments) != expected:
            raise ValueError(
                f"{surface!r} has segments {sorted(segments)}, expected "
                f"{sorted(expected)} -- a gap means a segment is missing rather than "
                "renumbered"
            )
        easting_km, northing_km = origins[surface]
        rebuilt = []
        for index in sorted(segments):
            dataset = segments[index].copy()
            dataset.attrs = {
                **dataset.attrs,
                "surface": surface,
                "origin_east_km": float(easting_km),
                "origin_north_km": float(northing_km),
            }
            rebuilt.append(TriangleMesh(dataset))
        meshes[surface] = rebuilt

    return meshes, pyproj.CRS(crs_name)


def write_mesh(
    meshes: Mapping[str, list[TriangleMesh]],
    crs: pyproj.CRS,
    path: Path | str,
    *,
    format: Format = Format.INFERRED,
    attrs: Mapping[str, Any] | None = None,
) -> None:
    """Write segments to an HDF5 file or a Zarr store.

    The layout is inferred from the extension unless ``format`` says otherwise.

    Raises
    ------
    ValueError
        If the format is not one a mesh can be written in: an SRF holds a rupture, not
        a surface.
    """
    path = Path(path)
    chosen = resolve(path, format)
    tree = to_datatree(meshes, crs, attrs=attrs)

    match chosen:
        case Format.NETCDF:
            tree.to_netcdf(path, engine="h5netcdf", mode="w")
        case Format.ZARR:
            tree.to_zarr(path, mode="w", consolidated=False)
        case _:
            raise ValueError(
                f"a mesh cannot be written as {chosen.value}: it holds a fault "
                "surface, not a rupture. Use .h5 or .zarr"
            )


def read_mesh(
    path: Path | str, *, format: Format = Format.INFERRED
) -> tuple[dict[str, list[TriangleMesh]], pyproj.CRS]:
    """Read segments back, from a version 3 file or a version 2 one.

    The layout is inferred from the extension unless ``format`` says otherwise. Returns
    the surface names mapped to their segments, and the CRS.

    Raises
    ------
    ValueError
        If the format is not one a mesh is written in.
    """
    path = Path(path)
    chosen = resolve(path, format)

    match chosen:
        case Format.NETCDF:
            tree = xr.open_datatree(path, engine="h5netcdf")
        case Format.ZARR:
            tree = xr.open_datatree(path, engine="zarr", consolidated=False)
        case _:
            raise ValueError(f"a mesh is not read from {chosen.value}")

    with tree:
        return from_datatree(tree)


__all__ = [
    "BOUNDARY_LABELS",
    "DEGENERATE_MASS_FRACTION",
    "SCHEMA_VERSION",
    "MongeFrame",
    "TriangleMesh",
    "build_fault",
    "build_point",
    "build_surface",
    "check_admissible",
    "fold_margin",
    "from_chart",
    "from_datatree",
    "implied_axes",
    "read_mesh",
    "remesh",
    "stated_axes",
    "to_datatree",
    "write_mesh",
]
