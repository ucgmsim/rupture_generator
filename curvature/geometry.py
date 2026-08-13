"""The mesh pair the whole experiment rests on: one surface, two geometries.

The comparison being made is *not* "a curved mesh against a flat mesh". It is one
mesh, twice: identical faces, identical per-vertex parameter coordinates ``(u, v)``,
and vertices that differ only in whether the Monge patch's normal displacement is
carried or dropped.

.. code-block:: text

    curved vertex:  X = O + u e_u + v e_v + h n     the real interface
    flat vertex:    X = O + u e_u + v e_v           the best-fit plane

Everything downstream follows from the two properties that construction has. The
**vertex count is the same**, so a generator seeded the same way hands both models
bit-identical white noise and every difference in the drawn fields is geometry rather
than RNG (:func:`same_white_noise`). And the **face indices are the same**, so face
``k`` of one model is face ``k`` of the other and every field can be differenced
pointwise without an interpolation step that would blur the thing being measured.

**The plane is the best-fit plane, which is generous to the literature.** Papers that
generate a rupture on a plane and project it onto a real interface rarely say how the
plane was constructed. The SVD fit :class:`~rupture_generator.triangular.mesh.MongeFrame`
uses minimises ``|grad h|``, which is the quantity that bounds every departure from
flatness measured here -- the area inflation, the path shortening, the metric factor on
the correlation length. So any other plane makes the literature approach look worse than
this experiment does.

**The mesh is built and then subdivided, not refined from the CFM triangulation.** The
CFM Hikurangi mesh has an area max/min of 4.28e+04, and the multigrid sampler's cost is
set by element shape: refining it costs 312 V-cycle iterations against 12, and produces
a field whose variance outliers ``sampling.standardise`` then divides the healthy part
of the slip distribution by. :func:`~rupture_generator.triangular.mesh.remesh` builds a
well-shaped mesh at a coarse spacing and ``spde.subdivided`` refines *that*, where
one-to-four subdivision's shape preservation is a virtue instead of a liability.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pyproj
from scipy import sparse

from rupture_generator.triangular import gocad
from rupture_generator.triangular.mesh import (
    MongeFrame,
    TriangleMesh,
    implied_axes,
    remesh,
)
from rupture_generator.triangular.spde import subdivided

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

HIKURANGI = Path("examples/cfm/Hikurangi.ts.gz")
"""The NZ CFM v1.0 Hikurangi subduction interface, as shipped."""

PUYSEGUR_FIORDLAND = Path("examples/cfm/Puysegur_Fiordland.ts.gz")
"""The Puysegur interface continued north beneath Fiordland: **the interface proper**.

The CFM ships two surfaces over this trench and they are the same surface where they
overlap -- 86% of one's vertices lie within 2 km of the other, and the median
nearest-vertex distance is zero. What separates them is where they stop.
:data:`PUYSEGUR` ends 29 km further south, and it ends **mid-slab**: its northern edge
sits at 24 km depth, which is not a boundary the interface has. This one runs on to
5.1 km depth beneath Fiordland, so its northern edge is the trench rather than a cut,
and it carries the segment that hosted the 2009 Mw 7.8 Dusky Sound interface event.

An interface truncated across dip would put a free boundary in the middle of the
seismogenic zone, and the study's taper and its correlated field both read the boundary.
"""

PUYSEGUR = Path("examples/cfm/Puyseguer.ts.gz")
"""The Puysegur interface alone -- the CFM's spelling, and its own file.

Kept and run because it is the **more curved** of the two (``|grad h|`` p90 0.653
against 0.563 on the built mesh) and so the stronger test of the thing this study
measures, and because it is the surface whose shipped triangulation
:func:`~rupture_generator.triangular.spde._refuse_starved_vertices` refuses: its closest
pair of vertices is 0.15 m apart, which leaves a vertex with 7.35e-07 of the median
share of the surface against a floor of 5e-06. The refusal is correct -- that vertex's
variance runs away and standardising by the sample spread shrinks the whole slip
distribution by 5.3.

It is also irrelevant here, because
:func:`~rupture_generator.triangular.mesh.remesh` **builds** a lattice on the surface
rather than refining the triangulation. Measured on this study's own mesh: the worst
lumped-mass ratio is 0.1660, which is 1/6.02 -- the 1/6 every vertex of a well-shaped
lattice has, and the flat twin's is 1/6 to four figures because in projection all faces
are congruent. Nothing was repaired by hand; the defect is simply not inherited. That is
the case for building rather than subdividing, made by a file that cannot be subdivided
at all.
"""

NZTM = pyproj.CRS("EPSG:2193")
"""The CRS the CFM files are in. Their GOCAD header says ``NAME Default`` and nothing
else, so this is stated by the caller exactly as a geometry config states it; the
coordinate ranges (eastings 1.50e6-2.20e6 m, northings 5.29e6-5.91e6 m) identify it."""

COARSE_SPACING_KM = 2.0
"""What :func:`~rupture_generator.triangular.mesh.remesh` is asked for before subdivision.

Two one-to-four subdivisions halve the edge length twice, so this is four times the
nominal 500 m the study runs at. The number that matters is the *delivered* median edge
length, which is reported rather than assumed -- see :attr:`MeshPair.median_edge_km`.

**Why the mesh is built coarse and refined rather than built at the target.** The
multigrid hierarchy has to be *nested*, and one-to-four subdivision is what makes it so;
a second call to ``remesh`` at a quarter of the spacing produces a mesh with no
relationship to the first.

**Why 2 km rather than the 4 km that reaches the same face count.** Subdivision cannot
move the surface at all -- :func:`~rupture_generator.triangular.spde.subdivided` puts
every new vertex on a coarse face, so ``h`` stays affine and ``grad h`` and the areas are
the parent's to twelve digits. The **base** spacing is therefore the only thing that sets
the built mesh's geometric fidelity, through the boundary staircase and through triangles
that chord across the source's own kinks. Measured by :mod:`curvature.resolution`, the
true/projected area ratio a built mesh reports converges as

======  ==============  ==================
base    faces           area ratio
======  ==============  ==================
8 km    5,192           1.026248
4 km    21,394          1.027416
2 km    86,850          1.028137
1 km    349,462         1.028466
500 m   1,402,906       1.028638
======  ==============  ==================

-- halving increments, so the limit is near 1.0287. A 4 km base with three subdivisions
and a 2 km base with two reach the same 1.39 M faces and the same 611 m median edge, but
the first carries the 4 km geometry and the second the 2 km one: 0.12% against 0.05% below
the converged ratio, for the same cost. The coarsest multigrid level is 44 thousand
vertices either way, which factorises in about a second.
"""

SUBDIVISIONS = 2
"""How many one-to-four refinements the multigrid hierarchy has above the built mesh.

**500 m is chosen against the geometry's own resolution, not against the machine.**
``remesh`` lifts by piecewise-linear interpolation *on the source faces*, so wherever the
built mesh is finer than the CFM triangulation -- whose median vertex spacing is 5.6 km --
the new triangles are coplanar sub-triangles of source faces and ``|grad h|``, area and
``h`` are the source's **exactly**. The interface carries no geometric information below
about 5.6 km, so 500 m already oversamples it elevenfold and every geometric quantity in
this study is the same number 400 m would give. :mod:`curvature.resolution` measures that
rather than asserting it.
"""


@dataclasses.dataclass(frozen=True)
class MeshPair:
    """One surface expressed twice, sharing topology and parameter coordinates.

    Attributes
    ----------
    faces : IntArray
        ``(F, 3)`` vertex indices. **The same array serves both models.**
    parameters_km : FloatArray
        ``(V, 2)`` per-vertex ``(u, v)``, along strike and down dip in the frame.
        **The same array serves both models**, by construction rather than by
        coincidence: the flat vertices are the curved ones with ``h`` dropped, and
        dropping ``h`` does not move ``(u, v)``.
    curved_km, flat_km : FloatArray
        ``(V, 3)`` vertex positions, absolute in :data:`NZTM`, kilometres, depth
        positive down.
    displacement_km : FloatArray
        ``(V,)`` the ``h`` that separates them -- the signed normal distance from the
        best-fit plane to the interface, which is the whole of the difference between
        the two models.
    frame : MongeFrame
        The reference plane both are written against.
    curved_levels, flat_levels : list
        The multigrid hierarchies, coarsest first, in the
        ``(vertices_km, faces, prolongation)`` form ``spde.MaternOperator`` takes. The
        prolongations are **shared**: one-to-four subdivision's transfer is linear
        interpolation on the coarse block plus a half-and-half row per edge midpoint,
        which reads the connectivity and never the positions.
    """

    faces: IntArray
    parameters_km: FloatArray
    curved_km: FloatArray
    flat_km: FloatArray
    displacement_km: FloatArray
    frame: MongeFrame
    curved_levels: list[tuple[FloatArray, IntArray, sparse.csr_matrix]]
    flat_levels: list[tuple[FloatArray, IntArray, sparse.csr_matrix]]
    base: TriangleMesh
    """The coarse mesh the hierarchy stands on, kept only so that :meth:`chart` can
    hand its frame, origin and surface name to a refined container."""

    def chart(self, vertices_km: FloatArray) -> TriangleMesh:
        """One of the two models as a :class:`TriangleMesh`, for writing a rupture file.

        The analysis works in plain arrays, because the sampler and the eikonal solver
        do; this exists only at the file seam, where the native rupture format wants a
        container carrying the frame and the parameter coordinates.

        ``with_triangulation`` keeps the frame exactly rather than refitting it, which
        is what makes the flat twin's ``(u, v)`` identical to the curved mesh's instead
        of merely close -- a refit to the projected points would move the chart under
        its own parameter coordinates.

        Parameters
        ----------
        vertices_km : FloatArray
            ``(V, 3)`` -- :attr:`curved_km` or :attr:`flat_km`.

        Returns
        -------
        TriangleMesh
            Admissible; the builder checks.
        """
        return self.base.with_triangulation(
            vertices_km, self.faces, np.zeros(len(self.faces), dtype=np.int64)
        )

    @property
    def vertex_count(self) -> int:
        """How many vertices each model has. They are the same number; that is the point."""
        return int(self.curved_km.shape[0])

    @property
    def face_count(self) -> int:
        """How many faces each model has."""
        return int(self.faces.shape[0])

    @property
    def median_edge_km(self) -> float:
        """The median edge length of the curved mesh, in kilometres.

        Quoted rather than the requested spacing, because the built lattice is sampled
        in the *parameter* plane and lifted, so the true edge is longer than the
        parameter edge by the metric factor and the delivered resolution is not the
        number that was asked for.
        """
        corners = self.curved_km[self.faces]
        return float(np.median(np.linalg.norm(corners[:, 1] - corners[:, 0], axis=-1)))

    def areas_km2(self, vertices_km: FloatArray) -> FloatArray:
        """Face areas of one of the two models, ``(F,)`` square kilometres.

        Parameters
        ----------
        vertices_km : FloatArray
            ``(V, 3)`` -- :attr:`curved_km` or :attr:`flat_km`.

        Returns
        -------
        FloatArray
            Half the cross product of two edges, per face.
        """
        corners = vertices_km[self.faces]
        return 0.5 * np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=-1,
        )

    def centres_km(self, vertices_km: FloatArray) -> FloatArray:
        """Face centres of one model, ``(F, 3)``, the mean of the three corners."""
        return vertices_km[self.faces].mean(axis=1)

    def slopes(self) -> FloatArray:
        """``grad h`` per face in the parameter plane, ``(F, 2)``, dimensionless.

        ``h`` is a vertex quantity and each face is a triangle in ``(u, v)``, so ``h``
        is affine over the face and its gradient is exact rather than a difference
        approximation. This is the quantity that bounds every departure from flatness
        the study measures, and the one the best-fit plane minimises by construction.

        The identity worth checking against it, and the reason it is here: a Monge
        patch's true area exceeds its projected area by exactly
        ``sqrt(1 + |grad h|^2)`` per face, so
        ``areas_km2(curved) / areas_km2(flat)`` and this must agree to round-off. They
        are computed by completely different routes -- one a cross product in three
        dimensions, the other a two-by-two solve in the parameter plane -- so agreement
        is evidence that the flat twin really is the curved mesh's own projection.

        Returns
        -------
        FloatArray
            ``(F, 2)`` the derivatives along ``e_u`` and ``e_v``.
        """
        corners = self.parameters_km[self.faces]
        heights = self.displacement_km[self.faces]
        first = corners[:, 1] - corners[:, 0]
        second = corners[:, 2] - corners[:, 0]
        # Cramer's rule on [first; second] g = [dh1; dh2]; the determinant is twice the
        # parameter-plane area, which `check_admissible` has already refused to be zero.
        determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        rise_first = heights[:, 1] - heights[:, 0]
        rise_second = heights[:, 2] - heights[:, 0]
        return np.stack(
            [
                (rise_first * second[:, 1] - rise_second * first[:, 1]) / determinant,
                (rise_second * first[:, 0] - rise_first * second[:, 0]) / determinant,
            ],
            axis=-1,
        )

    def face_parameters_km(self) -> FloatArray:
        """Face centres in ``(u, v)``, ``(F, 2)``. Shared, like everything in parameter space."""
        return self.parameters_km[self.faces].mean(axis=1)

    def to_lonlat(self, points_km: FloatArray) -> tuple[FloatArray, FloatArray]:
        """WGS84 longitude and latitude of positions in the mesh's CRS.

        The one crossing to geographic coordinates in this experiment, and it exists
        only so the three hypocentres can be named after real places.

        Parameters
        ----------
        points_km : FloatArray
            ``(..., 3)`` or ``(..., 2)`` positions, absolute in :data:`NZTM`, kilometres.

        Returns
        -------
        tuple of FloatArray
            Degrees east and degrees north.
        """
        transformer = pyproj.Transformer.from_crs(
            NZTM, pyproj.CRS("EPSG:4326"), always_xy=True
        )
        longitude, latitude = transformer.transform(
            np.asarray(points_km)[..., 0] * 1000.0,
            np.asarray(points_km)[..., 1] * 1000.0,
        )
        return np.asarray(longitude), np.asarray(latitude)


def flatten(frame: MongeFrame, vertices_km: FloatArray) -> FloatArray:
    """The same points with their normal displacement removed.

    ``X -> O + u e_u + v e_v``, which is the orthogonal projection onto the best-fit
    plane. Applied to every level of the hierarchy as well as to the finest mesh, so
    the flat model's coarse grids describe the flat model's own geometry rather than
    the curved one's.

    Parameters
    ----------
    frame : MongeFrame
        The reference plane.
    vertices_km : FloatArray
        ``(V, 3)`` positions.

    Returns
    -------
    FloatArray
        ``(V, 3)`` positions on the plane.
    """
    parameters = frame.project(vertices_km)
    parameters[:, 2] = 0.0
    return frame.lift(parameters)


def build_pair(
    path: Path | str = HIKURANGI,
    *,
    spacing_km: float = COARSE_SPACING_KM,
    levels: int = SUBDIVISIONS,
) -> MeshPair:
    """Read the interface, build the mesh, and make its flat twin.

    Parameters
    ----------
    path : Path or str, optional
        The GOCAD TSurf to read.
    spacing_km : float, optional
        The spacing :func:`~rupture_generator.triangular.mesh.remesh` builds at, before
        subdivision.
    levels : int, optional
        How many one-to-four subdivisions to apply.

    Returns
    -------
    MeshPair
    """
    path = Path(path)
    surface = gocad.read_tsurf(path)
    vertices_km, faces = surface.vertices_km, surface.parts[0]
    # The CFM file states no strike or dip, so they are read off the fitted plane's
    # steepest descent -- the geologist's strike and dip of that plane, which uses only
    # the normal and the vertical and so has no sign to guess.
    strike_deg, dip_deg, dips_left = implied_axes(vertices_km)
    base = remesh(
        vertices_km,
        faces,
        spacing_km,
        strike_deg=strike_deg,
        dip_deg=dip_deg,
        dips_left=dips_left,
        # The file names the surface, so the segment cannot end up named after a
        # different interface than the one it was read from. ``Hikurangi.ts.gz`` gives
        # ``hikurangi``, which is the name the study's existing rupture files carry.
        surface=path.name.split(".")[0].lower(),
    )

    frame = base.frame
    curved_levels: list[tuple[FloatArray, IntArray, sparse.csr_matrix]] = []
    current = (base.vertices_km(), base.faces())
    for _ in range(levels):
        finer_vertices, finer_faces, prolongation = subdivided(*current)
        curved_levels.append((current[0], current[1], prolongation))
        current = (finer_vertices, finer_faces)
    curved_km, mesh_faces = current

    parameters = frame.project(curved_km)
    return MeshPair(
        faces=mesh_faces,
        parameters_km=parameters[:, :2].copy(),
        curved_km=curved_km,
        flat_km=flatten(frame, curved_km),
        displacement_km=parameters[:, 2].copy(),
        frame=frame,
        curved_levels=curved_levels,
        flat_levels=[
            (flatten(frame, level_vertices), level_faces, prolongation)
            for level_vertices, level_faces, prolongation in curved_levels
        ],
        base=base,
    )


def same_white_noise(vertex_count: int, seed: int) -> tuple[bool, float]:
    """Whether two identically seeded generators draw the same noise at this size.

    The claim the whole experiment rests on, checked rather than assumed. It is true
    because ``MaternOperator.draw`` consumes exactly ``rng.standard_normal(V)`` and both
    models have the same ``V`` -- but "the seeds will draw differently" is the obvious
    worry about a paired design, so it is measured and reported.

    Parameters
    ----------
    vertex_count : int
        How many deviates each generator is asked for.
    seed : int
        The seed both are built from.

    Returns
    -------
    tuple
        Whether the two vectors are bit-identical, and the largest absolute difference
        between them.
    """
    first = np.random.default_rng(seed).standard_normal(vertex_count)
    second = np.random.default_rng(seed).standard_normal(vertex_count)
    return bool(np.array_equal(first, second)), float(np.abs(first - second).max())
