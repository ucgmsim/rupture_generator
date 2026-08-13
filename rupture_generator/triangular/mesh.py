"""The triangular mesh type: a fault segment as a Monge patch, triangulated.

A segment is a chart ``X(u, v) = O + u e_u + v e_v + h(u, v) n`` -- a reference plane
and a normal displacement over it -- with the parameter domain triangulated and the
triangulation lifted to R^3. :class:`TriangleMesh` wraps that as an `xarray.Dataset`
and carries **methods, not stored copies**, for cell centres, areas, local strike and
dip, arc lengths and boundaries. A derived quantity written down is a second
description of the geometry, free to drift from the first.

Everything is plain vector arithmetic in the projected CRS, in kilometres, and
positions are **offsets from a per-surface origin** -- both for the reasons
:mod:`rupture_generator.mesh` measures: on the ellipsoid a 60 km interface came out
1.4e-2 low in area, and an absolute NZTM northing rounds ~400 times worse than an
offset. This module adds no projection seam of its own; it is `mesh.py` that owns the
one crossing to WGS84.

**The frame splits its two jobs**, because neither choice alone is right.

``n`` is the **best-fit plane normal**, the smallest singular vector of the centred
corner cloud. That is the choice which minimises ``|grad h|`` by construction, and
``|grad h|`` is what bounds every departure from flatness the rest of the pipeline has
to absorb: the margin before the projection folds, the factor by which true surface
length exceeds parameter length, and the distortion the SPDE's diagonal anisotropy is
assuming away. Minimising it is not a convenience.

``e_u`` and ``e_v`` come from the **config's stated strike and dip**, never from the
SVD. The SVD's two in-plane singular vectors are the principal axes of the *point
cloud*: they coincide with strike and dip only when the patch is longer than it is
wide, they are degenerate on a square patch (where a 45-degree answer is as good a fit
as any), and their sign is arbitrary. An arbitrary sign is the reversed-strike failure
-- an SRF that looks entirely plausible and is physically backwards. Taking the axes
from the config also keeps the SPDE anisotropy tensor diagonal: Mai's two correlation
lengths are defined *along strike* and *down dip*, so if the parameter axes are strike
and dip the two lengths cannot mix.

On a planar fault this collapses exactly. The best-fit plane *is* the fault plane, so
``e_u`` is the stated strike and ``e_v`` the stated dip direction: measured across every
single-plane segment of the shipped ``beavan`` and ``kaikoura``, strike comes back to
5.7e-14 degrees and dip to 9.9e-14, against a one-degree rake bound. The lifted
triangles reproduce :meth:`~rupture_generator.mesh.RuptureMesh.areas_km2` to 4.4e-16
relative and :meth:`~rupture_generator.mesh.RuptureMesh.centres` to 5.3e-15 km, both f64
round-off at fault scale.

When a surface arrives with no config to state its axes -- a 3-D fault model, a version
2 file -- :func:`implied_axes` reads them off the fitted plane's **steepest descent**,
which is the geologist's strike and dip of that plane. That uses only the fitted normal
and the vertical, never the SVD's in-plane axes, so it is not the failure above: a patch
samples a plane, it does not change which way the plane dips.

**Admissibility replaces ``validate_chart``.** A Monge patch is a patch at all only if
the projection ``X -> (u, v)`` is injective, and on a triangulation that is checkable
as *every triangle is positively oriented in the parameter plane* -- no folds.
:func:`check_admissible` is the refusal and :func:`fold_margin` is the diagnostic that
goes with it; the former's docstring carries the measurements, on the shipped geometry
and on three real subduction interfaces, and says what the check does **not** catch.

**Element shape is the production constraint, so the mesh is built rather than
refined.** The multigrid sampler's cost is set by element shape, not resolution, and
one-to-four subdivision preserves shape exactly -- it splits each triangle into four
similar ones -- so a badly shaped source stays badly shaped at every level. Measured at
matched vertex count, a built lattice draws in 1.71 s against 61.4 s for a subdivided CFM
interface, a 36-fold penalty, with V-cycle counts flat at 12 against doubling per level.
:func:`remesh` is therefore the production path: it samples the parameter domain at a
target spacing, lifts onto the supplied surface, and reaches 100 m on full Hikurangi at
17.6 M vertices. Its docstring carries the measurements.

**The connectivity always comes from the surface, never from the projection**, and
there are exactly two ways in. :meth:`TriangleMesh.from_patches` takes a quad lattice
and splits every quad; :meth:`TriangleMesh.from_triangulation` takes faces that already
exist, which is what a 3-D fault model gives (see
:mod:`rupture_generator.triangular.gocad`). There is deliberately no constructor that
*infers* connectivity from points. An earlier one did -- Delaunay of the projected
``(u, v)`` -- and it was wrong in a way no shipped example could reveal, because
Delaunay triangulates the convex **hull** and a planar fault's footprint is convex. On
the Williams et al. (2013) Hikurangi interface it inflated a 21x21 block from 800 faces
to 866, area by 6.6%, and ``maximum_slope`` from 0.196 to 18.35;
:meth:`TriangleMesh.from_patches` records the measurement. Taking connectivity from the
surface is also what makes the fold check mean anything, since a triangulation *of* the
projection cannot be inverted with respect to it.

Inferring an outline from a bare point cloud is not merely unimplemented, it is
unreliable on this data, which is why it is absent rather than pending. Measured on the
Hikurangi lattice-with-holes, whose true outline is known from its occupancy mask: the
circumradius of an interior triangle ranges over 5.889 to 6.196 km and that of a
gap-spanning one *starts at 5.941*, so the two overlap and no alpha-shape threshold
separates them. The best available cut still admits 103 faces that are not interface.
For a quantity whose bound is exact, 97% correct is not correct.

**Storing ``(u, v)`` per vertex** is what makes the rest fall out rather than needing
geodesic machinery: the arc lengths are the metric factor ``sqrt(1 + |grad h|^2)`` read
off the per-face Jacobian and integrated, the hypocentre seam is a point-in-triangle
query, the boundary labels are read off the parameter coordinates, and the SPDE's
anisotropy is diagonal because the parameter axes *are* strike and dip.

MESH.md's storage box lists ``strike_arc_km`` and ``dip_arc_km`` among the per-vertex
arrays. They are :meth:`~TriangleMesh.strike_arc_km` and
:meth:`~TriangleMesh.dip_arc_km` **methods** here instead, because they are functions
of the nodes and this package does not write derived quantities down. The decision
MESH.md is making -- that both the parameter coordinates and the true surface arc
lengths are available, because their consumers differ -- is kept in full: the
covariance and the mesh are built on ``(u, v)``, while the hypocentre seam and the SRF
header want arc length, since "the hypocentre is 12 km along strike" means along the
fault.

**The parameter domain is currently exactly the fault, and that is a restriction this
module expects to lose.** The SPDE sampler's Neumann boundary condition reflects the
covariance in the domain boundary (Lindgren et al. appendix A.4), which is not an edge
effect at fault scale: Mai & Beroza figure 13's own 0.25-0.6 ratio bound puts a fault
between 1.7 and 4 correlation lengths across *by construction* -- ``colombia`` is 1.9 --
so a domain cropped to the fault has the reflection everywhere. The circulant sampler
does not suffer it because it pads and crops. The fix is the same one: triangulate a
parameter rectangle **extended past the fault** on all sides, sample on that, and crop
back to the fault.

:func:`padded_builder` is that, handed to the sampler as a callable so it never imports
this module: ``build(pad_strike_km, pad_dip_km)`` returns ``(vertices_km, faces,
parameters_uv, fault_faces)``, with the fault's own mesh and parameter coordinates
untouched and the pad a frame around it. The pad may be **coarser** than the fault, and
usually should be -- its only job is to move the boundary away, and resolving it at fault
resolution multiplies the vertex count several-fold for no modelling gain.

``h(u, v)`` is defined by the fault's own nodes, so the pad is an *extrapolation of the
reference surface*, not more fault, and the flattest available extrapolation is the one
used: ``h`` held at the nearest fault boundary node's value. Two properties make that the
right choice rather than the lazy one. It is continuous across the seam, which matters
because the SPDE assembles from the **lifted** triangles and a jump in ``h`` there is a
crease the cotangent Laplacian reads as real geometry -- moving the artefact the pad
exists to remove onto the fault edge instead of away from it. (That also rules out the
cheapest option, ``h = 0``: ``h`` is zero-*mean*, not zero at the edge.) And it adds no
curvature, which matters because :func:`check_admissible` applies to the padded mesh too:
at the ``|grad h| = 2.14`` a real interface reaches, the tangent plane is already 65
degrees off the reference plane, so anything continuing the *curvature* would turn past
vertical.

What this module deliberately does not do: it does not sample, solve or taper, and it
imports neither :mod:`rupture_generator.triangular.spde` nor
:mod:`rupture_generator.triangular.fim`. Those two take plain ``vertices_km`` and
``faces`` arrays so the three components stay decoupled. It also does not generate a
surface that folds -- :func:`check_admissible` refuses one honestly, but the escape
hatch for a fault that turns too far is a *ruled* reference surface, which is what
``build_fault``'s bend stretch already constructs, and nothing needs it yet.
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
from scipy.spatial import KDTree

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
    from collections.abc import Callable, Mapping, Sequence

    from rupture_generator.config.geometry import FaultConfig, PointConfig

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

SCHEMA_VERSION = 3
"""Bumped when a reader of an older file would get the wrong answer rather than an error.

Version 3 stores a *triangulation*: a flat node table, a face table, and the per-vertex
parameter coordinates. Versions 1 and 2 stored a structured ``(dip_node, strike_node)``
lattice whose connectivity was its shape, so a v2 reader handed a v3 file finds no
lattice, and a v3 reader handed a v2 file has to supply the connectivity itself --
which :func:`from_datatree` does, triangulating the lattice on the way in.
"""

NODE_DIM = "node"
"""The dim a vertex quantity lives on."""

FACE_DIMS = ("face",)
"""The dims a stage's field lives on: one value per triangle, flat.

Deliberately **not** nzcvm's ``i``/``j``/``k``. In that codebase those name structured
axes in ``GridSchema`` and (vertex, cell, corner) in ``TetrahedralMeshSchema`` at the
same time, so anything that aligns across the two broadcasts garbage without
complaining.
"""

CORNER_DIM = "corner"
"""The dim a face's vertex indices lie along. Its *size* is the cell arity, read from
``faces.shape`` rather than hard-coded, following ``TetrahedralMeshSchema``."""

NODE_VARIABLES = ("east_km", "north_km", "depth_km", "strike_km", "dip_km")
"""The chart's own geometry, per vertex: three positions and the two parameter
coordinates. What a mesh file stores, and the only thing it stores."""

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
"""How little surface a vertex may carry, against the median, before the mesh is refused.

A vertex's lumped mass is the area it carries -- a third of every triangle it touches --
and it is the diagonal of the SPDE sampler's mass matrix. A vertex with almost none is
barely constrained by the operator, so its marginal variance explodes; and because
``sampling.standardise`` divides the whole field by its sample standard deviation, a
couple of such vertices drag the *healthy* slip distribution down with them. This is the
silent-and-plausible failure class: the field still looks like a field and nothing
raises.

**The constant lives here rather than in the sampler** because it states a property of a
*mesh*, and this module owns meshes; the sampler imports it and keeps its own backstop
for the same condition. This check is the primary gate.

**Measured, not chosen.** The sampler's own sweep found a cliff rather than a slope: at a
worst mass ratio of 1.47e-05 the drawn field's healthy sample spread is 0.9996, and at
1.47e-06 it is 0.0487 -- a factor of twenty across a factor of ten in mass. This value is
the geometric middle of that decade, which is the only defensible place to stand when the
transition is that sharp: an order of magnitude of headroom on each side.

The three NZ CFM v1.0 subduction interfaces bracket it independently, as a fraction of
each mesh's own median lumped mass:

======================  =========  ===========  ==============  ==========
interface               V          worst ratio  below 5e-6      field std
======================  =========  ===========  ==============  ==========
Puyseguer               2597       7.3e-07      **1 vertex**    0.187
Hikurangi               5218       2.4e-05      none            0.996
Puysegur-Fiordland      2312       2.4e-05      none            1.000
======================  =========  ===========  ==============  ==========

Puyseguer's worst vertex inflates its own marginal variance to 9.5e6, driving the field's
standard deviation to 6.26 and suppressing the healthy part of the slip distribution
**5.3-fold**. Every lattice mesh this package builds sits at exactly 1/6 -- the worst a
lattice can do is a corner vertex touching two triangles -- so no mesh built here can
approach the line.

**The limitation, which matters and is not obvious.** The mass *ratio* is
refinement-invariant: 1-to-4 subdivision splits every triangle into four similar ones, so
a mesh's worst ratio is the same at every level and Hikurangi passes this gate at all of
them. The *damage* is not invariant. Measured over three refinements of Hikurangi, the
maximum per-vertex variance runs 9.2, 12.9, 257, and by the third the drawn field's
sample spread is 8.0 -- which is what ``standardise`` divides the whole segment by. So
**this gate does not protect a refined bad mesh**, and no threshold on a
refinement-invariant quantity could. The answer is not to refine a badly shaped mesh at
all: see :func:`remesh`, which builds a well-shaped one at a target resolution instead.
"""

BOUNDARY_LABELS = ("top", "bottom", "lateral")
"""What a boundary edge can be, which is what the taper needs to tell apart."""

_MINIMUM_IN_PLANE_LENGTH = float(np.sqrt(np.finfo(np.float64).eps))
"""How much of the stated strike must survive projection into the fitted plane.

Not a modelling choice -- an error bound. Projecting a unit vector out of a plane
leaves a residual of length ``L``, and the residual's *direction* carries an error of
about ``eps / L``, because the round-off in the subtraction is absolute while the
result being normalised is ``L``. At ``L = sqrt(eps) = 1.5e-8`` the recovered strike is
uncertain by ``sqrt(eps)`` radians -- 8.5e-7 degrees, still six orders inside the
one-degree rake bound -- and below it the frame would be reporting a strike it does not
know. Real geometry sits at ``L = 1``: the stated strike lies *in* the fitted plane to
round-off, which is what the planar collapse means.
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
    ``(sin, cos)`` rather than the mathematical ``(cos, sin)``. Named by the right-hand
    rule the same way :class:`~rupture_generator.config.geometry.PlaneConfig` is:
    walking the trace from its first point to its last, ``dips_left=False`` dips away
    to your right.

    Parameters
    ----------
    strike_deg : float
        Grid-north bearing of the trace direction.
    dip_deg : float
        Dip below horizontal, in ``(0, 90]``.
    dips_left : bool, optional
        Whether the plane dips to the left of the trace direction.

    Returns
    -------
    tuple of FloatArray
        The strike direction and the down-dip direction, both unit, both ``(3,)``.
        They are orthogonal for every dip, which is what makes the planar collapse in
        :meth:`MongeFrame.fit` exact rather than approximate.
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

    For data that arrives without a config: a 3-D fault model, or a version 2 mesh
    file. Fit the plane, take its **steepest descent** direction, and read the strike
    and dip off that -- which is the geologist's definition of a plane's strike and dip,
    and is unique up to the sign convention that a plane dips to the *right* of its
    strike (:class:`~rupture_generator.config.geometry.PlaneConfig`'s own default).

    This is **not** the failure mode the module docstring rejects. That one is taking
    the in-plane axes from the SVD's in-plane singular vectors, which are the point
    cloud's principal axes: degenerate on a square patch and arbitrary in sign. What
    happens here uses only the fitted **normal** -- which the design takes from the SVD
    anyway -- crossed with the vertical, so it has no dependence on the sampling at all
    and no sign to guess. A patch samples the plane; it does not change which way the
    plane dips.

    It is also a fixed point: feed the result to :meth:`MongeFrame.fit` and the frame
    that comes back has exactly these axes, because the stated strike already lies in
    the fitted plane.

    Parameters
    ----------
    points_km : FloatArray
        ``(n, 3)`` positions, components ``(east, north, depth)``, depth positive down.

    Returns
    -------
    tuple
        ``(strike_deg, dip_deg, dips_left)``, ready to pass to
        :meth:`TriangleMesh.from_triangulation`. ``dips_left`` is always ``False``: the
        sign is *chosen* here rather than discovered, which is the point.

    Raises
    ------
    ValueError
        If the fitted plane is horizontal, where a strike is not defined at all.
    """
    points = np.asarray(points_km, dtype=np.float64)
    _, _, rotation = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    normal = rotation[2]

    # The steepest descent direction in the plane: straight down, with whatever part
    # of it leaves the plane removed.
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
    an origin on the plane and an orthonormal triple with ``e_u x e_v = n``, so the
    parameter plane is right-handed about the normal and a positively oriented triangle
    in ``(u, v)`` is one the projection has not folded.

    ``eq=False`` for the same reason
    :class:`~rupture_generator.mesh.RuptureMesh` uses it: the generated ``__eq__``
    would compare arrays and hand back an array, which raises the moment anything puts
    a frame in an ``if``. Two frames are compared by comparing their axes.
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

        The normal is the smallest right singular vector of the centred cloud -- the
        least-squares plane, which minimises the normal displacement ``h`` and so keeps
        the margin before the projection folds as wide as this construction can. Its
        *sign* is arbitrary out of the SVD, so it is flipped to agree with
        ``cross(strike, dip)`` from the config; that is the only thing the config's
        numbers decide about the normal.

        The in-plane axes come from the config alone. See this module's docstring for
        why: the SVD's in-plane axes are the cloud's principal axes, degenerate on a
        square patch and arbitrary in sign.

        **What the plane is fitted to matters more than it looks.** MESH.md says the
        *corner* cloud, and on a planar patch that is right and costs nothing: bilinear
        subdivision makes every interior node an affine combination of the corners, so
        both span the same plane exactly. On a *curved* patch a plane through four
        corners is not a fit to anything -- it interpolates four points and ignores the
        surface between them.

        Nor is a fit to every node right, even weighted by the area each carries: that
        is a fit to the *sample*. Refine one plane of a segment and its nodes outvote
        the others, tilting a plane that should not have moved, and ``|grad h|`` --
        which is a statement about how the planes are oriented, not about how finely
        they were cut -- picks up a dependence on the subfault size. Measured across a
        factor of four in cell size on ``alpine_hope``: 4e-4 relative drift.

        So the callers pass a **quadrature of the surface** instead, and this is the
        weighted fit that consumes it; :func:`_surface_moment` builds it. The result is
        the exact continuous least-squares plane of the surface, which subdividing a
        triangle cannot move at all.

        Parameters
        ----------
        corners_km : FloatArray
            ``(n, 3)`` points to fit the plane through.
        strike_deg, dip_deg : float
            The config's stated geometry, in degrees.
        dips_left : bool, optional
            Whether the plane dips left of the trace direction.
        weights : FloatArray, optional
            ``(n,)`` non-negative weights, in square kilometres when they are the area
            each node carries. Unweighted by default, which is right for a cloud whose
            points already sample the surface evenly.

        Returns
        -------
        MongeFrame
            With ``origin_km`` at the weighted centroid, which lies on the fitted plane.

        Raises
        ------
        ValueError
            If fewer than three points are given, if the weights are the wrong shape or
            sum to nothing, or if the stated strike is perpendicular to the fitted plane
            -- a strike that lies along the normal names no direction in the patch, which
            means the stated geometry and the node positions are describing different
            surfaces.
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
        # sqrt on the rows is what turns an unweighted SVD into a weighted least
        # squares: the singular values are then sums of `w * residual^2`.
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
        """The same plane with its origin moved within it. Functional, never in place.

        Moving the origin *within* the plane leaves ``h`` untouched, which is what lets
        the builder put ``(u, v) = (0, 0)`` at the patch's shallow near corner without
        changing the surface it describes.

        Parameters
        ----------
        strike_km, dip_km : float
            How far to move the origin along each in-plane axis.

        Returns
        -------
        MongeFrame
            A new frame with the same axes.
        """
        return dataclasses.replace(
            self,
            origin_km=self.origin_km
            + strike_km * self.strike_axis
            + dip_km * self.dip_axis,
        )

    def project(self, points_km: FloatArray) -> FloatArray:
        """Points as ``(u, v, h)``: two parameter coordinates and a normal height.

        Parameters
        ----------
        points_km : FloatArray
            ``(..., 3)`` positions, offsets from the surface origin.

        Returns
        -------
        FloatArray
            ``(..., 3)``, the components along ``e_u``, ``e_v`` and ``n``.
        """
        basis = np.stack([self.strike_axis, self.dip_axis, self.normal], axis=1)
        return (np.asarray(points_km, dtype=np.float64) - self.origin_km) @ basis

    def lift(self, parameters_km: FloatArray) -> FloatArray:
        """The inverse of :meth:`project`: ``(u, v, h)`` back to a position.

        Parameters
        ----------
        parameters_km : FloatArray
            ``(..., 3)`` components along ``e_u``, ``e_v`` and ``n``.

        Returns
        -------
        FloatArray
            ``(..., 3)`` positions, offsets from the surface origin.
        """
        basis = np.stack([self.strike_axis, self.dip_axis, self.normal], axis=0)
        return self.origin_km + np.asarray(parameters_km, dtype=np.float64) @ basis

    @property
    def strike_deg(self) -> float:
        """The frame's own strike: the grid-north bearing of ``e_u``, in ``[0, 360)``."""
        return float(
            np.degrees(np.arctan2(self.strike_axis[0], self.strike_axis[1])) % 360.0
        )

    @property
    def dip_deg(self) -> float:
        """The frame's own dip: how far ``e_v`` plunges below horizontal, in degrees."""
        return float(np.degrees(np.arcsin(np.clip(self.dip_axis[2], -1.0, 1.0))))

    def to_attrs(self) -> dict[str, FloatArray]:
        """The frame as file attributes.

        Returns
        -------
        dict
            Four ``(3,)`` arrays, keyed by :data:`RESERVED_ATTRS` names.
        """
        return {
            "frame_origin_km": np.asarray(self.origin_km, dtype=np.float64),
            "strike_axis": np.asarray(self.strike_axis, dtype=np.float64),
            "dip_axis": np.asarray(self.dip_axis, dtype=np.float64),
            "normal": np.asarray(self.normal, dtype=np.float64),
        }

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any]) -> MongeFrame:
        """Read a frame back out of file attributes.

        Parameters
        ----------
        attrs : Mapping
            What :meth:`to_attrs` wrote.

        Returns
        -------
        MongeFrame
            The frame, with its axes exactly as stored -- not refitted, because a
            refit would move the chart under its own parameter coordinates.
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
        """Wrap a dataset in the layout :meth:`from_patches` builds.

        Parameters
        ----------
        dataset : xr.Dataset
            Carrying :data:`NODE_VARIABLES`, :data:`FACE_VARIABLES` and the frame in
            its attributes. Not the constructor to reach for -- use
            :meth:`from_patches`, or read a file.
        """
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
        """**The lattice builder.** Fit a frame, split every quad, lift.

        The reference frame is fitted to the patch corners and every node is projected
        to ``(u, v)``; the connectivity is then read straight off the lattice -- two
        triangles per quad, split on the ``(i, j) -> (i+1, j+1)`` diagonal -- and lifted
        by taking each vertex's own position. Patches that share nodes exactly, which is
        what a fused seam column is, are welded, so the seam is one row of vertices and
        the triangulations either side of it conform.

        **Not Delaunay of the projected points**, which is what this used to do, and the
        reason is a measurement on a real curved surface rather than a preference. The
        Williams et al. (2013) Hikurangi interface, largest fully populated 21x21 block:

        ==========================  =======  =======
        quantity                    lattice  Delaunay
        ==========================  =======  =======
        faces (must be ``2*20*20``) 800      866
        area, km^2                  28421.0  30418.4
        ``maximum_slope()``         0.196    18.35
        ==========================  =======  =======

        `scipy.spatial.Delaunay` triangulates the **convex hull** of the points it is
        given. A planar fault's parameter footprint is a convex quadrilateral, so the
        hull is the domain and nothing is added -- which is why every shipped example
        passed. A curved surface's footprint is slightly concave (about 1% here), and
        Delaunay fills each notch with faces that are not on the fault: 66 of them,
        carrying 1997.4 km^2, or **6.6% of an area whose bound is exact** because moment
        is. Their slopes reach 18.35 against a true maximum of 0.196, so
        :meth:`maximum_slope` and :func:`check_admissible` were reading hull slivers
        rather than the interface. And a sliver spanning a notch is a **shortcut edge**
        in the mesh graph, which the eikonal wavefront would propagate along and the
        SPDE would inherit as connectivity.

        Culling the slivers by patch membership cannot fix it: the test compared each
        face's centroid against the quad through the patch's four *corners*, and once
        the surface curves that quad is not the patch's parameter footprint. The
        connectivity was never unknown -- a lattice has it -- so taking it from the
        lattice is exact, ``O(n)``, preserves the boundary, and needs no hull reasoning
        at all. :meth:`from_triangulation` is the entry point for a surface that arrives
        with its own faces; there is no constructor that guesses them.

        **Extending the domain past the fault**, which the SPDE sampler's boundary
        reflection will want (see this module's docstring), needs three changes and no
        more. They are written down here rather than made, because where the extension
        lives is a decision about the sampler as much as about the container.

        1. The pad arrives as its **own argument**, not as another entry in ``patches``.
           ``patches`` is what the frame is fitted to, and fitting the reference plane
           to an extrapolation *of that plane* would be circular.
        2. Its faces are marked ``plane_of_face = -1``, and a ``fault_faces()``
           predicate reads it back. The pad is a lattice too, so it brings its own
           connectivity the same way a patch does, and it conforms along the fault
           boundary if it is laid out on the fault's own boundary nodes.
        3. Two places stop meaning the padded domain and start meaning the fault. The
           frame translation below takes its minimum over ``patches`` rather than over
           every vertex, so the fault still starts at ``(0, 0)`` and the pad is simply
           negative -- otherwise every stored parameter coordinate shifts and the
           hypocentre seam silently moves. And :meth:`arc_profile` integrates over the
           fault's faces rather than all of them, so the SRF's extents and
           :meth:`cell_index`'s bounds are the fault's rather than the pad's.

        Parameters
        ----------
        patches : Sequence of FloatArray
            One ``(n_i+1, n_j+1, 3)`` node lattice per config plane, ``i`` down dip and
            ``j`` along strike, positions offsets from the surface origin.
        strike_deg, dip_deg : float
            The config's stated geometry for this segment.
        dips_left : bool, optional
            Whether the segment dips left of its trace direction.
        origin_east_km, origin_north_km : float
            The surface origin, in the CRS, kilometres.
        surface : str
            The surface's name, which becomes the group name in files.

        Returns
        -------
        TriangleMesh
            With exactly ``2 * n_i * n_j`` faces summed over the patches, less any the
            weld shares.

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
        # Exact equality: a shared seam column is the *same* corner arithmetic run
        # twice in `build_fault`, so duplicates are bitwise identical rather than
        # close, and a tolerance here would weld two nodes that genuinely differ.
        # `inverse` is what carries lattice position into vertex index, and it has to
        # survive: `np.unique` sorts, so vertex order is *not* lattice order and
        # reshaping anything per-vertex back to the lattice gives silent nonsense.
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
            # near -> far -> opposite -> beside turns positively and both its triangles
            # do too. That is what makes `check_admissible` a test rather than a
            # tautology -- the orientation now comes from the surface's own lattice
            # instead of from a triangulation of the projection.
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
        """**The builder for a surface that arrives with its own faces.**

        What a 3-D fault model gives you. The NZ CFM v1.0 subduction interfaces are
        GOCAD TSurf files carrying 9236, 4090 and 4041 triangles of their own, and
        :mod:`rupture_generator.triangular.gocad` reads them straight into this.

        The connectivity is **kept exactly as given**. Nothing here triangulates,
        retriangulates, culls or repairs: the faces are the modeller's statement about
        what the surface is, and a builder that second-guessed them would be back in the
        convex-hull business :meth:`from_patches` documents its way out of. The one
        thing that *is* normalised is the global winding -- whether the modeller
        numbered each face anticlockwise seen from the hanging wall or the footwall is a
        convention rather than a fact, and all three CFM interfaces use the opposite one
        to this frame. Faces disagreeing with the *majority* are the folds, and those
        are refused; see the comment on the flip.

        That also makes :func:`check_admissible` mean something on these meshes.
        Orientation tested against a triangulation *of the projection* is a tautology --
        `scipy.spatial.Delaunay` orients every face positively by construction -- whereas
        orientation tested against the surface's own faces is a real question with a real
        answer. Measured: all three CFM interfaces have **zero** inverted triangles, so
        each is genuinely one Monge patch.

        The frame is fitted to **every vertex** rather than to four corners, because a
        triangulated interface has no corners. On a lattice patch the two are the same
        fit -- bilinear subdivision makes every interior node an affine combination of
        the corners -- so this is not a second convention, it is the same one where
        corners exist and the only available one where they do not. The in-plane axes
        still come from ``strike_deg`` and ``dip_deg`` and never from the SVD; see
        :func:`implied_axes` for what to pass when the surface arrives without a config
        to state them.

        Parameters
        ----------
        vertices_km : FloatArray
            ``(V, 3)`` positions, offsets from the surface origin, components
            ``(east, north, depth)`` with depth positive down.
        faces : IntArray
            ``(F, 3)`` zero-based vertex indices, wound so that each face is
            anticlockwise seen from the ``strike_deg``/``dip_deg`` side.
        strike_deg, dip_deg : float
            The stated geometry, in degrees.
        dips_left : bool, optional
            Whether the surface dips left of the strike direction.
        origin_east_km, origin_north_km : float, optional
            The surface origin, in the CRS, kilometres.
        surface : str
            The surface's name.
        plane_of_face : IntArray, optional
            ``(F,)`` provenance. Defaults to zeros -- one part.

        Returns
        -------
        TriangleMesh
            With ``faces`` unchanged.

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
        """Project through a settled frame and lay the arrays out.

        Parameters
        ----------
        vertices_km : FloatArray
            ``(V, 3)`` positions.
        faces : IntArray
            ``(F, 3)`` vertex indices.
        plane_of_face : IntArray
            ``(F,)`` provenance.
        frame : MongeFrame
            Already fitted and already translated.
        origin_east_km, origin_north_km : float
            The surface origin.
        surface : str
            The surface's name.

        Returns
        -------
        TriangleMesh
            Unchecked -- the caller runs :func:`check_admissible`.
        """
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
        """Lay the arrays out as the dataset, with no geometry decided here.

        Parameters
        ----------
        vertices_km : FloatArray
            ``(V, 3)`` node positions, offsets from the surface origin.
        parameters_km : FloatArray
            ``(V, 2)`` parameter coordinates.
        faces : IntArray
            ``(F, 3)`` vertex indices.
        plane_of_face : IntArray
            ``(F,)`` config-plane provenance.
        frame : MongeFrame
            The reference frame the parameter coordinates are in.
        origin_east_km, origin_north_km : float
            The surface origin, in the CRS, kilometres.
        surface : str
            The surface's name.

        Returns
        -------
        TriangleMesh
            The wrapped dataset.
        """
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
        """Node positions, ``(V, 3)``, components (east, north, depth).

        Returns
        -------
        FloatArray
            Offsets from the surface origin, in kilometres. This and :meth:`faces` are
            the pair the sampler and the eikonal solver take -- plain arrays, so the
            three components stay decoupled.
        """
        return np.stack(
            [
                self._dataset["east_km"].to_numpy(),
                self._dataset["north_km"].to_numpy(),
                self._dataset["depth_km"].to_numpy(),
            ],
            axis=-1,
        )

    def faces(self) -> IntArray:
        """Triangles as vertex indices, ``(F, 3)``.

        Returns
        -------
        IntArray
            Positively oriented in the parameter plane -- see :func:`check_admissible`.
        """
        return self._dataset["faces"].to_numpy()

    def parameters_km(self) -> FloatArray:
        """Per-vertex parameter coordinates ``(u, v)``, ``(V, 2)``.

        Returns
        -------
        FloatArray
            Along strike and down dip in the frame, in kilometres, both starting at
            zero at the patch's shallow near corner. These are what the covariance and
            the mesh are built on; :meth:`strike_arc_km` and :meth:`dip_arc_km` are the
            true surface lengths that go with them.
        """
        return np.stack(
            [
                self._dataset["strike_km"].to_numpy(),
                self._dataset["dip_km"].to_numpy(),
            ],
            axis=-1,
        )

    def planes(self) -> IntArray:
        """Which config plane each face came from, ``(F,)``.

        Returns
        -------
        IntArray
            Index into the segment's own patches, in trace order.
        """
        return self._dataset["plane_of_face"].to_numpy()

    # -------------------------------------------------------------- the fields

    def fields(self) -> frozenset[str]:
        """Every attached field's name.

        Returns
        -------
        frozenset of str
            The variables whose dims are exactly :data:`FACE_DIMS` and whose names are
            not the chart's own, so no second list has to be kept in step. Geometry is
            not in here; geometry is computed.
        """
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
            Naming the field and listing what this chart does carry -- a stage asking
            for a field nobody drew is a pipeline written in the wrong order.
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

        Returns
        -------
        TriangleMesh
            A new chart; the one this was called on is untouched.

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
        """This chart with those fields dropped. Functional, never in place.

        A name that is not there is not an error: dropping is a statement about the
        result, not a claim about the history.

        Returns
        -------
        TriangleMesh
            A new chart without them.
        """
        return self._with(self._dataset.drop_vars(names, errors="ignore"))

    @property
    def attrs(self) -> Mapping[str, Any]:
        """What this chart records about itself, read-only.

        A mutable view is a mutable chart, so this is a proxy.

        Returns
        -------
        Mapping
            The frame and the origin, plus whatever a stage recorded -- the truncation
            diagnostic, and on the one segment that holds it, where the rupture
            nucleated.
        """
        return types.MappingProxyType(dict(self._dataset.attrs))

    def with_attrs(self, **values: Any) -> TriangleMesh:
        """This chart with the attributes given in ``values``.

        Scalars by convention: these are written straight into a file's group
        attributes.

        Returns
        -------
        TriangleMesh
            A new chart carrying them.

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

        The one stage whose output is not a face field: a pulse per face, each its own
        length, so they are carried as CSR exactly as
        :meth:`~rupture_generator.mesh.RuptureMesh.with_pulses` carries them -- same dim
        names, same checks, so a reader of either file finds the same two arrays.

        Parameters
        ----------
        offsets : IntArray
            Where each face's pulse starts, length ``face_count + 1``.
        samples : FloatArray
            Every pulse, concatenated.

        Returns
        -------
        TriangleMesh
            A new chart carrying them.

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

    # ------------------------------------------------------- derived quantities

    def centres(self) -> FloatArray:
        """Face centres, ``(F, 3)`` -- the mean of the three corners.

        Returns
        -------
        FloatArray
            Positions, offsets from the surface origin.
        """
        return self.vertices_km()[self.faces()].mean(axis=1)

    def areas_km2(self) -> FloatArray:
        """Face areas, ``(F,)``.

        Returns
        -------
        FloatArray
            Half the cross product of two edges. This is exactly one of the two terms
            :meth:`~rupture_generator.mesh.RuptureMesh.areas_km2` sums -- that formula
            is already a two-triangle split, so the quad mesh's area is the sum of its
            triangles' by construction rather than by approximation.
        """
        corners = self.vertices_km()[self.faces()]
        return 0.5 * np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=-1,
        )

    def parameter_areas_km2(self) -> FloatArray:
        """**Signed** face areas in the parameter plane, ``(F,)``.

        Returns
        -------
        FloatArray
            Positive where the projection preserves orientation. A non-positive entry
            is a fold: two pieces of surface over one piece of plane. This is the whole
            of the admissibility test -- see :func:`check_admissible`.
        """
        corners = self.parameters_km()[self.faces()]
        first = corners[:, 1] - corners[:, 0]
        second = corners[:, 2] - corners[:, 0]
        return 0.5 * (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])

    def lumped_mass_km2(self) -> FloatArray:
        """Surface area carried by each vertex, ``(V,)``.

        A third of every triangle the vertex touches -- the barycentric dual area, which
        is also the diagonal of the SPDE sampler's lumped mass matrix. It sums to the
        mesh's total area exactly, which is what makes it a partition of the surface
        rather than an estimate of one.

        Returns
        -------
        FloatArray
            Square kilometres. :data:`DEGENERATE_MASS_FRACTION` says how small a share
            is too small, and why.
        """
        return _vertex_area_km2(self.vertices_km(), self.faces())

    def face_quality(self) -> FloatArray:
        """Shape quality of each face, ``(F,)``, in ``[0, 1]``.

        ``4 sqrt(3) A / (a^2 + b^2 + c^2)``: one for an equilateral triangle, zero for a
        degenerate one, and scale-free. A lattice split on its diagonal gives 0.866 for
        a square cell, which is where every mesh this package builds sits.

        Returns
        -------
        FloatArray
            Dimensionless. Reported by :func:`check_admissible` when it refuses a mesh,
            because "face 1966 has quality 4.8e-05" is a thing a modeller can act on and
            "the mass matrix is singular" is not.
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

        Returns
        -------
        FloatArray
            ``cross(X1 - X0, X2 - X0)``, normalised, so it points to the same side as
            the frame's own normal on every positively oriented face. A face of zero
            area gets the frame's normal rather than a NaN, which would travel silently
            into an SRF.
        """
        corners = self.vertices_km()[self.faces()]
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        magnitude = np.linalg.norm(normal, axis=-1)
        degenerate = magnitude == 0.0
        unit = normal / np.where(degenerate, 1.0, magnitude)[:, None]
        return np.where(degenerate[:, None], self.frame.normal, unit)

    def slope(self) -> FloatArray:
        """Per-face ``grad h = (dh/du, dh/dv)``, ``(F, 2)``, dimensionless.

        Computed **from the face normal, not from the affine map** ``dX/d(u, v)``, and
        the difference is not cosmetic. A Monge patch's normal is proportional to
        ``(-h_u, -h_v, 1)`` in the ``(e_u, e_v, n)`` basis, so reading ``grad h`` off the
        normal costs one cross product and one division, and it is well conditioned
        whenever the *triangle* is -- which is what actually matters.

        The obvious alternative inverts the parameter-space edge matrix, and that matrix
        is near-singular exactly when a face is a sliver in projection -- which real
        meshes have: the CFM Hikurangi interface's worst face has a shape quality of
        5.4e-04. On these three interfaces the two routes happen to agree, so the
        fragility is not something this repository has caught in the act; it is avoided
        on principle, because ``|grad h|`` is the number :func:`check_admissible` reports
        and the whole construction is budgeted against, and this route has no matrix
        inverse in it to be conditioned badly at all.

        Returns
        -------
        FloatArray
            ``|grad h|`` is ``tan`` of the angle between the face's normal and the
            frame's: zero where the surface is parallel to the reference plane, and
            growing without bound as it turns perpendicular to it. It is both the margin
            before the projection folds and the factor ``sqrt(1 + |grad h|^2)`` by which
            true surface length exceeds parameter length.
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
        """The worst ``|grad h|`` on the patch, dimensionless.

        Returns
        -------
        float
            Zero on a planar segment, to round-off.
        """
        return float(np.linalg.norm(self.slope(), axis=-1).max(initial=0.0))

    def strike_dip_deg(self) -> tuple[FloatArray, FloatArray]:
        """Per-face strike (grid north, ``[0, 360)``) and dip (``[0, 90]``).

        Both come from the face's normal rather than its edges: on a plane the two
        agree, and the normal is what keeps them right on a surface that is not one.
        The absolute value on the normal's vertical component makes the dip independent
        of the normal's sign.

        The strike's *sign* is fixed by the frame's ``e_u``, and therefore by the
        config. That is the departure from
        :meth:`~rupture_generator.mesh.RuptureMesh.strike_dip_deg`, which orients the
        strike by the cell's along-strike edges: a quad has two of those and a triangle
        has none, so the frame has to carry the orientation instead. It is worth being
        explicit about because a reversed strike produces an SRF that looks entirely
        plausible and is physically backwards.

        Everything here is built from the face's own normal and the frame, and nothing
        from the parameter-space affine map -- :meth:`slope` says why that map is not
        safe to read on a real mesh. A degenerate face reports dip 0 and the frame's
        strike, never NaN.

        This pair is also the **anisotropy frame** the SPDE sampler takes: the surface's
        own strike and dip, ``e_strike = normalise(z x n)`` and ``e_dip = n x e_strike``,
        rather than anything derived from ``(u, v)``. That is why the sign convention
        here is worth stating exactly rather than leaving to the caller.

        Returns
        -------
        tuple of FloatArray
            Strike and dip, each ``(F,)``, in degrees.
        """
        unit = self.face_normals()
        dip_deg = np.degrees(np.arccos(np.clip(np.abs(unit[..., 2]), 0.0, 1.0)))

        # cross(DOWN, n) is perpendicular to down (horizontal) and to the normal (in
        # the face's plane): the strike direction, up to sign.
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

        The **derived** half of MESH.md's pair. Parameter length and true surface
        length differ by exactly ``sqrt(1 + h_u^2)`` per element -- which this reads off
        the Jacobian rather than solving for -- and the map is that factor integrated
        across the patch:

        ``S(u) = integral of M(u') du'``, where ``M`` is the *area-weighted mean* of
        ``sqrt(1 + h_u^2)`` over the faces the patch has at ``u'``.

        Averaging across dip rather than following one line ``v = const`` is the
        deliberate choice, and it is the one the consumers want. "The hypocentre is 12
        km along strike" and the SRF's plane length are both statements about *the
        fault's* extent, which is one number per ``u``, not one per ``(u, v)``. It is
        also what makes the map strictly increasing, so :meth:`cell_index` can invert it
        and stay a query in the parameter plane -- a per-line arc length is not
        guaranteed to be a reparameterisation at all, because two neighbouring lines can
        differ in length.

        Exact where it has to be. Each face's dip extent is taken as constant across its
        own ``u`` span, so ``M`` is piecewise constant on the ``2F`` face endpoints and
        the integral is a sorted cumulative sum -- no quadrature grid, no bin count to
        choose, and ``O(F log F)``. Both the total surface area and the total parameter
        area are preserved exactly by that flattening, and on a planar patch ``M`` is
        identically one, so ``S(u) = u`` to round-off.

        Parameters
        ----------
        axis : int
            0 for strike, 1 for dip.

        Returns
        -------
        tuple of FloatArray
            Parameter knots, ascending, and the arc length at each -- the two arrays
            `numpy.interp` takes, in either direction.
        """
        parameters = self.parameters_km()[self.faces()][..., axis]
        low, high = parameters.min(axis=1), parameters.max(axis=1)
        knots = np.unique(np.concatenate([low, high]))
        if knots.size < 2:
            return knots, np.zeros_like(knots)

        # A triangle in the parameter plane is a tent in `axis`; flattening it to a
        # rectangle of the same area keeps both integrals below exact and leaves only
        # the *shape* within one cell approximate.
        density = self.parameter_areas_km2() / (high - low)
        metric = np.sqrt(1.0 + self.slope()[:, axis] ** 2)
        opens = np.searchsorted(knots, low)
        closes = np.searchsorted(knots, high)

        def active(weights: FloatArray) -> FloatArray:
            """Total weight of the faces spanning each interval between knots.

            A face opens at its low knot and closes at its high one, so a running sum
            of those two events gives every interval's total in one pass -- there is no
            per-interval search.
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

        Not ``u``, which is the *projected* length: the two differ by
        ``sqrt(1 + h_u^2)``, which is 1.000 on a planar patch, 1.18 at the
        ``|grad h| = 0.63`` the shipped ``alpine_hope`` reaches and 2.37 at the 2.14 a
        real subduction interface reaches. This is what the
        hypocentre spec and the SRF header extents want, because "the hypocentre is 12
        km along strike" means along the fault. See :meth:`arc_profile`.

        Returns
        -------
        FloatArray
            Kilometres, zero at the patch's near end.
        """
        knots, arc_km = self.arc_profile(0)
        return np.interp(self.parameters_km()[:, 0], knots, arc_km)

    def dip_arc_km(self) -> FloatArray:
        """True surface distance down dip, per vertex, ``(V,)``.

        Returns
        -------
        FloatArray
            Kilometres, zero at the patch's top edge. :meth:`strike_arc_km` says why
            this is not simply ``v``.
        """
        knots, arc_km = self.arc_profile(1)
        return np.interp(self.parameters_km()[:, 1], knots, arc_km)

    # ------------------------------------------------------------- the boundary

    def _half_edges(self) -> tuple[IntArray, IntArray, IntArray]:
        """Every directed edge, its face, and how many faces share it undirected.

        Returns
        -------
        tuple of IntArray
            ``(3F, 2)`` directed edges in face order, ``(3F,)`` face indices, and
            ``(3F,)`` incidence counts.
        """
        faces = self.faces()
        directed = np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
        )
        of_face = np.tile(np.arange(self.face_count, dtype=np.int64), 3)
        _, inverse, counts = np.unique(
            np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
        )
        return directed, of_face, counts[inverse.ravel()]

    def edges(self) -> IntArray:
        """Every undirected edge once, ``(E, 2)``, each pair sorted ascending.

        Returns
        -------
        IntArray
            Built as all ``3F`` directed edges, canonicalised as sorted pairs and
            deduplicated -- the same pass :meth:`boundary_edges` reads its counts from,
            so there is one implementation of edge incidence rather than one per
            consumer.
        """
        faces = self.faces()
        directed = np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
        )
        return np.unique(np.sort(directed, axis=1), axis=0)

    def boundary_edges(self, label: str | None = None) -> IntArray:
        """The edges incident to exactly one face, ``(B, 2)``.

        Directed rather than sorted: each is returned in the order its own face names
        it, so the interior lies to its left in the parameter plane and the outward
        normal is a rotation away. That is what :meth:`boundary_labels` reads.

        Parameters
        ----------
        label : str, optional
            One of :data:`BOUNDARY_LABELS`, to take only that part of the boundary.

        Returns
        -------
        IntArray
            Vertex index pairs.

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
        """The faces with at least one boundary edge, ``(b,)``, ascending.

        Parameters
        ----------
        label : str, optional
            One of :data:`BOUNDARY_LABELS`.

        Returns
        -------
        IntArray
            Face indices. What the propagation stage's edge search and the taper both
            want, from one implementation rather than three.

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

    def boundary_labels(self) -> np.ndarray:
        """Each boundary edge as ``top``, ``bottom`` or ``lateral``, ``(B,)``.

        Read straight off the parameter coordinates, which is one of the things storing
        ``(u, v)`` per vertex buys. For a positively oriented triangulation a boundary
        edge runs with the interior on its left, so its outward normal in the parameter
        plane is that direction turned a right angle clockwise. Whichever component of
        that normal dominates says which boundary it is: mostly ``-v`` is the top edge,
        mostly ``+v`` the bottom, and otherwise it runs down dip and is lateral.

        Dominance rather than an angle threshold, so there is no tolerance to justify:
        the two cases are separated by ``|n_v| = |n_u|``, which is the 45-degree
        diagonal and the only division that does not need a number chosen.

        Returns
        -------
        np.ndarray
            Strings, aligned with :meth:`boundary_edges`.
        """
        edges = self.boundary_edges()
        parameters = self.parameters_km()
        direction = parameters[edges[:, 1]] - parameters[edges[:, 0]]
        # Turn the edge direction a right angle clockwise: interior on the left means
        # this points out of the patch.
        outward_u, outward_v = direction[:, 1], -direction[:, 0]

        labels = np.full(len(edges), "lateral", dtype="<U7")
        along_dip = np.abs(outward_v) > np.abs(outward_u)
        labels[along_dip & (outward_v < 0.0)] = "top"
        labels[along_dip & (outward_v > 0.0)] = "bottom"
        return labels

    # ------------------------------------------------------- the hypocentre seam

    def cell_index(self, strike_km: float, dip_km: float) -> int:
        """The face containing an in-fault position, as one flat index.

        **The one narrow conversion seam** between the config's arc lengths and the
        pipeline's indices, and worth keeping narrow: a hypocentre one cell off in both
        directions correlates 0.99+ with the right answer while moving onsets by up to
        a second.

        The arguments are **true surface arc lengths**, not parameter coordinates,
        because that is what "12 km along strike" means. Each is inverted through
        :meth:`arc_profile` -- which is strictly increasing, so the inversion is exact
        -- and the query itself is then a point-in-triangle test in the parameter
        plane. On a planar patch the two coordinate systems are the same one and the
        inversion is the identity.

        This is not the SRF's ``shyp``, which is measured from the along-strike centre
        and converted by the SRF writer, and not a vertex index. A position on a shared
        edge belongs to the lower-numbered face, which is arbitrary but deterministic.

        Parameters
        ----------
        strike_km, dip_km : float
            Arc lengths from the patch's shallow near corner.

        Returns
        -------
        int
            A flat face index in ``[0, F)``.

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

        # Outside every face. Only round-off is forgiven, measured in kilometres so the
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
        answer is an ``(i, j)`` pair. A triangulation has one flat face index and no
        lattice position, so this is the identity -- and it exists so that
        :func:`~rupture_generator.propagation.causal_jump` can record a jump's cells
        without knowing which kind of chart it is looking at.

        Parameters
        ----------
        flat_index : int
            A face index in ``[0, F)``.

        Returns
        -------
        int
            ``flat_index``.
        """
        return int(flat_index)


# ============================================================================
# Admissibility -- what replaces `validate_chart`
# ============================================================================


def check_admissible(mesh: TriangleMesh) -> None:
    """Assert a segment is a Monge patch, and a mesh anything can be solved on.

    **Two claims, which fail in different ways and are both silent.**

    The first is about the projection rather than the surface: ``X -> (u, v)`` must be
    injective, or the chart describes two pieces of fault at one parameter point and
    every field drawn on it is doubly defined. On a triangulation that is exactly *every
    triangle is positively oriented in the parameter plane* -- a triangulation with no
    folds tiles its image once.

    The second is about the discretisation. No vertex may carry less than
    :data:`DEGENERATE_MASS_FRACTION` of the mesh's median lumped mass, because a vertex
    with negligible support is barely constrained by the SPDE operator, its marginal
    variance explodes, and ``sampling.standardise`` then divides the *whole* field by a
    standard deviation those few vertices dominate. On the CFM ``Puyseguer`` interface
    that suppresses the healthy slip distribution 5.3-fold while everything still looks
    like a slip distribution. That constant's docstring carries the measurement.

    **These degenerate faces are refused rather than dropped, and that is deliberate.**
    They arrive in the input file: they are the modeller's surface, not this package's
    artefact, and silently deleting faces from a surface someone supplied would change
    its area and its outline without saying so. (There is nothing left that this module
    drops on its own -- :meth:`TriangleMesh.from_patches` builds connectivity from the
    lattice and invents no triangles to cull.) Naming them and stopping lets the modeller
    remesh, or drop them deliberately and own the change.

    The sampler carries its own backstop for the same condition; this is the primary
    gate and does not assume the other one runs.

    No tolerance, deliberately. The quantity is the determinant of two parameter
    differences and the test is on its **sign**, which is what injectivity is; a
    zero-area face is a parameterisation that has collapsed and is refused on the same
    line rather than being waved through by a nearby epsilon.

    Measured here on the shipped examples at their own configured subfault sizes, as
    total trace turning against the worst ``|grad h|`` the surface carries:

    ============================  ======  =========  =========  ======  =======
    surface                       planes  turning    |grad h|   margin  cut
    ============================  ======  =========  =========  ======  =======
    ``beavan`` (7 faults)         1 each  0.0 deg    4.3e-14    1.000   0.1 km
    ``kaikoura`` (2 segments)     1 each  4.6 deg    1.7e-14    1.000   1.0 km
    ``colombia``                  1       0.0 deg    3.8e-13    1.000   0.1 km
    ``hope``                      2       19.7 deg   1.77e-01   0.984   1.0 km
    ``alpine_hope``: Caswell      7       43.8 deg   6.31e-01   0.844   0.25 km
    ``alpine_hope``: G to Jacksons  30    192.8 deg  2.25e-01   0.537   0.25 km
    ============================  ======  =========  =========  ======  =======

    ``Alpine: Caswell`` is the worst ``|grad h|`` of the 20 surfaces in
    ``alpine_hope``; ``Alpine: George to Jacksons`` is the worst fold margin. The
    numbers are re-measured by ``tests/triangular/test_trimesh.py`` -- the docstring is
    the record, the test is the measurement. ``alpine_hope`` is quoted at a 0.25 km cut
    rather than its shipped 0.1 km because the shipped cut is 2.1 million nodes and
    triangulating all of it needs about 3 GB; ``|grad h|`` came out **bit-identical**
    at 1.0, 0.25 and 0.1 km, which is what it should do, since it is a property of the
    planes' orientations rather than of how finely they were cut. The fold margin is
    not: it compares triangle areas, and at the shipped 0.1 km the same two surfaces
    give 0.836 and 0.537.

    So the shipped geometry reaches ``|grad h| = 0.63``, where true surface length
    exceeds parameter length by 18% and the projection is still comfortably injective.
    That is about twice MESH.md's recorded ``|grad h| <~ 0.33``, which was measured
    before this frame existed; the margin is still wide, and the surface that has the
    least of it is not the one that turns the most. ``Alpine: George to Jacksons``
    turns 193 degrees in total and is perfectly admissible, because its bends largely
    cancel. Total turning is a poor predictor and ``|grad h|`` is the real quantity,
    which is the other reason this check is the right home for the refusal.

    **Real subduction interfaces go far past that**, and the patch holds. The three NZ
    CFM v1.0 interfaces, read with their own connectivity by
    :mod:`rupture_generator.triangular.gocad`:

    ====================  ====  ====  =======  =====  =====  ======  ====  ========
    interface             V     F     bestdip  med    p90    max     inv   proj/true
    ====================  ====  ====  =======  =====  =====  ======  ====  ========
    Hikurangi             5218  9236  14.1     0.158  0.425  1.2142  **0**  0.636
    Puyseguer             2597  4090  21.2     0.101  0.881  2.1435  **0**  0.423
    Puysegur-Fiordland    2312  4041  22.8     0.132  0.771  1.9688  **0**  0.453
    ====================  ====  ====  =======  =====  =====  ======  ====  ========

    (``|grad h|`` quantiles area-weighted; the last column is the smallest ratio of
    projected to true face area.) So real geometry reaches 1.2 to 2.1, six times
    MESH.md's budget: the metric factor ``sqrt(1 + |grad h|^2)`` reaches 2.37, and
    parameter distance differs from true surface length by up to 137% *locally* against
    the 5% MESH.md sized against. Every one is still injective. The aggregate is far
    milder than the local worst -- end to end the strike extent exceeds its projection
    by 0.2 to 2.5% and the dip extent by 2.8 to 12.4% -- because the steep places are
    localised and :meth:`TriangleMesh.arc_profile` weights by area.

    ``Puyseguer`` is in that table but does not load: it fails the mesh-quality gate
    below. Its geometry is measured here through the same code with the gate lifted,
    because "the surface is fine and the discretisation is not" is exactly the
    distinction the two checks exist to draw.

    **Was this check vacuous? Yes, and it is not any more.** While the builder
    triangulated the projected points, `scipy.spatial.Delaunay` oriented every face
    positively by construction, so no surface could fail: on a synthetic fan of planes
    each turning 45 degrees, ``|grad h|`` climbed to 2.8 and *nothing inverted*. Now the
    connectivity comes from the surface -- the lattice in
    :meth:`TriangleMesh.from_patches`, the file's own faces in
    :meth:`TriangleMesh.from_triangulation` -- so orientation is a real question. The
    zeroes in the table above are therefore a measurement rather than a tautology, and
    ``tests/triangular/test_trimesh.py`` now builds a lattice that genuinely folds and
    watches this refuse it, which it could not do before.

    Two things it still does not catch, stated so nobody assumes otherwise. Patches that
    **overlap each other** in the parameter plane produce no inverted face, since each
    is individually fine; fusion already requires conforming planes that tile, so this
    is not reachable from a config, but it is not tested for either. And a frame that
    has stopped *meaning* strike and dip is not a fold: on that same 45-degree fan the
    best-fit plane rotates flat, the frame's own dip falling from 60 degrees at two
    planes to 0.015 at eight, and the patch remains a perfectly good graph over a plane
    that is no longer the fault's. The honest signal for that is ``|grad h|`` itself,
    which :meth:`TriangleMesh.maximum_slope` reports rather than refuses, because the
    CFM measurement above shows 2.0 is *normal* -- so any threshold would have to be
    argued from data nobody has, and reporting a number that can be looked at beats
    refusing against one that was invented.

    The refusal belongs here rather than in
    :data:`~rupture_generator.mesh.SHARPEST_BEND_DEG` because here it names the
    modelling assumption directly -- "this surface is a graph over a plane" -- rather
    than a per-bend proxy for it, and because it applies to *any* mesh, including one
    read from a file or refined by something that is not this builder.

    Parameters
    ----------
    mesh : TriangleMesh
        The segment to check.

    Raises
    ------
    ValueError
        For a fold, naming the worst face and its signed parameter area; or for a
        near-degenerate mesh, naming the starved vertices with their lumped mass and the
        faces around them with their quality. Both say what the caller can do.
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

    Parameters
    ----------
    mesh : TriangleMesh
        The segment to measure.

    Returns
    -------
    float
        ``min(signed area) / mean(signed area)``. One on a patch whose triangles are
        all the same size and orientation, zero at the fold, negative past it. A
        margin rather than a pass/fail, so a geometry that is nearly inadmissible can
        be *reported* before it stops being a patch at all.
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

    Not a refinement of the supplied triangulation -- a replacement for it. The parameter
    domain is sampled on a regular lattice at ``spacing_km``, every lattice node is lifted
    onto the source surface, and the connectivity comes from the lattice.

    **Why building beats subdividing, measured.** One-to-four subdivision splits each
    triangle into four *similar* ones, so it preserves element shape exactly: a badly
    shaped mesh stays badly shaped at every level, and the shape is what the multigrid
    sampler pays for. At matched vertex count:

    ==========================================  =======  ============  =======
    mesh                                        V        area max/min  draw
    ==========================================  =======  ============  =======
    built lattice                               263169   **1**         1.71 s
    CFM Hikurangi, subdivided 1-to-4            300345   4.28e+04      61.4 s
    ==========================================  =======  ============  =======

    A 36-fold penalty from shape alone, and it gets worse rather than better with
    refinement: V-cycle counts stay flat at 12 from 4 k to 4.2 M vertices on a built mesh
    against 43, 90, 173, 312 -- doubling per level -- on a subdivided one. That is the
    difference between 100 m being reachable and not, which is why this exists.

    **The interpolant is piecewise-linear on the source faces**, and that is a decision
    rather than a default. It is what the source surface *means*: a triangulation is
    exactly the claim that the surface is planar within each face, so evaluating that
    claim is the only reading of the data that adds nothing to it. Two consequences worth
    stating. Every new vertex lies **exactly on** the source surface, so the result is
    inscribed rather than approximated, and where the new lattice is finer than the source
    the new triangles are coplanar sub-triangles of source faces -- so ``|grad h|`` there
    is the source's *exactly*, and area is preserved exactly. The only departures are new
    triangles that straddle two source faces, which chord across the source's own kinks,
    and the boundary.

    **The boundary is not guessed.** The earlier negative result stands -- no alpha-shape
    threshold recovers this outline, interior circumradius 5.889-6.196 km overlapping
    gap-spanning from 5.941 -- but none is needed, because the source triangulation
    *carries* its outline. A lattice node is kept exactly when it lies inside some source
    face, which is simultaneously the outline test and the precondition for interpolating
    ``h`` there; a lattice quad is kept when all four of its corners are. So the outline
    is exact data, resolved onto the lattice at ``spacing_km``: the boundary becomes a
    staircase whose deviation is bounded by one cell, and the area it costs is
    ``O(spacing x perimeter)`` and is reported by the tests rather than assumed small.

    **Element shape is then exact in the parameter plane and bounded in space.** Every
    face is half a lattice cell, so in projection all faces are congruent: measured on
    full Hikurangi at 100 m, parameter area max/min is ``1.000000000011``. What remains in
    *three* dimensions is the metric factor and nothing else -- the 3-D area spread came
    out ``1.5742`` against ``sqrt(1 + |grad h|max^2) = 1.5742``, agreeing to four figures,
    so the residual spread is the surface's own curvature rather than the mesh's shape,
    and no mesh over this surface can do better.

    Measured against the source, which is the comparison that matters:

    ==================  ========  ==============  =========  ===========  ======
    Hikurangi           V         param max/min   3-D ratio  min angle    area
    ==================  ========  ==============  =========  ===========  ======
    CFM source          5218      --              4.28e+04   0.018 deg    --
    built at 2 km       44049     1.0             1.553      31.6 deg     -1.49%
    built at 1 km       175988    1.0             1.565       31.1 deg    -0.75%
    built at 400 m      1100240   1.0             1.571       31.0 deg    -0.30%
    built at 100 m      17598707  1.0             1.574       31.0 deg    -0.075%
    ==================  ========  ==============  =========  ===========  ======

    Four orders of magnitude off the source's area ratio, and the minimum angle goes from
    0.018 degrees to 31. The area deficit is the boundary staircase and it **halves with
    the spacing**, exactly the first order the argument above predicts, so it is a
    resolution knob rather than a bias. ``|grad h|`` converges to the source's from below
    as the chording tightens: 1.006, 1.104, 1.188, 1.204, 1.212, 1.216 against the
    source's 1.2142.

    Cost, measured, since the point of this is reaching 100 m: 3.6 s and 0.9 GB at 400 m,
    **35.7 s and 10.8 GB at 100 m** for 17.6 M vertices and 35.2 M faces. The
    lattice-shaped intermediates dominate, and the peak is about 600 bytes per vertex.
    ``Puyseguer`` at 400 m is 397 k vertices in 1.1 s, and remeshing repairs it: its
    lumped-mass ratio goes from 7.3e-07, which :data:`DEGENERATE_MASS_FRACTION` refuses,
    to 1/6.

    Parameters
    ----------
    vertices_km : FloatArray
        ``(V, 3)`` source positions, offsets from the surface origin.
    faces : IntArray
        ``(F, 3)`` source connectivity. It need not pass :func:`check_admissible` -- a
        source whose elements are unusable is the case this exists for -- but it must be
        a graph over its own best-fit plane, since that is what makes the parameter
        lattice well defined.
    spacing_km : float
        The target edge length. A *request*, in the sense
        :func:`rupture_generator.mesh.cell_counts` uses: the parameter extents are cut
        into whole cells, so the realised spacing is the extent over the count. That is
        what makes the lattice land on the domain's corners, and it is what makes this
        reduce exactly to :meth:`TriangleMesh.from_patches` on a planar fault.
    strike_deg, dip_deg : float
        The stated geometry of the source.
    dips_left : bool, optional
        Whether the source dips left of the strike direction.
    origin_east_km, origin_north_km : float, optional
        The surface origin, in the CRS, kilometres.
    surface : str
        The name the result carries.

    Returns
    -------
    TriangleMesh
        Admissible, all-congruent, at the realised spacing.

    Raises
    ------
    ValueError
        For a source this cannot read (see :func:`_fit_surface`), a non-positive
        spacing, or a spacing so coarse that no lattice cell falls entirely inside the
        surface -- which names the extents so the caller can see by how much.
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

    # One pass per source face, scattering into the lattice rather than searching the
    # lattice for each face: F is thousands while the lattice is millions, and a face
    # only ever touches the nodes in its own bounding box.
    for index in range(len(connectivity)):
        if determinant[index] == 0.0:
            continue
        # `floor` and `ceil` the wrong way round on purpose, widening the candidate
        # block by up to one node on each side. A source face's extent is very often
        # *exactly* a lattice coordinate -- always, on the planar fault this has to
        # reduce to -- and then the tight bound is an integer that round-off pushes to
        # either side, silently dropping the whole boundary ring. The barycentric test
        # below rejects whatever is genuinely outside, so widening costs work and
        # nothing else.
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
        # A node on a shared edge is inside both faces and both give the same height,
        # so overwriting is harmless; the tolerance is a *length* rather than a
        # barycentric fraction, so it means the same thing on every face shape.
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


def padded_builder(
    mesh: TriangleMesh, *, pad_spacing_km: float | None = None
) -> Callable[[float, float], tuple[FloatArray, IntArray, FloatArray, IntArray]]:
    """A callable that rebuilds a segment with its parameter domain extended outwards.

    What the SPDE sampler needs and the shape it needs it in: the sampler never imports
    this module, so the padding arrives as a function it can call. The returned callable
    takes the pad widths and hands back plain arrays.

    **Why the pad exists.** The SPDE's Neumann boundary condition reflects the covariance
    in the domain boundary (Lindgren et al. appendix A.4), and at fault scale that is not
    an edge effect: Mai & Beroza figure 13's own 0.25-0.6 ratio bound puts a fault between
    1.7 and 4 correlation lengths across *by construction*, so a domain cropped to the
    fault has the reflection everywhere. The circulant sampler never suffered it because
    it pads and crops; this is the same remedy.

    **The pad may be coarser than the fault, and usually should be.** Its only job is to
    move the boundary away, so resolving it at fault resolution buys nothing and costs a
    great deal -- a uniform pad two correlation lengths wide multiplies the vertex count
    by roughly five, which at 100 m puts a crustal segment into the millions for no
    modelling gain. ``pad_spacing_km`` defaults to the fault's own realised spacing, and
    is worth setting larger.

    **What the patch means out there.** ``h(u, v)`` is defined by the fault's own nodes, so
    the pad is an *extrapolation of the reference surface*, and this uses the flattest one
    available: ``h`` is held at the value of the nearest fault boundary node along each
    outward direction, which is continuous across the seam and adds no curvature. Both
    properties are load-bearing rather than tidy. Continuity, because the sampler
    assembles from the **lifted** triangles and a jump in ``h`` at the seam is a crease the
    cotangent Laplacian reads as real geometry -- which would move the artefact the pad
    exists to remove onto the fault edge instead of away from it. And no added curvature,
    because :func:`check_admissible` applies to the padded mesh too: at the ``|grad h| =
    2.14`` a real interface reaches, a tangent-plane extension is already 65 degrees off
    the reference plane and anything continuing the *curvature* would turn past vertical.

    Parameters
    ----------
    mesh : TriangleMesh
        The fault, already built. Its frame and parameter coordinates are kept exactly, so
        the fault's own ``(u, v)`` do not move when the pad is added -- otherwise every
        stored coordinate shifts and the hypocentre seam moves silently.
    pad_spacing_km : float, optional
        Lattice spacing in the pad. Defaults to the fault's median edge length in the
        parameter plane.

    Returns
    -------
    callable
        ``build(pad_strike_km, pad_dip_km)`` returning ``(vertices_km, faces,
        parameters_uv, fault_faces)``: positions ``(V, 3)``, connectivity ``(F, 3)``,
        parameter coordinates ``(V, 2)``, and the indices of the faces that are **on the
        fault** -- which is what the sampler crops back to. Zero pad widths give the
        fault's own arrays unchanged.
    """
    frame = mesh.frame
    fault_uv = mesh.parameters_km()
    fault_vertices = mesh.vertices_km()
    fault_faces = mesh.faces()
    fault_height = frame.project(fault_vertices)[:, 2]

    edges = mesh.edges()
    default_spacing_km = float(
        np.median(
            np.linalg.norm(fault_uv[edges[:, 0]] - fault_uv[edges[:, 1]], axis=-1)
        )
    )
    spacing_km = pad_spacing_km if pad_spacing_km is not None else default_spacing_km

    def build(
        pad_strike_km: float, pad_dip_km: float
    ) -> tuple[FloatArray, IntArray, FloatArray, IntArray]:
        """Extend the domain by the given pad widths and return plain arrays.

        Parameters
        ----------
        pad_strike_km, pad_dip_km : float
            How far to extend past the fault along each parameter axis, on both sides.

        Returns
        -------
        tuple
            ``(vertices_km, faces, parameters_uv, fault_faces)``.

        Raises
        ------
        ValueError
            For a negative pad width, or a pad the spacing cannot resolve at all.
        """
        if pad_strike_km < 0.0 or pad_dip_km < 0.0:
            raise ValueError(
                f"the pad is {pad_strike_km} by {pad_dip_km} km; a negative pad is not "
                "a smaller domain, it is a crop, and cropping is the sampler's own step"
            )
        if pad_strike_km == 0.0 and pad_dip_km == 0.0:
            return (
                fault_vertices,
                fault_faces,
                fault_uv,
                np.arange(len(fault_faces), dtype=np.int64),
            )

        low = fault_uv.min(axis=0) - np.array([pad_strike_km, pad_dip_km])
        high = fault_uv.max(axis=0) + np.array([pad_strike_km, pad_dip_km])
        counts = np.maximum(np.round((high - low) / spacing_km).astype(np.int64), 1)
        grid_u = np.linspace(low[0], high[0], counts[0] + 1)
        grid_v = np.linspace(low[1], high[1], counts[1] + 1)
        lattice_v, lattice_u = np.meshgrid(grid_v, grid_u, indexing="ij")

        # Flat extrapolation: hold `h` at the nearest fault vertex. Nearest-neighbour
        # rather than anything smoother precisely because it adds no curvature, and it
        # is continuous where it matters -- at the seam the nearest fault vertex *is*
        # the boundary node, so the pad meets the fault at the fault's own height.
        query = np.stack([lattice_u.ravel(), lattice_v.ravel()], axis=-1)
        nearest = KDTree(fault_uv).query(query, k=1)[1]
        height = fault_height[nearest].reshape(lattice_u.shape)

        # Everything strictly inside the fault's parameter bounding box is the fault's
        # own business; the pad is the frame around it. Cells wholly inside the box are
        # dropped and the fault's own vertices spliced in instead, so the fault keeps
        # exactly the mesh it was built with.
        inside = (
            (lattice_u > fault_uv[:, 0].min())
            & (lattice_u < fault_uv[:, 0].max())
            & (lattice_v > fault_uv[:, 1].min())
            & (lattice_v < fault_uv[:, 1].max())
        )
        drop = inside[:-1, :-1] & inside[:-1, 1:] & inside[1:, 1:] & inside[1:, :-1]
        keep = ~drop

        used = np.zeros(lattice_u.shape, dtype=bool)
        for row, column in ((0, 0), (0, 1), (1, 1), (1, 0)):
            used[row : row + keep.shape[0], column : column + keep.shape[1]] |= keep
        numbering = np.full(used.shape, -1, dtype=np.int64)
        numbering[used] = np.arange(int(used.sum())) + len(fault_uv)

        pad_uv = np.stack([lattice_u[used], lattice_v[used]], axis=-1)
        pad_vertices = frame.lift(
            np.stack([pad_uv[:, 0], pad_uv[:, 1], height[used]], axis=-1)
        )

        near = numbering[:-1, :-1][keep]
        far = numbering[:-1, 1:][keep]
        opposite = numbering[1:, 1:][keep]
        beside = numbering[1:, :-1][keep]
        pad_faces = np.stack(
            [
                np.stack([near, far, opposite], axis=-1),
                np.stack([near, opposite, beside], axis=-1),
            ],
            axis=1,
        ).reshape(-1, 3)

        return (
            np.concatenate([fault_vertices, pad_vertices]),
            np.concatenate([fault_faces, pad_faces]),
            np.concatenate([fault_uv, pad_uv]),
            np.arange(len(fault_faces), dtype=np.int64),
        )

    return build


def _fit_surface(
    vertices_km: FloatArray,
    faces: IntArray,
    *,
    strike_deg: float,
    dip_deg: float,
    dips_left: bool = False,
) -> tuple[FloatArray, IntArray, MongeFrame]:
    """Validate a supplied triangulation, fit its frame, and settle its winding.

    Shared by :meth:`TriangleMesh.from_triangulation` and :func:`remesh`, which need the
    same front half for different back halves -- and, importantly, ``remesh`` needs it on
    surfaces that :func:`check_admissible` *refuses*, since repairing those is the whole
    reason it exists.

    Returns
    -------
    tuple
        The vertices as float64, the faces wound to agree with the frame, and the frame
        with its origin already moved to the parameter domain's low corner.

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

    # Which way a modeller wound their triangles is a convention, not a fact about the
    # surface: it says whether they numbered each face anticlockwise seen from the
    # hanging wall or from the footwall. All three CFM interfaces are wound the opposite
    # way to this frame, and reading that as 9236 folds would be reading a file-format
    # convention as geology. What is *not* a convention is whether the faces agree with
    # each other -- that is exactly injectivity -- so the total signed area fixes the
    # convention (it is plus or minus the true area whenever the projection is
    # injective, so its sign is unambiguous and needs no tolerance) and
    # `check_admissible` then refuses every face that disagrees.
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

    Returns the three edge midpoints of every face weighted by a third of its area.
    That rule integrates any **quadratic** over a triangle exactly, and the second
    moment a least-squares plane fit needs is quadratic, so the fit is the exact
    continuous one: ``integral (x - c)(x - c)^T dA`` over the surface.

    Exactness is the point, not elegance. Fitting to the *vertices* -- even weighted by
    the area each carries -- is a fit to the sample: refine a segment's planes unevenly
    and the plane tilts, so ``|grad h|`` picks up a dependence on the subfault size that
    it has no business having. Measured on ``alpine_hope``'s first surface across a
    factor of four in cell size: the vertex fit moved ``|grad h|`` by 4e-4 relative,
    this one leaves it identical to 1e-15, because subdividing a planar triangle does
    not change any integral over it.
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
    """How much surface each vertex carries: a third of every triangle it touches.

    The barycentric-dual area, and the weight that turns :meth:`MongeFrame.fit` from a
    fit to the *sample* into a fit to the *surface*. Summing to the total area exactly
    is what makes it a partition rather than an estimate.
    """
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

    What the config said, read back out of the geometry it produced: the trace bearing
    is the top edge's chord and the dip is the near column's plunge. Exact for a chart
    this package built, which is what makes reading a version 2 file give the same
    frame the config would have.
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

    The compatibility path: a version 2 mesh file holds a ``(dip_node, strike_node)``
    lattice and nothing else, and this is what turns one into a segment. A fused chart
    is split back into its constant-plane blocks first, so the seam column is shared
    exactly as it was written and ``plane_of_face`` survives the round trip.

    Parameters
    ----------
    chart : RuptureMesh
        A structured chart, fused or not.

    Returns
    -------
    TriangleMesh
        With the frame the config's own numbers would have produced -- see
        :func:`_stated_geometry`.
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

    The trace, the bends and the seam sharing are
    :func:`rupture_generator.mesh.build_fault`'s and :func:`rupture_generator.mesh.fuse`'s
    -- this is the same geometry, retriangulated. A *fused segment* is the unit rather
    than a fault or a plane, because that is the largest run of planes that is one
    surface, and a fault whose planes hang differently is two surfaces whether or not
    they touch.

    The frame's stated strike is the segment's top-edge **chord**, first trace point to
    last. On a single-plane segment that is the plane's own bearing exactly, which is
    what makes the planar collapse exact; on a bent one it is the direction the segment
    goes overall, and every departure from it is carried by ``h``. Dip and dip
    direction come from the segment's first plane, and fusion has already required
    every plane in the segment to state the same ones.

    Parameters
    ----------
    fault : FaultConfig
        The digitised geometry.
    crs : pyproj.CRS
        The projected CRS to build in.

    Returns
    -------
    list of TriangleMesh
        One per segment, in trace order.

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
    """A point source as two triangles.

    A point source is the pipeline with constant fields, not a special type, so it is
    an ordinary one-quad patch -- and its strike and dip are the only ones a config
    states outright rather than implying through a trace, which makes it the cleanest
    test of the frame.

    Parameters
    ----------
    point : PointConfig
        The catalogue entry.
    crs : pyproj.CRS
        The projected CRS to build in.

    Returns
    -------
    list of TriangleMesh
        One mesh, of two faces.
    """
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
    """Discretise one surface: the dispatch a triangular mesh CLI would call.

    Parameters
    ----------
    surface : FaultConfig or PointConfig
        What the geometry file said.
    crs : pyproj.CRS
        The projected CRS to build in.

    Returns
    -------
    list of TriangleMesh
        One per segment.
    """
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
    """Lay segments out as a tree: one group per segment, nested under its surface.

    Only the geometry is stored -- vertices, faces, the parameter coordinates and the
    frame. Centres, areas, strike, dip and both arc lengths are functions of those, so
    they are computed on read and never written.

    Parameters
    ----------
    meshes : Mapping of str to list of TriangleMesh
        Surface name to its segments, in trace order.
    crs : pyproj.CRS
        The frame every position is in. Stored once, in the root.
    attrs : Mapping, optional
        Extra root attributes -- the config verbatim, a title.

    Returns
    -------
    xr.DataTree
        With ``/<surface>/segment_<n>`` groups.
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
        # One origin per surface, as JSON because an attribute is a scalar or an
        # array and this is a mapping.
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
    :func:`rupture_generator.formats.mesh.from_datatree` and triangulated on the way in
    by :func:`from_chart` -- the nodes are the geometry, so the surface comes back
    identical and only the connectivity is new.

    Parameters
    ----------
    tree : xr.DataTree
        What :func:`to_datatree` wrote, or what version 2 wrote.

    Returns
    -------
    tuple
        Surface name to its segments, and the CRS they are in.

    Raises
    ------
    ValueError
        If the tree carries no CRS, a surface has no recorded origin, or a surface's
        segments are numbered with a gap -- a gap means a segment is missing rather
        than renumbered.
    """
    version = int(tree.attrs.get("schema_version", 1))
    if version < SCHEMA_VERSION:
        # A local import: `formats.mesh` is the structured track's file seam, and this
        # is the one place the triangular track reads from it. Keeping it here means
        # `triangular.mesh` can be imported without dragging in the older format.
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

    # Keyed by the *stored* segment index rather than by the order the groups come back
    # in, because Zarr does not preserve order and HDF5 does -- trusting iteration
    # order is green in one container and silently permutes the fault in the other.
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

    Parameters
    ----------
    meshes : Mapping of str to list of TriangleMesh
        Surface name to its segments.
    crs : pyproj.CRS
        The frame every position is in.
    path : Path or str
        Where to write it.
    format : Format, optional
        Which layout; inferred from the extension by default.
    attrs : Mapping, optional
        Extra root attributes.

    Raises
    ------
    ValueError
        If the format is not one a mesh can be written in. An SRF holds a rupture, not
        a surface, and there is nothing sensible to put in its slip columns.
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

    Parameters
    ----------
    path : Path or str
        The file or store.
    format : Format, optional
        Which layout; inferred from the extension by default.

    Returns
    -------
    tuple
        Surface name to its segments, and the CRS.

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
    "padded_builder",
    "read_mesh",
    "remesh",
    "stated_axes",
    "to_datatree",
    "write_mesh",
]
