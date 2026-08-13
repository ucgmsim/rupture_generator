"""Does the geometry change between 500 m and 400 m? Measured, not asserted.

The study runs at 500 m rather than at the 400 m the machinery reaches, and the
justification is a claim about where the *geometry's* information lives rather than
about what the machine can hold. :func:`~rupture_generator.triangular.mesh.remesh` lifts
each lattice node onto the source surface by piecewise-linear interpolation **on the
source faces**, so wherever the built mesh is finer than the CFM triangulation the new
triangles are coplanar sub-triangles of source faces. On those, ``h``, ``grad h`` and
area are the source's *exactly* rather than approximately. The CFM Hikurangi interface
has a median vertex spacing near 5.6 km, so 500 m oversamples it elevenfold and the only
faces that can differ between two such meshes are the ones straddling a source face's
edge, which chord across the source's own kinks, plus the boundary staircase.

That is an argument. This module is the measurement, run across three resolutions:

- the **total area ratio** true/projected, which is the number the moment error rests on;
- the **distribution of** ``|grad h|``, which is what bounds every departure from
  flatness, quoted at percentiles rather than by its maximum -- a maximum over two
  million faces is an order statistic and would move between resolutions even if the
  distribution did not;
- the identity ``area_curved / area_flat = sqrt(1 + |grad h|^2)``, which is exact for a
  Monge patch and is computed here by two unrelated routes.

What 500 m does *not* give is the 400-500 m octave of **slip** heterogeneity. That is a
real loss and it is stated rather than folded into the geometric argument, because the
slip field's correlation length is 21.5 km down dip and its structure continues below the
geometry's own resolution.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from curvature.geometry import COARSE_SPACING_KM, build_pair

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

SUBDIVISION_LADDER = tuple(
    (
        f"{COARSE_SPACING_KM / 2**levels:g} km nominal",
        COARSE_SPACING_KM,
        levels,
    )
    for levels in range(4)
)
"""Resolutions reached by subdividing **one** base mesh, as ``(label, spacing, levels)``.

Derived from :data:`~curvature.geometry.COARSE_SPACING_KM` rather than written out, so
changing the study's base spacing cannot leave this ladder describing a different mesh
than its labels claim. The study is the third rung; the fourth is one refinement past it,
which is what brackets the claim from below.

This is the ladder the study's own hierarchy sits on, and the answer it gives is
stronger than convergence: the geometry is *exactly* invariant, because
:func:`~rupture_generator.triangular.spde.subdivided` places each new vertex at an edge
midpoint of a coarse face, which lies **in that face's plane**. So ``h`` stays affine
over every sub-triangle, ``grad h`` is the parent's to round-off, and the areas sum to
the parent's. Refining this way cannot change the surface; it changes only how finely the
surface is sampled.
"""

BUILD_LADDER = (8.0, 4.0, 2.0, 1.0, 0.5)
"""Base spacings the mesh is **built** at, in kilometres, with no subdivision.

The comparison that is not a tautology. Building at a different spacing resamples the
source surface: the boundary becomes a finer staircase, and triangles that straddle two
source faces chord across the source's own kinks instead of lying in one of them. Both
effects are ``O(spacing)``, so this ladder is where any real resolution dependence in the
geometry has to show up -- and where the claim that 500 m and 400 m give the same answer
is either earned or refuted.
"""

PERCENTILES = (50.0, 90.0, 99.0, 99.9)
"""Where ``|grad h|`` is read. Percentiles rather than the maximum, which over two
million faces is an order statistic of the tail rather than a property of the surface."""


def survey() -> dict:
    """Run both ladders and report the geometry at every rung.

    Returns
    -------
    dict
        ``by_subdivision`` and ``by_build_spacing``, each keyed by resolution label. The
        first shows that refining a mesh cannot move the surface; the second shows how
        far the *built* mesh's geometry still depends on the spacing it was built at,
        which is the question 500 m against 400 m actually asks.
    """
    return {
        "by_subdivision": _rungs(SUBDIVISION_LADDER),
        "by_build_spacing": _rungs(
            [(f"built {km:g} km", km, 0) for km in BUILD_LADDER]
        ),
    }


def _rungs(ladder: Sequence[tuple[str, float, int]]) -> dict:
    """Measure the geometry of each ``(label, spacing_km, levels)`` in a ladder.

    Parameters
    ----------
    ladder : sequence
        Triples of label, base spacing in kilometres, and subdivision count.

    Returns
    -------
    dict
        Keyed by label, with the vertex and face counts, the achieved median edge
        length, the total areas and their ratio, the ``|grad h|`` percentiles, and the
        worst disagreement between the two routes to the metric factor.
    """
    survey: dict[str, dict] = {}
    for label, spacing_km, levels in ladder:
        pair = build_pair(spacing_km=spacing_km, levels=levels)
        curved_km2 = pair.areas_km2(pair.curved_km)
        flat_km2 = pair.areas_km2(pair.flat_km)
        slope = np.linalg.norm(pair.slopes(), axis=-1)
        metric = np.sqrt(1.0 + slope**2)
        survey[label] = {
            "vertices": pair.vertex_count,
            "faces": pair.face_count,
            "median_edge_km": pair.median_edge_km,
            "area_curved_km2": float(curved_km2.sum()),
            "area_flat_km2": float(flat_km2.sum()),
            "area_ratio_true_over_projected": float(curved_km2.sum() / flat_km2.sum()),
            "displacement_h_min_km": float(pair.displacement_km.min()),
            "displacement_h_max_km": float(pair.displacement_km.max()),
            "displacement_h_rms_km": float(pair.displacement_km.std()),
            "slope_grad_h_percentiles": {
                f"p{value:g}": float(np.percentile(slope, value))
                for value in PERCENTILES
            },
            "slope_grad_h_max": float(slope.max()),
            "slope_grad_h_area_weighted_mean": float(
                np.average(slope, weights=curved_km2)
            ),
            # The two routes to the same number: a cross product in three dimensions
            # against a two-by-two solve in the parameter plane. Agreement is evidence
            # that the flat twin is the curved mesh's own orthogonal projection.
            "metric_factor_identity_max_relative_error": float(
                np.abs(curved_km2 / flat_km2 / metric - 1.0).max()
            ),
        }
        del pair
    return survey
