"""The experiment: two interfaces, three conditions, one set of numbers out.

Run as ``uv run python -m curvature.run`` for Hikurangi, and
``uv run python -m curvature.run puysegur_fiordland`` or ``... puyseguer`` for the two
Puysegur surfaces. Each invocation writes its own rasters -- ``curvature/data/arrays.npz``
for Hikurangi, ``curvature/data/<interface>.npz`` otherwise -- and **merges** its groups
into ``curvature/results.json``, leaving every group it did not produce untouched. So the
three runs can be made in any order and repeated singly, and a rerun of one interface
cannot silently drop another's numbers. Nothing here plots.

A second argument is the magnitude: ``uv run python -m curvature.run hikurangi 9.11``.
It defaults to :data:`~curvature.model.MAGNITUDE`, and everything else the run reads --
the mesh, the seeds, the hypocentres, the sample interval, the six rows and the
counterfactual -- is fixed, so the magnitude is the only difference between two runs and
the second is a **control on magnitude dependence** rather than a second experiment. The
prediction it tests is that the travel-time differences and the moment's area term do not
move: the eikonal solver reads geometry and the velocity model and never the event, and
the area ratio is a property of the two surfaces. What is expected to move is everything
the event sizes -- the correlation lengths, the slip, the rise times and the corner
frequency. :func:`merged` and :func:`tag` are what keep the two magnitudes' outputs
beside each other rather than one on top of the other.

One process per interface, because :func:`peak_memory_gb` reads a high-water mark: two
interfaces in one process would report the larger one's peak for both, which is a bound
rather than a measurement.

The run matrix
--------------

======================  ==========  ================  =====================================
scenario                hypocentre  velocity model    what it isolates
======================  ==========  ================  =====================================
``central_constant``    50%         constant          pure geometry: area, path length,
                                                      correlation structure
``central_standard``    50%         standard 1-D      geometry **and** depth
``northern_standard``   20%         standard 1-D      spatial variability
``southern_standard``   80%         standard 1-D      spatial variability
``northern_constant``   20%         constant          geometry alone, at the north end
``southern_constant``   80%         constant          geometry alone, at the south end
======================  ==========  ================  =====================================

The first two rows are the controlled pair. Same hypocentre, same white noise, same mesh
pair; **only the velocity model changes**, and with it whether anything in the run reads
depth at all. Their difference is the depth-driven contribution, measured rather than
argued. :mod:`curvature.model` documents exactly what the constant setting neutralises
and how.

Three hypocentres rather than more, because they are for spatial variability rather than
for statistics: one realisation cannot support a distribution over hypocentres, and three
along-strike positions at a fixed down-dip position show whether the effect is uniform
along the interface, which is the question a fourth would not answer better. The seeds are
shared across all six rows, so they are not independent samples and are not treated as
any.

The last two rows are the controlled pair repeated at the other two hypocentres, and they
turn a **single-hypocentre attribution into a spatial one**. With a control at the centre
only, the standard rows at the ends report a total and the split is not readable there:
the three totals vary along strike by 4.3 s on Hikurangi and by 3.7 to 7.8 s on Puysegur,
and which mechanism carries that variation is exactly the question the control exists to
answer. Asking it at one hypocentre and assuming the answer at the other two would be
assuming the thing under test. Every surface now runs all six rows, which is what makes
"does the geometric term move with the hypocentre, or only the depth term?" a measurement
rather than an inference from the middle of the fault.

:attr:`Interface.decomposed_sites` still exists rather than being folded away, because it
is what a rerun of a subset would be configured through and because it is the honest
statement of which hypocentres a given group's numbers were split at.

The third condition
-------------------

:func:`true_depth_report` adds a row the rows above cannot supply. The flat model is
wrong about two separable things -- **where its subfaults are** and **what rock they are
told they are in** -- and the run matrix only brackets them: the constant control has
neither error and the standard row has both. The counterfactual has the first and not the
second, and it is what a refactor that assigned material properties from the mesh
geometry rather than from the rupture sampler would deliver:

=================================  =============================================
condition                          isolates
=================================  =============================================
constant velocity                  geometry alone
standard, **true-depth** materials geometry + correct materials
standard, flat-depth materials     geometry + wrong materials -- the status quo
=================================  =============================================

The gap between the two standard rows is what the refactor is worth. The gap between the
true-depth row and the curved model is what it cannot buy, because the paths, the areas
and the metric on the correlated field are still the plane's.

The counterfactual runs at every hypocentre in :attr:`Interface.decomposed_sites`, beside
the constant control it is there to be compared against. The two are different models --
one has no depth anywhere, the other has the interface's depth everywhere -- and they
agree on Hikurangi's central hypocentre to 0.10 s out of a 7.53 s standard error, which
is what makes the cheap control a usable stand-in for the expensive refactor. That
agreement is a measurement rather than an identity, so it is measured again at every
hypocentre the control is run at rather than carried across;
:func:`by_site` reports the gap beside the error it has to be small against.
"""

from __future__ import annotations

import dataclasses
import json
import resource
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np

from curvature import analysis, model, resolution
from curvature.geometry import (
    COARSE_SPACING_KM,
    HIKURANGI,
    PUYSEGUR,
    PUYSEGUR_FIORDLAND,
    SUBDIVISIONS,
    MeshPair,
    build_pair,
    same_white_noise,
)
from rupture_generator import timing
from rupture_generator.moment import (
    MAGNITUDE_COEFFICIENT,
    cumulative_moment,
    seismic_moment_nm,
)
from rupture_generator.sampling import (
    MAI_MAXIMUM_RATIO,
    correlation_lengths,
    standardise,
)
from rupture_generator.timing import alpha_t
from rupture_generator.triangular import fim
from rupture_generator.triangular.fim import DegradedSeed
from rupture_generator.triangular.spde import BOUNDARY_FOLDING_LENGTHS

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

HERE = Path(__file__).resolve().parent
RASTER_SPACING_KM = 0.5
"""The parameter-plane cell size every map and every correlation estimate uses.

The mesh was built by sampling the parameter plane at 2 km and subdivided twice, so its
own parameter spacing is 0.5 km and each cell collects the two triangles of one lattice
quad. Matching it is what keeps the raster close to lossless.
"""

DIP_POSITION_KM = 180.0
"""Where along dip every Hikurangi hypocentre sits, in parameter kilometres.

One value for all three, so the along-strike comparison is not confounded by depth. It is
the shallowest down-dip position present at all three along-strike locations -- the
interface's south-western end has no shallow part in this triangulation -- and it puts the
hypocentre at 16 to 20 km true depth, inside the seismogenic zone.
"""

PUYSEGUR_DIP_POSITION_KM = 70.0
"""The same choice on Puysegur, whose down-dip parameter extent is 173 km against 332.

Chosen by the criterion rather than by scaling Hikurangi's number: the shallowest
down-dip position present at all three along-strike locations that puts every one of the
three hypocentres in the seismogenic zone. At 70 km the true depths are 17.4, 25.0 and
15.1 km on :data:`~curvature.geometry.PUYSEGUR_FIORDLAND` and 16.2, 24.5 and 16.6 km on
:data:`~curvature.geometry.PUYSEGUR`, against Hikurangi's 19.4, 19.6 and 16.0 km. The
central hypocentre is the deep one here because this interface is deepest mid-strike,
which is the geometry rather than the choice.
"""

STRIKE_FRACTIONS = {"northern": 0.2, "central": 0.5, "southern": 0.8}
"""Where along strike each hypocentre sits, as a fraction of the parameter extent.

``u`` increases south-west along Hikurangi's frame strike of 218.5 degrees, so
``northern`` is the East Cape end and ``southern`` the Marlborough end. On Puysegur the
frame strike is 16.4 degrees and ``u`` increases north-east, so ``northern`` is the
Puysegur Trench's southern end and ``southern`` the Fiordland end -- the names are
positions in the parameter plane, and :func:`hypocentres` records the real geography in
longitude and latitude so the two cannot be confused.
"""

SPECTRAL_BAND_HZ = (0.02, 0.5)
"""Where the high-frequency falloff exponent is fitted, in hertz.

Above the corner -- an Mw 8.5 whole-interface rupture corners near 0.01 Hz -- and below
the 6 Hz where a 500 m subfault crossed in 0.167 s stops resolving anything. A single
realisation's spectrum is not a straight line, so the band is quoted with the number.
"""

MAGNITUDE_LADDER = (7.5, 8.0, 8.5, 9.0, 9.5)
"""Magnitudes the interface is tested against before one is chosen.

The choice is not free: the magnitude sets the correlation lengths, and a domain has two
different ways of being too small for them. Mai & Beroza (2002) figure 13 bounds the
correlation length at :data:`~rupture_generator.sampling.MAI_MAXIMUM_RATIO` of the source
dimension, past which the relations are being extrapolated; and Lindgren et al. (2011)
appendix A.4 bounds how close the field may come to the boundary before the SPDE's
Neumann folding *is* the field, which
:func:`~rupture_generator.triangular.spde.boundary_folds` reads as
``2 * BOUNDARY_FOLDING_LENGTHS`` correlation lengths across the narrower axis.
:func:`magnitude_ladder` measures both at every rung so the chosen magnitude is a
reported pass rather than an assertion.
"""


@dataclasses.dataclass(frozen=True)
class Scenario:
    """One row of the run matrix."""

    name: str
    site: str
    velocity: model.VelocityModel


BASE_SCENARIOS = (
    Scenario("central_constant", "central", model.CONSTANT),
    Scenario("central_standard", "central", model.STANDARD),
    Scenario("northern_standard", "northern", model.STANDARD),
    Scenario("southern_standard", "southern", model.STANDARD),
)
"""The four rows every interface runs, in the order ``results.json`` has always had them.

Appended to rather than rebuilt, so an interface that decomposes at more than one
hypocentre gains rows at the end and the four a published document reads keep their
names, their order and their numbers.
"""


@dataclasses.dataclass(frozen=True)
class Interface:
    """One subduction interface, and the study choices that are its own.

    Attributes
    ----------
    name : str
        Also the stem of this interface's raster file, and the key its groups take in
        ``results.json``.
    path : Path
        The GOCAD TSurf to build the mesh pair on.
    dip_position_km : float
        Where along dip the three hypocentres sit, in parameter kilometres.
    decomposed_sites : tuple of str, optional
        Where the flat model's error is split into its geometric and depth parts, rather
        than only reported whole. Each named site gets a constant-velocity control and a
        true-depth counterfactual beside its standard row, which is the complete
        three-way decomposition at that hypocentre. Every interface the study runs names
        all three; the default is the centre alone, which is the smallest run that still
        supports the attribution the module docstring describes.
    """

    name: str
    path: Path
    dip_position_km: float
    decomposed_sites: tuple[str, ...] = ("central",)

    @property
    def scenarios(self) -> tuple[Scenario, ...]:
        """Every row of this interface's run matrix.

        Returns
        -------
        tuple of Scenario
            :data:`BASE_SCENARIOS`, then a constant-velocity control at each decomposed
            site the base rows do not already carry one for.
        """
        already = {row.name for row in BASE_SCENARIOS}
        return BASE_SCENARIOS + tuple(
            Scenario(f"{site}_constant", site, model.CONSTANT)
            for site in self.decomposed_sites
            if f"{site}_constant" not in already
        )


HIKURANGI_INTERFACE = Interface(
    "hikurangi", HIKURANGI, DIP_POSITION_KM, decomposed_sites=tuple(STRIKE_FRACTIONS)
)
PUYSEGUR_INTERFACES = (
    Interface(
        "puysegur_fiordland",
        PUYSEGUR_FIORDLAND,
        PUYSEGUR_DIP_POSITION_KM,
        decomposed_sites=tuple(STRIKE_FRACTIONS),
    ),
    Interface(
        "puyseguer",
        PUYSEGUR,
        PUYSEGUR_DIP_POSITION_KM,
        decomposed_sites=tuple(STRIKE_FRACTIONS),
    ),
)
INTERFACES = {
    interface.name: interface
    for interface in (HIKURANGI_INTERFACE, *PUYSEGUR_INTERFACES)
}

TRUE_DEPTH_CONDITIONS = (("truedepth", True), ("truedepth_materials_only", False))
"""The counterfactual's two readings of "materials", and whether the depth ramps move.

See :func:`~curvature.model.true_depth_materials` for what each includes and why the
first is the one the study reports.
"""


def peak_memory_gb() -> float:
    """This process's peak resident set, in gigabytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1.0e6


def magnitude_ladder(pair: MeshPair, chosen: float = model.MAGNITUDE) -> dict:
    """Which magnitudes this interface can carry the correlation structure of.

    Two independent bounds, both read off the parameter extents rather than off the
    area, because both are statements about how many correlation lengths fit across the
    domain. Reported at every rung of :data:`MAGNITUDE_LADDER` including the ones that
    fail, so the magnitude the study runs at is a choice with its alternatives beside it.

    Parameters
    ----------
    pair : MeshPair
    chosen : float, optional
        The magnitude actually run. Added as its own rung when it is not one of
        :data:`MAGNITUDE_LADDER`'s half-steps, so the two bounds are reported *at* the
        magnitude the numbers beside them were produced at rather than interpolated
        between the rungs either side of it.

    Returns
    -------
    dict
        Per magnitude: the two correlation lengths, their ratios to the two parameter
        extents, how many correlation lengths span each axis, and whether each bound
        holds.
    """
    extent_strike_km = float(np.ptp(pair.parameters_km[:, 0]))
    extent_dip_km = float(np.ptp(pair.parameters_km[:, 1]))
    ladder: dict[str, dict] = {}
    for magnitude in sorted({*MAGNITUDE_LADDER, chosen}):
        structure = correlation_lengths(magnitude)
        ratios = (
            structure.correlation_length_strike_km / extent_strike_km,
            structure.correlation_length_dip_km / extent_dip_km,
        )
        spans = (1.0 / ratios[0], 1.0 / ratios[1])
        ladder[f"mw_{magnitude:g}"] = {
            "correlation_length_strike_km": structure.correlation_length_strike_km,
            "correlation_length_dip_km": structure.correlation_length_dip_km,
            "mai_ratio_strike": ratios[0],
            "mai_ratio_dip": ratios[1],
            "mai_maximum_ratio": MAI_MAXIMUM_RATIO,
            "within_mai_range": bool(max(ratios) <= MAI_MAXIMUM_RATIO),
            "correlation_lengths_across_strike": spans[0],
            "correlation_lengths_across_dip": spans[1],
            "boundary_folding_lengths_needed": 2.0 * BOUNDARY_FOLDING_LENGTHS,
            "clear_of_boundary_folding": bool(
                min(spans) >= 2.0 * BOUNDARY_FOLDING_LENGTHS
            ),
        }
    return {
        "parameter_extent_strike_km": extent_strike_km,
        "parameter_extent_dip_km": extent_dip_km,
        "chosen_magnitude": chosen,
        "by_magnitude": ladder,
    }


def hypocentres(pair: MeshPair, dip_position_km: float = DIP_POSITION_KM) -> dict:
    """Locate each hypocentre and say where in the world it is.

    The face is chosen by nearest parameter coordinate, and **the same face index serves
    both models** -- which is the point of the paired design, since the two geometries
    have identical ``(u, v)`` and so the same parameter position is the same subfault.

    Parameters
    ----------
    pair : MeshPair
    dip_position_km : float, optional
        Where along dip all three sit. Defaults to Hikurangi's; see
        :data:`PUYSEGUR_DIP_POSITION_KM` for why Puysegur needs its own.

    Returns
    -------
    dict
        Per site: the face index, its parameter coordinates, and its longitude,
        latitude and depth in each of the two models.
    """
    parameters = pair.face_parameters_km()
    extent_km = float(parameters[:, 0].max())
    curved_centres = pair.centres_km(pair.curved_km)
    flat_centres = pair.centres_km(pair.flat_km)

    located: dict[str, dict] = {}
    for site, fraction in STRIKE_FRACTIONS.items():
        target = np.array([fraction * extent_km, dip_position_km])
        face = int(np.argmin(np.linalg.norm(parameters - target, axis=1)))
        curved_longitude, curved_latitude = pair.to_lonlat(curved_centres[face])
        flat_longitude, flat_latitude = pair.to_lonlat(flat_centres[face])
        located[site] = {
            "face_index": face,
            "strike_fraction": fraction,
            "strike_km": float(parameters[face, 0]),
            "dip_km": float(parameters[face, 1]),
            "curved_longitude_deg": float(curved_longitude),
            "curved_latitude_deg": float(curved_latitude),
            "curved_depth_km": float(curved_centres[face, 2]),
            "flat_longitude_deg": float(flat_longitude),
            "flat_latitude_deg": float(flat_latitude),
            "flat_depth_km": float(flat_centres[face, 2]),
            "depth_error_km": float(flat_centres[face, 2] - curved_centres[face, 2]),
        }
    return located


def geometry_report(pair: MeshPair) -> dict:
    """Everything about the two geometries that does not depend on a rupture.

    The area ratio, the displacement ``h``, the depth error ``Delta z`` -- which is the
    driver of every depth-mediated effect -- and the slope distribution.

    Parameters
    ----------
    pair : MeshPair

    Returns
    -------
    dict
    """
    curved_km2 = pair.areas_km2(pair.curved_km)
    flat_km2 = pair.areas_km2(pair.flat_km)
    curved_centres = pair.centres_km(pair.curved_km)
    flat_centres = pair.centres_km(pair.flat_km)
    depth_error_km = flat_centres[:, 2] - curved_centres[:, 2]
    slope = np.linalg.norm(pair.slopes(), axis=-1)
    identical, largest = same_white_noise(pair.vertex_count, model.SEEDS["slip"])

    return {
        "vertices": pair.vertex_count,
        "faces": pair.face_count,
        "median_edge_km": pair.median_edge_km,
        "build_spacing_km": COARSE_SPACING_KM,
        "subdivisions": SUBDIVISIONS,
        "parameter_extent_strike_km": float(np.ptp(pair.parameters_km[:, 0])),
        "parameter_extent_dip_km": float(np.ptp(pair.parameters_km[:, 1])),
        "frame_strike_deg": pair.frame.strike_deg,
        "frame_dip_deg": pair.frame.dip_deg,
        "frame_normal_from_vertical_deg": float(
            np.degrees(np.arccos(abs(pair.frame.normal[2])))
        ),
        "white_noise_bit_identical": identical,
        "white_noise_max_difference": largest,
        "area_true_km2": float(curved_km2.sum()),
        "area_projected_km2": float(flat_km2.sum()),
        "area_ratio_true_over_projected": float(curved_km2.sum() / flat_km2.sum()),
        **analysis.spread(
            curved_km2 / flat_km2, "per_face_area_ratio", "dimensionless"
        ),
        **analysis.spread(pair.displacement_km, "displacement_h", "km"),
        **analysis.spread(depth_error_km, "depth_error_flat_minus_curved", "km"),
        **analysis.weighted_spread(
            depth_error_km, curved_km2, "depth_error_flat_minus_curved", "km"
        ),
        "faces_flat_above_sea_level": int((flat_centres[:, 2] < 0.0).sum()),
        "fraction_flat_above_sea_level": float((flat_centres[:, 2] < 0.0).mean()),
        "area_flat_above_sea_level_km2": float(
            curved_km2[flat_centres[:, 2] < 0.0].sum()
        ),
        "curved_depth_min_km": float(curved_centres[:, 2].min()),
        "curved_depth_max_km": float(curved_centres[:, 2].max()),
        "flat_depth_min_km": float(flat_centres[:, 2].min()),
        "flat_depth_max_km": float(flat_centres[:, 2].max()),
        **analysis.spread(slope, "slope_grad_h", "dimensionless"),
        "slope_area_weighted_mean": float(np.average(slope, weights=curved_km2)),
    }


def moment_report(
    pair: MeshPair,
    pattern: FloatArray,
    curved: model.Materials,
    flat: model.Materials,
    magnitude: float = model.MAGNITUDE,
) -> dict:
    """The moment error of the flat model, split into its area and rigidity parts.

    The flat model scales its slip pattern so that ``sum(mu_flat A_flat f)`` hits the
    target. Projecting that slip onto the true interface applies it to ``mu_true`` and
    ``A_true`` instead, so the moment it *delivers* differs from the moment it *reports*
    by exactly

    .. code-block:: text

        delivered / target = S(mu_true, A_true) / S(mu_flat, A_flat)
                           = [S(mu_true, A_true) / S(mu_true, A_flat)]   the area part
                           x [S(mu_true, A_flat) / S(mu_flat, A_flat)]   the rigidity part

    -- an exact factorisation, so the two parts multiply to the whole and whether they
    compound or cancel is read off their signs rather than assumed. The **other** order of
    the same factorisation is reported too: the two orderings differ only by the
    interaction between the area error and the rigidity error, so quoting both says how
    separable the effects are without a separate calculation.

    The pattern is the **flat model's own**, because that is what the workflow being
    criticised produces.

    Passing the *same* materials as both arguments is what the true-depth counterfactual
    does, and the factorisation then reports a rigidity part of exactly 1 -- not by a
    special case but because ``mu_true / mu_flat`` is 1 at every face.

    Parameters
    ----------
    pair : MeshPair
    pattern : FloatArray
        ``(F,)`` the flat model's dimensionless slip pattern.
    curved, flat : model.Materials
        Sampled at each model's own depths.
    magnitude : float, optional
        The target the flat model scaled its pattern to. Only the two moments and the
        mean slip read it: the three ratios are ratios of the same fold and the
        magnitude cancels out of them exactly, which is why the area and rigidity terms
        are properties of the geometry rather than of the event.

    Returns
    -------
    dict
    """
    curved_km2 = pair.areas_km2(pair.curved_km)
    flat_km2 = pair.areas_km2(pair.flat_km)

    def fold(rigidity_pa: FloatArray, areas_km2: FloatArray) -> float:
        """``sum(mu A f)`` in newton-metres per unit of the dimensionless pattern."""
        return float(np.sum(rigidity_pa * areas_km2 * 1.0e6 * pattern))

    true_true = fold(curved.rigidity_pa, curved_km2)
    true_flat = fold(curved.rigidity_pa, flat_km2)
    flat_true = fold(flat.rigidity_pa, curved_km2)
    flat_flat = fold(flat.rigidity_pa, flat_km2)

    target_nm = seismic_moment_nm(magnitude)
    rigidity_ratio = curved.rigidity_pa / flat.rigidity_pa
    return {
        "target_moment_nm": target_nm,
        "curved_reported_moment_nm": target_nm,
        "flat_reported_moment_nm": target_nm,
        "flat_delivered_moment_nm": target_nm * true_true / flat_flat,
        "flat_delivered_over_target": true_true / flat_flat,
        # The magnitude the flat model's rupture actually is, once its slip sits on
        # the true surface. Hanks & Kanamori equation 7 read backwards.
        "flat_delivered_magnitude": float(
            np.log10(target_nm * true_true / flat_flat) / 1.5 - MAGNITUDE_COEFFICIENT
        ),
        "area_contribution_ratio": true_true / true_flat,
        "rigidity_contribution_ratio": true_flat / flat_flat,
        "area_contribution_ratio_other_ordering": flat_true / flat_flat,
        "rigidity_contribution_ratio_other_ordering": true_true / flat_true,
        "interaction_ratio": (true_true / true_flat) / (flat_true / flat_flat),
        **analysis.spread(
            rigidity_ratio, "rigidity_ratio_true_over_flat", "dimensionless"
        ),
        "faces_with_rigidity_error": int((rigidity_ratio != 1.0).sum()),
        "fraction_with_rigidity_error": float((rigidity_ratio != 1.0).mean()),
        "faces_in_different_velocity_layer": int((curved.layer != flat.layer).sum()),
        "fraction_in_different_velocity_layer": float(
            (curved.layer != flat.layer).mean()
        ),
        **analysis.spread(
            (flat.layer - curved.layer).astype(np.float64),
            "layer_offset_flat_minus_curved",
            "layers",
        ),
        "mean_slip_curved_m": float(
            np.sum(
                curved_km2
                * model.slip_metres(pattern, curved.rigidity_pa, curved_km2, magnitude)
            )
            / curved_km2.sum()
        ),
    }


def correlation_report(
    pair: MeshPair, fields: dict, magnitude: float = model.MAGNITUDE
) -> tuple[dict, dict]:
    """Delivered correlation lengths, against **surface** separation, for both models.

    Two measurements, and they answer different questions. The **Gaussian** is the
    sampler's own output and is what the SPDE's covariance is a statement about; the
    **slip pattern** is that field after truncation at zero and the edge taper, which is
    what actually reaches an SRF. Reporting only the first would flatter the sampler and
    only the second would confound the geometry with the truncation.

    Parameters
    ----------
    pair : MeshPair
    fields : dict
        ``{"curved": Fields, "flat": Fields}``.
    magnitude : float, optional
        Sets the correlation lengths the delivered ones are measured against, so it must
        be the magnitude ``fields`` were drawn at.

    Returns
    -------
    tuple
        The summary dict, and the raw profiles for plotting.
    """
    parameters = pair.face_parameters_km()
    height, _, _ = analysis.rasterise(
        parameters, pair.displacement_km[pair.faces].mean(axis=1), RASTER_SPACING_KM
    )
    structure = model.covariance(magnitude)
    asked = {
        "strike": structure.correlation_length_strike_km,
        "dip": structure.correlation_length_dip_km,
    }

    slope = pair.slopes()
    areas_km2 = pair.areas_km2(pair.curved_km)
    summary: dict = {
        "asked_correlation_length_strike_km": asked["strike"],
        "asked_correlation_length_dip_km": asked["dip"],
        "correlation_level_at_one_length": analysis.HALF_CORRELATION,
        # How much longer the true surface is than its projection, along each axis
        # separately. These are what a projected field's correlation length is
        # stretched by, so they are the prediction the profiles are measured against.
        "metric_factor_along_strike_area_weighted": float(
            np.average(np.sqrt(1.0 + slope[:, 0] ** 2), weights=areas_km2)
        ),
        "metric_factor_down_dip_area_weighted": float(
            np.average(np.sqrt(1.0 + slope[:, 1] ** 2), weights=areas_km2)
        ),
    }
    profiles: dict = {}
    # The three readings that matter, and the fourth for symmetry. "on_true" measures
    # separation along the real interface; "on_plane" measures it in the parameter
    # plane, which is the plane's own surface distance. A **flat field on the true
    # surface** is the literature workflow's actual delivered field, and is the only
    # combination that can disagree with the model it was asked for.
    readings = (
        ("curved_on_true", "curved", height),
        ("curved_on_plane", "curved", None),
        ("flat_on_plane", "flat", None),
        ("flat_on_true", "flat", height),
    )
    for reading, geometry, surface in readings:
        drawn = fields[geometry]
        for quantity, values in (
            ("gaussian", drawn.gaussian),
            ("slip_pattern", drawn.pattern),
        ):
            grid, _, _ = analysis.rasterise(parameters, values, RASTER_SPACING_KM)
            for direction, axis in (("strike", 1), ("dip", 0)):
                separation_km, correlation, count = analysis.correlation_profile(
                    grid, surface, RASTER_SPACING_KM, axis=axis
                )
                key = f"{reading}_{quantity}_{direction}"
                profiles[f"{key}_separation_km"] = separation_km
                profiles[f"{key}_correlation"] = correlation
                profiles[f"{key}_pairs"] = count
                delivered = analysis.delivered_length_km(separation_km, correlation)
                summary[f"delivered_length_{key}_km"] = delivered
                summary[f"delivered_over_asked_{key}"] = delivered / asked[direction]

    for quantity in ("gaussian", "slip_pattern"):
        for direction in ("strike", "dip"):
            on_plane = summary[
                f"delivered_length_flat_on_plane_{quantity}_{direction}_km"
            ]
            on_true = summary[
                f"delivered_length_flat_on_true_{quantity}_{direction}_km"
            ]
            summary[f"projection_stretch_{quantity}_{direction}"] = on_true / on_plane
            summary[f"flat_over_curved_delivered_{quantity}_{direction}"] = (
                on_true
                / summary[f"delivered_length_curved_on_true_{quantity}_{direction}_km"]
            )

    summary["pointwise_correlation_gaussian"] = analysis.pearson(
        fields["curved"].gaussian, fields["flat"].gaussian
    )
    summary["pointwise_correlation_slip_pattern"] = analysis.pearson(
        fields["curved"].pattern, fields["flat"].pattern
    )
    summary["pointwise_correlation_rise_time"] = analysis.pearson(
        fields["curved"].rise_time_s, fields["flat"].rise_time_s
    )
    summary["pointwise_correlation_rake"] = analysis.pearson(
        fields["curved"].rake_deg, fields["flat"].rake_deg
    )
    summary["gaussian_spread_curved"] = float(fields["curved"].draw.std())
    summary["gaussian_spread_flat"] = float(fields["flat"].draw.std())
    summary["gaussian_spread_ratio_flat_over_curved"] = float(
        fields["flat"].draw.std() / fields["curved"].draw.std()
    )
    summary["pointwise_correlation_after_standardise"] = analysis.pearson(
        standardise(fields["curved"].draw), standardise(fields["flat"].draw)
    )
    return summary, profiles


def arrival_report(
    delta_travel_s: FloatArray,
    delta_onset_s: FloatArray,
    areas_curved_km2: FloatArray,
    duration_s: float,
) -> dict:
    """How far apart two models' rupture fronts are, summarised the one way.

    Factored out so the true-depth row is measured with the same code as the four rows
    of the run matrix -- a second summary written out again would be a second chance to
    quote a differently defined median in the same table.

    Parameters
    ----------
    delta_travel_s, delta_onset_s : FloatArray
        ``(F,)`` flat minus curved, seconds. Onset is travel time plus the drawn
        perturbation, so the two differ by however much the perturbation differs.
    areas_curved_km2 : FloatArray
        ``(F,)`` the true area each face carries, for the area-weighted summary.
    duration_s : float
        The curved model's own last arrival, which is what the differences are large or
        small *against*.

    Returns
    -------
    dict
    """
    return {
        **analysis.spread(delta_travel_s, "delta_travel_time_flat_minus_curved", "s"),
        **analysis.weighted_spread(
            delta_travel_s,
            areas_curved_km2,
            "delta_travel_time_flat_minus_curved",
            "s",
        ),
        **analysis.spread(delta_onset_s, "delta_onset_flat_minus_curved", "s"),
        "delta_travel_time_p90_as_fraction_of_duration": float(
            np.percentile(delta_travel_s, 90.0) / duration_s
        ),
        "delta_travel_time_max_as_fraction_of_duration": float(
            np.abs(delta_travel_s).max() / duration_s
        ),
        "delta_travel_time_median_as_fraction_of_duration": float(
            np.median(delta_travel_s) / duration_s
        ),
        "fraction_of_faces_flat_arrives_early": float((delta_travel_s < 0.0).mean()),
    }


def spectrum_report(
    scenario: str,
    geometry: str,
    times_s: FloatArray,
    accumulated: FloatArray,
    saved: dict,
) -> dict:
    """One model's moment rate function, its spectrum, and the arrays a figure needs.

    Parameters
    ----------
    scenario, geometry : str
        Name the saved arrays.
    times_s : FloatArray
        ``(T,)`` seconds from the first onset.
    accumulated : FloatArray
        ``(2, T)`` the moment rate under the two weightings -- what the model reports on
        its own areas and rigidities, and what it delivers once its slip sits on the
        true surface.
    saved : dict
        Mutated: the timeline, the two rates and the spectrum are added to it.

    Returns
    -------
    dict
        The scalars, keyed by geometry.
    """
    saved[f"mrf_times_{scenario}_{geometry}_s"] = times_s.astype(np.float32)
    saved[f"mrf_rate_{scenario}_{geometry}_reported_nm_s"] = accumulated[0].astype(
        np.float32
    )
    saved[f"mrf_rate_{scenario}_{geometry}_projected_nm_s"] = accumulated[1].astype(
        np.float32
    )
    frequency_hz, amplitude = model.amplitude_spectrum(
        accumulated[0], model.SAMPLE_INTERVAL_S
    )
    saved[f"spectrum_frequency_{scenario}_{geometry}_hz"] = frequency_hz.astype(
        np.float32
    )
    saved[f"spectrum_amplitude_{scenario}_{geometry}_nm"] = amplitude.astype(np.float32)
    return {
        f"moment_from_mrf_{geometry}_reported_nm": float(
            cumulative_moment(times_s, accumulated[0])[-1]
        ),
        f"moment_from_mrf_{geometry}_projected_nm": float(
            cumulative_moment(times_s, accumulated[1])[-1]
        ),
        f"peak_moment_rate_{geometry}_reported_nm_s": float(accumulated[0].max()),
        f"mrf_duration_{geometry}_s": float(times_s[-1] - times_s[0]),
        f"corner_frequency_{geometry}_hz": analysis.corner_frequency_hz(
            frequency_hz, amplitude
        ),
        f"high_frequency_slope_{geometry}": analysis.high_frequency_slope(
            frequency_hz, amplitude, SPECTRAL_BAND_HZ
        ),
    }


def true_depth_report(
    pair: MeshPair,
    samplers: dict,
    taper_weight: FloatArray,
    areas: dict,
    curved: dict,
    located: dict,
    sites: tuple[str, ...],
    saved: dict,
    magnitude: float = model.MAGNITUDE,
) -> dict:
    """The counterfactual row, and the two numbers a refactor decision turns on.

    Everything about this model is the flat model's -- the plane's vertices, so the
    plane's shorter paths; the plane's areas; the plane's metric on the correlated
    field; the same white noise and the same shared taper -- **except** that the rock is
    read at the true interface depth. See
    :func:`~curvature.model.true_depth_materials` for what "the rock" is taken to
    include, and for the narrower reading reported beside it.

    The materials and the drawn fields depend on the condition alone, so they are built
    once per condition and the hypocentres loop inside. Only the eikonal solve and the
    moment rate are per hypocentre, which is why running three costs the front rather
    than the field.

    Parameters
    ----------
    pair : MeshPair
    samplers : dict
        ``{"curved": Sampler, "flat": Sampler}``. Only the flat one is used: the
        counterfactual is a flat model, and using the curved operator would be a
        different experiment.
    taper_weight : FloatArray
        ``(F,)`` shared, as everywhere.
    areas : dict
        ``{"curved": (F,), "flat": (F,)}`` square kilometres.
    curved : dict
        Per site, the curved standard model this is measured against: its ``materials``,
        ``travel_s`` and ``onset_s`` at that hypocentre, and the ``duration_s`` its
        front takes.
    located : dict
        Every hypocentre, from :func:`hypocentres`.
    sites : tuple of str
        Which of them to run, from :attr:`Interface.decomposed_sites`.
    saved : dict
        Mutated with the rasters, the polar coordinates and the spectra.
    magnitude : float, optional
        The event, passed on to the fields and the moment fold so the counterfactual is
        the same event as the rows it is compared against.

    Returns
    -------
    dict
    """
    speed = model.speed_params(pair, model.STANDARD)
    pulse = model.pulse_params(model.STANDARD)
    parameters = pair.face_parameters_km()
    true_depth_km = pair.centres_km(pair.curved_km)[:, 2]
    flat_depth_km = pair.centres_km(pair.flat_km)[:, 2]

    report: dict = {
        "hypocentre": located["central"],
        "hypocentres": {site: located[site] for site in sites},
        "included_in_materials": [
            "moment.sample_velocity_model: rigidity, at the true depth",
            "moment.sample_velocity_model: shear speed, at the true depth",
            "timing.SpeedParams.depth_factor: the ramp reads the true depth",
            "stages.RiseTimeParams.stretch_at: the ramp reads the true depth",
            "stages.RiseTimeParams.shallow_blend: the blend reads the true depth",
            "pulses.PulseParams.beta_at: the rising fraction reads the true depth",
        ],
        "left_flat_on_purpose": [
            "the vertices the eikonal front propagates over, so the paths stay short",
            "the face areas the moment folds over",
            "the SPDE operator, so the correlated field keeps the plane's metric",
            "the lateral taper, which is measured in the shared parameter plane",
        ],
        "conditions": {},
        "scenarios": {},
        "moment": {},
    }

    for condition, ramps_read_true_depth in TRUE_DEPTH_CONDITIONS:
        materials = model.true_depth_materials(
            true_depth_km,
            flat_depth_km,
            model.STANDARD,
            ramps_read_true_depth=ramps_read_true_depth,
        )
        report["conditions"][condition] = {
            "depth_ramps_read_true_depth": ramps_read_true_depth,
            **analysis.spread(materials.depth_km, "depth_the_ramps_read", "km"),
        }
        # The chart the depth-reading field stages see. `draw_fields` reads the depth
        # off the vertices it is handed and nothing else, so handing it the curved
        # vertices while the sampler stays the flat one is exactly this counterfactual:
        # true depths, the plane's covariance.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fields = model.draw_fields(
                pair,
                samplers["flat"],
                pair.curved_km if ramps_read_true_depth else pair.flat_km,
                model.STANDARD,
                taper_weight,
                magnitude,
            )
        slip_m = model.slip_metres(
            fields.pattern, materials.rigidity_pa, areas["flat"], magnitude
        )

        report["moment"][condition] = moment_report(
            pair, fields.pattern, materials, materials, magnitude
        )
        report["moment"][condition].update(analysis.spread(slip_m, "slip_flat", "m"))
        report["moment"][condition].update(
            analysis.spread(fields.rise_time_s, "rise_time_flat", "s")
        )
        report["moment"][condition].update(
            analysis.spread(
                timing.speed_field(
                    materials.depth_km, materials.shear_speed_km_s, speed
                ),
                "rupture_speed_flat",
                "km_s",
            )
        )

        # Slip and rise time are the condition's, not the hypocentre's -- nothing in
        # either reads where the rupture started -- so they are rasterised once and kept
        # under the central name they have always had, rather than once per site.
        for label, values in (
            (f"raster_slip_central_standard_{condition}_flat_m", slip_m),
            (
                f"raster_rise_time_central_standard_{condition}_flat_s",
                fields.rise_time_s,
            ),
        ):
            grid, _, _ = analysis.rasterise(parameters, values, RASTER_SPACING_KM)
            saved[label] = grid.astype(np.float32)

        for site in sites:
            face = located[site]["face_index"]
            name = f"{site}_standard_{condition}"

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", DegradedSeed)
                travel_s, seed_reports = model.travel_times(
                    pair, pair.flat_km, materials, speed, face
                )
            onset_s = model.onset_of(travel_s, fields.perturbation, face)

            delta_travel_s = travel_s - curved[site]["travel_s"]
            delta_onset_s = onset_s - curved[site]["onset_s"]
            entry: dict = {
                "site": site,
                "velocity_model": "standard",
                "materials_sampled_at": "true interface depth",
                "measured_against": (
                    "the curved model under the standard velocity model"
                ),
                "hypocentre": located[site],
                "rupture_duration_curved_s": curved[site]["duration_s"],
                "rupture_duration_flat_s": float(travel_s.max()),
                **arrival_report(
                    delta_travel_s,
                    delta_onset_s,
                    areas["curved"],
                    curved[site]["duration_s"],
                ),
                "seed_ball_warnings": [
                    str(caught_one.message) for caught_one in caught
                ],
                **{
                    f"seed_ball_{quantity}_flat": float(
                        max(getattr(seed, quantity) for seed in seed_reports)
                    )
                    for quantity in ("radius_km", "slowness_error_s")
                },
                "seed_ball_budget_s": float(fim.SEED_SLOWNESS_BUDGET_S),
            }

            weights = np.stack(
                [
                    areas["flat"] * 1.0e6 * materials.rigidity_pa,
                    areas["curved"] * 1.0e6 * curved[site]["materials"].rigidity_pa,
                ]
            )
            times_s, accumulated = model.moment_rate(
                slip_m,
                fields.rise_time_s,
                onset_s,
                materials.depth_km,
                weights,
                pulse,
            )
            entry.update(spectrum_report(name, "flat", times_s, accumulated, saved))
            report["scenarios"][name] = entry

            # The azimuth and the radius the polar figure needs are this hypocentre's
            # own, which the scenario loop has already saved under ``{site}_standard``;
            # carrying them again under a second name would be two copies of one array
            # in a 120 MB file.
            for label, values in (
                (f"raster_delta_travel_{name}_s", delta_travel_s),
                (f"raster_travel_flat_{name}_s", travel_s),
            ):
                grid, _, _ = analysis.rasterise(parameters, values, RASTER_SPACING_KM)
                saved[label] = grid.astype(np.float32)
            saved[f"polar_delta_travel_{name}_s"] = delta_travel_s.astype(np.float32)

    return report


def study(
    interface: Interface, magnitude: float = model.MAGNITUDE
) -> tuple[dict, dict]:
    """Run the whole matrix on one interface.

    Parameters
    ----------
    interface : Interface
    magnitude : float, optional
        The event. Everything else -- the mesh, the seeds, the hypocentres, the sample
        interval, the six rows and the counterfactual -- is held fixed, so a second
        magnitude is a control on magnitude dependence rather than a second experiment.
        Defaults to :data:`~curvature.model.MAGNITUDE`, which reproduces the study's own
        numbers.

    Returns
    -------
    tuple
        The results groups, and the rasters and profiles the figures draw from.
    """
    started = time.perf_counter()
    timings: dict[str, float] = {}

    step = time.perf_counter()
    pair = build_pair(interface.path)
    taper_weight = model.lateral_taper(pair)
    timings["build_and_taper_s"] = time.perf_counter() - step

    results: dict = {
        "geometry": geometry_report(pair),
        "hypocentres": hypocentres(pair, interface.dip_position_km),
        "settings": {
            "decomposed_sites": list(interface.decomposed_sites),
            "magnitude": magnitude,
            "sample_interval_s": model.SAMPLE_INTERVAL_S,
            "velocity_fraction": model.VELOCITY_FRACTION,
            "average_rake_deg": model.AVERAGE_RAKE_DEG,
            "coefficient_of_variation": model.COEFFICIENT_OF_VARIATION,
            "side_taper_fraction": model.SIDE_TAPER,
            "onset_scale_s": model.ONSET_SCALE_S,
            "raster_spacing_km": RASTER_SPACING_KM,
            "seeds": model.SEEDS,
            "faces_touched_by_taper": int((taper_weight < 1.0).sum()),
            "neutralised_in_the_constant_control": [
                "moment.sample_velocity_model: one layer, one shear speed, one density",
                "timing.SpeedParams.depth_factor: shallow_factor = deep_factor = 1",
                "stages.RiseTimeParams.stretch_at: shallow_factor = deep_factor = 1",
                (
                    "stages.RiseTimeParams.shallow_blend: ramp at -1000 km, weight is "
                    "1 at every face of both models"
                ),
                "pulses.PulseParams.beta_at: beta_shallow = beta_mid = beta_deep",
            ],
            "reads_no_depth_so_needs_no_neutralising": [
                "stages.RakeParams: base + sigma * Z",
                "stages.OnsetParams: a standardised correlated field",
            ],
            "nyquist_hz": 0.5 / model.SAMPLE_INTERVAL_S,
            "spectral_band_hz": list(SPECTRAL_BAND_HZ),
        },
        "scenarios": {},
    }
    if interface is not HIKURANGI_INTERFACE or magnitude != model.MAGNITUDE:
        # Hikurangi's Mw 8.5 magnitude was settled before this study and its group is
        # quoted as it stands, so the ladder is reported where it was actually used to
        # choose -- which is every surface at every other magnitude, Hikurangi included.
        results["settings"]["dip_position_km"] = interface.dip_position_km
        results["magnitude_ladder"] = magnitude_ladder(pair, magnitude)

    geometries = {"curved": pair.curved_km, "flat": pair.flat_km}
    levels = {"curved": pair.curved_levels, "flat": pair.flat_levels}

    step = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samplers = {
            name: model.Sampler(pair, vertices, levels[name], magnitude)
            for name, vertices in geometries.items()
        }
    timings["operator_setup_s"] = time.perf_counter() - step
    results["settings"]["spde_error_bound"] = {
        name: str(sampler.error) for name, sampler in samplers.items()
    }

    materials = {
        name: {
            geometry: model.materials_of(pair.centres_km(vertices)[:, 2], velocity)
            for geometry, vertices in geometries.items()
        }
        for name, velocity in (
            ("constant", model.CONSTANT),
            ("standard", model.STANDARD),
        )
    }

    step = time.perf_counter()
    fields: dict = {}
    for name, velocity in (("constant", model.CONSTANT), ("standard", model.STANDARD)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fields[name] = {
                geometry: model.draw_fields(
                    pair,
                    samplers[geometry],
                    vertices,
                    velocity,
                    taper_weight,
                    magnitude,
                )
                for geometry, vertices in geometries.items()
            }
    timings["draw_fields_s"] = time.perf_counter() - step

    areas = {
        geometry: pair.areas_km2(vertices) for geometry, vertices in geometries.items()
    }
    slip = {
        name: {
            geometry: model.slip_metres(
                fields[name][geometry].pattern,
                materials[name][geometry].rigidity_pa,
                areas[geometry],
                magnitude,
            )
            for geometry in geometries
        }
        for name in fields
    }

    velocities = {"constant": model.CONSTANT, "standard": model.STANDARD}
    for name in fields:
        results.setdefault("moment", {})[name] = moment_report(
            pair,
            fields[name]["flat"].pattern,
            materials[name]["curved"],
            materials[name]["flat"],
            magnitude,
        )
        results["moment"][name]["truncated_fraction_curved"] = fields[name][
            "curved"
        ].truncated_fraction
        results["moment"][name]["truncated_fraction_flat"] = fields[name][
            "flat"
        ].truncated_fraction
        results["moment"][name]["zero_slip_faces_curved"] = int(
            (fields[name]["curved"].pattern == 0.0).sum()
        )
        results["moment"][name]["zero_slip_faces_flat"] = int(
            (fields[name]["flat"].pattern == 0.0).sum()
        )
        for geometry in geometries:
            results["moment"][name].update(
                analysis.spread(slip[name][geometry], f"slip_{geometry}", "m")
            )
            results["moment"][name].update(
                analysis.spread(
                    fields[name][geometry].rise_time_s,
                    f"rise_time_{geometry}",
                    "s",
                )
            )
        # `stages.rise_time_field` clamps its output to exactly one sample interval, so
        # this is where the dt = 0.02 s floor binds -- and if it binds here, the pulse
        # kernel never sees a rise time it cannot represent and never raises
        # `UnrepresentableRiseTime`. Counted rather than assumed.
        for geometry in geometries:
            at_floor = (
                fields[name][geometry].rise_time_s <= model.SAMPLE_INTERVAL_S * 1.000001
            )
            results["moment"][name][f"faces_at_rise_time_floor_{geometry}"] = int(
                at_floor.sum()
            )
            results["moment"][name][f"fraction_at_rise_time_floor_{geometry}"] = float(
                at_floor.mean()
            )
            results["moment"][name][
                f"moment_fraction_at_rise_time_floor_{geometry}"
            ] = float(
                np.sum(
                    (
                        slip[name][geometry]
                        * areas[geometry]
                        * materials[name][geometry].rigidity_pa
                    )[at_floor]
                )
                / np.sum(
                    slip[name][geometry]
                    * areas[geometry]
                    * materials[name][geometry].rigidity_pa
                )
            )
        # The rupture speed itself, which is the channel the depth error reaches the
        # onset through. Split into its two factors so the document can say which one
        # carries the effect: the shear speed the velocity model returns at that depth,
        # and the depth ramp `timing.SpeedParams.depth_factor` applies on top of it.
        for geometry in geometries:
            depth_km = materials[name][geometry].depth_km
            params = model.speed_params(pair, velocities[name])
            results["moment"][name].update(
                analysis.spread(
                    timing.speed_field(
                        depth_km, materials[name][geometry].shear_speed_km_s, params
                    ),
                    f"rupture_speed_{geometry}",
                    "km_s",
                )
            )
            results["moment"][name].update(
                analysis.spread(
                    materials[name][geometry].shear_speed_km_s,
                    f"shear_speed_{geometry}",
                    "km_s",
                )
            )
            results["moment"][name].update(
                analysis.spread(
                    params.depth_factor(depth_km),
                    f"depth_factor_{geometry}",
                    "dimensionless",
                )
            )
        results["moment"][name]["rise_time_floor_s"] = model.SAMPLE_INTERVAL_S
        results["moment"][name]["unrepresentable_rise_time_raised"] = False
        results["moment"][name]["rise_time_ratio_flat_over_curved"] = analysis.spread(
            fields[name]["flat"].rise_time_s / fields[name]["curved"].rise_time_s,
            "rise_time_ratio",
            "dimensionless",
        )

    step = time.perf_counter()
    correlation, profiles = correlation_report(pair, fields["standard"], magnitude)
    results["correlation"] = correlation
    timings["correlation_s"] = time.perf_counter() - step

    # The rasters and profiles the figures draw from. Face fields are 1.39 M values, so
    # only the ones a figure actually uses are carried, and as rasters rather than as
    # face arrays.
    parameters = pair.face_parameters_km()
    curved_centres = pair.centres_km(pair.curved_km)
    flat_centres = pair.centres_km(pair.flat_km)
    saved: dict = dict(profiles)
    height_face = pair.displacement_km[pair.faces].mean(axis=1)
    for label, values in (
        ("displacement_h_km", height_face),
        ("depth_error_km", flat_centres[:, 2] - curved_centres[:, 2]),
        ("depth_curved_km", curved_centres[:, 2]),
        ("depth_flat_km", flat_centres[:, 2]),
        ("slope_grad_h", np.linalg.norm(pair.slopes(), axis=-1)),
        (
            "layer_crossed",
            (
                materials["standard"]["curved"].layer
                != materials["standard"]["flat"].layer
            ).astype(np.float64),
        ),
        (
            "rigidity_ratio",
            materials["standard"]["curved"].rigidity_pa
            / materials["standard"]["flat"].rigidity_pa,
        ),
        ("gaussian_curved", fields["standard"]["curved"].gaussian),
        ("gaussian_flat", fields["standard"]["flat"].gaussian),
    ):
        grid, axis_u, axis_v = analysis.rasterise(parameters, values, RASTER_SPACING_KM)
        saved[f"raster_{label}"] = grid.astype(np.float32)
    saved["raster_axis_u_km"] = axis_u
    saved["raster_axis_v_km"] = axis_v

    for name in fields:
        for geometry in geometries:
            for label, values in (
                (f"slip_{name}_{geometry}_m", slip[name][geometry]),
                (f"rise_time_{name}_{geometry}_s", fields[name][geometry].rise_time_s),
            ):
                grid, _, _ = analysis.rasterise(parameters, values, RASTER_SPACING_KM)
                saved[f"raster_{label}"] = grid.astype(np.float32)

    # Geodesic distance on the *true* surface from each hypocentre: a uniform unit
    # slowness makes the eikonal solution the distance itself, which is the radius the
    # polar plot needs and is not recoverable from the parameter plane.
    step = time.perf_counter()
    uniform = model.Materials(
        depth_km=np.zeros(pair.face_count),
        shear_speed_km_s=np.ones(pair.face_count),
        rigidity_pa=np.ones(pair.face_count),
        layer=np.zeros(pair.face_count, dtype=np.int64),
    )
    # `speed_field` returns `velocity_fraction / alpha_T * depth_factor * shear_speed`.
    # With the control's depth factor of exactly 1, a unit shear speed and a fraction
    # equal to alpha_T, that is exactly 1 km/s -- so the arrival time in seconds is the
    # geodesic distance in kilometres.
    unit_speed = dataclasses.replace(
        model.speed_params(pair, model.CONSTANT),
        velocity_fraction=alpha_t(pair.frame.dip_deg, model.AVERAGE_RAKE_DEG),
    )
    geodesic = {}
    for site, located in results["hypocentres"].items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geodesic[site], _ = model.travel_times(
                pair, pair.curved_km, uniform, unit_speed, located["face_index"]
            )
    timings["geodesic_s"] = time.perf_counter() - step

    step = time.perf_counter()
    seed_warnings: dict[str, list[str]] = {}
    curved_standard: dict[str, dict] = {}
    for scenario in interface.scenarios:
        located = results["hypocentres"][scenario.site]
        face = located["face_index"]
        name = scenario.velocity.name
        speed = model.speed_params(pair, scenario.velocity)
        pulse = model.pulse_params(scenario.velocity)

        travel: dict[str, FloatArray] = {}
        onset: dict[str, FloatArray] = {}
        seed_reports: dict[str, tuple] = {}
        for geometry, vertices in geometries.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", DegradedSeed)
                travel[geometry], seed_reports[geometry] = model.travel_times(
                    pair, vertices, materials[name][geometry], speed, face
                )
            seed_warnings[f"{scenario.name}_{geometry}"] = [
                str(entry.message) for entry in caught
            ]
            onset[geometry] = model.onset_of(
                travel[geometry], fields[name][geometry].perturbation, face
            )

        delta_travel_s = travel["flat"] - travel["curved"]
        delta_onset_s = onset["flat"] - onset["curved"]
        duration_s = float(travel["curved"].max())

        entry: dict = {
            "site": scenario.site,
            "velocity_model": name,
            "hypocentre": located,
            "rupture_duration_curved_s": duration_s,
            "rupture_duration_flat_s": float(travel["flat"].max()),
            **arrival_report(
                delta_travel_s, delta_onset_s, areas["curved"], duration_s
            ),
            "seed_ball_warnings": seed_warnings[f"{scenario.name}_curved"]
            + seed_warnings[f"{scenario.name}_flat"],
            # The analytic geodesic ball the front is seeded from assumes constant
            # slowness over its own radius, and this is what that costs at the ring, in
            # seconds. It is a bias on the hypocentre's own onset, present in both
            # models, and it belongs next to the onset differences so a reader can see
            # it is small against them rather than having to take that on trust.
            **{
                f"seed_ball_{quantity}_{geometry}": float(
                    max(getattr(report, quantity) for report in seed_reports[geometry])
                )
                for geometry in geometries
                for quantity in ("radius_km", "slowness_error_s")
            },
            "seed_ball_budget_s": float(fim.SEED_SLOWNESS_BUDGET_S),
        }

        for geometry in geometries:
            weights = np.stack(
                [
                    areas[geometry] * 1.0e6 * materials[name][geometry].rigidity_pa,
                    areas["curved"] * 1.0e6 * materials[name]["curved"].rigidity_pa,
                ]
            )
            times_s, accumulated = model.moment_rate(
                slip[name][geometry],
                fields[name][geometry].rise_time_s,
                onset[geometry],
                materials[name][geometry].depth_km,
                weights,
                pulse,
            )
            entry.update(
                spectrum_report(scenario.name, geometry, times_s, accumulated, saved)
            )

        entry["peak_moment_rate_ratio_flat_over_curved"] = (
            entry["peak_moment_rate_flat_reported_nm_s"]
            / entry["peak_moment_rate_curved_reported_nm_s"]
        )
        entry["corner_frequency_ratio_flat_over_curved"] = (
            entry["corner_frequency_flat_hz"] / entry["corner_frequency_curved_hz"]
        )
        results["scenarios"][scenario.name] = entry

        # What the counterfactual at this hypocentre will be measured against. Kept for
        # every standard row rather than for the central one alone, because the
        # counterfactual runs wherever the control does and its comparand is the curved
        # standard model at the same hypocentre.
        if scenario.velocity is model.STANDARD:
            curved_standard[scenario.site] = {
                "materials": materials[name]["curved"],
                "travel_s": travel["curved"],
                "onset_s": onset["curved"],
                "duration_s": duration_s,
            }

        grid, _, _ = analysis.rasterise(parameters, delta_travel_s, RASTER_SPACING_KM)
        saved[f"raster_delta_travel_{scenario.name}_s"] = grid.astype(np.float32)
        grid, _, _ = analysis.rasterise(parameters, travel["curved"], RASTER_SPACING_KM)
        saved[f"raster_travel_curved_{scenario.name}_s"] = grid.astype(np.float32)
        grid, _, _ = analysis.rasterise(parameters, travel["flat"], RASTER_SPACING_KM)
        saved[f"raster_travel_flat_{scenario.name}_s"] = grid.astype(np.float32)

        # The polar plot's own coordinates: azimuth in the shared parameter plane,
        # radius the true surface distance from the hypocentre.
        offset = parameters - parameters[face]
        saved[f"polar_azimuth_{scenario.name}_rad"] = np.arctan2(
            offset[:, 1], offset[:, 0]
        ).astype(np.float32)
        saved[f"polar_radius_{scenario.name}_km"] = geodesic[scenario.site].astype(
            np.float32
        )
        saved[f"polar_delta_travel_{scenario.name}_s"] = delta_travel_s.astype(
            np.float32
        )
    timings["scenarios_s"] = time.perf_counter() - step

    # The controlled attribution: the same difference with and without depth.
    control = results["scenarios"]["central_constant"]
    full = results["scenarios"]["central_standard"]
    results["attribution"] = {
        "geometric_delta_travel_median_s": control[
            "delta_travel_time_flat_minus_curved_median_s"
        ],
        "geometric_delta_travel_p90_s": control[
            "delta_travel_time_flat_minus_curved_p90_s"
        ],
        "geometric_delta_travel_max_absolute_s": control[
            "delta_travel_time_flat_minus_curved_max_absolute_s"
        ],
        "with_depth_delta_travel_median_s": full[
            "delta_travel_time_flat_minus_curved_median_s"
        ],
        "with_depth_delta_travel_p90_s": full[
            "delta_travel_time_flat_minus_curved_p90_s"
        ],
        "with_depth_delta_travel_max_absolute_s": full[
            "delta_travel_time_flat_minus_curved_max_absolute_s"
        ],
        "depth_share_of_delta_travel_median_s": full[
            "delta_travel_time_flat_minus_curved_median_s"
        ]
        - control["delta_travel_time_flat_minus_curved_median_s"],
        "geometric_moment_ratio": results["moment"]["constant"][
            "flat_delivered_over_target"
        ],
        "with_depth_moment_ratio": results["moment"]["standard"][
            "flat_delivered_over_target"
        ],
        "geometric_duration_ratio_flat_over_curved": control["rupture_duration_flat_s"]
        / control["rupture_duration_curved_s"],
        "with_depth_duration_ratio_flat_over_curved": full["rupture_duration_flat_s"]
        / full["rupture_duration_curved_s"],
    }

    step = time.perf_counter()
    results["true_depth"] = true_depth_report(
        pair,
        samplers,
        taper_weight,
        areas,
        curved_standard,
        results["hypocentres"],
        interface.decomposed_sites,
        saved,
        magnitude,
    )
    results["true_depth"]["refactor"] = refactor_numbers(results)
    results["attribution"]["by_site"] = by_site(results, interface.decomposed_sites)
    timings["true_depth_s"] = time.perf_counter() - step

    if interface is HIKURANGI_INTERFACE and magnitude == model.MAGNITUDE:
        # The survey builds meshes and measures their geometry; nothing in it reads the
        # event. Rerunning it at a second magnitude would spend four minutes producing
        # the same table, so the magnitude that already carries it keeps it.
        step = time.perf_counter()
        results["resolution"] = resolution.survey()
        timings["resolution_survey_s"] = time.perf_counter() - step

    timings["total_s"] = time.perf_counter() - started
    results["run"] = {**timings, "peak_memory_gb": peak_memory_gb()}
    return results, saved


def by_site(results: dict, sites: tuple[str, ...]) -> dict:
    """The same attribution the central pair supports, at every decomposed hypocentre.

    Three rows per site, from three runs that differ only in what they are allowed to
    read: the constant control has no depth anywhere, the counterfactual has the
    interface's depth, and the standard row has the plane's. So the **geometric** part is
    the control, the **depth** part is what the standard row adds to it, and the two sum
    to the whole by construction rather than by fitting.

    The fourth number is the one that says whether the cheap run stands in for the
    expensive one. ``constant_minus_true_depth_delta_travel_median_s`` is the gap between
    the control and the counterfactual, and it is reported beside the error it has to be
    small against: two models with completely different material fields agreeing to a
    tenth of a second on a several-second error is what licenses reading the control as a
    refactored model, and a site where they disagree is a site where it cannot be.

    ``geometric_fraction_of_delta_travel_median`` can be negative or exceed one, and that
    is information rather than a defect: the two parts are signed and need not share a
    sign, so a geometric part that runs early under a depth part that runs late gives a
    fraction outside ``[0, 1]`` and says the two mechanisms oppose each other.

    Parameters
    ----------
    results : dict
        This interface's groups, with ``scenarios`` and ``true_depth`` filled in.
    sites : tuple of str
        From :attr:`Interface.decomposed_sites`.

    Returns
    -------
    dict
        Keyed by site.
    """
    split: dict[str, dict] = {}
    for site in sites:
        control = results["scenarios"][f"{site}_constant"]
        full = results["scenarios"][f"{site}_standard"]
        counterfactual = results["true_depth"]["scenarios"][
            f"{site}_standard_truedepth"
        ]
        geometric = control["delta_travel_time_flat_minus_curved_median_s"]
        with_depth = full["delta_travel_time_flat_minus_curved_median_s"]
        refactored = counterfactual["delta_travel_time_flat_minus_curved_median_s"]
        split[site] = {
            "geometric_delta_travel_median_s": geometric,
            "geometric_delta_travel_p90_s": control[
                "delta_travel_time_flat_minus_curved_p90_s"
            ],
            "geometric_delta_travel_max_absolute_s": control[
                "delta_travel_time_flat_minus_curved_max_absolute_s"
            ],
            "with_depth_delta_travel_median_s": with_depth,
            "with_depth_delta_travel_p90_s": full[
                "delta_travel_time_flat_minus_curved_p90_s"
            ],
            "with_depth_delta_travel_max_absolute_s": full[
                "delta_travel_time_flat_minus_curved_max_absolute_s"
            ],
            "depth_share_of_delta_travel_median_s": with_depth - geometric,
            "geometric_fraction_of_delta_travel_median": geometric / with_depth,
            "geometric_delta_onset_median_s": control[
                "delta_onset_flat_minus_curved_median_s"
            ],
            "with_depth_delta_onset_median_s": full[
                "delta_onset_flat_minus_curved_median_s"
            ],
            "depth_share_of_delta_onset_median_s": full[
                "delta_onset_flat_minus_curved_median_s"
            ]
            - control["delta_onset_flat_minus_curved_median_s"],
            "true_depth_delta_travel_median_s": refactored,
            "true_depth_delta_onset_median_s": counterfactual[
                "delta_onset_flat_minus_curved_median_s"
            ],
            "constant_minus_true_depth_delta_travel_median_s": geometric - refactored,
            "constant_minus_true_depth_over_with_depth": abs(geometric - refactored)
            / abs(with_depth),
            "value_of_the_refactor_delta_travel_median_s": with_depth - refactored,
            "geometric_duration_ratio_flat_over_curved": control[
                "rupture_duration_flat_s"
            ]
            / control["rupture_duration_curved_s"],
            "with_depth_duration_ratio_flat_over_curved": full[
                "rupture_duration_flat_s"
            ]
            / full["rupture_duration_curved_s"],
        }
    return split


def refactor_numbers(results: dict) -> dict:
    """The two numbers a decision about the refactor turns on.

    **What the refactor is worth** is the gap between the two standard rows: the flat
    model's onset error with the plane's materials, minus its error with the interface's.
    **What it cannot buy** is the true-depth row's own remaining gap to the curved model,
    which is the plane's shorter paths and smaller areas and is not a material at all.

    Both are quoted on the median, the area-weighted median, the p90 and the largest
    absolute error, because a signed quantity that nearly cancels in the median can still
    be large at a subfault -- and on the delivered moment, where the same split is exact
    rather than statistical.

    Parameters
    ----------
    results : dict
        This interface's groups, with ``scenarios``, ``moment`` and ``true_depth``
        already filled in.

    Returns
    -------
    dict
    """
    status_quo = results["scenarios"]["central_standard"]
    control = results["scenarios"]["central_constant"]
    numbers: dict = {}
    for condition, _ in TRUE_DEPTH_CONDITIONS:
        counterfactual = results["true_depth"]["scenarios"][
            f"central_standard_{condition}"
        ]
        for quantity in (
            "delta_travel_time_flat_minus_curved_median_s",
            "delta_travel_time_flat_minus_curved_area_weighted_median_s",
            "delta_travel_time_flat_minus_curved_p90_s",
            "delta_travel_time_flat_minus_curved_max_absolute_s",
            "delta_onset_flat_minus_curved_median_s",
        ):
            numbers[f"{condition}_{quantity}"] = counterfactual[quantity]
            numbers[f"value_of_the_refactor_{condition}_{quantity}"] = (
                status_quo[quantity] - counterfactual[quantity]
            )
        numbers[f"irreducible_geometric_cost_{condition}_median_s"] = counterfactual[
            "delta_travel_time_flat_minus_curved_median_s"
        ]
        numbers[f"{condition}_moment_delivered_over_target"] = results["true_depth"][
            "moment"
        ][condition]["flat_delivered_over_target"]
        numbers[f"value_of_the_refactor_{condition}_moment_ratio"] = (
            results["true_depth"]["moment"][condition]["flat_delivered_over_target"]
            / results["moment"]["standard"]["flat_delivered_over_target"]
        )
    numbers["status_quo_delta_travel_median_s"] = status_quo[
        "delta_travel_time_flat_minus_curved_median_s"
    ]
    numbers["constant_control_delta_travel_median_s"] = control[
        "delta_travel_time_flat_minus_curved_median_s"
    ]
    numbers["status_quo_moment_delivered_over_target"] = results["moment"]["standard"][
        "flat_delivered_over_target"
    ]
    # How much of the standard row's error the counterfactual removes, as a fraction.
    # Read on the median because that is the number the study's headline quotes. It can
    # exceed 1, and that is information rather than a defect: it means the residual sits
    # on the *other* side of zero, so correcting the materials overshoots -- which is
    # what happens on an interface whose depth error is small enough that the plane's
    # shorter paths are comparable to it.
    numbers["fraction_of_onset_error_the_refactor_removes"] = 1.0 - (
        results["true_depth"]["scenarios"]["central_standard_truedepth"][
            "delta_travel_time_flat_minus_curved_median_s"
        ]
        / status_quo["delta_travel_time_flat_minus_curved_median_s"]
    )
    return numbers


def tag(magnitude: float) -> str:
    """The filename suffix that names a magnitude, empty at the study's own.

    The Mw 8.5 rasters, figures and rupture files are published under bare names and are
    read by paths written down elsewhere, so the baseline magnitude cannot gain a suffix
    without breaking them. Every other magnitude carries one, which is what keeps two
    magnitudes' outputs in one directory without either overwriting the other.

    Parameters
    ----------
    magnitude : float

    Returns
    -------
    str
        ``""`` at :data:`~curvature.model.MAGNITUDE`, else ``"_mw<magnitude>"``.
    """
    return "" if magnitude == model.MAGNITUDE else f"_mw{magnitude:g}"


def prefix(interface_name: str, magnitude: float) -> str:
    """What one interface's outputs at one magnitude are named with.

    One rule, here rather than in each of the three programs that writes files, because
    a figure, a raster and a rupture file that disagreed about the prefix would be three
    sets of outputs nobody could match up. Hikurangi at :data:`~curvature.model.MAGNITUDE`
    takes no prefix at all, which is what keeps the published figures at the paths the
    document already writes down.

    Parameters
    ----------
    interface_name : str
    magnitude : float

    Returns
    -------
    str
        Empty, or a trailing-underscore prefix such as ``puyseguer_mw8.67_``.
    """
    stem = "" if interface_name == HIKURANGI_INTERFACE.name else interface_name
    named = f"{stem}{tag(magnitude)}".lstrip("_")
    return f"{named}_" if named else ""


def merged(groups: dict, interface: Interface, magnitude: float) -> dict:
    """This interface's groups, folded into whatever ``results.json`` already holds.

    Reading the file back rather than writing it fresh is what lets the three
    interfaces be run in any order and singly: a rerun of one replaces its own groups
    and leaves the others exactly as they were, which is also the property the published
    document needs from Hikurangi's numbers.

    **Magnitude is a dimension of the file rather than a rewrite of it.** The baseline
    magnitude keeps the layout the document quotes -- Hikurangi at the top level, the two
    Puysegur surfaces under ``puysegur`` -- and every other magnitude lands under
    ``magnitudes/mw_<magnitude>/<interface>``, one flat shape for all three surfaces
    because nothing there is published under a path yet. So a control at a second
    magnitude adds keys and moves none, and the two are read side by side rather than one
    replacing the other.

    Parameters
    ----------
    groups : dict
        What :func:`study` returned.
    interface : Interface
    magnitude : float
        Which magnitude the groups were produced at.

    Returns
    -------
    dict
        The whole file's contents.
    """
    path = HERE / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    if magnitude != model.MAGNITUDE:
        by_magnitude = results.setdefault("magnitudes", {})
        by_magnitude.setdefault(f"mw_{magnitude:g}", {})[interface.name] = groups
        return results
    if interface is HIKURANGI_INTERFACE:
        results.update(groups)
        return results
    results.setdefault("puysegur", {})[interface.name] = groups
    return results


def replaced(path: Path, write: Callable[[Path], None]) -> None:
    """Write a file under a temporary name in its own directory and rename it into place.

    Both outputs here are read while they are being rewritten -- ``results.json`` by a
    document build and the rasters by :mod:`curvature.figures` -- and a plain write
    truncates before it fills, so a reader that arrives in between sees a zero-byte file
    and no error. :meth:`pathlib.Path.replace` is atomic within a filesystem, so a reader
    sees either the whole old file or the whole new one and never a half of either.

    The partial is removed if the write fails, so a crash leaves no rubble behind under a
    name nobody recognises.

    Parameters
    ----------
    path : Path
        Where the file belongs.
    write : callable
        Called with the temporary path. Its extension is ``path``'s, since writers that
        dispatch on it -- ``np.savez_compressed`` appends ``.npz`` when it is missing --
        would otherwise write somewhere else entirely.
    """
    partial = path.with_name(f".partial-{path.name}")
    try:
        write(partial)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    """Run one interface at one magnitude and merge its results in."""
    warnings.simplefilter("always", DegradedSeed)
    name = sys.argv[1] if len(sys.argv) > 1 else HIKURANGI_INTERFACE.name
    if name not in INTERFACES:
        raise SystemExit(f"no such interface {name!r}: choose from {list(INTERFACES)}")
    interface = INTERFACES[name]
    magnitude = float(sys.argv[2]) if len(sys.argv) > 2 else model.MAGNITUDE

    groups, saved = study(interface, magnitude)
    stem = "arrays" if interface is HIKURANGI_INTERFACE else interface.name
    (HERE / "data").mkdir(exist_ok=True)
    replaced(
        HERE / "data" / f"{stem}{tag(magnitude)}.npz",
        lambda partial: np.savez_compressed(partial, **saved),
    )
    replaced(
        HERE / "results.json",
        lambda partial: partial.write_text(
            json.dumps(merged(groups, interface, magnitude), indent=2, sort_keys=False)
        ),
    )
    print(json.dumps(groups["run"], indent=2))
    print(json.dumps(groups["attribution"], indent=2))
    print(json.dumps(groups["true_depth"]["refactor"], indent=2))


if __name__ == "__main__":
    main()
