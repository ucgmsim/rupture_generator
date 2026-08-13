"""One rupture model, run on whichever of the two geometries it is handed.

The stages are the package's own -- :mod:`rupture_generator.stages`,
:mod:`rupture_generator.moment`, :mod:`rupture_generator.timing`,
:mod:`rupture_generator.pulses` and :mod:`rupture_generator.triangular.fim` -- called
here rather than transcribed, so the two models differ in their *geometry* and in
nothing else. What this module adds is the wiring: a sampler bound to one set of
vertices, a taper that is deliberately shared, and the two velocity settings that make
the depth contribution measurable instead of arguable.

**Everything not geometric is held identical between the models, on purpose.** The seeds
per stage, the correlation lengths, the coefficient of variation, the taper weights, the
target moment, the scalar ``alpha_T`` geometric correction and the hypocentre's face
index are all one value used twice. A quantity that differed would be a second
explanation for every difference measured, and the experiment would stop being a
controlled one.

``alpha_T`` deserves the note. :func:`~rupture_generator.timing.alpha_t` takes a fault's
*average* dip and rake and returns a scalar that shortens the rise time and raises the
rupture speed. A literature workflow reads that average off the plane it built, so the
plane's own dip is used for both models here. The per-subfault depth dependence -- which
is where the flat model actually goes wrong -- is left free to differ, because that is
the thing being measured.

**The taper is shared, and that is the generous choice again.** The production side
taper ramps slip to zero over 2% of the fault's parameter extent. Both models have
identical ``(u, v)``, so a taper measured in the parameter plane is bit-identical between
them and contributes exactly nothing to any difference reported. Measuring it on the true
surface instead would make the flat model wrong in one more place.

The velocity control
--------------------

The flat model is wrong about two different things at once. Its **paths** are shorter,
because a chord across a curved surface is shorter than the surface, and its **depths**
are wrong, because the best-fit plane cuts through the interface rather than following
it. Both change the onset, both change the moment, and an argument about which dominates
is worth less than a measurement.

:data:`CONSTANT` is that measurement's control. It is not merely "constant velocities":
several stages read depth *directly* rather than through the velocity model, and a
control that left those live would smuggle a depth effect into the run that is supposed
to have none. Every one of them is neutralised, and each through the public API rather
than by patching:

======================================  ==============================================
what reads depth                        how it is neutralised
======================================  ==============================================
``moment.sample_velocity_model``         one layer, one shear speed, one density, so
                                         ``mu`` is a single number everywhere
``timing.SpeedParams.depth_factor``      ``shallow_factor = deep_factor = 1``, which
                                         makes the factor exactly 1 at every depth
``stages.RiseTimeParams.stretch_at``     ``shallow_factor = deep_factor = 1``, likewise
``stages.RiseTimeParams.shallow_blend``  a ramp placed at -1000 km, entirely above the
                                         shallowest face either model has (-17.3 km),
                                         so its weight is exactly 1 everywhere
``pulses.PulseParams.beta_at``           ``beta_shallow = beta_mid = beta_deep``, so
                                         the rising fraction does not vary with depth
======================================  ==============================================

Two stages that might be expected here are absent, and their absence is a finding rather
than an oversight. :class:`~rupture_generator.stages.RakeParams` and
:class:`~rupture_generator.stages.OnsetParams` read **no depth at all** -- rake is
``base + sigma * Z`` and the onset perturbation is a standardised correlated field, so
neither has a depth ramp to neutralise. The three ``DepthRamp`` consumers in this
package are rise time, rupture speed and the pulse's rising fraction, and that is the
complete list.

With all five neutralised the eikonal sees a **uniform slowness field** and the moment
fold sees a single rigidity, so every difference between the two geometries in a
:data:`CONSTANT` run is geometric. Subtracting that run from the matching
:data:`STANDARD` one is the depth-driven part.

The counterfactual
------------------

Those two settings bracket the flat model's two errors but do not separate them: the
control has neither the wrong depths nor the wrong rock, and :data:`STANDARD` has both.
:func:`true_depth_materials` builds the model in between -- the plane's positions, the
interface's rock -- which is what a refactor that assigned material properties from the
mesh geometry rather than from the rupture sampler would deliver. It is the only one of
the three that is not a run of the shipped pipeline, and the only reason it needs new
code is that ``Materials`` carries the depth the ramps read as well as the depth the rock
was sampled at, so the two can be told apart.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial import cKDTree

from curvature.geometry import MeshPair
from rupture_generator import moment, pulses, stages, timing
from rupture_generator.sampling import (
    VonKarmanFilterParameters,
    correlation_lengths,
)
from rupture_generator.triangular import fim, spde

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

MAGNITUDE = 8.5
"""The event. A whole-interface Hikurangi rupture, which is what makes the parameter
domain 821 x 329 km and so puts the fault 14.6 by 15.3 correlation lengths across --
comfortably outside the range where the SPDE's Neumann boundary folding is the field.

**The same magnitude serves Puysegur, and that is a choice rather than an inheritance.**
Puysegur is 66 to 76 thousand square kilometres against Hikurangi's 178, so the obvious
worry is that it cannot carry an Mw 8.5. Two bounds say what it can carry and
:func:`~curvature.run.magnitude_ladder` reports both at five magnitudes on each surface.
Neither bites here: the correlation lengths are 0.08 and 0.13 of the parameter extents,
against the 0.6 of Mai & Beroza figure 13's upper end, and the domain is 12.1 by 7.9
correlation lengths across, against the 4 the Neumann folding needs. The first magnitude
either bound refuses is 9.5, which folds at 3.9 by 3.7.

What settles it is that the magnitude is not free to differ. It sets the correlation
lengths, so running Puysegur at a smaller magnitude would change the field's structure
as well as the surface and the comparison would stop being about geometry. Mw 9.0 passes
both bounds too and is rejected for the same reason and one more: five and a half times
the moment on 40% of the area is a mean slip near ten metres, which is a different class
of event rather than the same event on a second interface.
"""

VELOCITY_FRACTION = 0.8
"""The rupture speed as a fraction of the shear speed, before the depth ramps."""

AVERAGE_RAKE_DEG = 90.0
"""Pure reverse slip, which is what a subduction interface does, and which puts
:func:`~rupture_generator.timing.alpha_t`'s rake factor at full strength."""

RISE_TIME_COEFFICIENT = 1.6
COEFFICIENT_OF_VARIATION = 0.75
SIDE_TAPER = 0.02
ONSET_SCALE_S = -0.35
BASE_RAKE_DEG = 90.0
RAKE_SIGMA_DEG = 15.0
"""Production values, from :class:`~rupture_generator.config.rupture.SlipConfig` and
:class:`~rupture_generator.config.rupture.TimingConfig`, except the base rake, which is
:data:`AVERAGE_RAKE_DEG` because this is a subduction thrust rather than the crustal
strike-slip event those defaults were written for."""

SAMPLE_INTERVAL_S = 0.02
"""The slip-rate pulse's sample interval, and the study's one departure from production.

Production is 0.005 s. This is four times that, and the reason is that **the mesh, not
the sample interval, is what limits the moment rate spectrum**. A 500 m subfault at a
~3 km/s rupture speed is crossed in 0.167 s, so the discretisation stops being
trustworthy above about 6 Hz; the Nyquist frequency at this interval is 25 Hz. The
binding constraint is therefore four times below the one being chosen, which is the
regime where the choice cannot shape the result. An Mw 8.5 corner frequency is near
0.01 Hz, three orders below either.

What it costs is real and is stated rather than hidden. ``dt`` sets a **floor on
representable rise time**: a pulse shorter than one sample integrates to nothing, and a
subfault slipping several metres would become indistinguishable from one that did not
slip. Quadrupling ``dt`` quadruples that floor.
:func:`~rupture_generator.stages.rise_time_field` already clamps its output to exactly
one sample interval, so the floor binds *there* rather than in the pulse kernel -- but
whether the kernel raises ``UnrepresentableRiseTime`` anyway is measured and reported,
not worked around.
"""

BOUNDARY_SAMPLES_PER_EDGE = 8
"""How finely the boundary polyline is resampled before distances are taken to it.

The same construction and the same number as
:data:`rupture_generator.triangular.pipeline.BOUNDARY_SAMPLES_PER_EDGE`: a polyline
sampled at spacing ``s`` reports a distance too large by at most ``s / 2``, so the
spacing is a fraction of whichever of the ramp width and the boundary edge length is
longer.
"""

SEEDS = {"slip": 20260813, "rise_time": 20260814, "rake": 20260815, "onset": 20260816}
"""One stream per stage, so changing one stage cannot move another's noise.

The **same** four integers serve both geometries and both velocity settings. That is the
experiment: identical vertex counts mean ``rng.standard_normal(V)`` returns the same
vector, so the two models are handed the same white noise and every difference is
geometry.
"""


@dataclasses.dataclass(frozen=True)
class VelocityModel:
    """A 1-D velocity model, and whether the depth ramps around it are live.

    Attributes
    ----------
    name : str
        ``standard`` or ``constant``.
    bottom_depth_km, shear_speed_km_s, density_g_cm3 : tuple of float
        The layers, shallow to deep. A depth below the last one takes that layer's
        properties rather than an extrapolation, and a depth *above* the first --
        which the flat model produces, because the best-fit plane rises above sea
        level over the trench-ward third of Hikurangi -- takes the first layer's.
    depth_ramps_live : bool
        Whether the depth ramps in the speed model, the rise-time model and the pulse
        shape are left as production has them. False is the control; see this module's
        docstring for the table of what that switches off and how.
    """

    name: str
    bottom_depth_km: tuple[float, ...]
    shear_speed_km_s: tuple[float, ...]
    density_g_cm3: tuple[float, ...]
    depth_ramps_live: bool


STANDARD = VelocityModel(
    name="standard",
    bottom_depth_km=(
        2.80,
        3.00,
        3.20,
        3.40,
        3.60,
        3.80,
        4.00,
        4.20,
        4.40,
        4.60,
        4.80,
        5.00,
        8.00,
        12.00,
        27.00,
        39.00,
        1038.00,
    ),
    shear_speed_km_s=(
        2.490,
        2.700,
        2.770,
        2.840,
        2.910,
        2.980,
        3.050,
        3.120,
        3.190,
        3.260,
        3.330,
        3.400,
        3.600,
        3.600,
        3.700,
        4.300,
        4.600,
    ),
    density_g_cm3=(
        2.440,
        2.480,
        2.490,
        2.510,
        2.520,
        2.540,
        2.560,
        2.580,
        2.600,
        2.610,
        2.630,
        2.660,
        2.720,
        2.720,
        2.830,
        3.120,
        3.330,
    ),
    depth_ramps_live=True,
)
"""The shipped ``examples/colombia.toml`` model verbatim.

Chosen because it is a real New Zealand subduction model that already reaches past this
interface's 75 km, and because its layering is **fine where it matters**: twelve
boundaries between 2.8 and 5.0 km, then 8, 12, 27 and 39 km. A model with three thick
layers would hide the layer-crossing effect a 20 km depth error produces; this one does
not, which is the property a controlled experiment wants from it.
"""

CONSTANT = VelocityModel(
    name="constant",
    bottom_depth_km=(1.0e6,),
    shear_speed_km_s=(3.5,),
    density_g_cm3=(2.8,),
    depth_ramps_live=False,
)
"""The control: one layer, and every depth ramp flattened. See the module docstring.

The shear speed and density are the middle of :data:`STANDARD`'s seismogenic range
rather than an average of it -- what matters is that they are *constants*, since every
quantity the control reports is a ratio between two models that share them.
"""

NEUTRALISED_BLEND = stages.DepthRamp(centre_km=-1000.0, half_width_km=1.0)
"""The rise-time model's shallow blend, placed where it cannot fire.

:meth:`~rupture_generator.stages.DepthRamp.weight` is 1 below the ramp, and this ramp
finishes at -999 km. The shallowest face either geometry produces is at -17.3 km, so the
weight is exactly 1 at every face of both models and the blend towards slip's own
Gaussian never happens. Setting it at the surface would *not* do -- the flat model's
trench-ward faces sit above sea level, so a ramp near zero would fire there and only
there, which is precisely the depth dependence the control exists to remove.
"""


@dataclasses.dataclass(frozen=True)
class Materials:
    """What rock each subfault is in, read at its own centre depth.

    Attributes
    ----------
    depth_km : FloatArray
        ``(F,)`` face centre depth, positive down. **Negative in the flat model
        wherever the best-fit plane rises above sea level.**
    shear_speed_km_s, rigidity_pa : FloatArray
        ``(F,)`` from the 1-D model.
    layer : IntArray
        ``(F,)`` which layer each face landed in. Compared between the models to count
        the faces the flat geometry puts in different rock entirely.
    """

    depth_km: FloatArray
    shear_speed_km_s: FloatArray
    rigidity_pa: FloatArray
    layer: IntArray


@dataclasses.dataclass(frozen=True)
class Fields:
    """The drawn per-face fields of one model, before the moment sizes them.

    Attributes
    ----------
    pattern : FloatArray
        ``(F,)`` the truncated, tapered, dimensionless slip pattern.
    gaussian : FloatArray
        ``(F,)`` the standardised Gaussian the pattern came from -- the field to
        correlate between models, because it is the one the two geometries act on
        directly and the one the truncation has not yet bent.
    draw : FloatArray
        ``(F,)`` the same field **before** standardising. Its own spread is what
        ``sampling.standardise`` divides out, so keeping it is what makes "which
        findings survive standardise" answerable.
    rise_time_s : FloatArray
        ``(F,)`` seconds.
    rake_deg : FloatArray
        ``(F,)`` degrees. Drawn because the native rupture file carries it, and
        because it is the one field with **no depth dependence at all** -- so any
        difference between the models in it is purely the sampler's geometry.
    perturbation : FloatArray
        ``(F,)`` dimensionless, standardised, correlated with slip.
    truncated_fraction : float
        What fraction of the fault the non-negativity truncation clipped.
    """

    pattern: FloatArray
    gaussian: FloatArray
    draw: FloatArray
    rise_time_s: FloatArray
    rake_deg: FloatArray
    perturbation: FloatArray
    truncated_fraction: float


class _Chart:
    """The whole of what a field stage asks of a chart: where its subfaults are.

    :mod:`rupture_generator.stages` takes a ``Chart`` protocol with a single
    ``centres()`` method, because three of its stages read nothing but the centre
    depths. This supplies it without dragging in a mesh container.
    """

    def __init__(self, centres_km: FloatArray) -> None:
        """Hold the face centres.

        Parameters
        ----------
        centres_km : FloatArray
            ``(F, 3)`` positions, depth last.
        """
        self._centres_km = centres_km

    def centres(self) -> FloatArray:
        """Subfault centres, ``(F, 3)``, depth last."""
        return self._centres_km


def covariance() -> VonKarmanFilterParameters:
    """Mai & Beroza's correlation lengths at :data:`MAGNITUDE`."""
    return correlation_lengths(MAGNITUDE)


def materials_of(depth_km: FloatArray, velocity: VelocityModel) -> Materials:
    """Sample a 1-D velocity model at each subfault's own depth.

    Parameters
    ----------
    depth_km : FloatArray
        ``(F,)`` face centre depths, positive down.
    velocity : VelocityModel

    Returns
    -------
    Materials
    """
    bottoms = np.asarray(velocity.bottom_depth_km)
    shear_speed, rigidity = moment.sample_velocity_model(
        depth_km,
        bottoms,
        np.asarray(velocity.shear_speed_km_s),
        np.asarray(velocity.density_g_cm3),
    )
    return Materials(
        depth_km=depth_km,
        shear_speed_km_s=shear_speed,
        rigidity_pa=rigidity,
        layer=moment.layer_of(depth_km, bottoms),
    )


def true_depth_materials(
    true_depth_km: FloatArray,
    flat_depth_km: FloatArray,
    velocity: VelocityModel,
    *,
    ramps_read_true_depth: bool,
) -> Materials:
    """The counterfactual: a flat model that reads its rock off the real interface.

    The refactor this measures is "assign material properties in the mesh geometry
    rather than in the rupture sampler". A mesh that carried its own properties would
    hand the flat twin the rigidity and shear speed of the interface the subfault
    actually sits on, while leaving the plane's shorter paths, its areas and its metric
    on the correlated field exactly as they are. This builds that model, and nothing
    else about the flat model changes.

    **What counts as a material is a decision, and it is made here.** Three stages read
    a subfault's depth *directly* rather than through the 1-D model:
    :meth:`~rupture_generator.timing.SpeedParams.depth_factor`,
    :attr:`~rupture_generator.stages.RiseTimeParams.stretch_at` with its shallow blend,
    and :meth:`~rupture_generator.pulses.PulseParams.beta_at`. They are ramps in depth,
    not in shape: each says what the rock does near the free surface and near the base
    of the seismogenic zone, so each is a property of *where the subfault sits* and
    would move with it under the refactor. ``ramps_read_true_depth=True`` is therefore
    the honest reading of the counterfactual and is the one the study reports.

    The other reading is reported beside it rather than argued away.
    ``ramps_read_true_depth=False`` corrects only the two quantities the 1-D velocity
    model returns -- rigidity and shear speed -- and leaves the ramps reading the
    plane's depth, which is what a narrower refactor that moved only ``mu`` and
    ``beta`` onto the mesh would deliver. The difference between the two is what the
    ramps are worth.

    Parameters
    ----------
    true_depth_km : FloatArray
        ``(F,)`` the face centre depths on the real interface, positive down. The rock
        is always sampled here; that is the whole of the counterfactual.
    flat_depth_km : FloatArray
        ``(F,)`` the same faces' depths on the best-fit plane, which is what the depth
        ramps read when ``ramps_read_true_depth`` is false.
    velocity : VelocityModel
    ramps_read_true_depth : bool
        Whether the three depth ramps move with the rock.

    Returns
    -------
    Materials
        Shear speed, rigidity and layer at the true depth; ``depth_km`` is whichever
        depth the ramps are to read, since that attribute is the only channel the
        ramps have.
    """
    sampled = materials_of(true_depth_km, velocity)
    if ramps_read_true_depth:
        return sampled
    return dataclasses.replace(sampled, depth_km=flat_depth_km)


def lateral_taper(pair: MeshPair, fraction: float = SIDE_TAPER) -> FloatArray:
    """The side taper, measured in the parameter plane and therefore shared.

    A rupture that slips right up to its boundary is unphysical, because the edges are
    where the fault stops. The production taper ramps slip to zero over ``fraction`` of
    the fault's along-strike extent, and
    :func:`rupture_generator.triangular.pipeline.taper_edges` measures that as a distance
    to the *lateral* part of the boundary in the parameter plane.

    Both models share ``(u, v)`` exactly, so this weight is bit-identical between them.
    Top and bottom tapers are zero in production and are not applied.

    Parameters
    ----------
    pair : MeshPair
        For its faces and parameter coordinates.
    fraction : float, optional
        The ramp width as a fraction of the along-strike parameter extent.

    Returns
    -------
    FloatArray
        ``(F,)`` weights in ``[0, 1]``.
    """
    faces = pair.faces
    parameters = pair.parameters_km
    width_km = fraction * float(np.ptp(parameters[:, 0]))

    # A boundary edge is a half-edge with no twin. Kept directed, in the order its own
    # face names it, so the interior lies to its left in the parameter plane and the
    # outward normal is that direction turned a right angle clockwise -- which is what
    # separates a lateral edge from the top and bottom ones.
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, first, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_index=True, return_counts=True
    )
    edges = directed[first[counts == 1]]

    direction = parameters[edges[:, 1]] - parameters[edges[:, 0]]
    outward_u, outward_v = direction[:, 1], -direction[:, 0]
    lateral = edges[np.abs(outward_v) <= np.abs(outward_u)]

    starts, ends = parameters[lateral[:, 0]], parameters[lateral[:, 1]]
    lengths_km = np.linalg.norm(ends - starts, axis=1)
    spacing_km = max(width_km, float(np.median(lengths_km))) / BOUNDARY_SAMPLES_PER_EDGE
    steps = np.maximum(1, np.ceil(lengths_km / spacing_km)).astype(np.int64)
    count = steps + 1
    offsets = np.cumsum(count) - count
    within = np.arange(int(count.sum())) - np.repeat(offsets, count)
    edge_of = np.repeat(np.arange(len(steps)), count)
    samples = (
        starts[edge_of]
        + (within / np.repeat(steps, count))[:, None] * (ends - starts)[edge_of]
    )

    distance_km, _ = cKDTree(samples).query(pair.face_parameters_km(), k=1, workers=-1)
    return np.clip(np.asarray(distance_km) / width_km, 0.0, 1.0)


class Sampler:
    """One assembled SPDE operator, reduced to faces, for one geometry.

    Held rather than rebuilt because the assembly and the multigrid setup are the
    expensive part and the noise is not: both velocity settings draw from the same
    operator, since the operator reads the surface and never the velocity model.
    """

    def __init__(self, pair: MeshPair, vertices_km: FloatArray, levels: list) -> None:
        """Assemble the operator on one of the two geometries.

        Parameters
        ----------
        pair : MeshPair
            For the shared faces and parameter coordinates.
        vertices_km : FloatArray
            ``(V, 3)`` this model's vertices -- the whole of what makes the two
            models different.
        levels : list
            This model's multigrid hierarchy, coarsest first.
        """
        self._faces = pair.faces
        self._operator = spde.MaternOperator(
            vertices_km, pair.faces, pair.parameters_km, covariance(), coarser=levels
        )

    @property
    def error(self) -> spde.ModelError:
        """Bolin & Kirchner theorem 3.3 instantiated for this mesh."""
        return self._operator.error

    def __call__(
        self,
        chart: object,
        asked: VonKarmanFilterParameters,
        rng: np.random.Generator,
    ) -> FloatArray:
        """One field of this model's covariance, one value per face.

        The signature :data:`rupture_generator.stages.FieldSampler` names, so the
        package's own stages take this unchanged.
        """
        del chart, asked
        return spde.face_values(self._operator.draw(rng), self._faces)


def draw_fields(
    pair: MeshPair,
    sampler: Sampler,
    vertices_km: FloatArray,
    velocity: VelocityModel,
    taper_weight: FloatArray,
) -> Fields:
    """The four drawn fields of one model, under one velocity setting.

    Slip, rake and the onset perturbation read no depth, so they are identical between
    the two velocity settings of one geometry; rise time does, through its stretch and
    its shallow blend, so it is not. All four are redrawn anyway, from the same seeds,
    because a stream that is re-consumed identically costs one draw and removes a class
    of reasoning about which fields could be shared.

    Parameters
    ----------
    pair : MeshPair
        For the shared faces.
    sampler : Sampler
        Bound to this model's geometry.
    vertices_km : FloatArray
        ``(V, 3)`` this model's vertices, for the face centres the depth ramps read.
    velocity : VelocityModel
        Whether the depth ramps are live.
    taper_weight : FloatArray
        ``(F,)`` from :func:`lateral_taper`, shared.

    Returns
    -------
    Fields
    """
    structure = covariance()
    chart = _Chart(pair.centres_km(vertices_km))
    slip_params = stages.SlipParams(
        covariance=structure,
        coefficient_of_variation=COEFFICIENT_OF_VARIATION,
        side_taper=SIDE_TAPER,
    )
    pattern, gaussian, draw = stages.slip_pattern(
        chart,
        slip_params,
        np.random.default_rng(SEEDS["slip"]),
        sampler=sampler,
        taper=lambda field, _params: field * taper_weight,
    )

    correction = timing.alpha_t(pair.frame.dip_deg, AVERAGE_RAKE_DEG)
    average_s = stages.average_rise_time_s(
        moment.seismic_moment_nm(MAGNITUDE), RISE_TIME_COEFFICIENT, correction
    )
    rise_params = (
        stages.RiseTimeParams(average_s=average_s)
        if velocity.depth_ramps_live
        else stages.RiseTimeParams(
            average_s=average_s,
            shallow_blend=NEUTRALISED_BLEND,
            shallow_factor=1.0,
            deep_factor=1.0,
        )
    )
    rise_time_s = stages.rise_time_field(
        chart,
        gaussian,
        draw,
        rise_params,
        np.random.default_rng(SEEDS["rise_time"]),
        structure,
        sample_interval_s=SAMPLE_INTERVAL_S,
        sampler=sampler,
    )
    rake_deg = stages.rake_field(
        chart,
        stages.RakeParams(
            covariance=structure,
            base_rake_deg=BASE_RAKE_DEG,
            sigma_deg=RAKE_SIGMA_DEG,
        ),
        np.random.default_rng(SEEDS["rake"]),
        sampler=sampler,
    )
    perturbation = stages.onset_perturbation(
        chart,
        draw,
        stages.OnsetParams(scale_s=ONSET_SCALE_S),
        np.random.default_rng(SEEDS["onset"]),
        structure,
        sampler=sampler,
    )
    return Fields(
        pattern=pattern,
        gaussian=gaussian,
        draw=draw,
        rise_time_s=rise_time_s,
        rake_deg=rake_deg,
        perturbation=perturbation,
        truncated_fraction=stages.truncated_fraction(gaussian, slip_params),
    )


def slip_metres(
    pattern: FloatArray,
    rigidity_pa: FloatArray,
    areas_km2: FloatArray,
    magnitude: float = MAGNITUDE,
) -> FloatArray:
    """Size a dimensionless pattern into slip, in metres, against a target moment.

    The one global fold, and the place the flat model's error enters the moment: a
    literature workflow divides by ``sum(mu A f)`` computed on the **plane**, so the
    factor it gets is wrong by exactly the ratio of that sum to the true one.

    Parameters
    ----------
    pattern : FloatArray
        ``(F,)`` dimensionless and non-negative.
    rigidity_pa, areas_km2 : FloatArray
        ``(F,)`` the rigidity and area the *scaling* is told to use.
    magnitude : float, optional
        The target.

    Returns
    -------
    FloatArray
        ``(F,)`` slip in metres.
    """
    return moment.scale_to_moment(
        [pattern], [rigidity_pa], [areas_km2], moment.seismic_moment_nm(magnitude)
    )[0]


def speed_params(pair: MeshPair, velocity: VelocityModel) -> timing.SpeedParams:
    """The rupture-speed model, with the **plane's** average dip and a thrust rake.

    The scalar geometric correction is deliberately the same object for both geometries
    -- see this module's docstring. What is not shared is the per-subfault depth the
    ramps and the shear speed are read at, which is the mechanism under test, and under
    :data:`CONSTANT` the ramps are switched off by setting both factors to exactly 1.

    Parameters
    ----------
    pair : MeshPair
        For the frame's dip.
    velocity : VelocityModel

    Returns
    -------
    timing.SpeedParams
    """
    factors = (
        {} if velocity.depth_ramps_live else {"shallow_factor": 1.0, "deep_factor": 1.0}
    )
    return timing.SpeedParams(
        velocity_fraction=VELOCITY_FRACTION,
        average_dip_deg=pair.frame.dip_deg,
        average_rake_deg=AVERAGE_RAKE_DEG,
        **factors,
    )


def pulse_params(velocity: VelocityModel) -> pulses.PulseParams:
    """How every subfault's slip-rate pulse is shaped and sampled.

    Under :data:`CONSTANT` the three depth-dependent rising fractions are set equal, so
    :meth:`~rupture_generator.pulses.PulseParams.beta_at` returns one number at every
    depth and the ramps around it cannot fire.

    Parameters
    ----------
    velocity : VelocityModel

    Returns
    -------
    pulses.PulseParams
    """
    shape = pulses.from_stype("OliuP2")
    if velocity.depth_ramps_live:
        return pulses.PulseParams(shape=shape, sample_interval_s=SAMPLE_INTERVAL_S)
    return pulses.PulseParams(
        shape=shape,
        beta_shallow=0.13,
        beta_mid=0.13,
        beta_deep=0.13,
        sample_interval_s=SAMPLE_INTERVAL_S,
    )


def travel_times(
    pair: MeshPair,
    vertices_km: FloatArray,
    materials: Materials,
    params: timing.SpeedParams,
    hypocentre_face: int,
    *,
    threads: int = 8,
) -> tuple[FloatArray, tuple[fim.SeedReport, ...]]:
    """First-arrival times per face, seconds, from one hypocentre, and the seed reports.

    The front is seeded at all three corners of the hypocentre's face, at ``t = 0``.
    Seeding one corner would leave that face arriving about ``0.4 h S`` late -- a bias
    in the one quantity the model's own perturbation gives no cover for -- and seeding
    the whole face is the faithful reading of a structured solver's seed cell.

    Parameters
    ----------
    pair : MeshPair
        For the shared faces.
    vertices_km : FloatArray
        ``(V, 3)`` the geometry the front propagates over, which sets the **paths**.
    materials : Materials
        This model's depths and shear speeds, which set the **speed**.
    params : timing.SpeedParams
    hypocentre_face : int
        Which face the rupture starts in -- the **same index** in both models, because
        the faces are the same faces.
    threads : int, optional
        Workers for the Rust backend. The answer is bit-identical at any thread count.

    Returns
    -------
    tuple
        ``(F,)`` arrivals in seconds -- the mean of each face's three corners, which is
        exactly the piecewise-linear solution evaluated at the centroid -- and the seed
        reports. The reports are returned rather than discarded because the analytic
        ball's radius is *derived* from the mesh and the velocity model, and its two
        bounds are checkable on the mesh handed in: a derived quantity nobody measures
        is a configured one that has stopped being written down.
    """
    speed_km_s = timing.speed_field(
        materials.depth_km, materials.shear_speed_km_s, params
    )
    seeds = [fim.Seed(int(vertex), 0.0) for vertex in pair.faces[hypocentre_face]]
    vertex_times_s, reports = fim.solve_with_report(
        vertices_km,
        pair.faces,
        1.0 / speed_km_s,
        seeds,
        backend=fim.KERNEL,
        threads=threads,
    )
    return fim.face_arrivals(pair.faces, vertex_times_s), reports


def onset_of(
    travel_time_s: FloatArray, perturbation: FloatArray, hypocentre_face: int
) -> FloatArray:
    """Onset from travel time plus the drawn perturbation, with the hypocentre pinned.

    Parameters
    ----------
    travel_time_s, perturbation : FloatArray
        ``(F,)`` from :func:`travel_times` and :func:`draw_fields`.
    hypocentre_face : int
        Pinned to zero perturbation, so its onset is exactly its travel time.

    Returns
    -------
    FloatArray
        ``(F,)`` seconds.
    """
    return stages.apply_perturbation(
        travel_time_s,
        perturbation,
        stages.OnsetParams(scale_s=ONSET_SCALE_S),
        hypocentre=hypocentre_face,
        delay_s=0.0,
    )


def moment_rate(
    slip_m: FloatArray,
    rise_time_s: FloatArray,
    onset_s: FloatArray,
    depth_km: FloatArray,
    weights_nm_per_m: FloatArray,
    params: pulses.PulseParams,
    *,
    chunk: int = 8192,
) -> tuple[FloatArray, FloatArray]:
    """The moment rate function, accumulated in chunks over the faces.

    :func:`rupture_generator.moment.moment_rate` builds the whole CSR pulse array first,
    and at 1.87 M faces with a mean rise time near 6 s that is about 1.9e9 samples -- 15 GB
    at f64, which is most of this machine's memory. The moment rate is a *sum* over
    faces, so no pulse has to outlive its own contribution: a chunk of faces is
    synthesised, added into a timeline fixed in advance, and discarded. Peak memory is
    ``chunk`` pulses rather than ``F`` of them, and the answer is identical.

    The arithmetic is that function's: each subfault's pulse is placed at its own onset,
    quantised to the sample interval, and weighted. Interpolating the onset instead
    would smear each pulse across two samples and change the peak, which is the number
    people read off this.

    **The sample interval is left at production's 0.005 s.** It is not free -- it sets
    the rise-time floor in :func:`~rupture_generator.stages.rise_time_field` and so is a
    modelling choice rather than a plotting one -- and incremental accumulation makes it
    affordable, so there is no reason to coarsen it. The accumulated timeline is a few
    hundred seconds, which is under a megabyte.

    Parameters
    ----------
    slip_m, rise_time_s, onset_s, depth_km : FloatArray
        ``(F,)`` per subfault. ``depth_km`` is read only for the pulse's rising
        fraction, and under :data:`CONSTANT` not even for that.
    weights_nm_per_m : FloatArray
        ``(k, F)`` or ``(F,)``: what each subfault's slip rate is multiplied by, which
        is ``mu A`` in newton-metres per metre of slip. Several weightings are
        accumulated in one pass because the expensive part is synthesising the pulses,
        and the flat model has two readings worth reporting -- the moment it *claims*,
        on its own areas and rigidities, and the moment it *delivers* once its slip is
        placed on the true surface.
    params : pulses.PulseParams
    chunk : int, optional
        How many subfaults are synthesised at once. Sets the peak memory, not the
        answer.

    Returns
    -------
    tuple of FloatArray
        Times in seconds from the first onset, and ``(k, T)`` moment rate in
        newton-metres per second.
    """
    weights = np.atleast_2d(weights_nm_per_m)
    interval_s = params.sample_interval_s
    first_s = float(onset_s.min())
    starts = np.rint((onset_s - first_s) / interval_s).astype(np.int64)
    # The longest pulse the shape can produce sets the timeline; `duration_scale` is
    # how much longer than the rise time the resolved shape runs.
    span = (
        int(
            starts.max()
            + np.ceil(rise_time_s.max() * params.shape.duration_scale / interval_s)
        )
        + 2
    )
    rate = np.zeros((len(weights), span), dtype=np.float64)

    for start in range(0, len(slip_m), chunk):
        stop = min(start + chunk, len(slip_m))
        offsets, samples = pulses.synthesise(
            depth_km[start:stop], slip_m[start:stop], rise_time_s[start:stop], params
        )
        lengths = np.diff(offsets)
        for local in range(stop - start):
            length = int(lengths[local])
            if length == 0:
                # A subfault the taper or the truncation left at zero slip has no
                # pulse at all -- no samples, which is not a pulse of zeros.
                continue
            index = start + local
            begin = int(starts[index])
            pulse = samples[offsets[local] : offsets[local] + length]
            for row in range(len(weights)):
                rate[row, begin : begin + length] += weights[row, index] * pulse

    return np.arange(span, dtype=np.float64) * interval_s + first_s, rate


def amplitude_spectrum(
    rate_nm_s: FloatArray, sample_interval_s: float = SAMPLE_INTERVAL_S
) -> tuple[FloatArray, FloatArray]:
    """The moment rate function's amplitude spectrum.

    Parameters
    ----------
    rate_nm_s : FloatArray
        The moment rate, newton-metres per second.
    sample_interval_s : float, optional

    Returns
    -------
    tuple of FloatArray
        Frequency in hertz and ``|M-dot(f)|`` in newton-metres. The zero-frequency
        value is the total moment, which is the check that the transform is scaled
        right.
    """
    spectrum = np.abs(np.fft.rfft(rate_nm_s)) * sample_interval_s
    return np.fft.rfftfreq(len(rate_nm_s), sample_interval_s), spectrum
