"""The pipeline: the one place the stage order is written down.

It is written down as :func:`generate`'s own body. A realisation goes in, a realisation
comes out, and each line between is a function of the same shape -- so the stage order
is code that runs rather than a table that has to be kept true.

.. code-block:: python

    realisation = propagate(geometry, ...)        # which fault triggers which
    realisation = attach_materials(realisation, ...)   # the rock each subfault is in
    realisation = draw_fields(realisation, ...)   # slip, rise time, rake, perturbation
    realisation = scale_moment(realisation, ...)  # the pattern becomes slip, in metres
    realisation = solve_onsets(realisation, ...)  # the wavefront, in causal order
    return synthesise_pulses(realisation, ...)    # a slip-rate pulse per subfault

Unlike the port's `generate`, that order is a **convention rather than a contract**.
Each calculation draws from its own substream, keyed by its own name and its segment's,
so reordering them, re-batching them, or changing one's parameters cannot change
another's noise. The port could not do that: its stages shared one stream in a fixed
order, and two dead fields were drawn and discarded on every run purely to keep that
order intact.

# Three shapes, and only three

Every function here has one of three shapes, and which one it is says what it may do.
Most are a **map over the segments** -- a dict comprehension whose per-segment closure
never sees another segment, which is what licenses a substream each. `scale_moment` is
a **fold**: the shared factor needs every segment's pattern, rigidity and area at once,
and it does not pretend otherwise. `solve_onsets` is the one **causal traversal**,
parents before children, because a child is seeded where its parent's front crossed
onto it.

# Drawing and timing are separate

Every random choice is made in `draw_fields` -- including the onset perturbation, which
is drawn there because it correlates against slip's own Gaussian, and *spent* in
`solve_onsets`. So the sampler reference never leaves the function that made it, and
the causal traversal is a pure function of its inputs. A point source takes the same
path with constant fields and a perturbation of zeros, which is why no stage below ever
asks what kind of source it has.

# Nothing here writes a file

The result is each input chart with `slip_m`, `rake_deg`, `rise_time_s`, `onset_s` and
pulses attached. Which of those a rupture file stores, and what the groups are called,
is `formats.rupture`'s to say -- so the stage order can be read without reading the
file layout, which is what this module used to make impossible.
"""

from __future__ import annotations

import itertools

import numpy as np

from rupture_generator import moment, propagation, pulses, stages, timing
from rupture_generator.config.geometry import (
    ComputedPropagation,
    GeometryConfig,
    PredeterminedPropagation,
    PropagationConfig,
)
from rupture_generator.config.rupture import (
    FieldConfig,
    HypocentreConfig,
    PerFaultSourceConfig,
    PointSourceConfig,
    RampConfig,
    RuptureConfig,
    SourceConfig,
    VelocityModelConfig,
)
from rupture_generator.mesh import RuptureMesh, build_surface, fuse, validate_chart
from rupture_generator.random import Streams
from rupture_generator.realisation import Realisation
from rupture_generator.sampling import FieldSampler, SpectralSampler

CALCULATIONS = ("propagation", "slip", "rise_time", "rake", "onset")
"""Every calculation that draws, by name.

Documentation rather than machinery -- `random.Streams` hashes the name, so this list
is not an index into anything and adding to it changes nothing. That is the point:
a calculation's noise is a function of the seed, the realisation, its own name and its
segment's name, and of nothing else.
"""


def _streams(config: RuptureConfig) -> Streams:
    """The event's randomness, split by name."""
    return Streams(config.random.seed, config.random.realisation)


def segments_of(geometry: GeometryConfig) -> Realisation:
    """S1-S3 over the whole geometry: every validated segment, named.

    A surface that fuses to one segment keeps its own name. One whose planes do not
    all share a seam becomes several, named ``surface:0``, ``surface:1`` -- because
    those parts are what actually rupture, and the causality tree is over the things
    that rupture. A predetermined propagation can name either form; the bare name is
    what a single-segment surface answers to.
    """
    segments: dict[str, RuptureMesh] = {}
    for surface in geometry.surfaces:
        charts = fuse(build_surface(surface, geometry.crs))
        for chart in charts:
            validate_chart(chart)
        segments |= named(surface.name, charts)
    return Realisation(segments, geometry.crs)


def named(surface: str, charts: list[RuptureMesh]) -> dict[str, RuptureMesh]:
    """One surface's charts, under the names the causality tree uses.

    A surface that fuses to one segment keeps its own name; one whose planes do not all
    share a seam becomes ``surface:0``, ``surface:1``. One function because both
    `segments_of` here and `generate_cli.named_segments`, which starts from a mesh file
    rather than a geometry, have to agree about it -- a rupture whose config names
    ``kaikoura`` and whose mesh yields ``kaikoura:0`` is a rupture nobody can select.
    """
    if len(charts) == 1:
        return {surface: charts[0]}
    return {f"{surface}:{index}": chart for index, chart in enumerate(charts)}


def charts_for(geometry: GeometryConfig, surface_name: str | None) -> Realisation:
    """The validated segments of one surface.

    Kept for callers that want a single surface rather than the whole system.

    Raises
    ------
    ValueError
        If the geometry holds several surfaces and none was named. Picking the first
        would run silently on a fault nobody chose, and the output would look exactly
        like the one that was wanted.
    """
    names = [surface.name for surface in geometry.surfaces]
    if surface_name is None:
        if len(names) != 1:
            raise ValueError(
                f"the geometry holds {len(names)} surfaces ({', '.join(names)}), so "
                "one has to be named"
            )
        surface_name = names[0]
    elif surface_name not in names:
        raise ValueError(
            f"no surface is called {surface_name!r}; the geometry holds "
            f"{', '.join(names)}"
        )

    surface = next(item for item in geometry.surfaces if item.name == surface_name)
    charts = fuse(build_surface(surface, geometry.crs))
    for chart in charts:
        validate_chart(chart)
    return Realisation(named(surface_name, charts), geometry.crs)


def causality_tree(
    segments: dict[str, RuptureMesh],
    config: PropagationConfig,
    root: str,
    rng: np.random.Generator,
) -> propagation.Tree:
    """Which segment triggers which, either sampled or as stated.

    Sampled from fault separations in the computed form, or taken verbatim in the
    predetermined one -- where the stated root has to be the segment the hypocentre
    is on, checked rather than assumed.

    Raises
    ------
    ValueError
        If the stated tree is not one, or the faults are too far apart to form a
        connected system.
    """
    names = list(segments)
    if len(names) == 1:
        return {names[0]: None}

    if isinstance(config, PredeterminedPropagation):
        tree: propagation.Tree = {name: config.parents.get(name) for name in names}
        propagation.check_tree(tree, names, root)
        return tree

    assert isinstance(config, ComputedPropagation)
    distances_km = {
        (first, second): propagation.closest_approach_km(
            segments[first], segments[second]
        )
        for first, second in itertools.combinations(names, r=2)
    }
    graph = propagation.jump_graph(
        distances_km,
        names,
        d0_km=config.d0_km,
        delta_km=config.delta_km,
        max_jump_km=config.max_jump_km,
    )
    edges = (
        propagation.sample_tree(graph, rng)
        if config.strategy == "sampled"
        else propagation.maximum_likelihood_tree(graph)
    )
    return propagation.root_tree(graph.faults, edges, root)


def generate(
    config: RuptureConfig,
    geometry: Realisation,
    *,
    propagation_config: PropagationConfig | None = None,
) -> Realisation:
    """Run the pipeline over a fault system.

    A realisation in, a realisation out. The geometry is a system nothing has been
    drawn on; the result is the same segments with the rupture attached to them, which
    is why the two are one type.

    Nothing here writes a file. The rupture is a `Realisation`, and what a rupture file
    stores is `formats.rupture`'s to say -- so the stage order can be read without
    reading the file layout.

    Parameters
    ----------
    config : RuptureConfig
        What the earthquake is.
    geometry : Realisation
        The validated charts and the frame they are in, from `segments_of` or
        `charts_for`.
    propagation_config : PropagationConfig, optional
        How the rupture crosses between them. Defaults to the computed form. A keyword
        rather than part of ``config`` because it describes the *geometry*, and arrives
        beside the meshes from whichever file they were read from.

    Returns
    -------
    Realisation

    Raises
    ------
    ValueError
        For a hypocentre off the fault, a propagation that is not a tree, an
        unrepresentable rise time (naming the subfault), or a rupture speed the front
        cannot travel at.
    """

    source = config.source
    source.check_segments(list(geometry))
    propagation_config = propagation_config or ComputedPropagation()

    realisation = propagate(geometry, propagation_config, config.hypocentre, config)
    realisation = attach_materials(realisation, config.velocity_model)

    if isinstance(source, PointSourceConfig):
        realisation = constant_fields(realisation, source, config.field)
    else:
        realisation = draw_fields(realisation, config)

    realisation = scale_moment(realisation, source)
    realisation = solve_onsets(realisation, config, propagation_config)
    return synthesise_pulses(realisation, config)


def propagate(
    realisation: Realisation,
    propagation_config: PropagationConfig,
    hypocentre: HypocentreConfig,
    config: RuptureConfig,
) -> Realisation:
    """Decide which segment triggers which, before any field is drawn.

    The tree is settled first because everything causal downstream reads it, and
    because deciding it from drawn fields would make the propagation a function of the
    noise rather than of the geometry.

    The **jumps** are not found here. `propagation.causal_jump` needs the parent's
    solved wavefront, which does not exist until the travel times are solved, so the
    tree is fixed here and the crossings are found in :func:`solve_onsets`. That split
    is `propagation.py`'s own division of labour.

    Raises
    ------
    ValueError
        If the hypocentre does not say which segment it is on when several rupture, if
        it names one that is not here, if a stated tree is not a tree, or if the faults
        are too far apart to form a connected system.
    """
    root = _root_of(realisation, hypocentre)
    return realisation.with_tree(
        causality_tree(
            dict(realisation),
            propagation_config,
            root,
            _streams(config).stream("propagation"),
        )
    )


def _root_of(realisation: Realisation, hypocentre: HypocentreConfig) -> str:
    """Which segment the rupture starts on, named or inferred.

    Inferred only when there is nothing to infer: with one segment the hypocentre can
    only be on it. With several, picking one would run a rupture on a fault nobody
    chose, and the output would look exactly like the one that was wanted.

    Raises
    ------
    ValueError
    """
    names = list(realisation)
    if hypocentre.fault is None:
        if len(names) != 1:
            raise ValueError(
                f"this rupture spans {len(names)} segments ({', '.join(names)}), so "
                "the hypocentre has to say which one it is on"
            )
        return names[0]
    if hypocentre.fault not in realisation:
        raise ValueError(
            f"the hypocentre is on {hypocentre.fault!r}, which is not one of "
            f"{', '.join(names)}"
        )
    return hypocentre.fault


def attach_materials(
    realisation: Realisation, velocity_model: VelocityModelConfig
) -> Realisation:
    """The rock each subfault is in: shear speed, rigidity and density.

    Sampled per subfault from the 1-D model at its own centre depth. Every later stage
    reads these off the chart rather than the model, so the moment fold, the wavefront
    and the SRF all describe the same rock -- the SRF used to resample the model from
    each subfault's written depth, which is the same number arrived at twice.

    ``density_g_cm3`` is a working field: the rupture file does not store it, and only
    the SRF, whose points state it, ever asks.
    """
    bottoms = np.asarray(velocity_model.bottom_depth_km)
    speeds = np.asarray(velocity_model.shear_speed_km_s)
    layer_densities = np.asarray(velocity_model.density_g_cm3)

    def attach(mesh: RuptureMesh) -> RuptureMesh:
        depth_km = mesh.centres()[..., 2]
        shear_speed, rigidity = moment.sample_velocity_model(
            depth_km, bottoms, speeds, layer_densities
        )
        return mesh.with_fields(
            shear_speed_kms=shear_speed,
            rigidity_pa=rigidity,
            density_g_cm3=layer_densities[moment.layer_of(depth_km, bottoms)],
        )

    return realisation.replace_segments(
        {name: attach(mesh) for name, mesh in realisation.items()}
    )


def draw_fields(realisation: Realisation, config: RuptureConfig) -> Realisation:
    """S4 to S6, and the onset perturbation: every field the rupture draws.

    One batch, one visit per segment, one covariance spec. They are together because
    three of the four are drawn *against slip's own Gaussian* -- and that Gaussian and
    the sampler reference it came with are local to this function, so nothing else in
    the pipeline has to carry them. The reference is a padded-grid spectrum, not
    representable as a cell field and not recoverable from one; drawing everything that
    needs it here is what keeps it from becoming a side channel.

    What leaves is four fields on each chart. The onset perturbation is *drawn* here
    and *spent* in :func:`solve_onsets`, which is what lets the causal traversal be
    deterministic: every random choice this rupture makes has been made by the time
    this returns.

    ``slip_pattern`` is deliberately not ``slip_m``. Its size is the moment fold's to
    set, and calling it something else until then is what stops a stage reading an
    unscaled field as if it were slip.
    """
    source = config.source
    streams = _streams(config)
    sampler: FieldSampler = SpectralSampler()

    def draw(name: str, mesh: RuptureMesh) -> RuptureMesh:
        covariance = source.covariance_of(name)
        slip_params = stages.SlipParams(
            covariance=covariance,
            coefficient_of_variation=config.slip.coefficient_of_variation,
            side_taper=config.slip.side_taper,
            top_taper=config.slip.top_taper,
            bottom_taper=config.slip.bottom_taper,
        )
        pattern, gaussian, reference = stages.slip_pattern(
            mesh, slip_params, streams.stream("slip", name), sampler
        )

        average_s = stages.average_rise_time_s(
            moment.seismic_moment_nm(source.magnitude_of(name)),
            source.rise_time_coefficient,
            timing.alpha_t(source.dip_of(name, mesh), source.rake_of(name)),
        )
        rise_time_s = stages.rise_time_field(
            mesh,
            gaussian,
            reference,
            _rise_time_params(config, average_s),
            streams.stream("rise_time", name),
            sampler,
            covariance,
            sample_interval_s=config.timing.sample_interval_s,
        )

        # Takes neither the Gaussian nor the reference, which is the statement that
        # rake is independent of slip: a patch that slips more has no reason to slip
        # in a different direction.
        rake_deg = stages.rake_field(
            mesh,
            stages.RakeParams(
                covariance=covariance,
                base_rake_deg=source.base_rake_deg_of(name, config.field.base_rake_deg),
                sigma_deg=config.slip.rake_sigma_deg,
            ),
            streams.stream("rake", name),
            sampler,
        )

        perturbation = stages.onset_perturbation(
            mesh,
            reference,
            _onset_params(config),
            streams.stream("onset", name),
            sampler,
            covariance,
        )

        return mesh.with_fields(
            slip_pattern=pattern,
            rise_time_s=rise_time_s,
            rake_deg=rake_deg,
            onset_perturbation=perturbation,
        ).with_attrs(
            truncated_fraction=stages.truncated_fraction(gaussian, slip_params)
        )

    return realisation.replace_segments(
        {name: draw(name, mesh) for name, mesh in realisation.items()}
    )


def constant_fields(
    realisation: Realisation, source: PointSourceConfig, field: FieldConfig
) -> Realisation:
    """The same four fields for a point source, given rather than drawn.

    A point source is this pipeline with constant fields, not a path of its own: it
    provides exactly the names :func:`draw_fields` provides, so everything downstream
    is unchanged and no later stage asks which kind of source it has.

    The rise time is **given**, and the geometric correction is deliberately *not*
    applied -- a rise time the caller chose has already accounted for the geometry --
    while the same source's rupture *speed* does get it, in :func:`solve_onsets`. The
    perturbation is a field of zeros rather than a missing one, which is what lets
    `stages.apply_perturbation` run unconditionally.
    """

    def constant(mesh: RuptureMesh) -> RuptureMesh:
        return mesh.with_fields(
            slip_pattern=np.ones(mesh.cell_counts),
            rise_time_s=np.full(mesh.cell_counts, source.rise_time_s),
            rake_deg=np.full(mesh.cell_counts, field.base_rake_deg),
            onset_perturbation=np.zeros(mesh.cell_counts),
        ).with_attrs(truncated_fraction=0.0)

    return realisation.replace_segments(
        {name: constant(mesh) for name, mesh in realisation.items()}
    )


def scale_moment(realisation: Realisation, source: SourceConfig) -> Realisation:
    """Size the slip pattern into slip, in metres. The one global fold.

    Either **one factor over every segment** -- so how the moment divides between
    faults is the fields' own -- or a target per fault, when the source states the
    division. The two are different models and the config says which; a hazard model
    that derived each fault's magnitude from its area has already decided the
    partition, and re-deriving it would discard that.

    Not a per-segment map, and not pretending to be: the shared factor needs every
    segment's pattern, rigidity and area at once. The four lists are built from one
    ``names``, in one order, and paired back by position -- get that wrong and each
    fault carries a plausible slip, the event total is still exactly right, and only
    the per-fault moments are swapped.
    """
    names = list(realisation)
    patterns = [realisation[name]["slip_pattern"] for name in names]
    rigidities = [realisation[name]["rigidity_pa"] for name in names]
    areas = [realisation[name].areas_km2() for name in names]

    if isinstance(source, PerFaultSourceConfig):
        scaled = moment.scale_each_to_moment(
            patterns,
            rigidities,
            areas,
            [moment.seismic_moment_nm(source.magnitude_of(name)) for name in names],
        )
    else:
        scaled = moment.scale_to_moment(
            patterns, rigidities, areas, moment.seismic_moment_nm(source.magnitude)
        )

    return realisation.replace_segments(
        {
            name: realisation[name].with_fields(slip_m=slip).without("slip_pattern")
            for name, slip in zip(names, scaled, strict=True)
        }
    ).with_moment(moment.seismic_moment_nm(source.magnitude))


def solve_onsets(
    realisation: Realisation,
    config: RuptureConfig,
    propagation_config: PropagationConfig,
) -> Realisation:
    """S7 and S8: the wavefront, in causal order, and where the front crossed.

    **Draws nothing.** Every random choice was made in :func:`draw_fields`; this solves
    the eikonal equation from the seeds the tree implies and spends the perturbation
    already on the chart. So the one traversal whose order matters is a pure function
    of its inputs, and re-running it cannot move a field.

    Parents first. The root is seeded at the hypocentre; every other segment is seeded
    where and when its parent's front crossed onto it, which is what makes a
    multi-segment rupture propagate rather than restart on each fault.
    """
    onset_params = _onset_params(config)
    jump_delay = _jump_delay(config)
    max_jump_km = (
        propagation_config.max_jump_km
        if isinstance(propagation_config, ComputedPropagation)
        else propagation.MAX_JUMP_KM
    )

    root = realisation.root
    hypocentre = realisation[root].cell_index(
        config.hypocentre.strike_km, config.hypocentre.dip_km
    )

    solved: dict[str, RuptureMesh] = {}
    jumps: dict[str, propagation.Jump] = {}

    for name in realisation.in_causal_order():
        mesh = realisation[name]
        parent = realisation.tree[name]

        if parent is None:
            seeds = [(*hypocentre, 0.0)]
            pinned: tuple[int, int] | None = hypocentre
            delay_s = config.timing.rupture_delay_s
        else:
            # Chosen on the parent's wavefront and timed on its onset. Choosing the
            # cell from the perturbed field would be an argmin over a hundred thousand
            # perturbed values -- an order statistic that finds the perturbation's
            # negative tail rather than the shape of the front -- while the time the
            # rupture actually reached the chosen cell is the onset's to report.
            jump = propagation.causal_jump(
                solved[parent],
                solved[parent]["wavefront_s"],
                mesh,
                jump_delay,
                parent_onset_s=solved[parent]["onset_s"],
                max_distance_km=max_jump_km,
            )
            jumps[name] = jump
            seeds = [(*jump.child_cell, jump.arrival_s)]
            # Triggered from elsewhere: no pin and no clamp, so this segment's onsets
            # stay absolute. That is what lets a rupture propagate between faults
            # rather than restarting on each.
            pinned = None
            delay_s = 0.0

        travel_time_s = timing.travel_times(
            mesh,
            mesh["shear_speed_kms"],
            _speed_params(config, name, mesh),
            seeds,
        )
        solved[name] = mesh.with_fields(
            # The solved wavefront, kept beside the perturbed onset because the two
            # answer different questions for a jump: see `causal_jump`.
            wavefront_s=travel_time_s + delay_s,
            onset_s=stages.apply_perturbation(
                travel_time_s,
                mesh["onset_perturbation"],
                onset_params,
                hypocentre=pinned,
                delay_s=delay_s,
            ),
        )

    # The hypocentre, in the root's own arc lengths and under the names a file uses,
    # so the writer copies it and no segment needs a special case. Clamped to the
    # fault's extent: a hypocentre at the far edge is on the last cell, and the arc
    # length naming it is the edge rather than something past it.
    root_mesh = solved[root]
    solved[root] = root_mesh.with_attrs(
        hypocentre_strike_km=min(
            config.hypocentre.strike_km, float(root_mesh.strike_arc_km()[-1])
        ),
        hypocentre_dip_km=min(
            config.hypocentre.dip_km, float(root_mesh.dip_arc_km()[-1])
        ),
    )

    return realisation.replace_segments(solved).with_jumps(jumps)


def synthesise_pulses(realisation: Realisation, config: RuptureConfig) -> Realisation:
    """S9: a slip-rate pulse per subfault, carrying that subfault's slip.

    The last stage, and the only one whose output is not a cell field -- a pulse per
    subfault, each its own length, so the charts carry them as CSR.
    """
    pulse_params = pulses.PulseParams(
        shape=pulses.from_stype(config.timing.slip_rate_shape or "OliuP2"),
        shallow_ramp=_ramp(config.timing.beta_shallow_ramp),
        mid_ramp=_ramp(config.timing.beta_mid_ramp),
        beta_shallow=config.timing.beta_shallow,
        beta_mid=config.timing.beta_mid,
        beta_deep=config.timing.beta_deep,
        sample_interval_s=config.timing.sample_interval_s,
    )

    def synthesise(mesh: RuptureMesh) -> RuptureMesh:
        offsets, samples = pulses.synthesise(
            mesh, mesh["slip_m"], mesh["rise_time_s"], pulse_params
        )
        return mesh.with_pulses(offsets, samples).with_attrs(
            sample_interval_s=config.timing.sample_interval_s
        )

    return realisation.replace_segments(
        {name: synthesise(mesh) for name, mesh in realisation.items()}
    )


def _onset_params(config: RuptureConfig) -> stages.OnsetParams:
    """How far the onset is perturbed from the wavefront, and how it follows slip.

    One object read in two places -- the correlation by the draw, the scale and spread
    by the application -- so the two cannot disagree about which model they are.
    """
    return stages.OnsetParams(
        scale_s=config.timing.rupture_time_scale,
        correlation=config.timing.rupture_time_correlation,
        sigma=config.timing.rupture_time_sigma,
    )


def _speed_params(
    config: RuptureConfig, name: str, mesh: RuptureMesh
) -> timing.SpeedParams:
    """How fast the front travels on one segment."""
    source = config.source
    return timing.SpeedParams(
        velocity_fraction=config.field.velocity_fraction,
        average_dip_deg=source.dip_of(name, mesh),
        average_rake_deg=source.rake_of(name),
        shallow=_ramp(config.timing.shallow_speed_ramp or config.timing.shallow_ramp),
        deep=_ramp(config.timing.deep_speed_ramp or config.timing.deep_ramp),
        shallow_factor=config.timing.shallow_speed_factor,
        deep_factor=config.timing.deep_speed_factor,
    )


def _jump_delay(config: RuptureConfig) -> propagation.JumpDelay:
    """How long the front takes to cross from one segment to the next.

    The gap is by definition on neither fault, so neither segment's own *sampled*
    materials describe the rock in it -- the shared velocity model does, read at the
    depth the front leaves from. One delay serves every edge of the tree, because the
    velocity model is one model and nothing here depends on which pair is crossing.
    """
    return propagation.DistanceOverVelocity(
        np.asarray(config.velocity_model.bottom_depth_km),
        np.asarray(config.velocity_model.shear_speed_km_s),
    )


def _ramp(config_ramp: RampConfig) -> stages.DepthRamp:
    return stages.DepthRamp(config_ramp.centre_km, config_ramp.half_width_km)


def _rise_time_params(config: RuptureConfig, average_s: float) -> stages.RiseTimeParams:
    timing_config = config.timing
    return stages.RiseTimeParams(
        average_s=average_s,
        correlation=timing_config.rise_time_correlation,
        sigma=timing_config.rise_time_sigma,
        slip_exponent=timing_config.slip_exponent,
        shallow_blend=_ramp(timing_config.rise_time_blend),
        shallow_stretch=_ramp(timing_config.shallow_ramp),
        deep_stretch=_ramp(timing_config.deep_ramp),
        shallow_factor=timing_config.shallow_rise_factor,
        deep_factor=timing_config.deep_rise_factor,
    )


__all__ = [
    "CALCULATIONS",
    "Realisation",
    "attach_materials",
    "causality_tree",
    "charts_for",
    "constant_fields",
    "draw_fields",
    "generate",
    "named",
    "propagate",
    "scale_moment",
    "segments_of",
    "solve_onsets",
    "synthesise_pulses",
]
