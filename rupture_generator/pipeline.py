"""The pipeline: the one place the stage order is written down, as :func:`generate`'s
own body.

Each stage has one of three shapes: a **map over the segments**, whose per-segment
closure never sees another segment, which is what licenses a substream each;
:func:`scale_moment`, a **fold** needing every segment's pattern, rigidity and area at
once; and :func:`solve_onsets`, the one **causal traversal**, parents before children.

Every random choice is made in :func:`draw_fields` and spent elsewhere, so the causal
traversal is a pure function of its inputs. Each calculation draws from its own
substream, keyed by its own name and its segment's, so reordering or re-batching the
stages cannot change any field's noise.

Nothing here writes a file: the result is each input chart with `slip_m`, `rake_deg`,
`rise_time_s`, `onset_s` and pulses attached.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from rupture_generator import moment, propagation, pulses, stages, timing
from rupture_generator.config.geometry import GeometryConfig
from rupture_generator.config.rupture import (
    ComputedPropagation,
    FieldConfig,
    HypocentreConfig,
    PerFaultSourceConfig,
    PointSourceConfig,
    PredeterminedPropagation,
    PropagationConfig,
    RampConfig,
    RuptureConfig,
    SourceConfig,
    VelocityModelConfig,
)
from rupture_generator.errors import ConfigError
from rupture_generator.mesh import RuptureMesh, build_surface, fuse, validate_chart
from rupture_generator.realisation import Realisation


def segments_of(geometry: GeometryConfig) -> Realisation:
    """S1-S3 over the whole geometry: every validated segment, named by `named`."""
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
    share a seam becomes ``surface:0``, ``surface:1``.
    """
    if len(charts) == 1:
        return {surface: charts[0]}
    return {f"{surface}:{index}": chart for index, chart in enumerate(charts)}


def charts_for(geometry: GeometryConfig, surface_name: str | None) -> Realisation:
    """The validated segments of one surface.

    Raises
    ------
    ConfigError
        If the geometry holds several surfaces and none was named.
    """
    names = [surface.name for surface in geometry.surfaces]
    if surface_name is None:
        if len(names) != 1:
            raise ConfigError(
                f"the geometry holds {len(names)} surfaces ({', '.join(names)}), so "
                "one has to be named"
            )
        surface_name = names[0]
    elif surface_name not in names:
        raise ConfigError(
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
) -> propagation.Tree[str | None]:
    """Which segment triggers which: sampled from fault separations, or as stated.

    In the predetermined form the stated root has to be the segment the hypocentre is
    on.

    Raises
    ------
    ConfigError
        If the stated tree is not one, or the faults are too far apart to form a
        connected system.
    """
    names = list(segments)
    if len(names) == 1:
        return {names[0]: None}

    if isinstance(config, PredeterminedPropagation):
        tree: propagation.Tree[str | None] = {
            name: config.parents.get(name) for name in names
        }
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
) -> Realisation:
    """Run the pipeline over a fault system.

    The geometry is **annotated in place**: the segments, the tree and the jumps are
    written onto the realisation passed in, which is also what is returned.

    Raises
    ------
    ConfigError
        For a hypocentre off the fault, a propagation that is not a tree, an
        unrepresentable rise time (naming the subfault), or a rupture speed the front
        cannot travel at.
    """

    source = config.source
    source.check_segments(list(geometry))

    realisation = propagate(
        geometry,
        config.propagation,
        config.hypocentre,
        config.random.stream("propagation"),
    )
    realisation = attach_materials(realisation, config.velocity_model)

    if isinstance(source, PointSourceConfig):
        realisation = constant_fields(realisation, source, config.field)
    else:
        realisation = draw_fields(realisation, config)

    realisation = scale_moment(realisation, source)
    realisation = solve_onsets(realisation, config)
    return synthesise_pulses(realisation, config)


def propagate(
    realisation: Realisation,
    propagation_config: PropagationConfig,
    hypocentre: HypocentreConfig,
    rng: np.random.Generator,
) -> Realisation:
    """Decide which segment triggers which, before any field is drawn.

    Settled first because deciding it from drawn fields would make the propagation a
    function of the noise rather than of the geometry. The **jumps** are not found
    here; `propagation.causal_jump` needs the parent's solved onsets.

    Raises
    ------
    ConfigError
        If the hypocentre names no segment when several rupture, or one that is not
        here; if a stated tree is not a tree; or if the faults are too far apart to
        form a connected system.
    """
    root = _root_of(realisation, hypocentre)
    realisation.tree = causality_tree(dict(realisation), propagation_config, root, rng)
    return realisation


def _root_of(realisation: Realisation, hypocentre: HypocentreConfig) -> str:
    """Which segment the rupture starts on, named or inferred.

    Raises
    ------
    ConfigError
        If several segments rupture and the hypocentre names none of them, or names
        one that is not here.
    """
    names = list(realisation)
    if hypocentre.fault is None:
        if len(names) != 1:
            raise ConfigError(
                f"this rupture spans {len(names)} segments ({', '.join(names)}), so "
                "the hypocentre has to say which one it is on"
            )
        return names[0]
    if hypocentre.fault not in realisation:
        raise ConfigError(
            f"the hypocentre is on {hypocentre.fault!r}, which is not one of "
            f"{', '.join(names)}"
        )
    return hypocentre.fault


def attach_materials(
    realisation: Realisation, velocity_model: VelocityModelConfig
) -> Realisation:
    """The rock each subfault is in: shear speed, rigidity and density.

    Sampled per subfault from the 1-D model at its own centre depth. Every later stage
    reads these off the chart rather than the model, so the moment fold, the front
    and the SRF all describe the same rock. ``density_g_cm3`` is there only because the
    SRF's points state it.
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

    for name, mesh in list(realisation.items()):
        realisation[name] = attach(mesh)
    return realisation


def draw_fields(realisation: Realisation, config: RuptureConfig) -> Realisation:
    """The four drawn fields: slip pattern, rise time, rake, onset displacement.

    The only place anything is drawn. Rise time and the onset displacement both
    correlate against slip's own Gaussian, so the Gaussian and the sampler reference are
    locals that never leave this function -- nothing downstream needs the latent, only
    the fields drawn from it.
    """
    source = config.source
    random = config.random

    def draw(name: str, mesh: RuptureMesh) -> RuptureMesh:
        covariance = source.covariance_of(name)
        slip_params = stages.SlipParams(
            covariance=covariance,
            coefficient_of_variation=config.slip.coefficient_of_variation,
            side_taper=config.slip.side_taper,
            top_taper=config.slip.top_taper,
            bottom_taper=config.slip.bottom_taper,
        )
        pattern, slip_latent = stages.slip_pattern(
            mesh, slip_params, random.stream("slip", name)
        )

        average_s = stages.average_rise_time_s(
            moment.seismic_moment_nm(source.magnitude_of(name)),
            source.rise_time_coefficient,
            timing.alpha_t(source.dip_of(name, mesh), source.rake_of(name)),
        )
        rise_time_s = stages.rise_time_field(
            mesh,
            slip_latent,
            slip_params.marginal,
            rise_time_model(config, average_s),
            random.stream("rise_time", name),
            covariance,
            sample_interval_s=config.timing.sample_interval_s,
        )

        # Takes neither the Gaussian nor the reference, which is the statement that
        # rake is independent of slip.
        rake_deg = stages.rake_field(
            mesh,
            stages.RakeParams(
                covariance=covariance,
                base_rake_deg=source.base_rake_deg_of(name, config.field.base_rake_deg),
                sigma_deg=config.slip.rake_sigma_deg,
            ),
            random.stream("rake", name),
        )

        onset_displacement_s = stages.onset_perturbation(
            mesh,
            slip_latent,
            slip_params.marginal,
            onset_model(config, name),
            random.stream("onset", name),
            covariance,
        )

        return mesh.with_fields(
            slip_pattern=pattern,
            rise_time_s=rise_time_s,
            rake_deg=rake_deg,
            onset_displacement_s=onset_displacement_s,
        )

    for name, mesh in list(realisation.items()):
        realisation[name] = draw(name, mesh)
    return realisation


def constant_fields(
    realisation: Realisation, source: PointSourceConfig, field: FieldConfig
) -> Realisation:
    """The same fields for a point source, given rather than drawn.

    Provides exactly the names :func:`draw_fields` does, so no later stage asks which
    kind of source it has. The rise time is **given**, and the geometric correction
    deliberately *not* applied -- a rise time the caller chose has already accounted
    for the geometry -- while the same source's rupture *speed* does get it, in
    :func:`solve_onsets`. The onset displacement is zeros rather than missing, so a
    point source takes the same code path and its front stays exactly radial.
    """

    def constant(mesh: RuptureMesh) -> RuptureMesh:
        return mesh.with_fields(
            slip_pattern=np.ones(mesh.cell_counts),
            rise_time_s=np.full(mesh.cell_counts, source.rise_time_s),
            rake_deg=np.full(mesh.cell_counts, field.base_rake_deg),
            onset_displacement_s=np.zeros(mesh.cell_counts),
        )

    for name, mesh in list(realisation.items()):
        realisation[name] = constant(mesh)
    return realisation


def scale_moment(realisation: Realisation, source: SourceConfig) -> Realisation:
    """Size the slip pattern into slip, in metres. The one global fold.

    Either **one factor over every segment** -- so how the moment divides between
    faults is the fields' own -- or a target per fault, when the source states the
    division. The four lists are built from one ``names``, in one order, and paired
    back by position: get that wrong and the event total is still exactly right and
    only the per-fault moments are swapped.
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

    for name, slip in zip(names, scaled, strict=True):
        realisation[name] = (
            realisation[name].with_fields(slip_m=slip).without("slip_pattern")
        )
    return realisation


def solve_onsets(realisation: Realisation, config: RuptureConfig) -> Realisation:
    """S7 and S8: the wavefront, in causal order, and where the front crossed.

    **Draws nothing**: it spends the displacement already on the chart, so the one
    order-dependent traversal is a pure function of its inputs. The root is seeded at
    the hypocentre; every other segment where and when its parent's front crossed onto
    it.

    Two steps per segment, and they compose. :func:`~rupture_generator.timing.travel_times`
    solves the coherent front over a smooth speed field, and
    :func:`~rupture_generator.stages.taper_onset` blends the displacement in from zero
    at the seed. Nothing needs pinning or shifting afterwards: the blend is zero at the
    seed, so the seed's onset is exactly the time it was seeded at, and the per-cell
    causal clamp keeps every other subfault strictly after it.

    There is consequently **one** timing field, ``onset_s``. The generator used to
    carry a second, ``wavefront_s``, holding the front before the displacement was
    added to it -- `propagation.causal_jump` chose departures on that one to keep an
    argmin off the displacement's negative tail. The clamp is what removes that tail,
    so the two fields would now agree on which cell is earliest, and there is only the
    one.
    """
    jump_delay = jump_model(config)
    max_jump_km = (
        config.propagation.max_jump_km
        if isinstance(config.propagation, ComputedPropagation)
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
            seed_cell, seed_time_s = hypocentre, 0.0
            delay_s = config.timing.rupture_delay_s
        else:
            jump = propagation.causal_jump(
                solved[parent],
                solved[parent]["onset_s"],
                mesh,
                jump_delay,
                max_distance_km=max_jump_km,
            )
            jumps[name] = jump
            seeds = [(*jump.child_cell, jump.arrival_s)]
            # Triggered from elsewhere: the seed carries the absolute arrival time, so
            # these onsets do not restart from zero on this fault. The taper is measured
            # from that arrival, so a jump point is protected the same way a hypocentre
            # is -- the segment still starts rupturing where the front reached it.
            seed_cell, seed_time_s = jump.child_cell, jump.arrival_s
            delay_s = 0.0

        travel_time_s = stages.taper_onset(
            timing.travel_times(
                mesh,
                mesh["shear_speed_kms"],
                speed_model(config, name, mesh),
                seeds,
            ),
            mesh["onset_displacement_s"],
            onset_model(config, name),
            seed_cell=seed_cell,
            seed_time_s=seed_time_s,
        )
        solved[name] = mesh.with_fields(onset_s=travel_time_s + delay_s)

    # The hypocentre, in the root's own arc lengths. Clamped to the fault's extent: a
    # hypocentre at the far edge is on the last cell.
    root_mesh = solved[root]
    solved[root] = root_mesh.with_attrs(
        hypocentre_strike_km=min(
            config.hypocentre.strike_km, float(root_mesh.strike_arc_km()[-1])
        ),
        hypocentre_dip_km=min(
            config.hypocentre.dip_km, float(root_mesh.dip_arc_km()[-1])
        ),
    )

    realisation.update(solved)
    realisation.jumps = jumps
    return realisation


def synthesise_pulses(realisation: Realisation, config: RuptureConfig) -> Realisation:
    """S9: a slip-rate pulse per subfault, carrying that subfault's slip.

    The only stage whose output is not a cell field: each pulse has its own length, so
    the charts carry them as CSR.
    """
    params = pulse_model(config)

    def synthesise(mesh: RuptureMesh) -> RuptureMesh:
        offsets, samples = pulses.synthesise(
            mesh.centres()[..., 2],
            mesh["slip_m"],
            mesh["rise_time_s"],
            params,
        )
        return mesh.with_pulses(offsets, samples).with_attrs(
            sample_interval_s=config.timing.sample_interval_s
        )

    for name, mesh in list(realisation.items()):
        realisation[name] = synthesise(mesh)
    return realisation


# --- The config boundary: what the file says, as the parameter objects a stage takes.


def onset_model(config: RuptureConfig, name: str) -> stages.OnsetParams:
    """How far this segment's onsets are displaced, and over how long they blend in.

    Per segment, because the spread follows the moment and a per-fault source gives
    each fault a magnitude of its own -- which is how the production workflow runs
    genslip, once per fault with that fault's ``mag=``.

    Always a parameter object: an offset and coefficient of zero is a coherent front,
    which draws nothing and leaves the solved times alone, and it takes the same path
    as any other.
    """
    timing = config.timing
    return stages.OnsetParams(
        scale_s=stages.onset_scale_s(
            moment.seismic_moment_nm(config.source.magnitude_of(name)),
            timing.rupture_time_offset_s,
            timing.rupture_time_coefficient,
        ),
        correlation=timing.rupture_time_correlation,
        blend_sigma=timing.rupture_time_blend_sigma,
    )


def speed_model(config: RuptureConfig, name: str, mesh: Any) -> timing.SpeedParams:
    """How fast the front travels on one segment."""
    source = config.source
    return timing.SpeedParams(
        velocity_fraction=config.field.velocity_fraction,
        average_dip_deg=source.dip_of(name, mesh),
        average_rake_deg=source.rake_of(name),
        shallow=depth_ramp(
            config.timing.shallow_speed_ramp or config.timing.shallow_ramp
        ),
        deep=depth_ramp(config.timing.deep_speed_ramp or config.timing.deep_ramp),
        shallow_factor=config.timing.shallow_speed_factor,
        deep_factor=config.timing.deep_speed_factor,
        minimum_fraction=config.timing.rupture_velocity_min_fraction,
        maximum_fraction=config.timing.rupture_velocity_max_fraction,
    )


def jump_model(config: RuptureConfig) -> propagation.JumpDelay:
    """How long the front takes to cross from one segment to the next.

    One delay serves every edge of the tree.
    """
    return propagation.DistanceOverVelocity(
        np.asarray(config.velocity_model.bottom_depth_km),
        np.asarray(config.velocity_model.shear_speed_km_s),
    )


def depth_ramp(config_ramp: RampConfig) -> stages.DepthRamp:
    """One config ramp -- a centre and a half width, in km -- as a `DepthRamp`."""
    return stages.DepthRamp(config_ramp.centre_km, config_ramp.half_width_km)


def pulse_model(config: RuptureConfig) -> pulses.PulseParams:
    """How every subfault's slip-rate pulse is shaped and sampled.

    Raises
    ------
    ConfigError
        For a slip-rate shape this package does not implement, named.
    """
    return pulses.PulseParams(
        shape=pulses.from_stype(config.timing.slip_rate_shape or "OliuP2"),
        shallow_ramp=depth_ramp(config.timing.beta_shallow_ramp),
        mid_ramp=depth_ramp(config.timing.beta_mid_ramp),
        beta_shallow=config.timing.beta_shallow,
        beta_mid=config.timing.beta_mid,
        beta_deep=config.timing.beta_deep,
        sample_interval_s=config.timing.sample_interval_s,
    )


def rise_time_model(config: RuptureConfig, average_s: float) -> stages.RiseTimeParams:
    """How long each subfault slips for, and how that follows slip and depth."""
    timing_config = config.timing
    return stages.RiseTimeParams(
        average_s=average_s,
        correlation=timing_config.rise_time_correlation,
        sigma=timing_config.rise_time_sigma,
        slip_exponent=timing_config.slip_exponent,
        shallow_blend=depth_ramp(timing_config.rise_time_blend),
        shallow_stretch=depth_ramp(timing_config.shallow_ramp),
        deep_stretch=depth_ramp(timing_config.deep_ramp),
        shallow_factor=timing_config.shallow_rise_factor,
        deep_factor=timing_config.deep_rise_factor,
    )


__all__ = [
    "Realisation",
    "attach_materials",
    "causality_tree",
    "charts_for",
    "constant_fields",
    "depth_ramp",
    "draw_fields",
    "generate",
    "jump_model",
    "named",
    "onset_model",
    "propagate",
    "pulse_model",
    "rise_time_model",
    "scale_moment",
    "segments_of",
    "solve_onsets",
    "speed_model",
    "synthesise_pulses",
]
