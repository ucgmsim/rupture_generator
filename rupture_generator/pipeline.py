"""The pipeline: the one place the stage order is written down.

.. code-block:: text

    S1  geometry            -> coarse mesh          mesh.build_surface
    S2  coarse mesh         -> chart                (same call; subdivision)
    S3  chart               -> validated chart      mesh.validate_chart
    S4  chart, Mw, rng      -> slip                 stages.slip_pattern + moment
    S5  chart, slip, rng    -> rise time            stages.rise_time_field
    S6  chart, rng          -> rake                 stages.rake_field
    S7  chart, seeds        -> travel time          timing.travel_times
    S8  travel time, rng    -> onset                stages.onset_times
    S9  slip, rise, onset   -> pulses               pulses.synthesise

Unlike the port's `generate`, this order is a **convention rather than a contract**.
Each stage draws from its own substream, named for the stage and the segment, so
reordering them or changing one stage's parameters cannot change another's noise. The
port could not do that: its stages shared one stream in a fixed order, and two dead
fields were drawn and discarded on every run purely to keep that order intact.

# Several faults, one earthquake

A rupture may cross between faults. `propagation.py` decides which fault triggers
which, before any field is drawn; this module then walks that tree in topological
order, and for each edge asks the parent's *solved wavefront* where and when the
front crossed. So S4 to S6 run per segment with independent substreams, S7 and S8 run
in causal order, and only the moment scaling is global -- one factor over every
segment, so the partition of moment between faults is the fields' own rather than an
artefact of scaling each alone.

The result is each input chart with `slip`, `rake`, `rise_time`, `onset` and pulses
attached -- which is exactly what the rupture file stores, so "the pipeline returns an
annotated mesh" and "the pipeline's output is the file" are one statement.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyproj
import xarray as xr

from rupture_generator import moment, propagation, pulses, stages, timing
from rupture_generator.config.geometry import (
    ComputedPropagation,
    GeometryConfig,
    PredeterminedPropagation,
    PropagationConfig,
)
from rupture_generator.config.rupture import (
    FiniteSourceConfig,
    PointSourceConfig,
    RampConfig,
    RuptureConfig,
)
from rupture_generator.formats import rupture as rupture_format
from rupture_generator.mesh import RuptureMesh, build_surface, fuse, validate_chart
from rupture_generator.sampling import (
    CovarianceSpec,
    FieldSampler,
    SpectralSampler,
    correlation_lengths,
)

STAGE_STREAMS = ("propagation", "slip", "rise_time", "rake", "onset")
"""The named substreams, one per stage that draws.

Spawned in this order from the event seed, and each spawned again per segment. The
*names* are what matter -- they make a stage's noise a function of the seed and its
own identity rather than of what ran before it.
"""


@dataclasses.dataclass(frozen=True)
class Realisation:
    """One generated rupture: the annotated charts, and how it propagated.

    Attributes
    ----------
    segments : dict of str to xr.Dataset
        One per segment, keyed by the name the causality tree uses.
    tree : propagation.Tree
        Which segment triggered which. The root is mapped to ``None``.
    jumps : dict of str to propagation.Jump
        Where and when the front crossed onto each triggered segment.
    moment_newton_m : float
        The moment the whole event carries -- the target, closed exactly across every
        segment together.
    hypocentre : tuple of int
        The ``(i, j)`` cell the rupture nucleated at, on the root segment.
    truncated_fraction : float
        What fraction of the fault the slip truncation clipped. A diagnostic: a large
        value says the requested variation was not really achievable.
    """

    segments: dict[str, xr.Dataset]
    tree: propagation.Tree
    jumps: dict[str, propagation.Jump]
    moment_newton_m: float
    hypocentre: tuple[int, int]
    truncated_fraction: float

    @property
    def root(self) -> str:
        """The segment the rupture started on."""
        return next(name for name, parent in self.tree.items() if parent is None)


def _stream(config: RuptureConfig, stage: str, segment: int) -> np.random.Generator:
    """The generator for one stage on one segment.

    Keyed by position rather than by name so the stream is a function of the seed,
    the realisation, the stage and the segment -- and of nothing else. Adding a stage
    to the end of :data:`STAGE_STREAMS` therefore leaves every existing stage's noise
    alone, which is the property that lets stages be added at all.
    """
    return np.random.default_rng(
        np.random.SeedSequence(
            entropy=config.random.seed,
            spawn_key=(config.random.realisation, STAGE_STREAMS.index(stage), segment),
        )
    )


def _covariance(config: RuptureConfig) -> CovarianceSpec:
    """The field structure this source implies."""
    source = config.source
    if isinstance(source, FiniteSourceConfig):
        return correlation_lengths(
            source.magnitude,
            strike_offset=source.strike_offset,
            dip_offset=source.dip_offset,
        )
    # A point source draws no fields, but the stages still want a spec; one cell has
    # no structure to describe, so any positive length does.
    return CovarianceSpec(1.0, 1.0)


def segments_of(geometry: GeometryConfig) -> dict[str, RuptureMesh]:
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
        if len(charts) == 1:
            segments[surface.name] = charts[0]
        else:
            for index, chart in enumerate(charts):
                segments[f"{surface.name}:{index}"] = chart
    return segments


def charts_for(geometry: GeometryConfig, surface_name: str | None) -> list[RuptureMesh]:
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
    segments = fuse(build_surface(surface, geometry.crs))
    for segment in segments:
        validate_chart(segment)
    return segments


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
        for index, first in enumerate(names)
        for second in names[index + 1 :]
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
    segments: dict[str, RuptureMesh] | list[RuptureMesh],
    crs: pyproj.CRS,
    *,
    propagation_config: PropagationConfig | None = None,
    sampler: FieldSampler | None = None,
) -> Realisation:
    """Run the pipeline over a fault system.

    Parameters
    ----------
    config : RuptureConfig
        What the earthquake is.
    segments : dict of str to RuptureMesh, or list of RuptureMesh
        The validated charts. A list is taken as the segments of one unnamed surface,
        which is the single-fault case.
    crs : pyproj.CRS
        The frame they are in, for the one projection seam.
    propagation_config : PropagationConfig, optional
        How the rupture crosses between them. Defaults to the computed form.
    sampler : FieldSampler, optional
        Defaults to the spectral sampler. The seam a Matern sampler enters through.

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
    sampler = sampler or SpectralSampler()
    covariance = _covariance(config)
    source = config.source
    propagation_config = propagation_config or ComputedPropagation()

    if not isinstance(segments, dict):
        segments = (
            {"fault": segments[0]}
            if len(segments) == 1
            else {f"fault:{index}": chart for index, chart in enumerate(segments)}
        )

    names = list(segments)
    root = config.hypocentre.fault
    if root is None:
        if len(names) != 1:
            raise ValueError(
                f"this rupture spans {len(names)} segments ({', '.join(names)}), so "
                "the hypocentre has to say which one it is on"
            )
        root = names[0]
    elif root not in segments:
        raise ValueError(
            f"the hypocentre is on {root!r}, which is not one of {', '.join(names)}"
        )

    tree = causality_tree(
        segments, propagation_config, root, _stream(config, "propagation", 0)
    )

    max_jump_km = (
        propagation_config.max_jump_km
        if isinstance(propagation_config, ComputedPropagation)
        else propagation.MAX_JUMP_KM
    )
    order = list(propagation.in_topological_order(tree))
    position = {name: index for index, name in enumerate(names)}

    target_moment = moment.seismic_moment_nm(source.magnitude)
    correction = timing.alpha_t(source.average_dip_deg, source.average_rake_deg)
    is_point = isinstance(source, PointSourceConfig)

    # ---- Materials ------------------------------------------------------------
    materials = {}
    for name, mesh in segments.items():
        materials[name] = moment.sample_velocity_model(
            mesh.centres()[..., 2],
            np.asarray(config.velocity_model.bottom_depth_km),
            np.asarray(config.velocity_model.shear_speed_km_s),
            np.asarray(config.velocity_model.density_g_cm3),
        )

    # ---- S4: slip, drawn per segment and scaled once ---------------------------
    slip_params = stages.SlipParams(
        covariance=covariance,
        coefficient_of_variation=config.slip.coefficient_of_variation,
        side_taper=config.slip.side_taper,
        top_taper=config.slip.top_taper,
        bottom_taper=config.slip.bottom_taper,
    )
    patterns: dict[str, np.ndarray] = {}
    gaussians: dict[str, np.ndarray] = {}
    references: dict[str, object] = {}
    clipped = 0.0
    for name, mesh in segments.items():
        if is_point:
            # A point source is this pipeline with constant fields, not a separate
            # path: S4 to S6 become constants and S7 to S9 are unchanged.
            patterns[name] = np.ones(mesh.cell_counts)
            gaussians[name] = np.zeros(mesh.cell_counts)
            references[name] = None
        else:
            pattern, gaussian, reference = stages.slip_pattern(
                mesh, slip_params, _stream(config, "slip", position[name]), sampler
            )
            patterns[name] = pattern
            gaussians[name] = gaussian
            references[name] = reference
            clipped = max(clipped, stages.truncated_fraction(gaussian, slip_params))

    # One factor over every segment, so the partition of moment between faults is
    # the fields' own rather than an artefact of scaling each alone.
    scaled = moment.scale_to_moment(
        [patterns[name] for name in names],
        [materials[name][1] for name in names],
        [segments[name].areas_km2() for name in names],
        target_moment,
    )
    slips = dict(zip(names, scaled, strict=True))

    # ---- S5, S6: rise time and rake, per segment ------------------------------
    rise_times: dict[str, np.ndarray] = {}
    rakes: dict[str, np.ndarray] = {}
    for name, mesh in segments.items():
        if is_point:
            # Given rather than derived, and the geometric correction is *not*
            # applied: a rise time the caller chose has already accounted for the
            # geometry.
            rise_times[name] = np.full(mesh.cell_counts, source.rise_time_s)
            rakes[name] = np.full(mesh.cell_counts, config.field.base_rake_deg)
            continue

        average_s = stages.average_rise_time_s(
            target_moment, source.rise_time_coefficient, correction
        )
        rise_times[name] = stages.rise_time_field(
            mesh,
            gaussians[name],
            references[name],
            _rise_time_params(config, average_s),
            _stream(config, "rise_time", position[name]),
            sampler,
            covariance,
            sample_interval_s=config.timing.sample_interval_s,
        )
        rakes[name] = stages.rake_field(
            mesh,
            stages.RakeParams(
                covariance=covariance,
                base_rake_deg=config.field.base_rake_deg,
                sigma_deg=config.slip.rake_sigma_deg,
            ),
            _stream(config, "rake", position[name]),
            sampler,
        )

    # ---- S7, S8: the wavefront, in causal order -------------------------------
    speed_params = timing.SpeedParams(
        velocity_fraction=config.field.velocity_fraction,
        average_dip_deg=source.average_dip_deg,
        average_rake_deg=source.average_rake_deg,
        shallow=_ramp(config.timing.shallow_speed_ramp or config.timing.shallow_ramp),
        deep=_ramp(config.timing.deep_speed_ramp or config.timing.deep_ramp),
        shallow_factor=config.timing.shallow_speed_factor,
        deep_factor=config.timing.deep_speed_factor,
    )
    onset_params = stages.OnsetParams(
        scale_s=config.timing.rupture_time_scale,
        correlation=config.timing.rupture_time_correlation,
        sigma=config.timing.rupture_time_sigma,
        delay_s=config.timing.rupture_delay_s,
    )

    hypocentre = segments[root].cell_index(
        config.hypocentre.strike_km, config.hypocentre.dip_km
    )
    onsets: dict[str, np.ndarray] = {}
    jumps: dict[str, propagation.Jump] = {}

    for name in order:
        mesh = segments[name]
        parent = tree[name]

        if parent is None:
            seeds = [(*hypocentre, 0.0)]
            pinned: tuple[int, int] | None = hypocentre
            delay_s = onset_params.delay_s
        else:
            # The parent's *arrival* is what the front jumps from -- its onset, not
            # its unperturbed wavefront, because the perturbation is part of when the
            # rupture actually got there.
            jump = propagation.causal_jump(
                segments[parent],
                onsets[parent],
                mesh,
                _jump_delay(config, materials, parent, name),
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
            mesh, materials[name][0], speed_params, seeds
        )
        reference = references[name]
        if reference is None:
            onsets[name] = travel_time_s + delay_s
        else:
            onsets[name] = stages.onset_times(
                mesh,
                travel_time_s,
                reference,
                dataclasses.replace(onset_params, delay_s=delay_s),
                _stream(config, "onset", position[name]),
                sampler,
                covariance,
                hypocentre=pinned,
            )

    # ---- S9: pulses -----------------------------------------------------------
    pulse_params = pulses.PulseParams(
        shape=pulses.from_stype(config.timing.slip_rate_shape or "OliuP2"),
        shallow_ramp=_ramp(config.timing.beta_shallow_ramp),
        mid_ramp=_ramp(config.timing.beta_mid_ramp),
        beta_shallow=config.timing.beta_shallow,
        beta_mid=config.timing.beta_mid,
        beta_deep=config.timing.beta_deep,
        sample_interval_s=config.timing.sample_interval_s,
    )

    datasets: dict[str, xr.Dataset] = {}
    for name, mesh in segments.items():
        offsets, samples = pulses.synthesise(
            mesh, slips[name], rise_times[name], pulse_params
        )
        datasets[name] = rupture_format.to_dataset(
            mesh,
            crs,
            slip_m=slips[name],
            rake_deg=rakes[name],
            onset_s=onsets[name],
            rise_time_s=rise_times[name],
            pulse_offsets=offsets,
            pulse_samples=samples,
            sample_interval_s=config.timing.sample_interval_s,
            moment_newton_m=target_moment,
            # Only the segment the rupture nucleated on records a hypocentre;
            # writing one into every group claimed several hypocentres for one
            # earthquake.
            hypocentre_km=(
                (
                    min(config.hypocentre.strike_km, float(mesh.strike_arc_km()[-1])),
                    min(config.hypocentre.dip_km, float(mesh.dip_arc_km()[-1])),
                )
                if name == root
                else None
            ),
        )

    return Realisation(
        segments=datasets,
        tree=tree,
        jumps=jumps,
        moment_newton_m=target_moment,
        hypocentre=hypocentre,
        truncated_fraction=clipped,
    )


def _jump_delay(
    config: RuptureConfig,
    materials: dict[str, tuple[np.ndarray, np.ndarray]],
    parent: str,
    child: str,
) -> propagation.JumpDelay:
    """How long the front takes to cross from one segment to the next.

    The gap is by definition on neither fault, so neither segment's own shear speeds
    describe the rock in it; the mean over the two is the nearest thing available and
    is what the default model uses.
    """
    del config
    speed_km_s = 0.5 * (
        float(np.mean(materials[parent][0])) + float(np.mean(materials[child][0]))
    )
    return propagation.DistanceOverVelocity(speed_km_s)


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
    "STAGE_STREAMS",
    "Realisation",
    "causality_tree",
    "charts_for",
    "generate",
    "segments_of",
]
