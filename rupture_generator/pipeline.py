"""The pipeline: the one place the stage order is written down.

.. code-block:: text

    S1  geometry            -> coarse mesh          mesh.build_surface
    S2  coarse mesh         -> chart                (same call; subdivision)
    S3  chart               -> validated chart      mesh.validate_chart
    S4  chart, Mw, rng      -> slip                 stages.slip_pattern + moment
    S5  chart, slip, rng    -> rise time            stages.rise_time_field
    S6  chart, rng          -> rake                 stages.rake_field
    S7  chart, hypocentre   -> travel time          timing.travel_times
    S8  travel time, rng    -> onset                stages.onset_times
    S9  slip, rise, onset   -> pulses               pulses.synthesise

Unlike the port's `generate`, this order is a **convention rather than a contract**.
Each stage draws from its own substream, spawned by name from the event seed, so
reordering them or changing one stage's parameters cannot change another's noise. The
port could not do that: its stages shared one stream in a fixed order, and two dead
fields were drawn and discarded on every run purely to keep that order intact.

The result is the input chart with `slip`, `rake`, `rise_time`, `onset` and pulses
attached -- which is exactly what the rupture file stores, so "the pipeline returns an
annotated mesh" and "the pipeline's output is the file" are one statement.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyproj
import xarray as xr

from rupture_generator import moment, pulses, stages, timing
from rupture_generator.config.geometry import GeometryConfig
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

STAGE_STREAMS = ("slip", "rise_time", "rake", "onset")
"""The named substreams, one per stage that draws.

Spawned in this order from the event seed. The *names* are what matter -- they are
what makes a stage's noise a function of the seed and its own identity, rather than of
what ran before it.
"""


@dataclasses.dataclass(frozen=True)
class Realisation:
    """One generated rupture: the annotated charts, and what they cost.

    Attributes
    ----------
    segments : list of xr.Dataset
        One per segment, in the shape the rupture file stores.
    moment_newton_m : float
        The moment the whole event carries -- the target, closed exactly.
    hypocentre : tuple of int
        The ``(i, j)`` cell the rupture started from, on ``segments[0]``.
    truncated_fraction : float
        What fraction of the fault the slip truncation clipped. A diagnostic: a large
        value says the requested variation was not really achievable.
    """

    segments: list[xr.Dataset]
    moment_newton_m: float
    hypocentre: tuple[int, int]
    truncated_fraction: float


def _substreams(config: RuptureConfig) -> dict[str, np.random.Generator]:
    """A named generator per stage, from the one event seed.

    ``realisation`` selects an independent stream from the same seed, which is what
    makes a campaign restartable: realisation 7 of seed 1234 is reproducible without
    generating the six before it.
    """
    root = np.random.SeedSequence(
        entropy=config.random.seed, spawn_key=(config.random.realisation,)
    )
    return {
        name: np.random.default_rng(child)
        for name, child in zip(
            STAGE_STREAMS, root.spawn(len(STAGE_STREAMS)), strict=True
        )
    }


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


def charts_for(geometry: GeometryConfig, surface_name: str | None) -> list[RuptureMesh]:
    """S1-S3: the validated segments of one surface.

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


def generate(
    config: RuptureConfig,
    segments: list[RuptureMesh],
    crs: pyproj.CRS,
    *,
    sampler: FieldSampler | None = None,
) -> Realisation:
    """Run the pipeline over a surface's segments.

    Parameters
    ----------
    config : RuptureConfig
        What the earthquake is.
    segments : list of RuptureMesh
        The validated charts, from :func:`charts_for`.
    crs : pyproj.CRS
        The frame they are in, for the one projection seam.
    sampler : FieldSampler, optional
        Defaults to the spectral sampler. The seam a Matern sampler enters through.

    Returns
    -------
    Realisation

    Raises
    ------
    ValueError
        For a hypocentre off the fault, an unrepresentable rise time (naming the
        subfault), or a rupture speed the front cannot travel at.
    """
    sampler = sampler or SpectralSampler()
    rngs = _substreams(config)
    covariance = _covariance(config)
    source = config.source

    # Multi-segment propagation is Phase 4's; until then a surface that split into
    # several segments is a rupture nobody has said how to cross.
    if len(segments) != 1:
        raise ValueError(
            f"this surface is {len(segments)} segments -- its planes do not share a "
            "seam, so a rupture front would have to jump between them. Crossing a "
            "segment boundary needs the propagation stage, which is not written"
        )
    (mesh,) = segments

    hypocentre = mesh.cell_index(config.hypocentre.strike_km, config.hypocentre.dip_km)
    depth_km = mesh.centres()[..., 2]
    shear_speed_km_s, rigidity_pa = moment.sample_velocity_model(
        depth_km,
        np.asarray(config.velocity_model.bottom_depth_km),
        np.asarray(config.velocity_model.shear_speed_km_s),
        np.asarray(config.velocity_model.density_g_cm3),
    )

    target_moment = moment.seismic_moment_nm(source.magnitude)
    correction = timing.alpha_t(source.average_dip_deg, source.average_rake_deg)

    # ---- S4: slip -------------------------------------------------------------
    slip_params = stages.SlipParams(
        covariance=covariance,
        coefficient_of_variation=config.slip.coefficient_of_variation,
        side_taper=config.slip.side_taper,
        top_taper=config.slip.top_taper,
        bottom_taper=config.slip.bottom_taper,
    )
    if isinstance(source, PointSourceConfig):
        # A point source is this pipeline with constant fields, not a separate path:
        # S4 to S6 become constants and S7 to S9 are unchanged.
        pattern = np.ones(mesh.cell_counts)
        gaussian = np.zeros(mesh.cell_counts)
        reference = None
        clipped = 0.0
    else:
        pattern, gaussian, reference = stages.slip_pattern(
            mesh, slip_params, rngs["slip"], sampler
        )
        clipped = stages.truncated_fraction(gaussian, slip_params)

    (slip_m,) = moment.scale_to_moment(
        [pattern], [rigidity_pa], [mesh.areas_km2()], target_moment
    )

    # ---- S5: rise time --------------------------------------------------------
    if isinstance(source, PointSourceConfig):
        # Given rather than derived, and the geometric correction is *not* applied:
        # a rise time the caller chose has already accounted for the geometry.
        rise_time_s = np.full(mesh.cell_counts, source.rise_time_s)
    else:
        average_s = stages.average_rise_time_s(
            target_moment, source.rise_time_coefficient, correction
        )
        rise_time_s = stages.rise_time_field(
            mesh,
            gaussian,
            reference,
            _rise_time_params(config, average_s),
            rngs["rise_time"],
            sampler,
            covariance,
            sample_interval_s=config.timing.sample_interval_s,
        )

    # ---- S6: rake -------------------------------------------------------------
    rake_params = stages.RakeParams(
        covariance=covariance,
        base_rake_deg=config.field.base_rake_deg,
        sigma_deg=config.slip.rake_sigma_deg,
    )
    if isinstance(source, PointSourceConfig):
        rake_deg = np.full(mesh.cell_counts, config.field.base_rake_deg)
    else:
        rake_deg = stages.rake_field(mesh, rake_params, rngs["rake"], sampler)

    # ---- S7: wavefront --------------------------------------------------------
    speed_params = timing.SpeedParams(
        velocity_fraction=config.field.velocity_fraction,
        average_dip_deg=source.average_dip_deg,
        average_rake_deg=source.average_rake_deg,
        shallow=_ramp(config.timing.shallow_speed_ramp or config.timing.shallow_ramp),
        deep=_ramp(config.timing.deep_speed_ramp or config.timing.deep_ramp),
        shallow_factor=config.timing.shallow_speed_factor,
        deep_factor=config.timing.deep_speed_factor,
    )
    travel_time_s = timing.travel_times(
        mesh, shear_speed_km_s, speed_params, [(*hypocentre, 0.0)]
    )

    # ---- S8: onset ------------------------------------------------------------
    onset_params = stages.OnsetParams(
        scale_s=config.timing.rupture_time_scale,
        correlation=config.timing.rupture_time_correlation,
        sigma=config.timing.rupture_time_sigma,
        delay_s=config.timing.rupture_delay_s,
    )
    if reference is None:
        onset_s = travel_time_s + onset_params.delay_s
    else:
        onset_s = stages.onset_times(
            mesh,
            travel_time_s,
            reference,
            onset_params,
            rngs["onset"],
            sampler,
            covariance,
            hypocentre=hypocentre,
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
    offsets, samples = pulses.synthesise(mesh, slip_m, rise_time_s, pulse_params)

    strike_arc = mesh.strike_arc_km()
    dip_arc = mesh.dip_arc_km()
    dataset = rupture_format.to_dataset(
        mesh,
        crs,
        slip_m=slip_m,
        rake_deg=rake_deg,
        onset_s=onset_s,
        rise_time_s=rise_time_s,
        pulse_offsets=offsets,
        pulse_samples=samples,
        sample_interval_s=config.timing.sample_interval_s,
        moment_newton_m=target_moment,
        hypocentre_km=(
            min(config.hypocentre.strike_km, float(strike_arc[-1])),
            min(config.hypocentre.dip_km, float(dip_arc[-1])),
        ),
    )

    return Realisation(
        segments=[dataset],
        moment_newton_m=target_moment,
        hypocentre=hypocentre,
        truncated_fraction=clipped,
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


__all__ = ["STAGE_STREAMS", "Realisation", "charts_for", "generate"]
