"""Kinematic rupture generation.

The library is plain data and plain functions, and the usual run is five calls::

    charts = {name: chart.subdivide(0.1) for name, chart in
              geometry_from_geojson(handle, crs).items()}
    realisation = sample_path(Realisation(charts, crs), hypocentre, rng=rng)
    materials = {name: sample_materials(chart, shear_speed_km_s=vs, density_g_cm3=rho)
                 for name, chart in charts.items()}
    ruptures = generate(realisation, materials, sources, RuptureSettings(), seed=seed)
    write_rupture(path, realisation, ruptures, materials, dt_s=0.005)

:mod:`rupture_generator.config` builds the same inputs from a TOML file, and the
command line runs that file. Embeddings of slip covariances are cached between calls
by :func:`rupture_generator.sampling.sampler`; ``sampler.cache_clear()`` releases them.
"""

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.formats.srf import write_rupture
from rupture_generator.geometry import Geometry, Segment, geometry_from_geojson
from rupture_generator.rupture.generator import (
    JumpRule,
    RakeSettings,
    RiseSettings,
    RuptureSettings,
    SegmentRupture,
    SlipSettings,
    TimingSettings,
    generate,
    generate_segment,
    jump_seed,
)
from rupture_generator.rupture.materials import (
    Layers,
    Materials,
    constant_sampler,
    interpolated_sampler,
    layered_1d_sampler,
    ramp_sampler,
    sample_materials,
    velocity_model,
)
from rupture_generator.rupture.propagator import JumpModel, likeliest_path, sample_path
from rupture_generator.rupture.realisation import Hypocentre, Realisation
from rupture_generator.rupture.source import (
    MAI_BEROZA,
    CorrelationRelation,
    SegmentSource,
    magnitude_from_moment,
    moment_from_magnitude,
    segment_source,
    split_moment,
)

__all__ = [
    "MAI_BEROZA",
    "CorrelationRelation",
    "Geometry",
    "Hypocentre",
    "JumpModel",
    "JumpRule",
    "Layers",
    "Materials",
    "RakeSettings",
    "Realisation",
    "RiseSettings",
    "RuptureGeneratorError",
    "RuptureSettings",
    "Segment",
    "SegmentRupture",
    "SegmentSource",
    "SlipSettings",
    "TimingSettings",
    "constant_sampler",
    "generate",
    "generate_segment",
    "geometry_from_geojson",
    "interpolated_sampler",
    "jump_seed",
    "layered_1d_sampler",
    "likeliest_path",
    "magnitude_from_moment",
    "moment_from_magnitude",
    "ramp_sampler",
    "sample_materials",
    "sample_path",
    "segment_source",
    "split_moment",
    "velocity_model",
    "write_rupture",
]
