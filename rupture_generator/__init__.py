"""Kinematic rupture generation.

The library is plain data and plain functions, and the usual run is five calls::

    charts = {name: chart.subdivide(0.1) for name, chart in
              geometry_from_geojson(handle, crs).items()}
    realisation = sample_path(Realisation(charts, crs), hypocentre, rng=rng)
    medium = layered_medium(Layers(bottom_depth_km), shear_speed_km_s, density_g_cm3)
    drawn = generate(realisation, medium, sources, RuptureSettings(), seed=seed)
    write_rupture(path, drawn)

:func:`write_fields` stores the drawn fields instead, a fraction of the SRF's size,
and :func:`~rupture_generator.formats.fields.fields_to_srf` makes the SRF from them
later.

:mod:`rupture_generator.config` builds the same inputs from a TOML file, and the
command line runs that file. :func:`rupture_generator.sampling.sampler` keeps the
last 64 covariance embeddings, two per segment, so drawing a fault system of up to
32 segments again reuses every one. ``sampler.cache_clear()`` releases them.
"""

from importlib.metadata import version

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.formats.fields import read_fields, write_fields
from rupture_generator.formats.srf import write_rupture
from rupture_generator.geometry import Geometry, Segment, geometry_from_geojson
from rupture_generator.rupture.generator import (
    FaultProfiles,
    JumpRule,
    PulseSettings,
    RakeSettings,
    RiseSettings,
    RuptureSettings,
    SegmentRupture,
    SlipSettings,
    TimingSettings,
    first_arrival,
    generate,
    generate_segment,
)
from rupture_generator.rupture.medium import (
    Layers,
    Medium,
    SpatialField,
    constant_field,
    interpolated_field,
    layered_field,
    layered_medium,
    ramp_field,
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

__version__ = version("rupture-generator")
"""The installed version, from the latest git tag when it was built."""

__all__ = [
    "MAI_BEROZA",
    "CorrelationRelation",
    "FaultProfiles",
    "Geometry",
    "Hypocentre",
    "JumpModel",
    "JumpRule",
    "Layers",
    "Medium",
    "PulseSettings",
    "RakeSettings",
    "Realisation",
    "RiseSettings",
    "RuptureGeneratorError",
    "RuptureSettings",
    "Segment",
    "SegmentRupture",
    "SegmentSource",
    "SlipSettings",
    "SpatialField",
    "TimingSettings",
    "constant_field",
    "first_arrival",
    "generate",
    "generate_segment",
    "geometry_from_geojson",
    "interpolated_field",
    "layered_field",
    "layered_medium",
    "likeliest_path",
    "magnitude_from_moment",
    "moment_from_magnitude",
    "ramp_field",
    "read_fields",
    "sample_path",
    "segment_source",
    "split_moment",
    "write_fields",
    "write_rupture",
]
