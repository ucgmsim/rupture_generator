"""Fault geometry: one chart type, and the GeoJSON sections and surfaces that build it."""

from rupture_generator.geometry.crs import check_coverage, projected_crs
from rupture_generator.geometry.geojson import (
    NSHM_ALIASES,
    Segment,
    geometry_from_geojson,
)
from rupture_generator.geometry.geometry import (
    CellArray,
    CellMask,
    CellSelection,
    Geometry,
    NodeArray,
)
from rupture_generator.geometry.surface import flatten, read_tsurf, surface_chart

__all__ = [
    "NSHM_ALIASES",
    "CellArray",
    "CellMask",
    "CellSelection",
    "Geometry",
    "NodeArray",
    "Segment",
    "check_coverage",
    "flatten",
    "geometry_from_geojson",
    "projected_crs",
    "read_tsurf",
    "surface_chart",
]
