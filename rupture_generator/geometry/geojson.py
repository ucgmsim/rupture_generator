"""Fault sections as GeoJSON, read into charts.

A fault system is one ``FeatureCollection``. Each ``Feature`` is a section: a named
surface trace as a ``LineString`` in WGS84 longitude and latitude (:rfc:`7946`), hung
on one dip and one depth range. That is OpenQuake's ``simpleFaultSource`` and the
data model of the GEM, USGS and New Zealand hazard models, so a hazard model's file
loads here and opens in QGIS. A section whose dip changes along strike is two
sections.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self, TextIO

import numpy as np
import pyproj

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.geometry.geometry import Geometry

type TraceArray = np.ndarray[tuple[int, int], np.dtype[np.float64]]
"""Surface trace, ``(n+1, 2)``: longitude and latitude in WGS84 degrees."""

NSHM_ALIASES: Mapping[str, str] = {
    "FaultName": "name",
    "DipDeg": "dip_deg",
    "DipDir": "dip_direction_deg",
    "UpDepth": "upper_depth_km",
    "LowDepth": "lower_depth_km",
}
"""The New Zealand hazard model's property names, mapped onto this module's own."""

PARALLEL_TOLERANCE_DEG = 5.0
"""How close to a plane's strike a dip direction may point before it picks no side:
within a few degrees the side is whichever way rounding went."""

_GEOD = pyproj.Geod(ellps="WGS84")


@dataclass(frozen=True)
class Segment:
    """One planar fault section, its trace ordered so the fault dips to the right."""

    trace: TraceArray
    name: str
    dip_deg: float
    upper_depth_km: float
    lower_depth_km: float

    @classmethod
    def from_feature(
        cls, feature: Mapping[str, Any], aliases: Mapping[str, str] | None = None
    ) -> Self:
        """A segment from a GeoJSON ``Feature`` with a ``LineString`` trace.

        ``aliases`` renames properties before they are read; pass
        :data:`NSHM_ALIASES` for a New Zealand hazard model file.

        Raises
        ------
        RuptureGeneratorError
            For a missing property, a dip outside ``(0, 90]``, a depth range that does
            not reach downward, or a dip direction that does not pick one side of every
            plane.
        """
        properties = {
            (aliases or {}).get(key, key): value
            for key, value in (feature.get("properties") or {}).items()
        }
        try:
            name = str(properties["name"])
            dip_deg = float(properties["dip_deg"])
            dip_direction_deg = float(properties["dip_direction_deg"])
            upper_depth_km = float(properties["upper_depth_km"])
            lower_depth_km = float(properties["lower_depth_km"])
        except KeyError as missing:
            raise RuptureGeneratorError(
                f"a section has no {missing} property"
            ) from None
        if not 0.0 < dip_deg <= 90.0:
            raise RuptureGeneratorError(
                f"{name}: a dip of {dip_deg} degrees is not in (0, 90]"
            )
        if not 0.0 <= upper_depth_km < lower_depth_km:
            raise RuptureGeneratorError(
                f"{name}: {upper_depth_km} to {lower_depth_km} km is not a depth range"
            )

        trace = np.asarray(feature["geometry"]["coordinates"], dtype=np.float64)
        if trace.ndim != 2 or trace.shape[1] != 2 or len(trace) < 2:
            raise RuptureGeneratorError(
                f"{name}: the trace is shaped {trace.shape}; it wants at least two "
                "longitude, latitude pairs"
            )
        # Each plane's strike on the ellipsoid; the dip direction only picks a side.
        strikes_deg, _, _ = _GEOD.inv(
            trace[:-1, 0], trace[:-1, 1], trace[1:, 0], trace[1:, 1]
        )
        offsets_deg = (dip_direction_deg - np.asarray(strikes_deg)) % 360.0
        from_strike_deg = np.minimum(offsets_deg % 180.0, 180.0 - offsets_deg % 180.0)
        if np.any(from_strike_deg < PARALLEL_TOLERANCE_DEG):
            raise RuptureGeneratorError(
                f"{name}: the dip direction {dip_direction_deg:g} runs along a plane's "
                "strike, so it picks no side of the trace"
            )
        right = offsets_deg < 180.0
        if not (right.all() or not right.any()):
            raise RuptureGeneratorError(
                f"{name}: the dip direction {dip_direction_deg:g} falls to the left of "
                "some planes and the right of others, so the trace doubles back past "
                "it; split the section at the reversal"
            )
        if not right[0]:
            trace = trace[::-1].copy()

        return cls(
            trace=trace,
            name=name,
            dip_deg=dip_deg,
            upper_depth_km=upper_depth_km,
            lower_depth_km=lower_depth_km,
        )

    def to_geometry(self, transformer: pyproj.Transformer) -> Geometry:
        """The coarsest chart: one cell per plane, each plane keeping its own seam.

        ``transformer`` takes WGS84 longitude and latitude to the projected CRS, in
        metres. Each plane hangs perpendicular to its own projected strike.
        :meth:`Geometry.subdivide` cuts the chart to a working resolution.
        """
        east_m, north_m = transformer.transform(self.trace[:, 0], self.trace[:, 1])
        trace_km = np.column_stack([east_m, north_m]) / 1000.0

        steps_km = np.diff(trace_km, axis=0)
        down_dip_rad = np.arctan2(steps_km[:, 0], steps_km[:, 1]) + np.pi / 2
        depth_km = self.lower_depth_km - self.upper_depth_km
        reach_km = depth_km / np.tan(np.radians(self.dip_deg))
        down = np.column_stack(
            [
                reach_km * np.sin(down_dip_rad),
                reach_km * np.cos(down_dip_rad),
                np.full(len(steps_km), depth_km),
            ]
        )

        surface = np.column_stack(
            [trace_km, np.full(len(trace_km), self.upper_depth_km)]
        )
        # Each plane's near then far node, so every plane carries its own seam.
        top = np.stack([surface[:-1], surface[1:]], axis=1).reshape(-1, 3)
        bottom = top + np.repeat(down, 2, axis=0)
        planes = len(steps_km)
        return Geometry(
            nodes=np.stack([top, bottom]),
            occupied=np.ones((1, planes), dtype=bool),
            plane_cells=(1,) * planes,
        )


def geometry_from_geojson(
    handle: TextIO, crs: pyproj.CRS, *, aliases: Mapping[str, str] | None = None
) -> dict[str, Geometry]:
    """Read a ``FeatureCollection`` of sections into coarse charts, keyed by name.

    ``crs`` is the projected frame the charts are built in, and the one their
    :class:`~rupture_generator.rupture.realisation.Realisation` records.

    Raises
    ------
    RuptureGeneratorError
        If the document is not a collection of ``LineString`` sections, or two
        sections share a name.
    """
    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    def _hook(obj: dict[str, Any]) -> Any:
        match obj:
            case {"type": "Feature", "geometry": {"type": "LineString"}}:
                return Segment.from_feature(obj, aliases)
            case {"type": "Feature", "geometry": {"type": kind}}:
                raise RuptureGeneratorError(
                    f"a section is a LineString trace, not a {kind}"
                )
            case {"type": "FeatureCollection", "features": [*segments]}:
                charts = {
                    segment.name: segment.to_geometry(transformer)
                    for segment in segments
                }
                if len(charts) != len(segments):
                    raise RuptureGeneratorError("two sections share a name")
                return charts
            case _:
                return obj

    charts = json.load(handle, object_hook=_hook)
    if (
        not isinstance(charts, dict)
        or not charts
        or not all(isinstance(chart, Geometry) for chart in charts.values())
    ):
        raise RuptureGeneratorError("a fault system is a FeatureCollection of sections")
    return charts
