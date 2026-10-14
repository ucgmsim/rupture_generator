"""The projected frame a fault system's charts use, checked against its faults.

Positions are kilometres east and north in a projected CRS. The CRS has to measure in
metres and keep lengths true where the faults are. There's no default frame: a frame
made for one region distorts another, and nothing downstream would notice.
"""

from collections.abc import Mapping

import numpy as np
import pyproj

from rupture_generator.errors import RuptureGeneratorError

AREA_MARGIN_DEG = 2.0
"""How far past its area of use, in degrees, a CRS may still place a fault.

An area of use follows the land the CRS was made for, and offshore faults lie past it:
the Puysegur end of the Alpine fault is 0.8 degrees west of NZTM's."""

MAXIMUM_SCALE_ERROR = 0.01
"""How far the CRS may stretch a length at any trace point, as a fraction."""

_LISTED = 5
"""How many faults an error names before it summarises the rest."""


def projected_crs(crs: pyproj.CRS | str | None) -> pyproj.CRS:
    """Read a CRS, and refuse one that isn't projected in metres.

    Parameters
    ----------
    crs : pyproj.CRS, str or None
        The CRS, or anything :class:`pyproj.CRS` reads, such as ``"EPSG:32611"``.

    Returns
    -------
    pyproj.CRS
        The CRS.

    Raises
    ------
    RuptureGeneratorError
        If ``crs`` is missing or unreadable, isn't projected, or measures its axes in
        anything but metres.
    """
    if crs is None or crs == "":
        raise RuptureGeneratorError(
            "a projected CRS in metres is required, and there's no default; choose "
            "one made for the faults' region, such as their UTM zone"
        )
    try:
        frame = crs if isinstance(crs, pyproj.CRS) else pyproj.CRS(crs)
    except pyproj.exceptions.CRSError as error:
        raise RuptureGeneratorError(f"{crs!r} is not a CRS: {error}") from error
    if not frame.is_projected:
        raise RuptureGeneratorError(
            f"{_describe(frame)} is not a projected CRS; positions are kilometres, so "
            "the frame has to be one"
        )
    units = sorted({axis.unit_name for axis in frame.axis_info})
    if units != ["metre"]:
        raise RuptureGeneratorError(
            f"{_describe(frame)} measures its axes in {', '.join(units)}; positions "
            "are kilometres, so choose a CRS in metres made for the faults' region"
        )
    return frame


def check_coverage(crs: pyproj.CRS, traces: Mapping[str, np.ndarray]) -> None:
    """Refuse a CRS made for somewhere other than the faults.

    A fault must lie within :data:`AREA_MARGIN_DEG` of the CRS's area of use, where
    the CRS states one, and the CRS must stretch no length at a trace point by more
    than :data:`MAXIMUM_SCALE_ERROR`. The second catches a CRS with no area of use,
    and one that covers the world, such as Web Mercator, at a latitude it distorts.

    Parameters
    ----------
    crs : pyproj.CRS
        The projected CRS.
    traces : Mapping of str to np.ndarray
        Each fault's trace, ``(n, 2)``, in WGS84 longitude and latitude, by name.

    Raises
    ------
    RuptureGeneratorError
        If a fault lies outside the CRS's area of use, or the CRS stretches lengths at
        a fault by more than the tolerance.
    """
    if not traces:
        return
    points = np.concatenate([np.asarray(trace) for trace in traces.values()])
    advice = f"choose a CRS made for the faults' region, such as {_utm_zone(points)}"

    area = crs.area_of_use
    if area is not None:
        outside = [
            name
            for name, trace in traces.items()
            if not np.all(_within(np.asarray(trace), area.bounds))
        ]
        if outside:
            span = _span(np.concatenate([np.asarray(traces[name]) for name in outside]))
            raise RuptureGeneratorError(
                f"{_describe(crs)} is made for longitude {area.west:g} to "
                f"{area.east:g} and latitude {area.south:g} to {area.north:g}, and "
                f"the faults {_names(outside)} lie outside it, at {span}; {advice}"
            )

    factors = pyproj.Proj(crs).get_factors(points[:, 0], points[:, 1])
    stretch = np.maximum(
        np.abs(np.asarray(factors.meridional_scale) - 1.0),
        np.abs(np.asarray(factors.parallel_scale) - 1.0),
    )
    if not np.all(stretch <= MAXIMUM_SCALE_ERROR):
        counts = np.cumsum([len(trace) for trace in traces.values()])
        worst = int(np.argmax(stretch))
        name = list(traces)[int(np.searchsorted(counts, worst, side="right"))]
        raise RuptureGeneratorError(
            f"{_describe(crs)} stretches lengths by up to {float(stretch[worst]):.1%} "
            f"at the faults, most at {name!r}, past the {MAXIMUM_SCALE_ERROR:.0%} "
            f"tolerance; {advice}"
        )


def _describe(crs: pyproj.CRS) -> str:
    """A CRS as its code and name, for an error message."""
    authority = crs.to_authority()
    code = ":".join(authority) if authority else crs.to_string()
    return f"{code} ({crs.name})"


def _within(
    points: np.ndarray, bounds: tuple[float, float, float, float]
) -> np.ndarray:
    """Whether each point lies within ``bounds``, widened by the margin."""
    west, south, east, north = bounds
    lon, lat = points[:, 0], points[:, 1]
    inside_lat = (lat >= south - AREA_MARGIN_DEG) & (lat <= north + AREA_MARGIN_DEG)
    # A box that crosses the antimeridian has its west edge east of its east edge.
    from_west = (lon - (west - AREA_MARGIN_DEG)) % 360.0
    width = (east - west) % 360.0 or 360.0
    return inside_lat & (from_west <= width + 2.0 * AREA_MARGIN_DEG)


def _names(names: list[str]) -> str:
    """Up to :data:`_LISTED` names, and a count of the rest."""
    listed = ", ".join(repr(name) for name in names[:_LISTED])
    rest = len(names) - _LISTED
    return f"{listed} and {rest} more" if rest > 0 else listed


def _span(points: np.ndarray) -> str:
    """The longitudes and latitudes the points cover."""
    lon, lat = points[:, 0], points[:, 1]
    return (
        f"longitude {lon.min():g} to {lon.max():g} and latitude {lat.min():g} to "
        f"{lat.max():g}"
    )


def _utm_zone(points: np.ndarray) -> str:
    """The WGS84 UTM zone at the points' centre, as an EPSG code."""
    lon, lat = (float(value) for value in points.mean(axis=0))
    zone = int((lon + 180.0) // 6.0) % 60 + 1
    hemisphere = "N" if lat >= 0.0 else "S"
    code = (32600 if lat >= 0.0 else 32700) + zone
    return f"EPSG:{code} (UTM zone {zone}{hemisphere})"


__all__ = [
    "AREA_MARGIN_DEG",
    "MAXIMUM_SCALE_ERROR",
    "check_coverage",
    "projected_crs",
]
