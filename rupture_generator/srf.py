"""Writing a drawn rupture as a Standard Rupture Format file.

The SRF is in CGS -- slip in centimetres, area in square centimetres, slip rate in
centimetres per second -- and this module is the only place those units appear. Depth
stays in kilometres and positions are WGS84 longitude and latitude.

Every plane of every segment is one SRF plane, in the order the segments were drawn.
Its points run down dip in rows, strike fastest, as the format orders them. A plane's
strike is the geodesic azimuth along its top edge, so the projection's grid
convergence never reaches the file.
"""

import itertools
from collections.abc import Mapping

import numpy as np
import pyproj

from rupture_generator._kernels import synthesise_pulses
from rupture_generator.geometry import CellArray, Geometry
from rupture_generator.rupture.generator import SegmentRupture
from rupture_generator.rupture.realisation import Hypocentre, Realisation
from rupture_generator.srf_parser import (
    PyCsrMatrix,
    PySrfFile,
    PySrfMetadata,
    PySrfPlane,
    write_srf,
)

CM_PER_M = 100.0
CM2_PER_KM2 = 1.0e10
M_PER_KM = 1000.0

NO_HYPOCENTRE = -999.0
"""``shyp`` and ``dhyp`` on a plane the rupture did not start on."""

_GEOD = pyproj.Geod(ellps="WGS84")


def _plane_slices(geometry: Geometry) -> list[slice]:
    """Each plane's cell columns, in trace order."""
    edges = np.cumsum([0, *geometry.plane_cells])
    return [slice(int(a), int(b)) for a, b in itertools.pairwise(edges)]


def _to_lon_lat(
    transformer: pyproj.Transformer, east_km: np.ndarray, north_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = transformer.transform(east_km * M_PER_KM, north_km * M_PER_KM)
    return np.asarray(lon), np.asarray(lat)


def _planes(
    geometry: Geometry,
    hypocentre: Hypocentre | None,
    transformer: pyproj.Transformer,
) -> tuple[list[PySrfPlane], list[float]]:
    """One header per plane, and each plane's strike for its points.

    ``hypocentre`` is the realisation's if it lies on this segment, and only the plane
    holding it gets a ``shyp`` and ``dhyp``: along strike from the plane's top centre,
    and down dip from its top edge.
    """
    arc_km = geometry.strike_arc_km
    width_km = float(geometry.dip_arc_km[-1])
    headers, strikes = [], []
    for columns, plane in zip(
        _plane_slices(geometry), geometry.plane_nodes(), strict=True
    ):
        start_km, end_km = float(arc_km[columns.start]), float(arc_km[columns.stop])
        (lon0, lon1), (lat0, lat1) = _to_lon_lat(
            transformer, plane[0, [0, -1], 0], plane[0, [0, -1], 1]
        )
        strike_deg, _, _ = _GEOD.inv(lon0, lat0, lon1, lat1)
        strike_deg %= 360.0
        centre_lon, centre_lat = _to_lon_lat(
            transformer, plane[0, [0, -1], 0].mean(), plane[0, [0, -1], 1].mean()
        )
        on_plane = hypocentre is not None and start_km <= hypocentre.strike_km <= end_km
        headers.append(
            PySrfPlane(
                elon=float(centre_lon),
                elat=float(centre_lat),
                nstk=columns.stop - columns.start,
                ndip=geometry.cells[0],
                len=end_km - start_km,
                wid=width_km,
                stk=strike_deg,
                dip=float(geometry.dip_deg[:, columns].mean()),
                dtop=float(plane[0, 0, 2]),
                shyp=(
                    hypocentre.strike_km - 0.5 * (start_km + end_km)
                    if on_plane
                    else NO_HYPOCENTRE
                ),
                dhyp=hypocentre.dip_km if on_plane else NO_HYPOCENTRE,
            )
        )
        strikes.append(strike_deg)
        if on_plane:
            hypocentre = None
    return headers, strikes


def _plane_major(geometry: Geometry, field: CellArray) -> np.ndarray:
    """A cell field flattened in SRF order: plane by plane, then dip rows, strike
    fastest."""
    return np.concatenate(
        [field[:, columns].ravel() for columns in _plane_slices(geometry)]
    )


def write_rupture(
    path: str,
    realisation: Realisation,
    ruptures: Mapping[str, SegmentRupture],
    *,
    dt_s: float,
    beta: Mapping[str, CellArray] | None = None,
) -> None:
    """Write drawn segments as an SRF (version 1).

    ``beta`` is the Liu-Archuleta-Hartzell rising fraction per cell, by segment; a
    segment without one gets a single-sample impulse per subfault.

    Raises
    ------
    ValueError
        From pulse synthesis, for a slipping subfault whose rise time rounds to no
        samples at ``dt_s`` or a ``beta`` outside ``(0, 0.5]``.
    """
    transformer = pyproj.Transformer.from_crs(
        realisation.crs, "EPSG:4326", always_xy=True
    )
    planes: list[PySrfPlane] = []
    columns: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "lon",
            "lat",
            "dep",
            "stk",
            "dip",
            "area",
            "tinit",
            "rake",
            "slip1",
            "rise",
        )
    }
    offsets: list[np.ndarray] = [np.zeros(1, dtype=np.int64)]
    samples: list[np.ndarray] = []

    for name, rupture in ruptures.items():
        geometry = rupture.geometry
        on_segment = (
            realisation.hypocentre
            if realisation.hypocentre and realisation.hypocentre.segment == name
            else None
        )
        headers, strikes = _planes(geometry, on_segment, transformer)
        planes.extend(headers)

        centres = geometry.centres
        lon, lat = _to_lon_lat(
            transformer,
            _plane_major(geometry, centres[..., 0]),
            _plane_major(geometry, centres[..., 1]),
        )
        slip_m, rise_s = (
            _plane_major(geometry, rupture.slip_m),
            _plane_major(geometry, rupture.rise_time_s),
        )
        columns["lon"].append(lon)
        columns["lat"].append(lat)
        columns["dep"].append(_plane_major(geometry, centres[..., 2]))
        columns["stk"].append(
            np.repeat(strikes, [geometry.cells[0] * n for n in geometry.plane_cells])
        )
        columns["dip"].append(_plane_major(geometry, geometry.dip_deg))
        columns["area"].append(_plane_major(geometry, geometry.areas_km2) * CM2_PER_KM2)
        columns["tinit"].append(_plane_major(geometry, rupture.onset_s))
        columns["rake"].append(_plane_major(geometry, rupture.rake_deg))
        columns["slip1"].append(slip_m * CM_PER_M)
        columns["rise"].append(rise_s)

        segment_beta = (
            None
            if beta is None or name not in beta
            else _plane_major(geometry, beta[name])
        )
        row_offsets, rates = synthesise_pulses(slip_m, rise_s, dt_s, segment_beta)
        offsets.append(row_offsets[1:] + offsets[-1][-1])
        samples.append(rates * CM_PER_M)

    count = sum(len(lon) for lon in columns["lon"])
    metadata = PySrfMetadata(
        **{
            name: np.concatenate(parts).astype(np.float32)
            for name, parts in columns.items()
        },
        dt=np.full(count, dt_s, dtype=np.float32),
    )
    slip_rate = PyCsrMatrix(
        row_ptr=np.concatenate(offsets).astype(np.uint64),
        data=np.concatenate(samples).astype(np.float32),
    )
    write_srf(PySrfFile(planes, metadata, slip_rate), path)


__all__ = ["NO_HYPOCENTRE", "write_rupture"]
