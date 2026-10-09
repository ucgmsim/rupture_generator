"""A drawn rupture as a Standard Rupture Format file, written and read back.

Version 2.0, which records each point's shear speed and density. The SRF is in CGS.
Slip is in centimetres and area in square centimetres. Slip rate and shear speed are
in centimetres per second, and density is in grams per cubic centimetre. This module is the only
place those units appear. Depth stays in kilometres and positions are WGS84 longitude
and latitude.

Every plane of every segment is one SRF plane, in the order :func:`generate` drew the
segments. Its points run down dip in rows, with the along-strike index varying first,
as the format orders them. A plane's strike is the geodesic azimuth along its top
edge, so the projection's grid convergence never reaches the file.
"""

import dataclasses
import itertools
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyproj

from rupture_generator._kernels import synthesise_pulses
from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.geometry import CellArray, Geometry
from rupture_generator.rupture.generator import SegmentRupture
from rupture_generator.rupture.medium import rigidity_pa
from rupture_generator.rupture.realisation import Hypocentre, Realisation
from rupture_generator.srf_parser import (
    PyCsrMatrix,
    PySrfFile,
    PySrfMetadata,
    PySrfPlane,
    parse_srf,
    write_srf,
)

CM_PER_M = 100.0
CM_PER_KM = 1.0e5
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

    ``hypocentre`` is the realisation's if it lies on this segment. The plane holding
    it gets a ``shyp`` and ``dhyp``, along strike from the plane's top centre and down
    dip from its top edge, and every other plane gets neither.
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
    """A cell field flattened in SRF order.

    Plane by plane, then dip rows, with the along-strike index varying first.
    """
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
    """Write drawn segments, and the rock each one read, as an SRF.

    Parameters
    ----------
    path : str
        Where to write the file.
    realisation : Realisation
        The fault system, for its projection and its hypocentre.
    ruptures : Mapping of str to SegmentRupture
        The drawn segments by name, written in the mapping's order.
    dt_s : float
        The slip-rate sample interval in seconds.
    beta : Mapping of str to CellArray, optional
        The Liu-Archuleta-Hartzell rising fraction per cell, by segment. A segment
        without one gets a single-sample impulse per subfault.

    Raises
    ------
    RuptureGeneratorError
        For a slipping subfault whose rise time rounds to no samples at ``dt_s``, or a
        ``beta`` outside ``(0, 0.5]``.
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
            "vs",
            "density",
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
        columns["vs"].append(
            _plane_major(geometry, rupture.shear_speed_km_s) * CM_PER_KM
        )
        columns["density"].append(_plane_major(geometry, rupture.density_g_cm3))

        segment_beta = (
            None
            if beta is None or name not in beta
            else _plane_major(geometry, beta[name])
        )
        try:
            row_offsets, rates = synthesise_pulses(slip_m, rise_s, dt_s, segment_beta)
        except ValueError as error:
            raise RuptureGeneratorError(f"{name}: {error}") from None
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


@dataclasses.dataclass(frozen=True, eq=False)
class SrfRupture:
    """An SRF read back in SI: one array per point column, in file order.

    Parameters
    ----------
    planes : list of PySrfPlane
        The plane headers, as the file states them.
    hypocentre : tuple of float or None
        ``(longitude, latitude, depth_km)``, or ``None`` if no plane records one.
    lon_deg, lat_deg : np.ndarray
        Each point's WGS84 longitude and latitude in degrees.
    depth_km : np.ndarray
        Each point's depth in kilometres.
    strike_deg, dip_deg : np.ndarray
        Each point's strike and dip in degrees.
    area_m2 : np.ndarray
        Each point's area in square metres.
    onset_s : np.ndarray
        When each point starts to slip, in seconds.
    dt_s : np.ndarray
        Each point's slip-rate sample interval in seconds.
    rake_deg : np.ndarray
        Each point's rake in degrees.
    slip_m : np.ndarray
        Each point's slip, in metres.
    rise_time_s : np.ndarray
        Each point's rise time in seconds.
    shear_speed_km_s : np.ndarray
        The shear speed at each point in kilometres per second.
    density_g_cm3 : np.ndarray
        The density at each point in grams per cubic centimetre.
    pulse_offsets : np.ndarray
        Row offsets into ``pulses_m_s``: point ``k``'s slip rate is
        ``pulses_m_s[pulse_offsets[k]:pulse_offsets[k + 1]]``, from its own onset.
    pulses_m_s : np.ndarray
        Every point's slip rate in metres per second, end to end, in single precision
        as the file holds it.
    """

    planes: list[PySrfPlane]
    hypocentre: tuple[float, float, float] | None
    lon_deg: np.ndarray
    lat_deg: np.ndarray
    depth_km: np.ndarray
    strike_deg: np.ndarray
    dip_deg: np.ndarray
    area_m2: np.ndarray
    onset_s: np.ndarray
    dt_s: np.ndarray
    rake_deg: np.ndarray
    slip_m: np.ndarray
    rise_time_s: np.ndarray
    shear_speed_km_s: np.ndarray
    density_g_cm3: np.ndarray
    pulse_offsets: np.ndarray
    pulses_m_s: np.ndarray

    @property
    def rigidity_pa(self) -> np.ndarray:
        """np.ndarray: Each point's rigidity in pascals, from the file's own rock."""
        return rigidity_pa(self.shear_speed_km_s, self.density_g_cm3)


def _hypocentre(plane: PySrfPlane) -> tuple[float, float, float]:
    """The inverse of :func:`_planes`: ``shyp`` along strike from the top centre, then
    ``dhyp`` down dip."""
    dip = np.radians(plane.dip)
    lon, lat, _ = _GEOD.fwd(plane.elon, plane.elat, plane.stk, plane.shyp * M_PER_KM)
    lon, lat, _ = _GEOD.fwd(
        lon, lat, plane.stk + 90.0, plane.dhyp * np.cos(dip) * M_PER_KM
    )
    return float(lon), float(lat), plane.dtop + plane.dhyp * float(np.sin(dip))


def read_rupture(path: str | Path) -> SrfRupture:
    """Read a version 2 SRF, as :func:`write_rupture` writes, into SI.

    Only the ``slip1`` component comes back, the only one this package writes.

    Parameters
    ----------
    path : str or Path
        The SRF to read.

    Returns
    -------
    SrfRupture
        The file's points and pulses, in SI.

    Raises
    ------
    RuptureGeneratorError
        If the file isn't an SRF, or is version 1 and so lacks the rock.
    OSError
        If reading the file fails.
    """
    try:
        srf = parse_srf(Path(path).read_bytes())
    except ValueError as error:
        raise RuptureGeneratorError(f"{path}: {error}") from None
    points = srf.metadata
    if points.vs is None or points.density is None:
        raise RuptureGeneratorError(
            f"{path} is a version 1 SRF, which records no shear speed or density"
        )
    pulses = srf.slipt1.data
    pulses /= CM_PER_M
    located = [plane for plane in srf.planes if plane.shyp != NO_HYPOCENTRE]
    return SrfRupture(
        planes=list(srf.planes),
        hypocentre=_hypocentre(located[0]) if located else None,
        lon_deg=points.lon.astype(np.float64),
        lat_deg=points.lat.astype(np.float64),
        depth_km=points.dep.astype(np.float64),
        strike_deg=points.stk.astype(np.float64),
        dip_deg=points.dip.astype(np.float64),
        area_m2=points.area.astype(np.float64) / CM_PER_M**2,
        onset_s=points.tinit.astype(np.float64),
        dt_s=points.dt.astype(np.float64),
        rake_deg=points.rake.astype(np.float64),
        slip_m=points.slip1.astype(np.float64) / CM_PER_M,
        rise_time_s=points.rise.astype(np.float64),
        shear_speed_km_s=points.vs.astype(np.float64) / CM_PER_KM,
        density_g_cm3=points.density.astype(np.float64),
        pulse_offsets=srf.slipt1.row_ptr.astype(np.int64),
        pulses_m_s=pulses,
    )


__all__ = ["NO_HYPOCENTRE", "SrfRupture", "read_rupture", "write_rupture"]
