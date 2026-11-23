"""A drawn rupture as a Standard Rupture Format file, written and read back.

Version 2.0, which records each point's shear speed and density. The SRF is in CGS.
Slip is in centimetres and area in square centimetres. Slip rate and shear speed are
in centimetres per second, and density is in grams per cubic centimetre. This module is the only
place those units appear. Depth stays in kilometres and positions are WGS84 longitude
and latitude.

Every plane of every segment is one SRF plane, in the order :func:`generate` drew the
segments. Its points run down dip in rows, with the along-strike index varying first,
as the format orders them, unoccupied cells included as points that never slip.

Each point's strike is its cell's own, which
:func:`~rupture_generator.geometry.crs.true_azimuth_deg` turns from
:attr:`Geometry.strike` into a true azimuth. Rake, measured from strike, means the same
however the chart's grid lies, and no strike includes the projection's grid
convergence.
"""

import dataclasses
import itertools
import secrets
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyproj
import srf

from rupture_generator._kernels import synthesise_pulses
from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.geometry import CellArray, CellMask, Geometry, NodeArray
from rupture_generator.geometry.crs import to_lon_lat, true_azimuth_deg
from rupture_generator.geometry.geometry import bordering
from rupture_generator.rupture.generator import SegmentRupture
from rupture_generator.rupture.medium import rigidity_pa
from rupture_generator.rupture.realisation import Hypocentre, Realisation

CM_PER_M = 100.0
CM_PER_KM = 1.0e5
CM2_PER_KM2 = 1.0e10

NO_HYPOCENTRE = -999.0
"""``shyp`` and ``dhyp`` on a plane the rupture did not start on."""


def _plane_slices(geometry: Geometry) -> list[slice]:
    """Each plane's cell columns, in trace order."""
    edges = np.cumsum([0, *geometry.plane_cells])
    return [slice(int(a), int(b)) for a, b in itertools.pairwise(edges)]


def _top_centre_km(nodes: NodeArray, occupied: CellMask) -> np.ndarray:
    """The point halfway along the top outline of a plane's occupied cells, in plan.

    The outline runs along the top edge of each column's shallowest occupied cell, in
    column order, joined from one column to the next.
    """
    columns = np.flatnonzero(occupied.any(axis=0))
    rows = occupied[:, columns].argmax(axis=0)
    corners = np.stack([nodes[rows, columns], nodes[rows, columns + 1]], axis=1)
    outline = corners.reshape(-1, 3)[:, :2]
    steps_km = np.linalg.norm(np.diff(outline, axis=0), axis=-1)
    arc_km = np.concatenate([[0.0], np.cumsum(steps_km)])
    half_km = 0.5 * arc_km[-1]
    k = min(int(np.searchsorted(arc_km, half_km, side="right")) - 1, len(steps_km) - 1)
    fraction = (half_km - arc_km[k]) / steps_km[k] if steps_km[k] > 0 else 0.0
    return outline[k] + fraction * (outline[k + 1] - outline[k])


def _planes(
    name: str,
    geometry: Geometry,
    strike_deg: CellArray,
    hypocentre: Hypocentre | None,
    transformer: pyproj.Transformer,
) -> list[srf.Plane]:
    """One header per plane, summarising the plane's occupied cells.

    A header's ``dtop`` is the shallowest depth of an occupied cell's node, and its
    centre lies halfway along the occupied cells' top outline. Its ``dip`` is their
    mean dip and its ``stk`` their circular mean strike. ``nstk`` and
    ``ndip`` count every cell, as the points do, and ``len`` and ``wid`` are the
    chart's own extents along its top row and first column, so ``len / nstk`` and
    ``wid / ndip`` are the steps between points. In an SRF the points record the
    geometry and the plane headers only summarise it, so a header need not describe a
    rectangle.

    ``hypocentre`` is the realisation's if it lies on this segment. The plane holding
    it gets a ``shyp`` and ``dhyp`` in the chart's coordinates, along the top row from
    the plane's middle and down the first column, and every other plane gets neither.
    A reader places them in steps of ``len / nstk`` and ``wid / ndip``, so each is
    the hypocentre's fractional cell index times that step. Off the fault a curved
    chart's top row and first column step unevenly, and the distances along them
    would name another cell.

    Raises
    ------
    RuptureGeneratorError
        For a plane with no occupied cell, which has no fault to summarise.
    """
    arc_km = geometry.strike_arc_km
    width_km = float(geometry.dip_arc_km[-1])
    if hypocentre is not None:
        strike_cells = float(
            np.interp(hypocentre.strike_km, arc_km, np.arange(arc_km.size))
        )
        dip_cells = float(
            np.interp(
                hypocentre.dip_km, geometry.dip_arc_km, np.arange(geometry.cells[0] + 1)
            )
        )
    headers = []
    for number, (columns, plane) in enumerate(
        zip(_plane_slices(geometry), geometry.plane_nodes(), strict=True)
    ):
        occupied = geometry.occupied[:, columns]
        if not occupied.any():
            raise RuptureGeneratorError(
                f"{name}: plane {number} has no occupied cell, so no fault for an SRF "
                "plane header to summarise"
            )
        start_km, end_km = float(arc_km[columns.start]), float(arc_km[columns.stop])
        centre_lon, centre_lat = to_lon_lat(
            transformer, *_top_centre_km(plane, occupied)
        )
        azimuth_rad = np.radians(strike_deg[:, columns][occupied])
        mean_strike_deg = np.degrees(
            np.arctan2(np.sin(azimuth_rad).sum(), np.cos(azimuth_rad).sum())
        )
        on_plane = hypocentre is not None and start_km <= hypocentre.strike_km <= end_km
        nstk = columns.stop - columns.start
        shyp, dhyp = (
            (
                ((strike_cells - columns.start) / nstk - 0.5) * (end_km - start_km),
                dip_cells / geometry.cells[0] * width_km,
            )
            if on_plane
            else (NO_HYPOCENTRE, NO_HYPOCENTRE)
        )
        headers.append(
            srf.Plane(
                elon=float(centre_lon),
                elat=float(centre_lat),
                nstk=nstk,
                ndip=geometry.cells[0],
                len=end_km - start_km,
                wid=width_km,
                stk=float(mean_strike_deg % 360.0),
                dip=float(geometry.dip_deg[:, columns][occupied].mean()),
                dtop=float(plane[..., 2][bordering(occupied, (0, 1))].min()),
                shyp=shyp,
                dhyp=dhyp,
            )
        )
        if on_plane:
            hypocentre = None
    return headers


def _plane_points(
    name: str,
    rupture: SegmentRupture,
    columns: slice,
    strike_deg: CellArray,
    transformer: pyproj.Transformer,
) -> tuple[srf.Points, srf.SlipRate]:
    """One plane's points and slip-rate rows, in SRF order and SRF units.

    Raises
    ------
    RuptureGeneratorError
        For a slipping subfault whose rise time rounds to no samples at the segment's
        ``dt_s``, or a ``beta`` outside ``(0, 0.5]``.
    """
    geometry, dt_s, beta = rupture.geometry, rupture.dt_s, rupture.beta

    def cells(field: CellArray) -> np.ndarray:
        return np.ascontiguousarray(field[:, columns]).ravel()

    centres = geometry.centres[:, columns]
    lon, lat = to_lon_lat(transformer, centres[..., 0].ravel(), centres[..., 1].ravel())
    slip_m, rise_s = cells(rupture.slip_m), cells(rupture.rise_time_s)
    try:
        row_offsets, rates = synthesise_pulses(
            slip_m, rise_s, dt_s, None if beta is None else cells(beta)
        )
    except ValueError as error:
        raise RuptureGeneratorError(f"{name}: {error}") from error

    def single(values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)

    # The kernel leaves the pulse empty where slip is too small to resolve. The file
    # records that point as not slipping.
    slip_m = np.where(np.diff(row_offsets) > 0, slip_m, 0.0)

    points = srf.Points(
        lon=single(lon),
        lat=single(lat),
        dep=single(centres[..., 2].ravel()),
        stk=single(cells(strike_deg)),
        dip=single(cells(geometry.dip_deg)),
        area=single(cells(geometry.areas_km2) * CM2_PER_KM2),
        tinit=single(cells(rupture.onset_s)),
        dt=np.full(lon.size, dt_s, dtype=np.float32),
        rake=single(cells(rupture.rake_deg)),
        slip1=single(slip_m * CM_PER_M),
        vs=single(cells(rupture.shear_speed_km_s) * CM_PER_KM),
        density=single(cells(rupture.density_g_cm3)),
    )
    slip_rate = srf.SlipRate(
        offsets=row_offsets.astype(np.uintp), values=single(rates * CM_PER_M)
    )
    return points, slip_rate


def write_rupture(
    path: str | Path,
    realisation: Realisation[SegmentRupture],
    *,
    comments: Sequence[str] = (),
) -> None:
    """Write drawn segments, and the rock each one read, as an SRF.

    The file goes to a hidden sibling of ``path`` first and replaces ``path`` only once
    it's whole. A failed write leaves ``path`` as it was.

    Parameters
    ----------
    path : str or Path
        Where to write the file.
    realisation : Realisation of SegmentRupture
        The drawn fault system, written in its segments' order. Its projection and
        hypocentre locate the planes, and each segment's pulse sets its slip rates.
    comments : sequence of str, optional
        The header's comment lines, such as the run's record, one line each.

    Raises
    ------
    RuptureGeneratorError
        For a ``beta`` outside ``(0, 0.5]``, a slipping subfault's negative or
        non-finite rise time, a plane with no occupied cell to summarise in its
        header, or points or comments the SRF writer refuses. A rise
        time under half of ``dt_s`` is no error: that subfault slips within one
        sample.
    OSError
        If writing the file fails.
    """
    path = Path(path)
    transformer = pyproj.Transformer.from_crs(
        realisation.crs, "EPSG:4326", always_xy=True
    )
    hypocentre = realisation.hypocentre
    planes: list[srf.Plane] = []
    strikes: dict[str, CellArray] = {}
    for name, rupture in realisation.items():
        geometry = rupture.geometry
        strikes[name] = true_azimuth_deg(
            transformer, geometry.centres[..., :2], geometry.strike
        )
        on_segment = hypocentre if hypocentre and hypocentre.segment == name else None
        planes.extend(_planes(name, geometry, strikes[name], on_segment, transformer))

    partial = path.with_name(f".{path.name}.{secrets.token_hex(4)}.partial")
    try:
        with srf.Writer(partial, planes, comments=comments) as writer:
            for name, rupture in realisation.items():
                for columns in _plane_slices(rupture.geometry):
                    writer.write(
                        *_plane_points(
                            name, rupture, columns, strikes[name], transformer
                        )
                    )
        partial.replace(path)
    except srf.SrfError as error:
        raise RuptureGeneratorError(f"{path}: {error}") from error
    finally:
        partial.unlink(missing_ok=True)


@dataclasses.dataclass(frozen=True, eq=False)
class SrfRupture:
    """An SRF read back in SI: one array per point column, in file order.

    Parameters
    ----------
    planes : tuple of srf.Plane
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
    pulse_duration_s : np.ndarray
        How long each point's slip-rate pulse runs in the file, in seconds, which is
        its sample count times its ``dt``. Since an SRF has no rise time column, this
        is the nearest thing to one it holds, and a poor one: a single-sample impulse
        spans a few samples whatever the generator drew for its rise time.
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

    planes: tuple[srf.Plane, ...]
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
    pulse_duration_s: np.ndarray
    shear_speed_km_s: np.ndarray
    density_g_cm3: np.ndarray
    pulse_offsets: np.ndarray
    pulses_m_s: np.ndarray

    @property
    def rigidity_pa(self) -> np.ndarray:
        """np.ndarray: Each point's rigidity in pascals, from the file's own rock."""
        return rigidity_pa(self.shear_speed_km_s, self.density_g_cm3)


def _along(index: float, count: int) -> tuple[int, float]:
    """The point a fractional index starts from, and how far past it, extrapolating."""
    if count == 1:
        return 0, 0.0
    start = min(max(int(np.floor(index)), 0), count - 2)
    return start, index - start


def _hypocentre(plane: srf.Plane, positions: np.ndarray) -> tuple[float, float, float]:
    """The inverse of :func:`_planes`, read off the plane's own points.

    ``shyp`` and ``dhyp`` are fractions of ``len`` and ``wid`` along the plane's own
    edges, and the points are its cell centres, so the hypocentre is a bilinear
    reading of them. Within half a cell of an edge the reading extrapolates, which is
    exact on a flat plane.

    Parameters
    ----------
    plane : srf.Plane
        The header holding the hypocentre.
    positions : np.ndarray
        That plane's points, ``(ndip, nstk, 3)``: longitude, latitude and depth.
    """
    j, u = _along((plane.shyp / plane.len + 0.5) * plane.nstk - 0.5, plane.nstk)
    i, v = _along(plane.dhyp / plane.wid * plane.ndip - 0.5, plane.ndip)
    right, below = min(j + 1, plane.nstk - 1), min(i + 1, plane.ndip - 1)
    top = (1.0 - u) * positions[i, j] + u * positions[i, right]
    bottom = (1.0 - u) * positions[below, j] + u * positions[below, right]
    lon, lat, depth = (1.0 - v) * top + v * bottom
    return float(lon), float(lat), float(depth)


def read_rupture(path: str | Path) -> SrfRupture:
    """Read a version 2 SRF, as :func:`write_rupture` writes, into SI.

    Only the ``slip1`` component comes back, the only one this package writes. A
    regular file is memory-mapped. This function reads anything else, such as a pipe,
    whole.

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
    path = Path(path)
    try:
        # srf.read memory-maps a path, and refuses a pipe or a terminal as empty.
        parsed = srf.read(path if path.is_file() else path.read_bytes())
    except srf.SrfError as error:
        message = str(error)
        if not message.startswith(f"{path}:"):
            message = f"{path}: {message}"
        raise RuptureGeneratorError(message) from error
    points = parsed.points
    if points.vs is None or points.density is None:
        raise RuptureGeneratorError(
            f"{path} is a version 1 SRF, which records no shear speed or density"
        )
    pulses = parsed.slip_rate.values / np.float32(CM_PER_M)
    positions = np.column_stack([points.lon, points.lat, points.dep]).astype(np.float64)
    hypocentre = None
    for plane, run in zip(parsed.planes, parsed.plane_points, strict=True):
        if plane.shyp != NO_HYPOCENTRE:
            grid = positions[run].reshape(plane.ndip, plane.nstk, 3)
            hypocentre = _hypocentre(plane, grid)
            break
    return SrfRupture(
        planes=tuple(parsed.planes),
        hypocentre=hypocentre,
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
        pulse_duration_s=parsed.rise,
        shear_speed_km_s=points.vs.astype(np.float64) / CM_PER_KM,
        density_g_cm3=points.density.astype(np.float64),
        pulse_offsets=parsed.slip_rate.offsets.astype(np.int64),
        pulses_m_s=pulses,
    )


__all__ = ["NO_HYPOCENTRE", "SrfRupture", "read_rupture", "write_rupture"]
