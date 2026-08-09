"""Leaving the projected frame: where a mesh becomes longitude and latitude.

`genslip::mesh` works in a Cartesian coordinate reference system the modeller named --
NZTM2000, a UTM zone -- and never leaves it. That is what makes every quantity it
derives an exact identity rather than an approximation carrying a curvature error. This
module is the one seam where the projection is undone, and it is the only place in the
package that imports `pyproj`.

Two things cross, and only one of them is a plain transformation.

Positions
    A `Transformer` and nothing else. Kilometres in, metres to the projection,
    longitude and latitude out.

Strike
    **Plus the grid convergence angle.** Grid north is not true north: a projection's
    northing axis points along the *central meridian*, and away from it the two diverge
    by the convergence angle, which is what `Proj.get_factors(...).meridian_convergence`
    reports. `MESH.md` has how large that gets in NZTM2000, and why it matters against
    the rake bound in `ENGINEERING_RULES.md`. A strike written without it is wrong by
    more than the format can express.

Dip and rake cross unchanged. Both are angles *within* the fault plane, measured from
directions that rotate together with it, so the convergence cancels.

The sign convention is `true = grid + convergence`, and it is asserted rather than
assumed: `tests/test_projection.py` steps ten kilometres along grid north, asks
`pyproj.Geod` what azimuth that actually was, and compares. Getting it backwards is a
two-degree error in the middle of the country and a ten-degree error across it, which
is exactly the kind of thing that looks plausible on a map.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import pyproj

from rupture_generator._core import RefinedMesh
from rupture_generator.units import CM2_PER_KM2, M_PER_KM

if TYPE_CHECKING:
    from rupture_generator.assemble import SubfaultGeometry

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

WGS84 = pyproj.CRS("EPSG:4326")
"""What an SRF's coordinates are, and what everything downstream of one expects."""


@dataclasses.dataclass(frozen=True)
class Located:
    """One patch of a mesh, in longitude and latitude.

    Every array is cell-centred and shaped ``(dip, strike)`` -- the order
    ``crates/genslip/src/grid.rs`` fixes, strike varying fastest -- except that
    ``to_subfault_geometry`` flattens them, because that is what an SRF stores.

    Attributes
    ----------
    longitude_deg, latitude_deg : FloatArray
        Where each subfault's centre is, on WGS84.
    depth_km : FloatArray
        How deep its centre is. Unchanged by the projection.
    strike_deg : FloatArray
        Clockwise from **true** north: the mesh's grid strike plus the convergence
        angle at that subfault.
    dip_deg : FloatArray
        Below horizontal. Unchanged by the projection.
    area_km2 : FloatArray
        Unchanged by the projection, which is a statement about the projection rather
        than about this code -- see the note in ``project_patch``.
    """

    longitude_deg: FloatArray
    latitude_deg: FloatArray
    depth_km: FloatArray
    strike_deg: FloatArray
    dip_deg: FloatArray
    area_km2: FloatArray


def grid_convergence_deg(
    crs: pyproj.CRS, longitude_deg: FloatArray, latitude_deg: FloatArray
) -> FloatArray:
    """The angle from true north to grid north, in degrees, at each point.

    Add it to a grid azimuth to get a true one. Zero on the projection's central
    meridian and growing away from it -- in NZTM2000, whose central meridian is 173
    degrees east, it runs from about -3.4 degrees at East Cape to +5.0 in Fiordland.

    Parameters
    ----------
    crs : pyproj.CRS
        The projected CRS the grid azimuth was measured in.
    longitude_deg, latitude_deg : FloatArray
        Where to evaluate it. The convergence varies across a fault, so this is
        per subfault rather than one number for the mesh.

    Returns
    -------
    FloatArray
        Degrees, the same shape as the inputs.
    """
    factors = pyproj.Proj(crs).get_factors(longitude_deg, latitude_deg)
    return np.asarray(factors.meridian_convergence, dtype=np.float64)


def project_patch(mesh: RefinedMesh, patch: int, crs: pyproj.CRS) -> Located:
    """Turn one patch of a mesh into longitude, latitude and true-north angles.

    Parameters
    ----------
    mesh : RefinedMesh
        The mesh, whose positions are offsets from ``mesh.origin`` in kilometres.
    patch : int
        Which patch -- one per plane of the fault.
    crs : pyproj.CRS
        The projected CRS the mesh was built in.

    Returns
    -------
    Located
        Cell-centred arrays, shaped ``(dip, strike)``.

    Notes
    -----
    The origin is added back here and nowhere else -- the one point where the large
    number has to appear, and it appears once. ``RefinedMesh`` holds offsets rather than
    absolute positions for the reason its own docstring gives.

    Area is taken from the mesh unchanged. A projection does distort area -- NZTM's
    scale factor is 0.9996 on its central meridian -- but the distortion applies to the
    *coordinates*, and the fault's true area is the one the modeller specified in the
    CRS they chose for that region. Correcting it here would be applying a second
    opinion about a number nobody asked to be reinterpreted.
    """
    east_km, north_km, _ = mesh.cell_centres(patch)
    origin = mesh.origin
    to_wgs84 = pyproj.Transformer.from_crs(crs, WGS84, always_xy=True)

    longitude_deg, latitude_deg = to_wgs84.transform(
        (origin.easting_km + east_km) * M_PER_KM,
        (origin.northing_km + north_km) * M_PER_KM,
    )
    longitude_deg = np.asarray(longitude_deg, dtype=np.float64)
    latitude_deg = np.asarray(latitude_deg, dtype=np.float64)

    grid_strike_deg = mesh.strike_deg(patch)
    true_strike_deg = np.mod(
        grid_strike_deg + grid_convergence_deg(crs, longitude_deg, latitude_deg),
        360.0,
    )

    return Located(
        longitude_deg=longitude_deg,
        latitude_deg=latitude_deg,
        depth_km=mesh.cell_centres(patch)[2],
        strike_deg=true_strike_deg,
        dip_deg=mesh.dip_deg(patch),
        area_km2=mesh.areas_km2(patch),
    )


def to_projected(
    crs: pyproj.CRS, longitude_deg: float, latitude_deg: float
) -> tuple[float, float]:
    """A longitude and latitude as an easting and northing, in **kilometres**.

    The way in. A trace is digitised in longitude and latitude, and the mesh is built in
    the CRS, so this runs once per trace point on the way to
    :class:`~rupture_generator._core.Fault`.

    Returns
    -------
    tuple of float
        Easting and northing, in kilometres.
    """
    easting_m, northing_m = pyproj.Transformer.from_crs(
        WGS84, crs, always_xy=True
    ).transform(longitude_deg, latitude_deg)
    return easting_m / M_PER_KM, northing_m / M_PER_KM


def to_subfault_geometry(
    mesh: RefinedMesh, patch: int, crs: pyproj.CRS
) -> SubfaultGeometry:
    """One patch as the arrays ``assemble.to_srf_file`` has always asked for.

    The seam this whole module exists to reach. ``assemble.py``'s docstring has said
    since it was written that *"the subfault coordinates arrive in ``SubfaultGeometry``,
    from whoever discretised the fault, because that is the only place that knows how
    the mesh became a grid"*. This is that place.

    Flattened along-strike-fastest, which is the order the core produces every field in
    and the order an SRF stores points in.

    Returns
    -------
    SubfaultGeometry
        One value per subfault, with ``area_cm2`` in the square centimetres the format
        stores and the moment sum is expressed in.
    """
    from rupture_generator.assemble import SubfaultGeometry

    located = project_patch(mesh, patch, crs)
    return SubfaultGeometry(
        longitude_deg=located.longitude_deg.ravel(),
        latitude_deg=located.latitude_deg.ravel(),
        depth_km=located.depth_km.ravel(),
        strike_deg=located.strike_deg.ravel(),
        dip_deg=located.dip_deg.ravel(),
        area_cm2=located.area_km2.ravel() * CM2_PER_KM2,
    )
