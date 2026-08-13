"""Turning a triangulated rupture into an SRF file: one point per triangle.

`assemble.py` writes a lattice segment as a PLANE whose ``NSTK x NDIP`` says how its
points are laid out. A triangulation has no such layout, so this writes **one PLANE per
segment with ``NSTK = n_triangles`` and ``NDIP = 1``**, and every triangle carries its
own longitude, latitude, depth, strike, dip, rake, area and slip as a point. The header's
geometry is a summary rather than a construction, and nothing reconstructs a subfault
from it.

**That is safe for SW4, and the reason is in SW4's source rather than in this docstring's
confidence.** Both readers `sscanf` the plane fields, print them and discard them: in the
text reader the plane variables are scoped to the loop body and die at its closing brace
(`sw4/src/parseInputFile.C:6249-6261`), and the HDF5 reader `free`s the metadata on the
line before it opens the ``POINTS`` dataset (`sw4/src/readhdf5.C:1066`). The point count
comes from the ``POINTS`` line alone and there is no ``nstk * ndip == npts`` check
anywhere. So the header is geometrically inert to SW4.

**It is not inert to everything else**, and that is the caveat rather than a detail. Any
consumer that rebuilds geometry from the header gets nonsense from this file --
including this package's own `scripts/view.py`, which strides quads out of
``strike_count`` and ``dip_count``. A viewer for a triangular rupture has to read the
mesh file, where the faces are; MESH.md's phase 3 is where that happens.

**HDF5 is the chosen path**, which settles a question the text path leaves open: the two
disagree about the shear modulus, because the text reader ignores ``VS`` and ``DEN`` and
takes ``mu`` from the SW4 grid while the HDF5 reader uses the file's own. This module
fills ``VS`` and ``DEN`` from the materials stage, so the moment the file states is the
moment the pipeline scaled to -- and :func:`moment_newton_m` computes it from the file's
own columns, in the file's own units, so it can be checked against SW4's printed
``made %i point moment tensor sources`` tally. Choosing HDF5 also removes the low-slip
filter as a constraint: a single-precision text build drops any point whose
``dt sum(sdot)`` is under 1e-4 m and warns only ten times, which is exactly what an edge
taper produces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyproj

from rupture_generator.assemble import POINT_COLUMNS, srf_file
from rupture_generator.mesh import WGS84, grid_convergence_deg
from rupture_generator.realisation import (
    HYPOCENTRE_DIP_KM,
    HYPOCENTRE_STRIKE_KM,
    Realisation,
)
from rupture_generator.srf import PlaneHeader, SrfFile
from rupture_generator.triangular.pipeline import SegmentGeometry
from rupture_generator.units import (
    CM2_PER_M2,
    CM_PER_KM,
    CM_PER_M,
    M2_PER_KM2,
    M_PER_KM,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rupture_generator.triangular.mesh import TriangleMesh

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


def project_faces(
    mesh: TriangleMesh, geometry: SegmentGeometry, crs: pyproj.CRS
) -> dict[str, FloatArray]:
    """Face centres in WGS84, with true-north strike and dip, one value per face.

    The triangular counterpart of `mesh.project_cells`, and the same two crossings: the
    origin is added back, and strike crosses from grid north to true north with the grid
    convergence evaluated **per face** rather than once for the segment. Dip and area
    cross unchanged -- dip is an angle within the surface, and the fault's true area is
    the one the modeller specified in the CRS they chose.

    Parameters
    ----------
    mesh : TriangleMesh
        The segment, for its strike and dip per face.
    geometry : SegmentGeometry
        Its hoisted geometry, for the centres.
    crs : pyproj.CRS
        The projected frame the positions are in.

    Returns
    -------
    dict of str to FloatArray
        ``longitude_deg``, ``latitude_deg``, ``depth_km``, ``strike_deg``, ``dip_deg``,
        each ``(F,)``.
    """
    centres_km = geometry.centres_km
    origin_east_km, origin_north_km = mesh.origin_km
    to_wgs84 = pyproj.Transformer.from_crs(crs, WGS84, always_xy=True)
    longitude_deg, latitude_deg = to_wgs84.transform(
        (origin_east_km + centres_km[:, 0]) * M_PER_KM,
        (origin_north_km + centres_km[:, 1]) * M_PER_KM,
    )
    longitude_deg = np.asarray(longitude_deg, dtype=np.float64)
    latitude_deg = np.asarray(latitude_deg, dtype=np.float64)

    grid_strike_deg, dip_deg = mesh.strike_dip_deg()
    return {
        "longitude_deg": longitude_deg,
        "latitude_deg": latitude_deg,
        "depth_km": centres_km[:, 2],
        "strike_deg": np.mod(
            grid_strike_deg + grid_convergence_deg(crs, longitude_deg, latitude_deg),
            360.0,
        ),
        "dip_deg": dip_deg,
    }


def plane_header(
    mesh: TriangleMesh,
    geometry: SegmentGeometry,
    located: Mapping[str, FloatArray],
    *,
    hypocentre_km: tuple[float, float] | None,
) -> PlaneHeader:
    """The PLANE record for one triangulated segment.

    ``NSTK`` is the segment's triangle count and ``NDIP`` is 1, which is how the point
    block's length is stated without claiming a lattice. Everything else is a **summary
    of the segment**, not a construction of it: the centre is the mean of the face
    centres, the length and width are the segment's own arc extents, the strike and dip
    are the reference frame's, and the top depth is the shallowest node. A reader that
    treats those as a rectangle gets a rectangle that does not exist -- see this module's
    docstring for why SW4 does not, and who does.

    ``shyp`` keeps the one convention conversion `assemble.plane_header` makes: the SRF
    measures the hypocentre from the plane's along-strike **centre**, where the config
    and the mesh measure from the ``u = 0`` end.

    Parameters
    ----------
    mesh : TriangleMesh
        The segment.
    geometry : SegmentGeometry
        Its hoisted geometry.
    located : Mapping of str to FloatArray
        :func:`project_faces`' output, passed in because the projection is the expensive
        part and the point columns want it too.
    hypocentre_km : tuple of float, optional
        Where the rupture started, in this segment's own arc lengths, or ``None`` for a
        segment that does not hold it. The format has no way to say "not here", so a
        segment without the hypocentre records zeros.

    Returns
    -------
    PlaneHeader
    """
    length_km = float(mesh.arc_profile(0)[1][-1])
    width_km = float(mesh.arc_profile(1)[1][-1])
    frame = mesh.frame
    strike_km, dip_km = hypocentre_km or (0.0, 0.0)

    return PlaneHeader(
        centre_longitude_deg=float(located["longitude_deg"].mean()),
        centre_latitude_deg=float(located["latitude_deg"].mean()),
        strike_count=geometry.face_count,
        dip_count=1,
        length_km=length_km,
        width_km=width_km,
        strike_deg=frame.strike_deg,
        dip_deg=frame.dip_deg,
        top_depth_km=float(geometry.vertices_km[:, 2].min()),
        hypocentre_strike_km=strike_km - length_km / 2.0 if hypocentre_km else 0.0,
        hypocentre_dip_km=dip_km,
    )


def to_srf_file(
    realisation: Realisation,
    geometries: Mapping[str, SegmentGeometry] | None = None,
) -> SrfFile:
    """Assemble an SRF version 2.0 file from a generated triangular rupture.

    One PLANE per segment in the realisation's own order, and one point per triangle.
    The unit conversions and the CSR assembly are `assemble.srf_file`'s, unchanged: this
    module's whole job is the per-face columns and the header, because everything after
    that is the same file.

    Parameters
    ----------
    realisation : Realisation
        A rupture that has been all the way through
        :func:`rupture_generator.triangular.pipeline.generate`.
    geometries : Mapping of str to SegmentGeometry, optional
        The hoisted geometry of each segment. Derived here if omitted, which costs one
        pass over the faces per segment -- pass the pipeline's own to avoid it.

    Returns
    -------
    SrfFile
        Version 2.0, one PLANE record per segment.

    Raises
    ------
    KeyError
        If a segment is missing a field the format needs, which is a realisation that
        has not been all the way through the pipeline.
    """
    headers: list[PlaneHeader] = []
    columns: dict[str, list[np.ndarray]] = {name: [] for name in POINT_COLUMNS}
    pulse_lengths: list[np.ndarray] = []
    samples_of: list[np.ndarray] = []

    for name, mesh in realisation.items():
        geometry = SegmentGeometry.of(mesh) if geometries is None else geometries[name]
        located = project_faces(mesh, geometry, realisation.crs)

        hypocentre_km = (
            (
                float(mesh.attrs[HYPOCENTRE_STRIKE_KM]),
                float(mesh.attrs[HYPOCENTRE_DIP_KM]),
            )
            if HYPOCENTRE_STRIKE_KM in mesh.attrs
            else None
        )
        headers.append(
            plane_header(mesh, geometry, located, hypocentre_km=hypocentre_km)
        )

        interval_s = float(mesh.attrs["sample_interval_s"])

        # SI leaves the package here, in the one place it does: slip and area cross into
        # the format's own units, and depth, angles and times are already what it wants.
        columns["longitude_deg"].append(located["longitude_deg"])
        columns["latitude_deg"].append(located["latitude_deg"])
        columns["depth_km"].append(located["depth_km"])
        columns["strike_deg"].append(located["strike_deg"])
        columns["dip_deg"].append(located["dip_deg"])
        columns["area_cm2"].append(geometry.areas_km2 * M2_PER_KM2 * CM2_PER_M2)
        columns["onset_s"].append(mesh["onset_s"])
        columns["sample_interval_s"].append(np.full(geometry.face_count, interval_s))
        columns["rake_deg"].append(mesh["rake_deg"])
        columns["slip_cm"].append(mesh["slip_m"] * CM_PER_M)
        columns["rise_time_s"].append(mesh["rise_time_s"])
        columns["shear_speed_cm_s"].append(mesh["shear_speed_kms"] * CM_PER_KM)
        columns["density_g_cm3"].append(mesh["density_g_cm3"])

        offsets, samples = mesh.pulses  # ty: ignore[not-iterable]
        pulse_lengths.append(np.diff(offsets))
        samples_of.append(samples)

    return srf_file(headers, columns, pulse_lengths, samples_of)


def moment_newton_m(srf: SrfFile) -> float:
    """The moment an SRF file states, summed from its own columns.

    Not the moment the pipeline scaled to -- **the one the file carries**, in the units
    it carries it in, so that it can be compared against what a consumer computes.
    MESH.md asks for exactly this comparison against SW4's printed
    ``made %i point moment tensor sources`` tally, because that is the only check that
    catches SW4's low-slip filter quietly eating tapered edges.

    In CGS, rigidity is ``rho vs^2`` in dyne per square centimetre and a dyne-centimetre
    is 1e-7 newton-metres; the columns are float32, so this reads them in float64 and
    the answer is good to about six figures rather than to round-off.

    Parameters
    ----------
    srf : SrfFile
        A version 2.0 file, whose points carry ``VS`` and ``DEN``.

    Returns
    -------
    float
        Newton-metres.

    Raises
    ------
    ValueError
        If the points carry no material properties, which is a version 1.0 file, where
        the shear modulus comes from whatever grid the file is run against and the
        question has no answer here.
    """
    points = srf.points
    if points.shear_speed_cm_s is None or points.density_g_cm3 is None:
        raise ValueError(
            "these points carry no shear speed or density, so the file states no "
            "rigidity and no moment. That is a version 1.0 SRF, whose moment depends on "
            "the velocity model it is run against rather than on the file"
        )
    rigidity_dyne_cm2 = points.density_g_cm3.astype(np.float64) * (
        points.shear_speed_cm_s.astype(np.float64) ** 2
    )
    moment_dyne_cm = float(
        np.sum(
            rigidity_dyne_cm2
            * points.area_cm2.astype(np.float64)
            * points.slip_cm.astype(np.float64)
        )
    )
    return moment_dyne_cm * 1.0e-7


__all__ = ["moment_newton_m", "plane_header", "project_faces", "to_srf_file"]
