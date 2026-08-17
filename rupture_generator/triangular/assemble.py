"""Turning a triangulated rupture into an SRF file: one point per triangle.

A triangulation has no ``NSTK x NDIP`` layout, so each segment becomes one PLANE with
``NSTK = n_triangles`` and ``NDIP = 1``, and every triangle carries its own longitude,
latitude, depth, strike, dip, rake, area and slip as a point. The header's geometry is
a summary, not a construction.

That is safe for SW4: both readers `sscanf` the plane fields, print them and discard
them (`sw4/src/parseInputFile.C:6249-6261`, `sw4/src/readhdf5.C:1066`), the point count
comes from the ``POINTS`` line alone, and there is no ``nstk * ndip == npts`` check. It
is not inert to everything else -- any consumer that rebuilds geometry from the header
gets nonsense, including this package's own `scripts/view.py`.

HDF5 is the chosen path. The text reader ignores ``VS`` and ``DEN`` and takes ``mu``
from the SW4 grid where the HDF5 reader uses the file's own; this module fills them
from the materials stage, so the moment the file states is the moment the pipeline
scaled to. HDF5 also avoids the text path's low-slip filter, which drops any point
whose ``dt sum(sdot)`` is under 1e-4 m -- exactly what an edge taper produces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pyproj

from rupture_generator import pulses
from rupture_generator.assemble import POINT_COLUMNS, srf_file
from rupture_generator.mesh import WGS84, grid_convergence_deg
from rupture_generator.realisation import (
    HYPOCENTRE_DIP_KM,
    HYPOCENTRE_STRIKE_KM,
    Realisation,
)
from rupture_generator.srf import PlaneHeader, Points, SrfFile, Sw4Hdf5Stream
from rupture_generator.triangular.pipeline import (
    STREAM_BUDGET_BYTES,
    SegmentGeometry,
    face_blocks,
)
from rupture_generator.units import (
    CM2_PER_M2,
    CM_PER_KM,
    CM_PER_M,
    M2_PER_KM2,
    M_PER_KM,
    SRF_FLOAT,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from rupture_generator.triangular.mesh import TriangleMesh

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]


def project_faces(
    mesh: TriangleMesh, geometry: SegmentGeometry, crs: pyproj.CRS
) -> dict[str, FloatArray]:
    """Face centres in WGS84, with true-north strike and dip, one value per face.

    The triangular counterpart of `mesh.project_cells`: the origin is added back, and
    strike crosses from grid north to true north with the grid convergence evaluated
    **per face**. Dip crosses unchanged, being an angle within the surface.

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

    ``NSTK`` is the segment's triangle count and ``NDIP`` is 1, which states the point
    block's length without claiming a lattice. Everything else summarises the segment:
    the centre is the mean of the face centres, the length and width are its own arc
    extents, the strike and dip are the reference frame's, and the top depth is the
    shallowest node.

    ``shyp`` keeps `assemble.plane_header`'s one convention conversion: the SRF
    measures the hypocentre from the plane's along-strike **centre**, where the config
    and the mesh measure from the ``u = 0`` end. ``hypocentre_km`` is ``None`` for a
    segment that does not hold it, which the format can only record as zeros.
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
    The unit conversions and the CSR assembly are `assemble.srf_file`'s, unchanged.
    ``geometries`` is derived here if omitted, at one pass over the faces per segment --
    pass the pipeline's own to avoid it.

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


def write_sw4_hdf5(
    path: Path | str,
    realisation: Realisation,
    params: pulses.PulseParams,
    geometries: Mapping[str, SegmentGeometry] | None = None,
    *,
    budget_bytes: int = STREAM_BUDGET_BYTES,
) -> None:
    """Write a generated triangular rupture straight out as an SRF-HDF5 file.

    Pulse synthesis and writing are fused: attaching every pulse of every subfault to
    the mesh first is 15 GB of ``f64`` at a 400 m cut on the CFM Hikurangi interface,
    so the pulses are synthesised here a block of faces at a time, converted, appended
    and dropped. :data:`STREAM_BUDGET_BYTES` bounds the peak and :func:`face_blocks` is
    where a block comes from; a rupture that fits in memory goes through
    :func:`to_srf_file` instead, and the two share `srf.Sw4Hdf5Stream`.

    Pulses already attached to the realisation are **not** read; the kernel is
    deterministic in its inputs, so what is written is what ``synthesise_pulses`` would
    have attached. Run the pipeline with ``synthesise=False`` and nothing is computed
    twice.

    Parameters
    ----------
    path : Path or str
        Where to write. Overwritten.
    realisation : Realisation
        A rupture that has been through
        :func:`~rupture_generator.triangular.pipeline.generate`, with or without its
        pulses.
    params : pulses.PulseParams
        How each pulse is shaped and sampled. Its ``sample_interval_s`` is the file's
        ``dt``.
    geometries : Mapping of str to SegmentGeometry, optional
        The hoisted geometry of each segment. Derived here if omitted.
    budget_bytes : int, optional
        See :data:`STREAM_BUDGET_BYTES`.

    Raises
    ------
    KeyError
        If a segment is missing a field the format needs, which is a realisation that
        has not been all the way through the pipeline.
    ValueError
        For a subfault whose rise time the sample interval cannot represent. The
        message names the segment and the block as well as the subfault, because the
        kernel numbers subfaults within the block it was handed.
    """
    with Sw4Hdf5Stream(path, "2.0") as stream:
        for name, mesh in realisation.items():
            geometry = (
                SegmentGeometry.of(mesh) if geometries is None else geometries[name]
            )
            located = project_faces(mesh, geometry, realisation.crs)

            hypocentre_km = (
                (
                    float(mesh.attrs[HYPOCENTRE_STRIKE_KM]),
                    float(mesh.attrs[HYPOCENTRE_DIP_KM]),
                )
                if HYPOCENTRE_STRIKE_KM in mesh.attrs
                else None
            )
            stream.plane(
                plane_header(mesh, geometry, located, hypocentre_km=hypocentre_km)
            )

            # SI leaves the package here, for the whole segment at once: the point
            # columns are one float per face, so only the pulses need blocking.
            depth_km = geometry.depth_km
            slip_m = mesh["slip_m"]
            rise_time_s = mesh["rise_time_s"]
            columns = {
                "longitude_deg": located["longitude_deg"],
                "latitude_deg": located["latitude_deg"],
                "depth_km": located["depth_km"],
                "strike_deg": located["strike_deg"],
                "dip_deg": located["dip_deg"],
                "area_cm2": geometry.areas_km2 * M2_PER_KM2 * CM2_PER_M2,
                "onset_s": mesh["onset_s"],
                "sample_interval_s": np.full(
                    geometry.face_count, params.sample_interval_s
                ),
                "rake_deg": mesh["rake_deg"],
                "slip_cm": slip_m * CM_PER_M,
                "rise_time_s": rise_time_s,
                "shear_speed_cm_s": mesh["shear_speed_kms"] * CM_PER_KM,
                "density_g_cm3": mesh["density_g_cm3"],
            }

            # Twelve bytes a live sample: the kernel's own f64, and the float32 it
            # is narrowed into for the file.
            for block in face_blocks(
                rise_time_s,
                params,
                budget_bytes,
                8 + np.dtype(SRF_FLOAT).itemsize,
            ):
                try:
                    offsets, samples = pulses.synthesise(
                        depth_km[block], slip_m[block], rise_time_s[block], params
                    )
                except ValueError as error:
                    raise ValueError(
                        f"synthesising the pulses of segment {name!r} over faces "
                        f"{block.start} to {block.stop}, where the kernel numbers "
                        f"subfaults from {block.start}: {error}"
                    ) from error

                # Converted into its final buffer rather than concatenated into one, so
                # only the destination is live beside the kernel's own output.
                block_samples = np.empty(samples.size, dtype=SRF_FLOAT)
                np.multiply(samples, CM_PER_M, out=block_samples, casting="unsafe")
                del samples

                stream.points(
                    Points(
                        **{
                            column: np.asarray(values[block], dtype=SRF_FLOAT)
                            for column, values in columns.items()
                        }
                    ),
                    np.diff(offsets),
                    block_samples,
                )


def hdf5_moment_newton_m(path: Path | str) -> float:
    """The moment an SRF-HDF5 file states, summed from its own ``POINTS`` columns.

    :func:`moment_newton_m` for a file too large to read: the same arithmetic on the
    same four columns, read a slab at a time off disk, and never touching ``SR1`` --
    which carries no moment, since the kernel normalises every pulse so that
    ``dt sum(sdot)`` is the point's own slip. It catches a streaming writer dropping or
    duplicating a block of points.

    Returns
    -------
    float
        Newton-metres.

    Raises
    ------
    ValueError
        For a version 1.0 file, whose points carry no material properties, so the
        moment depends on the velocity model it is run against rather than on the file.
    """
    with h5py.File(path, "r") as file:
        if float(file.attrs["VERSION"]) < 2.0:
            raise ValueError(
                f"{path} declares SRF version {float(file.attrs['VERSION'])}, whose "
                "points carry no shear speed or density, so the file states no "
                "rigidity and no moment"
            )
        points = file["POINTS"]
        moment_dyne_cm = 0.0
        # A slab at a time: a million records of `SW4_POINTS_DTYPE` is 68 MB, large
        # enough that the per-read overhead does not show.
        for start in range(0, points.shape[0], 1 << 20):
            slab = points[start : start + (1 << 20)]
            moment_dyne_cm += float(
                np.sum(
                    slab["DEN"].astype(np.float64)
                    * slab["VS"].astype(np.float64) ** 2
                    * slab["AREA"].astype(np.float64)
                    * slab["SLIP1"].astype(np.float64)
                )
            )
    return moment_dyne_cm * 1.0e-7


def moment_newton_m(srf: SrfFile) -> float:
    """The moment an SRF file states, summed from its own columns.

    Not the moment the pipeline scaled to -- **the one the file carries**, so that it
    can be compared against SW4's printed ``made %i point moment tensor sources``
    tally, the only check that catches SW4's low-slip filter eating tapered edges.

    In CGS, rigidity is ``rho vs^2`` in dyne per square centimetre and a dyne-centimetre
    is 1e-7 newton-metres; the columns are float32, so this reads them in float64.

    Returns
    -------
    float
        Newton-metres.

    Raises
    ------
    ValueError
        If the points carry no material properties, which is a version 1.0 file, where
        the shear modulus comes from whatever grid the file is run against.
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


__all__ = [
    "hdf5_moment_newton_m",
    "moment_newton_m",
    "plane_header",
    "project_faces",
    "to_srf_file",
    "write_sw4_hdf5",
]
