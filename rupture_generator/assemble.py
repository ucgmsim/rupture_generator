"""Turning generated segments into an SRF file.

The pipeline produces physics on charts, in SI. An SRF wants flat arrays in CGS, one
PLANE record per segment, points ordered along strike fastest within each segment in
turn. This is the translation, and it is the **only** place in the package where
metres become centimetres.

# Where the plane header comes from

From the mesh, not from a reconstruction. genslip recomputes a plane centre from a
fault width and a dip with a tangent-plane approximation that is off by 43 m on a
crustal fault and 1.9 km at subduction scale; the caller here already has the exact
answer, because the mesh was built in a projected frame where the quantity is an
identity, so this asks for it rather than deriving it worse.

# The one convention conversion

An SRF's ``shyp`` is measured from the along-strike **centre** of the plane, where the
config and the mesh both measure from its ``j = 0`` end. That subtraction happens here,
at the seam that writes the format that wants it, and nowhere else.

# One PLANE per segment

genslip emits one PLANE record per segment and orders its points by segment rather
than by anything in the input geometry, so a bent or multi-segment fault is written as
several planes whose points follow in the same order. `SrfFile.planes` has always been
a list; what was missing was a caller that filled it with more than one entry.
"""

from __future__ import annotations

import numpy as np
import scipy as sp
import xarray as xr

from rupture_generator.formats.rupture import mesh_of
from rupture_generator.srf import PlaneHeader, Points, SrfFile
from rupture_generator.units import CM2_PER_M2, CM_PER_KM, CM_PER_M, SRF_FLOAT


def plane_header(
    segment: xr.Dataset, *, hypocentre_km: tuple[float, float] | None
) -> PlaneHeader:
    """The PLANE record for one segment, read off its own geometry.

    Parameters
    ----------
    segment : xr.Dataset
        A rupture-file segment.
    hypocentre_km : tuple of float, optional
        Where the rupture started, in this segment's own arc lengths, or ``None`` for
        a segment that does not hold it. The format has no way to say "not here", so
        a segment without the hypocentre records zeros -- which is what genslip does,
        and what a reader of a multi-plane SRF has to know already.

    Returns
    -------
    PlaneHeader
    """
    mesh = mesh_of(segment)
    length_km = float(mesh.strike_arc_km()[-1])
    width_km = float(mesh.dip_arc_km()[-1])
    cells_i, cells_j = mesh.cell_counts

    strike_km, dip_km = hypocentre_km or (0.0, 0.0)

    return PlaneHeader(
        centre_longitude_deg=float(segment["centre_longitude_deg"].mean()),
        centre_latitude_deg=float(segment["centre_latitude_deg"].mean()),
        strike_count=cells_j,
        dip_count=cells_i,
        length_km=length_km,
        width_km=width_km,
        strike_deg=float(segment["strike_deg"].to_numpy()[0, 0]),
        dip_deg=float(segment["dip_deg"].to_numpy()[0, 0]),
        top_depth_km=float(segment["node_depth_km"].min()),
        # The SRF measures the hypocentre from the plane's along-strike centre; the
        # config and the mesh measure from its j = 0 end.
        hypocentre_strike_km=strike_km - length_km / 2.0 if hypocentre_km else 0.0,
        hypocentre_dip_km=dip_km,
    )


def to_srf_file(
    segments: list[xr.Dataset],
    shear_speeds_km_s: list[np.ndarray],
    densities_g_cm3: list[np.ndarray],
) -> SrfFile:
    """Assemble an SRF version 2.0 file from generated segments.

    Parameters
    ----------
    segments : list of xr.Dataset
        The rupture-file segments, in the order their planes should appear.
    shear_speeds_km_s, densities_g_cm3 : list of np.ndarray
        Per segment, one value per subfault, in the units a velocity model is written
        in. Version 2.0 carries both per point -- shear speed in centimetres per
        second, which this converts.

    Returns
    -------
    SrfFile
        Version 2.0, one PLANE record per segment.

    Raises
    ------
    ValueError
        If a segment's material arrays do not describe its subfaults.
    """
    headers: list[PlaneHeader] = []
    columns: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "longitude_deg",
            "latitude_deg",
            "depth_km",
            "strike_deg",
            "dip_deg",
            "area_cm2",
            "onset_s",
            "sample_interval_s",
            "rake_deg",
            "slip_cm",
            "rise_time_s",
            "shear_speed_cm_s",
            "density_g_cm3",
        )
    }
    pulse_lengths: list[np.ndarray] = []
    pulse_samples: list[np.ndarray] = []

    for segment, shear_speed_km_s, density_g_cm3 in zip(
        segments, shear_speeds_km_s, densities_g_cm3, strict=True
    ):
        subfaults = segment.sizes["i"] * segment.sizes["j"]
        for name, values in (
            ("shear_speed_km_s", shear_speed_km_s),
            ("density_g_cm3", density_g_cm3),
        ):
            if np.asarray(values).size != subfaults:
                raise ValueError(
                    f"{name} has {np.asarray(values).size} entries for {subfaults} "
                    f"subfaults on segment {segment.attrs.get('surface')}"
                )

        hypocentre_km = (
            (
                float(segment.attrs["hypocentre_strike_km"]),
                float(segment.attrs["hypocentre_dip_km"]),
            )
            if "hypocentre_strike_km" in segment.attrs
            else None
        )
        headers.append(plane_header(segment, hypocentre_km=hypocentre_km))

        def flat(name: str, dataset: xr.Dataset = segment) -> np.ndarray:
            return dataset[name].to_numpy().ravel()

        interval_s = float(segment.attrs["sample_interval_s"])

        # SI leaves the package here. Slip and area cross into the format's own
        # units; depth, angles and times are already what it wants.
        columns["longitude_deg"].append(flat("centre_longitude_deg"))
        columns["latitude_deg"].append(flat("centre_latitude_deg"))
        columns["depth_km"].append(flat("centre_depth_km"))
        columns["strike_deg"].append(flat("strike_deg"))
        columns["dip_deg"].append(flat("dip_deg"))
        columns["area_cm2"].append(flat("area_m2") * CM2_PER_M2)
        columns["onset_s"].append(flat("onset_s"))
        columns["sample_interval_s"].append(np.full(subfaults, interval_s))
        columns["rake_deg"].append(flat("rake_deg"))
        columns["slip_cm"].append(flat("slip_m") * CM_PER_M)
        columns["rise_time_s"].append(flat("rise_time_s"))
        columns["shear_speed_cm_s"].append(
            np.asarray(shear_speed_km_s, dtype=np.float64).ravel() * CM_PER_KM
        )
        columns["density_g_cm3"].append(
            np.asarray(density_g_cm3, dtype=np.float64).ravel()
        )

        offsets = segment["slip_rate_offset"].to_numpy()
        pulse_lengths.append(np.diff(offsets))
        pulse_samples.append(segment["slip_rate"].to_numpy() * CM_PER_M)

    points = Points(
        **{
            name: np.concatenate(values).astype(SRF_FLOAT)
            for name, values in columns.items()
        }
    )

    # The offsets are rebuilt across segments rather than concatenated: each
    # segment's own start at zero, and the SRF's points are one run.
    lengths = np.concatenate(pulse_lengths) if pulse_lengths else np.empty(0, np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    samples = (
        np.concatenate(pulse_samples) if pulse_samples else np.empty(0, np.float64)
    )
    longest = int(lengths.max()) if len(lengths) else 0
    # A sample's column is its position within its own pulse, which is what makes the
    # ragged set of pulses a CSR matrix as wide as the longest of them.
    within = (
        np.concatenate([np.arange(length, dtype=np.int64) for length in lengths])
        if longest
        else np.empty(0, dtype=np.int64)
    )

    return SrfFile(
        version="2.0",
        planes=headers,
        points=points,
        slip_rate=sp.sparse.csr_array(
            (samples.astype(SRF_FLOAT), within, offsets),
            shape=(len(points.longitude_deg), longest),
        ),
    )


__all__ = ["plane_header", "to_srf_file"]
