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

        pulse_lengths.append(np.diff(segment["slip_rate_offset"].to_numpy()))

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
    longest = int(lengths.max()) if len(lengths) else 0

    # **The samples are converted into their final buffer, not concatenated into it.**
    # `np.concatenate` of one scaled array per segment holds the pieces and the result
    # at once, and both in float64 -- four copies of a quantity the format stores as
    # float32. On the shipped twenty-fault scenario that is 944 million samples, so
    # each copy is 7.6 GB and the peak was over 20 GB for a 3.8 GB answer.
    #
    # `np.multiply` with `out=` does the unit conversion and the narrowing in one pass,
    # straight into the slice that segment owns, so only the destination is ever live.
    samples = np.empty(int(offsets[-1]), dtype=SRF_FLOAT)
    at = 0
    for segment in segments:
        source = segment["slip_rate"].to_numpy()
        np.multiply(
            source, CM_PER_M, out=samples[at : at + source.size], casting="unsafe"
        )
        at += source.size

    # A sample's column is its position within its own pulse, which is what makes the
    # ragged set of pulses a CSR matrix as wide as the longest of them.
    #
    # **Built as a cumulative sum in a single buffer**, because the obvious spellings
    # are not single-buffer. `concatenate([arange(n) for n in lengths])` builds an array
    # object per subfault -- two million of them, all alive at once. `arange(total) -
    # repeat(starts, lengths)` is vectorised but materialises three arrays the length of
    # the samples, and at this size each one is 3.8 GB.
    #
    # Instead: every sample's column is one more than its predecessor's, except at the
    # start of a row where it drops back to zero. So write those increments -- ones, and
    # `1 - previous length` at each row start -- and integrate them in place.
    #
    # Only non-empty rows get a reset: an empty row shares its start position with the
    # next one, and duplicate fancy-index writes keep the last, which would take the
    # empty row's length instead of the real previous row's.
    #
    # int32 because a column is bounded by the longest pulse, and scipy wants `indices`
    # and `indptr` in one dtype -- which int32 can carry as long as there are fewer
    # than 2^31 samples in total. Past that, int64 and twice the memory is the only
    # option, so the choice is made on the actual count rather than assumed.
    index_dtype = np.int32 if samples.size < np.iinfo(np.int32).max else np.int64
    within = np.ones(samples.size, dtype=index_dtype)
    if within.size:
        within[0] = 0
        occupied = lengths > 0
        starts = offsets[:-1][occupied]
        within[starts[1:]] = 1 - lengths[occupied][:-1]
        np.cumsum(within, out=within)

    return SrfFile(
        version="2.0",
        planes=headers,
        points=points,
        slip_rate=sp.sparse.csr_array(
            (samples, within, offsets.astype(index_dtype)),
            shape=(len(points.longitude_deg), longest),
        ),
    )


__all__ = ["plane_header", "to_srf_file"]
