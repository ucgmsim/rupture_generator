"""Turning generated segments into an SRF file.

The pipeline produces physics on charts, in SI. An SRF wants flat arrays in CGS, one
PLANE record per segment, points ordered along strike fastest within each segment in
turn. This is the translation, and it is the **only** place in the package where
metres become centimetres.

The plane header comes **from the mesh**, not from a reconstruction. Recomputing a
plane centre from a fault width and a dip needs a tangent-plane approximation that is
off by 43 m on a crustal fault and 1.9 km at subduction scale; the mesh was built in a
projected frame where the quantity is an identity, so this asks it rather than deriving
it worse.

The one convention conversion: an SRF's ``shyp`` is measured from the along-strike
**centre** of the plane, where the config and the mesh both measure from its ``j = 0``
end. That subtraction happens here, at the seam that writes the format wanting it.

One PLANE record per segment, its points ordered by segment, so a bent or
multi-segment fault is written as several planes whose points follow in the same order.
"""

from __future__ import annotations

import numpy as np
import scipy as sp
import xarray as xr

from rupture_generator.mesh import RuptureMesh, project_cells
from rupture_generator.realisation import (
    HYPOCENTRE_DIP_KM,
    HYPOCENTRE_STRIKE_KM,
    Realisation,
)
from rupture_generator.srf import PlaneHeader, Points, SrfFile
from rupture_generator.units import (
    CM2_PER_M2,
    CM_PER_KM,
    CM_PER_M,
    M2_PER_KM2,
    SRF_FLOAT,
)


def plane_header(
    mesh: RuptureMesh, located: xr.Dataset, *, hypocentre_km: tuple[float, float] | None
) -> PlaneHeader:
    """The PLANE record for one segment, read off its own chart.

    Parameters
    ----------
    mesh : RuptureMesh
        The chart the rupture is on.
    located : xr.Dataset
        `mesh.project_cells`' output -- the only thing here that needs the frame.
        Passed in because :func:`to_srf_file` wants it for the point columns as well,
        and the projection is the expensive part.
    hypocentre_km : tuple of float, optional
        Where the rupture started, in this segment's own arc lengths, or ``None`` for
        a segment that does not hold it. The format has no way to say "not here", so
        a segment without the hypocentre records zeros, which a reader of a multi-plane
        SRF has to know already.

    Returns
    -------
    PlaneHeader
    """
    length_km = float(mesh.strike_arc_km()[-1])
    width_km = float(mesh.dip_arc_km()[-1])
    cells_i, cells_j = mesh.cell_counts

    strike_km, dip_km = hypocentre_km or (0.0, 0.0)

    return PlaneHeader(
        centre_longitude_deg=float(located["centre_longitude_deg"].mean()),
        centre_latitude_deg=float(located["centre_latitude_deg"].mean()),
        strike_count=cells_j,
        dip_count=cells_i,
        length_km=length_km,
        width_km=width_km,
        strike_deg=float(located["strike_deg"].to_numpy()[0, 0]),
        dip_deg=float(located["dip_deg"].to_numpy()[0, 0]),
        top_depth_km=float(mesh.nodes()[..., 2].min()),
        # The SRF measures the hypocentre from the plane's along-strike centre; the
        # config and the mesh measure from its j = 0 end.
        hypocentre_strike_km=strike_km - length_km / 2.0 if hypocentre_km else 0.0,
        hypocentre_dip_km=dip_km,
    )


POINT_COLUMNS = (
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
"""Every column a version 2.0 point block carries, in the order `Points` declares them.

Named once because two writers fill them -- this one from a lattice and
:func:`rupture_generator.triangular.assemble.to_srf_file` from a triangulation -- and a
column one of them forgot would be a file of zeros that still parses.
"""


def srf_file(
    headers: list[PlaneHeader],
    columns: dict[str, list[np.ndarray]],
    pulse_lengths: list[np.ndarray],
    pulse_samples: list[np.ndarray],
) -> SrfFile:
    """One SRF file from per-segment columns already in the format's own units.

    The half of the writing that is **not** about the shape of a chart: concatenating
    each segment's columns into one run of points, rebuilding the CSR offsets across
    segments, and narrowing the samples into their final buffer. A triangulated rupture
    reaches this with different columns and a different header and needs none of it
    written twice -- which matters more than it looks, because most of what is here is
    memory discipline rather than arithmetic (see the comments), and a second
    transcription would be a second place to get 20 GB of peak for a 3.8 GB answer.

    Parameters
    ----------
    headers : list of PlaneHeader
        One per segment, in the order their points follow.
    columns : dict of str to list of np.ndarray
        Keyed by :data:`POINT_COLUMNS`, one flat array per segment, already converted.
    pulse_lengths : list of np.ndarray
        Each segment's per-subfault pulse lengths.
    pulse_samples : list of np.ndarray
        Each segment's concatenated slip-rate samples, in metres per second.

    Returns
    -------
    SrfFile
        Version 2.0.
    """
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
    for source in pulse_samples:
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


def to_srf_file(
    realisation: Realisation,
) -> SrfFile:
    """Assemble an SRF version 2.0 file from a generated rupture.

    One PLANE per segment, in the realisation's own order.

    Version 2.0 carries shear speed and density per point, and both are fields the
    materials stage attached, so nothing here re-reads the velocity model. That
    resampling used to happen in `generate_cli`, from each subfault's *stored centre
    depth* -- a second reading of a quantity the pipeline had already computed, and one
    that could disagree with it.

    Parameters
    ----------
    realisation : Realisation
        A rupture that has been through the pipeline.

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

    for mesh in realisation.values():
        cells_i, cells_j = mesh.cell_counts
        subfaults = cells_i * cells_j
        located = project_cells(mesh, realisation.crs)

        hypocentre_km = (
            (
                float(mesh.attrs[HYPOCENTRE_STRIKE_KM]),
                float(mesh.attrs[HYPOCENTRE_DIP_KM]),
            )
            if HYPOCENTRE_STRIKE_KM in mesh.attrs
            else None
        )
        headers.append(plane_header(mesh, located, hypocentre_km=hypocentre_km))

        def projected(name: str, dataset: xr.Dataset = located) -> np.ndarray:
            return dataset[name].to_numpy().ravel()

        interval_s = float(mesh.attrs["sample_interval_s"])

        # SI leaves the package here. Slip and area cross into the format's own
        # units; depth, angles and times are already what it wants.
        columns["longitude_deg"].append(projected("centre_longitude_deg"))
        columns["latitude_deg"].append(projected("centre_latitude_deg"))
        columns["depth_km"].append(projected("centre_depth_km"))
        columns["strike_deg"].append(projected("strike_deg"))
        columns["dip_deg"].append(projected("dip_deg"))
        columns["area_cm2"].append(projected("area_km2") * M2_PER_KM2 * CM2_PER_M2)
        columns["onset_s"].append(mesh["onset_s"].ravel())
        columns["sample_interval_s"].append(np.full(subfaults, interval_s))
        columns["rake_deg"].append(mesh["rake_deg"].ravel())
        columns["slip_cm"].append(mesh["slip_m"].ravel() * CM_PER_M)
        columns["rise_time_s"].append(mesh["rise_time_s"].ravel())
        columns["shear_speed_cm_s"].append(mesh["shear_speed_kms"].ravel() * CM_PER_KM)
        columns["density_g_cm3"].append(mesh["density_g_cm3"].ravel())

        offsets, samples = mesh.pulses  # ty: ignore[not-iterable]
        pulse_lengths.append(np.diff(offsets))
        samples_of.append(samples)

    return srf_file(headers, columns, pulse_lengths, samples_of)


__all__ = ["POINT_COLUMNS", "plane_header", "srf_file", "to_srf_file"]
