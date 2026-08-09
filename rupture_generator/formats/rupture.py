"""The rupture file: a generated model, and the mesh it was generated on.

An ``xr.DataTree``, **one group per segment**, written to HDF5 or Zarr. A segment is
the pipeline's unit -- one chart, one field of each kind, one wavefront -- so a group
is exactly what a stage produced, and "the pipeline returns an annotated mesh" and
"the pipeline's output is the file" are the same statement.

A segment may span several config planes where their seams coincide; the ``plane``
coordinate on the strike axis records which one each cell column came from, so the
provenance survives without the file having to be cut the way the config was written.

# Self-contained on purpose

Each group carries the node positions as well as the fields, so a rupture file is
everything a viewer or a consumer needs. The alternative -- a rupture that refers to a
mesh file -- is a pair that can be separated, and a slip field without its geometry is
a grid of numbers.

# Two coordinate systems, and neither is redundant

The **nodes** are projected offsets, exactly as the mesh file holds them: they are the
geometry, and they are what a renderer draws. The **cell** variables are WGS84, because
they are what an SRF is written from and what every consumer downstream expects. One is
input and the other is derived output.

# Units are SI

Slip in metres, slip rate in metres per second, moment in newton-metres, area in
square metres. The centimetres the SRF format wants appear in `srf.py` and nowhere
else. Every variable carries its unit, which is the discipline that stopped shear speed
being written in km/s where the SRF wants cm/s.

# The pulses are ragged, so they are stored as CSR

Each subfault's pulse has its own length. ``data``/``indices``/``indptr`` is the layout
the kernel already produces and `scipy.sparse.csr_array` already wants, so nothing is
translated on either side -- and the column index of a sample is its position *within
its own pulse*, not within the rupture.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyproj
import xarray as xr

from rupture_generator.formats import Format, resolve
from rupture_generator.mesh import RuptureMesh, project_cells
from rupture_generator.units import M2_PER_KM2

if TYPE_CHECKING:
    from collections.abc import Mapping

SCHEMA_VERSION = 2
"""Bumped from the port's format: the units are SI and a group is a segment.

A reader of a version-1 file would take metres for centimetres, which is the kind of
disagreement a version number exists for.
"""

CELL_VARIABLES = {
    "centre_longitude_deg": ("degrees_east", "Subfault centre, WGS84"),
    "centre_latitude_deg": ("degrees_north", "Subfault centre, WGS84"),
    "centre_depth_km": ("kilometres", "Subfault centre depth, positive downwards"),
    "strike_deg": ("degrees", "Strike, clockwise from TRUE north"),
    "dip_deg": ("degrees", "Dip, below horizontal"),
    "area_m2": ("square metres", "Subfault area"),
    "slip_m": ("metres", "Total slip"),
    "rake_deg": ("degrees", "Slip direction within the plane"),
    "onset_s": ("seconds", "When the rupture front arrives"),
    "rise_time_s": ("seconds", "How long the subfault slips for"),
}
"""Every variable carries its unit and a sentence.

``strike_deg`` says **TRUE** in capitals because the mesh file's does not: that one is
grid north, and the difference reaches five degrees. A reader who takes one for the
other gets a mechanism wrong by more than the SRF can express.
"""

NODE_VARIABLES = {
    "node_east_km": ("kilometres", "Easting offset from the mesh origin"),
    "node_north_km": ("kilometres", "Northing offset from the mesh origin"),
    "node_depth_km": ("kilometres", "Node depth, positive downwards"),
}


def to_dataset(
    mesh: RuptureMesh,
    crs: pyproj.CRS,
    *,
    slip_m: np.ndarray,
    rake_deg: np.ndarray,
    onset_s: np.ndarray,
    rise_time_s: np.ndarray,
    pulse_offsets: np.ndarray,
    pulse_samples: np.ndarray,
    sample_interval_s: float,
    moment_newton_m: float,
    hypocentre_km: tuple[float, float] | None = None,
) -> xr.Dataset:
    """One segment's rupture and geometry, as a dataset.

    Parameters
    ----------
    mesh : RuptureMesh
        The chart the fields live on.
    crs : pyproj.CRS
        The frame its nodes are in.
    slip_m, rake_deg, onset_s, rise_time_s : np.ndarray
        Cell fields on ``(i, j)``.
    pulse_offsets, pulse_samples : np.ndarray
        The CSR pulses from the kernel: ``pulse_samples[offsets[k]:offsets[k+1]]`` is
        subfault ``k``'s, flattened along strike fastest.
    sample_interval_s, moment_newton_m : float
    hypocentre_km : tuple of float, optional
        Where the rupture started, in **this segment's own** arc lengths. Omitted on
        a segment that does not hold it -- writing it into every group claimed three
        hypocentres for one earthquake.

    Returns
    -------
    xr.Dataset
    """
    located = project_cells(mesh, crs)
    cells_i, cells_j = mesh.cell_counts

    cells = {
        "centre_longitude_deg": located["centre_longitude_deg"].to_numpy(),
        "centre_latitude_deg": located["centre_latitude_deg"].to_numpy(),
        "centre_depth_km": located["centre_depth_km"].to_numpy(),
        "strike_deg": located["strike_deg"].to_numpy(),
        "dip_deg": located["dip_deg"].to_numpy(),
        "area_m2": located["area_km2"].to_numpy() * M2_PER_KM2,
        "slip_m": slip_m,
        "rake_deg": rake_deg,
        "onset_s": onset_s,
        "rise_time_s": rise_time_s,
    }

    offsets = np.asarray(pulse_offsets, dtype=np.int64)
    lengths = np.diff(offsets)
    # The column of a sample is its position within its own pulse, which is what makes
    # this a CSR matrix of shape (subfault, longest pulse).
    columns = (
        np.concatenate([np.arange(length, dtype=np.int64) for length in lengths])
        if len(lengths)
        else np.empty(0, dtype=np.int64)
    )

    nodes = mesh.nodes()
    data_vars: dict[str, Any] = {
        name: (
            ("i", "j"),
            np.asarray(values, dtype=np.float64),
            {"units": CELL_VARIABLES[name][0], "long_name": CELL_VARIABLES[name][1]},
        )
        for name, values in cells.items()
    }
    data_vars |= {
        name: (
            ("i_node", "j_node"),
            nodes[..., axis],
            {"units": unit, "long_name": description},
        )
        for axis, (name, (unit, description)) in enumerate(NODE_VARIABLES.items())
    }
    data_vars |= {
        "slip_rate": (
            "sample",
            np.asarray(pulse_samples, dtype=np.float64),
            {
                "units": "metres per second",
                "long_name": "Slip-rate pulses, concatenated (CSR data)",
            },
        ),
        "slip_rate_column": (
            "sample",
            columns,
            {"long_name": "Sample position within its own pulse (CSR indices)"},
        ),
        "slip_rate_offset": (
            "cell_edge",
            offsets,
            {"long_name": "Where each subfault's pulse starts (CSR indptr)"},
        ),
    }

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "strike_km": (
                "j_node",
                mesh.strike_arc_km(),
                {
                    "units": "kilometres",
                    "long_name": "Along strike from the j = 0 edge",
                },
            ),
            "dip_km": (
                "i_node",
                mesh.dip_arc_km(),
                {"units": "kilometres", "long_name": "Down dip from the top edge"},
            ),
            "plane": (
                "j",
                mesh.planes(),
                {"long_name": "Which config plane this cell column came from"},
            ),
        },
        attrs={
            "surface": mesh.surface,
            "strike_count": cells_j,
            "dip_count": cells_i,
            "sample_interval_s": sample_interval_s,
            "moment_newton_m": moment_newton_m,
            "origin_east_km": mesh.origin_km[0],
            "origin_north_km": mesh.origin_km[1],
            **(
                {
                    "hypocentre_strike_km": hypocentre_km[0],
                    "hypocentre_dip_km": hypocentre_km[1],
                }
                if hypocentre_km
                else {}
            ),
        },
    )
    return dataset


def to_datatree(
    segments: Mapping[str, xr.Dataset],
    crs: pyproj.CRS,
    *,
    attrs: Mapping[str, Any] | None = None,
) -> xr.DataTree:
    """Assemble segment datasets into an event tree.

    Keys are ``"<surface>/segment_<n>"``. The root carries the CRS and whatever the
    caller records about the run -- the config verbatim, the seed, and (once there is
    more than one segment) the causality tree.
    """
    tree = xr.DataTree.from_dict(dict(segments))
    tree.attrs = {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "crs": crs.to_string(),
        **dict(attrs or {}),
    }
    return tree


def write_rupture(
    tree: xr.DataTree, path: Path | str, *, format: Format = Format.INFERRED
) -> None:
    """Write a rupture tree.

    Raises
    ------
    ValueError
        If asked for a format this does not write. Text SRF and SW4's SRF-HDF5 go
        through `rupture_generator.srf`, which owns those layouts.
    """
    path = Path(path)
    chosen = resolve(path, format)

    match chosen:
        case Format.NETCDF:
            tree.to_netcdf(path, engine="h5netcdf", mode="w")
        case Format.ZARR:
            tree.to_zarr(path, mode="w", consolidated=False)
        case _:
            raise ValueError(
                f"{chosen.value} is not written here -- see rupture_generator.srf"
            )


def read_rupture(path: Path | str, *, format: Format = Format.INFERRED) -> xr.DataTree:
    """Read a rupture tree back.

    Returned open, because a caller usually wants a group or two rather than all of it.
    """
    path = Path(path)
    chosen = resolve(path, format)

    match chosen:
        case Format.NETCDF:
            return xr.open_datatree(path, engine="h5netcdf")
        case Format.ZARR:
            return xr.open_datatree(path, engine="zarr", consolidated=False)
        case _:
            raise ValueError(f"a rupture is not read from {chosen.value}")


def segments_in(tree: xr.DataTree) -> list[tuple[str, int, xr.Dataset]]:
    """Every segment in a rupture tree, in a stable order.

    Sorted on the group's own name rather than on iteration order, because **Zarr does
    not preserve order**: eleven groups written in sequence come back in neither
    insertion nor lexicographic order, while HDF5 preserves insertion. A reader that
    trusted iteration order is green in one container and silently permutes the fault
    in the other.

    Returns
    -------
    list of (str, int, xr.Dataset)
        Surface name, segment index, and the segment's dataset.
    """
    found = []
    for path, node in tree.subtree_with_keys:
        if not node.has_data or "slip_m" not in node.dataset:
            continue
        name = Path(path).name
        surface = node.attrs.get("surface") or Path(path).parent.name
        index = int(name.rsplit("_", 1)[-1]) if "_" in name else 0
        found.append((str(surface), index, node.dataset))
    return sorted(found, key=lambda entry: (entry[0], entry[1]))


def mesh_of(dataset: xr.Dataset) -> RuptureMesh:
    """The chart a segment was generated on, rebuilt from its stored nodes.

    Lossless: the nodes are the geometry, so every derived quantity comes back
    identical rather than close.
    """
    return RuptureMesh.from_nodes(
        dataset["node_east_km"].to_numpy(),
        dataset["node_north_km"].to_numpy(),
        dataset["node_depth_km"].to_numpy(),
        origin_east_km=float(dataset.attrs["origin_east_km"]),
        origin_north_km=float(dataset.attrs["origin_north_km"]),
        surface=str(dataset.attrs["surface"]),
        plane_of_column=dataset["plane"].to_numpy(),
    )


__all__ = [
    "CELL_VARIABLES",
    "NODE_VARIABLES",
    "SCHEMA_VERSION",
    "mesh_of",
    "read_rupture",
    "segments_in",
    "to_dataset",
    "to_datatree",
    "write_rupture",
]
