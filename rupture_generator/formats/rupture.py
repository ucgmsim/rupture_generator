"""The rupture file: a generated model, and the mesh it was generated on.

An ``xr.DataTree``, **one group per segment**, written to HDF5 or Zarr. A segment is
the pipeline's unit -- one chart, one field of each kind, one wavefront -- so a group
is exactly what a stage produced, and "the pipeline returns an annotated mesh" and
"the pipeline's output is the file" are the same statement.

A segment may span several config planes where their seams coincide; the ``plane``
coordinate on the strike axis records which one each cell column came from, so the
provenance survives without the file having to be cut the way the config was written.

Each group carries the node positions as well as the fields, so a rupture file is
everything a viewer or a consumer needs: a slip field without its geometry is a grid of
numbers, and a pair of files can be separated.

Two coordinate systems, neither redundant. The **nodes** are projected offsets, exactly
as the mesh file holds them -- the geometry, and what a renderer draws. The **cell**
variables are WGS84, what an SRF is written from and what consumers expect.

Units are SI: slip in metres, slip rate in metres per second, moment in newton-metres,
area in square metres. The centimetres the SRF format wants appear in `srf.py` and
nowhere else, and every variable carries its unit.

The pulses are ragged, so they are stored as CSR. ``data``/``indptr`` is the layout the
kernel already produces and `scipy.sparse.csr_array` already wants, so nothing is
translated on either side.

**The ``indices`` of that triple are not stored**: every pulse starts at column zero and
runs contiguously, so a sample's column is a function of ``indptr`` alone. Writing them
down doubles the file -- 7.6 GB of int64 restating 7.6 GB of float64 on the shipped
twenty-fault scenario, the array that used to exhaust memory before the rupture could be
written at all. `assemble.to_srf_file` rebuilds them where `scipy.sparse` insists.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyproj
import xarray as xr

from rupture_generator.formats import Format, resolve
from rupture_generator.mesh import RuptureMesh, project_cells
from rupture_generator.realisation import Realisation
from rupture_generator.units import M2_PER_KM2

if TYPE_CHECKING:
    from collections.abc import Mapping

SCHEMA_VERSION = 2
"""Version 2 made the units SI and a group a segment.

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
    "rigidity_pa": ("pascals", "Shear rigidity at subfault"),
    "shear_speed_kms": ("kilometres per second", "Shear wave speed at subfault"),
}
"""Every variable carries its unit and a sentence.

``strike_deg`` says **TRUE** in capitals because the mesh file's does not: that one is
grid north, and the difference reaches five degrees.
"""

NODE_VARIABLES = {
    "node_east_km": ("kilometres", "Easting offset from the mesh origin"),
    "node_north_km": ("kilometres", "Northing offset from the mesh origin"),
    "node_depth_km": ("kilometres", "Node depth, positive downwards"),
}


FILE_FIELDS = (
    "rigidity_pa",
    "shear_speed_kms",
    "slip_m",
    "rake_deg",
    "onset_s",
    "rise_time_s",
)
"""The fields a rupture file stores, read off the chart by name.

The whitelist keeps the pipeline's *working* fields out: a file is a rupture rather
than a trace of how one was made.
"""


def to_dataset(
    mesh: RuptureMesh,
    crs: pyproj.CRS,
    *,
    segment_name: str | None = None,
    sample_interval_s: float,
    moment_newton_m: float,
) -> xr.Dataset:
    """One segment's rupture and geometry, as a dataset.

    The fields are **on the chart**, so this reads them off rather than being told
    them: a field the pipeline stops producing fails here, naming the segment and the
    field, rather than at a call site the pipeline no longer has.

    The hypocentre is not a parameter either. Only the segment the rupture nucleated on
    carries one, in its own attrs, so this copies what the chart records -- writing it
    into every group claimed three hypocentres for one earthquake.

    Parameters
    ----------
    mesh : RuptureMesh
        A chart the pipeline has finished with: the file's fields, and its pulses.
    crs : pyproj.CRS
        The frame its nodes are in -- the one projection seam.
    segment_name : str, optional
        What the causality tree calls this segment. Stored because a surface can yield
        several segments, which would otherwise be distinguishable only by the group
        name -- a convention rather than a record.
    sample_interval_s, moment_newton_m : float

    Raises
    ------
    KeyError
        If a field the file stores is not on the chart. A realisation that has not been
        all the way through the pipeline is not a rupture, and writing one with holes in
        it produces a file whose readers fail instead.
    ValueError
        If the chart carries no pulses.
    """
    located = project_cells(mesh, crs)
    cells_i, cells_j = mesh.cell_counts

    pulses = mesh.pulses
    if pulses is None:
        raise ValueError(
            f"{mesh.surface} has no slip-rate pulses, so it has not been through the "
            "whole pipeline"
        )
    offsets, pulse_samples = pulses

    cells = {
        "centre_longitude_deg": located["centre_longitude_deg"].to_numpy(),
        "centre_latitude_deg": located["centre_latitude_deg"].to_numpy(),
        "centre_depth_km": located["centre_depth_km"].to_numpy(),
        "strike_deg": located["strike_deg"].to_numpy(),
        "dip_deg": located["dip_deg"].to_numpy(),
        "area_m2": located["area_km2"].to_numpy() * M2_PER_KM2,
        **{name: mesh[name] for name in FILE_FIELDS},
    }

    offsets = np.asarray(offsets, dtype=np.int64)

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
            "segment": segment_name or mesh.surface,
            "strike_count": cells_j,
            "dip_count": cells_i,
            "sample_interval_s": sample_interval_s,
            "moment_newton_m": moment_newton_m,
            "origin_east_km": mesh.origin_km[0],
            "origin_north_km": mesh.origin_km[1],
            # Whatever the chart recorded about itself: the truncation diagnostic, and
            # -- on the one segment that nucleated -- the hypocentre, already under the
            # names the file uses. `RESERVED_ATTRS` is what makes this splat safe.
            **{
                name: value
                for name, value in mesh.attrs.items()
                if name not in ("surface", "origin_east_km", "origin_north_km")
            },
        },
    )
    return dataset


def to_datatree(
    realisation: Realisation,
    *,
    attrs: Mapping[str, Any] | None = None,
) -> xr.DataTree:
    """A generated rupture as an event tree: one group per segment.

    **The seam the pipeline does not cross.** `pipeline.generate` produces a
    `Realisation` and stops. Which fields the file stores, what the groups are called,
    and which working fields are dropped are decided here -- which removes the
    inversion the pipeline used to carry, where it imported this module, called it once
    per segment, and so made the file layout something you had to read in order to read
    the stage order.

    Parameters
    ----------
    realisation : Realisation
        A rupture that has been through the pipeline.
    attrs : Mapping, optional
        The **caller's** provenance: a title, the config verbatim, the seed and the
        realisation index. What the rupture *is* -- the frame, the causality tree, the
        jumps, the event moment -- comes off the realisation and is written either way.
    """
    # Once, not once per segment: the property sums over every chart, so reading it
    # inside the comprehension would make the writer quadratic in the segment count.
    moment_newton_m = realisation.moment_newton_m
    tree = xr.DataTree.from_dict(
        {
            # A colon is a path separator to a datatree, and a fused surface's parts
            # are called `kaikoura:0`. The segment's own name is in its attrs, which is
            # what `segments_in` reads it back by.
            f"{name.replace(':', '_')}/segment": to_dataset(
                mesh,
                realisation.crs,
                segment_name=name,
                sample_interval_s=float(mesh.attrs["sample_interval_s"]),
                moment_newton_m=moment_newton_m,
            )
            for name, mesh in realisation.items()
        }
    )
    tree.attrs = {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "crs": realisation.crs.to_string(),
        "causality_tree": json.dumps(realisation.tree),
        "jumps": json.dumps(
            {
                name: dataclasses.asdict(jump)
                for name, jump in realisation.jumps.items()
            }
        ),
        "moment_newton_m": moment_newton_m,
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


def segments_in(tree: xr.DataTree) -> list[tuple[str, xr.Dataset]]:
    """Every segment in a rupture tree, in a stable order.

    Zarr does not preserve order when saved.

    Returns
    -------
    list of (str, xr.Dataset)
        The segment's name -- what the causality tree calls it -- and its dataset.
    """
    found = []
    for path, node in tree.subtree_with_keys:
        if not node.has_data or "slip_m" not in node.dataset:
            continue
        segment = node.attrs.get("segment") or node.attrs.get("surface") or path
        found.append((str(segment), path, node.dataset))
    return [
        (segment, dataset)
        for (segment, _, dataset) in sorted(found, key=lambda e: e[1])
    ]


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
    "FILE_FIELDS",
    "NODE_VARIABLES",
    "SCHEMA_VERSION",
    "mesh_of",
    "read_rupture",
    "segments_in",
    "to_dataset",
    "to_datatree",
    "write_rupture",
]
