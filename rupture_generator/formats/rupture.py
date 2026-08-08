"""The rupture file: a generated model, and the mesh it was generated on.

An ``xr.DataTree``, one group per plane, written to HDF5 or Zarr. `FORMAT.md` says what
is in it and why; this reads and writes it.

# Self-contained on purpose

Each group carries the node positions as well as the fields, so a rupture file is
everything a viewer or a consumer needs. The alternative -- a rupture that refers to a
mesh file -- is a pair that can be separated, and a slip field without its geometry is a
grid of numbers.

# Two coordinate systems, and neither is redundant

The **nodes** are projected offsets, exactly as the mesh file holds them: they are the
geometry, and they are what a renderer draws. The **cell** variables are WGS84, because
they are what an SRF is written from and what every consumer downstream expects.

One is input and the other is derived output, which is why this is not the two-
descriptions-of-one-thing the repo forbids elsewhere -- and
`tests/test_rupture_format.py` asserts the derivation still holds after a round trip.

# The pulses are ragged, so they are stored as CSR

Each subfault's pulse has its own length. `data`/`indices`/`indptr` is the layout the
core already produces and `scipy.sparse.csr_array` already wants, so nothing is
translated on either side -- and the column index of a sample is its position *within
its own pulse*, not within the rupture.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyproj
import xarray as xr

from rupture_generator._core import GeneratedRupture, RefinedMesh
from rupture_generator.formats import Format, resolve
from rupture_generator.mesh import Located, project_patch
from rupture_generator.units import CM2_PER_KM2

if TYPE_CHECKING:
    from collections.abc import Mapping

SCHEMA_VERSION = 1

CELL_VARIABLES = {
    "centre_longitude_deg": ("degrees_east", "Subfault centre, WGS84"),
    "centre_latitude_deg": ("degrees_north", "Subfault centre, WGS84"),
    "centre_depth_km": ("kilometres", "Subfault centre depth, positive downwards"),
    "strike_deg": ("degrees", "Strike, clockwise from TRUE north"),
    "dip_deg": ("degrees", "Dip, below horizontal"),
    "area_cm2": ("square centimetres", "Subfault area, as an SRF stores it"),
    "slip_cm": ("centimetres", "Total slip"),
    "rake_deg": ("degrees", "Slip direction within the plane"),
    "onset_s": ("seconds", "When the rupture front arrives"),
    "rise_time_s": ("seconds", "How long the subfault slips for"),
}
"""Every variable carries its unit and a sentence.

`strike_deg` says **TRUE** in capitals because the mesh file's does not: that one is grid
north, and the difference reaches five degrees. A reader who takes one for the other gets
a mechanism wrong by more than the SRF can express.
"""


def to_dataset(
    rupture: GeneratedRupture,
    mesh: RefinedMesh,
    patch: int,
    crs: pyproj.CRS,
    *,
    located: Located | None = None,
    hypocentre_km: tuple[float, float] | None = None,
) -> xr.Dataset:
    """One plane's rupture and geometry, as a dataset.

    Parameters
    ----------
    rupture : GeneratedRupture
        The generated model, flat over subfaults, along-strike fastest.
    mesh, patch : RefinedMesh and int
        Where the fault is.
    crs : pyproj.CRS
        The frame the mesh is in.
    located : Located, optional
        The projection, if it has already been done. Recomputed when omitted.
    hypocentre_km : tuple of float, optional
        Where the rupture started, in the plane's own arc lengths.

    Returns
    -------
    xr.Dataset
        Node variables on ``(dip_node, strike_node)``, cell variables on
        ``(dip, strike)``, and the pulses as CSR on ``(sample,)`` and ``(cell_edge,)``.
    """
    located = located or project_patch(mesh, patch, crs)
    strike_count, dip_count = mesh.cell_extents(patch)
    shape = (dip_count, strike_count)

    east_km, north_km, depth_km = mesh.node_positions(patch)
    cells = {
        "centre_longitude_deg": located.longitude_deg,
        "centre_latitude_deg": located.latitude_deg,
        "centre_depth_km": located.depth_km,
        "strike_deg": located.strike_deg,
        "dip_deg": located.dip_deg,
        "area_cm2": located.area_km2 * CM2_PER_KM2,
        "slip_cm": rupture.slip_cm.reshape(shape),
        "rake_deg": rupture.rake_deg.reshape(shape),
        "onset_s": rupture.onset_s.reshape(shape),
        "rise_time_s": rupture.rise_time_s.reshape(shape),
    }

    offsets = np.asarray(rupture.slip_rate_offsets, dtype=np.int64)
    lengths = np.diff(offsets)
    # The column of a sample is its position within its own pulse, which is what makes
    # this a CSR matrix of shape (subfault, longest pulse).
    columns = (
        np.concatenate([np.arange(length, dtype=np.int64) for length in lengths])
        if len(lengths)
        else np.empty(0, dtype=np.int64)
    )

    data_vars = {
        name: (("dip", "strike"), np.asarray(values, dtype=np.float64), _attrs(name))
        for name, values in cells.items()
    }
    data_vars |= {
        "node_east_km": (
            ("dip_node", "strike_node"),
            east_km,
            {"units": "kilometres", "long_name": "Easting offset from the mesh origin"},
        ),
        "node_north_km": (
            ("dip_node", "strike_node"),
            north_km,
            {"units": "kilometres", "long_name": "Northing offset from the mesh origin"},
        ),
        "node_depth_km": (
            ("dip_node", "strike_node"),
            depth_km,
            {"units": "kilometres", "long_name": "Node depth, positive downwards"},
        ),
        "slip_rate": (
            "sample",
            np.asarray(rupture.slip_rate, dtype=np.float64),
            {
                "units": "centimetres per second",
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

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "strike_km": (
                "strike_node",
                mesh.strike_arc_km(patch),
                {"units": "kilometres", "long_name": "Along strike from the i = 0 edge"},
            ),
            "dip_km": (
                "dip_node",
                mesh.dip_arc_km(patch),
                {"units": "kilometres", "long_name": "Down dip from the top edge"},
            ),
        },
        attrs={
            "plane": patch,
            "strike_count": strike_count,
            "dip_count": dip_count,
            "sample_interval_s": rupture.sample_interval_s,
            "moment_dyne_cm": rupture.moment_dyne_cm,
            "alpha_t": rupture.alpha_t,
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


def _attrs(name: str) -> dict[str, str]:
    unit, description = CELL_VARIABLES[name]
    return {"units": unit, "long_name": description}


def to_datatree(
    planes: Mapping[str, xr.Dataset],
    crs: pyproj.CRS,
    origins: Mapping[str, tuple[float, float]],
    *,
    attrs: Mapping[str, Any] | None = None,
) -> xr.DataTree:
    """Assemble plane datasets into a tree.

    Keys are ``"<surface>/plane_<n>"``, as in the mesh file, so the two are read the
    same way.
    """
    tree = xr.DataTree.from_dict(dict(planes))
    tree.attrs = {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "crs": crs.to_string(),
        "origins": json.dumps({name: list(origin) for name, origin in origins.items()}),
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


def planes_in(tree: xr.DataTree) -> list[tuple[str, int, xr.Dataset]]:
    """Every plane in a rupture tree, in order.

    Sorted on the stored ``plane`` attribute rather than on iteration order, because
    **Zarr does not preserve order** -- see `formats/mesh.py`, where the same trap cost
    a real bug.

    Returns
    -------
    list of (str, int, xr.Dataset)
        Surface name, plane index, and the plane's dataset.
    """
    found = []
    for path, node in tree.subtree_with_keys:
        if not node.has_data or "slip_cm" not in node.dataset:
            continue
        surface = Path(path).parent.name or Path(path).name
        found.append((surface, int(node.attrs["plane"]), node.dataset))
    return sorted(found, key=lambda entry: (entry[0], entry[1]))


__all__ = [
    "CELL_VARIABLES",
    "SCHEMA_VERSION",
    "planes_in",
    "read_rupture",
    "to_dataset",
    "to_datatree",
    "write_rupture",
]
