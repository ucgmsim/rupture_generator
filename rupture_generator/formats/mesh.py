"""The mesh file: node positions, one group per plane.

An ``xr.DataTree``, written to HDF5 or Zarr. One refined plane -- a chart -- per
group: the ``(dip_node, strike_node)`` arrays are its vertices, with its topology
implied by their shape, which is why there is no face list in the file: a structured
grid's connectivity *is* its shape.

Positions are **offsets from the mesh's origin**, in kilometres, in the CRS named in
the root attributes -- the same thing :class:`~rupture_generator.mesh.RuptureMesh`
holds, for the same reason. The origin is stored once, in the root attributes, and
added back by whoever wants a coordinate rather than a shape.

# Only the geometry is stored

Cell centres, areas, strike and dip are all functions of the nodes, so they are
computed on read and never written. A stored quantity that could have been derived is
a second description free to drift from the first.

# The dims are renamed at this seam and nowhere else

In memory a chart's node dims are ``(i_node, j_node)`` -- ``i`` down-dip, ``j`` along
strike. The file keeps its original ``(dip_node, strike_node)`` spelling, so readers
that never import this package keep working. The rename happens here, in both
directions, and nothing else in the package knows both spellings.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyproj
import xarray as xr

from rupture_generator.config.geometry import ComputedPropagation, PropagationConfig
from rupture_generator.formats import Format, resolve
from rupture_generator.mesh import RuptureMesh

if TYPE_CHECKING:
    from collections.abc import Mapping

SCHEMA_VERSION = 1
"""Bumped when a reader of an older file would get the wrong answer rather than an error."""


def to_datatree(
    meshes: Mapping[str, list[RuptureMesh]],
    crs: pyproj.CRS,
    *,
    propagation: PropagationConfig | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> xr.DataTree:
    """Lay charts out as a tree: one group per plane, nested under its surface.

    Parameters
    ----------
    meshes : Mapping of str to list of RuptureMesh
        Surface name to its per-plane charts, in trace order.
    crs : pyproj.CRS
        The frame every position is in. Stored once, in the root.
    propagation : PropagationConfig, optional
        How a rupture crosses between these surfaces. Defaults to the computed form.
    attrs : Mapping, optional
        Extra root attributes -- the config verbatim, a title.

    Returns
    -------
    xr.DataTree
        With ``/<surface>/plane_<n>`` groups.
    """
    groups: dict[str, xr.Dataset] = {}
    origins: dict[str, list[float]] = {}

    for name, charts in meshes.items():
        origins[name] = list(charts[0].origin_km)
        for index, chart in enumerate(charts):
            dip_cells, strike_cells = chart.cell_counts
            dataset = (
                chart.node_dataset()
                .rename({"i_node": "dip_node", "j_node": "strike_node"})
                .assign_coords(
                    strike_km=(
                        "strike_node",
                        chart.strike_arc_km(),
                        {
                            "units": "kilometres",
                            "long_name": "Distance along strike from the i = 0 edge",
                        },
                    ),
                    dip_km=(
                        "dip_node",
                        chart.dip_arc_km(),
                        {
                            "units": "kilometres",
                            "long_name": "Distance down dip from the top edge",
                        },
                    ),
                )
            )
            dataset.attrs = {
                "surface": name,
                "plane": index,
                "strike_count": strike_cells,
                "dip_count": dip_cells,
            }
            groups[f"{name}/plane_{index}"] = dataset

    tree = xr.DataTree.from_dict(groups)
    tree.attrs = {
        "schema_version": SCHEMA_VERSION,
        "created": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "crs": crs.to_string(),
        # One origin per surface, as JSON because an attribute is a scalar or an
        # array and this is a mapping. Read back by `from_datatree` and nothing else.
        "origins": json.dumps(origins),
        # How a rupture crosses between these surfaces is a property of the fault
        # system, so it travels with the fault system. The `geometry_config` attribute
        # the CLI also writes is the input file verbatim -- provenance, for a human
        # and for reproducing a run -- where this is the operative copy the pipeline
        # reads back.
        "propagation": json.dumps((propagation or ComputedPropagation()).to_dict()),
        **dict(attrs or {}),
    }
    return tree


def from_datatree(
    tree: xr.DataTree,
) -> tuple[dict[str, list[RuptureMesh]], pyproj.CRS, PropagationConfig]:
    """Rebuild charts from a tree.

    The inverse of :func:`to_datatree`, and lossless: the nodes are the geometry, so
    everything derived from them comes back identical rather than close.

    Returns
    -------
    tuple
        Surface name to per-plane charts, the CRS they are in, and how a rupture
        crosses between them.

    Raises
    ------
    ValueError
        If the tree carries no CRS, or a surface has no recorded origin. Both mean
        the file is not one of these, and reading it as one would put the fault
        somewhere.
    """
    crs_name = tree.attrs.get("crs")
    if crs_name is None:
        raise ValueError("the file has no crs attribute, so its positions mean nothing")
    origins = json.loads(tree.attrs.get("origins", "{}"))

    # Keyed by the *stored* plane index rather than by the order the groups come
    # back in, because **Zarr does not preserve order** -- see the git history's
    # MESH.md for the measurement. HDF5 does preserve insertion, so trusting
    # iteration order is green in one container and silently permutes the fault in
    # the other -- and since the order varies between runs, it is the kind of thing
    # that fails somewhere else, intermittently, long after.
    by_surface: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for path, node in tree.subtree_with_keys:
        if not node.has_data or "east_km" not in node.dataset:
            continue
        dataset = node.dataset
        surface = node.attrs.get("surface") or Path(path).parent.name
        plane = int(node.attrs["plane"])

        planes = by_surface.setdefault(surface, {})
        if plane in planes:
            raise ValueError(f"{surface!r} has two planes numbered {plane}")
        planes[plane] = (
            dataset["east_km"].to_numpy(),
            dataset["north_km"].to_numpy(),
            dataset["depth_km"].to_numpy(),
        )

    meshes: dict[str, list[RuptureMesh]] = {}
    for surface, planes in by_surface.items():
        if surface not in origins:
            raise ValueError(f"{surface!r} has no origin, so its offsets mean nothing")
        expected = set(range(len(planes)))
        if set(planes) != expected:
            raise ValueError(
                f"{surface!r} has planes {sorted(planes)}, expected {sorted(expected)} "
                "-- a gap means a plane is missing rather than renumbered"
            )
        easting_km, northing_km = origins[surface]
        meshes[surface] = [
            RuptureMesh.from_nodes(
                *planes[index],
                origin_east_km=easting_km,
                origin_north_km=northing_km,
                surface=surface,
                plane_of_column=np.full(
                    planes[index][0].shape[1] - 1, index, dtype=np.int64
                ),
            )
            for index in sorted(planes)
        ]

    stored = tree.attrs.get("propagation")
    propagation = (
        PropagationConfig.from_dict(json.loads(stored))
        if stored
        else ComputedPropagation()
    )
    return meshes, pyproj.CRS(crs_name), propagation


def write_mesh(
    meshes: Mapping[str, list[RuptureMesh]],
    crs: pyproj.CRS,
    path: Path | str,
    *,
    format: Format = Format.INFERRED,
    propagation: PropagationConfig | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> None:
    """Write charts to an HDF5 file or a Zarr store.

    Raises
    ------
    ValueError
        If the format is not one a mesh can be written in. An SRF holds a rupture,
        not a surface, and there is nothing sensible to put in its slip columns.
    """
    path = Path(path)
    chosen = resolve(path, format)
    tree = to_datatree(meshes, crs, propagation=propagation, attrs=attrs)

    match chosen:
        case Format.NETCDF:
            tree.to_netcdf(path, engine="h5netcdf", mode="w")
        case Format.ZARR:
            tree.to_zarr(path, mode="w", consolidated=False)
        case _:
            raise ValueError(
                f"a mesh cannot be written as {chosen.value}: it holds a fault "
                "surface, not a rupture. Use .h5 or .zarr"
            )


def read_mesh(
    path: Path | str, *, format: Format = Format.INFERRED
) -> tuple[dict[str, list[RuptureMesh]], pyproj.CRS, PropagationConfig]:
    """Read charts back.

    Returns
    -------
    tuple
        Surface name to per-plane charts, the CRS, and the propagation.

    Raises
    ------
    ValueError
        If the format is not one a mesh is written in.
    """
    path = Path(path)
    chosen = resolve(path, format)

    match chosen:
        case Format.NETCDF:
            tree = xr.open_datatree(path, engine="h5netcdf")
        case Format.ZARR:
            tree = xr.open_datatree(path, engine="zarr", consolidated=False)
        case _:
            raise ValueError(f"a mesh is not read from {chosen.value}")

    with tree:
        return from_datatree(tree)


__all__ = [
    "SCHEMA_VERSION",
    "from_datatree",
    "read_mesh",
    "to_datatree",
    "write_mesh",
]
