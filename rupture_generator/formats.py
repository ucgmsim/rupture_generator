"""This package's own files: which format a path means, and the two layouts inside.

One ``xr.DataTree`` per file, written to HDF5 or Zarr, in one of two layouts:

**A mesh** is a fault surface with nothing drawn on it -- ``/<surface>/plane_<n>``, one
group per chart in trace order. The ``(dip_node, strike_node)`` arrays are its vertices
and there is no face list, because a structured grid's connectivity *is* its shape.

**A rupture** is a generated model and the mesh it was generated on -- one group per
segment, a segment being the pipeline's unit of one chart, one field of each kind and
one wavefront. Each group carries its node positions too, so a rupture file is
everything a viewer or a consumer needs.

Positions are offsets from an origin stored once in the root, in kilometres, in the CRS
the root names. Only geometry is stored: centres, areas, strike and dip are functions
of the nodes, computed on read and never written. The node dims are renamed at this
seam and nowhere else -- in memory ``(i_node, j_node)``, ``i`` down dip and ``j`` along
strike; in the file the original ``(dip_node, strike_node)``, so outside readers keep
working.

Units are SI: slip in metres, slip rate in metres per second, moment in newton-metres,
area in square metres. The centimetres the SRF format wants appear in `srf.py` and
nowhere else, and every variable carries its unit.

The pulses are ragged, so they are stored as CSR ``data``/``indptr`` -- the layout the
kernel already produces and `scipy.sparse.csr_array` already wants. **The ``indices`` of
that triple are not stored**: every pulse starts at column zero and runs contiguously,
so a sample's column is a function of ``indptr`` alone, and writing them down would add
7.6 GB of int64 to the shipped twenty-fault scenario. `assemble.to_srf_file` rebuilds
them where `scipy.sparse` insists.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyproj
import xarray as xr

from rupture_generator.errors import FormatError
from rupture_generator.mesh import RuptureMesh, project_cells
from rupture_generator.units import M2_PER_KM2

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rupture_generator.realisation import Realisation

SCHEMA_VERSION = 2
"""Bumped when a reader of an older file would get a wrong answer rather than an error.

Version 2 made the units SI, made a rupture group a segment, and stopped storing
``propagation`` on a mesh, which is a property of the earthquake rather than of the
surfaces. A version 1 reader would take metres for centimetres.
"""


class Format(StrEnum):
    """A container, and the layout inside it.

    Examples
    --------
    >>> from rupture_generator.formats import Format
    >>> Format.NETCDF == "netcdf"
    True
    """

    INFERRED = auto()
    """Work it out from the path. What the CLI passes unless told otherwise."""

    NETCDF = auto()
    """This package's own layout, in one HDF5 file. The default for a rupture."""

    ZARR = auto()
    """The same layout, as a Zarr store. A directory rather than a file."""

    SRF = auto()
    """The Standard Rupture Format, as text. Six significant figures, no provenance."""

    SRF_HDF5 = auto()
    """SW4's SRF-in-HDF5, specified in someone else's manual."""


def from_path(path: Path | str) -> Format:
    """Infer a format from a path's extension.

    ``.srf.h5`` is SW4's SRF-in-HDF5 and ``.h5`` is this package's own, so inference
    looks at the last *two* suffixes first. Both are HDF5, so getting it backwards is
    not noticed downstream until something reads a dataset that is not there.

    Returns
    -------
    Format
        Never :attr:`Format.INFERRED`.

    Raises
    ------
    FormatError
        If the extension names nothing.

    Examples
    --------
    >>> from pathlib import Path
    >>> from rupture_generator.formats import from_path
    >>> from_path(Path("rupture.h5"))
    <Format.NETCDF: 'netcdf'>
    >>> from_path(Path("rupture.srf.h5"))
    <Format.SRF_HDF5: 'srf_hdf5'>
    """
    path = Path(path)

    if "".join(path.suffixes[-2:]).lower() in {".srf.h5", ".srf.hdf5"}:
        return Format.SRF_HDF5

    by_suffix = {
        ".h5": Format.NETCDF,
        ".hdf5": Format.NETCDF,
        ".nc": Format.NETCDF,
        ".zarr": Format.ZARR,
        ".srf": Format.SRF,
    }
    suffix = path.suffix.lower()
    if suffix in by_suffix:
        return by_suffix[suffix]

    raise FormatError(
        f"no format for {path.name!r}. Give one of "
        f"{sorted(by_suffix)} or .srf.h5, or say --format"
    )


def resolve(path: Path | str, format: Format = Format.INFERRED) -> Format:
    """The format to use: the one given, or the one the path implies.

    Returns
    -------
    Format
        Never :attr:`Format.INFERRED`.
    """
    return from_path(path) if format is Format.INFERRED else format


# The container, which is the same question for either layout.


def _write_tree(tree: xr.DataTree, path: Path, chosen: Format, holds: str) -> None:
    """Write a tree in whichever of the two containers was asked for.

    Raises
    ------
    FormatError
        For a format this does not write. Text SRF and SW4's SRF-HDF5 go through
        `rupture_generator.srf`, which owns those layouts.
    """
    match chosen:
        case Format.NETCDF:
            tree.to_netcdf(path, engine="h5netcdf", mode="w")
        case Format.ZARR:
            tree.to_zarr(path, mode="w", consolidated=False)
        case _:
            raise FormatError(
                f"a {holds} cannot be written as {chosen.value}. Use .h5 or .zarr, or "
                "see rupture_generator.srf for the SRF layouts"
            )


def _open_tree(path: Path, chosen: Format, holds: str) -> xr.DataTree:
    """Open a tree from either container, left open for the caller to close."""
    match chosen:
        case Format.NETCDF:
            return xr.open_datatree(path, engine="h5netcdf")
        case Format.ZARR:
            return xr.open_datatree(path, engine="zarr", consolidated=False)
        case _:
            raise FormatError(f"a {holds} is not read from {chosen.value}")


# The mesh layout: a fault surface, with nothing drawn on it.


def _mesh_tree(
    meshes: Mapping[str, list[RuptureMesh]],
    crs: pyproj.CRS,
    *,
    attrs: Mapping[str, Any] | None = None,
) -> xr.DataTree:
    """Lay charts out as a tree, the CRS and origins stored once in the root."""
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
        # One origin per surface, as JSON because an attribute is a scalar or an array
        # and this is a mapping.
        "origins": json.dumps(origins),
        **dict(attrs or {}),
    }
    return tree


def _meshes_from_tree(
    tree: xr.DataTree,
) -> tuple[dict[str, list[RuptureMesh]], pyproj.CRS]:
    """Rebuild charts from a tree, losslessly: the nodes are the geometry.

    Raises
    ------
    FormatError
        If the tree carries no CRS, or a surface has no recorded origin, or its planes
        are not numbered contiguously from zero.
    """
    crs_name = tree.attrs.get("crs")
    if crs_name is None:
        raise FormatError(
            "the file has no crs attribute, so its positions mean nothing"
        )
    origins = json.loads(tree.attrs.get("origins", "{}"))

    # Keyed by the *stored* plane index rather than by iteration order, because Zarr
    # does not preserve order and HDF5 does: trusting it permutes the fault in one
    # container and not the other.
    by_surface: dict[str, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for path, node in tree.subtree_with_keys:
        if not node.has_data or "east_km" not in node.dataset:
            continue
        dataset = node.dataset
        surface = node.attrs.get("surface") or Path(path).parent.name
        plane = int(node.attrs["plane"])

        planes = by_surface.setdefault(surface, {})
        if plane in planes:
            raise FormatError(f"{surface!r} has two planes numbered {plane}")
        planes[plane] = (
            dataset["east_km"].to_numpy(),
            dataset["north_km"].to_numpy(),
            dataset["depth_km"].to_numpy(),
        )

    meshes: dict[str, list[RuptureMesh]] = {}
    for surface, planes in by_surface.items():
        if surface not in origins:
            raise FormatError(f"{surface!r} has no origin, so its offsets mean nothing")
        expected = set(range(len(planes)))
        if set(planes) != expected:
            raise FormatError(
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

    return meshes, pyproj.CRS(crs_name)


def write_mesh(
    meshes: Mapping[str, list[RuptureMesh]],
    crs: pyproj.CRS,
    path: Path | str,
    *,
    format: Format = Format.INFERRED,
    attrs: Mapping[str, Any] | None = None,
) -> None:
    """Write charts to an HDF5 file or a Zarr store."""
    path = Path(path)
    _write_tree(
        _mesh_tree(meshes, crs, attrs=attrs), path, resolve(path, format), "mesh"
    )


def read_mesh(
    path: Path | str, *, format: Format = Format.INFERRED
) -> tuple[dict[str, list[RuptureMesh]], pyproj.CRS]:
    """Read charts back: surface name to per-plane charts, and their CRS."""
    path = Path(path)
    with _open_tree(path, resolve(path, format), "mesh") as tree:
        return _meshes_from_tree(tree)


# The rupture layout: a generated model, and the mesh it was generated on.

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

``strike_deg`` says **TRUE** in capitals because the mesh layout's does not: that one is
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

A whitelist, so the pipeline's *working* fields stay out.
"""


def segment_dataset(
    mesh: RuptureMesh,
    crs: pyproj.CRS,
    *,
    segment_name: str | None = None,
    sample_interval_s: float,
    moment_newton_m: float,
) -> xr.Dataset:
    """One segment's rupture and geometry, as a dataset.

    The fields and the hypocentre are **on the chart**, so this reads them off rather
    than being told them. Only the segment the rupture nucleated on carries a
    hypocentre; ``segment_name`` is stored because one surface can yield several
    segments, which the group name alone records only by convention.

    Raises
    ------
    KeyError
        If a field the file stores is not on the chart, so the realisation has not been
        all the way through the pipeline.
    FormatError
        If the chart carries no pulses.
    """
    located = project_cells(mesh, crs)
    cells_i, cells_j = mesh.cell_counts

    pulses = mesh.pulses
    if pulses is None:
        raise FormatError(
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

    return xr.Dataset(
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


def rupture_tree(
    realisation: Realisation,
    *,
    attrs: Mapping[str, Any] | None = None,
) -> xr.DataTree:
    """A generated rupture as an event tree: one group per segment.

    **The seam the pipeline does not cross.** Which fields the file stores, what the
    groups are called, and which working fields are dropped are decided here.

    ``attrs`` is the **caller's** provenance: a title, the config verbatim, the seed and
    the realisation index. What the rupture *is* -- the frame, the causality tree, the
    jumps, the event moment -- comes off the realisation and is written either way.
    """
    # Once, not once per segment: the property sums over every chart, so reading it
    # inside the comprehension would make the writer quadratic in the segment count.
    moment_newton_m = realisation.moment_newton_m
    tree = xr.DataTree.from_dict(
        {
            # A colon is a path separator to a datatree, and a fused surface's parts
            # are called `kaikoura:0`. `segments_in` reads the name back from attrs.
            f"{name.replace(':', '_')}/segment": segment_dataset(
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
            {name: dataclasses.asdict(jump) for name, jump in realisation.jumps.items()}
        ),
        "moment_newton_m": moment_newton_m,
        **dict(attrs or {}),
    }
    return tree


def write_rupture(
    tree: xr.DataTree, path: Path | str, *, format: Format = Format.INFERRED
) -> None:
    """Write a rupture tree."""
    path = Path(path)
    _write_tree(tree, path, resolve(path, format), "rupture")


def read_rupture(path: Path | str, *, format: Format = Format.INFERRED) -> xr.DataTree:
    """Read a rupture tree back.

    Returned open, because a caller usually wants a group or two rather than all of it.
    """
    path = Path(path)
    return _open_tree(path, resolve(path, format), "rupture")


def segments_in(tree: xr.DataTree) -> list[tuple[str, xr.Dataset]]:
    """Every segment in a rupture tree, sorted by group path, since Zarr does not.

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


__all__ = [
    "CELL_VARIABLES",
    "FILE_FIELDS",
    "NODE_VARIABLES",
    "SCHEMA_VERSION",
    "Format",
    "from_path",
    "read_mesh",
    "read_rupture",
    "resolve",
    "rupture_tree",
    "segment_dataset",
    "segments_in",
    "write_mesh",
    "write_rupture",
]
