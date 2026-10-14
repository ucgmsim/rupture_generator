"""A drawn rupture's fields as an xarray DataTree in HDF5, and an SRF made from them.

An SRF stores each subfault's slip-rate series, hundreds of samples per cell. Those
follow from slip, rise time and the pulse shape, so a file of the fields is a fraction
of the size and makes the same SRF on demand. The file keeps every field in double
precision, so the SRF written from a file read back is byte-identical to one written
from the draw.

The root group stores the projection as WKT in ``crs``, the hypocentre in
``hypocentre_segment``, ``hypocentre_strike_km`` and ``hypocentre_dip_km``, and the
pulse's ``dt_s``. Its ``segment`` variable names the segments in drawing order, the
order the SRF writes them in, and ``parent`` names the segment that triggered each
one, empty for the hypocentre's. Each segment is a group of its own name. Its cell
fields are on ``(dip, strike)`` with their units in ``units``, its chart's nodes on
``(node_dip, node_strike, axis)``, its cell centres on ``(dip, strike, axis)``, and its
three statistics in attributes.
"""

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import xarray as xr

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.formats.srf import write_rupture
from rupture_generator.geometry import CellArray, Geometry
from rupture_generator.rupture.generator import SegmentRupture
from rupture_generator.rupture.realisation import Hypocentre, Realisation

SUFFIXES = (".h5", ".nc")
"""The suffixes of a fields file."""

_ENGINE = "h5netcdf"
_CELL = ("dip", "strike")
_AXIS = ["east", "north", "depth"]
_FIELDS = {
    "slip_m": "m",
    "rise_time_s": "s",
    "rake_deg": "degree",
    "onset_s": "s",
    "speed_km_s": "km s-1",
    "shear_speed_km_s": "km s-1",
    "density_g_cm3": "g cm-3",
}
"""Each cell field of :class:`SegmentRupture`, with its units."""
_STATISTICS = ("alpha_t", "rise_time_mean_s", "onset_scale_s")
_COMPRESSION = {"zlib": True, "complevel": 4}


@dataclasses.dataclass(frozen=True, eq=False)
class StoredRupture:
    """A drawn rupture read from a fields file, ready for an SRF.

    Attributes
    ----------
    realisation : Realisation of SegmentRupture
        The drawn fault system.
    dt_s : float
        The slip-rate sample interval in seconds.
    beta : dict of str to CellArray or None
        The pulse's rising fraction per cell by segment, or ``None`` for impulses.
    """

    realisation: Realisation[SegmentRupture]
    dt_s: float
    beta: dict[str, CellArray] | None


def _segment(
    rupture: SegmentRupture, beta: CellArray | None
) -> tuple[xr.Dataset, dict[str, dict]]:
    """One segment's group, and the compression for each of its variables."""
    geometry = rupture.geometry
    variables: dict[str, tuple] = {
        name: (_CELL, getattr(rupture, name), {"units": units})
        for name, units in _FIELDS.items()
    }
    variables["occupied"] = (_CELL, geometry.occupied)
    if beta is not None:
        variables["beta"] = (_CELL, beta, {"units": "1"})
    variables["nodes_km"] = (
        ("node_dip", "node_strike", "axis"),
        geometry.nodes,
        {"units": "km"},
    )
    variables["centres_km"] = (_CELL + ("axis",), geometry.centres, {"units": "km"})
    variables["plane_cells"] = ("plane", np.asarray(geometry.plane_cells))
    dataset = xr.Dataset(
        variables,
        coords={"axis": _AXIS},
        attrs={name: getattr(rupture, name) for name in _STATISTICS},
    )
    encoding = {
        str(name): _COMPRESSION
        for name, variable in dataset.data_vars.items()
        if variable.ndim >= 2
    }
    return dataset, encoding


def write_fields(
    path: Path | str,
    realisation: Realisation[SegmentRupture],
    *,
    dt_s: float,
    beta: Mapping[str, CellArray] | None = None,
) -> None:
    """Write drawn segments, and what an SRF of them needs, as a DataTree in HDF5.

    Parameters
    ----------
    path : Path or str
        Where to write the file.
    realisation : Realisation of SegmentRupture
        The drawn fault system.
    dt_s : float
        The slip-rate sample interval in seconds.
    beta : Mapping of str to CellArray, optional
        The Liu-Archuleta-Hartzell rising fraction per cell, by segment.

    Raises
    ------
    RuptureGeneratorError
        For a segment name containing ``/``, which separates groups.
    OSError
        If writing the file fails.
    """
    path = Path(path)
    names = list(realisation)
    awkward = [name for name in names if "/" in name]
    if awkward:
        raise RuptureGeneratorError(
            f"{awkward} contain '/', which separates the groups of {path}"
        )
    attrs: dict[str, str | float] = {
        "crs": realisation.crs.to_wkt(),
        "dt_s": dt_s,
    }
    if (hypocentre := realisation.hypocentre) is not None:
        attrs |= {
            "hypocentre_segment": hypocentre.segment,
            "hypocentre_strike_km": hypocentre.strike_km,
            "hypocentre_dip_km": hypocentre.dip_km,
        }
    variables = {"segment": ("segment", names)}
    if realisation.tree:
        variables["parent"] = (
            "segment",
            [realisation.tree[name] or "" for name in names],
        )
    groups: dict[str, xr.Dataset] = {"/": xr.Dataset(variables, attrs=attrs)}
    encoding: dict[str, dict[str, dict]] = {}
    for name, rupture in realisation.items():
        segment_beta = None if beta is None else beta.get(name)
        groups[name], encoding[f"/{name}"] = _segment(rupture, segment_beta)
    try:
        xr.DataTree.from_dict(groups).to_netcdf(path, engine=_ENGINE, encoding=encoding)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def read_fields(path: Path | str) -> StoredRupture:
    """Read a file :func:`write_fields` wrote.

    Parameters
    ----------
    path : Path or str
        The file.

    Returns
    -------
    StoredRupture
        The drawn system and its pulse.

    Raises
    ------
    RuptureGeneratorError
        If the file lacks a segment or a variable :func:`write_fields` writes.
    OSError
        If reading the file fails.
    """
    path = Path(path)
    with xr.open_datatree(path, engine=_ENGINE) as tree:
        try:
            root = tree.dataset
            names = [str(name) for name in root["segment"].values]
            segments, beta = {}, {}
            for name in names:
                group = tree[name].dataset
                geometry = Geometry(
                    nodes=group["nodes_km"].values,
                    occupied=group["occupied"].values.astype(bool),
                    plane_cells=tuple(int(n) for n in group["plane_cells"].values),
                )
                values: dict[str, Any] = {f: group[f].values for f in _FIELDS}
                values |= {s: float(group.attrs[s]) for s in _STATISTICS}
                segments[name] = SegmentRupture(geometry=geometry, **values)
                if "beta" in group:
                    beta[name] = group["beta"].values
            parents = (
                {
                    name: str(parent) or None
                    for name, parent in zip(names, root["parent"].values, strict=True)
                }
                if "parent" in root
                else {}
            )
            hypocentre = (
                Hypocentre(
                    str(root.attrs["hypocentre_segment"]),
                    float(root.attrs["hypocentre_strike_km"]),
                    float(root.attrs["hypocentre_dip_km"]),
                )
                if "hypocentre_segment" in root.attrs
                else None
            )
            crs = pyproj.CRS.from_wkt(str(root.attrs["crs"]))
            dt_s = float(root.attrs["dt_s"])
        except KeyError as error:
            raise RuptureGeneratorError(
                f"{path} has no {error}; it is not a fields file"
            ) from error
    return StoredRupture(
        realisation=Realisation(segments, crs, hypocentre, parents),
        dt_s=dt_s,
        beta=beta or None,
    )


def fields_to_srf(source: Path | str, destination: Path | str) -> None:
    """Write the SRF of the rupture stored in a fields file.

    Parameters
    ----------
    source : Path or str
        The fields file :func:`write_fields` wrote.
    destination : Path or str
        Where to write the SRF.

    Raises
    ------
    RuptureGeneratorError
        If ``source`` isn't a fields file, or the pulse kernel refuses a pulse.
    OSError
        If reading or writing a file fails.
    """
    stored = read_fields(source)
    write_rupture(
        str(destination), stored.realisation, dt_s=stored.dt_s, beta=stored.beta
    )


__all__ = ["SUFFIXES", "StoredRupture", "fields_to_srf", "read_fields", "write_fields"]
