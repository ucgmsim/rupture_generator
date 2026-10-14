"""A drawn rupture's fields as an xarray DataTree in HDF5, and an SRF made from them.

An SRF stores each subfault's slip-rate series, hundreds of samples per cell. Those
follow from slip, rise time and the pulse shape, so a file of the fields is a fraction
of the size and makes the same SRF on demand. The file keeps every field in double
precision, so the SRF written from a file read back is byte-identical to one written
from the draw.

The root group stores the projection as WKT in ``crs``, the hypocentre in
``hypocentre_segment``, ``hypocentre_strike_km`` and ``hypocentre_dip_km``. Its
``segment`` variable names the segments in drawing order, the order the SRF writes
them in, and ``parent`` names the segment that triggered each one, empty for the
hypocentre's. Each segment is a group of its own name. Its cell fields are on ``(dip, strike)`` with their units in
``units``, its chart's nodes on ``(node_dip, node_strike, axis)``, its cell centres on
``(dip, strike, axis)``, and its three statistics and its pulse's ``dt_s`` in
attributes.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyproj

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.formats.srf import write_rupture
from rupture_generator.geometry import Geometry
from rupture_generator.rupture.generator import SegmentRupture
from rupture_generator.rupture.realisation import Hypocentre, Realisation

if TYPE_CHECKING:
    import xarray as xr

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
_SCALARS = ("alpha_t", "rise_time_mean_s", "onset_scale_s", "dt_s")
"""Each scalar of :class:`SegmentRupture`, kept as an attribute."""
_COMPRESSION = {"zlib": True, "complevel": 4}


def _segment(rupture: SegmentRupture) -> tuple[xr.Dataset, dict[str, dict]]:
    """One segment's group, and the compression for each of its variables."""
    import xarray as xr

    geometry = rupture.geometry
    variables: dict[str, tuple] = {
        name: (_CELL, getattr(rupture, name), {"units": units})
        for name, units in _FIELDS.items()
    }
    variables["occupied"] = (_CELL, geometry.occupied)
    if rupture.beta is not None:
        variables["beta"] = (_CELL, rupture.beta, {"units": "1"})
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
        attrs={name: getattr(rupture, name) for name in _SCALARS},
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
    attrs: Mapping[str, str | int | float] | None = None,
) -> None:
    """Write drawn segments, and what an SRF of them needs, as a DataTree in HDF5.

    Parameters
    ----------
    path : Path or str
        Where to write the file.
    realisation : Realisation of SegmentRupture
        The drawn fault system.
    attrs : Mapping of str to str, int or float, optional
        More attributes for the root, such as the run's record.

    Raises
    ------
    RuptureGeneratorError
        For a segment name containing ``/``, which separates groups.
    OSError
        If writing the file fails.
    """
    import xarray as xr

    path = Path(path)
    names = list(realisation)
    awkward = [name for name in names if "/" in name]
    if awkward:
        raise RuptureGeneratorError(
            f"{awkward} contain '/', which separates the groups of {path}"
        )
    root_attrs: dict[str, str | int | float] = {
        **(attrs or {}),
        "crs": realisation.crs.to_wkt(),
    }
    if (hypocentre := realisation.hypocentre) is not None:
        root_attrs |= {
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
    groups: dict[str, xr.Dataset] = {"/": xr.Dataset(variables, attrs=root_attrs)}
    encoding: dict[str, dict[str, dict]] = {}
    for name, rupture in realisation.items():
        groups[name], encoding[f"/{name}"] = _segment(rupture)
    try:
        xr.DataTree.from_dict(groups).to_netcdf(path, engine=_ENGINE, encoding=encoding)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def read_fields(path: Path | str) -> Realisation[SegmentRupture]:
    """Read a file :func:`write_fields` wrote.

    Parameters
    ----------
    path : Path or str
        The file.

    Returns
    -------
    Realisation of SegmentRupture
        The drawn fault system.

    Raises
    ------
    RuptureGeneratorError
        If the file lacks a segment or a variable :func:`write_fields` writes.
    OSError
        If reading the file fails.
    """
    import xarray as xr

    path = Path(path)
    with xr.open_datatree(path, engine=_ENGINE) as tree:
        try:
            root = tree.dataset
            names = [str(name) for name in root["segment"].values]
            segments = {}
            for name in names:
                group = tree[name].dataset
                geometry = Geometry(
                    nodes=group["nodes_km"].values,
                    occupied=group["occupied"].values.astype(bool),
                    plane_cells=tuple(int(n) for n in group["plane_cells"].values),
                )
                values: dict[str, Any] = {f: group[f].values for f in _FIELDS}
                values |= {s: float(group.attrs[s]) for s in _SCALARS}
                values["beta"] = group["beta"].values if "beta" in group else None
                segments[name] = SegmentRupture(geometry=geometry, **values)
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
        except KeyError as error:
            raise RuptureGeneratorError(
                f"{path} has no {error}; it is not a fields file"
            ) from error
    return Realisation(segments, crs, hypocentre, parents)


def fields_to_srf(
    source: Path | str, destination: Path | str, *, comments: Sequence[str] = ()
) -> None:
    """Write the SRF of the rupture stored in a fields file.

    Parameters
    ----------
    source : Path or str
        The fields file :func:`write_fields` wrote.
    destination : Path or str
        Where to write the SRF.
    comments : sequence of str, optional
        The SRF header's comment lines, such as the run's record.

    Raises
    ------
    RuptureGeneratorError
        If ``source`` isn't a fields file, or the pulse kernel refuses a pulse.
    OSError
        If reading or writing a file fails.
    """
    write_rupture(str(destination), read_fields(source), comments=comments)


__all__ = ["SUFFIXES", "fields_to_srf", "read_fields", "write_fields"]
