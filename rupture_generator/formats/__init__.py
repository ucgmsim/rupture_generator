"""Which file format a path means, and who writes it.

One enum, one inference rule and one dispatch: a :class:`Format` with an ``INFERRED``
member, a :func:`from_path` that reads the extension, and a ``match`` per writer.

``.srf.h5`` is SW4's SRF-in-HDF5 and ``.h5`` is this package's own format, so inference
looks at the last *two* suffixes first. Getting that backwards writes a native rupture
where a consumer expects SW4's layout -- both are HDF5, so nothing downstream notices
until it reads a dataset that is not there.
"""

from enum import StrEnum, auto
from pathlib import Path


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
    """The Standard Rupture Format, as text. Six significant figures and no provenance."""

    SRF_HDF5 = auto()
    """SW4's SRF-in-HDF5. Someone else's layout, specified in someone else's manual."""


def from_path(path: Path | str) -> Format:
    """Infer a format from a path's extension.

    Parameters
    ----------
    path : Path or str
        Where the file is going, or came from.

    Returns
    -------
    Format
        Never :attr:`Format.INFERRED`.

    Raises
    ------
    ValueError
        If the extension names nothing. Guessing would write a rupture in a layout
        nobody asked for.

    Examples
    --------
    >>> from pathlib import Path
    >>> from rupture_generator.formats import from_path
    >>> from_path(Path("rupture.h5"))
    <Format.NETCDF: 'netcdf'>
    >>> from_path(Path("rupture.srf.h5"))
    <Format.SRF_HDF5: 'srf_hdf5'>
    >>> from_path(Path("rupture.zarr"))
    <Format.ZARR: 'zarr'>
    """
    path = Path(path)

    # Two suffixes first: `.srf.h5` is SW4's and `.h5` is ours, and the specific one has
    # to win or a native file goes out wearing SW4's name.
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

    raise ValueError(
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


__all__ = ["Format", "from_path", "resolve"]
