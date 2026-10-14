"""The record of a run, which states the config, software and inputs it ran with.

A record is the config as TOML, every default written out, followed by a
``[provenance]`` table. :func:`~rupture_generator.config.load` reads past that table,
so a record reruns as it stands. The geometry's path, and each input's, is relative to
the record's own location, so a record moved with its inputs still runs. The record
gives a path that shares only the filesystem root with it in full. In an SRF the
record is the header's comments, one per line of TOML, and in a fields file it's the
root's ``run_record`` attribute. Either way it's part of the file it describes.
"""

import dataclasses
import datetime
import hashlib
import importlib.metadata
import platform
import subprocess
import tomllib
from pathlib import Path

from mashumaro.codecs import BasicDecoder
from mashumaro.codecs.toml import TOMLEncoder
from mashumaro.mixins.toml import TOMLDialect

from rupture_generator import __version__
from rupture_generator import config as rupture_config
from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.formats.fields import RECORD_ATTRIBUTE, SUFFIXES
from rupture_generator.formats.srf import read_comments

DEPENDENCIES = (
    "numpy",
    "scipy",
    "pyproj",
    "xarray",
    "h5netcdf",
    "mashumaro",
    "ucgmsim-srf",
)
"""The packages whose versions a record states."""

ATTRIBUTE = RECORD_ATTRIBUTE
"""The fields file's root attribute that holds its record."""


@dataclasses.dataclass(frozen=True)
class Input:
    """A file a run read.

    Attributes
    ----------
    path : str
        Where it was, relative to the record.
    sha256 : str
        Its SHA-256 digest, in hexadecimal. For an SRF a rerun read, the digest of
        the record its header holds: hashing the whole file would read gigabytes for
        the few kilobytes the rerun used.
    """

    path: str
    sha256: str


@dataclasses.dataclass(frozen=True, kw_only=True)
class Provenance:
    """``[provenance]``: what a run ran with.

    Attributes
    ----------
    rupture_generator : str
        This package's version.
    commit : str, optional
        The git commit the package ran from, when it ran from a checkout.
    modified : bool, optional
        Whether that checkout had uncommitted changes.
    python : str
        The Python version.
    platform : str
        The operating system and machine.
    dependencies : dict of str to str
        The version of each package in :data:`DEPENDENCIES`.
    seed : int
        The seed the run drew from.
    created : datetime.datetime
        When the run started, in UTC.
    inputs : list of Input
        The rupture file and the geometry it names.
    """

    rupture_generator: str
    commit: str | None = None
    modified: bool | None = None
    python: str
    platform: str
    dependencies: dict[str, str]
    seed: int
    created: datetime.datetime
    inputs: list[Input]


@dataclasses.dataclass(frozen=True)
class _Table:
    provenance: Provenance


_ENCODER = TOMLEncoder(_Table)
_DECODER = BasicDecoder(Provenance, default_dialect=TOMLDialect)


def _relative(path: Path, directory: Path) -> Path:
    """``path`` relative to ``directory``, or absolute if they share only the root."""
    path, directory = path.resolve(), directory.resolve()
    if path.parts[:2] != directory.parts[:2]:
        return path
    return path.relative_to(directory, walk_up=True)


def _digest(path: Path) -> str:
    """An input's SHA-256: an SRF's record's, any other file's whole contents'."""
    if path.suffix == ".srf":
        return hashlib.sha256(_record(path).encode()).hexdigest()
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _git(*arguments: str) -> str:
    """Run git in this package's directory, and return what it prints."""
    return subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).parent,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def _checkout() -> tuple[str | None, bool | None]:
    """The commit this package runs from and whether it's modified, outside git none."""
    try:
        # A package installed inside some other repository is in no commit of its own.
        _git("ls-files", "--error-unmatch", "__init__.py")
        return _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain", "."))
    except OSError, subprocess.CalledProcessError:
        return None, None


def provenance(source: Path, config: rupture_config.RuptureConfig) -> Provenance:
    """State what a run of ``config`` runs with, its inputs as absolute paths.

    Parameters
    ----------
    source : Path
        The rupture file the config came from.
    config : RuptureConfig
        The loaded config.

    Returns
    -------
    Provenance
        The versions, platform, seed and inputs.

    Raises
    ------
    OSError
        If reading an input fails.
    """
    commit, modified = _checkout()
    return Provenance(
        rupture_generator=__version__,
        commit=commit,
        modified=modified,
        python=platform.python_version(),
        platform=platform.platform(),
        dependencies={name: importlib.metadata.version(name) for name in DEPENDENCIES},
        seed=config.seed,
        created=datetime.datetime.now(datetime.UTC).replace(microsecond=0),
        inputs=[
            Input(str(path.resolve()), _digest(path))
            for path in (Path(source), config.geometry.path)
        ],
    )


def _render(
    config: rupture_config.RuptureConfig,
    stated: Provenance,
    location: Path,
    origin: Path | None = None,
) -> str:
    """The record as TOML, its paths relative to ``location``'s directory.

    The config's geometry path is absolute. The inputs' paths are absolute, or
    relative to ``origin``'s directory when the record came from another file.
    """
    directory = location.parent
    stated = dataclasses.replace(
        stated,
        inputs=[
            Input(
                str(
                    _relative(
                        Path(i.path) if origin is None else origin.parent / i.path,
                        directory,
                    )
                ),
                i.sha256,
            )
            for i in stated.inputs
        ],
    )
    geometry = dataclasses.replace(
        config.geometry, path=_relative(config.geometry.path, directory)
    )
    relative = dataclasses.replace(config, geometry=geometry)
    return f"{rupture_config.dump(relative)}\n{_ENCODER.encode(_Table(stated))}"


def run_record(
    source: Path, config: rupture_config.RuptureConfig, location: Path
) -> str:
    """Write the record of a run as TOML.

    Parameters
    ----------
    source : Path
        The rupture file the config came from.
    config : RuptureConfig
        The loaded config.
    location : Path
        The file the record goes in, an SRF or a fields file. Paths in the record are
        relative to its directory.

    Returns
    -------
    str
        The config, every default written out, then its ``[provenance]``.

    Raises
    ------
    RuptureGeneratorError
        If ``source`` is an SRF without a record.
    OSError
        If reading an input fails.
    """
    return _render(config, provenance(source, config), location)


def relocate(record: str, origin: Path, location: Path) -> str:
    """Rewrite a record that ``origin`` holds for ``location``, its paths moved there.

    The SRF made from a fields file gets the fields file's record this way. It's the
    record of the draw, with paths that still lead to the inputs from the SRF.

    Parameters
    ----------
    record : str
        The record, as TOML.
    origin : Path
        The file that holds it, which its paths are relative to.
    location : Path
        The file it goes in.

    Returns
    -------
    str
        The record as TOML, its paths relative to ``location``'s directory.

    Raises
    ------
    RuptureGeneratorError
        If ``record`` doesn't read as one.
    """
    config, stated = _parse(record, origin)
    return _render(config, stated, location, origin)


def _record(path: Path) -> str:
    """The record a file holds as TOML: an SRF's header, a fields file's attribute,
    or a TOML file whole.

    Raises
    ------
    RuptureGeneratorError
        If the SRF or fields file doesn't contain a record.
    """
    import xarray as xr

    path = Path(path)
    if path.suffix == ".srf":
        comments = read_comments(path)
        if comments:
            return "\n".join(comments)
        where = "header comments"
    elif path.suffix in SUFFIXES:
        with xr.open_datatree(path, engine="h5netcdf") as tree:
            if ATTRIBUTE in tree.attrs:
                return str(tree.attrs[ATTRIBUTE])
        where = f"{ATTRIBUTE} attribute"
    else:
        return path.read_text()
    raise RuptureGeneratorError(f"{path} has no record: it has no {where}")


def _parse(text: str, path: Path) -> tuple[rupture_config.RuptureConfig, Provenance]:
    config = rupture_config.loads(text, path.parent, name=str(path))
    try:
        table = tomllib.loads(text)[rupture_config.PROVENANCE]
    except KeyError as error:
        raise RuptureGeneratorError(
            f"{path} has no [{rupture_config.PROVENANCE}] table"
        ) from error
    return config, _DECODER.decode(table)


def read_config(path: Path) -> rupture_config.RuptureConfig:
    """Read the config a rupture file states, or the one an SRF's record ran.

    Parameters
    ----------
    path : Path
        A rupture file or a record as TOML, or an SRF or fields file holding a
        record.

    Returns
    -------
    RuptureConfig
        The config, its geometry path resolved.

    Raises
    ------
    RuptureGeneratorError
        If the file doesn't contain a record, or describes no rupture.
    OSError
        If reading the file fails.
    """
    path = Path(path)
    return rupture_config.loads(_record(path), path.parent, name=str(path))


def read_record(
    path: Path,
) -> tuple[rupture_config.RuptureConfig, Provenance]:
    """Read the record of a run, from the SRF or fields file holding it.

    Parameters
    ----------
    path : Path
        An SRF or a fields file, or a record as a TOML file of its own.

    Returns
    -------
    tuple of (RuptureConfig, Provenance)
        The config the run ran, its geometry path resolved, and its provenance.

    Raises
    ------
    RuptureGeneratorError
        If the file doesn't contain a record.
    OSError
        If reading the file fails.
    """
    path = Path(path)
    return _parse(_record(path), path)


__all__ = [
    "ATTRIBUTE",
    "DEPENDENCIES",
    "Input",
    "Provenance",
    "provenance",
    "read_config",
    "read_record",
    "relocate",
    "run_record",
]
