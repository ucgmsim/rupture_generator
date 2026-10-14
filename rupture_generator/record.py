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

import srf
from mashumaro.codecs import BasicDecoder
from mashumaro.codecs.toml import TOMLEncoder
from mashumaro.mixins.toml import TOMLDialect

from rupture_generator import __version__
from rupture_generator import config as rupture_config
from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.formats.fields import SUFFIXES

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

ATTRIBUTE = "run_record"
"""The fields file's root attribute that holds its record."""


@dataclasses.dataclass(frozen=True)
class Input:
    """A file a run read.

    Attributes
    ----------
    path : str
        Where it was, relative to the record.
    sha256 : str
        Its SHA-256 digest, in hexadecimal.
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


def _sha256(path: Path) -> str:
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
            Input(str(path.resolve()), _sha256(path))
            for path in (Path(source), config.geometry.path)
        ],
    )


def _render(
    config: rupture_config.RuptureConfig,
    stated: Provenance,
    origin: Path,
    location: Path,
) -> str:
    """The record as TOML, its paths moved from ``origin``'s directory to ``location``'s.

    The config's geometry path is absolute already. The inputs' paths are relative to
    ``origin``'s directory, or absolute.
    """
    directory = location.parent
    stated = dataclasses.replace(
        stated,
        inputs=[
            Input(str(_relative(origin.parent / i.path, directory)), i.sha256)
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
    OSError
        If reading an input fails.
    """
    return _render(config, provenance(source, config), location, location)


def relocated(path: Path, location: Path) -> str | None:
    """Rewrite the record in a file for another file, its paths relative to that.

    The SRF made from a fields file gets the fields file's record this way. It's the
    record of the draw, with paths that still lead to the inputs from the SRF.

    Parameters
    ----------
    path : Path
        A file holding a record, an SRF or a fields file.
    location : Path
        The file the record goes in.

    Returns
    -------
    str or None
        The record as TOML, or ``None`` when ``path`` doesn't contain one.

    Raises
    ------
    RuptureGeneratorError
        If the record in ``path`` doesn't read as one.
    OSError
        If reading the file fails.
    """
    path = Path(path)
    text = _text(path)
    if text is None:
        return None
    config, stated = _parse(text, path)
    return _render(config, stated, path, location)


def _text(path: Path) -> str | None:
    """The record a file holds as TOML, or ``None`` when it holds none."""
    import xarray as xr

    if path.suffix == ".srf":
        try:
            comments = srf.read_comments(path)
        except srf.SrfError as error:
            raise RuptureGeneratorError(str(error)) from error
        return "\n".join(comments) if comments else None
    if path.suffix in SUFFIXES:
        with xr.open_datatree(path, engine="h5netcdf") as tree:
            return str(tree.attrs[ATTRIBUTE]) if ATTRIBUTE in tree.attrs else None
    return path.read_text()


def _parse(text: str, path: Path) -> tuple[rupture_config.RuptureConfig, Provenance]:
    config = rupture_config.loads(text, path.parent, name=str(path))
    try:
        table = tomllib.loads(text)[rupture_config.PROVENANCE]
    except KeyError as error:
        raise RuptureGeneratorError(
            f"{path} has no [{rupture_config.PROVENANCE}] table"
        ) from error
    return config, _DECODER.decode(table)


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
    text = _text(path)
    if text is None:
        where = "header comments" if path.suffix == ".srf" else f"{ATTRIBUTE} attribute"
        raise RuptureGeneratorError(f"{path} has no record: it has no {where}")
    return _parse(text, path)


__all__ = [
    "ATTRIBUTE",
    "DEPENDENCIES",
    "Input",
    "Provenance",
    "provenance",
    "read_record",
    "relocated",
    "run_record",
]
