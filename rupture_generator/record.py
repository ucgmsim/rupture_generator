"""The record of a run, which states the config, software and inputs it ran with.

A record is the config as TOML, every default written out, followed by a
``[provenance]`` table. In an SRF it's the header's comments, one per line of TOML,
and in a fields file it's the root's ``run_record`` attribute. It records how a file
was made. Nothing reads it back.
"""

import dataclasses
import datetime
import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

from mashumaro.codecs.toml import TOMLEncoder

from rupture_generator import __version__
from rupture_generator import config as rupture_config

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


@dataclasses.dataclass(frozen=True)
class Input:
    """A file a run read.

    Attributes
    ----------
    path : str
        Its absolute path.
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


_ENCODER = TOMLEncoder(dict[str, Provenance])


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
    """State what a run of ``config`` runs with.

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
    inputs = []
    for path in (Path(source), config.geometry.path):
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        inputs.append(Input(str(path.resolve()), digest))
    return Provenance(
        rupture_generator=__version__,
        commit=commit,
        modified=modified,
        python=platform.python_version(),
        platform=platform.platform(),
        dependencies={name: importlib.metadata.version(name) for name in DEPENDENCIES},
        seed=config.seed,
        created=datetime.datetime.now(datetime.UTC).replace(microsecond=0),
        inputs=inputs,
    )


def run_record(source: Path, config: rupture_config.RuptureConfig) -> str:
    """Write the record of a run as TOML.

    Parameters
    ----------
    source : Path
        The rupture file the config came from.
    config : RuptureConfig
        The loaded config.

    Returns
    -------
    str
        The config, every default written out, then its ``[provenance]``.

    Raises
    ------
    OSError
        If reading an input fails.
    """
    stated = provenance(source, config)
    return f"{rupture_config.dump(config)}\n{_ENCODER.encode({'provenance': stated})}"


__all__ = ["DEPENDENCIES", "Input", "Provenance", "provenance", "run_record"]
