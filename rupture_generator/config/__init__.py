"""Configuration: what a fault is, and what the earthquake on it is.

Two files, because they have different lifetimes. A geometry is digitised once and
reused across every realisation and every magnitude run on it; a source is what varies.
``rupture-generator mesh`` reads the first and ``generate`` reads the second.

Importing this module is what registers the tagged unions. `mashumaro`'s
``Discriminator(include_subtypes=True)`` resolves a ``type`` tag against whatever
subclasses *exist*, so they have to have been imported; the walk below does that, and
dropping a new surface or source kind into its module is the whole of adding one.

Deliberately eager rather than lazy: a tag that fails to resolve because a module was
never imported produces "no such type" for a type that is right there.
"""

import importlib
import pkgutil

from rupture_generator.config.core import ConfigObject, field_path
from rupture_generator.config.geometry import GeometryConfig, SurfaceConfig
from rupture_generator.config.rupture import RuptureConfig, read_config, read_geometry


def _register_subtypes() -> None:
    """Import every module in the package, so its tagged subclasses exist."""
    for _finder, name, _is_package in pkgutil.walk_packages(__path__, f"{__name__}."):
        importlib.import_module(name)


_register_subtypes()

__all__ = [
    "ConfigObject",
    "GeometryConfig",
    "RuptureConfig",
    "SurfaceConfig",
    "field_path",
    "read_config",
    "read_geometry",
]
