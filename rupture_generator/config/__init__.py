"""Configuration: what a fault is, and what the earthquake on it is.

Two files, because they have different lifetimes. A geometry is digitised once and
reused across every realisation and every magnitude run on it; a source is what varies.
``rupture-generator mesh`` reads the first and ``generate`` reads the second.

# Importing this module is what registers the tagged unions

`mashumaro`'s ``Discriminator(include_subtypes=True)`` resolves a ``type`` tag against
whatever subclasses *exist*, which means they have to have been imported. The walk below
does that, so dropping a new slip-rate shape or surface kind into its module is the
whole of adding one -- there is no list to keep in step, which is the point.

Deliberately eager rather than lazy. A tag that fails to resolve because a module was
never imported produces "no such type" for a type that is right there, and the
resolution is invisible from the message.
"""

import importlib
import pkgutil

from rupture_generator.config.core import ConfigObject, field_path
from rupture_generator.config.geometry import GeometryConfig, SurfaceConfig
from rupture_generator.config.rupture import RuptureConfig, read_config, read_geometry
from rupture_generator.config.slip_rate import SlipRateShapeConfig


def _register_subtypes() -> None:
    """Import every module in the package, so its tagged subclasses exist."""
    for _finder, name, _is_package in pkgutil.walk_packages(__path__, f"{__name__}."):
        importlib.import_module(name)


_register_subtypes()

__all__ = [
    "ConfigObject",
    "GeometryConfig",
    "RuptureConfig",
    "SlipRateShapeConfig",
    "SurfaceConfig",
    "field_path",
    "read_config",
    "read_geometry",
]
