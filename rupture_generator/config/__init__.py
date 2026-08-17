"""Configuration: what a fault is, and what the earthquake on it is."""

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
