"""What a fault surface looks like written down.

The input to ``rupture-generator mesh``. It describes where the fault is and nothing
about the earthquake on it, which is `rupture.py`'s job.

Traces are written in longitude and latitude; ``crs`` names the projected coordinate
reference system the mesh is *built* in, and the subcommand converts once on the way
in. Connectivity is structural: a fault is an origin and a list of planes, each giving
only where its top edge *ends*, so two planes that do not meet cannot be written down.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import pyproj
from mashumaro.types import Discriminator, SerializationStrategy

from rupture_generator.config.core import ConfigObject
from rupture_generator.config.validation import (
    DepthKm,
    DipDeg,
    Latitude,
    Longitude,
    NonEmptyStr,
    PositiveFloat,
    PositiveInt,
    StrikeDeg,
)


class CrsStrategy(SerializationStrategy, use_annotations=True):
    """A CRS as whatever `pyproj` accepts: ``"EPSG:2193"``, ``2193``, WKT, PROJ."""

    def serialize(self, value: pyproj.CRS) -> str:
        """The authority code, where there is one."""
        return value.to_string()

    def deserialize(self, value: str | int) -> pyproj.CRS:
        """Whatever `pyproj.CRS` accepts."""
        return pyproj.CRS(value)


CRS = dataclasses.field(
    metadata={"serialization_strategy": CrsStrategy()},
)


@dataclasses.dataclass
class LonLat(ConfigObject):
    """A point on the surface: two named fields, so a config cannot swap them."""

    longitude_deg: Longitude
    latitude_deg: Latitude


@dataclasses.dataclass
class Discretisation(ConfigObject):
    """How finely a plane is cut, given one way or the other.

    ``subfault_size_km`` is a *request*: the plane is cut into whole cells, so the size
    actually used is the plane's own length over the resulting count. ``strike_count``
    and ``dip_count`` say it exactly.
    """

    subfault_size_km: PositiveFloat | None = None
    strike_count: PositiveInt | None = None
    dip_count: PositiveInt | None = None

    def __post_init__(self) -> None:
        """Validate the fields, then check exactly one form of discretisation."""
        super().__post_init__()

        by_size = self.subfault_size_km is not None
        by_count = self.strike_count is not None or self.dip_count is not None

        if by_size and by_count:
            self.refuse(
                "subfault_size_km",
                "give either a subfault size or explicit counts, not both -- they "
                "would disagree and there is no rule for which wins",
            )
        if not by_size and not by_count:
            self.refuse(
                "subfault_size_km",
                "a plane needs a subfault size, or a strike_count and a dip_count",
            )
        if by_count and (self.strike_count is None or self.dip_count is None):
            missing = "dip_count" if self.dip_count is None else "strike_count"
            self.refuse(missing, "explicit counts need both a strike and a dip count")


@dataclasses.dataclass
class PlaneConfig(ConfigObject):
    """One plane of a fault: where its top edge ends, and how it hangs from it.

    Where the top edge *begins* is the previous plane's ``end``, or the fault's
    ``origin``. Dip, depth and discretisation are per plane.
    """

    end: LonLat
    dip_deg: DipDeg
    bottom_depth_km: PositiveFloat
    discretisation: Discretisation
    dip_direction: Literal["right", "left"] = "right"

    @property
    def dips_left(self) -> bool:
        """Whether the plane dips left, walking the trace from its first point."""
        return self.dip_direction == "left"


@dataclasses.dataclass
class SurfaceConfig(ConfigObject):
    """A named fault surface. Tagged, so a file can hold several kinds."""

    class Config(ConfigObject.Config):
        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclasses.dataclass
class FaultConfig(SurfaceConfig):
    """A connected run of planes.

    ``top_depth_km`` is the depth in kilometres of the trace they all hang from.
    """

    name: NonEmptyStr
    origin: LonLat
    planes: list[PlaneConfig]
    top_depth_km: DepthKm = 0.0
    type: Literal["fault"] = "fault"

    def __post_init__(self) -> None:
        """Validate the fields, then the invariants between them."""
        super().__post_init__()
        if not self.planes:
            self.refuse("planes", "a fault needs at least one plane")
        for plane in self.planes:
            if plane.bottom_depth_km <= self.top_depth_km:
                self.refuse(
                    "planes",
                    f"a plane reaches {plane.bottom_depth_km} km, which is not below "
                    f"the fault's top at {self.top_depth_km} km",
                )


@dataclasses.dataclass
class PointConfig(SurfaceConfig):
    """A point source: one subfault, given by its centre rather than a top edge."""

    name: NonEmptyStr
    centre: LonLat
    depth_km: DepthKm
    strike_deg: StrikeDeg
    dip_deg: DipDeg
    size_km: PositiveFloat = 1.0
    type: Literal["point"] = "point"


@dataclasses.dataclass
class GeometryConfig(ConfigObject):
    """A whole geometry file: a CRS, and the surfaces in it.

    Examples
    --------
    TOML::

        schema_version = 1
        crs = "EPSG:2193"

        [[surfaces]]
        type = "fault"
        name = "kaikoura"
        origin = { longitude_deg = 173.00, latitude_deg = -42.60 }
        top_depth_km = 0.0

        [[surfaces.planes]]
        end = { longitude_deg = 173.40, latitude_deg = -42.40 }
        dip_deg = 70.0
        bottom_depth_km = 15.0
        discretisation = { subfault_size_km = 1.0 }

        [[surfaces.planes]]
        end = { longitude_deg = 173.90, latitude_deg = -42.10 }
        dip_deg = 55.0
        bottom_depth_km = 12.0
        discretisation = { subfault_size_km = 1.0 }
    """

    crs: pyproj.CRS = CRS
    surfaces: list[SurfaceConfig] = dataclasses.field(default_factory=list)
    schema_version: int = 1
    title: str | None = None

    def __post_init__(self) -> None:
        """Validate the fields, then the invariants between them."""
        super().__post_init__()
        if not self.surfaces:
            self.refuse("surfaces", "a geometry needs at least one surface")

        names = [surface.name for surface in self.surfaces]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            self.refuse(
                "surfaces",
                f"two surfaces are called {sorted(duplicates)!r}; names become group "
                "names in the mesh file, so they have to be distinct",
            )

        if not self.crs.is_projected:
            self.refuse(
                "crs",
                f"{self.crs.to_string()!r} is not a projected CRS. The mesh is built in "
                "a Cartesian frame -- give a projection covering the region, such as "
                "EPSG:2193 for New Zealand, rather than a geographic CRS like EPSG:4326",
            )
