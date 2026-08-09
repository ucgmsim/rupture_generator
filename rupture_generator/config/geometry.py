"""What a fault surface looks like written down.

The input to ``rupture-generator mesh``. It describes *where the fault is* and nothing
about the earthquake on it, which is `rupture.py`'s job -- the two are separate files
because a geometry is reused across realisations and a source is not.

# Positions are longitude and latitude; the mesh is built in a projection

A trace is digitised in longitude and latitude and that is how it is written here.
``crs`` names the projected coordinate reference system the mesh is *built* in, and the
subcommand converts once on the way in. Which CRS is a real choice with real
consequences -- it is the frame every derived quantity is exact in, and its distortion
over the region is the modeller's to judge -- so it is stated rather than assumed.

# Connectivity is structural, and there is no `union` type

A ``[[fault]]`` is an origin and a list of planes, each giving only where its top edge
*ends*. The near end is the previous plane's far end, so two planes that do not meet
cannot be written down. `crates/genslip/src/mesh.rs` has the argument in full.

An earlier draft of this had a ``union`` type for disjoint geometries. It is not here
because the top-level list already *is* the union: several ``[[fault]]`` entries in one
file are several unconnected surfaces, and a type that said so as well would be a second
way to spell the same thing.
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
    """A coordinate reference system as whatever `pyproj` accepts.

    ``"EPSG:2193"``, ``2193``, a WKT string, a PROJ string. Serialising gives back the
    authority code where there is one, so a config that round-trips does not grow a page
    of WKT it did not start with.
    """

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
    """A point on the surface, as it is written in a catalogue or a trace file.

    Two fields rather than a bare pair, so a config cannot silently swap them. The
    ordering mistake is otherwise invisible in New Zealand, where a longitude of 172 and
    a latitude of -43 are both plausible-looking numbers and only one of them is in
    range -- which is exactly the near miss that makes it worth spelling out.
    """

    longitude_deg: Longitude
    latitude_deg: Latitude


@dataclasses.dataclass
class Discretisation(ConfigObject):
    """How finely a plane is cut, given one way or the other.

    ``subfault_size_km`` is the usual one and is a *request*: the plane is cut into
    whole cells, so the size actually used is the plane's own length over the resulting
    count. ``strike_count`` and ``dip_count`` say it exactly, for a fixture or a
    comparison that needs a particular grid.

    The rounding happens in the subcommand rather than here, because it needs the
    plane's length and width, which need the projection. What is here is the request and
    the refusal to accept both forms at once.
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

    Where the top edge *begins* is not here -- it is the previous plane's ``end``, or
    the fault's ``origin``.

    Dip, depth and discretisation are all per plane. A multi-segment fault whose
    segments dip differently and reach different depths is ordinary, and each of its
    planes becomes its own mesh.
    """

    end: LonLat
    dip_deg: DipDeg
    bottom_depth_km: PositiveFloat
    discretisation: Discretisation
    dip_direction: Literal["right", "left"] = "right"

    @property
    def dips_left(self) -> bool:
        """Whether the plane dips left of the trace direction.

        Named by the right-hand rule: walking the trace from its first point to its
        last, ``"right"`` dips away to your right.
        """
        return self.dip_direction == "left"


@dataclasses.dataclass
class SurfaceConfig(ConfigObject):
    """A named fault surface. Tagged, so a file can hold several kinds."""

    class Config(ConfigObject.Config):
        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclasses.dataclass
class FaultConfig(SurfaceConfig):
    """A connected run of planes.

    ``top_depth_km`` belongs to the fault rather than to each plane: it is the depth of
    the trace they all hang from. A segment starting deeper than its neighbour does not
    touch it and is a different fault.
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
    """A point source: one subfault, centred where it is told.

    Given by its *centre* rather than by a top edge, because that is how a catalogue
    gives one. The conversion to a one-cell plane is the library's.
    """

    name: NonEmptyStr
    centre: LonLat
    depth_km: DepthKm
    strike_deg: StrikeDeg
    dip_deg: DipDeg
    size_km: PositiveFloat = 1.0
    type: Literal["point"] = "point"


@dataclasses.dataclass
class PropagationConfig(ConfigObject):
    """How a rupture crosses between the surfaces of a fault system.

    Tagged, because the two ways of answering are different in kind rather than in
    degree: either the tree is *computed* from how far apart the faults are, or it is
    *stated*. A file that says nothing gets the computed form with its defaults, so a
    geometry with one surface never has to mention this at all.
    """

    class Config(ConfigObject.Config):
        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclasses.dataclass
class ComputedPropagation(PropagationConfig):
    """Sample which fault triggers which from how far apart they are.

    The probability that a rupture jumps a gap follows Shaw & Dieterich (2007):
    certain within ``delta_km``, decaying with characteristic length ``d0_km``, and
    beyond ``max_jump_km`` not a jump anyone models. The tree is then drawn from the
    distribution those probabilities imply -- or, with ``strategy =
    "maximum_likelihood"``, taken as its single most likely member, which is what a
    campaign wanting the modal scenario rather than a sample asks for.
    """

    strategy: Literal["sampled", "maximum_likelihood"] = "sampled"
    d0_km: PositiveFloat = 3.0
    delta_km: PositiveFloat = 1.0
    max_jump_km: PositiveFloat = 15.0
    type: Literal["computed"] = "computed"


@dataclasses.dataclass
class PredeterminedPropagation(PropagationConfig):
    """State which fault triggers which, rather than sampling it.

    ``parents`` maps each triggered fault to the one that triggered it. The fault
    that appears as nobody's child is the root, and it must be the one the hypocentre
    is on -- stated in the rupture config, checked here, so the tree is written down
    once rather than twice.

    Examples
    --------
    TOML::

        [propagation]
        type = "predetermined"
        parents = { kelly = "hope", conway = "kelly" }
    """

    parents: dict[str, str] = dataclasses.field(default_factory=dict)
    type: Literal["predetermined"] = "predetermined"

    def __post_init__(self) -> None:
        """Validate the fields, then the shape of what they describe."""
        super().__post_init__()
        if not self.parents:
            self.refuse(
                "parents",
                "a predetermined propagation needs to say which fault triggers "
                "which; with one surface there is nothing to state, so use the "
                "computed form or omit the section",
            )
        for child, parent in self.parents.items():
            if child == parent:
                self.refuse("parents", f"{child!r} cannot trigger itself")


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
    propagation: PropagationConfig = dataclasses.field(
        default_factory=lambda: ComputedPropagation()
    )
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

        if isinstance(self.propagation, PredeterminedPropagation):
            named = set(self.propagation.parents) | set(
                self.propagation.parents.values()
            )
            # A propagation names *segments*, and a surface whose planes do not all
            # share a seam becomes several of them -- ``surface:0``, ``surface:1``.
            # Only the surface part can be checked here, because how many segments a
            # surface yields is not known until it has been meshed and fused. The
            # whole name is checked against the real segments when the rupture runs.
            unknown = {
                name for name in named if name.split(":", 1)[0] not in set(names)
            }
            if unknown:
                self.refuse(
                    "propagation",
                    f"the propagation names {', '.join(sorted(unknown))}, which "
                    f"{'is' if len(unknown) == 1 else 'are'} not on any surface in "
                    f"this geometry ({', '.join(sorted(names))})",
                )

        if not self.crs.is_projected:
            self.refuse(
                "crs",
                f"{self.crs.to_string()!r} is not a projected CRS. The mesh is built in "
                "a Cartesian frame -- give a projection covering the region, such as "
                "EPSG:2193 for New Zealand, rather than a geographic CRS like EPSG:4326",
            )
