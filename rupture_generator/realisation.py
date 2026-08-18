"""One rupture: the charts it happens on, and how it crossed between them.

A :class:`Realisation` maps segment name to chart, plus the frame those charts are in
and the tree saying which segment triggered which. The **same type** describes a fault
system before anything has been drawn on it and after the whole pipeline has run.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, MutableMapping

import pyproj

from rupture_generator import moment, propagation
from rupture_generator.errors import PropagationError
from rupture_generator.mesh import RuptureMesh

TRUNCATED_FRACTION = "truncated_fraction"

HYPOCENTRE_STRIKE_KM = "hypocentre_strike_km"
HYPOCENTRE_DIP_KM = "hypocentre_dip_km"


@dataclasses.dataclass
class Realisation(MutableMapping[str, RuptureMesh]):
    """A fault system, before or after annotation.

    Attributes
    ----------
    segments : dict of str to RuptureMesh
        One chart per segment.
    crs : pyproj.CRS
        The projected frame those charts are in.
    tree : propagation.Tree of str or None
        Which segment triggers which; a root maps to ``None``. Keyed by the child.
    jumps : propagation.Tree of propagation.Jump
        Where and when the front crossed onto each triggered segment.
    """

    segments: dict[str, RuptureMesh]
    crs: pyproj.CRS
    tree: propagation.Tree[str | None] = dataclasses.field(default_factory=dict)
    jumps: propagation.Tree[propagation.Jump] = dataclasses.field(default_factory=dict)

    def __getitem__(self, name: str) -> RuptureMesh:
        """Get rupture mesh for a segment with `name`."""
        return self.segments[name]

    def __setitem__(self, name: str, rupture_mesh: RuptureMesh) -> None:
        """Update a rupture mesh at `name` to `rupture_mesh`."""
        self.segments[name] = rupture_mesh

    def __delitem__(self, name: str) -> None:
        """Drop a segment. The tree is not rewritten to match."""
        del self.segments[name]

    def __iter__(self) -> Iterator[str]:
        """An iterator over segments in insertion order."""
        return iter(self.segments)

    def __len__(self) -> int:
        """How many segments rupture."""
        return len(self.segments)

    def __repr__(self) -> str:
        """The segments and what is on them."""
        fields = sorted(set().union(*(mesh.fields() for mesh in self.values())))
        return (
            f"{type(self).__name__}({', '.join(self.segments)}; "
            f"fields: {', '.join(fields) or 'none'})"
        )

    @property
    def root(self) -> str:
        """The name of the segment the rupture started on.

        Raises
        ------
        PropagationError
            If nothing has propagated yet, or if the tree has several roots.
        """
        roots = [name for name, parent in self.tree.items() if parent is None]
        match roots:
            case [root]:
                return root
            case []:
                raise PropagationError(
                    "This rupture has no root; nothing has propagated yet"
                )
            case _:
                raise PropagationError("This rupture has multiple roots")

    @property
    def hypocentre(self) -> tuple[int, int] | int:
        """The subfault the rupture nucleated at, on the root segment.

        Labelled the way that segment's own chart labels a subfault: an ``(i, j)``
        cell on a lattice, a flat face index on a triangulation.

        Raises
        ------
        KeyError
            If the root records no hypocentre, so the wavefront has not been solved.
        """
        root = self[self.root]
        return root.cell_index(
            float(root.attrs[HYPOCENTRE_STRIKE_KM]),
            float(root.attrs[HYPOCENTRE_DIP_KM]),
        )

    @property
    def moment_newton_m(self) -> float:
        """The whole event's seismic moment, newton-metres.

        What the fields carry, summed over every segment, rather than the target they
        were scaled to; the two agree by construction.

        Raises
        ------
        KeyError
            If a segment carries no slip, so the moment fold has not run.
        """
        return sum(
            moment.moment_of(mesh["slip_m"], mesh["rigidity_pa"], mesh.areas_km2())
            for mesh in self.values()
        )

    @property
    def truncated_fraction(self) -> float:
        """The worst segment's slip truncation, as a diagnostic."""
        return max(
            (
                float(mesh.attrs[TRUNCATED_FRACTION])
                for mesh in self.values()
                if TRUNCATED_FRACTION in mesh.attrs
            ),
            default=0.0,
        )

    def in_causal_order(self) -> Iterator[str]:
        """Segment names in order of propagation causality."""
        return propagation.in_topological_order(self.tree)


__all__ = [
    "HYPOCENTRE_DIP_KM",
    "HYPOCENTRE_STRIKE_KM",
    "TRUNCATED_FRACTION",
    "Realisation",
]
