import dataclasses
from collections.abc import Iterator, Mapping
from graphlib import TopologicalSorter
from types import MappingProxyType

import pyproj

from rupture_generator.geometry import Geometry, GeometryError


@dataclasses.dataclass(frozen=True)
class Jump:
    """Where and when a rupture front crossed from one chart to the next.

    Cells are labelled as their own chart labels them. ``arrival_s`` is the departure
    plus the delay, and the seed time the child's onsets are solved from.
    """

    parent_cell: tuple[int, int]
    child_cell: tuple[int, int]
    distance_km: float
    departure_s: float
    arrival_s: float


@dataclasses.dataclass(frozen=True, eq=False)
class Realisation(Mapping[str, Geometry]):
    """A fault system, before or after anything is drawn on it.

    A read-only mapping from segment name to chart. The same type describes the system
    before propagation, when ``tree`` and ``jumps`` are empty, and after the whole
    pipeline has run.

    Attributes
    ----------
    segments : Mapping of str to Geometry
        One chart per segment.
    crs : pyproj.CRS
        The projected frame every chart's positions are offsets in.
    tree : Mapping of str to str or None
        Which segment triggered which, keyed by the child; a root maps to ``None``.
        Empty until propagation; otherwise names exactly the segments and is a forest.
    jumps : Mapping of str to Jump
        Where and when the front crossed onto each triggered segment, keyed by the child.
    """

    segments: Mapping[str, Geometry]
    crs: pyproj.CRS
    tree: Mapping[str, str | None] = dataclasses.field(default_factory=dict)
    jumps: Mapping[str, Jump] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the invariants, then make the mappings read-only."""
        if not self.segments:
            raise GeometryError("a realisation needs at least one segment")
        if not self.crs.is_projected:
            raise GeometryError(
                f"{self.crs.to_string()!r} is not a projected CRS; positions are "
                "kilometre offsets, so the frame has to be one"
            )
        if self.tree:
            self._check_forest()
        object.__setattr__(self, "segments", MappingProxyType(dict(self.segments)))
        object.__setattr__(self, "tree", MappingProxyType(dict(self.tree)))
        object.__setattr__(self, "jumps", MappingProxyType(dict(self.jumps)))

    def _check_forest(self) -> None:
        names = set(self.segments)
        if set(self.tree) != names:
            raise GeometryError(
                f"the tree names {sorted(self.tree)} and the segments are {sorted(names)}"
            )
        children = {child for child, parent in self.tree.items() if parent is not None}
        if set(self.jumps) != children:
            raise GeometryError(
                f"jumps are recorded for {sorted(self.jumps)} but the triggered "
                f"segments are {sorted(children)}"
            )
        for name in self.tree:
            seen: set[str] = set()
            current: str | None = name
            while current is not None:
                if current in seen:
                    raise GeometryError(f"the tree has a cycle through {current!r}")
                seen.add(current)
                current = self.tree[current]

    # ------------------------------------------------------------ the mapping

    def __getitem__(self, name: str) -> Geometry:
        """The chart for a segment."""
        return self.segments[name]

    def __iter__(self) -> Iterator[str]:
        """Segment names in insertion order."""
        return iter(self.segments)

    def __len__(self) -> int:
        """How many segments."""
        return len(self.segments)

    def __repr__(self) -> str:
        """The segments and the union of what is on them."""
        names = sorted(set().union(*(set(chart.fields) for chart in self.values())))
        return (
            f"{type(self).__name__}({', '.join(self.segments)}; "
            f"fields: {', '.join(names) or 'none'})"
        )

    # -------------------------------------------------------------- the writes

    def replace(self, **charts: Geometry) -> Realisation:
        """This realisation with some segments' charts swapped. The only write path.

        Raises
        ------
        GeometryError
            For a name that is not a segment: the tree and jumps name segments, and a
            new one would leave them inconsistent.
        """
        unknown = sorted(set(charts) - set(self.segments))
        if unknown:
            raise GeometryError(
                f"{unknown} are not segments of this realisation; it has "
                f"{sorted(self.segments)}"
            )
        return dataclasses.replace(self, segments={**self.segments, **charts})

    def propagated(
        self, tree: Mapping[str, str | None], jumps: Mapping[str, Jump]
    ) -> Realisation:
        """This realisation with its trigger forest recorded."""
        return dataclasses.replace(self, tree=tree, jumps=jumps)

    @property
    def root(self) -> str:
        """The segment the rupture started on.

        Raises
        ------
        GeometryError
            If nothing has propagated yet, or if the forest has several trees.
        """
        roots = [name for name, parent in self.tree.items() if parent is None]
        match roots:
            case [root]:
                return root
            case []:
                raise GeometryError("nothing has propagated yet, so there is no root")
            case _:
                raise GeometryError(f"the rupture has several roots: {sorted(roots)}")

    def in_causal_order(self) -> Iterator[tuple[str, str | None, Geometry]]:
        """Segment names parents-first, so a child's parent has always been visited."""
        sorter = TopologicalSorter()

        for segment in self.segments:
            if parent := self.tree.get(segment):
                sorter.add(segment, parent)
            else:
                sorter.add(segment)

        for seg in sorter.static_order():
            parent = self.tree.get(seg)
            yield seg, parent, self.segments[seg]


__all__ = [
    "Jump",
    "Realisation",
]
