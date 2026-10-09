import dataclasses
import graphlib
from collections.abc import Iterator, Mapping

import pyproj

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.geometry import Geometry


@dataclasses.dataclass(frozen=True)
class Hypocentre:
    """Where the rupture nucleated: a segment, and two arc lengths on its chart.

    Arc lengths rather than indices, so the hypocentre survives the chart being recut
    at a different resolution.
    """

    segment: str
    strike_km: float
    dip_km: float


@dataclasses.dataclass(frozen=True, eq=False)
class Realisation(Mapping[str, Geometry]):
    """A fault system: its charts, where the rupture starts, and the order it spreads.

    A read-only mapping from segment name to chart.

    Attributes
    ----------
    segments : Mapping of str to Geometry
        One chart per segment.
    crs : pyproj.CRS
        The projected frame every chart's positions are in.
    hypocentre : Hypocentre or None
        Where the rupture nucleates. A lone segment needs only this to be drawn.
    tree : Mapping of str to str or None
        Which segment triggered which, keyed by the child, the hypocentre's segment
        mapped to ``None``. Empty until propagation; otherwise one tree over exactly
        the segments.
    """

    segments: Mapping[str, Geometry]
    crs: pyproj.CRS
    hypocentre: Hypocentre | None = None
    tree: Mapping[str, str | None] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the invariants."""
        if not self.segments:
            raise RuptureGeneratorError("a realisation needs at least one segment")
        if not self.crs.is_projected:
            raise RuptureGeneratorError(
                f"{self.crs.to_string()!r} is not a projected CRS; positions are "
                "kilometres, so the frame has to be one"
            )
        if self.hypocentre and self.hypocentre.segment not in self.segments:
            raise RuptureGeneratorError(
                f"the rupture starts on {self.hypocentre.segment!r}, which is not one "
                f"of {sorted(self.segments)}"
            )
        if self.tree:
            self._check_tree()

    def _check_tree(self) -> None:
        names = set(self.segments)
        if set(self.tree) != names or not set(self.tree.values()) <= names | {None}:
            raise RuptureGeneratorError(
                f"the tree names {sorted(self.tree)} and the segments are {sorted(names)}"
            )
        roots = [name for name, parent in self.tree.items() if parent is None]
        start = self.hypocentre.segment if self.hypocentre else None
        if roots != [start]:
            raise RuptureGeneratorError(
                f"the tree is rooted at {roots} and the rupture starts on {start!r}; a "
                "rupture is one tree grown from its hypocentre"
            )
        try:
            self._sorter().prepare()
        except graphlib.CycleError as cycle:
            raise RuptureGeneratorError(
                f"the tree has a cycle through {cycle.args[1]}"
            ) from None

    def _sorter(self) -> graphlib.TopologicalSorter:
        return graphlib.TopologicalSorter(
            {child: [parent] if parent else [] for child, parent in self.tree.items()}
        )

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
        """The segments, not the charts."""
        return f"{type(self).__name__}({', '.join(self.segments)})"

    # -------------------------------------------------------------- the walk

    def propagated(
        self, tree: Mapping[str, str | None], hypocentre: Hypocentre
    ) -> Realisation:
        """This realisation with where it starts and how it spreads recorded."""
        return dataclasses.replace(self, tree=tree, hypocentre=hypocentre)

    def in_causal_order(self) -> Iterator[tuple[str, str | None, Geometry]]:
        """Segments parents-first, so a child's parent has always been visited.

        Raises
        ------
        RuptureGeneratorError
            If there is no hypocentre, or several segments and no tree to order them.
        """
        if self.hypocentre is None:
            raise RuptureGeneratorError(
                "the realisation has no hypocentre to start from"
            )
        if not self.tree:
            if len(self.segments) > 1:
                raise RuptureGeneratorError(
                    f"{sorted(self.segments)} have not been propagated, so nothing says "
                    "which triggers which"
                )
            yield self.hypocentre.segment, None, self.segments[self.hypocentre.segment]
            return
        for name in self._sorter().static_order():
            yield name, self.tree[name], self.segments[name]


__all__ = ["Hypocentre", "Realisation"]
