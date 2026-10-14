"""The fault system a rupture runs over, with its hypocentre and triggering tree."""

import dataclasses
import graphlib
from collections.abc import Iterator, Mapping

import pyproj

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.geometry import Geometry


@dataclasses.dataclass(frozen=True)
class Hypocentre:
    """Location of rupture nucleation.

    Attributes
    ----------
    segment : str
        The segment the rupture starts on.
    strike_km : float
        Distance along strike from the chart's first edge, in kilometres.
    dip_km : float
        Distance down dip from the chart's top edge, in kilometres.
    """

    segment: str
    strike_km: float
    dip_km: float

    @classmethod
    def from_fractions(
        cls, segment: str, geometry: Geometry, strike: float, dip: float
    ) -> Hypocentre:
        """Convert a hypocentre from fractions of a chart's extent.

        Parameters
        ----------
        segment : str
            The segment the rupture starts on.
        geometry : Geometry
            That segment's chart.
        strike : float
            Fraction of the chart's length along strike, in ``[0, 1]``.
        dip : float
            Fraction of the chart's width down dip, in ``[0, 1]``.

        Returns
        -------
        Hypocentre
            The hypocentre at those fractions, as arc lengths.

        Raises
        ------
        RuptureGeneratorError
            If either fraction lies outside ``[0, 1]``.
        """
        if not (0.0 <= strike <= 1.0 and 0.0 <= dip <= 1.0):
            raise RuptureGeneratorError(
                f"a hypocentre at fractions ({strike}, {dip}) is off the fault; "
                "fractions lie in [0, 1]"
            )
        return cls(
            segment,
            strike * float(geometry.strike_arc_km[-1]),
            dip * float(geometry.dip_arc_km[-1]),
        )


@dataclasses.dataclass(frozen=True, eq=False)
class Realisation[S](Mapping[str, S]):
    """A fault system's segments, with its hypocentre and triggering tree.

    ``S`` is the type of each segment's value. Until
    :func:`~rupture_generator.rupture.generator.generate` draws the fields it's the
    segment's :class:`~rupture_generator.geometry.Geometry`, and ``generate`` turns a
    ``Realisation[Geometry]`` into a ``Realisation[SegmentRupture]``.

    Attributes
    ----------
    segments : Mapping of str to S
        Each segment's value, by name.
    crs : pyproj.CRS
        The projected frame every chart's positions are in.
    hypocentre : Hypocentre or None
        Where the rupture nucleates. A system of one segment needs nothing more.
    tree : Mapping of str to str or None
        Each segment's parent, the one that triggered it. The hypocentre's segment
        maps to ``None``. It's empty before propagation, and after that it's one tree
        over exactly the segments.
    """

    segments: Mapping[str, S]
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

    def __getitem__(self, name: str) -> S:
        """The value of the segment called ``name``."""
        return self.segments[name]

    def __iter__(self) -> Iterator[str]:
        """Segment names in insertion order."""
        return iter(self.segments)

    def __len__(self) -> int:
        """How many segments."""
        return len(self.segments)

    def __repr__(self) -> str:
        """The segment names, not what they hold."""
        return f"{type(self).__name__}({', '.join(self.segments)})"

    # -------------------------------------------------------------- the walk

    def propagated(
        self, tree: Mapping[str, str | None], hypocentre: Hypocentre
    ) -> Realisation[S]:
        """Record where the rupture starts and how it spreads.

        Parameters
        ----------
        tree : Mapping of str to str or None
            Each segment's parent, with ``None`` for the hypocentre's segment.
        hypocentre : Hypocentre
            Where the rupture nucleates.

        Returns
        -------
        Realisation
            A copy of this realisation with the tree and hypocentre set.
        """
        return dataclasses.replace(self, tree=tree, hypocentre=hypocentre)

    def with_segments[T](self, segments: Mapping[str, T]) -> Realisation[T]:
        """Replace every segment's value, with the same frame, hypocentre and tree.

        Parameters
        ----------
        segments : Mapping of str to T
            Each segment's new value, over the same names.

        Returns
        -------
        Realisation of T
            This realisation over ``segments``.

        Raises
        ------
        RuptureGeneratorError
            If ``segments`` names different segments than the tree.
        """
        return Realisation(segments, self.crs, self.hypocentre, self.tree)

    def in_causal_order(self) -> Iterator[tuple[str, str | None, S]]:
        """Walk the segments parents-first.

        Each segment comes after the one that triggered it, so a caller drawing
        them in this order has always drawn the parent already.

        Yields
        ------
        tuple of (str, str or None, S)
            The segment's name, its parent's name (``None`` for the hypocentre's
            segment) and what it holds.

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
