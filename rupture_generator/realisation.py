"""One rupture: the charts it happens on, and how it crossed between them.

A :class:`Realisation` is a mapping of segment name to :class:`~rupture_generator.mesh.RuptureMesh`,
plus the frame those charts are in and the tree saying which segment triggered which.
The **same type** describes a fault system before anything has been drawn on it and
after the whole pipeline has run, so `pipeline.generate` is a function from a
realisation to a realisation, and a stage is a function of the same shape.

That is what lets the pipeline be a recipe rather than a machine. Every stage takes
this and returns this; the fields a stage attaches are on the charts; and the argument
about which stage runs when is the order of the lines in `generate`.

# Why it is not a MutableMapping

Reading a segment is ordinary indexing. *Replacing* one goes through
:meth:`Realisation.replace_segments`, which cannot change the key set. That one
restriction is what keeps the tree and the jumps honest: which faults exist is the
geometry's to say, fixed when the system is built, so no stage can leave the tree
naming a segment that is no longer there. A stage that wanted to add or drop a fault
would be describing a different earthquake, and that is a constructor.

# What lives here and what does not

The charts carry their own fields and their own attrs. What is here is what belongs to
the *system*: the frame, the tree, the jumps, and the moment the whole event was scaled
to. Anything a segment could answer is a property computed from the segments -- the same
argument `mesh.py` makes for never storing an area it can compute.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping

import pyproj

from rupture_generator import propagation
from rupture_generator.mesh import RuptureMesh

TRUNCATED_FRACTION = "truncated_fraction"
"""A segment's own slip-truncation diagnostic, in its chart's attrs."""

HYPOCENTRE_STRIKE_KM = "hypocentre_strike_km"
HYPOCENTRE_DIP_KM = "hypocentre_dip_km"
"""Where the rupture nucleated, in the root segment's own arc lengths.

On the chart rather than here, and under the file's own names, so the writer copies
them rather than being told them and the root needs no special case. The cell index is
derived from them -- see :attr:`Realisation.hypocentre`.
"""


@dataclasses.dataclass(frozen=True)
class Realisation(Mapping[str, RuptureMesh]):
    """A fault system, before or after annotation.

    Attributes
    ----------
    segments : dict of str to RuptureMesh
        One per segment, keyed by the name the causality tree uses. **Iteration order
        is the order the geometry built them in, and it is stable.** The moment fold
        reads patterns, rigidities and areas into parallel lists and pairs the answer
        back by position, so an order that varied between two passes would hand one
        fault another's target and leave the event total exactly right.
    crs : pyproj.CRS
        The projected frame every chart's offsets are in. It travels with the segments
        because it is what makes them mean anything -- a node position without its
        frame is three numbers -- and because carrying it here is what lets `generate`
        stop taking it: the pipeline never projects, and the writer that does can ask.
    tree : propagation.Tree
        Which segment triggered which; a root maps to ``None``. Before propagation has
        run, every segment is its own root, which is the honest description of a system
        nobody has said anything about yet.
    jumps : dict of str to propagation.Jump
        Where and when the front crossed onto each triggered segment, keyed by the
        **child**: a child has exactly one parent, so its name names the edge.
    moment_newton_m : float, optional
        What the whole event was scaled to, or ``None`` before it has been. The target
        rather than a recomputation of it -- the two agree to round-off by construction,
        and a test that recomputes the moment from the segments and compares it to this
        is asserting that closure rather than restating it.
    """

    segments: dict[str, RuptureMesh]
    crs: pyproj.CRS
    tree: propagation.Tree = dataclasses.field(default_factory=dict)
    jumps: dict[str, propagation.Jump] = dataclasses.field(default_factory=dict)
    moment_newton_m: float | None = None

    def __post_init__(self) -> None:
        """Copy the mappings, and refuse a system that is not one.

        Copied because a caller's dict would otherwise be a way to write through a
        frozen object. Each check names a mistake a stage can make, and names it here
        rather than failing later inside a traversal.

        An empty ``tree`` is normalised to the unpropagated forest -- every segment its
        own root -- so there is no second spelling of "nothing has been decided yet"
        for the rest of the package to test for.

        Raises
        ------
        ValueError
            If the system holds no segments, if the tree does not name exactly the
            segments, if a parent is not a segment, or if a jump lands on a segment
            nothing triggers.
        """
        if not self.segments:
            raise ValueError("a realisation is at least one segment")

        object.__setattr__(self, "segments", dict(self.segments))
        object.__setattr__(self, "jumps", dict(self.jumps))
        tree = dict(self.tree) if self.tree else dict.fromkeys(self.segments)
        object.__setattr__(self, "tree", tree)

        if set(tree) != set(self.segments):
            raise ValueError(
                f"the causality tree names {', '.join(sorted(tree))}, and this "
                f"rupture is {', '.join(self.segments)}"
            )
        for child, parent in tree.items():
            if parent is not None and parent not in self.segments:
                raise ValueError(
                    f"{child} is triggered by {parent}, which is not a segment of "
                    f"this rupture ({', '.join(self.segments)})"
                )
        for child in self.jumps:
            if tree.get(child) is None:
                raise ValueError(
                    f"there is a jump onto {child}, which nothing triggers"
                )

    # ------------------------------------------------------------------ reading

    def __getitem__(self, name: str) -> RuptureMesh:
        """One segment's chart, by the name the causality tree uses."""
        return self.segments[name]

    def __iter__(self) -> Iterator[str]:
        """The segment names, in the order the geometry built them."""
        return iter(self.segments)

    def __len__(self) -> int:
        """How many segments rupture."""
        return len(self.segments)

    def __repr__(self) -> str:
        """The segments and what is on them, not the charts themselves."""
        fields = sorted(set().union(*(mesh.fields() for mesh in self.values())))
        return (
            f"{type(self).__name__}({', '.join(self.segments)}; "
            f"fields: {', '.join(fields) or 'none'})"
        )

    @property
    def root(self) -> str:
        """The name of the segment the rupture started on.

        The name rather than the chart, because callers index the tree and the config
        with it.

        Raises
        ------
        ValueError
            If the system has several roots -- distinguishing a system propagation has
            not run on from a tree that genuinely disagrees with itself, because the
            first is a pipeline in the wrong order and the second is a bad tree.
        """
        roots = [name for name, parent in self.tree.items() if parent is None]
        if len(roots) == 1:
            return roots[0]
        if len(roots) == len(self.segments):
            raise ValueError(
                "nothing has propagated yet, so every segment is still its own root; "
                "the tree is decided before any field is drawn"
            )
        raise ValueError(
            f"this rupture has {len(roots)} roots ({', '.join(roots)}), and an "
            "earthquake starts in one place"
        )

    @property
    def hypocentre(self) -> tuple[int, int]:
        """The ``(i, j)`` cell the rupture nucleated at, on the root segment.

        Read off the root chart's own recorded arc lengths through
        `RuptureMesh.cell_index` -- the one narrow conversion seam, and `DEFECTS.md`
        17's exact subject. Derived rather than stored because the index and the arc
        lengths are one fact twice, and the arc lengths are what the file keeps.

        Raises
        ------
        KeyError
            If the root records no hypocentre, which is a rupture that has not been
            through the wavefront solve.
        """
        root = self[self.root]
        return root.cell_index(
            float(root.attrs[HYPOCENTRE_STRIKE_KM]),
            float(root.attrs[HYPOCENTRE_DIP_KM]),
        )

    @property
    def truncated_fraction(self) -> float:
        """The worst segment's slip truncation, as a diagnostic.

        A max rather than a mean: it answers "was the requested variation achievable
        anywhere on this rupture", and a fault that clipped badly is not excused by
        three that did not. Zero where no segment recorded one -- a point source draws
        no field, so nothing was truncated.
        """
        return max(
            (
                float(mesh.attrs[TRUNCATED_FRACTION])
                for mesh in self.values()
                if TRUNCATED_FRACTION in mesh.attrs
            ),
            default=0.0,
        )

    def in_causal_order(self) -> Iterator[str]:
        """Every segment, each after the one that triggers it.

        The order the wavefront is solved in, and the only order in the pipeline that
        is not the geometry's.
        """
        return propagation.in_topological_order(self.tree)

    # ------------------------------------------------------------------ writing

    def replace_segments(self, segments: Mapping[str, RuptureMesh]) -> Realisation:
        """This system with some of its segments replaced. Functional, never in place.

        A mapping rather than keyword arguments, because a segment name is not an
        identifier: a surface whose planes do not all share a seam is called
        ``kaikoura:0``.

        The mapping may be **partial** -- the causal solve replaces one segment at a
        time -- and it may **not introduce a name**. That refusal is the whole
        invariant: nothing else here has to check that the tree and the segments agree,
        because nothing can make them disagree. The receiver's order is kept, whatever
        order the argument arrives in, which is what stops a causal-order pass silently
        reordering the moment fold's parallel lists.

        Raises
        ------
        ValueError
            Naming the segments that are not part of this system, and listing the ones
            that are. A mistyped segment name is otherwise a silent no-op that drops a
            whole fault's fields.
        """
        unknown = set(segments) - set(self.segments)
        if unknown:
            raise ValueError(
                f"{', '.join(sorted(unknown))} is not a segment of this rupture "
                f"({', '.join(self.segments)}); replacing cannot add a fault"
            )
        return dataclasses.replace(
            self, segments={**self.segments, **dict(segments)} if segments else self.segments
        )

    def with_tree(self, tree: propagation.Tree) -> Realisation:
        """This system, with who triggered whom decided. Functional, never in place.

        The propagation's one output. The checks are in :meth:`__post_init__`, where
        every other route in is checked too.
        """
        return dataclasses.replace(self, tree=tree)

    def with_jumps(self, jumps: Mapping[str, propagation.Jump]) -> Realisation:
        """This system, with where and when the front crossed each edge recorded.

        Separate from :meth:`with_tree` because the two are decided at opposite ends of
        the pipeline: the tree before any field is drawn, the jumps from the solved
        wavefront. That separation is `propagation.py`'s stated division of labour, and
        one setter for both would hide it.
        """
        return dataclasses.replace(self, jumps=dict(jumps))

    def with_moment(self, moment_newton_m: float) -> Realisation:
        """This system, with the moment it was scaled to. Functional, never in place."""
        return dataclasses.replace(self, moment_newton_m=moment_newton_m)


__all__ = [
    "HYPOCENTRE_DIP_KM",
    "HYPOCENTRE_STRIKE_KM",
    "TRUNCATED_FRACTION",
    "Realisation",
]
