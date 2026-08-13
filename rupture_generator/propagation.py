"""Which faults rupture, in what order, and where the front crosses between them.

A multi-segment earthquake is a tree: each fault has exactly one triggering parent,
and the root is where the rupture nucleated. This module builds that tree and then
finds, for every edge of it, the point and time at which the front jumps.

**The tree decides who triggers whom; the wavefront decides where and when.** The tree
comes from fault separations -- sampled from them or stated outright -- and is fixed
before any field is drawn. The jump points come from the solved wavefront on the
parent, so a rupture that reaches the far end of a fault early jumps from there rather
than from wherever the two faults happen to be closest.

Arriving somewhere is necessary for a jump and nowhere near sufficient. What triggers
the next segment is the stress concentration of an *arrested* rupture tip, so
:func:`causal_jump` searches only the parent's edge cells, where the front runs out of
fault; the citations are at that function. Those two rules bracket the failure modes on
either side -- closest approach jumps too late and at the surface, earliest arrival
over every cell jumps too early, from the wake of a front that never stopped. Neither
needs a minimum jump depth to fix, and none is configured here.

Distances are measured in the projected frame, where a distance is an exact identity
rather than an approximation carrying a curvature error. There is no geodesy here.

Wilson's algorithm and Kruskal's are written out rather than imported: forty lines and
twenty-five, against a dependency whose sampler semantics would become part of this
package's contract. Writing them here also lets them take a `numpy.random.Generator`,
which is what makes a sampled tree reproducible from the event seed. It buys a sharper
test, too -- enumerating every spanning tree and weighting it by
``prod(w) * prod(1 - w)`` is a genuinely different algorithm from a loop-erased random
walk, so the enumeration is a *reference* rather than a second reading of the sampler.
"""

from __future__ import annotations

import dataclasses
import graphlib
import itertools
import math
from typing import TYPE_CHECKING, Protocol

import numpy as np

from rupture_generator import moment

if TYPE_CHECKING:
    from collections.abc import Iterator

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]


class Chart(Protocol):
    """What this module asks of a fault segment, and no more.

    A protocol rather than a concrete mesh type because every question here is about
    positions and about where the fault stops, and neither depends on whether the
    segment is a lattice or a triangulation. Both mesh containers satisfy it as they
    stand.
    """

    @property
    def surface(self) -> str:
        """The segment's name, for the message when two faults are too far apart."""
        ...

    @property
    def origin_km(self) -> tuple[float, float]:
        """The easting and northing every position on this chart is an offset from."""
        ...

    def centres(self) -> FloatArray:
        """Subfault centres, positions with depth last."""
        ...

    def boundary_faces(self) -> IntArray:
        """Flat indices of the subfaults where the fault runs out."""
        ...

    def cell_key(self, flat_index: int) -> tuple[int, int] | int:
        """How this chart labels the subfault at a flat index."""
        ...


type Tree[T] = dict[str, T]
"""A rupture causality tree: each fault mapped to the fault that triggered it, and
the root mapped to ``None``."""

MAX_JUMP_KM = 15.0
"""How far a rupture front can jump between faults at all.

Beyond this the Shaw-Dieterich probability is small enough that including the edge
only adds numerical noise to the sampler, and the fault pair is not a jump anyone
models. Edges longer than this are removed before the graph is built rather than
given a tiny weight, so a fault beyond reach of every other is *disconnected* and
says so.
"""

PROBABILITY_CAP = 0.99
"""The largest jump probability an edge may carry.

The sampler reweights each edge to ``w / (1 - w)``, which diverges as ``w``
approaches one. A pair of faults a metre apart is certain to jump for every practical
purpose, and capping keeps that certainty finite.
"""


def shaw_dieterich(
    distance_km: float | FloatArray,
    *,
    d0_km: float = 3.0,
    delta_km: float = 1.0,
) -> float | FloatArray:
    """The probability that a rupture jumps a gap of a given width.

    .. math:: P(d) = \\min\\left(1, e^{-(d - \\delta) / d_0}\\right)

    Shaw, B. E., & Dieterich, J. H. (2007). Probabilities for jumping fault segment
    stepovers. *Geophysical Research Letters* **34**(1).

    Certain within ``delta_km`` -- faults that nearly touch always break together --
    and decaying with a characteristic length ``d0_km`` beyond it.
    """
    return np.minimum(1.0, np.exp(-(np.asarray(distance_km) - delta_km) / d0_km))


@dataclasses.dataclass(frozen=True)
class JumpGraph:
    """Faults, and how likely the rupture is to jump between each pair.

    Attributes
    ----------
    faults : tuple of str
        The fault names, in a fixed order that indexes ``weights``.
    weights : FloatArray
        Symmetric ``(n, n)``, with **zero meaning no edge at all** rather than an
        impossible one. A fault out of reach of every other makes the graph
        disconnected, which is a refusal.
    """

    faults: tuple[str, ...]
    weights: FloatArray

    def __post_init__(self) -> None:
        """Check the weights describe a symmetric graph over these faults."""
        count = len(self.faults)
        if self.weights.shape != (count, count):
            raise ValueError(f"the weights are {self.weights.shape} for {count} faults")
        if not np.allclose(self.weights, self.weights.T):
            raise ValueError("a jump is as likely in one direction as the other")

    @property
    def edges(self) -> list[tuple[int, int, float]]:
        """Every present edge once, as ``(u, v, weight)`` with ``u < v``."""
        return [
            (u, v, float(self.weights[u, v]))
            for u, v in itertools.combinations(range(len(self.faults)), 2)
            if self.weights[u, v] > 0.0
        ]

    def is_connected(self) -> bool:
        """Whether every fault is reachable from every other."""
        count = len(self.faults)
        if count == 0:
            return False
        seen = {0}
        stack = [0]
        while stack:
            node = stack.pop()
            for neighbour in range(count):
                if neighbour not in seen and self.weights[node, neighbour] > 0.0:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return len(seen) == count


def jump_graph(
    distances_km: dict[tuple[str, str], float],
    faults: list[str],
    *,
    d0_km: float = 3.0,
    delta_km: float = 1.0,
    max_jump_km: float = MAX_JUMP_KM,
) -> JumpGraph:
    """Turn fault separations into jump probabilities.

    Parameters
    ----------
    distances_km : dict
        Keyed by unordered fault pairs -- either ordering is accepted -- giving the
        closest approach between them in kilometres.
    faults : list of str
        Every fault, in the order the graph will index them.
    d0_km, delta_km, max_jump_km : float
        Shaw-Dieterich parameters, and the gap beyond which a pair gets no edge.
    """
    count = len(faults)
    index = {name: position for position, name in enumerate(faults)}
    weights = np.zeros((count, count), dtype=np.float64)

    for (first, second), distance_km in distances_km.items():
        if first == second or distance_km >= max_jump_km:
            continue
        probability = min(
            float(shaw_dieterich(distance_km, d0_km=d0_km, delta_km=delta_km)),
            PROBABILITY_CAP,
        )
        u, v = index[first], index[second]
        weights[u, v] = weights[v, u] = probability

    return JumpGraph(tuple(faults), weights)


def _sampling_weights(graph: JumpGraph) -> FloatArray:
    """The edge weights that make a weighted spanning-tree sampler give the right trees.

    The distribution wanted is

    .. math:: P(T) \\propto \\prod_{e \\in T} w(e) \\prod_{e \\notin T} (1 - w(e))

    -- a tree is likely when the jumps it makes are likely *and* the jumps it does
    not make are unlikely. Factor out the constant:

    .. math::
        \\prod_{e \\in T} w \\prod_{e \\notin T} (1 - w)
        = \\left[\\prod_{\\text{all } e} (1 - w)\\right]
          \\prod_{e \\in T} \\frac{w}{1 - w}

    The bracket does not depend on ``T``, so sampling a spanning tree with probability
    proportional to the product of ``w / (1 - w)`` over its edges gives exactly the
    distribution above. That is a weighted uniform spanning tree, which is what
    :func:`sample_tree` draws.
    """
    weights = np.zeros_like(graph.weights)
    present = graph.weights > 0.0
    weights[present] = graph.weights[present] / (1.0 - graph.weights[present])
    return weights


def sample_tree(graph: JumpGraph, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Draw a spanning tree, with each tree's probability what the model says.

    Wilson's algorithm: grow the tree by loop-erased random walks. Starting from any
    vertex not yet in the tree, walk at random -- stepping to a neighbour with
    probability proportional to that edge's weight -- until reaching a vertex already
    in the tree, erasing loops as they close by simply overwriting each vertex's
    onward step. The path that survives is added, and the process repeats.

    Wilson, D. B. (1996). Generating random spanning trees more quickly than the
    cover time. *STOC '96*. The algorithm samples a spanning tree with probability
    proportional to the product of its edge weights, which with the reweighting in
    :func:`_sampling_weights` is the distribution the jump model asks for.

    Returns
    -------
    list of tuple
        The tree's edges as ``(u, v)`` index pairs, undirected and unrooted.

    Raises
    ------
    ValueError
        If the graph is disconnected: there is no tree to sample, and returning a
        forest would be a rupture that started twice.
    """
    if not graph.is_connected():
        raise ValueError(
            "these faults do not form a connected system: at least one is beyond "
            f"{MAX_JUMP_KM} km of every other, so no rupture reaches it. Remove it, "
            "or generate it as its own earthquake"
        )

    weights = _sampling_weights(graph)
    count = len(graph.faults)
    if count == 1:
        return []

    # The walk's step distribution, per vertex.
    totals = weights.sum(axis=1)

    in_tree = np.zeros(count, dtype=bool)
    next_vertex = np.full(count, -1, dtype=np.int64)
    in_tree[0] = True

    for start in range(1, count):
        if in_tree[start]:
            continue
        # Walk until the tree is reached, overwriting the onward step at each vertex.
        # Overwriting *is* the loop erasure: a revisited vertex forgets the excursion
        # it made last time.
        walker = start
        while not in_tree[walker]:
            step = rng.random() * totals[walker]
            cumulative = np.cumsum(weights[walker])
            neighbour = int(np.searchsorted(cumulative, step, side="right"))
            next_vertex[walker] = min(neighbour, count - 1)
            walker = next_vertex[walker]
        # Retrace the surviving path and adopt it.
        walker = start
        while not in_tree[walker]:
            in_tree[walker] = True
            walker = next_vertex[walker]

    return [
        (vertex, int(next_vertex[vertex])) for vertex in range(count) if vertex != 0
    ]


def maximum_likelihood_tree(graph: JumpGraph) -> list[tuple[int, int]]:
    """The single most likely tree, rather than a draw from the distribution.

    Maximising ``prod w / (1 - w)`` over trees is maximising the sum of
    ``log w - log(1 - w)``, which is a maximum spanning tree under those edge scores,
    so Kruskal's algorithm gives it exactly. Used when a campaign wants the modal
    scenario rather than a sample.

    Returns
    -------
    list of tuple
        The tree's edges as ``(u, v)`` index pairs.

    Raises
    ------
    ValueError
        If the graph is disconnected.
    """
    if not graph.is_connected():
        raise ValueError(
            "these faults do not form a connected system, so there is no spanning "
            "tree to maximise over"
        )

    scored = sorted(
        (
            (math.log(weight) - math.log1p(-weight), u, v)
            for u, v, weight in graph.edges
        ),
        reverse=True,
    )

    parent = list(range(len(graph.faults)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    edges: list[tuple[int, int]] = []
    for _score, u, v in scored:
        root_u, root_v = find(u), find(v)
        if root_u != root_v:
            parent[root_u] = root_v
            edges.append((u, v))
    return edges


def root_tree(
    faults: tuple[str, ...], edges: list[tuple[int, int]], root: str
) -> Tree[str | None]:
    """Orient an undirected tree away from the fault the rupture started on.

    Returns
    -------
    Tree of str or None
        Each fault mapped to its triggering parent; the root mapped to ``None``.

    Raises
    ------
    ValueError
        If the root is not one of the faults, or some fault is unreachable from it.
    """
    if root not in faults:
        raise ValueError(
            f"the rupture starts on {root!r}, which is not one of {', '.join(faults)}"
        )

    neighbours: dict[int, list[int]] = {index: [] for index in range(len(faults))}
    for u, v in edges:
        neighbours[u].append(v)
        neighbours[v].append(u)

    start = faults.index(root)
    tree: Tree[str | None] = {root: None}
    stack = [start]
    seen = {start}
    while stack:
        node = stack.pop()
        for neighbour in neighbours[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                tree[faults[neighbour]] = faults[node]
                stack.append(neighbour)

    if len(seen) != len(faults):
        missing = sorted(set(faults) - set(tree))
        raise ValueError(
            f"{', '.join(missing)} cannot be reached from {root!r}, so the tree does "
            "not describe one rupture"
        )
    return tree


def check_tree(tree: Tree[str | None], faults: list[str], root: str) -> None:
    """Refuse a stated tree that is not one.

    Every refusal here is a mistake someone can make in a config file, and each names
    what is wrong rather than failing later with a cycle in a traversal.

    Raises
    ------
    ValueError
        If a parent is not a fault, a fault is missing, the root disagrees with where
        the rupture starts, or the parent map contains a cycle.
    """
    known = set(faults)
    for child, parent in tree.items():
        if child not in known:
            raise ValueError(
                f"the propagation names {child!r}, which is not a surface in this "
                f"geometry ({', '.join(sorted(known))})"
            )
        if parent is not None and parent not in known:
            raise ValueError(
                f"{child!r} is triggered by {parent!r}, which is not a surface in "
                "this geometry"
            )

    roots = [child for child, parent in tree.items() if parent is None]
    if len(roots) != 1:
        raise ValueError(
            f"the propagation has {len(roots)} faults with no parent "
            f"({', '.join(sorted(roots))}); a rupture starts in one place"
        )
    if roots[0] != root:
        raise ValueError(
            f"the propagation is rooted at {roots[0]!r} but the hypocentre is on "
            f"{root!r}. One of the two is wrong, and this refuses rather than "
            "choosing"
        )

    missing = known - set(tree)
    if missing:
        raise ValueError(
            f"{', '.join(sorted(missing))} has no entry in the propagation, so "
            "nothing says whether it ruptures"
        )

    # Walk each fault to the root; a cycle never gets there.
    for fault in tree:
        seen = set()
        node: str | None = fault
        while node is not None:
            if node in seen:
                raise ValueError(
                    f"the propagation loops: {fault!r} is its own ancestor, so no "
                    "fault in that loop is ever triggered"
                )
            seen.add(node)
            node = tree[node]


def in_topological_order(tree: Tree[str | None]) -> Iterator[str]:
    """Every fault, each after the one that triggers it.

    The order the pipeline solves in: a child's wavefront needs its parent's, so a
    parent is always finished first.
    """
    sorter = graphlib.TopologicalSorter()
    for node, parent in tree.items():
        if parent is not None:
            sorter.add(node, parent)
        else:
            sorter.add(node)

    yield from sorter.static_order()


# ============================================================================
# Jump delay -- a model, not a formula
# ============================================================================


class JumpDelay(Protocol):
    """How long a rupture takes to cross a gap of a given width, from a given depth.

    Two arguments, because the rock in the gap is not the same rock at every depth and
    a model that pretends otherwise makes a shallow crossing look as fast as a deep
    one. Anything else a model needs it closes over when it is built.
    """

    def __call__(self, distance_km: FloatArray, depth_km: FloatArray) -> FloatArray:
        """Seconds, elementwise over gap widths and the depths they are left from.

        ``depth_km`` is the **departure** depth: the front leaves an arrested tip at
        that depth and crosses rock described by it.

        Must be **non-negative, and monotone in distance at fixed depth**: from one
        departure point, a wider gap never crosses faster than a narrower one.
        :func:`causal_jump` relies on that to search nearest neighbours rather than
        every pair, which is what makes a million-subfault rupture tractable.
        """
        ...


@dataclasses.dataclass(frozen=True)
class Instantaneous:
    """The front crosses the gap in no time.

    Not physical, and useful precisely for that: the jump point is wherever the
    parent's front arrives earliest against the geometry alone, which makes it the
    control case for testing that a delay model changed something.
    """

    def __call__(self, distance_km: FloatArray, depth_km: FloatArray) -> FloatArray:
        """Zero, shaped like the distances."""
        del depth_km
        return np.zeros_like(distance_km)


@dataclasses.dataclass(frozen=True)
class DistanceOverVelocity:
    """The gap is crossed at the shear speed of the depth the front left from.

    The default model. The gap is on neither fault, so neither fault's *sampled*
    materials describe it; the shared 1-D velocity model does, read at the departure
    depth -- the front leaves an arrested tip and crosses the rock that tip is in.

    There is deliberately no constant-speed variant: a mean over parts of both faults
    nowhere near the gap is what let a crossing at the surface trace look as fast as
    one at seismogenic depth.
    """

    bottom_depth_km: FloatArray
    shear_speed_km_s: FloatArray

    def __post_init__(self) -> None:
        """Refuse a model the front cannot cross a gap at."""
        speeds = np.asarray(self.shear_speed_km_s, dtype=np.float64)
        if speeds.shape != np.asarray(self.bottom_depth_km, dtype=np.float64).shape:
            raise ValueError(
                f"the velocity model has {speeds.size} shear speeds for "
                f"{np.size(self.bottom_depth_km)} layer bottoms"
            )
        if not speeds.size:
            raise ValueError(
                "a jump crosses rock, and this velocity model has no layers"
            )
        if not (speeds > 0.0).all():
            raise ValueError(
                f"a jump crosses the gap at {float(speeds.min())} km/s, which never "
                "arrives"
            )

    def __call__(self, distance_km: FloatArray, depth_km: FloatArray) -> FloatArray:
        """Seconds: the gap width over the shear speed at the departure depth."""
        layer = moment.layer_of(depth_km, self.bottom_depth_km)
        speed = np.asarray(self.shear_speed_km_s, dtype=np.float64)[layer]
        return np.asarray(distance_km) / speed


# ============================================================================
# The causal jump
# ============================================================================


@dataclasses.dataclass(frozen=True)
class Jump:
    """Where and when a rupture front crossed from one fault to the next.

    Attributes
    ----------
    parent_cell, child_cell : tuple of int, or int
        The subfaults the front left from and arrived at, labelled the way their own
        chart labels a subfault: ``(i, j)`` on a lattice, a flat face index on a
        triangulation. Either indexes that chart's own fields, which is the property
        the label exists for -- see
        :meth:`~rupture_generator.mesh.RuptureMesh.cell_key`.
    distance_km : float
        The gap it crossed.
    departure_s : float
        When the front reached the parent's jump-off point.
    arrival_s : float
        When it reached the child -- the departure plus the delay, and the seed time
        the child's wavefront is solved from.
    from_edge : bool
        Whether the front left from an edge of the parent. ``False`` records that no
        edge cell was within reach and the search fell back to the whole chart -- a
        child sitting off the *face* of its parent rather than off an end. Carried on
        the jump rather than logged, because a fallback nobody can see is a second
        model running silently.
    """

    parent_cell: tuple[int, int] | int
    child_cell: tuple[int, int] | int
    distance_km: float
    departure_s: float
    arrival_s: float
    from_edge: bool = True


def causal_jump(
    parent: Chart,
    parent_wavefront_s: FloatArray,
    child: Chart,
    delay: JumpDelay,
    *,
    parent_onset_s: FloatArray | None = None,
    max_distance_km: float = MAX_JUMP_KM,
) -> Jump:
    """Where the front crosses to the child fault, and when it gets there.

    .. math::
        (p^*, c^*) = \\arg\\min_{p \\in \\partial P,\\; c \\in C}
        \\left[\\, t_P(p) + \\mathrm{delay}\\left(\\|X_P(p) - X_C(c)\\|,\\;
        z_P(p)\\right) \\right]

    **The front jumps from where it arrests, not from wherever it passes.** That is
    the ``\\partial``: candidates are the parent's edge cells, the places the rupture
    runs out of fault and stops. The trigger for a jump is the stress concentration of
    an arrested rupture tip -- its stopping phase -- rather than the wavefront sweeping
    by earlier. Oglesby (2008), *BSSA* **98**, 440, found jumps succeed when donor slip
    terminates abruptly and fail when it tapers; Kase & Kuge (2001), *GJI* **147**, 330,
    found triggering follows the front reaching the fault edge by about a second; Fliss,
    Bhat, Dmowska & Rice (2005), *JGR* **110**, B06312, work the mechanism out for the
    Landers backward branch, where the rupture arrests, radiates, and re-nucleates.

    Without the restriction the minimisation takes a cell deep in the wake of the
    front -- far from anywhere the rupture stops, and radiating essentially nothing
    towards the child -- because a chord through intact rock at the shear speed always
    beats the front crawling along the fault at a fraction of it. First arrival is
    necessary for a jump and nowhere near sufficient, and treating it as sufficient is
    what made every jump too early.

    **No depth rule, and none is needed.** All four edges are candidates, the surface
    trace included, and the arrival time decides between them: the shallow reduction in
    :mod:`rupture_generator.timing` already makes the surface trace a late arrival, and
    the delay is charged at the shear speed of the depth the front leaves from, which is
    lowest there too. A jump that goes deep does so because the earthquake got there
    first, not because a minimum depth was configured.

    **Not the closest pair either.** The minimisation is over arrival time, so a front
    that reaches a distant edge of the parent early will jump from there in preference
    to a nearer edge it reaches late. Closest approach is a fact about the geometry;
    this is a fact about the earthquake, and only one of them knows which way the front
    was travelling.

    **Only pairs within the jump limit are candidates**, and that bound is physics
    rather than an optimisation. A rupture crosses a gap at roughly the shear speed
    but propagates along a fault at a *fraction* of it, so without the bound the
    minimisation discovers that leaving from the hypocentre and crossing tens of
    kilometres of intact rock beats travelling there along the fault -- measured at a
    28 km "jump" on the shipped two-segment example, arriving before the front had
    covered a third of the first fault. The gap model is fitted to stepovers of a few
    kilometres and says nothing about that. The same limit decides which faults are
    connected at all, so any pair the tree contains has at least one candidate.

    The search is over nearest neighbours rather than over pairs, and that is exact
    rather than an approximation: from one departure point the delay never decreases
    with distance, so that point's earliest arrival is always to the closest point on
    the other fault. It is also what makes the stage tractable -- the two largest
    faults of the shipped scenario have 145 billion pairs between them and 37,740
    nearest neighbours, and restricting to edges cuts even that by two orders of
    magnitude.

    **Where a front runs out of fault is the chart's own answer**, not this function's:
    a lattice's perimeter is its first and last rows and columns, and a triangulation's
    is the faces with an edge no second face shares. Both spell it ``boundary_faces()``
    and both return flat indices, so nothing here branches on which kind of chart it
    has, and neither does the label a :class:`Jump` records -- see
    :meth:`~rupture_generator.mesh.RuptureMesh.cell_key`.

    Parameters
    ----------
    parent, child : Chart
        The two charts. Both hold offsets from their own surface origins, so the
        origins are added back here before differencing -- two faults digitised
        against different origins would otherwise be compared in different frames.
    parent_wavefront_s : FloatArray
        When the front reached each of the parent's subfaults. **The field the choice
        is made on**, and it should be the solved wavefront rather than the perturbed
        onset: an argmin over a hundred thousand perturbed cells is an order statistic
        that selects the perturbation's negative tail, not the shape of the front.
    delay : JumpDelay
    parent_onset_s : FloatArray, optional
        **The field the clock is read from**, when it differs from the one the choice
        is made on. Choosing *where* the front left is a question about the wavefront;
        choosing *when* it left is a question about that one cell, and there the
        perturbation is part of the answer. Defaults to the wavefront, so a caller
        with one field passes one field.
    max_distance_km : float
        The widest gap a jump may cross.

    Returns
    -------
    Jump

    Raises
    ------
    ValueError
        If no pair is within the limit, which means these two faults are not close
        enough to be part of one rupture at all.
    """
    parent_origin = np.array([*parent.origin_km, 0.0])
    child_origin = np.array([*child.origin_km, 0.0])

    wavefront = np.asarray(parent_wavefront_s, dtype=np.float64).reshape(-1)
    departures = (
        wavefront
        if parent_onset_s is None
        else np.asarray(parent_onset_s, dtype=np.float64).reshape(-1)
    )
    all_points = (parent.centres() + parent_origin).reshape(-1, 3)
    to_points = (child.centres() + child_origin).reshape(-1, 3)

    # **Only each candidate's nearest child can win.** From one departure point the
    # depth is fixed, so the delay there is a function of distance alone and never
    # decreases with it:
    #
    #     min_c [ t_P(p) + delay(d(p, c), z(p)) ]
    #         =  t_P(p) + delay( min_c d(p, c), z(p) )
    #
    # which turns a search over every pair into one nearest-neighbour query per
    # candidate. Exact for any delay monotone in distance at fixed depth, which is the
    # one thing :class:`JumpDelay` asks of an implementation.
    from scipy.spatial import cKDTree

    tree = cKDTree(to_points)
    candidates = parent.boundary_faces()
    candidate_points = all_points[candidates]
    nearest_km, nearest = tree.query(candidate_points, k=1, workers=-1)
    from_edge = bool((nearest_km <= max_distance_km).any())

    if not from_edge:
        # A child off the *face* of its parent rather than off an end -- a splay, or a
        # fault passing beneath the middle of another. The front never arrests within
        # reach, so there is no arrest to jump from and the whole chart is searched
        # instead. Recorded on the Jump rather than passed over in silence.
        candidates = np.arange(all_points.shape[0])
        candidate_points = all_points
        nearest_km, nearest = tree.query(candidate_points, k=1, workers=-1)

    reachable = nearest_km <= max_distance_km
    if not reachable.any():
        raise ValueError(
            f"{parent.surface!r} and {child.surface!r} come no closer than "
            f"{float(nearest_km.min()):.2f} km, past the {max_distance_km:.1f} km a "
            "rupture jumps, so the front never crosses between them"
        )

    delays_s = delay(nearest_km, candidate_points[:, 2])

    # Chosen on the wavefront, timed on the onset. The argmin picks the cell; the cell
    # is then asked when the rupture actually got there.
    chosen = int(
        np.argmin(np.where(reachable, wavefront[candidates] + delays_s, np.inf))
    )
    from_cell = int(candidates[chosen])
    to_cell = int(nearest[chosen])
    departure_s = float(departures[from_cell])

    return Jump(
        parent_cell=parent.cell_key(from_cell),
        child_cell=child.cell_key(to_cell),
        distance_km=float(nearest_km[chosen]),
        departure_s=departure_s,
        arrival_s=departure_s + float(delays_s[chosen]),
        from_edge=from_edge,
    )


def closest_approach_km(first: Chart, second: Chart) -> float:
    """How near two faults come to each other, in kilometres.

    What the jump probability is a function of. Measured between cell centres rather
    than between surfaces, which understates the true closest approach by up to half
    a subfault -- immaterial against a 15 km cutoff and a 3 km decay length, and it
    keeps this the same quantity the jump search minimises over.

    Over nearest neighbours rather than over pairs: the minimum over every pair is the
    minimum over each point's nearest, so the tree gives the same number without the
    matrix. The dense form held an ``(n_first, n_second, 3)`` difference, which on the
    two largest faults of the shipped scenario is 145 billion pairs -- 3.5 TB.
    """
    from scipy.spatial import cKDTree

    from_points = (first.centres() + np.array([*first.origin_km, 0.0])).reshape(-1, 3)
    to_points = (second.centres() + np.array([*second.origin_km, 0.0])).reshape(-1, 3)
    nearest_km, _ = cKDTree(to_points).query(from_points, k=1, workers=-1)
    return float(nearest_km.min())


__all__ = [
    "MAX_JUMP_KM",
    "PROBABILITY_CAP",
    "Chart",
    "DistanceOverVelocity",
    "Instantaneous",
    "Jump",
    "JumpDelay",
    "JumpGraph",
    "Tree",
    "causal_jump",
    "check_tree",
    "closest_approach_km",
    "in_topological_order",
    "jump_graph",
    "maximum_likelihood_tree",
    "root_tree",
    "sample_tree",
    "shaw_dieterich",
]
