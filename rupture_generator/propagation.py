"""Which faults rupture, in what order, and where the front crosses between them.

A multi-segment earthquake is a tree: each fault has exactly one triggering parent, and
the root is where the rupture nucleated. The tree comes from fault separations and is
fixed before any field is drawn; the jump points come from the solved wavefront on the
parent, so a rupture that reaches the far end of a fault early jumps from there rather
than from wherever the two faults happen to be closest.

Distances are measured in the projected frame, where a distance is an exact identity
rather than an approximation carrying a curvature error. There is no geodesy here.

References
----------
Kase, Y., & Kuge, K. (2001). Rupture propagation beyond fault discontinuities:
significance of fault strike and location.
*Geophysical Journal International*, 147(2), 330-342.

Oglesby, D. D. (2008). Rupture termination and jump on parallel offset faults.
*Bulletin of the Seismological Society of America*, 98(1), 440-447.

Shaw, B. E., & Dieterich, J. H. (2007). Probabilities for jumping fault segment
stepovers. *Geophysical Research Letters*, 34(1), L01307.

Wilson, D. B. (1996). Generating random spanning trees more quickly than the cover
time. *Proceedings of the 28th ACM Symposium on Theory of Computing*, 296-303.
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
    """What this module asks of a fault segment: positions, and where it stops."""

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

Beyond this the Shaw & Dieterich (2007) probability only adds noise to the sampler, so
longer edges are removed rather than given a tiny weight and a fault beyond reach of
every other is *disconnected*.
"""

PROBABILITY_CAP = 0.99
"""The largest jump probability an edge may carry.

The sampler reweights each edge to ``w / (1 - w)``, which diverges as ``w`` approaches
one; the cap keeps a pair of faults a metre apart finite.
"""


def shaw_dieterich(
    distance_km: float | FloatArray,
    *,
    d0_km: float = 3.0,
    delta_km: float = 1.0,
) -> float | FloatArray:
    """The probability that a rupture jumps a gap of a given width.

    Shaw & Dieterich (2007): certain within ``delta_km`` and decaying with
    characteristic length ``d0_km`` beyond it.

    .. math:: P(d) = \\min\\left(1, e^{-(d - \\delta) / d_0}\\right)
    """
    return np.minimum(1.0, np.exp(-(np.asarray(distance_km) - delta_km) / d0_km))


@dataclasses.dataclass(frozen=True)
class JumpGraph:
    """Faults, and how likely the rupture is to jump between each pair.

    ``weights`` is symmetric ``(n, n)`` over ``faults`` in order, and zero means no
    edge at all rather than an impossible one.
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

    ``distances_km`` gives closest approach in kilometres, keyed by fault pairs in
    either ordering; ``faults`` fixes the order the graph indexes them in. Every other
    argument is a length in kilometres.
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

    and its constant factor comes out, leaving a weighted uniform spanning tree over
    ``w / (1 - w)`` -- which is what :func:`sample_tree` draws.

    .. math::
        \\prod_{e \\in T} w \\prod_{e \\notin T} (1 - w)
        = \\left[\\prod_{\\text{all } e} (1 - w)\\right]
          \\prod_{e \\in T} \\frac{w}{1 - w}
    """
    weights = np.zeros_like(graph.weights)
    present = graph.weights > 0.0
    weights[present] = graph.weights[present] / (1.0 - graph.weights[present])
    return weights


def sample_tree(graph: JumpGraph, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Draw a spanning tree, with each tree's probability what the model says.

    Wilson (1996): from any vertex not yet in the tree, walk at random -- stepping to a
    neighbour with probability proportional to that edge's weight -- until reaching a
    vertex already in the tree, overwriting each vertex's onward step to erase loops as
    they close. The tree comes out with probability proportional to the product of its
    edge weights, which under :func:`_sampling_weights` is the distribution wanted. Its
    edges are returned as ``(u, v)`` index pairs, undirected and unrooted.

    Raises
    ------
    ValueError
        If the graph is disconnected: returning a forest would be a rupture that
        started twice.
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
    ``log w - log(1 - w)``, so Kruskal's maximum spanning tree gives it exactly. Its
    edges come back as ``(u, v)`` index pairs.

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
    """Every fault, each after the one that triggers it: the pipeline's solve order."""
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
    """How long a rupture takes to cross a gap of a given width, from a given depth."""

    def __call__(self, distance_km: FloatArray, depth_km: FloatArray) -> FloatArray:
        """Seconds, elementwise over gap widths and the **departure** depths.

        Must be non-negative and monotone in distance at fixed depth, which is what
        lets :func:`causal_jump` search nearest neighbours rather than every pair.
        """
        ...


@dataclasses.dataclass(frozen=True)
class Instantaneous:
    """The front crosses the gap in no time: the control case, not a physical one."""

    def __call__(self, distance_km: FloatArray, depth_km: FloatArray) -> FloatArray:
        """Zero, shaped like the distances."""
        del depth_km
        return np.zeros_like(distance_km)


@dataclasses.dataclass(frozen=True)
class DistanceOverVelocity:
    """The gap is crossed at the shear speed of the depth the front left from.

    The default model. The gap is on neither fault, so the shared 1-D velocity model
    describes it rather than either fault's sampled materials.
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

    The cells are labelled as their own chart labels one: ``(i, j)`` on a lattice, a
    flat face index on a triangulation. ``arrival_s`` is the departure plus the delay,
    and the seed time the child's wavefront is solved from. ``from_edge`` is ``False``
    when no edge cell was within reach and the search fell back to the whole chart.
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

    Candidates are the parent's ``boundary_faces()``, where the front runs out of fault
    and arrests: the trigger is the stress concentration of an arrested tip rather than
    the wavefront sweeping by. Oglesby (2008) found jumps succeed when donor slip
    terminates abruptly and fail when it tapers, Kase & Kuge (2001) that triggering
    follows the front reaching the fault edge by about a second, and Fliss, Bhat,
    Dmowska & Rice (2005) work the mechanism out for the Landers backward branch.

    .. math::
        (p^*, c^*) = \\arg\\min_{p \\in \\partial P,\\; c \\in C}
        \\left[\\, t_P(p) + \\mathrm{delay}\\left(\\|X_P(p) - X_C(c)\\|,\\;
        z_P(p)\\right) \\right]

    Parameters
    ----------
    parent, child : Chart
        The two charts, whose origins are added back before differencing.
    parent_wavefront_s : FloatArray
        Seconds to each of the parent's subfaults, and the field the choice is made on.
        Pass the solved wavefront rather than the perturbed onset: an argmin over a
        hundred thousand perturbed cells selects the perturbation's negative tail.
    delay : JumpDelay
        How long the crossing takes.
    parent_onset_s : FloatArray, optional
        The field the clock is read from. Defaults to the wavefront.
    max_distance_km : float
        The widest gap a jump may cross, in kilometres.

    Raises
    ------
    ValueError
        If no pair is within the limit, so the two faults are not close enough to be
        part of one rupture at all.
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

    # Only each candidate's nearest child can win, so this is one nearest-neighbour
    # query per candidate rather than a search over pairs. Exact for any delay monotone
    # in distance at fixed depth, which is the one thing `JumpDelay` asks for.
    from scipy.spatial import cKDTree

    tree = cKDTree(to_points)
    candidates = parent.boundary_faces()
    candidate_points = all_points[candidates]
    nearest_km, nearest = tree.query(candidate_points, k=1, workers=-1)
    from_edge = bool((nearest_km <= max_distance_km).any())

    if not from_edge:
        # A child off the *face* of its parent rather than off an end -- a splay, say.
        # No arrest is within reach, so the whole chart is searched instead.
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

    # Chosen on the wavefront, timed on the onset.
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

    What the jump probability is a function of. Between cell centres rather than
    surfaces, understating the true approach by up to half a subfault -- immaterial
    against a 15 km cutoff and a 3 km decay length.
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
