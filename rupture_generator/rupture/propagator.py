"""Which faults rupture, and in what order.

A multi-segment earthquake is a tree: every fault has exactly one triggering parent,
and the root is where the rupture nucleated. The tree is fixed before any field is
drawn, because it is a statement about fault separations rather than about slip; where
and *when* the front crossed is not decided here, since a crossing time is read off
the parent's solved onsets.

Two ways to choose the tree, from one model. Each gap either transmits the rupture or
does not, with a probability that decays with its width, so a whole tree has a
probability and the two natural questions are "draw one" and "which is likeliest".
:func:`sample_tree` answers the first by Wilson's algorithm and
:func:`maximum_likelihood_tree` the second by Kruskal's.

Distances are measured in the projected frame, where they are exact identities, and by
the charts themselves, so no origin is read here.

References
----------
Kase, Y., & Kuge, K. (2001). Rupture propagation beyond fault discontinuities:
significance of fault strike and location. *Geophysical Journal International*,
147(2), 330-342.

Oglesby, D. D. (2008). Rupture termination and jump on parallel offset faults.
*Bulletin of the Seismological Society of America*, 98(1), 440-447.

Shaw, B. E., & Dieterich, J. H. (2007). Probabilities for jumping fault segment
stepovers. *Geophysical Research Letters*, 34(1), L01307.

Wilson, D. B. (1996). Generating random spanning trees more quickly than the cover
time. *Proceedings of the 28th ACM Symposium on Theory of Computing*, 296-303.
"""

import dataclasses
import itertools
import math
from collections.abc import Mapping

import numpy as np

from rupture_generator.rupture.realisation import Realisation

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

type Tree = dict[str, str | None]
"""Each fault mapped to the fault that triggered it, and the root mapped to ``None``."""


class PropagationError(ValueError):
    """A fault system no single rupture runs through."""


def shaw_dieterich(
    distance_km: float | FloatArray, *, d0_km: float = 3.0, delta_km: float = 1.0
) -> FloatArray:
    """The probability that a rupture jumps a gap of a given width.

    Shaw & Dieterich (2007): certain within ``delta_km``, and decaying with
    characteristic length ``d0_km`` beyond it.

    .. math:: P(d) = \\min\\left(1, e^{-(d - \\delta) / d_0}\\right)
    """
    return np.minimum(
        1.0, np.exp(-(np.asarray(distance_km, dtype=np.float64) - delta_km) / d0_km)
    )


@dataclasses.dataclass(frozen=True)
class JumpModel:
    """How readily a rupture crosses from one fault to the next.

    ``d0_km`` and ``delta_km`` are :func:`shaw_dieterich`'s. Past ``max_jump_km`` the
    probability only adds noise to the sampler, so longer gaps carry no edge at all and
    a fault beyond reach of every other leaves the system disconnected rather than
    merely unlikely.

    ``probability_cap`` is the largest probability an edge may carry, because the
    reweighting in :func:`_sampling_weights` diverges as the probability approaches
    one. A gap that certain is a fault that should have been one segment.
    """

    d0_km: float = 3.0
    delta_km: float = 1.0
    max_jump_km: float = 15.0
    probability_cap: float = 0.99

    def __post_init__(self) -> None:
        """Refuse a model no gap answers to."""
        if self.d0_km <= 0.0:
            raise PropagationError(
                f"the decay length is a length, got {self.d0_km} km; at zero every gap "
                "is impassable and the system is never connected"
            )
        if self.delta_km < 0.0:
            raise PropagationError(f"a gap width has no sign, got {self.delta_km} km")
        if self.max_jump_km <= 0.0:
            raise PropagationError(
                f"nothing can jump {self.max_jump_km} km, so no fault triggers any other"
            )
        if not 0.0 < self.probability_cap < 1.0:
            raise PropagationError(
                f"the probability cap lies in (0, 1), got {self.probability_cap}; at 1 "
                "the sampler's weights are infinite"
            )


DEFAULT_JUMP_MODEL = JumpModel()
"""What a caller gets who does not say: Shaw & Dieterich's own decay length."""


@dataclasses.dataclass(frozen=True)
class JumpGraph:
    """Faults, and how likely the rupture is to jump between each pair.

    ``weights`` is symmetric ``(n, n)`` over ``faults`` in order. Zero means no edge at
    all rather than an impossible one: the two are the same thing for a spanning tree,
    and saying it once keeps the walk in :func:`sample_tree` off pairs the model has
    already ruled out.
    """

    faults: tuple[str, ...]
    weights: FloatArray

    def __post_init__(self) -> None:
        """Check the weights describe a symmetric graph over these faults."""
        count = len(self.faults)
        if self.weights.shape != (count, count):
            raise PropagationError(
                f"the weights are {self.weights.shape} for {count} faults"
            )
        if not np.allclose(self.weights, self.weights.T):
            raise PropagationError("a jump is as likely in one direction as the other")
        if np.any(self.weights < 0.0):
            raise PropagationError("a probability has no sign")
        if np.any(np.diag(self.weights) != 0.0):
            raise PropagationError("a fault does not trigger itself")

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
            for neighbour in np.flatnonzero(self.weights[stack.pop()] > 0.0):
                if int(neighbour) not in seen:
                    seen.add(int(neighbour))
                    stack.append(int(neighbour))
        return len(seen) == count


def separations_km(realisation: Realisation) -> dict[tuple[str, str], float]:
    """The closest approach between every pair of segments, in kilometres.

    Each chart measures to the next itself, so the two frames and their origins stay
    inside the container. Measured between the faults' *edges*, which is where a front
    can leave one and land on the other.
    """
    return {
        (near, far): float(
            realisation[near].nearest_cells_to(realisation[far])[2].min()
        )
        for near, far in itertools.combinations(realisation, 2)
    }


def jump_graph(
    distances_km: Mapping[tuple[str, str], float],
    faults: tuple[str, ...],
    model: JumpModel = DEFAULT_JUMP_MODEL,
) -> JumpGraph:
    """Turn fault separations into jump probabilities.

    ``distances_km`` is closest approach keyed by pairs in either ordering, as
    :func:`separations_km` returns; ``faults`` fixes the order the graph indexes them
    in, which is the order every edge list here refers to.

    Raises
    ------
    PropagationError
        If a distance names a fault the graph does not hold.
    """
    index = {name: position for position, name in enumerate(faults)}
    weights = np.zeros((len(faults), len(faults)), dtype=np.float64)

    for (near, far), distance_km in distances_km.items():
        unknown = sorted({near, far} - set(index))
        if unknown:
            raise PropagationError(
                f"a separation names {unknown}, which is not among {sorted(faults)}"
            )
        if near == far or distance_km >= model.max_jump_km:
            continue
        probability = min(
            float(
                shaw_dieterich(distance_km, d0_km=model.d0_km, delta_km=model.delta_km)
            ),
            model.probability_cap,
        )
        u, v = index[near], index[far]
        weights[u, v] = weights[v, u] = probability

    return JumpGraph(tuple(faults), weights)


def _sampling_weights(graph: JumpGraph) -> FloatArray:
    """The edge weights a spanning-tree sampler needs to give the right trees.

    The distribution wanted is every gap deciding for itself, conditioned on the result
    being one rupture:

    .. math:: P(T) \\propto \\prod_{e \\in T} w(e) \\prod_{e \\notin T} (1 - w(e))

    Multiplying and dividing by :math:`\\prod_{\\text{all } e} (1 - w)` takes the second
    product out as a constant, leaving a weighted uniform spanning tree over
    :math:`w / (1 - w)` -- which is what Wilson's algorithm draws.
    """
    weights = np.zeros_like(graph.weights)
    present = graph.weights > 0.0
    weights[present] = graph.weights[present] / (1.0 - graph.weights[present])
    return weights


def _disconnected(graph: JumpGraph) -> PropagationError:
    return PropagationError(
        f"{', '.join(graph.faults)} do not form a connected system: at least one is "
        "beyond reach of every other, so no single rupture gets to it. Drop it, widen "
        "the model's maximum jump, or generate it as its own earthquake"
    )


def sample_tree(graph: JumpGraph, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Draw a spanning tree, each tree as likely as the model says it is.

    Wilson's algorithm. From a vertex not yet in the tree, walk at random, stepping to
    a neighbour with probability proportional to that edge's weight, until arriving at
    a vertex the tree already holds; then adopt the path walked. Overwriting each
    vertex's onward step *is* the loop erasure, since a vertex reached a second time
    forgets the excursion it made the first time, and what survives is a simple path.
    The tree comes out with probability proportional to the product of its edge
    weights, which under :func:`_sampling_weights` is the distribution wanted.

    The edges come back as undirected, unrooted ``(u, v)`` index pairs. Which vertex
    the algorithm grows from does not bias the result: the weights are symmetric, so
    the walk is reversible and the measure is the same from any start.

    Raises
    ------
    PropagationError
        If the graph is disconnected, since a forest would be a rupture that started
        in more than one place.
    """
    if not graph.is_connected():
        raise _disconnected(graph)

    count = len(graph.faults)
    if count == 1:
        return []

    weights = _sampling_weights(graph)
    # The neighbours each vertex actually has, and their running totals, built once
    # rather than per step. Drawing among the edges that exist is also what keeps
    # round-off from stepping across a pair the model excluded, which a cumulative row
    # over every vertex allows.
    neighbours = [np.flatnonzero(row > 0.0) for row in weights]
    totals = [
        np.cumsum(row[where]) for row, where in zip(weights, neighbours, strict=True)
    ]

    in_tree = np.zeros(count, dtype=bool)
    onward = np.full(count, -1, dtype=np.int64)
    in_tree[0] = True

    for start in range(1, count):
        if in_tree[start]:
            continue
        walker = start
        while not in_tree[walker]:
            running = totals[walker]
            step = rng.random() * float(running[-1])
            choice = min(
                int(np.searchsorted(running, step, side="right")), running.size - 1
            )
            onward[walker] = neighbours[walker][choice]
            walker = int(onward[walker])
        # Retrace what survived the erasure and adopt it.
        walker = start
        while not in_tree[walker]:
            in_tree[walker] = True
            walker = int(onward[walker])

    return [(vertex, int(onward[vertex])) for vertex in range(1, count)]


def maximum_likelihood_tree(graph: JumpGraph) -> list[tuple[int, int]]:
    """The single likeliest tree, rather than a draw from the distribution.

    Maximising :math:`\\prod w / (1 - w)` over trees is maximising the sum of
    :math:`\\log w - \\log (1 - w)`, so Kruskal's maximum spanning tree gives it
    exactly. The edges come back as ``(u, v)`` index pairs.

    Raises
    ------
    PropagationError
        If the graph is disconnected.
    """
    if not graph.is_connected():
        raise _disconnected(graph)

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


def log_likelihood(graph: JumpGraph, edges: list[tuple[int, int]]) -> float:
    """How likely a tree is under the model, up to the constant every tree shares.

    :math:`\\sum_{e \\in T} \\log w - \\log (1 - w)`, the quantity
    :func:`maximum_likelihood_tree` maximises, so two trees on one graph are
    comparable by it and a sampled tree can be scored against the likeliest.
    """
    return sum(
        math.log(w) - math.log1p(-w)
        for w in (float(graph.weights[u, v]) for u, v in edges)
    )


def root_tree(faults: tuple[str, ...], edges: list[tuple[int, int]], root: str) -> Tree:
    """Orient an undirected tree away from the fault the rupture started on.

    Raises
    ------
    PropagationError
        If the root is not one of the faults, or some fault is unreachable from it.
    """
    if root not in faults:
        raise PropagationError(
            f"the rupture starts on {root!r}, which is not one of {', '.join(faults)}"
        )

    neighbours: dict[int, list[int]] = {index: [] for index in range(len(faults))}
    for u, v in edges:
        neighbours[u].append(v)
        neighbours[v].append(u)

    start = faults.index(root)
    tree: Tree = {root: None}
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
        raise PropagationError(
            f"{', '.join(missing)} cannot be reached from {root!r}, so the tree does "
            "not describe one rupture"
        )
    return tree


def _propagated(
    realisation: Realisation, root: str, edges: list[tuple[int, int]]
) -> Realisation:
    return realisation.propagated(root_tree(tuple(realisation), edges, root))


def sample_path(
    realisation: Realisation,
    *,
    root: str,
    rng: np.random.Generator,
    model: JumpModel = DEFAULT_JUMP_MODEL,
) -> Realisation:
    """One rupture path through a fault system, drawn from the model.

    Measures the segments' separations, turns them into jump probabilities, draws a
    tree by :func:`sample_tree` and orients it away from ``root``. The result is the
    realisation with its tree recorded, ready for the generator.

    Raises
    ------
    PropagationError
        If the system is disconnected, or the root is not one of its segments.
    """
    graph = jump_graph(separations_km(realisation), tuple(realisation), model)
    return _propagated(realisation, root, sample_tree(graph, rng))


def likeliest_path(
    realisation: Realisation, *, root: str, model: JumpModel = DEFAULT_JUMP_MODEL
) -> Realisation:
    """The likeliest rupture path through a fault system.

    :func:`sample_path`'s counterpart: the same graph, and the tree
    :func:`maximum_likelihood_tree` picks out of it. Deterministic, so it takes no
    generator -- which makes it what to use when a study wants one representative
    rupture per system rather than an ensemble.

    Raises
    ------
    PropagationError
        If the system is disconnected, or the root is not one of its segments.
    """
    graph = jump_graph(separations_km(realisation), tuple(realisation), model)
    return _propagated(realisation, root, maximum_likelihood_tree(graph))


__all__ = [
    "DEFAULT_JUMP_MODEL",
    "JumpGraph",
    "JumpModel",
    "PropagationError",
    "Tree",
    "jump_graph",
    "likeliest_path",
    "log_likelihood",
    "maximum_likelihood_tree",
    "root_tree",
    "sample_path",
    "sample_tree",
    "separations_km",
    "shaw_dieterich",
]
