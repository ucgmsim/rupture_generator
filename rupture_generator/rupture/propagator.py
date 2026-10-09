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
:func:`maximum_likelihood_tree` the second as a maximum spanning tree.

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

import numpy as np
import scipy as sp

from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.rupture.realisation import Hypocentre, Realisation

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

type Tree = dict[str, str | None]
"""Each fault mapped to the fault that triggered it, and the root mapped to ``None``."""


@dataclasses.dataclass(frozen=True)
class JumpModel:
    """How readily a rupture crosses from one fault to the next.

    Past ``max_jump_km`` the probability only adds noise to the sampler, so longer gaps
    carry no edge at all and a fault beyond reach of every other leaves the system
    disconnected rather than merely unlikely.

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
            raise RuptureGeneratorError(
                f"the decay length is a length, got {self.d0_km} km; at zero every gap "
                "is impassable and the system is never connected"
            )
        if self.delta_km < 0.0:
            raise RuptureGeneratorError(
                f"a gap width has no sign, got {self.delta_km} km"
            )
        if self.max_jump_km <= 0.0:
            raise RuptureGeneratorError(
                f"nothing can jump {self.max_jump_km} km, so no fault triggers any other"
            )
        if not 0.0 < self.probability_cap < 1.0:
            raise RuptureGeneratorError(
                f"the probability cap lies in (0, 1), got {self.probability_cap}; at 1 "
                "the sampler's weights are infinite"
            )

    def probability(self, distance_km: float) -> float:
        """The probability that a rupture jumps a gap of a given width.

        Shaw & Dieterich (2007): certain within ``delta_km``, and decaying with
        characteristic length ``d0_km`` beyond it, then capped and cut off.

        .. math:: P(d) = \\min\\left(1, e^{-(d - \\delta) / d_0}\\right)
        """
        if distance_km >= self.max_jump_km:
            return 0.0
        decay = np.exp(-(distance_km - self.delta_km) / self.d0_km)
        return float(min(decay, self.probability_cap))


DEFAULT_JUMP_MODEL = JumpModel()
"""Shaw & Dieterich's own decay length."""


@dataclasses.dataclass(frozen=True)
class JumpGraph:
    """Faults, and how likely the rupture is to jump between each pair.

    ``weights`` is symmetric ``(n, n)`` over ``faults`` in order. Zero means no edge at
    all rather than an impossible one: the two are the same thing for a spanning tree.
    """

    faults: tuple[str, ...]
    weights: FloatArray

    def is_connected(self) -> bool:
        """Whether every fault is reachable from every other."""
        components, _ = sp.sparse.csgraph.connected_components(
            self.weights > 0.0, directed=False
        )
        return components == 1


def jump_graph(
    realisation: Realisation, model: JumpModel = DEFAULT_JUMP_MODEL
) -> JumpGraph:
    """Jump probabilities between every pair of segments.

    Measured between the faults' *edges*, which is where a front can leave one and land
    on the other.
    """
    faults = tuple(realisation)
    weights = np.zeros((len(faults), len(faults)), dtype=np.float64)
    for (u, near), (v, far) in itertools.combinations(enumerate(faults), 2):
        distance_km = float(
            realisation[near].nearest_cells_to(realisation[far])[2].min()
        )
        weights[u, v] = weights[v, u] = model.probability(distance_km)
    return JumpGraph(faults, weights)


def _log_odds(probability: FloatArray) -> FloatArray:
    return np.log(probability) - np.log1p(-probability)


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
    weights[present] = np.exp(_log_odds(graph.weights[present]))
    return weights


def _disconnected(graph: JumpGraph) -> RuptureGeneratorError:
    return RuptureGeneratorError(
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
    RuptureGeneratorError
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
    :math:`\\log w - \\log (1 - w)`, so the maximum spanning tree gives it exactly. The
    log odds are shifted to positive costs, since every spanning tree has the same
    number of edges and a zero cost reads as no edge. The edges come back as ``(u, v)``
    index pairs.

    Raises
    ------
    RuptureGeneratorError
        If the graph is disconnected.
    """
    if not graph.is_connected():
        raise _disconnected(graph)
    if len(graph.faults) == 1:
        return []
    present = graph.weights > 0.0
    score = _log_odds(graph.weights[present])
    cost = np.zeros_like(graph.weights)
    cost[present] = score.max() + 1.0 - score
    u, v = sp.sparse.csgraph.minimum_spanning_tree(cost).nonzero()
    return list(zip(u.tolist(), v.tolist(), strict=True))


def root_tree(faults: tuple[str, ...], edges: list[tuple[int, int]], root: str) -> Tree:
    """Orient an undirected tree away from the fault the rupture started on.

    Raises
    ------
    RuptureGeneratorError
        If the root is not one of the faults, or some fault is unreachable from it.
    """
    if root not in faults:
        raise RuptureGeneratorError(
            f"the rupture starts on {root!r}, which is not one of {', '.join(faults)}"
        )
    count = len(faults)
    adjacency = np.zeros((count, count))
    for u, v in edges:
        adjacency[u, v] = 1.0
    start = faults.index(root)
    order, predecessors = sp.sparse.csgraph.breadth_first_order(
        adjacency, start, directed=False, return_predecessors=True
    )
    if len(order) != count:
        missing = sorted(set(faults) - {faults[k] for k in order})
        raise RuptureGeneratorError(
            f"{', '.join(missing)} cannot be reached from {root!r}, so the tree does "
            "not describe one rupture"
        )
    return {faults[k]: None if k == start else faults[predecessors[k]] for k in order}


def sample_path(
    realisation: Realisation,
    hypocentre: Hypocentre,
    *,
    rng: np.random.Generator,
    model: JumpModel = DEFAULT_JUMP_MODEL,
) -> Realisation:
    """One rupture path through a fault system, drawn from the model.

    Measures the segments' separations, turns them into jump probabilities, draws a
    tree by :func:`sample_tree` and orients it away from the hypocentre. The result is
    the realisation with its tree recorded, ready for the generator.

    Raises
    ------
    RuptureGeneratorError
        If the system is disconnected, or the hypocentre is not on one of its segments.
    """
    graph = jump_graph(realisation, model)
    tree = root_tree(graph.faults, sample_tree(graph, rng), hypocentre.segment)
    return realisation.propagated(tree, hypocentre)


def likeliest_path(
    realisation: Realisation,
    hypocentre: Hypocentre,
    *,
    model: JumpModel = DEFAULT_JUMP_MODEL,
) -> Realisation:
    """The likeliest rupture path through a fault system.

    :func:`sample_path`'s counterpart: the same graph, and the tree
    :func:`maximum_likelihood_tree` picks out of it. Deterministic, so it takes no
    generator -- which makes it what to use when a study wants one representative
    rupture per system rather than an ensemble.

    Raises
    ------
    RuptureGeneratorError
        If the system is disconnected, or the hypocentre is not on one of its segments.
    """
    graph = jump_graph(realisation, model)
    tree = root_tree(graph.faults, maximum_likelihood_tree(graph), hypocentre.segment)
    return realisation.propagated(tree, hypocentre)


__all__ = [
    "DEFAULT_JUMP_MODEL",
    "JumpGraph",
    "JumpModel",
    "Tree",
    "jump_graph",
    "likeliest_path",
    "maximum_likelihood_tree",
    "root_tree",
    "sample_path",
    "sample_tree",
]
