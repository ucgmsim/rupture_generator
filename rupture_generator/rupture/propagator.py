"""Which faults rupture, and in what order.

A multi-segment earthquake is a tree. Each fault but the root has one triggering
parent, and the root is the fault the rupture nucleated on. This module fixes the tree before
the generator draws any field, because the tree depends on fault separations and not
on slip. Where and when the front crossed belongs to the generator, which reads a
crossing time off the parent's solved onsets.

One model gives two ways to choose the tree. Each gap transmits the rupture with a
probability that decays with its width, which gives every tree a probability. That
leaves two natural questions: draw one tree, or find the likeliest.
:func:`_sample_tree` answers the first by Wilson's algorithm and
:func:`_maximum_likelihood_tree` answers the second as a maximum spanning tree.

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
time. *Proceedings of the Twenty-Eighth ACM Symposium on Theory of Computing*, 296-303.
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

    Past ``max_jump_km`` the probability only adds noise to the sampler. A longer gap
    doesn't get an edge, and a fault farther than that from every other fault leaves
    the system disconnected.

    ``probability_cap`` bounds the probability on any edge, because the reweighting in
    :func:`_sampling_weights` diverges as the probability approaches one. A gap that
    certain belongs inside one segment.

    Attributes
    ----------
    d0_km : float
        The decay length of the jump probability beyond ``delta_km``, in kilometres.
        Shaw & Dieterich's value is 3 km.
    delta_km : float
        The gap width below which a jump is certain, in kilometres.
    max_jump_km : float
        The widest gap a rupture can jump, in kilometres.
    probability_cap : float
        The largest probability an edge may have, in ``(0, 1)``.
    """

    d0_km: float = 3.0
    delta_km: float = 1.0
    max_jump_km: float = 15.0
    probability_cap: float = 0.99

    def __post_init__(self) -> None:
        """Refuse parameters that make the model meaningless."""
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
        """Return the probability that a rupture jumps a gap of a given width.

        Shaw & Dieterich (2007) make a jump certain within ``delta_km`` and let the
        probability decay with characteristic length ``d0_km`` beyond it. This model
        then caps it at ``probability_cap`` and cuts it off at ``max_jump_km``.

        .. math:: P(d) = \\min\\left(1, e^{-(d - \\delta) / d_0}\\right)

        Parameters
        ----------
        distance_km : float
            The gap width, in kilometres.

        Returns
        -------
        float
            The jump probability, zero at or beyond ``max_jump_km``.
        """
        if distance_km >= self.max_jump_km:
            return 0.0
        decay = np.exp(-(distance_km - self.delta_km) / self.d0_km)
        return float(min(decay, self.probability_cap))

    def reach_km(self, nearest_km: float, rng: np.random.Generator) -> float:
        """Draw the widest gap a rupture that did jump can cross.

        Read :meth:`probability` as a survival function: ``P(d)`` is the chance a
        rupture crosses a gap of at least ``d``, so the widest gap it crosses is
        ``delta_km`` plus an exponential of mean ``d0_km``. The tree says the rupture
        crossed the nearest gap, ``nearest_km``. The exponential has no memory, so
        with that crossing known the distance is the gap (or ``delta_km``, if wider)
        plus a fresh exponential. This method caps the draw at ``max_jump_km``, but
        never below the gap the rupture crossed.

        Parameters
        ----------
        nearest_km : float
            The narrowest gap between the parent and the child, in kilometres.
        rng : numpy.random.Generator
            The generator the exponential comes from.

        Returns
        -------
        float
            The widest gap the jump can cross, in kilometres.
        """
        reach = max(self.delta_km, nearest_km) + rng.exponential(self.d0_km)
        return float(min(reach, max(self.max_jump_km, nearest_km)))


DEFAULT_JUMP_MODEL = JumpModel()
"""Shaw & Dieterich's own decay length."""


@dataclasses.dataclass(frozen=True)
class _JumpGraph:
    """Faults, and how likely the rupture is to jump between each pair.

    ``weights`` is symmetric ``(n, n)`` over ``faults`` in order. Zero means no edge.
    For a spanning tree, an impossible edge and a missing one are the same thing.
    """

    faults: tuple[str, ...]
    weights: FloatArray

    def is_connected(self) -> bool:
        """Report whether a path joins every fault to every other.

        Returns
        -------
        bool
            True if the graph has one connected component.
        """
        components, _ = sp.sparse.csgraph.connected_components(
            self.weights > 0.0, directed=False
        )
        return components == 1


def _jump_graph(
    realisation: Realisation, model: JumpModel = DEFAULT_JUMP_MODEL
) -> _JumpGraph:
    """Measure the jump probability between every pair of segments.

    The distance is between the faults' edges, where a front can leave one fault and
    start the other.
    """
    faults = tuple(realisation)
    weights = np.zeros((len(faults), len(faults)), dtype=np.float64)
    for (u, near), (v, far) in itertools.combinations(enumerate(faults), 2):
        distance_km = float(
            realisation[near].nearest_cells_to(realisation[far])[2].min()
        )
        weights[u, v] = weights[v, u] = model.probability(distance_km)
    return _JumpGraph(faults, weights)


def _log_odds(probability: FloatArray) -> FloatArray:
    return np.log(probability) - np.log1p(-probability)


def _sampling_weights(graph: _JumpGraph) -> FloatArray:
    """Turn jump probabilities into the weights a spanning-tree sampler needs.

    The target distribution treats each gap as an independent trial, conditioned on the
    result being one rupture:

    .. math:: P(T) \\propto \\prod_{e \\in T} w(e) \\prod_{e \\notin T} (1 - w(e))

    Multiplying and dividing by :math:`\\prod_{\\text{all } e} (1 - w)` takes the second
    product out as a constant, and leaves a weighted uniform spanning tree over
    :math:`w / (1 - w)`, which Wilson's algorithm draws.
    """
    weights = np.zeros_like(graph.weights)
    present = graph.weights > 0.0
    weights[present] = np.exp(_log_odds(graph.weights[present]))
    return weights


def _disconnected(graph: _JumpGraph) -> RuptureGeneratorError:
    return RuptureGeneratorError(
        f"{', '.join(graph.faults)} do not form a connected system: at least one is "
        "beyond reach of every other, so no single rupture gets to it. Drop it, widen "
        "the model's maximum jump, or generate it as its own earthquake"
    )


def _sample_tree(graph: _JumpGraph, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Draw a spanning tree, each tree as likely as the model says.

    This is Wilson's algorithm. From a vertex outside the tree, walk at random, stepping
    to a neighbour with probability proportional to that edge's weight, until the walk
    meets a vertex in the tree. Then add the walked path to the tree. Overwriting each
    vertex's onward step performs the loop erasure: on a second visit a vertex forgets
    its first excursion, so the recorded path has no loops. The tree comes out with
    probability proportional to the product of its edge weights, which under
    :func:`_sampling_weights` is the target distribution.

    It returns undirected, unrooted ``(u, v)`` index pairs. The starting vertex
    doesn't bias the result: the weights are symmetric, so the walk is reversible and
    the measure is the same from any start.

    Raises
    ------
    RuptureGeneratorError
        If no path joins every fault to every other, since a forest would be a
        rupture that started in more than one place.
    """
    if not graph.is_connected():
        raise _disconnected(graph)

    count = len(graph.faults)
    if count == 1:
        return []

    weights = _sampling_weights(graph)
    # Each vertex's neighbours and their running totals, built once for the whole
    # walk. Drawing only among existing edges also stops round-off from stepping
    # across a pair the model excluded, which a cumulative row over every vertex
    # would allow.
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
        # Retrace the path left after the erasure and add it to the tree.
        walker = start
        while not in_tree[walker]:
            in_tree[walker] = True
            walker = int(onward[walker])

    return [(vertex, int(onward[vertex])) for vertex in range(1, count)]


def _maximum_likelihood_tree(graph: _JumpGraph) -> list[tuple[int, int]]:
    """Find the likeliest tree.

    Maximising :math:`\\prod w / (1 - w)` over trees is maximising the sum of
    :math:`\\log w - \\log (1 - w)`, so the maximum spanning tree gives it exactly. This
    function shifts the log odds to positive costs, since every spanning tree has the
    same number of edges and a zero cost reads as no edge. It returns the edges as
    ``(u, v)`` index pairs.

    Raises
    ------
    RuptureGeneratorError
        If no path joins every fault to every other.
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


def _root_tree(
    faults: tuple[str, ...], edges: list[tuple[int, int]], root: str
) -> Tree:
    """Orient an undirected tree away from the fault the rupture started on.

    Raises
    ------
    RuptureGeneratorError
        If the root isn't one of the faults, or no path leads from it to some fault.
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
    """Draw one rupture path through a fault system from the model.

    The jump probabilities come from the segments' separations. A tree drawn by
    :func:`_sample_tree` over them, oriented away from the hypocentre, is the path.

    Parameters
    ----------
    realisation : Realisation
        The fault system.
    hypocentre : Hypocentre
        Where the rupture nucleates.
    rng : numpy.random.Generator
        The generator that draws the tree.
    model : JumpModel
        How readily the rupture jumps a gap.

    Returns
    -------
    Realisation
        The realisation with its hypocentre and tree recorded, ready for the generator.

    Raises
    ------
    RuptureGeneratorError
        If no path joins every segment to every other, or the hypocentre isn't on
        one of the segments.
    """
    graph = _jump_graph(realisation, model)
    tree = _root_tree(graph.faults, _sample_tree(graph, rng), hypocentre.segment)
    return realisation.propagated(tree, hypocentre)


def likeliest_path(
    realisation: Realisation,
    hypocentre: Hypocentre,
    *,
    model: JumpModel = DEFAULT_JUMP_MODEL,
) -> Realisation:
    """Find the likeliest rupture path through a fault system.

    This is the counterpart of :func:`sample_path`: the same graph, with the tree
    :func:`_maximum_likelihood_tree` picks from it. The result is deterministic, so it
    takes no generator. Use it for one representative rupture per system in place of
    an ensemble.

    Parameters
    ----------
    realisation : Realisation
        The fault system.
    hypocentre : Hypocentre
        Where the rupture nucleates.
    model : JumpModel
        How readily the rupture jumps a gap.

    Returns
    -------
    Realisation
        The realisation with its hypocentre and tree recorded, ready for the generator.

    Raises
    ------
    RuptureGeneratorError
        If no path joins every segment to every other, or the hypocentre isn't on
        one of the segments.
    """
    graph = _jump_graph(realisation, model)
    tree = _root_tree(graph.faults, _maximum_likelihood_tree(graph), hypocentre.segment)
    return realisation.propagated(tree, hypocentre)


__all__ = [
    "DEFAULT_JUMP_MODEL",
    "JumpModel",
    "Tree",
    "likeliest_path",
    "sample_path",
]
