"""Properties of the causality tree and the jumps between faults.

The tree sampler is tested against **brute-force enumeration** of the distribution it
is meant to draw from. That is a real reference rather than a second reading of the
subject: enumerating every spanning tree and weighting it by ``prod(w) * prod(1 - w)``
is a different algorithm from a loop-erased random walk, and the two agreeing is
evidence about both. It is also the machinery the source module carried in production
and this one does not -- the enumeration is exponential, so it belongs in a test over
four faults and nowhere near a run.

Statistical assertions here carry multinomial error, stated at the assertion: with
``n`` draws, a tree of probability ``p`` is seen ``p ± sqrt(p(1-p)/n)`` of the time.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rupture_generator import propagation
from rupture_generator.config.geometry import (
    ComputedPropagation,
    PredeterminedPropagation,
)
from rupture_generator.mesh import RuptureMesh
from rupture_generator.pipeline import causality_tree
from rupture_generator.propagation import (
    DistanceOverVelocity,
    Instantaneous,
    JumpGraph,
    causal_jump,
    check_tree,
    in_topological_order,
    jump_graph,
    maximum_likelihood_tree,
    root_tree,
    sample_tree,
    shaw_dieterich,
)

SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def enumerate_trees(graph: JumpGraph) -> dict[frozenset, float]:
    """Every spanning tree and its probability, by brute force.

    ``P(T) proportional to prod(w in T) * prod(1 - w not in T)`` -- the model's own
    definition, evaluated directly over all ``n - 1`` edge subsets that happen to
    span. Exponential, and the reason this is a test and not a library function.
    """
    edges = graph.edges
    count = len(graph.faults)
    probabilities: dict[frozenset, float] = {}

    for chosen in itertools.combinations(edges, count - 1):
        parent = list(range(count))

        def find(node: int, parent: list[int] = parent) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        spans = True
        for u, v, _ in chosen:
            root_u, root_v = find(u), find(v)
            if root_u == root_v:
                spans = False
                break
            parent[root_u] = root_v
        if not spans:
            continue

        key = frozenset((u, v) for u, v, _ in chosen)
        probability = 1.0
        for u, v, weight in edges:
            probability *= weight if (u, v) in key else (1.0 - weight)
        probabilities[key] = probability

    total = sum(probabilities.values())
    return {key: value / total for key, value in probabilities.items()}


def _graph() -> JumpGraph:
    """Four faults with a spread of jump probabilities, and every pair connected."""
    weights = np.array(
        [
            [0.00, 0.80, 0.30, 0.05],
            [0.80, 0.00, 0.60, 0.20],
            [0.30, 0.60, 0.00, 0.70],
            [0.05, 0.20, 0.70, 0.00],
        ]
    )
    return JumpGraph(("a", "b", "c", "d"), weights)


def _chart(
    *, east_km: float = 0.0, north_km: float = 0.0, cells: int = 6, name: str = "f"
) -> RuptureMesh:
    """A flat unit-spaced chart, offset to wherever the test wants a fault."""
    across, down = np.meshgrid(
        np.arange(cells + 1, dtype=float), np.arange(cells + 1, dtype=float)
    )
    return RuptureMesh.from_nodes(
        down + east_km,
        across + north_km,
        np.full_like(across, 5.0),
        origin_east_km=0.0,
        origin_north_km=0.0,
        surface=name,
    )


# ============================================================================
# The jump probability
# ============================================================================


@given(
    distance_km=st.floats(min_value=0.0, max_value=60.0, allow_nan=False),
    d0_km=st.floats(min_value=0.5, max_value=10.0, allow_nan=False),
    delta_km=st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
)
def test_a_jump_becomes_less_likely_with_distance(
    distance_km: float, d0_km: float, delta_km: float
) -> None:
    """Shaw-Dieterich is a probability, certain up close and decaying beyond.

    The three claims that make it usable: it never leaves ``(0, 1]``, it is exactly 1
    within the certainty distance -- faults that nearly touch always break together
    -- and it is monotone, so a fault further away is never more likely to be
    triggered than a nearer one.
    """
    probability = float(shaw_dieterich(distance_km, d0_km=d0_km, delta_km=delta_km))
    assert 0.0 < probability <= 1.0

    if distance_km <= delta_km:
        assert probability == 1.0

    further = float(shaw_dieterich(distance_km + 1.0, d0_km=d0_km, delta_km=delta_km))
    assert further <= probability


def test_faults_beyond_the_limit_get_no_edge_at_all() -> None:
    """Past the jump limit a pair is absent from the graph, not merely improbable.

    The difference matters: an edge with a tiny weight is a jump the sampler can
    still make, where no edge at all means a fault out of reach of every other makes
    the system *disconnected*, which is refused. A rupture that started twice is not
    one earthquake.
    """
    graph = jump_graph(
        {("a", "b"): 2.0, ("a", "c"): 40.0, ("b", "c"): 38.0},
        ["a", "b", "c"],
    )
    assert graph.weights[0, 1] > 0.0
    assert graph.weights[0, 2] == 0.0
    assert not graph.is_connected()

    with pytest.raises(ValueError, match="connected"):
        sample_tree(graph, np.random.default_rng(0))


# ============================================================================
# The sampler, against enumeration
# ============================================================================


@pytest.mark.slow
def test_the_sampler_draws_trees_at_their_own_probabilities() -> None:
    """Sampled frequencies match the enumerated distribution.

    The property the whole tree apparatus rests on. Wilson's algorithm samples a
    spanning tree with probability proportional to the product of its edge weights;
    reweighting each edge to ``w / (1 - w)`` turns that into the distribution the jump
    model asks for, because the leftover ``prod(1 - w)`` over *all* edges is a constant
    that cancels. If that derivation were wrong -- or the reweighting omitted -- the
    trees would still be trees and the ruptures would still look plausible.

    Sixty thousand draws puts the standard error at the largest cell near 0.002.
    The assertion allows **four** of those rather than three, because sixteen cells
    are tested at once: at three standard errors each, the chance that at least one
    exceeds by luck alone is about four percent, which would make this test flaky
    rather than informative. Four puts the family-wise rate near a thousandth.
    """
    graph = _graph()
    expected = enumerate_trees(graph)
    assert len(expected) == 16

    draws = 60_000
    rng = np.random.default_rng(20260809)
    seen: dict[frozenset, int] = {}
    for _ in range(draws):
        key = frozenset(tuple(sorted(edge)) for edge in sample_tree(graph, rng))
        seen[key] = seen.get(key, 0) + 1

    # Every tree with any real probability should turn up.
    assert set(seen) == set(expected)

    for key, probability in expected.items():
        realised = seen[key] / draws
        standard_error = (probability * (1.0 - probability) / draws) ** 0.5
        assert abs(realised - probability) <= 4.0 * standard_error + 1.0e-4


def test_the_most_likely_tree_is_the_argmax_of_the_distribution() -> None:
    """Kruskal on ``log w - log(1 - w)`` picks the same tree enumeration does.

    Maximising a product of ``w / (1 - w)`` is maximising a sum of its logarithms,
    which is what a maximum spanning tree does -- so this is an identity between two
    computations, and enumeration is the one that follows the definition.
    """
    graph = _graph()
    expected = enumerate_trees(graph)

    chosen = frozenset(tuple(sorted(edge)) for edge in maximum_likelihood_tree(graph))
    assert chosen == max(expected, key=lambda key: expected[key])


@SETTINGS
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_every_sampled_tree_is_a_tree(seed: int) -> None:
    """``n - 1`` edges, no cycle, everything reachable from the root.

    Asserted through :func:`root_tree`, which is what the pipeline actually uses: it
    refuses an unreachable fault, so a sampled forest would be caught here rather
    than becoming a rupture with a fault that never breaks.
    """
    graph = _graph()
    edges = sample_tree(graph, np.random.default_rng(seed))

    assert len(edges) == len(graph.faults) - 1

    for root in graph.faults:
        tree = root_tree(graph.faults, edges, root)
        assert set(tree) == set(graph.faults)
        assert [name for name, parent in tree.items() if parent is None] == [root]
        # Every fault reaches the root by following parents, so there is no cycle.
        for fault in tree:
            steps = 0
            node = fault
            while tree[node] is not None:
                node = tree[node]
                steps += 1
                assert steps <= len(graph.faults)
            assert node == root


def test_a_sampled_tree_is_reproducible() -> None:
    """The same seed gives the same tree, so a rupture is reproducible from its seed."""
    graph = _graph()
    first = sample_tree(graph, np.random.default_rng(7))
    second = sample_tree(graph, np.random.default_rng(7))
    assert first == second


def test_one_fault_is_a_tree_with_no_edges() -> None:
    """The degenerate case the single-fault path takes."""
    graph = JumpGraph(("solo",), np.zeros((1, 1)))
    assert sample_tree(graph, np.random.default_rng(0)) == []
    assert root_tree(("solo",), [], "solo") == {"solo": None}


# ============================================================================
# A stated tree
# ============================================================================


def test_a_stated_tree_is_used_verbatim() -> None:
    """Predetermined mode does what it says, and does not sample.

    Asserted by giving it a tree the sampler would essentially never draw -- the
    chain through the *least* likely edges -- and checking it comes back.
    """
    segments = {
        "a": _chart(name="a"),
        "b": _chart(east_km=8.0, name="b"),
        "c": _chart(east_km=16.0, name="c"),
    }
    stated = PredeterminedPropagation(parents={"b": "a", "c": "b"})

    tree = causality_tree(segments, stated, "a", np.random.default_rng(0))
    assert tree == {"a": None, "b": "a", "c": "b"}


def test_a_stated_tree_that_is_not_one_is_refused() -> None:
    """Each way of writing it down wrong is refused by name.

    These are mistakes someone makes in a geometry file, so each says what is wrong
    rather than failing later inside a traversal.
    """
    faults = ["a", "b", "c"]

    # A cycle that leaves a root elsewhere: `a` roots the tree, and `b` and `c`
    # trigger each other. A cycle with no root at all is caught one check earlier,
    # as a rupture that starts nowhere.
    with pytest.raises(ValueError, match="is its own ancestor"):
        check_tree({"a": None, "b": "c", "c": "b", "d": "a"}, [*faults, "d"], "a")

    with pytest.raises(ValueError, match="starts in one place"):
        check_tree({"a": "c", "b": "a", "c": "b"}, faults, "a")

    with pytest.raises(ValueError, match="not a surface"):
        check_tree({"a": None, "b": "a", "c": "elsewhere"}, faults, "a")

    with pytest.raises(ValueError, match="no entry in the propagation"):
        check_tree({"a": None, "b": "a"}, faults, "a")

    with pytest.raises(ValueError, match="faults with no parent"):
        check_tree({"a": None, "b": None, "c": "a"}, faults, "a")


def test_a_stated_root_must_be_where_the_hypocentre_is() -> None:
    """The tree says where the rupture starts and so does the hypocentre.

    They are two statements of one fact, so they can disagree -- and this refuses
    rather than choosing. Silently preferring either would move the nucleation point
    to a fault the config did not name.
    """
    with pytest.raises(ValueError, match="rooted at 'a' but the hypocentre is on 'b'"):
        check_tree({"a": None, "b": "a"}, ["a", "b"], "b")


def test_topological_order_puts_a_parent_before_its_children() -> None:
    """The order the wavefront is solved in: a child's seed needs its parent's onsets."""
    tree = {"a": None, "b": "a", "c": "a", "d": "c"}
    order = list(in_topological_order(tree))

    assert set(order) == set(tree)
    for fault, parent in tree.items():
        if parent is not None:
            assert order.index(parent) < order.index(fault)


# ============================================================================
# The causal jump
# ============================================================================


def test_the_jump_leaves_from_where_the_front_arrives_first() -> None:
    """Not the closest pair -- the earliest one.

    Two faults side by side, with the parent's front arriving at its far end long
    before its near end. With no delay the jump leaves from the early corner even
    though a nearer point exists, because what the rupture does is arrive somewhere
    and carry on, not find the shortest gap on a map. This is the whole difference
    from fitting jumps by closest approach.
    """
    parent = _chart(cells=6, name="parent")
    child = _chart(east_km=10.0, cells=6, name="child")

    # The front sweeps the parent from high j to low j: the *far* end is early.
    onset_s = np.tile(np.arange(6, dtype=float)[::-1], (6, 1))

    jump = causal_jump(parent, onset_s, child, Instantaneous())

    assert jump.departure_s == 0.0
    # Column 5 is the early end, and the jump leaves from it.
    assert jump.parent_cell[1] == 5
    assert jump.arrival_s == 0.0


def test_a_delay_moves_the_jump_towards_the_shorter_gap() -> None:
    """With a delay, distance starts to matter again -- and is traded against time.

    The control is the same geometry with no delay. A model that ignored its delay
    argument, or applied it with the wrong sign, would give the same answer in both.
    """
    parent = _chart(cells=6, name="parent")
    child = _chart(east_km=10.0, cells=6, name="child")
    onset_s = np.tile(np.arange(6, dtype=float)[::-1], (6, 1))

    instant = causal_jump(parent, onset_s, child, Instantaneous())
    delayed = causal_jump(parent, onset_s, child, DistanceOverVelocity(3.0))

    assert delayed.arrival_s > instant.arrival_s
    assert delayed.distance_km <= instant.distance_km


def test_the_jump_is_within_the_distance_a_rupture_jumps() -> None:
    """The bound is physics, not an optimisation.

    A rupture crosses a gap at roughly the shear speed but travels along a fault at a
    fraction of it, so an unbounded search finds that leaving the hypocentre and
    crossing tens of kilometres of intact rock beats propagating there -- measured at
    a 28 km "jump" on the shipped two-segment example, arriving before the front had
    covered a third of the first fault. The gap model is fitted to stepovers of a few
    kilometres and says nothing about that.
    """
    parent = _chart(cells=6, name="parent")
    child = _chart(east_km=30.0, cells=6, name="child")
    onset_s = np.zeros((6, 6))

    with pytest.raises(ValueError, match="past the"):
        causal_jump(parent, onset_s, child, Instantaneous(), max_distance_km=15.0)

    near = _chart(east_km=10.0, cells=6, name="near")
    jump = causal_jump(parent, onset_s, near, Instantaneous(), max_distance_km=15.0)
    assert jump.distance_km <= 15.0


def test_a_jump_never_arrives_before_it_leaves() -> None:
    """Causality, on one edge: the arrival is the departure plus a delay.

    The delay is non-negative for every model here, so the arrival cannot precede the
    departure -- which is the per-edge half of the whole-rupture causality property.
    """
    parent = _chart(cells=6, name="parent")
    child = _chart(east_km=8.0, cells=6, name="child")
    rng = np.random.default_rng(3)
    onset_s = rng.random((6, 6)) * 5.0

    for delay in (Instantaneous(), DistanceOverVelocity(3.2)):
        jump = causal_jump(parent, onset_s, child, delay)
        assert jump.arrival_s >= jump.departure_s
        assert jump.departure_s == pytest.approx(onset_s[jump.parent_cell])


def test_a_delay_needs_a_speed_the_front_can_travel_at() -> None:
    """A zero or negative crossing speed never arrives."""
    with pytest.raises(ValueError, match="never arrives"):
        DistanceOverVelocity(0.0)


def test_the_default_delay_is_distance_over_velocity() -> None:
    """And it is what its name says, elementwise."""
    delay = DistanceOverVelocity(4.0)
    assert delay(np.array([0.0, 8.0, 12.0])).tolist() == [0.0, 2.0, 3.0]
    assert Instantaneous()(np.array([0.0, 8.0])).tolist() == [0.0, 0.0]


# ============================================================================
# The computed mode, through the pipeline's own entry point
# ============================================================================


def test_a_computed_tree_connects_faults_within_reach() -> None:
    """Three faults in a line, each within jumping distance of the next."""
    segments = {
        "a": _chart(name="a"),
        "b": _chart(east_km=12.0, name="b"),
        "c": _chart(east_km=24.0, name="c"),
    }
    tree = causality_tree(
        segments, ComputedPropagation(), "a", np.random.default_rng(11)
    )

    assert set(tree) == {"a", "b", "c"}
    assert tree["a"] is None
    assert set(in_topological_order(tree)) == {"a", "b", "c"}


def test_the_maximum_likelihood_strategy_is_deterministic() -> None:
    """No draw is made, so the generator is irrelevant and the answer is the same."""
    segments = {
        "a": _chart(name="a"),
        "b": _chart(east_km=9.0, name="b"),
        "c": _chart(east_km=20.0, name="c"),
    }
    config = ComputedPropagation(strategy="maximum_likelihood")

    first = causality_tree(segments, config, "a", np.random.default_rng(1))
    second = causality_tree(segments, config, "a", np.random.default_rng(999))
    assert first == second


def test_faults_out_of_reach_of_each_other_are_refused() -> None:
    """A system nothing can propagate across is not one earthquake."""
    segments = {
        "a": _chart(name="a"),
        "far": _chart(east_km=200.0, name="far"),
    }
    with pytest.raises(ValueError, match="connected"):
        causality_tree(segments, ComputedPropagation(), "a", np.random.default_rng(0))


def test_closest_approach_is_symmetric_and_zero_for_touching_faults() -> None:
    """What the jump probability is a function of."""
    first = _chart(name="a")
    second = _chart(east_km=10.0, name="b")

    forward = propagation.closest_approach_km(first, second)
    backward = propagation.closest_approach_km(second, first)
    assert forward == pytest.approx(backward)
    # Cell centres, not surfaces: the charts' node edges are 4 km apart, and their
    # nearest centres half a cell further in on each side.
    assert forward == pytest.approx(5.0, abs=1e-9)

    assert propagation.closest_approach_km(first, first) == pytest.approx(0.0)
