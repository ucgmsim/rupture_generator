import numpy as np
import pytest

from rupture_generator import (
    Geometry,
    Hypocentre,
    JumpModel,
    Realisation,
    RuptureGeneratorError,
    RuptureModel,
    first_arrival,
    generate,
    likeliest_path,
    sample_path,
    segment_source,
)
from rupture_generator.rupture.propagator import (
    JumpGraph,
    _jump_graph,
    _maximum_likelihood_tree,
    _root_tree,
    _sample_tree,
)


@pytest.fixture
def unpropagated(scenario):
    realisation = scenario.realisation
    return Realisation(
        realisation.segments, realisation.crs, hypocentre=realisation.hypocentre
    )


@pytest.fixture
def hypocentre(scenario):
    hypocentre = scenario.realisation.hypocentre
    assert hypocentre is not None
    return hypocentre


def _far_copy(geometry, offset_km):
    return Geometry(
        nodes=geometry.nodes + np.array([offset_km, 0.0, 0.0]),
        occupied=geometry.occupied,
        plane_cells=geometry.plane_cells,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"d0_km": 0.0}, "decay length"),
        ({"delta_km": -1.0}, "no sign"),
        ({"max_jump_km": 0.0}, "nothing can jump"),
        ({"probability_cap": 1.0}, "probability cap"),
        ({"reach": "mean"}, "a reach is 'sampled' or 'reference'"),
    ],
)
def test_a_model_no_gap_answers_to_is_refused(kwargs, message):
    with pytest.raises(RuptureGeneratorError, match=message):
        JumpModel(**kwargs)


def test_the_jump_probability_is_capped_decays_and_stops():
    model = JumpModel()
    assert model.probability(0.0) == model.probability_cap
    distances = np.linspace(model.delta_km + 0.1, model.max_jump_km - 0.1, 20)
    probabilities = [model.probability(d) for d in distances]
    assert np.all(np.diff(probabilities) < 0.0)
    assert model.probability(model.max_jump_km) == 0.0


@pytest.mark.parametrize("pick", ["likeliest", "sampled"])
def test_a_path_is_one_tree_grown_from_the_hypocentre(unpropagated, hypocentre, pick):
    if pick == "likeliest":
        realisation = likeliest_path(unpropagated, hypocentre)
    else:
        rng = np.random.default_rng(3)
        realisation = sample_path(unpropagated, hypocentre, rng=rng)
    assert set(realisation.tree) == set(unpropagated)
    assert [name for name, parent in realisation.tree.items() if parent is None] == [
        hypocentre.segment
    ]
    assert len(list(realisation.in_causal_order())) == len(unpropagated)


def test_the_likeliest_path_is_deterministic(unpropagated, hypocentre):
    first = likeliest_path(unpropagated, hypocentre)
    assert likeliest_path(unpropagated, hypocentre).tree == first.tree


def test_one_segment_needs_no_jump(unpropagated, hypocentre):
    name = hypocentre.segment
    alone = Realisation({name: unpropagated[name]}, unpropagated.crs, hypocentre)
    rng = np.random.default_rng(0)
    for realisation in (
        likeliest_path(alone, hypocentre),
        sample_path(alone, hypocentre, rng=rng),
    ):
        assert realisation.tree == {name: None}


@pytest.mark.parametrize("pick", ["likeliest", "sampled"])
def test_a_fault_beyond_reach_of_every_other_is_refused(unpropagated, hypocentre, pick):
    segments = dict(unpropagated.segments)
    segments["far away"] = _far_copy(unpropagated[hypocentre.segment], 500.0)
    system = Realisation(segments, unpropagated.crs, hypocentre)
    with pytest.raises(RuptureGeneratorError, match="connected system"):
        if pick == "likeliest":
            likeliest_path(system, hypocentre)
        else:
            sample_path(system, hypocentre, rng=np.random.default_rng(0))


def test_a_tree_roots_only_at_one_of_its_faults():
    with pytest.raises(RuptureGeneratorError, match="not one of"):
        _root_tree(("a", "b"), [(0, 1)], "c")
    with pytest.raises(RuptureGeneratorError, match="cannot be reached"):
        _root_tree(("a", "b"), [], "a")


def _grid(east, north, depth):
    """A chart from node coordinates broadcast to ``(rows + 1, columns + 1)``."""
    nodes = np.stack(np.broadcast_arrays(east, north, depth), axis=-1).astype(float)
    cells = (nodes.shape[0] - 1, nodes.shape[1] - 1)
    return Geometry(nodes, np.ones(cells, bool), (cells[1],))


@pytest.fixture
def splayed(scenario):
    """A gently dipping interface, and a vertical splay rooted mid-way across it."""
    i, j = np.mgrid[0:21, 0:21]
    interface = _grid(j, i, 10.0 + 0.1 * i)
    i, j = np.mgrid[0:10, 0:11]
    splay = _grid(5.0 + j, 10.0, 2.0 + i)
    return Realisation(
        {"interface": interface, "splay": splay},
        scenario.realisation.crs,
        hypocentre=Hypocentre("interface", 10.0, 2.0),
    )


def test_an_interior_jump_measures_the_gap_between_intersecting_faults(splayed):
    interface, splay = splayed["interface"], splayed["splay"]
    edge_km = splay.nearest_cells_to(interface)[2].min()
    interior_km = splay.nearest_cells_to(interface, interior=True)[2].min()
    assert interior_km < 1.0 < edge_km

    edge = _jump_graph(splayed, JumpModel())
    interior = _jump_graph(splayed, JumpModel(interior=True))
    assert interior.weights[0, 1] == JumpModel().probability_cap
    assert edge.weights[0, 1] < interior.weights[0, 1]


def test_an_interior_jump_starts_the_splay_where_it_meets_the_interface(
    scenario, splayed
):
    sources = {name: segment_source(6.5, 90.0) for name in splayed}
    alone = Realisation(
        {"interface": splayed["interface"]},
        splayed.crs,
        hypocentre=splayed.hypocentre,
    )
    parent = generate(alone, scenario.medium, sources, RuptureModel(), seed=1)
    parent = parent["interface"]
    seeds = {
        interior: first_arrival(
            parent,
            splayed["splay"],
            scenario.medium,
            JumpModel(d0_km=1e-12, delta_km=0.0, interior=interior),
            np.random.default_rng(0),
        )
        for interior in (False, True)
    }
    # The splay's bottom row lies on the interface.
    assert seeds[True].cell[0] == splayed["splay"].cells[0] - 1
    assert seeds[True].time_s < seeds[False].time_s


@pytest.mark.parametrize("pick", ["likeliest", "sampled"])
def test_the_reach_never_changes_the_tree(unpropagated, hypocentre, pick):
    trees = []
    for reach in ("sampled", "reference"):
        model = JumpModel(reach=reach)
        if pick == "likeliest":
            realisation = likeliest_path(unpropagated, hypocentre, model=model)
        else:
            realisation = sample_path(
                unpropagated, hypocentre, rng=np.random.default_rng(3), model=model
            )
        trees.append(realisation.tree)
    assert trees[0] == trees[1]


@pytest.mark.parametrize("pick", ["likeliest", "sampled"])
def test_a_disconnected_system_names_every_fault_and_the_remedies(pick):
    graph = JumpGraph(("alpine", "hope", "kekerengu"), np.zeros((3, 3)))
    with pytest.raises(RuptureGeneratorError) as raised:
        if pick == "likeliest":
            _maximum_likelihood_tree(graph)
        else:
            _sample_tree(graph, np.random.default_rng(0))
    assert str(raised.value) == (
        "alpine, hope, kekerengu do not form a connected system: at least one is "
        "beyond reach of every other, so no single rupture gets to it. Drop it, widen "
        "the model's maximum jump, or generate it as its own earthquake"
    )


def test_a_chain_has_only_one_spanning_tree_to_sample():
    # a - c - b: the walk from b must pass through c, which then needs no walk of
    # its own.
    weights = np.zeros((3, 3))
    weights[0, 2] = weights[2, 0] = 0.3
    weights[1, 2] = weights[2, 1] = 0.6
    graph = JumpGraph(("a", "b", "c"), weights)
    for seed in range(5):
        assert _sample_tree(graph, np.random.default_rng(seed)) == [(1, 2), (2, 0)]


def test_sampled_trees_follow_the_product_of_jump_odds():
    # Each tree of a triangle omits one edge, so P(tree) is proportional to the
    # odds w / (1 - w) of the other two.
    probabilities = {(0, 1): 0.2, (0, 2): 0.5, (1, 2): 0.8}
    weights = np.zeros((3, 3))
    for (u, v), p in probabilities.items():
        weights[u, v] = weights[v, u] = p
    graph = JumpGraph(("a", "b", "c"), weights)
    odds = {edge: p / (1.0 - p) for edge, p in probabilities.items()}
    expected = {
        omitted: float(np.prod([o for edge, o in odds.items() if edge != omitted]))
        for omitted in odds
    }
    total = sum(expected.values())

    rng = np.random.default_rng(11)
    draws = 20_000
    counts = dict.fromkeys(odds, 0)
    for _ in range(draws):
        edges = {(min(u, v), max(u, v)) for u, v in _sample_tree(graph, rng)}
        assert len(edges) == 2
        (omitted,) = set(odds) - edges
        counts[omitted] += 1
    for omitted, weight in expected.items():
        share = weight / total
        spread = np.sqrt(share * (1.0 - share) / draws)
        assert abs(counts[omitted] / draws - share) < 5.0 * spread
