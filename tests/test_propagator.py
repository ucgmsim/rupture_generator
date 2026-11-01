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
from rupture_generator.rupture.propagator import _jump_graph, _root_tree


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
