import numpy as np
import pytest

from rupture_generator import (
    Geometry,
    JumpModel,
    Realisation,
    RuptureGeneratorError,
    likeliest_path,
    sample_path,
)
from rupture_generator.rupture.propagator import _root_tree


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
