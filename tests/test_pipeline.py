import dataclasses

import numpy as np
import pytest
import scipy as sp

from rupture_generator import (
    GaussianRake,
    Geometry,
    Hypocentre,
    Realisation,
    RuptureGeneratorError,
    RuptureModel,
    generate,
)
from rupture_generator.rupture.generator import (
    FIELDS_STREAM,
    JUMP_STREAM,
    JumpSeed,
    SegmentRupture,
    _segment_rng,
    _travel_times,
    first_arrival,
)
from rupture_generator.rupture.medium import Medium, constant_field
from rupture_generator.rupture.models import embed
from rupture_generator.rupture.propagator import JumpModel


def test_every_segment_carries_its_moment(scenario, ruptures):
    for name, rupture in ruptures.items():
        geometry, rigidity = rupture.geometry, rupture.rigidity_pa
        moment = np.sum(rigidity * geometry.areas_km2 * 1e6 * rupture.slip_m)
        assert moment == pytest.approx(scenario.sources[name].moment_nm, rel=1e-12)


def test_fields_are_finite_and_slip_is_positive(ruptures):
    for rupture in ruptures.values():
        for field in (
            rupture.slip_m,
            rupture.rise_time_s,
            rupture.rake_deg,
            rupture.onset_s,
        ):
            assert field.shape == rupture.geometry.cells
            assert np.all(np.isfinite(field))
        assert rupture.slip_m.min() >= 0.0


def test_rise_time_mean_is_the_moments(ruptures):
    for rupture in ruptures.values():
        occupied = rupture.geometry.occupied
        assert rupture.rise_time_s[occupied].mean() == pytest.approx(
            rupture.statistics["rise.mean_s"]
        )


def test_the_rupture_starts_at_the_hypocentre(scenario, ruptures):
    hypocentre = scenario.realisation.hypocentre
    root = ruptures[hypocentre.segment]
    cell = root.geometry.cell_at(hypocentre.strike_km, hypocentre.dip_km)
    assert root.onset_s[cell] == 0.0
    assert root.onset_s.min() >= 0.0
    for name, rupture in ruptures.items():
        if name != hypocentre.segment:
            assert rupture.onset_s.min() > 0.0


def test_a_seed_reproduces_and_segments_are_independent(scenario, ruptures):
    again = generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        scenario.model,
        seed=scenario.seed,
    )
    for name in ruptures:
        assert np.array_equal(ruptures[name].slip_m, again[name].slip_m)
        assert np.array_equal(ruptures[name].onset_s, again[name].onset_s)

    root = scenario.realisation.hypocentre.segment
    alone = Realisation(
        {root: scenario.realisation[root]},
        scenario.realisation.crs,
        hypocentre=scenario.realisation.hypocentre,
    )
    solo = generate(
        alone,
        scenario.medium,
        scenario.sources,
        scenario.model,
        seed=scenario.seed,
    )
    assert np.array_equal(solo[root].slip_m, ruptures[root].slip_m)


def test_the_jump_rule_is_replaceable(scenario):
    calls = []

    def recording(parent, child, medium, model, rng):
        calls.append(child)
        return first_arrival(parent, child, medium, model, rng)

    generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        scenario.model,
        seed=scenario.seed,
        jump_rule=recording,
    )
    assert len(calls) == len(scenario.realisation) - 1


def _first_jump(scenario, ruptures):
    name, parent = next(
        (child, up) for child, up in scenario.realisation.tree.items() if up
    )
    return ruptures[parent], ruptures[name].geometry


def _pairs_within(child, parent, reach_km):
    """Every pair of edge cells within ``reach_km``, by brute force."""
    landings, child_km = child.jump_cells()
    departures, parent_km = parent.jump_cells()
    distance_km = np.linalg.norm(child_km[:, None] - parent_km[None], axis=-1)
    c, p = np.nonzero(distance_km <= reach_km)
    return (landings[0][c], landings[1][c]), (departures[0][p], departures[1][p])


def test_a_jump_crosses_the_nearest_gap_at_the_shear_speed(scenario, ruptures):
    # With a decay length near zero, the jump crosses only the nearest gap.
    drawn, child = _first_jump(scenario, ruptures)
    model = JumpModel(d0_km=1e-12, delta_km=0.0)
    seed = first_arrival(drawn, child, scenario.medium, model, np.random.default_rng(0))
    _, _, gap_km = child.nearest_cells_to(drawn.geometry)
    landings, departures = _pairs_within(
        child, drawn.geometry, float(gap_km.min()) + 1e-9
    )
    arrival_s = drawn.onset_s[departures] + scenario.medium.crossing_time_s(
        drawn.geometry.centres[departures], child.centres[landings]
    )
    assert seed.time_s == pytest.approx(float(arrival_s.min()))


def test_a_jump_lands_within_its_reach(scenario, ruptures):
    drawn, child = _first_jump(scenario, ruptures)
    model = JumpModel()
    landings, _, gap_km = child.nearest_cells_to(drawn.geometry)
    for draw in range(20):
        reach_km = model.reach_km(float(gap_km.min()), np.random.default_rng(draw))
        seed = first_arrival(
            drawn, child, scenario.medium, model, np.random.default_rng(draw)
        )
        landed = (landings[0] == seed.cell[0]) & (landings[1] == seed.cell[1])
        assert gap_km[landed].min() <= reach_km


def test_how_a_front_jumps_never_moves_the_fields(scenario, ruptures):
    elsewhere = generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        scenario.model,
        seed=scenario.seed,
        jump_propagation_model=JumpModel(d0_km=10.0),
    )
    for name, rupture in ruptures.items():
        assert np.array_equal(rupture.slip_m, elsewhere[name].slip_m)
        assert np.array_equal(rupture.rake_deg, elsewhere[name].rake_deg)


FIELDS = ("slip_m", "rise_time_s", "rake_deg", "onset_s", "speed_km_s")


def test_the_same_settings_by_name_draw_the_same_rupture(scenario, ruptures):
    by_name = generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        dict.fromkeys(scenario.realisation, scenario.model),
        seed=scenario.seed,
        jump_propagation_model=scenario.jump_model,
    )
    for name, rupture in ruptures.items():
        for field in FIELDS:
            assert np.array_equal(
                getattr(rupture, field), getattr(by_name[name], field)
            )


def test_settings_on_one_segment_leave_the_others_alone(scenario, ruptures):
    root = scenario.realisation.hypocentre.segment
    other = next(name for name in scenario.realisation if name != root)
    settings = dict.fromkeys(scenario.realisation, scenario.model)
    settings[other] = dataclasses.replace(
        scenario.model, rake=GaussianRake(sigma_deg=0.0)
    )
    drawn = generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        settings,
        seed=scenario.seed,
        jump_propagation_model=scenario.jump_model,
    )
    for field in FIELDS:
        assert np.array_equal(
            getattr(ruptures[root], field), getattr(drawn[root], field)
        )
    assert np.all(drawn[other].rake_deg == scenario.sources[other].rake_deg)
    # Each model draws from a generator of its own. The other fields never change.
    for field in FIELDS:
        if field != "rake_deg":
            assert np.array_equal(
                getattr(ruptures[other], field), getattr(drawn[other], field)
            )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda settings, name: settings.pop(name), "have no model"),
        (lambda settings, _: settings.update(nope=RuptureModel()), "not segments"),
    ],
    ids=["missing", "unknown"],
)
def test_settings_by_name_cover_exactly_the_segments(scenario, change, message):
    settings = dict.fromkeys(scenario.realisation, scenario.model)
    change(settings, next(iter(settings)))
    with pytest.raises(RuptureGeneratorError, match=message):
        generate(
            scenario.realisation,
            scenario.medium,
            scenario.sources,
            settings,
            seed=scenario.seed,
        )


@pytest.mark.parametrize("nearest_km", [0.0, 0.5, 4.0, 40.0])
def test_a_reach_is_conditioned_on_the_gap_it_crossed(nearest_km):
    model = JumpModel()
    rng = np.random.default_rng(1)
    reaches = np.array([model.reach_km(nearest_km, rng) for _ in range(4000)])
    assert reaches.min() >= nearest_km
    assert reaches.max() <= max(model.max_jump_km, nearest_km)
    if nearest_km + 5 * model.d0_km < model.max_jump_km:
        floor = max(model.delta_km, nearest_km)
        assert reaches.mean() == pytest.approx(floor + model.d0_km, rel=0.1)


@pytest.mark.parametrize("nearest_km", [0.0, 0.5, 4.0, 13.0, 40.0])
def test_a_reference_reach_is_the_mean_and_draws_nothing(nearest_km):
    model = JumpModel(reach="reference")
    rng = np.random.default_rng(1)
    state = rng.bit_generator.state
    reach_km = model.reach_km(nearest_km, rng)
    assert rng.bit_generator.state == state
    floor = max(model.delta_km, nearest_km)
    assert reach_km == min(floor + model.d0_km, max(model.max_jump_km, nearest_km))


def test_a_reference_reach_is_the_mean_of_the_sampled_one():
    sampled = JumpModel(max_jump_km=1e6)
    reference = dataclasses.replace(sampled, reach="reference")
    rng = np.random.default_rng(2)
    reaches = [sampled.reach_km(4.0, rng) for _ in range(20000)]
    assert reference.reach_km(4.0, rng) == pytest.approx(np.mean(reaches), rel=0.02)


def _jumping_with(seed):
    """A jump rule that draws from its own seed, in place of the jump stream's."""

    def rule(parent, child, medium, model, _):
        return first_arrival(parent, child, medium, model, np.random.default_rng(seed))

    return rule


def test_a_reference_jump_is_the_same_whatever_its_generator(scenario):
    model = JumpModel(reach="reference")
    drawn = [
        generate(
            scenario.realisation,
            scenario.medium,
            scenario.sources,
            scenario.model,
            seed=scenario.seed,
            jump_rule=_jumping_with(seed),
            jump_propagation_model=model,
        )
        for seed in range(4)
    ]
    for name in scenario.realisation:
        for other in drawn[1:]:
            for field in FIELDS:
                assert np.array_equal(
                    getattr(drawn[0][name], field), getattr(other[name], field)
                )


def test_a_reference_jump_lands_at_the_first_arrival_within_the_mean_reach(
    scenario, ruptures
):
    drawn, child = _first_jump(scenario, ruptures)
    model = JumpModel(reach="reference")
    _, _, gap_km = child.nearest_cells_to(drawn.geometry)
    reach_km = model.reach_km(float(gap_km.min()), np.random.default_rng(0))
    landings, departures = _pairs_within(child, drawn.geometry, reach_km)
    seeds = {
        first_arrival(drawn, child, scenario.medium, model, np.random.default_rng(k))
        for k in range(10)
    }
    assert len(seeds) == 1
    # The same generators move a sampled jump, so the test has something to fix.
    sampled = {
        first_arrival(
            drawn, child, scenario.medium, JumpModel(), np.random.default_rng(k)
        )
        for k in range(10)
    }
    assert len(sampled) > 1
    arrival_s = drawn.onset_s[departures] + scenario.medium.crossing_time_s(
        drawn.geometry.centres[departures], child.centres[landings]
    )
    assert seeds.pop().time_s == float(arrival_s.min())


def _vertical(start, end, cells):
    """A one-row vertical chart from ``start`` to ``end``, 1 km deep."""
    t = np.linspace(0.0, 1.0, cells + 1)
    xy = np.asarray(start) + t[:, None] * (np.asarray(end) - np.asarray(start))
    nodes = np.stack(
        [np.column_stack([xy, np.full(cells + 1, depth)]) for depth in (0.0, 1.0)]
    )
    return Geometry(nodes, np.ones((1, cells), bool), (cells,))


@pytest.mark.parametrize(
    "bound_km_s", [3.5, 7.0, np.inf], ids=["tight", "loose", "none"]
)
def test_a_jump_can_depart_from_any_parent_cell_within_reach(bound_km_s):
    # A 40 km parent ruptures west to east at 0.8 Vs, and a short child runs north
    # from 2 km off its middle. Cutting the corner from an earlier parent cell, at the
    # full shear speed, beats crossing from the parent cell nearest any child cell.
    # The speed bound only prunes, so every bound finds the same crossing.
    vs = 3.5
    medium = Medium(constant_field(vs), constant_field(2.7), bound_km_s)
    geometry = _vertical((0.0, 0.0), (40.0, 0.0), 40)
    onset = geometry.centres[..., 0] / (0.8 * vs)
    zero = np.zeros_like(onset)
    parent = SegmentRupture(
        geometry=geometry,
        slip_m=zero,
        rise_time_s=zero,
        rake_deg=zero,
        onset_s=onset,
        speed_km_s=zero + 0.8 * vs,
        shear_speed_km_s=zero + vs,
        density_g_cm3=zero + 2.7,
        dt_s=0.05,
        beta=None,
    )
    child = _vertical((20.5, 2.0), (20.5, 6.0), 4)
    model = JumpModel(reach="reference")
    seed = first_arrival(parent, child, medium, model, np.random.default_rng(0))

    _, _, gap_km = child.nearest_cells_to(geometry)
    reach_km = model.reach_km(float(gap_km.min()), np.random.default_rng(0))
    p, c = geometry.centres.reshape(-1, 3), child.centres.reshape(-1, 3)
    distance_km = np.linalg.norm(p[:, None] - c[None], axis=-1)
    arrival_s = onset.reshape(-1)[:, None] + distance_km / vs
    arrival_s[distance_km > reach_km] = np.inf
    _, first = np.unravel_index(np.argmin(arrival_s), arrival_s.shape)
    assert seed.time_s == pytest.approx(float(arrival_s.min()))
    assert seed.cell == (0, int(first))


def test_the_default_reach_draws_as_before():
    # By default a model draws the same distance it drew before it had a choice.
    model = JumpModel()
    assert model.reach == "sampled"
    for nearest_km in (0.0, 4.0, 40.0):
        exponential = np.random.default_rng(5).exponential(model.d0_km)
        before = min(
            max(model.delta_km, nearest_km) + exponential,
            max(model.max_jump_km, nearest_km),
        )
        assert model.reach_km(nearest_km, np.random.default_rng(5)) == before


def test_each_rupture_keeps_the_rock_it_read(scenario, ruptures):
    for rupture in ruptures.values():
        centres = rupture.geometry.centres
        np.testing.assert_array_equal(
            rupture.shear_speed_km_s, scenario.medium.shear_speed_km_s(centres)
        )
        np.testing.assert_array_equal(
            rupture.density_g_cm3, scenario.medium.density_g_cm3(centres)
        )


def test_default_settings_are_valid():
    RuptureModel()


@pytest.mark.parametrize(
    "build",
    [
        lambda r: Realisation(
            r.segments, r.crs, hypocentre=Hypocentre("nope", 0.0, 0.0)
        ),
        lambda r: Realisation(
            r.segments, r.crs, r.hypocentre, {name: name for name in r}
        ),
    ],
    ids=["unknown hypocentre segment", "cyclic tree"],
)
def test_inconsistent_realisations_are_refused(scenario, build):
    with pytest.raises(RuptureGeneratorError):
        build(scenario.realisation)


def test_high_slip_ruptures_early(scenario, ruptures):
    # A positive correlation pulls high-slip patches ahead of the coherent front,
    # as genslip's negative tsfac does.
    assert scenario.model.perturbation.correlation > 0.0
    hypocentre = scenario.realisation.hypocentre
    root = ruptures[hypocentre.segment]
    seed = JumpSeed(root.geometry.cell_at(hypocentre.strike_km, hypocentre.dip_km), 0.0)
    front_s = _travel_times(root.geometry, root.speed_km_s, seed)
    occupied = root.geometry.occupied
    rho = sp.stats.spearmanr(
        root.slip_m[occupied], (root.onset_s - front_s)[occupied]
    ).statistic
    assert rho < -0.3


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_a_seed_that_is_not_a_non_negative_integer_is_refused(scenario, seed):
    with pytest.raises(RuptureGeneratorError, match="non-negative integer"):
        generate(
            scenario.realisation,
            scenario.medium,
            scenario.sources,
            scenario.model,
            seed=seed,
        )


def test_a_jump_draws_apart_from_any_segments_fields(scenario):
    # A segment named like the old jump key must not share the jump's stream.
    name = next(iter(scenario.realisation))
    jump = _segment_rng(scenario.seed, JUMP_STREAM, name).random(4)
    fields = _segment_rng(scenario.seed, FIELDS_STREAM, f"jump:{name}").random(4)
    same = _segment_rng(scenario.seed, FIELDS_STREAM, name).random(4)
    assert not np.array_equal(jump, fields)
    assert not np.array_equal(jump, same)


def test_a_second_draw_reuses_every_embedding(scenario):
    embed.cache_clear()
    misses = []
    for _ in range(2):
        generate(
            scenario.realisation,
            scenario.medium,
            scenario.sources,
            scenario.model,
            seed=scenario.seed,
        )
        misses.append(embed.cache_info().misses)
    # Slip's latent and rake each embed once per segment; rise time and the onset
    # displacement draw from slip's embedding.
    assert misses == [2 * len(scenario.realisation)] * 2
