import dataclasses

import numpy as np
import pytest
import scipy as sp

from rupture_generator import (
    Hypocentre,
    RakeSettings,
    Realisation,
    RuptureGeneratorError,
    RuptureSettings,
    generate,
)
from rupture_generator.rupture.generator import (
    FIELDS_STREAM,
    JUMP_STREAM,
    JumpSeed,
    _segment_rng,
    _travel_times,
    first_arrival,
)
from rupture_generator.rupture.propagator import JumpModel
from rupture_generator.sampling import sampler
from rupture_generator.sampling.field import SAMPLER_CACHE_SIZE


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
            rupture.rise_time_mean_s
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
        scenario.settings,
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
        scenario.settings,
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
        scenario.settings,
        seed=scenario.seed,
        jump_rule=recording,
    )
    assert len(calls) == len(scenario.realisation) - 1


def _first_jump(scenario, ruptures):
    name, parent = next(
        (child, up) for child, up in scenario.realisation.tree.items() if up
    )
    return ruptures[parent], ruptures[name].geometry


def test_a_jump_crosses_the_nearest_gap_at_the_shear_speed(scenario, ruptures):
    # With a decay length near zero, the jump crosses only the nearest gap.
    drawn, child = _first_jump(scenario, ruptures)
    model = JumpModel(d0_km=1e-12, delta_km=0.0)
    seed = first_arrival(drawn, child, scenario.medium, model, np.random.default_rng(0))
    landings, departures, gap_km = child.nearest_cells_to(drawn.geometry)
    nearest = gap_km <= gap_km.min() + 1e-9
    arrival_s = drawn.onset_s[departures] + scenario.medium.crossing_time_s(
        drawn.geometry.centres[departures], child.centres[landings]
    )
    assert seed.time_s == pytest.approx(float(arrival_s[nearest].min()))


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
        scenario.settings,
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
        dict.fromkeys(scenario.realisation, scenario.settings),
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
    settings = dict.fromkeys(scenario.realisation, scenario.settings)
    settings[other] = dataclasses.replace(
        scenario.settings, rake=RakeSettings(sigma_deg=0.0)
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
    assert np.array_equal(drawn[other].slip_m, ruptures[other].slip_m)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda settings, name: settings.pop(name), "have no settings"),
        (lambda settings, _: settings.update(nope=RuptureSettings()), "not segments"),
    ],
    ids=["missing", "unknown"],
)
def test_settings_by_name_cover_exactly_the_segments(scenario, change, message):
    settings = dict.fromkeys(scenario.realisation, scenario.settings)
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
    RuptureSettings()


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
    assert scenario.settings.timing.correlation > 0.0
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
            scenario.settings,
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
    sampler.cache_clear()
    for _ in range(2):
        generate(
            scenario.realisation,
            scenario.medium,
            scenario.sources,
            scenario.settings,
            seed=scenario.seed,
        )
    assert sampler.cache_info().hits == 2 * len(scenario.realisation)
    assert SAMPLER_CACHE_SIZE >= 2 * 20
