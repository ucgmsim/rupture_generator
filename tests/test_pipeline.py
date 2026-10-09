import numpy as np
import pytest

from rupture_generator import (
    Hypocentre,
    Realisation,
    RuptureGeneratorError,
    RuptureSettings,
    generate,
)
from rupture_generator.rupture.generator import jump_seed
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
        return jump_seed(parent, child, medium, model, rng)

    generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        scenario.settings,
        seed=scenario.seed,
        jump=recording,
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
    seed = jump_seed(drawn, child, scenario.medium, model, np.random.default_rng(0))
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
        seed = jump_seed(
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
        jump_model=JumpModel(d0_km=10.0),
    )
    for name, rupture in ruptures.items():
        assert np.array_equal(rupture.slip_m, elsewhere[name].slip_m)
        assert np.array_equal(rupture.rake_deg, elsewhere[name].rake_deg)


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
