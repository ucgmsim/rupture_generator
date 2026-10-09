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


def test_every_segment_carries_its_moment(scenario, ruptures):
    for name, rupture in ruptures.items():
        geometry, rigidity = rupture.geometry, scenario.materials[name].rigidity_pa
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
        scenario.materials,
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
        scenario.materials,
        scenario.sources,
        scenario.settings,
        seed=scenario.seed,
    )
    assert np.array_equal(solo[root].slip_m, ruptures[root].slip_m)


def test_the_jump_rule_is_replaceable(scenario):
    calls = []

    def recording(parent, child):
        calls.append(child)
        return jump_seed(parent, child)

    generate(
        scenario.realisation,
        scenario.materials,
        scenario.sources,
        scenario.settings,
        seed=scenario.seed,
        jump=recording,
    )
    assert len(calls) == len(scenario.realisation) - 1


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
