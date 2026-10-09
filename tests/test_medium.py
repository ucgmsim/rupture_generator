import dataclasses

import numpy as np
import pytest

from rupture_generator import (
    FaultProfiles,
    Layers,
    Medium,
    RuptureGeneratorError,
    constant_field,
    generate,
    layered_medium,
)


def test_a_uniform_crossing_is_length_over_speed():
    medium = Medium(constant_field(3.0), constant_field(2.7))
    start = np.array([[0.0, 0.0, 5.0], [1.0, 2.0, 3.0]])
    end = np.array([[3.0, 4.0, 5.0], [1.0, 2.0, 3.0]])
    np.testing.assert_allclose(medium.crossing_time_s(start, end), [5.0 / 3.0, 0.0])


def test_a_layered_crossing_is_the_harmonic_mean():
    # Straight down through 2 km at 2 km/s then 2 km at 4 km/s: 1 s + 0.5 s. The
    # arithmetic mean of the speeds, 3 km/s, would give 4/3 s.
    medium = layered_medium(Layers(np.array([2.0, 10.0])), [2.0, 4.0], [2.5, 2.7])
    crossing_s = medium.crossing_time_s(np.array([0, 0, 0.0]), np.array([0, 0, 4.0]))
    assert crossing_s == pytest.approx(1.5)


def test_fields_take_any_leading_shape():
    medium = layered_medium(Layers(np.array([2.0, 10.0])), [2.0, 4.0], [2.5, 2.7])
    positions = np.zeros((4, 5, 6, 3))
    assert medium.shear_speed_km_s(positions).shape == (4, 5, 6)
    assert medium.rigidity_pa(positions).shape == (4, 5, 6)
    assert medium.crossing_time_s(positions, positions + 1.0).shape == (4, 5, 6)


@pytest.mark.parametrize(
    ("profile", "value", "message"),
    [
        ("rise_time_slip_weight", 1.5, "slip weight"),
        ("rupture_speed_factor", 0.0, "rupture speed factor"),
        ("rise_time_factor", -1.0, "rise-time factor"),
    ],
)
def test_a_profile_out_of_range_is_refused(scenario, profile, value, message):
    profiles = dataclasses.replace(
        scenario.settings.profiles, **{profile: constant_field(value)}
    )
    settings = dataclasses.replace(scenario.settings, profiles=profiles)
    with pytest.raises(RuptureGeneratorError, match=message):
        generate(
            scenario.realisation,
            scenario.medium,
            scenario.sources,
            settings,
            seed=scenario.seed,
        )


def test_a_medium_that_is_not_rock_is_refused(scenario):
    medium = Medium(constant_field(-1.0), constant_field(2.7))
    with pytest.raises(RuptureGeneratorError, match="shear speed"):
        generate(
            scenario.realisation,
            medium,
            scenario.sources,
            scenario.settings,
            seed=scenario.seed,
        )


def test_profiles_default_to_unmodified():
    profiles = FaultProfiles()
    positions = np.zeros((3, 3))
    for field in dataclasses.fields(profiles):
        np.testing.assert_array_equal(getattr(profiles, field.name)(positions), 1.0)
