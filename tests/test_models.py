"""Each model draws apart from the others, and the generator checks what they return."""

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest
import scipy as sp
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from rupture_generator import (
    Drawn,
    RuptureGeneratorError,
    SlipField,
    gaussian_anamorphosis,
    generate,
)
from rupture_generator import config as rupture_config
from rupture_generator.formats.srf import write_rupture
from rupture_generator.rupture.models import CUBE_ROOT_DYNE_CM, _floored

from .conftest import EXAMPLE, SPACING_KM

README = Path(__file__).parents[1] / "README.md"


@pytest.fixture(scope="module")
def given_slip():
    """The README's ``GivenSlip``, run as the README states it."""
    block = next(
        block
        for block in re.findall(r"```python\n(.*?)```", README.read_text(), re.DOTALL)
        if "class GivenSlip" in block
    )
    namespace: dict = {}
    exec(compile(block, str(README), "exec"), namespace)  # noqa: S102
    return namespace["GivenSlip"]


def _generate(scenario, model):
    return generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        model,
        seed=scenario.seed,
        jump_propagation_model=scenario.jump_model,
    )


def _with_slips(scenario, given_slip, slips):
    return _generate(
        scenario,
        {
            name: dataclasses.replace(scenario.model, slip=given_slip(slips[name]))
            for name in scenario.realisation
        },
    )


def _moment(rupture, slip_m):
    return float(
        np.sum(rupture.rigidity_pa * rupture.geometry.areas_km2 * 1e6 * slip_m)
    )


@pytest.fixture(scope="module")
def flipped(ruptures):
    """Each drawn slip reversed along strike: a slip the seed would never draw."""
    return {
        name: np.fliplr(rupture.slip_m).copy() for name, rupture in ruptures.items()
    }


def test_each_statistic_names_its_model(ruptures):
    for rupture in ruptures.values():
        assert set(rupture.statistics) == {
            "rise.mean_s",
            "rise.alpha_t",
            "speed.alpha_t",
            "perturbation.scale_s",
        }


def test_a_function_serves_as_a_model(scenario, ruptures):
    def level(context, rng):
        return Drawn(np.full(context.geometry.cells, 42.0))

    drawn = _generate(scenario, dataclasses.replace(scenario.model, rake=level))
    for name, rupture in drawn.items():
        assert np.all(rupture.rake_deg == 42.0)
        assert np.array_equal(rupture.onset_s, ruptures[name].onset_s)


def test_a_given_slip_is_kept_and_the_rest_drawn_around_it(
    scenario, ruptures, given_slip, flipped
):
    drawn = _with_slips(scenario, given_slip, flipped)
    for name, rupture in drawn.items():
        assert rupture.slip_m.tobytes() == flipped[name].tobytes()
        # Rake and speed never read slip's latent, and draw from their own streams.
        assert np.array_equal(rupture.rake_deg, ruptures[name].rake_deg)
        assert np.array_equal(rupture.speed_km_s, ruptures[name].speed_km_s)
        assert not np.array_equal(rupture.rise_time_s, ruptures[name].rise_time_s)
        # Rise time follows the given slip's ranks, not the drawn slip's.
        occupied = rupture.geometry.occupied
        assert (
            sp.stats.spearmanr(
                rupture.slip_m[occupied], rupture.rise_time_s[occupied]
            ).statistic
            > 0.5
        )


def test_a_given_slip_sets_the_moment_the_kinematics_follow(
    scenario, given_slip, flipped
):
    once = _with_slips(scenario, given_slip, flipped)
    # A tenth more moment lies within the magnitude tolerance.
    larger = _with_slips(
        scenario, given_slip, {name: 1.1 * slip for name, slip in flipped.items()}
    )
    coefficient = scenario.model.rise.coefficient * CUBE_ROOT_DYNE_CM
    for name, rupture in once.items():
        assert rupture.statistics["rise.mean_s"] == pytest.approx(
            coefficient
            * np.cbrt(_moment(rupture, flipped[name]))
            * rupture.statistics["rise.alpha_t"]
        )
        # Normal scores ignore scale, so only the mean moves.
        np.testing.assert_allclose(
            larger[name].rise_time_s, rupture.rise_time_s * np.cbrt(1.1), rtol=1e-12
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda slip: slip[:-1], "slip is shaped"),
        (lambda slip: np.where(slip == slip.max(), -1.0, slip), "slip runs"),
        (lambda slip: np.where(slip == slip.max(), np.nan, slip), "slip runs"),
        (lambda slip: 0.0 * slip, "zero moment"),
        (lambda slip: 2.0 * slip, "moment magnitude of"),
    ],
    ids=["shape", "negative", "nan", "zero", "moment"],
)
def test_the_generator_refuses_a_slip_that_does_not_fit(
    scenario, ruptures, change, message
):
    name = scenario.realisation.hypocentre.segment
    slip_m = change(ruptures[name].slip_m)

    def bad(context, rng):
        return SlipField(slip_m, np.zeros(slip_m.shape), context.source.covariance)

    with pytest.raises(RuptureGeneratorError, match=message):
        _generate(scenario, dataclasses.replace(scenario.model, slip=bad))


@pytest.mark.parametrize(
    ("field", "returned", "message"),
    [
        ("rise", lambda cells: Drawn(np.ones((1, 1))), "rise time is shaped"),
        ("rise", lambda cells: Drawn(-np.ones(cells)), "rise time runs"),
        ("rake", lambda cells: Drawn(np.full(cells, np.inf)), "rake runs"),
        ("speed", lambda cells: Drawn(np.zeros(cells)), "rupture speed runs"),
    ],
)
def test_the_generator_refuses_a_field_that_does_not_fit(
    scenario, field, returned, message
):
    def bad(context, *_):
        return returned(context.geometry.cells)

    with pytest.raises(RuptureGeneratorError, match=message):
        _generate(scenario, dataclasses.replace(scenario.model, **{field: bad}))


@given(
    hnp.arrays(
        np.float64,
        hnp.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=8),
        elements=st.integers(0, 5).map(float),
    ),
    st.data(),
)
def test_normal_scores_follow_the_values_and_ignore_off_fault_cells(values, data):
    occupied = data.draw(hnp.arrays(bool, values.shape))
    scores = gaussian_anamorphosis(values, occupied)
    assert np.all(scores[~occupied] == 0.0)
    on, at = values[occupied], scores[occupied]
    order = np.argsort(on, kind="stable")
    # Monotone, with equal values sharing a score.
    assert np.all(np.diff(at[order]) >= 0.0)
    assert np.all((np.diff(on[order]) == 0.0) <= (np.diff(at[order]) == 0.0))
    # Distinct values' mid-ranks are symmetric about the median. Their scores sum to
    # nothing.
    if np.unique(on).size == on.size:
        assert abs(at.sum()) <= 1e-9 * max(on.size, 1)


# ------------------------------------------------------------- the rise floor

DT_S = 0.005
"""The floor the workflow sets, its slip-rate sample interval."""

DEPTH_FACTOR = rupture_config.Profile(
    depth_km=[0.0, 3.0, 10.0, 14.0], values=[2.0, 1.0, 1.0, 2.0]
)
"""Twice the rise time at the top and the bottom, as the workflow sets it."""


def _rise_draw(magnitude, *, minimum_s, seed=7, **rise):
    """The example at the test resolution, both segments near ``magnitude``."""
    loaded = rupture_config.load(EXAMPLE)
    scenario = rupture_config.build(
        dataclasses.replace(
            loaded,
            seed=seed,
            geometry=dataclasses.replace(loaded.geometry, spacing_km=SPACING_KM),
            source=dataclasses.replace(
                loaded.source, magnitudes={"bent": magnitude, "next": magnitude - 0.3}
            ),
            rise=dataclasses.replace(
                loaded.rise, minimum_s=minimum_s, factor=DEPTH_FACTOR, **rise
            ),
        )
    )
    return _generate(scenario, scenario.model)


FLOOR_CASES = [
    (spread, exponent, magnitude)
    for spread in (0.5, 0.75)
    for exponent in (0.5, 1.0, 1.5)
    for magnitude in (4.5, 6.0, 7.5)
    # test_a_floor_past_the_marginals_reach_names_the_segment draws this one.
    if (spread, exponent, magnitude) != (0.75, 1.5, 4.5)
]


@pytest.mark.parametrize(("spread", "exponent", "magnitude"), FLOOR_CASES)
@pytest.mark.parametrize("seed", [1, 2])
def test_every_occupied_cell_keeps_the_rise_floor_and_the_mean_holds(
    spread, exponent, magnitude, seed
):
    drawn = _rise_draw(
        magnitude,
        minimum_s=DT_S,
        seed=seed,
        family="truncated_normal",
        coefficient_of_variation=spread,
        slip_exponent=exponent,
    )
    for rupture in drawn.values():
        rise_s = rupture.rise_time_s[rupture.geometry.occupied]
        assert rise_s.min() >= DT_S
        assert rise_s.mean() == pytest.approx(
            rupture.statistics["rise.mean_s"], rel=1e-12
        )
        assert 0.0 < rupture.statistics["rise.minimum"] < 1.0
        # A cell drops to the floor only where the drawn mean runs over its expected
        # value, and then few do: a quarter of a percent at Mw 4.5.
        assert 0.0 <= rupture.statistics["rise.floored"] < 0.01


def test_the_floor_lifts_rise_times_a_pulse_cannot_sample(tmp_path):
    # Small, uncorrelated with slip, and linear in the marginal: some slipping cell
    # draws a rise time that rounds to no samples at all.
    rise = {
        "family": "truncated_normal",
        "coefficient_of_variation": 0.5,
        "slip_exponent": 1.0,
        "correlation": 0.0,
    }
    bare = _rise_draw(4.5, minimum_s=None, **rise)
    assert min(r.rise_time_s[r.geometry.occupied].min() for r in bare.values()) < (
        0.5 * DT_S
    )
    with pytest.raises(RuptureGeneratorError, match="rounds to zero samples"):
        write_rupture(str(tmp_path / "bare.srf"), bare)

    floored = _rise_draw(4.5, minimum_s=DT_S, **rise)
    for name, rupture in floored.items():
        assert rupture.rise_time_s[rupture.geometry.occupied].min() >= DT_S
        # The floor reshapes rise time alone.
        assert np.array_equal(rupture.slip_m, bare[name].slip_m)
        assert np.array_equal(rupture.rake_deg, bare[name].rake_deg)
    write_rupture(str(tmp_path / "floored.srf"), floored)


def test_a_floor_no_segment_can_keep_names_the_segment():
    with pytest.raises(
        RuptureGeneratorError,
        match="segment 'bent': a rise-time floor of 100.0 s is not below",
    ):
        _rise_draw(6.0, minimum_s=100.0)


def test_a_floor_past_the_marginals_reach_names_the_segment():
    # At Mw 4.2 the floor is a seventh of the mean rise time, and at a power of 1.5
    # it needs a cut past 0.2475, 99% of the way to the 0.25 a spread of 0.75 allows.
    with pytest.raises(
        RuptureGeneratorError,
        match="segment 'next': a rise-time floor of 0.005 s .* cannot be cut that high",
    ):
        _rise_draw(4.5, minimum_s=DT_S, slip_exponent=1.5)


@pytest.mark.parametrize("minimum_s", [0.05, 0.4, 0.8, 0.99])
@pytest.mark.parametrize("seed", range(4))
def test_holding_cells_at_the_floor_keeps_the_mean(minimum_s, seed):
    rng = np.random.default_rng(seed)
    pattern = rng.lognormal(sigma=1.0, size=(12, 20))
    occupied = rng.random(pattern.shape) < 0.8
    rise_s, floored = _floored(pattern, occupied, 1.0, minimum_s)
    inside, scaled = rise_s[occupied], pattern[occupied]
    assert inside.min() >= minimum_s
    assert inside.mean() == pytest.approx(1.0, rel=1e-12)
    held = inside == minimum_s
    assert floored == pytest.approx(held.mean())
    # One scale lifts the cells over the floor, and that scale leaves each cell at the
    # floor under it.
    ratios = inside[~held] / scaled[~held]
    np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)
    assert np.all(scaled[held] * ratios[0] <= minimum_s)


def test_a_floor_under_every_cell_is_the_plain_rescale():
    pattern = np.random.default_rng(0).lognormal(size=(6, 9))
    occupied = np.ones(pattern.shape, bool)
    rise_s, floored = _floored(pattern, occupied, 2.0, 1e-6)
    assert floored == 0.0
    assert rise_s.tobytes() == (pattern * (2.0 / float(pattern.mean()))).tobytes()
