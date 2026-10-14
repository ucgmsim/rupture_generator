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
from rupture_generator.rupture.models import CUBE_ROOT_DYNE_CM

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
        occupied = rupture.geometry.occupied
        assert rupture.rise_time_s[occupied].mean() == pytest.approx(
            rupture.statistics["rise.mean_s"]
        )


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
