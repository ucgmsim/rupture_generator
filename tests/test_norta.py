import numpy as np
import pytest
import scipy as sp
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from rupture_generator import RuptureGeneratorError
from rupture_generator.sampling import NORMAL, Marginal, latent_correlation
from rupture_generator.sampling.norta import (
    NORTA_CORRELATION_SLACK,
    _correlation_series,
    _doubly_truncated_normal_location,
    _doubly_truncated_normal_spread,
)

GAMMA = Marginal("gamma", 0.75)

LATENTS = hnp.arrays(
    np.float64,
    hnp.array_shapes(max_dims=3, max_side=5),
    elements=st.floats(-10.0, 10.0),
)


@given(LATENTS)
def test_the_normal_marginal_is_the_identity(latent):
    transformed = NORMAL.apply(latent)
    assert transformed.dtype == np.float64
    np.testing.assert_array_equal(transformed, latent)


@given(hnp.arrays(np.float64, (4,), elements=st.floats(-1.0, 1.0)))
def test_two_normal_marginals_need_no_correction(target):
    np.testing.assert_array_equal(latent_correlation(NORMAL, NORMAL, target), target)


def test_integer_latents_come_back_as_floats():
    transformed = NORMAL.apply(np.array([-1, 0, 2]))
    assert transformed.dtype == np.float64
    np.testing.assert_array_equal(transformed, [-1.0, 0.0, 2.0])


def test_a_normal_paired_with_a_gamma_corrects_linearly():
    # The normal transform is He_1 alone, so g(rho) = rho corr(Z, h(Z)): the
    # strongest correlation the pair can share is corr(Z, h(Z)), here by quadrature.
    quantile = GAMMA.apply  # within 6e-7 of the direct quantile
    product, _ = sp.integrate.quad(
        lambda z: z * float(quantile(np.array([z]))[0]) * sp.stats.norm.pdf(z),
        -12.0,
        12.0,
        limit=200,
    )
    strongest = product / GAMMA.coefficient_of_variation
    target = np.linspace(-0.9, 0.9, 7) * strongest
    np.testing.assert_allclose(
        latent_correlation(NORMAL, GAMMA, target), target / strongest, atol=1e-5
    )
    # Swapping the pair transposes nothing: the series is symmetric.
    np.testing.assert_allclose(
        latent_correlation(GAMMA, NORMAL, target),
        latent_correlation(NORMAL, GAMMA, target),
        atol=1e-12,
    )


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_a_correlation_the_marginals_cannot_share_is_refused(sign):
    ceiling = float(np.sum(_correlation_series(NORMAL, GAMMA)))
    assert 0.9 < ceiling < 1.0
    # By the series' odd symmetry the floor is minus the ceiling.
    with pytest.raises(RuptureGeneratorError, match="Gaussian copula"):
        latent_correlation(NORMAL, GAMMA, np.array([0.0, sign * (ceiling + 1e-6)]))
    # Round-off past the extreme is clipped to the extreme latent correlation.
    edge = sign * (ceiling + 0.5 * NORTA_CORRELATION_SLACK)
    np.testing.assert_allclose(
        latent_correlation(NORMAL, GAMMA, np.array([edge])), [sign], atol=1e-9
    )


def test_an_unimplemented_family_fails_when_first_transformed():
    marginal = Marginal("lognormal", 0.5)  # ty: ignore[invalid-argument-type] not a MarginalFamily, on purpose
    with pytest.raises(ValueError, match="marginal distribution: lognormal"):
        marginal.apply(np.zeros(3))


def _cut_normal_moments(location, scale, maximum):
    """Mean and coefficient of variation of the normal cut to [0, maximum], by quadrature."""

    def moment(power):
        return sp.integrate.quad(
            lambda x: x**power * sp.stats.norm.pdf(x, location, scale),
            0.0,
            maximum,
            points=[min(max(location, 0.0), maximum)],
        )[0]

    mass, first, second = moment(0), moment(1), moment(2)
    mean = first / mass
    return mean, np.sqrt(second / mass - mean * mean) / mean


@pytest.mark.parametrize("maximum", [1.5, 4.3])
@pytest.mark.parametrize("scale", [0.1, 1.0, 5.0])
def test_the_doubly_truncated_normal_parent_puts_the_cut_mean_at_one(scale, maximum):
    location = _doubly_truncated_normal_location(scale, maximum)
    mean, spread = _cut_normal_moments(location, scale, maximum)
    assert mean == pytest.approx(1.0, abs=1e-9)
    assert _doubly_truncated_normal_spread(scale, maximum) == pytest.approx(
        spread, rel=1e-8
    )


def test_a_narrow_parent_is_barely_cut():
    # Ten parent deviations from either cut, the parent and its cut agree.
    assert _doubly_truncated_normal_location(0.05, 2.0) == pytest.approx(1.0, abs=1e-12)
    assert _doubly_truncated_normal_spread(0.05, 2.0) == pytest.approx(0.05, rel=1e-12)


def test_the_doubly_truncated_normal_stays_on_its_support():
    marginal = Marginal("doubly_truncated_normal", 0.3, 2.5)
    latent = np.linspace(-40.0, 40.0, 4001)
    values = marginal.apply(latent)
    assert np.all(np.diff(values) >= 0.0)
    assert values[0] == pytest.approx(0.0, abs=1e-9)
    # The table stops NORTA_TAIL short of the top, where the density is thin.
    assert 2.5 - 1e-6 < values[-1] <= 2.5
    # The transformed standard normal has the marginal's unit mean and spread.
    nodes, weights = np.polynomial.hermite_e.hermegauss(160)
    weights = weights / np.sqrt(2.0 * np.pi)
    transformed = marginal.apply(nodes)
    mean = weights @ transformed
    assert mean == pytest.approx(1.0, abs=1e-5)
    assert np.sqrt(weights @ (transformed - mean) ** 2) == pytest.approx(0.3, abs=1e-5)
