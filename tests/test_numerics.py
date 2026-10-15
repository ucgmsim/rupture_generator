import io
import itertools
import json

import numpy as np
import pyproj
import pytest
import scipy as sp

from rupture_generator import geometry_from_geojson, moment_from_magnitude
from rupture_generator.rupture.source import segment_source
from rupture_generator.sampling import (
    Covariance,
    Grid,
    Marginal,
    PreCorrected,
    VonKarman,
)
from rupture_generator.sampling.field import (
    MAXIMUM_VARIANCE_DEFICIT,
    WRAP_TOLERANCE,
    _embed,
)

TRACE = [[172.0, -42.0], [172.3, -41.8], [172.7, -41.7]]

SECTION = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": TRACE,
            },
            "properties": {
                "name": "bent",
                "dip_deg": 60,
                "dip_direction_deg": 300,
                "upper_depth_km": 0,
                "lower_depth_km": 15,
            },
        }
    ],
}


def _bent():
    return geometry_from_geojson(
        io.StringIO(json.dumps(SECTION)), pyproj.CRS("EPSG:2193")
    )["bent"]


def test_a_bent_trace_hangs_as_one_translated_sheet():
    coarse = _bent()
    # Every node drops by the same vector, so neighbouring planes share their
    # junction all the way down and no seam opens.
    down = coarse.nodes[1] - coarse.nodes[0]
    np.testing.assert_allclose(down, down[:1].repeat(len(down), axis=0), atol=1e-12)
    reach_km = 15.0 / np.tan(np.radians(60.0))
    assert np.linalg.norm(down[0, :2]) == pytest.approx(reach_km, rel=1e-12)
    assert down[0, 2] == pytest.approx(15.0)


def test_a_plane_off_square_to_the_dip_direction_dips_more_steeply():
    trace = np.array(TRACE)
    strikes_deg, _, _ = pyproj.Geod(ellps="WGS84").inv(
        trace[:-1, 0], trace[:-1, 1], trace[1:, 0], trace[1:, 1]
    )
    across = np.abs(np.sin(np.radians(300.0 - strikes_deg)))
    expected = np.degrees(np.arctan(np.tan(np.radians(60.0)) / across))
    # The trace dips to the left. The section reverses it, and its planes run backwards.
    # The 0.05 degrees is the projection's grid convergence, which the geodesic
    # strikes here don't see.
    np.testing.assert_allclose(_bent().dip_deg[0], expected[::-1], atol=0.05)


def test_subdivided_planes_keep_their_area_and_dip():
    coarse = _bent()
    fine = coarse.subdivide(0.5)
    assert fine.areas_km2.sum() == pytest.approx(coarse.areas_km2.sum(), rel=1e-12)
    for plane, (start, stop) in enumerate(
        itertools.pairwise(np.cumsum([0, *fine.plane_cells]))
    ):
        np.testing.assert_allclose(
            fine.dip_deg[:, start:stop], coarse.dip_deg[0, plane], atol=1e-9
        )
    assert fine.plane_cells == (70, 67)


def test_the_dct_embedding_is_the_wrapped_fft():
    covariance = Covariance(
        PreCorrected(VonKarman(), Marginal("truncated_exponential", 0.9)), (5.0, 3.0)
    )
    grid = Grid((35, 137), (0.5, 0.5))
    amplitudes, padded = _embed(grid, covariance)
    lags = [
        np.arange(e // 2 + 1) * r / ln
        for e, r, ln in zip(padded, grid.resolution_km, covariance.lengths_km)
    ]
    quadrant = covariance.correlation(np.hypot(lags[0][:, None], lags[1][None, :]))

    def wrap(e):
        return np.minimum(np.arange(e), e - np.arange(e))

    spectrum = np.fft.fft2(quadrant[np.ix_(wrap(padded[0]), wrap(padded[1]))]).real
    np.testing.assert_allclose(
        amplitudes**2,
        np.maximum(spectrum[: padded[0] // 2 + 1, : padded[1] // 2 + 1], 0.0),
        atol=1e-12 * spectrum.max(),
    )


def _target(covariance, grid, scale=(1.0, 1.0)):
    lags = [
        np.arange(e) * r / (ln * k)
        for e, r, ln, k in zip(
            grid.shape, grid.resolution_km, covariance.lengths_km, scale
        )
    ]
    return covariance.correlation(np.hypot(lags[0][:, None], lags[1][None, :]))


@pytest.mark.parametrize(
    ("shape", "magnitude"),
    [((80, 150), 6.5), ((80, 150), 7.5), ((125, 300), 7.5)],
    ids=["8x15 km Mw6.5", "8x15 km Mw7.5", "12x30 km Mw7.5"],
)
def test_the_embedding_delivers_the_covariance_within_its_tolerances(shape, magnitude):
    # The covariance the draws have is exactly the inverse transform of the clipped
    # spectrum, read at a grid lag h through the wrap as min(h, P - h), with no draws.
    grid = Grid(shape, (0.1, 0.1))
    covariance = segment_source(magnitude, 90.0).covariance
    amplitudes, padded = _embed(grid, covariance)
    quadrant = sp.fft.idctn(amplitudes**2, type=1)
    wrapped = [
        np.minimum(np.arange(e), p - np.arange(e)) for e, p in zip(shape, padded)
    ]
    delivered = quadrant[np.ix_(*wrapped)]

    error = np.abs(delivered - _target(covariance, grid)).max()
    assert error <= WRAP_TOLERANCE + MAXIMUM_VARIANCE_DEFICIT

    # The lengths the delivered covariance corresponds to: what the tolerances are
    # really answerable to, since the regression that sets them scatters by far more.
    fit = sp.optimize.least_squares(
        lambda scale: (_target(covariance, grid, scale) - delivered)[::4, ::4].ravel(),
        [1.0, 1.0],
    )
    np.testing.assert_allclose(fit.x, 1.0, atol=1e-3)


def test_moment_magnitude_is_hanks_kanamori_equation_7():
    assert np.log10(moment_from_magnitude(6.0) * 1e7) == pytest.approx(
        1.5 * 6.0 + 16.05
    )


@pytest.mark.parametrize(
    "marginal",
    [
        Marginal("truncated_exponential", 0.9),
        Marginal("gamma", 0.75),
        Marginal("doubly_truncated_normal", 0.6, 4.3),
        Marginal("truncated_normal", 0.5, minimum=0.2),
        Marginal("gamma", 0.75, minimum=0.01),
    ],
)
def test_the_tabulated_transform_matches_the_quantile(marginal):
    from rupture_generator.sampling.norta import NORTA_TAIL, _distribution

    latent = np.random.default_rng(0).standard_normal(100_000)
    exact = _distribution(marginal).ppf(
        np.clip(sp.stats.norm.cdf(latent), NORTA_TAIL, 1 - NORTA_TAIL)
    )
    np.testing.assert_allclose(marginal.apply(latent), exact, rtol=2e-6)


@pytest.mark.parametrize("maximum", [1.5, 2.0, 4.3, 10.0])
@pytest.mark.parametrize("fraction", [0.05, 0.5, 0.99])
def test_a_doubly_truncated_normal_hits_its_mean_spread_and_maximum(maximum, fraction):
    from rupture_generator.sampling.norta import (
        _distribution,
        _doubly_truncated_normal_ceiling,
    )

    spread = fraction * _doubly_truncated_normal_ceiling(maximum)
    fitted = _distribution(Marginal("doubly_truncated_normal", spread, maximum))
    assert fitted.mean() == pytest.approx(1.0, abs=1e-9)
    assert fitted.std() == pytest.approx(spread, abs=1e-7)
    assert fitted.support() == pytest.approx((0.0, maximum), abs=1e-9)


def test_the_doubly_truncated_normal_approaches_the_truncated_exponential():
    from rupture_generator.sampling.norta import (
        _distribution,
        _doubly_truncated_normal_ceiling,
    )

    # 0.90 puts the truncated exponential's largest slip at about 4.3 mean slips.
    exponential = _distribution(Marginal("truncated_exponential", 0.9))
    maximum = float(exponential.support()[1])
    # The fit stops widening its parent short of the limit, so it falls just below.
    assert 0.0 < 0.9 - _doubly_truncated_normal_ceiling(maximum) < 5e-4


CUT_FAMILIES = [
    ("truncated_normal", 0.5),
    ("gamma", 0.75),
    ("gamma", 0.2),
    ("gamma", 2.0),
]


@pytest.mark.parametrize(("family", "spread"), CUT_FAMILIES)
@pytest.mark.parametrize("fraction", [1e-6, 0.01, 0.5, 0.95])
def test_a_cut_marginal_keeps_its_mean_and_spread_above_its_minimum(
    family, spread, fraction
):
    from rupture_generator.sampling import largest_minimum
    from rupture_generator.sampling.norta import NORTA_TAIL, _distribution

    minimum = fraction * largest_minimum(family, spread)
    marginal = Marginal(family, spread, minimum=minimum)
    fitted = _distribution(marginal)
    assert fitted.support()[0] == pytest.approx(minimum, rel=1e-9)
    assert fitted.mean() == pytest.approx(1.0, abs=1e-9)
    assert fitted.std() == pytest.approx(spread, rel=1e-7)
    # The moments by quadrature over the quantile function, apart from the fit's own.
    edges = (NORTA_TAIL, 1.0 - NORTA_TAIL)
    mean = sp.integrate.quad(fitted.ppf, *edges, limit=400, epsabs=1e-11)[0]
    square = sp.integrate.quad(
        lambda quantile: fitted.ppf(quantile) ** 2, *edges, limit=400, epsabs=1e-11
    )[0]
    assert mean == pytest.approx(1.0, abs=1e-5)
    assert np.sqrt(square - mean * mean) == pytest.approx(spread, rel=1e-4)
    assert marginal.power_mean(1.0) == pytest.approx(1.0, abs=1e-6)
    assert marginal.power_mean(2.0) == pytest.approx(1.0 + spread**2, rel=1e-3)
    # The transform doesn't give a value under the cut.
    drawn = marginal.apply(np.random.default_rng(1).standard_normal(200_000))
    assert drawn.min() >= minimum * (1.0 - 1e-12)
    assert drawn.mean() == pytest.approx(1.0, abs=0.01 * spread)


# A gamma of spread 2 has shape 1/4 and puts 0.4% of its values under 1e-9, so even
# that cut moves it measurably. The families here put almost none there.
@pytest.mark.parametrize(("family", "spread"), CUT_FAMILIES[:3])
def test_a_marginal_cut_at_almost_nothing_is_the_uncut_one(family, spread):
    uncut = Marginal(family, spread)
    assert Marginal(family, spread, minimum=0.0) == uncut
    latent = np.linspace(-5.0, 5.0, 1001)
    np.testing.assert_allclose(
        Marginal(family, spread, minimum=1e-9).apply(latent),
        uncut.apply(latent),
        rtol=1e-4,
        atol=1e-8,
    )


@pytest.mark.parametrize(("family", "spread"), CUT_FAMILIES)
def test_norta_corrects_for_the_cut_marginal(family, spread):
    from rupture_generator.sampling import largest_minimum, latent_correlation
    from rupture_generator.sampling.norta import (
        _correlation_series,
        _hermite_expansion,
    )

    marginal = Marginal(family, spread, minimum=0.5 * largest_minimum(family, spread))
    uncut = Marginal(family, spread)
    # The expansion is of the cut transform: unit mean, the stated spread.
    expansion = _hermite_expansion(marginal)
    assert expansion.coefficients[0] == pytest.approx(1.0, abs=1e-6)
    assert expansion.spread == pytest.approx(spread, rel=2e-3)
    # The cut changes the map from latent to marginal correlation, and the inverse
    # follows it.
    series = _correlation_series(marginal, marginal)
    assert not np.allclose(series, _correlation_series(uncut, uncut))
    target = np.array([0.1, 0.5, 0.9])
    latent = latent_correlation(marginal, marginal, target)
    delivered = np.polynomial.polynomial.polyval(latent, np.r_[0.0, series])
    np.testing.assert_allclose(delivered, target, atol=1e-8)
    lags = np.array([0.0, 1.0])
    corrected = PreCorrected(lambda lag: np.exp(-lag), marginal)
    np.testing.assert_allclose(
        corrected(lags), latent_correlation(marginal, marginal, np.exp(-lags))
    )


def test_a_normal_has_no_power_mean():
    from rupture_generator import RuptureGeneratorError

    with pytest.raises(RuptureGeneratorError, match="no power mean"):
        Marginal().power_mean(0.5)
