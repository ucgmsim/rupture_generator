import io
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

SECTION = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[172.0, -42.0], [172.3, -41.8], [172.7, -41.7]],
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


def test_subdivided_planes_keep_their_area_and_dip():
    coarse = geometry_from_geojson(
        io.StringIO(json.dumps(SECTION)), pyproj.CRS("EPSG:2193")
    )["bent"]
    fine = coarse.subdivide(0.5)
    width_km = 15.0 / np.sin(np.radians(60.0))
    assert fine.areas_km2.sum() == pytest.approx(
        coarse.strike_arc_km[-1] * width_km, rel=1e-12
    )
    np.testing.assert_allclose(fine.dip_deg, 60.0, atol=1e-9)
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
    "marginal", [Marginal("truncated_exponential", 0.9), Marginal("gamma", 0.75)]
)
def test_the_tabulated_transform_matches_the_quantile(marginal):
    from rupture_generator.sampling.norta import NORTA_TAIL, _distribution

    latent = np.random.default_rng(0).standard_normal(100_000)
    exact = _distribution(marginal).ppf(
        np.clip(sp.stats.norm.cdf(latent), NORTA_TAIL, 1 - NORTA_TAIL)
    )
    np.testing.assert_allclose(marginal.apply(latent), exact, rtol=2e-6)
