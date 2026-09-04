"""Properties of the sampler, the moment closure, and the field stages.

Three kinds of tolerance appear here, and which one a test uses is a statement about
what kind of claim it is making.

**Identities** are asserted at ``1e-12`` relative or exactly. A blend written as
``rho*a + sqrt(1-rho^2)*b`` either is that expression or is not; there is no
estimator error in it, and asserting it as an identity rather than as a sample
correlation is what makes the assertion sharp. A rho of 0.8 implemented as 0.5 is a
factor of a million against the identity and well under one standard error against a
Pearson coefficient computed on a fault-sized sample.

**Constructions** are asserted at ``1e-9`` relative -- six orders above the f64
round-off floor at fault scale. A moment that closes by construction, a mean that a
normalisation was chosen to produce: these are exact in exact arithmetic, and the
tolerance is arithmetic slack only.

**Statistical** claims carry their estimator's error, derived at the assertion. A
field's realised spread is a sample statistic over roughly ``area / (lambda_s *
lambda_d)`` independent patches -- not over its subfault count, which is what makes
a fault-sized sample much smaller than it looks -- so the tolerance is written in
terms of that effective count and the test says so.

No test here pins a number the C produced.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rupture_generator.errors import ConfigError
from rupture_generator.mesh import RuptureMesh
from rupture_generator.moment import (
    moment_of,
    rigidity_pa,
    sample_velocity_model,
    scale_to_moment,
    seismic_moment_nm,
)
from rupture_generator.sampling import (
    CORRELATION_LENGTH_TOLERANCE,
    MINIMUM_EMBEDDING,
    NORMAL,
    SUZUKI_COEFFICIENTS,
    DegradedCorrelation,
    HybridFilterParameters,
    Marginal,
    VonKarmanFilterParameters,
    _crossing_km,
    _delivered_lengths,
    _distribution,
    _embed,
    _nonstationary_covariance,
    attainable_correlation,
    correlate_fields,
    correlation_lengths,
    latent_correlation,
    standardise,
    transformed_correlation,
    von_karman_correlation,
    von_karman_field,
)
from rupture_generator.stages import (
    CAUSAL_MARGIN,
    DepthRamp,
    OnsetParams,
    RakeParams,
    RiseTimeParams,
    SlipParams,
    onset_perturbation,
    onset_scale_s,
    rake_field,
    rise_time_field,
    slip_pattern,
    taper_edges,
    taper_onset,
)
from rupture_generator.timing import (
    MAXIMUM_VELOCITY_FRACTION,
    OFF_FAULT_SLOWNESS_FACTOR,
    RAYLEIGH_VELOCITY_FRACTION,
    SpeedParams,
    alpha_t,
    speed_field,
    travel_times,
)
from tests.strategies import (
    MAGNITUDES,
    SEEDS,
    charts_with_covariances,
    depth_ramps,
    planar_charts,
)

IDENTITY = 1.0e-12
"""For a relation that is an expression, not an estimate."""

CONSTRUCTION = 1.0e-9
"""For a quantity a construction makes exact, with room for f64 arithmetic only."""

SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
"""Twenty-five examples. Each one draws a whole field over a padded grid and
transforms it, so the cost per example is real; the shrinking is what earns its keep,
not the count."""


def _rng(seed: int = 1234) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed))


def _flat_chart(cells_i: int, cells_j: int, *, depth_km: float = 5.0) -> RuptureMesh:
    """A horizontal unit-spaced chart, for tests about arithmetic rather than geometry."""
    east, north = np.meshgrid(
        np.arange(cells_j + 1, dtype=float), np.arange(cells_i + 1, dtype=float)
    )
    return RuptureMesh.from_nodes(
        north,
        east,
        np.full_like(east, depth_km),
        origin_east_km=0.0,
        origin_north_km=0.0,
        surface="flat",
    )


def _sample(
    mesh: RuptureMesh,
    covariance: VonKarmanFilterParameters,
    rng: np.random.Generator,
) -> np.ndarray:
    """One standardised field on a chart."""
    return standardise(von_karman_field(mesh, covariance, rng))


# ============================================================================
# The sampler
# ============================================================================


@SETTINGS
@given(drawn=charts_with_covariances(), seed=SEEDS)
def test_a_sampled_field_is_standardised(
    drawn: tuple[RuptureMesh, VonKarmanFilterParameters], seed: int
) -> None:
    """The sampler's output contract: zero mean, unit population variance.

    Exact by construction -- `standardise` subtracts the mean and divides by the
    spread -- so this is asserted at the arithmetic tolerance rather than at an
    estimator's. It is what lets every stage downstream write ``1 + cov*Z`` and mean
    it, instead of carrying its own normalisation.
    """
    mesh, covariance = drawn
    field = _sample(mesh, covariance, _rng(seed))

    assert field.shape == mesh.cell_counts
    assert float(field.mean()) == pytest.approx(0.0, abs=CONSTRUCTION)
    assert float(field.std()) == pytest.approx(1.0, rel=CONSTRUCTION)


def test_a_one_cell_chart_gives_the_zero_field_rather_than_nan() -> None:
    """A constant field has no structure to scale, so scaling it is not defined.
    mesh, covariance = drawn

    Not a hypothetical: the mesh CLI produces a one-cell chart for any plane shorter
    than half the requested subfault size. Dividing by the spread there gave infinity,
    then infinity times zero, and a whole SRF of NaN slip, NaN rake and NaN slip-rate
    samples was written with no error raised anywhere -- which is the worst possible
    failure, because every consumer downstream accepted the file.
    """
    field = _sample(_flat_chart(1, 1), VonKarmanFilterParameters(0.3, 0.3), _rng())

    assert field.shape == (1, 1)
    assert np.isfinite(field).all()
    assert field[0, 0] == 0.0


@SETTINGS
@given(drawn=charts_with_covariances(), seed=SEEDS)
def test_a_realised_field_is_real_and_finite(
    drawn: tuple[RuptureMesh, VonKarmanFilterParameters], seed: int
) -> None:
    """The spectrum inverse-transforms to a real field of the chart's own shape.

    The real part of the inverse transform is one of the two independent fields the
    complex draw carries, so realness is structural rather than checked -- what is
    worth asserting is that nothing infinite survives, and that the crop takes the
    fault's own corner of the padded grid rather than some other rectangle. A chart of
    the wrong shape here is a field silently transposed or offset from the geometry it
    is meant to describe.
    """
    mesh, covariance = drawn
    field = von_karman_field(mesh, covariance, _rng(seed))

    assert field.shape == mesh.cell_counts
    assert not np.iscomplexobj(field)
    assert np.isfinite(field).all()


@SETTINGS
@given(drawn=charts_with_covariances())
def test_the_embedding_delivers_the_correlation_length_it_was_asked_for(
    drawn: tuple[RuptureMesh, VonKarmanFilterParameters],
) -> None:
    """The contract, in the units the model is parameterised in.

    Circulant embedding is exact where its eigenvalues are non-negative, and on a fine
    grid they are not quite -- a smooth covariance sampled far below its own
    correlation length folds slightly onto itself. Clipping the negatives is the
    standard remedy; what makes it a remedy rather than a fudge is that the cost is
    *measured*, in the quantity a seismologist would weigh.

    So this asserts what `_embed` promises: that the correlation length the field
    actually has is within `CORRELATION_LENGTH_TOLERANCE` of the one the magnitude
    implied. The old spectral sampler could make no such statement at any tolerance --
    at production padding its delivered correlation length was out by a factor of 2pi.
    """
    mesh, covariance = drawn
    embedding = _embed(mesh.cell_counts, mesh.spacing_km(), covariance)

    assert embedding.correlation_length_error <= CORRELATION_LENGTH_TOLERANCE
    assert embedding.eigenvalues.min() >= 0.0
    # And the covariance it delivers is a covariance: unit variance at zero lag.
    delivered = np.fft.ifft2(embedding.eigenvalues).real
    assert delivered[0, 0] == pytest.approx(1.0, abs=0.05)


@SETTINGS
@given(hurst=st.floats(min_value=0.2, max_value=0.9))
def test_a_correlation_length_is_where_the_field_has_forgotten_half_of_itself(
    hurst: float,
) -> None:
    """What ``a`` *means*, pinned against the paper rather than against this code.

    Mai & Beroza equation (1) makes the argument of the ACF a distance in correlation
    lengths, so ``C(1)`` is the correlation at a separation of exactly one ``a``. At
    the paper's ``H = 0.75`` that is 0.5005 -- so a correlation length is, to within a
    rounding, the separation at which the field has forgotten half of itself.

    This is the assertion that a wavenumber convention cannot slip past. The same model
    written as a power spectrum needs ``k`` declared angular or in cycles, and getting
    it wrong scales every correlation length by ``2*pi`` while leaving the fields
    entirely plausible. Stated as a covariance there is nothing to declare.
    """
    assert float(von_karman_correlation(np.array([0.0]), hurst)[0]) == pytest.approx(
        1.0, abs=IDENTITY
    )
    assert float(von_karman_correlation(np.array([1.0]), 0.75)[0]) == pytest.approx(
        0.5005, abs=1.0e-4
    )
    # Monotone in separation, whatever the roughness: structure only decorrelates.
    distances = np.linspace(0.0, 8.0, 64)
    assert np.all(np.diff(von_karman_correlation(distances, hurst)) <= 0.0)


@SETTINGS
@given(drawn=charts_with_covariances())
def test_the_embedding_pads_to_at_least_twice_the_fault(
    drawn: tuple[RuptureMesh, VonKarmanFilterParameters],
) -> None:
    """A Toeplitz matrix of n lags needs 2n-2 to embed, so a fraction will not do.

    The old rule padded by ten percent, which is a statement about the fault rather
    than about the covariance -- and the covariance is the thing that wraps.
    """
    mesh, covariance = drawn
    extents = _embed(mesh.cell_counts, mesh.spacing_km(), covariance).extents

    for padded, cells in zip(extents, mesh.cell_counts, strict=True):
        assert padded >= MINIMUM_EMBEDDING * cells


def test_a_covariance_too_large_for_its_fault_degrades_rather_than_refusing() -> None:
    """A generator has to generate, so an impossible covariance warns and carries on.

    The segment still appears in the file, with the largest slip patches its grid can
    carry and a `DegradedCorrelation` saying what it got instead of what was asked for.
    Refusing would be defensible for a library and useless for a production run: a
    twenty-fault scenario should not fail because one 3.6 km-wide segment carries a
    magnitude implying 4.4 km asperities.

    A caller who wants the refusal has it -- the warning is a category, so
    ``simplefilter("error", DegradedCorrelation)`` turns it into one.
    """
    mesh = _flat_chart(4, 4)
    huge = VonKarmanFilterParameters(1e3, 1e3)

    with pytest.warns(DegradedCorrelation, match="cannot carry correlation lengths"):
        embedding = _embed(mesh.cell_counts, mesh.spacing_km(), huge)

    # Still a covariance, and still the fault's own shape: degraded, not broken.
    assert embedding.eigenvalues.min() >= 0.0
    assert np.isfinite(embedding.eigenvalues).all()
    field = von_karman_field(mesh, huge, _rng())
    assert field.shape == mesh.cell_counts
    assert np.isfinite(field).all()


@pytest.mark.slow
def test_the_realised_correlation_follows_the_requested_one() -> None:
    """Over many draws, the sample correlation tracks rho.

    Statistical, and deliberately loose: a field's independent patches number about
    ``area / (lambda_s * lambda_d)``, which on this chart is roughly 25 -- so a single
    draw's correlation carries a standard error near ``1/sqrt(25) = 0.2``. Averaging
    over 40 draws brings that to about 0.03, and the assertion allows 0.06, two of
    those.

    The identity test above is what actually pins the blend. This one is here because
    an identity in the wavenumber domain says nothing on its own about whether the two
    fields a *caller* receives are correlated at all -- the standardisation, and the
    fact that each is cropped from a padded grid, sit between.
    """
    mesh = _flat_chart(24, 24)
    covariance = VonKarmanFilterParameters(4.0, 4.0)

    for rho in (0.0, 0.5, 0.9):
        realised = []
        for seed in range(40):
            rng = _rng(seed)
            reference = von_karman_field(mesh, covariance, rng)
            independent = von_karman_field(mesh, covariance, rng)
            first = standardise(reference)
            other = standardise(correlate_fields(reference, independent, rho))
            realised.append(float(np.corrcoef(first.ravel(), other.ravel())[0, 1]))
        assert float(np.mean(realised)) == pytest.approx(rho, abs=0.06)


@given(magnitude=MAGNITUDES)
def test_correlation_lengths_follow_the_published_relation(magnitude: float) -> None:
    """Mai & Beroza's own formula, evaluated independently of the implementation.

    The exponents are the paper's own fractions. Equation (5) reads ``log(a_z) ~
    -1.5 + (1/3) Mw``, so a third is what was published and ``0.3333`` would be a
    transcription of it -- a difference of 0.06% at M8, far inside the paper's own
    scatter, but there is no reason to carry an approximation of a number the source
    states exactly.
    """
    covariance = correlation_lengths(magnitude)

    assert covariance.correlation_length_strike_km == pytest.approx(
        10.0 ** (0.5 * magnitude - 2.50), rel=CONSTRUCTION
    )
    assert covariance.correlation_length_dip_km == pytest.approx(
        10.0 ** (magnitude / 3.0 - 1.50), rel=CONSTRUCTION
    )


@given(smaller=MAGNITUDES, larger=MAGNITUDES)
def test_a_bigger_earthquake_has_bigger_patches(smaller: float, larger: float) -> None:
    """Correlation lengths grow with magnitude, on both axes.

    The monotonicity is the physical content of the relation -- a larger event has
    larger asperities -- and it holds whatever the offsets are, so it survives someone
    reconfiguring them.
    """
    if smaller > larger:
        smaller, larger = larger, smaller

    small = correlation_lengths(smaller)
    big = correlation_lengths(larger)
    assert big.correlation_length_strike_km >= small.correlation_length_strike_km
    assert big.correlation_length_dip_km >= small.correlation_length_dip_km


@given(magnitude=MAGNITUDES)
def test_mai_is_a_value_of_the_four_coefficients(magnitude: float) -> None:
    """Stating Mai's own numbers as a custom relation reproduces Mai exactly.

    The property the ``custom`` option rests on: the defaults are a *value* of the
    parameterisation rather than a branch beside it. If they were not -- if the
    published relation reached the corner by a path the four coefficients cannot
    express -- then a config stating Somerville's coefficients would not be
    Somerville either, and the option would be advertising more than it does.

    Exact equality rather than a tolerance, because it is the same expression
    evaluated on the same numbers, not a re-derivation.
    """
    assert correlation_lengths(
        magnitude,
        strike_offset=2.50,
        dip_offset=1.50,
        strike_exponent=0.5,
        dip_exponent=1.0 / 3.0,
    ) == correlation_lengths(magnitude)


@given(
    magnitude=MAGNITUDES,
    strike_offset=st.floats(-1.0, 4.0),
    dip_offset=st.floats(-1.0, 4.0),
    strike_exponent=st.floats(0.0, 1.0),
    dip_exponent=st.floats(0.0, 1.0),
)
def test_a_corner_relation_is_a_line_in_log_length(
    magnitude: float,
    strike_offset: float,
    dip_offset: float,
    strike_exponent: float,
    dip_exponent: float,
) -> None:
    """Whatever the coefficients, ``log10 lambda == exponent * Mw - offset``.

    Asserted against the formula rather than against Mai's numbers, because this is
    the claim a config file makes when it states four coefficients of its own: the
    file's numbers are the ones the field is drawn with, on both axes, and the axes
    do not share a coefficient. Swapping the strike and dip coefficients is the
    failure this catches and the shipped relation cannot -- both of Mai's axes are
    positive and similar in size.
    """
    covariance = correlation_lengths(
        magnitude,
        strike_offset=strike_offset,
        dip_offset=dip_offset,
        strike_exponent=strike_exponent,
        dip_exponent=dip_exponent,
    )

    assert np.log10(covariance.correlation_length_strike_km) == pytest.approx(
        strike_exponent * magnitude - strike_offset, abs=CONSTRUCTION
    )
    assert np.log10(covariance.correlation_length_dip_km) == pytest.approx(
        dip_exponent * magnitude - dip_offset, abs=CONSTRUCTION
    )


# ============================================================================
# Moment
# ============================================================================


@given(magnitude=MAGNITUDES)
def test_the_moment_relation_is_hanks_and_kanamori_equation_seven(
    magnitude: float,
) -> None:
    """``log10 M0 [N m] == 1.5 Mw + 9.0499505``.

    One of the four wrong numbers. Equation 4 is a *different* relation with the
    constant 10.73 in its dyne-centimetre form, and production selects equation 7; a
    port that defaulted to eq. 4 read 1.109 times too much moment and mean slip
    against every config that leaves the default alone. The error is a clean
    multiplicative factor, so nothing about the *shape* of a rupture can see it --
    which is why the relation is asserted here directly rather than inferred from an
    output.
    """
    # From the published dyne-centimetre form, converted here rather than quoted: a
    # newton-metre is 1e7 dyne-centimetres, so the log constant is 7 smaller.
    expected = 1.5 * (magnitude + 10.699967) - 7.0
    assert np.log10(seismic_moment_nm(magnitude)) == pytest.approx(
        expected, rel=CONSTRUCTION
    )


def test_rigidity_of_crustal_rock_is_about_thirty_gigapascals() -> None:
    """A physical anchor, not a pinned number.

    3.2 km/s and 2.6 g/cm^3 is ordinary crust, and ``rho v_s^2`` for it is 27 GPa. The
    assertion is loose because what it is checking is the *unit conversion* -- SI
    pascals, not the dyne per square centimetre the C worked in -- and being out by
    the factor of ten that separates them is not a near miss.
    """
    value = float(rigidity_pa(np.array([3.2]), np.array([2.6]))[0])
    assert value == pytest.approx(2.66e10, rel=0.01)


@SETTINGS
@given(drawn=charts_with_covariances(), magnitude=MAGNITUDES, seed=SEEDS)
def test_the_scaled_slip_carries_the_target_moment(
    drawn: tuple[RuptureMesh, VonKarmanFilterParameters],
    magnitude: float,
    seed: int,
) -> None:
    """``sum(mu * A * s) == M0``.

    A tautology on its own -- the field is divided by exactly that sum -- so what the
    test is worth is the **registration**: that the areas come from the mesh rather
    than from a nominal product of spacings, that the rigidity is sampled at each
    subfault's own depth, and that the accumulation is in float64. The C folds through
    single precision, which on a hundred thousand subfaults costs about 6e-5 relative
    -- six missing subfaults' worth, where in float64 one missing subfault is visible.
    """
    mesh, covariance = drawn
    pattern, _ = slip_pattern(mesh, SlipParams(covariance=covariance), _rng(seed))
    depth_km = mesh.centres()[..., 2]
    _, rigidity = sample_velocity_model(
        depth_km,
        np.array([1.0, 5.0, 12.0, 1000.0]),
        np.array([1.8, 3.2, 3.5, 4.6]),
        np.array([2.1, 2.5, 2.7, 3.2]),
    )

    target = seismic_moment_nm(magnitude)
    (slip_m,) = scale_to_moment([pattern], [rigidity], [mesh.areas_km2()], target)

    assert moment_of(slip_m, rigidity, mesh.areas_km2()) == pytest.approx(
        target, rel=CONSTRUCTION
    )
    assert (slip_m >= 0.0).all()


def test_two_segments_are_scaled_by_one_shared_factor() -> None:
    """The joint scaling: segment moments sum to the target, individually hitting no
    target of their own.

    This is the whole content of "scale once, jointly". A per-segment scaling would
    make the total right and the *partition between faults* an artefact of how each
    pattern happened to normalise -- which is a statement about the sampler rather
    than about the earthquake. Here the partition is the patterns' own.
    """
    first = _flat_chart(6, 10)
    second = _flat_chart(6, 4)
    patterns = [np.full(first.cell_counts, 1.0), np.full(second.cell_counts, 3.0)]
    rigidities = [np.full(mesh.cell_counts, 3.0e10) for mesh in (first, second)]
    areas = [mesh.areas_km2() for mesh in (first, second)]

    target = seismic_moment_nm(7.0)
    slips = scale_to_moment(patterns, rigidities, areas, target)

    moments = [
        moment_of(slip, rigidity, area)
        for slip, rigidity, area in zip(slips, rigidities, areas, strict=True)
    ]
    assert sum(moments) == pytest.approx(target, rel=CONSTRUCTION)
    # The second segment's pattern is three times the first's on 40% of the area, so
    # it carries more moment -- and neither carries half.
    assert moments[1] > moments[0]
    assert moments[0] != pytest.approx(target / 2, rel=0.01)
    # One factor: the ratio of scaled to unscaled is the same on both.
    ratios = [
        float((slip / pattern).mean())
        for slip, pattern in zip(slips, patterns, strict=True)
    ]
    assert ratios[0] == pytest.approx(ratios[1], rel=CONSTRUCTION)


def test_a_pattern_that_carries_no_moment_is_refused() -> None:
    """A field that is zero everywhere cannot be scaled to carry anything.

    Dividing would give infinity, and the alternative to refusing is a rupture whose
    every subfault slips an infinite amount.
    """
    mesh = _flat_chart(4, 4)
    with pytest.raises(ConfigError, match="carries no moment"):
        scale_to_moment(
            [np.zeros(mesh.cell_counts)],
            [np.full(mesh.cell_counts, 3.0e10)],
            [mesh.areas_km2()],
            seismic_moment_nm(6.0),
        )


def test_a_depth_on_a_layer_boundary_takes_the_layer_above() -> None:
    """The boundary convention, and the clamp below the model.

    Both are choices rather than consequences. A fault whose top edge sits exactly on
    a boundary is ordinary, and the two readings give it different rock; a subfault
    below the deepest layer is a modelling error, and extrapolating a velocity model
    past its own bottom invents properties nobody supplied.
    """
    bottoms = np.array([1.0, 5.0, 12.0])
    speeds = np.array([1.8, 3.2, 3.5])
    densities = np.array([2.1, 2.5, 2.7])

    on_boundary, _ = sample_velocity_model(
        np.array([1.0, 5.0]), bottoms, speeds, densities
    )
    assert on_boundary.tolist() == [1.8, 3.2]

    below, _ = sample_velocity_model(np.array([999.0]), bottoms, speeds, densities)
    assert below.tolist() == [3.5]

    # Per subfault, not per row: a two-dimensional depth field gives a
    # two-dimensional answer, which is what a bent chart needs.
    grid, _ = sample_velocity_model(
        np.array([[0.5, 3.0], [8.0, 20.0]]), bottoms, speeds, densities
    )
    assert grid.shape == (2, 2)
    assert grid.tolist() == [[1.8, 3.2], [3.5, 3.5]]


# ============================================================================
# Slip and its taper
# ============================================================================


@SETTINGS
@given(drawn=charts_with_covariances(), magnitude=MAGNITUDES, seed=SEEDS)
def test_a_slip_pattern_is_never_negative(
    drawn: tuple[RuptureMesh, VonKarmanFilterParameters],
    magnitude: float,
    seed: int,
) -> None:
    """Slip is non-negative, because this is a model of slip and not of deficit.

    Not by clipping: the pattern's marginal is a truncated normal, a distribution
    whose support is the positive half-line and whose mean and coefficient of
    variation are the ones asked for. genslip reaches non-negativity by truncating
    ``1 + cov * Z``, which leaves a point mass at zero and neither of those two.
    """
    mesh, covariance = drawn
    pattern, _ = slip_pattern(
        mesh,
        SlipParams(covariance=covariance),
        _rng(seed),
    )
    assert (pattern >= 0.0).all()


@given(
    cells_i=st.integers(min_value=6, max_value=20),
    cells_j=st.integers(min_value=8, max_value=30),
    side=st.floats(min_value=0.0, max_value=0.4, allow_nan=False),
    top=st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
    bottom=st.floats(min_value=0.0, max_value=0.3, allow_nan=False),
)
def test_the_taper_is_the_product_of_two_one_dimensional_ramps(
    cells_i: int, cells_j: int, side: float, top: float, bottom: float
) -> None:
    """Separability, asserted by building the two ramps independently.

    This is the design decision the taper carries. genslip has two profiles that agree
    while the ramps stay apart and disagree once they overlap -- in its dip bands the
    right-hand ramp overwrites the left rather than compounding with it, so a cell two
    ramps reach is damped once in one place and twice in another. A taper is a
    statement about each edge separately, so the separable form is the one that means
    what the name says.
    """
    params = SlipParams(
        covariance=VonKarmanFilterParameters(1.5, 1.5),
        side_taper=side,
        top_taper=top,
        bottom_taper=bottom,
    )
    field = np.ones((cells_i, cells_j))
    tapered = taper_edges(field, params)

    # The one-dimensional ramps, from the definition rather than from the code.
    def width(fraction: float, extent: int) -> int:
        return max(0, int(fraction * extent + 0.5))

    along = np.ones(cells_j)
    for offset in range(width(side, cells_j)):
        ramp = (offset + 1) / width(side, cells_j)
        along[offset] *= ramp
        along[cells_j - 1 - offset] *= ramp
    down = np.ones(cells_i)
    for offset in range(width(top, cells_i)):
        down[offset] *= (offset + 1) / width(top, cells_i)
    for offset in range(width(bottom, cells_i)):
        down[cells_i - 1 - offset] *= (offset + 1) / width(bottom, cells_i)

    assert np.allclose(tapered, np.outer(down, along), rtol=CONSTRUCTION)


def test_slip_reaches_the_free_surface_at_full_amplitude() -> None:
    """``top_taper`` is zero in production, and that is deliberate.

    A surface-rupturing earthquake slips at the surface; tapering there would model
    every event as buried. The sides are tapered because the fault's along-strike ends
    are where it stops, which is a different statement about a different edge.
    """
    params = SlipParams(covariance=VonKarmanFilterParameters(1.5, 1.5))
    assert params.top_taper == 0.0

    # Long enough along strike that a 2% taper is a whole cell: the width rounds to
    # nearest, so on anything under about 25 cells the production taper is no taper at
    # all -- genslip's rule, faithfully, and worth knowing before reading a short
    # fault's edges as untapered by design.
    tapered = taper_edges(np.ones((8, 100)), params)

    assert np.all(tapered[0, :] == tapered[1, :])
    # The along-strike ends are damped, though.
    assert tapered[0, 0] < 1.0
    assert tapered[0, -1] < 1.0


def test_overlapping_tapers_are_refused() -> None:
    """Past a half of the fault a taper stops being a statement about its edges.

    Refusing makes the region where genslip's two profiles disagree unrepresentable,
    which is preferable to picking one of them silently.
    """
    params = SlipParams(covariance=VonKarmanFilterParameters(1.5, 1.5), side_taper=0.8)
    with pytest.raises(ConfigError, match="overlap"):
        taper_edges(np.ones((6, 10)), params)


# ============================================================================
# Rise time
# ============================================================================


@SETTINGS
@given(drawn=charts_with_covariances(min_cells=6), magnitude=MAGNITUDES, seed=SEEDS)
def test_the_mean_rise_time_is_the_requested_average(
    drawn: tuple[RuptureMesh, VonKarmanFilterParameters],
    magnitude: float,
    seed: int,
) -> None:
    """The normalisation closes the mean by construction, except where the floor binds.

    The normalising constant is the mean of the stretched pattern, so dividing by it
    and multiplying by the average is an identity. The floor raises the shortest
    subfaults to one sample, which can only push the realised mean *up* -- and that
    floor is physics rather than slack: a pulse shorter than one sample cannot be
    represented at all.
    """
    mesh, covariance = drawn
    rng = _rng(seed)
    _, slip_latent = slip_pattern(mesh, SlipParams(covariance=covariance), rng)

    average_s = 1.5
    interval_s = 0.005
    rise = rise_time_field(
        mesh,
        slip_latent,
        SlipParams(covariance=covariance).marginal,
        RiseTimeParams(average_s=average_s),
        rng,
        covariance,
        sample_interval_s=interval_s,
    )

    assert (rise >= interval_s - CONSTRUCTION).all()
    assert float(rise.mean()) >= average_s * (1.0 - CONSTRUCTION)
    # The floor can only lift a handful of subfaults on a well-conditioned field, so
    # the mean stays within a few percent of the target rather than drifting.
    assert float(rise.mean()) == pytest.approx(average_s, rel=0.15)


@given(ramp=depth_ramps(), factor=st.floats(min_value=1.0, max_value=4.0))
def test_the_depth_stretch_is_flat_between_its_ramps(
    ramp: DepthRamp, factor: float
) -> None:
    """``1 + excess`` outside each ramp and exactly 1 between them.

    The asymmetry is deliberate and is the original's: each branch measures from its
    own ramp's *far* end, which is what makes the value exactly one at both inner
    edges rather than nearly one. A version that measured from the near end would be
    continuous, plausible, and one at the wrong place.
    """
    deep = DepthRamp(ramp.centre_km + 20.0, ramp.half_width_km)
    params = RiseTimeParams(
        average_s=1.0,
        shallow_stretch=ramp,
        deep_stretch=deep,
        shallow_factor=factor,
        deep_factor=factor,
    )

    assert params.stretch_at(np.array([ramp.shallow_km - 1.0]))[0] == pytest.approx(
        factor, rel=CONSTRUCTION
    )
    assert params.stretch_at(np.array([deep.deep_km + 1.0]))[0] == pytest.approx(
        factor, rel=CONSTRUCTION
    )
    midpoint = 0.5 * (ramp.deep_km + deep.shallow_km)
    assert params.stretch_at(np.array([midpoint]))[0] == pytest.approx(
        1.0, rel=CONSTRUCTION
    )


def test_a_slip_exponent_below_the_floor_is_refused() -> None:
    """Below 0.1 genslip abandons the correlated field for independent lognormal noise.

    That is a different model, not a limiting case of this one, and it is not
    implemented -- so it is refused rather than silently skipped, which is what the
    original's early return amounts to.
    """
    mesh = _flat_chart(6, 6)
    rng = _rng()
    _, slip_latent = slip_pattern(
        mesh, SlipParams(covariance=VonKarmanFilterParameters(1.8, 1.8)), rng
    )

    with pytest.raises(ConfigError, match="slip exponent"):
        rise_time_field(
            mesh,
            slip_latent,
            SlipParams(covariance=VonKarmanFilterParameters(1.8, 1.8)).marginal,
            RiseTimeParams(average_s=1.0, slip_exponent=0.05),
            rng,
            VonKarmanFilterParameters(1.8, 1.8),
            sample_interval_s=0.005,
        )


# ============================================================================
# Rake
# ============================================================================


@SETTINGS
@given(mesh=planar_charts(min_cells=6), seed=SEEDS)
def test_the_rake_field_carries_degrees(mesh: RuptureMesh, seed: int) -> None:
    """Mean at the base rake, population spread at ``sigma_deg`` -- in **degrees**.

    `DEFECTS.md` 14: handing the slip field's dimensionless coefficient of variation
    (0.75) to the rake field in place of its sigma (15 degrees) gave every rake a
    spread of three quarters of a degree where the model wants fifteen -- a factor of
    twenty, on every fault, regardless of geometry. The two quantities are never bare
    numbers in the same expression, and their names carry the difference; this asserts
    the units actually arrive.
    """
    params = RakeParams(covariance=VonKarmanFilterParameters(1.8, 1.8))
    rake = rake_field(mesh, params, _rng(seed))

    assert float(rake.mean()) == pytest.approx(params.base_rake_deg, abs=CONSTRUCTION)
    assert float(rake.std()) == pytest.approx(params.sigma_deg, rel=CONSTRUCTION)


# ============================================================================
# The geometric correction, and the wrong number it was
# ============================================================================


@given(
    dip_deg=st.floats(min_value=0.0, max_value=90.0, allow_nan=False),
    rake_deg=st.floats(min_value=-180.0, max_value=180.0, allow_nan=False),
)
def test_the_geometric_correction_is_bounded(dip_deg: float, rake_deg: float) -> None:
    """``alpha_t`` lies in ``[1/1.1, 1]``, so it can move things by at most a tenth.

    The bound is what makes the correction a correction. Its coefficient is a literal
    for a reason: the sibling high-frequency port read the same constant from a deck
    whose "use the default" sentinel was -99, and when the deck reader was deleted the
    sentinel went through literally and gave every non-strike-slip fault a negative
    corner frequency.
    """
    value = alpha_t(dip_deg, rake_deg)
    assert 1.0 / 1.1 <= value <= 1.0


@given(rake_deg=st.floats(min_value=-180.0, max_value=0.0, allow_nan=False))
def test_the_correction_is_off_for_normal_faulting(rake_deg: float) -> None:
    """Negative rake means normal faulting, where the correction does not apply.

    Graves & Pitarka's adjustment is for reverse-slip geometries, where the free
    surface sits closer to the fault plane. It being exactly one elsewhere is what
    lets a strike-slip rupture be checked against the uncorrected model.
    """
    if rake_deg in (-180.0, 0.0):
        return
    assert alpha_t(60.0, rake_deg) == 1.0


def test_a_vertical_strike_slip_fault_is_the_calibration_point() -> None:
    """``alpha_t == 1`` exactly, so the whole correction is a no-op there."""
    assert alpha_t(90.0, 0.0) == 1.0
    assert alpha_t(90.0, 180.0) == 1.0
    assert alpha_t(90.0, 90.0) == 1.0


def test_a_dip_outside_the_plane_is_refused() -> None:
    """genslip answers a dip of 120 degrees with a factor of *zero*.

    That is a rupture with the correction silently switched off, and it is
    indistinguishable in the output from a vertical fault. Refusing names the mistake
    where it was made.
    """
    with pytest.raises(ConfigError, match="not a fault plane"):
        alpha_t(120.0, 90.0)
    with pytest.raises(ConfigError, match="not a fault plane"):
        alpha_t(-5.0, 90.0)


def test_the_correction_speeds_up_a_dipping_reverse_rupture() -> None:
    """The missing division, which was one of the four wrong numbers.

    A dip-45 pure-reverse source takes the correction at full strength, so its rupture
    speed is ``1/alpha_t`` times the raw fraction and every travel time is ``alpha_t``
    times the uncorrected one -- about 9% faster. The port did not divide at all, so
    every such rupture ran that much slow; and because the error is a smooth scaling of
    the whole onset field, nothing about the rupture's shape could see it.

    The vertical strike-slip case in the same test is the control: there the
    correction is exactly one, so the two solves agree bit for bit.
    """
    mesh = _flat_chart(10, 20, depth_km=8.0)
    shear_speed = np.full(mesh.cell_counts, 3.4)
    seeds = [(5, 10, 0.0)]

    reverse = SpeedParams(
        velocity_fraction=0.8, average_dip_deg=45.0, average_rake_deg=90.0
    )
    correction = alpha_t(45.0, 90.0)
    assert correction == pytest.approx(1.0 / 1.1, rel=CONSTRUCTION)

    corrected = travel_times(mesh, shear_speed, reverse, seeds)
    # The same fraction on a geometry the correction does not touch: a vertical
    # strike-slip fault, where alpha_t is exactly one.
    uncorrected = travel_times(
        mesh,
        shear_speed,
        SpeedParams(
            velocity_fraction=0.8,
            average_dip_deg=90.0,
            average_rake_deg=0.0,
        ),
        seeds,
    )
    assert np.allclose(corrected, uncorrected * correction, rtol=CONSTRUCTION)

    strike_slip = SpeedParams(
        velocity_fraction=0.8, average_dip_deg=90.0, average_rake_deg=0.0
    )
    plain = travel_times(mesh, shear_speed, strike_slip, seeds)
    assert np.array_equal(
        plain,
        travel_times(
            mesh,
            shear_speed,
            SpeedParams(
                velocity_fraction=0.8, average_dip_deg=90.0, average_rake_deg=180.0
            ),
            seeds,
        ),
    )


@given(ramp=depth_ramps(), factor=st.floats(min_value=0.2, max_value=1.0))
def test_the_speed_depth_factor_is_one_on_its_plateau(
    ramp: DepthRamp, factor: float
) -> None:
    """Reduced at both ends of the depth range and exactly 1 between.

    The plateau being *exactly* one is the assertion worth making: it is where the
    configured velocity fraction means what it says, and a version whose branches met
    at 0.999 would be a fault whose middle ruptured slightly slow for no stated reason.
    """
    deep = DepthRamp(ramp.centre_km + 20.0, ramp.half_width_km)
    params = SpeedParams(
        velocity_fraction=0.8,
        average_dip_deg=90.0,
        average_rake_deg=0.0,
        shallow=ramp,
        deep=deep,
        shallow_factor=factor,
        deep_factor=factor,
    )

    assert params.depth_factor(np.array([ramp.shallow_km - 1.0]))[0] == pytest.approx(
        factor, rel=CONSTRUCTION
    )
    assert params.depth_factor(np.array([deep.deep_km + 1.0]))[0] == pytest.approx(
        factor, rel=CONSTRUCTION
    )
    midpoint = 0.5 * (ramp.deep_km + deep.shallow_km)
    assert params.depth_factor(np.array([midpoint]))[0] == pytest.approx(
        1.0, rel=CONSTRUCTION
    )


@given(
    fraction=st.floats(min_value=0.4, max_value=0.9, allow_nan=False),
    increase=st.floats(min_value=1.05, max_value=2.0, allow_nan=False),
)
def test_a_faster_rupture_never_arrives_later(fraction: float, increase: float) -> None:
    """The wavefront is monotone in speed, everywhere at once.

    A property of the eikonal equation rather than of this stage, but asserted through
    the stage because that is where the speed field is assembled -- and an assembly
    that inverted the depth factor, or divided where it should multiply, would break
    the monotonicity while leaving every arrival finite and plausible.

    Capped at the **Rayleigh** fraction rather than at 1. A background above ``cR`` is
    not a configuration this model has: `speed_field` skips the mode-II forbidden zone
    by shifting everything above ``cR`` onto the supershear branch, so a background
    there would be silently converted to a supershear rupture. `SpeedParams` refuses it
    instead, which is why this sweep stops where it does.
    """
    mesh = _flat_chart(8, 16, depth_km=10.0)
    shear_speed = np.full(mesh.cell_counts, 3.3)
    seeds = [(4, 8, 0.0)]

    def solve(value: float) -> np.ndarray:
        return travel_times(
            mesh,
            shear_speed,
            SpeedParams(
                velocity_fraction=value, average_dip_deg=90.0, average_rake_deg=0.0
            ),
            seeds,
        )

    slower = solve(fraction)
    faster = solve(min(fraction * increase, RAYLEIGH_VELOCITY_FRACTION))
    assert (faster <= slower + CONSTRUCTION).all()


def test_a_speed_the_front_cannot_travel_at_is_refused_by_name() -> None:
    """Non-positive speed is refused in the stage's own vocabulary.

    The solver inverts the speed, so a zero would surface from inside a kernel as
    something about slowness. Naming the subfault, and saying which inputs to look at,
    is the difference between a diagnosis and a stack trace.
    """
    mesh = _flat_chart(4, 4, depth_km=5.0)
    shear_speed = np.full(mesh.cell_counts, 3.0)
    shear_speed[2, 2] = 0.0

    params = SpeedParams(
        velocity_fraction=0.8, average_dip_deg=90.0, average_rake_deg=0.0
    )
    with pytest.raises(ConfigError, match=r"subfault \(2, 2\)"):
        speed_field(mesh.centres()[..., 2], shear_speed, params)


# ============================================================================
# The onset displacement, its blend, and the registration it keeps -- DEFECTS.md 17
# ============================================================================


def _onset_setup(
    mesh: RuptureMesh, seed: int
) -> tuple[np.ndarray, object, np.random.Generator, VonKarmanFilterParameters]:
    """The slip draw the onset displacement correlates against, on any chart.

    The correlation lengths are a third of the chart's own extent rather than fixed
    kilometres -- Mai figure 13's median ratio -- so a generated chart of any size
    carries structure the model would actually put on it.
    """
    cells_i, cells_j = mesh.cell_counts
    strike_km, dip_km = mesh.spacing_km()
    covariance = VonKarmanFilterParameters(
        cells_j * strike_km / 3.0, cells_i * dip_km / 3.0
    )
    rng = _rng(seed)
    _, reference = slip_pattern(mesh, SlipParams(covariance=covariance), rng)
    shear_speed = np.full(mesh.cell_counts, 3.3)
    return shear_speed, reference, rng, covariance


def _blended_onset(
    mesh: RuptureMesh,
    seed: int,
    hypocentre: tuple[int, int],
    *,
    delay_s: float = 0.0,
    scale_s: float = 0.45,
    blend_sigma: float = 4.0,
) -> tuple[np.ndarray, SpeedParams, np.ndarray]:
    """The onset field this generator actually writes, and what shaped it.

    Both steps, in the order the pipeline runs them: the coherent solve over a speed
    field that is a function of depth alone, then the displacement blended in from the
    seed. Returns the onsets, the speed parameters, and the speed field the solve ran
    over.
    """
    shear_speed, reference, rng, covariance = _onset_setup(mesh, seed)
    params = OnsetParams(scale_s=scale_s, blend_sigma=blend_sigma)
    displacement = onset_perturbation(
        mesh,
        reference,
        SlipParams(covariance=covariance).marginal,
        params,
        rng,
        covariance,
    )
    speed = SpeedParams(
        velocity_fraction=0.8, average_dip_deg=90.0, average_rake_deg=0.0
    )
    travel = travel_times(mesh, shear_speed, speed, [(*hypocentre, 0.0)])
    onset = taper_onset(travel, displacement, params, seed_cell=hypocentre) + delay_s
    return onset, speed, speed_field(mesh.centres()[..., 2], shear_speed, speed)


@SETTINGS
@given(
    mesh=planar_charts(min_cells=6, max_cells=14),
    seed=SEEDS,
    delay_s=st.floats(min_value=0.0, max_value=3.0, allow_nan=False),
)
def test_the_hypocentre_onset_is_the_delay_and_the_earliest(
    mesh: RuptureMesh, seed: int, delay_s: float
) -> None:
    """The registration property. `DEFECTS.md` 17.

    Two claims: the hypocentre's onset is **exactly** the delay, and the delay is the
    **minimum** of the whole onset field. Both hold with the displacement *on* and with
    nothing pinned or shifted afterwards. The first is the blend -- its weight is the
    travel time over the blend length, exactly zero at the seed, so the displacement
    there is multiplied by zero whatever it drew. The second is the causal clamp, which
    holds every other cell's weight at or under ``tau / (c * dip)`` and so its onset
    strictly after the seed's.

    That is what the construction buys. Adding the displacement to the solved times
    unblended, as the port did, loses both: the cell the rupture started from was not
    the earliest thing in the file unless its displacement was pinned to zero by hand,
    and a high-slip patch beside it was pulled earlier still unless the field was
    clamped -- a subfault radiating before the event it belongs to. Both fixes were
    corrections to a field that had already been made wrong.

    The measured cost of getting registration wrong: a hypocentre one cell off in each
    direction gave onset fields correlating 0.92 to 0.997 with the truth while
    differing by up to 1.05 s. The front still expanded smoothly and onset still
    started at zero, so every diagnostic that asked whether the shape was right said
    yes.
    """
    cells_i, cells_j = mesh.cell_counts
    hypocentre = (cells_i // 3, cells_j // 2)
    onset, _, _ = _blended_onset(mesh, seed, hypocentre, delay_s=delay_s)

    assert float(onset[hypocentre]) == pytest.approx(delay_s, abs=CONSTRUCTION)
    assert float(onset.min()) == pytest.approx(delay_s, abs=CONSTRUCTION)


@SETTINGS
@given(mesh=planar_charts(min_cells=6, max_cells=14), seed=SEEDS)
def test_the_blended_onset_never_precedes_the_seed(
    mesh: RuptureMesh, seed: int
) -> None:
    """No subfault ruptures before the front that seeded it, at any blend width.

    This is what the per-cell clamp is for, and it is asserted across blend widths
    including ones far too narrow for the draw -- at ``blend_sigma = 0.25`` the blend
    term reaches 1 almost immediately and the clamp is doing all the work, which is
    precisely the case a fixed multiple of the scale cannot survive on its own.

    The claim is one-sided by construction. A first-arrival field is Lipschitz in the
    slowness, so the *coherent* front bounds its own neighbour differences; the
    displacement added on top of it has no such bound -- its gradient is the gradient
    of whatever field was drawn, divided by nothing -- and that is the deliberate trade
    this mechanism makes. What is not traded away is the ordering against the seed.
    """
    cells_i, cells_j = mesh.cell_counts
    hypocentre = (cells_i // 3, cells_j // 2)
    for blend_sigma in (0.25, 4.0, 16.0):
        onset, _, _ = _blended_onset(mesh, seed, hypocentre, blend_sigma=blend_sigma)
        assert float(onset.min()) >= -CONSTRUCTION
        assert np.unravel_index(int(np.argmin(onset)), onset.shape) == hypocentre


@SETTINGS
@given(mesh=planar_charts(min_cells=6, max_cells=14), seed=SEEDS)
def test_the_coherent_front_is_lipschitz_in_the_slowness(
    mesh: RuptureMesh, seed: int
) -> None:
    """The solve's own guarantee, on the field the blend starts from.

    ``T(b) <= T(a) + d * s_ab`` for adjacent cells, because the front reaching ``a``
    reaches ``b`` within ``d * s_ab``. So the difference between neighbours is bounded
    by the spacing times the largest slowness the solve ran over -- and that largest
    slowness is set by ``minimum_fraction``, which is the whole content of the causal
    band.

    Asserted globally and therefore exactly: any path's slowness is at most the chart's
    largest, so no tolerance is needed. The sharper local statement -- against the
    larger slowness of each *pair* -- holds to a measured 1.025 rather than exactly,
    because the fast sweeping update is a diagonal-aware stencil and reaches past the
    two cells being differenced; that 2.5% is the scheme's discretisation and not the
    model's, so the bound asserted here is the one that owes nothing to either.
    """
    cells_i, cells_j = mesh.cell_counts
    shear_speed, _, _, _ = _onset_setup(mesh, seed)
    speed_params = SpeedParams(
        velocity_fraction=0.8, average_dip_deg=90.0, average_rake_deg=0.0
    )
    travel = travel_times(
        mesh, shear_speed, speed_params, [(cells_i // 3, cells_j // 2, 0.0)]
    )
    speed = speed_field(mesh.centres()[..., 2], shear_speed, speed_params)
    strike_km, dip_km = mesh.spacing_km()

    # The slowness the solve actually ran over, wall included: an unoccupied cell is
    # slowed rather than removed, and a difference across one is still a difference.
    slowness = 1.0 / speed
    occupied = mesh.occupied()
    if not occupied.all():
        slowness = np.where(occupied, slowness, slowness * OFF_FAULT_SLOWNESS_FACTOR)
    slowest = float(slowness.max())

    for axis, spacing_km in ((0, dip_km), (1, strike_km)):
        gap_s = np.abs(np.diff(travel, axis=axis))
        assert np.all(gap_s <= spacing_km * slowest + CONSTRUCTION)


@SETTINGS
@given(mesh=planar_charts(min_cells=8, max_cells=14), seed=SEEDS)
def test_the_speed_field_stays_on_one_of_the_two_causal_branches(
    mesh: RuptureMesh, seed: int
) -> None:
    """The band is two branches with the forbidden zone between them, and stays so.

    Three claims. The realised speed never passes ``sqrt(2) Vs``, the Burridge-Andrews
    speed. It never lands in the mode-II forbidden zone, ``cR`` to ``Vs``, where
    in-plane rupture has no steady speed to hold -- genslip's ``[rvfmin, rvfmax]`` is a
    plain clip and allows every value between, the zone included. And it stays strictly
    positive, since the solver inverts it.

    Driven with depth factors above 1, which is the only way this model can reach
    either wall: the corrected background is held sub-Rayleigh when the parameters are
    validated, and the shipped depth profile only ever reduces it.
    """
    shear_speed, _, _, _ = _onset_setup(mesh, seed)
    speed = speed_field(
        mesh.centres()[..., 2],
        shear_speed,
        SpeedParams(
            velocity_fraction=0.8,
            average_dip_deg=90.0,
            average_rake_deg=0.0,
            shallow_factor=1.6,
            deep_factor=1.9,
        ),
    )
    fraction = speed / shear_speed

    assert np.all(fraction <= MAXIMUM_VELOCITY_FRACTION + CONSTRUCTION)
    assert np.all(speed > 0.0)

    inside = (fraction > RAYLEIGH_VELOCITY_FRACTION + CONSTRUCTION) & (
        fraction < 1.0 - CONSTRUCTION
    )
    assert not inside.any(), (
        f"{int(inside.sum())} subfaults sit between the Rayleigh speed and the shear "
        "speed, where an in-plane front has no steady speed; the gap is skipped by a "
        "monotone shift, not clipped into"
    )


@SETTINGS
@given(
    mesh=planar_charts(min_cells=8, max_cells=16),
    seed=SEEDS,
    scale_s=st.floats(min_value=0.05, max_value=0.8, allow_nan=False),
)
def test_the_blended_onset_keeps_the_seed_the_earliest(
    mesh: RuptureMesh, seed: int, scale_s: float
) -> None:
    """The whole point of the blend, over a range of displacement scales.

    Asserted against the *untapered* field, which fails, so the test is about the blend
    and not about the arithmetic.
    """
    shear_speed, reference, rng, covariance = _onset_setup(mesh, seed)
    cells_i, cells_j = mesh.cell_counts
    at = (cells_i // 3, cells_j // 2)
    params = OnsetParams(scale_s=scale_s)
    displacement = onset_perturbation(
        mesh,
        reference,
        SlipParams(covariance=covariance).marginal,
        params,
        rng,
        covariance,
    )
    travel = travel_times(
        mesh,
        shear_speed,
        SpeedParams(velocity_fraction=0.8, average_dip_deg=90.0, average_rake_deg=0.0),
        [(*at, 0.0)],
    )
    tapered = taper_onset(travel, displacement, params, seed_cell=at)

    assert float(tapered[at]) == pytest.approx(0.0, abs=CONSTRUCTION)
    assert float(tapered.min()) == pytest.approx(0.0, abs=CONSTRUCTION)
    assert np.unravel_index(int(np.argmin(tapered)), tapered.shape) == at


def _blend_weight(
    params: OnsetParams, *, seconds_per_cell: float = 0.5, seed: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A radial front and the weight :func:`taper_onset` realised on it.

    Returns ``(travel, centred displacement, weight, defined)``. The seed cell's weight
    is unrecoverable -- its displacement is zero there by construction, so the quotient
    is 0/0 -- and ``defined`` is the mask that excludes it.

    ``seconds_per_cell`` sets how long the rupture is against the blend zone, which is
    the whole of what decides how much of the displacement survives. The default puts
    the far corner at 18 s, a Mw 7.1 duration, where 4 sigma of a 0.35 s scale is 8% of
    the way across.
    """
    cells = (40, 60)
    at = (20, 30)
    grid = np.hypot(
        (np.arange(cells[0]) - at[0])[:, None], (np.arange(cells[1]) - at[1])[None, :]
    )
    travel = seconds_per_cell * grid
    displacement = params.scale_s * _rng(seed).standard_normal(cells)

    tapered = taper_onset(travel, displacement, params, seed_cell=at)
    centred = displacement - displacement[at]
    defined = centred != 0.0
    weight = np.ones_like(centred)
    weight[defined] = (tapered - travel)[defined] / centred[defined]
    return travel, centred, weight, defined


def _blend_and_clamp(
    params: OnsetParams, travel: np.ndarray, centred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The two terms of the weight, computed independently of the implementation."""
    ramp = np.minimum(1.0, travel / (params.blend_sigma * params.scale_s))
    dip = np.maximum(-centred, 0.0)
    with np.errstate(divide="ignore", over="ignore"):
        clamp = travel / (CAUSAL_MARGIN * np.maximum(dip, 1e-300))
    return ramp, clamp


def test_the_blend_is_four_sigma_wide_and_delivers_the_rest_whole() -> None:
    """What the shipped default does on a fault of the size it was chosen for.

    Inside ``blend_sigma * scale_s`` of travel time the weight is the linear ramp;
    outside it the displacement is delivered whole. On an 18 s rupture that zone is the
    first 1.4 s, so the realised spread comes out within a fraction of a per cent of the
    scale asked for -- which is the point of stating the width in sigmas rather than in
    seconds, and the reason 4 is a usable default rather than a fault-specific number.

    Both halves are asserted, because a blend that quietly delivered less than it
    claimed would still pass a causality test.
    """
    params = OnsetParams(scale_s=0.35, blend_sigma=4.0)
    travel, centred, weight, defined = _blend_weight(params)
    ramp, clamp = _blend_and_clamp(params, travel, centred)

    blend_s = params.blend_sigma * params.scale_s
    assert blend_s == pytest.approx(1.4)

    # Nothing on a fault this long is clamped: the clamp is for cells the front has
    # barely left, and the blend has already suppressed those.
    assert np.all(clamp[defined] >= ramp[defined])
    assert np.allclose(weight[defined], ramp[defined], atol=IDENTITY)

    # Out past the zone -- 99% of this fault -- the displacement is delivered whole.
    outside = defined & (travel >= blend_s)
    assert float(outside.mean()) > 0.95
    assert np.allclose(weight[outside], 1.0, atol=IDENTITY)

    realised = weight * centred
    assert float(realised.std()) == pytest.approx(float(centred.std()), rel=0.01)


def test_the_causal_clamp_binds_where_the_blend_alone_would_not() -> None:
    """The second term, on a front short enough for it to matter.

    A 1.3 s front against a 0.35 s scale is the regime a fixed blend width cannot
    survive on its own -- the front leaves the blend zone while the draw still has dips
    deeper than the travel time admits. There the clamp is the binding term, and it is
    per cell: two neighbours at the same travel time are held back differently because
    their own draws differ.

    Asserted as the identity ``w = min(ramp, clamp, 1)``, plus the consequence that
    makes it worth having.
    """
    params = OnsetParams(scale_s=0.35, blend_sigma=1.0)
    travel, centred, weight, defined = _blend_weight(params, seconds_per_cell=0.036)
    ramp, clamp = _blend_and_clamp(params, travel, centred)

    clamped = defined & (clamp < ramp)
    assert clamped.any(), "the fixture no longer exercises the clamp"
    assert float(clamped.mean()) < 0.1

    expected = np.minimum(ramp, clamp)
    assert np.allclose(weight[defined], expected[defined], atol=IDENTITY)
    # And the consequence: nothing precedes the seed, which sits at travel zero.
    assert float((travel + weight * centred).min()) >= 0.0


def test_the_blend_is_the_largest_admissible_ramp() -> None:
    """Linear in the travel time, and that is not an arbitrary choice.

    Keeping the onset after the seed is exactly ``w(tau) * (-delta) < tau``, so writing
    ``s = tau / (c e)`` the admissible weights are those with ``sup w(s)/s <= 1``. The
    linear ramp attains that bound, so nothing delivers more displacement at the same
    safety, and any sub-linear ramp breaks it near zero however wide the blend is made.

    Measured on the front where the clamp binds, so the bound comes out at
    ``1 / CAUSAL_MARGIN`` -- the margin that keeps the seed the strict minimum rather
    than a tie.
    """
    params = OnsetParams(scale_s=0.35, blend_sigma=1.0)
    travel, centred, weight, defined = _blend_weight(params, seconds_per_cell=0.036)

    away = defined & (travel > 0.0)
    ratio = (weight[away] * np.maximum(-centred, 0.0)[away] / travel[away]).max()
    assert ratio <= 1.0
    assert ratio == pytest.approx(1.0 / CAUSAL_MARGIN, rel=0.05)


def test_a_wider_blend_delivers_less_of_the_displacement() -> None:
    """The knob does what it says, monotonically.

    The blend is a suppression, so widening it can only remove displacement -- and once
    the zone is comparable to the rupture's own duration it removes most of it. That is
    the trade the parameter exposes: a smoother start, paid for in delivered spread.
    """
    spreads = []
    for blend_sigma in (1.0, 4.0, 16.0, 64.0):
        params = OnsetParams(scale_s=0.35, blend_sigma=blend_sigma)
        _, centred, weight, _ = _blend_weight(params)
        spreads.append(float((weight * centred).std()))

    assert spreads == sorted(spreads, reverse=True)
    # 64 sigma is 22.4 s against an 18 s rupture: the whole fault is inside the zone.
    assert spreads[-1] < 0.5 * spreads[0]


def test_the_onset_scale_follows_the_production_workflow_curve() -> None:
    """The amplitude relation, pinned against the numbers genslip is actually given.

    `workflow`'s `default_parameters/root/defaults.yaml` sets `tsfac_bzero = -0.1` and
    `tsfac_slope = -0.5` and leaves `tsfac_main` null, so genslip derives the amplitude
    per fault from the moment. None of the four version overlays override them. These
    are the seconds that produces, and they are what this package has to reproduce for
    a rupture generated here to carry the same timing heterogeneity as one generated
    through the production workflow.

    Signs: the workflow states both as negatives because genslip *subtracts* its
    `tsfac`. A spread has none, and `onset_scale_s` takes the magnitudes.
    """
    workflow = {
        5.00: 0.135,
        6.00: 0.212,
        6.45: 0.288,
        7.00: 0.454,
        7.10: 0.497,
        7.50: 0.729,
        8.00: 1.219,
        9.00: 3.640,
    }
    for magnitude, expected_s in workflow.items():
        scale_s = onset_scale_s(seismic_moment_nm(magnitude), 0.1, 0.5)
        assert scale_s == pytest.approx(expected_s, abs=5.0e-4)


def test_the_onset_scale_is_an_offset_plus_a_cube_root_of_moment() -> None:
    """Each term on its own, so a sign or unit error in either cannot hide.

    The offset is the whole of the spread at zero moment, and doubling the coefficient
    doubles only the part above it. Without the offset the relation is self-similar and
    the spread would fall to nothing at small magnitudes, where the recordings say it
    does not: at Mw 5 the offset is 74% of the total, and at Mw 9 it is 2.7%.
    """
    assert onset_scale_s(0.0, 0.1, 0.5) == pytest.approx(0.1)
    assert onset_scale_s(1.0e18, 0.0, 0.5) == pytest.approx(
        2.0 * onset_scale_s(1.0e18, 0.0, 0.25)
    )
    # Cube root, so a thousandfold moment is a tenfold spread above the offset.
    small = onset_scale_s(1.0e18, 0.0, 0.5)
    assert onset_scale_s(1.0e21, 0.0, 0.5) == pytest.approx(10.0 * small)
    # Both zero is a coherent front, exactly.
    assert onset_scale_s(1.0e21, 0.0, 0.0) == 0.0

    for magnitude, offset_share in ((5.0, 0.74), (9.0, 0.027)):
        total = onset_scale_s(seismic_moment_nm(magnitude), 0.1, 0.5)
        assert 0.1 / total == pytest.approx(offset_share, rel=0.05)


def test_the_coefficient_is_in_the_published_dyne_centimetre_units() -> None:
    """The unit conversion, asserted rather than trusted to a comment.

    ``coefficient`` is per cube-root dyne-centimetre at a scale of 1e-9 -- genslip's
    own convention, and the one :func:`average_rise_time_s` already takes. This package
    is otherwise SI, so the conversion is the thing most likely to be silently wrong,
    and it is a factor of ``(1e7)^(1/3) = 215.4`` rather than 1e7.
    """
    moment_nm = 1.0e19
    moment_dyne_cm = moment_nm * 1.0e7

    above_offset = onset_scale_s(moment_nm, 0.0, 0.5)
    assert above_offset == pytest.approx(0.5e-9 * moment_dyne_cm ** (1.0 / 3.0))
    # And that is 215.4x what treating newton-metres as dyne-centimetres would give.
    assert above_offset / (0.5e-9 * moment_nm ** (1.0 / 3.0)) == pytest.approx(
        215.443, rel=1.0e-4
    )


def test_a_blend_with_no_width_is_refused() -> None:
    """Zero sigma is not a coherent front; it is a front with no blend at all."""
    with pytest.raises(ConfigError, match="not a width"):
        OnsetParams(scale_s=0.35, blend_sigma=0.0)


def test_a_zero_scale_is_a_coherent_front_and_draws_nothing() -> None:
    """The off switch, and it has to be bit-exact.

    A displacement field that only *nearly* vanished would make a coherent run
    irreproducible, and the draw is skipped rather than scaled to zero, so turning the
    mechanism off costs no sample.
    """
    mesh = _flat_chart(16, 24)
    covariance = VonKarmanFilterParameters(4.0, 2.5)
    slip_params = SlipParams(covariance=covariance)
    _, reference = slip_pattern(mesh, slip_params, _rng(5))

    displacement = onset_perturbation(
        mesh,
        reference,
        slip_params.marginal,
        OnsetParams(scale_s=0.0),
        _rng(6),
        covariance,
    )

    assert np.array_equal(displacement, np.zeros(mesh.cell_counts))

    travel = np.arange(displacement.size, dtype=np.float64).reshape(displacement.shape)
    assert np.array_equal(
        taper_onset(travel, displacement, OnsetParams(scale_s=0.0), seed_cell=(0, 0)),
        travel,
    )


@SETTINGS
@given(drawn=charts_with_covariances(), seed=SEEDS)
def test_the_same_seed_gives_the_same_field(
    drawn: tuple[RuptureMesh, VonKarmanFilterParameters], seed: int
) -> None:
    """Reproducibility, asserted bit for bit rather than approximately."""
    mesh, covariance = drawn
    first = _sample(mesh, covariance, _rng(seed))
    second = _sample(mesh, covariance, _rng(seed))
    assert np.array_equal(first, second)


def test_one_stages_parameters_do_not_disturb_another_stages_noise() -> None:
    """The property that replaces the whole draw-order apparatus.

    The port drew every field from one stream in a fixed order, so a stage's noise
    depended on what ran before it -- to the point that two dead fields were drawn and
    discarded on every run purely to keep the order intact, and a band-pass was
    forbidden from ever removing a draw. With a named substream per stage that is all
    unrepresentable: changing the rupture speed leaves the onset displacement's draw
    identical, so the two can be reordered, retried or tested alone.

    Asserted bit for bit, which the previous model could not do. There, the
    displacement was recoverable only as ``onset - travel``, so the assertion had to
    carry the round-off of adding a sub-second field to a multi-second arrival and
    subtracting it back off. Now the draw is a separate array that never meets the
    solve, and the claim is exact.

    The second half is what keeps it from being vacuous: the same displacement, spent
    over two different speeds, must give *different* onsets. Otherwise a stage that
    silently ignored the field would pass the first assertion.
    """
    mesh = _flat_chart(8, 16, depth_km=9.0)
    covariance = VonKarmanFilterParameters(4.0, 2.5)
    shear_speed = np.full(mesh.cell_counts, 3.3)
    hypocentre = (4, 8)
    params = OnsetParams(scale_s=0.45)

    def draw_and_solve(velocity_fraction: float) -> tuple[np.ndarray, np.ndarray]:
        rng = _rng(99)
        slip_params = SlipParams(covariance=covariance)
        _, reference = slip_pattern(mesh, slip_params, rng)
        displacement = onset_perturbation(
            mesh, reference, slip_params.marginal, params, _rng(7), covariance
        )
        travel = travel_times(
            mesh,
            shear_speed,
            SpeedParams(
                velocity_fraction=velocity_fraction,
                average_dip_deg=90.0,
                average_rake_deg=0.0,
            ),
            [(*hypocentre, 0.0)],
        )
        onset = taper_onset(travel, displacement, params, seed_cell=hypocentre)
        return displacement, onset

    slow_field, slow_onset = draw_and_solve(0.6)
    fast_field, fast_onset = draw_and_solve(0.9)

    assert np.array_equal(slow_field, fast_field)
    assert not np.allclose(slow_onset, fast_onset)


MARGINALS = (
    NORMAL,
    Marginal("truncated_normal", 0.75),
    Marginal("truncated_exponential", 0.75),
    Marginal("gamma", 0.75),
)
"""One of every family a field may take, at the coefficient of variation this package
ships. The normal is in the list because the identity is a case the fit has to handle,
not an exemption from it."""


@pytest.mark.parametrize("marginal", MARGINALS)
def test_a_marginal_has_the_mean_and_spread_it_was_asked_for(
    marginal: Marginal,
) -> None:
    """What replaces genslip's clip, asserted on the distribution rather than a field.

    `1 + cov * Z` truncated at zero is a normal with a point mass at zero: neither its
    mean nor its spread is the one configured. A fitted marginal has both. Asserted
    through the frozen distribution's own moments, so this is about the fit and not
    about a sample.
    """
    if marginal.is_normal:
        # The identity: mean 0 and unit spread, and no fitting to check.
        assert marginal.apply(np.zeros(3)).tolist() == [0.0, 0.0, 0.0]
        return

    distribution = _distribution(marginal)
    mean, variance = distribution.mean(), distribution.var()
    assert float(mean) == pytest.approx(1.0, rel=1.0e-6)
    assert float(np.sqrt(variance) / mean) == pytest.approx(
        marginal.coefficient_of_variation, rel=1.0e-6
    )


@pytest.mark.parametrize("marginal", MARGINALS)
def test_the_correlation_series_sums_to_one(marginal: Marginal) -> None:
    """`g(1) = 1`: a field can be perfectly correlated with itself.

    The auto series' coefficients are the whole variance of the transformed field
    decomposed by Hermite order, so they sum to exactly 1 for a converged expansion.
    This is what `NORTA_ORDER`'s docstring claims at 20 terms, and it is the cheapest
    check that the expansion has converged at all -- a truncation that had not would
    leave the sum short.
    """
    assert float(attainable_correlation(marginal, marginal)) == pytest.approx(
        1.0, abs=1.0e-9
    )


@pytest.mark.parametrize("marginal", MARGINALS)
def test_the_pre_correction_inverts_the_correlation_map(marginal: Marginal) -> None:
    """`g(g_inv(rho)) == rho`, which is the entire content of the pre-correction.

    Round-tripped through the tabulated inverse, so this measures the tabulation as
    well as the series. `NORTA_INVERSE_POINTS`' docstring puts the round trip at 2e-10;
    the tolerance here is loose enough to be about the inversion rather than about the
    last bit of the interpolation.
    """
    targets = np.array([-0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 0.9])
    latent = latent_correlation(marginal, marginal, targets)
    delivered = transformed_correlation(marginal, marginal, latent)
    assert np.allclose(delivered, targets, atol=1.0e-8)


def test_the_pre_correction_asks_for_more_than_it_wants() -> None:
    """The direction of the correction, which a round trip alone would not catch.

    A monotone nonlinear map destroys correlation, so to *deliver* 0.8 the sampler has
    to be asked for more than 0.8 -- and under `NORMAL` for exactly 0.8, because there
    the map is the identity. Both halves matter: a pre-correction with the sign wrong
    still round-trips.
    """
    target = np.array(0.8)
    assert float(latent_correlation(NORMAL, NORMAL, target)) == pytest.approx(
        0.8, abs=1.0e-12
    )
    for marginal in (Marginal("truncated_normal", 0.75), Marginal("gamma", 0.75)):
        assert float(latent_correlation(marginal, marginal, target)) > 0.8


def test_a_correlation_two_marginals_cannot_share_is_refused() -> None:
    """The Gaussian-copula ceiling, and a refusal rather than the nearest answer.

    Two fields pushed through *different* monotone maps cannot be perfectly correlated
    even when their latents are, so `g_fg(1) < 1` and a target above it has no latent
    at all. Returning the closest one that exists would answer a question the caller
    did not ask, so `latent_correlation` raises.
    """
    slip = Marginal("truncated_normal", 0.75)
    ceiling = attainable_correlation(slip, NORMAL)
    assert ceiling < 1.0

    # Just inside is fine; just outside is refused.
    latent_correlation(slip, NORMAL, np.array(ceiling - 1.0e-6))
    with pytest.raises(ConfigError, match="under a Gaussian copula"):
        latent_correlation(slip, NORMAL, np.array(ceiling + 1.0e-3))


@pytest.mark.parametrize("marginal", MARGINALS)
def test_the_field_carries_the_correlation_length_not_the_latent(
    marginal: Marginal,
) -> None:
    """Where the fitted correlation length lands: on the field that gets written out.

    The embedding is of a *pre-corrected* covariance, so its latent has a different
    correlation length from the field the marginal produces. What the model is
    parameterised on is the field's, and `_attempt` measures the delivered lengths
    after the marginal for exactly that reason. genslip fits the latent's and writes
    out the other one.
    """
    cell_counts = (32, 64)
    spacing_km = (0.5, 0.5)
    covariance = VonKarmanFilterParameters(6.0, 4.0)
    embedding = _embed(cell_counts, spacing_km, covariance, marginal)

    delivered = np.fft.ifft2(embedding.eigenvalues).real
    lengths = _delivered_lengths(
        delivered, cell_counts, spacing_km, covariance, marginal
    )
    for got, wanted in zip(
        lengths,
        (
            covariance.correlation_length_strike_km,
            covariance.correlation_length_dip_km,
        ),
        strict=True,
    ):
        assert got == pytest.approx(wanted, rel=CORRELATION_LENGTH_TOLERANCE)


def test_a_slip_patterns_marginal_is_the_one_it_was_given() -> None:
    """The stage, not the sampler: what `slip_pattern` actually writes out.

    Statistical, and deliberately over a large chart -- a marginal is a statement about
    the population and a fault-sized sample of a correlated field carries far fewer
    independent values than it has cells. Asserted on the mean and spread, and on
    non-negativity, which is the property a truncated marginal exists to give.

    The largest slip is asserted too, because for the truncated exponential it is not a
    separate fact: the family is parameterised by where the cut falls, so the spread
    *is* the maximum, and 0.90 is 4.30 mean slips. That is the whole reason production
    draws from it -- Thingbaijam & Mai (2016) fit the ratio rather than the spread.
    """
    mesh = _flat_chart(64, 128)
    covariance = VonKarmanFilterParameters(6.0, 6.0)
    params = SlipParams(
        covariance=covariance, coefficient_of_variation=0.90, side_taper=0.0
    )
    assert params.marginal.family == "truncated_exponential"

    values = np.concatenate(
        [slip_pattern(mesh, params, _rng(seed))[0].ravel() for seed in range(6)]
    )
    assert float(values.min()) > 0.0
    assert float(values.mean()) == pytest.approx(1.0, abs=0.05)
    assert float(values.std() / values.mean()) == pytest.approx(0.90, abs=0.12)
    # The support's own end, not a quantile: no draw can exceed it. Against 4.30 rather
    # than 4.30 sample means, because the marginal is unit-mean by construction and the
    # bound is a property of the population.
    assert float(values.max()) < 4.30

    # The other family still works, and still says what it says.
    normal = dataclasses.replace(params, marginal_family="truncated_normal")
    other = np.concatenate(
        [slip_pattern(mesh, normal, _rng(seed))[0].ravel() for seed in range(6)]
    )
    assert float(other.min()) > 0.0
    assert float(other.std() / other.mean()) == pytest.approx(0.90, abs=0.12)
    # And has no such end: a normal's tail is unbounded wherever it is truncated below.
    assert float(other.max()) > 4.30


def test_the_shallow_blend_does_not_notch_the_variance() -> None:
    """The hybrid covariance is unit-variance *through* the transition, not just at it.

    Blending two correlation functions is not the same as blending two fields: a naive
    weighted sum of two normalised covariances dips below one at intermediate depths,
    which would show up as a band of suppressed slip across the transition. Asserted at
    every dip row rather than at the two ends, because the ends are where a notch is
    absent by construction.
    """
    hybrid = _hybrid((12.0, 5.0), (3.0, 1.5))
    downdip_km = np.linspace(0.0, 25.0, 40)
    strike_km, dip_km, hurst = hybrid.profile(downdip_km)

    # Zero strike lag, so this is the covariance of each row with every other; its
    # diagonal is each row's own variance.
    covariance = _nonstationary_covariance(
        np.zeros(1), downdip_km, strike_km, dip_km, hurst
    )
    variance = np.diag(covariance[0])

    assert np.allclose(variance, 1.0, atol=1.0e-9), (
        f"variance dips to {float(variance.min()):.6f} across the transition; a "
        "weighted sum of two normalised covariances would, and the averaged length "
        "tensor is what avoids it"
    )


# ============================================================================
# The hybrid profile: correlation lengths that change with depth
# ============================================================================


def _dipping_chart(
    cells_i: int, cells_j: int, *, dip_deg: float = 90.0, spacing_km: float = 1.0
) -> RuptureMesh:
    """A chart with a depth range, which is what a depth profile needs to bite on."""
    along = np.arange(cells_j + 1) * spacing_km
    down = np.arange(cells_i + 1) * spacing_km
    east = np.tile(down[:, None] * np.cos(np.radians(dip_deg)), (1, cells_j + 1))
    north = np.tile(along[None, :], (cells_i + 1, 1))
    depth = np.tile(down[:, None] * np.sin(np.radians(dip_deg)), (1, cells_j + 1))
    return RuptureMesh.from_nodes(
        east,
        north,
        depth,
        origin_east_km=0.0,
        origin_north_km=0.0,
        surface="dipping",
    )


def _hybrid(
    shallow: tuple[float, float], deep: tuple[float, float]
) -> HybridFilterParameters:
    return HybridFilterParameters(
        shallow=VonKarmanFilterParameters(*shallow),
        deep=VonKarmanFilterParameters(*deep),
        transition_depth_km=10.0,
        transition_half_width_km=3.0,
    )


def _strike_length(
    fields: np.ndarray, row: int, spacing_km: float, hurst: float
) -> float:
    """The along-strike correlation length an ensemble shows at one dip row."""
    band = fields[:, row, :]
    band = band - band.mean()
    cells = band.shape[1]
    acf = np.array(
        [float((band[:, : cells - k] * band[:, k:]).mean()) for k in range(cells // 2)]
    )
    half = float(von_karman_correlation(np.array([1.0]), hurst)[0])
    return _crossing_km(acf / acf[0], half, spacing_km)


def test_a_hybrid_profile_is_log_linear_between_its_ends() -> None:
    """A correlation length is a scale, so the transition interpolates in the log.

    Halfway between 2 km and 18 km is 6 km, not 10: the arithmetic midpoint would put
    most of the change in the shallow half of the ramp and almost none in the deep
    half, which is not a transition between two relations so much as a jump near one
    of them.
    """
    hybrid = _hybrid((18.0, 18.0), (2.0, 2.0))
    depth = np.array([0.0, 7.0, 10.0, 13.0, 30.0])
    strike, dip, hurst = hybrid.profile(depth)

    # Flat outside the ramp, and exactly the two ends there.
    assert strike[0] == pytest.approx(18.0, rel=IDENTITY)
    assert strike[-1] == pytest.approx(2.0, rel=IDENTITY)
    assert strike[1] == pytest.approx(18.0, rel=IDENTITY)
    assert strike[3] == pytest.approx(2.0, rel=IDENTITY)
    # The geometric mean at the centre.
    assert strike[2] == pytest.approx(np.sqrt(18.0 * 2.0), rel=IDENTITY)
    assert dip[2] == pytest.approx(np.sqrt(18.0 * 2.0), rel=IDENTITY)
    # One spectral falloff across the whole profile, for a pair that shares one.
    assert float(np.ptp(hurst)) == pytest.approx(0.0, abs=IDENTITY)


def test_a_hybrid_with_equal_ends_is_the_stationary_sampler_exactly() -> None:
    """The special case is an identity, not an approximation.

    A covariance that does not vary with depth is Toeplitz down dip; a Toeplitz
    operator embeds in a circulant one, and a circulant operator's eigenvectors are the
    DFT. So the general sampler's factorisation *is* the stationary sampler's
    transform, and dispatching to it costs nothing and changes nothing -- asserted
    bit-for-bit rather than statistically, because there is no estimator in the way.
    """
    mesh = _dipping_chart(16, 32)
    stationary = VonKarmanFilterParameters(4.0, 4.0)
    hybrid = _hybrid((4.0, 4.0), (4.0, 4.0))
    assert hybrid.is_stationary

    general = von_karman_field(mesh, hybrid, _rng(11))
    special = von_karman_field(mesh, stationary, _rng(11))

    assert np.array_equal(general, special)


def test_the_hybrid_covariance_has_unit_variance_at_every_depth() -> None:
    """What makes one marginal a statement about the whole fault.

    Paciorek & Schervish (2006) build the quadratic form from the *averaged* length
    tensor and divide by the determinant ratio, which is what holds the diagonal at 1
    while the lengths underneath it change. Without it the shallow and deep halves
    would carry different variances and the marginal transform would mean two
    different things on one fault.
    """
    hybrid = _hybrid((12.0, 5.0), (3.0, 1.5))
    depth = np.linspace(0.0, 24.0, 40)
    strike, dip, hurst = hybrid.profile(depth)
    downdip = np.linspace(0.0, 24.0, 40)

    covariance = _nonstationary_covariance(np.array([0.0]), downdip, strike, dip, hurst)

    assert np.diag(covariance[0]) == pytest.approx(1.0, abs=CONSTRUCTION)
    # Symmetric, which the factorisation assumes and does not check.
    assert covariance[0] == pytest.approx(covariance[0].T, abs=CONSTRUCTION)


@pytest.mark.parametrize("marginal", [NORMAL, Marginal("truncated_normal", 0.75)])
def test_the_hybrid_field_carries_its_profile_at_every_depth(
    marginal: Marginal,
) -> None:
    """The point of the whole sampler, measured where it is supposed to differ.

    Statistical: the correlation length is read off an ensemble ACF at a shallow row
    and a deep one, and both have to land on their own end of the profile rather than
    on some average of the two. Under a NORTA marginal the *field* has to carry them,
    which is what the pre-correction is for -- so this runs for both.
    """
    mesh = _dipping_chart(24, 96)
    hybrid = _hybrid((12.0, 5.0), (3.0, 1.5))
    depth = mesh.centres()[..., 2].mean(axis=1)
    strike, _, hurst = hybrid.profile(depth)

    rng = _rng(5)
    fields = marginal.apply(
        np.array([von_karman_field(mesh, hybrid, rng, marginal) for _ in range(300)])
    )

    for row in (0, mesh.cell_counts[0] - 1):
        delivered = _strike_length(fields, row, mesh.spacing_km()[0], float(hurst[row]))
        assert delivered == pytest.approx(float(strike[row]), rel=0.12)

    # And the two ends really are different, so the test above is not passing on a
    # field that happens to sit between them.
    assert strike[0] > 3.0 * strike[-1]


def test_a_hybrid_field_takes_the_marginal_it_was_given() -> None:
    """NORTA and the depth profile compose: the pre-correction is pointwise.

    It applies to a nonstationary covariance the same way it applies to a stationary
    one -- entry by entry, on correlations -- and `g_inv(1) = 1` keeps the unit
    diagonal that makes the marginal one statement about the whole fault.
    """
    mesh = _dipping_chart(24, 96)
    marginal = Marginal("truncated_normal", 0.75)
    rng = _rng(7)
    pooled = marginal.apply(
        np.array(
            [
                von_karman_field(mesh, _hybrid((12.0, 5.0), (3.0, 1.5)), rng, marginal)
                for _ in range(60)
            ]
        )
    )

    assert (pooled >= 0.0).all()
    assert float(pooled.mean()) == pytest.approx(1.0, abs=0.03)
    assert float(pooled.std()) == pytest.approx(0.75, rel=0.05)


def test_suzukis_down_dip_length_saturates_and_mais_does_not() -> None:
    """The shallow branch is a relation, not a multiple of the deep one.

    genslip pairs Suzuki et al. (2022) shallow with Mai & Beroza (2002) deep, and the
    two are not a factor apart on either axis: Suzuki's dip length stops growing above
    its corner magnitude while Mai's keeps going, so the ratio between the branches
    depends on the earthquake. Past about M8.6 the *shallow* branch is the shorter one
    down dip -- which is a thing to know before turning the hybrid on for a subduction
    interface, and is why this is asserted rather than assumed.
    """
    saturation = SUZUKI_COEFFICIENTS["dip_saturation_magnitude"]

    below = correlation_lengths(6.0, **SUZUKI_COEFFICIENTS)
    at = correlation_lengths(saturation, **SUZUKI_COEFFICIENTS)
    above = correlation_lengths(8.0, **SUZUKI_COEFFICIENTS)

    assert below.correlation_length_dip_km < at.correlation_length_dip_km
    assert above.correlation_length_dip_km == pytest.approx(
        at.correlation_length_dip_km, rel=IDENTITY
    )
    # The strike axis has no corner and keeps growing.
    assert above.correlation_length_strike_km > at.correlation_length_strike_km
    # Mai does not saturate on either axis.
    assert (
        correlation_lengths(8.0).correlation_length_dip_km
        > correlation_lengths(saturation).correlation_length_dip_km
    )
    # And the crossover is real: at M9 Suzuki is the shorter of the two down dip.
    assert (
        correlation_lengths(9.0, **SUZUKI_COEFFICIENTS).correlation_length_dip_km
        < correlation_lengths(9.0).correlation_length_dip_km
    )
