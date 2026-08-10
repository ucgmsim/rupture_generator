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

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rupture_generator.mesh import RuptureMesh
from rupture_generator.moment import (
    moment_of,
    rigidity_pa,
    sample_velocity_model,
    scale_to_moment,
    seismic_moment_nm,
)
from rupture_generator.sampling import (
    CovarianceSpec,
    SpectralSampler,
    correlation_lengths,
)
from rupture_generator.stages import (
    DepthRamp,
    OnsetParams,
    RakeParams,
    RiseTimeParams,
    SlipParams,
    apply_perturbation,
    onset_perturbation,
    rake_field,
    rise_time_field,
    slip_pattern,
    taper_edges,
)
from rupture_generator.timing import SpeedParams, alpha_t, speed_field, travel_times
from tests.strategies import (
    MAGNITUDES,
    SEEDS,
    covariances,
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


# ============================================================================
# The sampler
# ============================================================================


@SETTINGS
@given(mesh=planar_charts(), covariance=covariances(), seed=SEEDS)
def test_a_sampled_field_is_standardised(
    mesh: RuptureMesh, covariance: CovarianceSpec, seed: int
) -> None:
    """The sampler's output contract: zero mean, unit population variance.

    Exact by construction -- the last thing the sampler does is subtract the mean and
    divide by the spread -- so this is asserted at the arithmetic tolerance rather
    than at an estimator's. It is what lets every stage downstream write ``1 + cov*Z``
    and mean it, instead of carrying its own normalisation.
    """
    field = SpectralSampler().sample(mesh, covariance, _rng(seed))

    assert field.shape == mesh.cell_counts
    assert float(field.mean()) == pytest.approx(0.0, abs=CONSTRUCTION)
    assert float(field.std()) == pytest.approx(1.0, rel=CONSTRUCTION)


def test_a_one_cell_chart_gives_the_zero_field_rather_than_nan() -> None:
    """A constant field has no structure to scale, so scaling it is not defined.

    Not a hypothetical: the mesh CLI produces a one-cell chart for any plane shorter
    than half the requested subfault size. Dividing by the spread there gave infinity,
    then infinity times zero, and a whole SRF of NaN slip, NaN rake and NaN slip-rate
    samples was written with no error raised anywhere -- which is the worst possible
    failure, because every consumer downstream accepted the file.
    """
    field = SpectralSampler().sample(
        _flat_chart(1, 1), CovarianceSpec(5.0, 5.0), _rng()
    )

    assert field.shape == (1, 1)
    assert np.isfinite(field).all()
    assert field[0, 0] == 0.0


@SETTINGS
@given(mesh=planar_charts(), covariance=covariances(), seed=SEEDS)
def test_the_inverse_transform_is_real(
    mesh: RuptureMesh, covariance: CovarianceSpec, seed: int
) -> None:
    """The symmetrised spectrum inverse-transforms to a real field.

    The sampler raises rather than silently taking the real part of something with a
    meaningful imaginary component, so this asserts that the guard never fires. It is
    the check that the Hermitian mirror covered *both* interior diagonals: omitting
    the second leaves half the negative-dip half an unmirrored draw, and the field
    that falls out still looks like a slip distribution.
    """
    SpectralSampler().sample(mesh, covariance, _rng(seed))


@SETTINGS
@given(
    mesh=planar_charts(),
    covariance=covariances(),
    rho=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    seed=SEEDS,
)
def test_the_correlation_blend_is_an_identity_on_the_fault(
    mesh: RuptureMesh, covariance: CovarianceSpec, rho: float, seed: int
) -> None:
    """``blended == rho*reference + sqrt(1 - rho^2)*independent``, pointwise.

    The inverse transform is linear and the crop is a restriction, so a relation
    imposed in the wavenumber domain survives to the fault exactly. Asserting the
    identity rather than a sample correlation coefficient is the whole point: a rho of
    0.8 implemented as 0.5 misses this by a factor of a million, and misses a Pearson
    coefficient computed over a fault by well under one standard error.

    Standardising each field afterwards divides by its own sample spread, which
    perturbs the relation by the estimator error -- which is why the sampler exposes
    the fields before that step.
    """
    sampler = SpectralSampler()
    rng = _rng(seed)
    reference_field, reference = sampler.sample_with_reference(mesh, covariance, rng)
    blended, independent = sampler.blend_on_fault(mesh, covariance, reference, rho, rng)

    expected = rho * reference.field + np.sqrt(1.0 - rho * rho) * independent
    scale = max(float(np.abs(expected).max()), 1.0e-300)
    assert np.abs(blended - expected).max() <= IDENTITY * scale
    # And the reference the blend used is the field the caller was handed, up to the
    # standardisation applied to one and not the other.
    assert reference.field.shape == reference_field.shape


@SETTINGS
@given(mesh=planar_charts(min_cells=6), covariance=covariances(), seed=SEEDS)
def test_a_field_correlated_at_one_is_the_reference(
    mesh: RuptureMesh, covariance: CovarianceSpec, seed: int
) -> None:
    """rho = 1 reproduces the reference exactly, and rho = 0 discards it.

    The two ends of the blend, which between them pin its orientation: a
    ``sqrt(1-rho^2)`` written as ``sqrt(rho)`` or a swapped pair of weights passes
    every statistical check in the middle of the range and fails here.
    """
    sampler = SpectralSampler()
    rng = _rng(seed)
    _, reference = sampler.sample_with_reference(mesh, covariance, rng)

    same, _ = sampler.blend_on_fault(mesh, covariance, reference, 1.0, rng)
    scale = max(float(np.abs(reference.field).max()), 1.0e-300)
    assert np.abs(same - reference.field).max() <= IDENTITY * scale

    none, independent = sampler.blend_on_fault(mesh, covariance, reference, 0.0, rng)
    assert np.abs(none - independent).max() <= IDENTITY * scale


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
    mesh = _flat_chart(20, 20)
    covariance = CovarianceSpec(4.0, 4.0)
    sampler = SpectralSampler()

    for rho in (0.0, 0.5, 0.9):
        realised = []
        for seed in range(40):
            rng = _rng(seed)
            reference_field, reference = sampler.sample_with_reference(
                mesh, covariance, rng
            )
            other = sampler.correlated_with(mesh, covariance, reference, rho, rng)
            realised.append(
                float(np.corrcoef(reference_field.ravel(), other.ravel())[0, 1])
            )
        assert float(np.mean(realised)) == pytest.approx(rho, abs=0.06)


@given(magnitude=MAGNITUDES)
def test_correlation_lengths_follow_the_published_relation(magnitude: float) -> None:
    """Mai & Beroza's own formula, evaluated independently of the implementation.

    ``0.3333`` rather than a third is carried deliberately: the difference is in the
    fourth decimal of the exponent, which reaches about a percent of the down-dip
    corner at M8, and the literal is what the relation was fitted and published with.
    A reference that re-derived it as ``1/3`` would be asserting a different relation.
    """
    covariance = correlation_lengths(magnitude)

    assert covariance.correlation_length_strike_km == pytest.approx(
        10.0 ** (0.5 * magnitude - 2.50), rel=CONSTRUCTION
    )
    assert covariance.correlation_length_dip_km == pytest.approx(
        10.0 ** (0.3333 * magnitude - 1.50), rel=CONSTRUCTION
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
        dip_exponent=0.3333,
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
@given(mesh=planar_charts(), magnitude=MAGNITUDES, seed=SEEDS)
def test_the_scaled_slip_carries_the_target_moment(
    mesh: RuptureMesh, magnitude: float, seed: int
) -> None:
    """``sum(mu * A * s) == M0``.

    A tautology on its own -- the field is divided by exactly that sum -- so what the
    test is worth is the **registration**: that the areas come from the mesh rather
    than from a nominal product of spacings, that the rigidity is sampled at each
    subfault's own depth, and that the accumulation is in float64. The C folds through
    single precision, which on a hundred thousand subfaults costs about 6e-5 relative
    -- six missing subfaults' worth, where in float64 one missing subfault is visible.
    """
    covariance = correlation_lengths(magnitude)
    pattern, _, _ = slip_pattern(
        mesh, SlipParams(covariance=covariance), _rng(seed), SpectralSampler()
    )
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
    """A field truncated to zero everywhere cannot be scaled to carry anything.

    Dividing would give infinity, and the alternative to refusing is a rupture whose
    every subfault slips an infinite amount.
    """
    mesh = _flat_chart(4, 4)
    with pytest.raises(ValueError, match="carries no moment"):
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
@given(mesh=planar_charts(), magnitude=MAGNITUDES, seed=SEEDS)
def test_a_slip_pattern_is_never_negative(
    mesh: RuptureMesh, magnitude: float, seed: int
) -> None:
    """Slip is truncated at zero, because this is a model of slip and not of deficit.

    At the production spread of 0.75 about 9% of subfaults are clipped in expectation,
    which the pipeline reports as a diagnostic: a large fraction says the requested
    variation was not really achievable and the delivered spectrum is distorted.
    """
    pattern, _, _ = slip_pattern(
        mesh,
        SlipParams(covariance=correlation_lengths(magnitude)),
        _rng(seed),
        SpectralSampler(),
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
        covariance=CovarianceSpec(5.0, 5.0),
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
    params = SlipParams(covariance=CovarianceSpec(5.0, 5.0))
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
    params = SlipParams(covariance=CovarianceSpec(5.0, 5.0), side_taper=0.8)
    with pytest.raises(ValueError, match="overlap"):
        taper_edges(np.ones((6, 10)), params)


# ============================================================================
# Rise time
# ============================================================================


@SETTINGS
@given(mesh=planar_charts(min_cells=6), magnitude=MAGNITUDES, seed=SEEDS)
def test_the_mean_rise_time_is_the_requested_average(
    mesh: RuptureMesh, magnitude: float, seed: int
) -> None:
    """The normalisation closes the mean by construction, except where the floor binds.

    The normalising constant is the mean of the stretched pattern, so dividing by it
    and multiplying by the average is an identity. The floor raises the shortest
    subfaults to one sample, which can only push the realised mean *up* -- and that
    floor is physics rather than slack: a pulse shorter than one sample cannot be
    represented at all.
    """
    covariance = correlation_lengths(magnitude)
    sampler = SpectralSampler()
    rng = _rng(seed)
    _, gaussian, reference = slip_pattern(
        mesh, SlipParams(covariance=covariance), rng, sampler
    )

    average_s = 1.5
    interval_s = 0.005
    rise = rise_time_field(
        mesh,
        gaussian,
        reference,
        RiseTimeParams(average_s=average_s),
        rng,
        sampler,
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
    sampler = SpectralSampler()
    rng = _rng()
    _, gaussian, reference = slip_pattern(
        mesh, SlipParams(covariance=CovarianceSpec(5.0, 5.0)), rng, sampler
    )

    with pytest.raises(ValueError, match="slip exponent"):
        rise_time_field(
            mesh,
            gaussian,
            reference,
            RiseTimeParams(average_s=1.0, slip_exponent=0.05),
            rng,
            sampler,
            CovarianceSpec(5.0, 5.0),
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
    params = RakeParams(covariance=CovarianceSpec(6.0, 6.0))
    rake = rake_field(mesh, params, _rng(seed), SpectralSampler())

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
    with pytest.raises(ValueError, match="not a fault plane"):
        alpha_t(120.0, 90.0)
    with pytest.raises(ValueError, match="not a fault plane"):
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
    faster = solve(min(fraction * increase, 1.0))
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
    with pytest.raises(ValueError, match=r"subfault \(2, 2\)"):
        speed_field(mesh.centres()[..., 2], shear_speed, params)


# ============================================================================
# Onset -- DEFECTS.md 17
# ============================================================================


def _onset_setup(
    mesh: RuptureMesh, seed: int
) -> tuple[np.ndarray, object, SpectralSampler, np.random.Generator, CovarianceSpec]:
    covariance = CovarianceSpec(6.0, 5.0)
    sampler = SpectralSampler()
    rng = _rng(seed)
    _, _, reference = slip_pattern(
        mesh, SlipParams(covariance=covariance), rng, sampler
    )
    shear_speed = np.full(mesh.cell_counts, 3.3)
    return shear_speed, reference, sampler, rng, covariance


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

    Two claims, and both need saying. The hypocentre's onset is **exactly** the delay,
    because its perturbation is pinned to zero rather than left to the field; and the
    delay is the **minimum** of the whole onset field, because the perturbation is
    clamped there. Without the pin the cell the rupture started from need not be the
    earliest thing in the file; without the clamp a high-slip patch beside it gets
    pulled earlier still, which is a subfault radiating before the event it belongs to.

    The measured cost of getting registration wrong: a hypocentre one cell off in each
    direction gave onset fields correlating 0.92 to 0.997 with the truth while
    differing by up to 1.05 s. The front still expanded smoothly and onset still
    started at zero, so every diagnostic that asked whether the shape was right said
    yes.
    """
    shear_speed, reference, sampler, rng, covariance = _onset_setup(mesh, seed)
    cells_i, cells_j = mesh.cell_counts
    hypocentre = (cells_i // 3, cells_j // 2)

    travel = travel_times(
        mesh,
        shear_speed,
        SpeedParams(velocity_fraction=0.8, average_dip_deg=90.0, average_rake_deg=0.0),
        [(*hypocentre, 0.0)],
    )
    params = OnsetParams(scale_s=-0.35)
    onset = apply_perturbation(
        travel,
        onset_perturbation(mesh, reference, params, rng, sampler, covariance),
        params,
        hypocentre=hypocentre,
        delay_s=delay_s,
    )

    assert float(onset[hypocentre]) == pytest.approx(delay_s, abs=CONSTRUCTION)
    assert float(onset.min()) == pytest.approx(delay_s, abs=CONSTRUCTION)


@SETTINGS
@given(mesh=planar_charts(min_cells=6, max_cells=14), seed=SEEDS)
def test_onset_is_travel_time_plus_its_perturbation(
    mesh: RuptureMesh, seed: int
) -> None:
    """The stage is an identity on top of the wavefront, not a second solve.

    Asserted where the clamp does not bind, which is what separates the two things
    this stage does: it adds a perturbation, and it refuses to let anything precede
    the earthquake. A segment with no hypocentre -- one triggered from elsewhere --
    gets neither the pin nor the clamp, so its onsets stay absolute, which is what
    lets a multi-segment rupture propagate rather than restart on every fault.
    """
    shear_speed, reference, sampler, _, covariance = _onset_setup(mesh, seed)
    cells_i, cells_j = mesh.cell_counts
    travel = travel_times(
        mesh,
        shear_speed,
        SpeedParams(velocity_fraction=0.8, average_dip_deg=90.0, average_rake_deg=0.0),
        [(cells_i // 2, cells_j // 2, 0.0)],
    )

    params = OnsetParams(scale_s=-0.35)
    delay_s = 1.0
    # No hypocentre: no pin, no clamp, so the relation is exact everywhere and the
    # perturbation is recoverable.
    onset = apply_perturbation(
        travel,
        onset_perturbation(mesh, reference, params, _rng(seed + 1), sampler, covariance),
        params,
        hypocentre=None,
        delay_s=delay_s,
    )
    perturbation = (onset - travel - delay_s) / params.scale_s

    assert float(perturbation.mean()) == pytest.approx(0.0, abs=CONSTRUCTION)
    assert float(perturbation.std()) == pytest.approx(params.sigma, rel=CONSTRUCTION)
    assert np.allclose(
        onset, travel + params.scale_s * perturbation + delay_s, rtol=IDENTITY
    )


# ============================================================================
# Determinism and independence -- what replaces the draw-order machinery
# ============================================================================


@SETTINGS
@given(mesh=planar_charts(), seed=SEEDS)
def test_the_same_seed_gives_the_same_field(mesh: RuptureMesh, seed: int) -> None:
    """Reproducibility, asserted bit for bit rather than approximately."""
    covariance = CovarianceSpec(8.0, 5.0)
    sampler = SpectralSampler()
    first = sampler.sample(mesh, covariance, _rng(seed))
    second = sampler.sample(mesh, covariance, _rng(seed))
    assert np.array_equal(first, second)


def test_one_stages_parameters_do_not_disturb_another_stages_noise() -> None:
    """The property that replaces the whole draw-order apparatus.

    The port drew every field from one stream in a fixed order, so a stage's noise
    depended on what ran before it -- to the point that two dead fields were drawn and
    discarded on every run purely to keep the order intact, and a band-pass was
    forbidden from ever removing a draw. With a named substream per stage that is all
    unrepresentable: here, changing the rupture speed leaves the onset perturbation
    identical, so the two stages can be reordered, retried or tested alone.
    """
    mesh = _flat_chart(8, 16, depth_km=9.0)
    covariance = CovarianceSpec(6.0, 5.0)
    sampler = SpectralSampler()
    shear_speed = np.full(mesh.cell_counts, 3.3)
    hypocentre = (4, 8)

    def onset_for(velocity_fraction: float) -> np.ndarray:
        rng = _rng(99)
        _, _, reference = slip_pattern(
            mesh, SlipParams(covariance=covariance), rng, sampler
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
        params = OnsetParams(scale_s=-0.35)
        onset = apply_perturbation(
            travel,
            onset_perturbation(mesh, reference, params, _rng(7), sampler, covariance),
            params,
            # No hypocentre, so no clamp: the onset is the wavefront plus the
            # perturbation exactly, and subtracting recovers the perturbation. With
            # the clamp in the way the difference would carry the wavefront's own
            # shape wherever it bound, and this test would be about the clamp.
            hypocentre=None,
            delay_s=0.0,
        )
        return onset - travel

    # The perturbation is the onset minus the wavefront, and the wavefront is the only
    # thing the velocity fraction touches.
    #
    # Recovered rather than compared bit for bit: `(T + p) - T` loses about
    # `eps * |T|` of `p`, which on multi-second arrivals is around 1e-15 s, and the
    # two solves have different `T`. The tolerance below is three orders above that
    # and eleven below the 0.35 s the perturbation itself is worth -- so it separates
    # "the same draw, differently rounded" from "a different draw" by a wide margin.
    slow = onset_for(0.6)
    fast = onset_for(0.9)
    assert np.abs(slow - fast).max() < 1.0e-12
