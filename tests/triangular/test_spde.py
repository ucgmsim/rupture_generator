"""The SPDE sampler, against analytic covariances and against the circulant sampler.

The four gates of `MESH.md`'s phase 0 are `test_planar_covariance_matches_analytic`
(gate 1), `test_matches_the_circulant_sampler` (gate 2),
`test_warped_patch_tracks_surface_separation` (gate 3) and `test_bound_is_a_bound`
(gate 4). The rest either check reductions to known answers or are the studies the
module's constants quote.

Covariances here come from `MaternOperator.covariance_column`, which is exact,
rather than from draws. A Monte Carlo estimate of a correlation carries an error
of order ``1/sqrt(draws)``, which at any affordable number of draws is larger than
the 1e-2 discretisation error being measured; `test_covariance_column_is_the_draws`
is what ties the exact route back to the field that actually comes out.
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from rupture_generator import sampling
from rupture_generator.sampling import (
    CORRELATION_LENGTH_TOLERANCE,
    MAXIMUM_DOUBLINGS,
    DegradedCorrelation,
    VonKarmanFilterParameters,
    correlation_lengths,
    von_karman_correlation,
)
from rupture_generator.triangular import spde

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

MAGNITUDE = 7.0
"""The magnitude the studies run at. Mai & Beroza's relations put it at 10 km along
strike and 6.81 km down dip, which is anisotropic enough that a bug swapping the two
axes cannot pass."""


DIP_DEG = 45.0
"""The dip the study patches are built on.

Any dip well clear of :data:`spde.DEGENERATE_DIP_SINE` would do; 45 degrees is
the one that makes the strike and dip directions equally far from the vertical,
so a test that confused them could not pass by symmetry.
"""


def monge_frame(dip_deg: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(e_u, e_v, n)`` for a plane dipping ``dip_deg``, depth positive down.

    ``e_u`` is horizontal, so it *is* the plane's strike; ``e_v`` is the steepest
    descent, so it *is* the dip direction. That identity is what
    `test_planar_frame_is_the_planes_strike_and_dip` checks the sampler
    reproduces from the face normal alone.
    """
    dip = np.radians(dip_deg)
    frame_u = np.array([1.0, 0.0, 0.0])
    frame_v = np.array([0.0, np.cos(dip), np.sin(dip)])
    return frame_u, frame_v, np.cross(frame_u, frame_v)


def grid_mesh(
    cells_u: int,
    cells_v: int,
    step_u: float,
    step_v: float,
    height: Callable[[FloatArray, FloatArray], FloatArray] | None = None,
    dip_deg: float = DIP_DEG,
) -> tuple[FloatArray, IntArray, FloatArray, tuple[int, int]]:
    """A regular triangulation of a rectangle on a **dipping** plane.

    The base plane dips at ``dip_deg`` so that the surface has a strike and a dip
    at all: the sampler takes its frame from the face normal, and a patch lying
    flat at constant depth is exactly the degeneracy
    :data:`spde.DEGENERATE_DIP_SINE` describes. Depth is the third component and
    positive down, matching the package.

    ``X = u e_u + v e_v + h(u, v) n``, with ``e_u`` horizontal (strike), ``e_v``
    down dip and ``n`` their normal -- the Monge patch of `MESH.md` Component 1.
    Each quad is cut on the same diagonal, which is the worst case for the
    cotangent Laplacian's isotropy and therefore the right one to test on.
    """
    frame_u, frame_v, normal = monge_frame(dip_deg)
    u = np.arange(cells_u + 1) * step_u
    v = np.arange(cells_v + 1) * step_v
    grid_u, grid_v = np.meshgrid(u, v, indexing="ij")
    parameters = np.stack([grid_u.ravel(), grid_v.ravel()], axis=-1)
    lift = np.zeros(grid_u.size) if height is None else height(grid_u, grid_v).ravel()
    vertices = (
        parameters[:, 0, None] * frame_u
        + parameters[:, 1, None] * frame_v
        + lift[:, None] * normal
    )

    index = np.arange((cells_u + 1) * (cells_v + 1)).reshape(cells_u + 1, cells_v + 1)
    corner = [index[:-1, :-1].ravel(), index[1:, :-1].ravel()]
    corner += [index[1:, 1:].ravel(), index[:-1, 1:].ravel()]
    faces = np.concatenate(
        [
            np.stack([corner[0], corner[1], corner[2]], axis=-1),
            np.stack([corner[0], corner[2], corner[3]], axis=-1),
        ]
    )
    return vertices, faces, parameters, (cells_u + 1, cells_v + 1)


def square_mesh(
    lengths: int,
    per_length: int,
    covariance: VonKarmanFilterParameters,
    height: Callable[[FloatArray, FloatArray], FloatArray] | None = None,
    dip_deg: float = DIP_DEG,
) -> tuple[FloatArray, IntArray, FloatArray, tuple[int, int]]:
    """A ``lengths`` x ``lengths`` correlation-length domain cut ``per_length`` a length."""
    return grid_mesh(
        lengths * per_length,
        lengths * per_length,
        covariance.correlation_length_strike_km / per_length,
        covariance.correlation_length_dip_km / per_length,
        height=height,
        dip_deg=dip_deg,
    )


def centre_profile(
    operator: spde.MaternOperator,
    shape: tuple[int, int],
    covariance: VonKarmanFilterParameters,
    radii: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """The exact correlation from the centre vertex, along each axis, at ``radii``.

    Returns the along-strike and down-dip profiles and the analytic values, all
    at the same dimensionless separations, so the comparison is like for like.
    """
    row, column = shape[0] // 2, shape[1] // 2
    column_values = operator.covariance_column(row * shape[1] + column).reshape(shape)
    variance = column_values[row, column]
    strike = column_values[row : row + radii.size, column] / variance
    dip = column_values[row, column : column + radii.size] / variance
    return strike, dip, von_karman_correlation(radii, covariance.hurst), variance


def quiet(function: Callable[..., Any], *args: Any, **keywords: Any) -> Any:
    """Run something that legitimately warns about a small domain."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DegradedCorrelation)
        return function(*args, **keywords)


# ---------------------------------------------------------------- the model


def test_matern_exponent_is_mais_hurst_on_a_surface() -> None:
    # Lindgren et al. (2011) equation (2): alpha = nu + d/2, and the von Karman
    # Hurst exponent is nu. Mai's 0.75 on a 2-manifold is the non-integer 1.75
    # that forces the rational approximation in the first place.
    assert spde.matern_exponent(sampling.HURST) == pytest.approx(1.75)
    assert spde.matern_exponent(sampling.HURST) / 2.0 == pytest.approx(0.875)


def test_von_karman_at_one_correlation_length() -> None:
    # The number the whole parameterisation means, and the anchor every profile
    # in this file is read against.
    assert von_karman_correlation(np.array([1.0]))[0] == pytest.approx(0.5005, abs=1e-4)


# --------------------------------------------------- the rational approximation


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_rational_roots_are_real_and_negative(order: int) -> None:
    # Stahl (2003) puts the poles and zeros of the best rational approximation of
    # x^s on the negative real axis. The solver depends on it: a negative root
    # makes the shifted matrix M + |r| K, which is symmetric positive definite.
    fit = spde.rational_approximation(0.875, order, spde._interval_floor(order))
    assert fit.numerator_roots.size == order
    assert fit.denominator_roots.size == order + 1
    assert (fit.numerator_roots < 0.0).all()
    assert (fit.denominator_roots < 0.0).all()


@pytest.mark.parametrize("order", [1, 2, 3])
def test_rational_approximation_reproduces_the_power(order: int) -> None:
    # The factored form the solver uses, against x^-beta on the interval it was
    # fitted for. Bolin & Kirchner equation (3.9).
    beta = 0.875
    floor = spde._interval_floor(order)
    fit = spde.rational_approximation(beta, order, floor)
    eigenvalue = np.exp(np.linspace(0.0, -np.log(floor), 4000))
    numerator = fit.numerator_leading * np.prod(
        [1.0 - root * eigenvalue for root in fit.numerator_roots], axis=0
    )
    denominator = fit.denominator_leading * np.prod(
        [1.0 - root * eigenvalue for root in fit.denominator_roots], axis=0
    )
    assert np.abs(numerator / denominator - eigenvalue**-beta).max() == pytest.approx(
        fit.supremum_error, rel=1e-6
    )
    # Stahl's rate, which is what theorem 3.3's rational term is bounded by.
    assert fit.supremum_error < np.exp(-2.0 * np.pi * np.sqrt(0.125 * order))


def test_rational_order_is_the_cheapest_that_converges() -> None:
    # RATIONAL_ORDER's justification. At h = 0.177 the m = 1 fit's error is what
    # limits the answer and m = 3 buys nothing over m = 2.
    covariance = correlation_lengths(MAGNITUDE)
    vertices, faces, parameters, shape = square_mesh(16, 8, covariance)
    radii = np.arange(3 * 8 + 1) / 8
    errors = {}
    for order in (1, 2, 3):
        operator = spde.MaternOperator(
            vertices, faces, parameters, covariance, order=order
        )
        strike, dip, analytic, _ = centre_profile(operator, shape, covariance, radii)
        errors[order] = max(
            float(np.abs(strike - analytic).max()), float(np.abs(dip - analytic).max())
        )
    assert errors[1] == pytest.approx(1.77e-2, abs=2e-3)
    assert errors[2] == pytest.approx(1.08e-2, abs=2e-3)
    assert errors[3] == pytest.approx(1.13e-2, abs=2e-3)
    # m = 1 is materially worse; m = 3 is not materially better.
    assert errors[1] > 1.4 * errors[2]
    assert errors[3] > 0.9 * errors[2]
    assert spde.RATIONAL_ORDER == 2


def test_lawson_approaches_the_minimax_error_from_below() -> None:
    # LAWSON_ITERATIONS' justification, and the reason it is not smaller: the fit
    # starts at least squares and climbs towards minimax, so stopping early makes
    # the reported supremum error -- and hence the model error -- optimistic.
    floor = spde._interval_floor(2)
    original = spde.LAWSON_ITERATIONS
    try:
        errors = {}
        for iterations in (16, 32, 64, 128, 256):
            spde.LAWSON_ITERATIONS = iterations
            spde.rational_approximation.cache_clear()
            errors[iterations] = spde.rational_approximation(
                0.875, 2, floor
            ).supremum_error
    finally:
        spde.LAWSON_ITERATIONS = original
        spde.rational_approximation.cache_clear()

    assert sorted(errors.values()) == list(errors.values())
    reference = errors[256]
    assert abs(errors[128] - reference) / reference < 5e-3
    assert abs(errors[16] - reference) / reference > 5e-2
    assert spde.LAWSON_ITERATIONS == 128


# ------------------------------------------------------------- the assembly


def test_isotropic_stiffness_is_the_cotangent_laplacian() -> None:
    # The reduction that says the assembly is the Laplace-Beltrami operator:
    # with one correlation length on both axes, Lindgren et al. (2011) appendix
    # A.2's G_ij(T) = e_i . e_j / (4|T|), scaled by a^2. The triangle is a
    # general one in three dimensions, so the check runs through the lifted path.
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.4], [0.2, 1.3, -0.3]])
    faces = np.array([[0, 1, 2]])
    length = 2.0
    covariance = VonKarmanFilterParameters(length, length)

    lumped, stiffness, _ = spde._assemble(vertices, faces, covariance)
    edge = np.stack(
        [
            vertices[2] - vertices[1],
            vertices[0] - vertices[2],
            vertices[1] - vertices[0],
        ]
    )
    area = 0.5 * np.linalg.norm(
        np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    )
    expected = length**2 * (edge @ edge.T) / (4.0 * area)
    assert stiffness.toarray() == pytest.approx(expected)
    # Mass lumping: |T|/3 at each corner (appendix A.2's C-tilde).
    assert lumped == pytest.approx(np.full(3, area / 3.0))


def test_stiffness_annihilates_a_constant() -> None:
    # A Laplacian has the constants in its kernel however it is assembled, on any
    # geometry. The one identity that holds for every anisotropy and every lift.
    covariance = correlation_lengths(MAGNITUDE)
    vertices, faces, _, _ = square_mesh(
        2, 4, covariance, height=lambda u, v: 0.2 * u + 0.1 * v**2 / 40.0
    )
    _, stiffness, _ = spde._assemble(vertices, faces, covariance)
    assert np.abs(stiffness @ np.ones(vertices.shape[0])).max() < 1e-9


def test_planar_frame_is_the_planes_strike_and_dip() -> None:
    # The planar-collapse identity `MESH.md` asks be kept: on a planar fault the
    # frame taken from the face normal must reproduce the plane's own strike and
    # dip exactly. Here that is the Monge frame (e_u, e_v) the mesh was built on,
    # recovered without the sampler ever seeing a parameter coordinate.
    covariance = correlation_lengths(MAGNITUDE)
    for dip_deg in (10.0, 45.0, 89.0):
        frame_u, frame_v, _ = monge_frame(dip_deg)
        vertices, faces, _, _ = square_mesh(1, 2, covariance, dip_deg=dip_deg)
        _, _, strike, dip, sine_dip = spde._surface_frames(vertices, faces)
        # Up to sign, which H cannot see: it uses only the outer products.
        assert np.abs(strike @ frame_u) == pytest.approx(np.ones(strike.shape[0]))
        assert np.abs(dip @ frame_v) == pytest.approx(np.ones(dip.shape[0]))
        assert sine_dip == pytest.approx(
            np.full(sine_dip.shape, np.sin(np.radians(dip_deg)))
        )


def test_frame_follows_the_surface_not_the_parameterisation() -> None:
    # Tilt the patch about the dip axis. The surface's strike stays horizontal --
    # it is *defined* horizontal -- but rotates within the horizontal plane, and
    # the dip direction stays perpendicular to it. A frame taken from dX/du would
    # instead have left the horizontal with the tilt.
    covariance = correlation_lengths(MAGNITUDE)
    slope = 0.5
    vertices, faces, _, _ = square_mesh(1, 2, covariance, height=lambda u, v: slope * u)
    _, _, strike, dip, _ = spde._surface_frames(vertices, faces)
    assert np.abs(strike[:, 2]).max() < 1e-12, "strike must stay horizontal"
    assert np.abs((strike * dip).sum(axis=-1)).max() < 1e-12
    # The rotation is exactly atan(slope / sin(base dip)) within the horizontal.
    frame_u, _, _ = monge_frame(DIP_DEG)
    turn = np.degrees(np.arccos(np.abs(strike @ frame_u)))
    expected = np.degrees(np.arctan(slope / np.sin(np.radians(DIP_DEG))))
    assert turn == pytest.approx(np.full(turn.shape, expected))
    # dX/du, the frame the first draft used, does leave the horizontal here.
    assert abs((frame_u + slope * monge_frame(DIP_DEG)[2])[2]) > 0.3


def test_anisotropy_fades_where_the_surface_flattens() -> None:
    # Where a patch is too flat to have a strike, the anisotropy is faded to
    # isotropic with the geometric mean -- and the geometric mean is what keeps
    # det H, hence the marginal variance, unmoved through the transition.
    covariance = correlation_lengths(MAGNITUDE)
    strike_length = covariance.correlation_length_strike_km
    dip_length = covariance.correlation_length_dip_km

    horizontal = np.array([0.0])
    lambda_s, lambda_d = spde._anisotropy(horizontal, covariance)
    assert lambda_s == pytest.approx(lambda_d)
    assert lambda_s[0] == pytest.approx(strike_length * dip_length)

    steep = np.array([1.0])
    lambda_s, lambda_d = spde._anisotropy(steep, covariance)
    assert lambda_s[0] == pytest.approx(strike_length**2)
    assert lambda_d[0] == pytest.approx(dip_length**2)

    # det H is preserved at every point of the fade, which is what stops the
    # degeneracy drawing a variance step across the shallow contour.
    sweep = np.linspace(0.0, 1.0, 41)
    lambda_s, lambda_d = spde._anisotropy(sweep, covariance)
    assert lambda_s * lambda_d == pytest.approx(
        np.full(sweep.shape, (strike_length * dip_length) ** 2)
    )
    # Monotone, and complete by the threshold.
    assert np.all(np.diff(lambda_s) >= -1e-12)
    assert lambda_s[sweep >= spde.DEGENERATE_DIP_SINE][0] == pytest.approx(
        strike_length**2
    )


def test_degenerate_dip_threshold_is_derived() -> None:
    # One degree of angular error swinging the frame by a right angle's worth.
    assert spde.DEGENERATE_DIP_SINE == pytest.approx(1.0 / 45.0)
    assert np.degrees(np.arcsin(spde.DEGENERATE_DIP_SINE)) == pytest.approx(
        1.273, abs=1e-3
    )


def test_a_horizontal_patch_is_isotropic_and_finite() -> None:
    # The degeneracy exercised end to end: a patch at constant depth has no
    # strike at all, and must still produce a field rather than a NaN.
    covariance = correlation_lengths(MAGNITUDE)
    vertices, faces, _, _ = square_mesh(8, 4, covariance, dip_deg=0.0)
    _, _, _, _, sine_dip = spde._surface_frames(vertices, faces)
    assert sine_dip.max() == 0.0
    operator = quiet(spde.MaternOperator, vertices, faces, None, covariance)
    field = operator.draw(np.random.default_rng(1))
    assert np.isfinite(field).all()
    # Isotropic, with the geometric mean: the mesh width uses the same H, so it
    # comes out at the geometric-mean length rather than either axis's.
    assert operator.error.mesh_width > 0.0


def test_refuses_geometry_it_cannot_read() -> None:
    covariance = correlation_lengths(MAGNITUDE)
    vertices, faces, parameters, _ = square_mesh(1, 2, covariance)

    with pytest.raises(ValueError, match="lifted positions"):
        spde.MaternOperator(vertices[:, :2], faces, parameters, covariance)
    with pytest.raises(ValueError, match="one .u, v. pair per vertex"):
        spde.MaternOperator(vertices, faces, parameters[:-1], covariance)
    with pytest.raises(ValueError, match="triangles"):
        spde.MaternOperator(vertices, faces[:, :2], parameters, covariance)
    with pytest.raises(ValueError, match="indexes vertices"):
        spde.MaternOperator(vertices, faces + vertices.shape[0], parameters, covariance)

    broken = vertices.copy()
    broken[0, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        spde.MaternOperator(broken, faces, parameters, covariance)


def test_refuses_a_face_with_no_area() -> None:
    covariance = VonKarmanFilterParameters(1.0, 1.0)
    faces = np.array([[0, 1, 2]])
    with pytest.raises(ValueError, match="no area"):
        spde.MaternOperator(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            faces,
            None,
            covariance,
        )


def test_a_folded_parameter_domain_no_longer_concerns_the_sampler() -> None:
    # A face with area in three dimensions but none in projection is a fold. It
    # used to be refused here, because the frame was read off dX/du. The frame is
    # now the surface's own, so a fold is invisible to the operator -- it is the
    # mesh builder's admissibility question, about whether (u, v) can address the
    # fault, not about whether the field can be drawn on it.
    covariance = VonKarmanFilterParameters(1.0, 1.0)
    folded = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 1.0]])
    operator = quiet(
        spde.MaternOperator, vertices, np.array([[0, 1, 2]]), folded, covariance
    )
    assert np.isfinite(operator.draw(np.random.default_rng(0))).all()


def test_refuses_a_vertex_starved_of_area() -> None:
    # The CFM Puyseguer defect: a near-duplicate vertex carrying one needle
    # triangle, which leaves it almost no area, runs its variance away and -- via
    # `standardise` -- shrinks the whole segment.
    covariance = correlation_lengths(MAGNITUDE)
    vertices, faces, _, shape = square_mesh(4, 4, covariance)
    anchor = (shape[0] // 2) * shape[1] + shape[1] // 2
    step = vertices[anchor + 1] - vertices[anchor]
    twin = vertices[anchor] + 1.0e-7 * step
    grown = np.vstack([vertices, twin])
    needle = np.vstack([faces, [[anchor, len(vertices), anchor + shape[1]]]])

    with pytest.raises(ValueError, match="starved|left .* km\\^2"):
        quiet(spde.MaternOperator, grown, needle, None, covariance)
    # And the message names the vertex and both areas, so the mesh can be fixed.
    try:
        quiet(spde.MaternOperator, grown, needle, None, covariance)
    except ValueError as refusal:
        assert str(len(vertices)) in str(refusal)
        assert "median" in str(refusal)
        assert "remesh" in str(refusal)


def test_a_healthy_mesh_is_far_from_the_starvation_floor() -> None:
    # The floor has to be somewhere the meshes this package builds never go.
    covariance = correlation_lengths(MAGNITUDE)
    vertices, faces, _, _ = square_mesh(8, 4, covariance)
    lumped, _, _ = spde._assemble(vertices, faces, covariance)
    ratio = lumped.min() / np.median(lumped)
    assert ratio > 1000.0 * spde.MINIMUM_LUMPED_MASS_RATIO


# ----------------------------------------------------------------- warnings


def test_warns_past_mai_figure_13() -> None:
    # Carried over from `sampling._warn_if_degraded`: a statement about the
    # model's validity, which survives the change of method.
    covariance = VonKarmanFilterParameters(9.0, 9.0)
    vertices, faces, parameters, _ = grid_mesh(8, 8, 1.0, 1.0)
    # A ratio past 0.6 is under 1.7 correlation lengths across, so the folding
    # warning necessarily fires too; this test is about the other one.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Neumann.*")
        with pytest.warns(DegradedCorrelation, match="figure 13"):
            spde.MaternOperator(vertices, faces, parameters, covariance)


def test_warns_when_the_folding_is_the_field() -> None:
    covariance = correlation_lengths(MAGNITUDE)
    vertices, faces, parameters, _ = square_mesh(2, 4, covariance)
    with pytest.warns(DegradedCorrelation, match="Neumann"):
        spde.MaternOperator(vertices, faces, parameters, covariance)


def test_quiet_on_a_domain_that_carries_its_covariance() -> None:
    covariance = correlation_lengths(MAGNITUDE)
    vertices, faces, parameters, _ = square_mesh(8, 4, covariance)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spde.MaternOperator(vertices, faces, parameters, covariance)


# -------------------------------------------------------------- the sampler


def test_draw_is_the_rngs_alone() -> None:
    covariance = correlation_lengths(MAGNITUDE)
    vertices, faces, parameters, _ = square_mesh(8, 4, covariance)
    operator = spde.MaternOperator(vertices, faces, parameters, covariance)
    first = operator.draw(np.random.default_rng(4))
    second = operator.draw(np.random.default_rng(4))
    assert first == pytest.approx(second)
    assert np.isfinite(first).all()
    assert first.shape == (vertices.shape[0],)


def test_the_operator_is_mesh_native() -> None:
    """The cleanest statement of what the surface frame bought.

    Nothing in the operator reads the parameter coordinates, so the same surface
    addressed through a different chart draws the *same field*, bit for bit. The
    parameter coordinates survive only so the two segment-size checks can report.
    """
    covariance = correlation_lengths(MAGNITUDE)
    vertices, faces, parameters, _ = square_mesh(
        8, 4, covariance, height=lambda u, v: 0.3 * u + 0.15 * v
    )
    with_chart = spde.MaternOperator(vertices, faces, parameters, covariance).draw(
        np.random.default_rng(6)
    )
    without = spde.MaternOperator(vertices, faces, None, covariance).draw(
        np.random.default_rng(6)
    )
    assert with_chart == pytest.approx(without, rel=0.0, abs=0.0)

    # A different chart on the same geometry: skew and stretch (u, v) arbitrarily.
    skewed = np.column_stack(
        [3.0 * parameters[:, 0] + 0.7 * parameters[:, 1], -2.0 * parameters[:, 1]]
    )
    reparameterised = quiet(
        spde.MaternOperator, vertices, faces, skewed, covariance
    ).draw(np.random.default_rng(6))
    assert reparameterised == pytest.approx(without, rel=0.0, abs=0.0)


def test_matern_field_is_one_operators_draw() -> None:
    covariance = correlation_lengths(MAGNITUDE)
    vertices, faces, parameters, _ = square_mesh(8, 4, covariance)
    field = spde.matern_field(
        vertices, faces, parameters, covariance, np.random.default_rng(9)
    )
    operator = spde.MaternOperator(vertices, faces, parameters, covariance)
    assert field == pytest.approx(operator.draw(np.random.default_rng(9)))


def test_covariance_column_is_the_draws() -> None:
    # What licenses every other test in this file to use the exact covariance
    # instead of sampling: the two agree to the Monte Carlo error of the draws.
    covariance = correlation_lengths(MAGNITUDE)
    vertices, faces, parameters, shape = square_mesh(8, 4, covariance)
    operator = spde.MaternOperator(vertices, faces, parameters, covariance)
    probe = (shape[0] // 2) * shape[1] + shape[1] // 2

    rng = np.random.default_rng(17)
    draws = 3000
    total = np.zeros(vertices.shape[0])
    for _ in range(draws):
        field = operator.draw(rng)
        total += field * field[probe]
    empirical = total / draws
    exact = operator.covariance_column(probe)
    # A covariance estimated from n draws carries a standard error of order
    # sigma^2 / sqrt(n); at 3000 draws that is ~2e-2, and the two agree to it.
    assert np.abs(empirical - exact).max() < 0.1
    assert empirical[probe] == pytest.approx(exact[probe], rel=0.05)


def test_covariance_column_refuses_a_vertex_that_is_not_there() -> None:
    covariance = correlation_lengths(MAGNITUDE)
    vertices, faces, parameters, _ = square_mesh(8, 4, covariance)
    operator = spde.MaternOperator(vertices, faces, parameters, covariance)
    with pytest.raises(ValueError, match="off a mesh"):
        operator.covariance_column(vertices.shape[0])


# ------------------------------------------------------------ face reduction


def test_face_values_is_the_finite_element_field_at_the_centroid() -> None:
    # Not a choice of reduction: the P1 basis functions are each 1/3 at the
    # centroid, so the mean of the corners is the field's own value there.
    values = np.array([1.0, 4.0, 7.0, -2.0])
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    assert spde.face_values(values, faces) == pytest.approx([4.0, 3.0])


def test_face_values_refuses_a_field_that_is_not_the_meshs() -> None:
    faces = np.array([[0, 1, 2]])
    with pytest.raises(ValueError, match="one value per vertex"):
        spde.face_values(np.zeros((3, 2)), faces)
    with pytest.raises(ValueError, match="indexes vertex"):
        spde.face_values(np.zeros(2), faces)


# ------------------------------------------------------------------- gate 1


@pytest.mark.slow
def test_planar_covariance_matches_analytic() -> None:
    """Gate 1: the delivered covariance is Mai & Beroza's von Karman ACF."""
    covariance = correlation_lengths(MAGNITUDE)
    assert covariance.correlation_length_strike_km == pytest.approx(10.0)
    assert covariance.correlation_length_dip_km == pytest.approx(6.813, abs=1e-3)

    per_length = 8
    vertices, faces, parameters, shape = square_mesh(16, per_length, covariance)
    operator = spde.MaternOperator(vertices, faces, parameters, covariance)
    radii = np.arange(3 * per_length + 1) / per_length
    strike, dip, analytic, variance = centre_profile(operator, shape, covariance, radii)

    assert operator.error.mesh_width == pytest.approx(0.1768, abs=1e-3)
    # The anisotropy is right if and only if the two axes deliver the *same*
    # dimensionless profile from correlation lengths that differ by 47%.
    assert strike == pytest.approx(dip, abs=1e-6)
    assert np.abs(strike - analytic).max() < 1.2e-2
    assert np.abs(dip - analytic).max() < 1.2e-2
    # The marginal variance is the continuum one to within the discretisation.
    assert variance == pytest.approx(1.0, abs=0.03)

    # Read as a correlation length, which is the units `sampling.py` states its
    # own tolerance in: within CORRELATION_LENGTH_TOLERANCE of the target.
    half = float(von_karman_correlation(np.array([1.0]), covariance.hurst)[0])
    below = int(np.flatnonzero(strike <= half)[0])
    above, under = strike[below - 1], strike[below]
    delivered = radii[below - 1] + (above - half) / (above - under) / per_length
    assert abs(delivered - 1.0) < sampling.CORRELATION_LENGTH_TOLERANCE


# ------------------------------------------------------------------- gate 2


@pytest.mark.slow
def test_matches_the_circulant_sampler() -> None:
    """Gate 2: the same covariance the lattice sampler delivers, on a regular mesh."""
    covariance = correlation_lengths(MAGNITUDE)
    per_length = 8
    cells = 16 * per_length
    spacing = (
        covariance.correlation_length_strike_km / per_length,
        covariance.correlation_length_dip_km / per_length,
    )

    embedding = sampling._embed((cells, cells), spacing, covariance)
    delivered = np.fft.ifft2(embedding.eigenvalues).real
    delivered = delivered / delivered[0, 0]

    vertices, faces, parameters, shape = square_mesh(16, per_length, covariance)
    operator = spde.MaternOperator(vertices, faces, parameters, covariance)
    radii = np.arange(3 * per_length + 1) / per_length
    strike, dip, analytic, _ = centre_profile(operator, shape, covariance, radii)

    # The circulant path delivers the ACF exactly -- it evaluates it at lattice
    # points -- so this comparison is the SPDE's discretisation error and nothing
    # else. Worth asserting rather than assuming, since it is what makes the
    # circulant sampler usable as a reference at all.
    assert np.abs(delivered[0, : radii.size] - analytic).max() < 1e-6
    assert np.abs(delivered[: radii.size, 0] - analytic).max() < 1e-6

    assert np.abs(strike - delivered[: radii.size, 0]).max() < 1.2e-2
    assert np.abs(dip - delivered[0, : radii.size]).max() < 1.2e-2


# ------------------------------------------------------------------- gate 3


def surface_prediction(
    slope: float, covariance: VonKarmanFilterParameters, dip_deg: float = DIP_DEG
) -> tuple[float, float]:
    """Exact delivered correlation lengths on a plane tilted by ``h = slope * u``.

    Everything is constant on a tilted plane, so ``H`` can be written down and the
    answer read off rather than simulated. The correlation length along a
    parameter direction ``X_d`` is where the ``H`` metric reaches one, that is
    ``1 / sqrt(X_d^T H^-1 X_d)``; returned in units of the strike and dip
    correlation lengths so that 1.0 means "unmoved".
    """
    frame_u, frame_v, normal = monge_frame(dip_deg)
    tangent_u = frame_u + slope * normal
    tangent_v = frame_v
    tilted = np.cross(tangent_u, tangent_v)
    tilted = tilted / np.linalg.norm(tilted)
    strike = np.cross([0.0, 0.0, 1.0], tilted)
    strike = strike / np.linalg.norm(strike)
    dip = np.cross(tilted, strike)

    strike_length = covariance.correlation_length_strike_km
    dip_length = covariance.correlation_length_dip_km
    inverse = (
        np.outer(strike, strike) / strike_length**2 + np.outer(dip, dip) / dip_length**2
    )
    along_u = 1.0 / np.sqrt(tangent_u @ inverse @ tangent_u)
    along_v = 1.0 / np.sqrt(tangent_v @ inverse @ tangent_v)
    return along_u / strike_length, along_v / dip_length


@pytest.mark.slow
def test_warped_patch_tracks_the_surface_frame():
    """Gate 3: the lifted assembly makes the field Matern *on the surface*.

    Run at the gradients the real CFM interfaces exhibit, not only the 0.33 of
    `MESH.md`: the shipped Puyseguer surface reaches ``|grad h| = 2.05``, where
    the surface and projected metrics differ by 124% rather than 5%.

    The reference is exact. On a tilted plane ``H`` is constant, so the delivered
    correlation length along each parameter axis is a closed form
    (:func:`surface_prediction`) -- no arc length and no geodesic anywhere. A
    sampler assembling from the projected triangles, or one that ignored how the
    tilt rotates strike and dip, would deliver 1.0 on both axes instead.

    Both axes are checked, which is what makes this a test of the anisotropy
    tensor and not merely of the metric: the tilt rotates the frame within the
    horizontal plane, so the two correlation lengths are redistributed between
    the parameter directions rather than simply rescaled.
    """
    covariance = correlation_lengths(MAGNITUDE)
    per_length = 8
    half = float(von_karman_correlation(np.array([1.0]), covariance.hurst)[0])

    def delivered(slope: float) -> tuple[float, float, float]:
        stretch = np.sqrt(1.0 + slope**2)
        # Hold the *surface* resolution fixed, so a steeper patch is not also a
        # coarser mesh and the two effects cannot be confused.
        cells_u = round(16 * per_length * stretch)
        vertices, faces, parameters, shape = grid_mesh(
            cells_u,
            16 * per_length,
            16 * covariance.correlation_length_strike_km / cells_u,
            covariance.correlation_length_dip_km / per_length,
            height=(lambda u, v: slope * u) if slope else None,
        )
        operator = spde.MaternOperator(vertices, faces, parameters, covariance)
        row, column = shape[0] // 2, shape[1] // 2
        values = operator.covariance_column(row * shape[1] + column).reshape(shape)

        def crossing(profile: FloatArray, lag: FloatArray) -> float:
            below = int(np.flatnonzero(profile <= half)[0])
            high, low = profile[below - 1], profile[below]
            return float(
                lag[below - 1]
                + (high - half) / (high - low) * (lag[below] - lag[below - 1])
            )

        steps_u = int(3 * per_length * stretch)
        parameter_u = parameters[:, 0].reshape(shape)[:, 0]
        along_u = crossing(
            values[row : row + steps_u + 1, column] / values[row, column],
            (parameter_u[row : row + steps_u + 1] - parameter_u[row])
            / covariance.correlation_length_strike_km,
        )
        steps_v = 3 * per_length
        parameter_v = parameters[:, 1].reshape(shape)[0, :]
        along_v = crossing(
            values[row, column : column + steps_v + 1] / values[row, column],
            (parameter_v[column : column + steps_v + 1] - parameter_v[column])
            / covariance.correlation_length_dip_km,
        )
        return along_u, along_v, operator.error.mesh_width

    flat_u, flat_v, flat_width = delivered(0.0)
    predicted_u, predicted_v = surface_prediction(0.0, covariance)
    assert predicted_u == pytest.approx(1.0)
    assert predicted_v == pytest.approx(1.0)
    # The flat case's own discretisation bias, which the tilted cases inherit.
    assert flat_u == pytest.approx(0.986, abs=3e-3)
    assert flat_v == pytest.approx(0.986, abs=3e-3)

    for slope in (0.33, 1.0, 2.0):
        along_u, along_v, width = delivered(slope)
        # The geometric stretch was compensated; what is left is the anisotropy
        # being redistributed by the frame rotation, which is the effect under
        # test rather than a change of resolution.
        assert width == pytest.approx(flat_width, rel=0.3)
        surface_u, surface_v = surface_prediction(slope, covariance)
        corrected_u, corrected_v = along_u / flat_u, along_v / flat_v

        # The dip axis is held a little looser than the strike axis: the
        # discretisation bias divided out is the flat mesh's, and the tilt does
        # not change the effective resolution equally on the two axes.
        assert corrected_u == pytest.approx(surface_u, abs=0.02)
        assert corrected_v == pytest.approx(surface_v, abs=0.04)
        # Decisively closer to the surface reading than to the projected one,
        # which predicts 1.0 on both axes.
        assert abs(corrected_u - surface_u) < 0.2 * abs(corrected_u - 1.0)


@pytest.mark.slow
def test_warped_patch_discrimination_grows_with_gradient():
    # `MESH.md` justified gate 3 at |grad h| = 0.33, where surface and projected
    # separation differ by 5.3%. The real interfaces reach 2.05, where they
    # differ by 124% -- so the plan's gate is the weakest case, not the typical
    # one, and the margin only ever grows.
    covariance = correlation_lengths(MAGNITUDE)
    margins = []
    for slope in (0.33, 1.0, 2.0):
        surface_u, _ = surface_prediction(slope, covariance)
        margins.append(abs(1.0 - surface_u))
    # Measured against the exact H metric: the tilt both stretches the surface
    # and rotates strike and dip within it, so the parameter axes move further
    # than the 5.3% metric factor alone would put them.
    assert margins == pytest.approx([0.0959, 0.3991, 0.6364], abs=2e-3)
    assert margins == sorted(margins)
    assert margins[2] > 6.0 * margins[0]
    # The metric factor at the steepest real face on the shipped CFM data.
    assert np.sqrt(1.0 + 2.048**2) == pytest.approx(2.279, abs=1e-3)


# ------------------------------------------------------------------- gate 4


@pytest.mark.slow
def test_bound_is_a_bound() -> None:
    """Gate 4: the reported model error bounds the measured one, at the right rate.

    Bolin & Kirchner theorem 3.3 carries a constant "independent of h, m" and
    otherwise unknown, pinned here as `MODEL_ERROR_CONSTANT`. What makes the
    reported number a bound rather than a fit is that the ratio of measured error
    to bracketed quantity stays *flat* while the mesh refines: a drifting ratio
    would mean the rate was wrong.
    """
    covariance = correlation_lengths(MAGNITUDE)
    ratios = []
    for per_length in (1, 2, 4, 8):
        vertices, faces, parameters, shape = square_mesh(16, per_length, covariance)
        operator = spde.MaternOperator(vertices, faces, parameters, covariance)
        radii = np.arange(3 * per_length + 1) / per_length
        strike, dip, analytic, _ = centre_profile(operator, shape, covariance, radii)
        measured = max(
            float(np.abs(strike - analytic).max()), float(np.abs(dip - analytic).max())
        )
        error = operator.error
        assert error.bound == pytest.approx(
            spde.MODEL_ERROR_CONSTANT
            * (error.finite_element_term + error.rational_term)
        )
        assert measured < error.bound
        ratios.append(measured / error.bound)

    # A factor of 8 in h and a factor of 4 in the bracketed quantity, with the
    # ratio moving by a factor of three and never leaving the band.
    assert min(ratios) > 0.25
    assert max(ratios) <= 1.0
    assert ratios == pytest.approx([0.274, 0.706, 0.832, 0.582], abs=0.06)


@pytest.mark.slow
def test_bound_terms_are_theorem_3_3() -> None:
    # The two exponents at Mai's beta, spelled out: the finite element term
    # converges at h^0.75 and the rational term *diverges* at h^-1.25, which is
    # the theorem's honest statement that refining without raising m stops
    # helping. Remark 3.4 is the counterpart.
    covariance = correlation_lengths(MAGNITUDE)
    coarse_terms = []
    for per_length in (4, 8):
        vertices, faces, parameters, _ = square_mesh(8, per_length, covariance)
        operator = quiet(spde.MaternOperator, vertices, faces, parameters, covariance)
        coarse_terms.append(operator.error)
    first, second = coarse_terms
    scale = second.mesh_width / first.mesh_width
    assert second.finite_element_term / first.finite_element_term == pytest.approx(
        scale**0.75, rel=1e-6
    )
    assert second.rational_term / first.rational_term == pytest.approx(
        scale**-1.25, rel=1e-6
    )


# --------------------------------------------------------- measured studies


@pytest.mark.slow
def test_boundary_folding_is_lindgren_a4() -> None:
    """The measurement `_warn_if_folded` quotes, and the difference from circulant.

    Lindgren et al. (2011) appendix A.4 theorem 1: the Neumann covariance is the
    Matern one reflected at the boundary. On a fault-sized domain that is not an
    edge effect.
    """
    covariance = correlation_lengths(MAGNITUDE)
    per_length = 8
    target = float(von_karman_correlation(np.array([1.0]), covariance.hurst)[0])
    observed = {}
    for lengths in (2, 4, 8, 16):
        vertices, faces, parameters, shape = square_mesh(
            lengths, per_length, covariance
        )
        operator = quiet(spde.MaternOperator, vertices, faces, parameters, covariance)
        row, column = shape[0] // 2, shape[1] // 2
        values = operator.covariance_column(row * shape[1] + column).reshape(shape)
        observed[lengths] = (
            float(values[row, column]),
            float(values[row + per_length, column] / values[row, column]),
        )

    variances = {k: v[0] for k, v in observed.items()}
    correlations = {k: v[1] for k, v in observed.items()}
    assert variances[2] == pytest.approx(2.586, abs=0.05)
    assert variances[4] == pytest.approx(1.168, abs=0.02)
    assert variances[16] == pytest.approx(1.022, abs=0.02)
    assert correlations[2] == pytest.approx(0.878, abs=0.02)
    assert correlations[4] == pytest.approx(0.584, abs=0.02)
    assert correlations[16] == pytest.approx(0.495, abs=0.02)
    # Monotone towards the target as the domain grows, and only the widest
    # domains are anywhere near it.
    assert correlations[2] > correlations[4] > correlations[8] > target - 0.01

    # The circulant sampler is unaffected at every one of those sizes, because it
    # pads the embedding and crops. This is the difference between the samplers.
    for lengths in (2, 4, 16):
        cells = lengths * per_length
        embedding = sampling._embed(
            (cells, cells),
            (
                covariance.correlation_length_strike_km / per_length,
                covariance.correlation_length_dip_km / per_length,
            ),
            covariance,
        )
        delivered = np.fft.ifft2(embedding.eigenvalues).real
        assert delivered[0, per_length] / delivered[0, 0] == pytest.approx(
            target, abs=1e-3
        )


@pytest.mark.slow
def test_face_values_smoothing_converges() -> None:
    """The table in :func:`face_values`: the P1 within-element artefact goes with h."""
    covariance = correlation_lengths(MAGNITUDE)
    ratios = {}
    for lengths, per_length in ((16, 2), (16, 4), (16, 8), (8, 16)):
        vertices, faces, parameters, shape = square_mesh(
            lengths, per_length, covariance
        )
        operator = quiet(spde.MaternOperator, vertices, faces, parameters, covariance)
        centre = (shape[0] // 2) * shape[1] + shape[1] // 2
        probe = int(np.flatnonzero((faces == centre).any(axis=1))[0])
        columns = np.stack([operator.covariance_column(int(v)) for v in faces[probe]])
        face_to_face = columns.mean(axis=0)[faces].mean(axis=1)
        vertex = operator.covariance_column(centre)
        ratios[round(operator.error.mesh_width, 3)] = float(
            face_to_face[probe] / vertex[centre]
        )

    assert ratios[0.707] == pytest.approx(0.783, abs=0.01)
    assert ratios[0.354] == pytest.approx(0.897, abs=0.01)
    assert ratios[0.177] == pytest.approx(0.957, abs=0.01)
    assert ratios[0.088] == pytest.approx(0.983, abs=0.01)
    # Monotone towards 1 as the mesh refines, which is what "artefact" means.
    assert sorted(ratios.values()) == list(ratios.values())


@pytest.mark.slow
def test_fixed_fitting_interval_beats_the_mesh_spectrum() -> None:
    """Why :func:`_interval_floor` does not follow the mesh, measured.

    Widening the fit to cover the whole discrete spectrum makes it formally valid
    there and the delivered covariance three and a half times worse, because the
    supremum norm weights modes the field gives almost no variance to.
    """
    covariance = correlation_lengths(MAGNITUDE)
    per_length = 32
    vertices, faces, parameters, shape = square_mesh(8, per_length, covariance)
    radii = np.arange(3 * per_length + 1) / per_length

    results = {}
    original = spde._interval_floor
    try:
        for name, floor in (("fixed", None), ("spectrum", 1e-5), ("colombia", 1e-6)):
            if floor is None:
                spde._interval_floor = original
            else:
                spde._interval_floor = lambda order, value=floor: value
            operator = spde.MaternOperator(vertices, faces, parameters, covariance)
            strike, _, analytic, _ = centre_profile(operator, shape, covariance, radii)
            results[name] = (
                operator.error.rational_supremum_error,
                float(np.abs(strike - analytic).max()),
            )
    finally:
        spde._interval_floor = original

    # This mesh's spectrum reaches 8.1e-5, so covering it needs delta = 1e-5; the
    # `colombia` mesh reaches 5e-6 and would need 1e-6.
    assert results["fixed"][0] == pytest.approx(1.13e-2, rel=0.1)
    assert results["spectrum"][0] == pytest.approx(5.25e-2, rel=0.1)
    assert results["colombia"][0] == pytest.approx(1.08e-1, rel=0.1)
    assert results["fixed"][1] == pytest.approx(8.3e-3, abs=2e-3)
    assert results["spectrum"][1] == pytest.approx(1.62e-2, abs=3e-3)
    assert results["colombia"][1] == pytest.approx(4.35e-2, abs=5e-3)

    # Covering the spectrum is worse in the supremum norm *and*, which is the
    # point, in the covariance the field actually gets.
    assert results["spectrum"][1] > 1.7 * results["fixed"][1]
    assert results["colombia"][1] > 4.0 * results["fixed"][1]


@pytest.mark.slow
def test_draw_cost_at_fault_scale() -> None:
    """The profile `MESH.md`'s risks section asks for, on a `colombia`-sized mesh.

    185150 vertices and 367836 faces, the shipped example's 175 x 1058 nodes at
    0.1 km triangulated. Asserted only as order-of-magnitude ceilings, generously:
    the point is the record, and a wall-clock assertion that is tight is a
    flake on someone else's machine.

    **This is the memory high-water mark of the file**: the three SuperLU factors
    hold 34 M nonzeros between them and the process peaks around 0.9 GB at
    ``m = 2`` (1.2 GB at ``m = 3``, which is why the profile is not parameterised
    over the order). Everything else here runs under 100 MB. Give it 2 GB.
    """
    import time

    covariance = correlation_lengths(7.4)
    vertices, faces, parameters, _ = grid_mesh(1057, 174, 0.1, 0.1)
    assert vertices.shape[0] == 185150
    assert faces.shape[0] == 367836

    start = time.perf_counter()
    operator = quiet(spde.MaternOperator, vertices, faces, parameters, covariance)
    setup = time.perf_counter() - start

    rng = np.random.default_rng(2)
    operator.draw(rng)
    start = time.perf_counter()
    for _ in range(4):
        field = operator.draw(rng)
    per_draw = (time.perf_counter() - start) / 4

    # Measured: 2.5 s setup (0.5 s numpy assembly, 1.5 s SuperLU on three
    # matrices), 57 ms a draw, at m = 2. Four fields a segment is a quarter of a
    # second against a setup that is amortised over them.
    assert setup < 20.0
    assert per_draw < 1.0
    assert np.isfinite(field).all()
    assert operator.error.mesh_width == pytest.approx(0.0125, abs=1e-3)
    # The whole fault is inside the folding, which is the finding that matters.
    assert operator.error.boundary_reach > 17.4


@pytest.mark.slow
def test_irregular_delaunay_mesh_delivers_the_same_covariance() -> None:
    """The reason for the migration: nothing here needs a lattice.

    Every other test in this file runs on a regular grid, which is the one case
    the circulant sampler could also have done. This one triangulates jittered
    points in the parameter domain -- `MESH.md` Component 1's "triangulate the 2D
    parameter domain (Delaunay, always valid) and lift to 3D" -- so the faces have
    a spread of areas and shapes, and asserts the covariance is unmoved.

    The same points are triangulated twice, unjittered and jittered, so the
    comparison isolates irregularity from resolution. The error is read off every
    vertex at its own separation rather than binned by lag, because on an
    irregular mesh there is no lag to index by -- that is the whole difference.
    """
    from scipy.spatial import Delaunay

    covariance = correlation_lengths(MAGNITUDE)
    lengths, per_length = 12, 6
    step_u = covariance.correlation_length_strike_km / per_length
    step_v = covariance.correlation_length_dip_km / per_length
    count = lengths * per_length

    grid_u, grid_v = np.meshgrid(
        np.arange(count + 1) * step_u, np.arange(count + 1) * step_v, indexing="ij"
    )
    regular = np.stack([grid_u.ravel(), grid_v.ravel()], axis=-1)
    interior = (
        (regular[:, 0] > 0.0)
        & (regular[:, 0] < count * step_u)
        & (regular[:, 1] > 0.0)
        & (regular[:, 1] < count * step_v)
    )
    # Jitter the interior only, so the domain stays the rectangle it claims to be.
    rng = np.random.default_rng(31)
    jittered = regular + rng.uniform(-0.3, 0.3, regular.shape) * (
        [step_u, step_v] * interior[:, None]
    )

    def measure(parameters: FloatArray) -> tuple[float, float, float, float]:
        faces = np.asarray(Delaunay(parameters).simplices, dtype=np.int64)
        frame_u, frame_v, _ = monge_frame(DIP_DEG)
        vertices = parameters[:, 0, None] * frame_u + parameters[:, 1, None] * frame_v
        operator = spde.MaternOperator(vertices, faces, parameters, covariance)
        areas, _, _, _, _ = spde._surface_frames(vertices, faces)

        # A probe in the deep interior, so the folding is not what is measured.
        centre = np.array([count * step_u / 2.0, count * step_v / 2.0])
        probe = int(np.argmin(np.linalg.norm(parameters - centre, axis=1)))
        values = operator.covariance_column(probe)
        offset = parameters - parameters[probe]
        radius = np.sqrt(
            (offset[:, 0] / covariance.correlation_length_strike_km) ** 2
            + (offset[:, 1] / covariance.correlation_length_dip_km) ** 2
        )
        # The inner cut is at a quarter of a correlation length because the von
        # Karman ACF has a cusp at the origin that no piecewise-linear basis
        # resolves; the outer one keeps the Neumann folding out.
        near = (radius >= 0.25) & (radius <= 3.0)
        # Enough of the mesh that the maximum below is a claim about the field
        # rather than about a handful of vertices.
        assert near.sum() > 900
        error = np.abs(
            values[near] / values[probe]
            - von_karman_correlation(radius[near], covariance.hurst)
        )
        return (
            operator.error.mesh_width,
            float(areas.max() / areas.min()),
            float(error.max()),
            float(values[probe]),
        )

    width, spread, error, variance = measure(regular)
    assert spread == pytest.approx(1.0)
    assert width == pytest.approx(0.236, abs=0.01)
    assert error == pytest.approx(4.3e-3, abs=2e-3)
    assert variance == pytest.approx(1.0, abs=0.02)

    width, spread, error, variance = measure(jittered)
    # Genuinely irregular: an order of magnitude of face area, not a lattice in
    # disguise -- and the jitter lengthens the longest edge, so h grows with it.
    assert spread > 5.0
    assert width == pytest.approx(0.384, abs=0.02)
    # 2.7e-2 measured, against 2.1e-2 for a *regular* mesh at that same h read
    # off `test_bound_is_a_bound`'s refinement. So irregularity costs about 30%
    # beyond what the mesh width alone accounts for -- which is the price of
    # leaving Bolin & Kirchner's quasi-uniform assumption, and it is a price and
    # not a failure.
    assert error == pytest.approx(2.7e-2, abs=5e-3)
    assert variance == pytest.approx(1.0, abs=0.06)


# --------------------------------------------------------------- the padding


def padded_builder(
    lengths_u: float,
    lengths_v: float,
    per_length: int,
    covariance: VonKarmanFilterParameters,
) -> Callable[[float, float], spde.PaddedMesh]:
    """A stand-in for the container's padded builder, for a rectangular fault.

    Mirrors the contract `TriangleMesh.from_patches`' docstring sets out: the pad
    arrives as its own argument, it is a lattice conforming along the fault
    boundary, its faces are marked as not-fault, and the fault's parameter
    coordinates do not move when the pad changes -- the pad is simply negative.
    """
    step_u = covariance.correlation_length_strike_km / per_length
    step_v = covariance.correlation_length_dip_km / per_length
    fault_u = round(lengths_u * per_length)
    fault_v = round(lengths_v * per_length)

    def build(pad_u_km: float, pad_v_km: float) -> spde.PaddedMesh:
        cells_u = round(pad_u_km / step_u)
        cells_v = round(pad_v_km / step_v)
        # The fault keeps (0, 0); the pad is negative. Nothing about the fault's
        # own coordinates depends on how wide the pad is.
        u = (np.arange(-cells_u, fault_u + cells_u + 1)) * step_u
        v = (np.arange(-cells_v, fault_v + cells_v + 1)) * step_v
        grid_u, grid_v = np.meshgrid(u, v, indexing="ij")
        parameters = np.stack([grid_u.ravel(), grid_v.ravel()], axis=-1)
        frame_u, frame_v, _ = monge_frame(DIP_DEG)
        vertices = parameters[:, 0, None] * frame_u + parameters[:, 1, None] * frame_v
        index = np.arange(u.size * v.size).reshape(u.size, v.size)
        a, b = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
        c, d = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
        faces = np.concatenate(
            [np.stack([a, b, c], axis=-1), np.stack([a, c, d], axis=-1)]
        )
        centroid = parameters[faces].mean(axis=1)
        # Indices, not a mask -- see `spde.Padding.fault_faces`: what a consumer needs
        # is the fault's faces *in the fault's own order*, which a mask cannot state.
        fault = np.flatnonzero(
            (centroid[:, 0] > 0.0)
            & (centroid[:, 0] < fault_u * step_u)
            & (centroid[:, 1] > 0.0)
            & (centroid[:, 1] < fault_v * step_v)
        )
        return vertices, faces, parameters, fault

    return build


@pytest.mark.slow
def test_padding_beats_the_boundary_reflection_on_a_crustal_fault():
    """A fault 2 correlation lengths across, padded until the covariance is right.

    This is where the folding bites: measured, the interfaces are 8 to 16
    correlation lengths across and need nothing, while `colombia` is 1.9.
    """
    covariance = correlation_lengths(MAGNITUDE)
    per_length = 8
    build = padded_builder(2.0, 2.0, per_length, covariance)

    # Unpadded, the reflection *is* the field.
    vertices, faces, parameters, fault = build(0.0, 0.0)
    bare = quiet(spde.MaternOperator, vertices, faces, parameters, covariance)
    unpadded = spde._delivered_correlation_length(
        bare, parameters, fault, faces, covariance
    )
    assert unpadded > 1.4, "a 2-length domain should be badly folded"

    padded = quiet(spde.padded_operator, build, covariance)
    assert padded.correlation_length_error <= CORRELATION_LENGTH_TOLERANCE
    assert padded.delivered_correlation_length == pytest.approx(1.0, abs=0.02)
    # The first guess is Lindgren appendix A.4's own reach, so it should usually
    # be the answer rather than the start of a search.
    assert padded.pad_lengths == spde.BOUNDARY_FOLDING_LENGTHS
    assert padded.pad_km[0] == pytest.approx(
        spde.BOUNDARY_FOLDING_LENGTHS * covariance.correlation_length_strike_km
    )
    # And the pad really did shrink the error, by a lot.
    assert padded.correlation_length_error < 0.1 * abs(unpadded - 1.0)


@pytest.mark.slow
def test_padding_draws_only_the_fault():
    covariance = correlation_lengths(MAGNITUDE)
    build = padded_builder(2.0, 2.0, 6, covariance)
    padded = quiet(spde.padded_operator, build, covariance)
    field = padded.draw_on_faces(np.random.default_rng(3))
    assert field.shape == (padded.fault_faces.size,)
    assert np.isfinite(field).all()
    # The pad is a real fraction of the padded domain, so this is not a no-op.
    assert padded.fault_faces.size < 0.4 * len(padded.operator.faces)


def test_a_pad_that_touches_nothing_is_refused():
    """The bug this refusal exists for, reproduced: a pad that is not conforming.

    A pad whose vertices are its own -- surrounding the fault, overlapping it in the
    parameter plane, sharing not one node -- is a second connected component. The
    operator never couples to it, so the field on the fault is the unpadded one *exactly*
    and nothing about the pad's geometry looks wrong. This builds that mesh deliberately
    by renumbering the pad's vertices away from the fault's, and asserts the refusal
    names it.
    """
    covariance = correlation_lengths(MAGNITUDE)
    conforming = padded_builder(2.0, 2.0, 4, covariance)

    def adrift(pad_u_km: float, pad_v_km: float) -> spde.PaddedMesh:
        vertices, faces, parameters, fault = conforming(pad_u_km, pad_v_km)
        pad = np.setdiff1d(np.arange(len(faces)), fault)
        # Give every pad face its own copy of every vertex it uses: the same geometry,
        # in the same place, joined to nothing.
        fresh = faces[pad].ravel()
        renumbered = np.arange(fresh.size) + len(vertices)
        return (
            np.concatenate([vertices, vertices[fresh]]),
            np.concatenate([faces[fault], renumbered.reshape(-1, 3)]),
            np.concatenate([parameters, parameters[fresh]]),
            np.arange(len(fault)),
        )

    with pytest.raises(ValueError, match="connected components"):
        quiet(spde.padded_operator, adrift, covariance)


@pytest.mark.slow
def test_padding_candidates_double_from_lindgrens_reach():
    covariance = correlation_lengths(MAGNITUDE)
    candidates = spde._pad_candidates(covariance)
    assert candidates[0] == spde.BOUNDARY_FOLDING_LENGTHS
    assert candidates == [
        spde.BOUNDARY_FOLDING_LENGTHS * 2.0**k for k in range(len(candidates))
    ]
    assert len(candidates) == MAXIMUM_DOUBLINGS + 1


@pytest.mark.slow
def test_an_interface_sized_segment_needs_no_padding():
    # The measurement that says this is a crustal problem: at 8 correlation
    # lengths across, the unpadded domain already delivers the covariance.
    covariance = correlation_lengths(MAGNITUDE)
    build = padded_builder(8.0, 8.0, 8, covariance)
    vertices, faces, parameters, fault = build(0.0, 0.0)
    bare = quiet(spde.MaternOperator, vertices, faces, parameters, covariance)
    delivered = spde._delivered_correlation_length(
        bare, parameters, fault, faces, covariance
    )
    assert abs(delivered - 1.0) <= CORRELATION_LENGTH_TOLERANCE


@pytest.mark.slow
def test_the_tolerance_bounds_folding_and_discretisation_together():
    """Why padding cannot rescue a mesh too coarse to carry the covariance.

    Unlike `sampling._delivered_lengths`, which measures target and delivered with
    the same estimator so the grid's coarseness cancels, this check compares
    against the analytic 1.0 -- so the finite element bias is inside the tolerance
    alongside the folding. On an 8-correlation-length domain, where there is
    nothing left to fold, the *mesh* is what decides.
    """
    covariance = correlation_lengths(MAGNITUDE)
    errors = {}
    for per_length in (6, 8, 12):
        build = padded_builder(8.0, 8.0, per_length, covariance)
        vertices, faces, parameters, fault = build(0.0, 0.0)
        operator = quiet(spde.MaternOperator, vertices, faces, parameters, covariance)
        delivered = spde._delivered_correlation_length(
            operator, parameters, fault, faces, covariance
        )
        errors[round(operator.error.mesh_width, 3)] = abs(delivered - 1.0)

    # Falls with the mesh, not with the domain -- so it is discretisation.
    assert list(errors.values()) == sorted(errors.values(), reverse=True)
    assert errors[0.236] > CORRELATION_LENGTH_TOLERANCE
    assert errors[0.177] <= CORRELATION_LENGTH_TOLERANCE
    assert errors[0.118] < 0.5 * CORRELATION_LENGTH_TOLERANCE


# ------------------------------------------------------------- the multigrid


def hierarchy(
    vertices_km: FloatArray, faces: IntArray, levels: int
) -> tuple[FloatArray, IntArray, list[tuple[FloatArray, IntArray, object]]]:
    """Refine ``levels`` times, returning the finest mesh and the coarser stack."""
    coarser = []
    for _ in range(levels):
        finer_vertices, finer_faces, prolongation = spde.subdivided(vertices_km, faces)
        coarser.append((vertices_km, faces, prolongation))
        vertices_km, faces = finer_vertices, finer_faces
    return vertices_km, faces, coarser


def test_subdivided_is_one_to_four_and_keeps_the_geometry() -> None:
    covariance = correlation_lengths(MAGNITUDE)
    vertices, faces, _, _ = square_mesh(2, 2, covariance)
    refined_vertices, refined_faces, prolongation = spde.subdivided(vertices, faces)

    assert refined_faces.shape[0] == 4 * faces.shape[0]
    # Coarse vertices keep their own indices, so the prolongation's top block is
    # the identity -- which is what makes the hierarchy free.
    assert prolongation.shape == (refined_vertices.shape[0], vertices.shape[0])
    assert refined_vertices[: vertices.shape[0]] == pytest.approx(vertices)
    # Total area is unchanged: midpoints lie on the coarse faces, so this refines
    # the discretisation and not the surface.
    coarse_area, _, _, _, _ = spde._surface_frames(vertices, faces)
    fine_area, _, _, _, _ = spde._surface_frames(refined_vertices, refined_faces)
    assert fine_area.sum() == pytest.approx(coarse_area.sum())


def test_prolongation_is_exact_on_linear_functions() -> None:
    # Linear interpolation reproduces a linear function exactly, which is the
    # property that makes P the right transfer for a piecewise-linear basis.
    covariance = correlation_lengths(MAGNITUDE)
    vertices, faces, _, _ = square_mesh(2, 2, covariance)
    refined_vertices, _, prolongation = spde.subdivided(vertices, faces)
    slope = np.array([0.3, -0.7, 1.1])
    assert prolongation @ (vertices @ slope) == pytest.approx(refined_vertices @ slope)


def test_multigrid_matches_the_direct_solver() -> None:
    """The covariance is the same object whichever route computes it.

    Draws differ pointwise between the two routes -- the three chained solves
    amplify, see :data:`spde.ITERATIVE_TOLERANCE` -- but the covariance is a
    quadratic functional of the operator and does not.
    """
    covariance = correlation_lengths(MAGNITUDE)
    base, base_faces, _, _ = square_mesh(16, 2, covariance)
    vertices, faces, coarser = hierarchy(base, base_faces, 2)

    direct = spde.MaternOperator(vertices, faces, None, covariance)
    multigrid = spde.MaternOperator(vertices, faces, None, covariance, coarser=coarser)
    assert multigrid.error.mesh_width == pytest.approx(direct.error.mesh_width)

    probe = int(np.argmin(np.linalg.norm(vertices - vertices.mean(axis=0), axis=1)))
    exact = direct.covariance_column(probe)
    iterative = multigrid.covariance_column(probe)
    assert iterative[probe] == pytest.approx(exact[probe], rel=1e-8)
    assert np.abs(iterative - exact).max() / exact[probe] < 1e-8


def test_multigrid_iteration_count_does_not_grow_with_the_mesh() -> None:
    """Mesh independence, which is the whole reason for a V-cycle.

    Measured on a well-shaped hierarchy from 4 thousand to 4.2 million vertices,
    the outer count is flat at about 12. Here two levels are enough to show it is
    not growing with ``1/h`` the way a Jacobi-preconditioned solve does.
    """
    covariance = correlation_lengths(MAGNITUDE)
    counts = []
    for levels in (2, 3):
        base, base_faces, _, _ = square_mesh(16, 1, covariance)
        vertices, faces, coarser = hierarchy(base, base_faces, levels)
        operator = spde.MaternOperator(
            vertices, faces, None, covariance, coarser=coarser
        )
        operator.draw(np.random.default_rng(0))
        counts.append(max(max(solver.iterations) for solver in operator._solvers))
    assert all(count < 30 for count in counts), counts
    # Four times the vertices must not cost twice the iterations.
    assert counts[1] < 1.5 * counts[0]


def test_multigrid_refuses_a_hierarchy_in_the_wrong_order() -> None:
    covariance = correlation_lengths(MAGNITUDE)
    base, base_faces, _, _ = square_mesh(8, 2, covariance)
    vertices, faces, coarser = hierarchy(base, base_faces, 2)
    with pytest.raises(ValueError, match="coarsest-first"):
        spde.MaternOperator(
            vertices, faces, None, covariance, coarser=list(reversed(coarser))
        )


def test_the_mass_gate_is_blind_to_refinement() -> None:
    """Why :data:`spde.MINIMUM_LUMPED_MASS_RATIO` is a backstop and not a guarantee.

    Subdivision divides every area by four, so the ratio of the smallest lumped
    mass to the median is **exactly invariant** -- the gate reads the same number
    however fine the mesh gets. The damage is not invariant: measured on the CFM
    Hikurangi interface, sixty vertices of three hundred thousand take the healthy
    field's amplitude to 0.266 after standardising, at a mass ratio fourteen times
    *above* the floor, while a coarser mesh's smallest vertex sits below the floor
    and is harmless. That measurement lives in the module docstring, because it
    needs data this repository does not ship; the identity it turns on is here.
    """
    covariance = correlation_lengths(MAGNITUDE)
    vertices, faces, _, shape = square_mesh(8, 2, covariance)
    anchor = (shape[0] // 2) * shape[1] + shape[1] // 2
    step = vertices[anchor + 1] - vertices[anchor]
    vertices = np.vstack([vertices, vertices[anchor] + 3.0e-3 * step])
    faces = np.vstack([faces, [[anchor, len(vertices) - 1, anchor + shape[1]]]])

    ratios = []
    for _ in range(3):
        lumped, _, _ = spde._assemble(vertices, faces, covariance)
        ratios.append(float(lumped.min() / np.median(lumped)))
        vertices, faces, _ = spde.subdivided(vertices, faces)

    # Invariant to a few parts in a thousand -- the residual is the median moving
    # as refinement changes which vertices are typical, not the sliver healing.
    assert ratios[1] == pytest.approx(ratios[0], rel=0.05)
    assert ratios[2] == pytest.approx(ratios[0], rel=0.05)
    # And it stays above the floor throughout, so the gate admits every level.
    assert min(ratios) > spde.MINIMUM_LUMPED_MASS_RATIO
