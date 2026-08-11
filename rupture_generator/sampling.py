"""Correlated Gaussian random fields with von Karman correlations.

Sampled by **circulant embedding** (Dietrich & Newsam 1993; Wood & Chan 1994), which
is exact: on the fault's own grid the drawn field has precisely the target covariance,
to floating point, rather than an approximation of it.

# The model is a covariance, not a spectrum

Mai & Beroza (2002) equation (1) gives the von Karman autocorrelation directly:

.. math::

    C(r) = \\frac{G_H(r)}{G_H(0)}, \\qquad G_H(r) = r^H K_H(r), \\qquad
    r = \\sqrt{\\frac{x^2}{a_x^2} + \\frac{z^2}{a_z^2}}

``r`` is a **dimensionless** distance -- the separation measured in correlation
lengths, one per axis -- and ``C`` is the standard Matern correlation of smoothness
``H``. That expression is what this module implements, and implementing it is what
makes ``a_x`` and ``a_z`` unambiguous.

The paper also states the matching power spectrum,
``P(k) = a_x a_z / (1 + k^2)^{H+1}`` with ``k`` the dimensionless wavenumber
``sqrt(a_x^2 k_x^2 + a_z^2 k_z^2)``. Sampling *that* instead requires deciding whether
``k_x`` is angular or in cycles, and the two answers differ by a factor of ``2*pi`` in
the delivered correlation length -- an ambiguity no output can adjudicate, because both
give plausible-looking slip. Working from ``C(r)`` removes the question rather than
answering it.

# Why circulant embedding rather than shaping noise with a spectrum

Multiplying white noise by a sampled spectrum and inverse-transforming is cheaper, and
it is what this module used to do. Its field is only approximately the target: the
covariance it delivers is the target *wrapped* by the periodic grid and *aliased* by
the discrete spectrum, and nothing in the method reports how large either error is --
the padding that controls the first was a fraction of the fault, which is not a scale
the covariance knows about.

Circulant embedding instead writes the covariance down at the grid's own lags, embeds
it in a circulant matrix whose eigenvalues are that array's DFT, and draws by
multiplying their square roots into complex white noise. Exactness costs a padded grid
at least twice the fault on each axis, and carries one failure mode -- an embedding
whose eigenvalues are not all non-negative is not a covariance matrix, so it raises
rather than sampling something else.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

import numpy as np
from scipy.fft import next_fast_len
from scipy.special import gamma, kv

if TYPE_CHECKING:
    from rupture_generator.mesh import RuptureMesh

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

HURST = 0.75
"""The von Karman roughness exponent, and the only correlation shape left.

Mai & Beroza (2002) figure 11: the median over their 44 finite-source models is 0.75
for the circular average, with 0.71 along strike and 0.77 down dip and no dependence
on magnitude or faulting style. One number for both axes is theirs, not a
simplification made here.

It is the Matern smoothness ``nu``. At ``H = 0.5`` the von Karman correlation is the
exponential one, which is why the paper finds the two fit its data comparably.
"""

MINIMUM_EMBEDDING = 2
"""How many times the fault each padded axis is, before any enlargement.

A Toeplitz matrix of ``n`` lags embeds in a circulant of ``2n - 2``: the first row
carries ``c(0) ... c(n-1)`` and then the mirror ``c(n-2) ... c(1)``. Anything smaller
cannot hold the covariance, whatever fraction it is dressed up as -- which is the
argument the old ten-percent margin did not have.
"""

MAXIMUM_DOUBLINGS = 4
"""How far the embedding may be enlarged before it is refused.

Dietrich & Newsam's recipe is to double until every eigenvalue is non-negative. Four
doublings is a 16-fold grid on each axis, past which the covariance is not one this
grid can carry and saying so beats sampling something else.
"""

EIGENVALUE_TOLERANCE = 1.0e-10
"""How negative an eigenvalue may be, relative to the largest, and still be round-off.

The DFT of a valid covariance is non-negative exactly; what comes back is that
arithmetic in floating point, so the smallest eigenvalues scatter about zero. Ten
orders below the largest is round-off; anything deeper is the embedding failing.
"""


@dataclasses.dataclass(frozen=True)
class VonKarmanFilterParameters:
    """How far a field's structure reaches, and how rough it is.

    Attributes
    ----------
    correlation_length_strike_km, correlation_length_dip_km : float
        The patch size along strike and down dip. The corner of the spectrum is the
        ellipse through the reciprocals of these, so structure larger than them is
        flat and structure smaller falls off.
    hurst : float
        The von Karman roughness exponent. 0.75 is the only value production
        selects, and the same number is a Matern smoothness for the sampler that
        replaces this one.
    """

    correlation_length_strike_km: float
    correlation_length_dip_km: float
    hurst: float = HURST

    def __post_init__(self) -> None:
        """Refuse a spec that cannot describe a field."""
        for name in (
            "correlation_length_strike_km",
            "correlation_length_dip_km",
        ):
            value = getattr(self, name)
            if not (value > 0.0) or not np.isfinite(value):
                raise ValueError(f"{name} must be a positive length, got {value}")
        if not (0.0 < self.hurst < 1.0):
            raise ValueError(f"hurst must be in (0, 1), got {self.hurst}")


def correlation_lengths(
    magnitude: float,
    *,
    strike_offset: float = 2.50,
    dip_offset: float = 1.50,
    strike_exponent: float = 0.5,
    dip_exponent: float = 1.0 / 3.0,
) -> VonKarmanFilterParameters:
    """A corner relation's correlation lengths for a magnitude, in kilometres.

    .. math::

        \\lambda_{strike} = 10^{c_{s} M_w - a}, \\qquad
        \\lambda_{dip}    = 10^{c_{d} M_w - b}

    A corner relation is a straight line in log-length against magnitude, so those
    four numbers are the whole of one: an exponent and an offset per axis. **The
    defaults are Mai & Beroza (2002) equation (5)**, which reads

    .. math::

        \\log(a_{s}) \\approx -2.5 + \\tfrac{1}{2} M_w, \\qquad
        \\log(a_{z}) \\approx -1.5 + \\tfrac{1}{3} M_w

    -- what a config naming ``mai`` takes, where stating four of your own is what a
    config naming ``custom`` does. The exponents are the paper's own ``1/2`` and
    ``1/3``, written as fractions because that is how equation (5) writes them; the
    measured regressions they simplify are in its table 3, at 0.53 and 0.37 for all
    mechanisms together.

    These are a *simplification* the paper offers, and it says so: the same table
    gives coefficients per faulting style, and the scatter about equation (5) is
    ``sigma`` of 0.19 and 0.18 in log-length -- a factor of 1.5. A ``custom`` relation
    is how a caller who wants the strike-slip or dip-slip row states it.

    The three named relations this replaced -- Somerville, Suzuki, Given -- are still
    refused by name in the config rather than re-spelled here as coefficients,
    because a name is a claim output cannot adjudicate: `DEFECTS.md` 11 records that
    Mai and Somerville cross over at M7.37, so a comparison below that magnitude says
    whichever one you started from is right. A reader who has the coefficients and a
    reason can still state them as a ``custom`` relation, where the file says the
    numbers rather than a name that stands in for them.
    """
    return VonKarmanFilterParameters(
        correlation_length_strike_km=10.0
        ** (strike_exponent * magnitude - strike_offset),
        correlation_length_dip_km=10.0 ** (dip_exponent * magnitude - dip_offset),
    )


def von_karman_correlation(
    normalised_distance: FloatArray, hurst: float = HURST
) -> FloatArray:
    """Mai & Beroza (2002) equation (1)'s von Karman ACF, at a dimensionless distance.

    .. math:: C(r) = \\frac{G_H(r)}{G_H(0)} = \\frac{2^{1-H}}{\\Gamma(H)} r^H K_H(r)

    ``G_H(r) = r^H K_H(r)`` with ``K_H`` the modified Bessel function of the second
    kind; the normalisation is its own limit at the origin, ``G_H(0) = 2^{H-1}
    \\Gamma(H)``, which is what makes ``C(0) = 1`` rather than infinite.

    The argument is a **distance in correlation lengths**, not in kilometres -- the
    anisotropy lives in how that distance is formed (see :func:`_wrapped_distance`),
    so this function is one-dimensional and has no axes to confuse.

    ``C(1) = 0.5005`` at ``H = 0.75``: a separation of one correlation length is
    where the field has forgotten about half of itself. That number is the whole
    meaning of ``a``, and it is checkable against this function without running the
    sampler at all.
    """
    distance = np.asarray(normalised_distance, dtype=np.float64)
    correlation = np.ones_like(distance)
    # The limit at r = 0 is 1 and the expression there is 0 * inf, so it is taken
    # rather than evaluated.
    away = distance > 0.0
    scaled = distance[away]
    correlation[away] = (
        2.0 ** (1.0 - hurst) / gamma(hurst) * scaled**hurst * kv(hurst, scaled)
    )
    return correlation


def _wrapped_distance(
    extents: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
) -> FloatArray:
    """Distance from the origin to every cell of the periodic grid, in correlation lengths.

    **Wrapped**, which is what makes the embedding circulant: on a periodic grid of
    ``m`` cells the lag to cell ``p`` is ``min(p, m - p)``, because the short way round
    is the distance. Feeding unwrapped lags here builds a Toeplitz matrix instead, whose
    DFT is not its eigenvalues, and the field that falls out has no particular
    covariance at all.

    Each axis is divided by its own correlation length before the two are combined, so
    the result is the ``r`` of Mai & Beroza equation (1) -- anisotropy enters here and
    nowhere else.
    """
    padded_i, padded_j = extents
    strike_km, dip_km = spacing_km

    down_dip = np.minimum(np.arange(padded_i), padded_i - np.arange(padded_i))
    along_strike = np.minimum(np.arange(padded_j), padded_j - np.arange(padded_j))

    dip_lag = (down_dip * dip_km)[:, None] / parameters.correlation_length_dip_km
    strike_lag = (along_strike * strike_km)[None, :] / (
        parameters.correlation_length_strike_km
    )
    return np.sqrt(dip_lag**2 + strike_lag**2)


@functools.lru_cache(maxsize=32)
def _embed(
    cell_counts: tuple[int, int],
    spacing_km: tuple[float, float],
    parameters: VonKarmanFilterParameters,
) -> tuple[tuple[int, int], FloatArray]:
    """The circulant embedding of this covariance on this grid: extents and eigenvalues.

    The covariance is written down at the padded grid's wrapped lags; its 2-D DFT is
    the circulant matrix's eigenvalues, real by construction because the covariance is
    symmetric. If they are all non-negative the matrix is a covariance matrix and
    sampling from it is exact.

    When they are not, the embedding is enlarged and tried again -- Dietrich & Newsam's
    own recipe. A larger grid pushes the wrap further out, which is what makes the
    negative eigenvalues go away: they are the covariance folding back onto itself.

    Cached, because four fields are drawn per segment against one covariance and the
    Bessel evaluation over the padded grid is the expensive part of this module. The
    key is the whole of what the answer depends on, so two calls that agree on it are
    the same question.

    Returns
    -------
    tuple
        The padded extents ``(i, j)``, and the eigenvalues on that grid.

    Raises
    ------
    ValueError
        If no embedding up to :data:`MAXIMUM_DOUBLINGS` has non-negative eigenvalues.
        Refusing beats clipping: a clipped spectrum is a different covariance, and one
        nothing downstream could notice.
    """
    for doubling in range(MAXIMUM_DOUBLINGS + 1):
        scale = MINIMUM_EMBEDDING * 2**doubling
        extents = (
            int(next_fast_len(scale * cell_counts[0])),
            int(next_fast_len(scale * cell_counts[1])),
        )
        covariance = von_karman_correlation(
            _wrapped_distance(extents, spacing_km, parameters), parameters.hurst
        )
        eigenvalues = np.fft.fft2(covariance).real
        if eigenvalues.min() >= -EIGENVALUE_TOLERANCE * eigenvalues.max():
            # The remaining negatives are round-off about zero, not structure.
            return extents, np.maximum(eigenvalues, 0.0)

    raise ValueError(
        f"a von Karman covariance with correlation lengths "
        f"{parameters.correlation_length_strike_km:.3g} km along strike and "
        f"{parameters.correlation_length_dip_km:.3g} km down dip does not embed on a "
        f"{cell_counts[0]}x{cell_counts[1]} grid at "
        f"{spacing_km[0]:.3g}x{spacing_km[1]:.3g} km, even padded "
        f"{MINIMUM_EMBEDDING * 2**MAXIMUM_DOUBLINGS} times over -- the correlation "
        "length is too large for a fault this size to carry"
    )


def standardise(field: FloatArray) -> FloatArray:
    """Zero mean, unit sample variance.

    Not needed for the variance -- circulant embedding delivers ``C(0) = 1`` by
    construction -- but the stages are written as ``1 + cov * Z`` and mean the sample
    statistics, so this is what makes that arithmetic exact on each realisation rather
    than on average.
    """
    spread = float(field.std())
    # A one-cell chart has a single sample and hence no variance. The mesh CLI produces
    # one for any plane shorter than half the requested subfault size; dividing there
    # gave infinity, then infinity times zero, and an SRF of NaN that nothing refused.
    if spread == 0.0:
        return np.zeros_like(field)
    return (field - field.mean()) / spread


@dataclasses.dataclass(frozen=True, kw_only=True)
class ComplexField:
    """A drawn field in the wavenumber domain, before it is a field on a fault.

    What a later field correlates *against*. Held as the spectrum rather than as the
    realised array because the blend happens here, where the two fields share a set of
    eigenvalues, so what it correlates is the spatial structure rather than the values.

    It is also why this cannot be a cell field on a chart: it lives on the **padded**
    grid, and the crop to the fault is a restriction no inverse can undo. A stage that
    needs one keeps it in a local variable and spends it in the same function.

    Attributes
    ----------
    spectrum : np.ndarray
        Complex, shaped :attr:`extents`: the embedding's square-rooted eigenvalues
        times complex white noise.
    extents : tuple of int
        The padded grid drawn on, ``(i, j)``.
    cell_counts : tuple of int
        The fault's own shape, which :func:`realise_field` crops back to.
    """

    spectrum: np.ndarray
    extents: tuple[int, int]
    cell_counts: tuple[int, int]


def von_karman_field(
    mesh: RuptureMesh,
    covariance: VonKarmanFilterParameters,
    rng: np.random.Generator,
) -> ComplexField:
    """Draw a field with this covariance on this chart, in the wavenumber domain.

    The embedding's eigenvalues carry the structure and the noise carries the
    randomness, so the field is a function of the correlation lengths and the
    generator, and of nothing else.

    The noise is complex with **unit variance in each part** rather than jointly:
    :func:`realise_field` keeps the real part, and halving the variance to make the
    complex draw standard would halve the field's. The imaginary part is a second,
    independent field with the same covariance, and is discarded -- taking it would
    pair two calculations onto one draw, which is exactly the coupling the named
    substreams exist to prevent.
    """
    cell_counts = mesh.cell_counts
    extents, eigenvalues = _embed(cell_counts, mesh.spacing_km(), covariance)

    noise = rng.standard_normal(extents) + 1j * rng.standard_normal(extents)
    return ComplexField(
        spectrum=np.sqrt(eigenvalues) * noise,
        extents=extents,
        cell_counts=cell_counts,
    )


def correlate_fields(
    field_a: ComplexField, field_b: ComplexField, rho: float
) -> ComplexField:
    """A field correlated at ``rho`` with ``field_a``, in the wavenumber domain.

    ``rho * A + sqrt(1 - rho^2) * B``. Both operands have the same eigenvalues and the
    weights are the cosine and sine of one angle, so the result has them too: the
    correlation is set without disturbing the covariance, which is what makes the blend
    composable with whatever rescaling a stage applies afterwards.

    Because the inverse transform is linear and the crop is a restriction, the same
    relation holds **pointwise on the fault**. That identity is what a test should
    assert: a rho of 0.8 implemented as 0.5 is enormous against an identity and under
    one standard error against a sample correlation coefficient.
    """
    if not (-1.0 <= rho <= 1.0):
        raise ValueError(f"a correlation must be in [-1, 1], got {rho}")

    return dataclasses.replace(
        field_a,
        spectrum=(
            rho * field_a.spectrum + np.sqrt(1.0 - rho * rho) * field_b.spectrum
        ),
    )


def realise_field(field: ComplexField) -> FloatArray:
    """The field on the fault: inverse-transform the spectrum and crop.

    The ``sqrt(n)`` undoes numpy's inverse-transform normalisation, so what comes out
    has the covariance the eigenvalues describe rather than that divided by the grid.

    The fault takes the padded grid's **corner**. Every cell of the padded grid is a
    valid sample, so which corner is arbitrary; what matters is that the fault's own
    lags are all shorter than the wrap, which the embedding guarantees.

    Linear in the spectrum, and the crop is a restriction -- which is what lets a
    relation imposed in the wavenumber domain, such as :func:`correlate_fields`' blend,
    hold exactly pointwise on the fault.
    """
    realised = np.fft.ifft2(field.spectrum) * np.sqrt(
        field.extents[0] * field.extents[1]
    )
    cells_i, cells_j = field.cell_counts
    return np.ascontiguousarray(realised.real[:cells_i, :cells_j])


__all__ = [
    "HURST",
    "MAXIMUM_DOUBLINGS",
    "MINIMUM_EMBEDDING",
    "ComplexField",
    "VonKarmanFilterParameters",
    "correlate_fields",
    "correlation_lengths",
    "realise_field",
    "standardise",
    "von_karman_correlation",
    "von_karman_field",
]
