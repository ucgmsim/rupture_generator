"""Correlated Gaussian random fields on a chart, and the seam where they come from.

A slip distribution is not white noise: it has patches, and how big they are is a
function of magnitude. :class:`CovarianceSpec` says how big; a :class:`FieldSampler`
turns that into a field.

# Two samplers, one configuration

`SpectralSampler` shapes white noise by the square root of a power spectrum and
inverse-transforms it. That needs a **regular grid**, which is why S3 exists to assert
one, and it is the whole reason the temporary flatness constraint is temporary: a
`KernelSampler` sampling the same covariance on an arbitrary point set is the curved
geometry path, and `CovarianceSpec`'s correlation lengths map straight onto Matern
length scales (smoothness ``nu = hurst``). The *configuration* is already
sampler-independent; only the mechanism swaps.

Correlation between fields happens **inside** the sampler, in
:meth:`FieldSampler.correlated_with`, because the mechanism is sampler-specific: the
spectral sampler blends in the wavenumber domain, where the two fields share a
spectrum, so what gets correlated is the spatial *structure*. Blending after the
inverse transform would correlate the values and leave each field's structure
untouched -- a different model, and the wrong one. The pipeline only ever says "give
me a field at correlation rho with that one".

# What is private here

Padding, even extents, Nyquist rows, Hermitian symmetry: all of it is
`SpectralSampler`'s own business and nothing outside this module may know it exists.
The mesh does not, the stages do not, and the config does not.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from rupture_generator.mesh import RuptureMesh

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

HURST = 0.75
"""The von Karman roughness exponent, and the only spectral shape left.

Mai & Beroza (2002)'s own falloff, which is what the corner relation production
selects takes. The 2-D power spectrum is ``(1 + a) ** -(H + 1)``; the field is built
in *amplitude*, so the exponent is halved on the way in.
"""

BAND_PASS_ORDER = 4
"""How sharply the high-wavenumber roll-off cuts. genslip's own literal, and it stays
one: the parameter that could vary it fed a field nothing constructs."""

NYQUIST_FRACTION = 0.8
"""Where the roll-off sits, as a fraction of the grid's Nyquist wavenumber.

The shortest wavelength a grid can carry is twice its spacing; rolling off at 80% of
that wavenumber rather than at it keeps the last resolvable octave from being a step.
"""


@dataclasses.dataclass(frozen=True)
class CovarianceSpec:
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
            if not (value > 0.0) or not math.isfinite(value):
                raise ValueError(f"{name} must be a positive length, got {value}")
        if not (0.0 < self.hurst < 1.0):
            raise ValueError(f"hurst must be in (0, 1), got {self.hurst}")


def correlation_lengths(
    magnitude: float, *, strike_offset: float = 2.50, dip_offset: float = 1.50
) -> CovarianceSpec:
    """Mai & Beroza (2002)'s correlation lengths for a magnitude.

    .. math::

        \\lambda_{strike} = 10^{0.5 M_w - a}, \\qquad
        \\lambda_{dip}    = 10^{0.3333 M_w - b}

    in kilometres. **0.3333, not one third**: the difference is in the fourth decimal
    of the exponent, which at M8 is about a percent of the corner, and the literal is
    what the relation was fitted and published with.

    The three relations this replaced -- Somerville, Suzuki, Given -- are refused by
    name in the config, because output cannot adjudicate between them: `DEFECTS.md`
    11 records that Mai and Somerville cross over at M7.37, so a comparison below
    that magnitude says whichever one you started from is right.
    """
    return CovarianceSpec(
        correlation_length_strike_km=10.0 ** (0.5 * magnitude - strike_offset),
        correlation_length_dip_km=10.0 ** (0.3333 * magnitude - dip_offset),
    )


class Reference(Protocol):
    """A drawn field's internal state, for correlating a later field against it.

    Opaque on purpose. What it holds is sampler-specific -- the spectral sampler
    keeps a wavenumber-domain array, a kernel sampler would keep the draw itself --
    and the pipeline only ever passes it back.
    """


class FieldSampler(Protocol):
    """Something that can draw standardised Gaussian fields on a chart.

    Three methods rather than two, because the pipeline needs **three** fields
    correlated against one reference: rise time at 0.9 with slip, the onset
    perturbation at 0.8 with slip, and rake at nothing. A `correlated_pair` that
    drew a fresh reference each time would correlate each field with a different
    slip distribution than the one the rupture actually has -- which looks right in
    every marginal statistic and is wrong in exactly the way that matters.
    """

    def sample(
        self, mesh: RuptureMesh, covariance: CovarianceSpec, rng: np.random.Generator
    ) -> FloatArray:
        """A zero-mean, unit-variance field with the given covariance, on ``(i, j)``."""
        ...

    def sample_with_reference(
        self, mesh: RuptureMesh, covariance: CovarianceSpec, rng: np.random.Generator
    ) -> tuple[FloatArray, Reference]:
        """The same, plus the handle later fields correlate against."""
        ...

    def correlated_with(
        self,
        mesh: RuptureMesh,
        covariance: CovarianceSpec,
        reference: Reference,
        rho: float,
        rng: np.random.Generator,
    ) -> FloatArray:
        """A field correlated at ``rho`` with the field that produced ``reference``."""
        ...


@dataclasses.dataclass(frozen=True)
class SpectralReference:
    """A drawn field's shaped, symmetrised spectrum, and the field itself.

    What :meth:`SpectralSampler.correlated_with` blends against. Holding the
    spectrum rather than the field is the point: the blend happens in the wavenumber
    domain, where the two fields share a spectrum, so what it correlates is the
    spatial structure rather than the values.
    """

    spectrum: np.ndarray
    field: FloatArray


class SpectralSampler:
    """Fields by shaping white noise with a spectrum and inverse-transforming.

    Everything about the padded grid is private to this class. The output is always
    zero-mean and unit-variance over the fault, so a caller that wants a mean or a
    spread applies its own -- which is what makes the stages' arithmetic readable
    (`1 + cov * Z`, rather than a chain of divides and rescales).
    """

    def _padded_extents(self, cells_i: int, cells_j: int) -> tuple[int, int]:
        """The grid to transform on: the fault plus a wraparound margin, made even.

        The DFT is periodic, so structure running off one end of the fault reappears
        at the other; a margin absorbs it and the fault takes the padded grid's
        **corner**. Even because the symmetrisation addresses the Nyquist row and
        column directly.

        genslip's rule is ``even(floor(1.10 n))``, which for every extent below ten
        -- and for exactly 4 and 8 -- returns ``n`` itself: **no margin at all**.
        That is reproduced faithfully by the port and is not defensible, so this
        takes the larger of the ten-percent margin and two cells. A deliberate change
        of behaviour, not a port detail: `PLAN.md` section 2 makes output free to
        move where a property says it should.
        """

        def padded(count: int) -> int:
            wanted = max(count + 2, math.ceil(1.10 * count))
            return wanted + 1 if wanted % 2 else wanted

        return padded(cells_i), padded(cells_j)

    def _envelope(
        self,
        extents: tuple[int, int],
        spacing_km: tuple[float, float],
        covariance: CovarianceSpec,
    ) -> FloatArray:
        """The deterministic amplitude at each wavenumber: spectrum times band-pass.

        Wavenumbers are in **cycles** per kilometre, which is what ``fftfreq``
        returns and the only convention in which ``k * lambda`` is dimensionless.
        """
        padded_i, padded_j = extents
        strike_km, dip_km = spacing_km

        # `i` is down dip and `j` along strike, so the row frequencies are the dip
        # ones. The outer product below carries that.
        k_dip = np.fft.fftfreq(padded_i, d=dip_km)[:, None]
        k_strike = np.fft.fftfreq(padded_j, d=strike_km)[None, :]

        normalised = (k_strike * covariance.correlation_length_strike_km) ** 2 + (
            k_dip * covariance.correlation_length_dip_km
        ) ** 2
        # The square root of the power spectrum: the field is built in amplitude.
        # von Karman is a smooth roll-off, not a hard corner -- the power is 1 at the
        # origin, `2 ** -(H+1)` on the corner ellipse, and `a ** -(H+1)` far above.
        shape = (1.0 + normalised) ** (-(covariance.hurst + 1.0) / 2.0)

        # The band-pass keeps the field off the grid's own resolution limit, whose
        # wavelength is derived here and never written down as a constant: no
        # constant is right on two grids, and a fixed band was one of the four wrong
        # numbers -- measured at 80% per-subfault slip differences on the shipped
        # example, because band-limiting a spectrum genslip leaves alone is a
        # different earthquake with the same moment.
        shortest_wavelength_km = 2.0 * math.sqrt(strike_km * dip_km) / NYQUIST_FRACTION
        squared = k_strike**2 + k_dip**2
        high_cut = 1.0 + (squared * shortest_wavelength_km**2) ** BAND_PASS_ORDER

        # genslip pairs this with a low-wavenumber cut whose limit it then assigns a
        # value so large the term is exactly 1 everywhere. It is not reproduced: an
        # inert factor that underflows if anyone ever gives it a real limit is worse
        # than no factor. Structure longer than the fault is removed by the crop.
        return shape / high_cut

    def _noise(self, extents: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        """Unit-variance complex white noise over the padded grid.

        Split evenly between the real and imaginary parts, so ``E|W|^2 = 1``.
        """
        real = rng.standard_normal(extents)
        imaginary = rng.standard_normal(extents)
        return (real + 1j * imaginary) / np.sqrt(2.0)

    def _symmetrise(self, spectrum: np.ndarray) -> np.ndarray:
        """Impose ``F(-k) = conj(F(k))``, so the inverse transform is real.

        The four self-conjugate points -- the origin and the three Nyquist corners --
        map to themselves under conjugation, so they carry one real degree of freedom
        rather than two. The quadrature split took half their variance, and this
        gives it back before discarding their imaginary parts; without it the field's
        longest-wavelength components are systematically weak.

        Afterwards exactly one real degree of freedom survives per grid point, which
        is what a real field of that many points must have.
        """
        padded_i, padded_j = spectrum.shape
        half_i, half_j = padded_i // 2, padded_j // 2

        spectrum = spectrum.copy()
        for point in ((0, 0), (0, half_j), (half_i, 0), (half_i, half_j)):
            spectrum[point] = complex(spectrum[point].real * np.sqrt(2.0), 0.0)

        # Reflect the whole grid through the origin, then keep the half that was
        # drawn and take the conjugate reflection for the half that was not. The
        # kept half is rows 1..half_i-1 entire, plus rows 0 and half_i up to their
        # own Nyquist column.
        mirrored = np.conj(
            spectrum[
                np.ix_(
                    (-np.arange(padded_i)) % padded_i, (-np.arange(padded_j)) % padded_j
                )
            ]
        )
        rows = np.arange(padded_i)[:, None]
        columns = np.arange(padded_j)[None, :]
        drawn = ((rows >= 1) & (rows < half_i)) | (
            ((rows == 0) | (rows == half_i)) & (columns <= half_j)
        )
        return np.where(drawn, spectrum, mirrored)

    def _to_fault(
        self, spectrum: np.ndarray, cell_counts: tuple[int, int]
    ) -> FloatArray:
        """Inverse-transform, take the fault's corner of the padded grid.

        No normalisation constant: genslip's forward and inverse spacing factors --
        a discrete approximation to a continuous Fourier integral -- multiply out to
        exactly one, and the caller standardises anyway.
        """
        field = np.fft.ifft2(spectrum)
        largest = np.abs(field.real).max()
        if largest > 0.0 and np.abs(field.imag).max() > 1.0e-9 * largest:
            raise AssertionError(
                "the inverse transform is not real, so the spectrum was not "
                "Hermitian -- this is a bug in the symmetrisation, not in any input"
            )
        cells_i, cells_j = cell_counts
        return np.ascontiguousarray(field.real[:cells_i, :cells_j])

    def _standardise(self, field: FloatArray) -> FloatArray:
        """Zero mean, unit population variance -- the sampler's output contract.

        A **one-cell chart** has a field whose single value is its own mean, and the
        mesh CLI produces one for any plane shorter than half the requested subfault
        size. Dividing there gave infinity, then infinity times zero, and a whole SRF
        of NaN slip written with no error raised anywhere. A constant field has no
        structure to scale, so the answer is the zero field.
        """
        spread = float(field.std())
        if spread == 0.0:
            return np.zeros_like(field)
        return (field - field.mean()) / spread

    def _draw(
        self,
        mesh: RuptureMesh,
        covariance: CovarianceSpec,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """White noise shaped by the envelope, on the padded grid."""
        cells_i, cells_j = mesh.cell_counts
        extents = self._padded_extents(cells_i, cells_j)
        envelope = self._envelope(extents, mesh.spacing_km(), covariance)
        return self._noise(extents, rng) * envelope, (cells_i, cells_j)

    def sample(
        self, mesh: RuptureMesh, covariance: CovarianceSpec, rng: np.random.Generator
    ) -> FloatArray:
        """One standardised field on the chart's cells."""
        return self.sample_with_reference(mesh, covariance, rng)[0]

    def sample_with_reference(
        self, mesh: RuptureMesh, covariance: CovarianceSpec, rng: np.random.Generator
    ) -> tuple[FloatArray, SpectralReference]:
        """A field, and the shaped spectrum later fields blend against."""
        spectrum, cell_counts = self._draw(mesh, covariance, rng)
        symmetric = self._symmetrise(spectrum)
        field = self._to_fault(symmetric, cell_counts)
        return self._standardise(field), SpectralReference(symmetric, field)

    def correlated_with(
        self,
        mesh: RuptureMesh,
        covariance: CovarianceSpec,
        reference: SpectralReference,
        rho: float,
        rng: np.random.Generator,
    ) -> FloatArray:
        """A standardised field correlated at ``rho`` with the reference's field.

        The blend is ``rho * H_ref + sqrt(1 - rho^2) * H_new`` in the **wavenumber**
        domain. Both operands have unit variance and the weights are the cosine and
        sine of one angle, so the result does too: the correlation is set without
        disturbing the amplitude, which is what makes the blend composable with
        whatever rescaling a stage applies afterwards.

        Because the inverse transform is linear and the crop is a restriction, the
        same relation holds **pointwise on the fault** before standardisation. That
        identity is what a test should assert -- a rho of 0.8 implemented as 0.5 is
        enormous against an identity and under one standard error against a sample
        correlation coefficient -- and :meth:`blend_on_fault` exposes it.
        """
        return self._standardise(
            self.blend_on_fault(mesh, covariance, reference, rho, rng)[0]
        )

    def blend_on_fault(
        self,
        mesh: RuptureMesh,
        covariance: CovarianceSpec,
        reference: SpectralReference,
        rho: float,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, FloatArray]:
        """The blended field and its independent part, **before** standardising.

        Where the blend identity is still exact: standardising divides each field by
        its own sample spread, which perturbs the relation by the estimator error.
        """
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"a correlation must be in [-1, 1], got {rho}")

        spectrum, cell_counts = self._draw(mesh, covariance, rng)
        independent = self._symmetrise(spectrum)
        blended = rho * reference.spectrum + math.sqrt(1.0 - rho * rho) * independent
        return (
            self._to_fault(blended, cell_counts),
            self._to_fault(independent, cell_counts),
        )


__all__ = [
    "BAND_PASS_ORDER",
    "HURST",
    "NYQUIST_FRACTION",
    "CovarianceSpec",
    "FieldSampler",
    "SpectralSampler",
    "correlation_lengths",
]
