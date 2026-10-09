"""The von Karman correlation function."""

import dataclasses

import numpy as np
import scipy as sp

from rupture_generator.errors import RuptureGeneratorError

HURST = 0.75
"""Mai & Beroza (2002) figure 11: the median Hurst exponent over 44 finite-source
models, 0.71 along strike and 0.77 down dip."""


@dataclasses.dataclass(frozen=True)
class VonKarman:
    """Mai & Beroza (2002) equation (1), at a lag in correlation lengths.

    ``C(r) = 2^(1-H) / Gamma(H) * r^H * K_H(r)``, with ``K_H`` the modified Bessel
    function of the second kind. ``C(1) = 0.5005`` at ``H = 0.75``.

    Attributes
    ----------
    hurst : float
        The Hurst exponent ``H``, in ``(0, 1)``.
    """

    hurst: float = HURST

    def __post_init__(self) -> None:
        """Refuse a Hurst exponent outside ``(0, 1)``."""
        if not 0.0 < self.hurst < 1.0:
            raise RuptureGeneratorError(f"hurst must be in (0, 1), got {self.hurst}")

    def __call__(self, lag: np.ndarray) -> np.ndarray:
        """Evaluate the correlation at a lag in correlation lengths.

        At zero lag the formula reads 0/0, and the correlation takes its limit, 1.

        Parameters
        ----------
        lag : np.ndarray
            Lags in correlation lengths, of any shape.

        Returns
        -------
        np.ndarray
            The correlation at each lag.
        """
        lag = np.asarray(lag, dtype=np.float64)
        correlation = np.ones_like(lag)
        away = lag > 0.0
        scaled = lag[away]
        away_correlation = sp.special.kv(self.hurst, scaled)
        away_correlation *= scaled**self.hurst
        away_correlation *= 2.0 ** (1.0 - self.hurst) / sp.special.gamma(self.hurst)
        correlation[away] = away_correlation
        return correlation
