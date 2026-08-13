"""The measurements: rasterising, correlation against surface separation, and summaries.

Two things here are more than bookkeeping.

**Rasterising onto the shared parameter lattice.** Both models' faces carry the same
``(u, v)``, and the mesh was built by sampling that plane on a regular lattice, so
binning face values back onto a lattice of the same spacing is close to lossless and
gives every map and every correlation estimate one common frame. It is what makes a
difference map a difference rather than an interpolation artefact.

**Correlation against *surface* separation, which is the whole point of measurement 3.**
The SPDE operator is assembled from the lifted triangles, so the curved model's field has
its correlation length in true surface distance. The flat model's field has the same
length in *parameter* distance -- and parameter distance is shorter than surface distance
by the metric factor ``sqrt(1 + |grad h|^2)``. So projecting the flat model's slip onto
the real interface delivers a correlation length **inflated** by that factor, and the
inflation is local: it is largest exactly where the interface bends most.

Measuring that needs separation in kilometres *along the surface*, not in parameter
kilometres. On the raster that is exact rather than approximate: a row at fixed ``v`` is a
polyline in the ``(u, h)`` plane, and the true along-strike arc between two of its samples
is the sum of the segment lengths ``sqrt(du^2 + dh^2)``. :func:`correlation_profile` walks
that arc and bins by it, so the flat and curved models are compared at equal *physical*
separation, which is the comparison a practitioner cares about.
"""

from __future__ import annotations

import numpy as np

from rupture_generator.sampling import HURST, von_karman_correlation

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

HALF_CORRELATION = float(von_karman_correlation(np.array([1.0]), HURST)[0])
"""``C(1) = 0.5005`` at ``H = 0.75``: what the von Karman ACF is at one correlation
length. The whole meaning of the correlation length ``a``, and therefore the level whose
crossing :func:`delivered_length_km` reads a *delivered* length off."""

MAXIMUM_LAG_KM = 160.0
"""How far the correlation profiles reach, in kilometres.

Just under three times the 56.2 km along-strike correlation length, which is far enough
to see the profile flatten and short enough that the estimate at the far end still
averages over most of the fault.
"""


def rasterise(
    parameters_uv_km: FloatArray,
    values: FloatArray,
    spacing_km: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Bin per-face values onto a regular lattice in the parameter plane.

    The mean of the faces falling in each cell, and ``NaN`` where no face does -- which
    is the fault's own outline, since the parameter domain is not a rectangle.

    Parameters
    ----------
    parameters_uv_km : FloatArray
        ``(F, 2)`` face centres in ``(u, v)``. Shared between the models, which is why
        one raster geometry serves both.
    values : FloatArray
        ``(F,)`` the field to bin.
    spacing_km : float
        The cell size. Matching the mesh's own parameter spacing keeps this close to
        lossless; coarser would smooth the field being measured.

    Returns
    -------
    tuple of FloatArray
        The ``(n_v, n_u)`` grid, and the cell-centre ``u`` and ``v`` axes in kilometres.
    """
    origin = parameters_uv_km.min(axis=0)
    index = np.floor((parameters_uv_km - origin) / spacing_km).astype(np.int64)
    shape = (int(index[:, 1].max()) + 1, int(index[:, 0].max()) + 1)
    flat = index[:, 1] * shape[1] + index[:, 0]

    total = np.bincount(flat, weights=values, minlength=shape[0] * shape[1])
    count = np.bincount(flat, minlength=shape[0] * shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return (
        grid.reshape(shape),
        origin[0] + (np.arange(shape[1]) + 0.5) * spacing_km,
        origin[1] + (np.arange(shape[0]) + 0.5) * spacing_km,
    )


def _arc_lengths(
    height_km: FloatArray, spacing_km: float
) -> tuple[FloatArray, FloatArray]:
    """Cumulative surface arc length along the last axis of a raster of ``h``.

    A row at fixed ``v`` is the curve ``u -> (u, h(u))`` in the plane spanned by the
    parameter axis and the frame normal, so its length between two samples is the sum of
    ``sqrt(du^2 + dh^2)``. That is the true along-axis surface distance, and it is what
    the flat model's parameter distance is missing.

    **Gaps are counted, not propagated.** The fault's parameter footprint is not a
    rectangle, so rows have holes, and a pair whose two members straddle a hole has no
    arc length between them. The obvious implementation -- carry the gap as ``NaN`` into
    a cumulative sum -- also destroys every pair *after* the hole, which on this outline
    silently discards most of the long-lag statistics and biases the profile towards the
    start of each row. Instead the gap steps contribute zero to the running length and
    are counted separately, so a pair is rejected exactly when a gap lies between its
    members.

    Parameters
    ----------
    height_km : FloatArray
        ``(..., n)`` the normal displacement, ``NaN`` outside the fault.
    spacing_km : float
        The parameter spacing between samples.

    Returns
    -------
    tuple of FloatArray
        ``(..., n)`` cumulative arc length, and ``(..., n)`` how many gap steps have
        been crossed to reach each sample.
    """
    step = np.sqrt(spacing_km**2 + np.diff(height_km, axis=-1) ** 2)
    missing = ~np.isfinite(step)
    leading = np.zeros(height_km.shape[:-1] + (1,))
    return (
        np.concatenate(
            [leading, np.cumsum(np.where(missing, 0.0, step), axis=-1)], axis=-1
        ),
        np.concatenate([leading, np.cumsum(missing, axis=-1)], axis=-1),
    )


def correlation_profile(
    field: FloatArray,
    height_km: FloatArray | None,
    spacing_km: float,
    *,
    axis: int,
    maximum_lag_km: float = MAXIMUM_LAG_KM,
    bin_km: float | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Empirical correlation against **surface** separation along one parameter axis.

    The field is centred and scaled by its own valid-cell statistics first, so the
    profile starts at 1 and the estimate is a correlation rather than a covariance.
    Every ordered pair at every lag contributes, binned by the true surface distance
    between its two members.

    Parameters
    ----------
    field : FloatArray
        ``(n_v, n_u)`` raster from :func:`rasterise`, ``NaN`` outside the fault.
    height_km : FloatArray or None
        ``(n_v, n_u)`` the normal displacement of **the surface the separation is
        measured on**, which is not always the surface the field was generated on.
        ``None`` means the plane, where surface separation and parameter separation are
        the same thing -- stated by passing ``None`` rather than by passing zeros, so
        the two cases cannot be confused.

        The three combinations this study uses are the whole of measurement 3. A curved
        field on the curved surface is what the SPDE promised. A flat field on the plane
        is what a practitioner believes they have. **A flat field on the curved
        surface** is what they actually have once the slip is projected, and it is the
        only one of the three that can differ from the model.
    spacing_km : float
        The raster spacing.
    axis : int
        0 for down dip, 1 for along strike.
    maximum_lag_km : float, optional
        How far to reach.
    bin_km : float, optional
        Separation bin width. Defaults to the raster spacing, which is the finest bin
        that cannot be empty.

    Returns
    -------
    tuple of FloatArray
        Bin-centre separation in kilometres, the correlation there, and how many pairs
        each bin averaged.
    """
    bin_km = bin_km or spacing_km
    if axis == 0:
        field = field.T
        height_km = None if height_km is None else height_km.T

    valid = np.isfinite(field)
    centred = np.where(valid, field, 0.0)
    mean = centred.sum() / valid.sum()
    centred = np.where(valid, field - mean, 0.0)
    variance = float((centred**2).sum() / valid.sum())

    arc, gaps = (
        (None, None)
        if height_km is None
        else _arc_lengths(np.where(valid, height_km, np.nan), spacing_km)
    )

    bins = int(np.ceil(maximum_lag_km / bin_km)) + 1
    total = np.zeros(bins)
    count = np.zeros(bins)
    for lag in range(1, field.shape[1]):
        # The arc is longer than the parameter lag, so the loop runs past the lag the
        # cap would suggest and the cap is applied to the separation itself.
        if lag * spacing_km > maximum_lag_km:
            break
        pair = valid[:, lag:] & valid[:, :-lag]
        if not pair.any():
            continue
        if arc is None:
            separation = np.full(pair.shape, lag * spacing_km)
            unbroken = pair
        else:
            separation = arc[:, lag:] - arc[:, :-lag]
            unbroken = pair & (gaps[:, lag:] == gaps[:, :-lag])
        usable = unbroken & np.isfinite(separation) & (separation <= maximum_lag_km)
        if not usable.any():
            continue
        index = np.floor(separation[usable] / bin_km).astype(np.int64)
        product = (centred[:, lag:] * centred[:, :-lag])[usable]
        total += np.bincount(index, weights=product, minlength=bins)[:bins]
        count += np.bincount(index, minlength=bins)[:bins]

    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.where(
            count > 0, total / np.maximum(count, 1) / variance, np.nan
        )
    return (np.arange(bins) + 0.5) * bin_km, correlation, count


def delivered_length_km(
    separation_km: FloatArray,
    correlation: FloatArray,
    level: float = HALF_CORRELATION,
) -> float:
    """Where a measured correlation profile first falls through ``C(1)``.

    The correlation length of a von Karman field is defined by ``C(a) = 0.5005``, so
    reading the crossing off an empirical profile gives the length the sampler
    **delivered** rather than the one it was asked for. Linear between the two samples
    that straddle it.

    Parameters
    ----------
    separation_km, correlation : FloatArray
        From :func:`correlation_profile`.
    level : float, optional
        The level to cross. Defaults to :data:`HALF_CORRELATION`.

    Returns
    -------
    float
        Kilometres, or ``nan`` if the profile never falls that far.
    """
    finite = np.isfinite(correlation)
    distance, value = separation_km[finite], correlation[finite]
    below = np.flatnonzero(value < level)
    if not below.size or below[0] == 0:
        return float("nan")
    after = int(below[0])
    before = after - 1
    span = value[before] - value[after]
    if span <= 0.0:
        return float(distance[after])
    weight = (value[before] - level) / span
    return float(distance[before] + weight * (distance[after] - distance[before]))


def spread(values: FloatArray, name: str, unit: str) -> dict:
    """A distribution as the handful of numbers a document quotes.

    Parameters
    ----------
    values : FloatArray
        The samples.
    name : str
        What the quantity is, for the returned keys.
    unit : str
        Its unit, which goes into every key so a number cannot be quoted without one.

    Returns
    -------
    dict
        Mean, median, the 10th, 90th, 1st and 99th percentiles, both extremes, and the
        median of the absolute value -- which is the one that does not cancel when a
        signed quantity is symmetric about zero.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    percentiles = np.percentile(values, [1.0, 10.0, 50.0, 90.0, 99.0])
    return {
        f"{name}_mean_{unit}": float(values.mean()),
        f"{name}_median_{unit}": float(percentiles[2]),
        f"{name}_p1_{unit}": float(percentiles[0]),
        f"{name}_p10_{unit}": float(percentiles[1]),
        f"{name}_p90_{unit}": float(percentiles[3]),
        f"{name}_p99_{unit}": float(percentiles[4]),
        f"{name}_min_{unit}": float(values.min()),
        f"{name}_max_{unit}": float(values.max()),
        f"{name}_median_absolute_{unit}": float(np.median(np.abs(values))),
        f"{name}_max_absolute_{unit}": float(np.abs(values).max()),
    }


def weighted_spread(
    values: FloatArray, weights: FloatArray, name: str, unit: str
) -> dict:
    """The same summary weighted by area, for a quantity whose faces are unequal.

    A per-face median over a mesh whose faces differ in area by 1.5 is a median over
    *faces*, not over the fault. Where the distinction matters the area-weighted form is
    reported alongside.

    Parameters
    ----------
    values, weights : FloatArray
        ``(F,)`` the samples and the area each carries.
    name, unit : str
        As :func:`spread`.

    Returns
    -------
    dict
        The area-weighted mean and the area-weighted median.
    """
    order = np.argsort(values)
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    half = 0.5 * cumulative[-1]
    return {
        f"{name}_area_weighted_mean_{unit}": float(np.average(values, weights=weights)),
        f"{name}_area_weighted_median_{unit}": float(
            values[order][int(np.searchsorted(cumulative, half))]
        ),
    }


def pearson(first: FloatArray, second: FloatArray) -> float:
    """The correlation between two fields defined on the same faces.

    Meaningful here precisely because the two models share their white noise: without
    that, this would be a statistic about two independent realisations and would sit
    near zero whatever the geometry did.
    """
    return float(
        np.corrcoef(np.asarray(first).ravel(), np.asarray(second).ravel())[0, 1]
    )


def corner_frequency_hz(frequency_hz: FloatArray, amplitude_nm: FloatArray) -> float:
    """Where the amplitude spectrum has fallen to half its zero-frequency value.

    An omega-squared model would put the corner where the spectrum is ``1/sqrt(2)`` of
    the plateau, but a measured moment rate spectrum of a single realisation is not an
    omega-squared model and fitting one would import an assumption the comparison does
    not need. The half-power point is a *descriptive* statistic of the curve, read the
    same way for both models, which is all a difference between them requires.

    Parameters
    ----------
    frequency_hz, amplitude_nm : FloatArray
        From :func:`~curvature.model.amplitude_spectrum`.

    Returns
    -------
    float
        Hertz.
    """
    plateau = float(amplitude_nm[0])
    below = np.flatnonzero(amplitude_nm < 0.5 * plateau)
    if not below.size:
        return float("nan")
    after = int(below[0])
    before = max(after - 1, 0)
    if after == before:
        return float(frequency_hz[after])
    span = amplitude_nm[before] - amplitude_nm[after]
    weight = (amplitude_nm[before] - 0.5 * plateau) / span if span > 0 else 0.0
    return float(
        frequency_hz[before] + weight * (frequency_hz[after] - frequency_hz[before])
    )


def high_frequency_slope(
    frequency_hz: FloatArray,
    amplitude_nm: FloatArray,
    band_hz: tuple[float, float],
) -> float:
    """The log-log falloff exponent of a spectrum over one band.

    Fitted by least squares to ``log|M-dot|`` against ``log f``, so -2 is the
    omega-squared falloff. Quoted with the band it was measured over, because a single
    realisation's spectrum is not a straight line and the number depends on where it is
    read.

    Parameters
    ----------
    frequency_hz, amplitude_nm : FloatArray
    band_hz : tuple of float
        The low and high edges, in hertz.

    Returns
    -------
    float
        The exponent.
    """
    inside = (
        (frequency_hz >= band_hz[0])
        & (frequency_hz <= band_hz[1])
        & (amplitude_nm > 0.0)
    )
    return float(
        np.polyfit(np.log10(frequency_hz[inside]), np.log10(amplitude_nm[inside]), 1)[0]
    )
