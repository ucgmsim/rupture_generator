"""The one place colour is decided, so no figure picks its own.

Every value here comes from a validated palette rather than from taste, and each colour
does exactly one job:

- **categorical** -- identity, which model a line belongs to. Two series only, curved
  and flat, so slots 1 and 2 and no more. Validated all-pairs on the light surface: CVD
  ``dE`` 24.7 under protanopia, normal-vision ``dE`` 33.6, both contrasts past 3:1. The
  third slot is deliberately absent: it fails the 3:1 contrast check on this surface,
  and the third line one figure carries is distinguished by *frame* -- solid for
  measured on the true interface, dotted for measured on the plane -- rather than by a
  colour that would need a relief channel to be legible.
- **sequential** -- magnitude: slip, onset, ``|grad h|``. One hue, light to dark, so the
  reader sees the order in the lightness rather than having to learn a rainbow. A second
  simultaneous magnitude takes the orange ramp.
- **diverging** -- polarity: every difference between the two models, and the signed
  displacement ``h``. Two hues that read as opposite, with a **neutral grey midpoint**
  and a norm centred at zero, so "no difference" is the colour of nothing. A difference
  drawn on a sequential map would put its zero somewhere arbitrary and make a symmetric
  effect look one-sided, which is the single most misleading thing this study could do.

Light mode only, deliberately: these are PNGs embedded in a document rather than a page
that follows a reader's theme.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from numpy.typing import ArrayLike

CURVED = "#2a78d6"
FLAT = "#eb6834"
"""Categorical slots 1 and 2: the true interface and its best-fit plane."""

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MIDPOINT = "#f0efec"
"""Chart chrome, light mode. The grid and axes are hairlines one shade off the surface,
never dashed -- dashing reads as "projection" when it is only a grid."""

_BLUE_RAMP = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
_ORANGE_RAMP = (
    "#fde3d5",
    "#fbc9b0",
    "#f8ae8b",
    "#f59267",
    "#f17a4a",
    "#eb6834",
    "#d95926",
    "#c04c1f",
    "#a63f19",
    "#8b3314",
    "#70280f",
    "#551d0a",
    "#3b1406",
)
_RED_RAMP = (
    "#fbdcdc",
    "#f6bcbc",
    "#f09b9b",
    "#ea7b7b",
    "#e34948",
    "#d03b3b",
    "#b53232",
    "#992a2a",
    "#7d2222",
    "#611a1a",
)

MAGNITUDE = LinearSegmentedColormap.from_list("magnitude", _BLUE_RAMP)
MAGNITUDE_SECOND = LinearSegmentedColormap.from_list("magnitude_second", _ORANGE_RAMP)
"""Sequential ramps: one hue each, monotone light to dark."""

DIFFERENCE = LinearSegmentedColormap.from_list(
    "difference",
    [*reversed(_BLUE_RAMP[4:]), MIDPOINT, *_RED_RAMP[4:]],
)
"""Diverging blue-to-red through a neutral grey. Warm and cool poles read as opposite;
the grey midpoint reads as nothing, which is what zero has to look like."""


def centred(values: ArrayLike, quantile: float = 99.5) -> TwoSlopeNorm:
    """A diverging norm pinned to zero and clipped to a percentile of the data.

    Two decisions, both to stop one outlier from setting the whole scale. The centre is
    **exactly zero**, never the data's own midpoint, because the colour of zero is the
    only thing a difference map has to get right. The limits are symmetric and read at a
    high percentile of ``|value|`` rather than at its maximum, so the map shows the
    distribution rather than the extreme -- over a million faces the maximum is an order
    statistic of the tail.

    Parameters
    ----------
    values : array_like
        The field being drawn. ``NaN`` is ignored.
    quantile : float, optional
        Which percentile of the absolute value sets the limit.

    Returns
    -------
    TwoSlopeNorm
    """
    finite = np.asarray(values)[np.isfinite(values)]
    limit = float(np.percentile(np.abs(finite), quantile))
    limit = limit if limit > 0.0 else 1.0
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def apply() -> None:
    """Set the rcParams every figure in this study shares.

    Thin marks, hairline recessive grid and axes, generous padding, no top or right
    spine, and text in ink tokens rather than in a series colour -- a colour beside a
    label carries identity, the label itself never does.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": SECONDARY_INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "medium",
            "axes.titlelocation": "left",
            "axes.titlepad": 8,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": SECONDARY_INK,
            "ytick.labelcolor": SECONDARY_INK,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.labelcolor": SECONDARY_INK,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9,
            "text.color": INK,
            "figure.constrained_layout.use": True,
        }
    )
