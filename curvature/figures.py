"""Every figure the document embeds. Run as ``uv run --with matplotlib python -m curvature.figures``.

Reads ``curvature/data/*.npz`` and ``curvature/results.json``, writes
``curvature/figures/*.png``. It computes nothing the study depends on -- if a number
appears on a figure it also appears in ``results.json``.

One :class:`Plate` per interface **per magnitude**, and every figure takes one.
Hikurangi at :data:`~curvature.model.MAGNITUDE` carries no prefix, so it writes the same
filenames it always did and the published document's image paths keep resolving; each
Puysegur surface writes ``<interface>_<figure>.png`` beside them, and each further
magnitude writes ``<interface>_mw<magnitude>_<figure>.png``. Nothing about a figure is
per-plate except that prefix, the title and the numbers, which is the point of running
the same analysis on a second surface and at a second magnitude: two figures that differ
only in their numbers can be compared by eye.

Three conventions hold across all of them.

**The fault-plane view is the default frame.** Maps are drawn in the parameter plane,
``u`` along strike against ``v`` down dip with depth increasing downward. That is the
frame both models share exactly, so a difference map is a difference rather than an
interpolation artefact, and it is how slip models are conventionally shown. One figure
is in map view instead, and says so.

**Magnitudes take a one-hue sequential ramp; differences take a diverging ramp centred
at zero.** A difference drawn on a sequential map puts its zero at an arbitrary colour
and makes a symmetric effect look one-sided. See :mod:`curvature.style`.

**Dense fields are binned, never overplotted.** There are 1.39 million faces; a scatter
of them is a solid block that hides its own distribution and costs megabytes. Every map
here is a raster of cell means and the polar figures are polar heatmaps.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, ListedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from curvature import model, run, style
from curvature.geometry import HIKURANGI, PUYSEGUR, PUYSEGUR_FIORDLAND, build_pair

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

SCENARIO_LABELS = {
    "central_constant": "Central, constant velocity",
    "central_standard": "Central, standard 1-D",
    "northern_standard": "Northern, standard 1-D",
    "southern_standard": "Southern, standard 1-D",
}
"""The four rows :func:`onset_polar` draws, and the order of its two-by-two grid.

Deliberately not the whole run matrix. Every interface now runs six rows, and adding the
other two here would reflow a published figure to make room for them; they have
:func:`decomposition_by_site`, which is built around the comparison they support.
"""

SITES = ("northern", "central", "southern")
"""Along strike, in the order the parameter coordinate runs. See
:data:`curvature.run.STRIKE_FRACTIONS` for what the names mean geographically -- they are
positions in the parameter plane and the two interfaces run opposite ways round."""

TRUE_DEPTH_LABELS = {
    "central_constant": "Geometry alone\n(constant velocity)",
    "central_standard_truedepth": "Geometry + true-depth materials\n(what the refactor buys)",
    "central_standard": "Geometry + flat-depth materials\n(the status quo)",
}
"""The three-way decomposition, in the order the argument runs.

The keys are two scenario names and one true-depth scenario name, so :func:`_entry` has to
know where each lives; they are kept in one place because the *order* is the figure's
argument and reordering it would reverse the reading.
"""

REFERENCE_CORNER_HZ = 0.01
"""Roughly where an Mw 8.5 whole-interface rupture corners, for the spectrum's marker.

A round order of magnitude rather than a fitted number -- it is drawn to say which part
of the spectrum is the source and which is the discretisation, and the corner the run
actually delivered is measured and reported in ``results.json``. At another magnitude
:func:`_reference_corner_hz` moves it by self-similarity, since a marker left at an
Mw 8.5 corner on an Mw 9.11 spectrum would be reading the wrong decade.
"""

MINIMUM_SECTION_EXTENT_KM = 100.0
"""How far down dip a column must reach before :func:`sections` will draw it.

Without a floor the "flattest" column is a short sliver at the mesh edge, whose departure
from the plane is small only because it does not reach far enough down dip to depart. One
absolute length rather than a fraction of each interface's own extent, so the three
profiles on one figure and the profiles on two interfaces are the same measurement.
"""


@dataclasses.dataclass(frozen=True)
class Plate:
    """One interface's arrays, numbers and filename prefix.

    Attributes
    ----------
    arrays : dict
        Its ``data/*.npz``.
    results : dict
        Its groups out of ``results.json``: ``geometry``, ``scenarios``, ``moment``,
        ``correlation``, ``true_depth`` and, for Hikurangi only, ``resolution``.
    prefix : str
        Prepended to every filename, from :func:`curvature.run.prefix`. Empty for
        Hikurangi at the study's own magnitude, whose figures are already published
        under bare names.
    path : Path
        The surface, for the one figure that needs the mesh rather than a raster.
    label : str
        What the interface is called in a title. A plate at any magnitude but the
        study's own says which, since two magnitudes' figures are otherwise identical
        drawings of different numbers.
    magnitude : float
        The event these numbers are for. Read by the one figure that marks a reference
        frequency the magnitude sets; everything else on every figure is measured.
    """

    arrays: dict
    results: dict
    prefix: str
    path: Path
    label: str
    magnitude: float

    def figure(self, name: str) -> Path:
        """Where a figure of this name belongs for this interface."""
        return FIGURES / f"{self.prefix}{name}.png"

    def decomposed(self) -> bool:
        """Whether this interface carries a constant-velocity control at every site.

        The figure that splits the onset error by hypocentre needs one at each. Asked of
        the results rather than of a list of interface names, so a plate read back from a
        partial rerun cannot draw a row that was never run.
        """
        return all(f"{site}_constant" in self.results["scenarios"] for site in SITES)


INTERFACES = (
    ("hikurangi", "arrays", HIKURANGI, "Hikurangi"),
    (
        "puysegur_fiordland",
        "puysegur_fiordland",
        PUYSEGUR_FIORDLAND,
        "Puysegur-Fiordland",
    ),
    ("puyseguer", "puyseguer", PUYSEGUR, "Puysegur"),
)
"""Each interface's key in ``results.json``, its raster stem, its surface and its title.

Hikurangi's raster is ``arrays.npz`` rather than ``hikurangi.npz`` because that is the
name it has always had and :mod:`curvature.run` still writes it.
"""


def _plate(
    groups: dict | None,
    stem: str,
    prefix: str,
    path: Path,
    label: str,
    magnitude: float,
):
    """One plate, or ``None`` if that run has not been made.

    Both halves of the check matter and neither implies the other: a group can be in
    ``results.json`` while its raster is still being written, and a raster can outlive a
    group that a later run replaced. A figure drawn from one without the other would be
    two runs superimposed.

    Parameters
    ----------
    groups : dict or None
        The interface's groups, or ``None`` if the file has none under that key.
    stem : str
        The raster's name in ``curvature/data``, without the extension.
    prefix, path, label, magnitude
        As :class:`Plate`.

    Returns
    -------
    Plate or None
    """
    raster = HERE / "data" / f"{stem}.npz"
    if groups is None or not raster.exists():
        return None
    return Plate(
        arrays=dict(np.load(raster)),
        results=groups,
        prefix=prefix,
        path=path,
        label=label,
        magnitude=magnitude,
    )


def _plates() -> list[Plate]:
    """Every interface at every magnitude that has been run.

    The baseline magnitude first, in the order the document reads it, then each further
    magnitude in turn. A magnitude's plates are the *same* figures drawn from the *same*
    code -- only the prefix, the label and the numbers differ -- which is what makes the
    two sets comparable by eye rather than by argument.
    """
    results = json.loads((HERE / "results.json").read_text())
    plates = []
    for name, stem, path, label in INTERFACES:
        groups = (
            results if name == "hikurangi" else results.get("puysegur", {}).get(name)
        )
        plate = _plate(
            groups,
            stem,
            run.prefix(name, model.MAGNITUDE),
            path,
            label,
            model.MAGNITUDE,
        )
        if plate is not None:
            plates.append(plate)
    for key, by_interface in sorted(results.get("magnitudes", {}).items()):
        magnitude = float(key.removeprefix("mw_"))
        for name, stem, path, label in INTERFACES:
            plate = _plate(
                by_interface.get(name),
                f"{stem}{run.tag(magnitude)}",
                run.prefix(name, magnitude),
                path,
                f"{label} (Mw {magnitude:g})",
                magnitude,
            )
            if plate is not None:
                plates.append(plate)
    return plates


def _save(figure: Figure, path: Path) -> None:
    """Write one PNG under a temporary name in its own directory and rename it in.

    The document build and a browser both read these while a rerun is rewriting them, and
    ``savefig`` truncates before it fills -- so a reader arriving in between gets a
    zero-byte PNG and no error to say why. :meth:`pathlib.Path.replace` is atomic within a
    filesystem, so a reader sees either the whole old figure or the whole new one. The
    partial is removed if the write fails.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Closed by the caller, as before; this only writes it.
    path : Path
        Where the figure belongs, from :meth:`Plate.figure`.
    """
    partial = path.with_name(f".partial-{path.name}")
    try:
        figure.savefig(partial)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def _fault_plane(
    axes: Axes,
    grid: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    *,
    cmap: Colormap,
    norm: Normalize | None = None,
    **kwargs: object,
) -> AxesImage:
    """Draw one raster in the fault-plane frame and label its axes.

    Depth increases downward, which is the orientation a down-dip section is drawn in
    and the one every reader of a slip model expects.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
    grid : ndarray
        ``(n_v, n_u)``, ``NaN`` outside the fault.
    axis_u, axis_v : ndarray
        Cell-centre coordinates, kilometres.
    cmap : Colormap
    norm : Normalize, optional

    Returns
    -------
    matplotlib.image.AxesImage
    """
    image = axes.imshow(
        grid,
        origin="upper",
        aspect="equal",
        extent=(axis_u[0], axis_u[-1], axis_v[-1], axis_v[0]),
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        **kwargs,
    )
    axes.set_xlabel("Along strike, u (km)")
    axes.set_ylabel("Down dip, v (km)")
    axes.grid(False)
    return image


def _bar(figure: Figure, image: ScalarMappable, label: str) -> Colorbar:
    """A colourbar with its units in the label, which is the only way it is allowed."""
    bar = figure.colorbar(image, ax=image.axes, fraction=0.03, pad=0.02)
    bar.set_label(label, fontsize=8, color=style.SECONDARY_INK)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, color=style.MUTED, labelcolor=style.SECONDARY_INK)
    return bar


def _polar_mean(
    panel: Axes,
    plate: Plate,
    scenario: str,
    norm: Normalize,
    radius_from: str = "central_standard",
) -> ScalarMappable:
    """One polar heatmap of the mean onset difference, binned by azimuth and distance.

    Azimuth is measured in the shared parameter plane -- 0 is along strike towards
    increasing ``u``, ``+pi/2`` is straight down dip -- and radius is the **true surface
    distance** from the hypocentre on the curved interface, from a unit-slowness eikonal
    solve.

    Binned rather than scattered: with a million faces a scatter is opaque, and the mean
    in a bin is the quantity a reader is trying to see anyway.

    Parameters
    ----------
    panel : matplotlib.axes.Axes
        On a polar projection.
    plate : Plate
    scenario : str
        Which delta to draw.
    norm : Normalize
        Shared across the panels a reader is meant to compare.
    radius_from : str, optional
        Whose azimuth and radius to use. The counterfactual rows share the central
        hypocentre's, since they are the same hypocentre on the same surface.

    Returns
    -------
    matplotlib.cm.ScalarMappable
    """
    azimuth = plate.arrays[f"polar_azimuth_{radius_from}_rad"]
    radius = plate.arrays[f"polar_radius_{radius_from}_km"]
    delta = plate.arrays[f"polar_delta_travel_{scenario}_s"]

    azimuth_edges = np.linspace(-np.pi, np.pi, 121)
    radius_edges = np.linspace(0.0, float(np.percentile(radius, 99.5)), 101)
    total, _, _ = np.histogram2d(
        azimuth, radius, bins=(azimuth_edges, radius_edges), weights=delta
    )
    count, _, _ = np.histogram2d(azimuth, radius, bins=(azimuth_edges, radius_edges))
    with np.errstate(invalid="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)

    image = panel.pcolormesh(
        azimuth_edges,
        radius_edges,
        mean.T,
        cmap=style.DIFFERENCE,
        norm=norm,
        shading="auto",
    )
    # Zero along +u and increasing clockwise, so down dip is at the bottom of the dial
    # exactly as it is at the bottom of every fault-plane map here.
    panel.set_theta_zero_location("E")
    panel.set_theta_direction(-1)
    panel.set_rlabel_position(112.5)
    panel.tick_params(labelsize=7, colors=style.SECONDARY_INK)
    panel.grid(color=style.GRID, linewidth=0.6)
    panel.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    panel.set_xticklabels(
        ["+u", "", "down dip", "", "-u", "", "up dip", ""], fontsize=7
    )
    return image


def _horizontal_bar(figure: Figure, image: ScalarMappable, axes: object, label: str):
    """A shared horizontal colourbar under a row of panels."""
    bar = figure.colorbar(
        image, ax=axes, fraction=0.03, pad=0.06, orientation="horizontal"
    )
    bar.set_label(label, fontsize=8, color=style.SECONDARY_INK)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, color=style.MUTED, labelcolor=style.SECONDARY_INK)
    return bar


def _entry(plate: Plate, scenario: str) -> dict:
    """One row of the decomposition, whichever group it is filed under.

    The four rows of the run matrix live in ``scenarios`` and the counterfactual rows in
    ``true_depth``, because the published Hikurangi ``scenarios`` group is quoted as it
    stands and gaining a key would change it. A figure that draws all of them together
    should not have to know that, so this is where it is known.
    """
    if scenario in plate.results["scenarios"]:
        return plate.results["scenarios"][scenario]
    return plate.results["true_depth"]["scenarios"][scenario]


def depth_error(plate: Plate) -> None:
    """The driver of everything depth-mediated: how far the plane is from the interface.

    ``Delta z = z_flat - z_curved`` per face, its distribution, and the per-face area
    ratio that drives the moment error -- side by side, because their relative size is
    the study's first question and putting them on one figure answers it by inspection.

    The area panel's ramp reaches the next half above the metric factor at the 99th
    percentile of ``|grad h|``, which on Hikurangi is 1.5 and on the far more curved
    Puysegur surfaces is 2.5. Derived rather than fixed, because a ramp that stopped at
    Hikurangi's limit would saturate exactly where Puysegur is interesting.
    """
    arrays, results = plate.arrays, plate.results
    axis_u, axis_v = arrays["raster_axis_u_km"], arrays["raster_axis_v_km"]
    error = arrays["raster_depth_error_km"]
    geometry = results["geometry"]

    figure, panels = plt.subplots(3, 1, figsize=(9.5, 10.5))

    image = _fault_plane(
        panels[0],
        error,
        axis_u,
        axis_v,
        cmap=style.DIFFERENCE,
        norm=style.centred(error),
    )
    _bar(figure, image, "Depth error, flat - curved (km)")
    panels[0].set_title(
        f"Depth error of the best-fit plane: {geometry['depth_error_flat_minus_curved_min_km']:.1f} "
        f"to +{geometry['depth_error_flat_minus_curved_max_km']:.1f} km"
    )

    values = error[np.isfinite(error)]
    panels[1].hist(values, bins=160, color=style.CURVED, edgecolor="none")
    for label, key in (
        ("p10", "depth_error_flat_minus_curved_p10_km"),
        ("median", "depth_error_flat_minus_curved_median_km"),
        ("p90", "depth_error_flat_minus_curved_p90_km"),
    ):
        at = geometry[key]
        panels[1].axvline(at, color=style.SECONDARY_INK, linewidth=1.0)
        panels[1].annotate(
            f"{label} {at:+.1f}",
            (at, 0.94),
            xycoords=("data", "axes fraction"),
            fontsize=7,
            color=style.SECONDARY_INK,
            ha="left" if at >= 0 else "right",
        )
    panels[1].set_xlabel("Depth error, flat - curved (km)")
    panels[1].set_ylabel("Faces")
    panels[1].set_title(
        f"Signed, so it nearly cancels in the mean "
        f"({geometry['depth_error_flat_minus_curved_mean_km']:+.2f} km) while the "
        f"typical face is "
        f"{geometry['depth_error_flat_minus_curved_median_absolute_km']:.1f} km out"
    )

    # The metric factor is the *exact* per-face area ratio of a Monge patch, so this
    # panel and the total ratio quoted in its title are one quantity at two scales.
    metric = np.sqrt(1.0 + arrays["raster_slope_grad_h"] ** 2)
    top = float(
        np.ceil(np.sqrt(1.0 + geometry["slope_grad_h_p99_dimensionless"] ** 2) / 0.5)
        * 0.5
    )
    image = _fault_plane(
        panels[2], metric, axis_u, axis_v, cmap=style.MAGNITUDE, vmin=1.0, vmax=top
    )
    _bar(figure, image, "True area / projected area")
    panels[2].set_title(
        f"Area inflation sqrt(1 + |grad h|^2): total ratio "
        f"{geometry['area_ratio_true_over_projected']:.4f}"
    )

    _save(figure, plate.figure("depth_error"))
    plt.close(figure)


def sections(plate: Plate) -> None:
    """Down-dip sections: the true interface against the plane it is being replaced by.

    Three along-strike positions, chosen by what they show rather than by where they
    are: the column whose worst departure from the plane is largest, one at the median,
    and the flattest column on the interface. Together they say that the deviation is
    not uniform -- which is why a single "how curved is it" number understates the
    local error.
    """
    arrays = plate.arrays
    axis_u, axis_v = arrays["raster_axis_u_km"], arrays["raster_axis_v_km"]
    curved = arrays["raster_depth_curved_km"]
    flat = arrays["raster_depth_flat_km"]
    error = np.abs(arrays["raster_depth_error_km"])
    spacing_km = float(axis_v[1] - axis_v[0])

    with np.errstate(invalid="ignore"):
        worst = np.nanmax(np.where(np.isfinite(error), error, np.nan), axis=0)
    usable = np.flatnonzero(
        np.isfinite(worst)
        & (np.isfinite(curved).sum(axis=0) * spacing_km > MINIMUM_SECTION_EXTENT_KM)
    )
    order = usable[np.argsort(worst[usable])]
    chosen = {
        "flattest": int(order[0]),
        "typical": int(order[len(order) // 2]),
        "worst": int(order[-1]),
    }

    figure, panels = plt.subplots(3, 1, figsize=(8.5, 8.5), sharex=True)
    for panel, (label, column) in zip(panels, chosen.items(), strict=True):
        depth_curved = curved[:, column]
        depth_flat = flat[:, column]
        inside = np.isfinite(depth_curved)
        panel.plot(
            axis_v[inside],
            depth_curved[inside],
            color=style.CURVED,
            label="True interface",
        )
        panel.plot(
            axis_v[inside],
            depth_flat[inside],
            color=style.FLAT,
            label="Best-fit plane",
        )
        gap = depth_flat - depth_curved
        at = int(np.nanargmax(np.abs(np.where(inside, gap, np.nan))))
        panel.annotate(
            "",
            xy=(axis_v[at], depth_curved[at]),
            xytext=(axis_v[at], depth_flat[at]),
            arrowprops={"arrowstyle": "<->", "color": style.SECONDARY_INK, "lw": 1.0},
        )
        # The label goes on whichever side of the arrow has room: the worst departure
        # is often at a column's deep end, where a right-hand label runs off the axes.
        right_half = axis_v[at] > 0.5 * (axis_v[inside][0] + axis_v[inside][-1])
        panel.annotate(
            f"{gap[at]:+.1f} km",
            (axis_v[at], 0.5 * (depth_curved[at] + depth_flat[at])),
            fontsize=8,
            color=style.INK,
            ha="right" if right_half else "left",
            xytext=(-8 if right_half else 8, 0),
            textcoords="offset points",
        )
        panel.invert_yaxis()
        panel.set_ylabel("Depth (km)")
        panel.set_title(
            f"{label.capitalize()} column, u = {axis_u[column]:.0f} km along strike: "
            f"worst departure {worst[column]:.1f} km"
        )
    panels[0].legend(loc="lower left")
    panels[-1].set_xlabel("Down dip, v (km)")
    _save(figure, plate.figure("sections"))
    plt.close(figure)


def plan_view(plate: Plate) -> None:
    """Map view: where on the interface the plane is wrong, and where the hypocentres are.

    The one figure not in the fault-plane frame. It exists so the fault-plane maps can be
    located on a map of New Zealand, and it is drawn at the **true** positions, since the
    flat model's own map positions are up to 6 km away horizontally and plotting the
    difference at those would put the error in the wrong place.
    """
    pair = build_pair(plate.path)
    centres = pair.centres_km(pair.curved_km)
    flat_centres = pair.centres_km(pair.flat_km)
    longitude, latitude = pair.to_lonlat(centres)
    error = flat_centres[:, 2] - centres[:, 2]
    # Degrees on both axes, so a degree of longitude is cos(latitude) as long as one of
    # latitude. Evaluated at this interface's own mean latitude rather than at a
    # constant, which would be the wrong parallel on any surface but the first.
    aspect = 1.0 / np.cos(np.radians(float(np.mean(latitude))))

    figure, panels = plt.subplots(1, 2, figsize=(11.0, 6.5))
    for panel, values, cmap, norm, label, title in (
        (
            panels[0],
            centres[:, 2],
            style.MAGNITUDE,
            None,
            "Interface depth (km)",
            f"The {plate.label} interface, CFM v1.0",
        ),
        (
            panels[1],
            error,
            style.DIFFERENCE,
            style.centred(error),
            "Depth error, flat - curved (km)",
            "Where the best-fit plane departs from it",
        ),
    ):
        # Binned rather than scattered: 1.39 M markers is a solid block that hides its
        # own distribution and costs megabytes in the PNG.
        counts, edge_x, edge_y = np.histogram2d(
            longitude, latitude, bins=(360, 360), weights=values
        )
        totals, _, _ = np.histogram2d(longitude, latitude, bins=(edge_x, edge_y))
        with np.errstate(invalid="ignore"):
            grid = np.where(totals > 0, counts / np.maximum(totals, 1), np.nan)
        image = panel.pcolormesh(
            edge_x, edge_y, grid.T, cmap=cmap, norm=norm, shading="auto"
        )
        _bar(figure, image, label)
        panel.set_xlabel("Longitude (degrees east)")
        panel.set_ylabel("Latitude (degrees north)")
        panel.set_title(title)
        panel.set_aspect(aspect)
        panel.grid(False)

        for site, located in plate.results["hypocentres"].items():
            panel.plot(
                located["curved_longitude_deg"],
                located["curved_latitude_deg"],
                marker="*",
                markersize=14,
                color=style.INK,
                markeredgecolor=style.SURFACE,
                markeredgewidth=1.5,
                linestyle="none",
            )
            panel.annotate(
                site,
                (located["curved_longitude_deg"], located["curved_latitude_deg"]),
                xytext=(8, 4),
                textcoords="offset points",
                fontsize=8,
                color=style.INK,
            )

    _save(figure, plate.figure("plan_view"))
    plt.close(figure)


def moment(plate: Plate) -> None:
    """The moment error, split into the part the area causes and the part rigidity does.

    The bars are ratios against 1, which is what "no error" is, so the axis is drawn
    from 1 rather than from 0 -- a ratio chart baselined at zero would hide a 3% effect
    entirely. The two velocity settings sit side by side because their difference is the
    controlled attribution. The counterfactual's own split is in :func:`true_depth_moment`
    rather than here, so that these two bars stay where the document reads them.
    """
    arrays, results = plate.arrays, plate.results
    axis_u, axis_v = arrays["raster_axis_u_km"], arrays["raster_axis_v_km"]
    figure = plt.figure(figsize=(10.5, 8.5))
    grid = figure.add_gridspec(2, 2)
    bars = figure.add_subplot(grid[0, 0])
    spread = figure.add_subplot(grid[0, 1])
    layers = figure.add_subplot(grid[1, :])

    labels = ["Area only", "Rigidity only", "Combined, delivered / target"]
    keys = [
        "area_contribution_ratio",
        "rigidity_contribution_ratio",
        "flat_delivered_over_target",
    ]
    positions = np.arange(len(labels))
    for offset, (name, colour) in enumerate(
        (("constant", style.CURVED), ("standard", style.FLAT))
    ):
        values = [results["moment"][name][key] for key in keys]
        bars.barh(
            positions + (offset - 0.5) * 0.38,
            [value - 1.0 for value in values],
            height=0.34,
            left=1.0,
            color=colour,
            label=f"{name} velocity model",
        )
        for position, value in zip(
            positions + (offset - 0.5) * 0.38, values, strict=True
        ):
            bars.annotate(
                f"{value:.4f}",
                (value, position),
                xytext=(6 if value >= 1.0 else -6, 0),
                textcoords="offset points",
                va="center",
                ha="left" if value >= 1.0 else "right",
                fontsize=8,
                color=style.INK,
            )
    bars.axvline(1.0, color=style.AXIS, linewidth=1.0)
    bars.set_yticks(positions, labels)
    bars.set_xlabel("Ratio to the target moment (dimensionless)")
    bars.set_title("What the flat model's moment is wrong by")
    bars.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    bars.grid(axis="y", visible=False)
    bars.margins(x=0.18)

    ratio = arrays["raster_rigidity_ratio"]
    values = ratio[np.isfinite(ratio)]
    spread.hist(values, bins=120, color=style.CURVED, edgecolor="none")
    spread.axvline(1.0, color=style.SECONDARY_INK, linewidth=1.0)
    spread.set_xlabel("Rigidity ratio, true depth / flat depth (dimensionless)")
    spread.set_ylabel("Faces")
    spread.set_title(
        f"Per-subfault rigidity error: "
        f"{results['moment']['standard']['fraction_with_rigidity_error']:.1%} of faces wrong"
    )

    # Binary, so a two-step listed map with two ticks rather than a continuous ramp:
    # a colourbar running 0.0 to 1.0 would imply intermediate states that do not exist.
    crossed = arrays["raster_layer_crossed"]
    image = _fault_plane(
        layers,
        crossed,
        axis_u,
        axis_v,
        cmap=ListedColormap([style.MIDPOINT, style.CURVED]),
        vmin=0.0,
        vmax=1.0,
    )
    bar = _bar(figure, image, "Velocity layer in the flat model")
    bar.set_ticks([0.25, 0.75])
    bar.set_ticklabels(["same", "different"])
    layers.set_title(
        f"{results['moment']['standard']['fraction_in_different_velocity_layer']:.1%} of "
        "faces land in a different velocity layer in the flat model"
    )
    _save(figure, plate.figure("moment"))
    plt.close(figure)


def onset_polar(plate: Plate) -> None:
    """The headline: onset error by direction and distance from the hypocentre.

    Colour is ``t_flat - t_curved``, diverging about zero, so a direction in which the
    flat model runs early is a different colour from one where it runs late and neither
    is the colour of no difference. All four scenarios share one scale, which is what
    makes their relative size readable rather than a comparison of four colourbars.
    """
    scenarios = list(SCENARIO_LABELS)
    figure, panels = plt.subplots(
        2, 2, figsize=(10.5, 10.0), subplot_kw={"projection": "polar"}
    )
    every = np.concatenate(
        [plate.arrays[f"polar_delta_travel_{name}_s"] for name in scenarios]
    )
    norm = style.centred(every)

    for panel, name in zip(panels.ravel(), scenarios, strict=True):
        image = _polar_mean(panel, plate, name, norm, radius_from=name)
        entry = plate.results["scenarios"][name]
        panel.set_title(
            f"{SCENARIO_LABELS[name]}\nmedian "
            f"{entry['delta_travel_time_flat_minus_curved_median_s']:+.2f} s, largest "
            f"{entry['delta_travel_time_flat_minus_curved_max_absolute_s']:.1f} s",
            pad=14,
            fontsize=9,
        )

    _horizontal_bar(
        figure,
        image,
        panels,
        "Rupture front arrival, flat - curved (s).  Radius: surface distance from the "
        "hypocentre (km)",
    )
    _save(figure, plate.figure("onset_polar"))
    plt.close(figure)


def onset_polar_control(plate: Plate) -> None:
    """The geometric onset error alone, on a scale that can show it.

    The four-panel figure puts every scenario on one colour scale, which is what makes
    the attribution readable -- and which flattens the constant-velocity panel to a
    single tone, because the geometric effect is two orders of magnitude smaller. This
    is that panel on its own scale, so its *structure* is visible: the flat model's
    shorter paths, and where on the dial they are shortest.

    The two figures must be read together. Neither is complete: this one shows a pattern
    whose amplitude is negligible, and the other shows an amplitude whose pattern is
    invisible.
    """
    name = "central_constant"
    delta = plate.arrays[f"polar_delta_travel_{name}_s"]

    figure = plt.figure(figsize=(6.4, 6.6))
    panel = figure.add_subplot(projection="polar")
    image = _polar_mean(panel, plate, name, style.centred(delta), radius_from=name)
    entry = plate.results["scenarios"][name]
    panel.set_title(
        "Geometry alone: the same quantity at 1/30 of the scale\n"
        f"median {entry['delta_travel_time_flat_minus_curved_median_s']:+.3f} s, "
        f"largest {entry['delta_travel_time_flat_minus_curved_max_absolute_s']:.2f} s",
        pad=16,
    )
    _horizontal_bar(
        figure,
        image,
        panel,
        "Rupture front arrival, flat - curved (s).  Radius: surface distance (km)",
    )
    _save(figure, plate.figure("onset_polar_control"))
    plt.close(figure)


def onset_maps(plate: Plate) -> None:
    """The wavefront in both models, and the three attributions of their difference.

    The last three panels share one diverging scale, which is the whole point: the
    constant-velocity panel is the error the geometry alone causes, the standard panel is
    the error with depth added, and the true-depth panel is what is left once the
    materials are read off the real interface. A shared scale is what makes their
    relative size readable rather than a comparison of three colourbars.

    **The scale is set by the two panels that were published together**, so those two
    look exactly as they did and the counterfactual is drawn onto the same ruler rather
    than moving it.
    """
    arrays = plate.arrays
    axis_u, axis_v = arrays["raster_axis_u_km"], arrays["raster_axis_v_km"]
    figure, panels = plt.subplots(5, 1, figsize=(9.5, 16.2))

    top = float(
        np.nanmax(
            [
                np.nanmax(arrays["raster_travel_curved_central_standard_s"]),
                np.nanmax(arrays["raster_travel_flat_central_standard_s"]),
            ]
        )
    )
    for panel, key, title in (
        (
            panels[0],
            "raster_travel_curved_central_standard_s",
            (
                f"Rupture front on the true {plate.label} interface, central "
                "hypocentre, standard velocities"
            ),
        ),
        (
            panels[1],
            "raster_travel_flat_central_standard_s",
            "The same front on the best-fit plane",
        ),
    ):
        image = _fault_plane(
            panel, arrays[key], axis_u, axis_v, cmap=style.MAGNITUDE, vmin=0.0, vmax=top
        )
        _bar(figure, image, "Arrival time (s)")
        panel.set_title(title)

    both = np.concatenate(
        [
            arrays["raster_delta_travel_central_constant_s"].ravel(),
            arrays["raster_delta_travel_central_standard_s"].ravel(),
        ]
    )
    norm = style.centred(both)
    for panel, name in (
        (panels[2], "central_constant"),
        (panels[3], "central_standard"),
        (panels[4], "central_standard_truedepth"),
    ):
        image = _fault_plane(
            panel,
            arrays[f"raster_delta_travel_{name}_s"],
            axis_u,
            axis_v,
            cmap=style.DIFFERENCE,
            norm=norm,
        )
        _bar(figure, image, "Arrival, flat - curved (s)")
        entry = _entry(plate, name)
        label = SCENARIO_LABELS.get(
            name, "Central, standard 1-D, materials at the true depth"
        )
        panel.set_title(
            f"{label}: median "
            f"{entry['delta_travel_time_flat_minus_curved_median_s']:+.2f} s, "
            f"largest {entry['delta_travel_time_flat_minus_curved_max_absolute_s']:.1f} s"
        )
    _save(figure, plate.figure("onset_maps"))
    plt.close(figure)


def true_depth(plate: Plate) -> None:
    """The three-way decomposition: what the refactor buys, and what it cannot.

    The top row is the argument. Three polar panels on **one** scale, set by the status
    quo, so a reader sees at a glance that correcting the material assignment takes the
    onset error from the third panel to something indistinguishable from the first.

    The bottom row is the residue. The two small panels again on a scale of their own,
    because a difference of a tenth of a second has structure and it is not visible on a
    ruler that has to reach tens of seconds; and beside them the decomposition itself,
    with the two gaps the refactor decision turns on drawn as gaps rather than quoted as
    numbers.

    The three conditions are all flat models, so they are not the curved/flat pair the
    palette's two categorical slots are for. The residual condition takes the flat
    model's colour with a hollow marker, which is :mod:`curvature.style`'s own rule for
    a third series: distinguish it by frame, not by a colour that fails contrast.
    """
    figure = plt.figure(figsize=(12.6, 9.6))
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 0.9), wspace=0.30)

    names = list(TRUE_DEPTH_LABELS)
    every = np.concatenate(
        [plate.arrays[f"polar_delta_travel_{name}_s"] for name in names]
    )
    shared = style.centred(every)
    top_panels = [
        figure.add_subplot(grid[0, column], polar=True) for column in range(3)
    ]
    for panel, name in zip(top_panels, names, strict=True):
        image = _polar_mean(panel, plate, name, shared)
        entry = _entry(plate, name)
        panel.set_title(
            f"{TRUE_DEPTH_LABELS[name]}\nmedian "
            f"{entry['delta_travel_time_flat_minus_curved_median_s']:+.3f} s",
            pad=14,
            fontsize=8,
        )
    _horizontal_bar(
        figure,
        image,
        top_panels,
        "Rupture front arrival, flat - curved (s), on one shared scale.  Radius: "
        "surface distance from the hypocentre (km)",
    )

    residual = ("central_constant", "central_standard_truedepth")
    small = np.concatenate(
        [plate.arrays[f"polar_delta_travel_{name}_s"] for name in residual]
    )
    own = style.centred(small)
    bottom_panels = [
        figure.add_subplot(grid[1, column], polar=True) for column in range(2)
    ]
    for panel, name in zip(bottom_panels, residual, strict=True):
        image = _polar_mean(panel, plate, name, own)
        panel.set_title(
            f"{TRUE_DEPTH_LABELS[name].splitlines()[0]}, on its own scale",
            pad=12,
            fontsize=8,
        )
    _horizontal_bar(
        figure,
        image,
        bottom_panels,
        "The same quantity, rescaled to what is left (s)",
    )

    ladder = figure.add_subplot(grid[1, 2])
    medians = [
        _entry(plate, name)["delta_travel_time_flat_minus_curved_median_s"]
        for name in names
    ]
    faces = (
        {"color": style.CURVED},
        {"color": style.FLAT, "markerfacecolor": style.SURFACE},
        {"color": style.FLAT},
    )
    for position, (value, face) in enumerate(zip(medians, faces, strict=True)):
        ladder.plot(
            [0.0, value],
            [position, position],
            color=face["color"],
            linewidth=2.0,
            zorder=2,
        )
        ladder.plot(
            [value],
            [position],
            marker="o",
            markersize=9,
            markeredgewidth=2.0,
            markeredgecolor=face["color"],
            markerfacecolor=face.get("markerfacecolor", face["color"]),
            linestyle="none",
            zorder=3,
        )
        # Always to the right of the marker. The two residual rows sit a fraction of a
        # second on the negative side of zero, and a label placed outward from them runs
        # into the condition names on the axis.
        ladder.annotate(
            f"{value:+.3f} s",
            (value, position),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8,
            color=style.INK,
        )
    # One gap drawn as a gap: the one the refactor closes, which spans most of the axis.
    # The other -- what is left -- is the true-depth row's own position, already labelled
    # on it, so it is stated as a share rather than drawn as a second arrow. The two
    # residual rows sit within a tenth of a second of each other on an axis reaching
    # several seconds, and an arrow between them would be shorter than its own head.
    refactor = plate.results["true_depth"]["refactor"]
    bought = refactor[
        "value_of_the_refactor_truedepth_delta_travel_time_flat_minus_curved_median_s"
    ]
    ladder.annotate(
        "",
        xy=(medians[1], 1.5),
        xytext=(medians[2], 1.5),
        arrowprops={"arrowstyle": "<->", "color": style.SECONDARY_INK, "lw": 1.0},
    )
    ladder.annotate(
        f"what the refactor buys: {bought:+.3f} s",
        (0.5 * (medians[1] + medians[2]), 1.5),
        xytext=(0, 10),
        textcoords="offset points",
        fontsize=7,
        color=style.SECONDARY_INK,
        ha="center",
        va="bottom",
    )
    ladder.annotate(
        f"what is left: {refactor['irreducible_geometric_cost_truedepth_median_s']:+.3f} s,\n"
        f"{abs(refactor['irreducible_geometric_cost_truedepth_median_s'] / medians[2]):.1%}"
        " of the status quo",
        (0.02, 0.03),
        xycoords="axes fraction",
        fontsize=7,
        color=style.SECONDARY_INK,
        ha="left",
        va="bottom",
    )
    ladder.axvline(0.0, color=style.AXIS, linewidth=1.0)
    ladder.set_yticks(
        range(3), [TRUE_DEPTH_LABELS[name].splitlines()[0] for name in names]
    )
    ladder.set_xlabel("Median onset error, flat - curved (s)")
    ladder.set_title("The decomposition")
    ladder.grid(axis="y", visible=False)
    ladder.margins(x=0.34, y=0.26)

    _save(figure, plate.figure("true_depth"))
    plt.close(figure)


def true_depth_moment(plate: Plate) -> None:
    """The moment split under all three conditions, and where the rigidity error goes.

    The exact factorisation, so the reading is arithmetic rather than statistical: the
    counterfactual's rigidity part is **1 to machine precision**, because ``mu`` is read
    at the same depth on both sides of the ratio. What survives is the area part, and it
    is what a projection onto a curved surface costs whatever the materials do.
    """
    figure, panels = plt.subplots(1, 2, figsize=(11.0, 4.6))
    labels = ["Area only", "Rigidity only", "Combined, delivered / target"]
    keys = [
        "area_contribution_ratio",
        "rigidity_contribution_ratio",
        "flat_delivered_over_target",
    ]
    rows = (
        ("constant velocity", plate.results["moment"]["constant"], style.CURVED, True),
        (
            "standard, true-depth materials",
            plate.results["true_depth"]["moment"]["truedepth"],
            style.FLAT,
            False,
        ),
        (
            "standard, flat-depth materials",
            plate.results["moment"]["standard"],
            style.FLAT,
            True,
        ),
    )
    positions = np.arange(len(labels))
    for offset, (name, group, colour, filled) in enumerate(rows):
        values = [group[key] for key in keys]
        # Top to bottom in the order the legend lists them, which is the order the
        # argument runs in: control, counterfactual, status quo.
        panels[0].barh(
            positions + (1 - offset) * 0.28,
            [value - 1.0 for value in values],
            height=0.24,
            left=1.0,
            color=colour if filled else style.SURFACE,
            edgecolor=colour,
            linewidth=0.0 if filled else 1.6,
            label=name,
        )
        for position, value in zip(
            positions + (1 - offset) * 0.28, values, strict=True
        ):
            panels[0].annotate(
                f"{value:.4f}",
                (value, position),
                xytext=(6 if value >= 1.0 else -6, 0),
                textcoords="offset points",
                va="center",
                ha="left" if value >= 1.0 else "right",
                fontsize=7,
                color=style.INK,
            )
    panels[0].axvline(1.0, color=style.AXIS, linewidth=1.0)
    panels[0].set_yticks(positions, labels)
    panels[0].set_xlabel("Ratio to the target moment (dimensionless)")
    panels[0].set_title("The moment error under the three conditions")
    panels[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=1)
    panels[0].grid(axis="y", visible=False)
    panels[0].margins(x=0.24)

    # The onset gap on the right, so the two things the refactor changes -- when the
    # front arrives and how much moment it delivers -- are read off one plate.
    # Four signed statistics of one signed quantity. The largest *absolute* error is
    # reported in `results.json` and not here: an unsigned statistic on an axis whose
    # other rows carry a sign would read as a sixth condition arriving late.
    quantities = (
        ("p10", "delta_travel_time_flat_minus_curved_p10_s"),
        ("median", "delta_travel_time_flat_minus_curved_median_s"),
        (
            "area-weighted median",
            "delta_travel_time_flat_minus_curved_area_weighted_median_s",
        ),
        ("p90", "delta_travel_time_flat_minus_curved_p90_s"),
    )
    positions = np.arange(len(quantities))
    for offset, (name, scenario, colour, filled) in enumerate(
        (
            ("status quo", "central_standard", style.FLAT, True),
            ("true-depth materials", "central_standard_truedepth", style.FLAT, False),
        )
    ):
        entry = _entry(plate, scenario)
        values = [entry[quantity] for _, quantity in quantities]
        panels[1].barh(
            positions + (offset - 0.5) * 0.36,
            values,
            height=0.32,
            color=colour if filled else style.SURFACE,
            edgecolor=colour,
            linewidth=0.0 if filled else 1.6,
            label=name,
        )
        for position, value in zip(
            positions + (offset - 0.5) * 0.36, values, strict=True
        ):
            panels[1].annotate(
                f"{value:+.2f}",
                (value, position),
                xytext=(6 if value >= 0.0 else -6, 0),
                textcoords="offset points",
                va="center",
                ha="left" if value >= 0.0 else "right",
                fontsize=7,
                color=style.INK,
            )
    panels[1].axvline(0.0, color=style.AXIS, linewidth=1.0)
    panels[1].set_yticks(positions, [name for name, _ in quantities])
    panels[1].set_xlabel("Onset error, flat - curved (s)")
    panels[1].set_title("What is left of the onset error")
    panels[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=2)
    panels[1].grid(axis="y", visible=False)
    panels[1].margins(x=0.26)

    _save(figure, plate.figure("true_depth_moment"))
    plt.close(figure)


def decomposition_by_site(plate: Plate) -> None:
    """Does the geometric term move along strike the way the depth term does?

    :func:`true_depth` answers the refactor question at one hypocentre. This asks the
    other half of it: the three standard rows differ by several seconds along strike, and
    with a control at the centre alone that spread is a total whose two mechanisms cannot
    be told apart. Here every hypocentre has all three rows, so the split is read off the
    figure rather than inferred from the middle one.

    **The top row is on one shared scale and the bottom axis is not.** The three
    geometric panels are the same quantity at three hypocentres and comparing them is the
    point, so they share a norm; the ladder underneath has to hold a geometric term near
    a hundredth of a second beside a total near ten, and a scale that made both legible
    would be a lie about their relative size. The geometric rows *should* look like
    nothing there. That they do is the result.

    The palette is :func:`true_depth`'s, unchanged, so a reader moving between the two
    figures reads the same three conditions in the same three marks: the control filled
    in the curved model's colour, the counterfactual hollow, the status quo filled.
    """
    figure = plt.figure(figsize=(12.6, 9.0))
    grid = figure.add_gridspec(2, 3, height_ratios=(1.0, 0.95), wspace=0.30)

    controls = [f"{site}_constant" for site in SITES]
    shared = style.centred(
        np.concatenate(
            [plate.arrays[f"polar_delta_travel_{name}_s"] for name in controls]
        )
    )
    panels = [figure.add_subplot(grid[0, column], polar=True) for column in range(3)]
    for panel, site, name in zip(panels, SITES, controls, strict=True):
        image = _polar_mean(panel, plate, name, shared, radius_from=name)
        entry = plate.results["scenarios"][name]
        panel.set_title(
            f"{site.capitalize()}, constant velocity\nmedian "
            f"{entry['delta_travel_time_flat_minus_curved_median_s']:+.3f} s, largest "
            f"{entry['delta_travel_time_flat_minus_curved_max_absolute_s']:.2f} s",
            pad=14,
            fontsize=8,
        )
    _horizontal_bar(
        figure,
        image,
        panels,
        "Geometry alone: rupture front arrival, flat - curved (s), on one shared scale.  "
        "Radius: surface distance from the hypocentre (km)",
    )

    ladder = figure.add_subplot(grid[1, :])
    conditions = (
        ("geometry alone (constant velocity)", "constant", style.CURVED, False),
        ("geometry + true-depth materials", "standard_truedepth", style.FLAT, True),
        ("geometry + flat-depth materials (status quo)", "standard", style.FLAT, False),
    )
    for offset, (label, suffix, colour, hollow) in enumerate(conditions):
        positions = np.arange(len(SITES)) * 4.0 + offset
        values = [
            _entry(plate, f"{site}_{suffix}")[
                "delta_travel_time_flat_minus_curved_median_s"
            ]
            for site in SITES
        ]
        for position, value in zip(positions, values, strict=True):
            ladder.plot(
                [0.0, value],
                [position, position],
                color=colour,
                linewidth=2.0,
                zorder=2,
            )
            # Outward from zero, unlike :func:`true_depth`'s ladder. There the rows sit
            # close enough to zero that an outward label runs into the condition names;
            # here the names are at the far left and the stem is what an inward label
            # would be written across.
            ladder.annotate(
                f"{value:+.3f} s",
                (value, position),
                xytext=(10 if value >= 0.0 else -10, 0),
                textcoords="offset points",
                va="center",
                ha="left" if value >= 0.0 else "right",
                fontsize=8,
                color=style.INK,
            )
        ladder.plot(
            values,
            positions,
            marker="o",
            markersize=9,
            markeredgewidth=2.0,
            markeredgecolor=colour,
            markerfacecolor=style.SURFACE if hollow else colour,
            linestyle="none",
            label=label,
            zorder=3,
        )

    split = plate.results["attribution"]["by_site"]
    gaps = ", ".join(
        f"{site} {split[site]['constant_minus_true_depth_delta_travel_median_s']:+.3f} s"
        for site in SITES
    )
    ladder.axvline(0.0, color=style.AXIS, linewidth=1.0)
    ladder.set_yticks(
        np.arange(len(SITES)) * 4.0 + 1.0, [s.capitalize() for s in SITES]
    )
    ladder.invert_yaxis()
    ladder.set_xlabel("Median onset error, flat - curved (s)")
    ladder.set_title(
        "The same decomposition at each hypocentre.  Control minus counterfactual: "
        f"{gaps}"
    )
    ladder.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncols=3)
    ladder.grid(axis="y", visible=False)
    ladder.margins(x=0.20, y=0.10)

    _save(figure, plate.figure("decomposition_by_site"))
    plt.close(figure)


def slip_maps(plate: Plate) -> None:
    """Slip in both models and their difference, from the same white noise.

    The two slip panels share a scale so their amplitudes are comparable; the difference
    panel is diverging about zero. The fields are not two realisations -- they are one
    realisation of the noise, acted on by two geometries.
    """
    arrays, results = plate.arrays, plate.results
    axis_u, axis_v = arrays["raster_axis_u_km"], arrays["raster_axis_v_km"]
    curved = arrays["raster_slip_standard_curved_m"]
    flat = arrays["raster_slip_standard_flat_m"]
    top = float(np.nanpercentile(np.concatenate([curved.ravel(), flat.ravel()]), 99.5))

    figure, panels = plt.subplots(3, 1, figsize=(9.5, 10.5))
    for panel, values, title in (
        (panels[0], curved, "Slip generated on the true interface"),
        (panels[1], flat, "Slip generated on the plane, from the same white noise"),
    ):
        image = _fault_plane(
            panel, values, axis_u, axis_v, cmap=style.MAGNITUDE, vmin=0.0, vmax=top
        )
        _bar(figure, image, "Slip (m)")
        panel.set_title(title)

    difference = flat - curved
    image = _fault_plane(
        panels[2],
        difference,
        axis_u,
        axis_v,
        cmap=style.DIFFERENCE,
        norm=style.centred(difference),
    )
    _bar(figure, image, "Slip, flat - curved (m)")
    offset = float(np.nanmean(difference))
    panels[2].set_title(
        f"Difference: a {offset:+.3f} m offset from the moment rescaling, on a field "
        f"correlating {results['correlation']['pointwise_correlation_slip_pattern']:.3f} "
        "with its twin"
    )
    _save(figure, plate.figure("slip_maps"))
    plt.close(figure)


def correlation(plate: Plate) -> None:
    """Delivered correlation length against **surface** separation, for both models.

    The measurement that says what the literature approach does to the *structure* of
    slip rather than to its size. Both curves are read at the same physical separation on
    the true interface, so a flat model whose field has the right length in parameter
    kilometres shows up here as a field whose length is too long in real ones.
    """
    arrays = plate.arrays
    figure, panels = plt.subplots(1, 2, figsize=(11.0, 4.8))
    summary = plate.results["correlation"]
    level = summary["correlation_level_at_one_length"]
    readings = (
        ("curved_on_true", style.CURVED, "-", "Curved model, on the true interface"),
        ("flat_on_plane", style.FLAT, ":", "Flat model, on its own plane"),
        ("flat_on_true", style.FLAT, "-", "Flat model, projected onto the interface"),
    )

    for panel, direction, asked in (
        (panels[0], "strike", summary["asked_correlation_length_strike_km"]),
        (panels[1], "dip", summary["asked_correlation_length_dip_km"]),
    ):
        for offset, (reading, colour, dash, label) in enumerate(readings):
            key = f"{reading}_gaussian_{direction}"
            panel.plot(
                arrays[f"{key}_separation_km"],
                arrays[f"{key}_correlation"],
                color=colour,
                linestyle=dash,
                label=label,
            )
            delivered = summary[f"delivered_length_{key}_km"]
            if not np.isfinite(delivered):
                continue
            panel.plot(
                [delivered],
                [level],
                marker="o",
                color=colour,
                markersize=7,
                markeredgecolor=style.SURFACE,
                markeredgewidth=1.5,
                linestyle="none",
            )
            panel.annotate(
                f"{delivered:.1f} km",
                (delivered, level),
                xytext=(8, 14 - 13 * offset),
                textcoords="offset points",
                fontsize=8,
                color=style.INK,
            )
        panel.axhline(level, color=style.AXIS, linewidth=1.0)
        panel.axvline(asked, color=style.AXIS, linewidth=1.0)
        panel.annotate(
            f"asked {asked:.1f} km",
            (asked, 0.95),
            xycoords=("data", "axes fraction"),
            fontsize=7,
            color=style.MUTED,
            rotation=90,
            va="top",
            ha="right",
        )
        panel.set_xlabel(f"Surface separation along {direction} (km)")
        panel.set_ylabel("Correlation (dimensionless)")
        panel.set_title(f"Along {direction}")
        panel.set_xlim(0.0, 3.0 * asked)
        panel.set_ylim(0.0, 1.02)
    panels[0].legend(loc="upper right")
    _save(figure, plate.figure("correlation"))
    plt.close(figure)


def _log_binned(
    frequency_hz: np.ndarray, amplitude: np.ndarray, bins: int = 90
) -> tuple[np.ndarray, np.ndarray]:
    """A spectrum averaged in logarithmic frequency bins.

    Equal bins per decade, so the smoothing is uniform on the axis the spectrum is read
    on: at 0.01 Hz a bin is a few samples wide and the estimate is barely touched, while
    at 10 Hz it averages thousands and the periodogram's chi-squared scatter falls away.

    Parameters
    ----------
    frequency_hz, amplitude : ndarray
        The raw spectrum, positive frequencies only.
    bins : int, optional
        How many bins across the whole range.

    Returns
    -------
    tuple of ndarray
        Bin-centre frequency and mean amplitude, with empty bins dropped.
    """
    edges = np.logspace(np.log10(frequency_hz[0]), np.log10(frequency_hz[-1]), bins + 1)
    index = np.clip(np.searchsorted(edges, frequency_hz, side="right") - 1, 0, bins - 1)
    total = np.bincount(index, weights=amplitude, minlength=bins)
    count = np.bincount(index, minlength=bins)
    filled = count > 0
    centres = np.sqrt(edges[:-1] * edges[1:])
    return centres[filled], total[filled] / count[filled]


def _reference_corner_hz(magnitude: float) -> float:
    """:data:`REFERENCE_CORNER_HZ`, carried to another magnitude by self-similarity.

    A constant stress drop puts the corner at ``f_c ~ M_0^(-1/3)``, so a magnitude step
    moves it by ``10 ** (-magnitude_step / 2)``. Exact at
    :data:`~curvature.model.MAGNITUDE`, where it returns the published number unchanged.

    Parameters
    ----------
    magnitude : float

    Returns
    -------
    float
    """
    return REFERENCE_CORNER_HZ * 10.0 ** (-(magnitude - model.MAGNITUDE) / 2.0)


def moment_rate(plate: Plate) -> None:
    """The moment rate function and its amplitude spectrum, both models.

    The spectrum is the panel the comparison lives or dies on. It is drawn log-log with
    the two frequencies that bound where it can be believed marked: the corner of the
    plate's own event, and the ~6 Hz where a 500 m subfault crossed at 3 km/s stops
    resolving anything. Between those two the curves are the measurement; outside them
    they are the discretisation.
    """
    arrays, results = plate.arrays, plate.results
    figure, panels = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for column, name in enumerate(("central_constant", "central_standard")):
        times = arrays[f"mrf_times_{name}_curved_s"]
        for geometry, colour in (("curved", style.CURVED), ("flat", style.FLAT)):
            panels[0, column].plot(
                arrays[f"mrf_times_{name}_{geometry}_s"],
                arrays[f"mrf_rate_{name}_{geometry}_reported_nm_s"] / 1.0e19,
                color=colour,
                label=f"{geometry} interface",
                linewidth=1.4,
            )
            frequency = arrays[f"spectrum_frequency_{name}_{geometry}_hz"]
            amplitude = arrays[f"spectrum_amplitude_{name}_{geometry}_nm"]
            inside = frequency > 0
            # The raw periodogram of a *single* realisation is noise above about 1 Hz --
            # one draw gives one chi-squared sample per frequency -- so drawing it as a
            # solid line would claim a precision the estimate does not have. It is kept
            # underneath at low opacity, and the line is the log-binned mean, which is
            # how a source spectrum is read. The corner and falloff in `results.json`
            # come from the raw spectrum, not from this smoothing.
            panels[1, column].loglog(
                frequency[inside],
                amplitude[inside],
                color=colour,
                linewidth=0.5,
                alpha=0.25,
            )
            binned_hz, binned = _log_binned(frequency[inside], amplitude[inside])
            panels[1, column].loglog(
                binned_hz,
                binned,
                color=colour,
                label=f"{geometry} interface",
                linewidth=1.8,
            )
        panels[0, column].set_xlabel("Time since first onset (s)")
        panels[0, column].set_ylabel("Moment rate (1e19 N m / s)")
        panels[0, column].set_title(SCENARIO_LABELS[name])
        panels[0, column].set_xlim(times[0], times[-1])

        for at, note in (
            (_reference_corner_hz(plate.magnitude), f"Mw {plate.magnitude:g} corner"),
            (6.0, "500 m subfault limit"),
        ):
            panels[1, column].axvline(at, color=style.AXIS, linewidth=1.0)
            panels[1, column].annotate(
                note,
                (at, 0.5),
                xycoords=("data", "axes fraction"),
                xytext=(-4, 0),
                textcoords="offset points",
                fontsize=7,
                color=style.MUTED,
                rotation=90,
                rotation_mode="anchor",
                ha="left",
                va="bottom",
            )
        panels[1, column].set_xlabel("Frequency (Hz)")
        panels[1, column].set_ylabel("|Moment rate spectrum| (N m)")
        entry = results["scenarios"][name]
        panels[1, column].set_title(
            f"Falloff {entry['high_frequency_slope_curved']:.2f} curved, "
            f"{entry['high_frequency_slope_flat']:.2f} flat"
        )
        panels[1, column].set_xlim(1.0e-3, 25.0)
    panels[0, 0].legend(loc="upper right")
    _save(figure, plate.figure("moment_rate"))
    plt.close(figure)


def resolution(plate: Plate) -> None:
    """Whether the geometry this study measures depends on the mesh it is measured on.

    Two ladders with opposite answers, which is why they are on one figure. Subdividing a
    mesh cannot move the surface at all -- the points land on the faces they came from --
    so the area ratio is flat to twelve digits. Rebuilding at a finer spacing does move
    it, because the boundary staircase and the chording across the source's own kinks are
    both first order in the spacing. The study's 2 km build sits 0.05% below the limit.
    """
    ladder = plate.results["resolution"]["by_build_spacing"]
    spacings = [float(name.split()[1]) for name in ladder]
    ratios = [ladder[name]["area_ratio_true_over_projected"] for name in ladder]
    subdivision = plate.results["resolution"]["by_subdivision"]

    figure, panels = plt.subplots(1, 2, figsize=(10.5, 4.2))
    panels[0].plot(spacings, ratios, marker="o", color=style.CURVED)
    panels[0].set_xscale("log")
    panels[0].set_xticks(spacings, [f"{value:g}" for value in spacings])
    panels[0].minorticks_off()
    panels[0].set_xlabel("Spacing the mesh was built at (km)")
    panels[0].set_ylabel("True area / projected area")
    panels[0].set_title("Rebuilding finer does move the geometry")
    panels[0].invert_xaxis()
    panels[0].margins(y=0.18)
    for spacing, ratio in zip(spacings, ratios, strict=True):
        panels[0].annotate(
            f"{ratio:.6f}",
            (spacing, ratio),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=7,
            color=style.SECONDARY_INK,
            ha="center",
        )

    edges = [entry["median_edge_km"] * 1000.0 for entry in subdivision.values()]
    values = [entry["area_ratio_true_over_projected"] for entry in subdivision.values()]
    panels[1].plot(edges, values, marker="o", color=style.FLAT)
    panels[1].set_xscale("log")
    panels[1].set_xticks(edges, [f"{value:.0f}" for value in edges])
    panels[1].minorticks_off()
    panels[1].invert_xaxis()
    panels[1].set_xlabel("Median edge length after subdivision (m)")
    panels[1].set_ylabel("True area / projected area")
    panels[1].set_title("Subdividing the same mesh does not")
    # The same window as the left panel, so "flat" is flat against a scale on which the
    # rebuild effect is visible rather than against one auto-fitted to round-off.
    panels[1].set_ylim(*panels[0].get_ylim())
    span = max(values) - min(values)
    panels[1].annotate(
        f"spread {span:.1e} across an eightfold refinement,\nagainst "
        f"{max(ratios) - min(ratios):.1e} across the rebuild",
        (0.5, 0.12),
        xycoords="axes fraction",
        fontsize=8,
        color=style.SECONDARY_INK,
        ha="center",
    )
    _save(figure, plate.figure("resolution"))
    plt.close(figure)


def main() -> None:
    """Draw everything, for every interface that has been run."""
    style.apply()
    FIGURES.mkdir(parents=True, exist_ok=True)

    for plate in _plates():
        depth_error(plate)
        sections(plate)
        moment(plate)
        onset_polar(plate)
        onset_polar_control(plate)
        onset_maps(plate)
        true_depth(plate)
        true_depth_moment(plate)
        if plate.decomposed():
            decomposition_by_site(plate)
        slip_maps(plate)
        correlation(plate)
        moment_rate(plate)
        if "resolution" in plate.results:
            resolution(plate)
        plan_view(plate)

    for path in sorted(FIGURES.glob("*.png")):
        print(f"{path.name}: {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
