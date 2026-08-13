"""``rupture-generator view``: watch a rupture happen."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer
import xarray as xr
from scipy.spatial import KDTree

from rupture_generator.formats import Format, from_path
from rupture_generator.formats.rupture import read_rupture, segments_in
from rupture_generator.moment import cumulative_moment, moment_rate, rigidity_pa
from rupture_generator.scripts.errors import console
from rupture_generator.triangular.mesh import SCHEMA_VERSION as TRIANGULAR_SCHEMA
from rupture_generator.triangular.mesh import from_datatree, remesh
from rupture_generator.units import M2_PER_KM2

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from rupture_generator.triangular.mesh import TriangleMesh

    # A colour map already closed over its limits -- `hot` or `viridis` with the same
    # `low` and `high` the fault is drawn at, so a bar and a patch of fault showing the
    # same value show the same colour.
    Colouring = Callable[[np.ndarray], np.ndarray]

FIELDS = {
    "slip": ("slip_m", "metres", "hot"),
    "rise-time": ("rise_time_s", "seconds", "viridis"),
    "rake": ("rake_deg", "degrees", "quiver"),
}
"""Which variable each field view shows, its unit, and how it is drawn.

Slip gets **hot** because it is a magnitude with a meaningful zero: black where the
fault did not move, and brightening monotonically in luminance, so the asperities read
as bright patches at a glance. Rise time gets **viridis** because it has no meaningful
zero and wants a perceptually even ramp rather than an implied floor. Rake is an
*angle*, so no colour ramp is honest for it at all -- a cyclic quantity on a linear
scale puts 179 and -179 degrees at opposite ends -- and it is drawn as arrows instead.
"""

ANIMATED = "slip"
"""The only field that is a function of time. See the module note."""

DEFAULT_CELL_BUDGET = 50_000
"""How many cells to draw before striding the display down.

Chosen from what a viewer can usefully show rather than from what it can survive: at
50,000 cells a 1,600-pixel window has about six pixels per subfault, so striding
further would discard detail the screen could have resolved, and drawing more would
paint several subfaults into one pixel.
"""

TRIANGLES_PER_CELL = 2
"""How many display triangles one cell of :data:`DEFAULT_CELL_BUDGET` buys.

The budget is stated in *cells*, which on the structured track are quads, and two
triangles tile the footprint of one quad. So a triangulation drawn at twice the cell
budget puts the same number of pixels on each drawn face, and one number stays
comparable across both tracks rather than needing a second nobody can weigh against
the first.

It is a budget for the **animation** rather than for the renderer: the slip mesh is
logged again at every frame, so a million faces is not a slow first draw but a
gigabyte of mesh per second of timeline.
"""

PULSE_BLOCK_SAMPLES = 8_000_000
"""How many slip-rate samples to hold at once while carrying pulses onto the display.

The pulses are the only part of a rupture file that does not fit: the shipped 525 m
Hikurangi ruptures carry 400 M samples (3.2 GB of ``f64``) and the 400 m one 2.45 G
(19.6 GB), against 30 GB of machine. Everything else a viewer needs -- vertices, faces,
and one value per subfault per field -- is under 250 MB at 2.15 M faces, which is why
:func:`load` reads those whole and blocks only this.

A block of 8 M samples is 64 MB of ``f64``, and :func:`_carry_pulses` holds four arrays
of that size while it works on one. That is not what sets the peak: the aggregate it is
building is, at 0.24 GB for the 525 m rupture and 0.98 GB for the 400 m one, plus one
scatter temporary of the same size again. So the block size trades only against the
per-block overhead, and 8 M is far enough above the point where that matters -- the
whole 400 M sample pass takes 8.3 s, against 0.9 s to assign the subfaults and 55 s to
build the surface they are drawn on.
"""

AXIS_LINE = (90, 96, 104)
"""Grey enough to read as furniture rather than as data."""

AXIS_TEXT = (196, 202, 210)
"""A label is text: Rerun draws it in the entity's own colour, so a transparent point
carries an invisible label -- the background chip renders and the glyphs do not."""

MAX_ARROWS = 3_000
"""How many rake arrows to draw. Beyond this they overlap into a solid mass and stop
being readable as directions."""


@dataclasses.dataclass(frozen=True)
class Subfaults:
    """What the file says, at its own resolution, once the drawn mesh is coarser.

    A triangular rupture is drawn on a surface built for the screen (see
    :func:`decimate`), so the drawn cells are not the file's subfaults and a panel that
    *counts* them -- the statistics table, the field distributions -- would be
    describing the picture rather than the earthquake. These are the arrays those
    panels read instead. One field per array, no geometry: 66 MB at 1.39 M faces, which
    is the price of the histograms being about the rupture.
    """

    slip_m: np.ndarray
    rise_time_s: np.ndarray
    onset_s: np.ndarray
    rake_deg: np.ndarray
    area_m2: np.ndarray
    rigidity_pa: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class Segment:
    """One fault segment, in the local metric frame the viewer draws in.

    Every input format reduces to this, which is what lets everything below be written
    once. Positions are **metres east, north and up** from the rupture's own centroid:
    up rather than down because a viewer's vertical axis points up and every file
    format measures depth downwards.

    Attributes
    ----------
    cells : tuple of int, or None
        The lattice shape a structured file was stored on, which is what
        :func:`strided` skips through. ``None`` for a triangulation, which has no
        lattice to skip through and arrives already decimated -- see :func:`decimate`.
    corners_m : np.ndarray
        ``(cells, corners, 3)``, anticlockwise from the shallow near corner. Four
        corners from a quad lattice and three from a triangulation; nothing below reads
        the count, it fans whatever it is given. Unshared between cells on purpose --
        see the module note.
    rigidity_pa : np.ndarray or None
        Present when the file carries the material properties the generator sampled.
        An SRF version 2.0 and a version 3 rupture do; a version 2 rupture does not,
        and the moment release is then drawn at a nominal rigidity with the panel
        saying so.
    subfaults : Subfaults or None
        The file's own values where this segment has been decimated for the screen, and
        ``None`` where it has not -- in which case the segment *is* what the file says
        and :attr:`population` returns it.
    """

    name: str
    cells: tuple[int, int] | None
    corners_m: np.ndarray
    centres_m: np.ndarray
    slip_m: np.ndarray
    rise_time_s: np.ndarray
    onset_s: np.ndarray
    rake_deg: np.ndarray
    area_m2: np.ndarray
    strike_deg: np.ndarray
    dip_deg: np.ndarray
    pulse_offsets: np.ndarray
    pulse_samples: np.ndarray
    sample_interval_s: float
    hypocentre_m: np.ndarray | None = None
    rigidity_pa: np.ndarray | None = None
    subfaults: Subfaults | None = None

    def values(self, variable: str) -> np.ndarray:
        """One field, flattened over subfaults."""
        return getattr(self, variable).ravel()

    @property
    def population(self) -> Segment | Subfaults:
        """The subfaults the file holds, which is this segment unless it was decimated.

        Returns
        -------
        Segment or Subfaults
            Either way it carries ``slip_m``, ``rise_time_s``, ``onset_s``,
            ``rake_deg``, ``area_m2`` and ``rigidity_pa`` under those names, so a caller
            asking what the *rupture* looks like reads the same attributes whichever it
            gets.
        """
        return self.subfaults if self.subfaults is not None else self


# ============================================================================
# Colour
# ============================================================================


def viridis(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Perceptually even, and the right choice for a field with no meaningful zero."""
    return _ramp(_VIRIDIS_16, values, low, high)


def hot(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Black through red and yellow to white.

    Built from the definition rather than a table: the red channel saturates over the
    first three eighths, green over the next three, and blue only in the last quarter.
    Luminance rises monotonically, which is what makes a zero read as absent rather
    than as merely low.
    """
    fraction = _fraction(values, low, high)
    red = np.clip(fraction / 0.365, 0.0, 1.0)
    green = np.clip((fraction - 0.365) / 0.381, 0.0, 1.0)
    blue = np.clip((fraction - 0.746) / 0.254, 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).round().astype(np.uint8)


def _fraction(values: np.ndarray, low: float, high: float) -> np.ndarray:
    span = high - low
    if span <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - low) / span, 0.0, 1.0)


def _ramp(
    anchors: np.ndarray, values: np.ndarray, low: float, high: float
) -> np.ndarray:
    position = _fraction(values, low, high) * (len(anchors) - 1)
    lower = np.floor(position).astype(int)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weight = (position - lower)[:, None]
    blended = anchors[lower] * (1.0 - weight) + anchors[upper] * weight
    return blended.round().astype(np.uint8)


_VIRIDIS_16 = np.array(
    [
        (68, 1, 84),
        (71, 24, 106),
        (72, 45, 117),
        (69, 65, 125),
        (64, 84, 131),
        (57, 102, 135),
        (50, 119, 138),
        (44, 136, 139),
        (39, 152, 138),
        (39, 168, 133),
        (54, 183, 122),
        (85, 197, 104),
        (124, 208, 80),
        (169, 217, 51),
        (216, 222, 26),
        (253, 231, 37),
    ],
    dtype=np.float64,
)

COLOURMAPS = {"hot": hot, "viridis": viridis}


# ============================================================================
# Loading -- both formats reduce to Segment
# ============================================================================


def load(path: Path, max_cells: int = DEFAULT_CELL_BUDGET) -> tuple[list[Segment], str]:
    """Read a rupture, whatever it is written as.

    Parameters
    ----------
    path : Path
        A native rupture file of either schema, or a text SRF.
    max_cells : int, optional
        How many cells the drawn mesh may have. Only a version 3 file uses it here --
        a triangulation is decimated on the way in, because that is what bounds the
        pulses this has to read at all -- and the structured tracks are strided at
        draw time instead.

    Returns
    -------
    tuple
        The segments, and a sentence saying what was read and how faithfully.

    Raises
    ------
    ValueError
        For a format this cannot read, a native file that holds no rupture, or a
        version 3 file that holds a fault surface with nothing drawn on it.
    """
    chosen = from_path(path)
    if chosen in (Format.NETCDF, Format.ZARR):
        with read_rupture(path) as tree:
            # Dispatched on what the file says it is, not on which variables it
            # happens to carry: the two schemas share most of their vocabulary, so
            # guessing reads a triangulation as a lattice and fails somewhere else.
            version = int(tree.attrs.get("schema_version", 1))
            if version >= TRIANGULAR_SCHEMA:
                return _from_triangular_file(tree, max_cells)
            return _from_rupture_file(tree), "native rupture file"
    if chosen is Format.SRF:
        return _from_srf(path), "SRF, mesh reconstructed from subfault centres"
    raise ValueError(
        f"a rupture cannot be read from {chosen.value}: this reads the native format "
        "and text SRF"
    )


def _local_frame(
    positions: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """A common origin for every segment, and the stacked positions relative to it."""
    stacked = np.concatenate([p.reshape(-1, 3) for p in positions])
    return stacked.mean(axis=0), stacked


def _from_rupture_file(tree: xr.DataTree) -> list[Segment]:
    """The structured format, where the nodes are stored and nothing is reconstructed."""
    segments: list[Segment] = []
    found = segments_in(tree)
    if not found:
        raise ValueError("this file holds no rupture")

    origin = None
    for name, dataset in found:
        east = (
            dataset["node_east_km"].to_numpy() + float(dataset.attrs["origin_east_km"])
        ) * 1000.0
        north = (
            dataset["node_north_km"].to_numpy()
            + float(dataset.attrs["origin_north_km"])
        ) * 1000.0
        up = -dataset["node_depth_km"].to_numpy() * 1000.0
        nodes = np.stack([east, north, up], axis=-1)
        if origin is None:
            origin = np.array([nodes[..., 0].mean(), nodes[..., 1].mean(), 0.0])

        corners = (
            np.stack(
                [
                    nodes[:-1, :-1],
                    nodes[:-1, 1:],
                    nodes[1:, 1:],
                    nodes[1:, :-1],
                ],
                axis=2,
            ).reshape(-1, 4, 3)
            - origin
        )

        hypocentre = None
        if "hypocentre_strike_km" in dataset.attrs:
            hypocentre = _hypocentre_position(dataset, nodes, origin)

        segments.append(
            Segment(
                name=name,
                cells=(dataset.sizes["i"], dataset.sizes["j"]),
                corners_m=corners,
                centres_m=corners.mean(axis=1),
                slip_m=dataset["slip_m"].to_numpy(),
                rise_time_s=dataset["rise_time_s"].to_numpy(),
                onset_s=dataset["onset_s"].to_numpy(),
                rake_deg=dataset["rake_deg"].to_numpy(),
                area_m2=dataset["area_m2"].to_numpy(),
                strike_deg=dataset["strike_deg"].to_numpy(),
                dip_deg=dataset["dip_deg"].to_numpy(),
                pulse_offsets=dataset["slip_rate_offset"].to_numpy(),
                pulse_samples=dataset["slip_rate"].to_numpy(),
                rigidity_pa=dataset["rigidity_pa"].to_numpy(),
                sample_interval_s=float(dataset.attrs["sample_interval_s"]),
                hypocentre_m=hypocentre,
            )
        )
    return segments


def _hypocentre_position(
    dataset: xr.Dataset, nodes: np.ndarray, origin: np.ndarray
) -> np.ndarray:
    """Where the rupture nucleated, from the arc lengths the file records.

    The file stores the hypocentre the way the config states it -- as distances along
    strike and down dip -- so this walks the same arc lengths back to a position,
    which is the one conversion that keeps the marker on the cell the pipeline
    actually seeded.
    """
    strike_arc = dataset["strike_km"].to_numpy()
    dip_arc = dataset["dip_km"].to_numpy()
    j = int(
        np.searchsorted(
            strike_arc[1:-1], dataset.attrs["hypocentre_strike_km"], "right"
        )
    )
    i = int(np.searchsorted(dip_arc[1:-1], dataset.attrs["hypocentre_dip_km"], "right"))
    cell = nodes[i : i + 2, j : j + 2].reshape(-1, 3).mean(axis=0)
    return cell - origin


# ============================================================================
# The triangular track: a rupture drawn on a coarser surface of its own
# ============================================================================


def _from_triangular_file(
    tree: xr.DataTree, max_cells: int
) -> tuple[list[Segment], str]:
    """A version 3 rupture: a triangulation, redrawn at a resolution a screen can hold.

    Nothing here reconstructs geometry -- the file stores the vertices and the faces --
    but everything is *decimated*, and that is the difference from the structured path.
    A shipped 525 m Hikurangi rupture is 1.39 M faces and its pulses are 3.2 GB, so the
    drawn surface is rebuilt at a display resolution by :func:`decimate` and the file's
    own values are carried onto it: the picture is coarse, the numbers behind the
    panels are not.

    Measured end to end, under ``ulimit -v 12000000`` on a 30 GB machine, at the
    default 50,000 cell budget:

    ==================  =========  =========  =======  ========  ======  =====
    rupture             on disk    faces      drawn    pulses    load    peak
    ==================  =========  =========  =======  ========  ======  =====
    Hikurangi 525 m     3.4 GB     1,389,600  95,894   400 M     90 s    1.2 GB
    Hikurangi 400 m     20.0 GB    2,151,936  95,776   2.45 G    206 s   2.2 GB
    ==================  =========  =========  =======  ========  ======  =====

    The peak is a fifth of the smaller file and a ninth of the larger, which is the
    point: nothing here is proportional to the pulses.

    Returns
    -------
    tuple
        The segments, and a sentence naming both resolutions -- because a viewer
        drawing something other than what the file holds has to say so.
    """
    meshes, _ = from_datatree(tree)
    parts = [
        (surface if len(segments) == 1 else f"{surface}:{index}", mesh)
        for surface, segments in meshes.items()
        for index, mesh in enumerate(segments)
    ]
    if not parts:
        raise ValueError("this file holds no rupture")

    # A version 3 *mesh* file and a version 3 rupture are the same layout with and
    # without the fields, so the difference has to be looked for rather than inferred
    # from the schema -- and saying which field is missing is what tells a reader
    # they are looking at a surface nothing has been drawn on yet.
    wanted = ("slip_m", "rise_time_s", "onset_s", "rake_deg")
    for name, mesh in parts:
        missing = [field for field in wanted if field not in mesh]
        if missing:
            raise ValueError(
                f"segment {name!r} carries no {', '.join(missing)}, so this is a fault "
                "surface rather than a rupture on one. Generate a rupture first"
            )

    # The budget is for the picture, so it is shared out between segments the same way
    # `MAX_ARROWS` is: a twenty-segment fault system draws twenty coarser meshes rather
    # than twenty full ones.
    budget = max(TRIANGLES_PER_CELL * max_cells // len(parts), 1)

    segments: list[Segment] = []
    origin: np.ndarray | None = None
    for name, mesh in parts:
        segment, origin = _triangular_segment(mesh, name, budget, origin)
        segments.append(segment)

    drawn = sum(len(segment.corners_m) for segment in segments)
    subfaults = sum(mesh.face_count for _, mesh in parts)
    if drawn == subfaults:
        return segments, f"native triangular rupture file, {subfaults:,} faces"
    return segments, (
        f"native triangular rupture file, {subfaults:,} faces redrawn on {drawn:,}"
    )


def _triangular_segment(
    mesh: TriangleMesh, name: str, budget: int, origin: np.ndarray | None
) -> tuple[Segment, np.ndarray]:
    """One segment of a version 3 file, decimated, with its fields carried across.

    **How each field crosses is a decision per field, not one rule.**

    - **Slip** is the area-weighted mean of the subfaults in a display cell, and the
      cell's area is their total. That pair is what makes the drawn rupture carry the
      file's moment exactly rather than drifting with the display resolution, and
      `test_a_decimated_rupture_carries_the_files_moment` asserts it. Measured on a
      shipped 525 m rupture, 1.39 M faces onto 98,468: the drawn moment is the file's to
      2.2e-16 relative, and the total area to round-off. What the mean *does* cost is
      the extremes, and they are reported rather than hidden -- peak slip 2.399 m drawn
      against 2.438 m in the file, which is why the statistics panel and both static
      histograms read :attr:`Segment.population` instead.
    - **Rigidity** is weighted by ``area x slip`` rather than by area, which is the
      weight that makes ``mu_cell A_cell slip_cell`` equal the subfaults' own summed
      moment when rigidity varies across the cell. Where a cell did not slip there is
      no moment to preserve and it falls back to the area-weighted mean.
    - **Onset** is the *earliest* of the cell's subfaults: when the front reaches the
      cell is when it reaches the first of them. It is a real subfault's onset rather
      than an average, it is what the aggregated pulse below is measured from, and
      unlike a nearest-face pick it can never report the front arriving after the cell
      has already started to slip.
    - **Rake, rise time, strike and dip** are the nearest subfault's, so no drawn cell
      shows a value no subfault had -- the property the strided path kept, kept here.
      Averaging a rake is meaningless at the wrap and averaging a rise time smears the
      quantity being looked at.

    Returns
    -------
    tuple
        The segment, and the viewer's shared origin -- established by the first segment
        so that every later one lands beside it rather than about its own centre.
    """
    origin_km = mesh.origin_km
    if origin is None:
        vertices = mesh.vertices_km()
        origin = np.array(
            [
                (vertices[:, 0].mean() + origin_km[0]) * 1000.0,
                (vertices[:, 1].mean() + origin_km[1]) * 1000.0,
                0.0,
            ]
        )

    areas_m2 = mesh.areas_km2() * M2_PER_KM2
    slip_m = np.asarray(mesh["slip_m"], dtype=np.float64)
    rigidity = (
        np.asarray(mesh["rigidity_pa"], dtype=np.float64)
        if "rigidity_pa" in mesh
        else None
    )
    subfaults = Subfaults(
        slip_m=slip_m,
        rise_time_s=np.asarray(mesh["rise_time_s"], dtype=np.float64),
        onset_s=np.asarray(mesh["onset_s"], dtype=np.float64),
        rake_deg=np.asarray(mesh["rake_deg"], dtype=np.float64),
        area_m2=areas_m2,
        rigidity_pa=rigidity,
    )

    display, cell_of_face = decimate(mesh, budget)
    cells = display.face_count
    weight = np.bincount(cell_of_face, weights=areas_m2, minlength=cells)
    # A display cell with no subfault of its own is possible where the drawn surface
    # reaches past the file's -- it carries no area, no slip and no pulse, and dividing
    # by its area would put a NaN on the picture instead of nothing.
    safe = np.where(weight > 0.0, weight, 1.0)

    def carried(values: np.ndarray) -> np.ndarray:
        """A field's area-weighted mean over the subfaults in each display cell."""
        return (
            np.bincount(cell_of_face, weights=areas_m2 * values, minlength=cells) / safe
        )

    drawn_slip = carried(slip_m)
    drawn_rigidity = None
    if rigidity is not None:
        moment = np.bincount(
            cell_of_face, weights=areas_m2 * slip_m * rigidity, minlength=cells
        )
        slipped = weight * drawn_slip
        drawn_rigidity = np.where(
            slipped > 0.0,
            moment / np.where(slipped > 0.0, slipped, 1.0),
            carried(rigidity),
        )

    # The earliest onset in each cell, and the closest subfault to each cell's centre.
    onset_s = np.full(cells, np.inf)
    np.minimum.at(onset_s, cell_of_face, subfaults.onset_s)
    onset_s = np.where(np.isfinite(onset_s), onset_s, float(subfaults.onset_s.min()))
    nearest = KDTree(mesh.centres()).query(display.centres())[1]

    strike_deg, dip_deg = mesh.strike_dip_deg()
    corners_km = display.vertices_km()[display.faces()]
    corners_m = _view_metres(corners_km, origin_km, origin)

    offsets, samples = _carry_pulses(
        mesh, cell_of_face, onset_s, areas_m2 / safe[cell_of_face]
    )

    return (
        Segment(
            name=name,
            cells=None,
            corners_m=corners_m,
            centres_m=corners_m.mean(axis=1),
            slip_m=drawn_slip,
            rise_time_s=subfaults.rise_time_s[nearest],
            onset_s=onset_s,
            rake_deg=subfaults.rake_deg[nearest],
            area_m2=weight,
            strike_deg=strike_deg[nearest],
            dip_deg=dip_deg[nearest],
            pulse_offsets=offsets,
            pulse_samples=samples,
            sample_interval_s=float(mesh.attrs["sample_interval_s"]),
            hypocentre_m=_triangular_hypocentre(mesh, origin),
            rigidity_pa=drawn_rigidity,
            # Only where the drawn cells are not the file's faces. Undecimated, the
            # segment *is* what the file says and a second copy of it would be one
            # more thing that could disagree with the picture.
            subfaults=subfaults if display is not mesh else None,
        ),
        origin,
    )


def _view_metres(
    points_km: np.ndarray, origin_km: tuple[float, float], origin_m: np.ndarray
) -> np.ndarray:
    """``(..., 3)`` chart positions as metres east, north and **up** about the viewer.

    The one place the two conventions meet: a chart holds offsets from its surface
    origin in kilometres with depth positive downwards, and a viewer's vertical axis
    points up.
    """
    points = np.asarray(points_km, dtype=np.float64)
    return (
        np.stack(
            [
                points[..., 0] + origin_km[0],
                points[..., 1] + origin_km[1],
                -points[..., 2],
            ],
            axis=-1,
        )
        * 1000.0
        - origin_m
    )


def display_spacing_km(area_km2: float, faces: int) -> float:
    """The lattice spacing that cuts a surface of ``area_km2`` into about ``faces``.

    :func:`~rupture_generator.triangular.mesh.remesh` cuts the parameter domain into
    squares of the spacing it is given and splits each into two triangles, so
    ``faces = 2 A / d**2`` and the spacing is that inverted. Solved rather than
    searched for, and the answer is an over-estimate of the face count by exactly the
    metric factor ``sqrt(1 + |grad h|**2)`` -- the surface is larger than its own
    projection -- so the realised mesh comes in *under* budget, which is the safe side.
    Measured on the shipped 525 m Hikurangi rupture: 1.89 km asked for 100,000 faces
    and got 95,894.

    Parameters
    ----------
    area_km2 : float
        The surface's own area.
    faces : int
        How many triangles to aim for.

    Returns
    -------
    float
        Kilometres.
    """
    return math.sqrt(TRIANGLES_PER_CELL * area_km2 / faces)


def decimate(mesh: TriangleMesh, faces: int) -> tuple[TriangleMesh, np.ndarray]:
    """A coarser surface to draw, and which of its faces each subfault belongs to.

    **A triangulation cannot be strided.** Taking every seventh cell of a lattice --
    which is what :func:`strided_corners` does, and what a 2.15 M face rupture asks for
    at any usable budget -- leaves a coarser *lattice*, because a quad block has
    corners to span. Taking every 49th triangle leaves 49th of a surface: holes, not a
    coarser fault. So the display mesh is **built** rather than sieved, by
    :func:`~rupture_generator.triangular.mesh.remesh`, which lifts a lattice at the
    display spacing onto the file's own surface. Every drawn vertex lies exactly on the
    fault, the outline is the file's resolved to one display cell, and there are no
    holes.

    The subfaults are then assigned to display cells by **nearest centre**, which makes
    the assignment a partition: every subfault lands in exactly one cell, so a sum over
    display cells is a sum over the file's subfaults and the moment cannot leak. It
    costs 0.87 s for 1.39 M subfaults against 95,894 cells, which is nothing beside the
    remeshing.

    **The remeshing is what this costs**, and it is worth saying where the time goes
    rather than leaving it to be discovered: 54.9 s of the 90 s it takes to open a
    shipped 525 m rupture, and 2.7 minutes of the 3.4 for the 400 m one. `remesh` scans
    its *source* faces in a Python loop -- which is cheap for the pipeline, whose source
    is a 5,218-face CFM interface, and is the whole cost here, where the source is 1.39
    M faces and the target is 95 k. Vectorising that loop is the one change that would
    make this fast, and it belongs in ``triangular/mesh.py`` rather than here.

    Parameters
    ----------
    mesh : TriangleMesh
        The file's own segment.
    faces : int
        The display budget, in triangles.

    Returns
    -------
    tuple
        The display mesh, and ``(F,)`` display-cell indices, one per subfault. A mesh
        already inside the budget is returned unchanged, with the identity assignment,
        so a small rupture is drawn exactly as it was stored.

    Raises
    ------
    ValueError
        If the display mesh cannot be built, naming the spacing it was asked for.
        Drawing the file's own faces instead is not the fallback it looks like: the
        slip mesh is logged once per frame, so a million faces is a gigabyte of mesh
        per second of timeline.
    """
    if mesh.face_count <= faces:
        return mesh, np.arange(mesh.face_count)

    frame = mesh.frame
    spacing_km = display_spacing_km(float(mesh.areas_km2().sum()), faces)
    try:
        display = remesh(
            mesh.vertices_km(),
            mesh.faces(),
            spacing_km,
            strike_deg=frame.strike_deg,
            dip_deg=frame.dip_deg,
            origin_east_km=mesh.origin_km[0],
            origin_north_km=mesh.origin_km[1],
            surface=mesh.surface,
        )
    except ValueError as error:
        raise ValueError(
            f"{mesh.surface!r} cannot be drawn at {spacing_km:.3g} km, which is the "
            f"spacing a budget of {faces:,} faces asks of a surface of "
            f"{mesh.areas_km2().sum():,.0f} km2: {error}. Raise --max-cells"
        ) from error

    return display, KDTree(display.centres()).query(mesh.centres())[1]


def _carry_pulses(
    mesh: TriangleMesh,
    cell_of_face: np.ndarray,
    cell_onset_s: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Every subfault's slip-rate pulse, summed onto the display cell that holds it.

    A display cell's pulse is its subfaults' pulses, each weighted by its share of the
    cell's area and each laid down at its own onset relative to the cell's -- so the
    aggregate is the cell's mean slip rate through time, and integrates to exactly the
    area-weighted mean slip that :func:`_triangular_segment` draws. Nothing about the
    rupture's timing is averaged: a front sweeping across a display cell shows up as
    the cell's pulse being longer than any of its subfaults', which is what it is.

    **This is the only pass over the pulses, and it is blocked.** The rates arrive as
    an unread array (:attr:`~rupture_generator.triangular.mesh.TriangleMesh
    .pulse_rates`), and :data:`PULSE_BLOCK_SAMPLES` at a time are read, weighted and
    scattered. The aggregate is smaller than the file's by most of the decimation
    ratio: 30.6 M samples against 400 M on a shipped 525 m rupture and 122 M against
    2.45 G on the 400 m one, because a display cell holds one pulse where its ~15
    subfaults held one each. What it does not shrink by is the *whole* ratio -- the
    cell's pulse spans its subfaults' onsets as well as their rise times, which is
    exactly the front crossing it and is the thing worth watching.

    Onsets are quantised to the sample interval, exactly as
    :func:`~rupture_generator.moment.moment_rate` quantises them -- the same half-sample
    error, 0.01 s at the 0.02 s these files are written at, against onsets the model
    itself perturbs by ~0.35 s.

    Returns
    -------
    tuple
        The CSR ``(offsets, samples)`` of the display mesh's own pulses.

    Raises
    ------
    ValueError
        If the file carries no pulses, which is what a rupture generated with
        ``synthesise=False`` looks like: it has slip and onsets but never says how
        anything moved, so there is no animation to draw.
    """
    offsets, rates = mesh.pulse_offsets, mesh.pulse_rates
    if offsets is None or rates is None:
        raise ValueError(
            f"{mesh.surface!r} carries no slip-rate pulses, so there is nothing to "
            "animate. It was generated with `synthesise=False`; write it again with "
            "the pulse model, or view an SRF of it"
        )
    interval_s = float(mesh.attrs["sample_interval_s"])
    lengths = np.diff(offsets)

    # Where each subfault's pulse starts inside its cell's, in samples. Non-negative
    # because the cell's onset is the earliest of its own subfaults'.
    shift = np.rint(
        (np.asarray(mesh["onset_s"], dtype=np.float64) - cell_onset_s[cell_of_face])
        / interval_s
    ).astype(np.int64)

    reach = np.zeros(len(cell_onset_s), dtype=np.int64)
    np.maximum.at(reach, cell_of_face, shift + lengths)
    cell_offsets = np.concatenate([[0], np.cumsum(reach)])

    samples = np.zeros(int(cell_offsets[-1]), dtype=np.float64)
    # Sample `g` of subfault `i` lands at `base[i] + g` in the aggregate, which is the
    # whole of the scatter: no per-subfault Python loop over 1.39 M faces.
    base = cell_offsets[cell_of_face] + shift - offsets[:-1]
    for first, last in _face_blocks(offsets, PULSE_BLOCK_SAMPLES):
        low, high = int(offsets[first]), int(offsets[last])
        if high == low:
            continue
        block = np.asarray(rates[low:high], dtype=np.float64)
        block *= np.repeat(weight[first:last], lengths[first:last])
        target = np.repeat(base[first:last], lengths[first:last]) + np.arange(low, high)
        samples += np.bincount(target, weights=block, minlength=samples.size)

    return cell_offsets, samples


def _face_blocks(offsets: np.ndarray, block_samples: int) -> Iterator[tuple[int, int]]:
    """Face ranges holding about ``block_samples`` samples each.

    Cut on the pulses rather than on the faces, because a subfault's pulse is as long
    as its own rise time and the tapered edge of a fault has none at all: a fixed
    number of faces is a block whose size varies by orders of magnitude, and the
    budget is about bytes. A single subfault whose pulse is longer than the block is
    still read whole, since a pulse is the unit that can be placed.
    """
    faces = len(offsets) - 1
    first = 0
    while first < faces:
        last = (
            int(np.searchsorted(offsets, offsets[first] + block_samples, "right")) - 1
        )
        last = min(max(last, first + 1), faces)
        yield first, last
        first = last


def _triangular_hypocentre(mesh: TriangleMesh, origin: np.ndarray) -> np.ndarray | None:
    """Where the rupture nucleated, on a triangulation.

    The file records the hypocentre as the config states it -- arc lengths along strike
    and down dip -- so this asks the chart the same question the pipeline asked when it
    seeded the wavefront, :meth:`~rupture_generator.triangular.mesh.TriangleMesh
    .cell_index`, and takes the centre of the face that comes back. The marker is then
    on the subfault that nucleated rather than near it, and at the file's own
    resolution rather than the display's, which is the one place in the viewer where a
    cell's width would be a visible lie.
    """
    if "hypocentre_strike_km" not in mesh.attrs:
        return None
    face = mesh.cell_index(
        float(mesh.attrs["hypocentre_strike_km"]),
        float(mesh.attrs["hypocentre_dip_km"]),
    )
    corners = mesh.vertices_km()[mesh.faces()[face]]
    return _view_metres(corners.mean(axis=0), mesh.origin_km, origin)


def _from_srf(path: Path) -> list[Segment]:
    """An SRF, whose mesh is reconstructed from what the format does store.

    An SRF records subfault *centres*, not corners, so the quads here are built by
    stepping half a cell along strike and down dip from each centre, using the cell
    size the plane header implies. That is exact for the uniform planar grids an SRF
    describes and approximate for nothing else, which is the same assumption the
    format itself makes.

    **Not for a triangular rupture's SRF**, and there is nothing in the file that says
    so. `rupture_generator.triangular.assemble` writes one PLANE per segment with
    ``NSTK`` the triangle count and ``NDIP`` 1, whose length and width are a *summary*
    of the segment rather than a lattice -- so this would draw every subfault as a quad
    the full width of the fault. That header is inert to SW4 by design and there is no
    marker to detect it by, so the rule is the one MESH.md states: a triangular model is
    viewed from its own rupture file, where the faces are.
    """
    from rupture_generator.srf import read_srf

    srf = read_srf(path)
    points = srf.points

    latitude = np.asarray(points.latitude_deg, dtype=np.float64)
    longitude = np.asarray(points.longitude_deg, dtype=np.float64)
    depth_m = np.asarray(points.depth_km, dtype=np.float64) * 1000.0

    # A local tangent frame about the rupture's own centroid. Good to a few metres
    # over a fault system, which is far below what a viewer resolves.
    lat0, lon0 = float(latitude.mean()), float(longitude.mean())
    metres_per_degree = 111_320.0
    east = (longitude - lon0) * metres_per_degree * math.cos(math.radians(lat0))
    north = (latitude - lat0) * metres_per_degree
    centres = np.stack([east, north, -depth_m], axis=-1)

    segments: list[Segment] = []
    start = 0
    for index, plane in enumerate(srf.planes):
        count = plane.strike_count * plane.dip_count
        block = slice(start, start + count)
        start += count

        strike = np.radians(np.asarray(points.strike_deg[block], dtype=np.float64))
        dip = np.radians(np.asarray(points.dip_deg[block], dtype=np.float64))
        along = np.stack(
            [np.sin(strike), np.cos(strike), np.zeros_like(strike)], axis=-1
        )
        # Down dip, in east-north-up: the horizontal part turns ninety degrees off
        # strike and the vertical part goes down.
        down = np.stack(
            [
                np.sin(strike + np.pi / 2) * np.cos(dip),
                np.cos(strike + np.pi / 2) * np.cos(dip),
                -np.sin(dip),
            ],
            axis=-1,
        )
        half_length = plane.length_km * 1000.0 / plane.strike_count / 2.0
        half_width = plane.width_km * 1000.0 / plane.dip_count / 2.0

        middle = centres[block]
        corners = np.stack(
            [
                middle - along * half_length - down * half_width,
                middle + along * half_length - down * half_width,
                middle + along * half_length + down * half_width,
                middle - along * half_length + down * half_width,
            ],
            axis=1,
        )

        rise = np.asarray(points.rise_time_s[block], dtype=np.float64)
        rigidity = None
        if points.shear_speed_cm_s is not None and points.density_g_cm3 is not None:
            # cm/s back to km/s, and the density is already what rigidity_pa wants.
            rigidity = rigidity_pa(
                np.asarray(points.shear_speed_cm_s[block], dtype=np.float64) / 1.0e5,
                np.asarray(points.density_g_cm3[block], dtype=np.float64),
            )

        offsets = srf.slip_rate.indptr[block.start : block.stop + 1]
        samples = srf.slip_rate.data[offsets[0] : offsets[-1]] / 100.0

        segments.append(
            Segment(
                name=f"plane_{index}",
                cells=(plane.dip_count, plane.strike_count),
                corners_m=corners,
                centres_m=middle,
                slip_m=np.asarray(points.slip_cm[block], dtype=np.float64) / 100.0,
                rise_time_s=rise,
                onset_s=np.asarray(points.onset_s[block], dtype=np.float64),
                rake_deg=np.asarray(points.rake_deg[block], dtype=np.float64),
                area_m2=np.asarray(points.area_cm2[block], dtype=np.float64) / 1.0e4,
                strike_deg=np.asarray(points.strike_deg[block], dtype=np.float64),
                dip_deg=np.asarray(points.dip_deg[block], dtype=np.float64),
                pulse_offsets=offsets - offsets[0],
                pulse_samples=samples,
                sample_interval_s=float(points.sample_interval_s[block.start]),
                hypocentre_m=None,
                rigidity_pa=rigidity,
            )
        )
    return segments


# ============================================================================
# What the numbers say
# ============================================================================


NOMINAL_RIGIDITY_PA = 3.0e10
"""30 GPa, crustal rock. Used only where a file does not carry the rigidity the
generator sampled, and the panel says when that is."""


def statistics(segments: list[Segment]) -> str:
    """The numbers that say what kind of earthquake this is, and the moment release.

    Read off :attr:`Segment.population` rather than off the segments themselves, so
    every number here is a statement about the rupture the file holds -- its moment,
    its subfault count, its peak slip -- and not about the coarser surface a
    triangulation is drawn on. On the structured track the two are the same object.

    Returns
    -------
    str
        A markdown summary.
    """
    counted = [segment.population for segment in segments]
    exact = all(segment.rigidity_pa is not None for segment in counted)

    moment_nm = 0.0
    area_m2 = 0.0
    for segment in counted:
        rigidity = (
            segment.rigidity_pa
            if segment.rigidity_pa is not None
            else np.full(segment.slip_m.size, NOMINAL_RIGIDITY_PA)
        )
        moment_nm += float(
            np.sum(rigidity.ravel() * segment.area_m2.ravel() * segment.slip_m.ravel())
        )
        area_m2 += float(np.sum(segment.area_m2))

    magnitude = (math.log10(moment_nm) - 9.0499505) / 1.5 if moment_nm > 0 else 0.0

    starts = [float(segment.onset_s.min()) for segment in counted]
    # When the last subfault *stops* slipping, not when the last one starts: the
    # duration of the earthquake rather than of the front's travel.
    ends = [float((segment.onset_s + segment.rise_time_s).max()) for segment in counted]
    duration_s = max(ends) - min(starts)

    slip = np.concatenate([segment.slip_m.ravel() for segment in counted])
    caveat = (
        ""
        if exact
        else f" (at a nominal μ = {NOMINAL_RIGIDITY_PA / 1e9:.0f} GPa because rupture does not carry rigidity) "
    )
    lines = [
        "| | |",
        "| --- | --- |",
        f"| moment magnitude | **{magnitude:.2f}** {caveat}|",
        f"| seismic moment | {moment_nm:.3e} N m |",
        f"| fault area | {area_m2 / 1.0e6:,.0f} km² |",
        f"| rupture duration | {duration_s:.1f} s |",
        f"| segments | {len(segments)} |",
        f"| subfaults | {slip.size:,} |",
        f"| mean slip | {slip.mean():.2f} m |",
        f"| peak slip | {slip.max():.2f} m |",
        "",
    ]
    return "\n".join(lines)


def moment_release(
    segments: list[Segment], times_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Moment rate and cumulative moment over the display times.

    Both come from the same pulses, and the second is the running integral of the
    first, so a viewer comparing them is looking at one quantity two ways rather than
    at two computations that might disagree.
    """
    total = np.zeros_like(times_s)
    for segment in segments:
        rigidity = (
            segment.rigidity_pa.ravel()
            if segment.rigidity_pa is not None
            else np.full(segment.slip_m.size, NOMINAL_RIGIDITY_PA)
        )
        times, rate = moment_rate(
            segment.pulse_offsets,
            segment.pulse_samples,
            segment.onset_s.ravel(),
            segment.area_m2.ravel(),
            rigidity,
            sample_interval_s=segment.sample_interval_s,
            duration_s=float(times_s[-1] - times_s[0]),
        )
        total += np.interp(times_s, times, rate, left=0.0, right=0.0)
    return total, cumulative_moment(times_s, total)


class CumulativeSlip:
    """How much each subfault has slipped by a given time, in metres.

    Each subfault's pulse integrated up to `t`, placed at its own onset. This is what
    makes the animation a rupture rather than a slide show: at any moment, the part of
    the fault that has moved is exactly the part the front has reached.

    **One frame at a time, by design.** The whole animation as a `(frames, cells)`
    array is the obvious shape and it does not fit: at the quarter-second default the
    shipped twenty-fault scenario is 1,229 frames over 2 million subfaults, which is
    20 GB of float64 for a picture that only ever shows one row of it. Here the state
    is the rupture rather than the animation, so the cost stops depending on how finely
    the timeline is stepped.

    **Constructing this consumes `segment.pulse_samples`.** The rates are overwritten
    in place with their own running sum, because a separate integral would be a second
    copy of the largest array in the file. Build it after anything that reads the
    rates -- `moment_release` is the only such caller, and it runs first.
    """

    def __init__(self, segment: Segment) -> None:
        """Integrate the segment's pulses in place, into cumulative slip."""
        offsets = np.asarray(segment.pulse_offsets, dtype=np.int64)
        self.starts = offsets[:-1]
        self.lengths = np.diff(offsets)
        self.occupied = self.lengths > 0
        self.interval_s = segment.sample_interval_s
        self.onset_s = segment.onset_s.ravel()

        # One running sum over every pulse laid end to end, rather than one per row:
        # a row's own running total is the difference from the value just before it
        # starts. In float64 the shared accumulator is good to about one part in 10^9
        # of a pulse here, which is nine digits more than a colour ramp can show.
        self.integral = segment.pulse_samples
        np.cumsum(self.integral, out=self.integral)
        # Where a row starts partway through, its own zero is the value just before it.
        # Clipped because a row that starts at zero has nothing before it, and a segment
        # on which nothing slips has no integral to read at all.
        last = max(self.integral.size - 1, 0)
        self.before = np.where(
            self.starts > 0,
            self.integral[np.clip(self.starts - 1, 0, last)]
            if self.integral.size
            else 0.0,
            0.0,
        )

    def _sampled(self, index: np.ndarray) -> np.ndarray:
        """Each row's running total at its own `index`, zero where it has no pulse."""
        if not self.integral.size:
            return np.zeros(self.lengths.size, dtype=np.float64)
        # A row with no pulse is read at position zero and masked out afterwards. It
        # cannot be read at its own start: a subfault that does not slip has none, and
        # if it is the last one that start is one sample past the end of the integral.
        at = np.where(self.occupied, self.starts + index, 0)
        return np.where(
            self.occupied, (self.integral[at] - self.before) * self.interval_s, 0.0
        )

    def at(self, time_s: float) -> np.ndarray:
        """Slip so far at `time_s`, one value per subfault, flat."""
        into = np.floor((time_s - self.onset_s) / self.interval_s)
        index = np.clip(into, 0, np.maximum(self.lengths - 1, 0)).astype(np.int64)
        return np.where(into < 0, 0.0, self._sampled(index))

    def total(self) -> np.ndarray:
        """Each subfault's slip once its pulse has finished."""
        return self._sampled(np.maximum(self.lengths - 1, 0))


# ============================================================================
# Drawing
# ============================================================================


def stride_for(segments: list[Segment], budget: int) -> int:
    """How many cells to skip so the drawn mesh fits the budget.

    A stride rather than a resampling: every drawn cell is a real subfault with its
    real value, and the ones between are simply not drawn. Averaging into
    super-cells would paint values no subfault has.

    Only the lattice segments are counted, because they are the only ones a stride can
    thin -- a triangulation arrives from :func:`decimate` already inside the budget,
    and counting it would stride the lattices down on its behalf.
    """
    total = sum(
        segment.slip_m.size for segment in segments if segment.cells is not None
    )
    if total <= budget:
        return 1
    return math.ceil(math.sqrt(total / budget))


def drawn(segment: Segment, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Which cells to colour, and the corners to draw them on.

    The one place the two tracks differ in how they are thinned, and they differ
    because a stride means something on a lattice and nothing on a triangulation --
    :func:`decimate` says why.

    Returns
    -------
    tuple
        Flat indices into the segment's fields, and the matching
        ``(drawn, corners, 3)`` positions.
    """
    if segment.cells is None:
        return np.arange(len(segment.corners_m)), segment.corners_m
    return strided(segment, stride), strided_corners(segment, stride)


def _lattice(segment: Segment) -> tuple[int, int]:
    """The segment's lattice shape, or a refusal naming what to use instead.

    Raises
    ------
    ValueError
        For a triangulation, which has no lattice to skip through: a stride would leave
        holes rather than a coarser surface -- see :func:`decimate`.
    """
    if segment.cells is None:
        raise ValueError(
            f"{segment.name!r} is a triangulation, so it has no lattice to stride "
            "through. Draw it with `drawn`, which decimates it instead"
        )
    return segment.cells


def strided(segment: Segment, stride: int) -> np.ndarray:
    """The flat indices of the cells to draw."""
    cells_i, cells_j = _lattice(segment)
    rows = np.arange(0, cells_i, stride)
    columns = np.arange(0, cells_j, stride)
    return (rows[:, None] * cells_j + columns[None, :]).ravel()


def strided_corners(segment: Segment, stride: int) -> np.ndarray:
    """The drawn quads: one per `stride` x `stride` block, spanning the whole block.

    **The colour is one subfault's; the extent is the block's.** Drawing the sampled
    subfault at its own size instead leaves the other ``stride**2 - 1`` as holes, and
    on a 100 m mesh strided by seven that is a 100 m quad every 700 m -- which reads
    as a point cloud rather than as a fault. Neighbouring blocks take their shared
    edge from the same cell corners, so the drawn surface closes.

    This still paints no value that no subfault has, which is what `stride_for`
    refuses to give up: `strided` picks the one real subfault each block is coloured
    by, and nothing here averages.
    """
    cells_i, cells_j = _lattice(segment)
    rows = np.arange(0, cells_i, stride)
    columns = np.arange(0, cells_j, stride)
    # The last cell each block reaches; the final block is short where the stride does
    # not divide the grid.
    row_ends = np.minimum(rows + stride - 1, cells_i - 1)
    column_ends = np.minimum(columns + stride - 1, cells_j - 1)

    def flat(down: np.ndarray, along: np.ndarray) -> np.ndarray:
        return (down[:, None] * cells_j + along[None, :]).ravel()

    corners = segment.corners_m
    # Corner k of the block is corner k of the cell that sits at that corner of it --
    # the ordering `load` builds, anticlockwise from the shallow near corner.
    return np.stack(
        [
            corners[flat(rows, columns), 0],
            corners[flat(rows, column_ends), 1],
            corners[flat(row_ends, column_ends), 2],
            corners[flat(row_ends, columns), 3],
        ],
        axis=1,
    )


def slip_direction(segment: Segment, keep: np.ndarray) -> np.ndarray:
    """Unit vectors along which each subfault slipped, in east-north-up.

    Rake is measured within the fault plane from the strike direction towards up-dip,
    so the slip direction is ``cos(rake)`` along strike plus ``sin(rake)`` up dip.
    Drawing it needs the plane's own basis, which strike and dip give.
    """
    strike = np.radians(segment.strike_deg.ravel()[keep])
    dip = np.radians(segment.dip_deg.ravel()[keep])
    rake = np.radians(segment.rake_deg.ravel()[keep])

    along = np.stack([np.sin(strike), np.cos(strike), np.zeros_like(strike)], axis=-1)
    up_dip = np.stack(
        [
            -np.sin(strike + np.pi / 2) * np.cos(dip),
            -np.cos(strike + np.pi / 2) * np.cos(dip),
            np.sin(dip),
        ],
        axis=-1,
    )
    return along * np.cos(rake)[:, None] + up_dip * np.sin(rake)[:, None]


def rose(values_deg: np.ndarray, bins: int = 36) -> list[np.ndarray]:
    """A rake distribution as wedges on a circle.

    A rake is an *angle*, so a bar chart of it is misleading twice over: it puts -179
    and 179 degrees at opposite ends of the axis when they are a degree apart, and it
    invites reading the horizontal axis as a magnitude. A rose puts the wrap where it
    belongs, which is nowhere.

    Returns
    -------
    list of np.ndarray
        One closed polygon per bin, in a two-dimensional view's own coordinates.
    """
    counts, edges = np.histogram(
        np.mod(values_deg + 180.0, 360.0) - 180.0, bins=bins, range=(-180.0, 180.0)
    )
    longest = counts.max() or 1
    wedges = []
    for count, low, high in zip(counts, edges[:-1], edges[1:], strict=True):
        if count == 0:
            continue
        radius = count / longest
        angles = np.radians(np.linspace(low, high, 6))
        # Screen coordinates with y downwards, so a rake of zero points right and the
        # circle runs the way a compass does.
        arc = np.stack([radius * np.cos(angles), -radius * np.sin(angles)], axis=-1)
        wedges.append(np.vstack([[0.0, 0.0], arc, [0.0, 0.0]]))
    return wedges


def rose_axis(bins: int = 36) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """The reference circle a rake rose is read against, marked in degrees.

    :func:`rose` draws wedges whose angle is the rake and whose radius is a count
    normalised to the largest bin, and a bare set of wedges is a plot with no axis at
    all -- there is nothing on screen saying which way is zero, which way is positive,
    or how far round a wedge sits. That is the angular form of the same complaint as
    a bar chart indexed by bin number.

    So: the unit circle the longest wedge touches, a half-way circle, spokes on the
    eights, and a degree label against each. The wrap stays where :func:`rose` put it,
    which is nowhere.

    Returns
    -------
    tuple
        The guide polylines, the label positions, and the labels.
    """
    ticks = np.arange(-180.0, 180.0, 45.0)
    guides = []
    for radius in (0.5, 1.0):
        angles = np.radians(np.linspace(-180.0, 180.0, 4 * bins + 1))
        guides.append(
            np.stack([radius * np.cos(angles), -radius * np.sin(angles)], axis=-1)
        )
    for degrees in ticks:
        angle = math.radians(degrees)
        guides.append(
            np.array([[0.0, 0.0], [math.cos(angle), -math.sin(angle)]], dtype=float)
        )

    radians = np.radians(ticks)
    positions = np.stack([1.18 * np.cos(radians), -1.18 * np.sin(radians)], axis=-1)
    return guides, positions, [f"{int(degrees)}\N{DEGREE SIGN}" for degrees in ticks]


def require_rerun():
    """Import Rerun, or say how to get it.

    Guarded rather than imported at the top because `rerun-sdk` pulls pyarrow and
    pillow, and neither `mesh` nor `generate` needs a display. A cluster running a
    thousand realisations should not be installing a GUI stack.
    """
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:  # pragma: no cover - exercised by not having it
        message = (
            "rerun-sdk is required for `view`. Install it with:\n"
            "    pip install 'rupture-generator[vis]'"
        )
        raise ImportError(message) from error
    return rr, rrb


def layout(blueprint):  # noqa: ANN001 - Rerun's types
    """The panels: the fault, and three ways of reading the same numbers.

    Each selector is a tab strip rather than a command-line flag, because the whole
    point of a viewer is to look at one thing and then another without re-running it.
    """
    return blueprint.Blueprint(
        blueprint.Horizontal(
            blueprint.Vertical(
                blueprint.TextDocumentView(origin="/statistics", name="statistics"),
                blueprint.Tabs(
                    # Spatial rather than bar-chart views: a bar chart takes one
                    # colour for the whole chart, and these are coloured per bin to
                    # match the fault. See `_histogram`.
                    blueprint.Spatial2DView(origin="/histogram/slip", name="slip"),
                    blueprint.Spatial2DView(
                        origin="/histogram/rise_time", name="rise time"
                    ),
                    blueprint.Spatial2DView(origin="/histogram/rake", name="rake"),
                    name="distribution",
                ),
                blueprint.Tabs(
                    blueprint.TimeSeriesView(origin="/moment/rate", name="moment rate"),
                    blueprint.TimeSeriesView(
                        origin="/moment/cumulative", name="cumulative moment"
                    ),
                    name="moment release",
                ),
                row_shares=[2, 3, 3],
            ),
            blueprint.Tabs(
                blueprint.Spatial3DView(
                    origin="/fault",
                    name="slip",
                    contents=["/fault/slip/**", "/fault/hypocentre/**"],
                ),
                blueprint.Spatial3DView(
                    origin="/fault",
                    name="rise time",
                    contents=["/fault/rise_time/**", "/fault/hypocentre/**"],
                ),
                blueprint.Spatial3DView(
                    origin="/fault",
                    name="rake",
                    contents=["/fault/rake/**", "/fault/hypocentre/**"],
                ),
            ),
            column_shares=[1, 3],
        ),
        collapse_panels=True,
    )


def view(
    rupture: Annotated[
        Path,
        typer.Argument(
            help="Rupture file: the native format, or a text SRF.",
            exists=True,
            readable=True,
        ),
    ],
    time_step: Annotated[
        float, typer.Option(help="Seconds between animation frames.")
    ] = 0.25,
    bins: Annotated[int, typer.Option(help="Histogram bins.")] = 40,
    max_cells: Annotated[
        int,
        typer.Option(
            help="Cells to draw. A lattice is strided down to this; a triangulation is "
            "redrawn on a coarser surface of twice as many triangles."
        ),
    ] = DEFAULT_CELL_BUDGET,
    save: Annotated[
        Path | None,
        typer.Option(help="Write a .rrd recording instead of opening a window."),
    ] = None,
) -> None:
    """Show a rupture propagating, with its moment release and field distributions."""
    try:
        rerun, blueprint = require_rerun()
    except ImportError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    try:
        segments, provenance = load(rupture, max_cells)
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    rerun.init("rupture-generator", spawn=save is None)
    if save is not None:
        rerun.save(save)

    rerun.send_blueprint(layout(blueprint))
    log_rupture(rerun, segments, provenance, time_step, bins, max_cells)

    if save is not None:
        console.print(f"[green]wrote[/green] {save}")


def log_rupture(
    rerun,  # noqa: ANN001 - Rerun's module
    segments: list[Segment],
    provenance: str,
    time_step: float,
    bins: int,
    max_cells: int,
) -> None:
    """Log every panel, over the rupture's own timeline."""
    stride = stride_for(segments, max_cells)
    kept, quads = {}, {}
    for segment in segments:
        kept[segment.name], quads[segment.name] = drawn(segment, stride)

    # The provenance leads the panel because it is the one sentence that says whether
    # what is on screen is the file or a redrawing of it.
    summary = f"*{provenance}*\n\n{statistics(segments)}"
    rerun.log(
        "/statistics",
        rerun.TextDocument(summary, media_type="text/markdown"),
        static=True,
    )

    # The whole rupture on one clock: the earliest onset to the last pulse's end, from
    # the file's own subfaults rather than the drawn cells, so a decimated rupture's
    # timeline still reaches the last subfault to stop moving.
    counted = [segment.population for segment in segments]
    starts = [float(segment.onset_s.min()) for segment in counted]
    ends = [float((segment.onset_s + segment.rise_time_s).max()) for segment in counted]
    times_s = np.arange(min(starts), max(ends) + time_step, time_step)

    _log_static_fields(rerun, segments, kept, quads, bins)
    _log_hypocentre(rerun, segments)

    # `moment_release` reads the slip *rates*; building the clocks overwrites them with
    # their running sum. That order is a requirement, not a preference.
    rate, cumulative = moment_release(segments, times_s)
    rate_scale = _engineering_scale(rate)
    cumulative_scale = _engineering_scale(cumulative)
    _label_moment_axes(rerun, rate_scale, cumulative_scale)
    slipped = {segment.name: CumulativeSlip(segment) for segment in segments}
    peak = (
        max([float(clock.total().max()) for clock in slipped.values()] + [0.0]) or 1.0
    )

    for step, moment_s in enumerate(times_s):
        rerun.set_time("rupture", duration=float(moment_s))
        rerun.log("/moment/rate", rerun.Scalars(float(rate[step]) / rate_scale))
        rerun.log(
            "/moment/cumulative",
            rerun.Scalars(float(cumulative[step]) / cumulative_scale),
        )

        frame = []
        for segment in segments:
            indices = kept[segment.name]
            current = slipped[segment.name].at(float(moment_s))
            rerun.log(
                f"/fault/slip/{segment.name}",
                _mesh(rerun, quads[segment.name], hot(current[indices], 0.0, peak)),
            )
            frame.append(current)
        # The same map and the same limits the slip mesh is drawn with, a few lines up.
        # This one is over the *drawn* cells rather than the file's subfaults, because
        # slip-so-far is a function of time and only the drawn cells carry a pulse each
        # -- the static distributions above are the file's own.
        _histogram(
            rerun,
            "/histogram/slip",
            np.concatenate(frame),
            (0.0, peak),
            bins,
            lambda values: hot(values, 0.0, peak),
            "slip (m)",
        )


def _log_static_fields(
    rerun,  # noqa: ANN001
    segments: list[Segment],
    kept: dict[str, np.ndarray],
    quads: dict[str, np.ndarray],
    bins: int,
) -> None:
    """Rise time and rake, which are properties of the finished rupture.

    The two distributions are over the file's own subfaults -- neither field is a
    function of time, so nothing forces them down to the drawn cells -- while the
    meshes and arrows carry the drawn cells' values on the distributions' own limits.
    """
    counted = [segment.population for segment in segments]
    rise = np.concatenate([segment.rise_time_s.ravel() for segment in counted])
    rake = np.concatenate([segment.rake_deg.ravel() for segment in counted])
    low, high = float(rise.min()), float(rise.max())

    for segment in segments:
        indices = kept[segment.name]
        rerun.log(
            f"/fault/rise_time/{segment.name}",
            _mesh(
                rerun,
                quads[segment.name],
                viridis(segment.rise_time_s.ravel()[indices], low, high),
            ),
            static=True,
        )

    # Rake, as arrows along the direction each subfault slipped. Thinned further than
    # the mesh: arrows overlap into a solid mass long before cells do.
    for segment in segments:
        indices = kept[segment.name]
        budget = max(1, MAX_ARROWS // len(segments))
        if len(indices) > budget:
            indices = indices[:: math.ceil(len(indices) / budget)]
        slip = segment.slip_m.ravel()[indices]
        scale = float(np.linalg.norm(segment.corners_m[0, 1] - segment.corners_m[0, 0]))
        lengths = scale * 6.0 * slip / (slip.max() or 1.0)
        rerun.log(
            f"/fault/rake/{segment.name}",
            rerun.Arrows3D(
                origins=segment.centres_m[indices],
                vectors=slip_direction(segment, indices) * lengths[:, None],
                colors=hot(slip, 0.0, float(slip.max()) or 1.0),
            ),
            static=True,
        )

    _histogram(
        rerun,
        "/histogram/rise_time",
        rise,
        (low, high),
        bins,
        lambda values: viridis(values, low, high),
        "rise time (s)",
        static=True,
    )
    # The axis first, so the wedges draw over it rather than under it.
    guides, label_positions, labels = rose_axis()
    rerun.log(
        "/histogram/rake/axis",
        rerun.LineStrips2D(guides, colors=[(90, 96, 104)], radii=[0.004]),
        static=True,
    )
    rerun.log(
        "/histogram/rake/axis/labels",
        rerun.Points2D(
            label_positions,
            colors=[AXIS_TEXT],
            labels=labels,
            show_labels=True,
            radii=[0.004],
        ),
        static=True,
    )
    rerun.log(
        "/histogram/rake",
        rerun.LineStrips2D(rose(rake), colors=[(216, 222, 26)]),
        static=True,
    )


def _log_hypocentre(rerun, segments: list[Segment]) -> None:  # noqa: ANN001
    """Where the rupture nucleated, on whichever segment holds it."""
    for segment in segments:
        if segment.hypocentre_m is None:
            continue
        rerun.log(
            "/fault/hypocentre",
            rerun.Points3D(
                positions=[segment.hypocentre_m],
                radii=[max(400.0, float(np.abs(segment.corners_m).max()) / 120.0)],
                colors=[(255, 64, 64)],
                labels=[f"hypocentre ({segment.name})"],
            ),
            static=True,
        )


def _engineering_scale(values: np.ndarray) -> float:
    """The power of a thousand these numbers are most readable in.

    A moment is around ``1e19`` newton-metres and a moment rate around ``1e18`` per
    second, and Rerun 0.35 has no hook for formatting a tick label -- `ScalarAxis`
    carries a range and a zoom lock and nothing else. So the axis reads
    ``6.000000e18`` where it wants to read ``6.0``, and the only lever left is the
    number that goes in. Divide by the enclosing power of a thousand and say which one
    in the series name: the same information, in the two places a reader looks.

    A thousand rather than ten so the exponent is one an SI reader already has a word
    for, and so it stays put while the curve grows through an order of magnitude.
    """
    peak = float(np.max(np.abs(values))) if np.size(values) else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return 1.0
    return float(10.0 ** (3 * math.floor(math.log10(peak) / 3)))


def _label_moment_axes(rerun, rate_scale: float, cumulative_scale: float) -> None:  # noqa: ANN001
    """Name each moment series for the units it is actually plotted in.

    The scaling in :func:`_engineering_scale` is only honest if the exponent it
    removed is visible, and the series name is where a time-series view shows it.
    """

    def units(scale: float, unit: str) -> str:
        return unit if scale == 1.0 else f"1e{round(math.log10(scale))} {unit}"

    rerun.log(
        "/moment/rate",
        rerun.SeriesLines(names=[f"moment rate ({units(rate_scale, 'N m/s')})"]),
        static=True,
    )
    rerun.log(
        "/moment/cumulative",
        rerun.SeriesLines(
            names=[f"cumulative moment ({units(cumulative_scale, 'N m')})"]
        ),
        static=True,
    )


def _ticks(low: float, high: float, count: int = 5) -> np.ndarray:
    """Round positions to label an axis at, spanning ``low`` to ``high``."""
    if not np.isfinite([low, high]).all() or high <= low:
        return np.array([low])
    step = (high - low) / count
    magnitude = 10.0 ** math.floor(math.log10(step))
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        if step <= multiple * magnitude:
            step = multiple * magnitude
            break
    first = math.ceil(low / step) * step
    return np.arange(first, high + step / 2.0, step)


def _histogram(
    rerun,  # noqa: ANN001
    path: str,
    values: np.ndarray,
    limits: tuple[float, float],
    bins: int,
    colours: Colouring,
    unit: str,
    *,
    static: bool = False,
) -> None:
    """A histogram drawn as boxes, on the quantity's axis, in the fault's own colours.

    **Why this is drawn rather than logged as a `BarChart`.** Rerun's bar chart takes
    one colour for the whole chart -- its own documentation says "the color of the bar
    chart" -- and the component batch accepts an array of them without complaint while
    the visualiser draws only the first. So a `BarChart` coloured per bin comes out
    uniformly `hot(0)`, which is black, or uniformly `viridis(low)`, which is purple:
    the bars vanish rather than saying they could not be coloured. `Boxes2D` colours
    per box, so the histogram is assembled here and the panel is a 2-D view.

    That costs the axis a bar chart draws for itself, so one is drawn too -- and it is
    the axis that was the point. Counts against *bin number* read as a distribution
    over the quantity while meaning something that changes when the bin count does; a
    slip peak "at 7" means seven metres here.

    Each bar takes the colour its own bin centre has on the fault, from the same map
    at the same limits, which is what makes the two panels one instrument: a band of
    colour on the fault is findable in the distribution by its colour rather than by
    reading a number off one panel and hunting for it in the other.

    Drawn in a unit box with the real values on the labels, so the two axes stay
    legible against each other however far apart their magnitudes are -- slip runs to
    single-figure metres against counts in the tens of thousands.

    **Subfaults at rest are counted, not binned.** A cell the front has not reached
    has no slip and no rise time, and there are tens of thousands of them -- so they
    pile into the first bin and make a spike several times the height of the
    distribution, which then flattens everything that was worth looking at. They are
    not part of the distribution of slip on a slipping fault; they are the fault that
    has not slipped. Dropping them silently would be a lie about the sample size, so
    the count goes on the panel.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    moving = values > 0.0
    resting = int(values.size - np.count_nonzero(moving))
    counts, edges = np.histogram(values[moving], bins=bins, range=limits)

    centres = 0.5 * (edges[:-1] + edges[1:])
    low, high = float(edges[0]), float(edges[-1])
    tallest = float(counts.max()) or 1.0
    span = (high - low) or 1.0

    # Counts on a log height. The tail of a slip distribution is a handful of cells
    # against tens of thousands in the mode, and on a linear axis every bin outside the
    # mode is a line one pixel high -- present, unreadable, and easy to mistake for
    # empty. `log10(count + 1)` rather than `log10(count)` so that a bin holding a
    # single cell still stands above the baseline instead of vanishing into it.
    def height_of(count: np.ndarray | float) -> np.ndarray | float:
        return np.log10(np.asarray(count, dtype=np.float64) + 1.0) / math.log10(
            tallest + 1.0
        )

    across = (centres - low) / span
    heights = height_of(counts)
    width = float(edges[1] - edges[0]) / span

    # Screen coordinates run y downwards, so a bar of height h spans 0 to -h.
    rerun.log(
        path,
        rerun.Boxes2D(
            centers=np.stack([across, -heights / 2.0], axis=-1),
            half_sizes=np.stack(
                [np.full(len(counts), width / 2.0), heights / 2.0], axis=-1
            ),
            colors=colours(centres),
        ),
        static=static,
    )

    guides = [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, -1.0]]),
    ]
    positions, labels = [], []
    for value in _ticks(low, high):
        x = (value - low) / span
        guides.append(np.array([[x, 0.0], [x, 0.02]]))
        positions.append([x, 0.07])
        labels.append(f"{value:.3g}")
    decades = [0.0] + [
        10.0**power for power in range(math.ceil(math.log10(tallest)) + 1)
    ]
    for value in decades:
        if value > tallest:
            break
        y = -float(height_of(value))
        guides.append(np.array([[0.0, y], [-0.015, y]]))
        positions.append([-0.075, y])
        labels.append(f"{value:,.0f}")
    positions.append([0.5, 0.15])
    labels.append(unit)
    positions.append([-0.075, -1.04])
    labels.append("count (log)")
    if resting:
        positions.append([0.5, -1.09])
        labels.append(f"{resting:,} subfaults at rest, not shown")

    rerun.log(
        f"{path}/axis",
        rerun.LineStrips2D(guides, colors=[AXIS_LINE], radii=[0.0015]),
        static=static,
    )
    rerun.log(
        f"{path}/axis/labels",
        rerun.Points2D(
            np.array(positions),
            colors=[AXIS_TEXT],
            labels=labels,
            show_labels=True,
            radii=[0.002],
        ),
        static=static,
    )


def _mesh(rerun, corners: np.ndarray, colours: np.ndarray):  # noqa: ANN001
    """A `Mesh3D` of flat-coloured cells, fanned into triangles.

    The arity comes from ``corners.shape`` rather than being assumed: a quad lattice
    gives four corners and a triangulation three, and a fan from corner zero reduces to
    exactly the two-triangle split for a quad, so neither track pays for the other.
    """
    cells, arity = corners.shape[:2]
    vertices = corners.reshape(-1, 3).astype(np.float32)
    base = np.arange(cells) * arity
    triangles = np.concatenate(
        [
            np.stack([base, base + corner + 1, base + corner + 2], axis=-1)
            for corner in range(arity - 2)
        ]
    ).astype(np.uint32)
    return rerun.Mesh3D(
        vertex_positions=vertices,
        triangle_indices=triangles,
        vertex_colors=np.repeat(colours, arity, axis=0),
    )


__all__ = [
    "FIELDS",
    "CumulativeSlip",
    "Segment",
    "Subfaults",
    "decimate",
    "display_spacing_km",
    "drawn",
    "hot",
    "load",
    "moment_release",
    "rose",
    "statistics",
    "stride_for",
    "strided_corners",
    "view",
    "viridis",
]
