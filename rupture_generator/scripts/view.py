"""``rupture-generator view``: watch a rupture happen."""

from __future__ import annotations

import dataclasses
import itertools
import math
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer
import xarray as xr

from rupture_generator.errors import FormatError
from rupture_generator.formats import Format, from_path, read_rupture, segments_in
from rupture_generator.moment import cumulative_moment, moment_rate, rigidity_pa
from rupture_generator.scripts.render import console

if TYPE_CHECKING:
    from collections.abc import Callable

    # A colour map already closed over the same `low` and `high` the fault is drawn at.
    Colouring = Callable[[np.ndarray], np.ndarray]

FIELDS = {
    "slip": ("slip_m", "metres", "hot"),
    "onset": ("onset_s", "seconds", "viridis"),
    "rise-time": ("rise_time_s", "seconds", "viridis"),
    "rake": ("rake_deg", "degrees", "quiver"),
}
"""Which variable each field view shows, its unit, and how it is drawn."""

ANIMATED = "slip"
"""The only field that is a function of time."""

DEFAULT_CELL_BUDGET = 50_000
"""How many cells to draw before striding the display down.

At 50,000 cells a 1,600-pixel window has about six pixels per subfault, so striding
further discards detail the screen could have resolved.
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
class Segment:
    """One fault segment, in the local metric frame the viewer draws in.

    Every input format reduces to this. Positions are metres east, north and **up**
    from the rupture's own centroid, up rather than down because a viewer's vertical
    axis points up and every file format measures depth downwards.

    Attributes
    ----------
    cells : tuple of int
        The lattice shape the file was stored on, which is what :func:`strided` skips
        through.
    corners_m : np.ndarray
        ``(cells, corners, 3)``, anticlockwise from the shallow near corner. Nothing
        below reads the corner count, it fans whatever it is given. Unshared between
        cells.
    rigidity_pa : np.ndarray or None
        Present when the file carries the material properties the generator sampled.
        Otherwise the moment release is drawn at a nominal rigidity, with the panel
        saying so.
    plane : np.ndarray or None
        Which of the surface's planes each along-strike column came from, ``(cells_j,)``.
        A native file records it; an SRF has no equivalent, because there a plane is a
        segment and the question does not arise.
    occupied : np.ndarray or None
        Which cells are really fault, shaped like the lattice. None when every cell is,
        which is every fault built from a config and every SRF. A chart resampled from a
        modeller's outline is a rectangle and the fault inside it is not: the CFM
        Hikurangi interface fills about two thirds of the rectangle its outline spans.
    """

    name: str
    cells: tuple[int, int]
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
    plane: np.ndarray | None = None
    occupied: np.ndarray | None = None

    def values(self, variable: str) -> np.ndarray:
        """One field, flattened over subfaults."""
        return getattr(self, variable).ravel()

    @property
    def on_fault(self) -> np.ndarray:
        """Which subfaults are really fault, flat.

        The cells outside the outline are not a smaller kind of fault, they are not
        fault: they carry positions because the grid needs corners, and the generator
        walls the wavefront off from them, does not count them in the moment and does
        not write them to an SRF. A panel that reads them reads a rectangle the
        modeller never drew -- their onsets run to half an hour, because a walled-off
        cell is one the front crawled into rather than one it reached.
        """
        if self.occupied is None:
            # Sized from the lattice rather than from any one field, so that it indexes
            # every field alike and a triangulation, which has no lattice, is sized
            # from its cells.
            count = (
                math.prod(self.cells) if self.cells is not None else len(self.corners_m)
            )
            return np.ones(count, dtype=bool)
        return np.asarray(self.occupied, dtype=bool).ravel()

    def spanned(self) -> tuple[float, float]:
        """When the first subfault starts and the last one stops, in seconds.

        Over the fault rather than over the rectangle it was cut from.
        """
        mask = self.on_fault
        onset = self.values("onset_s")[mask]
        return (
            float(onset.min()),
            float((onset + self.values("rise_time_s")[mask]).max()),
        )


# Colour


def viridis(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Perceptually even, and the right choice for a field with no meaningful zero."""
    return _ramp(_VIRIDIS_16, values, low, high)


def hot(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Black through red and yellow to white.

    Matplotlib's ``hot`` segment boundaries: red saturates over the first three
    eighths, green over the next three, blue over the last quarter.
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


# Matplotlib's viridis at 16 evenly spaced anchors, rounded to 8-bit channels.
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


# Loading -- both formats reduce to Segment


def load(path: Path) -> tuple[list[Segment], str]:
    """Read a rupture, whatever it is written as.

    The display budget is not applied here: a structured rupture is strided at draw
    time, by :func:`stride_for`.

    Parameters
    ----------
    path : Path
        A native rupture file, or a text SRF.

    Returns
    -------
    tuple
        The segments, and a sentence saying what was read and how faithfully.

    Raises
    ------
    FormatError
        For a format this cannot read, or a native file that holds no rupture.
    """
    chosen = from_path(path)
    if chosen in (Format.NETCDF, Format.ZARR):
        with read_rupture(path) as tree:
            return _from_rupture_file(tree), "native rupture file"
    if chosen is Format.SRF:
        return _from_srf(path), "SRF, mesh reconstructed from subfault centres"
    raise FormatError(
        f"a rupture cannot be read from {chosen.value}: this reads the native format "
        "and text SRF"
    )


def _from_rupture_file(tree: xr.DataTree) -> list[Segment]:
    """The structured format, whose nodes are stored rather than reconstructed."""
    segments: list[Segment] = []
    found = segments_in(tree)
    if not found:
        raise FormatError("this file holds no rupture")

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
                plane=dataset["plane"].to_numpy() if "plane" in dataset else None,
                occupied=(
                    dataset["occupied"].to_numpy() if "occupied" in dataset else None
                ),
            )
        )
    return segments


def _hypocentre_position(
    dataset: xr.Dataset, nodes: np.ndarray, origin: np.ndarray
) -> np.ndarray:
    """Where the rupture nucleated, from the arc lengths the file records.

    The file stores the hypocentre as the config states it, in distances along strike
    and down dip, so this walks the same arc lengths back to a position.
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


def _from_srf(path: Path) -> list[Segment]:
    """An SRF, whose mesh is reconstructed from what the format does store.

    An SRF records subfault *centres*, not corners, so the quads here are built by
    stepping half a cell along strike and down dip from each centre, using the cell
    size the plane header implies. That is exact for the uniform planar grids an SRF
    describes and approximate for nothing else.

    A PLANE header that summarises its segment rather than describing a lattice cannot
    be read back this way; view such a rupture from its own native file.
    """
    from rupture_generator.srf import read_srf

    srf = read_srf(path)
    points = srf.points

    latitude = np.asarray(points.latitude_deg, dtype=np.float64)
    longitude = np.asarray(points.longitude_deg, dtype=np.float64)
    depth_m = np.asarray(points.depth_km, dtype=np.float64) * 1000.0

    # A local tangent frame about the rupture's own centroid, good to a few metres.
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


# What the numbers say


NOMINAL_RIGIDITY_PA = 3.0e10
"""30 GPa, crustal rock. Used only where a file does not carry the rigidity the
generator sampled, and the panel says when that is."""


def statistics(segments: list[Segment]) -> str:
    """The numbers that say what kind of earthquake this is, and the moment release.

    Read off the segments themselves, so every number is about the rupture the file
    holds rather than the coarser surface a triangulation is drawn on.

    Returns
    -------
    str
        A markdown summary.
    """
    counted = list(segments)
    exact = all(segment.rigidity_pa is not None for segment in counted)

    moment_nm = 0.0
    area_m2 = 0.0
    for segment in counted:
        rigidity = (
            segment.rigidity_pa
            if segment.rigidity_pa is not None
            else np.full(segment.slip_m.size, NOMINAL_RIGIDITY_PA)
        )
        # On the fault only. A resampled interface is stored on the rectangle its
        # outline spans, and counting that rectangle's area would report a fault
        # half again as large as the modeller drew.
        mask = segment.on_fault
        moment_nm += float(
            np.sum(
                rigidity.ravel()[mask]
                * segment.area_m2.ravel()[mask]
                * segment.slip_m.ravel()[mask]
            )
        )
        area_m2 += float(np.sum(segment.area_m2.ravel()[mask]))

    magnitude = (math.log10(moment_nm) - 9.0499505) / 1.5 if moment_nm > 0 else 0.0

    # When the last subfault *stops* slipping, not when the last one starts.
    spans = [segment.spanned() for segment in counted]
    duration_s = max(end for _, end in spans) - min(start for start, _ in spans)

    slip = np.concatenate(
        [segment.slip_m.ravel()[segment.on_fault] for segment in counted]
    )
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

    Both come from the same pulses, the second being the running integral of the first.
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

    Each subfault's pulse integrated up to `t`, placed at its own onset. One frame at a
    time: the whole animation as a ``(frames, cells)`` array is 20 GB of float64 for
    the shipped twenty-fault scenario at the quarter-second default.

    Constructing this **consumes** ``segment.pulse_samples``, overwriting the rates in
    place with their running sum, so build it after anything that reads them --
    `moment_release` is the only such caller, and it runs first.
    """

    def __init__(self, segment: Segment) -> None:
        """Integrate the segment's pulses in place, into cumulative slip."""
        offsets = np.asarray(segment.pulse_offsets, dtype=np.int64)
        self.starts = offsets[:-1]
        self.lengths = np.diff(offsets)
        self.occupied = self.lengths > 0
        self.interval_s = segment.sample_interval_s
        self.onset_s = segment.onset_s.ravel()

        # One running sum over every pulse laid end to end: a row's own running total
        # is the difference from the value just before it starts.
        self.integral = segment.pulse_samples
        np.cumsum(self.integral, out=self.integral)
        # Where a row starts partway through, its own zero is the value just before it.
        # Clipped because a row that starts at zero has nothing before it.
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
        # A row with no pulse is read at position zero and masked out afterwards: its
        # own start can be one sample past the end of the integral.
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


# Drawing


def stride_for(segments: list[Segment], budget: int) -> int:
    """How many cells to skip so the drawn mesh fits the budget.

    A stride rather than a resampling, so every drawn cell is a real subfault with its
    real value. Only the lattice segments are counted: a triangulation arrives from
    :func:`decimate` already inside the budget.
    """
    total = sum(
        segment.slip_m.size for segment in segments if segment.cells is not None
    )
    if total <= budget:
        return 1
    return math.ceil(math.sqrt(total / budget))


def drawn(segment: Segment, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Which cells to colour, and the corners to draw them on.

    A stride means something on a lattice and nothing on a triangulation --
    :func:`decimate` says why.

    Returns
    -------
    tuple
        Flat indices into the segment's fields, and the matching
        ``(drawn, corners, 3)`` positions.
    """
    if segment.cells is None:
        return np.arange(len(segment.corners_m)), segment.corners_m
    indices, corners = strided(segment, stride), strided_corners(segment, stride)
    # A block is drawn if the cell whose colour it carries is fault. At stride 1 that
    # is the outline exactly; striding coarsens it, the way striding coarsens
    # everything else here. Drawing the rest would put the rectangle on screen
    # instead of the fault.
    standing = segment.on_fault[indices]
    if standing.all():
        return indices, corners
    return indices[standing], corners[standing]


def strided(segment: Segment, stride: int) -> np.ndarray:
    """The flat indices of the cells to draw."""
    cells_i, cells_j = segment.cells
    rows = np.arange(0, cells_i, stride)
    columns = np.arange(0, cells_j, stride)
    return (rows[:, None] * cells_j + columns[None, :]).ravel()


def strided_corners(segment: Segment, stride: int) -> np.ndarray:
    """The drawn quads: one per `stride` x `stride` block, spanning the whole block.

    The colour is one subfault's and the extent is the block's, so the drawn surface
    closes rather than reading as a point cloud. Neighbouring blocks take their shared
    edge from the same cell corners, and nothing here averages.
    """
    cells_i, cells_j = segment.cells
    rows = np.arange(0, cells_i, stride)
    columns = np.arange(0, cells_j, stride)
    # The last cell each block reaches; the final block is short where the stride does
    # not divide the grid.
    row_ends = np.minimum(rows + stride - 1, cells_i - 1)
    column_ends = np.minimum(columns + stride - 1, cells_j - 1)

    def flat(down: np.ndarray, along: np.ndarray) -> np.ndarray:
        return (down[:, None] * cells_j + along[None, :]).ravel()

    corners = segment.corners_m
    # Corner k of the block is corner k of the cell at that corner of it: the ordering
    # `load` builds, anticlockwise from the shallow near corner.
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

    A rake is an angle, so a linear axis would put -179 and 179 degrees at opposite
    ends when they are a degree apart.

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
        # Screen coordinates with y downwards, so a rake of zero points right.
        arc = np.stack([radius * np.cos(angles), -radius * np.sin(angles)], axis=-1)
        wedges.append(np.vstack([[0.0, 0.0], arc, [0.0, 0.0]]))
    return wedges


def rose_axis(bins: int = 36) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """The reference circle a rake rose is read against, marked in degrees.

    The unit circle the longest wedge touches, a half-way circle, spokes on the
    eights, and a degree label against each.

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
    pillow, and neither `mesh` nor `generate` needs a display.
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
    """The panels: the fault, and four ways of reading the same numbers.

    Each 3D tab holds one field, its colourbar, and the hypocentre, and nothing else:
    the four colourbars stand in the same place in the scene, so a tab that let two of
    them through would draw one over the other.
    """
    return blueprint.Blueprint(
        blueprint.Horizontal(
            blueprint.Vertical(
                blueprint.TextDocumentView(origin="/statistics", name="statistics"),
                blueprint.Tabs(
                    # Spatial rather than bar-chart views: these are coloured per bin
                    # to match the fault. See `_histogram`.
                    blueprint.Spatial2DView(origin="/histogram/slip", name="slip"),
                    blueprint.Spatial2DView(
                        origin="/histogram/onset", name="onset time"
                    ),
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
                    # Where one plane of a fault meets the next: present, and off until
                    # a reader asks, which is what the blueprint panel is expanded for.
                    overrides={
                        "/fault/slip/joints": blueprint.EntityBehavior(visible=False)
                    },
                ),
                blueprint.Spatial3DView(
                    origin="/fault",
                    name="onset time",
                    contents=["/fault/onset/**", "/fault/hypocentre/**"],
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
        # The blueprint panel is the only way Rerun offers to switch an entity off, and
        # the things worth switching off are all here: the hypocentre's label and
        # leader, the isochrones, their labels, the colourbar. Collapsing it, which is
        # what `collapse_panels` does, hides the switches rather than the panels.
        blueprint.BlueprintPanel(state="expanded"),
        blueprint.SelectionPanel(state="collapsed"),
        blueprint.TimePanel(state="collapsed"),
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
        segments, provenance = load(rupture)
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


@dataclasses.dataclass(frozen=True)
class Drawing:
    """What every panel needs to know about how this rupture is being drawn.

    The cells that survived the display budget, the quads they are drawn on, the box
    all of them fit in, and how many bins the distributions get. Passed around as one
    thing because a panel needs most of it and none of it is a choice a panel makes.
    """

    kept: dict[str, np.ndarray]
    quads: dict[str, np.ndarray]
    bounds: tuple[np.ndarray, np.ndarray]
    bins: int

    @property
    def lift(self) -> float:
        """How far off the surface a label has to sit to stop z-fighting the mesh."""
        return 0.01 * _reach(self.bounds)


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
    drawing = Drawing(kept, quads, _scene_bounds(quads), bins)

    # The provenance leads the panel: it says whether what is on screen is the file.
    summary = f"*{provenance}*\n\n{statistics(segments)}"
    rerun.log(
        "/statistics",
        rerun.TextDocument(summary, media_type="text/markdown"),
        static=True,
    )

    # The whole rupture on one clock, from the file's own subfaults rather than the
    # drawn cells, so a decimated timeline still reaches the last subfault to stop.
    spans = [segment.spanned() for segment in segments]
    times_s = np.arange(
        min(start for start, _ in spans),
        max(end for _, end in spans) + time_step,
        time_step,
    )

    _log_static_fields(rerun, segments, drawing)
    _log_hypocentre(rerun, segments, drawing)

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
    _log_colourbar(
        rerun,
        "/fault/slip",
        (0.0, peak),
        lambda values: hot(values, 0.0, peak),
        "slip (m)",
        drawing.bounds,
    )
    _log_joints(rerun, segments, drawing)

    for step, moment_s in enumerate(times_s):
        rerun.set_time("rupture", duration=float(moment_s))
        rerun.log("/moment/rate", rerun.Scalars(float(rate[step]) / rate_scale))
        rerun.log(
            "/moment/cumulative",
            rerun.Scalars(float(cumulative[step]) / cumulative_scale),
        )

        frame = []
        for segment in segments:
            indices = drawing.kept[segment.name]
            current = slipped[segment.name].at(float(moment_s))
            rerun.log(
                f"/fault/slip/{segment.name}",
                _mesh(
                    rerun,
                    drawing.quads[segment.name],
                    hot(current[indices], 0.0, peak),
                ),
            )
            frame.append(current[segment.on_fault])
        # The same map and limits the slip mesh is drawn with, over the fault's own
        # subfaults: slip-so-far is a function of time and only they carry a pulse
        # each, and an off-fault cell reads as a subfault at rest that never was one.
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
    drawing: Drawing,
) -> None:
    """Onset time, rise time and rake, which are properties of the finished rupture.

    The distributions are over the file's own subfaults; the meshes and arrows carry
    the drawn cells' values on the distributions' own limits.
    """
    onset = _log_scalar_field(
        rerun,
        segments,
        drawing,
        "onset_s",
        "onset",
        "onset time (s)",
        zero_is_rest=False,
    )
    _log_isochrones(rerun, segments, drawing, contour_levels(*onset))
    _log_scalar_field(
        rerun, segments, drawing, "rise_time_s", "rise_time", "rise time (s)"
    )

    # Rake, as arrows along the direction each subfault slipped, thinned further than
    # the mesh: arrows overlap into a solid mass long before cells do. Length and
    # colour are normalised over every segment rather than each on its own peak, so
    # that two segments' arrows mean the same thing and one colourbar covers both.
    peak = max(float(segment.slip_m.max()) for segment in segments) or 1.0
    for segment in segments:
        indices = drawing.kept[segment.name]
        budget = max(1, MAX_ARROWS // len(segments))
        if len(indices) > budget:
            indices = indices[:: math.ceil(len(indices) / budget)]
        slip = segment.slip_m.ravel()[indices]
        scale = float(np.linalg.norm(segment.corners_m[0, 1] - segment.corners_m[0, 0]))
        rerun.log(
            f"/fault/rake/{segment.name}",
            rerun.Arrows3D(
                origins=segment.centres_m[indices],
                vectors=slip_direction(segment, indices)
                * (scale * 6.0 * slip / peak)[:, None],
                colors=hot(slip, 0.0, peak),
            ),
            static=True,
        )
    _log_colourbar(
        rerun,
        "/fault/rake",
        (0.0, peak),
        lambda values: hot(values, 0.0, peak),
        "slip (m), arrow colour",
        drawing.bounds,
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
    rake = np.concatenate(
        [segment.rake_deg.ravel()[segment.on_fault] for segment in segments]
    )
    rerun.log(
        "/histogram/rake",
        rerun.LineStrips2D(rose(rake), colors=[(216, 222, 26)]),
        static=True,
    )


def _log_scalar_field(
    rerun,  # noqa: ANN001
    segments: list[Segment],
    drawing: Drawing,
    variable: str,
    path: str,
    unit: str,
    *,
    zero_is_rest: bool = True,
) -> tuple[float, float]:
    """One scalar per subfault: the fault, its colourbar, and its distribution.

    Onset time and rise time are the same drawing: a static field on `viridis`, over
    limits taken from every subfault in the file rather than from the drawn ones, so
    striding the display cannot move the colour scale.

    Parameters
    ----------
    variable : str
        The `Segment` field to read.
    path : str
        The entity name, under ``/fault`` and ``/histogram``.
    zero_is_rest : bool
        Whether a zero means a subfault that never moved. True of rise time, and false
        of onset time, where zero is the arrival the rupture is timed from.

    Returns
    -------
    tuple
        The limits it was drawn on, for whatever else has to agree with them.
    """
    values = np.concatenate(
        [segment.values(variable)[segment.on_fault] for segment in segments]
    )
    low, high = float(values.min()), float(values.max())

    def colouring(binned: np.ndarray) -> np.ndarray:
        return viridis(binned, low, high)

    for segment in segments:
        rerun.log(
            f"/fault/{path}/{segment.name}",
            _mesh(
                rerun,
                drawing.quads[segment.name],
                viridis(
                    segment.values(variable)[drawing.kept[segment.name]], low, high
                ),
            ),
            static=True,
        )
    _log_colourbar(
        rerun, f"/fault/{path}", (low, high), colouring, unit, drawing.bounds
    )
    _histogram(
        rerun,
        f"/histogram/{path}",
        values,
        (low, high),
        drawing.bins,
        colouring,
        unit,
        static=True,
        zero_is_rest=zero_is_rest,
    )
    return low, high


def _log_isochrones(
    rerun,  # noqa: ANN001
    segments: list[Segment],
    drawing: Drawing,
    levels: np.ndarray,
) -> None:
    """Where the rupture front stood at each of a few round times.

    Traced on the file's own lattice rather than on the drawn one: the colour mesh is
    strided down to fit the display budget, and a front drawn on the strided lattice
    would be a staircase of the stride rather than the front the file records.

    Each line carries its own time, so nothing has to state the interval. The lines are
    drawn on both faces of the fault, a little clear of it: on the surface they z-fight
    the mesh, and on one face only they vanish behind it from half the viewpoints there
    are. Their labels are not, being text that draws over the fault either way.
    """
    lifted = {segment.name: _outward(segment) * drawing.lift for segment in segments}
    for segment in segments:
        onset = segment.onset_s.reshape(segment.cells).astype(np.float64)
        # `isochrones` takes a non-finite value as a front that never arrived, which is
        # what an off-fault cell is: the wavefront was walled off from it and crawled
        # in, so contouring its arrival would draw the wall rather than the front.
        onset = np.where(segment.on_fault.reshape(segment.cells), onset, np.nan)
        positions = segment.centres_m.reshape(*segment.cells, 3)

        drawn_levels, anchors, labels = [], [], []
        for level in levels:
            crossings = isochrones(onset, positions, float(level))
            if not len(crossings):
                continue
            drawn_levels.append(_both_sides(crossings, lifted[segment.name]))
            # One label a line, at the middle crossing it drew rather than at an end,
            # which puts it inside the fault instead of on its edge.
            anchors.append(crossings[len(crossings) // 2, 0] + lifted[segment.name])
            labels.append(f"{level:g} s")
        if not drawn_levels:
            continue

        rerun.log(
            f"/fault/onset/isochrones/{segment.name}",
            rerun.LineStrips3D(
                np.concatenate(drawn_levels), colors=[CONTOUR_LINE], radii=[UI_POINTS]
            ),
            static=True,
        )
        rerun.log(
            f"/fault/onset/isochrones/{segment.name}/labels",
            rerun.Points3D(
                np.array(anchors),
                colors=[CONTOUR_LINE],
                labels=labels,
                show_labels=True,
                radii=[UI_POINTS],
            ),
            static=True,
        )


def _log_joints(
    rerun,  # noqa: ANN001
    segments: list[Segment],
    drawing: Drawing,
) -> None:
    """Where one plane of a fault meets the next.

    A fault is stored as a single chart across all of its planes and the mesh is drawn
    continuously over them, so nothing on screen says where the trace was digitised.
    For a run of planes that happen to be collinear -- which is what a workflow makes
    when it cuts a straight trace into equal lengths -- that is indistinguishable from a
    fault that has only one, and the file is the only place the difference shows.

    Off by default, in the blueprint. This is a fact about how the input was written
    down rather than about the earthquake, and it is on the slip view because that is
    the one a reader looks at first.
    """
    for segment in segments:
        if segment.plane is None:
            continue
        boundaries = np.nonzero(np.diff(segment.plane))[0]
        if not boundaries.size:
            continue

        cells_i, cells_j = segment.cells
        corners = segment.corners_m.reshape(cells_i, cells_j, 4, 3)
        lift = _outward(segment) * drawing.lift
        # Corner 1 is the down-dip run of nodes on the far side of a column, and the
        # last row's corner 2 closes it at the bottom. Both faces, as for a contour.
        rerun.log(
            f"/fault/slip/joints/{segment.name}",
            rerun.LineStrips3D(
                [
                    side
                    for column in boundaries
                    for side in _both_sides(
                        np.concatenate(
                            [corners[:, column, 1], corners[-1:, column, 2]]
                        )[None],
                        lift,
                    )
                ],
                colors=[JOINT_LINE],
                radii=[UI_POINTS],
            ),
            static=True,
        )


def _log_hypocentre(
    rerun,  # noqa: ANN001
    segments: list[Segment],
    drawing: Drawing,
) -> None:
    """Where the rupture nucleated, on whichever segment holds it.

    The marker stays where the rupture actually started; the label stands off the
    surface on a leader line, because text logged at a position on the mesh z-fights
    it. The label is its own entity, and hiding it takes the leader with it.
    """
    for segment in segments:
        if segment.hypocentre_m is None:
            continue
        radius = max(400.0, float(np.abs(segment.corners_m).max()) / 120.0)
        anchor = segment.hypocentre_m + _outward(segment) * drawing.lift * 4.0
        rerun.log(
            f"/fault/hypocentre/{segment.name}",
            rerun.Points3D(
                positions=[segment.hypocentre_m], radii=[radius], colors=[(255, 64, 64)]
            ),
            static=True,
        )
        rerun.log(
            f"/fault/hypocentre/{segment.name}/label",
            rerun.Points3D(
                positions=[anchor],
                radii=[UI_POINTS],
                colors=[(255, 128, 128)],
                labels=[f"hypocentre ({segment.name})"],
                show_labels=True,
            ),
            static=True,
        )
        rerun.log(
            f"/fault/hypocentre/{segment.name}/label/leader",
            rerun.LineStrips3D(
                [np.stack([segment.hypocentre_m, anchor])],
                colors=[(255, 128, 128)],
                radii=[UI_POINTS],
            ),
            static=True,
        )


def _engineering_scale(values: np.ndarray) -> float:
    """The power of a thousand these numbers are most readable in.

    Rerun 0.35 has no hook for formatting a tick label, so the only lever is the number
    that goes in: divide by the enclosing power of a thousand and name it in the series
    label. A thousand rather than ten, so the exponent is one SI has a word for.
    """
    peak = float(np.max(np.abs(values))) if np.size(values) else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        return 1.0
    return float(10.0 ** (3 * math.floor(math.log10(peak) / 3)))


def _label_moment_axes(rerun, rate_scale: float, cumulative_scale: float) -> None:  # noqa: ANN001
    """Name each moment series for the units it is actually plotted in."""

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
    ticks = np.arange(first, high + step / 2.0, step)
    # The overshoot is there so a tick landing exactly on `high` survives the
    # comparison; it must not leave a label past the end of the axis.
    return ticks[ticks <= high + step * 1.0e-9]


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
    zero_is_rest: bool = True,
) -> None:
    """A histogram drawn as boxes, on the quantity's axis, in the fault's own colours.

    Rerun's `BarChart` takes one colour for the whole chart, so `Boxes2D` is used
    instead and the axis is drawn here. Subfaults at rest are counted, not binned --
    which is a statement about slip and rise time and not about every quantity, so
    `zero_is_rest` turns it off for one, like onset time, whose zero is a real value.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    moving = values > 0.0 if zero_is_rest else np.ones(values.shape, dtype=bool)
    resting = int(values.size - np.count_nonzero(moving))
    counts, edges = np.histogram(values[moving], bins=bins, range=limits)

    centres = 0.5 * (edges[:-1] + edges[1:])
    low, high = float(edges[0]), float(edges[-1])
    tallest = float(counts.max()) or 1.0
    span = (high - low) or 1.0

    # Counts on a log height, since the tail of a slip distribution is a handful of
    # cells against tens of thousands in the mode. `log10(count + 1)` rather than
    # `log10(count)` so a bin holding a single cell still stands above the baseline.
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


# Furniture: the scale a field is read against, and the times the front stood at


COLOURBAR_CELLS = 64
"""How many quads a colourbar is drawn as. Enough that the ramp reads as continuous."""

CONTOUR_LINE = (236, 240, 246)
"""Near-white, because an isochrone has to be followed across the whole of `viridis`."""

JOINT_LINE = (120, 200, 255)
"""Cool blue: `hot` runs black through red and yellow to white and has no blue in it,
so a joint drawn in this cannot be read as slip."""

CONTOUR_STEPS_S = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0, 120.0, 300.0)
"""The isochrone spacings worth offering, coarsest wins.

Round numbers a reader counts in: quarter-seconds for a single small fault, half a
minute for a subduction rupture. Anything not on this ladder -- 3.7 s, say -- makes the
reader do arithmetic to answer "how long between these two lines".
"""

TARGET_CONTOURS = 10
"""How many isochrones a fault can carry before they stop being separable."""

UI_POINTS = -1.0
"""Rerun reads a negative radius as a width in screen points rather than in metres.

Furniture wants that: a contour drawn 80 m wide is invisible on a scenario zoomed out
to a fault system, and a solid band on one zoomed in to a single subfault.
"""


def _scene_bounds(quads: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The box every drawn cell fits inside, as its low and high corners.

    Furniture is placed and sized from this rather than from any one segment, so that
    the colourbar of a five-fault scenario stands clear of all five.
    """
    corners = np.concatenate([quad.reshape(-1, 3) for quad in quads.values()])
    return corners.min(axis=0), corners.max(axis=0)


def _reach(bounds: tuple[np.ndarray, np.ndarray]) -> float:
    """How far the scene runs horizontally, which is what everything is scaled to."""
    low, high = bounds
    return float(max(high[0] - low[0], high[1] - low[1])) or 1.0


def _outward(segment: Segment) -> np.ndarray:
    """A unit vector off the fault surface, for lifting text clear of the mesh.

    A label logged at a position *on* the surface z-fights the mesh it is labelling.
    Up is not the fix -- a vertical fault contains it, which is most of New Zealand's
    crustal faults -- so this is the surface's own normal, taken from the corners of
    its first cell: along strike crossed with down dip.
    """
    cell = segment.corners_m[0]
    normal = np.cross(cell[1] - cell[0], cell[3] - cell[0])
    length = float(np.linalg.norm(normal))
    return normal / length if length else np.array([0.0, 0.0, 1.0])


def _both_sides(points: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """The same points lifted clear of a surface in both directions.

    One side is a coin flip. Every segment has its own normal, so lifting each by its
    own puts some of a fault system behind the surface from any viewpoint -- and no
    viewpoint fixes the rest, which is what makes it worse than a wrong guess: the
    reader turns the camera and a different subset of the contours disappears.

    Both copies are logged, and the mesh hides whichever one is behind it.

    For geometry only. Rerun draws a label over whatever is in front of it, so text
    mirrored this way is not hidden on the far side -- it is simply written twice.
    """
    return np.concatenate([points + offset, points - offset])


def _log_colourbar(
    rerun,  # noqa: ANN001
    path: str,
    limits: tuple[float, float],
    colouring: Colouring,
    unit: str,
    bounds: tuple[np.ndarray, np.ndarray],
) -> None:
    """The scale a field is coloured on, standing beside the fault.

    Rerun 0.35 has no screen-space overlay for a 3D view, so a colourbar has to be part
    of the scene rather than pinned to the window. It stands off the low corner of the
    scene's own bounding box, and is drawn as two strips crossed at right angles so
    that no camera azimuth catches it edge-on and loses it.

    The ticks come from :func:`_ticks`, the same helper the histogram axes use, so the
    two panels label the same field at the same values.
    """
    low_value, high_value = limits
    low, high = bounds
    reach = _reach(bounds)
    # As tall as the scene is deep, so it stands beside the fault rather than towering
    # over it -- with a floor, for a scene that is almost flat.
    length = max(0.6 * float(high[2] - low[2]), 0.12 * reach)
    width = 0.05 * length
    # Off the low corner in both horizontal directions, standing on the deepest cell.
    foot = np.array([low[0] - 0.08 * reach, low[1] - 0.08 * reach, low[2]])

    edges = foot[2] + np.linspace(0.0, length, COLOURBAR_CELLS + 1)
    centres = np.linspace(low_value, high_value, COLOURBAR_CELLS)
    quads = []
    for axis in (0, 1):
        span = np.zeros(3)
        span[axis] = width / 2.0
        for bottom, top in itertools.pairwise(edges):
            base = np.array([foot[0], foot[1], bottom])
            rise = np.array([0.0, 0.0, top - bottom])
            quads.append(
                np.stack(
                    [base - span, base + span, base + span + rise, base - span + rise]
                )
            )
    colours = np.tile(colouring(centres), (2, 1))
    rerun.log(
        f"{path}/colourbar",
        _mesh(rerun, np.stack(quads), colours),
        static=True,
    )

    positions, labels = [], []
    for value in _ticks(low_value, high_value):
        fraction = _fraction(np.array([value]), low_value, high_value)[0]
        positions.append(
            [foot[0] + width, foot[1] + width, foot[2] + fraction * length]
        )
        labels.append(f"{value:.3g}")
    positions.append([foot[0], foot[1], foot[2] + length + 0.06 * length])
    labels.append(unit)
    rerun.log(
        f"{path}/colourbar/labels",
        rerun.Points3D(
            np.array(positions),
            colors=[AXIS_TEXT],
            labels=labels,
            show_labels=True,
            radii=[UI_POINTS],
        ),
        static=True,
    )


def contour_levels(
    low: float, high: float, target: int = TARGET_CONTOURS
) -> np.ndarray:
    """Round times to draw isochrones at, spaced so the fault does not fill with lines.

    The coarsest spacing on :data:`CONTOUR_STEPS_S` that fits `target` lines across the
    range wins, which is what makes a thirty-second rupture come out in fives and a
    three-second one in halves without either being told which it is.

    The lowest onset is skipped: the front's own start is one subfault, and a contour
    through it is a point rather than a line.
    """
    span = high - low
    if not math.isfinite(span) or span <= 0.0:
        return np.array([])
    step = next(
        (step for step in CONTOUR_STEPS_S if span / step <= target), CONTOUR_STEPS_S[-1]
    )
    first = math.ceil(low / step) * step
    return np.arange(first if first > low else first + step, high, step)


def isochrones(values: np.ndarray, positions: np.ndarray, level: float) -> np.ndarray:
    """Where a lattice of values crosses `level`, as line segments in space.

    Marching squares, on the parameter lattice and then mapped onto the surface it is
    laid over, so a contour follows a curved fault rather than a plane through it.

    Written against the crossings rather than against the cells: the obvious version
    builds a ``(cells, 4, 3)`` array of every cell's four possible crossing points,
    which is 960 MB of float64 for a ten-million-cell fault and is all but empty --
    an isochrone touches the square root of the cells, not all of them.

    Parameters
    ----------
    values : np.ndarray
        ``(i, j)``. Non-finite entries take no part: a front that never arrived has no
        crossing rather than a crossing at infinity.
    positions : np.ndarray
        ``(i, j, 3)``, where each lattice point sits.
    level : float
        The value to trace.

    Returns
    -------
    np.ndarray
        ``(segments, 2, 3)``, each a straight run between two cell edges. Empty when
        the level lies outside the field.
    """
    finite = np.isfinite(values)
    above = np.where(finite, values, -np.inf) >= level
    # An edge is crossed when its ends straddle the level and both ends are real.
    down = (above[:, :-1] != above[:, 1:]) & finite[:, :-1] & finite[:, 1:]
    across = (above[:-1, :] != above[1:, :]) & finite[:-1, :] & finite[1:, :]

    # A cell's four edges, anticlockwise from the one along its low i side. Boolean per
    # cell rather than positions per cell: one byte each instead of ninety-six.
    cell = np.stack(
        [down[:-1, :], across[:, 1:], down[1:, :], across[:, :-1]], axis=-1
    ).reshape(-1, 4)
    rows, edges = np.nonzero(cell)
    if not rows.size:
        return np.empty((0, 2, 3))

    # Every cell crosses an even number of its edges, so `nonzero` -- ascending within
    # each row -- already pairs them. Two crossings is one segment either way round.
    # Four is a saddle, and pairing (0, 1) with (2, 3) resolves it one of the two ways
    # it can be resolved; nothing downstream can tell which.
    columns = values.shape[1] - 1
    i, j = rows // columns, rows % columns
    starts = (i + np.array([0, 0, 1, 0])[edges], j + np.array([0, 1, 0, 0])[edges])
    ends = (i + np.array([0, 1, 1, 1])[edges], j + np.array([1, 1, 1, 0])[edges])

    first, second = values[starts], values[ends]
    fraction = ((level - first) / (second - first))[:, None]
    crossings = positions[starts] + fraction * (positions[ends] - positions[starts])
    return crossings.reshape(-1, 2, 3)


def _mesh(rerun, corners: np.ndarray, colours: np.ndarray):  # noqa: ANN001
    """A `Mesh3D` of flat-coloured cells, fanned into triangles.

    The arity comes from ``corners.shape``: four for a quad lattice, three for a
    triangulation, and a fan from corner zero reduces to the two-triangle quad split.
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
    "Drawing",
    "Segment",
    "contour_levels",
    "drawn",
    "hot",
    "isochrones",
    "load",
    "moment_release",
    "rose",
    "statistics",
    "stride_for",
    "strided_corners",
    "view",
    "viridis",
]
