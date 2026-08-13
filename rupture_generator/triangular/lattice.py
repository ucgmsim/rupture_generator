"""The regular lattice over the parameter plane, and the two solvers that run on it.

**The geometry is curved and the solvers are flat.** A segment's shape is the Monge
patch ``X(u, v) = O + u e_u + v e_v + h(u, v) n`` that :mod:`~rupture_generator
.triangular.mesh` builds, and that is what supplies true areas, true depths, true
positions and the true outline. What runs *on* it is not a surface solver: both the
wavefront and the field sampler are computed on a regular ``(i, j)`` lattice over the
segment's own parameter rectangle -- the shadow of the fault on the plane it is a graph
over -- and their answers are projected back onto the faces.

That split is what this module is. It is not a compromise reached for want of a surface
solver; it is the arrangement the curvature study measured as keeping nearly all of the
value:

======================================  ==================  ===================
term                                    a flat model        this arrangement
======================================  ==================  ===================
moment delivered / target               0.9690              **1.0**, true areas
rigidity contribution                   0.9384              **1.0**, true depths
onset, median (Hikurangi)               +7.53 s             **-0.14 to -0.17 s**
======================================  ==================  ===================

The two large terms are removed *before they arise*, because area and depth are read
off the curved mesh and never off the lattice. What is left is the **metric error** --
path lengths measured in the plane rather than on the surface. Measured face by face
against the mesh solver this replaced, on both real interfaces: a median of -0.14 to
+0.03 s, a fifth percentile of -0.8 to -2.0 s, a worst cell at -6.0 s, and 12-30% of
the moment arriving more than half a second early. :func:`travel_times` carries the
full table. That is what this design costs; it is a modelling judgement rather than a
bug, and for scale the model's own deliberate onset perturbation is ~0.35 s and the
ruptures are 143 to 255 s long.

**The outline is carried by a slowness wall.** A lattice eikonal has no concept of "not
fault", and a subduction interface's parameter footprint is not convex -- Hikurangi is
10.3% concave against its hull and Puyseguer 14.2%, and the rectangle wastes 37-47% of
its cells. Off-fault cells are therefore raised to :data:`OFF_FAULT_SLOWNESS_FACTOR`
times the local slowness, which stops the front rather than slowing it. See that
constant for the measurements.

**Padding retires.** The SPDE sampler this replaced solved a PDE on a bounded domain
and its Neumann condition reflected the covariance back into the fault, which needed a
conforming pad built around every segment small enough to fold. Circulant embedding
pads and crops *by construction* (`sampling.py`), so the boundary problem does not
arise here at all -- including on the shipped crustal faults, which sit at 2.2-4
correlation lengths and used to warn.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

from rupture_generator import _kernels, sampling, timing

if TYPE_CHECKING:
    from rupture_generator.sampling import VonKarmanFilterParameters

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int64]]
BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]

OFF_FAULT_SLOWNESS_FACTOR = 10.0
"""What the slowness of an off-fault lattice cell is multiplied by.

**Derived from where the answer saturates, not chosen with a margin.** The wall was
swept over ×10, ×100, ×1000 and ×10\\ :sup:`5` on both the CFM Hikurangi and Puyseguer
interfaces at every study hypocentre, and

.. code-block:: text

    max | M(x10) - M(x10^5) |  =  0.0 s     exactly, everywhere on the fault

-- the arrival field is bit-identical from ×10 upward. There is nothing above ×10 left
to buy, so a larger factor would be a number with no measurement behind it. The wall is
finite because the kernel refuses non-finite slowness, and ×10 is where finite stops
mattering.

**It does not disturb the factorisation.** `eikonal.rs` warns that Fomel et al.'s
multiplicative split assumes a smooth medium with the singularity confined to the
source, and a hard slowness jump is exactly the structure that would trigger its
one-sided fallback. Measured rather than argued: on a uniform medium inside a
rectangular fault, where the exact answer is ``s·r``, the maximum error is
**1.3e-13 s** and it is *identical* with no wall and at ×10, ×10\\ :sup:`3`,
×10\\ :sup:`5` and ×10\\ :sup:`8`. ``τ ≡ 1`` survives the wall exactly.

**It stops the front rather than slowing it.** On an L-shaped fault whose geodesic must
round a reflex corner, the walled solve matches the closed-form around-the-corner
distance to **0.099 s** over a 20 s traverse, where the unwalled solve short-cuts
through the missing quadrant by 0.70 s median and 2.23 s worst.

**What it costs.** 1.0-1.5× the open solve: 2.2-2.7 s against 1.8-2.2 s on Hikurangi's
1650 × 664 lattice, against ~10 s for one meshFIM solve on the same interface's 1.39 M
faces.
"""


@dataclasses.dataclass(frozen=True)
class ParameterLattice:
    """A segment's parameter rectangle, cut into square cells at the mesh's own spacing.

    One of these is built per segment in
    :func:`~rupture_generator.triangular.pipeline.generate` and handed to both the
    sampler and the wavefront, because it is derived geometry and deriving it twice is
    the same mistake :class:`~rupture_generator.triangular.pipeline.SegmentGeometry`
    exists to avoid.

    **Axis order is the solvers' own, and it is not the parameter columns' order.**
    ``parameters_km`` is ``(u, v) = (strike, dip)``; a lattice grid is ``(i, j) =
    (dip, strike)``, which is what both :func:`~rupture_generator._kernels
    .eikonal_solve` and :mod:`~rupture_generator.sampling` index. The cells are square,
    so the two spacings are equal and only the *extents* carry the distinction -- which
    is precisely why it is written down here rather than left to be inferred.

    **Binning is lossless on a mesh built the way this package builds one, and it
    reduces exactly to the structured chart.** Both
    :func:`~rupture_generator.triangular.mesh.remesh` and
    :meth:`~rupture_generator.triangular.mesh.TriangleMesh.from_patches` cut the
    parameter plane into rectangles and split each into two triangles, whose centroids
    sit at ``1/3`` and ``2/3`` of the cell on each axis -- so both land in their own
    cell, the cells are the quads, and on a planar fault this lattice *is* the
    structured chart's grid. That is why the two tracks draw the same field there rather
    than merely a comparable one. On a triangulation that was not built that way -- a
    GOCAD surface used directly -- the spacing is the median parameter edge length per
    axis and the binning is a genuine average.

    Attributes
    ----------
    origin_km : FloatArray
        ``(2,)`` the ``(u, v)`` corner of cell ``(0, 0)``: the parameter domain's own
        minimum **vertex**, which on a mesh cut into quads is a quad corner.
    spacing_km : tuple of float
        Cell size, ``(strike, dip)`` -- the parameter columns' own order.
    shape : tuple of int
        ``(dip cells, strike cells)``.
    cell_of_face : IntArray
        ``(F,)`` the flat index into :attr:`shape` of the cell each face centre falls
        in. Both directions of transfer are this one array: binning is a
        ``bincount`` over it and projection is a gather from it.
    occupied : BoolArray
        :attr:`shape`, true where at least one face centre falls -- the fault's outline
        as the lattice sees it, and what the slowness wall is built from.
    """

    origin_km: FloatArray
    spacing_km: tuple[float, float]
    shape: tuple[int, int]
    cell_of_face: IntArray
    occupied: BoolArray

    @classmethod
    def of(cls, parameters_km: FloatArray, faces: IntArray) -> ParameterLattice:
        """Cut a segment's parameter rectangle at the spacing its own triangles have.

        **The spacing is read per axis, from the edges that move on that axis alone.**
        A quad split into two triangles contributes one edge along ``u``, one along
        ``v`` and one diagonal, so the median ``|Δu|`` over the edges with ``Δu ≠ 0``
        is the cell's own width and the median ``|Δv|`` its own height. One median over
        edge *lengths* would instead return something between the two on any mesh whose
        cells are not square, which is every mesh a config with different strike and dip
        subfault sizes produces.

        Parameters
        ----------
        parameters_km : FloatArray
            ``(V, 2)`` the vertices' ``(u, v)`` parameter coordinates.
        faces : IntArray
            ``(F, 3)`` vertex indices.

        Returns
        -------
        ParameterLattice

        Raises
        ------
        ValueError
            For a segment with no extent on one of its parameter axes -- there is no
            spacing to read from it, which is a degenerate chart rather than a fault.
        """
        corners_uv = parameters_km[faces]
        steps = np.abs(corners_uv - np.roll(corners_uv, 1, axis=1)).reshape(-1, 2)
        spacing = []
        for axis, name in enumerate(("strike", "dip")):
            moving = steps[steps[:, axis] > 0.0, axis]
            if not moving.size:
                raise ValueError(
                    f"every edge of this segment has the same {name} parameter "
                    "coordinate, so it has no extent along that axis and there is no "
                    "lattice spacing to read from it"
                )
            spacing.append(float(np.median(moving)))
        spacing_km = (spacing[0], spacing[1])

        # The origin is the parameter domain's own corner -- a **vertex**, not the first
        # face centre -- and that is load-bearing rather than tidy. On a mesh cut into
        # quads the two differ by a third of a cell, and that third is the whole margin:
        # a quad's two centroids sit at 1/3 and 2/3 of the cell from its corner, so
        # binning from the corner leaves a third of a cell of clearance to each edge,
        # while binning from the first centroid puts one of them exactly *on* an edge,
        # where f64 round-off in the subtraction drops it into the cell below. Measured
        # on a 6 x 6 quad mesh: one cell lost, and with it one cell of the fault.
        origin_km = parameters_km.min(axis=0)
        centres_uv = corners_uv.mean(axis=1)
        index = np.floor((centres_uv - origin_km) / np.asarray(spacing_km)).astype(
            np.int64
        )
        shape = (int(index[:, 1].max()) + 1, int(index[:, 0].max()) + 1)

        cell_of_face = index[:, 1] * shape[1] + index[:, 0]
        occupied = np.zeros(shape[0] * shape[1], dtype=bool)
        occupied[cell_of_face] = True
        return cls(
            origin_km=origin_km,
            spacing_km=spacing_km,
            shape=shape,
            cell_of_face=cell_of_face,
            occupied=occupied.reshape(shape),
        )

    @property
    def cell_counts(self) -> tuple[int, int]:
        """``(dip, strike)`` cell counts -- what the embedding is built on."""
        return self.shape

    @property
    def sampling_spacing_km(self) -> tuple[float, float]:
        """``(strike, dip)`` spacing -- the order the sampler takes."""
        return self.spacing_km

    @property
    def sweep_spacing_km(self) -> tuple[float, float]:
        """``(dip, strike)`` spacing, the order the fast sweep takes -- reversed."""
        return (self.spacing_km[1], self.spacing_km[0])

    def bin(self, values: FloatArray) -> FloatArray:
        """The mean of a per-face field in each cell; ``NaN`` where no face falls.

        Parameters
        ----------
        values : FloatArray
            ``(F,)`` one value per face.

        Returns
        -------
        FloatArray
            :attr:`shape`, with ``NaN`` off the fault.
        """
        cells = self.shape[0] * self.shape[1]
        total = np.bincount(self.cell_of_face, weights=values, minlength=cells)
        count = np.bincount(self.cell_of_face, minlength=cells)
        with np.errstate(invalid="ignore", divide="ignore"):
            binned = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        return binned.reshape(self.shape)

    def project(self, grid: FloatArray) -> FloatArray:
        """A lattice field read back onto the faces, one value per face.

        The exact inverse of :meth:`bin` in the sense that matters: a face takes the
        value of the cell its own centre falls in, so no interpolation stencil is
        involved and a face whose cell was seeded at ``t = 0`` reads exactly zero.

        Parameters
        ----------
        grid : FloatArray
            :attr:`shape`.

        Returns
        -------
        FloatArray
            ``(F,)``.
        """
        return np.asarray(grid, dtype=np.float64).reshape(-1)[self.cell_of_face]

    def cell_of(self, face: int) -> tuple[int, int]:
        """The ``(i, j)`` cell one face's centre falls in -- how a seed is placed.

        Parameters
        ----------
        face : int
            A face index.

        Returns
        -------
        tuple of int
            ``(dip index, strike index)``.
        """
        flat = int(self.cell_of_face[face])
        return divmod(flat, self.shape[1])


def _filled(grid: FloatArray, occupied: BoolArray) -> FloatArray:
    """Give every off-fault cell the value of the nearest on-fault one.

    **The fill rule, stated because it is a choice and not because it matters.**
    A rectangular lattice has to invent a medium off the fault, and the invention is
    arbitrary: nearest-neighbour continuation is the choice most generous to a lattice
    solver, because nothing about the medium then tells the sweep where the fault
    stops.

    Without :data:`OFF_FAULT_SLOWNESS_FACTOR` this choice is worth a **factor of five**
    in the error. On Puyseguer the front routes through a re-entrant bay where
    nearest-neighbour fill copies deep fast rock (0.307 s/km) into ground the real
    detour crosses as shallow slow rock (up to 0.51 s/km) -- so the invented medium is
    both absent *and* quick, and 80% of the unwalled error is that rather than pure
    geometric short-cutting, which is bounded at 0.76 s.

    With the wall the front does not enter these cells at all, so the rule is
    immaterial. It is written down anyway: a reader who found an arbitrary fill here
    with no comment would reasonably assume it had been chosen with care, and the point
    is that it has not been and does not need to be.

    Parameters
    ----------
    grid : FloatArray
        The binned field, ``NaN`` off the fault.
    occupied : BoolArray
        Where the fault is.

    Returns
    -------
    FloatArray
        The same grid with no ``NaN``.
    """
    from scipy import ndimage

    _, nearest = ndimage.distance_transform_edt(~occupied, return_indices=True)
    return grid[tuple(nearest)]


def travel_times(
    lattice: ParameterLattice,
    depth_km: FloatArray,
    shear_speed_km_s: FloatArray,
    params: timing.SpeedParams,
    seeds: list[tuple[int, int, float]],
) -> FloatArray:
    """S7: first-arrival times per face, by a walled factored sweep on the lattice.

    The four steps, and the one that is the point of the design:

    .. code-block:: text

        1  slowness per face, from the mesh's own TRUE centre depths
        2  binned onto the lattice, off-fault cells filled            (`_filled`)
        3  off-fault cells walled                    (`OFF_FAULT_SLOWNESS_FACTOR`)
        4  factored fast sweep, then projected back onto faces

    **Step 1 is why the geometry stays curved.** The speed field is
    :func:`~rupture_generator.timing.speed_field` unchanged -- it is elementwise in
    depth and shear speed -- and the depths it reads are the curved surface's, not the
    parameter plane's. The curvature study measured that substitution as removing
    essentially all of the timing error a flat model carries: a flat model's median
    onset error on Hikurangi is +7.53 s and this one's is -0.14 to -0.17 s, and the
    difference is almost entirely that the rock is sampled where the fault actually is.

    **What is left is the metric error**, which no wall and no fill removes: the sweep
    measures ``|Δ(u, v)|`` where the front travels ``|ΔX|``, so paths are short by the
    surface's own stretch.

    **Measured against the mesh solver this replaced**, face by face, on the two CFM
    subduction interfaces at a 0.5 km cut, before it was deleted. Both solves read the
    *same* true-depth slowness on the *same* faces, so the residual is the whole of what
    the swap costs. Signed, in seconds, negative meaning the hybrid arrives early:

    =========== ========= ====== ====== ====== ====== ====== ====== ====== ======
    interface   site      p01    p05    median p95    min    max    w<-0.5 w<-1
    =========== ========= ====== ====== ====== ====== ====== ====== ====== ======
    Hikurangi   northern  -1.68  -0.87  -0.14  +0.11  -2.84  +0.34  12.6%  1.8%
    Hikurangi   central   -1.41  -0.85  -0.07  +0.17  -4.06  +0.36  11.9%  3.1%
    Hikurangi   southern  -1.72  -1.05  -0.09  +0.06  -4.24  +0.36  19.7%  6.4%
    Puyseguer   northern  -2.77  -1.33  -0.09  +0.08  -4.56  +0.66  18.0%  9.5%
    Puyseguer   central   -1.83  -0.78  +0.03  +0.19  -3.71  +0.75  11.9%  5.1%
    Puyseguer   southern  -4.01  -1.97  -0.14  +0.06  -6.00  +0.90  29.8%  23.9%
    =========== ========= ====== ====== ====== ====== ====== ====== ====== ======

    The last two columns are **moment**-weighted: the share of the earthquake's own
    moment arriving more than half a second, and more than a second, early. The ruptures
    are 143 to 255 s long and the model's own deliberate onset perturbation is ~0.35 s.

    **Two of those columns are two errors of opposite sign.** The medians are *smaller*
    than the -0.14 to -0.30 s the metric error alone predicts, because the metric error
    is partly cancelled by the solver change: this is a second-order factored sweep where
    the mesh solver was first-order by design, and that difference is a systematic
    +0.08 to +0.14 s the other way. The tails do not cancel and are the honest number.

    **It also costs less.** 2.3-3.2 s against 10.7-11.1 s on Hikurangi's 1.39 M faces,
    0.8-1.0 s against 3.5 s on Puyseguer's 494 thousand -- 3.5 to 4.5 times faster, wall
    included.

    Parameters
    ----------
    lattice : ParameterLattice
        The segment's lattice.
    depth_km : FloatArray
        ``(F,)`` each face's **true** centre depth, positive down.
    shear_speed_km_s : FloatArray
        ``(F,)`` from the velocity model at those depths.
    params : timing.SpeedParams
        How fast the front travels here.
    seeds : list of tuple
        ``(i, j, t0_s)`` -- where the front starts and when. One triple for a
        hypocentre; the kernel takes several and returns the pointwise minimum, which
        is what first arrival from several sources means.

    Returns
    -------
    FloatArray
        ``(F,)`` travel times in seconds.

    Raises
    ------
    ValueError
        For a rupture speed the front cannot travel at, or a seed outside the lattice.
    """
    speed_km_s = timing.speed_field(depth_km, shear_speed_km_s, params)
    slowness = _filled(lattice.bin(1.0 / speed_km_s), lattice.occupied)
    walled = np.where(lattice.occupied, slowness, slowness * OFF_FAULT_SLOWNESS_FACTOR)
    grid = _kernels.eikonal_solve(
        np.ascontiguousarray(walled), lattice.sweep_spacing_km, seeds
    )
    return lattice.project(grid)


def draw_field(
    lattice: ParameterLattice,
    covariance: VonKarmanFilterParameters,
    rng: np.random.Generator,
) -> FloatArray:
    """One von Karman field on the lattice, projected onto the faces.

    :func:`~rupture_generator.sampling.von_karman_grid` on the segment's parameter
    rectangle, read back onto the faces. The embedding covers the whole rectangle
    including the cells no face falls in; those are simply never read, which costs the
    37-47% of a subduction interface's rectangle that is off-fault and buys the outline
    being irrelevant to the sampler.

    **The delivered correlation length is measured in PLANAR distance, and that is a
    modelling choice with a live alternative.** Circulant embedding builds the
    covariance from separations in the parameter plane; a surface-intrinsic sampler
    builds it from separations on the curved surface, and on Hikurangi the two differ
    by 8.7% along strike and 4.0% down dip -- of which only 0.5-0.9% is the projection
    stretch, the rest being that the Laplace-Beltrami operator on the real surface
    delivers a shorter length than the same operator on its shadow.

    The plane is chosen because **Mai & Beroza (2002) fitted 44 finite-source
    inversions on planar faults**: "along strike" in their equations (4) and (5) is a
    planar strike, the two definitions coincide on the sources the relations were
    regressed from, and reading them with a surface-intrinsic metric applies the
    numbers outside the geometry they were measured in.

    The argument the other way is real and is not settled here: a fitted surface frame's
    azimuth is a property of *how the patch is cut* rather than of the surface --
    splitting Hikurangi in two moves it from 218.5° to 211.6° and 223.7° -- so a
    segmented interface would get a 12-17° discontinuity at a seam for no physical
    reason, worth 10-25% in delivered along-strike length. That bites only when one
    surface is split into segments, which is not what this package does to an interface.

    Parameters
    ----------
    lattice : ParameterLattice
        The segment's lattice.
    covariance : VonKarmanFilterParameters
        The patch structure this segment's magnitude implies.
    rng : np.random.Generator
        The stage's own substream.

    Returns
    -------
    FloatArray
        ``(F,)`` one value per face.

    Raises
    ------
    ValueError
        If the embedding this segment needs is past
        :data:`~rupture_generator.sampling.MAXIMUM_EMBEDDING_CELLS`.
    """
    return lattice.project(
        sampling.von_karman_grid(
            lattice.cell_counts, lattice.sampling_spacing_km, covariance, rng
        )
    )


__all__ = [
    "OFF_FAULT_SLOWNESS_FACTOR",
    "ParameterLattice",
    "draw_field",
    "travel_times",
]
