"""The regular lattice over the parameter plane, and the two solvers that run on it.

A segment's shape is the Monge patch ``X(u, v) = O + u e_u + v e_v + h(u, v) n`` that
:mod:`~rupture_generator.triangular.mesh` builds, and that is what supplies true areas,
depths, positions and outline. The wavefront and the field sampler run instead on a
regular ``(i, j)`` lattice over the segment's own parameter rectangle -- the fault's
shadow on the plane it is a graph over -- and their answers are projected back onto its
faces. The cost is the metric error: path lengths measured in the plane rather than on
the surface. HYBRID.md carries the curvature study behind the arrangement.

A lattice eikonal has no concept of "not fault", so the outline is carried by a slowness
wall on the off-fault cells -- see :data:`OFF_FAULT_SLOWNESS_FACTOR`.

References
----------
Mai, P. M., & Beroza, G. C. (2002). A spatial random field model to characterize
complexity in earthquake slip. *Journal of Geophysical Research*, 107(B11), 2308.
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

Swept over ×10 to ×10\\ :sup:`5` on both CFM subduction interfaces: the arrival field is
bit-identical from ×10 upward, so there is nothing above it left to buy. The wall is
finite because the kernel refuses non-finite slowness. It stops the front rather than
slowing it, and does not disturb Fomel et al.'s multiplicative factorisation -- on a
uniform medium inside a rectangular fault the maximum error is 1.3e-13 s with the wall
and without it, so ``τ ≡ 1`` survives it exactly.
"""


@dataclasses.dataclass(frozen=True)
class ParameterLattice:
    """A segment's parameter rectangle, cut into square cells at the mesh's own spacing.

    Built once per segment in :func:`~rupture_generator.triangular.pipeline.generate`
    and handed to both the sampler and the wavefront.

    Axis order is the solvers' own and not the parameter columns': ``parameters_km`` is
    ``(u, v) = (strike, dip)``, while a lattice grid is ``(i, j) = (dip, strike)``,
    which is what :func:`~rupture_generator._kernels.eikonal_solve` and
    :mod:`~rupture_generator.sampling` index. The cells are square, so only the
    *extents* carry the distinction.

    Binning is lossless on a mesh this package built: its quads are split into two
    triangles whose centroids sit at ``1/3`` and ``2/3`` of the cell on each axis, so
    the cells are the quads and on a planar fault this lattice *is* the structured
    chart's grid. On a GOCAD surface used directly, the spacing is the median parameter
    edge length per axis and the binning is a genuine average.

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
        in. Binning is a ``bincount`` over it and projection is a gather from it.
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

        The spacing is read per axis, from the edges that move on that axis alone: the
        median ``|Δu|`` over the edges with ``Δu ≠ 0`` is the cell's own width and the
        median ``|Δv|`` its own height. One median over edge *lengths* would instead
        return something between the two on any mesh whose cells are not square.

        ``parameters_km`` is ``(V, 2)``, the vertices' ``(u, v)`` parameter coordinates,
        and ``faces`` is ``(F, 3)`` vertex indices.

        Returns
        -------
        ParameterLattice

        Raises
        ------
        ValueError
            For a segment with no extent on one of its parameter axes -- there is no
            spacing to read from it.
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

        # The origin is the parameter domain's own corner -- a vertex, not the first
        # face centre. A quad's two centroids sit at 1/3 and 2/3 of the cell from its
        # corner, so binning from the first centroid puts the other exactly *on* a cell
        # edge, where f64 round-off drops it into the cell below: one cell of fault
        # lost.
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
        """The mean of a per-face ``(F,)`` field in each cell; ``NaN`` where none falls.

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

        ``grid`` is :attr:`shape`. A face takes the value of the cell its own centre
        falls in -- a gather with no interpolation stencil, so a face whose cell was
        seeded at ``t = 0`` reads zero.

        Returns
        -------
        FloatArray
            ``(F,)``.
        """
        return np.asarray(grid, dtype=np.float64).reshape(-1)[self.cell_of_face]

    def cell_of(self, face: int) -> tuple[int, int]:
        """The ``(i, j)`` cell one face's centre falls in -- how a seed is placed.

        Returns
        -------
        tuple of int
            ``(dip index, strike index)``.
        """
        flat = int(self.cell_of_face[face])
        return divmod(flat, self.shape[1])


def _filled(grid: FloatArray, occupied: BoolArray) -> FloatArray:
    """Give every off-fault cell the value of the nearest on-fault one.

    A rectangular lattice has to invent a medium off the fault and the invention is
    arbitrary; nearest-neighbour continuation is the choice most generous to a lattice
    solver, since nothing about the medium then tells the sweep where the fault stops.
    With :data:`OFF_FAULT_SLOWNESS_FACTOR` the front does not enter these cells at all,
    so the rule is immaterial -- without it, it is worth a factor of five in the error.
    ``grid`` is the binned field, ``NaN`` off the fault, and ``occupied`` is where the
    fault is.

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

    .. code-block:: text

        1  slowness per face, from the mesh's own TRUE centre depths
        2  binned onto the lattice, off-fault cells filled            (`_filled`)
        3  off-fault cells walled                    (`OFF_FAULT_SLOWNESS_FACTOR`)
        4  factored fast sweep, then projected back onto faces

    Step 1 is why the geometry stays curved. The speed field is
    :func:`~rupture_generator.timing.speed_field` unchanged -- elementwise in depth and
    shear speed -- and the depths it reads are the curved surface's, not the parameter
    plane's, so the rock is sampled where the fault actually is.

    What is left is the metric error, which no wall and no fill removes: the sweep
    measures ``|Δ(u, v)|`` where the front travels ``|ΔX|``, so paths are short by the
    surface's own stretch. Measured face by face against the mesh solver this replaced,
    on both CFM subduction interfaces, that is a median of -0.14 to +0.03 s, a worst
    cell at -6.0 s, and 12-30% of the moment arriving more than half a second early,
    against ruptures 143 to 255 s long. HYBRID.md carries the full table.

    ``depth_km`` is ``(F,)`` each face's **true** centre depth, positive down, and
    ``shear_speed_km_s`` is ``(F,)`` from the velocity model at those depths.
    ``seeds`` are ``(i, j, t0_s)`` -- one triple for a hypocentre, and the kernel takes
    several and returns the pointwise minimum.

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
    including the 37-47% of a subduction interface's cells that no face falls in; those
    are simply never read, so the outline is irrelevant to the sampler.

    The delivered correlation length is measured in **planar** distance, because Mai &
    Beroza (2002) fitted 44 finite-source inversions on planar faults: "along strike" in
    their equations (4) and (5) is a planar strike, and reading them with a
    surface-intrinsic metric applies the numbers outside the geometry they were measured
    in. On Hikurangi the two metrics differ by 8.7% along strike and 4.0% down dip.

    ``covariance`` is the patch structure this segment's magnitude implies, and ``rng``
    the stage's own substream.

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
