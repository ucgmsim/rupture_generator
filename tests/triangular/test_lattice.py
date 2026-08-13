"""The parameter lattice, and the two properties the whole hybrid rests on.

The architecture is a curved mesh supplying areas, depths, positions and an outline, and
a *flat* lattice over its parameter plane carrying both solvers. Two things have to be
true for that to be honest, and neither is obvious:

- **the wall stops the front without disturbing the factorisation**, so a rectangular
  lattice can carry a non-convex fault's outline at all, and
- **the projection is a gather**, so the hypocentre's own subfault reads the seed time
  exactly rather than an interpolation of it.

Both are measured here against closed forms, not against a second transcription of the
code. What this file deliberately does *not* contain is the comparison against the mesh
solver it replaced: that solver is gone, the comparison needs the CFM interfaces and
minutes of solve time, and its numbers live in
:func:`~rupture_generator.triangular.lattice.travel_times`' docstring where a reader
meets them.
"""

from __future__ import annotations

import numpy as np
import pytest

from rupture_generator import _kernels
from rupture_generator.sampling import (
    MAXIMUM_EMBEDDING_CELLS,
    VonKarmanFilterParameters,
    von_karman_grid,
)
from rupture_generator.triangular.lattice import (
    OFF_FAULT_SLOWNESS_FACTOR,
    ParameterLattice,
    _filled,
)

SLOWNESS = 0.25
"""s/km, uniform, so ``s |x - x0|`` is the exact arrival and the wall is the only
variable."""


def _uniform_case(n: int, factor: float | None) -> np.ndarray:
    """A rectangular fault inside a larger rectangle, source at its centre.

    Inside the fault the exact arrival is ``s |x - x0|`` whatever happens outside,
    because the straight ray never leaves. So the error against that is a measurement of
    what the wall does to the factorisation and of nothing else.
    """
    spacing = 40.0 / n
    grid = np.full((n, n), SLOWNESS)
    mask = np.zeros((n, n), dtype=bool)
    low, high = n // 4, 3 * n // 4
    mask[low:high, low:high] = True
    if factor is not None:
        grid = np.where(mask, grid, grid * factor)

    seed = (n // 2, n // 2)
    times = _kernels.eikonal_solve(grid, (spacing, spacing), [(*seed, 0.0)])
    rows, columns = np.mgrid[0:n, 0:n]
    exact = SLOWNESS * spacing * np.hypot(rows - seed[0], columns - seed[1])
    return (times - exact)[mask]


@pytest.mark.parametrize("factor", (None, 10.0, 1.0e3, 1.0e5, 1.0e8))
def test_the_wall_leaves_the_factorisation_exact(factor: float | None) -> None:
    """**The concern that turned out to be unfounded**, so it is asserted rather than said.

    `eikonal.rs` warns that Fomel et al.'s multiplicative split assumes a smooth medium
    with the singularity confined to the source, and that its one-sided fallback costs a
    factor of six on a gradient. A hard slowness jump is exactly the structure that would
    trigger it -- so this measures, on a case whose answer is known in closed form.

    Measured at ``n = 200``: ``max |error|`` is **1.3e-13 s** with no wall and *the same
    1.3e-13 s* at every wall factor from ×10 to ×10^8. ``τ ≡ 1`` survives the wall
    exactly and the fallback is never reached.
    """
    error = _uniform_case(200, factor)
    assert np.abs(error).max() < 1.0e-12


def test_the_wall_saturates_at_ten() -> None:
    """Where :data:`OFF_FAULT_SLOWNESS_FACTOR` comes from: there is nothing above it.

    The constant is 10 because the answer stops changing there, which is the only
    defensible way to pick it -- a larger factor "to be safe" would be a number with no
    measurement behind it. On the two real interfaces the study measured
    ``max |M(x10) - M(x10^5)| = 0.0 s`` exactly; here the same statement is made on the
    non-convex case below, where the wall is doing work rather than sitting idle.
    """
    mask, _seed, _exact, _spacing = _elbow_geometry(300)
    ten = _elbow_arrivals(300, OFF_FAULT_SLOWNESS_FACTOR)
    huge = _elbow_arrivals(300, 1.0e5)
    # On the fault, which is the only place an arrival means anything -- inside the wall
    # the two solves are of course different media.
    assert np.array_equal(ten[mask], huge[mask])


def _elbow_geometry(n: int) -> tuple[np.ndarray, tuple[int, int], np.ndarray, float]:
    """An L-shaped fault, its seed, its exact geodesic arrivals, and the spacing.

    The reference is the exact distance *inside* the L: the straight line where the
    reflex corner is visible, and the two-segment path through the corner where it is
    not. So a walled solve is checked against an answer that owes nothing to another
    solver.
    """
    spacing = 40.0 / n
    arm = n // 3
    mask = np.zeros((n, n), dtype=bool)
    mask[:arm, :] = True
    mask[:, :arm] = True
    seed = (arm // 2, n - arm // 2)

    rows, columns = np.mgrid[0:n, 0:n]
    corner = (arm - 1, arm - 1)
    direct = np.hypot(rows - seed[0], columns - seed[1])
    via = np.hypot(corner[0] - seed[0], corner[1] - seed[1]) + np.hypot(
        rows - corner[0], columns - corner[1]
    )
    # The vertical arm sits below the top bar, so a straight line from the top bar's far
    # right crosses the missing quadrant: the corner is the only way in.
    detour = (rows >= arm) & (columns < arm)
    exact = SLOWNESS * spacing * np.where(detour, via, direct)
    return mask, seed, exact, spacing


def _elbow_arrivals(n: int, factor: float) -> np.ndarray:
    mask, seed, _exact, spacing = _elbow_geometry(n)
    grid = np.where(mask, SLOWNESS, SLOWNESS * factor)
    return _kernels.eikonal_solve(grid, (spacing, spacing), [(*seed, 0.0)])


def test_the_wall_stops_the_front_rather_than_slowing_it() -> None:
    """The property a mask has to have, on the only case where the two differ.

    A front that is merely *slowed* still crosses the gap and arrives early on the far
    side; one that is stopped goes round. On an L whose geodesic must round a reflex
    corner, the walled solve matches the closed-form around-the-corner distance to
    **0.099 s** over a 20 s traverse, where the same solve with no wall short-cuts
    through the missing quadrant by **0.70 s** median and 2.23 s worst.

    That gap -- two orders of magnitude between the wall's error and the error it
    removes -- is the whole argument for masking rather than living with an open solve.
    """
    n = 300
    mask, seed, exact, spacing = _elbow_geometry(n)
    rows, columns = np.mgrid[0:n, 0:n]
    arm = n // 3
    beyond = mask & (rows >= arm) & (columns < arm)

    walled = _elbow_arrivals(n, OFF_FAULT_SLOWNESS_FACTOR)
    open_solve = _kernels.eikonal_solve(
        np.full((n, n), SLOWNESS), (spacing, spacing), [(*seed, 0.0)]
    )

    assert np.abs((walled - exact)[beyond]).max() < 0.2
    assert np.median((open_solve - walled)[beyond]) < -0.5


def test_the_wall_makes_the_off_fault_fill_irrelevant() -> None:
    """Why :func:`~rupture_generator.triangular.lattice._filled` may be arbitrary.

    A rectangular lattice must invent a medium off the fault, and the invention changes
    the answer by a factor of five when the front is allowed into it -- 80% of the
    unwalled error on Puyseguer is nearest-neighbour fill copying deep fast rock into a
    bay the real detour crosses as shallow slow rock, rather than geometric
    short-cutting, which is bounded at 0.76 s.

    With the wall the front never enters those cells, so two completely different fills
    -- the nearest on-fault value, and the fastest rock anywhere on the fault, which is
    the most favourable possible invention -- give **bit-identical** arrivals on the
    fault. That is what licenses not choosing carefully.
    """
    n = 300
    mask, seed, _exact, spacing = _elbow_geometry(n)
    # A medium with real structure, so a fill that leaked would show up: slowness
    # doubling down the grid, which is a fault getting slower with depth.
    rows, _columns = np.mgrid[0:n, 0:n]
    on_fault = SLOWNESS * (1.0 + rows / n)
    binned = np.where(mask, on_fault, np.nan)

    nearest = _filled(binned, mask)
    fastest = np.where(mask, on_fault, on_fault[mask].min())

    arrivals = [
        _kernels.eikonal_solve(
            np.where(mask, grid, grid * OFF_FAULT_SLOWNESS_FACTOR),
            (spacing, spacing),
            [(*seed, 0.0)],
        )[mask]
        for grid in (nearest, fastest)
    ]
    assert np.array_equal(*arrivals)


# ============================================================================
# The lattice itself
# ============================================================================


def _quad_mesh(
    rows: int, columns: int, du: float, dv: float
) -> tuple[np.ndarray, np.ndarray]:
    """A ``rows x columns`` parameter lattice split into two triangles a quad.

    The shape both :func:`~rupture_generator.triangular.mesh.remesh` and
    :meth:`~rupture_generator.triangular.mesh.TriangleMesh.from_patches` produce.
    """
    u, v = np.meshgrid(
        np.arange(columns + 1) * du, np.arange(rows + 1) * dv, indexing="xy"
    )
    parameters = np.stack([u.ravel(), v.ravel()], axis=-1)
    index = np.arange((rows + 1) * (columns + 1)).reshape(rows + 1, columns + 1)
    near = index[:-1, :-1].ravel()
    far = index[:-1, 1:].ravel()
    below = index[1:, :-1].ravel()
    opposite = index[1:, 1:].ravel()
    faces = np.concatenate(
        [
            np.stack([near, far, opposite], axis=-1),
            np.stack([near, opposite, below], axis=-1),
        ]
    )
    return parameters, faces


def test_the_lattice_recovers_the_grid_the_mesh_was_cut_on() -> None:
    """Rectangular cells, and the spacing read per axis rather than from edge lengths.

    A quad contributes one edge along ``u``, one along ``v`` and one diagonal, so a
    median over edge *lengths* would return something between the two spacings on any
    mesh whose cells are not square -- which is every mesh a config with different strike
    and dip subfault sizes produces. Reading each axis from the edges that move on that
    axis alone is exact instead.
    """
    parameters, faces = _quad_mesh(5, 9, du=2.0, dv=0.5)
    lattice = ParameterLattice.of(parameters, faces)

    assert lattice.shape == (5, 9)
    assert lattice.spacing_km == pytest.approx((2.0, 0.5))
    assert lattice.sweep_spacing_km == pytest.approx((0.5, 2.0))
    assert lattice.occupied.all()
    # Exactly the two triangles of its own quad in every cell, which is what makes
    # binning lossless and projection an identity on a mesh built this way.
    assert np.array_equal(np.bincount(lattice.cell_of_face), np.full(45, 2))


def test_projection_is_a_gather_so_a_seeded_cell_reads_its_seed_exactly() -> None:
    """**Where the pinned hypocentre onset comes from**, and it is one line of arithmetic.

    `stages.apply_perturbation` zeroes the hypocentre's perturbation, so its onset is
    exactly the delay -- but only if its *travel time* is exactly zero. A scheme that
    interpolated a lattice field onto face centres would leave that a fraction of a cell
    late, in the one quantity `MESH.md` singles out as having no perturbation to hide
    behind. Reading each face's own cell cannot: the seeded cell holds ``t0`` and every
    face in it reads ``t0``.
    """
    parameters, faces = _quad_mesh(6, 6, du=1.0, dv=1.0)
    lattice = ParameterLattice.of(parameters, faces)
    grid = np.arange(36, dtype=np.float64).reshape(6, 6)

    projected = lattice.project(grid)
    assert np.array_equal(projected, grid.ravel()[lattice.cell_of_face])
    # And the round trip: binning a projected field gives that field back, because the
    # projection is constant on each cell.
    assert np.array_equal(lattice.bin(projected), grid)


def test_a_chart_with_no_extent_on_an_axis_is_refused_by_name() -> None:
    """A degenerate parameter domain has no spacing to read, so it says so."""
    parameters = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    with pytest.raises(ValueError, match="no extent along that axis"):
        ParameterLattice.of(parameters, faces)


# ============================================================================
# The embedding cap, which used to be a suggestion
# ============================================================================


def test_an_embedding_past_the_cap_is_refused_rather_than_allocated() -> None:
    """**The bypass, closed.** A bound that does not bind is worse than no bound.

    `_candidate_extents` used to break out of its search on the first over-cap candidate,
    return an empty list, and then fall through to ``candidates or [smallest]`` -- which
    handed back the minimum embedding *without checking it against the cap at all*. On the
    CFM Hikurangi interface cut at 100 m that is 111.8 M cells against a 67.1 M limit,
    allocated silently, and :data:`~rupture_generator.sampling.MAXIMUM_EMBEDDING_CELLS`
    reported nothing.

    A grid whose minimum embedding is past the cap has no field this machine can draw,
    because a circulant embedding is at least twice the grid on each axis and there is
    nothing smaller to try. So it refuses, names both counts, and names the two things a
    caller can do.
    """
    side = int(np.sqrt(MAXIMUM_EMBEDDING_CELLS)) // 2 + 64
    covariance = VonKarmanFilterParameters(1.0, 1.0)

    with pytest.raises(ValueError, match="past the") as raised:
        von_karman_grid((side, side), (0.1, 0.1), covariance, np.random.default_rng(0))
    message = str(raised.value)
    assert f"{MAXIMUM_EMBEDDING_CELLS:,}" in message
    assert "larger subfaults" in message


def test_a_grid_that_fits_still_draws() -> None:
    """The cap binds where it should and nowhere else.

    Every shipped example is three orders below it, and the CFM Hikurangi interface at
    400 m -- a 2075 x 830 lattice embedding to 6.9 M cells -- is a tenth of it. This is
    the smaller half of that statement: a grid under the cap is drawn without comment.
    """
    field = von_karman_grid(
        (32, 64),
        (1.0, 1.0),
        VonKarmanFilterParameters(8.0, 6.0),
        np.random.default_rng(3),
    )
    assert field.shape == (32, 64)
    assert np.isfinite(field).all()
