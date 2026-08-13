"""What the triangular eikonal solver must satisfy, and what it measurably costs.

Three kinds of test live here, and they are kept apart on purpose.

**The contract**, ported from `crates/kernels/tests/eikonal_contract.rs`. Every property
that file asserts of a first-arrival solver is asserted here of this one, with the
lattice replaced by a triangulation. It is the safety net `MESH.md` says to port first,
and the properties it deliberately does *not* assert -- distance monotonicity, a sharp
global Lipschitz bound, convergence of the worst *relative* error -- are false for a
discrete solution and are not asserted here either.

**The analytic checks.** Accuracy is judged against closed-form answers -- a uniform
medium, and a constant gradient where the rays are circular arcs -- never against
another implementation. The one place a second solver appears is the reduction test,
where agreeing with `_kernels.eikonal_solve` is the point rather than the standard.

**The measurements.** `MESH.md` requires four numbers reported rather than asserted:
the convergence slope for each of the two boundary conditions, the systematic error
against distance from the source, the ``r0`` sweep, and the jump cell. Those tests
print what they measured and assert only the bound the plan states, so the number in
the docstring and the number the test produces cannot drift apart.

Mesh sizes are capped at 257x257 vertices (66 thousand vertices, 132 thousand faces,
about 40 MB of corner table). Fu et al.'s own study goes to 1,048,576 vertices; the
sixteen-fold step past the cap is not taken here because this solver is numpy rather
than CUDA and the slope is already flat over four refinements.
"""

from __future__ import annotations

import itertools
import pathlib
import warnings
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from rupture_generator import _kernels
from rupture_generator.triangular import fim
from rupture_generator.triangular.fim import FloatArray, IntArray

# ============================================================================
# Meshes to solve on
# ============================================================================

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"


def lattice(
    cells: int, extent_km: float = 16.0, *, alternate: bool = False
) -> tuple[FloatArray, IntArray]:
    """A regular triangulation of a square, in the ``z = 0`` plane.

    ``alternate`` flips the diagonal on a chequerboard. Both splits are regular
    triangulations of the same square; the difference is whether the mesh has a
    preferred diagonal direction, which is the largest single term in the first-order
    error and worth being able to turn on and off.
    """
    axis = np.linspace(0.0, extent_km, cells)
    down, across = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack([across.ravel(), np.zeros(cells * cells), down.ravel()], axis=1)
    index = np.arange(cells * cells).reshape(cells, cells)
    top_left = index[:-1, :-1].ravel()
    bottom_left = index[1:, :-1].ravel()
    bottom_right = index[1:, 1:].ravel()
    top_right = index[:-1, 1:].ravel()
    if not alternate:
        faces = np.concatenate(
            [
                np.stack([top_left, bottom_left, bottom_right], axis=1),
                np.stack([top_left, bottom_right, top_right], axis=1),
            ]
        )
    else:
        rows, columns = np.meshgrid(
            np.arange(cells - 1), np.arange(cells - 1), indexing="ij"
        )
        flip = ((rows + columns) % 2 == 1).ravel()[:, None]
        faces = np.concatenate(
            [
                np.where(
                    flip,
                    np.stack([top_left, bottom_left, top_right], axis=1),
                    np.stack([top_left, bottom_left, bottom_right], axis=1),
                ),
                np.where(
                    flip,
                    np.stack([bottom_left, bottom_right, top_right], axis=1),
                    np.stack([top_left, bottom_right, top_right], axis=1),
                ),
            ]
        )
    return vertices, faces


def scattered(
    count: int, extent_km: float = 16.0, seed: int = 20260813
) -> tuple[FloatArray, IntArray]:
    """A Delaunay triangulation of scattered points, lifted flat.

    Unstructured, so it carries obtuse triangles and exercises the virtual-edge
    unfolding; Delaunay, so it is the triangulation `MESH.md`'s Component 1 actually
    builds, in the parameter domain where it maximises the minimum angle.
    """
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    edge = int(np.sqrt(count))
    axis = np.linspace(0.0, extent_km, edge)
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    inside = (grid > 0.0).all(axis=1) & (grid < extent_km).all(axis=1)
    grid[inside] += rng.uniform(-0.35, 0.35, size=(int(inside.sum()), 2)) * (
        extent_km / (edge - 1)
    )
    mesh = Delaunay(grid)
    vertices = np.stack([grid[:, 1], np.zeros(grid.shape[0]), grid[:, 0]], axis=1)
    return vertices, np.asarray(mesh.simplices, dtype=np.int64)


def warped(
    cells: int, extent_km: float = 16.0, amplitude_km: float = 1.6
) -> tuple[FloatArray, IntArray]:
    """A lattice lifted off its own plane -- a genuinely curved Monge patch.

    ``h(u, v) = A sin(pi u / L) sin(pi v / L)`` gives ``|grad h|`` up to about 0.31 at
    ``A = 1.6`` on a 16 km patch, which is the ``|grad h| <~ 0.33`` `MESH.md` measures
    on the worst shipped surface. Nothing here has an analytic answer; it is where the
    contract properties are checked on a surface that is not flat.
    """
    vertices, faces = lattice(cells, extent_km)
    lift = (
        amplitude_km
        * np.sin(np.pi * vertices[:, 0] / extent_km)
        * np.sin(np.pi * vertices[:, 2] / extent_km)
    )
    vertices = vertices.copy()
    vertices[:, 1] = lift
    return vertices, faces


def nearest(vertices: FloatArray, point: tuple[float, float, float]) -> int:
    """The vertex closest to a position, as a plain int."""
    return int(np.argmin(np.linalg.norm(vertices - np.asarray(point), axis=1)))


def gradient_arrival_s(
    vertices: FloatArray, source: int, v0: float, g: float
) -> FloatArray:
    """Exact first arrival for ``v(z) = v0 + g z``, where the rays are circular arcs.

    ``T = (1/g) arccosh(1 + g^2 d^2 / (2 v_1 v_2))`` with ``d`` the straight-line
    distance and ``v_1``, ``v_2`` the speeds at the two ends. It depends on the
    endpoints' depths only through their speeds and not on the path, which is what
    makes it usable per vertex -- and it is an analytic solution rather than a second
    reading of the solver.
    """

    def speed_at(depth_km: FloatArray) -> FloatArray:
        return v0 + g * depth_km

    distance_km = np.linalg.norm(vertices - vertices[source], axis=1)
    return (
        np.arccosh(
            1.0
            + g
            * g
            * distance_km**2
            / (2.0 * speed_at(vertices[source, 2]) * speed_at(vertices[:, 2]))
        )
        / g
    )


MESHES = {
    "lattice": lattice(25),
    "alternating lattice": lattice(25, alternate=True),
    "scattered": scattered(625),
    "warped": warped(25),
}


def uniform(faces: IntArray, slowness_s_per_km: float = 0.4) -> FloatArray:
    """One slowness everywhere."""
    return np.full(faces.shape[0], slowness_s_per_km)


def structured(vertices: FloatArray, faces: IntArray) -> FloatArray:
    """A smoothly heterogeneous slowness, varying by about a factor of two.

    Smooth, and deliberately: a slowness that jumped by 40% between adjacent faces
    would be a medium in which the analytic ball's ``S0`` is not representative of the
    ball at all, and the tests below would then be measuring how badly a nonsense
    velocity model breaks the boundary condition rather than how the solver behaves.
    A crustal gradient with a lateral wobble is the shape of the real thing.
    """
    depth_km = vertices[faces, 2].mean(axis=1)
    across_km = vertices[faces, 0].mean(axis=1)
    return 1.0 / (1.8 + 0.15 * depth_km + 0.5 * np.sin(0.4 * across_km))


def rough(faces: IntArray, seed: int = 7) -> FloatArray:
    """A slowness that jumps from face to face, for checks that do not seed a ball."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.6, size=faces.shape[0])


def one_ring(faces: IntArray, count: int) -> list[set[int]]:
    """Vertex-to-vertex adjacency as a list of sets."""
    ring = [set() for _ in range(count)]
    for a, b, c in faces:
        for first, second in ((a, b), (b, c), (c, a)):
            ring[first].add(int(second))
            ring[second].add(int(first))
    return ring


def edge_slowness(
    faces: IntArray, slowness: FloatArray
) -> dict[tuple[int, int], float]:
    """The slowest face carrying each edge, keyed by the sorted vertex pair."""
    slowest = {}
    for face, (a, b, c) in enumerate(faces):
        for first, second in ((a, b), (b, c), (c, a)):
            key = (min(int(first), int(second)), max(int(first), int(second)))
            slowest[key] = max(slowest.get(key, 0.0), float(slowness[face]))
    return slowest


# ============================================================================
# The contract, ported from crates/kernels/tests/eikonal_contract.rs
#
# Split in two, because the two halves are different kinds of statement.
#
# **Structural properties of the scheme** -- causality, the Lipschitz step, the
# sandwich between a straight line and a path along edges -- are asserted through
# `solve_from_boundary` with a single held vertex. That is meshFIM with nothing else in
# front of it, and there these are theorems about the update rule: the one-sided cap
# gives `T(c) <= T(n) + |edge| s` by construction and everything else follows.
#
# **Properties of the answer** are asserted through `solve`. The analytic ball is a
# *boundary condition*, not a solution: it asserts `T = S0 d` over a few cells, which is
# exactly right for a uniform medium and off by the seeding error otherwise. So it can
# and does hand back values a hair earlier than any path along the mesh would allow,
# and asserting the structural bounds through it would be asserting that a boundary
# condition is a consequence of the scheme, which it is not.
# ============================================================================


@pytest.mark.parametrize("name", list(MESHES))
def test_a_multi_seed_solve_is_the_min_of_its_single_seed_solves(name: str) -> None:
    """The property `MESH.md` says carries over from `eikonal.rs` unchanged.

    First arrival from several sources *is* the minimum over sources. Today the
    implementation solves each seed's ball separately and combines, so this is exact to
    the bit -- asserted that tightly on purpose, because it is the contract a future
    single-pass multi-source scheme would have to meet, and whoever writes one will
    have to loosen this equality here with the tolerance argued in front of them.
    """
    vertices, faces = MESHES[name]
    slowness = structured(vertices, faces)
    seeds = [
        fim.Seed(nearest(vertices, (2.0, 0.0, 3.0)), 0.0),
        fim.Seed(nearest(vertices, (13.0, 0.0, 12.0)), 1.25),
        fim.Seed(nearest(vertices, (8.0, 0.0, 1.0)), 0.4),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fim.DegradedSeed)
        together = fim.solve(vertices, faces, slowness, seeds)
        apart = np.min(
            [fim.solve(vertices, faces, slowness, [seed]) for seed in seeds], axis=0
        )
    assert together.tobytes() == apart.tobytes()


@pytest.mark.parametrize("name", list(MESHES))
def test_a_seed_never_ruptures_after_its_own_start_time(name: str) -> None:
    """And with one seed, it ruptures exactly then.

    The inequality is the multi-seed statement: an earlier wavefront may sweep past a
    later seed, and then that seed's ``t0`` is not the first arrival there. The
    equality for a lone seed is the pinned hypocentre `MESH.md` calls out as having no
    perturbation to hide behind -- it has to be exact, not close.
    """
    vertices, faces = MESHES[name]
    slowness = structured(vertices, faces)
    seeds = [
        fim.Seed(nearest(vertices, (3.0, 0.0, 3.0)), 0.0),
        fim.Seed(nearest(vertices, (12.0, 0.0, 12.0)), 2.0),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fim.DegradedSeed)
        times_s = fim.solve(vertices, faces, slowness, seeds)
        for seed in seeds:
            assert times_s[seed.vertex] <= seed.t0_s + 1e-12
            alone = fim.solve(vertices, faces, slowness, [seed])
        assert alone[seeds[-1].vertex] == seeds[-1].t0_s


@pytest.mark.parametrize("name", list(MESHES))
def test_every_vertex_ruptures_after_a_neighbour(name: str) -> None:
    """The correct form of "the front expands outward".

    Distance monotonicity is *false* in a heterogeneous medium: a fast channel reaches
    a far vertex before a slow one nearby. Causality is not -- a first arrival came
    from somewhere, and on a triangulation that somewhere is a vertex the update rule
    can see.

    **Over the extended one-ring, not the mesh one-ring.** Kimmel & Sethian's virtual
    edges reach past an obtuse wedge into the next triangle, and Fu et al. note they
    "are not considered part of the mesh; they are used only in the solver". So a vertex
    whose wedge was unfolded can legitimately be reached from a vertex it shares no edge
    with, and the mesh one-ring is then the wrong set to quantify over. Asserted over
    the set the update rule actually reads, which is the corner table's flanks -- on the
    lattices the two sets coincide, and on the Delaunay and warped meshes they do not.
    """
    vertices, faces = MESHES[name]
    slowness = rough(faces)
    source = nearest(vertices, (5.0, 0.0, 5.0))
    times_s = fim.solve_from_boundary(
        vertices, faces, slowness, np.array([source]), np.array([0.0])
    )
    corners = fim._corners(vertices, faces, slowness)
    for vertex in range(vertices.shape[0]):
        if vertex == source:
            continue
        wedges = slice(corners.start[vertex], corners.start[vertex + 1])
        upwind = np.concatenate([corners.left[wedges], corners.right[wedges]])
        assert (times_s[upwind] < times_s[vertex]).any(), (
            f"vertex {vertex} at {times_s[vertex]} is a local minimum; nothing reached it"
        )


@pytest.mark.parametrize("name", list(MESHES))
def test_neighbouring_vertices_are_lipschitz(name: str) -> None:
    """No edge is crossed faster than the slowest face that carries it.

    The *sharp* form, per edge, because a triangulation has no single spacing to state
    it in. Structural: the one-sided cap enforces ``T(c) <= T(n) + |edge| s`` by
    construction for every real edge, and ``s`` is at most the slowest face meeting that
    edge. A virtual edge is not a path, so the bound is quantified over real edges only.
    """
    vertices, faces = MESHES[name]
    slowness = rough(faces)
    source = nearest(vertices, (8.0, 0.0, 8.0))
    times_s = fim.solve_from_boundary(
        vertices, faces, slowness, np.array([source]), np.array([0.0])
    )
    for (first, second), worst in edge_slowness(faces, slowness).items():
        span_km = float(np.linalg.norm(vertices[first] - vertices[second]))
        step_s = abs(times_s[first] - times_s[second])
        assert step_s <= span_km * worst * (1.0 + 1e-9), (
            f"edge {first}-{second} jumps {step_s} s, past the "
            f"{span_km * worst} s crossing it at the slowest face takes"
        )


@pytest.mark.parametrize("name", list(MESHES))
def test_a_faster_medium_never_ruptures_later(name: str) -> None:
    """Two solves, and no analytic solution needed.

    It catches the error every bound is weakest against: a solver that inverts its
    input, using speed where slowness belongs, still produces a plausible field but
    reverses this.
    """
    vertices, faces = MESHES[name]
    slowness = structured(vertices, faces)
    seed = fim.Seed(nearest(vertices, (4.0, 0.0, 9.0)), 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fim.DegradedSeed)
        slow_s = fim.solve(vertices, faces, slowness, [seed])
        quick_s = fim.solve(vertices, faces, slowness * 0.8, [seed])
    assert (quick_s <= slow_s + 1e-12).all()


@pytest.mark.parametrize("name", list(MESHES))
def test_arrival_is_between_the_straight_line_and_the_mesh_path(name: str) -> None:
    """First arrival is sandwiched between a chord and an actual path along edges.

    The strongest check available on a medium with no closed-form solution, and the one
    that catches the errors worth catching: speed used where slowness belongs, a length
    in the wrong units, the seed on the wrong vertex. Above, Dijkstra over the mesh's
    own edges is *a* path through the medium, so first arrival cannot beat it -- and it
    is a genuinely different algorithm rather than a second reading of this one. Below,
    no path is shorter than the straight line and none faster than the fastest face,
    less one edge's slack for how the source vertex itself is discretised.
    """
    vertices, faces = MESHES[name]
    slowness = rough(faces)
    source = nearest(vertices, (6.0, 0.0, 11.0))
    times_s = fim.solve_from_boundary(
        vertices, faces, slowness, np.array([source]), np.array([0.0])
    )

    from scipy.sparse import coo_array
    from scipy.sparse.csgraph import dijkstra

    slowest = edge_slowness(faces, slowness)
    rows, columns, weights = [], [], []
    for (first, second), worst in slowest.items():
        span_km = float(np.linalg.norm(vertices[first] - vertices[second]))
        rows += [first, second]
        columns += [second, first]
        weights += [span_km * worst] * 2
    graph = coo_array(
        (weights, (rows, columns)), shape=(vertices.shape[0], vertices.shape[0])
    ).tocsr()
    assert (times_s <= dijkstra(graph, indices=source) * (1.0 + 1e-9)).all()

    straight_s = np.linalg.norm(vertices - vertices[source], axis=1) * float(
        slowness.min()
    )
    longest_km = max(
        float(np.linalg.norm(vertices[first] - vertices[second]))
        for first, second in slowest
    )
    slack_s = longest_km * float(slowness.max())
    assert (times_s >= straight_s - slack_s).all()


@pytest.mark.parametrize("name", ["lattice", "alternating lattice", "scattered"])
def test_a_uniform_medium_is_solved_exactly_inside_the_ball(name: str) -> None:
    """The seeded ball is the analytic solution, so inside it the error is rounding.

    Not a statement about the scheme -- outside the ball a first-order scheme leaves
    the ~1e-2 the convergence study below measures. It is a statement about the
    boundary condition: ``T = S0 d`` is the exact answer for a uniform medium, so if
    the ball is seeded correctly it agrees with the analytic solution to f64 round-off
    at every vertex it holds, and the *hypocentre itself is exact*.
    """
    vertices, faces = MESHES[name]
    slowness = uniform(faces)
    seed = fim.Seed(nearest(vertices, (8.0, 0.0, 8.0)), 3.5)
    times_s, (report,) = fim.solve_with_report(vertices, faces, slowness, [seed])

    distance_km = np.linalg.norm(vertices - vertices[seed.vertex], axis=1)
    inside = distance_km <= report.radius_km
    exact_s = 3.5 + 0.4 * distance_km
    assert np.abs(times_s[inside] - exact_s[inside]).max() < 1e-12
    assert times_s[seed.vertex] == 3.5


# ============================================================================
# The local solver, against the other published statement of it
# ============================================================================


def _loose_triangles(count: int, *, obtuse: bool, seed: int = 4) -> tuple:
    """Random triangles as a mesh of disjoint faces, plus a random flank arrival.

    Disjoint on purpose: each triangle is then its own isolated local-solve problem
    with no neighbour to be updated from, which is what makes the comparison below one
    wedge at a time.
    """
    rng = np.random.default_rng(seed)
    vertices, faces, slowness, gap_s = [], [], [], []
    while len(faces) < count:
        near_km, far_km = rng.uniform(0.2, 3.0, size=2)
        theta = rng.uniform(0.02, np.pi - 0.02)
        if (np.cos(theta) < 0.0) != obtuse:
            continue
        base = len(vertices)
        vertices += [
            [0.0, 0.0, 0.0],
            [near_km, 0.0, 0.0],
            [far_km * np.cos(theta), 0.0, far_km * np.sin(theta)],
        ]
        faces.append([base, base + 1, base + 2])
        slowness.append(rng.uniform(0.1, 1.0))
        chord_km = float(
            np.linalg.norm(np.array(vertices[-2]) - np.array(vertices[-1]))
        )
        # Lipschitz-feasible: a bigger gap than this is not a wavefront on this edge.
        gap_s.append(rng.uniform(0.0, slowness[-1] * chord_km))
    return (
        np.array(vertices),
        np.array(faces, dtype=np.int64),
        np.array(slowness),
        np.array(gap_s),
    )


def _fu_2011_equation_2_2(
    vertices: FloatArray, face: IntArray, slowness_s_per_km: float, gap_s: float
) -> tuple[bool, float]:
    """Fu et al. equation (2.2), minimised over ``lambda`` by brute force.

    ``Phi_3 = lambda Phi_12 + Phi_1 + f ||e_13 - lambda e_12||`` with ``Phi_1 = 0`` and
    ``Phi_12 = gap_s``. The paper accepts the update when the minimiser lands in
    ``[0, 1]`` -- the characteristic crosses the base edge between the two known
    vertices -- and otherwise clamps to the endpoints.

    A completely different parametrisation from the equation (4) quadratic the module
    solves, evaluated numerically rather than in closed form, so it is a *reference*
    and not a second transcription of the subject.
    """
    apex, near, far = vertices[face]
    to_near, along = near - apex, far - near
    lam = np.linspace(0.0, 1.0, 200_001)
    phi_s = lam * gap_s + slowness_s_per_km * np.linalg.norm(
        to_near[None, :] + lam[:, None] * along[None, :], axis=1
    )
    best = int(np.argmin(phi_s))
    return 0 < best < lam.size - 1, float(phi_s[best])


@pytest.mark.parametrize("obtuse", [False, True])
def test_the_local_solver_agrees_with_fu_et_als_own_on_acute_wedges(
    obtuse: bool,
) -> None:
    """Two published statements of one local solver, checked against each other.

    Kimmel & Sethian's equation (4) quadratic with its equation (5) acceptance test is
    what this module solves; Fu et al.'s equation (2.2) minimised over ``lambda`` is the
    same update written the other way round. They should be the same rule, and the
    module docstring says they are -- so it is worth knowing exactly where that stops
    being true.

    **Acute wedges: identical, on both counts.** The two accept the same wedges over
    400 random acute triangles, and where both accept they return the same arrival to
    1e-5 s, which is the resolution of the 200,001-point search the reference uses.

    **Obtuse wedges: they part company in half of cases**, always the same way -- Fu
    et al.'s stated ``lambda in [0, 1]`` accepts an interior minimum that lies *below*
    the later flank's own arrival, which would be an update taken from a vertex
    downwind of the answer. Equation (5)'s ``u < t`` rules exactly that out, and adding
    it to the reference restores agreement everywhere. The stricter rule is the one
    implemented, and this is the evidence for preferring it rather than a preference.
    """
    count = 400
    vertices, faces, slowness, gap_s = _loose_triangles(count, obtuse=obtuse)
    corners = fim._corners(vertices, faces, slowness)

    # The apex of each face is its corner 0; the flanks carry 0 and `gap_s`.
    times_s = np.full(vertices.shape[0], np.inf)
    times_s[faces[:, 1]] = 0.0
    times_s[faces[:, 2]] = gap_s
    apex_wedge = np.array(
        [int(np.flatnonzero(corners.apex == face[0])[0]) for face in faces]
    )
    solved_s = fim._candidates(corners, apex_wedge, times_s)
    # A wedge was accepted exactly when it beat its own one-sided edge fallback.
    fallback_s = np.minimum(
        slowness
        * np.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 0]], axis=1),
        gap_s
        + slowness
        * np.linalg.norm(vertices[faces[:, 2]] - vertices[faces[:, 0]], axis=1),
    )
    accepted = solved_s < fallback_s - 1e-12

    matched = agreed_value = 0
    for face in range(count):
        interior, reference_s = _fu_2011_equation_2_2(
            vertices, faces[face], float(slowness[face]), float(gap_s[face])
        )
        causal = reference_s > gap_s[face]
        if bool(accepted[face]) == (interior and causal):
            matched += 1
        if accepted[face] and interior:
            agreed_value += abs(solved_s[face] - reference_s) < 1e-5
    print(
        f"\n{'obtuse' if obtuse else 'acute'} wedges: equation (5) and equation (2.2) "
        f"plus causality agree on {matched}/{count}; where both accept, the arrivals "
        f"agree at {agreed_value}/{int(accepted.sum())}"
    )
    assert matched == count
    assert agreed_value == int(accepted.sum())


# ============================================================================
# Refusals: bad inputs are named, not solved around
# ============================================================================


def test_an_out_of_bounds_seed_is_refused_by_name() -> None:
    """`eikonal.rs`'s ``SeedOutOfBounds``, naming the seed and the mesh."""
    vertices, faces = MESHES["lattice"]
    with pytest.raises(ValueError, match=r"seed 1 at vertex 99999.*625 vertices"):
        fim.solve(vertices, faces, uniform(faces), [fim.Seed(0, 0.0), fim.Seed(99999)])


def test_a_face_no_wave_can_cross_is_refused_by_name() -> None:
    """`eikonal.rs`'s ``NonPositiveSlowness``, and for the reason it gives."""
    vertices, faces = MESHES["lattice"]
    for bad in (0.0, -0.3, np.nan, np.inf):
        slowness = uniform(faces)
        slowness[17] = bad
        with pytest.raises(ValueError, match=r"slowness on face 17.*unreachable"):
            fim.solve(vertices, faces, slowness, [fim.Seed(0)])


def test_a_seed_with_no_time_is_refused() -> None:
    """`eikonal.rs`'s ``NonFiniteSeedTime``: a NaN start poisons every cell it wins."""
    vertices, faces = MESHES["lattice"]
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match=r"seed 0 starts at t = .*not a time"):
            fim.solve(vertices, faces, uniform(faces), [fim.Seed(0, bad)])


def test_no_seeds_no_mesh_and_wrong_lengths_are_refused() -> None:
    """The remaining four of `eikonal.rs`'s variants, each by its own message."""
    vertices, faces = MESHES["lattice"]
    with pytest.raises(ValueError, match="no seeds"):
        fim.solve(vertices, faces, uniform(faces), [])
    with pytest.raises(ValueError, match="no surface to solve on"):
        fim.solve(vertices[:0], faces[:0], uniform(faces[:0]), [fim.Seed(0)])
    with pytest.raises(ValueError, match=r"has 11 values.*one per face"):
        fim.solve(vertices, faces, np.full(11, 0.4), [fim.Seed(0)])
    with pytest.raises(
        ValueError, match=r"face 0 is \[0, \d+, 99999\].*outside a mesh"
    ):
        broken = faces.copy()
        broken[0, 2] = 99999
        fim.solve(vertices, broken, uniform(faces), [fim.Seed(0)])


def test_a_triangle_with_no_thickness_is_refused() -> None:
    """A collinear corner has no gradient to read, and says so rather than returning one."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    with pytest.raises(ValueError, match="a triangle that thin is a line segment"):
        fim.solve(vertices, faces, uniform(faces), [fim.Seed(0)])


def test_a_component_with_no_seed_is_refused() -> None:
    """Two disjoint patches and one seed: the other patch says it was never reached."""
    left, left_faces = lattice(5, 4.0)
    right = left + np.array([100.0, 0.0, 0.0])
    vertices = np.concatenate([left, right])
    faces = np.concatenate([left_faces, left_faces + left.shape[0]])
    with pytest.raises(ValueError, match="never reached.*holds no seed"):
        fim.solve(vertices, faces, uniform(faces), [fim.Seed(0)])


# ============================================================================
# Gate 1 and gate 2a: the two boundary conditions, and their slopes
# ============================================================================

REFINEMENTS = (33, 65, 129, 257)
"""Vertex counts per side for the convergence study.

Fu et al. section 3.3.2 use seven meshes from 256 to 1,048,576 vertices. Four are used
here, 1,089 to 66,049 vertices, and the cap is deliberate: this solver is numpy rather
than the paper's CUDA, and the memory of the corner table grows with it. The slope is
already flat to two decimal places over these four, so the missing refinements would
add cost and not evidence.
"""

CIRCLE_KM = 3.0
"""The seeded radius for the smooth boundary condition, held fixed under refinement.

Fu et al.'s own number -- "a pair of circles of radius 3, where the domain is 16x16".
Fixed rather than mesh-derived because a boundary that shrank with ``h`` would be a
different boundary-value problem at every refinement, and the slope would measure the
sequence of problems rather than the convergence of one.
"""


def _slope(errors: list[float], spacings: list[float]) -> float:
    """Least-squares slope of log error against log spacing."""
    return float(
        np.polyfit(np.log(np.asarray(spacings)), np.log(np.asarray(errors)), 1)[0]
    )


def _refinement_errors(
    boundary: Callable[[int, FloatArray], IntArray],
) -> tuple[list[float], list[float], list[float]]:
    """L2 and Linf error against the analytic uniform solution, at each refinement."""
    extent_km, slowness_s_per_km = 16.0, 0.4
    spacings, l2, linf = [], [], []
    for cells in REFINEMENTS:
        vertices, faces = lattice(cells, extent_km)
        source = nearest(vertices, (extent_km / 2, 0.0, extent_km / 2))
        distance_km = np.linalg.norm(vertices - vertices[source], axis=1)
        held = boundary(source, distance_km)
        times_s = fim.solve_from_boundary(
            vertices,
            faces,
            uniform(faces, slowness_s_per_km),
            held,
            slowness_s_per_km * distance_km[held],
        )
        # Measured outside the seeded disc in both cases, so the two boundary
        # conditions are compared on the same set of vertices.
        outside = distance_km > CIRCLE_KM
        error_s = times_s[outside] - slowness_s_per_km * distance_km[outside]
        spacings.append(extent_km / (cells - 1))
        l2.append(float(np.sqrt(np.mean(error_s**2))))
        linf.append(float(np.abs(error_s).max()))
    return spacings, l2, linf


@pytest.mark.slow
def test_a_point_boundary_reproduces_fu_et_als_degraded_slope() -> None:
    """Gate 1: the *failure* first, because that is what shows the port is faithful.

    Fu et al. section 3.3.2: "For the point boundary conditions, the slope is
    less--showing the method is not first-order accurate for nonsmooth boundaries,
    which are inconsistent with the governing equations." They give no number for it;
    their figure 3.3(b) digitises to about 0.68.

    Measured here, on a 16x16 square at 0.4 s/km with a single seeded vertex:

    ======  ===========  ===========
    h (km)  L2 (s)       Linf (s)
    ======  ===========  ===========
    0.500   8.9948e-02   1.9150e-01
    0.250   5.5491e-02   1.1711e-01
    0.125   3.3420e-02   6.9799e-02
    0.0625  1.9696e-02   4.0715e-02
    ======  ===========  ===========

    Fitted slope 0.731 in L2 and 0.744 in Linf -- inside a twentieth of the digitised
    0.68, and unmistakably short of 1. A spike that passed a first-order test with a
    point source would be a spike that is not doing what the paper does.
    """
    spacings, l2, linf = _refinement_errors(
        lambda source, distance_km: np.array([source])
    )
    slope_l2, slope_linf = _slope(l2, spacings), _slope(linf, spacings)
    print(f"\npoint boundary: slope L2 {slope_l2:.3f}, Linf {slope_linf:.3f}")
    print(f"  h    {[f'{value:.4f}' for value in spacings]}")
    print(f"  L2   {[f'{value:.4e}' for value in l2]}")
    print(f"  Linf {[f'{value:.4e}' for value in linf]}")
    assert 0.5 < slope_l2 < 0.9, f"L2 slope {slope_l2}: not the reported degradation"
    assert 0.5 < slope_linf < 0.9


@pytest.mark.slow
def test_a_circular_boundary_recovers_first_order() -> None:
    """Gate 2a: Fu et al.'s slope 1.0, which is the whole argument for the ball.

    Their sentence: "For the circular boundary conditions, the slope of this graph is
    1.0, which is consistent to our claim that meshFIM is first-order accurate."

    Measured here, same square and same slowness, seeded on a disc of fixed radius 3:

    ======  ===========  ===========
    h (km)  L2 (s)       Linf (s)
    ======  ===========  ===========
    0.500   3.4982e-02   9.4360e-02
    0.250   1.7272e-02   4.7965e-02
    0.125   8.3878e-03   2.3705e-02
    0.0625  4.1418e-03   1.1794e-02
    ======  ===========  ===========

    Fitted slope 1.026 in L2 and 1.000 in Linf. Same solver, same mesh sequence, same
    error window as the point-boundary test above; only the boundary changed, and the
    slope went from 0.73 to 1.0. That is the claim `MESH.md` makes, measured.
    """
    spacings, l2, linf = _refinement_errors(
        lambda source, distance_km: np.flatnonzero(distance_km <= CIRCLE_KM)
    )
    slope_l2, slope_linf = _slope(l2, spacings), _slope(linf, spacings)
    print(f"\ncircular boundary: slope L2 {slope_l2:.3f}, Linf {slope_linf:.3f}")
    print(f"  h    {[f'{value:.4f}' for value in spacings]}")
    print(f"  L2   {[f'{value:.4e}' for value in l2]}")
    print(f"  Linf {[f'{value:.4e}' for value in linf]}")
    assert 0.9 < slope_l2 < 1.15, f"L2 slope {slope_l2}: not first order"
    assert 0.9 < slope_linf < 1.15


# ============================================================================
# Gate 2b: reduction to the Cartesian solver, as a systematic
# ============================================================================

PERTURBATION_S = 0.35
"""The onset displacement the model applies deliberately.

``rupture_time_scale = -0.35`` s in `examples/crustal.toml`. `MESH.md` argues this and
not `ENGINEERING_RULES.md`'s 0.05 s is the yardstick for a method change that is
expected to move numbers, because the 0.05 s bound answers "does this implementation
agree with that one" rather than "is this distinguishable".
"""


@pytest.mark.parametrize("cells", [33, 65])
def test_the_wavefront_agrees_with_the_cartesian_solver(cells: int) -> None:
    """Gate 2b: the systematic against source distance, not the RMS.

    A first-order bias does not average out across realisations and it grows with
    distance from the source, so an RMS would report it as smaller than it is. This
    bins the difference by distance and prints the mean in each bin.

    On a 16 km square of uniform 0.4 s/km, where `_kernels.eikonal_solve` is exact to
    1e-13 and so stands in for the analytic answer:

    ======  =============  ============  =============
    h (km)  bias at 2 km   bias at 8 km  worst vertex
    ======  =============  ============  =============
    0.500   +3.4 ms        +37 ms        +116 ms
    0.250   +1.3 ms        +28 ms        +80 ms
    ======  =============  ============  =============

    The bias is positive at every distance -- meshFIM is *late*, which is what an
    upwind scheme confined to mesh edges must be -- and it grows monotonically with
    distance, from a few milliseconds near the ball to about 5% of the deliberate
    perturbation at the far corner. The worst vertex at 1 km-class spacing is 0.116 s:
    twice `ENGINEERING_RULES.md`'s 0.05 s and a third of the 0.35 s the model displaces
    every onset by on purpose. `MESH.md` predicts exactly that trade and says the
    0.05 s bound is the wrong test to apply to it.
    """
    extent_km, slowness_s_per_km = 16.0, 0.4
    vertices, faces = lattice(cells, extent_km)
    spacing_km = extent_km / (cells - 1)
    source = (cells // 2, cells // 2)
    seed = fim.Seed(source[0] * cells + source[1], 0.0)

    triangular_s, (report,) = fim.solve_with_report(
        vertices, faces, uniform(faces, slowness_s_per_km), [seed]
    )
    cartesian_s = _kernels.eikonal_solve(
        np.full((cells, cells), slowness_s_per_km),
        (spacing_km, spacing_km),
        [(*source, 0.0)],
    ).reshape(-1)

    distance_km = np.linalg.norm(vertices - vertices[seed.vertex], axis=1)
    exact_s = slowness_s_per_km * distance_km
    assert np.abs(cartesian_s - exact_s).max() < 1e-12, "the reference is not exact"

    bias_s = triangular_s - cartesian_s
    print(f"\nh = {spacing_km:.4f} km, r0 = {report.radius_km:.3f} km")
    edges = np.linspace(0.0, distance_km.max(), 9)
    for lower, upper in itertools.pairwise(edges):
        band = (distance_km >= lower) & (distance_km < upper)
        if band.sum() < 5:
            continue
        print(
            f"  r in [{lower:5.2f}, {upper:5.2f}) n={int(band.sum()):5d} "
            f"mean {bias_s[band].mean() * 1e3:+8.3f} ms  "
            f"worst {np.abs(bias_s[band]).max() * 1e3:8.3f} ms"
        )
    print(f"  worst overall {np.abs(bias_s).max() * 1e3:.3f} ms")

    # The pinned hypocentre, which carries no perturbation and is the registration
    # point every diagnostic is measured from, is exact rather than close.
    assert triangular_s[seed.vertex] == 0.0
    assert cartesian_s[seed.vertex] == 0.0
    # The gate `MESH.md` states: well under the deliberate perturbation.
    assert np.abs(bias_s).max() < PERTURBATION_S / 3.0
    # And late everywhere, which is the sign an upwind scheme on mesh edges must have.
    assert bias_s.min() > -1e-9
    # Monotone in distance, band by band: the bias is a systematic and not noise.
    means = [
        bias_s[(distance_km >= lower) & (distance_km < upper)].mean()
        for lower, upper in itertools.pairwise(edges)
        if ((distance_km >= lower) & (distance_km < upper)).sum() >= 5
    ]
    assert all(later >= earlier for earlier, later in itertools.pairwise(means))


# ============================================================================
# Gate 3: the r0 sweep -- a bound, not a choice
# ============================================================================


@pytest.mark.slow
@pytest.mark.parametrize(
    ("v0", "gradient", "spacing_km"),
    [(1.8, 0.15, 1.0), (1.8, 0.15, 0.25), (3.0, 0.06, 1.0)],
)
def test_the_seeded_radius_is_a_bound_and_not_a_choice(
    v0: float, gradient: float, spacing_km: float
) -> None:
    """Gate 3: onset error against ``r0``, and against the slowness across the ball.

    Solved on a constant-gradient medium, where the arrival is analytic, so both halves
    of the trade can be measured separately: the seeding error at the ring (which grows
    with ``r0``, because the ball asserts one slowness) and the discretisation bias in
    the far field (which shrinks with ``r0``, because more of the domain is analytic).

    What the sweep shows, and it is the answer to "is ``r0`` a free parameter":

    - **The two errors move in opposite directions**, so there is an interior minimum
      rather than a monotone preference. On ``v = 1.8 + 0.15 z`` at 1 km spacing the
      far-field bias falls from +70 ms to -132 ms across the sweep, while the seeding
      error climbs from 8 ms to 1700 ms. Neither alone would pin ``r0``; together they
      do.
    - **The minimum is flat, and :data:`fim.SEED_RING_DEPTH` sits in the middle of it.**
      Worst-vertex error, in milliseconds, by ring depth:

      ==================  ====  ====  ====  ====  ====  ====
      medium              1     2     3     4     6     8
      ==================  ====  ====  ====  ====  ====  ====
      1.8+0.15z, h=1      207   180   191   216   506   1050
      1.8+0.15z, h=0.25    83    73    66    63    65     73
      3.0+0.06z, h=1      196   167   152   147   166    303
      ==================  ====  ====  ====  ====  ====  ====

      Across **2 to 4 rings** the spread is 1.20x, 1.16x and 1.13x -- flat, and three is
      the middle of it. Outside that window ``r0`` starts to matter in both directions:
      at one ring the seeded boundary is a hexagon rather than a circle, and by six the
      seeding error has taken over (506 ms against 191 ms at three, on the steep
      gradient). That is what "derived and reported rather than configured" buys -- not
      that the choice is arbitrary, but that both sides of it are visible.
    - **The reported bound is a bound.** ``r0 max|S - S0|``, what
      :class:`fim.SeedReport` carries, exceeds the *measured* seeding error at every
      radius of every medium swept, by a factor between 2.1 and 3.6. Conservative by
      about three and never optimistic, which is what a bound has to be.
    """
    extent_km = 24.0
    cells = round(extent_km / spacing_km) + 1
    vertices, faces = lattice(cells, extent_km)
    slowness = 1.0 / (v0 + gradient * vertices[faces, 2].mean(axis=1))
    source = nearest(vertices, (extent_km / 2, 0.0, extent_km / 2))
    truth_s = gradient_arrival_s(vertices, source, v0, gradient)
    distance_km = np.linalg.norm(vertices - vertices[source], axis=1)
    source_slowness = float(slowness[(faces == source).any(axis=1)].mean())
    far = distance_km > extent_km / 3.0

    print(f"\nv = {v0} + {gradient} z km/s, h = {spacing_km} km")
    conservatism, worst = [], []
    for rings in (1, 2, 3, 4, 6, 8, 12):
        radius_km = rings * spacing_km * np.sqrt(2.0)
        held = np.flatnonzero(distance_km <= radius_km)
        if held.size < 3:
            continue
        seeded_error_s = float(
            np.abs(source_slowness * distance_km[held] - truth_s[held]).max()
        )
        covered = np.isin(faces, held).any(axis=1)
        bound_s = radius_km * float(np.abs(slowness[covered] - source_slowness).max())
        times_s = fim.solve_from_boundary(
            vertices, faces, slowness, held, source_slowness * distance_km[held]
        )
        error_s = times_s - truth_s
        print(
            f"  r0 = {radius_km:5.2f} km ({rings:2d} rings, {held.size:5d} seeded)  "
            f"reported bound {bound_s * 1e3:8.2f} ms  measured seed error "
            f"{seeded_error_s * 1e3:8.2f} ms  far bias {error_s[far].mean() * 1e3:+8.2f} "
            f"ms  worst {np.abs(error_s).max() * 1e3:8.2f} ms"
        )
        if seeded_error_s > 0.0:
            conservatism.append(bound_s / seeded_error_s)
        if abs(rings - fim.SEED_RING_DEPTH) <= 1:
            worst.append(float(np.abs(error_s).max()))

    assert min(conservatism) > 1.0, (
        f"the reported bound is optimistic by {1 / min(conservatism):.2f}x somewhere; "
        "a bound that can be beaten is an estimate"
    )
    assert max(conservatism) < 5.0, "the bound is so loose it says nothing"
    # Flat within one ring either side of the depth the module derives. Wider than that
    # it is not flat, which is the point: the window is real and has edges.
    assert max(worst) / min(worst) < 1.25, (
        f"worst-vertex error varies by {max(worst) / min(worst):.2f}x across the ring "
        "depths either side of SEED_RING_DEPTH, so the derived radius sits on a slope "
        "rather than in the flat window and has to be argued rather than derived"
    )


def test_the_seeded_ball_reports_both_of_its_bounds() -> None:
    """The two bounds are on the report, in the units each is stated in.

    `MESH.md`'s Risks section: ``r0`` is derived rather than configured precisely so
    that it is not a free parameter, and a derived quantity nobody measures is a
    configured one that stopped being written down. This asserts the measurement
    exists and is the right shape, not what it happens to equal.
    """
    vertices, faces = lattice(41, 16.0)
    slowness = 1.0 / (1.8 + 0.15 * vertices[faces, 2].mean(axis=1))
    seed = fim.Seed(nearest(vertices, (8.0, 0.0, 8.0)), 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fim.DegradedSeed)
        _, (report,) = fim.solve_with_report(vertices, faces, slowness, [seed])

    spacing_km = 16.0 / 40.0
    # The lower bound: three rings of a lattice whose rings reach along the diagonal.
    assert report.radius_km == pytest.approx(
        fim.SEED_RING_DEPTH * spacing_km * np.sqrt(2.0), rel=1e-9
    )
    assert report.seeded_vertices == 61
    assert 0.0 < report.boundary_radius_spread < 0.5
    # The upper bound, in seconds, and consistent with its own two factors.
    assert report.slowness_error_s == pytest.approx(
        report.radius_km * report.slowness_spread * report.source_slowness_s_per_km
    )
    assert report.unsplit_obtuse_wedges == 0
    assert report.sweeps > 0


def test_a_ball_the_velocity_model_does_not_justify_says_so() -> None:
    """The upper bound warns rather than refuses, and the message carries the numbers.

    A hypocentre a kilometre from a layer boundary is an ordinary earthquake. Refusing
    would answer a different question; saying nothing would make ``r0`` a parameter
    with a hidden assumption attached. `sampling.py`'s ``DegradedCorrelation`` is the
    register: the model you asked for is not the one you get, here is what you got.
    """
    vertices, faces = lattice(25, 24.0)
    # A layer boundary through the middle of the patch, at a 2x slowness contrast.
    slowness = np.where(vertices[faces, 2].mean(axis=1) < 12.0, 0.5, 0.25)
    seed = fim.Seed(nearest(vertices, (12.0, 0.0, 12.0)), 0.0)
    with pytest.warns(fim.DegradedSeed, match="past the .* budget"):
        _, (report,) = fim.solve_with_report(vertices, faces, slowness, [seed])
    assert report.slowness_error_s > fim.SEED_SLOWNESS_BUDGET_S


# ============================================================================
# Obtuse triangles and the virtual-edge unfolding
# ============================================================================


def fan(outer_km: float = 2.2) -> tuple[FloatArray, IntArray]:
    """A vertex every one of whose wedges is obtuse -- the case unfolding exists for.

    Three triangles around a centre vertex, so each wedge is 120 degrees, plus three
    outer triangles to unfold into. It is Kimmel & Sethian's figure 6 as a mesh.

    This shape is the *only* way to make the unfolding bind, and that is worth knowing
    rather than working around: at an ordinary vertex the six wedges tile the circle,
    so the directions an obtuse wedge cannot serve are served by its acute neighbours
    and the per-vertex minimum picks those instead. See
    :func:`test_the_unfolding_rarely_binds_on_the_meshes_this_codebase_builds`.
    """
    inner = np.radians([90.0, 210.0, 330.0])
    outer = np.radians([150.0, 270.0, 30.0])
    vertices = np.array(
        [[0.0, 0.0, 0.0]]
        + [[np.cos(angle), 0.0, np.sin(angle)] for angle in inner]
        + [[outer_km * np.cos(angle), 0.0, outer_km * np.sin(angle)] for angle in outer]
    )
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 4], [2, 3, 5], [3, 1, 6]],
        dtype=np.int64,
    )
    return vertices, faces


def test_an_obtuse_wedge_is_split_into_acute_ones() -> None:
    """Kimmel & Sethian section 4.2, as a statement about the geometry it produces.

    Three 120-degree wedges go in; three of them plus six 60-degree ones come out, and
    the virtual edge reaches the outer vertex at its true distance. The originals stay
    because the one-sided edge update they still offer is a valid upper bound on a real
    path, and an upper bound can only lose the per-vertex minimum.
    """
    vertices, faces = fan()
    corners = fim._corners(vertices, faces, uniform(faces))
    at_centre = corners.apex == 0
    assert int(at_centre.sum()) == 9, "three real wedges plus six virtual ones"
    cosines = np.sort(corners.cosine[at_centre])
    assert cosines[:3] == pytest.approx(-0.5), "the three 120-degree originals"
    assert cosines[3:] == pytest.approx(0.5), "six 60-degree wedges, none obtuse"
    assert corners.unsplit_obtuse == 0
    # The virtual edge's length is the real distance to the outer vertex, because on a
    # flat patch unfolding is the identity -- which is the check that the two-circle
    # construction places it where the geometry says rather than merely somewhere.
    virtual = at_centre & np.isin(corners.left, [4, 5, 6])
    assert corners.left_km[virtual] == pytest.approx(2.2)


def test_unfolding_makes_an_all_obtuse_vertex_exact_for_a_plane_wave() -> None:
    """And what it is worth, in seconds, on the case it exists for.

    A plane wave is linear, and the local solver's reconstruction is linear, so an
    accepted update reproduces it *exactly*. With three 120-degree wedges the update is
    rejected for most arrival directions and falls back to the edge form; with the
    wedges split it is accepted for every direction. Swept over 24 directions on a
    uniform medium, where the answer is known in closed form:

    - without unfolding, the centre is late by up to 13.6 ms;
    - with it, exactly right -- 0 s to f64 round-off, at every direction.

    That is the ``O(e_max / (pi - theta_max))`` degradation Kimmel & Sethian predict,
    turned off.
    """
    vertices, faces = fan()
    slowness_s_per_km = 0.4
    slowness = uniform(faces, slowness_s_per_km)
    held = np.arange(1, 7)

    worst_s = {}
    for limit in (0, fim.UNFOLD_LIMIT):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fim, "UNFOLD_LIMIT", limit)
            worst_s[limit] = max(
                abs(
                    fim.solve_from_boundary(
                        vertices, faces, slowness, held, plane_s[held]
                    )[0]
                    - plane_s[0]
                )
                for plane_s in _plane_waves(vertices, slowness_s_per_km)
            )
    print(
        f"\nall-obtuse vertex: without unfolding the centre is late by up to "
        f"{worst_s[0] * 1e3:.2f} ms; with it, {worst_s[fim.UNFOLD_LIMIT] * 1e3:.2e} ms"
    )
    assert worst_s[0] > 1e-3, "this vertex was supposed to be badly served without it"
    assert worst_s[fim.UNFOLD_LIMIT] < 1e-12


def _plane_waves(
    vertices: FloatArray, slowness_s_per_km: float, directions: int = 24
) -> Iterator[FloatArray]:
    """Exact arrivals for plane waves sweeping every direction in the patch's plane."""
    for degrees in np.linspace(0.0, 360.0, directions, endpoint=False):
        heading = np.array(
            [np.cos(np.radians(degrees)), 0.0, np.sin(np.radians(degrees))]
        )
        exact_s = slowness_s_per_km * (vertices @ heading)
        yield exact_s - exact_s.min() + 1.0


def sheared(
    cells: int, extent_km: float = 16.0, shear: float = 2.0, squash: float = 0.2
) -> tuple[FloatArray, IntArray]:
    """A lattice squashed and sheared until every face has an obtuse corner."""
    axis = np.linspace(0.0, extent_km, cells)
    down, across = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack(
        [
            (across + shear * down * squash).ravel(),
            np.zeros(cells * cells),
            (down * squash).ravel(),
        ],
        axis=1,
    )
    index = np.arange(cells * cells).reshape(cells, cells)
    top_left = index[:-1, :-1].ravel()
    bottom_left = index[1:, :-1].ravel()
    bottom_right = index[1:, 1:].ravel()
    top_right = index[:-1, 1:].ravel()
    return vertices, np.concatenate(
        [
            np.stack([top_left, bottom_left, bottom_right], axis=1),
            np.stack([top_left, bottom_right, top_right], axis=1),
        ]
    )


@pytest.mark.parametrize("name", ["sheared", "scattered"])
def test_unfolding_can_only_lower_an_arrival_never_raise_it(name: str) -> None:
    """The invariant, and how much section 4.2 is worth away from the degenerate case.

    The invariant first: the split wedges are *added* to the corner table rather than
    replacing the obtuse originals, so the per-vertex minimum can only fall. The
    originals are kept because the one-sided edge update they still offer is a valid
    upper bound on a real path along a real edge, and an upper bound never wins a
    minimum against a good answer -- which is how this gets `eikonal.rs`'s "only when no
    triangle produced a causal root" without an ordering.

    `MESH.md` argues obtuse elements should be "rare and mild" because the triangulation
    is chosen -- Delaunay in the parameter domain, which maximises the minimum angle.
    Measured, obtuse corners are not rare, and how much they cost depends entirely on
    how regular the mesh is:

    ==========  ==============  ======  =======  ===============  ================
    mesh        obtuse corners  wedges  unsplit  vertices changed  mean improvement
    ==========  ==============  ======  =======  ===============  ================
    sheared     800 of 2400     2895    60       15 of 441        0.0009 ms
    scattered   261 of 3456     522      0       549 of 625       7.4 ms
    ==========  ==============  ======  =======  ===============  ================

    On the *structured* sheared lattice it is worth nothing: the wedges tile the circle
    at every vertex, so the arrival directions an obtuse wedge cannot serve fall inside
    its acute neighbours and the minimum picks those. On the *unstructured* Delaunay
    mesh -- the one Component 1 actually builds -- it takes 7.6% off the mean relative
    error, 0.0195 to 0.0181, and up to 27 ms off a single vertex. Modest, real, and
    much smaller than the degenerate case :func:`fan` builds, where it is the whole
    answer.
    """
    vertices, faces = {"sheared": sheared(21), "scattered": scattered(625)}[name]
    slowness = uniform(faces)
    exact_s = 0.4 * np.linalg.norm(vertices - vertices[0], axis=1)

    answers = {}
    for limit in (0, fim.UNFOLD_LIMIT):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fim, "UNFOLD_LIMIT", limit)
            corners = fim._corners(vertices, faces, slowness)
            answers[limit] = (
                fim.solve_from_boundary(
                    vertices, faces, slowness, np.array([0]), np.array([0.0])
                ),
                corners.apex.size - 3 * faces.shape[0],
                corners.unsplit_obtuse,
            )

    without, _, obtuse = answers[0]
    with_it, virtual, unsplit = answers[fim.UNFOLD_LIMIT]
    changed = int((np.abs(with_it - without) > 1e-12).sum())
    relative = (np.abs(with_it - exact_s) / np.maximum(exact_s, 1e-9))[1:]
    improvement_s = without - with_it
    print(
        f"\n{name}: {obtuse} obtuse corners of {3 * faces.shape[0]}, {virtual} virtual "
        f"wedges, {unsplit} unsplit; the answer changes at {changed} of "
        f"{vertices.shape[0]} vertices by up to {improvement_s.max() * 1e3:.4f} ms "
        f"(mean {improvement_s.mean() * 1e3:.4f} ms); worst relative error "
        f"{relative.max():.4f}"
    )
    assert obtuse > 0, "this mesh was supposed to carry obtuse corners"
    assert virtual >= 2 * (obtuse - unsplit), "each split wedge should yield two"
    # The invariant, and the only thing here worth pinning.
    assert (improvement_s >= -1e-12).all(), "unfolding made an arrival later"


def test_a_boundary_edge_leaves_its_wedge_unsplit_and_counts_it() -> None:
    """There is nothing to unfold into at the edge of the mesh, and that is reported.

    A wedge that keeps the one-sided edge update is a silently degraded stencil unless
    somebody counts it, which is what :attr:`fim.SeedReport.unsplit_obtuse_wedges` is
    for. A single obtuse triangle has no neighbours at all, so every obtuse wedge of it
    is unsplit -- the smallest case where the count must be non-zero.
    """
    vertices = np.array(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 0.0, 0.25], [2.0, 0.0, -0.25]]
    )
    faces = np.array([[0, 1, 2]])
    corners = fim._corners(vertices, faces, uniform(faces))
    assert (corners.cosine < 0.0).any(), "vertex 2 was supposed to be obtuse"
    assert corners.unsplit_obtuse >= 1
    del vertices, faces


# ============================================================================
# Vertex arrivals to face arrivals
# ============================================================================


def test_face_arrivals_are_the_centroid_of_a_piecewise_linear_field() -> None:
    """The mean of the corners *is* the interpolated value at the centroid, exactly.

    Not an approximation to be checked to a tolerance: the solution is piecewise linear
    over each triangle by construction, and the centroid's barycentric coordinates are
    ``(1/3, 1/3, 1/3)``. Checked against a linear field, where the identity is exact
    and any other reduction fails it.
    """
    vertices, faces = MESHES["scattered"]
    linear = 0.3 * vertices[:, 0] - 0.7 * vertices[:, 2] + 1.5
    centroids = vertices[faces].mean(axis=1)
    at_centroid = 0.3 * centroids[:, 0] - 0.7 * centroids[:, 2] + 1.5
    assert fim.face_arrivals(faces, linear) == pytest.approx(at_centroid, abs=1e-12)


def test_face_arrivals_would_be_biased_early_by_the_minimum() -> None:
    """Why the mean and not the minimum, in the units the choice costs.

    A uniform medium on a regular lattice: taking the earliest corner rather than the
    centroid shifts *every* face early, by an amount set by the spacing rather than by
    the earthquake. That is a uniform bias in the one quantity `MESH.md` says has no
    perturbation to hide behind, and it is the argument recorded in
    :func:`fim.face_arrivals`'s docstring, measured.
    """
    vertices, faces = lattice(33, 16.0)
    times_s = fim.solve(vertices, faces, uniform(faces), [fim.Seed(0, 0.0)])
    by_mean = fim.face_arrivals(faces, times_s)
    by_minimum = times_s[faces].min(axis=1)
    shift_s = float((by_mean - by_minimum).mean())
    print(f"\nminimum-corner reduction is early by {shift_s * 1e3:.1f} ms on average")
    assert shift_s > 0.05, "the two reductions should differ by a measurable bias"
    assert (by_mean >= by_minimum - 1e-12).all()


def test_face_arrivals_refuses_a_field_of_the_wrong_length() -> None:
    """A field indexed by the wrong thing is named rather than gathered out of range."""
    _, faces = MESHES["lattice"]
    with pytest.raises(ValueError, match="outside a field of 3 vertex arrivals"):
        fim.face_arrivals(faces, np.zeros(3))


# ============================================================================
# The sweep count, and the ceiling on it
# ============================================================================


@pytest.mark.parametrize("name", list(MESHES))
def test_the_sweep_count_stays_well_inside_its_ceiling(name: str) -> None:
    """:data:`fim.MAX_SWEEP_FACTOR` is a ceiling on a bounded quantity, not a knob.

    One pass of the active list advances the front by at least one ring, so the
    seeded set's own eccentricity is a lower bound on the passes; re-entry is what
    could push it higher, and re-entry happens only where a faster path overtakes an
    earlier one. Measured on the four meshes here: 1.05 on the two lattices, 1.29 on
    the Delaunay mesh and 1.60 on the alternating lattice, whose criss-cross diagonals
    give the front two ways round every quad and so the most re-entry. A factor of four
    is two and a half times the worst headroom actually used, and the loop exits on an
    empty list rather than on the count.
    """
    vertices, faces = MESHES[name]
    seed = fim.Seed(nearest(vertices, (8.0, 0.0, 8.0)), 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", fim.DegradedSeed)
        _, (report,) = fim.solve_with_report(
            vertices, faces, structured(vertices, faces), [seed]
        )

    start, index = fim._adjacency(faces, vertices.shape[0])
    distance_km = np.linalg.norm(vertices - vertices[seed.vertex], axis=1)
    held = np.flatnonzero(distance_km <= report.radius_km)
    rings = int(fim._hops(start, index, held, vertices.shape[0]).max())
    ratio = report.sweeps / (rings + 1)
    print(f"\n{name}: {report.sweeps} sweeps over {rings} rings, ratio {ratio:.2f}")
    assert ratio < fim.MAX_SWEEP_FACTOR / 2.0, (
        f"{report.sweeps} sweeps over {rings} rings uses more than half the ceiling"
    )


# ============================================================================
# Gate 4: the jump cell, on the shipped geometries
# ============================================================================


@pytest.fixture
def structured_meshes() -> Callable[[str], list]:
    """The shipped geometries, built as `RuptureMesh` charts by the structured path.

    This gate is the one place the triangular track reaches into the working pipeline,
    and it reaches read-only: `propagation.causal_jump` takes two charts and a wavefront
    array, so handing it a wavefront this module solved needs no pipeline change at all.
    """
    from rupture_generator import pipeline
    from rupture_generator.config import read_geometry

    def load(name: str) -> list:
        geometry = read_geometry(EXAMPLES / f"{name}.geometry.toml")
        return list(pipeline.segments_of(geometry).segments.items())

    return load


CRUST_V0_KM_S = 1.8
CRUST_GRADIENT_PER_S = 0.15
"""A smooth crustal velocity gradient, in place of a layered model.

Smooth on purpose. A layered model would make the two solvers see different media --
the Cartesian one takes slowness per cell and this one per face, and across a layer
jump the face mean is a *different* medium rather than a discretisation of the same
one. With ``v = v0 + g z`` the two agree to ``O(h^2)`` and the comparison measures the
solvers rather than the reduction between them. ``1.8 + 0.15 z`` km/s spans
`examples/crustal.toml`'s 1.8 to 3.5 km/s over the seismogenic depth range.
"""


def _lattice_of_centres(cell_counts: tuple[int, int]) -> IntArray:
    """Triangulate a structured chart's cell centres, two triangles per quad.

    The vertices *are* the cells, so the triangular solve lands index for index on the
    array `propagation.causal_jump` expects and the two wavefronts can be handed to it
    unchanged. It is the sharpest comparison reachable without touching the pipeline:
    same geometry, same seed cell, same medium, same argmin.
    """
    rows, columns = cell_counts
    index = np.arange(rows * columns).reshape(rows, columns)
    top_left = index[:-1, :-1].ravel()
    bottom_left = index[1:, :-1].ravel()
    bottom_right = index[1:, 1:].ravel()
    top_right = index[:-1, 1:].ravel()
    return np.concatenate(
        [
            np.stack([top_left, bottom_left, bottom_right], axis=1),
            np.stack([top_left, bottom_right, top_right], axis=1),
        ]
    )


@pytest.mark.slow
@pytest.mark.parametrize("geometry", ["kaikoura", "beavan"])
def test_the_jump_cell_is_stable_between_solvers(
    structured_meshes: Callable[[str], list], geometry: str
) -> None:
    """Gate 4: what `propagation.causal_jump`'s argmin does when the wavefront changes.

    The argmin runs over the *raw* wavefront rather than the perturbed onset --
    deliberately, because an argmin over a hundred thousand perturbed values is an
    order statistic that finds the perturbation's negative tail. So the wavefront's
    shape selects where a multi-segment rupture crosses, and a systematic first-order
    bias moves that selection. `MESH.md` calls this one of the two places the model's
    own 0.35 s noise provides no cover.

    Every ordered pair of segments is tested, not just the causality tree's, because
    each pair is an independent argmin and the tree exercises one of them.

    **Measured, and the honest answer is mixed.**

    - **The arrival time is stable.** Worst difference 0.020 s across beavan's 38
      pairs and 0.154 s across kaikoura's 2 -- both under the 0.35 s the model
      displaces onsets by on purpose, and beavan's is under the 0.05 s onset bound.
    - **The cell is not.** 25 of 38 pairs agree exactly on beavan; the other 13 move
      the departure cell by 0.100 to 0.501 km. On kaikoura one of two moves by one
      cell, 0.992 km.

    Every disagreement is a near-tie: the wavefront gap between the two candidate
    departure cells is 1 to 40 ms on beavan, against cells 0.1 km apart. The objective
    is flat along the edge the front arrests on, so it is being resolved differently
    rather than being got wrong -- and that is a property of the geometry as much as of
    the solver. The assertion here is therefore on the arrival time and on the physical
    distance the cell moved, which are the quantities that mean something downstream;
    exact cell equality is printed, and would be the wrong thing to pin.
    """
    from rupture_generator import propagation

    segments = structured_meshes(geometry)
    solved = {}
    for name, chart in segments:
        rows, columns = chart.cell_counts
        centres_km = chart.centres().reshape(-1, 3)
        faces = _lattice_of_centres(chart.cell_counts)
        per_cell = 1.0 / (CRUST_V0_KM_S + CRUST_GRADIENT_PER_S * centres_km[:, 2])
        per_face = 1.0 / (
            CRUST_V0_KM_S + CRUST_GRADIENT_PER_S * centres_km[faces, 2].mean(axis=1)
        )
        source = (rows // 2, columns // 2)
        cartesian_s = _kernels.eikonal_solve(
            per_cell.reshape(rows, columns),
            chart.spacing_km()[::-1],
            [(*source, 0.0)],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", fim.DegradedSeed)
            triangular_s, (report,) = fim.solve_with_report(
                centres_km, faces, per_face, [fim.Seed(source[0] * columns + source[1])]
            )
        print(
            f"\n{name}: {rows}x{columns}, r0 = {report.radius_km:.3f} km, "
            f"reported seed bound {report.slowness_error_s * 1e3:.2f} ms, "
            f"{report.sweeps} sweeps, {report.unsplit_obtuse_wedges} unsplit wedges, "
            f"worst |fim - cartesian| = "
            f"{np.abs(triangular_s - cartesian_s.reshape(-1)).max() * 1e3:.1f} ms"
        )
        solved[name] = (chart, cartesian_s, triangular_s.reshape(rows, columns))

    delay = propagation.DistanceOverVelocity(np.array([1000.0]), np.array([3.5]))
    agreed = moved = 0
    worst_shift_s = worst_move_km = 0.0
    for parent, child in itertools.permutations([name for name, _ in segments], 2):
        chart, cartesian_s, triangular_s = solved[parent]
        try:
            from_cartesian = propagation.causal_jump(
                chart, cartesian_s, solved[child][0], delay
            )
            from_triangular = propagation.causal_jump(
                chart, triangular_s, solved[child][0], delay
            )
        except ValueError:
            continue  # further apart than a rupture jumps: not a pair at all
        shift_s = abs(from_triangular.arrival_s - from_cartesian.arrival_s)
        worst_shift_s = max(worst_shift_s, shift_s)
        if from_cartesian.parent_cell == from_triangular.parent_cell:
            agreed += 1
            continue
        moved += 1
        origin_km = np.array([*chart.origin_km, 0.0])
        centres_km = chart.centres() + origin_km
        move_km = float(
            np.linalg.norm(
                centres_km[from_cartesian.parent_cell]
                - centres_km[from_triangular.parent_cell]
            )
        )
        worst_move_km = max(worst_move_km, move_km)
        gap_s = abs(
            triangular_s[from_cartesian.parent_cell]
            - triangular_s[from_triangular.parent_cell]
        )
        print(
            f"  {parent[:18]:18s} -> {child[:18]:18s}  cartesian "
            f"{from_cartesian.parent_cell} vs triangular "
            f"{from_triangular.parent_cell}: {move_km:.3f} km apart, wavefront gap "
            f"between them {gap_s * 1e3:.2f} ms, arrival moved {shift_s * 1e3:+.2f} ms"
        )

    print(
        f"\n{geometry}: {agreed} pairs agree on the departure cell, {moved} move it; "
        f"worst move {worst_move_km:.3f} km, worst arrival shift "
        f"{worst_shift_s * 1e3:.1f} ms"
    )
    assert agreed + moved > 0, "no pair was close enough to jump between"
    # The gate `MESH.md` states, on the quantity that reaches the SRF.
    assert worst_shift_s < PERTURBATION_S / 2.0
    # And the departure cell, when it moves, moves within one subfault-ish rather than
    # to a different part of the fault.
    spacing_km = max(max(chart.spacing_km()) for _, chart in segments)
    assert worst_move_km <= 6.0 * spacing_km, (
        f"the departure cell moved {worst_move_km:.3f} km, which is not a tie being "
        "broken differently"
    )
