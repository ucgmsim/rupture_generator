//! What any first-arrival solver must satisfy, whatever scheme it uses.
//!
//! Carried from the port's contract and made generative: every property below is
//! quantified over generated grid shapes, spacings, slowness fields and seed sets,
//! because a property asserted at a single point is a smoke test, not a contract.
//! Accuracy is judged against **analytic truth** on the two media where truth is
//! known — uniform, and a constant gradient — never against another solver.
//!
//! # What is asserted, and what is deliberately not
//!
//! Several plausible-sounding properties are false for a discrete solution and are
//! called out here rather than left for someone to add:
//!
//! - **"Travel time increases with distance from the source"** is false in a
//!   heterogeneous medium — a fast channel reaches a distant cell before a slow one
//!   nearby. The correct universal statement is *causality*: every cell has a
//!   neighbour that ruptured earlier.
//! - **The Lipschitz bound `|T(a) − T(b)| ≤ ‖a − b‖₂ · s_max`** holds for the
//!   viscosity solution and is violated by a first-order scheme by ~20% on the
//!   diagonal — legitimate discretisation error, not a bug. Only the *neighbour* form
//!   is sharp.
//! - **Mesh convergence on max relative error** does not converge: the worst cell
//!   sits at a fixed grid offset from the source, so its relative error is
//!   resolution-independent. Absolute error is what converges.

mod common;

use _kernels::eikonal::{self, Error, Seed};
use common::{Grid, exact, grid, uniform_grid};
use proptest::prelude::*;

fn solve(grid: &Grid) -> Vec<f64> {
    eikonal::solve(&grid.slowness, grid.extent, grid.spacing_km, &grid.seeds)
        .expect("generated grids are valid inputs")
}

/// Straight-line distance between two nodes, each axis at its own spacing.
fn distance_km(a: (usize, usize), b: (usize, usize), spacing_km: (f64, f64)) -> f64 {
    let down = (exact(a.0) - exact(b.0)) * spacing_km.0;
    let across = (exact(a.1) - exact(b.1)) * spacing_km.1;
    (down * down + across * across).sqrt()
}

fn neighbours(i: usize, j: usize, extent: (usize, usize)) -> Vec<(usize, usize)> {
    let mut cells = Vec::with_capacity(4);
    if i > 0 {
        cells.push((i - 1, j));
    }
    if i + 1 < extent.0 {
        cells.push((i + 1, j));
    }
    if j > 0 {
        cells.push((i, j - 1));
    }
    if j + 1 < extent.1 {
        cells.push((i, j + 1));
    }
    cells
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// A multi-seed solve is the pointwise minimum of its single-seed solves.
    ///
    /// The property the seed contract rests on: first arrival from
    /// several sources *is* the minimum over sources. Today the implementation
    /// solves per seed and combines, so this is exact to the bit — asserted that
    /// tightly on purpose, because this test is the contract a future single-pass
    /// multi-source sweep has to meet, and whoever writes one will have to loosen
    /// this equality *here*, with the tolerance argued in front of them.
    #[test]
    fn a_multi_seed_solve_is_the_min_of_its_single_seed_solves(grid in grid(4)) {
        let combined = solve(&grid);

        let mut best = vec![f64::INFINITY; grid.slowness.len()];
        for seed in &grid.seeds {
            let alone = eikonal::solve(&grid.slowness, grid.extent, grid.spacing_km, &[*seed])
                .expect("a generated seed is valid alone");
            for (cell, arrival) in best.iter_mut().zip(&alone) {
                *cell = cell.min(*arrival);
            }
        }

        for (index, (&all, &min)) in combined.iter().zip(&best).enumerate() {
            prop_assert_eq!(
                all.to_bits(), min.to_bits(),
                "cell {}: {} from the joint solve, {} from the min of singles",
                index, all, min
            );
        }
    }

    /// No seed's cell ruptures after the time it was seeded at — and with one seed,
    /// it ruptures exactly then.
    ///
    /// The inequality is the multi-seed statement: an earlier wavefront is allowed
    /// to sweep past a later seed, and then the seed's `t0` is not the first
    /// arrival there.
    #[test]
    fn a_seed_never_ruptures_after_its_own_start_time(grid in grid(4)) {
        let times = solve(&grid);
        for seed in &grid.seeds {
            let arrival = times[grid.at(seed.i, seed.j)];
            prop_assert!(
                arrival <= seed.t0_s,
                "seed at ({}, {}) starts at {} but ruptures at {}",
                seed.i, seed.j, seed.t0_s, arrival
            );
        }
        if let [seed] = grid.seeds[..] {
            prop_assert_eq!(times[grid.at(seed.i, seed.j)].to_bits(), seed.t0_s.to_bits());
        }
    }

    /// Every non-seed cell has a neighbour that ruptured before it.
    ///
    /// The correct form of "the front expands outward". Distance monotonicity is
    /// *false* here: in a heterogeneous medium a fast channel reaches a far cell
    /// before a slow one nearby. Causality is not — a first arrival came from
    /// somewhere, and on a four-connected grid that somewhere is a face neighbour.
    #[test]
    fn every_cell_ruptures_after_a_neighbour(grid in grid(3)) {
        let times = solve(&grid);
        let (ni, nj) = grid.extent;
        for i in 0..ni {
            for j in 0..nj {
                if grid.is_seed(i, j) {
                    continue;
                }
                let here = times[grid.at(i, j)];
                let earlier = neighbours(i, j, grid.extent)
                    .into_iter()
                    .any(|(ni_, nj_)| times[grid.at(ni_, nj_)] < here);
                prop_assert!(
                    earlier,
                    "({}, {}) at {} is a local minimum; nothing reached it",
                    i, j, here
                );
            }
        }
    }

    /// Adjacent cells differ by no more than one cell's travel at the slowest speed.
    ///
    /// The *sharp* Lipschitz statement, per axis because the spacing is: the
    /// one-sided cap enforces `T(c) ≤ T(n) + h·s_c` structurally, and `s_c` is at
    /// most the grid's slowest cell. The relative slack admits the causal triangle
    /// roots, which satisfy the bound to discretisation rather than to rounding.
    #[test]
    fn neighbouring_cells_are_lipschitz(grid in grid(3)) {
        let times = solve(&grid);
        let (ni, nj) = grid.extent;
        let slowest = grid.slowest();
        for i in 0..ni {
            for j in 0..nj {
                let here = times[grid.at(i, j)];
                for (i2, j2) in neighbours(i, j, grid.extent) {
                    let bound = if i2 == i { grid.spacing_km.1 } else { grid.spacing_km.0 }
                        * slowest;
                    let step = (times[grid.at(i2, j2)] - here).abs();
                    prop_assert!(
                        step <= bound * (1.0 + 1e-6),
                        "({}, {}) to ({}, {}) jumps {} s, past the {} one cell at \
                         the slowest speed takes",
                        i, j, i2, j2, step, bound
                    );
                }
            }
        }
    }

    /// First arrival is sandwiched between the straight line and the lattice path.
    ///
    /// The strongest check available on a medium with no closed-form solution, and
    /// the one that catches the errors worth catching: speed used where slowness
    /// belongs, an axis's spacing on the wrong axis, the seed in the wrong cell.
    /// Above: the axis-only path is *a* path, so first arrival cannot beat it.
    /// Below: no path is shorter than the straight line and none faster than the
    /// medium's fastest cell, less one cell's slack for how the seed cell itself is
    /// discretised.
    #[test]
    fn arrival_is_between_the_straight_line_and_the_lattice_path(grid in grid(3)) {
        let times = solve(&grid);
        let (ni, nj) = grid.extent;
        let (h_i, h_j) = grid.spacing_km;
        let (slowest, fastest) = (grid.slowest(), grid.fastest());
        let slack = h_i.max(h_j) * slowest;

        for i in 0..ni {
            for j in 0..nj {
                let arrival = times[grid.at(i, j)];

                let along_axes = grid
                    .seeds
                    .iter()
                    .map(|seed| {
                        seed.t0_s
                            + (exact(i.abs_diff(seed.i)) * h_i + exact(j.abs_diff(seed.j)) * h_j)
                                * slowest
                    })
                    .fold(f64::INFINITY, f64::min);
                prop_assert!(
                    arrival <= along_axes * (1.0 + 1e-9),
                    "({}, {}) arrives at {}, later than the {} an axis-only path \
                     would take",
                    i, j, arrival, along_axes
                );

                let straight = grid
                    .seeds
                    .iter()
                    .map(|seed| {
                        seed.t0_s + distance_km((i, j), (seed.i, seed.j), grid.spacing_km) * fastest
                    })
                    .fold(f64::INFINITY, f64::min);
                prop_assert!(
                    arrival >= straight - slack,
                    "({}, {}) arrives at {}, sooner than the {} a straight line \
                     through the medium's fastest cell would take",
                    i, j, arrival, straight
                );
            }
        }
    }

    /// A faster medium never ruptures later.
    ///
    /// Two solves, and no analytic solution needed. It catches the error the
    /// sandwich bound is weakest against: a solver that inverts its input, using
    /// speed where slowness belongs, still produces a plausible field but reverses
    /// this.
    #[test]
    fn a_faster_medium_never_ruptures_later(grid in grid(3)) {
        let slow = solve(&grid);
        let faster: Vec<f64> = grid.slowness.iter().map(|slowness| slowness * 0.8).collect();
        let quick = eikonal::solve(&faster, grid.extent, grid.spacing_km, &grid.seeds)
            .expect("scaling slowness keeps it valid");
        for (index, (&was, &now)) in slow.iter().zip(&quick).enumerate() {
            prop_assert!(
                now <= was + 1e-12,
                "cell {} got later ({} from {}) when the medium got faster",
                index, now, was
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// On a uniform medium the solver is exact: `T = t0 + s·distance`, every cell.
    ///
    /// The factorisation's designed-in case — `τ ≡ 1` satisfies the discrete
    /// equations at any spacing, square cells or not, so the error is rounding, not
    /// truncation. Asserted at 1e-9 relative rather than the ~1e-14 measured,
    /// because the reference below accumulates its distance in a different order
    /// than the solver's `T₀`; both are far below the ~1e-2 a first-order scheme
    /// leaves on media that bend rays.
    #[test]
    fn a_uniform_medium_is_solved_exactly(grid in uniform_grid()) {
        let cells = grid.extent.0 * grid.extent.1;
        let times = eikonal::solve(
            &vec![grid.slowness; cells],
            grid.extent,
            grid.spacing_km,
            &[grid.seed],
        )
        .expect("a uniform medium is a valid input");

        for i in 0..grid.extent.0 {
            for j in 0..grid.extent.1 {
                let expected = grid.seed.t0_s
                    + grid.slowness
                        * distance_km((i, j), (grid.seed.i, grid.seed.j), grid.spacing_km);
                let error = (times[i * grid.extent.1 + j] - expected).abs();
                prop_assert!(
                    error <= 1e-9 * expected.max(1.0),
                    "({}, {}): {} against an exact {}",
                    i, j, times[i * grid.extent.1 + j], expected
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// Accuracy on the medium that discriminates: a constant gradient
// ---------------------------------------------------------------------------------

/// Slowness for `v(z) = v₀ + g·z`, `z` the down-dip depth on square cells of `h` km.
fn gradient_slowness(cells: usize, h: f64, v0: f64, g: f64) -> Vec<f64> {
    (0..cells * cells)
        .map(|index| 1.0 / (v0 + g * exact(index / cells) * h))
        .collect()
}

/// First-arrival time in a constant-gradient medium, exactly.
///
/// For `v(z) = v₀ + g·z` the rays are circular arcs and
///
/// ```text
///     T = (1/g) · arccosh(1 + g²·d² / (2·v₁·v₂))
/// ```
///
/// with `d` the straight-line distance and `v₁`, `v₂` the speeds at the two
/// endpoints. It depends on the endpoints' *depths* only through their speeds, not
/// on the path — which is what makes it usable as a per-cell reference, and it is an
/// analytic solution rather than a re-implementation of the subject (rule 5).
fn gradient_arrival_s(cell: (usize, usize), seed: (usize, usize), h: f64, v0: f64, g: f64) -> f64 {
    let speed_at = |depth_cells: usize| v0 + g * exact(depth_cells) * h;
    let distance = distance_km(cell, seed, (h, h));
    let cosh = 1.0 + g * g * distance * distance / (2.0 * speed_at(seed.0) * speed_at(cell.0));
    cosh.acosh() / g
}

/// Worst relative error against the analytic gradient solution, and the rounds the
/// sweep took.
fn gradient_error(cells: usize, h: f64, v0: f64, g: f64) -> (f64, usize) {
    let seed = Seed {
        i: cells / 2,
        j: cells / 2,
        t0_s: 0.0,
    };
    let (times, rounds) = eikonal::solve_with_rounds(
        &gradient_slowness(cells, h, v0, g),
        (cells, cells),
        (h, h),
        &[seed],
    )
    .expect("a gradient medium is a valid input");

    let mut worst = 0.0_f64;
    for i in 0..cells {
        for j in 0..cells {
            if (i, j) == (seed.i, seed.j) {
                continue;
            }
            let expected = gradient_arrival_s((i, j), (seed.i, seed.j), h, v0, g);
            worst = worst.max((times[i * cells + j] - expected).abs() / expected);
        }
    }
    (worst, rounds)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    /// Halving the spacing at least halves the error: first-order convergence.
    ///
    /// The discriminating accuracy measurement. A uniform medium rewards any scheme
    /// that is exact for plane waves; a gradient bends the rays into circular arcs
    /// and asks whether the scheme reproduces curvature. This is the check that
    /// told the two rejected solvers apart from the one that ships (`DEFECTS.md`
    /// 19): a scheme polluted by the source singularity does not converge at all.
    /// The bound is loose on purpose — 1.5 against a measured 2.0, where both
    /// rejected solvers sat at 1.01 and 1.03. Quantified over crustal-looking
    /// profiles, 1.5–3 km/s at the surface gaining 30–90 m/s per km.
    #[test]
    fn the_gradient_error_converges_at_first_order(v0 in 1.5_f64..3.0, g in 0.03_f64..0.09) {
        let (coarse, _) = gradient_error(33, 1.0, v0, g);
        let (fine, _) = gradient_error(65, 0.5, v0, g);
        let ratio = coarse / fine;
        prop_assert!(
            ratio > 1.5,
            "error {:.5e} at h=1.0 and {:.5e} at h=0.5 is a ratio of {:.2}; below \
             first order",
            coarse, fine, ratio
        );
    }
}

/// The round count is a property of the medium, not of the mesh.
///
/// Zhao's alternating orderings exist to give this: each ordering follows one family
/// of characteristics, so a fixed number of sweeps covers every ray direction however
/// fine the grid. It is the whole basis for the O(N) claim — a solver whose rounds
/// grew with the grid would be O(N log N) at best.
#[test]
fn the_sweep_count_does_not_grow_with_the_mesh() {
    let counts: Vec<(usize, usize)> = [(33, 1.0), (65, 0.5), (129, 0.25)]
        .into_iter()
        .map(|(cells, h)| (cells, gradient_error(cells, h, 2.0, 0.06).1))
        .collect();

    let first = counts[0].1;
    for (cells, rounds) in &counts {
        assert_eq!(
            *rounds, first,
            "a {cells}x{cells} grid took {rounds} rounds where 33x33 took {first}"
        );
    }
}

// ---------------------------------------------------------------------------------
// Refusals: bad inputs are named, not solved around
// ---------------------------------------------------------------------------------

const OK_SLOWNESS: f64 = 0.4;

fn valid_seed() -> Seed {
    Seed {
        i: 1,
        j: 2,
        t0_s: 0.0,
    }
}

#[test]
fn an_out_of_bounds_seed_is_refused_by_name() {
    let error = eikonal::solve(
        &[OK_SLOWNESS; 12],
        (3, 4),
        (0.5, 0.5),
        &[
            valid_seed(),
            Seed {
                i: 3,
                j: 0,
                t0_s: 0.0,
            },
        ],
    )
    .expect_err("seed 1 is off the down-dip edge");
    assert_eq!(
        error,
        Error::SeedOutOfBounds {
            seed: 1,
            i: 3,
            j: 0,
            ni: 3,
            nj: 4
        }
    );
    let message = error.to_string();
    assert!(
        message.contains("seed 1") && message.contains("(3, 0)") && message.contains("3x4"),
        "the refusal must name the seed and the grid: {message}"
    );
}

#[test]
fn a_cell_no_wave_can_cross_is_refused_by_name() {
    for bad in [0.0, -0.3, f64::NAN, f64::INFINITY] {
        let mut slowness = vec![OK_SLOWNESS; 12];
        slowness[7] = bad;
        let error = eikonal::solve(&slowness, (3, 4), (0.5, 0.5), &[valid_seed()])
            .expect_err("cell (1, 3) is not a medium");
        assert!(
            matches!(error, Error::NonPositiveSlowness { i: 1, j: 3, .. }),
            "{bad} at (1, 3) gave {error}"
        );
    }
}

#[test]
fn a_degenerate_spacing_is_refused_by_axis() {
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let error = eikonal::solve(&[OK_SLOWNESS; 12], (3, 4), (0.5, bad), &[valid_seed()])
            .expect_err("the along-strike spacing is not a length");
        assert!(
            matches!(
                error,
                Error::NonPositiveSpacing {
                    axis: "along-strike",
                    ..
                }
            ),
            "spacing {bad} gave {error}"
        );
    }
}

#[test]
fn no_seeds_no_grid_and_wrong_lengths_are_refused() {
    assert_eq!(
        eikonal::solve(&[OK_SLOWNESS; 12], (3, 4), (0.5, 0.5), &[]),
        Err(Error::NoSeeds)
    );
    assert_eq!(
        eikonal::solve(&[], (0, 4), (0.5, 0.5), &[valid_seed()]),
        Err(Error::EmptyGrid { ni: 0, nj: 4 })
    );
    assert_eq!(
        eikonal::solve(&[OK_SLOWNESS; 11], (3, 4), (0.5, 0.5), &[valid_seed()]),
        Err(Error::WrongLength {
            ni: 3,
            nj: 4,
            got: 11
        })
    );
}

#[test]
fn a_seed_with_no_time_is_refused() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = eikonal::solve(
            &[OK_SLOWNESS; 12],
            (3, 4),
            (0.5, 0.5),
            &[Seed {
                i: 0,
                j: 0,
                t0_s: bad,
            }],
        )
        .expect_err("a seed time must be finite");
        assert!(
            matches!(error, Error::NonFiniteSeedTime { seed: 0, .. }),
            "t0 = {bad} gave {error}"
        );
    }
}
