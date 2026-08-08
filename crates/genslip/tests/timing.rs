//! What the solver and the pipeline cost, measured rather than asserted.
//!
//! `#[ignore]`d, like the SRF parser's throughput tests and for the same reason: the
//! gate answers questions about behaviour and these answer one about cost. Run them
//! deliberately, before and after a change that claims to be faster.
//!
//! ```sh
//! cargo test -p genslip --release -- --ignored --nocapture solver_scaling
//! cargo test -p genslip --release -- --ignored --nocapture whole_rupture
//! ```
//!
//! **`--release` matters here** and nowhere else in this suite: the sweep is a tight
//! numeric loop and a debug build measures the borrow checker's bookkeeping rather
//! than the arithmetic.
//!
//! # What the scaling number means
//!
//! Fast sweeping is O(N): four sequential passes over the grid, a fixed number of
//! rounds, no heap and no sort. The way to see that in a measurement is **time per
//! cell**, which should stay flat as the grid grows. A scheme that sorted each
//! wavefront would show it rising as the square root of the cell count, and one with a
//! priority queue would show it rising logarithmically.
//!
//! For scale, the solver this replaced: genslip's `wafront2d.f` orders each expanding
//! ring with a *selection sort*, making it O(N^1.5) — and `SORT` is declared
//! `DIMENSION TI(400)`, so a fault more than 400 subfaults across would have read past
//! the end of its scratch array. Neither limit survives.
#![cfg(feature = "fftw")]

mod common;

use common::fixture;
use genslip::rupture::{EikonalSolver, FactoredSweep, Hypocentre, SpeedGrid};
use std::time::Instant;

/// A rupture-speed field with structure in both directions, as a real fault has.
fn speed_field(cells: usize) -> SpeedGrid {
    let values = (0..cells * cells)
        .map(|index| {
            #[expect(clippy::cast_precision_loss, reason = "grid indices")]
            let (strike, dip) = ((index % cells) as f32, (index / cells) as f32);
            2.0 + 0.7 * (dip * 0.05).tanh() + 0.3 * (strike * 0.02).sin()
        })
        .collect();
    SpeedGrid::new(cells, cells, values)
}

/// Time per cell, across two orders of magnitude of grid.
///
/// Flat means O(N). The absolute numbers are machine-dependent and not the point; the
/// shape of the column is.
#[test]
#[ignore = "measures time, not behaviour"]
fn solver_scaling() {
    println!(
        "\n{:>7} {:>10} {:>12} {:>8} {:>16}",
        "cells", "subfaults", "total", "rounds", "ns/cell/round"
    );

    for cells in [32_usize, 64, 128, 256, 512, 1024] {
        let speed = speed_field(cells);
        let hypocentre = Hypocentre {
            strike: cells / 3,
            dip: cells / 3,
        };

        // Best of three: the first pass warms the allocator and the page tables.
        let mut best = std::time::Duration::MAX;
        let mut rounds = 0;
        for _ in 0..3 {
            let mut solver = FactoredSweep::new();
            let start = Instant::now();
            let times = solver.solve(&speed, hypocentre, 1.0);
            best = best.min(start.elapsed());
            rounds = solver.rounds();
            std::hint::black_box(times.time(0, 0));
        }

        // Per cell AND per round. The round count is a property of the medium, and
        // these grids are physically bigger rather than more finely sampled -- same
        // spacing, more cells -- so the medium genuinely gains structure and the count
        // steps from 2 to 3. Dividing it out is what isolates the per-cell cost, which
        // is the O(N) claim. (Refining the *mesh* at a fixed domain leaves the count
        // alone; `the_sweep_count_does_not_grow_with_the_mesh` asserts that.)
        #[expect(clippy::cast_precision_loss, reason = "grid sizes are small")]
        let per_cell_round = best.as_nanos() as f64 / (cells * cells * rounds) as f64;
        println!(
            "{cells:>7} {:>10} {:>12?} {rounds:>8} {per_cell_round:>16.1}",
            cells * cells,
            best
        );
    }
    println!();
}

/// The whole pipeline, so the solver's share of it is visible.
///
/// Worth knowing before optimising anything: on a realistic fault the solve is a small
/// fraction of a rupture, which is dominated by the six spectral fields and their
/// transforms. A faster solver is worth having for accuracy and for dropping the
/// Fortran, not because it was the bottleneck.
#[test]
#[ignore = "measures time, not behaviour"]
fn whole_rupture() {
    let grid = fixture::fault();
    let mut best = std::time::Duration::MAX;
    for _ in 0..5 {
        let start = Instant::now();
        let model = genslip::realisation::generate(
            &mut genslip::rng::GenslipLcg::new(fixture::SEED),
            &mut genslip::fft::FftwFft::new(),
            &mut FactoredSweep::new(),
            &grid,
            &fixture::velocity_model(),
            fixture::source_spec(),
            fixture::slip_spec(),
            fixture::timing_spec(),
            fixture::hypocentre(),
        );
        best = best.min(start.elapsed());
        std::hint::black_box(model.moment_dyne_cm);
    }

    let subfaults = grid.extents.fault_strike * grid.extents.fault_dip;
    println!("\nwhole rupture, {subfaults} subfaults: {best:?}");

    // The solver alone, on the same grid, for the share.
    let (shear, _) = fixture::velocity_model().sample(grid.extents.fault_strike, &grid.depth_km);
    let fraction = genslip::taper::SlipField::from_values(
        grid.extents.fault_strike,
        grid.extents.fault_dip,
        grid.velocity_fraction.clone(),
    );
    let speed = genslip::rupture::speed_field(
        &shear,
        &fraction,
        &grid.depth_km,
        fixture::timing_spec().speed_profile,
    );

    let mut solve_best = std::time::Duration::MAX;
    for _ in 0..5 {
        let start = Instant::now();
        let times = FactoredSweep::new().solve(&speed, fixture::hypocentre(), 1.0);
        solve_best = solve_best.min(start.elapsed());
        std::hint::black_box(times.time(0, 0));
    }
    println!(
        "  of which the eikonal solve: {solve_best:?} ({:.1}%)\n",
        100.0 * solve_best.as_secs_f64() / best.as_secs_f64()
    );
}
