//! What must hold about a rupture, whatever the numbers come out as.
//!
//! A red here is a bug — never an argument. That is what separates this file from
//! the calibration and reference-agreement tests: nothing below has a tolerance
//! anyone can debate, because nothing below is about *values*. These are claims about
//! **registration** — which subfault is which, which axis is which, which guard fired
//! — and about structure.
//!
//! # Why registration rather than conservation
//!
//! The obvious invariants are the conservation ones, and they are worthless.
//! `sum(mu*A*s) == M0` holds for any area and any rigidity, because the code divides
//! by exactly that sum. `dt*sum(pulse) == slip` passes for a boxcar. `min(onset) ==
//! delay` is a subtraction asserting itself. Each is a tautology dressed as a law.
//!
//! Registration identities are not, because they relate two things the code does
//! *not* derive from each other — a configured index and a computed argmin, a
//! configured spread and a realised one, a guard's threshold and an emitted pulse.
//! Four of the five defects the corpus found are caught here, exactly, for free:
//!
//! | | caught by |
//! | --- | --- |
//! | `DEFECTS.md` 14 (rake spread 1/20th) | `the_rake_field` |
//! | `DEFECTS.md` 16 (pulse on a silent subfault) | `the_slip_rate_guard` |
//! | `DEFECTS.md` 17 (hypocentre a cell off) | `the_hypocentre` |
//! | `DEFECTS.md` 18 (slip spread 63% too wide) | the spread contract in `the_slip_field` |
//!
//! **That table is verified, not asserted.** `teeth.sh` alongside this file puts each
//! defect back into the library and expects the named contract to fail; a claim of
//! this kind decays silently otherwise, because a fixture that stops having silent
//! subfaults makes a guard test vacuous without making it red. Run it when changing
//! anything here.
//!
//! # Generic over the engines on purpose
//!
//! Every contract takes its FFT and its solver as parameters, so replacing either is
//! a configuration change rather than a test rewrite. `contract_for!` at the bottom
//! instantiates them; a second instantiation is one line.
#![cfg(all(feature = "fftw", feature = "wavefront-compat"))]

mod common;

use common::counting::{CountingSource, field_draw_count};
use common::fixture;
use common::stats::{self, lag_one_along_dip, lag_one_along_strike};
use common::tolerance::f32_sum_relative;
use genslip::fft::Fft;
use genslip::field::{CorrelationLengths, Spectrum2D};
use genslip::grid::Spectrum;
use genslip::realisation::{RuptureModel, generate};
use genslip::rng::GenslipLcg;
use genslip::rupture::{EikonalSolver, Hypocentre, TravelTimes};
use genslip::slip::{SpectrumSpec, generate_normalised};
use genslip::slip_rate::MIN_SLIP_CM;
use genslip::taper::{EdgeTapers, SlipField, taper_edges};
use num_complex::Complex32;

/// Generate the fixture rupture, optionally with the timing perturbation switched off.
fn rupture<F: Fft, E: EikonalSolver>(
    fft: &mut F,
    solver: &mut E,
    seed: i64,
    perturbed: bool,
) -> RuptureModel {
    let mut timing = fixture::timing_spec();
    if !perturbed {
        timing.rupture_time_scale = 0.0;
    }
    generate(
        &mut GenslipLcg::new(seed),
        fft,
        solver,
        &fixture::fault(),
        &fixture::velocity_model(),
        fixture::source_spec(),
        fixture::slip_spec(),
        timing,
        fixture::hypocentre(),
    )
}

/// The travel-time field alone, with no perturbation on top.
fn travel_times<E: EikonalSolver>(solver: &mut E, hypocentre: Hypocentre) -> TravelTimes {
    let grid = fixture::fault();
    let (shear_speed, _) =
        fixture::velocity_model().sample(grid.extents.fault_strike, &grid.depth_km);
    let velocity_fraction = SlipField::from_values(
        grid.extents.fault_strike,
        grid.extents.fault_dip,
        grid.velocity_fraction.clone(),
    );
    let speed = genslip::rupture::speed_field(
        &shear_speed,
        &velocity_fraction,
        &grid.depth_km,
        fixture::timing_spec().speed_profile,
    );
    solver.solve(&speed, hypocentre, f64::from(grid.spacing.strike_km))
}

// ---------------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------------

/// Rupture starts where it was told to start.
///
/// `DEFECTS.md` 17 put the source one cell along strike and one down dip. The onset
/// field it produced was smooth, started at zero, and correlated 0.92 to 0.997 with
/// the truth — every "is this the right shape" diagnostic said yes. What it could not
/// do is put the zero in the right place, and that is a fact rather than a tolerance.
fn the_hypocentre<F: Fft, E: EikonalSolver>(fft: &mut F, solver: &mut E) {
    let hypocentre = fixture::hypocentre();
    let travel = travel_times(solver, hypocentre);

    // Exactly zero, not nearly: the solver seeds this cell rather than computing it.
    assert_eq!(
        travel.time(hypocentre.strike, hypocentre.dip).to_bits(),
        0.0_f64.to_bits(),
        "travel time at the hypocentre must be exactly zero"
    );

    // And uniquely so. A solver that zeroed the whole grid would pass the line above.
    for dip in 0..travel.dip_count() {
        for strike in 0..travel.strike_count() {
            if (strike, dip) == (hypocentre.strike, hypocentre.dip) {
                continue;
            }
            assert!(
                travel.time(strike, dip) > 0.0,
                "({strike}, {dip}) is not later than the hypocentre"
            );
        }
    }

    // End to end, with the perturbation off so the earliest subfault IS the source:
    // the whole rupture starts at the configured delay, at the configured cell.
    let model = rupture(fft, solver, fixture::SEED, false);
    let onset = model.onset_s.as_slice();
    let at_hypocentre = model.onset_s.time(hypocentre.strike, hypocentre.dip);
    let earliest = onset.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        (at_hypocentre - earliest).abs() < 1e-12,
        "the earliest subfault is not the hypocentre: {at_hypocentre} vs {earliest}"
    );
    assert!(
        (at_hypocentre - f64::from(fixture::timing_spec().rupture_delay_s)).abs() < 1e-12,
        "onset at the hypocentre is {at_hypocentre}, not the configured delay"
    );
}

/// A subfault that does not slip emits no pulse, and one that does emits one.
///
/// The guard lives in genslip's SRF *loader*, outside the generator, so porting the
/// generator faithfully did not reproduce it — `DEFECTS.md` 16. On a tapered fault
/// this is every edge subfault.
fn the_slip_rate_guard<F: Fft, E: EikonalSolver>(fft: &mut F, solver: &mut E) {
    let model = rupture(fft, solver, fixture::SEED, true);
    let slip = model.slip.slip.as_slice();

    let mut silent = 0;
    for (index, pulse) in model.slip_rate.iter().enumerate() {
        let cm = slip[index].abs();
        if cm > MIN_SLIP_CM {
            assert!(
                !pulse.is_empty(),
                "subfault {index} slips {cm} cm and emits nothing"
            );
        } else {
            silent += 1;
            assert!(
                pulse.is_empty(),
                "subfault {index} slips {cm} cm, below {MIN_SLIP_CM}, and emits \
                 {} samples",
                pulse.len()
            );
        }
    }

    // The taper guarantees some. Without this the loop above is vacuously true, which
    // is exactly how the defect survived.
    assert!(silent > 0, "no subfault is below the slip threshold");
}

/// The rake field's spread is `rake_sigma_deg`, in **degrees**.
///
/// `DEFECTS.md` 14 handed it the slip field's coefficient of variation — 0.75,
/// dimensionless — where a spread of 15 degrees belonged, a factor of twenty. Both
/// numbers are in the fixture, so this cannot pass by coincidence.
fn the_rake_field<F: Fft, E: EikonalSolver>(fft: &mut F, solver: &mut E) {
    let model = rupture(fft, solver, fixture::SEED, true);
    let grid = fixture::fault();
    let spec = fixture::slip_spec();

    let deviation: Vec<f64> = model
        .rake_deg
        .as_slice()
        .iter()
        .zip(&grid.base_rake_deg)
        .map(|(rake, base)| f64::from(rake - base))
        .collect();
    let spread = stats::population_sigma(&deviation);

    // The rescale is a single-precision fold over every subfault, so the bound is the
    // fold's, not a chosen number.
    let tolerance = f64::from(spec.rake_sigma_deg) * f32_sum_relative(deviation.len()) * 10.0;
    assert!(
        (spread - f64::from(spec.rake_sigma_deg)).abs() < tolerance.max(1e-3),
        "rake spread {spread} degrees, configured {}",
        spec.rake_sigma_deg
    );

    // And it is not the slip field's spread, which is what it was.
    let slip_variation = f64::from(spec.spectrum.coefficient_of_variation);
    assert!(
        (spread - slip_variation).abs() > 1.0,
        "rake spread {spread} is indistinguishable from the slip CoV {slip_variation}"
    );
}

/// Strike is the fast axis, and a transposition would show.
///
/// Every statistic symmetric in the two axes — mean, moment, coefficient of
/// variation, a radially pooled power spectrum — is invariant under transposing the
/// grid, and so is a square fixture. This is the contract that is not.
fn the_axis_order<F: Fft>(fft: &mut F) {
    let grid = fixture::wide_fault();
    let mut spec: SpectrumSpec = fixture::spectrum_spec();
    // Deliberately far apart, so the claim is about the axes rather than about
    // whether the statistic can resolve 7.9 km from 5.8 km at 1 km spacing.
    spec.correlation = CorrelationLengths {
        strike: 12.0,
        dip: 3.0,
    };

    let generated = generate_normalised(
        &mut GenslipLcg::new(fixture::SEED),
        fft,
        grid.extents,
        grid.spacing,
        spec,
    );
    let field = stats::widen(generated.field.as_slice());

    let along = lag_one_along_strike(&field, grid.extents.fault_strike, grid.extents.fault_dip);
    let down = lag_one_along_dip(&field, grid.extents.fault_strike, grid.extents.fault_dip);
    assert!(
        along > down,
        "a field with a 12 km strike corner and a 3 km dip corner is smoother down \
         dip ({down:.4}) than along strike ({along:.4}); the axes are transposed"
    );
}

// ---------------------------------------------------------------------------------
// Structure
// ---------------------------------------------------------------------------------

/// The generated slip field's spread is the configured one, or Frankel's own.
///
/// `DEFECTS.md` 18 stretched a Frankel field where the original shifts it, leaving
/// slip correlated 0.993 with the truth and 63% too variable. The exception is the
/// physics, not an exemption: `normalises_from_its_minimum` says which spectra
/// honour the configured value and which set their own.
fn the_slip_field<F: Fft>(fft: &mut F) {
    let grid = fixture::fault();
    for shape in [
        Spectrum2D::Somerville,
        Spectrum2D::Mai,
        Spectrum2D::Frankel,
        Spectrum2D::Suzuki,
    ] {
        let mut spec = fixture::spectrum_spec();
        spec.shape = shape;
        spec.correlation = fixture::corner_lengths();

        let generated = generate_normalised(
            &mut GenslipLcg::new(fixture::SEED),
            fft,
            grid.extents,
            grid.spacing,
            spec,
        );
        let field = stats::widen(generated.field.as_slice());

        // Unit mean, whichever branch was taken.
        let mean = stats::mean(&field);
        assert!((mean - 1.0).abs() < 1e-4, "{shape:?}: mean {mean} is not 1");

        if shape.normalises_from_its_minimum() {
            // Shifted: the least-slipping subfault is exactly zero, and the
            // configured spread is ignored.
            let minimum = field.iter().copied().fold(f64::INFINITY, f64::min);
            assert!(
                minimum.abs() < 1e-9,
                "{shape:?}: shifted fields put their minimum at zero, got {minimum}"
            );
        } else {
            // Stretched: the spread is what was asked for.
            let spread = stats::population_sigma(&field);
            let wanted = f64::from(spec.coefficient_of_variation);
            assert!(
                (spread - wanted).abs() < 1e-3,
                "{shape:?}: spread {spread}, configured {wanted}"
            );
        }
    }
}

/// Hermitian symmetry, so the inverse transform is real.
///
/// Bitwise, because it is imposed by copying rather than computed: a conjugate that
/// merely nearly matches means the symmetry pass missed a cell.
fn the_spectrum_is_hermitian<F: Fft>(_fft: &mut F) {
    let (strike_count, dip_count) = (28, 16);
    let mut spectrum = Spectrum::zeros(strike_count, dip_count);
    genslip::field::correlated_field(
        &mut spectrum,
        &mut GenslipLcg::new(fixture::SEED),
        Spectrum2D::Mai,
        genslip::field::WavenumberStep {
            strike: 0.05,
            dip: 0.07,
        },
        fixture::corner_lengths(),
        genslip::field::WavelengthBand::new(1.5, 80.0),
        3.5,
    );

    // Negating an exact zero flips its sign bit, and `+0.0` and `-0.0` are the same
    // *value* -- which is what conjugate symmetry is about. Comparing raw bits would
    // report the self-conjugate points as broken, so zeros are canonicalised first.
    // Everything else stays bitwise, because the symmetry is imposed by copying.
    let canonical = |value: f32| {
        if value == 0.0 {
            0.0_f32.to_bits()
        } else {
            value.to_bits()
        }
    };

    for dip in 0..dip_count {
        for strike in 0..strike_count {
            let mirrored = spectrum[(
                (strike_count - strike) % strike_count,
                (dip_count - dip) % dip_count,
            )];
            let here = spectrum[(strike, dip)];
            assert_eq!(
                canonical(mirrored.re),
                canonical(here.re),
                "real part at ({strike}, {dip}) is not mirrored"
            );
            assert_eq!(
                canonical(mirrored.im),
                canonical(-here.im),
                "imaginary part at ({strike}, {dip}) is not conjugated"
            );
        }
    }

    // The four self-conjugate points must be real outright -- they are their own
    // mirror, so the check above is vacuous for them.
    for point in [
        (0, 0),
        (strike_count / 2, 0),
        (0, dip_count / 2),
        (strike_count / 2, dip_count / 2),
    ] {
        assert_eq!(
            spectrum[point].im.to_bits(),
            0.0_f32.to_bits(),
            "self-conjugate point {point:?} carries an imaginary part"
        );
    }
}

/// `correlate_with` is an algebraic identity, not a statistical property.
///
/// The map is elementwise-linear, so `t <- rho*r + sqrt(1-rho^2)*t` holds pointwise
/// and exactly. Testing it with a correlation coefficient would cost several
/// realisations and have about a millionth of the power: a rho of 0.8 implemented as
/// 0.5 is a 1e6 margin here and under one standard error there.
fn correlation_is_exact() {
    for rho in [0.0_f32, 0.3, 0.8, 1.0] {
        let mut target = Spectrum::zeros(4, 4);
        let mut reference = Spectrum::zeros(4, 4);
        for index in 0..16 {
            target.as_mut_slice()[index] = Complex32::new(0.0, 1.0);
            reference.as_mut_slice()[index] = Complex32::new(1.0, 0.0);
        }

        genslip::field::correlate_with(&mut target, &reference, rho);

        let residual = (1.0 - rho * rho).sqrt();
        for value in target.as_slice() {
            assert!(
                (value.re - rho).abs() < 1e-6 && (value.im - residual).abs() < 1e-6,
                "rho {rho}: got {value}, wanted {rho} + {residual}i"
            );
        }
    }
}

/// Tapering only ever reduces, and reduces the edge it was pointed at.
fn the_taper_is_a_contraction() {
    let (strike_count, dip_count) = (24, 14);
    let original =
        SlipField::from_values(strike_count, dip_count, vec![1.0; strike_count * dip_count]);

    let mut tapered = original.clone();
    taper_edges(
        &mut tapered,
        &EdgeTapers {
            sides: 0.1,
            top: 0.2,
            bottom: 0.0,
        },
    );

    for (after, before) in tapered.as_slice().iter().zip(original.as_slice()) {
        assert!(
            after.abs() <= before.abs() + f32::EPSILON,
            "the taper amplified {before} to {after}"
        );
        assert!(*after > 0.0, "the taper zeroed a subfault outright");
    }

    // Pointed up dip: the shallowest row is damped and the deepest is not. Getting
    // this backwards is invisible to any statistic over the whole field.
    let row = |dip: usize| tapered[(strike_count / 2, dip)];
    assert!(
        row(0) < row(dip_count - 1),
        "a top taper damped the deep edge: {} vs {}",
        row(0),
        row(dip_count - 1)
    );
}

/// Truncation leaves nothing negative.
fn truncation_leaves_no_negative_slip<F: Fft, E: EikonalSolver>(fft: &mut F, solver: &mut E) {
    let model = rupture(fft, solver, fixture::SEED, true);
    for (index, value) in model.slip.slip.as_slice().iter().enumerate() {
        assert!(*value >= 0.0, "subfault {index} slips {value} cm");
    }
}

// ---------------------------------------------------------------------------------
// The draw stream
// ---------------------------------------------------------------------------------

/// A rupture consumes exactly the randomness the pipeline accounts for.
///
/// Four spectral fields on the padded grid, plus two skipped fields on the refined
/// grid that genslip builds and never uses (`PRUNED.md`) but whose draws are not
/// optional. Counted through a decorator rather than through the LCG's exposed state,
/// so the same assertion covers the modern generator.
///
/// **Count is not order.** Two stages drawing the same number of deviates could be
/// swapped and this would still pass; what catches that is the corpus, pointwise,
/// because a permutation changes every field downstream.
fn the_draw_count<F: Fft, E: EikonalSolver>(fft: &mut F, solver: &mut E) {
    let grid = fixture::fault();
    let mut counting = CountingSource::new(GenslipLcg::new(fixture::SEED));

    let _ = generate(
        &mut counting,
        fft,
        solver,
        &grid,
        &fixture::velocity_model(),
        fixture::source_spec(),
        fixture::slip_spec(),
        fixture::timing_spec(),
        fixture::hypocentre(),
    );

    let (padded_strike, padded_dip) = (grid.extents.padded_strike, grid.extents.padded_dip);
    let refined = padded_strike.max(padded_dip) * 3;
    let expected =
        4 * field_draw_count(padded_strike, padded_dip) + 2 * field_draw_count(refined, refined);

    assert_eq!(
        counting.gaussians(),
        expected,
        "slip, rake and the two perturbations are {} each; the two skipped fields are \
         {} each",
        field_draw_count(padded_strike, padded_dip),
        field_draw_count(refined, refined)
    );

    // The skipped fields dominate, which is worth stating: they are the reason a
    // rupture costs what it does in randomness.
    assert!(
        2 * field_draw_count(refined, refined) > 4 * field_draw_count(padded_strike, padded_dip),
        "the skipped fields stopped being the larger consumer"
    );
}

/// Run every contract against one pair of engines.
macro_rules! contract_for {
    ($name:ident, $fft:expr, $solver:expr) => {
        mod $name {
            use super::*;

            #[test]
            fn rupture_starts_at_the_hypocentre() {
                the_hypocentre(&mut $fft, &mut $solver);
            }

            #[test]
            fn a_silent_subfault_emits_no_pulse() {
                the_slip_rate_guard(&mut $fft, &mut $solver);
            }

            #[test]
            fn the_rake_spread_is_in_degrees() {
                the_rake_field(&mut $fft, &mut $solver);
            }

            #[test]
            fn strike_is_the_fast_axis() {
                the_axis_order(&mut $fft);
            }

            #[test]
            fn the_slip_field_has_the_spread_its_spectrum_implies() {
                the_slip_field(&mut $fft);
            }

            #[test]
            fn the_spectrum_is_conjugate_symmetric() {
                the_spectrum_is_hermitian(&mut $fft);
            }

            #[test]
            fn tapering_only_reduces() {
                the_taper_is_a_contraction();
            }

            #[test]
            fn no_slip_is_negative() {
                truncation_leaves_no_negative_slip(&mut $fft, &mut $solver);
            }

            #[test]
            fn the_randomness_is_accounted_for() {
                the_draw_count(&mut $fft, &mut $solver);
            }
        }
    };
}

contract_for!(
    fftw_and_wavefront,
    genslip::fft::FftwFft::new(),
    genslip::rupture::Wavefront2d::new()
);

#[test]
fn correlating_two_spectra_is_exact() {
    correlation_is_exact();
}
