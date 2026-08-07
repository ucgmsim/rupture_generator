//! A whole rupture model, end to end.
//!
//! # What this gate is and is not
//!
//! It is **not** a parity test. There is no C function that produces a rupture model;
//! `main` does, and only into a file. Every stage is pinned individually elsewhere.
//!
//! What this pins is the *composition*: that the stages run in the right order, that
//! the draw sequence is what the original's is, and that a seed reproduces a model.
//! It is deliberately a small number of summary statistics rather than the whole
//! model, because the model is 10^4 numbers and a diff of those tells you nothing.
//!
//! **It goes red by design** when a commit changes the draw structure — count, order,
//! or generator. That is the gate asking for an adjudication, which is the correct
//! answer for exactly those commits. Re-record with `UPDATE_SNAPSHOT=1` and say why
//! in the commit message; a re-recorded snapshot with no explanation is
//! indistinguishable from a silently broken one.
#![cfg(all(feature = "fftw", feature = "wavefront-compat"))]

use genslip::fft::FftwFft;
use genslip::field::{Spectrum2D, WavelengthBand};
use genslip::realisation::{FaultGrid, SlipSpec, SourceSpec, TimingSpec, generate};
use genslip::rise_time::{DepthRamp, RiseTimeSpec, RiseTimeStretch, Weighting};
use genslip::rng::{GenslipLcg, Realisations as _};
use genslip::rupture::{Hypocentre, SpeedProfile, Wavefront2d};
use genslip::slip::{GridExtents, PerturbationSpec, SpectrumSpec, SubfaultSpacing};
use genslip::slip_rate::BetaProfile;
use genslip::source::{CornerRelation, Layer, MagnitudeScale, VelocityModel};
use genslip::taper::EdgeTapers;

/// A fault small enough to run in milliseconds and large enough that every depth
/// ramp has subfaults on both sides of it.
fn fault() -> FaultGrid {
    let (strike_count, dip_count) = (24, 14);
    FaultGrid {
        extents: GridExtents {
            fault_strike: strike_count,
            fault_dip: dip_count,
            padded_strike: 28,
            padded_dip: 16,
        },
        spacing: SubfaultSpacing {
            strike_km: 1.0,
            dip_km: 1.0,
        },
        // 0.5 km to 20.5 km: through the shallow rise-time ramp at 6.5 and the deep
        // one at 17.5.
        #[expect(clippy::cast_precision_loss, reason = "small test indices")]
        depth_km: (0..dip_count).map(|dip| 0.5 + dip as f32 * 1.5).collect(),
        base_rake_deg: vec![175.0; strike_count * dip_count],
        velocity_fraction: vec![0.8; strike_count * dip_count],
    }
}

fn velocity_model() -> VelocityModel {
    VelocityModel::new(vec![
        Layer {
            bottom_depth_km: 1.0,
            shear_speed_km_s: 1.8,
            density_g_cm3: 2.1,
        },
        Layer {
            bottom_depth_km: 5.0,
            shear_speed_km_s: 2.6,
            density_g_cm3: 2.4,
        },
        Layer {
            bottom_depth_km: 12.0,
            shear_speed_km_s: 3.2,
            density_g_cm3: 2.6,
        },
        Layer {
            bottom_depth_km: 30.0,
            shear_speed_km_s: 3.6,
            density_g_cm3: 2.7,
        },
    ])
}

fn source_spec() -> SourceSpec {
    SourceSpec {
        magnitude: 6.8,
        magnitude_scale: MagnitudeScale::Moment,
        corners: CornerRelation::Mai {
            strike_offset: 2.50,
            dip_offset: 1.50,
            circular: false,
        },
        modified_corners: false,
        rise_time_coefficient: 1.6,
        average_dip_deg: 60.0,
        average_rake_deg: 175.0,
    }
}

fn slip_spec() -> SlipSpec {
    SlipSpec {
        spectrum: SpectrumSpec {
            shape: Spectrum2D::Mai,
            // Overwritten by the magnitude relation inside `generate`.
            correlation: genslip::field::CorrelationLengths {
                strike: 1.0,
                dip: 1.0,
            },
            band: WavelengthBand::new(1.5, 80.0),
            coefficient_of_variation: 0.75,
            phase_shift: (0.0, 0.0),
        },
        tapers: EdgeTapers {
            sides: 0.02,
            top: 0.0,
            bottom: 0.0,
        },
        truncate_negative: true,
        water_level: 0.0,
        // genslip's `rake_sigma` default, in degrees. Deliberately not the 0.75
        // above: that is the slip field's coefficient of variation, dimensionless,
        // and confusing the two is `DEFECTS.md` 14.
        rake_sigma_deg: 15.0,
    }
}

fn timing_spec() -> TimingSpec {
    let shallow = DepthRamp {
        centre_km: 6.5,
        half_width_km: 1.5,
    };
    let deep = DepthRamp {
        centre_km: 17.5,
        half_width_km: 2.5,
    };
    TimingSpec {
        rupture_time: PerturbationSpec {
            correlation: 0.8,
            sigma: 1.0,
        },
        rupture_time_scale: -0.35,
        rupture_delay_s: 0.0,
        rise_time: RiseTimeSpec {
            perturbation: PerturbationSpec {
                correlation: 0.9,
                sigma: 0.75,
            },
            shallow_blend: DepthRamp {
                centre_km: 2.0,
                half_width_km: 1.0,
            },
            slip_exponent: 0.5,
        },
        rise_time_stretch: RiseTimeStretch {
            shallow,
            shallow_factor: 2.0,
            deep,
            deep_factor: 2.0,
        },
        rise_time_weighting: Weighting::Uniform,
        speed_profile: SpeedProfile {
            shallow,
            shallow_factor: 0.6,
            deep,
            deep_factor: 0.6,
        },
        beta: BetaProfile {
            shallow_ramp: DepthRamp {
                centre_km: 2.0,
                half_width_km: 1.0,
            },
            shallow: 0.5,
            mid_ramp: DepthRamp {
                centre_km: 6.5,
                half_width_km: 1.5,
            },
            mid: 0.13,
            deep: 0.13,
        },
        sample_interval_s: 0.005,
        max_samples: 100_000,
    }
}

/// Summary statistics, formatted so a diff points at which one moved.
fn summarise(seed: i64, realisation: u64) -> String {
    let grid = fault();
    let mut draws = GenslipLcg::new(seed).realisation(realisation);
    let mut fft = FftwFft::new();
    let mut solver = Wavefront2d::new();

    let model = generate(
        &mut draws,
        &mut fft,
        &mut solver,
        &grid,
        &velocity_model(),
        source_spec(),
        slip_spec(),
        timing_spec(),
        Hypocentre { strike: 12, dip: 8 },
    );

    let peak = |values: &[f32]| values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    #[expect(clippy::cast_precision_loss, reason = "small subfault counts")]
    let mean = |values: &[f32]| values.iter().sum::<f32>() / values.len() as f32;

    let onset = model.onset_s.as_slice();
    let total_samples: usize = model
        .slip_rate
        .iter()
        .map(genslip::slip_rate::SlipRate::len)
        .sum();
    let silent = model
        .slip_rate
        .iter()
        .filter(|pulse| pulse.is_empty())
        .count();

    format!(
        "moment      {:.6e}\n\
         alpha_t     {:.7}\n\
         slip mean   {:.5}\n\
         slip max    {:.5}\n\
         rake mean   {:.5}\n\
         rake max    {:.5}\n\
         onset max   {:.7}\n\
         rise mean   {:.7}\n\
         rise max    {:.7}\n\
         stf samples {total_samples}\n\
         stf silent  {silent}\n\
         final seed  {}\n",
        model.moment_dyne_cm,
        model.alpha_t,
        model.slip.average_cm,
        model.slip.maximum_cm,
        mean(model.rake_deg.as_slice()),
        peak(model.rake_deg.as_slice()),
        onset.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        mean(model.rise_time_s.as_slice()),
        peak(model.rise_time_s.as_slice()),
        genslip::rng::GenslipLcg::seed(draws),
    )
}

const SNAPSHOT: &str = include_str!("snapshots/realisation.txt");

#[test]
fn the_whole_model_matches_its_snapshot() {
    let produced = summarise(20_260_807, 0);

    if std::env::var_os("UPDATE_SNAPSHOT").is_some() {
        std::fs::write(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/snapshots/realisation.txt"
            ),
            &produced,
        )
        .expect("the snapshot directory exists");
        return;
    }

    assert_eq!(
        produced, SNAPSHOT,
        "\nthe model moved. If that was intended, re-record with UPDATE_SNAPSHOT=1 \
         and say why in the commit message.\n"
    );
}

#[test]
fn a_seed_reproduces_a_model() {
    assert_eq!(summarise(4242, 3), summarise(4242, 3));
}

#[test]
fn realisations_and_seeds_both_change_the_model() {
    let base = summarise(4242, 0);
    assert_ne!(
        base,
        summarise(4242, 1),
        "realisation index changed nothing"
    );
    assert_ne!(base, summarise(4243, 0), "seed changed nothing");
}

#[test]
fn the_moment_survives_the_pipeline() {
    // The one physical invariant that should hold end to end: the scaled slip field
    // carries the moment the magnitude implies. Held loosely, because the moment sum
    // is a single-precision fold over every subfault.
    let grid = fault();
    let mut draws = GenslipLcg::new(909);
    let mut fft = FftwFft::new();
    let mut solver = Wavefront2d::new();
    let model = generate(
        &mut draws,
        &mut fft,
        &mut solver,
        &grid,
        &velocity_model(),
        source_spec(),
        slip_spec(),
        timing_spec(),
        Hypocentre { strike: 12, dip: 8 },
    );

    let expected = genslip::source::seismic_moment(source_spec().magnitude, MagnitudeScale::Moment);
    assert!(
        (model.moment_dyne_cm - expected).abs() < expected * 1e-6,
        "moment {} vs {expected}",
        model.moment_dyne_cm
    );
    assert!(model.slip.average_cm > 0.0, "average slip is not positive");
}

// Deliberately not asserted:
//
// - The model itself, sample by sample. It is 10^4 numbers and a diff of those says
//   only "something moved", which is the least useful thing a gate can say.
// - Anything against the C binary. That is the Stage 0 corpus's job, and it compares
//   parsed SRF fields rather than these summaries.
