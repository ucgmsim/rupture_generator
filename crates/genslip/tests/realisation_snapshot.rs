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

mod common;

use common::fixture;
use genslip::fft::FftwFft;
use genslip::realisation::generate;
use genslip::rng::{GenslipLcg, Realisations as _};
use genslip::rupture::Wavefront2d;
use genslip::source::MagnitudeScale;

/// Summary statistics, formatted so a diff points at which one moved.
fn summarise(seed: i64, realisation: u64) -> String {
    let mut draws = GenslipLcg::new(seed).realisation(realisation);
    let mut fft = FftwFft::new();
    let mut solver = Wavefront2d::new();

    let model = generate(
        &mut draws,
        &mut fft,
        &mut solver,
        &fixture::fault(),
        &fixture::velocity_model(),
        fixture::source_spec(),
        fixture::slip_spec(),
        fixture::timing_spec(),
        fixture::hypocentre(),
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
    let produced = summarise(fixture::SEED, 0);

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
    let mut draws = GenslipLcg::new(909);
    let mut fft = FftwFft::new();
    let mut solver = Wavefront2d::new();
    let model = generate(
        &mut draws,
        &mut fft,
        &mut solver,
        &fixture::fault(),
        &fixture::velocity_model(),
        fixture::source_spec(),
        fixture::slip_spec(),
        fixture::timing_spec(),
        fixture::hypocentre(),
    );

    let expected =
        genslip::source::seismic_moment(fixture::source_spec().magnitude, MagnitudeScale::Moment);
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
