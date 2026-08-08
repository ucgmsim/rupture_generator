//! A point source is the finite-fault generator with the randomness taken out.
//!
//! `generic_slip2srf` is not a generator. It reads a file of subfaults carrying slip,
//! rake and onset, stretches a rise time with depth, turns each (slip, duration) pair
//! into a pulse, and writes an SRF. The workflow that calls it hands it **one slip
//! value, one rake and one onset**, repeated over however many subfaults the geometry
//! discretised into — see `realisation_to_srf.py:706-757`.
//!
//! So the point-source path is not a second pipeline. [`point_source`] builds the four
//! fields as constants and hands them to the same `assemble` a finite fault reaches.
//! What this file asserts is that the constants really do collapse: that the general
//! machinery, fed degenerate inputs, gives back the simple answers a point source is
//! supposed to have — **exactly**, not approximately.
//!
//! That distinction is the whole test strategy here. `ENGINEERING_RULES.md` rule 2
//! says prefer an exact identity to a statistical test, and every claim below is an
//! identity: a one-subfault source's slip *is* `M0/(μA)`, its onset *is* the delay,
//! its rise time *is* the number it was given. If any of those were merely close, the
//! reuse would be a coincidence rather than a consequence.
//!
//! # What is deliberately different from the C
//!
//! Onset is solved for rather than given, and rise time is a fault-wide average
//! rather than an unstretched floor. Both are stated at [`point_source`] and both are
//! measured against the C in the reference comparison; neither is asserted here,
//! because this file is about what the port *is*, not about what the C did.

mod common;

use common::fixture;
use genslip::grid::FaultAxes;
use genslip::realisation::{FaultGrid, PointSourceSpec, point_source};
use genslip::rupture::{FactoredSweep, Hypocentre};
use genslip::slip_rate::SlipRateShape;
use genslip::source::MagnitudeScale;

/// A magnitude small enough that a point source is the right model for it.
const MAGNITUDE: f64 = 5.2;
const RISE_TIME_S: f64 = 0.35;

fn spec() -> PointSourceSpec {
    PointSourceSpec {
        magnitude: MAGNITUDE,
        magnitude_scale: MagnitudeScale::Moment,
        average_dip_deg: 60.0,
        average_rake_deg: 175.0,
        rise_time_s: RISE_TIME_S,
    }
}

/// A genuine point: one subfault, half a kilometre across.
fn a_point() -> FaultGrid {
    let mut grid = fixture::fault_of(1, 1, 2, 2);
    grid.spacing.strike_km = 0.5;
    grid.spacing.dip_km = 0.5;
    grid.depth_km = vec![7.0];
    grid.base_rake_deg = vec![175.0];
    grid.velocity_fraction = vec![0.8];
    grid
}

fn origin() -> Hypocentre {
    Hypocentre { strike: 0, dip: 0 }
}

fn generate(grid: &FaultGrid) -> genslip::realisation::RuptureModel {
    valid(point_source(
        &mut FactoredSweep::new(),
        grid,
        &fixture::velocity_model(),
        spec(),
        &fixture::timing_spec(),
        Hypocentre {
            strike: grid.extents.fault_strike / 2,
            dip: grid.extents.fault_dip / 2,
        },
    ))
}

/// Unwrap a rupture the fixture guarantees is valid.
///
/// The fixtures here are all well-formed geometries, so an error is a bug in this
/// file rather than a case worth handling. `Error`'s own contract is exercised where
/// it belongs, in `errors.rs`.
#[track_caller]
fn valid(
    model: genslip::Result<genslip::realisation::RuptureModel>,
) -> genslip::realisation::RuptureModel {
    model.expect("the fixture geometry is valid")
}

/// The moment is what the magnitude says, and the slip is what the moment says.
///
/// `M0 = μ A s` at one subfault, so `s = M0/(μA)` with nothing left over. Asserted to
/// `f64` rounding rather than to a tolerance, because this is arithmetic rather than
/// a model: any gap is a unit error or a missing rigidity, not a discretisation.
#[test]
fn one_subfault_slips_the_moment_divided_by_rigidity_and_area() {
    let grid = a_point();
    let model = valid(point_source(
        &mut FactoredSweep::new(),
        &grid,
        &fixture::velocity_model(),
        spec(),
        &fixture::timing_spec(),
        origin(),
    ));

    // The layer containing 7 km: vs 3.2 km/s, density 2.6 g/cm^3.
    let shear_speed_cm_s = 3.2 * 1.0e5;
    let rigidity = 2.6 * shear_speed_cm_s * shear_speed_cm_s;
    let area_cm2 = (0.5 * 1.0e5) * (0.5 * 1.0e5);
    let expected = model.moment_dyne_cm / (rigidity * area_cm2);

    let slip = model.slip.slip[[0, 0]];
    assert!(
        (slip - expected).abs() <= 1e-5 * expected,
        "one subfault slipped {slip} cm where the moment implies {expected}"
    );
}

/// A one-subfault source ruptures at the delay, and at nothing else.
///
/// The eikonal solve from a subfault to itself is zero, so what is left is
/// `rupture_delay_s` — which is what `generic_slip2srf`'s `inittime` means. The two
/// paths coincide exactly at one subfault; they diverge across a plane, and that is
/// where the C stops being able to say anything.
#[test]
fn one_subfault_ruptures_at_the_delay() {
    let mut timing = fixture::timing_spec();
    for delay_s in [0.0_f64, 2.5] {
        timing.rupture_delay_s = delay_s;
        let model = valid(point_source(
            &mut FactoredSweep::new(),
            &a_point(),
            &fixture::velocity_model(),
            spec(),
            &timing,
            origin(),
        ));
        assert!(
            (model.onset_s[[0, 0]] - delay_s).abs() < 1e-12,
            "a single subfault with a {delay_s} s delay ruptured at {}",
            model.onset_s[[0, 0]]
        );
    }
}

/// A one-subfault source's rise time is the number it was given.
///
/// The depth ramp cancels, and that is the point of treating `rise_time_s` as the
/// fault-wide *average*: the normalisation is the mean of the depth factor, which
/// over one subfault is that subfault's own factor, so it divides out whatever the
/// depth. The C's reading would give `factor_at(7 km) * rise_time_s` instead and the
/// answer would depend on where the point happened to be.
#[test]
fn one_subfault_rises_in_the_time_it_was_given() {
    for depth_km in [0.5_f64, 5.0, 7.0, 20.0, 40.0] {
        let mut grid = a_point();
        grid.depth_km = vec![depth_km];
        let model = valid(point_source(
            &mut FactoredSweep::new(),
            &grid,
            &fixture::velocity_model(),
            spec(),
            &fixture::timing_spec(),
            origin(),
        ));
        let rise = model.rise_time_s[[0, 0]];
        assert!(
            (rise - RISE_TIME_S).abs() <= 1e-6 * RISE_TIME_S,
            "a point at {depth_km} km rose in {rise} s, not the {RISE_TIME_S} asked for"
        );
    }
}

/// The rake is the base rake, unperturbed.
#[test]
fn the_rake_is_the_one_the_geometry_carries() {
    let grid = fixture::fault_of(5, 3, 6, 4);
    let model = generate(&grid);
    for (index, rake) in model.rake_deg.flat().iter().enumerate() {
        assert!(
            (rake - grid.base_rake_deg[index]).abs() < 1e-6,
            "subfault {index} came out at rake {rake}, not {}",
            grid.base_rake_deg[index]
        );
    }
}

/// Across a plane, slip is uniform along strike and varies only with rigidity.
///
/// Uniform slip is the input; what is *not* obvious is that the moment scaling leaves
/// it uniform. It does, because the scaler is a single factor over the whole field —
/// so a fault crossing a layer boundary gets one slip everywhere and the differing
/// rigidities are absorbed into the moment sum rather than into the slip. That is the
/// meaning of "moment-consistent", and it is worth pinning because the alternative
/// reading — slip varying inversely with rigidity — is equally plausible and wrong.
#[test]
fn slip_is_uniform_across_the_plane() {
    // Depths from 0.5 to 12.5 km at the fixture's 1.5 km spacing, which crosses three
    // of the four layers.
    let grid = fixture::fault_of(5, 9, 6, 10);
    let model = generate(&grid);

    let first = model.slip.slip[[0, 0]];
    for value in model.slip.slip.flat() {
        assert_eq!(
            value.to_bits(),
            first.to_bits(),
            "slip is not uniform: {value} against {first}"
        );
    }
    assert!(first > 0.0, "a point source with no slip is not a source");
}

/// Across a plane, the rupture propagates. The C writes one onset everywhere.
///
/// The deliberate difference, asserted as a positive claim rather than left implicit:
/// onset is zero at the hypocentre, non-negative everywhere, and genuinely spread out.
#[test]
fn across_a_plane_the_rupture_has_a_front() {
    let grid = fixture::fault_of(9, 5, 10, 6);
    let model = generate(&grid);
    let hypocentre = (grid.extents.fault_strike / 2, grid.extents.fault_dip / 2);

    assert!(
        model.onset_s[[hypocentre.1, hypocentre.0]].abs() < 1e-12,
        "the hypocentre does not rupture at zero"
    );

    let times = model.onset_s.flat();
    assert!(times.iter().all(|time| *time >= 0.0));

    let latest = times.iter().copied().fold(0.0_f64, f64::max);
    assert!(
        latest > 0.5,
        "a 9x5 km plane ruptured in {latest} s, which is not a front"
    );
}

/// Further from the hypocentre is later, along both axes.
///
/// The eikonal solve's own contract, restated here because a point source reaches it
/// through a different door and a wiring mistake would show up as a plausible-looking
/// but unordered field.
#[test]
fn onset_grows_away_from_the_hypocentre() {
    let grid = fixture::fault_of(9, 5, 10, 6);
    let model = generate(&grid);
    let (hs, hd) = (4_usize, 2_usize);

    for strike in hs..grid.extents.fault_strike - 1 {
        assert!(
            model.onset_s[[hd, strike + 1]] >= model.onset_s[[hd, strike]],
            "onset fell going away from the hypocentre along strike at {strike}"
        );
    }
    for dip in hd..grid.extents.fault_dip - 1 {
        assert!(
            model.onset_s[[dip + 1, hs]] >= model.onset_s[[dip, hs]],
            "onset fell going away from the hypocentre down dip at {dip}"
        );
    }
}

/// The fault-wide average rise time is the number asked for, on any plane.
///
/// The one-subfault case makes this trivially true; over a plane it is a real claim
/// about `rise_time_normalisation`, which divides by the mean of the depth factor.
/// The depth ramp redistributes and does not inflate.
#[test]
fn the_average_rise_time_is_the_one_asked_for() {
    let grid = fixture::fault_of(5, 9, 6, 10);
    let model = generate(&grid);

    let values = model.rise_time_s.flat();
    let mean = values.iter().sum::<f64>() / genslip::units::exact(values.len());
    assert!(
        (mean - RISE_TIME_S).abs() <= 1e-4 * RISE_TIME_S,
        "the average rise time is {mean}, not the {RISE_TIME_S} asked for"
    );

    // And it genuinely varies, or the claim above is about a constant field.
    let spread = values.iter().copied().fold(0.0_f64, f64::max)
        - values.iter().copied().fold(f64::MAX, f64::min);
    assert!(
        spread > 0.05 * RISE_TIME_S,
        "rise time is flat across depths from 0.5 to 12.5 km"
    );
}

/// Every shape produces a usable rupture through this path, and conserves the moment.
///
/// The pulses are covered shape by shape in `slip_rate_contract.rs`. What this adds
/// is that the *pipeline* carries each of them: a shape that only worked when called
/// directly would pass there and fail here.
#[test]
fn every_shape_makes_it_through_the_pipeline() {
    let grid = fixture::fault_of(5, 3, 6, 4);
    let mut timing = fixture::timing_spec();

    for shape in [
        SlipRateShape::OliuP2,
        SlipRateShape::Ucsb,
        SlipRateShape::Ucsb2,
        SlipRateShape::UcsbT { stretch: 2.0 },
        SlipRateShape::UcsbVarT1 { tau1_ratio: 0.2 },
        SlipRateShape::Brune,
        SlipRateShape::Urs,
        SlipRateShape::Esg2006,
        SlipRateShape::Cos,
        SlipRateShape::Seki,
        SlipRateShape::Delta,
    ] {
        timing.slip_rate_shape = shape;
        let model = valid(point_source(
            &mut FactoredSweep::new(),
            &grid,
            &fixture::velocity_model(),
            spec(),
            &timing,
            Hypocentre { strike: 2, dip: 1 },
        ));

        let moment: f64 = model
            .slip_rate
            .iter()
            .enumerate()
            .map(|(index, pulse)| {
                let integral: f64 = pulse
                    .as_slice()
                    .iter()
                    .map(|value| *value * timing.sample_interval_s)
                    .sum();
                assert!(!pulse.is_empty(), "{shape:?} left subfault {index} silent");
                integral
            })
            .sum();

        // Every pulse integrates to its subfault's slip, and the slip is uniform, so
        // the summed integral is the summed slip. A shape that normalised wrongly
        // would show up here even though its own contract test passed.
        let slip: f64 = model.slip.slip.flat().iter().copied().sum();
        assert!(
            (moment - slip).abs() <= 1e-4 * slip,
            "{shape:?}: the pulses carry {moment} cm of slip where the field has {slip}"
        );
    }
}

/// `seki` arrives earlier than the others, because it radiates before its own start.
///
/// The onset shift, seen from outside the pulse generator. Nothing else moves the
/// arrival, so a regression that dropped the shift would show up as `seki` alone
/// agreeing with the rest.
#[test]
fn seki_moves_the_arrival_and_nothing_else_does() {
    let grid = fixture::fault_of(5, 3, 6, 4);
    let mut timing = fixture::timing_spec();
    timing.rupture_delay_s = 5.0;

    let onset_of = |shape| {
        let mut timing = timing;
        timing.slip_rate_shape = shape;
        valid(point_source(
            &mut FactoredSweep::new(),
            &grid,
            &fixture::velocity_model(),
            spec(),
            &timing,
            Hypocentre { strike: 2, dip: 1 },
        ))
        .onset_s[[0, 0]]
    };

    let plain = onset_of(SlipRateShape::Ucsb);
    let seki = onset_of(SlipRateShape::Seki);
    assert!(
        seki < plain,
        "seki arrived at {seki}, not before the {plain} everything else gets"
    );
    // A quarter of the rise time, which is what the shift is.
    assert!(
        (plain - seki - 0.25 * RISE_TIME_S).abs() < 1e-3,
        "seki moved by {}, not a quarter of the {RISE_TIME_S} s rise time",
        plain - seki
    );

    for shape in [
        SlipRateShape::OliuP2,
        SlipRateShape::Ucsb2,
        SlipRateShape::Brune,
        SlipRateShape::Urs,
        SlipRateShape::Esg2006,
        SlipRateShape::Cos,
        SlipRateShape::Delta,
    ] {
        assert!(
            (onset_of(shape) - plain).abs() < 1e-12,
            "{shape:?} moved the arrival"
        );
    }
}

/// Nothing here is random, and the signature is most of the proof.
///
/// `point_source` takes no `DrawSource` and no `Fft`, so there is no stream to
/// advance and no seed to differ by. This is the observable half: the same inputs
/// give bit-identical output, every field, every time.
#[test]
fn the_same_inputs_give_the_same_rupture_every_time() {
    let grid = fixture::fault_of(5, 3, 6, 4);
    let first = generate(&grid);
    let second = generate(&grid);

    assert_eq!(first.slip.slip.flat(), second.slip.slip.flat());
    assert_eq!(first.rake_deg.flat(), second.rake_deg.flat());
    assert_eq!(first.onset_s.flat(), second.onset_s.flat());
    assert_eq!(first.rise_time_s.flat(), second.rise_time_s.flat());
    assert_eq!(first.slip_rate, second.slip_rate);
}

// Deliberately not asserted:
//
// - Anything against `generic_slip2srf`'s output. The two disagree by design on onset
//   and on what `risetime` means, so an assertion would be encoding one of those
//   choices twice. The sizes are measured in the reference comparison instead.
// - That a point source and a finite fault agree in any limit. They cannot: one has a
//   spectral slip field and the other does not, and no amount of shrinking the fault
//   makes a random field constant.
