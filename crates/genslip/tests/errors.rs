//! What the crate refuses, and what it says when it does.
//!
//! Every one of these was a panic or a silent fallback before. The panics were only
//! half wrong — a message is a message — but a caller could not catch one, and the
//! Python boundary turned them into a `pyo3_runtime.PanicException` with a Rust
//! backtrace attached rather than into a `ValueError` naming the argument.
//!
//! The silent fallback was wholly wrong, and it has its own test in
//! `source_parity.rs`: a dip of 120° used to produce `alpha_t = 1.0`, which is
//! *exactly* what a vertical strike-slip fault gets. Not an error value, not a NaN —
//! the calibration point. See `error.rs`.
//!
//! # What is asserted here
//!
//! That the boundary refuses, and that it refuses with the *right variant*. Matching
//! on the variant rather than on the message means the wording can be improved
//! without touching a test, and means a check that fires for the wrong reason — a
//! shape error where an extent error belongs — is caught rather than passing on the
//! strength of having failed at all.

mod common;

use common::fixture;
use genslip::Error;
use genslip::fft::RustFft;
use genslip::realisation::{PointSourceSpec, generate, point_source};
use genslip::rng::GenslipLcg;
use genslip::rupture::{FactoredSweep, Hypocentre};
use genslip::source::{MagnitudeScale, geometry_correction};

/// Run the finite-fault generator on a fixture with one thing changed.
fn generated(
    grid: &genslip::realisation::FaultGrid,
    hypocentre: Hypocentre,
) -> genslip::Result<genslip::realisation::RuptureModel> {
    generate(
        &mut GenslipLcg::new(fixture::SEED),
        &mut RustFft::new(),
        &mut FactoredSweep::new(),
        grid,
        &fixture::velocity_model(),
        fixture::source_spec(),
        fixture::slip_spec(),
        &fixture::timing_spec(),
        hypocentre,
    )
}

fn a_point_source(
    grid: &genslip::realisation::FaultGrid,
    rise_time_s: f64,
) -> genslip::Result<genslip::realisation::RuptureModel> {
    point_source(
        &mut FactoredSweep::new(),
        grid,
        &fixture::velocity_model(),
        PointSourceSpec {
            magnitude: 5.2,
            magnitude_scale: MagnitudeScale::Moment,
            average_dip_deg: 60.0,
            average_rake_deg: 175.0,
            rise_time_s,
        },
        &fixture::timing_spec(),
        Hypocentre { strike: 0, dip: 0 },
    )
}

/// The error a call was refused with.
///
/// `Result<RuptureModel, _>` cannot be compared directly and should not be —
/// `RuptureModel` deriving `PartialEq` so that a test can say "this is an error"
/// would be the tail wagging the dog. This says the intended thing instead: the call
/// was refused, and here is what with.
#[track_caller]
fn refusal<T>(result: genslip::Result<T>) -> Error {
    match result {
        Ok(_) => panic!("expected a refusal, got a rupture"),
        Err(error) => error,
    }
}

/// The fixture itself is accepted, or nothing below means anything.
///
/// First, deliberately: a test file full of refusals proves nothing if the thing
/// being perturbed was already being refused.
#[test]
fn the_unmodified_fixture_is_accepted() {
    assert!(generated(&fixture::fault(), fixture::hypocentre()).is_ok());
    assert!(a_point_source(&fixture::fault(), 0.35).is_ok());
}

#[test]
fn a_hypocentre_off_the_fault_is_refused() {
    let grid = fixture::fault();
    let (strike_count, dip_count) = (grid.extents.fault_strike, grid.extents.fault_dip);

    for hypocentre in [
        Hypocentre {
            strike: strike_count,
            dip: 0,
        },
        Hypocentre {
            strike: 0,
            dip: dip_count,
        },
    ] {
        assert_eq!(
            refusal(generated(&fixture::fault(), hypocentre)),
            Error::HypocentreOffFault {
                strike: hypocentre.strike,
                dip: hypocentre.dip,
                strike_count,
                dip_count,
            }
        );
    }

    // The last subfault is on the fault. An off-by-one in the check would refuse it.
    assert!(
        generated(
            &fixture::fault(),
            Hypocentre {
                strike: strike_count - 1,
                dip: dip_count - 1
            }
        )
        .is_ok()
    );
}

/// The check that moved. Both entry points reach the same one.
///
/// It used to live in `crates/core`, written out twice, so a Rust caller got no check
/// at all and the two Python copies were one edit away from disagreeing with each
/// other and with the solver's own assertion.
#[test]
fn the_point_source_path_refuses_the_same_hypocentre() {
    let grid = fixture::fault();
    let result = point_source(
        &mut FactoredSweep::new(),
        &grid,
        &fixture::velocity_model(),
        PointSourceSpec {
            magnitude: 5.2,
            magnitude_scale: MagnitudeScale::Moment,
            average_dip_deg: 60.0,
            average_rake_deg: 175.0,
            rise_time_s: 0.35,
        },
        &fixture::timing_spec(),
        Hypocentre {
            strike: 999,
            dip: 0,
        },
    );
    assert!(matches!(refusal(result), Error::HypocentreOffFault { .. }));
}

#[test]
fn a_per_subfault_array_of_the_wrong_length_names_itself() {
    let mut grid = fixture::fault();
    grid.base_rake_deg.pop();
    let expected = fixture::STRIKE_COUNT * fixture::DIP_COUNT;

    assert_eq!(
        refusal(generated(&grid, fixture::hypocentre())),
        Error::Shape {
            what: "base_rake_deg",
            found: expected - 1,
            expected,
        }
    );
}

#[test]
fn a_depth_per_dip_row_is_required() {
    let mut grid = fixture::fault();
    grid.depth_km.push(40.0);

    assert_eq!(
        refusal(generated(&grid, fixture::hypocentre())),
        Error::Shape {
            what: "depth_km",
            found: fixture::DIP_COUNT + 1,
            expected: fixture::DIP_COUNT,
        }
    );
}

#[test]
fn an_odd_padded_extent_is_refused() {
    let mut grid = fixture::fault();
    grid.extents.padded_dip += 1;

    assert_eq!(
        refusal(generated(&grid, fixture::hypocentre())),
        Error::Extent {
            what: "padded",
            strike_count: fixture::PADDED_STRIKE,
            dip_count: fixture::PADDED_DIP + 1,
            constraint: "even",
        }
    );
}

#[test]
fn a_fault_larger_than_its_padding_is_refused() {
    let mut grid = fixture::fault();
    grid.extents.padded_strike = 2;

    assert!(matches!(
        refusal(generated(&grid, fixture::hypocentre())),
        Error::FaultLargerThanPadding { .. }
    ));
}

/// Zero and negative are equally wrong, and so is NaN.
///
/// NaN is the one worth having: it does not compare greater than zero, so a naive
/// `> 0.0` guard lets it through, and a NaN rise time produces a pulse of NaNs
/// rather than a complaint. `Error::require_positive` tests finiteness for that
/// reason.
#[test]
fn a_rise_time_that_is_not_positive_is_refused() {
    let grid = fixture::fault();
    for rise_time_s in [0.0_f64, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                refusal(a_point_source(&grid, rise_time_s)),
                Error::NotPositive {
                    what: "rise_time_s",
                    ..
                }
            ),
            "a rise time of {rise_time_s} was accepted"
        );
    }
}

/// Both ends of the dip range are *on* the fault plane, not outside it.
#[test]
fn the_dip_range_is_inclusive_at_both_ends() {
    assert!(geometry_correction(0.0, 90.0).is_ok());
    assert!(geometry_correction(90.0, 90.0).is_ok());
    assert!(geometry_correction(-0.001, 90.0).is_err());
    assert!(geometry_correction(90.001, 90.0).is_err());
}

/// The message names the input and the constraint, so a caller can act on it.
///
/// Asserted because the whole argument for an error type over a panic is that the
/// message reaches someone who can fix it. A variant whose `Display` said
/// "invalid input" would satisfy every other test in this file.
#[test]
fn every_message_names_what_was_wrong() {
    for (error, wanted) in [
        (
            Error::DipOutOfRange { degrees: 120.0 },
            vec!["dip", "120", "0..=90"],
        ),
        (
            Error::NotPositive {
                what: "rise_time_s",
                value: -1.0,
            },
            vec!["rise_time_s", "-1", "positive"],
        ),
        (
            Error::Shape {
                what: "base_rake_deg",
                found: 3,
                expected: 4,
            },
            vec!["base_rake_deg", "3", "4"],
        ),
        (
            Error::HypocentreOffFault {
                strike: 9,
                dip: 1,
                strike_count: 5,
                dip_count: 3,
            },
            vec!["hypocentre", "9", "5x3"],
        ),
    ] {
        let message = error.to_string();
        for fragment in wanted {
            assert!(
                message.contains(fragment),
                "{error:?} says {message:?}, which does not mention {fragment:?}"
            );
        }
    }
}

// Deliberately not asserted:
//
// - That the internal assertions fire. They are invariants this crate is responsible
//   for, established at the boundary above — a test for one would be a test that the
//   boundary check it duplicates does not work, which is what `error.rs` says the
//   split is for.
// - Exact message wording. The variant is the contract; the wording is prose, and
//   pinning it would make improving a message a test change.
