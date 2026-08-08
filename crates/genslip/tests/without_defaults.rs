//! `--no-default-features` generates a rupture, rather than merely compiling.
//!
//! Until the Fortran eikonal solver went, this configuration had no `EikonalSolver`
//! at all: `realisation::generate` was unreachable from the crate's own types, so the
//! Stage 3 endpoint the gate checked continuously was a compile check and nothing
//! more. That is the gap this closes.
#![cfg(not(feature = "fftw"))]

mod common;
use common::fixture;

#[test]
fn a_rupture_is_generated_without_fftw_or_a_fortran_solver() {
    // The perturbation off, so the earliest subfault IS the source: onset is then the
    // solver's output rather than the solver's output plus a slip-correlated field,
    // and the registration claim below means something.
    let mut timing = fixture::timing_spec();
    timing.rupture_time_scale = 0.0;

    let model = genslip::realisation::generate(
        &mut genslip::rng::GenslipLcg::new(fixture::SEED),
        &mut genslip::fft::RustFft::new(),
        &mut genslip::rupture::FactoredSweep::new(),
        &fixture::fault(),
        &fixture::velocity_model(),
        fixture::source_spec(),
        fixture::slip_spec(),
        timing,
        fixture::hypocentre(),
    );

    assert!(model.slip.average_cm > 0.0, "no slip");
    assert!(model.moment_dyne_cm > 0.0, "no moment");
    assert!(
        model.slip_rate.iter().any(|pulse| !pulse.is_empty()),
        "no subfault emitted a pulse"
    );

    let hypocentre = fixture::hypocentre();
    assert!(
        model.onset_s.time(hypocentre.strike, hypocentre.dip) < 1e-9,
        "the rupture does not start at the hypocentre"
    );
}
