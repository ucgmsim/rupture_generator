//! Link single-precision FFTW, when the `fftw` feature asks for it.
//!
//! **Temporary, and Stage 1 only.** The port reproduces genslip bit for bit, and
//! genslip's transforms are FFTW's, so the only way to match them exactly is to call
//! the same library. Stage 3 turns the feature off by default and then deletes it —
//! that swap is the single biggest deletion in the plan and it is what removes the
//! last C dependency from the runtime.
//!
//! `cargo build --no-default-features` already builds and passes the FFT contract
//! tests on `rustfft` alone, so the endpoint is reachable today rather than hoped for.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_FFTW");

    if std::env::var_os("CARGO_FEATURE_FFTW").is_none() {
        return;
    }

    println!("cargo:rustc-link-lib=dylib=fftw3f");
    // FFTW's planner is process-global mutable state and aborts if two threads enter
    // it at once. `fftwf_make_planner_thread_safe` lives in this companion library
    // and installs a lock inside FFTW itself, which is why it also covers the
    // planner calls the C oracle makes -- both link the same libfftw3f.
    println!("cargo:rustc-link-lib=dylib=fftw3f_threads");
}
