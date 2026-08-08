//! Link FFTW, when the `fftw` feature asks for it.
//!
//! **Temporary.** The port reproduced genslip bit for bit, which meant calling the
//! same transform it calls; the feature exists so that comparison was possible and it
//! is deleted once `fft::RustFft` becomes the only engine. The measured divergence
//! between the two is 7.06e-8 relative, recorded in `tests/fft_contract.rs` before the
//! swap so it can adjudicate it.
//!
//! `cargo build --no-default-features` builds without it, which is the Stage 3
//! endpoint — checked continuously by `gate.sh` so it cannot rot. Since the Fortran
//! eikonal solver went, that configuration generates ruptures rather than merely
//! compiling, and the whole suite runs in it.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_FFTW");

    if std::env::var_os("CARGO_FEATURE_FFTW").is_some() {
        println!("cargo:rustc-link-lib=dylib=fftw3f");
        // FFTW's planner is process-global mutable state and aborts if two threads
        // enter it at once. `fftwf_make_planner_thread_safe` lives in this companion
        // library and installs a lock inside FFTW itself, which is why it also covers
        // the planner calls the C oracle makes -- both link the same libfftw3f.
        println!("cargo:rustc-link-lib=dylib=fftw3f_threads");
    }
}
