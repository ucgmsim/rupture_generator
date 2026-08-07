//! Link the Stage 1 compatibility backends, when their features ask for them.
//!
//! Both are **temporary**. The port reproduces genslip bit for bit, which means
//! calling the same code it calls; each feature exists so that comparison is
//! possible and each is deleted once the scientific suite is the gate.
//!
//! | feature | links | replaced by |
//! | --- | --- | --- |
//! | `fftw` | system FFTW | `fft::RustFft` |
//! | `wavefront-compat` | genslip's Fortran eikonal solver, from an EMOD3D build | a fast-marching solver |
//!
//! `cargo build --no-default-features` builds with neither, which is the Stage 3
//! endpoint — checked continuously by `gate.sh` so it cannot rot.
//!
//! `wavefront-compat` needs `EMOD3D_BUILD_DIR` to point at an `EMOD3D` `CMake` build
//! tree, built without fast-math and without FP contraction. See
//! `genslip-oracle/build.rs`, which says the same thing at greater length.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_FFTW");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_WAVEFRONT_COMPAT");
    println!("cargo:rerun-if-env-changed=EMOD3D_BUILD_DIR");

    if std::env::var_os("CARGO_FEATURE_FFTW").is_some() {
        println!("cargo:rustc-link-lib=dylib=fftw3f");
        // FFTW's planner is process-global mutable state and aborts if two threads
        // enter it at once. `fftwf_make_planner_thread_safe` lives in this companion
        // library and installs a lock inside FFTW itself, which is why it also covers
        // the planner calls the C oracle makes -- both link the same libfftw3f.
        println!("cargo:rustc-link-lib=dylib=fftw3f_threads");
    }

    if std::env::var_os("CARGO_FEATURE_WAVEFRONT_COMPAT").is_some() {
        link_genslip_fortran();
    }
}

/// Link genslip's static libraries, for `wfront2d_`.
///
/// The whole archive comes along because `genrandv5.6` is one library; nothing else
/// in it is referenced.
fn link_genslip_fortran() {
    let build_dir = std::env::var_os("EMOD3D_BUILD_DIR").map_or_else(
        || {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../EMOD3D/build")
                .canonicalize()
                .unwrap_or_else(|_| PathBuf::from("../../../EMOD3D/build"))
        },
        PathBuf::from,
    );

    let genslip = build_dir.join("Genslip/v5.6.2");
    assert!(
        genslip.join("libgenrandv5.6.a").exists(),
        "genslip: the `wavefront-compat` feature needs an EMOD3D build, and \n\
         {} does not hold one.\n\
         Set EMOD3D_BUILD_DIR, or build without default features.",
        genslip.display(),
    );

    println!("cargo:rustc-link-search=native={}", genslip.display());
    println!(
        "cargo:rustc-link-search=native={}",
        build_dir.join("Getpar").display()
    );
    println!("cargo:rustc-link-lib=static=genrandv5.6");
    println!("cargo:rustc-link-lib=static=srfv5.6");
    println!("cargo:rustc-link-lib=static=get");
    println!("cargo:rustc-link-lib=dylib=gfortran");
    println!("cargo:rustc-link-lib=dylib=m");
}
