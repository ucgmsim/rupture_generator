//! Link the genslip v5.6.2 static libraries so their functions can be called directly.
//!
//! The oracle is a function call, not a fixture. `Genslip/v5.6.2/CMakeLists.txt`
//! already builds the physics as `genrandv5.6`, separate from `main()`, so there is
//! nothing to extract: point this at an EMOD3D build tree and every prototype in
//! `function.h` becomes callable from a Rust test.
//!
//! Set `EMOD3D_BUILD_DIR` to an `EMOD3D` `CMake` build directory. The default assumes a
//! sibling checkout.
//!
//! # Compiler flags are part of the contract
//!
//! `EMOD3D` must be built without fast-math and without FP contraction, or the oracle
//! is not a stable reference:
//!
//! ```sh
//! cmake -B build -DCMAKE_C_FLAGS="-O0 -fno-fast-math -ffp-contract=off" \
//!                -DCMAKE_Fortran_FLAGS="-O0 -fno-fast-math -ffp-contract=off"
//! ```
//!
//! At `-O2` the compiler fuses multiply-adds and reassociates, so the C moves under
//! us and a bit-equality failure stops meaning what we want it to mean.

use std::path::{Path, PathBuf};

/// Libraries to link, in dependency order. `genrandv5.6` references symbols in
/// `srfv5.6` and `get`, so it must come first.
const STATIC_LIBRARIES: [(&str, &str); 3] = [
    ("genrandv5.6", "Genslip/v5.6.2"),
    ("srfv5.6", "Genslip/v5.6.2"),
    ("get", "Getpar"),
];

fn default_build_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../EMOD3D/build")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from("../../../EMOD3D/build"))
}

fn main() {
    println!("cargo:rerun-if-env-changed=EMOD3D_BUILD_DIR");
    println!("cargo:rerun-if-changed=build.rs");

    let build_dir =
        std::env::var_os("EMOD3D_BUILD_DIR").map_or_else(default_build_dir, PathBuf::from);

    for (library, subdirectory) in STATIC_LIBRARIES {
        let directory = build_dir.join(subdirectory);
        let archive = directory.join(format!("lib{library}.a"));

        assert!(
            archive.exists(),
            "genslip-oracle: {} not found.\n\
             Set EMOD3D_BUILD_DIR to an EMOD3D CMake build directory (currently {}), \
             and build it with -O0 -fno-fast-math -ffp-contract=off.",
            archive.display(),
            build_dir.display(),
        );

        println!("cargo:rustc-link-search=native={}", directory.display());
        println!("cargo:rustc-link-lib=static={library}");
        println!("cargo:rerun-if-changed={}", archive.display());
    }

    // genslip's FFTs are single-precision FFTW, and wafront2d.f/fourg.f drag in the
    // gfortran runtime. Both are dynamic; only the genslip objects are vendored.
    println!("cargo:rustc-link-lib=dylib=fftw3f");
    println!("cargo:rustc-link-lib=dylib=gfortran");
    println!("cargo:rustc-link-lib=dylib=m");

    emit_gfortran_search_path();
}

/// gfortran's runtime is not on the default linker path on every distribution.
fn emit_gfortran_search_path() {
    let Ok(output) = std::process::Command::new("gfortran")
        .arg("-print-file-name=libgfortran.so")
        .output()
    else {
        return;
    };

    let path = String::from_utf8_lossy(&output.stdout);
    let path = Path::new(path.trim());
    if let Some(directory) = path.parent()
        && path.is_absolute()
    {
        println!("cargo:rustc-link-search=native={}", directory.display());
    }
}
