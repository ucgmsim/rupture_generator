//! genslip's random number generator (`Genslip/v5.6.2/misc.c:48`).
//!
//! One 31-bit truncated linear congruential generator drives every field genslip
//! produces, with the state carried explicitly in a `long` that callers thread
//! through by pointer. The whole reproducibility contract of the program rests on
//! it, which is why it is the first thing ported and the first thing pinned.
//!
//! Two details are load-bearing and easy to get wrong:
//!
//! * **The multiply happens in `long`.** On LP64 Linux that is 64-bit, so
//!   `state * 1103515245` does *not* wrap at 32 bits before the mask is applied.
//!   Reproducing this with `i32` gives a different stream.
//! * **A Gaussian costs exactly 12 uniforms.** `gaus_rand` is an Irwin-Hall sum of
//!   twelve uniforms, not Box-Muller and not a ziggurat. Substituting a better
//!   generator changes the draw count and desynchronises everything downstream.

unsafe extern "C" {
    /// `double sfrand(long *seed)` — one uniform on `[-1, 1)`, advancing `seed`.
    fn sfrand(seed: *mut core::ffi::c_long) -> f64;

    /// `double gaus_rand(float *sigma, float *mean, long *seed)` — one Gaussian,
    /// consuming twelve uniforms.
    fn gaus_rand(
        sigma: *mut core::ffi::c_float,
        mean: *mut core::ffi::c_float,
        seed: *mut core::ffi::c_long,
    ) -> f64;
}

/// Draw one uniform on `[-1, 1)`, advancing `seed` in place.
///
/// Safe because `seed` is an ordinary `&mut` to a scalar: there is no length
/// contract and no aliasing for the C to violate.
pub fn uniform(seed: &mut i64) -> f64 {
    // SAFETY: `sfrand` reads and writes exactly one `long` through this pointer,
    // which `&mut i64` guarantees is valid, aligned and uniquely borrowed.
    unsafe { sfrand(std::ptr::from_mut(seed)) }
}

/// Draw one Gaussian with the given standard deviation and mean, advancing `seed`
/// by twelve uniforms.
///
/// `sigma` and `mean` are `float` in the C and taken by pointer; they are read, not
/// written, but the signature is non-const so they are passed as mutable copies.
pub fn gaussian(sigma: f32, mean: f32, seed: &mut i64) -> f64 {
    let mut sigma = sigma;
    let mut mean = mean;
    // SAFETY: all three pointers address live, uniquely-borrowed locals of the
    // matching C types. `gaus_rand` mutates only `seed`.
    unsafe {
        gaus_rand(
            std::ptr::from_mut(&mut sigma),
            std::ptr::from_mut(&mut mean),
            std::ptr::from_mut(seed),
        )
    }
}

/// Draw `count` uniforms, returning them and leaving `seed` advanced.
pub fn uniforms(seed: &mut i64, count: usize) -> Vec<f64> {
    (0..count).map(|_| uniform(seed)).collect()
}
