//! genslip's wavenumber-domain field generators (`Genslip/v5.6.2/slip.c`).
//!
//! These take the grid as a bare `struct complex *` with the extents passed
//! alongside, so every wrapper here has a length contract the compiler cannot see.
//! Each one checks it and panics rather than trusting the caller: an oracle that can
//! be made to read out of bounds is worse than no oracle, because the comparison it
//! produces is meaningless rather than absent.

use crate::Complex;

unsafe extern "C" {
    /// `void kfilt_gaus2(struct complex *s0, int nx0, int ny0, float *dkx,
    /// float *dky, float *xl, float *yl, long *seed, int kflag,
    /// float *lambda_max, float *lambda_min)`
    fn kfilt_gaus2(
        s0: *mut Complex,
        nx0: core::ffi::c_int,
        ny0: core::ffi::c_int,
        dkx: *mut core::ffi::c_float,
        dky: *mut core::ffi::c_float,
        xl: *mut core::ffi::c_float,
        yl: *mut core::ffi::c_float,
        seed: *mut core::ffi::c_long,
        kflag: core::ffi::c_int,
        lambda_max: *mut core::ffi::c_float,
        lambda_min: *mut core::ffi::c_float,
    );

    /// `void kfilt_beta2(struct complex *s0, int nx0, int ny0, float *dkx,
    /// float *dky, float *hcoef, float *lambda_max, float *lambda_min, long *seed)`
    fn kfilt_beta2(
        s0: *mut Complex,
        nx0: core::ffi::c_int,
        ny0: core::ffi::c_int,
        dkx: *mut core::ffi::c_float,
        dky: *mut core::ffi::c_float,
        hcoef: *mut core::ffi::c_float,
        lambda_max: *mut core::ffi::c_float,
        lambda_min: *mut core::ffi::c_float,
        seed: *mut core::ffi::c_long,
    );

    /// `void hermit(struct complex *s0, int nx0, int ny0)`
    fn hermit(s0: *mut Complex, nx0: core::ffi::c_int, ny0: core::ffi::c_int);
}

/// Check that `grid` really holds `strike_count * dip_count` points.
fn check_extent(grid: &[Complex], strike_count: usize, dip_count: usize) {
    assert_eq!(
        grid.len(),
        strike_count * dip_count,
        "grid holds {} points but was described as {strike_count}x{dip_count}",
        grid.len(),
    );
    assert!(
        i32::try_from(strike_count).is_ok() && i32::try_from(dip_count).is_ok(),
        "extents must fit in a C int"
    );
}

/// Extents as C ints, after [`check_extent`] has passed.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    reason = "check_extent has already proved both fit in an i32"
)]
const fn extents(strike_count: usize, dip_count: usize) -> (i32, i32) {
    (strike_count as i32, dip_count as i32)
}

/// `kfilt_gaus2`: a correlated field with a corner set by the correlation lengths.
///
/// The amplitude scale is read from `grid[0]`'s magnitude on entry, so the caller
/// must seed it there — the C has no separate argument for it.
#[expect(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub fn correlated_field(
    grid: &mut [Complex],
    strike_count: usize,
    dip_count: usize,
    strike_step: f32,
    dip_step: f32,
    strike_correlation: f32,
    dip_correlation: f32,
    seed: &mut i64,
    kflag: i32,
    max_wavelength: f32,
    min_wavelength: f32,
) {
    check_extent(grid, strike_count, dip_count);
    let (nx0, ny0) = extents(strike_count, dip_count);

    let mut dkx = strike_step;
    let mut dky = dip_step;
    let mut xl = strike_correlation;
    let mut yl = dip_correlation;
    let mut lambda_max = max_wavelength;
    let mut lambda_min = min_wavelength;

    // SAFETY: `grid` is uniquely borrowed and check_extent has proved it holds
    // exactly nx0*ny0 elements, which is what the C indexes. Every other pointer
    // addresses a live local of the matching type.
    unsafe {
        kfilt_gaus2(
            grid.as_mut_ptr(),
            nx0,
            ny0,
            &raw mut dkx,
            &raw mut dky,
            &raw mut xl,
            &raw mut yl,
            std::ptr::from_mut(seed),
            kflag,
            &raw mut lambda_max,
            &raw mut lambda_min,
        );
    }
}

/// `kfilt_beta2`: a self-affine field, pure power law.
#[expect(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub fn self_affine_field(
    grid: &mut [Complex],
    strike_count: usize,
    dip_count: usize,
    strike_step: f32,
    dip_step: f32,
    hurst_exponent: f32,
    max_wavelength: f32,
    min_wavelength: f32,
    seed: &mut i64,
) {
    check_extent(grid, strike_count, dip_count);
    let (nx0, ny0) = extents(strike_count, dip_count);

    let mut dkx = strike_step;
    let mut dky = dip_step;
    let mut hcoef = hurst_exponent;
    let mut lambda_max = max_wavelength;
    let mut lambda_min = min_wavelength;

    // SAFETY: as above.
    unsafe {
        kfilt_beta2(
            grid.as_mut_ptr(),
            nx0,
            ny0,
            &raw mut dkx,
            &raw mut dky,
            &raw mut hcoef,
            &raw mut lambda_max,
            &raw mut lambda_min,
            std::ptr::from_mut(seed),
        );
    }
}

/// `hermit`: reflect the non-negative half into a Hermitian-symmetric whole.
pub fn impose_hermitian_symmetry(grid: &mut [Complex], strike_count: usize, dip_count: usize) {
    check_extent(grid, strike_count, dip_count);
    let (nx0, ny0) = extents(strike_count, dip_count);

    // SAFETY: as above.
    unsafe { hermit(grid.as_mut_ptr(), nx0, ny0) }
}

unsafe extern "C" {
    /// `void kfilter(struct complex *s0, int nx0, int ny0, float *dkx, float *dky,
    /// int ord, float *lambda_max, float *lambda_min)`
    fn kfilter(
        s0: *mut Complex,
        nx0: core::ffi::c_int,
        ny0: core::ffi::c_int,
        dkx: *mut core::ffi::c_float,
        dky: *mut core::ffi::c_float,
        ord: core::ffi::c_int,
        lambda_max: *mut core::ffi::c_float,
        lambda_min: *mut core::ffi::c_float,
    );

    /// `void get_mean_sigma_c(struct complex *x, int ns, int nd, float *avg,
    /// float *sig)`
    fn get_mean_sigma_c(
        x: *mut Complex,
        ns: core::ffi::c_int,
        nd: core::ffi::c_int,
        avg: *mut core::ffi::c_float,
        sig: *mut core::ffi::c_float,
    );
}

/// `kfilter`: band-pass an existing field at a caller-chosen order.
#[expect(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub fn band_pass(
    grid: &mut [Complex],
    strike_count: usize,
    dip_count: usize,
    strike_step: f32,
    dip_step: f32,
    order: i32,
    max_wavelength: f32,
    min_wavelength: f32,
) {
    check_extent(grid, strike_count, dip_count);
    let (nx0, ny0) = extents(strike_count, dip_count);

    let mut dkx = strike_step;
    let mut dky = dip_step;
    let mut lambda_max = max_wavelength;
    let mut lambda_min = min_wavelength;

    // SAFETY: as above -- uniquely borrowed, length checked against the extents.
    unsafe {
        kfilter(
            grid.as_mut_ptr(),
            nx0,
            ny0,
            &raw mut dkx,
            &raw mut dky,
            order,
            &raw mut lambda_max,
            &raw mut lambda_min,
        );
    }
}

/// `get_mean_sigma_c`: mean and population standard deviation of the real part.
///
/// Takes the grid by mutable reference only because the C signature is non-const;
/// it does not write.
pub fn mean_and_sigma(grid: &mut [Complex], strike_count: usize, dip_count: usize) -> (f32, f32) {
    check_extent(grid, strike_count, dip_count);
    let (ns, nd) = extents(strike_count, dip_count);

    let mut mean = 0.0_f32;
    let mut sigma = 0.0_f32;

    // SAFETY: as above; `avg` and `sig` address live locals.
    unsafe {
        get_mean_sigma_c(grid.as_mut_ptr(), ns, nd, &raw mut mean, &raw mut sigma);
    }

    (mean, sigma)
}
