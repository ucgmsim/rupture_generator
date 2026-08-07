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

unsafe extern "C" {
    /// `void shift_phase(struct complex *s0, int nx0, int ny0, float *dkx,
    /// float *dky, double *xshift, double *yshift)`
    fn shift_phase(
        s0: *mut Complex,
        nx0: core::ffi::c_int,
        ny0: core::ffi::c_int,
        dkx: *mut core::ffi::c_float,
        dky: *mut core::ffi::c_float,
        xshift: *mut core::ffi::c_double,
        yshift: *mut core::ffi::c_double,
    );

    /// `void taper_slip_all_r(float *sr, int nx, int ny, float *st, float *bt,
    /// float *tt)`
    fn taper_slip_all_r(
        sr: *mut core::ffi::c_float,
        nx: core::ffi::c_int,
        ny: core::ffi::c_int,
        st: *mut core::ffi::c_float,
        bt: *mut core::ffi::c_float,
        tt: *mut core::ffi::c_float,
    );
}

/// `shift_phase`: translate the field by applying a linear phase ramp.
pub fn shift_phase_of(
    grid: &mut [Complex],
    strike_count: usize,
    dip_count: usize,
    strike_step: f32,
    dip_step: f32,
    strike_shift: f64,
    dip_shift: f64,
) {
    check_extent(grid, strike_count, dip_count);
    let (nx0, ny0) = extents(strike_count, dip_count);

    let mut dkx = strike_step;
    let mut dky = dip_step;
    let mut xshift = strike_shift;
    let mut yshift = dip_shift;

    // SAFETY: as above -- uniquely borrowed, length checked against the extents.
    unsafe {
        shift_phase(
            grid.as_mut_ptr(),
            nx0,
            ny0,
            &raw mut dkx,
            &raw mut dky,
            &raw mut xshift,
            &raw mut yshift,
        );
    }
}

/// `taper_slip_all_r`: ramp slip to zero at the fault edges.
///
/// Operates on a real field, not a complex spectrum, so it takes a plain `f32`
/// slice. The extents here are the subfault counts, not the padded grid.
pub fn taper_edges(
    slip: &mut [f32],
    strike_count: usize,
    dip_count: usize,
    sides: f32,
    bottom: f32,
    top: f32,
) {
    assert_eq!(
        slip.len(),
        strike_count * dip_count,
        "slip holds {} values but was described as {strike_count}x{dip_count}",
        slip.len(),
    );
    assert!(
        i32::try_from(strike_count).is_ok() && i32::try_from(dip_count).is_ok(),
        "extents must fit in a C int"
    );
    let (nx, ny) = extents(strike_count, dip_count);

    let mut st = sides;
    let mut bt = bottom;
    let mut tt = top;

    // SAFETY: length checked against the extents; the C writes only within them.
    unsafe {
        taper_slip_all_r(
            slip.as_mut_ptr(),
            nx,
            ny,
            &raw mut st,
            &raw mut bt,
            &raw mut tt,
        );
    }
}

unsafe extern "C" {
    /// `void scale_slip_r_vsden(struct pointsource *ps, float *sr, int nstk,
    /// int ndip, int nys, float *dx, float *dy, float *dtop, float *dip,
    /// float *mom, float *savg, float *smax)`
    fn scale_slip_r_vsden(
        ps: *mut crate::PointSource,
        sr: *mut core::ffi::c_float,
        nstk: core::ffi::c_int,
        ndip: core::ffi::c_int,
        nys: core::ffi::c_int,
        dx: *mut core::ffi::c_float,
        dy: *mut core::ffi::c_float,
        dtop: *mut core::ffi::c_float,
        dip: *mut core::ffi::c_float,
        mom: *mut core::ffi::c_float,
        savg: *mut core::ffi::c_float,
        smax: *mut core::ffi::c_float,
    );
}

/// What `scale_slip_r_vsden` reports back through its out-parameters.
#[derive(Clone, Copy, Debug)]
pub struct SlipScalingResult {
    pub moment: f32,
    pub average: f32,
    pub maximum: f32,
}

/// `scale_slip_r_vsden`: scale slip to a target moment or average, writing the
/// result into `subfaults[i].slip`.
///
/// `dtop` is accepted and never read by the C; it is passed as zero.
///
/// # Panics
///
/// If `subfaults` does not hold one entry per subfault, or if the requested block
/// does not fit inside `field`.
#[expect(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub fn scale_slip(
    subfaults: &mut [crate::PointSource],
    field: &mut [f32],
    strike_count: usize,
    dip_count: usize,
    dip_offset: usize,
    strike_km: f32,
    dip_km: f32,
    dip_degrees: f32,
    moment: f32,
    average: f32,
) -> SlipScalingResult {
    assert_eq!(
        subfaults.len(),
        strike_count * dip_count,
        "got {} subfaults for a {strike_count}x{dip_count} fault",
        subfaults.len()
    );
    assert!(
        field.len() >= strike_count * (dip_count + dip_offset),
        "field holds {} values, too few for {dip_count} rows at offset {dip_offset}",
        field.len()
    );
    let (nstk, ndip) = extents(strike_count, dip_count);
    let nys = i32::try_from(dip_offset).expect("dip offset must fit in a C int");

    let mut dx = strike_km;
    let mut dy = dip_km;
    let mut dtop = 0.0_f32;
    let mut dip = dip_degrees;
    let mut mom = moment;
    let mut savg = average;
    let mut smax = 0.0_f32;

    // SAFETY: both buffers are uniquely borrowed and checked above to be at least as
    // large as the extents the C will index. Every scalar pointer addresses a live
    // local of the matching type.
    unsafe {
        scale_slip_r_vsden(
            subfaults.as_mut_ptr(),
            field.as_mut_ptr(),
            nstk,
            ndip,
            nys,
            &raw mut dx,
            &raw mut dy,
            &raw mut dtop,
            &raw mut dip,
            &raw mut mom,
            &raw mut savg,
            &raw mut smax,
        );
    }

    SlipScalingResult {
        moment: mom,
        average: savg,
        maximum: smax,
    }
}

unsafe extern "C" {
    /// `void fft2d_fftw(struct complex *xc, int n1, int n2, int isgn, float *d1,
    /// float *d2)`
    fn fft2d_fftw(
        xc: *mut Complex,
        n1: core::ffi::c_int,
        n2: core::ffi::c_int,
        isgn: core::ffi::c_int,
        d1: *mut core::ffi::c_float,
        d2: *mut core::ffi::c_float,
    );
}

/// Installs FFTW's internal planner lock, once per process.
static MAKE_PLANNER_THREAD_SAFE: std::sync::Once = std::sync::Once::new();

unsafe extern "C" {
    fn fftwf_make_planner_thread_safe();
}

/// Make FFTW's planner safe to call from more than one thread.
///
/// `fft2d_fftw` plans on **every** call, and FFTW's planner is process-global
/// mutable state that corrupts — and then segfaults or aborts — when two threads
/// enter it at once. genslip is one process doing one thing and never meets this. A
/// test binary running its tests in parallel meets it immediately, and the lock has
/// to cover *both* sides: the port installs it too, but whichever runs first must,
/// or the other races against an uninitialised planner.
fn make_planner_thread_safe() {
    // SAFETY: idempotent, and required before any planner call.
    MAKE_PLANNER_THREAD_SAFE.call_once(|| unsafe { fftwf_make_planner_thread_safe() });
}

/// `fft2d_fftw`: a 2-D transform, scaled by the product of the sample spacings.
///
/// `sign` is `-1` for the fault-to-wavenumber direction and `+1` for the reverse,
/// as the C spells it.
pub fn transform_2d(
    grid: &mut [Complex],
    strike_count: usize,
    dip_count: usize,
    sign: i32,
    first_spacing: f32,
    second_spacing: f32,
) {
    check_extent(grid, strike_count, dip_count);
    make_planner_thread_safe();
    let (n1, n2) = extents(strike_count, dip_count);

    let mut d1 = first_spacing;
    let mut d2 = second_spacing;

    // SAFETY: uniquely borrowed, length checked against the extents.
    unsafe {
        fft2d_fftw(grid.as_mut_ptr(), n1, n2, sign, &raw mut d1, &raw mut d2);
    }
}

unsafe extern "C" {
    /// `void get_rslow_stretch(float *rspd, int ns, int nd, double *rslw, int nsfd,
    /// int ndfd, int isoff, int idoff, float *tsf, long *seed)`
    fn get_rslow_stretch(
        rspd: *mut core::ffi::c_float,
        ns: core::ffi::c_int,
        nd: core::ffi::c_int,
        rslw: *mut core::ffi::c_double,
        nsfd: core::ffi::c_int,
        ndfd: core::ffi::c_int,
        isoff: core::ffi::c_int,
        idoff: core::ffi::c_int,
        tsf: *mut core::ffi::c_float,
        seed: *mut core::ffi::c_long,
    );

    /// `subroutine wfront2d(m, n, is, js, h, ns, ttime, slwns, ntot, ti, jm)`
    #[link_name = "wfront2d_"]
    fn wfront2d(
        m: *const core::ffi::c_int,
        n: *const core::ffi::c_int,
        is: *const core::ffi::c_int,
        js: *const core::ffi::c_int,
        h: *const core::ffi::c_double,
        ns: *const core::ffi::c_int,
        ttime: *mut core::ffi::c_double,
        slwns: *mut core::ffi::c_double,
        ntot: *const core::ffi::c_int,
        ti: *mut core::ffi::c_double,
        jm: *mut core::ffi::c_int,
    );
}

/// `get_rslow_stretch`: invert speeds into a padded slowness grid.
///
/// `jitter` is genslip's `rvel_rand`, a hardwired 0.0 with no `getpar` behind it, so
/// the lognormal branch it guards never fires and no deviates are drawn. Exposed
/// anyway so a test can prove that rather than assume it.
#[expect(clippy::too_many_arguments, reason = "mirrors the C signature")]
pub fn padded_slowness(
    speed: &mut [f32],
    strike_count: usize,
    dip_count: usize,
    padded_strike: usize,
    padded_dip: usize,
    strike_offset: usize,
    dip_offset: usize,
    jitter: f32,
    rng_state: &mut i64,
) -> Vec<f64> {
    assert_eq!(
        speed.len(),
        strike_count * dip_count,
        "speed extent mismatch"
    );
    let mut slowness = vec![0.0_f64; padded_strike * padded_dip];
    let mut jitter = jitter;

    // SAFETY: `speed` holds ns*nd elements and `slowness` holds nsfd*ndfd, which is
    // what the routine indexes. The offsets are the caller's and are checked by the
    // assertions above plus the vector's own length.
    unsafe {
        get_rslow_stretch(
            speed.as_mut_ptr(),
            i32::try_from(strike_count).expect("extent fits a C int"),
            i32::try_from(dip_count).expect("extent fits a C int"),
            slowness.as_mut_ptr(),
            i32::try_from(padded_strike).expect("extent fits a C int"),
            i32::try_from(padded_dip).expect("extent fits a C int"),
            i32::try_from(strike_offset).expect("offset fits a C int"),
            i32::try_from(dip_offset).expect("offset fits a C int"),
            &raw mut jitter,
            std::ptr::from_mut(rng_state),
        );
    }
    slowness
}

/// `wfront2d`: first-arrival times on a padded slowness grid.
///
/// `source_strike` and `source_dip` are **0-based** here; the conversion to Fortran's
/// 1-based indexing happens inside.
pub fn wavefront_times(
    slowness: &mut [f64],
    padded_strike: usize,
    padded_dip: usize,
    source_strike: usize,
    source_dip: usize,
    spacing_km: f64,
    ring_radius: usize,
) -> Vec<f64> {
    assert_eq!(
        slowness.len(),
        padded_strike * padded_dip,
        "slowness extent mismatch"
    );
    let mut times = vec![0.0_f64; slowness.len()];
    let mut time_scratch = vec![0.0_f64; padded_strike + padded_dip];
    let mut index_scratch = vec![0_i32; padded_strike + padded_dip];

    let m = i32::try_from(padded_strike).expect("extent fits a C int");
    let n = i32::try_from(padded_dip).expect("extent fits a C int");
    let is = i32::try_from(source_strike + 1).expect("index fits a C int");
    let js = i32::try_from(source_dip + 1).expect("index fits a C int");
    let ns = i32::try_from(ring_radius).expect("radius fits a C int");
    let ntot = i32::try_from(slowness.len()).expect("size fits a C int");

    // SAFETY: `times` and `slowness` are both `ntot` long, and both scratch buffers
    // are `m + n` long, as the routine's header requires.
    unsafe {
        wfront2d(
            &raw const m,
            &raw const n,
            &raw const is,
            &raw const js,
            &raw const spacing_km,
            &raw const ns,
            times.as_mut_ptr(),
            slowness.as_mut_ptr(),
            &raw const ntot,
            time_scratch.as_mut_ptr(),
            index_scratch.as_mut_ptr(),
        );
    }
    times
}

unsafe extern "C" {
    /// `void set_ll(float *elon, float *elat, float *slon, float *slat, float *sn,
    /// float *se)`
    fn set_ll(
        elon: *mut core::ffi::c_float,
        elat: *mut core::ffi::c_float,
        slon: *mut core::ffi::c_float,
        slat: *mut core::ffi::c_float,
        sn: *mut core::ffi::c_float,
        se: *mut core::ffi::c_float,
    );
}

/// `set_ll`: a kilometre offset from a point, as a longitude and latitude.
///
/// Returns `(longitude, latitude)` in degrees.
#[must_use]
pub fn offset_point(
    origin_longitude: f32,
    origin_latitude: f32,
    north_km: f32,
    east_km: f32,
) -> (f32, f32) {
    let mut elon = origin_longitude;
    let mut elat = origin_latitude;
    let mut slon = 0.0_f32;
    let mut slat = 0.0_f32;
    let mut north = north_km;
    let mut east = east_km;

    // SAFETY: all six pointers address live locals of the matching type. The routine
    // reads four and writes two.
    unsafe {
        set_ll(
            &raw mut elon,
            &raw mut elat,
            &raw mut slon,
            &raw mut slat,
            &raw mut north,
            &raw mut east,
        );
    }

    (slon, slat)
}

unsafe extern "C" {
    /// `void get_rspeed_vsden2(float *rspd, struct pointsource *ps, int nx, int ny,
    /// float *smax, float *savg, float *shal_vr, float *dmin1, float *dmax1,
    /// float *deep_vr, float *dmin2, float *dmax2, float *rvfmn, float *rvfmx,
    /// int scl_slip)`
    fn get_rspeed_vsden2(
        rspd: *mut core::ffi::c_float,
        ps: *mut crate::PointSource,
        nx: core::ffi::c_int,
        ny: core::ffi::c_int,
        smax: *mut core::ffi::c_float,
        savg: *mut core::ffi::c_float,
        shal_vr: *mut core::ffi::c_float,
        dmin1: *mut core::ffi::c_float,
        dmax1: *mut core::ffi::c_float,
        deep_vr: *mut core::ffi::c_float,
        dmin2: *mut core::ffi::c_float,
        dmax2: *mut core::ffi::c_float,
        rvfmn: *mut core::ffi::c_float,
        rvfmx: *mut core::ffi::c_float,
        scl_slip: core::ffi::c_int,
    );
}

/// The five depths and factors that shape the rupture-speed profile.
#[derive(Clone, Copy, Debug)]
pub struct SpeedProfileArgs {
    pub shallow_factor: f32,
    pub shallow_min_km: f32,
    pub shallow_max_km: f32,
    pub deep_factor: f32,
    pub deep_min_km: f32,
    pub deep_max_km: f32,
}

/// `get_rspeed_vsden2`: rupture speed per subfault, from shear speed and depth.
///
/// `scale_with_slip` is genslip's `fdrup_scale_slip`, configured off.
#[must_use]
pub fn rupture_speed(
    subfaults: &mut [crate::PointSource],
    strike_count: usize,
    dip_count: usize,
    profile: SpeedProfileArgs,
    scale_with_slip: bool,
) -> Vec<f32> {
    assert_eq!(
        subfaults.len(),
        strike_count * dip_count,
        "got {} subfaults for a {strike_count}x{dip_count} fault",
        subfaults.len()
    );
    let (nx, ny) = extents(strike_count, dip_count);
    let mut speeds = vec![0.0_f32; subfaults.len()];

    // Only read when `scale_with_slip` is set; passed anyway so the signature is
    // honest about what the C takes.
    let mut maximum_slip = subfaults
        .iter()
        .map(|subfault| subfault.slip)
        .fold(0.0_f32, f32::max);
    #[expect(clippy::cast_precision_loss, reason = "subfault counts are small")]
    let mut average_slip = subfaults.iter().map(|s| s.slip).sum::<f32>() / subfaults.len() as f32;

    let mut shallow_factor = profile.shallow_factor;
    let mut shallow_min = profile.shallow_min_km;
    let mut shallow_max = profile.shallow_max_km;
    let mut deep_factor = profile.deep_factor;
    let mut deep_min = profile.deep_min_km;
    let mut deep_max = profile.deep_max_km;
    let mut minimum_fraction = 0.25_f32;
    let mut maximum_fraction = 1.414_f32;

    // SAFETY: `speeds` and `subfaults` both hold nx*ny elements, which is what the
    // routine indexes; every scalar pointer addresses a live local.
    unsafe {
        get_rspeed_vsden2(
            speeds.as_mut_ptr(),
            subfaults.as_mut_ptr(),
            nx,
            ny,
            &raw mut maximum_slip,
            &raw mut average_slip,
            &raw mut shallow_factor,
            &raw mut shallow_min,
            &raw mut shallow_max,
            &raw mut deep_factor,
            &raw mut deep_min,
            &raw mut deep_max,
            &raw mut minimum_fraction,
            &raw mut maximum_fraction,
            i32::from(scale_with_slip),
        );
    }

    speeds
}
