//! The Python boundary.
//!
//! Five configuration groups and one call. The groups **mirror the core's own
//! decomposition** — `FaultGrid`, `SourceSpec`, `SlipSpec`, `TimingSpec`, and the
//! velocity model — so a configuration written down in YAML deserialises straight
//! into them rather than being translated field by field. A translation layer is
//! where the two descriptions of the same thing drift apart.
//!
//! # What crosses the boundary
//!
//! Arrays, not files. Geometry arrives as numpy from `rupture_generator.geometry`;
//! the model leaves as numpy. Nothing here reads or writes a GSF or an SRF — Python
//! owns every format, and `rupture_generator.srf` is what turns the arrays into one.
//!
//! # The GIL is released for the generation itself
//!
//! A rupture model takes tens of milliseconds on a small fault and seconds on a
//! large one, all of it in Rust with no Python objects touched. Releasing the GIL
//! lets a caller run several faults at once from a thread pool.
//!
//! **With one caveat**, and it is the reason `RustFft` is the destination: FFTW's
//! planner is process-global mutable state. `FftwFft` installs FFTW's own lock
//! before planning, which makes this safe, but it serialises the planning of every
//! fault in the process. `rustfft` needs no lock at all.

// `PyReadonlyArray` arguments are taken by value because that is how PyO3 extracts
// them; there is no by-reference form. The lint is right in general and wrong at
// every array argument in this file.
#![expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts PyReadonlyArray arguments by value"
)]

use genslip::fft::FftwFft;
use genslip::field::{CorrelationLengths, Spectrum2D, WavelengthBand};
use genslip::realisation::{self, RuptureModel};
use genslip::rise_time::{DepthRamp, RiseTimeSpec, RiseTimeStretch, Weighting};
use genslip::rng::{GenslipLcg, Realisations as _};
use genslip::rupture::{Hypocentre, SpeedProfile, Wavefront2d};
use genslip::slip::{GridExtents, PerturbationSpec, SpectrumSpec, SubfaultSpacing};
use genslip::slip_rate::BetaProfile;
use genslip::source::{CornerRelation, Layer, MagnitudeScale, VelocityModel};
use genslip::taper::EdgeTapers;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Which relation maps magnitude onto the slip spectrum's wavenumber corners.
///
/// The offsets and exponents live on the Python side, so this only names the family.
#[pyclass(eq, eq_int, from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpectrumModel {
    Somerville,
    Mai,
    Frankel,
    MaiSomerville,
    Suzuki,
    InputCorners,
}

impl SpectrumModel {
    const fn shape(self) -> Spectrum2D {
        match self {
            Self::Somerville => Spectrum2D::Somerville,
            Self::Mai => Spectrum2D::Mai,
            Self::Frankel => Spectrum2D::Frankel,
            Self::MaiSomerville => Spectrum2D::MaiSomerville,
            Self::Suzuki => Spectrum2D::Suzuki,
            Self::InputCorners => Spectrum2D::InputCorners,
        }
    }
}

/// How the fault-wide rise-time constant is averaged.
#[pyclass(eq, eq_int, from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RiseTimeWeighting {
    Uniform,
    BySlip,
    BySlipAndRuptureSpeed,
}

impl RiseTimeWeighting {
    const fn weighting(self) -> Weighting {
        match self {
            Self::Uniform => Weighting::Uniform,
            Self::BySlip => Weighting::BySlip,
            Self::BySlipAndRuptureSpeed => Weighting::BySlipAndRuptureSpeed,
        }
    }
}

/// A linear ramp between two depths, in kilometres.
#[pyclass(from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Copy, Debug)]
pub struct Ramp {
    #[pyo3(get)]
    pub centre_km: f32,
    #[pyo3(get)]
    pub half_width_km: f32,
}

#[pymethods]
impl Ramp {
    #[new]
    const fn new(centre_km: f32, half_width_km: f32) -> Self {
        Self {
            centre_km,
            half_width_km,
        }
    }

    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "PyO3 methods take &self; the receiver is not ours to choose"
    )]
    fn __repr__(&self) -> String {
        format!(
            "Ramp(centre_km={}, half_width_km={})",
            self.centre_km, self.half_width_km
        )
    }
}

impl Ramp {
    const fn ramp(self) -> DepthRamp {
        DepthRamp {
            centre_km: self.centre_km,
            half_width_km: self.half_width_km,
        }
    }
}

/// The discretised fault.
///
/// `depth_km` is one value per dip row; `base_rake_deg` and `velocity_fraction` are
/// one per subfault, along-strike index fastest. The padded extents are the
/// wraparound margin the generator needs and are the caller's to compute — genslip
/// derives them from the fault's share of a possibly multi-segment rupture.
#[pyclass(skip_from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Debug)]
pub struct FaultGrid {
    inner: realisation::FaultGrid,
}

#[pymethods]
impl FaultGrid {
    #[new]
    #[pyo3(signature = (
        fault_strike, fault_dip, padded_strike, padded_dip,
        strike_km, dip_km, depth_km, base_rake_deg, velocity_fraction,
    ))]
    #[expect(clippy::too_many_arguments, reason = "one flat constructor per group")]
    fn new(
        fault_strike: usize,
        fault_dip: usize,
        padded_strike: usize,
        padded_dip: usize,
        strike_km: f32,
        dip_km: f32,
        depth_km: PyReadonlyArray1<'_, f32>,
        base_rake_deg: PyReadonlyArray1<'_, f32>,
        velocity_fraction: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Self> {
        if fault_strike == 0 || fault_dip == 0 {
            return Err(PyValueError::new_err("a fault needs at least one subfault"));
        }
        if fault_strike > padded_strike || fault_dip > padded_dip {
            return Err(PyValueError::new_err(format!(
                "a {fault_strike}x{fault_dip} fault does not fit in a \
                 {padded_strike}x{padded_dip} grid"
            )));
        }
        // genslip rounds every padded extent up to even (`if(nstk2%2) nstk2++;` and
        // three siblings, genslip_v5.6.2.c:1471-1490) because the generators address
        // the Nyquist row and column directly. An odd extent means the caller sized
        // the grid wrongly, so it is refused here rather than panicking three layers
        // down.
        if !padded_strike.is_multiple_of(2) || !padded_dip.is_multiple_of(2) {
            return Err(PyValueError::new_err(format!(
                "padded extents must be even, got {padded_strike}x{padded_dip}"
            )));
        }
        if strike_km <= 0.0 || dip_km <= 0.0 {
            return Err(PyValueError::new_err(
                "subfault dimensions must be strictly positive",
            ));
        }

        let subfaults = fault_strike * fault_dip;
        let depth_km = depth_km.as_slice()?.to_vec();
        let base_rake_deg = base_rake_deg.as_slice()?.to_vec();
        let velocity_fraction = velocity_fraction.as_slice()?.to_vec();

        if depth_km.len() != fault_dip {
            return Err(PyValueError::new_err(format!(
                "depth_km has {} entries; a {fault_dip}-row fault needs one per row",
                depth_km.len()
            )));
        }
        for (name, values) in [
            ("base_rake_deg", &base_rake_deg),
            ("velocity_fraction", &velocity_fraction),
        ] {
            if values.len() != subfaults {
                return Err(PyValueError::new_err(format!(
                    "{name} has {} entries; a {fault_strike}x{fault_dip} fault needs \
                     {subfaults}",
                    values.len()
                )));
            }
        }

        Ok(Self {
            inner: realisation::FaultGrid {
                extents: GridExtents {
                    fault_strike,
                    fault_dip,
                    padded_strike,
                    padded_dip,
                },
                spacing: SubfaultSpacing { strike_km, dip_km },
                depth_km,
                base_rake_deg,
                velocity_fraction,
            },
        })
    }

    #[getter]
    const fn subfault_count(&self) -> usize {
        self.inner.extents.fault_strike * self.inner.extents.fault_dip
    }
}

/// A layered one-dimensional velocity model.
#[pyclass(skip_from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Debug)]
pub struct VelocityModel1D {
    inner: VelocityModel,
}

#[pymethods]
impl VelocityModel1D {
    /// Build from three parallel arrays, ordered shallow to deep.
    ///
    /// `bottom_depth_km` is the depth to the *bottom* of each layer.
    #[new]
    fn new(
        bottom_depth_km: PyReadonlyArray1<'_, f32>,
        shear_speed_km_s: PyReadonlyArray1<'_, f32>,
        density_g_cm3: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<Self> {
        let depths = bottom_depth_km.as_slice()?;
        let speeds = shear_speed_km_s.as_slice()?;
        let densities = density_g_cm3.as_slice()?;

        if depths.len() != speeds.len() || depths.len() != densities.len() {
            return Err(PyValueError::new_err(
                "bottom_depth_km, shear_speed_km_s and density_g_cm3 must be the \
                 same length",
            ));
        }
        if depths.is_empty() {
            return Err(PyValueError::new_err(
                "a velocity model needs at least one layer",
            ));
        }

        let layers = (0..depths.len())
            .map(|index| Layer {
                bottom_depth_km: depths[index],
                shear_speed_km_s: speeds[index],
                density_g_cm3: densities[index],
            })
            .collect();

        Ok(Self {
            inner: VelocityModel::new(layers),
        })
    }
}

/// What the earthquake is, before any field is drawn.
#[pyclass(skip_from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Copy, Debug)]
pub struct SourceSpec {
    inner: realisation::SourceSpec,
}

#[pymethods]
impl SourceSpec {
    /// Build the source description.
    ///
    /// `model` selects two things at once, because the original's `kmodel` does:
    /// the spectral falloff shape, and the relation mapping magnitude onto the
    /// wavenumber corners. **They do not partition the same way** — Frankel has a
    /// falloff of its own (`slip.c:1651`) while sharing Mai's corner relation, since
    /// the original's branch is `kmodel == MAI_FLAG || kmodel == FRANKEL_FLAG`
    /// (`genslip_v5.6.2.c:1303`). Routing it to Somerville instead is a different
    /// power law with different offsets, and every corner comes out wrong.
    ///
    /// `circular_average` forces the down-dip corner to equal the along-strike one.
    /// It reaches Somerville and Mai — the two relations whose branches test it —
    /// and not Suzuki, Input Corners, or the Mai-Somerville hybrid, whose branches
    /// do not. Under Somerville it is not merely equality: the original switches to
    /// a *third* offset, 1.825 rather than 1.72 and 1.93.
    #[new]
    #[pyo3(signature = (
        magnitude, model, strike_offset, dip_offset, *,
        use_moment_magnitude = true, modified_corners = false, circular_average = false,
        saturation_magnitude = 6.3, strike_exponent = 0.5, dip_exponent = 0.5,
        rise_time_coefficient = 1.6, average_dip_deg, average_rake_deg,
    ))]
    #[expect(clippy::too_many_arguments, reason = "one flat constructor per group")]
    const fn new(
        magnitude: f32,
        model: SpectrumModel,
        strike_offset: f32,
        dip_offset: f32,
        use_moment_magnitude: bool,
        modified_corners: bool,
        circular_average: bool,
        saturation_magnitude: f32,
        strike_exponent: f32,
        dip_exponent: f32,
        rise_time_coefficient: f32,
        average_dip_deg: f32,
        average_rake_deg: f32,
    ) -> Self {
        let corners = match model {
            SpectrumModel::Somerville => CornerRelation::Somerville {
                circular: circular_average,
            },
            // Frankel takes the Mai relation, not Somerville's: one branch serves
            // both in the original. Its falloff shape stays its own.
            SpectrumModel::Mai | SpectrumModel::Frankel => CornerRelation::Mai {
                strike_offset,
                dip_offset,
                circular: circular_average,
            },
            // The hybrid's branch tests neither `circular_average` nor the corner
            // parameters -- it evaluates Mai's literals. See `mapping.corner_offsets`.
            SpectrumModel::MaiSomerville => CornerRelation::Mai {
                strike_offset,
                dip_offset,
                circular: false,
            },
            SpectrumModel::Suzuki => CornerRelation::Suzuki {
                strike_offset,
                dip_offset,
                saturation_magnitude,
            },
            SpectrumModel::InputCorners => CornerRelation::Given {
                strike_offset,
                dip_offset,
                strike_exponent,
                dip_exponent,
            },
        };

        Self {
            inner: realisation::SourceSpec {
                magnitude,
                magnitude_scale: if use_moment_magnitude {
                    MagnitudeScale::Moment
                } else {
                    MagnitudeScale::Local
                },
                corners,
                modified_corners,
                rise_time_coefficient,
                average_dip_deg,
                average_rake_deg,
            },
        }
    }
}

/// How the slip field is shaped and trimmed.
#[pyclass(skip_from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Copy, Debug)]
pub struct SlipSpec {
    inner: realisation::SlipSpec,
    model: SpectrumModel,
}

#[pymethods]
impl SlipSpec {
    #[new]
    #[pyo3(signature = (
        model, *, coefficient_of_variation = 0.75,
        min_wavelength_km = 1.5, max_wavelength_km = 80.0,
        strike_shift = 0.0, dip_shift = 0.0,
        side_taper = 0.02, top_taper = 0.0, bottom_taper = 0.0,
        truncate_negative = true, water_level = 0.0,
    ))]
    #[expect(clippy::too_many_arguments, reason = "one flat constructor per group")]
    fn new(
        model: SpectrumModel,
        coefficient_of_variation: f32,
        min_wavelength_km: f32,
        max_wavelength_km: f32,
        strike_shift: f64,
        dip_shift: f64,
        side_taper: f32,
        top_taper: f32,
        bottom_taper: f32,
        truncate_negative: bool,
        water_level: f32,
    ) -> PyResult<Self> {
        if min_wavelength_km <= 0.0 || max_wavelength_km <= 0.0 {
            return Err(PyValueError::new_err(
                "wavelength limits must be strictly positive",
            ));
        }

        Ok(Self {
            model,
            inner: realisation::SlipSpec {
                spectrum: SpectrumSpec {
                    shape: model.shape(),
                    // Overwritten from the magnitude relation inside `generate`.
                    correlation: CorrelationLengths {
                        strike: 1.0,
                        dip: 1.0,
                    },
                    band: WavelengthBand::new(min_wavelength_km, max_wavelength_km),
                    coefficient_of_variation,
                    phase_shift: (strike_shift, dip_shift),
                },
                tapers: EdgeTapers {
                    sides: side_taper,
                    top: top_taper,
                    bottom: bottom_taper,
                },
                truncate_negative,
                water_level,
            },
        })
    }

    #[getter]
    const fn model(&self) -> SpectrumModel {
        self.model
    }
}

/// How rupture time and rise time relate to slip.
#[pyclass(skip_from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Copy, Debug)]
pub struct TimingSpec {
    inner: realisation::TimingSpec,
}

#[pymethods]
impl TimingSpec {
    /// Build the timing description.
    ///
    /// `shallow_ramp` and `deep_ramp` are the depth ramps that stretch **rise time**.
    /// Rupture *speed* has ramps of its own — the original reads four independent
    /// pairs, `risetimedep`/`risetimedep_range` and `deep_risetimedep`/
    /// `deep_risetimedep_range` against `shal_vrup_dep`/`shal_vrup_deprange` and
    /// `deep_vrup_dep`/`deep_vrup_deprange`. They share defaults, 6.5/1.5 and
    /// 17.5/2.5, so collapsing them into one pair is invisible until someone moves
    /// one and not the other.
    ///
    /// `shallow_speed_ramp` and `deep_speed_ramp` are therefore optional and fall
    /// back to the rise-time ramps, which reproduces the shared-default case exactly
    /// while leaving the divergent one expressible.
    ///
    /// The two also diverge without being configured differently: both deep ramps
    /// are pushed down to the hypocentre depth per realisation
    /// (`genslip_v5.6.2.c:2378-2381` and `:2974-2977`), each using its *own*
    /// half-width in that adjustment. Equal centres and unequal half-widths give
    /// unequal ramps at a deep hypocentre.
    #[new]
    #[pyo3(signature = (
        *, rupture_time_correlation = 0.8, rupture_time_sigma = 1.0,
        rupture_time_scale, rupture_delay_s = 0.0,
        rise_time_correlation = 0.9, rise_time_sigma = 0.75,
        rise_time_blend, slip_exponent = 0.5,
        shallow_ramp, shallow_rise_factor = 2.0,
        deep_ramp, deep_rise_factor = 2.0,
        shallow_speed_ramp = None, deep_speed_ramp = None,
        shallow_speed_factor = 0.6, deep_speed_factor = 0.6,
        weighting = RiseTimeWeighting::Uniform,
        beta_shallow_ramp, beta_shallow = 0.5,
        beta_mid_ramp, beta_mid = 0.13, beta_deep = 0.13,
        sample_interval_s = 0.005, max_samples = 100_000,
    ))]
    #[expect(clippy::too_many_arguments, reason = "one flat constructor per group")]
    const fn new(
        rupture_time_correlation: f32,
        rupture_time_sigma: f32,
        rupture_time_scale: f32,
        rupture_delay_s: f32,
        rise_time_correlation: f32,
        rise_time_sigma: f32,
        rise_time_blend: Ramp,
        slip_exponent: f32,
        shallow_ramp: Ramp,
        shallow_rise_factor: f32,
        deep_ramp: Ramp,
        deep_rise_factor: f32,
        shallow_speed_ramp: Option<Ramp>,
        deep_speed_ramp: Option<Ramp>,
        shallow_speed_factor: f32,
        deep_speed_factor: f32,
        weighting: RiseTimeWeighting,
        beta_shallow_ramp: Ramp,
        beta_shallow: f32,
        beta_mid_ramp: Ramp,
        beta_mid: f32,
        beta_deep: f32,
        sample_interval_s: f32,
        max_samples: usize,
    ) -> Self {
        // `Option::unwrap_or` is not const, and a match is clearer about the
        // fallback being "the rise-time ramp" rather than some neutral default.
        let shallow_speed_ramp = match shallow_speed_ramp {
            Some(ramp) => ramp,
            None => shallow_ramp,
        };
        let deep_speed_ramp = match deep_speed_ramp {
            Some(ramp) => ramp,
            None => deep_ramp,
        };

        Self {
            inner: realisation::TimingSpec {
                rupture_time: PerturbationSpec {
                    correlation: rupture_time_correlation,
                    sigma: rupture_time_sigma,
                },
                rupture_time_scale,
                rupture_delay_s,
                rise_time: RiseTimeSpec {
                    perturbation: PerturbationSpec {
                        correlation: rise_time_correlation,
                        sigma: rise_time_sigma,
                    },
                    shallow_blend: rise_time_blend.ramp(),
                    slip_exponent,
                },
                rise_time_stretch: RiseTimeStretch {
                    shallow: shallow_ramp.ramp(),
                    shallow_factor: shallow_rise_factor,
                    deep: deep_ramp.ramp(),
                    deep_factor: deep_rise_factor,
                },
                rise_time_weighting: weighting.weighting(),
                speed_profile: SpeedProfile {
                    shallow: shallow_speed_ramp.ramp(),
                    shallow_factor: shallow_speed_factor,
                    deep: deep_speed_ramp.ramp(),
                    deep_factor: deep_speed_factor,
                },
                beta: BetaProfile {
                    shallow_ramp: beta_shallow_ramp.ramp(),
                    shallow: beta_shallow,
                    mid_ramp: beta_mid_ramp.ramp(),
                    mid: beta_mid,
                    deep: beta_deep,
                },
                sample_interval_s,
                max_samples,
            },
        }
    }
}

/// A generated rupture model.
///
/// Every field is a flat array over subfaults, along-strike index fastest — the
/// order every other array in the pipeline uses. The slip-rate functions are ragged,
/// so they come back as one concatenated array plus the offsets that index into it,
/// which is the layout `scipy.sparse.csr_array` wants.
#[pyclass(module = "rupture_generator._core")]
pub struct GeneratedRupture {
    slip_cm: Vec<f32>,
    rake_deg: Vec<f32>,
    onset_s: Vec<f32>,
    rise_time_s: Vec<f32>,
    slip_rate: Vec<f32>,
    slip_rate_offsets: Vec<u64>,
    strike_count: usize,
    dip_count: usize,
    moment_dyne_cm: f32,
    alpha_t: f32,
    sample_interval_s: f32,
}

impl GeneratedRupture {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "onset times are seconds and the downstream format stores them as f32"
    )]
    fn from_model(model: &RuptureModel, sample_interval_s: f32) -> Self {
        let mut slip_rate = Vec::new();
        let mut slip_rate_offsets = Vec::with_capacity(model.slip_rate.len() + 1);
        slip_rate_offsets.push(0);
        for pulse in &model.slip_rate {
            slip_rate.extend_from_slice(pulse.as_slice());
            slip_rate_offsets.push(slip_rate.len() as u64);
        }

        Self {
            slip_cm: model.slip.slip.as_slice().to_vec(),
            rake_deg: model.rake_deg.as_slice().to_vec(),
            onset_s: model
                .onset_s
                .as_slice()
                .iter()
                .map(|time| *time as f32)
                .collect(),
            rise_time_s: model.rise_time_s.as_slice().to_vec(),
            slip_rate,
            slip_rate_offsets,
            strike_count: model.slip.slip.strike_count(),
            dip_count: model.slip.slip.dip_count(),
            moment_dyne_cm: model.moment_dyne_cm,
            alpha_t: model.alpha_t,
            sample_interval_s,
        }
    }
}

#[pymethods]
impl GeneratedRupture {
    /// Slip in centimetres, one per subfault.
    #[getter]
    fn slip_cm<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.slip_cm.clone().into_pyarray(py)
    }

    /// Rake in degrees, one per subfault.
    #[getter]
    fn rake_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.rake_deg.clone().into_pyarray(py)
    }

    /// Rupture onset in seconds, zero at the hypocentre.
    #[getter]
    fn onset_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.onset_s.clone().into_pyarray(py)
    }

    /// Rise time in seconds, one per subfault.
    #[getter]
    fn rise_time_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.rise_time_s.clone().into_pyarray(py)
    }

    /// Every slip-rate sample, concatenated. Index it with `slip_rate_offsets`.
    #[getter]
    fn slip_rate<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.slip_rate.clone().into_pyarray(py)
    }

    /// Where each subfault's pulse starts in `slip_rate`. One longer than the
    /// subfault count, so pulse `i` is `slip_rate[offsets[i]:offsets[i + 1]]`.
    #[getter]
    fn slip_rate_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        self.slip_rate_offsets.clone().into_pyarray(py)
    }

    /// Sample interval of the slip-rate functions, in seconds.
    #[getter]
    const fn sample_interval_s(&self) -> f32 {
        self.sample_interval_s
    }

    /// Seismic moment in dyne-cm.
    #[getter]
    const fn moment_dyne_cm(&self) -> f32 {
        self.moment_dyne_cm
    }

    /// The dip-and-rake correction that was applied to rise time and rupture speed.
    #[getter]
    const fn alpha_t(&self) -> f32 {
        self.alpha_t
    }

    /// `(strike_count, dip_count)`.
    #[getter]
    const fn shape(&self) -> (usize, usize) {
        (self.strike_count, self.dip_count)
    }

    /// Slip reshaped to `(dip, strike)`, which is C order for this layout.
    fn slip_grid<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let values = ndarray_from(&self.slip_cm, self.strike_count, self.dip_count)?;
        Ok(values.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "GeneratedRupture(shape=({}, {}), moment={:.4e})",
            self.strike_count, self.dip_count, self.moment_dyne_cm
        )
    }
}

fn ndarray_from(
    values: &[f32],
    strike_count: usize,
    dip_count: usize,
) -> PyResult<numpy::ndarray::Array2<f32>> {
    numpy::ndarray::Array2::from_shape_vec((dip_count, strike_count), values.to_vec())
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Generate one rupture model.
///
/// `realisation` selects an independent draw stream, so realisation *n* is
/// reproducible without generating the ones before it.
///
/// The GIL is released for the generation itself.
#[pyfunction]
#[pyo3(signature = (
    grid, velocity_model, source, slip, timing, *,
    seed, realisation = 0, hypocentre_strike, hypocentre_dip,
))]
#[expect(
    clippy::too_many_arguments,
    reason = "five config groups plus the seed"
)]
fn generate_rupture(
    py: Python<'_>,
    grid: &FaultGrid,
    velocity_model: &VelocityModel1D,
    source: &SourceSpec,
    slip: &SlipSpec,
    timing: &TimingSpec,
    seed: i64,
    realisation: u64,
    hypocentre_strike: usize,
    hypocentre_dip: usize,
) -> PyResult<GeneratedRupture> {
    let extents = grid.inner.extents;
    if hypocentre_strike >= extents.fault_strike || hypocentre_dip >= extents.fault_dip {
        return Err(PyValueError::new_err(format!(
            "hypocentre ({hypocentre_strike}, {hypocentre_dip}) is outside a {}x{} fault",
            extents.fault_strike, extents.fault_dip
        )));
    }

    let sample_interval_s = timing.inner.sample_interval_s;

    // Nothing below touches a Python object.
    let model = py.detach(|| {
        let mut draws = GenslipLcg::new(seed).realisation(realisation);
        let mut fft = FftwFft::new();
        let mut solver = Wavefront2d::new();
        realisation::generate(
            &mut draws,
            &mut fft,
            &mut solver,
            &grid.inner,
            &velocity_model.inner,
            source.inner,
            slip.inner,
            timing.inner,
            Hypocentre {
                strike: hypocentre_strike,
                dip: hypocentre_dip,
            },
        )
    });

    Ok(GeneratedRupture::from_model(&model, sample_interval_s))
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SpectrumModel>()?;
    module.add_class::<RiseTimeWeighting>()?;
    module.add_class::<Ramp>()?;
    module.add_class::<FaultGrid>()?;
    module.add_class::<VelocityModel1D>()?;
    module.add_class::<SourceSpec>()?;
    module.add_class::<SlipSpec>()?;
    module.add_class::<TimingSpec>()?;
    module.add_class::<GeneratedRupture>()?;
    module.add_function(wrap_pyfunction!(generate_rupture, module)?)?;
    Ok(())
}
