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
//! No caveat any more. It used to have one: FFTW's planner is process-global mutable
//! state and aborts if two threads enter it at once, so `FftwFft` had to install
//! FFTW's own lock. `RustFft` has no global state, and FFTW is gone.
//! before planning, which makes this safe, but it serialises the planning of every
//! fault in the process. `rustfft` needs no lock at all.

// `PyReadonlyArray` arguments are taken by value because that is how PyO3 extracts
// them; there is no by-reference form. The lint is right in general and wrong at
// every array argument in this file.
#![expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts PyReadonlyArray arguments by value"
)]

use genslip::fft::RustFft;
use genslip::field::{CorrelationLengths, Spectrum2D, WavelengthBand};
use genslip::grid::FaultAxes;
use genslip::realisation::{self, RuptureModel};
use genslip::rise_time::{DepthRamp, RiseTimeSpec, RiseTimeStretch, Weighting};
use genslip::rng::{GenslipLcg, Realisations as _};
use genslip::rupture::{FactoredSweep, Hypocentre, SpeedProfile};
use genslip::slip::{GridExtents, PerturbationSpec, SpectrumSpec, SubfaultSpacing};
use genslip::slip_rate::{BetaProfile, SlipRateShape as Shape};
use genslip::source::{CornerRelation, Layer, MagnitudeScale, VelocityModel};
use genslip::taper::EdgeTapers;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Every way the library can refuse, as one `ValueError`.
///
/// `genslip::Error`'s `Display` already names the input and the constraint it broke,
/// so there is nothing to add here and nothing to keep in step: a new variant reaches
/// Python with its own message the day it is written. This replaced two hand-written
/// hypocentre checks that duplicated the library's own, word for word and one
/// refactor away from disagreeing with it.
struct Refused(genslip::Error);

impl From<Refused> for PyErr {
    fn from(Refused(error): Refused) -> Self {
        PyValueError::new_err(error.to_string())
    }
}

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

/// Which slip-rate function every subfault gets.
///
/// A class with named constructors rather than an `enum`, because four of the shapes
/// carry a parameter and Python enums do not. `SlipRateShape.ucsb()` and
/// `SlipRateShape.ucsb_t(2.0)` are both shapes; only the second has anything to say
/// beyond its name.
///
/// [`from_stype`](SlipRateShape::from_stype) parses `generic_slip2srf`'s command-line
/// vocabulary, including the numeric suffix on `ucsb-T`.
#[pyclass(eq, frozen, from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlipRateShape {
    inner: Shape,
}

#[pymethods]
impl SlipRateShape {
    /// genslip's finite-fault default. The only shape whose parameter comes from a
    /// depth profile rather than from here.
    #[staticmethod]
    const fn oliu_p2() -> Self {
        Self {
            inner: Shape::OliuP2,
        }
    }

    /// `stype=ucsb`.
    #[staticmethod]
    const fn ucsb() -> Self {
        Self { inner: Shape::Ucsb }
    }

    /// `stype=ucsb2`.
    #[staticmethod]
    const fn ucsb2() -> Self {
        Self {
            inner: Shape::Ucsb2,
        }
    }

    /// `stype=ucsb-T<stretch>`.
    #[staticmethod]
    fn ucsb_t(stretch: f64) -> PyResult<Self> {
        if stretch <= 0.0 {
            return Err(PyValueError::new_err(
                "the ucsb-T stretch must be strictly positive",
            ));
        }
        Ok(Self {
            inner: Shape::UcsbT { stretch },
        })
    }

    /// `stype=ucsb-varT1`. The C defaults `tau1_ratio` to 0.13, which is `ucsb`.
    #[staticmethod]
    #[pyo3(signature = (tau1_ratio = 0.13))]
    fn ucsb_var_t1(tau1_ratio: f64) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&tau1_ratio) {
            return Err(PyValueError::new_err(
                "tau1_ratio is a fraction of the duration and must be in [0, 1]",
            ));
        }
        Ok(Self {
            inner: Shape::UcsbVarT1 { tau1_ratio },
        })
    }

    /// `stype=brune`. The duration is the rise time, not the C's slip-derived
    /// constant — see the Rust `SlipRateShape::Brune`.
    #[staticmethod]
    const fn brune() -> Self {
        Self {
            inner: Shape::Brune,
        }
    }

    /// `stype=urs`.
    #[staticmethod]
    const fn urs() -> Self {
        Self { inner: Shape::Urs }
    }

    /// `stype=esg2006`.
    #[staticmethod]
    const fn esg2006() -> Self {
        Self {
            inner: Shape::Esg2006,
        }
    }

    /// `stype=cos`.
    #[staticmethod]
    const fn cos() -> Self {
        Self { inner: Shape::Cos }
    }

    /// `stype=seki`. Moves each subfault's onset back a quarter of its rise time.
    #[staticmethod]
    const fn seki() -> Self {
        Self { inner: Shape::Seki }
    }

    /// `stype=delta`.
    #[staticmethod]
    const fn delta() -> Self {
        Self {
            inner: Shape::Delta,
        }
    }

    /// Parse `generic_slip2srf`'s `stype`.
    ///
    /// The one place the C's option vocabulary is decoded, including the numeric
    /// suffix `ucsb-T` accepts because the C dispatches on it with `strncmp`. An
    /// unrecognised name is an error here; the C falls through to `brune`, silently
    /// generating a different rupture from the one that was asked for.
    #[staticmethod]
    fn from_stype(stype: &str) -> PyResult<Self> {
        if let Some(suffix) = stype.strip_prefix("ucsb-T") {
            let stretch = if suffix.is_empty() {
                1.0
            } else {
                suffix.parse::<f64>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "{stype:?}: the text after `ucsb-T` must be a number"
                    ))
                })?
            };
            return Self::ucsb_t(stretch);
        }
        match stype {
            "ucsb" => Ok(Self::ucsb()),
            "ucsb2" => Ok(Self::ucsb2()),
            "ucsb-varT1" => Self::ucsb_var_t1(0.13),
            "brune" => Ok(Self::brune()),
            "urs" => Ok(Self::urs()),
            "esg2006" => Ok(Self::esg2006()),
            "cos" => Ok(Self::cos()),
            "seki" => Ok(Self::seki()),
            "delta" => Ok(Self::delta()),
            "OliuP2" => Ok(Self::oliu_p2()),
            other => Err(PyValueError::new_err(format!(
                "{other:?} is not a slip-rate function this understands"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!("SlipRateShape({:?})", self.inner)
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
    pub centre_km: f64,
    #[pyo3(get)]
    pub half_width_km: f64,
}

#[pymethods]
impl Ramp {
    #[new]
    const fn new(centre_km: f64, half_width_km: f64) -> Self {
        Self {
            centre_km,
            half_width_km,
        }
    }

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
        strike_km: f64,
        dip_km: f64,
        depth_km: PyReadonlyArray1<'_, f64>,
        base_rake_deg: PyReadonlyArray1<'_, f64>,
        velocity_fraction: PyReadonlyArray1<'_, f64>,
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
                // Per-subfault quantities are grids on the Rust side; the boundary
                // takes them flat, in the along-strike-fastest order every array
                // crossing here uses, and reshapes once.
                base_rake_deg: genslip::grid::from_values(fault_strike, fault_dip, base_rake_deg),
                velocity_fraction: genslip::grid::from_values(
                    fault_strike,
                    fault_dip,
                    velocity_fraction,
                ),
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
        bottom_depth_km: PyReadonlyArray1<'_, f64>,
        shear_speed_km_s: PyReadonlyArray1<'_, f64>,
        density_g_cm3: PyReadonlyArray1<'_, f64>,
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

    /// Depth to the bottom of each layer, in kilometres.
    #[getter]
    fn bottom_depth_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.layer_field(py, |layer| layer.bottom_depth_km)
    }

    /// Shear-wave speed in each layer, in kilometres per second.
    #[getter]
    fn shear_speed_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.layer_field(py, |layer| layer.shear_speed_km_s)
    }

    /// Density in each layer, in grams per cubic centimetre.
    #[getter]
    fn density_g_cm3<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.layer_field(py, |layer| layer.density_g_cm3)
    }

    /// Number of layers.
    fn __len__(&self) -> usize {
        self.inner.layers().len()
    }
}

impl VelocityModel1D {
    /// One column of the layer table, as a fresh array.
    ///
    /// Fresh rather than a view: a view would alias Rust-owned memory that Python
    /// could then write through, and the model is validated on construction. Three
    /// getters returning three arrays is also the shape the constructor takes, so a
    /// model round-trips through its own arguments.
    fn layer_field<'py>(
        &self,
        py: Python<'py>,
        of: impl Fn(&Layer) -> f64,
    ) -> Bound<'py, PyArray1<f64>> {
        self.inner
            .layers()
            .iter()
            .map(of)
            .collect::<Vec<_>>()
            .into_pyarray(py)
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
        magnitude: f64,
        model: SpectrumModel,
        strike_offset: f64,
        dip_offset: f64,
        use_moment_magnitude: bool,
        modified_corners: bool,
        circular_average: bool,
        saturation_magnitude: f64,
        strike_exponent: f64,
        dip_exponent: f64,
        rise_time_coefficient: f64,
        average_dip_deg: f64,
        average_rake_deg: f64,
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

/// What a point source is, over and above its geometry.
///
/// Deliberately not a [`SourceSpec`] with fields left blank. A point source has no
/// spectrum, so it has no corner relation, and it is *told* its rise time rather than
/// deriving one from the moment — so the two descriptions have four fields in common
/// and nothing else, and collapsing them would mean a caller filling in corner
/// exponents that are never read.
#[pyclass(skip_from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Copy, Debug)]
pub struct PointSourceSpec {
    inner: realisation::PointSourceSpec,
}

#[pymethods]
impl PointSourceSpec {
    /// Build the point-source description.
    ///
    /// `rise_time_s` is the **fault-wide average**, which the depth ramp then
    /// redistributes around. `generic_slip2srf` treats its `risetime` as the
    /// unstretched value instead, so its ramp can only lengthen and the realised
    /// average comes out above what was asked for. Reading it as the average is what
    /// makes a single-subfault source rise in exactly this many seconds whatever its
    /// depth.
    #[new]
    #[pyo3(signature = (
        magnitude, rise_time_s, *,
        average_dip_deg, average_rake_deg, use_moment_magnitude = true,
    ))]
    fn new(
        magnitude: f64,
        rise_time_s: f64,
        average_dip_deg: f64,
        average_rake_deg: f64,
        use_moment_magnitude: bool,
    ) -> PyResult<Self> {
        if rise_time_s <= 0.0 {
            return Err(PyValueError::new_err(
                "rise_time_s must be strictly positive",
            ));
        }
        Ok(Self {
            inner: realisation::PointSourceSpec {
                magnitude,
                magnitude_scale: if use_moment_magnitude {
                    MagnitudeScale::Moment
                } else {
                    MagnitudeScale::Local
                },
                average_dip_deg,
                average_rake_deg,
                rise_time_s,
            },
        })
    }
}

/// How the slip and rake fields are shaped and trimmed.
#[pyclass(skip_from_py_object, module = "rupture_generator._core")]
#[derive(Clone, Copy, Debug)]
pub struct SlipSpec {
    inner: realisation::SlipSpec,
    model: SpectrumModel,
}

#[pymethods]
impl SlipSpec {
    /// Build the slip and rake field description.
    ///
    /// `coefficient_of_variation` is the **slip** field's spread, dimensionless.
    /// `rake_sigma_deg` is the **rake** field's, in degrees. They are different
    /// quantities in different units, they are both spreads of a field drawn through
    /// the same spectrum, and for a long time this constructor had only the first —
    /// so the rake field silently took it. See `DEFECTS.md` 14.
    #[new]
    #[pyo3(signature = (
        model, *, coefficient_of_variation = 0.75, rake_sigma_deg = 15.0,
        min_wavelength_km = 1.5, max_wavelength_km = 80.0,
        strike_shift = 0.0, dip_shift = 0.0,
        side_taper = 0.02, top_taper = 0.0, bottom_taper = 0.0,
        truncate_negative = true, water_level = 0.0,
    ))]
    #[expect(clippy::too_many_arguments, reason = "one flat constructor per group")]
    fn new(
        model: SpectrumModel,
        coefficient_of_variation: f64,
        rake_sigma_deg: f64,
        min_wavelength_km: f64,
        max_wavelength_km: f64,
        strike_shift: f64,
        dip_shift: f64,
        side_taper: f64,
        top_taper: f64,
        bottom_taper: f64,
        truncate_negative: bool,
        water_level: f64,
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
                rake_sigma_deg,
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
        slip_rate_shape = None,
        sample_interval_s = 0.005, max_samples = 100_000,
    ))]
    #[expect(clippy::too_many_arguments, reason = "one flat constructor per group")]
    const fn new(
        rupture_time_correlation: f64,
        rupture_time_sigma: f64,
        rupture_time_scale: f64,
        rupture_delay_s: f64,
        rise_time_correlation: f64,
        rise_time_sigma: f64,
        rise_time_blend: Ramp,
        slip_exponent: f64,
        shallow_ramp: Ramp,
        shallow_rise_factor: f64,
        deep_ramp: Ramp,
        deep_rise_factor: f64,
        shallow_speed_ramp: Option<Ramp>,
        deep_speed_ramp: Option<Ramp>,
        shallow_speed_factor: f64,
        deep_speed_factor: f64,
        weighting: RiseTimeWeighting,
        beta_shallow_ramp: Ramp,
        beta_shallow: f64,
        beta_mid_ramp: Ramp,
        beta_mid: f64,
        beta_deep: f64,
        slip_rate_shape: Option<SlipRateShape>,
        sample_interval_s: f64,
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
                // `OliuP2` is genslip's finite-fault shape and stays the default, so
                // the other ten are opt-in rather than something a caller can select
                // by accident.
                slip_rate_shape: match slip_rate_shape {
                    Some(shape) => shape.inner,
                    None => Shape::OliuP2,
                },
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
    slip_cm: Vec<f64>,
    rake_deg: Vec<f64>,
    onset_s: Vec<f64>,
    rise_time_s: Vec<f64>,
    slip_rate: Vec<f64>,
    slip_rate_offsets: Vec<u64>,
    strike_count: usize,
    dip_count: usize,
    moment_dyne_cm: f64,
    alpha_t: f64,
    sample_interval_s: f64,
}

impl GeneratedRupture {
    fn from_model(model: &RuptureModel, sample_interval_s: f64) -> Self {
        let mut slip_rate = Vec::new();
        let mut slip_rate_offsets = Vec::with_capacity(model.slip_rate.len() + 1);
        slip_rate_offsets.push(0);
        for pulse in &model.slip_rate {
            slip_rate.extend_from_slice(pulse.as_slice());
            slip_rate_offsets.push(slip_rate.len() as u64);
        }

        Self {
            slip_cm: model.slip.slip.flat().to_vec(),
            rake_deg: model.rake_deg.flat().to_vec(),
            onset_s: model.onset_s.flat().to_vec(),
            rise_time_s: model.rise_time_s.flat().to_vec(),
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
    fn slip_cm<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.slip_cm.clone().into_pyarray(py)
    }

    /// Rake in degrees, one per subfault.
    #[getter]
    fn rake_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.rake_deg.clone().into_pyarray(py)
    }

    /// Rupture onset in seconds, zero at the hypocentre.
    #[getter]
    fn onset_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.onset_s.clone().into_pyarray(py)
    }

    /// Rise time in seconds, one per subfault.
    #[getter]
    fn rise_time_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.rise_time_s.clone().into_pyarray(py)
    }

    /// Every slip-rate sample, concatenated. Index it with `slip_rate_offsets`.
    #[getter]
    fn slip_rate<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
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
    const fn sample_interval_s(&self) -> f64 {
        self.sample_interval_s
    }

    /// Seismic moment in dyne-cm.
    #[getter]
    const fn moment_dyne_cm(&self) -> f64 {
        self.moment_dyne_cm
    }

    /// The dip-and-rake correction that was applied to rise time and rupture speed.
    #[getter]
    const fn alpha_t(&self) -> f64 {
        self.alpha_t
    }

    /// `(strike_count, dip_count)`.
    #[getter]
    const fn shape(&self) -> (usize, usize) {
        (self.strike_count, self.dip_count)
    }

    /// Slip reshaped to `(dip, strike)`, which is C order for this layout.
    fn slip_grid<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
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
    values: &[f64],
    strike_count: usize,
    dip_count: usize,
) -> PyResult<numpy::ndarray::Array2<f64>> {
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
    let sample_interval_s = timing.inner.sample_interval_s;

    // Nothing below touches a Python object.
    let model = py
        .detach(|| {
            let mut draws = GenslipLcg::new(seed).realisation(realisation);
            let mut fft = RustFft::new();
            let mut solver = FactoredSweep::new();
            realisation::generate(
                &mut draws,
                &mut fft,
                &mut solver,
                &grid.inner,
                &velocity_model.inner,
                source.inner,
                slip.inner,
                &timing.inner,
                Hypocentre {
                    strike: hypocentre_strike,
                    dip: hypocentre_dip,
                },
            )
        })
        .map_err(Refused)?;

    Ok(GeneratedRupture::from_model(&model, sample_interval_s))
}

/// Generate a point source: the same rupture model, with nothing drawn.
///
/// A point source is a plane whose slip, rake and perturbations are constant, so this
/// runs the same assembler [`generate_rupture`] does with those four fields built
/// rather than drawn. There is **no seed and no realisation**, because there is
/// nothing random: the same inputs give bit-identical output every time.
///
/// Two things differ from `generic_slip2srf`, deliberately. Onset is solved for from
/// the hypocentre rather than written as one number everywhere — at a single subfault
/// the two agree exactly, and across a discretised plane this has a rupture front
/// where the C has none. And `rise_time_s` is the fault-wide average rather than an
/// unstretched floor.
///
/// The GIL is released for the generation itself.
#[pyfunction]
#[pyo3(signature = (
    grid, velocity_model, point_source, timing, *,
    hypocentre_strike, hypocentre_dip,
))]
fn generate_point_source(
    py: Python<'_>,
    grid: &FaultGrid,
    velocity_model: &VelocityModel1D,
    point_source: &PointSourceSpec,
    timing: &TimingSpec,
    hypocentre_strike: usize,
    hypocentre_dip: usize,
) -> PyResult<GeneratedRupture> {
    let sample_interval_s = timing.inner.sample_interval_s;

    // Nothing below touches a Python object.
    let model = py
        .detach(|| {
            realisation::point_source(
                &mut FactoredSweep::new(),
                &grid.inner,
                &velocity_model.inner,
                point_source.inner,
                &timing.inner,
                Hypocentre {
                    strike: hypocentre_strike,
                    dip: hypocentre_dip,
                },
            )
        })
        .map_err(Refused)?;

    Ok(GeneratedRupture::from_model(&model, sample_interval_s))
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SpectrumModel>()?;
    module.add_class::<RiseTimeWeighting>()?;
    module.add_class::<SlipRateShape>()?;
    module.add_class::<Ramp>()?;
    module.add_class::<FaultGrid>()?;
    module.add_class::<VelocityModel1D>()?;
    module.add_class::<SourceSpec>()?;
    module.add_class::<PointSourceSpec>()?;
    module.add_class::<SlipSpec>()?;
    module.add_class::<TimingSpec>()?;
    module.add_class::<GeneratedRupture>()?;
    module.add_function(wrap_pyfunction!(generate_rupture, module)?)?;
    module.add_function(wrap_pyfunction!(generate_point_source, module)?)?;
    Ok(())
}
