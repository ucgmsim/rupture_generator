//! What the magnitude and the fault geometry imply, before any field is drawn.
//!
//! Everything the generators need that is not itself random: the seismic moment the
//! rupture has to carry, the wavenumber corners its slip spectrum turns over at, how
//! long the average subfault slips for, and the elastic properties at each depth.

use crate::error::{Error, Result};
use crate::grid::SlipField;
use crate::units;
use ndarray::{Array1, Axis};

/// Natural log of ten, the base of every magnitude relation here.
///
/// The original computes it as `log(10.0)` at run time. `f64::consts::LN_10` is the
/// same value to the last bit — the C's `log` is correctly rounded for an exact
/// argument — so this substitution is free.
const LN_10: f64 = std::f64::consts::LN_10;

/// Which magnitude convention a value is on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MagnitudeScale {
    /// Moment magnitude, Hanks & Kanamori (1979): `log10(M0) = 1.5*Mw + 16.1`.
    Moment,
    /// The variant genslip labels plain `M`, differing only in the constant.
    Local,
}

impl MagnitudeScale {
    /// The additive constant in `log10(M0) = 1.5*(mag + c)`.
    const fn coefficient(self) -> f64 {
        match self {
            Self::Moment => 10.73,
            Self::Local => 10.7,
        }
    }
}

/// Seismic moment in dyne-cm, from magnitude.
///
/// (orig. `genslip_v5.6.2.c:1250`)
#[must_use]
pub fn seismic_moment(magnitude: f64, scale: MagnitudeScale) -> f64 {
    (LN_10 * 1.5 * (magnitude + scale.coefficient())).exp()
}

/// Which relation sets the wavenumber corners of the slip spectrum.
///
/// The corners are where the spectrum turns over from flat to falling — physically,
/// the largest scale on which slip is still correlated. Every relation here is a
/// power law in magnitude; they differ in the exponents and the offsets, which come
/// from different inversions of different earthquake catalogues.
#[derive(Clone, Copy, Debug)]
pub enum CornerRelation {
    /// Somerville et al. (1999). Both corners scale as `10^(0.5*M)`.
    ///
    /// The `0.79818` subtracted from each offset is `log10(2*pi)`. genslip's comment
    /// argues at length that Somerville's corners carry that factor where Mai and
    /// Beroza's do not, and that removing it is what lets the two be compared
    /// without an ad-hoc adjustment.
    Somerville { circular: bool },
    /// Mai & Beroza (2002). The down-dip corner scales as `10^(M/3)`, not `10^(M/2)`.
    Mai {
        strike_offset: f64,
        dip_offset: f64,
        circular: bool,
    },
    /// Suzuki: along strike as Mai, down dip clamped above a saturation magnitude.
    Suzuki {
        strike_offset: f64,
        dip_offset: f64,
        saturation_magnitude: f64,
    },
    /// Offsets and magnitude exponents supplied directly.
    Given {
        strike_offset: f64,
        dip_offset: f64,
        strike_exponent: f64,
        dip_exponent: f64,
    },
}

/// Correlation lengths of the slip field, in kilometres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrelationLengths {
    pub strike_km: f64,
    pub dip_km: f64,
}

/// `10^(exponent*magnitude - offset)`, evaluated as the original evaluates it.
///
/// `offset` is `f64` because the two relations that supply it differ in width, and
/// the difference is visible: Somerville's offsets are inline `double` literals,
/// while Mai's and Suzuki's come from `float` getpar variables and are widened at
/// the call. Callers convert accordingly.
fn power_law(magnitude: f64, exponent: f64, offset: f64) -> f64 {
    (LN_10 * (exponent * magnitude - offset)).exp()
}

/// The correlation lengths a magnitude implies.
///
/// `modified_corners` overrides the relation entirely with `10^(0.5*M - 2.0)` in both
/// directions. It is applied *after* every branch in the original, so it silently
/// wins over the chosen relation rather than being an alternative to it.
///
/// (orig. `genslip_v5.6.2.c:1303-1370`)
#[must_use]
pub fn correlation_lengths(
    magnitude: f64,
    relation: CornerRelation,
    modified_corners: bool,
) -> CorrelationLengths {
    /// `log10(2*pi)`, subtracted from both Somerville offsets. See the variant's note.
    const TWO_PI_DECADES: f64 = 0.79818;

    if modified_corners {
        let length = power_law(magnitude, 0.5, 2.00);
        return CorrelationLengths {
            strike_km: length,
            dip_km: length,
        };
    }

    match relation {
        CornerRelation::Somerville { circular } => {
            // The original subtracts the two offsets separately --
            // `0.5*mag - 1.72 - 0.79818` -- and `(a - b) - c` is not `a - (b + c)`
            // in floating point. Folding them would move every corner.
            let somerville =
                |offset: f64| (LN_10 * (0.5 * magnitude - offset - TWO_PI_DECADES)).exp();

            if circular {
                let length = somerville(1.825);
                CorrelationLengths {
                    strike_km: length,
                    dip_km: length,
                }
            } else {
                CorrelationLengths {
                    strike_km: somerville(1.72),
                    dip_km: somerville(1.93),
                }
            }
        }
        CornerRelation::Mai {
            strike_offset,
            dip_offset,
            circular,
        } => {
            let strike_km = power_law(magnitude, 0.5, strike_offset);
            // 0.3333, not 1/3. Reproduced: the difference is in the fourth decimal
            // of the exponent, which at M8 is a percent of the corner.
            let dip_km = if circular {
                strike_km
            } else {
                power_law(magnitude, 0.3333, dip_offset)
            };
            CorrelationLengths { strike_km, dip_km }
        }
        CornerRelation::Suzuki {
            strike_offset,
            dip_offset,
            saturation_magnitude,
        } => CorrelationLengths {
            strike_km: power_law(magnitude, 0.5, strike_offset),
            dip_km: power_law(magnitude.min(saturation_magnitude), 0.5, dip_offset),
        },
        CornerRelation::Given {
            strike_offset,
            dip_offset,
            strike_exponent,
            dip_exponent,
        } => CorrelationLengths {
            strike_km: power_law(magnitude, strike_exponent, strike_offset),
            dip_km: power_law(magnitude, dip_exponent, dip_offset),
        },
    }
}

/// The rise-time and rupture-speed correction for dip and rake.
///
/// Graves & Pitarka's 2010 model was calibrated on strike-slip earthquakes. A
/// shallow-dipping reverse fault ruptures differently: the free surface is closer to
/// the whole fault plane, so slip is faster and the pulse shorter. `alpha_t` shortens
/// rise time and raises rupture speed together, by the same factor, as a function of
/// how far the geometry is from vertical strike-slip.
///
/// Both factors are 1 for a vertical strike-slip fault, so `alpha_t` is 1 and nothing
/// changes — which is the calibration point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeometryCorrection {
    /// `alpha_t` itself, in `[1/(1+c), 1]`.
    pub alpha_t: f64,
    /// How far from vertical the fault dips, 0 at 90 degrees and 1 at or below 45.
    pub dip_factor: f64,
    /// How close the rake is to pure reverse, 1 at 90 degrees and 0 at 0 or 180.
    pub rake_factor: f64,
}

/// The coefficient in `alpha_t = 1/(1 + f_D*f_R*c)`.
///
/// **Hardwired** at `genslip_v5.6.2.c:1416`, despite a comment 1200 lines earlier
/// listing `Calpha` as a `getpar` variable. The workflow configuration carries a
/// `corner_frequency_alpha` of 0.1 that never reaches this program.
///
/// Worth knowing because the sibling HF port had the same constant read from a deck,
/// with a sentinel of `-99.0` meaning "use the default" — and when its deck reader
/// was deleted, the sentinel went through literally and produced negative corner
/// frequencies for every non-strike-slip fault.
const ALPHA_COEFFICIENT: f64 = 0.1;

/// Where the dip factor stops falling: at or below this the fault is flat enough that
/// the correction is fully on. (`genslip_v5.6.2.c:1421`)
const DIP_FACTOR_PLATEAU_DEG: f64 = 45.0;
/// A vertical fault. The dip factor is zero here and the correction does nothing.
const VERTICAL_DIP_DEG: f64 = 90.0;
/// Pure reverse, where the rake factor peaks. (`genslip_v5.6.2.c:1432`)
const REVERSE_RAKE_DEG: f64 = 90.0;

/// Compute the dip and rake correction.
///
/// `average_rake_deg` is averaged over the fault before being wrapped into
/// `[-180, 180]`, so a fault straddling the wrap gives a mean that is not the mean of
/// the angles. Reproduced.
///
/// # Errors
///
/// [`Error::DipOutOfRange`] if `average_dip_deg` is outside 0–90.
///
/// **The original returns a factor of zero instead**, which is not an error value: it
/// is a rupture whose rise time and rupture speed are silently uncorrected, and
/// nothing downstream can tell it apart from a vertical fault. A dip of 120° is not a
/// fault plane, so this says so. See `error.rs`.
///
/// The *rake* has no such branch and needs none — it is wrapped into `[-180, 180]`
/// first, so every input lands in range by construction.
///
/// (orig. `genslip_v5.6.2.c:1416-1444`)
pub fn geometry_correction(
    average_dip_deg: f64,
    average_rake_deg: f64,
) -> Result<GeometryCorrection> {
    if !(0.0..=VERTICAL_DIP_DEG).contains(&average_dip_deg) {
        return Err(Error::DipOutOfRange {
            degrees: average_dip_deg,
        });
    }

    // One at or below the plateau, falling linearly to zero at vertical.
    let dip_factor = if average_dip_deg > DIP_FACTOR_PLATEAU_DEG {
        1.0 - (average_dip_deg - DIP_FACTOR_PLATEAU_DEG)
            / (VERTICAL_DIP_DEG - DIP_FACTOR_PLATEAU_DEG)
    } else {
        1.0
    };

    let mut rake = average_rake_deg;
    while rake < -180.0 {
        rake += 360.0;
    }
    while rake > 180.0 {
        rake -= 360.0;
    }

    // The original spells the magnitude `sqrt((r-90)*(r-90))`. That is `abs`, and
    // provably so rather than approximately: `sqrt(fl(x*x)) == |x|` exactly under
    // round-to-nearest wherever `x*x` neither overflows nor underflows, which on a
    // rake offset bounded by 90 it cannot. See `tests/float_identities.rs`.
    let rake_factor = if (0.0..=180.0).contains(&rake) {
        1.0 - (rake - REVERSE_RAKE_DEG).abs() / REVERSE_RAKE_DEG
    } else {
        // Only reachable for a negative rake -- normal faulting -- where the original
        // gives zero and means it: the correction is for reverse-slip geometries.
        0.0
    };

    let alpha_t = 1.0 / (1.0 + dip_factor * rake_factor * ALPHA_COEFFICIENT);

    Ok(GeometryCorrection {
        alpha_t,
        dip_factor,
        rake_factor,
    })
}

/// Average rise time in seconds, from moment.
///
/// `M0^(1/3)` scaling: rise time grows with the cube root of moment, which is the
/// same as saying it grows linearly with fault dimension.
///
/// The `alpha_t` correction is **not** applied here; the caller multiplies, because
/// the same factor also divides the rupture-speed fraction and doing both in one
/// place keeps them from drifting apart.
///
/// (orig. `genslip_v5.6.2.c:1412`)
#[must_use]
pub fn average_rise_time(moment_dyne_cm: f64, coefficient: f64) -> f64 {
    // `cbrt` rather than the original's `exp(log(M0)/3)`: one call instead of two, and
    // exact where the pair is not -- it lands on 3 for 27, which the exp/log pair
    // misses. `tests/float_identities.rs` pins that difference.
    coefficient * units::RISE_TIME_MOMENT_SCALE * moment_dyne_cm.cbrt()
}

/// One layer of a one-dimensional velocity model.
#[derive(Clone, Copy, Debug)]
pub struct Layer {
    /// Depth to the **bottom** of the layer, in km.
    pub bottom_depth_km: f64,
    /// Shear-wave speed, in km/s.
    pub shear_speed_km_s: f64,
    /// Density, in g/cm³.
    pub density_g_cm3: f64,
}

impl Layer {
    /// Rigidity in CGS units (dyne/cm²), `rho * vs^2`.
    #[must_use]
    pub fn rigidity(self) -> f64 {
        self.shear_speed_km_s * self.shear_speed_km_s * self.density_g_cm3 * units::RIGIDITY_SCALE
    }
}

/// A layered velocity model.
#[derive(Clone, Debug)]
pub struct VelocityModel {
    layers: Vec<Layer>,
}

impl VelocityModel {
    /// Build from layers ordered shallow to deep.
    ///
    /// # Panics
    ///
    /// If there are no layers.
    #[must_use]
    pub fn new(layers: Vec<Layer>) -> Self {
        assert!(
            !layers.is_empty(),
            "a velocity model needs at least one layer"
        );
        Self { layers }
    }

    /// The layers, shallow to deep.
    ///
    /// Read-only, because a model is validated on construction and nothing downstream
    /// has a reason to edit one in place. The Python binding needs it to hand back
    /// what it was given — a model the caller cannot read is one they have to keep a
    /// second copy of, and two copies drift.
    #[must_use]
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// The layer containing `depth_km`, or the deepest one if it falls below them all.
    ///
    /// Clamping rather than extrapolating is the original's behaviour and the right
    /// one: a subfault below the model is a modelling error, not a reason to invent
    /// properties for it. Note the search is a linear scan by depth, so a subfault
    /// exactly on a boundary belongs to the layer *above* it.
    #[must_use]
    pub fn layer_at(&self, depth_km: f64) -> Layer {
        let mut index = 0;
        while depth_km > self.layers[index].bottom_depth_km && index < self.layers.len() - 1 {
            index += 1;
        }
        self.layers[index]
    }

    /// Shear speed and rigidity at every subfault.
    ///
    /// Returned as two fields rather than a struct per subfault: the moment scaling
    /// wants rigidity alone and the rupture-speed field wants shear speed alone.
    ///
    /// # Panics
    ///
    /// If `depth_km` is empty.
    ///
    /// (orig. `load_vsden`, ruptime.c)
    #[must_use]
    pub fn sample(&self, strike_count: usize, depth_km: &[f64]) -> (SlipField, SlipField) {
        // One layer lookup per dip row, broadcast along strike -- the properties are
        // constant across a row because depth is.
        let layers: Vec<Layer> = depth_km.iter().map(|depth| self.layer_at(*depth)).collect();
        let column = |of: fn(&Layer) -> f64| {
            let values: Vec<f64> = layers.iter().map(&of).collect();
            Array1::from(values)
                .insert_axis(Axis(1))
                .broadcast((depth_km.len(), strike_count))
                .expect("a column broadcasts across a row")
                .to_owned()
        };

        (
            column(|layer| layer.shear_speed_km_s),
            column(|layer| layer.rigidity()),
        )
    }
}
