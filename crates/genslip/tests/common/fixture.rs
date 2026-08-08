//! One fault, configured the way the workflow configures faults.
//!
//! The numbers are genslip's own defaults where it has them and `crustal_small`'s
//! where it does not, so a test written here and a corpus case disagree about the
//! rupture rather than about the input.
//!
//! Everything is a plain function returning an owned value, so a test that wants a
//! variation takes one and edits the field it cares about:
//!
//! ```ignore
//! let mut slip = fixture::slip_spec();
//! slip.spectrum.shape = Spectrum2D::Frankel;
//! ```
//!
//! That is deliberately not a builder. A builder would need a method per field and
//! would have to be edited every time a spec gains one; struct update does the same
//! job and cannot fall behind.

use genslip::field::{CorrelationLengths, Spectrum2D, WavelengthBand};
use genslip::realisation::{FaultGrid, SlipSpec, SourceSpec, TimingSpec};
use genslip::rise_time::{DepthRamp, RiseTimeSpec, RiseTimeStretch, Weighting};
use genslip::rupture::{Hypocentre, SpeedProfile};
use genslip::slip::{GridExtents, PerturbationSpec, SpectrumSpec, SubfaultSpacing};
use genslip::slip_rate::{BetaProfile, OliuP2, SlipRateShape};
use genslip::source::{CornerRelation, Layer, MagnitudeScale, VelocityModel};
use genslip::taper::EdgeTapers;

/// Subfaults along strike, down dip, and the padded extents around them.
///
/// 24x14 is small enough to run in milliseconds and large enough that every depth
/// ramp has subfaults on both sides of it — which matters, because a ramp with no
/// subfault inside its transition is a ramp no test can see.
pub const STRIKE_COUNT: usize = 24;
pub const DIP_COUNT: usize = 14;
pub const PADDED_STRIKE: usize = 28;
pub const PADDED_DIP: usize = 16;

/// A deliberately **non-square** fault, for the properties a square one hides.
///
/// Any statistic symmetric in the two axes — mean, moment, coefficient of variation,
/// the radially-pooled power spectrum — is invariant under transposing the grid. So
/// is a square fixture. Tests that care which axis is which use this.
pub const WIDE_STRIKE: usize = 40;
pub const WIDE_DIP: usize = 10;
pub const WIDE_PADDED_STRIKE: usize = 44;
pub const WIDE_PADDED_DIP: usize = 12;

/// The default fault: 24x14 at 1 km, from 0.5 km down to 20 km.
///
/// The depth range is chosen to straddle every ramp in `timing_spec`: the shallow
/// rise-time and rupture-speed transition at 6.5 +/- 1.5 km, the deep one at
/// 17.5 +/- 2.5 km, and the beta transitions at 2.0 and 6.5 km.
#[must_use]
pub fn fault() -> FaultGrid {
    fault_of(STRIKE_COUNT, DIP_COUNT, PADDED_STRIKE, PADDED_DIP)
}

/// The non-square fault, for axis-sensitive properties.
#[must_use]
pub fn wide_fault() -> FaultGrid {
    fault_of(WIDE_STRIKE, WIDE_DIP, WIDE_PADDED_STRIKE, WIDE_PADDED_DIP)
}

/// A fault of any shape, with the fixture's depths, rake and velocity fraction.
///
/// # Panics
///
/// If the fault does not fit inside the padding, or either padded extent is odd —
/// the spectral generators address the Nyquist row and column directly.
#[must_use]
pub fn fault_of(
    strike_count: usize,
    dip_count: usize,
    padded_strike: usize,
    padded_dip: usize,
) -> FaultGrid {
    assert!(
        strike_count <= padded_strike && dip_count <= padded_dip,
        "a {strike_count}x{dip_count} fault does not fit in {padded_strike}x{padded_dip}"
    );
    assert!(
        padded_strike.is_multiple_of(2) && padded_dip.is_multiple_of(2),
        "padded extents must be even"
    );

    FaultGrid {
        extents: GridExtents {
            fault_strike: strike_count,
            fault_dip: dip_count,
            padded_strike,
            padded_dip,
        },
        spacing: SubfaultSpacing {
            strike_km: 1.0,
            dip_km: 1.0,
        },
        depth_km: (0..dip_count)
            .map(|dip| 0.5 + genslip::units::exact(dip) * 1.5)
            .collect(),
        base_rake_deg: genslip::grid::from_values(
            strike_count,
            dip_count,
            vec![175.0; strike_count * dip_count],
        ),
        velocity_fraction: genslip::grid::from_values(
            strike_count,
            dip_count,
            vec![0.8; strike_count * dip_count],
        ),
    }
}

/// Four crustal layers, shared with the corpus's `CRUSTAL_LAYERS`.
#[must_use]
pub fn velocity_model() -> VelocityModel {
    VelocityModel::new(vec![
        Layer {
            bottom_depth_km: 1.0,
            shear_speed_km_s: 1.8,
            density_g_cm3: 2.1,
        },
        Layer {
            bottom_depth_km: 5.0,
            shear_speed_km_s: 2.6,
            density_g_cm3: 2.4,
        },
        Layer {
            bottom_depth_km: 12.0,
            shear_speed_km_s: 3.2,
            density_g_cm3: 2.6,
        },
        Layer {
            bottom_depth_km: 30.0,
            shear_speed_km_s: 3.6,
            density_g_cm3: 2.7,
        },
    ])
}

/// M6.8 on the Mai corner relation, dipping 60 degrees with a 175-degree rake.
#[must_use]
pub fn source_spec() -> SourceSpec {
    SourceSpec {
        magnitude: 6.8,
        magnitude_scale: MagnitudeScale::Moment,
        corners: CornerRelation::Mai {
            strike_offset: 2.50,
            dip_offset: 1.50,
            circular: false,
        },
        modified_corners: false,
        rise_time_coefficient: 1.6,
        average_dip_deg: 60.0,
        average_rake_deg: 175.0,
    }
}

/// The slip field's spread is `0.75` **dimensionless**; the rake field's is `15`
/// **degrees**. Confusing the two is `DEFECTS.md` 14, so they are never both bare
/// numbers in the same expression here.
#[must_use]
pub fn slip_spec() -> SlipSpec {
    SlipSpec {
        spectrum: spectrum_spec(),
        tapers: EdgeTapers {
            sides: 0.02,
            top: 0.0,
            bottom: 0.0,
        },
        truncate_negative: true,
        water_level: 0.0,
        rake_sigma_deg: 15.0,
    }
}

/// The spectrum on its own, for tests that drive a kernel rather than a rupture.
///
/// The correlation lengths here are a **placeholder**: `realisation::generate`
/// overwrites them from the magnitude relation. A test calling `generate_normalised`
/// directly has to set them itself, and [`corner_lengths`] is what `generate` would
/// have used.
#[must_use]
pub fn spectrum_spec() -> SpectrumSpec {
    SpectrumSpec {
        shape: Spectrum2D::Mai,
        correlation: CorrelationLengths {
            strike: 1.0,
            dip: 1.0,
        },
        band: WavelengthBand::new(1.5, 80.0),
        coefficient_of_variation: 0.75,
        phase_shift: (0.0, 0.0),
    }
}

/// The correlation lengths the fixture's magnitude and relation imply, in km.
///
/// **Anisotropic** — along-strike is roughly twice down-dip at M6.8 — which is what
/// makes the fast-axis contract test able to tell the two directions apart.
#[must_use]
pub fn corner_lengths() -> CorrelationLengths {
    let spec = source_spec();
    let corners = genslip::source::correlation_lengths(spec.magnitude, spec.corners, false);
    CorrelationLengths {
        strike: corners.strike_km,
        dip: corners.dip_km,
    }
}

/// genslip's timing defaults, with the four ramp pairs kept distinct.
///
/// The rise-time and rupture-speed profiles share their configured centres and
/// half-widths, which is why `DEFECTS.md` 13 — one ramp pair reaching both — went
/// unnoticed. They are separate values here so a test that moves one moves only one.
#[must_use]
pub fn timing_spec() -> TimingSpec {
    let shallow = DepthRamp {
        centre_km: 6.5,
        half_width_km: 1.5,
    };
    let deep = DepthRamp {
        centre_km: 17.5,
        half_width_km: 2.5,
    };
    TimingSpec {
        rupture_time: PerturbationSpec {
            correlation: 0.8,
            sigma: 1.0,
        },
        rupture_time_scale: -0.35,
        rupture_delay_s: 0.0,
        rise_time: RiseTimeSpec {
            perturbation: PerturbationSpec {
                correlation: 0.9,
                sigma: 0.75,
            },
            shallow_blend: DepthRamp {
                centre_km: 2.0,
                half_width_km: 1.0,
            },
            slip_exponent: 0.5,
        },
        rise_time_stretch: RiseTimeStretch {
            shallow,
            shallow_factor: 2.0,
            deep,
            deep_factor: 2.0,
        },
        rise_time_weighting: Weighting::Uniform,
        slip_rate_shape: SlipRateShape::from(OliuP2),
        speed_profile: SpeedProfile {
            shallow,
            shallow_factor: 0.6,
            deep,
            deep_factor: 0.6,
        },
        beta: BetaProfile {
            shallow_ramp: DepthRamp {
                centre_km: 2.0,
                half_width_km: 1.0,
            },
            shallow: 0.5,
            mid_ramp: DepthRamp {
                centre_km: 6.5,
                half_width_km: 1.5,
            },
            mid: 0.13,
            deep: 0.13,
        },
        sample_interval_s: 0.005,
        max_samples: 100_000,
    }
}

/// Mid-fault along strike, below the shallow ramp and above the deep one.
///
/// Deliberately not the centre in both directions: an off-centre hypocentre is what
/// makes an axis transposition visible, and `(12, 8)` on a 24x14 fault is
/// asymmetric in dip.
#[must_use]
pub const fn hypocentre() -> Hypocentre {
    Hypocentre { strike: 12, dip: 8 }
}

/// The seed the fixture tests use unless they are varying it.
pub const SEED: i64 = 20_260_807;
