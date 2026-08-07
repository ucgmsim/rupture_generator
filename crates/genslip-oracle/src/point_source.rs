//! genslip's per-subfault record (`StandRupFormat/structure.h`).
//!
//! Twenty-one fields covering geometry, slip, rupture timing, material properties,
//! asperity masks and rupture-velocity factors — everything any stage of the program
//! might want about a subfault, in one struct that every stage is handed.
//!
//! **This mirror exists so the C can be called, and for no other reason.** It is a
//! layout requirement: `ps[i].mu` only reaches the right bytes if the Rust struct
//! has the same size and field offsets. Nothing in `genslip` uses it, and nothing
//! should — a function that needs rigidity should take rigidity.

/// Layout-compatible mirror of `struct pointsource`.
///
/// All fields are four bytes wide, so there is no padding to reason about.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PointSource {
    pub lon: f32,
    pub lat: f32,
    pub dep: f32,
    pub stk: f32,
    pub dip: f32,
    pub rak: f32,
    pub area: f32,
    pub slip: f32,
    pub rupt: f32,
    pub vs: f32,
    pub den: f32,
    /// Rigidity, in CMS units. The only field `scale_slip_r_vsden` reads.
    pub mu: f32,
    pub beta: f32,
    pub asp: f32,
    pub asp_mask: i32,
    pub subevt: f32,
    pub subevt_mask: i32,
    pub aseis: f32,
    pub rvf: f32,
    pub trise_fac: f32,
    pub rvf_fac: f32,
}

const _: () = {
    // 21 four-byte fields. If this ever fails the header has changed under us, and
    // every `ps[i]` in the oracle is addressing the wrong bytes.
    assert!(size_of::<PointSource>() == 21 * 4);
    assert!(align_of::<PointSource>() == 4);
};
