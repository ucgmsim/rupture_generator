# What was pruned, and why

Every genslip v5.6.2 parameter or code path this project deliberately does **not**
carry gets one line here, naming the evidence that made it inert. A pruned branch
someone later needs is a `git revert`; a silently-pruned branch is a mystery.

Two categories, and they have different risk profiles:

- **Dead in v5.6.2** — parsed or compiled but never reachable. Removing these cannot
  change output. Safe on inspection alone.
- **Inert under `workflow`'s `root/defaults.yaml`** — live code that the configuration
  never activates. Safe only after the draw audit below, because *inert is not the same
  as silent*: several branches consume from the shared RNG stream whether or not their
  result is used.

## The draw audit is a precondition

`genslip` runs one 31-bit LCG stream (`sfrand`, `misc.c:48`) through every field
generator, and `gaus_rand` consumes exactly 12 draws per Gaussian. A branch that
contributes nothing numerically can still advance that stream, and deleting it then
changes every field generated afterwards while still producing plausible output.

Two known cases, both configured to be numerically inert and **neither prunable**:

| Path | Configured | Still draws |
| --- | --- | --- |
| Fault roughness field | `alpha_rough: 0.0` | `24 · nstk3 · (ndip3/2 + 1)` |
| `tsfac2` | `tsfac2_sigma: 1.0e-10` | `24 · nstk3 · (ndip3/2 + 1)` |

Before removing anything from the second category, run the C with and without the
branch and compare the final seed. `dump_last_seed` exists for exactly this and
`root/defaults.yaml` already enables it.

---

## Pruned: dead in v5.6.2

| Parameter | Evidence |
| --- | --- |
| `tsfac_coef` | Pre-v5.4.1 moment-scaling coefficient (GP16 eq. A2). Declared at `genslip_v5.6.2.c:561`, parsed at `:973`, never read again — `tsfac_main` is built from `tsfac_bzero` and `tsfac_slope` at `:1256`. Was `RuptureTimePerturbation.coefficient`. |
| `wavelength_max` | Parsed at `genslip_v5.6.2.c:1092`, then overwritten unconditionally with `1.0e+15` at `:1236` (`/* hardwire for now 2016-10-21 */`). No user value can reach the filters. Was `SpatialFiltering.rake_max_wavelength`. |
| `rt_rand` | Parsed into `stfparams` at `genslip_v5.6.2.c:868`. Its only reader is `load_slip_srf_dd2` (`gslip_srf_subs.c:677`), which nothing calls — `main` uses `load_slip_srf_dd5_vsden` (`genslip_v5.6.2.c:2964`), which does not mention it. Rise-time perturbation comes from the correlated `rtime1`/`rtime2` fields instead. Was `RiseTimeParameters.perturbation_sigma_ln`. |

Still to prune from the config once confirmed the same way: `use_gaus` (deprecated
2023-03-05, read and never used), `stretch_kcorner`, `rand_rake_degs`, `reload_slip`
(hardwired to 1 at `:1982`).

Also dead and not to be ported: `fourg.f` in its entirety — reachable only from
`fft2d` (`slip.c:830`), which `main` never calls; every live transform goes through
`fft2d_fftw`.

## Pruned: inert under `root/defaults.yaml`

*(Pending the draw audit — nothing removed yet.)*

## Refused rather than ignored

A parameter set to a value the port cannot honour must raise, not be dropped. This is
not hypothetical: `workflow`'s default binary path still points at `genslip_v5.4.2`,
which does not know `beta_asp`, `beta_subevt`, `beta_*_depth`, `hyb_corlen_*` or
`rtime2slip_exp`, and `getpar` never asks for names it does not recognise — so those
five have been silently discarded in production. The same class of bug produced
`calpha = -99.0` in the HF port's defaults.
