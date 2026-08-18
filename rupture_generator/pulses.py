"""S9: slip-rate pulses, and the vocabulary that names their shapes.

:func:`from_stype` turns the ``stype`` spelling a config file uses into a resolved
shape; :func:`synthesise` hands slip, rise time and shape to the kernel.

One family is available: `OliuP2`, the Liu-Archuleta pulse whose rising fraction comes
from a depth profile, plus `delta` for the degenerate spike; the ``ucsb`` aliases are
parametrisations of the same kernel. :data:`REMOVED_SHAPES` are refused **by name**
rather than falling through to a default and silently generating a different rupture.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from rupture_generator import _kernels
from rupture_generator.errors import ConfigError
from rupture_generator.stages import DepthRamp

FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]

REMOVED_SHAPES = ("brune", "urs", "esg2006", "cos", "seki")
"""Shapes no longer available. Refused by name, not treated as typos, because the
production workflow's defaults file advertises them as valid spellings."""


@dataclasses.dataclass(frozen=True)
class ResolvedShape:
    """A slip-rate shape the kernel can synthesise, fully parametrised.

    ``kernel`` is ``"oliu_p"`` or ``"delta"``, the two the kernel knows.
    ``duration_scale`` multiplies the rise time before synthesis. ``beta`` is the
    rising fraction, in ``(0, 0.5]``, or ``None`` for "from the depth profile", which
    is what distinguishes `OliuP2` from its fixed-beta aliases.
    """

    kernel: str
    duration_scale: float = 1.0
    beta: float | None = None


_ALIASES = {
    # stype           kernel     duration_scale  beta
    "oliup2": ResolvedShape("oliu_p", 1.0, None),
    "ucsb": ResolvedShape("oliu_p", 1.0, 0.13),
    "ucsb2": ResolvedShape("oliu_p", 2.0, 0.065),
    "ucsb-vart1": ResolvedShape("oliu_p", 1.0, 0.13),
    "delta": ResolvedShape("delta"),
}
"""Each alias is ``oliu_p`` with the breakpoints moved, an identity established
sample for sample, which is what licenses collapsing them into one kernel."""


def from_stype(stype: str) -> ResolvedShape:
    """Resolve a config file's ``stype`` spelling to a kernel shape.

    ``stype`` is ``OliuP2``, ``delta``, one of the ``ucsb`` aliases (including
    ``ucsb-T<b>`` with its numeric suffix), or a removed name.

    Raises
    ------
    ConfigError
        For a removed shape, saying it was removed; for anything else, saying the name
        is unknown. A removed shape is a decision to revisit, an unknown one a typo.
    """
    lowered = stype.lower()

    if lowered in REMOVED_SHAPES:
        raise ConfigError(
            f"the slip-rate shape {stype!r} was removed in the pipeline rewrite: "
            "production selects the OliuP2 family, and the others go until someone "
            "asks for one with a reason. Use 'OliuP2' or 'delta'"
        )

    if lowered in _ALIASES:
        return _ALIASES[lowered]

    # `ucsb-T<b>` scales the duration by b and the rising fraction down to match --
    # the one spelling that carries a number.
    if lowered.startswith("ucsb-t"):
        suffix = stype[len("ucsb-T") :]
        if suffix == "":
            return ResolvedShape("oliu_p", 1.0, 0.13)
        try:
            scale = float(suffix)
        except ValueError:
            raise ConfigError(
                f"the text after 'ucsb-T' must be a number, got {suffix!r}"
            ) from None
        if scale <= 0.0:
            raise ConfigError(f"'ucsb-T' needs a positive scale, got {scale}")
        return ResolvedShape("oliu_p", scale, 0.13 / scale)

    raise ConfigError(
        f"no slip-rate shape is spelled {stype!r}. The vocabulary is 'OliuP2', "
        "'delta', and the ucsb aliases ('ucsb', 'ucsb2', 'ucsb-varT1', 'ucsb-T<b>')"
    )


@dataclasses.dataclass(frozen=True)
class PulseParams:
    """How each subfault's slip-rate pulse is shaped and sampled.

    ``shape`` is already resolved by :func:`from_stype`, so the kernel never sees a
    name it has to interpret. The betas are the rising fraction by depth -- shallow
    subfaults get the largest, a longer rising limb and so a less impulsive pulse near
    the free surface -- and are ignored when the shape carries its own fixed beta.
    ``sample_interval_s`` is the pulse's own sample rate, and the SRF's ``dt``.
    """

    shape: ResolvedShape
    shallow_ramp: DepthRamp = DepthRamp(2.0, 1.0)  # noqa: RUF009
    mid_ramp: DepthRamp = DepthRamp(6.5, 1.5)  # noqa: RUF009
    beta_shallow: float = 0.5
    beta_mid: float = 0.13
    beta_deep: float = 0.13
    sample_interval_s: float = 0.005

    def beta_at(self, depth_km: FloatArray) -> FloatArray:
        """The rising fraction at each depth, ramping between the three values."""
        shallow_weight = self.shallow_ramp.weight(depth_km)
        mid_weight = self.mid_ramp.weight(depth_km)
        return (
            self.beta_shallow
            + (self.beta_mid - self.beta_shallow) * shallow_weight
            + (self.beta_deep - self.beta_mid) * mid_weight
        )


def synthesise(
    depth_km: FloatArray,
    slip_m: FloatArray,
    rise_time_s: FloatArray,
    params: PulseParams,
) -> tuple[np.ndarray, np.ndarray]:
    """S9: a slip-rate pulse for every subfault, as CSR rows.

    The kernel guarantees ``dt * sum(pulse) == slip`` exactly, whatever the shape, and
    refuses -- naming the subfault -- one that slips at a rise time its shape cannot
    sample at this interval. Emitting nothing there instead dropped 0.63% of the
    moment on the seed-1234 fixture.

    Takes **depth rather than a chart**, and reads it only when the shape's rising
    fraction is depth-dependent. Returns offsets (length subfaults + 1) and
    concatenated samples in metres per second, flattened along strike fastest.
    """
    flat_slip = np.ascontiguousarray(slip_m, dtype=np.float64).ravel()
    flat_rise = np.ascontiguousarray(rise_time_s, dtype=np.float64).ravel()

    if params.shape.kernel == "delta":
        return _kernels.synthesise_pulses(
            flat_slip, flat_rise, params.sample_interval_s, "delta"
        )

    if params.shape.beta is None:
        beta = params.beta_at(np.asarray(depth_km, dtype=np.float64)).ravel()
    else:
        beta = np.full(flat_slip.shape, params.shape.beta, dtype=np.float64)

    return _kernels.synthesise_pulses(
        flat_slip,
        flat_rise * params.shape.duration_scale,
        params.sample_interval_s,
        "oliu_p",
        beta,
    )


__all__ = [
    "REMOVED_SHAPES",
    "PulseParams",
    "ResolvedShape",
    "from_stype",
    "synthesise",
]
