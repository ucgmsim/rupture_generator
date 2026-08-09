"""S9: slip-rate pulses, and the vocabulary that names their shapes.

Two things live here. The **vocabulary seam** -- :func:`from_stype`, which turns the
`stype` spelling a config file uses into a resolved shape -- and (from Phase 3) the
driver that hands slip, rise time and shape to ``crates/kernels``' pulse synthesis.

# The vocabulary shrank, and the seam says so

The port offered eleven shapes because genslip did. Production selects one family:
`OliuP2`, the Liu-Archuleta pulse whose rising fraction comes from a depth profile,
plus `delta` for the degenerate spike. The four proven aliases of the same kernel --
``ucsb``, ``ucsb2``, ``ucsb-varT1``, ``ucsb-T<b>`` -- collapse into parametrisations
of it: `crates/genslip`'s contract tests established, sample for sample, that each is
``oliu_p`` with the breakpoints moved, and the table in :data:`_ALIASES` is that
finding written down.

The rest -- ``brune``, ``urs``, ``esg2006``, ``cos``, ``seki`` -- are **removed**, and
the refusal says so by name. `defaults.yaml` in the production workflow advertises
them as valid ``stype`` values, so a config that names one deserves better than
"unknown shape": it gets told the shape existed and was removed, which is the
difference between a typo and a decision.

The C's own behaviour on an unrecognised ``stype`` is to fall through to ``brune``
and silently generate a different rupture. Anything not in the vocabulary is an
error here, whatever else changes.
"""

from __future__ import annotations

import dataclasses

REMOVED_SHAPES = ("brune", "urs", "esg2006", "cos", "seki")
"""Shapes the rewrite removed. Refused by name, not treated as typos, because the
production workflow's defaults file advertises them as valid spellings."""


@dataclasses.dataclass(frozen=True)
class ResolvedShape:
    """A slip-rate shape the kernel can synthesise, fully parametrised.

    Attributes
    ----------
    kernel : str
        ``"oliu_p"`` or ``"delta"`` -- the two shapes ``crates/kernels`` knows.
    duration_scale : float
        Multiplies the rise time before synthesis. 1 for the plain pulse; the
        ``ucsb2`` alias doubles the duration with the peak kept in place.
    beta : float or None
        The rising fraction, in ``(0, 0.5]``. ``None`` means "from the depth
        profile", which is what distinguishes `OliuP2` from its fixed-beta aliases.
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
"""Each alias is ``oliu_p`` with the breakpoints moved -- an identity the old
`slip_rate_contract.rs` asserted sample for sample, which is what licenses collapsing
them here instead of keeping five kernels."""


def from_stype(stype: str) -> ResolvedShape:
    """Resolve a config file's ``stype`` spelling to a kernel shape.

    Parameters
    ----------
    stype : str
        genslip's own spelling: ``OliuP2``, ``delta``, one of the ``ucsb`` aliases
        (including ``ucsb-T<b>`` with its numeric suffix), or a removed name.

    Returns
    -------
    ResolvedShape

    Raises
    ------
    ValueError
        For a removed shape, saying it was removed; for anything else, saying the
        name is unknown. The two messages differ because the reader's next step
        differs: a removed shape is a decision to revisit, an unknown one is a typo.
    """
    lowered = stype.lower()

    if lowered in REMOVED_SHAPES:
        raise ValueError(
            f"the slip-rate shape {stype!r} was removed in the pipeline rewrite: "
            "production selects the OliuP2 family, and the others go until someone "
            "asks for one with a reason. Use 'OliuP2' or 'delta'"
        )

    if lowered in _ALIASES:
        return _ALIASES[lowered]

    # `ucsb-T<b>` scales the duration by b and the rising fraction down to match --
    # the one spelling that carries a number, which is why the vocabulary is strings
    # rather than an enum.
    if lowered.startswith("ucsb-t"):
        suffix = stype[len("ucsb-T") :]
        if suffix == "":
            return ResolvedShape("oliu_p", 1.0, 0.13)
        try:
            scale = float(suffix)
        except ValueError:
            raise ValueError(
                f"the text after 'ucsb-T' must be a number, got {suffix!r}"
            ) from None
        if scale <= 0.0:
            raise ValueError(f"'ucsb-T' needs a positive scale, got {scale}")
        return ResolvedShape("oliu_p", scale, 0.13 / scale)

    raise ValueError(
        f"no slip-rate shape is spelled {stype!r}. The vocabulary is 'OliuP2', "
        "'delta', and the ucsb aliases ('ucsb', 'ucsb2', 'ucsb-varT1', 'ucsb-T<b>')"
    )


__all__ = ["REMOVED_SHAPES", "ResolvedShape", "from_stype"]
