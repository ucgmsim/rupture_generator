"""The eleven slip-rate shapes, as a tagged union.

Each has a ``type`` and, for four of them, a parameter. That is why this is a union of
dataclasses rather than an enum: ``ucsb_t`` carries a stretch and ``ucsb_var_t1`` a
ratio, and an enum member cannot. `_core.pyi` already models the same split for the same
reason, as named constructors on a class rather than as enum members.

Four of `generic_slip2srf`'s ten shapes turn out to be ``oliu_p2`` with the breakpoints
moved -- see `crates/genslip/src/slip_rate.rs`, which asserts the identity sample by
sample. They are still eleven separate names here, because a config names what the
modeller asked for rather than what it reduces to.

Adding one is a class and nothing else: the ``Discriminator`` finds subclasses, and
`__init__.py` imports this module so they exist to be found.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

from mashumaro.types import Discriminator

from rupture_generator import _core
from rupture_generator.config.core import ConfigObject
from rupture_generator.config.validation import PositiveFloat, UnitInterval


@dataclasses.dataclass
class SlipRateShapeConfig(ConfigObject):
    """Which slip-rate function every subfault gets."""

    class Config(ConfigObject.Config):
        discriminator = Discriminator(field="type", include_subtypes=True)

    def to_core(self) -> _core.SlipRateShape:
        """The compiled shape this names.

        Raises
        ------
        NotImplementedError
            Always, on the base. A shape that does not override this is a shape that
            was declared and never wired up, and saying so is better than returning a
            default that generates a different earthquake.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not say which compiled shape it is"
        )


@dataclasses.dataclass
class OliuP2(SlipRateShapeConfig):
    """genslip's finite-fault default. Its parameter comes from the beta depth profile
    rather than from here, which is what makes it the only shape with none."""

    type: Literal["oliu_p2"] = "oliu_p2"

    def to_core(self) -> _core.SlipRateShape:
        """The compiled shape this names."""
        return _core.SlipRateShape.oliu_p2()


@dataclasses.dataclass
class Ucsb(SlipRateShapeConfig):
    """``stype=ucsb``."""

    type: Literal["ucsb"] = "ucsb"

    def to_core(self) -> _core.SlipRateShape:
        """The compiled shape this names."""
        return _core.SlipRateShape.ucsb()


@dataclasses.dataclass
class Ucsb2(SlipRateShapeConfig):
    """``stype=ucsb2``."""

    type: Literal["ucsb2"] = "ucsb2"

    def to_core(self) -> _core.SlipRateShape:
        """The compiled shape this names."""
        return _core.SlipRateShape.ucsb2()


@dataclasses.dataclass
class UcsbT(SlipRateShapeConfig):
    """``ucsb-T``: the sinusoid with its rising limb stretched."""

    stretch: PositiveFloat
    type: Literal["ucsb_t"] = "ucsb_t"

    def to_core(self) -> _core.SlipRateShape:
        """The compiled shape this names."""
        return _core.SlipRateShape.ucsb_t(self.stretch)


@dataclasses.dataclass
class UcsbVarT1(SlipRateShapeConfig):
    """``ucsb-varT1``. The C defaults the ratio to 0.13, which is plain ``ucsb``."""

    tau1_ratio: UnitInterval = 0.13
    type: Literal["ucsb_var_t1"] = "ucsb_var_t1"

    def to_core(self) -> _core.SlipRateShape:
        """The compiled shape this names."""
        return _core.SlipRateShape.ucsb_var_t1(self.tau1_ratio)


@dataclasses.dataclass
class Brune(SlipRateShapeConfig):
    """Brune's pulse. Its duration is the rise time here, not the C's slip-derived
    constant -- the two differ by the ratio of two time constants, which
    `crates/genslip/tests/point_source.rs` asserts so the choice is hard to undo."""

    type: Literal["brune"] = "brune"

    def to_core(self) -> _core.SlipRateShape:
        """The compiled shape this names."""
        return _core.SlipRateShape.brune()


@dataclasses.dataclass
class Urs(SlipRateShapeConfig):
    """The shape whose depth ramp hid `DEFECTS.md`'s ``urs`` bug behind a `_ =>` arm."""

    type: Literal["urs"] = "urs"

    def to_core(self) -> _core.SlipRateShape:
        """The compiled shape this names."""
        return _core.SlipRateShape.urs()


@dataclasses.dataclass
class Esg2006(SlipRateShapeConfig):
    """``stype=esg2006``."""

    type: Literal["esg2006"] = "esg2006"

    def to_core(self) -> _core.SlipRateShape:
        """The compiled shape this names."""
        return _core.SlipRateShape.esg2006()


@dataclasses.dataclass
class Cos(SlipRateShapeConfig):
    """``stype=cos``."""

    type: Literal["cos"] = "cos"

    def to_core(self) -> _core.SlipRateShape:
        """The compiled shape this names."""
        return _core.SlipRateShape.cos()


@dataclasses.dataclass
class Seki(SlipRateShapeConfig):
    """The one shape that shifts its own onset."""

    type: Literal["seki"] = "seki"

    def to_core(self) -> _core.SlipRateShape:
        """The compiled shape this names."""
        return _core.SlipRateShape.seki()


@dataclasses.dataclass
class Delta(SlipRateShapeConfig):
    """A spike: all the slip in one sample."""

    type: Literal["delta"] = "delta"

    def to_core(self) -> _core.SlipRateShape:
        """The compiled shape this names."""
        return _core.SlipRateShape.delta()
