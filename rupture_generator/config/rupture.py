"""What the earthquake is: the input to ``rupture-generator generate``.

# One copy

The port kept three descriptions of a rupture model -- these dataclasses, a Rust spec
mirror, and a stub -- policed by a completeness test whose only job was to keep them
agreeing. Every one of the four wrong numbers found in the reduction sweep was a
disagreement between copies. This file is now the **only** copy: stages take frozen
parameter objects built from these classes, kernels take scalars and arrays, and there
is nothing left to mirror.

The one place the "configuration is the pipeline's vocabulary" rule bends is the
**hypocentre**, which is in-fault arc lengths here and a cell index in the pipeline.
That conversion is the mesh's, it is `DEFECTS.md` 17's exact subject, and it happens at
one seam -- ``mesh.RuptureMesh.cell_index`` -- with the convention written above it.

# The physics vocabulary shrank

Production selects one corner relation (``mai``), one spectral shape (``von_karman``),
and one slip-rate family (``OliuP2``). The others were documented knobs, so they are
**refused by name with a message saying they were removed** -- a reader who wrote
``model = "somerville"`` deserves to know it was a decision, not a typo. The evidence
that the config, not the output, is what adjudicates this selection is `DEFECTS.md`
11: the Mai/Somerville crossover at M7.37 makes output comparison unable to tell them
apart below that magnitude.

The corner relation is the one of the three that has a second option, and it is not a
second name: ``custom`` states the four coefficients in the file. That is deliberate.
A name asserts a published fit and nothing checks the assertion -- which is how
`DEFECTS.md` 11 happened -- whereas coefficients are what the pipeline actually uses
and are readable off the file. Anyone who needs a relation the rewrite removed writes
its numbers down rather than asking this package to remember them.
"""

from __future__ import annotations

import dataclasses
import math
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from mashumaro.codecs.json import JSONDecoder
from mashumaro.codecs.toml import TOMLDecoder
from mashumaro.codecs.yaml import YAMLDecoder
from mashumaro.types import Discriminator

from rupture_generator.config.core import ConfigObject
from rupture_generator.config.validation import (
    DepthKm,
    DipDeg,
    Magnitude,
    NonEmptyStr,
    PositiveFloat,
    RakeDeg,
    Seconds,
    UnitInterval,
    VelocityFraction,
    non_empty,
)
from rupture_generator.pulses import from_stype
from rupture_generator.sampling import CovarianceSpec, correlation_lengths

if TYPE_CHECKING:
    from rupture_generator.mesh import RuptureMesh

REMOVED_CORNER_MODELS = ("somerville", "suzuki", "given")
"""Corner relations the rewrite removed. Production's `defaults.yaml` sets
``srf.kmodel: 2`` -- Mai -- with no override on any path, so the others go by name.
What replaces them is ``custom``, which states the coefficients instead: a name is a
claim about a fit that output cannot check, and four numbers are checkable."""

CORNER_MODELS = ("mai", "custom")
"""The corner relations a source may name. ``mai`` is the published relation and takes
no coefficients; ``custom`` takes all four and no name."""

CORNER_COEFFICIENTS = ("strike_offset", "dip_offset", "strike_exponent", "dip_exponent")
"""What a corner relation *is*: an exponent and an offset per axis, in
``lambda = 10 ** (exponent * Mw - offset)`` -- see `sampling.correlation_lengths`."""

REMOVED_SPECTRUM_SHAPES = ("somerville", "frankel")
"""Spectral falloffs the rewrite removed. Von Karman (Hurst 0.75) is Mai's own
falloff, the shape ``kmodel: 2`` takes; the hybrid weights that fed the others are
inert in production."""


@dataclasses.dataclass
class RampConfig(ConfigObject):
    """A linear ramp between two depths, in kilometres."""

    centre_km: DepthKm
    half_width_km: PositiveFloat


@dataclasses.dataclass
class HypocentreConfig(ConfigObject):
    """Where the rupture starts, in the fault's own coordinates.

    Arc lengths rather than indices, and rather than the SRF's ``shyp``:
    ``strike_km`` from the ``j = 0`` end of the fault and ``dip_km`` from its top edge.
    Both are in-fault distances, so they mean the same thing whatever the fault is cut
    into -- which an index does not.

    ``fault`` names which surface the rupture nucleated on, and so which fault is the
    root of the causality tree. It lives here rather than with the propagation because
    it is a property of *this earthquake* rather than of the fault system: the same
    geometry, ruptured from a different fault, is a different tree. Omitted when the
    geometry has one surface, since there is nothing to choose.
    """

    strike_km: DepthKm
    dip_km: DepthKm
    fault: str | None = None


@dataclasses.dataclass
class VelocityModelConfig(ConfigObject):
    """A layered one-dimensional velocity model, ordered shallow to deep."""

    bottom_depth_km: list[float]
    shear_speed_km_s: list[float]
    density_g_cm3: list[float]

    def __post_init__(self) -> None:
        """Validate the fields, then the invariants between them."""
        super().__post_init__()
        lengths = {
            "bottom_depth_km": len(self.bottom_depth_km),
            "shear_speed_km_s": len(self.shear_speed_km_s),
            "density_g_cm3": len(self.density_g_cm3),
        }
        if len(set(lengths.values())) != 1:
            self.refuse(
                "bottom_depth_km",
                f"the three columns describe different numbers of layers: {lengths}",
            )
        non_empty(self.bottom_depth_km)
        for name in lengths:
            for value in getattr(self, name):
                if value <= 0.0:
                    self.refuse(
                        name, f"every layer's {name} must be positive, got {value}"
                    )


@dataclasses.dataclass
class SourceConfig(ConfigObject):
    """What the earthquake is. Tagged: a finite fault and a point source enter the
    pipeline differently -- one draws fields, the other is the constant case.

    # The source answers per segment

    A rupture over several faults asks the same five questions of each of them: what
    magnitude does this fault carry, which way does it slip, how does it dip, what
    patch structure does it have, and what does its rake field centre on. Two of the
    three sources answer from one number for the whole event, and the third answers
    from a dictionary -- so the *questions* are the same and only the answers differ,
    which is what makes them methods here rather than branches in the pipeline.

    The base answers all five from ``magnitude``, ``average_rake_deg`` and
    ``average_dip_deg``; :class:`PerFaultSourceConfig` overrides every one. The rule
    that keeps this from growing: the source answers **values**, and the pipeline does
    the arithmetic. ``alpha_t`` is physics, so it stays in `timing`, even though it is
    computed from two numbers that come from here.
    """

    class Config(ConfigObject.Config):
        discriminator = Discriminator(field="type", include_subtypes=True)

    def check_segments(self, segments: list[str]) -> None:
        """Refuse a source that does not describe this rupture's faults.

        A no-op for a source stating one magnitude for the event, which describes any
        number of faults by construction. Overridden where the source names faults and
        so can name the wrong ones.
        """

    def magnitude_of(self, segment: str) -> float:
        """The magnitude this segment carries."""
        return float(self.magnitude)  # ty: ignore[unresolved-attribute]

    def rake_of(self, segment: str) -> float:
        """This segment's mean rake, in degrees -- the mechanism, not the field.

        What `timing.alpha_t` and the rupture speed read. The rake *field*'s centre is
        :meth:`base_rake_deg_of`, which is a different number.
        """
        return float(self.average_rake_deg)  # ty: ignore[unresolved-attribute]

    def dip_of(self, segment: str, mesh: RuptureMesh) -> float:
        """This segment's mean dip, in degrees.

        Takes the chart because a source that does not state a dip reads it off the
        geometry, which is exact and one fewer thing written down twice.
        """
        return float(self.average_dip_deg)  # ty: ignore[unresolved-attribute]

    def base_rake_deg_of(self, segment: str, default_deg: float) -> float:
        """What this segment's rake *field* is centred on.

        Deliberately not :meth:`rake_of`. For a source stating one average rake, the
        field's centre is the ``[field]`` section's ``base_rake_deg`` and the average
        rake is what the geometric correction uses -- two numbers that happen to be
        175 degrees in every shipped example, so collapsing them would change every
        finite rupture's rake field with nothing going red.
        """
        return default_deg

    def covariance_of(self, segment: str) -> CovarianceSpec:
        """The patch structure this segment's magnitude implies.

        Per segment rather than per event, because correlation lengths scale with
        magnitude: a fault carrying an Mw 6.3 has smaller asperities than one carrying
        an Mw 7.9, and the event's summed magnitude would give the small fault patches
        larger than itself.
        """
        return correlation_lengths(self.magnitude_of(segment))


@dataclasses.dataclass
class FiniteSourceConfig(SourceConfig):
    """A finite fault.

    ``model`` is the corner relation only, not the spectral shape a `[slip]` section
    chooses independently as ``shape`` -- the two used to be one vocabulary, and
    nothing checked that a `[source]` and a `[slip]` section naming different
    relations agreed (`DEFECTS.md` 11).

    Two relations. ``mai`` is Mai & Beroza (2002) and carries no coefficients: they
    are the published fit, they live in `sampling.correlation_lengths`, and a file
    that overrode one of them would still be *called* mai while no longer being it.
    ``custom`` is the other way round -- no published name, and all four coefficients
    stated in the file, so what it is is readable from the file rather than from the
    version of this package that read it. That is the seam a removed relation comes
    back through: whoever has Somerville's coefficients and a reason writes them down.
    """

    magnitude: Magnitude
    average_dip_deg: DipDeg
    average_rake_deg: RakeDeg
    model: str = "mai"
    strike_offset: float | None = None
    dip_offset: float | None = None
    strike_exponent: float | None = None
    dip_exponent: float | None = None
    rise_time_coefficient: PositiveFloat = 1.6
    type: Literal["finite"] = "finite"

    def __post_init__(self) -> None:
        """Validate the fields, then the vocabulary."""
        super().__post_init__()
        if self.model in REMOVED_CORNER_MODELS:
            self.refuse(
                "model",
                f"the corner relation {self.model!r} was removed in the pipeline "
                "rewrite: production selects 'mai', and a relation of your own is "
                f"model = 'custom' with {', '.join(CORNER_COEFFICIENTS)} stated",
            )
        if self.model not in CORNER_MODELS:
            self.refuse("model", f"no corner relation is spelled {self.model!r}")
        missing = [name for name in CORNER_COEFFICIENTS if getattr(self, name) is None]
        if self.model == "mai":
            for name in CORNER_COEFFICIENTS:
                if getattr(self, name) is not None:
                    self.refuse(
                        name,
                        f"{name} is part of Mai & Beroza's published fit, so a 'mai' "
                        "relation does not take one -- overriding it would leave the "
                        "file naming a relation it is no longer using. Coefficients "
                        "of your own are model = 'custom', which takes all four",
                    )
        elif missing:
            self.refuse(
                missing[0],
                "a custom corner relation states all four of its coefficients, so "
                "that which relation it is does not depend on this package's "
                f"defaults; missing: {', '.join(missing)}",
            )
        else:
            for name in CORNER_COEFFICIENTS:
                value = getattr(self, name)
                if not math.isfinite(value):
                    self.refuse(name, f"must be a finite number, got {value}")
            for name in ("strike_exponent", "dip_exponent"):
                if getattr(self, name) < 0.0:
                    self.refuse(
                        name,
                        f"must be 0 or more, got {getattr(self, name)}: a negative "
                        "exponent gives a larger earthquake smaller asperities. Zero "
                        "is allowed, and is a correlation length of 10 ** -offset km "
                        "at every magnitude",
                    )

    def corner_coefficients(self) -> dict[str, float]:
        """The coefficients `sampling.correlation_lengths` should take, if any.

        Empty for ``mai``, which is what keeps the published numbers in exactly one
        place -- that function's own defaults -- rather than restating them here where
        the two copies could drift. Every one of the four wrong numbers the reduction
        sweep found was a disagreement between copies.
        """
        if self.model != "custom":
            return {}
        return {name: getattr(self, name) for name in CORNER_COEFFICIENTS}

    def covariance_of(self, segment: str) -> CovarianceSpec:
        """One structure for the whole event: one magnitude, one corner relation."""
        return correlation_lengths(self.magnitude, **self.corner_coefficients())


@dataclasses.dataclass
class PointSourceConfig(SourceConfig):
    """A point source.

    Not a finite source with fields left blank. There is no spectrum, so no corner
    relation, and ``rise_time_s`` is given rather than derived from the moment -- as
    the **fault-wide average**, which the depth ramp redistributes around.
    """

    magnitude: Magnitude
    rise_time_s: PositiveFloat
    average_dip_deg: DipDeg
    average_rake_deg: RakeDeg
    type: Literal["point"] = "point"

    def covariance_of(self, segment: str) -> CovarianceSpec:
        """Any positive lengths will do: a point source draws no fields.

        One cell has no structure to describe. The stages still want a spec, so this
        is the shape of "the question does not arise" -- not a claim about a spectrum.
        """
        return CovarianceSpec(1.0, 1.0)


@dataclasses.dataclass
class PerFaultSourceConfig(SourceConfig):
    """A rupture whose faults each carry a magnitude of their own.

    The other finite source states one magnitude for the event and lets the sampled
    fields decide how the moment divides between faults. This one states the division:
    each fault is scaled to its own target, and the event's magnitude is whatever they
    sum to. Both are defensible and they are different models -- a hazard model that
    derived each fault's magnitude from its own area has already decided the
    partition, and a pipeline that re-derived it would be discarding that.

    Rake is per fault for the same reason: a system that ruptures a strike-slip fault
    into a normal one has two mechanisms, and one number cannot carry both.

    Dip is **not** here. It is a property of the geometry, and every segment's mean
    dip is read from its own chart -- which is exact, and one fewer thing stated twice.
    """

    magnitudes: dict[str, float] = dataclasses.field(default_factory=dict)
    rakes: dict[str, float] = dataclasses.field(default_factory=dict)
    rise_time_coefficient: PositiveFloat = 1.6
    type: Literal["per_fault"] = "per_fault"

    def __post_init__(self) -> None:
        """Validate the fields, then the invariants between them."""
        super().__post_init__()
        if not self.magnitudes:
            self.refuse(
                "magnitudes",
                "a per-fault source needs a magnitude for each fault; with one fault "
                "and one magnitude, use a finite source",
            )
        missing = set(self.magnitudes) - set(self.rakes)
        if missing:
            self.refuse(
                "rakes",
                f"{', '.join(sorted(missing))} has a magnitude but no rake, and a "
                "fault that slips has a direction",
            )
        for name, magnitude in self.magnitudes.items():
            if not 3.0 <= magnitude <= 10.0:
                self.refuse(
                    "magnitudes",
                    f"{name} has magnitude {magnitude}, outside the [3, 10] this "
                    "generator models",
                )

    @property
    def magnitude(self) -> float:
        """The event's magnitude: what the parts sum to, in moment.

        Reported rather than configured. Summing magnitudes directly would be
        meaningless -- they are logarithms -- so this sums the moments and converts
        back, which is the only arithmetic here that means anything.
        """
        total = sum(
            10.0 ** (1.5 * (value + 6.0333003)) for value in self.magnitudes.values()
        )
        return (math.log10(total) - 9.0499505) / 1.5

    def check_segments(self, segments: list[str]) -> None:
        """Refuse magnitudes that do not name this rupture's faults, in either direction.

        Both ways round, because they are different mistakes. A magnitude for a fault
        that is not here is a name that did not match -- usually a surface that fused
        into ``name:0`` and ``name:1``. A fault with no magnitude would otherwise
        rupture carrying none, which is a fault that appears in the file and radiates
        nothing.

        Raises
        ------
        ValueError
        """
        unknown = set(self.magnitudes) - set(segments)
        if unknown:
            self.refuse(
                "magnitudes",
                f"the source gives magnitudes for {', '.join(sorted(unknown))}, which "
                f"are not segments of this rupture ({', '.join(segments)})",
            )
        missing = set(segments) - set(self.magnitudes)
        if missing:
            self.refuse(
                "magnitudes",
                f"{', '.join(sorted(missing))} has no magnitude, and a fault that "
                "ruptures carries moment",
            )

    def magnitude_of(self, segment: str) -> float:
        """This fault's own stated magnitude."""
        return float(self.magnitudes[segment])

    def rake_of(self, segment: str) -> float:
        """This fault's own stated rake."""
        return float(self.rakes[segment])

    def dip_of(self, segment: str, mesh: RuptureMesh) -> float:
        """This fault's mean dip, read off its chart.

        Dip is a property of the geometry and is not stated here -- see the class
        docstring. Exact, and one fewer thing written down twice.
        """
        return float(np.mean(mesh.strike_dip_deg()[1]))

    def base_rake_deg_of(self, segment: str, default_deg: float) -> float:
        """The fault's own rake. A system with two mechanisms centres two fields."""
        return self.rakes[segment]


def default_wavelength_band(strike_km: float, dip_km: float) -> tuple[float, float]:
    """The wavelength limits genslip picks when none is given, for this grid.

    No constant is right on two grids, which is why `SlipConfig` leaves both `None`
    rather than carrying a literal: genslip derives the low end from the grid itself,
    ``2*sqrt(dstk*ddip)/0.8`` -- 80% of the Nyquist wavelength of a grid whose spacing
    is the geometric mean of the strike and dip cell sizes, so the band-pass rolls off
    at 80% of the Nyquist *wavenumber* rather than at it exactly. The high end has no
    real bound: genslip's own is ``1.0e15``, assigned after the ``getpar`` that reads
    it, so nothing a config file says about it is ever seen.
    """
    return 2.0 * math.sqrt(strike_km * dip_km) / 0.8, 1.0e15


@dataclasses.dataclass
class SlipConfig(ConfigObject):
    """How the slip and rake fields are shaped and trimmed.

    ``coefficient_of_variation`` is the slip field's spread and is dimensionless;
    ``rake_sigma_deg`` is the rake field's and is in **degrees**. Handing one to the
    other is `DEFECTS.md` 14, which gave every rake a spread of 0.75 degrees where the
    original gives 15 -- a factor of twenty, on every fault. They are never both bare
    numbers in the same expression here, and their names carry the difference.

    ``min_wavelength_km`` and ``max_wavelength_km`` default to `None` rather than to a
    literal, because the right value depends on the grid the field is sampled on --
    see `default_wavelength_band`. Either can still be set explicitly, which is
    honoured over the derived value.
    """

    shape: str = "von_karman"
    coefficient_of_variation: PositiveFloat = 0.75
    rake_sigma_deg: PositiveFloat = 15.0
    min_wavelength_km: PositiveFloat | None = None
    max_wavelength_km: PositiveFloat | None = None
    side_taper: UnitInterval = 0.02
    top_taper: UnitInterval = 0.0
    bottom_taper: UnitInterval = 0.0

    def __post_init__(self) -> None:
        """Validate the fields, then the invariants between them."""
        super().__post_init__()
        if self.shape in REMOVED_SPECTRUM_SHAPES:
            self.refuse(
                "shape",
                f"the spectral shape {self.shape!r} was removed in the pipeline "
                "rewrite: production's corner relation takes von Karman, and the "
                "others go until someone asks for one with a reason",
            )
        if self.shape != "von_karman":
            self.refuse("shape", f"no spectral shape is spelled {self.shape!r}")
        # Only checkable when both are given -- a `None` is filled in from the grid
        # when the field is sampled, too late for this constructor to see, and always
        # self-consistent by construction (`default_wavelength_band`'s low end is
        # kilometres, its high end is 1e15).
        if (
            self.min_wavelength_km is not None
            and self.max_wavelength_km is not None
            and self.max_wavelength_km <= self.min_wavelength_km
        ):
            self.refuse(
                "max_wavelength_km",
                f"must be above min_wavelength_km ({self.min_wavelength_km}), "
                f"got {self.max_wavelength_km}",
            )


@dataclasses.dataclass
class TimingConfig(ConfigObject):
    """How rupture time and rise time relate to slip.

    ``shallow_ramp`` and ``deep_ramp`` stretch **rise time**. Rupture speed has ramps
    of its own, which default to the rise-time ones because that is the case the
    original's four independent parameters share; ``shallow_speed_ramp`` and
    ``deep_speed_ramp`` override them when they do not. `DEFECTS.md` 13 was one pair
    reaching both.
    """

    rupture_time_scale: float
    rise_time_blend: RampConfig
    shallow_ramp: RampConfig
    deep_ramp: RampConfig
    beta_shallow_ramp: RampConfig
    beta_mid_ramp: RampConfig
    rupture_time_correlation: float = 0.8
    rupture_time_sigma: PositiveFloat = 1.0
    rupture_delay_s: Seconds = 0.0
    rise_time_correlation: float = 0.9
    rise_time_sigma: PositiveFloat = 0.75
    slip_exponent: float = 0.5
    shallow_rise_factor: PositiveFloat = 2.0
    deep_rise_factor: PositiveFloat = 2.0
    shallow_speed_ramp: RampConfig | None = None
    deep_speed_ramp: RampConfig | None = None
    shallow_speed_factor: PositiveFloat = 0.6
    deep_speed_factor: PositiveFloat = 0.6
    # genslip's own `stype` spelling, parsed by `pulses.from_stype` -- including
    # `ucsb-T`'s numeric suffix, which is why this is a string and not a `Literal`.
    # The production workflow's `defaults.yaml` advertises removed shapes as valid
    # `stype` values, so the parse distinguishes "removed" from "unknown".
    slip_rate_shape: str | None = None
    beta_shallow: PositiveFloat = 0.5
    beta_mid: PositiveFloat = 0.13
    beta_deep: PositiveFloat = 0.13
    sample_interval_s: PositiveFloat = 0.005

    def __post_init__(self) -> None:
        """Validate the fields, then the one with its own vocabulary.

        Parsed here so an unrecognised or removed ``stype`` is refused when the file
        is read, naming the field, rather than partway through a generation run. The
        C falls through to ``brune`` on a name it does not know and silently produces
        a different rupture.
        """
        super().__post_init__()
        if self.slip_rate_shape is not None:
            try:
                from_stype(self.slip_rate_shape)
            except ValueError as error:
                self.refuse("slip_rate_shape", str(error))


@dataclasses.dataclass
class FieldConfig(ConfigObject):
    """The two per-subfault fields the geometry does not supply.

    Both are constants here. The stages take them per subfault, because a mesh may
    vary them; a config that could say so per subfault would need a way to address
    subfaults, which is a bigger thing than this needs to be yet.
    """

    base_rake_deg: RakeDeg = 175.0
    velocity_fraction: VelocityFraction = 0.8


@dataclasses.dataclass
class RandomConfig(ConfigObject):
    """Which stream of numbers, and where in it.

    One event seed. ``numpy.random.SeedSequence(seed)`` spawns every (stage, segment)
    pair its own named substream, so draw order inside the pipeline does not matter
    and changing one stage's parameters cannot change another's noise.
    ``realisation`` selects an independent stream from the same seed, which is what
    makes a campaign restartable.
    """

    seed: int = 1234
    realisation: int = 0

    def __post_init__(self) -> None:
        """Validate the fields, then the invariants between them."""
        super().__post_init__()
        if self.realisation < 0:
            self.refuse("realisation", f"must be 0 or more, got {self.realisation}")


DECODERS = {
    "toml": TOMLDecoder,
    "yaml": YAMLDecoder,
    "yml": YAMLDecoder,
    "json": JSONDecoder,
}
"""Which decoder an extension means. TOML is the default for anything unrecognised."""


@dataclasses.dataclass
class RuptureConfig(ConfigObject):
    """A whole generate config.

    Examples
    --------
    TOML::

        schema_version = 1
        title = "Crustal M6.2"

        [hypocentre]
        strike_km = 5.0
        dip_km = 4.0

        [velocity_model]
        bottom_depth_km  = [1.0, 5.0, 12.0, 1000.0]
        shear_speed_km_s = [1.8, 3.2, 3.5, 4.6]
        density_g_cm3    = [2.1, 2.5, 2.7, 3.2]

        [source]
        type = "finite"
        magnitude = 6.2
        average_dip_deg = 60.0
        average_rake_deg = 175.0

        [timing]
        rupture_time_scale = -0.35
        rise_time_blend   = { centre_km = 2.0,  half_width_km = 1.0 }
        shallow_ramp      = { centre_km = 6.5,  half_width_km = 1.5 }
        deep_ramp         = { centre_km = 17.5, half_width_km = 2.5 }
        beta_shallow_ramp = { centre_km = 2.0,  half_width_km = 1.0 }
        beta_mid_ramp     = { centre_km = 6.5,  half_width_km = 1.5 }

        [random]
        seed = 1234
    """

    hypocentre: HypocentreConfig
    velocity_model: VelocityModelConfig
    source: SourceConfig
    timing: TimingConfig
    slip: SlipConfig = dataclasses.field(default_factory=SlipConfig)
    field: FieldConfig = dataclasses.field(default_factory=FieldConfig)
    random: RandomConfig = dataclasses.field(default_factory=RandomConfig)
    schema_version: int = 1
    title: NonEmptyStr | None = None

    def __post_init__(self) -> None:
        """Validate the fields, then the invariants between them."""
        super().__post_init__()
        if isinstance(self.source, PointSourceConfig) and self.slip != SlipConfig():
            self.refuse(
                "slip",
                "a point source draws no fields, so a [slip] section would be read "
                "and ignored -- remove it, or use a finite source",
            )


def read_config(path: Path | str, format: str | None = None) -> RuptureConfig:
    """Read a generate config, in whichever of the three spellings it is written.

    Parameters
    ----------
    path : Path or str
        The file.
    format : str, optional
        ``"toml"``, ``"yaml"`` or ``"json"``. Inferred from the extension when
        omitted, defaulting to TOML.

    Returns
    -------
    RuptureConfig

    Raises
    ------
    InvalidFieldValue, MissingField
        If the file parses but does not describe a rupture.
    tomllib.TOMLDecodeError, json.JSONDecodeError, yaml.YAMLError
        If it does not parse. The CLI renders these differently -- a syntax error
        wants a line number and the line, and a validation error wants a key.
    """
    path = Path(path)
    chosen = format or path.suffix.lstrip(".").lower()
    return DECODERS.get(chosen, TOMLDecoder)(RuptureConfig).decode(path.read_text())


def read_geometry(path: Path | str, format: str | None = None):
    """Read a geometry config. The counterpart of :func:`read_config`.

    Returns
    -------
    GeometryConfig
    """
    from rupture_generator.config.geometry import GeometryConfig

    path = Path(path)
    chosen = format or path.suffix.lstrip(".").lower()
    return DECODERS.get(chosen, TOMLDecoder)(GeometryConfig).decode(path.read_text())


__all__ = [
    "CORNER_COEFFICIENTS",
    "CORNER_MODELS",
    "DECODERS",
    "REMOVED_CORNER_MODELS",
    "REMOVED_SPECTRUM_SHAPES",
    "FieldConfig",
    "FiniteSourceConfig",
    "HypocentreConfig",
    "PerFaultSourceConfig",
    "PointSourceConfig",
    "RampConfig",
    "RandomConfig",
    "RuptureConfig",
    "SlipConfig",
    "SourceConfig",
    "TimingConfig",
    "VelocityModelConfig",
    "default_wavelength_band",
    "read_config",
    "read_geometry",
    "tomllib",
]
