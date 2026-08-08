"""Configuration: the validators, the tagged unions, and what gets refused.

Three concerns, and one deliberate omission.

**The validators.** Property-tested where the constraint is a range, because a range is
exactly the kind of thing a single example gets right by accident -- an off-by-one on an
inclusive bound is invisible unless something lands on it, and hypothesis lands on it.

**The tagged unions.** Parametrised over *every* member, generated from the union rather
than listed, so a shape added without a `to_core` cannot slip past by not being in
someone's list.

**What is refused.** Every cross-field invariant, and the misspelt key -- because
`forbid_extra_keys` is the difference between a typo being an error and being a silently
different earthquake.

Not tested: that mashumaro decodes TOML, YAML and JSON correctly. That is the library's
job and it has its own suite. What *is* tested is that the three spellings produce the
same object here, which is a claim about this schema rather than about mashumaro.
"""

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from mashumaro.exceptions import ExtraKeysError, InvalidFieldValue, MissingField

from rupture_generator import _core
from rupture_generator.config import field_path, read_config, read_geometry
from rupture_generator.config.geometry import (
    Discretisation,
    FaultConfig,
    GeometryConfig,
    LonLat,
    PlaneConfig,
    PointConfig,
)
from rupture_generator.config.rupture import (
    FiniteSourceConfig,
    PointSourceConfig,
    RuptureConfig,
    SlipConfig,
    VelocityModelConfig,
)
from rupture_generator.config.slip_rate import SlipRateShapeConfig
from rupture_generator.config.validation import (
    at_least,
    at_most,
    in_range,
    non_empty,
    non_negative,
    positive,
)

SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

REALS = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


GEOMETRY_TOML = """
schema_version = 1
crs = "EPSG:2193"

[[surfaces]]
type = "fault"
name = "kaikoura"
origin = { longitude_deg = 173.00, latitude_deg = -42.60 }
top_depth_km = 0.0

[[surfaces.planes]]
end = { longitude_deg = 173.40, latitude_deg = -42.40 }
dip_deg = 70.0
bottom_depth_km = 15.0
discretisation = { subfault_size_km = 1.0 }
"""

RUPTURE_TOML = """
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
"""


def a_rupture(**overrides) -> dict:
    """The example config as a dict, with sections replaced."""
    import tomllib

    document = tomllib.loads(RUPTURE_TOML)
    document.update(overrides)
    return document


def a_geometry(**overrides) -> dict:
    import tomllib

    document = tomllib.loads(GEOMETRY_TOML)
    document.update(overrides)
    return document


def subclasses(base: type) -> list[type]:
    """Every concrete member of a tagged union, however deeply nested.

    Generated rather than listed. A union member that nobody remembered to add to a
    parametrise list is precisely the one with the untested `to_core`.
    """
    found = []
    for child in base.__subclasses__():
        found.extend(subclasses(child))
        if not child.__subclasses__():
            found.append(child)
    return found


class TestTheValidators:
    """Ranges, property-tested, because an inclusive bound is where they go wrong."""

    @given(value=REALS)
    @SETTINGS
    def test_positive_accepts_exactly_what_is_above_zero(self, value: float) -> None:
        if value > 0.0:
            assert positive(value) == value
        else:
            with pytest.raises(ValueError, match="greater than 0"):
                positive(value)

    @given(value=REALS)
    @SETTINGS
    def test_non_negative_accepts_zero(self, value: float) -> None:
        if value >= 0.0:
            assert non_negative(value) == value
        else:
            with pytest.raises(ValueError, match="0 or more"):
                non_negative(value)

    @given(limit=REALS, value=REALS)
    @SETTINGS
    def test_at_least_and_at_most_are_inclusive(
        self, limit: float, value: float
    ) -> None:
        if value >= limit:
            assert at_least(limit)(value) == value
        else:
            with pytest.raises(ValueError, match="or more"):
                at_least(limit)(value)

        if value <= limit:
            assert at_most(limit)(value) == value
        else:
            with pytest.raises(ValueError, match="or less"):
                at_most(limit)(value)

    @given(
        low=REALS,
        width=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
        value=REALS,
    )
    @SETTINGS
    def test_in_range_is_closed_by_default(
        self, low: float, width: float, value: float
    ) -> None:
        high = low + width
        if low <= value <= high:
            assert in_range(low, high)(value) == value
        else:
            with pytest.raises(ValueError, match="must be in"):
                in_range(low, high)(value)

    @given(
        low=REALS,
        width=st.floats(min_value=0.1, max_value=1e6, allow_nan=False),
        value=REALS,
    )
    @SETTINGS
    def test_in_range_can_open_its_lower_end(
        self, low: float, width: float, value: float
    ) -> None:
        """The distinction `DipDeg` rests on: `(0, 90]`, not `[0, 90]`.

        A horizontal fault has no strike and no down-dip direction, and the node
        placement would divide by a vanishing `tan(dip)`.
        """
        high = low + width
        check = in_range(low, high, open_low=True)
        if low < value <= high:
            assert check(value) == value
        else:
            with pytest.raises(ValueError, match="must be in"):
                check(value)

    def test_the_open_lower_bound_shows_in_the_message(self) -> None:
        """So a reader can tell which kind of range they broke."""
        with pytest.raises(ValueError, match=r"must be in \(0"):
            in_range(0.0, 90.0, open_low=True)(0.0)
        with pytest.raises(ValueError, match=r"must be in \[0"):
            in_range(0.0, 90.0)(-1.0)

    @pytest.mark.parametrize("empty", [[], "", {}])
    def test_non_empty_rejects_emptiness(self, empty: object) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            non_empty(empty)

    @pytest.mark.parametrize(
        "validator",
        [positive, non_negative, at_least(1.0), at_most(1.0), in_range(0.0, 1.0)],
    )
    def test_none_passes_every_validator(self, validator: Callable) -> None:
        """An optional field is absent, not invalid.

        Without this, every `X | None` field would need its own guard, and the one that
        forgot would reject a config for leaving out something optional.
        """
        assert validator(None) is None


class TestValidationReachesTheFields:
    """The `Annotated` sweep actually runs, and says which field failed."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dip_deg", 0.0),
            ("dip_deg", 90.5),
            ("dip_deg", -10.0),
            ("bottom_depth_km", 0.0),
            ("bottom_depth_km", -5.0),
        ],
    )
    def test_a_plane_refuses_an_impossible_value(
        self, field: str, value: float
    ) -> None:
        plane = {
            "end": {"longitude_deg": 173.4, "latitude_deg": -42.4},
            "dip_deg": 70.0,
            "bottom_depth_km": 15.0,
            "discretisation": {"subfault_size_km": 1.0},
        }
        plane[field] = value
        with pytest.raises(InvalidFieldValue) as caught:
            PlaneConfig.from_dict(plane)
        assert field in str(caught.value) or field == caught.value.field_name

    @pytest.mark.parametrize(
        ("longitude_deg", "latitude_deg"),
        [(200.0, -43.0), (-181.0, -43.0), (172.0, 91.0), (172.0, -90.5)],
    )
    def test_a_position_off_the_earth_is_refused(
        self, longitude_deg: float, latitude_deg: float
    ) -> None:
        with pytest.raises(InvalidFieldValue):
            LonLat.from_dict(
                {"longitude_deg": longitude_deg, "latitude_deg": latitude_deg}
            )

    @pytest.mark.parametrize("magnitude", [2.9, 10.1, 0.0, -6.0])
    def test_a_magnitude_that_is_not_an_earthquake_is_refused(
        self, magnitude: float
    ) -> None:
        with pytest.raises(InvalidFieldValue):
            FiniteSourceConfig.from_dict(
                {
                    "type": "finite",
                    "magnitude": magnitude,
                    "average_dip_deg": 60.0,
                    "average_rake_deg": 175.0,
                }
            )

    @given(magnitude=st.floats(min_value=3.0, max_value=10.0))
    @SETTINGS
    def test_every_real_magnitude_is_accepted(self, magnitude: float) -> None:
        source = FiniteSourceConfig.from_dict(
            {
                "type": "finite",
                "magnitude": magnitude,
                "average_dip_deg": 60.0,
                "average_rake_deg": 175.0,
            }
        )
        assert source.magnitude == magnitude


class TestTheTaggedUnions:
    """Every member resolves from its tag and knows what it is.

    Parametrised over the union itself, so a member that exists but was never wired up
    fails rather than going unnoticed.
    """

    @pytest.mark.parametrize(
        "shape_class", subclasses(SlipRateShapeConfig), ids=lambda c: c.__name__
    )
    def test_every_slip_rate_shape_resolves_from_its_tag_and_builds(
        self, shape_class: type
    ) -> None:
        """Decoding the tag gives this class back, and it names a compiled shape.

        The second half is what catches a shape that was declared and never wired up:
        the base `to_core` raises rather than returning a default, so a missing override
        fails here instead of silently generating a different earthquake.
        """
        # Whatever the shape requires, at a value in range for all of them: `stretch` is
        # positive and `tau1_ratio` is a fraction, so 1.0 serves both.
        required = {
            field.name: 1.0
            for field in dataclasses.fields(shape_class)
            if field.name != "type" and field.default is dataclasses.MISSING
        }
        tag = dataclasses.fields(shape_class)[-1].default

        # Through the *base*, so this exercises the discriminator rather than the class.
        shape = SlipRateShapeConfig.from_dict({**required, "type": tag})

        assert isinstance(shape, shape_class), f"{tag!r} resolved to {type(shape)}"
        assert isinstance(shape.to_core(), _core.SlipRateShape)

    def test_there_are_eleven_of_them(self) -> None:
        """The count the crate's own commit message claims.

        `crates/genslip/src/slip_rate.rs` is titled *"Eleven shapes, one trait, no
        catch-alls"*. If a twelfth appears in the core without one here, this and
        `test_config_completeness` disagree about the vocabulary.
        """
        assert len(subclasses(SlipRateShapeConfig)) == 11

    def test_the_base_refuses_to_guess(self) -> None:
        """A shape without a `to_core` raises rather than returning a default.

        Returning `oliu_p2` would be a shape nobody chose, generating a different
        earthquake and looking like a working config.
        """
        with pytest.raises(NotImplementedError, match="does not say"):
            SlipRateShapeConfig().to_core()

    @pytest.mark.parametrize(
        ("tag", "expected"),
        [("fault", FaultConfig), ("point", PointConfig)],
    )
    def test_a_surface_tag_selects_its_class(self, tag: str, expected: type) -> None:
        document = a_geometry()
        if tag == "point":
            document["surfaces"] = [
                {
                    "type": "point",
                    "name": "aftershock",
                    "centre": {"longitude_deg": 173.5, "latitude_deg": -42.3},
                    "depth_km": 8.0,
                    "strike_deg": 55.0,
                    "dip_deg": 60.0,
                }
            ]
        geometry = GeometryConfig.from_dict(document)
        assert isinstance(geometry.surfaces[0], expected)

    @pytest.mark.parametrize(
        ("tag", "expected"),
        [("finite", FiniteSourceConfig), ("point", PointSourceConfig)],
    )
    def test_a_source_tag_selects_its_class(self, tag: str, expected: type) -> None:
        source = {
            "type": tag,
            "magnitude": 6.2,
            "average_dip_deg": 60.0,
            "average_rake_deg": 175.0,
        }
        if tag == "point":
            source["rise_time_s"] = 1.5
        document = a_rupture(source=source)
        if tag == "point":
            document.pop("slip", None)
        config = RuptureConfig.from_dict(document)
        assert isinstance(config.source, expected)

    def test_an_unknown_tag_is_refused(self) -> None:
        document = a_rupture(
            source={
                "type": "telepathy",
                "magnitude": 6.2,
                "average_dip_deg": 60.0,
                "average_rake_deg": 175.0,
            }
        )
        with pytest.raises(Exception, match="telepathy|not found|Unable"):
            RuptureConfig.from_dict(document)


class TestCrossFieldInvariants:
    """What no single field could have caught, and which key gets blamed."""

    def test_a_discretisation_given_twice_is_refused(self) -> None:
        with pytest.raises(InvalidFieldValue, match="not both"):
            Discretisation.from_dict({"subfault_size_km": 1.0, "strike_count": 20})

    def test_a_discretisation_given_no_way_is_refused(self) -> None:
        with pytest.raises(InvalidFieldValue, match="needs a subfault size"):
            Discretisation.from_dict({})

    @pytest.mark.parametrize("given", ["strike_count", "dip_count"])
    def test_half_a_discretisation_is_refused(self, given: str) -> None:
        with pytest.raises(InvalidFieldValue, match="both a strike and a dip"):
            Discretisation.from_dict({given: 12})

    def test_a_plane_above_its_fault_top_is_refused(self) -> None:
        document = a_geometry()
        document["surfaces"][0]["top_depth_km"] = 20.0
        with pytest.raises(Exception) as caught:
            GeometryConfig.from_dict(document)
        path, innermost = field_path(caught.value)
        assert "not below" in str(innermost)
        assert path.endswith("planes"), path

    def test_a_fault_with_no_planes_is_refused(self) -> None:
        document = a_geometry()
        document["surfaces"][0]["planes"] = []
        with pytest.raises(Exception) as caught:
            GeometryConfig.from_dict(document)
        _, innermost = field_path(caught.value)
        assert "at least one plane" in str(innermost)

    def test_a_geometry_with_no_surfaces_is_refused(self) -> None:
        with pytest.raises(InvalidFieldValue, match="at least one surface"):
            GeometryConfig.from_dict(a_geometry(surfaces=[]))

    def test_two_surfaces_with_one_name_are_refused(self) -> None:
        """Names become group names in the mesh file, so a clash loses a fault."""
        document = a_geometry()
        document["surfaces"].append(dict(document["surfaces"][0]))
        with pytest.raises(InvalidFieldValue, match="two surfaces are called"):
            GeometryConfig.from_dict(document)

    def test_a_geographic_crs_is_refused_with_advice(self) -> None:
        """EPSG:4326 is the obvious wrong answer, so the message names the right one."""
        with pytest.raises(InvalidFieldValue, match="EPSG:2193"):
            GeometryConfig.from_dict(a_geometry(crs="EPSG:4326"))

    def test_a_ragged_velocity_model_is_refused(self) -> None:
        with pytest.raises(InvalidFieldValue, match="different numbers of layers"):
            VelocityModelConfig.from_dict(
                {
                    "bottom_depth_km": [1.0, 5.0],
                    "shear_speed_km_s": [1.8],
                    "density_g_cm3": [2.1, 2.4],
                }
            )

    def test_an_inverted_wavelength_band_is_refused(self) -> None:
        with pytest.raises(InvalidFieldValue, match="above min_wavelength_km"):
            SlipConfig.from_dict({"min_wavelength_km": 80.0, "max_wavelength_km": 1.5})

    def test_a_point_source_with_a_slip_section_is_refused(self) -> None:
        """Silence is the failure mode: a point source draws nothing, so a `[slip]`
        section would be parsed, validated and then never read."""
        document = a_rupture(
            source={
                "type": "point",
                "magnitude": 6.2,
                "rise_time_s": 1.5,
                "average_dip_deg": 60.0,
                "average_rake_deg": 175.0,
            },
            slip={"coefficient_of_variation": 0.9},
        )
        with pytest.raises(InvalidFieldValue, match="read and ignored|draws no fields"):
            RuptureConfig.from_dict(document)


class TestMisspellingsAreErrors:
    """`forbid_extra_keys`, which is the difference between a typo and a different
    earthquake.

    `README.md` records the alternative running in production: genslip's `getpar` never
    asks for names it does not recognise, so five parameters *"have been silently
    discarded"* for as long as the workflow has pointed at a binary that does not know
    them.
    """

    @pytest.mark.parametrize(
        "section", ["source", "slip", "timing", "random", "hypocentre"]
    )
    def test_an_unknown_key_is_refused(self, section: str) -> None:
        document = a_rupture()
        document.setdefault(section, {})["definitely_not_a_parameter"] = 1.0
        with pytest.raises(Exception, match="definitely_not_a_parameter|extra"):
            RuptureConfig.from_dict(document)

    @pytest.mark.parametrize(
        "misspelling",
        ["magnitide", "average_dip_degrees", "rake_deg", "Magnitude"],
    )
    def test_a_near_miss_is_refused_rather_than_ignored(self, misspelling: str) -> None:
        source: dict[str, object] = {
            "type": "finite",
            "magnitude": 6.2,
            "average_dip_deg": 60.0,
            "average_rake_deg": 175.0,
            misspelling: 7.0,
        }
        with pytest.raises(ExtraKeysError):
            FiniteSourceConfig.from_dict(source)

    def test_a_missing_required_field_is_named(self) -> None:
        """And `field_path` digs it out of the wrapping.

        A required field missing inside a tagged union arrives as an
        `InvalidFieldValue` about the *union*, with the `MissingField` underneath it in
        `__context__`. The outer message names `source`, which is true and useless; the
        inner one names `magnitude`, which is the key to add.
        """
        document = a_rupture()
        del document["source"]["magnitude"]
        with pytest.raises(Exception) as caught:
            RuptureConfig.from_dict(document)

        path, innermost = field_path(caught.value)
        assert isinstance(innermost, MissingField)
        assert innermost.field_name == "magnitude"
        assert path.startswith("source"), path


class TestTheThreeSpellings:
    """TOML, YAML and JSON describe the same rupture.

    A claim about this schema, not about mashumaro: a field that only round-trips in one
    format is one whose type the other decoders read differently.
    """

    @pytest.mark.parametrize("suffix", ["toml", "yaml", "yml", "json"])
    def test_a_config_reads_the_same_in_every_format(
        self, suffix: str, tmp_path: Path
    ) -> None:
        document = a_rupture()
        expected = RuptureConfig.from_dict(document)

        path = tmp_path / f"config.{suffix}"
        if suffix == "json":
            path.write_text(json.dumps(document))
        elif suffix in {"yaml", "yml"}:
            path.write_text(yaml.safe_dump(document))
        else:
            path.write_text(RUPTURE_TOML)

        assert read_config(path) == expected

    def test_an_unrecognised_extension_is_read_as_toml(self, tmp_path: Path) -> None:
        path = tmp_path / "config.conf"
        path.write_text(RUPTURE_TOML)
        assert read_config(path) == RuptureConfig.from_dict(a_rupture())

    def test_the_format_can_be_given_explicitly(self, tmp_path: Path) -> None:
        """For a config on stdin, or one whose name says nothing."""
        path = tmp_path / "config.txt"
        path.write_text(json.dumps(a_rupture()))
        assert read_config(path, format="json") == RuptureConfig.from_dict(a_rupture())

    def test_a_geometry_reads_too(self, tmp_path: Path) -> None:
        path = tmp_path / "geometry.toml"
        path.write_text(GEOMETRY_TOML)
        geometry = read_geometry(path)
        assert geometry.crs.is_projected
        assert geometry.surfaces[0].name == "kaikoura"


class TestDefaultsAreTheCores:
    """A field left out gets what the core would have used.

    `test_config_completeness` says every argument is *reachable*; this says the ones
    nobody sets are the ones the core has. A default that drifted would be a config
    silently overriding the library with a different number.
    """

    @pytest.mark.parametrize(
        ("config_class", "field_name", "expected"),
        [
            (SlipConfig, "coefficient_of_variation", 0.75),
            (SlipConfig, "rake_sigma_deg", 15.0),
            (SlipConfig, "min_wavelength_km", 1.5),
            (SlipConfig, "max_wavelength_km", 80.0),
            (SlipConfig, "side_taper", 0.02),
            (SlipConfig, "truncate_negative", True),
            (FiniteSourceConfig, "rise_time_coefficient", 1.6),
            (FiniteSourceConfig, "saturation_magnitude", 6.3),
            (FiniteSourceConfig, "strike_offset", 2.50),
            (FiniteSourceConfig, "dip_offset", 1.50),
        ],
    )
    def test_a_default_matches_the_stub(
        self, config_class: type, field_name: str, expected: object
    ) -> None:
        field = next(
            f for f in dataclasses.fields(config_class) if f.name == field_name
        )
        assert field.default == expected

    def test_the_rng_defaults_to_the_compatible_engine(self) -> None:
        """So a config that says nothing reproduces what the port has always produced.

        `pcg` is the better generator and the one to choose deliberately; making it the
        default would silently change every existing config's output.
        """
        from rupture_generator.config.rupture import RandomConfig

        assert RandomConfig().engine == "genslip_lcg"


class TestSurfacesKnowTheirShape:
    @pytest.mark.parametrize(
        ("direction", "expected"), [("right", False), ("left", True)]
    )
    def test_dip_direction_reads_as_a_flag(
        self, direction: str, expected: bool
    ) -> None:
        plane = PlaneConfig.from_dict(
            {
                "end": {"longitude_deg": 173.4, "latitude_deg": -42.4},
                "dip_deg": 70.0,
                "bottom_depth_km": 15.0,
                "discretisation": {"subfault_size_km": 1.0},
                "dip_direction": direction,
            }
        )
        assert plane.dips_left is expected

    def test_an_unknown_dip_direction_is_refused(self) -> None:
        with pytest.raises(InvalidFieldValue):
            PlaneConfig.from_dict(
                {
                    "end": {"longitude_deg": 173.4, "latitude_deg": -42.4},
                    "dip_deg": 70.0,
                    "bottom_depth_km": 15.0,
                    "discretisation": {"subfault_size_km": 1.0},
                    "dip_direction": "downwards",
                }
            )

    @pytest.mark.parametrize(
        ("count", "expected"),
        [(1, 2), (10, 12), (20, 22), (24, 26), (11, 12), (19, 20)],
    )
    def test_the_default_padding_is_genslips_rule(
        self, count: int, expected: int
    ) -> None:
        """`even(int(1.10 * n))`, which `tests/harness/mapping.py` checks against the
        binary itself at every extent from 2 to 40."""
        from rupture_generator.config.rupture import GridConfig

        assert GridConfig.default_padding(count) == expected

    @given(count=st.integers(min_value=1, max_value=500))
    @SETTINGS
    def test_the_padding_is_always_even_and_always_fits(self, count: int) -> None:
        """The two things the generators actually require of it.

        Even because they address the Nyquist row and column directly; at least the
        fault because it is a wraparound margin.
        """
        from rupture_generator.config.rupture import GridConfig

        padded = GridConfig.default_padding(count)
        assert padded % 2 == 0
        assert padded >= count
