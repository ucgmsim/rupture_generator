"""Construction refuses malformed input, with an error that says what was wrong."""

import io
import json

import numpy as np
import pyproj
import pytest

from rupture_generator import (
    CorrelatedPerturbation,
    CorrelatedRise,
    DrawnSlip,
    GaussianRake,
    Geometry,
    Hypocentre,
    Layers,
    Realisation,
    RuptureGeneratorError,
    ShearFractionSpeed,
    geometry_from_geojson,
    interpolated_field,
    layered_field,
    ramp_field,
)
from rupture_generator.geometry import NSHM_ALIASES
from rupture_generator.sampling import Marginal

NZTM = pyproj.CRS("EPSG:2193")


def _plane(cells=(1, 1)):
    """One flat plane, 1 km cells, ``cells`` down dip and along strike."""
    rows, columns = cells
    i, j = np.mgrid[0 : rows + 1, 0 : columns + 1].astype(float)
    nodes = np.stack([j, i, i], axis=-1)
    return Geometry(nodes=nodes, occupied=np.ones(cells, bool), plane_cells=(columns,))


def _section(**overrides):
    properties = {
        "name": "a",
        "dip_deg": 60.0,
        "dip_direction_deg": 90.0,
        "upper_depth_km": 0.0,
        "lower_depth_km": 10.0,
    }
    properties.update(overrides.pop("properties", {}))
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[172.0, -42.0], [172.0, -41.9]],
        },
        "properties": properties,
    }
    feature.update(overrides)
    return feature


def _read(*features, document=None):
    document = document or {"type": "FeatureCollection", "features": list(features)}
    return geometry_from_geojson(io.StringIO(json.dumps(document)), NZTM)


# ------------------------------------------------------------------- settings


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: CorrelatedRise(slip_exponent=0.05), "slip exponent"),
        (lambda: CorrelatedRise(correlation=1.5), "correlation lies in"),
        (lambda: GaussianRake(sigma_deg=-1.0), "no sign"),
        (lambda: ShearFractionSpeed(velocity_fraction=0.0), "velocity fraction"),
        (lambda: CorrelatedPerturbation(offset_s=-0.1), "magnitudes"),
        (lambda: CorrelatedPerturbation(blend_sigma=0.0), "no width"),
        (lambda: CorrelatedPerturbation(correlation=-1.5), "correlation lies in"),
    ],
)
def test_settings_out_of_range_are_refused(build, message):
    with pytest.raises(RuptureGeneratorError, match=message):
        build()


@pytest.mark.parametrize(
    ("family", "cov", "maximum", "message"),
    [
        ("gamma", 0.0, np.inf, "positive coefficient of variation"),
        ("truncated_normal", 5.0, np.inf, "truncated normal cannot"),
        ("truncated_exponential", 5.0, np.inf, "truncated exponential cannot"),
        ("gamma", 0.75, 4.0, "only a doubly_truncated_normal"),
        ("doubly_truncated_normal", 0.5, np.inf, "needs a finite maximum"),
        ("doubly_truncated_normal", 0.5, 1.0, "needs a finite maximum"),
        ("doubly_truncated_normal", 0.9, 4.3, "all but becomes the truncated"),
    ],
)
def test_a_marginal_its_family_cannot_have_is_refused(family, cov, maximum, message):
    with pytest.raises(RuptureGeneratorError, match=message):
        Marginal(family, cov, maximum)


def test_slip_cannot_follow_a_normal():
    with pytest.raises(RuptureGeneratorError, match="slip values are positive"):
        DrawnSlip(family="normal")


# ------------------------------------------------------------------- geometry


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: Geometry(np.zeros((1, 2, 3)), np.ones((0, 1), bool), (1,)),
            "at least one row",
        ),
        (
            lambda: Geometry(np.zeros((2, 2, 3)), np.ones((1, 1), bool), ()),
            "every plane",
        ),
        (
            lambda: Geometry(np.zeros((2, 3, 3)), np.ones((1, 1), bool), (1,)),
            "junction column",
        ),
        (
            lambda: Geometry(np.full((2, 2, 3), np.nan), np.ones((1, 1), bool), (1,)),
            "NaN",
        ),
        (
            lambda: Geometry(np.zeros((2, 2, 3)), np.ones((2, 2), bool), (1,)),
            "occupied",
        ),
    ],
)
def test_a_malformed_chart_is_refused(build, message):
    with pytest.raises(RuptureGeneratorError, match=message):
        build()


def test_a_chart_reads_as_its_shape():
    assert repr(_plane((2, 3))) == "Geometry(2x3 cells)"


def test_a_position_off_the_chart_is_refused():
    with pytest.raises(RuptureGeneratorError, match="off the fault"):
        _plane().cell_at(5.0, 0.5)


@pytest.mark.parametrize("spacing_km", [0.0, np.inf])
def test_a_subfault_size_that_is_no_length_is_refused(spacing_km):
    with pytest.raises(RuptureGeneratorError, match="positive length"):
        _plane().subdivide(spacing_km)


def test_a_partial_outline_cannot_be_resampled():
    chart = _plane((2, 2))
    holed = Geometry(chart.nodes, np.array([[True, False], [True, True]]), (2,))
    with pytest.raises(RuptureGeneratorError):
        holed.subdivide(0.5)


def test_an_empty_chart_has_no_distance_to_anything():
    chart = _plane()
    empty = Geometry(chart.nodes, np.zeros((1, 1), bool), (1,))
    with pytest.raises(RuptureGeneratorError, match="unoccupied"):
        chart.nearest_cells_to(empty)


# -------------------------------------------------------------------- geojson


@pytest.mark.parametrize(
    ("feature", "message"),
    [
        ({"properties": {"name": "a"}}, "has no"),
        ({"properties": {"dip_deg": 95.0}}, "not in \\(0, 90\\]"),
        ({"properties": {"upper_depth_km": 10.0}}, "not a depth range"),
        ({"properties": {"dip_direction_deg": 0.0}}, "runs along"),
    ],
)
def test_a_section_that_describes_no_fault_is_refused(feature, message):
    section = _section()
    if feature["properties"] == {"name": "a"}:
        del section["properties"]["dip_deg"]
    else:
        section["properties"].update(feature["properties"])
    with pytest.raises(RuptureGeneratorError, match=message):
        _read(section)


def test_a_missing_property_names_the_section_and_what_it_has():
    section = _section()
    del section["properties"]["dip_deg"]
    with pytest.raises(
        RuptureGeneratorError,
        match=r"^section 'a' has no 'dip_deg' property; it has \['dip_direction_deg', "
        r"'lower_depth_km', 'name', 'upper_depth_km'\]$",
    ):
        _read(section)


def test_a_section_without_a_name_is_named_by_its_id():
    section = _section(id="AF-1")
    del section["properties"]["name"]
    with pytest.raises(
        RuptureGeneratorError, match=r"^section 'AF-1' has no 'name' property; it has"
    ):
        _read(section)


def test_a_missing_aliased_property_names_both_spellings():
    section = _section()
    del section["properties"]["dip_deg"]
    with pytest.raises(RuptureGeneratorError, match="no 'dip_deg' or 'DipDeg'"):
        geometry_from_geojson(
            io.StringIO(
                json.dumps({"type": "FeatureCollection", "features": [section]})
            ),
            NZTM,
            aliases=NSHM_ALIASES,
        )


def test_a_trace_of_one_point_is_refused():
    section = _section()
    section["geometry"]["coordinates"] = [[172.0, -42.0]]
    with pytest.raises(RuptureGeneratorError, match="at least two"):
        _read(section)


def test_a_trace_that_doubles_back_past_its_dip_direction_is_refused():
    section = _section()
    section["geometry"]["coordinates"] = [
        [172.0, -42.0],
        [172.0, -41.9],
        [172.0, -42.05],
    ]
    with pytest.raises(RuptureGeneratorError, match="doubles back"):
        _read(section)


def test_a_section_is_a_line():
    section = _section()
    section["geometry"] = {"type": "Point", "coordinates": [172.0, -42.0]}
    with pytest.raises(RuptureGeneratorError, match="not a Point"):
        _read(section)


def test_a_section_without_a_geometry_is_refused():
    with pytest.raises(RuptureGeneratorError, match="^a: a section is a LineString"):
        _read(_section(geometry=None))


@pytest.mark.parametrize(
    ("properties", "message"),
    [
        ({"dip_deg": "steep"}, "a: dip_deg is 'steep', which is not a number"),
        ({"lower_depth_km": None}, "a: lower_depth_km is None, which is not a number"),
    ],
)
def test_a_property_that_is_not_a_number_is_refused(properties, message):
    with pytest.raises(RuptureGeneratorError, match=message):
        _read(_section(properties=properties))


def test_a_trace_that_is_not_coordinates_is_refused():
    section = _section()
    section["geometry"]["coordinates"] = [["east", "north"], [172.0, -41.9]]
    with pytest.raises(RuptureGeneratorError, match="a: the trace's coordinates"):
        _read(section)


def test_a_document_that_is_not_json_is_refused():
    with pytest.raises(RuptureGeneratorError, match="is not JSON"):
        geometry_from_geojson(io.StringIO('{"type": "FeatureCollection",'), NZTM)


def test_sections_have_their_own_names():
    with pytest.raises(RuptureGeneratorError, match="share a name"):
        _read(_section(), _section())


@pytest.mark.parametrize(
    "document", [{"type": "FeatureCollection", "features": []}, []]
)
def test_a_fault_system_is_a_collection_of_sections(document):
    with pytest.raises(RuptureGeneratorError, match="FeatureCollection"):
        _read(document=document)


# --------------------------------------------------------------------- medium


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: Layers(np.array([])), "non-empty"),
        (lambda: Layers(np.array([2.0, 1.0])), "do not increase"),
        (lambda: Layers(np.array([0.0, 1.0])), "not below the surface"),
        (
            lambda: layered_field(Layers(np.array([1.0, 2.0])), [1.0], name="vs"),
            "1 values for 2 layers",
        ),
        (
            lambda: layered_field(Layers(np.array([1.0])), [0.0], name="vs"),
            "positive, finite",
        ),
        (lambda: interpolated_field(np.array([0.0, 1.0]), np.array([1.0])), "1 values"),
        (
            lambda: interpolated_field(np.array([0.0, 1.0]), np.array([1.0, np.nan])),
            "not finite",
        ),
        (lambda: ramp_field(2.0, 0.0, 1.0, 2.0), "is a step"),
    ],
)
def test_a_medium_or_profile_that_is_no_model_is_refused(build, message):
    with pytest.raises(RuptureGeneratorError, match=message):
        build()


# ---------------------------------------------------------------- realisation


def test_a_realisation_needs_a_fault_in_a_projected_frame():
    with pytest.raises(RuptureGeneratorError, match="at least one segment"):
        Realisation({}, NZTM)
    with pytest.raises(RuptureGeneratorError, match="not a projected CRS"):
        Realisation({"a": _plane()}, pyproj.CRS("EPSG:4326"))
    with pytest.raises(RuptureGeneratorError, match="US survey foot"):
        Realisation({"a": _plane()}, pyproj.CRS("EPSG:2227"))


# ------------------------------------------------------------------------ crs


@pytest.mark.parametrize(
    ("crs", "message"),
    [
        (None, "a projected CRS in metres is required"),
        ("", "a projected CRS in metres is required"),
        ("nonsense", "'nonsense' is not a CRS"),
        ("EPSG:4326", "EPSG:4326 .* is not a projected CRS"),
        ("EPSG:2227", "EPSG:2227 .* measures its axes in US survey foot"),
    ],
)
def test_a_crs_that_is_not_projected_in_metres_is_refused(crs, message):
    with pytest.raises(RuptureGeneratorError, match=message):
        geometry_from_geojson(
            io.StringIO(json.dumps({"type": "FeatureCollection", "features": []})), crs
        )


def _utah():
    section = _section(properties={"name": "wasatch", "dip_direction_deg": 270.0})
    section["geometry"]["coordinates"] = [[-111.9, 40.7], [-111.8, 40.9]]
    return {"type": "FeatureCollection", "features": [section]}


def test_a_crs_made_for_elsewhere_is_refused():
    # NZTM would build Utah's faults half as big again, without a word.
    with pytest.raises(
        RuptureGeneratorError,
        match=r"EPSG:2193 .* is made for longitude 166.37 to 178.63 .* the faults "
        r"'wasatch' lie outside it, at longitude -111.9 to -111.8 .* EPSG:32612",
    ):
        geometry_from_geojson(io.StringIO(json.dumps(_utah())), NZTM)
    charts = geometry_from_geojson(io.StringIO(json.dumps(_utah())), "EPSG:32612")
    assert set(charts) == {"wasatch"}


def test_a_crs_that_stretches_the_faults_is_refused():
    # Web Mercator covers the world, and stretches lengths by a third at 42 S.
    with pytest.raises(RuptureGeneratorError, match="stretches lengths by up to 3"):
        geometry_from_geojson(
            io.StringIO(
                json.dumps({"type": "FeatureCollection", "features": [_section()]})
            ),
            "EPSG:3857",
        )


def test_an_offshore_fault_near_the_area_of_use_is_accepted():
    section = _section()
    # 0.8 degrees west of NZTM's area of use, like the Puysegur end of the Alpine
    # fault.
    section["geometry"]["coordinates"] = [[165.55, -46.0], [165.6, -45.9]]
    assert set(_read(section)) == {"a"}


def test_a_hypocentre_off_the_fault_is_refused():
    with pytest.raises(RuptureGeneratorError, match="off the fault"):
        Hypocentre.from_fractions("a", _plane(), 1.5, 0.5)


@pytest.mark.parametrize(
    ("tree", "message"),
    [
        ({"a": None}, "the tree names"),
        ({"a": None, "b": "nope"}, "the tree names"),
        ({"a": "b", "b": None}, "rooted at"),
    ],
)
def test_a_tree_that_is_not_the_rupture_is_refused(tree, message):
    segments = {"a": _plane(), "b": _plane()}
    with pytest.raises(RuptureGeneratorError, match=message):
        Realisation(segments, NZTM, Hypocentre("a", 0.5, 0.5), tree)


def test_an_unpropagated_system_has_no_order():
    segments = {"a": _plane(), "b": _plane()}
    with pytest.raises(RuptureGeneratorError, match="no hypocentre"):
        list(Realisation(segments, NZTM).in_causal_order())
    with pytest.raises(RuptureGeneratorError, match="not been propagated"):
        list(Realisation(segments, NZTM, Hypocentre("a", 0.5, 0.5)).in_causal_order())


def test_a_realisation_reads_as_its_segments():
    realisation = Realisation({"a": _plane()}, NZTM)
    assert "a" in repr(realisation)
    assert len(realisation) == 1
