"""Construction refuses malformed input, with an error that says what was wrong."""

import io
import json

import numpy as np
import pyproj
import pytest

from rupture_generator import (
    Geometry,
    Hypocentre,
    Layers,
    RakeSettings,
    Realisation,
    RiseSettings,
    RuptureGeneratorError,
    TimingSettings,
    geometry_from_geojson,
    interpolated_field,
    layered_field,
    ramp_field,
)
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
        (lambda: RiseSettings(slip_exponent=0.05), "slip exponent"),
        (lambda: RiseSettings(correlation=1.5), "correlation lies in"),
        (lambda: RakeSettings(sigma_deg=-1.0), "no sign"),
        (lambda: TimingSettings(velocity_fraction=0.0), "velocity fraction"),
        (lambda: TimingSettings(offset_s=-0.1), "magnitudes"),
        (lambda: TimingSettings(blend_sigma=0.0), "no width"),
    ],
)
def test_settings_out_of_range_are_refused(build, message):
    with pytest.raises(RuptureGeneratorError, match=message):
        build()


@pytest.mark.parametrize(
    ("family", "cov", "message"),
    [
        ("gamma", 0.0, "positive coefficient of variation"),
        ("truncated_normal", 5.0, "truncated normal cannot"),
        ("truncated_exponential", 5.0, "truncated exponential cannot"),
    ],
)
def test_a_marginal_its_family_cannot_have_is_refused(family, cov, message):
    with pytest.raises(RuptureGeneratorError, match=message):
        Marginal(family, cov)


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
            "seam column",
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


def test_one_plane_has_no_seams():
    assert _plane().seam_divergence_km().shape == (0,)


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
