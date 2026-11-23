import dataclasses
from pathlib import Path

import numpy as np
import pyproj
import pytest

from rupture_generator import Geometry, Realisation, SegmentRupture, config, generate
from rupture_generator.formats.srf import write_rupture

EXAMPLE = Path(__file__).parents[1] / "examples" / "two_faults.toml"

CFM = EXAMPLE.parent / "cfm"
"""The NZ Community Fault Model's subduction interfaces, as the examples ship them."""

NZTM = pyproj.CRS("EPSG:2193")

ORIGIN_KM = np.array([1600.0, 5300.0, 0.0])
"""Near the Alpine fault in NZTM, so positions carry NZTM's rounding."""

SPACING_KM = 0.5
"""The fixture's own resolution: the example is drawn finer than a test needs."""


@pytest.fixture(scope="session")
def scenario() -> config.Scenario:
    loaded = config.load(EXAMPLE)
    geometry = dataclasses.replace(loaded.geometry, spacing_km=SPACING_KM)
    return config.build(dataclasses.replace(loaded, geometry=geometry))


@pytest.fixture
def coarse_example(tmp_path):
    """The example as a rupture file at the fixture's resolution."""
    loaded = config.load(EXAMPLE)
    geometry = dataclasses.replace(loaded.geometry, spacing_km=SPACING_KM)
    path = tmp_path / "coarse.toml"
    path.write_text(config.dump(dataclasses.replace(loaded, geometry=geometry)))
    return path


@pytest.fixture(scope="session")
def ruptures(scenario: config.Scenario) -> Realisation[SegmentRupture]:
    return generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        scenario.model,
        seed=scenario.seed,
        jump_propagation_model=scenario.jump_model,
    )


@pytest.fixture(scope="session")
def srf_path(tmp_path_factory, scenario, ruptures):
    path = tmp_path_factory.mktemp("srf") / "rupture.srf"
    write_rupture(str(path), ruptures)
    return path


def tilted(strike_deg, dip_deg, cells=(3, 4), occupied=None):
    """A flat chart of 1 km cells along a frame strike, dipping to its right.

    Returns the chart and its strike as a unit vector in plan. Every cell is fault
    unless ``occupied`` says otherwise.
    """
    s, d = np.radians(strike_deg), np.radians(dip_deg)
    along = np.array([np.sin(s), np.cos(s), 0.0])
    down = np.array([np.cos(d) * np.cos(s), -np.cos(d) * np.sin(s), np.sin(d)])
    i, j = np.mgrid[0 : cells[0] + 1, 0 : cells[1] + 1].astype(float)
    nodes = ORIGIN_KM + i[..., None] * down + j[..., None] * along
    if occupied is None:
        occupied = np.ones(cells, bool)
    return Geometry(nodes, occupied, (cells[1],)), along[:2]


def holed():
    """A 3 x 4 chart of 1 km cells dipping 45 degrees east of frame north, with its
    top-left and bottom-right cells off the fault."""
    occupied = np.ones((3, 4), bool)
    occupied[0, 0] = occupied[2, 3] = False
    return tilted(0.0, 45.0, occupied=occupied)[0]


def srf_body(path: Path) -> bytes:
    """An SRF's bytes without its header comments, which hold the run's record."""
    with path.open("rb") as handle:
        return b"".join(line for line in handle if not line.startswith(b"#"))
