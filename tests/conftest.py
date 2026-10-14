import dataclasses
from pathlib import Path

import pytest

from rupture_generator import Realisation, SegmentRupture, config, generate
from rupture_generator.formats.srf import write_rupture

EXAMPLE = Path(__file__).parents[1] / "examples" / "two_faults.toml"

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
        scenario.settings,
        seed=scenario.seed,
        jump_propagation_model=scenario.jump_model,
    )


@pytest.fixture(scope="session")
def srf_path(tmp_path_factory, scenario, ruptures):
    path = tmp_path_factory.mktemp("srf") / "rupture.srf"
    write_rupture(str(path), ruptures)
    return path
