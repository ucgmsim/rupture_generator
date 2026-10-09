from pathlib import Path

import pytest

from rupture_generator import config, generate

EXAMPLE = Path(__file__).parents[1] / "examples" / "two_faults.toml"


@pytest.fixture(scope="session")
def scenario() -> config.Scenario:
    return config.build(config.load(EXAMPLE))


@pytest.fixture(scope="session")
def ruptures(scenario: config.Scenario) -> dict:
    return generate(
        scenario.realisation,
        scenario.materials,
        scenario.sources,
        scenario.settings,
        seed=scenario.seed,
    )
