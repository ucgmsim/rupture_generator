from pathlib import Path

import pytest

from rupture_generator import config, generate
from rupture_generator.formats.srf import write_rupture

EXAMPLE = Path(__file__).parents[1] / "examples" / "two_faults.toml"


@pytest.fixture(scope="session")
def scenario() -> config.Scenario:
    return config.build(config.load(EXAMPLE))


@pytest.fixture(scope="session")
def ruptures(scenario: config.Scenario) -> dict:
    return generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        scenario.settings,
        seed=scenario.seed,
        jump_model=scenario.jump_model,
    )


@pytest.fixture(scope="session")
def srf_path(tmp_path_factory, scenario, ruptures):
    path = tmp_path_factory.mktemp("srf") / "rupture.srf"
    write_rupture(
        str(path),
        scenario.realisation,
        ruptures,
        dt_s=scenario.dt_s,
        beta=scenario.beta,
    )
    return path
