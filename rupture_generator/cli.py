"""``rupture-generator CONFIG OUTPUT``: read a rupture file and write an SRF.

Everything here is a library call. Next to the SRF goes the config with every default
written out, ``OUTPUT.toml``, which records what the run used.
"""

import argparse
import sys
from pathlib import Path

from rupture_generator import config as rupture_config
from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.formats.srf import write_rupture
from rupture_generator.rupture.generator import generate


def main(argv: list[str] | None = None) -> int:
    """Run the command line.

    Parameters
    ----------
    argv : list of str, optional
        The arguments after the program name. ``None`` reads them from
        :data:`sys.argv`.

    Returns
    -------
    int
        The exit status: 0 on success, 2 when the input describes no rupture or a
        file read or write fails.
    """
    parser = argparse.ArgumentParser(
        prog="rupture-generator",
        description="Draw a kinematic rupture from a TOML file and write it as an SRF.",
    )
    parser.add_argument("config", type=Path, help="the rupture file")
    parser.add_argument("output", type=Path, help="where to write the SRF")
    args = parser.parse_args(argv)

    try:
        config = rupture_config.load(args.config)
        scenario = rupture_config.build(config)
        ruptures = generate(
            scenario.realisation,
            scenario.medium,
            scenario.sources,
            scenario.settings,
            seed=scenario.seed,
            jump_model=scenario.jump_model,
        )
        write_rupture(
            str(args.output),
            scenario.realisation,
            ruptures,
            dt_s=scenario.dt_s,
            beta=scenario.beta,
        )
    except (RuptureGeneratorError, OSError) as error:
        print(f"rupture-generator: {error}", file=sys.stderr)
        return 2
    args.output.with_name(args.output.name + ".toml").write_text(
        rupture_config.dump(config)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
