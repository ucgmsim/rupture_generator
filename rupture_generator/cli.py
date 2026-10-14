"""``rupture-generator INPUT OUTPUT``: draw a rupture, or turn stored fields into an SRF.

Everything here is a library call. A TOML ``INPUT`` is a rupture file to draw, and
``OUTPUT``'s suffix picks what to write: ``.srf`` for an SRF, or ``.h5`` or ``.nc`` for
the drawn fields, which make an SRF later. A fields file as ``INPUT`` writes its SRF to
``OUTPUT``.

Each draw keeps its run record, :mod:`rupture_generator.record`: the config with every
default written out, the software that ran it and the digests of its inputs. An SRF
stores it in its header's comment lines, and a fields file in its root attributes. The
SRF made from a fields file keeps the fields file's record.
"""

import argparse
import sys
from pathlib import Path

from rupture_generator import config as rupture_config
from rupture_generator.errors import RuptureGeneratorError
from rupture_generator.formats.fields import (
    RECORD_ATTRIBUTE,
    SUFFIXES,
    fields_to_srf,
    write_fields,
)
from rupture_generator.formats.srf import write_rupture
from rupture_generator.record import run_record
from rupture_generator.rupture.generator import generate


def _draw(source: Path, output: Path) -> None:
    """Draw the rupture ``source`` describes, and write it and its record."""
    if output.suffix != ".srf" and output.suffix not in SUFFIXES:
        raise RuptureGeneratorError(
            f"{output} ends in neither .srf nor one of {', '.join(SUFFIXES)}"
        )
    config = rupture_config.load(source)
    record = run_record(source, config)
    scenario = rupture_config.build(config)
    drawn = generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        scenario.settings,
        seed=scenario.seed,
        jump_propagation_model=scenario.jump_model,
    )
    if output.suffix == ".srf":
        write_rupture(str(output), drawn, comments=record.splitlines())
    else:
        write_fields(output, drawn, attrs={RECORD_ATTRIBUTE: record})


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
        description=(
            "Draw a kinematic rupture from a TOML file and write it as an SRF (.srf) "
            f"or as its fields ({', '.join(SUFFIXES)}). Given a fields file, write "
            "its SRF."
        ),
    )
    parser.add_argument("input", type=Path, help="the rupture file, or a fields file")
    parser.add_argument("output", type=Path, help="where to write the SRF or fields")
    args = parser.parse_args(argv)

    try:
        if args.input.suffix in SUFFIXES:
            if args.output.suffix != ".srf":
                raise RuptureGeneratorError(
                    f"{args.input} is a fields file, which writes an SRF, and "
                    f"{args.output} does not end in .srf"
                )
            fields_to_srf(args.input, args.output)
        else:
            _draw(args.input, args.output)
    except (RuptureGeneratorError, OSError) as error:
        print(f"rupture-generator: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
