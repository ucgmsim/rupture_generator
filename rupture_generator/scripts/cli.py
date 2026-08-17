"""``rupture-generator``: the whole command line.

Three steps, three subcommands, and the boundary between them is a file:

.. code-block:: text

    rupture-generator mesh     geometry.toml  mesh.h5
    rupture-generator generate config.toml    mesh.h5  rupture.h5
    rupture-generator view      rupture.h5

They are separate because their inputs have different lifetimes: a geometry is
digitised once and reused, a source config is what varies, and a rupture is the output.
"""

import typer

from rupture_generator.scripts import generate_cli, mesh_cli
from rupture_generator.scripts import view as view_cli

app = typer.Typer(
    help="Generate kinematic rupture models.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Generate kinematic rupture models."""


# Single commands rather than sub-groups: each does one thing, and
# `rupture-generator mesh build` is a word nobody needs to type.
app.command("mesh")(mesh_cli.mesh)
app.command("generate")(generate_cli.generate)
app.command("view")(view_cli.view)


if __name__ == "__main__":
    app()
