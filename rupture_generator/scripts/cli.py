"""``rupture-generator``: the whole command line.

Three steps, three subcommands, and the boundary between them is a file:

.. code-block:: text

    rupture-generator mesh     geometry.toml  mesh.h5
    rupture-generator generate config.toml    mesh.h5  rupture.h5
    rupture-generator view      rupture.h5

They are separate because their inputs have different lifetimes. A geometry is digitised
once and reused across every realisation run on it; a source config is what varies; a
rupture is the output. Fusing them would mean rebuilding the geometry on every run, and
a geometry rebuilt is a geometry that can come out different.

Each subcommand lives in its own module and is attached here, which is
``nzcvm/scripts/nzcvm_cli.py``'s shape and keeps every leaf independently runnable.
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
    """Generate kinematic rupture models.

    The callback is not decoration. Typer folds a one-command app into its root, so
    with a single command registered ``rupture-generator mesh`` would *be*
    ``rupture-generator``. Three are registered now, so it no longer folds -- but
    removing this would make the interface move again the next time one is taken away.
    It is also where a global option goes when there is one.
    """


# Single commands rather than sub-groups: each does one thing, and
# `rupture-generator mesh build` is a word nobody needs to type.
app.command("mesh")(mesh_cli.mesh)
app.command("generate")(generate_cli.generate)
app.command("view")(view_cli.view)


if __name__ == "__main__":
    app()
