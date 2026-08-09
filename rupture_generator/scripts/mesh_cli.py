"""``rupture-generator mesh``: a geometry description in, a mesh file out.

The step that turns a fault written down as a trace into subfault positions. It is its
own subcommand rather than part of ``generate`` because a geometry is digitised once
and reused -- across realisations, across magnitudes, across whole studies -- and
rebuilding it every run would be both wasteful and a chance for it to come out
different.

Everything geometric lives in `rupture_generator.mesh`; what is here is the I/O, the
error rendering, and the summary table. The table is the reason the rounding is worth
showing: a config asks for a subfault *size*, a mesh is cut into whole *cells*, and
the size actually used is printed beside the one asked for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pyproj
import typer
from rich.table import Table

from rupture_generator.config import read_geometry
from rupture_generator.config.geometry import GeometryConfig
from rupture_generator.formats import Format
from rupture_generator.formats.mesh import write_mesh
from rupture_generator.mesh import RuptureMesh, build_surface, project_cells
from rupture_generator.scripts.errors import console, load_config


def summarise(meshes: dict[str, list[RuptureMesh]], crs: pyproj.CRS) -> Table:
    """What was built, so a reader can see it is the fault they meant.

    The first thing anyone wants after discretising is to check the numbers, and the
    one they cannot get from the config alone is the subfault size *actually* used --
    a request of 1.0 km on a 39.7 km plane becomes 0.99 km, because the plane is cut
    into whole cells.

    Strike is reported **true**, not grid. The mesh works from the projection's
    northing axis, and in NZTM that is up to five degrees off true north -- a number
    in a summary table will be read as a compass bearing, so it has to be one.
    """
    table = Table(
        title=f"Mesh in {crs.to_string()}",
        title_justify="left",
        caption="strike and dip are of the first cell; strike is true north",
        caption_justify="left",
    )
    for column, justify in (
        ("surface", "left"),
        ("#", "right"),
        ("cells", "right"),
        ("cell km", "right"),
        ("extent km", "right"),
        ("strike/dip", "right"),
        ("depth km", "right"),
    ):
        table.add_column(column, justify=justify)

    for name, charts in meshes.items():
        for index, chart in enumerate(charts):
            dip_cells, strike_cells = chart.cell_counts
            strike_km, dip_km = chart.spacing_km()
            located = project_cells(chart, crs)
            depth_km = chart.dataset["depth_km"].to_numpy()
            table.add_row(
                name if index == 0 else "",
                str(index),
                f"{strike_cells}x{dip_cells}",
                f"{strike_km:.2f}x{dip_km:.2f}",
                f"{strike_km * strike_cells:.1f}x{dip_km * dip_cells:.1f}",
                f"{float(located['strike_deg'][0, 0]):.1f}/"
                f"{float(located['dip_deg'][0, 0]):.1f}",
                f"{depth_km.min():.1f}-{depth_km.max():.1f}",
            )
    return table


def mesh(
    geometry: Annotated[
        Path,
        typer.Argument(
            help="Geometry description to read (TOML, YAML or JSON).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[Path, typer.Argument(help="Mesh file to write.")],
    output_format: Annotated[
        Format,
        typer.Option(
            "--format", help="Output format. Usually inferred from the extension."
        ),
    ] = Format.INFERRED,
    quiet: Annotated[
        bool, typer.Option(help="Do not print the summary table.")
    ] = False,
) -> None:
    """Discretise a fault geometry into a mesh."""
    config: GeometryConfig = load_config(geometry, read_geometry)

    meshes: dict[str, list[RuptureMesh]] = {}
    for surface in config.surfaces:
        try:
            meshes[surface.name] = build_surface(surface, config.crs)
        except ValueError as error:
            console.print(f"[red]{surface.name}: {error}[/red]")
            raise typer.Exit(1) from error

    write_mesh(
        meshes,
        config.crs,
        output,
        format=output_format,
        propagation=config.propagation,
        attrs={
            "title": config.title or geometry.stem,
            "geometry_config": geometry.read_text(),
            "surfaces": json.dumps(list(meshes)),
        },
    )

    if not quiet:
        console.print(summarise(meshes, config.crs))
        console.print(f"[green]wrote[/green] {output}")
