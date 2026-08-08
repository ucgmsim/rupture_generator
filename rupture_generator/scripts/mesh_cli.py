"""``rupture-generator mesh``: a geometry description in, a mesh file out.

The step that turns a fault written down as a trace into subfault positions. It is its
own subcommand rather than part of ``generate`` because a geometry is digitised once and
reused -- across realisations, across magnitudes, across whole studies -- and rebuilding
it every run would be both wasteful and a chance for it to come out different.

# Where the rounding happens

A config asks for a subfault *size*; a mesh is cut into whole *cells*. Turning one into
the other is a rounding decision, and it happens here rather than in the library,
because this is where it can be shown: the summary table prints the size actually used
beside the one asked for.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Annotated

import numpy as np
import pyproj
import typer
import yaml
from mashumaro.exceptions import InvalidFieldValue, MissingField
from rich.table import Table

from rupture_generator._core import (
    Cuts,
    Fault,
    Plane,
    PointSource,
    Projected,
    build_fault_mesh,
    build_point_mesh,
)
from rupture_generator.config import read_geometry
from rupture_generator.config.geometry import (
    Discretisation,
    FaultConfig,
    GeometryConfig,
    LonLat,
    PointConfig,
)
from rupture_generator.formats import Format
from rupture_generator.formats.mesh import write_mesh
from rupture_generator.mesh import project_patch, to_projected
from rupture_generator.scripts.errors import (
    console,
    print_config_error,
    print_syntax_error,
)


def cell_counts(
    discretisation: Discretisation, length_km: float, width_km: float
) -> Cuts:
    """How many cells a plane gets, from a size or from explicit counts.

    A size is a *request*: the plane is cut into whole cells, so the size actually used
    is the plane's own length over the count. Rounded to nearest rather than down, and
    floored at one -- a plane shorter than the size asked for is still a plane, and
    zero cells is not a surface.

    Parameters
    ----------
    discretisation : Discretisation
        What the config asked for.
    length_km, width_km : float
        The plane's own dimensions, which is why this cannot happen at parse time.

    Returns
    -------
    Cuts
    """
    if discretisation.subfault_size_km is not None:
        size = discretisation.subfault_size_km
        return Cuts(
            max(1, round(length_km / size)),
            max(1, round(width_km / size)),
        )
    return Cuts(discretisation.strike_count, discretisation.dip_count)


def _projected(crs: pyproj.CRS, point: LonLat) -> Projected:
    return Projected(*to_projected(crs, point.longitude_deg, point.latitude_deg))


def build_surface(surface: FaultConfig | PointConfig, crs: pyproj.CRS):
    """Discretise one surface.

    Returns
    -------
    RefinedMesh
    """
    if isinstance(surface, PointConfig):
        return build_point_mesh(
            PointSource(
                _projected(crs, surface.centre),
                depth_km=surface.depth_km,
                strike_deg=surface.strike_deg,
                dip_deg=surface.dip_deg,
                size_km=surface.size_km,
            )
        )

    origin = _projected(crs, surface.origin)
    planes: list[Plane] = []
    cuts: list[Cuts] = []
    near = origin

    for plane in surface.planes:
        far = _projected(crs, plane.end)
        planes.append(
            Plane(
                far,
                dip_deg=plane.dip_deg,
                bottom_depth_km=plane.bottom_depth_km,
                dips_left=plane.dips_left,
            )
        )
        # The plane's own dimensions, which the discretisation is rounded against.
        length_km = float(
            np.hypot(
                far.easting_km - near.easting_km, far.northing_km - near.northing_km
            )
        )
        width_km = (plane.bottom_depth_km - surface.top_depth_km) / np.sin(
            np.deg2rad(plane.dip_deg)
        )
        cuts.append(cell_counts(plane.discretisation, length_km, float(width_km)))
        near = far

    return build_fault_mesh(
        Fault(origin, planes, top_depth_km=surface.top_depth_km), cuts
    )


def summarise(meshes: dict, crs: pyproj.CRS) -> Table:
    """What was built, so a reader can see it is the fault they meant.

    The first thing anyone wants after discretising is to check the numbers, and the one
    they cannot get from the config alone is the subfault size *actually* used -- a
    request of 1.0 km on a 39.7 km plane becomes 0.99 km, because the plane is cut into
    whole cells.

    Strike is reported **true**, not grid. The mesh works from the projection's northing
    axis, and in NZTM that is up to five degrees off true north -- a number in a summary
    table will be read as a compass bearing, so it has to be one.
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

    for name, mesh in meshes.items():
        for patch in range(mesh.patch_count):
            strike_count, dip_count = mesh.cell_extents(patch)
            strike_km, dip_km = mesh.spacing(patch)
            _, _, depth_km = mesh.node_positions(patch)
            located = project_patch(mesh, patch, crs)
            table.add_row(
                name if patch == 0 else "",
                str(patch),
                f"{strike_count}x{dip_count}",
                f"{strike_km:.2f}x{dip_km:.2f}",
                f"{strike_km * strike_count:.1f}x{dip_km * dip_count:.1f}",
                f"{float(located.strike_deg[0, 0]):.1f}/{float(located.dip_deg[0, 0]):.1f}",
                f"{depth_km.min():.1f}-{depth_km.max():.1f}",
            )
    return table


def load_geometry(geometry: Path) -> GeometryConfig:
    """Read a geometry file, rendering a failure rather than raising it.

    Only the decode is wrapped. Anything after it is a bug here rather than a mistake in
    the file, and a traceback is the right thing for that.
    """
    try:
        return read_geometry(geometry)
    except (InvalidFieldValue, MissingField) as error:
        print_config_error(error)
        raise typer.Exit(1) from error
    except tomllib.TOMLDecodeError as error:
        print_syntax_error(error, geometry.read_text(), "toml")
        raise typer.Exit(1) from error
    except json.JSONDecodeError as error:
        print_syntax_error(error, geometry.read_text(), "json")
        raise typer.Exit(1) from error
    except yaml.YAMLError as error:
        console.print(f"[red]{geometry}: {error}[/red]")
        raise typer.Exit(1) from error
    except Exception as error:
        print_config_error(error)
        raise typer.Exit(1) from error


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
    config = load_geometry(geometry)

    meshes = {}
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
        attrs={
            "title": config.title or geometry.stem,
            "geometry_config": geometry.read_text(),
            "surfaces": json.dumps(list(meshes)),
        },
    )

    if not quiet:
        console.print(summarise(meshes, config.crs))
        console.print(f"[green]wrote[/green] {output}")
