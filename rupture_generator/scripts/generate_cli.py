"""``rupture-generator generate``: a config and a mesh in, a rupture out.

The middle of the three subcommands. It reads a mesh file rather than a geometry
description because a geometry is digitised once and reused across every realisation
run on it -- rebuilding it per run would be both wasteful and a chance for it to come
out different -- so the boundary between the two steps is a file.

What is here is the I/O, the option handling and the summary. The pipeline is
`rupture_generator.pipeline`, and the stage order lives there and only there.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.table import Table

from rupture_generator import assemble, moment, pipeline
from rupture_generator.config import read_config
from rupture_generator.config.rupture import RuptureConfig
from rupture_generator.formats import Format, resolve
from rupture_generator.formats.mesh import read_mesh
from rupture_generator.formats.rupture import to_datatree, write_rupture
from rupture_generator.mesh import RuptureMesh, fuse, validate_chart
from rupture_generator.scripts.errors import console, load_config
from rupture_generator.srf import write_srf


def named_segments(
    meshes: dict[str, list[RuptureMesh]],
    surface: str | None,
    plane: int | None,
) -> dict[str, RuptureMesh]:
    """The validated segments to generate on, named as the causality tree names them.

    Without ``--surface`` every surface in the mesh takes part, which is what a
    multi-fault rupture is; with it, one does. A surface that fuses to a single
    segment keeps its own name, and one whose planes do not all share a seam becomes
    ``surface:0``, ``surface:1`` -- because those parts are what rupture.

    ``--plane`` selects **one plane's own chart**, before fusion: it is how a bent
    fault gets generated a plane at a time when the fused surface is not what was
    wanted. It needs a surface to be unambiguous.

    Raises
    ------
    ValueError
        If a name or plane index is not in the mesh, or ``--plane`` is given without
        a surface to apply it to.
    """
    if surface is not None and surface not in meshes:
        raise ValueError(
            f"the mesh holds no surface called {surface!r}; it has "
            f"{', '.join(sorted(meshes))}"
        )
    if plane is not None and surface is None and len(meshes) != 1:
        raise ValueError(
            "--plane says which plane of one surface to generate on, so it needs "
            f"--surface to say which of {', '.join(sorted(meshes))}"
        )

    chosen = {surface: meshes[surface]} if surface is not None else dict(meshes)

    segments: dict[str, RuptureMesh] = {}
    for name, charts in chosen.items():
        if plane is not None:
            if not 0 <= plane < len(charts):
                raise ValueError(
                    f"{name!r} has {len(charts)} planes, numbered 0 to "
                    f"{len(charts) - 1}, so there is no plane {plane}"
                )
            parts = [charts[plane]]
        else:
            parts = fuse(charts)

        for part in parts:
            validate_chart(part)
        if len(parts) == 1:
            segments[name] = parts[0]
        else:
            for index, part in enumerate(parts):
                segments[f"{name}:{index}"] = part
    return segments


def report(
    config: RuptureConfig,
    realisation: pipeline.Realisation,
    *,
    seed: int,
    realisation_index: int,
) -> Table:
    """What was generated, in the numbers someone would check it by.

    Slip and moment are the two anyone reads first: a mean slip that is an order out
    says the magnitude or the area is wrong, and both are visible here beside each
    other.
    """
    table = Table(
        title=config.title or "Rupture",
        title_justify="left",
        caption=f"seed {seed}, realisation {realisation_index}",
        caption_justify="left",
    )
    table.add_column("quantity", justify="left")
    table.add_column("value", justify="right")

    slip_m = np.concatenate(
        [
            segment["slip_m"].to_numpy().ravel()
            for segment in realisation.segments.values()
        ]
    )
    rise_s = np.concatenate(
        [
            segment["rise_time_s"].to_numpy().ravel()
            for segment in realisation.segments.values()
        ]
    )
    onset_s = np.concatenate(
        [
            segment["onset_s"].to_numpy().ravel()
            for segment in realisation.segments.values()
        ]
    )

    for name, value in (
        ("segments", ", ".join(realisation.segments)),
        ("subfaults", f"{slip_m.size}"),
        ("magnitude", f"{config.source.magnitude:.2f}"),
        ("moment", f"{realisation.moment_newton_m:.4g} N m"),
        ("slip mean / max", f"{slip_m.mean():.3f} / {slip_m.max():.3f} m"),
        ("rise time mean", f"{rise_s.mean():.3f} s"),
        ("onset range", f"{onset_s.min():.2f} to {onset_s.max():.2f} s"),
        ("truncated", f"{realisation.truncated_fraction:.1%}"),
    ):
        table.add_row(name, value)

    # Who triggered whom, and where the front crossed. On a single-fault rupture the
    # tree is one node and there is nothing to say.
    for child, jump in realisation.jumps.items():
        table.add_row(
            f"{realisation.tree[child]} to {child}",
            f"{jump.distance_km:.2f} km at {jump.arrival_s:.2f} s",
        )
    return table


def generate(
    config: Annotated[
        Path,
        typer.Argument(
            help="Rupture config to read (TOML, YAML or JSON).",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    mesh_path: Annotated[
        Path,
        typer.Argument(
            metavar="MESH",
            help="Mesh file from `rupture-generator mesh`.",
            exists=True,
            readable=True,
        ),
    ],
    output: Annotated[Path, typer.Argument(help="Rupture file to write.")],
    output_format: Annotated[
        Format,
        typer.Option(
            "--format", help="Output format. Usually inferred from the extension."
        ),
    ] = Format.INFERRED,
    surface: Annotated[
        str | None, typer.Option(help="Which surface, when the mesh holds several.")
    ] = None,
    plane: Annotated[
        int | None,
        typer.Option(
            help="Generate on one plane alone, rather than the fused surface."
        ),
    ] = None,
    seed: Annotated[
        int | None, typer.Option(help="Override the config's random seed.")
    ] = None,
    realisation: Annotated[
        int | None, typer.Option(help="Override the config's realisation index.")
    ] = None,
    quiet: Annotated[bool, typer.Option(help="Do not print the summary.")] = False,
) -> None:
    """Generate a kinematic rupture model on a mesh."""
    rupture_config: RuptureConfig = load_config(config, read_config)

    # An override replaces the file's value, and **what actually ran is what gets
    # recorded**: a rupture whose attrs disagree with how it was made cannot be
    # regenerated from itself.
    if seed is not None:
        rupture_config.random.seed = seed
    if realisation is not None:
        rupture_config.random.realisation = realisation

    meshes, crs, propagation = read_mesh(mesh_path)

    # Everything geometric and everything physical, in one place, so a refusal from
    # any of it renders the same way -- one red line naming the cause, rather than a
    # traceback for a mistake in a file.
    try:
        segments = named_segments(meshes, surface, plane)
        result = pipeline.generate(
            rupture_config, segments, crs, propagation_config=propagation
        )
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    chosen = resolve(output, output_format)
    if chosen in (Format.SRF, Format.SRF_HDF5):
        _write_srf(rupture_config, result, output)
    else:
        tree = to_datatree(
            {
                f"{name.replace(':', '_')}/segment": segment
                for name, segment in result.segments.items()
            },
            crs,
            attrs={
                "title": rupture_config.title or config.stem,
                "config": config.read_text(),
                # The causality tree, as JSON: which segment triggered which, and
                # where the front crossed onto each. Without it a multi-fault
                # rupture file is a set of faults that happen to be in one place.
                "causality_tree": json.dumps(result.tree),
                "jumps": json.dumps(
                    {
                        name: dataclasses.asdict(jump)
                        for name, jump in result.jumps.items()
                    }
                ),
                "seed": rupture_config.random.seed,
                "realisation": rupture_config.random.realisation,
                "moment_newton_m": result.moment_newton_m,
            },
        )
        write_rupture(tree, output, format=output_format)

    if not quiet:
        console.print(
            report(
                rupture_config,
                result,
                seed=rupture_config.random.seed,
                realisation_index=rupture_config.random.realisation,
            )
        )
        console.print(f"[green]wrote[/green] {output}")


def _write_srf(
    config: RuptureConfig, result: pipeline.Realisation, output: Path
) -> None:
    """The SRF path, which needs the material properties the rupture file does not
    store -- an SRF version 2.0 point carries shear speed and density, and the
    velocity model is the only thing that knows them."""
    shear_speeds = []
    densities = []
    bottoms = np.asarray(config.velocity_model.bottom_depth_km)
    speeds = np.asarray(config.velocity_model.shear_speed_km_s)
    layer_densities = np.asarray(config.velocity_model.density_g_cm3)

    for segment in result.segments.values():
        depth_km = segment["centre_depth_km"].to_numpy()
        shear_speed, _ = moment.sample_velocity_model(
            depth_km, bottoms, speeds, layer_densities
        )
        layer = np.minimum(
            np.searchsorted(bottoms, depth_km, side="left"), len(bottoms) - 1
        )
        shear_speeds.append(shear_speed.ravel())
        densities.append(layer_densities[layer].ravel())

    write_srf(
        output,
        assemble.to_srf_file(list(result.segments.values()), shear_speeds, densities),
    )


__all__ = ["generate", "named_segments", "report"]
