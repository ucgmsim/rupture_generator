"""``rupture-generator generate``: a config and a mesh in, a rupture out.

The middle step. It takes the two things that vary independently -- where the fault is,
and what the earthquake on it is -- and produces the model.

# A bent fault is one rupture, and its planes are fused

The generator works on a single `(i, j)` grid, but that does **not** mean one plane. A
fault whose trace bends is one continuous surface cut into one grid whose strike varies
along it -- genslip's `bent` corpus case is exactly that, 32x14 with the strike and rake
changing within the grid -- so the planes of a surface are concatenated along strike and
generated as one.

The grid is an index space. The eikonal sweep steps through `(i, j)` and the transform
runs on the array; neither knows that the strike changed at column 12. What they do
require is that the array *is* a rectangle: the same number of dip rows, and the same
spacing, throughout.

# What is not supported, and how it is detected

A **multi-segment** rupture -- planes with a gap between them, or where the dip angle,
the dip direction or the down-dip width changes -- is not one surface, and the generator
has no rupture front that crosses between two.

The check is geometric rather than a list of scalars: **plane k's last column of nodes
must coincide with plane k+1's first**. Two planes that share their top-edge vertex but
differ in dip, in dip direction or in width diverge below it, so one test covers all
three, and it covers them in the units they matter in -- kilometres of separation on the
fault -- rather than as a comparison of angles.

A gap does not arise inside a surface: connectivity is structural in the config, since a
plane says only where its top edge *ends*. Two surfaces in one geometry file are the gap
case, and they are two ruptures.

`--plane` opts out and generates on one plane alone.
"""

from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from rupture_generator import _core
from rupture_generator.config import read_config
from rupture_generator.config.rupture import (
    GridConfig,
    PointSourceConfig,
    RuptureConfig,
)
from rupture_generator.formats import Format, resolve
from rupture_generator.formats.mesh import read_mesh
from rupture_generator.formats.rupture import to_dataset, to_datatree, write_rupture
from rupture_generator.mesh import project_patch, to_subfault_geometry
from rupture_generator.scripts.errors import console, load_config

SEAM_TOLERANCE_KM = 1.0e-6
"""How far apart two planes' shared nodes may be and still be one surface.

A millimetre. Planes that genuinely share a column are built from the same trace vertex
and the same dip, so they agree to round-off -- around `1e-13` km at fault scale. Planes
that differ in dip, dip direction or width diverge by *kilometres* below the seam: the
`kaikoura` example, at 70 and 55 degrees, separates by 3.5 km at its deepest row.

Six orders above the floor and six below anything real, which is the widest gap available
and means the check never has to be argued about.
"""


def choose_surface(
    meshes: dict[str, _core.RefinedMesh], surface: str | None
) -> tuple[str, _core.RefinedMesh]:
    """Which surface to generate on, or a refusal saying what there was.

    Ambiguity is an error rather than a default. Picking the first of several would run
    silently on a fault nobody chose, and the output would look exactly like the one that
    was wanted.

    Raises
    ------
    typer.Exit
        With the choices listed, so the next command line writes itself.
    """
    if surface is None:
        if len(meshes) != 1:
            console.print(
                f"[red]the mesh holds {len(meshes)} surfaces "
                f"({', '.join(sorted(meshes))}); say --surface[/red]"
            )
            raise typer.Exit(1)
        surface = next(iter(meshes))
    elif surface not in meshes:
        console.print(
            f"[red]no surface {surface!r} in the mesh; it holds "
            f"{', '.join(sorted(meshes))}[/red]"
        )
        raise typer.Exit(1)
    return surface, meshes[surface]


@dataclasses.dataclass(frozen=True)
class Fused:
    """A surface's planes as the single grid the generator runs on.

    Attributes
    ----------
    planes : list of int
        Which planes were fused, in strike order.
    spans : list of tuple
        Each plane's columns in the fused grid, so the fields can be split back.
    depth_km : np.ndarray
        Cell-centre depth on ``(dip, strike)``, concatenated.
    strike_count, dip_count : int
    strike_km, dip_km : float
        The spacing every plane agreed on.
    """

    planes: list[int]
    spans: list[tuple[int, int]]
    depth_km: np.ndarray
    strike_count: int
    dip_count: int
    strike_km: float
    dip_km: float


def fuse(mesh: _core.RefinedMesh, surface: str, only: int | None = None) -> Fused:
    """Concatenate a surface's planes into one grid, or refuse and say why.

    See the module note for what is and is not one surface. `only` generates on a single
    plane instead, which is the opt-out.

    Raises
    ------
    typer.Exit
        If the planes do not form one grid, naming the seam and the gap in kilometres.
    """
    planes = [only] if only is not None else list(range(mesh.patch_count))

    extents = [mesh.cell_extents(patch) for patch in planes]
    dip_counts = {dip for _, dip in extents}
    if len(dip_counts) != 1:
        console.print(
            f"[red]{surface}: its planes are cut into {sorted(dip_counts)} rows down "
            "dip, so they are not one grid. Give them the same dip discretisation, or "
            "say --plane to generate on one.[/red]"
        )
        raise typer.Exit(1)

    for near, far in itertools.pairwise(planes):
        gap_km = seam_gap_km(mesh, near, far)
        if gap_km > SEAM_TOLERANCE_KM:
            console.print(
                f"[red]{surface}: planes {near} and {far} are {gap_km:.3f} km apart at "
                "their shared edge, so this is a multi-segment rupture -- the dip, the "
                "dip direction or the width changes between them. Generating across "
                "that needs a rupture front that crosses a seam, which is not written. "
                "Say --plane to generate on one.[/red]"
            )
            raise typer.Exit(1)

    columns = [strike for strike, _ in extents]
    spacings = [mesh.spacing(patch) for patch in planes]
    strike_km = mean_spacing(surface, "strike", [pair[0] for pair in spacings], columns)
    dip_km = mean_spacing(surface, "dip", [pair[1] for pair in spacings], columns)

    starts = np.cumsum([0, *columns])
    return Fused(
        planes=planes,
        spans=[
            (int(starts[index]), int(starts[index + 1])) for index in range(len(planes))
        ],
        depth_km=np.concatenate(
            [mesh.cell_centres(patch)[2] for patch in planes], axis=1
        ),
        strike_count=int(starts[-1]),
        dip_count=extents[0][1],
        strike_km=strike_km,
        dip_km=dip_km,
    )


SPACING_SPREAD = 0.10
"""How much the planes' cell sizes may differ before they are two resolutions.

The generator runs on one grid with one spacing, so a fused surface needs a single
number. **genslip does the same thing**: it is handed a GSF with a `ds` and a `dw` per
subfault and averages them into the `dstk` and `ddip` it uses everywhere, which
`tests/harness/gsf.py` reproduces as `mean_along_strike_km`.

The bound is what *rounding one requested size* can produce. A plane of length `L` cut at
size `s` gets cells of `L / round(L / s)`, which is within `s / 2L` of `s` -- under 2% on
a 27 km plane at 1 km, and reaching 10% only on a plane of five cells. More than that
means the planes were cut at genuinely different resolutions, which is a request rather
than a rounding, and averaging it would silently split the difference.
"""


def mean_spacing(
    surface: str, axis: str, values: list[float], weights: list[int]
) -> float:
    """One spacing for a fused surface: the mean over its subfaults.

    Weighted by cell count, which is what an average over subfaults *is* -- a plane with
    twice the cells contributes twice.

    Raises
    ------
    typer.Exit
        If the planes disagree by more than rounding could explain.
    """
    spread = (max(values) - min(values)) / min(values)
    if spread > SPACING_SPREAD:
        console.print(
            f"[red]{surface}: its planes have {axis} spacings "
            f"{[round(value, 3) for value in values]} km, a {spread:.0%} spread. The "
            "generator runs on one grid with one spacing, and that is too far apart to "
            "average -- give the planes the same subfault size, or say --plane.[/red]"
        )
        raise typer.Exit(1)
    return float(np.average(values, weights=weights))


def seam_gap_km(mesh: _core.RefinedMesh, near: int, far: int) -> float:
    """How far apart two planes are at their shared column, at the worst node.

    The geometric test that separates one bent surface from two segments. Planes sharing
    a trace vertex agree there whatever their dips, so the disagreement shows *below* the
    top edge and grows with depth -- which is why this is a maximum over the column
    rather than a comparison of the first node.
    """
    last = np.stack(mesh.node_positions(near))[:, :, -1]
    first = np.stack(mesh.node_positions(far))[:, :, 0]
    if last.shape != first.shape:
        # Different dip discretisations: the columns cannot be compared node for node,
        # and the planes are not one grid whatever their positions. `fuse` checks the
        # counts before it gets here, so this is the guard rather than the message.
        return float("inf")
    return float(np.linalg.norm(last - first, axis=0).max())


def fault_grid(fused: Fused, config: RuptureConfig) -> _core.FaultGrid:
    """The core's `FaultGrid`, built from a fused surface and the config's fields.

    Depth comes from the mesh, per subfault, concatenated across the planes. Rake and
    velocity fraction are constants from the config, broadcast -- the core takes them
    per subfault because a mesh may vary them, and a config that could say so per
    subfault would need a way to address subfaults.

    `velocity_fraction` is divided by `alpha_t` before it crosses into the core.
    genslip divides `rvfrac` and every subfault's rupture-speed fraction by the same
    correction it shortens rise time with (`genslip_v5.6.2.c:1443-1445`); the core
    applies `alpha_t` to rise time only, so a caller handing it a raw fraction gets a
    rupture up to 10% slow on a dip-45 reverse fault.
    """
    subfaults = fused.strike_count * fused.dip_count
    padding = config.grid
    correction = _core.alpha_t(
        config.source.average_dip_deg, config.source.average_rake_deg
    )
    return _core.FaultGrid(
        fused.strike_count,
        fused.dip_count,
        padding.padded_strike or GridConfig.default_padding(fused.strike_count),
        padding.padded_dip or GridConfig.default_padding(fused.dip_count),
        fused.strike_km,
        fused.dip_km,
        depth_km=fused.depth_km.ravel(),
        base_rake_deg=np.full(subfaults, config.field.base_rake_deg),
        velocity_fraction=np.full(
            subfaults, config.field.velocity_fraction / correction
        ),
    )


def hypocentre_cell(
    mesh: _core.RefinedMesh, fused: Fused, config: RuptureConfig
) -> tuple[int, int]:
    """Where the rupture starts, as a cell index.

    The config gives arc lengths -- along strike from the ``i = 0`` end, down dip from
    the top edge -- because those mean the same thing whatever the plane is cut into,
    which an index does not. `DEFECTS.md` 17 is this conversion going wrong by one cell
    in each direction and producing a rupture that was smooth, started at zero and
    correlated 0.99+ with the right one.

    On a fused surface the arc length runs along the *whole* fault, so a hypocentre past
    the first plane lands in a later one and its column is offset accordingly.

    Raises
    ------
    typer.Exit
        If the hypocentre is off the fault, with the extent it had to be inside.
    """
    remaining_km = config.hypocentre.strike_km
    for patch, (start, _) in zip(fused.planes, fused.spans, strict=True):
        length_km = float(mesh.strike_arc_km(patch)[-1])
        if remaining_km <= length_km or patch == fused.planes[-1]:
            try:
                strike, dip = mesh.cell_index(
                    patch, min(remaining_km, length_km), config.hypocentre.dip_km
                )
            except ValueError as error:
                console.print(f"[red]hypocentre: {error}[/red]")
                raise typer.Exit(1) from error
            if remaining_km > length_km:
                console.print(
                    f"[red]hypocentre: strike_km {config.hypocentre.strike_km} is past "
                    f"the fault's {sum(float(mesh.strike_arc_km(p)[-1]) for p in fused.planes):.2f} km"
                    "[/red]"
                )
                raise typer.Exit(1)
            return start + strike, dip
        remaining_km -= length_km
    raise typer.Exit(1)


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
        int | None, typer.Option(help="Which plane, when the surface has several.")
    ] = None,
    seed: Annotated[
        int | None, typer.Option(help="Override the config's random seed.")
    ] = None,
    realisation: Annotated[
        int | None, typer.Option(help="Override the config's realisation index.")
    ] = None,
    quiet: Annotated[bool, typer.Option(help="Do not print the summary.")] = False,
) -> None:
    """Generate a rupture model on a mesh."""
    settings = load_config(config, read_config)
    meshes, crs = read_mesh(mesh_path)
    name, mesh = choose_surface(meshes, surface)
    if plane is not None and not 0 <= plane < mesh.patch_count:
        console.print(
            f"[red]{name} has planes 0..{mesh.patch_count - 1}, not {plane}[/red]"
        )
        raise typer.Exit(1)
    # The command line wins over the file, and what actually ran is what gets recorded.
    used_seed = settings.random.seed if seed is None else seed
    used_realisation = (
        settings.random.realisation if realisation is None else realisation
    )

    # The geometry is inside the `try` as well as the generation. `fuse` reads
    # `mesh.spacing`, which refuses a non-uniform patch, and `hypocentre_cell` reads
    # `mesh.cell_index`, which refuses a hypocentre off the fault -- both `ValueError`
    # from the same `Refused` conversion the core uses. They used to sit above it, so a
    # mesh file that had been hand-edited into non-uniformity came back as a raw
    # traceback where every other refusal in this command prints one red line.
    try:
        fused = fuse(mesh, name, plane)
        grid = fault_grid(fused, settings)
        strike, dip = hypocentre_cell(mesh, fused, settings)

        if isinstance(settings.source, PointSourceConfig):
            rupture = _core.generate_point_source(
                grid,
                settings.velocity_model.to_core(),
                settings.source.to_core(),
                settings.timing.to_core(),
                hypocentre_strike=strike,
                hypocentre_dip=dip,
            )
        else:
            rupture = _core.generate_rupture(
                grid,
                settings.velocity_model.to_core(),
                settings.source.to_core(),
                settings.slip.to_core(fused.strike_km, fused.dip_km),
                settings.timing.to_core(),
                seed=used_seed,
                realisation=used_realisation,
                hypocentre_strike=strike,
                hypocentre_dip=dip,
            )
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error

    chosen = resolve(output, output_format)
    if chosen in {Format.SRF, Format.SRF_HDF5}:
        write_srf(rupture, mesh, fused, crs, settings, output, chosen)
    else:
        tree = to_datatree(
            {
                f"{name}/plane_{patch}": to_dataset(
                    slice_rupture(rupture, fused, index),
                    mesh,
                    patch,
                    crs,
                    hypocentre_km=plane_hypocentre(fused, settings, index, strike),
                )
                for index, patch in enumerate(fused.planes)
            },
            crs,
            {name: (mesh.origin.easting_km, mesh.origin.northing_km)},
            attrs={
                "title": settings.title or config.stem,
                "config": config.read_text(),
                "surface": name,
                "seed": used_seed,
                "realisation": used_realisation,
                "rng_engine": "pcg",
                "moment_dyne_cm": rupture.moment_dyne_cm,
                "alpha_t": rupture.alpha_t,
                "sample_interval_s": rupture.sample_interval_s,
            },
        )
        write_rupture(tree, output, format=chosen)

    if not quiet:
        report(rupture, name, fused, settings, used_seed, used_realisation)
    console.print(f"[green]wrote[/green] {output}")


@dataclasses.dataclass(frozen=True)
class PlaneRupture:
    """One plane's share of a fused rupture, shaped like a `GeneratedRupture`.

    Duck-typed rather than a real one, because the core builds those and this is a
    *view* of one that already exists. `to_dataset` reads exactly these names.
    """

    sample_interval_s: float
    moment_dyne_cm: float
    alpha_t: float
    slip_cm: np.ndarray
    rake_deg: np.ndarray
    onset_s: np.ndarray
    rise_time_s: np.ndarray
    slip_rate: np.ndarray
    slip_rate_offsets: np.ndarray


def plane_hypocentre(
    fused: Fused, settings: RuptureConfig, index: int, strike_cell: int
) -> tuple[float, float] | None:
    """Where the rupture started in *this* plane's arc lengths, or `None`.

    `to_dataset` documents `hypocentre_km` as the plane's own arc lengths, and the
    coordinate it is read against -- the group's `strike_km` -- runs from zero at each
    plane's own edge. The config gives one arc length across the whole fused surface,
    so writing it unchanged into every group put the hypocentre off the end of every
    plane but one, and claimed three hypocentres for one earthquake.

    A plane that does not contain the hypocentre gets no attribute rather than a wrong
    one; `to_dataset` omits the pair when this returns `None`.
    """
    start, stop = fused.spans[index]
    if not start <= strike_cell < stop:
        return None
    return (
        settings.hypocentre.strike_km - start * fused.strike_km,
        settings.hypocentre.dip_km,
    )


def slice_rupture(
    rupture: _core.GeneratedRupture, fused: Fused, index: int
) -> PlaneRupture:
    """The columns of a fused rupture that belong to one plane.

    The generator ran on the whole surface, so its fields are flat over
    `(dip, strike_total)` along-strike fastest. Splitting them means taking a *column
    range* out of each dip row, not a contiguous slice -- which is the kind of thing that
    looks right when the fault happens to have one row.

    The pulses come with it: each subfault's samples are found through the offsets and
    re-concatenated, so the plane's CSR is self-contained rather than indexing into the
    surface's.
    """
    start, stop = fused.spans[index]
    rows, columns = fused.dip_count, fused.strike_count
    keep = (
        np.arange(rows)[:, None] * columns + np.arange(start, stop)[None, :]
    ).ravel()

    offsets = np.asarray(rupture.slip_rate_offsets, dtype=np.int64)
    samples = np.asarray(rupture.slip_rate, dtype=np.float64)
    pulses = [samples[offsets[cell] : offsets[cell + 1]] for cell in keep]
    lengths = np.array([len(pulse) for pulse in pulses], dtype=np.int64)

    return PlaneRupture(
        sample_interval_s=rupture.sample_interval_s,
        moment_dyne_cm=rupture.moment_dyne_cm,
        alpha_t=rupture.alpha_t,
        slip_cm=rupture.slip_cm[keep],
        rake_deg=rupture.rake_deg[keep],
        onset_s=rupture.onset_s[keep],
        rise_time_s=rupture.rise_time_s[keep],
        slip_rate=(np.concatenate(pulses) if pulses else np.empty(0, dtype=np.float64)),
        slip_rate_offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
    )


def write_srf(
    rupture: _core.GeneratedRupture,
    mesh: _core.RefinedMesh,
    fused: Fused,
    crs: object,
    settings: RuptureConfig,
    output: Path,
    chosen: Format,
) -> None:
    """Assemble and write an SRF, in text or SW4's HDF5.

    One `PLANE` record per plane, which is what an SRF is for -- genslip emits one per
    segment too, and `README.md`'s second trap is that the point order then follows the
    *segments* rather than the GSF.

    The plane header genslip *recomputes* -- with a tangent-plane approximation off by
    43 m on a crustal fault and 1.9 km at subduction scale -- is derived here from the
    mesh, which knows where the fault actually is.
    """
    from rupture_generator import srf as srf_module
    from rupture_generator.assemble import to_srf_file

    if len(fused.planes) != 1:
        console.print(
            "[red]writing a bent fault to an SRF needs one PLANE record per plane, "
            "which `assemble.to_srf_file` does not build yet. Use .h5, or --plane.[/red]"
        )
        raise typer.Exit(1)

    patch = fused.planes[0]
    geometry = to_subfault_geometry(mesh, patch, crs)
    strike_count, dip_count = mesh.cell_extents(patch)
    strike_arc = mesh.strike_arc_km(patch)
    dip_arc = mesh.dip_arc_km(patch)
    located = project_patch(mesh, patch, crs)

    # The SRF's plane centre is the middle of the whole plane, and `shyp` is measured
    # from the along-strike *centre* rather than from the end -- which is the one place
    # the in-fault convention this package uses has to be converted.
    header = srf_module.PlaneHeader(
        centre_longitude_deg=float(np.mean(geometry.longitude_deg)),
        centre_latitude_deg=float(np.mean(geometry.latitude_deg)),
        strike_count=strike_count,
        dip_count=dip_count,
        length_km=float(strike_arc[-1]),
        width_km=float(dip_arc[-1]),
        strike_deg=float(located.strike_deg[0, 0]),
        dip_deg=float(located.dip_deg[0, 0]),
        top_depth_km=float(mesh.node_positions(patch)[2].min()),
        hypocentre_strike_km=settings.hypocentre.strike_km
        - float(strike_arc[-1]) / 2.0,
        hypocentre_dip_km=settings.hypocentre.dip_km,
    )

    shear_speed_km_s, density_g_cm3 = sample_materials(settings, geometry.depth_km)
    srf_file = to_srf_file(rupture, geometry, header, shear_speed_km_s, density_g_cm3)

    if chosen is Format.SRF:
        srf_module.write_srf(output, srf_file)
    else:
        srf_file.write_sw4_hdf5(output)


def sample_materials(
    settings: RuptureConfig, depth_km: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Shear speed and density at each subfault, from the layered model.

    The same clamping the core does: a subfault below the deepest layer takes that
    layer's properties rather than an extrapolation, because a subfault below the model
    is a modelling error and not a reason to invent numbers for it.
    """
    bottoms = np.asarray(settings.velocity_model.bottom_depth_km)
    layer = np.minimum(
        np.searchsorted(bottoms, depth_km, side="left"), len(bottoms) - 1
    )
    return (
        np.asarray(settings.velocity_model.shear_speed_km_s)[layer],
        np.asarray(settings.velocity_model.density_g_cm3)[layer],
    )


def report(
    rupture: _core.GeneratedRupture,
    surface: str,
    fused: Fused,
    settings: RuptureConfig,
    seed: int,
    realisation: int,
) -> None:
    """What was generated, in the numbers a reader checks first."""
    from rich.table import Table

    magnitude = settings.source.magnitude
    planes = (
        f"{len(fused.planes)} planes fused"
        if len(fused.planes) > 1
        else f"plane {fused.planes[0]}"
    )
    table = Table(title=f"{surface}, {planes}", title_justify="left", show_header=False)
    table.add_column("", style="bold")
    table.add_column("", justify="right")

    strike_count, dip_count = rupture.shape
    for label, value in (
        ("subfaults", f"{strike_count}x{dip_count}"),
        ("magnitude", f"{magnitude:.2f}"),
        ("moment", f"{rupture.moment_dyne_cm:.4g} dyne-cm"),
        (
            "slip",
            f"mean {rupture.slip_cm.mean():.1f}, max {rupture.slip_cm.max():.1f} cm",
        ),
        ("rise time", f"mean {rupture.rise_time_s.mean():.2f} s"),
        ("onset", f"0 to {rupture.onset_s.max():.2f} s"),
        ("random", f"pcg, seed {seed}, realisation {realisation}"),
    ):
        table.add_row(label, value)
    console.print(table)
