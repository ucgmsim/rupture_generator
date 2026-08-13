"""Native rupture files, one per run, for opening in the viewer.

Run as ``uv run python -m curvature.ruptures`` for Hikurangi and
``uv run python -m curvature.ruptures puysegur_fiordland`` or ``... puyseguer`` for the
Puysegur surfaces, matching :mod:`curvature.run`'s own invocation -- including its second
argument, the magnitude, which every filename then carries. It writes
``curvature/ruptures/<prefix><scenario>_<geometry>.rupture.h5``: every row of the
interface's own run matrix times the two geometries, plus the true-depth counterfactual at
each hypocentre the study decomposes at, which exists only as a flat model and so
contributes one file per hypocentre rather than two. Hikurangi takes no prefix, so the
eight files it already wrote keep their names and the rows added since land beside them.
Each file carries the geometry, the four per-subfault fields, the materials, the
wavefront, the onset and the slip-rate pulses.

**The pulses are streamed, not held.** At 1.39 M faces and a 5.8 s mean rise time sampled
at 0.02 s a rupture's pulses are about 4e8 samples, 3.2 GB of ``f64``, on a machine with
20 GB free and other work running.
:func:`~rupture_generator.triangular.pipeline.write_rupture_mesh` takes the pulse model
rather than the pulses and runs the synthesis a block of faces at a time, appending each
block to the file and dropping it, so the peak is one block. Nothing is approximated: the
stored arrays are the same two CSR arrays under the same names.

This module recomputes the rupture rather than reading it from
:mod:`curvature.run`'s output, and that is a deliberate cost. The fields are 1.39 M
values each and there are nine combinations of them, so carrying them between the two
programs would mean a gigabyte of intermediate on disk to save four minutes of arithmetic
-- and recomputing from the same seeds is also a check that the seeds do what the study
says they do.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pyproj

from curvature import model
from curvature.geometry import NZTM, MeshPair, build_pair
from curvature.run import INTERFACES, Interface, hypocentres, prefix
from rupture_generator.realisation import Realisation
from rupture_generator.triangular.mesh import TriangleMesh
from rupture_generator.triangular.pipeline import write_rupture_mesh

HERE = Path(__file__).resolve().parent
RUPTURES = HERE / "ruptures"

TRUE_DEPTH_CONDITION = "truedepth"
"""Which reading of the counterfactual gets a file.

The one the study reports -- every quantity a subfault's depth determines, read at the
true depth. The narrower ``truedepth_materials_only`` reading differs from it in the rise
time and the pulse shape alone, and ``results.json`` carries both; a second 3 GB file for
that difference would not be read.
"""


def _annotate(
    pair: MeshPair,
    vertices_km: np.ndarray,
    velocity: model.VelocityModel,
    materials: model.Materials,
    fields: model.Fields,
    slip_m: np.ndarray,
    travel_s: np.ndarray,
    onset_s: np.ndarray,
    face: int,
) -> TriangleMesh:
    """One model as a :class:`TriangleMesh` carrying everything a reader wants.

    The field names are the pipeline's own, so a file written here and a file written by
    ``rupture-generator generate`` are read by the same code.

    Parameters
    ----------
    pair : MeshPair
    vertices_km : FloatArray
        ``(V, 3)`` this model's vertices.
    velocity : model.VelocityModel
        Which model the densities came from, so the per-face density is read off the
        layers that were actually used rather than inferred from the result.
    materials : model.Materials
    fields : model.Fields
    slip_m, travel_s, onset_s : FloatArray
        ``(F,)`` per subfault.
    face : int
        The hypocentre's face, for the arc-length attributes a viewer reads.

    Returns
    -------
    TriangleMesh
    """
    chart = pair.chart(vertices_km)
    densities = np.asarray(velocity.density_g_cm3)
    density = densities[np.minimum(materials.layer, len(densities) - 1)]
    strike_arc_km = chart.strike_arc_km()[pair.faces[face]].mean()
    dip_arc_km = chart.dip_arc_km()[pair.faces[face]].mean()
    return chart.with_fields(
        slip_m=slip_m,
        rise_time_s=fields.rise_time_s,
        rake_deg=fields.rake_deg,
        onset_s=onset_s,
        wavefront_s=travel_s,
        onset_perturbation=fields.perturbation,
        shear_speed_kms=materials.shear_speed_km_s,
        rigidity_pa=materials.rigidity_pa,
        density_g_cm3=density,
    ).with_attrs(
        truncated_fraction=fields.truncated_fraction,
        hypocentre_strike_km=float(strike_arc_km),
        hypocentre_dip_km=float(dip_arc_km),
    )


def _write(chart: TriangleMesh, path: Path, pulse: object, started: float) -> None:
    """Stream one rupture to disk and say what it cost.

    **Written under a temporary name and renamed into place**, which is not tidiness.
    These files are 1 to 3 GB and take tens of seconds, they are *rewritten* whenever
    the study is rerun, and something else -- a viewer, another agent -- reads them from
    the same directory meanwhile. Writing in place fails both ways round: HDF5 takes an
    advisory lock, so a reader holding the old file makes ``h5py.File(path, "w")`` raise
    ``BlockingIOError`` *after* it has already truncated, which leaves a published
    rupture at zero bytes; and a reader that opens midway through a successful write
    sees a partial file with no way to tell. A fresh name cannot be locked by anyone,
    and :meth:`pathlib.Path.replace` is atomic within a filesystem, so a reader sees
    either the whole old file or the whole new one.

    The partial is removed if the write fails, so a crash cannot leave 3 GB of rubble
    behind under a name nobody recognises.

    Parameters
    ----------
    chart : TriangleMesh
        Annotated by :func:`_annotate`; its own ``surface`` names the segment, so a file
        cannot end up naming an interface it was not built on.
    path : Path
    pulse : pulses.PulseParams
    started : float
        When this rupture's arithmetic began, for the line printed after it.
    """
    realisation = Realisation(segments={chart.surface: chart}, crs=pyproj.CRS(NZTM))
    # A dotted prefix rather than a suffix: `write_rupture_mesh` dispatches the
    # streaming route on the extension, so the temporary name has to keep `.h5` and only
    # the stem is free to say what it is.
    partial = path.with_name(f".partial-{path.name}")
    try:
        write_rupture_mesh(realisation, partial, pulse)
        size = partial.stat().st_size
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    print(
        f"{path.name}: {size / 1e9:.2f} GB in {time.perf_counter() - started:.0f} s",
        flush=True,
    )


def write_interface(interface: Interface, magnitude: float = model.MAGNITUDE) -> None:
    """Recompute every run on one interface and write its rupture files.

    Parameters
    ----------
    interface : Interface
    magnitude : float, optional
        The event. Defaults to :data:`~curvature.model.MAGNITUDE`, which writes the
        files under the bare names they already have; any other magnitude carries
        :func:`~curvature.run.tag`'s suffix in every filename, so two magnitudes' files
        sit in one directory and neither can be mistaken for the other.
    """
    named = prefix(interface.name, magnitude)
    pair = build_pair(interface.path)
    taper_weight = model.lateral_taper(pair)
    located = hypocentres(pair, interface.dip_position_km)

    geometries = {"curved": pair.curved_km, "flat": pair.flat_km}
    levels = {"curved": pair.curved_levels, "flat": pair.flat_levels}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samplers = {
            name: model.Sampler(pair, vertices, levels[name], magnitude)
            for name, vertices in geometries.items()
        }

    velocities = {"constant": model.CONSTANT, "standard": model.STANDARD}
    materials = {
        name: {
            geometry: model.materials_of(pair.centres_km(vertices)[:, 2], velocity)
            for geometry, vertices in geometries.items()
        }
        for name, velocity in velocities.items()
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fields = {
            name: {
                geometry: model.draw_fields(
                    pair,
                    samplers[geometry],
                    vertices,
                    velocity,
                    taper_weight,
                    magnitude,
                )
                for geometry, vertices in geometries.items()
            }
            for name, velocity in velocities.items()
        }

    areas = {
        geometry: pair.areas_km2(vertices) for geometry, vertices in geometries.items()
    }
    slip = {
        name: {
            geometry: model.slip_metres(
                fields[name][geometry].pattern,
                materials[name][geometry].rigidity_pa,
                areas[geometry],
                magnitude,
            )
            for geometry in geometries
        }
        for name in velocities
    }

    for scenario in interface.scenarios:
        face = located[scenario.site]["face_index"]
        name = scenario.velocity.name
        speed = model.speed_params(pair, scenario.velocity)
        pulse = model.pulse_params(scenario.velocity)

        for geometry, vertices in geometries.items():
            started = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                travel_s, _ = model.travel_times(
                    pair, vertices, materials[name][geometry], speed, face
                )
            onset_s = model.onset_of(
                travel_s, fields[name][geometry].perturbation, face
            )
            chart = _annotate(
                pair,
                vertices,
                scenario.velocity,
                materials[name][geometry],
                fields[name][geometry],
                slip[name][geometry],
                travel_s,
                onset_s,
                face,
            )
            _write(
                chart,
                RUPTURES / f"{named}{scenario.name}_{geometry}.rupture.h5",
                pulse,
                started,
            )

    # The counterfactual, at each hypocentre the study decomposes at. A flat model in
    # every respect except which depth its rock is read at, so it is written on the flat
    # vertices and its file says `flat` -- the difference from `<site>_standard_flat` is
    # inside, in `rigidity_pa` and `shear_speed_kms`, which is exactly the difference the
    # refactor would make. Its materials and its fields are the same arrays whatever the
    # hypocentre is, so they are built once and only the front is per site.
    speed = model.speed_params(pair, model.STANDARD)
    pulse = model.pulse_params(model.STANDARD)
    true_depth = model.true_depth_materials(
        pair.centres_km(pair.curved_km)[:, 2],
        pair.centres_km(pair.flat_km)[:, 2],
        model.STANDARD,
        ramps_read_true_depth=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        counterfactual = model.draw_fields(
            pair,
            samplers["flat"],
            pair.curved_km,
            model.STANDARD,
            taper_weight,
            magnitude,
        )
    slip_m = model.slip_metres(
        counterfactual.pattern, true_depth.rigidity_pa, areas["flat"], magnitude
    )

    for site in interface.decomposed_sites:
        started = time.perf_counter()
        face = located[site]["face_index"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            travel_s, _ = model.travel_times(
                pair, pair.flat_km, true_depth, speed, face
            )
        onset_s = model.onset_of(travel_s, counterfactual.perturbation, face)
        chart = _annotate(
            pair,
            pair.flat_km,
            model.STANDARD,
            true_depth,
            counterfactual,
            slip_m,
            travel_s,
            onset_s,
            face,
        )
        _write(
            chart,
            RUPTURES / f"{named}{site}_standard_{TRUE_DEPTH_CONDITION}_flat.rupture.h5",
            pulse,
            started,
        )


def main() -> None:
    """Write one interface's rupture files, at one magnitude."""
    RUPTURES.mkdir(parents=True, exist_ok=True)
    name = sys.argv[1] if len(sys.argv) > 1 else "hikurangi"
    if name not in INTERFACES:
        raise SystemExit(f"no such interface {name!r}: choose from {list(INTERFACES)}")
    magnitude = float(sys.argv[2]) if len(sys.argv) > 2 else model.MAGNITUDE
    write_interface(INTERFACES[name], magnitude)


if __name__ == "__main__":
    main()
