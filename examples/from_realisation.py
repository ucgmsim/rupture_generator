"""Convert a workflow ``realisation.json`` into this package's two config files.

The workflow's realisation format is one JSON document holding everything a simulation
needs -- geometry, source, velocity model, seeds, and the parameters of four other
programs. This pulls out the parts that describe *the earthquake* and writes them as a
geometry file and a rupture config.

Run it as::

    python examples/from_realisation.py path/to/realisation.json examples/hope

which writes ``examples/hope.geometry.toml`` and ``examples/hope.toml``.

# What is carried across, and what is not

Carried: the fault corners, the causality tree, the per-fault magnitudes and rakes,
the velocity model, the hypocentre, the tapers, the rupture-speed profile and the
resolution.

**Not carried: the jump points.** The realisation records where the rupture crossed
between faults, fitted by closest approach. This pipeline computes them instead, from
the solved wavefront on the parent fault -- so importing them would be importing the
answer to a question this generator asks itself, and asks differently.

Also not carried: everything belonging to the other programs in the workflow --
`emod3d`, `hf`, `bb`, `im`, the domain and the 3-D velocity model. They describe how
the ground motion is simulated, not what the earthquake is.

# The corners are quads, four per plane

Each fault's ``corners`` list is a flat run of four-point groups, one per plane:
two points on the surface trace and two directly below them at the fault's bottom
depth. Consecutive planes share a trace point, so the trace is recovered by taking
the first point of each group and the last point of the final one.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def trace_and_geometry(corners: list[dict]) -> tuple[list[dict], float, float, str]:
    """One fault's trace points, its dip, its bottom depth, and which way it dips.

    The dip is recovered from the geometry rather than read from a field, because the
    realisation does not carry one: it carries the corner positions the dip produced.
    Taking the mean over the fault's planes is exact where they agree and is the only
    thing available where they do not.
    """
    planes = len(corners) // 4
    trace = [corners[4 * plane] for plane in range(planes)]
    trace.append(corners[4 * (planes - 1) + 1])

    latitudes = [point["latitude"] for point in corners]
    lat0 = float(np.mean(latitudes))
    east_per_degree = 111.32 * math.cos(math.radians(lat0))

    dips = []
    sides = []
    for plane in range(planes):
        top_a, top_b, bottom_b, _ = corners[4 * plane : 4 * plane + 4]
        # The down-dip step from the far trace point to the point below it.
        east = (bottom_b["longitude"] - top_b["longitude"]) * east_per_degree
        north = (bottom_b["latitude"] - top_b["latitude"]) * 110.57
        down = (bottom_b["depth"] - top_b["depth"]) / 1000.0
        horizontal = math.hypot(east, north)
        if horizontal > 1.0e-9:
            dips.append(math.degrees(math.atan2(down, horizontal)))
        else:
            dips.append(90.0)

        # Which side of the trace the fault hangs on: the sign of the cross product
        # of the along-strike step with the down-dip step, in the horizontal plane.
        along_east = (top_b["longitude"] - top_a["longitude"]) * east_per_degree
        along_north = (top_b["latitude"] - top_a["latitude"]) * 110.57
        sides.append(np.sign(along_east * north - along_north * east))

    bottom_depth_km = max(point["depth"] for point in corners) / 1000.0
    dip_deg = float(np.clip(np.mean(dips), 1.0, 90.0))
    # `right` in this package means the fault dips away to the right of the walk along
    # the trace, which is a negative cross product in an east-north frame.
    direction = "right" if float(np.mean(sides)) < 0 else "left"
    return trace, dip_deg, bottom_depth_km, direction


def simplify(
    trace: list[dict], dip_deg: float, bottom_depth_km: float, east_per_degree: float
) -> tuple[list[dict], float]:
    """Drop trace points the fault's own depth cannot support.

    A digitised trace can be finer than the surface beneath it. At a bend the two
    planes share a bottom column placed along the bisector, so the bottom edge of a
    plane shortens by about ``reach * sin(deflection / 2)`` at each end, where
    ``reach = depth / tan(dip)`` is how far the fault steps horizontally on its way
    down. When that exceeds the plane's own length the surface closes to a triangle
    at depth and there is no grid on it.

    Measured on this scenario: "Alpine: George to Jacksons" has a 0.50 km plane on a
    fault with a 3.5 km reach, and a 16 degree bend at one end swings its bottom edge
    0.50 km sideways -- the plane's whole length. Its bottom row came out 0.2 m wide
    against 100 m at the top, and chart validation refused it, correctly.

    So the trace is thinned until every plane keeps at least half its length at
    depth. This *moves the fault*, so the worst displacement is returned and the
    converter reports it: a few hundred metres on a 20 km-deep fault is well inside
    what the trace itself is known to, and a reader should be told rather than
    trusted to assume.

    Returns
    -------
    tuple
        The kept trace points, and the furthest any dropped point was from the line
        that replaced it, in kilometres.
    """
    reach_km = bottom_depth_km / math.tan(math.radians(dip_deg))
    kept = list(trace)
    worst_km = 0.0

    def position(point: dict) -> np.ndarray:
        return np.array(
            [point["longitude"] * east_per_degree, point["latitude"] * 110.57]
        )

    def survives(points: list[dict]) -> list[int]:
        """Which interior points leave a plane too short for its own depth."""
        offenders = []
        for index in range(1, len(points) - 1):
            before = position(points[index]) - position(points[index - 1])
            after = position(points[index + 1]) - position(points[index])
            lengths = (float(np.linalg.norm(before)), float(np.linalg.norm(after)))
            if min(lengths) < 1.0e-9:
                offenders.append(index)
                continue
            turn = math.degrees(
                math.acos(
                    float(
                        np.clip(
                            np.dot(before, after) / (lengths[0] * lengths[1]),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            lost_km = reach_km * math.sin(math.radians(turn) / 2.0)
            if min(lengths) < 2.0 * lost_km:
                offenders.append(index)
        return offenders

    while True:
        offenders = survives(kept)
        if not offenders or len(kept) <= 2:
            break

        # Drop the one whose removal moves the trace least.
        def deviation(index: int) -> float:
            a, b = position(kept[index - 1]), position(kept[index + 1])
            span = b - a
            length = float(np.linalg.norm(span))
            if length < 1.0e-12:
                return 0.0
            offset = position(kept[index]) - a
            return float(abs(np.cross(span, offset)) / length)

        victim = min(offenders, key=deviation)
        worst_km = max(worst_km, deviation(victim))
        kept.pop(victim)

    return kept, worst_km


def geometry_toml(realisation: dict, resolution_km: float) -> str:
    """The fault system, as a geometry file."""
    geometries = realisation["sources"]["source_geometries"]
    tree = realisation["rupture_propagation"]["rupture_causality_tree"]

    lines = [
        "# The Alpine-Hope joint rupture, converted from a workflow realisation.",
        "#",
        "# Every fault here is a run of planes hanging from a digitised trace. The",
        "# propagation is stated rather than sampled, because this scenario is a",
        "# specific one: the hazard model chose which fault triggers which, and",
        "# recomputing it would be generating a different earthquake.",
        "",
        "schema_version = 1",
        'crs = "EPSG:2193"',
        f'title = "{realisation["metadata"]["name"]}"',
        "",
    ]

    moved: list[str] = []
    for name, source in geometries.items():
        trace, dip_deg, bottom_depth_km, direction = trace_and_geometry(
            source["corners"]
        )
        lat0 = float(np.mean([point["latitude"] for point in trace]))
        simplified, worst_km = simplify(
            trace,
            dip_deg,
            bottom_depth_km,
            111.32 * math.cos(math.radians(lat0)),
        )
        if len(simplified) < len(trace):
            moved.append(
                f"{name}: {len(trace)} -> {len(simplified)} trace points, "
                f"worst deviation {worst_km * 1000:.0f} m"
            )
        trace = simplified
        lines += [
            "[[surfaces]]",
            'type = "fault"',
            f'name = "{name}"',
            (
                f"origin = {{ longitude_deg = {trace[0]['longitude']:.6f}, "
                f"latitude_deg = {trace[0]['latitude']:.6f} }}"
            ),
            "top_depth_km = 0.0",
            "",
        ]
        for point in trace[1:]:
            lines += [
                "[[surfaces.planes]]",
                (
                    f"end = {{ longitude_deg = {point['longitude']:.6f}, "
                    f"latitude_deg = {point['latitude']:.6f} }}"
                ),
                f"dip_deg = {dip_deg:.3f}",
                f"bottom_depth_km = {bottom_depth_km:.3f}",
                f'dip_direction = "{direction}"',
                f"discretisation = {{ subfault_size_km = {resolution_km} }}",
                "",
            ]

    if moved:
        lines = (
            lines[:6]
            + [
                "#",
                "# Traces thinned where the fault's own depth cannot support them:",
                "# at a bend the bottom edge shortens by the horizontal reach of the",
                "# dip, and a plane shorter than that closes to a triangle at depth.",
            ]
            + [f"#   {line}" for line in moved]
            + lines[6:]
        )

    parents = {child: parent for child, parent in tree.items() if parent is not None}
    lines += [
        "# Who triggers whom. The jump *points* are not carried across: this",
        "# generator finds them from the solved wavefront rather than by closest",
        "# approach, so importing them would import an answer to a question it asks",
        "# itself, and asks differently.",
        "[propagation]",
        'type = "predetermined"',
        "",
        # A table section rather than an inline table: TOML's inline tables are
        # single-line by definition, and twenty faults do not fit on one.
        "[propagation.parents]",
    ]
    for child, parent in parents.items():
        lines.append(f'"{child}" = "{parent}"')
    lines.append("")
    return "\n".join(lines)


def rupture_toml(realisation: dict) -> str:
    """The earthquake, as a rupture config."""
    magnitudes = realisation["magnitudes"]["magnitudes"]
    rakes = realisation["rakes"]["rakes"]
    tree = realisation["rupture_propagation"]["rupture_causality_tree"]
    hypocentre = realisation["rupture_propagation"]["hypocentre"]
    srf = realisation["srf"]
    velocity = realisation["rupture_velocity"]
    root = next(name for name, parent in tree.items() if parent is None)

    # The velocity model is layer thicknesses; this package indexes by the depth to
    # each layer's bottom, which is their running sum.
    layers = realisation["velocity_model_1d"]["model"]
    bottoms = np.cumsum([layer["thickness"] for layer in layers])

    total = sum(10.0 ** (1.5 * (m + 6.0333003)) for m in magnitudes.values())
    joint = (math.log10(total) - 9.0499505) / 1.5

    lines = [
        "# The Alpine-Hope joint rupture, converted from a workflow realisation.",
        "#",
        f"# Twenty faults, each with a magnitude of its own; together Mw {joint:.2f}.",
        "# The source is `per_fault` rather than `finite` because the hazard model",
        "# already decided how the moment divides between them -- deriving that",
        "# division again from one event magnitude would discard what it said.",
        "",
        "schema_version = 1",
        f'title = "{realisation["metadata"]["name"]}"',
        "",
        "# The realisation gives the hypocentre as fractions along strike and down",
        "# dip; this config uses in-fault arc lengths, so the fractions are resolved",
        "# against the root fault's own extent when the mesh is known. These are that",
        "# resolution for the meshed geometry.",
        "[hypocentre]",
        f'fault = "{root}"',
        "strike_km = 0.0  # filled in by the converter's --resolve step",
        "dip_km = 0.0",
        "",
        "[velocity_model]",
        "bottom_depth_km  = [" + ", ".join(f"{value:.2f}" for value in bottoms) + "]",
        "shear_speed_km_s = ["
        + ", ".join(f"{layer['Vs']:.3f}" for layer in layers)
        + "]",
        "density_g_cm3    = ["
        + ", ".join(f"{layer['rho']:.3f}" for layer in layers)
        + "]",
        "",
        "[source]",
        'type = "per_fault"',
        f"rise_time_coefficient = {srf['risetime_coef']}",
        "",
        "[source.magnitudes]",
    ]
    for name, magnitude in magnitudes.items():
        lines.append(f'"{name}" = {magnitude:.6f}')
    lines += ["", "[source.rakes]"]
    for name, rake in rakes.items():
        lines.append(f'"{name}" = {float(rake):.3f}')

    lines += [
        "",
        "[slip]",
        f"coefficient_of_variation = {srf['slip_sigma']}",
        f"side_taper = {srf['side_taper']}",
        f"top_taper = {srf['top_taper']}",
        f"bottom_taper = {srf['bot_taper']}",
        "",
        "[field]",
        f"velocity_fraction = {velocity['rvfrac']}",
        "",
        "[timing]",
        "rupture_time_scale = -0.35",
        "rise_time_blend   = { centre_km = 2.0,  half_width_km = 1.0 }",
        (
            f"shallow_ramp      = {{ centre_km = {velocity['shallow_depth']}, "
            f"half_width_km = {velocity['shallow_transition_range']} }}"
        ),
        (
            f"deep_ramp         = {{ centre_km = {velocity['deep_depth']}, "
            f"half_width_km = {velocity['deep_transition_range']} }}"
        ),
        "beta_shallow_ramp = { centre_km = 2.0,  half_width_km = 1.0 }",
        "beta_mid_ramp     = { centre_km = 6.5,  half_width_km = 1.5 }",
        f"shallow_speed_factor = {velocity['rvfrac_shal']}",
        f"deep_speed_factor = {velocity['rvfrac_deep']}",
        "",
        "[random]",
        f"seed = {realisation['seeds']['genslip_seed'] % (2**31)}",
        "realisation = 0",
        "",
        (
            f"# Hypocentre fractions from the realisation: s = {hypocentre['s']}, "
            f"d = {hypocentre['d']}"
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    """Read a realisation and write the two configs beside each other."""
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)

    realisation = json.loads(Path(sys.argv[1]).read_text())
    stem = Path(sys.argv[2])
    resolution_km = realisation["srf"]["resolution"]

    geometry_path = stem.with_suffix(".geometry.toml")
    rupture_path = stem.with_suffix(".toml")
    geometry_path.write_text(geometry_toml(realisation, resolution_km))
    rupture_path.write_text(rupture_toml(realisation))

    print(f"wrote {geometry_path}")
    print(f"wrote {rupture_path}")


if __name__ == "__main__":
    main()
