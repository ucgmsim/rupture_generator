"""Convert a workflow ``realisation.json`` into GeoJSON sections and a rupture config.

The workflow's realisation format is one JSON document holding everything a simulation
needs -- geometry, source, velocity model, seeds, and the parameters of four other
programs. This pulls out the parts that describe *the earthquake* and writes them as a
GeoJSON fault system and a rupture config.

Run it as::

    python examples/from_realisation.py path/to/realisation.json examples/hope

which writes ``examples/hope.geojson`` and ``examples/hope.toml``.

# What is carried across, and what is not

Carried: the fault traces, dips and depths, the causality tree, the per-fault
magnitudes and rakes, the velocity model, the hypocentre, the tapers, the rupture-speed
profile and the resolution.

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
import pyproj

from rupture_generator import config
from rupture_generator.config import (
    GeometryConfig,
    HypocentreConfig,
    MediumConfig,
    PerFault,
    Predetermined,
    Profile,
    ProfilesConfig,
    PulseConfig,
    Ramp,
    Rise,
    RuptureConfig,
    Slip,
    Timing,
)
from rupture_generator.rupture.source import moment_from_magnitude

GEOD = pyproj.Geod(ellps="WGS84")


def section(name: str, corners: list[dict]) -> dict:
    """One fault as a GeoJSON section: its trace, dip, depths and dip direction.

    The dip is recovered from the geometry rather than read from a field, because the
    realisation does not carry one: it carries the corner positions the dip produced.
    Taking the mean over the fault's planes is exact where they agree and is the only
    thing available where they do not.
    """
    planes = len(corners) // 4
    trace = [corners[4 * plane] for plane in range(planes)]
    trace.append(corners[4 * (planes - 1) + 1])

    lat0 = float(np.mean([point["latitude"] for point in corners]))
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
        dips.append(
            math.degrees(math.atan2(down, horizontal)) if horizontal > 1.0e-9 else 90.0
        )
        # Which side of the trace the fault hangs on: the sign of the cross product
        # of the along-strike step with the down-dip step, in the horizontal plane.
        along_east = (top_b["longitude"] - top_a["longitude"]) * east_per_degree
        along_north = (top_b["latitude"] - top_a["latitude"]) * 110.57
        sides.append(np.sign(along_east * north - along_north * east))

    # A negative cross product in an east-north frame is a fault dipping to the right
    # of the walk along the trace, a quarter turn clockwise from its strike.
    strike_deg, _, _ = GEOD.inv(
        trace[0]["longitude"], trace[0]["latitude"],
        trace[-1]["longitude"], trace[-1]["latitude"],
    )  # fmt: skip
    turn = 90.0 if float(np.mean(sides)) < 0 else -90.0
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [round(point["longitude"], 6), round(point["latitude"], 6)]
                for point in trace
            ],
        },
        "properties": {
            "name": name,
            "dip_deg": round(float(np.clip(np.mean(dips), 1.0, 90.0)), 3),
            "dip_direction_deg": round((strike_deg + turn) % 360.0, 3),
            "upper_depth_km": 0.0,
            "lower_depth_km": round(
                max(point["depth"] for point in corners) / 1000.0, 3
            ),
        },
    }


def fault_system(realisation: dict) -> str:
    """The faults, as a GeoJSON ``FeatureCollection`` with one feature per line."""
    features = [
        json.dumps(section(name, source["corners"]))
        for name, source in realisation["sources"]["source_geometries"].items()
    ]
    return (
        '{\n  "type": "FeatureCollection",\n  "features": [\n    '
        + ",\n    ".join(features)
        + "\n  ]\n}\n"
    )


def rupture(realisation: dict, geometry: Path) -> RuptureConfig:
    """The earthquake, as a rupture config.

    The depth profiles are the workflow's: the front slowed by ``rvfrac_shal`` and
    ``rvfrac_deep`` over its shallow and deep transitions, rise time doubled over the
    same ones, shallow rise time tied to slip above 3 km, and the pulse's rising
    fraction 0.5 above 1 km and 0.13 below 3 km.
    """
    tree = realisation["rupture_propagation"]["rupture_causality_tree"]
    hypocentre = realisation["rupture_propagation"]["hypocentre"]
    srf = realisation["srf"]
    velocity = realisation["rupture_velocity"]
    root = next(name for name, parent in tree.items() if parent is None)

    # The velocity model is layer thicknesses; this package indexes by the depth to
    # each layer's bottom, which is their running sum.
    layers = realisation["velocity_model_1d"]["model"]
    shallow, deep = velocity["shallow_depth"], velocity["deep_depth"]
    shallow_range = velocity["shallow_transition_range"]
    deep_range = velocity["deep_transition_range"]
    transitions = [
        shallow - shallow_range,
        shallow + shallow_range,
        deep - deep_range,
        deep + deep_range,
    ]
    return RuptureConfig(
        seed=realisation["seeds"]["genslip_seed"] % (2**31),
        geometry=GeometryConfig(
            path=geometry, crs="EPSG:2193", spacing_km=srf["resolution"]
        ),
        hypocentre=HypocentreConfig(
            segment=root, strike_fraction=hypocentre["s"], dip_fraction=hypocentre["d"]
        ),
        medium=MediumConfig(
            bottom_depth_km=np.cumsum(
                [layer["thickness"] for layer in layers]
            ).tolist(),
            shear_speed_km_s=[layer["Vs"] for layer in layers],
            density_g_cm3=[layer["rho"] for layer in layers],
        ),
        source=PerFault(
            magnitudes=realisation["magnitudes"]["magnitudes"],
            rakes={
                name: float(rake)
                for name, rake in realisation["rakes"]["rakes"].items()
            },
        ),
        propagation=Predetermined(
            parents={child: parent for child, parent in tree.items() if parent}
        ),
        profiles=ProfilesConfig(
            rise_time_factor=Profile(depth_km=transitions, values=[2.0, 1.0, 1.0, 2.0]),
            rise_time_slip_weight=Ramp(
                centre_km=2.0, half_width_km=1.0, shallow=0.0, deep=1.0
            ),
            rupture_speed_factor=Profile(
                depth_km=transitions,
                values=[velocity["rvfrac_shal"], 1.0, 1.0, velocity["rvfrac_deep"]],
            ),
        ),
        slip=Slip(
            coefficient_of_variation=srf["slip_sigma"],
            side_taper=srf["side_taper"],
            top_taper=srf["top_taper"],
            bottom_taper=srf["bot_taper"],
        ),
        rise=Rise(coefficient=srf["risetime_coef"]),
        timing=Timing(velocity_fraction=velocity["rvfrac"]),
        pulse=PulseConfig(
            beta=Ramp(centre_km=2.0, half_width_km=1.0, shallow=0.5, deep=0.13)
        ),
    )


def main() -> None:
    """Read a realisation and write the fault system and config beside each other."""
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)

    realisation = json.loads(Path(sys.argv[1]).read_text())
    stem = Path(sys.argv[2])
    geometry_path = stem.with_suffix(".geojson")
    rupture_path = stem.with_suffix(".toml")

    magnitudes = realisation["magnitudes"]["magnitudes"].values()
    total_nm = sum(moment_from_magnitude(magnitude) for magnitude in magnitudes)
    joint = (math.log10(total_nm * 1.0e7) - 16.05) / 1.5
    header = (
        f"# {realisation['metadata']['name']}, converted from a workflow realisation:\n"
        f"# {len(magnitudes)} faults, each with a magnitude of its own; together "
        f"Mw {joint:.2f}.\n\n"
    )

    geometry_path.write_text(fault_system(realisation))
    rupture_path.write_text(
        header + config.dump(rupture(realisation, Path(geometry_path.name)))
    )
    config.load(rupture_path)
    print(f"wrote {geometry_path}")
    print(f"wrote {rupture_path}")


if __name__ == "__main__":
    main()
