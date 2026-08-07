"""The Stage 0 fixture corpus: cases, their reference output, and how to rebuild it.

**Nothing here is part of `rupture_generator`.** A case is a geometry, a getpar set
and a seed; generating one runs the real genslip and stores what it produced. The
stored output is what `test_corpus.py` compares the port against, so the comparison
runs on a machine with no genslip binary and no EMOD3D build.

# Why the reference is stored rather than regenerated

A reference computed by the same run that checks it proves only that the run is
self-consistent. Storing it makes the comparison a claim about *this* genslip, built
with these flags, on the day it was recorded -- and makes a change in the port show up
as a diff rather than as two numbers that moved together.

Rebuild with:

```sh
GENSLIP_BINARY=... .venv/bin/python -m tests.harness.corpus
```

which overwrites everything under `tests/corpus/`. Do that only when the reference
genslip changes, and say so in the commit: it invalidates every comparison at once.

# What the spread is for

Each case exists to make some quantity stop being trivially constant, because a
mapping that reads subfault zero where it should average, or hardcodes a number that
should track the grid, passes on a single uniform plane:

| case | what it stops being constant |
| --- | --- |
| `crustal_small` | nothing -- it is the anchor, and the fixture the earlier commits used |
| `crustal_large` | the grid: 48x20 pads differently, and `wavelength_min` tracks the subfault size |
| `subduction` | the dip, the top depth, and `magC` -- Suzuki's down-dip corner saturates |
| `bent` | `strike_deg` and `rake_deg` *within* the grid, so `avgrak` and `alphaT` stop being every subfault's value |
| `frankel_corners` | the corner relation, which `DEFECTS.md` 11 got wrong |

`crustal_small` is deliberately the geometry `test_genslip_reference.py` already
drives, so the corpus and the older tests cannot disagree about the same fault.

# One case is a twin, and that is the point of it

`frankel_no_perturbation` is `frankel_corners` with `tsfac_main = 0` and nothing else
changed, built from it with `dataclasses.replace` so "nothing else" is a fact.

Onset is a travel time plus a slip-correlated perturbation, and two wrong things that
sum to a plausible field are indistinguishable in it. Setting `tsfac_main = 0` removes
the perturbation term exactly -- see `_perturbation_switched_off` -- so the twin's
onset is the eikonal solve alone. That is what closed `DEFECTS.md` 17: the whole onset
divergence was in the travel times, and none of it in the perturbation.

Frankel is the one worth twinning because it is the case whose slip still diverges. Its
perturbation is drawn correlated with slip, so its onset diverges too, and without the
twin there is no way to tell that from a travel-time regression.

# The GSF's order is not the SRF's, and only a multi-segment case shows it

`segno` is not inert when `seg_delay=0`, which is what it first looked like.
`init_plane_srf` (`gslip_srf_subs.c:6`) takes `nseg` to be `max(segno) + 1` and emits
**one `PLANE` block per segment**, writing the points segment by segment. A GSF laid
out along-strike-fastest *across* a join is therefore reordered on the way into the
SRF.

On `bent` the two orders disagree by up to **0.18 degrees of position** -- the same
subfault count, the same plausible-looking rupture, different subfaults. Comparing in
file order would have compared slip at one place against slip at another and reported
the difference as a port defect. `segment_order` is the permutation and
`test_corpus.py` pins it.

Nothing but a multi-segment case can show this: on all four single-plane cases the two
orders are identical, which is why `bent` earns its place twice over.

# What is deliberately absent

genslip's **`seg_delay` path is not ported** -- the rupture-speed reduction in a zone
around each segment boundary (`rvfac_seg`, `gwid`, `genslip_v5.6.2.c:3061-3092`) and
`get_rsegdelay`. `bent` is a bent *geometry* with `seg_delay=0`, which is the part
that can be compared. A case with the delays on would store a reference nothing can
check.
"""

from __future__ import annotations

import dataclasses
import gzip
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from rupture_generator import srf
from rupture_generator.srf import SrfFile
from tests.harness import serialise as utils
from tests.harness.genslip_config import KModel, Parameters, RuptureTimePerturbation
from tests.harness.gsf import (
    FloatArray,
    GsfSubfaults,
    on_a_plane,
    on_two_planes,
    read_gsf,
    write_gsf,
)
from tests.harness.test_unroll import _make_minimal_params

CORPUS = Path(__file__).resolve().parent.parent / "corpus"
"""Where the stored cases live. Committed, so the comparison needs no binary."""

# Four crustal layers, shared by every crustal case so a difference between them is a
# difference in the fault rather than in the medium.
CRUSTAL_LAYERS = (
    np.array([1.0, 5.0, 12.0, 30.0], dtype=np.float32),
    np.array([1.8, 2.6, 3.2, 3.6], dtype=np.float32),
    np.array([2.1, 2.4, 2.6, 2.7], dtype=np.float32),
)

# A deeper, faster column for the subduction case: its top edge is below the crustal
# model's second layer, so reusing that one would put the whole fault in one layer and
# make the depth-dependent rigidity constant.
SUBDUCTION_LAYERS = (
    np.array([3.0, 10.0, 25.0, 45.0, 80.0], dtype=np.float32),
    np.array([2.0, 2.9, 3.5, 3.9, 4.3], dtype=np.float32),
    np.array([2.2, 2.5, 2.8, 3.0, 3.3], dtype=np.float32),
)


@dataclasses.dataclass(frozen=True)
class Case:
    """One fixture: a fault, a parameter set, and a seed.

    The geometry is stored as a builder call rather than as the GSF alone, so the
    case can be rebuilt from scratch and so `test_corpus.py` can ask for the
    subfaults directly without parsing a file it also wrote.

    Attributes
    ----------
    name : str
        Its filename stem under `tests/corpus/`.
    why : str
        What this case makes non-constant that the others do not. Written down
        because a fixture whose purpose is not recorded gets deleted by the next
        person to look at the directory.
    build : Any
        A zero-argument callable returning the `GsfSubfaults`.
    strike_count, dip_count : int
        `nstk` and `ndip`.
    magnitude : float
        `mag`.
    seed : int
        The RNG seed.
    hypocentre_strike_km, hypocentre_dip_km : float
        `shypo` and `dhypo`, in kilometres. See `mapping.hypocentre_indices`.
    layers : tuple
        `(bottom_depth_km, shear_speed_km_s, density_g_cm3)`.
    overrides : dict
        getpar values differing from the shared fixture set.
    twin_of : str | None
        The case this one is a deliberate copy of, differing in one parameter so the
        two can be *differenced*. A twin is built with `dataclasses.replace` from the
        case it names, which is what makes "identical but for one field" a fact rather
        than an intention. It is exempt from the spread checks in `test_corpus.py`,
        which otherwise require every case to be a different fault.
    """

    name: str
    why: str
    build: Any
    strike_count: int
    dip_count: int
    magnitude: float
    seed: int
    hypocentre_strike_km: float
    hypocentre_dip_km: float
    layers: tuple[FloatArray, FloatArray, FloatArray]
    overrides: dict[str, Any] = dataclasses.field(default_factory=dict)
    twin_of: str | None = None

    def parameters(self) -> Parameters:
        """
        Returns
        -------
        Parameters
            The getpar set, the shared fixture defaults with this case's overrides.
        """
        return _make_minimal_params(
            read_gsf=True,
            read_erf=False,
            # Roughness off: with it on, the two spectral fields `PRUNED.md` describes
            # stop being numerically inert, and no case here is about roughness.
            alpha_rough=0.0,
            **self.overrides,
        )

    def geometry(self) -> GsfSubfaults:
        """
        Returns
        -------
        GsfSubfaults
            The discretised fault.
        """
        return self.build()


def _perturbation_switched_off() -> RuptureTimePerturbation:
    """The shared fixture's rupture-time perturbation, with `tsfac_main` set to zero.

    Taken from the fixture rather than written out again, so a twin built with it
    differs from its original in `tsfac_main` and in nothing else.

    Zero is *honoured*, not read as "unset": the sentinel is `-1.0e+15` and the guard
    is `tsfac_main > -1.0e+10` (`genslip_v5.6.2.c:3134`), so zero passes it and
    multiplies the perturbation away. Nothing else reads `tsfac_main`, and the
    perturbation field is drawn either way, so the draw stream is untouched -- which
    is what makes the difference between a twin and its original the perturbation
    alone.
    """
    return dataclasses.replace(
        _make_minimal_params().rupture_time_perturbation, main_value=0.0
    )


_FRANKEL = Case(
    name="frankel_corners",
    why="kmodel=Frankel, which DEFECTS.md 11 routed to the wrong corner relation",
    build=lambda: on_a_plane(
        strike_count=24,
        dip_count=16,
        along_strike_km=1.0,
        down_dip_km=1.0,
        centre_longitude_deg=171.5,
        centre_latitude_deg=-44.2,
        strike_deg=100.0,
        dip_deg=45.0,
        top_depth_km=2.0,
        rake_deg=90.0,
    ),
    strike_count=24,
    dip_count=16,
    magnitude=6.6,
    seed=20260811,
    hypocentre_strike_km=-4.0,
    hypocentre_dip_km=8.0,
    layers=CRUSTAL_LAYERS,
    overrides=dict(kmodel=KModel.FRANKEL),
)


CASES: tuple[Case, ...] = (
    Case(
        name="crustal_small",
        why="the anchor: the same 20x12 fault the reference tests already drive",
        build=lambda: on_a_plane(
            strike_count=20,
            dip_count=12,
            along_strike_km=0.5,
            down_dip_km=0.5,
            centre_longitude_deg=172.0,
            centre_latitude_deg=-43.5,
            strike_deg=45.0,
            dip_deg=80.0,
            top_depth_km=1.0,
            rake_deg=175.0,
        ),
        strike_count=20,
        dip_count=12,
        magnitude=6.2,
        seed=20260807,
        hypocentre_strike_km=0.0,
        hypocentre_dip_km=3.0,
        layers=CRUSTAL_LAYERS,
    ),
    Case(
        name="crustal_large",
        why="a larger grid and a different subfault size: nstk2/ndip2 and the "
        "Nyquist-derived wavelength_min both move",
        build=lambda: on_a_plane(
            strike_count=48,
            dip_count=20,
            along_strike_km=1.0,
            down_dip_km=1.0,
            centre_longitude_deg=173.2,
            centre_latitude_deg=-42.1,
            strike_deg=215.0,
            dip_deg=65.0,
            top_depth_km=0.0,
            rake_deg=155.0,
        ),
        strike_count=48,
        dip_count=20,
        magnitude=7.1,
        seed=20260808,
        # Off-centre along strike and deep enough down dip to push the deep ramps
        # below their configured 17.5 km -- the adjustment `mapping` reproduces.
        hypocentre_strike_km=-8.0,
        hypocentre_dip_km=15.0,
        layers=CRUSTAL_LAYERS,
    ),
    Case(
        name="subduction",
        why="shallow dip, deep top edge, a magnitude past magC so Suzuki's down-dip "
        "corner saturates, and a coarser dt; alphaT is well below 1 here and near "
        "1 everywhere else",
        build=lambda: on_a_plane(
            strike_count=48,
            dip_count=24,
            along_strike_km=4.0,
            down_dip_km=4.0,
            centre_longitude_deg=178.0,
            centre_latitude_deg=-39.0,
            strike_deg=20.0,
            dip_deg=18.0,
            top_depth_km=12.0,
            rake_deg=95.0,
        ),
        strike_count=48,
        dip_count=24,
        # 192 x 96 km, which genslip's own median relation puts at M8.27 -- so the
        # magnitude and the area agree. The earlier draft asked for M8.1 on a
        # 72 x 48 km fault, four magnitude units of slip crammed onto a small plane,
        # and every rise time came out absurd. A fixture has to be a rupture that
        # could happen, or the numbers it pins are of nothing.
        magnitude=8.2,
        seed=20260809,
        hypocentre_strike_km=40.0,
        hypocentre_dip_km=60.0,
        layers=SUBDUCTION_LAYERS,
        # magC below the magnitude, so the *saturated* branch of Suzuki's down-dip
        # corner is the one taken. At the documented 8.4 it would not be, and the
        # case would silently test the unclamped path the crustal cases already do.
        # dt is coarser here both because a subduction SRF is written that way and
        # because it makes `sample_interval_s` stop being one number across the
        # corpus.
        overrides=dict(kmodel=KModel.SUZUKI, magnitude_clamp=8.0, dt=0.05),
    ),
    Case(
        name="bent",
        why="strike and rake vary within the grid, so avgrak and alphaT stop being "
        "the value every subfault has",
        build=lambda: on_two_planes(
            strike_counts=(16, 16),
            dip_count=14,
            along_strike_km=1.0,
            down_dip_km=1.0,
            centre_longitude_deg=173.0,
            centre_latitude_deg=-42.0,
            strike_degs=(30.0, 65.0),
            dip_deg=55.0,
            top_depth_km=0.5,
            rake_degs=(100.0, 140.0),
        ),
        strike_count=32,
        dip_count=14,
        magnitude=6.9,
        seed=20260810,
        hypocentre_strike_km=4.0,
        hypocentre_dip_km=7.0,
        layers=CRUSTAL_LAYERS,
    ),
    _FRANKEL,
    dataclasses.replace(
        _FRANKEL,
        name="frankel_no_perturbation",
        why="frankel_corners with tsfac_main=0, so onset is the eikonal solve and "
        "nothing else -- the only way to check travel times on the one case whose "
        "slip, and therefore whose slip-correlated timing perturbation, diverges",
        overrides=_FRANKEL.overrides
        | dict(rupture_time_perturbation=_perturbation_switched_off()),
        twin_of="frankel_corners",
    ),
)

BY_NAME = {case.name: case for case in CASES}


def gsf_path(name: str) -> Path:
    """
    Parameters
    ----------
    name : str
        The case's name.

    Returns
    -------
    Path
        Where its geometry file is stored.
    """
    return CORPUS / f"{name}.gsf"


def args_path(name: str) -> Path:
    """
    Parameters
    ----------
    name : str
        The case's name.

    Returns
    -------
    Path
        Where its rendered argument list is stored.
    """
    return CORPUS / f"{name}.args"


def srf_path(name: str) -> Path:
    """
    Parameters
    ----------
    name : str
        The case's name.

    Returns
    -------
    Path
        Where its gzipped reference SRF is stored.
    """
    return CORPUS / f"{name}.srf.gz"


def segment_order(geometry: GsfSubfaults) -> np.ndarray:
    """The permutation taking GSF order to the order the SRF writes points in.

    genslip emits one `PLANE` block per distinct `segno` and writes each segment's
    points together (`gslip_srf_subs.c:6`), so a GSF whose rows run across a segment
    join comes out reordered. Within a segment the order is unchanged.

    For a single-segment fault this is the identity, which is why four of the five
    cases would never have caught its absence.

    Parameters
    ----------
    geometry : GsfSubfaults
        The subfaults as the GSF holds them.

    Returns
    -------
    np.ndarray
        Indices into GSF order, such that `array[segment_order(geometry)]` is in the
        SRF's order.
    """
    return np.concatenate(
        [
            np.flatnonzero(geometry.segment == segment)
            for segment in np.unique(geometry.segment)
        ]
    )


def load_reference(name: str) -> SrfFile:
    """Read a case's stored reference SRF.

    Decompressed into memory rather than mapped: these are fixtures of a few
    megabytes, and the streaming path `genslip_reference` uses exists for the
    multi-gigabyte files the parser is actually measured on.

    Parameters
    ----------
    name : str
        The case's name.

    Returns
    -------
    SrfFile
        What genslip produced when the corpus was built.

    Raises
    ------
    FileNotFoundError
        If the case has not been generated. `tests/corpus/README.md` says how.
    """
    path = srf_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; rebuild the corpus with "
            "`GENSLIP_BINARY=... python -m tests.harness.corpus`"
        )
    with gzip.open(path, "rb") as compressed:
        return srf.SrfFile.from_file(compressed.read())


def load_geometry(name: str) -> GsfSubfaults:
    """Read a case's stored geometry file.

    Parameters
    ----------
    name : str
        The case's name.

    Returns
    -------
    GsfSubfaults
        The subfaults genslip was given.
    """
    return read_gsf(gsf_path(name))


def load_arguments(name: str) -> dict[str, str]:
    """Read a case's stored argument list back into a mapping.

    Parameters
    ----------
    name : str
        The case's name.

    Returns
    -------
    dict[str, str]
        Every `name=value` genslip was invoked with, values unparsed.
    """
    return dict(
        line.split("=", 1)
        for line in args_path(name).read_text().splitlines()
        if line.strip()
    )


def run_port(case: Case) -> Any:
    """Generate the same rupture with the port, from the same `Parameters`.

    Every argument comes through `mapping`, which is the point: nothing here picks a
    value, so a disagreement with the stored reference is a disagreement in the
    physics or in the mapping, and never in what the two sides were asked for.

    The result is in **GSF order**, along-strike-fastest over the whole grid. The
    reference is in SRF order. `segment_order` is the permutation, and it is the
    identity for every single-segment case.

    Parameters
    ----------
    case : Case
        The case to generate.

    Returns
    -------
    GeneratedRupture
        The port's rupture.
    """
    from rupture_generator import generate_rupture
    from tests.harness import mapping

    geometry = case.geometry()
    parameters = case.parameters()
    bottom_depth_km, shear_speed_km_s, density_g_cm3 = case.layers

    derived = mapping.derive(
        geometry,
        parameters,
        magnitude=case.magnitude,
        strike_count=case.strike_count,
        dip_count=case.dip_count,
    )
    strike, dip = mapping.hypocentre_indices(
        case.hypocentre_strike_km, case.hypocentre_dip_km, geometry, derived
    )

    return generate_rupture(
        mapping.fault_grid(
            geometry,
            parameters,
            derived,
            strike_count=case.strike_count,
            dip_count=case.dip_count,
        ),
        mapping.velocity_model(bottom_depth_km, shear_speed_km_s, density_g_cm3),
        mapping.source_spec(geometry, parameters, magnitude=case.magnitude),
        mapping.slip_spec(geometry, parameters),
        mapping.timing_spec(
            geometry,
            parameters,
            derived,
            hypocentre_dip_km=case.hypocentre_dip_km,
        ),
        seed=case.seed,
        hypocentre_strike=strike,
        hypocentre_dip=dip,
    )


def generate(case: Case, genslip_path: Path) -> None:
    """Run genslip for one case and store everything it was given and produced.

    The `infile` and `velfile` arguments are rewritten to the names the corpus uses
    before the argument list is stored, so the stored list describes the case rather
    than the temporary directory it happened to run in.

    Parameters
    ----------
    case : Case
        The case to build.
    genslip_path : Path
        The genslip binary.
    """
    from tests.harness.genslip_reference import generate_segment_rupture

    CORPUS.mkdir(parents=True, exist_ok=True)
    geometry = case.geometry()
    bottom_depth_km, shear_speed_km_s, density_g_cm3 = case.layers

    reference = generate_segment_rupture(
        geometry,
        case.parameters(),
        genslip_path,
        magnitude=case.magnitude,
        strike_count=case.strike_count,
        dip_count=case.dip_count,
        seed=case.seed,
        hypocentre_strike_km=case.hypocentre_strike_km,
        hypocentre_dip_km=case.hypocentre_dip_km,
        bottom_depth_km=bottom_depth_km,
        shear_speed_km_s=shear_speed_km_s,
        density_g_cm3=density_g_cm3,
        keep_raw=True,
    )

    write_gsf(geometry, gsf_path(case.name))

    options = case.parameters().to_cmd() | dict(
        read_gsf=True,
        read_erf=False,
        read_slip_file=False,
        read_vsden=False,
        ns=1,
        nh=1,
        infile=gsf_path(case.name).name,
        velfile=f"{case.name}.1d",
        mag=case.magnitude,
        nstk=case.strike_count,
        ndip=case.dip_count,
        seed=case.seed,
        shypo=case.hypocentre_strike_km,
        dhypo=case.hypocentre_dip_km,
    )
    args_path(case.name).write_text(
        "\n".join(sorted(utils.serialise_options(options))) + "\n"
    )

    # genslip's own bytes, not a re-serialisation of the parsed model: see
    # `ReferenceRun.raw`. mtime=0 so rebuilding an unchanged case produces an
    # identical file rather than a diff that is only a timestamp.
    assert reference.raw is not None, "generate asked for keep_raw"
    with gzip.GzipFile(srf_path(case.name), "wb", mtime=0) as compressed:
        compressed.write(reference.raw)


def main() -> int:
    """Rebuild every case. Entry point for `python -m tests.harness.corpus`.

    Returns
    -------
    int
        A process exit status.
    """
    binary = os.environ.get("GENSLIP_BINARY")
    if not binary:
        print(
            "set GENSLIP_BINARY to a genslip v5.6.2 built with -std=gnu17",
            file=sys.stderr,
        )
        return 1

    # Named cases only, when named. Adding a case should not rewrite the others'
    # fixtures -- and rebuilding all of them anyway is the check that the binary in
    # hand is the one the stored references came from, since an unchanged case
    # rebuilds to identical bytes.
    wanted = sys.argv[1:]
    unknown = [name for name in wanted if name not in BY_NAME]
    if unknown:
        print(f"no such case: {', '.join(unknown)}", file=sys.stderr)
        return 1

    for case in CASES:
        if wanted and case.name not in wanted:
            continue
        generate(case, Path(binary))
        size = srf_path(case.name).stat().st_size
        print(f"{case.name:24s} {size / 1024:8.1f} KiB  {case.why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
