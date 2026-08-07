"""Drive the genslip binary for one fault segment.

This is the *reference* path, not the destination. It exists so the port has
something to compare against and so the Stage 0 fixture corpus can be generated;
it is the seam ``rupture_generator`` replaces.
"""

import dataclasses
import math
import mmap
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from rupture_generator import srf
from rupture_generator.srf import SrfFile
from tests.harness import serialise as utils
from tests.harness.genslip_config import Parameters
from tests.harness.gsf import FloatArray, GsfSubfaults, write_gsf

# genslip reports what it derived on stderr before it generates anything: the padded
# extents, the subfault spacing, alphaT, and the rise-time and rupture-velocity
# constants alphaT was applied to. Those are exactly the quantities `mapping.py` has
# to reconstruct from the getpar names, so the binary is their oracle and a
# transcription of the C is not needed to check them.
#
# `%13.5e` and `%.4f` mean the values arrive rounded. Every assertion against them
# allows for that; see `test_mapping.py`.
_DIAGNOSTIC = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_()]*)=\s*(?P<value>-?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|\d*\.\d+))"
)


def parse_diagnostics(stderr: bytes) -> dict[str, float]:
    """Read genslip's `name= value` progress reports off stderr.

    genslip has no machine-readable output beyond the SRF, but it prints what it
    derived -- `nstk2`, `ndip2`, `dstk`, `ddip`, `alphaT`, `trise_avg`, `rvfrac_avg`,
    `mom` -- in a uniform `name= value` shape. Parsing it turns the reference binary
    into an oracle for the *inputs* of the port as well as its outputs, which is what
    lets a wrong mapping be told apart from a wrong port.

    **The first occurrence of a name wins.** genslip prints
    `mag= 6.20 median mag= 5.78` on one line (line 1630), and a name here cannot
    contain a space, so the second pair parses as `mag` too -- letting the last win
    would silently report the median magnitude as the magnitude, which is a plausible
    number and the wrong one. Nothing genslip revises is printed twice: it resolves
    `mag` against `use_median_mag` at line 1268, well before it reports it.

    Names that recur as noise (`re`, `im`, `avg`, `min`, `max`, `sig`, from the field
    summaries) are meaningless under either rule and are not read by anything.

    Parameters
    ----------
    stderr : bytes
        Everything genslip wrote to stderr.

    Returns
    -------
    dict[str, float]
        Every `name= value` pair it printed, first occurrence winning.
    """
    text = stderr.decode(errors="replace")
    diagnostics: dict[str, float] = {}
    for match in _DIAGNOSTIC.finditer(text):
        diagnostics.setdefault(match.group("name"), float(match.group("value")))
    return diagnostics


@dataclasses.dataclass(frozen=True)
class ReferenceRun:
    """One invocation of genslip: what it produced and what it said it derived.

    Attributes
    ----------
    srf : SrfFile
        The rupture it wrote.
    diagnostics : dict[str, float]
        The `name= value` pairs it reported on stderr. See `parse_diagnostics`.
    raw : bytes | None
        Exactly the bytes genslip wrote, when `keep_raw` asked for them.

        The fixture corpus stores *these* rather than a re-serialisation of `srf`,
        because a round trip through this package's own parser and writer would
        launder any defect in either into the reference -- and the reference is
        supposed to be genslip's answer, not ours. `None` by default: an SRF can
        reach gigabytes and no other caller wants it resident.
    """

    srf: SrfFile
    diagnostics: dict[str, float]
    raw: bytes | None = None

# genslip reads the fault grid from a GSF file and writes SRF to stdout. These are not
# user choices -- they describe how this function drives the binary -- so they are set
# here rather than exposed on `Parameters`.
#
# ns AND nh ARE NOT OPTIONAL, whatever their defaults suggest. They are the counts of
# slip and hypocentre realisations to emit, they default to -1, and the loops that
# write the SRF are `for(js=0;js<ns;js++)` and `for(ih=0;ih<nh;ih++)`
# (genslip_v5.6.2.c:2966). Every path that computes them sits inside
# `if(read_erf == 1)`, so on the GSF path they stay -1, both loops run zero times, and
# genslip EXITS 0 HAVING WRITTEN NOTHING -- no diagnostic, a zero-byte SRF. One
# realisation per invocation is what the corpus wants regardless.
_DEFAULT_GEOMETRY_PARAMETERS = dict(
    read_gsf=True,
    read_erf=False,
    read_slip_file=False,
    read_vsden=False,
    ns=1,
    nh=1,
)


def _build_geometry_parameters(geometry_path: Path) -> dict[str, Any]:
    return _DEFAULT_GEOMETRY_PARAMETERS | dict(infile=geometry_path)


def write_velocity_model(
    bottom_depth_km: FloatArray,
    shear_speed_km_s: FloatArray,
    density_g_cm3: FloatArray,
    output: Path,
) -> None:
    """Write a velocity model as the 1D file genslip reads.

    One line per layer: thickness, P speed, S speed, density. The arguments are layer
    *bottoms*, matching `VelocityModel1D`'s constructor, so the thicknesses are
    differences.

    Takes the three arrays rather than a `VelocityModel1D` because that type is
    write-only from Python -- PyO3 exposes its constructor and no getters -- so a
    caller cannot hand this the model it built. Passing the arrays to both keeps one
    definition with two consumers, which is the point.

    The P speed is a placeholder. `read_Fvelmodel` parses it into the layer struct
    (`ruptime.c:286`) and nothing in genslip reads it back: rupture speed comes from
    the S speed and rigidity from `vs * vs * den`. Writing a Poisson solid's
    `sqrt(3) * vs` keeps the file plausible to anything else that reads it.

    Parameters
    ----------
    bottom_depth_km : FloatArray
        Depth of the bottom of each layer, shallow to deep.
    shear_speed_km_s : FloatArray
        S-wave speed in each layer.
    density_g_cm3 : FloatArray
        Density in each layer.
    output : Path
        Where to write them.
    """
    bottoms = np.asarray(bottom_depth_km, dtype=np.float64)
    shear = np.asarray(shear_speed_km_s, dtype=np.float64)
    density = np.asarray(density_g_cm3, dtype=np.float64)
    thickness = np.diff(bottoms, prepend=0.0)

    lines = [f"{len(bottoms)}"]
    lines.extend(
        f"{thick:10.4f} {math.sqrt(3.0) * vs:10.4f} {vs:10.4f} {den:10.4f}"
        for thick, vs, den in zip(thickness, shear, density, strict=True)
    )
    output.write_text("\n".join(lines) + "\n")


def generate_segment_rupture(
    geometry: GsfSubfaults,
    parameters: Parameters,
    genslip_path: Path,
    *,
    magnitude: float,
    strike_count: int,
    dip_count: int,
    seed: int,
    hypocentre_strike_km: float,
    hypocentre_dip_km: float,
    bottom_depth_km: FloatArray,
    shear_speed_km_s: FloatArray,
    density_g_cm3: FloatArray,
    keep_raw: bool = False,
) -> ReferenceRun:
    """Generate the rupture for one fault segment by invoking genslip.

    The keyword arguments are the ones genslip takes with `mstpar` on the GSF path
    (`genslip_v5.6.2.c:829`) or that have no representation in `Parameters`, so they
    cannot be omitted and are not defaulted here. `nstk` and `ndip` are among them
    because a GSF is a flat list of subfaults and does not say what shape the grid is.

    Parameters
    ----------
    geometry : GsfSubfaults
        The discretised fault segment, written out as a GSF file for genslip.
    parameters : Parameters
        Rupture generation parameters, rendered as ``name=value`` arguments.
    genslip_path : Path
        Path to the genslip binary. Not a member of ``parameters`` because it is
        not a genslip parameter -- putting it there would serialise it onto the
        command line as ``genslip_path=...``.
    magnitude : float
        The target magnitude, `mag`.
    strike_count, dip_count : int
        The grid's shape, `nstk` and `ndip`.
    seed : int
        The RNG seed.
    hypocentre_strike_km : float
        Hypocentre position along strike from the segment's centre, `shypo`.
    hypocentre_dip_km : float
        Hypocentre position down dip from the top edge, `dhypo`.
    bottom_depth_km, shear_speed_km_s, density_g_cm3 : FloatArray
        The 1D velocity model, written out as genslip's `velfile`. The same three
        arrays a caller passes to `VelocityModel1D`, so both sides are provably given
        the same layers -- see `write_velocity_model`.
    keep_raw : bool, optional
        Return the bytes genslip wrote alongside the parsed model. The fixture
        corpus wants them; nothing else does, and an SRF can reach gigabytes.

    Returns
    -------
    ReferenceRun
        The generated rupture, and the quantities genslip reported deriving.

    Raises
    ------
    subprocess.CalledProcessError
        If genslip exits non-zero. Its stderr is attached, which matters because
        genslip reports parameter and geometry errors there rather than by exit
        code alone.
    ValueError
        If genslip exits zero having written nothing, which is what it does when it
        is asked for zero realisations. `mmap` cannot map an empty file, so without
        this the failure arrives as an unexplained `ValueError` from the standard
        library.
    """
    with tempfile.TemporaryDirectory() as tmp_directory:
        tmp = Path(tmp_directory)
        gsf_path = tmp / "geometry.gsf"
        write_gsf(geometry, gsf_path)
        velocity_path = tmp / "velocity_model.1d"
        write_velocity_model(
            bottom_depth_km, shear_speed_km_s, density_g_cm3, velocity_path
        )

        options = (
            parameters.to_cmd()
            | _build_geometry_parameters(gsf_path)
            | dict(
                mag=magnitude,
                nstk=strike_count,
                ndip=dip_count,
                seed=seed,
                shypo=hypocentre_strike_km,
                dhypo=hypocentre_dip_km,
                velfile=velocity_path,
            )
        )
        cmd = [str(genslip_path), *utils.serialise_options(options)]

        # SRF goes to stdout and can reach gigabytes on a large fault, so it is
        # streamed to disk rather than buffered in the parent. stderr is small and
        # carries the diagnostics, so it is captured.
        srf_path = tmp / "rupture.srf"
        with srf_path.open("wb") as srf_file:
            completed = subprocess.run(
                cmd,
                stdout=srf_file,
                stderr=subprocess.PIPE,
                shell=False,
                check=False,
            )

        if completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode, cmd, stderr=completed.stderr
            )

        if srf_path.stat().st_size == 0:
            raise ValueError(
                "genslip exited 0 and wrote no SRF, which means it was asked for zero "
                "realisations -- check ns and nh. Its stderr was:\n"
                + completed.stderr.decode(errors="replace")
            )

        with (
            srf_path.open("rb") as srf_file,
            mmap.mmap(srf_file.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
        ):
            if hasattr(mapped, "madvise"):
                mapped.madvise(mmap.MADV_SEQUENTIAL)
            return ReferenceRun(
                srf=srf.SrfFile.from_file(mapped),
                diagnostics=parse_diagnostics(completed.stderr),
                raw=srf_path.read_bytes() if keep_raw else None,
            )
