"""Drive the genslip binary for one fault segment.

This is the *reference* path, not the destination. It exists so the port has
something to compare against and so the Stage 0 fixture corpus can be generated;
it is the seam ``rupture_generator`` replaces.
"""

import mmap
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from rupture_generator import srf
from tests.harness import serialise as utils
from tests.harness.genslip_config import Parameters
from rupture_generator.geometry import DiscretisedGeometry
from rupture_generator.srf import SrfFile

# genslip reads the fault grid from a GSF file and writes SRF to stdout. These four
# are not user choices -- they describe how this function drives the binary -- so
# they are set here rather than exposed on `Parameters`.
_DEFAULT_GEOMETRY_PARAMETERS = dict(
    read_gsf=True,
    read_erf=False,
    read_slip_file=False,
    read_vsden=False,
)


def _build_geometry_parameters(geometry_path: Path) -> dict[str, Any]:
    return _DEFAULT_GEOMETRY_PARAMETERS | dict(infile=geometry_path)


def write_gsf(geometry: DiscretisedGeometry, output: Path) -> None:
    raise NotImplementedError


def generate_segment_rupture(
    geometry: DiscretisedGeometry,
    parameters: Parameters,
    genslip_path: Path,
) -> SrfFile:
    """Generate the rupture for one fault segment by invoking genslip.

    Parameters
    ----------
    geometry : DiscretisedGeometry
        The discretised fault segment, written out as a GSF file for genslip.
    parameters : Parameters
        Rupture generation parameters, rendered as ``name=value`` arguments.
    genslip_path : Path
        Path to the genslip binary. Not a member of ``parameters`` because it is
        not a genslip parameter -- putting it there would serialise it onto the
        command line as ``genslip_path=...``.

    Returns
    -------
    SrfFile
        The generated rupture.

    Raises
    ------
    subprocess.CalledProcessError
        If genslip exits non-zero. Its stderr is attached, which matters because
        genslip reports parameter and geometry errors there rather than by exit
        code alone.
    """
    with tempfile.TemporaryDirectory() as tmp_directory:
        tmp = Path(tmp_directory)
        gsf_path = tmp / "geometry.gsf"
        write_gsf(geometry, gsf_path)

        options = parameters.to_cmd() | _build_geometry_parameters(gsf_path)
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

        with (
            srf_path.open("rb") as srf_file,
            mmap.mmap(srf_file.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
        ):
            if hasattr(mapped, "madvise"):
                mapped.madvise(mmap.MADV_SEQUENTIAL)
            return srf.SrfFile.from_file(mapped)
