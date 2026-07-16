import mmap
import shutil
import subprocess
import tempfile
from collections.abc import Buffer
from pathlib import Path
from typing import Any

from source_modelling import srf
from source_modelling.srf import SrfFile

from rupture_generator import utils

MAX_IN_MEMORY_BUFFER = 1 << 30


def file_backed(output: tempfile.SpooledTemporaryFile) -> bool:
    return output._rolled


def output_buffer(output: tempfile.SpooledTemporaryFile) -> Buffer:
    return output._file.getbuffer()


_DEFAULT_GEOMETRY_PARAMETERS = dict(
    read_gsf=True,
    read_erf=False,
    read_slip_file=False,
    read_vsden=False,
)


def _build_geometry_parameters(geometry_path: Path) -> dict[str, Any]:
    return _DEFAULT_GEOMETRY_PARAMETERS | dict(infile=geometry_path)


def generate_segment_slip(geometry: Geometry, parameters: Parameters) -> SrfFile:
    with (
        tempfile.NamedTemporaryFile() as tmp_gsf_output,
        tempfile.SpooledTemporaryFile(MAX_IN_MEMORY_BUFFER, mode="w+b") as output,
    ):
        tmp_gsf_path = Path(tmp_gsf_output.name)
        geometry.write_gsf(tmp_gsf_path)

        options = parameters.to_cmd()
        options.update(_build_geometry_parameters(tmp_gsf_path))

        cmd = [str(parameters.genslip_path)]
        cmd.extend(utils.serialise_options(options))

        with subprocess.Popen(
            cmd, check=True, shell=False, stdout=subprocess.PIPE
        ) as proc:
            if not proc.stdout:
                raise RuntimeError("Could not acquire stdout pipe.")
            shutil.copyfileobj(proc.stdout, output)

        output.flush()

        if file_backed(output):
            # We only have file-backed output for truly huge SRFs (> 1GB). Here
            # it makes sense to setup a memory map to read these.
            with mmap.mmap(output.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                if hasattr(mm, "madvise"):
                    mm.madvise(mmap.MADV_SEQUENTIAL)
                return srf.SrfFile.from_file(mm)
        else:
            return srf.SrfFile.from_file(output_buffer(output))
