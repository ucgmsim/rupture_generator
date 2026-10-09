"""Type stubs for the SRF extension; the classes are `crates/srf/src/pytypes.rs`."""

from collections.abc import Buffer

import numpy as np
from numpy.typing import NDArray

class PySrfPlane:
    elon: float
    elat: float
    nstk: int
    ndip: int
    len: float
    wid: float
    stk: float
    dip: float
    dtop: float
    shyp: float
    dhyp: float
    def __init__(
        self,
        elon: float,
        elat: float,
        nstk: int,
        ndip: int,
        len: float,
        wid: float,
        stk: float,
        dip: float,
        dtop: float,
        shyp: float,
        dhyp: float,
    ) -> None: ...

class PyCsrMatrix:
    row_ptr: NDArray[np.uint64]
    data: NDArray[np.float32]
    indices: NDArray[np.uint64] | None
    def __init__(
        self,
        row_ptr: NDArray[np.uint64],
        data: NDArray[np.float32],
        indices: NDArray[np.uint64] | None = None,
    ) -> None: ...

class PySrfMetadata:
    lon: NDArray[np.float32]
    lat: NDArray[np.float32]
    dep: NDArray[np.float32]
    stk: NDArray[np.float32]
    dip: NDArray[np.float32]
    area: NDArray[np.float32]
    tinit: NDArray[np.float32]
    dt: NDArray[np.float32]
    rake: NDArray[np.float32]
    slip1: NDArray[np.float32]
    rise: NDArray[np.float32]
    vs: NDArray[np.float32] | None
    density: NDArray[np.float32] | None
    def __init__(
        self,
        lon: NDArray[np.float32],
        lat: NDArray[np.float32],
        dep: NDArray[np.float32],
        stk: NDArray[np.float32],
        dip: NDArray[np.float32],
        area: NDArray[np.float32],
        tinit: NDArray[np.float32],
        dt: NDArray[np.float32],
        rake: NDArray[np.float32],
        slip1: NDArray[np.float32],
        rise: NDArray[np.float32],
        vs: NDArray[np.float32] | None = None,
        density: NDArray[np.float32] | None = None,
    ) -> None: ...

class PySrfFile:
    planes: list[PySrfPlane]
    metadata: PySrfMetadata
    slipt1: PyCsrMatrix
    def __init__(
        self,
        planes: list[PySrfPlane],
        metadata: PySrfMetadata,
        slipt1: PyCsrMatrix,
    ) -> None: ...

def parse_srf(buffer: Buffer) -> PySrfFile: ...
def write_srf(py_srf_file: PySrfFile, file_path: str) -> None: ...
