"""Type stubs for the Rust kernels; the signatures are `crates/kernels/src/lib.rs`."""

import numpy as np
from numpy.typing import NDArray

def eikonal_solve(
    slowness: NDArray[np.float64],
    spacing_km: tuple[float, float],
    seeds: list[tuple[int, int, float]],
) -> NDArray[np.float64]: ...
def synthesise_pulses(
    slip_m: NDArray[np.float64],
    rise_time_s: NDArray[np.float64],
    dt_s: float,
    shape: str,
    beta: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...
def circulant_draw(
    eigenvalues: NDArray[np.float64],
    cell_counts: tuple[int, int],
    seed: int,
) -> NDArray[np.float64]: ...
