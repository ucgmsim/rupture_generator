"""The GSF file, which is how genslip is told where the fault is.

**Nothing here is part of `rupture_generator`.** The library takes geometry as arrays;
only the binary reads it from a file, so the format lives with the rest of genslip's
vocabulary. `PRUNED.md` records the reader as deliberately not ported.

The format, as `read_gsfpars_vsden_as` reads it (`Genslip/v5.6.2/iofunc.c:521`):

```
# any number of comment lines
<subfault count>
lon lat dep ds dw stk dip rake slip tinit segno
... one such line per subfault
```

`slip` and `tinit` are inputs on the other path, where genslip is handed a slip
distribution instead of generating one. On the generation path it overwrites both, and
the convention is to write -1 and 0.

Four quantities genslip *derives* from this file rather than being told
(`iofunc.c:645-676`) are properties here, because a caller has to pass some of them
back on the command line and they have to agree.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

FloatArray = np.ndarray[tuple[int], np.dtype[np.float32]]

# genslip's own degrees-to-radians constant, truncated at nine digits
# (`iofunc.c:530`). Not math.radians(1): the difference is 3.6e-10 per degree, which
# is enough to move the last bits of a float32 dtop, and dtop is compared against the
# value the binary puts in its SRF header.
RADIANS_PER_DEGREE = 0.017453293

_COLUMNS = (
    "longitude_deg",
    "latitude_deg",
    "depth_km",
    "along_strike_km",
    "down_dip_km",
    "strike_deg",
    "dip_deg",
    "rake_deg",
    "slip_cm",
    "onset_s",
)
"""The float columns, in file order. `segment` is an int and is handled separately."""


@dataclasses.dataclass
class GsfSubfaults:
    """A GSF file's subfaults, one array per column, in file order.

    Attributes
    ----------
    longitude_deg, latitude_deg : FloatArray
        Where the subfault's centre is.
    depth_km : FloatArray
        How deep its centre is.
    along_strike_km, down_dip_km : FloatArray
        Its dimensions -- `ds` and `dw`. genslip averages these into the `dstk` and
        `ddip` it uses everywhere else.
    strike_deg, dip_deg, rake_deg : FloatArray
        Its orientation and slip direction.
    slip_cm, onset_s : FloatArray
        Overwritten by the generation path. Write -1 and 0.
    segment : np.ndarray
        Which fault segment it belongs to, zero-based.
    """

    longitude_deg: FloatArray
    latitude_deg: FloatArray
    depth_km: FloatArray
    along_strike_km: FloatArray
    down_dip_km: FloatArray
    strike_deg: FloatArray
    dip_deg: FloatArray
    rake_deg: FloatArray
    slip_cm: FloatArray
    onset_s: FloatArray
    segment: np.ndarray

    def __post_init__(self) -> None:
        """Check every column describes the same set of subfaults.

        Raises
        ------
        ValueError
            If the columns are not all the same length, or there are none.
        """
        lengths = {name: len(getattr(self, name)) for name in (*_COLUMNS, "segment")}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"GSF columns disagree on length: {lengths}")
        if not any(lengths.values()):
            raise ValueError("a GSF needs at least one subfault")

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The number of subfaults.
        """
        return len(self.longitude_deg)

    @property
    def mean_along_strike_km(self) -> float:
        """float: The `dstk` genslip derives, averaged in double as the C does."""
        return float(np.mean(self.along_strike_km, dtype=np.float64))

    @property
    def mean_down_dip_km(self) -> float:
        """float: The `ddip` genslip derives, averaged in double as the C does."""
        return float(np.mean(self.down_dip_km, dtype=np.float64))

    @property
    def mean_dip_deg(self) -> float:
        """float: The average dip genslip derives.

        Accumulated in **float32** and divided once, because that is what
        `*dip = *dip + psrc[i].dip` over a `float *dip` does -- unlike `ds` and `dw`,
        which the same loop accumulates into doubles.
        """
        total = np.float32(0.0)
        for dip in self.dip_deg:
            total = np.float32(total + dip)
        return float(total / np.float32(len(self)))

    @property
    def top_depth_km(self) -> float:
        """float: The `dtop` genslip derives, and puts in the SRF header.

        The shallowest subfault *centre*, lifted by half a subfault's vertical extent
        to reach the top edge, and floored at the surface.
        """
        shallowest = float(np.min(self.depth_km))
        half_width = 0.5 * self.mean_down_dip_km
        top = shallowest - half_width * math.sin(self.mean_dip_deg * RADIANS_PER_DEGREE)
        return max(top, 0.0)


def write_gsf(subfaults: GsfSubfaults, output: Path) -> None:
    """Write subfaults as a GSF file.

    Parameters
    ----------
    subfaults : GsfSubfaults
        The subfaults to write.
    output : Path
        Where to write them.
    """
    columns = [getattr(subfaults, name) for name in _COLUMNS]
    lines = [f"{len(subfaults)}"]
    lines.extend(
        " ".join(f"{column[index]:14.6f}" for column in columns)
        + f" {int(subfaults.segment[index]):4d}"
        for index in range(len(subfaults))
    )
    output.write_text("\n".join(lines) + "\n")


def read_gsf(gsf_ffp: Path) -> GsfSubfaults:
    """Read a GSF file.

    Parameters
    ----------
    gsf_ffp : Path
        The file to read.

    Returns
    -------
    GsfSubfaults
        Its subfaults.

    Raises
    ------
    ValueError
        If the file declares a subfault count it does not then supply.
    """
    lines = [
        line
        for line in gsf_ffp.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    declared = int(lines[0].split()[0])
    rows = [line.split() for line in lines[1:]]
    if len(rows) != declared:
        raise ValueError(
            f"{gsf_ffp} declares {declared} subfaults and supplies {len(rows)}"
        )

    values = np.array(
        [[float(field) for field in row[: len(_COLUMNS)]] for row in rows]
    )
    return GsfSubfaults(
        **{
            name: values[:, index].astype(np.float32)
            for index, name in enumerate(_COLUMNS)
        },
        segment=np.array([int(row[len(_COLUMNS)]) for row in rows]),
    )


def on_a_plane(
    strike_count: int,
    dip_count: int,
    along_strike_km: float,
    down_dip_km: float,
    centre_longitude_deg: float,
    centre_latitude_deg: float,
    strike_deg: float,
    dip_deg: float,
    top_depth_km: float,
    rake_deg: float,
) -> GsfSubfaults:
    """Lay subfaults out on one plane, along strike fastest.

    A flat-earth layout, deliberately: this produces *fixture input*, and what matters
    about a fixture is that it is fixed and that both sides are handed the same one.
    It is not a claim about where a fault is, and it is not the geodesy the library
    refuses to do -- see `assemble.py`.

    Parameters
    ----------
    strike_count, dip_count : int
        Subfaults along strike and down dip.
    along_strike_km, down_dip_km : float
        One subfault's dimensions.
    centre_longitude_deg, centre_latitude_deg : float
        Where the top edge's centre is.
    strike_deg, dip_deg : float
        The plane's orientation.
    top_depth_km : float
        Depth of the top edge.
    rake_deg : float
        Slip direction, uniform.

    Returns
    -------
    GsfSubfaults
        One subfault per grid cell.
    """
    # Kilometres per degree, at the plane's own latitude. A fixture needs subfaults a
    # plausible distance apart, not a defensible projection.
    km_per_degree_latitude = 111.195
    km_per_degree_longitude = km_per_degree_latitude * math.cos(
        math.radians(centre_latitude_deg)
    )

    strike_index, dip_index = np.meshgrid(np.arange(strike_count), np.arange(dip_count))
    strike_index = strike_index.ravel()
    dip_index = dip_index.ravel()

    # Distance from the top-edge centre: along strike, and down dip projected onto the
    # surface.
    along = (strike_index - 0.5 * (strike_count - 1)) * along_strike_km
    down = (dip_index + 0.5) * down_dip_km
    surface = down * math.cos(math.radians(dip_deg))
    depth = top_depth_km + down * math.sin(math.radians(dip_deg))

    strike_radians = math.radians(strike_deg)
    # Dip direction is 90 degrees clockwise of strike.
    dip_radians = strike_radians + math.pi / 2
    north = along * math.cos(strike_radians) + surface * math.cos(dip_radians)
    east = along * math.sin(strike_radians) + surface * math.sin(dip_radians)

    count = strike_count * dip_count
    return GsfSubfaults(
        longitude_deg=(centre_longitude_deg + east / km_per_degree_longitude).astype(
            np.float32
        ),
        latitude_deg=(centre_latitude_deg + north / km_per_degree_latitude).astype(
            np.float32
        ),
        depth_km=depth.astype(np.float32),
        along_strike_km=np.full(count, along_strike_km, dtype=np.float32),
        down_dip_km=np.full(count, down_dip_km, dtype=np.float32),
        strike_deg=np.full(count, strike_deg, dtype=np.float32),
        dip_deg=np.full(count, dip_deg, dtype=np.float32),
        rake_deg=np.full(count, rake_deg, dtype=np.float32),
        # genslip overwrites both on the generation path.
        slip_cm=np.full(count, -1.0, dtype=np.float32),
        onset_s=np.zeros(count, dtype=np.float32),
        segment=np.zeros(count, dtype=int),
    )
