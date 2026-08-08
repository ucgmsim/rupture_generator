"""Module for handling SRF (Standard Rupture Format) files.

This module provides classes and functions for reading and writing SRF files,
as well as representing their contents.
See https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+On+GM
for details on the SRF format.

The in-memory model is arrays, one per field, named for what they hold and in what
unit. `SrfFile.points` is a `Points`, not a table: `points.onset_s` is a float32
array of one onset per subfault, and delaying a rupture is `points.onset_s += 1`.
The names match `GeneratedRupture` and `assemble.SubfaultGeometry` field for field,
so assembling an SRF out of a generated rupture is a copy rather than a translation.

**Why not qcore.srf?**

You might use this module instead of the `qcore.srf` module because:

1. The `qcore.srf` module does not support writing SRF files.

2. The points are arrays, so they can be manipulated with vectorised operations.

3. There is better documentation for the new module than the old one.

You should use `qcore.srf` if you do not eventually intend to read all
points of the SRF file (it is memory efficient), or you are working
with code that already uses `qcore.srf`.

Classes: ``SrfFile`` (representation of an SRF file), ``PlaneHeader`` (one segment's
header entry), ``Points`` (the subfaults).

Functions: ``read_srf`` (read an SRF file into memory), ``write_srf`` (write an SRF
object to a filepath).

Examples
--------
>>> srf_file = srf.read_srf('/path/to/srf')
>>> srf_file.points.onset_s.max() # get the last time any point in the SRF ruptures
>>> srf_file.points.onset_s += 1 # delay all points by one second
>>> srf.write_srf('/path/to/srf', srf_file)
"""

import dataclasses
import itertools
import mmap
from collections.abc import Buffer, Sequence
from pathlib import Path
from typing import Self

import h5py
import numpy as np
import scipy as sp

from rupture_generator import srf_parser  # ty: ignore[unresolved-import]

FloatArray = np.ndarray[tuple[int], np.dtype[np.float32]]

SUPPORTED_VERSIONS = frozenset({"1.0", "2.0"})
"""SRF versions this module reads and writes.

Version 3.0 adds Vp and a full moment tensor per point. The parser does not
implement it, so it is not listed here — and the configuration schema should not
advertise it either.
"""


class ParseError(Exception):
    """Raised when a file is not valid SRF.

    Two lines, copied from `source_modelling.parse_utils` rather than depended on:
    it was the only symbol this module took from that file.
    """


@dataclasses.dataclass(frozen=True)
class PlaneHeader:
    """One segment's entry in the SRF header.

    This is the `PLANE` block as the file stores it, and nothing more: no projection
    and no geometry.

    `hypocentre_strike_km` is measured from the segment's along-strike **centre** and
    `hypocentre_dip_km` from its top edge — genslip's convention, and the one
    `realisation_to_srf.py` already converts into.
    """

    centre_longitude_deg: float
    centre_latitude_deg: float
    strike_count: int
    dip_count: int
    length_km: float
    width_km: float
    strike_deg: float
    dip_deg: float
    top_depth_km: float
    hypocentre_strike_km: float
    hypocentre_dip_km: float

    @property
    def subfault_count(self) -> int:  # numpydoc ignore=RT01
        """int: How many of the file's points belong to this segment."""
        return self.strike_count * self.dip_count


@dataclasses.dataclass
class Points:
    """The subfaults of an SRF, one array per field.

    Every array holds one value per subfault, in the file's order: along strike
    fastest, within each segment in turn.

    `shear_speed_cm_s` and `density_g_cm3` are the version 2.0 material properties.
    They are present together or not at all, which is what distinguishes a version
    2.0 point block from a version 1.0 one.

    Attributes
    ----------
    longitude_deg, latitude_deg : FloatArray
        Where the subfault is.
    depth_km : FloatArray
        How deep it is.
    strike_deg, dip_deg : FloatArray
        Its local orientation, which need not equal its segment's.
    area_cm2 : FloatArray
        Its area, in the square centimetres the format stores and the moment sum is
        expressed in.
    onset_s : FloatArray
        When it starts slipping. The SRF calls this `tinit`.
    sample_interval_s : FloatArray
        The timestep of its slip-rate function. The SRF calls this `dt`.
    rake_deg : FloatArray
        The slip direction.
    slip_cm : FloatArray
        Total slip.
    rise_time_s : FloatArray
        The duration of the slip-rate function, `nt1 * dt`. Derived on read rather
        than stored: the file holds `nt1`.
    shear_speed_cm_s : FloatArray | None
        Shear-wave speed, in **centimetres** per second — the file's unit, a factor
        of 1e5 away from the kilometres per second a velocity model is written in.
    density_g_cm3 : FloatArray | None
        Density.
    """

    longitude_deg: FloatArray
    latitude_deg: FloatArray
    depth_km: FloatArray
    strike_deg: FloatArray
    dip_deg: FloatArray
    area_cm2: FloatArray
    onset_s: FloatArray
    sample_interval_s: FloatArray
    rake_deg: FloatArray
    slip_cm: FloatArray
    rise_time_s: FloatArray
    shear_speed_cm_s: FloatArray | None = None
    density_g_cm3: FloatArray | None = None

    def __post_init__(self) -> None:
        """Check every array describes the same set of subfaults.

        A DataFrame refused ragged columns for free. Nothing else here would notice,
        and a column filled from the wrong array survives a round trip through the
        file.

        Raises
        ------
        ValueError
            If the arrays are not all the same length, or only one of the two
            material properties is present.
        """
        present = {
            field.name: values
            for field in dataclasses.fields(self)
            if (values := getattr(self, field.name)) is not None
        }
        lengths = {name: len(values) for name, values in present.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"point arrays disagree on length: {lengths}")

        material = ("shear_speed_cm_s", "density_g_cm3")
        if len(present.keys() & material) == 1:
            raise ValueError(
                "shear_speed_cm_s and density_g_cm3 are the version 2.0 material "
                "properties and go together; this has only "
                f"{next(iter(present.keys() & material))}"
            )

    @property
    def has_material_properties(self) -> bool:  # numpydoc ignore=RT01
        """bool: Whether the points carry version 2.0's shear speed and density."""
        return self.shear_speed_cm_s is not None

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The number of subfaults.
        """
        return len(self.longitude_deg)

    def __getitem__(self, subfaults: slice | np.ndarray) -> Self:
        """Take a subset of the subfaults.

        Parameters
        ----------
        subfaults : slice | np.ndarray
            A slice, or an array of indices or of booleans.

        Returns
        -------
        Self
            The selected subfaults, as their own `Points`.

        Raises
        ------
        TypeError
            If given a single index. One subfault's worth of scalars is not a
            `Points` and pretending otherwise produces arrays of rank zero.
        """
        if not isinstance(subfaults, slice | np.ndarray):
            raise TypeError(
                f"points are selected by a slice or an array, not {type(subfaults).__name__}"
            )
        return dataclasses.replace(
            self,
            **{
                field.name: values[subfaults]
                for field in dataclasses.fields(self)
                if (values := getattr(self, field.name)) is not None
            },
        )


class Segments(Sequence):
    """A read-only view of an SRF's points, one segment at a time.

    Parameters
    ----------
    planes : Sequence[PlaneHeader]
        The header of the SRF file.
    points : Points
        The points of the SRF file.
    """

    def __init__(self, planes: Sequence[PlaneHeader], points: Points) -> None:
        """Initialise the Segments object.

        Parameters
        ----------
        planes : Sequence[PlaneHeader]
            The header of the SRF file.
        points : Points
            The points of the SRF file.
        """
        self._planes = planes
        self._points = points

    # ty: slice overload missing to satisfy Sequence LSP; fix by adding
    # @overload stubs for int and slice once slice support is implemented.
    def __getitem__(self, index: int) -> Points:  # ty: ignore[invalid-method-override]
        """Get the points of the nth segment in the SRF.

        Parameters
        ----------
        index : int
            The index of the segment.

        Returns
        -------
        Points
            The points belonging to the nth segment.
        """
        if not isinstance(index, int):
            # NOTE: We are not covering this in test coverage because
            # we intend to support slicing in the future.
            raise TypeError(
                "Segment index must an integer, not slice or tuple"
            )  # pragma: no cover
        boundaries = list(
            itertools.accumulate(
                (plane.subfault_count for plane in self._planes), initial=0
            )
        )
        return self._points[boundaries[index] : boundaries[index + 1]]

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The number of segments in the SRF.
        """
        return len(self._planes)


SW4_PLANE_DTYPE = np.dtype(
    [
        ("ELON", "f4"),
        ("ELAT", "f4"),
        ("NSTK", "i4"),
        ("NDIP", "i4"),
        ("LEN", "f4"),
        ("WID", "f4"),
        ("STK", "f4"),
        ("DIP", "f4"),
        ("DTOP", "f4"),
        ("SHYP", "f4"),
        ("DHYP", "f4"),
    ]
)

SW4_POINTS_DTYPE = np.dtype(
    [
        ("LON", "f4"),
        ("LAT", "f4"),
        ("DEP", "f4"),
        ("STK", "f4"),
        ("DIP", "f4"),
        ("AREA", "f4"),
        ("TINIT", "f4"),
        ("DT", "f4"),
        ("VS", "f4"),
        ("DEN", "f4"),
        ("RAKE", "f4"),
        ("SLIP1", "f4"),
        ("NT1", "i4"),
        ("SLIP2", "f4"),
        ("NT2", "i4"),
        ("SLIP3", "f4"),
        ("NT3", "i4"),
    ]
)

_SW4_PLANE_FIELDS = {
    "ELON": "centre_longitude_deg",
    "ELAT": "centre_latitude_deg",
    "NSTK": "strike_count",
    "NDIP": "dip_count",
    "LEN": "length_km",
    "WID": "width_km",
    "STK": "strike_deg",
    "DIP": "dip_deg",
    "DTOP": "top_depth_km",
    "SHYP": "hypocentre_strike_km",
    "DHYP": "hypocentre_dip_km",
}
"""Which attribute of a `PlaneHeader` fills each SW4 plane field.

Named rather than positional. The two coordinates and the two hypocentre offsets are
adjacent and the same width, so a transposition would be silent.
"""

_SW4_POINT_FIELDS = {
    "LON": "longitude_deg",
    "LAT": "latitude_deg",
    "DEP": "depth_km",
    "STK": "strike_deg",
    "DIP": "dip_deg",
    "AREA": "area_cm2",
    "TINIT": "onset_s",
    "DT": "sample_interval_s",
    "RAKE": "rake_deg",
    "SLIP1": "slip_cm",
}
"""Which array of a `Points` fills each SW4 point field.

`VS` and `DEN` are version 2.0 only, `NT1` comes from the slip-rate matrix, and
`SLIP2`/`NT2`/`SLIP3`/`NT3` describe rake components this format does not carry and
stay zero.
"""


@dataclasses.dataclass
class SrfFile:
    """Representation of an SRF file.

    `version` is not inferred from what the points happen to carry. It is declared,
    and the constructor refuses a declaration the points contradict — which is the
    only way the two cannot disagree at write time.

    Attributes
    ----------
    version : str
        The version of this SrfFile, one of `SUPPORTED_VERSIONS`.
    planes : list[PlaneHeader]
        The header of the SRF file: one entry per segment.
    points : Points
        The subfaults, one array per field. See `Points`.
    slip_rate : csr_array
        A sparse array of the slip-rate function of every point, where
        `slip_rate[i, j]` is the slip rate of the ith subfault `j` samples after its
        own onset. Row `i` is as long as that subfault's pulse; the matrix is as wide
        as the longest one.

    References
    ----------
    SRF File Format Doc: https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+On+GM
    """

    version: str
    planes: list[PlaneHeader]
    points: Points
    slip_rate: sp.sparse.csr_array

    def __post_init__(self) -> None:
        """Check the declared version and the points agree.

        Raises
        ------
        ValueError
            If the version is not supported, or the material properties are present
            when it is 1.0 or absent when it is 2.0.
        """
        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError(
                f"unsupported SRF version {self.version!r}; "
                f"supported versions are {', '.join(sorted(SUPPORTED_VERSIONS))}"
            )
        if self.version == "2.0" and not self.points.has_material_properties:
            raise ValueError(
                "SRF version 2.0 carries vs and den per point, and these points have "
                "neither. Add them, or set version to '1.0'."
            )
        if self.version == "1.0" and self.points.has_material_properties:
            raise ValueError(
                "SRF version 1.0 has nowhere to put vs and den. Drop them, or set "
                "version to '2.0'."
            )

    @classmethod
    def from_file(cls, srf_ffp: Path | str | Buffer) -> Self:
        """Read an srf file from a filepath.

        Parameters
        ----------
        srf_ffp : Path | str | Buffer
            Either a path-like pointing to a file, or a buffer containg raw SRF bytes.

        Returns
        -------
        Self
            The SRFFile instance for this path.

        Raises
        ------
        ParseError
            If the file is not valid SRF.
        """
        try:
            if isinstance(srf_ffp, (Path, str)):
                with (
                    open(srf_ffp, "rb") as f,
                    mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm,
                ):
                    # Windows doesn't have madvise
                    if hasattr(mm, "madvise"):
                        mm.madvise(mmap.MADV_SEQUENTIAL)
                    py_srf = srf_parser.parse_srf(mm)
            else:
                py_srf = srf_parser.parse_srf(srf_ffp)
        except ValueError as parse_error:
            raise ParseError(str(parse_error)) from parse_error

        metadata = py_srf.metadata
        version = "2.0" if metadata.vs is not None else "1.0"

        planes = [
            PlaneHeader(
                centre_longitude_deg=plane.elon,
                centre_latitude_deg=plane.elat,
                strike_count=int(plane.nstk),
                dip_count=int(plane.ndip),
                length_km=plane.len,
                width_km=plane.wid,
                strike_deg=plane.stk,
                dip_deg=plane.dip,
                top_depth_km=plane.dtop,
                hypocentre_strike_km=plane.shyp,
                hypocentre_dip_km=plane.dhyp,
            )
            for plane in py_srf.planes
        ]

        points = Points(
            longitude_deg=metadata.lon,
            latitude_deg=metadata.lat,
            depth_km=metadata.dep,
            strike_deg=metadata.stk,
            dip_deg=metadata.dip,
            area_cm2=metadata.area,
            onset_s=metadata.tinit,
            sample_interval_s=metadata.dt,
            rake_deg=metadata.rake,
            slip_cm=metadata.slip1,
            rise_time_s=metadata.rise,
            shear_speed_cm_s=metadata.vs,
            density_g_cm3=metadata.density,
        )

        row_ptr = py_srf.slipt1.row_ptr
        data = py_srf.slipt1.data
        indices = py_srf.slipt1.indices

        n_timesteps = int(indices.max()) + 1 if len(indices) else 0
        slip_rate = sp.sparse.csr_array(
            (data, indices, row_ptr), shape=(len(row_ptr) - 1, n_timesteps)
        )

        return cls(version, planes, points, slip_rate)

    def write_srf(self, srf_ffp: str | Path) -> None:
        """Write an SRFFile object to a file.

        Parameters
        ----------
        srf_ffp : Path
            The path to the output SRF.
        """
        planes = [
            srf_parser.PySrfPlane(
                elon=plane.centre_longitude_deg,
                elat=plane.centre_latitude_deg,
                nstk=plane.strike_count,
                ndip=plane.dip_count,
                len=plane.length_km,
                wid=plane.width_km,
                stk=plane.strike_deg,
                dip=plane.dip_deg,
                dtop=plane.top_depth_km,
                shyp=plane.hypocentre_strike_km,
                dhyp=plane.hypocentre_dip_km,
            )
            for plane in self.planes
        ]

        def as_float32(values: FloatArray) -> FloatArray:
            return np.ascontiguousarray(values, dtype=np.float32)

        points = self.points
        metadata = srf_parser.PySrfMetadata(
            lon=as_float32(points.longitude_deg),
            lat=as_float32(points.latitude_deg),
            dep=as_float32(points.depth_km),
            stk=as_float32(points.strike_deg),
            dip=as_float32(points.dip_deg),
            area=as_float32(points.area_cm2),
            tinit=as_float32(points.onset_s),
            dt=as_float32(points.sample_interval_s),
            rake=as_float32(points.rake_deg),
            slip1=as_float32(points.slip_cm),
            rise=as_float32(points.rise_time_s),
            # The version and these two agree by construction -- see __post_init__ --
            # so the file cannot come out as a version other than `self.version`.
            vs=as_float32(points.shear_speed_cm_s)
            if points.shear_speed_cm_s is not None
            else None,
            density=as_float32(points.density_g_cm3)
            if points.density_g_cm3 is not None
            else None,
        )

        slipt1 = srf_parser.PyCsrMatrix(
            row_ptr=self.slip_rate.indptr.astype(np.uint64),
            indices=self.slip_rate.indices.astype(np.uint64),
            data=self.slip_rate.data.astype(np.float32),
        )

        py_srf_file = srf_parser.PySrfFile(planes, metadata, slipt1)
        srf_parser.write_srf(py_srf_file, str(srf_ffp))

    def write_sw4_hdf5(
        self,
        output_ffp: Path | str,
    ) -> None:
        """Write the SRF file in SW4's SRF-HDF5 format.

        Parameters
        ----------
        output_ffp : Path
            The path to the output HDF5 file.

        References
        ----------
        .. [1] Petersson, N.A. and B. Sjogreen (2017). SW4 v2.0.
           Computational Infrastructure of Geodynamics, Davis, CA.
           DOI: 10.5281/zenodo.1045297.
        .. [2] Petersson, N.A. and B. Sjogreen (2017). User's guide to
           SW4, version 2.0. Technical report LLNL-SM-741439, Lawrence
           Livermore National Laboratory, Livermore, CA.
           https://github.com/geodynamics/sw4/blob/master/doc/SW4_UsersGuide.pdf
        """
        plane_data = np.zeros(len(self.planes), dtype=SW4_PLANE_DTYPE)
        for sw4_field, attribute in _SW4_PLANE_FIELDS.items():
            plane_data[sw4_field] = [getattr(plane, attribute) for plane in self.planes]

        points_data = np.zeros(len(self.points), dtype=SW4_POINTS_DTYPE)
        for sw4_field, attribute in _SW4_POINT_FIELDS.items():
            points_data[sw4_field] = getattr(self.points, attribute)

        points_data["NT1"] = np.diff(self.slip_rate.indptr)
        if self.points.has_material_properties:
            points_data["VS"] = self.points.shear_speed_cm_s
            points_data["DEN"] = self.points.density_g_cm3

        with h5py.File(output_ffp, "w") as h5file:
            h5file.attrs.create("VERSION", np.float32(self.version))
            h5file.attrs.create("PLANE", plane_data)
            h5file.create_dataset("POINTS", data=points_data)
            h5file.create_dataset("SR1", data=self.slip_rate.data.astype(np.float32))

    @property
    def nt(self):  # numpydoc ignore=RT01
        """int: Samples in the longest slip-rate pulse.

        **Not** the rupture's duration in samples. Each row of `slip_rate` starts at
        column zero and the onset lives in `points.onset_s` as a float, so the matrix
        is as wide as the longest pulse rather than as wide as the rupture. For the
        duration, use `(points.onset_s.max() / dt) + nt`.
        """
        return self.slip_rate.shape[1]

    @property
    def dt(self):  # numpydoc ignore=RT01
        """float: time resolution of SRF."""
        return self.points.sample_interval_s[0]

    @property
    def segments(self) -> Segments:  # numpydoc ignore=RT01
        """Segments: A sequence of segments in the SRF."""
        return Segments(self.planes, self.points)


def read_srf(srf_ffp: Path | str | Buffer) -> SrfFile:
    """Read an SRF file into an SrfFile object.

    Parameters
    ----------
    srf_ffp : Path
        The filepath of the SRF file.

    Returns
    -------
    SrfFile
        The filepath of the SRF file.
    """
    return SrfFile.from_file(srf_ffp)


def write_srf(srf_ffp: str | Path, srf: SrfFile) -> None:
    """Write an SRF object to a filepath.

    Parameters
    ----------
    srf_ffp : Path
        The filepath to write the srf object to.
    srf : SrfFile
        The SRF object.
    """
    srf.write_srf(srf_ffp)
