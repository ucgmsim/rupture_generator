"""Reading and writing SRF (Standard Rupture Format) files.

The in-memory model is arrays, one per field, named for what they hold and in what
unit: `points.onset_s` is a float32 array of one onset per subfault, and delaying a
rupture is `points.onset_s += 1`. `assemble.py` fills these from a generated segment,
converting SI into the centimetres and dyne-centimetres the format stores.

This module reads the whole file into memory; use `qcore.srf` instead to stream one.
`SrfFile` cites the format.

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
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import h5py
import numpy as np
import scipy as sp

from rupture_generator import srf_parser  # ty: ignore[unresolved-import]

FloatArray = np.ndarray[tuple[int], np.dtype[np.float32]]

SUPPORTED_VERSIONS = frozenset({"1.0", "2.0"})
"""SRF versions this module reads and writes.

Version 3.0 adds Vp and a full moment tensor per point, which the parser does not
implement.
"""


class ParseError(Exception):
    """Raised when a file is not valid SRF."""


@dataclasses.dataclass(frozen=True)
class PlaneHeader:
    """One segment's entry in the SRF header: the `PLANE` block as the file stores it.

    Following the SRF's own convention, `hypocentre_strike_km` is measured from the
    segment's along-strike **centre** and `hypocentre_dip_km` from its top edge.
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
    fastest, within each segment in turn. `shear_speed_cm_s` and `density_g_cm3` are
    the version 2.0 material properties, present together or not at all. Each field's
    name says what it holds and in what unit, with three exceptions: ``onset_s`` is the
    SRF's ``tinit`` and ``sample_interval_s`` its ``dt``; ``rise_time_s`` is *derived*
    on read as ``nt1 * dt``; and ``shear_speed_cm_s`` is centimetres per second, 1e5
    from the kilometres per second a velocity model is written in.
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
        """Take a subset of the subfaults."""
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
    """A read-only view of an SRF's points, one segment at a time."""

    def __init__(self, planes: Sequence[PlaneHeader], points: Points) -> None:
        """Initialise the Segments object."""
        self._planes = planes
        self._points = points

    def __getitem__(self, index: int) -> Points:  # ty: ignore[invalid-method-override]
        """Get the points of the nth segment in the SRF."""
        if not isinstance(index, int):
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

Named rather than positional: the two coordinates and the two hypocentre offsets are
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
`SLIP2`/`NT2`/`SLIP3`/`NT3` are rake components this format does not carry.
"""


@dataclasses.dataclass
class SrfFile:
    """Representation of an SRF file.

    `version` is declared rather than inferred from what the points happen to carry,
    and the constructor refuses a declaration the points contradict. `slip_rate[i, j]`
    is the slip rate of the ith subfault `j` samples after its **own** onset, so row
    `i` is as long as that subfault's pulse and the matrix as wide as the longest.

    References
    ----------
    SRF File Format Doc.
    https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+On+GM
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
    def from_file(cls, srf_ffp: Path | str) -> Self:
        """Read an srf file from a filepath.

        Raises
        ------
        ParseError
            If the file is not valid SRF.
        """
        try:
            with (
                open(srf_ffp, "rb") as f,
                mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm,
            ):
                # Windows doesn't have madvise
                if hasattr(mm, "madvise"):
                    mm.madvise(mmap.MADV_SEQUENTIAL)
                py_srf = srf_parser.parse_srf(mm)
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
        """Write an SRFFile object to a file."""
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

        # No `indices`: every pulse starts at column zero, so `row_ptr` already says
        # which column a sample is in, and widening `indices` to `uint64` to hand it
        # over would be 7.6 GB of ignored buffer on the twenty-fault scenario.
        # `asarray` rather than `astype`: the data already arrives as float32.
        slipt1 = srf_parser.PyCsrMatrix(
            row_ptr=self.slip_rate.indptr.astype(np.uint64),
            data=np.asarray(self.slip_rate.data, dtype=np.float32),
        )

        py_srf_file = srf_parser.PySrfFile(planes, metadata, slipt1)
        srf_parser.write_srf(py_srf_file, str(srf_ffp))

    def write_sw4_hdf5(
        self,
        output_ffp: Path | str,
    ) -> None:
        """Write the SRF file in SW4's SRF-HDF5 format.

        The whole file in one block, because an `SrfFile` is the whole file already.
        A rupture too large to be one goes through `Sw4Hdf5Stream`, which this
        delegates to, so there is one statement of the layout rather than two.

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
        with Sw4Hdf5Stream(output_ffp, self.version) as stream:
            for plane in self.planes:
                stream.plane(plane)
            stream.points(
                self.points,
                np.diff(self.slip_rate.indptr),
                # `asarray`, not `astype`: already float32, and a copy here is 3.8 GB
                # on a twenty-fault rupture.
                np.asarray(self.slip_rate.data, dtype=np.float32),
            )

    @property
    def nt(self):  # numpydoc ignore=RT01
        """int: Samples in the longest slip-rate pulse.

        **Not** the rupture's duration in samples: every row of `slip_rate` starts at
        column zero and the onset is a float in `points.onset_s`. For the duration, use
        `(points.onset_s.max() / dt) + nt`.
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


SW4_POINTS_PER_CHUNK = 1 << 14
"""How many points one HDF5 chunk of the ``POINTS`` dataset holds.

`SW4_POINTS_DTYPE` is 68 bytes, so this is a 1.1 MB chunk -- above the ~64 KB below
which per-chunk B-tree overhead shows, and small enough that a reader wanting a few
points does not pull a hundred megabytes.
"""

SW4_SAMPLES_PER_CHUNK = 1 << 20
"""How many slip-rate samples one HDF5 chunk of ``SR1`` holds.

Four megabytes of float32, trading B-tree entries against the granularity of a partial
read. Independent of any memory budget a streaming writer works to: that bounds what is
held at once, this is the layout on disk.
"""


class Sw4Hdf5Stream:
    """SW4's SRF-HDF5 format, written a block of points at a time.

    The format is append-only in subfault order: ``POINTS`` is one record per subfault
    and ``SR1`` is every pulse concatenated in the same order, with each subfault's
    length in its own ``NT1`` field. There is no index array to rebuild across blocks,
    so a block can be written and forgotten; the only invariant across blocks is that
    ``NT1`` and the samples appended agree.

    ``PLANE`` is an attribute rather than a dataset and attributes cannot grow, so the
    headers are collected and written on close -- which is why this is a context
    manager. ``version`` is stored as the file's ``VERSION`` attribute in float32.

    Examples
    --------
    >>> with Sw4Hdf5Stream(path, "2.0") as stream:  # doctest: +SKIP
    ...     stream.plane(header)
    ...     for block, lengths, samples in blocks:
    ...         stream.points(block, lengths, samples)
    """

    def __init__(self, output_ffp: Path | str, version: str) -> None:
        """Hold where to write and what version to declare."""
        self._path = output_ffp
        self._version = version
        self._planes: list[PlaneHeader] = []
        self._file: h5py.File | None = None
        self._points: h5py.Dataset | None = None
        self._samples: h5py.Dataset | None = None

    def __enter__(self) -> Self:
        """Open the file and create both datasets empty and growable."""
        self._file = h5py.File(self._path, "w")
        self._file.attrs.create("VERSION", np.float32(self._version))
        self._points = self._file.create_dataset(
            "POINTS",
            shape=(0,),
            maxshape=(None,),
            dtype=SW4_POINTS_DTYPE,
            chunks=(SW4_POINTS_PER_CHUNK,),
        )
        self._samples = self._file.create_dataset(
            "SR1",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=(SW4_SAMPLES_PER_CHUNK,),
        )
        return self

    def __exit__(self, *exception: object) -> None:
        """Write the plane headers and close, whether or not the block raised."""
        assert self._file is not None
        plane_data = np.zeros(len(self._planes), dtype=SW4_PLANE_DTYPE)
        for sw4_field, attribute in _SW4_PLANE_FIELDS.items():
            plane_data[sw4_field] = [
                getattr(plane, attribute) for plane in self._planes
            ]
        self._file.attrs.create("PLANE", plane_data)
        self._file.close()
        self._file = self._points = self._samples = None

    def plane(self, header: PlaneHeader) -> None:
        """Add one PLANE record. Order is the order the points follow."""
        self._planes.append(header)

    def points(
        self,
        points: Points,
        pulse_lengths: np.ndarray,
        samples_cm_s: np.ndarray,
    ) -> None:
        """Append one block of subfaults and their concatenated pulses.

        Parameters
        ----------
        points : Points
            The block's columns, already in the format's own units. Material
            properties are written when there, which is what makes the file 2.0.
        pulse_lengths : np.ndarray
            ``(n,)`` samples in each of this block's pulses -- the ``NT1`` column.
        samples_cm_s : np.ndarray
            The block's pulses concatenated, in centimetres per second, as long as
            ``pulse_lengths.sum()``.

        Raises
        ------
        RuntimeError
            If called outside the context manager, where there is no open file.
        ValueError
            If the samples handed over are not as many as the lengths claim, which
            would shift every later subfault's pulse.
        """
        if self._points is None or self._samples is None:
            raise RuntimeError(
                "an Sw4Hdf5Stream writes inside its `with` block; outside it there is "
                "no open file"
            )
        expected = int(np.sum(pulse_lengths))
        if samples_cm_s.size != expected:
            raise ValueError(
                f"this block claims {expected} slip-rate samples across "
                f"{len(pulse_lengths)} subfaults and carries {samples_cm_s.size}. "
                "SR1 is every pulse concatenated in subfault order, so a block that "
                "disagrees with its own NT1 shifts every later subfault's pulse"
            )

        block = np.zeros(len(points), dtype=SW4_POINTS_DTYPE)
        for sw4_field, attribute in _SW4_POINT_FIELDS.items():
            block[sw4_field] = getattr(points, attribute)
        block["NT1"] = pulse_lengths
        if points.has_material_properties:
            block["VS"] = points.shear_speed_cm_s
            block["DEN"] = points.density_g_cm3

        at = self._points.shape[0]
        self._points.resize((at + len(block),))
        self._points[at:] = block

        at = self._samples.shape[0]
        self._samples.resize((at + samples_cm_s.size,))
        self._samples[at:] = samples_cm_s


def read_srf(srf_ffp: Path | str) -> SrfFile:
    """Read an SRF file into an SrfFile object."""
    return SrfFile.from_file(srf_ffp)


def write_srf(srf_ffp: str | Path, srf: SrfFile) -> None:
    """Write an SRF object to a filepath."""
    srf.write_srf(srf_ffp)
