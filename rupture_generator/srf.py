"""Module for handling SRF (Standard Rupture Format) files.

This module provides classes and functions for reading and writing SRF files,
as well as representing their contents.
See https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+On+GM
for details on the SRF format.

**Why not qcore.srf?**

You might use this module instead of the `qcore.srf` module because:

1. The `qcore.srf` module does not support writing SRF files.

2. Exposing SRF points as a pandas dataframe allows manipulation of
   the points using efficient vectorised operations. We use this in
   rupture propagation to delay ruptures by adding to the `tinit` column.

3. There is better documentation for the new module than the old one.

You should use `qcore.srf` if you do not eventually intend to read all
points of the SRF file (it is memory efficient), or you are working
with code that already uses `qcore.srf`.

Classes: ``SrfFile`` (representation of an SRF file).

Functions: ``read_srf`` (read an SRF file into memory), ``write_srf`` (write an SRF
object to a filepath).

Examples
--------
>>> srf_file = srf.read_srf('/path/to/srf')
>>> srf_file.points['tinit'].max() # get the last time any point in the SRF ruptures
>>> srf_file.points['tinit'] += 1 # delay all points by one second
>>> srf.write_srf('/path/to/srf', srf_file)
"""

import dataclasses
import mmap
from collections.abc import Buffer, Sequence
from pathlib import Path
from typing import Self

import h5py
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr

from rupture_generator import srf_parser  # ty: ignore[unresolved-import]

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

_SW4_POINTS_EXTERNAL_FIELDS = {"VS", "DEN", "NT1", "SLIP2", "NT2", "SLIP3", "NT3"}


class Segments(Sequence):
    """A read-only view for SRF segments.

    Parameters
    ----------
    header : pd.DataFrame
        The header of the SRF file.
    points : pd.DataFrame
        The points of the SRF file.
    """

    def __init__(self, header: pd.DataFrame, points: pd.DataFrame) -> None:
        """Initialise the Segments object.

        Parameters
        ----------
        header : pd.DataFrame
            The header of the SRF file.
        points : pd.DataFrame
            The points of the SRF file.
        """
        self._header = header
        self._points = points

    # ty: slice overload missing to satisfy Sequence LSP; fix by adding
    # @overload stubs for int and slice once slice support is implemented.
    def __getitem__(self, index: int) -> pd.DataFrame:  # ty: ignore[invalid-method-override]
        """Get the nth segment in the SRF.

        Parameters
        ----------
        index : int
            The index of the segment.

        Returns
        -------
        int
            The nth segment in the SRF.
        """
        if not isinstance(index, int):
            # NOTE: We are not covering this in test coverage because
            # we intend to support slicing in the future.
            raise TypeError(
                "Segment index must an integer, not slice or tuple"
            )  # pragma: no cover
        points_offset = (self._header["nstk"] * self._header["ndip"]).cumsum()
        if index == 0:
            return self._points.iloc[: points_offset.iloc[index]]
        return self._points.iloc[
            points_offset.iloc[index - 1] : points_offset.iloc[index]
        ]

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The number of segments in the SRF.
        """
        return len(self._header)


@dataclasses.dataclass
class SrfFile:
    """
    Representation of an SRF file.

    Attributes
    ----------
    version : str
        The version of this SrfFile
    header : pd.DataFrame
        A list of SrfSegment objects representing the header of the SRF file.
        The columns of the header are:

        - elon: The centre longitude of the plane.
        - elat: The centre latitude of the plane.
        - nstk: The number of patches along strike for the plane.
        - ndip: The number of patches along dip for the plane.
        - len: The length of the plane (in km).
        - wid: The width of the plane (in km).
        - stk: The plane strike.
        - dip: The plane dip.
        - dtop: The top of the plane.
        - shyp: The hypocentre location in strike coordinates.
        - dhyp: The hypocentre location in dip coordinates.


    points : pd.DataFrame
        A dataframe of the points (subfaults) in the SRF file. The columns are:

        - lon: longitude of the patch.
        - lat: latitude of the patch.
        - dep: depth of the patch (in kilometres).
        - stk: local strike.
        - dip: local dip.
        - area: area of the patch (in cm^2).
        - tinit: initial rupture time for this patch (in seconds).
        - dt: the timestep for all slipt columns (in seconds).
        - vs: shear-wave velocity at the patch (in cm/s). Version 2.0 only.
        - den: density at the patch (in g/cm^3). Version 2.0 only.
        - rake: local rake.
        - slip: total slip (in cm).
        - rise: total rise time (in seconds), computed as nt * dt.

        The vs and den columns are only present when version is "2.0". The
        rise column is computed from the SRF and is not written to disk. See
        the linked documentation on the SRF format for more details.

    slipt1_array : csr_array
        A sparse array containing the slip for each point and at each timestep, where
        slipt1_array[i, j] is the slip for the ith patch at time t = j * dt.

    References
    ----------
    SRF File Format Doc: https://wiki.canterbury.ac.nz/display/QuakeCore/File+Formats+Used+On+GM
    """

    version: str
    header: pd.DataFrame
    points: pd.DataFrame
    slipt1_array: sp.sparse.csr_array

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

        version = "2.0" if py_srf.metadata.vs is not None else "1.0"

        headers = pd.DataFrame(
            [
                {
                    "elon": plane.elon,
                    "elat": plane.elat,
                    "nstk": plane.nstk,
                    "ndip": plane.ndip,
                    "len": plane.len,
                    "wid": plane.wid,
                    "stk": plane.stk,
                    "dip": plane.dip,
                    "dtop": plane.dtop,
                    "shyp": plane.shyp,
                    "dhyp": plane.dhyp,
                }
                for plane in py_srf.planes
            ]
        )
        headers["nstk"] = headers["nstk"].astype(int)
        headers["ndip"] = headers["ndip"].astype(int)

        metadata = py_srf.metadata
        points_data = {
            "lon": metadata.lon,
            "lat": metadata.lat,
            "dep": metadata.dep,
            "stk": metadata.stk,
            "dip": metadata.dip,
            "area": metadata.area,
            "tinit": metadata.tinit,
            "dt": metadata.dt,
        }
        if version == "2.0":
            points_data["vs"] = metadata.vs
            points_data["den"] = metadata.density
        points_data["rake"] = metadata.rake
        points_data["slip"] = metadata.slip1
        points_data["rise"] = metadata.rise
        points_df = pd.DataFrame(points_data)

        row_ptr = py_srf.slipt1.row_ptr
        data = py_srf.slipt1.data
        indices = py_srf.slipt1.indices

        n_timesteps = int(indices.max()) + 1 if len(indices) else 0
        slipt1_array = sp.sparse.csr_array(
            (data, indices, row_ptr), shape=(len(row_ptr) - 1, n_timesteps)
        )

        return cls(
            version,
            headers,
            points_df,
            slipt1_array,
        )

    def write_srf(self, srf_ffp: str | Path) -> None:
        """Write an SRFFile object to a file.

        Parameters
        ----------
        srf_ffp : Path
            The path to the output SRF.

        """

        planes = [
            srf_parser.PySrfPlane(
                elon=row["elon"],
                elat=row["elat"],
                nstk=int(row["nstk"]),
                ndip=int(row["ndip"]),
                len=row["len"],
                wid=row["wid"],
                stk=row["stk"],
                dip=row["dip"],
                dtop=row["dtop"],
                shyp=row["shyp"],
                dhyp=row["dhyp"],
            )
            for _, row in self.header.iterrows()
        ]

        # The version written is `self.version`, and the point columns have to
        # agree with it. Previously the version was *inferred* from whether vs and
        # den happened to be present, so `self.version` was set on read and then
        # silently ignored on write -- a file could round-trip into a different
        # version than it came in as.
        if self.version not in SUPPORTED_VERSIONS:
            raise ValueError(
                f"cannot write SRF version {self.version!r}; "
                f"supported versions are {', '.join(sorted(SUPPORTED_VERSIONS))}"
            )

        has_material = "vs" in self.points and "den" in self.points
        if self.version == "2.0" and not has_material:
            raise ValueError(
                "SRF version 2.0 carries vs and den per point, and this file has "
                "neither. Add them, or set version to '1.0'."
            )

        metadata = srf_parser.PySrfMetadata(
            lon=self.points["lon"].to_numpy(dtype=np.float32),
            lat=self.points["lat"].to_numpy(dtype=np.float32),
            dep=self.points["dep"].to_numpy(dtype=np.float32),
            stk=self.points["stk"].to_numpy(dtype=np.float32),
            dip=self.points["dip"].to_numpy(dtype=np.float32),
            area=self.points["area"].to_numpy(dtype=np.float32),
            tinit=self.points["tinit"].to_numpy(dtype=np.float32),
            dt=self.points["dt"].to_numpy(dtype=np.float32),
            rake=self.points["rake"].to_numpy(dtype=np.float32),
            slip1=self.points["slip"].to_numpy(dtype=np.float32),
            rise=self.points["rise"].to_numpy(dtype=np.float32),
            vs=self.points["vs"].to_numpy(dtype=np.float32)
            if self.version == "2.0"
            else None,
            density=self.points["den"].to_numpy(dtype=np.float32)
            if self.version == "2.0"
            else None,
        )

        slipt1 = srf_parser.PyCsrMatrix(
            row_ptr=self.slip.indptr.astype(np.uint64),
            indices=self.slip.indices.astype(np.uint64),
            data=self.slip.data.astype(np.float32),
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
        plane_data = np.empty(len(self.header), dtype=SW4_PLANE_DTYPE)
        assert SW4_PLANE_DTYPE.names is not None
        for field in SW4_PLANE_DTYPE.names:
            plane_data[field] = self.header[field.lower()].values.astype(
                SW4_PLANE_DTYPE[field].type  # ty: ignore[invalid-argument-type]
            )  # ty: ignore[invalid-assignment]

        # Build POINTS structured array
        points_data: np.ndarray = np.zeros(len(self.points), dtype=SW4_POINTS_DTYPE)
        assert SW4_POINTS_DTYPE.names is not None
        for field in SW4_POINTS_DTYPE.names:
            if field in _SW4_POINTS_EXTERNAL_FIELDS:
                continue
            points_data[field] = self.points[
                "slip" if field == "SLIP1" else field.lower()
            ].values.astype(SW4_POINTS_DTYPE[field].type)  # ty: ignore

        points_data["NT1"] = np.diff(self.slipt1_array.indptr).astype(np.int32)
        if (
            self.version == "2.0"
        ):  # vs/den are mandatory in 2.0; missing columns will fail loudly
            points_data["VS"] = self.points["vs"].to_numpy().astype(np.float32)
            points_data["DEN"] = self.points["den"].to_numpy().astype(np.float32)

        with h5py.File(output_ffp, "w") as h5file:
            h5file.attrs.create("VERSION", np.float32(self.version))
            h5file.attrs.create("PLANE", plane_data)
            h5file.create_dataset("POINTS", data=points_data)
            h5file.create_dataset("SR1", data=self.slipt1_array.data.astype(np.float32))

    def write_hdf5(
        self, hdf5_ffp: Path, include_slip_time_function: bool = True
    ) -> None:
        """Write an SRFFile to disk in an HDF5 format using xarray's to_netcdf.

        Parameters
        ----------
        hdf5_ffp : Path
            The path to the HDF5 file to save to.
        include_slip_time_function : bool
            If True, include the slip time function in the HDF5
            output. Slower and outputs larger files.
        """

        self.to_xarray(include_slip_time_function=include_slip_time_function).to_netcdf(
            hdf5_ffp,
            engine="h5netcdf",
            encoding={
                # Apply compression to the 'data' variable of the sparse array
                "data": {"compression": "zlib", "complevel": 9},
                # Apply compression to the 'indices' variable of the sparse array
                "indices": {"compression": "zlib", "complevel": 9},
            }
            if include_slip_time_function
            else None,
        )

    @classmethod
    def from_hdf5(cls, hdf5_ffp: Path) -> Self:
        """
        Reads an SRFFile object from an HDF5 file.

        Parameters
        ----------
        hdf5_ffp : Path
            The file path to the HDF5 file.

        Returns
        -------
        SrfFile
            An instance of the SrfFile class reconstructed from the HDF5 data.
        """
        ds = xr.open_dataset(hdf5_ffp, engine="h5netcdf")

        header_data = {
            var_name[len("plane_") :]: ds[var_name].values
            for var_name in ds.data_vars
            if isinstance(var_name, str) and var_name.startswith("plane_")
        }
        header_df = pd.DataFrame(header_data)
        header_df[["nstk", "ndip"]] = header_df[["nstk", "ndip"]].astype(int)

        points_data = {
            col: ds[col].values
            for col in ds.data_vars
            if isinstance(col, str)
            and not col.startswith("plane_")
            and col not in {"data", "indices", "indptr"}
        }
        points_df = pd.DataFrame(points_data)

        data = ds["data"].values
        indices = ds["indices"].values
        indptr_saved = ds["indptr"].values
        reconstructed_indptr = np.append(indptr_saved, len(data))

        slipt1_array = sp.sparse.csr_array((data, indices, reconstructed_indptr))

        return cls(
            version=ds.attrs["version"],
            header=header_df,
            points=points_df,
            slipt1_array=slipt1_array,
        )

    def to_xarray(self, include_slip_time_function: bool = True) -> xr.Dataset:
        """Convert an SRFFile into an xarray dataset.

        Parameters
        ----------
        include_slip_time_function : bool, default False
            If True, include the slip time functions as well as the
            slip summaries in the SRF. Slower.

        Returns
        -------
        xr.Dataset
            An xarray dataset containing the information from an SRF
            file.
        """
        # Prepare data variables and coordinates for the header Dataset
        header_data_vars = {
            f"plane_{col}": ("segment", self.header[col].values)
            for col in self.header.columns
        }
        header_coords = {"segment": np.arange(len(self.header))}
        header_ds = xr.Dataset(header_data_vars, coords=header_coords)

        points_data_vars = {
            col: ("patch", self.points[col].values) for col in self.points.columns
        }
        points_coords = {"patch": np.arange(len(self.points))}
        points_ds = xr.Dataset(points_data_vars, coords=points_coords)

        datasets = [header_ds, points_ds]
        if include_slip_time_function:
            n_patches, n_timesteps = self.slipt1_array.shape
            slip_ds = xr.Dataset(
                {
                    "data": (("nz_idx",), self.slipt1_array.data),
                    "indices": (("nz_idx",), self.slipt1_array.indices),
                    "indptr": (
                        ("row",),
                        self.slipt1_array.indptr[:-1],
                    ),  # Apply slicing to the data
                },
                coords={
                    "row": np.arange(n_patches),
                    "col": np.arange(n_timesteps),
                },
                attrs={
                    "sparse_format": "csr",
                    "original_shape": self.slipt1_array.shape,
                    "units": "cm",
                    "description": "Slip for each patch at each timestep",
                },
            )
            datasets.append(slip_ds)
        ds = xr.merge(datasets)
        ds.attrs["version"] = self.version

        return ds

    @property
    def slip(self):  # numpydoc ignore=RT01
        "csr_array: A sparse array containing slip-time functions for each point."
        return self.slipt1_array


    @property
    def nt(self):  # numpydoc ignore=RT01
        """int: Samples in the longest slip-rate pulse.

        **Not** the rupture's duration in samples. Each row of `slipt1_array` starts
        at column zero and the onset lives in `points["tinit"]` as a float, so the
        matrix is as wide as the longest pulse rather than as wide as the rupture.
        For the duration, use `(points["tinit"].max() / dt) + nt`.
        """
        return self.slipt1_array.shape[1]

    @property
    def dt(self):  # numpydoc ignore=RT01
        """float: time resolution of SRF."""
        return self.points["dt"].iloc[0]

    @property
    def segments(self) -> Segments:  # numpydoc ignore=RT01
        """Segments: A sequence of segments in the SRF."""
        return Segments(self.header, self.points)



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
