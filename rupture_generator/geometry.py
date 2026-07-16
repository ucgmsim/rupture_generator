from dataclasses import dataclass

import numpy as np


@dataclass
class DiscretisedGeometry:
    x: np.ndarray
    y: np.ndarray
    depth: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    strike: np.ndarray
    dip: np.ndarray
    rake: np.ndarray
    segment_number: np.ndarray


VertexArray = np.ndarray[tuple[int, int], np.dtype[np.float32]]
FaceArray = np.ndarray[tuple[int, int], np.dtype[np.uint64]]
Point = np.ndarray[tuple[int,], np.dtype[np.float32]]


class Geometry:
    """Triangular mesh geometry"""

    vertices: np.ndarray
    faces: np.ndarray

    def discretise(self, resolution: float) -> DiscretisedGeometry:
        pass


def closest_point_pair(
    geometry_a: Geometry, geometry_b: Geometry
) -> tuple[Point, Point]:
    pass
