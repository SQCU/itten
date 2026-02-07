"""Core mesh data structure for 3D geometry."""

from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np


@dataclass
class Mesh:
    """
    3D mesh data structure with vertices, faces, and UV coordinates.

    Attributes:
        vertices: (N, 3) array of 3D vertex positions
        faces: (M, 3) array of triangle indices
        uvs: (N, 2) array of UV coordinates per vertex
    """
    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray     # (M, 3) triangle indices
    uvs: np.ndarray       # (N, 2) UV coordinates

    def __post_init__(self):
        """Ensure arrays are proper numpy arrays with correct dtypes."""
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int32)
        self.uvs = np.asarray(self.uvs, dtype=np.float64)

    def translate(self, x: float, y: float, z: float) -> 'Mesh':
        """
        Translate mesh by offset (x, y, z).

        Args:
            x: Translation along X axis
            y: Translation along Y axis
            z: Translation along Z axis

        Returns:
            New Mesh with translated vertices
        """
        offset = np.array([x, y, z], dtype=np.float64)
        return Mesh(
            vertices=self.vertices + offset,
            faces=self.faces.copy(),
            uvs=self.uvs.copy()
        )

    def rotate(self, axis: Union[str, Tuple[float, float, float]], angle: float) -> 'Mesh':
        """
        Rotate mesh around axis by angle (in radians).

        Args:
            axis: Either 'x', 'y', 'z' or a tuple (ax, ay, az) for arbitrary axis
            angle: Rotation angle in radians

        Returns:
            New Mesh with rotated vertices
        """
        # Get axis vector
        if isinstance(axis, str):
            axis_map = {
                'x': np.array([1.0, 0.0, 0.0]),
                'y': np.array([0.0, 1.0, 0.0]),
                'z': np.array([0.0, 0.0, 1.0])
            }
            axis_vec = axis_map[axis.lower()]
        else:
            axis_vec = np.array(axis, dtype=np.float64)
            axis_vec = axis_vec / np.linalg.norm(axis_vec)

        # Rodrigues' rotation formula
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Rotation matrix using Rodrigues' formula
        K = np.array([
            [0, -axis_vec[2], axis_vec[1]],
            [axis_vec[2], 0, -axis_vec[0]],
            [-axis_vec[1], axis_vec[0], 0]
        ])

        R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

        new_vertices = (R @ self.vertices.T).T

        return Mesh(
            vertices=new_vertices,
            faces=self.faces.copy(),
            uvs=self.uvs.copy()
        )

    def scale(self, sx: float, sy: float = None, sz: float = None) -> 'Mesh':
        """
        Scale mesh by factors (sx, sy, sz).

        If only sx is provided, uniform scaling is applied.

        Args:
            sx: Scale factor for X axis (or uniform if sy, sz not provided)
            sy: Scale factor for Y axis (optional)
            sz: Scale factor for Z axis (optional)

        Returns:
            New Mesh with scaled vertices
        """
        if sy is None:
            sy = sx
        if sz is None:
            sz = sx

        scale_vec = np.array([sx, sy, sz], dtype=np.float64)

        return Mesh(
            vertices=self.vertices * scale_vec,
            faces=self.faces.copy(),
            uvs=self.uvs.copy()
        )

    def copy(self) -> 'Mesh':
        """
        Create a deep copy of the mesh.

        Returns:
            New Mesh with copied data
        """
        return Mesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            uvs=self.uvs.copy()
        )

    @property
    def num_vertices(self) -> int:
        """Number of vertices in the mesh."""
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        """Number of faces (triangles) in the mesh."""
        return len(self.faces)

    def __repr__(self) -> str:
        return f"Mesh(vertices={self.num_vertices}, faces={self.num_faces})"
