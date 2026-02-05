"""
Mesh: Convert lattice to 3D mesh on convex surfaces (egg, torus, sphere).

Maps 2D lattice coordinates to 3D surface points with:
- UV parametrization
- Surface projection
- Geometry placement (triangles, parallelograms)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import numpy as np


@dataclass
class Vertex:
    """A 3D vertex with associated data."""
    position: np.ndarray  # (x, y, z)
    normal: np.ndarray = None  # Surface normal
    uv: Tuple[float, float] = (0.0, 0.0)
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    expansion: float = 0.0  # For visualization


@dataclass
class Face:
    """A triangular face."""
    vertex_indices: Tuple[int, int, int]
    normal: np.ndarray = None


@dataclass
class Mesh:
    """A 3D mesh with vertices and faces."""
    vertices: List[Vertex] = field(default_factory=list)
    faces: List[Face] = field(default_factory=list)


class Surface:
    """Base class for 3D surfaces."""

    def point_at(self, u: float, v: float) -> np.ndarray:
        """Get 3D point at UV coordinates."""
        raise NotImplementedError

    def normal_at(self, u: float, v: float) -> np.ndarray:
        """Get surface normal at UV coordinates."""
        raise NotImplementedError


class EggSurface(Surface):
    """
    Egg-shaped surface (prolate spheroid with taper).

    Parameters:
        radius: Base radius
        aspect: Height/width ratio
        taper: Amount of tapering toward top (0 = sphere, 1 = pointed)
    """

    def __init__(self, radius: float = 1.0, aspect: float = 1.3, taper: float = 0.3):
        self.radius = radius
        self.aspect = aspect
        self.taper = taper

    def point_at(self, u: float, v: float) -> np.ndarray:
        """
        Get 3D point on egg surface.

        u: 0-1, longitude (around the egg)
        v: 0-1, latitude (bottom to top)
        """
        # Spherical coordinates
        theta = u * 2 * np.pi  # Longitude
        phi = v * np.pi  # Latitude (0=top, pi=bottom)

        # Base sphere
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        # Apply aspect ratio
        z = z * self.aspect

        # Apply egg taper (narrower at top)
        taper_factor = 1.0 - self.taper * (1.0 - v)
        x = x * taper_factor
        y = y * taper_factor

        return np.array([x, y, z]) * self.radius

    def normal_at(self, u: float, v: float) -> np.ndarray:
        """Get surface normal at UV coordinates."""
        # Numerical normal via finite differences
        eps = 0.001
        p = self.point_at(u, v)
        pu = self.point_at(u + eps, v)
        pv = self.point_at(u, v + eps)

        du = pu - p
        dv = pv - p

        normal = np.cross(du, dv)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
        return normal


class TorusSurface(Surface):
    """
    Torus surface.

    Parameters:
        major_radius: Distance from center to tube center
        minor_radius: Tube radius
    """

    def __init__(self, major_radius: float = 2.0, minor_radius: float = 0.5):
        self.major_radius = major_radius
        self.minor_radius = minor_radius

    def point_at(self, u: float, v: float) -> np.ndarray:
        """
        Get 3D point on torus.

        u: 0-1, around major circle
        v: 0-1, around tube
        """
        theta = u * 2 * np.pi  # Major angle
        phi = v * 2 * np.pi  # Minor angle

        x = (self.major_radius + self.minor_radius * np.cos(phi)) * np.cos(theta)
        y = (self.major_radius + self.minor_radius * np.cos(phi)) * np.sin(theta)
        z = self.minor_radius * np.sin(phi)

        return np.array([x, y, z])

    def normal_at(self, u: float, v: float) -> np.ndarray:
        """Get surface normal."""
        theta = u * 2 * np.pi
        phi = v * 2 * np.pi

        # Normal points outward from tube
        nx = np.cos(phi) * np.cos(theta)
        ny = np.cos(phi) * np.sin(theta)
        nz = np.sin(phi)

        return np.array([nx, ny, nz])


class SphereSurface(Surface):
    """Simple sphere surface."""

    def __init__(self, radius: float = 1.0):
        self.radius = radius

    def point_at(self, u: float, v: float) -> np.ndarray:
        theta = u * 2 * np.pi
        phi = v * np.pi

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        return np.array([x, y, z]) * self.radius

    def normal_at(self, u: float, v: float) -> np.ndarray:
        return self.point_at(u, v) / self.radius


class LatticeToMesh:
    """
    Convert a 2D lattice to a 3D mesh on a surface.

    Maps lattice coordinates to UV, then UV to 3D points.
    Creates geometry (triangles/quads) at each lattice node.
    """

    def __init__(self, surface: Surface):
        self.surface = surface

    def lattice_to_uv(
        self,
        coord: Tuple[int, int],
        bounds: Tuple[int, int, int, int],
        margin: float = 0.1
    ) -> Tuple[float, float]:
        """
        Map lattice coordinate to UV.

        Args:
            coord: (x, y) lattice position
            bounds: (x_min, y_min, x_max, y_max)
            margin: UV margin at edges

        Returns:
            (u, v) in range [margin, 1-margin]
        """
        x_min, y_min, x_max, y_max = bounds
        x, y = coord

        # Normalize to [0, 1]
        if x_max > x_min:
            u = (x - x_min) / (x_max - x_min)
        else:
            u = 0.5

        if y_max > y_min:
            v = (y - y_min) / (y_max - y_min)
        else:
            v = 0.5

        # Apply margin
        u = margin + u * (1 - 2 * margin)
        v = margin + v * (1 - 2 * margin)

        return (u, v)

    def create_node_geometry(
        self,
        uv: Tuple[float, float],
        size: float,
        expansion: float,
        orientation: float = 0.0,
        geometry_type: str = "square",
        color: Tuple[float, float, float] = None
    ) -> Tuple[List[Vertex], List[Face]]:
        """
        Create geometry for a single lattice node.

        Args:
            uv: Center UV coordinates
            size: Size in UV space
            expansion: Local expansion value (affects color/size)
            orientation: Rotation angle in radians
            geometry_type: "triangle", "parallelogram", or "square"
            color: Override color (otherwise computed from expansion)

        Returns:
            (vertices, faces) for this node
        """
        u, v = uv
        vertices = []
        faces = []

        # Color based on expansion
        if color is None:
            # Blue (low expansion) to Green (high expansion)
            t = min(1.0, expansion / 3.0)
            color = (0.2, 0.3 + 0.5 * t, 0.8 - 0.4 * t)

        # Size scales with expansion
        actual_size = size * (0.5 + 0.5 * min(1.0, expansion / 2.0))

        # Create corner offsets based on geometry type
        cos_o = np.cos(orientation)
        sin_o = np.sin(orientation)

        if geometry_type == "triangle":
            # Equilateral triangle pointing in orientation direction
            offsets = [
                (actual_size * cos_o, actual_size * sin_o),
                (actual_size * np.cos(orientation + 2.1), actual_size * np.sin(orientation + 2.1)),
                (actual_size * np.cos(orientation - 2.1), actual_size * np.sin(orientation - 2.1)),
            ]
        elif geometry_type == "parallelogram":
            # Parallelogram aligned with orientation
            hw = actual_size * 0.8
            hh = actual_size * 0.4
            offsets = [
                (hw * cos_o - hh * sin_o, hw * sin_o + hh * cos_o),
                (-hw * cos_o - hh * sin_o, -hw * sin_o + hh * cos_o),
                (-hw * cos_o + hh * sin_o, -hw * sin_o - hh * cos_o),
                (hw * cos_o + hh * sin_o, hw * sin_o - hh * cos_o),
            ]
        else:  # square
            hw = actual_size * 0.5
            offsets = [
                (hw * cos_o - hw * sin_o, hw * sin_o + hw * cos_o),
                (-hw * cos_o - hw * sin_o, -hw * sin_o + hw * cos_o),
                (-hw * cos_o + hw * sin_o, -hw * sin_o - hw * cos_o),
                (hw * cos_o + hw * sin_o, hw * sin_o - hw * cos_o),
            ]

        # Create vertices
        for du, dv in offsets:
            vert_u = u + du
            vert_v = v + dv
            pos = self.surface.point_at(vert_u, vert_v)
            normal = self.surface.normal_at(vert_u, vert_v)
            vertex = Vertex(
                position=pos,
                normal=normal,
                uv=(vert_u, vert_v),
                color=color,
                expansion=expansion
            )
            vertices.append(vertex)

        # Create faces
        n = len(offsets)
        if n == 3:
            faces.append(Face(vertex_indices=(0, 1, 2)))
        elif n == 4:
            faces.append(Face(vertex_indices=(0, 1, 2)))
            faces.append(Face(vertex_indices=(0, 2, 3)))

        return vertices, faces

    def convert_lattice(
        self,
        nodes: Dict[Tuple[int, int], 'ExtrudedNode'],
        bounds: Tuple[int, int, int, int],
        node_size: float = 0.02,
        use_fiedler_orientation: bool = True
    ) -> Mesh:
        """
        Convert entire lattice to mesh.

        Args:
            nodes: Dict of coord -> ExtrudedNode
            bounds: Lattice bounds
            node_size: Base size of each node's geometry
            use_fiedler_orientation: Orient geometry by Fiedler gradient

        Returns:
            Complete mesh
        """
        mesh = Mesh()

        for coord, node in nodes.items():
            uv = self.lattice_to_uv(coord, bounds)

            # Use Fiedler gradient for orientation if available
            if use_fiedler_orientation and node.fiedler_gradient != (0.0, 0.0):
                orientation = np.arctan2(node.fiedler_gradient[1], node.fiedler_gradient[0])
            else:
                orientation = node.geometry_angle

            # Geometry type based on expansion
            if node.expansion < 0.5:
                geometry_type = "triangle"
            elif node.expansion < 1.5:
                geometry_type = "parallelogram"
            else:
                geometry_type = "square"

            # Create geometry for this node
            base_vertex_idx = len(mesh.vertices)
            vertices, faces = self.create_node_geometry(
                uv=uv,
                size=node_size,
                expansion=node.expansion,
                orientation=orientation,
                geometry_type=geometry_type
            )

            # Add to mesh with adjusted indices
            mesh.vertices.extend(vertices)
            for face in faces:
                adjusted_indices = tuple(i + base_vertex_idx for i in face.vertex_indices)
                mesh.faces.append(Face(vertex_indices=adjusted_indices, normal=face.normal))

        return mesh


def create_egg_mesh_from_territory(
    territory: 'TerritoryGraph',
    extrusion_state: 'ExtrusionState',
    node_size: float = 0.015
) -> Mesh:
    """
    Convenience function to create egg mesh from territory and extrusion state.
    """
    surface = EggSurface(radius=1.0, aspect=1.3, taper=0.3)
    converter = LatticeToMesh(surface)
    bounds = territory.get_bounds()

    return converter.convert_lattice(
        nodes=extrusion_state.nodes,
        bounds=bounds,
        node_size=node_size,
        use_fiedler_orientation=True
    )
