"""
Geometric primitives with proper UV mapping.
"""

import numpy as np
from .mesh import Mesh


def spherical_uv(vertices: np.ndarray) -> np.ndarray:
    """
    Map 3D vertices to UV using spherical projection.

    Args:
        vertices: (N, 3) array of vertex positions

    Returns:
        (N, 2) array of UV coordinates in [0, 1]
    """
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    unit = vertices / norms

    # theta = atan2(z, x), phi = asin(y)
    theta = np.arctan2(unit[:, 2], unit[:, 0])
    phi = np.arcsin(np.clip(unit[:, 1], -1, 1))

    u = (theta / (2 * np.pi) + 0.5) % 1.0
    v = phi / np.pi + 0.5

    return np.column_stack([u, v])


def Icosahedron(radius: float = 1.0, subdivisions: int = 0) -> Mesh:
    """
    Create an icosahedron mesh with spherical UV mapping.

    Args:
        radius: Radius of circumscribed sphere
        subdivisions: Number of subdivision iterations

    Returns:
        Mesh object
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2

    # 12 vertices of unit icosahedron
    vertices = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float64)

    # Normalize to unit sphere then scale
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms * radius

    # 20 triangular faces
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)

    # Apply subdivisions
    for _ in range(subdivisions):
        vertices, faces = _subdivide(vertices, faces, radius)

    # Compute spherical UVs
    uvs = spherical_uv(vertices)

    return Mesh(vertices=vertices, faces=faces, uvs=uvs)


def _subdivide(vertices: np.ndarray, faces: np.ndarray, radius: float) -> tuple:
    """
    Subdivide mesh by splitting each triangle into 4.

    Args:
        vertices: Current vertices
        faces: Current faces
        radius: Sphere radius for projection

    Returns:
        (new_vertices, new_faces)
    """
    edge_midpoints = {}
    new_vertices = list(vertices)
    new_faces = []

    def get_midpoint(v1_idx, v2_idx):
        """Get or create midpoint vertex."""
        edge = tuple(sorted([v1_idx, v2_idx]))
        if edge in edge_midpoints:
            return edge_midpoints[edge]

        # Create midpoint
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]
        mid = (v1 + v2) / 2

        # Project to sphere
        mid = mid / np.linalg.norm(mid) * radius

        new_idx = len(new_vertices)
        new_vertices.append(mid)
        edge_midpoints[edge] = new_idx
        return new_idx

    for face in faces:
        v0, v1, v2 = face

        # Get midpoints
        m01 = get_midpoint(v0, v1)
        m12 = get_midpoint(v1, v2)
        m20 = get_midpoint(v2, v0)

        # Create 4 new faces
        new_faces.append([v0, m01, m20])
        new_faces.append([v1, m12, m01])
        new_faces.append([v2, m20, m12])
        new_faces.append([m01, m12, m20])

    return np.array(new_vertices), np.array(new_faces, dtype=np.int32)


def Sphere(radius: float = 1.0, segments: int = 32) -> Mesh:
    """
    Create a UV sphere mesh.

    Args:
        radius: Sphere radius
        segments: Number of latitude/longitude divisions

    Returns:
        Mesh object
    """
    vertices = []
    faces = []
    uvs = []

    # Generate vertices
    for lat in range(segments + 1):
        theta = lat * np.pi / segments
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for lon in range(segments + 1):
            phi = lon * 2 * np.pi / segments

            x = radius * sin_theta * np.cos(phi)
            y = radius * cos_theta
            z = radius * sin_theta * np.sin(phi)

            vertices.append([x, y, z])
            uvs.append([lon / segments, 1 - lat / segments])

    vertices = np.array(vertices, dtype=np.float64)
    uvs = np.array(uvs, dtype=np.float64)

    # Generate faces
    for lat in range(segments):
        for lon in range(segments):
            v0 = lat * (segments + 1) + lon
            v1 = v0 + 1
            v2 = v0 + segments + 1
            v3 = v2 + 1

            if lat != 0:
                faces.append([v0, v2, v1])
            if lat != segments - 1:
                faces.append([v1, v2, v3])

    faces = np.array(faces, dtype=np.int32)

    return Mesh(vertices=vertices, faces=faces, uvs=uvs)


def Egg(radius: float = 1.0, pointiness: float = 0.25, segments: int = 32) -> Mesh:
    """
    Create an egg-shaped mesh.

    The egg is elongated along the Y axis with one end pointier.

    Args:
        radius: Base radius
        pointiness: How much to elongate the pointed end (0-1)
        segments: Number of latitude/longitude divisions

    Returns:
        Mesh object
    """
    vertices = []
    faces = []
    uvs = []

    # Generate vertices with egg deformation
    for lat in range(segments + 1):
        t = lat / segments  # 0 at top, 1 at bottom
        theta = t * np.pi

        # Egg deformation: more pointy at top (t=0)
        deform = 1.0 + pointiness * (1 - t) ** 2

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for lon in range(segments + 1):
            phi = lon * 2 * np.pi / segments

            x = radius * sin_theta * np.cos(phi)
            y = radius * cos_theta * deform
            z = radius * sin_theta * np.sin(phi)

            vertices.append([x, y, z])
            uvs.append([lon / segments, 1 - lat / segments])

    vertices = np.array(vertices, dtype=np.float64)
    uvs = np.array(uvs, dtype=np.float64)

    # Generate faces (same as sphere)
    for lat in range(segments):
        for lon in range(segments):
            v0 = lat * (segments + 1) + lon
            v1 = v0 + 1
            v2 = v0 + segments + 1
            v3 = v2 + 1

            if lat != 0:
                faces.append([v0, v2, v1])
            if lat != segments - 1:
                faces.append([v1, v2, v3])

    faces = np.array(faces, dtype=np.int32)

    return Mesh(vertices=vertices, faces=faces, uvs=uvs)
