"""
UV mapping and tangent frame utilities for texture-mesh operations.
"""

import numpy as np
from typing import Tuple
from ..geometry.mesh import Mesh


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector(s) along last axis."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm > 1e-10, v / norm, v)


def sample_bilinear(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """
    Bilinear texture sampling with wrapping.

    Args:
        texture: (H, W) or (H, W, C) texture array
        uv: (2,) or (..., 2) UV coordinates in [0, 1]

    Returns:
        Sampled values
    """
    h, w = texture.shape[:2]
    uv = np.asarray(uv)

    # Handle scalar UV
    if uv.ndim == 1:
        uv = uv.reshape(1, 2)
        squeeze = True
    else:
        squeeze = False

    # Wrap UVs to [0, 1)
    u = uv[..., 0] % 1.0
    v = uv[..., 1] % 1.0

    # Convert to pixel coordinates
    x = u * w - 0.5
    y = v * h - 0.5

    x0 = np.floor(x).astype(np.int32) % w
    y0 = np.floor(y).astype(np.int32) % h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h

    # Fractional parts
    fx = x - np.floor(x)
    fy = y - np.floor(y)

    # Sample four corners
    if texture.ndim == 2:
        v00 = texture[y0, x0]
        v01 = texture[y0, x1]
        v10 = texture[y1, x0]
        v11 = texture[y1, x1]
    else:
        v00 = texture[y0, x0, :]
        v01 = texture[y0, x1, :]
        v10 = texture[y1, x0, :]
        v11 = texture[y1, x1, :]
        fx = fx[..., np.newaxis]
        fy = fy[..., np.newaxis]

    result = (v00 * (1 - fx) * (1 - fy) +
              v01 * fx * (1 - fy) +
              v10 * (1 - fx) * fy +
              v11 * fx * fy)

    if squeeze:
        return result[0]
    return result


def apply_texture_to_mesh(mesh: Mesh, height_field: np.ndarray) -> Mesh:
    """
    Store height values as vertex attributes for rendering.

    Samples the height field at each vertex's UV coordinate and
    stores the result in mesh.vertex_heights.

    Args:
        mesh: Mesh with UV coordinates
        height_field: (H, W) height texture

    Returns:
        Same mesh with vertex_heights attribute added
    """
    # Sample height at each vertex's UV
    heights = np.array([
        sample_bilinear(height_field, uv) for uv in mesh.uvs
    ], dtype=np.float64)

    mesh.vertex_heights = heights
    return mesh


def compute_vertex_normals(mesh: Mesh) -> np.ndarray:
    """
    Compute per-vertex normals by averaging adjacent face normals.

    Args:
        mesh: Mesh with vertices and faces

    Returns:
        (N, 3) array of vertex normals
    """
    vertex_normals = np.zeros_like(mesh.vertices)

    # Compute face normals and accumulate to vertices
    for face in mesh.faces:
        v0, v1, v2 = mesh.vertices[face]
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)

        # Weight by area (face_normal magnitude is 2x area)
        vertex_normals[face[0]] += face_normal
        vertex_normals[face[1]] += face_normal
        vertex_normals[face[2]] += face_normal

    # Normalize
    return normalize(vertex_normals)


def project_to_tangent_plane(normal: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Project a vector onto the tangent plane defined by normal.

    Args:
        normal: (3,) surface normal (must be normalized)
        vector: (3,) vector to project

    Returns:
        (3,) projected vector (NOT normalized)
    """
    normal = np.asarray(normal)
    vector = np.asarray(vector)

    # Remove component along normal
    proj = vector - np.dot(vector, normal) * normal

    return proj


def compute_tangent_frame(mesh: Mesh, height_field: np.ndarray) -> Mesh:
    """
    Compute per-vertex tangent frames from texture gradient.

    The tangent direction follows the height gradient, which is
    crucial for anisotropic specular to elongate along bump features.

    Args:
        mesh: Mesh with vertices, faces, uvs
        height_field: (H, W) height texture

    Returns:
        Same mesh with tangents and vertex_normals attributes added
    """
    h, w = height_field.shape

    # First compute vertex normals
    vertex_normals = compute_vertex_normals(mesh)

    # Compute tangents from height gradient
    tangents = []
    du = 1.0 / w
    dv = 1.0 / h

    for i, uv in enumerate(mesh.uvs):
        # Sample height gradient at this UV
        u, v = uv

        # Central differences
        h_u_plus = sample_bilinear(height_field, np.array([u + du, v]))
        h_u_minus = sample_bilinear(height_field, np.array([u - du, v]))
        h_v_plus = sample_bilinear(height_field, np.array([u, v + dv]))
        h_v_minus = sample_bilinear(height_field, np.array([u, v - dv]))

        grad_u = (h_u_plus - h_u_minus) / (2 * du)
        grad_v = (h_v_plus - h_v_minus) / (2 * dv)

        # Convert gradient to 3D tangent
        # In UV space, gradient (grad_u, grad_v) points uphill
        # Map UV to XY for tangent estimation
        tangent_3d = np.array([grad_u, grad_v, 0.0], dtype=np.float64)

        # If gradient is too small, use default tangent
        grad_mag = np.linalg.norm(tangent_3d)
        if grad_mag < 1e-6:
            tangent_3d = np.array([1.0, 0.0, 0.0])

        # Project to tangent plane
        normal = vertex_normals[i]
        tangent = project_to_tangent_plane(normal, tangent_3d)

        # Normalize
        tangent_mag = np.linalg.norm(tangent)
        if tangent_mag > 1e-6:
            tangent = tangent / tangent_mag
        else:
            # Fallback: arbitrary perpendicular
            if abs(normal[0]) < 0.9:
                tangent = normalize(np.cross(normal, np.array([1, 0, 0])))
            else:
                tangent = normalize(np.cross(normal, np.array([0, 1, 0])))

        tangents.append(tangent)

    mesh.tangents = np.array(tangents, dtype=np.float64)
    mesh.vertex_normals = vertex_normals

    return mesh


def compute_tangent_from_uv(
    mesh: Mesh,
    face_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute tangent and bitangent for a face from UV mapping.

    Uses the UV-to-position mapping to determine how texture
    coordinates relate to surface directions.

    Args:
        mesh: Mesh with vertices, faces, uvs
        face_idx: Index of face to compute for

    Returns:
        (tangent, bitangent) as (3,) arrays
    """
    face = mesh.faces[face_idx]

    # Get vertices and UVs
    v0, v1, v2 = mesh.vertices[face]
    uv0, uv1, uv2 = mesh.uvs[face]

    # Position deltas
    dp1 = v1 - v0
    dp2 = v2 - v0

    # UV deltas
    duv1 = uv1 - uv0
    duv2 = uv2 - uv0

    # Solve for tangent and bitangent
    # [dp1] = [duv1.u  duv1.v] * [T]
    # [dp2]   [duv2.u  duv2.v]   [B]

    det = duv1[0] * duv2[1] - duv1[1] * duv2[0]

    if abs(det) < 1e-10:
        # Degenerate UV mapping, use geometric fallback
        tangent = normalize(dp1)
        bitangent = normalize(np.cross(np.cross(dp1, dp2), tangent))
    else:
        r = 1.0 / det
        tangent = r * (duv2[1] * dp1 - duv1[1] * dp2)
        bitangent = r * (-duv2[0] * dp1 + duv1[0] * dp2)
        tangent = normalize(tangent)
        bitangent = normalize(bitangent)

    return tangent, bitangent


def unwrap_spherical(vertices: np.ndarray) -> np.ndarray:
    """
    Generate spherical UV coordinates for vertices.

    Uses Y-up convention matching primitives.py:spherical_uv().

    Args:
        vertices: (N, 3) vertex positions (assumed roughly spherical)

    Returns:
        (N, 2) UV coordinates
    """
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    unit = vertices / np.maximum(norms, 1e-10)

    # Y-up convention (matches primitives.py)
    theta = np.arctan2(unit[:, 2], unit[:, 0])  # Changed from atan2(y, x)
    phi = np.arcsin(np.clip(unit[:, 1], -1, 1))  # Changed from acos(z)

    u = (theta / (2 * np.pi) + 0.5) % 1.0
    v = phi / np.pi + 0.5

    return np.column_stack([u, v])


def unwrap_cylindrical(
    vertices: np.ndarray,
    axis: str = 'z'
) -> np.ndarray:
    """
    Generate cylindrical UV coordinates for vertices.

    Args:
        vertices: (N, 3) vertex positions
        axis: Cylinder axis ('x', 'y', or 'z')

    Returns:
        (N, 2) UV coordinates
    """
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
    other = [i for i in range(3) if i != axis_idx]

    # U from angle around axis
    theta = np.arctan2(vertices[:, other[1]], vertices[:, other[0]])
    u = (theta + np.pi) / (2 * np.pi)

    # V from position along axis
    z = vertices[:, axis_idx]
    z_min, z_max = z.min(), z.max()
    if z_max - z_min > 1e-6:
        v = (z - z_min) / (z_max - z_min)
    else:
        v = np.full(len(z), 0.5)

    return np.stack([u, v], axis=1)


def create_uv_grid(
    width: int,
    height: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a UV-aligned planar grid mesh.

    Args:
        width: Number of vertices in U direction
        height: Number of vertices in V direction

    Returns:
        (vertices, faces, uvs) tuple
    """
    # Create UV coordinates
    u = np.linspace(0, 1, width)
    v = np.linspace(0, 1, height)
    uu, vv = np.meshgrid(u, v)

    # Flatten to create vertex arrays
    uvs = np.stack([uu.flatten(), vv.flatten()], axis=1)

    # Vertices: map UV to XY plane, Z=0
    vertices = np.zeros((width * height, 3), dtype=np.float64)
    vertices[:, 0] = uvs[:, 0] - 0.5
    vertices[:, 1] = uvs[:, 1] - 0.5

    # Create faces (two triangles per cell)
    faces = []
    for j in range(height - 1):
        for i in range(width - 1):
            # Quad corners
            v00 = j * width + i
            v01 = j * width + (i + 1)
            v10 = (j + 1) * width + i
            v11 = (j + 1) * width + (i + 1)

            # Two triangles
            faces.append([v00, v01, v10])
            faces.append([v01, v11, v10])

    return vertices, np.array(faces, dtype=np.int32), uvs
