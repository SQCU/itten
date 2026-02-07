"""
Mesh operations: fuse, chop, squash, etc.
"""

import numpy as np
from .mesh import Mesh
from .primitives import spherical_uv


def fuse(*meshes: Mesh) -> Mesh:
    """
    Combine multiple meshes by union of vertices and faces.

    This is a simple concatenation - for true boolean union,
    use boolean_union instead.

    Args:
        *meshes: Mesh objects to combine

    Returns:
        Combined Mesh
    """
    if not meshes:
        raise ValueError("At least one mesh required")

    if len(meshes) == 1:
        return meshes[0].copy()

    all_vertices = []
    all_faces = []
    all_uvs = []
    vertex_offset = 0

    for mesh in meshes:
        all_vertices.append(mesh.vertices)
        all_faces.append(mesh.faces + vertex_offset)
        all_uvs.append(mesh.uvs)
        vertex_offset += len(mesh.vertices)

    return Mesh(
        vertices=np.vstack(all_vertices),
        faces=np.vstack(all_faces),
        uvs=np.vstack(all_uvs)
    )


def chop(mesh: Mesh, plane_normal: tuple, plane_origin: tuple) -> Mesh:
    """
    Cut mesh by plane, keeping vertices above the plane.

    Simplified version: keeps faces where ALL vertices are above plane.
    For exact intersection, would need to split triangles at plane.

    Args:
        mesh: Input mesh
        plane_normal: Normal vector of cutting plane (points toward kept half)
        plane_origin: A point on the cutting plane

    Returns:
        Chopped Mesh
    """
    normal = np.array(plane_normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    origin = np.array(plane_origin, dtype=np.float64)

    # Signed distance from plane for each vertex
    # Positive means above plane (in direction of normal)
    distances = np.dot(mesh.vertices - origin, normal)

    # Keep vertices above plane (or on it)
    keep_vertex_mask = distances >= -1e-10

    # Keep faces where all 3 vertices are kept
    face_keep_mask = keep_vertex_mask[mesh.faces].all(axis=1)

    if not face_keep_mask.any():
        # No faces remaining - return empty mesh
        return Mesh(
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int32),
            uvs=np.zeros((0, 2))
        )

    # Build mapping from old indices to new indices
    # -1 means vertex is removed
    new_indices = np.full(len(mesh.vertices), -1, dtype=np.int32)
    new_indices[keep_vertex_mask] = np.arange(keep_vertex_mask.sum())

    # Filter vertices and UVs
    new_vertices = mesh.vertices[keep_vertex_mask]
    new_uvs = mesh.uvs[keep_vertex_mask]

    # Remap faces
    kept_faces = mesh.faces[face_keep_mask]
    new_faces = new_indices[kept_faces]

    return Mesh(vertices=new_vertices, faces=new_faces, uvs=new_uvs)


def squash(mesh: Mesh, axis: str = 'y', factor: float = 0.7) -> Mesh:
    """
    Scale mesh along an axis.

    factor=0.7 means 30% shorter along that axis.

    Args:
        mesh: Input mesh
        axis: 'x', 'y', or 'z'
        factor: Scale factor (1.0 = no change)

    Returns:
        Squashed Mesh
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis.lower() not in axis_map:
        raise ValueError(f"Invalid axis: {axis}")

    axis_idx = axis_map[axis.lower()]

    # Create scale factors
    scale = np.ones(3)
    scale[axis_idx] = factor

    return mesh.scale(scale[0], scale[1], scale[2])


def stretch(mesh: Mesh, axis: str = 'x', factor: float = 2.0) -> Mesh:
    """
    Stretch mesh along an axis.

    factor=2.0 means twice as long along that axis.

    Args:
        mesh: Input mesh
        axis: 'x', 'y', or 'z'
        factor: Scale factor (1.0 = no change)

    Returns:
        Stretched Mesh
    """
    return squash(mesh, axis, factor)


def translate(mesh: Mesh, x: float = 0, y: float = 0, z: float = 0) -> Mesh:
    """
    Translate mesh by offset.

    Args:
        mesh: Input mesh
        x, y, z: Translation offsets

    Returns:
        Translated Mesh
    """
    return mesh.translate(x, y, z)


def rotate(mesh: Mesh, axis: str = 'y', angle_degrees: float = 45.0) -> Mesh:
    """
    Rotate mesh around axis.

    Args:
        mesh: Input mesh
        axis: 'x', 'y', or 'z'
        angle_degrees: Rotation angle in degrees

    Returns:
        Rotated Mesh
    """
    angle_radians = np.radians(angle_degrees)
    return mesh.rotate(axis, angle_radians)


def center(mesh: Mesh) -> Mesh:
    """
    Center mesh at origin.

    Args:
        mesh: Input mesh

    Returns:
        Centered Mesh
    """
    centroid = mesh.vertices.mean(axis=0)
    return mesh.translate(-centroid[0], -centroid[1], -centroid[2])


def normalize_scale(mesh: Mesh, target_size: float = 2.0) -> Mesh:
    """
    Scale mesh so its bounding box fits within target_size.

    Args:
        mesh: Input mesh
        target_size: Maximum extent in any dimension

    Returns:
        Scaled Mesh
    """
    extents = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    max_extent = extents.max()

    if max_extent < 1e-10:
        return mesh.copy()

    scale_factor = target_size / max_extent
    return mesh.scale(scale_factor)
