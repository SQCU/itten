"""
PBR rendering with dichromatic reflection model.

Renders 3D meshes with height field textures using:
- Body reflection (warm diffuse)
- Interface reflection (cool specular)
- Fresnel blending
"""

import numpy as np
from typing import Optional, Tuple

# Import canonical implementation from texture.normals
from ..normals import height_to_normals


def render_mesh(
    mesh,
    height_field: np.ndarray,
    normal_map: Optional[np.ndarray] = None,
    output_size: int = 512,
    mode: str = 'dichromatic',
    **kwargs
) -> np.ndarray:
    """
    Render textured mesh.

    Args:
        mesh: Mesh object with vertices, faces, uvs
        height_field: 2D height values for texture
        normal_map: Optional normal map (computed from height if None)
        output_size: Output image size (square)
        mode: Rendering mode ('dichromatic', 'simple')
        **kwargs: Additional rendering parameters

    Returns:
        RGB image as uint8 array (H, W, 3)
    """
    if normal_map is None:
        normal_map = height_to_normals(height_field)

    if mode == 'dichromatic':
        return render_mesh_dichromatic(
            mesh, height_field, normal_map, output_size, **kwargs
        )
    else:
        return render_mesh_simple(
            mesh, height_field, normal_map, output_size, **kwargs
        )


def render_mesh_dichromatic(
    mesh,
    height_field: np.ndarray,
    normal_map: Optional[np.ndarray] = None,
    output_size: int = 512,
    body_color: Tuple[int, int, int] = (200, 120, 80),
    interface_color: Tuple[int, int, int] = (180, 200, 255),
    anisotropy: float = 0.8,
    light_angle: float = 15.0,
    bump_strength: float = 0.7,
    ambient: float = 0.15,
    background_color: Tuple[int, int, int] = (40, 40, 45)
) -> np.ndarray:
    """
    Dichromatic PBR rendering.

    Body reflection: warm, Lambertian diffuse
    Interface reflection: cool, anisotropic specular
    Fresnel blend between body and interface based on viewing angle.

    Args:
        mesh: Mesh object
        height_field: Height texture
        normal_map: Normal map (computed if None)
        output_size: Output image size
        body_color: RGB color for body reflection (warm)
        interface_color: RGB color for interface reflection (cool)
        anisotropy: Specular anisotropy (0-1)
        light_angle: Light elevation angle in degrees
        bump_strength: Normal map influence
        ambient: Ambient light intensity
        background_color: Background RGB

    Returns:
        RGB image as uint8 array
    """
    if normal_map is None:
        normal_map = height_to_normals(height_field, bump_strength)

    # Initialize output image
    img = np.zeros((output_size, output_size, 3), dtype=np.float32)
    depth_buffer = np.full((output_size, output_size), -np.inf)

    # Convert colors to float
    body = np.array(body_color, dtype=np.float32) / 255.0
    interface = np.array(interface_color, dtype=np.float32) / 255.0
    background = np.array(background_color, dtype=np.float32) / 255.0

    # Set background
    img[:, :] = background

    # Light direction (from above-front-left)
    light_elev = np.radians(light_angle)
    light_dir = np.array([
        np.cos(light_elev) * 0.5,
        np.sin(light_elev),
        np.cos(light_elev) * 0.5
    ])
    light_dir = light_dir / np.linalg.norm(light_dir)

    # View direction (looking down -Z)
    view_dir = np.array([0.0, 0.0, 1.0])

    # Get mesh bounds for camera setup
    vertices = mesh.vertices
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    center = (min_bounds + max_bounds) / 2
    extent = (max_bounds - min_bounds).max()

    # Rasterize each triangle
    for face_idx in range(len(mesh.faces)):
        face = mesh.faces[face_idx]
        v0, v1, v2 = vertices[face]
        uv0, uv1, uv2 = mesh.uvs[face]

        # Compute face normal
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        fn_len = np.linalg.norm(face_normal)
        if fn_len < 1e-10:
            continue
        face_normal = face_normal / fn_len

        # Back-face culling
        if np.dot(face_normal, view_dir) < 0:
            continue

        # Project to screen space (orthographic)
        scale = output_size * 0.4 / extent
        offset = np.array([output_size / 2, output_size / 2])

        p0 = ((v0[:2] - center[:2]) * scale + offset).astype(int)
        p1 = ((v1[:2] - center[:2]) * scale + offset).astype(int)
        p2 = ((v2[:2] - center[:2]) * scale + offset).astype(int)

        # Rasterize triangle
        _rasterize_triangle(
            img, depth_buffer, p0, p1, p2,
            v0, v1, v2, uv0, uv1, uv2,
            face_normal, height_field, normal_map,
            light_dir, view_dir,
            body, interface, anisotropy, ambient
        )

    # Convert to uint8
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def render_mesh_simple(
    mesh,
    height_field: np.ndarray,
    normal_map: Optional[np.ndarray] = None,
    output_size: int = 512,
    color: Tuple[int, int, int] = (180, 180, 180),
    ambient: float = 0.2,
    background_color: Tuple[int, int, int] = (40, 40, 45)
) -> np.ndarray:
    """
    Simple Lambertian rendering.

    Args:
        mesh: Mesh object
        height_field: Height texture
        normal_map: Normal map
        output_size: Output image size
        color: Surface RGB color
        ambient: Ambient light intensity
        background_color: Background RGB

    Returns:
        RGB image as uint8 array
    """
    if normal_map is None:
        normal_map = height_to_normals(height_field)

    img = np.zeros((output_size, output_size, 3), dtype=np.float32)
    depth_buffer = np.full((output_size, output_size), -np.inf)

    surface_color = np.array(color, dtype=np.float32) / 255.0
    background = np.array(background_color, dtype=np.float32) / 255.0
    img[:, :] = background

    light_dir = np.array([0.5, 0.7, 0.5])
    light_dir = light_dir / np.linalg.norm(light_dir)
    view_dir = np.array([0.0, 0.0, 1.0])

    vertices = mesh.vertices
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    center = (min_bounds + max_bounds) / 2
    extent = (max_bounds - min_bounds).max()

    for face_idx in range(len(mesh.faces)):
        face = mesh.faces[face_idx]
        v0, v1, v2 = vertices[face]
        uv0, uv1, uv2 = mesh.uvs[face]

        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        fn_len = np.linalg.norm(face_normal)
        if fn_len < 1e-10:
            continue
        face_normal = face_normal / fn_len

        if np.dot(face_normal, view_dir) < 0:
            continue

        scale = output_size * 0.4 / extent
        offset = np.array([output_size / 2, output_size / 2])

        p0 = ((v0[:2] - center[:2]) * scale + offset).astype(int)
        p1 = ((v1[:2] - center[:2]) * scale + offset).astype(int)
        p2 = ((v2[:2] - center[:2]) * scale + offset).astype(int)

        _rasterize_triangle_simple(
            img, depth_buffer, p0, p1, p2,
            v0, v1, v2, uv0, uv1, uv2,
            face_normal, height_field, normal_map,
            light_dir, surface_color, ambient
        )

    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def _rasterize_triangle(
    img, depth_buffer, p0, p1, p2,
    v0, v1, v2, uv0, uv1, uv2,
    face_normal, height_field, normal_map,
    light_dir, view_dir,
    body_color, interface_color, anisotropy, ambient
):
    """Rasterize a single triangle with dichromatic shading."""
    # Bounding box
    min_x = max(0, min(p0[0], p1[0], p2[0]))
    max_x = min(img.shape[1] - 1, max(p0[0], p1[0], p2[0]))
    min_y = max(0, min(p0[1], p1[1], p2[1]))
    max_y = min(img.shape[0] - 1, max(p0[1], p1[1], p2[1]))

    # Edge functions
    def edge_function(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    area = edge_function(p0, p1, p2)
    if abs(area) < 1e-10:
        return

    tex_h, tex_w = height_field.shape

    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            p = np.array([px, py])

            # Barycentric coordinates
            w0 = edge_function(p1, p2, p) / area
            w1 = edge_function(p2, p0, p) / area
            w2 = edge_function(p0, p1, p) / area

            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                # Interpolate depth
                depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]

                if depth > depth_buffer[py, px]:
                    depth_buffer[py, px] = depth

                    # Interpolate UV
                    u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]
                    v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]

                    # Wrap UVs
                    u = u % 1.0
                    v = v % 1.0

                    # Sample textures
                    tx = int(u * (tex_w - 1))
                    ty = int(v * (tex_h - 1))

                    height = height_field[ty, tx]
                    tex_normal = normal_map[ty, tx]

                    # Convert texture normal from [0,1] RGB encoding to [-1,1]
                    tex_normal = tex_normal * 2.0 - 1.0

                    # Perturb face normal with texture normal
                    normal = face_normal + tex_normal * 0.5
                    normal = normal / (np.linalg.norm(normal) + 1e-10)

                    # Diffuse (body reflection)
                    ndotl = max(0, np.dot(normal, light_dir))
                    diffuse = ndotl

                    # Specular (interface reflection)
                    half_vec = light_dir + view_dir
                    half_vec = half_vec / (np.linalg.norm(half_vec) + 1e-10)
                    ndoth = max(0, np.dot(normal, half_vec))
                    specular = ndoth ** (32 * (1 - anisotropy * 0.5))

                    # Fresnel (Schlick approximation)
                    ndotv = max(0.01, np.dot(normal, view_dir))
                    F0 = 0.04
                    fresnel = F0 + (1 - F0) * ((1 - ndotv) ** 5)

                    # Combine
                    color = (
                        (1 - fresnel) * body_color * (diffuse + ambient) +
                        fresnel * interface_color * specular
                    )

                    # Height-based ambient occlusion
                    ao = 0.8 + 0.2 * height
                    color = color * ao

                    img[py, px] = np.clip(color, 0, 1)


def _rasterize_triangle_simple(
    img, depth_buffer, p0, p1, p2,
    v0, v1, v2, uv0, uv1, uv2,
    face_normal, height_field, normal_map,
    light_dir, color, ambient
):
    """Rasterize a single triangle with simple Lambertian shading."""
    min_x = max(0, min(p0[0], p1[0], p2[0]))
    max_x = min(img.shape[1] - 1, max(p0[0], p1[0], p2[0]))
    min_y = max(0, min(p0[1], p1[1], p2[1]))
    max_y = min(img.shape[0] - 1, max(p0[1], p1[1], p2[1]))

    def edge_function(a, b, c):
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    area = edge_function(p0, p1, p2)
    if abs(area) < 1e-10:
        return

    tex_h, tex_w = height_field.shape

    for py in range(min_y, max_y + 1):
        for px in range(min_x, max_x + 1):
            p = np.array([px, py])

            w0 = edge_function(p1, p2, p) / area
            w1 = edge_function(p2, p0, p) / area
            w2 = edge_function(p0, p1, p) / area

            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]

                if depth > depth_buffer[py, px]:
                    depth_buffer[py, px] = depth

                    u = w0 * uv0[0] + w1 * uv1[0] + w2 * uv2[0]
                    v = w0 * uv0[1] + w1 * uv1[1] + w2 * uv2[1]
                    u = u % 1.0
                    v = v % 1.0

                    tx = int(u * (tex_w - 1))
                    ty = int(v * (tex_h - 1))

                    tex_normal = normal_map[ty, tx]
                    # Convert texture normal from [0,1] RGB encoding to [-1,1]
                    tex_normal = tex_normal * 2.0 - 1.0
                    normal = face_normal + tex_normal * 0.3
                    normal = normal / (np.linalg.norm(normal) + 1e-10)

                    ndotl = max(0, np.dot(normal, light_dir))

                    pixel_color = color * (ndotl + ambient)
                    img[py, px] = np.clip(pixel_color, 0, 1)
