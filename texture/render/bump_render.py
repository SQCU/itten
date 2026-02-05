"""
Bump and normal map rendering for egg surfaces.

Implements true bump mapping where:
- Height field displaces surface geometry along the surface normal
- Normal map perturbs lighting normals for shading calculations
- Result shows BOTH surface curvature AND embossed texture detail

This is different from color mapping - the texture actually affects
the 3D geometry and lighting, not just the surface color.
"""

import numpy as np
from typing import Tuple, Optional
from ..normals import height_to_normals
from .lighting import lambertian_diffuse, blinn_phong_specular, perturb_normal, compute_tbn_frame
from .ray_cast import ray_sphere_test, egg_deformation, spherical_uv_from_ray


def sample_texture_bilinear(texture: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Bilinear texture sampling with wrapping.

    Args:
        texture: (H, W) or (H, W, C) texture array
        u, v: UV coordinates in [0, 1], shape matching desired output

    Returns:
        Sampled values with shape of u/v (plus channel dim if texture has channels)
    """
    h, w = texture.shape[:2]

    # Wrap UVs to [0, 1)
    u = u % 1.0
    v = v % 1.0

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

    return result


def compute_egg_surface_normal(nx: np.ndarray, ny: np.ndarray, z: np.ndarray,
                                egg_factor: float = 0.25) -> np.ndarray:
    """
    Compute surface normal for egg geometry.

    The egg is a deformed sphere. This computes the analytical normal
    at each surface point accounting for the egg deformation.

    Args:
        nx, ny: Normalized screen coordinates
        z: Z depth on sphere surface
        egg_factor: Egg deformation amount

    Returns:
        (H, W, 3) surface normals pointing outward
    """
    # The egg deformation: egg_mod = 1 - egg_factor * ny
    # x_egg = nx / egg_mod, z_egg = z / egg_mod
    # We need the gradient of the implicit surface F(x,y,z) = 0

    # For egg shape: the surface is defined by
    # (x * egg_mod)^2 + y^2 + (z * egg_mod)^2 = 1
    # where egg_mod = 1 - egg_factor * y

    egg_mod = 1.0 - egg_factor * ny

    # For the deformed egg, the normal is approximately:
    # We can compute it from the sphere normal with egg correction
    surf_normal = np.stack([nx * egg_mod, ny, z * egg_mod], axis=-1)

    # Normalize
    surf_len = np.linalg.norm(surf_normal, axis=-1, keepdims=True)
    surf_len = np.where(surf_len > 1e-6, surf_len, 1.0)
    surf_normal = surf_normal / surf_len

    return surf_normal


def compute_displaced_normal(
    base_normal: np.ndarray,
    height_field: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    displacement_scale: float = 0.1
) -> np.ndarray:
    """
    Compute perturbed normal from height field displacement.

    This simulates what the normal WOULD be if the surface were
    displaced by the height field. Uses finite differences on
    the height field to compute the tangent-space perturbation.

    Args:
        base_normal: (H, W, 3) base surface normals
        height_field: (tex_h, tex_w) height values [0, 1]
        u, v: UV coordinates for each pixel
        displacement_scale: Multiplier for height perturbation

    Returns:
        (H, W, 3) perturbed normals
    """
    h, w = height_field.shape

    # Compute height gradient at each UV position using finite differences
    du = 1.0 / w
    dv = 1.0 / h

    # Sample heights at offset positions
    h_center = sample_texture_bilinear(height_field, u, v)
    h_u_plus = sample_texture_bilinear(height_field, u + du, v)
    h_u_minus = sample_texture_bilinear(height_field, u - du, v)
    h_v_plus = sample_texture_bilinear(height_field, u, v + dv)
    h_v_minus = sample_texture_bilinear(height_field, u, v - dv)

    # Central differences for gradient
    dh_du = (h_u_plus - h_u_minus) / (2 * du)
    dh_dv = (h_v_plus - h_v_minus) / (2 * dv)

    # Scale by displacement amount
    dh_du = dh_du * displacement_scale
    dh_dv = dh_dv * displacement_scale

    # Create tangent-space perturbation
    # The tangent space normal is (dh_du, dh_dv, 1) normalized
    # But we need to transform this to world space

    # For spherical surfaces, we compute approximate tangent and bitangent
    # Tangent points in the direction of increasing U (around the sphere)
    # Bitangent points in the direction of increasing V (up/down the sphere)

    # Compute tangent and bitangent from base normal
    # Use the "up" vector as a reference to compute tangent
    up = np.zeros_like(base_normal)
    up[..., 1] = 1.0  # Y-up

    # Handle poles where up is parallel to normal
    parallel_mask = np.abs(np.sum(base_normal * up, axis=-1)) > 0.99
    right = np.zeros_like(base_normal)
    right[..., 0] = 1.0  # X-right

    # Tangent = normalize(up - (up.n)*n) for non-polar regions
    # At poles, use right vector instead
    ref = np.where(parallel_mask[..., np.newaxis], right, up)

    # Project reference onto tangent plane
    tangent = ref - np.sum(ref * base_normal, axis=-1, keepdims=True) * base_normal
    tangent_len = np.linalg.norm(tangent, axis=-1, keepdims=True)
    tangent = np.where(tangent_len > 1e-6, tangent / tangent_len, tangent)

    # Bitangent = normal x tangent
    bitangent = np.cross(base_normal, tangent)

    # Perturb normal: n' = normalize(n - dh_du * tangent - dh_dv * bitangent)
    perturbed = (base_normal
                 - dh_du[..., np.newaxis] * tangent
                 - dh_dv[..., np.newaxis] * bitangent)

    # Normalize
    perturbed_len = np.linalg.norm(perturbed, axis=-1, keepdims=True)
    perturbed = np.where(perturbed_len > 1e-6, perturbed / perturbed_len, base_normal)

    return perturbed


def apply_normal_map_perturbation(
    base_normal: np.ndarray,
    normal_map: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    strength: float = 1.0
) -> np.ndarray:
    """
    Perturb base normals using a tangent-space normal map.

    This is separate from displacement - it directly modifies the
    lighting normal based on the texture's encoded normals.

    Args:
        base_normal: (H, W, 3) base surface normals
        normal_map: (tex_h, tex_w, 3) RGB-encoded normals [0, 1]
        u, v: UV coordinates
        strength: Perturbation strength [0, 1]

    Returns:
        (H, W, 3) perturbed normals
    """
    # Sample normal map
    tex_normal = sample_texture_bilinear(normal_map, u, v)

    # Convert from [0, 1] RGB encoding to [-1, 1] tangent space
    # R -> X (tangent), G -> Y (bitangent), B -> Z (normal)
    tex_normal = tex_normal * 2.0 - 1.0

    # Compute tangent frame from base normal
    up = np.zeros_like(base_normal)
    up[..., 1] = 1.0

    parallel_mask = np.abs(np.sum(base_normal * up, axis=-1)) > 0.99
    right = np.zeros_like(base_normal)
    right[..., 0] = 1.0

    ref = np.where(parallel_mask[..., np.newaxis], right, up)

    tangent = ref - np.sum(ref * base_normal, axis=-1, keepdims=True) * base_normal
    tangent_len = np.linalg.norm(tangent, axis=-1, keepdims=True)
    tangent = np.where(tangent_len > 1e-6, tangent / tangent_len, tangent)

    bitangent = np.cross(base_normal, tangent)

    # Transform texture normal from tangent space to world space
    # world_normal = T * tex_normal.x + B * tex_normal.y + N * tex_normal.z
    world_normal = (
        tangent * tex_normal[..., 0:1] +
        bitangent * tex_normal[..., 1:2] +
        base_normal * tex_normal[..., 2:3]
    )

    # Blend between base and perturbed based on strength
    result = base_normal * (1 - strength) + world_normal * strength

    # Normalize
    result_len = np.linalg.norm(result, axis=-1, keepdims=True)
    result = np.where(result_len > 1e-6, result / result_len, base_normal)

    return result


def render_bumped_egg(
    height_field: np.ndarray,
    normal_map: Optional[np.ndarray] = None,
    displacement_scale: float = 0.1,
    normal_strength: float = 0.7,
    output_size: int = 512,
    egg_factor: float = 0.25,
    light_dir: Tuple[float, float, float] = (0.5, 0.7, 0.5),
    body_color: Tuple[float, float, float] = (0.78, 0.47, 0.31),  # Warm terracotta
    specular_color: Tuple[float, float, float] = (0.8, 0.85, 1.0),  # Cool highlight
    ambient: float = 0.15,
    specular_power: float = 32.0,
    background_color: Tuple[int, int, int] = (30, 30, 45)
) -> np.ndarray:
    """
    Render an egg surface with bump mapping applied.

    This is TRUE bump/normal mapping where:
    1. Height field is used to compute displaced surface normals
       (simulating actual geometric displacement)
    2. Normal map perturbs the lighting normals for fine detail
    3. Combined result shows both surface curvature AND embossed texture

    The key insight: we use the height field for macro-displacement
    (large scale bumps) and the normal map for micro-detail lighting.

    Args:
        height_field: (H, W) height values normalized to [0, 1]
        normal_map: (H, W, 3) RGB-encoded normal map. If None, computed from height.
        displacement_scale: Scale of height displacement effect (0.05-0.3 typical)
        normal_strength: Strength of normal map perturbation (0-1)
        output_size: Output image resolution
        egg_factor: Egg shape deformation (0 = sphere, 0.3 = egg-like)
        light_dir: Light direction vector
        body_color: Diffuse surface color (RGB, 0-1)
        specular_color: Specular highlight color (RGB, 0-1)
        ambient: Ambient light intensity
        specular_power: Specular exponent
        background_color: Background RGB (0-255)

    Returns:
        (output_size, output_size, 3) RGB image as uint8
    """
    # Generate normal map from height if not provided
    if normal_map is None:
        normal_map = height_to_normals(height_field, strength=2.0)

    # Normalize light direction
    light = np.array(light_dir, dtype=np.float32)
    light = light / np.linalg.norm(light)

    # View direction (looking down -Z axis, so view is +Z)
    view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Create coordinate grids
    y_coords = np.arange(output_size, dtype=np.float32)
    x_coords = np.arange(output_size, dtype=np.float32)
    px, py = np.meshgrid(x_coords, y_coords)

    # Normalized device coordinates
    scale = 0.85
    nx = (px / output_size - 0.5) * 2 / scale
    ny = (0.5 - py / output_size) * 2 / scale

    # Ray-sphere intersection
    inside_mask, z = ray_sphere_test(nx, ny)

    # Apply egg deformation
    x_egg, z_egg, egg_mod = egg_deformation(nx, ny, z, egg_factor)

    # Compute UV coordinates
    u, v = spherical_uv_from_ray(nx, ny, z_egg, x_egg)

    # Compute base surface normal for the egg
    base_normal = compute_egg_surface_normal(nx, ny, z, egg_factor)

    # Apply height field displacement to normals
    # This simulates what the normal would be if surface was displaced
    displaced_normal = compute_displaced_normal(
        base_normal, height_field, u, v, displacement_scale
    )

    # Apply normal map perturbation on top of displacement
    # This adds fine detail from the normal map
    final_normal = apply_normal_map_perturbation(
        displaced_normal, normal_map, u, v, normal_strength
    )

    # Compute lighting using the perturbed normals
    # Lambertian diffuse
    ndotl = np.maximum(0, np.sum(final_normal * light, axis=-1))

    # Blinn-Phong specular
    half_vec = light + view_dir
    half_vec = half_vec / np.linalg.norm(half_vec)
    ndoth = np.maximum(0, np.sum(final_normal * half_vec, axis=-1))
    specular = ndoth ** specular_power

    # Fresnel effect - more specular at grazing angles
    ndotv = np.maximum(0.01, np.sum(final_normal * view_dir, axis=-1))
    fresnel = 0.04 + 0.96 * ((1.0 - ndotv) ** 5)

    # Sample height for ambient occlusion
    height_sample = sample_texture_bilinear(height_field, u, v)
    ao = 0.7 + 0.3 * height_sample  # Height-based AO

    # Combine lighting
    body = np.array(body_color)
    spec = np.array(specular_color)

    # Diffuse contribution (body reflection)
    diffuse_contrib = body[np.newaxis, np.newaxis, :] * (ndotl[..., np.newaxis] + ambient)

    # Specular contribution (interface reflection)
    specular_contrib = spec[np.newaxis, np.newaxis, :] * (specular * fresnel)[..., np.newaxis]

    # Final color with Fresnel blend
    color = (1 - fresnel[..., np.newaxis]) * diffuse_contrib + specular_contrib
    color = color * ao[..., np.newaxis]

    # Convert to uint8
    color = np.clip(color * 255, 0, 255).astype(np.uint8)

    # Create output with background
    img = np.full((output_size, output_size, 3), background_color, dtype=np.uint8)
    img[inside_mask] = color[inside_mask]

    return img


def render_color_only_egg(
    height_field: np.ndarray,
    output_size: int = 512,
    egg_factor: float = 0.25,
    light_dir: Tuple[float, float, float] = (0.5, 0.7, 0.5),
    base_color: Tuple[float, float, float] = (0.78, 0.47, 0.31),
    ambient: float = 0.25,
    background_color: Tuple[int, int, int] = (30, 30, 45)
) -> np.ndarray:
    """
    Render egg with height field as COLOR map only (no bump/displacement).

    This is the "before" comparison - the height values just modulate
    surface color, not the normals. The egg surface is smooth with
    no apparent 3D texture.

    Args:
        height_field: (H, W) height values [0, 1] - used as color intensity
        output_size: Output image resolution
        egg_factor: Egg shape deformation
        light_dir: Light direction
        base_color: Base surface color (RGB, 0-1)
        ambient: Ambient light intensity
        background_color: Background RGB (0-255)

    Returns:
        (output_size, output_size, 3) RGB image as uint8
    """
    # Normalize light direction
    light = np.array(light_dir, dtype=np.float32)
    light = light / np.linalg.norm(light)

    # Create coordinate grids
    y_coords = np.arange(output_size, dtype=np.float32)
    x_coords = np.arange(output_size, dtype=np.float32)
    px, py = np.meshgrid(x_coords, y_coords)

    # Normalized device coordinates
    scale = 0.85
    nx = (px / output_size - 0.5) * 2 / scale
    ny = (0.5 - py / output_size) * 2 / scale

    # Ray-sphere intersection
    inside_mask, z = ray_sphere_test(nx, ny)

    # Apply egg deformation
    x_egg, z_egg, egg_mod = egg_deformation(nx, ny, z, egg_factor)

    # Compute UV coordinates
    u, v = spherical_uv_from_ray(nx, ny, z_egg, x_egg)

    # Compute smooth surface normal (NO displacement)
    surf_normal = compute_egg_surface_normal(nx, ny, z, egg_factor)

    # Simple Lambertian shading with smooth normals
    ndotl = np.maximum(0, np.sum(surf_normal * light, axis=-1))

    # Sample height field as COLOR intensity only
    color_intensity = sample_texture_bilinear(height_field, u, v)

    # Modulate base color by texture (as if it were a decal)
    base = np.array(base_color)
    modulated_color = base * (0.5 + 0.5 * color_intensity[..., np.newaxis])

    # Apply lighting
    lighting = ndotl[..., np.newaxis] * 0.75 + ambient
    color = modulated_color * lighting

    # Convert to uint8
    color = np.clip(color * 255, 0, 255).astype(np.uint8)

    # Create output with background
    img = np.full((output_size, output_size, 3), background_color, dtype=np.uint8)
    img[inside_mask] = color[inside_mask]

    return img


def create_comparison_grid(
    images: list,
    labels: list,
    padding: int = 10,
    label_height: int = 30,
    font_color: Tuple[int, int, int] = (200, 200, 200)
) -> np.ndarray:
    """
    Create a side-by-side comparison grid of images.

    Args:
        images: List of (H, W, 3) uint8 images
        labels: List of label strings for each image
        padding: Padding between images
        label_height: Height reserved for labels
        font_color: Color for label text (approximate, uses simple rendering)

    Returns:
        Combined grid image
    """
    n_images = len(images)
    if n_images == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    img_h, img_w = images[0].shape[:2]

    # Grid layout: all in one row
    grid_w = n_images * img_w + (n_images + 1) * padding
    grid_h = img_h + 2 * padding + label_height

    grid = np.full((grid_h, grid_w, 3), 40, dtype=np.uint8)

    for i, (img, label) in enumerate(zip(images, labels)):
        x_offset = padding + i * (img_w + padding)
        y_offset = padding + label_height

        grid[y_offset:y_offset + img_h, x_offset:x_offset + img_w] = img

        # Simple label rendering (just a bar with approximate text position indicator)
        label_y = padding
        label_x = x_offset + img_w // 2 - len(label) * 3  # Approximate centering

        # Draw a small indicator for where label would be
        # (Full text rendering would require PIL, which we'll use if available)
        try:
            from PIL import Image, ImageDraw, ImageFont
            # Convert to PIL, draw text, convert back
            pil_grid = Image.fromarray(grid)
            draw = ImageDraw.Draw(pil_grid)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                font = ImageFont.load_default()

            # Get text size for centering
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_x = x_offset + (img_w - text_w) // 2
            draw.text((text_x, label_y + 5), label, fill=font_color, font=font)
            grid = np.array(pil_grid)
        except ImportError:
            # No PIL, just skip text
            pass

    return grid
