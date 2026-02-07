"""
Height field to normal map conversion.

Standard tangent-space normal map generation with strength control.
"""

import numpy as np
from typing import Tuple, List, Optional


def height_to_normals(
    height_field: np.ndarray,
    strength: float = 1.0,
    wrap: bool = True
) -> np.ndarray:
    """
    Convert height field to normal map.

    Args:
        height_field: 2D array of heights (any range, will be normalized)
        strength: Normal intensity multiplier (higher = more pronounced bumps)
        wrap: Whether to wrap edges (for tiling textures)

    Returns:
        3D array (H, W, 3) with RGB-encoded normals
        (0.5, 0.5, 1.0) = flat surface pointing up
    """
    h = height_field.astype(np.float64)
    height, width = h.shape

    # Normalize height to reasonable range
    h_min, h_max = h.min(), h.max()
    if h_max > h_min:
        h = (h - h_min) / (h_max - h_min)

    # Compute gradients using central differences
    dx = np.zeros_like(h)
    dy = np.zeros_like(h)

    if wrap:
        # Central differences with wrapping
        dx = np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)
        dy = np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)
        dx /= 2.0
        dy /= 2.0
    else:
        # Central differences, forward/backward at edges
        dx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / 2.0
        dx[:, 0] = h[:, 1] - h[:, 0]
        dx[:, -1] = h[:, -1] - h[:, -2]

        dy[1:-1, :] = (h[2:, :] - h[:-2, :]) / 2.0
        dy[0, :] = h[1, :] - h[0, :]
        dy[-1, :] = h[-1, :] - h[-2, :]

    # Scale gradients by strength
    dx *= strength
    dy *= strength

    # Compute normals: n = normalize(-dx, -dy, 1)
    normals = np.zeros((height, width, 3))
    normals[:, :, 0] = -dx  # X component (red channel)
    normals[:, :, 1] = -dy  # Y component (green channel)
    normals[:, :, 2] = 1.0  # Z component (blue channel)

    # Normalize to unit length
    norm = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True))
    norm = np.where(norm > 1e-10, norm, 1.0)
    normals = normals / norm

    # Convert from [-1, 1] to [0, 1] for RGB encoding
    normals = (normals + 1.0) / 2.0

    return normals


def normals_to_image(normals: np.ndarray) -> np.ndarray:
    """
    Convert normal map to uint8 image array.

    Args:
        normals: 3D array (H, W, 3) with values in [0, 1]

    Returns:
        uint8 array (H, W, 3) ready for PNG export
    """
    return (normals * 255).astype(np.uint8)


def combine_height_fields(
    fields: List[np.ndarray],
    weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    Combine multiple height fields with optional weights.

    Args:
        fields: List of 2D height arrays (same shape)
        weights: Optional weights for each field

    Returns:
        Combined height field
    """
    if not fields:
        raise ValueError("No height fields provided")

    if weights is None:
        weights = [1.0] * len(fields)

    # Ensure all fields have same shape
    shape = fields[0].shape
    for f in fields:
        if f.shape != shape:
            raise ValueError(f"Shape mismatch: {f.shape} vs {shape}")

    result = np.zeros(shape)
    for field, weight in zip(fields, weights):
        result += weight * field

    return result


def visualize_normals_lit(
    normals: np.ndarray,
    light_dir: Tuple[float, float, float] = (0.5, 0.8, 0.5)
) -> np.ndarray:
    """
    Visualize normal map with simple directional lighting.

    Args:
        normals: 3D array (H, W, 3) in [0, 1] range
        light_dir: Light direction (will be normalized)

    Returns:
        Grayscale uint8 array showing lighting
    """
    # Convert normals back to [-1, 1] range
    n = normals * 2 - 1

    # Normalize light direction
    light = np.array(light_dir)
    light = light / np.linalg.norm(light)

    # Dot product for Lambertian shading
    ndotl = np.sum(n * light, axis=2)
    ndotl = np.clip(ndotl, 0, 1)

    # Add ambient
    ambient = 0.2
    lit = ambient + (1 - ambient) * ndotl

    return (lit * 255).astype(np.uint8)


def normal_map_to_tangent_space(
    normal_map: np.ndarray,
    tangent: Tuple[float, float, float] = (1, 0, 0),
    bitangent: Tuple[float, float, float] = (0, 1, 0)
) -> np.ndarray:
    """
    Transform normal map to different tangent space.

    Args:
        normal_map: 3D array (H, W, 3) in [0, 1] range
        tangent: Tangent vector
        bitangent: Bitangent vector

    Returns:
        Transformed normal map
    """
    # Convert from [0, 1] to [-1, 1]
    n = normal_map * 2 - 1

    # Build TBN matrix
    t = np.array(tangent)
    b = np.array(bitangent)
    normal = np.cross(t, b)

    tbn = np.array([t, b, normal]).T

    # Transform normals
    height, width = normal_map.shape[:2]
    n_flat = n.reshape(-1, 3)
    transformed_flat = n_flat @ tbn
    transformed = transformed_flat.reshape(height, width, 3)

    # Convert back to [0, 1]
    return (transformed + 1.0) / 2.0
