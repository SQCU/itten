"""Ray casting utilities for egg/sphere rendering."""

import numpy as np


def ray_sphere_test(nx: np.ndarray, ny: np.ndarray) -> tuple:
    """
    Test rays against unit sphere.

    Args:
        nx: Normalized X coordinates
        ny: Normalized Y coordinates

    Returns:
        (inside_mask, z) tuple where inside_mask indicates valid hits
        and z is the sphere surface depth
    """
    r_sq = nx * nx + ny * ny
    inside_mask = r_sq <= 1.0
    z = np.sqrt(np.maximum(0, 1.0 - r_sq))
    return inside_mask, z


def egg_deformation(
    nx: np.ndarray,
    ny: np.ndarray,
    z: np.ndarray,
    egg_factor: float = 0.25
) -> tuple:
    """
    Apply egg deformation to sphere coordinates.

    Squeezes the bottom (y < 0) and expands the top (y > 0)
    to create an egg-like shape from a sphere.

    Args:
        nx: Normalized X coordinates
        ny: Normalized Y coordinates
        z: Sphere Z depth values
        egg_factor: Deformation amount (0 = sphere, 0.3 = egg-like)

    Returns:
        (x_egg, z_egg, egg_mod) tuple with deformed coordinates
    """
    egg_mod = 1.0 - egg_factor * ny
    x_egg = np.where(np.abs(egg_mod) > 0.01, nx / egg_mod, nx)
    z_egg = np.where(np.abs(egg_mod) > 0.01, z / egg_mod, z)
    return x_egg, z_egg, egg_mod


def spherical_uv_from_ray(
    nx: np.ndarray,
    ny: np.ndarray,
    z_egg: np.ndarray,
    x_egg: np.ndarray
) -> tuple:
    """
    Compute UV coordinates from ray intersection point.

    Uses spherical mapping where:
    - phi = angle from top (0 at north pole, pi at south pole)
    - theta = angle around vertical axis

    Args:
        nx: Normalized X coordinates (unused, kept for API consistency)
        ny: Normalized Y coordinates
        z_egg: Deformed Z coordinates
        x_egg: Deformed X coordinates

    Returns:
        (u, v) tuple with UV coordinates in [0, 1] range
    """
    phi = np.arccos(np.clip(ny, -1, 1))
    theta = np.arctan2(z_egg, x_egg)
    u = (theta / (2 * np.pi) + 0.5) % 1.0
    v = phi / np.pi
    return u, v
