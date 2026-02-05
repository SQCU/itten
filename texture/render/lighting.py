"""
Canonical lighting calculations for PBR rendering.

All lighting code should import from here - no reimplementation.
"""

import numpy as np
from typing import Tuple, Optional


def lambertian_diffuse(
    normal: np.ndarray,
    light_dir: np.ndarray
) -> np.ndarray:
    """
    Lambertian diffuse: N·L clamped to [0, 1].

    Args:
        normal: Surface normal(s), shape (..., 3)
        light_dir: Light direction (normalized), shape (3,)

    Returns:
        Diffuse factor(s), shape (...)
    """
    return np.maximum(0, np.sum(normal * light_dir, axis=-1))


def blinn_phong_specular(
    normal: np.ndarray,
    light_dir: np.ndarray,
    view_dir: np.ndarray,
    power: float = 32.0
) -> np.ndarray:
    """
    Blinn-Phong specular: (N·H)^power.

    Args:
        normal: Surface normal(s), shape (..., 3)
        light_dir: Light direction (normalized)
        view_dir: View direction (normalized)
        power: Specular exponent

    Returns:
        Specular factor(s), shape (...)
    """
    half_vec = light_dir + view_dir
    half_vec = half_vec / np.linalg.norm(half_vec)
    ndoth = np.maximum(0, np.sum(normal * half_vec, axis=-1))
    return ndoth ** power


def schlick_fresnel(
    ndotv: np.ndarray,
    F0: float = 0.04
) -> np.ndarray:
    """
    Schlick's Fresnel approximation.

    Args:
        ndotv: N·V (cosine of view angle)
        F0: Reflectance at normal incidence (0.04 for dielectrics)

    Returns:
        Fresnel factor
    """
    return F0 + (1.0 - F0) * ((1.0 - ndotv) ** 5)


def ward_anisotropic_specular(
    normal: np.ndarray,
    tangent: np.ndarray,
    light_dir: np.ndarray,
    view_dir: np.ndarray,
    roughness_u: float = 0.1,
    roughness_v: float = 0.1
) -> np.ndarray:
    """
    Ward anisotropic specular model.

    Args:
        normal: Surface normal(s)
        tangent: Tangent direction (along U)
        light_dir: Light direction
        view_dir: View direction
        roughness_u: Roughness along tangent
        roughness_v: Roughness along bitangent

    Returns:
        Specular factor(s)
    """
    bitangent = np.cross(normal, tangent)

    half_vec = light_dir + view_dir
    half_norm = np.linalg.norm(half_vec, axis=-1, keepdims=True)
    half_vec = half_vec / np.maximum(half_norm, 1e-10)

    hdott = np.sum(half_vec * tangent, axis=-1)
    hdotb = np.sum(half_vec * bitangent, axis=-1)
    hdotn = np.sum(half_vec * normal, axis=-1)

    exponent = -((hdott / roughness_u) ** 2 + (hdotb / roughness_v) ** 2)
    exponent = exponent / np.maximum(hdotn ** 2, 1e-10)

    spec = np.exp(exponent) / (4 * np.pi * roughness_u * roughness_v)
    ndotl = np.maximum(0, np.sum(normal * light_dir, axis=-1))

    return spec * ndotl


def compute_tbn_frame(
    base_normal: np.ndarray,
    tangent_hint: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute tangent-bitangent-normal frame.

    Args:
        base_normal: Surface normal, shape (3,) or (..., 3)
        tangent_hint: Optional tangent direction hint

    Returns:
        (tangent, bitangent, normal) tuple
    """
    if tangent_hint is None:
        # Default tangent along X axis, projected to tangent plane
        tangent_hint = np.array([1.0, 0.0, 0.0])

    # Gram-Schmidt orthogonalization
    normal = base_normal / np.linalg.norm(base_normal, axis=-1, keepdims=True)
    tangent = tangent_hint - np.sum(tangent_hint * normal, axis=-1, keepdims=True) * normal
    tangent = tangent / np.linalg.norm(tangent, axis=-1, keepdims=True)
    bitangent = np.cross(normal, tangent)

    return tangent, bitangent, normal


def perturb_normal(
    base_normal: np.ndarray,
    tex_normal: np.ndarray,
    bump_strength: float,
    tangent: Optional[np.ndarray] = None,
    bitangent: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Perturb surface normal by texture normal.

    Args:
        base_normal: Original surface normal
        tex_normal: Normal from texture (in tangent space or world space)
        bump_strength: Perturbation strength [0, 1]
        tangent: Optional tangent for TBN transform
        bitangent: Optional bitangent for TBN transform

    Returns:
        Perturbed normal (normalized)
    """
    if tangent is not None and bitangent is not None:
        # Transform tex_normal from tangent space to world space
        world_normal = (
            tex_normal[..., 0:1] * tangent +
            tex_normal[..., 1:2] * bitangent +
            tex_normal[..., 2:3] * base_normal
        )
        perturbed = base_normal + bump_strength * (world_normal - base_normal)
    else:
        # Simple additive perturbation
        perturbed = base_normal + bump_strength * tex_normal

    # Normalize
    norm = np.linalg.norm(perturbed, axis=-1, keepdims=True)
    return perturbed / np.maximum(norm, 1e-10)


def iridescence_color(
    ndotv: np.ndarray,
    base_hue: float = 0.0,
    period: float = 1.0
) -> np.ndarray:
    """
    Thin-film iridescence color based on viewing angle.

    Args:
        ndotv: N·V (cosine of view angle)
        base_hue: Base hue offset [0, 1]
        period: Color cycle period

    Returns:
        RGB color, shape (..., 3)
    """
    phase = (1.0 - ndotv) * period + base_hue

    # Simple HSV-like rainbow
    r = np.clip(np.abs(phase * 6.0 - 3.0) - 1.0, 0, 1)
    g = np.clip(2.0 - np.abs(phase * 6.0 - 2.0), 0, 1)
    b = np.clip(2.0 - np.abs(phase * 6.0 - 4.0), 0, 1)

    return np.stack([r, g, b], axis=-1)
