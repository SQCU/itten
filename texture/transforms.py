"""
Spectral transforms for texture synthesis.

Implements transforms from the spectral-transforms-compendium.md.
Uses GPU-accelerated spectral operations from spectral_ops_fast.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from typing import Tuple, Optional

from spectral_ops_fast import (
    build_weighted_image_laplacian,
    lanczos_k_eigenvectors,
    DEVICE
)


def eigenvector_phase_field(
    image: np.ndarray,
    theta: float = 0.5,
    eigenpair: Tuple[int, int] = (0, 1),
    edge_threshold: float = 0.1,
    num_iterations: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create phase field from eigenvector pair.

    Treats pairs of consecutive eigenvectors as complex numbers (real + imaginary)
    and extracts phase, creating smooth rotational fields that spiral around nodal points.

    The phase field creates psychedelic spiral patterns centered on topological defects.
    Where eigenvectors have zeros, the phase winds around creating vortex-like structures.

    Args:
        image: (H, W) grayscale image, values in [0, 1] or [0, 255]
        theta: Rotation angle [0, 1] - controls blend of phase and magnitude
        eigenpair: Tuple of eigenvector indices to use as (real, imag)
        edge_threshold: Carrier edge sensitivity for Laplacian weighting
        num_iterations: Lanczos iterations for eigenvector computation

    Returns:
        phase: (H, W) phase field in [-pi, pi], normalized to [0, 1]
        magnitude: (H, W) magnitude field, normalized to [0, 1]
    """
    H, W = image.shape

    # Convert image to tensor
    carrier = torch.tensor(image, dtype=torch.float32, device=DEVICE)
    if carrier.max() > 1.0:
        carrier = carrier / 255.0

    # Build weighted Laplacian
    L = build_weighted_image_laplacian(carrier, edge_threshold)

    # Compute eigenvectors - need at least max(eigenpair)+1
    num_evs = max(eigenpair) + 1
    eigenvectors, eigenvalues = lanczos_k_eigenvectors(
        L, num_evs, num_iterations
    )

    # Check we got enough eigenvectors
    if eigenvectors.shape[1] < num_evs:
        raise ValueError(f"Could not compute {num_evs} eigenvectors, got {eigenvectors.shape[1]}")

    # Get real and imaginary components from eigenvector pair
    ev_real = eigenvectors[:, eigenpair[0]].reshape(H, W)
    ev_imag = eigenvectors[:, eigenpair[1]].reshape(H, W)

    # Complex representation: phase and magnitude
    phase = np.arctan2(ev_imag, ev_real)  # Range [-pi, pi]
    magnitude = np.sqrt(ev_real**2 + ev_imag**2)

    # Normalize phase to [0, 1]
    phase_normalized = (phase + np.pi) / (2 * np.pi)

    # Normalize magnitude to [0, 1]
    mag_min, mag_max = magnitude.min(), magnitude.max()
    if mag_max > mag_min:
        magnitude_normalized = (magnitude - mag_min) / (mag_max - mag_min)
    else:
        magnitude_normalized = np.zeros_like(magnitude)

    # Blend phase and magnitude based on theta
    # theta=0 -> pure magnitude, theta=1 -> pure phase
    result = (1 - theta) * magnitude_normalized + theta * phase_normalized

    return result, magnitude_normalized


def fiedler_nodal_lines(
    image: np.ndarray,
    theta: float = 0.5,
    edge_threshold: float = 0.1,
    num_iterations: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract nodal lines from Fiedler vector.

    The Fiedler vector (second Laplacian eigenvector) finds the "most natural" way
    to bisect an image, following edges and contours rather than arbitrary lines.
    The nodal lines (zero-crossings) trace along image boundaries like hand-drawn curves.

    Args:
        image: (H, W) grayscale image, values in [0, 1] or [0, 255]
        theta: Blend factor [0, 1] - controls nodal line vs smooth Fiedler blend
               theta=0: smooth Fiedler field, theta=1: sharp nodal lines
        edge_threshold: Carrier edge sensitivity for Laplacian weighting
        num_iterations: Lanczos iterations for eigenvector computation

    Returns:
        result: (H, W) blended field combining Fiedler and nodal lines
        nodal_lines: (H, W) binary nodal line mask
    """
    H, W = image.shape

    # Convert image to tensor
    carrier = torch.tensor(image, dtype=torch.float32, device=DEVICE)
    if carrier.max() > 1.0:
        carrier = carrier / 255.0

    # Build weighted Laplacian
    L = build_weighted_image_laplacian(carrier, edge_threshold)

    # Get Fiedler vector (first non-trivial eigenvector)
    eigenvectors, eigenvalues = lanczos_k_eigenvectors(L, num_eigenvectors=1, num_iterations=num_iterations)
    fiedler = eigenvectors[:, 0].reshape(H, W)

    # Nodal lines are zero-crossings
    # Check for sign changes between adjacent pixels
    nodal_lines = np.zeros((H, W), dtype=np.float32)

    # Vertical crossings (between row i and row i+1)
    vertical_cross = (fiedler[:-1, :] * fiedler[1:, :]) < 0
    nodal_lines[:-1, :] = np.maximum(nodal_lines[:-1, :], vertical_cross.astype(np.float32))
    nodal_lines[1:, :] = np.maximum(nodal_lines[1:, :], vertical_cross.astype(np.float32))

    # Horizontal crossings (between col j and col j+1)
    horizontal_cross = (fiedler[:, :-1] * fiedler[:, 1:]) < 0
    nodal_lines[:, :-1] = np.maximum(nodal_lines[:, :-1], horizontal_cross.astype(np.float32))
    nodal_lines[:, 1:] = np.maximum(nodal_lines[:, 1:], horizontal_cross.astype(np.float32))

    # Normalize Fiedler to [0, 1]
    fiedler_min, fiedler_max = fiedler.min(), fiedler.max()
    if fiedler_max > fiedler_min:
        fiedler_normalized = (fiedler - fiedler_min) / (fiedler_max - fiedler_min)
    else:
        fiedler_normalized = np.zeros_like(fiedler)

    # Blend based on theta
    # theta=0 -> smooth Fiedler field, theta=1 -> sharp nodal lines
    result = (1 - theta) * fiedler_normalized + theta * nodal_lines

    return result, nodal_lines


def apply_transform(
    image: np.ndarray,
    transform_name: str,
    theta: float = 0.5,
    edge_threshold: float = 0.1,
    num_iterations: int = 50
) -> np.ndarray:
    """
    Apply a named spectral transform to an image.

    Args:
        image: (H, W) grayscale image
        transform_name: One of 'eigenvector_phase_field', 'fiedler_nodal_lines'
        theta: Transform-specific parameter [0, 1]
        edge_threshold: Carrier edge sensitivity
        num_iterations: Lanczos iterations

    Returns:
        (H, W) transformed image, normalized to [0, 1]
    """
    if transform_name == 'eigenvector_phase_field':
        result, _ = eigenvector_phase_field(
            image, theta=theta, edge_threshold=edge_threshold,
            num_iterations=num_iterations
        )
    elif transform_name == 'fiedler_nodal_lines':
        result, _ = fiedler_nodal_lines(
            image, theta=theta, edge_threshold=edge_threshold,
            num_iterations=num_iterations
        )
    else:
        raise ValueError(f"Unknown transform: {transform_name}")

    return result
