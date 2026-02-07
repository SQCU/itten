"""
Contour and nodal line extraction for texture synthesis.

Provides functions to extract meaningful contours from scalar fields
and eigenvector decompositions.
"""

import numpy as np
from typing import Tuple, Optional


def extract_contours(
    field: np.ndarray,
    num_levels: int = 5,
    line_width: float = 0.05
) -> np.ndarray:
    """
    Extract contour lines from a scalar field.

    Args:
        field: 2D scalar field
        num_levels: Number of contour levels
        line_width: Width of contour lines (fraction of value range)

    Returns:
        Binary image with 1 where contours exist
    """
    # Normalize field to [0, 1]
    f_min, f_max = field.min(), field.max()
    if f_max > f_min:
        field_norm = (field - f_min) / (f_max - f_min)
    else:
        return np.zeros_like(field)

    # Compute gradient magnitude for line detection
    grad_y = np.abs(np.roll(field_norm, 1, axis=0) - np.roll(field_norm, -1, axis=0)) / 2
    grad_x = np.abs(np.roll(field_norm, 1, axis=1) - np.roll(field_norm, -1, axis=1)) / 2
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Create contour image
    contours = np.zeros_like(field)

    # For each level, find where field crosses that value
    for level in np.linspace(0, 1, num_levels + 2)[1:-1]:
        # Distance to level
        dist_to_level = np.abs(field_norm - level)

        # Mark pixels close to level AND with high gradient
        threshold = line_width
        is_contour = (dist_to_level < threshold) & (grad_mag > 0.01)
        contours = np.maximum(contours, is_contour.astype(float) * (1 - dist_to_level / threshold))

    return contours


def extract_nodal_lines(
    eigenvector: np.ndarray,
    threshold: float = 0.1,
    min_gradient: float = 0.01
) -> np.ndarray:
    """
    Extract nodal lines from a single eigenvector.

    Nodal lines are where the eigenvector crosses zero.

    Args:
        eigenvector: 2D eigenvector field
        threshold: How close to zero counts as "on nodal line"
        min_gradient: Minimum gradient to distinguish actual nodal lines from flat regions

    Returns:
        Binary mask of nodal lines
    """
    ev = eigenvector
    ev_max = np.abs(ev).max()

    if ev_max < 1e-10:
        return np.zeros_like(ev)

    ev_normalized = ev / ev_max

    # Near zero check
    is_near_zero = np.abs(ev_normalized) < threshold

    # Gradient check (actual nodal line has gradient)
    grad_y = np.roll(ev, 1, axis=0) - np.roll(ev, -1, axis=0)
    grad_x = np.roll(ev, 1, axis=1) - np.roll(ev, -1, axis=1)
    grad_mag = np.sqrt(grad_y**2 + grad_x**2)

    is_nodal = is_near_zero & (grad_mag > min_gradient)

    return is_nodal.astype(np.float32)


def extract_partition_boundary(
    fiedler: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract partition boundary from Fiedler vector.

    The Fiedler vector partitions the domain into two regions
    based on sign. The boundary is the nodal line.

    Args:
        fiedler: 2D Fiedler vector field

    Returns:
        partition: Binary partition (0 or 1)
        boundary: Boundary mask
    """
    # Partition by sign
    partition = (fiedler >= 0).astype(np.float32)

    # Boundary: edges between partitions
    diff_up = np.abs(partition - np.roll(partition, 1, axis=0))
    diff_down = np.abs(partition - np.roll(partition, -1, axis=0))
    diff_left = np.abs(partition - np.roll(partition, 1, axis=1))
    diff_right = np.abs(partition - np.roll(partition, -1, axis=1))

    boundary = np.maximum.reduce([diff_up, diff_down, diff_left, diff_right])

    return partition, boundary


def extract_multi_eigenvector_contours(
    eigenvectors: np.ndarray,
    weights: Optional[np.ndarray] = None,
    num_levels: int = 5
) -> np.ndarray:
    """
    Extract contours from weighted sum of eigenvectors.

    Args:
        eigenvectors: (H, W, k) stack of eigenvector fields
        weights: (k,) weights for each eigenvector (default: uniform)
        num_levels: Number of contour levels

    Returns:
        Combined contour image
    """
    k = eigenvectors.shape[2]

    if weights is None:
        weights = np.ones(k) / k

    # Weighted sum of absolute eigenvector values
    combined_field = np.zeros(eigenvectors.shape[:2], dtype=np.float32)
    for i in range(k):
        combined_field += weights[i] * np.abs(eigenvectors[:, :, i])

    # Extract contours from combined field
    return extract_contours(combined_field, num_levels)


def carrier_edge_field(
    carrier: np.ndarray,
    smooth_sigma: float = 0.0
) -> np.ndarray:
    """
    Compute edge field from carrier image.

    Args:
        carrier: 2D carrier image
        smooth_sigma: Optional Gaussian smoothing before edge detection

    Returns:
        Edge magnitude field normalized to [0, 1]
    """
    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        carrier = gaussian_filter(carrier, sigma=smooth_sigma)

    # Gradient
    grad_y = np.roll(carrier, 1, axis=0) - np.roll(carrier, -1, axis=0)
    grad_x = np.roll(carrier, 1, axis=1) - np.roll(carrier, -1, axis=1)
    edges = np.sqrt(grad_x**2 + grad_y**2)

    # Normalize
    if edges.max() > edges.min():
        edges = (edges - edges.min()) / (edges.max() - edges.min())

    return edges
