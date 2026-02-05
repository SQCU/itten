"""
Unified texture synthesis core.

THE single source of truth for texture synthesis.
All other modules should import from here.

Primary API:
    synthesize() - THE function for texture synthesis
    TextureResult - Dataclass returned by synthesize()
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field
from typing import Union, Optional, Dict, Any, Tuple

# Canonical imports from spectral_ops_fast
from spectral_ops_fast import (
    build_weighted_image_laplacian,
    lanczos_k_eigenvectors,
    DEVICE
)
import torch


@dataclass
class TextureResult:
    """
    Result of texture synthesis.

    Attributes:
        height_field: (H, W) array normalized to [0, 1], ready for bump mapping
        normal_map: (H, W, 3) array normalized to [0, 1], RGB-encoded normals
        metadata: Dict with synthesis parameters and statistics
        diagnostics: Optional dict with intermediate results for debugging
    """
    height_field: np.ndarray
    normal_map: np.ndarray
    metadata: Dict[str, Any]
    diagnostics: Optional[Dict[str, Any]] = None


def normalize_to_01(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array to [0, 1] range.

    Handles edge cases where min == max by returning zeros.

    Args:
        arr: Input array of any shape

    Returns:
        Array normalized to [0, 1] range
    """
    arr_min = arr.min()
    arr_max = arr.max()

    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    else:
        return np.zeros_like(arr)


def _compute_spectral_eigenvectors(
    carrier: np.ndarray,
    num_eigenvectors: int = 8,
    edge_threshold: float = 0.1,
    lanczos_iterations: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute first k non-trivial eigenvectors of carrier-weighted Laplacian.

    Uses GPU-accelerated Lanczos iteration from spectral_ops_fast.

    Args:
        carrier: (H, W) carrier image
        num_eigenvectors: Number of eigenvectors to compute
        edge_threshold: Carrier edge sensitivity for Laplacian weighting
        lanczos_iterations: Number of Lanczos iterations

    Returns:
        eigenvectors: (H, W, k) array - each slice is an eigenvector reshaped to image
        eigenvalues: (k,) array
    """
    height, width = carrier.shape

    # Build weighted Laplacian
    carrier_tensor = torch.tensor(carrier, dtype=torch.float32, device=DEVICE)
    if carrier_tensor.max() > 1.0:
        carrier_tensor = carrier_tensor / 255.0

    L = build_weighted_image_laplacian(carrier_tensor, edge_threshold)

    # Compute eigenvectors using canonical Lanczos
    k = min(num_eigenvectors, lanczos_iterations - 1)
    eigenvectors_flat, eigenvalues = lanczos_k_eigenvectors(
        L, k, lanczos_iterations
    )

    # Reshape to images
    actual_k = eigenvectors_flat.shape[1]
    eigenvector_images = np.zeros((height, width, actual_k), dtype=np.float32)
    for i in range(actual_k):
        eigenvector_images[:, :, i] = eigenvectors_flat[:, i].reshape(height, width)

    return eigenvector_images, eigenvalues


def _resolve_carrier(
    carrier: Union[np.ndarray, str, 'CarrierInput'],
    output_size: Optional[int] = None
) -> np.ndarray:
    """
    Resolve carrier to numpy array.

    Args:
        carrier: Can be:
            - np.ndarray: Used directly
            - str: Pattern name ('amongus', 'checkerboard', 'dragon', 'noise')
            - CarrierInput: Object with .render() method
        output_size: Size for pattern generation (required if carrier is string)

    Returns:
        (H, W) numpy array normalized to [0, 1]
    """
    if isinstance(carrier, np.ndarray):
        arr = carrier.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr

    if isinstance(carrier, str):
        from .patterns import (
            generate_amongus, generate_checkerboard,
            generate_noise, generate_dragon_curve
        )

        size = output_size or 64
        pattern_map = {
            'amongus': lambda: generate_amongus(size),
            'checkerboard': lambda: generate_checkerboard(size),
            'noise': lambda: generate_noise(size),
            'dragon': lambda: generate_dragon_curve(size),
        }

        if carrier.lower() in pattern_map:
            return pattern_map[carrier.lower()]()
        else:
            raise ValueError(f"Unknown carrier pattern: {carrier}. "
                           f"Available: {list(pattern_map.keys())}")

    # Assume it's a CarrierInput-like object
    if hasattr(carrier, 'render'):
        return carrier.render()

    raise TypeError(f"Cannot resolve carrier of type {type(carrier)}")


def _resolve_operand(
    operand: Union[np.ndarray, str, 'OperandInput', None],
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Resolve operand to numpy array.

    Args:
        operand: Can be:
            - np.ndarray: Used directly (resized if needed)
            - str: Pattern name ('noise', 'checkerboard', 'solid')
            - OperandInput: Object with .render() method
            - None: Returns uniform 0.5 array
        target_shape: (H, W) shape that operand must match

    Returns:
        (H, W) numpy array normalized to [0, 1]
    """
    height, width = target_shape

    if operand is None:
        return np.full((height, width), 0.5, dtype=np.float32)

    if isinstance(operand, np.ndarray):
        arr = operand.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

        # Resize if needed
        if arr.shape != target_shape:
            from scipy.ndimage import zoom
            zoom_factors = (height / arr.shape[0], width / arr.shape[1])
            arr = zoom(arr, zoom_factors, order=1)

        return arr

    if isinstance(operand, str):
        from .patterns import generate_checkerboard, generate_noise

        pattern_map = {
            'noise': lambda: generate_noise(max(height, width)),
            'checkerboard': lambda: generate_checkerboard(max(height, width)),
            'solid': lambda: np.full((height, width), 0.5, dtype=np.float32),
        }

        if operand.lower() in pattern_map:
            arr = pattern_map[operand.lower()]()
            # Resize if needed
            if arr.shape != target_shape:
                from scipy.ndimage import zoom
                zoom_factors = (height / arr.shape[0], width / arr.shape[1])
                arr = zoom(arr, zoom_factors, order=1)
            return arr
        else:
            raise ValueError(f"Unknown operand pattern: {operand}. "
                           f"Available: {list(pattern_map.keys())}")

    # Assume it's an OperandInput-like object
    if hasattr(operand, 'render'):
        arr = operand.render()
        if arr.shape != target_shape:
            from scipy.ndimage import zoom
            zoom_factors = (height / arr.shape[0], width / arr.shape[1])
            arr = zoom(arr, zoom_factors, order=1)
        return arr

    raise TypeError(f"Cannot resolve operand of type {type(operand)}")


def _nodal_line_segment(
    carrier: np.ndarray,
    num_eigenvectors: int = 8,
    edge_threshold: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment carrier by Fiedler vector sign and extract nodal lines.

    Args:
        carrier: (H, W) carrier image
        num_eigenvectors: Number of eigenvectors to compute
        edge_threshold: Carrier edge sensitivity

    Returns:
        segment: (H, W) binary segmentation [0, 1]
        nodal_mask: (H, W) nodal line mask [0, 1]
    """
    # Compute eigenvectors
    eigenvectors, _ = _compute_spectral_eigenvectors(
        carrier, num_eigenvectors, edge_threshold
    )

    # Fiedler vector is first eigenvector
    fiedler = eigenvectors[:, :, 0]

    # Binary segmentation by sign
    segment = (fiedler >= 0).astype(np.float32)

    # Detect nodal lines (boundaries between partitions)
    diff_up = np.abs(segment - np.roll(segment, 1, axis=0))
    diff_down = np.abs(segment - np.roll(segment, -1, axis=0))
    diff_left = np.abs(segment - np.roll(segment, 1, axis=1))
    diff_right = np.abs(segment - np.roll(segment, -1, axis=1))

    nodal_mask = np.maximum.reduce([diff_up, diff_down, diff_left, diff_right])

    return segment, nodal_mask


def _spectral_etch_residual(
    carrier: np.ndarray,
    operand: np.ndarray,
    theta: float = 0.5,
    num_eigenvectors: int = 8,
    edge_threshold: float = 0.1
) -> np.ndarray:
    """
    Compute theta-weighted nodal lines modulated by operand.

    Args:
        carrier: (H, W) carrier image
        operand: (H, W) operand pattern
        theta: Rotation angle [0, 1]
        num_eigenvectors: Number of eigenvectors
        edge_threshold: Carrier edge sensitivity

    Returns:
        etch: (H, W) spectral etch residual [0, 1]
    """
    height, width = carrier.shape

    # Compute eigenvectors
    eigenvectors, _ = _compute_spectral_eigenvectors(
        carrier, num_eigenvectors, edge_threshold
    )

    actual_k = eigenvectors.shape[2]

    # Gaussian weighting centered at theta * k
    indices = np.arange(actual_k)
    center = theta * actual_k
    weights = np.exp(-((indices - center) ** 2) / 2.0)

    # Weighted sum of absolute eigenvector values
    spectral_field = np.sum(
        np.abs(eigenvectors) * weights[np.newaxis, np.newaxis, :],
        axis=2
    )

    # Extract nodal lines from spectral field
    spectral_norm = normalize_to_01(spectral_field)

    # Gradient magnitude
    grad_y = np.abs(np.roll(spectral_norm, 1, axis=0) - np.roll(spectral_norm, -1, axis=0)) / 2
    grad_x = np.abs(np.roll(spectral_norm, 1, axis=1) - np.roll(spectral_norm, -1, axis=1)) / 2
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    nodal_lines = grad_mag

    # Zero-crossing detection from individual eigenvectors
    zero_cross_sum = np.zeros((height, width), dtype=np.float32)

    for i in range(min(3, actual_k)):
        ev = eigenvectors[:, :, i]
        ev_max = np.abs(ev).max()
        if ev_max > 1e-10:
            ev_normalized = ev / ev_max

            is_near_zero = np.abs(ev_normalized) < 0.1

            grad_ev_y = np.roll(ev, 1, axis=0) - np.roll(ev, -1, axis=0)
            grad_ev_x = np.roll(ev, 1, axis=1) - np.roll(ev, -1, axis=1)
            grad_ev_mag = np.sqrt(grad_ev_y**2 + grad_ev_x**2)

            is_actual_nodal = is_near_zero & (grad_ev_mag > 0.01)

            weight = weights[i] if i < len(weights) else 0.0
            zero_cross_sum += weight * is_actual_nodal.astype(np.float32)

    # Combine gradient-based and zero-crossing nodal lines
    combined_nodal = nodal_lines + normalize_to_01(zero_cross_sum) * 0.5

    # Modulate by operand
    operand_norm = normalize_to_01(operand.astype(np.float32))
    etch = combined_nodal * (0.5 + 0.5 * operand_norm)

    return normalize_to_01(etch)


def _height_to_normals(
    height_field: np.ndarray,
    strength: float = 1.0,
    wrap: bool = True
) -> np.ndarray:
    """
    Convert height field to normal map.

    Args:
        height_field: 2D array of heights
        strength: Normal intensity multiplier
        wrap: Whether to wrap edges

    Returns:
        3D array (H, W, 3) with RGB-encoded normals [0, 1]
    """
    h = height_field.astype(np.float64)
    height, width = h.shape

    # Normalize height to reasonable range
    h_min, h_max = h.min(), h.max()
    if h_max > h_min:
        h = (h - h_min) / (h_max - h_min)

    # Compute gradients
    dx = np.zeros_like(h)
    dy = np.zeros_like(h)

    if wrap:
        dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / 2.0
        dy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) / 2.0
    else:
        dx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / 2.0
        dx[:, 0] = h[:, 1] - h[:, 0]
        dx[:, -1] = h[:, -1] - h[:, -2]

        dy[1:-1, :] = (h[2:, :] - h[:-2, :]) / 2.0
        dy[0, :] = h[1, :] - h[0, :]
        dy[-1, :] = h[-1, :] - h[-2, :]

    # Scale by strength
    dx *= strength
    dy *= strength

    # Compute normals
    normals = np.zeros((height, width, 3))
    normals[:, :, 0] = -dx
    normals[:, :, 1] = -dy
    normals[:, :, 2] = 1.0

    # Normalize to unit length
    norm = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True))
    norm = np.where(norm > 1e-10, norm, 1.0)
    normals = normals / norm

    # Convert from [-1, 1] to [0, 1]
    normals = (normals + 1.0) / 2.0

    return normals


def synthesize(
    carrier: Union[np.ndarray, str, 'CarrierInput'],
    operand: Union[np.ndarray, str, 'OperandInput', None] = None,
    *,
    theta: float = 0.5,
    gamma: float = 0.3,
    num_eigenvectors: int = 8,
    edge_threshold: float = 0.1,
    output_size: Optional[int] = None,
    normal_strength: float = 2.0,
    mode: str = 'spectral',
    preset: Optional[str] = None,
    return_diagnostics: bool = False
) -> TextureResult:
    """
    THE unified texture synthesis function.

    Combines spectral eigenvector decomposition with carrier-operand modulation
    to produce height fields suitable for bump/displacement mapping.

    Mathematical operation:
        1. Build carrier-weighted Laplacian L
        2. Compute eigenvectors v_1, ..., v_k of L
        3. Segment carrier via Fiedler vector sign
        4. Extract nodal lines (partition boundaries)
        5. Compute theta-weighted spectral etch
        6. Combine: height = segment * 0.5 + nodal_mask * 0.3 + gamma * etch

    Args:
        carrier: Input defining structure. Can be:
            - np.ndarray: (H, W) grayscale image
            - str: Pattern name ('amongus', 'checkerboard', 'dragon', 'noise')
            - CarrierInput: Object with .render() method

        operand: Modulation signal. Can be:
            - np.ndarray: (H, W) grayscale image
            - str: Pattern name ('noise', 'checkerboard', 'solid')
            - OperandInput: Object with .render() method
            - None: Uses uniform 0.5 (no modulation)

        theta: Rotation angle in [0, 1] controlling spectral emphasis.
            - 0: Coarse structure (Fiedler vector dominates)
            - 1: Fine structure (higher eigenvectors dominate)

        gamma: Etch residual strength [0, 1]. Higher = more nodal line influence.

        num_eigenvectors: Number of Laplacian eigenvectors to compute.

        edge_threshold: Carrier edge sensitivity for Laplacian weighting.
            Lower = sharper boundaries in diffusion.

        output_size: Target output size. Only used if carrier is a string.

        normal_strength: Multiplier for normal map generation.

        mode: Synthesis mode.
            - 'spectral': Full spectral etch (default)
            - 'blend': Simple blend of carrier and operand
            - 'simple': Just Fiedler segmentation

        preset: Optional parameter preset.
            - 'coarse': theta=0.1, gamma=0.2
            - 'balanced': theta=0.5, gamma=0.3
            - 'fine': theta=0.9, gamma=0.4
            - 'etch_heavy': theta=0.5, gamma=0.6

        return_diagnostics: If True, include intermediate results in output.

    Returns:
        TextureResult with:
            - height_field: (H, W) normalized to [0, 1]
            - normal_map: (H, W, 3) RGB-encoded normals
            - metadata: Dict with parameters and statistics
            - diagnostics: Optional dict with intermediate results

    Example:
        >>> from texture import synthesize
        >>> result = synthesize('amongus', 'checkerboard', theta=0.5)
        >>> result.height_field.shape
        (64, 64)
        >>> result.normal_map.shape
        (64, 64, 3)
    """
    # Apply preset if specified
    if preset is not None:
        presets = {
            "coarse": {"theta": 0.1, "gamma": 0.2},
            "balanced": {"theta": 0.5, "gamma": 0.3},
            "fine": {"theta": 0.9, "gamma": 0.4},
            "etch_heavy": {"theta": 0.5, "gamma": 0.6},
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(presets.keys())}")
        params = presets[preset]
        theta = params["theta"]
        gamma = params["gamma"]

    # Resolve carrier
    carrier_arr = _resolve_carrier(carrier, output_size)
    height, width = carrier_arr.shape

    # Resolve operand
    operand_arr = _resolve_operand(operand, (height, width))

    # Build metadata
    metadata = {
        'theta': theta,
        'gamma': gamma,
        'num_eigenvectors': num_eigenvectors,
        'edge_threshold': edge_threshold,
        'normal_strength': normal_strength,
        'mode': mode,
        'preset': preset,
        'output_shape': (height, width),
        'carrier_type': type(carrier).__name__,
        'operand_type': type(operand).__name__ if operand is not None else 'None',
    }

    diagnostics = {} if return_diagnostics else None

    # Compute based on mode
    if mode == 'simple':
        # Just Fiedler segmentation
        segment, nodal_mask = _nodal_line_segment(
            carrier_arr, num_eigenvectors, edge_threshold
        )
        height_field = segment * 0.7 + nodal_mask * 0.3

        if diagnostics is not None:
            diagnostics['segment'] = segment
            diagnostics['nodal_mask'] = nodal_mask

    elif mode == 'blend':
        # Simple blend without spectral etch
        segment, nodal_mask = _nodal_line_segment(
            carrier_arr, num_eigenvectors, edge_threshold
        )
        operand_norm = normalize_to_01(operand_arr)
        height_field = (
            segment * 0.4 +
            nodal_mask * 0.2 +
            operand_norm * 0.4
        )

        if diagnostics is not None:
            diagnostics['segment'] = segment
            diagnostics['nodal_mask'] = nodal_mask
            diagnostics['operand'] = operand_norm

    else:  # 'spectral' mode (default)
        # Full spectral etch
        segment, nodal_mask = _nodal_line_segment(
            carrier_arr, num_eigenvectors, edge_threshold
        )

        etch = _spectral_etch_residual(
            carrier_arr, operand_arr,
            theta=theta,
            num_eigenvectors=num_eigenvectors,
            edge_threshold=edge_threshold
        )

        # Combine components
        height_field = segment * 0.5 + nodal_mask * 0.3 + gamma * etch

        if diagnostics is not None:
            diagnostics['segment'] = segment
            diagnostics['nodal_mask'] = nodal_mask
            diagnostics['etch'] = etch
            diagnostics['carrier'] = carrier_arr
            diagnostics['operand'] = operand_arr

    # Normalize height field
    height_field = normalize_to_01(height_field)

    # Generate normal map
    normal_map = _height_to_normals(height_field, strength=normal_strength)

    # Add statistics to metadata
    metadata['height_min'] = float(height_field.min())
    metadata['height_max'] = float(height_field.max())
    metadata['height_mean'] = float(height_field.mean())
    metadata['height_std'] = float(height_field.std())

    return TextureResult(
        height_field=height_field,
        normal_map=normal_map,
        metadata=metadata,
        diagnostics=diagnostics
    )


# Convenience aliases
def quick_synthesize(
    carrier: Union[np.ndarray, str],
    operand: Union[np.ndarray, str, None] = None,
    preset: str = "balanced"
) -> TextureResult:
    """
    Quick texture synthesis with preset parameters.

    Presets:
        - "coarse": theta=0.1, gamma=0.2 (large-scale structure)
        - "balanced": theta=0.5, gamma=0.3 (balanced blend)
        - "fine": theta=0.9, gamma=0.4 (fine detail)
        - "etch_heavy": theta=0.5, gamma=0.6 (strong etch)

    Args:
        carrier: Carrier input (array, string, or CarrierInput)
        operand: Operand input (array, string, OperandInput, or None)
        preset: Parameter preset name

    Returns:
        TextureResult with synthesized texture
    """
    return synthesize(carrier, operand, preset=preset)
