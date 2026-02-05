"""
Spectral shader operations: lifted tensor-to-tensor pipeline.

This module provides the tensor-form operations for the spectral shader,
using primitives from:
- spectral_ops_fast.py (eigenvectors, Laplacians)
- block_rotation_ops.py (rotation transforms)
- translation_field_demo.py (translation fields)
- rasterization_demo.py (point-to-pixel rendering)

All operations are tensor-in, tensor-out. No pixel loops in hot paths.
torch.compile friendly: no dynamic shapes, consistent dtypes, vectorized ops.
"""
import torch
import math
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass


# Type alias for shape tuple to avoid repeated indexing
ShapeHW = Tuple[int, int]


# ============================================================
# 1. GATING
# ============================================================

def fiedler_gate(
    fiedler: torch.Tensor,      # (N,) or (H, W) Fiedler vector values
    threshold: float = 0.0,
    sharpness: float = 10.0
) -> torch.Tensor:
    """
    Soft gating from Fiedler vector via sigmoid.

    gate = sigmoid((fiedler - threshold) * sharpness)

    High gate (near 1): strong spectral resonance → thickening
    Low gate (near 0): weak resonance → shadow creation

    Args:
        fiedler: Fiedler vector (2nd eigenvector of Laplacian)
        threshold: center point of sigmoid
        sharpness: steepness of transition

    Returns:
        gate: same shape as fiedler, values in [0, 1]
    """
    return torch.sigmoid((fiedler - threshold) * sharpness)


def adaptive_threshold(
    fiedler: torch.Tensor,
    percentile: float = 50.0
) -> float:
    """
    Compute adaptive threshold as percentile of Fiedler values.

    Useful when Fiedler vector range varies across images.
    """
    flat = fiedler.flatten()
    k = int(len(flat) * percentile / 100.0)
    k = max(0, min(k, len(flat) - 1))
    sorted_vals, _ = torch.sort(flat)
    return sorted_vals[k].item()


# ============================================================
# 2. CYCLIC COLOR TRANSFORM
# ============================================================

def cyclic_color_transform(
    rgb: torch.Tensor,              # (N, 3) or (H, W, 3) RGB in [0, 1]
    rotation_strength: float = 0.3,
    contrast_strength: float = 0.5,
    phase_offset: float = 0.0
) -> torch.Tensor:
    """
    Cyclic color transform based on luminance.

    Maps luminance through a sinusoidal color wheel:
        new_c = 0.5 + amplitude * sin(2π * luminance + phase + channel_offset)

    Channel offsets are 0, 2π/3, 4π/3 for R, G, B (120° apart).

    Args:
        rgb: input colors
        rotation_strength: how much to rotate through color wheel (0-1)
        contrast_strength: amplitude of sinusoid (0-1)
        phase_offset: additional phase rotation

    Returns:
        transformed RGB, same shape as input
    """
    original_shape = rgb.shape
    if rgb.dim() == 3:
        rgb = rgb.reshape(-1, 3)

    # Compute luminance
    luminance = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]

    # Phase offsets for each channel (120 degrees apart): 0, 2*pi/3, 4*pi/3
    # Shape: (3,) for broadcasting over channels
    channel_phases = torch.tensor([0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0],
                                   device=rgb.device, dtype=rgb.dtype)

    # Sinusoidal mapping - vectorized over all channels
    amplitude = 0.4 * contrast_strength
    freq = 2.0 * math.pi * rotation_strength

    # luminance: (N,) -> (N, 1) for broadcasting with channel_phases (3,)
    # Result: (N, 3) sinusoidal values for all channels at once
    phases = phase_offset + channel_phases  # (3,)
    new_rgb = 0.5 + amplitude * torch.sin(freq * luminance.unsqueeze(-1) + phases)  # (N, 3)

    # Blend with original based on rotation_strength - single tensor op
    blend = rotation_strength
    result = (1 - blend) * rgb + blend * new_rgb

    if len(original_shape) == 3:
        result = result.reshape(original_shape)

    return torch.clamp(result, 0, 1)


def compute_shadow_colors(
    rgb: torch.Tensor,          # (N, 3) RGB
    effect_strength: float = 1.0
) -> torch.Tensor:
    """
    Shadow colors: stronger cyclic rotation with blue bias.
    """
    transformed = cyclic_color_transform(
        rgb,
        rotation_strength=0.3 * effect_strength,
        contrast_strength=0.8 * effect_strength
    )

    # Blue bias for shadow
    transformed = transformed.clone()
    transformed[..., 2] = transformed[..., 2] * 0.7 + 0.3  # boost blue
    transformed[..., 0] = transformed[..., 0] * 0.7        # reduce red

    return torch.clamp(transformed, 0, 1)


def compute_front_colors(
    rgb: torch.Tensor,          # (N, 3) RGB
    effect_strength: float = 1.0
) -> torch.Tensor:
    """
    Front colors: moderate cyclic rotation with teal/cyan bias.
    """
    transformed = cyclic_color_transform(
        rgb,
        rotation_strength=0.2 * effect_strength,
        contrast_strength=0.6 * effect_strength
    )

    # Teal/cyan bias for front
    transformed = transformed.clone()
    transformed[..., 1] = transformed[..., 1] * 0.8 + 0.2  # boost green
    transformed[..., 2] = transformed[..., 2] * 0.8 + 0.15 # boost blue
    transformed[..., 0] = transformed[..., 0] * 0.6        # reduce red

    return torch.clamp(transformed, 0, 1)


# ============================================================
# 3. SEGMENT DATA STRUCTURE
# ============================================================

@dataclass
class Segment:
    """
    A contiguous region extracted from the image.

    All coordinates are in pixel space (x, y).
    """
    mask: torch.Tensor          # (H, W) boolean mask in original image coords
    bbox: Tuple[int, int, int, int]  # (y0, y1, x0, x1)
    centroid: torch.Tensor      # (2,) as (x, y)
    coords: torch.Tensor        # (M, 2) pixel positions as (x, y)
    colors: torch.Tensor        # (M, 3) source RGB values
    label: int                  # segment ID


# ============================================================
# 4. SEGMENT EXTRACTION
# ============================================================

def extract_segments_from_contours(
    image_rgb: torch.Tensor,    # (H, W, 3) RGB image
    gate: torch.Tensor,         # (H, W) gate values
    gate_threshold: float = 0.5,
    contour_threshold: float = 1.0,
    min_pixels: int = 20,
    max_segments: int = 50,
    gray: Optional[torch.Tensor] = None  # (H, W) precomputed grayscale, avoids recomputation
) -> List[Segment]:
    """
    Extract segments from contour regions (v6-style).

    Finds contours via normalized intensity, then selects low-gate regions.
    Uses torch operations for contour detection, then extracts connected
    components via flood-fill style grouping.

    Args:
        image_rgb: source image
        gate: Fiedler gate values
        gate_threshold: select regions where gate < threshold
        contour_threshold: std devs from mean to be considered contour
        min_pixels: minimum segment size
        max_segments: maximum number to extract
        gray: precomputed grayscale tensor (optional, computed if None)

    Returns:
        List of Segment objects
    """
    device = image_rgb.device

    # Grayscale - use precomputed if provided
    if gray is None:
        gray = 0.299 * image_rgb[:,:,0] + 0.587 * image_rgb[:,:,1] + 0.114 * image_rgb[:,:,2]

    # LayerNorm style: find pixels far from mean
    gray_norm = (gray - gray.mean()) / (gray.std() + 1e-8)
    contours = torch.abs(gray_norm) > contour_threshold

    # Low gate intersection
    low_gate = gate < gate_threshold
    eligible = contours & low_gate

    if eligible.sum() < min_pixels * 3:
        # Fallback: use median gate on contours
        if contours.any():
            gate_on_contours = gate[contours]
            gate_median = torch.median(gate_on_contours)
            eligible = contours & (gate < gate_median)

    # Simple connected components via iterative labeling
    # (This is the one place we need component finding - use CPU scipy as fallback)
    labels = _torch_connected_components(eligible)

    return _labels_to_segments(image_rgb, labels, min_pixels, max_segments)


def _torch_connected_components(mask: torch.Tensor) -> torch.Tensor:
    """
    Simple connected components for a binary mask.

    Uses iterative label propagation. For production, could use
    scipy.ndimage.label or a CUDA kernel.

    torch.compile friendly: fixed iteration count, no python conditionals on tensors.
    """
    shape = mask.shape  # Keep shape as tuple, avoid H, W unpacking
    device = mask.device

    # Initialize labels: -1 for background, unique ID for each foreground pixel
    labels = torch.where(
        mask,
        torch.arange(shape[0] * shape[1], device=device).reshape(shape),
        torch.full(shape, -1, device=device, dtype=torch.long)
    )

    # Iterative propagation: each pixel takes minimum neighbor label
    # Fixed iteration count for torch.compile (converges well before 50 in practice)
    max_iters = 50

    for _ in range(max_iters):
        # For each of 4 directions, propagate minimum label (all vectorized)
        # Up
        labels[1:, :] = torch.where(
            mask[1:, :] & mask[:-1, :],
            torch.minimum(labels[1:, :], labels[:-1, :]),
            labels[1:, :]
        )
        # Down
        labels[:-1, :] = torch.where(
            mask[:-1, :] & mask[1:, :],
            torch.minimum(labels[:-1, :], labels[1:, :]),
            labels[:-1, :]
        )
        # Left
        labels[:, 1:] = torch.where(
            mask[:, 1:] & mask[:, :-1],
            torch.minimum(labels[:, 1:], labels[:, :-1]),
            labels[:, 1:]
        )
        # Right
        labels[:, :-1] = torch.where(
            mask[:, :-1] & mask[:, 1:],
            torch.minimum(labels[:, :-1], labels[:, 1:]),
            labels[:, :-1]
        )

    # Relabel to consecutive integers (vectorized via searchsorted)
    unique_labels = torch.unique(labels[labels >= 0])
    if unique_labels.numel() == 0:
        return torch.full(shape, -1, device=device, dtype=torch.long)

    # Create mapping: old_label -> new_label via searchsorted
    new_labels = torch.full_like(labels, -1)
    fg_mask = labels >= 0
    # searchsorted gives the index in unique_labels for each label value
    new_labels[fg_mask] = torch.searchsorted(unique_labels, labels[fg_mask])

    return new_labels


def _labels_to_segments(
    image_rgb: torch.Tensor,
    labels: torch.Tensor,
    min_pixels: int,
    max_segments: int
) -> List[Segment]:
    """Convert label map to list of Segment objects."""
    device = image_rgb.device

    unique_labels = torch.unique(labels[labels >= 0])
    segments = []

    for label_val in unique_labels:
        mask = labels == label_val
        pixel_count = mask.sum().item()

        if pixel_count < min_pixels:
            continue

        ys, xs = torch.where(mask)
        y0, y1 = ys.min().item(), ys.max().item() + 1
        x0, x1 = xs.min().item(), xs.max().item() + 1

        centroid = torch.tensor([xs.float().mean(), ys.float().mean()], device=device)
        coords = torch.stack([xs.float(), ys.float()], dim=-1)
        colors = image_rgb[ys, xs]
        local_mask = mask[y0:y1, x0:x1]

        segment = Segment(
            mask=local_mask,
            bbox=(y0, y1, x0, x1),
            centroid=centroid,
            coords=coords,
            colors=colors,
            label=label_val.item()
        )
        segments.append(segment)

        if len(segments) >= max_segments:
            break

    return segments


# ============================================================
# 5. BATCHED SEGMENT DATA PREPARATION
# ============================================================

def compute_segment_diameter(seg: Segment) -> float:
    """
    Compute segment diameter as greatest extent of convex hull.

    For efficiency, approximates as max of bbox diagonal and coord extent.
    """
    # Bbox diagonal
    y0, y1, x0, x1 = seg.bbox
    bbox_diag = ((y1 - y0) ** 2 + (x1 - x0) ** 2) ** 0.5

    # Coord extent (max pairwise distance approximation via extrema)
    if seg.coords.shape[0] < 2:
        return max(bbox_diag, 1.0)

    coords = seg.coords  # (M, 2) as (x, y)
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    coord_diag = ((x_max - x_min) ** 2 + (y_max - y_min) ** 2) ** 0.5

    return max(float(bbox_diag), float(coord_diag), 1.0)


def prepare_batched_segment_data(
    segments: List[Segment],
    translation_strength: float,
    shadow_offset: float,
    effect_strength: float,
    fiedler: Optional[torch.Tensor] = None,  # (H, W) for spectral modulation
    min_shadow_ratio: float = 0.1,           # Min shadow distance as ratio of diameter
    max_shadow_ratio: float = 0.9            # Max shadow distance as ratio of diameter
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare all segment data for batched scatter with diameter-based shadow projection.

    Shadow projection is modulated by:
    1. Segment diameter (larger segments → proportionally larger shadow offset)
    2. Local spectral complexity (high complexity → shorter shadow, stability)

    Shadow is projected OUTWARD from centroid, along the direction each point
    lies relative to its centroid, scaled by 0.1-0.9 of segment diameter.

    Returns:
        shadow_coords: (total_points, 2) all shadow positions
        shadow_colors: (total_points, 3) all shadow colors
        front_coords: (total_points, 2) all front positions
        front_colors: (total_points, 3) all front colors
    """
    if not segments:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        empty = torch.zeros((0, 2), device=device)
        empty_colors = torch.zeros((0, 3), device=device)
        return empty, empty_colors, empty.clone(), empty_colors.clone()

    device = segments[0].coords.device

    all_coords = []
    all_colors = []
    all_centroids = []
    all_diameters = []      # Per-point diameter (from parent segment)
    all_complexities = []   # Per-point spectral complexity

    # Compute local complexity from Fiedler if available
    if fiedler is not None:
        complexity_field = compute_local_spectral_complexity(fiedler, window_size=7)
    else:
        complexity_field = None

    # Collect all segment data with per-segment diameter
    for seg in segments:
        n_points = seg.coords.shape[0]
        diameter = compute_segment_diameter(seg)

        all_coords.append(seg.coords)
        all_colors.append(seg.colors)
        all_centroids.append(seg.centroid.unsqueeze(0).expand(n_points, -1))
        all_diameters.append(torch.full((n_points,), diameter, device=device))

        # Sample complexity at segment centroid
        if complexity_field is not None:
            cx, cy = int(seg.centroid[0].item()), int(seg.centroid[1].item())
            H, W = complexity_field.shape
            cx, cy = max(0, min(W-1, cx)), max(0, min(H-1, cy))
            complexity = complexity_field[cy, cx].item()
        else:
            complexity = 0.5  # Default mid-range
        all_complexities.append(torch.full((n_points,), complexity, device=device))

    # Stack into batched tensors
    coords = torch.cat(all_coords, dim=0)           # (N, 2)
    colors = torch.cat(all_colors, dim=0)           # (N, 3)
    centroids = torch.cat(all_centroids, dim=0)     # (N, 2)
    diameters = torch.cat(all_diameters, dim=0)     # (N,)
    complexities = torch.cat(all_complexities, dim=0)  # (N,)

    # Rotate all points 90° around their centroids (vectorized)
    rel = coords - centroids
    rotated = torch.stack([-rel[:, 1], rel[:, 0]], dim=-1) + centroids

    # Compute outward direction (unit vector from centroid to point)
    rel_rotated = rotated - centroids
    rel_dist = torch.sqrt(rel_rotated[:, 0]**2 + rel_rotated[:, 1]**2 + 1e-8)
    outward_dir = rel_rotated / rel_dist.unsqueeze(-1)  # (N, 2) unit vectors

    # Shadow distance = diameter * ratio, modulated by complexity
    # High complexity → shorter shadow (inverse modulation for stability)
    # complexity in [0, 1], map to shadow_ratio in [max, min] (inverted)
    shadow_ratio = max_shadow_ratio - complexities * (max_shadow_ratio - min_shadow_ratio)
    shadow_distance = diameters * shadow_ratio * effect_strength  # (N,)

    # Base translation for front (small fixed offset)
    base_translation = torch.tensor(
        [translation_strength * 0.3, translation_strength * 0.4],
        device=device
    )

    # Apply translations
    # Front: small rotation-aligned translation
    front_coords = rotated + base_translation.unsqueeze(0)

    # Shadow: project OUTWARD from centroid by shadow_distance
    shadow_coords = rotated + outward_dir * shadow_distance.unsqueeze(-1)

    # Compute colors (vectorized)
    shadow_colors = compute_shadow_colors(colors, effect_strength)
    front_colors = compute_front_colors(colors, effect_strength)

    return shadow_coords, shadow_colors, front_coords, front_colors


# ============================================================
# 6. LAYER-BASED SCATTER + HADAMARD COMPOSITE
# ============================================================

def scatter_to_layer(
    coords: torch.Tensor,       # (N, 2) as (x, y)
    colors: torch.Tensor,       # (N, 3) RGB
    H: int, W: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scatter points to a layer buffer with occupancy mask.

    Returns:
        layer: (H, W, 3) color buffer
        mask: (H, W) occupancy mask (1 where written, 0 elsewhere)

    For overlapping points within the same scatter, last-write-wins.
    """
    device = coords.device
    layer = torch.zeros((H, W, 3), device=device, dtype=colors.dtype)
    mask = torch.zeros((H, W), device=device, dtype=colors.dtype)

    if coords.shape[0] == 0:
        return layer, mask

    # Round to pixel coordinates and clamp
    px = coords[:, 0].round().long().clamp(0, W - 1)
    py = coords[:, 1].round().long().clamp(0, H - 1)

    # Scatter colors
    layer[py, px] = colors

    # Scatter occupancy (1s)
    mask[py, px] = 1.0

    return layer, mask


def composite_layers_hadamard(
    layers: List[torch.Tensor],     # List of (H, W, 3) color buffers
    masks: List[torch.Tensor],      # List of (H, W) occupancy masks
    priorities: List[float] = None  # Higher priority wins (default: later = higher)
) -> torch.Tensor:
    """
    Composite multiple layers using Hadamard masking.

    Priority order: later layers override earlier layers where both have content.

    Formula: output = sum_i( layer_i * mask_i * product_{j>i}(1 - mask_j) )

    For 2 layers (shadow, front):
        output = shadow * shadow_mask * (1 - front_mask) + front * front_mask
               = shadow * (shadow_mask - shadow_mask * front_mask) + front * front_mask

    This is the "front overwrites shadow" rule expressed as pure Hadamard ops.
    """
    if not layers:
        raise ValueError("Need at least one layer")

    shape_hw = layers[0].shape[:2]  # Keep as tuple
    device = layers[0].device

    if len(layers) == 1:
        return layers[0]

    if len(layers) == 2:
        # Optimized 2-layer case (shadow + front)
        shadow, front = layers[0], layers[1]
        # front_mask broadcasts over channel dim
        fm = masks[1].unsqueeze(-1)
        sm = masks[0].unsqueeze(-1)
        # Shadow visible where: shadow exists AND front doesn't
        # Front visible where: front exists
        return shadow * sm * (1 - fm) + front * fm

    # General N-layer case
    # Compute "visible mask" for each layer: it's visible where it exists
    # AND all higher-priority layers don't exist
    output = torch.zeros((*shape_hw, 3), device=device, dtype=layers[0].dtype)

    # Process in reverse priority order (highest priority last)
    accumulated_mask = torch.zeros((*shape_hw, 1), device=device, dtype=masks[0].dtype)

    for i in range(len(layers) - 1, -1, -1):
        layer_mask = masks[i].unsqueeze(-1)
        # This layer is visible where it exists and nothing above it exists
        visible = layer_mask * (1 - accumulated_mask)
        output = output + layers[i] * visible
        # Update accumulated mask
        accumulated_mask = torch.clamp(accumulated_mask + layer_mask, 0, 1)

    return output


def draw_all_segments_batched(
    output: torch.Tensor,           # (H, W, 3) base image
    segments: List[Segment],
    translation_strength: float,
    shadow_offset: float,
    effect_strength: float,
    fiedler: Optional[torch.Tensor] = None,  # (H, W) for spectral-modulated shadow projection
    z_buffer_mode: bool = False              # If True, don't overwrite existing non-black pixels
) -> torch.Tensor:
    """
    Draw all segments using batched scatter + Hadamard composite.

    This is the fully parallel version:
    1. Prepare all coords/colors in parallel (one concat + vectorized transforms)
    2. Scatter shadows to layer (one scatter op)
    3. Scatter fronts to layer (one scatter op)
    4. Hadamard composite: output = shadow * sm * (1-fm) + front * fm

    Shadow projection is modulated by:
    - Segment diameter: shadow distance = 0.1-0.9 of diameter
    - Local spectral complexity: high complexity → shorter shadow (stability)

    z_buffer_mode: If True, copy heads cannot overwrite pixels that already have
    content. This creates "stacking" behavior where segments fill in gaps rather
    than layering on top. Useful for demonstrating additive vs replacement semantics.

    No sequential dependencies between segments.
    """
    shape_hw = output.shape[:2]

    if not segments:
        return output

    # Step 1: Prepare batched data (vectorized) with spectral modulation
    shadow_coords, shadow_colors, front_coords, front_colors = prepare_batched_segment_data(
        segments, translation_strength, shadow_offset, effect_strength,
        fiedler=fiedler
    )

    # Step 2: Scatter to layers (2 parallel scatter ops)
    shadow_layer, shadow_mask = scatter_to_layer(shadow_coords, shadow_colors, shape_hw[0], shape_hw[1])
    front_layer, front_mask = scatter_to_layer(front_coords, front_colors, shape_hw[0], shape_hw[1])

    # Step 3: Hadamard composite (pure elementwise ops)
    segment_composite = composite_layers_hadamard(
        [shadow_layer, front_layer],
        [shadow_mask, front_mask]
    )

    # Step 4: Blend with base output where segments exist
    combined_mask = torch.clamp(shadow_mask + front_mask, 0, 1).unsqueeze(-1)

    if z_buffer_mode:
        # Z-buffer mode: only write to pixels that are "empty" (low luminance)
        # This prevents copy heads from overwriting existing content
        output_lum = 0.299 * output[:,:,0] + 0.587 * output[:,:,1] + 0.114 * output[:,:,2]
        empty_mask = (output_lum < 0.1).float().unsqueeze(-1)  # Consider dark pixels as empty
        combined_mask = combined_mask * empty_mask  # Only write where empty

    result = output * (1 - combined_mask) + segment_composite * combined_mask

    return result


# ============================================================
# 7. CHEBYSHEV-BASED SPECTRAL OPERATIONS (NO EIGENVECTOR COMPUTATION)
# ============================================================

def compute_spectral_gate_chebyshev(
    image_rgb: torch.Tensor,        # (H, W, 3) RGB image OR (H, W) grayscale
    L: Optional[torch.Tensor] = None,  # Sparse Laplacian (optional, built if None)
    center_ratio: float = 0.05,     # UNUSED (kept for API compat)
    width_ratio: float = 0.15,      # UNUSED (kept for API compat)
    order: int = 30,                # UNUSED (kept for API compat) - using Lanczos iterations instead
    edge_threshold: float = 0.1,    # For Laplacian construction
    sharpness: float = 10.0,        # Gate sigmoid sharpness
    percentile: float = 40.0,       # Percentile for adaptive threshold
    num_probes: int = 30,           # UNUSED (kept for API compat)
    lanczos_iterations: int = 30    # Lanczos iterations for Fiedler computation
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Compute spectral gate using GPU Lanczos Fiedler computation.

    Uses GPU-accelerated Lanczos iteration to compute the ACTUAL Fiedler vector
    (second smallest eigenvector of the Laplacian) on the full image graph.

    This provides exact spectral equivalence with the tiled approach but without
    tiling overhead. Suitable for images where full Laplacian fits in GPU memory.

    Mathematical operation:
        - Build weighted image Laplacian L
        - Lanczos iteration: L @ v = λ * v → extract v_2 (Fiedler)
        - Gate = sigmoid((Fiedler - threshold) * sharpness)

    Complexity: O(iterations * nnz(L)) where nnz ~ 4*H*W for 4-connectivity

    Args:
        image_rgb: Input image tensor
        L: Precomputed sparse Laplacian (built from image if None)
        edge_threshold: Edge sensitivity for Laplacian construction
        sharpness: Sigmoid sharpness for gate
        percentile: Percentile for adaptive threshold computation
        lanczos_iterations: Number of Lanczos iterations (more = better convergence)

    Returns:
        gate: (H, W) gate values in [0, 1]
        fiedler: (H, W) actual Fiedler vector
        L: The sparse Laplacian (for reuse)
        lambda_2: Second smallest eigenvalue
    """
    # Lazy import to avoid circular dependency
    from spectral_ops_fast import (
        build_weighted_image_laplacian,
        lanczos_fiedler_gpu
    )

    device = image_rgb.device

    # Handle RGB vs grayscale
    if image_rgb.dim() == 3 and image_rgb.shape[-1] == 3:
        H, W, _ = image_rgb.shape
        # Use grayscale for Laplacian (standard approach)
        carrier = 0.299 * image_rgb[:,:,0] + 0.587 * image_rgb[:,:,1] + 0.114 * image_rgb[:,:,2]
    else:
        H, W = image_rgb.shape
        carrier = image_rgb

    # Normalize carrier
    if carrier.max() > 1.0:
        carrier = carrier / 255.0

    # Build Laplacian if not provided
    if L is None:
        L = build_weighted_image_laplacian(carrier, edge_threshold)

    # Compute actual Fiedler vector via GPU Lanczos
    # This is O(iterations * nnz) - fast and exact
    fiedler_flat, lambda_2 = lanczos_fiedler_gpu(L, num_iterations=lanczos_iterations)

    # Reshape to image dimensions
    fiedler = fiedler_flat.reshape(H, W)

    # Compute adaptive threshold (percentile of Fiedler values)
    flat = fiedler.flatten()
    k = int(len(flat) * percentile / 100.0)
    k = max(0, min(k, len(flat) - 1))
    sorted_vals, _ = torch.sort(flat)
    threshold = sorted_vals[k].item()

    # Compute gate via sigmoid - same as explicit path
    gate = torch.sigmoid((fiedler - threshold) * sharpness)

    return gate, fiedler, L, lambda_2


def compute_complexity_chebyshev(
    spectral_field: torch.Tensor,   # (H, W) Chebyshev-filtered signal
    window_size: int = 7
) -> torch.Tensor:
    """
    Compute local spectral complexity from Chebyshev-filtered signal.

    This is equivalent to compute_local_spectral_complexity but accepts
    the Chebyshev-filtered signal instead of the Fiedler vector.

    The complexity captures gradient structure (kinks, corners) which is
    preserved by the Chebyshev filter since it's a linear operation.

    Args:
        spectral_field: (H, W) Chebyshev-filtered signal
        window_size: Local variance window

    Returns:
        complexity: (H, W) normalized complexity field
    """
    # Delegate to the existing function - it works on any 2D field
    return compute_local_spectral_complexity(spectral_field, window_size)


# ============================================================
# 7b. HIGH-GATE PATH: SPECTRALLY-MODULATED THICKENING (TENSOR OPS)
# ============================================================

def compute_local_spectral_complexity(
    fiedler: torch.Tensor,          # (H, W) Fiedler field
    window_size: int = 5
) -> torch.Tensor:
    """
    Compute local spectral complexity via tensor ops.

    High complexity = kinks, corners, pinches
    Low complexity = straight lines, smooth gradients

    All ops are convolutions or elementwise - no loops.
    """
    H, W = fiedler.shape
    device = fiedler.device
    dtype = fiedler.dtype

    # Sobel-style gradient via conv2d (tensor op)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0

    f_4d = fiedler.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    f_padded = torch.nn.functional.pad(f_4d, (1, 1, 1, 1), mode='reflect')

    grad_y = torch.nn.functional.conv2d(f_padded, sobel_y).squeeze()
    grad_x = torch.nn.functional.conv2d(f_padded, sobel_x).squeeze()
    grad_mag = torch.sqrt(grad_y**2 + grad_x**2 + 1e-8)

    # Local variance via conv2d: var = E[x²] - E[x]²
    pad = window_size // 2
    box_kernel = torch.ones(1, 1, window_size, window_size, device=device, dtype=dtype)
    box_kernel = box_kernel / (window_size * window_size)

    f_padded_box = torch.nn.functional.pad(f_4d, (pad, pad, pad, pad), mode='reflect')
    f_sq_padded = torch.nn.functional.pad(f_4d**2, (pad, pad, pad, pad), mode='reflect')

    local_mean = torch.nn.functional.conv2d(f_padded_box, box_kernel).squeeze()
    local_mean_sq = torch.nn.functional.conv2d(f_sq_padded, box_kernel).squeeze()
    local_var = (local_mean_sq - local_mean**2).clamp(min=0)

    # Combine: both indicate interesting structure
    complexity_raw = grad_mag + torch.sqrt(local_var + 1e-8)

    # Proper normalization: standardize then sigmoid
    # This spreads the distribution properly instead of clustering near 0
    # z-score centers at mean, sigmoid maps to [0, 1] with good spread
    z = (complexity_raw - complexity_raw.mean()) / (complexity_raw.std() + 1e-8)
    complexity = torch.sigmoid(z)  # Now centered ~0.5 with proper tails

    return complexity


def dilate_high_gate_regions(
    image_rgb: torch.Tensor,        # (H, W, 3) source image
    gate: torch.Tensor,             # (H, W) gate values
    gate_threshold: float = 0.5,
    dilation_radius: int = 2,
    fiedler: Optional[torch.Tensor] = None,
    modulation_strength: float = 0.3,       # Reduced: allows more growth (was 0.5)
    kernel_sigma_ratio: float = 0.6,        # sigma = radius * ratio (envelope sharpness)
    fill_threshold: float = 0.1,            # selection_mask threshold for filling
    gray: Optional[torch.Tensor] = None,    # (H, W) precomputed grayscale
    contours: Optional[torch.Tensor] = None # (H, W) precomputed contour mask
) -> torch.Tensor:
    """
    Structure-preserving thickening: Gaussian splat as MASK + transformed sampling.

    Two stages:
    1. Gaussian splat (conv2d) creates SELECTION MASK of where to thicken
    2. Fill selected locations by SAMPLING original with tiny transform (grid_sample)

    The transform = tiny translate toward contour + tiny rotation.
    This replicates structure instead of averaging - sparse patterns with internal
    voids/grain get thicker while PRESERVING their periodicity.

    Complexity REDUCES thickening (inverse modulation) for stability under
    autoregressive iteration - prevents runaway growth at kinks.

    All tensor ops: conv2d, grid_sample, elementwise.
    """
    H, W, _ = image_rgb.shape
    device = image_rgb.device
    dtype = image_rgb.dtype

    # Identify high-gate contour pixels - use precomputed if provided
    if contours is None:
        if gray is None:
            gray = 0.299 * image_rgb[:,:,0] + 0.587 * image_rgb[:,:,1] + 0.114 * image_rgb[:,:,2]
        gray_norm = (gray - gray.mean()) / (gray.std() + 1e-8)
        contours = torch.abs(gray_norm) > 1.0

    high_gate = gate > gate_threshold
    high_gate_contours = contours & high_gate

    if not high_gate_contours.any():
        return image_rgb.clone()

    # Compute complexity (inverse modulation: high complexity = LESS thickening)
    if fiedler is not None:
        complexity = compute_local_spectral_complexity(fiedler, window_size=7)
    else:
        complexity = torch.ones(H, W, device=device, dtype=dtype) * 0.5

    # STAGE 1: Gaussian splat to create selection mask
    # kernel_sigma_ratio controls envelope sharpness: lower = tighter, higher = spread
    r = dilation_radius
    k_size = 2 * r + 1
    y_grid = torch.arange(-r, r + 1, device=device, dtype=dtype)
    x_grid = torch.arange(-r, r + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
    dist_sq = xx**2 + yy**2
    sigma = dilation_radius * kernel_sigma_ratio  # Was hardcoded 0.6
    gaussian_kernel = torch.exp(-dist_sq / (2 * sigma**2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.max()
    kernel_4d = gaussian_kernel.view(1, 1, k_size, k_size)
    pad = r

    # Inverse complexity: higher complexity = less thickening (stability)
    # Higher modulation_strength → more uniform suppression → prevents sausage patterns
    # At modulation_strength=2.0: only complexity>0.45 gets significant suppression
    # At modulation_strength=5.0: complexity>0.18 starts getting suppressed
    inv_complexity = (1.0 - modulation_strength * complexity).clamp(min=0.05)
    weight_at_contours = high_gate_contours.float() * inv_complexity

    # Convolve to get selection mask
    weight_4d = weight_at_contours.unsqueeze(0).unsqueeze(0)
    weight_padded = torch.nn.functional.pad(weight_4d, (pad, pad, pad, pad), mode='constant', value=0)
    selection_mask = torch.nn.functional.conv2d(weight_padded, kernel_4d).squeeze()

    # Where to fill: not already contour, has nearby influence
    # fill_threshold controls growth: lower = more growth, higher = less
    fill_locations = (~contours) & (selection_mask > fill_threshold)

    if not fill_locations.any():
        return image_rgb.clone()

    # STAGE 2: Fill by sampling original with tiny transform
    # SwiGLU-style: gate decides where, then copy EXACT values (no interpolation)

    # Gradient of mask points toward contours (where to sample from)
    grad_y = torch.zeros_like(selection_mask)
    grad_x = torch.zeros_like(selection_mask)
    grad_y[1:, :] = selection_mask[1:, :] - selection_mask[:-1, :]
    grad_x[:, 1:] = selection_mask[:, 1:] - selection_mask[:, :-1]
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    dir_x = grad_x / grad_mag
    dir_y = grad_y / grad_mag

    # Build sampling grid: base + offset toward contour + tiny rotation
    yy_base = torch.arange(H, device=device, dtype=dtype)
    xx_base = torch.arange(W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(yy_base, xx_base, indexing='ij')

    # Offset toward contours (where to copy from)
    offset_scale = 1.2
    sample_y = grid_y + offset_scale * dir_y
    sample_x = grid_x + offset_scale * dir_x

    # Tiny rotation around sample offset
    angle = 0.03
    cos_a = torch.tensor(angle, device=device).cos()
    sin_a = torch.tensor(angle, device=device).sin()
    rel_y, rel_x = sample_y - grid_y, sample_x - grid_x
    sample_y = grid_y + cos_a * rel_y - sin_a * rel_x
    sample_x = grid_x + sin_a * rel_y + cos_a * rel_x

    # Normalize to [-1, 1] for grid_sample
    sample_grid = torch.stack([
        2.0 * sample_x / (W - 1) - 1.0,
        2.0 * sample_y / (H - 1) - 1.0
    ], dim=-1).unsqueeze(0)

    # NEAREST neighbor sampling: copy EXACT values, no fractional aliasing
    # This preserves the original value distribution (critical for dithers)
    image_4d = image_rgb.permute(2, 0, 1).unsqueeze(0)
    sampled = torch.nn.functional.grid_sample(
        image_4d, sample_grid, mode='nearest', padding_mode='border', align_corners=True
    )
    sampled_rgb = sampled.squeeze(0).permute(1, 2, 0)

    # Hard gate (SwiGLU-style): threshold the selection mask, then multiply
    # Only locations above threshold get the copied values
    hard_gate = (selection_mask > 0.2).float()  # threshold → binary
    gate_3d = (fill_locations.float() * hard_gate).unsqueeze(-1)  # (H, W, 1)

    # Output = gate * sampled + (1 - gate) * original
    # Where gate=1: exact copied value. Where gate=0: original unchanged.
    output = gate_3d * sampled_rgb + (1.0 - gate_3d) * image_rgb

    return output


# ============================================================
# 8. FULL SPECTRAL SHADER PASS
# ============================================================

def spectral_shader_pass(
    image_rgb: torch.Tensor,        # (H, W, 3) input RGB
    fiedler: torch.Tensor,          # (H, W) Fiedler vector
    config: Dict
) -> torch.Tensor:
    """
    Single pass of the spectral shader (fully parallel version).

    Pipeline:
    1. Fiedler → gate (elementwise sigmoid)
    2. High-gate path: dilate contours (parallel scatter)
    3. Low-gate path: extract segments (one connected-components pass)
    4. Batched segment transform: rotate + translate all points (vectorized)
    5. Scatter to shadow/front layers (2 parallel scatter ops)
    6. Hadamard composite: M_f * F + (1-M_f) * M_s * S (elementwise)

    No sequential dependencies between segments.

    Config keys:
        effect_strength: overall effect intensity (0-2)
        gate_threshold: Fiedler threshold for high/low split
        gate_sharpness: sigmoid steepness
        translation_strength: how far to translate segments
        shadow_offset: additional shadow displacement
        dilation_radius: thickening for high-gate regions
        min_segment_pixels: minimum segment size
        max_segments: maximum segments to process

    Returns:
        output_rgb: (H, W, 3) shaded result
    """
    device = image_rgb.device

    # Extract config with defaults
    effect_strength = config.get('effect_strength', 1.0)
    gate_threshold = config.get('gate_threshold', 0.0)
    gate_sharpness = config.get('gate_sharpness', 10.0)
    translation_strength = config.get('translation_strength', 20.0) * effect_strength
    shadow_offset = config.get('shadow_offset', 7.0) * effect_strength
    dilation_radius = int(config.get('dilation_radius', 2) * effect_strength)
    min_segment_pixels = config.get('min_segment_pixels', 20)
    max_segments = config.get('max_segments', 50)

    # 1. Compute gate (elementwise)
    gate = fiedler_gate(fiedler, gate_threshold, gate_sharpness)

    # Precompute grayscale and contours once for both high-gate and low-gate paths
    gray = 0.299 * image_rgb[:,:,0] + 0.587 * image_rgb[:,:,1] + 0.114 * image_rgb[:,:,2]
    gray_norm = (gray - gray.mean()) / (gray.std() + 1e-8)
    contours = torch.abs(gray_norm) > 1.0

    # 2. High-gate path: spectrally-modulated thickening (all tensor ops)
    # Fiedler field modulates thickening: kinks/curves get more, straight lines get less
    # ADSR-configurable: kernel_sigma_ratio (envelope), fill_threshold (growth), thicken_modulation (suppression)
    output = dilate_high_gate_regions(
        image_rgb, gate,
        gate_threshold=0.5,
        dilation_radius=dilation_radius,
        fiedler=fiedler,
        modulation_strength=config.get('thicken_modulation', 0.3),
        kernel_sigma_ratio=config.get('kernel_sigma_ratio', 0.6),
        fill_threshold=config.get('fill_threshold', 0.1),
        gray=gray,
        contours=contours
    )

    # 3. Low-gate path: extract segments (one CC pass)
    segments = extract_segments_from_contours(
        image_rgb, gate,
        gate_threshold=0.5,
        min_pixels=min_segment_pixels,
        max_segments=max_segments,
        gray=gray
    )

    # 4-6. Batched segment drawing (vectorized + 2 scatters + Hadamard)
    # Pass fiedler for diameter-based, spectrally-modulated shadow projection
    output = draw_all_segments_batched(
        output, segments,
        translation_strength=translation_strength,
        shadow_offset=shadow_offset,
        effect_strength=effect_strength,
        fiedler=fiedler,  # Local spectral structure for shadow modulation
        z_buffer_mode=config.get('z_buffer_mode', False)
    )

    return output


def spectral_shader_pass_chebyshev(
    image_rgb: torch.Tensor,        # (H, W, 3) input RGB
    config: Dict
) -> torch.Tensor:
    """
    Single pass of the spectral shader using Chebyshev filtering (Phase 1-2).

    This is the optimized path that avoids explicit eigenvector computation.
    Uses Chebyshev polynomial filtering on the graph Laplacian to compute:
    1. Spectral gate (replaces Fiedler-based gate)
    2. Spectral complexity (for thickening modulation)

    Performance: ~50-70% faster than explicit eigenvector computation.

    The Chebyshev-filtered signal captures the same low-frequency spectral
    structure as the Fiedler vector, which is sufficient for:
    - Gating (high/low spectral region separation)
    - Complexity modulation (gradient structure detection)

    Segment extraction still uses the Chebyshev-based gate, but does NOT
    compute segment spectral signatures (those require actual Fiedler values).

    Pipeline:
    1. Build image Laplacian (once)
    2. Chebyshev filter -> spectral_field (Fiedler-like, O(order * nnz))
    3. Adaptive threshold + sigmoid -> gate
    4. High-gate path: dilate contours with spectral complexity modulation
    5. Low-gate path: extract segments (one CC pass)
    6. Batched segment drawing

    Config keys:
        effect_strength: overall effect intensity (0-2)
        gate_sharpness: sigmoid steepness
        translation_strength: how far to translate segments
        shadow_offset: additional shadow displacement
        dilation_radius: thickening for high-gate regions
        min_segment_pixels: minimum segment size
        max_segments: maximum segments to process
        chebyshev_order: polynomial order (default 30)
        chebyshev_center_ratio: filter center as fraction of lambda_max (default 0.05)
        chebyshev_width_ratio: filter width as fraction of lambda_max (default 0.15)

    Returns:
        output_rgb: (H, W, 3) shaded result
    """
    device = image_rgb.device

    # Extract config with defaults
    effect_strength = config.get('effect_strength', 1.0)
    gate_sharpness = config.get('gate_sharpness', 10.0)
    translation_strength = config.get('translation_strength', 20.0) * effect_strength
    shadow_offset = config.get('shadow_offset', 7.0) * effect_strength
    dilation_radius = int(config.get('dilation_radius', 2) * effect_strength)
    min_segment_pixels = config.get('min_segment_pixels', 20)
    max_segments = config.get('max_segments', 50)

    # Chebyshev-specific parameters
    chebyshev_order = config.get('chebyshev_order', 30)
    center_ratio = config.get('chebyshev_center_ratio', 0.05)
    width_ratio = config.get('chebyshev_width_ratio', 0.15)

    # 1. Compute gate and spectral field via Chebyshev filtering (PHASE 1-2)
    # This replaces: fiedler = compute_local_eigenvectors_tiled_dither(...)[:,:,1]
    #                gate = fiedler_gate(fiedler, adaptive_threshold(fiedler, 40), sharpness)
    gate, spectral_field, L, lambda_max = compute_spectral_gate_chebyshev(
        image_rgb,
        L=None,  # Build Laplacian internally
        center_ratio=center_ratio,
        width_ratio=width_ratio,
        order=chebyshev_order,
        edge_threshold=0.1,
        sharpness=gate_sharpness,
        percentile=40.0
    )

    # Precompute grayscale and contours once for both high-gate and low-gate paths
    gray = 0.299 * image_rgb[:,:,0] + 0.587 * image_rgb[:,:,1] + 0.114 * image_rgb[:,:,2]
    gray_norm = (gray - gray.mean()) / (gray.std() + 1e-8)
    contours = torch.abs(gray_norm) > 1.0

    # 2. High-gate path: spectrally-modulated thickening (all tensor ops)
    # Uses spectral_field (Chebyshev-filtered) for complexity modulation
    # ADSR-configurable: kernel_sigma_ratio (envelope), fill_threshold (growth), thicken_modulation (suppression)
    output = dilate_high_gate_regions(
        image_rgb, gate,
        gate_threshold=0.5,
        dilation_radius=dilation_radius,
        fiedler=spectral_field,  # Chebyshev-filtered signal used for complexity
        modulation_strength=config.get('thicken_modulation', 0.3),
        kernel_sigma_ratio=config.get('kernel_sigma_ratio', 0.6),
        fill_threshold=config.get('fill_threshold', 0.1),
        gray=gray,
        contours=contours
    )

    # 3. Low-gate path: extract segments (one CC pass)
    # Gate already computed via Chebyshev - this is Phase 2
    segments = extract_segments_from_contours(
        image_rgb, gate,
        gate_threshold=0.5,
        min_pixels=min_segment_pixels,
        max_segments=max_segments,
        gray=gray
    )

    # 4-6. Batched segment drawing (vectorized + 2 scatters + Hadamard)
    # Pass spectral_field (Chebyshev-filtered) for shadow modulation
    output = draw_all_segments_batched(
        output, segments,
        translation_strength=translation_strength,
        shadow_offset=shadow_offset,
        effect_strength=effect_strength,
        fiedler=spectral_field,  # Chebyshev-filtered field for shadow modulation
        z_buffer_mode=config.get('z_buffer_mode', False)
    )

    return output


# ============================================================
# 9. TWO-IMAGE TRANSFER: A=target topology, B=source content
# ============================================================

def compute_segment_spectral_signature(
    segment: Segment,
    fiedler: torch.Tensor,      # (H, W) Fiedler field
) -> torch.Tensor:
    """
    Compute spectral signature of a segment for matching.

    Signature includes:
    - Mean Fiedler value in segment
    - Std of Fiedler value (how spectrally uniform)
    - Segment size (pixel count)
    - Aspect ratio of bounding box

    Returns (4,) feature vector.
    """
    y0, y1, x0, x1 = segment.bbox
    fiedler_shape = fiedler.shape  # Keep as tuple
    device = fiedler.device

    # Sample Fiedler at segment coords (direct shape indexing)
    xs = segment.coords[:, 0].long().clamp(0, fiedler_shape[1] - 1)
    ys = segment.coords[:, 1].long().clamp(0, fiedler_shape[0] - 1)
    fiedler_vals = fiedler[ys, xs]

    mean_f = fiedler_vals.mean()
    std_f = fiedler_vals.std() + 1e-8
    size = torch.tensor(fiedler_vals.shape[0], device=device, dtype=torch.float32)
    aspect = torch.tensor((x1 - x0) / max(y1 - y0, 1), device=device, dtype=torch.float32)

    return torch.stack([mean_f, std_f, (size + 1).log(), aspect])


def match_segments_by_topology(
    query_signatures: torch.Tensor,     # (Q, 4) signatures from A
    source_signatures: torch.Tensor,    # (S, 4) signatures from B
    top_k: int = 1
) -> torch.Tensor:
    """
    Find best matching source segments for each query.

    Uses L2 distance in signature space.

    Returns (Q, top_k) indices into source segments.
    """
    # Normalize signatures for fair comparison
    q_norm = (query_signatures - query_signatures.mean(0)) / (query_signatures.std(0) + 1e-8)
    s_norm = (source_signatures - source_signatures.mean(0)) / (source_signatures.std(0) + 1e-8)

    # Pairwise L2 distance
    # (Q, 1, 4) - (1, S, 4) -> (Q, S, 4) -> sum -> (Q, S)
    diff = q_norm.unsqueeze(1) - s_norm.unsqueeze(0)
    distances = (diff ** 2).sum(dim=-1)

    # Get top_k closest for each query
    _, indices = torch.topk(distances, k=min(top_k, distances.shape[1]), dim=1, largest=False)

    return indices


def transfer_segments_A_to_B(
    image_A: torch.Tensor,          # (H_A, W_A, 3) target image (WHERE)
    image_B: torch.Tensor,          # (H_B, W_B, 3) source image (WHAT) - can differ in size!
    fiedler_A: torch.Tensor,        # (H_A, W_A) Fiedler of A
    fiedler_B: torch.Tensor,        # (H_B, W_B) Fiedler of B
    config: Dict
) -> Tuple[List[Segment], List[Segment], torch.Tensor]:
    """
    Find segments in B that match A's topological queries.

    A and B can have DIFFERENT sizes. Matching is by spectral signature
    (scale-invariant features), not by pixel correspondence.

    Returns:
        query_segments: segments extracted from A (define WHERE to place)
        matched_segments: segments from B with matching topology (provide CONTENT)
        match_indices: which B segment matches each A segment
    """
    device = image_A.device

    gate_threshold = config.get('gate_threshold', 0.5)
    min_pixels = config.get('min_segment_pixels', 20)
    max_segments = config.get('max_segments', 50)

    # Compute gates for each image independently
    gate_A = fiedler_gate(fiedler_A, adaptive_threshold(fiedler_A, 40), 10.0)
    gate_B = fiedler_gate(fiedler_B, adaptive_threshold(fiedler_B, 40), 10.0)

    # Extract segments from both images (each in their own coordinate space)
    segments_A = extract_segments_from_contours(
        image_A, gate_A, gate_threshold=0.5,
        min_pixels=min_pixels, max_segments=max_segments
    )

    segments_B = extract_segments_from_contours(
        image_B, gate_B, gate_threshold=0.5,
        min_pixels=min_pixels, max_segments=max_segments * 2  # more candidates
    )

    if not segments_A or not segments_B:
        return segments_A, [], torch.tensor([], device=device)

    # Compute spectral signatures (scale-invariant: uses normalized features)
    # Signature: [mean_fiedler, std_fiedler, log(size), aspect_ratio]
    # These are comparable across different image sizes
    sigs_A = torch.stack([compute_segment_spectral_signature(s, fiedler_A) for s in segments_A])
    sigs_B = torch.stack([compute_segment_spectral_signature(s, fiedler_B) for s in segments_B])

    # Match A queries to B sources by spectral similarity
    match_indices = match_segments_by_topology(sigs_A, sigs_B, top_k=1).squeeze(-1)

    # Gather matched segments from B
    matched_segments = [segments_B[idx.item()] for idx in match_indices]

    return segments_A, matched_segments, match_indices


def two_image_shader_pass(
    image_A: torch.Tensor,          # (H_A, W_A, 3) target: defines WHERE, topology query
    image_B: torch.Tensor,          # (H_B, W_B, 3) source: provides WHAT, content (can differ in size!)
    fiedler_A: torch.Tensor,        # (H_A, W_A) precomputed Fiedler of A
    fiedler_B: torch.Tensor,        # (H_B, W_B) precomputed Fiedler of B
    config: Dict
) -> torch.Tensor:
    """
    Two-image spectral shader: topology from A, content from B.

    CRITICAL: A and B can have DIFFERENT sizes. No rescaling ever.
    Graphs have spectra; rescaling destroys spectra.

    Pipeline:
    1. Extract query segments from A (defines WHAT TOPOLOGY we seek, WHERE to place)
    2. Find matching segments in B by spectral signature
    3. Take B's segment AS-IS (B's shape, B's coords, B's colors)
    4. Recenter B's segment to A's query location
    5. Transform (rotate, translate into A's voids)
    6. Composite onto A

    A determines: WHERE segments go (centroid locations), WHAT topology we seek
    B determines: WHAT content (the actual segment pixels, colors, shape)
    """
    H_A, W_A, _ = image_A.shape
    device = image_A.device

    effect_strength = config.get('effect_strength', 1.0)
    translation_strength = config.get('translation_strength', 20.0) * effect_strength
    shadow_offset = config.get('shadow_offset', 7.0) * effect_strength
    dilation_radius = int(config.get('dilation_radius', 2) * effect_strength)

    # Compute gate for A (determines high/low regions in TARGET)
    gate_A = fiedler_gate(fiedler_A, adaptive_threshold(fiedler_A, 40), 10.0)

    # Step 1: High-gate path on A with spectrally-modulated thickening
    # Uses A's Fiedler to preserve expressive kinks/curves in A's structure
    # ADSR-configurable: kernel_sigma_ratio (envelope), fill_threshold (growth), thicken_modulation (suppression)
    output = dilate_high_gate_regions(
        image_A, gate_A,
        gate_threshold=0.5,
        dilation_radius=dilation_radius,
        fiedler=fiedler_A,
        modulation_strength=config.get('thicken_modulation', 0.3),
        kernel_sigma_ratio=config.get('kernel_sigma_ratio', 0.6),
        fill_threshold=config.get('fill_threshold', 0.1)
    )

    # Step 2: Find matching segments (A's topology query → B's content)
    segments_A, matched_B, match_indices = transfer_segments_A_to_B(
        image_A, image_B, fiedler_A, fiedler_B, config
    )

    if not segments_A or not matched_B:
        return output

    # Step 3: Take B's segments AS-IS, recenter to A's locations
    # B provides: shape, colors, relative structure
    # A provides: where to place (centroid location in A's space)
    transplanted_segments = []
    for seg_A, seg_B in zip(segments_A, matched_B):
        # B's segment content, recentered to A's location
        # offset = A's centroid - B's centroid
        offset = seg_A.centroid - seg_B.centroid

        # Translate B's coords to A's location
        transplanted_coords = seg_B.coords + offset.unsqueeze(0)

        # Compute new bbox in A's space
        xs = transplanted_coords[:, 0]
        ys = transplanted_coords[:, 1]
        new_bbox = (
            int(ys.min().item()), int(ys.max().item()) + 1,
            int(xs.min().item()), int(xs.max().item()) + 1
        )

        transplanted = Segment(
            mask=seg_B.mask,                # B's shape (actual content shape)
            bbox=new_bbox,                  # recomputed in A's space
            centroid=seg_A.centroid,        # A's location (WHERE)
            coords=transplanted_coords,     # B's structure, moved to A's location
            colors=seg_B.colors,            # B's actual colors (WHAT)
            label=seg_A.label
        )
        transplanted_segments.append(transplanted)

    # Step 4: Draw transplanted segments using A's distance field for translation
    # Pass fiedler_A for diameter-based, spectrally-modulated shadow projection
    output = draw_all_segments_batched(
        output, transplanted_segments,
        translation_strength=translation_strength,
        shadow_offset=shadow_offset,
        effect_strength=effect_strength,
        fiedler=fiedler_A,  # A's spectral structure for shadow modulation
        z_buffer_mode=config.get('z_buffer_mode', False)
    )

    return output


# ============================================================
# 10. UNIFIED FORWARD PASS
# ============================================================

def _compute_fiedler_from_tensor(
    image_tensor: torch.Tensor,
    tile_size: int = 64,
    overlap: int = 16,
    num_eigenvectors: int = 4,
    radii: List[int] = None,
    radius_weights: List[float] = None
) -> torch.Tensor:
    """
    Compute Fiedler vector from image tensor via spectral_ops_fast.

    Args:
        image_tensor: (H, W, 3) RGB tensor
        tile_size: tile size for eigenvector computation
        overlap: overlap between tiles
        num_eigenvectors: number of eigenvectors to compute
        radii: graph construction radii
        radius_weights: weights for each radius

    Returns:
        fiedler: (H, W) Fiedler vector as torch tensor
    """
    # Lazy import to avoid circular dependency at module load
    from spectral_ops_fast import compute_local_eigenvectors_tiled_dither

    if radii is None:
        radii = [1, 2, 3, 4, 5, 6]
    if radius_weights is None:
        radius_weights = [1.0, 0.6, 0.4, 0.3, 0.2, 0.1]

    # spectral_ops_fast now takes torch tensors directly
    evecs = compute_local_eigenvectors_tiled_dither(
        image_tensor, tile_size=tile_size, overlap=overlap, num_eigenvectors=num_eigenvectors,
        radii=radii, radius_weights=radius_weights
    )

    # Extract Fiedler (eigenvector index 1)
    return evecs[:, :, 1]


def shader_forwards(
    image_A: torch.Tensor,                      # (H, W, 3) primary/target image
    image_B: Optional[torch.Tensor] = None,     # (H_B, W_B, 3) source image (optional)
    fiedler_A: Optional[torch.Tensor] = None,   # (H, W) Fiedler of A (optional, computed if None)
    fiedler_B: Optional[torch.Tensor] = None,   # (H_B, W_B) Fiedler of B (optional)
    config: Optional[Dict] = None,
    use_chebyshev: Optional[bool] = None        # None = auto-detect, True = force Chebyshev, False = force explicit
) -> torch.Tensor:
    """
    Unified forward pass for spectral shader.

    Handles both single-image and two-image cases:
    - Single image (image_B is None): self-shading via spectral_shader_pass
    - Two images: topology from A, content from B via two_image_shader_pass

    Chebyshev mode (Phase 1-2 optimization):
    - In single-image mode, can use Chebyshev filtering instead of explicit eigenvectors
    - Chebyshev is ~50-70% faster for gate/complexity computation
    - Auto-enabled for single-image mode when fiedler_A is not precomputed
    - Two-image mode always uses explicit Fiedler (segment signatures need actual values)

    Args:
        image_A: Primary/target image tensor (H, W, 3)
        image_B: Optional source image for content transfer (H_B, W_B, 3)
        fiedler_A: Precomputed Fiedler vector for A (computed if None)
        fiedler_B: Precomputed Fiedler vector for B (computed if None, only used if image_B provided)
        config: Shader configuration dict (uses defaults if None)
        use_chebyshev: Whether to use Chebyshev filtering instead of explicit eigenvectors
            - None (default): Auto-detect. Use Chebyshev for single-image when fiedler not precomputed
            - True: Force Chebyshev path (single-image only)
            - False: Force explicit eigenvector path

    Returns:
        result: Shaded output tensor, same size as image_A
    """
    if config is None:
        config = {}

    # Two-image case: always use explicit Fiedler (segment signatures need actual values)
    if image_B is not None:
        # Compute Fiedler for A if not provided
        if fiedler_A is None:
            fiedler_A = _compute_fiedler_from_tensor(image_A)
        # Compute Fiedler for B if not provided
        if fiedler_B is None:
            fiedler_B = _compute_fiedler_from_tensor(image_B)
        return two_image_shader_pass(image_A, image_B, fiedler_A, fiedler_B, config)

    # Single image case: explicit path by default
    # NOTE: Chebyshev path is NOT equivalent to explicit path - it computes global
    # spectral structure while explicit computes LOCAL tiled spectral structure.
    # Use explicit path by default for correct behavior.
    if use_chebyshev is None:
        use_chebyshev = False  # Default to explicit tiled path (correct algorithm)

    if use_chebyshev:
        # WARNING: This path produces different results than explicit path
        # Only use if you explicitly want global spectral gating (non-standard)
        return spectral_shader_pass_chebyshev(image_A, config)
    else:
        # Original path: explicit Fiedler vector
        if fiedler_A is None:
            fiedler_A = _compute_fiedler_from_tensor(image_A)
        # Set adaptive threshold if not specified
        if 'gate_threshold' not in config:
            config = {**config, 'gate_threshold': adaptive_threshold(fiedler_A, 40.0)}
        return spectral_shader_pass(image_A, fiedler_A, config)


# ============================================================
# 11. AUTOREGRESSIVE DEPTH
# ============================================================

def shader_autoregressive(
    image: torch.Tensor,
    n_passes: int = 4,
    config: Optional[Dict] = None,
    schedule_fn: Optional[Callable[[Dict, int], Dict]] = None,
    use_chebyshev: Optional[bool] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Autoregressive shader: for n in range(n_passes): x = forwards(x, config); config = schedule_fn(config, n)

    Args:
        image: Input RGB tensor (H, W, 3)
        n_passes: Number of iterations
        config: Configuration dict (uses defaults if None)
        schedule_fn: Optional (config, pass_index) -> config transformer
        use_chebyshev: Whether to use Chebyshev filtering (default False)
            - None/False: Use explicit tiled eigenvector path (correct algorithm)
            - True: Force Chebyshev path (WARNING: not equivalent, global spectral only)

    Returns:
        final_output: Result after n_passes
        intermediates: List of outputs after each pass
    """
    if config is None:
        config = {}

    current = image
    intermediates = []

    for n in range(n_passes):
        current = shader_forwards(current, config=config, use_chebyshev=use_chebyshev)
        intermediates.append(current.clone())
        if schedule_fn is not None:
            config = schedule_fn(config, n)

    return current, intermediates


# ============================================================
# 12. DEMO / TEST (simplified via shader_forwards)
# ============================================================

def _load_image_to_tensor(img_path: str, device: torch.device) -> torch.Tensor:
    """Helper: load image file to (H, W, 3) tensor."""
    from PIL import Image
    import numpy as np
    img_pil = Image.open(img_path).convert('RGB')
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    return torch.tensor(img_np, dtype=torch.float32, device=device)


def _save_tensor_to_image(tensor: torch.Tensor, path: str):
    """Helper: save (H, W, 3) tensor to image file with mandatory timestamp."""
    from image_io import save_image
    return save_image(tensor, path, timestamp=True)


def demo_spectral_shader_ops():
    """Demo single-image shading via shader_forwards()."""
    print("=" * 60)
    print("SPECTRAL SHADER OPS DEMO")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load test image
    img_path = "demo_output/inputs/1bit redraw.png"
    image_rgb = _load_image_to_tensor(img_path, device)
    print(f"\nImage: {image_rgb.shape[1]}x{image_rgb.shape[0]}")

    # Compute Fiedler via I/O boundary helper (precompute for gate visualization)
    print("\nComputing dither-aware eigenvectors...")
    fiedler = _compute_fiedler_from_tensor(image_rgb)

    # Run shader via unified forward
    print("\nRunning spectral shader pass...")
    config = {
        'effect_strength': 1.0,
        'gate_sharpness': 10.0,
        'translation_strength': 15.0,
        'shadow_offset': 5.0,
        'dilation_radius': 2,
        'min_segment_pixels': 30,
        'max_segments': 40
    }

    output = shader_forwards(image_rgb, fiedler_A=fiedler, config=config)

    # Save results (timestamps auto-injected)
    _save_tensor_to_image(output, 'demo_output/spectral_shader_lifted.png')

    # Also save gate visualization
    gate_thresh = adaptive_threshold(fiedler, 40.0)
    gate = fiedler_gate(fiedler, gate_thresh, config['gate_sharpness'])
    _save_tensor_to_image(gate.unsqueeze(-1).expand(-1, -1, 3), 'demo_output/spectral_shader_gate.png')

    return output, fiedler, gate


def demo_two_image_transfer(img_A_path=None, img_B_path=None):
    """Demo two-image transfer via shader_forwards()."""
    print("=" * 60)
    print("TWO-IMAGE TRANSFER DEMO")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Default images
    img_A_path = img_A_path or "demo_output/inputs/1bit redraw.png"
    img_B_path = img_B_path or "demo_output/inputs/snek-heavy.png"

    # Load via helper (no rescaling - graphs have spectra)
    image_A = _load_image_to_tensor(img_A_path, device)
    image_B = _load_image_to_tensor(img_B_path, device)

    print(f"\nImage A (target): {img_A_path}")
    print(f"  Size: {image_A.shape[1]}x{image_A.shape[0]}")
    print(f"Image B (source): {img_B_path}")
    print(f"  Size: {image_B.shape[1]}x{image_B.shape[0]} (NO RESCALING)")

    # Run via unified forward (Fiedler computed internally)
    print("\nRunning two-image transfer (A=topology, B=content)...")
    config = {
        'effect_strength': 1.0,
        'translation_strength': 15.0,
        'shadow_offset': 5.0,
        'dilation_radius': 2,
        'min_segment_pixels': 30,
        'max_segments': 40
    }

    output = shader_forwards(image_A, image_B, config=config)

    # Save results (timestamps auto-injected)
    _save_tensor_to_image(output, 'demo_output/two_image_transfer.png')
    _save_tensor_to_image(image_B, 'demo_output/two_image_source_B.png')

    return output


def batch_random_pairs(n_pairs: int = 10):
    """Test random image pairs via shader_forwards()."""
    import random
    import os
    from datetime import datetime

    print("=" * 60)
    print("BATCH RANDOM PAIRS TEST")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Get all input images
    input_dir = "demo_output/inputs"
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    print(f"\nFound {len(image_files)} input images")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"demo_output/batch_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate random pairs (self-pairs OK = self-attention)
    random.seed(42)
    pairs = [(random.choice(image_files), random.choice(image_files)) for _ in range(n_pairs)]

    config = {
        'effect_strength': 1.0,
        'translation_strength': 15.0,
        'shadow_offset': 5.0,
        'dilation_radius': 2,
        'min_segment_pixels': 25,
        'max_segments': 40
    }

    results = []
    for i, (file_A, file_B) in enumerate(pairs):
        print(f"\nPair {i+1}/{n_pairs}: {file_A} -> {file_B}")

        path_A = os.path.join(input_dir, file_A)
        path_B = os.path.join(input_dir, file_B)

        image_A = _load_image_to_tensor(path_A, device)
        image_B = _load_image_to_tensor(path_B, device) if file_A != file_B else None

        print(f"  A: {image_A.shape[1]}x{image_A.shape[0]}")
        if image_B is not None:
            print(f"  B: {image_B.shape[1]}x{image_B.shape[0]}")

        try:
            output = shader_forwards(image_A, image_B, config=config)

            # Save with descriptive name
            name_A = os.path.splitext(file_A)[0][:20]
            name_B = os.path.splitext(file_B)[0][:20]
            output_name = f"{i+1:02d}_{name_A}_x_{name_B}.png"
            output_path = os.path.join(output_dir, output_name)

            _save_tensor_to_image(output, output_path)
            print(f"  Saved: {output_path}")
            results.append((file_A, file_B, "success", output_path))

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((file_A, file_B, f"error: {e}", None))

    # Summary
    successes = sum(1 for r in results if r[2] == "success")
    print(f"\nSuccess: {successes}/{n_pairs} in {output_dir}")
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--two-image":
        # Optional: specify images as args
        img_A = sys.argv[2] if len(sys.argv) > 2 else None
        img_B = sys.argv[3] if len(sys.argv) > 3 else None
        demo_two_image_transfer(img_A, img_B)
    elif len(sys.argv) > 1 and sys.argv[1] == "--batch":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        batch_random_pairs(n)
    else:
        demo_spectral_shader_ops()
