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
from typing import Dict, Tuple, Optional, List, Callable
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
# 2. PRIMITIVE HELPERS (deduplicated)
# ============================================================

def to_grayscale(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to grayscale via standard luminance weights."""
    if rgb.dim() == 2:
        if rgb.shape[-1] == 3:
            # (N, 3) -> (N,)
            return 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
        return rgb  # Already grayscale (H, W) or (N,)
    if rgb.dim() == 3:
        if rgb.shape[-1] == 3:
            # (H, W, 3) -> (H, W)
            return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        if rgb.shape[0] == 3:
            # (3, H, W) -> (H, W)
            return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    raise ValueError(f"Unsupported shape for grayscale conversion: {rgb.shape}")


def detect_contours(gray: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Detect contours as pixels far from mean intensity (layer-norm style)."""
    gray_norm = (gray - gray.mean()) / (gray.std() + 1e-8)
    return torch.abs(gray_norm) > threshold


# ============================================================
# 3. LOW-GATE TRANSFORM (PURE TENSOR - NO SEGMENT EXTRACTION)
# ============================================================

def apply_low_gate_transform(
    image_rgb: torch.Tensor,        # (H, W, 3) source image
    gate: torch.Tensor,             # (H, W) gate values in [0, 1]
    fiedler: torch.Tensor,          # (H, W) Fiedler vector
    contours: torch.Tensor,         # (H, W) boolean contour mask
    shadow_offset: float = 7.0,
    translation_strength: float = 20.0,
    effect_strength: float = 1.0
) -> torch.Tensor:
    """
    Pure tensor low-gate transform using grid_sample.

    Replaces: extract_segments_from_contours + draw_all_segments_batched

    Uses Fiedler gradient to determine local direction:
    - Gradient points along contours (tangent)
    - 90° to gradient = perpendicular = shadow/copy direction

    No segment extraction, no connected components, no coordinate unpacking.
    Just tensor ops: gradient, grid_sample, composite.
    """
    H, W, _ = image_rgb.shape
    device = image_rgb.device
    dtype = image_rgb.dtype

    # Low-gate selection: soft mask for where to apply transform
    # Invert gate: low gate = high selection weight
    low_gate_weight = (1.0 - gate) * contours.float()

    if low_gate_weight.sum() < 10:
        return image_rgb.clone()

    # Compute Fiedler gradient (tangent direction along contours)
    # Sobel gives gradient; gradient direction = tangent to level sets
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0

    f_4d = fiedler.unsqueeze(0).unsqueeze(0)
    f_padded = torch.nn.functional.pad(f_4d, (1, 1, 1, 1), mode='reflect')

    grad_x = torch.nn.functional.conv2d(f_padded, sobel_x).squeeze()
    grad_y = torch.nn.functional.conv2d(f_padded, sobel_y).squeeze()

    # Normalize gradient (avoid division by zero)
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    tangent_x = grad_x / grad_mag  # Unit tangent along contour
    tangent_y = grad_y / grad_mag

    # Perpendicular = 90° rotation of tangent
    perp_x = -tangent_y  # (rotate 90° CCW)
    perp_y = tangent_x

    # Build coordinate grids
    yy = torch.arange(H, device=device, dtype=dtype)
    xx = torch.arange(W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')

    # Shadow offset: sample from positions offset in perpendicular direction
    shadow_dist = shadow_offset * effect_strength
    shadow_sample_x = grid_x + perp_x * shadow_dist
    shadow_sample_y = grid_y + perp_y * shadow_dist

    # Front offset: small fixed translation
    front_dx = translation_strength * 0.3 * effect_strength
    front_dy = translation_strength * 0.4 * effect_strength
    front_sample_x = grid_x + front_dx
    front_sample_y = grid_y + front_dy

    # Normalize to [-1, 1] for grid_sample
    def to_grid(sx, sy):
        gx = 2.0 * sx / (W - 1) - 1.0
        gy = 2.0 * sy / (H - 1) - 1.0
        return torch.stack([gx, gy], dim=-1).unsqueeze(0)

    shadow_grid = to_grid(shadow_sample_x, shadow_sample_y)
    front_grid = to_grid(front_sample_x, front_sample_y)

    # Sample from source image
    image_4d = image_rgb.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    shadow_sampled = torch.nn.functional.grid_sample(
        image_4d, shadow_grid, mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(0).permute(1, 2, 0)  # (H, W, 3)

    front_sampled = torch.nn.functional.grid_sample(
        image_4d, front_grid, mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(0).permute(1, 2, 0)  # (H, W, 3)

    # Apply color transforms
    shadow_colors = compute_shadow_colors(shadow_sampled.reshape(-1, 3), effect_strength)
    shadow_colors = shadow_colors.reshape(H, W, 3)

    front_colors = compute_front_colors(front_sampled.reshape(-1, 3), effect_strength)
    front_colors = front_colors.reshape(H, W, 3)

    # Composite: shadow behind front, both masked by low_gate_weight
    # Layer order: base → shadow → front
    weight_3d = low_gate_weight.unsqueeze(-1)

    # Shadow layer (reduced weight so front shows through)
    shadow_weight = weight_3d * 0.6
    output = image_rgb * (1.0 - shadow_weight) + shadow_colors * shadow_weight

    # Front layer on top
    front_weight = weight_3d * 0.8
    output = output * (1.0 - front_weight) + front_colors * front_weight

    return output


# ============================================================
# 4. SEGMENT INFRASTRUCTURE + CROSS-ATTENTION TRANSFER
# ============================================================

@dataclass
class Segment:
    """A contiguous region extracted from the image. Coords as (x, y)."""
    mask: torch.Tensor          # (h, w) boolean local mask within bbox
    bbox: Tuple[int, int, int, int]  # (y0, y1, x0, x1)
    centroid: torch.Tensor      # (2,) as (x, y)
    coords: torch.Tensor        # (M, 2) pixel positions as (x, y)
    colors: torch.Tensor        # (M, 3) source RGB values
    label: int                  # segment ID


def _torch_connected_components(mask: torch.Tensor) -> torch.Tensor:
    """Connected components via 4-direction min + pointer jumping.

    Pointer jumping doubles propagation distance each iteration, reducing
    convergence from O(diameter) to O(log(diameter)) iterations.
    """
    H, W = mask.shape
    device = mask.device
    n = H * W
    flat_mask = mask.reshape(-1)
    parent = torch.arange(n, device=device, dtype=torch.long)
    parent[~flat_mask] = -1

    idx = torch.arange(n, device=device)
    row, col = idx // W, idx % W
    up_idx = (idx - W).clamp(0, n - 1)
    down_idx = (idx + W).clamp(0, n - 1)
    left_idx = (idx - 1).clamp(0, n - 1)
    right_idx = (idx + 1).clamp(0, n - 1)
    up_valid = flat_mask & (row > 0) & flat_mask[up_idx]
    down_valid = flat_mask & (row < H - 1) & flat_mask[down_idx]
    left_valid = flat_mask & (col > 0) & flat_mask[left_idx]
    right_valid = flat_mask & (col < W - 1) & flat_mask[right_idx]

    max_iter = int(math.ceil(math.log2(max(H, W) + 1))) + 5
    for _ in range(max_iter):
        old_parent = parent.clone()
        p = parent.clone()
        p = torch.where(up_valid, torch.minimum(p, parent[up_idx]), p)
        p = torch.where(down_valid, torch.minimum(p, parent[down_idx]), p)
        p = torch.where(left_valid, torch.minimum(p, parent[left_idx]), p)
        p = torch.where(right_valid, torch.minimum(p, parent[right_idx]), p)
        parent = p
        valid = parent >= 0
        jumped = parent[parent.clamp(0)]
        parent = torch.where(valid, jumped, parent)
        if torch.equal(parent, old_parent):
            break

    labels = parent.reshape(H, W)
    labels[~mask] = -1
    unique_labels = torch.unique(labels[labels >= 0])
    if unique_labels.numel() == 0:
        return labels
    mapping = torch.full((n,), -1, device=device, dtype=torch.long)
    mapping[unique_labels] = torch.arange(len(unique_labels), device=device)
    return torch.where(labels >= 0, mapping[labels.clamp(0)], labels)


def _labels_to_segments(image_rgb: torch.Tensor, labels: torch.Tensor,
                        min_pixels: int, max_segments: int) -> List[Segment]:
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
        segments.append(Segment(mask=local_mask, bbox=(y0, y1, x0, x1),
                                centroid=centroid, coords=coords,
                                colors=colors, label=label_val.item()))
        if len(segments) >= max_segments:
            break
    return segments


def extract_segments(img: torch.Tensor, gate: torch.Tensor,
                     gate_threshold: float = 0.5, min_pixels: int = 20,
                     max_segments: int = 50) -> List[Segment]:
    """Extract segments from low-gate contour regions."""
    H, W, _ = img.shape
    gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
    gray_norm = (gray - gray.mean()) / (gray.std() + 1e-8)
    contours = torch.abs(gray_norm) > 1.0
    low_gate = gate < gate_threshold
    eligible = contours & low_gate
    if eligible.sum() < min_pixels * 3:
        if contours.any():
            gate_on_contours = gate[contours]
            gate_median = torch.median(gate_on_contours)
            eligible = contours & (gate < gate_median)
    labels = _torch_connected_components(eligible)
    return _labels_to_segments(img, labels, min_pixels, max_segments)


def compute_segment_signature(segment: Segment, fiedler: torch.Tensor) -> torch.Tensor:
    """Compute 4D spectral signature: [mean_f, std_f, log_size, aspect]."""
    y0, y1, x0, x1 = segment.bbox
    H, W = fiedler.shape
    device = fiedler.device
    xs = segment.coords[:, 0].long().clamp(0, W-1)
    ys = segment.coords[:, 1].long().clamp(0, H-1)
    fiedler_vals = fiedler[ys, xs]
    mean_f = fiedler_vals.mean()
    std_f = fiedler_vals.std() + 1e-8
    size = torch.tensor(len(fiedler_vals), device=device, dtype=torch.float32)
    aspect = torch.tensor((x1 - x0) / max(y1 - y0, 1), device=device, dtype=torch.float32)
    return torch.stack([mean_f, std_f, torch.log(size + 1), aspect])


def match_segments(sigs_A: torch.Tensor, sigs_B: torch.Tensor) -> torch.Tensor:
    """Match A→B segments by L2 distance in normalized signature space. Returns (Q,) indices."""
    q_norm = (sigs_A - sigs_A.mean(0)) / (sigs_A.std(0) + 1e-8)
    s_norm = (sigs_B - sigs_B.mean(0)) / (sigs_B.std(0) + 1e-8)
    diff = q_norm.unsqueeze(1) - s_norm.unsqueeze(0)
    distances = (diff ** 2).sum(dim=-1)
    _, indices = torch.topk(distances, k=1, dim=1, largest=False)
    return indices.squeeze(-1)


def scatter_to_layer(coords: torch.Tensor, colors: torch.Tensor,
                     H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scatter points to a layer buffer with occupancy mask."""
    device = coords.device
    layer = torch.zeros((H, W, 3), device=device, dtype=colors.dtype)
    mask = torch.zeros((H, W), device=device, dtype=colors.dtype)
    if coords.shape[0] == 0:
        return layer, mask
    px = coords[:, 0].round().long().clamp(0, W - 1)
    py = coords[:, 1].round().long().clamp(0, H - 1)
    layer[py, px] = colors
    mask[py, px] = 1.0
    return layer, mask


def composite_layers_hadamard(shadow: torch.Tensor, shadow_mask: torch.Tensor,
                              front: torch.Tensor, front_mask: torch.Tensor) -> torch.Tensor:
    """Hadamard composite: front overwrites shadow where both exist."""
    fm = front_mask.unsqueeze(-1)
    sm = shadow_mask.unsqueeze(-1)
    return shadow * sm * (1 - fm) + front * fm


def cross_attention_transfer(
    image_A: torch.Tensor,          # (H_A, W_A, 3) target: defines WHERE
    image_B: torch.Tensor,          # (H_B, W_B, 3) source: provides WHAT
    fiedler_A: torch.Tensor,        # (H_A, W_A) Fiedler of A
    fiedler_B: torch.Tensor,        # (H_B, W_B) Fiedler of B
    gate_A: torch.Tensor,           # (H_A, W_A) gate values for A
    contours_A: torch.Tensor,       # (H_A, W_A) contour mask for A
    effect_strength: float = 1.0,
    shadow_offset: float = 7.0,
    translation_strength: float = 20.0,
    min_pixels: int = 20,
    max_segments: int = 50
) -> torch.Tensor:
    """
    Segment-based cross-attention transfer: extract, match, transplant, scatter.

    Pipeline:
    1. Extract connected-component segments from low-gate contours of A and B
    2. Compute 4D spectral signature per segment
    3. Match A→B by L2 distance in normalized signature space
    4. Transplant matched B fragments to A's centroid locations
    5. 90° rotation around centroids + translation toward voids
    6. Shadow/front color transforms
    7. Scatter-rasterize to layers + Hadamard composite
    """
    H_A, W_A, _ = image_A.shape
    device, dtype = image_A.device, image_A.dtype
    eff = effect_strength

    # 1. Extract segments from A
    segments_A = extract_segments(image_A, gate_A, gate_threshold=0.5,
                                  min_pixels=min_pixels, max_segments=max_segments)
    if not segments_A:
        return image_A

    # 2. Compute gate_B, extract segments from B
    thresh_B = adaptive_threshold(fiedler_B, 40)
    gate_B = fiedler_gate(fiedler_B, thresh_B, 10.0)
    segments_B = extract_segments(image_B, gate_B, gate_threshold=0.5,
                                  min_pixels=min_pixels, max_segments=max_segments * 2)
    if not segments_B:
        return image_A

    # 3. Compute spectral signatures and match
    sigs_A = torch.stack([compute_segment_signature(s, fiedler_A) for s in segments_A])
    sigs_B = torch.stack([compute_segment_signature(s, fiedler_B) for s in segments_B])
    match_idx = match_segments(sigs_A, sigs_B)

    # 4. Transplant matched B segments to A's centroid locations (vectorized)
    centroids_A = torch.stack([s.centroid for s in segments_A])
    centroids_B = torch.stack([s.centroid for s in segments_B])
    offsets = centroids_A - centroids_B[match_idx]

    all_coords = []
    all_colors = []
    all_centroids = []
    for i in range(len(segments_A)):
        seg_B = segments_B[match_idx[i]]
        all_coords.append(seg_B.coords + offsets[i].unsqueeze(0))
        all_colors.append(seg_B.colors)
        all_centroids.append(centroids_A[i].unsqueeze(0).expand(seg_B.coords.shape[0], -1))

    if not all_coords:
        return image_A

    coords = torch.cat(all_coords, dim=0)
    colors = torch.cat(all_colors, dim=0)
    centroids = torch.cat(all_centroids, dim=0)

    # 5. 90° rotation around centroids
    rel = coords - centroids
    rotated = torch.stack([-rel[:, 1], rel[:, 0]], dim=-1) + centroids

    # 6. Translation
    trans = translation_strength * eff
    base_translation = torch.tensor([trans * 0.5, trans * 0.7], device=device, dtype=dtype)
    shadow_extra = torch.tensor([shadow_offset * eff, shadow_offset * eff], device=device, dtype=dtype)

    front_coords = rotated + base_translation.unsqueeze(0)
    shadow_coords = rotated + base_translation.unsqueeze(0) + shadow_extra.unsqueeze(0)

    # 7. Color transforms
    shadow_colors = compute_shadow_colors(colors, eff)
    front_colors = compute_front_colors(colors, eff)

    # 8. Scatter to layers
    shadow_layer, shadow_mask = scatter_to_layer(shadow_coords, shadow_colors, H_A, W_A)
    front_layer, front_mask = scatter_to_layer(front_coords, front_colors, H_A, W_A)

    # 9. Hadamard composite
    segment_composite = composite_layers_hadamard(shadow_layer, shadow_mask, front_layer, front_mask)

    # 10. Blend with base image
    combined_mask = torch.clamp(shadow_mask + front_mask, 0, 1).unsqueeze(-1)
    return image_A * (1 - combined_mask) + segment_composite * combined_mask


# ============================================================
# 5. CYCLIC COLOR TRANSFORM
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
    luminance = to_grayscale(rgb)

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
# 5. CHEBYSHEV-BASED SPECTRAL OPERATIONS (NO EIGENVECTOR COMPUTATION)
# ============================================================

def compute_spectral_gate_chebyshev(
    image_rgb: torch.Tensor,        # (H, W, 3) RGB image OR (H, W) grayscale
    L: Optional[torch.Tensor] = None,  # Sparse Laplacian (optional, built if None)
    edge_threshold: float = 0.1,    # For Laplacian construction
    sharpness: float = 10.0,        # Gate sigmoid sharpness
    percentile: float = 40.0,       # Percentile for adaptive threshold
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
        carrier = to_grayscale(image_rgb)
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
    fiedler: torch.Tensor,          # (H, W) for complexity modulation
    gate_threshold: float = 0.5,
    dilation_radius: int = 2,
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
            gray = to_grayscale(image_rgb)
        contours = detect_contours(gray, threshold=1.0)

    high_gate = gate > gate_threshold
    high_gate_contours = contours & high_gate

    if not high_gate_contours.any():
        return image_rgb.clone()

    # Compute complexity (inverse modulation: high complexity = LESS thickening)
    complexity = compute_local_spectral_complexity(fiedler, window_size=7)

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
    Single pass of the spectral shader (pure tensor ops, no segment extraction).

    Pipeline:
    1. Fiedler → gate (elementwise sigmoid)
    2. High-gate path: dilate contours via conv2d + grid_sample
    3. Low-gate path: transform via Fiedler gradient direction + grid_sample

    All operations are tensor-to-tensor. No connected components, no pixel
    iteration, no coordinate unpacking. The Fiedler gradient provides local
    direction for transforms (perpendicular = shadow direction).

    Config keys:
        effect_strength: overall effect intensity (0-2)
        gate_threshold: Fiedler threshold for high/low split
        gate_sharpness: sigmoid steepness
        translation_strength: transform magnitude
        shadow_offset: shadow displacement
        dilation_radius: thickening for high-gate regions

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

    # 1. Compute gate (elementwise)
    gate = fiedler_gate(fiedler, gate_threshold, gate_sharpness)

    # Precompute grayscale and contours once for both high-gate and low-gate paths
    gray = to_grayscale(image_rgb)
    contours = detect_contours(gray, threshold=1.0)

    # 2. High-gate path: spectrally-modulated thickening (all tensor ops)
    output = dilate_high_gate_regions(
        image_rgb, gate, fiedler,
        gate_threshold=0.5,
        dilation_radius=dilation_radius,
        modulation_strength=config.get('thicken_modulation', 0.3),
        kernel_sigma_ratio=config.get('kernel_sigma_ratio', 0.6),
        fill_threshold=config.get('fill_threshold', 0.1),
        gray=gray,
        contours=contours
    )

    # 3. Low-gate path: pure tensor transform (no segment extraction!)
    # Uses Fiedler gradient for direction, grid_sample for transform
    output = apply_low_gate_transform(
        output, gate, fiedler, contours,
        shadow_offset=shadow_offset,
        translation_strength=translation_strength,
        effect_strength=effect_strength
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

    Pipeline:
    1. Lanczos Fiedler computation -> gate + spectral_field
    2. High-gate path: dilate contours with spectral complexity modulation
    3. Low-gate path: pure tensor transform via grid_sample (no segment extraction!)

    Config keys:
        effect_strength: overall effect intensity (0-2)
        gate_sharpness: sigmoid steepness
        translation_strength: how far to translate
        shadow_offset: shadow displacement
        dilation_radius: thickening for high-gate regions

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

    # 1. Compute gate and spectral field via Lanczos Fiedler
    gate, spectral_field, L, lambda_2 = compute_spectral_gate_chebyshev(
        image_rgb,
        L=None,
        edge_threshold=0.1,
        sharpness=gate_sharpness,
        percentile=40.0
    )

    # Precompute grayscale and contours once for both high-gate and low-gate paths
    gray = to_grayscale(image_rgb)
    contours = detect_contours(gray, threshold=1.0)

    # 2. High-gate path: spectrally-modulated thickening (all tensor ops)
    output = dilate_high_gate_regions(
        image_rgb, gate, spectral_field,
        gate_threshold=0.5,
        dilation_radius=dilation_radius,
        modulation_strength=config.get('thicken_modulation', 0.3),
        kernel_sigma_ratio=config.get('kernel_sigma_ratio', 0.6),
        fill_threshold=config.get('fill_threshold', 0.1),
        gray=gray,
        contours=contours
    )

    # 3. Low-gate path: pure tensor transform (no segment extraction!)
    output = apply_low_gate_transform(
        output, gate, spectral_field, contours,
        shadow_offset=shadow_offset,
        translation_strength=translation_strength,
        effect_strength=effect_strength
    )

    return output


# ============================================================
# 9. TWO-IMAGE TRANSFER (CROSS-ATTENTION BASED)
# ============================================================

def two_image_shader_pass(
    image_A: torch.Tensor,          # (H_A, W_A, 3) target: defines WHERE
    image_B: torch.Tensor,          # (H_B, W_B, 3) source: provides WHAT
    fiedler_A: torch.Tensor,        # (H_A, W_A) precomputed Fiedler of A
    fiedler_B: torch.Tensor,        # (H_B, W_B) precomputed Fiedler of B
    config: Dict
) -> torch.Tensor:
    """
    Two-image spectral shader: topology from A, content from B.

    CRITICAL: A and B can have DIFFERENT sizes. No rescaling ever.
    Graphs have spectra; rescaling destroys spectra.

    Pipeline:
    1. High-gate path: dilate contours on A (preserves A's structure)
    2. Segment-based cross-attention: extract segments, match by spectral
       signature, transplant B fragments to A locations, 90° rotate, scatter.
    """
    H_A, W_A, _ = image_A.shape
    device = image_A.device

    effect_strength = config.get('effect_strength', 1.0)
    dilation_radius = int(config.get('dilation_radius', 2) * effect_strength)

    # Compute gate and contours for A
    gate_A = fiedler_gate(fiedler_A, adaptive_threshold(fiedler_A, 40), 10.0)
    gray_A = to_grayscale(image_A)
    contours_A = detect_contours(gray_A, threshold=1.0)

    # Step 1: High-gate path on A with spectrally-modulated thickening
    output = dilate_high_gate_regions(
        image_A, gate_A, fiedler_A,
        gate_threshold=0.5,
        dilation_radius=dilation_radius,
        modulation_strength=config.get('thicken_modulation', 0.3),
        kernel_sigma_ratio=config.get('kernel_sigma_ratio', 0.6),
        fill_threshold=config.get('fill_threshold', 0.1)
    )

    # Step 2: Segment-based cross-attention transfer from B to A
    output = cross_attention_transfer(
        output, image_B, fiedler_A, fiedler_B, gate_A, contours_A,
        effect_strength=effect_strength,
        shadow_offset=config.get('shadow_offset', 7.0),
        translation_strength=config.get('translation_strength', 20.0),
        min_pixels=config.get('min_segment_pixels', 20),
        max_segments=config.get('max_segments', 50)
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
    fiedler_A: Optional[torch.Tensor] = None,   # Two-image only: Fiedler of A
    fiedler_B: Optional[torch.Tensor] = None,   # Two-image only: Fiedler of B
    config: Optional[Dict] = None,
    use_chebyshev: Optional[bool] = None        # None = auto-detect, True = force Chebyshev, False = force explicit
) -> torch.Tensor:
    """
    Unified forward pass for spectral shader.

    Handles both single-image and two-image cases:
    - Single image (image_B is None): self-shading via spectral_shader_pass
    - Two images: topology from A, content from B via two_image_shader_pass

    IMPORTANT: In single-image mode, Fiedler and threshold are ALWAYS recomputed
    from the current image. This ensures correct behavior in autoregressive loops
    where the image changes each iteration. The fiedler_A parameter is IGNORED
    in single-image mode.

    In two-image mode, fiedler_A/fiedler_B can be precomputed since images A and B
    don't change during the call.

    Args:
        image_A: Primary/target image tensor (H, W, 3)
        image_B: Optional source image for content transfer (H_B, W_B, 3)
        fiedler_A: Two-image mode only: precomputed Fiedler for A (computed if None)
        fiedler_B: Two-image mode only: precomputed Fiedler for B (computed if None)
        config: Shader configuration dict (uses defaults if None)
        use_chebyshev: Whether to use Chebyshev filtering instead of explicit eigenvectors

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
        # ALWAYS recompute Fiedler from current image - never use stale cached values
        fiedler_A = _compute_fiedler_from_tensor(image_A)
        # ALWAYS recompute threshold from current Fiedler - fixes AR staleness bug
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


