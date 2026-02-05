"""
Spectral shader v10.5: Unified thickening + shadow with cross-attention

BOTH effects at once:
  - HIGH GATE: Asymmetric thickening
  - LOW GATE: Shadow placement with geometry retrieval

Supports:
  - Self-attention: (target, target) - retrieve geometry from self
  - Cross-attention: (target, sample) - retrieve geometry from sample image

Cross-attention via spectral embedding:
  - Compute k-dimensional spectral signature per sample component
  - At each low-gate seed, find best-matching sample component
  - Retrieve that component's geometry, rotate, translate, color-transform
  - Scatter to target without overwriting target's graph shape
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import math
from spectral_ops_fast import (
    build_weighted_image_laplacian,
    lanczos_k_eigenvectors,
    heat_diffusion_sparse,
    DEVICE
)


# ============================================================
# Spectral embeddings for cross-attention
# ============================================================

def compute_spectral_embedding(carrier: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    Compute k-dimensional spectral embedding for each pixel.
    Returns (H*W, k) tensor of spectral coordinates.
    """
    H, W = carrier.shape
    device = carrier.device

    L = build_weighted_image_laplacian(carrier, edge_threshold=0.1)
    try:
        phi_np, _ = lanczos_k_eigenvectors(L, num_eigenvectors=k, num_iterations=50)
        phi = torch.tensor(phi_np, dtype=torch.float32, device=device)
    except:
        # Fallback: use carrier as single-dimensional embedding
        phi = carrier.flatten().unsqueeze(1)

    return phi  # (H*W, k)


def compute_component_signatures(
    components: list[torch.Tensor],  # list of (N_i, 2) coords
    phi: torch.Tensor,               # (H*W, k) spectral embedding
    W: int,                          # image width for coord→index
) -> list[torch.Tensor]:
    """
    Compute k-dimensional spectral signature for each connected component.
    Signature = mean spectral vector over component pixels.
    """
    signatures = []
    for coords in components:
        # Convert (y, x) to flat indices
        indices = coords[:, 0].long() * W + coords[:, 1].long()
        # Mean spectral vector
        comp_phi = phi[indices].mean(dim=0)  # (k,)
        signatures.append(comp_phi)
    return signatures


def cross_attention_retrieve(
    seed_phi: torch.Tensor,           # (k,) spectral vector at seed
    component_signatures: list[torch.Tensor],  # list of (k,) signatures
) -> int:
    """
    Find best-matching component via spectral attention.
    Returns index of component with highest |dot(seed, signature)|.
    """
    if not component_signatures:
        return -1

    scores = torch.stack([
        (seed_phi * sig).sum() for sig in component_signatures
    ])
    return scores.abs().argmax().item()


@dataclass
class UnifiedConfig:
    """All parameters for unified shader."""
    # Contour detection
    contour_threshold: float = 0.7

    # Gate parameters (threshold = -bias / (gamma - 1))
    # With gamma = 1.3: threshold = -bias / 0.3
    # High: bias = -0.18 → threshold = 0.6, fires when gate > 0.6
    # Low: bias = -0.12 → threshold = 0.4, fires when gate < 0.4
    gate_gamma_high: float = 1.3
    gate_bias_high: float = -0.18   # threshold 0.6
    gate_gamma_low: float = 1.3
    gate_bias_low: float = -0.12    # threshold 0.4 (NEGATIVE!)

    # === THICKENING (high gate) ===
    max_radius: int = 6
    r_inside: float = 2.0
    r_outside: float = 4.0
    grad_mod_inside: float = 1.0
    grad_mod_outside: float = 3.0
    asymmetry_mode: str = "edge_normal"

    # === SHADOW (low gate) ===
    min_segment_size: int = 8
    max_segments: int = 50
    shadow_distance: float = 20.0
    shadow_offset: float = 8.0  # additional offset for shadow layer

    # Per-fragment rotation: make each fragment orthogonal to LOCAL geometry
    # NOT a fixed angle - computed from local tangent at seed
    rotation_base: float = math.pi / 2  # base orthogonal rotation
    rotation_from_tangent: bool = True  # rotate to be orthogonal to local tangent

    # Color transforms
    shadow_hue_shift: float = 0.3
    shadow_blend: float = 0.5
    front_hue_shift: float = 0.2
    front_blend: float = 0.4

    # Cross-attention
    spectral_k: int = 8  # spectral embedding dimension
    cross_color_modulation: float = 0.6  # blend sample color with target-relative transform


# ============================================================
# Patch-based segment handling
# ============================================================

@dataclass
class SegmentPatch:
    """A segment as a mini-image patch."""
    mask: torch.Tensor      # (h, w) bool - which pixels are part of segment
    colors: torch.Tensor    # (h, w, 3) - colors at each position
    bbox: tuple             # (y0, y1, x0, x1) in original image
    centroid: torch.Tensor  # (2,) - center in original image coords
    principal_axis: torch.Tensor  # (2,) - direction of most variance


def extract_segment_patch(
    coords: torch.Tensor,   # (N, 2) segment coordinates
    img: torch.Tensor,      # (H, W, 3) source image
) -> SegmentPatch:
    """Extract a segment as a rectangular patch with mask and colors."""
    H, W = img.shape[:2]
    device = img.device

    y_coords = coords[:, 0].long()
    x_coords = coords[:, 1].long()

    # Bounding box
    y0, y1 = y_coords.min().item(), y_coords.max().item() + 1
    x0, x1 = x_coords.min().item(), x_coords.max().item() + 1
    h, w = y1 - y0, x1 - x0

    # Create patch mask and colors
    mask = torch.zeros(h, w, dtype=torch.bool, device=device)
    colors = torch.zeros(h, w, 3, device=device)

    # Fill in segment pixels
    local_y = y_coords - y0
    local_x = x_coords - x0
    mask[local_y, local_x] = True
    colors[local_y, local_x] = img[y_coords, x_coords]

    # Centroid in original coords
    centroid = coords.float().mean(dim=0)

    # Principal axis
    relative = coords.float() - centroid
    if coords.shape[0] > 2:
        cov = relative.T @ relative / coords.shape[0]
        _, eigenvectors = torch.linalg.eigh(cov)
        principal_axis = eigenvectors[:, -1]
    else:
        principal_axis = torch.tensor([1.0, 0.0], device=device)

    return SegmentPatch(
        mask=mask,
        colors=colors,
        bbox=(y0, y1, x0, x1),
        centroid=centroid,
        principal_axis=principal_axis,
    )


def rotate_patch(patch: SegmentPatch, angle: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rotate a patch by angle radians using COORDINATE ROTATION.
    No scaling or shearing - each pixel moves to its rotated position.

    Returns rotated (mask, colors) tensors with same-size output.
    """
    device = patch.mask.device
    h, w = patch.mask.shape

    # Get mask pixel coordinates
    mask_coords = patch.mask.nonzero(as_tuple=False).float()  # (N, 2) - (y, x)
    if mask_coords.shape[0] == 0:
        return patch.mask, patch.colors

    # Get colors at those positions
    colors_at_mask = patch.colors[patch.mask]  # (N, 3)

    # Center of patch
    center_y, center_x = h / 2.0, w / 2.0

    # Relative coordinates
    rel_y = mask_coords[:, 0] - center_y
    rel_x = mask_coords[:, 1] - center_x

    # Rotation matrix
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Rotate coordinates (NOT the inverse - we want forward rotation)
    rot_y = cos_a * rel_y - sin_a * rel_x
    rot_x = sin_a * rel_y + cos_a * rel_x

    # Compute new bounding box
    min_y, max_y = rot_y.min().item(), rot_y.max().item()
    min_x, max_x = rot_x.min().item(), rot_x.max().item()

    # New dimensions (add padding for safety)
    new_h = int(max_y - min_y) + 3
    new_w = int(max_x - min_x) + 3

    # New center
    new_center_y = new_h / 2.0
    new_center_x = new_w / 2.0

    # Shift rotated coords to new image space
    new_y = (rot_y - min_y + 1).round().long()  # +1 for padding
    new_x = (rot_x - min_x + 1).round().long()

    # Clamp to valid range
    new_y = new_y.clamp(0, new_h - 1)
    new_x = new_x.clamp(0, new_w - 1)

    # Create output tensors
    rotated_mask = torch.zeros(new_h, new_w, dtype=torch.bool, device=device)
    rotated_colors = torch.zeros(new_h, new_w, 3, device=device)

    # Scatter rotated pixels
    rotated_mask[new_y, new_x] = True
    rotated_colors[new_y, new_x] = colors_at_mask

    return rotated_mask, rotated_colors


# ============================================================
# Color transforms (from v6)
# ============================================================

def cyclic_color_transform(colors: torch.Tensor, hue_shift: float, blend: float) -> torch.Tensor:
    """
    Cyclic color transform with no fixed points at black/white.
    Works on (*, 3) tensors.
    """
    shape = colors.shape
    flat = colors.reshape(-1, 3)

    lum = 0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2]
    theta = 2 * math.pi * lum + hue_shift * math.pi

    cyc_r = 0.5 + 0.4 * torch.sin(theta)
    cyc_g = 0.5 + 0.4 * torch.sin(theta + 2 * math.pi / 3)
    cyc_b = 0.5 + 0.4 * torch.sin(theta + 4 * math.pi / 3)
    cyc = torch.stack([cyc_r, cyc_g, cyc_b], dim=-1)

    result = flat * (1 - blend) + cyc * blend
    return result.reshape(shape).clamp(0, 1)


def shadow_color(colors: torch.Tensor, cfg: UnifiedConfig) -> torch.Tensor:
    """Shadow: cyclic transform + blue bias."""
    transformed = cyclic_color_transform(colors, cfg.shadow_hue_shift, cfg.shadow_blend)
    transformed[..., 0] *= 0.7  # reduce red
    transformed[..., 2] = transformed[..., 2] * 0.7 + 0.3  # add blue
    return transformed.clamp(0, 1)


def front_color(colors: torch.Tensor, cfg: UnifiedConfig) -> torch.Tensor:
    """Front: cyclic transform + cyan bias."""
    transformed = cyclic_color_transform(colors, cfg.front_hue_shift, cfg.front_blend)
    transformed[..., 0] *= 0.6  # reduce red
    transformed[..., 1] = transformed[..., 1] * 0.8 + 0.2  # add green
    transformed[..., 2] = transformed[..., 2] * 0.8 + 0.15  # add blue
    return transformed.clamp(0, 1)


def cross_color_modulate(
    sample_colors: torch.Tensor,  # colors from sample component
    target_context: torch.Tensor,  # local color context from target
    blend: float,
) -> torch.Tensor:
    """
    Modulate sample colors relative to target context.
    Bends the sample's colors toward the target's local palette.
    """
    # Get luminance-based hue shift from target context
    target_lum = 0.299 * target_context[0] + 0.587 * target_context[1] + 0.114 * target_context[2]
    phase = target_lum * math.pi  # target luminance determines color phase

    # Apply cyclic transform with target-derived phase
    transformed = cyclic_color_transform(sample_colors, phase.item(), blend)

    # Blend with original sample colors
    return (sample_colors * (1 - blend) + transformed * blend).clamp(0, 1)


def compute_local_tangent_field(carrier: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute local tangent direction (along edges, not across them).
    Tangent is perpendicular to gradient.

    Returns (tan_y, tan_x) each (H, W).
    """
    H, W = carrier.shape

    # Gradient (perpendicular to edges)
    grad_y = torch.zeros_like(carrier)
    grad_x = torch.zeros_like(carrier)
    grad_y[:-1, :] = carrier[1:, :] - carrier[:-1, :]
    grad_x[:, :-1] = carrier[:, 1:] - carrier[:, :-1]

    # Tangent = rotate gradient 90° (along edges)
    tan_y = -grad_x
    tan_x = grad_y

    # Normalize
    mag = (tan_y**2 + tan_x**2).sqrt().clamp(min=1e-8)
    return tan_y / mag, tan_x / mag


def compute_optimal_placement(
    fragment_coords: torch.Tensor,  # (N, 2) relative coords
    fragment_axis: torch.Tensor,    # (2,) principal axis
    seed_pos: tuple,                # (y, x) seed position
    tangent_field: tuple,           # (tan_y, tan_x) each (H, W)
    geometry_mask: torch.Tensor,    # (H, W) bool - existing geometry
    distance_field: torch.Tensor,   # (H, W) distance to geometry
    void_direction: torch.Tensor,   # (H, W, 2) direction toward void
    H: int, W: int,
    num_angles: int = 12,
    num_distances: int = 8,
    max_distance: float = 30.0,
    w_source_parallel: float = 1.0,
    w_local_parallel: float = 0.5,
    w_intersection: float = 2.0,
    w_void: float = 0.3,
) -> tuple[float, float, torch.Tensor]:
    """
    Find optimal (rotation, translation) placement via energy minimization.

    Returns (best_angle, best_distance, final_coords).

    ALL TENSOR OPS - no Python loops over candidates.
    """
    device = fragment_coords.device
    N = fragment_coords.shape[0]
    seed_y, seed_x = seed_pos
    tan_y, tan_x = tangent_field

    # Get void direction at seed
    icy, icx = int(min(seed_y, H-1)), int(min(seed_x, W-1))
    vd = void_direction[icy, icx]  # (2,)

    # === 1. GENERATE CANDIDATE GRID ===
    angles = torch.linspace(0, 2 * math.pi * (1 - 1/num_angles), num_angles, device=device)  # (A,)
    distances = torch.linspace(5.0, max_distance, num_distances, device=device)  # (T,)

    A, T = num_angles, num_distances

    # Rotation matrices for all angles
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    # R[a] = [[cos, -sin], [sin, cos]]

    # Rotate fragment coords for each angle: (A, N, 2)
    # rotated_y = cos * y - sin * x
    # rotated_x = sin * y + cos * x
    fy, fx = fragment_coords[:, 0], fragment_coords[:, 1]  # (N,)
    rotated_y = cos_a.unsqueeze(1) * fy.unsqueeze(0) - sin_a.unsqueeze(1) * fx.unsqueeze(0)  # (A, N)
    rotated_x = sin_a.unsqueeze(1) * fy.unsqueeze(0) + cos_a.unsqueeze(1) * fx.unsqueeze(0)  # (A, N)

    # Rotate fragment axis for each angle: (A, 2)
    ax_y, ax_x = fragment_axis[0], fragment_axis[1]
    rotated_axis_y = cos_a * ax_y - sin_a * ax_x  # (A,)
    rotated_axis_x = sin_a * ax_y + cos_a * ax_x  # (A,)

    # Translate for each distance: (A, T, N, 2)
    # placed = rotated + seed + distance * void_dir
    trans_y = distances * vd[0]  # (T,)
    trans_x = distances * vd[1]  # (T,)

    placed_y = rotated_y.unsqueeze(1) + seed_y + trans_y.unsqueeze(0).unsqueeze(2)  # (A, T, N)
    placed_x = rotated_x.unsqueeze(1) + seed_x + trans_x.unsqueeze(0).unsqueeze(2)  # (A, T, N)

    # Clip to valid coords for gathering
    placed_y_idx = placed_y.long().clamp(0, H-1)
    placed_x_idx = placed_x.long().clamp(0, W-1)

    # === 2. GATHER CONTEXT AT ALL PLACEMENTS ===
    # Local tangent at placed positions
    local_tan_y = tan_y[placed_y_idx, placed_x_idx]  # (A, T, N)
    local_tan_x = tan_x[placed_y_idx, placed_x_idx]  # (A, T, N)

    # Geometry at placed positions
    on_geom = geometry_mask[placed_y_idx, placed_x_idx]  # (A, T, N) bool

    # Distance at placed positions
    dist_at = distance_field[placed_y_idx, placed_x_idx]  # (A, T, N)

    # === 3. COMPUTE ENERGIES ===

    # E_source_parallel: |dot(rotated_axis, original_axis)|
    # This is the same for all T, only depends on angle
    source_dot = rotated_axis_y * ax_y + rotated_axis_x * ax_x  # (A,)
    E_source = source_dot.abs().unsqueeze(1).expand(A, T)  # (A, T)

    # E_local_parallel: mean over N of |dot(rotated_axis, local_tangent)|
    # rotated_axis: (A, 2) -> (A, 1, 1, 2)
    # local_tan: (A, T, N, 2)
    dot_local = (rotated_axis_y.view(A, 1, 1) * local_tan_y +
                 rotated_axis_x.view(A, 1, 1) * local_tan_x)  # (A, T, N)
    E_local = dot_local.abs().mean(dim=2)  # (A, T)

    # E_intersection: count of pixels on geometry, discounted for orthogonal crossing
    orthog_discount = 1.0 - dot_local.abs()  # high when orthogonal
    intersection_penalty = on_geom.float() * (1.0 - 0.3 * orthog_discount)  # (A, T, N)
    E_intersection = intersection_penalty.sum(dim=2)  # (A, T)

    # E_void: reward for being in empty space (high distance = good)
    E_void = -dist_at.mean(dim=2)  # (A, T) - negative because we minimize

    # === 4. TOTAL ENERGY ===
    E_total = (w_source_parallel * E_source +
               w_local_parallel * E_local +
               w_intersection * E_intersection +
               w_void * E_void)  # (A, T)

    # === 5. SELECT MINIMUM ===
    flat_idx = E_total.argmin()
    best_a = (flat_idx // T).item()
    best_t = (flat_idx % T).item()

    best_angle = angles[best_a].item()
    best_distance = distances[best_t].item()

    # Compute final coords at best placement
    final_y = rotated_y[best_a] + seed_y + best_distance * vd[0]
    final_x = rotated_x[best_a] + seed_x + best_distance * vd[1]
    final_coords = torch.stack([final_y, final_x], dim=1)  # (N, 2)

    return best_angle, best_distance, final_coords


# ============================================================
# Scatter patch to output
# ============================================================

def scatter_patch(
    output: torch.Tensor,           # (H, W, 3) image to write to
    mask: torch.Tensor,             # (h, w) patch mask
    colors: torch.Tensor,           # (h, w, 3) patch colors
    position: tuple,                # (y, x) top-left position in output
    preserve_mask: torch.Tensor = None,  # (H, W) bool - don't overwrite these
) -> torch.Tensor:
    """Scatter a rotated patch to output at given position."""
    H, W = output.shape[:2]
    h, w = mask.shape
    py, px = int(position[0]), int(position[1])

    # Compute valid region (handle clipping at edges)
    src_y0 = max(0, -py)
    src_x0 = max(0, -px)
    src_y1 = min(h, H - py)
    src_x1 = min(w, W - px)

    dst_y0 = max(0, py)
    dst_x0 = max(0, px)
    dst_y1 = min(H, py + h)
    dst_x1 = min(W, px + w)

    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return output

    # Get source region
    src_mask = mask[src_y0:src_y1, src_x0:src_x1]
    src_colors = colors[src_y0:src_y1, src_x0:src_x1]

    # Apply to output
    if preserve_mask is not None:
        # Don't overwrite preserved pixels
        dst_preserve = preserve_mask[dst_y0:dst_y1, dst_x0:dst_x1]
        write_mask = src_mask & ~dst_preserve
    else:
        write_mask = src_mask

    # Write pixels
    output[dst_y0:dst_y1, dst_x0:dst_x1][write_mask] = src_colors[write_mask]

    return output


# ============================================================
# Asymmetric thickening (from v10.2)
# ============================================================

def compute_edge_normal_field(carrier: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized edge normal (gradient direction)."""
    grad_y = torch.zeros_like(carrier)
    grad_x = torch.zeros_like(carrier)
    grad_y[:-1, :] = carrier[1:, :] - carrier[:-1, :]
    grad_x[:, :-1] = carrier[:, 1:] - carrier[:, :-1]
    mag = (grad_y**2 + grad_x**2).sqrt().clamp(min=1e-8)
    return grad_y / mag, grad_x / mag


def generate_offset_kernel(max_radius: int, device) -> torch.Tensor:
    """Generate circular offset kernel."""
    offsets = []
    for dy in range(-max_radius, max_radius + 1):
        for dx in range(-max_radius, max_radius + 1):
            if dy*dy + dx*dx <= max_radius*max_radius and (dy != 0 or dx != 0):
                offsets.append([dy, dx])
    return torch.tensor(offsets, device=device, dtype=torch.float32)


def thicken_asymmetric(
    contour_coords: torch.Tensor,
    contour_colors: torch.Tensor,
    carrier: torch.Tensor,
    carrier_grad: torch.Tensor,
    H: int, W: int,
    cfg: UnifiedConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric thickening with inside/outside radius difference."""
    device = contour_coords.device
    N = contour_coords.shape[0]

    if N == 0:
        return contour_coords, contour_colors

    # Generate offset kernel
    offsets = generate_offset_kernel(cfg.max_radius, device)
    K = offsets.shape[0]

    # Candidate coordinates
    candidates = contour_coords.unsqueeze(1).float() + offsets.unsqueeze(0)
    candidates_flat = candidates.reshape(-1, 2)
    source_idx = torch.arange(N, device=device).unsqueeze(1).expand(-1, K).reshape(-1)

    # Distances
    offsets_repeated = offsets.unsqueeze(0).expand(N, -1, -1).reshape(-1, 2)
    distances = (offsets_repeated ** 2).sum(dim=1).sqrt()

    # Clip to bounds
    cand_y = candidates_flat[:, 0].long().clamp(0, H-1)
    cand_x = candidates_flat[:, 1].long().clamp(0, W-1)
    source_coords = contour_coords[source_idx]
    src_y = source_coords[:, 0].long().clamp(0, H-1)
    src_x = source_coords[:, 1].long().clamp(0, W-1)

    # Edge-normal mode for inside/outside
    normal_y, normal_x = compute_edge_normal_field(carrier)
    norm_y_src = normal_y[src_y, src_x]
    norm_x_src = normal_x[src_y, src_x]
    dot_product = offsets_repeated[:, 0] * norm_y_src + offsets_repeated[:, 1] * norm_x_src
    inside_mask = dot_product < 0

    # Gradient modulation
    grad_at_cand = carrier_grad[cand_y, cand_x]
    grad_norm = grad_at_cand / (carrier_grad.max() + 1e-8)

    # Effective radius
    effective_radius = torch.zeros_like(distances)
    effective_radius[inside_mask] = cfg.r_inside + cfg.grad_mod_inside * grad_norm[inside_mask]
    effective_radius[~inside_mask] = cfg.r_outside + cfg.grad_mod_outside * grad_norm[~inside_mask]

    # Accept
    in_bounds = (candidates_flat[:, 0] >= 0) & (candidates_flat[:, 0] < H) & \
                (candidates_flat[:, 1] >= 0) & (candidates_flat[:, 1] < W)
    accept = (distances < effective_radius) & in_bounds

    accepted_coords = candidates_flat[accept]
    accepted_colors = contour_colors[source_idx[accept]]

    # Combine with original
    all_coords = torch.cat([contour_coords.float(), accepted_coords], dim=0)
    all_colors = torch.cat([contour_colors, accepted_colors], dim=0)

    return all_coords, all_colors


# ============================================================
# Connected components
# ============================================================

def find_connected_components(mask: torch.Tensor, min_size: int, max_components: int) -> list[torch.Tensor]:
    """Find connected components via flood fill."""
    H, W = mask.shape
    device = mask.device
    remaining = mask.clone()
    components = []

    while remaining.any() and len(components) < max_components:
        seed_idx = remaining.nonzero(as_tuple=False)[0]
        component = torch.zeros(H, W, dtype=torch.bool, device=device)
        frontier = torch.zeros(H, W, dtype=torch.bool, device=device)
        frontier[seed_idx[0], seed_idx[1]] = True

        for _ in range(max(H, W)):
            expanded = frontier.clone()
            expanded[:-1, :] |= frontier[1:, :]
            expanded[1:, :] |= frontier[:-1, :]
            expanded[:, :-1] |= frontier[:, 1:]
            expanded[:, 1:] |= frontier[:, :-1]
            expanded &= remaining
            component |= expanded
            if not (expanded & ~frontier).any():
                break
            frontier = expanded

        remaining &= ~component
        coords = component.nonzero(as_tuple=False)
        if coords.shape[0] >= min_size:
            components.append(coords)

    return components


# ============================================================
# Void direction computation
# ============================================================

def compute_void_direction(contours: torch.Tensor, carrier: torch.Tensor, L) -> torch.Tensor:
    """Compute direction toward void (away from geometry)."""
    H, W = carrier.shape
    device = carrier.device

    # Heat diffusion from contours
    source = contours.float().flatten()
    diffused = heat_diffusion_sparse(L, source, alpha=0.1, iterations=30)
    distance = diffused.reshape(H, W)
    distance = 1.0 - distance / (distance.max() + 1e-8)

    # Gradient of distance field
    grad_y = torch.zeros_like(distance)
    grad_x = torch.zeros_like(distance)
    grad_y[:-1, :] = distance[1:, :] - distance[:-1, :]
    grad_x[:, :-1] = distance[:, 1:] - distance[:, :-1]
    mag = (grad_y**2 + grad_x**2).sqrt().clamp(min=1e-8)

    return torch.stack([grad_y / mag, grad_x / mag], dim=-1)


# ============================================================
# Unified shader
# ============================================================

def shader_unified(
    target: torch.Tensor,
    sample: torch.Tensor = None,  # If None, uses target (self-attention)
    cfg: UnifiedConfig = None,
) -> tuple[torch.Tensor, dict]:
    """
    Unified shader with BOTH thickening and shadow.

    HIGH GATE: Asymmetric thickening (always from target)
    LOW GATE: Shadow with geometry retrieval
      - Self-attention (sample=None or sample=target): retrieve from target
      - Cross-attention (sample≠target): retrieve geometry from sample

    Cross-attention retrieves GEOMETRY from sample at spectrally-matching
    locations, transforms it, and scatters to target without overwriting
    target's graph structure.
    """
    if cfg is None:
        cfg = UnifiedConfig()
    if sample is None:
        sample = target

    is_cross_attention = not torch.equal(target, sample)

    H_t, W_t = target.shape[:2]
    H_s, W_s = sample.shape[:2]
    device = target.device

    carrier_t = target.mean(dim=-1)
    carrier_s = sample.mean(dim=-1)
    L_t = build_weighted_image_laplacian(carrier_t, edge_threshold=0.1)

    # Contour detection on target
    carrier_t_norm = (carrier_t - carrier_t.mean()) / (carrier_t.std() + 1e-8)
    contours_t = carrier_t_norm.abs() > cfg.contour_threshold

    # Contour detection on sample (for component finding)
    carrier_s_norm = (carrier_s - carrier_s.mean()) / (carrier_s.std() + 1e-8)
    contours_s = carrier_s_norm.abs() > cfg.contour_threshold

    # Fiedler for gating (on target)
    try:
        eigenvectors, _ = lanczos_k_eigenvectors(L_t, num_eigenvectors=2, num_iterations=50)
        fiedler = torch.tensor(eigenvectors[:, -1], device=device, dtype=torch.float32).reshape(H_t, W_t)
        fiedler = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-8)
    except:
        fiedler = carrier_t

    # Spectral embeddings for cross-attention
    if is_cross_attention:
        phi_t = compute_spectral_embedding(carrier_t, cfg.spectral_k)
        phi_s = compute_spectral_embedding(carrier_s, cfg.spectral_k)
        # Ensure same dimension
        k_min = min(phi_t.shape[1], phi_s.shape[1])
        phi_t = phi_t[:, :k_min]
        phi_s = phi_s[:, :k_min]
    else:
        phi_t = None
        phi_s = None

    # Gates via gamma/bias parameterization
    # threshold = -bias / (gamma - 1)
    # high fires when gate > threshold, low fires when gate < threshold
    act = (fiedler - fiedler.mean()) / (fiedler.std() + 1e-8)
    gate = torch.sigmoid(act * 10.0)  # squash to [0, 1]

    # High gate: fires for upper portion
    high_modulated = cfg.gate_gamma_high * gate + cfg.gate_bias_high
    high_gate = high_modulated > gate

    # Low gate: fires for lower portion (NEGATIVE bias required!)
    low_modulated = cfg.gate_gamma_low * gate + cfg.gate_bias_low
    low_gate = low_modulated < gate

    stats = {
        'contours': contours_t.sum().item(),
        'high_gate': (contours_t & high_gate).sum().item(),
        'low_gate': (contours_t & low_gate).sum().item(),
        'is_cross_attention': is_cross_attention,
    }

    output = target.clone()

    # === BRANCH A: THICKENING (high gate) - always from target ===
    high_contours = contours_t & high_gate
    high_coords = high_contours.nonzero(as_tuple=False)

    if high_coords.shape[0] > 0:
        high_colors = target[high_coords[:, 0].long(), high_coords[:, 1].long()]

        # Carrier gradient for modulation
        grad_y = torch.zeros_like(carrier_t)
        grad_x = torch.zeros_like(carrier_t)
        grad_y[:-1, :] = carrier_t[1:, :] - carrier_t[:-1, :]
        grad_x[:, :-1] = carrier_t[:, 1:] - carrier_t[:, :-1]
        carrier_grad = (grad_y**2 + grad_x**2).sqrt()

        thick_coords, thick_colors = thicken_asymmetric(
            high_coords, high_colors, carrier_t, carrier_grad, H_t, W_t, cfg
        )

        # Scatter thickened pixels (preserve original contours)
        y_idx = thick_coords[:, 0].round().long().clamp(0, H_t-1)
        x_idx = thick_coords[:, 1].round().long().clamp(0, W_t-1)
        valid = ~contours_t[y_idx, x_idx]  # don't overwrite originals
        output[y_idx[valid], x_idx[valid]] = thick_colors[valid]

        stats['thickened'] = thick_coords.shape[0]

    # === BRANCH B: SHADOW (low gate) with ENERGY-MINIMIZING PLACEMENT ===
    # For cross-attention: find components in SAMPLE, place at target seeds
    # For self-attention: find components in target low-gate region

    # Pre-compute context fields for energy minimization
    tan_y, tan_x = compute_local_tangent_field(carrier_t)
    void_dir_field = compute_void_direction(contours_t, carrier_t, L_t)

    # Distance field (high = far from geometry = void)
    source = contours_t.float().flatten()
    diffused = heat_diffusion_sparse(L_t, source, alpha=0.1, iterations=30)
    distance_field = diffused.reshape(H_t, W_t)
    distance_field = 1.0 - distance_field / (distance_field.max() + 1e-8)

    if is_cross_attention:
        # Pre-segment SAMPLE into connected components
        sample_components = find_connected_components(contours_s, cfg.min_segment_size, cfg.max_segments)
        stats['sample_components'] = len(sample_components)

        if sample_components:
            # Compute spectral signatures for each sample component
            component_sigs = compute_component_signatures(sample_components, phi_s, W_s)

            # Find low-gate seeds in target
            low_contours = contours_t & low_gate
            seed_coords = low_contours.nonzero(as_tuple=False)

            # Subsample seeds for efficiency
            n_seeds = min(seed_coords.shape[0], 20)
            step = max(1, seed_coords.shape[0] // n_seeds)
            seeds = seed_coords[::step]

            stats['seeds_used'] = seeds.shape[0]
            stats['placements'] = []

            for seed in seeds:
                seed_y, seed_x = seed[0].item(), seed[1].item()
                seed_flat = seed_y * W_t + seed_x

                # Get seed's spectral vector
                seed_phi = phi_t[seed_flat]

                # Cross-attention: find best-matching sample component
                best_idx = cross_attention_retrieve(seed_phi, component_sigs)
                if best_idx < 0:
                    continue

                comp_coords = sample_components[best_idx]

                # Extract component as patch from SAMPLE
                patch = extract_segment_patch(comp_coords, sample)

                # === ENERGY-MINIMIZING PLACEMENT ===
                # Find optimal (rotation, translation) via grid search
                best_angle, best_dist, final_coords = compute_optimal_placement(
                    patch.centroid.new_zeros(comp_coords.shape[0], 2).copy_(
                        comp_coords.float() - patch.centroid
                    ),  # relative coords
                    patch.principal_axis,
                    (seed_y, seed_x),
                    (tan_y, tan_x),
                    contours_t,
                    distance_field,
                    void_dir_field,
                    H_t, W_t,
                    num_angles=12,
                    num_distances=8,
                    max_distance=cfg.shadow_distance,
                )

                stats['placements'].append({'angle': best_angle, 'dist': best_dist})

                # Rotate patch at optimal angle
                rotated_mask, rotated_colors = rotate_patch(patch, best_angle)

                # Get target context color at seed for modulation
                target_context = target[seed_y, seed_x]

                # Modulate colors: bend sample colors toward target's palette
                modulated_colors = cross_color_modulate(
                    rotated_colors, target_context, cfg.cross_color_modulation
                )

                # Compute placement position using optimal distance
                vd = void_dir_field[int(min(seed_y, H_t-1)), int(min(seed_x, W_t-1))]
                trans_y = vd[0].item() * best_dist
                trans_x = vd[1].item() * best_dist

                # Center the rotated patch at seed
                rh, rw = rotated_mask.shape
                base_y = seed_y + trans_y - rh / 2
                base_x = seed_x + trans_x - rw / 2

                # SHADOW layer (offset further)
                shadow_y = base_y + vd[0].item() * cfg.shadow_offset
                shadow_x = base_x + vd[1].item() * cfg.shadow_offset
                shadow_colors_mod = shadow_color(modulated_colors, cfg)
                output = scatter_patch(output, rotated_mask, shadow_colors_mod, (shadow_y, shadow_x), preserve_mask=contours_t)

                # FRONT layer
                front_colors_mod = front_color(modulated_colors, cfg)
                output = scatter_patch(output, rotated_mask, front_colors_mod, (base_y, base_x), preserve_mask=contours_t)

    else:
        # Self-attention: use energy-minimizing placement
        low_contours = contours_t & low_gate
        components = find_connected_components(low_contours, cfg.min_segment_size, cfg.max_segments)
        stats['segments'] = len(components)
        stats['placements'] = []

        if components:
            for comp_coords in components:
                # Extract as patch from target
                patch = extract_segment_patch(comp_coords, target)

                # Use centroid as seed for self-attention
                cy, cx = patch.centroid[0].item(), patch.centroid[1].item()

                # === ENERGY-MINIMIZING PLACEMENT ===
                relative_coords = comp_coords.float() - patch.centroid
                best_angle, best_dist, final_coords = compute_optimal_placement(
                    relative_coords,
                    patch.principal_axis,
                    (cy, cx),
                    (tan_y, tan_x),
                    contours_t,
                    distance_field,
                    void_dir_field,
                    H_t, W_t,
                    num_angles=12,
                    num_distances=8,
                    max_distance=cfg.shadow_distance,
                )

                stats['placements'].append({'angle': best_angle, 'dist': best_dist})

                # Rotate patch at optimal angle
                rotated_mask, rotated_colors = rotate_patch(patch, best_angle)

                # Compute placement position using optimal distance
                icy, icx = int(min(cy, H_t-1)), int(min(cx, W_t-1))
                vd = void_dir_field[icy, icx]
                trans_y = vd[0].item() * best_dist
                trans_x = vd[1].item() * best_dist

                # Center the rotated patch
                rh, rw = rotated_mask.shape
                base_y = cy + trans_y - rh / 2
                base_x = cx + trans_x - rw / 2

                # SHADOW layer (offset further)
                shadow_y = base_y + vd[0].item() * cfg.shadow_offset
                shadow_x = base_x + vd[1].item() * cfg.shadow_offset
                shadow_colors_t = shadow_color(rotated_colors, cfg)
                output = scatter_patch(output, rotated_mask, shadow_colors_t, (shadow_y, shadow_x), preserve_mask=contours_t)

                # FRONT layer (masks shadow)
                front_colors_t = front_color(rotated_colors, cfg)
                output = scatter_patch(output, rotated_mask, front_colors_t, (base_y, base_x), preserve_mask=contours_t)

    return output, stats


def demo():
    from PIL import Image
    import numpy as np

    inp = Path("demo_output/inputs")
    out_dir = Path("demo_output")

    # Load images
    images = {}
    for name in ["toof.png", "snek-heavy.png", "red-tonegraph.png"]:
        path = inp / name
        if path.exists():
            img = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
            images[name.split('.')[0]] = torch.tensor(img, dtype=torch.float32, device=DEVICE)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n[v10.5] Unified thickening + shadow with cross-attention")
    print(f"=" * 60)

    # === SELF-ATTENTION TESTS ===
    print(f"\n  [Self-attention tests]")

    if "toof" in images:
        toof_t = images["toof"]

        configs = [
            ("default", UnifiedConfig()),
            ("heavy_thicken", UnifiedConfig(r_outside=6.0, grad_mod_outside=4.0)),
        ]

        for name, cfg in configs:
            output, stats = shader_unified(toof_t, None, cfg)  # sample=None → self-attention

            print(f"\n    {name} (self):")
            print(f"      Contours: {stats['contours']}")
            print(f"      High gate: {stats['high_gate']} → Thickened: {stats.get('thickened', 0)}")
            print(f"      Low gate: {stats['low_gate']} → Segments: {stats.get('segments', 0)}")

            # Show placement statistics for self-attention
            placements = stats.get('placements', [])
            if placements:
                angles = [p['angle'] for p in placements]
                dists = [p['dist'] for p in placements]
                print(f"      Angles: min={min(angles):.2f}, max={max(angles):.2f}, mean={sum(angles)/len(angles):.2f} rad")
                print(f"      Distances: min={min(dists):.1f}, max={max(dists):.1f}")

            out_np = (output.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            out_path = out_dir / f"{timestamp}_self_{name}.png"
            Image.fromarray(out_np).save(out_path)
            print(f"      Saved: {out_path.name}")

    # === CROSS-ATTENTION TESTS ===
    print(f"\n  [Cross-attention tests]")

    cross_pairs = [
        ("snek-heavy", "red-tonegraph"),  # arrows/rectangles decorating snek
        ("toof", "red-tonegraph"),         # tonegraph geometry on toof
        ("snek-heavy", "toof"),            # toof geometry on snek
    ]

    for target_name, sample_name in cross_pairs:
        if target_name not in images or sample_name not in images:
            print(f"    Skipping {target_name} × {sample_name} (missing images)")
            continue

        target = images[target_name]
        sample = images[sample_name]

        # Resize sample to match target if needed
        H_t, W_t = target.shape[:2]
        H_s, W_s = sample.shape[:2]
        if H_t != H_s or W_t != W_s:
            # Use F.interpolate for resizing
            sample_resized = F.interpolate(
                sample.permute(2, 0, 1).unsqueeze(0),
                size=(H_t, W_t),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)
        else:
            sample_resized = sample

        cfg = UnifiedConfig(
            shadow_distance=25.0,
            shadow_offset=10.0,
            cross_color_modulation=0.5,
        )

        output, stats = shader_unified(target, sample_resized, cfg)

        print(f"\n    {target_name} × {sample_name}:")
        print(f"      Target contours: {stats['contours']}")
        print(f"      Sample components: {stats.get('sample_components', 0)}")
        print(f"      Seeds used: {stats.get('seeds_used', 0)}")
        print(f"      Thickened: {stats.get('thickened', 0)}")

        # Show placement statistics
        placements = stats.get('placements', [])
        if placements:
            angles = [p['angle'] for p in placements]
            dists = [p['dist'] for p in placements]
            print(f"      Angles: min={min(angles):.2f}, max={max(angles):.2f}, mean={sum(angles)/len(angles):.2f} rad")
            print(f"      Distances: min={min(dists):.1f}, max={max(dists):.1f}, mean={sum(dists)/len(dists):.1f}")

        out_np = (output.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        out_path = out_dir / f"{timestamp}_cross_{target_name}_by_{sample_name}.png"
        Image.fromarray(out_np).save(out_path)
        print(f"      Saved: {out_path.name}")

    # === MULTI-PASS SELF-ATTENTION ===
    if "toof" in images:
        print(f"\n  [Multi-pass self-attention]")
        current = images["toof"].clone()
        for p in range(4):
            strength = 0.8 ** p
            cfg = UnifiedConfig(
                r_outside=4.0 * strength,
                shadow_distance=20.0 * strength,
                shadow_offset=8.0 * strength,
            )
            current, stats = shader_unified(current, None, cfg)
            print(f"    Pass {p+1}: thickened={stats.get('thickened', 0)}, segments={stats.get('segments', 0)}")

        out_np = (current.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        out_path = out_dir / f"{timestamp}_self_4pass.png"
        Image.fromarray(out_np).save(out_path)
        print(f"    Saved: {out_path.name}")


if __name__ == "__main__":
    demo()
