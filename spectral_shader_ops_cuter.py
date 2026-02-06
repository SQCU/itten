"""Spectral shader operations: streamlined, fused tensor pipeline."""
import torch
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass

# Module-level cached Sobel kernels
_SOBEL_XY: Optional[torch.Tensor] = None
_SOBEL_DEVICE: Optional[torch.device] = None

def _get_sobel_kernels(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    global _SOBEL_XY, _SOBEL_DEVICE
    if _SOBEL_XY is None or _SOBEL_DEVICE != device:
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype) / 8.0
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype) / 8.0
        _SOBEL_XY, _SOBEL_DEVICE = torch.stack([sx, sy]).unsqueeze(1), device
    return _SOBEL_XY

@torch.compile
def compute_gradient_xy(field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute X/Y gradient in ONE conv2d call."""
    sobel = _get_sobel_kernels(field.device, field.dtype)
    f_padded = F.pad(field.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
    grad_xy = F.conv2d(f_padded, sobel)
    return grad_xy[0, 0], grad_xy[0, 1]

@torch.compile
def to_grayscale(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.dim() == 2:
        return 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2] if rgb.shape[-1] == 3 else rgb
    if rgb.dim() == 3:
        if rgb.shape[-1] == 3: return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        if rgb.shape[0] == 3: return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    raise ValueError(f"Unsupported shape: {rgb.shape}")

@torch.compile
def detect_contours(gray: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    return torch.abs((gray - gray.mean()) / (gray.std() + 1e-8)) > threshold

def adaptive_threshold(fiedler: torch.Tensor, percentile: float = 40.0) -> torch.Tensor:
    return torch.quantile(fiedler.flatten(), percentile / 100.0)

@torch.compile
def fill_empty_bins(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    n, device = len(values), values.device
    idx = torch.arange(n, device=device)
    forward_idx, _ = torch.cummax(torch.where(valid, idx, torch.zeros_like(idx)), dim=0)
    filled = values[forward_idx]
    backward_idx = torch.flip(torch.cummin(torch.flip(torch.where(valid, idx, torch.full_like(idx, n-1)), [0]), dim=0)[0], [0])
    return torch.where(forward_idx > 0, filled, values[backward_idx])

@torch.compile
def apply_color_transform(rgb: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    return (rgb * scales + biases).clamp(0, 1)

@torch.compile
def cyclic_color_transform(rgb: torch.Tensor, rot: float = 0.3, contrast: float = 0.5) -> torch.Tensor:
    shape = rgb.shape
    rgb = rgb.reshape(-1, 3) if rgb.dim() == 3 else rgb
    lum = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    phases = torch.tensor([0.0, 2*math.pi/3, 4*math.pi/3], device=rgb.device, dtype=rgb.dtype)
    new_rgb = 0.5 + 0.4 * contrast * torch.sin(2*math.pi*rot * lum.unsqueeze(-1) + phases)
    result = torch.lerp(rgb, new_rgb, rot).clamp(0, 1)
    return result.reshape(shape) if len(shape) == 3 else result

@torch.compile
def compute_shadow_colors(rgb: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    t = cyclic_color_transform(rgb, 0.3 * strength, 0.8 * strength)
    return apply_color_transform(t, torch.tensor([0.7, 1.0, 0.7], device=rgb.device, dtype=rgb.dtype),
                                    torch.tensor([0.0, 0.0, 0.3], device=rgb.device, dtype=rgb.dtype))

@torch.compile
def compute_front_colors(rgb: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    t = cyclic_color_transform(rgb, 0.2 * strength, 0.6 * strength)
    return apply_color_transform(t, torch.tensor([0.6, 0.8, 0.8], device=rgb.device, dtype=rgb.dtype),
                                    torch.tensor([0.0, 0.2, 0.15], device=rgb.device, dtype=rgb.dtype))

# ============================================================
# SEGMENT INFRASTRUCTURE (ported from backup for cross-attention)
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
    Typical: ~15 iterations for 512x512 instead of ~100.
    """
    H, W = mask.shape
    device = mask.device
    n = H * W

    # Work in flat (1D) space for pointer jumping
    flat_mask = mask.reshape(-1)
    parent = torch.arange(n, device=device, dtype=torch.long)
    parent[~flat_mask] = -1

    # Pre-compute neighbor indices + validity (once, not per iteration)
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

        # Vectorized 4-direction min (all directions simultaneously, no order dependency)
        p = parent.clone()
        p = torch.where(up_valid, torch.minimum(p, parent[up_idx]), p)
        p = torch.where(down_valid, torch.minimum(p, parent[down_idx]), p)
        p = torch.where(left_valid, torch.minimum(p, parent[left_idx]), p)
        p = torch.where(right_valid, torch.minimum(p, parent[right_idx]), p)
        parent = p

        # Pointer jumping: parent[i] = parent[parent[i]]
        valid = parent >= 0
        jumped = parent[parent.clamp(0)]
        parent = torch.where(valid, jumped, parent)

        if torch.equal(parent, old_parent):
            break

    # Vectorized label compaction (no Python loop over unique labels)
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


@torch.compile
def compute_local_spectral_complexity(fiedler: torch.Tensor, window_size: int = 5) -> torch.Tensor:
    device, dtype = fiedler.device, fiedler.dtype
    gx, gy = compute_gradient_xy(fiedler)
    grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
    pad, ws2 = window_size // 2, window_size ** 2
    box = torch.ones(1, 1, window_size, window_size, device=device, dtype=dtype) / ws2
    f_4d = fiedler.unsqueeze(0).unsqueeze(0)
    fp, fsp = F.pad(f_4d, (pad,)*4, 'reflect'), F.pad(f_4d**2, (pad,)*4, 'reflect')
    local_var = (F.conv2d(fsp, box) - F.conv2d(fp, box)**2).squeeze().clamp(min=0)
    raw = grad_mag + torch.sqrt(local_var + 1e-8)
    return torch.sigmoid((raw - raw.mean()) / (raw.std() + 1e-8))

def dilate_high_gate_fused(img: torch.Tensor, gate: torch.Tensor, fiedler: torch.Tensor,
                           gray: torch.Tensor, contours: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor, cfg: Dict) -> torch.Tensor:
    H, W, _ = img.shape
    device, dtype = img.device, img.dtype
    eff = cfg.get('effect_strength', 1.0)
    r = max(1, int(cfg.get('dilation_radius', 2) * eff))
    mod, fill_th, sigma_r = cfg.get('thicken_modulation', 0.3), cfg.get('fill_threshold', 0.1), cfg.get('kernel_sigma_ratio', 0.6)

    hgc = contours & (gate > cfg.get('high_gate_threshold', 0.5))
    has_c = hgc.any().float()
    # Inline spectral complexity using pre-computed gx, gy (avoids redundant Sobel)
    grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)
    ws = 7
    pad_c, ws2 = ws // 2, ws ** 2
    box = torch.ones(1, 1, ws, ws, device=device, dtype=dtype) / ws2
    f_4d = fiedler.unsqueeze(0).unsqueeze(0)
    fp, fsp = F.pad(f_4d, (pad_c,)*4, 'reflect'), F.pad(f_4d**2, (pad_c,)*4, 'reflect')
    local_var = (F.conv2d(fsp, box) - F.conv2d(fp, box)**2).squeeze().clamp(min=0)
    raw = grad_mag + torch.sqrt(local_var + 1e-8)
    cplx = torch.sigmoid((raw - raw.mean()) / (raw.std() + 1e-8))
    wt = hgc.float() * (1.0 - mod * cplx).clamp(min=0.05)

    coords = torch.arange(-r, r+1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    kern = torch.exp(-(xx**2 + yy**2) / (2 * (r * sigma_r)**2))
    kern = (kern / kern.max()).view(1, 1, 2*r+1, 2*r+1)
    sel = F.conv2d(F.pad(wt.unsqueeze(0).unsqueeze(0), (r,)*4, 'constant', 0), kern).squeeze()

    fill = (~contours) & (sel > fill_th)
    has_f = fill.any().float()

    mgy = F.pad(sel[1:, :] - sel[:-1, :], (0, 0, 0, 1))
    mgx = F.pad(sel[:, 1:] - sel[:, :-1], (0, 1, 0, 0))
    mm = torch.sqrt(mgx**2 + mgy**2 + 1e-8)
    dx, dy = mgx / mm, mgy / mm

    yb, xb = torch.arange(H, device=device, dtype=dtype), torch.arange(W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(yb, xb, indexing='ij')
    sy, sx = grid_y + 1.2*dy, grid_x + 1.2*dx
    c, s = math.cos(0.03), math.sin(0.03)
    ry, rx = sy - grid_y, sx - grid_x
    sy, sx = grid_y + c*ry - s*rx, grid_x + s*ry + c*rx
    sgrid = torch.stack([2*sx/(W-1)-1, 2*sy/(H-1)-1], dim=-1).unsqueeze(0)

    samp = F.grid_sample(img.permute(2,0,1).unsqueeze(0), sgrid, mode='nearest', padding_mode='border', align_corners=True)
    samp_rgb = samp.squeeze(0).permute(1, 2, 0)
    g3 = (fill.float() * (sel > 0.2).float()).unsqueeze(-1)
    result = torch.lerp(img, samp_rgb, g3)
    return torch.lerp(img, result, has_c * has_f)

def apply_low_gate_fused(img: torch.Tensor, gate: torch.Tensor, fiedler: torch.Tensor,
                         contours: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor, cfg: Dict) -> torch.Tensor:
    H, W, _ = img.shape
    device, dtype = img.device, img.dtype
    eff = cfg.get('effect_strength', 1.0)
    shad_off, trans = cfg.get('shadow_offset', 7.0) * eff, cfg.get('translation_strength', 20.0) * eff

    lgw = (1.0 - gate) * contours.float()
    has_m = (lgw.sum() >= 10).float()

    gm = torch.sqrt(gx**2 + gy**2 + 1e-8)
    tx, ty = gx / gm, gy / gm
    px, py = -ty, tx

    yy, xx = torch.arange(H, device=device, dtype=dtype), torch.arange(W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')

    shad_x, shad_y = grid_x + px * shad_off, grid_y + py * shad_off
    fr_x, fr_y = grid_x + trans * 0.3, grid_y + trans * 0.4

    def norm(sx, sy): return torch.stack([2*sx/(W-1)-1, 2*sy/(H-1)-1], dim=-1)
    comb_grid = torch.stack([norm(shad_x, shad_y), norm(fr_x, fr_y)], dim=0)

    img4 = img.permute(2, 0, 1).unsqueeze(0).expand(2, -1, -1, -1)
    samp = F.grid_sample(img4, comb_grid, mode='bilinear', padding_mode='border', align_corners=True)
    shad_s, fr_s = samp[0].permute(1, 2, 0), samp[1].permute(1, 2, 0)

    shad_c = compute_shadow_colors(shad_s.reshape(-1, 3), eff).reshape(H, W, 3)
    fr_c = compute_front_colors(fr_s.reshape(-1, 3), eff).reshape(H, W, 3)

    w3 = lgw.unsqueeze(-1)
    out = torch.lerp(img, shad_c, w3 * 0.6)
    out = torch.lerp(out, fr_c, w3 * 0.8)
    return torch.lerp(img, out, has_m)

def cross_attention_transfer(img_A: torch.Tensor, img_B: torch.Tensor, f_A: torch.Tensor, f_B: torch.Tensor,
                             gate_A: torch.Tensor, cont_A: torch.Tensor,
                             eff: float = 1.0, shadow_offset: float = 7.0,
                             translation_strength: float = 20.0,
                             min_pixels: int = 20, max_segments: int = 50) -> torch.Tensor:
    """Segment-based cross-attention transfer: extract, match, transplant, scatter."""
    H_A, W_A, _ = img_A.shape
    device, dtype = img_A.device, img_A.dtype

    # 1. Extract segments from A using pre-computed gate
    segments_A = extract_segments(img_A, gate_A, gate_threshold=0.5,
                                  min_pixels=min_pixels, max_segments=max_segments)
    if not segments_A:
        return img_A

    # 2. Compute gate_B, extract segments from B
    thresh_B = adaptive_threshold(f_B, 40.0)
    gate_B = torch.sigmoid((f_B - thresh_B) * 10.0)
    segments_B = extract_segments(img_B, gate_B, gate_threshold=0.5,
                                  min_pixels=min_pixels, max_segments=max_segments * 2)
    if not segments_B:
        return img_A

    # 3. Compute spectral signatures and match A→B
    sigs_A = torch.stack([compute_segment_signature(s, f_A) for s in segments_A])
    sigs_B = torch.stack([compute_segment_signature(s, f_B) for s in segments_B])
    match_idx = match_segments(sigs_A, sigs_B)

    # 4. Transplant matched B segments to A's centroid locations (vectorized)
    centroids_A = torch.stack([s.centroid for s in segments_A])       # (Q, 2)
    centroids_B = torch.stack([s.centroid for s in segments_B])       # (P, 2)
    offsets = centroids_A - centroids_B[match_idx]                    # (Q, 2) — no .item()

    all_coords = []
    all_colors = []
    all_centroids = []
    for i in range(len(segments_A)):
        seg_B = segments_B[match_idx[i]]
        all_coords.append(seg_B.coords + offsets[i].unsqueeze(0))
        all_colors.append(seg_B.colors)
        all_centroids.append(centroids_A[i].unsqueeze(0).expand(seg_B.coords.shape[0], -1))

    if not all_coords:
        return img_A

    coords = torch.cat(all_coords, dim=0)       # (N, 2) as (x, y)
    colors = torch.cat(all_colors, dim=0)        # (N, 3)
    centroids = torch.cat(all_centroids, dim=0)  # (N, 2)

    # 5. 90° rotation around centroids: (x, y) → (-y, x) relative to centroid
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

    # 10. Blend with base image where segments exist
    combined_mask = torch.clamp(shadow_mask + front_mask, 0, 1).unsqueeze(-1)
    return img_A * (1 - combined_mask) + segment_composite * combined_mask

def shader_pass_fused(img: torch.Tensor, fiedler: torch.Tensor, cfg: Dict) -> torch.Tensor:
    gray = to_grayscale(img)
    contours = detect_contours(gray)
    thresh = adaptive_threshold(fiedler, cfg.get('percentile', 40.0))
    gate = torch.sigmoid((fiedler - thresh) * cfg.get('gate_sharpness', 10.0))
    gx, gy = compute_gradient_xy(fiedler)
    out = dilate_high_gate_fused(img, gate, fiedler, gray, contours, gx, gy, cfg)
    return apply_low_gate_fused(out, gate, fiedler, contours, gx, gy, cfg)

def compute_spectral_gate_chebyshev(img: torch.Tensor, L=None, edge_th: float = 0.1,
                                     sharp: float = 10.0, pct: float = 40.0, lanczos_iter: int = 30):
    from spectral_ops_fast import build_weighted_image_laplacian, lanczos_fiedler_gpu
    H, W = (img.shape[:2] if img.dim() == 2 else img.shape[:2]) if img.dim() != 3 else (img.shape[0], img.shape[1])
    if img.dim() == 3 and img.shape[-1] == 3:
        H, W, _ = img.shape
        carrier = to_grayscale(img)
    else:
        carrier = img
    if carrier.max() > 1.0: carrier = carrier / 255.0
    if L is None: L = build_weighted_image_laplacian(carrier, edge_th)
    fiedler_flat, lam2 = lanczos_fiedler_gpu(L, num_iterations=lanczos_iter)
    fiedler = fiedler_flat.reshape(H, W)
    thresh = torch.quantile(fiedler.flatten(), pct / 100.0)
    return torch.sigmoid((fiedler - thresh) * sharp), fiedler, L, lam2

def _compute_fiedler_from_tensor(img: torch.Tensor, tile_size: int = 64, overlap: int = 16,
                                  num_evecs: int = 4, radii: List[int] = None, radius_weights: List[float] = None) -> torch.Tensor:
    from spectral_ops_fast import compute_local_eigenvectors_tiled_dither
    radii = radii or [1, 2, 3, 4, 5, 6]
    radius_weights = radius_weights or [1.0, 0.6, 0.4, 0.3, 0.2, 0.1]
    evecs = compute_local_eigenvectors_tiled_dither(img, tile_size=tile_size, overlap=overlap,
                                                     num_eigenvectors=num_evecs, radii=radii, radius_weights=radius_weights)
    return evecs[:, :, 1]

def two_image_shader_pass(img_A: torch.Tensor, img_B: torch.Tensor, f_A: torch.Tensor, f_B: torch.Tensor, cfg: Dict) -> torch.Tensor:
    eff = cfg.get('effect_strength', 1.0)
    gray_A, cont_A = to_grayscale(img_A), detect_contours(to_grayscale(img_A))
    thresh_A = adaptive_threshold(f_A, 40.0)
    gate_A = torch.sigmoid((f_A - thresh_A) * 10.0)
    gx, gy = compute_gradient_xy(f_A)
    out = dilate_high_gate_fused(img_A, gate_A, f_A, gray_A, cont_A, gx, gy, cfg)
    return cross_attention_transfer(out, img_B, f_A, f_B, gate_A, cont_A, eff,
                                    shadow_offset=cfg.get('shadow_offset', 7.0),
                                    translation_strength=cfg.get('translation_strength', 20.0),
                                    min_pixels=cfg.get('min_segment_pixels', 20),
                                    max_segments=cfg.get('max_segments', 50))

def shader_forwards(img_A: torch.Tensor, img_B: torch.Tensor = None, f_A: torch.Tensor = None,
                    f_B: torch.Tensor = None, config: Dict = None, use_chebyshev: bool = None) -> torch.Tensor:
    config = config or {}
    if img_B is not None:
        f_A = f_A if f_A is not None else _compute_fiedler_from_tensor(img_A)
        f_B = f_B if f_B is not None else _compute_fiedler_from_tensor(img_B)
        return two_image_shader_pass(img_A, img_B, f_A, f_B, config)
    use_chebyshev = use_chebyshev if use_chebyshev is not None else False
    if use_chebyshev:
        gate, fiedler, _, _ = compute_spectral_gate_chebyshev(img_A, sharp=config.get('gate_sharpness', 10.0), pct=40.0)
        gray, cont = to_grayscale(img_A), detect_contours(to_grayscale(img_A))
        gx, gy = compute_gradient_xy(fiedler)
        out = dilate_high_gate_fused(img_A, gate, fiedler, gray, cont, gx, gy, config)
        return apply_low_gate_fused(out, gate, fiedler, cont, gx, gy, config)
    return shader_pass_fused(img_A, _compute_fiedler_from_tensor(img_A), config)

def shader_autoregressive(img: torch.Tensor, n_passes: int = 4, config: Dict = None,
                          schedule_fn: Callable[[Dict, int], Dict] = None, use_chebyshev: bool = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    config, cur, ints = config or {}, img, []
    for n in range(n_passes):
        cur = shader_forwards(cur, config=config, use_chebyshev=use_chebyshev)
        ints.append(cur.clone())
        if schedule_fn: config = schedule_fn(config, n)
    return cur, ints
