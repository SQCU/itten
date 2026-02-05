"""
Spectral shader v13: SwiGLU-style smooth gating, no connected components, no iteration.

Variable names describe TENSOR OPERATIONS, not artistic metaphors.
- "gate_hi/lo" = smooth scalar field from swish, not boolean mask
- "affinity" = dot product of spectral bases, not "resonance"
- "perp_field" = 90° rotation of gradient vectors, not "orthogonal to tangent"
- "outward_field" = negative gradient of distance, not "void direction"

No:
- Connected components
- Per-pixel iteration
- Scipy
- Boolean masks
- Metaphor-laden variable names
"""
import torch
import torch.nn.functional as F
from spectral_ops_fast import (
    build_weighted_image_laplacian,
    lanczos_k_eigenvectors,
    heat_diffusion_sparse,
    DEVICE
)


def spectral_basis(carrier: torch.Tensor, k: int = 8):
    """φ: (H*W, k) spectral embedding."""
    H, W = carrier.shape
    L = build_weighted_image_laplacian(carrier, edge_threshold=0.1)
    phi_np, _ = lanczos_k_eigenvectors(L, k, 50)
    return torch.tensor(phi_np, device=carrier.device, dtype=torch.float32), L


def gradient_2d(field: torch.Tensor):
    """Returns (gy, gx) gradient fields, same shape as input."""
    gy = F.pad(field[1:] - field[:-1], (0, 0, 0, 1))
    gx = F.pad(field[:, 1:] - field[:, :-1], (0, 1, 0, 0))
    return gy, gx


def normalize_field(fy: torch.Tensor, fx: torch.Tensor):
    """Unit normalize a vector field."""
    mag = (fy**2 + fx**2).sqrt().clamp(min=1e-8)
    return fy / mag, fx / mag


def rotation_matrix_field(fy: torch.Tensor, fx: torch.Tensor):
    """
    Construct (H, W, 2, 2) rotation matrices from direction field.
    R @ [1, 0]^T = [fx, fy]^T (rotates x-axis to point along field)
    """
    fy, fx = normalize_field(fy, fx)
    # R = [[cos, -sin], [sin, cos]] where cos=fx, sin=fy
    R = torch.stack([
        torch.stack([fx, -fy], dim=-1),
        torch.stack([fy, fx], dim=-1)
    ], dim=-2)  # (H, W, 2, 2)
    return R


def make_base_grid(H: int, W: int, device):
    """Normalized [-1, 1] coordinate grid for grid_sample."""
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack([gx, gy], dim=-1)  # (H, W, 2) - note: grid_sample wants (x, y)


def offset_grid(base_grid: torch.Tensor, fy: torch.Tensor, fx: torch.Tensor,
                distance: float, H: int, W: int):
    """Offset grid coordinates along direction field."""
    # Convert pixel-space offset to normalized [-1, 1] space
    offset_y = fy * distance * 2.0 / H
    offset_x = fx * distance * 2.0 / W
    offset = torch.stack([offset_x, offset_y], dim=-1)  # (H, W, 2)
    return base_grid + offset


def color_matrix(angle: float):
    """3x3 hue rotation matrix."""
    c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    # Rotate in RG plane, keep B
    return torch.tensor([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=torch.float32)


# =============================================================================
# MAIN SHADER
# =============================================================================

def shader(
    target: torch.Tensor,
    sample: torch.Tensor = None,
    # Gate parameters (SwiGLU style: smooth, not boolean)
    w_hi: float = 2.0,      # scale for high gate swish input
    b_hi: float = 0.0,      # bias for high gate
    w_lo: float = 2.0,      # scale for low gate swish input
    b_lo: float = 0.0,      # bias for low gate
    # Geometric parameters
    spread_r: int = 3,      # spread radius (pixels)
    offset_dist: float = 15.0,  # offset distance (pixels)
    hue_angle: float = 0.8,     # color rotation (radians)
):
    """
    Spectral shader via SwiGLU-style gating.

    Two paths, both smooth-gated:
    1. HIGH: spread target colors perpendicular to Fiedler gradient
    2. LOW: cross-attention to sample, spatially offset along outward field
    """
    if sample is None:
        sample = target

    H, W = target.shape[:2]
    H_s, W_s = sample.shape[:2]
    device = target.device
    n_t, n_s = H * W, H_s * W_s

    carrier_t = target.mean(dim=-1)
    carrier_s = sample.mean(dim=-1)

    # === SPECTRAL BASES ===
    phi_t, L_t = spectral_basis(carrier_t)  # (n_t, k)
    phi_s, L_s = spectral_basis(carrier_s)  # (n_s, k)
    k = min(phi_t.shape[1], phi_s.shape[1])
    phi_t, phi_s = phi_t[:, :k], phi_s[:, :k]

    # === AFFINITY FIELD ===
    # How much each target pixel's spectral signature matches sample structure
    if n_t == n_s:
        # Same size: element-wise dot product
        affinity = (phi_t * phi_s).sum(dim=-1)  # (n_t,)
    else:
        # Different size: dot with mean sample signature
        sample_sig = phi_s.mean(dim=0)  # (k,)
        affinity = (phi_t * sample_sig).sum(dim=-1)  # (n_t,)

    # Normalize to zero-mean unit-variance
    affinity = (affinity - affinity.mean()) / (affinity.std() + 1e-8)

    # === SMOOTH GATES (SwiGLU: swish = x * sigmoid(x)) ===
    gate_hi = F.silu(w_hi * affinity + b_hi)  # high affinity → large positive
    gate_lo = F.silu(w_lo * (-affinity) + b_lo)  # low affinity → large positive

    # Reshape gates to image
    gate_hi_2d = gate_hi.reshape(H, W, 1)
    gate_lo_2d = gate_lo.reshape(H, W, 1)

    # === DIRECTION FIELDS ===
    # Fiedler = 2nd eigenvector, encodes global bipartition
    fiedler = phi_t[:, min(1, k-1)].reshape(H, W)
    grad_y, grad_x = gradient_2d(fiedler)
    grad_y, grad_x = normalize_field(grad_y, grad_x)

    # Perpendicular to gradient (90° rotation)
    perp_y, perp_x = -grad_x, grad_y

    # Outward field: negative gradient of distance from structure
    boundary = (carrier_t < 0.5).float().flatten()
    dist_field = heat_diffusion_sparse(L_t, boundary, 0.1, 30).reshape(H, W)
    out_y, out_x = gradient_2d(dist_field)
    out_y, out_x = normalize_field(-out_y, -out_x)  # negative = away from structure

    # === HIGH PATH: PERPENDICULAR SPREAD ===
    # Spread target colors along perp direction, gated by gate_hi
    # Simple implementation: sample from offset locations and blend
    base_grid = make_base_grid(H, W, device)

    spread_accum = torch.zeros_like(target)
    for r in range(1, spread_r + 1):
        for sign in [1.0, -1.0]:
            offset_grid_r = offset_grid(base_grid, perp_y * sign, perp_x * sign, r, H, W)
            sampled = F.grid_sample(
                target.permute(2, 0, 1).unsqueeze(0),  # (1, 3, H, W)
                offset_grid_r.unsqueeze(0),  # (1, H, W, 2)
                mode='bilinear', padding_mode='border', align_corners=True
            ).squeeze(0).permute(1, 2, 0)  # (H, W, 3)
            spread_accum = spread_accum + sampled
    spread = spread_accum / (2 * spread_r)  # average

    # === LOW PATH: CROSS-ATTENTION + OFFSET ===
    # Sparse attention: only boundary pixels attend to boundary pixels
    target_boundary = (carrier_t < 0.5).flatten()  # (n_t,) bool
    sample_boundary = (carrier_s < 0.5).flatten()  # (n_s,) bool

    t_idx = target_boundary.nonzero().squeeze(-1)  # boundary pixel indices in target
    s_idx = sample_boundary.nonzero().squeeze(-1)  # boundary pixel indices in sample

    # Subsample if still too large (cap at 8k x 8k attention)
    max_attn = 8000
    if t_idx.shape[0] > max_attn:
        t_idx = t_idx[torch.randperm(t_idx.shape[0], device=device)[:max_attn]]
    if s_idx.shape[0] > max_attn:
        s_idx = s_idx[torch.randperm(s_idx.shape[0], device=device)[:max_attn]]

    # Sparse attention: (|t_boundary|, |s_boundary|)
    phi_t_sparse = phi_t[t_idx]  # (n_t_sparse, k)
    phi_s_sparse = phi_s[s_idx]  # (n_s_sparse, k)

    attn_logits = phi_t_sparse @ phi_s_sparse.T  # manageable size
    attn_weights = F.softmax(attn_logits / (k ** 0.5), dim=-1)

    sample_flat = sample.reshape(n_s, 3)
    sample_sparse = sample_flat[s_idx]  # (n_s_sparse, 3)
    retrieved_sparse = attn_weights @ sample_sparse  # (n_t_sparse, 3)

    # Scatter back to full image
    retrieved = target.clone()  # start with target colors
    t_y, t_x = t_idx // W, t_idx % W
    retrieved[t_y, t_x] = retrieved_sparse

    # Rotation matrix field from outward direction
    R = rotation_matrix_field(out_y, out_x)  # (H, W, 2, 2)

    # Color rotation
    C = color_matrix(hue_angle).to(device)  # (3, 3)
    retrieved_colored = (retrieved @ C.T).clamp(0, 1)

    # Offset along outward direction
    offset_coords = offset_grid(base_grid, out_y, out_x, offset_dist, H, W)

    # Sample retrieved content at offset locations
    cross_attn_out = F.grid_sample(
        retrieved_colored.permute(2, 0, 1).unsqueeze(0),
        offset_coords.unsqueeze(0),
        mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(0).permute(1, 2, 0)

    # === GATED RESIDUAL COMBINATION ===
    output = target + gate_hi_2d * (spread - target) + gate_lo_2d * (cross_attn_out - target)

    return output.clamp(0, 1)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    from PIL import Image
    import numpy as np
    from datetime import datetime

    inp = Path("demo_output/inputs")
    out = Path("demo_output")

    def load(name):
        p = inp / name
        if p.exists():
            return torch.tensor(
                np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0,
                device=DEVICE
            )
        return None

    def save(img, name):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        Image.fromarray(
            (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        ).save(out / f"{ts}_v13_{name}.png")
        print(f"  saved: {name}")

    toof = load("toof.png")
    snek = load("snek-heavy.png")
    tone = load("red-tonegraph.png")

    print("\n[v13] SwiGLU-style smooth gating")

    if toof is not None:
        print("  toof self-attention:")
        save(shader(toof), "self_toof")

    if toof is not None and snek is not None:
        print("  toof × snek cross-attention:")
        save(shader(toof, snek), "toof_x_snek")

    if snek is not None and tone is not None:
        print("  snek × tone cross-attention:")
        save(shader(snek, tone), "snek_x_tone")
