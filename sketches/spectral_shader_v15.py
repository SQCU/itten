"""
Spectral shader v15: Clean (img, act) -> (img, act) signature.

1. Selection: spectral partition (not all pixels, some subset)
2a. Thicken: copy contiguous parts, paste displaced perpendicular to tangent
2b. Shadow: copy, ROTATE 90°, paste offset, color-tag to distinguish from 2a

Both thicken and shadow are:
    f(img: Tensor, act: Tensor) -> (img: Tensor, act: Tensor)

This separates "concrete pixel changes" from "abstract activation state".
"""
import torch
import torch.nn.functional as F
from spectral_ops_fast import (
    build_weighted_image_laplacian,
    lanczos_k_eigenvectors,
    heat_diffusion_sparse,
    DEVICE
)


# =============================================================================
# PRIMITIVES (pure tensor ops, no metaphors)
# =============================================================================

def spectral_basis(carrier: torch.Tensor, k: int = 8):
    """(H*W, k) spectral embedding + Laplacian."""
    L = build_weighted_image_laplacian(carrier, edge_threshold=0.1)
    phi_np, _ = lanczos_k_eigenvectors(L, k, 50)
    return torch.tensor(phi_np, device=carrier.device, dtype=torch.float32), L


def gradient_2d(f: torch.Tensor):
    """(gy, gx) same shape as f."""
    gy = F.pad(f[1:] - f[:-1], (0, 0, 0, 1))
    gx = F.pad(f[:, 1:] - f[:, :-1], (0, 1, 0, 0))
    return gy, gx


def normalize(fy: torch.Tensor, fx: torch.Tensor):
    """Unit normalize vector field."""
    mag = (fy**2 + fx**2).sqrt().clamp(min=1e-8)
    return fy / mag, fx / mag


def make_grid(H: int, W: int, device):
    """[-1,1] normalized grid for grid_sample."""
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack([gx, gy], dim=-1)  # (H, W, 2) as (x, y)


def sample_offset(img: torch.Tensor, fy: torch.Tensor, fx: torch.Tensor, dist: float):
    """Sample img at coords offset by (fy, fx) * dist."""
    H, W = img.shape[:2]
    grid = make_grid(H, W, img.device)
    # Convert pixel offset to normalized coords
    offset = torch.stack([fx * dist * 2 / W, fy * dist * 2 / H], dim=-1)
    return F.grid_sample(
        img.permute(2, 0, 1).unsqueeze(0),
        (grid + offset).unsqueeze(0),
        mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(0).permute(1, 2, 0)


def color_tag(rgb: torch.Tensor, phase: float = 1.0, blend: float = 0.7):
    """
    Cyclic color transform with NO fixed points at black or white.
    Black → color, white → color. Enables tracing effect provenance.

    Expressed as matmuls where possible:
    - luminance = rgb @ lum_weights
    - cyclic_rgb = 0.5 + 0.4 * sin(2π*lum + phase + channel_offsets)
    - output = (1-blend)*rgb + blend*cyclic  (linear combination)
    """
    device = rgb.device
    dtype = rgb.dtype
    pi = torch.pi

    # Luminance via matmul: (..., 3) @ (3, 1) -> (..., 1), then squeeze
    lum_weights = torch.tensor([[0.299], [0.587], [0.114]], device=device, dtype=dtype)
    lum = (rgb @ lum_weights).squeeze(-1)  # (...,)

    # Cyclic RGB: 120° phase offsets between channels
    phase_rad = phase * pi
    offsets = torch.tensor([0, 2*pi/3, 4*pi/3], device=device, dtype=dtype)

    # angles: (..., 1) + (3,) -> (..., 3)
    angles = 2 * pi * lum.unsqueeze(-1) + phase_rad + offsets
    cyclic_rgb = 0.5 + 0.4 * torch.sin(angles)

    # Blend: linear combination
    out = (1 - blend) * rgb + blend * cyclic_rgb

    return out.clamp(0, 1)


# =============================================================================
# 1. SELECTION: spectral partition -> activations
# =============================================================================

def compute_activations(target: torch.Tensor, sample: torch.Tensor,
                         gamma_hi: float = 1.5, bias_hi: float = -0.3,
                         gamma_lo: float = 0.5, bias_lo: float = 0.3):
    """
    Multi-threshold gates with gap healing via linear projection.

    Stack gates at [permissive, restrictive] thresholds → (H*W, 2)
    Linear project to get healed gate that fills gaps using full curve direction.

    Also stack tangent fields: [full_tangent, masked_tangent] → (H, W, 2)
    Linear project to get healed tangent that knows direction even at gaps.
    """
    H, W = target.shape[:2]
    device = target.device

    carrier_t = target.mean(dim=-1)
    carrier_s = sample.mean(dim=-1)

    # Spectral bases
    phi_t, L_t = spectral_basis(carrier_t)
    phi_s, _ = spectral_basis(carrier_s)
    k = min(phi_t.shape[1], phi_s.shape[1])
    phi_t, phi_s = phi_t[:, :k], phi_s[:, :k]

    # Affinity from sample boundary signature
    sample_boundary = (carrier_s < 0.5).flatten()
    sample_bnd_idx = sample_boundary.nonzero().squeeze(-1)
    if sample_bnd_idx.numel() > 0:
        sample_sig = phi_s[sample_bnd_idx].mean(dim=0)
    else:
        sample_sig = phi_s.mean(dim=0)
    affinity = (phi_t * sample_sig).sum(dim=-1)
    aff_norm = (affinity - affinity.mean()) / (affinity.std() + 1e-8)

    # Boundary weight
    boundary = (1.0 - carrier_t.flatten()) ** 2

    # === MULTI-THRESHOLD GATES: (H*W, 2) ===
    # dim 0 = permissive (full curves), dim 1 = restrictive (sparse but correct)
    thresholds_hi = torch.tensor([0.0, 0.3], device=device)  # [permissive, restrictive]
    thresholds_lo = torch.tensor([0.0, 0.3], device=device)

    # Stack: (H*W, 2)
    gate_hi_multi = torch.sigmoid((aff_norm.unsqueeze(-1) - thresholds_hi) * 3)
    gate_lo_multi = torch.sigmoid((thresholds_lo - aff_norm.unsqueeze(-1)) * 3)

    # Apply boundary weight
    gate_hi_multi = gate_hi_multi * boundary.unsqueeze(-1)
    gate_lo_multi = gate_lo_multi * boundary.unsqueeze(-1)

    # === LINEAR PROJECTION FOR GAP HEALING ===
    # [permissive, restrictive] @ W_heal → healed
    # W_heal: (2, 1) so (H*W, 2) @ (2, 1) → (H*W, 1)
    W_heal = torch.tensor([[0.3], [0.7]], device=device, dtype=torch.float32)

    gate_hi = (gate_hi_multi @ W_heal).squeeze(-1).reshape(H, W)
    gate_lo = (gate_lo_multi @ W_heal).squeeze(-1).reshape(H, W)

    # === TANGENT FIELD ===
    fiedler = phi_t[:, min(1, k-1)].reshape(H, W)
    tan_y_full, tan_x_full = gradient_2d(fiedler)
    tan_y_full, tan_x_full = normalize(tan_y_full, tan_x_full)

    # Stack tangent: [full, masked_by_restrictive] → (H, W, 2)
    gate_hi_restrictive = gate_hi_multi[:, 1].reshape(H, W)
    tan_y_masked = tan_y_full * gate_hi_restrictive
    tan_x_masked = tan_x_full * gate_hi_restrictive

    tan_y_stacked = torch.stack([tan_y_full, tan_y_masked], dim=-1)
    tan_x_stacked = torch.stack([tan_x_full, tan_x_masked], dim=-1)

    # Linear project: healed tangent uses full direction at gaps
    # (H, W, 2) @ (2, 1) → (H, W, 1) → squeeze
    tan_y = (tan_y_stacked @ W_heal).squeeze(-1)
    tan_x = (tan_x_stacked @ W_heal).squeeze(-1)
    tan_y, tan_x = normalize(tan_y, tan_x)

    # Outward field
    bnd_flat = (carrier_t < 0.5).float().flatten()
    dist_field = heat_diffusion_sparse(L_t, bnd_flat, 0.1, 30).reshape(H, W)
    out_y, out_x = gradient_2d(dist_field)
    out_y, out_x = normalize(-out_y, -out_x)

    return {
        'gate_hi': gate_hi,
        'gate_lo': gate_lo,
        'gate_hi_multi': gate_hi_multi.reshape(H, W, 2),  # for debugging
        'tan_y': tan_y, 'tan_x': tan_x,
        'out_y': out_y, 'out_x': out_x,
        'phi_t': phi_t, 'L_t': L_t,
    }


# =============================================================================
# 2a. THICKEN: copy, displace perpendicular, paste
# =============================================================================

def thicken(img: torch.Tensor, act: dict, radius: int = 3) -> tuple[torch.Tensor, dict]:
    """
    SCATTER: For HIGH-AFFINITY pixels (match sample), write perpendicular.
    Uses gate_hi: fires where target structure matches sample structure.

    (img, act) -> (img, act)
    """
    H, W = img.shape[:2]
    device = img.device
    gate = act['gate_hi']  # HIGH affinity gate
    tan_y, tan_x = act['tan_y'], act['tan_x']

    # Perpendicular direction
    perp_y, perp_x = -tan_x, tan_y

    # Find source pixels (where HIGH gate fires)
    src_mask = gate > 0.3
    src_idx = src_mask.nonzero()  # (N, 2) as [y, x]

    if src_idx.shape[0] == 0:
        return img, act

    new_img = img.clone()

    # For each radius, scatter source colors to offset destinations
    for r in range(1, radius + 1):
        for sign in [1.0, -1.0]:
            src_y, src_x = src_idx[:, 0], src_idx[:, 1]
            src_colors = img[src_y, src_x]

            py = perp_y[src_y, src_x]
            px = perp_x[src_y, src_x]

            dst_y = (src_y.float() + sign * py * r).round().long()
            dst_x = (src_x.float() + sign * px * r).round().long()

            valid = (dst_y >= 0) & (dst_y < H) & (dst_x >= 0) & (dst_x < W)
            dst_y, dst_x = dst_y[valid], dst_x[valid]
            colors = src_colors[valid]

            # Preserve structure: only write to background
            dst_is_bg = gate[dst_y, dst_x] < 0.3
            dst_y, dst_x = dst_y[dst_is_bg], dst_x[dst_is_bg]
            colors = colors[dst_is_bg]

            if dst_y.numel() > 0:
                new_img[dst_y, dst_x] = colors

    return new_img, act


# =============================================================================
# 2b. SHADOW: copy, rotate 90°, paste offset, color-tag
# =============================================================================

def shadow(img: torch.Tensor, act: dict, dist: float = 15.0, trace_len: int = 20) -> tuple[torch.Tensor, dict]:
    """
    FLOW FIELD approach: trace along tangent, deposit at perpendicular offset.

    Only propagate from LINKED pixels (have neighbors that also pass gate).
    Follow tangent flow, deposit color-tagged pixels along the trace.

    (img, act) -> (img, act)
    """
    H, W = img.shape[:2]
    device = img.device
    gate = act['gate_lo']
    gate_hi = act['gate_hi']
    out_y, out_x = act['out_y'], act['out_x']
    tan_y, tan_x = act['tan_y'], act['tan_x']

    new_img = img.clone()

    # === LINKAGE TEST: only propagate from pixels with gated neighbors ===
    # Dilate gate, then AND with original - keeps only pixels with nearby support
    gate_dilated = F.max_pool2d(
        gate.unsqueeze(0).unsqueeze(0),
        kernel_size=3, stride=1, padding=1
    ).squeeze()
    linked_mask = (gate > 0.3) & (gate_dilated > 0.5)  # has neighbors

    # Get seed points (linked pixels)
    seeds = linked_mask.nonzero()  # (N, 2)

    if seeds.shape[0] == 0:
        return img, act

    # Subsample seeds for efficiency
    max_seeds = 2000
    if seeds.shape[0] > max_seeds:
        idx = torch.randperm(seeds.shape[0], device=device)[:max_seeds]
        seeds = seeds[idx]

    # === TRACE FLOW AND DEPOSIT ===
    for i in range(seeds.shape[0]):
        sy, sx = seeds[i, 0].item(), seeds[i, 1].item()

        # Current position (float for smooth tracing)
        cy, cx = float(sy), float(sx)

        # Get seed color and tag it
        seed_color = img[sy, sx].unsqueeze(0)  # (1, 3)
        seed_color = color_tag(seed_color, phase=1.0, blend=0.8).squeeze(0)  # (3,)

        # Trace along tangent field
        for step in range(trace_len):
            iy, ix = int(cy), int(cx)

            # Bounds check
            if not (0 <= iy < H and 0 <= ix < W):
                break

            # Stop if we leave the linked region
            if not linked_mask[iy, ix]:
                break

            # Get local tangent and outward directions
            ty = tan_y[iy, ix].item()
            tx = tan_x[iy, ix].item()
            oy = out_y[iy, ix].item()
            ox = out_x[iy, ix].item()

            # Deposit at outward offset from current trace position
            dep_y = int(cy + oy * dist)
            dep_x = int(cx + ox * dist)

            if 0 <= dep_y < H and 0 <= dep_x < W:
                # Only deposit on background
                if gate_hi[dep_y, dep_x] < 0.3:
                    new_img[dep_y, dep_x] = seed_color

            # Step along tangent (alternate direction each seed for coverage)
            direction = 1.0 if i % 2 == 0 else -1.0
            cy += ty * direction
            cx += tx * direction

    return new_img, act


# =============================================================================
# MAIN SHADER
# =============================================================================

def shader(target: torch.Tensor, sample: torch.Tensor = None,
           thick_r: int = 4, shadow_dist: float = 20.0):
    """
    Spectral shader: selection -> thicken -> shadow.

    Clean composition of (img, act) -> (img, act) functions.
    """
    if sample is None:
        sample = target

    # 1. Selection
    act = compute_activations(target, sample)

    # 2a. Thicken
    img, act = thicken(target, act, radius=thick_r)

    # 2b. Shadow
    img, act = shadow(img, act, dist=shadow_dist)

    return img.clamp(0, 1)


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
        ).save(out / f"{ts}_v15_{name}.png")
        print(f"  {name}")

    toof = load("toof.png")
    snek = load("snek-heavy.png")
    tone = load("red-tonegraph.png")

    print("\n[v15] Clean (img, act) -> (img, act)")

    if toof is not None:
        print("  toof self:")
        save(shader(toof), "self_toof")

    if toof is not None and snek is not None:
        print("  toof × snek:")
        save(shader(toof, snek), "toof_x_snek")

    if snek is not None and tone is not None:
        print("  snek × tone:")
        save(shader(snek, tone), "snek_x_tone")
