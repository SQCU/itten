"""
Spectral shader v14: Honest depth. RMSNorm. Projections that subtract.

Two layers, each doing attention-like comparison + gated residual:
  Layer 1: spectral affinity → "where is structure?" (bulk selection)
  Layer 2: local contrast on layer 1 output → "where are edges of structure?" (contour selection)

Each layer:
  x = x + gate(rmsnorm(projection(attn(x)))) * transform(x)

No pretending one pass does what requires two.
"""
import torch
import torch.nn.functional as F
from spectral_ops_fast import (
    build_weighted_image_laplacian,
    lanczos_k_eigenvectors,
    heat_diffusion_sparse,
    DEVICE
)


def rmsnorm(x: torch.Tensor, eps: float = 1e-8):
    """RMS normalization: x / rms(x). No centering, just scale."""
    rms = (x ** 2).mean().sqrt().clamp(min=eps)
    return x / rms


def spectral_basis(carrier: torch.Tensor, k: int = 8):
    """Spectral embedding (H*W, k)."""
    L = build_weighted_image_laplacian(carrier, edge_threshold=0.1)
    phi_np, _ = lanczos_k_eigenvectors(L, k, 50)
    return torch.tensor(phi_np, device=carrier.device, dtype=torch.float32), L


def gradient_2d(field: torch.Tensor):
    """(gy, gx) gradient, same shape as input."""
    gy = F.pad(field[1:] - field[:-1], (0, 0, 0, 1))
    gx = F.pad(field[:, 1:] - field[:, :-1], (0, 1, 0, 0))
    return gy, gx


def gradient_magnitude(field: torch.Tensor):
    """Scalar gradient magnitude."""
    gy, gx = gradient_2d(field)
    return (gy**2 + gx**2).sqrt()


def normalize_field(fy: torch.Tensor, fx: torch.Tensor):
    """Unit normalize vector field."""
    mag = (fy**2 + fx**2).sqrt().clamp(min=1e-8)
    return fy / mag, fx / mag


def make_grid(H: int, W: int, device):
    """Normalized [-1,1] grid for grid_sample."""
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    return torch.stack([gx, gy], dim=-1)


def offset_sample(img: torch.Tensor, fy: torch.Tensor, fx: torch.Tensor, dist: float):
    """Sample img at coordinates offset by (fy, fx) * dist."""
    H, W = img.shape[:2]
    grid = make_grid(H, W, img.device)
    offset = torch.stack([fx * dist * 2 / W, fy * dist * 2 / H], dim=-1)
    coords = grid + offset
    return F.grid_sample(
        img.permute(2, 0, 1).unsqueeze(0),
        coords.unsqueeze(0),
        mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(0).permute(1, 2, 0)


def color_rotate(rgb: torch.Tensor, angle: float):
    """Hue rotation via matrix multiply."""
    c = torch.cos(torch.tensor(angle, device=rgb.device))
    s = torch.sin(torch.tensor(angle, device=rgb.device))
    # Rotate in RG plane
    C = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], device=rgb.device, dtype=rgb.dtype)
    return (rgb @ C.T).clamp(0, 1)


# =============================================================================
# LAYER 1: Spectral affinity → bulk structure selection
# =============================================================================

def layer1_structure_gate(phi_t: torch.Tensor, phi_s: torch.Tensor,
                          carrier_t: torch.Tensor, W_proj: torch.Tensor, H: int, W: int):
    """
    Computes "where is structure?" gate from spectral affinity.

    CRITICAL: Multiply by boundary weight so gate only fires on actual ink,
    not on background. This is what `& contours` did discretely.
    """
    k = phi_t.shape[1]

    # Affinity: how much does each target pixel match sample's spectral signature
    sample_sig = phi_s.mean(dim=0)  # (k,)
    affinity = (phi_t * sample_sig).sum(dim=-1)  # (n,)

    # RMSNorm
    normed = rmsnorm(affinity)

    # Threshold: only top values (above median) get positive input to sigmoid
    threshold = normed.median()
    centered = normed - threshold

    # Sigmoid gate
    gate = torch.sigmoid(centered * W_proj[0])

    # BOUNDARY WEIGHT: high on ink (dark pixels), low on background
    # This replaces the discrete `& contours` with smooth multiplication
    boundary_weight = 1.0 - carrier_t.flatten()  # dark pixels → high weight
    boundary_weight = boundary_weight ** 2  # sharpen: make background even lower

    # Gate only fires where there's actual structure
    gate = gate * boundary_weight

    return gate.reshape(H, W)


# =============================================================================
# LAYER 2: Local contrast → edge/contour selection
# =============================================================================

def layer2_edge_gate(structure_gate: torch.Tensor, carrier_t: torch.Tensor, W_edge: float = 5.0):
    """
    Computes "where are edges of structure?" from gradient of layer 1 output.

    High gradient magnitude = transition zone = contour.
    Also weighted by boundary so we don't process background.
    """
    # Gradient magnitude of the structure gate
    edge_response = gradient_magnitude(structure_gate)

    # RMSNorm + threshold (top half)
    normed = rmsnorm(edge_response.flatten())
    threshold = normed.median()
    centered = normed - threshold

    # Sigmoid with gain
    gate = torch.sigmoid(centered.reshape(structure_gate.shape) * W_edge)

    # Boundary weight: only fire on/near structure
    boundary_weight = 1.0 - carrier_t
    boundary_weight = boundary_weight ** 2

    return gate * boundary_weight


# =============================================================================
# MAIN SHADER: 2-layer ResNet structure
# =============================================================================

def shader(
    target: torch.Tensor,
    sample: torch.Tensor = None,
    # Layer 1 params
    W1_gain: float = 3.0,      # gain for structure gate sigmoid
    # Layer 2 params
    W2_gain: float = 5.0,      # gain for edge gate sigmoid
    # Transform params
    spread_r: int = 2,
    offset_dist: float = 12.0,
    hue_shift: float = 0.6,
):
    """
    2-layer spectral shader.

    Layer 1: spectral affinity → structure gate → spread transform
    Layer 2: gradient of L1 → edge gate → offset transform

    Each layer: x = x + gate * (transform - x)
    """
    if sample is None:
        sample = target

    H, W = target.shape[:2]
    H_s, W_s = sample.shape[:2]
    device = target.device

    carrier_t = target.mean(dim=-1)
    carrier_s = sample.mean(dim=-1)

    # === SPECTRAL BASES ===
    phi_t, L_t = spectral_basis(carrier_t)
    phi_s, L_s = spectral_basis(carrier_s)
    k = min(phi_t.shape[1], phi_s.shape[1])
    phi_t, phi_s = phi_t[:, :k], phi_s[:, :k]

    # === DIRECTION FIELDS ===
    fiedler = phi_t[:, min(1, k-1)].reshape(H, W)
    gy, gx = gradient_2d(fiedler)
    perp_y, perp_x = normalize_field(-gx, gy)  # perpendicular to Fiedler gradient

    # Outward field from heat diffusion
    boundary = (carrier_t < 0.5).float().flatten()
    dist_field = heat_diffusion_sparse(L_t, boundary, 0.1, 30).reshape(H, W)
    out_y, out_x = gradient_2d(dist_field)
    out_y, out_x = normalize_field(-out_y, -out_x)

    # === LAYER 1: Structure gate + spread ===
    gate1 = layer1_structure_gate(phi_t, phi_s, carrier_t, torch.tensor([W1_gain]), H, W)

    # Spread transform: sample from perpendicular offset, average
    spread = torch.zeros_like(target)
    for r in range(1, spread_r + 1):
        spread = spread + offset_sample(target, perp_y, perp_x, r)
        spread = spread + offset_sample(target, perp_y, perp_x, -r)
    spread = spread / (2 * spread_r)

    # Residual: x = x + gate * (transform - x)
    x = target + gate1.unsqueeze(-1) * (spread - target)

    # === LAYER 2: Edge gate + offset ===
    gate2 = layer2_edge_gate(gate1, carrier_t, W2_gain)

    # Cross-attention retrieval (sparse, on boundary pixels only)
    t_bnd = (carrier_t < 0.5).flatten()
    s_bnd = (carrier_s < 0.5).flatten()
    t_idx = t_bnd.nonzero().squeeze(-1)
    s_idx = s_bnd.nonzero().squeeze(-1)

    # Cap attention size
    max_attn = 6000
    if t_idx.shape[0] > max_attn:
        t_idx = t_idx[torch.randperm(t_idx.shape[0], device=device)[:max_attn]]
    if s_idx.shape[0] > max_attn:
        s_idx = s_idx[torch.randperm(s_idx.shape[0], device=device)[:max_attn]]

    if t_idx.shape[0] > 0 and s_idx.shape[0] > 0:
        phi_t_sp = phi_t[t_idx]
        phi_s_sp = phi_s[s_idx]

        attn = F.softmax(phi_t_sp @ phi_s_sp.T / (k ** 0.5), dim=-1)
        sample_flat = sample.reshape(-1, 3)[s_idx]
        retrieved_sp = attn @ sample_flat

        # Scatter to full image (start with current x)
        retrieved = x.clone()
        t_y, t_x = t_idx // W, t_idx % W
        retrieved[t_y, t_x] = retrieved_sp
    else:
        retrieved = x

    # Offset + color rotate
    offset_img = offset_sample(retrieved, out_y, out_x, offset_dist)
    offset_img = color_rotate(offset_img, hue_shift)

    # Residual
    x = x + gate2.unsqueeze(-1) * (offset_img - x)

    return x.clamp(0, 1)


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
        ).save(out / f"{ts}_v14_{name}.png")
        print(f"  saved: {name}")

    toof = load("toof.png")
    snek = load("snek-heavy.png")
    tone = load("red-tonegraph.png")

    print("\n[v14] 2-layer ResNet structure, RMSNorm, honest depth")

    if toof is not None:
        print("  toof self:")
        save(shader(toof), "self_toof")

    if toof is not None and snek is not None:
        print("  toof × snek:")
        save(shader(toof, snek), "toof_x_snek")

    if snek is not None and tone is not None:
        print("  snek × tone:")
        save(shader(snek, tone), "snek_x_tone")
