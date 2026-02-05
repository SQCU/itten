"""
Spectral shader v8: Implementing spec-shader-operations.md correctly.

The spec asks for:
1. Graph comparison → activation field (one matmul)
2. Two gates with INDEPENDENT (gamma, bias) pairs on SAME activation
3. High gate → thicken (orthogonal scatter)
4. Low gate → drop shadow (retrieve, transform, scatter)
5. ~10 lines main body, helpers for primitives

NO blur kernels. NO sharpen effects. Scatter writes only.
Color transforms as diagnostic indicators.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from spectral_ops_fast import (
    build_weighted_image_laplacian,
    iterative_spectral_transform,
    heat_diffusion_sparse,
    lanczos_k_eigenvectors,
    DEVICE
)

# ============================================================
# Geometric primitives
# ============================================================

def gradient_2d(field_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient of 2D field. Returns (grad_y, grad_x)."""
    H, W = field_2d.shape
    grad_y = torch.zeros_like(field_2d)
    grad_x = torch.zeros_like(field_2d)
    grad_y[:-1, :] = field_2d[1:, :] - field_2d[:-1, :]
    grad_x[:, :-1] = field_2d[:, 1:] - field_2d[:, :-1]
    return grad_y, grad_x


def normalize_vectors(vy: torch.Tensor, vx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize 2D vector field to unit length."""
    mag = (vy**2 + vx**2).sqrt().clamp(min=1e-8)
    return vy / mag, vx / mag


def color_rotate(rgb: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate hue by angle (in radians). Diagnostic indicator."""
    # Convert to simple hue rotation via matrix
    cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    # Approximate hue rotation in RGB space
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.333 * (r + g + b)
    r_c, g_c, b_c = r - lum, g - lum, b - lum
    # Rotate chromatic component
    new_r = lum + cos_a * r_c - sin_a * g_c
    new_g = lum + sin_a * r_c + cos_a * g_c
    new_b = b  # Keep blue axis stable for visibility
    return torch.stack([new_r, new_g, new_b], dim=-1).clamp(0, 1)


def find_connected_components(boundary_mask: torch.Tensor, H: int, W: int) -> list[torch.Tensor]:
    """
    Find connected components in boundary mask using flood fill.
    Returns list of index tensors, one per component.
    """
    device = boundary_mask.device
    visited = torch.zeros(H * W, dtype=torch.bool, device=device)
    boundary_idx = boundary_mask.nonzero(as_tuple=False).squeeze(-1)

    if boundary_idx.numel() == 0:
        return []

    boundary_set = set(boundary_idx.cpu().tolist())
    components = []

    for start in boundary_idx.cpu().tolist():
        if visited[start]:
            continue

        # BFS flood fill
        component = []
        queue = [start]
        while queue:
            idx = queue.pop(0)
            if visited[idx] or idx not in boundary_set:
                continue
            visited[idx] = True
            component.append(idx)

            # 4-connected neighbors
            y, x = idx // W, idx % W
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    nidx = ny * W + nx
                    if not visited[nidx] and nidx in boundary_set:
                        queue.append(nidx)

        if len(component) >= 5:  # minimum component size
            components.append(torch.tensor(component, device=device))

    return components


# ============================================================
# Core shader
# ============================================================

def shader(
    target: torch.Tensor,  # (H, W, 3) RGB
    sample: torch.Tensor,  # (H_s, W_s, 3) RGB
    gamma_h: float = 1.3,  # high gate gain (>1 for high threshold)
    bias_h: float = -0.15, # high gate bias (negative pushes threshold up)
    gamma_l: float = 1.3,  # low gate gain (>1 for low threshold)
    bias_l: float = 0.15,  # low gate bias (positive pushes threshold down into negatives)
    thicken_width: float = 2.0,
    shadow_distance: float = 15.0,
    theta: float = 0.3,    # spectral depth for comparison
) -> torch.Tensor:
    """
    Spectral shader per spec-shader-operations.md.

    Returns target with thickened high-resonance regions and
    shadowed low-resonance regions.
    """
    H_t, W_t = target.shape[:2]
    H_s, W_s = sample.shape[:2]
    n_t, n_s = H_t * W_t, H_s * W_s
    device = target.device

    # Grayscale carriers
    carrier_t = target.mean(dim=-1)
    carrier_s = sample.mean(dim=-1)

    # ========================================
    # 1. GRAPH COMPARISON (spectral embedding + cross-attention)
    # ========================================
    L_t = build_weighted_image_laplacian(carrier_t, edge_threshold=0.1)
    L_s = build_weighted_image_laplacian(carrier_s, edge_threshold=0.1)

    # Compute spectral embeddings: k eigenvector coefficients per pixel
    k = 8  # spectral dimension
    phi_t_np, _ = lanczos_k_eigenvectors(L_t, num_eigenvectors=k)  # (n_t, k)
    phi_s_np, _ = lanczos_k_eigenvectors(L_s, num_eigenvectors=k)  # (n_s, k)
    phi_t = torch.tensor(phi_t_np, dtype=torch.float32, device=device)
    phi_s = torch.tensor(phi_s_np, dtype=torch.float32, device=device)

    # Ensure consistent k dimension (lanczos might return fewer)
    k_t, k_s = phi_t.shape[1], phi_s.shape[1]
    k_min = min(k_t, k_s)
    phi_t = phi_t[:, :k_min]
    phi_s = phi_s[:, :k_min]

    # Activation: spectral similarity at each position
    # For same-size images: dot product of spectral vectors at each pixel
    # For different sizes: project target onto sample's BOUNDARY spectral signature
    sample_boundary = (carrier_s < 0.5).flatten()  # ink pixels
    sample_boundary_idx = sample_boundary.nonzero(as_tuple=False).squeeze(-1)

    if n_t == n_s:
        # Element-wise dot product: sum over k dimension
        activation = (phi_t * phi_s).sum(dim=1)  # (n_t,)
    else:
        # Cross-attention: each target pixel attends to sample BOUNDARY structure
        if sample_boundary_idx.numel() > 0:
            # Use mean spectral signature of sample ink pixels (not all pixels)
            sample_signature = phi_s[sample_boundary_idx].mean(dim=0)  # (k,)
            activation = (phi_t * sample_signature.unsqueeze(0)).sum(dim=1)  # (n_t,)
        else:
            activation = torch.zeros(n_t, device=device)

    # Normalize to [-1, 1] range for consistent gating
    act_mean = activation.mean()
    act_std = activation.std().clamp(min=1e-8)
    activation = (activation - act_mean) / act_std

    # ========================================
    # 2. SPARSE GATING (Hadamard with mostly-zeros)
    # ========================================
    # Target boundary: where there's ink (structure to operate on)
    target_boundary = (carrier_t < 0.5).flatten()  # dark pixels = ink

    # High gate: (gamma_h * act + bias_h) > act, constrained to boundary
    # Rearranges to: act > -bias_h / (gamma_h - 1) when gamma_h > 1
    raw_high_gate = (gamma_h * activation + bias_h) > activation
    high_gate = raw_high_gate & target_boundary  # only on ink

    # Low gate: (gamma_l * act + bias_l) < act, constrained to boundary
    # Rearranges to: act < -bias_l / (gamma_l - 1) when gamma_l > 1
    raw_low_gate = (gamma_l * activation + bias_l) < activation
    low_gate = raw_low_gate & target_boundary  # only on ink

    # Excluded middle: everything else
    excluded = ~high_gate & ~low_gate

    # Debug: gate statistics
    n_high = high_gate.sum().item()
    n_low = low_gate.sum().item()
    n_boundary = target_boundary.sum().item()
    n_both = (high_gate & low_gate).sum().item()
    print(f"    Gates: high={n_high}, low={n_low}, boundary={n_boundary}, both={n_both}")

    # ========================================
    # 3. LOCAL TANGENT (gradient of Fiedler vector)
    # ========================================
    # Fiedler vector is the 2nd eigenvector (index 1) - captures global structure
    fiedler = phi_t[:, min(1, phi_t.shape[1]-1)]  # 2nd eigenvector if available
    fiedler_2d = fiedler.reshape(H_t, W_t)

    # Tangent = gradient direction (along the curve)
    tan_y, tan_x = gradient_2d(fiedler_2d)
    tan_y, tan_x = normalize_vectors(tan_y, tan_x)

    # Orthogonal = perpendicular to tangent (across the curve)
    orth_y, orth_x = -tan_x, tan_y

    # ========================================
    # 4. VOID DIRECTION (negative gradient of distance field)
    # ========================================
    # Boundary = where there's ink (dark pixels in carrier)
    boundary = (carrier_t < 0.5).float().flatten()

    # Heat diffusion from boundary gives distance-like field
    distance = heat_diffusion_sparse(L_t, boundary, alpha=0.1, iterations=20)
    distance_2d = distance.reshape(H_t, W_t)

    # Void direction = negative gradient (toward emptiness)
    void_y, void_x = gradient_2d(distance_2d)
    void_y, void_x = normalize_vectors(-void_y, -void_x)

    # ========================================
    # 5. THICKEN (scatter high-gated pixels orthogonally)
    # ========================================
    output = target.clone()

    # Get high-gated pixel indices
    high_idx = high_gate.nonzero(as_tuple=False).squeeze(-1)

    if high_idx.numel() > 0:
        # Source positions
        src_y = high_idx // W_t
        src_x = high_idx % W_t

        # Get orthogonal direction at each source
        orth_y_src = orth_y[src_y, src_x]
        orth_x_src = orth_x[src_y, src_x]

        # Destination = source + orthogonal * width
        # Do both +ortho and -ortho for symmetric thickening
        for sign in [1.0, -1.0]:
            dst_y = (src_y.float() + sign * orth_y_src * thicken_width).round().long()
            dst_x = (src_x.float() + sign * orth_x_src * thicken_width).round().long()

            # Clip to bounds
            valid = (dst_y >= 0) & (dst_y < H_t) & (dst_x >= 0) & (dst_x < W_t)
            dst_y, dst_x = dst_y[valid], dst_x[valid]
            src_y_v, src_x_v = src_y[valid], src_x[valid]

            if dst_y.numel() > 0:
                # Preservation: don't overwrite existing structure
                dst_carrier = carrier_t[dst_y, dst_x]
                preserve = dst_carrier > 0.3  # only write to light pixels (void)

                dst_y, dst_x = dst_y[preserve], dst_x[preserve]
                src_y_v, src_x_v = src_y_v[preserve], src_x_v[preserve]

                if dst_y.numel() > 0:
                    # Scatter write with thicken color (diagnostic: darken)
                    src_colors = target[src_y_v, src_x_v]
                    # Thicken makes lines bolder - darken slightly
                    thicken_colors = src_colors * 0.7
                    output[dst_y, dst_x] = thicken_colors

    # ========================================
    # 6. DROP SHADOW (retrieve sample segment, transform, scatter)
    # ========================================
    # Pre-segment sample into connected components (sample_boundary already computed above)
    sample_components = find_connected_components(sample_boundary, H_s, W_s)
    if len(sample_components) == 0:
        return output  # no structure in sample to shadow

    # Compute spectral signature for each component (k-dimensional)
    component_spectral_sigs = []
    for comp_idx in sample_components:
        comp_phi = phi_s[comp_idx].mean(dim=0)  # (k,) mean spectral vector over component
        component_spectral_sigs.append(comp_phi)

    low_idx = low_gate.nonzero(as_tuple=False).squeeze(-1)

    if low_idx.numel() > 0:
        # Subsample seeds for efficiency
        step = max(1, low_idx.numel() // 12)
        seeds = low_idx[::step]

        for seed in seeds:
            seed_y = (seed // W_t).item()
            seed_x = (seed % W_t).item()

            # Local tangent at seed
            local_tan_y = tan_y[seed_y, seed_x].item()
            local_tan_x = tan_x[seed_y, seed_x].item()

            # Void direction at seed
            local_void_y = void_y[seed_y, seed_x].item()
            local_void_x = void_x[seed_y, seed_x].item()

            # Cross-attention: find best-matching sample component
            # seed_phi is a k-dimensional spectral vector
            seed_phi = phi_t[seed]  # (k,)
            # Component signatures should also be k-dimensional
            attention_scores = torch.stack([
                (seed_phi * sig).sum() for sig in component_spectral_sigs
            ])  # (num_components,)

            # Select component with highest absolute attention
            best_comp_idx = attention_scores.abs().argmax().item()
            segment_idx = sample_components[best_comp_idx]

            # Get segment positions and colors
            seg_y = segment_idx // W_s
            seg_x = segment_idx % W_s
            seg_colors = sample[seg_y, seg_x]

            # Compute segment centroid
            seg_cy = seg_y.float().mean()
            seg_cx = seg_x.float().mean()

            # Transform: rotate to align with local normal (perpendicular to tangent)
            angle = torch.atan2(torch.tensor(local_tan_y), torch.tensor(local_tan_x)).item()
            target_angle = angle + 3.14159 / 2  # rotate 90 degrees

            cos_a = torch.cos(torch.tensor(target_angle))
            sin_a = torch.sin(torch.tensor(target_angle))

            # Rotate segment around its centroid
            rel_y = seg_y.float() - seg_cy
            rel_x = seg_x.float() - seg_cx
            rot_y = cos_a * rel_y - sin_a * rel_x
            rot_x = sin_a * rel_y + cos_a * rel_x

            # Translate: displace into void from seed
            dst_y = (seed_y + rot_y + local_void_y * shadow_distance).round().long()
            dst_x = (seed_x + rot_x + local_void_x * shadow_distance).round().long()

            # Clip to bounds
            valid = (dst_y >= 0) & (dst_y < H_t) & (dst_x >= 0) & (dst_x < W_t)
            dst_y, dst_x = dst_y[valid], dst_x[valid]
            seg_colors_v = seg_colors[valid]

            if dst_y.numel() > 0:
                # Preservation: don't overwrite existing structure
                dst_carrier = carrier_t[dst_y, dst_x]
                preserve = dst_carrier > 0.3

                dst_y, dst_x = dst_y[preserve], dst_x[preserve]
                seg_colors_v = seg_colors_v[preserve]

                if dst_y.numel() > 0:
                    # Color rotation for diagnostic (shift hue)
                    shadow_colors = color_rotate(seg_colors_v, angle=1.0)
                    output[dst_y, dst_x] = shadow_colors

    return output


# ============================================================
# Demo
# ============================================================

def demo():
    from PIL import Image
    import numpy as np

    inp = Path("demo_output/inputs")
    out_dir = Path("demo_output")

    # Load test images
    toof = np.array(Image.open(inp / "offhand_pleometric.png").convert('RGB')).astype(np.float32) / 255.0
    tone = np.array(Image.open(inp / "red-tonegraph.png").convert('RGB')).astype(np.float32) / 255.0

    toof_t = torch.tensor(toof, dtype=torch.float32, device=DEVICE)
    tone_t = torch.tensor(tone, dtype=torch.float32, device=DEVICE)

    # Test configurations
    # Gate thresholds: high fires when act > -bias_h/(gamma_h-1)
    #                  low fires when act < -bias_l/(gamma_l-1)
    # For thin lineart, need permissive thresholds to catch boundary pixels
    # gamma=1.5, bias=-0.025 → threshold = 0.05 (high, catches act > 0.05)
    # gamma=1.5, bias=0.025 → threshold = -0.05 (low, catches act < -0.05)
    configs = [
        # name, target, sample, gamma_h, bias_h, gamma_l, bias_l
        ("v8_pleo_tone", toof_t, tone_t, 1.5, -0.025, 1.5, 0.025),
        ("v8_tone_pleo", tone_t, toof_t, 1.5, -0.025, 1.5, 0.025),
        ("v8_pleo_self", toof_t, toof_t, 1.5, -0.025, 1.5, 0.025),
        ("v8_pleo_tone_selective", toof_t, tone_t, 1.3, -0.1, 1.3, 0.1),  # more selective
    ]

    for name, target, sample, gh, bh, gl, bl in configs:
        print(f"[v8] {name}: gamma_h={gh}, bias_h={bh}, gamma_l={gl}, bias_l={bl}")
        result = shader(target, sample, gamma_h=gh, bias_h=bh, gamma_l=gl, bias_l=bl)

        # Save
        out_path = out_dir / f"{name}.png"
        result_np = (result.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(result_np).save(out_path)
        print(f"    Saved: {out_path}")

        # Threshold info
        thresh_h = -bh / (gh - 1) if gh != 1 else float('inf')
        thresh_l = -bl / (gl - 1) if gl != 1 else float('-inf')
        print(f"    Thresholds: high > {thresh_h:.2f}, low < {thresh_l:.2f}")


if __name__ == "__main__":
    demo()
