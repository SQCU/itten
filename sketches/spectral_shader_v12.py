"""
Spectral shader v12: Tensorized thickening + v8-style shadow (no fancy gating).
"""
import torch
import torch.nn.functional as F
from spectral_ops_fast import (
    build_weighted_image_laplacian,
    lanczos_k_eigenvectors,
    heat_diffusion_sparse,
    DEVICE
)


def compute_fields(carrier: torch.Tensor, L: torch.Tensor, phi: torch.Tensor):
    """Tangent, void direction, contours."""
    H, W = carrier.shape

    fiedler = phi[:, 1].reshape(H, W)
    gy = F.pad(fiedler[1:] - fiedler[:-1], (0, 0, 0, 1))
    gx = F.pad(fiedler[:, 1:] - fiedler[:, :-1], (0, 1, 0, 0))
    mag = (gy**2 + gx**2).sqrt().clamp(min=1e-8)
    tan_y, tan_x = gy / mag, gx / mag

    contours = (carrier - carrier.mean()).abs() > 0.5 * carrier.std()
    heat = heat_diffusion_sparse(L, contours.float().flatten(), 0.1, 30).reshape(H, W)
    hy = F.pad(heat[1:] - heat[:-1], (0, 0, 0, 1))
    hx = F.pad(heat[:, 1:] - heat[:, :-1], (0, 1, 0, 0))
    hmag = (hy**2 + hx**2).sqrt().clamp(min=1e-8)
    void_y, void_x = -hy / hmag, -hx / hmag

    return tan_y, tan_x, void_y, void_x, contours


def thicken(output: torch.Tensor, src_mask: torch.Tensor, phi: torch.Tensor,
            tan_y: torch.Tensor, tan_x: torch.Tensor, contours: torch.Tensor, max_r: int = 3):
    """Perpendicular scatter with spectral-modulated budget."""
    H, W, C = output.shape
    device = output.device

    src_idx = src_mask.nonzero()
    if src_idx.shape[0] == 0:
        return output
    M = src_idx.shape[0]

    flat_idx = src_idx[:, 0] * W + src_idx[:, 1]
    if phi.shape[1] > 2:
        energy = (phi[flat_idx, 2:] ** 2).sum(dim=1)
        energy = energy / (energy.max() + 1e-8)
        budget = max_r * (1 - 0.7 * energy)
    else:
        budget = torch.full((M,), max_r, device=device)

    ty = tan_y[src_idx[:, 0], src_idx[:, 1]]
    tx = tan_x[src_idx[:, 0], src_idx[:, 1]]
    perp_y, perp_x = -tx, ty

    offsets = torch.arange(-max_r, max_r + 1, device=device, dtype=torch.float32)
    offsets = offsets[offsets != 0]
    K = offsets.shape[0]

    dst_y = src_idx[:, 0:1].float() + perp_y[:, None] * offsets[None, :]
    dst_x = src_idx[:, 1:2].float() + perp_x[:, None] * offsets[None, :]

    within_budget = offsets.abs()[None, :] <= budget[:, None]
    in_bounds = (dst_y >= 0) & (dst_y < H) & (dst_x >= 0) & (dst_x < W)
    dst_y_int = dst_y.long().clamp(0, H-1)
    dst_x_int = dst_x.long().clamp(0, W-1)
    not_on_contour = ~contours[dst_y_int, dst_x_int]

    accept = within_budget & in_bounds & not_on_contour
    src_colors = output[src_idx[:, 0], src_idx[:, 1]]
    flat_dst = dst_y_int * W + dst_x_int

    output.view(-1, C)[flat_dst[accept]] = src_colors[:, None, :].expand(-1, K, -1)[accept]
    return output


def color_rotate(rgb: torch.Tensor, angle: float) -> torch.Tensor:
    """Hue rotation."""
    cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.333 * (r + g + b)
    r_c, g_c, b_c = r - lum, g - lum, b - lum
    new_r = lum + cos_a * r_c - sin_a * g_c
    new_g = lum + sin_a * r_c + cos_a * g_c
    return torch.stack([new_r, new_g, b], dim=-1).clamp(0, 1)


def find_components_tensored(boundary_flat: torch.Tensor, H: int, W: int, min_size: int = 5):
    """
    v8-style connected components but tensored.
    Returns list of flat index tensors.
    """
    device = boundary_flat.device
    boundary_2d = boundary_flat.reshape(H, W)
    visited = torch.zeros(H, W, dtype=torch.bool, device=device)
    components = []

    # Find all boundary positions
    positions = boundary_2d.nonzero()  # (N, 2)
    if positions.shape[0] == 0:
        return components

    # Iterate through unvisited seeds
    for i in range(positions.shape[0]):
        y, x = positions[i, 0].item(), positions[i, 1].item()
        if visited[y, x]:
            continue

        # Flood fill via dilation
        component = torch.zeros(H, W, dtype=torch.bool, device=device)
        component[y, x] = True

        for _ in range(max(H, W)):
            dilated = F.max_pool2d(component.float()[None, None], 3, 1, 1)[0, 0] > 0
            grown = dilated & boundary_2d & ~visited
            if not (grown & ~component).any():
                break
            component = component | grown

        visited = visited | component
        flat_idx = component.flatten().nonzero().squeeze(-1)
        if flat_idx.numel() >= min_size:
            components.append(flat_idx)

    return components


def shadow(output: torch.Tensor, phi_t: torch.Tensor, phi_s: torch.Tensor,
           low_gate: torch.Tensor, carrier_t: torch.Tensor,
           tan_y: torch.Tensor, tan_x: torch.Tensor,
           void_y: torch.Tensor, void_x: torch.Tensor,
           sample: torch.Tensor, sample_boundary: torch.Tensor,
           shadow_distance: float = 15.0, is_self: bool = False,
           L_t: torch.Tensor = None):
    """Shadow via attention-weighted copying. No segment iteration."""
    H, W = output.shape[:2]
    H_s, W_s = sample.shape[:2]
    device = output.device

    if is_self:
        # Self-attention: v6 style with segment iteration (keep working version)
        low_components = find_components_tensored(low_gate.flatten(), H, W, min_size=10)
        if not low_components:
            return output

        print(f"    shadow (self): {len(low_components)} components from low_gate")

        for comp_idx in low_components:
            if comp_idx.numel() < 5:
                continue

            comp_y = comp_idx // W
            comp_x = comp_idx % W
            cy, cx = comp_y.float().mean(), comp_x.float().mean()
            cy_int, cx_int = int(cy.item()), int(cx.item())
            cy_int = max(0, min(H-1, cy_int))
            cx_int = max(0, min(W-1, cx_int))

            seg_colors = output[comp_y.clamp(0, H-1), comp_x.clamp(0, W-1)]

            rel_y = comp_y.float() - cy
            rel_x = comp_x.float() - cx
            rot_y = -rel_x
            rot_x = rel_y

            local_void_y = void_y[cy_int, cx_int].item()
            local_void_x = void_x[cy_int, cx_int].item()

            dst_y = (cy + rot_y + local_void_y * shadow_distance).round().long()
            dst_x = (cx + rot_x + local_void_x * shadow_distance).round().long()

            valid = (dst_y >= 0) & (dst_y < H) & (dst_x >= 0) & (dst_x < W)
            dst_y, dst_x = dst_y[valid], dst_x[valid]
            seg_colors_v = seg_colors[valid]

            if dst_y.numel() > 0:
                preserve = carrier_t[dst_y, dst_x] > 0.3
                dst_y, dst_x = dst_y[preserve], dst_x[preserve]
                seg_colors_v = seg_colors_v[preserve]
                if dst_y.numel() > 0:
                    output[dst_y, dst_x] = color_rotate(seg_colors_v, 1.0)

        return output

    # CROSS-ATTENTION: counterfactual self-attention → match sample to residuals

    # 1. Build counterfactual buffer: what SELF-attention would produce
    counterfactual = output.clone()
    placements = []  # track what self-attn places and where

    low_components = find_components_tensored(low_gate.flatten(), H, W, min_size=10)
    if not low_components:
        print(f"    shadow (cross): no low_gate components")
        return output

    # Soft scale: filter by component size and activation strength
    # Keep only larger components (top 50% by size)
    comp_sizes = torch.tensor([c.numel() for c in low_components], device=device)
    size_thresh = comp_sizes.median()
    filtered_components = [c for c, s in zip(low_components, comp_sizes) if s >= size_thresh]

    # Compute heat field for open-direction selection
    contours_for_heat = (carrier_t - carrier_t.mean()).abs() > 0.5 * carrier_t.std()
    heat = heat_diffusion_sparse(L_t, contours_for_heat.float().flatten(), 0.1, 30).reshape(H, W)

    for comp_idx in filtered_components:
        if comp_idx.numel() < 5:
            continue

        comp_y = comp_idx // W
        comp_x = comp_idx % W
        cy, cx = comp_y.float().mean(), comp_x.float().mean()
        cy_int, cx_int = int(cy.item()), int(cx.item())
        cy_int = max(0, min(H-1, cy_int))
        cx_int = max(0, min(W-1, cx_int))

        # Source colors from target (what self would use)
        seg_colors = output[comp_y.clamp(0, H-1), comp_x.clamp(0, W-1)]

        # No rotation, no translation - just place at original positions
        dst_y = comp_y
        dst_x = comp_x

        valid = (dst_y >= 0) & (dst_y < H) & (dst_x >= 0) & (dst_x < W)
        dst_y_v, dst_x_v = dst_y[valid], dst_x[valid]
        seg_colors_v = seg_colors[valid]

        if dst_y_v.numel() > 0:
            # Write to counterfactual buffer (no preserve check for ablation)
            counterfactual[dst_y_v, dst_x_v] = color_rotate(seg_colors_v, 1.0)

            # Track placement: source component signature + destination indices
            src_sig = phi_t[comp_idx.clamp(0, phi_t.shape[0]-1)].mean(dim=0)
            placements.append({
                'src_sig': src_sig,
                'dst_flat': dst_y_v * W + dst_x_v,
                'src_idx': comp_idx,
            })

    if not placements:
        print(f"    shadow (cross): no placements from counterfactual")
        return output

    # 2. Pre-segment sample into components with signatures
    sample_components = find_components_tensored(sample_boundary, H_s, W_s, min_size=10)
    if not sample_components:
        print(f"    shadow (cross): no sample components")
        return output

    sample_sigs = torch.stack([phi_s[c.clamp(0, phi_s.shape[0]-1)].mean(dim=0) for c in sample_components])

    print(f"    shadow (cross): {len(placements)} counterfactual placements → {len(sample_components)} sample components")

    # 3. Collect fragments: for each placement, get sample component with intrinsic properties
    fragments = []
    for p in placements:
        scores = (p['src_sig'].unsqueeze(0) @ sample_sigs.T).abs().squeeze(0)
        best_idx = scores.argmax().item()
        best_comp = sample_components[best_idx]

        if best_comp.numel() < 5:
            continue

        # Sample component positions as (N, 2) tensor [y, x]
        samp_pos = torch.stack([
            (best_comp // W_s).float(),
            (best_comp % W_s).float()
        ], dim=1)  # (N, 2)

        samp_colors = sample[samp_pos[:, 0].long().clamp(0, H_s-1),
                             samp_pos[:, 1].long().clamp(0, W_s-1)]

        # Intrinsic properties of THIS sample component
        samp_centroid = samp_pos.mean(dim=0)  # (2,)
        samp_centered = samp_pos - samp_centroid

        cov = (samp_centered.T @ samp_centered) / (samp_centered.shape[0] + 1e-8)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        frag_length = 4.0 * eigvals[-1].sqrt()
        frag_axis = eigvecs[:, -1]

        # Destination centroid from target component
        src_pos = torch.stack([
            (p['src_idx'] // W).float(),
            (p['src_idx'] % W).float()
        ], dim=1)
        dst_centroid = src_pos.mean(dim=0)

        fragments.append({
            'rel_pos': samp_centered,  # positions relative to centroid
            'colors': samp_colors,
            'length': frag_length,
            'axis': frag_axis,
            'dst_centroid': dst_centroid,
            'samp_centroid': samp_centroid,  # original position in sample
        })

    if not fragments:
        print(f"    shadow (cross): no fragments collected")
        return output

    lengths = [f['length'].item() for f in fragments[:5]]
    print(f"    shadow (cross): collected {len(fragments)} fragments, lengths: {lengths}")

    # 4. Transform fragments: no rotation for now, just place at destination
    transformed = []
    for frag in fragments:
        # No rotation - just place at destination centroid
        new_pos = frag['rel_pos'] + frag['dst_centroid']

        transformed.append({
            'pos': new_pos,
            'colors': frag['colors'],
        })

    # 5. Collapse all fragments into output (sparse buffer approach)
    for t in transformed:
        pos_y = t['pos'][:, 0].round().long()
        pos_x = t['pos'][:, 1].round().long()
        valid = (pos_y >= 0) & (pos_y < H) & (pos_x >= 0) & (pos_x < W)
        pos_y, pos_x = pos_y[valid], pos_x[valid]
        colors = t['colors'][valid]

        if pos_y.numel() > 0:
            output[pos_y, pos_x] = color_rotate(colors, 1.0)

    return output


def align_spectral_bases(phi_t: torch.Tensor, phi_s: torch.Tensor,
                         H: int, W: int, H_s: int, W_s: int,
                         n_samples: int = 1000) -> torch.Tensor:
    """
    Procrustes alignment: rotate φ_s into φ_t's coordinate frame.
    Uses spatial correspondence (same relative position) as anchor points.
    """
    device = phi_t.device

    n_samples = min(n_samples, H * W, H_s * W_s)

    # Random pixel indices in target
    t_idx = torch.randperm(H * W, device=device)[:n_samples]
    t_y, t_x = t_idx // W, t_idx % W

    # Corresponding positions in sample (nearest neighbor by relative position)
    s_y = (t_y.float() * H_s / H).long().clamp(0, H_s - 1)
    s_x = (t_x.float() * W_s / W).long().clamp(0, W_s - 1)
    s_idx = s_y * W_s + s_x

    # Subset for alignment
    phi_t_sub = phi_t[t_idx]
    phi_s_sub = phi_s[s_idx]

    # Procrustes: R = U @ Vh where M = φ_s^T @ φ_t
    M = phi_s_sub.T @ phi_t_sub
    U, S, Vh = torch.linalg.svd(M)
    R = U @ Vh

    return phi_s @ R


def shader(target: torch.Tensor, sample: torch.Tensor = None,
           gamma_h: float = 1.3, bias_h: float = -0.15,
           gamma_l: float = 1.5, bias_l: float = -0.08,
           k: int = 8, thick_r: int = 3, shadow_dist: float = 15.0):
    """Spectral shader with Procrustes-aligned cross-attention."""
    if sample is None:
        sample = target

    H, W = target.shape[:2]
    H_s, W_s = sample.shape[:2]
    device = target.device

    carrier_t = target.mean(dim=-1)
    carrier_s = sample.mean(dim=-1)
    L_t = build_weighted_image_laplacian(carrier_t, edge_threshold=0.1)
    phi_t, _ = lanczos_k_eigenvectors(L_t, k, 50)
    phi_t = torch.tensor(phi_t, device=device, dtype=torch.float32)

    cross = not torch.equal(target, sample)
    if cross:
        L_s = build_weighted_image_laplacian(carrier_s, edge_threshold=0.1)
        phi_s, _ = lanczos_k_eigenvectors(L_s, k, 50)
        phi_s = torch.tensor(phi_s, device=device, dtype=torch.float32)
        phi_s = align_spectral_bases(phi_t, phi_s, H, W, H_s, W_s)
    else:
        phi_s = phi_t

    tan_y, tan_x, void_y, void_x, contours_t = compute_fields(carrier_t, L_t, phi_t)

    # v8: sample boundary = ink pixels
    sample_boundary = (carrier_s < 0.5).flatten()

    # Activation and gates
    fiedler = phi_t[:, 1].reshape(H, W)
    fiedler = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-8)
    act = torch.sigmoid((fiedler - fiedler.mean()) / (fiedler.std() + 1e-8) * 10)

    high_gate = ((gamma_h * act + bias_h) > act) & contours_t
    low_gate = ((gamma_l * act + bias_l) < act) & contours_t

    output = target.clone()
    output = thicken(output, high_gate, phi_t, tan_y, tan_x, contours_t, thick_r)
    output = shadow(output, phi_t, phi_s, low_gate, carrier_t,
                    tan_y, tan_x, void_y, void_x, sample, sample_boundary, shadow_dist,
                    is_self=not cross, L_t=L_t)

    return output


if __name__ == "__main__":
    from pathlib import Path
    from PIL import Image
    import numpy as np
    from datetime import datetime

    inp, out = Path("demo_output/inputs"), Path("demo_output")

    def load(name):
        p = inp / name
        return torch.tensor(np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0, device=DEVICE) if p.exists() else None

    toof, snek, tone = load("toof.png"), load("snek-heavy.png"), load("red-tonegraph.png")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save(img, name):
        Image.fromarray((img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)).save(out / f"{ts}_v12_{name}.png")
        print(f"  {name}")

    print(f"\n[v12] v8-style shadow")
    if toof is not None:
        save(shader(toof), "self_toof")
    if toof is not None and snek is not None:
        save(shader(toof, snek), "toof_x_snek")
    if snek is not None and tone is not None:
        save(shader(snek, tone), "snek_x_tone")
