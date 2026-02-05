"""
Spectral shader v11: Collapsed tensor-native implementation.
Target: <300 LOC, all operations as matmuls/einsum.
"""
import torch
import torch.nn.functional as F
import math
from spectral_ops_fast import (
    build_weighted_image_laplacian,
    lanczos_k_eigenvectors,
    heat_diffusion_sparse,
    DEVICE
)
from shadow_solver import solve_placements_vectorized, perpendicular_soft

# ============================================================
# PRIMITIVES (5 functions, each <15 lines)
# ============================================================

def gradient_2d(f: torch.Tensor) -> torch.Tensor:
    """(H, W) -> (H, W, 2) gradient field [dy, dx]."""
    g = torch.zeros(*f.shape, 2, device=f.device)
    g[:-1, :, 0] = f[1:, :] - f[:-1, :]
    g[:, :-1, 1] = f[:, 1:] - f[:, :-1]
    return g

def normalize_field(g: torch.Tensor) -> torch.Tensor:
    """(H, W, 2) -> (H, W, 2) unit vectors."""
    mag = g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return g / mag

def rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
    """(A,) angles -> (A, 2, 2) rotation matrices."""
    c, s = angles.cos(), angles.sin()
    R = torch.zeros(angles.shape[0], 2, 2, device=angles.device)
    R[:, 0, 0], R[:, 0, 1] = c, -s
    R[:, 1, 0], R[:, 1, 1] = s, c
    return R

def connected_components_sparse(mask: torch.Tensor, min_size: int, max_comps: int) -> torch.Tensor:
    """Returns (N, 3) tensor: [y, x, component_id] for all component pixels."""
    H, W = mask.shape
    remaining = mask.clone()
    all_coords = []
    comp_id = 0
    while remaining.any() and comp_id < max_comps:
        seed = remaining.nonzero()[0]
        frontier = torch.zeros_like(remaining)
        frontier[seed[0], seed[1]] = True
        component = frontier.clone()
        for _ in range(max(H, W)):
            expanded = F.max_pool2d(frontier.float().unsqueeze(0).unsqueeze(0), 3, 1, 1).squeeze() > 0
            expanded &= remaining
            if not (expanded & ~component).any():
                break
            component |= expanded
            frontier = expanded
        remaining &= ~component
        coords = component.nonzero()
        if coords.shape[0] >= min_size:
            ids = torch.full((coords.shape[0], 1), comp_id, device=mask.device)
            all_coords.append(torch.cat([coords, ids], dim=1))
            comp_id += 1
    return torch.cat(all_coords, dim=0) if all_coords else torch.zeros(0, 3, device=mask.device)

def cyclic_color(colors: torch.Tensor, phase: float, blend: float) -> torch.Tensor:
    """Cyclic color transform. colors: (..., 3)."""
    lum = (colors * torch.tensor([0.299, 0.587, 0.114], device=colors.device)).sum(dim=-1, keepdim=True)
    theta = 2 * math.pi * lum + phase
    offsets = torch.tensor([0, 2*math.pi/3, 4*math.pi/3], device=colors.device)
    cyc = 0.5 + 0.4 * (theta + offsets).sin()
    return colors * (1 - blend) + cyc * blend

def coords_to_flat(coords: torch.Tensor, shape: tuple) -> torch.Tensor:
    """(*, D) coords -> (*,) flat indices for shape (S0, S1, ..., S_{D-1})."""
    D = coords.shape[-1]
    strides = torch.tensor([1], device=coords.device)
    for s in reversed(shape[1:]):
        strides = torch.cat([strides, strides[-1:] * s])
    strides = strides.flip(0)  # (D,)
    return (coords * strides).sum(dim=-1).long()

def gather_nd(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Gather from field at coords. field: (S0,...,S_{D-1}, C), coords: (*, D) -> (*, C)."""
    shape = field.shape[:-1] if field.dim() > coords.shape[-1] else field.shape
    C = field.shape[-1] if field.dim() > len(shape) else 1
    flat = coords_to_flat(coords.clamp(min=0), shape)
    # Clamp flat indices to valid range
    flat = flat.clamp(0, field.view(-1, C).shape[0] - 1) if C > 1 else flat.clamp(0, field.numel() - 1)
    if C > 1:
        return field.view(-1, C)[flat]
    return field.view(-1)[flat]

def energy_minimize_placement(
    relative: torch.Tensor,      # (N, D) fragment coords relative to centroid
    frag_axis: torch.Tensor,     # (D,) principal axis of fragment
    seed: torch.Tensor,          # (D,) seed position
    seed_tangent: torch.Tensor,  # (D,) tangent direction at seed point
    tangent: torch.Tensor,       # (*shape, D) tangent field
    geom_mask: torch.Tensor,     # (*shape) bool - existing geometry
    dist_field: torch.Tensor,    # (*shape) distance to geometry
    void_dir: torch.Tensor,      # (D,) direction toward void at seed
    shape: tuple,                # spatial dimensions
    n_angles: int = 16, n_dists: int = 8, max_dist: float = 30.0,
    w_src: float = 2.0, w_seed: float = 4.0, w_loc: float = 1.0, w_int: float = 1.5, w_void: float = 0.8,
) -> tuple[float, float]:
    """Energy-minimizing (angle, distance) via batched grid search. D-agnostic.

    Key energies:
    - E_seed: orthogonal to LOCAL TANGENT at seed (PRIMARY - this is what user wants)
    - E_src: orthogonal to fragment's own axis (secondary)
    - E_loc: orthogonal to geometry at landing position
    Combined: fragments SPIN perpendicular to the contour they're decorating.
    """
    device = relative.device
    D = relative.shape[-1]
    N = relative.shape[0]

    # Candidate grid - bias toward orthogonal angles (π/2 ± wiggle)
    # Include angles near 90° and 270° with finer resolution
    base_angles = torch.linspace(0, 2*math.pi*(1-1/n_angles), n_angles, device=device)
    angles = base_angles
    dists = torch.linspace(5.0, max_dist, n_dists, device=device)
    A, T = n_angles, n_dists

    # Rotation matrices
    R = rotation_matrix(angles)

    # Rotate fragment and axis
    rotated = torch.einsum('nd,adc->anc', relative, R)  # (A, N, D)
    rot_axis = torch.einsum('d,adc->ac', frag_axis, R)  # (A, D)

    # Translate: (A, T, N, D) - all broadcasting, no D-specific indexing
    placed = rotated[:, None, :, :] + seed + dists[None, :, None, None] * void_dir

    # Clamp coords to valid range
    shape_t = torch.tensor(shape, device=device).float()
    placed_clamped = placed.clamp(min=0)
    for d in range(D):
        placed_clamped[..., d] = placed_clamped[..., d].clamp(max=shape_t[d] - 1)

    # Gather context via flat indexing - no y/x split
    placed_shape = placed_clamped.shape[:-1]  # (A, T, N)
    placed_flat = placed_clamped.reshape(-1, D)  # (A*T*N, D)

    local_tan = gather_nd(tangent, placed_flat).reshape(*placed_shape, D)  # (A, T, N, D)
    on_geom = gather_nd(geom_mask.unsqueeze(-1), placed_flat).reshape(*placed_shape)  # (A, T, N)
    at_dist = gather_nd(dist_field.unsqueeze(-1), placed_flat).reshape(*placed_shape)  # (A, T, N)

    # E_seed: penalize parallel to LOCAL TANGENT at seed - THIS IS PRIMARY
    # Fragment should be PERPENDICULAR to the contour it's decorating
    dot_seed = (rot_axis * seed_tangent).sum(dim=-1)  # (A,)
    E_seed = (dot_seed.abs() ** 2)[:, None].expand(A, T)  # squared for sharp penalty

    # E_source: penalize parallel to fragment's own source axis (secondary)
    dot_src = (rot_axis * frag_axis).sum(dim=-1)  # (A,)
    E_src = (dot_src.abs() ** 2)[:, None].expand(A, T)

    # E_local: penalize parallel to geometry at landing position
    dot_loc = (rot_axis[:, None, None, :] * local_tan).sum(dim=-1)
    E_loc = dot_loc.abs().mean(dim=-1)

    # E_intersection: overlap penalty, discounted for orthogonal crossing
    E_int = (on_geom.float() * (1.0 - 0.3 * (1.0 - dot_loc.abs()))).sum(dim=-1)

    # E_void: reward emptiness (negative = good)
    E_void = -at_dist.mean(dim=-1)

    # Total: E_seed dominates to enforce perpendicular to local contour
    E = w_seed * E_seed + w_src * E_src + w_loc * E_loc + w_int * E_int + w_void * E_void
    flat_idx = E.argmin()
    return angles[flat_idx // T].item(), dists[flat_idx % T].item()

# ============================================================
# MAIN SHADER (<100 lines)
# ============================================================

def shader_v11(
    target: torch.Tensor,
    sample: torch.Tensor = None,
    # Gate params: low_gate = (gamma * act + bias) < act
    # Threshold where gate flips: threshold = bias / (1 - gamma)
    # With gamma > 1: MORE negative bias → HIGHER threshold → MORE selections
    #   gamma_l=1.5, bias_l=-0.05 → threshold=0.10 → ~10% lowest activity
    #   gamma_l=1.5, bias_l=-0.10 → threshold=0.20 → ~20% lowest activity
    #   gamma_l=1.5, bias_l=-0.15 → threshold=0.30 → ~30% lowest activity
    gamma_h: float = 1.3, bias_h: float = -0.18,
    gamma_l: float = 1.5, bias_l: float = -0.08,  # threshold ~0.16 → sparse seeds
    # Thickening
    thick_radius: int = 4, r_in: float = 2.0, r_out: float = 2.0,
    # Shadow - sizes as fractions of image diameter, not magic pixel counts
    shadow_dist_frac: float = 0.05,   # shadow distance as fraction of diagonal
    shadow_offset_frac: float = 0.02, # offset between layers as fraction of diagonal
    min_seg_frac: float = 0.02,       # min segment length as fraction of diagonal
    max_seg_frac: float = 0.15,       # max segment length as fraction of diagonal
    # Spectral
    k: int = 8,
) -> tuple[torch.Tensor, dict]:
    """Unified spectral shader: thickening (high gate) + shadow (low gate)."""
    if sample is None:
        sample = target
    H, W = target.shape[:2]
    N = H * W
    device = target.device
    cross = not torch.equal(target, sample)

    # Convert fractional parameters to pixel values based on image diagonal
    diag = (H**2 + W**2) ** 0.5
    shadow_dist = shadow_dist_frac * diag
    shadow_offset = shadow_offset_frac * diag
    min_seg = max(3, int(min_seg_frac * diag))
    max_seg = max(min_seg + 1, int(max_seg_frac * diag))

    # === CARRIER & LAPLACIAN ===
    carrier_t = target.mean(dim=-1)
    carrier_s = sample.mean(dim=-1) if cross else carrier_t
    L_t = build_weighted_image_laplacian(carrier_t, edge_threshold=0.1)

    # === SPECTRAL EMBEDDING ===
    phi_t, _ = lanczos_k_eigenvectors(L_t, k, 50)
    phi_t = torch.tensor(phi_t, device=device, dtype=torch.float32)
    fiedler = phi_t[:, 1].reshape(H, W)
    fiedler = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-8)

    # === CONTOUR DETECTION (needed for tangent and distance) ===
    contours = (carrier_t - carrier_t.mean()).abs() > 0.7 * carrier_t.std()

    # === DIRECTION FIELDS (computed ONCE) ===
    grad = gradient_2d(carrier_t)  # (H, W, 2)
    grad_norm = normalize_field(grad)
    grad_mag = grad.norm(dim=-1)

    # Tangent from CONTOUR GEOMETRY via structure tensor
    # This gives actual edge direction, not carrier intensity gradient
    contour_grad = gradient_2d(contours.float())
    Jxx = F.avg_pool2d(contour_grad[..., 1:2].permute(2,0,1).unsqueeze(0)**2, 5, 1, 2).squeeze()
    Jyy = F.avg_pool2d(contour_grad[..., 0:1].permute(2,0,1).unsqueeze(0)**2, 5, 1, 2).squeeze()
    Jxy = F.avg_pool2d((contour_grad[..., 0:1] * contour_grad[..., 1:2]).permute(2,0,1).unsqueeze(0), 5, 1, 2).squeeze()
    theta = 0.5 * torch.atan2(2 * Jxy, Jxx - Jyy + 1e-8)
    tangent = torch.stack([theta.sin(), theta.cos()], dim=-1)  # (H, W, 2) edge direction

    # Distance field via heat diffusion
    dist_flat = heat_diffusion_sparse(L_t, contours.float().flatten(), 0.1, 30)
    distance = 1.0 - dist_flat.reshape(H, W) / (dist_flat.max() + 1e-8)
    void_dir = -normalize_field(gradient_2d(distance))

    # === GATING ===
    act = torch.sigmoid((fiedler - fiedler.mean()) / (fiedler.std() + 1e-8) * 10)
    high_gate = (gamma_h * act + bias_h) > act
    low_gate = (gamma_l * act + bias_l) < act

    output = target.clone()
    stats = {'contours': contours.sum().item(), 'high': (contours & high_gate).sum().item(),
             'low': (contours & low_gate).sum().item()}

    # === THICKENING (HIGH GATE) - V-PROJECTION STRATIFIED ===
    high_coords = (contours & high_gate).nonzero().float()  # (M, 2)
    if high_coords.shape[0] > 0:
        M = high_coords.shape[0]
        src_idx = high_coords.long()

        # === AFFINITY SCORE (Q·K^T analog) ===
        # Use higher eigenvectors to detect short-range spectral activity
        # High activity = bunching tendency = penalize with gamma < 1
        src_flat = src_idx[:, 0] * W + src_idx[:, 1]
        if phi_t.shape[1] > 2:
            # Short-range activity from eigenvectors 2+ (skip DC and Fiedler)
            high_freq_energy = (phi_t[src_flat, 2:] ** 2).sum(dim=1)  # (M,)
            high_freq_norm = high_freq_energy / (high_freq_energy.max() + 1e-8)
            # Gamma penalty: high short-range activity -> lower affinity for thickening
            gamma_penalty = 0.85  # < 1 penalizes bunching
            affinity = gamma_penalty ** (high_freq_norm * 3)  # (M,) in [gamma^3, 1]
        else:
            affinity = torch.ones(M, device=device)

        # === STRATIFIED BUDGET ALLOCATION ===
        # Sort by affinity, assign thickening budget per stratum
        # Top 25% affinity gets only 40% of budget (prevents hot spots)
        sorted_idx = affinity.argsort(descending=True)
        budget_per_source = torch.zeros(M, device=device)

        total_budget = M * 4  # total thickening pixels to distribute
        strata = [(0.25, 0.40), (0.25, 0.30), (0.25, 0.20), (0.25, 0.10)]  # (fraction of sources, fraction of budget)

        pos = 0
        for src_frac, budget_frac in strata:
            n_src = int(M * src_frac)
            stratum_budget = total_budget * budget_frac
            if n_src > 0:
                per_src = stratum_budget / n_src
                stratum_idx = sorted_idx[pos:pos + n_src]
                budget_per_source[stratum_idx] = per_src
            pos += n_src

        # === GENERATE CANDIDATES WITH BUDGET-LIMITED ACCEPTANCE ===
        offsets = []
        for dy in range(-thick_radius, thick_radius + 1):
            for dx in range(-thick_radius, thick_radius + 1):
                if 0 < dy*dy + dx*dx <= thick_radius*thick_radius:
                    offsets.append([dy, dx])
        offsets = torch.tensor(offsets, device=device, dtype=torch.float32)
        K = offsets.shape[0]

        candidates = high_coords.unsqueeze(1) + offsets.unsqueeze(0)  # (M, K, 2)
        cand_flat = candidates.reshape(-1, 2)
        cand_idx = cand_flat.long().clamp(min=0)
        cand_idx[:, 0] = cand_idx[:, 0].clamp(max=H-1)
        cand_idx[:, 1] = cand_idx[:, 1].clamp(max=W-1)

        dist_from_src = offsets.norm(dim=-1).unsqueeze(0).expand(M, -1)

        # Inside/outside
        normal_at_src = grad_norm[src_idx[:, 0], src_idx[:, 1]]
        dot = (offsets.unsqueeze(0) * normal_at_src.unsqueeze(1)).sum(dim=-1)
        inside = dot < 0
        r_eff = torch.where(inside,
                           torch.full_like(dist_from_src, r_in),
                           torch.full_like(dist_from_src, r_out))

        # Base acceptance (distance < radius)
        in_bounds = (cand_flat[:, 0] >= 0).reshape(M, K) & (cand_flat[:, 0] < H).reshape(M, K) & \
                    (cand_flat[:, 1] >= 0).reshape(M, K) & (cand_flat[:, 1] < W).reshape(M, K)
        base_accept = (dist_from_src < r_eff) & in_bounds

        # Budget-limited acceptance: for each source, accept up to budget_per_source[m] candidates
        # Sort candidates by distance, accept closest first up to budget
        accept = torch.zeros_like(base_accept)
        for m in range(M):
            mask_m = base_accept[m]
            if not mask_m.any():
                continue
            budget_m = int(budget_per_source[m].item())
            if budget_m <= 0:
                continue
            # Accept closest candidates first, up to budget
            dists_m = dist_from_src[m]
            valid_dists = torch.where(mask_m, dists_m, torch.tensor(float('inf'), device=device))
            _, order = valid_dists.sort()
            accept[m, order[:budget_m]] = mask_m[order[:budget_m]]

        # Scatter
        src_expand = torch.arange(M, device=device).unsqueeze(1).expand(-1, K)
        accepted_src = src_expand[accept]
        accepted_dst = cand_flat[accept.flatten()].long()
        src_colors = target[src_idx[:, 0], src_idx[:, 1]]
        accepted_dst_clamped = accepted_dst.clamp(min=0)
        accepted_dst_clamped[:, 0] = accepted_dst_clamped[:, 0].clamp(max=H-1)
        accepted_dst_clamped[:, 1] = accepted_dst_clamped[:, 1].clamp(max=W-1)
        flat_dst = coords_to_flat(accepted_dst_clamped, (H, W))
        output.view(-1, 3)[flat_dst] = src_colors[accepted_src]
        stats['thickened'] = accept.sum().item()

    # === SHADOW (LOW GATE) - BATCHED CROSS-ATTENTION ===
    H_s, W_s = sample.shape[:2]
    N_s = H_s * W_s
    if cross:
        L_s = build_weighted_image_laplacian(carrier_s, edge_threshold=0.1)
        phi_s, _ = lanczos_k_eigenvectors(L_s, k, 50)
        phi_s = torch.tensor(phi_s, device=device, dtype=torch.float32)
        contours_s = (carrier_s - carrier_s.mean()).abs() > 0.7 * carrier_s.std()
        comp_data = connected_components_sparse(contours_s, min_seg, max_seg)
    else:
        phi_s = phi_t
        comp_data = connected_components_sparse(contours & low_gate, min_seg, max_seg)

    if comp_data.shape[0] > 0:
        # Component signatures via scatter-mean
        comp_ids = comp_data[:, 2].long()
        comp_coords = comp_data[:, :2].long()
        n_comps = comp_ids.max().item() + 1
        # Use SAMPLE dimensions for sample indexing
        flat_idx = comp_coords[:, 0] * W_s + comp_coords[:, 1]

        # Signature = mean phi over component
        phi_at_comp = phi_s[flat_idx.clamp(0, N_s-1)]  # (total_pixels, k)
        sigs = torch.zeros(n_comps, phi_s.shape[1], device=device)
        counts = torch.zeros(n_comps, device=device)
        sigs.index_add_(0, comp_ids, phi_at_comp)
        counts.index_add_(0, comp_ids, torch.ones_like(comp_ids, dtype=torch.float32))
        sigs = sigs / counts.unsqueeze(1).clamp(min=1)  # (C, k)

        # Seeds from low gate - gating params control density
        low_coords = (contours & low_gate).nonzero()
        seeds = low_coords

        # Cross-attention: all seeds vs all signatures
        seed_flat = seeds[:, 0] * W + seeds[:, 1]
        seed_phi = phi_t[seed_flat.clamp(0, N-1)]  # (S, k)
        scores = seed_phi @ sigs.T  # (S, C) - THE CROSS-ATTENTION MATMUL
        best_comp = scores.abs().argmax(dim=1)  # (S,)

        S = seeds.shape[0]
        stats['seeds'] = S
        stats['components'] = n_comps
        stats['angles'] = []

        # === BATCH PREPARATION ===
        # Collect fragment data for all seeds, pad to max_N
        fragments_list = []
        frag_colors_list = []
        frag_axes_list = []
        valid_seed_mask = []

        for i in range(S):
            comp_mask = comp_ids == best_comp[i].item()
            if not comp_mask.any():
                valid_seed_mask.append(False)
                continue
            valid_seed_mask.append(True)
            frag_coords = comp_data[comp_mask, :2].float()
            frag_y = frag_coords[:, 0].long().clamp(0, H_s-1)
            frag_x = frag_coords[:, 1].long().clamp(0, W_s-1)
            colors = sample[frag_y, frag_x]
            centroid = frag_coords.mean(dim=0)
            relative = frag_coords - centroid

            # Principal axis via covariance
            if relative.shape[0] > 2:
                cov = relative.T @ relative / relative.shape[0]
                _, evecs = torch.linalg.eigh(cov)
                axis = evecs[:, -1]
            else:
                axis = torch.tensor([1.0, 0.0], device=device)

            fragments_list.append(relative)
            frag_colors_list.append(colors)
            frag_axes_list.append(axis)

        valid_seed_mask = torch.tensor(valid_seed_mask, device=device)
        if not valid_seed_mask.any():
            return output, stats

        # Filter to valid seeds
        valid_seeds = seeds[valid_seed_mask].float()  # (B, 2)
        B = valid_seeds.shape[0]

        # Pad fragments to max_N
        max_N = max(f.shape[0] for f in fragments_list)
        fragments_padded = torch.zeros(B, max_N, 2, device=device)
        frag_masks = torch.zeros(B, max_N, dtype=torch.bool, device=device)
        frag_colors_padded = torch.zeros(B, max_N, 3, device=device)

        for i, (frag, colors) in enumerate(zip(fragments_list, frag_colors_list)):
            n = frag.shape[0]
            fragments_padded[i, :n] = frag
            frag_masks[i, :n] = True
            frag_colors_padded[i, :n] = colors

        frag_axes = torch.stack(frag_axes_list)  # (B, 2)

        # Gather tangent and outward direction at seed positions
        # Use CARRIER GRADIENT (not distance field) - points from dark lines to light background
        seed_idx = valid_seeds.long().clamp(min=0)
        seed_idx[:, 0] = seed_idx[:, 0].clamp(max=H-1)
        seed_idx[:, 1] = seed_idx[:, 1].clamp(max=W-1)
        seed_tangents = tangent[seed_idx[:, 0], seed_idx[:, 1]]  # (B, 2)
        # grad_norm points outward from dark lines - better than void_dir for shadow direction
        outward_dirs = grad_norm[seed_idx[:, 0], seed_idx[:, 1]]  # (B, 2)

        # === BATCHED DIFFERENTIABLE SOLVE ===
        # Use local geometry only - direction determined by perpendicular weighted toward void
        # Higher distances push fragments into negative space for better budding
        opt_theta, opt_dist, shadow_dirs = solve_placements_vectorized(
            fragments_padded, frag_masks, frag_axes,
            valid_seeds, seed_tangents, outward_dirs,
            tangent, contours, distance,
            n_iters=25, lr_theta=0.5, lr_dist=0.15,
            dist_init=shadow_dist * 0.7, dist_min=shadow_dist * 0.3, dist_max=shadow_dist * 1.2,
            temperature=8.0,  # strong preference for outward-facing perpendicular
        )
        stats['angles'] = opt_theta.tolist()

        # === APPLY PLACEMENTS (still looped for scatter, but solver is batched) ===
        for i in range(B):
            n = frag_masks[i].sum().item()
            if n == 0:
                continue

            frag = fragments_padded[i, :n]
            colors = frag_colors_padded[i, :n]
            seed = valid_seeds[i]
            theta = opt_theta[i]
            dist = opt_dist[i]
            sdir = shadow_dirs[i]

            # Rotate fragment
            c, s = theta.cos(), theta.sin()
            rotated = torch.stack([c * frag[:, 0] - s * frag[:, 1],
                                   s * frag[:, 0] + c * frag[:, 1]], dim=-1)

            def scatter_layer(offset_mult, layer_colors):
                dst = (seed + rotated + sdir * offset_mult).long()
                shape_t = torch.tensor([H, W], device=device)
                in_bounds = ((dst >= 0) & (dst < shape_t)).all(dim=-1)
                dst_c = dst.clamp(min=0, max=shape_t.max().item()-1)
                flat_dst = coords_to_flat(dst_c, (H, W))
                not_on_contour = ~contours.view(-1)[flat_dst.clamp(0, H*W-1)]
                valid = in_bounds & not_on_contour
                flat_valid = coords_to_flat(dst[valid], (H, W)).clamp(0, H*W-1)
                output.view(-1, 3)[flat_valid] = layer_colors[valid].clamp(0, 1)

            scatter_layer(dist + shadow_offset,
                         cyclic_color(colors, 0.3, 0.5) * torch.tensor([0.7, 1.0, 1.3], device=device))
            scatter_layer(dist,
                         cyclic_color(colors, 0.2, 0.4) * torch.tensor([0.6, 1.2, 1.15], device=device))

    return output, stats


if __name__ == "__main__":
    from pathlib import Path
    from PIL import Image
    import numpy as np
    from datetime import datetime

    inp, out = Path("demo_output/inputs"), Path("demo_output")
    images = {}
    for name in ["toof.png", "snek-heavy.png", "red-tonegraph.png"]:
        p = inp / name
        if p.exists():
            images[name.split('.')[0]] = torch.tensor(
                np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0,
                device=DEVICE
            )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n[v11] Collapsed spectral shader")

    # Self-attention
    if "toof" in images:
        out_img, stats = shader_v11(images["toof"])
        print(f"  Self: {stats}")
        Image.fromarray((out_img.cpu().numpy() * 255).clip(0,255).astype(np.uint8)).save(out / f"{ts}_v11_self.png")

    # Cross-attention
    for t, s in [("snek-heavy", "red-tonegraph"), ("toof", "snek-heavy")]:
        if t in images and s in images:
            out_img, stats = shader_v11(images[t], images[s])
            print(f"  {t} x {s}: {stats}")
            Image.fromarray((out_img.cpu().numpy() * 255).clip(0,255).astype(np.uint8)).save(out / f"{ts}_v11_{t}_x_{s}.png")

    # 4-deep regression test
    print(f"\n  [4-pass regression]")
    for name in ["toof", "snek-heavy"]:
        if name not in images:
            continue
        current = images[name].clone()
        pass_stats = []
        for p in range(4):
            current, stats = shader_v11(current)
            pass_stats.append(stats)
            Image.fromarray((current.cpu().numpy() * 255).clip(0,255).astype(np.uint8)).save(
                out / f"{ts}_v11_{name}_pass{p+1}.png"
            )
        print(f"  {name} 4-pass:")
        for i, s in enumerate(pass_stats):
            print(f"    p{i+1}: contours={s['contours']}, high={s['high']}, low={s['low']}, thick={s.get('thickened',0)}, comps={s.get('components',0)}")
