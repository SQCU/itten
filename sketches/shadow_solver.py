"""
Differentiable shadow placement solver.
Replaces grid search with batched gradient descent on (θ, d) per fragment.

Key insight: anisotropic learning rates make rotation "cheap" and translation "expensive",
so the solver naturally prefers to rotate first, translate only when necessary.
"""
import torch
import torch.nn.functional as F
import math


def perpendicular_soft(tangent: torch.Tensor, void_grad: torch.Tensor, temperature: float = 5.0) -> torch.Tensor:
    """
    Compute shadow direction as perpendicular to tangent, weighted toward void.

    NO global bias - direction is purely determined by local geometry.

    tangent: (B, 2) or (2,) - tangent direction at seed points
    void_grad: (B, 2) or (2,) - gradient pointing toward void (outward from contours)

    Returns: (B, 2) or (2,) - shadow direction (perpendicular to tangent, biased toward void)
    """
    single = tangent.dim() == 1
    if single:
        tangent = tangent.unsqueeze(0)
        void_grad = void_grad.unsqueeze(0)

    # Two perpendicular options: rotate tangent by ±90°
    perp1 = torch.stack([-tangent[:, 1], tangent[:, 0]], dim=-1)  # rotate +90°
    perp2 = -perp1  # rotate -90°

    # Soft weighting: prefer the perpendicular that points toward void
    dot1 = (perp1 * void_grad).sum(dim=-1, keepdim=True)
    dot2 = (perp2 * void_grad).sum(dim=-1, keepdim=True)

    w1 = torch.exp(dot1 * temperature)
    w2 = torch.exp(dot2 * temperature)
    w_sum = w1 + w2 + 1e-8

    shadow_dir = (w1 * perp1 + w2 * perp2) / w_sum
    shadow_dir = shadow_dir / (shadow_dir.norm(dim=-1, keepdim=True) + 1e-8)

    return shadow_dir.squeeze(0) if single else shadow_dir


def batched_rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
    """(B,) angles -> (B, 2, 2) rotation matrices."""
    c, s = angles.cos(), angles.sin()
    R = torch.zeros(angles.shape[0], 2, 2, device=angles.device, dtype=angles.dtype)
    R[:, 0, 0], R[:, 0, 1] = c, -s
    R[:, 1, 0], R[:, 1, 1] = s, c
    return R


def gather_bilinear(field: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    Bilinear interpolation gather for differentiability.
    field: (H, W) or (H, W, C)
    coords: (*, 2) in [y, x] format, float
    Returns: (*, ) or (*, C)
    """
    H, W = field.shape[:2]
    has_channels = field.dim() == 3
    C = field.shape[2] if has_channels else 1
    if not has_channels:
        field = field.unsqueeze(-1)

    orig_shape = coords.shape[:-1]
    coords_flat = coords.reshape(-1, 2)  # (N, 2)

    y, x = coords_flat[:, 0], coords_flat[:, 1]

    # Clamp to valid range
    y = y.clamp(0, H - 1.001)
    x = x.clamp(0, W - 1.001)

    y0, x0 = y.floor().long(), x.floor().long()
    y1, x1 = (y0 + 1).clamp(max=H-1), (x0 + 1).clamp(max=W-1)

    wy1, wx1 = y - y0.float(), x - x0.float()
    wy0, wx0 = 1 - wy1, 1 - wx1

    # Gather corners
    v00 = field[y0, x0]  # (N, C)
    v01 = field[y0, x1]
    v10 = field[y1, x0]
    v11 = field[y1, x1]

    # Bilinear weights
    w00 = (wy0 * wx0).unsqueeze(-1)
    w01 = (wy0 * wx1).unsqueeze(-1)
    w10 = (wy1 * wx0).unsqueeze(-1)
    w11 = (wy1 * wx1).unsqueeze(-1)

    result = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11
    result = result.reshape(*orig_shape, C)

    return result.squeeze(-1) if not has_channels else result


def compute_placement_energy(
    theta: torch.Tensor,           # (B,) rotation angles
    dist: torch.Tensor,            # (B,) translation distances
    fragments: list,               # list of (N_i, 2) relative coords per fragment
    frag_axes: torch.Tensor,       # (B, 2) principal axes
    seeds: torch.Tensor,           # (B, 2) seed positions
    seed_tangents: torch.Tensor,   # (B, 2) tangent at seeds
    shadow_dirs: torch.Tensor,     # (B, 2) shadow projection direction
    tangent_field: torch.Tensor,   # (H, W, 2) tangent field
    geom_mask: torch.Tensor,       # (H, W) bool
    dist_field: torch.Tensor,      # (H, W) normalized distance
    # Energy weights
    w_parallel: float = 6.0,       # penalize parallel to seed tangent (PRIMARY)
    w_src_rot: float = 1.0,        # penalize not rotating from original axis
    w_local: float = 1.0,          # penalize parallel to local geometry at landing
    w_clip: float = 3.0,           # penalize intersection with geometry
    w_void: float = 1.5,           # reward distance from geometry
    w_ortho_discount: float = 0.4, # discount clipping when crossing orthogonally
) -> torch.Tensor:
    """
    Compute differentiable energy for batched fragment placements.
    Returns (B,) energy values.
    """
    device = theta.device
    B = theta.shape[0]
    H, W = geom_mask.shape

    # Rotation: explicit 2D
    c, s = theta.cos(), theta.sin()  # (B,)
    R = torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s, c], dim=-1)
    ], dim=1)  # (B, 2, 2)

    # Rotated axes
    ax_b, ay_b = frag_axes[:, 0], frag_axes[:, 1]
    rot_axes = torch.stack([c * ax_b - s * ay_b, s * ax_b + c * ay_b], dim=-1)  # (B, 2)

    # E_parallel: penalize when rotated axis is parallel to seed tangent
    # Use smooth approximation: 1 - |sin(angle between)|^2 ≈ |cos|^2 = dot^2
    dot_tangent = (rot_axes * seed_tangents).sum(dim=-1)  # (B,)
    E_parallel = dot_tangent ** 2  # high when parallel

    # E_src_rot: small penalty for not rotating away from original axis
    dot_src = (rot_axes * frag_axes).sum(dim=-1)  # (B,)
    E_src_rot = dot_src.abs()  # encourage rotation

    # Now compute energies that depend on actual placement positions
    # We'll sample a few representative points per fragment for efficiency
    E_local = torch.zeros(B, device=device)
    E_clip = torch.zeros(B, device=device)
    E_void = torch.zeros(B, device=device)

    for i in range(B):
        frag = fragments[i]  # (N_i, 2)
        N_i = frag.shape[0]
        if N_i == 0:
            continue

        # Subsample for efficiency (max 32 points)
        if N_i > 32:
            idx = torch.linspace(0, N_i - 1, 32, device=device).long()
            frag = frag[idx]
            N_i = 32

        # Rotate fragment
        rotated = frag @ R[i].T  # (N_i, 2)

        # Translate: seed + distance * shadow_dir
        placed = seeds[i] + rotated + dist[i] * shadow_dirs[i]  # (N_i, 2)

        # Gather local tangent at placed positions
        local_tan = gather_bilinear(tangent_field, placed)  # (N_i, 2)

        # E_local: penalize parallel to local tangent
        dot_local = (rot_axes[i:i+1] * local_tan).sum(dim=-1)  # (N_i,)
        E_local[i] = (dot_local ** 2).mean()

        # Gather geometry mask (soft via bilinear on float version)
        on_geom = gather_bilinear(geom_mask.float(), placed)  # (N_i,)

        # Orthogonality discount: less penalty if crossing at ~90°
        ortho = 1.0 - dot_local.abs()  # 1 when perpendicular, 0 when parallel
        clip_penalty = on_geom * (1.0 - w_ortho_discount * ortho)
        E_clip[i] = clip_penalty.mean()

        # E_void: reward being far from geometry (negative = good)
        void_dist = gather_bilinear(dist_field, placed)  # (N_i,)
        E_void[i] = -void_dist.mean()  # negative because high dist_field = far from geometry

    # Total energy
    E_total = (w_parallel * E_parallel +
               w_src_rot * E_src_rot +
               w_local * E_local +
               w_clip * E_clip +
               w_void * E_void)

    return E_total


def solve_placements_batched(
    fragments: list,               # list of (N_i, 2) relative coords
    frag_axes: torch.Tensor,       # (B, 2) principal axes
    seeds: torch.Tensor,           # (B, 2) seed positions
    seed_tangents: torch.Tensor,   # (B, 2) tangent at seeds
    void_grads: torch.Tensor,      # (B, 2) void direction at seeds
    tangent_field: torch.Tensor,   # (H, W, 2)
    geom_mask: torch.Tensor,       # (H, W)
    dist_field: torch.Tensor,      # (H, W)
    # Solver params
    n_iters: int = 15,
    lr_theta: float = 0.4,         # learning rate for rotation (FAST)
    lr_dist: float = 0.08,         # learning rate for translation (SLOW)
    dist_init: float = 12.0,
    dist_min: float = 3.0,
    dist_max: float = 35.0,
    temperature: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched differentiable solver for shadow placement.

    Returns:
        theta: (B,) optimal rotation angles
        dist: (B,) optimal translation distances
        shadow_dirs: (B, 2) shadow projection directions
    """
    device = seeds.device
    B = seeds.shape[0]

    # Compute shadow directions (perpendicular to tangent, weighted toward void)
    shadow_dirs = perpendicular_soft(seed_tangents, void_grads, temperature)  # (B, 2)

    # Initialize: start perpendicular to seed tangent
    # Find angle that makes frag_axis perpendicular to seed_tangent
    # If frag_axis · R(θ) · seed_tangent = 0, then θ = atan2(ax·ty - ay·tx, ax·tx + ay·ty) + π/2
    ax, ay = frag_axes[:, 0], frag_axes[:, 1]
    tx, ty = seed_tangents[:, 0], seed_tangents[:, 1]
    theta_perp = torch.atan2(ax * ty - ay * tx, ax * tx + ay * ty) + math.pi / 2

    # Parameters to optimize
    theta = theta_perp.clone().requires_grad_(True)
    dist = torch.full((B,), dist_init, device=device, requires_grad=True)

    # Gradient descent with anisotropic learning rates
    for iteration in range(n_iters):
        # Zero grads
        if theta.grad is not None:
            theta.grad.zero_()
        if dist.grad is not None:
            dist.grad.zero_()

        # Compute energy
        E = compute_placement_energy(
            theta, dist, fragments, frag_axes, seeds, seed_tangents, shadow_dirs,
            tangent_field, geom_mask, dist_field
        )

        # Backward
        E_total = E.sum()
        E_total.backward()

        # Update with anisotropic learning rates
        with torch.no_grad():
            theta -= lr_theta * theta.grad
            dist -= lr_dist * dist.grad
            dist.clamp_(dist_min, dist_max)

        # Re-enable gradients
        theta.requires_grad_(True)
        dist.requires_grad_(True)

    return theta.detach(), dist.detach(), shadow_dirs


def solve_placements_vectorized(
    fragments_padded: torch.Tensor,  # (B, max_N, 2) padded fragment coords
    frag_masks: torch.Tensor,        # (B, max_N) bool - valid points
    frag_axes: torch.Tensor,         # (B, 2) principal axes
    seeds: torch.Tensor,             # (B, 2) seed positions
    seed_tangents: torch.Tensor,     # (B, 2) tangent at seeds
    void_grads: torch.Tensor,        # (B, 2) void direction at seeds
    tangent_field: torch.Tensor,     # (H, W, 2)
    geom_mask: torch.Tensor,         # (H, W)
    dist_field: torch.Tensor,        # (H, W)
    # Solver params
    n_iters: int = 15,
    lr_theta: float = 0.4,
    lr_dist: float = 0.08,
    dist_init: float = 12.0,
    dist_min: float = 3.0,
    dist_max: float = 35.0,
    dist_target: float = None,  # target for E_min_dist penalty, defaults to dist_init * 1.5
    temperature: float = 5.0,
    # Energy weights - PARALLELISM IS WORSE THAN CLIPPING
    # Parallel non-clipping is a fixed point (sedimentary roughage)
    # Clipping at least allows finding anti-parallel escape directions
    w_parallel: float = 3.0,       # perpendicular to seed tangent
    w_src_rot: float = 0.3,        # mild rotation encouragement
    w_local: float = 12.0,         # DOMINANT: avoid parallel to ANY nearby geometry
    w_clip: float = 4.0,           # moderate: some clipping OK if it breaks parallelism
    w_void: float = 6.0,           # seek empty space
    w_ortho_discount: float = 0.8, # VERY generous - orthogonal crossing is fine
    w_min_dist: float = 4.0,       # push into void
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fully vectorized solver - no Python loops over fragments.
    All B fragments processed in parallel.

    Returns:
        theta: (B,) optimal rotation angles
        dist: (B,) optimal translation distances
        shadow_dirs: (B, 2) shadow projection directions
    """
    device = seeds.device
    B, max_N, _ = fragments_padded.shape
    H, W = geom_mask.shape

    # Compute shadow directions
    shadow_dirs = perpendicular_soft(seed_tangents, void_grads, temperature)  # (B, 2)

    # Initialize theta perpendicular to seed tangent
    ax, ay = frag_axes[:, 0], frag_axes[:, 1]
    tx, ty = seed_tangents[:, 0], seed_tangents[:, 1]
    theta_perp = torch.atan2(ax * ty - ay * tx, ax * tx + ay * ty) + math.pi / 2

    theta = theta_perp.clone()
    dist = torch.full((B,), dist_init, device=device)

    # Count valid points per fragment for proper averaging
    valid_counts = frag_masks.sum(dim=1).clamp(min=1).float()  # (B,)

    for iteration in range(n_iters):
        # Rotation: explicit 2D rotation (avoids bmm for stability)
        c, s = theta.cos(), theta.sin()  # (B,)

        # Rotated axes: R @ frag_axes where R = [[c,-s],[s,c]]
        ax, ay = frag_axes[:, 0], frag_axes[:, 1]
        rot_axes = torch.stack([c * ax - s * ay, s * ax + c * ay], dim=-1)  # (B, 2)

        # === ROTATION ENERGIES (don't depend on placement) ===
        # E_parallel: penalize parallel to seed tangent
        dot_tangent = (rot_axes * seed_tangents).sum(dim=-1)  # (B,)
        E_parallel = dot_tangent ** 2

        # E_src_rot: encourage rotation from original
        dot_src = (rot_axes * frag_axes).sum(dim=-1)
        E_src_rot = dot_src.abs()

        # === PLACEMENT-DEPENDENT ENERGIES ===
        # Rotate all fragment points: (B, max_N, 2)
        fx, fy = fragments_padded[..., 0], fragments_padded[..., 1]  # (B, max_N)
        rotated = torch.stack([c.unsqueeze(1) * fx - s.unsqueeze(1) * fy,
                               s.unsqueeze(1) * fx + c.unsqueeze(1) * fy], dim=-1)

        # Translate: seed + rotated + dist * shadow_dir
        # seeds: (B, 2) -> (B, 1, 2), dist: (B,) -> (B, 1, 1), shadow_dirs: (B, 2) -> (B, 1, 2)
        placed = seeds.unsqueeze(1) + rotated + dist.unsqueeze(1).unsqueeze(2) * shadow_dirs.unsqueeze(1)
        # placed: (B, max_N, 2)

        # Clamp to valid range for gathering
        placed_clamped = placed.clone()
        placed_clamped[..., 0] = placed_clamped[..., 0].clamp(0, H - 1.001)
        placed_clamped[..., 1] = placed_clamped[..., 1].clamp(0, W - 1.001)

        # Bilinear gather - flatten for efficiency
        placed_flat = placed_clamped.reshape(-1, 2)  # (B*max_N, 2)

        # Gather tangent field
        local_tan_flat = gather_bilinear(tangent_field, placed_flat)  # (B*max_N, 2)
        local_tan = local_tan_flat.reshape(B, max_N, 2)

        # Gather geometry mask
        on_geom_flat = gather_bilinear(geom_mask.float(), placed_flat)
        on_geom = on_geom_flat.reshape(B, max_N)

        # Gather distance field
        void_dist_flat = gather_bilinear(dist_field, placed_flat)
        void_dist = void_dist_flat.reshape(B, max_N)

        # E_local: parallel to local tangent
        dot_local = (rot_axes.unsqueeze(1) * local_tan).sum(dim=-1)  # (B, max_N)
        E_local_per_point = dot_local ** 2
        E_local = (E_local_per_point * frag_masks.float()).sum(dim=1) / valid_counts

        # E_clip: intersection with geometry, discounted for orthogonal crossing
        ortho = 1.0 - dot_local.abs()
        clip_per_point = on_geom * (1.0 - w_ortho_discount * ortho)
        E_clip = (clip_per_point * frag_masks.float()).sum(dim=1) / valid_counts

        # E_void: reward emptiness
        E_void = -(void_dist * frag_masks.float()).sum(dim=1) / valid_counts

        # E_min_dist: penalize staying too close to seed (push shadows outward)
        # Uses smooth exponential decay - strong penalty near seed, gentle far away
        # target_dist scales with solver's dist_init (no hardcoded pixel values)
        _target_dist = dist_target if dist_target is not None else dist_init * 1.5
        E_min_dist = torch.exp(-dist / _target_dist)  # high when dist is small

        # Total energy
        E = (w_parallel * E_parallel +
             w_src_rot * E_src_rot +
             w_local * E_local +
             w_clip * E_clip +
             w_void * E_void +
             w_min_dist * E_min_dist)

        # === GRADIENT COMPUTATION ===
        # Analytical gradients for key terms (avoiding autograd overhead)

        # dE_parallel/dtheta: d/dtheta (dot(R*ax, tx)^2)
        # Let u = R*ax, then dot(u, tx)^2 = (ux*tx + uy*ty)^2
        # R*ax = [c*ax - s*ay, s*ax + c*ay]
        # ux = c*ax - s*ay, uy = s*ax + c*ay
        # d(ux)/dtheta = -s*ax - c*ay = -uy, d(uy)/dtheta = c*ax - s*ay = ux
        # d(dot)/dtheta = -uy*tx + ux*ty
        # d(dot^2)/dtheta = 2*dot*(-uy*tx + ux*ty)
        ux, uy = rot_axes[:, 0], rot_axes[:, 1]
        grad_parallel = 2 * dot_tangent * (-uy * tx + ux * ty)

        # dE_src_rot/dtheta: similar pattern
        grad_src = torch.sign(dot_src) * (-uy * ax + ux * ay)

        # For placement-dependent terms, use finite difference (simpler, still fast)
        eps = 0.01
        theta_plus = theta + eps
        c_p, s_p = theta_plus.cos(), theta_plus.sin()
        rot_axes_p = torch.stack([c_p * ax - s_p * ay, s_p * ax + c_p * ay], dim=-1)

        rotated_p = torch.stack([c_p.unsqueeze(1) * fx - s_p.unsqueeze(1) * fy,
                                 s_p.unsqueeze(1) * fx + c_p.unsqueeze(1) * fy], dim=-1)
        placed_p = seeds.unsqueeze(1) + rotated_p + dist.unsqueeze(1).unsqueeze(2) * shadow_dirs.unsqueeze(1)
        placed_p_clamped = placed_p.clone()
        placed_p_clamped[..., 0] = placed_p_clamped[..., 0].clamp(0, H - 1.001)
        placed_p_clamped[..., 1] = placed_p_clamped[..., 1].clamp(0, W - 1.001)

        local_tan_p = gather_bilinear(tangent_field, placed_p_clamped.reshape(-1, 2)).reshape(B, max_N, 2)
        dot_local_p = (rot_axes_p.unsqueeze(1) * local_tan_p).sum(dim=-1)
        E_local_p = ((dot_local_p ** 2) * frag_masks.float()).sum(dim=1) / valid_counts

        on_geom_p = gather_bilinear(geom_mask.float(), placed_p_clamped.reshape(-1, 2)).reshape(B, max_N)
        ortho_p = 1.0 - dot_local_p.abs()
        E_clip_p = ((on_geom_p * (1.0 - w_ortho_discount * ortho_p)) * frag_masks.float()).sum(dim=1) / valid_counts

        void_dist_p = gather_bilinear(dist_field, placed_p_clamped.reshape(-1, 2)).reshape(B, max_N)
        E_void_p = -(void_dist_p * frag_masks.float()).sum(dim=1) / valid_counts

        grad_local_fd = (E_local_p - E_local) / eps
        grad_clip_fd = (E_clip_p - E_clip) / eps
        grad_void_fd = (E_void_p - E_void) / eps

        grad_theta = (w_parallel * grad_parallel +
                      w_src_rot * grad_src +
                      w_local * grad_local_fd +
                      w_clip * grad_clip_fd +
                      w_void * grad_void_fd)

        # Gradient for dist (finite difference on translation)
        dist_plus = dist + eps
        placed_d = seeds.unsqueeze(1) + rotated + dist_plus.unsqueeze(1).unsqueeze(2) * shadow_dirs.unsqueeze(1)
        placed_d_clamped = placed_d.clone()
        placed_d_clamped[..., 0] = placed_d_clamped[..., 0].clamp(0, H - 1.001)
        placed_d_clamped[..., 1] = placed_d_clamped[..., 1].clamp(0, W - 1.001)

        on_geom_d = gather_bilinear(geom_mask.float(), placed_d_clamped.reshape(-1, 2)).reshape(B, max_N)
        void_dist_d = gather_bilinear(dist_field, placed_d_clamped.reshape(-1, 2)).reshape(B, max_N)
        local_tan_d = gather_bilinear(tangent_field, placed_d_clamped.reshape(-1, 2)).reshape(B, max_N, 2)
        dot_local_d = (rot_axes.unsqueeze(1) * local_tan_d).sum(dim=-1)
        ortho_d = 1.0 - dot_local_d.abs()

        E_local_d = ((dot_local_d ** 2) * frag_masks.float()).sum(dim=1) / valid_counts
        E_clip_d = ((on_geom_d * (1.0 - w_ortho_discount * ortho_d)) * frag_masks.float()).sum(dim=1) / valid_counts
        E_void_d = -(void_dist_d * frag_masks.float()).sum(dim=1) / valid_counts

        # Analytical gradient for E_min_dist: d/ddist exp(-dist/target) = -exp(-dist/target)/target
        grad_min_dist = -E_min_dist / _target_dist

        grad_dist = (w_local * (E_local_d - E_local) +
                     w_clip * (E_clip_d - E_clip) +
                     w_void * (E_void_d - E_void)) / eps + w_min_dist * grad_min_dist

        # Update with anisotropic learning rates
        theta = theta - lr_theta * grad_theta
        dist = dist - lr_dist * grad_dist
        dist = dist.clamp(dist_min, dist_max)

    return theta, dist, shadow_dirs


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    B = 10
    max_N = 20
    H, W = 64, 64

    # Random test data
    fragments_padded = torch.randn(B, max_N, 2, device=device) * 5
    frag_masks = torch.rand(B, max_N, device=device) > 0.3
    frag_axes = F.normalize(torch.randn(B, 2, device=device), dim=-1)
    seeds = torch.rand(B, 2, device=device) * torch.tensor([H-1, W-1], device=device)
    seed_tangents = F.normalize(torch.randn(B, 2, device=device), dim=-1)
    void_grads = F.normalize(torch.randn(B, 2, device=device), dim=-1)
    tangent_field = F.normalize(torch.randn(H, W, 2, device=device), dim=-1)
    geom_mask = torch.rand(H, W, device=device) > 0.7
    dist_field = torch.rand(H, W, device=device)

    import time

    # Warmup
    for _ in range(3):
        theta, dist, shadow_dirs = solve_placements_vectorized(
            fragments_padded, frag_masks, frag_axes, seeds, seed_tangents, void_grads,
            tangent_field, geom_mask, dist_field, n_iters=15
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    n_runs = 50
    for _ in range(n_runs):
        theta, dist, shadow_dirs = solve_placements_vectorized(
            fragments_padded, frag_masks, frag_axes, seeds, seed_tangents, void_grads,
            tangent_field, geom_mask, dist_field, n_iters=15
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    print(f"Time per solve: {(t1-t0)/n_runs*1000:.2f}ms for B={B} fragments")
    print(f"Output shapes: theta={theta.shape}, dist={dist.shape}, shadow_dirs={shadow_dirs.shape}")
    print(f"Theta range: [{theta.min():.2f}, {theta.max():.2f}]")
    print(f"Dist range: [{dist.min():.2f}, {dist.max():.2f}]")
