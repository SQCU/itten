"""Minimal spectral ops for spectral shader. No Graph class, no scipy. Pure PyTorch sparse."""

import torch
from typing import Tuple, List

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_weighted_image_laplacian(carrier: torch.Tensor, edge_threshold: float = 0.1) -> torch.Tensor:
    """Build 4-connected weighted Laplacian. Weight = exp(-|color_diff|/threshold)."""
    device, (H, W) = carrier.device, carrier.shape
    n = H * W

    # Horizontal edges
    y_h, x_h = torch.arange(H, device=device), torch.arange(W - 1, device=device)
    yy_h, xx_h = torch.meshgrid(y_h, x_h, indexing='ij')
    idx_L, idx_R = (yy_h * W + xx_h).flatten(), (yy_h * W + xx_h).flatten() + 1
    w_h = torch.exp(-torch.abs(carrier[:, 1:] - carrier[:, :-1]).flatten() / edge_threshold)

    # Vertical edges
    y_v, x_v = torch.arange(H - 1, device=device), torch.arange(W, device=device)
    yy_v, xx_v = torch.meshgrid(y_v, x_v, indexing='ij')
    idx_T, idx_B = (yy_v * W + xx_v).flatten(), (yy_v * W + xx_v).flatten() + W
    w_v = torch.exp(-torch.abs(carrier[1:, :] - carrier[:-1, :]).flatten() / edge_threshold)

    # Off-diagonal and diagonal
    rows = torch.cat([idx_L, idx_R, idx_T, idx_B])
    cols = torch.cat([idx_R, idx_L, idx_B, idx_T])
    vals = torch.cat([-w_h, -w_h, -w_v, -w_v])

    degrees = torch.zeros(n, device=device)
    degrees.scatter_add_(0, idx_L, w_h)
    degrees.scatter_add_(0, idx_R, w_h)
    degrees.scatter_add_(0, idx_T, w_v)
    degrees.scatter_add_(0, idx_B, w_v)

    diag_idx = torch.arange(n, device=device)
    rows, cols = torch.cat([rows, diag_idx]), torch.cat([cols, diag_idx])
    vals = torch.cat([vals, degrees])

    return torch.sparse_coo_tensor(torch.stack([rows.long(), cols.long()]), vals, (n, n)).coalesce()


def lanczos_fiedler_gpu(L: torch.Tensor, num_iterations: int = 30, tol: float = 1e-10) -> Tuple[torch.Tensor, float]:
    """Lanczos for Fiedler vector (2nd smallest eigenvector)."""
    n, device = L.shape[0], L.device
    if n < 3:
        return torch.zeros(n, device=device, dtype=torch.float32), 0.0

    k = min(num_iterations, n - 1)
    torch.manual_seed(42)
    v = torch.randn(n, device=device, dtype=torch.float32)
    v = v - v.mean()
    v_norm = torch.linalg.norm(v)
    if v_norm < tol:
        v = torch.ones(n, device=device, dtype=torch.float32)
        v[::2] = -1.0
        v = v - v.mean()
        v_norm = torch.linalg.norm(v)
    v = v / v_norm

    V = torch.zeros((n, k + 1), device=device, dtype=torch.float32)
    V[:, 0] = v
    alphas, betas = torch.zeros(k, device=device, dtype=torch.float32), torch.zeros(k, device=device, dtype=torch.float32)

    actual_k = k
    for i in range(k):
        w = torch.sparse.mm(L, V[:, i].unsqueeze(1)).squeeze(1)
        alpha = torch.dot(V[:, i], w)
        alphas[i] = alpha
        w = w - alpha * V[:, i]
        if i > 0:
            w = w - betas[i - 1] * V[:, i - 1]
            V_prev = V[:, :i + 1]
            coeffs = torch.mm(V_prev.T, w.unsqueeze(1)).squeeze(1)
            w = w - torch.mm(V_prev, coeffs.unsqueeze(1)).squeeze(1)
        w = w - w.mean()
        beta = torch.linalg.norm(w)
        if beta < tol:
            actual_k = i + 1
            break
        betas[i] = beta
        V[:, i + 1] = w / beta

    if actual_k < 2:
        return torch.zeros(n, device=device, dtype=torch.float32), 0.0

    T = torch.diag(alphas[:actual_k])
    if actual_k > 1:
        off = torch.arange(actual_k - 1, device=device)
        T[off, off + 1] = betas[:actual_k - 1]
        T[off + 1, off] = betas[:actual_k - 1]

    eigenvalues, eigenvectors = torch.linalg.eigh(T)
    mask = eigenvalues > 1e-6
    if not mask.any():
        return torch.zeros(n, device=device, dtype=torch.float32), 0.0

    idx = torch.where(mask)[0][0].item()
    y = eigenvectors[:, idx].unsqueeze(1)
    return torch.mm(V[:, :actual_k], y).squeeze(1), eigenvalues[idx].item()


def _build_multiscale_laplacian(tile: torch.Tensor, radii: List[int], radius_weights: List[float], edge_threshold: float) -> torch.Tensor:
    """Multi-radius Laplacian for dither patterns."""
    device = tile.device
    is_rgb = tile.dim() == 3 and tile.shape[-1] == 3
    H, W = (tile.shape[0], tile.shape[1]) if is_rgb else tile.shape
    flat = tile.reshape(-1, 3) if is_rgb else tile.flatten()
    n = H * W

    offset_dy, offset_dx, offset_w = [], [], []
    for radius, rw in zip(radii, radius_weights):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if (dx == 0 and dy == 0) or dx * dx + dy * dy > radius * radius:
                    continue
                offset_dy.append(dy)
                offset_dx.append(dx)
                offset_w.append(rw)

    if not offset_dy:
        diag = torch.arange(n, device=device)
        return torch.sparse_coo_tensor(torch.stack([diag, diag]), torch.ones(n, device=device), (n, n)).coalesce()

    offset_dy = torch.tensor(offset_dy, device=device, dtype=torch.long)
    offset_dx = torch.tensor(offset_dx, device=device, dtype=torch.long)
    offset_w = torch.tensor(offset_w, device=device, dtype=torch.float32)

    yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=torch.long), torch.arange(W, device=device, dtype=torch.long), indexing='ij')
    yy_flat, xx_flat = yy.flatten(), xx.flatten()
    dst_y = yy_flat.unsqueeze(1) + offset_dy.unsqueeze(0)
    dst_x = xx_flat.unsqueeze(1) + offset_dx.unsqueeze(0)
    valid = (dst_y >= 0) & (dst_y < H) & (dst_x >= 0) & (dst_x < W)

    src_idx = torch.arange(n, device=device).unsqueeze(1).expand(-1, len(offset_dy))
    dst_idx = dst_y * W + dst_x
    src_flat, dst_flat = src_idx[valid], dst_idx[valid]
    weights_flat = offset_w.unsqueeze(0).expand(n, -1)[valid]

    color_diff = torch.norm(flat[src_flat] - flat[dst_flat], dim=-1) if is_rgb else torch.abs(flat[src_flat] - flat[dst_flat])
    edge_w = weights_flat * torch.exp(-color_diff / edge_threshold)
    rows, cols, vals = src_flat, dst_flat, -edge_w

    degrees = torch.zeros(n, device=device)
    degrees.scatter_add_(0, rows, -vals)

    diag = torch.arange(n, device=device)
    rows, cols = torch.cat([rows, diag]), torch.cat([cols, diag])
    vals = torch.cat([vals, degrees])
    return torch.sparse_coo_tensor(torch.stack([rows.long(), cols.long()]), vals, (n, n)).coalesce()


def _lanczos_tile(L: torch.Tensor, num_evecs: int, num_iter: int) -> torch.Tensor:
    """Lanczos for multiple eigenvectors on a tile."""
    n, device = L.shape[0], L.device
    k = min(num_iter, n - 1)
    if n < 3 or k < 2:
        return torch.zeros((n, num_evecs), device=device, dtype=torch.float32)

    torch.manual_seed(42 + n % 10000)
    v = torch.randn(n, device=device, dtype=torch.float32)
    v = (v - v.mean()) / torch.linalg.norm(v - v.mean()).clamp(min=1e-10)

    V = torch.zeros((n, k + 1), device=device, dtype=torch.float32)
    V[:, 0] = v
    alphas, betas = torch.zeros(k, device=device, dtype=torch.float32), torch.zeros(k, device=device, dtype=torch.float32)

    actual_k = k
    for i in range(k):
        w = torch.sparse.mm(L, V[:, i].unsqueeze(1)).squeeze(1)
        alphas[i] = torch.dot(V[:, i], w)
        w = w - alphas[i] * V[:, i]
        if i > 0:
            w = w - betas[i - 1] * V[:, i - 1]
            V_prev = V[:, :i + 1]
            coeffs = torch.mm(V_prev.T, w.unsqueeze(1)).squeeze(1)
            w = w - torch.mm(V_prev, coeffs.unsqueeze(1)).squeeze(1)
        w = w - w.mean()
        beta = torch.linalg.norm(w)
        if beta < 1e-10:
            actual_k = i + 1
            break
        betas[i] = beta
        V[:, i + 1] = w / beta

    if actual_k < 2:
        return torch.zeros((n, num_evecs), device=device, dtype=torch.float32)

    T = torch.diag(alphas[:actual_k])
    if actual_k > 1:
        off = torch.arange(actual_k - 1, device=device)
        T[off, off + 1] = betas[:actual_k - 1]
        T[off + 1, off] = betas[:actual_k - 1]

    eigenvalues, eigenvectors = torch.linalg.eigh(T)
    mask = eigenvalues > 1e-6
    valid_idx = torch.where(mask)[0]
    if len(valid_idx) == 0:
        return torch.zeros((n, num_evecs), device=device, dtype=torch.float32)

    take = min(num_evecs, len(valid_idx))
    evecs = torch.mm(V[:, :actual_k], eigenvectors[:, valid_idx[:take]])
    result = torch.zeros((n, num_evecs), device=device, dtype=torch.float32)
    result[:, :take] = evecs
    return result


def _blend_weights_1d(length: int, overlap: int, device: torch.device) -> torch.Tensor:
    """1D blend weights with linear taper."""
    w = torch.ones(length, device=device, dtype=torch.float32)
    if overlap > 0 and length > 2 * overlap:
        taper = torch.arange(1, overlap + 1, device=device, dtype=torch.float32) / (overlap + 1)
        w[:overlap], w[-overlap:] = taper, taper.flip(0)
    return w


def compute_local_eigenvectors_tiled_dither(
    image: torch.Tensor, tile_size: int = 64, overlap: int = 16, num_eigenvectors: int = 4,
    radii: List[int] = None, radius_weights: List[float] = None,
    edge_threshold: float = 0.15, lanczos_iterations: int = 30
) -> torch.Tensor:
    """Compute eigenvectors in overlapping tiles with multi-radius connectivity for dither."""
    radii = radii or [1, 2, 3, 4, 5, 6]
    radius_weights = radius_weights or [1.0, 0.6, 0.4, 0.3, 0.2, 0.1]
    device = image.device
    is_rgb = image.dim() == 3 and image.shape[-1] == 3
    H, W = (image.shape[0], image.shape[1]) if is_rgb else image.shape

    if image.max() > 1.0:
        image = image.to(dtype=torch.float32) / 255.0
    else:
        image = image.to(dtype=torch.float32)

    result = torch.zeros((H, W, num_eigenvectors), device=device, dtype=torch.float32)
    weights = torch.zeros((H, W), device=device, dtype=torch.float32)

    step = tile_size - overlap
    y_starts = list(range(0, max(1, H - tile_size + 1), step))
    if not y_starts or y_starts[-1] + tile_size < H:
        y_starts.append(max(0, H - tile_size))
    x_starts = list(range(0, max(1, W - tile_size + 1), step))
    if not x_starts or x_starts[-1] + tile_size < W:
        x_starts.append(max(0, W - tile_size))
    y_starts, x_starts = sorted(set(y_starts)), sorted(set(x_starts))

    for y in y_starts:
        for x in x_starts:
            y_end, x_end = min(y + tile_size, H), min(x + tile_size, W)
            tile = image[y:y_end, x:x_end]
            th, tw = (tile.shape[0], tile.shape[1]) if is_rgb else tile.shape
            if th < 4 or tw < 4:
                continue

            L = _build_multiscale_laplacian(tile, radii, radius_weights, edge_threshold)
            evecs = _lanczos_tile(L, num_eigenvectors, lanczos_iterations).reshape(th, tw, num_eigenvectors)
            blend = torch.outer(_blend_weights_1d(th, overlap, device), _blend_weights_1d(tw, overlap, device))

            for k in range(num_eigenvectors):
                result[y:y_end, x:x_end, k] += evecs[:, :, k] * blend
            weights[y:y_end, x:x_end] += blend

    return result / weights.unsqueeze(-1).clamp(min=1e-10)
