"""
Coarsened spectral computation via graph sparsification.

Key insight: Real image graphs aren't random graphs. They have structure:
- Large homogeneous regions (collapse to single node)
- Coherent edges (preserve as boundaries between supernodes)
- Hierarchical organization

Pipeline:
1. Segment image into superpixels/regions (O(n))
2. Build coarse graph on regions (~500 nodes)
3. Compute Fiedler on coarse graph (~1ms)
4. Map Fiedler values back to pixels (O(n))

Total: O(n) + O(coarse) ≈ O(n) with small constant
Target: <5ms for 256x256
"""

import torch
import torch.nn.functional as F
from typing import Tuple
import time


def slic_superpixels(
    image_rgb: torch.Tensor,
    n_segments: int = 500,
    compactness: float = 10.0,
    n_iterations: int = 5
) -> torch.Tensor:
    """
    Simple Linear Iterative Clustering (SLIC) superpixels.

    Pure PyTorch implementation. Returns (H, W) tensor of segment labels.

    Complexity: O(n_iterations × n_pixels × n_segments_per_pixel)
    With grid initialization, each pixel only considers ~9 nearby centers.
    Effective: O(n_iterations × n)
    """
    H, W = image_rgb.shape[:2]
    device = image_rgb.device
    dtype = image_rgb.dtype

    # Grid spacing
    S = int((H * W / n_segments) ** 0.5)
    n_h = max(1, H // S)
    n_w = max(1, W // S)
    actual_segments = n_h * n_w

    # Initialize cluster centers on grid
    cy = torch.linspace(S // 2, H - S // 2, n_h, device=device)
    cx = torch.linspace(S // 2, W - S // 2, n_w, device=device)
    centers_y, centers_x = torch.meshgrid(cy, cx, indexing='ij')
    centers_y = centers_y.flatten().long().clamp(0, H - 1)
    centers_x = centers_x.flatten().long().clamp(0, W - 1)

    # Center colors
    centers_color = image_rgb[centers_y, centers_x]  # (K, 3)
    K = len(centers_y)

    # Pixel coordinates
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    # Labels
    labels = torch.zeros(H, W, device=device, dtype=torch.long)

    for _ in range(n_iterations):
        # Assign pixels to nearest center (within 2S window)
        # For efficiency, compute distances to all centers and take argmin
        # In production, use spatial hashing for O(1) per pixel

        # Color distance: ||pixel_color - center_color||²
        pixel_colors = image_rgb.reshape(-1, 3)  # (n, 3)
        color_dist = torch.cdist(pixel_colors, centers_color, p=2)  # (n, K)

        # Spatial distance: ||pixel_pos - center_pos||² / S²
        pixel_y = yy.flatten().float()
        pixel_x = xx.flatten().float()
        center_y = centers_y.float()
        center_x = centers_x.float()

        dy = pixel_y.unsqueeze(1) - center_y.unsqueeze(0)  # (n, K)
        dx = pixel_x.unsqueeze(1) - center_x.unsqueeze(0)
        spatial_dist = torch.sqrt(dy**2 + dx**2) / S

        # Combined distance
        dist = color_dist + compactness * spatial_dist

        # Assign to nearest
        labels = dist.argmin(dim=1).reshape(H, W)

        # Update centers
        for k in range(K):
            mask = (labels == k)
            if mask.sum() > 0:
                centers_y[k] = yy[mask].float().mean().long().clamp(0, H - 1)
                centers_x[k] = xx[mask].float().mean().long().clamp(0, W - 1)
                centers_color[k] = image_rgb[mask].mean(dim=0)

    return labels


def fast_superpixels(
    image_rgb: torch.Tensor,
    n_segments: int = 500
) -> torch.Tensor:
    """
    Fast approximate superpixels via grid + local refinement.

    Much faster than SLIC, good enough for coarsening.
    """
    H, W = image_rgb.shape[:2]
    device = image_rgb.device

    # Grid spacing
    S = max(4, int((H * W / n_segments) ** 0.5))

    # Downsample image
    scale = S
    small_h, small_w = H // scale, W // scale

    if small_h < 2 or small_w < 2:
        # Image too small, just use uniform grid
        labels = torch.zeros(H, W, device=device, dtype=torch.long)
        idx = 0
        for i in range(0, H, S):
            for j in range(0, W, S):
                labels[i:i+S, j:j+S] = idx
                idx += 1
        return labels

    # Downsample
    img_4d = image_rgb.permute(2, 0, 1).unsqueeze(0)
    small = F.interpolate(img_4d, size=(small_h, small_w), mode='bilinear', align_corners=False)
    small = small.squeeze(0).permute(1, 2, 0)  # (small_h, small_w, 3)

    # Each small pixel is a superpixel
    # Label by position
    labels_small = torch.arange(small_h * small_w, device=device).reshape(small_h, small_w)

    # Upsample labels (nearest neighbor to preserve boundaries)
    labels = F.interpolate(
        labels_small.float().unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode='nearest'
    ).squeeze().long()

    return labels


def build_coarse_graph(
    image_rgb: torch.Tensor,
    labels: torch.Tensor,
    edge_threshold: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build sparse Laplacian on superpixel graph.

    Nodes = superpixels
    Edges = between adjacent superpixels, weighted by boundary strength

    Returns: (L_coarse, mean_colors, n_nodes)
    """
    H, W = labels.shape
    device = labels.device

    # Number of unique labels
    n_nodes = labels.max().item() + 1

    # Mean color per superpixel
    flat_labels = labels.flatten()
    flat_colors = image_rgb.reshape(-1, 3)

    color_sum = torch.zeros(n_nodes, 3, device=device)
    counts = torch.zeros(n_nodes, device=device)

    color_sum.scatter_add_(0, flat_labels.unsqueeze(1).expand(-1, 3), flat_colors)
    counts.scatter_add_(0, flat_labels, torch.ones_like(flat_labels, dtype=torch.float))

    mean_colors = color_sum / counts.unsqueeze(1).clamp(min=1)

    # Find adjacent superpixel pairs
    # Horizontal adjacency
    h_left = labels[:, :-1].flatten()
    h_right = labels[:, 1:].flatten()
    h_diff = h_left != h_right

    # Vertical adjacency
    v_top = labels[:-1, :].flatten()
    v_bottom = labels[1:, :].flatten()
    v_diff = v_top != v_bottom

    # Collect edges
    edges_i = torch.cat([h_left[h_diff], v_top[v_diff]])
    edges_j = torch.cat([h_right[h_diff], v_bottom[v_diff]])

    # Make symmetric and unique
    all_i = torch.cat([edges_i, edges_j])
    all_j = torch.cat([edges_j, edges_i])

    # Edge weights from color difference
    color_i = mean_colors[all_i]
    color_j = mean_colors[all_j]
    color_diff = torch.norm(color_i - color_j, dim=1)
    weights = torch.exp(-color_diff / edge_threshold)

    # Build sparse Laplacian
    # Off-diagonal: -w_ij
    rows = all_i
    cols = all_j
    vals = -weights

    # Diagonal: sum of weights
    degrees = torch.zeros(n_nodes, device=device)
    degrees.scatter_add_(0, all_i, weights)

    diag = torch.arange(n_nodes, device=device)
    rows = torch.cat([rows, diag])
    cols = torch.cat([cols, diag])
    vals = torch.cat([vals, degrees])

    L = torch.sparse_coo_tensor(
        torch.stack([rows.long(), cols.long()]),
        vals,
        (n_nodes, n_nodes)
    ).coalesce()

    return L, mean_colors, n_nodes


def lanczos_small(L: torch.Tensor, n_iter: int = 30) -> torch.Tensor:
    """
    Lanczos for small graphs. Can use dense methods if n < 1000.
    """
    n = L.shape[0]
    device = L.device

    if n < 3:
        return torch.zeros(n, device=device)

    if n < 1000:
        # Dense eigendecomposition is faster for small graphs
        # Use CPU for stability with small matrices
        L_dense = L.to_dense().cpu()
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L_dense)
            # Return second eigenvector (Fiedler)
            return eigenvectors[:, 1].to(device)
        except RuntimeError:
            # Fall back to Lanczos
            pass

    # Lanczos for larger graphs
    torch.manual_seed(42)
    v = torch.randn(n, device=device)
    v = v - v.mean()
    v = v / torch.linalg.norm(v)

    V = torch.zeros(n, n_iter + 1, device=device)
    V[:, 0] = v
    alphas = torch.zeros(n_iter, device=device)
    betas = torch.zeros(n_iter, device=device)

    for i in range(n_iter):
        w = torch.sparse.mm(L, V[:, i].unsqueeze(1)).squeeze()
        alphas[i] = torch.dot(V[:, i], w)
        w = w - alphas[i] * V[:, i]
        if i > 0:
            w = w - betas[i-1] * V[:, i-1]
        w = w - w.mean()
        beta = torch.linalg.norm(w)
        if beta < 1e-10:
            break
        betas[i] = beta
        V[:, i+1] = w / beta

    # Build tridiagonal matrix
    T = torch.diag(alphas)
    for i in range(n_iter - 1):
        T[i, i+1] = betas[i]
        T[i+1, i] = betas[i]

    eigenvalues, eigenvectors = torch.linalg.eigh(T)

    # Find smallest positive eigenvalue
    mask = eigenvalues > 1e-6
    if not mask.any():
        return torch.zeros(n, device=device)

    idx = torch.where(mask)[0][0]
    y = eigenvectors[:, idx]

    return torch.mm(V[:, :n_iter], y.unsqueeze(1)).squeeze()


def fiedler_coarsened(
    image_rgb: torch.Tensor,
    n_segments: int = 500,
    edge_threshold: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute Fiedler vector via graph coarsening.

    1. Segment into superpixels
    2. Build coarse graph
    3. Compute Fiedler on coarse graph
    4. Map back to pixels

    Returns: (fiedler, labels, compute_time_ms)
    """
    H, W = image_rgb.shape[:2]
    device = image_rgb.device

    start = time.perf_counter()

    # Step 1: Superpixels
    labels = fast_superpixels(image_rgb, n_segments)

    # Step 2: Coarse graph
    L_coarse, mean_colors, n_nodes = build_coarse_graph(image_rgb, labels, edge_threshold)

    # Step 3: Fiedler on coarse graph
    fiedler_coarse = lanczos_small(L_coarse, n_iter=min(30, n_nodes - 2))

    # Step 4: Map back to pixels
    fiedler = fiedler_coarse[labels]

    elapsed = (time.perf_counter() - start) * 1000

    return fiedler, labels, elapsed


def fiedler_hierarchical(
    image_rgb: torch.Tensor,
    levels: int = 3,
    base_segments: int = 100
) -> Tuple[torch.Tensor, float]:
    """
    Multi-level coarsening for better accuracy.

    Coarsen at multiple scales, blend results.
    """
    H, W = image_rgb.shape[:2]
    device = image_rgb.device

    start = time.perf_counter()

    fiedlers = []
    weights = []

    for level in range(levels):
        n_seg = base_segments * (2 ** level)
        fiedler, _, _ = fiedler_coarsened(image_rgb, n_segments=n_seg)
        fiedlers.append(fiedler)
        weights.append(2 ** level)  # More weight to finer levels

    # Weighted blend
    total_weight = sum(weights)
    result = sum(f * w for f, w in zip(fiedlers, weights)) / total_weight

    elapsed = (time.perf_counter() - start) * 1000

    return result, elapsed


# ============================================================
# Benchmark
# ============================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Target: <5ms for 256x256\n")

    for size in [128, 256, 512]:
        H, W = size, size
        print(f"=== {H}x{W} ({H*W} pixels) ===")

        # Test image with structure
        torch.manual_seed(42)
        y, x = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )
        image = torch.stack([x, y, 0.5 * torch.ones_like(x)], dim=-1)
        c1 = ((x - 0.25)**2 + (y - 0.25)**2 < 0.02).float()
        c2 = ((x - 0.75)**2 + (y - 0.75)**2 < 0.03).float()
        line = (torch.abs(y - 0.6) < 0.015).float() * (x > 0.2).float() * (x < 0.8).float()
        mask = (c1 + c2 + line).clamp(0, 1)
        image = image * (1 - mask.unsqueeze(-1)) + mask.unsqueeze(-1) * 0.9

        # Warmup
        for _ in range(3):
            _ = fiedler_coarsened(image, n_segments=500)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark coarsened Fiedler
        times = []
        for _ in range(10):
            fiedler_c, labels, elapsed = fiedler_coarsened(image, n_segments=500)
            times.append(elapsed)

        n_superpixels = labels.max().item() + 1
        mean_time = sum(times) / len(times)
        print(f"  coarsened ({n_superpixels} superpixels): {mean_time:.2f}ms ({1000/mean_time:.0f} Hz)")
        print(f"    fiedler range: [{fiedler_c.min():.3f}, {fiedler_c.max():.3f}]")

        # Compare with full Lanczos
        try:
            from spectral_ops_fast_cuter import build_weighted_image_laplacian, lanczos_fiedler_gpu

            gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
            L_full = build_weighted_image_laplacian(gray)

            # Warmup
            for _ in range(3):
                _ = lanczos_fiedler_gpu(L_full, num_iterations=30)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            times = []
            for _ in range(10):
                start = time.perf_counter()
                fiedler_full, _ = lanczos_fiedler_gpu(L_full, num_iterations=30)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

            fiedler_full = fiedler_full.reshape(H, W)
            mean_time_full = sum(times) / len(times)
            print(f"  full lanczos: {mean_time_full:.2f}ms ({1000/mean_time_full:.0f} Hz)")

            # Correlation
            fc = fiedler_c.flatten() - fiedler_c.mean()
            ff = fiedler_full.flatten() - fiedler_full.mean()
            corr = abs((fc * ff).sum() / (torch.sqrt((fc**2).sum() * (ff**2).sum()) + 1e-10)).item()
            print(f"    correlation with full: {corr:.3f}")
            print(f"    speedup: {mean_time_full/mean_time:.1f}x")

        except ImportError:
            print("  (full Lanczos not available)")

        print()

    print("✓ Benchmark complete")
