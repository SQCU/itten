"""
Fast local spectral structure detection for real-time graphics.

Target: ~100Hz (10ms) on 256x256 images.
Budget: ~100 sparse matrix-vector products per frame.

Key insight: We don't need GLOBAL Fiedler (graph bipartition).
We need LOCAL structure: curvature, branching, twist.

Approach:
1. Low-pass filter applied to POSITION signals (x, y)
2. Spectrally-smoothed positions reveal local graph structure
3. Deviation from original positions = local structural complexity
4. Gradient of smoothed field = local orientation
5. Hessian of smoothed field = local curvature

No eigenvector computation. Just polynomial filtering.
"""

import torch
import torch.nn.functional as F
from typing import Tuple
import time


def build_laplacian_fast(gray: torch.Tensor, edge_threshold: float = 0.1) -> torch.Tensor:
    """Build sparse 4-connected Laplacian. Optimized for speed."""
    device = gray.device
    H, W = gray.shape
    n = H * W

    if gray.max() > 1.0:
        gray = gray / 255.0

    # Horizontal edges
    w_h = torch.exp(-torch.abs(gray[:, 1:] - gray[:, :-1]).flatten() / edge_threshold)
    h_idx = torch.arange(H * (W - 1), device=device)
    h_left = h_idx + h_idx // (W - 1)
    h_right = h_left + 1

    # Vertical edges
    w_v = torch.exp(-torch.abs(gray[1:, :] - gray[:-1, :]).flatten() / edge_threshold)
    v_top = torch.arange((H - 1) * W, device=device)
    v_bottom = v_top + W

    # Sparse COO construction
    rows = torch.cat([h_left, h_right, v_top, v_bottom])
    cols = torch.cat([h_right, h_left, v_bottom, v_top])
    vals = torch.cat([-w_h, -w_h, -w_v, -w_v])

    # Degrees via scatter
    degrees = torch.zeros(n, device=device)
    degrees.scatter_add_(0, h_left, w_h)
    degrees.scatter_add_(0, h_right, w_h)
    degrees.scatter_add_(0, v_top, w_v)
    degrees.scatter_add_(0, v_bottom, w_v)

    diag = torch.arange(n, device=device)
    rows = torch.cat([rows, diag])
    cols = torch.cat([cols, diag])
    vals = torch.cat([vals, degrees])

    return torch.sparse_coo_tensor(torch.stack([rows.long(), cols.long()]), vals, (n, n)).coalesce()


def fast_lowpass(L: torch.Tensor, signal: torch.Tensor, order: int = 10, t: float = 0.25) -> torch.Tensor:
    """
    Fast low-pass filter via heat diffusion: (I - tL)^order @ signal

    Complexity: O(order × nnz(L)) = O(order × n) for grid graphs.
    For order=10, n=65536: ~10 sparse matvecs = ~1ms on GPU.
    """
    result = signal.clone()
    for _ in range(order):
        Lr = torch.sparse.mm(L, result if result.dim() == 2 else result.unsqueeze(1))
        if result.dim() == 1:
            Lr = Lr.squeeze(1)
        result = result - t * Lr
    return result


def local_spectral_structure(
    image_rgb: torch.Tensor,
    order: int = 10,
    t: float = 0.25
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute local spectral structure features in O(order × n) time.

    Returns:
        smoothed_gray: (H, W) - spectrally smoothed intensity
        structure_magnitude: (H, W) - local structural complexity
        orientation_x, orientation_y: (H, W) - local structure orientation (gradient direction)

    The key insight: local structure = how much does the graph Laplacian
    "pull" at this pixel? High Laplacian magnitude = near edges/branches.
    """
    H, W = image_rgb.shape[:2]
    device = image_rgb.device
    n = H * W

    # Grayscale
    gray = 0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1] + 0.114 * image_rgb[..., 2]

    # Build Laplacian
    L = build_laplacian_fast(gray)

    # Smooth the grayscale
    gray_flat = gray.flatten()
    smoothed_flat = fast_lowpass(L, gray_flat, order=order, t=t)
    smoothed_gray = smoothed_flat.reshape(H, W)

    # Structure magnitude = |L @ smoothed| = local second derivative
    # This captures how much the graph structure "curves" at each pixel
    Ls = torch.sparse.mm(L, smoothed_flat.unsqueeze(1)).squeeze()
    structure_magnitude = torch.abs(Ls).reshape(H, W)

    # Also compute local variance as structure indicator
    # Use a small pooling window to get local statistics
    pool_size = 5
    sm_4d = smoothed_gray.unsqueeze(0).unsqueeze(0)
    local_mean = F.avg_pool2d(sm_4d, pool_size, stride=1, padding=pool_size//2)
    local_var = F.avg_pool2d((sm_4d - local_mean)**2, pool_size, stride=1, padding=pool_size//2)
    local_std = torch.sqrt(local_var + 1e-8).squeeze()

    # Combine: structure = Laplacian magnitude + local variation
    structure_magnitude = structure_magnitude + local_std * 10

    # Orientation from Sobel gradient of smoothed field
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           device=device, dtype=gray.dtype).view(1, 1, 3, 3) / 8
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           device=device, dtype=gray.dtype).view(1, 1, 3, 3) / 8

    sm_padded = F.pad(sm_4d, (1, 1, 1, 1), mode='reflect')
    grad_x = F.conv2d(sm_padded, sobel_x).squeeze()
    grad_y = F.conv2d(sm_padded, sobel_y).squeeze()

    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    orientation_x = grad_x / grad_mag
    orientation_y = grad_y / grad_mag

    return smoothed_gray, structure_magnitude, orientation_x, orientation_y


def local_curvature(
    image_rgb: torch.Tensor,
    order: int = 10,
    t: float = 0.25
) -> torch.Tensor:
    """
    Compute local graph curvature via Laplacian of smoothed field.

    High curvature = corners, branch points, pinch points.
    Low curvature = straight lines, smooth regions.
    """
    H, W = image_rgb.shape[:2]
    device = image_rgb.device

    gray = 0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1] + 0.114 * image_rgb[..., 2]
    L = build_laplacian_fast(gray)

    # Smooth the grayscale
    smoothed = fast_lowpass(L, gray.flatten(), order=order, t=t)

    # Curvature = magnitude of Laplacian of smoothed field
    # L @ smoothed gives second derivative (discrete curvature)
    Ls = torch.sparse.mm(L, smoothed.unsqueeze(1)).squeeze()
    curvature = torch.abs(Ls).reshape(H, W)

    return curvature


def spectral_gate_fast(
    image_rgb: torch.Tensor,
    order: int = 10,
    t: float = 0.25,
    threshold_percentile: float = 40.0,
    sharpness: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast spectral gate using local structure instead of global Fiedler.

    Gates on local structural complexity:
    - High gate: complex structure (branch points, high curvature)
    - Low gate: simple structure (straight lines, smooth regions)
    """
    _, structure_mag, _, _ = local_spectral_structure(image_rgb, order, t)

    # Adaptive threshold
    flat = structure_mag.flatten()
    k = int(len(flat) * threshold_percentile / 100.0)
    threshold = torch.kthvalue(flat, max(1, min(k, len(flat))))[0]

    gate = torch.sigmoid((structure_mag - threshold) * sharpness)

    return gate, structure_mag


def spectral_correspondence_fast(
    image_A: torch.Tensor,
    image_B: torch.Tensor,
    order: int = 10,
    t: float = 0.25,
    n_bins: int = 256
) -> torch.Tensor:
    """
    Fast cross-image correspondence using local spectral structure.

    Matches pixels by local structural complexity rather than global Fiedler.
    """
    H_A, W_A = image_A.shape[:2]
    H_B, W_B = image_B.shape[:2]
    device = image_A.device

    # Compute structure for both images
    _, struct_A, _, _ = local_spectral_structure(image_A, order, t)
    _, struct_B, _, _ = local_spectral_structure(image_B, order, t)

    # Normalize to [0, 1] using joint range
    s_min = min(struct_A.min(), struct_B.min())
    s_max = max(struct_A.max(), struct_B.max())
    struct_A_norm = (struct_A - s_min) / (s_max - s_min + 1e-8)
    struct_B_norm = (struct_B - s_min) / (s_max - s_min + 1e-8)

    # Bin B pixels by structure
    bin_B = (struct_B_norm.flatten() * (n_bins - 1)).long().clamp(0, n_bins - 1)

    # Mean position per bin
    y_B = torch.arange(H_B, device=device).unsqueeze(1).expand(H_B, W_B).flatten().float()
    x_B = torch.arange(W_B, device=device).unsqueeze(0).expand(H_B, W_B).flatten().float()

    y_sum = torch.zeros(n_bins, device=device)
    x_sum = torch.zeros(n_bins, device=device)
    count = torch.zeros(n_bins, device=device)

    y_sum.scatter_add_(0, bin_B, y_B)
    x_sum.scatter_add_(0, bin_B, x_B)
    count.scatter_add_(0, bin_B, torch.ones_like(y_B))

    valid = count > 0
    y_mean = torch.zeros(n_bins, device=device)
    x_mean = torch.zeros(n_bins, device=device)
    y_mean[valid] = y_sum[valid] / count[valid]
    x_mean[valid] = x_sum[valid] / count[valid]

    # Fill empty bins
    for i in range(1, n_bins):
        if not valid[i]:
            y_mean[i], x_mean[i] = y_mean[i-1], x_mean[i-1]
    for i in range(n_bins - 2, -1, -1):
        if not valid[i]:
            y_mean[i], x_mean[i] = y_mean[i+1], x_mean[i+1]

    # Look up for A pixels
    bin_A = (struct_A_norm.flatten() * (n_bins - 1)).long().clamp(0, n_bins - 1)
    sample_y = y_mean[bin_A].reshape(H_A, W_A)
    sample_x = x_mean[bin_A].reshape(H_A, W_A)

    # Grid sample
    grid = torch.stack([
        2.0 * sample_x / (W_B - 1) - 1.0,
        2.0 * sample_y / (H_B - 1) - 1.0
    ], dim=-1).unsqueeze(0)

    image_B_4d = image_B.permute(2, 0, 1).unsqueeze(0)
    sampled = F.grid_sample(image_B_4d, grid, mode='bilinear',
                            padding_mode='border', align_corners=True)

    return sampled.squeeze(0).permute(1, 2, 0)


def local_geometric_features(
    image_rgb: torch.Tensor,
    order: int = 10,
    t: float = 0.25
) -> dict:
    """
    Compute full set of local geometric features for structure-aware shading.

    Returns dict with:
        - smoothed: spectrally smoothed intensity
        - laplacian_mag: |L @ smoothed| - edge/branch detection
        - gradient_mag: |∇smoothed| - edge strength
        - gradient_dir: (dx, dy) - local orientation
        - curvature: how fast gradient direction changes - corners/bends
        - divergence: ∇·gradient - sources/sinks
    """
    H, W = image_rgb.shape[:2]
    device = image_rgb.device
    dtype = image_rgb.dtype

    # Grayscale and Laplacian
    gray = 0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1] + 0.114 * image_rgb[..., 2]
    L = build_laplacian_fast(gray)

    # Smooth
    smoothed = fast_lowpass(L, gray.flatten(), order=order, t=t).reshape(H, W)

    # Laplacian magnitude (second derivative / curvature indicator)
    Ls = torch.sparse.mm(L, smoothed.flatten().unsqueeze(1)).squeeze().reshape(H, W)
    laplacian_mag = torch.abs(Ls)

    # Gradient via Sobel
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3) / 8
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype).view(1, 1, 3, 3) / 8

    sm_4d = smoothed.unsqueeze(0).unsqueeze(0)
    sm_pad = F.pad(sm_4d, (1, 1, 1, 1), mode='reflect')

    grad_x = F.conv2d(sm_pad, sobel_x).squeeze()
    grad_y = F.conv2d(sm_pad, sobel_y).squeeze()

    gradient_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    grad_dir_x = grad_x / gradient_mag
    grad_dir_y = grad_y / gradient_mag

    # Curvature = rate of change of gradient direction
    # Approximate via Laplacian of gradient direction (how fast does it twist)
    dir_x_4d = grad_dir_x.unsqueeze(0).unsqueeze(0)
    dir_y_4d = grad_dir_y.unsqueeze(0).unsqueeze(0)
    dir_x_pad = F.pad(dir_x_4d, (1, 1, 1, 1), mode='reflect')
    dir_y_pad = F.pad(dir_y_4d, (1, 1, 1, 1), mode='reflect')

    # Gradient of direction field
    ddx_x = F.conv2d(dir_x_pad, sobel_x).squeeze()
    ddx_y = F.conv2d(dir_x_pad, sobel_y).squeeze()
    ddy_x = F.conv2d(dir_y_pad, sobel_x).squeeze()
    ddy_y = F.conv2d(dir_y_pad, sobel_y).squeeze()

    # Curvature = magnitude of gradient-of-direction
    curvature = torch.sqrt(ddx_x**2 + ddx_y**2 + ddy_x**2 + ddy_y**2 + 1e-8)

    # Divergence = ∂dx/∂x + ∂dy/∂y (sources/sinks in direction field)
    divergence = ddx_x + ddy_y

    return {
        'smoothed': smoothed,
        'laplacian_mag': laplacian_mag,
        'gradient_mag': gradient_mag,
        'gradient_dir': (grad_dir_x, grad_dir_y),
        'curvature': curvature,
        'divergence': divergence,
    }


# ============================================================
# Benchmark
# ============================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Target: 10ms (100Hz)")
    print()

    # Test at multiple resolutions
    for size in [128, 256, 512]:
        H, W = size, size
        print(f"=== {H}x{W} ({H*W} pixels) ===")

        # Generate test image
        torch.manual_seed(42)
        y, x = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )
        image = torch.stack([x, y, 0.5 * torch.ones_like(x)], dim=-1)
        circle = ((x - 0.3)**2 + (y - 0.3)**2 < 0.04).float()
        image = image * (1 - circle.unsqueeze(-1)) + circle.unsqueeze(-1) * 0.8

        # Warmup
        for _ in range(3):
            _ = local_spectral_structure(image, order=10)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark local structure
        times = []
        for _ in range(10):
            start = time.perf_counter()
            smoothed, struct_mag, ori_x, ori_y = local_spectral_structure(image, order=10)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        mean_time = sum(times) / len(times) * 1000
        fps = 1000 / mean_time
        print(f"  local_spectral_structure: {mean_time:.2f}ms ({fps:.1f} Hz)")
        print(f"    structure range: [{struct_mag.min():.4f}, {struct_mag.max():.4f}]")

        # Benchmark curvature
        times = []
        for _ in range(10):
            start = time.perf_counter()
            curv = local_curvature(image, order=10)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        mean_time = sum(times) / len(times) * 1000
        fps = 1000 / mean_time
        print(f"  local_curvature: {mean_time:.2f}ms ({fps:.1f} Hz)")

        # Benchmark gate
        times = []
        for _ in range(10):
            start = time.perf_counter()
            gate, _ = spectral_gate_fast(image, order=10)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        mean_time = sum(times) / len(times) * 1000
        fps = 1000 / mean_time
        print(f"  spectral_gate_fast: {mean_time:.2f}ms ({fps:.1f} Hz)")
        print(f"    gate range: [{gate.min():.3f}, {gate.max():.3f}]")

        # Compare with Lanczos Fiedler
        try:
            from spectral_ops_fast_cuter import build_weighted_image_laplacian, lanczos_fiedler_gpu

            gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
            L = build_weighted_image_laplacian(gray)

            # Warmup
            for _ in range(3):
                _ = lanczos_fiedler_gpu(L, num_iterations=30)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            times = []
            for _ in range(10):
                start = time.perf_counter()
                fiedler, _ = lanczos_fiedler_gpu(L, num_iterations=30)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            mean_time = sum(times) / len(times) * 1000
            fps = 1000 / mean_time
            print(f"  lanczos_fiedler (30 iter): {mean_time:.2f}ms ({fps:.1f} Hz)")

        except ImportError:
            print("  (Lanczos not available)")

        print()

    print("✓ Benchmark complete")

    # Test geometric features
    print("\n=== Geometric Features Test ===")
    H, W = 256, 256
    torch.manual_seed(42)
    y, x = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing='ij'
    )
    # More complex test pattern: multiple circles, lines
    image = torch.stack([x * 0.5, y * 0.5, 0.3 * torch.ones_like(x)], dim=-1)
    # Add circles
    c1 = ((x - 0.25)**2 + (y - 0.25)**2 < 0.02).float()
    c2 = ((x - 0.75)**2 + (y - 0.75)**2 < 0.03).float()
    c3 = ((x - 0.5)**2 + (y - 0.3)**2 < 0.015).float()
    # Add line
    line = (torch.abs(y - 0.7) < 0.01).float() * (x > 0.2).float() * (x < 0.8).float()
    # Compose
    mask = (c1 + c2 + c3 + line).clamp(0, 1)
    image = image * (1 - mask.unsqueeze(-1)) + mask.unsqueeze(-1) * torch.tensor([0.9, 0.1, 0.1], device=device)

    start = time.perf_counter()
    features = local_geometric_features(image, order=10)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    feat_time = (time.perf_counter() - start) * 1000

    print(f"Time for all features: {feat_time:.2f}ms")
    print(f"Smoothed range: [{features['smoothed'].min():.3f}, {features['smoothed'].max():.3f}]")
    print(f"Laplacian mag range: [{features['laplacian_mag'].min():.4f}, {features['laplacian_mag'].max():.4f}]")
    print(f"Gradient mag range: [{features['gradient_mag'].min():.4f}, {features['gradient_mag'].max():.4f}]")
    print(f"Curvature range: [{features['curvature'].min():.4f}, {features['curvature'].max():.4f}]")
    print(f"Divergence range: [{features['divergence'].min():.4f}, {features['divergence'].max():.4f}]")

    # Compare with Fiedler
    try:
        from spectral_ops_fast_cuter import build_weighted_image_laplacian, lanczos_fiedler_gpu

        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        L = build_weighted_image_laplacian(gray)
        fiedler, _ = lanczos_fiedler_gpu(L, num_iterations=30)
        fiedler = fiedler.reshape(H, W)

        # Correlations
        def corr(a, b):
            a_flat = a.flatten() - a.mean()
            b_flat = b.flatten() - b.mean()
            return abs((a_flat * b_flat).sum() / (
                torch.sqrt((a_flat**2).sum() * (b_flat**2).sum()) + 1e-10
            )).item()

        print(f"\nCorrelations with Fiedler:")
        print(f"  smoothed: {corr(features['smoothed'], fiedler):.3f}")
        print(f"  laplacian_mag: {corr(features['laplacian_mag'], torch.abs(fiedler)):.3f}")
        print(f"  gradient_mag: {corr(features['gradient_mag'], torch.abs(fiedler)):.3f}")
        print(f"  curvature: {corr(features['curvature'], torch.abs(fiedler)):.3f}")

        # Key question: does local structure correlate with Fiedler GRADIENT?
        # Fiedler gradient = direction along contours
        fiedler_4d = fiedler.unsqueeze(0).unsqueeze(0)
        fiedler_pad = F.pad(fiedler_4d, (1, 1, 1, 1), mode='reflect')
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=image.dtype).view(1, 1, 3, 3) / 8
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=image.dtype).view(1, 1, 3, 3) / 8
        fiedler_grad_x = F.conv2d(fiedler_pad, sobel_x).squeeze()
        fiedler_grad_y = F.conv2d(fiedler_pad, sobel_y).squeeze()
        fiedler_grad_mag = torch.sqrt(fiedler_grad_x**2 + fiedler_grad_y**2)

        print(f"  gradient_mag vs |∇Fiedler|: {corr(features['gradient_mag'], fiedler_grad_mag):.3f}")

    except ImportError:
        print("(Fiedler comparison not available)")
