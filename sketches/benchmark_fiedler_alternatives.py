"""
Benchmark: Fiedler Vector Approximation Alternatives

Compares speed and correlation of different approaches to computing
Fiedler-like spectral signals for the shader pipeline.

Methods tested:
1. Lanczos (ground truth) - exact Fiedler vector
2. Heat Diffusion - (I - tL)^k @ random_vec
3. Chebyshev Filtering - polynomial approximation around low eigenvalues
4. SDF Gating - signed distance from contours (no spectral computation)
5. Bilateral Proxy - fast bilateral filter response (implicit spectral awareness)

Usage:
    python benchmark_fiedler_alternatives.py [image_path]
"""

import torch
import torch.nn.functional as F
import time
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Helper Functions
# =============================================================================

def to_grayscale(rgb: torch.Tensor) -> torch.Tensor:
    """RGB to grayscale."""
    if rgb.dim() == 2:
        return rgb
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def sobel_edges(gray: torch.Tensor) -> torch.Tensor:
    """Compute edge magnitude via Sobel."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           device=gray.device, dtype=gray.dtype).view(1, 1, 3, 3)

    g = gray.unsqueeze(0).unsqueeze(0)
    g_pad = F.pad(g, (1, 1, 1, 1), mode='reflect')

    gx = F.conv2d(g_pad, sobel_x).squeeze()
    gy = F.conv2d(g_pad, sobel_y).squeeze()

    return torch.sqrt(gx**2 + gy**2)


def correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()

    a_mean = a_flat - a_flat.mean()
    b_mean = b_flat - b_flat.mean()

    num = (a_mean * b_mean).sum()
    denom = torch.sqrt((a_mean**2).sum() * (b_mean**2).sum()) + 1e-10

    return abs(num / denom).item()


# =============================================================================
# Method 1: Lanczos (Ground Truth)
# =============================================================================

def fiedler_lanczos(gray: torch.Tensor, iterations: int = 30) -> torch.Tensor:
    """Compute true Fiedler vector via Lanczos iteration."""
    from spectral_ops_fast_cuter import build_weighted_image_laplacian, lanczos_fiedler_gpu

    H, W = gray.shape
    if gray.max() > 1.0:
        gray = gray / 255.0

    L = build_weighted_image_laplacian(gray, edge_threshold=0.1)
    fiedler_flat, _ = lanczos_fiedler_gpu(L, num_iterations=iterations)

    return fiedler_flat.reshape(H, W)


# =============================================================================
# Method 2: Heat Diffusion
# =============================================================================

def fiedler_heat_diffusion(gray: torch.Tensor, t: float = 0.1, iterations: int = 30) -> torch.Tensor:
    """
    Approximate Fiedler via heat diffusion: (I - tL)^k @ random_vec

    As k increases, this converges to smooth eigenvectors.
    With t small and k moderate, captures Fiedler-like structure.
    """
    from spectral_ops_fast_cuter import build_weighted_image_laplacian

    H, W = gray.shape
    n = H * W
    device = gray.device

    if gray.max() > 1.0:
        gray = gray / 255.0

    L = build_weighted_image_laplacian(gray, edge_threshold=0.1)

    # Random starting vector, mean-centered (orthogonal to constant eigenvector)
    torch.manual_seed(42)
    v = torch.randn(n, device=device, dtype=torch.float32)
    v = v - v.mean()
    v = v / torch.linalg.norm(v)

    # Heat diffusion: v_{k+1} = v_k - t * L @ v_k  (explicit Euler)
    # This attenuates high-frequency components faster than low-frequency
    for _ in range(iterations):
        Lv = torch.sparse.mm(L, v.unsqueeze(1)).squeeze(1)
        v = v - t * Lv
        v = v - v.mean()  # Keep orthogonal to constant
        v = v / torch.linalg.norm(v)

    return v.reshape(H, W)


# =============================================================================
# Method 3: Chebyshev Polynomial Filtering
# =============================================================================

def fiedler_chebyshev(gray: torch.Tensor, order: int = 10, center: float = 0.1) -> torch.Tensor:
    """
    Chebyshev polynomial filter targeting low eigenvalues.

    Uses low-pass filter with cutoff near `center`, then subtracts mean
    to get Fiedler-like structure (orthogonal to constant eigenvector).
    """
    from spectral_ops_fast_cuter import build_weighted_image_laplacian

    H, W = gray.shape
    n = H * W
    device = gray.device

    if gray.max() > 1.0:
        gray = gray / 255.0

    L = build_weighted_image_laplacian(gray, edge_threshold=0.1)

    # Spectral range for 4-connected weighted Laplacian
    # Maximum eigenvalue bounded by max degree * 2 ~ 8 for 4-connected
    lambda_max = 4.0

    # Random starting vector (mean-centered to exclude constant eigenvector)
    torch.manual_seed(42)
    v = torch.randn(n, device=device, dtype=torch.float32)
    v = v - v.mean()
    v = v / torch.linalg.norm(v)

    # Low-pass Chebyshev filter: h(x) = cos(K * arccos(1 - x/cutoff)) for x < cutoff
    # This amplifies eigenvalues near 0 (but we project out lambda_1=0)
    # Cutoff at `center` means we keep components with eigenvalue < center

    # Jackson-Chebyshev coefficients for smooth low-pass
    cutoff = center * lambda_max  # actual eigenvalue cutoff

    # Chebyshev recurrence on scaled operator: L_tilde = I - L/lambda_max
    # Maps eigenvalues from [0, lambda_max] to [1, -1]
    # Low eigenvalues of L -> high values of L_tilde (near 1)

    # T_0(L_tilde) @ v = v
    T_prev = v.clone()

    # T_1(L_tilde) @ v = L_tilde @ v = v - L @ v / lambda_max
    Lv = torch.sparse.mm(L, v.unsqueeze(1)).squeeze(1)
    T_curr = v - Lv / lambda_max

    # Accumulate with Jackson damping
    # h_k = (2 - delta_k0) * cos(k * theta_c) where theta_c = arccos(1 - 2*cutoff/lambda_max)
    theta_c = np.arccos(np.clip(1 - 2 * cutoff / lambda_max, -1, 1))

    def jackson(k, K):
        """Jackson damping to reduce Gibbs oscillation."""
        alpha = np.pi / (K + 2)
        return ((K - k + 1) * np.cos(alpha * k) + np.sin(alpha * k) / np.tan(alpha)) / (K + 1)

    c0 = theta_c / np.pi
    result = c0 * jackson(0, order) * T_prev

    c1 = 2 * np.sin(theta_c) / np.pi
    result = result + c1 * jackson(1, order) * T_curr

    for k in range(2, order + 1):
        # T_{k+1}(L_tilde) = 2 * L_tilde * T_k - T_{k-1}
        LT_curr = torch.sparse.mm(L, T_curr.unsqueeze(1)).squeeze(1)
        T_next = 2 * (T_curr - LT_curr / lambda_max) - T_prev

        ck = 2 * np.sin(k * theta_c) / (k * np.pi)
        result = result + ck * jackson(k, order) * T_next

        T_prev, T_curr = T_curr, T_next

    # Mean-center (project out constant eigenvector)
    result = result - result.mean()
    norm = torch.linalg.norm(result)
    if norm > 1e-10:
        result = result / norm

    return result.reshape(H, W)


# =============================================================================
# Method 4: SDF Gating (No Spectral Computation)
# =============================================================================

def fiedler_sdf(gray: torch.Tensor, percentile: float = 85.0) -> torch.Tensor:
    """
    Signed distance field from contours.

    Fast O(n log n) approximation using JFA (Jump Flooding Algorithm) style.
    This is NOT a spectral method but provides similar partition structure.
    """
    H, W = gray.shape
    device = gray.device

    if gray.max() > 1.0:
        gray = gray / 255.0

    # Detect contours via edge magnitude
    edges = sobel_edges(gray)
    threshold = torch.quantile(edges.flatten(), percentile / 100.0)
    contour_mask = edges > threshold

    # Approximate distance transform using iterative dilation
    # (Pure PyTorch, no scipy)
    dist = torch.zeros_like(gray)
    dist[contour_mask] = 0
    dist[~contour_mask] = float('inf')

    # Iterative distance propagation (approximate)
    kernel = torch.tensor([[1.414, 1.0, 1.414],
                           [1.0,   0.0, 1.0],
                           [1.414, 1.0, 1.414]], device=device, dtype=gray.dtype)
    kernel = kernel.view(1, 1, 3, 3)

    for _ in range(max(H, W) // 2):
        dist_pad = F.pad(dist.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
        # For each pixel, min of (neighbor + distance to neighbor)
        neighbors = F.unfold(dist_pad, 3).reshape(9, H, W)
        offsets = kernel.flatten()
        dist_candidates = neighbors + offsets.view(9, 1, 1)
        dist = dist_candidates.min(dim=0)[0]

    # Sign: positive outside contour (low edge), negative inside (high edge)
    # Use smoothed edge response as sign proxy
    edge_smooth = F.avg_pool2d(edges.unsqueeze(0).unsqueeze(0), 5, 1, 2).squeeze()
    sign = torch.where(edge_smooth > edge_smooth.median(), -1.0, 1.0)

    sdf = sign * dist

    # Normalize to similar range as Fiedler
    sdf = sdf - sdf.mean()
    sdf = sdf / (sdf.std() + 1e-10)

    return sdf


# =============================================================================
# Method 5: Bilateral Proxy
# =============================================================================

def fiedler_bilateral(gray: torch.Tensor, sigma_s: float = 5.0, sigma_r: float = 0.1) -> torch.Tensor:
    """
    Bilateral filter residual as spectral proxy.

    Bilateral filtering preserves edges while smoothing - the residual
    (original - filtered) captures edge-aligned structure similar to Fiedler.
    """
    H, W = gray.shape
    device = gray.device

    if gray.max() > 1.0:
        gray = gray / 255.0

    # Approximate bilateral via guided filter approach
    # Faster than full bilateral, similar spectral properties

    # Spatial Gaussian kernel
    k = int(3 * sigma_s)
    if k % 2 == 0:
        k += 1
    x = torch.arange(k, device=device, dtype=gray.dtype) - k // 2
    spatial_kernel = torch.exp(-x**2 / (2 * sigma_s**2))
    spatial_kernel = spatial_kernel / spatial_kernel.sum()

    # Separable spatial blur
    g_pad = F.pad(gray.unsqueeze(0).unsqueeze(0), (k//2, k//2, k//2, k//2), mode='reflect')

    # Horizontal pass
    blur_h = F.conv2d(g_pad, spatial_kernel.view(1, 1, 1, -1))
    # Vertical pass
    blur = F.conv2d(blur_h, spatial_kernel.view(1, 1, -1, 1)).squeeze()

    # Range weighting: downweight contributions from different intensities
    # Approximate by local variance scaling
    local_mean = blur
    local_var = F.avg_pool2d((gray.unsqueeze(0).unsqueeze(0) - local_mean.unsqueeze(0).unsqueeze(0))**2,
                              k, 1, k//2).squeeze()

    # Edge-preserving blend
    edge_weight = torch.exp(-local_var / (sigma_r**2 + 1e-10))
    filtered = edge_weight * blur + (1 - edge_weight) * gray

    # Residual captures edge structure
    residual = gray - filtered

    # Normalize
    residual = residual - residual.mean()
    residual = residual / (residual.std() + 1e-10)

    return residual


# =============================================================================
# Method 6: Power Iteration (Inverse)
# =============================================================================

def fiedler_power_inverse(gray: torch.Tensor, iterations: int = 30, shift: float = 0.01) -> torch.Tensor:
    """
    Inverse power iteration with shift.

    Computes (L - shift*I)^{-1} approximately via iterative solving,
    which converges to eigenvector of smallest eigenvalue > shift.
    """
    from spectral_ops_fast_cuter import build_weighted_image_laplacian

    H, W = gray.shape
    n = H * W
    device = gray.device

    if gray.max() > 1.0:
        gray = gray / 255.0

    L = build_weighted_image_laplacian(gray, edge_threshold=0.1)

    # Add shift to avoid singularity at lambda_1=0
    diag_idx = torch.arange(n, device=device)
    L_shifted_indices = L.indices()
    L_shifted_values = L.values().clone()

    # Find diagonal entries and add shift
    diag_mask = L_shifted_indices[0] == L_shifted_indices[1]
    L_shifted_values[diag_mask] += shift

    L_shifted = torch.sparse_coo_tensor(L_shifted_indices, L_shifted_values, (n, n)).coalesce()

    # Random starting vector
    torch.manual_seed(42)
    v = torch.randn(n, device=device, dtype=torch.float32)
    v = v - v.mean()
    v = v / torch.linalg.norm(v)

    # Approximate inverse via Jacobi iteration: (L_shifted)^{-1} @ v
    # x_{k+1} = D^{-1} @ (b - (L-D) @ x_k)
    # For Laplacian, D = diag(L)

    diag_vals = torch.zeros(n, device=device)
    diag_vals.scatter_(0, L_shifted_indices[0][diag_mask], L_shifted_values[diag_mask])
    D_inv = 1.0 / (diag_vals + 1e-10)

    for _ in range(iterations):
        # b = v (target)
        # Lx = L_shifted @ x
        Lx = torch.sparse.mm(L_shifted, v.unsqueeze(1)).squeeze(1)
        # Jacobi update: x = D^{-1} @ (v - (L - D) @ x) = D^{-1} @ v + x - D^{-1} @ L @ x
        # Simplified: x = x + D^{-1} @ (v - L @ x)
        residual = v - Lx
        v = v + 0.5 * D_inv * residual  # Damped Jacobi
        v = v - v.mean()
        v = v / torch.linalg.norm(v)

    return v.reshape(H, W)


# =============================================================================
# Benchmarking
# =============================================================================

def benchmark_method(name: str, method_fn, gray: torch.Tensor,
                     ground_truth: torch.Tensor, warmup: int = 3, runs: int = 10) -> Dict:
    """Benchmark a single method."""
    device = gray.device

    # Warmup
    for _ in range(warmup):
        _ = method_fn(gray)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = method_fn(gray)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    corr = correlation(result, ground_truth)

    return {
        'name': name,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'correlation': corr,
        'result': result
    }


def run_benchmark(image_path: Optional[str] = None):
    """Run full benchmark on test image."""

    # Load or generate test image
    if image_path and Path(image_path).exists():
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        rgb = torch.tensor(np.array(img), device=DEVICE, dtype=torch.float32) / 255.0
        print(f"Loaded: {image_path} ({rgb.shape[0]}x{rgb.shape[1]})")
    else:
        # Synthetic test pattern
        H, W = 256, 256
        y, x = torch.meshgrid(
            torch.linspace(0, 1, H, device=DEVICE),
            torch.linspace(0, 1, W, device=DEVICE),
            indexing='ij'
        )
        # Gradient + blobs pattern
        rgb = torch.stack([
            x * 0.5 + 0.25,
            y * 0.5 + 0.25,
            ((x - 0.5)**2 + (y - 0.5)**2).sqrt()
        ], dim=-1)
        # Add some structure
        circles = ((x - 0.3)**2 + (y - 0.3)**2 < 0.05).float() * 0.3
        circles += ((x - 0.7)**2 + (y - 0.7)**2 < 0.08).float() * 0.3
        rgb = rgb * (1 - circles.unsqueeze(-1)) + circles.unsqueeze(-1)
        print(f"Generated synthetic image: {H}x{W}")

    gray = to_grayscale(rgb)

    print("\n" + "="*60)
    print("Computing ground truth (Lanczos Fiedler)...")
    print("="*60)

    # Ground truth
    start = time.perf_counter()
    ground_truth = fiedler_lanczos(gray, iterations=50)
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    gt_time = time.perf_counter() - start
    print(f"Ground truth computed in {gt_time:.3f}s")

    # Methods to benchmark - ordered by expected quality
    methods = [
        # Lanczos variants (expected best)
        ("Lanczos-30", lambda g: fiedler_lanczos(g, iterations=30)),
        ("Lanczos-20", lambda g: fiedler_lanczos(g, iterations=20)),
        ("Lanczos-15", lambda g: fiedler_lanczos(g, iterations=15)),
        ("Lanczos-10", lambda g: fiedler_lanczos(g, iterations=10)),

        # Iterative methods
        ("Heat-Diffusion-30", lambda g: fiedler_heat_diffusion(g, t=0.1, iterations=30)),
        ("Heat-Diffusion-50", lambda g: fiedler_heat_diffusion(g, t=0.05, iterations=50)),
        ("Power-Inverse-30", lambda g: fiedler_power_inverse(g, iterations=30)),

        # Polynomial filters
        ("Chebyshev-10", lambda g: fiedler_chebyshev(g, order=10, center=0.1)),
        ("Chebyshev-20", lambda g: fiedler_chebyshev(g, order=20, center=0.05)),

        # Non-spectral proxies
        ("SDF", lambda g: fiedler_sdf(g, percentile=85.0)),
        ("Bilateral", lambda g: fiedler_bilateral(g, sigma_s=5.0, sigma_r=0.1)),
    ]

    print("\n" + "="*60)
    print("Benchmarking methods (warmup=3, runs=10)")
    print("="*60)

    results = []
    for name, fn in methods:
        print(f"  {name}...", end=" ", flush=True)
        try:
            r = benchmark_method(name, fn, gray, ground_truth)
            results.append(r)
            print(f"corr={r['correlation']:.3f}, time={r['mean_time']*1000:.1f}ms")
        except Exception as e:
            print(f"ERROR: {e}")

    # Summary table
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Method':<20} {'Correlation':>12} {'Time (ms)':>12} {'Speedup':>10}")
    print("-"*60)

    # Sort by correlation
    results.sort(key=lambda r: r['correlation'], reverse=True)

    for r in results:
        speedup = gt_time / r['mean_time']
        print(f"{r['name']:<20} {r['correlation']:>12.3f} {r['mean_time']*1000:>12.1f} {speedup:>10.1f}x")

    print("-"*60)
    print(f"{'Ground Truth':<20} {'1.000':>12} {gt_time*1000:>12.1f} {'1.0x':>10}")

    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    # Find best speed/correlation tradeoffs
    fast_accurate = [r for r in results if r['correlation'] > 0.7 and r['mean_time'] < gt_time * 0.5]
    if fast_accurate:
        best = max(fast_accurate, key=lambda r: r['correlation'] / r['mean_time'])
        print(f"Best speed/accuracy tradeoff: {best['name']}")
        print(f"  - Correlation: {best['correlation']:.3f}")
        print(f"  - Speedup: {gt_time/best['mean_time']:.1f}x")

    very_fast = [r for r in results if r['mean_time'] < gt_time * 0.2]
    if very_fast:
        best_fast = max(very_fast, key=lambda r: r['correlation'])
        print(f"\nFastest with decent correlation: {best_fast['name']}")
        print(f"  - Correlation: {best_fast['correlation']:.3f}")
        print(f"  - Speedup: {gt_time/best_fast['mean_time']:.1f}x")

    return results


if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Try default test images
    if image_path is None:
        for candidate in ["demo_output/inputs/1bit redraw.png",
                          "demo_output/inputs/snek-heavy.png",
                          "demo_output/inputs/toof.png"]:
            if Path(candidate).exists():
                image_path = candidate
                break

    results = run_benchmark(image_path)
