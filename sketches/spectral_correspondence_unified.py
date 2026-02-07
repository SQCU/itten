"""
Unified spectral correspondence using O(k·n) polynomial transforms.

This replaces explicit Fiedler eigenvector computation with polynomial-filtered
random probes - the core O(k·n) operation that motivates the repo.

Key insight: phi_A @ phi_B.T (spectral correspondence) can be approximated by
filtering the SAME random probes through different Laplacians. The filtered
probes capture how each graph transforms signals in the low-frequency band.

No explicit eigenvector computation. Just polynomial matrix-vector products.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def build_image_laplacian(gray: torch.Tensor, edge_threshold: float = 0.1) -> torch.Tensor:
    """Build sparse 4-connected weighted Laplacian from grayscale image."""
    device = gray.device
    H, W = gray.shape
    n = H * W

    if gray.max() > 1.0:
        gray = gray / 255.0

    # Horizontal edges: weight = exp(-|color_diff|/threshold)
    h_left = torch.arange(H * (W - 1), device=device)
    h_left = h_left + h_left // (W - 1)  # Skip last column indices
    h_right = h_left + 1
    w_h = torch.exp(-torch.abs(gray[:, 1:] - gray[:, :-1]).flatten() / edge_threshold)

    # Vertical edges
    v_top = torch.arange((H - 1) * W, device=device)
    v_bottom = v_top + W
    w_v = torch.exp(-torch.abs(gray[1:, :] - gray[:-1, :]).flatten() / edge_threshold)

    # Build sparse Laplacian
    rows = torch.cat([h_left, h_right, v_top, v_bottom])
    cols = torch.cat([h_right, h_left, v_bottom, v_top])
    vals = torch.cat([-w_h, -w_h, -w_v, -w_v])

    # Degrees
    degrees = torch.zeros(n, device=device)
    degrees.scatter_add_(0, h_left, w_h)
    degrees.scatter_add_(0, h_right, w_h)
    degrees.scatter_add_(0, v_top, w_v)
    degrees.scatter_add_(0, v_bottom, w_v)

    # Add diagonal
    diag_idx = torch.arange(n, device=device)
    rows = torch.cat([rows, diag_idx])
    cols = torch.cat([cols, diag_idx])
    vals = torch.cat([vals, degrees])

    return torch.sparse_coo_tensor(
        torch.stack([rows.long(), cols.long()]), vals, (n, n)
    ).coalesce()


def chebyshev_lowpass(
    L: torch.Tensor,
    signal: torch.Tensor,
    cutoff: float = 0.2,
    order: int = 20,
    lambda_max: float = 4.0
) -> torch.Tensor:
    """
    Apply Chebyshev low-pass filter emphasizing eigenvalues < cutoff.

    This is THE O(k·n) operation: k = order, n = signal length.
    Complexity: O(order × nnz(L)) ≈ O(k × n) for sparse L.
    """
    n = L.shape[0]
    device = L.device

    # Ensure signal is 2D: (n, num_channels)
    if signal.dim() == 1:
        signal = signal.unsqueeze(1)

    # Scale cutoff to Chebyshev domain
    # L has eigenvalues in [0, lambda_max]
    # We want to pass eigenvalues < cutoff
    # Map to [-1, 1]: x_scaled = 2*x/lambda_max - 1
    # Low eigenvalues map to x_scaled near -1

    # Chebyshev recurrence on scaled operator
    # T_0 = I, T_1 = L_scaled, T_{k+1} = 2*L_scaled*T_k - T_{k-1}

    # Jackson-Chebyshev coefficients for low-pass
    theta_c = torch.arccos(torch.clamp(torch.tensor(1 - 2 * cutoff / lambda_max, device=device), -1, 1))

    def jackson(k, K):
        alpha = torch.tensor(torch.pi / (K + 2), device=device)
        k_t = torch.tensor(k, device=device, dtype=torch.float32) if not isinstance(k, torch.Tensor) else k.float()
        K_t = torch.tensor(K, device=device, dtype=torch.float32) if not isinstance(K, torch.Tensor) else K.float()
        return ((K_t - k_t + 1) * torch.cos(alpha * k_t) +
                torch.sin(alpha * k_t) / torch.tan(alpha)) / (K_t + 1)

    # T_0 term
    c0 = theta_c / torch.pi
    T_prev = signal.clone()
    result = c0 * jackson(0, order) * T_prev

    # T_1 term: L_scaled @ signal = signal - L @ signal / lambda_max
    Ls = torch.sparse.mm(L, signal)
    T_curr = signal - Ls / lambda_max
    c1 = 2 * torch.sin(theta_c) / torch.pi
    result = result + c1 * jackson(1, order) * T_curr

    # Higher order terms
    for k in range(2, order + 1):
        LT = torch.sparse.mm(L, T_curr)
        T_next = 2 * (T_curr - LT / lambda_max) - T_prev

        ck = 2 * torch.sin(k * theta_c) / (k * torch.pi)
        result = result + ck * jackson(k, order) * T_next

        T_prev, T_curr = T_curr, T_next

    return result


def spectral_embedding_from_signal(
    L: torch.Tensor,
    signal: torch.Tensor,  # (n,) input signal to filter
    theta: float = 0.1,
    num_steps: int = 8,
    order: int = 15
) -> torch.Tensor:
    """
    Spectral embedding by filtering a signal through iterative band-pass.

    This IS the O(k·n) partial spectral transform:
    - theta controls which eigenfrequencies to emphasize
    - theta≈0 → low frequencies (smooth, Fiedler-like)
    - theta≈1 → high frequencies (detail)

    Complexity: O(num_steps × order × nnz(L)) = O(k·n)
    """
    n = L.shape[0]
    device = L.device

    # Ensure signal is (n, 1)
    if signal.dim() == 1:
        signal = signal.unsqueeze(1)

    # Estimate lambda_max (for normalized Laplacian, this is ≤ 2)
    # Quick estimation via power iteration
    v = torch.randn(n, device=device)
    for _ in range(5):
        v = torch.sparse.mm(L, v.unsqueeze(1)).squeeze()
        v = v / (torch.linalg.norm(v) + 1e-10)
    lambda_max = torch.dot(v, torch.sparse.mm(L, v.unsqueeze(1)).squeeze()).item()
    lambda_max = max(lambda_max, 1.0)

    # Iterative spectral filtering
    current = signal.clone()
    target_lambda = theta * lambda_max
    sigma = 0.3 * lambda_max / num_steps  # Band width

    for step in range(num_steps):
        center = (step + 0.5) * theta * lambda_max / num_steps

        # Chebyshev band-pass at this center
        # T_0 = I, T_1 = L_scaled, recurrence
        T_prev = current.clone()
        Ls = torch.sparse.mm(L, current)
        T_curr = current - Ls / lambda_max  # Scaled to [-1, 1]

        # Gaussian weight for this band
        weight = torch.exp(torch.tensor(-(center - target_lambda)**2 / (2 * sigma**2)))

        # Accumulate with low-order Chebyshev
        result = weight * T_prev
        for k in range(1, min(order, 5)):
            LT = torch.sparse.mm(L, T_curr)
            T_next = 2 * (T_curr - LT / lambda_max) - T_prev
            result = result + weight * T_curr * (0.8 ** k)  # Decay for higher orders
            T_prev, T_curr = T_curr, T_next

        current = result
        # Normalize to prevent growth/decay
        current = current / (torch.linalg.norm(current) + 1e-10) * torch.linalg.norm(signal)

    return current.squeeze()


def spectral_embedding(
    L: torch.Tensor,
    num_probes: int = 4,
    cutoff: float = 0.15,
    order: int = 20,
    seed: int = 42
) -> torch.Tensor:
    """
    Compute spectral embedding via filtered random probes.

    For correspondence, we filter the same random probes through different graphs.
    The difference in outputs reveals graph structure differences.
    """
    n = L.shape[0]
    device = L.device

    # Generate reproducible random probes
    torch.manual_seed(seed)
    probes = torch.randn(n, num_probes, device=device, dtype=torch.float32)

    # Project out constant component
    probes = probes - probes.mean(dim=0, keepdim=True)
    probes = probes / (torch.linalg.norm(probes, dim=0, keepdim=True) + 1e-10)

    # Filter each probe
    filtered_list = []
    for i in range(num_probes):
        filtered_list.append(
            spectral_embedding_from_signal(L, probes[:, i], theta=cutoff, num_steps=8, order=order)
        )

    filtered = torch.stack(filtered_list, dim=1)
    filtered = filtered / (torch.linalg.norm(filtered, dim=0, keepdim=True) + 1e-10)

    return filtered


def cross_attention_polynomial(
    image_A: torch.Tensor,  # (H_A, W_A, 3)
    image_B: torch.Tensor,  # (H_B, W_B, 3)
    num_probes: int = 4,
    cutoff: float = 0.15,
    order: int = 20,
    temperature: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cross-image attention using polynomial spectral embeddings.

    Instead of:
      - Compute Fiedler (Lanczos, explicit eigenvector)
      - Bin by Fiedler similarity

    We do:
      - Filter same probes through both Laplacians (O(k·n))
      - Soft attention based on embedding similarity

    This IS the O(k·n) polynomial transform for correspondence.

    Returns:
        sampled_B: (H_A, W_A, 3) - B's colors sampled at A's spectral positions
        attention_summary: (H_A, W_A) - attention entropy for visualization
    """
    device = image_A.device
    H_A, W_A, _ = image_A.shape
    H_B, W_B, _ = image_B.shape
    n_A, n_B = H_A * W_A, H_B * W_B

    # Convert to grayscale for Laplacian construction
    gray_A = 0.299 * image_A[..., 0] + 0.587 * image_A[..., 1] + 0.114 * image_A[..., 2]
    gray_B = 0.299 * image_B[..., 0] + 0.587 * image_B[..., 1] + 0.114 * image_B[..., 2]

    # Build Laplacians
    L_A = build_image_laplacian(gray_A)
    L_B = build_image_laplacian(gray_B)

    # Compute spectral embeddings using SAME random seed
    # This ensures we're filtering the same probes through both graphs
    phi_A = spectral_embedding(L_A, num_probes, cutoff, order, seed=42)  # (n_A, k)
    phi_B = spectral_embedding(L_B, num_probes, cutoff, order, seed=42)  # (n_B, k)

    # Soft attention: how much does each A pixel attend to each B pixel?
    # attention[i, j] = softmax_j(phi_A[i] · phi_B[j] / temperature)
    #
    # For efficiency with large images, we use chunked computation
    # or approximate with locality-sensitive hashing.
    # For now, direct computation (works for moderate image sizes):

    # Similarity matrix: phi_A @ phi_B.T gives (n_A, n_B)
    # This is O(n_A × n_B × k) - expensive for large images
    #
    # Optimization: use sparse attention or binning for large images

    if n_A * n_B > 1e8:  # Large images: fall back to binning
        return _binned_transfer(phi_A, phi_B, image_B, H_A, W_A, H_B, W_B)

    # Direct soft attention
    similarity = torch.mm(phi_A, phi_B.T)  # (n_A, n_B)
    attention = F.softmax(similarity / temperature, dim=1)  # (n_A, n_B)

    # Sample B colors weighted by attention
    colors_B = image_B.reshape(n_B, 3)  # (n_B, 3)
    sampled_colors = torch.mm(attention, colors_B)  # (n_A, 3)

    # Reshape to A's dimensions
    sampled_B = sampled_colors.reshape(H_A, W_A, 3)

    # Attention entropy for visualization (how peaked is attention?)
    entropy = -(attention * (attention + 1e-10).log()).sum(dim=1)
    attention_summary = entropy.reshape(H_A, W_A)

    return sampled_B, attention_summary


def _binned_transfer(
    phi_A: torch.Tensor,  # (n_A, k)
    phi_B: torch.Tensor,  # (n_B, k)
    image_B: torch.Tensor,
    H_A: int, W_A: int,
    H_B: int, W_B: int,
    n_bins: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Binned transfer for large images (O(n) instead of O(n²)).

    Uses first probe dimension as 1D coordinate for binning.
    This is analogous to Fiedler binning but using polynomial-filtered probe.
    """
    device = phi_A.device
    n_A, n_B = phi_A.shape[0], phi_B.shape[0]

    # Use first filtered probe as 1D embedding
    coord_A = phi_A[:, 0]
    coord_B = phi_B[:, 0]

    # Normalize to [0, 1]
    all_coords = torch.cat([coord_A, coord_B])
    c_min, c_max = all_coords.min(), all_coords.max()
    coord_A_norm = (coord_A - c_min) / (c_max - c_min + 1e-8)
    coord_B_norm = (coord_B - c_min) / (c_max - c_min + 1e-8)

    # Bin B pixels
    bin_B = (coord_B_norm * (n_bins - 1)).long().clamp(0, n_bins - 1)

    # Build lookup: mean (y, x) per bin
    y_B = torch.arange(H_B, device=device).repeat_interleave(W_B).float()
    x_B = torch.arange(W_B, device=device).repeat(H_B).float()

    y_sum = torch.zeros(n_bins, device=device)
    x_sum = torch.zeros(n_bins, device=device)
    count = torch.zeros(n_bins, device=device)

    y_sum.scatter_add_(0, bin_B, y_B)
    x_sum.scatter_add_(0, bin_B, x_B)
    count.scatter_add_(0, bin_B, torch.ones(n_B, device=device))

    # Mean position per bin
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

    # Look up for each A pixel
    bin_A = (coord_A_norm * (n_bins - 1)).long().clamp(0, n_bins - 1)
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
    sampled_B = sampled.squeeze(0).permute(1, 2, 0)

    # Bin density as attention summary
    attention_summary = count[bin_A].reshape(H_A, W_A).float()
    attention_summary = attention_summary / (attention_summary.max() + 1e-8)

    return sampled_B, attention_summary


# ============================================================
# Single-image spectral effects using polynomial transform
# ============================================================

def spectral_gate_polynomial(
    image_rgb: torch.Tensor,
    cutoff: float = 0.15,
    order: int = 20,
    threshold_percentile: float = 40.0,
    sharpness: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spectral gate using polynomial-filtered probe instead of Fiedler.

    Returns:
        gate: (H, W) soft mask in [0, 1]
        spectral_field: (H, W) the filtered probe (analogous to Fiedler)
    """
    H, W = image_rgb.shape[:2]
    device = image_rgb.device

    gray = 0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1] + 0.114 * image_rgb[..., 2]
    L = build_image_laplacian(gray)

    # Get spectral embedding (single probe = Fiedler approximation)
    phi = spectral_embedding(L, num_probes=1, cutoff=cutoff, order=order)
    spectral_field = phi[:, 0].reshape(H, W)

    # Adaptive threshold
    flat = spectral_field.flatten()
    k = int(len(flat) * threshold_percentile / 100.0)
    threshold = torch.kthvalue(flat, max(1, min(k, len(flat))))[0]

    # Sigmoid gate
    gate = torch.sigmoid((spectral_field - threshold) * sharpness)

    return gate, spectral_field


def spectral_gate_direct(
    image_rgb: torch.Tensor,
    theta: float = 0.1,
    num_steps: int = 8,
    order: int = 10,
    threshold_percentile: float = 40.0,
    sharpness: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spectral gate by filtering the IMAGE ITSELF through spectral transform.

    This is the direct O(k·n) approach:
    - Filter grayscale through graph Laplacian at low theta
    - The filtered image is smooth within regions, has gradients at edges
    - Use this as the gating field

    No eigenvector computation. Just apply the iterative spectral transform.
    """
    H, W = image_rgb.shape[:2]
    device = image_rgb.device

    gray = 0.299 * image_rgb[..., 0] + 0.587 * image_rgb[..., 1] + 0.114 * image_rgb[..., 2]
    L = build_image_laplacian(gray)

    # THE KEY OPERATION: filter the grayscale through the spectral transform
    # theta=0.1 emphasizes low frequencies → smooth, structure-aware field
    spectral_field = spectral_embedding_from_signal(
        L, gray.flatten(), theta=theta, num_steps=num_steps, order=order
    ).reshape(H, W)

    # Adaptive threshold
    flat = spectral_field.flatten()
    k = int(len(flat) * threshold_percentile / 100.0)
    threshold = torch.kthvalue(flat, max(1, min(k, len(flat))))[0]

    # Sigmoid gate
    gate = torch.sigmoid((spectral_field - threshold) * sharpness)

    return gate, spectral_field


def spectral_gradient_polynomial(
    image_rgb: torch.Tensor,
    cutoff: float = 0.15,
    order: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spectral gradient direction using polynomial-filtered probe.

    Returns:
        grad_x, grad_y: (H, W) gradient components (tangent direction)
    """
    H, W = image_rgb.shape[:2]
    device = image_rgb.device
    dtype = image_rgb.dtype

    _, spectral_field = spectral_gate_polynomial(image_rgb, cutoff, order)

    # Sobel gradient of spectral field
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           device=device, dtype=dtype).view(1, 1, 3, 3) / 8.0

    f_4d = spectral_field.unsqueeze(0).unsqueeze(0)
    f_padded = F.pad(f_4d, (1, 1, 1, 1), mode='reflect')

    grad_x = F.conv2d(f_padded, sobel_x).squeeze()
    grad_y = F.conv2d(f_padded, sobel_y).squeeze()

    return grad_x, grad_y


# ============================================================
# Demo / test
# ============================================================

if __name__ == "__main__":
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Generate test images
    H, W = 256, 256
    torch.manual_seed(123)

    # Image A: gradient + circles
    y, x = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing='ij'
    )
    image_A = torch.stack([x, y, 0.5 * torch.ones_like(x)], dim=-1)
    circle_A = ((x - 0.3)**2 + (y - 0.3)**2 < 0.04).float()
    image_A = image_A * (1 - circle_A.unsqueeze(-1)) + circle_A.unsqueeze(-1) * 0.8

    # Image B: different pattern
    image_B = torch.stack([1-x, y, torch.abs(x - 0.5)], dim=-1)
    circle_B = ((x - 0.7)**2 + (y - 0.6)**2 < 0.05).float()
    image_B = image_B * (1 - circle_B.unsqueeze(-1)) + circle_B.unsqueeze(-1) * 0.2

    print(f"Image A: {image_A.shape}")
    print(f"Image B: {image_B.shape}")

    # Test spectral gate (probe-based)
    print("\n--- Spectral Gate (probe-filtered) ---")
    start = time.perf_counter()
    gate, spec_field = spectral_gate_polynomial(image_A)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f"Time: {(time.perf_counter() - start)*1000:.1f}ms")
    print(f"Gate range: [{gate.min():.3f}, {gate.max():.3f}]")
    print(f"Spectral field range: [{spec_field.min():.3f}, {spec_field.max():.3f}]")

    # Test spectral gate (direct image filtering)
    print("\n--- Spectral Gate (direct image filter) ---")
    start = time.perf_counter()
    gate_direct, spec_direct = spectral_gate_direct(image_A)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f"Time: {(time.perf_counter() - start)*1000:.1f}ms")
    print(f"Gate range: [{gate_direct.min():.3f}, {gate_direct.max():.3f}]")
    print(f"Spectral field range: [{spec_direct.min():.3f}, {spec_direct.max():.3f}]")

    # Test cross-attention
    print("\n--- Cross-Attention (polynomial) ---")
    start = time.perf_counter()
    sampled_B, attn_summary = cross_attention_polynomial(image_A, image_B, num_probes=4)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f"Time: {(time.perf_counter() - start)*1000:.1f}ms")
    print(f"Sampled B range: [{sampled_B.min():.3f}, {sampled_B.max():.3f}]")
    print(f"Attention entropy range: [{attn_summary.min():.3f}, {attn_summary.max():.3f}]")

    # Compare with Lanczos Fiedler
    print("\n--- Comparison with Lanczos Fiedler ---")
    try:
        from spectral_ops_fast_cuter import build_weighted_image_laplacian, lanczos_fiedler_gpu

        gray_A = 0.299 * image_A[..., 0] + 0.587 * image_A[..., 1] + 0.114 * image_A[..., 2]
        L = build_weighted_image_laplacian(gray_A)

        start = time.perf_counter()
        fiedler_lanczos, _ = lanczos_fiedler_gpu(L, num_iterations=30)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        lanczos_time = time.perf_counter() - start

        fiedler_lanczos = fiedler_lanczos.reshape(H, W)
        lanc_flat = fiedler_lanczos.flatten()
        lanc_centered = lanc_flat - lanc_flat.mean()

        # Correlation: probe-filtered vs Lanczos
        poly_flat = spec_field.flatten()
        poly_centered = poly_flat - poly_flat.mean()
        corr_probe = (poly_centered * lanc_centered).sum() / (
            torch.sqrt((poly_centered**2).sum() * (lanc_centered**2).sum()) + 1e-10
        )

        # Correlation: direct-filtered vs Lanczos
        direct_flat = spec_direct.flatten()
        direct_centered = direct_flat - direct_flat.mean()
        corr_direct = (direct_centered * lanc_centered).sum() / (
            torch.sqrt((direct_centered**2).sum() * (lanc_centered**2).sum()) + 1e-10
        )

        print(f"Lanczos time: {lanczos_time*1000:.1f}ms")
        print(f"Correlation (probe-filtered vs Lanczos): {abs(corr_probe.item()):.3f}")
        print(f"Correlation (direct-image-filter vs Lanczos): {abs(corr_direct.item()):.3f}")

        # Key insight: does direct filter correlate with grayscale?
        gray_flat = gray_A.flatten()
        gray_centered = gray_flat - gray_flat.mean()
        corr_gray_direct = (direct_centered * gray_centered).sum() / (
            torch.sqrt((direct_centered**2).sum() * (gray_centered**2).sum()) + 1e-10
        )
        corr_gray_fiedler = (lanc_centered * gray_centered).sum() / (
            torch.sqrt((lanc_centered**2).sum() * (gray_centered**2).sum()) + 1e-10
        )
        print(f"Correlation (direct-filter vs grayscale): {abs(corr_gray_direct.item()):.3f}")
        print(f"Correlation (Fiedler vs grayscale): {abs(corr_gray_fiedler.item()):.3f}")

    except ImportError as e:
        print(f"Could not import Lanczos: {e}")

    print("\n✓ Unified polynomial spectral correspondence complete")
