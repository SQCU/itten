"""
Dither-aware connectivity: multi-scale graph construction.

Standard 4-connected graphs see dither patterns as isolated dots.
By building edges at multiple radii (or dither-frequency-aware edges),
we can make dither patterns spectrally connected.

Key insight: the Laplacian determines what "connected" means.
"""
import torch
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional


def build_radius_edges(
    H: int, W: int,
    radius: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge indices for all pixel pairs within given radius.

    Returns:
        src_indices: (E,) source pixel indices (flattened)
        dst_indices: (E,) destination pixel indices (flattened)
    """
    # Generate offset pairs within radius
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            if dx * dx + dy * dy <= radius * radius:
                offsets.append((dy, dx))

    # Build edges for all pixels
    src_list = []
    dst_list = []

    for dy, dx in offsets:
        # Valid source pixel range
        y_start = max(0, -dy)
        y_end = min(H, H - dy)
        x_start = max(0, -dx)
        x_end = min(W, W - dx)

        # Source pixels
        y_src = torch.arange(y_start, y_end, device=device)
        x_src = torch.arange(x_start, x_end, device=device)
        yy, xx = torch.meshgrid(y_src, x_src, indexing='ij')

        src_flat = yy.flatten() * W + xx.flatten()
        dst_flat = (yy.flatten() + dy) * W + (xx.flatten() + dx)

        src_list.append(src_flat)
        dst_list.append(dst_flat)

    src_indices = torch.cat(src_list)
    dst_indices = torch.cat(dst_list)

    return src_indices, dst_indices


def build_multiscale_laplacian(
    image: torch.Tensor,          # (H, W) grayscale in [0, 1]
    radii: List[int] = [1, 2, 4],
    radius_weights: List[float] = [1.0, 0.3, 0.1],
    intensity_threshold: float = 0.15
) -> torch.Tensor:
    """
    Build Laplacian with edges at multiple radii.

    Edge weight = radius_weight * exp(-|intensity_diff| / threshold)

    This connects dither patterns that have spacing matching the radii.
    """
    H, W = image.shape
    N = H * W
    device = image.device

    image_flat = image.flatten()

    all_src = []
    all_dst = []
    all_weights = []

    for radius, r_weight in zip(radii, radius_weights):
        src, dst = build_radius_edges(H, W, radius, device)

        # Intensity-based weight
        intensity_diff = torch.abs(image_flat[src] - image_flat[dst])
        edge_weight = r_weight * torch.exp(-intensity_diff / intensity_threshold)

        all_src.append(src)
        all_dst.append(dst)
        all_weights.append(edge_weight)

    src_indices = torch.cat(all_src)
    dst_indices = torch.cat(all_dst)
    weights = torch.cat(all_weights)

    # Build sparse adjacency
    indices = torch.stack([src_indices, dst_indices])
    adj = torch.sparse_coo_tensor(indices, weights, (N, N)).coalesce()

    # Laplacian: L = D - A
    degrees = torch.zeros(N, device=device)
    degrees.scatter_add_(0, src_indices, weights)

    # Off-diagonal: -A
    L_indices = torch.cat([indices, torch.stack([torch.arange(N, device=device)] * 2)], dim=1)
    L_values = torch.cat([-weights, degrees])

    L = torch.sparse_coo_tensor(L_indices, L_values, (N, N)).coalesce()

    return L


def compute_local_dither_frequency(
    image: torch.Tensor,
    patch_size: int = 16
) -> torch.Tensor:
    """
    Estimate local dither frequency via autocorrelation peaks.

    Returns (H, W) field of dominant local frequencies.
    """
    H, W = image.shape
    device = image.device

    freq_field = torch.zeros(H, W, device=device)

    # Simple approach: for each patch, find autocorrelation peaks
    # Peak spacing indicates dither frequency

    half = patch_size // 2

    for y in range(half, H - half, half):
        for x in range(half, W - half, half):
            patch = image[y-half:y+half, x-half:x+half]

            # Autocorrelation via FFT
            f = torch.fft.fft2(patch - patch.mean())
            power = torch.abs(f) ** 2
            autocorr = torch.fft.ifft2(power).real

            # Find first non-central peak (indicates periodicity)
            autocorr[half-2:half+3, half-2:half+3] = 0  # mask center

            peak_val = autocorr.max()
            if peak_val > 0.1 * autocorr.sum():
                peak_idx = (autocorr == peak_val).nonzero()[0]
                freq = torch.sqrt((peak_idx[0] - half)**2 + (peak_idx[1] - half)**2)
                freq_field[y-half:y+half, x-half:x+half] = freq

    return freq_field


def eigenvectors_sparse_lanczos(
    L: torch.Tensor,
    k: int = 10,
    num_iter: int = 50
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple Lanczos for sparse Laplacian."""
    N = L.shape[0]
    device = L.device

    # Random start
    torch.manual_seed(42)
    v = torch.randn(N, device=device)
    v = v - v.mean()
    v = v / torch.norm(v)

    # Lanczos iteration
    V = torch.zeros(N, k + 1, device=device)
    V[:, 0] = v

    alphas = torch.zeros(k, device=device)
    betas = torch.zeros(k, device=device)

    for i in range(min(k, num_iter)):
        w = torch.sparse.mm(L, V[:, i:i+1]).squeeze()

        alpha = torch.dot(V[:, i], w)
        alphas[i] = alpha

        w = w - alpha * V[:, i]
        if i > 0:
            w = w - betas[i-1] * V[:, i-1]

        # Reorthogonalize
        w = w - V[:, :i+1] @ (V[:, :i+1].T @ w)
        w = w - w.mean()

        beta = torch.norm(w)
        if beta < 1e-10:
            break

        betas[i] = beta
        if i + 1 < k + 1:
            V[:, i+1] = w / beta

    # Solve tridiagonal eigenproblem
    T = torch.diag(alphas[:k])
    for i in range(k-1):
        T[i, i+1] = betas[i]
        T[i+1, i] = betas[i]

    eigenvalues, eigenvecs_T = torch.linalg.eigh(T)
    eigenvectors = V[:, :k] @ eigenvecs_T

    return eigenvalues, eigenvectors


def demo_dither_connectivity():
    """
    Compare standard vs multi-scale connectivity on dithered image.
    """
    print("=" * 60)
    print("DITHER CONNECTIVITY DEMO")
    print("=" * 60)

    device = torch.device('cpu')

    # Load the dithered image - CROP a dither-rich region, don't downsample!
    # Downsampling destroys dither structure!
    img_path = "demo_output/inputs/1bit redraw.png"
    img_pil = Image.open(img_path).convert('L')

    # Crop to a region with dither patterns (the hair/top area)
    # This preserves dither structure while keeping computation tractable
    crop_box = (50, 20, 250, 150)  # (left, top, right, bottom) - 200x130 region
    img_pil = img_pil.crop(crop_box)
    scale = 1  # no scaling, native resolution

    img = torch.tensor(np.array(img_pil), dtype=torch.float32, device=device) / 255.0
    H, W = img.shape
    N = H * W

    print(f"\nImage: {W}x{H} = {N} pixels (downsampled {scale}x)")

    # Method 1: Standard 4-connected Laplacian
    print("\n--- Method 1: Standard 4-connected ---")
    L_standard = build_multiscale_laplacian(img, radii=[1], radius_weights=[1.0])
    eigenvalues_std, eigenvectors_std = eigenvectors_sparse_lanczos(L_standard, k=10)

    n_components_std = (eigenvalues_std.abs() < 1e-5).sum().item()
    print(f"Eigenvalues near 0: {n_components_std}")
    print(f"First 6 eigenvalues: {eigenvalues_std[:6].tolist()}")

    # Method 2: Multi-scale (radii 1, 2, 4)
    print("\n--- Method 2: Multi-scale (radii 1, 2, 4) ---")
    L_multi = build_multiscale_laplacian(img, radii=[1, 2, 4], radius_weights=[1.0, 0.5, 0.2])
    eigenvalues_multi, eigenvectors_multi = eigenvectors_sparse_lanczos(L_multi, k=10)

    n_components_multi = (eigenvalues_multi.abs() < 1e-5).sum().item()
    print(f"Eigenvalues near 0: {n_components_multi}")
    print(f"First 6 eigenvalues: {eigenvalues_multi[:6].tolist()}")

    # Method 3: Dither-scale (radii 1, 2, 3, 4, 5, 6) - finer coverage
    print("\n--- Method 3: Dither-scale (radii 1-6) ---")
    L_dither = build_multiscale_laplacian(
        img,
        radii=[1, 2, 3, 4, 5, 6],
        radius_weights=[1.0, 0.6, 0.4, 0.3, 0.2, 0.1]
    )
    eigenvalues_dither, eigenvectors_dither = eigenvectors_sparse_lanczos(L_dither, k=10)

    n_components_dither = (eigenvalues_dither.abs() < 1e-5).sum().item()
    print(f"Eigenvalues near 0: {n_components_dither}")
    print(f"First 6 eigenvalues: {eigenvalues_dither[:6].tolist()}")

    # Visualize Fiedler vectors (2nd eigenvector shows main partition)
    print("\nVisualizing Fiedler vectors (2nd eigenvector)...")

    visualize_fiedler_comparison(
        img, H, W,
        eigenvectors_std[:, 1].reshape(H, W),
        eigenvectors_multi[:, 1].reshape(H, W),
        eigenvectors_dither[:, 1].reshape(H, W),
        ["Standard (r=1)", "Multi-scale (r=1,2,4)", "Dither-scale (r=1-6)"]
    )

    return img, eigenvalues_std, eigenvalues_multi, eigenvalues_dither


def visualize_fiedler_comparison(img, H, W, fiedler1, fiedler2, fiedler3, labels):
    """Compare Fiedler vectors from different Laplacians."""
    from PIL import ImageDraw

    def to_image(field):
        # Normalize to [0, 255]
        f = field.numpy()
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        return (f * 255).astype(np.uint8)

    img_np = (img.numpy() * 255).astype(np.uint8)

    # Create comparison grid
    margin = 5
    label_h = 20
    panel_w, panel_h = W, H

    total_w = 4 * panel_w + 5 * margin
    total_h = panel_h + 2 * margin + label_h

    canvas = Image.new('RGB', (total_w, total_h), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    # Original image
    img_pil = Image.fromarray(img_np).convert('RGB')
    canvas.paste(img_pil, (margin, margin + label_h))
    draw.text((margin + 2, 2), "Original", fill=(255, 255, 255))

    # Fiedler vectors as colormaps
    fields = [fiedler1, fiedler2, fiedler3]
    for i, (field, label) in enumerate(zip(fields, labels)):
        field_img = to_image(field)

        # Apply colormap (blue-white-red)
        field_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        field_norm = field_img / 255.0

        # Blue for negative, red for positive (relative to median)
        median = 0.5
        field_rgb[:, :, 0] = (np.clip(field_norm - median, 0, 0.5) * 2 * 255).astype(np.uint8)  # red
        field_rgb[:, :, 2] = (np.clip(median - field_norm, 0, 0.5) * 2 * 255).astype(np.uint8)  # blue
        field_rgb[:, :, 1] = 255 - np.abs(field_img.astype(np.int16) - 128).astype(np.uint8)  # green at middle

        pil_field = Image.fromarray(field_rgb)
        x_pos = (i + 1) * (panel_w + margin) + margin
        canvas.paste(pil_field, (x_pos, margin + label_h))
        draw.text((x_pos + 2, 2), label, fill=(255, 255, 255))

    canvas.save('demo_output/dither_connectivity.png')
    print(f"\nSaved: demo_output/dither_connectivity.png")


if __name__ == "__main__":
    demo_dither_connectivity()
