"""
True carrier-operand spectral blend transforms.

Unlike self-convolution transforms, these functions take TWO textures:
- Carrier: provides spatial structure (WHERE things go)
- Operand: provides pattern (WHAT gets embedded)

The result shows BOTH signatures inextricably combined.
Theta controls the blend depth / spectral coupling strength.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from typing import Tuple, Optional
from scipy.ndimage import map_coordinates

from spectral_ops_fast import (
    build_weighted_image_laplacian,
    lanczos_k_eigenvectors,
    DEVICE
)


def spectral_embed(
    carrier: np.ndarray,
    operand: np.ndarray,
    theta: float = 0.5,
    num_eigenvectors: int = 16,
    edge_threshold: float = 0.1,
    num_iterations: int = 50
) -> np.ndarray:
    """
    Embed operand into carrier's spectral basis.

    The operand is projected onto the carrier's eigenvector basis, then
    reconstructed with weights that blend between operand-dominant and
    carrier-dominant representations.

    Args:
        carrier: (H, W) grayscale carrier image providing spatial structure
        operand: (H, W) grayscale operand image providing pattern to embed
        theta: Blend parameter [0, 1]
               theta=0: operand dominates (carrier structure faint)
               theta=0.5: balanced blend
               theta=1: carrier dominates (operand trace faint)
        num_eigenvectors: Number of eigenvectors for spectral basis
        edge_threshold: Carrier edge sensitivity for Laplacian weighting
        num_iterations: Lanczos iterations for eigenvector computation

    Returns:
        (H, W) blended result showing both carrier and operand signatures
    """
    H, W = carrier.shape

    # Ensure same size
    if operand.shape != carrier.shape:
        raise ValueError(f"Carrier {carrier.shape} and operand {operand.shape} must have same shape")

    # Normalize inputs to [0, 1]
    carrier_norm = carrier.astype(np.float32)
    if carrier_norm.max() > 1.0:
        carrier_norm = carrier_norm / 255.0

    operand_norm = operand.astype(np.float32)
    if operand_norm.max() > 1.0:
        operand_norm = operand_norm / 255.0

    # Build carrier's spectral representation
    carrier_t = torch.tensor(carrier_norm, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier_t, edge_threshold)

    # Compute carrier's eigenvectors
    eigenvectors, eigenvalues = lanczos_k_eigenvectors(
        L, num_eigenvectors, num_iterations
    )

    # Convert to tensor for operations
    evs = torch.tensor(eigenvectors, dtype=torch.float32, device=DEVICE)
    operand_flat = torch.tensor(operand_norm.flatten(), dtype=torch.float32, device=DEVICE)
    carrier_flat = torch.tensor(carrier_norm.flatten(), dtype=torch.float32, device=DEVICE)

    # Project operand onto carrier's eigenvectors: coeffs = V^T @ operand
    operand_coeffs = torch.mm(evs.T, operand_flat.unsqueeze(1)).squeeze()

    # Project carrier onto its own eigenvectors: coeffs = V^T @ carrier
    carrier_coeffs = torch.mm(evs.T, carrier_flat.unsqueeze(1)).squeeze()

    # Create frequency-dependent blending weights
    # Low frequencies (small eigenvalues) -> carry more carrier structure
    # High frequencies (large eigenvalues) -> carry more operand pattern
    k = len(eigenvalues)
    freq_weights = torch.linspace(0, 1, k, device=DEVICE)  # 0 at low freq, 1 at high freq

    # Blend coefficients based on theta and frequency
    # theta=0: favor operand (all frequencies from operand)
    # theta=1: favor carrier (all frequencies from carrier)
    # Intermediate: low freqs from carrier, high freqs from operand, modulated by theta

    # Carrier weight increases with theta and at low frequencies
    carrier_weight = theta + (1 - theta) * (1 - freq_weights) * 0.5
    # Operand weight is complement
    operand_weight = 1 - carrier_weight

    blended_coeffs = carrier_weight * carrier_coeffs + operand_weight * operand_coeffs

    # Reconstruct from blended coefficients
    result_flat = torch.mm(evs, blended_coeffs.unsqueeze(1)).squeeze()

    # Add residual from operand that wasn't captured by eigenvectors
    # This preserves high-frequency operand details not in the k eigenvectors
    operand_reconstructed = torch.mm(evs, operand_coeffs.unsqueeze(1)).squeeze()
    operand_residual = operand_flat - operand_reconstructed

    # Add residual weighted by (1 - theta) to preserve operand detail
    result_flat = result_flat + (1 - theta) * operand_residual

    result = result_flat.cpu().numpy().reshape(H, W)

    # Normalize to [0, 1]
    result_min, result_max = result.min(), result.max()
    if result_max > result_min:
        result = (result - result_min) / (result_max - result_min)

    return result


def spectral_warp_embed(
    carrier: np.ndarray,
    operand: np.ndarray,
    theta: float = 0.5,
    num_eigenvectors: int = 8,
    warp_scale: float = 5.0,
    edge_threshold: float = 0.1,
    num_iterations: int = 50
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Warp operand pattern along carrier's eigenvector gradients.

    The operand pattern deforms along the carrier's spectral flow,
    creating visible warping that respects carrier structure.

    Args:
        carrier: (H, W) grayscale carrier image providing warp field
        operand: (H, W) grayscale operand image to be warped
        theta: Controls warp strength and blend [0, 1]
               theta=0: minimal warp, operand nearly unchanged
               theta=0.5: moderate warp, visible deformation
               theta=1: maximum warp along carrier gradients
        num_eigenvectors: Number of eigenvectors for warp field
        warp_scale: Base warp displacement magnitude
        edge_threshold: Carrier edge sensitivity for Laplacian weighting
        num_iterations: Lanczos iterations for eigenvector computation

    Returns:
        result: (H, W) warped and blended result
        warp_field: Tuple of (warp_x, warp_y) displacement fields
    """
    H, W = carrier.shape

    # Ensure same size
    if operand.shape != carrier.shape:
        raise ValueError(f"Carrier {carrier.shape} and operand {operand.shape} must have same shape")

    # Normalize inputs to [0, 1]
    carrier_norm = carrier.astype(np.float32)
    if carrier_norm.max() > 1.0:
        carrier_norm = carrier_norm / 255.0

    operand_norm = operand.astype(np.float32)
    if operand_norm.max() > 1.0:
        operand_norm = operand_norm / 255.0

    # Build carrier's spectral representation
    carrier_t = torch.tensor(carrier_norm, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier_t, edge_threshold)

    # Compute carrier's eigenvectors
    eigenvectors, eigenvalues = lanczos_k_eigenvectors(
        L, num_eigenvectors, num_iterations
    )

    # Compute warp field from multiple eigenvector gradients
    warp_x = np.zeros((H, W), dtype=np.float32)
    warp_y = np.zeros((H, W), dtype=np.float32)

    for k in range(min(num_eigenvectors, eigenvectors.shape[1])):
        ev = eigenvectors[:, k].reshape(H, W)

        # Gradient of eigenvector
        grad_y = np.gradient(ev, axis=0)
        grad_x = np.gradient(ev, axis=1)

        # Weight by eigenvalue (lower = more structural = more influence)
        # Use inverse square root for decay
        if eigenvalues[k] > 1e-6:
            weight = 1.0 / np.sqrt(eigenvalues[k])
        else:
            weight = 1.0
        weight = min(weight, 10.0)  # Cap weight

        # Accumulate warp: perpendicular to gradient for flow-along effect
        # This creates warping that follows carrier contours
        warp_x += weight * (-grad_y)  # Perpendicular
        warp_y += weight * grad_x

    # Normalize warp field
    warp_mag = np.sqrt(warp_x**2 + warp_y**2)
    max_mag = warp_mag.max()
    if max_mag > 0:
        warp_x = warp_x / max_mag
        warp_y = warp_y / max_mag

    # Scale by theta and warp_scale
    warp_x = warp_x * theta * warp_scale
    warp_y = warp_y * theta * warp_scale

    # Create sampling coordinates
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    warped_x = np.clip(x_coords + warp_x, 0, W - 1).astype(np.float32)
    warped_y = np.clip(y_coords + warp_y, 0, H - 1).astype(np.float32)

    # Warp the operand
    warped_operand = map_coordinates(operand_norm, [warped_y, warped_x], order=1, mode='reflect')

    # Blend with carrier based on theta
    # theta=0: pure operand (unwarped)
    # theta=1: heavily warped operand mixed with carrier structure
    carrier_blend = theta * 0.3  # Carrier contributes up to 30% at theta=1
    result = (1 - carrier_blend) * warped_operand + carrier_blend * carrier_norm

    # Normalize result
    result_min, result_max = result.min(), result.max()
    if result_max > result_min:
        result = (result - result_min) / (result_max - result_min)

    return result, (warp_x, warp_y)


def compute_spatial_autocorrelation(
    image: np.ndarray,
    lags: list = [1, 2, 4, 8, 16]
) -> dict:
    """
    Compute spatial autocorrelation at multiple lags.

    This measures how correlated a pixel is with its neighbors at
    various distances. The structure of autocorrelation reveals
    the underlying spatial patterns.

    Args:
        image: (H, W) grayscale image
        lags: List of pixel distances to measure

    Returns:
        Dictionary mapping lag -> (horizontal_corr, vertical_corr, diagonal_corr)
    """
    H, W = image.shape
    img = image.astype(np.float64)

    # Normalize
    img = (img - img.mean()) / (img.std() + 1e-8)

    result = {}

    for lag in lags:
        if lag >= min(H, W):
            result[lag] = (0.0, 0.0, 0.0)
            continue

        # Horizontal autocorrelation
        if W > lag:
            h_corr = np.mean(img[:, :-lag] * img[:, lag:])
        else:
            h_corr = 0.0

        # Vertical autocorrelation
        if H > lag:
            v_corr = np.mean(img[:-lag, :] * img[lag:, :])
        else:
            v_corr = 0.0

        # Diagonal autocorrelation
        if H > lag and W > lag:
            d_corr = np.mean(img[:-lag, :-lag] * img[lag:, lag:])
        else:
            d_corr = 0.0

        result[lag] = (float(h_corr), float(v_corr), float(d_corr))

    return result


def generate_checkerboard(size: int = 64, block_size: int = 8) -> np.ndarray:
    """Generate a checkerboard pattern."""
    pattern = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                pattern[i, j] = 1.0
    return pattern


def generate_noise(size: int = 64, seed: int = 42) -> np.ndarray:
    """Generate uniform noise pattern."""
    np.random.seed(seed)
    return np.random.rand(size, size).astype(np.float32)


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def save_image_as_raw(image: np.ndarray, filepath: str):
    """Save image as raw grayscale bytes for inspection."""
    # Scale to 0-255 and save as uint8
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    np.save(filepath, img_uint8)


if __name__ == '__main__':
    from PIL import Image

    print("=" * 60)
    print("TRUE CARRIER-OPERAND SPECTRAL BLEND TEST")
    print("=" * 60)

    # Test configuration
    size = 64
    theta_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    lags = [1, 2, 4, 8, 16]
    output_dir = '/home/bigboi/itten/demo_output/carrier_operand'

    # Generate test patterns
    carrier = generate_checkerboard(size, block_size=8)
    operand = generate_noise(size, seed=42)

    print(f"\nCarrier: {size}x{size} checkerboard")
    print(f"Operand: {size}x{size} noise")
    print(f"Theta values: {theta_values}")
    print(f"Lags for autocorrelation: {lags}")

    # Save carrier and operand as PNG
    Image.fromarray((carrier * 255).astype(np.uint8)).save(f'{output_dir}/carrier.png')
    Image.fromarray((operand * 255).astype(np.uint8)).save(f'{output_dir}/operand.png')
    print(f"\nSaved carrier and operand to {output_dir}/")

    # Store results
    results = {
        'spectral_embed': {},
        'spectral_warp_embed': {}
    }

    # ===== Test spectral_embed =====
    print("\n" + "-" * 40)
    print("Testing spectral_embed...")
    print("-" * 40)

    embed_results = []
    for i, theta in enumerate(theta_values):
        print(f"  theta={theta}...", end=" ")
        result = spectral_embed(carrier, operand, theta=theta)
        embed_results.append(result)

        # Save result as PNG
        Image.fromarray((result * 255).astype(np.uint8)).save(
            f'{output_dir}/spectral_embed_theta_{theta:.1f}.png'
        )

        # Compute metrics
        autocorr = compute_spatial_autocorrelation(result, lags)
        psnr_carrier = psnr(result, carrier)
        psnr_operand = psnr(result, operand)

        results['spectral_embed'][theta] = {
            'autocorrelation': autocorr,
            'psnr_carrier': psnr_carrier,
            'psnr_operand': psnr_operand
        }

        print(f"PSNR(carrier)={psnr_carrier:.2f}dB, PSNR(operand)={psnr_operand:.2f}dB")

    # ===== Test spectral_warp_embed =====
    print("\n" + "-" * 40)
    print("Testing spectral_warp_embed...")
    print("-" * 40)

    warp_results = []
    for i, theta in enumerate(theta_values):
        print(f"  theta={theta}...", end=" ")
        result, (warp_x, warp_y) = spectral_warp_embed(carrier, operand, theta=theta)
        warp_results.append(result)

        # Save result as PNG
        Image.fromarray((result * 255).astype(np.uint8)).save(
            f'{output_dir}/spectral_warp_embed_theta_{theta:.1f}.png'
        )

        # Save warp magnitude
        warp_mag = np.sqrt(warp_x**2 + warp_y**2)
        warp_mag_norm = warp_mag / (warp_mag.max() + 1e-8)
        Image.fromarray((warp_mag_norm * 255).astype(np.uint8)).save(
            f'{output_dir}/warp_field_theta_{theta:.1f}.png'
        )

        # Compute metrics
        autocorr = compute_spatial_autocorrelation(result, lags)
        psnr_carrier = psnr(result, carrier)
        psnr_operand = psnr(result, operand)

        results['spectral_warp_embed'][theta] = {
            'autocorrelation': autocorr,
            'psnr_carrier': psnr_carrier,
            'psnr_operand': psnr_operand,
            'warp_magnitude': warp_mag.mean()
        }

        print(f"PSNR(carrier)={psnr_carrier:.2f}dB, PSNR(operand)={psnr_operand:.2f}dB, warp_mag={results['spectral_warp_embed'][theta]['warp_magnitude']:.2f}")

    # ===== Autocorrelation structure analysis =====
    print("\n" + "-" * 40)
    print("Autocorrelation Structure Analysis")
    print("-" * 40)

    # Compute autocorrelation for carrier and operand for reference
    carrier_autocorr = compute_spatial_autocorrelation(carrier, lags)
    operand_autocorr = compute_spatial_autocorrelation(operand, lags)

    print("\nCarrier (checkerboard) autocorrelation:")
    for lag in lags:
        h, v, d = carrier_autocorr[lag]
        print(f"  lag={lag:2d}: H={h:+.3f}, V={v:+.3f}, D={d:+.3f}")

    print("\nOperand (noise) autocorrelation:")
    for lag in lags:
        h, v, d = operand_autocorr[lag]
        print(f"  lag={lag:2d}: H={h:+.3f}, V={v:+.3f}, D={d:+.3f}")

    print("\nspectral_embed autocorrelation by theta:")
    for theta in theta_values:
        print(f"  theta={theta}:")
        autocorr = results['spectral_embed'][theta]['autocorrelation']
        for lag in lags:
            h, v, d = autocorr[lag]
            print(f"    lag={lag:2d}: H={h:+.3f}, V={v:+.3f}, D={d:+.3f}")

    print("\nspectral_warp_embed autocorrelation by theta:")
    for theta in theta_values:
        print(f"  theta={theta}:")
        autocorr = results['spectral_warp_embed'][theta]['autocorrelation']
        for lag in lags:
            h, v, d = autocorr[lag]
            print(f"    lag={lag:2d}: H={h:+.3f}, V={v:+.3f}, D={d:+.3f}")

    # ===== PSNR Summary =====
    print("\n" + "-" * 40)
    print("PSNR Summary")
    print("-" * 40)

    print("\nspectral_embed PSNR trends:")
    print("  theta  PSNR(carrier)  PSNR(operand)")
    for theta in theta_values:
        pc = results['spectral_embed'][theta]['psnr_carrier']
        po = results['spectral_embed'][theta]['psnr_operand']
        print(f"  {theta:.1f}    {pc:6.2f}         {po:6.2f}")

    print("\nspectral_warp_embed PSNR trends:")
    print("  theta  PSNR(carrier)  PSNR(operand)  warp_mag")
    for theta in theta_values:
        pc = results['spectral_warp_embed'][theta]['psnr_carrier']
        po = results['spectral_warp_embed'][theta]['psnr_operand']
        wm = results['spectral_warp_embed'][theta]['warp_magnitude']
        print(f"  {theta:.1f}    {pc:6.2f}         {po:6.2f}      {wm:5.2f}")

    # ===== Create composite image =====
    # Create a grid showing all results
    print("\n" + "-" * 40)
    print("Creating composite visualization...")
    print("-" * 40)

    # Composite for spectral_embed
    n_theta = len(theta_values)
    composite_embed = np.zeros((size * 2, size * (n_theta + 1)), dtype=np.float32)

    # First column: carrier and operand
    composite_embed[:size, :size] = carrier
    composite_embed[size:, :size] = operand

    # Remaining columns: results for each theta
    for i, (theta, result) in enumerate(zip(theta_values, embed_results)):
        composite_embed[:size, size * (i + 1):size * (i + 2)] = result

    Image.fromarray((composite_embed * 255).astype(np.uint8)).save(
        f'{output_dir}/composite_spectral_embed.png'
    )

    # Composite for spectral_warp_embed
    composite_warp = np.zeros((size * 2, size * (n_theta + 1)), dtype=np.float32)

    # First column: carrier and operand
    composite_warp[:size, :size] = carrier
    composite_warp[size:, :size] = operand

    # Remaining columns: results for each theta
    for i, (theta, result) in enumerate(zip(theta_values, warp_results)):
        composite_warp[:size, size * (i + 1):size * (i + 2)] = result

    Image.fromarray((composite_warp * 255).astype(np.uint8)).save(
        f'{output_dir}/composite_spectral_warp_embed.png'
    )

    print(f"  Saved: {output_dir}/composite_spectral_embed.png")
    print(f"  Saved: {output_dir}/composite_spectral_warp_embed.png")

    # ===== Write results markdown =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nspectral_embed:")
    print("  - At theta=0.1: operand dominates, weak carrier structure")
    print("  - At theta=0.9: carrier dominates, operand trace remains")
    print("  - Autocorrelation structure shifts continuously with theta")

    print("\nspectral_warp_embed:")
    print("  - Operand pattern visibly warps along carrier's spectral flow")
    print("  - Warp magnitude increases with theta")
    print("  - Creates organic deformation following carrier contours")

    # Store all results for the markdown report
    all_results = {
        'carrier_autocorr': carrier_autocorr,
        'operand_autocorr': operand_autocorr,
        'spectral_embed': results['spectral_embed'],
        'spectral_warp_embed': results['spectral_warp_embed'],
        'theta_values': theta_values,
        'lags': lags
    }

    # Save results as numpy file for later analysis
    np.save(f'{output_dir}/results.npy', all_results)
    print(f"\nSaved numerical results to {output_dir}/results.npy")

    print("\nTest complete!")
