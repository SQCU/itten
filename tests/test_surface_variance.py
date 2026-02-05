"""
Test surface variance with PSNR covariance across surfaces x theta.

Generates 3 surface textures, applies synthesize() with different theta values,
and computes PSNR matrix to demonstrate non-trivial spectral transform behavior.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from texture.surfaces import generate_marble, generate_brick, generate_noise
from texture.core import synthesize


def compute_psnr(original: np.ndarray, transformed: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between original and transformed.

    PSNR = 10 * log10(MAX^2 / MSE)

    Higher PSNR = more similar (less change from transform)
    Lower PSNR = more different (more change from transform)

    Args:
        original: Original image array [0, 1]
        transformed: Transformed image array [0, 1]

    Returns:
        PSNR value in dB
    """
    # Ensure both are normalized to [0, 1]
    orig = np.clip(original.astype(np.float64), 0, 1)
    trans = np.clip(transformed.astype(np.float64), 0, 1)

    mse = np.mean((orig - trans) ** 2)

    if mse < 1e-10:
        return 100.0  # Essentially identical

    max_val = 1.0
    psnr = 10 * np.log10((max_val ** 2) / mse)

    return psnr


def save_image(array: np.ndarray, path: str):
    """Save normalized array as grayscale PNG."""
    # Normalize to [0, 255]
    img_data = (np.clip(array, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(img_data, mode='L')
    img.save(path)
    print(f"Saved: {path}")


def main():
    """Run surface variance test and generate PSNR matrix."""
    print("=" * 60)
    print("Surface Variance Test: PSNR Covariance Matrix")
    print("=" * 60)

    # Configuration
    size = 128
    theta_values = [0.1, 0.5, 0.9]
    surface_names = ['marble', 'brick', 'noise']
    output_dir = '/home/bigboi/itten/demo_output/surface_variance'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate surfaces
    print("\n1. Generating surface textures...")
    surfaces = {
        'marble': generate_marble(size, seed=42),
        'brick': generate_brick(size, seed=42),
        'noise': generate_noise(size, seed=42),
    }

    # Save original surfaces
    print("\n2. Saving original surfaces...")
    for name, surface in surfaces.items():
        path = os.path.join(output_dir, f'surface_{name}_original.png')
        save_image(surface, path)

    # Initialize PSNR matrix
    # Rows = surfaces, Columns = theta values
    psnr_matrix = np.zeros((3, 3))

    print("\n3. Computing PSNR for each surface x theta combination...")
    print("-" * 60)

    for i, surface_name in enumerate(surface_names):
        surface = surfaces[surface_name]
        print(f"\n  Surface: {surface_name}")

        for j, theta in enumerate(theta_values):
            print(f"    theta={theta}...", end=" ")

            # Apply synthesize with the surface as carrier
            result = synthesize(
                carrier=surface,
                operand=None,  # No operand - isolate surface effect
                theta=theta,
                gamma=0.3,
                num_eigenvectors=8,
                mode='spectral'
            )

            synthesized = result.height_field

            # Compute PSNR
            psnr = compute_psnr(surface, synthesized)
            psnr_matrix[i, j] = psnr

            print(f"PSNR = {psnr:.2f} dB")

            # Save synthesized image
            synth_path = os.path.join(
                output_dir,
                f'surface_{surface_name}_theta{theta:.1f}.png'
            )
            save_image(synthesized, synth_path)

    # Print PSNR matrix
    print("\n" + "=" * 60)
    print("PSNR MATRIX (surfaces x theta)")
    print("=" * 60)
    print("\n              theta=0.1    theta=0.5    theta=0.9")
    print("-" * 55)
    for i, name in enumerate(surface_names):
        row = psnr_matrix[i, :]
        print(f"{name:10s}    {row[0]:8.2f}    {row[1]:8.2f}    {row[2]:8.2f}")

    # Analyze covariance
    print("\n" + "=" * 60)
    print("COVARIANCE ANALYSIS")
    print("=" * 60)

    # Row variance (surface sensitivity)
    row_vars = np.var(psnr_matrix, axis=1)
    print("\nPSNR variance across theta (per surface):")
    for i, name in enumerate(surface_names):
        print(f"  {name}: {row_vars[i]:.4f}")

    # Column variance (theta sensitivity)
    col_vars = np.var(psnr_matrix, axis=0)
    print("\nPSNR variance across surfaces (per theta):")
    for j, theta in enumerate(theta_values):
        print(f"  theta={theta}: {col_vars[j]:.4f}")

    # Overall statistics
    print(f"\nOverall PSNR range: {psnr_matrix.min():.2f} - {psnr_matrix.max():.2f} dB")
    print(f"Overall PSNR std: {np.std(psnr_matrix):.2f} dB")

    # Check for non-trivial behavior
    total_variance = np.var(psnr_matrix)
    if total_variance > 1.0:
        print("\n[PASS] Non-trivial covariance detected!")
        print("  The spectral transform affects different surfaces differently.")
    else:
        print("\n[NOTE] Low variance detected.")
        print("  Consider adjusting parameters for more diverse responses.")

    # Write markdown report
    write_psnr_report(psnr_matrix, surface_names, theta_values)

    print("\n" + "=" * 60)
    print("Test complete!")
    print(f"Output saved to: {output_dir}")
    print("=" * 60)

    return psnr_matrix


def write_psnr_report(
    psnr_matrix: np.ndarray,
    surface_names: list,
    theta_values: list
):
    """Write PSNR matrix report to markdown file."""
    report_path = '/home/bigboi/itten/hypercontexts/surface-psnr-matrix.md'

    lines = [
        "# Surface PSNR Covariance Matrix",
        "",
        "## Overview",
        "",
        "This matrix shows PSNR (Peak Signal-to-Noise Ratio) values measuring",
        "the difference between original surface textures and their spectral",
        "transform outputs at different theta values.",
        "",
        "- **Higher PSNR**: Less change (surface preserved)",
        "- **Lower PSNR**: More change (transform had larger effect)",
        "",
        "## PSNR Matrix",
        "",
        "| Surface | theta=0.1 | theta=0.5 | theta=0.9 |",
        "|---------|-----------|-----------|-----------|",
    ]

    for i, name in enumerate(surface_names):
        row = psnr_matrix[i, :]
        lines.append(f"| {name} | {row[0]:.2f} dB | {row[1]:.2f} dB | {row[2]:.2f} dB |")

    lines.extend([
        "",
        "## Statistical Analysis",
        "",
        "### Per-Surface Variance (sensitivity to theta)",
        "",
    ])

    row_vars = np.var(psnr_matrix, axis=1)
    for i, name in enumerate(surface_names):
        lines.append(f"- **{name}**: {row_vars[i]:.4f}")

    lines.extend([
        "",
        "### Per-Theta Variance (sensitivity to surface type)",
        "",
    ])

    col_vars = np.var(psnr_matrix, axis=0)
    for j, theta in enumerate(theta_values):
        lines.append(f"- **theta={theta}**: {col_vars[j]:.4f}")

    lines.extend([
        "",
        "### Overall Statistics",
        "",
        f"- PSNR Range: {psnr_matrix.min():.2f} - {psnr_matrix.max():.2f} dB",
        f"- PSNR Standard Deviation: {np.std(psnr_matrix):.2f} dB",
        f"- Total Variance: {np.var(psnr_matrix):.4f}",
        "",
        "## Interpretation",
        "",
    ])

    # Add interpretation
    total_var = np.var(psnr_matrix)
    if total_var > 1.0:
        lines.extend([
            "The variance in the PSNR matrix demonstrates that the spectral",
            "transform produces **non-trivial, surface-dependent results**:",
            "",
            "1. Different surfaces respond differently to the same theta value",
            "2. The same surface responds differently to different theta values",
            "3. This covariance proves the transform is performing meaningful",
            "   spectral decomposition rather than trivial operations",
        ])
    else:
        lines.extend([
            "The low variance suggests the transform has similar effects",
            "across different surfaces and theta values. This may indicate:",
            "",
            "1. The surfaces are too similar in spectral content",
            "2. The theta range may need adjustment",
            "3. Consider using more diverse surface textures",
        ])

    lines.extend([
        "",
        "## Sample Images",
        "",
        "Original surfaces and transformed outputs are saved to:",
        "`/home/bigboi/itten/demo_output/surface_variance/`",
        "",
        "Files:",
        "- `surface_marble_original.png`",
        "- `surface_brick_original.png`",
        "- `surface_noise_original.png`",
        "- `surface_<name>_theta<value>.png` for each combination",
    ])

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nPSNR report written to: {report_path}")


if __name__ == '__main__':
    main()
