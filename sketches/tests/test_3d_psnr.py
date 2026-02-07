#!/usr/bin/env python3
"""
Test PSNR when transforms are rendered to 3D surfaces.

Compares flat texture PSNR vs 3D rendered PSNR to verify that the
spectral transforms maintain perceptual variance when rendered to
a 3D egg surface.

Key insight: UV distortion, lighting, and perspective on 3D surfaces
may affect how transforms appear. We want meaningful PSNR variance
in both flat and 3D rendered views.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from pathlib import Path

from texture.transforms import eigenvector_phase_field
from texture.render.ray_cast import ray_sphere_test, egg_deformation, spherical_uv_from_ray


def generate_checkerboard(size: int, num_squares: int = 8) -> np.ndarray:
    """
    Generate a checkerboard pattern.

    Args:
        size: Output size in pixels
        num_squares: Number of squares per dimension

    Returns:
        (size, size) array with values in [0, 1]
    """
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)

    square_size = size // num_squares
    pattern = ((xx // square_size) + (yy // square_size)) % 2

    return pattern.astype(np.float32)


def generate_noise_texture(size: int, seed: int = 42) -> np.ndarray:
    """
    Generate smooth noise texture using Gaussian blur on random noise.

    Args:
        size: Output size in pixels
        seed: Random seed

    Returns:
        (size, size) array with values in [0, 1]
    """
    rng = np.random.default_rng(seed)
    noise = rng.random((size, size)).astype(np.float32)

    # Smooth with simple box filter
    from scipy.ndimage import uniform_filter
    smooth = uniform_filter(noise, size=size // 16)

    # Normalize to [0, 1]
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
    return smooth


def render_texture_to_egg(
    texture: np.ndarray,
    output_size: int = 400,
    egg_factor: float = 0.25,
    light_dir: tuple = (0.5, 0.7, 0.5)
) -> np.ndarray:
    """
    Render a 2D texture onto a 3D egg surface.

    Uses ray casting to project the texture onto an egg shape with
    proper UV mapping and Lambertian lighting.

    Args:
        texture: (H, W) grayscale texture in [0, 1]
        output_size: Output image size (square)
        egg_factor: Egg deformation factor (0 = sphere, 0.3 = egg)
        light_dir: Light direction (x, y, z)

    Returns:
        (output_size, output_size, 3) RGB image as uint8
    """
    # Normalize light direction
    light = np.array(light_dir, dtype=np.float32)
    light = light / np.linalg.norm(light)

    # Create coordinate grids
    y_coords = np.arange(output_size, dtype=np.float32)
    x_coords = np.arange(output_size, dtype=np.float32)
    px, py = np.meshgrid(x_coords, y_coords)

    # Normalized device coordinates
    scale = 0.85
    nx = (px / output_size - 0.5) * 2 / scale
    ny = (0.5 - py / output_size) * 2 / scale

    # Ray-sphere intersection
    inside_mask, z = ray_sphere_test(nx, ny)

    # Egg deformation
    x_egg, z_egg, egg_mod = egg_deformation(nx, ny, z, egg_factor)

    # Surface normal (vectorized)
    surf_normal = np.stack([nx * egg_mod, ny, z * egg_mod], axis=-1)
    surf_len = np.linalg.norm(surf_normal, axis=-1, keepdims=True)
    surf_len = np.where(surf_len > 1e-6, surf_len, 1.0)
    surf_normal = surf_normal / surf_len

    # Lighting (vectorized dot product)
    ndotl = np.maximum(0, np.sum(surf_normal * light, axis=-1))
    ambient = 0.25
    diffuse = 0.75 * ndotl
    lighting = ambient + diffuse

    # UV coordinates from spherical mapping
    u, v = spherical_uv_from_ray(nx, ny, z_egg, x_egg)

    # Sample texture at UV coordinates
    tex_h, tex_w = texture.shape
    tex_u_idx = (u * (tex_w - 1)).astype(np.int32)
    tex_v_idx = (v * (tex_h - 1)).astype(np.int32)
    tex_u_idx = np.clip(tex_u_idx, 0, tex_w - 1)
    tex_v_idx = np.clip(tex_v_idx, 0, tex_h - 1)

    sampled_texture = texture[tex_v_idx, tex_u_idx]

    # Apply texture and lighting
    # Use a blue-green gradient based on texture value
    base_r = 60 + 100 * sampled_texture
    base_g = 100 + 100 * sampled_texture
    base_b = 180 - 60 * sampled_texture

    color_r = base_r * lighting
    color_g = base_g * lighting
    color_b = base_b * lighting

    # Create output image
    background_color = [30, 30, 45]
    img = np.full((output_size, output_size, 3), background_color, dtype=np.uint8)
    img[inside_mask, 0] = np.clip(color_r[inside_mask], 0, 255).astype(np.uint8)
    img[inside_mask, 1] = np.clip(color_g[inside_mask], 0, 255).astype(np.uint8)
    img[inside_mask, 2] = np.clip(color_b[inside_mask], 0, 255).astype(np.uint8)

    return img


def compute_psnr(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.

    PSNR = 10 * log10(MAX^2 / MSE)

    Higher PSNR = more similar (less change)
    Lower PSNR = more different (more change)

    Args:
        img1: First image (uint8 or float)
        img2: Second image (uint8 or float)
        mask: Optional boolean mask to restrict comparison

    Returns:
        PSNR value in dB
    """
    # Convert to float [0, 1]
    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float64) / 255.0
    if img2.dtype == np.uint8:
        img2 = img2.astype(np.float64) / 255.0

    if mask is not None:
        diff = (img1[mask] - img2[mask]) ** 2
    else:
        diff = (img1 - img2) ** 2

    mse = np.mean(diff)

    if mse < 1e-10:
        return 100.0  # Essentially identical

    max_val = 1.0
    psnr = 10 * np.log10((max_val ** 2) / mse)

    return psnr


def compute_flat_psnr(texture1: np.ndarray, texture2: np.ndarray) -> float:
    """
    Compute PSNR between two flat textures.

    Args:
        texture1: (H, W) grayscale texture in [0, 1]
        texture2: (H, W) grayscale texture in [0, 1]

    Returns:
        PSNR value in dB
    """
    t1 = np.clip(texture1, 0, 1)
    t2 = np.clip(texture2, 0, 1)

    mse = np.mean((t1 - t2) ** 2)

    if mse < 1e-10:
        return 100.0

    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def get_egg_mask(output_size: int, scale: float = 0.85) -> np.ndarray:
    """
    Get a mask for the egg region (inside the sphere).

    Args:
        output_size: Image size
        scale: Same scale factor used in rendering

    Returns:
        Boolean mask of shape (output_size, output_size)
    """
    y_coords = np.arange(output_size, dtype=np.float32)
    x_coords = np.arange(output_size, dtype=np.float32)
    px, py = np.meshgrid(x_coords, y_coords)

    nx = (px / output_size - 0.5) * 2 / scale
    ny = (0.5 - py / output_size) * 2 / scale

    r_sq = nx * nx + ny * ny
    return r_sq <= 1.0


def save_image(array: np.ndarray, path: str):
    """Save image array to file."""
    if array.ndim == 2:
        # Grayscale
        img_data = (np.clip(array, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(img_data, mode='L')
    else:
        # RGB
        img = Image.fromarray(array, mode='RGB')

    img.save(path)
    print(f"  Saved: {path}")


def main():
    """Run 3D PSNR test."""
    print("=" * 70)
    print("3D RENDER PSNR VALIDATION TEST")
    print("=" * 70)
    print("\nMeasuring PSNR when transforms are rendered to 3D egg surfaces.")
    print("Compares flat PSNR vs 3D PSNR to validate spectral transform behavior.\n")

    # Configuration
    texture_size = 128
    render_size = 400
    theta_values = [0.1, 0.5, 0.9]
    output_dir = Path('/home/bigboi/itten/demo_output/3d_psnr')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate test textures
    print("1. Generating test textures...")
    textures = {
        'checkerboard': generate_checkerboard(texture_size, num_squares=8),
        'noise': generate_noise_texture(texture_size, seed=42),
    }

    # Save original textures
    print("\n2. Saving original textures...")
    for name, texture in textures.items():
        path = output_dir / f'texture_{name}_original.png'
        save_image(texture, str(path))

    # Get egg mask for masked PSNR computation
    egg_mask = get_egg_mask(render_size)
    print(f"\n   Egg mask coverage: {egg_mask.sum() / egg_mask.size * 100:.1f}%")

    # Results storage
    results = {}

    print("\n3. Processing transforms and computing PSNR...")
    print("-" * 70)

    for texture_name, base_texture in textures.items():
        print(f"\n  Texture: {texture_name}")
        results[texture_name] = {
            'flat_psnr': [],
            '3d_psnr': [],
            'theta': theta_values,
        }

        # Render baseline (untransformed) to 3D
        baseline_3d = render_texture_to_egg(base_texture, render_size)
        path = output_dir / f'3d_{texture_name}_baseline.png'
        save_image(baseline_3d, str(path))

        for theta in theta_values:
            print(f"\n    theta = {theta}:")

            # Apply eigenvector_phase_field transform
            try:
                transformed_flat, _ = eigenvector_phase_field(
                    base_texture,
                    theta=theta,
                    eigenpair=(0, 1),
                    edge_threshold=0.1,
                    num_iterations=50
                )
            except Exception as e:
                print(f"      [ERROR] Transform failed: {e}")
                results[texture_name]['flat_psnr'].append(float('nan'))
                results[texture_name]['3d_psnr'].append(float('nan'))
                continue

            # Save transformed flat texture
            path = output_dir / f'texture_{texture_name}_theta{theta:.1f}.png'
            save_image(transformed_flat, str(path))

            # Compute flat PSNR
            flat_psnr = compute_flat_psnr(base_texture, transformed_flat)
            results[texture_name]['flat_psnr'].append(flat_psnr)
            print(f"      Flat PSNR: {flat_psnr:.2f} dB")

            # Render transformed texture to 3D
            transformed_3d = render_texture_to_egg(transformed_flat, render_size)
            path = output_dir / f'3d_{texture_name}_theta{theta:.1f}.png'
            save_image(transformed_3d, str(path))

            # Compute 3D PSNR (using egg mask to exclude background)
            psnr_3d = compute_psnr(baseline_3d, transformed_3d, egg_mask)
            results[texture_name]['3d_psnr'].append(psnr_3d)
            print(f"      3D PSNR:   {psnr_3d:.2f} dB")

            # Compute difference
            diff = flat_psnr - psnr_3d
            print(f"      Diff (flat - 3D): {diff:+.2f} dB")

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nPSNR Table (dB):")
    print("-" * 70)
    print(f"{'Texture':<15} | {'theta':<8} | {'Flat PSNR':<12} | {'3D PSNR':<12} | {'Diff':<10}")
    print("-" * 70)

    for texture_name in textures.keys():
        for i, theta in enumerate(theta_values):
            flat_p = results[texture_name]['flat_psnr'][i]
            psnr_3d = results[texture_name]['3d_psnr'][i]
            diff = flat_p - psnr_3d
            print(f"{texture_name:<15} | {theta:<8.1f} | {flat_p:<12.2f} | {psnr_3d:<12.2f} | {diff:+10.2f}")
        print("-" * 70)

    # Compute statistics
    print("\nStatistical Analysis:")
    print("-" * 70)

    all_flat = []
    all_3d = []
    for texture_name in textures.keys():
        all_flat.extend(results[texture_name]['flat_psnr'])
        all_3d.extend(results[texture_name]['3d_psnr'])

    all_flat = np.array([x for x in all_flat if not np.isnan(x)])
    all_3d = np.array([x for x in all_3d if not np.isnan(x)])

    if len(all_flat) > 0:
        print(f"  Flat PSNR range:  {all_flat.min():.2f} - {all_flat.max():.2f} dB")
        print(f"  Flat PSNR std:    {all_flat.std():.2f} dB")
        print(f"  3D PSNR range:    {all_3d.min():.2f} - {all_3d.max():.2f} dB")
        print(f"  3D PSNR std:      {all_3d.std():.2f} dB")

        avg_diff = np.mean(all_flat - all_3d)
        print(f"\n  Average diff (flat - 3D): {avg_diff:+.2f} dB")

        # Correlation
        if len(all_flat) > 2:
            corr = np.corrcoef(all_flat, all_3d)[0, 1]
            print(f"  Correlation (flat vs 3D): {corr:.3f}")

    # Write results report
    write_results_report(results, textures.keys(), theta_values, output_dir)

    print("\n" + "=" * 70)
    print("Test complete!")
    print(f"Output saved to: {output_dir}")
    print("=" * 70)

    return results


def write_results_report(
    results: dict,
    texture_names: list,
    theta_values: list,
    output_dir: Path
):
    """Write results to markdown report."""
    report_path = Path('/home/bigboi/itten/hypercontexts/3d-render-psnr-results.md')

    lines = [
        "# 3D Render PSNR Results",
        "",
        "## Overview",
        "",
        "This report compares PSNR (Peak Signal-to-Noise Ratio) values between:",
        "- **Flat PSNR**: Comparison of flat 2D textures before/after transform",
        "- **3D PSNR**: Comparison of textures rendered onto a 3D egg surface",
        "",
        "Higher PSNR = more similar (less change from transform)",
        "Lower PSNR = more different (more change from transform)",
        "",
        "## Test Configuration",
        "",
        "- Transform: `eigenvector_phase_field`",
        "- Texture size: 128x128",
        "- 3D render size: 400x400",
        f"- Theta values: {theta_values}",
        "",
        "## Results Table",
        "",
        "| Texture | theta | Flat PSNR (dB) | 3D PSNR (dB) | Diff |",
        "|---------|-------|----------------|--------------|------|",
    ]

    all_flat = []
    all_3d = []

    for texture_name in texture_names:
        for i, theta in enumerate(theta_values):
            flat_p = results[texture_name]['flat_psnr'][i]
            psnr_3d = results[texture_name]['3d_psnr'][i]
            diff = flat_p - psnr_3d

            all_flat.append(flat_p)
            all_3d.append(psnr_3d)

            lines.append(
                f"| {texture_name} | {theta:.1f} | {flat_p:.2f} | {psnr_3d:.2f} | {diff:+.2f} |"
            )

    all_flat = np.array([x for x in all_flat if not np.isnan(x)])
    all_3d = np.array([x for x in all_3d if not np.isnan(x)])

    lines.extend([
        "",
        "## Statistical Summary",
        "",
        f"- **Flat PSNR range**: {all_flat.min():.2f} - {all_flat.max():.2f} dB",
        f"- **Flat PSNR std**: {all_flat.std():.2f} dB",
        f"- **3D PSNR range**: {all_3d.min():.2f} - {all_3d.max():.2f} dB",
        f"- **3D PSNR std**: {all_3d.std():.2f} dB",
        f"- **Average diff (flat - 3D)**: {np.mean(all_flat - all_3d):+.2f} dB",
        "",
    ])

    if len(all_flat) > 2:
        corr = np.corrcoef(all_flat, all_3d)[0, 1]
        lines.append(f"- **Correlation (flat vs 3D)**: {corr:.3f}")
        lines.append("")

    # Interpretation
    lines.extend([
        "## Interpretation",
        "",
    ])

    avg_diff = np.mean(all_flat - all_3d)
    flat_var = all_flat.std()
    threevd_var = all_3d.std()

    if flat_var > 1.0 and threevd_var > 1.0:
        lines.extend([
            "**Both flat and 3D PSNR show meaningful variance across theta values.**",
            "",
            "This indicates that:",
            "1. The spectral transform produces perceptually different outputs at different theta values",
            "2. This perceptual difference is preserved when rendered to a 3D surface",
            "3. The 3D rendering does not destroy the transform's spectral properties",
            "",
        ])
    elif flat_var > 1.0:
        lines.extend([
            "**Flat PSNR shows variance but 3D PSNR is more consistent.**",
            "",
            "This may indicate that:",
            "1. The transform produces different flat textures",
            "2. But the 3D rendering (lighting, UV mapping) normalizes the differences",
            "3. Consider adjusting rendering parameters for better sensitivity",
            "",
        ])
    else:
        lines.extend([
            "**Limited variance detected in both flat and 3D PSNR.**",
            "",
            "This may indicate:",
            "1. The transform parameters may need adjustment",
            "2. The test textures may not be sensitive to the eigenvector_phase_field transform",
            "3. Consider using different texture patterns or transform settings",
            "",
        ])

    if abs(avg_diff) > 2.0:
        lines.extend([
            f"**Note**: The average difference of {avg_diff:+.2f} dB between flat and 3D PSNR",
            "indicates that the 3D rendering process affects perceived difference.",
            "Positive difference means 3D rendering amplifies perceived change.",
            "Negative difference means 3D rendering reduces perceived change.",
            "",
        ])

    lines.extend([
        "## Sample Images",
        "",
        f"All images saved to: `{output_dir}/`",
        "",
        "### Files",
        "",
        "- `texture_<name>_original.png` - Original flat textures",
        "- `texture_<name>_theta<value>.png` - Transformed flat textures",
        "- `3d_<name>_baseline.png` - Original texture rendered to 3D egg",
        "- `3d_<name>_theta<value>.png` - Transformed texture rendered to 3D egg",
        "",
        "## Methodology",
        "",
        "1. Generate test textures (checkerboard, noise)",
        "2. For each texture and theta value:",
        "   - Apply `eigenvector_phase_field` transform",
        "   - Compute flat PSNR between original and transformed texture",
        "   - Render both original and transformed to 3D egg surface",
        "   - Compute 3D PSNR using egg mask (excluding background)",
        "3. Compare flat vs 3D PSNR to assess how 3D rendering affects perceptual change",
    ])

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n  Report written to: {report_path}")


if __name__ == '__main__':
    main()
