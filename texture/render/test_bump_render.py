#!/usr/bin/env python
"""
Test script for bump render functionality.

Generates textures using synthesize() and renders them with:
1. Color-only mapping (texture as decal)
2. Bump-mapped (texture as displacement + normal perturbation)

Sweeps theta=[0.1, 0.5, 0.9] to show different texture characters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from PIL import Image

from texture.core import synthesize
from texture.render.bump_render import (
    render_bumped_egg,
    render_color_only_egg,
    create_comparison_grid
)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def main():
    output_dir = "/home/bigboi/itten/demo_output/bump_render"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Bump Render Test: Color-Only vs Bump-Mapped")
    print("=" * 60)

    # Test theta values
    theta_values = [0.1, 0.5, 0.9]

    # Storage for images
    color_only_images = []
    bump_mapped_images = []
    all_images = []
    all_labels = []

    # Generate and render for each theta
    for theta in theta_values:
        print(f"\nProcessing theta={theta}...")

        # Synthesize texture using spectral embedding
        result = synthesize(
            'amongus',
            'noise',
            theta=theta,
            gamma=0.3,
            output_size=128,
            num_eigenvectors=8
        )

        height_field = result.height_field
        normal_map = result.normal_map

        print(f"  Height field shape: {height_field.shape}")
        print(f"  Height range: [{height_field.min():.3f}, {height_field.max():.3f}]")
        print(f"  Normal map shape: {normal_map.shape}")

        # Save height field as image
        height_img = (height_field * 255).astype(np.uint8)
        Image.fromarray(height_img).save(
            os.path.join(output_dir, f"height_theta_{theta:.1f}.png")
        )

        # Save normal map as image
        normal_img = (normal_map * 255).astype(np.uint8)
        Image.fromarray(normal_img).save(
            os.path.join(output_dir, f"normal_theta_{theta:.1f}.png")
        )

        # Render COLOR-ONLY (texture as decal, smooth egg surface)
        color_only = render_color_only_egg(
            height_field,
            output_size=512,
            egg_factor=0.25,
            light_dir=(0.5, 0.7, 0.5),
            base_color=(0.78, 0.47, 0.31),
            ambient=0.25
        )
        color_only_images.append(color_only)

        Image.fromarray(color_only).save(
            os.path.join(output_dir, f"color_only_theta_{theta:.1f}.png")
        )
        print(f"  Saved color_only_theta_{theta:.1f}.png")

        # Render BUMP-MAPPED (true displacement + normal perturbation)
        bump_mapped = render_bumped_egg(
            height_field,
            normal_map=normal_map,
            displacement_scale=0.15,  # Visible displacement
            normal_strength=0.7,       # Strong normal detail
            output_size=512,
            egg_factor=0.25,
            light_dir=(0.5, 0.7, 0.5),
            body_color=(0.78, 0.47, 0.31),
            specular_color=(0.8, 0.85, 1.0),
            ambient=0.15,
            specular_power=32.0
        )
        bump_mapped_images.append(bump_mapped)

        Image.fromarray(bump_mapped).save(
            os.path.join(output_dir, f"bump_mapped_theta_{theta:.1f}.png")
        )
        print(f"  Saved bump_mapped_theta_{theta:.1f}.png")

        # Add to comparison
        all_images.append(color_only)
        all_labels.append(f"Color θ={theta}")
        all_images.append(bump_mapped)
        all_labels.append(f"Bump θ={theta}")

    # Create comparison grid: color vs bump for each theta
    print("\nCreating comparison grid...")
    comparison_grid = create_comparison_grid(all_images, all_labels)
    Image.fromarray(comparison_grid).save(
        os.path.join(output_dir, "comparison_grid.png")
    )
    print("  Saved comparison_grid.png")

    # Create side-by-side pairs for each theta
    for i, theta in enumerate(theta_values):
        pair = create_comparison_grid(
            [color_only_images[i], bump_mapped_images[i]],
            [f"Color Only θ={theta}", f"Bump Mapped θ={theta}"]
        )
        Image.fromarray(pair).save(
            os.path.join(output_dir, f"pair_theta_{theta:.1f}.png")
        )
        print(f"  Saved pair_theta_{theta:.1f}.png")

    # Compute PSNR between adjacent theta values
    print("\n" + "=" * 60)
    print("PSNR Analysis")
    print("=" * 60)

    print("\nColor-only PSNR between adjacent theta values:")
    for i in range(len(theta_values) - 1):
        psnr = compute_psnr(color_only_images[i], color_only_images[i + 1])
        print(f"  theta={theta_values[i]:.1f} vs theta={theta_values[i+1]:.1f}: {psnr:.2f} dB")

    print("\nBump-mapped PSNR between adjacent theta values:")
    for i in range(len(theta_values) - 1):
        psnr = compute_psnr(bump_mapped_images[i], bump_mapped_images[i + 1])
        print(f"  theta={theta_values[i]:.1f} vs theta={theta_values[i+1]:.1f}: {psnr:.2f} dB")

    # PSNR between color-only and bump-mapped for same theta
    print("\nPSNR between color-only and bump-mapped (same theta):")
    for i, theta in enumerate(theta_values):
        psnr = compute_psnr(color_only_images[i], bump_mapped_images[i])
        print(f"  theta={theta:.1f}: {psnr:.2f} dB")

    print("\n" + "=" * 60)
    print("Output saved to:", output_dir)
    print("=" * 60)

    return output_dir


if __name__ == "__main__":
    main()
