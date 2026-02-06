#!/usr/bin/env python3
"""Demo 4: Texture Atlas Shader -- Phase E of TODO_DEMO_RECOVERY.md.

Demonstrates that the spectral shader applied to a flat texture atlas produces
"intelligible deformations" when UV-mapped to a 3D surface. The spectral
transformation respects the atlas geometry: local spectral structure in the
flat atlas maps to coherent 3D surface features after UV mapping.

Composition chain:
    atlas texture -> SpectralShader -> shaded atlas -> UV map -> 3D egg render

Uses ONLY the _even_cuter Module files:
    - spectral_shader_model.py: SpectralShader.from_config
    - spectral_renderer.py: HeightToNormals, BilinearSampler, EggSurfaceRenderer
    - image_io.py: load_image, save_image

Zero imports from spectral_ops_fast.py or spectral_ops_fns.py.
All code pure PyTorch -- no numpy (except final PIL save via image_io).

Usage:
    uv run demos/demo_texture_atlas.py
    uv run demos/demo_texture_atlas.py --image input_images/toof.png
    uv run demos/demo_texture_atlas.py --output-dir demo_output --resolution 256
"""

import argparse
import os
import sys
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from spectral_shader_model import SpectralShader
from spectral_renderer import HeightToNormals, BilinearSampler, EggSurfaceRenderer
from image_io import load_image, save_image


# ======================================================================
# Default config (same as spectral_shader_main.py DEFAULT_CONFIG)
# ======================================================================

DEFAULT_CONFIG = {
    "tile_size": 64,
    "overlap": 16,
    "num_eigenvectors": 4,
    "radii": [1, 2, 3, 4, 5, 6],
    "radius_weights": [1.0, 0.6, 0.4, 0.3, 0.2, 0.1],
    "edge_threshold": 0.15,
    "gate_sharpness": 8.0,
    "effect_strength": 1.0,
    "translation_strength": 15.0,
    "shadow_offset": 5.0,
    "dilation_radius": 2,
    "lanczos_iterations": 30,
}


# ======================================================================
# Atlas generation (procedural fallback)
# ======================================================================


def generate_checkerboard_atlas(
    size: int = 128, num_squares: int = 8, device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate a colored checkerboard atlas texture.

    This serves as a minimal test atlas: the grid structure makes it easy
    to see how spectral transformations deform the pattern under UV mapping.

    Returns:
        (size, size, 3) float32 tensor in [0, 1].
    """
    img = torch.zeros(size, size, 3, dtype=torch.float32, device=device)
    sq = size // num_squares

    for i in range(num_squares):
        for j in range(num_squares):
            r0, r1 = i * sq, (i + 1) * sq
            c0, c1 = j * sq, (j + 1) * sq
            if (i + j) % 2 == 0:
                # Warm color varying by position
                img[r0:r1, c0:c1, 0] = 0.8 + 0.2 * (i / num_squares)
                img[r0:r1, c0:c1, 1] = 0.3 + 0.4 * (j / num_squares)
                img[r0:r1, c0:c1, 2] = 0.2
            else:
                # Cool color varying by position
                img[r0:r1, c0:c1, 0] = 0.2
                img[r0:r1, c0:c1, 1] = 0.3 + 0.3 * (i / num_squares)
                img[r0:r1, c0:c1, 2] = 0.6 + 0.3 * (j / num_squares)

    return img


def generate_gradient_atlas(
    size: int = 128, device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate a gradient atlas with radial features.

    Returns:
        (size, size, 3) float32 tensor in [0, 1].
    """
    y = torch.linspace(0, 1, size, device=device).unsqueeze(1).expand(size, size)
    x = torch.linspace(0, 1, size, device=device).unsqueeze(0).expand(size, size)

    # Radial distance from center
    cx, cy = 0.5, 0.5
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Color channels: horizontal gradient, vertical gradient, radial
    img = torch.stack([x, y, 1.0 - r.clamp(0, 1)], dim=-1)
    return img


# ======================================================================
# Helper: grayscale conversion
# ======================================================================

def to_grayscale(img: torch.Tensor) -> torch.Tensor:
    """Convert (H, W, 3) RGB to (H, W) grayscale. Pure torch."""
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Demo 4: Texture Atlas Shader -- spectral shader on atlas -> UV -> 3D"
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path to atlas image. If not given, uses procedural atlas.")
    parser.add_argument("--output-dir", type=str, default="demo_output",
                        help="Output directory. Default demo_output.")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Egg render resolution. Default 256.")
    parser.add_argument("--n-passes", type=int, default=1,
                        help="Number of AR shader passes. Default 1.")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("DEMO 4: Texture Atlas Shader")
    print("  Spectral shader on atlas -> UV map -> 3D egg render")
    print("  Key: spectral transforms produce intelligible deformations")
    print("=" * 60)
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Render resolution: {args.resolution}")
    print(f"AR passes: {args.n_passes}")

    # ------------------------------------------------------------------
    # Step 1: Load or create atlas texture
    # ------------------------------------------------------------------
    print("\n--- Step 1: Load/create atlas texture ---")
    t0 = time.time()

    if args.image is not None:
        print(f"Loading atlas from: {args.image}")
        atlas = load_image(args.image, device=device)
        atlas_name = os.path.splitext(os.path.basename(args.image))[0]
    else:
        # Try to find an existing test image
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate_images = [
            os.path.join(project_root, "input_images", "toof.png"),
            os.path.join(project_root, "input_images", "snek-heavy.png"),
            os.path.join(project_root, "input_images", "offhand_pleometric.png"),
        ]
        atlas = None
        atlas_name = "procedural"
        for cpath in candidate_images:
            if os.path.exists(cpath):
                print(f"Found test image: {cpath}")
                atlas = load_image(cpath, device=device)
                atlas_name = os.path.splitext(os.path.basename(cpath))[0]
                break

        if atlas is None:
            print("No test images found. Generating procedural checkerboard atlas.")
            atlas = generate_checkerboard_atlas(128, device=device)
            atlas_name = "checkerboard"

    H, W = atlas.shape[0], atlas.shape[1]
    load_time = time.time() - t0
    print(f"Atlas: {atlas_name}, shape {H}x{W}, loaded in {load_time:.3f}s")

    # Save original atlas
    p = save_image(atlas, os.path.join(output_dir, f"demo4_atlas_original_{atlas_name}.png"))

    # ------------------------------------------------------------------
    # Step 2: Apply SpectralShader to the atlas
    # ------------------------------------------------------------------
    print("\n--- Step 2: Apply SpectralShader to atlas ---")
    t0 = time.time()

    shader = SpectralShader.from_config(DEFAULT_CONFIG)
    shader = shader.to(device)
    print(f"Model:\n{shader}")

    t_shader_start = time.time()
    with torch.no_grad():
        shaded_atlas, intermediates = shader(atlas, n_passes=args.n_passes, decay=0.85)
    shader_time = time.time() - t_shader_start
    print(f"Shader applied ({args.n_passes} pass(es)) in {shader_time:.3f}s")

    # Clamp to valid range
    shaded_atlas = shaded_atlas.clamp(0, 1)

    # Save shaded atlas
    p = save_image(
        shaded_atlas,
        os.path.join(output_dir, f"demo4_atlas_shaded_{atlas_name}.png"),
    )

    # ------------------------------------------------------------------
    # Step 3: Create normal maps from both atlases
    # ------------------------------------------------------------------
    print("\n--- Step 3: Generate normal maps ---")
    t0 = time.time()

    height_to_normals = HeightToNormals(strength=2.0)

    # Original atlas normal map
    original_gray = to_grayscale(atlas)
    original_normals = height_to_normals(original_gray)

    # Shaded atlas normal map
    shaded_gray = to_grayscale(shaded_atlas)
    shaded_normals = height_to_normals(shaded_gray)

    normals_time = time.time() - t0
    print(f"Normal maps computed in {normals_time:.3f}s")

    # Save normal maps
    save_image(
        original_normals,
        os.path.join(output_dir, f"demo4_normals_original_{atlas_name}.png"),
    )
    save_image(
        shaded_normals,
        os.path.join(output_dir, f"demo4_normals_shaded_{atlas_name}.png"),
    )

    # ------------------------------------------------------------------
    # Step 4: Render both on the egg surface
    # ------------------------------------------------------------------
    print("\n--- Step 4: Render on egg surface ---")

    renderer = EggSurfaceRenderer(
        resolution=args.resolution,
        egg_factor=0.25,
        bump_strength=1.0,
        light_dir=(0.5, 0.7, 1.0),
    ).to(device)

    # Render with original atlas
    print("  Rendering with original atlas...")
    t0 = time.time()
    render_original = renderer(atlas, original_normals)
    render_original_time = time.time() - t0
    print(f"  Original render: {render_original.shape}, {render_original_time:.3f}s")

    save_image(
        render_original,
        os.path.join(output_dir, f"demo4_egg_original_{atlas_name}.png"),
    )

    # Render with shaded atlas
    print("  Rendering with shaded atlas...")
    t0 = time.time()
    render_shaded = renderer(shaded_atlas, shaded_normals)
    render_shaded_time = time.time() - t0
    print(f"  Shaded render:   {render_shaded.shape}, {render_shaded_time:.3f}s")

    save_image(
        render_shaded,
        os.path.join(output_dir, f"demo4_egg_shaded_{atlas_name}.png"),
    )

    # ------------------------------------------------------------------
    # Step 5: Also demonstrate BilinearSampler for explicit UV sampling
    # ------------------------------------------------------------------
    print("\n--- Step 5: Explicit UV sampling demonstration ---")
    t0 = time.time()

    sampler = BilinearSampler()

    # Create a grid of UV coordinates (like a flat UV unwrap visualization)
    uv_res = min(args.resolution, 256)
    u_coords = torch.linspace(0, 1, uv_res, device=device).unsqueeze(0).expand(uv_res, -1)
    v_coords = torch.linspace(0, 1, uv_res, device=device).unsqueeze(1).expand(-1, uv_res)

    # Sample both atlases at the UV grid
    sampled_original = sampler(atlas, u_coords, v_coords)  # (uv_res, uv_res, 3)
    sampled_shaded = sampler(shaded_atlas, u_coords, v_coords)

    # Compute difference to show where spectral transform changed the atlas
    diff = torch.abs(sampled_shaded - sampled_original)
    # Amplify for visibility
    diff_amplified = (diff * 3.0).clamp(0, 1)

    uv_time = time.time() - t0
    print(f"  UV sampling ({uv_res}x{uv_res} grid) in {uv_time:.3f}s")

    save_image(
        diff_amplified,
        os.path.join(output_dir, f"demo4_atlas_diff_{atlas_name}.png"),
    )

    # ------------------------------------------------------------------
    # Step 6: Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(f"  Atlas load:         {load_time:.3f}s")
    print(f"  Spectral shader:    {shader_time:.3f}s  ({args.n_passes} pass(es))")
    print(f"  Normal maps:        {normals_time:.3f}s")
    print(f"  Egg render (orig):  {render_original_time:.3f}s")
    print(f"  Egg render (shaded):{render_shaded_time:.3f}s")
    print(f"  UV sampling:        {uv_time:.3f}s")
    total = load_time + shader_time + normals_time + render_original_time + render_shaded_time + uv_time
    print(f"  Total:              {total:.3f}s")

    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"  Original atlas:     demo4_atlas_original_{atlas_name}_*.png")
    print(f"  Shaded atlas:       demo4_atlas_shaded_{atlas_name}_*.png")
    print(f"  Normal map (orig):  demo4_normals_original_{atlas_name}_*.png")
    print(f"  Normal map (shaded):demo4_normals_shaded_{atlas_name}_*.png")
    print(f"  3D render (orig):   demo4_egg_original_{atlas_name}_*.png")
    print(f"  3D render (shaded): demo4_egg_shaded_{atlas_name}_*.png")
    print(f"  Atlas diff map:     demo4_atlas_diff_{atlas_name}_*.png")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("The spectral shader transforms the flat atlas texture in a way that")
    print("respects the local spectral structure of the image. When UV-mapped to")
    print("the egg surface:")
    print("  - Original atlas: baseline 3D appearance")
    print("  - Shaded atlas: spectrally-transformed deformations visible as")
    print("    coherent surface features (thickened contours, shadow displacement,")
    print("    spectral gating effects)")
    print("  - Diff map: highlights where the spectral transform acts most strongly")
    print("\nThe key demonstration: spectral transformations of the flat atlas produce")
    print("INTELLIGIBLE deformations on the 3D surface -- the spectral structure")
    print("respects the atlas geometry.")

    print("\nDemo 4 complete.")


if __name__ == "__main__":
    main()
