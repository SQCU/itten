#!/usr/bin/env python3
"""Demo 1: Spectral Normal Mapping — Phase E of demo recovery.

Composition chain:
    image -> Laplacian -> eigenvectors -> height field -> normals -> 3D egg render

Demonstrates:
1. Partial spectral transform of a pictographic shape via ImageLaplacianBuilder
   and GraphEmbedding for sparse edge/contour/curve detection.
2. Fiedler vector as height field -> normal map -> bump-mapped egg surface
   ("visual echoes" from spectral methods through normal mapping).
3. SpectralShaderBlock applied to the image using the Fiedler vector, then
   the shaded result converted to height -> normals -> second render
   ("second-order interactions").

Uses ONLY the _even_cuter Module files:
- spectral_graph_embedding.py: GraphEmbedding, ImageLaplacianBuilder
- spectral_shader_layers.py: SpectralShaderBlock
- spectral_renderer.py: HeightToNormals, EggSurfaceRenderer
- image_io.py: load_image, save_image

Zero imports from spectral_ops_fast.py or spectral_ops_fns.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from pathlib import Path

import torch

from spectral_graph_embedding import GraphEmbedding, ImageLaplacianBuilder
from spectral_shader_layers import SpectralShaderBlock
from spectral_renderer import HeightToNormals, EggSurfaceRenderer
from image_io import load_image, save_image


def create_synthetic_amongus(size: int = 64, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Create a synthetic Among Us-like shape as an RGB image.

    This is a fallback when no input image is provided. The shape has clear
    contours and interior regions that produce interesting spectral structure.

    Returns:
        (size, size, 3) float32 tensor in [0, 1].
    """
    img = torch.ones((size, size, 3), device=device, dtype=torch.float32) * 0.95

    cy, cx = size // 2, size // 2

    # Body: rounded rectangle
    body_top = int(size * 0.2)
    body_bot = int(size * 0.85)
    body_left = int(size * 0.25)
    body_right = int(size * 0.75)
    img[body_top:body_bot, body_left:body_right] = torch.tensor([0.2, 0.1, 0.8])

    # Visor: lighter rectangle near top
    visor_top = int(size * 0.25)
    visor_bot = int(size * 0.45)
    visor_left = int(size * 0.5)
    visor_right = int(size * 0.72)
    img[visor_top:visor_bot, visor_left:visor_right] = torch.tensor([0.6, 0.85, 0.9])

    # Backpack: small rectangle on the left
    bp_top = int(size * 0.35)
    bp_bot = int(size * 0.65)
    bp_left = int(size * 0.12)
    bp_right = int(size * 0.28)
    img[bp_top:bp_bot, bp_left:bp_right] = torch.tensor([0.15, 0.08, 0.7])

    # Legs: two rectangles at the bottom
    leg_top = int(size * 0.75)
    leg_bot = int(size * 0.92)
    leg1_left = int(size * 0.28)
    leg1_right = int(size * 0.45)
    leg2_left = int(size * 0.55)
    leg2_right = int(size * 0.72)
    img[leg_top:leg_bot, leg1_left:leg1_right] = torch.tensor([0.2, 0.1, 0.8])
    img[leg_top:leg_bot, leg2_left:leg2_right] = torch.tensor([0.2, 0.1, 0.8])

    # Gap between legs
    gap_left = int(size * 0.45)
    gap_right = int(size * 0.55)
    img[leg_top:leg_bot, gap_left:gap_right] = torch.tensor([0.95, 0.95, 0.95])

    # Round the body with a simple distance-from-center mask
    yy = torch.arange(size, device=device, dtype=torch.float32).unsqueeze(1)
    xx = torch.arange(size, device=device, dtype=torch.float32).unsqueeze(0)
    # Elliptical mask for body region
    ey = (yy - cy) / (size * 0.35)
    ex = (xx - cx) / (size * 0.28)
    dist = ey ** 2 + ex ** 2
    outside = dist > 1.0
    # Only apply to body region (not visor or backpack)
    body_region = (yy >= body_top) & (yy < body_bot - 5)
    mask_reset = outside & body_region & (xx >= body_left) & (xx < body_right)
    img[mask_reset.expand_as(img[:, :, 0]).unsqueeze(-1).expand_as(img)] = 0.95

    return img.clamp(0.0, 1.0)


def to_grayscale(img: torch.Tensor) -> torch.Tensor:
    """Convert (H, W, 3) RGB to (H, W) grayscale using luminance weights."""
    return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114


def main():
    parser = argparse.ArgumentParser(
        description="Demo 1: Spectral Normal Mapping — visual echoes and second-order interactions"
    )
    parser.add_argument(
        "--image", type=Path, default=None,
        help="Input image path. If not provided, uses a synthetic shape."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("demo_output"),
        help="Output directory (default: demo_output/)"
    )
    parser.add_argument(
        "--resolution", type=int, default=512,
        help="Egg render resolution (default: 512)"
    )
    parser.add_argument(
        "--num-eigenvectors", type=int, default=8,
        help="Number of eigenvectors to compute (default: 8)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cpu' or 'cuda' (default: auto-detect)"
    )
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # Step 1: Load or create image
    # ===================================================================
    t0 = time.time()
    if args.image is not None:
        print(f"[1/8] Loading image: {args.image}")
        image = load_image(args.image, device=device)
    else:
        # Look for a real image in input_images/
        project_root = Path(__file__).parent.parent
        candidate = project_root / "input_images" / "toof.png"
        if candidate.exists():
            print(f"[1/8] Loading default image: {candidate}")
            image = load_image(candidate, device=device)
        else:
            print("[1/8] Creating synthetic Among Us shape (no input image found)")
            image = create_synthetic_amongus(size=64, device=device)

    H, W = image.shape[0], image.shape[1]
    print(f"       Image size: {W}x{H}")
    t1 = time.time()
    print(f"       Time: {t1 - t0:.3f}s")

    # ===================================================================
    # Step 2: Build image Laplacian via ImageLaplacianBuilder
    # ===================================================================
    print(f"\n[2/8] Building image Laplacian ({H*W} nodes)...")
    t2 = time.time()

    # Use smaller radii for larger images to keep memory manageable
    if H * W > 10000:
        laplacian_builder = ImageLaplacianBuilder(
            radii=[1, 2, 3],
            radius_weights=[1.0, 0.6, 0.4],
            edge_threshold=0.15,
        ).to(device)
    else:
        laplacian_builder = ImageLaplacianBuilder(
            radii=[1, 2, 3, 4, 5, 6],
            radius_weights=[1.0, 0.6, 0.4, 0.3, 0.2, 0.1],
            edge_threshold=0.15,
        ).to(device)

    laplacian = laplacian_builder(image)
    n_edges = laplacian._nnz()
    t3 = time.time()
    print(f"       Laplacian: {laplacian.shape[0]}x{laplacian.shape[1]}, {n_edges} non-zeros")
    print(f"       Time: {t3 - t2:.3f}s")

    # ===================================================================
    # Step 3: Compute eigenvectors via GraphEmbedding
    # ===================================================================
    print(f"\n[3/8] Computing {args.num_eigenvectors} eigenvectors via Lanczos...")
    t4 = time.time()

    graph_embedding = GraphEmbedding(
        num_eigenvectors=args.num_eigenvectors,
        lanczos_iterations=30,
    ).to(device)

    eigvecs, eigvals = graph_embedding(laplacian)
    t5 = time.time()
    print(f"       Eigenvectors shape: {eigvecs.shape}")
    print(f"       Eigenvalues: {eigvals[:4].tolist()}")
    print(f"       Time: {t5 - t4:.3f}s")

    # Extract Fiedler vector as 2D height field
    fiedler = eigvecs[:, 0].reshape(H, W)
    print(f"       Fiedler range: [{fiedler.min().item():.4f}, {fiedler.max().item():.4f}]")

    # ===================================================================
    # Step 4: Convert Fiedler height to normal map
    # ===================================================================
    print("\n[4/8] Converting height field to normals...")
    t6 = time.time()

    height_to_normals = HeightToNormals(strength=2.0).to(device)
    normals = height_to_normals(fiedler)
    t7 = time.time()
    print(f"       Normals shape: {normals.shape}")
    print(f"       Time: {t7 - t6:.3f}s")

    # Save the normal map as a diagnostic
    save_image(normals, args.output_dir / "demo1_normals.png")

    # ===================================================================
    # Step 5: Render on egg surface — "visual echoes"
    # ===================================================================
    print(f"\n[5/8] Rendering egg surface (resolution={args.resolution})...")
    t8 = time.time()

    egg_renderer = EggSurfaceRenderer(
        resolution=args.resolution,
        egg_factor=0.25,
        bump_strength=1.0,
        light_dir=(0.5, 0.7, 1.0),
    ).to(device)

    # Ensure image is (H, W, 3) for the renderer
    texture = image.clone()
    if texture.dim() == 2:
        texture = texture.unsqueeze(-1).expand(-1, -1, 3)

    rendered_1 = egg_renderer(texture, normals)
    t9 = time.time()
    print(f"       Render shape: {rendered_1.shape}")
    print(f"       Time: {t9 - t8:.3f}s")

    save_path_1 = save_image(rendered_1, args.output_dir / "demo1_visual_echoes.png")
    print(f"       Saved: {save_path_1}")

    # ===================================================================
    # Step 6: Apply SpectralShaderBlock using the Fiedler vector
    # ===================================================================
    print("\n[6/8] Applying SpectralShaderBlock (spectral shader pass)...")
    t10 = time.time()

    shader_block = SpectralShaderBlock(config={
        "gate_sharpness": 10.0,
        "dilation_radius": 2,
        "effect_strength": 1.0,
        "shadow_offset": 7.0,
        "translation_strength": 20.0,
    }).to(device)

    # Ensure image is (H, W, 3) float32 in [0, 1]
    shader_input = image.clone()
    if shader_input.dim() == 2:
        shader_input = shader_input.unsqueeze(-1).expand(-1, -1, 3)

    shaded = shader_block(shader_input, fiedler)
    t11 = time.time()
    print(f"       Shaded shape: {shaded.shape}")
    print(f"       Time: {t11 - t10:.3f}s")

    save_image(shaded, args.output_dir / "demo1_shaded.png")

    # ===================================================================
    # Step 7: Convert shaded result to height -> normals
    # ===================================================================
    print("\n[7/8] Computing second-order normals from shaded output...")
    t12 = time.time()

    shaded_gray = to_grayscale(shaded)
    normals_2 = height_to_normals(shaded_gray)
    t13 = time.time()
    print(f"       Second-order normals shape: {normals_2.shape}")
    print(f"       Time: {t13 - t12:.3f}s")

    save_image(normals_2, args.output_dir / "demo1_normals_second_order.png")

    # ===================================================================
    # Step 8: Render again — "second-order interactions"
    # ===================================================================
    print(f"\n[8/8] Rendering second-order egg surface...")
    t14 = time.time()

    rendered_2 = egg_renderer(shaded, normals_2)
    t15 = time.time()
    print(f"       Render shape: {rendered_2.shape}")
    print(f"       Time: {t15 - t14:.3f}s")

    save_path_2 = save_image(rendered_2, args.output_dir / "demo1_second_order.png")
    print(f"       Saved: {save_path_2}")

    # ===================================================================
    # Summary
    # ===================================================================
    total = t15 - t0
    print("\n" + "=" * 60)
    print("Demo 1: Spectral Normal Mapping -- Complete")
    print("=" * 60)
    print(f"  Image:            {W}x{H} ({H*W} pixels)")
    print(f"  Eigenvectors:     {args.num_eigenvectors}")
    print(f"  Render resolution: {args.resolution}")
    print(f"  Total time:       {total:.2f}s")
    print()
    print("  Phase breakdown:")
    print(f"    Load image:       {t1 - t0:.3f}s")
    print(f"    Build Laplacian:  {t3 - t2:.3f}s")
    print(f"    Lanczos:          {t5 - t4:.3f}s")
    print(f"    Height->normals:  {t7 - t6:.3f}s")
    print(f"    Egg render #1:    {t9 - t8:.3f}s")
    print(f"    Shader block:     {t11 - t10:.3f}s")
    print(f"    2nd-order normals:{t13 - t12:.3f}s")
    print(f"    Egg render #2:    {t15 - t14:.3f}s")
    print()
    print("  Outputs:")
    print(f"    Visual echoes:       {args.output_dir}/demo1_visual_echoes_*.png")
    print(f"    Shaded image:        {args.output_dir}/demo1_shaded_*.png")
    print(f"    Second-order render: {args.output_dir}/demo1_second_order_*.png")
    print(f"    Normal maps:         {args.output_dir}/demo1_normals_*.png")
    print()
    print("  Composition chain:")
    print("    image -> Laplacian -> eigenvectors -> height -> normals -> 3D render")
    print("    image + Fiedler -> SpectralShaderBlock -> grayscale -> normals -> 3D render")


if __name__ == "__main__":
    main()
