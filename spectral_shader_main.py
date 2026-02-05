#!/usr/bin/env python3
"""
Spectral Shader Main - Clean entry point for spectral graph operations.

Composable CLI: any combination of single/paired images with any number of AR passes.

Examples:
    # Single image, 1 pass
    python spectral_shader_main.py image.png

    # Single image, 4 AR passes
    python spectral_shader_main.py image.png -p 4

    # Two images, 1 pass (cross-attention transfer)
    python spectral_shader_main.py target.png source.png

    # Two images, 4 AR passes (each pass uses cross-attention from source)
    python spectral_shader_main.py target.png source.png -p 4
"""
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, List

import torch

from spectral_shader_ops import (
    two_image_shader_pass,
    shader_forwards,
    spectral_shader_pass,
    _compute_fiedler_from_tensor,
    adaptive_threshold,
)
from spectral_ops_fast import compute_local_eigenvectors_tiled_dither
from image_io import save_image, load_image as _load_image


# Default shader config
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
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(path: Path) -> torch.Tensor:
    """Load image as RGB float32 [0,1] torch tensor on GPU if available."""
    return _load_image(path, device=DEVICE)


def compute_fiedler(rgb: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Compute Fiedler vector from RGB via tiled eigenvector computation."""
    evecs = compute_local_eigenvectors_tiled_dither(
        rgb,
        tile_size=cfg["tile_size"],
        overlap=cfg["overlap"],
        num_eigenvectors=cfg["num_eigenvectors"],
        radii=cfg["radii"],
        radius_weights=cfg["radius_weights"],
        edge_threshold=cfg["edge_threshold"],
    )
    return evecs[:, :, 1]  # Fiedler = eigenvector index 1


def run_shader(
    image_a: torch.Tensor,
    image_b: Optional[torch.Tensor],
    fiedler_a: torch.Tensor,
    fiedler_b: Optional[torch.Tensor],
    config: dict,
) -> torch.Tensor:
    """
    Run one shader pass. Dispatches to single or two-image based on whether B is provided.
    """
    if image_b is not None and fiedler_b is not None:
        # Two-image: cross-attention transfer
        return two_image_shader_pass(image_a, image_b, fiedler_a, fiedler_b, config)
    else:
        # Single image: self-shading
        config_with_threshold = {**config, 'gate_threshold': adaptive_threshold(fiedler_a, 40.0)}
        return spectral_shader_pass(image_a, fiedler_a, config_with_threshold)


def run_autoregressive(
    image_a: torch.Tensor,
    image_b: Optional[torch.Tensor],
    n_passes: int,
    config: dict,
    decay: float = 1.0,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Autoregressive shader loop. Works for both single and two-image modes.

    In two-image mode, B stays fixed while A evolves through AR passes.
    Each pass: A' = shader(A, B) using cross-attention from B.
    """
    current = image_a.clone()
    intermediates = []

    # Compute Fiedler for B once (it doesn't change)
    fiedler_b = None
    if image_b is not None:
        print("Computing Fiedler for source image B...")
        fiedler_b = compute_fiedler(image_b, config)

    current_config = config.copy()

    for i in range(n_passes):
        print(f"Pass {i+1}/{n_passes}...")

        # Recompute Fiedler for current A (it changes each pass)
        fiedler_a = compute_fiedler(current, config)

        # Run shader
        current = run_shader(current, image_b, fiedler_a, fiedler_b, current_config)
        intermediates.append(current.clone())

        # Apply decay
        if decay < 1.0:
            current_config = {**current_config, "effect_strength": current_config["effect_strength"] * decay}

    return current, intermediates


def main():
    parser = argparse.ArgumentParser(
        description="Spectral Shader - composable single/paired images with AR passes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.png                    # Single image, 1 pass
  %(prog)s image.png -p 4               # Single image, 4 AR passes
  %(prog)s target.png source.png        # Two images, 1 pass
  %(prog)s target.png source.png -p 4   # Two images, 4 AR passes
        """
    )

    parser.add_argument("image_a", type=Path, help="Primary/target image")
    parser.add_argument("image_b", type=Path, nargs="?", default=None,
                        help="Optional source image for cross-attention transfer")
    parser.add_argument("-p", "--passes", type=int, default=1,
                        help="Number of autoregressive passes (default: 1)")
    parser.add_argument("-d", "--decay", type=float, default=0.75,
                        help="Effect decay rate per pass (default: 0.75)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output path (default: auto-generated)")
    parser.add_argument("--save-intermediates", action="store_true",
                        help="Save intermediate passes")

    args = parser.parse_args()
    cfg = DEFAULT_CONFIG.copy()

    # Load images
    print(f"Loading {args.image_a.name}...")
    image_a = load_image(args.image_a)
    print(f"  A: {image_a.shape[1]}x{image_a.shape[0]}")

    image_b = None
    if args.image_b is not None:
        print(f"Loading {args.image_b.name}...")
        image_b = load_image(args.image_b)
        print(f"  B: {image_b.shape[1]}x{image_b.shape[0]}")

    # Describe what we're doing
    mode = "two-image" if image_b is not None else "single-image"
    print(f"\nMode: {mode}, {args.passes} pass{'es' if args.passes > 1 else ''}")

    # Run
    start_time = time.time()

    if args.passes == 1 and image_b is None:
        # Fast path: single image, single pass
        print("Computing Fiedler...")
        fiedler_a = compute_fiedler(image_a, cfg)
        print("Running shader...")
        result = run_shader(image_a, None, fiedler_a, None, cfg)
        intermediates = [result]
    else:
        # AR loop (handles both single and two-image)
        result, intermediates = run_autoregressive(
            image_a, image_b, args.passes, cfg, args.decay
        )

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f}s ({elapsed/args.passes:.2f}s per pass)")

    # Generate output path (save_image adds timestamp automatically)
    if args.output is None:
        stem = args.image_a.stem
        if image_b is not None:
            stem = f"{stem}_x_{args.image_b.stem}"
        suffix = f"_{args.passes}x" if args.passes > 1 else ""
        output = Path(f"demo_output/{stem}{suffix}.png")
    else:
        output = args.output

    output.parent.mkdir(parents=True, exist_ok=True)

    # Save final
    save_image(result, output)

    # Save intermediates if requested
    if args.save_intermediates and args.passes > 1:
        for i, inter in enumerate(intermediates):
            inter_path = output.parent / f"{output.stem}_pass{i+1}.png"
            save_image(inter, inter_path)


if __name__ == "__main__":
    main()
