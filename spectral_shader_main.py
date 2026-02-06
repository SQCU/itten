#!/usr/bin/env python3
"""
Spectral Shader Main - Clean entry point for spectral graph operations.

Composable CLI: any combination of single/paired images with any number of AR passes.
Uses SpectralShader.from_config() from spectral_shader_model.py (even_cuter path).

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

import torch

from spectral_shader_model import SpectralShader
from image_io import save_image, load_image as _load_image


# Quality presets based on Fiedler benchmark results
# Lanczos-30: 0.84 correlation (quality), Lanczos-15: 0.48 (fast)
QUALITY_PRESETS = {
    "fast": {
        "lanczos_iterations": 15,  # 0.48 correlation, 2x faster
        "tile_size": 96,           # Larger tiles = fewer computations
        "overlap": 8,              # Less overlap
    },
    "standard": {
        "lanczos_iterations": 30,  # 0.84 correlation (optimal)
        "tile_size": 64,
        "overlap": 16,
    },
    "quality": {
        "lanczos_iterations": 50,  # Ground truth
        "tile_size": 48,           # Smaller tiles = finer detail
        "overlap": 24,             # More overlap for smooth blending
    },
}

# Default shader config — accepted by SpectralShader.from_config()
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
    "lanczos_iterations": 30,  # Standard quality
    "min_segment_pixels": 25,
    "max_segments": 40,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(path: Path) -> torch.Tensor:
    """Load image as RGB float32 [0,1] torch tensor on GPU if available."""
    return _load_image(path, device=DEVICE)


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
    parser.add_argument("-q", "--quality", choices=["fast", "standard", "quality"],
                        default="standard",
                        help="Quality preset: fast (2x speed), standard, quality (best)")

    args = parser.parse_args()
    cfg = DEFAULT_CONFIG.copy()

    # Apply quality preset
    if args.quality in QUALITY_PRESETS:
        cfg.update(QUALITY_PRESETS[args.quality])
        print(f"Quality preset: {args.quality}")

    # Build model from config (even_cuter path)
    model = SpectralShader.from_config(cfg).to(DEVICE)
    print(f"Model: {model.__class__.__name__}")

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

    # Run via SpectralShader.forward()
    start_time = time.time()

    result, intermediates = model(
        image_a,
        image_b=image_b,
        n_passes=args.passes,
        decay=args.decay,
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
