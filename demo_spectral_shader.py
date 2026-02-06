#!/usr/bin/env python3
"""
Spectral Shader Demo - Generate proof-of-concept renders.

This script demonstrates graph spectral attention mechanisms:
- Self-attention: autoregressive refinement on single image
- Cross-attention: texture transfer between image pairs

Uses SpectralShader.from_config() from spectral_shader_model.py (even_cuter path).

Usage:
    uv run python demo_spectral_shader.py                    # Run all demos
    uv run python demo_spectral_shader.py --self             # Self-attention only
    uv run python demo_spectral_shader.py --cross            # Cross-attention only
    uv run python demo_spectral_shader.py --no-overwrite     # Z-buffer masking mode
"""
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from image_io import save_image, load_image
from spectral_shader_model import SpectralShader


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("demo_output")


# ADSR envelope for autoregressive passes
# Attack: aggressive growth | Decay: moderate | Sustain: gentle | Release: minimal
ADSR_CONFIGS = [
    {'thicken_modulation': 0.2, 'kernel_sigma_ratio': 0.8, 'fill_threshold': 0.05, 'dilation_radius': 3},
    {'thicken_modulation': 0.3, 'kernel_sigma_ratio': 0.6, 'fill_threshold': 0.1, 'dilation_radius': 2},
    {'thicken_modulation': 0.5, 'kernel_sigma_ratio': 0.5, 'fill_threshold': 0.15, 'dilation_radius': 2},
    {'thicken_modulation': 0.7, 'kernel_sigma_ratio': 0.4, 'fill_threshold': 0.2, 'dilation_radius': 1},
]

BASE_CONFIG = {
    'effect_strength': 1.0,
    'gate_sharpness': 10.0,
    'translation_strength': 15.0,
    'shadow_offset': 5.0,
    'min_segment_pixels': 25,
    'max_segments': 40,
}


def make_pass_model(pass_idx: int) -> SpectralShader:
    """Build a SpectralShader configured for the given ADSR pass index."""
    config = {**BASE_CONFIG, **ADSR_CONFIGS[min(pass_idx, len(ADSR_CONFIGS) - 1)]}
    return SpectralShader.from_config(config).to(DEVICE)


def run_self_attention(img_name: str, n_passes: int = 4, no_overwrite: bool = False):
    """
    Self-attention: autoregressive refinement on single image.

    The image attends to itself, with copy/rotate heads extracting
    and repositioning spectral segments iteratively.
    """
    print(f"\n{'='*60}")
    print(f"Self-Attention: {img_name}")
    print(f"{'='*60}")

    img = load_image(INPUT_DIR / img_name, DEVICE)
    print(f"Image: {img.shape[1]}x{img.shape[0]}")

    current = img.clone()
    passes = []

    for i in range(n_passes):
        phase = ['Attack', 'Decay', 'Sustain', 'Release'][min(i, 3)]
        model = make_pass_model(i)

        print(f"  Pass {i+1}/{n_passes} ({phase})")
        # Single-image, single pass via forward()
        result, _intermediates = model(current, n_passes=1)
        passes.append(result.clone())

        stem = img_name.replace('.png', '')
        mode = '_nooverwrite' if no_overwrite else ''
        save_image(result, OUTPUT_DIR / f"{stem}_self{mode}_pass{i+1}.png")
        current = result

    return passes


def run_cross_attention(target_name: str, source_name: str, n_passes: int = 4, no_overwrite: bool = False):
    """
    Cross-attention: texture transfer between image pairs.

    Target (A) provides topology/position queries.
    Source (B) provides content/texture to transfer.
    Spectral signatures match segments across images.
    """
    print(f"\n{'='*60}")
    print(f"Cross-Attention: {target_name} x {source_name}")
    print(f"{'='*60}")

    img_A = load_image(INPUT_DIR / target_name, DEVICE)
    img_B = load_image(INPUT_DIR / source_name, DEVICE)

    print(f"Target (A): {img_A.shape[1]}x{img_A.shape[0]}")
    print(f"Source (B): {img_B.shape[1]}x{img_B.shape[0]}")

    current = img_A.clone()
    passes = []

    for i in range(n_passes):
        phase = ['Attack', 'Decay', 'Sustain', 'Release'][min(i, 3)]
        model = make_pass_model(i)

        print(f"  Pass {i+1}/{n_passes} ({phase})")
        # Two-image, single pass via forward()
        result, _intermediates = model(current, image_b=img_B, n_passes=1)
        passes.append(result.clone())

        stem_a = target_name.replace('.png', '')
        stem_b = source_name.replace('.png', '')
        mode = '_nooverwrite' if no_overwrite else ''
        save_image(result, OUTPUT_DIR / f"{stem_a}_x_{stem_b}{mode}_pass{i+1}.png")
        current = result

    return passes


def make_comparison_grid(images: list, labels: list, cols: int = None) -> np.ndarray:
    """Stitch images into a labeled comparison grid."""
    n = len(images)
    if cols is None:
        cols = n
    rows = (n + cols - 1) // cols

    # Convert to numpy
    imgs = []
    for img in images:
        if isinstance(img, torch.Tensor):
            arr = (img.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = np.array(img) if hasattr(img, '__array__') else img
        imgs.append(arr)

    # Target size
    target_h = max(img.shape[0] for img in imgs)
    target_w = max(img.shape[1] for img in imgs)

    label_h = 30
    cell_h = target_h + label_h
    cell_w = target_w

    grid = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 255

    for i, (img, label) in enumerate(zip(imgs, labels)):
        r, c = i // cols, i % cols
        y_off = r * cell_h + label_h
        x_off = c * cell_w

        h, w = img.shape[:2]
        y_pad = (target_h - h) // 2
        x_pad = (target_w - w) // 2
        grid[y_off + y_pad:y_off + y_pad + h, x_off + x_pad:x_off + x_pad + w] = img

    return grid


def main():
    parser = argparse.ArgumentParser(description="Spectral Shader Demo")
    parser.add_argument("--self", action="store_true", help="Run self-attention demos only")
    parser.add_argument("--cross", action="store_true", help="Run cross-attention demos only")
    parser.add_argument("--no-overwrite", action="store_true",
                       help="Z-buffer mode: prevent copy heads from overwriting existing pixels")
    parser.add_argument("--passes", type=int, default=4, help="Number of ADSR passes")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_all = not args.self and not args.cross

    print(f"Device: {DEVICE}")
    print(f"Mode: {'Z-buffer (no overwrite)' if args.no_overwrite else 'Standard (overwrite allowed)'}")

    results = {}

    # Self-attention demos
    if run_all or args.self:
        results['toof_self'] = run_self_attention("toof.png", args.passes, args.no_overwrite)
        results['1bit_self'] = run_self_attention("1bit redraw.png", args.passes, args.no_overwrite)

    # Cross-attention demos
    if run_all or args.cross:
        results['toof_x_1bit'] = run_cross_attention(
            "toof.png", "1bit redraw.png", args.passes, args.no_overwrite
        )
        results['tonegraph_x_snek'] = run_cross_attention(
            "red-tonegraph.png", "snek-heavy.png", args.passes, args.no_overwrite
        )
        results['tonegraph_x_enso'] = run_cross_attention(
            "red-tonegraph.png", "mspaint-enso-i-couldnt-forget.png", args.passes, args.no_overwrite
        )

    print(f"\n{'='*60}")
    print("Demo complete! Outputs saved to demo_output/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
