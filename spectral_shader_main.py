#!/usr/bin/env python3
"""
Spectral Shader Main - Clean entry point for spectral graph operations.

Maps configs and inputs to spectral_shader_ops functions.

Supports two spectral computation modes:
1. Explicit eigenvector path (original): compute_local_eigenvectors_tiled_dither
2. Chebyshev filtering path (Phase 1-2): O(order * nnz) polynomial filtering

The Chebyshev path is ~50-70% faster for single-image shading where full
Fiedler vectors are not needed (gate/complexity computation only).
"""
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import torch

from spectral_shader_ops import (
    two_image_shader_pass,
    batch_random_pairs,
    shader_autoregressive,
    shader_forwards,
    spectral_shader_pass_chebyshev,
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
    "gate_threshold": 0.5,
    "gate_sharpness": 8.0,
    "thicken_radius": 2.5,
    "thicken_strength": 0.6,
}


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(path: Path) -> torch.Tensor:
    """Load image as RGB float32 [0,1] torch tensor on GPU if available."""
    return _load_image(path, device=DEVICE)


def compute_spectral(rgb: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Compute dither-aware tiled spectral eigenvectors from RGB (uses L2 color distance)."""
    return compute_local_eigenvectors_tiled_dither(
        rgb,
        tile_size=cfg["tile_size"],
        overlap=cfg["overlap"],
        num_eigenvectors=cfg["num_eigenvectors"],
        radii=cfg["radii"],
        radius_weights=cfg["radius_weights"],
        edge_threshold=cfg["edge_threshold"],
    )


def run_single(image_a: Path, image_b: Path, output: Path, cfg: dict):
    """Run two-image shader pass: A=topology, B=content."""
    rgb_a = load_image(image_a)
    rgb_b = load_image(image_b)

    print(f"A: {image_a.name} ({rgb_a.shape[1]}x{rgb_a.shape[0]})")
    print(f"B: {image_b.name} ({rgb_b.shape[1]}x{rgb_b.shape[0]})")

    # Compute spectral from RGB directly (L2 color distance for edge weights)
    evecs_a = compute_spectral(rgb_a, cfg)
    evecs_b = compute_spectral(rgb_b, cfg)

    # Extract Fiedler (eigenvector index 1)
    fiedler_a = evecs_a[:, :, 1]
    fiedler_b = evecs_b[:, :, 1]

    shader_config = {
        "gate_threshold": cfg["gate_threshold"],
        "gate_sharpness": cfg["gate_sharpness"],
        "thicken_radius": cfg["thicken_radius"],
        "thicken_strength": cfg["thicken_strength"],
    }

    result = two_image_shader_pass(
        image_A=rgb_a,
        image_B=rgb_b,
        fiedler_A=fiedler_a,
        fiedler_B=fiedler_b,
        config=shader_config,
    )

    save_image(result, output)


def run_batch(n_pairs: int):
    """Run batch random pairs test (uses default paths from batch_random_pairs)."""
    batch_random_pairs(n_pairs=n_pairs)


def run_autoregressive(image_path: Path, output: Path, n_passes: int, decay: float, cfg: dict, use_chebyshev: bool = True):
    """Run autoregressive shader with n_passes.

    Args:
        use_chebyshev: If True, use Chebyshev filtering (faster). If False, use explicit eigenvectors.
    """
    image_tensor = load_image(image_path)
    print(f"Image: {image_path.name} ({image_tensor.shape[1]}x{image_tensor.shape[0]})")
    print(f"Passes: {n_passes}")
    print(f"Mode: {'Chebyshev' if use_chebyshev else 'Explicit eigenvectors'}")

    shader_config = {
        "effect_strength": 1.0,
        # gate_threshold computed adaptively by shader_forwards
        "gate_sharpness": cfg["gate_sharpness"],
        "translation_strength": 15.0,
        "shadow_offset": 5.0,
        "dilation_radius": 2,
        "min_segment_pixels": 25,
        "max_segments": 40,
    }

    # Optional: decay schedule (set to None for identical passes)
    def decay_schedule(config, n):
        return {**config, "effect_strength": config["effect_strength"] * decay}

    start_time = time.time()

    final, intermediates = shader_autoregressive(
        image_tensor,
        n_passes=n_passes,
        config=shader_config,
        schedule_fn=decay_schedule if decay < 1.0 else None,
        use_chebyshev=use_chebyshev,
    )

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f}s ({elapsed/n_passes:.2f}s per pass)")

    # Save final
    save_image(final, output)

    # Save intermediates
    stem = output.stem
    for i, inter in enumerate(intermediates):
        inter_path = output.parent / f"{stem}_pass{i+1}.png"
        save_image(inter, inter_path)


def run_compare(image_path: Path, output_dir: Path, n_passes: int, decay: float, cfg: dict):
    """Run both Chebyshev and explicit paths and compare timing/output."""
    image_tensor = load_image(image_path)
    print(f"Image: {image_path.name} ({image_tensor.shape[1]}x{image_tensor.shape[0]})")
    print(f"Passes: {n_passes}")
    print("=" * 60)
    print("COMPARISON: Chebyshev vs Explicit Eigenvectors")
    print("=" * 60)

    shader_config = {
        "effect_strength": 1.0,
        "gate_sharpness": cfg["gate_sharpness"],
        "translation_strength": 15.0,
        "shadow_offset": 5.0,
        "dilation_radius": 2,
        "min_segment_pixels": 25,
        "max_segments": 40,
    }

    def decay_schedule(config, n):
        return {**config, "effect_strength": config["effect_strength"] * decay}

    # Run Chebyshev path
    print("\n1. Running Chebyshev path...")
    start_cheb = time.time()
    final_cheb, _ = shader_autoregressive(
        image_tensor.clone(),
        n_passes=n_passes,
        config=shader_config.copy(),
        schedule_fn=decay_schedule if decay < 1.0 else None,
        use_chebyshev=True,
    )
    time_cheb = time.time() - start_cheb
    print(f"   Chebyshev time: {time_cheb:.2f}s ({time_cheb/n_passes:.2f}s per pass)")

    # Run explicit path
    print("\n2. Running explicit eigenvector path...")
    start_expl = time.time()
    final_expl, _ = shader_autoregressive(
        image_tensor.clone(),
        n_passes=n_passes,
        config=shader_config.copy(),
        schedule_fn=decay_schedule if decay < 1.0 else None,
        use_chebyshev=False,
    )
    time_expl = time.time() - start_expl
    print(f"   Explicit time: {time_expl:.2f}s ({time_expl/n_passes:.2f}s per pass)")

    # Compare
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    speedup = time_expl / time_cheb if time_cheb > 0 else 0
    print(f"Chebyshev: {time_cheb:.2f}s")
    print(f"Explicit:  {time_expl:.2f}s")
    print(f"Speedup:   {speedup:.2f}x")
    print(f"Time saved: {time_expl - time_cheb:.2f}s ({100*(1 - time_cheb/time_expl):.1f}%)")

    # Compute output difference
    diff = torch.abs(final_cheb - final_expl)
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    print(f"\nOutput difference:")
    print(f"  Mean absolute: {mean_diff:.6f}")
    print(f"  Max absolute:  {max_diff:.6f}")

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    cheb_path = output_dir / f"{stem}_chebyshev.png"
    expl_path = output_dir / f"{stem}_explicit.png"
    diff_path = output_dir / f"{stem}_diff.png"

    save_image(final_cheb, cheb_path)
    save_image(final_expl, expl_path)

    # Save amplified difference visualization
    diff_vis = (diff * 10).clamp(0, 1)  # Amplify for visibility
    save_image(diff_vis, diff_path)

    return time_cheb, time_expl, mean_diff, max_diff


def main():
    parser = argparse.ArgumentParser(description="Spectral Shader")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Single pass
    p_single = sub.add_parser("single", help="Two-image shader pass")
    p_single.add_argument("image_a", type=Path, help="Topology source")
    p_single.add_argument("image_b", type=Path, help="Content source")
    p_single.add_argument("-o", "--output", type=Path, default=None)

    # Batch mode
    p_batch = sub.add_parser("batch", help="Batch random pairs")
    p_batch.add_argument("-n", "--pairs", type=int, default=10)

    # Autoregressive mode
    p_auto = sub.add_parser("auto", help="Autoregressive multi-pass shader")
    p_auto.add_argument("image", type=Path, help="Input image")
    p_auto.add_argument("-o", "--output", type=Path, default=None)
    p_auto.add_argument("-p", "--passes", type=int, default=4, help="Number of passes (2-8)")
    p_auto.add_argument("-d", "--decay", type=float, default=0.75, help="Effect decay rate per pass")
    p_auto.add_argument("--chebyshev", action="store_true",
                       help="Use global Chebyshev filtering (WARNING: not equivalent to explicit)")
    p_auto.add_argument("--explicit", action="store_true",
                       help="Use explicit tiled eigenvectors (default, correct algorithm)")

    # Comparison mode (Phase 1-2 testing)
    p_compare = sub.add_parser("compare", help="Compare Chebyshev vs explicit eigenvector paths")
    p_compare.add_argument("image", type=Path, help="Input image")
    p_compare.add_argument("-o", "--output-dir", type=Path, default=None)
    p_compare.add_argument("-p", "--passes", type=int, default=4, help="Number of passes")
    p_compare.add_argument("-d", "--decay", type=float, default=0.75, help="Effect decay rate per pass")

    args = parser.parse_args()
    cfg = DEFAULT_CONFIG.copy()

    if args.cmd == "single":
        out = args.output or Path(f"demo_output/shader_{datetime.now():%Y%m%d_%H%M%S}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        run_single(args.image_a, args.image_b, out, cfg)

    elif args.cmd == "batch":
        run_batch(args.pairs)

    elif args.cmd == "auto":
        out = args.output or Path(f"demo_output/auto_{args.passes}x_{datetime.now():%Y%m%d_%H%M%S}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        # Default to explicit (correct algorithm). Only use chebyshev if explicitly requested.
        use_chebyshev = args.chebyshev and not args.explicit
        run_autoregressive(args.image, out, args.passes, args.decay, cfg, use_chebyshev=use_chebyshev)

    elif args.cmd == "compare":
        out_dir = args.output_dir or Path(f"demo_output/compare_{datetime.now():%Y%m%d_%H%M%S}")
        run_compare(args.image, out_dir, args.passes, args.decay, cfg)


if __name__ == "__main__":
    main()
