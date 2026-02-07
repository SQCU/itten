#!/usr/bin/env python3
"""
Pipeline benchmark: 20 cross-attention pairs through SpectralShader.from_config().

Every pair writes its output image + per-pixel stats. A tensor op has failed if
we have no outputs of the same dimensionality that give statistical access to
what the operation did. Timing-only benchmarks can't catch a pleasing nothing.

Usage:
    uv run python benchmark_pipeline.py                  # Standard benchmark
    uv run python benchmark_pipeline.py --dtype bfloat16 # bfloat16 benchmark
    uv run python benchmark_pipeline.py --pairs 5        # Quick smoke test
"""
import argparse
import itertools
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from spectral_shader_model import SpectralShader
from image_io import load_image, save_image


INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("demo_output/benchmark")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_pairs(n_pairs: int = 20):
    """Generate cross-attention pairs from available input images."""
    images = sorted(INPUT_DIR.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No .png files in {INPUT_DIR}")

    # All ordered pairs (A, B) where A != B
    all_pairs = [(a, b) for a, b in itertools.permutations(images, 2)]
    # Deterministic subset
    return all_pairs[:n_pairs]


def tensor_stats(t: torch.Tensor) -> dict:
    """Per-output stats that catch identity, zeros, constant, and NaN failures."""
    f = t.float()
    return {
        "shape": list(t.shape),
        "mean": f.mean().item(),
        "std": f.std().item(),
        "min": f.min().item(),
        "max": f.max().item(),
        "nonzero_frac": (f.abs() > 1e-6).float().mean().item(),
        "nan_count": torch.isnan(f).sum().item(),
        "inf_count": torch.isinf(f).sum().item(),
    }


def diff_stats(result: torch.Tensor, input_a: torch.Tensor) -> dict:
    """Stats on (result - input_a) to detect identity/near-identity ops."""
    # Crop to common size (cross-attention may change resolution)
    h = min(result.shape[0], input_a.shape[0])
    w = min(result.shape[1], input_a.shape[1])
    d = (result[:h, :w].float() - input_a[:h, :w].float())
    return {
        "diff_mean_abs": d.abs().mean().item(),
        "diff_max_abs": d.abs().max().item(),
        "diff_nonzero_frac": (d.abs() > 1e-6).float().mean().item(),
        "pixels_changed_pct": (d.abs() > 1.0 / 255).float().mean().item() * 100,
    }


def run_benchmark(n_pairs: int, compute_dtype: torch.dtype, n_passes: int):
    pairs = get_image_pairs(n_pairs)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dtype_tag = "bf16" if compute_dtype == torch.bfloat16 else "f32"
    run_dir = OUTPUT_DIR / f"{ts}_{dtype_tag}_{n_pairs}pairs"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Pairs: {len(pairs)}")
    print(f"Passes per pair: {n_passes}")
    print(f"compute_dtype: {compute_dtype}")
    print(f"Output dir: {run_dir}")
    print()

    config = {"compute_dtype": compute_dtype}
    model = SpectralShader.from_config(config).to(DEVICE)

    # Warmup (1 pair, not timed, still write output to catch init-only bugs)
    img_a = load_image(pairs[0][0], DEVICE)
    img_b = load_image(pairs[0][1], DEVICE)
    warmup_result, _ = model(img_a, image_b=img_b, n_passes=1)
    torch.cuda.synchronize() if DEVICE.type == "cuda" else None
    save_image(warmup_result, run_dir / "warmup.png", timestamp=False)

    # Benchmark
    times = []
    all_stats = []
    for i, (path_a, path_b) in enumerate(pairs):
        img_a = load_image(path_a, DEVICE)
        img_b = load_image(path_b, DEVICE)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        result, intermediates = model(img_a, image_b=img_b, n_passes=n_passes)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # Write the image — this is the dimensionality-preserving output
        a_name = path_a.stem[:15]
        b_name = path_b.stem[:15]
        img_path = run_dir / f"{i:02d}_{a_name}_x_{b_name}.png"
        save_image(result, img_path, timestamp=False)

        # Stats on the output tensor itself
        r_stats = tensor_stats(result)
        d_stats = diff_stats(result, img_a)

        pair_record = {
            "pair": i,
            "a": path_a.name,
            "b": path_b.name,
            "h": img_a.shape[0],
            "w": img_a.shape[1],
            "elapsed_ms": round(elapsed * 1000, 1),
            "output": r_stats,
            "vs_input": d_stats,
        }
        all_stats.append(pair_record)

        times.append(elapsed)
        h, w = img_a.shape[:2]

        # Flag suspicious outputs inline
        flags = []
        if r_stats["std"] < 0.01:
            flags.append("LOW_VARIANCE")
        if r_stats["nan_count"] > 0:
            flags.append("HAS_NAN")
        if d_stats["pixels_changed_pct"] < 1.0:
            flags.append("NEAR_IDENTITY")
        if r_stats["nonzero_frac"] < 0.1:
            flags.append("MOSTLY_ZERO")
        flag_str = f"  !! {' '.join(flags)}" if flags else ""

        print(
            f"  [{i+1:2d}/{len(pairs)}] {a_name:>15s} x {b_name:<15s}  "
            f"{h:3d}x{w:3d}  {elapsed*1000:7.0f}ms  "
            f"std={r_stats['std']:.3f} diff={d_stats['pixels_changed_pct']:.1f}%{flag_str}"
        )

    # Summary
    total = sum(times)
    mean = total / len(times)
    median = sorted(times)[len(times) // 2]
    fastest = min(times)
    slowest = max(times)

    # Aggregate health checks
    all_stds = [s["output"]["std"] for s in all_stats]
    all_diffs = [s["vs_input"]["pixels_changed_pct"] for s in all_stats]
    total_nans = sum(s["output"]["nan_count"] for s in all_stats)

    summary = {
        "run": ts,
        "dtype": dtype_tag,
        "n_pairs": len(pairs),
        "n_passes": n_passes,
        "device": str(DEVICE),
        "timing": {
            "total_s": round(total, 1),
            "mean_ms": round(mean * 1000),
            "median_ms": round(median * 1000),
            "fastest_ms": round(fastest * 1000),
            "slowest_ms": round(slowest * 1000),
        },
        "health": {
            "min_output_std": round(min(all_stds), 4),
            "mean_output_std": round(sum(all_stds) / len(all_stds), 4),
            "min_pixels_changed_pct": round(min(all_diffs), 1),
            "mean_pixels_changed_pct": round(sum(all_diffs) / len(all_diffs), 1),
            "total_nan_pixels": total_nans,
        },
        "pairs": all_stats,
    }

    # Write full stats to JSON
    stats_path = run_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"{'='*72}")
    print(f"  Total:   {total:.1f}s")
    print(f"  Mean:    {mean*1000:.0f}ms per pair")
    print(f"  Median:  {median*1000:.0f}ms per pair")
    print(f"  Fastest: {fastest*1000:.0f}ms")
    print(f"  Slowest: {slowest*1000:.0f}ms")
    print(f"  ---")
    print(f"  Output std range:     [{min(all_stds):.4f}, {max(all_stds):.4f}]")
    print(f"  Pixels changed range: [{min(all_diffs):.1f}%, {max(all_diffs):.1f}%]")
    print(f"  Total NaN pixels:     {total_nans}")
    print(f"  ---")
    print(f"  Images: {run_dir}/ ({len(pairs)} outputs + warmup)")
    print(f"  Stats:  {stats_path}")
    print(f"{'='*72}")

    # Hard failure: if everything looks like identity, say so
    if max(all_diffs) < 1.0:
        print("\n  !! WARNING: No pair changed >1% of pixels. Pipeline may be a no-op.")
    if total_nans > 0:
        print(f"\n  !! WARNING: {total_nans} NaN pixels across all outputs.")

    return times


def main():
    parser = argparse.ArgumentParser(description="Pipeline benchmark")
    parser.add_argument("--pairs", type=int, default=20, help="Number of pairs (default: 20)")
    parser.add_argument("--passes", type=int, default=1, help="AR passes per pair (default: 1)")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16"],
        default="float32",
        help="Compute dtype (default: float32)",
    )
    args = parser.parse_args()

    compute_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    run_benchmark(args.pairs, compute_dtype, args.passes)


if __name__ == "__main__":
    main()
