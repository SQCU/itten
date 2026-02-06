"""
Diagnostic: Shader B cross-attention mask validation.

Tests whether Shader B deposits fragments into negative space (non-contour areas)
rather than overwriting contour pixels.

Usage:
    uv run test_shader_b_diagnostic.py --label before
    uv run test_shader_b_diagnostic.py --label after

Compares results if both 'before' and 'after' directories exist.
"""

import sys
import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from image_io import load_image, save_image
from spectral_shader_ops_cuter import (
    two_image_shader_pass,
    to_grayscale,
    detect_contours,
)
from spectral_ops_fast import compute_local_eigenvectors_tiled_dither


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("demo_output/shader_b_diag")

ADSR_CONFIGS = [
    {"thicken_modulation": 0.2, "kernel_sigma_ratio": 0.8, "fill_threshold": 0.05, "dilation_radius": 3},
    {"thicken_modulation": 0.3, "kernel_sigma_ratio": 0.6, "fill_threshold": 0.1, "dilation_radius": 2},
    {"thicken_modulation": 0.5, "kernel_sigma_ratio": 0.5, "fill_threshold": 0.15, "dilation_radius": 2},
    {"thicken_modulation": 0.7, "kernel_sigma_ratio": 0.4, "fill_threshold": 0.2, "dilation_radius": 1},
]

BASE_CONFIG = {
    "effect_strength": 1.0,
    "gate_sharpness": 10.0,
    "translation_strength": 15.0,
    "shadow_offset": 5.0,
}


def compute_fiedler(img: torch.Tensor) -> torch.Tensor:
    """Compute Fiedler vector (2nd eigenvector of graph Laplacian)."""
    evecs = compute_local_eigenvectors_tiled_dither(
        img, tile_size=64, overlap=16, num_eigenvectors=4
    )
    return evecs[:, :, 1]


def compute_pass_metrics(output: torch.Tensor, original: torch.Tensor, contours: torch.Tensor):
    """Compute diagnostic metrics for a single pass."""
    diff = (output - original).abs()  # (H, W, 3)
    max_channel_diff = diff.max(dim=-1).values  # (H, W)

    nonzero_mask = max_channel_diff > (1.0 / 255.0)
    significant_mask = max_channel_diff > (10.0 / 255.0)

    nonzero_count = nonzero_mask.sum().item()
    significant_count = significant_mask.sum().item()
    total_energy = (diff ** 2).sum().item()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()

    # Fraction of changed pixels that are on contours vs negative space
    contours_bool = contours.bool()
    changed_on_contour = (nonzero_mask & contours_bool).sum().item()
    changed_off_contour = (nonzero_mask & ~contours_bool).sum().item()
    total_changed = changed_on_contour + changed_off_contour
    frac_on_contour = changed_on_contour / max(total_changed, 1)
    frac_off_contour = changed_off_contour / max(total_changed, 1)

    return {
        "nonzero_count": nonzero_count,
        "significant_count": significant_count,
        "total_energy": total_energy,
        "mean_diff": mean_diff,
        "max_diff": max_diff,
        "changed_on_contour": changed_on_contour,
        "changed_off_contour": changed_off_contour,
        "frac_on_contour": frac_on_contour,
        "frac_off_contour": frac_off_contour,
    }


def run_diagnostic(label: str):
    """Run the full 4-pass diagnostic and save outputs."""
    out_dir = OUTPUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Shader B Cross-Attention Diagnostic  [{label.upper()}]")
    print(f"{'='*70}")

    # Load images
    img_A = load_image(INPUT_DIR / "red-tonegraph.png", DEVICE)
    img_B = load_image(INPUT_DIR / "snek-heavy.png", DEVICE)
    original_A = img_A.clone()

    print(f"Image A (red-tonegraph): {img_A.shape[1]}x{img_A.shape[0]}")
    print(f"Image B (snek-heavy):    {img_B.shape[1]}x{img_B.shape[0]}")

    # Compute original contours for the fraction analysis
    gray_A = to_grayscale(original_A)
    contours_A = detect_contours(gray_A)
    contour_count = contours_A.sum().item()
    total_pixels = contours_A.numel()
    print(f"Original contour pixels: {int(contour_count)} / {total_pixels} ({100*contour_count/total_pixels:.1f}%)")

    # Precompute source Fiedler (constant across passes)
    print("\nComputing Fiedler for source (B)...")
    fiedler_B = compute_fiedler(img_B)

    current = img_A.clone()
    all_metrics = []

    for i in range(4):
        phase = ["Attack", "Decay", "Sustain", "Release"][i]
        cfg = {**BASE_CONFIG, **ADSR_CONFIGS[i]}

        print(f"\n  Pass {i+1}/4 ({phase})")

        # Compute Fiedler for current (evolving) image
        fiedler_current = compute_fiedler(current)

        # Run the cuter two-image shader pass
        result = two_image_shader_pass(current, img_B, fiedler_current, fiedler_B, cfg)

        # Compute metrics against the ORIGINAL image A (not current)
        metrics = compute_pass_metrics(result, original_A, contours_A)
        all_metrics.append(metrics)

        # Save output
        save_image(result, out_dir / f"pass{i+1}_output.png", timestamp=False)

        # Save amplified diff
        diff = (result - original_A).abs()
        diff_amplified = (diff * 5.0).clamp(0, 1)
        save_image(diff_amplified, out_dir / f"pass{i+1}_diff.png", timestamp=False)

        # Save binary diff mask
        max_channel_diff = diff.max(dim=-1).values
        binary_mask = (max_channel_diff > (1.0 / 255.0)).float()
        binary_rgb = binary_mask.unsqueeze(-1).expand(-1, -1, 3)
        save_image(binary_rgb, out_dir / f"pass{i+1}_diff_binary.png", timestamp=False)

        current = result

    # Print metrics table
    print(f"\n{'='*70}")
    print(f"  METRICS TABLE  [{label.upper()}]")
    print(f"{'='*70}")
    print(f"{'Pass':<6} {'NonZero':>10} {'Signif':>10} {'Energy':>12} {'MeanDiff':>10} {'MaxDiff':>8} {'OnCont%':>8} {'OffCont%':>9}")
    print("-" * 79)
    for i, m in enumerate(all_metrics):
        print(
            f"  {i+1:<4} {m['nonzero_count']:>10,} {m['significant_count']:>10,} "
            f"{m['total_energy']:>12.2f} {m['mean_diff']:>10.6f} {m['max_diff']:>8.4f} "
            f"{100*m['frac_on_contour']:>7.1f}% {100*m['frac_off_contour']:>8.1f}%"
        )

    # Trend analysis
    print(f"\n  TREND ANALYSIS:")
    nz = [m["nonzero_count"] for m in all_metrics]
    increasing = all(nz[i] <= nz[i + 1] for i in range(len(nz) - 1))
    if increasing:
        print(f"  [PASS] Non-zero pixel count increases across passes (Shader B fills more negative space)")
    else:
        print(f"  [WARN] Non-zero pixel count does NOT monotonically increase")
    for i in range(1, len(nz)):
        delta = nz[i] - nz[i - 1]
        pct = 100 * delta / max(nz[i - 1], 1)
        print(f"    Pass {i} -> {i+1}: {delta:+,} pixels ({pct:+.1f}%)")

    # Contour vs negative-space summary
    print(f"\n  CONTOUR vs NEGATIVE SPACE (final pass):")
    m = all_metrics[-1]
    print(f"    Changed pixels on contours:     {m['changed_on_contour']:>10,} ({100*m['frac_on_contour']:.1f}%)")
    print(f"    Changed pixels in neg. space:    {m['changed_off_contour']:>10,} ({100*m['frac_off_contour']:.1f}%)")
    if m["frac_off_contour"] > 0.5:
        print(f"    [PASS] Majority of changes are in negative space (expected after fix)")
    else:
        print(f"    [WARN] Majority of changes are ON contours (buggy mask?)")

    return all_metrics


def print_comparison(before_metrics, after_metrics):
    """Print side-by-side comparison of before/after metrics."""
    print(f"\n{'='*90}")
    print(f"  BEFORE vs AFTER COMPARISON")
    print(f"{'='*90}")
    print(f"{'Pass':<6} {'--- BEFORE ---':>30}   {'--- AFTER ---':>30}   {'Delta':>10}")
    print(f"{'':6} {'NonZero':>10} {'OnCont%':>8} {'OffCont%':>9}   {'NonZero':>10} {'OnCont%':>8} {'OffCont%':>9}   {'NZ delta':>10}")
    print("-" * 92)
    for i in range(4):
        b = before_metrics[i]
        a = after_metrics[i]
        delta_nz = a["nonzero_count"] - b["nonzero_count"]
        print(
            f"  {i+1:<4} "
            f"{b['nonzero_count']:>10,} {100*b['frac_on_contour']:>7.1f}% {100*b['frac_off_contour']:>8.1f}%   "
            f"{a['nonzero_count']:>10,} {100*a['frac_on_contour']:>7.1f}% {100*a['frac_off_contour']:>8.1f}%   "
            f"{delta_nz:>+10,}"
        )

    print(f"\n  INTERPRETATION:")
    b_final = before_metrics[-1]
    a_final = after_metrics[-1]
    print(f"    Before fix: {100*b_final['frac_on_contour']:.1f}% changes on contours, {100*b_final['frac_off_contour']:.1f}% in negative space")
    print(f"    After  fix: {100*a_final['frac_on_contour']:.1f}% changes on contours, {100*a_final['frac_off_contour']:.1f}% in negative space")
    if a_final["frac_off_contour"] > b_final["frac_off_contour"]:
        print(f"    [PASS] Fix increased negative-space deposition as expected")
    else:
        print(f"    [WARN] Fix did not increase negative-space deposition")


def main():
    parser = argparse.ArgumentParser(description="Shader B mask diagnostic")
    parser.add_argument("--label", type=str, required=True, help="Run label (e.g. 'before' or 'after')")
    parser.add_argument("--compare", action="store_true", help="Only print comparison (skip run)")
    args = parser.parse_args()

    if args.compare:
        # Load metrics from saved files (we save them as torch tensors)
        before_path = OUTPUT_DIR / "before" / "metrics.pt"
        after_path = OUTPUT_DIR / "after" / "metrics.pt"
        if before_path.exists() and after_path.exists():
            before_metrics = torch.load(before_path, weights_only=False)
            after_metrics = torch.load(after_path, weights_only=False)
            print_comparison(before_metrics, after_metrics)
        else:
            print("Need both before/ and after/ metrics.pt to compare.")
        return

    metrics = run_diagnostic(args.label)

    # Save metrics for later comparison
    metrics_path = OUTPUT_DIR / args.label / "metrics.pt"
    torch.save(metrics, metrics_path)
    print(f"\nMetrics saved to {metrics_path}")

    # If both exist, print comparison automatically
    before_path = OUTPUT_DIR / "before" / "metrics.pt"
    after_path = OUTPUT_DIR / "after" / "metrics.pt"
    if before_path.exists() and after_path.exists():
        before_metrics = torch.load(before_path, weights_only=False)
        after_metrics = torch.load(after_path, weights_only=False)
        print_comparison(before_metrics, after_metrics)


if __name__ == "__main__":
    main()
