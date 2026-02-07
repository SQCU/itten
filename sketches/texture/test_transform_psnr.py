"""
Test script for spectral transform PSNR validation.

Compares spectral transforms against Gaussian blur baseline.
Goal: transforms should produce HIGHER PSNR residual than blur
(meaning more visual change from original).

PSNR formula: 10 * log10(MAX^2 / MSE) where MSE = mean((a-b)^2)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple
from datetime import datetime

from texture.transforms import eigenvector_phase_field, fiedler_nodal_lines


def compute_psnr(original: np.ndarray, transformed: np.ndarray) -> float:
    """
    Compute PSNR between original and transformed images.

    PSNR = 10 * log10(MAX^2 / MSE)

    Higher PSNR = smaller difference (more similar)
    Lower PSNR = larger difference (more visual change)

    For residual analysis:
    - We want transforms that produce LOW PSNR (high visual change)
    - "PSNR residual" means the PSNR of (original, transformed)

    Args:
        original: Reference image
        transformed: Transformed image

    Returns:
        PSNR value in dB
    """
    # Ensure both are normalized to [0, 1]
    orig = original.astype(np.float64)
    trans = transformed.astype(np.float64)

    if orig.max() > 1.0:
        orig = orig / 255.0
    if trans.max() > 1.0:
        trans = trans / 255.0

    mse = np.mean((orig - trans) ** 2)

    if mse < 1e-10:
        return float('inf')

    max_val = 1.0  # Since we normalize to [0, 1]
    psnr = 10 * np.log10(max_val ** 2 / mse)

    return psnr


def generate_carrier_pattern(size: int = 64) -> np.ndarray:
    """
    Generate a test carrier pattern with interesting structure.

    Creates a pattern with edges, gradients, and texture for
    spectral analysis.

    Args:
        size: Image dimension (size x size)

    Returns:
        (size, size) numpy array in [0, 1]
    """
    # Create coordinate grids
    y, x = np.ogrid[:size, :size]
    y = y.astype(np.float64) / size
    x = x.astype(np.float64) / size

    # Combine multiple patterns for rich spectral content
    # 1. Radial gradient from center
    cx, cy = 0.5, 0.5
    radial = np.sqrt((x - cx)**2 + (y - cy)**2)

    # 2. Diagonal bands
    diagonal = np.sin(10 * np.pi * (x + y))

    # 3. Circular pattern
    circular = np.sin(8 * np.pi * radial)

    # 4. Noise component for texture
    np.random.seed(42)
    noise = np.random.rand(size, size) * 0.2

    # Combine
    pattern = 0.3 * radial + 0.25 * (diagonal * 0.5 + 0.5) + 0.25 * (circular * 0.5 + 0.5) + 0.2 * noise

    # Normalize to [0, 1]
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())

    return pattern.astype(np.float32)


def apply_gaussian_blur(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Apply Gaussian blur as baseline transform.

    Args:
        image: Input image
        sigma: Blur sigma

    Returns:
        Blurred image
    """
    blurred = gaussian_filter(image, sigma=sigma)
    return blurred


def run_psnr_comparison(
    image: np.ndarray,
    theta_values: List[float],
    blur_sigma: float = 2.0
) -> Dict:
    """
    Run PSNR comparison between transforms and blur baseline.

    Args:
        image: Test image
        theta_values: List of theta values to test
        blur_sigma: Gaussian blur sigma for baseline

    Returns:
        Dictionary with results
    """
    results = {
        'theta_values': theta_values,
        'blur_sigma': blur_sigma,
        'image_shape': image.shape,
        'blur_psnr': None,
        'eigenvector_phase_field': {},
        'fiedler_nodal_lines': {},
    }

    # Compute blur baseline
    blurred = apply_gaussian_blur(image, sigma=blur_sigma)
    blur_psnr = compute_psnr(image, blurred)
    results['blur_psnr'] = blur_psnr

    print(f"Baseline Gaussian blur (sigma={blur_sigma}): PSNR = {blur_psnr:.2f} dB")
    print("-" * 60)

    # Test Eigenvector Phase Field
    print("\nEigenvector Phase Field Transform:")
    for theta in theta_values:
        try:
            transformed, _ = eigenvector_phase_field(image, theta=theta)
            psnr = compute_psnr(image, transformed)
            results['eigenvector_phase_field'][theta] = {
                'psnr': psnr,
                'beats_blur': psnr < blur_psnr  # Lower PSNR = more change
            }
            status = "BEATS BLUR" if psnr < blur_psnr else "under blur"
            print(f"  theta={theta:.1f}: PSNR = {psnr:.2f} dB  [{status}]")
        except Exception as e:
            print(f"  theta={theta:.1f}: ERROR - {e}")
            results['eigenvector_phase_field'][theta] = {'psnr': None, 'error': str(e)}

    # Test Fiedler Nodal Lines
    print("\nFiedler Nodal Lines Transform:")
    for theta in theta_values:
        try:
            transformed, _ = fiedler_nodal_lines(image, theta=theta)
            psnr = compute_psnr(image, transformed)
            results['fiedler_nodal_lines'][theta] = {
                'psnr': psnr,
                'beats_blur': psnr < blur_psnr
            }
            status = "BEATS BLUR" if psnr < blur_psnr else "under blur"
            print(f"  theta={theta:.1f}: PSNR = {psnr:.2f} dB  [{status}]")
        except Exception as e:
            print(f"  theta={theta:.1f}: ERROR - {e}")
            results['fiedler_nodal_lines'][theta] = {'psnr': None, 'error': str(e)}

    return results


def generate_markdown_report(results: Dict) -> str:
    """
    Generate markdown report from results.

    Args:
        results: Dictionary with PSNR results

    Returns:
        Markdown string
    """
    lines = [
        "# Spectral Transform PSNR Validation Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        "This report compares spectral transforms against a Gaussian blur baseline.",
        "**Goal**: Spectral transforms should produce **LOWER** PSNR than blur,",
        "indicating more visual change from the original image.",
        "",
        "PSNR Formula: `10 * log10(MAX^2 / MSE)` where `MSE = mean((a-b)^2)`",
        "",
        "- **Lower PSNR** = larger difference from original = more visual transformation",
        "- **Higher PSNR** = smaller difference from original = less transformation",
        "",
        "## Test Configuration",
        "",
        f"- Image size: {results['image_shape'][0]}x{results['image_shape'][1]}",
        f"- Gaussian blur sigma: {results['blur_sigma']}",
        f"- Theta values tested: {results['theta_values']}",
        "",
        "## Baseline",
        "",
        f"**Gaussian Blur (sigma={results['blur_sigma']})**: PSNR = {results['blur_psnr']:.2f} dB",
        "",
        "---",
        "",
        "## Eigenvector Phase Field Transform",
        "",
        "Creates phase fields from eigenvector pairs, producing psychedelic spiral patterns",
        "centered on topological defects.",
        "",
        "| Theta | PSNR (dB) | vs Blur |",
        "|-------|-----------|---------|",
    ]

    for theta in results['theta_values']:
        data = results['eigenvector_phase_field'].get(theta, {})
        psnr = data.get('psnr')
        if psnr is not None:
            beats = data.get('beats_blur', False)
            status = "**BEATS BLUR**" if beats else "under blur"
            lines.append(f"| {theta:.1f} | {psnr:.2f} | {status} |")
        else:
            error = data.get('error', 'Unknown error')
            lines.append(f"| {theta:.1f} | ERROR | {error} |")

    # Count successes for phase field
    phase_beats = sum(1 for t in results['theta_values']
                      if results['eigenvector_phase_field'].get(t, {}).get('beats_blur', False))
    phase_total = len(results['theta_values'])

    lines.extend([
        "",
        f"**Summary**: {phase_beats}/{phase_total} theta values beat the blur baseline.",
        "",
        "---",
        "",
        "## Fiedler Nodal Lines Transform",
        "",
        "Extracts zero-crossings of the Fiedler vector to create natural partition",
        "boundaries that respect image structure.",
        "",
        "| Theta | PSNR (dB) | vs Blur |",
        "|-------|-----------|---------|",
    ])

    for theta in results['theta_values']:
        data = results['fiedler_nodal_lines'].get(theta, {})
        psnr = data.get('psnr')
        if psnr is not None:
            beats = data.get('beats_blur', False)
            status = "**BEATS BLUR**" if beats else "under blur"
            lines.append(f"| {theta:.1f} | {psnr:.2f} | {status} |")
        else:
            error = data.get('error', 'Unknown error')
            lines.append(f"| {theta:.1f} | ERROR | {error} |")

    # Count successes for fiedler
    fiedler_beats = sum(1 for t in results['theta_values']
                        if results['fiedler_nodal_lines'].get(t, {}).get('beats_blur', False))
    fiedler_total = len(results['theta_values'])

    lines.extend([
        "",
        f"**Summary**: {fiedler_beats}/{fiedler_total} theta values beat the blur baseline.",
        "",
        "---",
        "",
        "## Overall Assessment",
        "",
    ])

    total_beats = phase_beats + fiedler_beats
    total_tests = phase_total + fiedler_total

    if total_beats == total_tests:
        lines.append("**PASS**: All spectral transforms produce more visual change than Gaussian blur.")
    elif total_beats > total_tests // 2:
        lines.append(f"**PARTIAL PASS**: {total_beats}/{total_tests} transform configurations beat blur.")
    else:
        lines.append(f"**NEEDS WORK**: Only {total_beats}/{total_tests} configurations beat blur.")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Spectral transforms that beat blur are creating mathematically meaningful",
        "  visual modifications that go beyond simple smoothing.",
        "- Higher theta values typically increase the visual change for phase-based transforms.",
        "- The Fiedler nodal lines create structural boundaries that can dramatically",
        "  differ from the original image at high theta values.",
        "",
        "---",
        "",
        "*Generated by texture/test_transform_psnr.py*",
    ])

    return "\n".join(lines)


def main():
    """Run PSNR validation and generate report."""
    print("=" * 60)
    print("Spectral Transform PSNR Validation")
    print("=" * 60)

    # Generate test carrier pattern
    print("\nGenerating test carrier pattern...")
    image = generate_carrier_pattern(size=64)
    print(f"Image shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")

    # Define theta values to test
    theta_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Run comparison
    print("\nRunning PSNR comparison...")
    print("=" * 60)
    results = run_psnr_comparison(image, theta_values, blur_sigma=2.0)

    # Generate markdown report
    print("\n" + "=" * 60)
    print("Generating markdown report...")
    report = generate_markdown_report(results)

    # Write report to file
    report_path = "/home/bigboi/itten/hypercontexts/transform-psnr-results.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Report written to: {report_path}")
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
