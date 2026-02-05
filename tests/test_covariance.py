"""
2D Covariance Validation for Spectral Transforms.

Tests new transforms (eigenvector_phase_field, fiedler_nodal_lines) against
three surfaces (marble, brick, noise) across five theta values to demonstrate
non-trivial 2D covariance.

Success criteria:
- row_var (theta sensitivity) > 0.5 for at least 2 surfaces
- col_var (surface sensitivity) > 0.5 for at least 3 theta values
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from texture.surfaces import generate_marble, generate_brick, generate_noise
from texture.transforms import eigenvector_phase_field, fiedler_nodal_lines


def compute_psnr(original: np.ndarray, transformed: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        original: Reference image
        transformed: Transformed image

    Returns:
        PSNR in dB. Higher values mean more similarity.
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
        return 100.0  # Essentially identical

    max_val = 1.0
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr


def run_covariance_validation():
    """
    Run the full 2D covariance validation.

    Returns:
        dict with all results
    """
    print("=" * 60)
    print("2D Covariance Validation for Spectral Transforms")
    print("=" * 60)

    # Parameters
    SIZE = 64
    THETA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
    SURFACE_NAMES = ['marble', 'brick', 'noise']
    TRANSFORM_NAMES = ['eigenvector_phase_field', 'fiedler_nodal_lines']

    # Generate surfaces
    print("\n[1] Generating surfaces at {}x{}...".format(SIZE, SIZE))
    surfaces = {
        'marble': generate_marble(SIZE, seed=42),
        'brick': generate_brick(SIZE, seed=42),
        'noise': generate_noise(SIZE, seed=42),
    }
    for name, surf in surfaces.items():
        print(f"    {name}: shape={surf.shape}, range=[{surf.min():.3f}, {surf.max():.3f}]")

    # Initialize result matrices
    # Each matrix is (3 surfaces) x (5 theta values)
    results = {}

    for transform_name in TRANSFORM_NAMES:
        print(f"\n[2] Testing {transform_name}...")
        matrix = np.zeros((len(SURFACE_NAMES), len(THETA_VALUES)))

        for i, surf_name in enumerate(SURFACE_NAMES):
            surface = surfaces[surf_name]
            print(f"    Surface: {surf_name}")

            for j, theta in enumerate(THETA_VALUES):
                # Apply transform
                if transform_name == 'eigenvector_phase_field':
                    transformed, _ = eigenvector_phase_field(surface, theta=theta)
                else:  # fiedler_nodal_lines
                    transformed, _ = fiedler_nodal_lines(surface, theta=theta)

                # Compute PSNR
                psnr = compute_psnr(surface, transformed)
                matrix[i, j] = psnr
                print(f"        theta={theta}: PSNR={psnr:.2f} dB")

        results[transform_name] = {
            'matrix': matrix,
            'surface_names': SURFACE_NAMES,
            'theta_values': THETA_VALUES,
        }

    # Compute variance metrics
    print("\n[3] Computing variance metrics...")

    for transform_name in TRANSFORM_NAMES:
        matrix = results[transform_name]['matrix']

        # Row variance: variance across theta for each surface
        # This measures how sensitive the transform is to theta for each surface
        row_variances = np.var(matrix, axis=1)  # (3,) - one per surface

        # Column variance: variance across surfaces for each theta
        # This measures how differently the transform behaves on different surfaces
        col_variances = np.var(matrix, axis=0)  # (5,) - one per theta

        results[transform_name]['row_variances'] = row_variances
        results[transform_name]['col_variances'] = col_variances

        print(f"\n    {transform_name}:")
        print(f"        Row variances (theta sensitivity per surface):")
        for i, surf_name in enumerate(SURFACE_NAMES):
            print(f"            {surf_name}: {row_variances[i]:.4f}")

        print(f"        Column variances (surface sensitivity per theta):")
        for j, theta in enumerate(THETA_VALUES):
            print(f"            theta={theta}: {col_variances[j]:.4f}")

    # Check success criteria
    print("\n[4] Checking success criteria...")

    all_passed = True
    for transform_name in TRANSFORM_NAMES:
        row_variances = results[transform_name]['row_variances']
        col_variances = results[transform_name]['col_variances']

        # Criterion 1: row_var > 0.5 for at least 2 surfaces
        row_passes = np.sum(row_variances > 0.5)
        row_criterion = row_passes >= 2

        # Criterion 2: col_var > 0.5 for at least 3 theta values
        col_passes = np.sum(col_variances > 0.5)
        col_criterion = col_passes >= 3

        passed = row_criterion and col_criterion
        all_passed = all_passed and passed

        status = "PASS" if passed else "FAIL"
        print(f"\n    {transform_name}: {status}")
        print(f"        Row criterion (>=2 surfaces with var>0.5): {row_passes}/3 - {'PASS' if row_criterion else 'FAIL'}")
        print(f"        Col criterion (>=3 thetas with var>0.5): {col_passes}/5 - {'PASS' if col_criterion else 'FAIL'}")

        results[transform_name]['row_criterion'] = row_criterion
        results[transform_name]['col_criterion'] = col_criterion
        results[transform_name]['passed'] = passed
        results[transform_name]['row_passes'] = row_passes
        results[transform_name]['col_passes'] = col_passes

    results['all_passed'] = all_passed

    print("\n" + "=" * 60)
    print(f"Overall result: {'ALL CRITERIA MET' if all_passed else 'SOME CRITERIA NOT MET'}")
    print("=" * 60)

    return results


def write_results_markdown(results: dict, output_path: str):
    """
    Write comprehensive results to markdown file.
    """
    lines = []
    lines.append("# 2D Covariance Validation Results")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    overall = "PASS" if results['all_passed'] else "FAIL"
    lines.append(f"**Overall Result: {overall}**")
    lines.append("")
    lines.append("Testing spectral transforms against surfaces to demonstrate non-trivial 2D covariance.")
    lines.append("")
    lines.append("### Success Criteria")
    lines.append("- row_var (theta sensitivity) > 0.5 for at least 2 surfaces")
    lines.append("- col_var (surface sensitivity) > 0.5 for at least 3 theta values")
    lines.append("")

    for transform_name in ['eigenvector_phase_field', 'fiedler_nodal_lines']:
        r = results[transform_name]
        matrix = r['matrix']
        surface_names = r['surface_names']
        theta_values = r['theta_values']
        row_variances = r['row_variances']
        col_variances = r['col_variances']

        lines.append(f"## Transform: `{transform_name}`")
        lines.append("")
        lines.append(f"**Result: {'PASS' if r['passed'] else 'FAIL'}**")
        lines.append("")

        # PSNR Matrix
        lines.append("### PSNR Matrix (dB)")
        lines.append("")
        lines.append("| Surface | " + " | ".join([f"theta={t}" for t in theta_values]) + " |")
        lines.append("|---------|" + "|".join(["--------"] * len(theta_values)) + "|")

        for i, surf_name in enumerate(surface_names):
            row = [f"{matrix[i, j]:.2f}" for j in range(len(theta_values))]
            lines.append(f"| {surf_name} | " + " | ".join(row) + " |")

        lines.append("")

        # Row Variances
        lines.append("### Row Variances (Theta Sensitivity per Surface)")
        lines.append("")
        lines.append("| Surface | Variance | Status |")
        lines.append("|---------|----------|--------|")
        for i, surf_name in enumerate(surface_names):
            status = "PASS" if row_variances[i] > 0.5 else "-"
            lines.append(f"| {surf_name} | {row_variances[i]:.4f} | {status} |")

        lines.append("")
        lines.append(f"**Surfaces with var > 0.5: {r['row_passes']}/3** (need >= 2)")
        lines.append("")

        # Column Variances
        lines.append("### Column Variances (Surface Sensitivity per Theta)")
        lines.append("")
        lines.append("| Theta | Variance | Status |")
        lines.append("|-------|----------|--------|")
        for j, theta in enumerate(theta_values):
            status = "PASS" if col_variances[j] > 0.5 else "-"
            lines.append(f"| {theta} | {col_variances[j]:.4f} | {status} |")

        lines.append("")
        lines.append(f"**Thetas with var > 0.5: {r['col_passes']}/5** (need >= 3)")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    lines.append("### What the metrics mean")
    lines.append("")
    lines.append("- **Row variance** measures how much PSNR changes across theta values for a fixed surface.")
    lines.append("  - High row variance = transform is sensitive to theta parameter")
    lines.append("  - This is expected: as theta changes, the output should change")
    lines.append("")
    lines.append("- **Column variance** measures how much PSNR changes across surfaces for a fixed theta.")
    lines.append("  - High column variance = transform responds differently to different surface textures")
    lines.append("  - This is the key insight: the transform doesn't just apply the same operation regardless of input")
    lines.append("")
    lines.append("### 2D Covariance")
    lines.append("")
    lines.append("Having both high row and column variance demonstrates **non-separable 2D covariance**:")
    lines.append("- The transform's behavior depends on BOTH the theta parameter AND the input surface")
    lines.append("- This is more interesting than a transform that only depends on theta")
    lines.append("- It shows the spectral structure of the input matters")
    lines.append("")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nResults written to: {output_path}")


if __name__ == '__main__':
    # Run validation
    results = run_covariance_validation()

    # Write results markdown
    output_path = '/home/bigboi/itten/hypercontexts/covariance-validation-results.md'
    write_results_markdown(results, output_path)
