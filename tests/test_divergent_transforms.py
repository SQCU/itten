"""
Test Divergent Spectral Transforms

These tests verify that the new non-linear spectral transforms produce results
that DIVERGE from both input operands, rather than just blending them.

Success criteria:
    - PSNR(carrier, result) < 15dB - result diverged from carrier
    - PSNR(operand, result) < 15dB - result diverged from operand

Failure condition:
    - PSNR > 25dB means the transform just reproduced an input
"""

import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image
from spectral_ops_fast import (
    eigenvector_phase_field,
    spectral_contour_sdf,
    commute_time_distance_field,
    spectral_warp,
    spectral_subdivision_blend,
)
from texture.render.bump_render import render_bumped_egg


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.

    Lower PSNR = more different images.
    PSNR > 25dB = very similar
    PSNR < 15dB = substantially different
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Normalize to [0, 1]
    if img1.max() > 1:
        img1 = img1 / 255.0
    if img2.max() > 1:
        img2 = img2 / 255.0

    # Resize if needed
    if img1.shape != img2.shape:
        from scipy.ndimage import zoom
        h, w = img2.shape[:2]
        if img1.ndim == 2 and img2.ndim == 2:
            zoom_h = h / img1.shape[0]
            zoom_w = w / img1.shape[1]
            img1 = zoom(img1, (zoom_h, zoom_w), order=1)
        elif img1.ndim == 2:
            # img1 is grayscale, img2 is color - convert
            zoom_h = h / img1.shape[0]
            zoom_w = w / img1.shape[1]
            img1 = zoom(img1, (zoom_h, zoom_w), order=1)
            # Take mean across channels for comparison
            img2 = img2.mean(axis=-1) if img2.ndim == 3 else img2

    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0  # Identical images

    max_pixel = 1.0
    psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_val


def autocorrelation(img: np.ndarray, lags: list = [1, 4, 8, 16]) -> dict:
    """
    Compute autocorrelation at specified lags.

    Returns dict with lag -> correlation value.
    """
    img = img.astype(np.float64)
    if img.max() > 1:
        img = img / 255.0
    if img.ndim == 3:
        img = img.mean(axis=-1)

    # Flatten for 1D autocorrelation
    flat = img.flatten()
    n = len(flat)
    mean = flat.mean()
    var = np.var(flat)

    results = {}
    for lag in lags:
        if lag >= n:
            results[lag] = 0.0
            continue

        # Compute correlation at this lag
        corr = np.sum((flat[:-lag] - mean) * (flat[lag:] - mean)) / ((n - lag) * var + 1e-10)
        results[lag] = corr

    return results


def load_image_grayscale(path: str) -> np.ndarray:
    """Load image as grayscale float array."""
    img = Image.open(path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    if img.mode == 'RGB':
        img = img.convert('L')
    return np.array(img, dtype=np.float32) / 255.0


def create_checkerboard(size: int = 128, block_size: int = 16) -> np.ndarray:
    """Create a checkerboard pattern."""
    arr = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            if ((i // block_size) + (j // block_size)) % 2 == 0:
                arr[i, j] = 1.0
    return arr


def create_noise(size: int = 128, seed: int = 42) -> np.ndarray:
    """Create random noise pattern."""
    np.random.seed(seed)
    return np.random.rand(size, size).astype(np.float32)


def test_divergent_transforms():
    """
    Main test function for divergent spectral transforms.

    Tests each transform and verifies divergence from both inputs.
    """
    print("=" * 60)
    print("DIVERGENT SPECTRAL TRANSFORMS TEST")
    print("=" * 60)

    # Output directory
    output_dir = project_root / "demo_output" / "divergent_transforms"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create test images
    carrier_dir = project_root / "demo_output" / "3d_bump_sweep" / "carriers"

    # Try to load real textures, fall back to synthetic
    carriers = {}
    operands = {}

    # Check for amongus carrier
    amongus_path = carrier_dir / "amongus.png"
    if amongus_path.exists():
        carriers['amongus'] = load_image_grayscale(str(amongus_path))
        print(f"Loaded carrier: amongus ({carriers['amongus'].shape})")

    # Load other operands
    operand_dir = project_root / "demo_output" / "3d_bump_sweep" / "operands"
    if operand_dir.exists():
        for op_file in operand_dir.glob("*.png"):
            name = op_file.stem
            if name not in operands:
                operands[name] = load_image_grayscale(str(op_file))
                print(f"Loaded operand: {name} ({operands[name].shape})")

    # Fall back to synthetic textures if needed
    if not carriers:
        carriers['checkerboard'] = create_checkerboard(128, 16)
        carriers['noise'] = create_noise(128, seed=42)
        print("Using synthetic carriers: checkerboard, noise")

    if not operands:
        operands['checker_fine'] = create_checkerboard(128, 8)
        operands['noise_2'] = create_noise(128, seed=123)
        print("Using synthetic operands: checker_fine, noise_2")

    # Define transforms to test
    transforms = {
        'eigenvector_phase_field': eigenvector_phase_field,
        'spectral_contour_sdf': spectral_contour_sdf,
        'commute_time_distance_field': commute_time_distance_field,
        'spectral_warp': spectral_warp,
        'spectral_subdivision_blend': spectral_subdivision_blend,
    }

    # Theta values to test
    theta_values = [0.1, 0.5, 0.9]

    # Results storage
    results = []

    # Get first carrier and operand for testing
    carrier_name = list(carriers.keys())[0]
    carrier = carriers[carrier_name]

    operand_name = list(operands.keys())[0]
    operand = operands[operand_name]

    print(f"\nUsing carrier: {carrier_name} ({carrier.shape})")
    print(f"Using operand: {operand_name} ({operand.shape})")
    print()

    # Test each transform
    for transform_name, transform_fn in transforms.items():
        print(f"\n{'='*60}")
        print(f"Testing: {transform_name}")
        print(f"{'='*60}")

        for theta in theta_values:
            print(f"\n  theta={theta}:")

            try:
                # Apply transform
                result = transform_fn(carrier, operand, theta=theta)

                # Compute metrics
                psnr_carrier = psnr(carrier, result)
                psnr_operand = psnr(operand, result)

                # Compute autocorrelation
                autocorr_carrier = autocorrelation(carrier)
                autocorr_operand = autocorrelation(operand)
                autocorr_result = autocorrelation(result)

                # Check divergence criteria
                diverged_from_carrier = psnr_carrier < 15
                diverged_from_operand = psnr_operand < 15
                success = diverged_from_carrier and diverged_from_operand

                status = "SUCCESS" if success else "PARTIAL" if (diverged_from_carrier or diverged_from_operand) else "FAILED"

                print(f"    PSNR(carrier, result): {psnr_carrier:.2f} dB {'< 15dB OK' if diverged_from_carrier else '> 15dB WARN'}")
                print(f"    PSNR(operand, result): {psnr_operand:.2f} dB {'< 15dB OK' if diverged_from_operand else '> 15dB WARN'}")
                print(f"    Autocorr carrier lag=8: {autocorr_carrier.get(8, 0):.4f}")
                print(f"    Autocorr result  lag=8: {autocorr_result.get(8, 0):.4f}")
                print(f"    Status: {status}")

                # Store result
                results.append({
                    'transform': transform_name,
                    'theta': theta,
                    'psnr_carrier': psnr_carrier,
                    'psnr_operand': psnr_operand,
                    'autocorr_carrier': autocorr_carrier,
                    'autocorr_result': autocorr_result,
                    'diverged_carrier': diverged_from_carrier,
                    'diverged_operand': diverged_from_operand,
                    'success': success,
                    'result': result,
                })

                # Save texture output
                texture_path = output_dir / f"{transform_name}_theta{theta}_texture.png"
                result_uint8 = (result * 255).astype(np.uint8)
                Image.fromarray(result_uint8).save(str(texture_path))

                # Render as bump-mapped egg
                try:
                    egg_render = render_bumped_egg(
                        result,
                        displacement_scale=0.15,
                        output_size=512,
                    )
                    egg_path = output_dir / f"{transform_name}_theta{theta}_egg.png"
                    Image.fromarray(egg_render).save(str(egg_path))
                    print(f"    Saved: {egg_path.name}")
                except Exception as e:
                    print(f"    Render failed: {e}")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'transform': transform_name,
                    'theta': theta,
                    'error': str(e),
                    'success': False,
                })

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successes = [r for r in results if r.get('success', False)]
    partials = [r for r in results if not r.get('success', False) and (r.get('diverged_carrier', False) or r.get('diverged_operand', False))]
    failures = [r for r in results if not r.get('success', False) and not (r.get('diverged_carrier', False) or r.get('diverged_operand', False))]

    print(f"\nTotal tests: {len(results)}")
    print(f"Full divergence (both PSNR < 15dB): {len(successes)}")
    print(f"Partial divergence (one PSNR < 15dB): {len(partials)}")
    print(f"Failed (both PSNR > 15dB): {len(failures)}")

    if successes:
        print("\nSuccessful transforms:")
        for r in successes:
            print(f"  - {r['transform']} @ theta={r['theta']}: carrier={r['psnr_carrier']:.1f}dB, operand={r['psnr_operand']:.1f}dB")

    print(f"\nOutputs saved to: {output_dir}")

    return results


def generate_results_markdown(results: list, output_path: str):
    """Generate markdown results file."""

    lines = [
        "# Divergent Transforms Results",
        "",
        "## Summary",
        "",
        "Testing non-linear spectral transforms that produce results DIVERGING from both operands.",
        "",
        "**Success criteria:**",
        "- PSNR(carrier, result) < 15dB - result diverged from carrier",
        "- PSNR(operand, result) < 15dB - result diverged from operand",
        "",
        "**Failure condition:**",
        "- PSNR > 25dB means the transform just reproduced an input",
        "",
        "## Results Table",
        "",
        "| Transform | Theta | PSNR Carrier | PSNR Operand | Diverged? |",
        "|-----------|-------|--------------|--------------|-----------|",
    ]

    for r in results:
        if 'error' in r:
            lines.append(f"| {r['transform']} | {r['theta']} | ERROR | ERROR | No |")
        else:
            diverged = "YES" if r['success'] else "PARTIAL" if (r.get('diverged_carrier') or r.get('diverged_operand')) else "No"
            lines.append(f"| {r['transform']} | {r['theta']} | {r['psnr_carrier']:.2f} dB | {r['psnr_operand']:.2f} dB | {diverged} |")

    lines.extend([
        "",
        "## Autocorrelation Analysis",
        "",
        "Autocorrelation at lag=8 shows structural differences:",
        "",
    ])

    for r in results:
        if 'autocorr_result' in r:
            ac_carrier = r.get('autocorr_carrier', {}).get(8, 0)
            ac_result = r.get('autocorr_result', {}).get(8, 0)
            diff = abs(ac_carrier - ac_result)
            lines.append(f"- **{r['transform']} @ theta={r['theta']}**: carrier AC={ac_carrier:.4f}, result AC={ac_result:.4f}, diff={diff:.4f}")

    lines.extend([
        "",
        "## Key Mathematical Insights",
        "",
        "The transforms achieve divergence through non-linear operations:",
        "",
        "1. **eigenvector_phase_field**: `arctan2(ev1, ev2)` creates spiral patterns at topological defects",
        "2. **spectral_contour_sdf**: Distance transforms create smooth gradients not in discrete inputs",
        "3. **commute_time_distance_field**: Squared eigenvector differences weighted by inverse eigenvalues",
        "4. **spectral_warp**: Coordinate remapping via interpolation is inherently non-linear",
        "5. **spectral_subdivision_blend**: Sign thresholding creates boundaries not in either input",
        "",
        "## Demo Outputs",
        "",
        "Rendered eggs saved to: `demo_output/divergent_transforms/`",
        "",
    ])

    # List output files
    output_dir = Path(output_path).parent.parent / "demo_output" / "divergent_transforms"
    if output_dir.exists():
        lines.append("### Texture Files")
        lines.append("")
        for f in sorted(output_dir.glob("*_texture.png")):
            lines.append(f"- `{f.name}`")

        lines.append("")
        lines.append("### Rendered Eggs")
        lines.append("")
        for f in sorted(output_dir.glob("*_egg.png")):
            lines.append(f"- `{f.name}`")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    results = test_divergent_transforms()

    # Generate markdown results
    results_path = project_root / "hypercontexts" / "divergent-transforms-results.md"
    generate_results_markdown(results, str(results_path))
