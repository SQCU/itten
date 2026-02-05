"""
Proper Divergent Spectral Transforms Test

Fixes previous test issues:
1. Uses all 5 theta values: [0.1, 0.3, 0.5, 0.7, 0.9]
2. Uses natural images (not noise) as carriers/operands
3. Uses checkerboards (always better than noise for PSNR judgment)
4. Uses scattered amongus (NOT dense tiling)
5. Checks for color textures from 3d_psnr

Success criteria:
    - PSNR(carrier, result) < 15dB - result diverged from carrier
    - PSNR(operand, result) < 15dB - result diverged from operand
"""

import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
from texture.patterns import (
    generate_checkerboard,
    generate_amongus_scattered,
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

    # Handle different shapes
    if img1.shape != img2.shape:
        from scipy.ndimage import zoom
        h, w = img2.shape[:2]
        if img1.ndim == 2 and img2.ndim == 2:
            zoom_h = h / img1.shape[0]
            zoom_w = w / img1.shape[1]
            img1 = zoom(img1, (zoom_h, zoom_w), order=1)
        elif img1.ndim == 2:
            zoom_h = h / img1.shape[0]
            zoom_w = w / img1.shape[1]
            img1 = zoom(img1, (zoom_h, zoom_w), order=1)
            img2 = img2.mean(axis=-1) if img2.ndim == 3 else img2
        elif img2.ndim == 2:
            img1 = img1.mean(axis=-1) if img1.ndim == 3 else img1
            zoom_h = h / img1.shape[0]
            zoom_w = w / img1.shape[1]
            img1 = zoom(img1, (zoom_h, zoom_w), order=1)

    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0  # Identical images

    max_pixel = 1.0
    psnr_val = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_val


def load_image_grayscale(path: str, target_size: int = 128) -> np.ndarray:
    """Load image as grayscale float array, resized to target size."""
    img = Image.open(path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    if img.mode == 'RGB':
        img = img.convert('L')

    # Resize to target size
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    return np.array(img, dtype=np.float32) / 255.0


def run_proper_test():
    """
    Run proper divergent transforms test with correct inputs.
    """
    print("=" * 70)
    print("PROPER DIVERGENT SPECTRAL TRANSFORMS TEST")
    print("=" * 70)
    print("\nFixes from previous test:")
    print("  1. Uses all 5 theta values: [0.1, 0.3, 0.5, 0.7, 0.9]")
    print("  2. Uses natural images (snek-heavy.png, toof.png)")
    print("  3. Uses checkerboards (better than noise for PSNR)")
    print("  4. Uses scattered amongus (NOT dense tiling)")
    print()

    # Output directory
    output_dir = project_root / "demo_output" / "divergent_proper"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Size for all textures
    SIZE = 128

    # === LOAD CARRIERS ===
    carriers = {}

    # Carrier 1: Natural image (snek-heavy.png)
    snek_path = project_root / "demo_output" / "inputs" / "snek-heavy.png"
    if snek_path.exists():
        carriers['snek_heavy'] = load_image_grayscale(str(snek_path), SIZE)
        print(f"Loaded carrier: snek-heavy.png ({carriers['snek_heavy'].shape})")
    else:
        print(f"WARNING: {snek_path} not found")

    # Carrier 2: Checkerboard (block_size=16)
    carriers['checkerboard_16'] = generate_checkerboard(SIZE, tile_size=16)
    print(f"Generated carrier: checkerboard_16 ({carriers['checkerboard_16'].shape})")

    # Carrier 3: Scattered amongus (3 copies)
    carriers['amongus_scattered_3'] = generate_amongus_scattered(SIZE, num_copies=3, seed=None)
    print(f"Generated carrier: amongus_scattered_3 ({carriers['amongus_scattered_3'].shape})")

    # === LOAD OPERANDS ===
    operands = {}

    # Operand 1: Natural image (toof.png)
    toof_path = project_root / "demo_output" / "inputs" / "toof.png"
    if toof_path.exists():
        operands['toof'] = load_image_grayscale(str(toof_path), SIZE)
        print(f"Loaded operand: toof.png ({operands['toof'].shape})")
    else:
        print(f"WARNING: {toof_path} not found")

    # Operand 2: Checkerboard (block_size=8 - different from carrier)
    operands['checkerboard_8'] = generate_checkerboard(SIZE, tile_size=8)
    print(f"Generated operand: checkerboard_8 ({operands['checkerboard_8'].shape})")

    # Operand 3: Scattered amongus (4 copies, seed=42 - different from carrier)
    operands['amongus_scattered_4'] = generate_amongus_scattered(SIZE, num_copies=4, seed=42)
    print(f"Generated operand: amongus_scattered_4 ({operands['amongus_scattered_4'].shape})")

    # === SAVE INPUT TEXTURES ===
    input_dir = output_dir / "inputs"
    input_dir.mkdir(exist_ok=True)

    for name, arr in carriers.items():
        path = input_dir / f"carrier_{name}.png"
        Image.fromarray((arr * 255).astype(np.uint8)).save(str(path))

    for name, arr in operands.items():
        path = input_dir / f"operand_{name}.png"
        Image.fromarray((arr * 255).astype(np.uint8)).save(str(path))

    print(f"\nSaved input textures to: {input_dir}")

    # === CHECK FOR COLOR TEXTURES FROM 3D_PSNR ===
    color_textures = {}
    color_dir = project_root / "demo_output" / "3d_psnr"
    if color_dir.exists():
        for f in color_dir.glob("*.png"):
            color_textures[f.stem] = str(f)
        print(f"\nFound {len(color_textures)} color textures in 3d_psnr/")
        for name in list(color_textures.keys())[:5]:
            print(f"  - {name}")
        if len(color_textures) > 5:
            print(f"  ... and {len(color_textures) - 5} more")

    # === DEFINE TRANSFORMS ===
    transforms = {
        'eigenvector_phase_field': eigenvector_phase_field,
        'spectral_contour_sdf': spectral_contour_sdf,
        'commute_time_distance_field': commute_time_distance_field,
        'spectral_warp': spectral_warp,
        'spectral_subdivision_blend': spectral_subdivision_blend,
    }

    # === ALL 5 THETA VALUES ===
    theta_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # === RUN TESTS ===
    results = []
    total_tests = len(carriers) * len(operands) * len(transforms) * len(theta_values)
    test_count = 0

    print(f"\nRunning {total_tests} tests...")
    print("-" * 70)

    for carrier_name, carrier in carriers.items():
        for operand_name, operand in operands.items():
            for transform_name, transform_fn in transforms.items():
                for theta in theta_values:
                    test_count += 1
                    test_id = f"{carrier_name}_{operand_name}_{transform_name}_t{theta}"

                    try:
                        # Apply transform
                        result = transform_fn(carrier, operand, theta=theta)

                        # Compute PSNR metrics
                        psnr_carrier = psnr(carrier, result)
                        psnr_operand = psnr(operand, result)

                        # Check divergence criteria
                        diverged_carrier = psnr_carrier < 15
                        diverged_operand = psnr_operand < 15
                        success = diverged_carrier and diverged_operand

                        status = "OK" if success else "PARTIAL" if (diverged_carrier or diverged_operand) else "FAIL"

                        # Store result
                        results.append({
                            'carrier': carrier_name,
                            'operand': operand_name,
                            'transform': transform_name,
                            'theta': theta,
                            'psnr_carrier': psnr_carrier,
                            'psnr_operand': psnr_operand,
                            'diverged_carrier': diverged_carrier,
                            'diverged_operand': diverged_operand,
                            'success': success,
                            'status': status,
                        })

                        # Save texture output
                        texture_path = output_dir / f"{test_id}_texture.png"
                        result_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
                        Image.fromarray(result_uint8).save(str(texture_path))

                        # Render as bump-mapped egg for successful/partial cases
                        if success or (diverged_carrier or diverged_operand):
                            try:
                                egg_render = render_bumped_egg(
                                    result,
                                    displacement_scale=0.15,
                                    output_size=512,
                                )
                                egg_path = output_dir / f"{test_id}_egg.png"
                                Image.fromarray(egg_render).save(str(egg_path))
                            except Exception as e:
                                pass  # Skip render failures silently

                        # Progress output
                        if test_count % 25 == 0 or test_count == total_tests:
                            print(f"  [{test_count}/{total_tests}] {test_id}: PSNR(c)={psnr_carrier:.1f}dB PSNR(o)={psnr_operand:.1f}dB [{status}]")

                    except Exception as e:
                        results.append({
                            'carrier': carrier_name,
                            'operand': operand_name,
                            'transform': transform_name,
                            'theta': theta,
                            'error': str(e),
                            'success': False,
                            'status': 'ERROR',
                        })
                        print(f"  [{test_count}/{total_tests}] {test_id}: ERROR - {e}")

    # === ANALYZE RESULTS ===
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    valid_results = [r for r in results if 'error' not in r]
    successes = [r for r in valid_results if r['success']]
    partials = [r for r in valid_results if not r['success'] and (r['diverged_carrier'] or r['diverged_operand'])]
    failures = [r for r in valid_results if r['status'] == 'FAIL']
    errors = [r for r in results if 'error' in r]

    print(f"\nTotal tests: {len(results)}")
    print(f"  Full divergence (both PSNR < 15dB): {len(successes)} ({100*len(successes)/len(results):.1f}%)")
    print(f"  Partial divergence (one PSNR < 15dB): {len(partials)} ({100*len(partials)/len(results):.1f}%)")
    print(f"  Failed (both PSNR >= 15dB): {len(failures)} ({100*len(failures)/len(results):.1f}%)")
    print(f"  Errors: {len(errors)}")

    # === BY TRANSFORM ANALYSIS ===
    print("\n--- By Transform ---")
    for transform_name in transforms.keys():
        t_results = [r for r in valid_results if r['transform'] == transform_name]
        t_success = sum(1 for r in t_results if r['success'])
        t_partial = sum(1 for r in t_results if not r['success'] and (r['diverged_carrier'] or r['diverged_operand']))
        avg_psnr_c = np.mean([r['psnr_carrier'] for r in t_results])
        avg_psnr_o = np.mean([r['psnr_operand'] for r in t_results])
        print(f"  {transform_name}:")
        print(f"    Success: {t_success}/{len(t_results)}, Partial: {t_partial}/{len(t_results)}")
        print(f"    Avg PSNR: carrier={avg_psnr_c:.1f}dB, operand={avg_psnr_o:.1f}dB")

    # === BY THETA ANALYSIS ===
    print("\n--- By Theta ---")
    for theta in theta_values:
        t_results = [r for r in valid_results if r['theta'] == theta]
        t_success = sum(1 for r in t_results if r['success'])
        t_partial = sum(1 for r in t_results if not r['success'] and (r['diverged_carrier'] or r['diverged_operand']))
        avg_psnr_c = np.mean([r['psnr_carrier'] for r in t_results])
        avg_psnr_o = np.mean([r['psnr_operand'] for r in t_results])
        print(f"  theta={theta}: Success={t_success}/{len(t_results)}, Partial={t_partial}, Avg PSNR(c)={avg_psnr_c:.1f}dB, PSNR(o)={avg_psnr_o:.1f}dB")

    # === BY CARRIER ANALYSIS ===
    print("\n--- By Carrier ---")
    for carrier_name in carriers.keys():
        c_results = [r for r in valid_results if r['carrier'] == carrier_name]
        c_success = sum(1 for r in c_results if r['success'])
        avg_psnr_c = np.mean([r['psnr_carrier'] for r in c_results])
        print(f"  {carrier_name}: Success={c_success}/{len(c_results)}, Avg PSNR(carrier)={avg_psnr_c:.1f}dB")

    # === BY OPERAND ANALYSIS ===
    print("\n--- By Operand ---")
    for operand_name in operands.keys():
        o_results = [r for r in valid_results if r['operand'] == operand_name]
        o_success = sum(1 for r in o_results if r['success'])
        avg_psnr_o = np.mean([r['psnr_operand'] for r in o_results])
        print(f"  {operand_name}: Success={o_success}/{len(o_results)}, Avg PSNR(operand)={avg_psnr_o:.1f}dB")

    # === BEST AND WORST RESULTS ===
    if valid_results:
        # Sort by combined PSNR (lower is better for divergence)
        sorted_results = sorted(valid_results, key=lambda r: r['psnr_carrier'] + r['psnr_operand'])

        print("\n--- Best Divergent Results (lowest combined PSNR) ---")
        for r in sorted_results[:5]:
            print(f"  {r['carrier']}+{r['operand']} @ {r['transform']} theta={r['theta']}")
            print(f"    PSNR(c)={r['psnr_carrier']:.1f}dB, PSNR(o)={r['psnr_operand']:.1f}dB [{r['status']}]")

        print("\n--- Worst Results (highest combined PSNR) ---")
        for r in sorted_results[-5:]:
            print(f"  {r['carrier']}+{r['operand']} @ {r['transform']} theta={r['theta']}")
            print(f"    PSNR(c)={r['psnr_carrier']:.1f}dB, PSNR(o)={r['psnr_operand']:.1f}dB [{r['status']}]")

    print(f"\nOutputs saved to: {output_dir}")

    return results, carriers, operands


def generate_results_markdown(results: list, output_path: str):
    """Generate comprehensive markdown results file."""

    valid_results = [r for r in results if 'error' not in r]
    successes = [r for r in valid_results if r['success']]
    partials = [r for r in valid_results if not r['success'] and (r['diverged_carrier'] or r['diverged_operand'])]
    failures = [r for r in valid_results if r['status'] == 'FAIL']

    # Get unique values
    transforms = sorted(set(r['transform'] for r in valid_results))
    thetas = sorted(set(r['theta'] for r in valid_results))
    carriers = sorted(set(r['carrier'] for r in valid_results))
    operands = sorted(set(r['operand'] for r in valid_results))

    lines = [
        "# Divergent Transforms Proper Test Results",
        "",
        "## Test Configuration",
        "",
        "This test fixes issues from the previous divergent transforms test:",
        "",
        "| Issue | Previous | Fixed |",
        "|-------|----------|-------|",
        "| Theta values | Only 0.1 and 0.9 | All 5: [0.1, 0.3, 0.5, 0.7, 0.9] |",
        "| Carriers | Noise fields | Natural images + checkerboard + scattered amongus |",
        "| Operands | Noise fields | Natural images + checkerboard + scattered amongus |",
        "| Amongus | Dense tiling | Scattered with random transforms |",
        "",
        "## Carriers Used",
        "",
    ]

    for c in carriers:
        if 'snek' in c:
            lines.append(f"- **{c}**: Natural image (snek-heavy.png)")
        elif 'checkerboard' in c:
            lines.append(f"- **{c}**: Checkerboard with block_size=16")
        elif 'amongus' in c:
            lines.append(f"- **{c}**: Scattered amongus (3 copies, random transforms)")

    lines.extend([
        "",
        "## Operands Used",
        "",
    ])

    for o in operands:
        if 'toof' in o:
            lines.append(f"- **{o}**: Natural image (toof.png)")
        elif 'checkerboard' in o:
            lines.append(f"- **{o}**: Checkerboard with block_size=8 (different from carrier)")
        elif 'amongus' in o:
            lines.append(f"- **{o}**: Scattered amongus (4 copies, seed=42, different from carrier)")

    lines.extend([
        "",
        "## Summary Statistics",
        "",
        f"- **Total tests**: {len(results)}",
        f"- **Full divergence** (both PSNR < 15dB): {len(successes)} ({100*len(successes)/len(results):.1f}%)",
        f"- **Partial divergence** (one PSNR < 15dB): {len(partials)} ({100*len(partials)/len(results):.1f}%)",
        f"- **Failed** (both PSNR >= 15dB): {len(failures)} ({100*len(failures)/len(results):.1f}%)",
        "",
        "**Success criteria:**",
        "- PSNR(carrier, result) < 15dB means result diverged from carrier",
        "- PSNR(operand, result) < 15dB means result diverged from operand",
        "- Full success requires BOTH conditions",
        "",
    ])

    # Results by transform
    lines.extend([
        "## Results by Transform",
        "",
        "| Transform | Success | Partial | Failed | Avg PSNR(carrier) | Avg PSNR(operand) |",
        "|-----------|---------|---------|--------|-------------------|-------------------|",
    ])

    for transform in transforms:
        t_results = [r for r in valid_results if r['transform'] == transform]
        t_success = sum(1 for r in t_results if r['success'])
        t_partial = sum(1 for r in t_results if not r['success'] and (r['diverged_carrier'] or r['diverged_operand']))
        t_fail = sum(1 for r in t_results if r['status'] == 'FAIL')
        avg_c = np.mean([r['psnr_carrier'] for r in t_results])
        avg_o = np.mean([r['psnr_operand'] for r in t_results])
        lines.append(f"| {transform} | {t_success} | {t_partial} | {t_fail} | {avg_c:.1f} dB | {avg_o:.1f} dB |")

    # Results by theta
    lines.extend([
        "",
        "## Results by Theta",
        "",
        "| Theta | Success | Partial | Failed | Avg PSNR(carrier) | Avg PSNR(operand) |",
        "|-------|---------|---------|--------|-------------------|-------------------|",
    ])

    for theta in thetas:
        t_results = [r for r in valid_results if r['theta'] == theta]
        t_success = sum(1 for r in t_results if r['success'])
        t_partial = sum(1 for r in t_results if not r['success'] and (r['diverged_carrier'] or r['diverged_operand']))
        t_fail = sum(1 for r in t_results if r['status'] == 'FAIL')
        avg_c = np.mean([r['psnr_carrier'] for r in t_results])
        avg_o = np.mean([r['psnr_operand'] for r in t_results])
        lines.append(f"| {theta} | {t_success} | {t_partial} | {t_fail} | {avg_c:.1f} dB | {avg_o:.1f} dB |")

    # Full results table
    lines.extend([
        "",
        "## Full Results Table",
        "",
        "| Carrier | Operand | Transform | Theta | PSNR(c) | PSNR(o) | Status |",
        "|---------|---------|-----------|-------|---------|---------|--------|",
    ])

    # Sort by status (OK first, then PARTIAL, then FAIL)
    status_order = {'OK': 0, 'PARTIAL': 1, 'FAIL': 2, 'ERROR': 3}
    sorted_results = sorted(valid_results, key=lambda r: (status_order.get(r['status'], 4), r['psnr_carrier'] + r['psnr_operand']))

    for r in sorted_results:
        lines.append(f"| {r['carrier']} | {r['operand']} | {r['transform']} | {r['theta']} | {r['psnr_carrier']:.1f} dB | {r['psnr_operand']:.1f} dB | {r['status']} |")

    # Key insights
    lines.extend([
        "",
        "## Key Insights",
        "",
        "### Why Checkerboard is Better than Noise",
        "",
        "Checkerboards have:",
        "- Sharp edges that spectral methods can identify",
        "- Regular periodicity that shows up in eigenvalues",
        "- Clear visual features for PSNR comparison",
        "",
        "Noise has:",
        "- No coherent structure",
        "- Random features that don't survive spectral operations",
        "- Misleading PSNR (noise vs noise is always low PSNR)",
        "",
        "### Why Scattered Amongus is Better than Dense Tiling",
        "",
        "Scattered (random transforms):",
        "- Each copy has unique position, rotation, scale, shear",
        "- Recognizable features without dense periodicity",
        "- Tests how transforms handle isolated features",
        "",
        "Dense tiling:",
        "- Creates strong periodic signal that dominates spectrum",
        "- Hides subtle divergent effects",
        "- Not realistic test case",
        "",
        "### Transform Behavior Analysis",
        "",
    ])

    # Add transform-specific insights
    for transform in transforms:
        t_results = [r for r in valid_results if r['transform'] == transform]
        success_rate = sum(1 for r in t_results if r['success']) / len(t_results) * 100

        if 'phase_field' in transform:
            lines.append(f"**{transform}** ({success_rate:.0f}% success)")
            lines.append("- Uses arctan2 of eigenvector pairs to create spiral patterns")
            lines.append("- Creates topological defects (vortices) at eigenvector zeros")
            lines.append("")
        elif 'contour_sdf' in transform:
            lines.append(f"**{transform}** ({success_rate:.0f}% success)")
            lines.append("- Computes distance to eigenvector iso-contours")
            lines.append("- Creates smooth gradients not in discrete inputs")
            lines.append("")
        elif 'commute_time' in transform:
            lines.append(f"**{transform}** ({success_rate:.0f}% success)")
            lines.append("- Uses eigenvalue-weighted sum of squared eigenvector differences")
            lines.append("- Creates organic distance fields respecting graph topology")
            lines.append("")
        elif 'warp' in transform:
            lines.append(f"**{transform}** ({success_rate:.0f}% success)")
            lines.append("- Uses eigenvector gradients for displacement field")
            lines.append("- Coordinate remapping is inherently non-linear")
            lines.append("")
        elif 'subdivision' in transform:
            lines.append(f"**{transform}** ({success_rate:.0f}% success)")
            lines.append("- Recursively subdivides by Fiedler vector sign")
            lines.append("- Creates stained-glass patterns with operand statistics")
            lines.append("")

    lines.extend([
        "",
        "## Output Files",
        "",
        f"Results saved to: `demo_output/divergent_proper/`",
        "",
        "- `inputs/`: Carrier and operand textures",
        "- `*_texture.png`: Raw transform outputs",
        "- `*_egg.png`: Bump-mapped 3D renders (for successful/partial cases)",
        "",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nResults markdown written to: {output_path}")


if __name__ == "__main__":
    results, carriers, operands = run_proper_test()

    # Generate markdown results
    results_path = project_root / "hypercontexts" / "divergent-proper-results.md"
    generate_results_markdown(results, str(results_path))
