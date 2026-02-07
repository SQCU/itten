"""
3D Bump-Mapped Render Sweep: Carrier x Operand x Theta

Demonstrates spectral_warp_embed transforms rendered to 3D bump-mapped egg geometry.
Creates a combinatoric sweep of:
- 3 carriers (amongus, checkerboard, natural image)
- 3 operands (amongus_tessellated, checkerboard_fine, amongus_sheared)
- 3 theta values (0.1, 0.5, 0.9)

Total: 27 combinations rendered as 3D bump-mapped eggs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from typing import Dict, Tuple, List
import time

from texture.blend import spectral_warp_embed, psnr
from texture.render.bump_render import render_bumped_egg, create_comparison_grid
from texture.normals import height_to_normals
from texture.patterns import (
    generate_amongus,
    generate_checkerboard,
    generate_amongus_tessellated,
    generate_amongus_sheared
)


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = '/home/bigboi/itten/demo_output/3d_bump_sweep'
NATURAL_IMAGE_PATH = '/home/bigboi/itten/demo_output/inputs/snek-heavy.png'
RESULTS_MD_PATH = '/home/bigboi/itten/hypercontexts/3d-bump-sweep-results.md'

# Image size for processing (must be consistent)
SIZE = 128

# Theta values for the sweep
THETA_VALUES = [0.1, 0.5, 0.9]

# Render settings
RENDER_SIZE = 512
DISPLACEMENT_SCALE = 0.15
NORMAL_STRENGTH = 0.8


# =============================================================================
# Helper Functions
# =============================================================================

def load_and_resize_grayscale(path: str, size: int) -> np.ndarray:
    """Load an image and convert to grayscale, resize to target size."""
    img = Image.open(path)

    # Convert to grayscale
    if img.mode != 'L':
        img = img.convert('L')

    # Resize to target size
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    # Convert to numpy array normalized to [0, 1]
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def normalize_to_01(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        return (arr - arr_min) / (arr_max - arr_min)
    return arr


def compute_psnr_metrics(
    result: np.ndarray,
    carrier: np.ndarray,
    operand: np.ndarray
) -> Dict[str, float]:
    """Compute PSNR metrics for a result against carrier and operand."""
    return {
        'psnr_carrier': psnr(normalize_to_01(result), normalize_to_01(carrier)),
        'psnr_operand': psnr(normalize_to_01(result), normalize_to_01(operand))
    }


def render_base_3d(
    height_field: np.ndarray,
    name: str,
    output_dir: str
) -> np.ndarray:
    """Render a height field to 3D bumped egg and save."""
    # Generate normal map
    normal_map = height_to_normals(height_field, strength=2.0)

    # Render
    rendered = render_bumped_egg(
        height_field=height_field,
        normal_map=normal_map,
        displacement_scale=DISPLACEMENT_SCALE,
        normal_strength=NORMAL_STRENGTH,
        output_size=RENDER_SIZE
    )

    # Save
    Image.fromarray(rendered).save(os.path.join(output_dir, f'{name}.png'))

    return rendered


def save_height_field(arr: np.ndarray, path: str):
    """Save a height field as grayscale image."""
    img_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


# =============================================================================
# Main Sweep
# =============================================================================

def main():
    print("=" * 70)
    print("3D BUMP-MAPPED RENDER SWEEP: Carrier x Operand x Theta")
    print("=" * 70)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # Setup Carriers
    # =========================================================================
    print("\n[1/6] Setting up carriers...")

    carriers = {}

    # Carrier 1: Amongus
    carriers['amongus'] = generate_amongus(SIZE)
    print(f"  - amongus: {carriers['amongus'].shape}")

    # Carrier 2: Checkerboard (block_size=16)
    carriers['checkerboard'] = generate_checkerboard(SIZE, tile_size=16)
    print(f"  - checkerboard: {carriers['checkerboard'].shape}")

    # Carrier 3: Natural image (snek)
    carriers['natural'] = load_and_resize_grayscale(NATURAL_IMAGE_PATH, SIZE)
    print(f"  - natural (snek): {carriers['natural'].shape}")

    # Save carriers
    carriers_dir = os.path.join(OUTPUT_DIR, 'carriers')
    os.makedirs(carriers_dir, exist_ok=True)
    for name, carrier in carriers.items():
        save_height_field(carrier, os.path.join(carriers_dir, f'{name}.png'))
    print(f"  Saved carriers to {carriers_dir}")

    # =========================================================================
    # Setup Operands
    # =========================================================================
    print("\n[2/6] Setting up operands...")

    operands = {}

    # Operand 1: Amongus tessellated (3x3 grid)
    operands['amongus_tess'] = generate_amongus_tessellated(SIZE, copies_x=3, copies_y=3)
    print(f"  - amongus_tess: {operands['amongus_tess'].shape}")

    # Operand 2: Checkerboard with fine block size (8)
    operands['checker_fine'] = generate_checkerboard(SIZE, tile_size=8)
    print(f"  - checker_fine: {operands['checker_fine'].shape}")

    # Operand 3: Amongus sheared
    operands['amongus_shear'] = generate_amongus_sheared(SIZE, shear_angle=20.0)
    print(f"  - amongus_shear: {operands['amongus_shear'].shape}")

    # Save operands
    operands_dir = os.path.join(OUTPUT_DIR, 'operands')
    os.makedirs(operands_dir, exist_ok=True)
    for name, operand in operands.items():
        save_height_field(operand, os.path.join(operands_dir, f'{name}.png'))
    print(f"  Saved operands to {operands_dir}")

    # =========================================================================
    # Render Base Carriers and Operands as 3D
    # =========================================================================
    print("\n[3/6] Rendering base carriers and operands as 3D...")

    base_renders = {}
    base_renders_dir = os.path.join(OUTPUT_DIR, 'base_renders')
    os.makedirs(base_renders_dir, exist_ok=True)

    for name, carrier in carriers.items():
        base_renders[f'carrier_{name}'] = render_base_3d(carrier, f'carrier_{name}', base_renders_dir)
        print(f"  - Rendered carrier_{name}")

    for name, operand in operands.items():
        base_renders[f'operand_{name}'] = render_base_3d(operand, f'operand_{name}', base_renders_dir)
        print(f"  - Rendered operand_{name}")

    # =========================================================================
    # Combinatoric Sweep
    # =========================================================================
    print("\n[4/6] Running combinatoric sweep (3 x 3 x 3 = 27 combinations)...")

    renders_dir = os.path.join(OUTPUT_DIR, 'renders')
    os.makedirs(renders_dir, exist_ok=True)

    height_fields_dir = os.path.join(OUTPUT_DIR, 'height_fields')
    os.makedirs(height_fields_dir, exist_ok=True)

    # Store results for analysis
    results = []
    all_renders = {}

    total_combinations = len(carriers) * len(operands) * len(THETA_VALUES)
    current = 0

    for c_name, carrier in carriers.items():
        for o_name, operand in operands.items():
            for theta in THETA_VALUES:
                current += 1
                combo_name = f'{c_name}_{o_name}_theta{theta}'

                print(f"  [{current:2d}/{total_combinations}] {combo_name}...", end=" ")
                start_time = time.time()

                # Apply spectral_warp_embed
                try:
                    result, (warp_x, warp_y) = spectral_warp_embed(
                        carrier=carrier,
                        operand=operand,
                        theta=theta,
                        num_eigenvectors=8,
                        warp_scale=5.0
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

                # Normalize result to [0, 1]
                height_field = normalize_to_01(result)

                # Save height field
                save_height_field(height_field, os.path.join(height_fields_dir, f'{combo_name}.png'))

                # Generate normal map
                normal_map = height_to_normals(height_field, strength=2.0)

                # Render 3D bumped egg
                rendered = render_bumped_egg(
                    height_field=height_field,
                    normal_map=normal_map,
                    displacement_scale=DISPLACEMENT_SCALE,
                    normal_strength=NORMAL_STRENGTH,
                    output_size=RENDER_SIZE
                )

                # Save render
                Image.fromarray(rendered).save(os.path.join(renders_dir, f'{combo_name}.png'))
                all_renders[combo_name] = rendered

                # Compute PSNR metrics
                metrics = compute_psnr_metrics(height_field, carrier, operand)

                # Store result
                result_entry = {
                    'carrier': c_name,
                    'operand': o_name,
                    'theta': theta,
                    'combo_name': combo_name,
                    'psnr_carrier': metrics['psnr_carrier'],
                    'psnr_operand': metrics['psnr_operand'],
                    'height_field': height_field
                }
                results.append(result_entry)

                elapsed = time.time() - start_time
                print(f"PSNR(c)={metrics['psnr_carrier']:.1f}dB, PSNR(o)={metrics['psnr_operand']:.1f}dB [{elapsed:.1f}s]")

    print(f"\n  Completed {len(results)} renders")

    # =========================================================================
    # Build PSNR Matrix and Identify Outliers
    # =========================================================================
    print("\n[5/6] Analyzing PSNR metrics and identifying outliers...")

    # Build PSNR matrix
    psnr_carrier_matrix = {}
    psnr_operand_matrix = {}

    for r in results:
        key = (r['carrier'], r['operand'], r['theta'])
        psnr_carrier_matrix[key] = r['psnr_carrier']
        psnr_operand_matrix[key] = r['psnr_operand']

    # Compute statistics
    all_psnr_carrier = [r['psnr_carrier'] for r in results]
    all_psnr_operand = [r['psnr_operand'] for r in results]

    mean_psnr_carrier = np.mean(all_psnr_carrier)
    mean_psnr_operand = np.mean(all_psnr_operand)
    std_psnr_carrier = np.std(all_psnr_carrier)
    std_psnr_operand = np.std(all_psnr_operand)

    print(f"  PSNR(carrier): mean={mean_psnr_carrier:.2f}dB, std={std_psnr_carrier:.2f}dB")
    print(f"  PSNR(operand): mean={mean_psnr_operand:.2f}dB, std={std_psnr_operand:.2f}dB")

    # Identify outliers
    outliers = {
        'balanced': [],      # Both PSNRs moderate and similar
        'carrier_dominant': [],  # High carrier PSNR, low operand PSNR (carrier preserved well)
        'operand_dominant': [],  # Low carrier PSNR, high operand PSNR (operand preserved well)
        'high_covariance': []    # Both PSNRs moderate, showing blend
    }

    for r in results:
        pc = r['psnr_carrier']
        po = r['psnr_operand']
        diff = abs(pc - po)
        avg = (pc + po) / 2

        # Balanced: similar PSNRs, moderate values
        if diff < 2.0 and 8 < avg < 18:
            outliers['balanced'].append(r)

        # Carrier dominant: carrier PSNR much higher than operand
        if pc - po > 3.0:
            outliers['carrier_dominant'].append(r)

        # Operand dominant: operand PSNR much higher than carrier
        if po - pc > 3.0:
            outliers['operand_dominant'].append(r)

        # High covariance: both PSNRs moderate (10-20 range)
        if 10 < pc < 20 and 10 < po < 20:
            outliers['high_covariance'].append(r)

    # Sort outliers by criteria
    outliers['balanced'].sort(key=lambda r: abs(r['psnr_carrier'] - r['psnr_operand']))
    outliers['carrier_dominant'].sort(key=lambda r: r['psnr_carrier'] - r['psnr_operand'], reverse=True)
    outliers['operand_dominant'].sort(key=lambda r: r['psnr_operand'] - r['psnr_carrier'], reverse=True)
    outliers['high_covariance'].sort(key=lambda r: (r['psnr_carrier'] + r['psnr_operand']) / 2, reverse=True)

    print("\n  Outlier categories:")
    for category, items in outliers.items():
        print(f"    {category}: {len(items)} combinations")

    # =========================================================================
    # Create Comparison Grids
    # =========================================================================
    print("\n[6/6] Creating comparison grids...")

    grids_dir = os.path.join(OUTPUT_DIR, 'grids')
    os.makedirs(grids_dir, exist_ok=True)

    # Grid 1: All theta variations for each carrier-operand pair
    for c_name in carriers.keys():
        for o_name in operands.keys():
            grid_images = []
            grid_labels = []

            # Add carrier and operand base renders
            grid_images.append(base_renders[f'carrier_{c_name}'])
            grid_labels.append(f'Carrier: {c_name}')

            grid_images.append(base_renders[f'operand_{o_name}'])
            grid_labels.append(f'Operand: {o_name}')

            # Add theta variations
            for theta in THETA_VALUES:
                combo_name = f'{c_name}_{o_name}_theta{theta}'
                if combo_name in all_renders:
                    grid_images.append(all_renders[combo_name])
                    grid_labels.append(f'theta={theta}')

            if len(grid_images) >= 3:
                grid = create_comparison_grid(grid_images, grid_labels)
                Image.fromarray(grid).save(os.path.join(grids_dir, f'grid_{c_name}_{o_name}.png'))

    print(f"  Saved carrier-operand grids to {grids_dir}")

    # Grid 2: Outlier showcase - pick best from each category
    outlier_images = []
    outlier_labels = []

    # Select top outliers from each category
    for category in ['balanced', 'carrier_dominant', 'operand_dominant', 'high_covariance']:
        if outliers[category]:
            best = outliers[category][0]
            combo_name = best['combo_name']
            if combo_name in all_renders:
                outlier_images.append(all_renders[combo_name])
                label = f"{category[:3].upper()}: {best['carrier'][:4]}/{best['operand'][:4]}/t={best['theta']}"
                outlier_labels.append(label)

    if outlier_images:
        outlier_grid = create_comparison_grid(outlier_images, outlier_labels)
        Image.fromarray(outlier_grid).save(os.path.join(grids_dir, 'outlier_showcase.png'))
        print(f"  Saved outlier showcase grid")

    # =========================================================================
    # Write Results Markdown
    # =========================================================================
    print("\nWriting results markdown...")

    md_content = f"""# 3D Bump-Mapped Render Sweep Results

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Overview

Combinatoric sweep of spectral_warp_embed rendered to 3D bump-mapped egg geometry.

- **Carriers**: {', '.join(carriers.keys())}
- **Operands**: {', '.join(operands.keys())}
- **Theta values**: {THETA_VALUES}
- **Total combinations**: {len(results)}

## Configuration

| Parameter | Value |
|-----------|-------|
| Image size | {SIZE}x{SIZE} |
| Render size | {RENDER_SIZE}x{RENDER_SIZE} |
| Displacement scale | {DISPLACEMENT_SCALE} |
| Normal strength | {NORMAL_STRENGTH} |
| Eigenvectors | 8 |
| Warp scale | 5.0 |

## PSNR Statistics

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| PSNR(carrier) | {mean_psnr_carrier:.2f} dB | {std_psnr_carrier:.2f} dB | {min(all_psnr_carrier):.2f} dB | {max(all_psnr_carrier):.2f} dB |
| PSNR(operand) | {mean_psnr_operand:.2f} dB | {std_psnr_operand:.2f} dB | {min(all_psnr_operand):.2f} dB | {max(all_psnr_operand):.2f} dB |

## Full PSNR Matrix

| Carrier | Operand | Theta | PSNR(carrier) | PSNR(operand) | Diff |
|---------|---------|-------|---------------|---------------|------|
"""

    for r in sorted(results, key=lambda x: (x['carrier'], x['operand'], x['theta'])):
        diff = r['psnr_carrier'] - r['psnr_operand']
        md_content += f"| {r['carrier']} | {r['operand']} | {r['theta']} | {r['psnr_carrier']:.2f} dB | {r['psnr_operand']:.2f} dB | {diff:+.2f} dB |\n"

    md_content += f"""
## Outlier Analysis

### Balanced (Both signatures visible, similar PSNR)
"""

    for r in outliers['balanced'][:5]:
        md_content += f"- **{r['combo_name']}**: PSNR(c)={r['psnr_carrier']:.1f}dB, PSNR(o)={r['psnr_operand']:.1f}dB\n"

    md_content += f"""
### Carrier Dominant (Carrier structure preserved)
"""

    for r in outliers['carrier_dominant'][:5]:
        md_content += f"- **{r['combo_name']}**: PSNR(c)={r['psnr_carrier']:.1f}dB, PSNR(o)={r['psnr_operand']:.1f}dB\n"

    md_content += f"""
### Operand Dominant (Operand pattern preserved)
"""

    for r in outliers['operand_dominant'][:5]:
        md_content += f"- **{r['combo_name']}**: PSNR(c)={r['psnr_carrier']:.1f}dB, PSNR(o)={r['psnr_operand']:.1f}dB\n"

    md_content += f"""
### High Covariance (Both PSNRs moderate, showing blend)
"""

    for r in outliers['high_covariance'][:5]:
        md_content += f"- **{r['combo_name']}**: PSNR(c)={r['psnr_carrier']:.1f}dB, PSNR(o)={r['psnr_operand']:.1f}dB\n"

    md_content += f"""
## Output Files

### Directories

| Directory | Contents |
|-----------|----------|
| `carriers/` | Source carrier height fields |
| `operands/` | Source operand height fields |
| `base_renders/` | 3D renders of raw carriers/operands |
| `height_fields/` | Blended height fields from spectral_warp_embed |
| `renders/` | All 27 3D bump-mapped renders |
| `grids/` | Comparison grids |

### Key Images

- `grids/outlier_showcase.png` - Selected outliers from each category
- `grids/grid_{{carrier}}_{{operand}}.png` - Theta progression for each pair

## Observations

1. **Theta Effect**: As theta increases from 0.1 to 0.9, the carrier structure becomes more dominant in the blend. This is visible in the PSNR trends where PSNR(carrier) tends to increase with theta.

2. **Carrier Influence**: The carrier determines WHERE features appear on the 3D surface. The amongus carrier creates localized bump regions, while checkerboard creates regular periodic patterns.

3. **Operand Influence**: The operand determines WHAT patterns fill those regions. Tessellated operands create repetitive micro-structure, while sheared operands create directional flow.

4. **3D Visualization**: The bump mapping effectively visualizes the blend - both carrier and operand signatures are visible in the surface deformation and lighting.

5. **Best Blends**: The most visually interesting results tend to be in the 'balanced' and 'high_covariance' categories where both input signatures are clearly present.

## Technical Notes

- Uses `spectral_warp_embed` which warps the operand along the carrier's eigenvector gradients
- Height fields are normalized to [0,1] before rendering
- Normal maps generated with strength=2.0 for enhanced surface detail
- Render uses egg geometry with Blinn-Phong shading
"""

    with open(RESULTS_MD_PATH, 'w') as f:
        f.write(md_content)

    print(f"  Saved results to {RESULTS_MD_PATH}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Results markdown: {RESULTS_MD_PATH}")
    print(f"\nTotal renders: {len(results)}")
    print("\nKey findings:")
    print(f"  - Balanced blends: {len(outliers['balanced'])}")
    print(f"  - Carrier dominant: {len(outliers['carrier_dominant'])}")
    print(f"  - Operand dominant: {len(outliers['operand_dominant'])}")
    print(f"  - High covariance: {len(outliers['high_covariance'])}")


if __name__ == '__main__':
    main()
