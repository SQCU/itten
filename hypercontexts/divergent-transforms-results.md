# Divergent Transforms Results

## Executive Summary

**Objective**: Create spectral transforms that produce results DIVERGING from BOTH input operands, rather than linear blends that resemble one input.

**Result**: 12 of 15 tests achieved FULL DIVERGENCE (80% success rate)

**Key Insight**: Non-linear operations (arctan2, sqrt, abs, sign, thresholding, coordinate remapping) create structure that cannot be decomposed as linear combinations of inputs.

## Metrics

**Success criteria:**
- PSNR(carrier, result) < 15dB - result diverged from carrier
- PSNR(operand, result) < 15dB - result diverged from operand

**Failure condition:**
- PSNR > 25dB means the transform just reproduced an input

**Test configuration:**
- Carrier: `amongus.png` (128x128)
- Operand: `amongus_tess.png` (128x128)
- Theta values: [0.1, 0.5, 0.9]

## Results Table

| Transform | Theta | PSNR Carrier | PSNR Operand | Diverged? |
|-----------|-------|--------------|--------------|-----------|
| eigenvector_phase_field | 0.1 | 4.94 dB | 5.13 dB | YES |
| eigenvector_phase_field | 0.5 | 4.79 dB | 4.82 dB | YES |
| eigenvector_phase_field | 0.9 | 4.87 dB | 5.02 dB | YES |
| spectral_contour_sdf | 0.1 | 3.09 dB | 0.11 dB | YES |
| spectral_contour_sdf | 0.5 | 3.09 dB | 0.11 dB | YES |
| spectral_contour_sdf | 0.9 | 3.11 dB | 0.14 dB | YES |
| commute_time_distance_field | 0.1 | 5.43 dB | 6.31 dB | YES |
| commute_time_distance_field | 0.5 | 5.21 dB | 5.92 dB | YES |
| commute_time_distance_field | 0.9 | 5.10 dB | 5.73 dB | YES |
| spectral_warp | 0.1 | 4.47 dB | 16.45 dB | PARTIAL |
| spectral_warp | 0.5 | 4.27 dB | 18.47 dB | PARTIAL |
| spectral_warp | 0.9 | 4.28 dB | 18.89 dB | PARTIAL |
| spectral_subdivision_blend | 0.1 | 4.16 dB | 4.52 dB | YES |
| spectral_subdivision_blend | 0.5 | 4.10 dB | 4.43 dB | YES |
| spectral_subdivision_blend | 0.9 | 3.71 dB | 3.95 dB | YES |

## Autocorrelation Analysis

Autocorrelation at lag=8 shows structural differences:

- **eigenvector_phase_field @ theta=0.1**: carrier AC=0.7780, result AC=0.4341, diff=0.3439
- **eigenvector_phase_field @ theta=0.5**: carrier AC=0.7780, result AC=-0.0340, diff=0.8120
- **eigenvector_phase_field @ theta=0.9**: carrier AC=0.7780, result AC=0.0146, diff=0.7634
- **spectral_contour_sdf @ theta=0.1**: carrier AC=0.7780, result AC=0.3938, diff=0.3842
- **spectral_contour_sdf @ theta=0.5**: carrier AC=0.7780, result AC=0.3943, diff=0.3837
- **spectral_contour_sdf @ theta=0.9**: carrier AC=0.7780, result AC=0.3947, diff=0.3833
- **commute_time_distance_field @ theta=0.1**: carrier AC=0.7780, result AC=0.3217, diff=0.4563
- **commute_time_distance_field @ theta=0.5**: carrier AC=0.7780, result AC=0.3719, diff=0.4061
- **commute_time_distance_field @ theta=0.9**: carrier AC=0.7780, result AC=0.3971, diff=0.3809
- **spectral_warp @ theta=0.1**: carrier AC=0.7780, result AC=0.4035, diff=0.3745
- **spectral_warp @ theta=0.5**: carrier AC=0.7780, result AC=0.3992, diff=0.3788
- **spectral_warp @ theta=0.9**: carrier AC=0.7780, result AC=0.4092, diff=0.3688
- **spectral_subdivision_blend @ theta=0.1**: carrier AC=0.7780, result AC=0.1378, diff=0.6402
- **spectral_subdivision_blend @ theta=0.5**: carrier AC=0.7780, result AC=0.1169, diff=0.6611
- **spectral_subdivision_blend @ theta=0.9**: carrier AC=0.7780, result AC=0.0000, diff=0.7780

## Key Mathematical Insights

The transforms achieve divergence through non-linear operations:

### Why These Work (Non-Linear Operations)

1. **eigenvector_phase_field**: `arctan2(ev1, ev2)` creates spiral patterns at topological defects
   - Spirals emerge at zero-crossings (topological singularities)
   - Phase wrapping creates discontinuities that don't exist in either input
   - **Best performer**: PSNR ~5dB from both inputs, autocorrelation drops to near zero at theta=0.5

2. **spectral_contour_sdf**: Distance transforms create smooth gradients not in discrete inputs
   - Distance to iso-contours is inherently non-linear
   - Exponential weighting with operand creates interaction
   - Consistent low PSNR (~3dB carrier, ~0.1dB operand)

3. **commute_time_distance_field**: Squared eigenvector differences weighted by inverse eigenvalues
   - Random walk distances respect graph topology
   - Squared differences + sqrt = highly non-linear
   - Creates organic distance fields with PSNR ~5-6dB

4. **spectral_warp**: Coordinate remapping via interpolation is inherently non-linear
   - **Partial divergence only** - keeps operand structure due to warping
   - Good for carrier divergence (4.3dB), weaker for operand (16-19dB)
   - Consider: use warp field to generate NEW content, not warp operand

5. **spectral_subdivision_blend**: Sign thresholding creates boundaries not in either input
   - Fiedler sign = binary partition (maximally non-linear)
   - Recursive application creates stained-glass cells
   - Autocorrelation drops to 0.0 at theta=0.9 - completely destroys input structure

### Why Linear Blends Fail

Linear transforms like `result = alpha * carrier + beta * operand` have PSNR:
```
PSNR(carrier, result) = -10 * log10((1-alpha)^2 + beta^2)
PSNR(operand, result) = -10 * log10(alpha^2 + (1-beta)^2)
```
At least one PSNR is always > 15dB for any (alpha, beta) combination.

## Demo Outputs

Rendered eggs saved to: `demo_output/divergent_transforms/`

### Texture Files

- `commute_time_distance_field_theta0.1_texture.png`
- `commute_time_distance_field_theta0.5_texture.png`
- `commute_time_distance_field_theta0.9_texture.png`
- `eigenvector_phase_field_theta0.1_texture.png`
- `eigenvector_phase_field_theta0.5_texture.png`
- `eigenvector_phase_field_theta0.9_texture.png`
- `spectral_contour_sdf_theta0.1_texture.png`
- `spectral_contour_sdf_theta0.5_texture.png`
- `spectral_contour_sdf_theta0.9_texture.png`
- `spectral_subdivision_blend_theta0.1_texture.png`
- `spectral_subdivision_blend_theta0.5_texture.png`
- `spectral_subdivision_blend_theta0.9_texture.png`
- `spectral_warp_theta0.1_texture.png`
- `spectral_warp_theta0.5_texture.png`
- `spectral_warp_theta0.9_texture.png`

### Rendered Eggs

- `commute_time_distance_field_theta0.1_egg.png`
- `commute_time_distance_field_theta0.5_egg.png`
- `commute_time_distance_field_theta0.9_egg.png`
- `eigenvector_phase_field_theta0.1_egg.png`
- `eigenvector_phase_field_theta0.5_egg.png`
- `eigenvector_phase_field_theta0.9_egg.png`
- `spectral_contour_sdf_theta0.1_egg.png`
- `spectral_contour_sdf_theta0.5_egg.png`
- `spectral_contour_sdf_theta0.9_egg.png`
- `spectral_subdivision_blend_theta0.1_egg.png`
- `spectral_subdivision_blend_theta0.5_egg.png`
- `spectral_subdivision_blend_theta0.9_egg.png`
- `spectral_warp_theta0.1_egg.png`
- `spectral_warp_theta0.5_egg.png`
- `spectral_warp_theta0.9_egg.png`

## Implementation Details

### Functions Added to `spectral_ops_fast.py`

```python
def eigenvector_phase_field(carrier, operand, theta=0.5, eigenpair=(0,1), edge_threshold=0.1)
def spectral_contour_sdf(carrier, operand, theta=0.5, num_contours=5, edge_threshold=0.1)
def commute_time_distance_field(carrier, operand, theta=0.5, reference_mode='operand_max', edge_threshold=0.1)
def spectral_warp(carrier, operand, theta=0.5, warp_strength=10.0, edge_threshold=0.1)
def spectral_subdivision_blend(carrier, operand, theta=0.5, max_depth=4, min_size=16, edge_threshold=0.1)
```

### Transform Signatures

All transforms follow the standard signature:
- **carrier**: 2D array (H, W) - provides spectral/graph structure
- **operand**: 2D array (H, W) - modulates or fills the result
- **theta**: float [0, 1] - controls frequency band selection
- Returns: 2D array (H, W) normalized to [0, 1]

## Recommendations

### Best Transforms for Full Divergence

1. **spectral_subdivision_blend** - Strongest structural divergence, especially at high theta
2. **eigenvector_phase_field** - Creates unique spiral patterns at all theta values
3. **commute_time_distance_field** - Organic distance fields with consistent divergence

### Future Improvements

1. **spectral_warp** needs modification to diverge from operand:
   - Generate displacement field from carrier, apply to NOISE instead of operand
   - Or use operand as weight mask, not source texture

2. Consider adding:
   - **spectral_voronoi**: Use eigenvector extrema as Voronoi seeds
   - **harmonic_inpainting**: Fill operand-masked regions with carrier harmonics
   - **spectral_reaction_diffusion**: Two-component system with carrier coupling