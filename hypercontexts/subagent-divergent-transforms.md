# Subagent Handoff: Port Divergent Spectral Transforms

## Problem
Current transforms are LINEAR combinations of operands:
```
result ≈ α * carrier + β * operand
```
This means result is always "similar to" one of the operands.

## What We Need
Transforms where result DIVERGES from both operands:
- Different spatial covariance statistics than either input
- Different norm and mean values
- PSNR(carrier, result) AND PSNR(operand, result) both show the result is NOT just a blend

## Failure Metric
If result has high PSNR with either operand (>25dB), the transform failed - it just reproduced an input.

## Success Metric
- PSNR(carrier, result) < 15dB (diverged from carrier)
- PSNR(operand, result) < 15dB (diverged from operand)
- Autocorrelation structure differs from both inputs

## Transforms to Port (from compendium)

Read `/home/bigboi/itten/hypercontexts/spectral-transforms-compendium.md` and port these to `/home/bigboi/itten/spectral_ops_fast.py`:

### 1. Eigenvector Phase Field (Transform 4)
Non-linear: uses arctan2 of eigenvector pairs → creates spiral patterns that don't exist in either input.

### 2. Spectral Contour SDF (Transform 2)
Non-linear: computes distance to eigenvector iso-contours → creates distance field that's structurally different from input.

### 3. Commute Time Distance (Transform 18)
Non-linear: uses eigenvalue-weighted sum of squared differences → creates organic distance fields.

### 4. Spectral Warp Field (Transform 15)
Non-linear: uses eigenvector gradients to define displacement field → physically warps one image by another's structure.

### 5. Recursive Spectral Subdivision (Transform 10)
Non-linear: recursive Fiedler bisection → creates stained-glass cells that don't exist in either input.

## Implementation Location

Add functions to `/home/bigboi/itten/spectral_ops_fast.py`:
```python
def eigenvector_phase_field(carrier, operand, theta=0.5):
    """Create phase field from carrier eigenvectors, modulated by operand."""

def spectral_contour_sdf(carrier, operand, num_contours=5, theta=0.5):
    """SDF from carrier's spectral contours, weighted by operand."""

def commute_time_distance_field(carrier, operand, reference_mode='operand_max'):
    """Commute time from reference point, carrier provides graph."""

def spectral_warp(carrier, operand, warp_strength=0.5, theta=0.5):
    """Warp operand by carrier's eigenvector gradient flow."""

def spectral_subdivision_blend(carrier, operand, max_depth=4, theta=0.5):
    """Recursively subdivide by carrier's Fiedler, fill with operand values."""
```

## Testing

For each transform:
1. Load carrier from `demo_output/3d_psnr/` (the nice color textures)
2. Load operand (use another image from same dir, or amongus)
3. Apply transform at theta=[0.1, 0.5, 0.9]
4. Compute:
   - PSNR(carrier, result)
   - PSNR(operand, result)
   - Autocorrelation at lag=[1,4,8,16]
   - Mean and std of result
5. Verify DIVERGENCE: both PSNRs < 15dB

## Render

Use `render_bumped_egg()` to render results as 3D bump maps.
Save to `/home/bigboi/itten/demo_output/divergent_transforms/`

## Output
- Updated `/home/bigboi/itten/spectral_ops_fast.py` with 3-5 new transforms
- Test script `/home/bigboi/itten/tests/test_divergent_transforms.py`
- Results `/home/bigboi/itten/hypercontexts/divergent-transforms-results.md`
- Demo renders in `/home/bigboi/itten/demo_output/divergent_transforms/`
