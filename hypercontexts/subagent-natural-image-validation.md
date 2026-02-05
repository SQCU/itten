# Subagent Handoff: Natural Image Spectral Validation

## Problem
Previous tests used noise and checkerboard images which are pathological:
- Periodic features at same scale across entire image
- Hard to interpret spectral differences
- Not representative of "natural" structure

## Natural Images Available
`/home/bigboi/itten/demo_output/inputs/`:
- `1bit redraw.png` - binary with varied shapes
- `mspaint-boykisser-i-couldnt-forget-iii.png` - character art
- `mspaint-enso-i-couldnt-forget-ii.png` - circular brush stroke
- `mspaint-enso-i-couldnt-forget.png` - circular brush stroke variant
- `snek-heavy.png` - larger image with detail
- `toof.png` - character art

## Mission
Validate spectral transforms on natural images to show they work on non-pathological inputs.

## Tests to Run

### 1. Carrier-Operand Blend with Natural Images
```python
# Use natural image as carrier, different natural image as operand
carrier = load_image('snek-heavy.png')
operand = load_image('toof.png')

for theta in [0.1, 0.5, 0.9]:
    result = spectral_embed(carrier, operand, theta)
    # Measure PSNR to both inputs
```

### 2. Autocorrelation Structure Analysis
Natural images have NON-UNIFORM autocorrelation:
- Compare autocorrelation at different lags
- Show that transforms change the structure (not just intensity)

### 3. Transform Comparison Grid
For each natural image:
- Original
- eigenvector_phase_field at theta=0.1, 0.5, 0.9
- spectral_embed with noise operand at theta=0.1, 0.5, 0.9
- spectral_warp_embed at theta=0.1, 0.5, 0.9

### 4. PSNR Covariance Matrix
Build matrix: natural_images × transforms × theta
Show that covariance exists (transforms respond differently to different natural images)

## Success Criteria
- Transforms produce visible changes on natural images
- PSNR variance across theta is meaningful (not just noise)
- Different natural images produce different PSNR profiles
- Autocorrelation structure changes with theta (not just scales)

## Output
- `/home/bigboi/itten/tests/test_natural_images.py`
- `/home/bigboi/itten/demo_output/natural_image_transforms/`
  - Comparison grids for each image
  - PSNR matrix heatmap
- `/home/bigboi/itten/hypercontexts/natural-image-validation-results.md`
