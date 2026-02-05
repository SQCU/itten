# v15 Handoff: What Went Wrong and How to Proceed

## The Failure Mode

v15 development entered an immature editing loop: making changes → output doesn't crash → declaring progress → repeat. Multiple iterations produced essentially identical or worse output while claiming success.

The rotation/flow/shadow operation was never correctly implemented despite many attempts. The outputs were either:
- Scattered confetti (point coordinate rotation destroys structure)
- Identical to non-rotation baseline (flow tracing that deposits but doesn't transform)
- Patch-based rotation that didn't align with curve geometry

## What Should Have Been Done

**The reference implementations v5 and v6 work.** They produce the correct visual output. Instead of guessing what the operation should do, the correct approach was:

```python
# Orchestration script to compare implementations
source = load_image("toof.png")

output_v6 = resnet_spectral_shader_v6.shader(source)
output_v15 = spectral_shader_v15.shader(source)

# Extract residuals (what each shader ADDED)
residual_v6 = output_v6 - source
residual_v15 = output_v15 - source

# Now compare:
# - Where do the residuals differ spatially?
# - What is the distribution of residual values?
# - What geometric properties do the v6 residuals have that v15 lacks?

# Statistics to compute:
# - residual magnitude histograms
# - spatial autocorrelation (are v6 residuals more structured/connected?)
# - orientation analysis (do v6 residuals follow curve tangents?)
```

This empirical comparison would have revealed exactly what v6's shadow operation produces that v15's attempts did not.

## Why This Was Deducible

1. **v6 exists and works** - it's presented as a "unit test" using wrong ops (scipy, loops) but producing correct output types
2. **The spec describes the visual effect** - thickening, drop shadows, rotation relative to tangent
3. **Comparing working vs broken implementations is standard debugging**

Instead of comparing outputs, v15 development:
- Tried to reimplement from spec description alone
- Made assumptions about what operations meant
- Declared success based on "doesn't crash" rather than "matches reference"

## Key Technical Insights Reached (But Not Properly Applied)

1. **Rotation requires structure**: A 1x1 pixel can't rotate. Need linkage → segmentation → then rotation is defined.

2. **Patch abstraction is wrong**: Bounding boxes of spectral segments aren't curve segments. v6 uses connected components on actual contour pixels.

3. **Flow field could work**: Tracing tangent while depositing at offset naturally aligns with curves - but implementation didn't achieve actual rotation/transformation.

4. **Spectral segmentation ≠ curve segmentation**: Eigenvector sign partitions give "regions with similar graph properties" not "pieces of the curve."

## What v6 Actually Does (That v15 Doesn't)

From `resnet_spectral_shader_v6.py:272-307`:

```python
# Extract RECTANGULAR PATCH containing segment
rgb_patch = img_rgb[y0:y1, x0:x1].copy()

# ROTATE THE ENTIRE PATCH as dense 2D array
rotated_mask = np.rot90(mask, k=1)
rotated_rgb = np.rot90(rgb_patch, k=1)

# Place at new location, respecting the rotated mask
for dy in range(rh):
    for dx in range(rw):
        if rotated_mask[dy, dx]:
            output[new_y0 + dy, new_x0 + dx] = rotated_rgb[dy, dx]
```

The key: segments come from `scipy.ndimage.label()` which finds CONNECTED COMPONENTS on actual contour pixels. These are real curve pieces. Then `rot90` rotates the patch as a coherent unit.

v15's spectral segmentation produces regions that don't correspond to curve geometry, so rotating them doesn't produce curve-aligned shadows.

## Next Steps for Future Sessions

1. **Run comparison script first** - before any more implementation, empirically measure what v6 produces vs current v15

2. **Extract what makes v6 segments "correct"** - connected components on boundary pixels, not spectral partitioning

3. **Consider hybrid approach** - use spectral basis for gating/selection, but connected components for segmentation before rotation

4. **Define success criterion** - "v15 residuals match v6 residuals within tolerance" not "code runs without errors"

## Files

- `spectral_shader_v15.py` - current state, flow-field shadow (doesn't actually rotate)
- `resnet_spectral_shader_v6.py` - reference implementation with correct output
- `spectral_shader_v8.py` - earlier reference with different approach
- `hypercontexts/spec-shader-operations.md` - the spec (describes what, not how)
- `hypercontexts/spec-resnet-structure.md` - constraints on implementation style
