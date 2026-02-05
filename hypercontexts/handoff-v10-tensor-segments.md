# Handoff: Spectral Shader v10 - Tensor-Native Segment Operations

*Session Date: 2026-02-01*

---

## Summary

This session rebuilt the shader from first principles after diagnosing why v7g produced "running code that doesn't implement the spec." The core realization: previous implementations substituted "things that sort of do something similar" instead of the actual tensor operations specified.

Key achievement: **Segment rotation is now one matmul**, not per-pixel iteration. The pipeline properly composes scatter → gather → fuse → rotate → translate → scatter.

---

## What Works

### 1. Tensor-Native Segment Pipeline

```
scatter(mask) → coords (N, 2)
    → gather_colors(coords, img) → colors (N, 3)
    → fuse_properties(coords) → centroid, relative_coords, principal_axis
    → rotate_segment(relative, θ) → rotated (N, 2)  # ONE MATMUL
    → translate(rotated + centroid, void_dir * distance)
    → scatter_back(output, final_coords, transformed_colors)
```

### 2. Proper Thickening

- Morphological dilation via tensor shifts (no scipy)
- Color propagation from nearest original contour pixel
- Fiedler-gated: only high_gate regions get thickened
- Single pass: 567 → 6851 pixels

### 3. Connected Component Finding

- Iterative flood fill using tensor operations
- No scipy.ndimage.label dependency
- Finds contiguous curve segments as coordinate tensors

### 4. 90° Rotation via Matmul

```python
R = torch.tensor([[0, -1], [1, 0]])  # cos(90)=0, sin(90)=1
rotated_relative = relative_coords @ R.T
```

The entire segment rotates together. Curved lines become perpendicular curves.

### 5. Shadow + Front Layering

- Shadow drawn first at offset position with blue-biased cyclic color
- Front drawn second, masking shadow, with cyan-biased cyclic color
- Creates depth illusion like v6

### 6. Multi-Pass Accumulation

- Passes compound: 1113 → 7080 → 13520 → 18325 contours
- Effect strength decays (0.75^pass)
- Geometry accretes around original forms

---

## What Doesn't Work (Yet)

### 1. Width Modulation from Local Curvature

v6 had "ukiyo-e" effect where thickening varied with local graph structure. Current v10 has uniform dilation radius. The Fiedler boundary provides some asymmetry, but not the smooth width variation.

**Fix**: Modulate `thicken_radius` per-pixel based on local spectral features (e.g., Fiedler gradient magnitude, local graph curvature).

### 2. Segment Count is Low

v10 finds only 3 segments in first pass vs v6's many. The connected component finder might be too aggressive about merging, or the low_gate region is too small.

**Fix**: Tune gate_threshold, or segment based on spectral discontinuities rather than just connectivity.

### 3. Lanczos Fails on Modified Images

Multi-pass recomputes Fiedler each pass, but Lanczos fails on images with large uniform regions (created by thickening). Falls back to carrier-based pseudo-Fiedler.

**Fix**: Cache original image's Fiedler, or use iterative_spectral_transform which is more robust.

### 4. Cross-Image Shading Not Implemented

v10 only does self-shading (target=sample). The original spec has target/sample cross-attention for retrieving segments from one image to place on another.

---

## Key Files

```
spectral_shader_v10.py      # Current tensor-native implementation (~400 LOC)
  - scatter_segment()        # mask → coords
  - gather_segment_colors()  # coords → colors
  - fuse_segment_properties() # coords → centroid, relative, principal_axis
  - rotate_segment()         # relative @ R.T (the matmul!)
  - translate_segment()      # coords + translation
  - scatter_back()           # write to output with preservation
  - thicken_segment()        # dilate + color propagate
  - find_connected_components_tensor()  # flood fill via tensor shifts

spectral_shader_v9_ffn.py   # Earlier attempt with multi-head attention thickening
spectral_shader_v8.py       # Earlier attempt, lost segment rotation

hypercontexts/
  spec-shader-operations.md      # The spec we're implementing
  spec-resnet-structure.md       # Code structure principles
  handoff-v7g-cross-attention.md # Previous session state
```

---

## The Spec vs Current State

| Spec Requirement | v10 Status |
|-----------------|------------|
| Graph comparison (one matmul) | ✓ Fiedler via Lanczos |
| Sparse gating (γ*act + b vs act) | ✓ Fiedler sigmoid threshold |
| Local normal computation | ✓ Gradient of carrier |
| Thicken: orthogonal displacement | ✓ Isotropic dilation + Fiedler gating |
| Shadow: retrieve + rotate + translate | ✓ Connected components + matmul rotation |
| Preservation constraints | ✓ scatter_back checks existing_mask |
| ~10 lines main body | ✗ Still ~50 lines, could compress |

---

## Architecture Insight: Why Matmul for Rotation

The spec says operations should be tensor-to-tensor. Rotation of a point cloud is:

```
[y']   [cos θ  -sin θ] [y]
[x'] = [sin θ   cos θ] [x]
```

For N points, this is (N, 2) @ (2, 2) = (N, 2). One matmul. The entire segment—curves, corners, all of it—rotates together because matrix multiplication distributes over the point set.

This is why v6's `np.rot90(mask)` worked: it's rotating the entire 2D array. The tensor equivalent is coordinate transform via matmul, which generalizes to arbitrary angles, not just 90°.

---

## Next Steps

1. **Width modulation**: Compute per-pixel thickening radius from `|∇fiedler|` or local Laplacian curvature

2. **More segments**: Either lower gate threshold or use spectral clustering to find segment boundaries

3. **Cross-image attention**: Implement the full spec where low-gate seeds retrieve segments from sample image

4. **Compress main body**: The shader() function should be ~10-15 lines of composition, with helpers doing the work

5. **Test on other images**: snek-heavy.png, 1bit redraw.png, tonegraph

---

## Test Commands

```bash
# Run v10
uv run python spectral_shader_v10.py

# Outputs
demo_output/v10_toof_1x.png   # single pass
demo_output/v10_toof_4x.png   # four passes

# Compare to v6
uv run python resnet_spectral_shader_v6.py
# demo_output/resnet_v6_toof_labeled.png
```

---

## Session Learnings

1. **"Code runs" ≠ "code is correct"**: Typed operations (float→float) always run. The spec requires specific geometric meanings that are easy to approximate wrongly.

2. **Read the reference implementation**: v6 had connected component labeling and np.rot90 that I lost in v8/v9. Actually reading the code revealed the data flow.

3. **Tensor thinking**: Rotation is matmul. Dilation is iterative tensor shifts. Nearest-neighbor is argmin over broadcasted distance matrix. Every operation has a tensor form.

4. **Pipeline order matters**: scatter → gather → transform → scatter. You can't rotate pixels in place—you gather them, transform the gathered tensor, then scatter back.

5. **Gates have independent parameters**: High gate and low gate need separate (γ, b) pairs. Like a $13,000 hardware unit with four knobs.

---

*Previous: handoff-v7g-cross-attention.md*
*Related: spec-shader-operations.md, spec-resnet-structure.md*
