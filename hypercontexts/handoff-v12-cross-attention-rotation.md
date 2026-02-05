# Handoff: v12 Cross-Attention Rotation/Translation

*Session Date: 2026-02-03*

---

## Current State

`spectral_shader_v12.py` now has working cross-attention that:
1. Computes counterfactual self-attention (what self WOULD place)
2. Matches sample components to counterfactual placements via spectral dot product
3. Copies sample fragment geometry to target locations
4. Fragment intrinsic properties (length, axis, centroid) computed at selection time

**Baseline output**: Fragments placed at destination centroids with original sample orientation, no rotation, no translation.

---

## What Works

### Procrustes Alignment
Aligning φ_s to φ_t's coordinate frame before cross-attention. Cheap (O(k³) for k eigenvectors), makes cross-attention scores meaningful.

### Counterfactual Self-Attention
Computing what self-attention WOULD produce, then using those placements as queries to find matching sample components. This separates WHERE (from self) from WHAT (from sample).

### Component Signatures
Pre-segmenting sample into components, computing mean spectral signature per component, doing argmax selection. This gives discrete shape copying instead of color blending.

### Fragment Intrinsic Properties
Computing length/axis from sample component's own pixel positions at selection time, not from destination positions later.

---

## What Was Eliminated

### 1. Dense Pixel-to-Pixel Attention
**Tried**: `A = phi_t @ phi_s.T` for all pixels
**Problem**: OOM (256GB for moderate images), and produces color blending not shape copying
**Eliminated because**: Wrong granularity entirely

### 2. Attention Without Magnitude Gating
**Tried**: Row-normalized attention writes to all target pixels
**Problem**: Every pixel gets painted (uniform color wash)
**Eliminated because**: Normalization guarantees weight=1 regardless of match quality

### 3. Negative Space as Target Mask
**Tried**: `target_mask = ~contours` (all non-contour pixels)
**Problem**: 250K target pixels, writes everywhere
**Eliminated because**: Wrong target selection - should be sparse seeds, not all negative space

### 4. Void Direction for Translation
**Tried**: Heat diffusion from contours, gradient points "away from ink"
**Problem**: Points INTO locally closed regions (hair, enclosed shapes), not into open space
**Eliminated because**: Topologically misleading - "away from nearest contour" ≠ "into open geometry"

### 5. Fiedler Gradient for Translation Direction
**Tried**: `sign(fiedler_value) × fiedler_gradient` to push "deeper into partition"
**Problem**: Partition structure ≠ geometric openness. Pushed fragments into dense regions.
**Eliminated because**: Graph partition is not a proxy for spatial density

### 6. 4-Region Spectral Partition
**Tried**: `sign(fiedler) × sign(ev2)` to create coarse regions, snap to "brightest" region
**Problem**: Arbitrary region count, brightness heuristic, pile-ups at boundaries
**Eliminated because**: Not principled, reintroduces pixel-space heuristics

### 7. Heat Threshold for Negative Space
**Tried**: `heat < heat.median()` or `heat < quantile(heat, 0.3)`
**Problem**: Heat distribution heavily skewed (median often 0), threshold meaningless
**Eliminated because**: Distribution shape breaks the threshold approach

### 8. Per-Pixel Nudge (Non-Uniform)
**Tried**: Computing nudge direction/magnitude per destination pixel
**Problem**: Fragments become sparse/fragmented as pixels get different displacements
**Eliminated because**: Breaks segment coherence

### 9. Target Tangent for Rotation Direction
**Tried**: `tan_y, tan_x` at destination centroid to define "local curve orthogonal"
**Problem**: All fragments rotate same direction (tangent field is smooth), and direction was INTO image not away
**Eliminated because**: Target tangent alone doesn't capture per-fragment context

### 10. Sample Tangent + Target Tangent Matching
**Tried**: Get sample tangent at fragment origin, target tangent at destination, rotate to "clash"
**Problem**: Still produces uniform inward rotation
**Eliminated because**: The tangent fields encode partition structure, not curve direction in the intuitive sense

---

## Open Questions

### How to Define "Orthogonal to Local Curvature"?
The goal: fragments should clash with (be perpendicular to) the local curve direction at their placement site.

**Not answered by**:
- Fiedler gradient (partition boundary, not curve tangent)
- Void direction (topologically confused by enclosed regions)
- Target tangent field (smooth, doesn't give per-fragment variation)

**Possibly answered by**:
- The relationship between fragment shape and surrounding placed fragments
- A second spectral analysis on the placed-fragments image
- Something about the SOURCE component's context in the sample image

### How to Determine Translation Direction (+/-)?
Given an orthogonal direction, which way along it should fragments translate?

**Not answered by**:
- Fiedler sign (partition membership ≠ "outward")
- Heat/void direction (points into enclosed regions)
- Carrier brightness (pixel heuristic)

**Possibly answered by**:
- Global image structure (toward boundaries?)
- Density of other fragments (away from clusters?)
- Something intrinsic to the fragment's role in sample image

### Should Rotation Be Per-Fragment or Global?
Current approaches give either:
- Uniform rotation (all fragments same direction)
- No meaningful variation

Need: rotation that varies based on each fragment's local context

---

## Code Structure Notes

### x/y Notation Problem
Separate `tan_y, tan_x` variables obscure shape mismatches. Better to use `(N, 2)` tensors with column indices.

### In-Place Mutation Problem
Writing to `output` during iteration causes fragments to overwrite each other. Should collect into sparse buffers, collapse at end.

### Stateless Function Decomposition
```python
def place_fragment(src_idx, sample_comp, sample_img, src_centroid):
    """Returns (positions: (N,2), colors: (N,3), length, axis)"""

def compute_rotation(frag_axis, local_context) -> rotation_matrix:
    """Returns (2,2) rotation matrix"""

def translate_fragment(positions, direction, distance):
    """Returns new positions (N,2)"""

def collapse_buffers(list_of_positions, list_of_colors, H, W):
    """Sparse write to output. Returns (H,W,3)"""
```

---

## Files

- `spectral_shader_v12.py`: Current implementation
- `spectral_shader_v8.py`: Reference for seed→component→place structure
- `resnet_spectral_shader_v6.py`: Reference for self-attention shadow that works

---

*Previous: handoff-v12-cross-attention-mystery.md*
