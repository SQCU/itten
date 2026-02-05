# Handoff: v12 Cross-Attention Mystery

*Session Date: 2026-02-02*

---

## The Paradox

**Self-attention (v6 style)** produces:
- Fewer, larger segments (15 components for toof)
- Long coherent strokes
- Visually lush results

**Cross-attention** produces:
- More, smaller segments (even with identical placement logic)
- Fragmented, "hairy" copies
- Worse visual quality

This is backwards from expectation. Cross-attention should find BETTER matches (more options in sample space), yet produces WORSE results.

---

## What Was Tried

### 1. Seed-based retrieval (v8 style)
```
for each low_gate PIXEL:
    cross-attend to sample components
    copy best match
```
**Result**: 159+ seeds each copying the same component → massive over-copying

### 2. Component-to-component matching
```
for each SAMPLE component:
    find best target location
    place once
```
**Result**: All components stack at same "best" location (bottom-right corner)

### 3. Target-to-sample matching with used_targets tracking
```
for each SAMPLE component:
    find best TARGET component (skip used)
    place at target centroid
```
**Result**: Dashed-line appearance, global-scale features dropped randomly

### 4. SwiGLU-style gating
```
combined_gate = sigmoid(self_activity) * sigmoid(cross_activity)
placement_mask = low_gate & (combined > mean)
```
**Result**: 53-152 tiny fragmented placements vs 15 large self-attention placements

### 5. Unified placement logic (current)
```
# WHERE: target's low_gate components (same as self)
# WHAT: sample components via cross-attention
for each TARGET component:
    find best SAMPLE component
    copy with affinity gating
```
**Result**: Same component COUNT as self (15), but segments are still shorter/worse

---

## The Strange Observation

With identical placement logic:
- Self: copies TARGET component to nearby void → long strokes
- Cross: copies SAMPLE component to target location → short fragments

**Same 15 target components. Same placement locations. Different segment lengths.**

The difference is WHAT gets copied:
- Self: the target component itself (already fits the location)
- Cross: a sample component (may not fit the location's geometry)

---

## Hypotheses

### H1: Affinity gating is too aggressive for cross-attention
```python
copy_gate = affinity > affinity.mean()
```
For self-attention, segment φ and seed φ are from the SAME image, so affinity is naturally high across most of the segment.

For cross-attention, sample φ and target φ are from DIFFERENT spectral bases. Affinity might be systematically lower, causing more aggressive truncation.

**Test**: Compare affinity distributions for self vs cross cases.

### H2: Sample components don't match target geometry
When we copy a sample component to a target location:
- The sample component has its own shape/orientation
- The target location has different local geometry
- The mismatch causes visual fragmentation

v6 avoids this by always copying from self (component fits its own location).

### H3: Spectral basis mismatch
φ_t and φ_s are eigenvectors of DIFFERENT Laplacians. The dot product `φ_t @ φ_s.T` may not be meaningful in the same way as `φ_t @ φ_t.T`.

Cross-attention between different spectral bases might need:
- Basis alignment (Procrustes)
- Learned projection
- Different similarity metric

### H4: Scale mismatch
Sample components might be at different scales than target components. Copying a large sample component to a small target location (or vice versa) causes fragmentation.

---

## What v6 Does Differently

v6's shadow transform:
1. Finds segments in LOW-GATE region of SAME image
2. Rotates 90°
3. Translates by gradient direction
4. Places with color transform

Key insight: **v6 never does cross-image attention for shadows**. It only uses spectral gating to SELECT regions, then copies from self.

The "cross" in v6 is target×sample for ACTIVATION (Fiedler gating), not for COPYING.

---

## Possible Fixes (Not Yet Tried)

### A: Copy target structure, color from sample
Instead of copying sample GEOMETRY, copy target geometry but RECOLOR using sample's palette/statistics.

### B: Warp sample to target
Use the spectral correspondence to WARP sample components to fit target geometry before copying.

### C: Abandon cross-component copying
Use cross-attention only for:
- Gating (which regions are active)
- Color selection
Keep geometry copying as self-attention only.

### D: Investigate the affinity distribution
Print/visualize affinity values for self vs cross. If cross affinities are systematically lower, adjust the gating threshold.

---

## Code State

```
spectral_shader_v12.py  ~320 lines
- thicken(): working, spectral-modulated perpendicular scatter
- shadow():
  - self path: v6-style, 15 components, good results
  - cross path: same placement logic, shorter/worse segments
```

---

## Key Question

Why does copying SAMPLE structure produce worse results than copying TARGET structure, even when:
- Placement locations are identical
- Component counts are identical
- Affinity gating is identical

The answer likely involves spectral basis compatibility or geometric fit.

---

*Previous: handoff-v11-architectural-pivot.md*
