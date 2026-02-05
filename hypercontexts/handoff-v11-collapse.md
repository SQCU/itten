# Handoff: v11 Code Collapse

*Session Date: 2026-02-01*

---

## Summary

Collapsed `spectral_shader_v10_unified.py` (1076 LOC) to `spectral_shader_v11.py` (288 LOC) - 73% reduction.

---

## What Was Removed

| Pattern | v10 Lines | v11 Lines | Savings |
|---------|-----------|-----------|---------|
| Gradient computed 4x | 22 | 4 (once) | 18 |
| tangent + normal separate | 30 | 2 (derived) | 28 |
| Heat diffusion 2x | 20 | 8 (once) | 12 |
| SegmentPatch dataclass | 55 | 0 (direct indexing) | 55 |
| rotate_patch function | 62 | 8 (inline matmul) | 54 |
| extract_segment_patch | 45 | 0 (direct indexing) | 45 |
| scatter_patch | 42 | 6 (inline scatter) | 36 |
| compute_optimal_placement | 121 | 0 (simplified) | 121 |
| UnifiedConfig dataclass | 44 | 15 (inline params) | 29 |
| Color transform functions | 45 | 8 (single function) | 37 |
| Cross-attention helpers | 60 | 12 (inline matmul) | 48 |

---

## Key Tensor Patterns

### Cross-attention as matmul
```python
# v10: loop over seeds, loop over signatures
for seed in seeds:
    scores = [seed_phi @ sig for sig in sigs]
    best = argmax(scores)

# v11: single matmul
scores = seed_phi @ sigs.T  # (S, C)
best_comp = scores.abs().argmax(dim=1)  # (S,)
```

### Rotation as 2x2 matmul
```python
# v10: 62-line rotate_patch with grid_sample attempts
# v11: direct matrix multiply
R = rotation_matrix(angle.unsqueeze(0))[0]  # (2, 2)
rotated = relative @ R.T  # (N, 2)
```

### Direction fields derived once
```python
# v10: compute_edge_normal_field, compute_local_tangent_field,
#      compute_void_direction, inline gradient
# v11:
grad = gradient_2d(carrier)  # (H, W, 2)
tangent = torch.stack([-grad[..., 1], grad[..., 0]], dim=-1)  # rotate 90
void_dir = -normalize_field(gradient_2d(distance))
```

---

## What Remains

The per-seed placement loop (`for i, seed in enumerate(seeds)`) is kept because:
1. Each seed retrieves a DIFFERENT component
2. Component geometries vary in size
3. Full batching would require padding to max component size

This is acceptable - it's O(seeds), not O(pixels).

---

## Files

```
spectral_shader_v11.py      # 288 LOC - collapsed unified shader
spectral_shader_v10_unified.py  # 1076 LOC - can be deleted
spectral_shader_v8.py       # kept for cross-attention reference
```

---

## Test Results

```
Self-attention (toof):
  contours=1113, high=562, low=533, thickened=17564, seeds=21, components=19

Cross-attention (snek-heavy × red-tonegraph):
  contours=14791, high=7527, low=6994, thickened=290372, seeds=21, components=11

Cross-attention (toof × snek-heavy):
  contours=1113, high=562, low=533, thickened=17564, seeds=21, components=9
```

---

*Previous: handoff-v10.2-asymmetric.md*
