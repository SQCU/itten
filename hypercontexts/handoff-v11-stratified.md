# Handoff: Spectral Shader v11 - Stratified Thickening & Energy-Minimizing Shadows

*Session Date: 2026-02-02*

---

## Summary

Collapsed v10 (1076 LOC) to v11 (484 LOC) with proper tensor-native operations. Added V-projection style stratified thickening to prevent bunching, and energy-minimizing shadow placement with structure tensor tangent for true perpendicular rotation.

---

## What Works

### 1. Thickening (High Gate) - V-Projection Stratified

```python
# Gamma penalty on short-range spectral activity
high_freq_energy = (phi_t[src_flat, 2:] ** 2).sum(dim=1)
affinity = gamma_penalty ** (high_freq_norm * 3)  # gamma=0.85

# Stratified budget allocation
strata = [(0.25, 0.40),  # top 25% affinity → 40% budget
          (0.25, 0.30),  # next 25% → 30%
          (0.25, 0.20),  # next 25% → 20%
          (0.25, 0.10)]  # bottom 25% → 10%
```

**Effect**: Prevents self-amplifying bunching. High spectral activity regions get less thickening, forcing spread to quieter areas.

### 2. Shadow Placement (Low Gate) - Energy Minimizing

```python
# Energy terms
E_seed = |dot(rot_axis, seed_tangent)|²  # perpendicular to LOCAL contour (w=4.0)
E_src  = |dot(rot_axis, frag_axis)|²     # perpendicular to source (w=2.0)
E_loc  = |dot(rot_axis, local_tan)|      # perpendicular to landing geometry (w=1.0)
E_int  = overlap * (1 - orthog_discount) # avoid intersection (w=1.5)
E_void = -distance_field                  # seek emptiness (w=0.8)
```

**Key fix**: Tangent computed from **structure tensor of contour geometry**, not carrier gradient:
```python
contour_grad = gradient_2d(contours.float())
Jxx, Jyy, Jxy = smoothed_outer_product(contour_grad)
theta = 0.5 * atan2(2*Jxy, Jxx - Jyy)
tangent = [sin(theta), cos(theta)]  # actual edge direction
```

### 3. Cross-Attention Retrieval

```python
scores = seed_phi @ sigs.T  # (S, C) - single matmul
best_comp = scores.abs().argmax(dim=1)
```

Component signatures computed via scatter-mean over spectral embeddings.

---

## Growth Metrics (toof 4-pass)

| Pass | Contours | Thickened | Ratio |
|------|----------|-----------|-------|
| 1 | 1,113 | 1,960 | 1.8x |
| 2 | 2,563 | 4,284 | 1.7x |
| 3 | 4,359 | 6,678 | 1.5x |
| 4 | 6,441 | 11,340 | 1.8x |

Growth ratio stable at ~1.5-1.8x per pass (was explosive before stratification).

---

## TODO

### Vectorize the Thickener

Current budget-limited acceptance has a Python loop:
```python
for m in range(M):
    # Accept closest candidates up to budget_per_source[m]
```

Should be replaced with batched top-k or scatter operations:
```python
# Possible approach: argsort + cumsum + scatter
sorted_dists, order = dist_from_src.sort(dim=1)
cumsum = base_accept.cumsum(dim=1)
accept = (cumsum <= budget_per_source.unsqueeze(1)) & base_accept
```

This would eliminate the O(M) Python loop.

---

## Files

```
spectral_shader_v11.py      # 484 LOC - current unified implementation
spectral_shader_v10_unified.py  # 1076 LOC - can be deleted
spectral_shader_v8.py       # kept for cross-attention reference
```

---

## Key Parameters

```python
# Gating (threshold = -bias / (gamma - 1))
gamma_h=1.3, bias_h=-0.18  # high gate threshold ~0.6
gamma_l=1.3, bias_l=-0.12  # low gate threshold ~0.4

# Thickening
thick_radius=4, r_in=2.0, r_out=2.0
gamma_penalty=0.85  # short-range spectral penalty

# Shadow
shadow_dist=20.0, shadow_offset=8.0
w_seed=4.0, w_src=2.0, w_loc=1.0, w_int=1.5, w_void=0.8
```

---

## Architecture Insight

The V-projection analogy:
- **Q** = candidate thickening positions
- **K** = source contour positions
- **V** = budget allocation (inverse of spectral affinity)

Instead of `softmax(QK^T) @ V`, we do:
1. Compute affinity from spectral energy
2. Apply gamma penalty to high-frequency components
3. Stratify budget by affinity rank
4. Accept candidates up to per-source budget

This prevents attention-style "winner take all" concentration.

---

## Test Commands

```bash
# Run v11
uv run python spectral_shader_v11.py

# Quick single-pass test
uv run python -c "
from spectral_shader_v11 import shader_v11, DEVICE
from PIL import Image
import numpy as np
import torch

img = np.array(Image.open('demo_output/inputs/toof.png').convert('RGB')) / 255.0
out, stats = shader_v11(torch.tensor(img, device=DEVICE, dtype=torch.float32))
print(stats)
"
```

---

*Previous: handoff-v11-collapse.md, handoff-v10.2-asymmetric.md*
