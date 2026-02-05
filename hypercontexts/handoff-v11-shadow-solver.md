# Handoff: Differentiable Shadow Placement Solver

*Session Date: 2026-02-02*

---

## Problem

Cross-attention shadow placement was placing fragments **on** contours rather than **away** from them. The old grid search found positions where fragments fit geometrically (perpendicular to local tangent) rather than projecting outward like drop shadows.

---

## Solution: Batched Differentiable Solver

Replaced `energy_minimize_placement()` (per-fragment grid search) with `solve_placements_vectorized()` (batched gradient descent).

### Key Insights

1. **Anisotropic learning rates**: Rotation is "cheap" (lr=0.5), translation is "expensive" (lr=0.2). Solver naturally prefers rotating to perpendicular first, then translating to void.

2. **Global shadow direction**: Blend of local perpendicular (contour-aware) and global direction (down-right, simulating light from top-left). Prevents shadows from following curves into other geometry.

```python
# perpendicular_soft() blends:
local_dir = perpendicular_to_tangent(weighted_by_void_gradient)
global_dir = [0.6, 0.8]  # down-right
shadow_dir = (1 - global_bias) * local_dir + global_bias * global_dir
```

3. **Minimum distance penalty**: Exponential decay pushes fragments away from seed.

```python
E_min_dist = exp(-dist / target_dist)  # high when dist is small
```

---

## Energy Function

**Critical insight**: PARALLELISM IS WORSE THAN CLIPPING
- Parallel non-clipping is a fixed point → sedimentary roughage accumulation
- Clipping at least allows fragments to find anti-parallel escape directions

```python
E = (w_parallel * E_parallel +    # perpendicular to seed tangent (3.0)
     w_src_rot * E_src_rot +      # encourage rotation (0.3)
     w_local * E_local +          # DOMINANT: avoid parallel to ANY geometry (12.0)
     w_clip * E_clip +            # moderate: some clipping OK (4.0)
     w_void * E_void +            # seek empty space (6.0)
     w_min_dist * E_min_dist)     # push into void (4.0)
```

**Key tuning**: `w_local` DOMINATES (12.0) - fragments must avoid parallel configurations even if it means some intersection. `w_ortho_discount=0.8` makes orthogonal crossing almost free.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  BATCH PREPARATION (still has Python loop for variable-size)   │
│  - Collect fragments, pad to max_N                              │
│  - Compute principal axes via covariance                        │
│  - Gather seed tangents and outward directions                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  solve_placements_vectorized() - ALL TENSOR OPS                 │
│                                                                 │
│  for iteration in range(25):                                    │
│    # Rotation (element-wise 2D, not bmm - CUDA stable)          │
│    rot_axes = [c*ax - s*ay, s*ax + c*ay]                       │
│    rotated = [c*fx - s*fy, s*fx + c*fy]                        │
│                                                                 │
│    # Placement                                                  │
│    placed = seeds + rotated + dist * shadow_dirs                │
│                                                                 │
│    # Bilinear gather (differentiable)                          │
│    local_tan = gather_bilinear(tangent_field, placed)          │
│    on_geom = gather_bilinear(geom_mask, placed)                │
│    void_dist = gather_bilinear(dist_field, placed)             │
│                                                                 │
│    # Energy computation (batched over B fragments)              │
│    E = weighted_sum(E_parallel, E_clip, E_void, E_min_dist...) │
│                                                                 │
│    # Gradient: analytical for rotation, finite diff for rest   │
│    grad_theta = analytical(E_parallel) + fd(E_local, E_clip)   │
│    grad_dist = fd(E_local, E_clip, E_void) + analytical(E_min) │
│                                                                 │
│    # Anisotropic update                                         │
│    theta -= lr_theta * grad_theta  # fast rotation             │
│    dist -= lr_dist * grad_dist     # slow translation          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  SCATTER (still looped per fragment for now)                    │
│  - Apply rotation and translation                               │
│  - Two layers: shadow + highlight offset                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Performance

- **Before**: Grid search over 16 angles × 8 distances per fragment (sequential)
- **After**: 25 gradient iterations, all B fragments in parallel
- **Benchmark**: ~85ms for B=10 fragments on CUDA

---

## Files

```
shadow_solver.py              # NEW - differentiable solver (490 LOC)
spectral_shader_v11.py        # Updated to use solve_placements_vectorized
```

---

## Visual Comparison

**Before** (grid search, void_dir):
- Arrows following curves
- Square on face plane
- Elements ON contours

**After** (differentiable solver, global shadow bias):
- Consistent down-right offset
- Elements in void space
- Proper drop shadow projection

---

## Fragment Length & Budding Surface

Short fragments create a growth bottleneck:
- Tiny stubs provide minimal "budding surface" for subsequent passes
- Next iteration has few seed points to work with
- Growth stalls or produces uniform sediment

Solution: `min_seg=15, max_seg=80` (was 8, 50)

Longer fragments → more surface area → more diverse seeds for next pass → branching growth that can "bend away" from both the previous bud AND the base graph.

---

## TODO

### 1. Vectorize Scatter
The final scatter loop could be batched using `scatter_nd` with proper masking.

### 2. Learned Solver
The solver's trajectories are deterministic given inputs. Could train a small MLP:
```python
(seed_tangent, outward_dir, frag_axis, local_geometry_features) → (theta, dist)
```
Skip 25 iterations, single forward pass. Only worth it if solver becomes bottleneck.

### 3. Configurable Light Direction
Currently hardcoded `[0.6, 0.8]` (down-right). Could expose as parameter for artistic control.

---

*Previous: handoff-v11-stratified.md*
