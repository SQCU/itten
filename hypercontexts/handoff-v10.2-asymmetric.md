# Handoff: Spectral Shader v10.2 - Asymmetric Width Modulation

*Session Date: 2026-02-01*

---

## Summary

This session added asymmetric width modulation - the "jagged exciting flexibility" from v6 - using proper tensor ops. The key insight: **treat dilation candidates as abstract activations and refine them before scattering**.

---

## Proof Sketch: The Data Flow

```
1. GATHER contour pixels
   C = mask.nonzero() → (N, 2) coords

2. GENERATE candidate cloud
   offsets = circular_kernel(max_radius) → (K, 2)
   candidates = C.unsqueeze(1) + offsets.unsqueeze(0) → (N, K, 2)
   candidates_flat → (N*K, 2)
   source_idx = which contour spawned each → (N*K,)
   distances = ||offset|| → (N*K,)

3. COMPUTE spectral context per candidate
   Option A - Fiedler mode:
     fiedler_diff = fiedler[candidate] - fiedler[source]
     inside = fiedler_diff < 0  (toward low-Fiedler)

   Option B - Edge-normal mode:
     normal = ∇carrier / |∇carrier|  (points toward light/paper)
     dot = offset · normal[source]
     inside = dot < 0  (against normal = toward dark/ink)

4. COMPUTE effective radius
   grad_norm = fiedler_grad[candidate] / max(fiedler_grad)
   r_eff[inside] = r_inside + grad_mod_inside * grad_norm
   r_eff[outside] = r_outside + grad_mod_outside * grad_norm

5. FILTER (branchless)
   accept = (distance < r_eff) & in_bounds
   accepted_coords = candidates_flat[accept]
   accepted_colors = contour_colors[source_idx[accept]]

6. SCATTER
   output[accepted_coords] = accepted_colors
```

**All tensor ops. No per-pixel loops. No branches.**

---

## What Works

### 1. Asymmetric Inside/Outside Radii
```python
cfg = AsymmetricConfig(
    r_inside=2.0,   # thicken 2px toward inside
    r_outside=5.0,  # thicken 5px toward outside
)
```
Creates strokes that are heavier on one side, like ukiyo-e woodblock prints.

### 2. Two Asymmetry Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `edge_normal` | inside/outside based on local line geometry | Consistent stroke weight direction |
| `fiedler` | inside/outside based on global spectral partition | Emphasize spectral structure |

### 3. Gradient-Driven Modulation
```python
cfg = AsymmetricConfig(
    grad_mod_inside=2.0,   # spectral edges boost inside radius
    grad_mod_outside=4.0,  # spectral edges boost outside radius more
)
```
High-gradient (spectrally active) regions get more thickening.

### 4. Measured Asymmetry Ratios

| Config | Edge-normal ratio | Fiedler ratio |
|--------|-------------------|---------------|
| symmetric (3.0/3.0) | 1.53 | 1.09 |
| outside_heavy (2.0/5.0) | 9.66 | 6.74 |
| extreme (1.0/6.0) | 44.79 | 27.79 |

---

## Key Fixes from v10.1

1. **Low gate was dead**: bias sign was wrong (needed negative for gamma > 1)
2. **Width modulation was flat**: Fiedler gradient too small (~0.0003), switched to carrier gradient (~0.01-1.4)
3. **Uniform dilation**: Now asymmetric with inside/outside control

---

## Files

```
spectral_shader_v10_sweep.py      # Parameter sweep infrastructure
spectral_shader_v10_asymmetric.py # Asymmetric thickening implementation
  - AsymmetricConfig              # All parameters in one place
  - thicken_asymmetric()          # The core algorithm
  - compute_edge_normal_field()   # For edge-normal mode
```

---

## Test Commands

```bash
# Run asymmetric demo
uv run python spectral_shader_v10_asymmetric.py

# Run parameter sweep
uv run python spectral_shader_v10_sweep.py sweep

# Run field diagnostics
uv run python spectral_shader_v10_sweep.py diag
```

---

## Next Steps

1. **Integrate with segment rotation**: v10's low-gate shadow/rotation branch should use same asymmetric thickening

2. **Multi-pass with asymmetry decay**: Effect strength decays, but asymmetry ratio could also evolve

3. **Cross-image attention**: Port v8's spectral embedding retrieval into this framework

4. **Test on thin linework**: Single-pixel-width lines are the generality test

---

## Architecture Insight: Why This Works

The v6 "jagged flexibility" came from the Fiedler gate boundary intersecting with uniform dilation. Our approach is more general:

1. **Candidates are abstract**: We generate them but don't commit until filtered
2. **Filtering is spectral-aware**: Inside/outside determined by local/global spectral structure
3. **Radii are independent**: r_inside and r_outside can differ arbitrarily
4. **Gradient boosts edges**: Spectrally active regions thicken more

This gives fine-grained control while remaining fully vectorized.

---

---

## v10.3: Energy-Minimizing Shadow Placement

### The Energy Function

```
E_total = w₁·E_tangent_source   // parallel to source = bad
        + w₂·E_tangent_crossing // parallel crossing = bad
        + w₃·E_intersection     // overlap = bad (less bad if orthogonal)
        + w₄·E_void             // empty space = good (negative)
        + w₅·E_downright        // down-right bias (tie-breaker)
```

### Tensor Formulation

```
1. GENERATE candidate grid
   θ_grid = linspace(0, 2π, num_angles)     → (A,)
   t_grid = linspace(0, max_dist, num_dists) → (D,)

2. COMPUTE all rotations in parallel
   R_all = make_rotation_matrices(θ_grid)    → (A, 2, 2)
   rotated_coords = segment @ R_all.T         → (A, N, 2)
   rotated_axis = principal_axis @ R_all.T    → (A, 2)

3. COMPUTE all translations in parallel
   void_dir = void_direction[centroid]        → (2,)
   translations = t_grid ⊗ void_dir           → (D, 2)
   final_coords = rotated + centroid + trans  → (A, D, N, 2)

4. GATHER context at all candidate positions
   tangent_at_pos = tangent_field[final]      → (A, D, N, 2)
   distance_at_pos = distance_field[final]    → (A, D, N)
   geometry_at_pos = geometry_mask[final]     → (A, D, N)

5. COMPUTE energies (all batched)
   E_tangent_source = |rotated_axis · original_axis|  → (A,) broadcast to (A, D)
   E_tangent_crossing = (geometry * |axis · local_tangent|).sum()
   E_intersection = geometry.sum() * (1 - orthogonal_discount)
   E_void = -distance_at_pos.mean()
   E_downright = -(y + x) / (H + W)

6. SELECT optimal placement
   best_θ, best_t = argmin(E_total)           → scalar indices
```

### Key Behaviors

| Energy Weight | Effect |
|---------------|--------|
| w_tangent ↑ | Shadows rotate further from source orientation |
| w_intersection ↑ | Shadows avoid overlapping existing geometry |
| w_void ↑ | Shadows pushed into empty space |
| w_orthogonal_cross ↑ | Less penalty for crossing geometry at right angles |
| w_downright_bias ↑ | Shadows prefer down-and-right placement |

### Measured Results

| Config | Mean Angle | Mean Distance | Energy Range |
|--------|------------|---------------|--------------|
| balanced | 2.81 rad (161°) | 38.4 | -0.58 to -0.17 |
| tangent_heavy | 4.54 rad (260°) | 36.6 | -0.38 to +0.39 |
| void_seeking | 1.77 rad (101°) | 38.4 | -2.08 to -1.86 |

Shadows now prefer ~90° rotations (orthogonal to source) and push into void space.

---

## v10.5: Cross-Attention Extension

Added (target, sample) cross-attention to retrieve GEOMETRY from sample image and place at spectrally-matching locations in target.

### Algorithm

```
1. SPECTRAL EMBEDDING
   phi_t = lanczos(L_target, k=8)  → (H_t*W_t, k)
   phi_s = lanczos(L_sample, k=8)  → (H_s*W_s, k)

2. PRE-SEGMENT SAMPLE
   components = find_connected_components(contours_sample)
   signatures = [mean(phi_s[component]) for component in components]

3. FOR EACH LOW-GATE SEED IN TARGET
   seed_phi = phi_t[seed_index]  → (k,)

   # Cross-attention retrieval
   scores = [dot(seed_phi, sig) for sig in signatures]
   best_component = argmax(|scores|)

   # Extract geometry as patch
   patch = extract_patch(sample, best_component)

   # Transform: rotate + translate toward void
   rotated = rotate_patch(patch, θ=π/2)
   placed = translate_to_seed(rotated, seed, void_direction)

   # Color modulation: bend sample colors toward target palette
   colors = cross_color_modulate(rotated_colors, target_context)

   # Scatter shadow + front layers
   scatter(output, placed, colors, preserve=target_contours)
```

### Key Properties

1. **Geometry retrieval**: Copies connected components, not individual pixels
2. **Spectral matching**: k-dimensional dot product finds graph-similar structures
3. **Color modulation**: Sample colors bend toward target's local palette
4. **Preservation**: Never overwrites target's graph structure

### Test Results

| Target | Sample | Sample Components | Seeds Used |
|--------|--------|-------------------|------------|
| snek-heavy | red-tonegraph | 11 | 21 |
| toof | red-tonegraph | 11 | 21 |
| snek-heavy | toof | 23 | 21 |

Arrows/rectangles from tonegraph extrude from snek-heavy at spectrally-matched locations.

---

*Previous: handoff-v10-tensor-segments.md*
