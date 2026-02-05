# Spectral Compute Shader Handoff

## Summary

Six demo scripts demonstrate spectral graph transforms as compute shaders for image processing. Each script represents a distinct approach to the core pattern: **use `iterative_spectral_transform(L, signal, theta)` as the central operation** where one image's graph structure (Laplacian) filters another image's content (signal).

---

## Scripts (Entry Points)

### 1. `sdf_gated_spectral.py`
**Pattern**: SDF-weighted gating before spectral transform

```
img → contours → Voronoi co-graph → L
img → SDF → weights on co-graph nodes
signal = aggregate(img) * sdf_weights
output = scatter(spectral_transform(L, signal, θ))
```

**Key insight**: SDF creates smooth gradients from binary images, enabling meaningful gating. The SDF gates the signal BEFORE spectral transform, not after.

**Imports**: `image_cograph_spectral`

---

### 2. `phase_controlled_spectral.py`
**Pattern**: Eigenvector phase as spatial control field

```
img → Voronoi co-graph → L
eigenvectors(L) → phase field (sign patterns)
phase_field controls which transform to apply where
```

**Key insight**: Eigenvector signs create natural spatial partitions. Phase field derived from spectral decomposition controls local operations.

**Imports**: `image_cograph_spectral`

---

### 3. `cograph_demo_viz.py`
**Pattern**: Voronoi co-graph visualization

```
img → contour seeds → Voronoi → graph visualization
Shows: seeds, cell boundaries, adjacency edges
```

**Key insight**: Visualizes the intermediate co-graph structure that mediates between dense pixel space and sparse graph operations.

**Imports**: `image_cograph_spectral`

---

### 4. `cross_pixel_transform.py` + `pixel_transform_shaders.py`
**Pattern**: Spectral control fields driving pixel shaders

```
img_structure → co-graph → spectral decomposition → control fields
control fields = {intensity, phase, frequency}
pixel_shaders(img_content, control_fields) → output
```

**Key insight**: Spectral decomposition extracts control fields (intensity, phase, frequency) that parameterize pixel-level operations (shimmer, thicken, displace). Two-stage: spectral extracts structure, pixel shaders apply it.

**Imports**: `abstract_intermediate_pipeline`, `pixel_transform_shaders`

---

### 5. `smooth_compute_vcell_pixel.py`
**Pattern**: Smooth compute/pixel shader separation

```
COMPUTE SHADER:
  img_kernel → Voronoi L_kernel
  signal = aggregate(img_input, cell_map)
  transformed = spectral_transform(L_kernel, signal, θ)

PIXEL SHADER:
  output = scatter(transformed, cell_map)
  optionally: modulate original image
```

**Key insight**: Clean separation between graph-level compute (spectral transform) and pixel-level rendering (scatter). The V-cell defines both the graph topology and the pixel-to-node mapping.

**Imports**: `spectral_ops_fast` only

---

### 6. `vcell_compute_shader.py`
**Pattern**: Spectral transform IS the compute shader output

```
img_kernel → Voronoi → L_kernel, cell_map
signal = aggregate(img_input, cell_map)
output = scatter(spectral_transform(L_kernel, signal, θ))
# NOT: output = img_input * mask
```

**Key insight**: The transformed signal IS the output, not a mask multiplied with the original. This is the correct "compute shader" pattern where the spectral transform produces the structural result directly.

**Imports**: `spectral_ops_fast` only

---

## Modules (Dependencies)

### `spectral_ops_fast.py`
Core spectral operations library. All scripts depend on this.

**Key exports**:
- `Graph` - graph structure with adjacency + coords
- `Graph.laplacian(normalized=True)` - compute graph Laplacian
- `iterative_spectral_transform(L, signal, theta, num_steps=8)` - THE core operation
- `build_weighted_image_laplacian(img, edge_threshold)` - dense pixel graph from image
- `lanczos_k_eigenvectors(L, num_eigenvectors)` - extract eigenvectors
- `DEVICE` - torch device (cuda/cpu)

**`iterative_spectral_transform` signature**:
```python
def iterative_spectral_transform(L, signal, theta, num_steps=8):
    """
    Apply Chebyshev polynomial filter at spectral position theta.
    theta=0: emphasize low frequencies (Fiedler, global structure)
    theta=1: emphasize high frequencies (local detail)
    """
```

---

### `image_cograph_spectral.py`
Voronoi co-graph construction from images.

**Key exports**:
- `build_voronoi_cograph(img, n_seeds)` → (Graph, cell_map, n_nodes)
- `aggregate_to_cograph(img, cell_map, n_nodes)` → signal tensor
- `scatter_from_cograph(signal, cell_map)` → image array

Used by: `sdf_gated_spectral`, `phase_controlled_spectral`, `cograph_demo_viz`

---

### `abstract_intermediate_pipeline.py`
Creates abstract intermediate data from bitmap images.

**Key exports**:
- `create_abstract_from_bitmap(img)` - generates smooth control fields from binary
- `filter_eigenvector_phase(img)` - extracts phase patterns

Used by: `cross_pixel_transform`, `pixel_transform_shaders`

---

### `pixel_transform_shaders.py`
Pixel-level transformation operations controlled by spectral fields.

**Key exports**:
- `pixel_coral_shimmer(img, control)` - shimmer effect
- `pixel_contour_thicken(img, control)` - edge thickening
- `pixel_displace_by_field(img, field)` - displacement mapping
- `abstract_color_to_transforms(img, selector)` - selector-driven transforms

Used by: `cross_pixel_transform`

---

## Architectural Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                      IMAGE A (structure)                        │
│                             │                                   │
│                             ▼                                   │
│                    ┌─────────────────┐                         │
│                    │ Build Co-graph  │                         │
│                    │ (Voronoi/dense) │                         │
│                    └────────┬────────┘                         │
│                             │                                   │
│                             ▼                                   │
│                    ┌─────────────────┐                         │
│                    │   Laplacian L   │ ◄── encodes A's edges   │
│                    └────────┬────────┘                         │
│                             │                                   │
│     IMAGE B (content)       │                                   │
│            │                │                                   │
│            ▼                ▼                                   │
│     ┌────────────┐  ┌──────────────────────┐                   │
│     │ Aggregate  │  │ iterative_spectral_  │                   │
│     │ to nodes   │──│ transform(L, sig, θ) │                   │
│     └────────────┘  └──────────┬───────────┘                   │
│                                │                                │
│                                ▼                                │
│                       ┌──────────────┐                         │
│                       │   Scatter    │                         │
│                       │  to pixels   │                         │
│                       └──────┬───────┘                         │
│                              │                                  │
│                              ▼                                  │
│                    OUTPUT: A's structure                        │
│                            B's content                          │
│                            θ controls frequency                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Anti-Pattern (Avoided)

**WRONG**: `output = original_image * spectral_mask`

This "sums" the spectral result onto the original, burying the spectral transform's effect.

**RIGHT**: `output = scatter(spectral_transform(L, signal, θ))`

The spectral transform IS the operation producing the result.

---

## Theta Parameter

- `θ = 0.0`: Low frequency / global / Fiedler-like (smooth, coarse structure)
- `θ = 1.0`: High frequency / local / fine detail
- Sweep `θ ∈ [0, 1]`: Shows distributional shift across spectral positions

---

## Asymmetry

`transform(L_A, signal_B, θ) ≠ transform(L_B, signal_A, θ)`

Because `L_A ≠ L_B`. A's Laplacian encodes A's edge structure. Filtering B through A's Laplacian imposes A's discontinuities on B's content.

---

## Files Removed

The following demo scripts were removed as weaker demonstrations:
- `cross_natural_pattern.py`
- `graph_graph_spectral.py`
- `cross_graph_spectral.py`
- `shared_cograph_fusion.py`
- `structural_transfer.py`
- `sdf_path_graph.py`
- `structural_spectral_output.py`
- `structure_preserving_spectral.py`
- `infinite_bisect.py`
- `graph_view.py`
- `texture_tui.py`
- `run_all.py`
- `test_chebyshev_*.py`
- `benchmark_spectral.py`
- `analyze_benchmark.py`
