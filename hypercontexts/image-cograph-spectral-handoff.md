# Image↔Co-Graph Spectral Control - Handoff Document

*Session Date: 2026-01-31*
*Status: Productive implementation phase, patterns established*

---

## Context Bar
```
[██████████░░░░░░░░░░░░░░░░░░░░░░░░░] ~28% (56k/200k est)
Velocity: ▂▅▇█ (ramping productive)
```

---

## Problem Statement

Claude subagents were getting distracted from the core question:

> "How do we turn arbitrary 2D tensors into something graph-like-enough that we can apply graph spectral transforms, then use those transforms to *control/modulate* the original image?"

The failure mode was: agents would explore spectral ops, then just **sum the spectral output onto the image** like a DCT blur overlay. This produces nothing interesting.

---

## Solution: Non-Additive Control Patterns

Three scripts demonstrate the pattern, totaling ~250 LOC:

### 1. `image_cograph_spectral.py` (95 LOC)
**Core pipeline: Image → Sparse Co-Graph → Spectral Transform → Multiplicative Gate**

```
Image → Contours (gradient thresh) → Voronoi seeds → Sparse adjacency graph
     → Aggregate pixel values to graph nodes
     → iterative_spectral_transform(Laplacian, signal, theta)
     → Scatter back to pixels
     → MULTIPLY with original (not add)
```

Key functions:
- `extract_contour_points()` - gradient magnitude → seed points
- `voronoi_graph()` - seeds → sparse `Graph` with adjacency
- `pixel_to_cell_map()` / `aggregate_to_cells()` - dense↔sparse mapping
- `scatter_to_pixels()` - sparse→dense

### 2. `sdf_gated_spectral.py` (85 LOC)
**ADT Composition: SDF gates the signal BEFORE spectral transform**

```
SDF (distance from contours) → weights per Voronoi cell
Weighted signal = raw_signal * (0.3 + 0.7 * sdf_weights)
Spectral transform operates on the pre-weighted signal
```

This demonstrates: **one ADT (SDF) controlling another ADT (spectral transform)** - not summing them.

### 3. `phase_controlled_spectral.py` (120 LOC)
**Eigenvector Phase Field as Selector (from Compendium Transform 4)**

```
Phase field = arctan2(eigenvector_1, eigenvector_0)  # from IMAGE Laplacian
Signal_A = raw co-graph aggregation (no spectral)
Signal_B = SDF-weighted + spectral transformed

Output = cos(phase)*A + sin(phase)*B  # SELECTION, not sum
       → then multiply with original image
```

The phase vortices around eigenvector nodal points create natural switching boundaries between signals.

---

## Anti-Patterns (DO NOT DO)

```python
# BAD: Linear sum onto output
output = image + alpha * spectral_effect  # looks like blurry DCT

# BAD: Treating spectral output as "the result"
output = spectral_transform(image)  # loses the original

# BAD: Additive blending
output = 0.5 * image + 0.5 * spectral_thing  # still linear combination
```

## Correct Patterns (DO THIS)

```python
# GOOD: Multiplicative gating
output = image * (0.1 + 0.9 * spectral_mask)

# GOOD: Signal selection via phase
blend = (cos(phase) + 1) / 2  # phase controls choice
output = blend * signal_A + (1 - blend) * signal_B

# GOOD: Pre-transform weighting (ADT gates ADT)
weighted_signal = raw_signal * sdf_weights
transformed = spectral_transform(weighted_signal)  # SDF shaped input

# GOOD: Manufacture intermediates for filters that need gradients
sdf = distance_transform(binary_image)  # bitmap → gradients
radial = sweep_gradients_at_contours(image)  # sparse → dense variation
abstract = sdf * radial  # combine for richer structure
filtered = heat_diffusion(abstract)  # NOW the filter has something to work with
output = image * filtered  # use as control signal
```

---

## Architecture Diagrams

### Single-Image Flow (with phase control)
```
Image ──┬── Contours ──┬── Voronoi seeds ── Co-graph ── L_cograph
        │              │                         │
        │              ├── SDF ──────────────────┼── pre-weights signal
        │              │                         │
        │              └── Phase field ──────────┼── selects A vs B
        │                  (from L_image)        │
        │                                        │
        └── pixel intensity ─────────────────────┴── signal_A (raw)
                                                      signal_B (weighted+spectral)
                                                          │
                                          phase_select(A, B) → mult gate → output
```

### Two-Image Cross-Mutation Flow
```
Image_A ────────────────────────────────────────── Image_B
    │                                                  │
    ▼                                                  ▼
Graph_A (voronoi/path)                        Graph_B (voronoi/path)
    │                                                  │
    │         ┌────── heat_diffusion ──────┐          │
    │         │                            │          │
    ▼         ▼                            ▼          ▼
    └── L_A' = scale_edges(L_A, heat(L_B)) ◄──────────┘
              │                            │
    ┌─────────►  L_B' = scale_edges(L_B, heat(L_A)) ──┘
    │                            │
    ▼                            ▼
spectral(L_A', sig_A)    spectral(L_B', sig_B)
    │                            │
    ▼                            ▼
gate(Image_A)              gate(Image_B)
    │                            │
    ▼                            ▼
Output_A (carries B's      Output_B (carries A's
 structural influence)      structural influence)
```

### Alternative Graph Extraction (SDF-Path)
```
Image → SDF → segment into level bands → iso-contours
                                              │
                                    sample points along paths
                                              │
                                    build graph: path edges + proximity
                                              │
                                    "abstract path image" (renderable SVG-like)
                                              │
                                    use for spectral OR cross-mutation
```

**Key insight**: Multiple ways to lift a graph from an image. Each gives different spectral bases. They can all cross-modulate each other.

---

## Key APIs (from spectral_ops_fast.py)

```python
# Graph construction
Graph.from_image(image, connectivity=4, edge_threshold=0.1)
build_weighted_image_laplacian(carrier, edge_threshold)

# Eigenvectors
lanczos_k_eigenvectors(L, num_eigenvectors)  # returns (evecs, evals)
lanczos_fiedler_gpu(L)  # fast second eigenvector

# Spectral filtering
iterative_spectral_transform(L, signal, theta, num_steps=8)
chebyshev_filter(L, signal, center, width, order)
heat_diffusion_sparse(L, signal, alpha, iterations)
```

---

## Test Images

Located in `demo_output/inputs/`:
- `toof.png` - simple tooth outline, good for seeing Voronoi structure
- `snek-heavy.png` - witch character with lineart, shows contour following
- `1bit redraw.png` - face with dithering, shows phase vortices on features

Output panels are labeled: Original | Intermediate | θ=0.25 | θ=0.5 | θ=0.75

---

## File Manifest

```
◆ image_cograph_spectral.py    - base pipeline (95 LOC)
◆ cograph_demo_viz.py          - visualization with co-graph overlay (55 LOC)
◆ sdf_gated_spectral.py        - SDF→spectral ADT composition (85 LOC)
◆ phase_controlled_spectral.py - phase-based signal selection (120 LOC)
◆ cross_graph_spectral.py      - two-image cross-mutation (180 LOC)
◆ sdf_path_graph.py            - alternative graph extraction via paths (200 LOC)
◇ spectral_ops_fast.py         - core spectral ops library (~3000 LOC)
◇ hypercontexts/spectral-transforms-compendium.md - 25 transforms reference
```

**Total new code: ~735 LOC across 6 scripts**

### 7. `smooth_compute_vcell_pixel.py` (~200 LOC)
**Separated compute/pixel shader architecture**
- Compute shaders operate on pixel-level Laplacian (smooth output)
- Pixel shaders use V-cell as rendering effect (posterize, edges)
- V-cells never leak into compute path

### 8. `abstract_intermediate_pipeline.py` (~250 LOC)
**Manufacture intermediates that satisfy filter constraints**

Key insight: Don't say "filters don't work on bitmaps" - MAKE data they can work with:
```
Bitmap → SDF (instant gradients!)
      → Radial sweep (center-based falloff)
      → Contour sweep (iso-band structure)
      → Products (combined features)
            ↓
      Heat diffusion, bandpass, phase (NOW WORK)
            ↓
      Control signals for gating/selection
```

**Updated total: ~1185 LOC across 8 scripts**

---

---

## Recent Additions

### 4. `cross_graph_spectral.py` (~180 LOC)
**Two-Image Cross-Mutation: Graph A mutated by Graph B and vice versa**

```
image_A, image_B → graph_A, graph_B
heat_diffusion(graph_B) → scales edge weights of graph_A → graph_A'
heat_diffusion(graph_A) → scales edge weights of graph_B → graph_B'
spectral(graph_A', signal_A) → modulate image_A
spectral(graph_B', signal_B) → modulate image_B
```

Key function: `cross_modulate_edge_weights()` - one graph's heat flow scales another's edges.

### 5. `sdf_path_graph.py` (~200 LOC)
**Alternative Graph Extraction: SDF → Segment → Path → Abstract Graph**

Instead of Voronoi from contour seeds:
1. Compute SDF from image contours
2. Extract iso-contours at multiple levels
3. Sample points along these contour paths
4. Build graph: nodes = path samples, edges = path connectivity + spatial proximity

This produces a different spectral basis - concentric/following distance bands rather than radial cells.

Also renders an **abstract path image** (SVG-like) as an intermediate representation.

---

## Graph Extraction Methods Comparison

| Method | Nodes | Edges | Character |
|--------|-------|-------|-----------|
| Voronoi | Cell centers from contour seeds | Cell adjacency | Radial, cellular |
| SDF-Path | Samples along iso-contours | Path + proximity | Concentric, skeletal |

Both can be cross-modulated. The choice affects which "spectral directions" are available for control.

---

## Next Directions (Unstarted)

From the compendium, other transforms that fit the "control not sum" pattern:

1. **Transform 5: Expansion-Gated** - local λ₂ estimate as texture density control
2. **Transform 15: Spectral Warp Field** - eigenvector gradients warp coordinates
3. **Transform 18: Commute Time Distance** - random walk distance as threshold
4. **Diffusion Coupling** (from Conspiracy thread) - `L_B = f(exp(-t*L_A))`, one Laplacian's heat flow defines another's edge weights

The general principle: **spectral quantities control OTHER operations** (warping, thresholding, weighting, selection) rather than being summed into the output.

---

## For Continuing Agents

1. Read this document first
2. Run the existing scripts to see outputs: `uv run python <script>.py`
3. When adding new spectral operations, ask: "Is this being summed or is it controlling something?"
4. The compendium at `hypercontexts/spectral-transforms-compendium.md` has 25 transforms with code
5. Avoid the self-gas trap: "wow these spectral patterns look cool" → sum onto image → blurry nothing

---

*Handoff Version: 1.0*
*Parent Session: image-cograph-spectral exploration*
