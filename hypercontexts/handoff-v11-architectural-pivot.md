# Handoff: Architectural Pivot - Activations Not Pixels

*Session Date: 2026-02-02*

---

## Session Summary

Started debugging drop shadow placement issues, ended with fundamental architectural realization.

### What Was Attempted

1. **Differentiable shadow solver** (`shadow_solver.py`) - batched gradient descent for fragment placement
2. **Energy function tuning** - parallelism penalties, void-seeking, minimum distance
3. **Global shadow bias** - tried forcing down-right offset (caused systematic drift)
4. **Gate parameter tuning** - discovered bias sign was backwards, causing seed explosion

### What Broke

- Removed `max_seeds` cap → 533 seeds per pass → GPU saturation
- Bias sign confusion: more negative bias_l → MORE selections, not fewer
- Fragment placement still "sedimentary" - parallel layers accumulating

---

## The Core Problem

**We're treating spectral shaders as PIXELS IN → PIXELS OUT**

Current flow:
```
pixels
  → spectral decomposition (graph!)
  → IMMEDIATELY DISCARD GRAPH
  → pixel coordinates
  → local pixel scattering
  → pixels
```

We do expensive spectral analysis to get a graph structure, then throw it away and go back to greedy pixel operations.

---

## The Reframe

**Should be: ACTIVATIONS IN → PIXELS OUT**

```
spectral embedding φ (N × k)     ←── ACTIVATIONS IN
graph Laplacian L
Fiedler vector (structure)
         │
         ▼
operations in embedding space    ←── TRANSFORM GRAPH, NOT PIXELS
  - thickening: edge dilation in φ-space
  - shadow: subgraph projection in φ-space
         │
         ▼
rasterize final structure        ←── PIXELS OUT (once, at the end)
```

---

## What This Means Concretely

### Thickening (Current - Wrong)
```python
# For each HIGH-GATE PIXEL:
#   generate offset candidates (PIXELS)
#   accept based on budget (PIXELS)
#   scatter colors (PIXELS)
```

### Thickening (Should Be)
```python
# Identify high-activity GRAPH EDGES (from φ)
# Dilate edges in embedding space (add parallel edges to graph)
# Rasterize dilated graph at the end
```

### Shadow (Current - Wrong)
```python
# For each LOW-GATE PIXEL (seed):
#   retrieve PIXEL FRAGMENT via cross-attention
#   rotate PIXEL COORDINATES
#   scatter PIXELS at offset
```

### Shadow (Should Be)
```python
# Identify low-activity GRAPH REGION
# Cross-attention retrieves SUBGRAPH (not pixel blob)
# Project subgraph in embedding space (translation in φ)
# Rasterize projected subgraph
```

---

## Key Questions for Next Session

1. **What does "dilate an edge in φ-space" mean mathematically?**
   - Add rows to φ with interpolated values?
   - Modify the Laplacian to add parallel connectivity?

2. **What does "project a subgraph" mean?**
   - Translate φ values for a subset of nodes?
   - Create new nodes with offset embedding coordinates?

3. **How do we rasterize from φ back to pixels?**
   - φ gives us an embedding per pixel - but new "virtual" nodes don't have pixel positions
   - Need inverse mapping: φ-space position → pixel position

4. **Does the gating even make sense in this framing?**
   - High/low gate selected PIXELS based on Fiedler value
   - Should instead select GRAPH STRUCTURES (edges, connected regions)

---

## Files Changed This Session

```
shadow_solver.py                    # NEW - differentiable placement solver (now questionable)
spectral_shader_v11.py              # Modified gates, parameters, removed max_seeds
hypercontexts/handoff-v11-shadow-solver.md  # Previous handoff (partially obsolete)
```

---

## The Uncomfortable Truth

The entire v11 architecture may need rethinking. We have:
- Spectral decomposition ✓ (good, gives us graph structure)
- Graph-aware gating ✓ (selects regions by spectral properties)
- **Pixel-based operations ✗** (throws away graph, scatters pixels)
- **Pixel-based budgeting ✗** (counts pixels, not graph features)

The solver, the energy function, the scatter loops - all operating on pixel coordinates when they should operate on graph structure.

---

## Possible Paths Forward

### Path A: Fix current architecture
- Keep pixel operations but add proper budgeting
- Shadow pixels ≤ f(thickening pixels)
- Faster but doesn't address fundamental issue

### Path B: Graph-native operations
- Thickening = Laplacian modification (add edges)
- Shadow = subgraph embedding translation
- Rasterize once at end
- Principled but significant rewrite

### Path C: Hybrid
- Use spectral embedding to GUIDE pixel operations
- But define budgets/constraints in graph terms
- "Add N graph edges worth of thickening" not "scatter to N pixels"

---

## Relevant Literature Pointers

- Spectral mesh processing (Taubin, 1995) - smoothing/enhancement via eigenfunction manipulation
- Graph signal processing - filtering defined on graph Laplacian eigenbasis
- Neural implicit representations - continuous functions, rasterize at query time

---

*Previous: handoff-v11-shadow-solver.md*
*The shadow solver work may be salvageable if reframed as operating on graph embeddings rather than pixel coordinates.*
