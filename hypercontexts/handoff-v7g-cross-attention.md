# Handoff: Spectral Shader v7g - Cross-Attention Retrieval

*Session Date: 2026-02-01/02*

---

## Current State

v7g implements cross-attention retrieval with spatial gating. It's the first version that shows proper structure transfer in at least one direction.

### What Works

1. **tone→toof**: Copies curved toof linework onto tonegraph. Purple curved lines visible, matching toof's graph structure.

2. **Spatial gating**: Effects are localized near activation sources via locality gate `exp(-dist/radius)`. No more scattered patches everywhere.

3. **Lazy cross-attention**: Computes attention rows on-demand per seed, avoiding 262k×262k memory explosion.

4. **Boundary-weighted selection**: Sample boundary pixels weighted by cross-attention, then flood-filled along sample graph.

5. **Soft blending**: `output = (1-α)*target + α*warped_sample` with α from combined gate.

### What Doesn't Work

1. **toof→tone**: Produces blobby green patches, not sharp arrow/rectangle shapes. Tonegraph's disconnected components merge during flood fill.

2. **Spectral cross-attention is weak**: Raw attention values are tiny (max ~0.005) and uniform. Normalization makes everything look high-attention. Had to fall back to boundary-weighting.

3. **No true rotation**: The 90° rotation is applied, but shapes are still blobby enough that rotation isn't visually apparent.

4. **Thickening**: Heat kernel dilation exists but produces minimal visible effect.

---

## Key Files

```
resnet_spectral_shader_v7g.py    # Current implementation (~350 LOC)
  - laplacian(), lanczos()        # Spectral primitives
  - shader()                       # Main shader with cross-attention retrieval
  - mask_centroid(), mask_orientation()  # Segment properties
  - warp_mask_and_color()         # Rigid body transform via grid_sample

hypercontexts/
  spec-shader-operations.md       # What thicken/shadow should do
  spec-resnet-structure.md        # Code structure principles
  spec-segment-operations.md      # Flood fill, fuzz annihilator, rigid body
  spec-cross-attention-retrieval.md  # Current approach
```

---

## Architecture

```
INPUT: target (H_t, W_t, 3), sample (H_s, W_s, 3)

1. SPECTRAL EMBEDDINGS
   L_t = laplacian(target_gray)
   L_s = laplacian(sample_gray)
   phi_t = lanczos(L_t, k=12)  # (n_t, k)
   phi_s = lanczos(L_s, k=12)  # (n_s, k)

2. ACTIVATION (low-rank cross-attention sum)
   act = phi_t @ (phi_s.T @ sample_signal)  # (n_t,)
   act = act * boundary_mask  # only on edges

3. SEED SELECTION
   seeds = topk(act, n_seeds)

4. FOR EACH SEED:
   a. Compute cross-attention row: seed_attn = phi_t[seed] @ phi_s.T
   b. Weight by sample boundary: weighted = seed_attn * bnd_s
   c. Flood fill from high points along sample graph
   d. Get sample mask (sample-shaped region)
   e. Compute centroid, orientation
   f. Warp via grid_sample (rotate 90°, translate toward void)
   g. Apply with spatial gating (locality falloff from seed)

5. THICKEN (optional)
   Heat kernel dilation of high-activation boundary

OUTPUT: target with sample-shaped content at high-activation locations
```

---

## The Core Problem (Still Unsolved)

The spectral cross-attention should give us "soft intersection" - regions that are graph-similar in BOTH target and sample. But:

1. **Weak signal**: phi_t @ phi_s.T produces near-uniform scores when images are structurally different.

2. **No learned projections**: Real cross-attention uses Q=W_q·x, K=W_k·x. We're using raw spectral coefficients. The user suggested this is why it doesn't work well.

3. **Sample selection ignores target structure**: We select sample regions based on sample boundary + weak attention. The selection isn't constrained to match target's local geometry.

---

## What the User Said Needs to Happen

From the conversation:

> "the analogue to an attention resnet model needs to be more literal. the partial graph spectral transform IS an attention operation; throwing a learned linear projection in front of each side would satisfy the formalism that makes Q(operand A) * transpose(K(operand_B)) a cross attention lookup"

> "you might need to compute a self-attention wrt A, to be able to use this 'self attention' output as well as the 'cross attn' activation output as operands for a bank of linear functions (those are called a FFN in deep learning town)"

Translation:
- Self-attention on target: `phi_t @ phi_t.T`
- Self-attention on sample: `phi_s @ phi_s.T`
- Cross-attention: `phi_t @ phi_s.T`
- Combine via linear projections (FFN-like)

The projections transform the spectral space to make different images comparable.

---

## Suggested Next Steps

1. **Component segmentation**: Before flood fill, segment sample boundary into distinct connected components. Select ONE component per seed, not merged blobs.

2. **Stronger attention signal**: Try different spectral weighting, or use boundary Laplacian (edges only) instead of full image Laplacian.

3. **Linear projections**: Add W_q, W_k matrices (can be hand-designed, not learned) to transform spectral embeddings before attention.

4. **Self-attention integration**: Use `phi_t @ phi_t.T` to understand target's local structure, gate cross-attention by local coherence.

5. **Activation as spatial mask**: Instead of just locality falloff, use the actual activation map as the gate. Copies appear where `act > threshold`, shaped by activation contours.

---

## Test Commands

```bash
# Run v7g
uv run python resnet_spectral_shader_v7g.py

# Outputs in demo_output/
# v7g_toof_tone_c.png      - single pass
# v7g_toof_tone_cccc.png   - four passes
# v7g_tone_toof_c.png      - reverse direction
```

---

## Visual Results Summary

| Direction | Pattern | Result |
|-----------|---------|--------|
| toof→tone | c | Subtle green patches near contour |
| toof→tone | cccc | More green/cyan patches, localized |
| tone→toof | c | Purple curved lines (toof structure!) |

The tone→toof direction works because toof has simple connected curves. The toof→tone direction struggles because tonegraph has complex disconnected components.

---

## Code Quality

- ~350 LOC, pure torch (numpy only in demo I/O)
- No pixel-level for loops in shader core
- Lazy attention (compute rows on demand)
- grid_sample for rigid body transforms

Still has one `for seed in seeds` loop that could be batched, but it's over seeds (8-16), not pixels.

---

*Previous: v7f (flood fill segments, diamond nuggets)*
*Related: spec-cross-attention-retrieval.md*
