# Cross-Attention Retrieval: Shape-Preserving Selection

## The Problem with Flood Fill

v7f uses uniform dilation from seed points. This gives square nuggets regardless of source/target structure. Toof→tone should copy arrow shapes, tone→toof should copy linework shapes. Uniform fill ignores this.

## The Insight: Spectral Transform IS Attention

The partial spectral transform is literally an attention operation:
- `phi_t`: (n_t, k) spectral embedding of target
- `phi_s`: (n_s, k) spectral embedding of sample
- `phi_t @ phi_s.T`: (n_t, n_s) cross-attention matrix

Entry (i, j) measures spectral correspondence between target pixel i and sample pixel j.

## Cross-Attention Retrieval

For a target seed point, the cross-attention row directly retrieves sample structure:

```
seed_idx = high_activation_target_pixel
seed_attn = cross_attn[seed_idx, :]  # (n_s,) scores over sample

# High-scoring sample pixels form contiguous region
# Shape of region = sample's local graph structure
sample_mask = seed_attn > (seed_attn.max() * threshold)
```

Why contiguous? Spectral similarity implies graph proximity (Fiedler property). Pixels spectrally similar to each other are graph-adjacent. So high-attention sample pixels cluster into connected components shaped by sample's edges.

## Self-Attention for Local Structure

Also compute self-attention on each graph:
```
self_attn_t = phi_t @ phi_t.T  # (n_t, n_t) target structure
self_attn_s = phi_s @ phi_s.T  # (n_s, n_s) sample structure
```

Uses:
- Determine local tangent direction at target seed (for rotation alignment)
- Find void regions (low self-attention = isolated from structure)
- Weight cross-attention by local context

## FFN Analogy: Combining Attention Outputs

In transformers, attention outputs feed into FFN (linear projections). Here:

```
# Inputs
x_self = self_attn_t[seed_idx, :]      # (n_t,) local target context
x_cross = cross_attn[seed_idx, :]      # (n_s,) sample correspondence

# "FFN" = linear combinations
activation = gamma_self * x_self + gamma_cross * x_cross + bias
```

Or more literally: stack self/cross outputs, apply learned projection:
```
combined = stack([x_self, x_cross])  # conceptually
output = W @ combined + b            # linear projection
```

We're not learning W, but we can set it to meaningful values:
- W_self: weight for local structure
- W_cross: weight for correspondence
- bias: threshold for selection

## The Full Pipeline

```
1. Compute spectral bases
   phi_t = lanczos(L_t, k)  # target embedding
   phi_s = lanczos(L_s, k)  # sample embedding

2. Compute attention matrices
   self_t = phi_t @ phi_t.T      # (n_t, n_t)
   self_s = phi_s @ phi_s.T      # (n_s, n_s)
   cross = phi_t @ phi_s.T       # (n_t, n_s)

3. Find target seeds (high activation on boundary)
   act = cross.sum(dim=1)        # aggregate cross-attention
   seeds = topk(act * boundary)

4. For each seed, retrieve sample-shaped region
   sample_scores = cross[seed, :]           # (n_s,) attention over sample
   sample_mask = sample_scores > threshold  # sample-shaped!

5. Get sample colors/structure within mask
   sample_content = sample_rgb[sample_mask]

6. Compute transform
   # Target local tangent from self-attention gradient
   local_structure = self_t[seed, :].reshape(H, W)
   tangent = gradient(local_structure)

   # Sample mask orientation
   sample_orient = principal_axis(sample_mask)

   # Rotation to align sample to target tangent
   angle = target_tangent_angle - sample_orient + pi/2  # orthogonal

7. Warp and place
   warped = affine_transform(sample_mask, sample_content, angle, translation)
   output[void_region] = warped[void_region]
```

## Why This Gives Shape-Preserving Selection

- **No flood fill**: Shapes come directly from cross-attention scores
- **Sample structure preserved**: High-attention region follows sample's graph edges
- **Target structure respected**: Self-attention guides placement
- **Soft intersection**: Regions high in BOTH attentions = present in both graphs

## DNN Analogy

A DNN is a compute shader with extra channels:
- RGB (3 channels) → feature space (hundreds of channels)
- Selections/filters = multiplying feature channels by weights
- Each layer transforms feature space via attention + FFN

We're doing the same with graph spectral features:
- Image pixels → spectral coefficients (k channels)
- Cross-attention = soft correspondence lookup
- Self-attention = local structure encoding
- "FFN" = combining these via fixed linear projections

The spectral embedding IS a learned representation (learned by the Laplacian eigendecomposition). The attention IS differentiable soft selection. We just skip backprop and use semantically meaningful fixed weights.
