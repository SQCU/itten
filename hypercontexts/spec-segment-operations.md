# Segment-Level Operations: Why Pixels Can't Branch

## SwiGLU-Like Fuzz Annihilator

Scattered pixel selections are garbage. Use Laplacian to measure and crush non-contiguous fuzz.

### Local Fuzz Detection

The Laplacian applied to a signal measures local discontinuity:
```
local_fuzz = (L @ x).abs()  # high where pixel differs from neighbors
```

Isolated pixels have high local_fuzz. Contiguous regions have low local_fuzz.

### SwiGLU-Style Gating

Multiply signal by sigmoid of negative fuzz — kills isolated pixels:
```
gate = sigmoid(-local_fuzz * temperature)  # 0 for fuzzy, 1 for contiguous
x_clean = x * gate  # ANNIHILATES scattered pixels
```

Temperature controls harshness. High temperature = aggressive bitcrushing.

### Global Contiguity Penalty

For even stronger crushing, penalize selections with high total cut:
```
cut_ratio = (x @ L @ x) / (x @ x + eps)  # how fuzzy is the whole selection
contiguity = exp(-cut_ratio * scale)
x_clean = x * contiguity  # scale selection by its coherence
```

Hairy scattered selections get multiplied toward zero. Clean contiguous selections pass through.

### Combined Annihilator
```python
def annihilate_fuzz(x, L, temp=5.0):
    local_fuzz = (L @ x).abs()
    gate = torch.sigmoid(-local_fuzz * temp)
    x_gated = x * gate
    # Optional: also penalize global incoherence
    cut = x_gated @ L @ x_gated
    mass = x_gated @ x_gated + 1e-8
    coherence = torch.exp(-cut / mass)
    return x_gated * coherence
```

## Thickener That Actually Thickens

### Why Pixel-by-Pixel Scatter Fails

Current approach iterates pixels that share a segment ID:
```python
for each pixel in segment:
    rotate pixel's coordinates
    scatter pixel to destination
```

This treats pixels independently. They move alone, creating scattered noise. No fill-in occurs because neighboring source pixels land at non-neighboring destinations.

### Correct: Segment as Rigid Body

A segment must transform as a SINGLE UNIT:
```python
# 1. Extract segment as boolean mask
seg_mask = (segment_id == seg)  # (H, W) boolean

# 2. Compute segment properties ONCE
centroid = mask_weighted_mean(coords, seg_mask)
orientation = principal_axis(seg_mask)

# 3. Build transformation matrix for ENTIRE segment
T = rotation(angle) @ translation(offset)

# 4. Apply SINGLE affine warp to segment
#    This moves the entire mask as a connected unit
warped_coords = T @ (coords - centroid) + new_centroid

# 5. Stamp onto output (union, not replacement)
output = output | warp(seg_mask, T)
```

When the segment moves as a rigid body, neighboring pixels STAY neighboring. The shape stays connected. Overlapping regions create fill-in.

### Thickening = Copy + Small Offset + Union

```python
# Original segment mask
original = seg_mask

# Copy with small translation along normal
normal = perpendicular_to(segment_orientation)
offset_coords = coords + normal * (thickness / 2)
offset_copy = sample(seg_mask, offset_coords)  # grid_sample or equivalent

# Union creates thickening
thickened = original | offset_copy
```

When offset < segment width, original and copy overlap. The union fills the gap, creating a wider stroke.

### Heat Kernel Alternative

Diffuse ink along graph edges:
```python
ink = seg_mask.float()
diffused = heat_kernel(L, ink, t=thickness_param)
thickened = (diffused > lower_threshold)
output = output | thickened
```

Heat kernel spreads to graph neighbors. Threshold controls width. Result is always connected.

### Key Insight

- **Scatter individual pixels** → noise cloud, no fill-in
- **Transform segment as rigid body** → connected shape, fill-in from overlap
- **Heat kernel dilation** → spread along graph, always connected

The fuzz in v7d output (cyan/magenta scattered dots) comes from pixel-by-pixel scatter. Proper segment transforms would produce coherent rotated shapes.

## The Problem

You can't rotate a 1x1 pixel. The v7c code scatters individual pixels, each translated by its local void direction. This produces parallel sedimentation:

- Pixel at (y, x) on a curve → moves to (y + vy, x + vx)
- Neighboring pixel at (y', x') on same curve → moves to (y' + vy', x' + vx')
- Since vy, vx are roughly constant along the curve (void direction is smooth), the scattered pixels form a shape parallel to the original curve

No rotation ever happens. No orthogonal structure. Future passes see parallel texture, produce more parallel texture. The "halo" in v7c_toof_tone_c.png is parallel to the toof's contour — evidence of this failure.

## To Get Branching Arboreal Afterimages

The shadow of a horizontal segment must be VERTICAL (rotated 90°). Then when you apply the rule again, the shadow of that vertical segment is HORIZONTAL (rotated 90° from vertical). This creates perpendicular branching.

Each generation rotates 90° from its parent. After k generations, you have structure at k different orientations, creating the tree/arbor effect.

## Proof Sketch: Segment-Level Shadow

### 1. Segment Identification via Spectral Clustering

The Fiedler vector φ₁ (second eigenvector of Laplacian) partitions the graph. To verify this produces contiguous segments:

```
# Fiedler vector from Laplacian
phi = lanczos(L, k=8)
fiedler = phi[:, 1]  # second eigenvector (first non-constant)

# Segment assignment by quantizing Fiedler value
segment_id = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min())  # normalize to [0,1]
segment_id = (segment_id * num_segments).floor().long()

# VERIFY contiguity: for each segment, compute internal connectivity
for seg in range(num_segments):
    mask = (segment_id == seg).float()
    internal_edges = mask @ L @ mask  # quadratic form
    boundary_cost = internal_edges / mask.sum()
    # Low boundary_cost confirms segment is internally connected
```

The Fiedler vector minimizes the ratio cut by construction (it's the solution to the relaxed normalized cut problem). Quantizing it into bands preserves this property — each band has low cut cost relative to its size.

Alternative: use multiple eigenvectors for finer segmentation:
```
# Project onto first k eigenvectors
coords = phi[:, 1:k]  # (N, k-1) spectral embedding

# K-means or quantization in spectral space
segment_id = kmeans(coords, num_segments)  # or: (coords * scale).floor()
```

Higher eigenvectors capture finer structure. Using k eigenvectors gives k-way partitioning.

### 2. Segment Properties via scatter_add Reduction

For each segment, compute centroid and orientation:

```
# Counts
count[seg] = scatter_add(ones, segment_ids)

# Centroids
sum_y[seg] = scatter_add(y_coords, segment_ids)
sum_x[seg] = scatter_add(x_coords, segment_ids)
centroid_y = sum_y / count
centroid_x = sum_x / count

# Second moments (for orientation)
dy = y_coords - centroid_y[segment_ids]
dx = x_coords - centroid_x[segment_ids]
mu20[seg] = scatter_add(dx * dx, segment_ids)
mu02[seg] = scatter_add(dy * dy, segment_ids)
mu11[seg] = scatter_add(dx * dy, segment_ids)

# Principal axis orientation
orientation[seg] = 0.5 * atan2(2 * mu11, mu20 - mu02)
```

### 3. Segment Transform (Copy-Rotate-Translate)

The actual shadow operation on a segment:

```
# Target orientation is ORTHOGONAL to source
target_orientation = source_orientation + π/2

# Rotation matrix for the difference
θ = target_orientation - source_orientation  # = π/2
R = [[cos(θ), -sin(θ)],
     [sin(θ),  cos(θ)]]

# New centroid: translate toward void
new_centroid = source_centroid + void_direction * distance

# Transform each pixel in segment
local_coord = pixel_coord - source_centroid
rotated_coord = R @ local_coord
dest_coord = new_centroid + rotated_coord
```

### 4. Scatter with Structure Preservation

Only write to void regions:

```
valid = boundary[dest_coord] < threshold
output[dest_coord[valid]] = color_transform(source[pixel_coord[valid]])
```

## Key Tensor Operations

| Operation | Tensor Form |
|-----------|-------------|
| Segment assignment | `seg_id = (fiedler * k).floor().long()` |
| Segment centroids | `scatter_add(coords, seg_id) / scatter_add(ones, seg_id)` |
| Segment moments | `scatter_add(dx*dx, seg_id)`, etc. |
| Per-pixel segment lookup | `centroid[seg_id[pixel]]` — index_select |
| Rotation | `R @ local_coords` — batched matmul |
| Scatter write | `output.index_put_(dest_coords, colors)` |

All operations are tensor-to-tensor. No iteration over segments. The segment structure emerges from spectral clustering; the transforms apply uniformly via gather/scatter.

## Cyclic Color Transform (No Fixed Points)

The color rotation must have no absorbing states. Black shouldn't map to black, white shouldn't map to white.

```python
def cyclic_color(rgb, phase):
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    theta = 2 * pi * lum + phase * pi
    new_r = 0.5 + 0.4 * sin(theta)
    new_g = 0.5 + 0.4 * sin(theta + 2*pi/3)
    new_b = 0.5 + 0.4 * sin(theta + 4*pi/3)
    return stack([new_r, new_g, new_b], dim=-1)
```

At lum=0: θ = phase*π, outputs are (0.5 + 0.4*sin(phase*π), ...) — never (0,0,0)
At lum=1: θ = 2π + phase*π, same — never (1,1,1)

The phase parameter rotates the hue. Different phases give different colors. Repeated application cycles through hues rather than collapsing to black or white.

## Why This Produces Branching

Pass 1: Horizontal segment → shadow rotated 90° → vertical segment in void
Pass 2: That vertical segment → shadow rotated 90° → horizontal segment further out
Pass 3: That horizontal → vertical, etc.

Each generation is orthogonal to its parent. The tree grows perpendicular branches at each level. The color cycles through phases, so each generation has distinct hue.

Without segment-level rotation, you only get parallel sedimentation — each pixel moves individually, no collective rotation, no orthogonal structure, no branching.

## Proper Gating Formula

The gating must use the full formula with gamma and bias:

```
high_gate = (gamma_h * activation + bias_h) > activation
low_gate = (gamma_l * activation + bias_l) < activation
```

Not just `1.3 * activation > activation`. The gamma and bias parameters control the selectivity. When gamma ≠ 1, you get actual outlier detection relative to a transformed baseline.

## Selection from Sample Must Also Be Spectrally Coherent

The drop shadow copies segments FROM the sample image. This selection must use the Laplacian to measure and enforce contiguity.

### Measuring Contiguity via Laplacian Quadratic Form

For a selection vector s (soft or binary), the Laplacian quadratic form measures how much the selection cuts graph edges:

```
cut_cost = s^T @ L_sample @ s = Σ_edges w_ij * (s_i - s_j)²
```

- Low cut_cost → selection is contiguous (neighbors have similar selection values)
- High cut_cost → selection is scattered (selection jumps across edges)

### Enforcing Contiguity via Spectral Smoothing

Raw attention scores may be scattered. Project through partial spectral transform to smooth:

```
# Raw selection from cross-attention
attn = phi_target @ phi_sample.T          # (N_t, N_s)
selection_raw = attn[target_idx, :]       # (N_s,) scores for one target location

# Smooth via partial spectral transform (project to k eigenvectors and back)
selection_spectral = phi_sample @ (phi_sample.T @ selection_raw)

# Or smooth via heat kernel (diffuse along graph edges)
selection_smooth = heat_kernel(L_sample, selection_raw, t=1.0)
```

Both operations respect graph structure: spectral projection keeps only low-frequency (smooth) components, heat kernel diffuses values along edges.

### Threshold Smoothed Selection

After smoothing, threshold to get binary mask:

```
selection_mask = selection_smooth > (selection_smooth.mean() + k * selection_smooth.std())
```

The smoothed selection concentrates on contiguous regions. Thresholding carves out a connected component.

### Verify Contiguity (Optional Check)

Compute cut cost of final selection:

```
cut_cost = selection_mask @ L_sample @ selection_mask
contiguity_score = 1.0 - cut_cost / (selection_mask.sum() * lambda_max)
```

High contiguity_score confirms the selection respects graph structure.

### Full Selection Pipeline

```
1. Cross-attention: attn = phi_t @ phi_s.T
2. For target location i: raw_scores = attn[i, :]
3. Smooth: scores = phi_s @ (phi_s.T @ raw_scores)  # partial spectral transform
4. Threshold: mask = scores > (scores.mean() + k * scores.std())
5. Verify: cut = mask @ L_s @ mask  # should be low relative to mask.sum()
```

This gives contiguous sample segments, not scattered pixels.

## Caricaturize BOTH Images

The shader must exaggerate features of BOTH:

1. **Target caricature**: Thickening bolds the target's own contours. Drop shadows add echoes around target structure. The target becomes more emphatic.

2. **Sample caricature**: The copied segments carry sample's texture/color. When rotated and placed, they bring sample's character into the composition. Repeated application propagates sample features.

Spectral methods apply to BOTH graphs:
- Target spectral basis → WHERE effects happen
- Sample spectral basis → WHAT gets copied
- Cross-attention → HOW they relate

This is why it's "cross-image spectral grafting" — structure from one graph gets grafted onto another, with both contributing to the caricature.
