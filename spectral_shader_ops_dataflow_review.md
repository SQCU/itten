# Spectral Shader Ops - Data Flow and Performance Review

**File Reviewed:** `/home/bigboi/itten/spectral_shader_ops.py`
**Reference Documents:**
- `/home/bigboi/itten/spectral_shader_review_const.md` (prior review)
- `/home/bigboi/itten/what_is_a_shader.md` (data flow specification)

**Reviewer:** Claude Opus 4.5
**Date:** 2026-02-04

---

## 1. Redundant Encapsulating/Wrapping Functions

### Functions Flagged for Potential Inlining

| Function | Location | Issue | Recommendation |
|----------|----------|-------|----------------|
| `compute_shadow_colors` | Lines 126-144 | Wraps `cyclic_color_transform` then applies 3 scalar multiplications. Only 4 lines of actual transformation beyond the call. | **Keep** - The blue bias logic (lines 140-143) is non-trivial enough to warrant encapsulation. However, consider fusing with `compute_front_colors` into a single function that returns both. |
| `compute_front_colors` | Lines 147-166 | Same pattern as above with different bias constants. | **Keep** - Same reasoning. Fuse into `compute_both_colors(rgb, effect_strength) -> (shadow, front)` to avoid calling `cyclic_color_transform` twice. |
| `fiedler_gate` | Lines 28-49 | Single line of actual computation: `torch.sigmoid((fiedler - threshold) * sharpness)` | **Keep** - The docstring provides semantic meaning ("gate" vs "sigmoid"), and the function documents the threshold/sharpness contract. Single-line computation with good abstraction. |
| `adaptive_threshold` | Lines 52-65 | 5 lines of logic wrapped in a function. | **Keep** - The percentile-based threshold is reused in multiple places and encapsulates a specific algorithm choice. |

### Pure Wrappers Identified

**None in `spectral_shader_ops.py`.** All functions add computation or semantic meaning beyond simple delegation.

Note: The prior review (`spectral_shader_review_const.md`) identified `run_batch` in `spectral_shader_main.py` (lines 95-97) as a pure wrapper. That function is not in this file.

---

## 2. Parallelization Opportunities

### Current State of Issues Identified in Prior Review

#### 2.1 Connected Components Loop - Lines 269-296

**Status:** Still serial, as identified in prior review.

```python
# Line 269
max_iters = 50

for _ in range(max_iters):  # Python loop
    # Up - Line 274-278
    labels[1:, :] = torch.where(...)
    # Down - Line 280-284
    labels[:-1, :] = torch.where(...)
    # Left - Line 286-290
    labels[:, 1:] = torch.where(...)
    # Right - Line 292-296
    labels[:, :-1] = torch.where(...)
```

**Analysis:** Each iteration performs 4 vectorized tensor operations, but the Python loop prevents torch.compile from optimizing the entire iteration sequence. The operations within each iteration are correctly vectorized.

**Parallelization Opportunity:**
- The 4 directional propagations within each iteration could be combined into a single max-pool style operation using `torch.nn.functional.max_pool2d` with appropriate kernel, but this changes the algorithm semantics (simultaneous vs sequential direction updates).
- Alternative: Use `scipy.ndimage.label` on CPU or a CUDA-specific connected components kernel.

#### 2.2 Segment Extraction Loop - Lines 324-351

**Status:** Still serial, as identified in prior review.

```python
# Line 324
for label_val in unique_labels:
    mask = labels == label_val           # Line 325 - one comparison per label
    pixel_count = mask.sum().item()      # Line 326 - one reduction per label

    if pixel_count < min_pixels:         # Line 328
        continue

    ys, xs = torch.where(mask)           # Line 331 - one where per label
    # ... bbox, centroid, coords computation
```

**Analysis:** For N unique labels, this performs N equality comparisons, N reductions, and N `torch.where` calls. Each operation is vectorized, but the Python loop adds overhead.

**Parallelization Opportunity:**
```python
# Batched alternative approach:
counts = torch.bincount(labels.flatten() + 1)[1:]  # Skip -1 background
valid_labels = unique_labels[counts >= min_pixels]  # Filter in one op
# Then use scatter/gather operations for batched coordinate extraction
```

### Additional Parallelization Opportunities Found

#### 2.3 Spectral Signature Computation - Lines 945-946

```python
# Line 945-946
sigs_A = torch.stack([compute_segment_spectral_signature(s, fiedler_A) for s in segments_A])
sigs_B = torch.stack([compute_segment_spectral_signature(s, fiedler_B) for s in segments_B])
```

**Issue:** Python list comprehension iterates over segments sequentially.

**Analysis:** Each call to `compute_segment_spectral_signature` (lines 841-870) accesses:
- `segment.coords` for indexing into Fiedler
- Computes mean, std, size, aspect ratio

**Parallelization Opportunity:** Pre-batch all segment coordinates and compute signatures in parallel:
```python
# Pad coordinates to max segment size, use batch indexing into Fiedler
# Compute mean/std via masked operations
```

#### 2.4 Segment Transplantation Loop - Lines 1013-1038

```python
# Line 1014
for seg_A, seg_B in zip(segments_A, matched_B):
    offset = seg_A.centroid - seg_B.centroid  # Line 1017
    transplanted_coords = seg_B.coords + offset.unsqueeze(0)  # Line 1020
    # ... create Segment object
```

**Issue:** Python loop over segment pairs.

**Analysis:** The offset computation and coordinate translation are simple tensor additions that could be batched.

**Parallelization Opportunity:** Batch all centroid offsets and apply via broadcasting, then split back into segments. However, the Segment dataclass structure may require keeping this as a loop for object creation.

---

## 3. Serial Blocking Latency Analysis

### Latency Hotspots Ranked by Estimated Cost

| Rank | Function | Location | Estimated Relative Cost | Bottleneck Type |
|------|----------|----------|------------------------|-----------------|
| 1 | `_torch_connected_components` | Lines 248-309 | **HIGH** (50 iterations x 4 tensor ops) | Python loop prevents fusion; O(iterations x H x W) |
| 2 | `dilate_high_gate_regions` | Lines 617-747 | **MEDIUM-HIGH** | Two heavy ops: Gaussian conv2d + grid_sample; O(H x W x kernel_size^2) for conv, O(H x W) for grid_sample |
| 3 | `extract_segments_from_contours` | Lines 192-245 | **MEDIUM** | Calls `_torch_connected_components` (major cost) + grayscale/contour detection (minor) |
| 4 | `draw_all_segments_batched` | Lines 517-559 | **MEDIUM-LOW** | Two scatter ops + Hadamard composite; O(N_points) for scatter, O(H x W) for composite |
| 5 | `compute_local_spectral_complexity` | Lines 566-614 | **LOW** | Two conv2d ops (Sobel + box filter); well-optimized tensor operations |

### Detailed Cost Breakdown

#### `_torch_connected_components` (Lines 248-309)

**Operations per iteration:**
- 4 tensor slice operations: `labels[1:, :]`, `labels[:-1, :]`, etc.
- 4 `torch.where` calls with tensor conditions
- 4 `torch.minimum` calls

**Total:** 50 iterations x 12 tensor operations = 600 tensor operations in Python loop

**Why this is the bottleneck:**
1. Python loop prevents torch.compile from fusing iterations
2. Each iteration requires synchronization before the next
3. Convergence typically happens well before 50 iterations, but no early exit (by design, for torch.compile compatibility)

#### `dilate_high_gate_regions` (Lines 617-747)

**Heavy operations:**
1. **Gaussian conv2d** (Line 686): `torch.nn.functional.conv2d(weight_padded, kernel_4d)`
   - Cost: O(H x W x (2*radius+1)^2)
   - With `dilation_radius=2`: kernel is 5x5 = 25 multiplications per pixel

2. **grid_sample** (Lines 733-735): `torch.nn.functional.grid_sample(image_4d, sample_grid, mode='nearest', ...)`
   - Cost: O(H x W) with memory access patterns dependent on sample_grid offsets

**Supporting operations (lower cost):**
- Gradient computation (lines 698-704): 4 tensor subtractions
- Meshgrid creation (lines 707-709): memory allocation
- Grid normalization (lines 725-728): 4 elementwise ops

#### `draw_all_segments_batched` (Lines 517-559)

**Operations:**
1. `prepare_batched_segment_data`: O(total_points) - one-time batching
2. Two `scatter_to_layer` calls: O(total_points) each for index assignment
3. `composite_layers_hadamard`: O(H x W) - pure elementwise

**Why this is lower cost:**
- Operations are parallel and well-suited for GPU
- No Python loops in hot path
- Scatter operations are O(points), not O(pixels)

---

## 4. Data Flow Verification Against Documentation

### Comparison with `/home/bigboi/itten/what_is_a_shader.md`

| Documented Function | Implementation | Line Numbers | Match Status |
|---------------------|----------------|--------------|--------------|
| `two_image_shader_pass` | Implemented | 957-1048 | **MATCHES** |
| `adaptive_threshold` | Implemented | 52-65 | **MATCHES** |
| `fiedler_gate` | Implemented | 28-49 | **MATCHES** |
| `dilate_high_gate_regions` | Implemented | 617-747 | **MATCHES** |
| `compute_local_spectral_complexity` | Implemented | 566-614 | **MATCHES** |
| `transfer_segments_A_to_B` | Implemented | 900-954 | **MATCHES** |
| `extract_segments_from_contours` | Implemented | 192-245 | **MATCHES** |
| `_torch_connected_components` | Implemented | 248-309 | **MATCHES** |
| `_labels_to_segments` | Implemented | 312-353 | **MATCHES** |
| `compute_segment_spectral_signature` | Implemented | 841-870 | **MATCHES** |
| `match_segments_by_topology` | Implemented | 873-897 | **MATCHES** |
| `draw_all_segments_batched` | Implemented | 517-559 | **MATCHES** |
| `prepare_batched_segment_data` | Implemented | 360-420 | **MATCHES** |
| `compute_shadow_colors` | Implemented | 126-144 | **MATCHES** |
| `compute_front_colors` | Implemented | 147-166 | **MATCHES** |
| `cyclic_color_transform` | Implemented | 72-123 | **MATCHES** |
| `scatter_to_layer` | Implemented | 427-458 | **MATCHES** |
| `composite_layers_hadamard` | Implemented | 461-514 | **MATCHES** |

### Deviations Found

**None.** The implementation matches the documented data flow exactly.

### Call Graph Verification

The documented call order in `what_is_a_shader.md` (lines 199-257) matches the actual implementation:

1. `two_image_shader_pass` calls:
   - `adaptive_threshold` - **Line 990** calls `adaptive_threshold(fiedler_A, 40)`
   - `fiedler_gate` - **Line 990** calls `fiedler_gate(fiedler_A, ...)`
   - `dilate_high_gate_regions` - **Lines 994-1000**
   - `transfer_segments_A_to_B` - **Lines 1003-1004**
   - `draw_all_segments_batched` - **Lines 1041-1046**

2. `transfer_segments_A_to_B` calls:
   - `fiedler_gate` - **Lines 925-926**
   - `extract_segments_from_contours` - **Lines 929-937**
   - `compute_segment_spectral_signature` - **Lines 945-946**
   - `match_segments_by_topology` - **Line 949**

3. `extract_segments_from_contours` calls:
   - `_torch_connected_components` - **Line 243**
   - `_labels_to_segments` - **Line 245**

All documented relationships are correctly implemented.

---

## 5. Unfused Operations

### 5.1 Double Grayscale Computation

**Locations:**
- Line 223-224 (in `extract_segments_from_contours`)
- Line 649-650 (in `dilate_high_gate_regions`)

```python
# Both locations compute:
gray = 0.299 * image_rgb[:,:,0] + 0.587 * image_rgb[:,:,1] + 0.114 * image_rgb[:,:,2]
```

**Current mitigation:** Lines 799-803 in `spectral_shader_pass` precompute grayscale and contours, passing them via optional parameters:
```python
# Lines 800-803
gray = 0.299 * image_rgb[:,:,0] + 0.587 * image_rgb[:,:,1] + 0.114 * image_rgb[:,:,2]
gray_norm = (gray - gray.mean()) / (gray.std() + 1e-8)
contours = torch.abs(gray_norm) > 1.0
```

**Issue:** This precomputation exists in `spectral_shader_pass` but NOT in `two_image_shader_pass` (lines 957-1048). The two-image path recomputes grayscale twice.

**Fusion opportunity:** Add optional `gray`/`contours` parameters to `two_image_shader_pass` and precompute once.

### 5.2 Shadow and Front Color Computation

**Location:** Lines 417-418

```python
shadow_colors = compute_shadow_colors(colors, effect_strength)
front_colors = compute_front_colors(colors, effect_strength)
```

**Issue:** Both functions call `cyclic_color_transform` internally:
- `compute_shadow_colors` (line 133): `cyclic_color_transform(rgb, rotation_strength=0.3*e, contrast_strength=0.8*e)`
- `compute_front_colors` (line 154): `cyclic_color_transform(rgb, rotation_strength=0.2*e, contrast_strength=0.6*e)`

The base luminance computation is duplicated:
```python
# Inside cyclic_color_transform, line 100:
luminance = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
```

**Fusion opportunity:**
```python
def compute_shadow_and_front_colors(rgb, effect_strength):
    luminance = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
    # Compute both transforms sharing luminance
    shadow = _apply_cyclic_with_luminance(luminance, rgb, 0.3*e, 0.8*e, bias='blue')
    front = _apply_cyclic_with_luminance(luminance, rgb, 0.2*e, 0.6*e, bias='teal')
    return shadow, front
```

### 5.3 Min/Max Normalization

**Location:** Lines 610-612

```python
c_min, c_max = complexity.min(), complexity.max()
complexity = (complexity - c_min) / (c_max - c_min + 1e-8)
```

**Issue:** Two passes over the tensor (one for min, one for max).

**Fusion opportunity:** Use `torch.aminmax()` for single-pass:
```python
c_min, c_max = torch.aminmax(complexity)
```

### 5.4 Gate Computation in transfer_segments_A_to_B

**Location:** Lines 924-926

```python
gate_A = fiedler_gate(fiedler_A, adaptive_threshold(fiedler_A, 40), 10.0)
gate_B = fiedler_gate(fiedler_B, adaptive_threshold(fiedler_B, 40), 10.0)
```

**Issue:** `gate_A` is also computed in `two_image_shader_pass` at line 990:
```python
gate_A = fiedler_gate(fiedler_A, adaptive_threshold(fiedler_A, 40), 10.0)
```

This means `gate_A` is computed twice when `transfer_segments_A_to_B` is called from `two_image_shader_pass`.

**Fusion opportunity:** Pass precomputed `gate_A` to `transfer_segments_A_to_B`:
```python
def transfer_segments_A_to_B(..., gate_A=None, gate_B=None):
    if gate_A is None:
        gate_A = fiedler_gate(fiedler_A, adaptive_threshold(fiedler_A, 40), 10.0)
    ...
```

---

## Summary of Findings

### Redundant Functions
- **0 pure wrappers** identified for removal
- **2 functions** (`compute_shadow_colors`, `compute_front_colors`) could be fused but provide good semantic separation

### Parallelization Opportunities
| Issue | Location | Priority |
|-------|----------|----------|
| Connected components Python loop | Lines 269-296 | HIGH |
| Segment extraction loop | Lines 324-351 | MEDIUM |
| Spectral signature list comprehension | Lines 945-946 | LOW |
| Segment transplantation loop | Lines 1014-1038 | LOW |

### Latency Hotspots (Ranked)
1. `_torch_connected_components` - 50-iteration Python loop
2. `dilate_high_gate_regions` - Gaussian conv2d + grid_sample
3. `extract_segments_from_contours` - Dominated by #1
4. `draw_all_segments_batched` - Well-parallelized scatter operations
5. `compute_local_spectral_complexity` - Efficient conv2d operations

### Data Flow Verification
- **18/18 documented functions implemented**
- **0 deviations from specification**
- Call graph matches documentation exactly

### Unfused Operations
| Issue | Locations | Estimated Savings |
|-------|-----------|-------------------|
| Double grayscale in two-image path | Lines 649-650 (called twice) | 1 tensor pass over H x W |
| Double luminance in shadow/front colors | Lines 100 (called twice) | 1 tensor pass over N points |
| Separate min/max calls | Lines 610-611 | 1 tensor pass over H x W |
| Double gate_A computation | Lines 926 + 990 | 1 sigmoid over H x W |

---

*Review complete. The implementation is architecturally sound and matches the data flow documentation. Primary optimization targets are the connected components loop and grayscale/gate recomputation in the two-image path.*
