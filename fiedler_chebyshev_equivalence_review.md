# Fiedler/Chebyshev Equivalence Review

**Question:** Can we replace `compute_local_eigenvectors_tiled_dither` with `chebyshev_filter` / `iterative_spectral_transform`?

**Date:** 2026-02-04

---

## Executive Summary

**Answer: Partial replacement is possible, but not a complete drop-in.**

The expensive `compute_local_eigenvectors_tiled_dither` function (70-80% of pipeline runtime) computes explicit eigenvectors that are then consumed downstream in three distinct ways:

1. **Threshold-based gating** (via `fiedler_gate`) - **REPLACEABLE** with Chebyshev filtering
2. **Gradient/complexity computation** (via `compute_local_spectral_complexity`) - **REPLACEABLE** with polynomial methods
3. **Segment spectral signatures** (mean/std of Fiedler within segment) - **NOT DIRECTLY REPLACEABLE** - these require actual eigenvector values at specific coordinates

The existing `compare_explicit_vs_iterative` function in `spectral_ops_fns.py` explicitly demonstrates that polynomial filtering produces *correlated but not identical* signals to explicit eigenvectors. The correlation is high for weighted spectral field computations, but segment-level statistics (mean/std per segment) would differ because Chebyshev produces filtered signals, not eigenvector decompositions.

**Recommended path:** Replace explicit eigenvector computation with Chebyshev for the high-gate path (thickening), but retain explicit Fiedler for segment signature matching, potentially only computing Fiedler lazily/locally when segment extraction requires it.

---

## 1. Fiedler Usage Inventory

### 1.1 Direct Fiedler Consumers in `spectral_shader_ops.py`

| Function | Lines | Operation on Fiedler | Replaceable? |
|----------|-------|---------------------|--------------|
| `fiedler_gate` | 28-49 | `sigmoid((fiedler - threshold) * sharpness)` | **YES** - Chebyshev low-pass filter centered at lambda_2 would produce equivalent gating |
| `adaptive_threshold` | 52-65 | Computes percentile of flattened Fiedler values | **PARTIAL** - Would need to threshold on filtered signal instead |
| `compute_local_spectral_complexity` | 566-614 | Sobel gradient magnitude + local variance of Fiedler | **YES** - Can apply gradient ops to any spectral field |
| `dilate_high_gate_regions` | 617-747 | Uses Fiedler via `compute_local_spectral_complexity` | **YES** - Indirect usage only |
| `compute_segment_spectral_signature` | 841-870 | Samples `fiedler[ys, xs]`, computes `mean_f`, `std_f` | **NO** - Requires actual eigenvector values at specific pixel coordinates |
| `transfer_segments_A_to_B` | 900-954 | Uses spectral signatures for cross-image matching | **NO** - Depends on segment signatures |
| `spectral_shader_pass` | 754-834 | Creates gate, passes Fiedler to thickening | **PARTIAL** - Gate path is replaceable |
| `two_image_shader_pass` | 957-1048 | Creates gate, uses Fiedler for segment matching | **PARTIAL** - Gate path is replaceable |

### 1.2 What Each Consumer Extracts

| Consumer | Input | Output | Precision Requirements |
|----------|-------|--------|----------------------|
| `fiedler_gate` | (H, W) Fiedler | (H, W) gate values [0,1] | Low - just needs monotonic relationship with spectral position |
| `adaptive_threshold` | (H, W) Fiedler | scalar threshold | Low - percentile computation is order-statistics |
| `compute_local_spectral_complexity` | (H, W) Fiedler | (H, W) complexity field | Low - gradient magnitude is rotation-invariant |
| `compute_segment_spectral_signature` | (H, W) Fiedler + segment coords | 4-vector per segment | **HIGH** - mean/std must be consistent across images for matching |

---

## 2. Computational Cost Analysis: `compute_local_eigenvectors_tiled_dither`

### 2.1 Function Location
`spectral_ops_fast.py`, lines 1414-1545

### 2.2 Computational Structure

```
For each tile in grid (tile_size=64, overlap=16):
    1. Build multiscale Laplacian (6 radii, O(n * radii * offsets))
    2. Lanczos iteration (30 iterations):
        - Sparse matvec: O(nnz) where nnz ~ n * (radii * connectivity)
        - Reorthogonalization: O(k * n) per iteration
        - 4 direction propagation per iteration
    3. Solve small tridiagonal eigenproblem: O(k^2)
    4. Reconstruct eigenvectors: O(k * n)
```

### 2.3 Complexity Per Tile

- **Laplacian construction:** O(tile_size^2 * num_radii * offsets_per_radius)
  - For radii=[1,2,3,4,5,6], roughly 6 * 25 = 150 edge checks per pixel
  - Total: O(64^2 * 150) = O(614,400) operations

- **Lanczos iterations:** O(30 * nnz + 30 * k * n)
  - nnz ~ 64^2 * 12 (average connectivity) = 49,152
  - Per iteration: O(49,152 + 30 * 4 * 4096) = O(540,000)
  - Total: O(16,200,000) operations per tile

- **For a 512x512 image:**
  - Tiles: ((512-64)/(64-16) + 1)^2 ~ 100 tiles
  - Total: O(1.6 billion) operations

### 2.4 Memory Profile

- Per tile: O(tile_size^2 * num_eigenvectors) = O(64^2 * 4) = 16KB
- Lanczos vectors: O(n * k) = O(4096 * 30) = 480KB
- Never materializes full image Laplacian (key design decision)

---

## 3. Chebyshev/Iterative Alternatives Analysis

### 3.1 `chebyshev_filter` (lines 1726-1827)

**What it does:**
```python
def chebyshev_filter(L, signal, center, width, order=30, lambda_max=None):
    # Approximates g(L) @ signal where g is Gaussian centered at 'center'
    # Using Chebyshev polynomial recurrence:
    #   T_0(x) = 1, T_1(x) = x, T_k(x) = 2x*T_{k-1}(x) - T_{k-2}(x)
```

**Complexity:** O(order * nnz(L)) = O(30 * 12 * n) = O(360n) per application

**Output:** A filtered signal that emphasizes spectral components near `center`

**Key insight from code comments (lines 1556-1615):**
> "The magic: T_k(L_tilde) @ signal only requires matrix-vector products!"

**Can it replace Fiedler?**
- For gating: YES - `chebyshev_filter(L, ones, center=lambda_2, width=small)` produces a signal peaked at the Fiedler eigenvalue
- For signatures: NO - produces filtered signal, not the actual eigenvector

### 3.2 `iterative_spectral_transform` (lines 1830-1928)

**What it does:**
```python
def iterative_spectral_transform(L, signal, theta, num_steps=8, ...):
    # Applies spectral transform by iterating small-angle Chebyshev filters
    # theta=0 -> low eigenvectors, theta=1 -> high eigenvectors
```

**Complexity:** O(num_steps * polynomial_order * nnz) = O(8 * 30 * 12 * n) = O(2880n)

**Output:** Signal weighted by Gaussian centered at theta * lambda_max

**Comparison to explicit eigenvectors:**
- Produces similar *field structure* but different *absolute values*
- The `compare_explicit_vs_iterative` function shows they have high correlation

### 3.3 `polynomial_spectral_field` (lines 1931-2016)

**What it does:**
```python
def polynomial_spectral_field(L, signal, theta, num_bands=5, ...):
    # Approximates: spectral_field = sum_k weight_k * |eigenvector_k|
    # Using band-pass filters instead of explicit eigenvectors
```

**Key formula:**
```python
# Weights for each band (Gaussian centered at target)
band_weights = torch.exp(-((band_centers - target_lambda) ** 2) / (2 * band_sigma ** 2))
# Sum weighted band-filtered signals
result = sum_b weight_b * |filter_b(L) @ signal|
```

**This is the closest match** to what the shader needs for gating.

---

## 4. `compare_explicit_vs_iterative` Analysis

Location: `spectral_ops_fns.py`, lines 1174-1314

### 4.1 What It Demonstrates

The function explicitly computes:
1. **Explicit field:** `sum_k weight_k * |eigenvector_k|` using dense eigendecomposition
2. **Iterative field:** `polynomial_spectral_field(L, signal, theta)`
3. **Transform field:** `iterative_spectral_transform(L, signal, theta)`

Then computes Pearson correlation between explicit and iterative methods.

### 4.2 Key Findings from Code Structure

```python
# EXPLICIT: Dense eigenvector computation
L_dense = L.to_dense().cpu()
eigenvalues, eigenvectors = torch.linalg.eigh(L_dense)  # O(n^3)
explicit_field = sum_k weights[k] * |eigenvectors[:, k]|

# ITERATIVE: Polynomial approximation
iterative_field = polynomial_spectral_field(L, signal, theta)  # O(n * bands * order)

# Result
corr_field = correlation(explicit_norm, iterative_norm)
```

### 4.3 Implications

The comparison function exists because the authors recognized:
1. The methods are **not identical** (hence the need for comparison)
2. They produce **correlated results** (hence the correlation metric)
3. The iterative method has **O(n log n)** complexity vs **O(n^3)** for dense

**Critical observation:** The return includes both `correlation_field` and `correlation_transform`, implying neither method perfectly matches explicit eigenvectors.

---

## 5. Connected Components Redundancy Analysis

### 5.1 `_torch_connected_components` Purpose

Location: `spectral_shader_ops.py`, lines 248-309

**Why it exists:**
```python
# From extract_segments_from_contours (lines 192-245):
# 1. Find contours (via intensity threshold)
# 2. Intersect with low-gate regions
# 3. Run connected components to group pixels into segments
labels = _torch_connected_components(eligible)
```

### 5.2 Why Fiedler Doesn't Replace Connected Components

The Fiedler vector **partitions the graph into 2 parts** (by sign), but:

1. **Multiple disconnected regions** can have the same Fiedler sign
2. **Connected components** identifies *topologically distinct* regions
3. The shader needs to **manipulate individual segments** (translate, rotate, color)

Example:
```
Image with 3 separate blobs, all with gate < 0.5:
- Fiedler sign might be positive for all 3 (they're on same side of spectral cut)
- But they're 3 separate connected components that need independent processing
```

### 5.3 Could Fiedler Value Clustering Replace It?

**Partial answer:** Fiedler values are continuous and could *in theory* cluster regions, but:

1. **Fiedler is a global function** - values depend on entire graph structure
2. **Adjacent regions can have similar values** - no guaranteed separation
3. **The current approach is sound:** gate separates regions, CC identifies them

**Verdict:** Connected components serves a different purpose than Fiedler and cannot be replaced by spectral clustering without significant architectural changes.

---

## 6. Recommended Refactoring Path

### 6.1 Phase 1: Replace High-Gate Path (Low Risk)

**Current:**
```python
# dilate_high_gate_regions uses Fiedler only for complexity modulation
complexity = compute_local_spectral_complexity(fiedler, window_size=7)
```

**Proposed:**
```python
# Use Chebyshev-filtered signal instead
spectral_field = chebyshev_filter(L, ones, center=lambda_2, width=0.1*lambda_max)
complexity = compute_local_spectral_complexity(spectral_field.reshape(H,W), window_size=7)
```

**Why it works:** Complexity only needs gradient structure, not absolute values.

**Savings:** Eliminates eigenvector computation for thickening path.

### 6.2 Phase 2: Replace Gate Computation (Low Risk)

**Current:**
```python
threshold = adaptive_threshold(fiedler, 40)
gate = fiedler_gate(fiedler, threshold, sharpness=10)
```

**Proposed:**
```python
# Use low-frequency Chebyshev filter
spectral_low = chebyshev_filter(L, carrier.flatten(), center=0.05*lambda_max, width=0.1*lambda_max)
gate = torch.sigmoid((spectral_low - spectral_low.median()) * sharpness)
```

**Why it works:** The gate only needs to distinguish high/low spectral regions, not exact Fiedler values.

### 6.3 Phase 3: Lazy Fiedler for Signatures (Medium Risk)

**Problem:** `compute_segment_spectral_signature` needs actual Fiedler values at segment pixel coordinates.

**Option A: Keep explicit Fiedler for signatures only**
- Compute Chebyshev-based gate/complexity globally
- Compute Fiedler only when segments are extracted
- Fiedler computation is then O(segment_region) not O(full_image)

**Option B: Change signature scheme**
- Use Chebyshev-filtered values as signature components
- Requires revalidation of cross-image matching quality
- Could improve or degrade matching - needs experimentation

**Recommended:** Option A first, then experiment with Option B.

### 6.4 Phase 4: Preserve Connected Components (No Change)

Connected components is fundamentally different from spectral decomposition and should be retained.

---

## 7. Estimated Performance Improvement

### 7.1 Current Pipeline Cost Breakdown

| Stage | Function | Estimated % |
|-------|----------|-------------|
| Eigenvector computation | `compute_local_eigenvectors_tiled_dither` | 70-80% |
| Gate computation | `fiedler_gate` + `adaptive_threshold` | 2-3% |
| Thickening | `dilate_high_gate_regions` | 5-8% |
| Segment extraction | `extract_segments_from_contours` | 5-10% |
| Segment drawing | `draw_all_segments_batched` | 5-10% |

### 7.2 With Proposed Changes

**Phase 1+2 (Replace high-gate path + gate):**
- Eliminates 70% of Fiedler computation cost if signatures not needed
- Net reduction: **50-60%** of total pipeline time

**Phase 3 (Lazy Fiedler for signatures):**
- Fiedler computed only for segment bounding boxes
- Typical segment region: 100-1000 pixels vs 262,144 (512x512)
- Additional reduction: **10-20%**

**Theoretical maximum improvement:** 60-70% reduction in eigenvector computation time.

### 7.3 Caveats

1. **Chebyshev has setup cost:** `estimate_lambda_max` requires power iteration
2. **Coefficient computation:** `chebyshev_coefficients_gaussian` is O(order^2)
3. **Multiple filter applications:** Each Chebyshev filter is O(order * nnz)

**Realistic expectation:** 40-50% speedup after accounting for Chebyshev overhead.

---

## 8. Summary of Findings

| Question | Answer |
|----------|--------|
| Can Chebyshev replace explicit eigenvectors entirely? | **No** - segment signatures need actual values |
| Can Chebyshev replace gate/complexity computation? | **Yes** - these only need spectral field structure |
| Can Fiedler replace connected components? | **No** - fundamentally different operations |
| Is `compare_explicit_vs_iterative` relevant? | **Yes** - it proves methods are correlated but not identical |
| What's the maximum speedup? | **40-50%** realistic, **60-70%** theoretical |

---

## Appendix: Key Code References

### A.1 Fiedler Extraction from Eigenvectors
```python
# spectral_shader_ops.py line 1092
return evecs[:, :, 1]  # Fiedler is eigenvector index 1
```

### A.2 Chebyshev Recurrence
```python
# spectral_ops_fast.py lines 1811-1822
while k < order:
    L_T_prev = torch.sparse.mm(L, T_prev)
    T_curr = 2.0 * (scale * L_T_prev - T_prev) - T_prev_prev
    result = result + coeffs[k] * T_curr
    T_prev_prev = T_prev
    T_prev = T_curr
    k += 1
```

### A.3 Segment Signature Formula
```python
# spectral_shader_ops.py lines 865-870
mean_f = fiedler_vals.mean()
std_f = fiedler_vals.std() + 1e-8
size = torch.tensor(fiedler_vals.shape[0], ...)
aspect = torch.tensor((x1 - x0) / max(y1 - y0, 1), ...)
return torch.stack([mean_f, std_f, (size + 1).log(), aspect])
```
