# Spectral Shader Divergence Review

**Date:** 2026-02-04
**Issue:** Chebyshev path produces non-equivalent results to explicit Fiedler path

---

## Critical Finding: Local vs Global Spectral Structure

**The tiled eigenvector approach computes LOCAL spectral structure per tile.**
**Global Fiedler/Chebyshev approaches compute GLOBAL spectral structure.**

These are fundamentally different operations:

| Approach | What it computes | Range | Correlation with Tiled |
|----------|-----------------|-------|----------------------|
| Tiled Fiedler | Local partition per 64x64 tile | [-0.14, 0.12] | 1.0 (reference) |
| Global Lanczos | Single partition of entire image | [-0.03, 0.03] | 0.006 |
| Chebyshev low-pass | Smoothed image intensity | [0, 1] | 0.15 |
| Eigenvector magnitude approx | Averaged |v_k| across eigenvectors | [0.2, 1] | 0.08 |

**Conclusion:** No global spectral operation can approximate local tiled eigenvectors.
The tiled approach IS the correct algorithm for the shader - it captures local spectral
structure at each location, which is what thickening modulation requires.

The "fast Chebyshev path" should either:
1. Be removed entirely (use explicit tiled path)
2. Use a tiled Chebyshev approach (same complexity, no speedup)
3. Accept non-equivalence and design for different behavior

---

## Measured Divergence (Quantitative)

| Metric | Explicit Path | Chebyshev Path | Divergence |
|--------|---------------|----------------|------------|
| Gate correlation | 1.0 | 0.15 | **85% uncorrelated** |
| High-gate IoU | 1.0 | 0.18 | **82% non-overlapping** |
| High complexity pixels | 236 | 29,512 | **125x more** |
| Segment sizes | ~50 px | ~200 px | **4x larger** |
| Segment centroids | (335, 98) | (192, 93) | **Different regions** |

---

## Algorithmic Divergences

### 1. Gate Computation

**Reference (Explicit Fiedler):**
```python
fiedler = eigenvector[:, :, 1]  # Second smallest eigenvector
threshold = adaptive_threshold(fiedler, 40.0)  # 40th percentile
gate = sigmoid((fiedler - threshold) * sharpness)
```

**Spectral meaning:** Fiedler vector partitions the graph into two connected components. Gate selects one partition (high/low Fiedler values).

**Current (Chebyshev):**
```python
spectral_field = chebyshev_filter(L, signal, center=0.05*lambda_max, width=0.15*lambda_max)
gate = sigmoid((threshold - spectral_field) * sharpness)  # INVERTED
```

**Spectral meaning:** Low-pass filter smooths the signal. Gate now selects *structured regions* (where smoothing differs from original), NOT spectral partition regions.

**Linear algebra divergence:**
- Fiedler: `v_2` such that `L @ v_2 = λ_2 * v_2` - eigenvector structure
- Low-pass: `g(L) @ s` where `g` is Gaussian at low eigenvalues - smoothing operation
- These are **fundamentally different operations** - one partitions, one smooths

---

### 2. Complexity Modulation

**Reference:**
```python
complexity = compute_local_spectral_complexity(fiedler, window_size=7)
# complexity = gradient_magnitude(fiedler) + local_variance(fiedler)
inv_complexity = (1.0 - modulation_strength * complexity).clamp(min=0.1)
weight_at_contours = high_gate_contours.float() * inv_complexity
```

**Spectral meaning:** Fiedler gradient measures rate of change in graph partition membership. High gradient = partition boundary = kink/corner. Inverse modulation = **LESS thickening at kinks** (stability).

**Current (Chebyshev):**
```python
complexity = compute_local_spectral_complexity(spectral_field, window_size=7)
# complexity = gradient_magnitude(low_pass) + local_variance(low_pass)
```

**Spectral meaning:** Low-pass gradient measures rate of change in smoothed intensity. This captures **image edges**, not spectral partition boundaries.

**Measured divergence:**
- Explicit complexity high at 236 pixels (true spectral kinks)
- Chebyshev complexity high at 29,512 pixels (all image edges)
- Correlation: 0.65 (significant but not equivalent)

---

### 3. Selection Targets

**Reference behavior:**
- High gate: One spectral partition → thickening applied
- Low gate: Other partition → segment extraction for shadows
- Modulation: Spectral complexity reduces thickening at partition boundaries

**Current behavior:**
- High gate: Structured/edge regions → thickening everywhere with edges
- Low gate: Smooth regions → segment extraction from flat areas
- Modulation: Image-edge complexity (not spectral) → wrong regions suppressed

---

## Available Approximators (Not Being Used)

The codebase has proper spectral approximators in `spectral_ops_fast.py`:

### `approximate_eigenvector_magnitude_field(L, theta, num_probes=30)`
Approximates: `spectral_field_i = Σ_k weight_k * |v_k[i]|`

This uses stochastic trace estimation to approximate actual eigenvector magnitudes without computing eigenvectors. **Should be used for gate computation.**

### `polynomial_spectral_field(L, signal, theta, num_bands=5)`
Band-pass filtering with multiple bands approximating eigenvector structure.

### `iterative_spectral_transform(L, signal, theta, num_steps=8)`
Iterative approach to spectral position estimation.

---

## Required Fixes

### Fix 1: Use Proper Spectral Approximator for Gate

Replace:
```python
filtered = chebyshev_filter(L, signal, center=0.05*lambda_max, width=0.15*lambda_max)
spectral_field = filtered.reshape(H, W)
```

With:
```python
from spectral_ops_fast import approximate_eigenvector_magnitude_field

# theta=0.0 targets lowest eigenvectors (including Fiedler at λ_2)
spectral_field = approximate_eigenvector_magnitude_field(
    L, theta=0.0,  # Low eigenvalue region
    num_probes=30,
    polynomial_order=40,
    lambda_max=lambda_max,
    num_eigenvectors_approx=4  # Focus on first few eigenvectors
).reshape(H, W)
```

### Fix 2: Compute Complexity from Approximated Eigenvector Field

The complexity should be computed from the spectral field that approximates eigenvector structure, not from a simple low-pass signal.

### Fix 3: Restore Proper Gate Semantics

The gate should NOT be inverted. If using proper eigenvector approximation:
```python
gate = torch.sigmoid((spectral_field - threshold) * sharpness)
```

The inversion was a band-aid for the low-pass filter producing wrong values.

---

## Mathematical Summary

| Operation | Reference | Chebyshev (Current) | Proper Approximation |
|-----------|-----------|---------------------|---------------------|
| Gate signal | `v_2` (Fiedler eigenvector) | `g_{LP}(L) @ s` (low-pass) | `Σ_k w_k |v_k|` (eigenvector approx) |
| Complexity | `∇v_2` (partition boundary) | `∇(g_{LP} @ s)` (image edges) | `∇(Σ_k w_k |v_k|)` (spectral boundary) |
| Selectivity | Spectral partition | Image structure | Spectral partition (approximate) |

The current Chebyshev implementation answers: "Where are edges in the smoothed image?"
The reference implementation answers: "Which spectral partition does this pixel belong to?"

These are **not equivalent questions** and cannot produce equivalent results.
