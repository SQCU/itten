# Spectral Ops Fast Code Review

**File Reviewed:**
- `/home/bigboi/itten/spectral_ops_fast.py`

**Reference Documents:**
- `/home/bigboi/itten/what_is_a_shader.md` (data flow documentation)
- `/home/bigboi/itten/hypercontexts/constitutional-orientation.md` (coding standards)
- `/home/bigboi/itten/spectral_shader_review_const.md` (format reference)

**Reviewer:** Claude Opus 4.5
**Date:** 2026-02-04

---

## Executive Summary

The `spectral_ops_fast.py` module implements GPU-accelerated spectral graph operations using sparse tensor algebra. The code demonstrates sophisticated understanding of Lanczos iteration, Chebyshev polynomial filtering, and graph Laplacian construction. However, it contains **critical violations** of the numpy-free computation principle: numpy is used extensively in computational hot paths, not just at I/O boundaries.

**Strengths:**
- Well-documented mathematical framework (especially the Chebyshev section, lines 1563-1630)
- Proper use of sparse tensor operations for Laplacian construction and matvec
- GPU-first design with `DEVICE` selection
- Graph dataclass with clean property-based interface
- Comprehensive eigenvector computation via Lanczos

**Critical Issues:**
- **14 functions** use numpy for computation, not just I/O
- Numpy arrays are created, manipulated, and returned from computational functions
- Conversion round-trips (numpy -> torch -> numpy) waste memory and break GPU pipelines

**Architecture Issue:**
- Several functions have overly complex signatures that should follow `x, h, **kwargs` pattern
- Config key repacking instead of kwargs passthrough

**Overall Assessment:** The mathematical algorithms are correct and well-implemented, but the numpy usage violates the core principle that "numpy should only exist at PIL I/O boundaries, never in computation." This requires **critical rewrites** of multiple functions.

---

## CRITICAL NUMPY VIOLATIONS

These functions use numpy for **computation**, not just I/O. This is an end-of-world priority violation per the review specification.

---

### 1. `lanczos_k_eigenvectors` (Lines 642-748)

**Location:** Lines 718-719, 736, 742, 748

```python
# Line 718-719: Returns numpy arrays
def lanczos_k_eigenvectors(
    ...
) -> Tuple[np.ndarray, np.ndarray]:  # VIOLATION: returns numpy

# Line 736: Creates numpy zeros for return
if len(valid_indices) == 0:
    return np.zeros((n, num_eigenvectors)), np.zeros(num_eigenvectors)

# Line 742: Converts eigenvalues to numpy
selected_eigenvalues = eigenvalues[selected_indices].cpu().numpy()

# Line 748: Converts eigenvectors to numpy for return
return eigenvectors_full.cpu().numpy(), selected_eigenvalues
```

**Issue:** This function does all computation on GPU with torch, then converts to numpy for return. Any caller must convert back to torch to continue GPU computation.

**Recommended Fix:** Return torch tensors. Let callers convert at the I/O boundary if needed.

```python
def lanczos_k_eigenvectors(
    L: torch.Tensor,
    num_eigenvectors: int,
    ...
) -> Tuple[torch.Tensor, torch.Tensor]:  # Return torch, not numpy
    ...
    return eigenvectors_full, eigenvalues[selected_indices]
```

---

### 2. `compute_local_eigenvectors_tiled` (Lines 1055-1162)

**Location:** Lines 1056-1062, 1085-1091, 1139, 1156-1161

```python
# Line 1056-1062: Takes numpy, returns numpy
def compute_local_eigenvectors_tiled(
    image: np.ndarray,  # VIOLATION: numpy input
    ...
) -> np.ndarray:  # VIOLATION: numpy output

# Lines 1085-1091: Numpy operations on image
if image.max() > 1.0:
    image = image.astype(np.float32) / 255.0  # numpy division
else:
    image = image.astype(np.float32)  # numpy cast

# Lines 1091-1092: Numpy accumulator arrays
result = np.zeros((height, width, num_eigenvectors), dtype=np.float32)
weights = np.zeros((height, width), dtype=np.float32)

# Line 1139: Numpy outer product for blend weights
blend_weight = np.outer(blend_y, blend_x)  # VIOLATION: numpy computation

# Lines 1156-1161: Numpy division in normalization loop
mask = weights > 1e-10
k_idx = 0
while k_idx < num_eigenvectors:
    result[:, :, k_idx][mask] /= weights[mask]  # numpy division
    k_idx += 1
```

**Issue:** This entire function operates in numpy space with occasional torch calls for tile processing. The accumulator (`result`, `weights`), blend weight computation, and normalization are all numpy.

**Recommended Fix:** Rewrite to use torch tensors throughout. The blend weight outer product, accumulation, and normalization can all be vectorized torch operations.

---

### 3. `_compute_tile_eigenvectors` (Lines 1165-1273)

**Location:** Lines 1166-1170, 1188, 1265-1270

```python
# Lines 1166-1170: Takes and returns numpy
def _compute_tile_eigenvectors(
    tile: np.ndarray,  # VIOLATION: numpy input
    ...
) -> np.ndarray:  # VIOLATION: numpy output

# Line 1188: Returns numpy zeros
if n < 3 or k < 1:
    return np.zeros((tile_h, tile_w, num_eigenvectors), dtype=np.float32)

# Lines 1265-1270: Converts to numpy for reshape and return
evs_cpu = eigenvectors_full.cpu().numpy()  # conversion
result = np.zeros((tile_h, tile_w, num_eigenvectors), dtype=np.float32)

k_idx = 0
while k_idx < take_count:
    result[:, :, k_idx] = evs_cpu[:, k_idx].reshape(tile_h, tile_w)  # numpy reshape
    k_idx += 1

return result
```

**Issue:** Takes numpy tile, converts to torch for Laplacian/Lanczos, converts back to numpy for return. The reshape and accumulation happen in numpy.

**Recommended Fix:** Take torch tensor input, return torch tensor output. Let the caller handle I/O conversion.

---

### 4. `_create_blend_weights_1d` (Lines 1276-1299)

**Location:** Entire function (Lines 1276-1299)

```python
def _create_blend_weights_1d(length: int, overlap: int) -> np.ndarray:
    """Create 1D blending weights..."""
    weights = np.ones(length, dtype=np.float32)  # VIOLATION

    if overlap <= 0 or length <= 2 * overlap:
        return weights

    # Taper at start
    i = 0
    while i < overlap:
        weights[i] = (i + 1) / (overlap + 1)  # numpy indexing
        i += 1

    # Taper at end
    i = 0
    while i < overlap:
        weights[length - 1 - i] = (i + 1) / (overlap + 1)
        i += 1

    return weights
```

**Issue:** Pure numpy function creating blend weights. Called from `compute_local_eigenvectors_tiled` which then uses `np.outer` on the result.

**Recommended Fix:** Rewrite as torch:

```python
def _create_blend_weights_1d(length: int, overlap: int, device: torch.device) -> torch.Tensor:
    weights = torch.ones(length, dtype=torch.float32, device=device)
    if overlap <= 0 or length <= 2 * overlap:
        return weights
    # Vectorized taper
    taper = torch.arange(1, overlap + 1, device=device, dtype=torch.float32) / (overlap + 1)
    weights[:overlap] = taper
    weights[-overlap:] = taper.flip(0)
    return weights
```

---

### 5. `_compute_tile_eigenvectors_multiscale` (Lines 1306-1425)

**Location:** Lines 1307-1313, 1340-1341, 1391, 1417-1423

```python
# Lines 1307-1313: Takes and returns numpy
def _compute_tile_eigenvectors_multiscale(
    tile: np.ndarray,  # VIOLATION
    ...
) -> np.ndarray:  # VIOLATION

# Lines 1340-1341: Returns numpy zeros
if n < 3 or k < 1:
    return np.zeros((tile_h, tile_w, num_eigenvectors), dtype=np.float32)

# Line 1391: Returns numpy zeros
if actual_k < 2:
    return np.zeros((tile_h, tile_w, num_eigenvectors), dtype=np.float32)

# Lines 1417-1423: Numpy conversion and reshape
evs_cpu = eigenvectors_full.cpu().numpy()
result = np.zeros((tile_h, tile_w, num_eigenvectors), dtype=np.float32)

k_idx = 0
while k_idx < take_count:
    result[:, :, k_idx] = evs_cpu[:, k_idx].reshape(tile_h, tile_w)
    k_idx += 1

return result
```

**Issue:** Near-duplicate of `_compute_tile_eigenvectors` with same numpy violations. Uses multiscale Laplacian but same numpy I/O pattern.

**Recommended Fix:** Same as `_compute_tile_eigenvectors` - take/return torch tensors.

---

### 6. `compute_local_eigenvectors_tiled_dither` (Lines 1428-1560)

**Location:** Lines 1429-1437, 1471-1478, 1534, 1552-1557

```python
# Lines 1429-1437: Takes numpy, returns numpy
def compute_local_eigenvectors_tiled_dither(
    image: np.ndarray,  # VIOLATION
    ...
) -> np.ndarray:  # VIOLATION

# Lines 1471-1478: Numpy array operations
if image.max() > 1.0:
    image = image.astype(np.float32) / 255.0
else:
    image = image.astype(np.float32)

result = np.zeros((height, width, num_eigenvectors), dtype=np.float32)
weights = np.zeros((height, width), dtype=np.float32)

# Line 1534: Numpy outer product
blend_weight = np.outer(blend_y, blend_x)

# Lines 1552-1557: Numpy division
mask = weights > 1e-10
k_idx = 0
while k_idx < num_eigenvectors:
    result[:, :, k_idx][mask] /= weights[mask]
    k_idx += 1
```

**Issue:** Near-duplicate of `compute_local_eigenvectors_tiled` with same numpy violations. This is the dither-aware variant.

**Recommended Fix:** Same pattern - torch throughout.

---

### 7. `fast_spectral_etch_field` (Lines 2209-2266)

**Location:** Lines 2210-2215, 2257-2266

```python
# Lines 2210-2215: Takes numpy, returns numpy
def fast_spectral_etch_field(
    carrier: np.ndarray,  # VIOLATION
    ...
) -> np.ndarray:  # VIOLATION

# Lines 2257-2266: Numpy operations for reshape and normalization
field_np = field.cpu().numpy().reshape(height, width)  # conversion

f_min, f_max = field_np.min(), field_np.max()  # numpy min/max
if f_max > f_min:
    field_np = (field_np - f_min) / (f_max - f_min)  # numpy division
else:
    field_np = np.zeros_like(field_np)  # numpy zeros

return field_np
```

**Issue:** Entry point function takes numpy carrier, does torch computation, returns numpy. The normalization at the end is pure numpy.

**Recommended Fix:** Take torch tensor, return torch tensor. Normalization can be `(field - field.min()) / (field.max() - field.min() + 1e-10)`.

---

### 8. `Graph.coord_of` (Lines 78-85)

**Location:** Lines 84-85

```python
def coord_of(self, node: int) -> Optional[Tuple[float, ...]]:
    """Get coordinates of a node, if available."""
    if self.coords is None:
        return None
    if node < 0 or node >= self.num_nodes:
        return None
    coord = self.coords[node].cpu().numpy()  # VIOLATION: converts to numpy
    return tuple(coord.tolist())  # then to list then to tuple
```

**Issue:** Converts torch tensor to numpy to extract coordinate, then converts to Python tuple. The numpy step is unnecessary.

**Recommended Fix:**
```python
coord = self.coords[node].cpu()
return tuple(coord.tolist())  # direct from torch tensor
```

---

### 9. `chebyshev_coefficients_gaussian` (Lines 1687-1738)

**Location:** Line 1717

```python
# Line 1717: Uses np.pi
theta = (2 * j - 1) * np.pi / (2 * num_quadrature)
```

**Issue:** Uses `np.pi` constant. While this is minor (it's just a constant), it's an unnecessary numpy dependency.

**Recommended Fix:** Use `math.pi` or `torch.pi`.

---

### 10. Import statement (Line 10)

**Location:** Line 10

```python
import numpy as np
```

**Issue:** Numpy is imported at module level and used throughout. For a module claiming "All numerical hot paths are vectorized tensor operations" (line 4), this import enables violations of that principle.

**Recommended Fix:** After rewriting functions to use torch, remove this import or restrict to a specific I/O section.

---

## Summary of Critical Numpy Functions

| Function | Lines | Input Type | Output Type | Numpy Operations |
|----------|-------|------------|-------------|------------------|
| `lanczos_k_eigenvectors` | 642-748 | torch | **numpy** | zeros, conversion |
| `compute_local_eigenvectors_tiled` | 1055-1162 | **numpy** | **numpy** | zeros, astype, outer, division |
| `_compute_tile_eigenvectors` | 1165-1273 | **numpy** | **numpy** | zeros, reshape |
| `_create_blend_weights_1d` | 1276-1299 | int | **numpy** | ones, indexing |
| `_compute_tile_eigenvectors_multiscale` | 1306-1425 | **numpy** | **numpy** | zeros, reshape |
| `compute_local_eigenvectors_tiled_dither` | 1428-1560 | **numpy** | **numpy** | zeros, astype, outer, division |
| `fast_spectral_etch_field` | 2209-2266 | **numpy** | **numpy** | reshape, min, max, division, zeros_like |
| `Graph.coord_of` | 78-85 | torch | tuple | unnecessary conversion |
| `chebyshev_coefficients_gaussian` | 1687-1738 | float | torch | np.pi constant |

**Total: 9 functions with numpy violations, plus module-level import**

---

## MEDIUM PRIORITY: Signature Issues

These functions have input signatures more complex than `x, h, **kwargs_for_config`.

---

### 1. `compute_local_eigenvectors_tiled` (Lines 1055-1062)

```python
def compute_local_eigenvectors_tiled(
    image: np.ndarray,
    tile_size: int = 64,
    overlap: int = 16,
    num_eigenvectors: int = 4,
    edge_threshold: float = 0.1,
    lanczos_iterations: int = 30
) -> np.ndarray:
```

**Issue:** 6 parameters including 5 configuration values. Should be `image, **kwargs` where kwargs contains tile/overlap/eigenvector config.

**Recommended Signature:**
```python
def compute_local_eigenvectors_tiled(
    image: torch.Tensor,
    **config
) -> torch.Tensor:
    tile_size = config.get('tile_size', 64)
    overlap = config.get('overlap', 16)
    ...
```

---

### 2. `compute_local_eigenvectors_tiled_dither` (Lines 1428-1437)

```python
def compute_local_eigenvectors_tiled_dither(
    image: np.ndarray,
    tile_size: int = 64,
    overlap: int = 16,
    num_eigenvectors: int = 4,
    radii: List[int] = [1, 2, 3, 4, 5, 6],
    radius_weights: List[float] = [1.0, 0.6, 0.4, 0.3, 0.2, 0.1],
    edge_threshold: float = 0.15,
    lanczos_iterations: int = 30
) -> np.ndarray:
```

**Issue:** 8 parameters including 7 configuration values. Even worse than the base tiled function.

**Recommended Signature:**
```python
def compute_local_eigenvectors_tiled_dither(
    image: torch.Tensor,
    **config
) -> torch.Tensor:
```

---

### 3. `_compute_tile_eigenvectors_multiscale` (Lines 1306-1313)

```python
def _compute_tile_eigenvectors_multiscale(
    tile: np.ndarray,
    num_eigenvectors: int,
    radii: List[int],
    radius_weights: List[float],
    edge_threshold: float,
    lanczos_iterations: int
) -> np.ndarray:
```

**Issue:** 6 required parameters. As a private helper, this could receive kwargs from the parent.

---

### 4. `build_multiscale_image_laplacian` (Lines 942-1048)

```python
def build_multiscale_image_laplacian(
    carrier: torch.Tensor,
    radii: List[int] = [1, 2, 3, 4, 5, 6],
    radius_weights: List[float] = [1.0, 0.6, 0.4, 0.3, 0.2, 0.1],
    edge_threshold: float = 0.1
) -> torch.Tensor:
```

**Issue:** Mutable default arguments (`List[int] = [1, 2, 3, 4, 5, 6]`). This is a Python anti-pattern - mutable defaults can cause subtle bugs.

**Recommended Fix:**
```python
def build_multiscale_image_laplacian(
    carrier: torch.Tensor,
    radii: Optional[List[int]] = None,
    radius_weights: Optional[List[float]] = None,
    edge_threshold: float = 0.1
) -> torch.Tensor:
    if radii is None:
        radii = [1, 2, 3, 4, 5, 6]
    if radius_weights is None:
        radius_weights = [1.0, 0.6, 0.4, 0.3, 0.2, 0.1]
```

---

### 5. `iterative_spectral_transform` (Lines 1845-1943)

```python
def iterative_spectral_transform(
    L: torch.Tensor,
    signal: torch.Tensor,
    theta: float,
    num_steps: int = 8,
    polynomial_order: int = 30,
    lambda_max: Optional[float] = None,
    sigma_factor: float = 0.3
) -> torch.Tensor:
```

**Issue:** 7 parameters. The algorithm-specific parameters (`num_steps`, `polynomial_order`, `sigma_factor`) should be bundled.

---

## OTHER FINDINGS BY SEVERITY

---

## High Priority

### 1. Duplicate Code: `_compute_tile_eigenvectors` vs `_compute_tile_eigenvectors_multiscale`

**Location:** Lines 1165-1273 vs 1306-1425

**Issue:** These two functions are 90% identical. The only difference is which Laplacian builder is called:
- Line 1183: `L = build_weighted_image_laplacian(tile_tensor, edge_threshold)`
- Line 1335: `L = build_multiscale_image_laplacian(tile_tensor, radii, radius_weights, edge_threshold)`

Everything else (Lanczos iteration, tridiagonal solve, eigenvector reconstruction) is copy-pasted.

**Recommended Fix:** Extract shared Lanczos logic into a helper:

```python
def _lanczos_eigenvectors_from_laplacian(
    L: torch.Tensor,
    tile_shape: Tuple[int, int],
    num_eigenvectors: int,
    lanczos_iterations: int
) -> torch.Tensor:
    """Core Lanczos logic for a pre-built Laplacian."""
    ...

def _compute_tile_eigenvectors(tile, num_eigenvectors, edge_threshold, lanczos_iterations):
    L = build_weighted_image_laplacian(tile, edge_threshold)
    return _lanczos_eigenvectors_from_laplacian(L, tile.shape, num_eigenvectors, lanczos_iterations)

def _compute_tile_eigenvectors_multiscale(tile, num_eigenvectors, radii, radius_weights, edge_threshold, lanczos_iterations):
    L = build_multiscale_image_laplacian(tile, radii, radius_weights, edge_threshold)
    return _lanczos_eigenvectors_from_laplacian(L, tile.shape[:2], num_eigenvectors, lanczos_iterations)
```

---

### 2. Duplicate Code: `compute_local_eigenvectors_tiled` vs `compute_local_eigenvectors_tiled_dither`

**Location:** Lines 1055-1162 vs 1428-1560

**Issue:** These functions are ~85% identical. Both do:
1. Normalize image
2. Create result/weights accumulators
3. Compute tile positions
4. Loop over tiles calling helper
5. Blend and normalize

The only differences:
- Which tile helper is called
- Print statements in dither version
- RGB handling in dither version

**Recommended Fix:** Extract shared tiling logic:

```python
def _tiled_eigenvector_computation(
    image: torch.Tensor,
    tile_processor: Callable,
    tile_size: int,
    overlap: int,
    num_eigenvectors: int,
    **processor_kwargs
) -> torch.Tensor:
    """Generic tiled eigenvector computation with pluggable tile processor."""
    ...
```

---

### 3. While loops where tensor operations suffice

**Location:** Multiple places

The docstring claims "Graph traversal uses while loops (explicitly allowed, not hot path)" (line 5), but while loops are also used in computational sections:

```python
# Line 1110-1153: Nested while loops for tile iteration
y_idx = 0
while y_idx < len(y_starts):
    ...
    x_idx = 0
    while x_idx < len(x_starts):
        ...

# Lines 1157-1159: While loop for normalization
k_idx = 0
while k_idx < num_eigenvectors:
    result[:, :, k_idx][mask] /= weights[mask]
    k_idx += 1
```

**Issue:** The tile iteration loop is justified (non-uniform work per tile), but the normalization while loop could be vectorized:

```python
# Vectorized alternative
result[mask.unsqueeze(-1).expand_as(result)] /= weights[mask].unsqueeze(-1)
# Or simply:
result = result / weights.unsqueeze(-1).clamp(min=1e-10)
```

---

### 4. Redundant `.coalesce()` calls

**Location:** Lines 57-58, 89-90, 111

```python
# Lines 57-58 in Graph.neighbors
indices = self.adjacency.coalesce().indices()
values = self.adjacency.coalesce().values()  # coalesce called twice

# Lines 89-90 in Graph.edge_weight
indices = self.adjacency.coalesce().indices()
values = self.adjacency.coalesce().values()  # coalesce called twice again
```

**Issue:** `coalesce()` is an expensive operation that deduplicates and sorts sparse tensor indices. Calling it twice in succession wastes computation.

**Recommended Fix:**
```python
adj = self.adjacency.coalesce()
indices = adj.indices()
values = adj.values()
```

---

## Medium Priority

### 5. Magic numbers without documentation

**Location:** Multiple places

```python
# Line 235, 246: Diagonal edge weight scaling
weights_d1 = torch.exp(-diff_d1 / edge_threshold) * 0.707  # 1/sqrt(2)
weights_d2 = torch.exp(-diff_d2 / edge_threshold) * 0.707

# Line 1684: Safety margin for lambda_max
return lambda_est * 1.05  # Why 1.05?

# Line 1914-1915: Minimum sigma
if sigma < 0.05 * lambda_max:
    sigma = 0.05 * lambda_max  # Why 0.05?

# Line 2173: Filter width
filter_width = lambda_max / (num_eigenvectors_approx + 1)  # Why this formula?
```

**Recommendation:** Add brief comments explaining the rationale, especially for the Chebyshev filter parameters which have mathematical significance.

---

### 6. Inconsistent device handling

**Location:** Throughout

Some functions use `DEVICE` global, others accept `device` parameter, some do both:

```python
# Line 14: Global device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Line 156: Parameter with default
def from_image(cls, image, ..., device: Optional[torch.device] = None) -> 'Graph':
    if device is None:
        device = DEVICE  # Falls back to global

# Line 543, 549, etc.: Uses DEVICE directly
v = torch.randn(n, device=DEVICE, dtype=torch.float32)
```

**Recommendation:** Consistently use device parameter throughout, falling back to a sensible default. The global `DEVICE` pattern makes testing difficult.

---

### 7. Type annotation inconsistencies

**Location:** Various

```python
# Line 17-21: Protocol with type hints
class GraphView(Protocol):
    def neighbors(self, node: int) -> List[int]: ...

# Line 468-471: Function returns Tuple but docstring says "(L_sparse, node_list, node_to_idx)"
def build_sparse_laplacian(
    graph: GraphView,
    active_nodes: Set[int]
) -> Tuple[torch.Tensor, List[int], Dict[int, int]]:

# Lines 1687-1738: Returns torch.Tensor but function name suggests coefficients
def chebyshev_coefficients_gaussian(...) -> torch.Tensor:
```

**Issue:** The return type `torch.Tensor` for `chebyshev_coefficients_gaussian` is correct but could be more specific (1D tensor of coefficients).

---

### 8. Potential numerical instability in Lanczos

**Location:** Lines 551-557, 601-606

```python
# Lines 551-557: Division by possibly small norm
v_norm = torch.linalg.norm(v)
if v_norm < tol:
    v = torch.ones(n, device=DEVICE, dtype=torch.float32)
    v[::2] = -1.0
    v = v - v.mean()
    v_norm = torch.linalg.norm(v)
v = v / v_norm  # What if v_norm is still < tol?
```

**Issue:** After the fallback initialization, there's no check that `v_norm` is now large enough. For a graph with all-zero Laplacian, this could still divide by a tiny number.

**Recommended Fix:** Add assertion or use safe division:
```python
v = v / (v_norm + 1e-10)
```

---

## Low Priority

### 9. Print statements in production code

**Location:** Lines 1497, 1548, 1559

```python
print(f"Processing {total_tiles} tiles ({len(y_starts)}x{len(x_starts)}) with radii {radii}...")
...
print(f"  {tile_idx}/{total_tiles} tiles...")
...
print(f"Done. Output shape: {result.shape}")
```

**Issue:** Progress printing should use `logging` module or be optional via a `verbose` parameter.

---

### 10. Unused variable in `Graph.neighbors`

**Location:** Line 58

```python
def neighbors(self, node: int) -> List[int]:
    indices = self.adjacency.coalesce().indices()
    values = self.adjacency.coalesce().values()  # values never used

    mask = indices[0] == node
    neighbor_indices = indices[1][mask]
    return neighbor_indices.cpu().tolist()
```

**Issue:** `values` is computed but never used. This wastes memory for large sparse tensors.

---

### 11. Late dataclass import

**Location:** Line 24

```python
from dataclasses import dataclass
```

**Issue:** Import is after the Protocol definition at line 17. While not strictly incorrect, it's conventional to group all imports at the top.

---

### 12. Long mathematical comment block

**Location:** Lines 1563-1630

**Issue:** The 70-line comment block explaining Chebyshev polynomials is excellent documentation, but it could be moved to a separate markdown file or module docstring for better discoverability.

**Recommendation:** Keep inline but add a brief summary at the top pointing to the detailed explanation:
```python
# See lines 1563-1630 for detailed mathematical derivation
```

---

## Positive Observations

1. **Excellent mathematical documentation** (Lines 1563-1630): The Chebyshev polynomial filtering explanation is thorough and correct.

2. **Clean Graph dataclass** (Lines 27-466): The `Graph` class provides a clean interface with useful methods and proper sparse tensor handling.

3. **Correct Lanczos implementation** (Lines 530-639): The Lanczos iteration with full reorthogonalization is numerically stable.

4. **Good use of sparse operations**: Functions like `build_weighted_image_laplacian` and `build_multiscale_image_laplacian` correctly construct sparse Laplacians without dense intermediate matrices.

5. **Proper eigenvalue estimation** (Lines 1634-1684): Power iteration for `lambda_max` with convergence checking.

6. **1D signal handling** (Lines 1787-1791, 1839-1840): Functions properly handle both 1D and 2D signals with `was_1d` pattern.

7. **Edge case handling**: Empty graph handling (lines 299-307, 408-416), small graph handling (lines 542-543), and degenerate Laplacian handling (lines 1024-1029).

---

## Summary of Recommendations

### Critical (Must Fix)
1. Rewrite all 9 numpy-using functions to use torch tensors
2. Remove `import numpy as np` after torch rewrite
3. Change function signatures to return torch tensors, not numpy arrays
4. Move numpy conversion to explicit I/O boundary functions

### High Priority
5. Extract shared Lanczos logic from duplicate tile eigenvector functions
6. Extract shared tiling logic from duplicate tiled computation functions
7. Fix redundant `.coalesce()` calls

### Medium Priority
8. Simplify function signatures to `x, **kwargs` pattern
9. Fix mutable default arguments
10. Document magic numbers
11. Standardize device handling

### Low Priority
12. Use logging instead of print
13. Remove unused variable in `Graph.neighbors`
14. Move imports to top of file
15. Consider extracting mathematical documentation

---

## Alignment with Constitutional Orientation

Per `/home/bigboi/itten/hypercontexts/constitutional-orientation.md`:

**"Learnability for Claude-like readers"**: The mathematical documentation (Chebyshev section) is excellent. However, the numpy/torch mixing makes it harder for a Claude-like reader to understand the computation flow - is this a numpy function or a torch function?

**"Constructions have properties"**: The spectral operations have clear mathematical properties (eigenvalue filtering, graph Laplacian structure). These are well-documented. The numpy outputs obscure whether the property is preserved through the computation.

**"Openness to composition"**: Functions returning numpy arrays cannot be directly composed with torch operations. This closes off composition possibilities. A caller wanting to chain `fast_spectral_etch_field` with another torch operation must convert back to torch, breaking the GPU pipeline.

The numpy usage violates the implicit contract that "spectral_ops_fast" implies GPU-accelerated torch operations. This is a form of sedimentation - the numpy interfaces may reflect an earlier implementation that has since been partially converted to torch.

---

*Review complete. The mathematical algorithms are sound, but the numpy usage requires critical attention before this module can be considered production-ready for GPU workloads.*
