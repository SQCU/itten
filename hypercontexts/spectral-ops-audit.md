# Spectral Ops Fast Audit

## File Location
`/home/bigboi/itten/spectral_ops_fast.py` (~3189 lines)

## Overview
This is a comprehensive library of GPU-accelerated spectral graph operations built on `torch.sparse`. It contains:
- Graph data structures and construction utilities
- Lanczos-based eigenvector computation
- Heat diffusion operations
- Chebyshev polynomial filtering
- Iterative spectral transforms (the key function for v7_b)
- Divergent spectral transforms for image processing

---

## All Available Functions/Kernels

### 1. Graph Data Structures

#### `Graph` (dataclass)
**Location:** Lines 27-466
**Purpose:** Canonical graph type for all spectral operations

**Attributes:**
- `adjacency: torch.Tensor` - Sparse (n x n) adjacency matrix
- `coords: Optional[torch.Tensor]` - (n, d) coordinate tensor

**Methods:**
- `num_nodes` -> int
- `device` -> torch.device
- `neighbors(node: int)` -> List[int]
- `degree(node: int)` -> int
- `seed_nodes()` -> List[int]
- `coord_of(node: int)` -> Optional[Tuple[float, ...]]
- `edge_weight(u: int, v: int)` -> float
- `laplacian(normalized: bool = False)` -> torch.Tensor (sparse Laplacian)

**Class Methods:**
- `from_image(image, connectivity=4, edge_threshold=0.1, device=None)` -> Graph
- `from_lattice(lattice, device=None)` -> Graph
- `from_graphview(graph, seed_nodes=None, max_hops=10, device=None)` -> Graph

---

### 2. Graph Traversal Utilities

#### `expand_neighborhood(graph, seed_nodes, hops)` -> Set[int]
**Location:** Lines 468-495
**Purpose:** Expand seed nodes by given number of hops
**Note:** Uses while loops (allowed for graph traversal, not hot path)

#### `build_sparse_laplacian(graph, active_nodes)` -> Tuple[torch.Tensor, List[int], Dict[int, int]]
**Location:** Lines 498-557
**Purpose:** Build sparse Laplacian as torch.sparse_coo_tensor
**Returns:** (L_sparse, node_list, node_to_idx)

---

### 3. Lanczos Eigenvector Computation

#### `lanczos_fiedler_gpu(L, num_iterations=30, tol=1e-10)` -> Tuple[torch.Tensor, float]
**Location:** Lines 560-669
**Purpose:** GPU Lanczos iteration to compute Fiedler vector
**Returns:** (fiedler_vector, lambda2)
**Complexity:** O(k * n * iterations)

#### `lanczos_k_eigenvectors(L, num_eigenvectors, num_iterations=50, tol=1e-10)` -> Tuple[np.ndarray, np.ndarray]
**Location:** Lines 672-778
**Purpose:** Compute first k non-trivial eigenvectors using GPU Lanczos
**Returns:** (eigenvectors: (n, k), eigenvalues: (k,))
**Complexity:** O(k * n * iterations) instead of O(n^3) for dense solver

#### `local_fiedler_vector(graph, seed_nodes, num_iterations=30, hop_expansion=2)` -> Tuple[Dict[int, float], float]
**Location:** Lines 781-819
**Purpose:** Compute approximate Fiedler vector for local neighborhood
**Returns:** (fiedler_dict, approximate_lambda2)

#### `local_expansion_estimate(graph, node, radius=2, num_lanczos=10)` -> float
**Location:** Lines 822-839
**Purpose:** Estimate local expansion (lambda2) around a single node

#### `expansion_map_batched(graph, nodes, radius=2, k=15)` -> Dict[int, float]
**Location:** Lines 842-950
**Purpose:** Compute lambda_2 for ALL nodes using batched tensor operations

---

### 4. Graph Bisection/Partitioning

#### `local_bisect(graph, seed_nodes, num_iterations=30, hop_expansion=2)` -> Tuple[Set[int], Set[int], int]
**Location:** Lines 953-1000
**Purpose:** Bisect local neighborhood by sign of Fiedler vector
**Returns:** (partition_A, partition_B, cut_edge_count)

---

### 5. Spectral Embeddings

#### `spectral_node_embedding(graph, seed_nodes, embedding_dim=8, hop_expansion=3, num_iterations=50)` -> Dict[int, np.ndarray]
**Location:** Lines 1003-1114
**Purpose:** Compute local spectral embedding of nodes
**Returns:** Dict mapping node_id -> embedding vector (embedding_dim,)

---

### 6. Heat Diffusion Operations

#### `heat_diffusion_sparse(L, signal, alpha=0.1, iterations=10)` -> torch.Tensor
**Location:** Lines 1121-1159
**Purpose:** Heat diffusion on sparse Laplacian: x_{t+1} = x_t - alpha * L @ x_t
**Args:**
- `L`: sparse Laplacian (n x n)
- `signal`: input signal (n,) or (n, c) for multi-channel
- `alpha`: diffusion step size
- `iterations`: number of diffusion steps (controls "rotation angle")
**Returns:** Diffused signal, same shape as input

#### `build_image_laplacian(height, width, weights=None, device=None)` -> torch.Tensor
**Location:** Lines 1162-1237
**Purpose:** Build sparse Laplacian for image grid with optional edge weights
**Returns:** Sparse Laplacian (H*W, H*W)

#### `build_weighted_image_laplacian(carrier, edge_threshold=0.1)` -> torch.Tensor
**Location:** Lines 1240-1305
**Purpose:** Build weighted Laplacian from carrier image
**Key:** Edge weights = exp(-|carrier_diff| / threshold)
**Returns:** Sparse Laplacian (H*W, H*W)

---

### 7. Dict-Based Interface (Infinite Graph Compatibility)

#### `local_laplacian_matvec(graph, x, active_nodes=None)` -> Dict[int, float]
**Location:** Lines 1312-1408
**Purpose:** Compute Lx where L is graph Laplacian using dict-based sparse vectors
**Use Case:** Infinite graphs where full Laplacian can't be built upfront

---

### 8. Tiled Local Eigenvector Computation

#### `compute_local_eigenvectors_tiled(image, tile_size=64, overlap=16, num_eigenvectors=4, edge_threshold=0.1, lanczos_iterations=30)` -> np.ndarray
**Location:** Lines 1415-1522
**Purpose:** Compute eigenvectors in overlapping tiles, blend results
**Returns:** eigenvector_images: (H, W, k) array of blended eigenvector fields
**Complexity:** O(n * k * iters) where k << n, suitable for arbitrarily large images
**Memory:** O(tile_size^2 * k) per tile, not O(H*W)

#### `_compute_tile_eigenvectors(tile, num_eigenvectors, edge_threshold, lanczos_iterations)` -> np.ndarray
**Location:** Lines 1525-1633
**Purpose:** Internal - compute eigenvectors for a single tile

#### `_create_blend_weights_1d(length, overlap)` -> np.ndarray
**Location:** Lines 1636-1659
**Purpose:** Internal - create 1D blending weights with smooth falloff

---

### 9. Spectral Navigation

#### `spectral_direction_to_goal(graph, current, goal, radius=2, num_lanczos=15)` -> int
**Location:** Lines 1662-1738
**Purpose:** Compute best neighbor to move toward goal using local spectral structure
**Returns:** best_neighbor node ID

---

### 10. Chebyshev Polynomial Filtering (KEY FOR v7_b)

#### `estimate_lambda_max(L, num_iterations=20, tol=1e-6)` -> float
**Location:** Lines 1877-1927
**Purpose:** Estimate largest eigenvalue of L using power iteration
**Returns:** Approximate lambda_max (with 1.05x safety margin)

#### `chebyshev_coefficients_gaussian(center, width, order, num_quadrature=200)` -> torch.Tensor
**Location:** Lines 1930-1981
**Purpose:** Compute Chebyshev coefficients for a Gaussian filter
**Returns:** Tensor of Chebyshev coefficients (order,)

#### `chebyshev_filter(L, signal, center, width, order=30, lambda_max=None)` -> torch.Tensor
**Location:** Lines 1984-2085
**Purpose:** Apply Chebyshev polynomial filter to signal on graph Laplacian
**Complexity:** O(order * nnz(L))
**Key Insight:** Approximates g(L) @ signal where g is Gaussian centered at 'center', WITHOUT computing eigenvectors

---

### 11. Iterative Spectral Transform (PRIMARY FUNCTION FOR v7_b)

#### `iterative_spectral_transform(L, signal, theta, num_steps=8, polynomial_order=30, lambda_max=None, sigma_factor=0.3)` -> torch.Tensor
**Location:** Lines 2088-2186
**Purpose:** Apply spectral transform by iterating small-angle Chebyshev filters

**Args:**
- `L: torch.Tensor` - Sparse Laplacian matrix (n x n)
- `signal: torch.Tensor` - Input signal (n,) or (n, c) for multi-channel
- `theta: float` - Target rotation angle in [0, 1]
  - theta=0: Emphasize low eigenvectors (Fiedler, coarse structure)
  - theta=1: Emphasize high eigenvectors (fine spectral detail)
- `num_steps: int` - Number of iterative steps (more = smoother transition)
- `polynomial_order: int` - Chebyshev polynomial degree per step
- `lambda_max: Optional[float]` - Largest eigenvalue (estimated if not provided)
- `sigma_factor: float` - Width of each filter as fraction of step size

**Returns:** Transformed signal emphasizing eigenvectors near theta * lambda_max

**Complexity:** O(num_steps * polynomial_order * nnz(L))
- For sparse graphs: O(num_steps * polynomial_order * n) = O(n log n) with proper tuning

**Algorithm:**
1. Divide [0, theta] into num_steps intervals
2. For each step, apply Chebyshev filter centered at that spectral location
3. Each filter incrementally shifts spectral emphasis
4. Normalize after each step to prevent amplitude decay/growth

---

### 12. Related Polynomial Filtering Functions

#### `polynomial_spectral_field(L, signal, theta, num_bands=5, polynomial_order=30, lambda_max=None)` -> torch.Tensor
**Location:** Lines 2189-2273
**Purpose:** Compute spectral field weighted by theta using polynomial filtering
**Key:** Polynomial approximation to explicit eigenvector formula

#### `spectral_projection_filter(L, signal, theta, num_probes=10, polynomial_order=30, lambda_max=None)` -> torch.Tensor
**Location:** Lines 2277-2359
**Purpose:** Approximate projection onto theta-weighted eigenvector subspace using random probes

#### `approximate_eigenvector_magnitude_field(L, theta, num_probes=30, polynomial_order=40, lambda_max=None, num_eigenvectors_approx=8)` -> torch.Tensor
**Location:** Lines 2362-2449
**Purpose:** Approximate weighted sum of eigenvector magnitudes using stochastic filtering
**Key:** Core technique for eigenvalue-free spectral computation (stochastic trace estimation)

#### `fast_spectral_etch_field(carrier, theta=0.5, num_probes=30, polynomial_order=40, edge_threshold=0.1)` -> np.ndarray
**Location:** Lines 2452-2509
**Purpose:** Main entry point for O(n log n) spectral field computation
**Returns:** Spectral field as 2D array (H, W), normalized to [0, 1]

---

### 13. Comparison/Testing

#### `compare_explicit_vs_iterative(carrier, theta=0.5, num_eigenvectors=8, polynomial_order=30, num_steps=8, edge_threshold=0.1)` -> Dict[str, np.ndarray]
**Location:** Lines 2512-2645
**Purpose:** Compare explicit eigenvector computation vs iterative polynomial approximation
**Returns:** Dict with explicit_field, iterative_field, difference, correlation

#### `test_tiled_eigenvectors(image_size=256, tile_size=64)`
**Location:** Lines 1745-1798
**Purpose:** Test tiled eigenvector computation on synthetic image

---

### 14. Divergent Spectral Transforms (Non-Linear Operations)

#### `eigenvector_phase_field(carrier, operand, theta=0.5, eigenpair=(0,1), edge_threshold=0.1)` -> np.ndarray
**Location:** Lines 2665-2749
**Purpose:** Create phase field from carrier eigenvector pairs, modulated by operand
**Non-Linear:** Uses arctan2 of eigenvector pairs to create spiral patterns

#### `spectral_contour_sdf(carrier, operand, theta=0.5, num_contours=5, edge_threshold=0.1)` -> np.ndarray
**Location:** Lines 2752-2833
**Purpose:** Generate signed distance field from spectral contours
**Non-Linear:** Distance to eigenvector iso-contours

#### `commute_time_distance_field(carrier, operand, theta=0.5, reference_mode='operand_max', edge_threshold=0.1)` -> np.ndarray
**Location:** Lines 2836-2930
**Purpose:** Compute commute time distance from reference points
**Non-Linear:** CT(i,j) = sum_k (phi_k(i) - phi_k(j))^2 / lambda_k

#### `spectral_warp(carrier, operand, theta=0.5, warp_strength=10.0, edge_threshold=0.1)` -> np.ndarray
**Location:** Lines 2933-3030
**Purpose:** Warp operand using carrier's eigenvector gradient flow
**Non-Linear:** Coordinate remapping via interpolation

#### `spectral_subdivision_blend(carrier, operand, theta=0.5, max_depth=4, min_size=16, edge_threshold=0.1)` -> np.ndarray
**Location:** Lines 3033-3188
**Purpose:** Recursively subdivide by carrier's Fiedler, fill with operand statistics
**Non-Linear:** Sign thresholding on Fiedler vectors creates hierarchical cells

---

## Functions Suitable for v7_b

### PRIMARY: `iterative_spectral_transform`
This is the main function v7_b should use. It provides:
- O(n log n) complexity instead of O(n^3) for explicit eigenvectors
- `theta` parameter maps directly to rotation angle concept
- No eigenvector computation needed
- Works with sparse Laplacians from `build_weighted_image_laplacian`

### Supporting Functions v7_b Needs:
1. `build_weighted_image_laplacian(carrier, edge_threshold)` - Build graph from carrier image
2. `estimate_lambda_max(L)` - Get spectral radius for scaling
3. `chebyshev_filter(L, signal, center, width, order, lambda_max)` - Direct spectral filtering
4. `heat_diffusion_sparse(L, signal, alpha, iterations)` - Simple diffusion alternative
5. `lanczos_fiedler_gpu(L)` - If explicit Fiedler is still needed somewhere

### Alternative High-Level Entry Points:
- `fast_spectral_etch_field(carrier, theta)` - Complete spectral field computation
- `polynomial_spectral_field(L, signal, theta)` - Band-weighted spectral field

---

## What v7_b Might Be Reimplementing

Based on the parent context, v7_b likely reimplements:

1. **Laplacian construction** - Use `build_weighted_image_laplacian` instead
2. **Eigenvector computation** - Use `lanczos_k_eigenvectors` or avoid entirely with `iterative_spectral_transform`
3. **Spectral filtering/rotation** - Use `iterative_spectral_transform` with theta parameter
4. **Distance field computation** - Use `commute_time_distance_field` or Chebyshev-based diffusion
5. **Segment extraction** - Use `local_bisect` or `spectral_subdivision_blend` patterns

---

## spectral_funcs_fast.py Status

**Does NOT exist.** The parent context mentions it as an "Architecture Target" for high-level composed functions.

Currently, `spectral_ops_fast.py` contains both:
- Low-level kernels (Lanczos, Chebyshev recurrence, sparse matvec)
- High-level composed functions (fast_spectral_etch_field, divergent transforms)

A future refactor might split these into:
- `spectral_ops_fast.py`: Pure kernels (Lanczos, Chebyshev, sparse operations)
- `spectral_funcs_fast.py`: Composed functions (etch field, transforms, image processing)

---

## Summary: Key Functions for v7_b Refactor

```python
# Primary function to replace explicit eigenvector computation:
from spectral_ops_fast import (
    iterative_spectral_transform,  # Main spectral transform
    build_weighted_image_laplacian, # Build Laplacian from carrier
    estimate_lambda_max,            # Get lambda_max for scaling
    chebyshev_filter,               # Direct band-pass filtering
    heat_diffusion_sparse,          # Simple diffusion alternative
    lanczos_fiedler_gpu,            # If explicit Fiedler still needed
    lanczos_k_eigenvectors,         # If explicit eigenvectors needed
    Graph,                          # Canonical graph type
    DEVICE,                         # GPU/CPU device
)
```

The key insight is that `iterative_spectral_transform` provides the theta-parameterized spectral rotation that v7_b needs, without O(n^3) eigenvector decomposition.
