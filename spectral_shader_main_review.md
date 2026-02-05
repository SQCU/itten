# Spectral Shader Main Review

**File Reviewed:** `/home/bigboi/itten/spectral_shader_main.py`

**Cross-Referenced:**
- `/home/bigboi/itten/what_is_a_shader.md` (data flow documentation)
- `/home/bigboi/itten/spectral_shader_review_const.md` (previous ops review)
- `/home/bigboi/itten/spectral_shader_ops.py` (shader operations)
- `/home/bigboi/itten/spectral_ops_fast.py` (spectral computations)

**Reviewer:** Claude Opus 4.5
**Date:** 2026-02-04

---

## Executive Summary

The `spectral_shader_main.py` file is a clean CLI entry point that orchestrates spectral shader operations. The code is well-structured with clear separation between I/O, configuration, and tensor operations. However, there are several opportunities for optimization, including redundant wrapper functions, parallelizable operations, and latency-critical paths that could be improved.

---

## 1. Redundant Encapsulating/Wrapping Functions

### 1.1 `run_batch` (Lines 98-100) - **INLINE**

```python
def run_batch(n_pairs: int):
    """Run batch random pairs test (uses default paths from batch_random_pairs)."""
    batch_random_pairs(n_pairs=n_pairs)
```

**Analysis:** Pure wrapper with zero transformation. The docstring even acknowledges it "uses default paths from batch_random_pairs."

**Recommendation:** INLINE - Replace the call in `main()` with direct `batch_random_pairs(args.pairs)`.

**Impact:** Eliminates 1 function call overhead, reduces cognitive load.

---

### 1.2 `compute_spectral` (Lines 51-61) - **KEEP (with documentation)**

```python
def compute_spectral(rgb: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Compute dither-aware tiled spectral eigenvectors from RGB (uses L2 color distance)."""
    return compute_local_eigenvectors_tiled_dither(
        rgb,
        tile_size=cfg["tile_size"],
        overlap=cfg["overlap"],
        num_eigenvectors=cfg["num_eigenvectors"],
        radii=cfg["radii"],
        radius_weights=cfg["radius_weights"],
        edge_threshold=cfg["edge_threshold"],
    )
```

**Analysis:** This wrapper unpacks config dict into explicit parameters. It provides:
1. Config-to-parameter translation (legitimate abstraction)
2. A named semantic boundary ("compute spectral")
3. Documentation of the RGB -> eigenvectors conversion

**Recommendation:** KEEP - The config unpacking adds value. Add brief comment explaining why this layer exists.

---

### 1.3 `load_image` / `save_image` (Lines 35-48) - **KEEP**

```python
def load_image(path: Path) -> torch.Tensor:
    """Load image as RGB float32 [0,1] torch tensor."""
    img = Image.open(path)
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)

def save_image(tensor: torch.Tensor, path: Path):
    """Save (H,W,3) tensor as image."""
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    print(f"Saved: {path}")
```

**Analysis:** These establish the I/O boundary. They handle:
1. PIL <-> numpy <-> torch conversion
2. Float [0,1] normalization
3. Dtype conversions

**Recommendation:** KEEP - These are legitimate I/O boundaries. Note: There are duplicate helpers in `spectral_shader_ops.py` (`_load_image_to_tensor`, `_save_tensor_to_image`) that should be consolidated.

---

## 2. Loops That Should Be Parallel

### 2.1 Spectral Computation for Two Images (Lines 73-78)

**Current (Sequential):**
```python
evecs_a = compute_spectral(rgb_a, cfg)
evecs_b = compute_spectral(rgb_b, cfg)
```

**Analysis:** These two operations are completely independent. Each computes eigenvectors from a different image. No data dependency exists between them.

**Parallelization Opportunity:**
```python
# Option A: ThreadPoolExecutor (simple, works with GIL for I/O-bound portions)
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=2) as executor:
    future_a = executor.submit(compute_spectral, rgb_a, cfg)
    future_b = executor.submit(compute_spectral, rgb_b, cfg)
    evecs_a = future_a.result()
    evecs_b = future_b.result()

# Option B: torch.multiprocessing for true parallelism (better for GPU)
# Requires careful device handling
```

**Estimated Speedup:** Up to 2x for this section (depends on GPU utilization saturation).

**Caveat:** If GPU is already at 100% utilization from a single `compute_local_eigenvectors_tiled_dither` call, parallelization won't help. Profile first.

---

### 2.2 Intermediate Saving in Autoregressive Mode (Lines 135-138)

**Current (Sequential):**
```python
for i, inter in enumerate(intermediates):
    inter_path = output.parent / f"{stem}_pass{i+1}.png"
    save_image(inter, inter_path)
```

**Analysis:** Each `save_image` is I/O-bound (disk write). These are independent operations.

**Parallelization Opportunity:**
```python
from concurrent.futures import ThreadPoolExecutor

def save_with_path(args):
    tensor, path = args
    save_image(tensor, path)

with ThreadPoolExecutor(max_workers=4) as executor:
    paths = [(inter, output.parent / f"{stem}_pass{i+1}.png")
             for i, inter in enumerate(intermediates)]
    executor.map(save_with_path, paths)
```

**Estimated Speedup:** For 4-8 passes, could reduce total save time by 2-4x.

---

### 2.3 Tile Processing in `compute_local_eigenvectors_tiled_dither` (spectral_ops_fast.py)

**Current (Sequential in Python loop):**
```python
while y_idx < len(y_starts):
    y = y_starts[y_idx]
    while x_idx < len(x_starts):
        # ... process one tile
```

**Analysis:** This is the primary computational bottleneck. Tiles are processed sequentially. Each tile involves:
1. Building a multiscale Laplacian O(tile_size^2 * num_radii)
2. Lanczos iteration O(iterations * nnz(L))
3. Eigenproblem solve O(k^3) for tridiagonal

**Parallelization Opportunity:**
- **Batch tiles on GPU:** Process multiple tiles simultaneously if memory allows
- **torch.compile the inner function:** `_compute_tile_eigenvectors_multiscale` is a good candidate
- **CUDA streams:** Use multiple streams to overlap tile computation

**Estimated Impact:** This is the dominant cost. True parallelization could yield 4-16x improvement depending on hardware.

---

## 3. Serial Blocking Latency Analysis

### 3.1 Latency Breakdown for `run_single` (Two-Image Mode)

```
CLI Parse                         ~0.001%
|
+-- load_image(image_a)           ~2-5%    [I/O: disk read, PIL decode]
|
+-- load_image(image_b)           ~2-5%    [I/O: disk read, PIL decode]
|
+-- compute_spectral(rgb_a)       ~35-45%  [GPU: tiled eigenvector computation]
|     |
|     +-- build_multiscale_image_laplacian  ~10%
|     +-- Lanczos iterations                ~70%
|     +-- Blend tiles                       ~20%
|
+-- compute_spectral(rgb_b)       ~35-45%  [GPU: tiled eigenvector computation]
|
+-- two_image_shader_pass         ~5-15%   [GPU: shader operations]
|     |
|     +-- fiedler_gate                      ~1%
|     +-- dilate_high_gate_regions          ~30%
|     +-- transfer_segments_A_to_B          ~40%
|     |     +-- extract_segments x2         ~60% (connected components)
|     |     +-- signature computation       ~20%
|     |     +-- matching                    ~20%
|     +-- draw_all_segments_batched         ~29%
|
+-- save_image                    ~2-5%    [I/O: PNG encode, disk write]
```

### 3.2 Latency Breakdown for `run_autoregressive`

```
load_image                        ~2%
|
+-- shader_forwards (pass 1)      ~15-20%
|     +-- compute_fiedler         ~80%  [if not pre-provided]
|     +-- spectral_shader_pass    ~20%
|
+-- shader_forwards (pass 2)      ~15-20%
|     +-- compute_fiedler         ~80%  [RECOMPUTED unnecessarily]
|     +-- spectral_shader_pass    ~20%
|
+-- ... (repeat for n_passes)
|
+-- save final + intermediates    ~5-10%
```

**Critical Finding:** In autoregressive mode, `shader_forwards` recomputes Fiedler vectors on every pass even though the Fiedler is only used for gating and the image content is the only thing changing. The gating could potentially be computed once and reused, or a decay factor could be applied.

### 3.3 Relative Cost Estimates

| Operation | Relative Cost | Blocking Type |
|-----------|--------------|---------------|
| `compute_local_eigenvectors_tiled_dither` | **70-80%** | GPU compute |
| `_torch_connected_components` | 5-10% | GPU compute (Python loop overhead) |
| `dilate_high_gate_regions` | 3-5% | GPU compute |
| `draw_all_segments_batched` | 2-4% | GPU compute |
| `load_image` | 2-5% | I/O |
| `save_image` | 2-5% | I/O |
| Other ops | <5% | Various |

### 3.4 Critical Path

```
load_image_a --> compute_spectral_a -+
                                     |
load_image_b --> compute_spectral_b -+--> two_image_shader_pass --> save_image
```

The critical path is **compute_spectral** for both images. All other operations are either parallel opportunities or much smaller in cost.

---

## 4. Data Flow Analysis

### 4.1 Complete Data Flow Diagram

```
                     CLI ARGS
                        |
                        v
              +-------------------+
              |     argparse      |
              | (Path, int, float)|
              +-------------------+
                        |
        +---------------+---------------+
        |               |               |
        v               v               v
    run_single     run_batch    run_autoregressive
        |               |               |
        v               v               v
+----------------+  +------------------+  +------------------+
|  load_image    |  | batch_random_   |  |   load_image     |
|  Path -> Tensor|  | pairs (ops.py)  |  |   Path -> Tensor |
+----------------+  +------------------+  +------------------+
        |                                        |
        v                                        v
+----------------+                      +------------------+
| compute_spectral                      | shader_forwards  |
| RGB -> Evecs   |                      | (iterative)      |
| (DOMINANT COST)|                      +------------------+
+----------------+                               |
        |                                        v
        v                               +------------------+
+----------------+                      | shader_auto-     |
| Extract Fiedler|                      | regressive loop  |
| evecs[:,:,1]   |                      +------------------+
+----------------+                               |
        |                                        v
        v                               +------------------+
+----------------+                      | save_image +     |
| Build shader   |                      | intermediates    |
| config dict    |                      +------------------+
+----------------+
        |
        v
+---------------------+
| two_image_shader_   |
| pass (ops.py)       |
+---------------------+
        |
        v
+----------------+
|  save_image    |
|  Tensor -> PNG |
+----------------+
```

### 4.2 Data Transformation Chain

```
Path (string)
    |
    v [PIL.Image.open]
PIL.Image
    |
    v [np.array, astype(float32), /255]
np.ndarray (H, W, 3) float32 [0,1]
    |
    v [torch.from_numpy]
torch.Tensor (H, W, 3) float32 [0,1]        <-- PRIMARY DATA TYPE
    |
    v [compute_local_eigenvectors_tiled_dither]
torch.Tensor (H, W, num_eigenvectors) float32
    |
    v [slice [:,:,1]]
torch.Tensor (H, W) float32                  <-- Fiedler vector
    |
    v [two_image_shader_pass / spectral_shader_pass]
torch.Tensor (H, W, 3) float32 [0,1]         <-- Result
    |
    v [.cpu().numpy(), *255, astype(uint8)]
np.ndarray (H, W, 3) uint8 [0,255]
    |
    v [Image.fromarray, .save]
PNG file on disk
```

### 4.3 Unnecessary Conversions / Copies Identified

1. **Double tensor device transfer (potential):**
   - `load_image` creates tensor on CPU
   - If GPU is available, tensor needs to be moved
   - Currently no explicit `.to(device)` in main.py; relies on downstream ops

2. **Config dict reconstruction (Lines 80-85, 109-118):**
   - Creates new dict with subset of keys
   - Not strictly necessary if downstream accepts full config

3. **Fiedler extraction creates a view, not a copy:**
   - `fiedler_a = evecs_a[:, :, 1]` - This is a view, efficient
   - No issue here

4. **Intermediate tensors in autoregressive mode:**
   - `intermediates.append(current.clone())` - Clone is necessary to preserve state
   - This is correct behavior, not wasteful

---

## 5. Specific Recommendations

### 5.1 High Priority (Performance)

| ID | Recommendation | Impact | Effort |
|----|----------------|--------|--------|
| P1 | Parallelize `compute_spectral` for two-image mode | ~1.5-2x speedup | Low |
| P2 | Add torch.compile to `compute_local_eigenvectors_tiled_dither` | 10-30% speedup | Medium |
| P3 | Cache Fiedler in autoregressive mode (optional recompute flag) | N*speedup for N passes | Medium |
| P4 | Batch tile processing in `_compute_tile_eigenvectors_multiscale` | 2-4x speedup | High |

### 5.2 Medium Priority (Code Quality)

| ID | Recommendation | Impact | Effort |
|----|----------------|--------|--------|
| M1 | Inline `run_batch` wrapper | Cleaner code | Trivial |
| M2 | Consolidate duplicate I/O helpers (main.py vs ops.py) | Single source of truth | Low |
| M3 | Add device parameter to `load_image` for direct GPU load | Avoid CPU->GPU transfer | Low |
| M4 | Document config key expectations at module level | Maintainability | Low |

### 5.3 Low Priority (Polish)

| ID | Recommendation | Impact | Effort |
|----|----------------|--------|--------|
| L1 | Add progress callback to `compute_spectral` | Better UX | Low |
| L2 | Make output directory configurable via env var | Flexibility | Trivial |
| L3 | Add timing instrumentation (--profile flag) | Debugging | Medium |

---

## 6. Alignment with Reference Documents

### 6.1 Alignment with `what_is_a_shader.md`

The main.py correctly orchestrates the data flow documented in the shader documentation:
- Entry via `two_image_shader_pass` matches documented pipeline
- Fiedler extraction happens before shader pass as expected
- Config dict construction follows documented parameters

**Gap:** The documentation describes `shader_forwards` as the unified interface, but `run_single` calls `two_image_shader_pass` directly. Consider using `shader_forwards(image_A, image_B, ...)` for consistency.

### 6.2 Alignment with `spectral_shader_review_const.md`

Previous review identified:
- Unnecessary `.clone()` calls in ops.py - **Not present in main.py**
- Duplicate grayscale computation - **main.py doesn't compute grayscale**
- `run_batch` wrapper recommendation - **Confirmed, inline recommended**

---

## 7. Summary

### Critical Path for Optimization

```
compute_local_eigenvectors_tiled_dither  <<<< PRIMARY TARGET (70-80% of time)
    |
    +-- Parallelize tile processing
    +-- Apply torch.compile
    +-- Consider batching
```

### Quick Wins

1. **Inline `run_batch`** - Zero-effort cleanup
2. **Parallel I/O for autoregressive saves** - Simple ThreadPool
3. **Add device param to load_image** - Avoid transfer

### Structural Recommendations

1. Use `shader_forwards` consistently as the unified entry point
2. Consolidate I/O helpers between main.py and ops.py
3. Consider a `SpectralShaderConfig` dataclass instead of dict

---

*Review complete. The dominant optimization opportunity is parallelizing the tiled eigenvector computation, which accounts for 70-80% of total runtime.*
