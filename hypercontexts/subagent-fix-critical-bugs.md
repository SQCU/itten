# Hypercontext: Fix Critical Bugs (Priority 1)

## Mission
Fix the 4 critical redundancies that cause actual bugs.

## 1. Spherical UV Axis Mismatch (Peril 9/10)

**Problem**: Two different axis conventions:
```python
# primitives.py (Y-up):
theta = atan2(z, x), phi = asin(y)

# uv_mapping.py (Z-up):
theta = atan2(y, x), phi = acos(z)
```

**Fix**: Standardize on Y-up convention (matches render_egg.py usage).

Update `/home/bigboi/itten/texture_synth/render/uv_mapping.py:270` `unwrap_spherical()`:
```python
def unwrap_spherical(vertices: np.ndarray) -> np.ndarray:
    """Spherical UV mapping with Y-up convention."""
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    unit = vertices / np.maximum(norms, 1e-10)

    # Y-up convention (matches primitives.py)
    theta = np.arctan2(unit[:, 2], unit[:, 0])  # Changed from atan2(y, x)
    phi = np.arcsin(np.clip(unit[:, 1], -1, 1))  # Changed from acos(z)

    u = (theta / (2 * np.pi) + 0.5) % 1.0
    v = phi / np.pi + 0.5

    return np.column_stack([u, v])
```

## 2. height_to_normals ×3 (Peril 9/10)

**Problem**: Three implementations with different behaviors:
- `texture_synth/render/pbr.py:14`
- `texture_synth_v2/normals.py:11` (CANONICAL - has wrap option)
- `texture_editor/core.py:204`

**Fix**: Make pbr.py and texture_editor/core.py import from normals.py.

In `/home/bigboi/itten/texture_synth/render/pbr.py`, replace the function with:
```python
# Import canonical implementation
from texture_synth_v2.normals import height_to_normals
```

In `/home/bigboi/itten/texture_editor/core.py`, replace `heightfield_to_normalmap` with:
```python
from texture_synth_v2.normals import height_to_normals as heightfield_to_normalmap
```

## 3. Lanczos ×3 (Peril 9/10)

**Problem**: Three Lanczos implementations:
- `spectral_ops_fast.py:116` - GPU, CANONICAL
- `texture_synth_v2/spectral_etch.py:88` `_lanczos_eigenvectors()` - DUPLICATE
- `infinite_bisect.py:180` `local_fiedler_krylov()` - CPU duplicate

**Fix for spectral_etch.py**:
The `_lanczos_eigenvectors()` function should be replaced with a call to spectral_ops_fast.
But it needs multi-eigenvector support. Add to spectral_ops_fast.py if not present, then import.

Check if `spectral_ops_fast.py` has `lanczos_k_eigenvectors()`. If not, the existing `_lanczos_eigenvectors` in spectral_etch.py should be MOVED to spectral_ops_fast.py and imported.

**Fix for infinite_bisect.py**:
Replace `local_fiedler_krylov()` with import from spectral_ops_fast:
```python
from spectral_ops_fast import local_fiedler_vector

def local_fiedler_krylov(graph, seed_nodes, hops=2, iterations=30):
    """Wrapper for backward compatibility."""
    fiedler_dict, lambda2 = local_fiedler_vector(
        graph, list(seed_nodes), iterations, hops
    )
    return fiedler_dict, lambda2
```

## 4. synthesize_texture ×3 (Peril 9/10)

**Problem**: Three implementations:
- `texture_synth_v2/synthesize.py:225` - CANONICAL
- `texture_synth/synthesis/core.py:19` - polynomial approximation
- `texture_synth_v2/demo_unified.py:233` - inline fallback

**Fix**:
1. Delete the inline fallback in demo_unified.py, require proper import
2. In synthesis/core.py, either:
   a. Import from texture_synth_v2/synthesize.py, or
   b. Keep as alternative (polynomial) with different name

In `/home/bigboi/itten/texture_synth_v2/demo_unified.py`, find and remove the fallback `synthesize_texture` function. Replace with proper import.

## Verification
After fixes, run:
```python
# Test imports work
from texture_synth_v2.normals import height_to_normals
from texture_synth_v2.synthesize import synthesize_texture
from spectral_ops_fast import local_fiedler_vector

# Test UV consistency
from texture_synth.geometry.primitives import spherical_uv
from texture_synth.render.uv_mapping import unwrap_spherical
# These should give same results for same vertices
```
