# Hypercontext: Delete Dead Code and Deprecated Files

## Mission
Remove dead code, deprecated files, and merge duplicates.

## Files to Delete

### 1. `/home/bigboi/itten/main.py` - Dead placeholder
```python
# Current content is just:
# print("Hello from itten!")
```
**Action**: Delete file

### 2. Merge demo_unified.py into cli.py

`texture_synth_v2/demo_unified.py` duplicates 90% of `texture_synth_v2/cli.py`.

**Action**:
1. Move any unique functionality from demo_unified.py to cli.py
2. Delete demo_unified.py
3. Update any imports

### 3. Deprecate V1 Files

These are superseded by V2:
- `/home/bigboi/itten/lattice_extrusion.py` → use `lattice_extrusion_v2/`
- `/home/bigboi/itten/texture_synthesis.py` → use `texture_synth_v2/`

**Action**: Add deprecation warning at top of each:
```python
import warnings
warnings.warn(
    "This module is deprecated. Use lattice_extrusion_v2 instead.",
    DeprecationWarning,
    stacklevel=2
)
```

Or delete if nothing imports them.

### 4. Clean up infinite_bisect.py

After moving Lanczos to spectral_ops_fast, remove redundant implementations:
- `LocalLaplacian` class (lines 107-149) - use `local_laplacian_matvec`
- `local_fiedler_krylov()` (lines 180-280) - use `local_fiedler_vector`
- `bisect()` and `count_cut_edges()` - use `local_bisect`

Keep only `InfiniteGraph` class for testing purposes.

## Code to Inline-Delete

### render_egg.py Tangent Frame Duplication

The pattern:
```python
tangent = np.array([1.0, 0.0, 0.0])
bitangent = np.cross(base_normal, tangent)
```

Appears at lines 381-392, 562-570, 792-797, 943-948.

**Action**: After creating lighting.py, replace with:
```python
from texture_synth.render.lighting import compute_tbn_frame
tangent, bitangent, normal = compute_tbn_frame(base_normal)
```

## Update __init__.py Files

After deletions, ensure package __init__.py files don't try to import deleted modules.

Check:
- `texture_synth_v2/__init__.py`
- `texture_synth/__init__.py`

## Verification

After cleanup:
```bash
# Check no broken imports
python -c "import texture_synth"
python -c "import texture_synth_v2"
python -c "import lattice_extrusion_v2"
python -c "import bisect_viz"
python -c "import lattice_mosaic"
python -c "import texture_editor"

# Run integration test
python texture_tui.py -c "icosahedron amonguswrapped" -o cleanup_test/
```

## Summary of Deletions

| File | Action | Reason |
|------|--------|--------|
| `main.py` | DELETE | Dead code |
| `demo_unified.py` | MERGE → cli.py, then DELETE | 90% duplicate |
| `lattice_extrusion.py` | DEPRECATE or DELETE | V1, superseded |
| `texture_synthesis.py` | DEPRECATE or DELETE | V1, superseded |
| `infinite_bisect.py` | REFACTOR (remove duplicates) | Keep InfiniteGraph only |

## Estimated Impact

- ~500 lines of dead/duplicate code removed
- 2 deprecated V1 files marked
- 1 duplicate file merged
- Cleaner import structure
