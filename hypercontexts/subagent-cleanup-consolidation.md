# Hypercontext: Cleanup & Consolidation

## Mission
Execute the cleanup actions identified in the audit. Fix duplication, consolidate outputs, delete orphans.

## Priority 5 Actions (URGENT)

### 1. Unify height_to_normals()

**Problem**: Two implementations with DIFFERENT output ranges:
- `texture_synth/render/pbr.py:14` returns [-1, 1]
- `texture_synth_v2/normals.py:11` returns [0, 1] for RGB encoding

**Action**:
1. Keep `texture_synth_v2/normals.py` as canonical (RGB encoding is standard)
2. Update `texture_synth/render/pbr.py` to import from there:
```python
from texture_synth_v2.normals import height_to_normals
```
3. Or copy the implementation and ensure consistent output

## Priority 4 Actions

### 2. Consolidate pattern generators

**Problem**: Three locations with amongus/checkerboard:
- `texture_synth_v2/patterns.py` - vectorized, canonical
- `texture_synth/inputs/carriers.py` - OOP wrapper with loops
- `texture_editor/core.py` - inline

**Action**:
1. Update `texture_synth/inputs/carriers.py` to import from `texture_synth_v2/patterns.py`:
```python
from texture_synth_v2.patterns import generate_amongus, generate_checkerboard
```
2. Have AmongusCarrier._generate_base() call the imported function

### 3. Consolidate 3D egg renderers

**Problem**: `texture_synth_v2/render_egg.py` and `texture_editor/render.py` both have render_3d_egg()

**Action**: Keep `texture_synth_v2/render_egg.py` as canonical. It has more features (PBR variants).

## Priority 3 Actions

### 4. Delete deprecated file
```bash
rm spectral_ops_DEPRECATED.py
```

### 5. Delete orphaned PNG files from project root
```bash
rm -f fast_spectral_etch_demo.png
rm -f texture_egg.png
rm -f theta_dependent_test.png
rm -f filter_selectivity_test.png
rm -f theta_comparison.png
rm -f spectral_etch_angles.png
rm -f combined_nodal_lines.png
rm -f test_dichromatic_render.png
rm -f spectral_comparison_visual.png
```

### 6. Consolidate output directories

Move all scattered outputs to unified structure:
```bash
mkdir -p outputs/demos outputs/tests

# Move demo outputs
mv texture_synth_outputs/* outputs/demos/ 2>/dev/null || true
mv texture_synth_outputs_v2/* outputs/demos/ 2>/dev/null || true
mv demo_outputs/* outputs/demos/ 2>/dev/null || true

# Move test outputs
mv cli_test/* outputs/tests/ 2>/dev/null || true
mv cli_test2/* outputs/tests/ 2>/dev/null || true
mv outputs_pbr_test/* outputs/tests/ 2>/dev/null || true
mv test_trace_output/* outputs/tests/ 2>/dev/null || true

# Remove empty directories
rmdir texture_synth_outputs texture_synth_outputs_v2 demo_outputs cli_test cli_test2 outputs_pbr_test test_trace_output 2>/dev/null || true
```

### 7. Update .gitignore
Add:
```
outputs/
*.png
!docs/*.png
```

## Code Changes Required

### texture_synth/render/pbr.py
Change the height_to_normals implementation to match texture_synth_v2/normals.py output range [0,1].

### texture_synth/inputs/carriers.py
Import and use functions from texture_synth_v2/patterns.py instead of reimplementing.

### Demo/test files
Update hardcoded paths to use outputs/ directory:
- `test_chebyshev_final.py`
- `texture_synth_v2/demo_unified.py`
- `texture_synth_v2/spectral_etch.py`
- `texture_synth_v2/synthesize.py`
- `texture_synth_v2/demo_nodal_lines.py`

## Verification

After cleanup:
1. Run integration test to ensure nothing broke
2. Check that `outputs/` is the only output directory
3. Verify no duplicate implementations remain
4. Confirm .gitignore excludes outputs

## Deliverables
1. Delete orphaned files
2. Consolidate output directories
3. Fix height_to_normals duplication
4. Fix pattern generator duplication
5. Update .gitignore
6. Report what was changed
