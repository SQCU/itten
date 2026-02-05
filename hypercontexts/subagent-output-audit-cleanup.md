# Hypercontext: Output Target Audit & Cleanup

## Mission
1. Find ALL places where renders are written to disk (scattered output targets)
2. Identify root cause of "bespoke output targets" - why different rendering approaches exist
3. Consolidate all output files into unified output target
4. Report on code duplication with priority/peril analysis

## Phase 1: Audit Output Targets

### Find all PNG write locations
```bash
grep -rn "\.save\(" --include="*.py" .
grep -rn "Image.fromarray" --include="*.py" .
grep -rn "imsave\|imwrite" --include="*.py" .
grep -rn "\.png" --include="*.py" .
```

### Find all output directories referenced
```bash
grep -rn "output" --include="*.py" . | grep -i "dir\|path\|folder"
find . -name "*.png" -type f
```

### Categorize each output location
For each found location, document:
- File path and line number
- What is being rendered
- Where it's being saved (hardcoded path? parameter?)
- What renderer is being used
- Is it using the canonical RenderTrace or bespoke?

## Phase 2: Root Cause Analysis

### Questions to answer:
1. How many distinct "to PNG" implementations exist?
2. How many distinct rendering pipelines exist?
3. Why were they created separately instead of reusing?
4. Which ones predate the unified system?
5. Which ones duplicate logic from texture_synth/render/?

### Expected root causes:
- **Historical**: Files created before unified system existed
- **Test files**: Quick tests with inline rendering
- **Demo files**: Standalone demos with hardcoded outputs
- **Fragmentation**: Different developers/sessions creating parallel solutions

## Phase 3: Consolidation

### Unified output target
All renders should go to: `outputs/` (configurable via RenderTrace)

### For each scattered output:
1. If it's a test file output → move to `outputs/tests/`
2. If it's a demo output → move to `outputs/demos/`
3. If it's development artifact → delete or move to `outputs/dev/`
4. Update the code to use RenderTrace instead of bespoke saving

### Files to potentially clean up:
- Root directory PNGs (bisect.png, texture_egg.png, etc.)
- Scattered output directories (texture_synth_outputs/, texture_synth_outputs_v2/, etc.)
- Test output directories (outputs_pbr_test/, test_trace_output/, etc.)

## Phase 4: Duplication Report

### For each duplicated pattern, rate:

**Priority** (1-5, 5 = fix immediately):
- 5: Active bug risk, will cause confusion
- 4: Maintenance burden, multiple places to update
- 3: Code smell, should fix but not urgent
- 2: Minor redundancy
- 1: Acceptable duplication (e.g., similar but distinct use cases)

**Peril** (1-5, 5 = high risk):
- 5: Different implementations give different results silently
- 4: One gets updated, others become stale
- 3: Wastes developer time understanding duplicates
- 2: Makes codebase harder to navigate
- 1: Cosmetic issue only

### Categories of duplication to check:
1. **Rendering pipelines**: How many ways to render a mesh/texture?
2. **PNG saving**: How many Image.save() patterns?
3. **Normal map generation**: How many height_to_normals()?
4. **Laplacian building**: How many build_laplacian()?
5. **Eigenvector computation**: How many Lanczos implementations?
6. **Pattern generation**: How many amongus/checkerboard generators?

## Deliverables

1. **Audit table**: Every output location with file:line, what, where, which renderer
2. **Root cause report**: Why fragmentation occurred
3. **Consolidation actions**: What was moved/deleted/updated
4. **Duplication report**: Table of duplicates with priority/peril scores
5. **Recommendations**: What refactoring is still needed

## File Structure After Cleanup

```
outputs/                    ← ALL renders go here
├── demos/                  ← Demo outputs
├── tests/                  ← Test outputs
└── sessions/               ← TUI session outputs
    └── session_YYYYMMDD_HHMMSS/
        ├── step_0000_*.png
        ├── step_0001_*.png
        └── manifest.json
```

## Code Changes Needed

For each bespoke save location, change from:
```python
# BAD: bespoke
Image.fromarray(img).save("my_output.png")
```

To:
```python
# GOOD: unified
from texture_synth.render import RenderTrace
trace = RenderTrace("outputs/demos")
trace.save_raw(img, "my_output")
```
