# Spectral Shader Guiding Constraints

## Core Definitions

### 1. THICKEN Shader
- Grabs SEGMENTS (actual graph content, not empty space)
- Copies and translates TINY amounts
- Result: BOLDS existing lines in colinear/parallel way
- Does NOT delete, does NOT cover unrelated content

### 2. DROP SHADOW Shader
- Grabs SEGMENTS (actual graph content, not empty space)
- Copies + translates ENOUGH that shadows don't cover their source
- Rotates color to add ORTHOGONAL texture (different from source)
- Result: Adds new texture, does NOT bold, does NOT erase

### 3. Both Effects
- Use partial spectral transform to EXAGGERATE by CONTRAST and CONJUNCTION
- Are NOT here to delete image contents
- Operate on EXISTING CONTENT, not empty space

## Critical Guiding Constraint

**Regressive application (N times) should yield EXTREME CARICATURE of starting image-as-graph**

NOT:
- Image where all graph features are gone
- Image erased to solid color
- Graph features fused together into blob

This is NOT an if/then or switch case - it's a constraint that tells us what counts as progress.

## The Gating Metaphor

Gates in this codebase = **SwiGLU-style combination**:
```
output = signal_A * gate(signal_B)
```

Key properties:
- Combines abstract data types via MULTIPLIES and ADDS
- Results in NONLINEAR TRUNCATIONS (things that don't match both get zeroed)
- NOT binary thresholding
- NOT if/then branching

This is a RESIDUAL NET:
- Should be simple enough to NOT need learned parameters
- Implements OOD-generalizing texture effects
- Gate = multiplicative combination, not selection predicate

## Bug Identified: v7b Spectral Selection

### What went wrong:
`soft_spectral_select()` grabs regions based on Fiedler/gate affinity BUT:
- Does NOT check if region has ACTUAL CONTENT
- Selects all-white bounding boxes (no contrast = no content)
- Color-rotates empty space → creates solid teal rectangles
- These rectangles ERASE target content

### The original v6 constraint (correct):
```python
contours = np.abs(img_norm) > 1.0  # Must be ACTUAL CONTENT
low_gate = gate < gate_threshold
eligible = contours & low_gate      # BOTH constraints via AND (multiply)
```

### The fix needed:
Selection must verify CONTENT EXISTS:
- Sample pixel must have contrast (be part of actual drawing)
- Empty/flat regions must not be selectable regardless of spectral match
- Affinity should be GATED by content existence (multiply, not threshold)

## Implementation Pattern

```python
# WRONG: Select by spectral affinity alone
affinity = spectral_match(sample, target)
selection = affinity > threshold

# RIGHT: Gate spectral affinity BY content existence
content_mask = has_actual_content(sample)  # contrast, edges, etc
affinity = spectral_match(sample, target)
selection = affinity * content_mask  # Multiplicative gating
```

The multiplication ensures: no content → no selection, regardless of spectral match.
