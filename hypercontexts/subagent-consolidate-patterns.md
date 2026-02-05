# Hypercontext: Consolidate Pattern Generators (Priority 2-3)

## Mission
Fix pattern generator duplication - make all code import from canonical sources.

## Canonical Sources
- `texture_synth_v2/patterns.py` - amongus, checkerboard, gradients
- `texture_synth/inputs/svg_parser.py` - dragon curve, SVG parsing

## Fixes Required

### 1. texture_editor/core.py:252 - Loop-based Amongus (Peril 8/10)

**Current** (SLOW, loop-based):
```python
def generate_demo_bitmap(pattern: str = "amongus", size: int = 64):
    if pattern == "amongus":
        # ... nested for loops ...
```

**Fix**: Import from patterns.py
```python
from texture_synth_v2.patterns import generate_amongus, generate_checkerboard

def generate_demo_bitmap(pattern: str = "amongus", size: int = 64):
    if pattern == "amongus":
        return generate_amongus(size)
    elif pattern == "checkerboard":
        return generate_checkerboard(size, tile_size=size // 8)
    # ... other patterns ...
```

### 2. texture_synth/inputs/operands.py:12 - Loop-based Checkerboard

**Current** (loop-based):
```python
class CheckerboardOperand(OperandInput):
    def _generate(self) -> np.ndarray:
        pattern = np.zeros((self.size, self.size))
        for i in range(self.cells):
            for j in range(self.cells):
                if (i + j) % 2 == 0:
                    # ...
```

**Fix**: Import from patterns.py
```python
from texture_synth_v2.patterns import generate_checkerboard

class CheckerboardOperand(OperandInput):
    def _generate(self) -> np.ndarray:
        return generate_checkerboard(self.size, tile_size=self.size // self.cells)
```

### 3. texture_synth/inputs/carriers.py:106-166 - Duplicate Dragon Curve

**Current**: DragonCurveCarrier has its own L-system implementation

**Fix**: Import from svg_parser.py
```python
from .svg_parser import dragon_curve, rasterize_curve

class DragonCurveCarrier(CarrierInput):
    def _generate_base(self) -> np.ndarray:
        points = dragon_curve(self.iterations)
        return rasterize_curve(points, self.size, self.line_width)
```

Delete the duplicate `_dragon_curve_points()` and `_rasterize_curve()` methods.

### 4. render_3d_egg Duplicate

**Files**:
- `texture_synth_v2/render_egg.py:66` - CANONICAL (full-featured)
- `texture_editor/render.py:145` - DUPLICATE

**Fix**: Make texture_editor import from render_egg.py
```python
from texture_synth_v2.render_egg import render_3d_egg
```

Or if texture_editor needs a simpler version, have it call the canonical one with simplified parameters.

### 5. demo_unified.py Inline Fallback

**File**: `texture_synth_v2/demo_unified.py:233`

**Fix**: Delete the inline `synthesize_texture` fallback function. Use proper import:
```python
from texture_synth_v2.synthesize import synthesize_texture
```

If the fallback was for when synthesis failed, add proper error handling instead.

## Verification

After fixes, verify:
```python
# All patterns work
from texture_synth_v2.patterns import generate_amongus, generate_checkerboard
from texture_synth.inputs import AmongusCarrier, CheckerboardCarrier, DragonCurveCarrier

carrier = AmongusCarrier(64)
img = carrier.render()
assert img.shape == (64, 64)

dragon = DragonCurveCarrier(64, iterations=10)
img = dragon.render()
assert img.shape == (64, 64)

# Run demos
from texture_editor.core import generate_demo_bitmap
bitmap = generate_demo_bitmap("amongus", 64)
assert bitmap.shape == (64, 64)
```
