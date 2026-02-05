# v6 Canonical Patterns - Reference for v7_b Refactor

## Summary Statistics

- **Total LOC**: 445 lines
- **Imports Section**: Lines 18-27 (10 lines)
- **Core Functions**: 12 functions
- **Comments/Docstrings**: Minimal, focused
- **Architecture**: Single-file application that imports low-level kernels from `spectral_ops_fast.py`

---

## Imports vs Local Implementation

### IMPORTED from spectral_ops_fast.py:
```python
from spectral_ops_fast import (
    build_weighted_image_laplacian, lanczos_k_eigenvectors, DEVICE
)
```

Only 3 things imported:
1. `build_weighted_image_laplacian` - constructs the graph Laplacian
2. `lanczos_k_eigenvectors` - computes eigenvectors via Lanczos iteration
3. `DEVICE` - torch device constant

### IMPORTED from standard libraries:
```python
import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage import sobel, binary_dilation, label, center_of_mass
from scipy.ndimage import distance_transform_edt
```

### IMPLEMENTED LOCALLY (application-specific):
- Fiedler gating (`compute_fiedler_gate`)
- Color dilation (`dilate_with_color_propagation`)
- Color transforms (`cyclic_color_transform`, `compute_shadow_color`, `compute_front_color`)
- Segment extraction (`extract_segments`)
- Segment drawing (`draw_segment_v6`, `compute_translation`)
- Pass orchestration (`single_pass_v6`, `stacked_shader_v6`)
- Demo/visualization (`run_v6_demo`)

---

## Core Architecture (12 Functions)

| Function | Lines | Purpose |
|----------|-------|---------|
| `compute_fiedler_gate` | 34-43 | Spectral bipartition via Fiedler vector |
| `dilate_with_color_propagation` | 50-132 | True color-preserving thickening |
| `cyclic_color_transform` | 139-186 | Nonlinear color transform (no fixed points) |
| `compute_shadow_color` | 188-198 | Shadow color with blue bias |
| `compute_front_color` | 201-212 | Front color with teal/cyan bias |
| `extract_segments` | 219-255 | Extract contour segments from low-gate regions |
| `compute_translation` | 258-269 | Translation distance from gradient |
| `draw_segment_v6` | 272-308 | Draw rotated segment with shadow |
| `single_pass_v6` | 314-363 | Single shader pass orchestration |
| `stacked_shader_v6` | 366-389 | Stack multiple passes with decay |
| `run_v6_demo` | 396-432 | Demo visualization |
| `__main__` | 435-444 | Entry point |

---

## CANONICAL PATTERNS

### 1. Fiedler/Gate Computation (Lines 34-43)

```python
def compute_fiedler_gate(img_gray, edge_threshold=0.12):
    """Fiedler vector bipartition."""
    H, W = img_gray.shape
    carrier = torch.tensor(img_gray, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier, edge_threshold=edge_threshold)
    eigenvectors, _ = lanczos_k_eigenvectors(L, num_eigenvectors=2, num_iterations=50)
    fiedler = eigenvectors[:, -1].reshape(H, W)
    fiedler = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-8)
    gate = 1.0 / (1.0 + np.exp(-(fiedler - 0.5) * 10.0))
    return fiedler, gate
```

**Key characteristics:**
- Takes grayscale image, returns (fiedler, gate)
- Uses `num_eigenvectors=2`, takes `[:, -1]` (second eigenvector = Fiedler)
- Normalizes Fiedler to [0,1]
- Gate is sigmoid of centered Fiedler: `1/(1+exp(-(f-0.5)*10))`
- ~10 lines total

### 2. Segment Extraction (Lines 219-255)

```python
def extract_segments(img_rgb, gate, min_pixels, max_segments, gate_threshold):
    """Extract contour segments from low-gate regions."""
    img_gray = img_rgb.mean(axis=2)
    H, W = img_gray.shape

    # LayerNorm style adaptive thresholding
    img_norm = (img_gray - img_gray.mean()) / (img_gray.std() + 1e-8)
    contours = np.abs(img_norm) > 1.0
    low_gate = gate < gate_threshold
    eligible = contours & low_gate

    if eligible.sum() < min_pixels * 3:
        gate_median = np.median(gate[contours]) if contours.any() else 0.5
        eligible = contours & (gate < gate_median)

    labeled, num_features = label(eligible)
    segments = []

    for seg_id in range(1, min(num_features + 1, max_segments * 2)):
        mask = labeled == seg_id
        ys, xs = np.where(mask)
        if len(ys) < min_pixels:
            continue

        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1

        segments.append({
            'mask': mask[y0:y1, x0:x1],
            'bbox': (y0, y1, x0, x1),
            'center': (center_of_mass(mask[y0:y1, x0:x1])[0] + y0,
                       center_of_mass(mask[y0:y1, x0:x1])[1] + x0),
        })
        if len(segments) >= max_segments:
            break

    return segments
```

**Key characteristics:**
- Contour detection via LayerNorm-style thresholding: `abs(normalized) > 1.0`
- Low-gate selection: `gate < gate_threshold`
- Fallback to median gate if too few pixels
- Uses `scipy.ndimage.label` for connected components
- Returns list of dicts with `mask`, `bbox`, `center`
- ~35 lines total

### 3. Color Transforms (Lines 139-212)

**Core cyclic transform:**
```python
def cyclic_color_transform(rgb, rotation_strength, contrast_strength):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    phase = rotation_strength * np.pi

    # Cyclic channels with 120-degree phase offsets
    new_r = 0.5 + 0.4 * np.sin(2 * np.pi * lum + phase + 0)
    new_g = 0.5 + 0.4 * np.sin(2 * np.pi * lum + phase + 2*np.pi/3)
    new_b = 0.5 + 0.4 * np.sin(2 * np.pi * lum + phase + 4*np.pi/3)

    blend = min(contrast_strength * 0.5, 0.9)
    out_r = r * (1 - blend) + new_r * blend
    # ... similar for g, b

    # Contrast boost around mean
    mean = (out_r + out_g + out_b) / 3
    contrast_factor = 1.0 + contrast_strength * 0.3
    out_r = mean + (out_r - mean) * contrast_factor
    # ...

    return np.clip(np.stack([out_r, out_g, out_b], axis=-1), 0, 1)
```

**Shadow color (blue bias):**
```python
def compute_shadow_color(rgb, effect_strength):
    transformed = cyclic_color_transform(rgb, 0.3 * effect_strength, 0.8 * effect_strength)
    transformed[:, :, 2] = transformed[:, :, 2] * 0.7 + 0.3  # boost blue
    transformed[:, :, 0] = transformed[:, :, 0] * 0.7        # reduce red
    return np.clip(transformed, 0, 1)
```

**Front color (teal/cyan bias):**
```python
def compute_front_color(rgb, effect_strength):
    transformed = cyclic_color_transform(rgb, 0.2 * effect_strength, 0.6 * effect_strength)
    transformed[:, :, 1] = transformed[:, :, 1] * 0.8 + 0.2   # boost green
    transformed[:, :, 2] = transformed[:, :, 2] * 0.8 + 0.15  # boost blue
    transformed[:, :, 0] = transformed[:, :, 0] * 0.6         # reduce red
    return np.clip(transformed, 0, 1)
```

### 4. Drawing Segments (Lines 272-308)

```python
def draw_segment_v6(output_rgb, segment, img_rgb, translation, effect_strength):
    """Draw rotated segment with cyclic color transform."""
    H, W, _ = output_rgb.shape

    mask = segment['mask']
    y0, y1, x0, x1 = segment['bbox']

    rgb_patch = img_rgb[y0:y1, x0:x1].copy()
    rotated_mask = np.rot90(mask, k=1)
    rotated_rgb = np.rot90(rgb_patch, k=1)
    rh, rw = rotated_mask.shape

    ty, tx = translation
    new_y0, new_x0 = y0 + ty, x0 + tx

    shadow_offset = int(7 * effect_strength)
    sy0, sx0 = new_y0 + shadow_offset, new_x0 + shadow_offset

    shadow_colors = compute_shadow_color(rotated_rgb, effect_strength)
    front_colors = compute_front_color(rotated_rgb, effect_strength)

    # Draw shadow first (nested loops)
    for dy in range(rh):
        for dx in range(rw):
            if rotated_mask[dy, dx]:
                py, px = sy0 + dy, sx0 + dx
                if 0 <= py < H and 0 <= px < W:
                    output_rgb[py, px] = shadow_colors[dy, dx]

    # Draw front (masks shadow)
    for dy in range(rh):
        for dx in range(rw):
            if rotated_mask[dy, dx]:
                py, px = new_y0 + dy, new_x0 + dx
                if 0 <= py < H and 0 <= px < W:
                    output_rgb[py, px] = front_colors[dy, dx]
```

**Key characteristics:**
- 90-degree rotation via `np.rot90(mask, k=1)`
- Shadow drawn first at offset, front drawn second (overwrites)
- Per-pixel loop with bounds checking
- Color transforms computed on the rotated patch
- ~35 lines total

---

## Configuration Pattern

```python
config = {
    'edge_threshold': 0.10,
    'max_thickness': 5,
    'min_segment_pixels': 10,
    'max_segments': 2000,
    'base_gate_threshold': 0.65,
    'base_translation': 30,
    'base_effect_strength': 1.0,
    'decay_rate': 0.75,
    'min_effect_strength': 0.05,
}
```

## Pass Orchestration Pattern

```python
def single_pass_v6(img_rgb, pass_number, config):
    # 1. Compute decayed effect_strength
    effect_strength = config['base_effect_strength'] * (config['decay_rate'] ** (pass_number - 1))
    effect_strength = max(effect_strength, config['min_effect_strength'])

    # 2. Compute gate threshold (also decays)
    gate_threshold = config['base_gate_threshold'] - (pass_number - 1) * 0.08
    gate_threshold = max(gate_threshold, 0.25)

    # 3. Compute Fiedler gate
    fiedler, gate = compute_fiedler_gate(img_gray, edge_threshold=config['edge_threshold'])

    # 4. Branch A: Color dilation
    output_rgb = dilate_with_color_propagation(...)

    # 5. Branch B: Segment extraction + rotation
    segments = extract_segments(...)
    for segment in segments:
        ty, tx = compute_translation(...)
        draw_segment_v6(...)

    return output_rgb, len(segments)
```

---

## Style Notes for v7_b Compliance

1. **Docstrings**: Single-line, minimal (e.g., `"""Fiedler vector bipartition."""`)
2. **Comments**: Section headers with `# ===` bars, inline comments sparse
3. **Function size**: Most functions 10-40 lines
4. **No classes**: Pure functions only
5. **NumPy-first**: Torch only for spectral ops (Laplacian, eigenvectors)
6. **Explicit loops**: Pixel-level drawing uses nested for loops, not vectorized
7. **Configuration**: Single dict passed through, not scattered globals
8. **Return values**: Functions return (result, count) tuples for stats

---

## Target for v7_b

v7_b should:
- Be ~445-500 LOC (matching v6)
- Import from `spectral_ops_fast.py` (and potentially `spectral_funcs_fast.py`)
- NOT reimplement low-level kernels
- Follow the same function decomposition pattern
- Use the same config dict pattern
- Have similar docstring/comment density
