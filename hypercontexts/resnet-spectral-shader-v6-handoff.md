# ResNet Spectral Shader v6 - Handoff Document

*Session Date: 2026-02-01*
*Status: Working implementation with novel visual effects*

---

## Summary

v6 implements a ResNet-like spectral compute shader with:
- **True color-preserving dilation** (copies source colors, no darkening)
- **Unified `effect_strength` parameter** (scales all sub-calculations)
- **Cyclic color transform without black/white fixed points**
- **Stacked passes with exponential decay**

Produces genuinely novel computer graphics effects not seen in conventional 2D painting software.

---

## Key Innovations in v6

### 1. True Color-Preserving Dilation
```python
# OLD (v5): Darkened pixels during thickening
output[thicken_mask] *= 0.15  # Crushes colors to black

# NEW (v6): Copies source color to dilated pixels
# Find nearest original contour pixel, copy its color
output[new_pixel] = img_rgb[nearest_contour_pixel]
```

### 2. Cyclic Color Transform (No Fixed Points)
```python
# Sinusoidal mapping: even black/white evolve
new_r = 0.5 + 0.4 * sin(2π * luminance + phase + 0)
new_g = 0.5 + 0.4 * sin(2π * luminance + phase + 2π/3)
new_b = 0.5 + 0.4 * sin(2π * luminance + phase + 4π/3)
```
- Pure black (0,0,0) → maps to a color
- Pure white (1,1,1) → maps to a color
- Colors continue evolving at extremes

### 3. Unified Effect Strength
Single parameter scales everything:
- Dilation radius
- Translation distance
- Shadow offset
- Color rotation intensity
- Contrast boost

```python
effect_strength = base * (decay_rate ** (pass - 1))
# Pass 1: 1.000
# Pass 2: 0.750
# Pass 3: 0.562
# Pass 4: 0.422
# ...floor at 0.200
```

---

## Configuration

```python
config = {
    'edge_threshold': 0.10,
    'max_thickness': 5,
    'min_segment_pixels': 10,
    'max_segments': 50,
    'base_gate_threshold': 0.65,
    'base_translation': 30,

    # Unified effect strength
    'base_effect_strength': 1.0,
    'decay_rate': 0.75,
    'min_effect_strength': 0.2,
}
```

---

## Known Issues

### 1. Grayscale Collapse on Input
**Location**: `run_v6_demo()` line 394
```python
img = np.array(Image.open(image_path).convert('L'))  # <-- LOSES COLOR
```
**Impact**: Purple/colored input images lose their color immediately.
**Fix**: Load as RGB, use luminance for structure detection but preserve color for output.

### 2. Light Stroke Detection Failure
**Location**: `extract_segments()` and `dilate_with_color_propagation()`
```python
contours = img_gray < 0.4  # Only detects DARK pixels
```
**Impact**: Light strokes on white background (like enso paintings) have 0% shader affinity.
**Fix**: Use edge detection (gradient magnitude) instead of threshold, or adaptive thresholding.

### 3. Enso Images Specifically
The mspaint-enso images are light gray strokes (~0.7 luminance) on white (~1.0).
Since `0.7 < 0.4` is False, no contours are detected → 0 segments → no effect.

---

## Files

```
resnet_spectral_shader_v6.py     # Canonical v6 (~420 LOC)
resnet_spectral_shader_v5.py     # Previous (non-stationary orbits)
resnet_spectral_shader_v4.py     # Drop shadows, segment extraction
resnet_stacked_demo.py           # Stacked pass demo
```

---

## Output Samples

```
demo_output/resnet_v6_snek-heavy_labeled.png   # Complex line art
demo_output/resnet_v6_toof_labeled.png         # Simple closed curve
demo_output/resnet_v6_1bit redraw_labeled.png  # Dithered texture
```

Visual characteristics:
- Green/teal/purple palette from cyclic transform
- Orthogonal rotated segments with drop shadows
- Colors accumulate across passes (no black crushing)
- Depth:2 often optimal, depth:4-8 for saturated effects

---

## Architecture

```
INPUT (currently grayscale-collapsed)
    │
    ▼
[SPECTRAL] Fiedler bipartition → gate
    │
    ├── gate > threshold ──► [DILATION] Copy source colors to new pixels
    │
    └── gate < threshold ──► [SEGMENTS] Extract → Rotate 90° → Translate
                                   │
                                   ├── SHADOW: cyclic_transform(stronger)
                                   └── FRONT:  cyclic_transform(moderate)
    │
    ▼
effect_strength *= decay_rate
    │
    ▼
REPEAT (feed output to next pass)
```

---

## Effect Strength Attenuation Table

| Pass | effect_strength | gate_threshold |
|------|-----------------|----------------|
| 1 | 1.000 | 0.65 |
| 2 | 0.750 | 0.57 |
| 3 | 0.562 | 0.49 |
| 4 | 0.422 | 0.41 |
| 5 | 0.316 | 0.33 |
| 6 | 0.237 | 0.25 |
| 7 | 0.200 | 0.25 |
| 8 | 0.200 | 0.25 |

---

## Recommended Configurations

| Use Case | Passes | Notes |
|----------|--------|-------|
| Subtle enhancement | 1-2 | Clean orthogonal accents |
| Balanced effect | 2-3 | **Sweet spot** for most images |
| Saturated/artistic | 4-6 | Dense color accumulation |
| Maximum effect | 8 | Full coverage, may lose detail |

---

## For Continuing Agents

1. **Run**: `uv run python resnet_spectral_shader_v6.py`
2. **Key functions**:
   - `dilate_with_color_propagation()` - true color dilation
   - `cyclic_color_transform()` - no-fixed-point color rotation
   - `single_pass_v6()` - unified effect_strength application
3. **To fix grayscale collapse**: Load as RGB, extract luminance for Laplacian/gating but keep RGB for output
4. **To fix light stroke detection**: Replace `img_gray < 0.4` with gradient-based edge detection

---

## Visual Effect Characteristics

These effects are novel because:
1. **Spectral gating** creates structure-aware spatial partitioning
2. **Cyclic color transform** avoids fixed points, allowing continuous evolution
3. **Stacked application** with decay creates depth without saturation collapse
4. **Contour rotation** creates orthogonal structure impossible with standard filters

Conventional 2D painting software uses conservative brush engines that don't explore this parameter space.

---

*Handoff Version: 6.0*
*Parent: resnet-spectral-shader-v5-handoff.md*
