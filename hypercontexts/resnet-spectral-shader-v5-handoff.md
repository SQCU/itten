# ResNet Spectral Shader v5 - Handoff Document

*Session Date: 2026-02-01*
*Status: Working implementation with non-stationary color orbits*

---

## Summary

Implemented a ResNet-like spectral compute shader that demonstrates:
1. **Spectral gating** (Fiedler bipartition)
2. **Contour extraction + 90° rotation**
3. **Drop shadows with z-buffering**
4. **Non-stationary color orbits** (colors shift relative to current hue)
5. **Stacked passes with attenuated gating**

---

## Version History

| Version | Key Changes |
|---------|-------------|
| v1 | Initial attempt - effects too subtle |
| v2 | Stronger effects, proper color rotation matrix |
| v3 | Fiedler-based gating for meaningful bipartition |
| v4 | Contour segment extraction, drop shadows, explicit blue coloring |
| **v5** | Non-stationary color orbits, color-preserving thickening, stacked passes |

---

## Files

```
resnet_spectral_shader_v5.py     # Canonical implementation (~400 LOC)
resnet_spectral_shader_v4.py     # Previous version (static colors)
resnet_stacked_demo.py           # Stacked pass demonstration (v4-based)
hypercontexts/resnet-spectral-shader-protocol.md  # Protocol document
```

---

## v5 Architecture

```
INPUT (grayscale bitmap)
    │
    ▼
[SPECTRAL] Fiedler vector → bipartition gate
    │
    ├─── gate > threshold ──► [BRANCH_A] Color-preserving thickening
    │                              │
    │                              ▼
    │                         Darken existing colors (preserve hue)
    │
    └─── gate < threshold ──► [BRANCH_B] Segment extraction
                                   │
                                   ▼
                              Rotate 90°, translate
                                   │
                              ┌────┴────┐
                              │         │
                           SHADOW    FRONT
                           (orbit+)  (orbit)
                              │         │
                              ▼         ▼
                         Draw shadow first (z-buffer behind)
                         Draw front (masks shadow)
    │
    ▼
[RESIDUAL] Output feeds next pass
    │
    ▼
REPEAT with attenuated gate + increased orbit strength
```

---

## Non-Stationary Color Orbit

The key innovation in v5: **color rotation is relative to current pixel color**.

```python
# Orbit logic:
# - Gray pixels → rotate toward blue
# - Blue pixels → rotate toward magenta (to increase LOCAL contrast)
# - Magenta → rotate toward red
# Creates a spiral through color space

orbit_strength = 1.0 + (pass_number - 1) * 0.25
# Pass 1: orbit=1.00
# Pass 4: orbit=1.75
# Pass 8: orbit=2.75
```

This ensures re-copied segments don't stay the same color - they orbit further through color space with each pass.

---

## Attenuation Schedule

| Pass | Gate Threshold | Orbit Strength | Effect Strength |
|------|----------------|----------------|-----------------|
| 1 | 0.65 | 1.00 | 1.00 |
| 2 | 0.57 | 1.25 | 0.88 |
| 3 | 0.49 | 1.50 | 0.76 |
| 4 | 0.41 | 1.75 | 0.64 |
| 5 | 0.33 | 2.00 | 0.52 |
| 6 | 0.25 | 2.25 | 0.50 |
| 7 | 0.25 | 2.50 | 0.50 |
| 8 | 0.25 | 2.75 | 0.50 |

- **Gate threshold decreases**: Later passes require higher activation to trigger edits
- **Orbit strength increases**: Later passes rotate colors more aggressively
- **Effect strength decreases**: Later passes have smaller thickening/translation

---

## Configuration

```python
config = {
    'edge_threshold': 0.10,        # Laplacian edge weight sensitivity
    'max_thickness': 5,            # Maximum contour thickening
    'min_segment_pixels': 10,      # Minimum segment size
    'max_segments': 45,            # Max segments per pass
    'base_gate_threshold': 0.65,   # Starting gate threshold
    'translation_distance': 28,    # Perpendicular displacement
}
```

---

## Test Images

| Image | Characteristics | Behavior |
|-------|-----------------|----------|
| snek-heavy.png | Complex line art | Rich segment extraction, visible color orbits |
| toof.png | Simple closed curve | Heavy thickening, scattered rotations |
| 1bit redraw.png | Dithered texture | Dither dots become "contours", mass thickening |

---

## Output Examples

Generated files in `demo_output/`:
- `resnet_v5_snek-heavy_labeled.png`
- `resnet_v5_toof_labeled.png`
- `resnet_v5_1bit redraw_labeled.png`

Grid layout:
```
ORIGINAL     | 1x (orbit=1.0)  | 2x (orbit=1.25)
4x (orbit=1.75) | 8x (orbit=2.75) | DIFFERENCE
```

---

## Key Behaviors Demonstrated

1. **Orthogonal shapes** ✓ - 90° rotated contour segments create perpendicular marks
2. **Contour-only copies** ✓ - Only line pixels copied, bbox background transparent
3. **Translation perpendicular to contour** ✓ - Displacement along gradient direction
4. **Drop shadow** ✓ - Blue shadow behind teal/cyan front, z-buffered
5. **Non-stationary color orbit** ✓ - Colors shift relative to current hue
6. **Stacked passes** ✓ - Same rule applied iteratively with attenuation
7. **Color-preserving thickening** ✓ - Existing colors darkened, not replaced

---

## Known Issues / Future Work

1. **Thickening dominates at high passes** - Consider reducing thickening for deeper passes
2. **Orbit could be more dramatic** - Increase base rotation angle for more visible hue shifts
3. **Fiedler not strictly required** - Could use `iterative_spectral_transform` output as gate
4. **1bit/dithered images** - Stipple patterns treated as contours, may need special handling

---

## For Continuing Agents

1. Run `uv run python resnet_spectral_shader_v5.py` to regenerate outputs
2. The non-stationary orbit logic is in `relative_color_rotation()` and `compute_shadow_color_orbit()`
3. Stacked passes are in `stacked_shader_v5()` with the attenuation schedule
4. If colors aren't orbiting enough, increase `base_rotation` parameter

---

## Dependencies

- `spectral_ops_fast.py` - Core spectral operations
- torch, numpy, PIL, scipy

---

*Handoff Version: 5.0*
*Parent: spectral-compute-shader-handoff.md*
