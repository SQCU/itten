# Session Handoff: Spectral Blend + Bump Rendering

## What Was Built

### 1. True Carrier-Operand Spectral Blend (`texture/blend.py`)

**Problem solved**: Previous transforms were self-convolutions (texture × itself).

**New transforms**:
- `spectral_embed(carrier, operand, theta)` - project operand onto carrier's eigenvector basis
- `spectral_warp_embed(carrier, operand, theta)` - warp operand along carrier's spectral gradients

**Evidence of true blend**:
```
spectral_embed PSNR:
  theta=0.1: carrier=4.85dB, operand=34.24dB (operand dominates)
  theta=0.9: carrier=5.59dB, operand=14.13dB (carrier dominates)

spectral_warp_embed autocorrelation at lag=8:
  operand alone: H≈0 (no structure)
  theta=0.9:     H=-0.404 (carrier's periodic anti-correlation transferred!)
```

### 2. Bump/Normal Map 3D Rendering (`texture/render/bump_render.py`)

**Problem solved**: Textures were used as color decals, not bump maps.

**New capability**:
- `render_bumped_egg(height_field, normal_map, displacement_scale, normal_strength)`
- True displacement + normal perturbation + Blinn-Phong lighting

**Evidence it works**:
```
PSNR between theta=0.1 and theta=0.9:
  Color-only: 51.62 dB (minimal variation)
  Bump-mapped: 29.69 dB (20 dB MORE variation!)
```

### 3. Amongus Variations (`texture/patterns.py`)

6 new functions for transformed amongus:
- `generate_amongus_stretched(size, stretch_x, stretch_y)`
- `generate_amongus_sheared(size, shear_angle)`
- `generate_amongus_rotated(size, angle)`
- `generate_amongus_tessellated(size, copies_x, copies_y)`
- `generate_amongus_warped(size, warp_strength)`
- `generate_amongus_random_transform(size, seed)`

## Key Metrics Achieved

| Metric | Target | Achieved |
|--------|--------|----------|
| PSNR residual > blur | > blur's 20.6 dB | Phase Field: 9-12 dB ✅ |
| Autocorrelation structure changes | not just intensity | lag=8 goes 0 → -0.404 ✅ |
| Bump vs color-only variance | bump > color | 20 dB more variance ✅ |
| Dual PSNR dependence | both change | carrier + operand both respond ✅ |

## Demo Output Locations

```
demo_output/
├── carrier_operand/    - Spectral embed/warp results, warp fields
├── bump_render/        - Color vs bump comparisons at multiple theta
├── amongus_variations/ - All 9 amongus transforms in grid
├── surface_variance/   - Marble/brick/noise × theta matrix
├── 3d_psnr/           - Flat vs 3D PSNR comparison
└── lattice_compose/   - Lattice extrusion on textured surface
```

## Current Module Structure

```
texture/
├── core.py         - synthesize() main API
├── transforms.py   - eigenvector_phase_field, fiedler_nodal_lines
├── blend.py        - spectral_embed, spectral_warp_embed (NEW)
├── surfaces.py     - marble, brick, noise generators
├── patterns.py     - amongus + 6 variations, checkerboard, dragon, noise
└── render/
    └── bump_render.py - render_bumped_egg, render_color_only_egg (NEW)
```

## Next Steps

1. **Combine amongus variations with spectral blend**:
   - `spectral_embed(amongus_rotated, amongus_tessellated, theta)`
   - Test if different amongus×amongus combos produce distinct results

2. **Bump render the spectral blends**:
   - Generate height/normal from `spectral_warp_embed(amongus, noise, theta)`
   - Render with `render_bumped_egg()`
   - Measure PSNR variance across (carrier_transform, theta) grid

3. **Spatial covariance sweep**:
   - The warp transform showed good autocorrelation structure change
   - Verify this persists in bump-rendered 3D output
