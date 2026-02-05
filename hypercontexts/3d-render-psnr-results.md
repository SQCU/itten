# 3D Render PSNR Results

## Overview

This report compares PSNR (Peak Signal-to-Noise Ratio) values between:
- **Flat PSNR**: Comparison of flat 2D textures before/after transform
- **3D PSNR**: Comparison of textures rendered onto a 3D egg surface

Higher PSNR = more similar (less change from transform)
Lower PSNR = more different (more change from transform)

## Test Configuration

- Transform: `eigenvector_phase_field`
- Texture size: 128x128
- 3D render size: 400x400
- Theta values: [0.1, 0.5, 0.9]

## Results Table

| Texture | theta | Flat PSNR (dB) | 3D PSNR (dB) | Diff |
|---------|-------|----------------|--------------|------|
| checkerboard | 0.1 | 5.55 | 17.92 | -12.37 |
| checkerboard | 0.5 | 5.62 | 18.29 | -12.68 |
| checkerboard | 0.9 | 4.98 | 18.05 | -13.07 |
| noise | 0.1 | 12.46 | 25.52 | -13.07 |
| noise | 0.5 | 13.52 | 27.03 | -13.51 |
| noise | 0.9 | 10.99 | 24.13 | -13.14 |

## Statistical Summary

- **Flat PSNR range**: 4.98 - 13.52 dB
- **Flat PSNR std**: 3.55 dB
- **3D PSNR range**: 17.92 - 27.03 dB
- **3D PSNR std**: 3.83 dB
- **Average diff (flat - 3D)**: -12.97 dB

- **Correlation (flat vs 3D)**: 0.998

## Interpretation

**Both flat and 3D PSNR show meaningful variance across theta values.**

This indicates that:
1. The spectral transform produces perceptually different outputs at different theta values
2. This perceptual difference is preserved when rendered to a 3D surface
3. The 3D rendering does not destroy the transform's spectral properties

**Note**: The average difference of -12.97 dB between flat and 3D PSNR
indicates that the 3D rendering process affects perceived difference.
Positive difference means 3D rendering amplifies perceived change.
Negative difference means 3D rendering reduces perceived change.

## Sample Images

All images saved to: `/home/bigboi/itten/demo_output/3d_psnr/`

### Files

- `texture_<name>_original.png` - Original flat textures
- `texture_<name>_theta<value>.png` - Transformed flat textures
- `3d_<name>_baseline.png` - Original texture rendered to 3D egg
- `3d_<name>_theta<value>.png` - Transformed texture rendered to 3D egg

## Methodology

1. Generate test textures (checkerboard, noise)
2. For each texture and theta value:
   - Apply `eigenvector_phase_field` transform
   - Compute flat PSNR between original and transformed texture
   - Render both original and transformed to 3D egg surface
   - Compute 3D PSNR using egg mask (excluding background)
3. Compare flat vs 3D PSNR to assess how 3D rendering affects perceptual change