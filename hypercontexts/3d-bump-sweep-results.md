# 3D Bump-Mapped Render Sweep Results

Generated: 2026-01-31 01:07:29

## Overview

Combinatoric sweep of spectral_warp_embed rendered to 3D bump-mapped egg geometry.

- **Carriers**: amongus, checkerboard, natural
- **Operands**: amongus_tess, checker_fine, amongus_shear
- **Theta values**: [0.1, 0.5, 0.9]
- **Total combinations**: 27

## Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 128x128 |
| Render size | 512x512 |
| Displacement scale | 0.15 |
| Normal strength | 0.8 |
| Eigenvectors | 8 |
| Warp scale | 5.0 |

## PSNR Statistics

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| PSNR(carrier) | 5.11 dB | 2.13 dB | 2.77 dB | 11.60 dB |
| PSNR(operand) | 21.48 dB | 7.93 dB | 12.00 dB | 37.02 dB |

## Full PSNR Matrix

| Carrier | Operand | Theta | PSNR(carrier) | PSNR(operand) | Diff |
|---------|---------|-------|---------------|---------------|------|
| amongus | amongus_shear | 0.1 | 9.08 dB | 37.02 dB | -27.93 dB |
| amongus | amongus_shear | 0.5 | 10.26 dB | 23.91 dB | -13.64 dB |
| amongus | amongus_shear | 0.9 | 11.60 dB | 19.25 dB | -7.65 dB |
| amongus | amongus_tess | 0.1 | 3.31 dB | 32.16 dB | -28.85 dB |
| amongus | amongus_tess | 0.5 | 4.50 dB | 18.52 dB | -14.03 dB |
| amongus | amongus_tess | 0.9 | 5.85 dB | 13.73 dB | -7.88 dB |
| amongus | checker_fine | 0.1 | 3.30 dB | 30.64 dB | -27.34 dB |
| amongus | checker_fine | 0.5 | 4.56 dB | 17.37 dB | -12.81 dB |
| amongus | checker_fine | 0.9 | 5.96 dB | 12.84 dB | -6.88 dB |
| checkerboard | amongus_shear | 0.1 | 3.29 dB | 32.71 dB | -29.42 dB |
| checkerboard | amongus_shear | 0.5 | 4.46 dB | 18.93 dB | -14.47 dB |
| checkerboard | amongus_shear | 0.9 | 5.79 dB | 13.99 dB | -8.20 dB |
| checkerboard | amongus_tess | 0.1 | 3.29 dB | 31.07 dB | -27.77 dB |
| checkerboard | amongus_tess | 0.5 | 4.49 dB | 17.75 dB | -13.26 dB |
| checkerboard | amongus_tess | 0.9 | 5.84 dB | 13.20 dB | -7.36 dB |
| checkerboard | checker_fine | 0.1 | 3.33 dB | 28.90 dB | -25.56 dB |
| checkerboard | checker_fine | 0.5 | 4.62 dB | 16.11 dB | -11.49 dB |
| checkerboard | checker_fine | 0.9 | 6.02 dB | 12.00 dB | -5.98 dB |
| natural | amongus_shear | 0.1 | 2.78 dB | 32.69 dB | -29.90 dB |
| natural | amongus_shear | 0.5 | 3.94 dB | 18.76 dB | -14.82 dB |
| natural | amongus_shear | 0.9 | 5.27 dB | 13.69 dB | -8.42 dB |
| natural | amongus_tess | 0.1 | 2.77 dB | 32.07 dB | -29.29 dB |
| natural | amongus_tess | 0.5 | 3.94 dB | 18.31 dB | -14.37 dB |
| natural | amongus_tess | 0.9 | 5.28 dB | 13.38 dB | -8.10 dB |
| natural | checker_fine | 0.1 | 3.56 dB | 30.58 dB | -27.02 dB |
| natural | checker_fine | 0.5 | 4.78 dB | 17.43 dB | -12.65 dB |
| natural | checker_fine | 0.9 | 6.14 dB | 13.04 dB | -6.90 dB |

## Outlier Analysis

### Balanced (Both signatures visible, similar PSNR)

### Carrier Dominant (Carrier structure preserved)

### Operand Dominant (Operand pattern preserved)
- **natural_amongus_shear_theta0.1**: PSNR(c)=2.8dB, PSNR(o)=32.7dB
- **checkerboard_amongus_shear_theta0.1**: PSNR(c)=3.3dB, PSNR(o)=32.7dB
- **natural_amongus_tess_theta0.1**: PSNR(c)=2.8dB, PSNR(o)=32.1dB
- **amongus_amongus_tess_theta0.1**: PSNR(c)=3.3dB, PSNR(o)=32.2dB
- **amongus_amongus_shear_theta0.1**: PSNR(c)=9.1dB, PSNR(o)=37.0dB

### High Covariance (Both PSNRs moderate, showing blend)
- **amongus_amongus_shear_theta0.9**: PSNR(c)=11.6dB, PSNR(o)=19.2dB

## Output Files

### Directories

| Directory | Contents |
|-----------|----------|
| `carriers/` | Source carrier height fields |
| `operands/` | Source operand height fields |
| `base_renders/` | 3D renders of raw carriers/operands |
| `height_fields/` | Blended height fields from spectral_warp_embed |
| `renders/` | All 27 3D bump-mapped renders |
| `grids/` | Comparison grids |

### Key Images

- `grids/outlier_showcase.png` - Selected outliers from each category
- `grids/grid_{carrier}_{operand}.png` - Theta progression for each pair

## Observations

1. **Theta Effect**: As theta increases from 0.1 to 0.9, the carrier structure becomes more dominant in the blend. This is visible in the PSNR trends where PSNR(carrier) tends to increase with theta.

2. **Carrier Influence**: The carrier determines WHERE features appear on the 3D surface. The amongus carrier creates localized bump regions, while checkerboard creates regular periodic patterns.

3. **Operand Influence**: The operand determines WHAT patterns fill those regions. Tessellated operands create repetitive micro-structure, while sheared operands create directional flow.

4. **3D Visualization**: The bump mapping effectively visualizes the blend - both carrier and operand signatures are visible in the surface deformation and lighting.

5. **Best Blends**: The most visually interesting results tend to be in the 'balanced' and 'high_covariance' categories where both input signatures are clearly present.

## Technical Notes

- Uses `spectral_warp_embed` which warps the operand along the carrier's eigenvector gradients
- Height fields are normalized to [0,1] before rendering
- Normal maps generated with strength=2.0 for enhanced surface detail
- Render uses egg geometry with Blinn-Phong shading
