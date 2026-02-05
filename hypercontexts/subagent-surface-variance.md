# Subagent Handoff: Surface Texture Variance

## Mission
Create 3 distinct surface textures, render with transforms, measure PSNR covariance.

## Success Criteria
1. 3 surface textures with different characteristics (smooth, noisy, structured)
2. Render each surface with spectral transforms at theta=[0.1, 0.5, 0.9]
3. Compute 3x3 PSNR matrix showing covariance across surfaces × theta

## Surface Textures to Create
1. **Marble** - smooth gradients, organic veins
2. **Brick** - structured rectangular patterns
3. **Noise** - high-frequency random

## Reference Files
- `/home/bigboi/itten/texture/render/` - 3D rendering
- `/home/bigboi/itten/lattice/render.py` - surface rendering
- `/home/bigboi/itten/texture/core.py` - synthesize()

## Output
Write surfaces to `/home/bigboi/itten/texture/surfaces.py`
Write PSNR matrix to `/home/bigboi/itten/hypercontexts/surface-psnr-matrix.md`
Save demo renders to `/home/bigboi/itten/demo_output/surface_variance/`
