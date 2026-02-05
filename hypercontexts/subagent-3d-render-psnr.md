# Subagent Handoff: 3D Render PSNR Validation

## Mission
Measure PSNR residuals when transforms are rendered to 3D surfaces, not just flat textures.

## Context
User requirement: "high-ish in psnr residual when rendered to a 3d surface"
Current tests only measure flat texture PSNR.

## Steps
1. Read existing render code:
   - `/home/bigboi/itten/lattice/render.py` (render_3d_egg_mesh)
   - `/home/bigboi/itten/texture/render/`

2. Create test that:
   - Generates a surface texture (e.g., marble)
   - Renders it to 3D egg surface → baseline_3d
   - Applies transform at theta=[0.1, 0.5, 0.9]
   - Renders transformed to same 3D view → transformed_3d
   - Computes PSNR(baseline_3d, transformed_3d)

3. Compare 3D PSNR to flat PSNR:
   - Flat PSNR: PSNR(flat_surface, flat_transformed)
   - 3D PSNR: PSNR(3d_surface, 3d_transformed)
   - We want both to show meaningful variance

## Output
Write to `/home/bigboi/itten/hypercontexts/3d-render-psnr-results.md`
Save example renders to `/home/bigboi/itten/demo_output/3d_psnr/`
