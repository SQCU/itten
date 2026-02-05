# Subagent Handoff: Bump/Normal Maps as Actual Bump/Normal Maps in 3D

## Problem
Current 3D renders use textures as color maps, not as bump/normal maps.
We need to USE the height field for displacement and normal map for lighting.

## What "Bump Map" Means
- Height field displaces surface geometry along normal
- Normal map perturbs surface normals for lighting calculation
- Result: surface appears to have 3D relief/embossing

## Implementation

Create `/home/bigboi/itten/texture/render/bump_render.py`:

1. `render_bumped_surface(base_mesh, height_field, normal_map, displacement_scale=0.1)`
   - Apply height_field as vertex displacement
   - Use normal_map to perturb lighting normals
   - Render with Lambertian + specular lighting

2. `render_embossed_egg(height_field, normal_map, displacement_scale=0.1)`
   - Egg surface with bump mapping applied
   - Should see both:
     - Overall egg curvature through embossing
     - Embossing texture through lighting

## Test Configuration

1. Generate texture using spectral_embed(amongus, noise, theta)
2. Extract height_field and normal_map
3. Render to egg surface WITH bump mapping
4. Sweep theta=[0.1, 0.5, 0.9]
5. Compare:
   - Color-only render (current behavior)
   - Bump-mapped render (new behavior)

## Visual Success Criteria
- Can see surface curvature THROUGH the embossing texture
- Can see embossing texture THROUGH the lighting
- Different theta values produce visibly different embossing depth/character

## PSNR Measurement
- Compute PSNR between adjacent theta values
- Bump-mapped renders should show HIGHER PSNR variance than color-only
  (because geometry changes are more dramatic than color changes)

## Output
- `/home/bigboi/itten/texture/render/bump_render.py`
- `/home/bigboi/itten/demo_output/bump_render/`
  - `color_only_theta_*.png`
  - `bump_mapped_theta_*.png`
  - `comparison_grid.png`
- `/home/bigboi/itten/hypercontexts/bump-render-results.md`
