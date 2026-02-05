# Subagent Handoff: 3D Bump/Normal Map Render Sweep

## Problem
We implemented `texture/render/bump_render.py` with `render_bumped_egg()` but never demonstrated it with a combinatoric sweep of transforms.

## Mission
Create a combinatoric sweep of (carrier × operand × theta) rendered to 3D bump-mapped geometry, then select PSNR outliers for presentation.

## Sweep Configuration

### Carriers (3):
- `amongus` (structured, recognizable)
- `checkerboard` (periodic, grid)
- Natural image from `demo_output/inputs/snek-heavy.png`

### Operands (3):
- `amongus_tessellated` (periodic amongus)
- `checkerboard` with different block size
- Edge-detected natural image (once CG operands exist, use that)

### Theta values (3):
- 0.1, 0.5, 0.9

### Total combinations: 3 × 3 × 3 = 27

## For Each Combination

1. Apply `spectral_warp_embed(carrier, operand, theta)` or `spectral_embed`
2. Extract height_field and normal_map
3. Render with `render_bumped_egg(height_field, normal_map, ...)`
4. Compute:
   - PSNR(carrier_3d_render, result_3d_render)
   - PSNR(operand_3d_render, result_3d_render)
5. Store results in matrix

## Select Outliers

After sweep, identify:
- **High covariance outliers**: combinations where BOTH PSNR values are moderate (shows both signatures)
- **Carrier-dominant**: high operand PSNR, low carrier PSNR
- **Operand-dominant**: opposite
- **Balanced**: both PSNRs similar and moderate

## Output

### Images
Save to `/home/bigboi/itten/demo_output/3d_bump_sweep/`:
- All 27 renders as `{carrier}_{operand}_theta{theta}.png`
- Outlier comparison grid showing selected interesting cases
- PSNR heatmap visualization

### Data
- Full PSNR matrix as CSV/markdown table
- Outlier selections with justification

### Report
`/home/bigboi/itten/hypercontexts/3d-bump-sweep-results.md`

## Implementation

```python
from texture.blend import spectral_warp_embed
from texture.render.bump_render import render_bumped_egg
from texture.patterns import generate_amongus, generate_checkerboard, generate_amongus_tessellated

carriers = {
    'amongus': generate_amongus(128),
    'checkerboard': generate_checkerboard(128, block_size=16),
    'natural': load_and_resize('snek-heavy.png', 128)
}

operands = {
    'amongus_tess': generate_amongus_tessellated(128, copies_x=4, copies_y=4),
    'checker_fine': generate_checkerboard(128, block_size=8),
    'natural_edges': sobel_edges(load_image('toof.png'))
}

for c_name, carrier in carriers.items():
    for o_name, operand in operands.items():
        for theta in [0.1, 0.5, 0.9]:
            result = spectral_warp_embed(carrier, operand, theta)
            # ... render and measure
```

## Success Criteria
- 27 bump-mapped 3D renders produced
- PSNR matrix shows variance (not all same)
- Outlier grid demonstrates variety of effects
- Can visually see both carrier and operand influence in selected renders
