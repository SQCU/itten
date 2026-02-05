# Hypercontext: PBR Texture Differentiation Subagent

## Mission
Find/implement a PBR surface material that more vividly differentiates fractional spectral transform generated normal map textures across θ parameter values.

## Current State
- Current renderer: `texture_synth_v2/render_egg.py`
- `render_pbr_surface()` uses: tilt_angle=30°, metallic=0.8, roughness=0.2, light_angle=20°, bump_strength=0.5
- Existing outputs in `texture_synth_outputs_v2/demo_case_*_pbr.png`

## Problem
The eigenvector nodal line etchings at different θ values (0, 0.2, 0.4, 0.6, 0.8, 1.0) are not sufficiently differentiated visually. Need materials that make spectral differences pop.

## Key Files
```
texture_synth_v2/render_egg.py    - PBR rendering (render_pbr_surface, render_comparison_pbr)
texture_synth_v2/normals.py       - Height to normal map conversion
texture_synth_v2/synthesize.py    - synthesize_texture(carrier, operand, theta, gamma)
texture_synth_v2/demo_unified.py  - CLI for generating demos
```

## What Makes Nodal Lines Special
- Nodal lines = zero crossings of eigenvectors (unambiguously spectral)
- θ=0: Fiedler vector (coarsest partition boundary)
- θ=1: Higher eigenvectors (finer spectral detail)
- The DIRECTION and DENSITY of nodal lines change with θ

## Research Directions

### 1. Anisotropic Materials
Materials that respond differently to bump direction:
- Brushed metal (anisotropic specular along one axis)
- Woven fabric (cross-hatch patterns visible at grazing angles)
- Hair/fur shaders (fibers aligned with surface gradient)

### 2. Subsurface Scattering
Materials where light penetrates and scatters:
- Jade/marble (veins visible through translucency)
- Skin/wax (softens harsh transitions)
- Could make nodal line *depth* visible

### 3. Iridescence
Color shifts based on viewing angle:
- Soap bubbles, oil films, pearl
- Thin-film interference depends on surface normal
- Could map θ to hue/iridescence period

### 4. Extreme Grazing Angles
- Current: 20° light angle
- Try: 5-10° for raking light (archaeological style)
- Fresnel effects become dominant

### 5. Two-Tone / Toon Materials
- Discrete lighting levels
- Binary shadow/highlight boundary follows bump contours exactly
- Makes nodal lines appear as sharp shadow edges

## Output Deliverables
1. Modified `render_pbr_surface()` or new rendering function
2. Updated `render_comparison_pbr()` if needed
3. Demo images showing θ sweep with improved differentiation
4. Brief explanation of why chosen material works

## Commands to Test
```bash
# Generate current demos for baseline
python texture_synth_v2/demo_unified.py --all --pbr --size 96 --output outputs_test/

# Quick synthesis test
python -c "
from texture_synth_v2 import synthesize_texture, generate_varied_amongus, generate_checkerboard
carrier = generate_varied_amongus(64)
operand = generate_checkerboard(64)
height = synthesize_texture(carrier, operand, theta=0.5)
print(f'Height: {height.shape}, [{height.min():.3f}, {height.max():.3f}]')
"
```

## Constraints
- Keep render times reasonable (avoid ray tracing if possible)
- Output must be viewable as static PNG
- Must work with existing normal map format
