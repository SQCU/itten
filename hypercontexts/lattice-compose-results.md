# Lattice Extrusion + Texture Transform Composition Results

## Summary

This document reports the results of demonstrating that lattice extrusion composes with texture transforms on 3D surfaces.

**Conclusion: Composition VERIFIED.** Lattice extrusion and texture transforms combine to create visually distinct results at different theta values.

## Demo Execution

**Demo script:** `/home/bigboi/itten/demos/lattice_texture_compose.py`
**Output directory:** `/home/bigboi/itten/demo_output/lattice_compose/`

### What the Demo Does

1. **Creates a TerritoryGraph** with 3 islands connected by 2 narrow bridges (439 nodes total)
2. **Applies ExpansionGatedExtruder** to create 3D geometry based on local spectral expansion
3. **Synthesizes textures** using `texture.synthesize()` at different theta values (0.1, 0.5, 0.9)
4. **Renders composed results** showing lattice geometry "perched" on textured egg surfaces
5. **Generates comparison images** for visual inspection

## Key Results

### Lattice Extrusion Statistics

| Metric | Value |
|--------|-------|
| Total nodes | 439 |
| Islands | 3 |
| Bridges | 2 |
| Extrusion layers | 5 (layer 5 to layer 9) |

Layer distribution:
- Layer 5: 3 nodes (lowest, bottleneck regions)
- Layer 6: 118 nodes
- Layer 7: 229 nodes (majority)
- Layer 8: 87 nodes
- Layer 9: 2 nodes (highest, island centers)

### Texture Synthesis at Different Theta

| Theta | Height Mean | Height Std | Interpretation |
|-------|-------------|------------|----------------|
| 0.1 | 0.257 | 0.235 | Coarse structure, Fiedler-dominated |
| 0.5 | 0.279 | 0.250 | Balanced, mixed eigenvector influence |
| 0.9 | 0.279 | 0.249 | Fine structure, higher eigenvectors |

### Composition Observations

1. **Base Surface Changes with Theta**
   - theta=0.1: Large, smooth texture regions following Fiedler partitioning
   - theta=0.5: Mixed texture scales, moderate nodal line prominence
   - theta=0.9: Finer texture detail, more frequent nodal lines

2. **Lattice Geometry Integration**
   - Nodes sit "on top" of the textured surface
   - Island nodes (green) extruded higher (layers 7-9)
   - Bridge nodes (orange) remain lower (layers 5-6)
   - Node brightness correlates with extrusion layer

3. **Visual Composition Effects**
   - Texture coloring visible on egg surface between lattice nodes
   - Lighting affected by both surface normals and texture normal map
   - Lattice node colors remain distinct but integrate with background

## Output Files

| File | Description |
|------|-------------|
| `00_carrier_pattern.png` | Territory pattern used as carrier for texture synthesis |
| `01_texture_theta_0.1.png` | Texture at theta=0.1 (coarse) |
| `01_texture_theta_0.5.png` | Texture at theta=0.5 (balanced) |
| `01_texture_theta_0.9.png` | Texture at theta=0.9 (fine) |
| `02_composed_theta_0.1.png` | Composed render at theta=0.1 |
| `02_composed_theta_0.5.png` | Composed render at theta=0.5 |
| `02_composed_theta_0.9.png` | Composed render at theta=0.9 |
| `03_texture_comparison.png` | Side-by-side texture comparison |
| `04_composed_comparison.png` | Side-by-side composed comparison |
| `05_expansion_heatmap.png` | Local expansion values visualization |
| `06_bare_lattice.png` | Lattice without texture (baseline) |

## Technical Details

### Lattice Module Components Used

- `TerritoryGraph` - Graph structure with island/bridge regions
- `create_islands_and_bridges()` - Helper to create demo territory
- `ExpansionGatedExtruder` - Spectral-dependent extrusion
- `FiedlerAlignedGeometry` - Orientation based on Fiedler vector
- `render_3d_egg_mesh()` - 3D rendering on egg surface

### Texture Module Components Used

- `synthesize()` - Main texture synthesis function
- `TextureResult` - Result with height_field and normal_map

### Key Insight Confirmed

The handoff document stated:
> "Lattice extrusion creates geometry that 'perches' on the surface. The texture transform should affect both the base surface appearance AND how the geometry visually integrates."

This is confirmed by the demo:
1. Texture (controlled by theta) provides the base surface coloring
2. Normal maps from texture affect surface lighting
3. Lattice nodes are rendered on top with their own colors
4. The composition creates a unified visual where:
   - Textured surface is visible between nodes
   - Node brightness comes from extrusion layer
   - Node colors come from island/bridge membership

## Reproducibility

To reproduce these results:

```bash
cd /home/bigboi/itten
uv run python demos/lattice_texture_compose.py
```

Output will be saved to `/home/bigboi/itten/demo_output/lattice_compose/`
