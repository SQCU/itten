# Amongus Pattern Variations - Results

## Summary

Successfully created 6 new amongus pattern variation functions in `/home/bigboi/itten/texture/patterns.py`. These variations are designed for use as carriers and operands in spectral blend testing.

## New Functions Added

### 1. `generate_amongus_stretched(size, stretch_x=1.5, stretch_y=1.0)`

Applies horizontal and/or vertical stretch transformation.

- `stretch_x > 1.0`: Compresses figure horizontally (appears taller)
- `stretch_y > 1.0`: Compresses figure vertically (appears wider)
- Uses `scipy.ndimage.map_coordinates` for smooth interpolation

### 2. `generate_amongus_sheared(size, shear_angle=15)`

Applies shear transformation (parallelogram distortion).

- `shear_angle`: Angle in degrees for horizontal shear
- Positive angles shear to the right, negative to the left

### 3. `generate_amongus_rotated(size, angle=45)`

Applies rotation transformation.

- `angle`: Counter-clockwise rotation in degrees
- Uses `scipy.ndimage.rotate` with bilinear interpolation

### 4. `generate_amongus_tessellated(size, copies_x=3, copies_y=3)`

Creates a tiled grid of amongus figures.

- `copies_x`, `copies_y`: Number of tiles in each direction
- Each tile is centered within its grid cell

### 5. `generate_amongus_warped(size, warp_strength=0.3, seed=None)`

Applies smooth random warping via displacement field.

- `warp_strength`: Controls displacement magnitude (0.0 to 1.0)
- Uses low-frequency noise grid interpolated with cubic splines
- `seed`: Optional random seed for reproducibility

### 6. `generate_amongus_random_transform(size, seed=None)`

Applies random combination of all transforms.

- Randomly selects parameters for stretch, shear, rotation, and warp
- Parameter ranges chosen to produce recognizable but varied results
- `seed`: Optional random seed for reproducibility

## Output Files

All files saved to `/home/bigboi/itten/demo_output/amongus_variations/`:

### Individual Variations (128x128 pixels)

| File | Description | Fill Ratio |
|------|-------------|------------|
| `original.png` | Base amongus pattern | 42.5% |
| `stretched.png` | Stretched (1.5x, 1.0y) | 28.1% |
| `sheared.png` | Sheared (15 degrees) | 42.5% |
| `rotated.png` | Rotated (45 degrees) | 42.5% |
| `tessellated.png` | Tessellated (3x3 grid) | 40.3% |
| `warped.png` | Warped (strength 0.15) | 43.7% |
| `random_combo1.png` | Random transform (seed=1) | 33.2% |
| `random_combo2.png` | Random transform (seed=2) | 45.9% |
| `random_combo3.png` | Random transform (seed=3) | 47.7% |

### Comparison Grid

- `grid.png`: 3x3 comparison grid showing all 9 variations with labels

## Usage for Spectral Blending

These variations can be used as:

1. **Carriers**: Structure that guides spectral decomposition
   - Tessellated pattern provides repeating structure
   - Stretched/sheared patterns provide directional bias

2. **Operands**: Pattern to embed into other carriers
   - Warped patterns add organic variation
   - Rotated patterns test orientation invariance

3. **Both**: Amongus x Amongus spectral blends
   - Combine different transforms for complex visual effects
   - Random transforms enable automated variation generation

## Implementation Notes

- All functions return `np.float32` arrays normalized to [0, 1]
- Size parameter controls output dimensions (square images)
- Transformations use `scipy.ndimage` for high-quality interpolation
- All functions preserve the binary nature of the pattern (values near 0 or 1)

## Demo Script

Generator script: `/home/bigboi/itten/demo_output/amongus_variations/generate_demo.py`

Run with:
```bash
uv run python /home/bigboi/itten/demo_output/amongus_variations/generate_demo.py
```
