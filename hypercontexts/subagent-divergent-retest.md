# Subagent Handoff: Retest Divergent Transforms Properly

## Problems with Previous Test

1. **Only theta 0.1 and 0.9** - need intermediate values [0.1, 0.3, 0.5, 0.7, 0.9]
2. **Used noise fields** - should use natural images, checkerboards, geometric shapes
3. **Didn't use color textures** from `demo_output/3d_psnr/` as specified
4. **Wrong amongus approach** - dense tiling is bad. Should be single shape + random transforms

## Correct Test Configuration

### Carriers (use these, NOT noise):
1. Natural image from `demo_output/inputs/snek-heavy.png`
2. Checkerboard (always better than noise for PSNR judgment)
3. Single amongus with random transforms (NOT tessellated grid)

### Operands (use these, NOT noise):
1. Natural image from `demo_output/inputs/toof.png` (different from carrier)
2. Checkerboard with different block size
3. Single amongus with DIFFERENT random transforms

### Theta values:
[0.1, 0.3, 0.5, 0.7, 0.9] - need the intermediates!

### Color textures from 3d_psnr:
Check `demo_output/3d_psnr/` for the nice color renders and USE those.

## Fix Amongus Generation

DON'T do this (dense tiling):
```python
generate_amongus_tessellated(size, copies_x=4, copies_y=4)  # BAD
```

DO this (single shape + random transforms):
```python
def generate_amongus_scattered(size, num_copies=3, seed=None):
    """
    Single amongus shape with random copy+transform operations.

    Each copy gets random:
    - Position (translate)
    - Rotation (small range, e.g., -30 to +30 degrees)
    - Scale (small range, e.g., 0.7 to 1.3)
    - Shear (small range)
    """
    rng = np.random.default_rng(seed)
    base = generate_amongus(size // 2)  # Smaller base
    result = np.zeros((size, size))

    for i in range(num_copies):
        # Random transform parameters
        angle = rng.uniform(-30, 30)
        scale = rng.uniform(0.7, 1.3)
        shear = rng.uniform(-0.2, 0.2)
        tx = rng.uniform(0, size * 0.6)
        ty = rng.uniform(0, size * 0.6)

        # Apply transforms and composite
        transformed = apply_affine(base, angle, scale, shear, tx, ty)
        result = np.maximum(result, transformed)

    return result
```

## Test Script Requirements

Create `/home/bigboi/itten/tests/test_divergent_proper.py`:

1. Load ACTUAL good textures:
   - `demo_output/inputs/snek-heavy.png` as carrier
   - `demo_output/inputs/toof.png` as operand
   - Checkerboard (64x64, block_size=8)
   - Scattered amongus (NOT tessellated)

2. Test ALL 5 divergent transforms from spectral_ops_fast.py:
   - eigenvector_phase_field
   - spectral_contour_sdf
   - commute_time_distance_field
   - spectral_warp
   - spectral_subdivision_blend

3. Sweep theta = [0.1, 0.3, 0.5, 0.7, 0.9]

4. Measure divergence:
   - PSNR(carrier, result)
   - PSNR(operand, result)
   - Both should be < 15dB for divergence

5. Use color renders - load from 3d_psnr if available

## Output
- Add `generate_amongus_scattered()` to `/home/bigboi/itten/texture/patterns.py`
- `/home/bigboi/itten/tests/test_divergent_proper.py`
- Results to `/home/bigboi/itten/hypercontexts/divergent-proper-results.md`
- Renders to `/home/bigboi/itten/demo_output/divergent_proper/`
