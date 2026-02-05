# Itten: Graph Spectral Attention Shader

**Status: Proof of Concept / Work in Progress**

This repository demonstrates what "copy heads" and "read heads" in a graph spectral attention model *could* implement using extremely shallow layer counts without relying on complex parameter-only learned priors.

## What This Is

The kernels and shaders here are **unfinished and need rewriting**. They exist to demonstrate:

1. **Spectral segment matching** - using Fiedler vectors (2nd eigenvector of the graph Laplacian) to identify and match structurally similar regions across images
2. **Copy/rotate operations** - physically repositioning pixel segments based on spectral signatures
3. **Autoregressive refinement** - ADSR-envelope controlled iterative shading

The codebase shows that meaningful visual transformations can emerge from spectral graph operations alone, without learned parameters.

## Self-Attention vs Cross-Attention

### Self-Attention (Single Image Autoregressive)

When an image "attends to itself," the shader:
- Identifies spectral segments (connected regions with similar graph Laplacian structure)
- Copies and rotates segments into nearby voids
- Thickens contours based on local spectral complexity
- Iteratively refines over multiple ADSR-envelope controlled passes

![Self-attention progression](demo_output/readme_self_attention.png)

The progression shows feature accretion from the same source: existing structure propagates and densifies without external input. Color distortion is minimal because source and target share the same palette.

### Cross-Attention (Two Image Transfer)

When image A (target) attends to image B (source):
- A provides **topology queries**: WHERE to place segments, WHAT spectral structure to seek
- B provides **content**: WHAT pixels/colors to actually copy
- Spectral signatures match segments across images despite different sizes

![Cross-attention: toof x 1bit](demo_output/readme_cross_attention.png)

The color distortion filter reveals which visual properties come from which source:
- Structure/position: from target (A)
- Texture/color: from source (B)

### Source Texture Comparison

The same target topology with different source textures:

![Source comparison](demo_output/readme_source_comparison.png)

## Z-Buffer Mode (Sum Contributions)

By default, when multiple segments write to the same pixel, last-write-wins (replacement semantics). With `--no-overwrite` flag, overlapping contributions are summed and averaged:

```bash
uv run python demo_spectral_shader.py --no-overwrite
```

In z-buffer mode:
- Overlapping segment pixels are **summed** then normalized by count
- Multiple segments writing to the same location produce blended colors
- Prevents any single segment from dominating shared regions

This demonstrates the difference between:
- **Replacement**: later segments overwrite earlier (standard mode, last-write-wins)
- **Blending**: overlapping segments blend colors (z-buffer mode, sum/average)

**Note:** The effect is only visible when segments actually overlap. With few/small segments, both modes may produce identical output.

## Running the Demos

```bash
# Install dependencies
uv sync

# Run all demos (self + cross attention, standard mode)
uv run python demo_spectral_shader.py

# Self-attention only
uv run python demo_spectral_shader.py --self

# Cross-attention only
uv run python demo_spectral_shader.py --cross

# Z-buffer mode (no overwrite)
uv run python demo_spectral_shader.py --no-overwrite

# Customize passes
uv run python demo_spectral_shader.py --passes 8
```

## Configuration (ADSR Envelope)

The shader uses an Attack-Decay-Sustain-Release envelope to control growth rate across passes:

| Phase   | `thicken_modulation` | `kernel_sigma_ratio` | `fill_threshold` | `dilation_radius` |
|---------|---------------------|---------------------|------------------|-------------------|
| Attack  | 0.2 (aggressive)    | 0.8 (spread)        | 0.05 (more fill) | 3                 |
| Decay   | 0.3                 | 0.6                 | 0.1              | 2                 |
| Sustain | 0.5                 | 0.5                 | 0.15             | 2                 |
| Release | 0.7 (minimal)       | 0.4 (tight)         | 0.2 (less fill)  | 1                 |

These parameters can be adjusted in `demo_spectral_shader.py` or passed via config dict.

## Key Files

- `spectral_shader_ops.py` - Core shader operations (segment extraction, copy heads, compositing)
- `spectral_ops_fast.py` - Vectorized spectral graph operations (Laplacian, Lanczos, Fiedler)
- `demo_spectral_shader.py` - Demo script with ADSR envelope
- `image_io.py` - Timestamped image I/O (prevents accidental overwrites)

## Technical Details

### Spectral Operations

1. **Graph Laplacian**: Built from image with multi-radius connectivity (handles dithered images)
2. **Fiedler Vector**: 2nd eigenvector via tiled Lanczos iteration
3. **Spectral Gate**: Sigmoid on Fiedler separates high/low spectral regions
4. **Complexity Field**: Standardize + sigmoid normalized gradient magnitude

### Copy Head Mechanics

1. **Segment Extraction**: Connected components in low-gate regions
2. **Spectral Signature**: `[mean_fiedler, std_fiedler, log(size), aspect_ratio]`
3. **Cross-Image Matching**: L2 distance in signature space
4. **Shadow Projection**: Diameter-based, complexity-modulated outward offset

## What Could Be Built

This demonstrates primitives for:
- Texture synthesis via spectral matching
- Style transfer without neural networks
- Graph-based attention mechanisms
- Procedural content generation

The operations shown here could be implemented as attention heads in a transformer-like architecture, where spectral signatures serve as queries/keys and pixel content serves as values.

## License

MIT
