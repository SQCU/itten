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

| File | Role |
|------|------|
| `spectral_shader_model.py` | `SpectralShader` nn.Module — primary entry point (`from_config()`) |
| `spectral_shader_layers.py` | nn.Module building blocks (gate, thicken, shadow, cross-attention) |
| `spectral_embedding_layer.py` | `SpectralEmbedding` — tiled Lanczos Fiedler computation as nn.Module |
| `spectral_shader_ops_cuter.py` | Functional implementation the Modules delegate to |
| `spectral_shader_ops.py` | Original verbose implementation (retained for regression tests) |
| `spectral_ops_fast.py` | Numerical kernel (Laplacian construction, Lanczos iteration, tiled eigenvectors) |
| `demo_spectral_shader.py` | Demo script with ADSR envelope |
| `spectral_shader_main.py` | Composable CLI (single/paired images, AR passes, quality presets) |
| `image_io.py` | Timestamped image I/O (prevents accidental overwrites) |

Both `spectral_shader_main.py` and `demo_spectral_shader.py` use `SpectralShader.from_config()`.

## Performance: Reference vs Module Path

Benchmark on 20 pairs of 512x512 structured textures, each pair processed through one
self-attention pass + one cross-attention pass (CUDA, RTX-class GPU):

| | Total (20 pairs) | Per pair | Per image |
|---|---|---|---|
| **Fiedler only** | 107.2s | — | 2679 ms |
| **Reference** (`spectral_shader_ops.shader_forwards`) | 136.3s | 6813 ms | — |
| **Module** (`SpectralShader.from_config().forward`) | 114.2s | 5708 ms | — |

The nn.Module path (`spectral_shader_model.py`) is **16% faster** than calling
`spectral_shader_ops.py` functions directly (1.19x speedup). Both produce
**bit-exact identical output** (verified by e2e regression tests, 0.000000 mean
absolute difference).

### Fused kernels

The Module path benefits from several fused operations in `SpectralEmbedding` that
the reference path does not use:

| Fusion | What it does | Savings |
|--------|-------------|---------|
| **Fused Lanczos recurrence** | 3-term recurrence absorbed into reorthogonalization; alpha extracted from projection coefficients | ~60 kernel launches/tile |
| **Pre-computed Laplacian offsets** | Multi-radius neighbor offsets computed once in `__init__`, stored as buffers | Eliminates ~169-iter Python loop per tile |
| **Pointer-jumping connected components** | O(log diameter) iterations instead of O(diameter); vectorized 4-direction min | ~3x fewer iterations |
| **Batched blend accumulation** | Single `evecs * blend.unsqueeze(-1)` instead of per-eigenvector loop | k kernel launches → 1 |
| **Inline spectral complexity** | Reuses pre-computed Sobel gx/gy instead of recomputing inside dilation | Eliminates redundant Sobel pass |
| **Vectorized segment transplant** | Pre-computed centroid offsets as tensor ops, no `.item()` CPU syncs | Eliminates per-segment CPU roundtrip |

### Where the time goes

```
Fiedler eigenvector computation    94%   ← tiled Lanczos (spectral_embedding_layer.py)
Shader ops (gate/thicken/shadow)    5%   ← segment extract, match, rotate, composite
Module dispatch + Python/CUDA       1%
```

The Fiedler bottleneck dominates: each 512x512 image requires ~2.7s for tiled Lanczos
(121 tiles at 64x64 with 16px overlap, 30 iterations each). A self+cross pair computes
3 Fiedler vectors (1 for self, 2 for cross), accounting for the majority of per-pair cost.

### Implications

- **Module path is the correct default.** It is faster than the reference path thanks to
  fused kernels, and buys nn.Module composability, `torch.compile` compatibility on
  shader submodules, and a clean `forward()` API.
- **Optimization effort belongs in the Fiedler computation**, not the shader ops.
  Cutting Lanczos from 30 to 15 iterations halves the Fiedler time at 0.48 correlation
  (vs 0.84 at 30 iters) — use `-q fast` in `spectral_shader_main.py` for this tradeoff.
- The reference path (`spectral_shader_ops.py`) is retained for regression testing only.

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
