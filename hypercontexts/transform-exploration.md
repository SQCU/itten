# Transform Exploration: Spectral Graph Operations for Texture Synthesis

## Context

We have spectral graph operations in `spectral_ops_fast.py`:
- Laplacian construction (image → graph)
- Lanczos eigenvector computation
- Heat diffusion
- Chebyshev polynomial filtering
- Local Fiedler vectors
- Expansion estimates

## What Graph Ops Can Do

Graph operations on images treat pixels as nodes with edges to neighbors. This enables:
- **Nodal lines**: Zero-crossings of eigenvectors create natural partitions
- **Diffusion**: Heat spreads along graph edges (respects image structure)
- **Spectral filtering**: Low-pass (smooth) to high-pass (edges) via eigenvalue weighting
- **SDFs**: Signed distance from spectral contours
- **Topology linking**: Contiguous regions become connected components

## Underrepresented Transforms to Explore

Looking for transforms that:
1. Use partial spectral decomposition creatively
2. Compose carrier + operand in non-obvious ways
3. Create high aesthetic variance (measurable by PSNR residual)
4. Are "stunning, bedazzling, highly spatially covariant"

## Subagent Mission

Generate 5 rounds of:
1. 5 underrepresented transforms with implementation sketches
2. 1 fictional GameFAQs demoscene argument about how an effect "really worked"

Focus on:
- Spectral contour SDFs
- Eigenvector phase relationships
- Multi-scale spectral pyramids
- Graph topology manipulation
- Anisotropic diffusion variants

---
*Status: EXPLORING*
