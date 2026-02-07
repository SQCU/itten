# Handoff: Spectral Shader Optimization

## Session Summary

This session investigated optimizing the Fiedler vector computation for the spectral shader pipeline, targeting 100Hz (10ms) for real-time graphics demos. Current implementation runs at ~67Hz (15ms).

## Key Files Created This Session

| File | Purpose |
|------|---------|
| `GRAPH_COARSENING_RESEARCH.md` | **START HERE** - Literature review with 16 citations on spectral-preserving graph coarsening |
| `FIEDLER_BENCHMARK_RESULTS.md` | Benchmark of Fiedler approximation methods (heat diffusion, Chebyshev, etc.) - all failed |
| `benchmark_fiedler_alternatives.py` | Code for benchmarking alternative Fiedler approximations |
| `spectral_local_structure.py` | Fast (300Hz) local geometric features - but doesn't capture global spectral position |
| `spectral_coarsened.py` | Naive coarsening attempt - failed (0.002 correlation) |
| `spectral_correspondence_unified.py` | Unified polynomial spectral correspondence experiments |

## Key Files (Pre-existing)

| File | Purpose |
|------|---------|
| `spectral_shader_ops.py` | Main shader implementation with Fiedler-based gating and cross-attention |
| `spectral_shader_ops_cuter.py` | Streamlined 269-line version, torch.compile compatible |
| `spectral_ops_fast.py` | Spectral operations including `lanczos_fiedler_gpu`, `iterative_spectral_transform` |
| `spectral_ops_fast_cuter.py` | Minimal 263-line spectral ops |
| `spectral_shader_main.py` | CLI entry point with quality presets (fast/standard/quality) |

## What We Learned

### 1. Alternative Fiedler Approximations Don't Work
Tested: heat diffusion, Chebyshev filtering, SDF, bilateral proxy, power iteration.
Result: All <0.5 correlation with true Fiedler. Lanczos is uniquely good at extracting the 2nd eigenvector.

### 2. Local Features ≠ Global Spectral Position
Fast local features (gradient magnitude, curvature) correlate 0.39 with |∇Fiedler| (edge locations) but 0.02 with Fiedler values (global partition). These are orthogonal information.

### 3. Naive Coarsening Destroys Spectral Properties
Superpixel-based coarsening gave 0.002 correlation - the coarse Fiedler is a completely different bipartition.

### 4. Proper Coarsening Is a Research Problem
Graph coarsening that preserves spectral properties requires principled approaches from the literature:
- **GRACLUS**: Avoid eigenvectors entirely via kernel k-means equivalence
- **Local Variation (Loukas)**: Explicit spectral preservation guarantees
- **GRASS**: Near-linear, preserves first k eigenvalues

## The Core Tension

The spectral shader needs:
1. **Local geometry** (curvature, branching) → fast, 300Hz achievable
2. **Global correspondence** (Fiedler-based cross-image transfer) → expensive, 67Hz

The Fiedler captures "which side of the natural graph cut" - this is genuinely different from local features and can't be approximated cheaply without proper coarsening.

## Recommended Next Steps

### Option A: GRACLUS Path (Most Promising)
The GRACLUS paper shows normalized cut ≡ weighted kernel k-means. If normalized cut is what we actually want, we can optimize the same objective without eigenvectors.
- Read: Dhillon et al. 2007, link in `GRAPH_COARSENING_RESEARCH.md`
- Try: Port GRACLUS logic to PyTorch

### Option B: Loukas Coarsening
Proper spectral-preserving coarsening with theoretical guarantees.
- Code: https://github.com/loukasa/graph-coarsening
- Try: Coarsen to 500 nodes, dense eigensolve, map back

### Option C: GPU LOBPCG
If exact Fiedler needed, use GPU. 7x speedup reported.
- Try: `cugraph.spectralBalancedCutClustering()`

### Option D: Async/Memoization
Accept 67Hz Fiedler, compute async, cache per-image.
- Use fast local features (300Hz) for per-frame effects
- Use cached Fiedler for correspondence

## Architecture Context

```
spectral_shader_main.py          # CLI entry point
    ├── compute_fiedler()        # THE BOTTLENECK (15ms)
    │   └── compute_local_eigenvectors_tiled_dither()
    │       └── lanczos_fiedler_gpu()  # Lanczos iteration
    │
    └── run_shader()
        ├── two_image_shader_pass()    # Cross-attention transfer
        │   └── cross_attention_transfer()  # Uses Fiedler for correspondence
        │
        └── spectral_shader_pass()     # Single-image effects
            ├── fiedler_gate()         # Gate by Fiedler threshold
            ├── dilate_high_gate_regions()
            └── apply_low_gate_transform()  # Uses ∇Fiedler for direction
```

## Questions for Next Session

1. Is GRACLUS-style kernel k-means sufficient for the visual effect, or do we truly need eigenvectors?
2. Can we precompute Fiedler for a library of source images and use cached values for real-time transfer?
3. Would a learned embedding (train on Fiedler → predict from local features) work for this domain?

## Citations to Follow

From `GRAPH_COARSENING_RESEARCH.md`:
- **Immediate**: Loukas 2019 (spectral coarsening theory), Dhillon 2007 (GRACLUS)
- **If GPU path**: cuGraph docs, Sphynx paper
- **Deep theory**: Spielman-Srivastava 2011 (why spectral sparsification works)
