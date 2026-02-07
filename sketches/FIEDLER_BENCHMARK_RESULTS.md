# Fiedler Approximation Benchmark Results

## Summary

**Lanczos iteration is the clear winner** for Fiedler vector approximation. All alternative methods (heat diffusion, Chebyshev filtering, SDF gating, bilateral proxy) fail to achieve competitive correlation with the true Fiedler vector.

## Results (256x256 synthetic + 384x1152 real image)

| Method            | Correlation | Time (ms) | Speedup |
|-------------------|-------------|-----------|---------|
| **Lanczos-30**    | 0.83-0.84   | 13-16     | 20x     |
| Lanczos-20        | 0.62        | 9-10      | 29x     |
| Lanczos-15        | 0.48        | 7-8       | 37x     |
| Lanczos-10        | 0.33        | 5-6       | 50x     |
| Power-Inverse-30  | 0.28        | 7-10      | 31-37x  |
| Heat-Diffusion-30 | 0.27        | 6-7       | 44-46x  |
| Chebyshev-20      | 0.20        | 5         | 53-59x  |
| SDF               | 0.02-0.17   | 8-48      | 7-33x   |
| Bilateral         | <0.02       | 0.4-0.5   | 620x+   |

**Ground truth**: Lanczos-50 (282-321ms)

## Key Findings

1. **Lanczos dominates**: Even 10 iterations (0.33 corr) beats all non-Lanczos methods
2. **Lanczos-30 is optimal**: 0.84 correlation at 21x speedup - diminishing returns beyond this
3. **Lanczos-15 is minimum viable**: 0.48 correlation acceptable for preview/fast modes
4. **Alternative spectral methods fail**: Heat diffusion, power iteration, Chebyshev all plateau at ~0.27
5. **Non-spectral proxies fail completely**: SDF/Bilateral have near-zero correlation

## Why Alternatives Fail

- **Heat diffusion/Power iteration**: Converge to *first* eigenvector (constant), not second (Fiedler)
- **Chebyshev**: Band-pass targeting is imprecise; eigenvalue localization requires many terms
- **SDF**: Captures edge structure but not graph bipartition semantics
- **Bilateral**: Only implicit spectral awareness; no actual eigenvalue information

## Recommendations

### For Production Pipeline
```python
# Current optimal: Lanczos-30
fiedler, _ = lanczos_fiedler_gpu(L, num_iterations=30)  # 0.84 correlation
```

### For Fast Preview
```python
# Acceptable quality with 2x speedup
fiedler, _ = lanczos_fiedler_gpu(L, num_iterations=15)  # 0.48 correlation
```

### Pipeline Optimization Strategy
The real performance gain comes from **avoiding redundant Fiedler computation**, not from faster approximations:

1. **Global vs Tiled**: Use single global Fiedler instead of per-tile (current cuter version)
2. **Laplacian caching**: Reuse L across AR passes when image doesn't change much
3. **Reduced overlap**: Tiles with less overlap = fewer total Fiedler computations
4. **Adaptive iterations**: 15 iterations for fast mode, 30 for quality mode

## Implementation Notes

The `spectral_ops_fast_cuter.py` already uses Lanczos-30 as default. No changes needed to the core algorithm - focus optimization effort on:
- Reducing number of tiles processed
- Caching Laplacians across passes
- Parallelizing independent tile computations
