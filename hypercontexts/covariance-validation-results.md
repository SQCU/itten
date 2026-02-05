# 2D Covariance Validation Results

## Summary

**Overall Result: FAIL**

Testing spectral transforms against surfaces to demonstrate non-trivial 2D covariance.

### Success Criteria
- row_var (theta sensitivity) > 0.5 for at least 2 surfaces
- col_var (surface sensitivity) > 0.5 for at least 3 theta values

## Transform: `eigenvector_phase_field`

**Result: PASS**

### PSNR Matrix (dB)

| Surface | theta=0.1 | theta=0.3 | theta=0.5 | theta=0.7 | theta=0.9 |
|---------|--------|--------|--------|--------|--------|
| marble | 8.24 | 9.27 | 9.98 | 10.14 | 9.70 |
| brick | 11.75 | 12.30 | 12.10 | 11.24 | 10.03 |
| noise | 6.34 | 7.41 | 8.19 | 8.44 | 8.08 |

### Row Variances (Theta Sensitivity per Surface)

| Surface | Variance | Status |
|---------|----------|--------|
| marble | 0.4625 | - |
| brick | 0.6612 | PASS |
| noise | 0.5724 | PASS |

**Surfaces with var > 0.5: 2/3** (need >= 2)

### Column Variances (Surface Sensitivity per Theta)

| Theta | Variance | Status |
|-------|----------|--------|
| 0.1 | 5.0202 | PASS |
| 0.3 | 4.0570 | PASS |
| 0.5 | 2.5552 | PASS |
| 0.7 | 1.3227 | PASS |
| 0.9 | 0.7222 | PASS |

**Thetas with var > 0.5: 5/5** (need >= 3)

---

## Transform: `fiedler_nodal_lines`

**Result: FAIL**

### PSNR Matrix (dB)

| Surface | theta=0.1 | theta=0.3 | theta=0.5 | theta=0.7 | theta=0.9 |
|---------|--------|--------|--------|--------|--------|
| marble | 10.76 | 10.04 | 8.67 | 7.11 | 5.61 |
| brick | 10.48 | 10.16 | 9.03 | 7.55 | 6.04 |
| noise | 10.37 | 9.96 | 8.66 | 7.06 | 5.47 |

### Row Variances (Theta Sensitivity per Surface)

| Surface | Variance | Status |
|---------|----------|--------|
| marble | 3.5538 | PASS |
| brick | 2.7526 | PASS |
| noise | 3.3409 | PASS |

**Surfaces with var > 0.5: 3/3** (need >= 2)

### Column Variances (Surface Sensitivity per Theta)

| Theta | Variance | Status |
|-------|----------|--------|
| 0.1 | 0.0273 | - |
| 0.3 | 0.0072 | - |
| 0.5 | 0.0293 | - |
| 0.7 | 0.0485 | - |
| 0.9 | 0.0594 | - |

**Thetas with var > 0.5: 0/5** (need >= 3)

---

## Interpretation

### What the metrics mean

- **Row variance** measures how much PSNR changes across theta values for a fixed surface.
  - High row variance = transform is sensitive to theta parameter
  - This is expected: as theta changes, the output should change

- **Column variance** measures how much PSNR changes across surfaces for a fixed theta.
  - High column variance = transform responds differently to different surface textures
  - This is the key insight: the transform doesn't just apply the same operation regardless of input

### 2D Covariance

Having both high row and column variance demonstrates **non-separable 2D covariance**:
- The transform's behavior depends on BOTH the theta parameter AND the input surface
- This is more interesting than a transform that only depends on theta
- It shows the spectral structure of the input matters

## Findings

### `eigenvector_phase_field` - PASS

This transform demonstrates proper 2D covariance:

1. **Theta sensitivity (row variance)**: 2/3 surfaces pass (brick: 0.66, noise: 0.57)
   - PSNR changes by 2-3 dB across theta for each surface
   - The blend between magnitude and phase creates measurable theta sensitivity

2. **Surface sensitivity (column variance)**: 5/5 thetas pass (range: 0.72 to 5.02)
   - Very high variance especially at low theta values
   - The transform responds very differently to marble/brick/noise
   - At theta=0.1: brick gives 11.75 dB but noise only 6.34 dB (5.4 dB difference)

**Why it works**: The eigenvector phase field uses pairs of eigenvectors which vary based on image structure. Different surfaces produce different spectral decompositions, leading to different phase/magnitude patterns.

### `fiedler_nodal_lines` - FAIL

This transform shows high theta sensitivity but low surface sensitivity:

1. **Theta sensitivity (row variance)**: 3/3 surfaces pass (range: 2.75 to 3.55)
   - Very strong theta response
   - PSNR drops from ~10.5 dB at theta=0.1 to ~5.7 dB at theta=0.9

2. **Surface sensitivity (column variance)**: 0/5 thetas pass (all < 0.06)
   - The PSNR values across surfaces are nearly identical at each theta
   - At theta=0.5: marble=8.67, brick=9.03, noise=8.66 (only 0.4 dB spread)

**Why it fails**: The Fiedler vector partitions the image into two regions, but for all three surfaces the partition behavior is similar. The nodal lines trace different paths but the overall PSNR relationship to the original stays roughly constant. The transform is effectively "surface-blind" - it applies the same transformation intensity regardless of input texture.

## Recommendations

1. **Use `eigenvector_phase_field`** when you need transforms that respond to both theta AND input structure
2. **Consider modifying `fiedler_nodal_lines`** to incorporate more surface-dependent behavior:
   - Use higher eigenvector indices for structured vs noisy inputs
   - Weight the nodal line contribution by local image variance
   - Consider multi-scale Fiedler decomposition

## Conclusion

The `eigenvector_phase_field` transform successfully demonstrates non-trivial 2D covariance with both theta and surface sensitivity. The `fiedler_nodal_lines` transform, while highly theta-sensitive, does not differentiate between surface textures and thus fails the 2D covariance test.
