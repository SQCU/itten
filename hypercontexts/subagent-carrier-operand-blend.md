# Subagent Handoff: True Carrier-Operand Spectral Blend

## Problem Statement
Current transforms are self-convolutions - texture transformed with itself.
We need fn(carrier, operand) that embeds operand's "spirit" into carrier's structure.

## The Goal
Create a transform where:
- Carrier provides spatial structure (where things go)
- Operand provides texture pattern (what gets embedded)
- Result shows BOTH signatures inextricably combined
- Theta controls the blend depth / spectral coupling strength

## From Compendium (Transform 12)
```python
def carrier_operand_spectral_blend(carrier, operand, blend_strength=0.5):
    """Blend operand into carrier using spectral structure."""
    # Build carrier's spectral representation
    L = build_weighted_image_laplacian(carrier)
    eigenvectors = lanczos_k_eigenvectors(L, num_eigenvectors=8)

    # Project operand onto carrier's eigenvectors
    projected_coeffs = eigenvectors.T @ operand.flatten()

    # Reconstruct with weighted eigenvectors (favor low frequencies)
    weights = exp(-arange(k) / 3)
    weighted_coeffs = projected_coeffs * weights

    blended = eigenvectors @ weighted_coeffs
    return (1 - blend_strength) * carrier + blend_strength * blended
```

## What to Implement

Create `/home/bigboi/itten/texture/blend.py` with:

1. `spectral_embed(carrier, operand, theta)` - Embed operand into carrier's spectral basis
   - theta=0: operand dominates, carrier structure faint
   - theta=0.5: balanced blend
   - theta=1: carrier dominates, operand trace faint

2. `spectral_warp_embed(carrier, operand, theta)` - Use carrier's eigenvector gradients to WARP operand
   - The operand pattern deforms along carrier's spectral flow
   - Should show visible warping of operand features

3. Measure spatial covariance change:
   - For each theta, compute spatial autocorrelation of result
   - The autocorrelation structure should CHANGE with theta (not just intensity)

## Test Configuration
- Carrier: checkerboard (structured)
- Operand: noise (high-frequency)
- Theta sweep: [0.1, 0.3, 0.5, 0.7, 0.9]
- Measure: spatial autocorrelation at lag=[1,2,4,8,16] pixels

## Success Criteria
- Visual inspection shows BOTH carrier and operand signatures
- Spatial autocorrelation structure changes with theta (not just scales)
- PSNR(carrier, result) and PSNR(operand, result) both show theta dependence

## Output
- `/home/bigboi/itten/texture/blend.py`
- `/home/bigboi/itten/hypercontexts/carrier-operand-blend-results.md`
- Demo renders to `/home/bigboi/itten/demo_output/carrier_operand/`
