# Deletion Record: v7/v9/v10 Consolidation

*Session Date: 2026-02-01*

---

## Summary

Consolidated shader implementations by deleting intermediate versions that were superseded by `spectral_shader_v10_unified.py`. Kept `spectral_shader_v8.py` for contrastive study of cross-attention patterns.

---

## Deletion Order

### v7 Family (7 files, 145KB total)

These were iterative attempts at cross-attention that substituted "things that sort of do something similar" instead of actual tensor operations.

```
1. resnet_spectral_shader_v7.py      (23KB) - initial cross-attention attempt
2. resnet_spectral_shader_v7_b.py    (39KB) - bloated with debugging
3. resnet_spectral_shader_v7_b_clean.py (23KB) - cleanup attempt
4. resnet_spectral_shader_v7c.py     (7KB)  - simplified but lost features
5. resnet_spectral_shader_v7d.py     (10KB) - added back some features
6. resnet_spectral_shader_v7e.py     (14KB) - more features
7. resnet_spectral_shader_v7f.py     (15KB) - almost there
8. resnet_spectral_shader_v7g.py     (14KB) - "running code that doesn't implement spec"
```

**Why deleted**: Didn't achieve tensor-native operations. Accumulated complexity without correct semantics.

### v9 (1 file)

```
9. spectral_shader_v9_ffn.py         (17KB) - multi-head attention thickening attempt
```

**Why deleted**: Over-engineered attention mechanism that didn't improve on simpler approaches.

### v10 Intermediates (4 files)

```
10. spectral_shader_v10.py           (16KB) - initial tensor-native attempt
11. spectral_shader_v10_sweep.py     (30KB) - parameter sweep infrastructure
12. spectral_shader_v10_asymmetric.py (16KB) - asymmetric thickening experiments
13. spectral_shader_v10_shadow.py    (21KB) - energy-minimizing shadow experiments
```

**Why deleted**: Features consolidated into v10_unified. Sweep infrastructure was useful but now embedded in unified version's parameterization.

---

## Retained Files

### spectral_shader_v8.py (KEPT)

**Reason**: Best example of cross-attention codomain selection. Worth contrastive study for:
- Spectral embedding comparison (`phi_t * phi_s`)
- Pre-segmentation of sample into components
- Per-seed retrieval of best-matching component
- k-dimensional spectral signature matching

Key pattern to preserve:
```python
# Cross-attention: each target pixel attends to sample BOUNDARY structure
sample_signature = phi_s[sample_boundary_idx].mean(dim=0)  # (k,)
activation = (phi_t * sample_signature.unsqueeze(0)).sum(dim=1)  # (n_t,)
```

### spectral_shader_v10_unified.py (KEPT)

**Reason**: Consolidated implementation with:
- Parameterized gamma/bias gates
- Asymmetric thickening (inside/outside radii)
- Patch-based shadow with proper rotation
- Shadow + front layering
- Color transforms from v6

---

## What Was Learned

1. **v7 series**: Cross-attention is hard. Easy to write code that runs but doesn't implement the spec.

2. **v9**: Multi-head attention for thickening is overkill. Simple asymmetric radii achieve the effect.

3. **v10 series**:
   - Width modulation needs carrier gradient, not Fiedler gradient (too small)
   - Low gate needs NEGATIVE bias (threshold = -bias / (gamma - 1))
   - Patches must be rotated as images (grid_sample), not as coordinate lists
   - Shadow + front layering requires two scatter passes

---

## Next Step

Extend v10_unified to (target, sample) cross-attention, porting the spectral embedding retrieval from v8 into the unified tensor-native framework.

---

*Related: handoff-v10.2-asymmetric.md, handoff-v10-tensor-segments.md*
