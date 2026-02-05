# Hypercontext: Spectral Shader Ops Refactor

## Context Bar
[████████████████░░░░░░░░░░░░░░░░░░░] ~45%

## Session Goal
Refactor spectral_shader_ops.py to eliminate redundancy and align with modern attention-style architecture patterns.

## Files Modified This Session
◆ spectral_shader_ops.py [5 edits] - main target
◆ spectral_ops_fast.py [3 edits] - RGB edge weights added
◆ spectral_shader_main.py [4 edits] - clean CLI created

## Key Decisions Made
- RGB L2 distance for edge weights (not grayscale)
- SwiGLU-style hard gating with nearest neighbor sampling
- No image rescaling ever (graphs have spectra)
- Hadamard compositing for layers: M_f ⊙ F + (1-M_f) ⊙ M_s ⊙ S

## Current State of spectral_shader_ops.py
~1350 lines with these entry points:
- `demo_spectral_shader_ops()` - single image demo
- `demo_two_image_transfer()` - two image demo
- `batch_random_pairs()` - batch testing

All three share ~70% similar setup/forward logic (the "triplicate forwards disease").

## Refactor Requirements

### 1. Extract `shader_forwards()`
Shared forward pass logic:
- Load/prep images → eigenvectors → Fiedler → gate → thicken → segments → composite

### 2. Remove double-indexing anti-pattern
Bad: `H, W = tensor.shape[:2]` then using H, W everywhere
Good: Keep tensor, index directly or use shape tuple

### 3. Replace convolutions with sparse/spectral ops where possible
Current convolutions in:
- `compute_local_spectral_complexity()` - Sobel + box variance via conv2d
- `dilate_high_gate_regions()` - Gaussian kernel conv

These should become sparse masked matmuls or use local spectral methods.

### 4. Target architecture style
Modern resnet+attention pattern:
- `x = x + attention(norm(x))`
- Composable blocks
- No redundant reshapes
- Compilable (torch.compile friendly)

## Open Questions
? Should shader_forwards return intermediate tensors for debugging, or just final result?
? Deprecate demo functions entirely in favor of CLI?
