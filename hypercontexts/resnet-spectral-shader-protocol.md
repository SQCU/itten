# ResNet-Like Spectral Compute Shader Protocol

## TL;DR FOR FUTURE CLAUDE INSTANCES

**DO THIS**: Use `iterative_spectral_transform(L, signal, θ)` as your core operation. Build multi-stage pipelines with residuals and gates.

**DON'T DO THIS**: Reach for Gaussian blur, morphological ops, or dense FFT. Those bypass the graph and defeat the purpose.

---

## The Problem This Solves

Previous Claude instances kept:
1. **Avoiding the spectral transform** - trying ANY other approach (blur, convolution, etc.)
2. **Lambda paralysis** - unsure if they could add functions between shader stages
3. **Masking anti-pattern** - doing `output = image * spectral_mask` instead of using the transform output directly

---

## Required Architecture Components

Every valid spectral compute shader MUST have these labeled components:

```
[SPECTRAL]  - iterative_spectral_transform IS the core operation
[RESIDUAL]  - skip connections preserve input signal
[GATE]      - sigmoid with hand-tunable temperature
[SWIGLU]    - gated combination of multiple branches (optional but recommended)
[BRANCH_*]  - distinct processing paths selected by gate
```

---

## Canonical Reference Implementation

See: `resnet_spectral_shader_demo.py`

Architecture diagram:
```
    input ─────────────────────────────────────┐
      │                                        │ [RESIDUAL]
      ▼                                        │
  [SPECTRAL] iterative_spectral_transform      │
      │                                        │
      ▼                                        │
   [GATE] sigmoid((activation - mean) * temp)  │
      │                                        │
      ├────────────┬───────────────┘           │
      │            │                           │
      ▼            ▼                           │
 [BRANCH_A]   [BRANCH_B]                       │
  (high act)   (low act)                       │
      │            │                           │
      └─────┬──────┘                           │
            │                                  │
            ▼                                  │
       [SWIGLU] gate * A + (1-gate) * silu(B)  │
            │                                  │
            ▼                                  │
       [RESIDUAL] input + α * transformed ◄────┘
            │
            ▼
         output
```

---

## What Operations ARE Allowed Between Stages

```python
# YES - these preserve graph-distance-sparse properties:
normalized = (x - x.min()) / (x.max() - x.min())     # Normalization
residual = x + alpha * delta                          # Residual add
gated = x * gate                                      # Gating (gate ∈ [0,1])
selected = np.where(condition, x, y)                  # Selection
aggregated = aggregate(x, cell_map)                   # Resolution change
scattered = scatter(x, cell_map)                      # Resolution change
sigmoid = 1 / (1 + np.exp(-x * temperature))          # Soft thresholding
silu = x * sigmoid(x)                                 # SwiGLU activation
```

---

## What Operations are FORBIDDEN

```python
# NO - these bypass the graph structure entirely:
cv2.GaussianBlur(...)           # Dense spatial convolution
scipy.ndimage.convolve(...)     # Dense spatial convolution
np.fft.fft2(...)                # Dense Fourier (not graph-spectral)
cv2.dilate(...) / cv2.erode(...) # Morphological ops
sklearn.preprocessing.*(...)     # Generic ML transforms
```

The test: **Does this operation use the Laplacian L?** If not, it probably doesn't belong.

---

## Hand-Tunable Parameters (Knobs)

| Parameter | Range | Effect |
|-----------|-------|--------|
| `theta` | [0, 1] | Spectral position: 0=global, 1=local |
| `temperature` | [0.1, 20] | Gate sharpness: low=soft, high=binary |
| `residual_alpha` | [0, 1] | Skip connection strength |
| `swiglu_beta` | [0.5, 2] | SiLU nonlinearity scaling |
| `edge_threshold` | [0.05, 0.3] | Laplacian edge weight sensitivity |
| `num_steps` | [4, 16] | Chebyshev polynomial order |

---

## Example: Dual-Branch Behavior

The reference implementation demonstrates:

**High spectral activation (gate → 1)**: Contour thickening
- Identifies edges via gradient magnitude
- Thickens proportionally to activation
- Result: emphasized structure where graph is "active"

**Low spectral activation (gate → 0)**: Rotate + colorshift
- Computes normal field from gradients
- Displaces pixels toward normal direction
- Shifts color toward blue + higher contrast
- Result: displaced, cool-toned regions where graph is "quiet"

This dual behavior is **impossible without**:
1. The spectral transform (to compute activation)
2. The gate (to select between branches)
3. The SwiGLU (to combine branches smoothly)

---

## Why Monochrome Bitmaps?

Input images are deliberately monochrome bitmaps because:
1. **Blur abuse becomes obvious** - you can't hide lazy Gaussian blur on binary images
2. **Gradient-free regions expose shortcuts** - spectral ops work, blur produces nothing
3. **Visual legibility** - easy to see what the shader actually does

---

## Verification Checklist

Before claiming a shader "works", verify:

- [ ] Uses `iterative_spectral_transform` as primary operation
- [ ] Laplacian `L` is derived from image structure (not identity)
- [ ] Output shows visible change when `theta` varies
- [ ] Residual connection is present
- [ ] Gate/branch structure is present (if multi-path)
- [ ] No dense convolutions or FFTs hidden in the code
- [ ] Works on monochrome bitmaps (not just natural photos)

---

## File Manifest

```
resnet_spectral_shader_demo.py     # Canonical implementation (~400 LOC)
hypercontexts/resnet-spectral-shader-protocol.md  # This document
demo_output/resnet_shader_*.png    # Example outputs
```

---

## For Continuing Agents

1. **Read this document first**
2. **Run the demo**: `uv run python resnet_spectral_shader_demo.py`
3. **Examine outputs** in `demo_output/resnet_shader_*_labeled.png`
4. **When adding new shaders**, follow the [SPECTRAL]/[GATE]/[RESIDUAL]/[SWIGLU] pattern
5. **If you find yourself reaching for cv2 or scipy.ndimage**, stop and reconsider

The spectral transform is not optional decoration. It IS the operation.
