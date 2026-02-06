# TODO: _even_cuter spectral layer respecification

## What this is

Respecification of `spectral_shader_ops_cuter.py` and `spectral_ops_fast_cuter.py` into
`nn.Module` form, making the DNN-equivalent properties of each construction discoverable
to future readers (both human and Claude-like).

**Not a rewrite.** Same tensor operations, same properties, different vocabulary.
The _cuter files remain canonical. The _even_cuter files cite them as origin.

## Why this matters

The _cuter pipeline implements:
- A spatial transformer (dilate_high_gate_fused)
- Discretized cross-attention (cross_attention_transfer)
- An implicit fixed-point layer (Lanczos iteration)
- GLU-style gating (fiedler_gate + high/low path routing)
- Learned-equivalent color transforms (cyclic + channel affine)

These are expressed in shader culture (flat functions, config dicts, global buffers).
Respecifying them in DNN culture (nn.Module, register_buffer, typed forward signatures)
makes the composition space visible to readers who know the DNN literature.

Constitutional test: a haiku-class model reading `SpectralThickening` with a docstring
citing Jaderberg 2015 will propose compositions with deformable convolutions, optical flow,
displacement-field modules. The same model reading `dilate_high_gate_fused` will produce
locally-correct-but-closed output.

## Source files (DO NOT MODIFY)

| File | Lines | Role |
|------|-------|------|
| `spectral_shader_ops_cuter.py` | 270 | Fused shader operations |
| `spectral_ops_fast_cuter.py` | 264 | Minimal Lanczos + Laplacian |

## Target files (TO CREATE)

| File | Derives from | Priority |
|------|-------------|----------|
| `spectral_shader_layers.py` | `spectral_shader_ops_cuter.py` | P0 |
| `spectral_embedding_layer.py` | `spectral_ops_fast_cuter.py` | P1 |
| `spectral_shader_model.py` | composes the above | P2 |

---

## P0: `spectral_shader_layers.py`

Source: `spectral_shader_ops_cuter.py`

### Structural gaps to close

**Gap 4 (highest value): SpectralThickening as Spatial Transformer**

Source function: `dilate_high_gate_fused` (cuter lines 91-130)

What it does:
1. Gaussian kernel construction -> conv2d (the "attention map")
2. Gradient of convolved mask -> displacement field
3. grid_sample at displaced positions -> spatial warp
4. Hard-threshold gating -> SwiGLU-style gate

Module spec:
```
class SpectralThickening(nn.Module):
    """Structure-preserving thickening via spectrally-gated spatial transformer.

    Constructs a displacement field from the gradient of a Gaussian-convolved
    spectral selection mask, then warps the input via grid_sample. The spectral
    complexity of the Fiedler field inversely modulates the selection mask,
    preventing runaway growth at high-curvature regions under AR iteration.

    Architecture analog: Spatial Transformer Network (Jaderberg et al. 2015)
    with spectral complexity as the gating mechanism (cf. GLU, Dauphin et al. 2017;
    SwiGLU, Shazeer 2020).

    The Gaussian kernel sigma and selection threshold are the "learned" parameters
    in the DNN analog - here set to fixed values that work for the spectral shader
    demonstration.
    """
    def __init__(self, dilation_radius=2, sigma_ratio=0.6, fill_threshold=0.1,
                 modulation_strength=0.3, rotation_angle=0.03):
        # register_buffer for Gaussian kernel (recomputed if radius changes)
        # store config as attributes, not dict

    def forward(self, img, gate, fiedler, contours):
        # same tensor ops as dilate_high_gate_fused
        # returns thickened image
```

Properties to preserve:
- Nearest-neighbor sampling (preserves dither patterns exactly)
- Inverse complexity modulation (stability under AR iteration)
- Hard gate threshold (SwiGLU-style, not soft attention)

**Gap 5: SpectralCrossAttention as discretized dot-product attention**

Source function: `cross_attention_transfer` (cuter lines 167-201)

What it does:
1. Normalize Fiedler values to [0,1] -> spectral embedding
2. Bin assignment via floor -> hard codebook assignment
3. scatter_add for mean position per bin -> value computation
4. Lookup + grid_sample -> attention output

Module spec:
```
class SpectralCrossAttention(nn.Module):
    """Spectral correspondence via discretized attention over Fiedler embeddings.

    Q: fiedler_A normalized to bins ("what spectral position am I at?")
    K: fiedler_B normalized to bins ("what spectral positions exist in B?")
    V: mean spatial position per bin in B ("where in B is that spectral position?")

    Hard assignment (binning) replaces softmax. This is intentional sparsification:
    O(n * n_bins) instead of O(n^2). n_bins controls the attention resolution.

    Architecture analog: VQ-VAE codebook lookup (van den Oord et al. 2017) composed
    with spatial sampling. The bin-to-position lookup is a rank-1 approximation of
    the full spectral correspondence matrix phi_A @ phi_B^T.

    For soft attention variant: replace binning with softmax(fiedler_A @ fiedler_B^T),
    but this is O(n_A * n_B) and defeats the sparsification purpose.
    """
    def __init__(self, n_bins=256):
        ...

    def forward(self, img_A, img_B, fiedler_A, fiedler_B, gate_A, contours_A,
                effect_strength=1.0):
        ...
```

Properties to preserve:
- Different-size images (no rescaling - spectral properties are intrinsic)
- Forward-backward fill for empty bins (handles sparse Fiedler distributions)
- Low-gate masking for composite (only transfers where gate is low)

**Mechanical gaps (lower value, handle if time permits):**

- Gap 1: Module-level `_SOBEL_XY` global -> `register_buffer` in a GradientModule
- Gap 2: Config dict -> constructor arguments throughout
- Gap 3: Hardcoded color scales/biases -> stored as buffers with `requires_grad=False`
  (names them as "this is where a learnable parameter would go")

### Module composition

```
class SpectralShaderBlock(nn.Module):
    """Single pass of spectral shader.

    Pipeline:
        fiedler -> gate (sigmoid)
        high-gate path: SpectralThickening (spatial transformer)
        low-gate path: SpectralShadow (displacement + color transform)

    Config is stored at construction time, not passed per-call.
    Data flows through forward(); config is frozen in the module.
    """
    def __init__(self, config):
        self.gate = SpectralGate(sharpness=config.gate_sharpness)
        self.thickening = SpectralThickening(...)
        self.shadow = SpectralShadow(...)

    def forward(self, img, fiedler):
        gate = self.gate(fiedler)
        gray = to_grayscale(img)
        contours = detect_contours(gray)
        out = self.thickening(img, gate, fiedler, contours)
        out = self.shadow(out, gate, fiedler, contours)
        return out
```

---

## P1: `spectral_embedding_layer.py`

Source: `spectral_ops_fast_cuter.py`

### Gap 6: Lanczos as implicit fixed-point layer

Source functions:
- `build_weighted_image_laplacian` (cuter lines 9-41)
- `lanczos_fiedler_gpu` (cuter lines 44-101)
- `_build_multiscale_laplacian` (cuter lines 104-151)
- `_lanczos_tile` (cuter lines 154-206)
- `compute_local_eigenvectors_tiled_dither` (cuter lines 218-263)

Module spec:
```
class SpectralEmbedding(nn.Module):
    """Compute Fiedler vector via tiled Lanczos iteration.

    This is an implicit layer: the forward pass finds the fixed point
    of the eigenvalue equation L @ v = lambda * v for the 2nd smallest
    eigenvalue. The iteration count controls approximation quality.

    Architecture analog: Deep Equilibrium Models (Bai et al. 2019) -
    the output is defined implicitly as a fixed point, not as the
    result of a finite chain of explicit layers.

    For differentiable backward pass (future work): implicit differentiation
    via the resolvent d(v)/d(L) = -(L - lambda*I)^{-1} (dL) v.
    See Giles 2008 "Extended Collection of Matrix Derivative Results".

    AR cache semantics: the Fiedler of a static image doesn't change.
    In two-image mode, source embedding can be cached. Target embedding
    must be recomputed after each AR mutation step. This is analogous to
    KV-cache invalidation in autoregressive transformers: unchanged prefix
    tokens keep their KV entries; the new token requires fresh computation.
    """
    def __init__(self, tile_size=64, overlap=16, num_eigenvectors=4,
                 radii=[1,2,3,4,5,6], radius_weights=[1.0,0.6,0.4,0.3,0.2,0.1],
                 edge_threshold=0.15, lanczos_iterations=30):
        ...

    def forward(self, image):
        # returns (H, W) Fiedler vector
        # internally: tile, build Laplacian per tile, Lanczos per tile, blend
        ...
```

Properties to preserve:
- Tiled computation (memory O(tile_size^2), not O(H*W))
- Multi-radius connectivity (dither pattern awareness)
- Overlap blending (smooth transitions between tiles)
- Mean subtraction in Lanczos (deflates trivial eigenvector)
- Full reorthogonalization (numerical stability for small lambda_2)

### The AR vs depth distinction (must be architecturally explicit)

```
class SpectralShaderAR(nn.Module):
    """Autoregressive spectral shader.

    IMPORTANT DISTINCTION from deeper networks:
    - More depth = more SpectralShaderBlock layers processing the SAME input.
      The SpectralEmbedding is computed once; all blocks share it.
    - AR iteration = running the whole model on MUTATED output.
      The SpectralEmbedding must be RECOMPUTED because the input changed.

    In attention terms:
    - Depth: all layers share the same KV cache (input unchanged)
    - AR: KV cache for target is invalidated (input mutated)
    - Source KV cache (if two-image) survives (source unchanged)
    """
    def __init__(self, shader_block, embedding_layer):
        ...

    def forward(self, image, n_passes=1, source=None, source_fiedler=None):
        current = image
        for i in range(n_passes):
            fiedler = self.embedding_layer(current)  # RECOMPUTE: input changed
            current = self.shader_block(current, fiedler)
        return current
```

---

## P2: `spectral_shader_model.py`

Composes P0 and P1 into a complete model.

```
class SpectralShader(nn.Module):
    """Complete spectral shader as nn.Module.

    Single-image mode: SpectralEmbedding -> SpectralShaderBlock
    Two-image mode: SpectralEmbedding(A), SpectralEmbedding(B) -> SpectralCrossAttentionBlock
    AR mode: loop over forward(), recomputing embedding each step

    This IS a DNN model definition. The operations are the same as the
    spectral_shader_ops_cuter.py functions. The vocabulary is different:
    nn.Module protocol, register_buffer, typed forward signatures.

    The purpose is discoverability: a reader who knows the DNN literature
    can see that this model contains a spatial transformer, discretized
    cross-attention, and an implicit spectral embedding layer, and can
    propose compositions with any module that produces/consumes these
    intermediate representations.
    """
```

---

## Subagent workflow

### Agent 1: Reader + Specifier
- Reads `spectral_shader_ops_cuter.py` and `spectral_ops_fast_cuter.py`
- Reads this TODO file
- For each target module: writes the full `nn.Module` class with docstrings,
  `__init__` signature, `forward` signature, and inline comments mapping
  each operation to its _cuter source line
- Does NOT write tensor implementation (leaves `forward` body as pseudocode
  with explicit cuter line citations)
- Output: skeleton files with complete interfaces

### Agent 2: Transcriber
- Reads the skeleton files from Agent 1
- Reads the _cuter source files
- Fills in `forward` bodies by transcribing _cuter tensor ops into Module form
- Handles register_buffer for kernels, grid caching, etc.
- Preserves exact numerical behavior (same ops, same order, same dtypes)
- Output: complete .py files

### Agent 3: Runner + Debugger
- Imports both _cuter and _even_cuter versions
- Runs both on same input images
- Asserts tensor equality (or near-equality within float32 tolerance)
- Fixes any transcription errors
- Runs torch.compile on the Module versions to verify compatibility
- Output: passing tests, confirmed numerical equivalence

---

## Future work (after _even_cuter)

### Fused Triton kernel for Lanczos inner loop
The bottleneck is 30 iterations of:
- sparse matvec (L @ v)
- dot product (v^T @ w)
- vector subtract + scale
- reorthogonalization (matmul with V)
- mean subtraction
- normalization

A Triton kernel that fuses this keeps Krylov vectors in SRAM, does one kernel
launch for the full iteration. This is the performance path for Question B.

### Differentiable backward through eigendecomposition
For end-to-end training of a model that uses SpectralEmbedding:
- d(fiedler)/d(image) via chain rule through Laplacian construction + Lanczos
- Lanczos backward: implicit differentiation via resolvent
- Laplacian backward: d(L)/d(image) is straightforward (edge weights are differentiable)
- Reference: Giles 2008, Wang et al. 2019 "Backpropagation-Friendly Eigendecomposition"

### Soft attention variant of SpectralCrossAttention
Replace hard binning with:
```python
# Q = fiedler_A projected, K = fiedler_B projected
# attention = softmax(Q @ K^T / sqrt(d))
# V = spatial positions in B
# output = attention @ V -> sample B at attended positions
```
This is O(n_A * n_B) but would allow gradient flow through the correspondence.

---

## Citations for docstrings

- Jaderberg et al. 2015: "Spatial Transformer Networks" (NeurIPS)
- Dauphin et al. 2017: "Language Modeling with Gated Convolutional Networks" (ICML)
- Shazeer 2020: "GLU Variants Improve Transformer" (arXiv:2002.05202)
- van den Oord et al. 2017: "Neural Discrete Representation Learning" (NeurIPS) [VQ-VAE]
- Bai et al. 2019: "Deep Equilibrium Models" (NeurIPS)
- Giles 2008: "Extended Collection of Matrix Derivative Results for Forward and Reverse Mode AD"
- Wang et al. 2019: "Backpropagation-Friendly Eigendecomposition" (NeurIPS)
- Dhillon et al. 2007: "Weighted Graph Cuts without Eigenvectors" (IEEE TPAMI) [GRACLUS]
- Loukas 2019: "Graph Reduction with Spectral and Cut Guarantees" (JMLR)
