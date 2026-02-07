"""Complete spectral shader model — P2 composition of P0 and P1.

This file composes the layer modules from P0 (spectral_shader_layers.py) and
P1 (spectral_embedding_layer.py) into a single nn.Module that can be used as
a DNN model definition. The operations are identical to spectral_shader_main.py
and spectral_shader_ops_cuter.py; the vocabulary is different: nn.Module
protocol, typed forward signatures, and composable submodule graph.

The purpose is discoverability. A reader who knows the DNN literature can see
from the module inventory below exactly what this model contains, and can
propose compositions with any module that produces or consumes these
intermediate representations.

DNN analogs table (from TODO_EVEN_CUTER.md)
-------------------------------------------
+---------------------------+------------------------------------------+-----------------------------+
| This module               | DNN analog                               | Citation                    |
+---------------------------+------------------------------------------+-----------------------------+
| SpectralEmbedding         | Implicit / fixed-point layer             | Bai et al. 2019 (DEQ)      |
| SpectralGate              | GLU / SwiGLU gate                        | Dauphin 2017, Shazeer 2020 |
| SpectralThickening        | Spatial Transformer Network              | Jaderberg et al. 2015      |
| SpectralShadow            | Displacement + channel affine            | --                          |
| SpectralCrossAttention    | VQ-VAE codebook + spatial sampling       | van den Oord et al. 2017   |
| SpectralShaderBlock       | Single residual-style block              | --                          |
| SpectralShaderAR          | Autoregressive wrapper (KV invalidation) | --                          |
| SpectralShader (this)     | Complete model                           | --                          |
+---------------------------+------------------------------------------+-----------------------------+

Module inventory
----------------
    SpectralShader           -- Complete model. Entry point for readers.
    SpectralCrossAttentionBlock -- Two-image block: thickening on A + cross-attention from B.

Usage
-----
    from spectral_shader_model import SpectralShader

    # From the default config dict (backward-compatible with spectral_shader_main.py):
    model = SpectralShader.from_config(DEFAULT_CONFIG)

    # Single-image, single pass:
    result, intermediates = model(image_a)

    # Two-image cross-attention, 4 AR passes with decay:
    result, intermediates = model(image_a, image_b=image_b, n_passes=4, decay=0.75)
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from spectral_shader_layers import (
    SpectralCrossAttention,
    SpectralGate,
    SpectralShaderBlock,
    SpectralShadow,
    SpectralThickening,
    adaptive_threshold,
    detect_contours,
    to_grayscale,
)
from spectral_embedding_layer import SpectralEmbedding, SpectralShaderAR


# ===================================================================
# SpectralCrossAttentionBlock
# ===================================================================
class SpectralCrossAttentionBlock(nn.Module):
    """Two-image shader block: thickening on A, then cross-attention from B.

    This mirrors the two_image_shader_pass() function in
    spectral_shader_ops_cuter.py (line 237-244). The pipeline is:

        1. Compute gate and contours for image A (via SpectralGate).
        2. Apply SpectralThickening to A (high-gate path, spatial transformer).
        3. Apply SpectralCrossAttention: transfer spectral structure from B
           into A's low-gate regions (discretized attention lookup).

    The shadow displacement path (SpectralShadow) is NOT applied in two-image
    mode — the cross-attention transfer replaces it. This matches the cuter
    source: two_image_shader_pass calls dilate_high_gate_fused then
    cross_attention_transfer, skipping apply_low_gate_fused.

    This block accepts the same (image, fiedler, source=, source_fiedler=)
    signature expected by SpectralShaderAR for cross-attention mode.

    Source: spectral_shader_ops_cuter.py lines 237-244
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__()
        cfg = config or {}

        self.gate = SpectralGate(
            sharpness=cfg.get("gate_sharpness", 10.0),
        )
        self.thickening = SpectralThickening(
            dilation_radius=cfg.get("dilation_radius", 2),
            sigma_ratio=cfg.get("kernel_sigma_ratio", 0.6),
            fill_threshold=cfg.get("fill_threshold", 0.1),
            modulation_strength=cfg.get("thicken_modulation", 0.3),
            rotation_angle=cfg.get("rotation_angle", 0.03),
            high_gate_threshold=cfg.get("high_gate_threshold", 0.5),
        )
        self.cross_attention = SpectralCrossAttention(
            shadow_offset=cfg.get("shadow_offset", 7.0),
            translation_strength=cfg.get("translation_strength", 20.0),
            min_pixels=cfg.get("min_segment_pixels", 20),
            max_segments=cfg.get("max_segments", 50),
        )

        self.percentile = cfg.get("percentile", 40.0)
        self.effect_strength = cfg.get("effect_strength", 1.0)

    def forward(
        self,
        img: torch.Tensor,
        fiedler: torch.Tensor,
        source: Optional[torch.Tensor] = None,
        source_fiedler: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply thickening + cross-attention transfer.

        Args:
            img: (H, W, 3) target image A in [0, 1].
            fiedler: (H, W) Fiedler vector of A.
            source: (H_B, W_B, 3) source image B. Required.
            source_fiedler: (H_B, W_B) Fiedler vector of B. Required.

        Returns:
            (H, W, 3) image A with thickening applied and spectral
            structure transferred from B.
        """
        if source is None or source_fiedler is None:
            raise ValueError(
                "SpectralCrossAttentionBlock requires both source and "
                "source_fiedler. For single-image mode, use SpectralShaderBlock."
            )

        # cuter:239-241 — gate and contours for A
        gray_A = to_grayscale(img)
        contours_A = detect_contours(gray_A)
        thresh_A = adaptive_threshold(fiedler, self.percentile)
        gate_A = self.gate(fiedler, thresh_A)

        # cuter:243 — high-gate thickening (spatial transformer on A)
        out = self.thickening(
            img, gate_A, fiedler, contours_A, self.effect_strength
        )

        # cuter:244 — cross-attention transfer from B into A's low-gate regions
        out = self.cross_attention(
            out,
            source,
            fiedler,
            source_fiedler,
            gate_A,
            contours_A,
            self.effect_strength,
        )

        return out


# ===================================================================
# SpectralShader — complete model
# ===================================================================
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

    Depth vs AR iteration
    ---------------------
    - **Depth** = more SpectralShaderBlock layers processing the SAME input.
      The SpectralEmbedding is computed once; all blocks share it. The
      Laplacian eigenvectors are invariant because the input is unchanged.
      Analog: a multi-layer transformer where all layers share one KV cache.

    - **AR iteration** = running the whole model on the MUTATED output.
      The SpectralEmbedding MUST be recomputed because the input changed.
      The Laplacian is a function of pixel values; when pixels change, the
      graph changes, and the Fiedler vector changes with it.
      Analog: autoregressive generation where each new token invalidates
      the KV cache for subsequent positions.

    In two-image mode, the source Fiedler is computed once and cached
    across all AR passes (the source image is never mutated). This is
    the encoder-decoder attention pattern: encoder KV cache (source) is
    computed once; decoder KV cache (target) grows/invalidates per step.

    Config compatibility
    --------------------
    The ``from_config()`` class method accepts the same dict format as
    ``spectral_shader_main.py``'s ``DEFAULT_CONFIG``, extracting the
    relevant keys for each submodule. This allows drop-in replacement
    of the procedural pipeline with the Module-based one.

    Submodule graph
    ---------------
    ::

        SpectralShader
        +-- embedding: SpectralEmbedding (implicit fixed-point layer)
        +-- self_shader: SpectralShaderBlock (single-image: gate + thicken + shadow)
        |   +-- gate: SpectralGate
        |   +-- thickening: SpectralThickening (spatial transformer)
        |   +-- shadow: SpectralShadow (displacement + color transform)
        +-- cross_shader: SpectralCrossAttentionBlock (two-image: thicken + cross-attn)
        |   +-- gate: SpectralGate
        |   +-- thickening: SpectralThickening (spatial transformer)
        |   +-- cross_attention: SpectralCrossAttention (VQ-VAE codebook lookup)
        +-- ar_self: SpectralShaderAR (wraps self_shader for AR iteration)
        +-- ar_cross: SpectralShaderAR (wraps cross_shader for AR iteration)
    """

    def __init__(
        self,
        embedding: SpectralEmbedding,
        self_shader: SpectralShaderBlock,
        cross_shader: SpectralCrossAttentionBlock,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # Mixed-precision: compute_dtype controls Zone B (shader ops, image I/O).
        # Zone A (Lanczos iteration inside SpectralEmbedding) always runs float32.
        # The Fiedler vector exits Zone A as float32 and is cast to compute_dtype
        # at the boundary. All downstream shader ops inherit compute_dtype.
        # Default float32 = no-op (bit-exact with prior behavior).
        self.compute_dtype = compute_dtype

        # Core submodules
        self.embedding = embedding
        self.self_shader = self_shader
        self.cross_shader = cross_shader

        # AR wrappers — compose embedding + shader block for multi-pass.
        # These use SpectralShaderAR which handles Fiedler recomputation
        # at each AR step (KV-cache invalidation semantics).
        self.ar_self = SpectralShaderAR(self_shader, embedding)
        self.ar_cross = SpectralShaderAR(cross_shader, embedding)

    @classmethod
    def from_config(cls, config: Dict) -> "SpectralShader":
        """Construct a SpectralShader from a config dict.

        Accepts the same format as ``spectral_shader_main.py``'s
        ``DEFAULT_CONFIG``. Keys are dispatched to the appropriate
        submodule constructors.

        Args:
            config: Configuration dictionary. All keys are optional;
                defaults match DEFAULT_CONFIG.

        Returns:
            Fully constructed SpectralShader module.

        Example::

            from spectral_shader_main import DEFAULT_CONFIG
            model = SpectralShader.from_config(DEFAULT_CONFIG)
        """
        cfg = config.copy() if config else {}

        # Mixed-precision dtype for Zone B (shader ops).
        # Accepts torch.dtype or string ("float32", "bfloat16").
        compute_dtype_raw = cfg.get("compute_dtype", torch.float32)
        if isinstance(compute_dtype_raw, str):
            compute_dtype = getattr(torch, compute_dtype_raw)
        else:
            compute_dtype = compute_dtype_raw

        # P1: SpectralEmbedding — implicit layer (Lanczos iteration)
        # output_dtype = compute_dtype: Fiedler exits Zone A as float32,
        # cast to compute_dtype at the boundary.
        embedding = SpectralEmbedding(
            tile_size=cfg.get("tile_size", 64),
            overlap=cfg.get("overlap", 16),
            num_eigenvectors=cfg.get("num_eigenvectors", 4),
            radii=cfg.get("radii", [1, 2, 3, 4, 5, 6]),
            radius_weights=cfg.get("radius_weights", [1.0, 0.6, 0.4, 0.3, 0.2, 0.1]),
            edge_threshold=cfg.get("edge_threshold", 0.15),
            lanczos_iterations=cfg.get("lanczos_iterations", 30),
            output_dtype=compute_dtype,
        )

        # P0: SpectralShaderBlock — single-image pipeline (gate + thicken + shadow)
        self_shader = SpectralShaderBlock(cfg)

        # P0 + cross-attention: two-image pipeline (thicken + cross-attention)
        cross_shader = SpectralCrossAttentionBlock(cfg)

        return cls(embedding, self_shader, cross_shader, compute_dtype=compute_dtype)

    def forward(
        self,
        image_a: torch.Tensor,
        image_b: Optional[torch.Tensor] = None,
        n_passes: int = 1,
        decay: float = 1.0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the spectral shader.

        Dispatches to single-image or two-image mode based on whether
        ``image_b`` is provided. Supports multi-pass AR iteration with
        optional effect-strength decay.

        Single-image mode (image_b is None):
            Each AR pass: embed(A) -> SpectralShaderBlock(A, fiedler_A) -> A'

        Two-image mode (image_b is provided):
            Source embedding computed once (cached).
            Each AR pass: embed(A) -> SpectralCrossAttentionBlock(A, fiedler_A, B, fiedler_B) -> A'

        Args:
            image_a: (H, W, 3) target/primary image in [0, 1].
            image_b: (H_B, W_B, 3) optional source image for cross-attention.
                May differ in size from image_a.
            n_passes: Number of autoregressive passes. Default 1.
                Each pass recomputes the Fiedler vector of the mutated target.
            decay: Effect-strength decay per pass. Default 1.0 (no decay).
                After pass i, effect_strength is multiplied by decay.
                Applied by scaling the effect_strength config in the shader
                blocks. A value of 0.75 matches spectral_shader_main.py
                default behavior.

        Returns:
            Tuple of:
                - final: (H, W, 3) output image after all passes.
                - intermediates: list of (H, W, 3) images, one per pass.
                  intermediates[-1] == final.
        """
        intermediates: List[torch.Tensor] = []

        # Zone B entry: cast images to compute_dtype (bfloat16 or float32).
        # The embedding layer internally runs Lanczos in float32 (Zone A)
        # and casts its output to compute_dtype via output_dtype.
        image_a = image_a.to(dtype=self.compute_dtype)

        if image_b is not None:
            image_b = image_b.to(dtype=self.compute_dtype)
            # Two-image mode: cross-attention transfer
            # Source embedding is computed ONCE — source is never mutated.
            source_fiedler = self.embedding(image_b)

            current = image_a
            current_effect = self.cross_shader.effect_strength
            original_effect = current_effect

            for _i in range(n_passes):
                # Set current effect strength (decays each pass)
                self.cross_shader.effect_strength = current_effect
                self.cross_shader.thickening.modulation_strength = (
                    self.cross_shader.thickening.modulation_strength
                )

                # RECOMPUTE target Fiedler — input was mutated by previous pass
                target_fiedler = self.embedding(current)

                # Apply cross-attention block
                current = self.cross_shader(
                    current,
                    target_fiedler,
                    source=image_b,
                    source_fiedler=source_fiedler,
                )

                intermediates.append(current.clone())

                # Decay effect strength for next pass
                if decay < 1.0:
                    current_effect = current_effect * decay

            # Restore original effect strength (module should be stateless
            # across calls)
            self.cross_shader.effect_strength = original_effect

            # Zone B exit: cast back to float32 for I/O (PIL/save_image expect float32).
            return current.float(), [i.float() for i in intermediates]

        else:
            # Single-image mode: self-shading
            current = image_a
            current_effect = self.self_shader.effect_strength
            original_effect = current_effect

            for _i in range(n_passes):
                # Set current effect strength
                self.self_shader.effect_strength = current_effect
                self.self_shader.shadow.effect_strength = current_effect

                # RECOMPUTE Fiedler — input was mutated by previous pass
                fiedler = self.embedding(current)

                # Apply shader block
                current = self.self_shader(current, fiedler)

                intermediates.append(current.clone())

                # Decay effect strength
                if decay < 1.0:
                    current_effect = current_effect * decay

            # Restore original effect strength
            self.self_shader.effect_strength = original_effect
            self.self_shader.shadow.effect_strength = original_effect

        # Zone B exit: cast back to float32 for I/O (PIL/save_image expect float32).
        return current.float(), [i.float() for i in intermediates]

    def forward_single_pass(
        self,
        image: torch.Tensor,
        image_b: Optional[torch.Tensor] = None,
        fiedler: Optional[torch.Tensor] = None,
        fiedler_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a single shader pass with optional pre-computed Fiedler vectors.

        This is the lower-level API for callers who manage their own
        embedding computation and AR loop. Useful when:
        - The Fiedler vector was computed externally (e.g., from a
          different Laplacian or with different parameters).
        - The caller wants to inspect/modify the Fiedler between passes.
        - Performance: avoid recomputing the Fiedler when the image
          has not changed.

        Args:
            image: (H, W, 3) target image in [0, 1].
            image_b: (H_B, W_B, 3) optional source image.
            fiedler: (H, W) pre-computed Fiedler of image. If None,
                computed via self.embedding.
            fiedler_b: (H_B, W_B) pre-computed Fiedler of image_b. If None
                and image_b is provided, computed via self.embedding.

        Returns:
            (H, W, 3) processed image.
        """
        # Zone B entry: cast to compute_dtype
        image = image.to(dtype=self.compute_dtype)

        if fiedler is None:
            fiedler = self.embedding(image)
        else:
            fiedler = fiedler.to(dtype=self.compute_dtype)

        if image_b is not None:
            image_b = image_b.to(dtype=self.compute_dtype)
            if fiedler_b is None:
                fiedler_b = self.embedding(image_b)
            else:
                fiedler_b = fiedler_b.to(dtype=self.compute_dtype)
            # Zone B exit: cast back to float32
            return self.cross_shader(
                image, fiedler, source=image_b, source_fiedler=fiedler_b
            ).float()
        else:
            return self.self_shader(image, fiedler).float()

    def __repr__(self) -> str:
        """Readable summary of the model graph."""
        lines = [
            f"{self.__class__.__name__}(",
            f"  compute_dtype={self.compute_dtype},",
            f"  embedding: {self.embedding.__class__.__name__}("
            f"tile_size={self.embedding.tile_size}, "
            f"overlap={self.embedding.overlap}, "
            f"lanczos_iter={self.embedding.lanczos_iterations}, "
            f"output_dtype={self.embedding.output_dtype})",
            f"  self_shader: {self.self_shader.__class__.__name__}("
            f"gate_sharpness={self.self_shader.gate.sharpness}, "
            f"effect={self.self_shader.effect_strength})",
            f"  cross_shader: {self.cross_shader.__class__.__name__}("
            f"max_segments={self.cross_shader.cross_attention.max_segments}, "
            f"effect={self.cross_shader.effect_strength})",
            ")",
        ]
        return "\n".join(lines)
