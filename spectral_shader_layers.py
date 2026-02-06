"""nn.Module respecification of spectral_shader_ops_cuter.py.

This file contains the same tensor operations as spectral_shader_ops_cuter.py,
expressed in DNN vocabulary: nn.Module subclasses, register_buffer for cached
tensors, typed forward signatures, and docstrings citing the DNN literature.

The _cuter file remains canonical. This file cites it as origin.

Module inventory:
    SpectralGate           -- Fiedler sigmoid gating (cuter:207)
    SpectralThickening     -- Spatial transformer with spectral gating (cuter:91-130)
    SpectralShadow         -- Displacement + cyclic color transform (cuter:132-165)
    SpectralCrossAttention -- Discretized dot-product attention (cuter:167-201)
    SpectralShaderBlock    -- Composes gate + thickening + shadow into a single pass

Utility functions (to_grayscale, detect_contours, etc.) are imported from
spectral_shader_ops_cuter.py to avoid duplication. Only the stateful operations
that benefit from Module form are respecified here.
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Imports from the canonical _cuter source (not duplicated)
# ---------------------------------------------------------------------------
from spectral_shader_ops_cuter import (
    to_grayscale,
    detect_contours,
    adaptive_threshold,
    fill_empty_bins,
    compute_gradient_xy,
    compute_local_spectral_complexity,
    apply_color_transform,
    cyclic_color_transform,
    compute_shadow_colors,
    compute_front_colors,
    extract_segments,
    compute_segment_signature,
    match_segments,
    scatter_to_layer,
    composite_layers_hadamard,
)

# ---------------------------------------------------------------------------
# Module-level constant: Sobel kernels (registered as buffer in modules)
# ---------------------------------------------------------------------------
# cuter:14-16 - Sobel X and Y kernels, /8 normalization, stacked as (2,1,3,3)
_SOBEL_X = torch.tensor(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
) / 8.0
_SOBEL_Y = torch.tensor(
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
) / 8.0
_SOBEL_XY = torch.stack([_SOBEL_X, _SOBEL_Y]).unsqueeze(1)  # (2, 1, 3, 3)


# ===================================================================
# SpectralGate
# ===================================================================
class SpectralGate(nn.Module):
    """Fiedler sigmoid gating: soft selection of high/low spectral regions.

    Computes gate = sigmoid((fiedler - threshold) * sharpness), producing a
    [0, 1] mask that routes pixels to the high-gate path (thickening) or
    low-gate path (shadow/cross-attention).

    In GLU terms (Dauphin et al. 2017), the Fiedler field is the "gate input"
    and the image is the "value input". The sigmoid sharpness controls the
    gate hardness: at sharpness -> inf this becomes a hard binary gate
    (SwiGLU-style, Shazeer 2020).

    Source: spectral_shader_ops_cuter.py line 207
    """

    def __init__(self, sharpness: float = 10.0) -> None:
        super().__init__()
        # cuter:207 - gate_sharpness parameter
        self.sharpness = sharpness

    def forward(self, fiedler: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """Compute spectral gate from Fiedler field.

        Args:
            fiedler: (H, W) Fiedler vector field.
            threshold: scalar threshold (typically from adaptive_threshold).

        Returns:
            gate: (H, W) tensor in [0, 1].
        """
        # cuter:207 - torch.sigmoid((fiedler - thresh) * sharpness)
        return torch.sigmoid((fiedler - threshold) * self.sharpness)


# ===================================================================
# SpectralThickening
# ===================================================================
class SpectralThickening(nn.Module):
    """Structure-preserving thickening via spectrally-gated spatial transformer.

    Constructs a displacement field from the gradient of a Gaussian-convolved
    spectral selection mask, then warps the input via grid_sample. The spectral
    complexity of the Fiedler field inversely modulates the selection mask,
    preventing runaway growth at high-curvature regions under AR iteration.

    Architecture analog: Spatial Transformer Network (Jaderberg et al. 2015,
    "Spatial Transformer Networks", NeurIPS) with spectral complexity as the
    gating mechanism (cf. GLU, Dauphin et al. 2017, "Language Modeling with
    Gated Convolutional Networks", ICML; SwiGLU, Shazeer 2020, "GLU Variants
    Improve Transformer", arXiv:2002.05202).

    The Gaussian kernel sigma and selection threshold are the "learned"
    parameters in the DNN analog -- here set to fixed values that work for
    the spectral shader demonstration.

    Key properties preserved from cuter source:
        - Nearest-neighbor sampling (preserves dither patterns exactly)
        - Inverse complexity modulation (stability under AR iteration)
        - Hard gate threshold at 0.5 (SwiGLU-style, not soft attention)

    Source: spectral_shader_ops_cuter.py lines 91-130
    """

    def __init__(
        self,
        dilation_radius: int = 2,
        sigma_ratio: float = 0.6,
        fill_threshold: float = 0.1,
        modulation_strength: float = 0.3,
        rotation_angle: float = 0.03,
        high_gate_threshold: float = 0.5,
        complexity_window: int = 7,
    ) -> None:
        super().__init__()
        # cuter:97 - config parameters stored as typed attributes
        self.dilation_radius = dilation_radius
        self.sigma_ratio = sigma_ratio
        self.fill_threshold = fill_threshold
        self.modulation_strength = modulation_strength
        self.rotation_angle = rotation_angle
        self.high_gate_threshold = high_gate_threshold
        self.complexity_window = complexity_window

        # cuter:104-107 - Gaussian kernel, precomputed for the default radius.
        # register_buffer keeps it on the correct device and in state_dict.
        kernel = self._build_gaussian_kernel(dilation_radius, sigma_ratio)
        self.register_buffer("gaussian_kernel", kernel)
        self._kernel_radius = dilation_radius

    # ---- helpers ----

    @staticmethod
    def _build_gaussian_kernel(radius: int, sigma_ratio: float) -> torch.Tensor:
        """Build a 2-D Gaussian kernel normalized to max=1.

        Returns:
            (1, 1, 2*radius+1, 2*radius+1) float32 tensor.
        """
        # cuter:104-107
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        sigma = radius * sigma_ratio
        kern = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kern = kern / kern.max()
        return kern.view(1, 1, 2 * radius + 1, 2 * radius + 1)

    def _ensure_kernel(self, radius: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return the Gaussian kernel, rebuilding if radius changed at runtime."""
        if radius != self._kernel_radius:
            kernel = self._build_gaussian_kernel(radius, self.sigma_ratio).to(
                device=device, dtype=dtype
            )
            # Update buffer in-place (stays on correct device)
            self.gaussian_kernel = kernel
            self._kernel_radius = radius
            return kernel
        return self.gaussian_kernel.to(device=device, dtype=dtype)

    # ---- forward ----

    def forward(
        self,
        img: torch.Tensor,
        gate: torch.Tensor,
        fiedler: torch.Tensor,
        contours: torch.Tensor,
        effect_strength: float = 1.0,
    ) -> torch.Tensor:
        """Apply spectrally-gated thickening to the image.

        Args:
            img: (H, W, 3) RGB image in [0, 1].
            gate: (H, W) spectral gate in [0, 1] from SpectralGate.
            fiedler: (H, W) Fiedler vector field.
            contours: (H, W) boolean contour mask.
            effect_strength: global effect multiplier (cuter cfg['effect_strength']).

        Returns:
            (H, W, 3) thickened image.
        """
        H, W, _ = img.shape
        device, dtype = img.device, img.dtype

        # cuter:95-97 - resolve effective radius and config scalars
        r = max(1, int(self.dilation_radius * effect_strength))
        mod = self.modulation_strength
        fill_th = self.fill_threshold

        # cuter:99-100 - high-gate contour mask and presence flag
        hgc = contours & (gate > self.high_gate_threshold)
        has_c = hgc.any().float()

        # cuter:101-102 - inverse complexity modulation
        cplx = compute_local_spectral_complexity(fiedler, self.complexity_window)
        wt = hgc.float() * (1.0 - mod * cplx).clamp(min=0.05)

        # cuter:104-108 - Gaussian convolution of the weighted mask
        kern = self._ensure_kernel(r, device, dtype)
        sel = F.conv2d(
            F.pad(wt.unsqueeze(0).unsqueeze(0), (r, r, r, r), "constant", 0),
            kern,
        ).squeeze()

        # cuter:110-111 - fill mask: non-contour pixels where selection > threshold
        fill = (~contours) & (sel > fill_th)
        has_f = fill.any().float()

        # cuter:113-116 - gradient of selection map -> displacement direction
        mgy = F.pad(sel[1:, :] - sel[:-1, :], (0, 0, 0, 1))
        mgx = F.pad(sel[:, 1:] - sel[:, :-1], (0, 1, 0, 0))
        mm = torch.sqrt(mgx ** 2 + mgy ** 2 + 1e-8)
        dx = mgx / mm
        dy = mgy / mm

        # cuter:118-124 - build sampling grid with displacement + rotation
        yb = torch.arange(H, device=device, dtype=dtype)
        xb = torch.arange(W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yb, xb, indexing="ij")

        # cuter:120 - 1.2x displacement along gradient
        sy = grid_y + 1.2 * dy
        sx = grid_x + 1.2 * dx

        # cuter:121-123 - small rotation for visual richness
        c = math.cos(self.rotation_angle)
        s = math.sin(self.rotation_angle)
        ry = sy - grid_y
        rx = sx - grid_x
        sy = grid_y + c * ry - s * rx
        sx = grid_x + s * ry + c * rx

        # cuter:124 - normalize to [-1, 1] for grid_sample
        sgrid = torch.stack([2 * sx / (W - 1) - 1, 2 * sy / (H - 1) - 1], dim=-1).unsqueeze(0)

        # cuter:126-127 - nearest-neighbor spatial warp (preserves dither patterns)
        samp = F.grid_sample(
            img.permute(2, 0, 1).unsqueeze(0),
            sgrid,
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        )
        samp_rgb = samp.squeeze(0).permute(1, 2, 0)

        # cuter:128-129 - composite with hard gate threshold
        g3 = (fill.float() * (sel > 0.2).float()).unsqueeze(-1)
        result = torch.lerp(img, samp_rgb, g3)

        # cuter:130 - guard: no-op if no contours or no fill pixels
        return torch.lerp(img, result, has_c * has_f)


# ===================================================================
# SpectralShadow
# ===================================================================
class SpectralShadow(nn.Module):
    """Spectrally-gated shadow and front-color displacement layer.

    Computes two displacement fields from the Fiedler gradient:
        1. Shadow path: perpendicular to gradient, offset by shadow_offset.
        2. Front-color path: fixed translation.

    Displaced pixels receive cyclic color transforms (channel affine) before
    compositing under the low-gate mask.

    The Sobel kernels for gradient computation are stored as registered
    buffers (replacing the module-level _SOBEL_XY global in the cuter source).

    Key properties preserved:
        - Sobel gradient computation via single conv2d call
        - Cyclic color transform (luminance -> phase-shifted sinusoidal RGB)
        - Channel affine (per-channel scale + bias)
        - Low-gate masking: effects only where gate is low (shadow region)

    Source: spectral_shader_ops_cuter.py lines 132-165
    """

    def __init__(
        self,
        shadow_offset: float = 7.0,
        translation_strength: float = 20.0,
        effect_strength: float = 1.0,
    ) -> None:
        super().__init__()
        # cuter:137 - displacement parameters
        self.shadow_offset = shadow_offset
        self.translation_strength = translation_strength
        self.effect_strength = effect_strength

        # cuter:14-16 / Gap 1 - Sobel kernels as registered buffer
        # Shape: (2, 1, 3, 3) -- [0] is Sobel-X, [1] is Sobel-Y
        self.register_buffer("sobel_xy", _SOBEL_XY.clone())

    def forward(
        self,
        img: torch.Tensor,
        gate: torch.Tensor,
        fiedler: torch.Tensor,
        contours: torch.Tensor,
        effect_strength: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply shadow and front-color displacement to the image.

        Args:
            img: (H, W, 3) RGB image in [0, 1].
            gate: (H, W) spectral gate in [0, 1].
            fiedler: (H, W) Fiedler vector field.
            contours: (H, W) boolean contour mask.
            effect_strength: override for the instance-level effect_strength.

        Returns:
            (H, W, 3) image with shadow/front color applied.
        """
        H, W, _ = img.shape
        device, dtype = img.device, img.dtype
        eff = effect_strength if effect_strength is not None else self.effect_strength

        # cuter:137 - scale offsets by effect strength
        shad_off = self.shadow_offset * eff
        trans = self.translation_strength * eff

        # cuter:139-140 - low-gate weight and presence check
        lgw = (1.0 - gate) * contours.float()
        has_m = (lgw.sum() >= 10).float()

        # cuter:20-25 (via compute_gradient_xy) - Fiedler gradient
        # We use the registered Sobel buffers instead of the global cache.
        sobel = self.sobel_xy.to(dtype=dtype)
        f_padded = F.pad(
            fiedler.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
        )
        grad_xy = F.conv2d(f_padded, sobel)
        gx = grad_xy[0, 0]  # (H, W)
        gy = grad_xy[0, 1]  # (H, W)

        # cuter:142-144 - gradient direction and perpendicular
        gm = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        tx = gx / gm
        ty = gy / gm
        px = -ty
        py = tx

        # cuter:146-147 - coordinate grids
        yy = torch.arange(H, device=device, dtype=dtype)
        xx = torch.arange(W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

        # cuter:149-150 - displacement fields
        shad_x = grid_x + px * shad_off
        shad_y = grid_y + py * shad_off
        fr_x = grid_x + trans * 0.3
        fr_y = grid_y + trans * 0.4

        # cuter:152-153 - normalize and stack grids for batched grid_sample
        def norm(sx: torch.Tensor, sy: torch.Tensor) -> torch.Tensor:
            return torch.stack([2 * sx / (W - 1) - 1, 2 * sy / (H - 1) - 1], dim=-1)

        comb_grid = torch.stack(
            [norm(shad_x, shad_y), norm(fr_x, fr_y)], dim=0
        )  # (2, H, W, 2)

        # cuter:155-157 - batched bilinear sampling
        img4 = img.permute(2, 0, 1).unsqueeze(0).expand(2, -1, -1, -1)
        samp = F.grid_sample(
            img4, comb_grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        shad_s = samp[0].permute(1, 2, 0)  # (H, W, 3)
        fr_s = samp[1].permute(1, 2, 0)    # (H, W, 3)

        # cuter:159-160 - cyclic color transforms (channel affine)
        shad_c = compute_shadow_colors(shad_s.reshape(-1, 3), eff).reshape(H, W, 3)
        fr_c = compute_front_colors(fr_s.reshape(-1, 3), eff).reshape(H, W, 3)

        # cuter:162-165 - composite under low-gate mask
        w3 = lgw.unsqueeze(-1)
        out = torch.lerp(img, shad_c, w3 * 0.6)
        out = torch.lerp(out, fr_c, w3 * 0.8)
        return torch.lerp(img, out, has_m)


# ===================================================================
# SpectralCrossAttention
# ===================================================================
class SpectralCrossAttention(nn.Module):
    """Segment-based spectral cross-attention transfer.

    Extracts connected-component segments from low-gate contours of A and B,
    computes 4D spectral signatures per segment, matches A→B by L2 distance,
    then transplants matched B fragments to A's centroid locations with 90°
    rotation + translation. Scatter-rasterizes to shadow/front layers and
    composites via Hadamard masking.

    Architecture analog: VQ-VAE codebook lookup (van den Oord et al. 2017,
    "Neural Discrete Representation Learning", NeurIPS) where the codebook
    is the segment signature space, and the lookup produces spatial fragments
    rather than feature vectors.

    NOT torch.compile(fullgraph=True) compatible due to iterative connected
    components (same constraint as SpectralEmbedding).

    Source: total_backup_for_stupid_idiots/itten/spectral_shader_ops.py
    """

    def __init__(self, shadow_offset: float = 7.0,
                 translation_strength: float = 20.0,
                 min_pixels: int = 20, max_segments: int = 50) -> None:
        super().__init__()
        self.shadow_offset = shadow_offset
        self.translation_strength = translation_strength
        self.min_pixels = min_pixels
        self.max_segments = max_segments

    def forward(
        self,
        img_A: torch.Tensor,
        img_B: torch.Tensor,
        fiedler_A: torch.Tensor,
        fiedler_B: torch.Tensor,
        gate_A: torch.Tensor,
        contours_A: torch.Tensor,
        effect_strength: float = 1.0,
    ) -> torch.Tensor:
        """Transfer spectral structure from image B onto image A.

        Args:
            img_A: (H_A, W_A, 3) target image.
            img_B: (H_B, W_B, 3) source image (may differ in size).
            fiedler_A: (H_A, W_A) Fiedler field of A.
            fiedler_B: (H_B, W_B) Fiedler field of B.
            gate_A: (H_A, W_A) spectral gate of A in [0, 1].
            contours_A: (H_A, W_A) boolean contour mask of A.
            effect_strength: global effect multiplier.

        Returns:
            (H_A, W_A, 3) image A with matched B fragments transplanted.
        """
        from spectral_shader_ops_cuter import cross_attention_transfer
        return cross_attention_transfer(
            img_A, img_B, fiedler_A, fiedler_B, gate_A, contours_A,
            eff=effect_strength,
            shadow_offset=self.shadow_offset,
            translation_strength=self.translation_strength,
            min_pixels=self.min_pixels,
            max_segments=self.max_segments,
        )


# ===================================================================
# SpectralShaderBlock
# ===================================================================
class SpectralShaderBlock(nn.Module):
    """Single pass of the spectral shader, composing gate + thickening + shadow.

    Pipeline:
        1. fiedler -> adaptive_threshold -> SpectralGate -> gate in [0,1]
        2. image -> to_grayscale -> detect_contours
        3. high-gate path: SpectralThickening (spatial transformer warp)
        4. low-gate path: SpectralShadow (displacement + color transform)

    Config is stored at construction time via typed submodule constructors.
    Data flows through forward(); config is frozen in the module graph.

    Source: spectral_shader_ops_cuter.py lines 203-210 (shader_pass_fused)
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__()
        cfg = config or {}

        # cuter:207 - gate sharpness
        self.gate = SpectralGate(
            sharpness=cfg.get("gate_sharpness", 10.0),
        )

        # cuter:209 - thickening submodule
        self.thickening = SpectralThickening(
            dilation_radius=cfg.get("dilation_radius", 2),
            sigma_ratio=cfg.get("kernel_sigma_ratio", 0.6),
            fill_threshold=cfg.get("fill_threshold", 0.1),
            modulation_strength=cfg.get("thicken_modulation", 0.3),
            rotation_angle=cfg.get("rotation_angle", 0.03),
            high_gate_threshold=cfg.get("high_gate_threshold", 0.5),
        )

        # cuter:210 - shadow submodule
        self.shadow = SpectralShadow(
            shadow_offset=cfg.get("shadow_offset", 7.0),
            translation_strength=cfg.get("translation_strength", 20.0),
            effect_strength=cfg.get("effect_strength", 1.0),
        )

        # Store percentile for adaptive thresholding
        self.percentile = cfg.get("percentile", 40.0)
        self.effect_strength = cfg.get("effect_strength", 1.0)

    def forward(self, img: torch.Tensor, fiedler: torch.Tensor) -> torch.Tensor:
        """Run one shader pass: gate -> thicken -> shadow.

        Args:
            img: (H, W, 3) RGB image in [0, 1].
            fiedler: (H, W) Fiedler vector field.

        Returns:
            (H, W, 3) processed image.
        """
        # cuter:204-205 - preprocessing
        gray = to_grayscale(img)
        contours = detect_contours(gray)

        # cuter:206-207 - adaptive threshold and gating
        thresh = adaptive_threshold(fiedler, self.percentile)
        gate = self.gate(fiedler, thresh)

        # cuter:209 - high-gate path: spatial transformer thickening
        out = self.thickening(img, gate, fiedler, contours, self.effect_strength)

        # cuter:210 - low-gate path: shadow displacement + color transform
        out = self.shadow(out, gate, fiedler, contours, self.effect_strength)

        return out
