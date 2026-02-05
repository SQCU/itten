"""
RESNET-LIKE SPECTRAL COMPUTE SHADER - CANONICAL REFERENCE IMPLEMENTATION

This shader demonstrates the REQUIRED structure for graph-property-exposing shaders.
It is deliberately "signposted" - every architectural choice is labeled and justified.

BEHAVIOR:
  - WHERE spectral activation > mean: THICKEN contour lines (edge emphasis)
  - WHERE spectral activation < mean: COPY pixels, ROTATE toward normal, SHIFT color blue+contrast

REQUIRED COMPONENTS (all present, all labeled):
  [SPECTRAL]  - iterative_spectral_transform is THE core operation
  [RESIDUAL]  - skip connections preserve input signal
  [GATE]      - sigmoid with hand-tunable temperature
  [SWIGLU]    - gated linear unit combining two branches
  [BRANCH_A]  - high-activation path (thicken)
  [BRANCH_B]  - low-activation path (rotate+colorshift)

INPUT: Monochrome bitmap (deliberate choice - blur abuse becomes visually obvious)
OUTPUT: RGB image showing dual-branch effect
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage import sobel, binary_dilation, distance_transform_edt
from scipy.spatial import cKDTree, Voronoi

from spectral_ops_fast import (
    build_weighted_image_laplacian, iterative_spectral_transform,
    Graph, DEVICE
)


# =============================================================================
# [SPECTRAL] CORE OPERATION - This is THE non-trivial transform
# =============================================================================

def compute_spectral_activation(L, signal, theta=0.5, num_steps=8):
    """
    [SPECTRAL] The spectral transform IS the operation.

    Returns per-node activation that encodes graph-distance-sparse properties.
    This CANNOT be replicated with Gaussian blur or convolution.
    """
    activation = iterative_spectral_transform(L, signal, theta=theta, num_steps=num_steps)
    return activation


# =============================================================================
# [GATE] HAND-TUNABLE SIGMOID - Controls branch selection
# =============================================================================

def spectral_gate(activation, temperature=5.0):
    """
    [GATE] Sigmoid gate with hand-tunable temperature.

    gate → 1 where activation > mean (high spectral response)
    gate → 0 where activation < mean (low spectral response)

    Temperature controls sharpness:
      - temp=1: soft blend
      - temp=10: near-binary selection
      - temp=0.1: almost uniform 0.5
    """
    mean_act = activation.mean()
    centered = activation - mean_act
    gate = torch.sigmoid(centered * temperature)
    return gate


# =============================================================================
# [BRANCH_A] HIGH ACTIVATION PATH - Contour thickening
# =============================================================================

def branch_thicken_contours(image_np, activation_np, thickness_scale=2.0):
    """
    [BRANCH_A] Where spectral activation is HIGH, thicken contour lines.

    This identifies edges and dilates them proportionally to local activation.
    The activation controls HOW MUCH thickening happens - not a binary mask.
    """
    H, W = image_np.shape

    # Edge detection (gradient magnitude)
    gx = sobel(image_np, axis=1)
    gy = sobel(image_np, axis=0)
    edges = np.sqrt(gx**2 + gy**2)
    edges = edges / (edges.max() + 1e-8)

    # Activation-modulated thickening
    # Higher activation = more dilation iterations
    activation_norm = (activation_np - activation_np.min()) / (activation_np.max() - activation_np.min() + 1e-8)

    # Create thickness field: edges * activation
    thickness_field = edges * activation_norm * thickness_scale

    # Apply morphological thickening based on local thickness
    # We use distance transform to create smooth thickness gradients
    edge_mask = edges > 0.1
    if edge_mask.sum() > 0:
        dist = distance_transform_edt(~edge_mask)
        thickened = np.exp(-dist / (1 + thickness_field * 3))
    else:
        thickened = edges

    return thickened


# =============================================================================
# [BRANCH_B] LOW ACTIVATION PATH - Rotate + copy + color shift
# =============================================================================

def compute_normal_field(image_np, activation_np):
    """
    Compute normal direction field from high-activation regions.
    Normal = perpendicular to gradient (tangent to contours).
    """
    gx = sobel(image_np, axis=1)
    gy = sobel(image_np, axis=0)

    # Normal is perpendicular to gradient: rotate 90 degrees
    # gradient (gx, gy) -> normal (-gy, gx)
    nx = -gy
    ny = gx

    # Normalize
    mag = np.sqrt(nx**2 + ny**2) + 1e-8
    nx = nx / mag
    ny = ny / mag

    return nx, ny


def branch_rotate_copy_colorshift(image_np, activation_np, normal_x, normal_y,
                                   rotation_strength=0.3, blue_shift=0.2, contrast_boost=1.3):
    """
    [BRANCH_B] Where spectral activation is LOW:
      1. Copy pixel slices
      2. Rotate toward normal direction
      3. Shift color toward blue + higher contrast

    This creates an "off-normal displacement" effect in low-activity regions.
    """
    H, W = image_np.shape

    # Activation-based modulation (inverse - low activation = more effect)
    activation_norm = (activation_np - activation_np.min()) / (activation_np.max() - activation_np.min() + 1e-8)
    low_activation = 1.0 - activation_norm  # invert: high where original is low

    # Create coordinate grids
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # Displacement toward normal direction, scaled by low_activation
    displacement_x = normal_x * low_activation * rotation_strength * W * 0.05
    displacement_y = normal_y * low_activation * rotation_strength * H * 0.05

    # Sample coordinates (with displacement)
    sample_x = np.clip(xx + displacement_x, 0, W-1).astype(int)
    sample_y = np.clip(yy + displacement_y, 0, H-1).astype(int)

    # Copy pixels from displaced locations
    rotated = image_np[sample_y, sample_x]

    # Contrast boost (centered on 0.5)
    contrasted = 0.5 + (rotated - 0.5) * contrast_boost
    contrasted = np.clip(contrasted, 0, 1)

    # Blue shift preparation (will be applied in RGB output)
    # Store the "blue shift amount" as a separate channel
    blue_amount = low_activation * blue_shift

    return contrasted, blue_amount


# =============================================================================
# [SWIGLU] GATED COMBINATION OF BRANCHES
# =============================================================================

def swiglu_combine(branch_a, branch_b, gate, beta=1.0):
    """
    [SWIGLU] SwiGLU-like gated linear unit.

    output = gate * branch_a + (1 - gate) * silu(branch_b * beta)

    The SiLU (swish) on branch_b adds non-linearity to the low-activation path.
    beta is hand-tunable scaling.
    """
    # SiLU = x * sigmoid(x), implemented in numpy
    scaled = branch_b * beta
    sigmoid_scaled = 1.0 / (1.0 + np.exp(-scaled))
    branch_b_silu = branch_b * sigmoid_scaled

    combined = gate * branch_a + (1 - gate) * branch_b_silu
    return combined


# =============================================================================
# [RESIDUAL] SKIP CONNECTION
# =============================================================================

def residual_add(original, transformed, alpha=0.6):
    """
    [RESIDUAL] Skip connection preserving input signal.

    output = original + alpha * transformed

    alpha controls how much the transformation affects the output.
    alpha=0: pure original (no effect)
    alpha=1: full transformation added
    """
    return original + alpha * transformed


# =============================================================================
# FULL PIPELINE - The ResNet-like shader
# =============================================================================

def resnet_spectral_shader(image_np, config=None):
    """
    FULL RESNET-LIKE SPECTRAL SHADER PIPELINE

    Architecture:

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
      thicken     rotate+color                     │
          │            │                           │
          └─────┬──────┘                           │
                │                                  │
                ▼                                  │
           [SWIGLU] gated combination              │
                │                                  │
                ▼                                  │
           [RESIDUAL] ◄────────────────────────────┘
                │
                ▼
             output

    Returns: (output_rgb, intermediates_dict)
    """
    if config is None:
        config = {
            'theta': 0.5,           # Spectral position
            'num_steps': 8,         # Chebyshev iterations
            'temperature': 5.0,     # Gate sharpness
            'thickness_scale': 2.0, # Branch A intensity
            'rotation_strength': 0.3,  # Branch B displacement
            'blue_shift': 0.3,      # Branch B color shift
            'contrast_boost': 1.4,  # Branch B contrast
            'swiglu_beta': 1.0,     # SwiGLU scaling
            'residual_alpha': 0.5,  # Skip connection strength
            'edge_threshold': 0.15, # Laplacian edge weight threshold
        }

    H, W = image_np.shape
    intermediates = {}

    # =========================================================================
    # STAGE 0: Build graph Laplacian from image
    # =========================================================================
    carrier = torch.tensor(image_np, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier, edge_threshold=config['edge_threshold'])
    signal = carrier.flatten()

    intermediates['input'] = image_np.copy()

    # =========================================================================
    # STAGE 1: [SPECTRAL] Compute spectral activation
    # =========================================================================
    activation = compute_spectral_activation(L, signal, theta=config['theta'],
                                              num_steps=config['num_steps'])
    activation_np = activation.cpu().numpy().reshape(H, W)
    activation_norm = (activation_np - activation_np.min()) / (activation_np.max() - activation_np.min() + 1e-8)

    intermediates['spectral_activation'] = activation_norm.copy()

    # =========================================================================
    # STAGE 2: [GATE] Compute gate from activation
    # =========================================================================
    gate = spectral_gate(activation, temperature=config['temperature'])
    gate_np = gate.cpu().numpy().reshape(H, W)

    intermediates['gate'] = gate_np.copy()

    # =========================================================================
    # STAGE 3: [BRANCH_A] High activation path - thicken contours
    # =========================================================================
    branch_a = branch_thicken_contours(image_np, activation_np,
                                        thickness_scale=config['thickness_scale'])

    intermediates['branch_a_thicken'] = branch_a.copy()

    # =========================================================================
    # STAGE 4: [BRANCH_B] Low activation path - rotate + colorshift
    # =========================================================================
    normal_x, normal_y = compute_normal_field(image_np, activation_np)
    branch_b, blue_amount = branch_rotate_copy_colorshift(
        image_np, activation_np, normal_x, normal_y,
        rotation_strength=config['rotation_strength'],
        blue_shift=config['blue_shift'],
        contrast_boost=config['contrast_boost']
    )

    intermediates['branch_b_rotate'] = branch_b.copy()
    intermediates['blue_shift_amount'] = blue_amount.copy()
    intermediates['normal_field'] = np.stack([normal_x, normal_y], axis=-1)

    # =========================================================================
    # STAGE 5: [SWIGLU] Gated combination
    # =========================================================================
    combined = swiglu_combine(branch_a, branch_b, gate_np, beta=config['swiglu_beta'])

    intermediates['swiglu_combined'] = combined.copy()

    # =========================================================================
    # STAGE 6: [RESIDUAL] Skip connection
    # =========================================================================
    output_gray = residual_add(image_np, combined - image_np, alpha=config['residual_alpha'])
    output_gray = np.clip(output_gray, 0, 1)

    intermediates['residual_output'] = output_gray.copy()

    # =========================================================================
    # STAGE 7: Apply color shift (low activation regions get blue tint)
    # =========================================================================
    # Convert grayscale to RGB with blue shift
    output_rgb = np.zeros((H, W, 3), dtype=np.float32)

    # Base grayscale
    output_rgb[:, :, 0] = output_gray * (1 - blue_amount * 0.3)  # R reduced
    output_rgb[:, :, 1] = output_gray * (1 - blue_amount * 0.1)  # G slightly reduced
    output_rgb[:, :, 2] = output_gray * (1 + blue_amount * 0.2)  # B boosted

    output_rgb = np.clip(output_rgb, 0, 1)

    return output_rgb, intermediates


# =============================================================================
# VISUALIZATION - Show all intermediate stages
# =============================================================================

def visualize_pipeline(image_path, output_dir, config=None):
    """Generate visualization showing all pipeline stages."""

    # Load monochrome bitmap
    img = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
    H, W = img.shape

    # Run shader
    output_rgb, intermediates = resnet_spectral_shader(img, config)

    # Build visualization grid
    # Row 1: Input → Spectral Activation → Gate
    # Row 2: Branch A (thicken) → Branch B (rotate) → Blue shift amount
    # Row 3: SwiGLU combined → Residual output → Final RGB

    def to_rgb(arr):
        """Convert grayscale to RGB for display."""
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1)
        return arr

    def colormap_activation(arr):
        """Colormap for activation/gate visualization."""
        rgb = np.zeros((*arr.shape, 3))
        rgb[:, :, 0] = arr  # Red channel = value
        rgb[:, :, 2] = 1 - arr  # Blue channel = inverse
        return rgb

    row1 = np.concatenate([
        to_rgb(intermediates['input']),
        colormap_activation(intermediates['spectral_activation']),
        colormap_activation(intermediates['gate']),
    ], axis=1)

    row2 = np.concatenate([
        to_rgb(intermediates['branch_a_thicken']),
        to_rgb(intermediates['branch_b_rotate']),
        colormap_activation(intermediates['blue_shift_amount']),
    ], axis=1)

    row3 = np.concatenate([
        to_rgb(intermediates['swiglu_combined']),
        to_rgb(intermediates['residual_output']),
        output_rgb,
    ], axis=1)

    grid = np.concatenate([row1, row2, row3], axis=0)
    grid = (grid * 255).clip(0, 255).astype(np.uint8)

    # Save
    stem = Path(image_path).stem
    out_path = Path(output_dir) / f"resnet_shader_{stem}.png"
    Image.fromarray(grid).save(out_path)
    print(f"Saved: {out_path}")

    # Add labels
    labeled = Image.fromarray(grid)
    draw = ImageDraw.Draw(labeled)

    labels = [
        ["INPUT", "[SPECTRAL] Activation", "[GATE] sigmoid"],
        ["[BRANCH_A] Thicken", "[BRANCH_B] Rotate", "Blue Shift Amount"],
        ["[SWIGLU] Combined", "[RESIDUAL] Output", "FINAL RGB"],
    ]

    for row_idx, row_labels in enumerate(labels):
        for col_idx, label in enumerate(row_labels):
            x = col_idx * W + 5
            y = row_idx * H + 5
            # Draw with shadow for visibility
            draw.text((x+1, y+1), label, fill=(0, 0, 0))
            draw.text((x, y), label, fill=(255, 100, 100))

    labeled_path = Path(output_dir) / f"resnet_shader_{stem}_labeled.png"
    labeled.save(labeled_path)
    print(f"Saved: {labeled_path}")

    return output_rgb, intermediates


# =============================================================================
# CONFIG VARIATIONS - Demonstrate hand-tunability
# =============================================================================

def run_config_sweep(image_path, output_dir):
    """Run shader with different configs to show tunability."""

    img = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
    H, W = img.shape

    configs = {
        'default': {
            'theta': 0.5, 'temperature': 5.0, 'thickness_scale': 2.0,
            'rotation_strength': 0.3, 'blue_shift': 0.3, 'contrast_boost': 1.4,
            'residual_alpha': 0.5, 'edge_threshold': 0.15, 'num_steps': 8, 'swiglu_beta': 1.0,
        },
        'sharp_gate': {
            'theta': 0.5, 'temperature': 15.0, 'thickness_scale': 2.0,  # Higher temp = sharper
            'rotation_strength': 0.3, 'blue_shift': 0.3, 'contrast_boost': 1.4,
            'residual_alpha': 0.5, 'edge_threshold': 0.15, 'num_steps': 8, 'swiglu_beta': 1.0,
        },
        'soft_gate': {
            'theta': 0.5, 'temperature': 1.0, 'thickness_scale': 2.0,  # Lower temp = softer
            'rotation_strength': 0.3, 'blue_shift': 0.3, 'contrast_boost': 1.4,
            'residual_alpha': 0.5, 'edge_threshold': 0.15, 'num_steps': 8, 'swiglu_beta': 1.0,
        },
        'high_theta': {
            'theta': 0.9, 'temperature': 5.0, 'thickness_scale': 2.0,  # High freq emphasis
            'rotation_strength': 0.3, 'blue_shift': 0.3, 'contrast_boost': 1.4,
            'residual_alpha': 0.5, 'edge_threshold': 0.15, 'num_steps': 8, 'swiglu_beta': 1.0,
        },
        'low_theta': {
            'theta': 0.1, 'temperature': 5.0, 'thickness_scale': 2.0,  # Low freq emphasis
            'rotation_strength': 0.3, 'blue_shift': 0.3, 'contrast_boost': 1.4,
            'residual_alpha': 0.5, 'edge_threshold': 0.15, 'num_steps': 8, 'swiglu_beta': 1.0,
        },
        'strong_residual': {
            'theta': 0.5, 'temperature': 5.0, 'thickness_scale': 2.0,
            'rotation_strength': 0.3, 'blue_shift': 0.3, 'contrast_boost': 1.4,
            'residual_alpha': 0.9, 'edge_threshold': 0.15, 'num_steps': 8, 'swiglu_beta': 1.0,
        },
    }

    outputs = []
    for name, config in configs.items():
        output_rgb, _ = resnet_spectral_shader(img, config)
        outputs.append((name, output_rgb))

    # Build comparison grid
    n_configs = len(outputs)
    cols = 3
    rows = (n_configs + cols - 1) // cols

    grid_h = rows * H
    grid_w = cols * W
    grid = np.ones((grid_h, grid_w, 3), dtype=np.float32)

    for idx, (name, rgb) in enumerate(outputs):
        row = idx // cols
        col = idx % cols
        y0, y1 = row * H, (row + 1) * H
        x0, x1 = col * W, (col + 1) * W
        grid[y0:y1, x0:x1] = rgb

    grid = (grid * 255).clip(0, 255).astype(np.uint8)

    stem = Path(image_path).stem
    out_path = Path(output_dir) / f"resnet_shader_{stem}_configs.png"
    Image.fromarray(grid).save(out_path)

    # Labels
    labeled = Image.fromarray(grid)
    draw = ImageDraw.Draw(labeled)
    for idx, (name, _) in enumerate(outputs):
        row = idx // cols
        col = idx % cols
        x = col * W + 5
        y = row * H + 5
        draw.text((x+1, y+1), name, fill=(0, 0, 0))
        draw.text((x, y), name, fill=(255, 200, 100))

    labeled.save(Path(output_dir) / f"resnet_shader_{stem}_configs_labeled.png")
    print(f"Saved config sweep: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    # Test images (monochrome bitmaps)
    test_images = [
        "demo_output/inputs/toof.png",
        "demo_output/inputs/snek-heavy.png",
        "demo_output/inputs/1bit redraw.png",
    ]

    for img_path in test_images:
        if Path(img_path).exists():
            print(f"\n{'='*60}")
            print(f"Processing: {img_path}")
            print('='*60)

            # Full pipeline visualization
            visualize_pipeline(img_path, str(output_dir))

            # Config sweep
            run_config_sweep(img_path, str(output_dir))
