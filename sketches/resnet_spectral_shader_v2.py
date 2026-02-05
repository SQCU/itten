"""
RESNET-LIKE SPECTRAL COMPUTE SHADER v2 - VISIBLE EFFECTS

BEHAVIOR (must be visually obvious):
  - WHERE spectral activation > mean: THICKEN existing contour pixels (dilation)
  - WHERE spectral activation < mean:
      * COPY slices of pixels
      * ROTATE copies to normal direction (orthogonal to contour)
      * COLOR SHIFT copies toward blue + high contrast

This version creates PALPABLE visual effects, not subtle modulations.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage import sobel, binary_dilation, binary_erosion, label, distance_transform_edt
from scipy.spatial import cKDTree

from spectral_ops_fast import (
    build_weighted_image_laplacian, iterative_spectral_transform,
    DEVICE
)


# =============================================================================
# [SPECTRAL] CORE OPERATION
# =============================================================================

def compute_spectral_activation(img_np, theta=0.5, edge_threshold=0.15):
    """
    [SPECTRAL] Compute per-pixel spectral activation.
    Returns activation map same size as input.
    """
    H, W = img_np.shape
    carrier = torch.tensor(img_np, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier, edge_threshold=edge_threshold)
    signal = carrier.flatten()

    activation = iterative_spectral_transform(L, signal, theta=theta, num_steps=8)
    activation_np = activation.cpu().numpy().reshape(H, W)

    # Normalize to [0, 1]
    activation_np = (activation_np - activation_np.min()) / (activation_np.max() - activation_np.min() + 1e-8)
    return activation_np


# =============================================================================
# [GATE] BINARY GATE FROM SPECTRAL ACTIVATION
# =============================================================================

def compute_gate(activation, temperature=5.0):
    """
    [GATE] Binary-ish gate: 1 where activation > mean, 0 where < mean.
    Temperature controls softness.
    """
    mean_act = activation.mean()
    centered = activation - mean_act
    gate = 1.0 / (1.0 + np.exp(-centered * temperature))
    return gate


# =============================================================================
# [BRANCH_A] HIGH ACTIVATION: THICKEN CONTOUR LINES
# =============================================================================

def extract_contour_mask(img_np, threshold=0.5):
    """Extract binary contour mask from image."""
    # For monochrome bitmaps: dark pixels are contours
    return img_np < threshold


def thicken_contours_gated(img_np, gate, max_thickness=5):
    """
    [BRANCH_A] Thicken contour lines WHERE gate is high.

    - Identifies contour pixels (dark lines in bitmap)
    - Dilates them by amount proportional to local gate value
    - Returns thickened binary mask
    """
    H, W = img_np.shape
    contour_mask = extract_contour_mask(img_np, threshold=0.5)

    # Create thickness map from gate (high gate = more thickness)
    # Quantize gate to discrete thickness levels
    thickness_levels = (gate * max_thickness).astype(int)

    # Build thickened result by iterative dilation
    result = np.zeros((H, W), dtype=np.float32)

    # For each thickness level, dilate and mask
    for t in range(1, max_thickness + 1):
        # Dilate contours by t iterations
        dilated = contour_mask.copy()
        for _ in range(t):
            dilated = binary_dilation(dilated)

        # Apply where thickness_level >= t
        mask = thickness_levels >= t
        result[mask & dilated] = 1.0

    # Also include original contours
    result[contour_mask] = 1.0

    return result


# =============================================================================
# [BRANCH_B] LOW ACTIVATION: ROTATED COPIES + COLOR SHIFT
# =============================================================================

def compute_normal_field(img_np):
    """
    Compute normal direction (perpendicular to gradient/contour).
    Returns (nx, ny) unit normal field.
    """
    gx = sobel(img_np, axis=1).astype(np.float32)
    gy = sobel(img_np, axis=0).astype(np.float32)

    # Normal is perpendicular to gradient: (gx, gy) -> (-gy, gx)
    nx = -gy
    ny = gx

    # Normalize
    mag = np.sqrt(nx**2 + ny**2) + 1e-8
    nx = nx / mag
    ny = ny / mag

    return nx, ny


def color_rotation_matrix(angle_degrees):
    """
    Build a 3x3 rotation matrix that rotates in RGB color space
    around the luminance axis (1,1,1)/sqrt(3) toward blue.

    This is a proper color space rotation, not just channel scaling.
    """
    # Rotation around the (1,1,1) axis (gray line)
    # This preserves luminance while shifting hue

    angle = np.radians(angle_degrees)
    c = np.cos(angle)
    s = np.sin(angle)

    # Rodrigues rotation formula around axis (1,1,1)/sqrt(3)
    # R = I + sin(θ)K + (1-cos(θ))K²
    # where K is the skew-symmetric matrix of the axis

    # For axis (1,1,1)/sqrt(3):
    # K = [[ 0, -1,  1],
    #      [ 1,  0, -1],
    #      [-1,  1,  0]] / sqrt(3)

    sqrt3 = np.sqrt(3)

    # Direct formula for rotation around (1,1,1):
    # Each element computed from Rodrigues formula
    m = (1 - c) / 3

    R = np.array([
        [c + m,      m - s/sqrt3, m + s/sqrt3],
        [m + s/sqrt3, c + m,      m - s/sqrt3],
        [m - s/sqrt3, m + s/sqrt3, c + m     ]
    ])

    return R


def apply_color_rotation(r, g, b, rotation_matrix, contrast_boost=1.3):
    """
    Apply color rotation matrix to RGB channels.
    Also applies contrast boost around 0.5 midpoint.
    """
    H, W = r.shape

    # Stack into (H, W, 3)
    rgb = np.stack([r, g, b], axis=-1)

    # Apply contrast boost first (around 0.5)
    rgb = 0.5 + (rgb - 0.5) * contrast_boost

    # Apply rotation: (H, W, 3) @ (3, 3).T
    rotated = np.einsum('ijk,lk->ijl', rgb, rotation_matrix)

    # Clip to valid range
    rotated = np.clip(rotated, 0, 1)

    return rotated[:, :, 0], rotated[:, :, 1], rotated[:, :, 2]


def create_rotated_copies(img_np, gate, normal_x, normal_y,
                          copy_distance=15, num_copies=3,
                          color_rotation_angle=60, contrast_boost=1.4):
    """
    [BRANCH_B] Create rotated/displaced copies WHERE gate is LOW.

    - Identifies regions where gate < 0.5 (low spectral activation)
    - Copies pixel slices from those regions
    - Displaces copies along normal direction (orthogonal to contours)
    - Applies COLOR ROTATION (not just scaling) toward blue
    - Boosts contrast
    """
    H, W = img_np.shape

    # Low activation mask (invert gate)
    low_activation = 1.0 - gate
    low_mask = low_activation > 0.5

    # Create coordinate grids
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # Output accumulator (RGB)
    output_r = np.zeros((H, W), dtype=np.float32)
    output_g = np.zeros((H, W), dtype=np.float32)
    output_b = np.zeros((H, W), dtype=np.float32)
    copy_count = np.zeros((H, W), dtype=np.float32)

    # Build color rotation matrix (rotate toward blue)
    color_rot = color_rotation_matrix(color_rotation_angle)

    # Create multiple displaced copies at different distances
    for i in range(1, num_copies + 1):
        dist = copy_distance * i / num_copies

        # Displacement along normal direction
        dx = normal_x * dist * low_activation
        dy = normal_y * dist * low_activation

        # Sample coordinates
        sample_x = np.clip(xx + dx, 0, W - 1).astype(int)
        sample_y = np.clip(yy + dy, 0, H - 1).astype(int)

        # Sample pixels from displaced locations (grayscale -> RGB)
        sampled = img_np[sample_y, sample_x]

        # Only add copies where displacement is significant
        displacement_mag = np.sqrt(dx**2 + dy**2)
        copy_mask = low_mask & (displacement_mag > 2)

        # Convert grayscale to RGB for color rotation
        sampled_r = sampled.copy()
        sampled_g = sampled.copy()
        sampled_b = sampled.copy()

        # Apply color rotation + contrast boost
        rot_r, rot_g, rot_b = apply_color_rotation(
            sampled_r, sampled_g, sampled_b,
            color_rot, contrast_boost=contrast_boost
        )

        # Accumulate with fade based on distance
        fade = 1.0 - (i - 1) / num_copies * 0.3
        output_r[copy_mask] += rot_r[copy_mask] * fade
        output_g[copy_mask] += rot_g[copy_mask] * fade
        output_b[copy_mask] += rot_b[copy_mask] * fade
        copy_count[copy_mask] += fade

    # Average where copies overlap
    copy_count = np.maximum(copy_count, 1)
    output_r /= copy_count
    output_g /= copy_count
    output_b /= copy_count

    return output_r, output_g, output_b, low_mask


# =============================================================================
# [SWIGLU] GATED COMBINATION
# =============================================================================

def swiglu_combine_rgb(img_np, branch_a_mask, branch_b_rgb, gate):
    """
    [SWIGLU] Combine branches with gated blending.

    - High gate: show thickened contours (branch A) as dark lines
    - Low gate: show rotated blue copies (branch B)
    """
    H, W = img_np.shape
    branch_b_r, branch_b_g, branch_b_b, low_mask = branch_b_rgb

    # Start with grayscale base
    output_r = img_np.copy()
    output_g = img_np.copy()
    output_b = img_np.copy()

    # Apply branch A: thickened contours shown as dark
    # Where branch_a_mask is True and gate is high, darken
    high_gate_mask = gate > 0.5
    thicken_region = (branch_a_mask > 0.5) & high_gate_mask  # Convert to bool properly
    output_r[thicken_region] = 0.05
    output_g[thicken_region] = 0.05
    output_b[thicken_region] = 0.1  # Slight blue tint even on thickened lines

    # Apply branch B: color-rotated copies where gate is low
    low_gate_mask = gate <= 0.5
    has_copy_data = (branch_b_r + branch_b_g + branch_b_b) > 0.05
    blend_mask = low_gate_mask & has_copy_data

    # Blend factor based on how low the gate is (stronger effect where gate is lower)
    blend = np.clip((0.5 - gate) * 2.0, 0, 1)  # 0 at gate=0.5, 1 at gate=0

    # SwiGLU-style: x * sigmoid(x) for the blend factor
    blend_silu = blend * (1.0 / (1.0 + np.exp(-blend * 3)))

    output_r[blend_mask] = (output_r[blend_mask] * (1 - blend_silu[blend_mask]) +
                           branch_b_r[blend_mask] * blend_silu[blend_mask])
    output_g[blend_mask] = (output_g[blend_mask] * (1 - blend_silu[blend_mask]) +
                           branch_b_g[blend_mask] * blend_silu[blend_mask])
    output_b[blend_mask] = (output_b[blend_mask] * (1 - blend_silu[blend_mask]) +
                           branch_b_b[blend_mask] * blend_silu[blend_mask])

    return np.stack([output_r, output_g, output_b], axis=-1)


# =============================================================================
# [RESIDUAL] SKIP CONNECTION
# =============================================================================

def residual_blend(original_rgb, transformed_rgb, alpha=0.7):
    """
    [RESIDUAL] Blend transformed result with original.
    """
    # Original as RGB
    if original_rgb.ndim == 2:
        original_rgb = np.stack([original_rgb] * 3, axis=-1)

    output = original_rgb * (1 - alpha) + transformed_rgb * alpha
    return np.clip(output, 0, 1)


# =============================================================================
# FULL PIPELINE
# =============================================================================

def resnet_spectral_shader_v2(img_np, config=None):
    """
    Full pipeline with visible effects.
    """
    if config is None:
        config = {
            'theta': 0.5,
            'temperature': 8.0,
            'max_thickness': 4,
            'copy_distance': 25,
            'num_copies': 5,
            'color_rotation_angle': 70,  # Degrees toward blue
            'contrast_boost': 1.5,
            'residual_alpha': 0.9,
            'edge_threshold': 0.12,
        }

    H, W = img_np.shape
    intermediates = {}

    # Store input
    intermediates['input'] = img_np.copy()

    # [SPECTRAL] Compute activation
    activation = compute_spectral_activation(
        img_np, theta=config['theta'], edge_threshold=config['edge_threshold']
    )
    intermediates['activation'] = activation.copy()

    # [GATE] Compute gate
    gate = compute_gate(activation, temperature=config['temperature'])
    intermediates['gate'] = gate.copy()

    # [BRANCH_A] Thicken contours where gate is high
    branch_a = thicken_contours_gated(img_np, gate, max_thickness=config['max_thickness'])
    intermediates['branch_a'] = branch_a.copy()

    # [BRANCH_B] Rotated copies where gate is low
    normal_x, normal_y = compute_normal_field(img_np)
    intermediates['normal_x'] = normal_x.copy()
    intermediates['normal_y'] = normal_y.copy()

    branch_b_r, branch_b_g, branch_b_b, low_mask = create_rotated_copies(
        img_np, gate, normal_x, normal_y,
        copy_distance=config['copy_distance'],
        num_copies=config['num_copies'],
        color_rotation_angle=config['color_rotation_angle'],
        contrast_boost=config['contrast_boost']
    )
    intermediates['branch_b_r'] = branch_b_r.copy()
    intermediates['branch_b_b'] = branch_b_b.copy()
    intermediates['low_mask'] = low_mask.astype(float)

    # [SWIGLU] Combine branches
    combined = swiglu_combine_rgb(
        img_np, branch_a, (branch_b_r, branch_b_g, branch_b_b, low_mask), gate
    )
    intermediates['combined'] = combined.copy()

    # [RESIDUAL] Blend with original
    original_rgb = np.stack([img_np] * 3, axis=-1)
    output = residual_blend(original_rgb, combined, alpha=config['residual_alpha'])
    intermediates['output'] = output.copy()

    return output, intermediates


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_v2(image_path, output_dir):
    """Visualize pipeline with clear intermediate stages."""

    img = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
    H, W = img.shape

    output, intermediates = resnet_spectral_shader_v2(img)

    def to_rgb(arr):
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1)
        return arr

    def colormap(arr):
        """Red-blue colormap."""
        rgb = np.zeros((*arr.shape, 3))
        rgb[:, :, 0] = arr
        rgb[:, :, 2] = 1 - arr
        return rgb

    def normal_to_rgb(nx, ny):
        """Visualize normal field as RGB."""
        rgb = np.zeros((nx.shape[0], nx.shape[1], 3))
        rgb[:, :, 0] = (nx + 1) / 2  # R = x direction
        rgb[:, :, 1] = (ny + 1) / 2  # G = y direction
        rgb[:, :, 2] = 0.5
        return rgb

    # Row 1: Input | Activation | Gate
    row1 = np.concatenate([
        to_rgb(intermediates['input']),
        colormap(intermediates['activation']),
        colormap(intermediates['gate']),
    ], axis=1)

    # Row 2: Branch A (thickened) | Normal field | Branch B blue channel
    row2 = np.concatenate([
        to_rgb(intermediates['branch_a']),
        normal_to_rgb(intermediates['normal_x'], intermediates['normal_y']),
        to_rgb(intermediates['branch_b_b']),  # Show blue channel of copies
    ], axis=1)

    # Row 3: Low mask | Combined | Final output
    row3 = np.concatenate([
        to_rgb(intermediates['low_mask']),
        intermediates['combined'],
        intermediates['output'],
    ], axis=1)

    grid = np.concatenate([row1, row2, row3], axis=0)
    grid = (grid * 255).clip(0, 255).astype(np.uint8)

    stem = Path(image_path).stem
    out_path = Path(output_dir) / f"resnet_v2_{stem}.png"
    Image.fromarray(grid).save(out_path)
    print(f"Saved: {out_path}")

    # Labels
    labeled = Image.fromarray(grid)
    draw = ImageDraw.Draw(labeled)
    labels = [
        ["INPUT", "[SPECTRAL] Activation", "[GATE]"],
        ["[BRANCH_A] Thickened", "Normal Field", "[BRANCH_B] Blue copies"],
        ["Low Activation Mask", "[SWIGLU] Combined", "FINAL OUTPUT"],
    ]
    for ri, row_labels in enumerate(labels):
        for ci, lbl in enumerate(row_labels):
            x, y = ci * W + 5, ri * H + 5
            draw.text((x+1, y+1), lbl, fill=(0, 0, 0))
            draw.text((x, y), lbl, fill=(255, 255, 0))

    labeled.save(Path(output_dir) / f"resnet_v2_{stem}_labeled.png")
    print(f"Saved labeled")

    return output, intermediates


def run_theta_sweep(image_path, output_dir):
    """Show how theta changes the gate and thus the effect."""

    img = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
    H, W = img.shape

    thetas = [0.1, 0.3, 0.5, 0.7, 0.9]
    outputs = []

    for theta in thetas:
        config = {
            'theta': theta,
            'temperature': 8.0,
            'max_thickness': 4,
            'copy_distance': 25,
            'num_copies': 5,
            'color_rotation_angle': 70,
            'contrast_boost': 1.5,
            'residual_alpha': 0.9,
            'edge_threshold': 0.12,
        }
        output, _ = resnet_spectral_shader_v2(img, config)
        outputs.append(output)

    # Build grid
    row = np.concatenate(outputs, axis=1)
    row = (row * 255).clip(0, 255).astype(np.uint8)

    stem = Path(image_path).stem
    out_path = Path(output_dir) / f"resnet_v2_{stem}_theta_sweep.png"
    Image.fromarray(row).save(out_path)

    # Labels
    labeled = Image.fromarray(row)
    draw = ImageDraw.Draw(labeled)
    for i, theta in enumerate(thetas):
        x = i * W + 5
        draw.text((x+1, 6), f"θ={theta}", fill=(0, 0, 0))
        draw.text((x, 5), f"θ={theta}", fill=(255, 255, 0))

    labeled.save(Path(output_dir) / f"resnet_v2_{stem}_theta_sweep_labeled.png")
    print(f"Saved theta sweep: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

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

            visualize_v2(img_path, str(output_dir))
            run_theta_sweep(img_path, str(output_dir))
