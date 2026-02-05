"""
RESNET-LIKE SPECTRAL COMPUTE SHADER v3 - ACTUAL VISIBLE EFFECTS

Fixes from v2:
1. Gate uses FIEDLER VECTOR SIGN - creates actual bipartition, not uniform activation
2. Branch B creates ROTATED patches, not just displaced pixels
3. Color rotation is more aggressive

BEHAVIOR:
  - Fiedler+ regions (one side of graph cut): THICKEN contours
  - Fiedler- regions (other side): COPY patches, ROTATE 90°, SHIFT to blue
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage import sobel, binary_dilation, rotate as scipy_rotate
from scipy.ndimage import uniform_filter, map_coordinates

from spectral_ops_fast import (
    build_weighted_image_laplacian, iterative_spectral_transform,
    lanczos_k_eigenvectors, DEVICE
)


# =============================================================================
# [SPECTRAL] FIEDLER VECTOR - Creates meaningful bipartition
# =============================================================================

def compute_fiedler_gate(img_np, edge_threshold=0.12):
    """
    [SPECTRAL] Use Fiedler vector (2nd eigenvector) for gating.

    The Fiedler vector naturally bipartitions the graph - it's THE spectral
    operation for finding the "natural cut" of the image structure.

    Returns:
        fiedler: The Fiedler vector reshaped to image
        gate: Soft gate based on Fiedler sign (1 where positive, 0 where negative)
    """
    H, W = img_np.shape
    carrier = torch.tensor(img_np, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier, edge_threshold=edge_threshold)

    # Get Fiedler vector (2nd smallest eigenvector)
    eigenvectors, eigenvalues = lanczos_k_eigenvectors(L, num_eigenvectors=2, num_iterations=50)

    # Fiedler is the 2nd eigenvector (index 1, since index 0 is constant)
    # lanczos_k_eigenvectors returns numpy arrays already
    fiedler = eigenvectors[:, -1].reshape(H, W)

    # Normalize
    fiedler = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-8)

    # Gate: soft threshold around 0.5 (midpoint after normalization)
    temperature = 10.0
    gate = 1.0 / (1.0 + np.exp(-(fiedler - 0.5) * temperature))

    return fiedler, gate


def compute_spectral_activation(img_np, theta=0.5, edge_threshold=0.12):
    """
    Also compute standard spectral activation for comparison/combination.
    """
    H, W = img_np.shape
    carrier = torch.tensor(img_np, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier, edge_threshold=edge_threshold)
    signal = carrier.flatten()

    activation = iterative_spectral_transform(L, signal, theta=theta, num_steps=8)
    activation_np = activation.cpu().numpy().reshape(H, W)
    activation_np = (activation_np - activation_np.min()) / (activation_np.max() - activation_np.min() + 1e-8)

    return activation_np


# =============================================================================
# [BRANCH_A] HIGH GATE (Fiedler+): THICKEN CONTOURS
# =============================================================================

def extract_contours(img_np, threshold=0.5):
    """Extract contour mask (dark pixels in bitmap)."""
    return img_np < threshold


def thicken_contours_in_region(img_np, gate, max_thickness=5):
    """
    [BRANCH_A] Thicken contour lines where gate > 0.5

    Returns a mask of thickened contours.
    """
    H, W = img_np.shape
    contours = extract_contours(img_np)

    # Thickness proportional to gate value
    thickness_map = (gate * max_thickness).astype(int)

    result = np.zeros((H, W), dtype=np.float32)

    # Build progressively thickened versions
    dilated = contours.copy()
    for t in range(1, max_thickness + 1):
        dilated = binary_dilation(dilated)
        # Apply where thickness_map >= t AND gate > 0.5
        apply_mask = (thickness_map >= t) & (gate > 0.5)
        result[apply_mask & dilated] = 1.0

    # Always include original contours
    result[contours] = 1.0

    return result


# =============================================================================
# [BRANCH_B] LOW GATE (Fiedler-): ROTATED COPIES + COLOR SHIFT
# =============================================================================

def compute_local_orientation(img_np, window=7):
    """
    Compute local contour orientation using structure tensor.
    Returns angle field in radians.
    """
    gx = sobel(img_np, axis=1).astype(np.float64)
    gy = sobel(img_np, axis=0).astype(np.float64)

    # Structure tensor components
    Ixx = uniform_filter(gx * gx, size=window)
    Iyy = uniform_filter(gy * gy, size=window)
    Ixy = uniform_filter(gx * gy, size=window)

    # Orientation from structure tensor (perpendicular to gradient)
    angle = 0.5 * np.arctan2(2 * Ixy, Iyy - Ixx)

    return angle


def create_rotated_patch_copies(img_np, gate, patch_size=11, rotation_angle=90,
                                 sample_stride=8):
    """
    [BRANCH_B] Create ACTUALLY ROTATED copies of local patches.

    Where gate < 0.5 (Fiedler-):
    1. Extract local patch
    2. Rotate it 90 degrees (orthogonal)
    3. Place rotated patch back

    This creates visible orthogonal structures.
    """
    H, W = img_np.shape
    half = patch_size // 2

    # Output buffer
    output = img_np.copy()
    contribution_count = np.ones((H, W), dtype=np.float32)

    # Only process where gate is low (Fiedler-)
    low_gate_mask = gate < 0.5

    # Sample positions on a grid (for efficiency)
    ys = np.arange(half, H - half, sample_stride)
    xs = np.arange(half, W - half, sample_stride)

    for cy in ys:
        for cx in xs:
            # Check if this position is in low-gate region
            if not low_gate_mask[cy, cx]:
                continue

            # Extract patch
            patch = img_np[cy - half:cy + half + 1, cx - half:cx + half + 1]

            if patch.shape != (patch_size, patch_size):
                continue

            # Rotate patch 90 degrees
            rotated_patch = np.rot90(patch, k=1)  # 90 degrees counterclockwise

            # Blend factor based on how low the gate is
            blend = (0.5 - gate[cy, cx]) * 2  # 0 at gate=0.5, 1 at gate=0
            blend = min(max(blend, 0), 1)

            # Place rotated patch back
            y0, y1 = cy - half, cy + half + 1
            x0, x1 = cx - half, cx + half + 1

            output[y0:y1, x0:x1] = (output[y0:y1, x0:x1] * (1 - blend * 0.7) +
                                    rotated_patch * blend * 0.7)
            contribution_count[y0:y1, x0:x1] += blend * 0.3

    # Normalize by contribution
    output = output / np.maximum(contribution_count, 1)

    return output


def color_rotation_matrix(angle_degrees):
    """
    Rotation matrix around the gray axis (1,1,1) in RGB space.
    Positive angle rotates toward blue.
    """
    angle = np.radians(angle_degrees)
    c = np.cos(angle)
    s = np.sin(angle)
    sqrt3 = np.sqrt(3)

    m = (1 - c) / 3

    R = np.array([
        [c + m,       m - s/sqrt3, m + s/sqrt3],
        [m + s/sqrt3, c + m,       m - s/sqrt3],
        [m - s/sqrt3, m + s/sqrt3, c + m      ]
    ])

    return R


def apply_color_rotation_to_image(gray_img, rotation_angle=80, contrast=1.6):
    """
    Apply color rotation + contrast boost to a grayscale image.
    Returns RGB.
    """
    H, W = gray_img.shape

    # Contrast boost around 0.5
    contrasted = 0.5 + (gray_img - 0.5) * contrast
    contrasted = np.clip(contrasted, 0, 1)

    # Start with grayscale -> RGB
    rgb = np.stack([contrasted, contrasted, contrasted], axis=-1)

    # Apply rotation
    rot_matrix = color_rotation_matrix(rotation_angle)
    rotated = np.einsum('ijk,lk->ijl', rgb, rot_matrix)

    return np.clip(rotated, 0, 1)


# =============================================================================
# [SWIGLU] GATED COMBINATION
# =============================================================================

def swiglu_combine(img_np, thickened_mask, rotated_colored, gate):
    """
    [SWIGLU] Combine branches:
    - High gate (Fiedler+): Show thickened contours as dark
    - Low gate (Fiedler-): Show rotated+colored patches
    """
    H, W = img_np.shape

    # Start with original as grayscale RGB
    output = np.stack([img_np, img_np, img_np], axis=-1)

    # High gate region: apply thickened contours as dark lines
    high_gate = gate > 0.5
    thicken_region = (thickened_mask > 0.5) & high_gate

    # Thickened lines: dark with slight blue tint
    output[thicken_region, 0] = 0.02
    output[thicken_region, 1] = 0.02
    output[thicken_region, 2] = 0.08

    # Low gate region: blend in rotated+colored
    low_gate = gate <= 0.5

    # Blend factor: stronger where gate is lower
    blend = np.clip((0.5 - gate) * 2.5, 0, 1)
    blend_3d = blend[:, :, np.newaxis]

    # SwiGLU-style activation on blend
    silu_blend = blend_3d * (1.0 / (1.0 + np.exp(-blend_3d * 4)))

    # Apply blending only in low gate region
    for c in range(3):
        output[:, :, c] = np.where(
            low_gate,
            output[:, :, c] * (1 - silu_blend[:, :, 0]) + rotated_colored[:, :, c] * silu_blend[:, :, 0],
            output[:, :, c]
        )

    return output


# =============================================================================
# [RESIDUAL] SKIP CONNECTION
# =============================================================================

def residual_add(original_rgb, transformed_rgb, alpha=0.85):
    """[RESIDUAL] Blend with original."""
    return np.clip(original_rgb * (1 - alpha) + transformed_rgb * alpha, 0, 1)


# =============================================================================
# FULL PIPELINE
# =============================================================================

def resnet_spectral_shader_v3(img_np, config=None):
    """
    Full pipeline using Fiedler vector for meaningful bipartition.
    """
    if config is None:
        config = {
            'edge_threshold': 0.10,
            'max_thickness': 5,
            'patch_size': 13,
            'sample_stride': 6,
            'color_rotation': 85,
            'contrast': 1.7,
            'residual_alpha': 0.9,
        }

    H, W = img_np.shape
    intermediates = {}

    intermediates['input'] = img_np.copy()

    # [SPECTRAL] Compute Fiedler vector and gate
    fiedler, gate = compute_fiedler_gate(img_np, edge_threshold=config['edge_threshold'])
    intermediates['fiedler'] = fiedler.copy()
    intermediates['gate'] = gate.copy()

    # [BRANCH_A] Thicken contours where gate > 0.5
    thickened = thicken_contours_in_region(img_np, gate, max_thickness=config['max_thickness'])
    intermediates['thickened'] = thickened.copy()

    # [BRANCH_B] Rotated patches where gate < 0.5
    rotated_gray = create_rotated_patch_copies(
        img_np, gate,
        patch_size=config['patch_size'],
        sample_stride=config['sample_stride']
    )
    intermediates['rotated_gray'] = rotated_gray.copy()

    # Apply color rotation to rotated patches
    rotated_colored = apply_color_rotation_to_image(
        rotated_gray,
        rotation_angle=config['color_rotation'],
        contrast=config['contrast']
    )
    intermediates['rotated_colored'] = rotated_colored.copy()

    # [SWIGLU] Combine branches
    combined = swiglu_combine(img_np, thickened, rotated_colored, gate)
    intermediates['combined'] = combined.copy()

    # [RESIDUAL] Blend with original
    original_rgb = np.stack([img_np, img_np, img_np], axis=-1)
    output = residual_add(original_rgb, combined, alpha=config['residual_alpha'])
    intermediates['output'] = output.copy()

    return output, intermediates


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_v3(image_path, output_dir):
    """Visualize v3 pipeline."""

    img = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
    H, W = img.shape

    output, intermediates = resnet_spectral_shader_v3(img)

    def to_rgb(arr):
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1)
        return arr

    def colormap_diverging(arr):
        """Blue-white-red colormap for Fiedler (centered at 0.5)."""
        rgb = np.zeros((*arr.shape, 3))
        # Below 0.5: blue
        below = arr < 0.5
        rgb[below, 2] = 1 - 2 * arr[below]  # More blue where arr is low
        rgb[below, 0] = 2 * arr[below]
        rgb[below, 1] = 2 * arr[below]
        # Above 0.5: red
        above = arr >= 0.5
        rgb[above, 0] = 2 * (arr[above] - 0.5) + 0.5
        rgb[above, 1] = 1 - 2 * (arr[above] - 0.5)
        rgb[above, 2] = 1 - 2 * (arr[above] - 0.5)
        return np.clip(rgb, 0, 1)

    # Row 1: Input | Fiedler vector | Gate
    row1 = np.concatenate([
        to_rgb(intermediates['input']),
        colormap_diverging(intermediates['fiedler']),
        colormap_diverging(intermediates['gate']),
    ], axis=1)

    # Row 2: Thickened (Branch A) | Rotated gray (Branch B) | Rotated colored
    row2 = np.concatenate([
        to_rgb(intermediates['thickened']),
        to_rgb(intermediates['rotated_gray']),
        intermediates['rotated_colored'],
    ], axis=1)

    # Row 3: Combined | Final output | Side-by-side comparison
    comparison = np.concatenate([
        to_rgb(intermediates['input'])[:, :W//2],
        intermediates['output'][:, W//2:]
    ], axis=1)

    row3 = np.concatenate([
        intermediates['combined'],
        intermediates['output'],
        comparison,
    ], axis=1)

    grid = np.concatenate([row1, row2, row3], axis=0)
    grid = (grid * 255).clip(0, 255).astype(np.uint8)

    stem = Path(image_path).stem
    out_path = Path(output_dir) / f"resnet_v3_{stem}.png"
    Image.fromarray(grid).save(out_path)
    print(f"Saved: {out_path}")

    # Labels
    labeled = Image.fromarray(grid)
    draw = ImageDraw.Draw(labeled)
    labels = [
        ["INPUT", "[SPECTRAL] Fiedler", "[GATE] Fiedler sign"],
        ["[BRANCH_A] Thickened", "[BRANCH_B] Rotated", "Rotated + Color"],
        ["[SWIGLU] Combined", "FINAL OUTPUT", "Input|Output"],
    ]
    for ri, row_labels in enumerate(labels):
        for ci, lbl in enumerate(row_labels):
            x, y = ci * W + 5, ri * H + 5
            draw.text((x+1, y+1), lbl, fill=(0, 0, 0))
            draw.text((x, y), lbl, fill=(255, 255, 100))

    labeled.save(Path(output_dir) / f"resnet_v3_{stem}_labeled.png")
    print(f"Saved labeled")

    return output, intermediates


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
            visualize_v3(img_path, str(output_dir))
