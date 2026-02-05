"""
RESNET-LIKE SPECTRAL COMPUTE SHADER v4 - CONTOUR EXTRACTION + ROTATION + DROP SHADOW

Key changes from v3:
1. Extract actual CONTOUR SEGMENTS (not patches)
2. Copy ONLY contour pixels (bbox is transparent elsewhere)
3. Rotate segment 90 degrees (orthogonal)
4. Translate to new position
5. Draw DROP SHADOW: second copy, bluer, z-buffered behind first

The spectral basis (Fiedler vector) tells us WHERE to apply effects.
The contour extraction uses the image structure directly.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage import sobel, binary_dilation, label, center_of_mass
from scipy.ndimage import distance_transform_edt

from spectral_ops_fast import (
    build_weighted_image_laplacian, lanczos_k_eigenvectors, DEVICE
)


# =============================================================================
# [SPECTRAL] FIEDLER VECTOR FOR GATING
# =============================================================================

def compute_fiedler_gate(img_np, edge_threshold=0.12):
    """
    [SPECTRAL] Fiedler vector creates meaningful bipartition.
    """
    H, W = img_np.shape
    carrier = torch.tensor(img_np, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier, edge_threshold=edge_threshold)

    eigenvectors, eigenvalues = lanczos_k_eigenvectors(L, num_eigenvectors=2, num_iterations=50)
    fiedler = eigenvectors[:, -1].reshape(H, W)
    fiedler = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-8)

    temperature = 10.0
    gate = 1.0 / (1.0 + np.exp(-(fiedler - 0.5) * temperature))

    return fiedler, gate


# =============================================================================
# [BRANCH_A] HIGH GATE: THICKEN CONTOURS
# =============================================================================

def thicken_contours_gated(img_np, gate, max_thickness=5):
    """[BRANCH_A] Thicken contours where gate > 0.5"""
    H, W = img_np.shape
    contours = img_np < 0.5

    thickness_map = (gate * max_thickness).astype(int)
    result = np.zeros((H, W), dtype=np.float32)

    dilated = contours.copy()
    for t in range(1, max_thickness + 1):
        dilated = binary_dilation(dilated)
        apply_mask = (thickness_map >= t) & (gate > 0.5)
        result[apply_mask & dilated] = 1.0

    result[contours] = 1.0
    return result


# =============================================================================
# [BRANCH_B] LOW GATE: EXTRACT, ROTATE, TRANSLATE CONTOUR SEGMENTS
# =============================================================================

def extract_contour_segments(img_np, gate, min_pixels=20, max_segments=50,
                              gate_threshold=0.6):
    """
    Extract contour segments from LOW-GATE (Fiedler-) regions.

    Returns list of segments, each containing:
        - mask: boolean mask of contour pixels within local bbox
        - bbox: (y0, y1, x0, x1) bounding box in image coords
        - center: (cy, cx) center of mass
        - orientation: local orientation angle

    gate_threshold: pixels with gate < threshold are eligible (default 0.6 to be more inclusive)
    """
    H, W = img_np.shape

    # Find contour pixels (dark pixels in bitmap)
    contours = img_np < 0.5

    # Use more inclusive threshold - process contours where gate is below threshold
    # This ensures we get segments even when Fiedler partition is weak
    low_gate = gate < gate_threshold
    eligible = contours & low_gate

    # If we got nothing, try with ALL contours (fallback for simple images)
    if eligible.sum() < min_pixels * 3:
        # Fallback: use regions where gate is in the lower HALF of its range
        gate_median = np.median(gate[contours])
        low_gate = gate < gate_median
        eligible = contours & low_gate

    # Connected component labeling
    labeled, num_features = label(eligible)

    segments = []
    for seg_id in range(1, min(num_features + 1, max_segments * 2)):
        mask = labeled == seg_id
        ys, xs = np.where(mask)

        if len(ys) < min_pixels:
            continue

        # Bounding box
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1

        # Extract local mask within bbox
        local_mask = mask[y0:y1, x0:x1]

        # Center of mass
        cy, cx = center_of_mass(local_mask)
        cy += y0
        cx += x0

        # Local orientation from covariance/PCA
        ys_local = ys - cy
        xs_local = xs - cx
        cov = np.cov(xs_local, ys_local)
        if cov.shape == (2, 2):
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Principal direction
            principal = eigvecs[:, -1]
            angle = np.arctan2(principal[1], principal[0])
        else:
            angle = 0

        segments.append({
            'mask': local_mask,
            'bbox': (y0, y1, x0, x1),
            'center': (cy, cx),
            'angle': angle,
            'num_pixels': len(ys),
        })

        if len(segments) >= max_segments:
            break

    # Sort by size, take largest
    segments.sort(key=lambda s: s['num_pixels'], reverse=True)
    return segments[:max_segments]


def compute_translation_direction(img_np, cy, cx, offset_distance=20):
    """
    Compute translation direction based on local gradient (perpendicular to contour).
    """
    H, W = img_np.shape

    # Get local gradient at center
    gx = sobel(img_np, axis=1)
    gy = sobel(img_np, axis=0)

    # Sample gradient at center (clamped to bounds)
    iy = int(np.clip(cy, 0, H-1))
    ix = int(np.clip(cx, 0, W-1))

    local_gx = gx[iy, ix]
    local_gy = gy[iy, ix]

    # Normalize
    mag = np.sqrt(local_gx**2 + local_gy**2) + 1e-8
    nx = local_gx / mag
    ny = local_gy / mag

    # Translation along gradient direction (perpendicular to contour)
    tx = int(nx * offset_distance)
    ty = int(ny * offset_distance)

    return ty, tx


def color_rotation_matrix(angle_degrees):
    """Rotation around gray axis (1,1,1) toward blue."""
    angle = np.radians(angle_degrees)
    c = np.cos(angle)
    s = np.sin(angle)
    sqrt3 = np.sqrt(3)
    m = (1 - c) / 3

    return np.array([
        [c + m,       m - s/sqrt3, m + s/sqrt3],
        [m + s/sqrt3, c + m,       m - s/sqrt3],
        [m - s/sqrt3, m + s/sqrt3, c + m      ]
    ])


def draw_segment_with_drop_shadow(output_rgb, segment, img_np,
                                   translation, rotation_k=1,
                                   front_color_rotation=60,
                                   shadow_color_rotation=100,
                                   shadow_offset=(4, 4),
                                   front_contrast=1.4,
                                   shadow_contrast=1.2):
    """
    Draw a contour segment with rotation, translation, and drop shadow.

    Z-buffer order:
    1. Draw shadow first (bluer, offset behind)
    2. Draw front copy (masks shadow where they overlap)

    ONLY contour pixels are drawn - bbox background is transparent.
    """
    H, W, _ = output_rgb.shape

    mask = segment['mask']
    y0, y1, x0, x1 = segment['bbox']

    # Rotate the mask 90 degrees (or rotation_k * 90)
    rotated_mask = np.rot90(mask, k=rotation_k)
    rh, rw = rotated_mask.shape

    # Get original pixel values from image
    original_values = img_np[y0:y1, x0:x1].copy()
    rotated_values = np.rot90(original_values, k=rotation_k)

    # Apply translation
    ty, tx = translation
    new_y0 = y0 + ty
    new_x0 = x0 + tx
    new_y1 = new_y0 + rh
    new_x1 = new_x0 + rw

    # Shadow position (further offset)
    sy0 = new_y0 + shadow_offset[0]
    sx0 = new_x0 + shadow_offset[1]
    sy1 = sy0 + rh
    sx1 = sx0 + rw

    # Color rotation matrices
    front_rot = color_rotation_matrix(front_color_rotation)
    shadow_rot = color_rotation_matrix(shadow_color_rotation)

    # --- DRAW SHADOW FIRST (z-buffer: behind) ---
    if 0 <= sy0 < H and 0 <= sx0 < W:
        # Clip to image bounds
        clip_sy0 = max(0, sy0)
        clip_sy1 = min(H, sy1)
        clip_sx0 = max(0, sx0)
        clip_sx1 = min(W, sx1)

        # Corresponding region in rotated mask/values
        mask_y0 = clip_sy0 - sy0
        mask_y1 = mask_y0 + (clip_sy1 - clip_sy0)
        mask_x0 = clip_sx0 - sx0
        mask_x1 = mask_x0 + (clip_sx1 - clip_sx0)

        if mask_y1 > mask_y0 and mask_x1 > mask_x0:
            local_mask = rotated_mask[mask_y0:mask_y1, mask_x0:mask_x1]
            local_values = rotated_values[mask_y0:mask_y1, mask_x0:mask_x1]

            # SHADOW COLOR: Must be visibly blue even for dark source pixels
            # Instead of just color rotation, we CREATE a blue color and modulate by value
            # Dark contours (low value) → medium blue shadow
            # Light contours (high value) → lighter blue shadow

            # Base shadow color: obviously blue
            shadow_base_r = 0.15
            shadow_base_g = 0.25
            shadow_base_b = 0.55

            # Modulate by source luminance (inverted - dark contours get brighter shadow)
            # This ensures dark lines produce VISIBLE blue shadows
            lum_boost = 1.0 - local_values * 0.5  # dark (0) → 1.0, light (1) → 0.5

            shadow_rgb = np.zeros((*local_values.shape, 3))
            shadow_rgb[:, :, 0] = shadow_base_r * lum_boost * shadow_contrast
            shadow_rgb[:, :, 1] = shadow_base_g * lum_boost * shadow_contrast
            shadow_rgb[:, :, 2] = shadow_base_b * lum_boost * shadow_contrast
            shadow_rgb = np.clip(shadow_rgb, 0, 1)

            # Draw ONLY where mask is True (contour pixels only)
            for dy in range(local_mask.shape[0]):
                for dx in range(local_mask.shape[1]):
                    if local_mask[dy, dx]:
                        py = clip_sy0 + dy
                        px = clip_sx0 + dx
                        output_rgb[py, px] = shadow_rgb[dy, dx]

    # --- DRAW FRONT COPY (z-buffer: in front, masks shadow) ---
    if 0 <= new_y0 < H and 0 <= new_x0 < W:
        # Clip to image bounds
        clip_y0 = max(0, new_y0)
        clip_y1 = min(H, new_y1)
        clip_x0 = max(0, new_x0)
        clip_x1 = min(W, new_x1)

        mask_y0 = clip_y0 - new_y0
        mask_y1 = mask_y0 + (clip_y1 - clip_y0)
        mask_x0 = clip_x0 - new_x0
        mask_x1 = mask_x0 + (clip_x1 - clip_x0)

        if mask_y1 > mask_y0 and mask_x1 > mask_x0:
            local_mask = rotated_mask[mask_y0:mask_y1, mask_x0:mask_x1]
            local_values = rotated_values[mask_y0:mask_y1, mask_x0:mask_x1]

            # FRONT COPY: Cyan/teal tint to contrast with blue shadow
            # Dark contours → darker teal, light contours → lighter teal

            # Base front color: teal/cyan (contrasts with blue shadow)
            front_base_r = 0.08
            front_base_g = 0.35
            front_base_b = 0.38

            # Modulate: dark contours stay dark-ish, but tinted
            lum_factor = 0.3 + local_values * 0.4  # range 0.3-0.7 based on source

            front_rgb = np.zeros((*local_values.shape, 3))
            front_rgb[:, :, 0] = front_base_r + (1 - front_base_r) * local_values * 0.3
            front_rgb[:, :, 1] = front_base_g * lum_factor * front_contrast
            front_rgb[:, :, 2] = front_base_b * lum_factor * front_contrast
            front_rgb = np.clip(front_rgb, 0, 1)

            # Draw ONLY where mask is True
            for dy in range(local_mask.shape[0]):
                for dx in range(local_mask.shape[1]):
                    if local_mask[dy, dx]:
                        py = clip_y0 + dy
                        px = clip_x0 + dx
                        output_rgb[py, px] = front_rgb[dy, dx]


def process_branch_b(img_np, gate, output_rgb, config):
    """
    [BRANCH_B] Full processing: extract segments, rotate, translate, draw with shadow.
    """
    segments = extract_contour_segments(
        img_np, gate,
        min_pixels=config['min_segment_pixels'],
        max_segments=config['max_segments'],
        gate_threshold=config['gate_threshold']
    )

    for segment in segments:
        cy, cx = segment['center']

        # Compute translation direction (perpendicular to local contour)
        ty, tx = compute_translation_direction(
            img_np, cy, cx,
            offset_distance=config['translation_distance']
        )

        draw_segment_with_drop_shadow(
            output_rgb, segment, img_np,
            translation=(ty, tx),
            rotation_k=1,  # 90 degrees
            front_color_rotation=config['front_color_rotation'],
            shadow_color_rotation=config['shadow_color_rotation'],
            shadow_offset=config['shadow_offset'],
            front_contrast=config['front_contrast'],
            shadow_contrast=config['shadow_contrast']
        )

    return len(segments)


# =============================================================================
# [SWIGLU] COMBINATION
# =============================================================================

def swiglu_combine(img_np, thickened_mask, output_with_copies, gate):
    """
    [SWIGLU] Combine:
    - High gate: thickened contours (dark)
    - Low gate: already has rotated copies drawn
    """
    H, W = img_np.shape

    # High gate region: apply thickened contours
    high_gate = gate > 0.5
    thicken_region = (thickened_mask > 0.5) & high_gate

    # Darken thickened contours with slight color
    output_with_copies[thicken_region, 0] = 0.02
    output_with_copies[thicken_region, 1] = 0.02
    output_with_copies[thicken_region, 2] = 0.05

    return output_with_copies


# =============================================================================
# [RESIDUAL] SKIP CONNECTION
# =============================================================================

def residual_blend(original_rgb, transformed_rgb, alpha=0.9):
    """[RESIDUAL] Blend transformed with original."""
    return np.clip(original_rgb * (1 - alpha) + transformed_rgb * alpha, 0, 1)


# =============================================================================
# FULL PIPELINE
# =============================================================================

def resnet_spectral_shader_v4(img_np, config=None):
    """
    Full pipeline with contour extraction + rotation + drop shadow.
    """
    if config is None:
        config = {
            'edge_threshold': 0.10,
            'max_thickness': 5,
            'min_segment_pixels': 10,
            'max_segments': 60,
            'gate_threshold': 0.65,
            'translation_distance': 30,
            'front_color_rotation': 55,  # Not used with new color scheme
            'shadow_color_rotation': 100,  # Not used with new color scheme
            'shadow_offset': (8, 8),  # Larger offset for visible drop shadow
            'front_contrast': 1.4,
            'shadow_contrast': 1.5,  # Brighter shadow
            'residual_alpha': 0.95,
        }

    H, W = img_np.shape
    intermediates = {}

    intermediates['input'] = img_np.copy()

    # [SPECTRAL] Fiedler gate
    fiedler, gate = compute_fiedler_gate(img_np, edge_threshold=config['edge_threshold'])
    intermediates['fiedler'] = fiedler.copy()
    intermediates['gate'] = gate.copy()

    # [BRANCH_A] Thicken contours in high-gate regions
    thickened = thicken_contours_gated(img_np, gate, max_thickness=config['max_thickness'])
    intermediates['thickened'] = thickened.copy()

    # Start with grayscale base for output
    output_rgb = np.stack([img_np, img_np, img_np], axis=-1).copy()

    # [BRANCH_B] Extract, rotate, translate, draw with shadow in low-gate regions
    num_segments = process_branch_b(img_np, gate, output_rgb, config)
    intermediates['after_branch_b'] = output_rgb.copy()
    print(f"  Processed {num_segments} contour segments")

    # [SWIGLU] Combine with thickening
    combined = swiglu_combine(img_np, thickened, output_rgb, gate)
    intermediates['combined'] = combined.copy()

    # [RESIDUAL] Blend with original
    original_rgb = np.stack([img_np, img_np, img_np], axis=-1)
    output = residual_blend(original_rgb, combined, alpha=config['residual_alpha'])
    intermediates['output'] = output.copy()

    return output, intermediates


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_v4(image_path, output_dir):
    """Visualize v4 pipeline."""

    img = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
    H, W = img.shape
    print(f"  Image size: {W}x{H}")

    output, intermediates = resnet_spectral_shader_v4(img)

    def to_rgb(arr):
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1)
        return arr

    def colormap_diverging(arr):
        """Blue-white-red for Fiedler."""
        rgb = np.zeros((*arr.shape, 3))
        below = arr < 0.5
        rgb[below, 2] = 1 - 2 * arr[below]
        rgb[below, 0] = 2 * arr[below]
        rgb[below, 1] = 2 * arr[below]
        above = arr >= 0.5
        rgb[above, 0] = 2 * (arr[above] - 0.5) + 0.5
        rgb[above, 1] = 1 - 2 * (arr[above] - 0.5)
        rgb[above, 2] = 1 - 2 * (arr[above] - 0.5)
        return np.clip(rgb, 0, 1)

    # Row 1: Input | Fiedler | Gate
    row1 = np.concatenate([
        to_rgb(intermediates['input']),
        colormap_diverging(intermediates['fiedler']),
        colormap_diverging(intermediates['gate']),
    ], axis=1)

    # Row 2: Thickened (A) | After Branch B | Combined
    row2 = np.concatenate([
        to_rgb(intermediates['thickened']),
        intermediates['after_branch_b'],
        intermediates['combined'],
    ], axis=1)

    # Row 3: Output | Side-by-side | Difference
    diff = np.abs(intermediates['output'] - to_rgb(intermediates['input']))
    diff = diff / (diff.max() + 1e-8)  # Normalize for visibility

    comparison = np.concatenate([
        to_rgb(intermediates['input'])[:, :W//2],
        intermediates['output'][:, W//2:]
    ], axis=1)

    row3 = np.concatenate([
        intermediates['output'],
        comparison,
        diff * 3,  # Amplify difference for visibility
    ], axis=1)

    grid = np.concatenate([row1, row2, row3], axis=0)
    grid = (grid * 255).clip(0, 255).astype(np.uint8)

    stem = Path(image_path).stem
    out_path = Path(output_dir) / f"resnet_v4_{stem}.png"
    Image.fromarray(grid).save(out_path)
    print(f"  Saved: {out_path}")

    # Labels
    labeled = Image.fromarray(grid)
    draw = ImageDraw.Draw(labeled)
    labels = [
        ["INPUT", "[SPECTRAL] Fiedler", "[GATE]"],
        ["[BRANCH_A] Thickened", "[BRANCH_B] Rotated+Shadow", "[SWIGLU] Combined"],
        ["FINAL OUTPUT", "Input|Output", "Difference (3x)"],
    ]
    for ri, row_labels in enumerate(labels):
        for ci, lbl in enumerate(row_labels):
            x, y = ci * W + 5, ri * H + 5
            draw.text((x+1, y+1), lbl, fill=(0, 0, 0))
            draw.text((x, y), lbl, fill=(255, 255, 100))

    labeled.save(Path(output_dir) / f"resnet_v4_{stem}_labeled.png")

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
            visualize_v4(img_path, str(output_dir))
