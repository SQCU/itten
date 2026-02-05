"""
RESNET-LIKE SPECTRAL COMPUTE SHADER v5 - NON-STATIONARY COLOR ORBITS

Fixes from v4:
1. Thickening PRESERVES existing colors (darkens but doesn't replace)
2. Color rotation is RELATIVE to current pixel color (non-stationary)
   - Gray → blue
   - Blue → rotate toward contrasting hue (magenta/cyan orbit)
   - Each pass orbits further through color space

The orbit creates cumulative, non-stationary color shifts across passes.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage import sobel, binary_dilation, label, center_of_mass

from spectral_ops_fast import (
    build_weighted_image_laplacian, lanczos_k_eigenvectors, DEVICE
)


# =============================================================================
# [SPECTRAL] FIEDLER VECTOR FOR GATING
# =============================================================================

def compute_fiedler_gate(img_gray, edge_threshold=0.12):
    """[SPECTRAL] Fiedler vector creates meaningful bipartition."""
    H, W = img_gray.shape
    carrier = torch.tensor(img_gray, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier, edge_threshold=edge_threshold)

    eigenvectors, eigenvalues = lanczos_k_eigenvectors(L, num_eigenvectors=2, num_iterations=50)
    fiedler = eigenvectors[:, -1].reshape(H, W)
    fiedler = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-8)

    temperature = 10.0
    gate = 1.0 / (1.0 + np.exp(-(fiedler - 0.5) * temperature))

    return fiedler, gate


# =============================================================================
# [BRANCH_A] THICKEN - NOW PRESERVES EXISTING COLORS
# =============================================================================

def thicken_contours_color_preserving(img_rgb, gate, max_thickness=5):
    """
    [BRANCH_A] Thicken contours while PRESERVING existing colors.

    Instead of replacing with black, we:
    1. Identify contour pixels (dark in grayscale)
    2. Dilate the contour MASK
    3. For new pixels in dilated region: darken existing color, don't replace
    """
    H, W, _ = img_rgb.shape
    img_gray = img_rgb.mean(axis=2)

    # Original contours
    contours = img_gray < 0.4

    # Thickness based on gate
    thickness_map = (gate * max_thickness).astype(int)

    # Track which pixels get thickened
    thicken_mask = np.zeros((H, W), dtype=bool)

    dilated = contours.copy()
    for t in range(1, max_thickness + 1):
        dilated = binary_dilation(dilated)
        apply_mask = (thickness_map >= t) & (gate > 0.5)
        thicken_mask |= (apply_mask & dilated & ~contours)  # New pixels only

    # For thickened pixels: darken existing color but preserve hue
    output = img_rgb.copy()

    for c in range(3):
        # Darken by multiplying with factor, preserving relative color
        output[thicken_mask, c] = output[thicken_mask, c] * 0.15

    # Original contours get very dark but preserve slight color
    output[contours, 0] = output[contours, 0] * 0.1
    output[contours, 1] = output[contours, 1] * 0.1
    output[contours, 2] = output[contours, 2] * 0.1 + 0.02  # Slight blue bias

    return output, thicken_mask


# =============================================================================
# NON-STATIONARY COLOR ROTATION - ORBITS THROUGH COLOR SPACE
# =============================================================================

def compute_current_hue(rgb):
    """
    Compute approximate hue angle for RGB pixels.
    Returns angle in radians, 0 = red, 2π/3 = green, 4π/3 = blue
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Simple hue approximation using atan2
    # Project onto RG plane relative to blue
    hue = np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)

    return hue


def relative_color_rotation(rgb, base_rotation=60, contrast_boost=1.4):
    """
    NON-STATIONARY color rotation: rotation direction depends on current color.

    - Gray pixels → rotate toward blue (standard)
    - Blue pixels → rotate toward magenta/purple (contrasting)
    - Magenta pixels → rotate toward red/orange
    - Creates an ORBIT through color space

    The key insight: "increasing local contrast" from blue means moving AWAY from blue.
    """
    H, W, _ = rgb.shape

    # Compute current hue
    current_hue = compute_current_hue(rgb)

    # Compute saturation (how colorful vs gray)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    saturation = (max_c - min_c) / (max_c + 1e-8)

    # Luminance
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    # Rotation angle depends on current hue:
    # - Gray (low saturation) → standard rotation toward blue
    # - Blue (hue near -2π/3) → rotate toward magenta (+60°)
    # - Magenta → rotate toward red
    # This creates a counterclockwise orbit

    # Base rotation in radians
    base_rad = np.radians(base_rotation)

    # Modify rotation based on saturation and hue
    # High saturation + blue hue → rotate more (to escape blue)
    blue_hue = -2 * np.pi / 3  # Blue is at -120° or -2π/3
    hue_distance_from_blue = np.abs(current_hue - blue_hue)

    # If already blue-ish (small distance), rotate MORE to increase contrast
    blue_factor = 1.0 + saturation * (1.0 - hue_distance_from_blue / np.pi) * 0.5

    rotation_angle = base_rad * blue_factor

    # Apply rotation in RGB space (simplified - rotate in RB plane)
    output = np.zeros_like(rgb)

    cos_r = np.cos(rotation_angle)
    sin_r = np.sin(rotation_angle)

    # Rotation primarily affects R and B channels (toward blue, then toward magenta)
    # This is a simplified hue rotation
    output[:, :, 0] = r * cos_r + b * sin_r * 0.3  # R gets some B influence
    output[:, :, 1] = g * 0.85 + b * 0.15 * sin_r  # G slightly affected
    output[:, :, 2] = b * cos_r + (1 - r) * sin_r * 0.5 + 0.1  # B boosted, inverse R influence

    # Apply contrast boost
    mean_lum = luminance.mean()
    for c in range(3):
        output[:, :, c] = mean_lum + (output[:, :, c] - mean_lum) * contrast_boost

    return np.clip(output, 0, 1)


def compute_shadow_color_orbit(rgb, orbit_strength=1.0):
    """
    Shadow color: rotate FURTHER than front copy, creating depth through color.

    Shadow is always "one step ahead" in the orbit.
    """
    # Apply stronger rotation for shadow
    shadow = relative_color_rotation(rgb, base_rotation=90 * orbit_strength, contrast_boost=1.3)

    # Additional blue push for shadow depth
    shadow[:, :, 2] = shadow[:, :, 2] * 0.8 + 0.25  # Guaranteed blue component
    shadow[:, :, 0] = shadow[:, :, 0] * 0.6  # Reduce red

    return np.clip(shadow, 0, 1)


def compute_front_color_orbit(rgb, orbit_strength=1.0):
    """
    Front copy: rotate in orbit, but less than shadow.
    Creates teal/cyan tones that contrast with blue shadow.
    """
    front = relative_color_rotation(rgb, base_rotation=50 * orbit_strength, contrast_boost=1.5)

    # Teal/cyan bias
    front[:, :, 1] = front[:, :, 1] * 0.9 + 0.15  # Green boost
    front[:, :, 2] = front[:, :, 2] * 0.9 + 0.12  # Blue component
    front[:, :, 0] = front[:, :, 0] * 0.5  # Reduce red

    return np.clip(front, 0, 1)


# =============================================================================
# [BRANCH_B] SEGMENT EXTRACTION AND ROTATION
# =============================================================================

def extract_contour_segments(img_rgb, gate, min_pixels=10, max_segments=50, gate_threshold=0.65):
    """Extract contour segments from low-gate regions."""
    img_gray = img_rgb.mean(axis=2) if img_rgb.ndim == 3 else img_rgb
    H, W = img_gray.shape

    contours = img_gray < 0.4
    low_gate = gate < gate_threshold
    eligible = contours & low_gate

    if eligible.sum() < min_pixels * 3:
        gate_median = np.median(gate[contours]) if contours.sum() > 0 else 0.5
        low_gate = gate < gate_median
        eligible = contours & low_gate

    labeled, num_features = label(eligible)
    segments = []

    for seg_id in range(1, min(num_features + 1, max_segments * 2)):
        mask = labeled == seg_id
        ys, xs = np.where(mask)

        if len(ys) < min_pixels:
            continue

        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        local_mask = mask[y0:y1, x0:x1]

        cy, cx = center_of_mass(local_mask)
        cy += y0
        cx += x0

        segments.append({
            'mask': local_mask,
            'bbox': (y0, y1, x0, x1),
            'center': (cy, cx),
            'num_pixels': len(ys),
        })

        if len(segments) >= max_segments:
            break

    segments.sort(key=lambda s: s['num_pixels'], reverse=True)
    return segments[:max_segments]


def compute_translation_direction(img_gray, cy, cx, offset_distance=20):
    """Compute translation along gradient (perpendicular to contour)."""
    H, W = img_gray.shape
    gx = sobel(img_gray, axis=1)
    gy = sobel(img_gray, axis=0)

    iy = int(np.clip(cy, 0, H-1))
    ix = int(np.clip(cx, 0, W-1))

    local_gx = gx[iy, ix]
    local_gy = gy[iy, ix]

    mag = np.sqrt(local_gx**2 + local_gy**2) + 1e-8
    nx = local_gx / mag
    ny = local_gy / mag

    return int(ny * offset_distance), int(nx * offset_distance)


def draw_segment_with_color_orbit(output_rgb, segment, img_rgb,
                                   translation, rotation_k=1,
                                   shadow_offset=(6, 6),
                                   orbit_strength=1.0):
    """
    Draw segment with NON-STATIONARY color orbit.

    Colors are computed RELATIVE to current pixel colors, so:
    - Gray → blue/teal
    - Blue → magenta/purple
    - Each pass orbits further
    """
    H, W, _ = output_rgb.shape

    mask = segment['mask']
    y0, y1, x0, x1 = segment['bbox']

    # Extract RGB patch and rotate
    rgb_patch = img_rgb[y0:y1, x0:x1].copy()
    rotated_mask = np.rot90(mask, k=rotation_k)
    rotated_rgb = np.rot90(rgb_patch, k=rotation_k)
    rh, rw = rotated_mask.shape

    ty, tx = translation
    new_y0, new_x0 = y0 + ty, x0 + tx
    new_y1, new_x1 = new_y0 + rh, new_x0 + rw

    sy0, sx0 = new_y0 + shadow_offset[0], new_x0 + shadow_offset[1]
    sy1, sx1 = sy0 + rh, sx0 + rw

    # Compute orbit colors based on CURRENT pixel colors
    shadow_colors = compute_shadow_color_orbit(rotated_rgb, orbit_strength)
    front_colors = compute_front_color_orbit(rotated_rgb, orbit_strength)

    # Draw shadow first
    if 0 <= sy0 < H and 0 <= sx0 < W:
        clip_sy0, clip_sy1 = max(0, sy0), min(H, sy1)
        clip_sx0, clip_sx1 = max(0, sx0), min(W, sx1)
        mask_y0 = clip_sy0 - sy0
        mask_y1 = mask_y0 + (clip_sy1 - clip_sy0)
        mask_x0 = clip_sx0 - sx0
        mask_x1 = mask_x0 + (clip_sx1 - clip_sx0)

        if mask_y1 > mask_y0 and mask_x1 > mask_x0:
            local_mask = rotated_mask[mask_y0:mask_y1, mask_x0:mask_x1]
            local_shadow = shadow_colors[mask_y0:mask_y1, mask_x0:mask_x1]

            for dy in range(local_mask.shape[0]):
                for dx in range(local_mask.shape[1]):
                    if local_mask[dy, dx]:
                        py, px = clip_sy0 + dy, clip_sx0 + dx
                        output_rgb[py, px] = local_shadow[dy, dx]

    # Draw front (masks shadow)
    if 0 <= new_y0 < H and 0 <= new_x0 < W:
        clip_y0, clip_y1 = max(0, new_y0), min(H, new_y1)
        clip_x0, clip_x1 = max(0, new_x0), min(W, new_x1)
        mask_y0 = clip_y0 - new_y0
        mask_y1 = mask_y0 + (clip_y1 - clip_y0)
        mask_x0 = clip_x0 - new_x0
        mask_x1 = mask_x0 + (clip_x1 - clip_x0)

        if mask_y1 > mask_y0 and mask_x1 > mask_x0:
            local_mask = rotated_mask[mask_y0:mask_y1, mask_x0:mask_x1]
            local_front = front_colors[mask_y0:mask_y1, mask_x0:mask_x1]

            for dy in range(local_mask.shape[0]):
                for dx in range(local_mask.shape[1]):
                    if local_mask[dy, dx]:
                        py, px = clip_y0 + dy, clip_x0 + dx
                        output_rgb[py, px] = local_front[dy, dx]


# =============================================================================
# SINGLE PASS WITH ORBIT STRENGTH
# =============================================================================

def single_pass_v5(img_rgb, pass_number, config):
    """
    Single pass with:
    - Color-preserving thickening
    - Non-stationary color orbits
    - Orbit strength increases with pass number (deeper = more rotation)
    """
    H, W = img_rgb.shape[:2]
    img_gray = img_rgb.mean(axis=2)

    # Attenuate gate threshold
    attenuation_step = 0.08
    gate_threshold = config['base_gate_threshold'] - (pass_number - 1) * attenuation_step
    gate_threshold = max(gate_threshold, 0.25)

    # Orbit strength INCREASES with pass number (more rotation for deeper passes)
    orbit_strength = 1.0 + (pass_number - 1) * 0.25

    # Effect strength for thickening
    effect_strength = 1.0 - (pass_number - 1) * 0.12
    effect_strength = max(effect_strength, 0.5)

    print(f"    Pass {pass_number}: gate={gate_threshold:.2f}, orbit={orbit_strength:.2f}, effect={effect_strength:.2f}")

    # Compute gate
    fiedler, gate = compute_fiedler_gate(img_gray, edge_threshold=config['edge_threshold'])

    output_rgb = img_rgb.copy()

    # [BRANCH A] Color-preserving thickening
    thickness = int(config['max_thickness'] * effect_strength)
    output_rgb, _ = thicken_contours_color_preserving(output_rgb, gate, max_thickness=thickness)

    # [BRANCH B] Segment extraction with color orbit
    segments = extract_contour_segments(
        output_rgb, gate,
        min_pixels=config['min_segment_pixels'],
        max_segments=int(config['max_segments'] * effect_strength),
        gate_threshold=gate_threshold
    )

    translation_distance = int(config['translation_distance'] * effect_strength)

    for segment in segments:
        cy, cx = segment['center']
        ty, tx = compute_translation_direction(img_gray, cy, cx, offset_distance=translation_distance)

        draw_segment_with_color_orbit(
            output_rgb, segment, output_rgb,  # Use CURRENT output for color orbit
            translation=(ty, tx),
            rotation_k=1,
            shadow_offset=(int(7 * effect_strength), int(7 * effect_strength)),
            orbit_strength=orbit_strength
        )

    return output_rgb, len(segments)


def stacked_shader_v5(img_np, num_passes, config=None):
    """Apply shader num_passes times with non-stationary color orbits."""
    if config is None:
        config = {
            'edge_threshold': 0.10,
            'max_thickness': 5,
            'min_segment_pixels': 10,
            'max_segments': 45,
            'base_gate_threshold': 0.65,
            'translation_distance': 28,
        }

    if img_np.ndim == 2:
        current = np.stack([img_np, img_np, img_np], axis=-1)
    else:
        current = img_np.copy()

    total_segments = 0

    for pass_num in range(1, num_passes + 1):
        current, num_segments = single_pass_v5(current, pass_num, config)
        total_segments += num_segments

    return current, total_segments


# =============================================================================
# VISUALIZATION
# =============================================================================

def run_stacked_v5(image_path, output_dir):
    """Run v5 stacked demo."""
    img = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
    H, W = img.shape
    print(f"  Image: {W}x{H}")

    pass_counts = [1, 2, 4, 8]
    outputs = []

    for num_passes in pass_counts:
        print(f"\n  Running {num_passes}x passes (v5 - color orbit):")
        output, total_segments = stacked_shader_v5(img, num_passes)
        outputs.append((num_passes, output, total_segments))
        print(f"    Total segments: {total_segments}")

    def to_rgb(arr):
        if arr.ndim == 2:
            return np.stack([arr, arr, arr], axis=-1)
        return arr

    original_rgb = to_rgb(img)

    row1 = np.concatenate([original_rgb, outputs[0][1], outputs[1][1]], axis=1)

    diff = np.abs(outputs[3][1] - original_rgb)
    diff = diff / (diff.max() + 1e-8) * 2
    diff = np.clip(diff, 0, 1)

    row2 = np.concatenate([outputs[2][1], outputs[3][1], diff], axis=1)

    grid = np.concatenate([row1, row2], axis=0)
    grid = (grid * 255).clip(0, 255).astype(np.uint8)

    stem = Path(image_path).stem
    out_path = Path(output_dir) / f"resnet_v5_{stem}.png"
    Image.fromarray(grid).save(out_path)
    print(f"\n  Saved: {out_path}")

    # Labels
    labeled = Image.fromarray(grid)
    draw = ImageDraw.Draw(labeled)
    labels = [
        ["ORIGINAL", "1x (orbit=1.0)", "2x (orbit=1.25)"],
        ["4x (orbit=1.75)", "8x (orbit=2.75)", "DIFFERENCE"],
    ]
    for ri, row_labels in enumerate(labels):
        for ci, lbl in enumerate(row_labels):
            x, y = ci * W + 5, ri * H + 5
            draw.text((x+1, y+1), lbl, fill=(0, 0, 0))
            draw.text((x, y), lbl, fill=(255, 255, 100))

    labeled.save(Path(output_dir) / f"resnet_v5_{stem}_labeled.png")

    return outputs


if __name__ == "__main__":
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    test_images = [
        "demo_output/inputs/snek-heavy.png",
        "demo_output/inputs/toof.png",
        "demo_output/inputs/1bit redraw.png",
    ]

    for img_path in test_images:
        if Path(img_path).exists():
            print(f"\n{'='*60}")
            print(f"Processing: {img_path}")
            print('='*60)
            run_stacked_v5(img_path, str(output_dir))
