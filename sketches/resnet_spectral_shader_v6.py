"""
RESNET SPECTRAL SHADER v6 - UNIFIED EFFECT STRENGTH + TRUE COLOR DILATION

Key changes:
1. Thickening DILATES colors (copies source color to new pixels, no darkening)
2. Unified `effect_strength` parameter scales ALL sub-calculations
3. Nonlinear color transform WITHOUT fixed points at black/white (cyclic)

The effect_strength parameter controls:
- Thickening receptive field (dilation radius)
- Rotation/shadow translation distance
- Color contrast rotation intensity
- Shadow offset distance

Linear decay of effect_strength across layers gives clean attenuation.
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
# [SPECTRAL] FIEDLER GATING
# =============================================================================

def compute_fiedler_gate(img_gray, edge_threshold=0.12):
    """Fiedler vector bipartition."""
    H, W = img_gray.shape
    carrier = torch.tensor(img_gray, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier, edge_threshold=edge_threshold)
    eigenvectors, _ = lanczos_k_eigenvectors(L, num_eigenvectors=2, num_iterations=50)
    fiedler = eigenvectors[:, -1].reshape(H, W)
    fiedler = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-8)
    gate = 1.0 / (1.0 + np.exp(-(fiedler - 0.5) * 10.0))
    return fiedler, gate


# =============================================================================
# TRUE COLOR-PRESERVING THICKENING (Dilation with color propagation)
# =============================================================================

def dilate_with_color_propagation(img_rgb, gate, max_thickness, effect_strength):
    """
    TRUE color-preserving thickening:
    - Identifies contour pixels and their colors
    - Dilates the contour MASK
    - For NEW pixels: copies color from NEAREST original contour pixel
    - NO darkening, NO color modification

    effect_strength scales the dilation radius.
    """
    H, W, _ = img_rgb.shape
    img_gray = img_rgb.mean(axis=2)

    # Original contours - LayerNorm style adaptive thresholding
    # Detects pixels significantly different from mean (works for light OR dark strokes)
    img_norm = (img_gray - img_gray.mean()) / (img_gray.std() + 1e-8)
    contours = np.abs(img_norm) > 1.0

    # Effective thickness based on gate and effect_strength
    thickness = int(max_thickness * effect_strength)
    thickness = max(thickness, 1)

    # High-gate region gets thickening
    high_gate = gate > 0.5

    # Dilate contour mask
    dilated = contours.copy()
    for _ in range(thickness):
        dilated = binary_dilation(dilated)

    # New pixels = dilated but not original contours, and in high-gate region
    new_pixels = dilated & ~contours & high_gate

    if not new_pixels.any():
        return img_rgb.copy()

    output = img_rgb.copy()

    # For new pixels, find nearest original contour pixel and copy its color
    # Use distance transform to find nearest contour
    if contours.any():
        # Distance from each pixel to nearest contour
        dist_to_contour = distance_transform_edt(~contours)

        # Get coordinates of new pixels
        new_ys, new_xs = np.where(new_pixels)

        # For each new pixel, find nearest contour pixel
        contour_ys, contour_xs = np.where(contours)

        if len(contour_ys) > 0:
            # Build simple nearest-neighbor lookup
            # For efficiency, sample if too many pixels
            for i in range(len(new_ys)):
                ny, nx = new_ys[i], new_xs[i]

                # Find nearest contour pixel (simple search in local neighborhood)
                # Search in expanding squares
                found = False
                for radius in range(1, thickness + 5):
                    y0, y1 = max(0, ny - radius), min(H, ny + radius + 1)
                    x0, x1 = max(0, nx - radius), min(W, nx + radius + 1)

                    local_contours = contours[y0:y1, x0:x1]
                    if local_contours.any():
                        # Find first contour pixel in this region
                        local_ys, local_xs = np.where(local_contours)
                        # Pick closest
                        dists = (local_ys - (ny - y0))**2 + (local_xs - (nx - x0))**2
                        closest_idx = np.argmin(dists)
                        src_y = y0 + local_ys[closest_idx]
                        src_x = x0 + local_xs[closest_idx]

                        # Copy color from source contour pixel
                        output[ny, nx] = img_rgb[src_y, src_x]
                        found = True
                        break

                if not found:
                    # Fallback: use original pixel color (shouldn't happen often)
                    pass

    return output


# =============================================================================
# NONLINEAR COLOR TRANSFORM WITHOUT BLACK/WHITE FIXED POINTS
# =============================================================================

def cyclic_color_transform(rgb, rotation_strength, contrast_strength):
    """
    Nonlinear color transform with NO fixed points at black or white.

    Uses sinusoidal/cyclic mapping so:
    - Pure black (0,0,0) → shifts to a color (not black)
    - Pure white (1,1,1) → shifts to a color (not white)

    This allows colors to continue evolving even at extremes.

    rotation_strength: how much to rotate in color space
    contrast_strength: how much to boost local contrast
    """
    H, W, _ = rgb.shape

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Luminance
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # CYCLIC transform: use sine waves to create color that has no fixed points
    # The key: even at lum=0 or lum=1, the output is non-trivial

    phase = rotation_strength * np.pi  # rotation as phase shift

    # Cyclic color channels with phase offsets
    # These functions have no fixed points at 0 or 1
    new_r = 0.5 + 0.4 * np.sin(2 * np.pi * lum + phase + 0)
    new_g = 0.5 + 0.4 * np.sin(2 * np.pi * lum + phase + 2*np.pi/3)
    new_b = 0.5 + 0.4 * np.sin(2 * np.pi * lum + phase + 4*np.pi/3)

    # Blend with original based on contrast_strength
    # Higher contrast = more of the cyclic transform
    blend = min(contrast_strength * 0.5, 0.9)

    out_r = r * (1 - blend) + new_r * blend
    out_g = g * (1 - blend) + new_g * blend
    out_b = b * (1 - blend) + new_b * blend

    # Additional contrast boost around mean
    mean = (out_r + out_g + out_b) / 3
    contrast_factor = 1.0 + contrast_strength * 0.3
    out_r = mean + (out_r - mean) * contrast_factor
    out_g = mean + (out_g - mean) * contrast_factor
    out_b = mean + (out_b - mean) * contrast_factor

    return np.clip(np.stack([out_r, out_g, out_b], axis=-1), 0, 1)


def compute_shadow_color(rgb, effect_strength):
    """Shadow: stronger cyclic rotation, more blue bias."""
    transformed = cyclic_color_transform(
        rgb,
        rotation_strength=0.3 * effect_strength,
        contrast_strength=0.8 * effect_strength
    )
    # Blue bias for shadow
    transformed[:, :, 2] = transformed[:, :, 2] * 0.7 + 0.3
    transformed[:, :, 0] = transformed[:, :, 0] * 0.7
    return np.clip(transformed, 0, 1)


def compute_front_color(rgb, effect_strength):
    """Front: moderate cyclic rotation, teal/cyan bias."""
    transformed = cyclic_color_transform(
        rgb,
        rotation_strength=0.2 * effect_strength,
        contrast_strength=0.6 * effect_strength
    )
    # Teal/cyan bias for front
    transformed[:, :, 1] = transformed[:, :, 1] * 0.8 + 0.2
    transformed[:, :, 2] = transformed[:, :, 2] * 0.8 + 0.15
    transformed[:, :, 0] = transformed[:, :, 0] * 0.6
    return np.clip(transformed, 0, 1)


# =============================================================================
# SEGMENT EXTRACTION + ROTATION
# =============================================================================

def extract_segments(img_rgb, gate, min_pixels, max_segments, gate_threshold):
    """Extract contour segments from low-gate regions."""
    img_gray = img_rgb.mean(axis=2)
    H, W = img_gray.shape

    # LayerNorm style adaptive thresholding
    img_norm = (img_gray - img_gray.mean()) / (img_gray.std() + 1e-8)
    contours = np.abs(img_norm) > 1.0
    low_gate = gate < gate_threshold
    eligible = contours & low_gate

    if eligible.sum() < min_pixels * 3:
        gate_median = np.median(gate[contours]) if contours.any() else 0.5
        eligible = contours & (gate < gate_median)

    labeled, num_features = label(eligible)
    segments = []

    for seg_id in range(1, min(num_features + 1, max_segments * 2)):
        mask = labeled == seg_id
        ys, xs = np.where(mask)
        if len(ys) < min_pixels:
            continue

        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1

        segments.append({
            'mask': mask[y0:y1, x0:x1],
            'bbox': (y0, y1, x0, x1),
            'center': (center_of_mass(mask[y0:y1, x0:x1])[0] + y0,
                       center_of_mass(mask[y0:y1, x0:x1])[1] + x0),
        })
        if len(segments) >= max_segments:
            break

    return segments


def compute_translation(img_gray, cy, cx, base_distance, effect_strength):
    """Translation distance scaled by effect_strength."""
    H, W = img_gray.shape
    gx, gy = sobel(img_gray, axis=1), sobel(img_gray, axis=0)

    iy, ix = int(np.clip(cy, 0, H-1)), int(np.clip(cx, 0, W-1))
    mag = np.sqrt(gx[iy, ix]**2 + gy[iy, ix]**2) + 1e-8

    distance = base_distance * effect_strength
    ty = int(gy[iy, ix] / mag * distance)
    tx = int(gx[iy, ix] / mag * distance)
    return ty, tx


def draw_segment_v6(output_rgb, segment, img_rgb, translation, effect_strength):
    """Draw rotated segment with cyclic color transform."""
    H, W, _ = output_rgb.shape

    mask = segment['mask']
    y0, y1, x0, x1 = segment['bbox']

    rgb_patch = img_rgb[y0:y1, x0:x1].copy()
    rotated_mask = np.rot90(mask, k=1)
    rotated_rgb = np.rot90(rgb_patch, k=1)
    rh, rw = rotated_mask.shape

    ty, tx = translation
    new_y0, new_x0 = y0 + ty, x0 + tx

    shadow_offset = int(7 * effect_strength)
    sy0, sx0 = new_y0 + shadow_offset, new_x0 + shadow_offset

    shadow_colors = compute_shadow_color(rotated_rgb, effect_strength)
    front_colors = compute_front_color(rotated_rgb, effect_strength)

    # Draw shadow
    for dy in range(rh):
        for dx in range(rw):
            if rotated_mask[dy, dx]:
                py, px = sy0 + dy, sx0 + dx
                if 0 <= py < H and 0 <= px < W:
                    output_rgb[py, px] = shadow_colors[dy, dx]

    # Draw front (masks shadow)
    for dy in range(rh):
        for dx in range(rw):
            if rotated_mask[dy, dx]:
                py, px = new_y0 + dy, new_x0 + dx
                if 0 <= py < H and 0 <= px < W:
                    output_rgb[py, px] = front_colors[dy, dx]


# =============================================================================
# SINGLE PASS WITH UNIFIED EFFECT_STRENGTH
# =============================================================================

def single_pass_v6(img_rgb, pass_number, config):
    """
    Single pass with UNIFIED effect_strength controlling all sub-calculations.

    effect_strength scales:
    - Thickening radius
    - Translation distance
    - Shadow offset
    - Color rotation intensity
    - Contrast boost
    """
    H, W = img_rgb.shape[:2]
    img_gray = img_rgb.mean(axis=2)

    # UNIFIED effect_strength: linear decay across passes
    effect_strength = config['base_effect_strength'] * (config['decay_rate'] ** (pass_number - 1))
    effect_strength = max(effect_strength, config['min_effect_strength'])

    # Gate threshold also decays
    gate_threshold = config['base_gate_threshold'] - (pass_number - 1) * 0.08
    gate_threshold = max(gate_threshold, 0.25)

    print(f"    Pass {pass_number}: effect_strength={effect_strength:.3f}, gate={gate_threshold:.2f}")

    fiedler, gate = compute_fiedler_gate(img_gray, edge_threshold=config['edge_threshold'])

    output_rgb = img_rgb.copy()

    # [BRANCH A] True color-preserving dilation
    output_rgb = dilate_with_color_propagation(
        output_rgb, gate,
        max_thickness=config['max_thickness'],
        effect_strength=effect_strength
    )

    # [BRANCH B] Segment extraction + rotation with cyclic colors
    max_segs = int(config['max_segments'] * effect_strength)
    segments = extract_segments(output_rgb, gate,
                                min_pixels=config['min_segment_pixels'],
                                max_segments=max(max_segs, 5),
                                gate_threshold=gate_threshold)

    for segment in segments:
        cy, cx = segment['center']
        ty, tx = compute_translation(img_gray, cy, cx,
                                     config['base_translation'],
                                     effect_strength)
        draw_segment_v6(output_rgb, segment, output_rgb, (ty, tx), effect_strength)

    return output_rgb, len(segments)


def stacked_shader_v6(img_np, num_passes, config=None):
    """Stack passes with unified effect_strength decay."""
    if config is None:
        config = {
            'edge_threshold': 0.10,
            'max_thickness': 5,
            'min_segment_pixels': 10,
            'max_segments': 2000,
            'base_gate_threshold': 0.65,
            'base_translation': 30,
            # Unified effect strength parameters
            'base_effect_strength': 1.0,
            'decay_rate': 0.75,  # Each pass = 75% of previous
            'min_effect_strength': 0.05,
        }

    current = np.stack([img_np]*3, axis=-1) if img_np.ndim == 2 else img_np.copy()
    total_segments = 0

    for p in range(1, num_passes + 1):
        current, n = single_pass_v6(current, p, config)
        total_segments += n

    return current, total_segments


# =============================================================================
# VISUALIZATION
# =============================================================================

def run_v6_demo(image_path, output_dir):
    """Run v6 demo with unified effect strength."""
    img = np.array(Image.open(image_path).convert('RGB')).astype(np.float32) / 255.0
    H, W = img.shape[:2]
    print(f"  Image: {W}x{H}")

    pass_counts = [1, 2, 4, 8]
    outputs = []

    for n in pass_counts:
        print(f"\n  {n}x passes (v6 - unified effect_strength):")
        out, segs = stacked_shader_v6(img, n)
        outputs.append((n, out, segs))
        print(f"    Segments: {segs}")

    def to_rgb(a):
        return np.stack([a]*3, axis=-1) if a.ndim == 2 else a

    orig = to_rgb(img)
    row1 = np.concatenate([orig, outputs[0][1], outputs[1][1]], axis=1)
    diff = np.abs(outputs[3][1] - orig)
    diff = np.clip(diff / (diff.max() + 1e-8) * 2, 0, 1)
    row2 = np.concatenate([outputs[2][1], outputs[3][1], diff], axis=1)
    grid = (np.concatenate([row1, row2], axis=0) * 255).clip(0, 255).astype(np.uint8)

    stem = Path(image_path).stem
    Image.fromarray(grid).save(Path(output_dir) / f"resnet_v6_{stem}.png")

    # Labels
    labeled = Image.fromarray(grid)
    draw = ImageDraw.Draw(labeled)
    for ri, row in enumerate([["ORIGINAL", "1x", "2x"], ["4x", "8x", "DIFF"]]):
        for ci, lbl in enumerate(row):
            draw.text((ci*W+5+1, ri*H+5+1), lbl, fill=(0,0,0))
            draw.text((ci*W+5, ri*H+5), lbl, fill=(255,255,100))
    labeled.save(Path(output_dir) / f"resnet_v6_{stem}_labeled.png")
    print(f"\n  Saved: resnet_v6_{stem}_labeled.png")


if __name__ == "__main__":
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    for img_path in ["demo_output/inputs/snek-heavy.png",
                     "demo_output/inputs/toof.png",
                     "demo_output/inputs/1bit redraw.png"]:
        if Path(img_path).exists():
            print(f"\n{'='*60}\n{img_path}\n{'='*60}")
            run_v6_demo(img_path, str(output_dir))
