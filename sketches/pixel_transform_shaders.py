"""
Pixel shaders that actually transform pixels, not just modulate brightness.

Key insight: spectral/abstract outputs are SELECTORS for pixel operations:
- Abstract "color" at pixel → which transform to apply
- Control signal → how much transform to apply
- Operations: copy/paste, rotate, shear, scale, displace, thicken

Input bitmap is monochrome, but abstract intermediate assigns "color" that
selects which geometric transform applies to that pixel's neighborhood.
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import sobel, distance_transform_edt, map_coordinates
from scipy.ndimage import binary_dilation, generate_binary_structure
from abstract_intermediate_pipeline import (
    create_abstract_from_bitmap, filter_heat_diffusion,
    filter_bandpass, filter_eigenvector_phase
)
from spectral_ops_fast import DEVICE


# =============================================================================
# PIXEL TRANSFORM OPERATIONS
# =============================================================================

def pixel_displace_by_field(img: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                            strength: float = 5.0) -> np.ndarray:
    """Displace pixels by vector field (dx, dy)."""
    H, W = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # Apply displacement
    new_x = xx + strength * dx
    new_y = yy + strength * dy

    # Clamp to valid range
    new_x = np.clip(new_x, 0, W - 1)
    new_y = np.clip(new_y, 0, H - 1)

    # Sample with interpolation
    displaced = map_coordinates(img, [new_y, new_x], order=1, mode='reflect')
    return displaced


def pixel_copy_attended_to_unattended(img: np.ndarray, attention: np.ndarray,
                                       threshold: float = 0.6) -> np.ndarray:
    """
    Copy pixels from high-attention regions to low-attention regions.
    Creates echo/stamp effect.
    """
    H, W = img.shape
    attended_mask = attention > threshold
    unattended_mask = attention < (1 - threshold)

    # Get attended pixel coordinates
    att_ys, att_xs = np.where(attended_mask)
    if len(att_xs) == 0:
        return img

    # For unattended pixels, sample from attended pixels with offset
    result = img.copy()
    unatt_ys, unatt_xs = np.where(unattended_mask)

    if len(unatt_xs) > 0 and len(att_xs) > 0:
        # For each unattended pixel, copy from a corresponding attended pixel
        # Use modular indexing to cycle through attended pixels
        n_att = len(att_xs)
        for i, (uy, ux) in enumerate(zip(unatt_ys, unatt_xs)):
            src_idx = i % n_att
            result[uy, ux] = img[att_ys[src_idx], att_xs[src_idx]]

    return result


def pixel_local_rotate(img: np.ndarray, angle_field: np.ndarray,
                       kernel_size: int = 5) -> np.ndarray:
    """
    Apply local rotation to pixel neighborhoods based on angle field.
    angle_field values in [0, 1] map to [0, 2π].
    """
    H, W = img.shape
    result = np.zeros_like(img)
    pad = kernel_size // 2

    # Pad image
    padded = np.pad(img, pad, mode='reflect')

    for y in range(H):
        for x in range(W):
            angle = angle_field[y, x] * 2 * np.pi

            # Extract local patch
            patch = padded[y:y+kernel_size, x:x+kernel_size]

            # Rotate patch (simplified: just sample at rotated coords)
            cy, cx = pad, pad
            cos_a, sin_a = np.cos(angle), np.sin(angle)

            # Sample center of rotated patch
            # For efficiency, just take weighted average based on rotation
            # This is a simplified local rotation effect
            dy = int(round(sin_a * pad * 0.5))
            dx = int(round(cos_a * pad * 0.5))

            sy = np.clip(pad + dy, 0, kernel_size - 1)
            sx = np.clip(pad + dx, 0, kernel_size - 1)

            result[y, x] = patch[sy, sx]

    return result


def pixel_local_scale(img: np.ndarray, scale_field: np.ndarray) -> np.ndarray:
    """
    Local scaling: scale_field > 0.5 zooms in, < 0.5 zooms out.
    Implemented as displacement toward/away from local center.
    """
    H, W = img.shape

    # Scale field to displacement magnitude
    # 0.5 = no displacement, 0 = zoom out (displace away), 1 = zoom in (displace toward)
    scale = (scale_field - 0.5) * 2  # [-1, 1]

    # Compute local gradient direction (toward high-contrast features)
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    mag = np.sqrt(gx**2 + gy**2) + 1e-8

    # Displacement toward/away from features
    dx = scale * gx / mag * 3
    dy = scale * gy / mag * 3

    return pixel_displace_by_field(img, dx, dy, strength=1.0)


def pixel_local_shear(img: np.ndarray, shear_field: np.ndarray,
                      direction: str = 'horizontal') -> np.ndarray:
    """
    Local shearing based on shear_field.
    shear_field in [0, 1] maps to shear amount.
    """
    H, W = img.shape
    shear = (shear_field - 0.5) * 2  # [-1, 1]

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    if direction == 'horizontal':
        # Shear x based on y position and shear field
        dx = shear * (yy - H/2) / H * 10
        dy = np.zeros_like(dx)
    else:
        # Shear y based on x position
        dy = shear * (xx - W/2) / W * 10
        dx = np.zeros_like(dy)

    return pixel_displace_by_field(img, dx, dy, strength=1.0)


def pixel_contour_thicken(img: np.ndarray, thickness_field: np.ndarray,
                          base_thickness: int = 1, max_thickness: int = 5) -> np.ndarray:
    """
    Thicken contours/strokes based on thickness_field.
    Dark pixels (strokes) are dilated by varying amounts.
    """
    H, W = img.shape

    # Detect strokes (dark pixels)
    stroke_mask = img < 0.5

    # Quantize thickness field to integer dilation amounts
    thickness = base_thickness + (thickness_field * (max_thickness - base_thickness)).astype(int)

    result = np.ones_like(img)

    # Apply varying dilation
    for t in range(base_thickness, max_thickness + 1):
        mask_for_t = stroke_mask & (thickness >= t)
        if mask_for_t.any():
            struct = generate_binary_structure(2, 1)
            dilated = binary_dilation(mask_for_t, struct, iterations=t)
            result[dilated] = 0

    return result


def pixel_coral_shimmer(img: np.ndarray, phase_field: np.ndarray,
                        amplitude: float = 3.0, frequency: float = 8.0) -> np.ndarray:
    """
    Apply shimmery displacement based on phase field.
    Creates organic, coral-like warping.
    """
    H, W = img.shape

    # Phase field creates sinusoidal displacement
    # The "coral" effect comes from phase discontinuities creating swirls
    dx = amplitude * np.sin(phase_field * frequency * 2 * np.pi)
    dy = amplitude * np.cos(phase_field * frequency * 2 * np.pi)

    # Add some noise for organic feel
    noise_x = np.random.randn(H, W) * 0.5
    noise_y = np.random.randn(H, W) * 0.5

    return pixel_displace_by_field(img, dx + noise_x, dy + noise_y, strength=1.0)


# =============================================================================
# ABSTRACT COLOR → TRANSFORM SELECTOR
# =============================================================================

def abstract_color_to_transforms(img: np.ndarray, abstract_color: np.ndarray) -> np.ndarray:
    """
    Use abstract "color" (from phase/spectral) to select which transform applies.

    Quantize abstract_color into bands, each band gets a different transform:
      - Band 0: no transform (identity)
      - Band 1: local rotate
      - Band 2: local scale
      - Band 3: local shear
      - Band 4: coral shimmer
    """
    H, W = img.shape
    n_bands = 5

    # Quantize to bands
    bands = (abstract_color * n_bands).astype(int)
    bands = np.clip(bands, 0, n_bands - 1)

    result = img.copy()

    # Apply transforms per band
    # Band 0: identity (keep original)

    # Band 1: rotate
    mask1 = bands == 1
    if mask1.any():
        rotated = pixel_local_rotate(img, abstract_color, kernel_size=5)
        result[mask1] = rotated[mask1]

    # Band 2: scale
    mask2 = bands == 2
    if mask2.any():
        scaled = pixel_local_scale(img, abstract_color)
        result[mask2] = scaled[mask2]

    # Band 3: shear
    mask3 = bands == 3
    if mask3.any():
        sheared = pixel_local_shear(img, abstract_color, 'horizontal')
        result[mask3] = sheared[mask3]

    # Band 4: shimmer
    mask4 = bands == 4
    if mask4.any():
        shimmered = pixel_coral_shimmer(img, abstract_color, amplitude=2.0)
        result[mask4] = shimmered[mask4]

    return result


def combined_transform_pipeline(img: np.ndarray, control_field: np.ndarray,
                                 transform_selector: np.ndarray) -> np.ndarray:
    """
    Full pipeline:
    1. transform_selector chooses which operation
    2. control_field modulates how much
    """
    # First apply selector-based transforms
    transformed = abstract_color_to_transforms(img, transform_selector)

    # Then blend with original based on control strength
    result = img * (1 - control_field) + transformed * control_field

    return result


# =============================================================================
# DEMO
# =============================================================================

def run_pixel_transform_demo(input_path: str, output_dir: str):
    """Demo showing actual pixel transforms, not just brightness modulation."""
    img = np.array(Image.open(input_path).convert('L')).astype(np.float32) / 255.0
    H, W = img.shape

    # Get abstract intermediates and filters
    abstracts = create_abstract_from_bitmap(img)
    sdf = abstracts['sdf']
    phase = filter_eigenvector_phase(abstracts['contour_sweep'])
    heat = filter_heat_diffusion(abstracts['sdf'], t=0.25)

    # === Individual pixel transforms ===
    panels = []

    # Row 1: Original + control fields
    row1 = [
        img,
        sdf,
        phase,
        heat,
    ]

    # Row 2: Individual transforms
    coral = pixel_coral_shimmer(img, phase, amplitude=4.0, frequency=6.0)
    thickened = pixel_contour_thicken(img, heat, base_thickness=1, max_thickness=4)
    rotated = pixel_local_rotate(img, phase, kernel_size=7)
    copy_paste = pixel_copy_attended_to_unattended(img, heat, threshold=0.5)

    row2 = [coral, thickened, rotated, copy_paste]

    # Row 3: Transform selector (phase chooses transform type)
    selector_result = abstract_color_to_transforms(img, phase)
    combined = combined_transform_pipeline(img, heat, phase)
    # Shear by SDF
    sheared = pixel_local_shear(img, sdf, 'horizontal')
    # Scale by heat
    scaled = pixel_local_scale(img, heat)

    row3 = [selector_result, combined, sheared, scaled]

    # Row 4: Stacked transforms
    # Coral then thicken
    coral_thick = pixel_contour_thicken(coral, heat, base_thickness=1, max_thickness=3)
    # Rotate then shimmer
    rotate_shimmer = pixel_coral_shimmer(rotated, phase, amplitude=2.0)
    # Full pipeline with different selector
    full1 = combined_transform_pipeline(coral, sdf, phase)
    full2 = combined_transform_pipeline(thickened, phase, heat)

    row4 = [coral_thick, rotate_shimmer, full1, full2]

    # Build grid
    def make_row(imgs):
        return np.concatenate([np.stack([np.clip(x, 0, 1)]*3, axis=-1) for x in imgs], axis=1)

    grid = np.concatenate([make_row(r) for r in [row1, row2, row3, row4]], axis=0)
    grid = (grid * 255).clip(0, 255).astype(np.uint8)

    stem = Path(input_path).stem
    out_path = Path(output_dir) / f"pixel_xform_{stem}.png"
    Image.fromarray(grid).save(out_path)
    print(f"Saved: {out_path}")

    # Labeled
    from PIL import ImageDraw
    labeled = Image.fromarray(grid)
    draw = ImageDraw.Draw(labeled)
    labels = [
        ["Original", "SDF", "Phase", "Heat"],
        ["Coral Shimmer", "Contour Thick", "Local Rotate", "Copy→Paste"],
        ["Phase→Xform", "Combined", "Shear(SDF)", "Scale(Heat)"],
        ["Coral+Thick", "Rotate+Shimmer", "Pipeline A", "Pipeline B"],
    ]
    for row_idx, row_labels in enumerate(labels):
        for col_idx, lbl in enumerate(row_labels):
            draw.text((col_idx * W + 3, row_idx * H + 3), lbl, fill=(255, 50, 50))

    labeled.save(Path(output_dir) / f"pixel_xform_{stem}_labeled.png")
    print(f"Saved labeled")


if __name__ == "__main__":
    out_dir = "demo_output"
    for inp_name in ["toof.png", "snek-heavy.png", "1bit redraw.png"]:
        inp = Path("demo_output/inputs") / inp_name
        if inp.exists():
            run_pixel_transform_demo(str(inp), out_dir)
