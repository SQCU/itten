"""
Two-image cross-pixel-transforms with theta sweeps.

Architecture:
  image_A, image_B (e.g., same source with parametric variations)
      ↓
  abstract_A, abstract_B (SDF, phase from each)
      ↓
  cross_control_for_A = f(abstract_B, theta)  # B's structure controls A's transform
  cross_control_for_B = f(abstract_A, theta)  # A's structure controls B's transform
      ↓
  pixel_transform(A, control=cross_from_B)  # A transformed by B's spectral structure
  pixel_transform(B, control=cross_from_A)  # B transformed by A's spectral structure

This shows distributional shifts as theta varies and inputs change parametrically.
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import rotate, zoom, shift
from scipy.ndimage import sobel, distance_transform_edt
from abstract_intermediate_pipeline import (
    create_abstract_from_bitmap, filter_eigenvector_phase, filter_heat_diffusion
)
from pixel_transform_shaders import (
    pixel_coral_shimmer, pixel_contour_thicken, pixel_local_shear,
    pixel_displace_by_field, abstract_color_to_transforms
)
from spectral_ops_fast import (
    build_weighted_image_laplacian, iterative_spectral_transform,
    DEVICE
)


# =============================================================================
# PARAMETRIC IMAGE VARIATIONS
# =============================================================================

def create_parametric_variations(img: np.ndarray) -> dict:
    """
    Create parametric variations of source image.
    Returns dict of named variations.
    """
    H, W = img.shape
    variations = {'original': img}

    # Rotations
    for angle in [15, 30, 45]:
        rotated = rotate(img, angle, reshape=False, mode='constant', cval=1.0)
        variations[f'rot_{angle}'] = rotated

    # Stretches
    stretch_h = zoom(img, (1.0, 1.3), mode='constant', cval=1.0)
    stretch_h = stretch_h[:H, :W] if stretch_h.shape[1] >= W else np.pad(
        stretch_h, ((0, 0), (0, W - stretch_h.shape[1])), constant_values=1.0)[:H, :W]
    variations['stretch_h'] = stretch_h[:H, :W]

    stretch_v = zoom(img, (1.3, 1.0), mode='constant', cval=1.0)
    stretch_v = stretch_v[:H, :W] if stretch_v.shape[0] >= H else np.pad(
        stretch_v, ((0, H - stretch_v.shape[0]), (0, 0)), constant_values=1.0)[:H, :W]
    variations['stretch_v'] = stretch_v[:H, :W]

    # Shifted
    variations['shift_r'] = shift(img, (0, 20), mode='constant', cval=1.0)
    variations['shift_d'] = shift(img, (20, 0), mode='constant', cval=1.0)

    return variations


# =============================================================================
# CROSS-SPECTRAL CONTROL EXTRACTION
# =============================================================================

def extract_cross_control(img_source: np.ndarray, img_target: np.ndarray,
                          theta: float = 0.5) -> dict:
    """
    Extract control signals from source to apply to target.

    Returns multiple control fields derived from source's spectral structure
    at the given theta, shaped to influence target's pixels.
    """
    H, W = img_source.shape

    # Build abstract from source
    abstracts = create_abstract_from_bitmap(img_source)
    sdf = abstracts['sdf']
    contour_sweep = abstracts['contour_sweep']

    # Spectral transform on source's abstract at given theta
    carrier = torch.tensor(contour_sweep, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier, edge_threshold=0.1)
    signal = carrier.flatten()

    transformed = iterative_spectral_transform(L, signal, theta=theta, num_steps=6)
    spectral_field = transformed.cpu().numpy().reshape(H, W)
    spectral_field = (spectral_field - spectral_field.min()) / (spectral_field.max() - spectral_field.min() + 1e-8)

    # Phase from source
    phase = filter_eigenvector_phase(contour_sweep)

    # Heat diffusion for smooth control
    heat = filter_heat_diffusion(sdf, t=0.2)

    return {
        'spectral': spectral_field,
        'phase': phase,
        'heat': heat,
        'sdf': sdf,
    }


# =============================================================================
# CROSS PIXEL TRANSFORMS
# =============================================================================

def apply_cross_transform(img_target: np.ndarray, controls: dict,
                          transform_type: str = 'shimmer') -> np.ndarray:
    """
    Apply pixel transform to target using control signals from source.
    """
    if transform_type == 'shimmer':
        # Phase from source creates shimmer displacement on target
        return pixel_coral_shimmer(img_target, controls['phase'],
                                   amplitude=4.0, frequency=5.0)

    elif transform_type == 'thicken':
        # Heat from source controls stroke thickness on target
        return pixel_contour_thicken(img_target, controls['heat'],
                                     base_thickness=1, max_thickness=4)

    elif transform_type == 'shear':
        # Spectral from source controls shearing on target
        return pixel_local_shear(img_target, controls['spectral'], 'horizontal')

    elif transform_type == 'displace':
        # SDF gradients from source displace target
        gx = sobel(controls['sdf'], axis=1)
        gy = sobel(controls['sdf'], axis=0)
        return pixel_displace_by_field(img_target, gx, gy, strength=8.0)

    elif transform_type == 'selector':
        # Phase selects transform type per pixel
        return abstract_color_to_transforms(img_target, controls['phase'])

    elif transform_type == 'combined':
        # Chain multiple transforms
        t1 = pixel_coral_shimmer(img_target, controls['phase'], amplitude=2.0)
        t2 = pixel_contour_thicken(t1, controls['heat'], max_thickness=3)
        return t2

    return img_target


def cross_transform_pair(img_a: np.ndarray, img_b: np.ndarray,
                         theta: float = 0.5,
                         transform_type: str = 'combined') -> tuple:
    """
    Cross-transform a pair of images.
    A is transformed by B's spectral structure, B by A's.
    """
    # Extract controls
    controls_from_a = extract_cross_control(img_a, img_b, theta)
    controls_from_b = extract_cross_control(img_b, img_a, theta)

    # Apply cross transforms
    a_transformed = apply_cross_transform(img_a, controls_from_b, transform_type)
    b_transformed = apply_cross_transform(img_b, controls_from_a, transform_type)

    return a_transformed, b_transformed


# =============================================================================
# THETA SWEEP VISUALIZATION
# =============================================================================

def theta_sweep_cross_transform(img_a: np.ndarray, img_b: np.ndarray,
                                 thetas: list = [0.2, 0.4, 0.6, 0.8],
                                 transform_type: str = 'combined') -> list:
    """
    Sweep theta and show how cross-transform changes.
    Returns list of (a_transformed, b_transformed) tuples.
    """
    results = []
    for theta in thetas:
        a_t, b_t = cross_transform_pair(img_a, img_b, theta, transform_type)
        results.append((a_t, b_t))
    return results


# =============================================================================
# DEMO
# =============================================================================

def run_cross_transform_demo(input_path: str, output_dir: str):
    """
    Demo: parametric variations cross-transforming each other across theta.
    """
    img = np.array(Image.open(input_path).convert('L')).astype(np.float32) / 255.0
    H, W = img.shape

    # Create parametric variations
    variations = create_parametric_variations(img)

    # Select pairs to cross-transform
    pairs = [
        ('original', 'rot_30'),
        ('original', 'stretch_h'),
        ('rot_15', 'rot_45'),
    ]

    thetas = [0.2, 0.5, 0.8]

    all_rows = []

    for pair_name_a, pair_name_b in pairs:
        img_a = variations[pair_name_a]
        img_b = variations[pair_name_b]

        # Row: [A, B, A×B@θ1, A×B@θ2, A×B@θ3, B×A@θ1, B×A@θ2, B×A@θ3]
        # Simplified: [A, B, then theta sweep of A transformed by B]

        row_panels = [img_a, img_b]

        for theta in thetas:
            a_t, b_t = cross_transform_pair(img_a, img_b, theta, 'combined')
            row_panels.append(a_t)

        # Add one B transformed by A for comparison
        _, b_by_a = cross_transform_pair(img_a, img_b, 0.5, 'combined')
        row_panels.append(b_by_a)

        all_rows.append(row_panels)

    # Also show different transform types on one pair
    img_a = variations['original']
    img_b = variations['rot_30']

    transform_types = ['shimmer', 'thicken', 'displace', 'selector']
    type_row = [img_a, img_b]
    for tt in transform_types:
        a_t, _ = cross_transform_pair(img_a, img_b, 0.5, tt)
        type_row.append(a_t)
    all_rows.append(type_row)

    # Build grid - need to handle different row lengths
    max_cols = max(len(r) for r in all_rows)

    def make_row(imgs, target_cols):
        # Pad with white if needed
        while len(imgs) < target_cols:
            imgs.append(np.ones((H, W)))
        row = np.concatenate([np.stack([np.clip(x, 0, 1)]*3, axis=-1) for x in imgs], axis=1)
        return row

    grid = np.concatenate([make_row(r, max_cols) for r in all_rows], axis=0)
    grid = (grid * 255).clip(0, 255).astype(np.uint8)

    stem = Path(input_path).stem
    out_path = Path(output_dir) / f"cross_xform_{stem}.png"
    Image.fromarray(grid).save(out_path)
    print(f"Saved: {out_path}")

    # Labeled
    from PIL import ImageDraw
    labeled = Image.fromarray(grid)
    draw = ImageDraw.Draw(labeled)

    row_labels = [
        ["orig", "rot30", "A×B θ=.2", "θ=.5", "θ=.8", "B×A"],
        ["orig", "strH", "A×B θ=.2", "θ=.5", "θ=.8", "B×A"],
        ["r15", "r45", "A×B θ=.2", "θ=.5", "θ=.8", "B×A"],
        ["orig", "rot30", "shimmer", "thicken", "displace", "selector"],
    ]

    for row_idx, labels in enumerate(row_labels):
        for col_idx, lbl in enumerate(labels):
            if col_idx < max_cols:
                draw.text((col_idx * W + 3, row_idx * H + 3), lbl, fill=(255, 50, 50))

    labeled.save(Path(output_dir) / f"cross_xform_{stem}_labeled.png")
    print(f"Saved labeled")


def run_variation_grid(input_path: str, output_dir: str):
    """
    Show grid of all variations and their cross-interactions.
    """
    img = np.array(Image.open(input_path).convert('L')).astype(np.float32) / 255.0
    H, W = img.shape

    variations = create_parametric_variations(img)
    var_names = ['original', 'rot_15', 'rot_30', 'stretch_h', 'stretch_v']

    # Grid: rows = source of control, cols = target of transform
    # Cell (i,j) = variations[j] transformed by controls from variations[i]

    grid_rows = []

    # Header row: just the variations
    header = [variations[n] for n in var_names]
    grid_rows.append(header)

    # Each subsequent row: one source controlling all targets
    for src_name in var_names[:3]:  # Limit to 3 source rows for size
        row = []
        src_img = variations[src_name]

        for tgt_name in var_names:
            tgt_img = variations[tgt_name]

            if src_name == tgt_name:
                # Self: just show original
                row.append(tgt_img)
            else:
                # Cross-transform
                controls = extract_cross_control(src_img, tgt_img, theta=0.5)
                transformed = apply_cross_transform(tgt_img, controls, 'combined')
                row.append(transformed)

        grid_rows.append(row)

    # Build image
    def make_row(imgs):
        return np.concatenate([np.stack([np.clip(x, 0, 1)]*3, axis=-1) for x in imgs], axis=1)

    grid = np.concatenate([make_row(r) for r in grid_rows], axis=0)
    grid = (grid * 255).clip(0, 255).astype(np.uint8)

    stem = Path(input_path).stem
    out_path = Path(output_dir) / f"var_grid_{stem}.png"
    Image.fromarray(grid).save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    out_dir = "demo_output"
    for inp_name in ["toof.png", "snek-heavy.png"]:
        inp = Path("demo_output/inputs") / inp_name
        if inp.exists():
            run_cross_transform_demo(str(inp), out_dir)
            run_variation_grid(str(inp), out_dir)
