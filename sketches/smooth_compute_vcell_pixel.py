"""
Separated compute/pixel shader architecture:

COMPUTE SHADER (spectral, smooth output):
  - Uses pixel-level image Laplacian (not cell-based)
  - Spectral transform produces smooth gradients
  - No V-cell structure visible in compute output

PIXEL SHADER (V-cell as visual effect):
  - V-cell segmentation applied to PIXELS
  - Operations: posterize per cell, edge drawing, local contrast
  - V-cells modify the image, not the compute signal

The key: compute_signal is smooth, pixel_effect is structured.
Final: shaderfunc(image, smooth_compute_signal, vcell_pixel_effect) -> output
"""
import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage import sobel
from scipy.spatial import Voronoi, cKDTree
from spectral_ops_fast import (
    build_weighted_image_laplacian, iterative_spectral_transform,
    heat_diffusion_sparse, chebyshev_filter, estimate_lambda_max,
    DEVICE
)


# =============================================================================
# COMPUTE SHADER: Pixel-level spectral (smooth output, no cells)
# =============================================================================

def compute_shader_smooth_spectral(img: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """
    Spectral transform on the PIXEL-LEVEL image graph.
    Output is smooth - no cell structure visible.
    """
    H, W = img.shape
    carrier = torch.tensor(img, dtype=torch.float32, device=DEVICE)

    # Pixel-level Laplacian (each pixel is a node, edges to neighbors)
    L = build_weighted_image_laplacian(carrier, edge_threshold=0.15)

    # Signal is the image itself (flattened)
    signal = carrier.flatten()

    # Spectral transform - smooth because graph is pixel-level
    transformed = iterative_spectral_transform(L, signal, theta=theta, num_steps=6)

    # Reshape back to image
    result = transformed.cpu().numpy().reshape(H, W)

    # Normalize to [0, 1]
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result


def compute_shader_heat_diffusion(img: np.ndarray, t: float = 0.3) -> np.ndarray:
    """
    Heat diffusion on pixel graph - creates smooth blur respecting edges.
    """
    H, W = img.shape
    carrier = torch.tensor(img, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier, edge_threshold=0.1)
    signal = carrier.flatten()

    diffused = heat_diffusion_sparse(L, signal, alpha=t, iterations=30)
    result = diffused.cpu().numpy().reshape(H, W)
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result


def compute_shader_bandpass(img: np.ndarray, band_center: float = 0.5,
                            band_width: float = 0.3) -> np.ndarray:
    """
    Chebyshev bandpass filter on pixel graph - isolates spectral band smoothly.
    """
    H, W = img.shape
    carrier = torch.tensor(img, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier, edge_threshold=0.1)
    signal = carrier.flatten()

    lambda_max = estimate_lambda_max(L)
    filtered = chebyshev_filter(L, signal,
                                center=band_center * lambda_max,
                                width=band_width * lambda_max,
                                order=25, lambda_max=lambda_max)

    result = filtered.cpu().numpy().reshape(H, W)
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result


# =============================================================================
# PIXEL SHADER: V-cell as visual effect on pixels
# =============================================================================

def extract_vcell_segmentation(img: np.ndarray, n_seeds: int = 80) -> tuple:
    """
    Extract V-cell segmentation for pixel shader use.
    Returns (cell_map, seeds, edges) for rendering effects.
    """
    H, W = img.shape
    gx, gy = sobel(img, axis=1), sobel(img, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    ys, xs = np.where(grad_mag > np.percentile(grad_mag, 85))

    if len(xs) < n_seeds:
        extra = n_seeds - len(xs)
        ys = np.concatenate([ys, np.random.randint(0, H, extra)])
        xs = np.concatenate([xs, np.random.randint(0, W, extra)])

    indices = np.random.choice(len(xs), n_seeds, replace=False)
    seeds = np.column_stack([xs[indices], ys[indices]])

    # Cell map
    tree = cKDTree(seeds)
    yy, xx = np.mgrid[0:H, 0:W]
    _, cell_ids = tree.query(np.column_stack([xx.ravel(), yy.ravel()]))
    cell_map = cell_ids.reshape(H, W)

    # Voronoi edges
    vor = Voronoi(seeds)
    edges = []
    for ridge in vor.ridge_points:
        i, j = ridge
        if 0 <= i < n_seeds and 0 <= j < n_seeds:
            edges.append((i, j))

    return cell_map, seeds, edges


def pixel_shader_posterize_per_cell(img: np.ndarray, cell_map: np.ndarray,
                                     levels: int = 4) -> np.ndarray:
    """
    Posterize image, but compute average per V-cell (flat color per cell).
    """
    n_cells = cell_map.max() + 1
    result = np.zeros_like(img)

    for cell_id in range(n_cells):
        mask = cell_map == cell_id
        if mask.sum() > 0:
            avg = img[mask].mean()
            # Quantize to levels
            quantized = np.round(avg * (levels - 1)) / (levels - 1)
            result[mask] = quantized

    return result


def pixel_shader_cell_edges(img: np.ndarray, cell_map: np.ndarray,
                            edge_strength: float = 0.3) -> np.ndarray:
    """
    Darken pixels at V-cell boundaries.
    """
    H, W = img.shape
    edge_mask = np.zeros((H, W), dtype=bool)

    # Detect where cell_map changes
    edge_mask[:-1, :] |= cell_map[:-1, :] != cell_map[1:, :]
    edge_mask[:, :-1] |= cell_map[:, :-1] != cell_map[:, 1:]

    result = img.copy()
    result[edge_mask] *= (1 - edge_strength)
    return result


def pixel_shader_local_contrast(img: np.ndarray, cell_map: np.ndarray,
                                 strength: float = 1.5) -> np.ndarray:
    """
    Enhance contrast within each V-cell independently.
    """
    n_cells = cell_map.max() + 1
    result = np.zeros_like(img)

    for cell_id in range(n_cells):
        mask = cell_map == cell_id
        if mask.sum() > 0:
            cell_vals = img[mask]
            cell_mean = cell_vals.mean()
            # Enhance deviation from mean
            enhanced = cell_mean + strength * (cell_vals - cell_mean)
            result[mask] = np.clip(enhanced, 0, 1)

    return result


# =============================================================================
# COMBINED: shaderfunc(image, compute_signal, pixel_effect) -> output
# =============================================================================

def combined_shader(img: np.ndarray, compute_signal: np.ndarray,
                    pixel_effect: np.ndarray, blend_mode: str = 'multiply') -> np.ndarray:
    """
    Combine smooth compute signal with structured pixel effect.

    blend_mode options:
      - 'multiply': output = img * compute * pixel_effect
      - 'screen': additive-ish blend
      - 'compute_gate': compute gates how much pixel_effect applies
    """
    if blend_mode == 'multiply':
        # Both modulate the image multiplicatively
        return img * (0.2 + 0.8 * compute_signal) * (0.3 + 0.7 * pixel_effect)

    elif blend_mode == 'screen':
        # Screen blend
        base = img * (0.3 + 0.7 * compute_signal)
        return 1 - (1 - base) * (1 - 0.3 * pixel_effect)

    elif blend_mode == 'compute_gate':
        # Compute signal controls how much the pixel effect applies
        # Where compute is high, pixel_effect dominates; where low, original dominates
        return img * (1 - compute_signal) + pixel_effect * compute_signal

    else:
        return img * compute_signal


def run_demo(input_path: str, output_dir: str):
    """
    Demo: smooth compute + structured pixel shader, no V-cells in compute output.
    """
    img = np.array(Image.open(input_path).convert('L')).astype(np.float32) / 255.0
    H, W = img.shape

    # === COMPUTE SHADERS (smooth, no cell structure) ===
    compute_spectral = compute_shader_smooth_spectral(img, theta=0.5)
    compute_heat = compute_shader_heat_diffusion(img, t=0.25)
    compute_band = compute_shader_bandpass(img, band_center=0.4, band_width=0.3)

    # === PIXEL SHADER PREP (V-cell segmentation) ===
    cell_map, seeds, edges = extract_vcell_segmentation(img, n_seeds=60)

    # === PIXEL SHADERS (structured effects on pixels) ===
    pixel_posterize = pixel_shader_posterize_per_cell(img, cell_map, levels=5)
    pixel_edges = pixel_shader_cell_edges(img, cell_map, edge_strength=0.5)
    pixel_contrast = pixel_shader_local_contrast(img, cell_map, strength=1.8)

    # === BUILD OUTPUT PANELS ===
    panels = []

    # Row 1: Original, Compute outputs (should be smooth)
    row1 = [
        img,
        compute_spectral,
        compute_heat,
        compute_band,
    ]
    panels.append(np.concatenate([np.stack([x]*3, axis=-1) for x in row1], axis=1))

    # Row 2: Pixel shader effects (structured)
    row2 = [
        img,
        pixel_posterize,
        pixel_edges,
        pixel_contrast,
    ]
    panels.append(np.concatenate([np.stack([x]*3, axis=-1) for x in row2], axis=1))

    # Row 3: Combined - compute gates pixel effect
    combined1 = combined_shader(img, compute_spectral, pixel_posterize, 'compute_gate')
    combined2 = combined_shader(img, compute_heat, pixel_edges, 'multiply')
    combined3 = combined_shader(img, compute_band, pixel_contrast, 'multiply')
    # One more: heat diffusion gating posterize
    combined4 = combined_shader(img, compute_heat, pixel_posterize, 'compute_gate')

    row3 = [img, combined1, combined2, combined3]
    panels.append(np.concatenate([np.stack([x]*3, axis=-1) for x in row3], axis=1))

    # Stack rows
    combined_img = np.concatenate([(p * 255).clip(0, 255).astype(np.uint8) for p in panels], axis=0)

    stem = Path(input_path).stem
    out_path = Path(output_dir) / f"smooth_compute_{stem}.png"
    Image.fromarray(combined_img).save(out_path)
    print(f"Saved: {out_path}")

    # Labeled
    labeled = Image.fromarray(combined_img)
    draw = ImageDraw.Draw(labeled)
    # Row labels
    row_labels = [
        ["Original", "Compute:Spectral", "Compute:Heat", "Compute:Bandpass"],
        ["Original", "Pixel:Posterize", "Pixel:Edges", "Pixel:Contrast"],
        ["Original", "Spectral⊗Poster", "Heat⊗Edges", "Band⊗Contrast"],
    ]
    for row_idx, labels in enumerate(row_labels):
        for col_idx, lbl in enumerate(labels):
            draw.text((col_idx * W + 5, row_idx * H + 5), lbl, fill=(255, 80, 80))

    labeled.save(Path(output_dir) / f"smooth_compute_{stem}_labeled.png")
    print(f"Saved labeled")


if __name__ == "__main__":
    out_dir = "demo_output"
    for inp_name in ["toof.png", "snek-heavy.png", "1bit redraw.png"]:
        inp = Path("demo_output/inputs") / inp_name
        if inp.exists():
            run_demo(str(inp), out_dir)
