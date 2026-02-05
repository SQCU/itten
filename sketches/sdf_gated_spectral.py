"""
ADT composition: SDF gates co-graph signal before spectral transform.

Pipeline:
  Image → Contours → SDF (distance field)
                  → Voronoi co-graph
  SDF weights co-graph node signal → spectral transform → modulate image

This demonstrates ADT-gating-ADT: the SDF (one ADT derived from image)
modulates the input to the spectral transform (another ADT operation).
"""
import torch
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, sobel
from pathlib import Path
from image_cograph_spectral import (
    extract_contour_points, voronoi_graph, pixel_to_cell_map,
    aggregate_to_cells, scatter_to_pixels
)
from spectral_ops_fast import iterative_spectral_transform, DEVICE


def compute_sdf(img_gray: np.ndarray, threshold_pct: float = 85) -> np.ndarray:
    """Compute signed distance field from contours."""
    gx, gy = sobel(img_gray, axis=1), sobel(img_gray, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    threshold = np.percentile(grad_mag, threshold_pct)
    contour_mask = grad_mag > threshold

    # Distance from contour (unsigned)
    dist_outside = distance_transform_edt(~contour_mask)
    dist_inside = distance_transform_edt(contour_mask)

    # SDF: positive outside contour, negative inside
    sdf = dist_outside - dist_inside
    return sdf


def sdf_to_cell_weights(sdf: np.ndarray, cell_map: np.ndarray, n_cells: int) -> torch.Tensor:
    """Average SDF values per Voronoi cell → weight per node."""
    weights = np.zeros(n_cells)
    counts = np.zeros(n_cells)
    flat_sdf, flat_map = sdf.ravel(), cell_map.ravel()
    np.add.at(weights, flat_map, flat_sdf)
    np.add.at(counts, flat_map, 1)
    counts[counts == 0] = 1
    avg = weights / counts
    # Normalize to [0, 1] for gating
    avg = (avg - avg.min()) / (avg.max() - avg.min() + 1e-8)
    return torch.tensor(avg, dtype=torch.float32, device=DEVICE)


def run_sdf_gated(input_path: str, output_dir: str, n_seeds: int = 70,
                  thetas: list = [0.3, 0.5, 0.7]):
    """Full pipeline with SDF gating, theta sweep."""
    img = np.array(Image.open(input_path).convert('L')).astype(np.float32) / 255.0
    h, w = img.shape

    # Derive ADTs from image
    seeds = extract_contour_points(img, n_seeds)
    sdf = compute_sdf(img)
    cograph = voronoi_graph(seeds, h, w)
    L = cograph.laplacian(normalized=True)
    cell_map = pixel_to_cell_map(seeds, h, w)

    # Base signal: image intensity per cell
    base_signal = aggregate_to_cells(img, cell_map, len(seeds))

    # SDF gating weights per cell
    sdf_weights = sdf_to_cell_weights(sdf, cell_map, len(seeds))

    # Gated signal: SDF modulates the base signal before spectral transform
    gated_signal = base_signal * (0.3 + 0.7 * sdf_weights)

    panels = []

    # Panel 1: Original
    panels.append((np.stack([img]*3, axis=-1) * 255).astype(np.uint8))

    # Panel 2: SDF visualization
    sdf_viz = (sdf - sdf.min()) / (sdf.max() - sdf.min() + 1e-8)
    panels.append((np.stack([sdf_viz]*3, axis=-1) * 255).astype(np.uint8))

    # Panel 3: SDF-weighted co-graph signal (no spectral yet)
    gated_raw = scatter_to_pixels(gated_signal, cell_map)
    gated_raw = (gated_raw - gated_raw.min()) / (gated_raw.max() - gated_raw.min() + 1e-8)
    panels.append((np.stack([gated_raw]*3, axis=-1) * 255).astype(np.uint8))

    # Panels 4+: Spectral transform at different thetas
    for theta in thetas:
        transformed = iterative_spectral_transform(L, gated_signal, theta=theta, num_steps=5)
        mod_mask = scatter_to_pixels(transformed, cell_map)
        mod_mask = (mod_mask - mod_mask.min()) / (mod_mask.max() - mod_mask.min() + 1e-8)
        output = img * (0.15 + 0.85 * mod_mask)
        panels.append((np.stack([output]*3, axis=-1) * 255).clip(0,255).astype(np.uint8))

    # Concatenate and save
    combined = np.concatenate(panels, axis=1)
    stem = Path(input_path).stem
    out_path = Path(output_dir) / f"sdf_gated_{stem}.png"
    Image.fromarray(combined).save(out_path)
    print(f"Saved: {out_path}")

    # Also save labeled version
    labeled = Image.fromarray(combined)
    draw = ImageDraw.Draw(labeled)
    labels = ["Original", "SDF", "SDF*Signal", f"θ={thetas[0]}", f"θ={thetas[1]}", f"θ={thetas[2]}"]
    for i, lbl in enumerate(labels):
        draw.text((i * w + 5, 5), lbl, fill=(255, 80, 80))
    labeled_path = Path(output_dir) / f"sdf_gated_{stem}_labeled.png"
    labeled.save(labeled_path)
    print(f"Saved: {labeled_path}")

    return combined


if __name__ == "__main__":
    out_dir = "demo_output"
    for inp_name in ["toof.png", "snek-heavy.png", "1bit redraw.png"]:
        inp = Path("demo_output/inputs") / inp_name
        if inp.exists():
            run_sdf_gated(str(inp), out_dir, n_seeds=80, thetas=[0.25, 0.5, 0.75])
