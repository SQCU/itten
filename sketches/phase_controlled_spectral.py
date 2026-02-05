"""
Phase-controlled spectral gating.

The eigenvector phase field controls WHICH signal dominates at each pixel,
not by summing but by angular interpolation:

    output = cos(phase) * signal_A + sin(phase) * signal_B

This is multiplicative/selective control, not additive blending.
The phase vortices around nodal points create natural switching boundaries.
"""
import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage import sobel, distance_transform_edt
from image_cograph_spectral import (
    extract_contour_points, voronoi_graph, pixel_to_cell_map,
    aggregate_to_cells, scatter_to_pixels
)
from spectral_ops_fast import (
    Graph, iterative_spectral_transform, DEVICE,
    build_weighted_image_laplacian, lanczos_k_eigenvectors
)


def compute_phase_field(img: np.ndarray, eigenpair: tuple = (0, 1)) -> tuple:
    """
    Eigenvector phase field from compendium Transform 4.
    Returns (phase, magnitude) where phase is in [-pi, pi].
    """
    H, W = img.shape
    carrier = torch.tensor(img, dtype=torch.float32, device=DEVICE)
    L = build_weighted_image_laplacian(carrier, edge_threshold=0.1)

    evs, _ = lanczos_k_eigenvectors(L, num_eigenvectors=max(eigenpair) + 1)

    # evs may be numpy or torch depending on version
    if hasattr(evs, 'cpu'):
        evs = evs.cpu().numpy()
    ev_real = evs[:, eigenpair[0]].reshape(H, W)
    ev_imag = evs[:, eigenpair[1]].reshape(H, W)

    phase = np.arctan2(ev_imag, ev_real)  # [-pi, pi]
    magnitude = np.sqrt(ev_real**2 + ev_imag**2)

    return phase, magnitude


def compute_sdf(img_gray: np.ndarray) -> np.ndarray:
    """SDF from contours."""
    gx, gy = sobel(img_gray, axis=1), sobel(img_gray, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    contour_mask = grad_mag > np.percentile(grad_mag, 85)
    dist_out = distance_transform_edt(~contour_mask)
    dist_in = distance_transform_edt(contour_mask)
    return dist_out - dist_in


def run_phase_controlled(input_path: str, output_dir: str, n_seeds: int = 70,
                         thetas: list = [0.3, 0.5, 0.7]):
    """
    Pipeline: phase field controls blend between:
      - signal_A: raw co-graph aggregated image intensity
      - signal_B: SDF-weighted + spectral transformed signal

    The phase field acts as a selector, not an additive effect.
    """
    img = np.array(Image.open(input_path).convert('L')).astype(np.float32) / 255.0
    H, W = img.shape

    # === Build ADTs ===
    seeds = extract_contour_points(img, n_seeds)
    cograph = voronoi_graph(seeds, H, W)
    L_cograph = cograph.laplacian(normalized=True)
    cell_map = pixel_to_cell_map(seeds, H, W)

    # SDF for weighting
    sdf = compute_sdf(img)
    sdf_norm = (sdf - sdf.min()) / (sdf.max() - sdf.min() + 1e-8)

    # Phase field for control
    phase, phase_mag = compute_phase_field(img, eigenpair=(0, 1))

    # === Two signals to select between ===
    # Signal A: raw image intensity on co-graph (no spectral transform)
    signal_A = aggregate_to_cells(img, cell_map, len(seeds))
    signal_A_pixels = scatter_to_pixels(signal_A, cell_map)
    signal_A_pixels = (signal_A_pixels - signal_A_pixels.min()) / (signal_A_pixels.max() - signal_A_pixels.min() + 1e-8)

    # SDF weights for signal B (keep on CPU for numpy ops)
    sdf_weights = (
        np.bincount(cell_map.ravel(), weights=sdf_norm.ravel(), minlength=len(seeds)) /
        (np.bincount(cell_map.ravel(), minlength=len(seeds)) + 1e-8)
    )
    signal_A_np = signal_A.cpu().numpy() if hasattr(signal_A, 'cpu') else signal_A
    signal_B_base = signal_A_np * (0.3 + 0.7 * sdf_weights)

    panels = []

    # Panel 1: Original
    panels.append((np.stack([img]*3, axis=-1) * 255).astype(np.uint8))

    # Panel 2: Phase field visualization (hue from phase)
    phase_viz = (phase + np.pi) / (2 * np.pi)  # [0, 1]
    panels.append((np.stack([phase_viz]*3, axis=-1) * 255).astype(np.uint8))

    # Panel 3: Signal A (raw, no transform)
    panels.append((np.stack([signal_A_pixels]*3, axis=-1) * 255).astype(np.uint8))

    # Panels 4+: Phase-controlled blend at different thetas
    for theta in thetas:
        # Signal B: SDF-weighted + spectral transformed
        signal_B_torch = torch.tensor(signal_B_base, dtype=torch.float32, device=DEVICE)
        transformed = iterative_spectral_transform(L_cograph, signal_B_torch, theta=theta, num_steps=5)
        signal_B_pixels = scatter_to_pixels(transformed, cell_map)
        signal_B_pixels = (signal_B_pixels - signal_B_pixels.min()) / (signal_B_pixels.max() - signal_B_pixels.min() + 1e-8)

        # === PHASE CONTROL: angular interpolation, not sum ===
        # cos(phase) in [−1, 1], shift to [0, 1] for blend weight
        blend_weight = (np.cos(phase) + 1) / 2  # 0 = pure B, 1 = pure A

        # The phase field SELECTS between A and B
        controlled = blend_weight * signal_A_pixels + (1 - blend_weight) * signal_B_pixels

        # Apply to original image as gating (multiplicative, not additive)
        output = img * (0.1 + 0.9 * controlled)
        panels.append((np.stack([output]*3, axis=-1) * 255).clip(0, 255).astype(np.uint8))

    # Concatenate
    combined = np.concatenate(panels, axis=1)
    stem = Path(input_path).stem
    out_path = Path(output_dir) / f"phase_ctrl_{stem}.png"
    Image.fromarray(combined).save(out_path)
    print(f"Saved: {out_path}")

    # Labeled version
    labeled = Image.fromarray(combined)
    draw = ImageDraw.Draw(labeled)
    labels = ["Original", "Phase Field", "Signal A (raw)",
              f"Phase×(A,B) θ={thetas[0]}", f"θ={thetas[1]}", f"θ={thetas[2]}"]
    for i, lbl in enumerate(labels):
        draw.text((i * W + 5, 5), lbl, fill=(255, 80, 80))
    labeled_path = Path(output_dir) / f"phase_ctrl_{stem}_labeled.png"
    labeled.save(labeled_path)
    print(f"Saved: {labeled_path}")


if __name__ == "__main__":
    out_dir = "demo_output"
    for inp_name in ["toof.png", "snek-heavy.png", "1bit redraw.png"]:
        inp = Path("demo_output/inputs") / inp_name
        if inp.exists():
            run_phase_controlled(str(inp), out_dir, n_seeds=80, thetas=[0.25, 0.5, 0.75])
