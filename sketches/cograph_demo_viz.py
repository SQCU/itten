"""
Visualization: show co-graph structure and spectral modulation at different thetas.
"""
import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from image_cograph_spectral import (
    extract_contour_points, voronoi_graph, pixel_to_cell_map,
    aggregate_to_cells, scatter_to_pixels
)
from spectral_ops_fast import iterative_spectral_transform, DEVICE


def draw_voronoi_edges(img_rgb: np.ndarray, seeds: np.ndarray, graph) -> np.ndarray:
    """Draw co-graph edges on image."""
    out = img_rgb.copy()
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    adj = graph.adjacency.coalesce()
    indices = adj.indices().cpu().numpy()
    for k in range(indices.shape[1]):
        i, j = indices[0, k], indices[1, k]
        if i < j:
            x1, y1 = seeds[i]
            x2, y2 = seeds[j]
            draw.line([(x1, y1), (x2, y2)], fill=(255, 100, 100), width=1)
    for x, y in seeds:
        draw.ellipse([x-2, y-2, x+2, y+2], fill=(255, 50, 50))
    return np.array(pil)


def run_viz(input_path: str, output_path: str, n_seeds: int = 60):
    img = np.array(Image.open(input_path).convert('L')).astype(np.float32) / 255.0
    h, w = img.shape
    img_rgb = np.stack([img]*3, axis=-1)

    seeds = extract_contour_points(img, n_seeds)
    cograph = voronoi_graph(seeds, h, w)
    L = cograph.laplacian(normalized=True)
    cell_map = pixel_to_cell_map(seeds, h, w)
    signal = aggregate_to_cells(img, cell_map, len(seeds))

    # Panel 1: original
    p1 = (img_rgb * 255).astype(np.uint8)

    # Panel 2: co-graph overlay
    p2 = draw_voronoi_edges((img_rgb * 255).astype(np.uint8), seeds, cograph)

    # Panels 3-5: different theta values
    thetas = [0.2, 0.5, 0.8]
    panels = [p1, p2]
    for theta in thetas:
        transformed = iterative_spectral_transform(L, signal, theta=theta, num_steps=4)
        mod_mask = scatter_to_pixels(transformed, cell_map)
        mod_mask = (mod_mask - mod_mask.min()) / (mod_mask.max() - mod_mask.min() + 1e-8)
        out = img * (0.2 + 0.8 * mod_mask)
        out_rgb = np.stack([out]*3, axis=-1)
        panels.append((out_rgb * 255).clip(0, 255).astype(np.uint8))

    # Concatenate horizontally
    combined = np.concatenate(panels, axis=1)
    Image.fromarray(combined).save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    for inp_name in ["toof.png", "snek-heavy.png", "1bit redraw.png"]:
        inp = Path("demo_output/inputs") / inp_name
        if inp.exists():
            out = Path("demo_output") / f"cograph_viz_{inp.stem}.png"
            run_viz(str(inp), str(out))
