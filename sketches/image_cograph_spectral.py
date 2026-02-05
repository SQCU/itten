"""
Image → Contour → Voronoi co-graph → Spectral transform → Modulate image.

The idea: derive a sparse graph FROM the dense pixel grid, then use spectral
transforms on the sparse graph to gate/modulate the original image.
"""
import torch
import numpy as np
from PIL import Image
from scipy.spatial import Voronoi
from scipy.ndimage import sobel, label
from pathlib import Path
from spectral_ops_fast import Graph, iterative_spectral_transform, DEVICE


def extract_contour_points(img_gray: np.ndarray, n_points: int = 64) -> np.ndarray:
    """Get contour points via gradient magnitude thresholding."""
    gx, gy = sobel(img_gray, axis=1), sobel(img_gray, axis=0)
    grad_mag = np.sqrt(gx**2 + gy**2)
    threshold = np.percentile(grad_mag, 85)
    contour_mask = grad_mag > threshold
    ys, xs = np.where(contour_mask)
    if len(xs) < n_points:
        return np.column_stack([xs, ys])
    indices = np.random.choice(len(xs), n_points, replace=False)
    return np.column_stack([xs[indices], ys[indices]])


def voronoi_graph(seeds: np.ndarray, h: int, w: int) -> Graph:
    """Build sparse graph from Voronoi cell adjacency."""
    vor = Voronoi(seeds)
    n = len(seeds)
    edges = set()
    for ridge in vor.ridge_points:
        i, j = ridge
        if i >= 0 and j >= 0 and i < n and j < n:
            edges.add((min(i,j), max(i,j)))
    if not edges:
        # fallback: connect each seed to nearest neighbors
        from scipy.spatial import cKDTree
        tree = cKDTree(seeds)
        for i in range(n):
            _, idx = tree.query(seeds[i], k=4)
            for j in idx[1:]:
                edges.add((min(i,j), max(i,j)))
    rows, cols, vals = [], [], []
    for i, j in edges:
        rows.extend([i, j]); cols.extend([j, i]); vals.extend([1.0, 1.0])
    indices = torch.tensor([rows, cols], device=DEVICE)
    values = torch.tensor(vals, dtype=torch.float32, device=DEVICE)
    adj = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    coords = torch.tensor(seeds, dtype=torch.float32, device=DEVICE)
    return Graph(adjacency=adj, coords=coords)


def pixel_to_cell_map(seeds: np.ndarray, h: int, w: int) -> np.ndarray:
    """Map each pixel to its nearest Voronoi cell."""
    from scipy.spatial import cKDTree
    tree = cKDTree(seeds)
    yy, xx = np.mgrid[0:h, 0:w]
    pixels = np.column_stack([xx.ravel(), yy.ravel()])
    _, cell_ids = tree.query(pixels)
    return cell_ids.reshape(h, w)


def aggregate_to_cells(img: np.ndarray, cell_map: np.ndarray, n_cells: int) -> torch.Tensor:
    """Average pixel values per Voronoi cell → signal on graph nodes."""
    signal = np.zeros(n_cells)
    counts = np.zeros(n_cells)
    flat_img, flat_map = img.ravel(), cell_map.ravel()
    np.add.at(signal, flat_map, flat_img)
    np.add.at(counts, flat_map, 1)
    counts[counts == 0] = 1
    return torch.tensor(signal / counts, dtype=torch.float32, device=DEVICE)


def scatter_to_pixels(cell_signal: torch.Tensor, cell_map: np.ndarray) -> np.ndarray:
    """Expand cell values back to pixel grid."""
    sig_np = cell_signal.cpu().numpy()
    return sig_np[cell_map]


def run(input_path: str, output_path: str, n_seeds: int = 64, theta: float = 0.5):
    """Main pipeline: image → cograph spectral → modulated image."""
    img = np.array(Image.open(input_path).convert('L')).astype(np.float32) / 255.0
    h, w = img.shape

    # 1. Extract contour points as Voronoi seeds
    seeds = extract_contour_points(img, n_seeds)

    # 2. Build sparse Voronoi co-graph
    cograph = voronoi_graph(seeds, h, w)
    L = cograph.laplacian(normalized=True)

    # 3. Map pixels to cells, aggregate to graph signal
    cell_map = pixel_to_cell_map(seeds, h, w)
    signal = aggregate_to_cells(img, cell_map, len(seeds))

    # 4. Spectral transform on sparse co-graph
    transformed = iterative_spectral_transform(L, signal, theta=theta, num_steps=4)

    # 5. Scatter back to pixels → modulation mask
    mod_mask = scatter_to_pixels(transformed, cell_map)
    mod_mask = (mod_mask - mod_mask.min()) / (mod_mask.max() - mod_mask.min() + 1e-8)

    # 6. Modulate original image (gating)
    output = img * (0.3 + 0.7 * mod_mask)  # lerp between dimmed and full

    # Save
    out_img = Image.fromarray((output * 255).clip(0, 255).astype(np.uint8))
    out_img.save(output_path)
    print(f"Saved: {output_path}")
    return output, mod_mask, seeds


if __name__ == "__main__":
    inp = Path("demo_output/inputs/toof.png")
    out = Path("demo_output/cograph_spectral_output.png")
    run(str(inp), str(out), n_seeds=80, theta=0.6)
