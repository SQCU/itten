"""
V-cell as compute shader: the spectral transform IS the transformation, not a mask.

Instead of: output = image * spectral_mask  (summing)
Do:         output = scatter(spectral_transform(L_vcell, signal))

The Voronoi graph from image B becomes a compute kernel that transforms image A.
The transformed signal IS the output.
"""
import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from scipy.ndimage import sobel, zoom
from scipy.spatial import Voronoi, cKDTree
from spectral_ops_fast import Graph, iterative_spectral_transform, DEVICE


def load_img(path, size):
    img = np.array(Image.open(path).convert('L')).astype(np.float32) / 255.0
    H, W = size
    h, w = img.shape
    if (h, w) != (H, W):
        img = zoom(img, (H/h, W/w), order=1)[:H, :W]
    return img


def make_checker(H, W, cell=24):
    c = np.zeros((H, W), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            if ((y // cell) + (x // cell)) % 2 == 0:
                c[y, x] = 1.0
    return c


def build_vcell_kernel(img_kernel, n_seeds, H, W):
    """Build Voronoi graph from kernel image - this IS the compute kernel."""
    gx, gy = sobel(img_kernel, axis=1), sobel(img_kernel, axis=0)
    grad = np.sqrt(gx**2 + gy**2)
    ys, xs = np.where(grad > np.percentile(grad, 85))
    if len(xs) < n_seeds:
        ys = np.concatenate([ys, np.random.randint(0, H, n_seeds - len(ys))])
        xs = np.concatenate([xs, np.random.randint(0, W, n_seeds - len(xs))])
    idx = np.random.choice(len(xs), n_seeds, replace=False)
    seeds = np.column_stack([xs[idx], ys[idx]])

    # Voronoi adjacency
    vor = Voronoi(seeds)
    edges = set()
    for ridge in vor.ridge_points:
        i, j = ridge
        if 0 <= i < n_seeds and 0 <= j < n_seeds:
            edges.add((min(i,j), max(i,j)))
    if not edges:
        tree = cKDTree(seeds)
        for i in range(n_seeds):
            _, nbrs = tree.query(seeds[i], k=4)
            for j in nbrs[1:]:
                edges.add((min(i,j), max(i,j)))

    rows, cols, vals = [], [], []
    for i, j in edges:
        rows.extend([i, j]); cols.extend([j, i]); vals.extend([1.0, 1.0])

    adj = torch.sparse_coo_tensor(
        torch.tensor([rows, cols], device=DEVICE),
        torch.tensor(vals, dtype=torch.float32, device=DEVICE),
        (n_seeds, n_seeds)
    ).coalesce()

    graph = Graph(adjacency=adj, coords=torch.tensor(seeds, dtype=torch.float32, device=DEVICE))
    L = graph.laplacian(normalized=True)

    # Cell map for pixel shader (scatter)
    tree = cKDTree(seeds)
    yy, xx = np.mgrid[0:H, 0:W]
    _, cell_ids = tree.query(np.column_stack([xx.ravel(), yy.ravel()]))
    cell_map = cell_ids.reshape(H, W)

    return L, cell_map, n_seeds


def aggregate(img, cell_map, n_nodes):
    """Pixel shader: image -> graph signal."""
    signal = np.bincount(cell_map.ravel(), weights=img.ravel(), minlength=n_nodes)
    counts = np.bincount(cell_map.ravel(), minlength=n_nodes).astype(float)
    counts[counts == 0] = 1
    return torch.tensor(signal / counts, dtype=torch.float32, device=DEVICE)


def scatter(signal, cell_map):
    """Pixel shader: graph signal -> image."""
    return signal.cpu().numpy()[cell_map]


def compute_shader(L_kernel, signal_input, theta):
    """
    THE compute shader: spectral transform on V-cell graph.
    L_kernel defines the kernel structure, signal_input is transformed through it.
    """
    return iterative_spectral_transform(L_kernel, signal_input, theta, num_steps=8)


def run(output_dir):
    H, W = 256, 256
    n_seeds = 80
    thetas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Input images
    img_a = load_img('demo_output/inputs/snek-heavy.png', (H, W))
    img_b = load_img('demo_output/inputs/toof.png', (H, W))
    checker = make_checker(H, W, 24)
    amongus = load_img('demo_output/3d_bump_sweep/carriers/amongus.png', (H, W))

    pairs = [
        ('snek', img_a, 'checker', checker),
        ('snek', img_a, 'amongus', amongus),
        ('toof', img_b, 'amongus', amongus),
        ('snek', img_a, 'toof', img_b),
    ]

    for name_in, img_input, name_kernel, img_kernel in pairs:
        # Build V-cell compute kernel from kernel image
        L_kernel, cell_map, n_nodes = build_vcell_kernel(img_kernel, n_seeds, H, W)

        # Aggregate input image to kernel's graph nodes
        signal_input = aggregate(img_input, cell_map, n_nodes)

        rows = []

        # Row 1: input, kernel, raw aggregated (no spectral)
        raw_agg = scatter(signal_input, cell_map)
        row1 = [img_input, img_kernel, raw_agg]

        # Row 2: theta sweep - compute shader output (NOT multiplied with original!)
        row2 = []
        for theta in thetas:
            # THE COMPUTE SHADER: spectral transform IS the operation
            transformed = compute_shader(L_kernel, signal_input, theta)
            # THE PIXEL SHADER: scatter to pixels
            output = scatter(transformed, cell_map)
            # Normalize for visualization
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)
            row2.append(output)

        # Pad row1 to match
        while len(row1) < len(row2):
            row1.append(np.ones((H, W)))

        rows = [row1, row2]

        def make_row(imgs):
            return np.concatenate([np.stack([np.clip(x,0,1)]*3, axis=-1) for x in imgs], axis=1)

        grid = np.concatenate([make_row(r) for r in rows], axis=0)
        grid = (grid * 255).clip(0, 255).astype(np.uint8)

        out_path = Path(output_dir) / f"vcell_shader_{name_in}_by_{name_kernel}.png"
        Image.fromarray(grid).save(out_path)
        print(f"Saved: {out_path}")

        # Labels
        labeled = Image.fromarray(grid)
        draw = ImageDraw.Draw(labeled)
        labels = [
            ["input", "kernel", "aggregated"] + [""]*(len(thetas)-3),
            [f"θ={t}" for t in thetas],
        ]
        for ri, row_labels in enumerate(labels):
            for ci, lbl in enumerate(row_labels):
                draw.text((ci*W+3, ri*H+3), lbl, fill=(255,50,50))
        labeled.save(Path(output_dir) / f"vcell_shader_{name_in}_by_{name_kernel}_labeled.png")


if __name__ == "__main__":
    run("demo_output")
