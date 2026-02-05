"""
Spectral connected components: the answer is already in φ.

If a graph has c connected components, eigenvalue 0 has multiplicity c.
The corresponding eigenvectors are indicator functions for each component.
No BFS/DFS needed - just read the null space.
"""
import torch
import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple


def build_test_graph_with_components() -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Build a graph with 3 visually distinct connected components.

    Returns:
        coords: (N, 2) node positions
        adjacency: (N, N) sparse adjacency matrix
        true_labels: ground truth component labels
    """
    device = torch.device('cpu')

    # Component 0: small cluster (nodes 0-4)
    c0_coords = torch.tensor([
        [1.0, 1.0], [1.5, 1.2], [1.2, 1.6], [0.8, 1.4], [1.3, 0.8]
    ], device=device)
    c0_edges = [(0,1), (1,2), (2,3), (3,0), (0,4), (1,4)]

    # Component 1: line segment (nodes 5-9)
    c1_coords = torch.tensor([
        [3.0, 0.5], [3.5, 0.7], [4.0, 0.9], [4.5, 1.1], [5.0, 1.3]
    ], device=device)
    c1_edges = [(5,6), (6,7), (7,8), (8,9)]

    # Component 2: triangle (nodes 10-12)
    c2_coords = torch.tensor([
        [2.0, 2.5], [2.8, 2.3], [2.4, 3.0]
    ], device=device)
    c2_edges = [(10,11), (11,12), (12,10)]

    # Combine
    coords = torch.cat([c0_coords, c1_coords, c2_coords], dim=0)
    N = coords.shape[0]  # 13 nodes

    all_edges = c0_edges + c1_edges + c2_edges

    # Build sparse adjacency
    rows, cols = [], []
    for i, j in all_edges:
        rows.extend([i, j])
        cols.extend([j, i])

    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.ones(len(rows), dtype=torch.float32, device=device)
    adjacency = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

    # Ground truth labels
    true_labels = torch.tensor([0,0,0,0,0, 1,1,1,1,1, 2,2,2], dtype=torch.long, device=device)

    return coords, adjacency, true_labels


def laplacian_from_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Compute Laplacian L = D - A from sparse adjacency."""
    N = adj.shape[0]
    device = adj.device

    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()

    # Degree via scatter
    degrees = torch.zeros(N, device=device)
    degrees.scatter_add_(0, indices[0], values)

    # L = D - A: negate off-diagonal, add diagonal
    L_indices = torch.cat([indices, torch.stack([torch.arange(N, device=device)]*2)], dim=1)
    L_values = torch.cat([-values, degrees])

    L = torch.sparse_coo_tensor(L_indices, L_values, (N, N)).coalesce()
    return L


def eigenvectors_dense(L: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute smallest k eigenvectors via dense eigendecomposition.
    (For small demo graphs; use Lanczos for large graphs)
    """
    L_dense = L.to_dense()
    eigenvalues, eigenvectors = torch.linalg.eigh(L_dense)
    return eigenvalues[:k], eigenvectors[:, :k]


def components_from_spectrum(
    eigenvalues: torch.Tensor,  # (k,)
    eigenvectors: torch.Tensor, # (N, k)
    tol: float = 1e-6
) -> Tuple[torch.Tensor, int]:
    """
    Read connected components from spectral decomposition.

    Eigenvectors with eigenvalue ≈ 0 are component indicator functions.
    Each node belongs to whichever indicator is non-zero there.

    Returns:
        labels: (N,) component labels
        n_components: number of components found
    """
    null_mask = eigenvalues.abs() < tol
    n_components = null_mask.sum().item()

    if n_components == 0:
        # Graph is connected, everything is component 0
        return torch.zeros(eigenvectors.shape[0], dtype=torch.long), 1

    # Null space eigenvectors are component indicators
    indicators = eigenvectors[:, null_mask]  # (N, c)

    # Each node's component = which indicator has largest magnitude
    labels = indicators.abs().argmax(dim=-1)  # (N,)

    return labels, n_components


def demo_spectral_components():
    """
    Demonstrate: eigenvalues near 0 reveal connected components,
    and their eigenvectors are indicator functions we can just read.
    """
    print("=" * 60)
    print("SPECTRAL CONNECTED COMPONENTS DEMO")
    print("=" * 60)

    # Build test graph
    coords, adj, true_labels = build_test_graph_with_components()
    N = coords.shape[0]

    print(f"\nGraph: {N} nodes, 3 true components")
    print(f"  Component 0: nodes 0-4 (cluster)")
    print(f"  Component 1: nodes 5-9 (line)")
    print(f"  Component 2: nodes 10-12 (triangle)")

    # Compute Laplacian
    L = laplacian_from_adjacency(adj)

    # Compute eigenvectors (use k=6 to see structure)
    eigenvalues, eigenvectors = eigenvectors_dense(L, k=6)

    print(f"\nFirst 6 eigenvalues:")
    for i, ev in enumerate(eigenvalues):
        marker = " ← near zero (component indicator)" if ev.abs() < 1e-6 else ""
        print(f"  λ_{i} = {ev:.6f}{marker}")

    # Extract components from spectrum
    labels, n_found = components_from_spectrum(eigenvalues, eigenvectors, tol=1e-6)

    print(f"\nFound {n_found} components from null space")
    print(f"\nNode labels (spectral vs true):")
    print(f"  Node:     {' '.join(f'{i:3d}' for i in range(N))}")
    print(f"  Spectral: {' '.join(f'{l:3d}' for l in labels.tolist())}")
    print(f"  True:     {' '.join(f'{l:3d}' for l in true_labels.tolist())}")

    # Check if labeling matches (up to permutation)
    # Components might be numbered differently, but structure should match
    match = True
    for c in range(3):
        true_mask = true_labels == c
        spectral_vals = labels[true_mask].unique()
        if len(spectral_vals) != 1:
            match = False
            break

    print(f"\nComponent structure matches: {'✓' if match else '✗'}")

    # Show eigenvector values (the indicators)
    print(f"\nNull-space eigenvector values (component indicators):")
    null_mask = eigenvalues.abs() < 1e-6
    null_vecs = eigenvectors[:, null_mask]
    print(f"  {'Node':<6}", end="")
    for i in range(null_vecs.shape[1]):
        print(f"{'φ_' + str(i):<10}", end="")
    print("  Component")
    print("  " + "-" * 50)
    for n in range(N):
        print(f"  {n:<6}", end="")
        for i in range(null_vecs.shape[1]):
            print(f"{null_vecs[n, i]:< 10.4f}", end="")
        print(f"  {labels[n].item()}")

    # Visualize
    visualize_components(coords, adj, labels, true_labels, eigenvalues, eigenvectors)

    return coords, adj, labels, eigenvalues, eigenvectors


def visualize_components(coords, adj, spectral_labels, true_labels, eigenvalues, eigenvectors):
    """Draw the graph with components colored."""
    coords_np = coords.numpy()

    # Image setup with equal aspect ratio
    x_min, x_max = coords_np[:, 0].min() - 0.5, coords_np[:, 0].max() + 0.5
    y_min, y_max = coords_np[:, 1].min() - 0.5, coords_np[:, 1].max() + 0.5

    ppu = 80  # pixels per unit
    margin = 40

    w = int((x_max - x_min) * ppu) + 2 * margin
    h = int((y_max - y_min) * ppu) + 2 * margin

    img = Image.new('RGB', (w, h), 'white')
    draw = ImageDraw.Draw(img)

    def to_px(x, y):
        px = margin + (x - x_min) * ppu
        py = margin + (y_max - y) * ppu
        return int(px), int(py)

    colors = [(220, 60, 60), (60, 180, 60), (60, 60, 220)]

    # Draw edges
    adj_dense = adj.to_dense()
    N = coords_np.shape[0]
    for i in range(N):
        for j in range(i+1, N):
            if adj_dense[i, j] > 0:
                p1 = to_px(coords_np[i, 0], coords_np[i, 1])
                p2 = to_px(coords_np[j, 0], coords_np[j, 1])
                c = spectral_labels[i].item()
                draw.line([p1, p2], fill=colors[c], width=2)

    # Draw nodes
    for i in range(N):
        px, py = to_px(coords_np[i, 0], coords_np[i, 1])
        c = spectral_labels[i].item()
        draw.ellipse([px-8, py-8, px+8, py+8], fill=colors[c], outline=(0,0,0))
        draw.text((px-3, py-6), str(i), fill=(255,255,255))

    # Title
    draw.text((10, 5), "Connected components from spectrum (no BFS/DFS)", fill=(0,0,0))

    # Legend
    n_null = (eigenvalues.abs() < 1e-6).sum().item()
    draw.text((10, h-60), f"Eigenvalues ≈ 0: {n_null} (= num components)", fill=(0,0,0))
    draw.text((10, h-40), "Colors = component labels read from null-space eigenvectors", fill=(0,0,0))
    draw.text((10, h-20), "No iteration, no recursion - just φ[:, λ≈0].argmax()", fill=(0,0,0))

    img.save('demo_output/spectral_components.png')
    print(f"\nSaved: demo_output/spectral_components.png")


if __name__ == "__main__":
    demo_spectral_components()
