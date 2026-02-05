"""
Layout algorithms for node positioning.

Provides force-directed and spectral layouts for graph visualization.
All functions are pure - no side effects, deterministic given same input.
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def force_directed_layout(
    nodes: List[int],
    edges: List[Tuple[int, int]],
    positions: Optional[Dict[int, Tuple[float, float]]] = None,
    iterations: int = 50,
    k: float = 1.0,
    temperature: float = 0.1,
    rng_seed: Optional[int] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Compute force-directed layout using Fruchterman-Reingold algorithm.

    Args:
        nodes: List of node IDs
        edges: List of (source, target) tuples
        positions: Optional initial positions
        iterations: Number of iterations
        k: Optimal distance between nodes
        temperature: Initial temperature for simulated annealing
        rng_seed: Random seed for reproducibility

    Returns:
        Dictionary mapping node ID to (x, y) position
    """
    if not nodes:
        return {}

    rng = np.random.default_rng(rng_seed)
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Initialize positions
    if positions is None:
        pos = rng.random((n, 2)) * 2 - 1  # [-1, 1]
    else:
        pos = np.zeros((n, 2))
        for node, (x, y) in positions.items():
            if node in node_to_idx:
                pos[node_to_idx[node]] = [x, y]
        # Random init for new nodes
        for i, node in enumerate(nodes):
            if node not in positions:
                pos[i] = rng.random(2) * 2 - 1

    # Build adjacency for quick lookup
    adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for src, tgt in edges:
        if src in node_to_idx and tgt in node_to_idx:
            i, j = node_to_idx[src], node_to_idx[tgt]
            adj[i].add(j)
            adj[j].add(i)

    # Optimal distance
    area = 4.0  # [-1, 1] x [-1, 1]
    k_optimal = k * np.sqrt(area / max(n, 1))

    temp = temperature
    for iteration in range(iterations):
        disp = np.zeros((n, 2))

        # Repulsive forces between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[i] - pos[j]
                dist = np.linalg.norm(delta)
                if dist < 0.001:
                    delta = rng.random(2) * 0.01
                    dist = np.linalg.norm(delta)

                # Repulsive force: k^2 / d
                force = (k_optimal ** 2 / dist) * (delta / dist)
                disp[i] += force
                disp[j] -= force

        # Attractive forces along edges
        for i, neighbors in adj.items():
            for j in neighbors:
                if j > i:  # Process each edge once
                    delta = pos[i] - pos[j]
                    dist = np.linalg.norm(delta)
                    if dist > 0.001:
                        # Attractive force: d^2 / k
                        force = (dist ** 2 / k_optimal) * (delta / dist)
                        disp[i] -= force
                        disp[j] += force

        # Apply displacement with temperature limit
        for i in range(n):
            disp_norm = np.linalg.norm(disp[i])
            if disp_norm > 0:
                pos[i] += (disp[i] / disp_norm) * min(disp_norm, temp)

        # Cool down
        temp *= 0.95

        # Keep nodes in bounds
        pos = np.clip(pos, -1, 1)

    return {nodes[i]: (float(pos[i, 0]), float(pos[i, 1])) for i in range(n)}


def spectral_layout(
    nodes: List[int],
    edges: List[Tuple[int, int]],
    rng_seed: Optional[int] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Compute spectral layout using eigenvectors of Laplacian.

    Uses the 2nd and 3rd smallest eigenvectors for x, y coordinates.
    Falls back to force-directed if graph is too small or disconnected.

    Args:
        nodes: List of node IDs
        edges: List of (source, target) tuples
        rng_seed: Random seed for reproducibility

    Returns:
        Dictionary mapping node ID to (x, y) position
    """
    if len(nodes) < 3:
        return force_directed_layout(nodes, edges, rng_seed=rng_seed)

    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Build Laplacian matrix
    L = np.zeros((n, n))
    for src, tgt in edges:
        if src in node_to_idx and tgt in node_to_idx:
            i, j = node_to_idx[src], node_to_idx[tgt]
            L[i, j] = -1
            L[j, i] = -1
            L[i, i] += 1
            L[j, j] += 1

    # Compute eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # Use 2nd and 3rd smallest eigenvectors
        # (1st is constant for connected graph)
        x = eigenvectors[:, 1]
        y = eigenvectors[:, 2] if n > 2 else np.zeros(n)

        # Normalize to [-1, 1]
        if np.std(x) > 1e-10:
            x = (x - np.mean(x)) / (2 * np.std(x))
        if np.std(y) > 1e-10:
            y = (y - np.mean(y)) / (2 * np.std(y))

        x = np.clip(x, -1, 1)
        y = np.clip(y, -1, 1)

        return {nodes[i]: (float(x[i]), float(y[i])) for i in range(n)}
    except np.linalg.LinAlgError:
        return force_directed_layout(nodes, edges, rng_seed=rng_seed)


def compute_layout(
    nodes: List[int],
    edges: List[Tuple[int, int]],
    method: str = "force",
    positions: Optional[Dict[int, Tuple[float, float]]] = None,
    iterations: int = 50,
    rng_seed: Optional[int] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Compute node layout using specified method.

    Args:
        nodes: List of node IDs
        edges: List of (source, target) tuples
        method: "force" or "spectral"
        positions: Optional initial positions (for incremental layout)
        iterations: Number of iterations (force-directed only)
        rng_seed: Random seed for reproducibility

    Returns:
        Dictionary mapping node ID to (x, y) position
    """
    if method == "spectral":
        # Spectral layout, then refine with force-directed
        pos = spectral_layout(nodes, edges, rng_seed=rng_seed)
        return force_directed_layout(
            nodes, edges, positions=pos, iterations=iterations // 2, rng_seed=rng_seed
        )
    else:
        return force_directed_layout(
            nodes, edges, positions=positions, iterations=iterations, rng_seed=rng_seed
        )
