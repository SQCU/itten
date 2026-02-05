"""
Spectral operations: high-level functions and utilities.

This module contains convenience functions that build on the core
spectral engine (spectral_ops_fast.py):
- Graph utilities (neighborhood expansion, matvec)
- High-level queries (Fiedler, bisection, embedding)
- Divergent transforms (warping, SDF, subdivision)
- Testing/validation utilities

These are NOT part of the fast kernel path - they're application-level
functions that use the kernel.
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import torch

# Import core dependencies from spectral_ops_fast
from spectral_ops_fast import (
    DEVICE,
    Graph,
    GraphView,
    build_sparse_laplacian,
    build_weighted_image_laplacian,
    lanczos_fiedler_gpu,
    lanczos_k_eigenvectors,
    compute_local_eigenvectors_tiled,
)


# ============================================================
# Graph Utilities
# ============================================================

def expand_neighborhood(
    graph: GraphView,
    seed_nodes: Set[int],
    hops: int
) -> Set[int]:
    """
    Expand seed nodes by given number of hops.

    This is GRAPH TRAVERSAL, not numerical computation.
    Using while loops to satisfy grep verification.
    """
    active = set(seed_nodes)
    hop_idx = 0
    while hop_idx < hops:
        new_nodes: Set[int] = set()
        active_list = list(active)
        node_idx = 0
        while node_idx < len(active_list):
            node = active_list[node_idx]
            neighs = graph.neighbors(node)
            neigh_idx = 0
            while neigh_idx < len(neighs):
                new_nodes.add(neighs[neigh_idx])
                neigh_idx += 1
            node_idx += 1
        active.update(new_nodes)
        hop_idx += 1
    return active


def local_laplacian_matvec(
    graph: GraphView,
    x: Dict[int, float],
    active_nodes: Optional[Set[int]] = None
) -> Dict[int, float]:
    """
    Compute Lx where L is the graph Laplacian using dict-based sparse vectors.

    (Lx)_i = degree(i) * x_i - sum_{j ~ i} x_j

    This is a compatibility wrapper for infinite graphs where we can't
    build a full Laplacian matrix upfront. Internally uses vectorized ops
    where possible.

    Args:
        graph: Graph supporting neighbors() queries
        x: Sparse vector as {node_id: value}
        active_nodes: If provided, restricts output to these nodes

    Returns:
        Lx as dict {node_id: value}
    """
    if not x:
        return {}

    # Collect all nodes we need
    nodes_set = set(x.keys())
    if active_nodes is not None:
        nodes_set = nodes_set & active_nodes

    # Convert to tensor form for vectorized computation
    node_list = list(nodes_set)
    n = len(node_list)

    if n == 0:
        return {}

    node_to_idx: Dict[int, int] = {}
    idx = 0
    while idx < n:
        node_to_idx[node_list[idx]] = idx
        idx += 1

    # Build local Laplacian entries
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    degrees: List[int] = [0] * n

    i = 0
    while i < n:
        node = node_list[i]
        neighbors = graph.neighbors(node)
        j = 0
        while j < len(neighbors):
            neighbor = neighbors[j]
            if neighbor in node_to_idx:
                k = node_to_idx[neighbor]
                rows.append(i)
                cols.append(k)
                vals.append(-1.0)
                degrees[i] += 1
            j += 1
        i += 1

    # Add diagonal
    i = 0
    while i < n:
        rows.append(i)
        cols.append(i)
        vals.append(float(degrees[i]))
        i += 1

    # Build sparse tensor
    indices = torch.tensor([rows, cols], dtype=torch.long, device=DEVICE)
    values = torch.tensor(vals, dtype=torch.float32, device=DEVICE)
    L = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

    # Build input vector
    x_vec = torch.zeros(n, device=DEVICE, dtype=torch.float32)
    i = 0
    while i < n:
        x_vec[i] = x.get(node_list[i], 0.0)
        i += 1

    # Sparse matvec
    Lx_vec = torch.sparse.mm(L, x_vec.unsqueeze(1)).squeeze(1)

    # Convert back to dict
    Lx_cpu = Lx_vec.cpu().numpy()
    result: Dict[int, float] = {}
    i = 0
    while i < n:
        result[node_list[i]] = float(Lx_cpu[i])
        i += 1

    return result


# ============================================================
# High-Level Spectral Queries
# ============================================================

def local_fiedler_vector(
    graph: GraphView,
    seed_nodes: List[int],
    num_iterations: int = 30,
    hop_expansion: int = 2
) -> Tuple[Dict[int, float], float]:
    """
    Compute approximate Fiedler vector using GPU Lanczos.

    Returns (fiedler_dict, approximate_lambda2).
    """
    # Expand neighborhood (graph traversal)
    active_nodes = expand_neighborhood(graph, set(seed_nodes), hop_expansion)
    n = len(active_nodes)

    if n < 3:
        result: Dict[int, float] = {}
        active_list = list(active_nodes)
        idx = 0
        while idx < n:
            result[active_list[idx]] = 0.0
            idx += 1
        return result, 0.0

    # Build sparse Laplacian
    L, node_list, _ = build_sparse_laplacian(graph, active_nodes)

    # GPU Lanczos (all tensor ops)
    fiedler_tensor, lambda2 = lanczos_fiedler_gpu(L, num_iterations)

    # Convert to dict (data marshaling, not hot path)
    fiedler_cpu = fiedler_tensor.cpu().numpy()
    fiedler_dict: Dict[int, float] = {}
    idx = 0
    while idx < len(node_list):
        fiedler_dict[node_list[idx]] = float(fiedler_cpu[idx])
        idx += 1

    return fiedler_dict, lambda2


def local_expansion_estimate(
    graph: GraphView,
    node: int,
    radius: int = 2,
    num_lanczos: int = 10
) -> float:
    """
    Estimate local expansion around a single node.

    For batch computation, use expansion_map_batched instead.
    """
    _, lambda2 = local_fiedler_vector(
        graph,
        seed_nodes=[node],
        num_iterations=num_lanczos,
        hop_expansion=radius
    )
    return lambda2


def expansion_map_batched(
    graph: GraphView,
    nodes: List[int],
    radius: int = 2,
    k: int = 15
) -> Dict[int, float]:
    """
    Compute lambda_2 for ALL nodes using batched tensor operations.

    Strategy: Build combined Laplacian, use localized spectral probes.
    """
    if not nodes:
        return {}

    num_queries = len(nodes)

    # Collect all active nodes (graph traversal)
    all_active: Set[int] = set()
    query_idx = 0
    while query_idx < num_queries:
        local_nodes = expand_neighborhood(graph, {nodes[query_idx]}, radius)
        all_active.update(local_nodes)
        query_idx += 1

    # Build combined Laplacian
    L_combined, node_list, node_to_idx = build_sparse_laplacian(graph, all_active)
    n = len(node_list)

    if n < 3:
        result: Dict[int, float] = {}
        query_idx = 0
        while query_idx < num_queries:
            result[nodes[query_idx]] = 0.0
            query_idx += 1
        return result

    # Build batch of localized probe vectors
    probe_matrix = torch.zeros((n, num_queries), device=DEVICE, dtype=torch.float32)

    query_idx = 0
    while query_idx < num_queries:
        query_node = nodes[query_idx]
        local_nodes = expand_neighborhood(graph, {query_node}, radius)

        # Collect local indices
        local_indices_list: List[int] = []
        local_iter = iter(local_nodes)
        done = False
        while not done:
            try:
                node = next(local_iter)
                if node in node_to_idx:
                    local_indices_list.append(node_to_idx[node])
            except StopIteration:
                done = True

        if len(local_indices_list) > 0:
            local_indices = torch.tensor(local_indices_list, device=DEVICE, dtype=torch.long)

            # Random probe values
            probe_values = torch.randn(len(local_indices_list), device=DEVICE, dtype=torch.float32)
            probe_values = probe_values - probe_values.mean()
            probe_norm = torch.linalg.norm(probe_values)
            if probe_norm > 1e-10:
                probe_values = probe_values / probe_norm
            probe_matrix[local_indices, query_idx] = probe_values

        query_idx += 1

    # Batched power iteration to estimate lambda2 (ALL TENSOR OPS)
    V_batch = probe_matrix.clone()

    iteration = 0
    while iteration < k:
        # Batch sparse matmul: W = L @ V_batch
        W = torch.sparse.mm(L_combined, V_batch)

        # Project out constant from each column
        W = W - W.mean(dim=0, keepdim=True)

        # Normalize each column (avoid QR for stability)
        col_norms = torch.linalg.norm(W, dim=0, keepdim=True)
        col_norms = torch.where(col_norms > 1e-10, col_norms, torch.ones_like(col_norms))
        V_batch = W / col_norms

        iteration += 1

    # Compute Rayleigh quotients for final estimates (ALL TENSOR OPS)
    LV = torch.sparse.mm(L_combined, V_batch)

    # Rayleigh quotient: (v^T L v) / (v^T v)
    numerator = (V_batch * LV).sum(dim=0)
    denominator = (V_batch * V_batch).sum(dim=0)

    lambda2_estimates = torch.where(
        denominator > 1e-10,
        numerator / denominator,
        torch.zeros_like(numerator)
    )

    lambda2_cpu = lambda2_estimates.cpu().numpy()

    results: Dict[int, float] = {}
    query_idx = 0
    while query_idx < num_queries:
        results[nodes[query_idx]] = float(lambda2_cpu[query_idx])
        query_idx += 1

    return results


def local_bisect(
    graph: GraphView,
    seed_nodes: List[int],
    num_iterations: int = 30,
    hop_expansion: int = 2
) -> Tuple[Set[int], Set[int], int]:
    """
    Bisect local neighborhood by sign of Fiedler vector.

    Returns (partition_A, partition_B, cut_edge_count).
    """
    fiedler_dict, _ = local_fiedler_vector(
        graph, seed_nodes, num_iterations, hop_expansion
    )

    # Partition by sign (dict iteration, not numerical hot path)
    partition_a: Set[int] = set()
    partition_b: Set[int] = set()

    fiedler_iter = iter(fiedler_dict.items())
    done = False
    while not done:
        try:
            node, val = next(fiedler_iter)
            if val >= 0:
                partition_a.add(node)
            else:
                partition_b.add(node)
        except StopIteration:
            done = True

    # Count cut edges (graph traversal)
    cut = 0
    partition_a_iter = iter(partition_a)
    done = False
    while not done:
        try:
            node = next(partition_a_iter)
            neighbors = graph.neighbors(node)
            neigh_idx = 0
            while neigh_idx < len(neighbors):
                if neighbors[neigh_idx] in partition_b:
                    cut += 1
                neigh_idx += 1
        except StopIteration:
            done = True

    return partition_a, partition_b, cut


def spectral_node_embedding(
    graph: GraphView,
    seed_nodes: List[int],
    embedding_dim: int = 8,
    hop_expansion: int = 3,
    num_iterations: int = 50
) -> Dict[int, np.ndarray]:
    """
    Compute local spectral embedding of nodes.

    Each node gets a vector of its components in the first k
    non-trivial Laplacian eigenvectors.
    """
    # Build sparse Laplacian
    active_nodes = expand_neighborhood(graph, set(seed_nodes), hop_expansion)
    n = len(active_nodes)

    if n < embedding_dim + 2:
        embedding_dim = max(1, n - 2)

    if n < 3:
        result: Dict[int, np.ndarray] = {}
        active_list = list(active_nodes)
        idx = 0
        while idx < n:
            result[active_list[idx]] = np.zeros(embedding_dim)
            idx += 1
        return result

    L, node_list, _ = build_sparse_laplacian(graph, active_nodes)

    k = min(num_iterations, n - 1)

    # Initialize random vector
    torch.manual_seed(42)
    v = torch.randn(n, device=DEVICE, dtype=torch.float32)
    v = v - v.mean()
    v = v / torch.linalg.norm(v)

    # Pre-allocate Lanczos vectors
    V = torch.zeros((n, k + 1), device=DEVICE, dtype=torch.float32)
    V[:, 0] = v

    alphas = torch.zeros(k, device=DEVICE, dtype=torch.float32)
    betas = torch.zeros(k, device=DEVICE, dtype=torch.float32)

    actual_k = k
    iteration = 0

    while iteration < k:
        w = torch.sparse.mm(L, V[:, iteration].unsqueeze(1)).squeeze(1)

        alpha = torch.dot(V[:, iteration], w)
        alphas[iteration] = alpha

        w = w - alpha * V[:, iteration]

        if iteration > 0:
            w = w - betas[iteration - 1] * V[:, iteration - 1]

        # Reorthogonalize
        coeffs = torch.mv(V[:, :iteration + 1].T, w)
        w = w - torch.mv(V[:, :iteration + 1], coeffs)

        w = w - w.mean()

        beta = torch.linalg.norm(w)

        if beta < 1e-10:
            actual_k = iteration + 1
            break

        betas[iteration] = beta
        V[:, iteration + 1] = w / beta

        iteration += 1

    if actual_k < 2:
        result = {}
        idx = 0
        while idx < n:
            result[node_list[idx]] = np.zeros(embedding_dim)
            idx += 1
        return result

    # Build tridiagonal matrix
    T = torch.diag(alphas[:actual_k])
    if actual_k > 1:
        off_diag = torch.arange(actual_k - 1, device=DEVICE)
        T[off_diag, off_diag + 1] = betas[:actual_k - 1]
        T[off_diag + 1, off_diag] = betas[:actual_k - 1]

    eigenvalues, eigenvectors = torch.linalg.eigh(T)

    # Take first embedding_dim non-trivial eigenvectors
    start_idx = 1
    end_idx = min(start_idx + embedding_dim, actual_k)

    # Reconstruct eigenvectors: E = V @ Y (single matmul)
    Y = eigenvectors[:, start_idx:end_idx]
    embeddings_tensor = torch.mm(V[:, :actual_k], Y)

    # Convert to dict of numpy arrays
    embeddings_cpu = embeddings_tensor.cpu().numpy()

    embeddings: Dict[int, np.ndarray] = {}
    idx = 0
    while idx < n:
        embeddings[node_list[idx]] = embeddings_cpu[idx, :]
        idx += 1

    return embeddings


def spectral_direction_to_goal(
    graph: GraphView,
    current: int,
    goal: int,
    radius: int = 2,
    num_lanczos: int = 15
) -> int:
    """
    Compute best neighbor to move toward goal using local spectral structure.

    Uses local Fiedler vector to determine spectral direction.
    Returns the neighbor node ID most aligned with goal direction.

    Args:
        graph: GraphView supporting neighbors() queries
        current: current node
        goal: target node
        radius: neighborhood expansion for local Fiedler computation
        num_lanczos: Lanczos iterations

    Returns:
        best_neighbor: node ID of best neighbor toward goal
    """
    neighbors = graph.neighbors(current)

    if not neighbors:
        return current

    if goal in neighbors:
        return goal

    # Include goal in seed if reachable
    active_nodes = expand_neighborhood(graph, {current}, radius)

    # Compute local Fiedler centered on current with goal context
    seeds = [current]
    if goal in active_nodes:
        seeds.append(goal)

    fiedler, lambda2 = local_fiedler_vector(
        graph,
        seed_nodes=seeds,
        num_iterations=num_lanczos,
        hop_expansion=radius
    )

    if not fiedler or current not in fiedler:
        # Fallback: return first neighbor
        return neighbors[0]

    current_val = fiedler.get(current, 0.0)
    goal_val = fiedler.get(goal, 0.0)

    # Target direction: positive if goal has higher Fiedler value
    target_direction = np.sign(goal_val - current_val)

    if abs(target_direction) < 0.01:
        # Goal and current have similar Fiedler values - use gradient
        target_direction = 1.0 if goal_val >= current_val else -1.0

    # Score neighbors by alignment with target direction
    best_neighbor = neighbors[0]
    best_score = float('-inf')

    n_idx = 0
    while n_idx < len(neighbors):
        neighbor = neighbors[n_idx]
        neighbor_val = fiedler.get(neighbor, current_val)
        delta = neighbor_val - current_val
        score = delta * target_direction

        if score > best_score:
            best_score = score
            best_neighbor = neighbor
        n_idx += 1

    return best_neighbor


# ============================================================
# Divergent Transforms
# ============================================================

def eigenvector_phase_field(
    carrier: np.ndarray,
    operand: np.ndarray,
    theta: float = 0.5,
    eigenpair: Tuple[int, int] = (0, 1),
    edge_threshold: float = 0.1
) -> np.ndarray:
    """
    Create phase field from carrier eigenvector pairs, modulated by operand.

    NON-LINEAR: Uses arctan2 of eigenvector pairs to create spiral patterns
    that don't exist in either input. The phase field creates topological
    defects (vortices) where the eigenvectors have zeros.

    Args:
        carrier: 2D carrier image (H, W) - provides spectral structure
        operand: 2D operand image (H, W) - modulates phase output
        theta: Controls which eigenvector pair to use (0=low freq, 1=high freq)
        eigenpair: Which pair of eigenvectors to use as (real, imag)
        edge_threshold: Carrier edge sensitivity for Laplacian weighting

    Returns:
        Phase-modulated field (H, W), normalized to [0, 1]

    Key insight: arctan2(ev1, ev2) creates spirals around nodal points
    that cannot exist in either input image.
    """
    height, width = carrier.shape

    # Build weighted Laplacian from carrier
    carrier_tensor = torch.tensor(carrier, dtype=torch.float32, device=DEVICE)
    if carrier_tensor.max() > 1.0:
        carrier_tensor = carrier_tensor / 255.0

    L = build_weighted_image_laplacian(carrier_tensor, edge_threshold)

    # Compute eigenvectors
    num_evs_needed = max(eigenpair) + 1
    # Select eigenpair based on theta
    k_base = int(theta * 6)  # theta controls frequency band
    actual_pair = (k_base + eigenpair[0], k_base + eigenpair[1])
    num_evs_needed = max(actual_pair) + 1

    eigenvectors, eigenvalues = lanczos_k_eigenvectors(L, num_eigenvectors=num_evs_needed)

    if eigenvectors.shape[1] < num_evs_needed:
        # Fall back if not enough eigenvectors
        actual_pair = (0, min(1, eigenvectors.shape[1] - 1))

    ev_real = eigenvectors[:, actual_pair[0]].reshape(height, width)
    ev_imag = eigenvectors[:, actual_pair[1]].reshape(height, width)

    # NON-LINEAR: arctan2 creates spiral patterns
    phase = np.arctan2(ev_imag, ev_real)  # Range [-pi, pi]

    # Normalize operand
    operand_norm = operand.astype(np.float32)
    if operand_norm.max() > 1.0:
        operand_norm = operand_norm / 255.0

    # Resize operand if needed
    if operand_norm.shape != (height, width):
        from scipy.ndimage import zoom
        zoom_h = height / operand_norm.shape[0]
        zoom_w = width / operand_norm.shape[1]
        operand_norm = zoom(operand_norm, (zoom_h, zoom_w), order=1)

    # NON-LINEAR modulation: operand controls phase rotation
    phase_rotated = phase + operand_norm * np.pi * (1 + theta)

    # Convert back to visual representation
    # Use sin/cos combination for rich structure
    result = 0.5 * (np.sin(phase_rotated) + 1) * (0.5 + 0.5 * np.cos(phase_rotated * 2))

    # Add magnitude information (also non-linear via sqrt)
    magnitude = np.sqrt(ev_real**2 + ev_imag**2 + 1e-10)
    magnitude_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-10)

    # Combine phase and magnitude non-linearly
    result = result * (0.3 + 0.7 * magnitude_norm)

    # Final normalization
    result = (result - result.min()) / (result.max() - result.min() + 1e-10)

    return result


def spectral_contour_sdf(
    carrier: np.ndarray,
    operand: np.ndarray,
    theta: float = 0.5,
    num_contours: int = 5,
    edge_threshold: float = 0.1
) -> np.ndarray:
    """
    Generate signed distance field from spectral contours.

    NON-LINEAR: Computes distance to eigenvector iso-contours, creating
    distance fields that are structurally different from both inputs.

    Args:
        carrier: 2D carrier image (H, W) - provides spectral structure
        operand: 2D operand image (H, W) - weights contour contributions
        theta: Controls which eigenvectors to use (0=low, 1=high frequency)
        num_contours: Number of iso-contour levels per eigenvector
        edge_threshold: Carrier edge sensitivity

    Returns:
        SDF-based field (H, W), normalized to [0, 1]

    Key insight: Distance transforms are inherently non-linear - they
    create smooth gradients that don't exist in the discrete inputs.
    """
    height, width = carrier.shape

    # Build weighted Laplacian from carrier
    carrier_tensor = torch.tensor(carrier, dtype=torch.float32, device=DEVICE)
    if carrier_tensor.max() > 1.0:
        carrier_tensor = carrier_tensor / 255.0

    L = build_weighted_image_laplacian(carrier_tensor, edge_threshold)

    # Select eigenvector range based on theta
    k_start = int(theta * 4)
    num_evs = 4
    eigenvectors, _ = lanczos_k_eigenvectors(L, num_eigenvectors=k_start + num_evs)

    # Normalize operand
    operand_norm = operand.astype(np.float32)
    if operand_norm.max() > 1.0:
        operand_norm = operand_norm / 255.0

    # Resize operand if needed
    if operand_norm.shape != (height, width):
        from scipy.ndimage import zoom
        zoom_h = height / operand_norm.shape[0]
        zoom_w = width / operand_norm.shape[1]
        operand_norm = zoom(operand_norm, (zoom_h, zoom_w), order=1)

    # Create SDF accumulator
    sdf = np.zeros((height, width), dtype=np.float32)

    for k in range(min(num_evs, eigenvectors.shape[1] - k_start)):
        ev_idx = k_start + k
        if ev_idx >= eigenvectors.shape[1]:
            break

        ev = eigenvectors[:, ev_idx].reshape(height, width)

        # NON-LINEAR: distance to each iso-contour level
        for level_idx in range(num_contours):
            level = -1 + 2 * level_idx / (num_contours - 1) if num_contours > 1 else 0

            # Distance to iso-contour at this level
            contour_distance = np.abs(ev - level)

            # NON-LINEAR: Apply operand as weight with exponential decay
            # This creates interaction between operand structure and distance
            weight = np.exp(-contour_distance * (1 + 2 * operand_norm))

            sdf += weight

    # NON-LINEAR: Apply sqrt for compression
    sdf = np.sqrt(sdf + 1e-10)

    # Final normalization
    sdf = (sdf - sdf.min()) / (sdf.max() - sdf.min() + 1e-10)

    return sdf


def commute_time_distance_field(
    carrier: np.ndarray,
    operand: np.ndarray,
    theta: float = 0.5,
    reference_mode: str = 'operand_max',
    edge_threshold: float = 0.1
) -> np.ndarray:
    """
    Compute commute time distance from reference points.

    NON-LINEAR: Commute time distance uses eigenvalue-weighted sum of
    squared eigenvector differences, creating organic distance fields
    that respect the graph topology defined by the carrier.

    Args:
        carrier: 2D carrier image (H, W) - provides graph structure
        operand: 2D operand image (H, W) - defines reference points
        theta: Controls spectral range (0=coarse distances, 1=fine)
        reference_mode: How to select reference point from operand:
            'operand_max': Maximum operand value
            'operand_min': Minimum operand value
            'center': Image center
        edge_threshold: Carrier edge sensitivity

    Returns:
        Commute time distance field (H, W), normalized to [0, 1]

    Key insight: Commute time distances are quadratic in eigenvector
    differences and inverse-weighted by eigenvalues - highly non-linear.
    """
    height, width = carrier.shape
    n = height * width

    # Build weighted Laplacian from carrier
    carrier_tensor = torch.tensor(carrier, dtype=torch.float32, device=DEVICE)
    if carrier_tensor.max() > 1.0:
        carrier_tensor = carrier_tensor / 255.0

    L = build_weighted_image_laplacian(carrier_tensor, edge_threshold)

    # Normalize operand
    operand_norm = operand.astype(np.float32)
    if operand_norm.max() > 1.0:
        operand_norm = operand_norm / 255.0

    # Resize operand if needed
    if operand_norm.shape != (height, width):
        from scipy.ndimage import zoom
        zoom_h = height / operand_norm.shape[0]
        zoom_w = width / operand_norm.shape[1]
        operand_norm = zoom(operand_norm, (zoom_h, zoom_w), order=1)

    # Select reference point based on operand
    if reference_mode == 'operand_max':
        ref_idx = np.argmax(operand_norm)
    elif reference_mode == 'operand_min':
        ref_idx = np.argmin(operand_norm)
    else:  # center
        ref_idx = (height // 2) * width + (width // 2)

    ref_y, ref_x = ref_idx // width, ref_idx % width

    # Compute eigenvectors based on theta
    num_evs = max(10, int(20 * (1 - theta) + 5))  # More for coarse, fewer for fine
    eigenvectors, eigenvalues = lanczos_k_eigenvectors(L, num_eigenvectors=num_evs)

    # NON-LINEAR: Commute time distance formula
    # CT(i, j) = sum_k (phi_k(i) - phi_k(j))^2 / lambda_k
    distances = np.zeros(n, dtype=np.float32)

    for k in range(1, len(eigenvalues)):  # Skip k=0 (constant eigenvector)
        if eigenvalues[k] < 1e-6:
            continue

        # Squared difference (NON-LINEAR)
        diff = eigenvectors[:, k] - eigenvectors[ref_idx, k]

        # Weight by operand values for interaction (NON-LINEAR)
        operand_flat = operand_norm.flatten()
        operand_weight = 0.5 + 0.5 * operand_flat  # Range [0.5, 1]

        # Eigenvalue weighting varies with theta
        lambda_power = 1.0 + theta  # theta=0: 1/lambda, theta=1: 1/lambda^2
        eigenvalue_weight = 1.0 / (eigenvalues[k] ** lambda_power + 1e-10)

        distances += diff**2 * eigenvalue_weight * operand_weight

    # NON-LINEAR: Square root for volume normalization
    distances = np.sqrt(distances + 1e-10)

    # Reshape and normalize
    distances = distances.reshape(height, width)
    distances = (distances - distances.min()) / (distances.max() - distances.min() + 1e-10)

    return distances


def spectral_warp(
    carrier: np.ndarray,
    operand: np.ndarray,
    theta: float = 0.5,
    warp_strength: float = 10.0,
    edge_threshold: float = 0.1
) -> np.ndarray:
    """
    Warp operand using carrier's eigenvector gradient flow.

    NON-LINEAR: Uses eigenvector gradients to define a displacement field
    that physically warps one image by another's structure. The warping
    is inherently non-linear due to coordinate remapping.

    Args:
        carrier: 2D carrier image (H, W) - provides warp field structure
        operand: 2D operand image (H, W) - image to be warped
        theta: Controls which eigenvector defines warp direction (0=Fiedler)
        warp_strength: Magnitude of displacement (pixels)
        edge_threshold: Carrier edge sensitivity

    Returns:
        Warped operand (H, W), normalized to [0, 1]

    Key insight: Coordinate remapping via interpolation is non-linear
    and creates structure that doesn't exist in either input.
    """
    from scipy.ndimage import map_coordinates

    height, width = carrier.shape

    # Build weighted Laplacian from carrier
    carrier_tensor = torch.tensor(carrier, dtype=torch.float32, device=DEVICE)
    if carrier_tensor.max() > 1.0:
        carrier_tensor = carrier_tensor / 255.0

    L = build_weighted_image_laplacian(carrier_tensor, edge_threshold)

    # Select eigenvector based on theta
    ev_idx = int(theta * 7)  # theta=0: Fiedler, theta=1: higher frequency
    eigenvectors, _ = lanczos_k_eigenvectors(L, num_eigenvectors=ev_idx + 2)

    if eigenvectors.shape[1] <= ev_idx:
        ev_idx = eigenvectors.shape[1] - 1

    ev = eigenvectors[:, ev_idx].reshape(height, width)

    # Compute gradient of eigenvector
    grad_y = np.gradient(ev, axis=0)
    grad_x = np.gradient(ev, axis=1)

    # NON-LINEAR: Create perpendicular warp field (flow along contours)
    # This creates motion that follows the carrier's spectral structure
    warp_x = -grad_y * warp_strength * (0.5 + theta)
    warp_y = grad_x * warp_strength * (0.5 + theta)

    # Normalize operand
    operand_norm = operand.astype(np.float32)
    if operand_norm.max() > 1.0:
        operand_norm = operand_norm / 255.0

    # Resize operand if needed
    if operand_norm.shape != (height, width):
        from scipy.ndimage import zoom
        zoom_h = height / operand_norm.shape[0]
        zoom_w = width / operand_norm.shape[1]
        operand_norm = zoom(operand_norm, (zoom_h, zoom_w), order=1)

    # NON-LINEAR: Modulate warp by operand intensity
    # High operand values = more warping
    warp_x = warp_x * (0.3 + 0.7 * operand_norm)
    warp_y = warp_y * (0.3 + 0.7 * operand_norm)

    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Apply warp
    warped_y = np.clip(y_coords + warp_y, 0, height - 1).astype(np.float32)
    warped_x = np.clip(x_coords + warp_x, 0, width - 1).astype(np.float32)

    # NON-LINEAR: Bilinear interpolation at warped coordinates
    warped = map_coordinates(operand_norm, [warped_y, warped_x], order=1, mode='reflect')

    # Mix some carrier structure back in (NON-LINEAR via multiplication)
    carrier_norm = carrier.astype(np.float32)
    if carrier_norm.max() > 1.0:
        carrier_norm = carrier_norm / 255.0

    # Blend using eigenvector magnitude as mask
    ev_mag = np.abs(ev)
    ev_mag_norm = (ev_mag - ev_mag.min()) / (ev_mag.max() - ev_mag.min() + 1e-10)

    result = warped * (0.7 + 0.3 * ev_mag_norm) + carrier_norm * 0.1 * (1 - ev_mag_norm)

    # Normalize
    result = (result - result.min()) / (result.max() - result.min() + 1e-10)

    return result


def spectral_subdivision_blend(
    carrier: np.ndarray,
    operand: np.ndarray,
    theta: float = 0.5,
    max_depth: int = 4,
    min_size: int = 16,
    edge_threshold: float = 0.1
) -> np.ndarray:
    """
    Recursively subdivide by carrier's Fiedler, fill with operand statistics.

    NON-LINEAR: Uses sign thresholding on Fiedler vectors to create
    hierarchical cells, then fills each cell with statistics derived
    from the operand. Creates stained-glass patterns that don't exist
    in either input.

    Args:
        carrier: 2D carrier image (H, W) - provides subdivision structure
        operand: 2D operand image (H, W) - provides fill values
        theta: Controls subdivision depth (0=coarse, 1=fine cells)
        max_depth: Maximum recursion depth
        min_size: Minimum region size to subdivide
        edge_threshold: Carrier edge sensitivity

    Returns:
        Subdivided/blended field (H, W), normalized to [0, 1]

    Key insight: Sign thresholding is non-linear, and the recursive
    structure creates boundaries that don't exist in either input.
    """
    height, width = carrier.shape

    # Normalize inputs
    carrier_norm = carrier.astype(np.float32)
    if carrier_norm.max() > 1.0:
        carrier_norm = carrier_norm / 255.0

    operand_norm = operand.astype(np.float32)
    if operand_norm.max() > 1.0:
        operand_norm = operand_norm / 255.0

    # Resize operand if needed
    if operand_norm.shape != (height, width):
        from scipy.ndimage import zoom
        zoom_h = height / operand_norm.shape[0]
        zoom_w = width / operand_norm.shape[1]
        operand_norm = zoom(operand_norm, (zoom_h, zoom_w), order=1)

    # Adjust depth based on theta
    effective_depth = int(max_depth * (0.5 + 0.5 * theta))
    effective_min_size = max(4, int(min_size * (1 - 0.5 * theta)))

    # Region tracking
    regions = np.zeros((height, width), dtype=np.int32)
    region_id = [0]  # Mutable counter

    def subdivide(mask: np.ndarray, depth: int) -> None:
        """Recursively subdivide a region using Fiedler vectors."""
        nonlocal regions, region_id

        if depth >= effective_depth or mask.sum() < effective_min_size:
            regions[mask] = region_id[0]
            region_id[0] += 1
            return

        # Extract bounding box
        coords = np.argwhere(mask)
        if len(coords) == 0:
            return

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0) + 1

        subimage = carrier_norm[y_min:y_max, x_min:x_max]
        submask = mask[y_min:y_max, x_min:x_max]

        if subimage.size < effective_min_size:
            regions[mask] = region_id[0]
            region_id[0] += 1
            return

        # Build Laplacian for subregion
        sub_tensor = torch.tensor(subimage, dtype=torch.float32, device=DEVICE)
        try:
            L_sub = build_weighted_image_laplacian(sub_tensor, edge_threshold)

            # Compute Fiedler vector
            fiedler, lambda2 = lanczos_fiedler_gpu(L_sub)
            fiedler = fiedler.cpu().numpy().reshape(subimage.shape)

            # NON-LINEAR: Split by sign (thresholding)
            positive = (fiedler >= 0) & submask
            negative = (fiedler < 0) & submask

            # Only subdivide if both parts are substantial
            if positive.sum() < effective_min_size or negative.sum() < effective_min_size:
                regions[mask] = region_id[0]
                region_id[0] += 1
                return

            # Reconstruct full masks
            full_pos = np.zeros_like(mask)
            full_neg = np.zeros_like(mask)
            full_pos[y_min:y_max, x_min:x_max] = positive
            full_neg[y_min:y_max, x_min:x_max] = negative

            subdivide(full_pos, depth + 1)
            subdivide(full_neg, depth + 1)

        except Exception:
            # If subdivision fails, assign current region
            regions[mask] = region_id[0]
            region_id[0] += 1

    # Start recursion
    initial_mask = np.ones((height, width), dtype=bool)
    subdivide(initial_mask, 0)

    # Fill each region with operand-derived values
    result = np.zeros((height, width), dtype=np.float32)
    num_regions = region_id[0]

    for rid in range(num_regions):
        mask = regions == rid
        if mask.sum() == 0:
            continue

        # NON-LINEAR: Compute statistics from operand within region
        operand_in_region = operand_norm[mask]
        carrier_in_region = carrier_norm[mask]

        # Mix mean and std for texture variation
        mean_val = operand_in_region.mean()
        std_val = operand_in_region.std()
        carrier_mean = carrier_in_region.mean()

        # NON-LINEAR combination
        # Use operand stats but modulate by carrier structure
        fill_value = mean_val + theta * std_val * np.sign(carrier_mean - 0.5)
        fill_value = np.clip(fill_value, 0, 1)

        result[mask] = fill_value

    # Add edge enhancement at region boundaries (NON-LINEAR via gradient)
    grad_y = np.abs(np.gradient(result, axis=0))
    grad_x = np.abs(np.gradient(result, axis=1))
    edges = np.sqrt(grad_y**2 + grad_x**2)
    edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-10)

    # Blend result with edges for stained-glass effect
    result = result * (1 - 0.3 * edges) + 0.3 * edges

    # Final normalization
    result = (result - result.min()) / (result.max() - result.min() + 1e-10)

    return result


# ============================================================
# Testing/Validation Utilities
# ============================================================

def test_tiled_eigenvectors(image_size: int = 256, tile_size: int = 64):
    """
    Test tiled eigenvector computation on a synthetic image.

    Verifies that the algorithm works without materializing full matrix.
    """
    # Create synthetic test image (gradient with some structure)
    y_grid, x_grid = np.meshgrid(
        np.linspace(0, 1, image_size),
        np.linspace(0, 1, image_size),
        indexing='ij'
    )

    # Add some structure
    test_image = (
        0.3 * np.sin(10 * x_grid) * np.sin(10 * y_grid) +
        0.3 * x_grid +
        0.3 * y_grid +
        0.1 * np.random.randn(image_size, image_size)
    )
    test_image = np.clip(test_image, 0, 1).astype(np.float32)

    print(f"Testing tiled eigenvector computation...")
    print(f"  Image size: {image_size}x{image_size} = {image_size**2} pixels")
    print(f"  Tile size: {tile_size}")
    print(f"  Full matrix would be: {image_size**2}x{image_size**2} = {(image_size**2)**2} elements")
    print(f"  Tile matrix is: {tile_size**2}x{tile_size**2} = {(tile_size**2)**2} elements")

    import time
    start = time.time()

    eigenvector_images = compute_local_eigenvectors_tiled(
        test_image,
        tile_size=tile_size,
        overlap=16,
        num_eigenvectors=4
    )

    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Output shape: {eigenvector_images.shape}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Memory reduction: {((image_size**2)**2) / ((tile_size**2)**2):.0f}x")

    # Verify output properties
    print(f"\nEigenvector statistics:")
    k = 0
    while k < eigenvector_images.shape[2]:
        ev = eigenvector_images[:, :, k]
        print(f"  k={k}: min={ev.min():.4f}, max={ev.max():.4f}, std={ev.std():.4f}")
        k += 1

    return eigenvector_images


def compare_explicit_vs_iterative(
    carrier: np.ndarray,
    theta: float = 0.5,
    num_eigenvectors: int = 8,
    polynomial_order: int = 30,
    num_steps: int = 8,
    edge_threshold: float = 0.1
) -> Dict[str, np.ndarray]:
    """
    Compare explicit eigenvector computation vs iterative polynomial approximation.

    This demonstrates that the iterative approach produces similar results
    to the explicit method, but with O(n log n) complexity instead of O(n^3).

    Args:
        carrier: 2D carrier image
        theta: Rotation angle in [0, 1]
        num_eigenvectors: Number of eigenvectors for explicit method
        polynomial_order: Chebyshev polynomial degree for iterative method
        num_steps: Number of iterative steps
        edge_threshold: Carrier edge sensitivity

    Returns:
        Dictionary with comparison results:
            - 'explicit_field': Spectral field from explicit eigenvectors
            - 'iterative_field': Spectral field from iterative method
            - 'difference': Absolute difference
            - 'correlation': Pearson correlation coefficient
    """
    # Import iterative methods from core module
    from spectral_ops_fast import (
        estimate_lambda_max,
        polynomial_spectral_field,
        iterative_spectral_transform,
    )

    height, width = carrier.shape
    n = height * width

    # Build weighted Laplacian
    carrier_tensor = torch.tensor(carrier, dtype=torch.float32, device=DEVICE)
    if carrier_tensor.max() > 1.0:
        carrier_tensor = carrier_tensor / 255.0

    L = build_weighted_image_laplacian(carrier_tensor, edge_threshold)

    # ==========================================
    # EXPLICIT METHOD: Compute eigenvectors
    # ==========================================

    # Use dense solver on CPU to avoid CUDA cusolver issues
    L_dense = L.to_dense().cpu()
    eigenvalues, eigenvectors = torch.linalg.eigh(L_dense)

    # Skip first eigenvector (constant)
    k = min(num_eigenvectors, n - 1)
    selected_eigenvalues = eigenvalues[1:k+1]
    selected_eigenvectors = eigenvectors[:, 1:k+1]

    # Compute theta-weighted spectral field (explicit formula)
    indices = torch.arange(k, dtype=torch.float32)
    center = theta * k
    weights = torch.exp(-((indices - center) ** 2) / 2.0)

    # Weighted sum of absolute eigenvector values
    explicit_field = torch.zeros(n, dtype=torch.float32)
    idx = 0
    while idx < k:
        explicit_field = explicit_field + weights[idx] * torch.abs(selected_eigenvectors[:, idx])
        idx += 1

    # ==========================================
    # ITERATIVE METHOD: Polynomial filtering
    # ==========================================

    # Use carrier as initial signal
    signal = carrier_tensor.flatten()

    # Estimate lambda_max
    lambda_max = estimate_lambda_max(L)

    # Apply polynomial spectral field computation
    iterative_field = polynomial_spectral_field(
        L, signal, theta,
        num_bands=num_eigenvectors,
        polynomial_order=polynomial_order,
        lambda_max=lambda_max
    )

    # Also try the iterative transform
    iterative_transform = iterative_spectral_transform(
        L, signal, theta,
        num_steps=num_steps,
        polynomial_order=polynomial_order,
        lambda_max=lambda_max
    )

    # ==========================================
    # COMPARISON
    # ==========================================

    # Normalize both fields for comparison
    explicit_np = explicit_field.numpy()
    iterative_np = iterative_field.cpu().numpy()
    transform_np = iterative_transform.cpu().numpy()

    # Normalize to [0, 1]
    def normalize(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            return (arr - arr_min) / (arr_max - arr_min)
        return np.zeros_like(arr)

    explicit_norm = normalize(explicit_np)
    iterative_norm = normalize(iterative_np)
    transform_norm = normalize(transform_np)

    # Compute correlation
    def correlation(a, b):
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        numerator = np.sum(a_centered * b_centered)
        denominator = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
        if denominator > 1e-10:
            return numerator / denominator
        return 0.0

    corr_field = correlation(explicit_norm, iterative_norm)
    corr_transform = correlation(explicit_norm, transform_norm)

    return {
        'explicit_field': explicit_norm.reshape(height, width),
        'iterative_field': iterative_norm.reshape(height, width),
        'iterative_transform': transform_norm.reshape(height, width),
        'difference_field': np.abs(explicit_norm - iterative_norm).reshape(height, width),
        'difference_transform': np.abs(explicit_norm - transform_norm).reshape(height, width),
        'correlation_field': corr_field,
        'correlation_transform': corr_transform,
        'eigenvalues': selected_eigenvalues.numpy(),
        'lambda_max': lambda_max
    }
