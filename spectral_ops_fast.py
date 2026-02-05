"""
Fast spectral operations using torch.sparse on GPU.

All numerical hot paths are vectorized tensor operations.
Graph traversal uses while loops (explicitly allowed, not hot path).
GPU-first with CPU fallback.
"""

from typing import Dict, List, Set, Tuple, Optional, Protocol
import numpy as np
import torch

# Device selection: GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphView(Protocol):
    """Graph you can query but not enumerate."""
    def neighbors(self, node: int) -> List[int]: ...
    def degree(self, node: int) -> int: ...
    def seed_nodes(self) -> List[int]: ...


from dataclasses import dataclass


@dataclass
class Graph:
    """
    Canonical graph for all spectral operations.

    This is THE graph type for itten. All other graph types should convert to/from this.

    Attributes:
        adjacency: Sparse (n x n) adjacency/weight matrix as torch.sparse_coo_tensor
        coords: Optional (n, d) coordinate tensor for visualization/pathfinding

    The adjacency matrix stores edge weights. For unweighted graphs, use 1.0.
    Self-loops are NOT included in adjacency (diagonal is 0).
    """
    adjacency: torch.Tensor  # Sparse (n x n)
    coords: Optional[torch.Tensor] = None  # (n, d)

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self.adjacency.shape[0]

    @property
    def device(self) -> torch.device:
        """Device where graph tensors reside."""
        return self.adjacency.device

    def neighbors(self, node: int) -> List[int]:
        """Get neighbors of a node."""
        # Extract from sparse adjacency
        adj = self.adjacency.coalesce()
        indices = adj.indices()

        # Find edges starting from this node
        mask = indices[0] == node
        neighbor_indices = indices[1][mask]

        return neighbor_indices.cpu().tolist()

    def degree(self, node: int) -> int:
        """Get degree of a node."""
        return len(self.neighbors(node))

    def seed_nodes(self) -> List[int]:
        """Return some seed nodes for exploration."""
        n = self.num_nodes
        if n == 0:
            return []
        # Return first few nodes
        return list(range(min(5, n)))

    def coord_of(self, node: int) -> Optional[Tuple[float, ...]]:
        """Get coordinates of a node, if available."""
        if self.coords is None:
            return None
        if node < 0 or node >= self.num_nodes:
            return None
        coord = self.coords[node].cpu()
        return tuple(coord.tolist())

    def edge_weight(self, u: int, v: int) -> float:
        """Get weight of edge (u, v). Returns 0 if no edge."""
        adj = self.adjacency.coalesce()
        indices = adj.indices()
        values = adj.values()

        # Find edge (u, v)
        mask = (indices[0] == u) & (indices[1] == v)
        if mask.any():
            return values[mask][0].item()
        return 0.0

    def laplacian(self, normalized: bool = False) -> torch.Tensor:
        """
        Compute graph Laplacian.

        Args:
            normalized: If True, compute symmetric normalized Laplacian D^{-1/2} L D^{-1/2}

        Returns:
            Sparse Laplacian matrix (n x n)
        """
        n = self.num_nodes
        device = self.device

        adj = self.adjacency.coalesce()
        indices = adj.indices()
        values = adj.values()

        # Compute degrees via scatter_add
        degrees = torch.zeros(n, device=device, dtype=values.dtype)
        degrees.scatter_add_(0, indices[0], values)

        if normalized:
            # Symmetric normalized: I - D^{-1/2} A D^{-1/2}
            # Off-diagonal: -A[i,j] / sqrt(d_i * d_j)
            # Diagonal: 1
            d_inv_sqrt = torch.zeros(n, device=device, dtype=values.dtype)
            nonzero_mask = degrees > 1e-10
            d_inv_sqrt[nonzero_mask] = 1.0 / torch.sqrt(degrees[nonzero_mask])

            # Scale edge weights
            scaled_values = -values * d_inv_sqrt[indices[0]] * d_inv_sqrt[indices[1]]

            # Add diagonal (all 1s for nodes with edges)
            diag_indices = torch.arange(n, device=device)
            all_indices = torch.cat([indices, torch.stack([diag_indices, diag_indices])], dim=1)
            all_values = torch.cat([scaled_values, torch.ones(n, device=device, dtype=values.dtype)])

            L = torch.sparse_coo_tensor(all_indices, all_values, (n, n)).coalesce()
        else:
            # Unnormalized: L = D - A
            # Off-diagonal: -A[i,j]
            # Diagonal: degree[i]
            off_diag_values = -values

            diag_indices = torch.arange(n, device=device)
            all_indices = torch.cat([indices, torch.stack([diag_indices, diag_indices])], dim=1)
            all_values = torch.cat([off_diag_values, degrees])

            L = torch.sparse_coo_tensor(all_indices, all_values, (n, n)).coalesce()

        return L

    @classmethod
    def from_image(
        cls,
        image,
        connectivity: int = 4,
        edge_threshold: float = 0.1,
        device: Optional[torch.device] = None
    ) -> 'Graph':
        """
        Create graph from 2D image grid.

        Each pixel becomes a node. Edges connect adjacent pixels with weights
        based on intensity similarity: w = exp(-|diff| / threshold).

        Args:
            image: 2D array (H, W) - numpy array or torch tensor
            connectivity: 4 or 8 (4-connected or 8-connected grid)
            edge_threshold: Controls edge weight decay based on intensity difference
            device: Target device (defaults to DEVICE)

        Returns:
            Graph with image grid connectivity
        """
        if device is None:
            device = DEVICE

        # Convert to tensor
        if isinstance(image, np.ndarray):
            img = torch.tensor(image, dtype=torch.float32, device=device)
        else:
            img = image.to(device=device, dtype=torch.float32)

        # Normalize to [0, 1]
        if img.max() > 1.0:
            img = img / 255.0

        height, width = img.shape
        n = height * width

        # Build edge indices and weights
        rows_list = []
        cols_list = []
        weights_list = []

        # Horizontal edges: (y, x) -- (y, x+1)
        y_h = torch.arange(height, device=device)
        x_h = torch.arange(width - 1, device=device)
        yy_h, xx_h = torch.meshgrid(y_h, x_h, indexing='ij')

        idx_left = yy_h.flatten() * width + xx_h.flatten()
        idx_right = idx_left + 1

        diff_h = torch.abs(img[:, 1:] - img[:, :-1]).flatten()
        weights_h = torch.exp(-diff_h / edge_threshold)

        rows_list.extend([idx_left, idx_right])
        cols_list.extend([idx_right, idx_left])
        weights_list.extend([weights_h, weights_h])

        # Vertical edges: (y, x) -- (y+1, x)
        y_v = torch.arange(height - 1, device=device)
        x_v = torch.arange(width, device=device)
        yy_v, xx_v = torch.meshgrid(y_v, x_v, indexing='ij')

        idx_top = yy_v.flatten() * width + xx_v.flatten()
        idx_bottom = idx_top + width

        diff_v = torch.abs(img[1:, :] - img[:-1, :]).flatten()
        weights_v = torch.exp(-diff_v / edge_threshold)

        rows_list.extend([idx_top, idx_bottom])
        cols_list.extend([idx_bottom, idx_top])
        weights_list.extend([weights_v, weights_v])

        # 8-connectivity: add diagonal edges
        if connectivity == 8:
            # Diagonal 1: (y, x) -- (y+1, x+1)
            y_d = torch.arange(height - 1, device=device)
            x_d = torch.arange(width - 1, device=device)
            yy_d, xx_d = torch.meshgrid(y_d, x_d, indexing='ij')

            idx_tl = yy_d.flatten() * width + xx_d.flatten()
            idx_br = idx_tl + width + 1

            diff_d1 = torch.abs(img[1:, 1:] - img[:-1, :-1]).flatten()
            weights_d1 = torch.exp(-diff_d1 / edge_threshold) * 0.707  # Scale by 1/sqrt(2)

            rows_list.extend([idx_tl, idx_br])
            cols_list.extend([idx_br, idx_tl])
            weights_list.extend([weights_d1, weights_d1])

            # Diagonal 2: (y, x+1) -- (y+1, x)
            idx_tr = yy_d.flatten() * width + xx_d.flatten() + 1
            idx_bl = idx_tr + width - 1

            diff_d2 = torch.abs(img[1:, :-1] - img[:-1, 1:]).flatten()
            weights_d2 = torch.exp(-diff_d2 / edge_threshold) * 0.707

            rows_list.extend([idx_tr, idx_bl])
            cols_list.extend([idx_bl, idx_tr])
            weights_list.extend([weights_d2, weights_d2])

        # Concatenate all edges
        all_rows = torch.cat(rows_list)
        all_cols = torch.cat(cols_list)
        all_weights = torch.cat(weights_list)

        # Build sparse adjacency
        indices = torch.stack([all_rows.long(), all_cols.long()])
        adjacency = torch.sparse_coo_tensor(indices, all_weights, (n, n)).coalesce()

        # Build coordinate tensor: (node_id) -> (y, x) normalized to [0, 1]
        node_y = torch.arange(height, device=device).float().unsqueeze(1).expand(height, width).flatten()
        node_x = torch.arange(width, device=device).float().unsqueeze(0).expand(height, width).flatten()
        coords = torch.stack([node_y / max(1, height - 1), node_x / max(1, width - 1)], dim=1)

        return cls(adjacency=adjacency, coords=coords)

    @classmethod
    def from_lattice(cls, lattice, device: Optional[torch.device] = None) -> 'Graph':
        """
        Create graph from a lattice structure (TerritoryGraph or similar).

        The lattice must have:
        - all_node_ids() or all_nodes(): returns iterable of node IDs
        - neighbors(node_id): returns list of neighbor IDs
        - Optionally node_coord(node_id): returns (x, y) tuple

        Args:
            lattice: Lattice structure with graph interface
            device: Target device (defaults to DEVICE)

        Returns:
            Graph with lattice connectivity
        """
        if device is None:
            device = DEVICE

        # Get all nodes
        if hasattr(lattice, 'all_node_ids'):
            node_ids = list(lattice.all_node_ids())
        elif hasattr(lattice, 'all_nodes'):
            # TerritoryGraph uses coords, need to convert via node_id
            all_coords = lattice.all_nodes()
            node_ids = [lattice.node_id(c) for c in all_coords if lattice.node_id(c) is not None]
        else:
            raise ValueError("Lattice must have all_node_ids() or all_nodes() method")

        n = len(node_ids)
        if n == 0:
            return cls(
                adjacency=torch.sparse_coo_tensor(
                    torch.zeros((2, 0), dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.float32, device=device),
                    (0, 0)
                ),
                coords=None
            )

        # Create mapping from original IDs to contiguous 0..n-1
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        # Build edges
        rows = []
        cols = []

        idx = 0
        while idx < n:
            node_id = node_ids[idx]
            neighbors = lattice.neighbors(node_id)
            j = 0
            while j < len(neighbors):
                neighbor_id = neighbors[j]
                if neighbor_id in id_to_idx:
                    rows.append(idx)
                    cols.append(id_to_idx[neighbor_id])
                j += 1
            idx += 1

        # Build sparse adjacency (unweighted = 1.0)
        if rows:
            indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
            values = torch.ones(len(rows), dtype=torch.float32, device=device)
            adjacency = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
        else:
            adjacency = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.float32, device=device),
                (n, n)
            )

        # Build coordinates if available
        coords = None
        if hasattr(lattice, 'node_coord'):
            coord_list = []
            idx = 0
            while idx < n:
                coord = lattice.node_coord(node_ids[idx])
                if coord is not None:
                    coord_list.append([float(coord[0]), float(coord[1])])
                else:
                    coord_list.append([0.0, 0.0])
                idx += 1
            coords = torch.tensor(coord_list, dtype=torch.float32, device=device)

        return cls(adjacency=adjacency, coords=coords)

    @classmethod
    def from_graphview(
        cls,
        graph: GraphView,
        seed_nodes: Optional[List[int]] = None,
        max_hops: int = 10,
        device: Optional[torch.device] = None
    ) -> 'Graph':
        """
        Create Graph from a GraphView by expanding from seed nodes.

        Since GraphView can represent infinite graphs, we expand from seeds
        up to max_hops to get a finite subgraph.

        Args:
            graph: GraphView supporting neighbors() queries
            seed_nodes: Starting nodes (defaults to graph.seed_nodes())
            max_hops: Maximum expansion distance
            device: Target device (defaults to DEVICE)

        Returns:
            Graph containing nodes within max_hops of seeds
        """
        if device is None:
            device = DEVICE

        if seed_nodes is None:
            seed_nodes = graph.seed_nodes()

        # Expand neighborhood
        active = set(seed_nodes)
        hop = 0
        while hop < max_hops:
            new_nodes = set()
            active_list = list(active)
            i = 0
            while i < len(active_list):
                neighbors = graph.neighbors(active_list[i])
                j = 0
                while j < len(neighbors):
                    new_nodes.add(neighbors[j])
                    j += 1
                i += 1
            if not new_nodes - active:
                break
            active.update(new_nodes)
            hop += 1

        node_ids = sorted(active)
        n = len(node_ids)

        if n == 0:
            return cls(
                adjacency=torch.sparse_coo_tensor(
                    torch.zeros((2, 0), dtype=torch.long, device=device),
                    torch.zeros(0, dtype=torch.float32, device=device),
                    (0, 0)
                ),
                coords=None
            )

        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        # Build edges
        rows = []
        cols = []

        idx = 0
        while idx < n:
            node_id = node_ids[idx]
            neighbors = graph.neighbors(node_id)
            j = 0
            while j < len(neighbors):
                neighbor_id = neighbors[j]
                if neighbor_id in id_to_idx:
                    rows.append(idx)
                    cols.append(id_to_idx[neighbor_id])
                j += 1
            idx += 1

        if rows:
            indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
            values = torch.ones(len(rows), dtype=torch.float32, device=device)
            adjacency = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
        else:
            adjacency = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.float32, device=device),
                (n, n)
            )

        # Try to get coordinates if graph has coord_of method
        coords = None
        if hasattr(graph, 'coord_of'):
            coord_list = []
            has_coords = True
            idx = 0
            while idx < n:
                coord = graph.coord_of(node_ids[idx])
                if coord is not None:
                    coord_list.append(list(coord))
                else:
                    has_coords = False
                    break
                idx += 1
            if has_coords and coord_list:
                coords = torch.tensor(coord_list, dtype=torch.float32, device=device)

        return cls(adjacency=adjacency, coords=coords)


def build_sparse_laplacian(
    graph: GraphView,
    active_nodes: Set[int]
) -> Tuple[torch.Tensor, List[int], Dict[int, int]]:
    """
    Build sparse Laplacian as torch.sparse_coo_tensor.

    Returns: (L_sparse, node_list, node_to_idx)

    Graph traversal phase collects COO entries.
    Tensor construction is a single vectorized call.
    """
    node_list = list(active_nodes)
    n = len(node_list)

    # Build index mapping
    node_to_idx: Dict[int, int] = {}
    idx = 0
    while idx < n:
        node_to_idx[node_list[idx]] = idx
        idx += 1

    # Collect edges - GRAPH TRAVERSAL, not hot path
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    degrees: List[int] = []

    i = 0
    while i < n:
        node = node_list[i]
        neighbors = graph.neighbors(node)
        deg = 0
        j = 0
        while j < len(neighbors):
            neighbor = neighbors[j]
            if neighbor in node_to_idx:
                k = node_to_idx[neighbor]
                rows.append(i)
                cols.append(k)
                vals.append(-1.0)
                deg += 1
            j += 1
        degrees.append(deg)
        i += 1

    # Add diagonal entries
    i = 0
    while i < n:
        rows.append(i)
        cols.append(i)
        vals.append(float(degrees[i]))
        i += 1

    # Build sparse tensor - SINGLE VECTORIZED CALL
    indices = torch.tensor([rows, cols], dtype=torch.long, device=DEVICE)
    values = torch.tensor(vals, dtype=torch.float32, device=DEVICE)
    L = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

    return L, node_list, node_to_idx


def lanczos_fiedler_gpu(
    L: torch.Tensor,
    num_iterations: int = 30,
    tol: float = 1e-10
) -> Tuple[torch.Tensor, float]:
    """
    GPU Lanczos iteration - ALL TENSOR OPS, NO PYTHON LOOPS IN MATVEC.

    Returns: (fiedler_vector, lambda2)
    """
    n = L.shape[0]

    if n < 3:
        return torch.zeros(n, device=DEVICE, dtype=torch.float32), 0.0

    k = min(num_iterations, n - 1)

    # Initialize random vector, project out constant
    torch.manual_seed(42)
    v = torch.randn(n, device=DEVICE, dtype=torch.float32)
    v = v - v.mean()  # Project out constant
    v_norm = torch.linalg.norm(v)

    if v_norm < tol:
        v = torch.ones(n, device=DEVICE, dtype=torch.float32)
        v[::2] = -1.0
        v = v - v.mean()
        v_norm = torch.linalg.norm(v)

    v = v / v_norm

    # Pre-allocate Lanczos vectors as matrix columns
    V = torch.zeros((n, k + 1), device=DEVICE, dtype=torch.float32)
    V[:, 0] = v

    # Tridiagonal matrix elements
    alphas = torch.zeros(k, device=DEVICE, dtype=torch.float32)
    betas = torch.zeros(k, device=DEVICE, dtype=torch.float32)

    actual_k = k
    iteration = 0

    # Lanczos iteration - while loop for iteration count, NOT for node traversal
    while iteration < k:
        # SPARSE MATVEC: w = L @ v (single tensor op)
        w = torch.sparse.mm(L, V[:, iteration].unsqueeze(1)).squeeze(1)

        # alpha = v^T @ w (single tensor op)
        alpha = torch.dot(V[:, iteration], w)
        alphas[iteration] = alpha

        # w = w - alpha * v (tensor op)
        w = w - alpha * V[:, iteration]

        # w = w - beta * v_prev (tensor op)
        if iteration > 0:
            w = w - betas[iteration - 1] * V[:, iteration - 1]

        # Full reorthogonalization for numerical stability (tensor ops)
        # coeffs = V^T @ w, then w = w - V @ coeffs
        if iteration > 0:
            V_prev = V[:, :iteration + 1]
            coeffs = torch.mm(V_prev.T, w.unsqueeze(1)).squeeze(1)
            w = w - torch.mm(V_prev, coeffs.unsqueeze(1)).squeeze(1)

        # Project out constant (tensor op)
        w = w - w.mean()

        # beta = ||w|| (tensor op)
        beta = torch.linalg.norm(w)

        if beta < tol:
            actual_k = iteration + 1
            break

        betas[iteration] = beta
        V[:, iteration + 1] = w / beta

        iteration += 1

    if actual_k < 2:
        return torch.zeros(n, device=DEVICE, dtype=torch.float32), 0.0

    # Build tridiagonal matrix T (tensor indexing, no loops)
    T = torch.diag(alphas[:actual_k])

    if actual_k > 1:
        # Off-diagonal assignment via indexing
        off_diag = torch.arange(actual_k - 1, device=DEVICE)
        T[off_diag, off_diag + 1] = betas[:actual_k - 1]
        T[off_diag + 1, off_diag] = betas[:actual_k - 1]

    # Solve eigenproblem (single call)
    eigenvalues, eigenvectors = torch.linalg.eigh(T)

    # Find smallest non-trivial eigenvalue using tensor ops
    mask = eigenvalues > 1e-6
    if not mask.any():
        return torch.zeros(n, device=DEVICE, dtype=torch.float32), 0.0

    valid_indices = torch.where(mask)[0]
    idx = valid_indices[0].item()

    lambda2 = eigenvalues[idx].item()

    # Reconstruct Fiedler vector: fiedler = V @ y (single matmul)
    y = eigenvectors[:, idx].unsqueeze(1)
    fiedler = torch.mm(V[:, :actual_k], y).squeeze(1)

    return fiedler, lambda2


def lanczos_k_eigenvectors(
    L: torch.Tensor,
    num_eigenvectors: int,
    num_iterations: int = 50,
    tol: float = 1e-10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute first k non-trivial eigenvectors using GPU Lanczos.

    This is the canonical multi-eigenvector Lanczos implementation.
    O(k * n * iterations) instead of O(n^3) for dense solver.

    Args:
        L: Sparse Laplacian matrix (n x n)
        num_eigenvectors: Number of eigenvectors to compute
        num_iterations: Lanczos iterations
        tol: Convergence tolerance

    Returns:
        eigenvectors: (n, k) array of eigenvectors as columns
        eigenvalues: (k,) array of eigenvalues
    """
    n = L.shape[0]
    k = min(num_iterations, n - 1)

    # Initialize random vector, project out constant
    torch.manual_seed(42)
    v = torch.randn(n, device=DEVICE, dtype=torch.float32)
    v = v - v.mean()
    v_norm = torch.linalg.norm(v)
    if v_norm < tol:
        v = torch.ones(n, device=DEVICE, dtype=torch.float32)
        v[::2] = -1.0
        v = v - v.mean()
        v_norm = torch.linalg.norm(v)
    v = v / v_norm

    # Pre-allocate Lanczos vectors
    V = torch.zeros((n, k + 1), device=DEVICE, dtype=torch.float32)
    V[:, 0] = v

    alphas = torch.zeros(k, device=DEVICE, dtype=torch.float32)
    betas = torch.zeros(k, device=DEVICE, dtype=torch.float32)

    actual_k = k
    iteration = 0

    while iteration < k:
        # Sparse matvec
        w = torch.sparse.mm(L, V[:, iteration].unsqueeze(1)).squeeze(1)

        alpha = torch.dot(V[:, iteration], w)
        alphas[iteration] = alpha

        w = w - alpha * V[:, iteration]
        if iteration > 0:
            w = w - betas[iteration - 1] * V[:, iteration - 1]

        # Reorthogonalize
        if iteration > 0:
            V_prev = V[:, :iteration + 1]
            coeffs = torch.mm(V_prev.T, w.unsqueeze(1)).squeeze(1)
            w = w - torch.mm(V_prev, coeffs.unsqueeze(1)).squeeze(1)

        # Project out constant
        w = w - w.mean()

        beta = torch.linalg.norm(w)
        if beta < tol:
            actual_k = iteration + 1
            break

        betas[iteration] = beta
        V[:, iteration + 1] = w / beta
        iteration += 1

    if actual_k < 2:
        return torch.zeros((n, num_eigenvectors), device=DEVICE, dtype=torch.float32), torch.zeros(num_eigenvectors, device=DEVICE, dtype=torch.float32)

    # Build tridiagonal matrix
    T = torch.diag(alphas[:actual_k])
    if actual_k > 1:
        off_diag = torch.arange(actual_k - 1, device=DEVICE)
        T[off_diag, off_diag + 1] = betas[:actual_k - 1]
        T[off_diag + 1, off_diag] = betas[:actual_k - 1]

    # Solve small eigenproblem
    eigenvalues, eigenvectors_T = torch.linalg.eigh(T)

    # Select first num_eigenvectors non-trivial (skip near-zero)
    mask = eigenvalues > 1e-6
    valid_indices = torch.where(mask)[0]

    if len(valid_indices) == 0:
        return torch.zeros((n, num_eigenvectors), device=DEVICE, dtype=torch.float32), torch.zeros(num_eigenvectors, device=DEVICE, dtype=torch.float32)

    # Take first num_eigenvectors valid ones
    take_count = min(num_eigenvectors, len(valid_indices))
    selected_indices = valid_indices[:take_count]

    selected_eigenvalues = eigenvalues[selected_indices]

    # Reconstruct eigenvectors: E = V @ Y
    Y = eigenvectors_T[:, selected_indices]
    eigenvectors_full = torch.mm(V[:, :actual_k], Y)

    return eigenvectors_full, selected_eigenvalues


# ============================================================
# Heat diffusion on sparse Laplacian (for texture/image ops)
# ============================================================

def heat_diffusion_sparse(
    L: torch.Tensor,
    signal: torch.Tensor,
    alpha: float = 0.1,
    iterations: int = 10
) -> torch.Tensor:
    """
    Heat diffusion on sparse Laplacian: x_{t+1} = x_t - alpha * L @ x_t

    This is the core spectral smoothing operation.
    The number of iterations controls the "rotation angle" toward spectral domain.

    Args:
        L: sparse Laplacian (n x n)
        signal: input signal (n,) or (n, c) for multi-channel
        alpha: diffusion step size
        iterations: number of diffusion steps

    Returns:
        Diffused signal, same shape as input
    """
    x = signal.clone()

    # Handle 1D case
    was_1d = x.dim() == 1
    if was_1d:
        x = x.unsqueeze(1)

    iteration = 0
    while iteration < iterations:
        # Single sparse matmul per iteration
        Lx = torch.sparse.mm(L, x)
        x = x - alpha * Lx
        iteration += 1

    if was_1d:
        x = x.squeeze(1)

    return x


def build_image_laplacian(
    height: int,
    width: int,
    weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Build sparse Laplacian for image grid with optional edge weights.

    If weights is None, builds uniform 4-connected grid Laplacian.
    If weights is provided, should be (height, width, 4) for 4 neighbor weights.

    Returns sparse Laplacian on specified device.
    """
    if device is None:
        device = DEVICE

    n = height * width

    # Build indices for 4-connectivity
    # Horizontal edges: (y, x) -- (y, x+1)
    # Vertical edges: (y, x) -- (y+1, x)

    rows_list = []
    cols_list = []
    vals_list = []
    degrees = torch.zeros(n, device=device, dtype=torch.float32)

    # Horizontal edges (vectorized index computation)
    y_h = torch.arange(height, device=device)
    x_h = torch.arange(width - 1, device=device)
    yy_h, xx_h = torch.meshgrid(y_h, x_h, indexing='ij')

    idx_left = yy_h.flatten() * width + xx_h.flatten()
    idx_right = idx_left + 1

    if weights is not None:
        w_h = weights[:, :-1, 0].flatten()  # right neighbor weights
    else:
        w_h = torch.ones(len(idx_left), device=device)

    # Vertical edges
    y_v = torch.arange(height - 1, device=device)
    x_v = torch.arange(width, device=device)
    yy_v, xx_v = torch.meshgrid(y_v, x_v, indexing='ij')

    idx_top = yy_v.flatten() * width + xx_v.flatten()
    idx_bottom = idx_top + width

    if weights is not None:
        w_v = weights[:-1, :, 1].flatten()  # bottom neighbor weights
    else:
        w_v = torch.ones(len(idx_top), device=device)

    # Concatenate all edges
    all_rows = torch.cat([idx_left, idx_right, idx_top, idx_bottom,
                          idx_right, idx_left, idx_bottom, idx_top])
    all_cols = torch.cat([idx_right, idx_left, idx_bottom, idx_top,
                          idx_left, idx_right, idx_top, idx_bottom])
    all_vals = torch.cat([-w_h, -w_h, -w_v, -w_v,
                          -w_h, -w_h, -w_v, -w_v])

    # Compute degrees via scatter_add
    degrees.scatter_add_(0, all_rows[:len(all_rows)//2],
                         -all_vals[:len(all_vals)//2])

    # Add diagonal
    diag_idx = torch.arange(n, device=device)
    all_rows = torch.cat([all_rows, diag_idx])
    all_cols = torch.cat([all_cols, diag_idx])
    all_vals = torch.cat([all_vals, degrees])

    indices = torch.stack([all_rows.long(), all_cols.long()])
    L = torch.sparse_coo_tensor(indices, all_vals, (n, n)).coalesce()

    return L


def build_weighted_image_laplacian(
    carrier: torch.Tensor,
    edge_threshold: float = 0.1
) -> torch.Tensor:
    """
    Build weighted Laplacian from carrier image.

    Edge weights = exp(-|carrier_diff| / threshold)
    Near carrier edges: weak connections (anisotropic diffusion)
    In smooth regions: strong connections (isotropic diffusion)

    Args:
        carrier: (H, W) image tensor
        edge_threshold: controls edge sensitivity

    Returns:
        Sparse Laplacian (H*W, H*W)
    """
    device = carrier.device
    height, width = carrier.shape
    n = height * width

    # Compute edge weights from carrier differences (vectorized)
    diff_h = torch.abs(carrier[:, 1:] - carrier[:, :-1])
    diff_v = torch.abs(carrier[1:, :] - carrier[:-1, :])

    weights_h = torch.exp(-diff_h / edge_threshold)
    weights_v = torch.exp(-diff_v / edge_threshold)

    # Build COO indices
    y_h = torch.arange(height, device=device)
    x_h = torch.arange(width - 1, device=device)
    yy_h, xx_h = torch.meshgrid(y_h, x_h, indexing='ij')
    idx_left = (yy_h * width + xx_h).flatten()
    idx_right = idx_left + 1
    w_h = weights_h.flatten()

    y_v = torch.arange(height - 1, device=device)
    x_v = torch.arange(width, device=device)
    yy_v, xx_v = torch.meshgrid(y_v, x_v, indexing='ij')
    idx_top = (yy_v * width + xx_v).flatten()
    idx_bottom = idx_top + width
    w_v = weights_v.flatten()

    # Off-diagonal entries (both directions)
    rows = torch.cat([idx_left, idx_right, idx_top, idx_bottom])
    cols = torch.cat([idx_right, idx_left, idx_bottom, idx_top])
    vals = torch.cat([-w_h, -w_h, -w_v, -w_v])

    # Compute degrees
    degrees = torch.zeros(n, device=device)
    degrees.scatter_add_(0, idx_left, w_h)
    degrees.scatter_add_(0, idx_right, w_h)
    degrees.scatter_add_(0, idx_top, w_v)
    degrees.scatter_add_(0, idx_bottom, w_v)

    # Add diagonal
    diag_idx = torch.arange(n, device=device)
    rows = torch.cat([rows, diag_idx])
    cols = torch.cat([cols, diag_idx])
    vals = torch.cat([vals, degrees])

    indices = torch.stack([rows.long(), cols.long()])
    L = torch.sparse_coo_tensor(indices, vals, (n, n)).coalesce()

    return L


def build_multiscale_image_laplacian(
    carrier: torch.Tensor,
    radii: List[int] = [1, 2, 3, 4, 5, 6],
    radius_weights: List[float] = [1.0, 0.6, 0.4, 0.3, 0.2, 0.1],
    edge_threshold: float = 0.1
) -> torch.Tensor:
    """
    Build weighted Laplacian with multi-radius connectivity for dither patterns.

    VECTORIZED VERSION: Precomputes all offsets and processes them in batches
    instead of nested Python loops. ~10x faster than naive loop implementation.

    Dither patterns create isolated dots at radius 1. By adding edges at
    multiple radii, we connect pixels that are part of the same dither pattern
    but don't directly touch.

    Edge weight = radius_weight * exp(-color_diff / threshold)
    where color_diff is L2 norm for RGB or abs diff for grayscale.

    Args:
        carrier: (H, W) grayscale OR (H, W, 3) RGB image tensor
        radii: list of connectivity radii (e.g., [1, 2, 3, 4, 5, 6])
        radius_weights: weight multiplier for each radius level
        edge_threshold: controls edge sensitivity

    Returns:
        Sparse Laplacian (H*W, H*W)
    """
    device = carrier.device

    # Detect if RGB or grayscale
    is_rgb = carrier.dim() == 3 and carrier.shape[-1] == 3

    if is_rgb:
        height, width, _ = carrier.shape
        carrier_flat = carrier.reshape(-1, 3)  # (H*W, 3)
    else:
        height, width = carrier.shape
        carrier_flat = carrier.flatten()  # (H*W,)

    n = height * width

    # VECTORIZED: Precompute all (dy, dx, weight) triplets for all radii
    max_r = max(radii)
    offset_dy = []
    offset_dx = []
    offset_weight = []

    for radius, r_weight in zip(radii, radius_weights):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                if dx * dx + dy * dy > radius * radius:
                    continue
                offset_dy.append(dy)
                offset_dx.append(dx)
                offset_weight.append(r_weight)

    num_offsets = len(offset_dy)
    if num_offsets == 0:
        # Degenerate case: return identity Laplacian
        diag_idx = torch.arange(n, device=device)
        indices = torch.stack([diag_idx, diag_idx])
        vals = torch.ones(n, device=device)
        return torch.sparse_coo_tensor(indices, vals, (n, n)).coalesce()

    offset_dy = torch.tensor(offset_dy, device=device, dtype=torch.long)
    offset_dx = torch.tensor(offset_dx, device=device, dtype=torch.long)
    offset_weight = torch.tensor(offset_weight, device=device, dtype=torch.float32)

    # Create full pixel coordinate grid once
    y_grid = torch.arange(height, device=device, dtype=torch.long)
    x_grid = torch.arange(width, device=device, dtype=torch.long)
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing='ij')
    yy_flat = yy.flatten()  # (H*W,)
    xx_flat = xx.flatten()  # (H*W,)

    # For each pixel, compute destination coords for all offsets
    # dst_y[i, j] = y[i] + dy[j], dst_x[i, j] = x[i] + dx[j]
    dst_y = yy_flat.unsqueeze(1) + offset_dy.unsqueeze(0)  # (H*W, num_offsets)
    dst_x = xx_flat.unsqueeze(1) + offset_dx.unsqueeze(0)  # (H*W, num_offsets)

    # Mask valid destinations (within image bounds)
    valid_mask = (dst_y >= 0) & (dst_y < height) & (dst_x >= 0) & (dst_x < width)

    # Source indices: each pixel index, broadcast across offsets
    src_idx = torch.arange(n, device=device, dtype=torch.long).unsqueeze(1).expand(-1, num_offsets)

    # Destination indices
    dst_idx = dst_y * width + dst_x

    # Flatten and apply mask
    src_flat = src_idx[valid_mask]
    dst_flat = dst_idx[valid_mask]
    weights_flat = offset_weight.unsqueeze(0).expand(n, -1)[valid_mask]

    # Compute color differences for valid edges
    if is_rgb:
        color_diff = torch.norm(carrier_flat[src_flat] - carrier_flat[dst_flat], dim=-1)
    else:
        color_diff = torch.abs(carrier_flat[src_flat] - carrier_flat[dst_flat])

    # Compute edge weights
    edge_weights = weights_flat * torch.exp(-color_diff / edge_threshold)

    # Collect results
    all_rows = src_flat
    all_cols = dst_flat
    all_vals = -edge_weights  # Off-diagonal is negative

    if len(all_rows) == 0:
        # Degenerate case: return identity Laplacian
        diag_idx = torch.arange(n, device=device)
        indices = torch.stack([diag_idx, diag_idx])
        vals = torch.ones(n, device=device)
        return torch.sparse_coo_tensor(indices, vals, (n, n)).coalesce()

    # all_rows/cols/vals are already tensors from vectorized computation
    rows = all_rows
    cols = all_cols
    vals = all_vals

    # Compute degrees (sum of absolute off-diagonal values per row)
    degrees = torch.zeros(n, device=device)
    degrees.scatter_add_(0, rows, -vals)  # vals are negative, so negate

    # Add diagonal entries
    diag_idx = torch.arange(n, device=device)
    rows = torch.cat([rows, diag_idx])
    cols = torch.cat([cols, diag_idx])
    vals = torch.cat([vals, degrees])

    indices = torch.stack([rows.long(), cols.long()])
    L = torch.sparse_coo_tensor(indices, vals, (n, n)).coalesce()

    return L


# ============================================================
# Tiled Local Eigenvector Computation (Algorithm 1: Sparse Texture Etching)
# ============================================================

def compute_local_eigenvectors_tiled(
    image: torch.Tensor,
    tile_size: int = 64,
    overlap: int = 16,
    num_eigenvectors: int = 4,
    edge_threshold: float = 0.1,
    lanczos_iterations: int = 30
) -> torch.Tensor:
    """
    Compute eigenvectors in overlapping tiles, blend results.

    This is O(n * k * iters) where k << n, suitable for arbitrarily large images.
    Never materializes full image Laplacian - only tile-sized Laplacians.

    Args:
        image: 2D tensor (H, W) - can be arbitrarily large
        tile_size: size of each tile (default 64)
        overlap: overlap between adjacent tiles for blending (default 16)
        num_eigenvectors: eigenvectors per tile (default 4)
        edge_threshold: carrier edge sensitivity
        lanczos_iterations: iterations for Lanczos per tile

    Returns:
        eigenvector_images: (H, W, k) tensor of blended eigenvector fields

    Memory usage: O(tile_size^2 * k) per tile, not O(H*W)
    """
    device = image.device
    height, width = image.shape

    # Ensure image is float and normalized
    if image.max() > 1.0:
        image = image.to(dtype=torch.float32) / 255.0
    else:
        image = image.to(dtype=torch.float32)

    # Output accumulator and weight map for blending
    result = torch.zeros((height, width, num_eigenvectors), dtype=torch.float32, device=device)
    weights = torch.zeros((height, width), dtype=torch.float32, device=device)

    # Compute tile positions with overlap
    step = tile_size - overlap

    y_starts = list(range(0, height - tile_size + 1, step))
    if not y_starts or y_starts[-1] + tile_size < height:
        y_starts.append(max(0, height - tile_size))

    x_starts = list(range(0, width - tile_size + 1, step))
    if not x_starts or x_starts[-1] + tile_size < width:
        x_starts.append(max(0, width - tile_size))

    # Remove duplicates and sort
    y_starts = sorted(set(y_starts))
    x_starts = sorted(set(x_starts))

    tile_idx = 0
    y_idx = 0
    while y_idx < len(y_starts):
        y = y_starts[y_idx]
        x_idx = 0
        while x_idx < len(x_starts):
            x = x_starts[x_idx]

            # Extract tile
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            tile = image[y:y_end, x:x_end]

            tile_h, tile_w = tile.shape

            if tile_h < 4 or tile_w < 4:
                x_idx += 1
                continue

            # Compute eigenvectors for this tile using local Lanczos
            tile_eigenvectors = _compute_tile_eigenvectors(
                tile,
                num_eigenvectors,
                edge_threshold,
                lanczos_iterations
            )

            # Create blending weights (smooth falloff at edges)
            blend_y = _create_blend_weights_1d(tile_h, overlap, device)
            blend_x = _create_blend_weights_1d(tile_w, overlap, device)
            blend_weight = torch.outer(blend_y, blend_x)

            # Accumulate with blending
            k_actual = tile_eigenvectors.shape[2]
            k_idx = 0
            while k_idx < k_actual:
                result[y:y_end, x:x_end, k_idx] += (
                    tile_eigenvectors[:, :, k_idx] * blend_weight
                )
                k_idx += 1
            weights[y:y_end, x:x_end] += blend_weight

            tile_idx += 1
            x_idx += 1
        y_idx += 1

    # Normalize by accumulated weights (vectorized torch division)
    weights_expanded = weights.unsqueeze(-1).clamp(min=1e-10)
    result = result / weights_expanded

    return result


def _compute_tile_eigenvectors(
    tile: torch.Tensor,
    num_eigenvectors: int,
    edge_threshold: float,
    lanczos_iterations: int
) -> torch.Tensor:
    """
    Compute eigenvectors for a single tile using GPU Lanczos.

    Returns (tile_h, tile_w, k) tensor.
    """
    tile_h, tile_w = tile.shape
    n = tile_h * tile_w
    device = tile.device

    # Build weighted Laplacian for this tile only
    L = build_weighted_image_laplacian(tile, edge_threshold)

    k = min(num_eigenvectors, lanczos_iterations - 1, n - 2)

    if n < 3 or k < 1:
        return torch.zeros((tile_h, tile_w, num_eigenvectors), dtype=torch.float32, device=device)

    # GPU Lanczos
    torch.manual_seed(42 + hash((tile_h, tile_w)) % 10000)
    v = torch.randn(n, device=device, dtype=torch.float32)
    v = v - v.mean()
    v_norm = torch.linalg.norm(v)
    if v_norm < 1e-10:
        v = torch.ones(n, device=device, dtype=torch.float32)
        v[::2] = -1.0
        v = v - v.mean()
        v_norm = torch.linalg.norm(v)
    v = v / v_norm

    actual_iters = min(lanczos_iterations, n - 1)
    V = torch.zeros((n, actual_iters + 1), device=device, dtype=torch.float32)
    V[:, 0] = v

    alphas = torch.zeros(actual_iters, device=device, dtype=torch.float32)
    betas = torch.zeros(actual_iters, device=device, dtype=torch.float32)

    actual_k = actual_iters
    iteration = 0

    while iteration < actual_iters:
        w = torch.sparse.mm(L, V[:, iteration].unsqueeze(1)).squeeze(1)
        alpha = torch.dot(V[:, iteration], w)
        alphas[iteration] = alpha

        w = w - alpha * V[:, iteration]
        if iteration > 0:
            w = w - betas[iteration - 1] * V[:, iteration - 1]

        # Reorthogonalize
        if iteration > 0:
            V_prev = V[:, :iteration + 1]
            coeffs = torch.mm(V_prev.T, w.unsqueeze(1)).squeeze(1)
            w = w - torch.mm(V_prev, coeffs.unsqueeze(1)).squeeze(1)

        w = w - w.mean()
        beta = torch.linalg.norm(w)

        if beta < 1e-10:
            actual_k = iteration + 1
            break

        betas[iteration] = beta
        V[:, iteration + 1] = w / beta
        iteration += 1

    if actual_k < 2:
        return torch.zeros((tile_h, tile_w, num_eigenvectors), dtype=torch.float32, device=device)

    # Build and solve tridiagonal eigenproblem
    T = torch.diag(alphas[:actual_k])
    if actual_k > 1:
        off_diag = torch.arange(actual_k - 1, device=device)
        T[off_diag, off_diag + 1] = betas[:actual_k - 1]
        T[off_diag + 1, off_diag] = betas[:actual_k - 1]

    eigenvalues, eigenvectors_T = torch.linalg.eigh(T)

    # Select non-trivial eigenvectors
    mask = eigenvalues > 1e-6
    valid_indices = torch.where(mask)[0]

    if len(valid_indices) == 0:
        return torch.zeros((tile_h, tile_w, num_eigenvectors), dtype=torch.float32, device=device)

    take_count = min(num_eigenvectors, len(valid_indices))
    selected_indices = valid_indices[:take_count]

    # Reconstruct full eigenvectors
    Y = eigenvectors_T[:, selected_indices]
    eigenvectors_full = torch.mm(V[:, :actual_k], Y)

    # Reshape to tile images using torch
    result = torch.zeros((tile_h, tile_w, num_eigenvectors), dtype=torch.float32, device=device)

    k_idx = 0
    while k_idx < take_count:
        result[:, :, k_idx] = eigenvectors_full[:, k_idx].reshape(tile_h, tile_w)
        k_idx += 1

    return result


def _create_blend_weights_1d(length: int, overlap: int, device: torch.device) -> torch.Tensor:
    """
    Create 1D blending weights with smooth falloff at boundaries.

    Center region has weight 1, edges taper linearly over overlap distance.
    """
    weights = torch.ones(length, dtype=torch.float32, device=device)

    if overlap <= 0 or length <= 2 * overlap:
        return weights

    # Vectorized taper
    taper = torch.arange(1, overlap + 1, device=device, dtype=torch.float32) / (overlap + 1)
    weights[:overlap] = taper
    weights[-overlap:] = taper.flip(0)

    return weights


# ============================================================
# Dither-aware tiled spectral: multi-radius connectivity
# ============================================================

def _compute_tile_eigenvectors_multiscale(
    tile: torch.Tensor,
    num_eigenvectors: int,
    radii: List[int],
    radius_weights: List[float],
    edge_threshold: float,
    lanczos_iterations: int
) -> torch.Tensor:
    """
    Compute eigenvectors for a single tile using multi-radius Laplacian.

    This connects dither patterns that don't directly touch by adding
    edges at multiple radii.

    Args:
        tile: (H, W) grayscale OR (H, W, 3) RGB tensor

    Returns (tile_h, tile_w, k) tensor.
    """
    device = tile.device
    if tile.dim() == 3:
        tile_h, tile_w, _ = tile.shape
    else:
        tile_h, tile_w = tile.shape
    n = tile_h * tile_w

    # Build multi-radius Laplacian for this tile
    L = build_multiscale_image_laplacian(tile, radii, radius_weights, edge_threshold)

    k = min(num_eigenvectors, lanczos_iterations - 1, n - 2)

    if n < 3 or k < 1:
        return torch.zeros((tile_h, tile_w, num_eigenvectors), dtype=torch.float32, device=device)

    # GPU Lanczos (same as standard version)
    torch.manual_seed(42 + hash((tile_h, tile_w)) % 10000)
    v = torch.randn(n, device=device, dtype=torch.float32)
    v = v - v.mean()
    v_norm = torch.linalg.norm(v)
    if v_norm < 1e-10:
        v = torch.ones(n, device=device, dtype=torch.float32)
        v[::2] = -1.0
        v = v - v.mean()
        v_norm = torch.linalg.norm(v)
    v = v / v_norm

    actual_iters = min(lanczos_iterations, n - 1)
    V = torch.zeros((n, actual_iters + 1), device=device, dtype=torch.float32)
    V[:, 0] = v

    alphas = torch.zeros(actual_iters, device=device, dtype=torch.float32)
    betas = torch.zeros(actual_iters, device=device, dtype=torch.float32)

    actual_k = actual_iters
    iteration = 0

    while iteration < actual_iters:
        w = torch.sparse.mm(L, V[:, iteration].unsqueeze(1)).squeeze(1)
        alpha = torch.dot(V[:, iteration], w)
        alphas[iteration] = alpha

        w = w - alpha * V[:, iteration]
        if iteration > 0:
            w = w - betas[iteration - 1] * V[:, iteration - 1]

        # Reorthogonalize
        if iteration > 0:
            V_prev = V[:, :iteration + 1]
            coeffs = torch.mm(V_prev.T, w.unsqueeze(1)).squeeze(1)
            w = w - torch.mm(V_prev, coeffs.unsqueeze(1)).squeeze(1)

        w = w - w.mean()
        beta = torch.linalg.norm(w)

        if beta < 1e-10:
            actual_k = iteration + 1
            break

        betas[iteration] = beta
        V[:, iteration + 1] = w / beta
        iteration += 1

    if actual_k < 2:
        return torch.zeros((tile_h, tile_w, num_eigenvectors), dtype=torch.float32, device=device)

    # Build and solve tridiagonal eigenproblem
    T = torch.diag(alphas[:actual_k])
    if actual_k > 1:
        off_diag = torch.arange(actual_k - 1, device=device)
        T[off_diag, off_diag + 1] = betas[:actual_k - 1]
        T[off_diag + 1, off_diag] = betas[:actual_k - 1]

    eigenvalues, eigenvectors_T = torch.linalg.eigh(T)

    # Select non-trivial eigenvectors
    mask = eigenvalues > 1e-6
    valid_indices = torch.where(mask)[0]

    if len(valid_indices) == 0:
        return torch.zeros((tile_h, tile_w, num_eigenvectors), dtype=torch.float32, device=device)

    take_count = min(num_eigenvectors, len(valid_indices))
    selected_indices = valid_indices[:take_count]

    # Reconstruct full eigenvectors
    Y = eigenvectors_T[:, selected_indices]
    eigenvectors_full = torch.mm(V[:, :actual_k], Y)

    # Reshape to tile images using torch
    result = torch.zeros((tile_h, tile_w, num_eigenvectors), dtype=torch.float32, device=device)

    k_idx = 0
    while k_idx < take_count:
        result[:, :, k_idx] = eigenvectors_full[:, k_idx].reshape(tile_h, tile_w)
        k_idx += 1

    return result


def compute_local_eigenvectors_tiled_dither(
    image: torch.Tensor,
    tile_size: int = 64,
    overlap: int = 16,
    num_eigenvectors: int = 4,
    radii: List[int] = [1, 2, 3, 4, 5, 6],
    radius_weights: List[float] = [1.0, 0.6, 0.4, 0.3, 0.2, 0.1],
    edge_threshold: float = 0.15,
    lanczos_iterations: int = 30
) -> torch.Tensor:
    """
    Compute eigenvectors in overlapping tiles with dither-aware multi-radius connectivity.

    For dithered images, standard 4-connected graphs see dots as isolated.
    Multi-radius connectivity (radii 1-6) connects dither patterns that
    belong together but don't directly touch.

    This is O(n * k * iters) per tile, suitable for arbitrarily large images.
    Never materializes full image Laplacian - only tile-sized Laplacians.

    Args:
        image: (H, W) grayscale OR (H, W, 3) RGB tensor - can be arbitrarily large
        tile_size: size of each tile (default 64)
        overlap: overlap between adjacent tiles for blending (default 16)
        num_eigenvectors: eigenvectors per tile (default 4)
        radii: list of connectivity radii (default [1,2,3,4,5,6] for typical dither)
        radius_weights: weight multiplier for each radius level
        edge_threshold: carrier edge sensitivity (higher = more tolerant)
        lanczos_iterations: iterations for Lanczos per tile

    Returns:
        eigenvector_images: (H, W, k) tensor of blended eigenvector fields

    Memory usage: O(tile_size^2 * k) per tile, not O(H*W)
    """
    device = image.device

    # Detect if RGB or grayscale
    is_rgb = image.dim() == 3 and image.shape[-1] == 3

    if is_rgb:
        height, width, _ = image.shape
    else:
        height, width = image.shape

    # Ensure image is float and normalized
    if image.max() > 1.0:
        image = image.to(dtype=torch.float32) / 255.0
    else:
        image = image.to(dtype=torch.float32)

    # Output accumulator and weight map for blending
    result = torch.zeros((height, width, num_eigenvectors), dtype=torch.float32, device=device)
    weights = torch.zeros((height, width), dtype=torch.float32, device=device)

    # Compute tile positions with overlap
    step = tile_size - overlap

    y_starts = list(range(0, height - tile_size + 1, step))
    if not y_starts or y_starts[-1] + tile_size < height:
        y_starts.append(max(0, height - tile_size))

    x_starts = list(range(0, width - tile_size + 1, step))
    if not x_starts or x_starts[-1] + tile_size < width:
        x_starts.append(max(0, width - tile_size))

    # Remove duplicates and sort
    y_starts = sorted(set(y_starts))
    x_starts = sorted(set(x_starts))

    total_tiles = len(y_starts) * len(x_starts)
    print(f"Processing {total_tiles} tiles ({len(y_starts)}x{len(x_starts)}) with radii {radii}...")

    tile_idx = 0
    y_idx = 0
    while y_idx < len(y_starts):
        y = y_starts[y_idx]
        x_idx = 0
        while x_idx < len(x_starts):
            x = x_starts[x_idx]

            # Extract tile (preserves RGB channels if present)
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            tile = image[y:y_end, x:x_end]

            if is_rgb:
                tile_h, tile_w, _ = tile.shape
            else:
                tile_h, tile_w = tile.shape

            if tile_h < 4 or tile_w < 4:
                x_idx += 1
                continue

            # Compute eigenvectors for this tile using multi-radius Laplacian
            tile_eigenvectors = _compute_tile_eigenvectors_multiscale(
                tile,
                num_eigenvectors,
                radii,
                radius_weights,
                edge_threshold,
                lanczos_iterations
            )

            # Create blending weights (smooth falloff at edges)
            blend_y = _create_blend_weights_1d(tile_h, overlap, device)
            blend_x = _create_blend_weights_1d(tile_w, overlap, device)
            blend_weight = torch.outer(blend_y, blend_x)

            # Accumulate with blending
            k_actual = tile_eigenvectors.shape[2]
            k_idx = 0
            while k_idx < k_actual:
                result[y:y_end, x:x_end, k_idx] += (
                    tile_eigenvectors[:, :, k_idx] * blend_weight
                )
                k_idx += 1
            weights[y:y_end, x:x_end] += blend_weight

            tile_idx += 1
            if tile_idx % 20 == 0:
                print(f"  {tile_idx}/{total_tiles} tiles...")
            x_idx += 1
        y_idx += 1

    # Normalize by accumulated weights (vectorized torch division)
    weights_expanded = weights.unsqueeze(-1).clamp(min=1e-10)
    result = result / weights_expanded

    print(f"Done. Output shape: {result.shape}")
    return result


# ============================================================
# Chebyshev Polynomial Filtering and Iterative Spectral Transform
# ============================================================
#
# MATHEMATICAL FRAMEWORK
# ======================
#
# The key insight: spectral filtering g(L) where L is the graph Laplacian
# can be approximated using Chebyshev polynomials without computing eigenvectors.
#
# 1. CHEBYSHEV POLYNOMIALS
#    ---------------------
#    Chebyshev polynomials T_k(x) form an orthogonal basis on [-1, 1] with:
#      T_0(x) = 1
#      T_1(x) = x
#      T_k(x) = 2x * T_{k-1}(x) - T_{k-2}(x)   [recurrence relation]
#
#    Any smooth function f(x) on [-1, 1] can be approximated:
#      f(x) ≈ sum_{k=0}^{K} c_k * T_k(x)
#
# 2. SPECTRAL FILTERING VIA CHEBYSHEV
#    ---------------------------------
#    For a Laplacian L with eigenvalues in [0, lambda_max]:
#      - Rescale: L_tilde = 2*L/lambda_max - I  (eigenvalues in [-1, 1])
#      - Filter: g(L) = sum_k c_k * T_k(L_tilde)
#
#    The magic: T_k(L_tilde) @ signal only requires matrix-vector products!
#      T_0(L_tilde) @ x = x
#      T_1(L_tilde) @ x = L_tilde @ x
#      T_k(L_tilde) @ x = 2 * L_tilde @ T_{k-1}(L_tilde) @ x - T_{k-2}(L_tilde) @ x
#
#    Cost: O(K * nnz(L)) for K terms, where nnz(L) is number of nonzeros.
#    For sparse graphs with bounded degree: O(K * n)
#
# 3. GAUSSIAN SPECTRAL FILTER
#    -------------------------
#    To emphasize eigenvectors with eigenvalue near lambda_target:
#      g(lambda) = exp(-(lambda - lambda_target)^2 / (2 * sigma^2))
#
#    The Chebyshev coefficients for this Gaussian are:
#      c_k = (2/pi) * integral_{-1}^{1} g(x) * T_k(x) / sqrt(1-x^2) dx
#
#    For a Gaussian, these coefficients decay exponentially with k,
#    so only ~20-50 terms are needed for high accuracy.
#
# 4. ITERATIVE SPECTRAL TRANSFORM
#    -----------------------------
#    The "rotation angle" theta parameterizes which eigenvectors dominate:
#      - theta=0: Low eigenvectors (Fiedler, coarse structure)
#      - theta=1: High eigenvectors (fine spectral detail)
#
#    Instead of computing eigenvectors explicitly:
#      spectral_transform(theta) @ signal ≈ g_theta(L) @ signal
#
#    Where g_theta is a Gaussian filter centered at theta * lambda_max.
#
#    For large theta, we can iterate small theta steps:
#      - Each step shifts the spectral center slightly
#      - The composition approximates a larger rotation
#      - Total complexity: O(num_steps * polynomial_order * n)
#
# 5. COMPLEXITY ANALYSIS
#    --------------------
#    - Explicit eigenvectors: O(n^3) or O(k*n*iterations) for Lanczos
#    - Chebyshev filtering: O(order * nnz(L)) per application
#    - For image grids: nnz(L) = O(n), so O(order * n) per filter
#    - With O(log n) precision steps: O(n log n) total
#
# ============================================================


def estimate_lambda_max(
    L: torch.Tensor,
    num_iterations: int = 20,
    tol: float = 1e-6
) -> float:
    """
    Estimate largest eigenvalue of L using power iteration.

    This is needed to rescale L for Chebyshev polynomials.

    Args:
        L: Sparse Laplacian matrix (n x n)
        num_iterations: Maximum iterations for power method
        tol: Convergence tolerance

    Returns:
        Approximate lambda_max (largest eigenvalue)
    """
    n = L.shape[0]

    # Random starting vector
    torch.manual_seed(123)
    v = torch.randn(n, device=L.device, dtype=torch.float32)
    v = v / torch.linalg.norm(v)

    lambda_est = 0.0
    iteration = 0

    while iteration < num_iterations:
        # w = L @ v
        w = torch.sparse.mm(L, v.unsqueeze(1)).squeeze(1)

        # Rayleigh quotient
        lambda_new = torch.dot(v, w).item()

        # Normalize
        w_norm = torch.linalg.norm(w)
        if w_norm < 1e-10:
            break
        v = w / w_norm

        # Check convergence
        if abs(lambda_new - lambda_est) < tol * abs(lambda_new):
            lambda_est = lambda_new
            break

        lambda_est = lambda_new
        iteration += 1

    # Safety margin: eigenvalue could be slightly underestimated
    return lambda_est * 1.05


def chebyshev_coefficients_gaussian(
    center: float,
    width: float,
    order: int,
    num_quadrature: int = 200
) -> torch.Tensor:
    """
    Compute Chebyshev coefficients for a Gaussian filter.

    The filter is: g(x) = exp(-(x - center)^2 / (2 * width^2))
    where x is the rescaled eigenvalue in [-1, 1].

    Uses Chebyshev-Gauss quadrature for coefficient computation:
      c_k = (2/pi) * sum_j g(cos(theta_j)) * T_k(cos(theta_j)) * weight_j

    Args:
        center: Center of Gaussian in rescaled domain [-1, 1]
        width: Width (sigma) of Gaussian in rescaled domain
        order: Number of Chebyshev terms (polynomial degree)
        num_quadrature: Number of quadrature points

    Returns:
        Tensor of Chebyshev coefficients (order,)
    """
    # Compute on CPU to avoid CUDA dtype issues with mv
    # This is only done once per filter application, so cost is negligible

    # Chebyshev-Gauss quadrature points: theta_j = (2j-1)*pi / (2*N)
    j = torch.arange(1, num_quadrature + 1, dtype=torch.float32)
    theta = (2 * j - 1) * torch.pi / (2 * num_quadrature)
    x = torch.cos(theta)  # Quadrature points in [-1, 1]

    # Evaluate Gaussian at quadrature points
    if width > 1e-10:
        g_vals = torch.exp(-((x - center) ** 2) / (2 * width ** 2))
    else:
        # Delta function approximation
        g_vals = torch.zeros_like(x)
        closest_idx = torch.argmin(torch.abs(x - center))
        g_vals[closest_idx] = 1.0

    # Compute Chebyshev polynomials at quadrature points
    # T_k(cos(theta)) = cos(k * theta)
    k = torch.arange(order, dtype=torch.float32)
    T_k_vals = torch.cos(k.unsqueeze(1) * theta.unsqueeze(0))  # (order, num_quad)

    # Coefficients via quadrature
    # c_k = (2/N) * sum_j g(x_j) * T_k(x_j), with c_0 scaled by 1/2
    coeffs = (2.0 / num_quadrature) * torch.mv(T_k_vals, g_vals)
    coeffs[0] = coeffs[0] / 2.0  # Standard Chebyshev convention

    return coeffs


def chebyshev_filter(
    L: torch.Tensor,
    signal: torch.Tensor,
    center: float,
    width: float,
    order: int = 30,
    lambda_max: Optional[float] = None
) -> torch.Tensor:
    """
    Apply Chebyshev polynomial filter to signal on graph Laplacian.

    Approximates g(L) @ signal where g is a Gaussian centered at 'center'
    with standard deviation 'width', without computing eigenvectors.

    The filter emphasizes eigenvectors with eigenvalue near 'center'.

    Mathematical operation:
        1. Rescale L: L_tilde = 2*L/lambda_max - I  (eigenvalues in [-1, 1])
        2. Compute Chebyshev coefficients for Gaussian filter
        3. Apply filter: result = sum_k c_k * T_k(L_tilde) @ signal
           Using Chebyshev recurrence (only matrix-vector products needed)

    Complexity: O(order * nnz(L)), where nnz(L) is number of nonzeros.

    Args:
        L: Sparse Laplacian matrix (n x n)
        signal: Input signal (n,) or (n, c) for multi-channel
        center: Center of filter in eigenvalue space [0, lambda_max]
        width: Width (sigma) of filter in eigenvalue space
        order: Number of Chebyshev polynomial terms (higher = more accurate)
        lambda_max: Largest eigenvalue (estimated if not provided)

    Returns:
        Filtered signal, same shape as input
    """
    n = L.shape[0]
    device = L.device

    # Estimate lambda_max if not provided
    if lambda_max is None:
        lambda_max = estimate_lambda_max(L)

    # Ensure lambda_max is positive
    if lambda_max < 1e-10:
        lambda_max = 1.0

    # Handle 1D case
    was_1d = signal.dim() == 1
    if was_1d:
        signal = signal.unsqueeze(1)

    # Rescale center and width to [-1, 1] domain
    # Original eigenvalue lambda -> rescaled x = 2*lambda/lambda_max - 1
    # So lambda = (x + 1) * lambda_max / 2
    # If center is in [0, lambda_max], rescaled_center is in [-1, 1]
    rescaled_center = 2 * center / lambda_max - 1.0
    rescaled_width = 2 * width / lambda_max

    # Clamp rescaled_center to valid range
    rescaled_center = max(-1.0, min(1.0, rescaled_center))

    # Compute Chebyshev coefficients
    coeffs = chebyshev_coefficients_gaussian(rescaled_center, rescaled_width, order)
    coeffs = coeffs.to(device)

    # Apply filter using Chebyshev recurrence
    # T_0(L_tilde) @ x = x
    # T_1(L_tilde) @ x = L_tilde @ x
    # T_k(L_tilde) @ x = 2 * L_tilde @ T_{k-1} - T_{k-2}

    # First, precompute the scaling factor for L_tilde = 2*L/lambda_max - I
    # L_tilde @ x = (2/lambda_max) * L @ x - x
    scale = 2.0 / lambda_max

    # T_0 term
    T_prev_prev = signal  # T_0 @ signal = signal
    result = coeffs[0] * T_prev_prev

    if order > 1:
        # T_1 term: T_1(L_tilde) @ signal = L_tilde @ signal
        L_signal = torch.sparse.mm(L, signal)
        T_prev = scale * L_signal - signal  # L_tilde @ signal
        result = result + coeffs[1] * T_prev

    # Remaining terms via recurrence
    k = 2
    while k < order:
        # T_k(L_tilde) @ signal = 2 * L_tilde @ T_{k-1} - T_{k-2}
        L_T_prev = torch.sparse.mm(L, T_prev)
        T_curr = 2.0 * (scale * L_T_prev - T_prev) - T_prev_prev

        result = result + coeffs[k] * T_curr

        # Shift for next iteration
        T_prev_prev = T_prev
        T_prev = T_curr
        k += 1

    if was_1d:
        result = result.squeeze(1)

    return result


def iterative_spectral_transform(
    L: torch.Tensor,
    signal: torch.Tensor,
    theta: float,
    num_steps: int = 8,
    polynomial_order: int = 30,
    lambda_max: Optional[float] = None,
    sigma_factor: float = 0.3
) -> torch.Tensor:
    """
    Apply spectral transform by iterating small-angle Chebyshev filters.

    This approximates the effect of weighting eigenvectors by a Gaussian
    centered at theta * lambda_max, without computing eigenvectors.

    The theta parameter controls spectral emphasis:
        - theta=0: Emphasize low eigenvectors (Fiedler, coarse structure)
        - theta=1: Emphasize high eigenvectors (fine spectral detail)

    Algorithm:
        1. Divide [0, theta] into num_steps intervals
        2. For each step, apply Chebyshev filter centered at that spectral location
        3. Each filter incrementally shifts spectral emphasis

    Complexity: O(num_steps * polynomial_order * nnz(L))
    For sparse graphs: O(num_steps * polynomial_order * n) = O(n log n) with proper tuning

    Args:
        L: Sparse Laplacian matrix (n x n)
        signal: Input signal (n,) or (n, c) for multi-channel
        theta: Target rotation angle in [0, 1]
        num_steps: Number of iterative steps (more = smoother transition)
        polynomial_order: Chebyshev polynomial degree per step
        lambda_max: Largest eigenvalue (estimated if not provided)
        sigma_factor: Width of each filter as fraction of step size

    Returns:
        Transformed signal emphasizing eigenvectors near theta * lambda_max
    """
    device = L.device

    # Estimate lambda_max if not provided
    if lambda_max is None:
        lambda_max = estimate_lambda_max(L)

    if lambda_max < 1e-10:
        lambda_max = 1.0

    # Handle edge cases
    if theta <= 0:
        # At theta=0, return signal weighted by low-frequency component
        # Use filter centered at small eigenvalue
        return chebyshev_filter(
            L, signal,
            center=0.1 * lambda_max,
            width=0.2 * lambda_max,
            order=polynomial_order,
            lambda_max=lambda_max
        )

    if num_steps < 1:
        num_steps = 1

    # Step size in eigenvalue space
    delta_theta = theta / num_steps
    delta_lambda = delta_theta * lambda_max

    # Width of each filter (controls overlap between steps)
    sigma = sigma_factor * delta_lambda
    if sigma < 0.05 * lambda_max:
        sigma = 0.05 * lambda_max  # Minimum width for stability

    current = signal.clone()

    step = 0
    while step < num_steps:
        # Current spectral center
        current_theta = (step + 0.5) * delta_theta  # Center of this step's interval
        current_lambda = current_theta * lambda_max

        # Apply Chebyshev filter for this step
        current = chebyshev_filter(
            L, current,
            center=current_lambda,
            width=sigma,
            order=polynomial_order,
            lambda_max=lambda_max
        )

        # Normalize to prevent amplitude decay/growth
        current_norm = torch.linalg.norm(current)
        if current_norm > 1e-10:
            signal_norm = torch.linalg.norm(signal)
            if signal_norm > 1e-10:
                current = current * (signal_norm / current_norm)

        step += 1

    return current


def polynomial_spectral_field(
    L: torch.Tensor,
    signal: torch.Tensor,
    theta: float,
    num_bands: int = 5,
    polynomial_order: int = 30,
    lambda_max: Optional[float] = None
) -> torch.Tensor:
    """
    Compute spectral field weighted by theta using polynomial filtering.

    This is the polynomial approximation to the explicit eigenvector formula:
        spectral_field = sum_k weight_k * |eigenvector_k|
    where weights are Gaussian-distributed around theta * num_eigenvectors.

    Instead of computing eigenvectors, we use band-pass filters:
        spectral_field = sum_b filter_b(L) @ |signal|
    where each filter_b emphasizes a band of eigenvalues weighted by
    distance from the target theta.

    Args:
        L: Sparse Laplacian matrix (n x n)
        signal: Input signal (n,) - typically the carrier flattened
        theta: Rotation angle in [0, 1]
        num_bands: Number of spectral bands to sum
        polynomial_order: Chebyshev polynomial degree
        lambda_max: Largest eigenvalue (estimated if not provided)

    Returns:
        Spectral field (n,) weighted by theta
    """
    device = L.device
    n = L.shape[0]

    if lambda_max is None:
        lambda_max = estimate_lambda_max(L)

    if lambda_max < 1e-10:
        lambda_max = 1.0

    # Target eigenvalue
    target_lambda = theta * lambda_max

    # Create bands spanning [0, lambda_max]
    # Each band gets a Gaussian weight based on distance from target
    band_centers = torch.linspace(0.05 * lambda_max, 0.95 * lambda_max, num_bands, device=device)
    band_width = (lambda_max / num_bands) * 0.6  # Overlapping bands

    # Weights for each band (Gaussian centered at target)
    band_sigma = lambda_max / (2 * num_bands)  # Spread of weighting
    band_weights = torch.exp(-((band_centers - target_lambda) ** 2) / (2 * band_sigma ** 2))

    # Normalize weights
    weight_sum = band_weights.sum()
    if weight_sum > 1e-10:
        band_weights = band_weights / weight_sum

    # Handle 1D signal
    was_1d = signal.dim() == 1
    if was_1d:
        signal = signal.unsqueeze(1)

    # Sum weighted band-filtered signals
    result = torch.zeros_like(signal)

    band_idx = 0
    while band_idx < num_bands:
        center = band_centers[band_idx].item()
        weight = band_weights[band_idx].item()

        if weight > 1e-10:
            filtered = chebyshev_filter(
                L, signal,
                center=center,
                width=band_width,
                order=polynomial_order,
                lambda_max=lambda_max
            )
            result = result + weight * torch.abs(filtered)

        band_idx += 1

    if was_1d:
        result = result.squeeze(1)

    return result


def spectral_projection_filter(
    L: torch.Tensor,
    signal: torch.Tensor,
    theta: float,
    num_probes: int = 10,
    polynomial_order: int = 30,
    lambda_max: Optional[float] = None
) -> torch.Tensor:
    """
    Approximate projection onto theta-weighted eigenvector subspace using random probes.

    This uses a stochastic approach to approximate the explicit eigenvector formula:
        result = sum_k weight_k * (v_k @ signal) * v_k

    The approach:
        1. Generate random probe vectors
        2. Filter each probe through a Gaussian band-pass centered at theta
        3. Project signal onto filtered probes
        4. Reconstruct using weighted probe responses

    This produces a signal that emphasizes the spectral components near theta * lambda_max.

    Args:
        L: Sparse Laplacian matrix (n x n)
        signal: Input signal (n,)
        theta: Rotation angle in [0, 1]
        num_probes: Number of random probe vectors
        polynomial_order: Chebyshev polynomial degree
        lambda_max: Largest eigenvalue (estimated if not provided)

    Returns:
        Filtered signal (n,) emphasizing components near theta * lambda_max
    """
    device = L.device
    n = L.shape[0]

    if lambda_max is None:
        lambda_max = estimate_lambda_max(L)

    if lambda_max < 1e-10:
        lambda_max = 1.0

    # Target spectral center
    target_lambda = theta * lambda_max

    # Filter width - controls spectral selectivity
    filter_width = 0.2 * lambda_max

    # Handle 1D signal
    was_1d = signal.dim() == 1
    if was_1d:
        signal = signal.unsqueeze(1)

    # Generate random probe vectors
    torch.manual_seed(42)
    probes = torch.randn(n, num_probes, device=device, dtype=torch.float32)

    # Project out constant component from each probe
    probes = probes - probes.mean(dim=0, keepdim=True)

    # Normalize probes
    probe_norms = torch.linalg.norm(probes, dim=0, keepdim=True)
    probes = probes / (probe_norms + 1e-10)

    # Filter probes through band-pass centered at target
    filtered_probes = chebyshev_filter(
        L, probes,
        center=target_lambda,
        width=filter_width,
        order=polynomial_order,
        lambda_max=lambda_max
    )

    # Project signal onto filtered probes: coeffs = (filtered_probes)^T @ signal
    coeffs = torch.mm(filtered_probes.T, signal)  # (num_probes, c)

    # Reconstruct: result = filtered_probes @ coeffs / num_probes
    result = torch.mm(filtered_probes, coeffs) / num_probes

    if was_1d:
        result = result.squeeze(1)

    return result


def approximate_eigenvector_magnitude_field(
    L: torch.Tensor,
    theta: float,
    num_probes: int = 30,
    polynomial_order: int = 40,
    lambda_max: Optional[float] = None,
    num_eigenvectors_approx: int = 8
) -> torch.Tensor:
    """
    Approximate the weighted sum of eigenvector magnitudes using stochastic filtering.

    This approximates the explicit formula:
        spectral_field_i = sum_k weight_k * |v_k[i]|

    The approach (stochastic trace estimation):
        1. Generate random probe vectors z_j (Rademacher or Gaussian)
        2. Apply band-pass filter: y_j = g(L) @ z_j
        3. The approximation: sum_j |y_j|^2 / num_probes
           approximates sum_k weight_k * |v_k|^2

    For the magnitude field, we take square root.

    This is the core technique for eigenvalue-free spectral computation.

    Args:
        L: Sparse Laplacian matrix (n x n)
        theta: Rotation angle in [0, 1] - controls which eigenvalues to emphasize
        num_probes: Number of random probe vectors (more = better approximation)
        polynomial_order: Chebyshev polynomial degree
        lambda_max: Largest eigenvalue (estimated if not provided)
        num_eigenvectors_approx: Approximate number of eigenvectors in original formula
                                 (used to scale theta appropriately)

    Returns:
        Approximation to weighted eigenvector magnitude field (n,)
    """
    device = L.device
    n = L.shape[0]

    if lambda_max is None:
        lambda_max = estimate_lambda_max(L)

    if lambda_max < 1e-10:
        lambda_max = 1.0

    # In the explicit formula, theta * num_eigenvectors gives the eigenvector index
    # to center on. We need to convert this to eigenvalue space.
    # Eigenvalues roughly scale as k^2 / n for a grid, but we use linear scaling
    # for simplicity: target_lambda = theta * lambda_max

    target_lambda = theta * lambda_max

    # Filter width controls spectral selectivity
    # Wider = smoother transition between eigenvector weights
    filter_width = lambda_max / (num_eigenvectors_approx + 1)

    # Generate random Rademacher vectors (+/-1 entries)
    # These give unbiased trace estimates
    torch.manual_seed(12345)
    probes = torch.sign(torch.randn(n, num_probes, device=device, dtype=torch.float32))

    # Project out constant component
    probes = probes - probes.mean(dim=0, keepdim=True)

    # Apply Gaussian band-pass filter centered at target_lambda
    # This approximates: g(L) where g(lambda) = exp(-(lambda - target)^2 / (2*sigma^2))
    filtered_probes = chebyshev_filter(
        L, probes,
        center=target_lambda,
        width=filter_width,
        order=polynomial_order,
        lambda_max=lambda_max
    )

    # The filtered probes satisfy: filtered_j = sum_k g(lambda_k) * (v_k @ z_j) * v_k
    # Taking |filtered|^2 and averaging over probes:
    #   E[|filtered_i|^2] = sum_k g(lambda_k)^2 * v_k[i]^2
    #
    # If we want sum_k g(lambda_k) * |v_k[i]| instead of sum_k g^2 * v_k^2,
    # we need a different approach. Instead, use |filtered| directly
    # and leverage that it's approximately proportional to the desired quantity.

    # Method 1: Direct magnitude averaging
    # This approximates: sqrt( sum_k g^2(lambda_k) * v_k^2 )
    magnitude_field_squared = (filtered_probes ** 2).mean(dim=1)
    magnitude_field = torch.sqrt(magnitude_field_squared + 1e-10)

    return magnitude_field


def fast_spectral_etch_field(
    carrier: torch.Tensor,
    theta: float = 0.5,
    num_probes: int = 30,
    polynomial_order: int = 40,
    edge_threshold: float = 0.1
) -> torch.Tensor:
    """
    Compute spectral etch field without eigenvector decomposition.

    This is the main entry point for O(n log n) spectral field computation.
    It approximates the explicit eigenvector-based spectral field using
    polynomial filtering and stochastic trace estimation.

    The spectral field emphasizes different graph structures based on theta:
        - theta=0: Coarse structure (Fiedler, low-frequency)
        - theta=1: Fine structure (high-frequency eigenvectors)

    Args:
        carrier: 2D carrier tensor (H, W)
        theta: Rotation angle in [0, 1]
        num_probes: Number of random probes (more = better approximation)
        polynomial_order: Chebyshev polynomial degree
        edge_threshold: Carrier edge sensitivity for Laplacian weighting

    Returns:
        Spectral field as 2D tensor (H, W), normalized to [0, 1]
    """
    device = carrier.device
    height, width = carrier.shape

    # Ensure float and normalized
    carrier_tensor = carrier.to(dtype=torch.float32)
    if carrier_tensor.max() > 1.0:
        carrier_tensor = carrier_tensor / 255.0

    L = build_weighted_image_laplacian(carrier_tensor, edge_threshold)

    # Estimate lambda_max
    lambda_max = estimate_lambda_max(L)

    # Compute approximate eigenvector magnitude field
    field = approximate_eigenvector_magnitude_field(
        L, theta,
        num_probes=num_probes,
        polynomial_order=polynomial_order,
        lambda_max=lambda_max
    )

    # Reshape and normalize using torch operations
    field = field.reshape(height, width)

    f_min = field.min()
    f_max = field.max()
    if f_max > f_min:
        field = (field - f_min) / (f_max - f_min)
    else:
        field = torch.zeros_like(field)

    return field


