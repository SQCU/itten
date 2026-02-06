"""Graph-level spectral embedding as nn.Modules — Phase A foundation.

Three modules that enable all four demos:
- GraphEmbedding: eigenvectors + eigenvalues on ANY sparse Laplacian
- LocalSpectralProbe: local Fiedler + lambda_2 around query nodes
- ImageLaplacianBuilder: multi-radius weighted image Laplacian

These compose with SpectralEmbedding (which handles tiling/blending for images)
and with downstream demo modules (renderer, lattice, pathfinder).

Source: spectral_ops_fast.py, spectral_ops_fns.py, spectral_ops_fast_cuter.py
Priority: P1 (Phase A of TODO_DEMO_RECOVERY.md)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple


class GraphEmbedding(nn.Module):
    """Compute eigenvectors/eigenvalues of a sparse Laplacian via Lanczos.

    This is the general case of SpectralEmbedding. SpectralEmbedding constructs
    an image Laplacian internally; GraphEmbedding accepts any prebuilt Laplacian.

    Architecture analog: Deep Equilibrium Model (Bai et al. 2019, NeurIPS).
    The output is defined implicitly as the fixed point of L @ v = lambda * v.
    The Lanczos iteration is the fixed-point solver: it converges to the
    eigenvector that satisfies the eigenvalue equation. The tridiagonal
    projection is the "acceleration" (analogous to Anderson acceleration in DEQ).

    Returns both eigenvectors AND eigenvalues. Eigenvalues are load-bearing:
    lambda_2 (algebraic connectivity) gates lattice extrusion and measures
    local expansion. Discarding eigenvalues discards information.

    Parameters
    ----------
    num_eigenvectors : int
        Number of non-trivial eigenvectors to return. Default 4.
    lanczos_iterations : int
        Maximum Lanczos iterations (Krylov subspace dimension). Default 30.
    """

    def __init__(self, num_eigenvectors: int = 4, lanczos_iterations: int = 30):
        super().__init__()
        self.num_eigenvectors = num_eigenvectors
        self.lanczos_iterations = lanczos_iterations
        self.tol = 1e-10

    def forward(self, L: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute first k non-trivial eigenvectors and eigenvalues.

        Transcribed from spectral_ops_fast.py:lanczos_k_eigenvectors (lines 643-749).
        Preserves: mean subtraction, full reorthogonalization, GPU sparse matvec,
        deterministic seeding.

        Args:
            L: Sparse Laplacian (n, n) as torch.sparse_coo_tensor

        Returns:
            eigenvectors: (n, k) — columns are eigenvectors, Fiedler is [:, 0]
            eigenvalues: (k,) — lambda_2 is [0]
        """
        n = L.shape[0]
        device = L.device
        num_eigenvectors = self.num_eigenvectors
        k = min(self.lanczos_iterations, n - 1)

        # spectral_ops_fast.py:668-678 — initialize random vector, project out constant
        torch.manual_seed(42)
        v = torch.randn(n, device=device, dtype=torch.float32)
        v = v - v.mean()  # spectral_ops_fast.py:671 — mean subtraction (deflates trivial eigenvector)
        v_norm = torch.linalg.norm(v)
        if v_norm < self.tol:
            # spectral_ops_fast.py:674-677 — fallback for degenerate initial vector
            v = torch.ones(n, device=device, dtype=torch.float32)
            v[::2] = -1.0
            v = v - v.mean()
            v_norm = torch.linalg.norm(v)
        v = v / v_norm

        # spectral_ops_fast.py:681-685 — pre-allocate Lanczos vectors and tridiagonal entries
        V = torch.zeros((n, k + 1), device=device, dtype=torch.float32)
        V[:, 0] = v
        alphas = torch.zeros(k, device=device, dtype=torch.float32)
        betas = torch.zeros(k, device=device, dtype=torch.float32)

        actual_k = k
        iteration = 0

        # spectral_ops_fast.py:690-717 — Lanczos iteration
        while iteration < k:
            # spectral_ops_fast.py:692 — sparse matvec: w = L @ v_i
            w = torch.sparse.mm(L, V[:, iteration].unsqueeze(1)).squeeze(1)

            # spectral_ops_fast.py:694 — Rayleigh coefficient: alpha_i = v_i^T @ w
            alpha = torch.dot(V[:, iteration], w)
            alphas[iteration] = alpha

            # spectral_ops_fast.py:697 — three-term recurrence
            w = w - alpha * V[:, iteration]
            if iteration > 0:
                # spectral_ops_fast.py:699 — subtract beta_{i-1} * v_{i-1}
                w = w - betas[iteration - 1] * V[:, iteration - 1]

            # spectral_ops_fast.py:702-705 — full reorthogonalization against ALL previous vectors
            if iteration > 0:
                V_prev = V[:, :iteration + 1]
                coeffs = torch.mm(V_prev.T, w.unsqueeze(1)).squeeze(1)
                w = w - torch.mm(V_prev, coeffs.unsqueeze(1)).squeeze(1)

            # spectral_ops_fast.py:708 — mean subtraction (project out constant eigenvector)
            w = w - w.mean()

            # spectral_ops_fast.py:710-713 — check for Krylov breakdown
            beta = torch.linalg.norm(w)
            if beta < self.tol:
                actual_k = iteration + 1
                break

            # spectral_ops_fast.py:715-716
            betas[iteration] = beta
            V[:, iteration + 1] = w / beta
            iteration += 1

        # spectral_ops_fast.py:719-720 — need at least 2 iterations
        if actual_k < 2:
            return (
                torch.zeros((n, num_eigenvectors), device=device, dtype=torch.float32),
                torch.zeros(num_eigenvectors, device=device, dtype=torch.float32),
            )

        # spectral_ops_fast.py:723-727 — build tridiagonal matrix T
        T = torch.diag(alphas[:actual_k])
        if actual_k > 1:
            off_diag = torch.arange(actual_k - 1, device=device)
            T[off_diag, off_diag + 1] = betas[:actual_k - 1]
            T[off_diag + 1, off_diag] = betas[:actual_k - 1]

        # spectral_ops_fast.py:730 — solve small eigenproblem
        eigenvalues, eigenvectors_T = torch.linalg.eigh(T)

        # spectral_ops_fast.py:733-734 — select non-trivial eigenvalues (skip near-zero)
        mask = eigenvalues > 1e-6
        valid_indices = torch.where(mask)[0]

        if len(valid_indices) == 0:
            return (
                torch.zeros((n, num_eigenvectors), device=device, dtype=torch.float32),
                torch.zeros(num_eigenvectors, device=device, dtype=torch.float32),
            )

        # spectral_ops_fast.py:740-741 — take first num_eigenvectors valid ones
        take_count = min(num_eigenvectors, len(valid_indices))
        selected_indices = valid_indices[:take_count]
        selected_eigenvalues = eigenvalues[selected_indices]

        # spectral_ops_fast.py:746-747 — reconstruct eigenvectors: E = V @ Y
        Y = eigenvectors_T[:, selected_indices]
        eigenvectors_full = torch.mm(V[:, :actual_k], Y)

        # Zero-pad if we got fewer eigenvectors than requested
        if take_count < num_eigenvectors:
            padded_evecs = torch.zeros((n, num_eigenvectors), device=device, dtype=torch.float32)
            padded_evecs[:, :take_count] = eigenvectors_full
            padded_evals = torch.zeros(num_eigenvectors, device=device, dtype=torch.float32)
            padded_evals[:take_count] = selected_eigenvalues
            return padded_evecs, padded_evals

        return eigenvectors_full, selected_eigenvalues


class LocalSpectralProbe(nn.Module):
    """Compute local spectral properties around query points in a graph.

    Extracts a subgraph within `hop_radius` hops of the query node,
    builds a local Laplacian, computes Fiedler vector and lambda_2.

    This is the graph-local analog of SpectralEmbedding: it sees only
    a neighborhood, not the full graph. The pathfinder uses this to
    make locally-optimal decisions that approximate global Dijkstra
    with empirically low transport distance.

    Architecture analog: local attention window (Beltagy et al. 2020,
    Longformer) -- restricts computation to a local context, accepts
    approximation error in exchange for O(local) cost.

    Source: spectral_ops_fns.py:local_fiedler_vector (lines 169-207),
    local_expansion_estimate (lines 210-227),
    expansion_map_batched (lines 230+)

    Parameters
    ----------
    hop_radius : int
        Number of BFS hops to expand around each query node. Default 2.
    lanczos_iterations : int
        Lanczos iterations for local eigendecomposition. Default 15.
    """

    def __init__(self, hop_radius: int = 2, lanczos_iterations: int = 15):
        super().__init__()
        self.hop_radius = hop_radius
        self.lanczos_iterations = lanczos_iterations
        # Internal GraphEmbedding for the hot path (Lanczos on local subgraph)
        # Uses num_eigenvectors=1 since we only need Fiedler for local probes
        self._graph_embedding = GraphEmbedding(
            num_eigenvectors=1, lanczos_iterations=lanczos_iterations
        )

    def _expand_neighborhood(
        self, adjacency: torch.Tensor, seed_nodes: Set[int], hops: int
    ) -> Set[int]:
        """BFS expansion by hops on a sparse adjacency matrix.

        Transcribed from spectral_ops_fns.py:expand_neighborhood (lines 36-63).
        This is graph traversal, not a hot-path numerical operation.

        Args:
            adjacency: Sparse (n, n) adjacency matrix (coalesced)
            seed_nodes: Starting node indices
            hops: Number of BFS hops

        Returns:
            Set of all node indices within `hops` of any seed node
        """
        adj = adjacency.coalesce()
        indices = adj.indices()  # (2, nnz)
        rows = indices[0]
        cols = indices[1]

        # spectral_ops_fns.py:47-63 — BFS expansion
        active = set(seed_nodes)
        hop_idx = 0
        while hop_idx < hops:
            new_nodes: Set[int] = set()
            for node in list(active):
                # Find neighbors: all cols where row == node
                # spectral_ops_fns.py:55-58 — neighbor lookup
                mask = rows == node
                neighbors = cols[mask].cpu().tolist()
                for n_id in neighbors:
                    new_nodes.add(n_id)
            active.update(new_nodes)
            hop_idx += 1
        return active

    def _build_local_laplacian(
        self, adjacency: torch.Tensor, active_nodes: Set[int]
    ) -> Tuple[torch.Tensor, List[int], Dict[int, int]]:
        """Build sparse Laplacian for a subgraph defined by active_nodes.

        Transcribed from spectral_ops_fast.py:build_sparse_laplacian (lines 469-528).

        Args:
            adjacency: Full sparse (n, n) adjacency matrix
            active_nodes: Node indices to include in the subgraph

        Returns:
            L: Sparse Laplacian (m, m) where m = len(active_nodes)
            node_list: Ordered list of node indices
            node_to_idx: Mapping from global node index to local index
        """
        adj = adjacency.coalesce()
        adj_indices = adj.indices()
        adj_values = adj.values()
        device = adjacency.device

        node_list = sorted(active_nodes)
        m = len(node_list)

        # spectral_ops_fast.py:485-489 — build index mapping
        node_to_idx: Dict[int, int] = {}
        for idx, node in enumerate(node_list):
            node_to_idx[node] = idx

        # spectral_ops_fast.py:492-513 — collect edges
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        degrees = [0.0] * m

        adj_rows = adj_indices[0].cpu().tolist()
        adj_cols = adj_indices[1].cpu().tolist()
        adj_vals = adj_values.cpu().tolist()

        for r, c, v in zip(adj_rows, adj_cols, adj_vals):
            if r in node_to_idx and c in node_to_idx:
                li = node_to_idx[r]
                lj = node_to_idx[c]
                rows.append(li)
                cols.append(lj)
                vals.append(-abs(v))  # off-diagonal is negative weight
                degrees[li] += abs(v)

        # spectral_ops_fast.py:516-521 — add diagonal (degree) entries
        for i in range(m):
            rows.append(i)
            cols.append(i)
            vals.append(degrees[i])

        # spectral_ops_fast.py:524-526 — build sparse tensor
        if len(rows) == 0:
            # Degenerate: no edges
            diag = torch.arange(m, device=device)
            L = torch.sparse_coo_tensor(
                torch.stack([diag, diag]),
                torch.zeros(m, device=device, dtype=torch.float32),
                (m, m),
            ).coalesce()
            return L, node_list, node_to_idx

        indices_t = torch.tensor([rows, cols], dtype=torch.long, device=device)
        values_t = torch.tensor(vals, dtype=torch.float32, device=device)
        L = torch.sparse_coo_tensor(indices_t, values_t, (m, m)).coalesce()

        return L, node_list, node_to_idx

    def forward(
        self,
        adjacency: torch.Tensor,
        query_nodes: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """Compute local Fiedler vectors and expansion estimates around query nodes.

        Transcribed from spectral_ops_fns.py:local_fiedler_vector (lines 169-207)
        and expansion_map_batched (lines 230+).

        The key insight: expand neighborhood -> build local Laplacian -> Lanczos -> return.
        For the hot path (Lanczos on local subgraph), uses GraphEmbedding internally.

        Args:
            adjacency: Sparse (n, n) graph adjacency
            query_nodes: (q,) node indices to probe around
            coords: Optional (n, d) node coordinates (unused in computation,
                    available for downstream consumers)

        Returns:
            fiedler_maps: Dict mapping query node index -> local Fiedler tensor
                         keyed by global node IDs (as in spectral_ops_fns.py)
            expansion_estimates: (q,) lambda_2 per query node
        """
        device = adjacency.device
        q_list = query_nodes.cpu().tolist()
        num_queries = len(q_list)

        fiedler_maps: Dict[int, torch.Tensor] = {}
        expansion_estimates = torch.zeros(num_queries, device=device, dtype=torch.float32)

        # spectral_ops_fns.py:181-206 — per-query local Fiedler computation
        for qi, query_node in enumerate(q_list):
            # spectral_ops_fns.py:181 — expand neighborhood
            active_nodes = self._expand_neighborhood(
                adjacency, {query_node}, self.hop_radius
            )
            m = len(active_nodes)

            # spectral_ops_fns.py:184-191 — handle tiny subgraphs
            if m < 3:
                local_fiedler = torch.zeros(m, device=device, dtype=torch.float32)
                fiedler_maps[query_node] = local_fiedler
                expansion_estimates[qi] = 0.0
                continue

            # spectral_ops_fns.py:194 — build local Laplacian
            L_local, node_list, node_to_idx = self._build_local_laplacian(
                adjacency, active_nodes
            )

            # spectral_ops_fns.py:197 — GPU Lanczos via GraphEmbedding (hot path)
            eigenvectors, eigenvalues = self._graph_embedding(L_local)

            # spectral_ops_fns.py:200-206 — extract Fiedler vector
            fiedler_tensor = eigenvectors[:, 0]  # Fiedler is first non-trivial
            lambda2 = eigenvalues[0].item()

            # Store as tensor indexed by position in node_list
            fiedler_maps[query_node] = fiedler_tensor
            expansion_estimates[qi] = lambda2

        return fiedler_maps, expansion_estimates


class ImageLaplacianBuilder(nn.Module):
    """Build weighted image Laplacian with multi-radius connectivity.

    Constructs a graph where pixels are nodes and edges connect nearby
    pixels with weights based on color similarity. Multi-radius connectivity
    captures dither patterns that don't directly touch.

    Factored from SpectralEmbedding to allow reuse: the same Laplacian
    builder works for texture synthesis, normal mapping, and atlas operations.

    Source: spectral_ops_fast_cuter.py:_build_multiscale_laplacian (lines 104-151)

    Parameters
    ----------
    radii : list of int
        Connectivity radii for multi-scale Laplacian. Default [1,2,3,4,5,6].
    radius_weights : list of float
        Weight for each radius. Default [1.0, 0.6, 0.4, 0.3, 0.2, 0.1].
    edge_threshold : float
        Color difference threshold for edge weighting. Weights are
        exp(-|color_diff| / edge_threshold). Default 0.15.
    """

    def __init__(
        self,
        radii: Optional[List[int]] = None,
        radius_weights: Optional[List[float]] = None,
        edge_threshold: float = 0.15,
    ):
        super().__init__()
        self.edge_threshold = edge_threshold

        # register_buffer for radii and weights so they move with .to(device)
        _radii = radii if radii is not None else [1, 2, 3, 4, 5, 6]
        _radius_weights = radius_weights if radius_weights is not None else [1.0, 0.6, 0.4, 0.3, 0.2, 0.1]
        self.register_buffer("radii", torch.tensor(_radii, dtype=torch.long))
        self.register_buffer("radius_weights", torch.tensor(_radius_weights, dtype=torch.float32))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Build sparse Laplacian for the image graph.

        Transcribed from spectral_ops_fast_cuter.py:_build_multiscale_laplacian
        (lines 104-151). Preserves: multi-radius offset enumeration,
        bounds checking, color-similarity edge weighting, scatter_add degree
        computation.

        Args:
            image: (H, W) grayscale or (H, W, 3) RGB image tensor.
                   Values in [0, 255] or [0, 1]; auto-detected and normalized.

        Returns:
            Sparse COO Laplacian of shape (n, n) where n = H * W.
        """
        device = image.device

        # Normalize to [0, 1] float32
        if image.max() > 1.0:
            image = image.to(dtype=torch.float32) / 255.0
        else:
            image = image.to(dtype=torch.float32)

        # spectral_ops_fast_cuter.py:107-109 — detect RGB vs grayscale, get dimensions
        is_rgb = image.dim() == 3 and image.shape[-1] == 3
        H, W = (image.shape[0], image.shape[1]) if is_rgb else image.shape
        flat = image.reshape(-1, 3) if is_rgb else image.flatten()
        n = H * W

        # spectral_ops_fast_cuter.py:112-120 — enumerate (dy, dx, weight) offsets
        radii_list = self.radii.tolist()
        rw_list = self.radius_weights.tolist()
        offset_dy, offset_dx, offset_w = [], [], []
        for radius, rw in zip(radii_list, rw_list):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if (dx == 0 and dy == 0) or dx * dx + dy * dy > radius * radius:
                        continue
                    offset_dy.append(dy)
                    offset_dx.append(dx)
                    offset_w.append(rw)

        # spectral_ops_fast_cuter.py:122-124 — fallback if no offsets (degenerate)
        if not offset_dy:
            diag = torch.arange(n, device=device)
            return torch.sparse_coo_tensor(
                torch.stack([diag, diag]),
                torch.ones(n, device=device),
                (n, n),
            ).coalesce()

        # spectral_ops_fast_cuter.py:126-128 — tensorize offsets
        offset_dy_t = torch.tensor(offset_dy, device=device, dtype=torch.long)
        offset_dx_t = torch.tensor(offset_dx, device=device, dtype=torch.long)
        offset_w_t = torch.tensor(offset_w, device=device, dtype=torch.float32)

        # spectral_ops_fast_cuter.py:130-133 — compute destination coordinates
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.long),
            torch.arange(W, device=device, dtype=torch.long),
            indexing="ij",
        )
        yy_flat, xx_flat = yy.flatten(), xx.flatten()
        dst_y = yy_flat.unsqueeze(1) + offset_dy_t.unsqueeze(0)  # (n, num_offsets)
        dst_x = xx_flat.unsqueeze(1) + offset_dx_t.unsqueeze(0)

        # spectral_ops_fast_cuter.py:134 — bounds check
        valid = (dst_y >= 0) & (dst_y < H) & (dst_x >= 0) & (dst_x < W)

        # spectral_ops_fast_cuter.py:136-139 — gather valid indices and weights
        src_idx = torch.arange(n, device=device).unsqueeze(1).expand(-1, len(offset_dy))
        dst_idx = dst_y * W + dst_x
        src_flat = src_idx[valid]
        dst_flat = dst_idx[valid]
        weights_flat = offset_w_t.unsqueeze(0).expand(n, -1)[valid]

        # spectral_ops_fast_cuter.py:141-142 — edge weights = radius_weight * exp(-color_diff / threshold)
        if is_rgb:
            color_diff = torch.norm(flat[src_flat] - flat[dst_flat], dim=-1)
        else:
            color_diff = torch.abs(flat[src_flat] - flat[dst_flat])
        edge_w = weights_flat * torch.exp(-color_diff / self.edge_threshold)

        # spectral_ops_fast_cuter.py:143 — off-diagonal entries (negative)
        rows = src_flat
        cols = dst_flat
        vals = -edge_w

        # spectral_ops_fast_cuter.py:145-146 — degree vector via scatter_add
        degrees = torch.zeros(n, device=device)
        degrees.scatter_add_(0, rows, -vals)  # -(-edge_w) = edge_w

        # spectral_ops_fast_cuter.py:148-151 — append diagonal, build sparse COO
        diag = torch.arange(n, device=device)
        rows = torch.cat([rows, diag])
        cols = torch.cat([cols, diag])
        vals = torch.cat([vals, degrees])
        return torch.sparse_coo_tensor(
            torch.stack([rows.long(), cols.long()]), vals, (n, n)
        ).coalesce()
