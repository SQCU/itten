"""Spectral embedding as nn.Module — respecification of spectral_ops_fast_cuter.py.

This file respecifies the flat functions in spectral_ops_fast_cuter.py (264 lines)
into nn.Module form. The tensor operations are identical; the vocabulary is changed
from shader culture (flat functions, config dicts) to DNN culture (nn.Module,
register_buffer, typed forward signatures).

The _cuter file remains canonical. This file cites it as origin.

Source: spectral_ops_fast_cuter.py
Priority: P1 (see TODO_EVEN_CUTER.md)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from spectral_triton_kernels import ell_spmv_batched


class SpectralEmbedding(nn.Module):
    """Compute Fiedler vector via tiled Lanczos iteration over a multi-radius
    weighted image Laplacian.

    This is an **implicit layer**: the forward pass finds the fixed point of
    the eigenvalue equation L @ v = lambda * v for the 2nd smallest eigenvalue
    (the Fiedler value). The iteration count controls approximation quality,
    not network depth. There are no learnable parameters — the Laplacian is
    constructed from image content, and Lanczos converges to the true eigenvector.

    Architecture analog
    -------------------
    Deep Equilibrium Models (Bai et al. 2019, NeurIPS) define the output as
    an implicit fixed point z* = f(z*, x) rather than as the result of a
    finite chain of explicit layers. The Lanczos iteration is the fixed-point
    solver: it converges to the eigenvector that satisfies L @ v = lambda * v.
    The tridiagonal projection is the "acceleration" (analogous to Anderson
    acceleration in DEQ).

    AR cache semantics
    ------------------
    The Fiedler vector of a static image does not change. In two-image mode
    (cross-attention transfer), the **source** embedding can be computed once
    and cached for all AR passes. The **target** embedding must be recomputed
    after each AR mutation step, because the input has changed. This is
    analogous to KV-cache invalidation in autoregressive transformers:
    unchanged prefix tokens keep their KV entries; the new token (mutated
    target) requires fresh computation.

    Differentiable backward (future work)
    --------------------------------------
    For end-to-end training through the eigendecomposition, implicit
    differentiation via the resolvent gives:
        d(v)/d(L) = -(L - lambda * I)^{-1} (dL) v
    See Giles 2008, "Extended Collection of Matrix Derivative Results for
    Forward and Reverse Mode AD"; Wang et al. 2019, "Backpropagation-Friendly
    Eigendecomposition" (NeurIPS).

    Tiled computation
    -----------------
    The image is processed in overlapping tiles of size tile_size x tile_size.
    Each tile's Laplacian is O(tile_size^2) x O(tile_size^2), so memory is
    O(tile_size^4) per tile rather than O((H*W)^2) for the full image. Tiles
    are blended with linear taper weights in the overlap region to produce
    smooth transitions. This is a standard domain-decomposition strategy.

    Multi-radius connectivity
    -------------------------
    The Laplacian connects each pixel to neighbors at multiple radii
    (default [1,2,3,4,5,6]) with decaying weights. This is essential for
    dither-pattern-aware spectral decomposition: a checkerboard dither has
    all radius-1 neighbors different, but radius-2 neighbors identical.
    Without multi-radius connectivity, the Fiedler vector would see every
    dither pixel as a separate spectral cluster.

    Parameters
    ----------
    tile_size : int
        Side length of each square tile (pixels). Default 64.
    overlap : int
        Overlap between adjacent tiles (pixels). Default 16.
    num_eigenvectors : int
        Number of eigenvectors to compute per tile. The Fiedler vector is
        eigenvector index 0 (first non-trivial). Default 4.
    radii : list of int
        Connectivity radii for multi-scale Laplacian. Default [1,2,3,4,5,6].
    radius_weights : list of float
        Weight for each radius. Default [1.0, 0.6, 0.4, 0.3, 0.2, 0.1].
    edge_threshold : float
        Color difference threshold for edge weighting. Weights are
        exp(-|color_diff| / edge_threshold). Default 0.15.
    lanczos_iterations : int
        Maximum Lanczos iterations per tile. Default 30.
    """

    def __init__(
        self,
        tile_size: int = 64,
        overlap: int = 16,
        num_eigenvectors: int = 4,
        radii: Optional[List[int]] = None,
        radius_weights: Optional[List[float]] = None,
        edge_threshold: float = 0.15,
        lanczos_iterations: int = 30,
        output_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.tile_size = tile_size
        self.overlap = overlap
        self.num_eigenvectors = num_eigenvectors
        self.edge_threshold = edge_threshold
        self.lanczos_iterations = lanczos_iterations

        # Mixed-precision boundary: Lanczos (Zone A) always runs float32.
        # The Fiedler vector is cast to output_dtype at the very end of forward().
        # Default float32 = no-op (bit-exact with prior behavior).
        self.output_dtype = output_dtype

        # Store radii and weights as registered buffers so they move with
        # .to(device) and appear in state_dict (but are not learnable).
        _radii = radii if radii is not None else [1, 2, 3, 4, 5, 6]
        _radius_weights = radius_weights if radius_weights is not None else [1.0, 0.6, 0.4, 0.3, 0.2, 0.1]
        self.register_buffer(
            "radii", torch.tensor(_radii, dtype=torch.long)
        )
        self.register_buffer(
            "radius_weights", torch.tensor(_radius_weights, dtype=torch.float32)
        )

        # Numerical tolerance for Lanczos breakdown detection.
        # Not a learnable parameter — this is a convergence criterion.
        self.tol = 1e-10

        # Pre-compute Laplacian offsets once (eliminates ~169-iteration
        # Python loop from _build_multiscale_laplacian per tile).
        offset_dy, offset_dx, offset_w = [], [], []
        for radius, rw in zip(_radii, _radius_weights):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if (dx == 0 and dy == 0) or dx * dx + dy * dy > radius * radius:
                        continue
                    offset_dy.append(dy)
                    offset_dx.append(dx)
                    offset_w.append(rw)
        self.register_buffer(
            "_offset_dy", torch.tensor(offset_dy, dtype=torch.long)
        )
        self.register_buffer(
            "_offset_dx", torch.tensor(offset_dx, dtype=torch.long)
        )
        self.register_buffer(
            "_offset_w", torch.tensor(offset_w, dtype=torch.float32)
        )

        # Tier A: Pre-compute ELL sparsity structure for full tiles.
        # The column indices and validity mask depend only on tile dimensions
        # and offset config, NOT on image content. Only edge weights change.
        self._precompute_ell_structure()

        # Tier B: batch size for multi-tile Lanczos
        self.tile_batch_size = 8

    def _precompute_ell_structure(self):
        """Pre-compute ELL column indices and validity mask for a full tile.

        Eliminates per-tile meshgrid, broadcast, bounds check, and masking.
        The ELL structure is registered as buffers so it moves with .to(device).
        """
        H = W = self.tile_size
        n = H * W
        num_offsets = self._offset_dy.shape[0]

        if num_offsets == 0:
            self.register_buffer("_ell_col", torch.zeros(n, 0, dtype=torch.long))
            self.register_buffer("_ell_valid", torch.zeros(n, 0, dtype=torch.bool))
            self.register_buffer("_ell_base_w", torch.zeros(n, 0))
            return

        # Pixel coordinates for full tile
        yy = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1)
        xx = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1)

        # Destination coordinates for all (pixel, offset) pairs
        dst_y = yy.unsqueeze(1) + self._offset_dy.unsqueeze(0)  # (n, num_offsets)
        dst_x = xx.unsqueeze(1) + self._offset_dx.unsqueeze(0)

        # Validity mask (in-bounds)
        valid = (dst_y >= 0) & (dst_y < H) & (dst_x >= 0) & (dst_x < W)

        # Column indices (clamped for safe gather; invalid masked by val=0)
        ell_col = (dst_y * W + dst_x).clamp(0, n - 1)

        self.register_buffer("_ell_col", ell_col.long())        # (n, num_offsets)
        self.register_buffer("_ell_valid", valid)                # (n, num_offsets)
        # Store ELL base weights in output_dtype for bandwidth savings.
        # When output_dtype=bfloat16, this halves the largest static buffer.
        # Values are constants in [0.1, 1.0] — bfloat16 precision (~0.1%) is
        # far below sensitivity threshold. Upcast to float32 before Lanczos math.
        _ell_base_w = self._offset_w.unsqueeze(0).expand(n, -1).clone()
        self.register_buffer(
            "_ell_base_w",
            _ell_base_w.to(dtype=self.output_dtype)             # (n, num_offsets)
        )

    # ------------------------------------------------------------------
    # Private helpers — each maps to a function in spectral_ops_fast_cuter.py
    # ------------------------------------------------------------------

    def _build_multiscale_laplacian(
        self, tile: torch.Tensor
    ) -> torch.Tensor:
        """Multi-radius weighted Laplacian for a single tile.

        Source: spectral_ops_fast_cuter.py lines 104-151
        (_build_multiscale_laplacian)

        Constructs a sparse (n x n) Laplacian where n = tile_h * tile_w.
        Each pixel is connected to neighbors within each radius in
        self.radii, weighted by the corresponding entry in
        self.radius_weights and by exp(-|color_diff| / edge_threshold).

        Parameters
        ----------
        tile : torch.Tensor
            (H, W) grayscale or (H, W, 3) RGB tile.

        Returns
        -------
        torch.Tensor
            Sparse COO Laplacian of shape (n, n).
        """
        device = tile.device
        # --- cuter line 107-109: detect RGB vs grayscale, get dimensions ---
        is_rgb = tile.dim() == 3 and tile.shape[-1] == 3
        H, W = (tile.shape[0], tile.shape[1]) if is_rgb else tile.shape
        flat = tile.reshape(-1, 3) if is_rgb else tile.flatten()
        n = H * W

        # Use pre-computed offset tensors (moved from per-tile Python loop to __init__)
        offset_dy_t = self._offset_dy
        offset_dx_t = self._offset_dx
        offset_w_t = self._offset_w

        # --- fallback if no offsets (degenerate) ---
        if offset_dy_t.numel() == 0:
            diag = torch.arange(n, device=device)
            return torch.sparse_coo_tensor(
                torch.stack([diag, diag]),
                torch.ones(n, device=device),
                (n, n),
            ).coalesce().to_sparse_csr()

        # --- cuter lines 130-133: compute destination coordinates ---
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.long),
            torch.arange(W, device=device, dtype=torch.long),
            indexing="ij",
        )
        yy_flat, xx_flat = yy.flatten(), xx.flatten()
        dst_y = yy_flat.unsqueeze(1) + offset_dy_t.unsqueeze(0)  # (n, num_offsets)
        dst_x = xx_flat.unsqueeze(1) + offset_dx_t.unsqueeze(0)

        # --- cuter line 134: bounds check ---
        valid = (dst_y >= 0) & (dst_y < H) & (dst_x >= 0) & (dst_x < W)

        # --- cuter lines 136-139: gather valid indices and weights ---
        src_idx = torch.arange(n, device=device).unsqueeze(1).expand(-1, offset_dy_t.shape[0])
        dst_idx = dst_y * W + dst_x
        src_flat = src_idx[valid]
        dst_flat = dst_idx[valid]
        weights_flat = offset_w_t.unsqueeze(0).expand(n, -1)[valid]

        # --- cuter lines 141-142: edge weights = radius_weight * exp(-color_diff / threshold) ---
        if is_rgb:
            color_diff = torch.norm(flat[src_flat] - flat[dst_flat], dim=-1)
        else:
            color_diff = torch.abs(flat[src_flat] - flat[dst_flat])
        edge_w = weights_flat * torch.exp(-color_diff / self.edge_threshold)

        # --- cuter line 143: off-diagonal entries (negative) ---
        rows = src_flat
        cols = dst_flat
        vals = -edge_w

        # --- cuter lines 145-146: degree vector via scatter_add ---
        degrees = torch.zeros(n, device=device)
        degrees.scatter_add_(0, rows, -vals)  # -(-edge_w) = edge_w

        # --- cuter lines 148-151: append diagonal, build sparse COO ---
        diag = torch.arange(n, device=device)
        rows = torch.cat([rows, diag])
        cols = torch.cat([cols, diag])
        vals = torch.cat([vals, degrees])
        return torch.sparse_coo_tensor(
            torch.stack([rows.long(), cols.long()]), vals, (n, n)
        ).coalesce().to_sparse_csr()

    def _lanczos_tile(
        self, L: torch.Tensor, num_evecs: int, num_iter: int
    ) -> torch.Tensor:
        """Lanczos iteration for multiple eigenvectors on a single tile.

        Source: spectral_ops_fast_cuter.py lines 154-206
        (_lanczos_tile)

        This is the core implicit fixed-point solver. The Lanczos algorithm
        projects the n x n Laplacian onto a k-dimensional Krylov subspace,
        producing a tridiagonal matrix T whose eigenvalues approximate those
        of L. The Ritz vectors (back-projected eigenvectors of T) approximate
        the true eigenvectors.

        Key numerical properties preserved:
        - **Mean subtraction** (cuter line 179): after each matvec, the mean
          of w is subtracted. This deflates the trivial eigenvector (constant
          vector, eigenvalue 0) without explicitly computing it. Without this,
          the Lanczos basis would waste dimensions approximating lambda_1 = 0.
        - **Full reorthogonalization** (cuter lines 176-178): w is
          orthogonalized against ALL previous Lanczos vectors, not just the
          last two. This prevents ghost eigenvalues from loss of orthogonality
          in finite precision. Essential when lambda_2 is small (near-connected
          graph).
        - **Deterministic seeding** (cuter line 161): manual_seed(42 + n % 10000)
          for reproducibility.

        Parameters
        ----------
        L : torch.Tensor
            Sparse COO Laplacian, shape (n, n).
        num_evecs : int
            Number of non-trivial eigenvectors to return.
        num_iter : int
            Maximum Lanczos iterations (Krylov subspace dimension).

        Returns
        -------
        torch.Tensor
            Shape (n, num_evecs). Columns are eigenvectors corresponding to
            the num_evecs smallest non-trivial eigenvalues.
        """
        n, device = L.shape[0], L.device
        k = min(num_iter, n - 1)

        # --- cuter lines 158-159: early exit for tiny tiles ---
        if n < 3 or k < 2:
            return torch.zeros((n, num_evecs), device=device, dtype=torch.float32)

        # --- cuter lines 161-163: deterministic initial vector, mean-subtracted ---
        torch.manual_seed(42 + n % 10000)
        v = torch.randn(n, device=device, dtype=torch.float32)
        v = (v - v.mean()) / torch.linalg.norm(v - v.mean()).clamp(min=self.tol)

        # --- cuter lines 165-167: allocate Krylov basis and tridiagonal entries ---
        V = torch.zeros((n, k + 1), device=device, dtype=torch.float32)
        V[:, 0] = v
        alphas = torch.zeros(k, device=device, dtype=torch.float32)
        betas = torch.zeros(k, device=device, dtype=torch.float32)

        # --- Fused Lanczos iteration ---
        # Key optimization: the 3-term recurrence (w -= alpha*v_i, w -= beta*v_{i-1})
        # is absorbed into the full reorthogonalization. Since V[:, :i+1] contains
        # both v_i and v_{i-1}, projecting w onto V and subtracting removes those
        # components automatically. Alpha is extracted from the projection coefficients.
        # This saves 2 kernel launches per iteration (60 total for k=30).
        actual_k = k
        for i in range(k):
            # Sparse matvec: w = L @ v_i
            w = torch.sparse.mm(L, V[:, i].unsqueeze(1)).squeeze(1)

            # Full reorthogonalization subsumes 3-term recurrence
            V_used = V[:, : i + 1]
            coeffs = torch.mm(V_used.T, w.unsqueeze(1)).squeeze(1)
            alphas[i] = coeffs[i]  # alpha_i = v_i^T @ w (extracted from projection)
            w = w - torch.mm(V_used, coeffs.unsqueeze(1)).squeeze(1)

            # Mean subtraction: deflates trivial eigenvector (constant, lambda=0)
            w = w - w.mean()

            # Check for breakdown (Krylov subspace exhausted)
            beta = torch.linalg.norm(w)
            if beta < self.tol:
                actual_k = i + 1
                break
            betas[i] = beta
            V[:, i + 1] = w / beta

        # --- cuter lines 187-188: need at least 2 iterations for a non-trivial result ---
        if actual_k < 2:
            return torch.zeros((n, num_evecs), device=device, dtype=torch.float32)

        # --- cuter lines 190-194: build tridiagonal matrix T ---
        T = torch.diag(alphas[:actual_k])
        if actual_k > 1:
            off = torch.arange(actual_k - 1, device=device)
            T[off, off + 1] = betas[: actual_k - 1]
            T[off + 1, off] = betas[: actual_k - 1]

        # --- cuter lines 196-200: eigendecompose T, skip trivial eigenvalues ---
        eigenvalues, eigenvectors = torch.linalg.eigh(T)
        mask = eigenvalues > 1e-6
        valid_idx = torch.where(mask)[0]
        if len(valid_idx) == 0:
            return torch.zeros((n, num_evecs), device=device, dtype=torch.float32)

        # --- cuter lines 202-206: back-project Ritz vectors, zero-pad if needed ---
        take = min(num_evecs, len(valid_idx))
        evecs = torch.mm(V[:, :actual_k], eigenvectors[:, valid_idx[:take]])
        result = torch.zeros((n, num_evecs), device=device, dtype=torch.float32)
        result[:, :take] = evecs
        return result

    def _lanczos_tile_with_eigenvalues(
        self, L: torch.Tensor, num_evecs: int, num_iter: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lanczos iteration returning eigenvectors AND eigenvalues.

        Identical to _lanczos_tile except it also returns the eigenvalues
        corresponding to each eigenvector. Used by forward_with_eigenvalues.

        The Lanczos iteration itself is identical — this method differs only
        in the return path: it extracts eigenvalues from the tridiagonal
        eigenproblem alongside the Ritz vectors.

        Parameters
        ----------
        L : torch.Tensor
            Sparse COO Laplacian, shape (n, n).
        num_evecs : int
            Number of non-trivial eigenvectors to return.
        num_iter : int
            Maximum Lanczos iterations (Krylov subspace dimension).

        Returns
        -------
        evecs : torch.Tensor
            Shape (n, num_evecs). Columns are eigenvectors.
        evals : torch.Tensor
            Shape (num_evecs,). Corresponding eigenvalues.
        """
        n, device = L.shape[0], L.device
        k = min(num_iter, n - 1)

        if n < 3 or k < 2:
            return (
                torch.zeros((n, num_evecs), device=device, dtype=torch.float32),
                torch.zeros(num_evecs, device=device, dtype=torch.float32),
            )

        # Deterministic initial vector, mean-subtracted (identical to _lanczos_tile)
        torch.manual_seed(42 + n % 10000)
        v = torch.randn(n, device=device, dtype=torch.float32)
        v = (v - v.mean()) / torch.linalg.norm(v - v.mean()).clamp(min=self.tol)

        V = torch.zeros((n, k + 1), device=device, dtype=torch.float32)
        V[:, 0] = v
        alphas = torch.zeros(k, device=device, dtype=torch.float32)
        betas = torch.zeros(k, device=device, dtype=torch.float32)

        # Fused Lanczos iteration (same optimization as _lanczos_tile)
        actual_k = k
        for i in range(k):
            w = torch.sparse.mm(L, V[:, i].unsqueeze(1)).squeeze(1)
            V_used = V[:, : i + 1]
            coeffs = torch.mm(V_used.T, w.unsqueeze(1)).squeeze(1)
            alphas[i] = coeffs[i]
            w = w - torch.mm(V_used, coeffs.unsqueeze(1)).squeeze(1)
            w = w - w.mean()
            beta = torch.linalg.norm(w)
            if beta < self.tol:
                actual_k = i + 1
                break
            betas[i] = beta
            V[:, i + 1] = w / beta

        if actual_k < 2:
            return (
                torch.zeros((n, num_evecs), device=device, dtype=torch.float32),
                torch.zeros(num_evecs, device=device, dtype=torch.float32),
            )

        T = torch.diag(alphas[:actual_k])
        if actual_k > 1:
            off = torch.arange(actual_k - 1, device=device)
            T[off, off + 1] = betas[: actual_k - 1]
            T[off + 1, off] = betas[: actual_k - 1]

        eigenvalues, eigenvectors = torch.linalg.eigh(T)
        mask = eigenvalues > 1e-6
        valid_idx = torch.where(mask)[0]
        if len(valid_idx) == 0:
            return (
                torch.zeros((n, num_evecs), device=device, dtype=torch.float32),
                torch.zeros(num_evecs, device=device, dtype=torch.float32),
            )

        take = min(num_evecs, len(valid_idx))
        evecs = torch.mm(V[:, :actual_k], eigenvectors[:, valid_idx[:take]])

        result_evecs = torch.zeros((n, num_evecs), device=device, dtype=torch.float32)
        result_evecs[:, :take] = evecs

        result_evals = torch.zeros(num_evecs, device=device, dtype=torch.float32)
        result_evals[:take] = eigenvalues[valid_idx[:take]]

        return result_evecs, result_evals

    # ------------------------------------------------------------------
    # Tier A+B+C: Batched ELL Laplacian + Lanczos
    # ------------------------------------------------------------------

    def _build_laplacian_ell_batched(
        self, tiles: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched Laplacian construction using pre-computed ELL structure.

        Tier A optimization: the sparsity pattern is pre-computed in __init__.
        Only the color-dependent edge weights are computed per tile.

        Parameters
        ----------
        tiles : list of torch.Tensor
            B tiles, each (tile_size, tile_size) or (tile_size, tile_size, 3).
            All must be the same (full) tile size.

        Returns
        -------
        ell_val : (B, n, max_nnz) float32 — off-diagonal edge weights
        degree : (B, n) float32 — diagonal (row sums)
        """
        B = len(tiles)
        device = tiles[0].device
        is_rgb = tiles[0].dim() == 3 and tiles[0].shape[-1] == 3
        C = 3 if is_rgb else 1
        n = self.tile_size * self.tile_size

        # Stack tiles into batch
        flat_batch = torch.stack([
            t.reshape(-1, C) for t in tiles
        ])  # (B, n, C)

        # Gather neighbor colors using pre-computed column indices
        dst_colors = flat_batch[:, self._ell_col]       # (B, n, max_nnz, C)
        src_colors = flat_batch.unsqueeze(2)             # (B, n, 1, C)

        # Color difference
        if is_rgb:
            color_diff = torch.norm(src_colors - dst_colors, dim=-1)
        else:
            color_diff = (src_colors - dst_colors).abs().squeeze(-1)

        # Edge weights = base_weight * exp(-color_diff / threshold)
        # Upcast _ell_base_w to float32 for Lanczos Zone A math.
        # When output_dtype=bfloat16, base_w is stored compressed; upcast here
        # ensures Lanczos sees float32 (no precision change in computation).
        base_w = self._ell_base_w.float()  # (n, max_nnz) — upcast if bfloat16
        ell_val = base_w.unsqueeze(0) * torch.exp(
            -color_diff / self.edge_threshold
        )
        ell_val = ell_val * self._ell_valid.unsqueeze(0).float()  # zero invalid

        # Degree = sum of edge weights per row
        degree = ell_val.sum(dim=-1)  # (B, n)

        return ell_val, degree

    def _lanczos_batched(
        self,
        ell_val: torch.Tensor,
        degree: torch.Tensor,
        num_evecs: int,
        num_iter: int,
    ) -> List[torch.Tensor]:
        """Batched Lanczos iteration using ELL SpMV.

        Tier B+C optimization: processes multiple tiles simultaneously.
        Uses Triton ELL SpMV kernel (Tier C) and torch.bmm for batched
        reorthogonalization.

        Parameters
        ----------
        ell_val : (B, n, max_nnz) float32 — off-diagonal edge weights
        degree : (B, n) float32 — diagonal values
        num_evecs : int — eigenvectors to return per tile
        num_iter : int — Lanczos iterations

        Returns
        -------
        list of torch.Tensor
            B tensors each of shape (n, num_evecs).
        """
        B, n, _ = ell_val.shape
        device = ell_val.device
        k = min(num_iter, n - 1)

        if n < 3 or k < 2:
            return [torch.zeros((n, num_evecs), device=device) for _ in range(B)]

        # Deterministic init (same for all tiles — same n → same seed)
        torch.manual_seed(42 + n % 10000)
        v = torch.randn(n, device=device, dtype=torch.float32)
        v = (v - v.mean()) / torch.linalg.norm(v - v.mean()).clamp(min=self.tol)

        V = torch.zeros(B, n, k + 1, device=device, dtype=torch.float32)
        V[:, :, 0] = v.unsqueeze(0)
        alphas = torch.zeros(B, k, device=device, dtype=torch.float32)
        betas = torch.zeros(B, k, device=device, dtype=torch.float32)

        for i in range(k):
            # Batched ELL SpMV: w[b] = L[b] @ V[b, :, i]
            w = ell_spmv_batched(
                ell_val, degree, self._ell_col, V[:, :, i]
            )  # (B, n)

            # Batched reorthogonalization (fused: subsumes 3-term recurrence)
            # Element-wise + sum avoids CUBLAS strided-batch shape limits
            V_used = V[:, :, : i + 1]                                    # (B, n, i+1)
            coeffs = (V_used * w.unsqueeze(2)).sum(dim=1)                # (B, i+1)
            alphas[:, i] = coeffs[:, i]
            w = w - (V_used * coeffs.unsqueeze(1)).sum(dim=2)            # (B, n)

            # Mean subtraction (deflate trivial eigenvector)
            w = w - w.mean(dim=1, keepdim=True)

            # Norms (skip early termination in batched mode — rare and adds complexity)
            beta = torch.linalg.norm(w, dim=1)  # (B,)
            betas[:, i] = beta
            V[:, :, i + 1] = w / beta.unsqueeze(1).clamp(min=self.tol)

        # Batched tridiagonal eigensolve
        T = torch.zeros(B, k, k, device=device, dtype=torch.float32)
        diag_idx = torch.arange(k, device=device)
        T[:, diag_idx, diag_idx] = alphas
        if k > 1:
            off = torch.arange(k - 1, device=device)
            T[:, off, off + 1] = betas[:, : k - 1]
            T[:, off + 1, off] = betas[:, : k - 1]

        eigenvalues, eigenvectors = torch.linalg.eigh(T)  # batched

        # Back-project Ritz vectors (per-tile: valid eigenvalues may differ)
        results = []
        for b in range(B):
            mask = eigenvalues[b] > 1e-6
            valid_idx = torch.where(mask)[0]
            if len(valid_idx) == 0:
                results.append(
                    torch.zeros(n, num_evecs, device=device, dtype=torch.float32)
                )
                continue
            take = min(num_evecs, len(valid_idx))
            evecs = torch.mm(V[b, :, :k], eigenvectors[b, :, valid_idx[:take]])
            result = torch.zeros(n, num_evecs, device=device, dtype=torch.float32)
            result[:, :take] = evecs
            results.append(result)

        return results

    def _process_tile_batch(
        self,
        tiles: List[torch.Tensor],
        num_evecs: int,
        with_eigenvalues: bool = False,
    ) -> List:
        """Process a batch of full-size tiles through ELL Laplacian + batched Lanczos.

        Combines Tier A (pre-computed ELL structure), Tier B (batched processing),
        and Tier C (Triton ELL SpMV) into one pipeline.

        Parameters
        ----------
        tiles : list of torch.Tensor
            All tiles must be full size (tile_size x tile_size).
        num_evecs : int
            Number of eigenvectors per tile.
        with_eigenvalues : bool
            If True, also return eigenvalues per tile.

        Returns
        -------
        list of torch.Tensor (or list of tuples if with_eigenvalues)
            Each element is (n, num_evecs) eigenvector tensor.
        """
        # Build batched Laplacians using pre-computed ELL structure (Tier A)
        ell_val, degree = self._build_laplacian_ell_batched(tiles)

        if not with_eigenvalues:
            # Batched Lanczos (Tier B + C)
            return self._lanczos_batched(
                ell_val, degree, num_evecs, self.lanczos_iterations
            )
        else:
            # For eigenvalue variant, fall back to per-tile (eigenvalue
            # extraction is tile-specific and rarely on the hot path)
            results = []
            for b in range(len(tiles)):
                L = self._build_multiscale_laplacian(tiles[b])
                results.append(
                    self._lanczos_tile_with_eigenvalues(
                        L, num_evecs, self.lanczos_iterations
                    )
                )
            return results

    @staticmethod
    def _blend_weights_1d(
        length: int, overlap: int, device: torch.device
    ) -> torch.Tensor:
        """1D linear taper blend weights for tile overlap.

        Source: spectral_ops_fast_cuter.py lines 209-215
        (_blend_weights_1d)

        Produces a 1D weight vector of the given length with linear ramps in
        the first and last `overlap` elements. The 2D blend mask is the outer
        product of the horizontal and vertical 1D weights.

        Parameters
        ----------
        length : int
            Tile dimension (height or width).
        overlap : int
            Number of pixels in the taper region.
        device : torch.device
            Target device.

        Returns
        -------
        torch.Tensor
            Shape (length,), values in (0, 1].
        """
        # --- cuter lines 211-214 ---
        w = torch.ones(length, device=device, dtype=torch.float32)
        if overlap > 0 and length > 2 * overlap:
            taper = torch.arange(1, overlap + 1, device=device, dtype=torch.float32) / (
                overlap + 1
            )
            w[:overlap] = taper
            w[-overlap:] = taper.flip(0)
        return w

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Compute the Fiedler vector (2nd smallest Laplacian eigenvector) of
        the input image via tiled Lanczos iteration.

        Source: spectral_ops_fast_cuter.py lines 218-263
        (compute_local_eigenvectors_tiled_dither)

        Pipeline:
        1. Normalize image to [0, 1] float32.
        2. Compute tile grid with overlap.
        3. For each tile:
           a. Build multi-radius weighted Laplacian (_build_multiscale_laplacian)
           b. Run Lanczos iteration (_lanczos_tile)
           c. Accumulate blended eigenvectors
        4. Normalize by accumulated blend weights.
        5. Return eigenvector index 0 (the Fiedler vector) as (H, W).

        Parameters
        ----------
        image : torch.Tensor
            Input image. Shape (H, W) for grayscale or (H, W, 3) for RGB.
            Values in [0, 255] or [0, 1]; auto-detected and normalized.

        Returns
        -------
        torch.Tensor
            Fiedler vector, shape (H, W). Values are continuous; sign is
            arbitrary (eigenvectors are defined up to sign).
        """
        device = image.device

        # --- cuter lines 227-228: detect shape ---
        is_rgb = image.dim() == 3 and image.shape[-1] == 3
        H, W = (image.shape[0], image.shape[1]) if is_rgb else image.shape

        # --- cuter lines 230-233: normalize to [0,1] float32 ---
        if image.max() > 1.0:
            image = image.to(dtype=torch.float32) / 255.0
        else:
            image = image.to(dtype=torch.float32)

        # --- cuter lines 235-236: allocate accumulation buffers ---
        num_eigenvectors = self.num_eigenvectors
        result = torch.zeros(
            (H, W, num_eigenvectors), device=device, dtype=torch.float32
        )
        weights = torch.zeros((H, W), device=device, dtype=torch.float32)

        # --- cuter lines 238-245: compute tile grid ---
        step = self.tile_size - self.overlap
        y_starts = list(range(0, max(1, H - self.tile_size + 1), step))
        if not y_starts or y_starts[-1] + self.tile_size < H:
            y_starts.append(max(0, H - self.tile_size))
        x_starts = list(range(0, max(1, W - self.tile_size + 1), step))
        if not x_starts or x_starts[-1] + self.tile_size < W:
            x_starts.append(max(0, W - self.tile_size))
        y_starts = sorted(set(y_starts))
        x_starts = sorted(set(x_starts))

        # --- Tier B: collect tiles, batch full-size, fallback for edge tiles ---
        ts = self.tile_size
        full_tiles = []       # (tile_tensor, y, x) for full-size tiles
        edge_tiles = []       # (tile_tensor, y, x, th, tw) for non-standard tiles

        for y in y_starts:
            for x in x_starts:
                y_end = min(y + ts, H)
                x_end = min(x + ts, W)
                tile = image[y:y_end, x:x_end]
                th, tw = (tile.shape[0], tile.shape[1]) if is_rgb else tile.shape
                if th < 4 or tw < 4:
                    continue
                if th == ts and tw == ts:
                    full_tiles.append((tile, y, x))
                else:
                    edge_tiles.append((tile, y, x, th, tw))

        # Process full-size tiles in batches (Tier A+B+C fast path)
        BS = self.tile_batch_size
        for batch_start in range(0, len(full_tiles), BS):
            batch = full_tiles[batch_start : batch_start + BS]
            tiles_list = [t for t, _, _ in batch]
            evecs_list = self._process_tile_batch(tiles_list, num_eigenvectors)

            for (_, y, x), evecs_flat in zip(batch, evecs_list):
                evecs = evecs_flat.reshape(ts, ts, num_eigenvectors)
                blend = torch.outer(
                    self._blend_weights_1d(ts, self.overlap, device),
                    self._blend_weights_1d(ts, self.overlap, device),
                )
                result[y : y + ts, x : x + ts] += evecs * blend.unsqueeze(-1)
                weights[y : y + ts, x : x + ts] += blend

        # Process edge tiles sequentially (rare — only image boundaries)
        for tile, y, x, th, tw in edge_tiles:
            L = self._build_multiscale_laplacian(tile)
            evecs = self._lanczos_tile(
                L, num_eigenvectors, self.lanczos_iterations
            ).reshape(th, tw, num_eigenvectors)
            blend = torch.outer(
                self._blend_weights_1d(th, self.overlap, device),
                self._blend_weights_1d(tw, self.overlap, device),
            )
            y_end = min(y + ts, H)
            x_end = min(x + ts, W)
            result[y:y_end, x:x_end] += evecs * blend.unsqueeze(-1)
            weights[y:y_end, x:x_end] += blend

        # --- cuter line 263: normalize by blend weights ---
        all_evecs = result / weights.unsqueeze(-1).clamp(min=self.tol)

        # Return the Fiedler vector (eigenvector 0 = first non-trivial)
        # Zone A -> Zone B boundary: cast from float32 to output_dtype.
        # When output_dtype=bfloat16, this is where precision crosses the boundary.
        # When output_dtype=float32, this is a no-op.
        return all_evecs[:, :, 0].to(dtype=self.output_dtype)

    def forward_all(self, image: torch.Tensor) -> torch.Tensor:
        """Compute all num_eigenvectors eigenvectors, not just the Fiedler.

        This is the full output of compute_local_eigenvectors_tiled_dither.
        Useful when downstream modules need higher-order spectral features.

        Parameters
        ----------
        image : torch.Tensor
            Input image. Shape (H, W) or (H, W, 3).

        Returns
        -------
        torch.Tensor
            Shape (H, W, num_eigenvectors). Column 0 is the Fiedler vector.
        """
        device = image.device
        is_rgb = image.dim() == 3 and image.shape[-1] == 3
        H, W = (image.shape[0], image.shape[1]) if is_rgb else image.shape

        if image.max() > 1.0:
            image = image.to(dtype=torch.float32) / 255.0
        else:
            image = image.to(dtype=torch.float32)

        num_eigenvectors = self.num_eigenvectors
        result = torch.zeros(
            (H, W, num_eigenvectors), device=device, dtype=torch.float32
        )
        weights = torch.zeros((H, W), device=device, dtype=torch.float32)

        step = self.tile_size - self.overlap
        y_starts = list(range(0, max(1, H - self.tile_size + 1), step))
        if not y_starts or y_starts[-1] + self.tile_size < H:
            y_starts.append(max(0, H - self.tile_size))
        x_starts = list(range(0, max(1, W - self.tile_size + 1), step))
        if not x_starts or x_starts[-1] + self.tile_size < W:
            x_starts.append(max(0, W - self.tile_size))
        y_starts = sorted(set(y_starts))
        x_starts = sorted(set(x_starts))

        # Batched tile processing (same strategy as forward)
        ts = self.tile_size
        full_tiles = []
        edge_tiles = []
        for y in y_starts:
            for x in x_starts:
                y_end = min(y + ts, H)
                x_end = min(x + ts, W)
                tile = image[y:y_end, x:x_end]
                th, tw = (tile.shape[0], tile.shape[1]) if is_rgb else tile.shape
                if th < 4 or tw < 4:
                    continue
                if th == ts and tw == ts:
                    full_tiles.append((tile, y, x))
                else:
                    edge_tiles.append((tile, y, x, th, tw))

        BS = self.tile_batch_size
        for batch_start in range(0, len(full_tiles), BS):
            batch = full_tiles[batch_start : batch_start + BS]
            tiles_list = [t for t, _, _ in batch]
            evecs_list = self._process_tile_batch(tiles_list, num_eigenvectors)
            for (_, y, x), evecs_flat in zip(batch, evecs_list):
                evecs = evecs_flat.reshape(ts, ts, num_eigenvectors)
                blend = torch.outer(
                    self._blend_weights_1d(ts, self.overlap, device),
                    self._blend_weights_1d(ts, self.overlap, device),
                )
                result[y : y + ts, x : x + ts] += evecs * blend.unsqueeze(-1)
                weights[y : y + ts, x : x + ts] += blend

        for tile, y, x, th, tw in edge_tiles:
            L = self._build_multiscale_laplacian(tile)
            evecs = self._lanczos_tile(
                L, num_eigenvectors, self.lanczos_iterations
            ).reshape(th, tw, num_eigenvectors)
            blend = torch.outer(
                self._blend_weights_1d(th, self.overlap, device),
                self._blend_weights_1d(tw, self.overlap, device),
            )
            y_end = min(y + ts, H)
            x_end = min(x + ts, W)
            result[y:y_end, x:x_end] += evecs * blend.unsqueeze(-1)
            weights[y:y_end, x:x_end] += blend

        return (result / weights.unsqueeze(-1).clamp(min=self.tol)).to(dtype=self.output_dtype)

    def forward_with_eigenvalues(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (H, W) Fiedler vector AND (H, W) tiled eigenvalue map.

        Same pipeline as forward() but uses _lanczos_tile_with_eigenvalues
        to also extract eigenvalues per tile. The eigenvalue map tiles the
        first non-trivial eigenvalue (lambda_2) across each tile's spatial
        extent, blended the same way as the eigenvectors.

        This is needed by the lattice demo: lambda_2 (algebraic connectivity)
        gates extrusion depth. Previously eigenvalues were computed then
        discarded; this method preserves them.

        The existing forward() and forward_all() signatures are NOT modified.

        Parameters
        ----------
        image : torch.Tensor
            Input image. Shape (H, W) or (H, W, 3).

        Returns
        -------
        fiedler : torch.Tensor
            Fiedler vector, shape (H, W).
        eigenvalue_map : torch.Tensor
            Tiled eigenvalue map, shape (H, W). Each pixel gets the lambda_2
            of its tile, blended across overlapping tiles.
        """
        device = image.device

        is_rgb = image.dim() == 3 and image.shape[-1] == 3
        H, W = (image.shape[0], image.shape[1]) if is_rgb else image.shape

        if image.max() > 1.0:
            image = image.to(dtype=torch.float32) / 255.0
        else:
            image = image.to(dtype=torch.float32)

        num_eigenvectors = self.num_eigenvectors
        result = torch.zeros(
            (H, W, num_eigenvectors), device=device, dtype=torch.float32
        )
        eigenvalue_accum = torch.zeros((H, W), device=device, dtype=torch.float32)
        weights = torch.zeros((H, W), device=device, dtype=torch.float32)

        step = self.tile_size - self.overlap
        y_starts = list(range(0, max(1, H - self.tile_size + 1), step))
        if not y_starts or y_starts[-1] + self.tile_size < H:
            y_starts.append(max(0, H - self.tile_size))
        x_starts = list(range(0, max(1, W - self.tile_size + 1), step))
        if not x_starts or x_starts[-1] + self.tile_size < W:
            x_starts.append(max(0, W - self.tile_size))
        y_starts = sorted(set(y_starts))
        x_starts = sorted(set(x_starts))

        for y in y_starts:
            for x in x_starts:
                y_end = min(y + self.tile_size, H)
                x_end = min(x + self.tile_size, W)
                tile = image[y:y_end, x:x_end]
                th, tw = (tile.shape[0], tile.shape[1]) if is_rgb else tile.shape
                if th < 4 or tw < 4:
                    continue

                L = self._build_multiscale_laplacian(tile)
                evecs, evals = self._lanczos_tile_with_eigenvalues(
                    L, num_eigenvectors, self.lanczos_iterations
                )
                evecs = evecs.reshape(th, tw, num_eigenvectors)

                blend = torch.outer(
                    self._blend_weights_1d(th, self.overlap, device),
                    self._blend_weights_1d(tw, self.overlap, device),
                )

                for k in range(num_eigenvectors):
                    result[y:y_end, x:x_end, k] += evecs[:, :, k] * blend
                # Tile lambda_2 (first eigenvalue = index 0) across the tile
                eigenvalue_accum[y:y_end, x:x_end] += evals[0] * blend
                weights[y:y_end, x:x_end] += blend

        all_evecs = result / weights.unsqueeze(-1).clamp(min=self.tol)
        eigenvalue_map = eigenvalue_accum / weights.clamp(min=self.tol)

        return all_evecs[:, :, 0].to(dtype=self.output_dtype), eigenvalue_map.to(dtype=self.output_dtype)


class SpectralShaderAR(nn.Module):
    """Autoregressive spectral shader wrapper.

    Makes the distinction between **depth** and **AR iteration**
    architecturally explicit in the module graph.

    Depth vs AR
    -----------
    - **Depth** = more layers processing the **same** input. The spectral
      embedding (Fiedler vector) is computed once and shared across all
      layers. This is valid because the input has not changed, so the
      Laplacian — and therefore its eigenvectors — remain the same.
      Analog: a multi-layer transformer where all layers share the same
      KV cache computed from the input.

    - **AR iteration** = running the entire shader on the **mutated** output
      of the previous pass. The spectral embedding **must** be recomputed
      because the input image has changed. The Laplacian is a function of
      pixel values; when pixels change, the graph changes, and the Fiedler
      vector changes with it.
      Analog: autoregressive generation where each new token invalidates
      the cached attention for subsequent positions.

    Source KV cache
    ---------------
    In two-image mode (cross-attention transfer), the **source** image is
    never mutated. Its Fiedler vector can be computed once and passed as
    ``source_fiedler`` to avoid redundant computation across AR passes.
    The target's Fiedler vector is always recomputed.

    This is the same pattern as encoder-decoder attention in seq2seq
    transformers: the encoder KV cache (source) is computed once; the
    decoder KV cache (target) grows with each autoregressive step.

    Parameters
    ----------
    shader_block : nn.Module
        A single shader pass module (e.g., SpectralShaderBlock from
        spectral_shader_layers.py). Must accept (image, fiedler) and
        return a modified image.
    embedding_layer : SpectralEmbedding
        The spectral embedding module that computes the Fiedler vector.
    """

    def __init__(
        self,
        shader_block: nn.Module,
        embedding_layer: SpectralEmbedding,
    ):
        super().__init__()
        self.shader_block = shader_block
        self.embedding_layer = embedding_layer

    def forward(
        self,
        image: torch.Tensor,
        n_passes: int = 1,
        source: Optional[torch.Tensor] = None,
        source_fiedler: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run n_passes of autoregressive shader application.

        Each pass:
        1. Recompute Fiedler vector of current (mutated) image.
        2. Apply shader_block(current_image, fiedler) -> new image.
        3. Repeat with new image as input.

        The source Fiedler (if provided) is computed once and cached — it
        does not change because the source image is never mutated.

        Parameters
        ----------
        image : torch.Tensor
            Input image, shape (H, W) or (H, W, 3).
        n_passes : int
            Number of AR iterations. Default 1.
        source : torch.Tensor, optional
            Source image for cross-attention mode.
        source_fiedler : torch.Tensor, optional
            Pre-computed Fiedler vector of source. If source is provided
            but source_fiedler is not, it will be computed once here.

        Returns
        -------
        torch.Tensor
            Shader output after n_passes AR iterations.
        """
        # Source embedding: compute once, cache for all AR passes.
        # Source image is never mutated -> its Fiedler is invariant.
        if source is not None and source_fiedler is None:
            source_fiedler = self.embedding_layer(source)

        current = image
        for _pass_idx in range(n_passes):
            # RECOMPUTE: the target embedding must be recomputed because
            # current was mutated by the previous pass. This is the
            # KV-cache invalidation analog.
            fiedler = self.embedding_layer(current)

            # Apply the shader block. The block signature depends on
            # whether we're in single-image or cross-attention mode.
            if source is not None:
                # Cross-attention mode: pass source info to shader_block.
                # The shader_block is responsible for handling the source
                # arguments (e.g., SpectralCrossAttentionBlock).
                current = self.shader_block(
                    current, fiedler, source=source, source_fiedler=source_fiedler
                )
            else:
                # Single-image mode.
                current = self.shader_block(current, fiedler)

        return current
