"""Triton kernels for spectral embedding operations.

Provides ELL-format sparse matrix-vector multiply (SpMV) for the
graph Laplacian, with PyTorch fallback for CPU / non-Triton environments.

ELL (ELLPACK) format stores each row's non-zeros at a fixed width:
- col_indices: (n, max_nnz) — column index per entry
- values: (n, max_nnz) — edge weight per entry
- degree: (n,) — diagonal (sum of edge weights per row)

For the graph Laplacian L, the SpMV computes:
    y[i] = degree[i] * x[i] - sum_j(val[i,j] * x[col[i,j]])

The ELL format is natural here because the grid Laplacian has
near-uniform row lengths (~284 neighbors for radii [1..6], with
boundary rows slightly fewer). Padding invalid entries with val=0
wastes <5% of storage for a 64x64 tile.

Architecture analog: this is the "implicit layer" matvec — the
equivalent of a single forward pass through a DEQ layer's fixed-point
iteration. Fusing the SpMV with the Lanczos reorthogonalization
(Tier C goal) would be the equivalent of fusing attention + FFN.
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _ell_spmv_kernel(
        ell_val_ptr,
        degree_ptr,
        ell_col_ptr,
        x_ptr,
        y_ptr,
        n,
        max_nnz: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Single-tile ELL SpMV: y = D*x - V*x (Laplacian matvec)."""
        pid = tl.program_id(0)
        row_start = pid * BLOCK_SIZE
        row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
        row_mask = row_offsets < n

        # Diagonal: degree[i] * x[i]
        x_diag = tl.load(x_ptr + row_offsets, mask=row_mask, other=0.0)
        deg = tl.load(degree_ptr + row_offsets, mask=row_mask, other=0.0)
        acc = deg * x_diag

        # Off-diagonal: -sum_j(val[i,j] * x[col[i,j]])
        base = row_offsets * max_nnz
        for j in range(max_nnz):
            col = tl.load(ell_col_ptr + base + j, mask=row_mask, other=0)
            val = tl.load(ell_val_ptr + base + j, mask=row_mask, other=0.0)
            x_neighbor = tl.load(x_ptr + col, mask=row_mask, other=0.0)
            acc = acc - val * x_neighbor

        tl.store(y_ptr + row_offsets, acc, mask=row_mask)

    @triton.jit
    def _batched_ell_spmv_kernel(
        ell_val_ptr,
        degree_ptr,
        ell_col_ptr,
        x_ptr,
        y_ptr,
        n,
        max_nnz: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Batched ELL SpMV: y[b] = D[b]*x[b] - V[b]*x[b].

        ell_col is shared across the batch (same sparsity structure).
        ell_val and degree differ per batch element (image-dependent weights).
        """
        batch_id = tl.program_id(1)
        pid = tl.program_id(0)
        row_start = pid * BLOCK_SIZE
        row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
        row_mask = row_offsets < n

        # Batch offsets
        b_off_n = batch_id * n
        b_off_nnz = batch_id * n * max_nnz

        # Diagonal
        x_diag = tl.load(x_ptr + b_off_n + row_offsets, mask=row_mask, other=0.0)
        deg = tl.load(degree_ptr + b_off_n + row_offsets, mask=row_mask, other=0.0)
        acc = deg * x_diag

        # Off-diagonal (ell_col is NOT batched — shared structure)
        ell_base = row_offsets * max_nnz
        for j in range(max_nnz):
            col = tl.load(ell_col_ptr + ell_base + j, mask=row_mask, other=0)
            val = tl.load(ell_val_ptr + b_off_nnz + ell_base + j, mask=row_mask, other=0.0)
            x_neighbor = tl.load(x_ptr + b_off_n + col, mask=row_mask, other=0.0)
            acc = acc - val * x_neighbor

        tl.store(y_ptr + b_off_n + row_offsets, acc, mask=row_mask)


# ---------------------------------------------------------------------------
# Python dispatch (selects Triton or PyTorch)
# ---------------------------------------------------------------------------

BLOCK_SIZE = 128  # rows per Triton program — tunable


def ell_spmv(
    ell_val: torch.Tensor,
    degree: torch.Tensor,
    ell_col: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """ELL-format SpMV for a single tile.

    Parameters
    ----------
    ell_val : (n, max_nnz) float32 — off-diagonal edge weights
    degree : (n,) float32 — diagonal (row sums)
    ell_col : (n, max_nnz) int64 — column indices
    x : (n,) float32 — input vector

    Returns
    -------
    y : (n,) float32 — L @ x
    """
    n = x.shape[0]

    if HAS_TRITON and x.is_cuda:
        max_nnz = ell_col.shape[1]
        y = torch.empty_like(x)
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _ell_spmv_kernel[grid](
            ell_val, degree, ell_col, x, y,
            n, max_nnz, BLOCK_SIZE,
        )
        return y
    else:
        # PyTorch fallback
        neighbor_vals = x[ell_col]  # (n, max_nnz)
        off_diag = (ell_val * neighbor_vals).sum(dim=-1)
        return degree * x - off_diag


def ell_spmv_batched(
    ell_val: torch.Tensor,
    degree: torch.Tensor,
    ell_col: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Batched ELL-format SpMV.

    Parameters
    ----------
    ell_val : (B, n, max_nnz) float32 — off-diagonal weights per tile
    degree : (B, n) float32 — diagonals per tile
    ell_col : (n, max_nnz) int64 — column indices (shared structure)
    x : (B, n) float32 — input vectors

    Returns
    -------
    y : (B, n) float32 — L[b] @ x[b] for each b
    """
    B, n = x.shape

    if HAS_TRITON and x.is_cuda:
        max_nnz = ell_col.shape[1]
        y = torch.empty_like(x)
        grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE, B)
        _batched_ell_spmv_kernel[grid](
            ell_val.contiguous(),
            degree.contiguous(),
            ell_col.contiguous(),
            x.contiguous(),
            y,
            n, max_nnz, BLOCK_SIZE,
        )
        return y
    else:
        # PyTorch fallback: gather + reduce
        neighbor_vals = x[:, ell_col]  # (B, n, max_nnz)
        off_diag = (ell_val * neighbor_vals).sum(dim=-1)  # (B, n)
        return degree * x - off_diag
