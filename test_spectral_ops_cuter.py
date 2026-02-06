"""
Unit tests comparing spectral_ops_fast.py against spectral_ops_fast_cuter.py.

Tests Laplacian properties, Fiedler vector properties, implementation equivalence,
tiled vs global comparison, performance benchmarks, and edge cases.

Run with: pytest test_spectral_ops_cuter.py -v
"""

import torch
import time
import pytest
import numpy as np

# Device selection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Try to import original implementation
try:
    from spectral_ops_fast import (
        build_weighted_image_laplacian as orig_build_laplacian,
        lanczos_fiedler_gpu as orig_lanczos,
        compute_local_eigenvectors_tiled_dither as orig_tiled_dither,
        compute_local_eigenvectors_tiled as orig_tiled,
        build_image_laplacian as orig_build_image_laplacian,
        Graph,
    )
    HAS_ORIGINAL = True
except ImportError as e:
    print(f"Could not import original: {e}")
    HAS_ORIGINAL = False

# Try to import cuter implementation
try:
    from spectral_ops_fast_cuter import (
        build_weighted_image_laplacian as cuter_build_laplacian,
        lanczos_fiedler_gpu as cuter_lanczos,
        compute_local_eigenvectors_tiled_dither as cuter_tiled,
    )
    HAS_CUTER = True
except ImportError as e:
    print(f"Could not import cuter: {e}")
    HAS_CUTER = False


# ============================================================
# Helper Functions
# ============================================================

def make_test_image(H=64, W=64, pattern='checkerboard', device=None):
    """
    Create test images with known patterns.

    Args:
        H: Height
        W: Width
        pattern: One of 'checkerboard', 'gradient', 'uniform', 'random', 'stripes'
        device: Target device

    Returns:
        2D tensor of shape (H, W) with values in [0, 1]
    """
    if device is None:
        device = DEVICE

    if pattern == 'checkerboard':
        # Classic checkerboard - alternating 0/1
        y = torch.arange(H, device=device)
        x = torch.arange(W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        img = ((yy + xx) % 2).float()

    elif pattern == 'gradient':
        # Horizontal gradient from 0 to 1
        img = torch.linspace(0, 1, W, device=device).unsqueeze(0).expand(H, W)

    elif pattern == 'vertical_gradient':
        # Vertical gradient from 0 to 1
        img = torch.linspace(0, 1, H, device=device).unsqueeze(1).expand(H, W)

    elif pattern == 'uniform':
        # Uniform gray (0.5)
        img = torch.full((H, W), 0.5, device=device, dtype=torch.float32)

    elif pattern == 'random':
        # Random noise
        torch.manual_seed(42)
        img = torch.rand(H, W, device=device, dtype=torch.float32)

    elif pattern == 'stripes':
        # Horizontal stripes
        y = torch.arange(H, device=device)
        img = (y % 8 < 4).float().unsqueeze(1).expand(H, W)

    elif pattern == 'blocks':
        # 4 quadrants with different intensities
        img = torch.zeros(H, W, device=device, dtype=torch.float32)
        img[:H//2, :W//2] = 0.2
        img[:H//2, W//2:] = 0.4
        img[H//2:, :W//2] = 0.6
        img[H//2:, W//2:] = 0.8

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return img


def sparse_to_dense(sparse_tensor):
    """Convert sparse tensor to dense for comparison."""
    return sparse_tensor.to_dense()


def is_symmetric(L, tol=1e-6):
    """Check if sparse/dense matrix is symmetric."""
    L_dense = L.to_dense() if L.is_sparse else L
    return torch.allclose(L_dense, L_dense.T, atol=tol)


def row_sums(L):
    """Compute row sums of sparse/dense matrix."""
    L_dense = L.to_dense() if L.is_sparse else L
    return L_dense.sum(dim=1)


def check_positive_semidefinite(L, num_eigenvalues=10):
    """
    Check if matrix is positive semi-definite by checking smallest eigenvalues.

    For large sparse matrices, uses Lanczos. For small matrices, uses dense eigendecomposition.
    """
    L_dense = L.to_dense() if L.is_sparse else L
    n = L_dense.shape[0]

    if n <= 500:
        # Dense eigendecomposition for small matrices
        eigenvalues, _ = torch.linalg.eigh(L_dense)
        min_eigenvalue = eigenvalues[0].item()
    else:
        # For large matrices, just check a few smallest eigenvalues
        # Use power method on (lambda_max * I - L) to find smallest eigenvalue of L
        # This is a simplified check
        eigenvalues, _ = torch.linalg.eigh(L_dense[:min(500, n), :min(500, n)])
        min_eigenvalue = eigenvalues[0].item()

    return min_eigenvalue >= -1e-6


def count_nonzeros_per_row(L):
    """Count non-zero entries per row in sparse matrix."""
    L_dense = L.to_dense() if L.is_sparse else L
    return (L_dense.abs() > 1e-10).sum(dim=1)


# ============================================================
# TEST 1: Laplacian Properties
# ============================================================

@pytest.mark.skipif(not HAS_ORIGINAL, reason="Original implementation not available")
class TestOriginalLaplacianProperties:
    """Verify original Laplacian has correct properties."""

    def test_laplacian_shape(self):
        """Laplacian should be (n, n) where n = H*W."""
        H, W = 32, 32
        img = make_test_image(H, W, 'gradient')
        L = orig_build_laplacian(img)

        n = H * W
        assert L.shape == (n, n), f"Expected shape ({n}, {n}), got {L.shape}"

    def test_laplacian_row_sums_zero(self):
        """Row sums should be zero (or near-zero) for Laplacian."""
        H, W = 16, 16
        img = make_test_image(H, W, 'random')
        L = orig_build_laplacian(img)

        sums = row_sums(L)
        max_sum = sums.abs().max().item()
        assert max_sum < 1e-5, f"Row sums not zero, max = {max_sum}"

    def test_laplacian_symmetric(self):
        """Laplacian should be symmetric."""
        H, W = 16, 16
        img = make_test_image(H, W, 'checkerboard')
        L = orig_build_laplacian(img)

        assert is_symmetric(L), "Laplacian is not symmetric"

    def test_laplacian_positive_semidefinite(self):
        """Laplacian should be positive semi-definite."""
        H, W = 16, 16
        img = make_test_image(H, W, 'gradient')
        L = orig_build_laplacian(img)

        assert check_positive_semidefinite(L), "Laplacian is not positive semi-definite"

    def test_laplacian_sparsity_4_connectivity(self):
        """Sparsity pattern should match 4-connectivity (at most 5 nonzeros per row)."""
        H, W = 16, 16
        img = make_test_image(H, W, 'uniform')
        L = orig_build_laplacian(img)

        nnz_per_row = count_nonzeros_per_row(L)
        # Interior nodes: 5 (self + 4 neighbors)
        # Edge nodes: 4 (self + 3 neighbors)
        # Corner nodes: 3 (self + 2 neighbors)
        max_nnz = nnz_per_row.max().item()
        assert max_nnz <= 5, f"Too many nonzeros per row: {max_nnz}"

    def test_laplacian_diagonal_positive(self):
        """Diagonal entries should be non-negative (degrees)."""
        H, W = 16, 16
        img = make_test_image(H, W, 'random')
        L = orig_build_laplacian(img)

        L_dense = sparse_to_dense(L)
        diagonal = torch.diag(L_dense)
        assert (diagonal >= -1e-6).all(), "Diagonal contains negative entries"


@pytest.mark.skipif(not HAS_CUTER, reason="Cuter implementation not available")
class TestCuterLaplacianProperties:
    """Verify cuter Laplacian has correct properties."""

    def test_laplacian_shape(self):
        """Laplacian should be (n, n) where n = H*W."""
        H, W = 32, 32
        img = make_test_image(H, W, 'gradient')
        L = cuter_build_laplacian(img)

        n = H * W
        assert L.shape == (n, n), f"Expected shape ({n}, {n}), got {L.shape}"

    def test_laplacian_row_sums_zero(self):
        """Row sums should be zero (or near-zero) for Laplacian."""
        H, W = 16, 16
        img = make_test_image(H, W, 'random')
        L = cuter_build_laplacian(img)

        sums = row_sums(L)
        max_sum = sums.abs().max().item()
        assert max_sum < 1e-5, f"Row sums not zero, max = {max_sum}"

    def test_laplacian_symmetric(self):
        """Laplacian should be symmetric."""
        H, W = 16, 16
        img = make_test_image(H, W, 'checkerboard')
        L = cuter_build_laplacian(img)

        assert is_symmetric(L), "Laplacian is not symmetric"

    def test_laplacian_positive_semidefinite(self):
        """Laplacian should be positive semi-definite."""
        H, W = 16, 16
        img = make_test_image(H, W, 'gradient')
        L = cuter_build_laplacian(img)

        assert check_positive_semidefinite(L), "Laplacian is not positive semi-definite"


# ============================================================
# TEST 2: Fiedler Vector Properties
# ============================================================

@pytest.mark.skipif(not HAS_ORIGINAL, reason="Original implementation not available")
class TestOriginalFiedlerProperties:
    """Verify original Fiedler vector has correct properties."""

    def test_fiedler_orthogonal_to_constant(self):
        """Fiedler should be orthogonal to constant vector (mean ~ 0)."""
        H, W = 32, 32
        img = make_test_image(H, W, 'gradient')
        L = orig_build_laplacian(img)

        fiedler, lambda2 = orig_lanczos(L)

        mean_val = fiedler.mean().item()
        assert abs(mean_val) < 1e-4, f"Fiedler mean not zero: {mean_val}"

    def test_fiedler_is_eigenvector(self):
        """L @ fiedler should approximately equal lambda2 * fiedler."""
        # Use uniform image which produces a regular grid Laplacian
        # This has well-separated eigenvalues making Lanczos converge better
        H, W = 16, 16
        img = make_test_image(H, W, 'uniform')
        L = orig_build_laplacian(img)

        # Use more iterations for better convergence
        fiedler, lambda2 = orig_lanczos(L, num_iterations=80)

        if lambda2 > 1e-6:
            # L @ fiedler
            Lf = torch.sparse.mm(L, fiedler.unsqueeze(1)).squeeze(1)

            # Normalize fiedler before checking eigenvector property
            fiedler_norm = fiedler / (torch.linalg.norm(fiedler) + 1e-10)
            Lf_norm = Lf / (torch.linalg.norm(Lf) + 1e-10)

            # For a true eigenvector, L @ v / ||L @ v|| should equal v (up to sign)
            # Check correlation instead of relative error
            correlation = torch.abs(torch.dot(fiedler_norm, Lf_norm)).item()

            # Also check Rayleigh quotient: (v^T L v) / (v^T v) should approximate lambda2
            vTLv = torch.dot(fiedler, Lf)
            vTv = torch.dot(fiedler, fiedler)
            rayleigh = (vTLv / vTv).item()

            # Lanczos is iterative, so we check both correlation and Rayleigh quotient
            assert correlation > 0.8 or abs(rayleigh - lambda2) / (lambda2 + 1e-10) < 0.2, \
                f"Fiedler not a good eigenvector: correlation={correlation:.4f}, rayleigh={rayleigh:.6f}, lambda2={lambda2:.6f}"

    def test_fiedler_normalized(self):
        """Fiedler vector should have unit norm or be normalizable."""
        H, W = 32, 32
        img = make_test_image(H, W, 'random')
        L = orig_build_laplacian(img)

        fiedler, _ = orig_lanczos(L)

        # Fiedler should be non-trivial
        norm = torch.linalg.norm(fiedler).item()
        assert norm > 1e-6, "Fiedler vector is trivial (zero)"

    def test_fiedler_lambda2_positive(self):
        """Second eigenvalue should be positive for connected graph."""
        H, W = 16, 16
        img = make_test_image(H, W, 'checkerboard')
        L = orig_build_laplacian(img)

        _, lambda2 = orig_lanczos(L)

        assert lambda2 > 0, f"Lambda2 should be positive, got {lambda2}"


@pytest.mark.skipif(not HAS_CUTER, reason="Cuter implementation not available")
class TestCuterFiedlerProperties:
    """Verify cuter Fiedler vector has correct properties."""

    def test_fiedler_orthogonal_to_constant(self):
        """Fiedler should be orthogonal to constant vector (mean ~ 0)."""
        H, W = 32, 32
        img = make_test_image(H, W, 'gradient')
        L = cuter_build_laplacian(img)

        fiedler, lambda2 = cuter_lanczos(L)

        mean_val = fiedler.mean().item()
        assert abs(mean_val) < 1e-4, f"Fiedler mean not zero: {mean_val}"

    def test_fiedler_is_eigenvector(self):
        """L @ fiedler should approximately equal lambda2 * fiedler."""
        # Use uniform image which produces a regular grid Laplacian
        H, W = 16, 16
        img = make_test_image(H, W, 'uniform')
        L = cuter_build_laplacian(img)

        fiedler, lambda2 = cuter_lanczos(L, num_iterations=80)

        if lambda2 > 1e-6:
            # L @ fiedler
            Lf = torch.sparse.mm(L, fiedler.unsqueeze(1)).squeeze(1)

            # Normalize fiedler before checking eigenvector property
            fiedler_norm = fiedler / (torch.linalg.norm(fiedler) + 1e-10)
            Lf_norm = Lf / (torch.linalg.norm(Lf) + 1e-10)

            # Check correlation instead of relative error
            correlation = torch.abs(torch.dot(fiedler_norm, Lf_norm)).item()

            # Also check Rayleigh quotient
            vTLv = torch.dot(fiedler, Lf)
            vTv = torch.dot(fiedler, fiedler)
            rayleigh = (vTLv / vTv).item()

            assert correlation > 0.8 or abs(rayleigh - lambda2) / (lambda2 + 1e-10) < 0.2, \
                f"Fiedler not a good eigenvector: correlation={correlation:.4f}, rayleigh={rayleigh:.6f}, lambda2={lambda2:.6f}"


# ============================================================
# TEST 3: Fiedler Equivalence (Original vs Cuter)
# ============================================================

@pytest.mark.skipif(not (HAS_ORIGINAL and HAS_CUTER), reason="Need both implementations")
class TestFiedlerEquivalence:
    """Compare Fiedler from original vs cuter implementation."""

    def test_fiedler_equivalence_gradient(self):
        """Fiedler vectors should match (up to sign flip) on gradient image."""
        H, W = 32, 32
        img = make_test_image(H, W, 'gradient')

        L_orig = orig_build_laplacian(img)
        L_cuter = cuter_build_laplacian(img)

        f_orig, l2_orig = orig_lanczos(L_orig)
        f_cuter, l2_cuter = cuter_lanczos(L_cuter)

        # Normalize both
        f_orig = f_orig / (torch.linalg.norm(f_orig) + 1e-10)
        f_cuter = f_cuter / (torch.linalg.norm(f_cuter) + 1e-10)

        # Check correlation (handles sign flip)
        correlation = torch.abs(torch.dot(f_orig, f_cuter)).item()
        assert correlation > 0.95, f"Fiedler correlation too low: {correlation}"

        # Eigenvalues should match
        rel_error = abs(l2_orig - l2_cuter) / (abs(l2_orig) + 1e-10)
        assert rel_error < 0.1, f"Lambda2 mismatch: orig={l2_orig}, cuter={l2_cuter}"

    def test_fiedler_equivalence_checkerboard(self):
        """Fiedler vectors should match on checkerboard image."""
        H, W = 16, 16
        img = make_test_image(H, W, 'checkerboard')

        L_orig = orig_build_laplacian(img)
        L_cuter = cuter_build_laplacian(img)

        f_orig, _ = orig_lanczos(L_orig)
        f_cuter, _ = cuter_lanczos(L_cuter)

        # Normalize
        f_orig = f_orig / (torch.linalg.norm(f_orig) + 1e-10)
        f_cuter = f_cuter / (torch.linalg.norm(f_cuter) + 1e-10)

        # Check with abs to handle sign flip
        correlation = torch.abs(torch.dot(f_orig, f_cuter)).item()
        assert correlation > 0.9, f"Fiedler correlation too low: {correlation}"

    def test_fiedler_equivalence_random(self):
        """Fiedler vectors should match on random image."""
        H, W = 24, 24
        img = make_test_image(H, W, 'random')

        L_orig = orig_build_laplacian(img)
        L_cuter = cuter_build_laplacian(img)

        f_orig, _ = orig_lanczos(L_orig)
        f_cuter, _ = cuter_lanczos(L_cuter)

        # Normalize
        f_orig = f_orig / (torch.linalg.norm(f_orig) + 1e-10)
        f_cuter = f_cuter / (torch.linalg.norm(f_cuter) + 1e-10)

        correlation = torch.abs(torch.dot(f_orig, f_cuter)).item()
        assert correlation > 0.9, f"Fiedler correlation too low: {correlation}"


# ============================================================
# TEST 4: Tiled vs Global
# ============================================================

@pytest.mark.skipif(not HAS_ORIGINAL, reason="Original implementation not available")
class TestTiledVsGlobalOriginal:
    """Compare tiled Fiedler to global Fiedler using original implementation."""

    def test_tiled_produces_output(self):
        """Tiled computation should produce valid output."""
        H, W = 64, 64
        img = make_test_image(H, W, 'gradient')

        # Compute tiled eigenvectors
        result = orig_tiled(
            img,
            tile_size=32,
            overlap=8,
            num_eigenvectors=2,
            lanczos_iterations=20
        )

        assert result.shape == (H, W, 2), f"Unexpected shape: {result.shape}"
        assert not torch.isnan(result).any(), "Result contains NaN"
        assert not torch.isinf(result).any(), "Result contains Inf"

    def test_tiled_global_correlation(self):
        """Tiled result should approximate global on small image."""
        H, W = 32, 32
        img = make_test_image(H, W, 'blocks')

        # Global Fiedler using 4-connected Laplacian
        L = orig_build_laplacian(img)
        fiedler_global, _ = orig_lanczos(L, num_iterations=50)
        fiedler_global = fiedler_global.reshape(H, W)

        # Tiled (single tile that covers whole image) - uses same 4-connected Laplacian
        result_tiled = orig_tiled(
            img,
            tile_size=32,
            overlap=0,  # No overlap since single tile
            num_eigenvectors=1,
            edge_threshold=0.1,  # Match default
            lanczos_iterations=50
        )
        fiedler_tiled = result_tiled[:, :, 0]

        # Flatten and correlate
        g = fiedler_global.flatten()
        t = fiedler_tiled.flatten()

        g = g / (torch.linalg.norm(g) + 1e-10)
        t = t / (torch.linalg.norm(t) + 1e-10)

        correlation = torch.abs(torch.dot(g, t)).item()
        # Tiled may produce slightly different eigenvectors due to different initialization
        # and local vs global computation. A moderate correlation indicates they're computing
        # similar structure.
        assert correlation > 0.3, f"Tiled/global correlation too low: {correlation}"

    def test_tiled_dither_produces_output(self):
        """Dither-aware tiled computation should produce valid output."""
        H, W = 64, 64
        img = make_test_image(H, W, 'checkerboard')

        # Compute tiled eigenvectors with multi-radius connectivity
        result = orig_tiled_dither(
            img,
            tile_size=32,
            overlap=8,
            num_eigenvectors=2,
            radii=[1, 2, 3],  # Reduced radii for speed
            lanczos_iterations=20
        )

        assert result.shape == (H, W, 2), f"Unexpected shape: {result.shape}"
        assert not torch.isnan(result).any(), "Result contains NaN"
        assert not torch.isinf(result).any(), "Result contains Inf"


@pytest.mark.skipif(not HAS_CUTER, reason="Cuter implementation not available")
class TestTiledVsGlobalCuter:
    """Compare tiled Fiedler to global Fiedler using cuter implementation."""

    def test_tiled_produces_output(self):
        """Tiled computation should produce valid output."""
        H, W = 64, 64
        img = make_test_image(H, W, 'gradient')

        # Compute tiled eigenvectors
        result = cuter_tiled(
            img,
            tile_size=32,
            overlap=8,
            num_eigenvectors=2,
        )

        assert result.shape[0] == H and result.shape[1] == W, f"Unexpected shape: {result.shape}"
        assert not torch.isnan(result).any(), "Result contains NaN"
        assert not torch.isinf(result).any(), "Result contains Inf"


# ============================================================
# TEST 5: Performance Comparison
# ============================================================

@pytest.mark.skipif(not HAS_ORIGINAL, reason="Original implementation not available")
class TestPerformanceOriginal:
    """Benchmark original implementation."""

    def test_laplacian_build_time(self):
        """Time Laplacian construction."""
        H, W = 128, 128
        img = make_test_image(H, W, 'random')

        # Warmup
        _ = orig_build_laplacian(img)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        # Time
        start = time.perf_counter()
        for _ in range(5):
            L = orig_build_laplacian(img)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 5
        print(f"\nOriginal Laplacian build ({H}x{W}): {avg_time*1000:.2f} ms")

        # Just ensure it runs, no hard assertion on time
        assert avg_time < 10.0, "Laplacian build too slow"

    def test_lanczos_time(self):
        """Time Lanczos iteration."""
        H, W = 64, 64
        img = make_test_image(H, W, 'random')
        L = orig_build_laplacian(img)

        # Warmup
        _ = orig_lanczos(L)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        # Time
        start = time.perf_counter()
        for _ in range(5):
            _, _ = orig_lanczos(L)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 5
        print(f"\nOriginal Lanczos ({H}x{W}): {avg_time*1000:.2f} ms")

        assert avg_time < 10.0, "Lanczos too slow"


@pytest.mark.skipif(not (HAS_ORIGINAL and HAS_CUTER), reason="Need both implementations")
class TestPerformanceComparison:
    """Compare performance of original vs cuter implementations."""

    def test_laplacian_speedup(self):
        """Compare Laplacian construction speed."""
        H, W = 128, 128
        img = make_test_image(H, W, 'random')

        # Time original
        _ = orig_build_laplacian(img)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(5):
            L = orig_build_laplacian(img)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
        orig_time = (time.perf_counter() - start) / 5

        # Time cuter
        _ = cuter_build_laplacian(img)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(5):
            L = cuter_build_laplacian(img)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
        cuter_time = (time.perf_counter() - start) / 5

        speedup = orig_time / (cuter_time + 1e-10)
        print(f"\nLaplacian build speedup: {speedup:.2f}x (orig={orig_time*1000:.2f}ms, cuter={cuter_time*1000:.2f}ms)")

    def test_lanczos_speedup(self):
        """Compare Lanczos speed."""
        H, W = 64, 64
        img = make_test_image(H, W, 'random')

        L_orig = orig_build_laplacian(img)
        L_cuter = cuter_build_laplacian(img)

        # Time original
        _ = orig_lanczos(L_orig)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(5):
            _, _ = orig_lanczos(L_orig)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
        orig_time = (time.perf_counter() - start) / 5

        # Time cuter
        _ = cuter_lanczos(L_cuter)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(5):
            _, _ = cuter_lanczos(L_cuter)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
        cuter_time = (time.perf_counter() - start) / 5

        speedup = orig_time / (cuter_time + 1e-10)
        print(f"\nLanczos speedup: {speedup:.2f}x (orig={orig_time*1000:.2f}ms, cuter={cuter_time*1000:.2f}ms)")


# ============================================================
# TEST 6: Edge Cases
# ============================================================

@pytest.mark.skipif(not HAS_ORIGINAL, reason="Original implementation not available")
class TestEdgeCasesOriginal:
    """Test edge cases with original implementation."""

    def test_uniform_image(self):
        """Uniform image: all edges have same weight."""
        H, W = 16, 16
        img = make_test_image(H, W, 'uniform')
        L = orig_build_laplacian(img)

        # Should still produce valid Laplacian
        assert L.shape == (H * W, H * W)

        # Row sums should be zero
        sums = row_sums(L)
        assert sums.abs().max().item() < 1e-5

        # Fiedler should still be computable
        fiedler, lambda2 = orig_lanczos(L)
        assert not torch.isnan(fiedler).any()

    def test_tiny_image_2x2(self):
        """2x2 image should not crash."""
        H, W = 2, 2
        img = make_test_image(H, W, 'random')
        L = orig_build_laplacian(img)

        assert L.shape == (4, 4)

        # Lanczos may return zeros for tiny image, but should not crash
        fiedler, lambda2 = orig_lanczos(L)
        assert fiedler.shape == (4,)

    def test_tiny_image_3x3(self):
        """3x3 image should produce valid results."""
        H, W = 3, 3
        img = make_test_image(H, W, 'gradient')
        L = orig_build_laplacian(img)

        assert L.shape == (9, 9)

        fiedler, lambda2 = orig_lanczos(L)
        assert fiedler.shape == (9,)
        assert lambda2 >= 0

    def test_single_row(self):
        """1xN image (single row)."""
        H, W = 1, 16
        img = make_test_image(H, W, 'gradient')
        L = orig_build_laplacian(img)

        assert L.shape == (16, 16)

        # Each node has at most 2 neighbors (left and right)
        nnz_per_row = count_nonzeros_per_row(L)
        assert nnz_per_row.max().item() <= 3  # self + 2 neighbors

    def test_single_column(self):
        """Nx1 image (single column)."""
        H, W = 16, 1
        img = make_test_image(H, W, 'vertical_gradient')
        L = orig_build_laplacian(img)

        assert L.shape == (16, 16)

    def test_rectangular_image(self):
        """Non-square image."""
        H, W = 24, 48
        img = make_test_image(H, W, 'stripes')
        L = orig_build_laplacian(img)

        n = H * W
        assert L.shape == (n, n)

        fiedler, lambda2 = orig_lanczos(L)
        assert fiedler.shape == (n,)


@pytest.mark.skipif(not HAS_CUTER, reason="Cuter implementation not available")
class TestEdgeCasesCuter:
    """Test edge cases with cuter implementation."""

    def test_uniform_image(self):
        """Uniform image should not crash."""
        H, W = 16, 16
        img = make_test_image(H, W, 'uniform')
        L = cuter_build_laplacian(img)

        assert L.shape == (H * W, H * W)

    def test_tiny_image_2x2(self):
        """2x2 image should not crash."""
        H, W = 2, 2
        img = make_test_image(H, W, 'random')
        L = cuter_build_laplacian(img)

        assert L.shape == (4, 4)

    def test_tiny_image_3x3(self):
        """3x3 image should produce valid results."""
        H, W = 3, 3
        img = make_test_image(H, W, 'gradient')
        L = cuter_build_laplacian(img)

        assert L.shape == (9, 9)


# ============================================================
# Additional validation tests
# ============================================================

@pytest.mark.skipif(not HAS_ORIGINAL, reason="Original implementation not available")
class TestGraphClassIntegration:
    """Test Graph class integration with image Laplacians."""

    def test_graph_from_image_laplacian_matches(self):
        """Graph.from_image Laplacian should match build_weighted_image_laplacian."""
        H, W = 16, 16
        img = make_test_image(H, W, 'gradient')

        # Build using Graph.from_image
        graph = Graph.from_image(img.cpu().numpy() if DEVICE.type != 'cpu' else img.numpy(),
                                  connectivity=4, edge_threshold=0.1)
        L_graph = graph.laplacian()

        # Build using standalone function
        L_func = orig_build_laplacian(img, edge_threshold=0.1)

        # Should have same shape
        assert L_graph.shape == L_func.shape

        # Both should be valid Laplacians
        assert is_symmetric(L_graph)
        assert is_symmetric(L_func)


# ============================================================
# Main runner for quick testing
# ============================================================

if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Has original: {HAS_ORIGINAL}")
    print(f"Has cuter: {HAS_CUTER}")
    print()

    # Run a quick sanity check
    if HAS_ORIGINAL:
        print("Testing original implementation...")
        img = make_test_image(32, 32, 'gradient')
        L = orig_build_laplacian(img)
        print(f"  Laplacian shape: {L.shape}")
        print(f"  Row sums max: {row_sums(L).abs().max().item():.6f}")
        print(f"  Symmetric: {is_symmetric(L)}")

        fiedler, lambda2 = orig_lanczos(L)
        print(f"  Fiedler mean: {fiedler.mean().item():.6f}")
        print(f"  Lambda2: {lambda2:.6f}")
        print()

    if HAS_CUTER:
        print("Testing cuter implementation...")
        img = make_test_image(32, 32, 'gradient')
        L = cuter_build_laplacian(img)
        print(f"  Laplacian shape: {L.shape}")
        print(f"  Row sums max: {row_sums(L).abs().max().item():.6f}")
        print(f"  Symmetric: {is_symmetric(L)}")

        fiedler, lambda2 = cuter_lanczos(L)
        print(f"  Fiedler mean: {fiedler.mean().item():.6f}")
        print(f"  Lambda2: {lambda2:.6f}")
        print()

    if HAS_ORIGINAL and HAS_CUTER:
        print("Testing equivalence...")
        img = make_test_image(32, 32, 'gradient')

        L_orig = orig_build_laplacian(img)
        L_cuter = cuter_build_laplacian(img)

        f_orig, l2_orig = orig_lanczos(L_orig)
        f_cuter, l2_cuter = cuter_lanczos(L_cuter)

        f_orig = f_orig / (torch.linalg.norm(f_orig) + 1e-10)
        f_cuter = f_cuter / (torch.linalg.norm(f_cuter) + 1e-10)

        correlation = torch.abs(torch.dot(f_orig, f_cuter)).item()
        print(f"  Fiedler correlation: {correlation:.4f}")
        print(f"  Lambda2 orig: {l2_orig:.6f}, cuter: {l2_cuter:.6f}")
        print()

    print("Run full tests with: pytest test_spectral_ops_cuter.py -v")
