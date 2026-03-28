"""Test recursive reduction on a valid full N×N unitary polynomial (no complement needed)."""

import numpy as np
import pytest

from src.reduce import recursive_reduction
from src.verify import verify
from tests.conftest import make_random_achievable


def make_full_unitary_polynomial(d: int, N: int, seed: int = 0):
    """Generate a valid N×N unitary polynomial by forward simulation.
    
    This creates a polynomial U(z) that is unitary for all |z|=1,
    without needing any complement computation.
    """
    rng = np.random.default_rng(seed)
    
    # Random unitaries in U(N)
    def random_unitary(size):
        A = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
        Q, _ = np.linalg.qr(A)
        return Q
    
    R_list = [random_unitary(N) for _ in range(d + 1)]
    ell_list = rng.integers(1, N // 2 + 1, size=d).tolist()
    
    # Forward simulation to get polynomial coefficients
    from src.utils import unit_circle_points
    K = max(8 * (d + 1), 512)
    z_pts = unit_circle_points(K)
    
    U_vals = np.zeros((K, N, N), dtype=np.complex128)
    for ki in range(K):
        z = z_pts[ki]
        M = R_list[0].copy()
        for k in range(1, d + 1):
            ell = ell_list[k - 1]
            C_k = np.eye(N, dtype=np.complex128)
            C_k[:ell, :ell] *= z
            M = R_list[k] @ C_k @ M
        U_vals[ki] = M
    
    # Recover polynomial coefficients via IFFT
    all_coeffs = np.fft.fft(U_vals, axis=0) / K
    U_coeffs = all_coeffs[:d + 1]
    
    return U_coeffs, R_list, ell_list


def test_reduction_on_full_unitary():
    """Test that reduction works correctly on a valid full N×N unitary polynomial.
    
    This tests the reduction algorithm without any complement computation,
    using a polynomial that is already a full unitary.
    """
    d, N = 4, 4
    U_coeffs, R_list_expected, ell_list_expected = make_full_unitary_polynomial(d, N, seed=42)
    
    # Verify U(z) is unitary on unit circle
    from src.utils import unit_circle_points, poly_eval_batch
    z_pts = unit_circle_points(256)
    U_vals = poly_eval_batch(U_coeffs, z_pts)
    
    for k in range(len(z_pts)):
        err = np.linalg.norm(U_vals[k] @ U_vals[k].conj().T - np.eye(N), "fro")
        assert err < 1e-10, f"U(z[{k}]) not unitary: err={err:.2e}"
    
    # Run reduction
    result = recursive_reduction(U_coeffs)
    
    # Check output structure
    assert len(result.R_list) == d + 1
    assert len(result.ell_list) == d
    
    # All R_k should be unitary
    for k, R in enumerate(result.R_list):
        assert R.shape == (N, N)
        err = np.linalg.norm(R @ R.conj().T - np.eye(N), "fro")
        assert err < 1e-10, f"R_list[{k}] not unitary: err={err:.2e}"
    
    # Forward simulation should reconstruct U(z)
    for ki in range(min(10, len(z_pts))):  # Test first 10 points
        z = z_pts[ki]
        M = result.R_list[0].copy()
        for k in range(1, d + 1):
            ell = result.ell_list[k - 1]
            C_k = np.eye(N, dtype=np.complex128)
            C_k[:ell, :ell] *= z
            M = result.R_list[k] @ C_k @ M
        
        err = np.linalg.norm(M - U_vals[ki], "fro")
        assert err < 1e-8, f"Reconstruction error at z[{ki}]: {err:.2e}"


def test_reduction_preserves_unitarity():
    """Test that the reduction preserves unitarity throughout the recursion."""
    d, N = 6, 8
    U_coeffs, _, _ = make_full_unitary_polynomial(d, N, seed=99)
    
    result = recursive_reduction(U_coeffs)
    
    # Verify all intermediate steps preserve unitarity
    from src.utils import unit_circle_points, poly_eval_batch
    z_pts = unit_circle_points(128)
    
    for ki in range(len(z_pts)):
        z = z_pts[ki]
        M = result.R_list[0].copy()
        
        # Check after each step
        for k in range(1, d + 1):
            ell = result.ell_list[k - 1]
            C_k = np.eye(N, dtype=np.complex128)
            C_k[:ell, :ell] *= z
            M = result.R_list[k] @ C_k @ M
            
            # M should remain unitary
            err = np.linalg.norm(M @ M.conj().T - np.eye(N), "fro")
            assert err < 1e-8, f"Lost unitarity at step {k}, z[{ki}]: err={err:.2e}"


if __name__ == "__main__":
    test_reduction_on_full_unitary()
    test_reduction_preserves_unitarity()
    print("✓ All tests passed!")
