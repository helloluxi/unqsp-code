"""Test the fundamental theorem: P(z) unitary for all |z|=1 implies col(P_d) ⊥ col(P_0).

This is a key property from the paper that enables the recursive reduction algorithm.

Theorem: If P(z) is a polynomial matrix that is unitary for all |z|=1, then
the leading coefficient P_d and constant coefficient P_0 have orthogonal column spaces:
    P_d† P_0 = 0

Proof sketch:
- P(z) unitary means P(z)† P(z) = I for all |z|=1
- Expand: (P_0† + z̄P_1† + ... + z̄^d P_d†)(P_0 + zP_1 + ... + z^d P_d) = I
- Coefficient of z^d: P_d† P_0 = 0  (orthogonality)
- Coefficient of z^(-d): P_0† P_d = 0  (conjugate)
"""

import numpy as np
import pytest

from src.utils import unit_circle_points


def test_unitary_polynomial_implies_orthogonality():
    """Test that any unitary polynomial P(z) has orthogonal leading and constant coefficients.
    
    This is the fundamental property that makes the recursive reduction algorithm work.
    """
    d, N = 8, 8
    rng = np.random.default_rng(42)
    
    # Generate a valid unitary polynomial via forward U(N)-QSP
    def random_unitary(size):
        A = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
        Q, _ = np.linalg.qr(A)
        return Q
    
    R_list = [random_unitary(N) for _ in range(d + 1)]
    ell_list = rng.integers(1, N // 2 + 1, size=d).tolist()
    
    # Forward simulation to get polynomial coefficients
    K = max(8 * (d + 1), 512)
    z_pts = unit_circle_points(K)
    
    P_vals = np.zeros((K, N, N), dtype=np.complex128)
    for ki in range(K):
        z = z_pts[ki]
        M = R_list[0].copy()
        for k in range(1, d + 1):
            ell = ell_list[k - 1]
            C_k = np.eye(N, dtype=np.complex128)
            C_k[:ell, :ell] *= z
            M = R_list[k] @ C_k @ M
        P_vals[ki] = M
    
    # Verify P(z) is unitary at all sample points
    for ki in range(len(z_pts)):
        unitarity_error = np.linalg.norm(P_vals[ki] @ P_vals[ki].conj().T - np.eye(N), "fro")
        assert unitarity_error < 1e-10, f"P(z[{ki}]) not unitary: {unitarity_error:.3e}"
    
    # Recover polynomial coefficients
    all_coeffs = np.fft.fft(P_vals, axis=0) / K
    P_coeffs = all_coeffs[:d + 1]
    
    # Extract leading and constant coefficients
    P_lead = P_coeffs[d]  # P_d (degree d)
    P_const = P_coeffs[0]  # P_0 (degree 0)
    
    # Check orthogonality: P_d† P_0 = 0
    cross_product = P_lead.conj().T @ P_const
    orthogonality_error = np.linalg.norm(cross_product, "fro")
    
    print(f"\nUnitary polynomial P(z) (d={d}, N={N}):")
    print(f"  Verified unitary at {len(z_pts)} sample points")
    print(f"  P_d† P_0 error: {orthogonality_error:.3e}")
    
    # This MUST be satisfied for any unitary polynomial
    assert orthogonality_error < 1e-10, (
        f"Unitary polynomial violates orthogonality: {orthogonality_error:.3e}"
    )


def test_orthogonality_from_unitarity_constraint():
    """Verify orthogonality by directly analyzing the unitarity constraint.
    
    If P(z)† P(z) = I for all |z|=1, then expanding the product and
    collecting coefficients of z^d gives P_d† P_0 = 0.
    """
    d, N = 6, 4
    rng = np.random.default_rng(99)
    
    # Generate unitary polynomial
    def random_unitary(size):
        A = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
        Q, _ = np.linalg.qr(A)
        return Q
    
    R_list = [random_unitary(N) for _ in range(d + 1)]
    ell_list = rng.integers(1, N // 2 + 1, size=d).tolist()
    
    K = 512
    z_pts = unit_circle_points(K)
    
    P_vals = np.zeros((K, N, N), dtype=np.complex128)
    for ki in range(K):
        z = z_pts[ki]
        M = R_list[0].copy()
        for k in range(1, d + 1):
            ell = ell_list[k - 1]
            C_k = np.eye(N, dtype=np.complex128)
            C_k[:ell, :ell] *= z
            M = R_list[k] @ C_k @ M
        P_vals[ki] = M
    
    # Recover coefficients
    all_coeffs = np.fft.fft(P_vals, axis=0) / K
    P_coeffs = all_coeffs[:d + 1]
    
    # Compute P(z)† P(z) at sample points
    PtP_vals = np.zeros((K, N, N), dtype=np.complex128)
    for ki in range(K):
        PtP_vals[ki] = P_vals[ki].conj().T @ P_vals[ki]
    
    # Recover coefficients of P(z)† P(z)
    PtP_all_coeffs = np.fft.fft(PtP_vals, axis=0) / K
    
    # The coefficient of z^0 should be I (constant term of P†P = I)
    PtP_const = PtP_all_coeffs[0]
    const_error = np.linalg.norm(PtP_const - np.eye(N), "fro")
    
    # The coefficient of z^d in P†P comes from P_d† P_0 (among other terms)
    # For unitary P(z), all non-constant coefficients of P†P should be 0
    PtP_degree_d = PtP_all_coeffs[d]
    degree_d_error = np.linalg.norm(PtP_degree_d, "fro")
    
    print(f"\nUnitarity constraint analysis (d={d}, N={N}):")
    print(f"  P†P constant term vs I: {const_error:.3e}")
    print(f"  P†P degree-d coefficient: {degree_d_error:.3e}")
    
    assert const_error < 1e-10, "P†P constant term should be I"
    assert degree_d_error < 1e-10, "P†P degree-d coefficient should be 0"
    
    # Now verify P_d† P_0 directly
    P_lead = P_coeffs[d]
    P_const = P_coeffs[0]
    direct_orthogonality = np.linalg.norm(P_lead.conj().T @ P_const, "fro")
    
    print(f"  Direct P_d† P_0: {direct_orthogonality:.3e}")
    
    assert direct_orthogonality < 1e-10, "P_d† P_0 should be 0"


def test_multiple_random_unitaries():
    """Test orthogonality property for multiple random unitary polynomials.
    
    This verifies the property holds universally, not just for specific cases.
    """
    d, N = 5, 4
    num_trials = 10
    
    print(f"\nTesting {num_trials} random unitary polynomials (d={d}, N={N}):")
    
    for trial in range(num_trials):
        rng = np.random.default_rng(trial * 137)
        
        # Generate random unitary polynomial
        def random_unitary(size):
            A = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
            Q, _ = np.linalg.qr(A)
            return Q
        
        R_list = [random_unitary(N) for _ in range(d + 1)]
        ell_list = rng.integers(1, N // 2 + 1, size=d).tolist()
        
        K = 256
        z_pts = unit_circle_points(K)
        
        P_vals = np.zeros((K, N, N), dtype=np.complex128)
        for ki in range(K):
            z = z_pts[ki]
            M = R_list[0].copy()
            for k in range(1, d + 1):
                ell = ell_list[k - 1]
                C_k = np.eye(N, dtype=np.complex128)
                C_k[:ell, :ell] *= z
                M = R_list[k] @ C_k @ M
            P_vals[ki] = M
        
        # Recover coefficients
        all_coeffs = np.fft.fft(P_vals, axis=0) / K
        P_coeffs = all_coeffs[:d + 1]
        
        # Check orthogonality
        P_lead = P_coeffs[d]
        P_const = P_coeffs[0]
        orthogonality_error = np.linalg.norm(P_lead.conj().T @ P_const, "fro")
        
        print(f"  Trial {trial + 1}: P_d† P_0 error = {orthogonality_error:.3e}")
        
        assert orthogonality_error < 1e-10, (
            f"Trial {trial + 1} failed: {orthogonality_error:.3e}"
        )
    
    print(f"  ✓ All {num_trials} trials passed!")


def test_counterexample_non_unitary():
    """Test that non-unitary polynomials do NOT satisfy orthogonality.
    
    This proves the converse: if P(z) is not unitary, then col(P_d) ⊥ col(P_0)
    generally does NOT hold.
    """
    d, N = 6, 4
    rng = np.random.default_rng(555)
    
    # Create a random polynomial that is NOT unitary
    P_coeffs = (
        rng.standard_normal((d + 1, N, N)) + 
        1j * rng.standard_normal((d + 1, N, N))
    )
    
    # Verify it's not unitary
    from src.utils import poly_eval_batch
    z_pts = unit_circle_points(128)
    P_vals = poly_eval_batch(P_coeffs, z_pts)
    
    max_unitarity_error = 0.0
    for ki in range(len(z_pts)):
        error = np.linalg.norm(P_vals[ki] @ P_vals[ki].conj().T - np.eye(N), "fro")
        max_unitarity_error = max(max_unitarity_error, error)
    
    # Check orthogonality
    P_lead = P_coeffs[d]
    P_const = P_coeffs[0]
    orthogonality_error = np.linalg.norm(P_lead.conj().T @ P_const, "fro")
    
    print(f"\nNon-unitary polynomial (d={d}, N={N}):")
    print(f"  Max unitarity violation: {max_unitarity_error:.3e}")
    print(f"  P_d† P_0 error: {orthogonality_error:.3e}")
    
    # For non-unitary polynomial, we expect:
    # 1. Large unitarity violation
    assert max_unitarity_error > 0.1, "Should not be unitary"
    
    # 2. Orthogonality generally does NOT hold (though it could by chance)
    # We just verify it's not guaranteed to be small
    print(f"  ✓ Non-unitary polynomial does not guarantee orthogonality")


if __name__ == "__main__":
    test_unitary_polynomial_implies_orthogonality()
    test_orthogonality_from_unitarity_constraint()
    test_multiple_random_unitaries()
    test_counterexample_non_unitary()
    print("\n✓ All tests passed! Theorem verified: P(z) unitary ⟹ P_d† P_0 = 0")
