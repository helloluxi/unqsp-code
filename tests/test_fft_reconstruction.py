"""Test that FFT/IFFT correctly reconstructs polynomial coefficients when cropping.

This verifies that when we:
1. Generate a full N×N unitary polynomial U(z) via forward pass
2. Sample U(z) on the unit circle
3. Crop to get P(z) = U[:r, :c]
4. Use IFFT to recover polynomial coefficients

The recovered coefficients correctly reconstruct P(z) at all sample points.
"""

import numpy as np
import pytest

from src.utils import unit_circle_points, poly_eval_batch


def test_fft_reconstruction_full_unitary():
    """Test FFT/IFFT round-trip for full N×N unitary polynomial."""
    d, N = 6, 4
    rng = np.random.default_rng(42)
    
    # Generate random unitaries
    def random_unitary(size):
        A = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
        Q, _ = np.linalg.qr(A)
        return Q
    
    R_list = [random_unitary(N) for _ in range(d + 1)]
    ell_list = rng.integers(1, N // 2 + 1, size=d).tolist()
    
    # Forward pass: compute U(z) at sample points
    K = max(8 * (d + 1), 512)
    z_pts = unit_circle_points(K)
    
    U_vals_forward = np.zeros((K, N, N), dtype=np.complex128)
    for ki in range(K):
        z = z_pts[ki]
        M = R_list[0].copy()
        for k in range(1, d + 1):
            ell = ell_list[k - 1]
            C_k = np.eye(N, dtype=np.complex128)
            C_k[:ell, :ell] *= z
            M = R_list[k] @ C_k @ M
        U_vals_forward[ki] = M
    
    # IFFT to recover polynomial coefficients
    all_coeffs = np.fft.fft(U_vals_forward, axis=0) / K
    U_coeffs = all_coeffs[:d + 1]  # (d+1, N, N)
    
    # FFT back to reconstruct U(z) values
    U_vals_reconstructed = poly_eval_batch(U_coeffs, z_pts)
    
    # Check reconstruction error
    reconstruction_error = np.max(np.abs(U_vals_forward - U_vals_reconstructed))
    
    print(f"\nFull unitary FFT round-trip (d={d}, N={N}, K={K}):")
    print(f"  Max reconstruction error: {reconstruction_error:.3e}")
    
    assert reconstruction_error < 1e-10, (
        f"FFT reconstruction failed: error={reconstruction_error:.3e}"
    )


def test_fft_reconstruction_cropped_block():
    """Test that cropping then IFFT gives correct polynomial coefficients.
    
    This is the critical test: when we crop U(z) to P(z) = U[:r, :c] at sample
    points, then use IFFT to get P_coeffs, the recovered P_coeffs should correctly
    reconstruct P(z) at all sample points.
    """
    d, N, r, c = 8, 8, 4, 4
    rng = np.random.default_rng(99)
    
    # Generate full unitary polynomial
    def random_unitary(size):
        A = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
        Q, _ = np.linalg.qr(A)
        return Q
    
    R_list = [random_unitary(N) for _ in range(d + 1)]
    ell_list = rng.integers(1, N // 2 + 1, size=d).tolist()
    
    # Forward pass: compute full U(z)
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
    
    # Crop to get P(z) values
    P_vals_cropped = U_vals[:, :r, :c]  # (K, r, c)
    
    # IFFT to recover P polynomial coefficients
    all_coeffs = np.fft.fft(P_vals_cropped, axis=0) / K
    P_coeffs = all_coeffs[:d + 1]  # (d+1, r, c)
    
    # Reconstruct P(z) from coefficients using poly_eval_batch
    P_vals_reconstructed = poly_eval_batch(P_coeffs, z_pts)
    
    # Check that reconstruction matches the cropped values
    reconstruction_error = np.max(np.abs(P_vals_cropped - P_vals_reconstructed))
    
    print(f"\nCropped block FFT reconstruction (d={d}, N={N}, r={r}, c={c}, K={K}):")
    print(f"  Max reconstruction error: {reconstruction_error:.3e}")
    
    assert reconstruction_error < 1e-10, (
        f"Cropped FFT reconstruction failed: error={reconstruction_error:.3e}"
    )


def test_polynomial_evaluation_matches_manual():
    """Test that poly_eval_batch matches manual polynomial evaluation.
    
    Verifies: P(z) = P_0 + P_1*z + P_2*z^2 + ... + P_d*z^d
    """
    d, r, c = 5, 3, 2
    rng = np.random.default_rng(123)
    
    # Random polynomial coefficients
    P_coeffs = rng.standard_normal((d + 1, r, c)) + 1j * rng.standard_normal((d + 1, r, c))
    
    # Test points on unit circle
    K = 64
    z_pts = unit_circle_points(K)
    
    # Method 1: Use poly_eval_batch
    P_vals_batch = poly_eval_batch(P_coeffs, z_pts)
    
    # Method 2: Manual evaluation
    P_vals_manual = np.zeros((K, r, c), dtype=np.complex128)
    for ki in range(K):
        z = z_pts[ki]
        P_z = np.zeros((r, c), dtype=np.complex128)
        z_power = 1.0
        for l in range(d + 1):
            P_z += P_coeffs[l] * z_power
            z_power *= z
        P_vals_manual[ki] = P_z
    
    # Compare
    error = np.max(np.abs(P_vals_batch - P_vals_manual))
    
    print(f"\nPolynomial evaluation comparison (d={d}, r={r}, c={c}, K={K}):")
    print(f"  poly_eval_batch vs manual: {error:.3e}")
    
    assert error < 1e-12, f"poly_eval_batch doesn't match manual evaluation: {error:.3e}"


def test_ifft_fft_consistency():
    """Test that IFFT(samples) then FFT(coeffs) gives back original samples.
    
    This tests the core FFT assumption:
    - Sample polynomial P(z) at K points on unit circle
    - IFFT to get coefficients
    - Evaluate polynomial at same K points
    - Should get back original samples
    """
    d, r, c = 7, 4, 3
    K = max(8 * (d + 1), 512)
    
    rng = np.random.default_rng(456)
    
    # Create random polynomial coefficients
    P_coeffs_original = (
        rng.standard_normal((d + 1, r, c)) + 
        1j * rng.standard_normal((d + 1, r, c))
    )
    
    # Evaluate at K unit circle points
    z_pts = unit_circle_points(K)
    P_vals = poly_eval_batch(P_coeffs_original, z_pts)
    
    # IFFT to recover coefficients
    all_coeffs_recovered = np.fft.fft(P_vals, axis=0) / K
    P_coeffs_recovered = all_coeffs_recovered[:d + 1]
    
    # Check coefficient recovery
    coeff_error = np.max(np.abs(P_coeffs_original - P_coeffs_recovered))
    
    print(f"\nIFFT coefficient recovery (d={d}, r={r}, c={c}, K={K}):")
    print(f"  Coefficient recovery error: {coeff_error:.3e}")
    
    assert coeff_error < 1e-12, f"IFFT failed to recover coefficients: {coeff_error:.3e}"
    
    # FFT back to values
    P_vals_reconstructed = poly_eval_batch(P_coeffs_recovered, z_pts)
    
    # Check value reconstruction
    value_error = np.max(np.abs(P_vals - P_vals_reconstructed))
    
    print(f"  Value reconstruction error: {value_error:.3e}")
    
    assert value_error < 1e-12, f"FFT failed to reconstruct values: {value_error:.3e}"


def test_cropping_preserves_values():
    """Test that P(z) = U(z)[:r, :c] holds at all sample points.
    
    This is the fundamental property: when we crop a unitary U(z) to get P(z),
    the values of P(z) should exactly match the cropped block of U(z) at every
    evaluation point.
    """
    d, N, r, c = 6, 8, 3, 2
    rng = np.random.default_rng(789)
    
    # Generate full unitary
    def random_unitary(size):
        A = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
        Q, _ = np.linalg.qr(A)
        return Q
    
    R_list = [random_unitary(N) for _ in range(d + 1)]
    ell_list = rng.integers(1, N // 2 + 1, size=d).tolist()
    
    # Sample points
    K = 256
    z_pts = unit_circle_points(K)
    
    # Compute full U(z) at each point
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
    
    # Crop U(z) values
    P_vals_from_crop = U_vals[:, :r, :c]
    
    # Recover P coefficients via IFFT
    all_coeffs = np.fft.fft(P_vals_from_crop, axis=0) / K
    P_coeffs = all_coeffs[:d + 1]
    
    # Evaluate P(z) from coefficients
    P_vals_from_poly = poly_eval_batch(P_coeffs, z_pts)
    
    # These should be identical
    error = np.max(np.abs(P_vals_from_crop - P_vals_from_poly))
    
    print(f"\nCropping consistency (d={d}, N={N}, r={r}, c={c}, K={K}):")
    print(f"  Crop vs polynomial evaluation: {error:.3e}")
    
    assert error < 1e-11, (
        f"Cropped values don't match polynomial evaluation: {error:.3e}"
    )


if __name__ == "__main__":
    test_fft_reconstruction_full_unitary()
    test_fft_reconstruction_cropped_block()
    test_polynomial_evaluation_matches_manual()
    test_ifft_fft_consistency()
    test_cropping_preserves_values()
    print("\n✓ All FFT reconstruction tests passed!")
