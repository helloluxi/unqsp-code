"""Shared helpers: polynomial evaluation, unitary completion, Vandermonde batch eval."""

import numpy as np


def poly_eval_horner(coeffs: np.ndarray, z: complex) -> np.ndarray:
    """Evaluate polynomial matrix at scalar z using Horner's method.

    Args:
        coeffs: shape (d+1, r, c), coeffs[l] is coefficient of z^l
        z: evaluation point

    Returns:
        shape (r, c) matrix P(z)
    """
    d = len(coeffs) - 1
    result = coeffs[d].copy()
    for l in range(d - 1, -1, -1):
        result = result * z + coeffs[l]
    return result


def poly_eval_batch(coeffs: np.ndarray, z_points: np.ndarray) -> np.ndarray:
    """Evaluate polynomial matrix at K points via Vandermonde contraction.

    Args:
        coeffs: shape (d+1, r, c)
        z_points: shape (K,) complex array

    Returns:
        shape (K, r, c) array of P(z_k)
    """
    d = len(coeffs) - 1
    K = len(z_points)
    # Build Vandermonde: V[k, l] = z_k^l
    V = np.ones((K, d + 1), dtype=np.complex128)
    for l in range(1, d + 1):
        V[:, l] = V[:, l - 1] * z_points
    # Contract: result[k] = sum_l V[k,l] * coeffs[l]
    return np.einsum("kl,lrc->krc", V, coeffs)


def complete_to_unitary(A: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Complete a (N, m) approximately-isometric matrix to a (N, N) unitary.

    Re-orthogonalizes A's columns (to guarantee unitarity) while preserving
    column directions via phase correction. The first m columns of the result
    point in the same directions as A's columns up to small numerical errors.

    Args:
        A: shape (N, m), m <= N, approximately orthonormal columns
        rng: optional random generator for reproducibility

    Returns:
        shape (N, N) unitary U with U[:, :m] ≈ A (same directions)
    """
    if rng is None:
        rng = np.random.default_rng()
    N, m = A.shape

    # Thin QR of A → orthonormal Q0 spanning col(A)
    Q0, R0 = np.linalg.qr(A, mode="reduced")
    # Fix column phases: LAPACK QR can give arbitrary complex phases in R0's diagonal.
    # Multiplying Q0[:, k] by conj(phase(R0[k,k])) aligns it with A[:, k]'s direction.
    r_diag = np.diag(R0)
    phases = r_diag / np.abs(r_diag)
    Q0 = Q0 * phases.conj()  # broadcast: each col k multiplied by conj(phases[k])

    if m == N:
        return Q0

    # Build N-m orthonormal fill columns orthogonal to Q0
    rand_fill = rng.standard_normal((N, N - m)) + 1j * rng.standard_normal((N, N - m))
    rand_fill = rand_fill - Q0 @ (Q0.conj().T @ rand_fill)
    Q_fill, _ = np.linalg.qr(rand_fill, mode="reduced")
    return np.concatenate([Q0, Q_fill], axis=1)


def build_projector_matrix(ell: int, N: int) -> np.ndarray:
    """Build Π = diag(1,...,1, 0,...,0) with ell ones, size NxN."""
    Pi = np.zeros((N, N), dtype=np.complex128)
    Pi[:ell, :ell] = np.eye(ell, dtype=np.complex128)
    return Pi


def unit_circle_points(K: int) -> np.ndarray:
    """Return K equally-spaced points on the unit circle."""
    thetas = 2 * np.pi * np.arange(K) / K
    return np.exp(1j * thetas)
