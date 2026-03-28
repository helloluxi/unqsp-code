"""Tests for Module 3: complement finding."""

import numpy as np
import pytest

from src.complement import find_complement
from src.utils import poly_eval_batch, unit_circle_points
from tests.conftest import make_random_achievable


def isometry_residual(P_coeffs, Q_coeffs, K=256):
    """Max ||I - P†P - Q†Q||_F over K unit-circle points."""
    z_pts = unit_circle_points(K)
    P_vals = poly_eval_batch(P_coeffs, z_pts)
    c = P_coeffs.shape[2]
    Ic = np.eye(c, dtype=np.complex128)

    if Q_coeffs.shape[1] == 0:
        residuals = [np.linalg.norm(Ic - P_vals[k].conj().T @ P_vals[k], "fro")
                     for k in range(K)]
    else:
        Q_vals = poly_eval_batch(Q_coeffs, z_pts)
        residuals = [
            np.linalg.norm(
                Ic - P_vals[k].conj().T @ P_vals[k] - Q_vals[k].conj().T @ Q_vals[k],
                "fro"
            )
            for k in range(K)
        ]
    return float(np.max(residuals))


def test_complement_2x2_in_4():
    """Small 2x2 P in N=4 ancilla: Q should complete the isometry."""
    P = make_random_achievable(d=4, N=4, r=2, c=2, seed=10)
    result = find_complement(P, N=4, n_iters=5_000, n_restarts=2)
    assert result.Q_coeffs.shape == (5, 2, 2)
    res = isometry_residual(P, result.Q_coeffs)
    assert res < 0.05, f"Isometry residual too large: {res:.4f}"


def test_complement_skip_when_N_equals_r():
    """When N == r (P fills entire ancilla), Q must be empty and P itself is unitary-valued."""
    # Construct a unitary-valued P: use constant unitary (d=0)
    N = 4
    rng = np.random.default_rng(99)
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    U, _ = np.linalg.qr(A)
    P = np.zeros((1, N, N), dtype=np.complex128)
    P[0] = U
    result = find_complement(P, N=N, n_iters=100, n_restarts=1)
    assert result.Q_coeffs.shape == (1, 0, N)


def test_complement_column_vector():
    """P of shape (d+1, r, 1): Q should be an (r,)-dimensional complement."""
    P = make_random_achievable(d=6, N=8, r=4, c=1, seed=20)
    result = find_complement(P, N=8, n_iters=5_000, n_restarts=2)
    assert result.Q_coeffs.shape == (7, 4, 1)
    res = isometry_residual(P, result.Q_coeffs)
    assert res < 0.05, f"Isometry residual too large: {res:.4f}"
