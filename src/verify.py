"""Module 5: Forward simulation to verify circuit parameters."""

from dataclasses import dataclass

import numpy as np

from .utils import poly_eval_batch, unit_circle_points


@dataclass
class VerificationResult:
    max_error: float
    mean_error: float
    errors: np.ndarray   # shape (K,), per-sample Frobenius errors


def verify(
    R_list: list[np.ndarray],
    ell_list: list[int],
    P_coeffs: np.ndarray,
    K: int = 256,
    transposed: bool = False,
) -> VerificationResult:
    """Forward-simulate the U(N)-QSP circuit and compare to target P.

    Args:
        R_list:   d+1 unitaries, each (N, N) complex128
        ell_list: d integers, projector cutoffs
        P_coeffs: shape (d+1, r, c), target polynomial matrix
        K:        number of test points on unit circle
        transposed: if True, circuit was built on P^T, so extract P^T and transpose

    Returns:
        VerificationResult
    """
    d = len(R_list) - 1
    N = R_list[0].shape[0]
    r, c = P_coeffs.shape[1], P_coeffs.shape[2]

    z_pts = unit_circle_points(K)
    P_target = poly_eval_batch(P_coeffs, z_pts)  # (K, r, c)

    errors = np.zeros(K, dtype=np.float64)
    for ki in range(K):
        z = z_pts[ki]
        M = R_list[0].copy()
        for k in range(1, d + 1):
            ell = ell_list[k - 1]
            # C_k = block_diag(z * I_ell, I_{N-ell})
            C_k = np.eye(N, dtype=np.complex128)
            C_k[:ell, :ell] *= z
            M = R_list[k] @ C_k @ M
        
        if transposed:
            # Circuit was built on P^T, so M encodes P^T in top-left (c, r) block
            # Extract and transpose to get P
            P_circuit = M[:c, :r].T
        else:
            # Circuit encodes P in top-left (r, c) block
            P_circuit = M[:r, :c]
        
        errors[ki] = np.linalg.norm(P_circuit - P_target[ki], "fro")

    return VerificationResult(
        max_error=float(np.max(errors)),
        mean_error=float(np.mean(errors)),
        errors=errors,
    )
