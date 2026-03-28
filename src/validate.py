"""Module 2: Validate a polynomial matrix P(z) for U(N)-QSP achievability."""

from dataclasses import dataclass

import numpy as np

from .utils import poly_eval_batch, unit_circle_points


@dataclass
class ValidationResult:
    passed: bool
    max_sigma: float          # max singular value over sampled unit circle
    min_sigma: float          # min singular value over sampled unit circle
    worst_angle: float        # angle (radians) where max_sigma occurs
    true_degree: int          # degree after stripping near-zero leading coefficients
    message: str


def validate(
    P_coeffs: np.ndarray,
    N: int | None = None,
    eps_tol: float = 1e-4,
    eps_zero: float = 1e-12,
) -> ValidationResult:
    """Check achievability of P(z) for U(N)-QSP per paper specification.

    Per paper lines 1-4 of PLAN.md:
    - If B(z) fills the row of full unitary (r=N): validate B(z)†B(z) = I for all unit z
    - If B(z) fills the column of full unitary (c=N): validate B(z)B(z)† = I for all unit z
    - Otherwise (fills neither): test all singular values ≤ 1 for all unit z

    Args:
        P_coeffs: shape (d+1, r, c), dtype complex128
        N:        ancilla dimension (power of 2). Enables strictness checking.
        eps_tol:  tolerance for singular value violations (default 1e-4)
        eps_zero: threshold to strip near-zero leading coefficient slices

    Returns:
        ValidationResult
    """
    P_coeffs = np.asarray(P_coeffs, dtype=np.complex128)
    d, r, c = P_coeffs.shape

    # --- degree check ---
    true_degree = 0
    for l in range(d - 1, -1, -1):
        if np.max(np.abs(P_coeffs[l])) > eps_zero:
            true_degree = l
            break

    # --- sample unit circle ---
    d_poly = d - 1
    K = max(4 * (d_poly + 1), 512)
    z_pts = unit_circle_points(K)
    P_vals = poly_eval_batch(P_coeffs, z_pts)  # (K, r, c)

    # --- compute singular values ---
    all_svd = [np.linalg.svd(P_vals[k], compute_uv=False) for k in range(K)]
    sigmas_max = np.array([s[0] for s in all_svd])
    sigmas_min = np.array([s[-1] if len(s) > 0 else 0.0 for s in all_svd])
    max_sigma = float(np.max(sigmas_max))
    min_sigma = float(np.min(sigmas_min))
    worst_k = int(np.argmax(sigmas_max))
    worst_angle = float(2 * np.pi * worst_k / K)

    # --- determine strictness mode per paper ---
    fills_rows = (N is not None and r == N)
    fills_cols = (N is not None and c == N)

    if fills_rows:
        # B(z) fills rows → must satisfy B(z)†B(z) = I_c (column isometry)
        # Check by verifying all singular values = 1
        if max_sigma > 1.0 + eps_tol:
            msg = (
                f"FAIL (fills rows): σ_max = {max_sigma:.6f} > 1 at θ = {worst_angle:.4f} rad. "
                f"Must satisfy B†B = I."
            )
            return ValidationResult(False, max_sigma, min_sigma, worst_angle, true_degree, msg)
        if min_sigma < 1.0 - eps_tol:
            worst_min_k = int(np.argmin(sigmas_min))
            worst_min_angle = float(2 * np.pi * worst_min_k / K)
            msg = (
                f"FAIL (fills rows): σ_min = {min_sigma:.6f} < 1 at θ = {worst_min_angle:.4f} rad. "
                f"Must satisfy B†B = I."
            )
            return ValidationResult(False, max_sigma, min_sigma, worst_min_angle, true_degree, msg)
        msg = f"PASS (fills rows, B†B=I): σ ∈ [{min_sigma:.6f}, {max_sigma:.6f}] ≈ 1 (degree={true_degree})"
    elif fills_cols:
        # B(z) fills columns → must satisfy B(z)B(z)† = I_r (row isometry)
        # Check by verifying all singular values = 1
        if max_sigma > 1.0 + eps_tol:
            msg = (
                f"FAIL (fills cols): σ_max = {max_sigma:.6f} > 1 at θ = {worst_angle:.4f} rad. "
                f"Must satisfy BB† = I."
            )
            return ValidationResult(False, max_sigma, min_sigma, worst_angle, true_degree, msg)
        if min_sigma < 1.0 - eps_tol:
            worst_min_k = int(np.argmin(sigmas_min))
            worst_min_angle = float(2 * np.pi * worst_min_k / K)
            msg = (
                f"FAIL (fills cols): σ_min = {min_sigma:.6f} < 1 at θ = {worst_min_angle:.4f} rad. "
                f"Must satisfy BB† = I."
            )
            return ValidationResult(False, max_sigma, min_sigma, worst_min_angle, true_degree, msg)
        msg = f"PASS (fills cols, BB†=I): σ ∈ [{min_sigma:.6f}, {max_sigma:.6f}] ≈ 1 (degree={true_degree})"
    else:
        # General case: fills neither row nor column → all singular values ≤ 1
        if max_sigma > 1.0 + eps_tol:
            msg = (
                f"FAIL: σ_max = {max_sigma:.6f} > 1 at θ = {worst_angle:.4f} rad "
                f"(sample {worst_k}/{K})"
            )
            return ValidationResult(False, max_sigma, min_sigma, worst_angle, true_degree, msg)
        msg = f"PASS: σ_max = {max_sigma:.6f} ≤ 1 (degree={true_degree})"

    return ValidationResult(True, max_sigma, min_sigma, worst_angle, true_degree, msg)
