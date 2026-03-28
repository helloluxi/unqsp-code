"""Module 4: Recursive degree reduction to extract R_list and ell_list."""

from dataclasses import dataclass

import numpy as np

from .utils import complete_to_unitary, build_projector_matrix, poly_eval_batch, unit_circle_points


@dataclass
class ReductionResult:
    R_list: list[np.ndarray]   # d+1 unitaries, each (N, N) complex128
    ell_list: list[int]        # d integers


def _numerical_rank(s: np.ndarray, rank_tol: float = 1e-10) -> int:
    """Compute numerical rank from singular values using relative threshold.

    A singular value is considered nonzero if it exceeds rank_tol * s[0].
    The default 1e-10 is robust against accumulated floating-point drift
    across multiple reduction steps while still detecting genuine rank.
    """
    if len(s) == 0 or s[0] == 0:
        return 0
    tol = s[0] * rank_tol
    return int(np.sum(s > tol))


def recursive_reduction(
    Phi_coeffs: np.ndarray,
    rng: np.random.Generator | None = None,
    orth_warn_tol: float = 1e-3,
) -> ReductionResult:
    """Extract R_list and ell_list from tall isometry Phi_coeffs via degree reduction.

    Args:
        Phi_coeffs: shape (d+1, N, c), dtype complex128. Phi_coeffs[l] = l-th coefficient.
        rng: random generator for unitary completion (reproducibility)
        orth_warn_tol: warn if col(Phi_lead) and col(Phi_const) are not orthogonal

    Returns:
        ReductionResult with R_list (length d+1) and ell_list (length d)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    Phi_coeffs = np.asarray(Phi_coeffs, dtype=np.complex128)
    d, N, c = Phi_coeffs.shape[0] - 1, Phi_coeffs.shape[1], Phi_coeffs.shape[2]

    # Estimate isometry residual to calibrate adaptive rank threshold.
    # Singular values below O(sqrt(residual)) in Phi's coefficients are
    # artifacts of imperfect complementation, not genuine rank.
    K_check = max(4 * (d + 1), 256)
    z_check = unit_circle_points(K_check)
    Phi_check = poly_eval_batch(Phi_coeffs, z_check)
    Ic = np.eye(c, dtype=np.complex128)
    iso_res = max(
        np.linalg.norm(Phi_check[k].conj().T @ Phi_check[k] - Ic, "fro")
        for k in range(K_check)
    )
    # Adaptive rank tolerance: sqrt(residual) with safety factor, floor at 1e-10
    rank_tol = max(np.sqrt(iso_res) * 10, 1e-10)

    # Mutable working copy
    Phi = [Phi_coeffs[l].copy() for l in range(d + 1)]

    R_list = [None] * (d + 1)
    ell_list = [0] * d

    I_N = np.eye(N, dtype=np.complex128)

    for step in range(d, 0, -1):
        Phi_lead = Phi[step]   # (N, c)
        Phi_const = Phi[0]     # (N, c)

        # --- SVD of leading and constant coefficient ---
        U_lead, s_lead, _ = np.linalg.svd(Phi_lead, full_matrices=True)
        U_const, s_const, _ = np.linalg.svd(Phi_const, full_matrices=True)

        r_lead = _numerical_rank(s_lead, rank_tol)
        r_const = _numerical_rank(s_const, rank_tol)

        # --- ensure orthogonality by projecting col(Phi_0) out of col(Phi_d) ---
        # The isometry condition guarantees Phi_0†Phi_d ≈ 0, but numerical
        # complement errors can cause column space overlap.  Project out the
        # lead column space from the constant columns so orthogonality holds
        # by construction, then recompute the effective constant rank.
        lead_basis = U_lead[:, :r_lead]
        const_cols = U_const[:, :r_const].copy()
        const_cols -= lead_basis @ (lead_basis.conj().T @ const_cols)
        # Re-orthogonalize and compute effective rank of projected constant space
        if r_const > 0:
            Q_c, R_c = np.linalg.qr(const_cols, mode="reduced")
            s_proj = np.abs(np.diag(R_c))
            r_const = int(np.sum(s_proj > s_proj[0] * rank_tol)) if s_proj[0] > 0 else 0
            const_basis = Q_c[:, :r_const]
        else:
            const_basis = np.zeros((N, 0), dtype=np.complex128)

        # --- orthogonality check (on cleaned column spaces) ---
        if r_lead > 0 and r_const > 0:
            cross = np.linalg.norm(
                lead_basis.conj().T @ const_basis, "fro"
            )
            if cross > orth_warn_tol:
                raise ValueError(
                    f"Step {step}: col(Phi_lead) ⊥ col(Phi_const) violation = {cross:.3e}. "
                    "Complement is inaccurate. Per paper requirement, orthogonality must be maintained. "
                    "Increase complement optimization iterations (n_iters, lbfgs_iters) or use better initial data."
                )

        # --- choose ℓ = r_lead ---
        ell = r_lead
        ell_list[step - 1] = ell

        if r_lead + r_const > N:
            raise ValueError(
                f"Step {step}: r_lead ({r_lead}) + r_const ({r_const}) > N ({N}). "
                "Isometry property violated. The input Phi is not a valid column isometry."
            )

        # --- construct R_step ---
        # Build basis: [lead_basis | null_fill | const_basis]
        # arranged so first ell cols = col(Phi_lead), last r_const cols ⊥ col(Phi_lead)
        n_fill = N - r_lead - r_const
        if n_fill > 0:
            rand_fill = rng.standard_normal((N, n_fill)) + 1j * rng.standard_normal((N, n_fill))
            rand_fill -= lead_basis @ (lead_basis.conj().T @ rand_fill)
            if r_const > 0:
                rand_fill -= const_basis @ (const_basis.conj().T @ rand_fill)
            B = np.concatenate([lead_basis, rand_fill, const_basis], axis=1)
        else:
            B = np.concatenate([lead_basis, const_basis], axis=1)

        R_step, _ = np.linalg.qr(B, mode="complete")
        R_list[step] = R_step

        # --- degree-reduction update ---
        R_inv = R_step.conj().T
        Pi = build_projector_matrix(ell, N)
        I_minus_Pi = I_N - Pi

        new_Phi = [None] * step
        for l in range(step):
            new_Phi[l] = Pi @ (R_inv @ Phi[l + 1]) + I_minus_Pi @ (R_inv @ Phi[l])

        Phi = new_Phi

    # --- Final step: R_0 from constant isometry Phi[0] ---
    # Re-orthogonalize only here to correct accumulated floating-point drift.
    # Phi[0] is always (N, c) tall matrix since we transpose before reduction if needed
    R_list[0] = complete_to_unitary(Phi[0], rng=rng)

    return ReductionResult(R_list, ell_list)
