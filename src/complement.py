"""Module 3: Find polynomial complement Q via PyTorch optimization."""

from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from .utils import unit_circle_points


@dataclass
class ComplementResult:
    Q_coeffs: np.ndarray   # shape (d+1, N-r, c) if fill_rows, (d+1, r, N-c) if fill_cols; empty if skipped
    final_loss: float
    max_residual: float
    fill_mode: str  # "skip" (fills rows/cols already), "rows" (stacked vertically), or "cols" (concatenated horizontally)


def _eval_batch_torch(
    coeffs: torch.Tensor, z_pts: torch.Tensor
) -> torch.Tensor:
    """Evaluate polynomial at K points.

    Args:
        coeffs: (d+1, rows, c) complex tensor
        z_pts:  (K,) complex tensor

    Returns:
        (K, rows, c) complex tensor
    """
    d = coeffs.shape[0] - 1
    K = z_pts.shape[0]
    V = torch.ones(K, d + 1, dtype=torch.cdouble, device=coeffs.device)
    for l in range(1, d + 1):
        V[:, l] = V[:, l - 1] * z_pts
    return torch.einsum("kl,lrc->krc", V, coeffs)


def _compute_loss(Q_coeffs, P_vals, z_pts, Ic, alpha: float = 0.0):
    """Isometry residual loss.

    alpha=0  → mean squared Frobenius.
    alpha>0  → log-mean-exp soft-max approximation (fixed alpha, consistent
               landscape — required for L-BFGS stability).
               alpha should be calibrated so alpha × max(norms_sq) ≈ 10.
    """
    Q_vals = _eval_batch_torch(Q_coeffs, z_pts)
    PtP = torch.einsum("kri,krj->kij", P_vals.conj(), P_vals)
    QtQ = torch.einsum("kri,krj->kij", Q_vals.conj(), Q_vals)
    R = Ic.unsqueeze(0) - PtP - QtQ
    norms_sq = torch.sum(torch.abs(R) ** 2, dim=(-1, -2))   # (K,)
    if alpha <= 0.0:
        return torch.mean(norms_sq)
    logK = torch.log(torch.tensor(float(norms_sq.shape[0]),
                                   dtype=torch.double, device=norms_sq.device))
    return (torch.logsumexp(alpha * norms_sq, dim=0) - logK) / alpha


def _spectral_factor_init(
    P_vals_np: np.ndarray,
    d: int,
    N_compl: int,
    c: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Better-than-random initial Q via pointwise Cholesky + polynomial fit.

    At each unit-circle sample z_k, computes a Cholesky factor of
    I - P(z_k)†P(z_k) and uses IFFT to recover a polynomial Q whose values
    approximate this factorization. This provides a warm-start that's much
    closer to the true spectral factor than random initialization.

    Returns Q_init of shape (d+1, N_compl, c), dtype complex128.
    """
    K = P_vals_np.shape[0]
    Ic = np.eye(c, dtype=np.complex128)

    # Sample a random N_compl x N_compl unitary for rotating the Cholesky factors
    # consistently across samples (avoids phase discontinuities)
    Q_vals = np.zeros((K, N_compl, c), dtype=np.complex128)
    n_use = min(N_compl, c)  # G is cxc so has at most c eigenvectors
    for k in range(K):
        G = Ic - P_vals_np[k].conj().T @ P_vals_np[k]
        # G should be PSD; clip small negative eigenvalues
        w, v = np.linalg.eigh(G)
        w = np.maximum(w, 0)
        # Take the top n_use eigenvalues (largest); remaining rows stay zero
        idx = np.argsort(w)[::-1][:n_use]
        Q_vals[k, :n_use, :] = (v[:, idx] * np.sqrt(w[idx])).T  # (n_use, c)

    # Polynomial fit via FFT (truncate to degree d)
    Q_coeffs = np.fft.fft(Q_vals, axis=0)[:d + 1] / K
    return Q_coeffs


def _run_adam(
    P_vals: torch.Tensor,
    d: int,
    N_compl: int,
    c: int,
    n_iters: int,
    seed: int,
    device: torch.device,
    z_pts: torch.Tensor,
    Q_init: np.ndarray | None = None,
    pbar: tqdm | None = None,
) -> tuple[torch.Tensor, float]:
    """Single Adam optimization run. Returns (Q_coeffs_tensor, final_loss)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    if Q_init is not None:
        Q_coeffs = torch.tensor(Q_init, dtype=torch.cdouble, device=device).requires_grad_(True)
    else:
        Q_re = torch.randn(d + 1, N_compl, c, generator=gen, dtype=torch.double, device=device)
        Q_im = torch.randn(d + 1, N_compl, c, generator=gen, dtype=torch.double, device=device)
        Q_coeffs = torch.view_as_complex(
            torch.stack([Q_re, Q_im], dim=-1).contiguous()
        ) * 0.01
        Q_coeffs = Q_coeffs.requires_grad_(True)

    lr_high, lr_low = 1e-3, 1e-5
    switch_iter = int(0.7 * n_iters)
    optimizer = torch.optim.Adam([Q_coeffs], lr=lr_high)
    Ic = torch.eye(c, dtype=torch.cdouble, device=device)

    for it in range(n_iters):
        if it == switch_iter:
            for pg in optimizer.param_groups:
                pg["lr"] = lr_low
        optimizer.zero_grad()
        loss = _compute_loss(Q_coeffs, P_vals, z_pts, Ic)
        loss.backward()
        optimizer.step()
        if pbar is not None and it % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.2e}", refresh=False)
            pbar.update(100)

    return Q_coeffs.detach(), float(loss.item())


def _run_lbfgs(
    Q_init: torch.Tensor,
    P_vals: torch.Tensor,
    z_pts: torch.Tensor,
    n_iters: int,
    pbar: tqdm | None = None,
) -> tuple[torch.Tensor, float]:
    """L-BFGS refinement starting from Q_init. Converges to tighter tolerances than Adam."""
    device = Q_init.device
    c = P_vals.shape[2]
    Ic = torch.eye(c, dtype=torch.cdouble, device=device)
    Q_coeffs = Q_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [Q_coeffs], lr=1.0, max_iter=20, tolerance_grad=1e-12, tolerance_change=1e-14,
        history_size=50, line_search_fn="strong_wolfe",
    )
    final_loss = [float("inf")]

    # Calibrate alpha once so alpha × max(norms_sq) ≈ 10 at the start.
    # This keeps the soft-max meaningful without changing the landscape per step.
    with torch.no_grad():
        Q_vals0 = _eval_batch_torch(Q_coeffs.detach(), z_pts)
        PtP0 = torch.einsum("kri,krj->kij", P_vals.conj(), P_vals)
        QtQ0 = torch.einsum("kri,krj->kij", Q_vals0.conj(), Q_vals0)
        R0 = Ic.unsqueeze(0) - PtP0 - QtQ0
        max0 = torch.sum(torch.abs(R0) ** 2, dim=(-1, -2)).max().clamp(min=1e-30).item()
        alpha = 10.0 / max0

    for _ in range(n_iters):
        def closure():
            optimizer.zero_grad()
            loss = _compute_loss(Q_coeffs, P_vals, z_pts, Ic, alpha=alpha)
            loss.backward()
            final_loss[0] = float(loss.item())
            return loss
        optimizer.step(closure)
        if pbar is not None:
            pbar.set_postfix(loss=f"{final_loss[0]:.2e}", refresh=False)
            pbar.update(1)
        if final_loss[0] < 1e-20:
            break

    return Q_coeffs.detach(), final_loss[0]


def find_complement(
    P_coeffs: np.ndarray,
    N: int,
    n_iters: int = 10_000,
    n_restarts: int = 3,
    lbfgs_iters: int = 200,
    eps_complete: float = 1e-2,
    K_factor: int = 8,
    verbose: bool = False,
) -> ComplementResult:
    """Find polynomial complement Q such that [P; Q] is an isometry on the unit circle.

    Per paper PLAN.md lines 6-9:
    - If B(z) fills the row or column of full unitary: skip (no complement needed)
    - Otherwise: choose to fill row or column depending on which is closer to filling,
      then find B_1(z) via numerical optimization such that:
        * If filling rows: B_1(z)†B_1(z) = I - B(z)†B(z), then stack [B(z); B_1(z)]
        * If filling cols: B_1(z)B_1(z)† = I - B(z)B(z)†, then concatenate [B(z) B_1(z)]

    Uses Adam (n_iters iterations, n_restarts restarts) followed by L-BFGS refinement
    (lbfgs_iters outer iterations, each with up to 20 line-search steps). L-BFGS
    converges to much tighter tolerances than Adam and is essential for avoiding
    large singular-vector misalignment in the reduction step.

    Args:
        P_coeffs: shape (d+1, r, c), complex128
        N: ancilla dimension (power of 2, N >= r)
        n_iters: Adam iterations per restart
        n_restarts: number of independent restarts
        lbfgs_iters: outer L-BFGS iterations after best Adam result
        eps_complete: max residual threshold for warning
        K_factor: K = max(K_factor*(d+1), 512) sample points

    Returns:
        ComplementResult
    """
    P_coeffs = np.asarray(P_coeffs, dtype=np.complex128)
    d, r, c = P_coeffs.shape[0] - 1, P_coeffs.shape[1], P_coeffs.shape[2]
    
    # Per paper: if fills row (r=N) or column (c=N), skip complement
    fills_rows = (r == N)
    fills_cols = (c == N)
    
    if fills_rows or fills_cols:
        # No complement needed - B(z) already fills the unitary
        Q_empty = np.zeros((d + 1, 0, c), dtype=np.complex128)
        K = max(K_factor * (d + 1), 512)
        from .utils import poly_eval_batch
        z_pts_np = unit_circle_points(K)
        P_vals = poly_eval_batch(P_coeffs, z_pts_np)
        Ic = np.eye(c, dtype=np.complex128)
        Ir = np.eye(r, dtype=np.complex128)
        
        # Verify isometry condition
        residuals = []
        for k in range(K):
            if fills_rows:
                # Should satisfy P†P = I_c
                res = np.linalg.norm(Ic - P_vals[k].conj().T @ P_vals[k], "fro")
            else:
                # Should satisfy PP† = I_r
                res = np.linalg.norm(Ir - P_vals[k] @ P_vals[k].conj().T, "fro")
            residuals.append(res)
        residuals = np.array(residuals)
        return ComplementResult(Q_empty, float(np.mean(residuals**2)), float(np.max(residuals)), "skip")
    
    # Choose whether to fill rows or columns based on which is closer
    # Closer to filling rows if N - r < N - c, i.e., r > c
    fill_rows = (r >= c)
    
    if fill_rows:
        # Fill rows: find Q such that [P; Q] is column isometry, i.e., [P; Q]†[P; Q] = I
        N_compl = N - r
        P_work = P_coeffs  # shape (d+1, r, c)
        work_r, work_c = r, c
    else:
        # Fill columns: find Q such that [P Q] is row isometry, i.e., [P Q][P Q]† = I
        # Transpose the problem: work with P^T to find Q^T, then transpose back
        # P^T has shape (d+1, c, r), and we'll find Q^T of shape (d+1, N-c, r)
        N_compl = N - c
        P_work = np.transpose(P_coeffs, (0, 2, 1))  # shape (d+1, c, r)
        work_r, work_c = c, r

    # Special case: no complement needed (shouldn't reach here given above checks)
    if N_compl == 0:
        Q_empty = np.zeros((d + 1, 0, c), dtype=np.complex128)
        K = max(K_factor * (d + 1), 512)
        from .utils import poly_eval_batch
        z_pts_np = unit_circle_points(K)
        P_vals = poly_eval_batch(P_coeffs, z_pts_np)
        Ic = np.eye(c, dtype=np.complex128)
        residuals = np.array([
            np.linalg.norm(Ic - P_vals[k].conj().T @ P_vals[k], "fro")
            for k in range(K)
        ])
        return ComplementResult(Q_empty, float(np.mean(residuals**2)), float(np.max(residuals)), "skip")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    K = max(K_factor * (d + 1), 512)
    z_pts_np = unit_circle_points(K)
    z_pts = torch.tensor(z_pts_np, dtype=torch.cdouble, device=device)

    P_t = torch.tensor(P_work, dtype=torch.cdouble, device=device)
    P_vals = _eval_batch_torch(P_t, z_pts).detach()

    # Spectral-factor warm start: pointwise Cholesky + IFFT → polynomial init
    rng_np = np.random.default_rng(42)
    Q_sf_init = _spectral_factor_init(P_vals.cpu().numpy(), d, N_compl, work_c, rng_np)

    # --- Adam phase: multiple restarts ---
    best_Q = None
    best_loss = float("inf")
    for restart in range(n_restarts):
        q_init = Q_sf_init if restart == 0 else None
        pbar = tqdm(total=n_iters, desc=f"Adam restart {restart+1}/{n_restarts}",
                    leave=True) if verbose else None
        Q_t, loss = _run_adam(P_vals, d, N_compl, work_c, n_iters, seed=restart * 137,
                               device=device, z_pts=z_pts, Q_init=q_init, pbar=pbar)
        if pbar is not None:
            pbar.close()
        if loss < best_loss:
            best_loss = loss
            best_Q = Q_t

    # --- L-BFGS refinement phase ---
    if lbfgs_iters > 0:
        pbar = tqdm(total=lbfgs_iters, desc="L-BFGS", leave=True) if verbose else None
        best_Q, best_loss = _run_lbfgs(best_Q, P_vals, z_pts, lbfgs_iters, pbar=pbar)
        if pbar is not None:
            pbar.close()

    Q_coeffs = best_Q.cpu().numpy()
    
    # If we filled columns (transposed), transpose Q back
    if not fill_rows:
        # Q_coeffs is currently (d+1, N-c, r), transpose to (d+1, r, N-c)
        Q_coeffs = np.transpose(Q_coeffs, (0, 2, 1))

    # Trim near-zero leading coefficient
    if np.max(np.abs(Q_coeffs[-1])) < 1e-12:
        Q_coeffs[-1] = 0.0

    # Compute max residual on unit circle
    from .utils import poly_eval_batch
    z_pts_np2 = unit_circle_points(K)
    P_vals_np = poly_eval_batch(P_coeffs, z_pts_np2)
    V = np.ones((K, d + 1), dtype=np.complex128)
    for l in range(1, d + 1):
        V[:, l] = V[:, l - 1] * z_pts_np2
    Q_vals_np = np.einsum("kl,lrc->krc", V, Q_coeffs)
    
    # Verify the correct isometry condition based on fill_rows
    if fill_rows:
        # [P; Q] should satisfy [P; Q]†[P; Q] = I_c
        Ic_np = np.eye(c, dtype=np.complex128)
        residuals = np.array([
            np.linalg.norm(
                Ic_np - P_vals_np[k].conj().T @ P_vals_np[k] - Q_vals_np[k].conj().T @ Q_vals_np[k],
                "fro"
            )
            for k in range(K)
        ])
    else:
        # [P Q] should satisfy [P Q][P Q]† = I_r
        Ir_np = np.eye(r, dtype=np.complex128)
        residuals = np.array([
            np.linalg.norm(
                Ir_np - P_vals_np[k] @ P_vals_np[k].conj().T - Q_vals_np[k] @ Q_vals_np[k].conj().T,
                "fro"
            )
            for k in range(K)
        ])
    max_res = float(np.max(residuals))

    if max_res > eps_complete:
        raise ValueError(
            f"Complement residual {max_res:.2e} > {eps_complete:.2e}. "
            "Complementation failed to achieve required accuracy. "
            "Increase n_iters, lbfgs_iters, or K_factor to improve convergence."
        )

    fill_mode = "rows" if fill_rows else "cols"
    return ComplementResult(Q_coeffs, best_loss, max_res, fill_mode)
