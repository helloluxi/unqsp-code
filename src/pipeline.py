"""Module 6: Top-level U(N)-QSP compilation pipeline."""

from dataclasses import dataclass

import numpy as np

from .validate import validate, ValidationResult
from .complement import find_complement, ComplementResult
from .reduce import recursive_reduction, ReductionResult
from .verify import verify, VerificationResult


@dataclass
class PipelineResult:
    R_list: list[np.ndarray]
    ell_list: list[int]
    max_error: float
    validation: ValidationResult
    complement: ComplementResult
    reduction: ReductionResult
    verification: VerificationResult


def compile_unqsp(
    P_coeffs: np.ndarray,
    N: int,
    n_iters: int = 10_000,
    n_restarts: int = 3,
    verify_K: int = 256,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> PipelineResult:
    """Full U(N)-QSP compilation pipeline.

    Per paper lines 524-572: Q is computed via spectral factorization (numerically)
    to form the tall isometry Φ=[P;Q]. The R block that completes this to a square
    unitary is NOT needed. The recursion works on Φ to extract circuit parameters.

    Args:
        P_coeffs: shape (d+1, r, c), polynomial matrix to encode
        N:        ancilla dimension (power of 2, N >= r)
        n_iters:  Adam iterations for complement finding
        n_restarts: number of restarts for complement optimizer
        verify_K: number of test points for final verification
        rng:      random generator for reproducibility
        verbose:  print progress

    Returns:
        PipelineResult with all intermediate and final outputs
    """
    P_coeffs = np.asarray(P_coeffs, dtype=np.complex128)
    r = P_coeffs.shape[1]

    if N < r:
        raise ValueError(f"Ancilla dimension N={N} must be >= r={r}")
    if (N & (N - 1)) != 0:
        raise ValueError(f"N={N} must be a power of 2")

    # 1. Validate
    val = validate(P_coeffs, N=N)
    if not val.passed:
        raise ValueError(f"Validation failed: {val.message}")

    # 2. Find complement Q via numerical spectral factorization
    comp = find_complement(P_coeffs, N, n_iters=n_iters, n_restarts=n_restarts, verbose=verbose)

    # 3. Stack into tall isometry Φ based on fill mode
    # The reduction algorithm works on column isometries (tall matrices)
    # If we filled columns (row isometry), transpose to work with column isometry
    transposed = False
    if comp.fill_mode == "skip":
        # P already fills rows or columns, no complement needed
        if r == N:
            # Fills rows: P is (N, c), already column isometry
            Phi_coeffs = P_coeffs
        else:
            # Fills columns: P is (r, N), transpose to (N, r) column isometry
            Phi_coeffs = np.transpose(P_coeffs, (0, 2, 1))
            transposed = True
    elif comp.fill_mode == "rows":
        # Stack vertically: Φ = [P; Q], shape (d+1, N, c), column isometry
        Phi_coeffs = np.concatenate([P_coeffs, comp.Q_coeffs], axis=1)
    else:  # fill_mode == "cols"
        # Concatenate horizontally: Φ = [P Q], shape (d+1, r, N), row isometry
        # Transpose to (d+1, N, r) to make it column isometry
        Phi_coeffs = np.transpose(np.concatenate([P_coeffs, comp.Q_coeffs], axis=2), (0, 2, 1))
        transposed = True

    # 4. Recursive reduction on Φ (always column isometry)
    red = recursive_reduction(Phi_coeffs, rng=rng)

    # 5. Verify (handle transposed case)
    verif = verify(red.R_list, red.ell_list, P_coeffs, K=verify_K, transposed=transposed)

    return PipelineResult(
        R_list=red.R_list,
        ell_list=red.ell_list,
        max_error=verif.max_error,
        validation=val,
        complement=comp,
        reduction=red,
        verification=verif,
    )
