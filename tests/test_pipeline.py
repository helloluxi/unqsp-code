"""End-to-end pipeline tests: d=16, N=16.

Case A: P of shape (17, 16, 1)  — tall column vector
Case B: P of shape (17,  4, 4)  — square block in large ancilla
"""

import numpy as np
import pytest

ERROR_TOL = 1e-3   # generous tolerance: optimizer may not reach machine precision


def test_pipeline_case_a_column_vector(pipeline_case_a_result):
    """Case A: P is 16x1 (full-ancilla column vector isometry)."""
    P, result, N, D = pipeline_case_a_result

    assert len(result.R_list) == D + 1
    assert len(result.ell_list) == D

    # All R_k should be unitary
    for k, R in enumerate(result.R_list):
        err = np.linalg.norm(R @ R.conj().T - np.eye(N), "fro")
        assert err < 1e-8, f"R_list[{k}] not unitary: {err:.2e}"

    assert result.max_error < ERROR_TOL, (
        f"Case A reconstruction error {result.max_error:.4f} > {ERROR_TOL}"
    )


def test_pipeline_case_b_square_block(pipeline_case_b_result):
    """Case B: P is 4x4 partial isometry in a 16-dimensional ancilla."""
    P, result, N, D = pipeline_case_b_result

    assert len(result.R_list) == D + 1
    assert len(result.ell_list) == D

    for k, R in enumerate(result.R_list):
        err = np.linalg.norm(R @ R.conj().T - np.eye(N), "fro")
        assert err < 1e-8, f"R_list[{k}] not unitary: {err:.2e}"

    assert result.max_error < ERROR_TOL, (
        f"Case B reconstruction error {result.max_error:.4f} > {ERROR_TOL}"
    )
