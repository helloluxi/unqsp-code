"""Tests for Module 4: recursive degree reduction."""

import numpy as np
import pytest

from src.reduce import recursive_reduction
from src.verify import verify


def test_reduction_unitaries_are_unitary(reduction_4_4_2_2_seed30):
    """All R_k in R_list should be unitary."""
    P, Phi, result = reduction_4_4_2_2_seed30
    N = 4
    for k, R in enumerate(result.R_list):
        assert R.shape == (N, N), f"R_list[{k}] has wrong shape"
        err = np.linalg.norm(R @ R.conj().T - np.eye(N), "fro")
        assert err < 1e-10, f"R_list[{k}] not unitary: err={err:.2e}"


def test_reduction_ell_list_length(reduction_5_4_2_2_seed31):
    """ell_list should have length d."""
    d = 5
    P, Phi, result = reduction_5_4_2_2_seed31
    assert len(result.ell_list) == d
    assert len(result.R_list) == d + 1


def test_reduction_ell_in_valid_range(reduction_4_8_4_2_seed32):
    """All ell values should be in [0, N]."""
    N = 8
    P, Phi, result = reduction_4_8_4_2_seed32
    for ell in result.ell_list:
        assert 0 <= ell <= N, f"ell={ell} out of range [0, {N}]"
