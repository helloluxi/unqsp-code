"""Tests for Module 2: validation."""

import numpy as np
import pytest

from src.validate import validate
from tests.conftest import make_random_achievable


def test_validate_passes_on_achievable():
    """Achievable polynomial should pass validation."""
    P = make_random_achievable(d=4, N=4, r=2, c=2, seed=1)
    result = validate(P)
    assert result.passed, result.message


def test_validate_fails_on_oversized():
    """Polynomial with singular values > 1 should fail."""
    # Constant P = 2*I_2  =>  sigma = 2 > 1
    P = np.zeros((3, 2, 2), dtype=np.complex128)
    P[0] = 2.0 * np.eye(2)
    result = validate(P)
    assert not result.passed
    assert result.max_sigma > 1.0


def test_validate_passes_on_isometry():
    """Constant isometry (orthonormal columns) should pass."""
    # P = [I_2; 0] as a constant (degree-0) 4x2 isometry
    P = np.zeros((1, 4, 2), dtype=np.complex128)
    P[0, :2, :2] = np.eye(2)
    result = validate(P)
    assert result.passed
    assert result.max_sigma <= 1.0 + 1e-6


def test_validate_true_degree():
    """True degree should be reported correctly."""
    # degree-2 but leading coefficient is near-zero
    P = np.zeros((5, 2, 2), dtype=np.complex128)
    P[2] = 0.5 * np.eye(2)   # true degree = 2
    result = validate(P)
    assert result.true_degree == 2
