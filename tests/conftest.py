"""Shared fixtures: generate random achievable polynomial matrices."""

import numpy as np
import pytest
from functools import lru_cache

from src.utils import unit_circle_points, build_projector_matrix
from src.complement import find_complement
from src.reduce import recursive_reduction
from src.pipeline import compile_unqsp


def make_random_achievable(d: int, N: int, r: int, c: int, seed: int = 0) -> np.ndarray:
    """Generate a random achievable polynomial matrix P of shape (d+1, r, c).

    Strategy: draw random R_list ∈ U(N) and ell_list, run the forward circuit to
    obtain a full (NxN) polynomial unitary, then extract the top-left (r, c) block.
    The extracted block is always achievable because it is a sub-block of a unitary.

    Args:
        d:    polynomial degree
        N:    ancilla dimension (power of 2)
        r:    number of rows of P
        c:    number of columns of P (c <= r <= N)
        seed: random seed

    Returns:
        P_coeffs: shape (d+1, r, c), complex128
    """
    rng = np.random.default_rng(seed)

    # Random unitaries in U(N)
    def random_unitary(size):
        A = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
        Q, _ = np.linalg.qr(A)
        return Q

    R_list = [random_unitary(N) for _ in range(d + 1)]
    # Random projector cutoffs: ell_k in {1, ..., N//2} to keep things well-conditioned
    ell_list = rng.integers(1, N // 2 + 1, size=d).tolist()

    # Forward simulation: compute circuit unitary polynomial at K sample points,
    # then recover coefficients via IFFT.
    K = max(8 * (d + 1), 512)
    z_pts = unit_circle_points(K)

    # M_vals[k] = full NxN circuit unitary at z_pts[k]
    M_vals = np.zeros((K, N, N), dtype=np.complex128)
    for ki in range(K):
        z = z_pts[ki]
        M = R_list[0].copy()
        for k in range(1, d + 1):
            ell = ell_list[k - 1]
            C_k = np.eye(N, dtype=np.complex128)
            C_k[:ell, :ell] *= z
            M = R_list[k] @ C_k @ M
        M_vals[ki] = M

    # Extract (r, c) block and recover polynomial coefficients via IFFT
    P_vals = M_vals[:, :r, :c]  # (K, r, c)

    # Recover polynomial coefficients from unit-circle samples.
    # P(z_k) = sum_l P_l z_k^l  =>  P_l = (1/K) * sum_k P(z_k) * z_k^{-l}
    #                               = np.fft.fft(P_vals, axis=0)[l] / K
    all_coeffs = np.fft.fft(P_vals, axis=0) / K  # (K, r, c)
    P_coeffs = all_coeffs[:d + 1]

    return P_coeffs


# Cache for expensive complement computations
_complement_cache = {}


def get_cached_complement(d: int, N: int, r: int, c: int, seed: int, n_iters: int = 8_000, n_restarts: int = 2):
    """Get or compute complement Q for given parameters, with caching.
    
    Cache key is (d, N, r, c, seed, n_iters, n_restarts).
    Returns: (P_coeffs, Q_coeffs)
    """
    cache_key = (d, N, r, c, seed, n_iters, n_restarts)
    
    if cache_key not in _complement_cache:
        P = make_random_achievable(d=d, N=N, r=r, c=c, seed=seed)
        comp = find_complement(P, N=N, n_iters=n_iters, n_restarts=n_restarts)
        _complement_cache[cache_key] = (P.copy(), comp.Q_coeffs.copy())
    
    # Return copies to prevent mutation
    P, Q = _complement_cache[cache_key]
    return P.copy(), Q.copy()


def make_phi_cached(d: int, N: int, r: int, c: int, seed: int, n_iters: int = 8_000, n_restarts: int = 2):
    """Make a Phi isometry using cached complement computation.
    
    Returns: (P_coeffs, Phi_coeffs) where Phi = [P; Q]
    """
    P, Q = get_cached_complement(d, N, r, c, seed, n_iters, n_restarts)
    Phi = np.concatenate([P, Q], axis=1)
    return P, Phi


# Pytest fixtures using cached computations
@pytest.fixture(scope="session")
def phi_4_4_2_2_seed30():
    """Cached Phi for d=4, N=4, r=2, c=2, seed=30."""
    return make_phi_cached(d=4, N=4, r=2, c=2, seed=30)


@pytest.fixture(scope="session")
def phi_5_4_2_2_seed31():
    """Cached Phi for d=5, N=4, r=2, c=2, seed=31."""
    return make_phi_cached(d=5, N=4, r=2, c=2, seed=31)


@pytest.fixture(scope="session")
def phi_4_8_4_2_seed32():
    """Cached Phi for d=4, N=8, r=4, c=2, seed=32."""
    return make_phi_cached(d=4, N=8, r=4, c=2, seed=32)


# Cached reduction results (most expensive operation)
@pytest.fixture(scope="session")
def reduction_4_4_2_2_seed30(phi_4_4_2_2_seed30):
    """Cached reduction result for d=4, N=4, r=2, c=2, seed=30."""
    P, Phi = phi_4_4_2_2_seed30
    result = recursive_reduction(Phi)
    return P, Phi, result


@pytest.fixture(scope="session")
def reduction_5_4_2_2_seed31(phi_5_4_2_2_seed31):
    """Cached reduction result for d=5, N=4, r=2, c=2, seed=31."""
    P, Phi = phi_5_4_2_2_seed31
    result = recursive_reduction(Phi)
    return P, Phi, result


@pytest.fixture(scope="session")
def reduction_4_8_4_2_seed32(phi_4_8_4_2_seed32):
    """Cached reduction result for d=4, N=8, r=4, c=2, seed=32."""
    P, Phi = phi_4_8_4_2_seed32
    result = recursive_reduction(Phi)
    return P, Phi, result


# Cached pipeline results for expensive end-to-end tests
@pytest.fixture(scope="session")
def pipeline_case_a_result():
    """Cached pipeline result for Case A: d=16, N=16, r=16, c=1, seed=42."""
    D, N, r, c = 16, 16, 16, 1
    P = make_random_achievable(d=D, N=N, r=r, c=c, seed=42)
    result = compile_unqsp(
        P, N=N, n_iters=10_000, n_restarts=3, verify_K=128, rng=np.random.default_rng(7)
    )
    return P, result, N, D


@pytest.fixture(scope="session")
def pipeline_case_b_result():
    """Cached pipeline result for Case B: d=16, N=16, r=4, c=4, seed=99."""
    D, N, r, c = 16, 16, 4, 4
    P = make_random_achievable(d=D, N=N, r=r, c=c, seed=99)
    result = compile_unqsp(
        P, N=N, n_iters=10_000, n_restarts=3, verify_K=128, rng=np.random.default_rng(13)
    )
    return P, result, N, D


@pytest.fixture
def rng():
    return np.random.default_rng(123)
