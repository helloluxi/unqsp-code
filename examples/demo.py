"""Demo: compile a degree-8 polynomial matrix block into U(8)-QSP circuit parameters.

Note on parameter choice: for a random U(N) circuit, the top-r×c sub-block always
has σ_max close to 1 when r ≈ N/2 or c ≈ N/2 (random subblock theorem). This makes
the spectral complement problem near-degenerate and hard to optimize. Keep r, c < N/2
for well-conditioned problems, or use a larger N (e.g. N=16 for a 4×4 block).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from tests.conftest import make_random_achievable
from src.pipeline import compile_unqsp

D, N, R, C = 8, 8, 4, 4   # r=c=2 < N/2=4: well-conditioned complement problem

print(f"Target: degree-{D} polynomial, P shape ({R}x{C}), ancilla N={N}")
P = make_random_achievable(d=D, N=N, r=R, c=C, seed=0)

result = compile_unqsp(
    P, N=N, 
    n_iters=2_000,      # Reduced from 10_000 for faster execution
    n_restarts=1,       # Keep single restart
    rng=np.random.default_rng(0), 
    verbose=True
)

print(f"Validation : {result.validation.message}")
print(f"Complement : max residual = {result.complement.max_residual:.2e}")
print(f"Circuit    : {len(result.R_list)} unitaries, ell_list = {result.ell_list}")
print(f"Max error  : {result.max_error:.2e}  (‖P_circuit - P‖_F over 256 test points)")
