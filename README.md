# Quantum Signal Processing and Quantum Singular Value Transformation on U(N)

Code for the numerical experiments in [Lu, Liu & Lin, Quantum (2026)](https://quantum-journal.org/papers/q-2026-03-27-2048/)

## Overview

This paper generalizes quantum signal processing (QSP) and quantum singular value transformation (QSVT) from U(2) to U(N), enabling multiple polynomial transformations simultaneously from a single block-encoded input. Key results include N-interval decision and a quantum amplitude estimation algorithm achieving the Heisenberg limit without adaptive measurements.

## TODO

- [ ] Working in progress: full pipeline in `src/`
- [ ] Gate-level compilation with qiskit

## Notebooks (`samples/`)

| Notebook | Paper Section | Description |
|---|---|---|
| `samples/bqsp.ipynb` | Sec. 3.1 | Block encoding error via PCA of random matrices with decay patterns |
| `samples/4_dec.ipynb` | Sec. 3.2 | 4-interval decision: U(4)-QSP vs. two-stage U(2)-QSP comparison |
| `samples/eigval.ipynb` | Sec. 3.2 | Eigenvalue scaling of the sinc kernel matrix vs. degree and gap |
| `samples/qae.ipynb` | Sec. 3.3 | Quantum amplitude estimation: RMSE and error probability scaling |

## Compilation Pipeline (`src/`)

A full U(N)-QSP compilation pipeline that takes a target polynomial matrix block `P(z)` and returns the circuit parameters `R_0, ..., R_d ∈ U(N)` and projector cutoffs `ℓ_1, ..., ℓ_d`:

1. **Validate** — checks `σ_max(P(e^{iθ})) ≤ 1` (or exact isometry when `r=N` or `c=N`)
2. **Complement** — finds polynomial `Q(z)` such that `[P; Q]` is an isometry on the unit circle, via spectral-factor warm-start + Adam + L-BFGS
3. **Reduce** — recursively extracts `R_k` and `ℓ_k` from the tall isometry via SVD-based degree reduction
4. **Verify** — forward-simulates the recovered circuit and reports max reconstruction error

Run the demo with `python examples/demo.py`, or run tests with `pytest`.

## Setup

**Requirements:** Python 3.9+, CUDA-capable GPU (optional but recommended).

```bash
pip install -r requirements.txt
```

In addition, install PyTorch with CUDA support from https://pytorch.org/get-started/locally/ if you have a compatible GPU.

## Citation

```bibtex
@article{lu2026unqsp,
  title={Quantum Signal Processing and Quantum Singular Value Transformation on {$U(N)$}},
  author={Lu, Xi and Liu, Yuan and Lin, Hongwei},
  journal={Quantum},
  volume={10},
  pages={2048},
  year={2026},
  doi={10.22331/q-2026-03-27-2048},
  url={https://quantum-journal.org/papers/q-2026-03-27-2048/}
}
```
