# Quantum Signal Processing and Quantum Singular Value Transformation on U(N)

Code for the numerical experiments in [arXiv:2408.01439](https://arxiv.org/abs/2408.01439)

## Overview

This paper generalizes quantum signal processing (QSP) and quantum singular value transformation (QSVT) from U(2) to U(N), enabling multiple polynomial transformations simultaneously from a single block-encoded input. Key results include N-interval decision and a quantum amplitude estimation algorithm achieving the Heisenberg limit without adaptive measurements.

## Notebooks

| Notebook | Paper Section | Description |
|---|---|---|
| `bqsp.ipynb` | Sec. 3.1 | Block encoding error via PCA of random matrices with decay patterns |
| `4_dec.ipynb` | Sec. 3.2 | 4-interval decision: U(4)-QSP vs. two-stage U(2)-QSP comparison |
| `eigval.ipynb` | Sec. 3.2 | Eigenvalue scaling of the sinc kernel matrix vs. degree and gap |
| `qae.ipynb` | Sec. 3.3 | Quantum amplitude estimation: RMSE and error probability scaling |

## Setup

**Requirements:** Python 3.9+, CUDA-capable GPU (optional but recommended).

```bash
pip install -r requirements.txt
```

In addition, install PyTorch with CUDA support from https://pytorch.org/get-started/locally/ if you have a compatible GPU.

## Citation

```bibtex
@article{lu2025unqsp,
  title={Quantum Signal Processing and Quantum Singular Value Transformation on {$U(N)$}},
  author={Lu, Xi and Liu, Yuan and Lin, Hongwei},
  journal={arXiv preprint arXiv:2408.01439},
  year={2024}
}
```
