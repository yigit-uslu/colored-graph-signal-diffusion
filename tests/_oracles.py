"""Dense baselines for the matrix-free operators — TEST ONLY (never imported by src/ccd).

These materialise ``S`` and use an explicit eigendecomposition / matrix exponential as
the ground truth that the matvec-only Chebyshev code is checked against. Computed in
float64 so the baseline error is far below the float32 implementation error we measure.
"""

from __future__ import annotations

import torch


def dense_apply(f, op, v):
    """``f(S) v`` via a dense eigendecomposition ``U f(Λ) Uᵀ v`` (float64)."""
    S = op._dense().double()
    evals, U = torch.linalg.eigh(S)
    vv = v.double()
    fvals = f(evals)
    if vv.dim() == 1:
        return U @ (fvals * (U.transpose(-2, -1) @ vv))
    return U @ (fvals[:, None] * (U.transpose(-2, -1) @ vv))


def dense_expm_apply(c, op, v):
    """``exp(-c S) v`` via a dense matrix exponential (float64) — independent of the eigh path."""
    S = op._dense().double()
    return torch.linalg.matrix_exp(-float(c) * S) @ v.double()


def dense_solve(A, b, *, power=-1.0):
    """``A^{power} b`` for a dense SPD ``A`` via eigendecomposition (float64) — TEST-ONLY baseline.

    ``power=-1`` gives the inverse solve ``A⁻¹ b``; ``power=-0.5`` the inverse square root ``A^{-1/2} b``.
    """
    evals, U = torch.linalg.eigh(A.double())
    bb = b.double()
    fvals = evals ** float(power)
    if bb.dim() == 1:
        return U @ (fvals * (U.transpose(-2, -1) @ bb))
    return U @ (fvals[:, None] * (U.transpose(-2, -1) @ bb))


def rel_err(approx, exact) -> float:
    """Relative L2 error ``‖approx - exact‖ / ‖exact‖`` (both cast to float64)."""
    a = approx.double()
    e = exact.double()
    return float((a - e).norm() / (e.norm() + 1e-30))
