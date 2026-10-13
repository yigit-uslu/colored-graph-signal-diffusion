"""Chebyshev–Clenshaw application of ``f(S)v`` — no eigendecomposition (Phase I0).

Given a scalar function ``f`` and an enclosing interval ``[lo, hi] ⊇ spec(S)``, we
apply ``f(S)`` to a vector in three stages, touching ``S`` only as a matvec:

1. **Shift** the spectrum to ``[-1, 1]``: ``S̃ = a S + b I`` with ``a = 2/(hi-lo)``,
   ``b = -(hi+lo)/(hi-lo)``; ``S̃ w = a (S w) + b w`` is one matvec plus an axpy.
2. **Coefficients** ``c_k`` from a discrete cosine transform of the *scalar* ``f`` at
   the Chebyshev nodes (``O(K log K)`` scalar work, independent of ``d``).
3. **Apply** ``f_K(S)v = Σ_k c_k T_k(S̃) v`` by the Clenshaw recurrence — ``K`` matvecs,
   unconditionally stable since ``‖T_k(S̃)‖₂ ≤ 1``.

By the spectral-mapping theorem the operator error inherits the scalar error, so a
degree ``K`` that resolves ``f`` as a scalar function on ``[lo, hi]`` resolves
``f(S)``. The eigenvectors of ``S`` never appear.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor

from ccd.operators.linop import CovarianceOperator

ScalarFn = Callable[[Tensor], Tensor]


def _cheb_nodes(lo: float, hi: float, K: int, dtype: torch.dtype, device) -> Tensor:
    """The ``K+1`` Chebyshev nodes of ``[lo, hi]`` (mapped from ``[-1, 1]``)."""
    j = torch.arange(K + 1, dtype=dtype, device=device)
    xi = torch.cos(math.pi * (j + 0.5) / (K + 1))
    return 0.5 * (hi - lo) * xi + 0.5 * (hi + lo)


def cheb_coeffs(
    f: ScalarFn, lo: float, hi: float, K: int, *, dtype: torch.dtype = torch.float32, device="cpu"
) -> Tensor:
    """Chebyshev coefficients ``c_0..c_K`` of ``f`` on ``[lo, hi]`` via the DCT (scalar)."""
    lam = _cheb_nodes(lo, hi, K, dtype, device)
    fj = f(lam)                                                   # (K+1,)
    k = torch.arange(K + 1, dtype=dtype, device=device)
    j = torch.arange(K + 1, dtype=dtype, device=device)
    cos_mat = torch.cos(math.pi * k[:, None] * (j[None, :] + 0.5) / (K + 1))
    c = (cos_mat @ fj) * (2.0 / (K + 1))
    c[0] = c[0] * 0.5
    return c


def chebyshev_degree(
    f: ScalarFn, lo: float, hi: float, *, tol: float = 1e-4, k_max: int = 256,
    dtype: torch.dtype = torch.float32, device="cpu",
) -> int:
    """Smallest degree whose Chebyshev coefficient tail falls below ``tol`` (capped at ``k_max``).

    Computed from the scalar ``f`` only (no matvecs) — drives adaptive degree
    selection and the efficiency study. The default ``tol=1e-4`` is sized for
    float32: in single precision the coefficient tail saturates near the rounding
    floor (~1e-7), so a much smaller ``tol`` would chase noise and over-resolve.
    """
    c = cheb_coeffs(f, lo, hi, k_max, dtype=dtype, device=device)
    absc = c.abs()
    above = (absc > tol * absc.max()).nonzero().flatten()
    return int(above.max()) if above.numel() > 0 else 0


def _shifted(op: CovarianceOperator, w: Tensor, a: float, b: float) -> Tensor:
    """``S̃ w = a (S w) + b w`` — one matvec plus an axpy."""
    return a * op.matvec(w) + b * w


def apply_fun(
    f: ScalarFn,
    op: CovarianceOperator,
    v: Tensor,
    lo: float,
    hi: float,
    K: int | None = None,
    *,
    tol: float = 1e-4,
    k_max: int = 256,
) -> Tensor:
    """Apply ``f(S)`` to ``v`` by Chebyshev–Clenshaw using exactly ``K`` matvecs.

    ``K=None`` selects the degree adaptively from the coefficient tail (:func:`chebyshev_degree`).
    """
    dtype, device = v.dtype, v.device
    if K is None:
        K = max(chebyshev_degree(f, lo, hi, tol=tol, k_max=k_max, dtype=dtype, device=device), 1)
    c = cheb_coeffs(f, lo, hi, K, dtype=dtype, device=device)
    if K == 0:
        return c[0] * v
    a = 2.0 / (hi - lo)
    b = -(hi + lo) / (hi - lo)
    # Clenshaw, started at k=K (where the recurrence's b_{K+1}=0 makes the first matvec trivial).
    y1 = c[K] * v
    y2 = torch.zeros_like(v)
    for k in range(K - 1, 0, -1):
        y0 = 2.0 * _shifted(op, y1, a, b) - y2 + c[k] * v
        y2, y1 = y1, y0
    return _shifted(op, y1, a, b) - y2 + c[0] * v


class ChebyshevBasis:
    """Amortised Clenshaw basis: build ``T_k(S̃) v`` once (``K`` matvecs), reuse for many ``f``.

    The recurrence vectors depend only on ``S̃`` and ``v`` — not on ``f`` — so several
    operators that share the argument ``v`` (e.g. ``m_t``, ``Σ_t^{1/2}``, ``V_t`` applied
    to the same vector, or one operator at many time levels) cost ``K`` matvecs in
    total, not ``K`` per function.
    """

    def __init__(self, op: CovarianceOperator, v: Tensor, lo: float, hi: float, K: int):
        if K < 1:
            raise ValueError(f"K must be >= 1; got {K}")
        self.lo, self.hi, self.K = float(lo), float(hi), int(K)
        a = 2.0 / (hi - lo)
        b = -(hi + lo) / (hi - lo)
        basis = [v, _shifted(op, v, a, b)]                       # T_0, T_1
        for _ in range(2, K + 1):
            basis.append(2.0 * _shifted(op, basis[-1], a, b) - basis[-2])
        self.T = torch.stack(basis, dim=0)                       # (K+1, ...)

    def contract(self, f_or_coeffs) -> Tensor:
        """Evaluate ``Σ_k c_k T_k(S̃) v`` for a function or precomputed coefficients — no matvecs."""
        if callable(f_or_coeffs):
            c = cheb_coeffs(f_or_coeffs, self.lo, self.hi, self.K, dtype=self.T.dtype, device=self.T.device)
        else:
            c = f_or_coeffs
        return torch.tensordot(c, self.T, dims=([0], [0]))
