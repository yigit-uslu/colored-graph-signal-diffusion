"""Matvec-only spectrum bracketing for the Chebyshev filter (Phase I0).

A Chebyshev filter is evaluated on a rescaled spectrum and *diverges* outside its
interval, so we need an enclosing ``[λ_lo, λ_hi] ⊇ spec(S)`` — not the eigenvalues.

* ``λ_lo = τ`` is an **exact** lower bound for ``S = Ĉ_n + τI`` (``Ĉ_n ⪰ 0``), tight
  when ``n < d``.
* ``λ_hi`` comes from a few power-iteration steps (matvecs only): the Rayleigh
  quotient converges to ``λ_max`` from below, and we inflate by the residual plus a
  small relative margin so the interval safely encloses the spectrum.

No eigendecomposition and no matrix exponential are formed.
"""

from __future__ import annotations

import torch

from ccd.operators.linop import CovarianceOperator


def spectrum_bracket(
    op: CovarianceOperator,
    *,
    n_iter: int = 30,
    generator: torch.Generator | None = None,
    inflate: float = 1e-2,
) -> tuple[float, float]:
    """Return ``(λ_lo, λ_hi)`` with ``spec(S) ⊆ [λ_lo, λ_hi]`` using matvecs only.

    ``λ_lo = op.tau`` (exact). ``λ_hi`` = power-iteration Rayleigh quotient inflated
    by its residual and a ``(1 + inflate)`` relative margin.
    """
    d = op.shape[0]
    lo = float(op.tau)

    q = torch.randn(d, generator=generator, dtype=op.dtype, device=op.device)
    q = q / q.norm()
    for _ in range(n_iter):
        w = op.matvec(q)
        nrm = w.norm()
        if float(nrm) == 0.0:
            break
        q = w / nrm

    w = op.matvec(q)
    lam = float(torch.dot(q, w))               # Rayleigh quotient ≤ λ_max, → λ_max
    resid = float((w - lam * q).norm())
    hi = lam * (1.0 + inflate) + resid + 1e-6
    return lo, hi
