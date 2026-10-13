"""One-shot colored forward draw and the data-factor terminal draw (Phase I1).

The closed-form marginal means no SDE is simulated: a single ``x_t`` is drawn in one shot
from ``x_t = m_t x_0 + Σ_t^{1/2} ε``. The terminal ``x_T ~ N(0, S)`` is drawn at ``O(nd)``
via the **data-factor route** (correctness fix #4) — ``x_T = (1/√n) X z' + √τ z''`` has
covariance ``(1/n)XXᵀ + τI = S`` exactly, with no Cholesky factor and no ``d×d`` matrix.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ccd.forward.marginals import ColoredForward
from ccd.operators.linop import CovarianceOperator


def forward_draw(fwd: ColoredForward, x0: Tensor, eps: Tensor, F: float) -> Tensor:
    """One-shot colored forward sample ``x_t = m_t x_0 + Σ_t^{1/2} ε`` (no SDE simulation)."""
    return fwd.mean(x0, F) + fwd.sigma_half(eps, F)


def whiten(fwd: ColoredForward, x: Tensor, F: float, *, require_converged: bool = True, **solver_kw) -> Tensor:
    """Whiten ``x`` by ``V_F^{-1/2}`` (matrix-free, Phase I4): if ``x ~ N(0, V_F)`` the result is
    ``~ N(0, I)``.

    Also the steep half of the near-data-end encode ``V_{F_hi}^{1/2} V_{F_lo}^{-1/2}`` — computing
    ``V_{F_lo}^{-1/2} x`` here (via the rational solver) instead of the combined Chebyshev factor keeps
    the encode accurate as ``F_lo → 0`` (see the round-trip test). Delegates to
    :meth:`ColoredForward.v_inverse_half`; extra solver knobs (``tol``, ``maxiter``, ``n_poles``,
    ``method``) pass through ``**solver_kw``.

    With ``require_converged=True`` (default) it **raises** ``RuntimeError`` if the solve did not reach
    tolerance — a partial whitening must never be used silently by a downstream consumer.
    """
    y, info = fwd.v_inverse_half(x, F, **solver_kw)
    if require_converged:
        conv = info["converged"]
        ok = conv if isinstance(conv, bool) else bool(conv.all())
        if not ok:
            raise RuntimeError(
                f"whiten: the V_F^{{-1/2}} solve did not converge at F={F} "
                f"(relres={info['relres']}); refusing to return a partial whitening."
            )
    return y


def terminal_draw(
    op: CovarianceOperator,
    *,
    batch: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw ``x_T ~ N(0, S)`` at ``O(nd)`` via the data-factor route (no Cholesky).

    Returns ``(d,)`` or ``(d, batch)``.
    """
    n, d = op.n, op.d
    shape_n = (n,) if batch is None else (n, batch)
    shape_d = (d,) if batch is None else (d, batch)
    z1 = torch.randn(shape_n, generator=generator, dtype=op.dtype, device=op.device)
    z2 = torch.randn(shape_d, generator=generator, dtype=op.dtype, device=op.device)
    return (op.X @ z1) / (op.n ** 0.5) + (op.tau ** 0.5) * z2
