"""Oracle tests for the Phase I1 colored draws.

The forward and data-factor terminal draws are checked by their second-order statistics:
the empirical covariance of many draws must match ``V_t`` (forward) and ``S`` (terminal).
These are Monte-Carlo gates (fixed seed, large N, loose tolerance) — the tight analytic
correctness of the operators themselves is covered in ``test_marginals.py``.
"""

from __future__ import annotations

import pytest
import torch

from ccd.forward import ColoredForward, forward_draw, phi_power, terminal_draw, whiten
from ccd.operators import apply_fun, make_synthetic_operator, spectrum_bracket
from _oracles import rel_err

MC_TOL = 0.1
N = 40_000


def _setup(d=32, n=400, tau=0.05, seed=0):
    op = make_synthetic_operator(n, d, tau=tau, generator=torch.Generator().manual_seed(seed))
    lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(seed + 1))
    return op, lo, hi


def test_terminal_draw_covariance_is_S():
    op, _, _ = _setup()
    xT = terminal_draw(op, batch=N, generator=torch.Generator().manual_seed(7))   # (d, N)
    cov = (xT @ xT.transpose(-2, -1)) / N
    assert rel_err(cov, op._dense()) < MC_TOL                 # E[cov] = S, no Cholesky


def test_forward_draw_covariance_is_Vt():
    op, lo, hi = _setup()
    fwd = ColoredForward(op, lo, hi, phi=phi_power(1.0))
    F = 1.0
    g = torch.Generator().manual_seed(9)
    # x0 ~ N(0, Σ0 = Ĉ_n) via the data factor; ε ~ N(0, I), independent.
    x0 = (op.X @ torch.randn(op.n, N, generator=g, dtype=op.dtype)) / (op.n ** 0.5)
    eps = torch.randn(op.d, N, generator=g, dtype=op.dtype)
    xt = forward_draw(fwd, x0, eps, F)
    cov = (xt @ xt.transpose(-2, -1)) / N
    evals, U = torch.linalg.eigh(op._dense().double())
    Vt = U @ torch.diag(evals - op.tau * torch.exp(-F * evals)) @ U.transpose(-2, -1)
    assert rel_err(cov, Vt) < MC_TOL


# ------------------------------------------------------------------------- whiten (Phase I4)

def test_whiten_covariance_is_identity():
    # If x ~ N(0, V_F) then V_F^{-1/2} x ~ N(0, I): the empirical covariance of whitened draws ≈ I.
    op = make_synthetic_operator(200, 16, tau=0.05, generator=torch.Generator().manual_seed(3),
                                 dtype=torch.float64)
    lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(4))
    fwd = ColoredForward(op, lo, hi, phi=phi_power(1.0))
    F = 2.0
    g = torch.Generator().manual_seed(5)
    z = torch.randn(op.d, N, generator=g, dtype=torch.float64)
    # draw x ~ N(0, V_F) by applying V_F^{1/2} to standard normal
    x = apply_fun(lambda lam: torch.sqrt(torch.clamp(fwd._vmarg_fn(F)(lam), min=0.0)), op, z, lo, hi, K=200)
    y = whiten(fwd, x, F, tol=1e-9, maxiter=8000)
    cov = (y @ y.transpose(-2, -1)) / N
    assert rel_err(cov, torch.eye(op.d, dtype=torch.float64)) < MC_TOL


def test_whiten_refuses_unconverged():
    op = make_synthetic_operator(400, 48, tau=0.05, generator=torch.Generator().manual_seed(0),
                                 dtype=torch.float64)
    lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(1))
    fwd = ColoredForward(op, lo, hi, phi=phi_power(1.0))
    v = torch.randn(op.d, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    with pytest.raises(RuntimeError):
        whiten(fwd, v, 0.01, tol=1e-12, maxiter=2)               # steep + starved -> not converged
    # ...but the partial result is returned (finite) when the caller opts out of the convergence gate.
    y = whiten(fwd, v, 0.01, tol=1e-12, maxiter=2, require_converged=False)
    assert torch.isfinite(y).all()


def test_whiten_refuses_unconverged_batched():
    # Mixed-convergence batch (the tensor branch of the converged check): a zero column converges
    # instantly while stiff columns do not, so bool(converged.all()) is False and whiten must raise.
    op = make_synthetic_operator(400, 48, tau=0.05, generator=torch.Generator().manual_seed(0),
                                 dtype=torch.float64)
    lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(1))
    fwd = ColoredForward(op, lo, hi, phi=phi_power(1.0))
    cols = torch.randn(op.d, 2, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    V = torch.cat([torch.zeros(op.d, 1, dtype=torch.float64), cols], dim=1)   # col 0 converges at once
    with pytest.raises(RuntimeError):
        whiten(fwd, V, 0.01, tol=1e-12, maxiter=2)
