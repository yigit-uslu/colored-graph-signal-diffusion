"""End-to-end verification of the FORWARD SDE itself — not just its closed form.

I1/I2 gate the closed-form marginals (``m_t, Σ_t^{1/2}, V_t``) and the draws against a
dense-eigendecomposition oracle, but they never simulate the stochastic process: the link
from the SDE to those marginals lives only in the on-paper derivation. This test closes that
loop. It integrates the colored forward SDE (with ``β ≡ 1``, so ``F = T``)

    dx = −½ φ(S) x dt + (S φ(S))^{1/2} dW,        φ(λ) = λ^a,

by **Euler–Maruyama**, which uses only the drift/diffusion *coefficients*
(``−½φ(S)`` and ``(Sφ(S))^{1/2}``) and never the closed form ``m_t``/``Σ_t`` — so agreement
is a genuine, independent confirmation that the closed form is the SDE's marginal:

  (i)   conditional mean        E[x_F | x_0]      → m_F x_0,
  (ii)  conditional covariance  Cov(x_F | x_0)    → Σ_F = S(I − e^{−Fφ(S)}),
  (iii) marginal covariance     (x_0 ~ N(0,Σ_0))  → V_F = S − τ e^{−Fφ(S)},
  (iv)  Gaussianity             excess kurtosis per eigen-band ≈ 0,
  (v)   stationarity            at large F the law is N(0,S) under the dynamics.

The operators are applied in the exact eigenbasis (the process is diagonal there since all
operators commute), so the test isolates *SDE ↔ marginal* correctness from the Chebyshev
engine. Combined with I1/I2 (Chebyshev ≈ dense closed form), this gives SDE ≈ production code.

Monte-Carlo + Euler–Maruyama: fixed seed, many paths, small step, loose tolerances. The
integrator is explicit, so the colourings simulated here are the non-stiff ``a ≥ 0`` members
(``a < 0`` is condition-number-stiff for explicit EM — a known property, see the trade-off
table in the review note — and would need an implicit scheme).
"""

from __future__ import annotations

import pytest
import torch

from ccd.operators import make_synthetic_operator
from _oracles import rel_err

# Full Euler–Maruyama simulation: heavy (~2 min). Opt-in via `-m slow`; excluded from the
# default fast suite. This is the end-to-end SDE↔marginal check, not a per-commit gate.
pytestmark = pytest.mark.slow

MEAN_TOL = 0.03      # conditional mean is low-variance
COV_TOL = 0.06       # MC + Euler–Maruyama covariance
KURT_TOL = 0.2       # excess kurtosis (N large ⇒ MC error ~ sqrt(24/N) ≈ 0.025)


def _spectrum(d=16, n=200, tau=0.05, seed=0):
    op = make_synthetic_operator(n, d, tau=tau, generator=torch.Generator().manual_seed(seed))
    evals, U = torch.linalg.eigh(op._dense().double())   # exact eigenbasis (test oracle)
    return op, evals, U


def _em_paths(evals, U, a, F, n_steps, N, gen, y0):
    """Euler–Maruyama paths of the forward SDE, returned in x-space ``(d, N)`` at time ``F``.

    In eigen-coordinates ``y = Uᵀx`` the SDE is ``d bands`` of independent scalar OU,
    ``dy_i = −½ λ_i^a y_i dt + λ_i^{(1+a)/2} dW_i``; the EM update uses the *linear* drift
    factor ``(1 − ½λ^a dt)`` (NOT the exact ``e^{−½λ^a dt}``), so it never sees the closed form.
    """
    phi = evals ** a
    g = evals ** (0.5 * (1.0 + a))               # diffusion coeff (Sφ(S))^{1/2} per band
    dt = F / n_steps
    fac = 1.0 - 0.5 * phi * dt
    sig = g * dt ** 0.5
    y = y0.clone()
    for _ in range(n_steps):
        y = fac[:, None] * y + sig[:, None] * torch.randn(y.shape, generator=gen, dtype=torch.float64)
    return U @ y


@pytest.mark.parametrize("a", [0.0, 0.5, 1.0])
def test_sde_conditional_mean_matches_mt(a):
    op, evals, U = _spectrum()
    F, N, n_steps = 1.0, 20_000, 600
    gen = torch.Generator().manual_seed(101)
    x0 = torch.randn(op.d, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    y0 = (U.transpose(-2, -1) @ x0)[:, None].expand(op.d, N).contiguous()   # same x0 on every path
    xF = _em_paths(evals, U, a, F, n_steps, N, gen, y0)
    emp_mean = xF.mean(dim=1)
    mF_x0 = U @ (torch.exp(-0.5 * F * evals ** a) * (U.transpose(-2, -1) @ x0))   # closed-form mean
    assert rel_err(emp_mean, mF_x0) < MEAN_TOL


@pytest.mark.parametrize("a", [0.0, 0.5, 1.0])
def test_sde_conditional_covariance_matches_sigmat(a):
    op, evals, U = _spectrum()
    F, N, n_steps = 1.0, 40_000, 600
    gen = torch.Generator().manual_seed(202)
    y0 = torch.zeros(op.d, N, dtype=torch.float64)                       # x0 = 0 ⇒ Cov = Σ_F
    xF = _em_paths(evals, U, a, F, n_steps, N, gen, y0)
    cov = (xF @ xF.transpose(-2, -1)) / N
    sigma_F = U @ torch.diag(evals * (1.0 - torch.exp(-F * evals ** a))) @ U.transpose(-2, -1)
    assert rel_err(cov, sigma_F) < COV_TOL


def test_sde_marginal_covariance_matches_vt():
    op, evals, U = _spectrum()
    # F=3.5 puts V_F squarely in the transient — far from BOTH endpoints — so the τ-decay term
    # τe^{−Fφ} is genuinely resolved. (At F=1 it is only ~τ of ‖V‖, and the "no-decay" form
    # V=Σ_0 would slip under COV_TOL; the guard below pins that down permanently.)
    a, F, N, n_steps = 1.0, 3.5, 40_000, 600
    gen = torch.Generator().manual_seed(303)
    # x0 ~ N(0, Σ_0): in eigen-coords, band i has variance (Σ_0)_ii = λ_i − τ.
    y0 = torch.sqrt(evals - op.tau)[:, None] * torch.randn(op.d, N, generator=gen, dtype=torch.float64)
    xF = _em_paths(evals, U, a, F, n_steps, N, gen, y0)
    cov = (xF @ xF.transpose(-2, -1)) / N
    v_F = U @ torch.diag(evals - op.tau * torch.exp(-F * evals ** a)) @ U.transpose(-2, -1)
    assert rel_err(cov, v_F) < COV_TOL
    # Non-vacuity guard: the simulated marginal must be distinguishable from the F=0 form Σ_0
    # (= "the τ-decay never happened"). rel_err(V_F, Σ_0) ≈ 0.086 here, well above COV_TOL, so a
    # bug collapsing the decay term is caught rather than tolerated.
    sigma0 = U @ torch.diag(evals - op.tau) @ U.transpose(-2, -1)
    assert rel_err(cov, sigma0) > COV_TOL


def test_sde_marginal_is_gaussian():
    # The linear drift + additive (state-independent) noise make the marginal exactly Gaussian;
    # confirm the simulated per-band law has ~zero excess kurtosis.
    op, evals, U = _spectrum()
    a, F, N, n_steps = 1.0, 1.0, 40_000, 600
    gen = torch.Generator().manual_seed(404)
    y0 = torch.zeros(op.d, N, dtype=torch.float64)
    xF = _em_paths(evals, U, a, F, n_steps, N, gen, y0)
    yF = U.transpose(-2, -1) @ xF                                        # back to decoupled bands
    yc = yF - yF.mean(dim=1, keepdim=True)
    m2 = (yc ** 2).mean(dim=1)
    m4 = (yc ** 4).mean(dim=1)
    excess_kurt = m4 / m2 ** 2 - 3.0
    assert float(excess_kurt.abs().max()) < KURT_TOL


def test_sde_converges_to_stationary_N0S():
    # Under its own dynamics (not just the F→∞ limit of the formula) the law reaches N(0,S).
    op, evals, U = _spectrum()
    a, F, N, n_steps = 1.0, 200.0, 40_000, 4000   # ½φ_max·dt ≈ 0.026 ⇒ stable, ~10 slow-band τ's
    gen = torch.Generator().manual_seed(505)
    y0 = torch.zeros(op.d, N, dtype=torch.float64)
    xF = _em_paths(evals, U, a, F, n_steps, N, gen, y0)
    cov = (xF @ xF.transpose(-2, -1)) / N
    assert rel_err(cov, op._dense().double()) < COV_TOL                 # Cov(x_∞) ≈ S
