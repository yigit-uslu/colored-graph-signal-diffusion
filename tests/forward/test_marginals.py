"""Oracle tests for the Phase I1 colored forward marginals.

Gates: each operator vs a dense eigendecomposition baseline; the ``V_t`` identity;
Lyapunov stationarity ``V_t → S``; the scalar ``a=0`` reduction to the existing
``VPSchedule``; and the ``(λ−τ)/λ`` SNR prefactor (correctness fix #1). float32 path,
float64 baselines, ``rel_err < 1e-3`` gate.
"""

from __future__ import annotations

import math

import pytest
import torch

from ccd.diffusion.schedule import VPSchedule, make_beta_schedule
from ccd.forward import ColoredForward, phi_power, snr
from ccd.operators import make_synthetic_operator, spectrum_bracket
from _oracles import dense_apply, rel_err

TOL = 1e-3


def _setup(d=48, n=300, tau=0.05, seed=0):
    op = make_synthetic_operator(n, d, tau=tau, generator=torch.Generator().manual_seed(seed))
    lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(seed + 1))
    v = torch.randn(d, generator=torch.Generator().manual_seed(seed + 2))
    return op, lo, hi, v


@pytest.mark.parametrize("a", [1.0, 0.5, 0.0])
def test_marginals_match_dense(a):
    op, lo, hi, v = _setup()
    fwd = ColoredForward(op, lo, hi, phi=phi_power(a))
    F = 2.0
    f_m = lambda lam: torch.exp(-0.5 * F * lam ** a)
    f_s = lambda lam: torch.sqrt(torch.clamp(lam * (1.0 - torch.exp(-F * lam ** a)), min=0.0))
    f_v = lambda lam: lam - op.tau * torch.exp(-F * lam ** a)
    assert rel_err(fwd.mean(v, F), dense_apply(f_m, op, v)) < TOL
    assert rel_err(fwd.sigma_half(v, F), dense_apply(f_s, op, v)) < TOL
    assert rel_err(fwd.v_marginal(v, F), dense_apply(f_v, op, v)) < TOL


def test_vt_identity_assembled():
    # V_t = m_t Σ0 m_t + Σ_t  ==  S − τ e^{−Fφ(S)}, verified at the dense (per-eigenvalue) level.
    op, _, _, _ = _setup()
    a, F = 1.0, 1.5
    evals = torch.linalg.eigvalsh(op._dense().double())
    phi = evals ** a
    mt = torch.exp(-0.5 * F * phi)                      # eigenvalues of m_t
    sigma0 = evals - op.tau                             # eigenvalues of Σ0 = S − τI
    sigmat = evals * (1.0 - torch.exp(-F * phi))        # eigenvalues of Σ_t
    v_assembled = mt * sigma0 * mt + sigmat
    v_formula = evals - op.tau * torch.exp(-F * phi)
    assert float((v_assembled - v_formula).abs().max()) < 1e-9


def test_lyapunov_stationarity():
    # As F → ∞: m_t → 0 and V_t → S (the stationary law is N(0,S)).
    op, lo, hi, v = _setup()
    fwd = ColoredForward(op, lo, hi, phi=phi_power(1.0))
    F = 400.0
    assert rel_err(fwd.v_marginal(v, F), op.matvec(v)) < 1e-3
    assert float(fwd.mean(v, F).norm() / v.norm()) < 1e-3


def test_scalar_a0_reduces_to_vpschedule():
    # At a=0 the mean is the SCALAR whitened-VP mean; with F = −log ᾱ_t it matches VPSchedule.
    betas = make_beta_schedule("linear", 50)
    sched = VPSchedule(betas)
    t = 25
    abar = float(sched.alphas_cumprod[t])
    F = -math.log(abar)
    op, lo, hi, v = _setup()
    fwd = ColoredForward(op, lo, hi, phi=phi_power(0.0))
    assert math.isclose(math.exp(-F / 2), float(sched.sqrt_alphas_cumprod[t]), rel_tol=1e-4)
    assert math.isclose(math.sqrt(1 - math.exp(-F)), float(sched.sqrt_one_minus_alphas_cumprod[t]), rel_tol=1e-4)
    # structurally: at a=0, m_t is a scalar multiple of the identity
    assert rel_err(fwd.mean(v, F), math.exp(-F / 2) * v) < 1e-4


def test_snr_prefactor():
    op, _, _, _ = _setup()
    F, tau = 2.0, op.tau
    phi = phi_power(1.0)
    lam = torch.linspace(0.1, 1.0, 7, dtype=torch.float64)
    s = snr(lam, F, phi, tau)
    signal = torch.exp(-F * phi(lam)) * (lam - tau)
    noise = lam * (1.0 - torch.exp(-F * phi(lam)))
    assert rel_err(s, signal / noise) < 1e-4
    # the (λ−τ)/λ prefactor is genuinely present (≠ the naive 1/(e^{Fφ}−1))
    naive = 1.0 / (torch.exp(F * phi(lam)) - 1.0)
    assert float((s / naive - (lam - tau) / lam).abs().max()) < 1e-6
    assert float(((lam - tau) / lam).min()) < 0.9          # prefactor materially below 1
