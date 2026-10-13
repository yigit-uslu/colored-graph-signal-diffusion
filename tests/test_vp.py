"""The standard-diffusion (VP) baseline in `ccd.vp`, checked against exact Gaussian laws.

On Gaussian data of covariance C the denoiser is closed form (`GaussianOracleDenoiser` for the
posterior mean, `GaussianOraclePosteriorSampler` for posterior draws), so each sampler has a known
right answer and no learning is involved. Every check is paired with wrong versions of the code it
must reject: a check that passes a broken sampler is not evidence. CPU only, float64, a few seconds.

Run: conda run -n torch_env python -m pytest tests/test_vp.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.stats import chi2

import ccd.vp as vp
from ccd.operators import CovarianceOperator

D = 6
TAU = 0.01


def _correlation(seed: int = 0) -> torch.Tensor:
    """A correlation matrix (unit diagonal, as VP's standardised data has) with a spread spectrum."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    C = Q @ np.diag(np.geomspace(3.0, 0.05, D)) @ Q.T
    s = np.sqrt(np.diag(C))
    return torch.tensor(C / np.outer(s, s), dtype=torch.float64)


def _sym_power(A: torch.Tensor, p: float) -> torch.Tensor:
    w, U = torch.linalg.eigh(A)
    return U @ torch.diag(w**p) @ U.T


C = _correlation()
EYE = torch.eye(D, dtype=torch.float64)


# --------------------------------------------------------------------------- ancestral sampler

N_SAMPLES = 20_000
#: Six ancestral steps from r = 1e3 down to the read-out level 0.02.
R_GRID = np.geomspace(1e3, 0.02, 7).tolist()
#: (n/2)‖Ŵ − I‖²_F on the whitened cloud is asymptotically χ² with d(d+1)/2 degrees of freedom.
#: Gate at a false-alarm probability of 1e-6, read from that law rather than chosen by hand.
DOF = D * (D + 1) // 2
THRESHOLD = float(chi2.isf(1e-6, DOF))


def _whitened_deviation(X: torch.Tensor) -> float:
    """`(n/2)·‖Ŵ − I‖²_F` with `Ŵ = YYᵀ/n`, `Y = C^{-1/2} X`; the mean is known to be zero."""
    Y = _sym_power(C, -0.5) @ X
    W = Y @ Y.T / X.shape[1]
    return 0.5 * X.shape[1] * float(((W - EYE) ** 2).sum())


def _ancestral_cloud(seed: int = 0) -> torch.Tensor:
    """The production sampler, started from VP's stationary N(0, I), with exact posterior draws.

    With exact draws each ancestral step is exact, so at any step count the cloud is N(0, C) up to
    the start: the true marginal at r = 1e3 is `I + α²(C − I)` with `α² = 1/(1 + 1e6) ≈ 1e-6`, far
    below the gate's detection floor of about 0.08 in `‖Ŵ − I‖_F` at this sample size.
    """
    g = torch.Generator().manual_seed(seed)
    oracle = vp.GaussianOraclePosteriorSampler(C, generator=g)
    x_T = torch.randn(D, N_SAMPLES, generator=g, dtype=torch.float64)
    return vp.VPAncestralSampler(oracle).integrate(x_T, R_GRID, generator=g)


def test_the_ancestral_sampler_with_exact_posterior_draws_generates_the_data_covariance():
    assert _whitened_deviation(_ancestral_cloud()) < THRESHOLD


_COEFFS = vp.vp_ancestral_coeffs


def _coeffs_from(transition_var, c_d_alpha):
    """`vp_ancestral_coeffs` rebuilt with one ingredient swapped, to construct a wrong sampler."""
    def coeffs(r_a, r_b):
        a_a, s_a, a_b, s_b = vp.vp_alpha(r_a), vp.vp_sigma(r_a), vp.vp_alpha(r_b), vp.vp_sigma(r_b)
        a_ab = a_a / a_b
        s_ab_sq = transition_var(s_a, s_b, a_ab)
        return (a_ab * s_b**2 / s_a**2, s_ab_sq * c_d_alpha(a_a, a_b) / s_a**2,
                math.sqrt(max(s_ab_sq * s_b**2 / s_a**2, 0.0)))
    return coeffs


WRONG_SAMPLERS = {
    "noise_dropped": ("vp_ancestral_coeffs", lambda r_a, r_b: (*_COEFFS(r_a, r_b)[:2], 0.0)),
    "variance_passed_as_std": ("vp_ancestral_coeffs",
                               lambda r_a, r_b: (*_COEFFS(r_a, r_b)[:2], _COEFFS(r_a, r_b)[2] ** 2)),
    "c_d_uses_alpha_a": ("vp_ancestral_coeffs",
                         _coeffs_from(lambda s_a, s_b, a_ab: s_a**2 - a_ab**2 * s_b**2,
                                      lambda a_a, a_b: a_a)),
    "transition_variance_drops_alpha_ab": ("vp_ancestral_coeffs",
                                           _coeffs_from(lambda s_a, s_b, a_ab: s_a**2 - s_b**2,
                                                        lambda a_a, a_b: a_b)),
    # The schedule itself: σ(r) without its square root no longer tends to 1, so N(0, I) is no
    # longer the law the sampler starts from.
    "sigma_without_sqrt": ("vp_sigma", lambda r: r / (1.0 + r * r)),
}


@pytest.mark.parametrize("name", list(WRONG_SAMPLERS))
def test_each_wrong_ancestral_sampler_is_rejected(monkeypatch, name):
    attr, wrong = WRONG_SAMPLERS[name]
    monkeypatch.setattr(vp, attr, wrong)
    assert _whitened_deviation(_ancestral_cloud()) > THRESHOLD


def test_a_no_op_wrapper_of_the_coefficients_is_accepted(monkeypatch):
    # The control for the test above: substitution alone must not trip the gate, or the rejections
    # would show only that something was swapped, not that it was wrong.
    monkeypatch.setattr(vp, "vp_ancestral_coeffs", lambda r_a, r_b: _COEFFS(r_a, r_b))
    assert _whitened_deviation(_ancestral_cloud()) < THRESHOLD


# --------------------------------------------------------------------------- probability-flow ODE

F_MIN, F_MAX = 0.02, 20.0
LEVELS = vp.VPLevelMap(phi0=1.0, tr_C=float(C.trace()), tr_S=float(C.trace()) + D * TAU)


def _marginal(F: float) -> torch.Tensor:
    r = LEVELS.r_of_F(F)
    return vp.vp_alpha(r) ** 2 * C + vp.vp_sigma(r) ** 2 * EYE


def _transport_error(n_steps: int, scheme: str = "heun") -> float:
    """Relative error of the covariance the PF-ODE carries from F_MAX to F_MIN, with no sampling.

    With the exact denoiser the ODE is linear, so integrating the columns of `Σ_{F_MAX}^{1/2}` gives
    the transported covariance exactly; in the continuum it is the marginal at F_MIN. No final
    denoise: that would compare a different quantity (the posterior mean's covariance).
    """
    ode = vp.VPReversePFODE(vp.GaussianOracleDenoiser(C), LEVELS)
    grid = np.geomspace(F_MAX, F_MIN, n_steps + 1).tolist()
    out = ode.integrate(_sym_power(_marginal(F_MAX), 0.5), grid, scheme=scheme, denoise_final=False)
    target = _marginal(F_MIN)
    return float(torch.linalg.norm(out @ out.T - target) / torch.linalg.norm(target))


def _observed_order(scheme: str = "heun") -> float:
    return math.log2(_transport_error(128, scheme) / _transport_error(256, scheme))


#: Heun is second order. A wrong drift is a different field, so its error stops falling at the
#: wrong limit (order 0), and the first-order Euler step falls at order 1. 1.5 separates 2 from both.
ORDER_FLOOR = 1.5


def test_the_pf_ode_with_the_exact_denoiser_transports_the_marginal_at_second_order():
    assert _observed_order() > ORDER_FLOOR


_DRIFT = vp.VPReversePFODE.drift

WRONG_DRIFTS = {
    "a_F_as_phi0_over_F": (vp.VPLevelMap, "a_of_F", lambda self, F: self.phi0 / F),
    "half_dropped": (vp.VPReversePFODE, "drift", lambda self, x, F: 2.0 * _DRIFT(self, x, F)),
    "alpha_prefactor_dropped": (vp.VPReversePFODE, "drift",
                                lambda self, x, F: _DRIFT(self, x, F) / vp.vp_alpha(self.map.r_of_F(F))),
}


@pytest.mark.parametrize("name", list(WRONG_DRIFTS))
def test_each_wrong_pf_ode_drift_is_rejected(monkeypatch, name):
    owner, attr, wrong = WRONG_DRIFTS[name]
    monkeypatch.setattr(owner, attr, wrong)
    assert _observed_order() < ORDER_FLOOR


def test_a_first_order_step_is_rejected():
    assert _observed_order("euler") < ORDER_FLOOR


# --------------------------------------------------------------------------- denoiser and loss

def test_the_vp_loss_is_the_v_prediction_mse_and_the_head_is_alpha_x():
    g = torch.Generator().manual_seed(0)
    frames = torch.randn(40, D, generator=g, dtype=torch.float64) @ _sym_power(C, 0.5)
    op = CovarianceOperator(((frames - frames.mean(0)) * math.sqrt(40 / 39)).T.contiguous(), tau=TAU)
    ev = torch.linalg.eigvalsh(op.matvec(EYE))
    body = vp.VPBody(vp.GraphOnlyForward(op, float(ev.min()), float(ev.max())),
                     width=4, n_layers=2, K_gf=2).double()
    den = vp.VPDenoiser(body)
    x0 = torch.randn(D, 16, generator=g, dtype=torch.float64)
    eps = torch.randn(D, 16, generator=g, dtype=torch.float64)
    # σ² ≈ 0.33 here, so a loss weighted 1/σ or left unweighted would miss by a factor of 1.75 or more.
    r = 0.7
    a, s = vp.vp_alpha(r), vp.vp_sigma(r)
    v_target = s * x0 - a * eps
    v_mse = float(((body(a * x0 + s * eps, r) - v_target) ** 2).mean())
    assert float(vp.vp_loss(den, x0, r, eps=eps)) == pytest.approx(v_mse, rel=1e-12)
    np.testing.assert_allclose(den.head(x0, r).numpy(), (a * x0).numpy(), rtol=0, atol=0)
