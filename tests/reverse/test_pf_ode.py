"""Phase I3a — the Gaussian-scoped reverse probability-flow ODE sampler.

The reverse PF-ODE of the colored Gaussian process rescales each spectral band by
``√(v_{F_b}/v_{F_a})`` with ``v_F(λ) = λ − τ e^{−Fφ(λ)}`` (:mod:`ccd.reverse.pf_ode`). This
is a *smooth* function of ``S`` — no eigendecomposition, no explicit inverse — so it is applied
by the same Chebyshev–Clenshaw engine as the forward marginals. Gated here:

1. **The step matches the closed form** — ``scheme="exact"`` reproduces the dense-eig transport
   ``g(S) = √(V_{F_b}/V_{F_a})`` for every family member ``a ∈ {−1, 0, ½, 1}``.
2. **The exact scheme is a genuine (grid-independent) integrator** — telescoping per-step factors
   give the same result on a coarse grid, a fine grid, and one shot.
3. **Euler converges to it** — the explicit-Euler discretisation error falls ``~O(ΔF)`` toward
   the exact transport as the grid refines (the sense in which this is a real time-stepper).
4. **It closes the generative loop** — the reverse transports the prior ``N(0, V_{F_max})`` back
   to the data covariance ``Ĉ_n = S − τI``, both as an exact operator identity (fast) and as an
   empirical Monte-Carlo covariance (slow), with a non-vacuity guard against "did nothing".

float32/float64 paths, float64 dense baselines, ``rel_err`` gates.
"""

from __future__ import annotations

import pytest
import torch

from ccd.forward import phi_power
from ccd.operators import apply_fun, make_synthetic_operator, spectrum_bracket
from ccd.reverse import GaussianReversePFODE, reverse_f_grid
from _oracles import dense_apply, rel_err

TOL = 1e-3
FAMILY = [-1.0, 0.0, 0.5, 1.0]
# Many-step integration applies one ApplyFun per step, so a per-step Chebyshev truncation
# error accumulates ~linearly in the step count. To study *integration* (temporal) error in
# isolation we hold the spectral degree fixed and high, resolving each per-step operator to
# ~machine precision (the boundary layer of the transport near λ=τ needs a generous degree).
K_FINE = 160


def _setup(d=48, n=400, tau=0.05, seed=0, dtype=torch.float32):
    # n > d ⇒ Ĉ_n full rank ⇒ V_0 = Ĉ_n nonsingular, so the reverse reaches F=0 in the smooth
    # regime (the rank-deficient / steep-inverse case is the deferred I3+ increment).
    g = torch.Generator().manual_seed(seed)
    op = make_synthetic_operator(n, d, tau=tau, generator=g, dtype=dtype)
    lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(seed + 1))
    v = torch.randn(d, generator=torch.Generator().manual_seed(seed + 2), dtype=dtype)
    return op, lo, hi, v


def _v_fn(op, phi, F):
    return lambda lam: lam - op.tau * torch.exp(-F * phi(lam))


# --------------------------------------------------------- 1. exact step vs closed form
@pytest.mark.parametrize("a", FAMILY)
def test_exact_step_matches_dense_transport(a):
    op, lo, hi, v = _setup()
    phi = phi_power(a)
    rev = GaussianReversePFODE(op, lo, hi, phi=phi)
    F_a, F_b = 3.0, 0.7
    v_a, v_b = _v_fn(op, phi, F_a), _v_fn(op, phi, F_b)
    g = lambda lam: torch.sqrt(torch.clamp(v_b(lam) / v_a(lam), min=0.0))  # exact PF-ODE factor
    approx = rev.step(v, F_a, F_b, scheme="exact")
    assert rel_err(approx, dense_apply(g, op, v)) < TOL


def test_step_rejects_forward_direction():
    op, lo, hi, v = _setup()
    rev = GaussianReversePFODE(op, lo, hi, phi=phi_power(1.0))
    with pytest.raises(ValueError):
        rev.step(v, 1.0, 2.0)                # F_b > F_a is the forward direction


# --------------------------------------------------- 2. exact scheme is grid-independent
def test_exact_scheme_is_grid_independent():
    op, lo, hi, v = _setup(dtype=torch.float64)
    phi = phi_power(1.0)
    rev = GaussianReversePFODE(op, lo, hi, phi=phi, K=K_FINE)
    F_max, F_min = 3.0, 1.0
    v_a, v_b = _v_fn(op, phi, F_max), _v_fn(op, phi, F_min)
    g_true = lambda lam: torch.sqrt(torch.clamp(v_b(lam) / v_a(lam), min=0.0))
    exact_dense = dense_apply(g_true, op, v)               # closed-form √(V_{F_min}/V_{F_max})
    one_shot = rev.step(v, F_max, F_min, scheme="exact")
    coarse = rev.integrate(v, reverse_f_grid(F_max, F_min, 3), scheme="exact")
    fine = rev.integrate(v, reverse_f_grid(F_max, F_min, 40), scheme="exact")
    # Per-step factors telescope, so every grid realises the same closed-form transport
    # (to ~machine precision once the spectral degree is resolved).
    for out in (one_shot, coarse, fine):
        assert rel_err(out, exact_dense) < 1e-6


# ------------------------------------------------------- 3. Euler converges to the exact flow
def test_euler_converges_to_exact_transport():
    op, lo, hi, v = _setup(dtype=torch.float64)
    phi = phi_power(1.0)
    rev = GaussianReversePFODE(op, lo, hi, phi=phi, K=K_FINE)
    F_max, F_min = 3.0, 1.0                             # F_min > 0 keeps 1/v_F bounded (smooth regime)
    v_a, v_b = _v_fn(op, phi, F_max), _v_fn(op, phi, F_min)
    g_true = lambda lam: torch.sqrt(torch.clamp(v_b(lam) / v_a(lam), min=0.0))
    exact = dense_apply(g_true, op, v)                  # dense closed-form transport (ground truth)
    errs = []
    for n in (32, 64, 128, 256):
        approx = rev.integrate(v, reverse_f_grid(F_max, F_min, n), scheme="euler")
        errs.append(rel_err(approx, exact))
    # Explicit Euler is O(ΔF): monotone decrease, ≈halving per doubling, small final error.
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    assert errs[0] / errs[-1] > 6.0          # ≈8× over the 8× refinement (first order)
    assert errs[-2] / errs[-1] > 1.8         # last doubling ~halves the error (order ≈ 1)
    assert errs[-1] < 1e-4


# ------------------------------------------- 4a. generative identity (exact, deterministic)
def test_exact_transport_maps_prior_cov_to_data_cov():
    # The reverse covariance action g(S) V_{F_max}(S) g(S) must equal the data covariance Ĉ_n,
    # checked as an operator identity on a random vector (no Monte-Carlo noise).
    op, lo, hi, w = _setup(dtype=torch.float64)
    phi = phi_power(1.0)
    rev = GaussianReversePFODE(op, lo, hi, phi=phi)
    F_max = 6.0
    g = rev.transport_fn(F_max, 0.0)
    v_fmax = _v_fn(op, phi, F_max)
    lhs = apply_fun(g, op, apply_fun(v_fmax, op, apply_fun(g, op, w, lo, hi), lo, hi), lo, hi)
    c_hat = op.matvec(w) - op.tau * w        # Ĉ_n w = (S − τI) w
    assert rel_err(lhs, c_hat) < TOL


# ------------------------------------------------- 4b. generative loop (Monte-Carlo, slow)
@pytest.mark.slow
@pytest.mark.parametrize("a", [0.0, 0.5, 1.0])
def test_reverse_transports_prior_to_data_covariance(a):
    op, lo, hi, _ = _setup(d=16, n=200, seed=7, dtype=torch.float64)
    phi = phi_power(a)
    rev = GaussianReversePFODE(op, lo, hi, phi=phi)
    F_max, N = 5.0, 40_000

    # Draw the terminal x_T ~ N(0, V_{F_max}) exactly (V_{F_max}^{1/2} applied to white noise),
    # then run the reverse PF-ODE to the data end F=0 over a short grid.
    z = torch.randn(op.d, N, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    sig_fmax = lambda lam: torch.sqrt(torch.clamp(_v_fn(op, phi, F_max)(lam), min=0.0))
    x_T = apply_fun(sig_fmax, op, z, lo, hi)
    x_0 = rev.integrate(x_T, reverse_f_grid(F_max, 0.0, 8), scheme="exact")

    cov = (x_0 @ x_0.transpose(-2, -1)) / N
    c_hat = op._dense() - op.tau * torch.eye(op.d, dtype=torch.float64)   # Ĉ_n
    assert rel_err(cov, c_hat) < 0.05
    # Non-vacuity: the samples were reshaped from the prior N(0,S); reject the "did nothing" form.
    assert rel_err(cov, op._dense()) > 0.05
