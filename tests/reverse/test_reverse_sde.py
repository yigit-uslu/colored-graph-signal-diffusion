"""Phase I3b-1 — the Gaussian-scoped stochastic reverse-time SDE sampler.

The reverse SDE of the colored Gaussian process (:mod:`ccd.reverse.sde`) admits an exact
matrix-free ancestral transition ``x_{F_b} = μ(S) x_{F_a} + √q(S) ξ`` with smooth per-band
coefficients ``μ, √q`` — no eigendecomposition, no explicit inverse. A stochastic sampler cannot
be gated on a single trajectory, so the checks are:

1. **The coefficients are correct** — the closed forms satisfy exact marginal preservation
   ``μ² v_{F_a} + q = v_{F_b}`` for every family member ``a ∈ {−1, 0, ½, 1}`` (deterministic), and
   the conditional-mean operator ``μ(S)`` matches the dense eigendecomposition.
2. **Euler–Maruyama is consistent** — its per-band variance recursion converges ``~O(ΔF)`` to the
   marginal ``v_F`` as the grid refines (deterministic).
3. **It closes the generative loop** — the exact sampler transports the prior ``N(0, V_{F_max})``
   to the data covariance ``Ĉ_n = S − τI`` (Monte-Carlo), is marginal-preserving grid-independently,
   produces Gaussian samples, and — for the EM scheme — realises ``V_F`` at a fine grid.

float64 throughout for low baseline error; ``rel_err`` gates.
"""

from __future__ import annotations

import pytest
import torch

from ccd.forward import phi_power
from ccd.operators import apply_fun, make_synthetic_operator, spectrum_bracket
from ccd.reverse import GaussianReverseSDE, reverse_f_grid
from _oracles import dense_apply, rel_err

TOL = 1e-3
COV_TOL = 0.06
KURT_TOL = 0.2
FAMILY = [-1.0, 0.0, 0.5, 1.0]


def _setup(d=48, n=400, tau=0.05, seed=0, dtype=torch.float64):
    # n > d ⇒ Ĉ_n full rank ⇒ V_0 nonsingular, so the reverse reaches F=0 in the smooth regime.
    g = torch.Generator().manual_seed(seed)
    op = make_synthetic_operator(n, d, tau=tau, generator=g, dtype=dtype)
    lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(seed + 1))
    v = torch.randn(d, generator=torch.Generator().manual_seed(seed + 2), dtype=dtype)
    return op, lo, hi, v


def _v_fn(op, phi, F):
    return lambda lam: lam - op.tau * torch.exp(-F * phi(lam))


def _draw_prior(op, phi, lo, hi, F_max, N, seed):
    """x_T ~ N(0, V_{F_max}) exactly, via V_{F_max}^{1/2} applied to white noise (d, N)."""
    z = torch.randn(op.d, N, generator=torch.Generator().manual_seed(seed), dtype=op.dtype)
    sig = lambda lam: torch.sqrt(torch.clamp(_v_fn(op, phi, F_max)(lam), min=0.0))
    return apply_fun(sig, op, z, lo, hi)


# ---------------------------------------------------- 1. exact coefficients are correct
@pytest.mark.parametrize("a", FAMILY)
def test_exact_coeffs_preserve_marginal(a):
    # The defining correctness property of the ancestral step: μ² v_{F_a} + q = v_{F_b}, checked
    # as a per-band scalar identity at the true eigenvalues (dense, test-only).
    op, lo, hi, _ = _setup()
    phi = phi_power(a)
    rev = GaussianReverseSDE(op, lo, hi, phi=phi)
    F_a, F_b = 3.0, 0.7
    evals = torch.linalg.eigvalsh(op._dense())
    v_a, v_b = _v_fn(op, phi, F_a), _v_fn(op, phi, F_b)
    mu = rev.mean_fn(F_a, F_b)(evals)
    q = rev.noise_std_fn(F_a, F_b)(evals) ** 2
    assert rel_err(mu ** 2 * v_a(evals) + q, v_b(evals)) < 1e-6


def test_exact_conditional_mean_matches_dense():
    # The mean operator μ(S) is applied correctly by the matrix-free Chebyshev path.
    op, lo, hi, v = _setup()
    phi = phi_power(1.0)
    rev = GaussianReverseSDE(op, lo, hi, phi=phi)
    F_a, F_b = 3.0, 0.7
    approx = apply_fun(rev.mean_fn(F_a, F_b), op, v, lo, hi)
    assert rel_err(approx, dense_apply(rev.mean_fn(F_a, F_b), op, v)) < TOL


def test_step_rejects_forward_direction():
    op, lo, hi, v = _setup()
    rev = GaussianReverseSDE(op, lo, hi, phi=phi_power(1.0))
    with pytest.raises(ValueError):
        rev.step(v, 1.0, 2.0)                # F_b > F_a is the forward direction


# --------------------------------------------- 2. Euler–Maruyama consistency (deterministic)
def test_em_variance_recursion_converges_to_marginal():
    # Propagate the per-band variance through the EM update deterministically (no sampling), using
    # the sampler's own coefficient functions: ν ← s(λ)² ν + ΔF·(λφ). It must converge to v_{F_min}.
    op, lo, hi, _ = _setup()
    phi = phi_power(1.0)
    rev = GaussianReverseSDE(op, lo, hi, phi=phi)
    F_max, F_min = 3.0, 0.5
    evals = torch.linalg.eigvalsh(op._dense())
    g2 = rev.diffusion_std_fn()(evals) ** 2
    errs = []
    for n in (16, 32, 64, 128):
        grid = [float(f) for f in reverse_f_grid(F_max, F_min, n)]
        nu = _v_fn(op, phi, F_max)(evals)
        for F_a, F_b in zip(grid[:-1], grid[1:]):
            nu = rev.em_drift_fn(F_a, F_b)(evals) ** 2 * nu + (F_a - F_b) * g2
        errs.append(rel_err(nu, _v_fn(op, phi, F_min)(evals)))
    assert all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    assert errs[-2] / errs[-1] > 1.8         # ≈halving per doubling (order ≈ 1)
    assert errs[-1] < 5e-3


# ------------------------------------------------- 3. generative loop (Monte-Carlo, slow)
@pytest.mark.slow
@pytest.mark.parametrize("a", [0.0, 0.5, 1.0])
def test_exact_sde_transports_prior_to_data_covariance(a):
    op, lo, hi, _ = _setup(d=16, n=200, seed=7)
    phi = phi_power(a)
    rev = GaussianReverseSDE(op, lo, hi, phi=phi)
    F_max, N = 5.0, 40_000
    gen = torch.Generator().manual_seed(11)
    x_T = _draw_prior(op, phi, lo, hi, F_max, N, seed=1)
    x_0 = rev.integrate(x_T, reverse_f_grid(F_max, 0.0, 8), scheme="exact", generator=gen)

    cov = (x_0 @ x_0.transpose(-2, -1)) / N
    c_hat = op._dense() - op.tau * torch.eye(op.d, dtype=op.dtype)         # Ĉ_n
    assert rel_err(cov, c_hat) < COV_TOL
    assert rel_err(cov, op._dense()) > COV_TOL       # non-vacuity: not the prior N(0,S)


@pytest.mark.slow
def test_exact_sde_marginal_is_grid_independent():
    # The exact ancestral step preserves the marginal at any grid: a 1-step and a 16-step reverse
    # from F_max to an intermediate F both realise N(0, V_F).
    op, lo, hi, _ = _setup(d=16, n=200, seed=7)
    phi = phi_power(1.0)
    rev = GaussianReverseSDE(op, lo, hi, phi=phi)
    F_max, F_mid, N = 5.0, 1.5, 40_000
    x_T = _draw_prior(op, phi, lo, hi, F_max, N, seed=2)
    v_mid = dense_apply(_v_fn(op, phi, F_mid), op, torch.eye(op.d, dtype=op.dtype))  # V_{F_mid}
    for n in (1, 16):
        gen = torch.Generator().manual_seed(3)
        x = rev.integrate(x_T, reverse_f_grid(F_max, F_mid, n), scheme="exact", generator=gen)
        cov = (x @ x.transpose(-2, -1)) / N
        assert rel_err(cov, v_mid) < COV_TOL


@pytest.mark.slow
def test_reverse_sde_marginal_is_gaussian():
    # Per-band excess kurtosis ≈ 0: the reverse SDE keeps the marginal Gaussian.
    op, lo, hi, _ = _setup(d=16, n=200, seed=7)
    phi = phi_power(1.0)
    rev = GaussianReverseSDE(op, lo, hi, phi=phi)
    F_max, F_mid, N = 5.0, 1.5, 60_000
    evals, U = torch.linalg.eigh(op._dense())
    gen = torch.Generator().manual_seed(4)
    x_T = _draw_prior(op, phi, lo, hi, F_max, N, seed=5)
    x = rev.integrate(x_T, reverse_f_grid(F_max, F_mid, 8), scheme="exact", generator=gen)
    y = U.transpose(-2, -1) @ x                              # decorrelate into the eigenbasis
    m2 = (y ** 2).mean(dim=1)
    m4 = (y ** 4).mean(dim=1)
    excess_kurt = (m4 / m2 ** 2 - 3.0).abs()
    assert float(excess_kurt.max()) < KURT_TOL


@pytest.mark.slow
def test_em_sde_covariance_matches_marginal():
    # The Euler–Maruyama sampler realises the marginal V_{F_min} at a fine grid (weak accuracy).
    op, lo, hi, _ = _setup(d=16, n=200, seed=7)
    phi = phi_power(1.0)
    rev = GaussianReverseSDE(op, lo, hi, phi=phi)
    F_max, F_min, N = 5.0, 0.5, 30_000
    gen = torch.Generator().manual_seed(6)
    x_T = _draw_prior(op, phi, lo, hi, F_max, N, seed=8)
    x = rev.integrate(x_T, reverse_f_grid(F_max, F_min, 160), scheme="euler_maruyama", generator=gen)
    cov = (x @ x.transpose(-2, -1)) / N
    v_min = dense_apply(_v_fn(op, phi, F_min), op, torch.eye(op.d, dtype=op.dtype))   # V_{F_min}
    assert rel_err(cov, v_min) < COV_TOL
