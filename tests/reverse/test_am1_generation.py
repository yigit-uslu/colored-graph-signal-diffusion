"""End-to-end generation gate under the coarse-to-fine clock a=−1 (Phase V3b stretch).

The harder colouring: ``φ=λ^{-1}`` noises the high-variance block-factor modes *slowest*, so they survive to a
large ``F_max≈40`` (a ~124× per-band-clock spread). Two things make this regime work where the a=0 gate's
recipe fails:

* **The ancestral integrator** (``scheme="ancestral"``) — plain Euler–Maruyama *diverges* on this stiff clock
  (verified: head cov 2.31 vs ancestral 0.074); the ancestral step integrates the stiff linear part exactly.
* **Gradient clipping** — the scalar EDM weight spans ~64× over ``F∈[0.05,40]`` (20.6 at F=0.05 → 0.32 at
  F=40), so a single high-weight low-F batch explodes the gradient (observed pre-clip norm 41.3, loss spike to
  17.5) and diverges the body. Clipping to 1.0 stabilises it (loss → 0.17, body ≈0 at high F, +26% MMSE at
  F=0.5). This is the one training change vs the a=0 gate (``test_v3b_generation``).

As in the a=0 gate, only the **seed-robust** axis is asserted — bimodality (factor kurtosis) and the scale-free
standardised-factor energy distance — with a covariance divergence-sanity. Reference run (single seed, 4k
steps, 128 ancestral steps, N=3000): learned factor kurtosis ≈ −1.80 (real −1.77) vs head +0.03; std-factor
ED ≈ 0.0011 vs head 0.0062 (~5.6×); ambient ED 0.007 vs 0.022. The covariance (learned 0.088 vs head 0.077)
is *not* tightened relative to a=0 — it stays the seed-sensitive axis (see the summary doc / [[v3b-seed-
sensitivity]]), so it is reported, not gated. ``@pytest.mark.slow`` (trains ~4k steps + two 128-step
ancestral generations; GPU-strongly-preferred).
"""

from __future__ import annotations

import copy

import pytest
import torch

from ccd.denoiser import BlockGMM, ColoredDenoiser, ColoredEDM, VNNBody, edm_loss, sample_F_loguniform
from ccd.forward import ColoredForward, phi_power, terminal_draw
from ccd.operators import spectrum_bracket
from ccd.reverse import LearnedReverseSDE, reverse_f_grid
from _gen_metrics import block_avg, energy_distance, excess_kurtosis, standardize   # shared test-only metrics


def _train_am1(dev, *, steps=4000, k0=2.0, init_seed=0, data_seed=7):
    """Train ``D_θ`` on the block GMM under the coarse-to-fine clock ``a=−1``, with gradient clipping (the
    wide ``F∈[0.05,40]`` EDM weight explodes grads otherwise). Same toy/body as the a=0 gate."""
    torch.manual_seed(init_seed)
    toy = BlockGMM(d=48, n_blocks=8, rho=0.7, sep=0.98, idio_frac=0.1, tau=0.05, dtype=torch.float32, device=dev)
    op = toy.make_operator(n=6000, generator=torch.Generator(device=dev).manual_seed(1))
    lo, hi = spectrum_bracket(op, generator=torch.Generator(device=dev).manual_seed(2))
    fwd = ColoredForward(op, lo, hi, phi=phi_power(-1.0), K=64)
    den = ColoredDenoiser(ColoredEDM(fwd), VNNBody(fwd, width=96, n_layers=3, K_gf=5, mehler_envelope=True, k0=k0))
    opt = torch.optim.Adam(den.body.parameters(), lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(data_seed)
    for _ in range(steps):
        F = sample_F_loguniform(0.05, 40.0)
        x0 = toy.sample(256, generator=gen)
        opt.zero_grad()
        edm_loss(den, x0, F, generator=gen, c_in_kw=dict(tol=1e-4, maxiter=300)).backward()
        torch.nn.utils.clip_grad_norm_(den.body.parameters(), 1.0)   # tame the wide-F EDM-weight gradient
        opt.step()
    den.body.eval()
    return toy, fwd, den, op, lo, hi


@pytest.mark.slow
def test_generation_recovers_bimodal_modes_am1():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    toy, fwd, den, op, lo, hi = _train_am1(dev)

    # paired samplers, both ancestral (EM diverges under a=−1); head = same path, body decoder zeroed --------
    learned = LearnedReverseSDE(den, K=64, c_in_kw=dict(tol=1e-4, maxiter=300))
    head_den = copy.deepcopy(den)
    with torch.no_grad():
        for p in head_den.body.decoder.parameters():
            p.zero_()
    head = LearnedReverseSDE(head_den, K=64, c_in_kw=dict(tol=1e-3, maxiter=200, require_converged=False))

    N_GEN, N_STEPS, F_MAX, F_MIN = 3000, 128, 40.0, 0.02
    grid = reverse_f_grid(F_MAX, F_MIN, N_STEPS, spacing="equidistribute", weight="abs",
                          phi=phi_power(-1.0), tau=0.05, lo=lo, hi=hi)
    xT = terminal_draw(op, batch=N_GEN, generator=torch.Generator(device=dev).manual_seed(100))
    x_learned = learned.integrate(xT, grid, scheme="ancestral", generator=torch.Generator(device=dev).manual_seed(200))
    x_head = head.integrate(xT, grid, scheme="ancestral", generator=torch.Generator(device=dev).manual_seed(200))
    x_real = toy.sample(N_GEN, generator=torch.Generator(device=dev).manual_seed(300))

    bm = list(toy.non_gaussian_blocks)
    f_learned, f_head, f_real = block_avg(toy, x_learned), block_avg(toy, x_head), block_avg(toy, x_real)
    k_learned = float(excess_kurtosis(f_learned)[bm].mean())
    k_head = float(excess_kurtosis(f_head)[bm].mean())
    k_real = float(excess_kurtosis(f_real)[bm].mean())

    # (1) [ROBUST] bimodality recovered by the learned ancestral sampler, and NOT by the analytic head ------
    assert k_real < -0.3
    assert k_head > -0.15
    assert k_learned < 0.5 * k_real
    assert abs(k_learned - k_real) < abs(k_head - k_real)

    # (2) [ROBUST] scale-free shape distance (~5.6× better than head, seed-invariant) ----------------------
    ed_std_learned = energy_distance(standardize(f_learned), standardize(f_real))
    ed_std_head = energy_distance(standardize(f_head), standardize(f_real))
    assert ed_std_learned < 0.4 * ed_std_head

    # (3) sanity: the a=−1 ancestral sampler produced a sane N(0,~Ĉ_n), not a divergent draw (covariance is
    #     seed-sensitive and NOT tightened by a=−1; reported in the doc, not gated) -------------------------
    Chat = op._dense() - op.tau * torch.eye(op.d, dtype=x_real.dtype, device=dev)
    cov_relerr = float(((x_learned @ x_learned.T) / x_learned.shape[1] - Chat).norm() / Chat.norm())
    assert cov_relerr < 0.5
