"""End-to-end generation gate (Phase V3b) — the payoff of the whole VNN-denoiser workstream.

Train the colored-EDM denoiser ``D_θ`` on the in-model block GMM under the whitened-VP colouring
``a=0`` (``φ≡1``: uniform clock, prior ``N(0,S)`` reached at a moderate ``F_max``), then generate with the
learned reverse SDE (:class:`ccd.reverse.LearnedReverseSDE`) and show it crosses the **App-B.4 boundary the
analytic Gaussian score cannot**: it recovers the block factors' *bimodality*.

The comparison is a clean paired ablation. The head baseline is the **same** ``integrate`` code path with the
body's decoder zeroed (⇒ ``D_θ = c_skip(S) x`` = the exact Gaussian head, proven step-for-step equal to
:class:`GaussianReverseSDE` in the V3a collapse test), run on the **same terminal draws and the same
step-noise generator seed**. So the *only* difference between the two sample sets is the trained body — any
gap in bimodality, energy distance, or covariance is attributable to it alone.

A 4-seed sweep (see ``docs/agent_summaries/v3b_generation_gate.md``) separated what is **robust** from what is
**seed-sensitive**, and the gate is built on the robust part:

* **bimodality recovered [robust]** — the learned block-factor marginals are strongly bimodal (mean excess
  kurtosis ``≈ −1.85`` across every seed, vs real ``≈ −1.77``), while the head stays Gaussian (``≈ +0.02``);
  the analytic score *structurally* cannot produce this. This is the App-B.4 boundary.
* **scale-free shape distance ↓ [robust]** — standardising each block-factor marginal strips the covariance
  scale, isolating the recovered non-Gaussian *shape*; there the learned energy distance to real is ``≈ 5.6×``
  smaller than the head's, invariant to seed. This is the primary distributional gate.
* **covariance / ambient energy distance [seed-sensitive → NOT gated]** — a *good* init makes the learned
  sampler beat the head on ambient energy distance and covariance too, but the covariance faithfulness of an
  EM + learned-score sampler tracks body quality, and GPU-kernel nondeterminism makes even a fixed seed a
  fresh draw over ~3k training steps (the ambient-ED tail runs to ``≈ 2×`` the head's). So we do **not** gate
  on these — only a divergence sanity that a valid ``N(0,~Ĉ_n)`` was produced; the reference numbers and the
  full 4-seed spread are in the summary doc.

The training seed is pinned for reproducibility of the setup (not the exact FP trajectory); the *robust* gates
hold for every seed in the sweep, so pinning does not manufacture the result. Marked ``slow`` (trains ~3k
steps + generates through 128 reverse steps; GPU-preferred but CPU-tolerable at ``d=48``). Fast
structural/collapse coverage of the learned sampler lives in ``test_learned.py``.
"""

from __future__ import annotations

import copy

import pytest
import torch

from ccd.denoiser import BlockGMM, ColoredDenoiser, ColoredEDM, VNNBody, edm_loss, sample_F_loguniform
from ccd.forward import ColoredForward, forward_draw, phi_power, terminal_draw
from ccd.operators import spectrum_bracket
from ccd.reverse import LearnedReverseSDE, reverse_f_grid
from _gen_metrics import block_avg, energy_distance, excess_kurtosis, standardize   # shared test-only metrics


# ------------------------------------------------------------------ train D_θ under a=0

def _train_a0(dev, *, steps=3000, envelope=True, k0=2.0, init_seed=0, data_seed=7):
    """Train ``D_θ`` on the block GMM under the whitened-VP colouring ``a=0`` over ``[F_min, F_max]``.

    Envelope on (``η=m_F^{k0}``, forcing ``F_θ→0`` at high noise) is the V1 boundary condition
    ``E[x_0|x_F]→0`` and, on a good init, keeps the learned sampler's covariance below the head's — see the
    summary doc. The global seed pins the body init + the F-schedule for reproducibility. Returns the toy,
    forward, denoiser, and the (op, lo, hi) bracket.
    """
    torch.manual_seed(init_seed)                              # body init + the (generator-less) F-schedule
    toy = BlockGMM(d=48, n_blocks=8, rho=0.7, sep=0.98, idio_frac=0.1, tau=0.05, dtype=torch.float32, device=dev)
    op = toy.make_operator(n=6000, generator=torch.Generator(device=dev).manual_seed(1))
    lo, hi = spectrum_bracket(op, generator=torch.Generator(device=dev).manual_seed(2))
    fwd = ColoredForward(op, lo, hi, phi=phi_power(0.0), K=32)
    body = VNNBody(fwd, width=96, n_layers=3, K_gf=5, mehler_envelope=envelope, k0=k0)
    den = ColoredDenoiser(ColoredEDM(fwd), body)
    opt = torch.optim.Adam(den.body.parameters(), lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(data_seed)  # x0 draws + EDM noise
    for _ in range(steps):
        F = sample_F_loguniform(0.05, 6.0)
        x0 = toy.sample(256, generator=gen)
        opt.zero_grad()
        edm_loss(den, x0, F, generator=gen, c_in_kw=dict(tol=1e-4, maxiter=300)).backward()
        opt.step()
    den.body.eval()
    return toy, fwd, den, op, lo, hi


@pytest.mark.slow
def test_generation_recovers_bimodal_modes():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    toy, fwd, den, op, lo, hi = _train_a0(dev)

    # paired samplers: the head is the SAME integrate path with the body zeroed (⇒ Gaussian head) ---------
    learned = LearnedReverseSDE(den, K=32, c_in_kw=dict(tol=1e-4, maxiter=300))
    head_den = copy.deepcopy(den)
    with torch.no_grad():
        for p in head_den.body.decoder.parameters():
            p.zero_()
    # c_in feeds the zeroed body ⇒ its value is discarded; loose tol + require_converged=False so the (still
    # executed) solve is cheap and can never raise on a non-converged-but-irrelevant head solve.
    head = LearnedReverseSDE(head_den, K=32, c_in_kw=dict(tol=1e-3, maxiter=200, require_converged=False))

    N_GEN, N_STEPS, F_MAX = 4000, 128, 6.0
    grid = reverse_f_grid(F_MAX, 0.05, N_STEPS, spacing="equidistribute", weight="abs",
                          phi=phi_power(0.0), tau=0.05, lo=lo, hi=hi)
    xT = terminal_draw(op, batch=N_GEN, generator=torch.Generator(device=dev).manual_seed(100))
    x_learned = learned.integrate(xT, grid, generator=torch.Generator(device=dev).manual_seed(200))
    x_head = head.integrate(xT, grid, generator=torch.Generator(device=dev).manual_seed(200))
    x_real = toy.sample(N_GEN, generator=torch.Generator(device=dev).manual_seed(300))

    bm = list(toy.non_gaussian_blocks)
    f_learned, f_head, f_real = block_avg(toy, x_learned), block_avg(toy, x_head), block_avg(toy, x_real)
    k_learned = float(excess_kurtosis(f_learned)[bm].mean())
    k_head = float(excess_kurtosis(f_head)[bm].mean())
    k_real = float(excess_kurtosis(f_real)[bm].mean())

    # (1) [ROBUST] bimodality recovered by the learned sampler, and NOT by the analytic head --------------
    assert k_real < -0.3                                       # the toy really is bimodal
    assert k_head > -0.15                                      # the Gaussian head cannot be bimodal (≈0)
    assert k_learned < 0.5 * k_real                            # learned ≥ half as bimodal as the real data
    assert abs(k_learned - k_real) < abs(k_head - k_real)      # and much closer to real than the head is

    # (2) [ROBUST] scale-free shape distance: standardising per block strips the covariance scale, leaving
    #     pure shape — the learned marginals match the real bimodal shape ~5.6× better than the head, and this
    #     holds across every seed in the sweep (the primary distributional gate). ---------------------------
    ed_std_learned = energy_distance(standardize(f_learned), standardize(f_real))
    ed_std_head = energy_distance(standardize(f_head), standardize(f_real))
    assert ed_std_learned < 0.4 * ed_std_head

    # (3) sanity only. The covariance / ambient-energy-distance comparison to the head is genuinely
    #     seed-sensitive with a heavy tail — a good init beats the head on both, but an EM + learned-score
    #     sampler's covariance faithfulness tracks body quality, and GPU-kernel nondeterminism makes even a
    #     fixed seed a fresh draw over 3k steps. So we do NOT gate on it (the 4-seed spread is in the doc); we
    #     only guard that the sampler produced a sane ``N(0,~Ĉ_n)`` draw rather than diverging. --------------
    Chat = op._dense() - op.tau * torch.eye(op.d, dtype=x_real.dtype, device=dev)
    cov_relerr = float(((x_learned @ x_learned.T) / x_learned.shape[1] - Chat).norm() / Chat.norm())
    assert cov_relerr < 0.5                                    # covariance is right up to the score-error budget
