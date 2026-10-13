"""The colored-EDM denoiser ``D_θ``, its loss, and the validation toys (Phase V2).

Fast structural tests pin the wiring (``D_θ`` collapses to the head when the body is zero, the loss/weight/
sampler contracts, the toys' covariance/non-Gaussianity structure). Three ``@pytest.mark.slow`` gates actually
train (GPU if available): (1) on the block GMM the learned ``D_θ`` beats the analytic Wiener head; (2) on the
variance-matched Gaussian reference the body stays ≈0 (the head is already exact there); (3) the
**honest-boundary control** — on ``GMMinDCT`` the non-Gaussianity is strong but *delocalised* (out-of-model),
so the body stays ≈0 despite real structure, pinning the in-model boundary the positive gate relies on.
"""

from __future__ import annotations

import math

import pytest
import torch

from ccd.denoiser import (
    BlockGMM, ColoredDenoiser, ColoredEDM, GMMinDCT, VNNBody,
    dct_matrix, edm_loss, edm_weight, sample_F_loguniform,
)
from ccd.forward import ColoredForward, forward_draw, phi_power
from ccd.operators import spectrum_bracket
from _oracles import rel_err


def _pipeline(d=16, n_blocks=4, *, phi=None, K=64, dtype=torch.float64, seed=0, gaussian=False):
    toy = BlockGMM(d=d, n_blocks=n_blocks, sep=0.95, idio_frac=0.2, tau=0.05, dtype=dtype)
    op = toy.make_operator(n=1500, generator=torch.Generator().manual_seed(seed), gaussian=gaussian)
    lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(seed + 1))
    fwd = ColoredForward(op, lo, hi, phi=phi or phi_power(-1.0), K=K)
    body = VNNBody(fwd, width=16, n_layers=2, K_gf=4).to(dtype)
    return toy, fwd, ColoredDenoiser(ColoredEDM(fwd), body)


# ------------------------------------------------------------------ D_θ structure
def test_dtheta_equals_head_when_body_zero():
    # F_θ ≡ 0 (decoder zeroed) ⇒ D_θ = c_skip x + c_out·0 = c_skip x = the analytic head, exactly.
    toy, fwd, den = _pipeline()
    with torch.no_grad():
        for p in den.body.decoder.parameters():
            p.zero_()
    x = toy.sample(6, generator=torch.Generator().manual_seed(3))
    assert rel_err(den(x, 1.0), den.head(x, 1.0)) == 0.0


@pytest.mark.parametrize("shape", [(16,), (16, 5)])
def test_dtheta_shapes(shape):
    toy, fwd, den = _pipeline()
    x = torch.randn(shape, dtype=torch.float64)
    assert den(x, 1.0).shape == torch.Size(shape)


def test_rejects_mismatched_fwd():
    toy, fwd, den = _pipeline()
    _, fwd2, _ = _pipeline(seed=5)
    other_body = VNNBody(fwd2, width=8, n_layers=1, K_gf=2).double()
    with pytest.raises(ValueError):
        ColoredDenoiser(den.edm, other_body)          # body wraps a different ColoredForward


# ------------------------------------------------------------------ loss / weight / F sampler
def test_edm_weight_positive_and_decreasing():
    toy, fwd, den = _pipeline()
    ws = [edm_weight(fwd, F) for F in (0.3, 1.0, 4.0)]
    assert all(w > 0 for w in ws) and ws[0] > ws[1] > ws[2]   # 1/mean posterior variance ↓ in F


def test_sample_F_loguniform_range_shape_and_guard():
    g = torch.Generator().manual_seed(0)
    Fs = [sample_F_loguniform(0.2, 8.0, generator=g) for _ in range(500)]
    assert all(0.2 <= F <= 8.0 for F in Fs)
    # log-uniform shape: ~half the draws fall below the geometric mean (linear-uniform gives ~0.14, so this
    # actually pins the *distribution*, not just the support).
    gmean = math.sqrt(0.2 * 8.0)
    assert 0.4 < sum(F < gmean for F in Fs) / len(Fs) < 0.6
    for bad in [(0.0, 1.0), (2.0, 1.0), (-1.0, 1.0)]:
        with pytest.raises(ValueError):
            sample_F_loguniform(*bad)


def test_edm_loss_matches_manual_and_grads_body():
    toy, fwd, den = _pipeline()
    x0 = toy.sample(8, generator=torch.Generator().manual_seed(4))
    eps = torch.randn_like(x0)
    F = 0.8
    manual = (den(forward_draw(fwd, x0, eps, F), F) - x0).pow(2).mean()
    assert rel_err(edm_loss(den, x0, F, eps=eps, weight=1.0), manual) < 1e-10
    # the DEFAULT-weight path (the one training uses) must actually apply edm_weight(fwd, F)
    assert rel_err(edm_loss(den, x0, F, eps=eps), edm_weight(fwd, F) * manual) < 1e-10
    assert abs(edm_weight(fwd, F) - 1.0) > 1e-3         # and that weight ≠ 1 here, so the check has teeth
    edm_loss(den, x0, F, eps=eps).backward()
    grads = [p.grad for p in den.body.parameters()]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0.0


# ------------------------------------------------------------------ BlockGMM toy structure
def test_block_variance_matched():
    # the block GMM and its Gaussian reference share the covariance S (differ only in higher cumulants).
    toy = BlockGMM(d=32, n_blocks=8, sep=0.97, idio_frac=0.15, dtype=torch.float64)
    Xm = toy.sample(60000, generator=torch.Generator().manual_seed(0))
    Xg = toy.sample(60000, generator=torch.Generator().manual_seed(1), gaussian=True)
    assert rel_err((Xm @ Xm.T) / Xm.shape[1], (Xg @ Xg.T) / Xg.shape[1]) < 0.05


def test_block_node_marginals_are_bimodal():
    # node-domain non-Gaussianity: each block-node marginal is sub-Gaussian (bimodal) ⇒ negative kurtosis.
    toy = BlockGMM(d=32, n_blocks=8, sep=0.97, idio_frac=0.15, dtype=torch.float64)
    X = toy.sample(60000, generator=torch.Generator().manual_seed(0))
    kurt = (X ** 4).mean(1) / (X ** 2).mean(1) ** 2 - 3.0
    assert float(kurt.max()) < -0.3


def test_block_covariance_is_block_structured():
    toy = BlockGMM(d=32, n_blocks=8, sep=0.97, idio_frac=0.15, dtype=torch.float64)
    X = toy.sample(60000, generator=torch.Generator().manual_seed(0))
    C = (X @ X.T) / X.shape[1]
    same = toy.block_of[:, None] == toy.block_of[None, :]
    assert float(C[same].abs().mean()) > 20.0 * float(C[~same].abs().mean())


def test_block_rejects_indivisible_d():
    with pytest.raises(ValueError):
        BlockGMM(d=30, n_blocks=8)


# ------------------------------------------------------------------ GMMinDCT (out-of-model control)
def test_dct_matrix_orthonormal():
    W = dct_matrix(24, dtype=torch.float64)
    assert float((W @ W.T - torch.eye(24, dtype=torch.float64)).abs().max()) < 1e-12


def test_dct_toy_variance_matched_and_dct_aligned():
    toy = GMMinDCT(d=24, bimodal_axes=(1, 2), sep=0.9, dtype=torch.float64)
    Xm = toy.sample(60000, generator=torch.Generator().manual_seed(0))
    Xg = toy.sample(60000, generator=torch.Generator().manual_seed(1), gaussian=True)
    assert rel_err((Xm @ Xm.T) / Xm.shape[1], (Xg @ Xg.T) / Xg.shape[1]) < 0.05
    S = (Xm @ Xm.T) / Xm.shape[1] + toy.tau * torch.eye(24, dtype=torch.float64)
    DSD = toy.W @ S @ toy.W.T                                  # S is diagonalised by the DCT basis
    off = (DSD - torch.diag(torch.diagonal(DSD))).abs().max() / torch.diagonal(DSD).abs().max()
    assert float(off) < 0.05


def test_dct_toy_nongaussianity_is_delocalised():
    # the crux of the out-of-model control: GMMinDCT is STRONGLY bimodal in the DCT (eigen) basis but ≈Gaussian
    # in the NODE basis — the non-Gaussian direction is a delocalised eigenvector, so a node-domain σ sees each
    # eigen-coordinate diluted by ~1/√d and cannot bend it (why the trained body stays inert, gate #3 below).
    toy = GMMinDCT(d=48, bimodal_axes=(1, 2, 3, 4, 5, 6), sep=0.95, rho=1.0, dtype=torch.float64)
    X = toy.sample(60000, generator=torch.Generator().manual_seed(0))

    def kurt(V):
        Vc = V - V.mean(1, keepdim=True)
        return Vc.pow(4).mean(1) / Vc.pow(2).mean(1).pow(2) - 3.0

    kz = kurt(toy.W @ X)                                        # DCT (eigen) coordinates
    kx = kurt(X)                                               # node coordinates
    bax = list(toy.non_gaussian_axes)
    assert float(kz[bax].max()) < -0.5                         # strongly bimodal along the DCT modes (real structure)
    assert float(kx.abs().max()) < 0.2                         # yet every node marginal is ≈Gaussian (delocalised)


# ------------------------------------------------------------------ slow training gates (GPU-preferred)
def _train_block(gaussian, *, dev, steps=1200, seed=7):
    """Train ``D_θ`` on the block GMM (or its Gaussian reference) with the coarse-to-fine colouring ``a=-1``
    (which preserves the high-variance factor modes longest, keeping the bimodal structure alive across F)."""
    toy = BlockGMM(d=48, n_blocks=8, rho=0.7, sep=0.98, idio_frac=0.1, tau=0.05, dtype=torch.float32, device=dev)
    op = toy.make_operator(n=6000, generator=torch.Generator(device=dev).manual_seed(1), gaussian=gaussian)
    lo, hi = spectrum_bracket(op, generator=torch.Generator(device=dev).manual_seed(2))
    fwd = ColoredForward(op, lo, hi, phi=phi_power(-1.0), K=32)
    den = ColoredDenoiser(ColoredEDM(fwd), VNNBody(fwd, width=96, n_layers=3, K_gf=5))
    opt = torch.optim.Adam(den.body.parameters(), lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(steps):
        F = sample_F_loguniform(0.1, 3.0)
        x0 = toy.sample(256, generator=gen, gaussian=gaussian)
        opt.zero_grad()
        edm_loss(den, x0, F, generator=gen, c_in_kw=dict(tol=1e-4, maxiter=300)).backward()
        opt.step()
    den.body.eval()
    return toy, fwd, den, gen


def _improvement(toy, fwd, den, gen, F, gaussian):
    with torch.no_grad():
        x0 = toy.sample(6000, generator=gen, gaussian=gaussian)
        xF = forward_draw(fwd, x0, torch.randn_like(x0), F)
        D, h = den(xF, F, tol=1e-4, maxiter=300), den.head(xF, F)
        impr = 1.0 - float((D - x0).pow(2).mean()) / float((h - x0).pow(2).mean())
        delta = float((D - h).norm() / h.norm())
    return impr, delta


@pytest.mark.slow
def test_beats_wiener_head_on_block_gmm():
    # the load-bearing V2 gate: on node-local (block) non-Gaussianity the learned D_θ beats the analytic
    # Wiener head. Observed ≈+22% at F=0.7 after 1400 steps; assert >8% with an active body.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    toy, fwd, den, gen = _train_block(False, dev=dev)
    impr, delta = _improvement(toy, fwd, den, gen, 0.7, gaussian=False)
    assert impr > 0.08 and delta > 0.10


@pytest.mark.slow
def test_body_vanishes_on_gaussian_reference():
    # control: on the variance-matched Gaussian reference the head is the exact MMSE, so the trained body
    # must stay ≈0 and add no spurious "gain" (both toys share S, so this isolates the higher cumulants).
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    toy, fwd, den, gen = _train_block(True, dev=dev, seed=9)
    for F in (0.3, 0.7, 1.0):
        impr, delta = _improvement(toy, fwd, den, gen, F, gaussian=True)
        assert delta < 0.05 and impr < 0.02


def _train_dct(dev, *, steps=1500, seed=7):
    """Train ``D_θ`` on the GMMinDCT out-of-model toy under the coarse-to-fine colouring ``a=-1`` (same recipe
    as ``_train_block``; a=-1 keeps the high-variance low-frequency DCT modes — the bimodal ones — alive across
    F, giving the body its *best* chance). The non-Gaussianity lives along delocalised DCT eigenvectors, out of
    the node-local VNN's reach."""
    toy = GMMinDCT(d=48, bimodal_axes=(1, 2, 3, 4, 5, 6), sep=0.95, rho=1.0, tau=0.05, dtype=torch.float32, device=dev)
    op = toy.make_operator(n=6000, generator=torch.Generator(device=dev).manual_seed(1))
    lo, hi = spectrum_bracket(op, generator=torch.Generator(device=dev).manual_seed(2))
    fwd = ColoredForward(op, lo, hi, phi=phi_power(-1.0), K=32)
    den = ColoredDenoiser(ColoredEDM(fwd), VNNBody(fwd, width=96, n_layers=3, K_gf=5))
    opt = torch.optim.Adam(den.body.parameters(), lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(steps):
        F = sample_F_loguniform(0.1, 3.0)
        x0 = toy.sample(256, generator=gen)
        opt.zero_grad()
        edm_loss(den, x0, F, generator=gen, c_in_kw=dict(tol=1e-4, maxiter=300)).backward()
        opt.step()
    den.body.eval()
    return toy, fwd, den, gen


@pytest.mark.slow
def test_body_vanishes_on_dct_out_of_model():
    # the HONEST-BOUNDARY control: GMMinDCT is strongly non-Gaussian (DCT-axis kurtosis ≈ −1.6, see
    # test_dct_toy_nongaussianity_is_delocalised) but that structure is *delocalised*, so the node-local VNN
    # cannot exploit it — the trained body stays inert (impr ≈ 0, delta ≈ 0), the mirror image of the +22%
    # BlockGMM gain. This pins the in-model boundary experimentally: D_θ is not a universal non-Gaussian
    # learner, only a node-local one — so the positive gate isn't just generic overfitting capacity. (Observed
    # impr 0.000 and delta ≤ 0.009 at 1500 steps; the block gate reaches impr > 0.08, delta > 0.10.)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    toy, fwd, den, gen = _train_dct(dev)
    for F in (0.3, 0.7, 1.0):
        impr, delta = _improvement(toy, fwd, den, gen, F, gaussian=False)
        assert impr < 0.02 and delta < 0.04


# ------------------------------------------------------------------ per-band (whitened) loss
# `edm_loss_perband` supplies the faithful c_out^-2 weight that `edm_loss` deliberately scalarises.
# On a rank-deficient Ĉ_n the weight is unbounded (c_out² = Σ_0·Σ_F/V_F vanishes wherever Ĉ_n does),
# so the support restriction is a precondition, not a refinement — these tests pin both halves.

def _eig_of_S(fwd):
    """Dense (evals, evecs) of S = Ĉ_n + τI, descending. Test-side only — src stays matrix-free."""
    d = fwd.op.d
    S = torch.stack([fwd.op.matvec(e) for e in torch.eye(d, dtype=torch.float64)], dim=1)
    w, V = torch.linalg.eigh(0.5 * (S + S.transpose(0, 1)))
    idx = torch.argsort(w, descending=True)
    return w[idx].contiguous(), V[:, idx].contiguous()


def test_perband_uniform_full_rank_equals_edm_loss():
    """`uniform` at full rank is `edm_loss` exactly: U is orthogonal, so Parseval makes the
    projection a no-op and the weight is the same scalar. Pins the projection wiring."""
    from ccd.denoiser import edm_loss_perband

    toy, fwd, den = _pipeline()
    ev, U = _eig_of_S(fwd)
    x0 = toy.sample(8, generator=torch.Generator().manual_seed(11))
    eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(12), dtype=x0.dtype)
    for F in (0.3, 2.0):
        a = edm_loss_perband(den, x0, F, evals=ev, evecs=U, mode="uniform", rank=None, eps=eps)
        b = edm_loss(den, x0, F, eps=eps)
        assert rel_err(a, b) < 1e-10


def test_perband_scalar_mode_is_edm_loss():
    from ccd.denoiser import edm_loss_perband

    toy, fwd, den = _pipeline()
    ev, U = _eig_of_S(fwd)
    x0 = toy.sample(6, generator=torch.Generator().manual_seed(13))
    eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(14), dtype=x0.dtype)
    a = edm_loss_perband(den, x0, 1.0, evals=ev, evecs=U, mode="scalar", eps=eps)
    assert rel_err(a, edm_loss(den, x0, 1.0, eps=eps)) < 1e-12


def test_perband_whiten_matches_closed_form():
    """⟨r, W r⟩ with W = c_out^-2 equals the per-band whitened residual computed directly."""
    from ccd.denoiser import cout_sq_bands, edm_loss_perband

    toy, fwd, den = _pipeline()
    ev, U = _eig_of_S(fwd)
    x0 = toy.sample(8, generator=torch.Generator().manual_seed(15))
    eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(16), dtype=x0.dtype)
    F, k = 1.5, 8
    got = edm_loss_perband(den, x0, F, evals=ev, evecs=U, mode="whiten", rank=k, eps=eps)
    with torch.no_grad():
        r = den(forward_draw(fwd, x0, eps, F), F) - x0
        r_eig = U[:, :k].transpose(0, 1) @ r
        want = ((r_eig ** 2) / cout_sq_bands(fwd, F, ev[:k])[:, None]).mean()
    assert rel_err(got, want) < 1e-10


def test_perband_ignores_the_discarded_subspace():
    """A residual living entirely outside the retained support contributes exactly zero — which is
    the whole point: the null space of Ĉ_n must not drive training."""
    from ccd.denoiser import cout_sq_bands

    toy, fwd, den = _pipeline()
    ev, U = _eig_of_S(fwd)
    k = 6
    r_tail = U[:, k:] @ torch.randn(U.shape[0] - k, 4, generator=torch.Generator().manual_seed(17),
                                    dtype=torch.float64)
    w = 1.0 / cout_sq_bands(fwd, 1.0, ev[:k])
    assert float((w[:, None] * (U[:, :k].transpose(0, 1) @ r_tail) ** 2).mean()) < 1e-24


def test_perband_weight_is_nonnegative_on_a_rank_deficient_spectrum():
    """Σ_0 = λ−τ goes *negative* from round-off on exactly-null directions; the clamp keeps the
    quadratic form PSD instead of silently rewarding error there."""
    from ccd.denoiser import cout_sq_bands

    toy, fwd, den = _pipeline()
    ev, _ = _eig_of_S(fwd)
    ev_deficient = torch.cat([ev[:-3], torch.full((3,), fwd.op.tau, dtype=ev.dtype)])   # exact nulls
    for F in (0.05, 1.0, 20.0):
        c2 = cout_sq_bands(fwd, F, ev_deficient)
        assert bool((c2 >= 0).all()) and bool(torch.isfinite(c2).all())
        assert float(c2[-3:].abs().max()) < 1e-12                     # null directions carry no signal


def test_perband_whiten_rejects_a_support_touching_the_null_space():
    from ccd.denoiser import edm_loss_perband

    toy, fwd, den = _pipeline()
    ev, U = _eig_of_S(fwd)
    ev_bad = ev.clone(); ev_bad[-1] = fwd.op.tau                       # an exactly-null band inside support
    x0 = toy.sample(4, generator=torch.Generator().manual_seed(18))
    with pytest.raises(ValueError, match="vanishes on the retained support"):
        edm_loss_perband(den, x0, 1.0, evals=ev_bad, evecs=U, mode="whiten", rank=None)


def test_perband_weight_contributes_no_gradient_to_phi():
    """The gauge-critical wiring test.

    A learnable ``φ`` *legitimately* receives gradient through ``forward_draw`` and ``c_out`` — that is
    how the L8–L10 learnable-φ runs train. What must **not** happen is the *weight* contributing: an
    undetached ``W = 1/c_out²`` is minimised by ``φ→∞`` (blow the weight up, drive the loss to zero),
    which would silently corrupt those runs. So this is a differential test: our loss must reproduce
    the hand-built **detached-weight** gradient exactly, and the undetached weight must differ.
    """
    from ccd.forward import PhiLogPoly
    from ccd.denoiser import cout_sq_bands, edm_loss_perband

    def build():
        torch.manual_seed(0)                                       # VNNBody init draws from the GLOBAL rng:
        toy = BlockGMM(d=16, n_blocks=4, sep=0.95, idio_frac=0.2,  # without this the three builds get
                       tau=0.05, dtype=torch.float64)              # different bodies and the grads are
        op = toy.make_operator(n=1500, generator=torch.Generator().manual_seed(0))   # incomparable.
        lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(1))
        phi = PhiLogPoly.from_power(-1.0, lo, hi, degree=2, dtype=torch.float64)   # LEARNABLE φ
        fwd = ColoredForward(op, lo, hi, phi=phi, K=64)
        den = ColoredDenoiser(ColoredEDM(fwd), VNNBody(fwd, width=16, n_layers=2, K_gf=4).to(torch.float64))
        return toy, fwd, phi, den

    F, k = 1.0, 6
    x0 = BlockGMM(d=16, n_blocks=4, sep=0.95, idio_frac=0.2, tau=0.05,
                  dtype=torch.float64).sample(6, generator=torch.Generator().manual_seed(19))
    eps = torch.randn(x0.shape, generator=torch.Generator().manual_seed(20), dtype=torch.float64)

    def phi_grad(kind):
        toy, fwd, phi, den = build()
        ev, U = _eig_of_S(fwd)
        if kind == "ours":
            loss = edm_loss_perband(den, x0, F, evals=ev, evecs=U, mode="whiten", rank=k, eps=eps)
        else:
            r = den(forward_draw(fwd, x0, eps, F), F) - x0
            w = 1.0 / cout_sq_bands(fwd, F, ev[:k])                # φ-dependent, grad-carrying
            if kind == "detached":
                w = w.detach()
            loss = (w[:, None] * (U[:, :k].transpose(0, 1) @ r).pow(2)).mean()
        loss.backward()
        body_grad = max(float(p.grad.abs().max()) for p in den.body.parameters() if p.grad is not None)
        return phi.theta.grad.clone(), body_grad

    g_ours, body_ours = phi_grad("ours")
    g_detached, _ = phi_grad("detached")
    g_leaky, _ = phi_grad("leaky")

    assert torch.allclose(g_ours, g_detached, rtol=0, atol=0)      # our W is detached — bit-identical
    assert float((g_ours - g_leaky).abs().max()) > 1e-6            # ...and it genuinely matters
    assert body_ours > 0.0                                         # the body still learns


def test_perband_support_restores_weight_to_the_signal_carrying_modes():
    """The measured motivation: unrestricted, the whitened weight puts ~all of its mass on the
    near-null bulk; restricted, the top modes get a material share."""
    from ccd.denoiser import cout_sq_bands

    toy, fwd, den = _pipeline(d=32, n_blocks=4)
    ev, _ = _eig_of_S(fwd)
    F = 1.0
    w_all = 1.0 / cout_sq_bands(fwd, F, ev).clamp_min(1e-30)
    w_top = 1.0 / cout_sq_bands(fwd, F, ev[:8])
    share_all = float(w_all[:8].sum() / w_all.sum())
    share_top = 1.0
    assert share_all < 0.25 < share_top          # restriction materially re-weights toward the top


# ---------------------------------------------------------------- the head identity (Tier-1 STEP 0)
# c_out² = Σ_0Σ_F/V_F *is* the posterior variance of x_0 given x_F, so the analytic head's whitened
# per-band loss is the constant 1 — exactly, for every band, every F and every colouring. That makes
# the whitened objective's cross-arm comparison a parameter-free point prediction rather than a
# matched-SNR exercise, and it is the premise the whole Tier-1 gate rests on. These pin it.

def _identity_pipeline(a=-1.0, *, n=600, linear=False):
    """A small pipeline whose x0 is the WHOLE operator cloud.

    ``n`` is well below :func:`_pipeline`'s 1500 on purpose: the identity needs ``x0`` to be all of
    ``X`` (so ``E[x0x0ᵀ] = XXᵀ/n = Ĉ_n`` holds by construction rather than in expectation), and every
    ``edm_loss_perband`` call then costs ``O(d·n²)`` in the Chebyshev applies.
    """
    torch.manual_seed(0)                                           # VNNBody init draws from GLOBAL rng
    toy = BlockGMM(d=16, n_blocks=4, sep=0.95, idio_frac=0.2, tau=0.05, dtype=torch.float64)
    op = toy.make_operator(n=n, generator=torch.Generator().manual_seed(0))
    lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(1))
    fwd = ColoredForward(op, lo, hi, phi=phi_power(a), K=64)
    body = VNNBody(fwd, width=16, n_layers=2, K_gf=4,
                   act=torch.nn.Identity() if linear else None).to(torch.float64)
    return toy, fwd, ColoredDenoiser(ColoredEDM(fwd), body)


def _head_only(den):
    """Zero the (bias-free) decoder so ``F_θ ≡ 0`` and ``D_θ`` is exactly the analytic head."""
    with torch.no_grad():
        den.body.decoder.weight.zero_()
    return den


def _mean_whitened(den, x0, F, ev, U, k, *, n_draw, seed):
    """Whitened per-band loss averaged over ``n_draw`` independent ``eps`` draws."""
    from ccd.denoiser import edm_loss_perband

    with torch.no_grad():
        vals = [float(edm_loss_perband(
            den, x0, F, evals=ev, evecs=U, mode="whiten", rank=k, scale=1.0,
            eps=torch.randn(x0.shape, generator=torch.Generator().manual_seed(seed + i),
                            dtype=x0.dtype)))
            for i in range(n_draw)]
    return sum(vals) / len(vals)


@pytest.mark.parametrize("a", [0.0, -1.0, -2.0])
def test_perband_whiten_head_identity_is_one_for_every_colouring(a):
    """E[(c_skip x_F − x_0)_k²] = c_out,k² exactly ⇒ the whitened head loss is 1, independent of φ.

    ``x0`` is the **whole** cloud that defines the operator, because the identity needs
    ``E[x0 x0ᵀ] = Ĉ_n`` and ``Ĉ_n = XXᵀ/n`` is that cloud's second moment *by construction*. Using
    the full X removes the x0-sampling noise, leaving only the ``eps`` draw.
    """
    toy, fwd, den = _identity_pipeline(a)
    _head_only(den)
    ev, U = _eig_of_S(fwd)
    x0 = fwd.op.X                                                  # the very columns that built Ĉ_n
    for F in (0.05, 1.0, 20.0):
        got = _mean_whitened(den, x0, F, ev, U, 8, n_draw=32, seed=100)
        assert abs(got - 1.0) < 0.02, f"a={a} F={F}: whitened head loss {got:.5f} != 1"


def test_perband_whiten_head_identity_is_rank_independent():
    """The target is 1 per band, so the mean over any retained support is 1 — no rank dependence."""
    toy, fwd, den = _identity_pipeline()
    _head_only(den)
    ev, U = _eig_of_S(fwd)
    got = [_mean_whitened(den, fwd.op.X, 1.0, ev, U, k, n_draw=32, seed=200) for k in (4, 8, 12)]
    assert max(abs(g - 1.0) for g in got) < 0.02
    assert max(got) - min(got) < 0.02


def test_perband_whiten_head_identity_predicts_a_second_moment_mismatch():
    """The identity's precondition, made quantitative: if ``x0``'s per-band second moment is ``Sy_k``
    rather than ``Σ_0,k``, the loss is ``1 + (Σ_F/V_F)_k (Sy_k/Σ_0,k − 1)``, NOT 1.

    This is the trap: evaluating on a held-out cloud silently converts a parameter-free constant into
    an **arm-dependent** number, because the leading factor ``Σ_F/V_F`` depends on ``φ`` and ``F``. A
    few-percent artifact of exactly this shape was measured in `scripts/calibrate_scale.py`, which drew
    ``x0`` from the held-out ``Xg_real`` instead of the operator's own ``Xg_op``.
    """
    toy, fwd, den = _identity_pipeline()
    _head_only(den)
    ev, U = _eig_of_S(fwd)
    k, F = 8, 1.0
    # a deliberately mismatched cloud: treble the leading band, so Sy_0/Σ_0,0 = 9
    X = fwd.op.X
    x0 = X + 2.0 * (U[:, :1] @ (U[:, :1].transpose(0, 1) @ X))
    tau = fwd.op.tau
    sig0, lam = (ev[:k] - tau), ev[:k]
    e = torch.exp(-F * fwd.phi(lam))
    ratio = (lam * (1.0 - e)) / (lam - tau * e)                    # Σ_F/V_F ∈ (0, 1]
    sy = (U[:, :k].transpose(0, 1) @ x0).pow(2).mean(dim=1)        # realised per-band second moment
    want = float((1.0 + ratio * (sy / sig0 - 1.0)).mean())
    got = _mean_whitened(den, x0, F, ev, U, k, n_draw=32, seed=300)
    assert want - 1.0 > 0.05, f"the mismatch must be big enough to be a real test; got {want:.4f}"
    assert abs(got - want) / want < 0.02, f"measured {got:.5f} vs closed form {want:.5f}"


def test_perband_whiten_linear_body_is_one_plus_mean_gain_squared():
    """The sharp form: for a *linear* body of per-band gain ``p_k`` the whitened loss is exactly
    ``1 + mean_k p_k²`` — ``c_in² = 1/V_F`` cancels the cross term — so 1 is an attained floor.

    With ``act=Identity`` and no bias anywhere on the signal path, ``VNNBody`` is a polynomial in
    ``S̃`` and hence band-diagonal, so ``p_k`` is well defined as ``diag(Uᵀ F_θ(U))``. This is what
    makes the Tier-1 gate readable: the trained linear body's excess over 1 *is* its mean squared gain.
    """
    toy, fwd, den = _identity_pipeline(linear=True)
    ev, U = _eig_of_S(fwd)
    k, F = 8, 1.0
    with torch.no_grad():
        P = U.transpose(0, 1) @ den.body(U, F)
        p = torch.diagonal(P)[:k]
        off = float((P - torch.diag(torch.diagonal(P))).abs().max())
    assert off < 1e-8, f"a linear body must be band-diagonal; off-diagonal {off:.3e}"
    want = float((1.0 + p.pow(2)).mean())
    got = _mean_whitened(den, fwd.op.X, F, ev, U, k, n_draw=32, seed=400)
    assert want > 1.02, "the body's gain must be large enough for this to be a real test"
    assert abs(got - want) / want < 0.02, f"measured {got:.5f} vs 1+mean p² = {want:.5f}"
    assert got > 1.0                                               # ...and the floor holds
