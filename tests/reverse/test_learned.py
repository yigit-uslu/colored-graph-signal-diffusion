"""The learned-score reverse SDE sampler (Phase V3a).

The load-bearing test is the **analytic collapse**: with the learned denoiser forced to the Gaussian head
(``D_θ = c_skip(S) x``), the learned reverse-SDE drift/step must reproduce the analytic
:class:`GaussianReverseSDE` Euler–Maruyama drift/step to machine precision — this pins the folded-drift
algebra ``b = −½φ(S)x − a_F(S)(m_F(S)D_θ − x)`` and the ``Σ_F^{-1}`` cancellation. The rest are structural
(shapes/dtype/GPU, the end-to-end integrate, and the step guards).
"""

from __future__ import annotations

import pytest
import torch

from ccd.denoiser import BlockGMM, ColoredDenoiser, ColoredEDM, VNNBody
from ccd.forward import ColoredForward, phi_power, terminal_draw
from ccd.operators import apply_fun, spectrum_bracket
from ccd.reverse import GaussianReverseSDE, LearnedReverseSDE, reverse_f_grid
from _oracles import rel_err

# c_in is DISCARDED when the body is zeroed (D_θ = c_skip x), so it only has to converge — a loose tol keeps
# the collapse tests from spending minutes on an irrelevant tol=1e-10 CG solve.
CIK = dict(tol=1e-3, maxiter=200)
_HEAD_CACHE: dict = {}


def _head_sampler(a, *, d=16, K=200, dtype=torch.float64, dev="cpu"):
    """A LearnedReverseSDE whose denoiser is forced to the analytic head (decoder zeroed ⇒ F_θ≡0).

    Cached per colouring — the collapse suite reuses one operator/denoiser across all F parametrisations.
    """
    key = (a, d, K, dtype, dev)
    if key not in _HEAD_CACHE:
        toy = BlockGMM(d=d, n_blocks=4, sep=0.9, idio_frac=0.2, tau=0.05, dtype=dtype, device=dev)
        op = toy.make_operator(n=800, generator=torch.Generator(device=dev).manual_seed(0))
        lo, hi = spectrum_bracket(op, generator=torch.Generator(device=dev).manual_seed(1))
        fwd = ColoredForward(op, lo, hi, phi=phi_power(a), K=K)
        body = VNNBody(fwd, width=16, n_layers=2, K_gf=4).to(dtype)
        with torch.no_grad():
            for p in body.decoder.parameters():
                p.zero_()
        den = ColoredDenoiser(ColoredEDM(fwd), body)
        gauss = GaussianReverseSDE(op, lo, hi, phi=phi_power(a), K=K)
        _HEAD_CACHE[key] = (op, lo, hi, gauss, LearnedReverseSDE(den, K=K, c_in_kw=CIK))
    return _HEAD_CACHE[key]


# ------------------------------------------------------------------ the analytic collapse
@pytest.mark.parametrize("a", [0.0, 1.0, -1.0])
@pytest.mark.parametrize("F_a,F_b", [(1.0, 0.8), (3.0, 2.0), (0.5, 0.3)])
def test_drift_collapses_to_analytic(a, F_a, F_b):
    # learned drift with D_θ = head ⇒ x − ΔF·b(x,F_a) equals the analytic EM update apply_fun(em_drift).
    op, lo, hi, gauss, learned = _head_sampler(a)
    x = torch.randn(op.d, 4, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    analytic = apply_fun(gauss.em_drift_fn(F_a, F_b), op, x, lo, hi, gauss.K)
    learned_det = x - (F_a - F_b) * learned.drift(x, F_a)
    assert rel_err(learned_det, analytic) < 1e-6


@pytest.mark.parametrize("a", [0.0, 1.0, -1.0])
def test_em_step_collapses_to_gaussian(a):
    # full EM step (drift + noise) matches the Gaussian euler_maruyama step given the SAME noise draw.
    op, lo, hi, gauss, learned = _head_sampler(a)
    x = torch.randn(op.d, 3, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    g = gauss.step(x, 1.2, 0.9, scheme="euler_maruyama", generator=torch.Generator().manual_seed(123))
    l = learned.step(x, 1.2, 0.9, generator=torch.Generator().manual_seed(123))
    assert rel_err(l, g) < 1e-6


def test_diffusion_matches_gaussian():
    op, lo, hi, gauss, learned = _head_sampler(0.0)
    xi = torch.randn(op.d, 3, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    gn = apply_fun(gauss.diffusion_std_fn(), op, xi, lo, hi, gauss.K)
    ln = apply_fun(learned.diffusion_std_fn(), op, xi, lo, hi, learned.K)
    assert rel_err(ln, gn) == 0.0


@pytest.mark.parametrize("a", [0.0, 1.0, -1.0])
@pytest.mark.parametrize("F_a,F_b", [(1.2, 0.9), (3.0, 2.0), (6.0, 0.05)])
def test_ancestral_mean_collapses_to_gaussian(a, F_a, F_b):
    # the ancestral analogue of the drift collapse: with D_θ = head, the deterministic mean
    # c_x(S)x + c_d(S)D_θ equals the Gaussian sampler's EXACT reverse mean μ(S)x band-for-band. This pins the
    # c_x, c_d coefficients (and the Σ_a cancellation) to machine precision at real, non-degenerate steps —
    # the endpoints in test_ancestral_coeff_limits are passed even by a dropped-½ exponent bug.
    op, lo, hi, gauss, learned = _head_sampler(a)
    x = torch.randn(op.d, 4, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    d = learned.denoiser(x, F_a, **learned.c_in_kw)                  # = c_skip(S) x (body zeroed)
    c_x, c_d, _ = learned._ancestral_coeffs(F_a, F_b)
    mean = learned._apply(c_x, x) + learned._apply(c_d, d)
    mu = apply_fun(gauss.mean_fn(F_a, F_b), op, x, lo, hi, gauss.K)
    assert rel_err(mean, mu) < 1e-5


@pytest.mark.parametrize("a", [0.0, -1.0])
def test_ancestral_noise_identity(a):
    # pin the noise coefficient q = R Σ_b/Σ_a and the derivation's claimed identity R = Σ_a − T²Σ_b, plus q≥0.
    op, lo, hi, gauss, learned = _head_sampler(a)
    phi = learned.phi
    lam = torch.linspace(lo, hi, 64, dtype=torch.float64)
    F_a, F_b = 3.0, 1.0
    sig = lambda F: -lam * torch.expm1(-F * phi(lam))               # Σ_F = λ(1−e^{−Fφ})
    T = torch.exp(-0.5 * (F_a - F_b) * phi(lam))
    R = -lam * torch.expm1(-(F_a - F_b) * phi(lam))
    assert float((R - (sig(F_a) - T ** 2 * sig(F_b))).abs().max()) < 1e-10     # R = Σ_a − T²Σ_b
    _, _, ns = learned._ancestral_coeffs(F_a, F_b)
    assert float((ns(lam) ** 2 - R * sig(F_b) / sig(F_a)).abs().max()) < 1e-10  # q = R Σ_b/Σ_a
    assert float(ns(lam).min()) >= 0.0                                          # q ≥ 0


# ------------------------------------------------------------------ structure / integrate / guards
def _trained_like_sampler(*, dtype=torch.float32, dev="cpu"):
    """A LearnedReverseSDE with a (randomly-initialised) real body — exercises the D_θ path, not just head."""
    toy = BlockGMM(d=16, n_blocks=4, sep=0.9, idio_frac=0.2, tau=0.05, dtype=dtype, device=dev)
    op = toy.make_operator(n=2000, generator=torch.Generator(device=dev).manual_seed(0))
    lo, hi = spectrum_bracket(op, generator=torch.Generator(device=dev).manual_seed(1))
    fwd = ColoredForward(op, lo, hi, phi=phi_power(0.0), K=32)
    den = ColoredDenoiser(ColoredEDM(fwd), VNNBody(fwd, width=32, n_layers=2, K_gf=4).to(dtype))
    return op, lo, hi, LearnedReverseSDE(den, K=32, c_in_kw=dict(tol=1e-4, maxiter=500))


@pytest.mark.parametrize("scheme", ["euler_maruyama", "ancestral"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(16,), (16, 5)])
def test_shapes_and_dtype(dtype, shape, scheme):
    # covers BOTH schemes through a real (non-zeroed) body — so the ancestral D_θ + c_in wiring is exercised,
    # not just the analytic-head marginal test.
    op, lo, hi, learned = _trained_like_sampler(dtype=dtype)
    x = torch.randn(shape, dtype=dtype)
    out = learned.step(x, 1.0, 0.7, scheme=scheme, generator=torch.Generator().manual_seed(0))
    assert out.shape == torch.Size(shape) and out.dtype == dtype and torch.isfinite(out).all()


def test_integrate_and_denoise_final():
    op, lo, hi, learned = _trained_like_sampler()
    grid = reverse_f_grid(6.0, 0.05, 16, spacing="equidistribute", phi=phi_power(0.0), tau=0.05, lo=lo, hi=hi)
    xT = terminal_draw(op, batch=8, generator=torch.Generator().manual_seed(2))
    out = learned.integrate(xT, grid, generator=torch.Generator().manual_seed(3))
    assert out.shape == (op.d, 8) and torch.isfinite(out).all()
    # denoise_final=True returns the clean D_θ(x_{F_min}) ⇒ differs from the raw last iterate.
    raw = learned.integrate(xT, grid, generator=torch.Generator().manual_seed(3), denoise_final=False)
    assert rel_err(out, raw) > 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_runs_on_gpu():
    op, lo, hi, learned = _trained_like_sampler(dtype=torch.float32, dev="cuda")
    grid = reverse_f_grid(6.0, 0.05, 12, spacing="equidistribute", phi=phi_power(0.0), tau=0.05, lo=lo, hi=hi)
    xT = terminal_draw(op, batch=8, generator=torch.Generator(device="cuda").manual_seed(2))
    out = learned.integrate(xT, grid, generator=torch.Generator(device="cuda").manual_seed(3))
    assert out.device.type == "cuda" and torch.isfinite(out).all()


def test_step_guards():
    op, lo, hi, learned = _trained_like_sampler()
    x = torch.randn(op.d, dtype=torch.float32)
    with pytest.raises(ValueError):
        learned.step(x, 0.5, 1.0)                             # F_b > F_a
    assert torch.equal(learned.step(x, 1.0, 1.0), x)          # F_b == F_a is a no-op


# ------------------------------------------------------------------ the ancestral (stiff-stable) scheme
def test_step_scheme_validation():
    op, lo, hi, learned = _trained_like_sampler()
    with pytest.raises(ValueError):
        learned.step(torch.zeros(op.d), 1.0, 0.5, scheme="bogus")


@pytest.mark.parametrize("a", [0.0, -1.0])
def test_ancestral_coeff_limits(a):
    # the derived per-band step's endpoints: ΔF→0 ⇒ c_x→1, c_d→0 (no move); F_b→0 ⇒ c_d→1, c_x→0 (x→D_θ).
    op, lo, hi, gauss, learned = _head_sampler(a)
    lam = torch.linspace(lo, hi, 64, dtype=torch.float64)
    cx0, cd0, ns0 = learned._ancestral_coeffs(3.0, 3.0 - 1e-7)
    assert float((cx0(lam) - 1).abs().max()) < 1e-4 and float(cd0(lam).abs().max()) < 1e-4
    assert float(ns0(lam).max()) < 1e-2                              # noise std → 0 (as √ΔF)
    cxL, cdL, nsL = learned._ancestral_coeffs(3.0, 1e-8)
    assert float((cdL(lam) - 1).abs().max()) < 1e-4 and float(cxL(lam).abs().max()) < 1e-4


_MARGINAL_CACHE: dict = {}


def _gauss_head_case(a):
    """A zeroed-body (⇒ ``D_θ = c_skip x`` = Gaussian head) sampler for the marginal-correctness test, built
    light (small ``d``, float32, low apply degree ``K``, cheap low-iter ``c_in`` — discarded when the body is
    0) and GPU-preferred, so a full 128-step generation is quick. Cached per colouring."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a not in _MARGINAL_CACHE:
        toy = BlockGMM(d=20, n_blocks=5, sep=0.9, idio_frac=0.2, tau=0.05, dtype=torch.float32, device=dev)
        op = toy.make_operator(n=1500, generator=torch.Generator(device=dev).manual_seed(0))
        lo, hi = spectrum_bracket(op, generator=torch.Generator(device=dev).manual_seed(1))
        fwd = ColoredForward(op, lo, hi, phi=phi_power(a), K=48)
        body = VNNBody(fwd, width=8, n_layers=1, K_gf=2)
        with torch.no_grad():
            for p in body.decoder.parameters():
                p.zero_()
        den = ColoredDenoiser(ColoredEDM(fwd), body)
        head = LearnedReverseSDE(den, K=48, c_in_kw=dict(tol=0.1, maxiter=20, require_converged=False))
        _MARGINAL_CACHE[a] = (op, lo, hi, head, dev)
    return _MARGINAL_CACHE[a]


@pytest.mark.slow
@pytest.mark.parametrize("a,F_max,F_min", [(0.0, 6.0, 0.05), (-1.0, 40.0, 0.02)])
def test_ancestral_generates_gaussian_marginal(a, F_max, F_min):
    # with D_θ = head the ancestral sampler must reach N(0,Ĉ_n) for BOTH clocks — and, crucially, stay stable
    # under the stiff a=-1 clock (124x per-band-rate spread) where Euler–Maruyama diverges.
    op, lo, hi, head, dev = _gauss_head_case(a)
    Chat = op._dense() - op.tau * torch.eye(op.d, dtype=op.dtype, device=dev)
    grid = reverse_f_grid(F_max, F_min, 128, spacing="equidistribute", weight="abs",
                          phi=phi_power(a), tau=op.tau, lo=lo, hi=hi)
    xT = terminal_draw(op, batch=1500, generator=torch.Generator(device=dev).manual_seed(1))

    def cov(x):
        return rel_err((x @ x.T) / x.shape[1], Chat)

    def gen(scheme):
        return head.integrate(xT, grid, scheme=scheme,
                              generator=torch.Generator(device=dev).manual_seed(2), denoise_final=False)

    x_anc = gen("ancestral")
    assert cov(x_anc) < 0.15 and bool(torch.isfinite(x_anc).all())  # ancestral ~reaches N(0,Ĉ_n)
    if a == -1.0:                                                    # the raison d'être of the scheme
        assert cov(gen("euler_maruyama")) > 4 * cov(x_anc)         # EM blows up where ancestral stays sound
