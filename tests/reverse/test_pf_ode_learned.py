"""The learned-score deterministic PF-ODE sampler (Phase V3b).

Mirrors ``test_learned.py`` for the deterministic sibling. The load-bearing test is the **analytic collapse**:
with the learned denoiser forced to the Gaussian head (``D_θ = c_skip(S) x``), the learned PF-ODE **euler**
step reproduces :class:`ccd.reverse.GaussianReversePFODE`'s euler step to machine precision — pinning the
half-score drift ``b = −½φ(S)x − ½a_F(S)(m_F(S)D_θ − x)`` and its reduction to the analytic PF-ODE band
``c_F = ½φτe^{−Fφ}/v_F``. Two further properties distinguish it from the reverse SDE: it is **deterministic**
(no ``generator`` — bit-identical across calls on a deterministic backend; on GPU it still inherits
CUDA-kernel nondeterminism through ``D_θ``), and it is **not stiff** under the coarse-to-fine clock (``c_F``
vanishes as ``F→∞`` where the SDE drift → ½φ), so plain ``euler`` / ``heun`` stay stable at **both** clocks —
no exponential integrator needed. The rest are structural (shapes/dtype/GPU, Heun's 2nd-order accuracy, the
integrate + denoise-final path, and the step guards).
"""

from __future__ import annotations

import pytest
import torch

from ccd.denoiser import BlockGMM, ColoredDenoiser, ColoredEDM, VNNBody
from ccd.forward import ColoredForward, phi_power, terminal_draw
from ccd.operators import spectrum_bracket
from ccd.reverse import GaussianReversePFODE, LearnedReversePFODE, reverse_f_grid
from _oracles import rel_err

# c_in is DISCARDED when the body is zeroed (D_θ = c_skip x), so it only has to converge — a loose tol keeps
# the collapse tests off a minutes-long tol=1e-10 CG solve.
CIK = dict(tol=1e-3, maxiter=200)
_HEAD_CACHE: dict = {}


def _head_pfode(a, *, d=16, K=200, dtype=torch.float64, dev="cpu"):
    """A LearnedReversePFODE forced to the analytic head (decoder zeroed ⇒ F_θ≡0), paired with the
    :class:`GaussianReversePFODE` on the *same* operator. Cached per colouring."""
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
        gauss = GaussianReversePFODE(op, lo, hi, phi=phi_power(a), K=K)
        _HEAD_CACHE[key] = (op, lo, hi, gauss, LearnedReversePFODE(den, K=K, c_in_kw=CIK))
    return _HEAD_CACHE[key]


# ------------------------------------------------------------------ the analytic collapse
@pytest.mark.parametrize("a", [0.0, 1.0, -1.0])
@pytest.mark.parametrize("F_a,F_b", [(1.0, 0.8), (3.0, 2.0), (0.5, 0.3)])
def test_euler_step_collapses_to_gaussian(a, F_a, F_b):
    # learned PF-ODE euler step with D_θ = head equals GaussianReversePFODE's euler step (independent code
    # path) — this pins the half-score drift b = −½φx − ½a_F(m_F D_θ − x) and its Σ_F cancellation.
    op, lo, hi, gauss, learned = _head_pfode(a)
    x = torch.randn(op.d, 4, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    g = gauss.step(x, F_a, F_b, scheme="euler")
    l = learned.step(x, F_a, F_b, scheme="euler")
    assert rel_err(l, g) < 1e-6


@pytest.mark.parametrize("a", [0.0, 1.0, -1.0])
def test_heun_more_accurate_than_euler(a):
    # with D_θ = head the drift is exactly linear, so both explicit schemes converge to GaussianReversePFODE's
    # EXACT transport — and Heun (2nd order) is strictly closer than Euler at the same step count. Also proves
    # both schemes stay bounded (in particular at a=−1, where the PF-ODE is *not* stiff).
    op, lo, hi, gauss, learned = _head_pfode(a)
    grid = reverse_f_grid(6.0, 0.05, 24, spacing="equidistribute", weight="rel",
                          phi=phi_power(a), tau=0.05, lo=lo, hi=hi)
    xT = terminal_draw(op, batch=64, generator=torch.Generator().manual_seed(2))
    x_exact = gauss.integrate(xT, grid, scheme="exact")
    e_eu = rel_err(learned.integrate(xT, grid, scheme="euler", denoise_final=False), x_exact)
    e_he = rel_err(learned.integrate(xT, grid, scheme="heun", denoise_final=False), x_exact)
    assert e_he < e_eu and e_he < 5e-3


# ------------------------------------------------------------------ structure / integrate / guards
def _trained_like_pfode(*, dtype=torch.float32, dev="cpu"):
    """A LearnedReversePFODE with a (randomly-initialised) real body — exercises the D_θ path, not just head."""
    toy = BlockGMM(d=16, n_blocks=4, sep=0.9, idio_frac=0.2, tau=0.05, dtype=dtype, device=dev)
    op = toy.make_operator(n=2000, generator=torch.Generator(device=dev).manual_seed(0))
    lo, hi = spectrum_bracket(op, generator=torch.Generator(device=dev).manual_seed(1))
    fwd = ColoredForward(op, lo, hi, phi=phi_power(0.0), K=32)
    den = ColoredDenoiser(ColoredEDM(fwd), VNNBody(fwd, width=32, n_layers=2, K_gf=4).to(dtype))
    return op, lo, hi, LearnedReversePFODE(den, K=32, c_in_kw=dict(tol=1e-4, maxiter=500))


@pytest.mark.parametrize("scheme", ["euler", "heun"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(16,), (16, 5)])
def test_shapes_and_dtype(dtype, shape, scheme):
    op, lo, hi, pf = _trained_like_pfode(dtype=dtype)
    x = torch.randn(shape, dtype=dtype)
    out = pf.step(x, 1.0, 0.7, scheme=scheme)
    assert out.shape == torch.Size(shape) and out.dtype == dtype and torch.isfinite(out).all()


def test_deterministic():
    # the defining property vs the SDE: no generator ⇒ the same x_T and grid give the same output — bit-for-bit
    # on this CPU fixture (on GPU D_θ inherits CUDA-kernel nondeterminism, so repeats agree only statistically).
    op, lo, hi, pf = _trained_like_pfode()
    grid = reverse_f_grid(6.0, 0.05, 12, spacing="equidistribute", phi=phi_power(0.0), tau=0.05, lo=lo, hi=hi)
    xT = terminal_draw(op, batch=8, generator=torch.Generator().manual_seed(2))
    a = pf.integrate(xT, grid, scheme="heun")
    b = pf.integrate(xT, grid, scheme="heun")
    assert torch.equal(a, b)


def test_integrate_and_denoise_final():
    op, lo, hi, pf = _trained_like_pfode()
    grid = reverse_f_grid(6.0, 0.05, 16, spacing="equidistribute", phi=phi_power(0.0), tau=0.05, lo=lo, hi=hi)
    xT = terminal_draw(op, batch=8, generator=torch.Generator().manual_seed(2))
    out = pf.integrate(xT, grid)
    assert out.shape == (op.d, 8) and torch.isfinite(out).all()
    # denoise_final=True returns the clean D_θ(x_{F_min}) ⇒ differs from the raw last iterate.
    raw = pf.integrate(xT, grid, denoise_final=False)
    assert rel_err(out, raw) > 1e-4


def test_step_scheme_validation():
    op, lo, hi, pf = _trained_like_pfode()
    with pytest.raises(ValueError):
        pf.step(torch.zeros(op.d), 1.0, 0.5, scheme="ancestral")  # an SDE scheme name is invalid for the PF-ODE


def test_step_guards():
    op, lo, hi, pf = _trained_like_pfode()
    x = torch.randn(op.d, dtype=torch.float32)
    with pytest.raises(ValueError):
        pf.step(x, 0.5, 1.0)                                       # F_b > F_a
    assert torch.equal(pf.step(x, 1.0, 1.0), x)                    # F_b == F_a is a no-op


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_runs_on_gpu():
    op, lo, hi, pf = _trained_like_pfode(dtype=torch.float32, dev="cuda")
    grid = reverse_f_grid(6.0, 0.05, 12, spacing="equidistribute", phi=phi_power(0.0), tau=0.05, lo=lo, hi=hi)
    xT = terminal_draw(op, batch=8, generator=torch.Generator(device="cuda").manual_seed(2))
    out = pf.integrate(xT, grid)
    assert out.device.type == "cuda" and torch.isfinite(out).all()


# ------------------------------------------------------------------ the marginal (both clocks; a=−1 stability)
_MARGINAL_CACHE: dict = {}


def _gauss_head_pfode(a):
    """A zeroed-body (⇒ ``D_θ = c_skip x`` = Gaussian head) PF-ODE sampler for the marginal-correctness test,
    built light (small ``d``, float32, low degree ``K``, cheap ``c_in`` — discarded when the body is 0) and
    GPU-preferred. Cached per colouring."""
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
        pf = LearnedReversePFODE(den, K=48, c_in_kw=dict(tol=0.1, maxiter=20, require_converged=False))
        _MARGINAL_CACHE[a] = (op, lo, hi, pf, dev)
    return _MARGINAL_CACHE[a]


@pytest.mark.slow
@pytest.mark.parametrize("a,F_max,F_min", [(0.0, 6.0, 0.05), (-1.0, 40.0, 0.02)])
def test_pfode_generates_gaussian_marginal(a, F_max, F_min):
    # the deterministic PF-ODE with D_θ = head must reach N(0,Ĉ_n) at BOTH clocks — and, unlike the reverse
    # SDE (where explicit Euler–Maruyama diverges at a=−1), stay stable under the stiff a=−1 clock with the
    # PLAIN explicit schemes, because the PF-ODE drift c_F vanishes as F→∞ (no stiffness).
    op, lo, hi, pf, dev = _gauss_head_pfode(a)
    Chat = op._dense() - op.tau * torch.eye(op.d, dtype=op.dtype, device=dev)
    grid = reverse_f_grid(F_max, F_min, 128, spacing="equidistribute", weight="rel",
                          phi=phi_power(a), tau=op.tau, lo=lo, hi=hi)
    xT = terminal_draw(op, batch=1500, generator=torch.Generator(device=dev).manual_seed(1))

    def cov(x):
        return rel_err((x @ x.T) / x.shape[1], Chat)

    x_he = pf.integrate(xT, grid, scheme="heun", denoise_final=False)
    x_eu = pf.integrate(xT, grid, scheme="euler", denoise_final=False)
    assert cov(x_he) < 0.15 and bool(torch.isfinite(x_he).all())
    assert cov(x_eu) < 0.15 and bool(torch.isfinite(x_eu).all())   # explicit Euler stable even at a=−1
