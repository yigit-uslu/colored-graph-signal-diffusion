"""Phase I2 — the colouring family as a first-class tested axis + the efficiency claim.

Two things are gated here:

1. **The family is a path knob, not a prior knob.** Every member ``a ∈ {−1, 0, ½, 1}``
   shares *both* endpoints — ``V_0 = Ĉ_n`` (clean data) and ``V_∞ = S`` (stationary law
   ``N(0,S)``) — independent of the colouring. What the colouring changes is the
   *generation-order signature*: the order in which spectral bands cross unit SNR
   (``F*=ln2/φ``) flips with ``sign(a)`` — fine→coarse for ``a>0``, coarse→fine for
   ``a<0``, none at ``a=0``. (``a=−1`` marginals-vs-dense also completes the I1 coverage,
   which only spanned ``a ≥ 0``.)

2. **The construction is efficient and eigendecomposition-free.** An adaptive Chebyshev
   degree ``K ≪ d``, spending *exactly* ``K`` matvecs, reproduces the dense-eig marginal
   to ``1e-3`` — the operational meaning of "constructed without eigendecomposition".

float32 path, float64 baselines, ``rel_err < 1e-3`` gate.
"""

from __future__ import annotations

import math

import pytest
import torch

from ccd.forward import ColoredForward, f_star, generation_order, phi_power
from ccd.operators import chebyshev_degree, make_synthetic_operator, spectrum_bracket
from _oracles import dense_apply, rel_err

TOL = 1e-3
FAMILY = [-1.0, 0.0, 0.5, 1.0]


def _setup(d=64, n=400, tau=0.05, seed=0):
    op = make_synthetic_operator(n, d, tau=tau, generator=torch.Generator().manual_seed(seed))
    lo, hi = spectrum_bracket(op, generator=torch.Generator().manual_seed(seed + 1))
    v = torch.randn(d, generator=torch.Generator().manual_seed(seed + 2))
    return op, lo, hi, v


# --------------------------------------------------------------- prior preservation
@pytest.mark.parametrize("a", FAMILY)
def test_endpoints_are_colouring_independent(a):
    # Both ends of the path are shared by the whole family: only the route differs.
    op, lo, hi, v = _setup()
    fwd = ColoredForward(op, lo, hi, phi=phi_power(a))
    # F = 0:  V_0 = S − τ e^{0} = S − τI = Ĉ_n, identical for every a.
    c_hat = op.matvec(v) - op.tau * v
    assert rel_err(fwd.v_marginal(v, 0.0), c_hat) < TOL
    # F → ∞:  V_∞ = S, identical for every a (stationary law N(0,S)).
    assert rel_err(fwd.v_marginal(v, 400.0), op.matvec(v)) < TOL
    # and the mean collapses, m_∞ → 0, for every a.
    assert float(fwd.mean(v, 400.0).norm() / v.norm()) < 5e-2


def test_marginals_match_dense_negative_a():
    # a = −1 (coarse→fine) completes the family: I1's test_marginals only spanned a ≥ 0.
    op, lo, hi, v = _setup()
    fwd = ColoredForward(op, lo, hi, phi=phi_power(-1.0))
    F = 2.0
    f_m = lambda lam: torch.exp(-0.5 * F * lam ** (-1.0))
    f_s = lambda lam: torch.sqrt(torch.clamp(lam * (1.0 - torch.exp(-F * lam ** (-1.0))), min=0.0))
    f_v = lambda lam: lam - op.tau * torch.exp(-F * lam ** (-1.0))
    assert rel_err(fwd.mean(v, F), dense_apply(f_m, op, v)) < TOL
    assert rel_err(fwd.sigma_half(v, F), dense_apply(f_s, op, v)) < TOL
    assert rel_err(fwd.v_marginal(v, F), dense_apply(f_v, op, v)) < TOL


# --------------------------------------------------------- generation-order signature
def test_generation_order_signature():
    op, _, _, _ = _setup(d=64)
    evals = torch.linalg.eigvalsh(op._dense().double())     # ascending, distinct
    asc = torch.argsort(evals)
    desc = torch.argsort(evals, descending=True)

    # a > 0: fine→coarse — small λ (large F*) generated first ⇒ ascending-λ order.
    for a in (0.5, 1.0):
        assert torch.equal(generation_order(phi_power(a), evals), asc)
    # a < 0: coarse→fine — large λ generated first ⇒ descending-λ order.
    assert torch.equal(generation_order(phi_power(-1.0), evals), desc)
    # a = 0: no order — every band crosses unit SNR at the same F* = ln2.
    fs0 = f_star(phi_power(0.0), evals)
    assert float(fs0.max() - fs0.min()) < 1e-9
    # (The a=1 order being the exact reverse of a=−1 follows from the asc/desc gates above;
    # asserting flip(desc)==asc separately would be a tautology, so it is omitted.)


def test_f_star_matches_closed_form():
    # F*(λ) = ln2 · λ^{−a} for the power-law family.
    import math

    evals = torch.linspace(0.05, 1.05, 11, dtype=torch.float64)
    for a in FAMILY:
        fs = f_star(phi_power(a), evals)
        assert rel_err(fs, math.log(2.0) * evals ** (-a)) < 1e-6


# ------------------------------------------------------------------ efficiency claim
def test_efficient_construction_is_eig_free_and_cheap():
    # Tall regime (d ≫ n): a dense eig would be O(d^3). The matrix-free path instead
    # spends an adaptive K ≪ d matvecs and matches the dense-eig marginal.
    op, lo, hi, v = _setup(d=512, n=128, seed=3)
    fwd = ColoredForward(op, lo, hi, phi=phi_power(1.0))
    F = 4.0
    f_m = lambda lam: torch.exp(-0.5 * F * lam)

    K_pred = max(chebyshev_degree(f_m, lo, hi), 1)             # scalar-only, no matvecs
    op.reset_count()
    out = fwd.mean(v, F)
    K_used = op.matvec_count

    assert K_used == K_pred                                    # *exactly* K matvecs
    assert K_used < op.d // 8                                  # K ≪ d (here ~tens vs 512)
    assert rel_err(out, dense_apply(f_m, op, v)) < TOL         # ...and it is correct


# ------------------------------------------------------------------ two-point-pinned shifted power law
# `ShiftedPowerRate` pins BOTH ends of φ (coarsest clock ln2/FLOOR, finest c·F_min), so its clock
# range matches TiledRate's by construction instead of being whatever (λ_max/λ_min)^|a| happens to
# be. These tests are hermetic: they build their own wide spectrum (the `_setup` fixture's ratio is
# only ~17.5, below R^(1/|a|), so the pin is genuinely unreachable there and would raise).

PIN_A = [-1.0, -1.5, -2.0]
PIN_FLOOR, PIN_FMIN, PIN_C = 0.077, 0.02, 3.0
PIN_LO, PIN_HI = 0.05, 8.86                                    # matches the real bracket's 177× ratio


def _pin_spectrum(n=256):
    """Strictly distinct, log-spaced spectrum spanning the same ratio as the real d=256 bracket."""
    return torch.logspace(math.log10(PIN_LO), math.log10(PIN_HI), n, dtype=torch.float64)


def _pin_R():
    return math.log(2.0) / (PIN_C * PIN_FMIN * PIN_FLOOR)      # = 150.0319…


def _pin(a, **kw):
    from ccd.forward import ShiftedPowerRate

    kw = dict(FLOOR=PIN_FLOOR, F_min=PIN_FMIN, c=PIN_C, **kw)
    return ShiftedPowerRate(a, PIN_LO, PIN_HI, **kw)


@pytest.mark.parametrize("a", PIN_A)
def test_shifted_power_two_point_pin(a):
    """Both pins hold, so every arm shares one clock range regardless of `a`."""
    r, R = _pin(a), _pin_R()
    assert float(r(torch.tensor(PIN_HI, dtype=torch.float64))) == pytest.approx(PIN_FLOOR, rel=1e-12)
    assert float(r(torch.tensor(PIN_LO, dtype=torch.float64))) == pytest.approx(R * PIN_FLOOR, rel=1e-10)
    assert r.range() == pytest.approx(R, rel=1e-10)
    coarse, fine = r.clocks()
    assert coarse == pytest.approx(math.log(2.0) / PIN_FLOOR, rel=1e-12)
    assert fine == pytest.approx(PIN_C * PIN_FMIN, rel=1e-10)   # finest clock clears the sampler floor


@pytest.mark.parametrize("a", PIN_A)
def test_shifted_power_normphi_is_a_noop(a):
    """φ(λ_max)=FLOOR by construction ⇒ the script-side NormPhi rescale is ×1."""
    ev = _pin_spectrum()
    assert PIN_FLOOR / float(_pin(a)(ev).min()) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("a", PIN_A)
def test_shifted_power_strictly_decreasing(a):
    """Strict monotonicity ⇒ f_star / generation_order stay well posed (no shelf ties)."""
    ev = _pin_spectrum(1000)                                   # strictly distinct by construction
    phi = _pin(a)(ev)
    assert bool((phi[1:] < phi[:-1]).all())                    # φ strictly decreasing in λ
    order = generation_order(_pin(a), ev)
    assert torch.equal(order, torch.argsort(ev, descending=True))


@pytest.mark.parametrize("a", PIN_A)
def test_shifted_power_chebyshev_K40_beats_plain_power(a):
    """The shift is analytic on [lo,hi] (pole at −s<0), so K=40 resolves it far better than λ^a.

    float64 and small/mid F: this is where the pinned law's geometric convergence is the point.
    (In the production float32 path both families floor at ~3e-6.)
    """
    from ccd.operators.chebyshev import cheb_coeffs

    ev = _pin_spectrum()
    r = _pin(a)
    plain_scale = PIN_FLOOR * PIN_HI ** (-a)                   # same one-end pin as NormPhi gives λ^a
    for F in (0.02, 0.5):
        m_shift = lambda lam: torch.exp(-0.5 * F * r(lam))
        m_plain = lambda lam: torch.exp(-0.5 * F * plain_scale * lam ** a)
        errs = []
        for fn in (m_shift, m_plain):
            c = cheb_coeffs(fn, PIN_LO, PIN_HI, 40, dtype=torch.float64)
            x = (2.0 * ev - (PIN_HI + PIN_LO)) / (PIN_HI - PIN_LO)
            approx = sum(ck * torch.cos(k * torch.acos(x.clamp(-1.0, 1.0))) for k, ck in enumerate(c))
            errs.append(float((approx - fn(ev)).abs().max()))
        assert errs[0] < 1e-3                                  # absolute gate
        assert errs[0] <= errs[1]                              # ...and never worse than the plain arm


@pytest.mark.parametrize("a", PIN_A)
def test_shifted_power_v_monotone_guard(a):
    """The τ|a|/(e(λ_min+s))<1 guard is sufficient for v_F monotone at EVERY F (_v_bracket's precondition)."""
    tau = 0.05
    r = _pin(a, tau=tau)                                       # constructs ⇒ guard passed
    assert tau * abs(a) / (math.e * (PIN_LO + r.s)) < 1.0
    ev = torch.logspace(math.log10(PIN_LO), math.log10(PIN_HI), 20_001, dtype=torch.float64)
    for F in (PIN_FMIN, 0.5, 9.0):
        v = ev - tau * torch.exp(-F * r(ev))
        assert bool((v[1:] >= v[:-1]).all())                   # non-decreasing ⇒ grid min is the true inf


def test_shifted_power_rejects_a_ge_zero():
    """a=0 has R=1 by definition and cannot be two-point pinned; a>0 breaks the pin's orientation."""
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError, match="needs a < 0"):
            _pin(bad)


def test_shifted_power_rejects_unreachable_R():
    """Pinning can only SHRINK the range: R above the unpinned (λ_max/λ_min)^|a| must raise."""
    unpinned = (PIN_HI / PIN_LO) ** 1.0                        # a=-1 ⇒ 177.2 < 321
    assert unpinned < 321.0
    with pytest.raises(ValueError, match="unreachable"):
        _pin(-1.0, R=321.0)


@pytest.mark.parametrize("a", PIN_A)
def test_shifted_power_effective_a_is_nominal_not_local(a):
    """effective_a is the arm label; the realised slope is local_a/weighted_a and is strictly milder."""
    r = _pin(a)
    assert r.effective_a() == a
    ev = _pin_spectrum()
    local = r.local_a(ev)
    assert torch.allclose(local, a * ev / (ev + r.s), rtol=1e-12)
    assert bool((local > a).all()) and bool((local < 0.0).all())   # |realised| < |nominal| everywhere
    w = torch.ones_like(ev)
    assert r.weighted_a(ev.numpy(), w.numpy()) == pytest.approx(float(local.mean()), rel=1e-10)
