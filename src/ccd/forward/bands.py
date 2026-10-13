r"""Band functions of the colored forward/reverse process — the single source of truth.

Every scalar ``λ → value`` the operator path feeds to :func:`~ccd.operators.chebyshev.apply_fun`
lives here, and **the audit must audit these same objects**. Before this module they were inline
closures inside :mod:`ccd.forward.marginals`, :mod:`ccd.denoiser.edm` and
:mod:`ccd.reverse.learned`; auditing a re-typed copy of an inline closure audits the copy, not the
code that runs. The factories are module-level and pure — they close over ``(phi, tau, F…)`` and
return a function of ``λ`` alone — so `cheb_audit` and the operator path can hold the *same*
callable.

Nothing here applies an operator or touches a tensor's shape: these are scalar functions evaluated
either on an eigenvalue grid (audit, float64) or at the Chebyshev nodes (`apply_fun`).

**The split (implementation plan §2.1b, Remedy A).** ``Σ_F^{1/2} = √(λ·h)`` and
``q^{1/2} = √(λ·g)`` each carry a bare ``√λ``, whose branch point at ``λ=0`` sits a distance ``τ``
below a bracket of width ≈197 on the fMRI operator — no admissible Chebyshev degree resolves it
(measured 1.5e-4 at K=256, decaying algebraically). Since functions of ``S`` commute, the factor
splits exactly:

    N(0, Σ_F)  =  h^{1/2}(S) · N(0, S),        h = 1 − e^{−Fφ}
    N(0, q)    =  g^{1/2}(S) · N(0, S),        q = λ·g

and ``N(0,S)`` is drawn exactly by the O(nd) data-factor route with no spectral function at all.
``h`` and ``g`` are analytic on the bracket and spectral to machine precision by K≈64. The unsplit
forms are kept here **only** so the audit can report them as the rejected route and so the
teeth-control (§3.3) can inject one deliberately — they must never reach a production draw.
"""

from __future__ import annotations

from typing import Callable

import torch

from .rate import ColouringRate

__all__ = [
    "ScalarBandFn",
    "c_d_fn", "c_out_fn", "c_out_sq_fn", "c_skip_fn", "c_x_fn",
    "g_fn", "g_sqrt_fn", "h_fn", "h_sqrt_fn",
    "m_fn", "q_fn", "q_sqrt_fn", "sigma_fn", "sigma_sqrt_fn", "v_fn",
]

ScalarBandFn = Callable[[torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------- forward marginals


def m_fn(phi: ColouringRate, F: float) -> ScalarBandFn:
    """``m_F(λ) = e^{−½Fφ(λ)}`` — the forward mean factor. Spectral (1.2e-14 at K=64)."""
    return lambda lam: torch.exp(-0.5 * F * phi(lam))


def h_fn(phi: ColouringRate, F: float) -> ScalarBandFn:
    """``h_F(λ) = 1 − e^{−Fφ(λ)}`` — the SPLIT forward-noise factor (Remedy A).

    Analytic on the bracket: no ``√λ``, no pole. This is the half of ``Σ_F`` that stays matrix-free.
    """
    return lambda lam: -torch.expm1(-F * phi(lam))


def h_sqrt_fn(phi: ColouringRate, F: float) -> ScalarBandFn:
    """``√h_F`` — what multiplies an exact ``N(0,S)`` draw to give ``N(0,Σ_F)``."""
    h = h_fn(phi, F)
    return lambda lam: torch.sqrt(torch.clamp(h(lam), min=0.0))


def sigma_fn(phi: ColouringRate, F: float) -> ScalarBandFn:
    """``Σ_F(λ) = λ(1 − e^{−Fφ}) = λ·h_F``."""
    return lambda lam: -lam * torch.expm1(-F * phi(lam))


def sigma_sqrt_fn(phi: ColouringRate, F: float) -> ScalarBandFn:
    """``Σ_F^{1/2} = √(λ·h_F)`` — **UNSPLIT**. Carries the ``√λ`` branch point.

    Kept for the audit's rejected-route row and for the §3.3 teeth-control's deliberate defect.
    Never route a production draw through this; use ``h_sqrt_fn`` on a data-factor ``N(0,S)``.
    """
    sig = sigma_fn(phi, F)
    return lambda lam: torch.sqrt(torch.clamp(sig(lam), min=0.0))


def v_fn(phi: ColouringRate, tau: float, F: float) -> ScalarBandFn:
    """``v_F(λ) = λ − τe^{−Fφ}`` — the marginal variance. Smooth and bounded; the steepness is in
    its *inverse*, which the Krylov solver owns."""
    return lambda lam: lam - tau * torch.exp(-F * phi(lam))


# --------------------------------------------------------------------------- EDM preconditioners
#
# Both are DENSE-dispositioned on a ridge-dominated spectrum (Remedy C): `c_out` carries √(λ−τ), a
# branch point exactly AT the bracket's lower endpoint, and `c_skip` runs 0→1 across a boundary
# layer of relative width ~8e-8. Measured at K=256: 2.1e-2 and 3.9e-1. They are defined here anyway
# so the dense backend and the audit's rejected-route row share one definition with the legacy
# matrix-free path that other targets still use.


def c_skip_fn(phi: ColouringRate, tau: float, F: float) -> ScalarBandFn:
    """``c_skip = (λ−τ)m_F/v_F`` — the Wiener-head preconditioner."""
    def f(lam):
        m = torch.exp(-0.5 * F * phi(lam))
        return (lam - tau) * m / (lam - tau * m * m)
    return f


def c_out_sq_fn(phi: ColouringRate, tau: float, F: float) -> ScalarBandFn:
    """``c_out² = Σ_0Σ_F/V_F`` with ``Σ_0 = λ−τ``."""
    def f(lam):
        e = torch.exp(-F * phi(lam))
        return (lam - tau) * (lam * (1.0 - e)) / (lam - tau * e)
    return f


def c_out_fn(phi: ColouringRate, tau: float, F: float) -> ScalarBandFn:
    cout2 = c_out_sq_fn(phi, tau, F)
    return lambda lam: torch.sqrt(torch.clamp(cout2(lam), min=0.0))


# --------------------------------------------------------------------------- ancestral step
#
# Pair-indexed: these take (F_a, F_b), not a single F. A one-argument `draw(F, …)` cannot express
# the ancestral noise at all — `q = λ·g(F_a, F_b)`.


def c_x_fn(phi: ColouringRate, F_a: float, F_b: float) -> ScalarBandFn:
    """``c_x = T Σ_b/Σ_a``, ``T = e^{−½ΔFφ}``. A ratio in which ``λ`` genuinely cancels."""
    sig_a, sig_b, dF = sigma_fn(phi, F_a), sigma_fn(phi, F_b), F_a - F_b
    return lambda lam: torch.exp(-0.5 * dF * phi(lam)) * sig_b(lam) / sig_a(lam)


def c_d_fn(phi: ColouringRate, F_a: float, F_b: float) -> ScalarBandFn:
    """``c_d = R m_b/Σ_a``, ``R = λ(1−e^{−ΔFφ})``. ``λ`` cancels here too."""
    sig_a, dF = sigma_fn(phi, F_a), F_a - F_b

    def f(lam):
        R = -lam * torch.expm1(-dF * phi(lam))
        return R * torch.exp(-0.5 * F_b * phi(lam)) / sig_a(lam)
    return f


def g_fn(phi: ColouringRate, F_a: float, F_b: float) -> ScalarBandFn:
    """``g = (1−e^{−ΔFφ})(1−e^{−F_bφ})/(1−e^{−F_aφ})`` — the SPLIT ancestral noise factor.

    ``q = R Σ_b/Σ_a`` where every ``Σ`` is ``λ·h``, so exactly one power of ``λ`` survives:
    ``q = λ·g``. The bare ``λ`` factors out cleanly and ``g`` is analytic — this is the ancestral
    counterpart of ``h``, and it is what makes the reverse step's noise draw matrix-free.
    """
    h_d, h_b, h_a = h_fn(phi, F_a - F_b), h_fn(phi, F_b), h_fn(phi, F_a)
    return lambda lam: h_d(lam) * h_b(lam) / h_a(lam)


def g_sqrt_fn(phi: ColouringRate, F_a: float, F_b: float) -> ScalarBandFn:
    """``√g`` — what multiplies an exact ``N(0,S)`` draw to give ``N(0,q)``."""
    g = g_fn(phi, F_a, F_b)
    return lambda lam: torch.sqrt(torch.clamp(g(lam), min=0.0))


def q_fn(phi: ColouringRate, F_a: float, F_b: float) -> ScalarBandFn:
    """``q = R Σ_b/Σ_a = λ·g``."""
    g = g_fn(phi, F_a, F_b)
    return lambda lam: lam * g(lam)


def q_sqrt_fn(phi: ColouringRate, F_a: float, F_b: float) -> ScalarBandFn:
    """``q^{1/2} = √(λ·g)`` — **UNSPLIT**, carries ``√λ``. Same warning as `sigma_sqrt_fn`."""
    q = q_fn(phi, F_a, F_b)
    return lambda lam: torch.sqrt(torch.clamp(q(lam), min=0.0))
