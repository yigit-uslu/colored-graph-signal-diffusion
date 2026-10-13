"""Gaussian-scoped stochastic reverse-time SDE sampler (Phase I3b-1).

The stochastic counterpart of the deterministic PF-ODE (:mod:`ccd.reverse.pf_ode`). Anderson's
reverse-time SDE of the colored forward process (``β≡1``, so ``F = t``), run ``F: F_max → 0``, is

    dx = [−½ φ(S) x + Sφ(S) V_F^{-1} x] dF + (Sφ(S))^{1/2} dW̄

— the score term now carries **full weight** (the PF-ODE used ½) and the diffusion term the
PF-ODE dropped is restored. Per spectral band ``λ`` this is a linear (OU) SDE, so it has an
**exact matrix-free transition** — the continuous-time DDPM ancestral step. With the forward
being a time-homogeneous OU, ``(x_{F_a}, x_{F_b})`` is jointly Gaussian and the reverse posterior
``p(x_{F_b} | x_{F_a}) = N(μ(S) x_{F_a}, q(S))`` has, per band (``ΔF = F_a − F_b > 0``),

    μ(λ) = e^{−½ ΔF φ(λ)} · v_{F_b}(λ) / v_{F_a}(λ),
    q(λ) = v_{F_b}(λ) · (1 − e^{−ΔF φ(λ)} · v_{F_b}(λ) / v_{F_a}(λ)),   v_F(λ) = λ − τ e^{−Fφ(λ)},

with ``q ≥ 0`` and exact marginal preservation ``μ² v_{F_a} + q = v_{F_b}``. Both ``μ`` and
``√q`` are smooth functions of ``S`` on ``[τ, λ_hi]`` (each ``apply_fun``-able), so — exactly as
in I3a — the sampler needs **no eigendecomposition and no explicit inverse**: the ``1/v_F`` inside
the coefficients is bounded for any ``F ≥ F_min > 0``, and the steep inverse ``V_t^{-1}``
(shifted-CG / Zolotarev) stays deferred to ``F_min → 0`` with a rank-deficient ``Ĉ_n`` or a
learned score.

The ``"euler_maruyama"`` scheme discretises the reverse SDE directly with drift coefficient
``d_F(λ) = φ(λ)(λ + τ e^{−Fφ}) / (2 v_F)`` and diffusion ``(λφ(λ))^{1/2}``; it preserves the
marginal only to ``O(ΔF)`` and is used to exhibit convergence to the exact dynamics.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ccd.operators.chebyshev import apply_fun
from ccd.operators.linop import CovarianceOperator
from ccd.forward.rate import ColouringRate, phi_power


class GaussianReverseSDE:
    """Stochastic reverse-time SDE sampler for the colored Gaussian process.

    Args mirror :class:`ccd.reverse.pf_ode.GaussianReversePFODE`. Two schemes (see :meth:`step`):
    ``"exact"`` applies the closed-form ancestral transition ``x_{F_b} = μ(S) x_{F_a} + √q(S) ξ``
    — marginal-preserving at *any* grid, hence the production stochastic sampler;
    ``"euler_maruyama"`` is the direct SDE discretisation, ``O(ΔF)`` marginal error.

    Each stochastic step draws fresh standard-normal noise from the supplied ``generator``.
    """

    def __init__(
        self,
        op: CovarianceOperator,
        lo: float,
        hi: float,
        *,
        phi: ColouringRate | None = None,
        K: int | None = None,
    ):
        self.op = op
        self.lo = float(lo)
        self.hi = float(hi)
        self.phi = phi if phi is not None else phi_power(1.0)
        self.K = K

    # ----- per-band scalar functions f(λ) fed to ApplyFun (no matrix exponential, no inverse) -----

    def _v_fn(self, F: float):
        """Marginal band variance ``v_F(λ) = λ − τ e^{−Fφ(λ)}`` (= the ``V_F`` band)."""
        tau = self.op.tau
        return lambda lam: lam - tau * torch.exp(-F * self.phi(lam))

    def mean_fn(self, F_a: float, F_b: float):
        """Exact ancestral mean coefficient ``μ(λ) = e^{−½ΔF φ} v_{F_b}/v_{F_a}`` (``ΔF=F_a−F_b``)."""
        dF = float(F_a) - float(F_b)
        v_a, v_b, phi = self._v_fn(F_a), self._v_fn(F_b), self.phi
        return lambda lam: torch.exp(-0.5 * dF * phi(lam)) * v_b(lam) / v_a(lam)

    def noise_std_fn(self, F_a: float, F_b: float):
        """Exact ancestral noise std ``√q(λ) = √(v_{F_b}(1 − e^{−ΔF φ} v_{F_b}/v_{F_a}))``."""
        dF = float(F_a) - float(F_b)
        v_a, v_b, phi = self._v_fn(F_a), self._v_fn(F_b), self.phi

        def f(lam: Tensor) -> Tensor:
            ratio = torch.exp(-dF * phi(lam)) * v_b(lam) / v_a(lam)
            return torch.sqrt(torch.clamp(v_b(lam) * (1.0 - ratio), min=0.0))

        return f

    def em_drift_fn(self, F_a: float, F_b: float):
        """Euler–Maruyama linear factor ``s(λ) = 1 − ΔF·d_{F_a}(λ)`` (drift over the step)."""
        h = float(F_a) - float(F_b)
        tau, phi = self.op.tau, self.phi

        def s(lam: Tensor) -> Tensor:
            e = torch.exp(-F_a * phi(lam))
            v = lam - tau * e
            d = phi(lam) * (lam + tau * e) / (2.0 * v)     # ½ φ (λ + τ e^{−F_a φ}) / v_{F_a}
            return 1.0 - h * d

        return s

    def diffusion_std_fn(self):
        """Diffusion coefficient ``g(λ) = (λ φ(λ))^{1/2}`` (noise scales by ``g·√ΔF`` per step)."""
        phi = self.phi
        return lambda lam: torch.sqrt(torch.clamp(lam * phi(lam), min=0.0))

    # ----------------------------------------------------------------- public API

    def step(
        self,
        x: Tensor,
        F_a: float,
        F_b: float,
        *,
        scheme: str = "exact",
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """One stochastic reverse step ``F_a → F_b`` (``F_b ≤ F_a``); draws fresh noise."""
        F_a, F_b = float(F_a), float(F_b)
        if F_b > F_a:
            raise ValueError(f"reverse step needs F_b <= F_a; got F_a={F_a}, F_b={F_b}")
        if F_b == F_a:
            return x
        xi = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
        if scheme == "exact":
            mean = apply_fun(self.mean_fn(F_a, F_b), self.op, x, self.lo, self.hi, self.K)
            noise = apply_fun(self.noise_std_fn(F_a, F_b), self.op, xi, self.lo, self.hi, self.K)
            return mean + noise
        if scheme == "euler_maruyama":
            drift = apply_fun(self.em_drift_fn(F_a, F_b), self.op, x, self.lo, self.hi, self.K)
            noise = apply_fun(self.diffusion_std_fn(), self.op, xi, self.lo, self.hi, self.K)
            return drift + (F_a - F_b) ** 0.5 * noise
        raise ValueError(f"scheme must be 'exact' or 'euler_maruyama'; got {scheme!r}")

    def integrate(
        self,
        x_T: Tensor,
        F_grid,
        *,
        scheme: str = "exact",
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Integrate the reverse SDE along a decreasing ``F_grid``; ``x_T`` is the draw at ``F_grid[0]``."""
        Fs = [float(f) for f in F_grid]
        x = x_T
        for F_a, F_b in zip(Fs[:-1], Fs[1:]):
            x = self.step(x, F_a, F_b, scheme=scheme, generator=generator)
        return x
