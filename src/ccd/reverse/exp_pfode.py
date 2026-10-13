r"""Exponential (Lawson–Euler) integrator for the learned colored PF-ODE.

🛑 **Why this is a separate module and not a scheme inside `LearnedReversePFODE`.** It was written
there first, and that broke the **Stage A seal**: `src/ccd/reverse/learned.py` and `src/ccd/vp.py` are
both inside `stage_a_manifest.json`'s 49-file closure, so adding a branch to either made every Stage A
consumer refuse with ``NotSealable``. `scripts/fmri_lowstep_eval.py` is *not* in that closure, so a new
module reached only from the evaluator keeps a sealed benchmark sealed. The cost is one subclass; the
alternative was re-sealing Stage A to add an ablation.

**The scheme.** The learned PF-ODE drift is

    b(x, F) = −½ φ(S) x − ½ a_F(S) ( m_F(S) D_θ(x, F) − x ).

With the body zeroed, ``D_θ = c_skip(S) x = m_F Σ_0/v_F · x`` and this reduces *band for band* to the
analytic drift ``c_F(λ) = ½ φ τ e^{−Fφ} / v_F`` — verified to 5e-16 across the fMRI spectrum before a
line of this was written, and re-checked as a test. That ``c_F`` is the linear part an exponential
integrator solves exactly; its integrating factor over ``F_a → F_b`` is the closed form

    E(λ) = exp ∫_{F_a}^{F_b} c_F dF = √( v_{F_b}(λ) / v_{F_a}(λ) ),      v_F(λ) = λ − τ e^{−Fφ(λ)},

which is exactly the transport :class:`ccd.reverse.GaussianReversePFODE` applies in closed form. Taking
one frozen Euler step in the residual ``N = b − c_F(S)x`` and transporting it gives

    x_{F_b} = √(v_b/v_a) · [ x_{F_a} − ΔF · N(x_{F_a}, F_a) ],           ΔF = F_a − F_b.

⭐ **What separates it from ``"euler"``**: with the body zeroed ``N ≡ 0``, so this returns the *exact*
Gaussian transport at **any** step size, where ``"euler"`` only reproduces that class's Euler step. It
is first-order in the learned residual, the same order ``"ancestral"`` is on the SDE side — which is
what makes the sampler ablation clean: it holds ``ancestral``'s integrator type while dropping the
noise, and holds ``euler``'s determinism while changing the integrator.

⚠️ **There is deliberately no VP analogue.** VP's zero-body denoiser ``α x`` carries the score of
``N(0, I)`` — VP's own stationary law — so its analytic head drift is *identically zero*, the
integrating factor is ``exp(0) = 1``, and the exponential step **is** the Euler step. On the VP arm the
exponential-vs-explicit axis is degenerate; the evaluator routes ``exp_pfode`` to VP's ``"euler"`` and
the ablation gates the two clouds as bit-identical rather than hiding the coincidence.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ccd.reverse.learned import LearnedReversePFODE

__all__ = ["ExponentialPFODE"]


class ExponentialPFODE(LearnedReversePFODE):
    """:class:`LearnedReversePFODE` plus ``scheme="exp"``; ``"euler"``/``"heun"`` are inherited."""

    def _v_fn(self, F: float):
        """Marginal band variance ``v_F(λ) = λ − τ e^{−Fφ(λ)}`` — the band the transport is built from."""
        tau, phi = self.op.tau, self.phi
        return lambda lam: lam - tau * torch.exp(-F * phi(lam))

    def _head_drift_fn(self, F: float):
        """The analytic drift the PF-ODE has with the body zeroed: ``c_F(λ) = ½ φ τ e^{−Fφ}/v_F``."""
        tau, phi = self.op.tau, self.phi
        v = self._v_fn(F)
        return lambda lam: 0.5 * phi(lam) * tau * torch.exp(-F * phi(lam)) / v(lam)

    def _transport_fn(self, F_a: float, F_b: float):
        """The integrating factor ``√(v_{F_b}/v_{F_a})`` — the exact linear PF-ODE transport."""
        v_a, v_b = self._v_fn(F_a), self._v_fn(F_b)
        return lambda lam: torch.sqrt(torch.clamp(v_b(lam) / v_a(lam), min=0.0))

    def step(self, x: Tensor, F_a: float, F_b: float, *, scheme: str = "heun") -> Tensor:
        """One deterministic reverse step. ``scheme="exp"`` is handled here; the rest delegate up."""
        if scheme != "exp":
            return super().step(x, F_a, F_b, scheme=scheme)
        F_a, F_b = float(F_a), float(F_b)
        if F_b > F_a:
            raise ValueError(f"reverse step needs F_b <= F_a; got F_a={F_a}, F_b={F_b}")
        if F_b == F_a:
            return x
        with torch.no_grad():
            dF = F_a - F_b
            resid = self.drift(x, F_a) - self._apply(self._head_drift_fn(F_a), x)
            return self._apply(self._transport_fn(F_a, F_b), x - dF * resid)

    def integrate(self, x_T: Tensor, F_grid, *, scheme: str = "heun",
                  denoise_final: bool = True) -> Tensor:
        """Same contract as the parent; re-implemented only so ``"exp"`` reaches :meth:`step`."""
        if scheme != "exp":
            return super().integrate(x_T, F_grid, scheme=scheme, denoise_final=denoise_final)
        Fs = [float(f) for f in F_grid]
        x = x_T
        for F_a, F_b in zip(Fs[:-1], Fs[1:]):
            x = self.step(x, F_a, F_b, scheme="exp")
        if denoise_final:
            with torch.no_grad():
                x = self.denoiser(x, Fs[-1], **self.c_in_kw)
        return x
