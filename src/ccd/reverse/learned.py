"""Reverse-time samplers driven by a *learned* colored score (Phase V3a / V3b).

The Gaussian samplers (:mod:`ccd.reverse.sde`, :mod:`ccd.reverse.pf_ode`) integrate the reverse dynamics with
the analytic score ``−V_F^{-1}x``, so their drift is *linear* in ``x`` and collapses to a single
function-of-``S`` apply. With a learned denoiser ``D_θ`` the score is the **colored Tweedie**
``s_F = Σ_F^{-1}(m_F D_θ − x)``, and the drift becomes nonlinear in ``x`` (through ``D_θ``). Crucially, in the
sampler the score only ever appears as ``Sφ(S) s_F``, and the ``S`` cancels the one in ``Σ_F = S(I − e^{−Fφ})``,
so **no new steep inverse** is needed — the reverse-SDE drift is

    b(x, F) = −½ φ(S) x − a_F(S) · (m_F(S) D_θ(x, F) − x),     a_F(λ) = φ(λ) / (1 − e^{−F φ(λ)}),

with ``a_F, m_F, φ`` all smooth ``apply_fun``s. The only steep piece is ``c_in = V_F^{-1/2}`` *inside* ``D_θ``
(built in V0/I4). One can check ``b`` collapses to the analytic drift rate ``d(λ) = ½φ(λ+τe^{−Fφ})/v_F`` when
``D_θ = c_skip(S) x`` (that is the V3a test).

Two samplers share the plumbing (:class:`_LearnedReverse`):

* :class:`LearnedReverseSDE` — the **stochastic** reverse-time SDE. Two integrators (``scheme=``):
  ``"euler_maruyama"`` (default) — the explicit SDE discretisation, cheap and adequate for a **non-stiff**
  clock (``a=0``) but it **diverges** when the per-band clock ``Fφ(λ)`` spans a large dynamic range
  (``a=−1``); and ``"ancestral"`` — the stiff-stable exponential / data-prediction step
  (:meth:`~LearnedReverseSDE._ancestral_coeffs`), the reverse posterior ``q(x_{F_b}|x_{F_a}, x_0)`` with
  ``x_0 ← D_θ`` that integrates the stiff *linear* part **exactly** (its mean is exact but, being a
  point-estimate step, it slightly under-injects noise). Use it for ``a=−1``.
* :class:`LearnedReversePFODE` — the **deterministic** probability-flow ODE (learned analogue of
  :class:`ccd.reverse.GaussianReversePFODE`): same marginals, no noise, the score at half weight. Its drift
  ``c_F = ½φτe^{−Fφ}/v_F`` *vanishes* as ``F→∞`` (where the SDE drift → ½φ), so it is **not stiff** at either
  clock — plain ``"euler"`` / ``"heun"`` stay stable, no exponential integrator needed.

Sampling runs under ``no_grad``; the denoiser is duck-typed (needs ``.fwd`` and to be callable
``D_θ(x, F, **c_in_kw)``) so this module stays free of the PyG import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ccd.forward import bands
from ccd.operators.chebyshev import apply_fun

if TYPE_CHECKING:                                            # annotation only — no runtime PyG import
    from ccd.denoiser.denoiser import ColoredDenoiser


class _LearnedReverse:
    """Shared plumbing for the learned reverse samplers (SDE + PF-ODE).

    Holds the operator / spectrum / colouring handles taken from the trained denoiser, the matrix-free
    :func:`ccd.operators.apply_fun` wrapper, and the smooth per-band score bands ``m_F, a_F`` (no inverse, no
    matrix exponential). Subclasses add the drift and the per-scheme step/integrate.

    Args:
        denoiser: a trained ``D_θ`` (e.g. :class:`ccd.denoiser.ColoredDenoiser`) — must expose ``.fwd`` (the
            :class:`ColoredForward` carrying ``op``, ``[lo, hi]``, ``φ``) and be callable ``D_θ(x, F, **c_in_kw)``.
        K: Chebyshev degree for the drift applies (``None`` ⇒ adaptive per apply). This is the degree of the
            *smooth* drift operators only; the steep ``c_in`` degree lives inside ``D_θ``.
        c_in_kw: solver kwargs forwarded to ``D_θ`` each step (e.g. ``tol``, ``maxiter`` for the ``c_in`` CG).
    """

    def __init__(self, denoiser: "ColoredDenoiser", *, K: int | None = None,
                 c_in_kw: dict | None = None, backend=None):
        """``backend=None`` keeps the historical matrix-free path exactly.

        Supplying a :class:`~ccd.operators.backends.SpectralBackend` makes every apply in the
        reverse pass exact — which is the fMRI benchmark's primary configuration, where the
        matrix-free route is measured not to resolve.
        """
        self.denoiser = denoiser
        self.fwd = denoiser.fwd
        self.op = self.fwd.op
        self.lo, self.hi = self.fwd.lo, self.fwd.hi
        self.phi = self.fwd.phi
        self.K = K
        self.c_in_kw = dict(c_in_kw or {})
        self.backend = backend

    def _apply(self, f, v: Tensor) -> Tensor:
        """The single choke point for every function-of-``S`` apply in the reverse pass.

        ``backend`` (see ``__init__``) routes all of them — including the direct ``q^{1/2} ξ`` at
        :meth:`step` — through an exact dense eigensystem when supplied. Note what that means: the
        *unsplit* ancestral noise form, which the split was invented to avoid, is correct again once
        the apply is exact. The split exists to rescue the **matrix-free** route, not the law.
        """
        if self.backend is not None:
            return self.backend.apply(f, v)
        return apply_fun(f, self.op, v, self.lo, self.hi, self.K)

    # ----- per-band scalar functions f(λ) fed to ApplyFun (all smooth; no inverse, no matrix exponential) -----

    def _mean_fn(self, F: float):
        """Mean operator band ``m_F(λ) = e^{−½ F φ(λ)}``."""
        phi = self.phi
        return lambda lam: torch.exp(-0.5 * F * phi(lam))

    def _a_fn(self, F: float):
        """Folded score coefficient ``a_F(λ) = φ(λ) / (1 − e^{−F φ(λ)})`` (``−expm1`` keeps small-``Fφ`` exact)."""
        phi = self.phi
        return lambda lam: phi(lam) / (-torch.expm1(-F * phi(lam)))


class LearnedReverseSDE(_LearnedReverse):
    """Stochastic reverse-time sampler with a learned colored Tweedie score.

    Constructor args are inherited from :class:`_LearnedReverse` (``denoiser``, ``K``, ``c_in_kw``). Two
    integrators (``scheme=`` on :meth:`step` / :meth:`integrate`): ``"euler_maruyama"`` (default, non-stiff
    clocks) and the stiff-stable ``"ancestral"`` (use for ``a=−1``).
    """

    def diffusion_std_fn(self):
        """Diffusion coefficient ``g(λ) = (λ φ(λ))^{1/2}`` — identical to the Gaussian sampler's."""
        phi = self.phi
        return lambda lam: torch.sqrt(torch.clamp(lam * phi(lam), min=0.0))

    def _ancestral_coeffs(self, F_a: float, F_b: float):
        """Per-band coefficients of the stiff-stable **ancestral / exponential** step (``ΔF = F_a − F_b``):

            x_{F_b} = c_x(S) x_{F_a} + c_d(S) D_θ(x_{F_a}, F_a) + q(S)^{1/2} ξ

        This is the reverse posterior ``q(x_{F_b} | x_{F_a}, x_0)`` of the colored forward OU with the point
        estimate ``x_0 ← D_θ`` (the DDPM/DPM-Solver "data-prediction" ancestral step). With
        ``Σ_F(λ)=λ(1−e^{−Fφ})`` the *conditional* forward-noise variance, ``m_F=e^{−½Fφ}``,
        ``T=e^{−½ΔFφ}`` and ``R(λ)=λ(1−e^{−ΔFφ})=Σ_{ΔF}``, a Gaussian-product gives

            c_x = T Σ_b/Σ_a,     c_d = R m_b/Σ_a,     q = R Σ_b/Σ_a.

        Every band is a smooth bounded function of ``S`` (``apply_fun``-able) — and unlike Euler–Maruyama the
        stiff *linear* part is integrated **exactly**, so it stays stable when the per-band clock ``F φ(λ)``
        spans a large dynamic range (e.g. ``a=−1``, where ``φ=λ^{-1}`` and EM diverges). Limits: ``ΔF→0`` ⇒
        ``c_x→1, c_d,q→0`` (no move); ``F_b→0`` ⇒ ``Σ_b→0, R→Σ_a`` ⇒ ``x_{F_b}→D_θ`` (built-in final denoise).
        The **mean** ``c_x + c_d·c_skip`` equals the Gaussian sampler's exact reverse mean band-for-band; the
        **noise** ``q`` is the conditional posterior variance given a *known* ``x_0``, so with the point
        estimate it under-injects by ``c_d²·Var(x_0|x_{F_a})`` (the standard data-prediction under-dispersion,
        ``O(ΔF)``), not the marginal-preserving variance of the linear ``GaussianReverseSDE`` "exact" scheme.

        Delegated to :mod:`ccd.forward.bands` so the audit holds these same callables (plan §2.1).
        ``noise_std`` is the direct ``q^{1/2} = √(λ·g)``. Its ``√λ`` is a branch point, so **as a
        Chebyshev series** it does not resolve on a ridge-dominated spectrum — but with an exact
        ``backend`` the direct form is correct, and it is what the fMRI benchmark's primary path
        uses, from one shared ``(d, batch)`` innovation block (plan §2.1b-ter).

        The split alternative — ``bands.g_sqrt_fn`` on a data-factor ``N(0,S)`` draw, exploiting
        ``q = λ·g`` — rescues the *matrix-free* route and is production only there (plan §2.1b,
        Remedy A). It costs an extra ``(n, batch)`` factor block on its own stream, which is why the
        primary path no longer pays for it.
        """
        return (bands.c_x_fn(self.phi, F_a, F_b),
                bands.c_d_fn(self.phi, F_a, F_b),
                bands.q_sqrt_fn(self.phi, F_a, F_b))

    # ----------------------------------------------------------------- drift / step / integrate

    def drift(self, x: Tensor, F: float) -> Tensor:
        """Reverse-SDE drift ``b(x, F) = −½ φ(S) x − a_F(S)(m_F(S) D_θ(x, F) − x)`` (nonlinear in ``x``)."""
        d = self.denoiser(x, F, **self.c_in_kw)                       # D_θ(x, F)
        resid = self._apply(self._mean_fn(F), d) - x                  # m_F(S) D_θ − x
        return -0.5 * self._apply(self.phi, x) - self._apply(self._a_fn(F), resid)

    def step(
        self,
        x: Tensor,
        F_a: float,
        F_b: float,
        *,
        scheme: str = "euler_maruyama",
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """One reverse step ``F_a → F_b`` (``F_b ≤ F_a``); draws fresh noise.

        ``scheme="euler_maruyama"`` (default) is the explicit SDE discretisation — cheap and fine for a
        non-stiff clock (e.g. ``a=0``), but it **diverges** when the per-band clock ``Fφ(λ)`` spans a large
        dynamic range (e.g. ``a=−1``). ``scheme="ancestral"`` is the stiff-stable exponential/data-prediction
        step (:meth:`_ancestral_coeffs`) — the linear part exact — and is the one to use there.
        """
        if scheme not in ("euler_maruyama", "ancestral"):
            raise ValueError(f"scheme must be 'euler_maruyama' or 'ancestral'; got {scheme!r}")
        F_a, F_b = float(F_a), float(F_b)
        if F_b > F_a:
            raise ValueError(f"reverse step needs F_b <= F_a; got F_a={F_a}, F_b={F_b}")
        if F_b == F_a:
            return x
        with torch.no_grad():
            if scheme == "ancestral":
                d = self.denoiser(x, F_a, **self.c_in_kw)             # x_0 prediction
                c_x, c_d, noise_std = self._ancestral_coeffs(F_a, F_b)
                xi = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
                return self._apply(c_x, x) + self._apply(c_d, d) + self._apply(noise_std, xi)
            dF = F_a - F_b                                            # ΔF > 0  (Euler–Maruyama)
            b = self.drift(x, F_a)
            xi = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)
            noise = self._apply(self.diffusion_std_fn(), xi)
            return x - dF * b + dF ** 0.5 * noise

    def integrate(
        self,
        x_T: Tensor,
        F_grid,
        *,
        scheme: str = "euler_maruyama",
        generator: torch.Generator | None = None,
        denoise_final: bool = True,
    ) -> Tensor:
        """Integrate the reverse SDE along a decreasing ``F_grid``; ``x_T`` is the draw at ``F_grid[0]``.

        ``scheme`` selects the per-step integrator (see :meth:`step`): ``"euler_maruyama"`` (default) or the
        stiff-stable ``"ancestral"``. With ``denoise_final=True`` (default) the output is the clean estimate
        ``D_θ(x_{F_min}, F_min)`` (EDM-style) rather than the still-noisy last iterate.
        """
        Fs = [float(f) for f in F_grid]
        x = x_T
        for F_a, F_b in zip(Fs[:-1], Fs[1:]):
            x = self.step(x, F_a, F_b, scheme=scheme, generator=generator)
        if denoise_final:
            with torch.no_grad():
                x = self.denoiser(x, Fs[-1], **self.c_in_kw)
        return x


class LearnedReversePFODE(_LearnedReverse):
    """Deterministic probability-flow ODE reverse sampler with a learned colored Tweedie score.

    The learned analogue of :class:`ccd.reverse.GaussianReversePFODE`. The PF-ODE shares the reverse SDE's
    marginals but drops the noise and carries the score at **half weight** (Anderson): per band its drift is
    the SDE drift's linear part plus a half-strength score correction,

        dx/dF = b(x, F) = −½ φ(S) x − ½ a_F(S) (m_F(S) D_θ(x, F) − x).

    With ``D_θ = c_skip(S) x`` (the head) this collapses band-for-band to the analytic PF-ODE drift
    ``c_F(λ) = ½ φ τ e^{−Fφ}/v_F`` — the V3a-style collapse test. Unlike the reverse SDE the PF-ODE is **not
    stiff** under a coarse-to-fine clock: ``c_F`` *vanishes* as ``F→∞`` (the SDE drift → ½φ, the 124× spread
    that diverges EM at ``a=−1``) and is bounded for ``F ≥ F_min > 0``, so the explicit schemes below stay
    stable at **both** clocks — no exponential integrator is needed. Being deterministic it also removes the
    ancestral scheme's sampling noise (a cleaner readout of the trained score); it does **not** remove
    training run-noise (that lives in the weights).

    Constructor args are inherited from :class:`_LearnedReverse`. Two integration schemes (see :meth:`step`):
    ``"heun"`` (default) — the 2nd-order predictor–corrector (trapezoidal / EDM) step, ``O(ΔF²)``; ``"euler"`` —
    explicit Euler, ``O(ΔF)``, and the scheme that collapses to ``GaussianReversePFODE``'s Euler step to
    machine precision when the body is zeroed.
    """

    def drift(self, x: Tensor, F: float) -> Tensor:
        """PF-ODE drift ``b(x, F) = −½ φ(S) x − ½ a_F(S)(m_F(S) D_θ(x, F) − x)`` — the reverse-SDE drift with the
        score term at half weight (deterministic; ``dx/dF = b``)."""
        d = self.denoiser(x, F, **self.c_in_kw)                       # D_θ(x, F)
        resid = self._apply(self._mean_fn(F), d) - x                  # m_F(S) D_θ − x
        return -0.5 * self._apply(self.phi, x) - 0.5 * self._apply(self._a_fn(F), resid)

    def step(self, x: Tensor, F_a: float, F_b: float, *, scheme: str = "heun") -> Tensor:
        """One deterministic reverse step ``F_a → F_b`` (``F_b ≤ F_a``); no noise, no generator.

        ``scheme="heun"`` (default) is the 2nd-order predictor–corrector (trapezoidal) step — a Euler predictor
        ``x̃ = x − ΔF·b(x,F_a)`` then the average-slope corrector ``x − ½ΔF·(b(x,F_a) + b(x̃,F_b))``. ``"euler"``
        is the explicit first-order step. Both need ``F_b > 0`` (the ``a_F`` band diverges at ``F=0``);
        integrate to a positive ``F_min`` and use ``denoise_final`` for the clean final estimate.
        """
        if scheme not in ("euler", "heun"):
            raise ValueError(f"scheme must be 'euler' or 'heun'; got {scheme!r}")
        F_a, F_b = float(F_a), float(F_b)
        if F_b > F_a:
            raise ValueError(f"reverse step needs F_b <= F_a; got F_a={F_a}, F_b={F_b}")
        if F_b == F_a:
            return x
        with torch.no_grad():
            dF = F_a - F_b                                            # ΔF > 0
            b_a = self.drift(x, F_a)
            x_euler = x - dF * b_a
            if scheme == "euler":
                return x_euler
            b_b = self.drift(x_euler, F_b)                            # Heun corrector at the predicted endpoint
            return x - 0.5 * dF * (b_a + b_b)

    def integrate(self, x_T: Tensor, F_grid, *, scheme: str = "heun", denoise_final: bool = True) -> Tensor:
        """Integrate the reverse PF-ODE along a decreasing ``F_grid`` (``x_T`` is the draw at ``F_grid[0]``).

        Deterministic — no injected sampling noise and no ``generator``, so it is bit-identical across calls on
        a deterministic backend (CPU). On GPU it still inherits CUDA-kernel nondeterminism through ``D_θ`` (the
        TAGConv scatter-add), like any GPU computation — so repeats agree statistically, not bit-for-bit.
        ``scheme`` selects the per-step integrator (see :meth:`step`); with ``denoise_final=True`` (default) the
        output is the clean estimate ``D_θ(x_{F_min}, F_min)`` (EDM-style) rather than the last ODE iterate.
        """
        Fs = [float(f) for f in F_grid]
        x = x_T
        for F_a, F_b in zip(Fs[:-1], Fs[1:]):
            x = self.step(x, F_a, F_b, scheme=scheme)
        if denoise_final:
            with torch.no_grad():
                x = self.denoiser(x, Fs[-1], **self.c_in_kw)
        return x
