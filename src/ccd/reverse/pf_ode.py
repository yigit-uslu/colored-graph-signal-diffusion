"""Gaussian-scoped reverse-time probability-flow ODE sampler (Phase I3a).

In the Gaussian idealisation the data is ``x_0 ~ N(0, Ĉ_n)`` with ``Ĉ_n = S − τI``, so the
colored forward marginal is *exactly* ``p_F = N(0, V_F)`` with ``V_F = S − τ e^{−Fφ(S)}``
(:mod:`ccd.forward.marginals`). The analytic score is then ``∇log p_F(x) = −V_F^{-1} x`` and the
probability-flow ODE of the colored forward SDE (``β≡1``, so ``F = t``)

    dx/dF = −½ φ(S) x − ½ Sφ(S) ∇log p_F(x) = ½ Sφ(S) V_F^{-1} x − ½ φ(S) x

reduces, per spectral band ``λ`` (all operators are functions of ``S`` and commute), to the
scalar linear ODE

    dx_λ/dF = c_F(λ) x_λ,   c_F(λ) = ½ φ(λ) τ e^{−Fφ(λ)} / v_F(λ),   v_F(λ) = λ − τ e^{−Fφ(λ)},

using ``λ − v_F(λ) = τ e^{−Fφ(λ)}``. Its exact flow from ``F_a`` to ``F_b`` multiplies each band
by ``√(v_{F_b}(λ)/v_{F_a}(λ))`` — the deterministic PF-ODE of a mean-zero Gaussian just rescales
every mode by its standard-deviation ratio.

Two facts keep this in the *smooth* Chebyshev regime — no eigendecomposition and, crucially,
**no explicit inverse**:

* the exact per-step factor ``g(λ)=√(v_{F_b}(λ)/v_{F_a}(λ))`` is a smooth function of ``S`` on
  ``[τ, λ_hi]`` (finite even at ``F_b = 0``, where ``g(λ_min)`` stays bounded because ``v_0 =
  λ − τ`` is the clean-data variance);
* the Euler drift ``c_F(λ)`` contains ``1/v_F`` which, for any ``F ≥ F_min > 0``, is bounded
  (``v_F(λ) ≥ τ(1 − e^{−Fφ(τ)}) > 0``) — so each step is a plain :func:`ccd.operators.apply_fun`.

The steep inverse ``V_t^{-1}`` (needing shifted-CG / Zolotarev poles) only appears as
``F_min → 0`` with a *rank-deficient* ``Ĉ_n`` (``V_0`` singular) or once a *learned* score
replaces the analytic one — both deferred past I3a.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ccd.operators.chebyshev import apply_fun
from ccd.operators.linop import CovarianceOperator
from ccd.forward.rate import ColouringRate, phi_power


def _interp_increasing(xq: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Linear interpolation of ``fp`` at query points ``xq``, for strictly increasing ``xp``."""
    idx = torch.searchsorted(xp, xq).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    t = (xq - x0) / (x1 - x0).clamp_min(1e-30)
    return f0 + t * (f1 - f0)


def reverse_f_grid(
    F_max: float,
    F_min: float,
    n_steps: int,
    *,
    spacing: str = "uniform",
    phi: ColouringRate | None = None,
    tau: float | None = None,
    lo: float | None = None,
    hi: float | None = None,
    lam_sample: Tensor | None = None,
    lam_bar: float | None = None,
    weight: str = "rel",
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Decreasing grid ``F_max = F[0] > … > F[n_steps] = F_min`` (``n_steps`` reverse steps).

    Step placement matters only for the **approximate** schemes (``"euler"`` / ``"euler_maruyama"``);
    the exact schemes telescope and are grid-independent. The I3b-2 study (see
    ``scripts/bench_step_placement.py``) found the ``F*``-quantile / generation-order grid to be a
    *poor* choice — it targets where bands are generated, not where the integrator is stiff, and is
    degenerate at ``a=0`` — so it is intentionally **not** offered here (it lives only in the bench as
    a diagnostic). The spacings below are all **matrix-free** (no eigenvalues, respecting the
    ``no-eig in src/ccd`` guard):

    * ``"uniform"`` — linear in ``F``.
    * ``"geometric"`` — log-spaced in ``F`` (``F_min>0``); resolution near the data end.
    * ``"uniform_logsnr"`` — uniform in a reference band's log-SNR ``ν=−log(e^{Fφ̄}−1)`` (EDM/VDM
      style). Needs ``phi`` and a reference ``lam_bar`` (defaults to ``√(lo·hi)``); ``F_min>0``.
    * ``"stiffness"`` — equispaces ``ln v_F`` of the stiffest band ``λ=lo`` (``v_F=λ−τe^{−Fφ}``).
      Needs ``phi, tau, lo``; ``F_min>0``. Never collapses; matrix-free from the bracket alone.
    * ``"equidistribute"`` — the error-equidistribution (arc-length) grid: nodes equalise
      ``∫ρ dF`` with activity density ``ρ(F)=Σ_λ|d\\,{\\rm log}\\,v_F/dF|`` (``weight="rel"``, for the
      PF-ODE) or ``Σ_λ|dv_F/dF|`` (``weight="abs"``, for the SDE). The study's most consistent
      winner. Needs ``phi, tau`` and a band sample ``lam_sample`` (defaults to a log-spaced proxy
      over ``[lo,hi]``; pass the true spectrum / a DOS estimate to sharpen it); ``F_min>0``.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1; got {n_steps}")
    if F_max <= F_min:
        raise ValueError(f"need F_max > F_min; got F_max={F_max}, F_min={F_min}")

    if spacing == "uniform":
        return torch.linspace(F_max, F_min, n_steps + 1, dtype=dtype)

    if spacing == "geometric":
        if F_min <= 0:
            raise ValueError("geometric spacing requires F_min > 0")
        return torch.logspace(math.log10(F_max), math.log10(F_min), n_steps + 1, dtype=dtype)

    # ---- the placement-study spacings all need F_min > 0 (log-SNR / log-variance diverge at F=0) ----
    if F_min <= 0:
        raise ValueError(f"spacing {spacing!r} requires F_min > 0")

    if spacing == "uniform_logsnr":
        if phi is None:
            raise ValueError("uniform_logsnr requires phi")
        if lam_bar is None:
            if lo is None or hi is None:
                raise ValueError("uniform_logsnr requires lam_bar or (lo, hi)")
            lam_bar = (float(lo) * float(hi)) ** 0.5
        phibar = float(phi(torch.tensor(float(lam_bar), dtype=dtype)))
        nu = lambda F: -math.log(math.expm1(F * phibar))          # log-SNR of the reference band
        nus = torch.linspace(nu(F_max), nu(F_min), n_steps + 1, dtype=dtype)
        grid = torch.log1p(torch.exp(-nus)) / phibar
        grid[0], grid[-1] = F_max, F_min
        return grid

    if spacing == "stiffness":
        if phi is None or tau is None or lo is None:
            raise ValueError("stiffness requires phi, tau, lo")
        lam_w = float(lo)
        phi_w = float(phi(torch.tensor(lam_w, dtype=dtype)))
        lnv = lambda F: math.log(lam_w - float(tau) * math.exp(-F * phi_w))
        ws = torch.linspace(lnv(F_max), lnv(F_min), n_steps + 1, dtype=dtype)
        grid = -torch.log((lam_w - torch.exp(ws)) / float(tau)) / phi_w
        grid[0], grid[-1] = F_max, F_min
        return grid

    if spacing == "equidistribute":
        if phi is None or tau is None:
            raise ValueError("equidistribute requires phi and tau")
        if lam_sample is None:
            if lo is None or hi is None:
                raise ValueError("equidistribute requires lam_sample or (lo, hi)")
            lam_sample = torch.logspace(math.log10(float(lo)), math.log10(float(hi)), 256, dtype=dtype)
        lam = torch.as_tensor(lam_sample, dtype=dtype).reshape(-1)
        F_aux = torch.linspace(F_min, F_max, 2000, dtype=dtype)                  # increasing
        ph = phi(lam)
        e = torch.exp(-F_aux[:, None] * ph[None, :])
        dvdF = float(tau) * ph[None, :] * e                                      # τ φ e^{−Fφ}
        if weight == "rel":
            v = (lam[None, :] - float(tau) * e).clamp_min(1e-12)
            dens = (dvdF / v).sum(dim=1)
        elif weight == "abs":
            dens = dvdF.sum(dim=1)
        else:
            raise ValueError(f"weight must be 'rel' or 'abs'; got {weight!r}")
        C = torch.zeros(F_aux.numel(), dtype=dtype)
        C[1:] = torch.cumsum(0.5 * (dens[1:] + dens[:-1]) * (F_aux[1:] - F_aux[:-1]), dim=0)
        targets = torch.linspace(0.0, float(C[-1]), n_steps + 1, dtype=dtype)
        grid = torch.flip(_interp_increasing(targets, C, F_aux), dims=[0])       # decreasing
        grid[0], grid[-1] = F_max, F_min
        return grid

    raise ValueError(
        f"spacing must be one of uniform/geometric/uniform_logsnr/stiffness/equidistribute; "
        f"got {spacing!r}"
    )


class GaussianReversePFODE:
    """Deterministic PF-ODE reverse sampler for the colored Gaussian process.

    Args mirror :class:`ccd.forward.marginals.ColoredForward`: the operator ``S``, its spectrum
    bracket ``[lo, hi] ⊇ spec(S)``, the colouring rate ``φ`` (default ``φ(λ)=λ`` — the original
    ``S``-colouring), and a fixed Chebyshev degree ``K`` (``None`` ⇒ adaptive per apply).

    Two integration schemes (see :meth:`step`): ``"exact"`` applies the closed-form per-step
    factor ``√(v_{F_b}/v_{F_a})`` — exact for this linear-Gaussian ODE at *any* grid, so it is
    the production Gaussian sampler; ``"euler"`` is an explicit-Euler discretisation of the drift
    ``c_F`` with ``O(ΔF)`` error, used to exhibit convergence to the exact transport.
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

    def _transport_fn(self, F_a: float, F_b: float):
        """Exact PF-ODE factor ``g(λ) = √(v_{F_b}(λ) / v_{F_a}(λ))`` for the step ``F_a → F_b``."""
        v_a, v_b = self._v_fn(F_a), self._v_fn(F_b)
        return lambda lam: torch.sqrt(torch.clamp(v_b(lam) / v_a(lam), min=0.0))

    def _euler_fn(self, F_a: float, dF: float):
        """One explicit-Euler step operator ``s(λ) = 1 + dF·c_{F_a}(λ)`` (``dF = F_b − F_a < 0``)."""
        tau, phi = self.op.tau, self.phi

        def s(lam: Tensor) -> Tensor:
            e = torch.exp(-F_a * phi(lam))
            v = lam - tau * e
            c = 0.5 * phi(lam) * tau * e / v            # ½ φ τ e^{−F_a φ} / v_{F_a}
            return 1.0 + dF * c

        return s

    # ----------------------------------------------------------------- public API

    def transport_fn(self, F_a: float, F_b: float):
        """The scalar exact-transport function ``g(λ)`` for ``F_a → F_b`` (for oracles/inspection)."""
        return self._transport_fn(float(F_a), float(F_b))

    def step(self, x: Tensor, F_a: float, F_b: float, *, scheme: str = "exact") -> Tensor:
        """One reverse step ``F_a → F_b`` (``F_b ≤ F_a``) applied to ``x`` via ApplyFun."""
        F_a, F_b = float(F_a), float(F_b)
        if F_b > F_a:
            raise ValueError(f"reverse step needs F_b <= F_a; got F_a={F_a}, F_b={F_b}")
        if F_b == F_a:
            return x
        if scheme == "exact":
            f = self._transport_fn(F_a, F_b)
        elif scheme == "euler":
            f = self._euler_fn(F_a, F_b - F_a)
        else:
            raise ValueError(f"scheme must be 'exact' or 'euler'; got {scheme!r}")
        return apply_fun(f, self.op, x, self.lo, self.hi, self.K)

    def integrate(self, x_T: Tensor, F_grid, *, scheme: str = "exact") -> Tensor:
        """Integrate the reverse PF-ODE along a decreasing ``F_grid``.

        ``x_T`` is the terminal draw at ``F_grid[0]``; returns the state at ``F_grid[-1]``.
        With ``scheme="exact"`` the per-step factors telescope, so the result is independent of
        the grid; with ``scheme="euler"`` it converges to that exact transport as the grid refines.
        """
        Fs = [float(f) for f in F_grid]
        x = x_T
        for F_a, F_b in zip(Fs[:-1], Fs[1:]):
            x = self.step(x, F_a, F_b, scheme=scheme)
        return x
