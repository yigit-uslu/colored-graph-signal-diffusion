"""Colored forward marginals ``m_t, Σ_t^{1/2}, V_t`` as matrix-free functions of S (Phase I1).

For colouring rate ``φ`` and integrated rate ``F = ∫₀ᵗ β``, the closed-form Gaussian
marginal of the colored forward process is

    m_t        = exp(−½ F φ(S))                         (mean operator)
    Σ_t        = S (I − exp(−F φ(S)))                    (conditional covariance)
    V_t        = m_t Σ_0 m_t + Σ_t = S − τ exp(−F φ(S)) (marginal covariance)

with ``Σ_0 = Ĉ_n = S − τI`` and stationary law ``N(0, S)``. Each is applied to a vector
by Chebyshev–Clenshaw (:func:`ccd.operators.apply_fun`) — no matrix exponential is
formed and no eigendecomposition is taken. A one-shot forward draw is then
``x_t = m_t x_0 + Σ_t^{1/2} ε`` (see :mod:`ccd.forward.sampling`).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ccd.operators.chebyshev import apply_fun, chebyshev_degree
from ccd.operators.linop import CovarianceOperator
from ccd.operators.rational import invsqrt_poles, n_poles_for
from ccd.operators.solvers import cg, shifted_cg
from ccd.forward import bands
from ccd.forward.rate import ColouringRate, phi_power


class ColoredForward:
    """Matrix-free colored forward marginals for a fixed colouring rate ``φ``.

    Args:
        op: the covariance operator ``S``.
        lo, hi: spectrum bracket ``[λ_lo, λ_hi] ⊇ spec(S)``.
        phi: colouring rate ``φ`` (default ``φ(λ)=λ`` — the original ``S``-colouring).
        K: Chebyshev degree; ``None`` selects it adaptively per call.
    """

    def __init__(
        self,
        op: CovarianceOperator,
        lo: float,
        hi: float,
        *,
        phi: ColouringRate | None = None,
        K: int | None = None,
        backend=None,
    ):
        self.op = op
        self.lo = float(lo)
        self.hi = float(hi)
        self.phi = phi if phi is not None else phi_power(1.0)
        self.K = K
        # `backend=None` keeps the historical matrix-free path exactly. A SpectralBackend makes
        # every apply below exact -- the fMRI benchmark's primary configuration, where the
        # matrix-free route is measured not to resolve at any admissible degree.
        self.backend = backend

    def _apply(self, f, v: Tensor) -> Tensor:
        if self.backend is not None:
            return self.backend.apply(f, v)
        return apply_fun(f, self.op, v, self.lo, self.hi, self.K)

    # ----- scalar functions f(λ) fed to ApplyFun (no matrix exponential, no eigendecomposition) -----

    # Delegated to `ccd.forward.bands` so the audit holds the SAME callables the operator path
    # applies (implementation plan §2.1's required refactor). Arithmetic unchanged.

    def _mean_fn(self, F: float):
        return bands.m_fn(self.phi, F)

    def _sigma_half_fn(self, F: float):
        """``Σ_F^{1/2}`` — the direct form, carrying ``√λ``.

        ``√λ`` is a branch point at the bracket's lower endpoint, so **as a Chebyshev series** this
        fails the audit at every admissible degree on a ridge-dominated spectrum. That is a
        statement about the *approximation*, not the law: with an exact ``backend`` this form is
        correct and is what the fMRI benchmark's primary path uses (plan §2.1b-ter).

        The alternative — ``bands.h_sqrt_fn`` applied to a data-factor ``N(0,S)`` draw, which never
        forms ``√λ`` — exists to rescue the *matrix-free* route, and is production only there
        (the secondary operator-stability experiment, and the §3.3 teeth-control).
        """
        return bands.sigma_sqrt_fn(self.phi, F)

    def _h_half_fn(self, F: float):
        """``√(1−e^{−Fφ})`` — the split factor (Remedy A). Spectral: 1.1e-7 at K=24."""
        return bands.h_sqrt_fn(self.phi, F)

    def _vmarg_fn(self, F: float):
        return bands.v_fn(self.phi, self.op.tau, F)

    # ----------------------------------------------------------------- operator applications

    def mean(self, v: Tensor, F: float) -> Tensor:
        """Apply ``m_t = exp(−½ F φ(S))`` to ``v``."""
        return self._apply(self._mean_fn(F), v)

    def sigma_half(self, v: Tensor, F: float) -> Tensor:
        """Apply ``Σ_t^{1/2} = (S(I − exp(−F φ(S))))^{1/2}`` to ``v``."""
        return self._apply(self._sigma_half_fn(F), v)

    def v_marginal(self, v: Tensor, F: float) -> Tensor:
        """Apply the marginal covariance ``V_t = S − τ exp(−F φ(S))`` to ``v``."""
        return self._apply(self._vmarg_fn(F), v)

    # ----------------------------------------------------- steep inverses V_t^{-1}, V_t^{-1/2} (Phase I4)

    def _vmarg_matvec(self, F: float, K: int | None = None):
        """A closure ``w ↦ V_t w = v_F(S) w`` for the matrix-free solvers.

        ``v_F`` is the *smooth, bounded* marginal-variance scalar, so its Chebyshev degree is small and
        ``F``-mild — the steepness lives only in the *inverse*, which the Krylov solver handles. The
        degree is fixed once here (resolving ``v_F`` to the operator's precision floor) so every matvec
        in the solve uses the same ``K``; it must stay on ``v_F``, never on ``1/v_F``.
        """
        vfn = self._vmarg_fn(F)
        op, lo, hi = self.op, self.lo, self.hi
        if K is None:
            K = self.K
        if K is None:
            cheb_tol = 1e-4 if op.dtype == torch.float32 else 1e-9
            K = max(chebyshev_degree(vfn, lo, hi, tol=cheb_tol, dtype=op.dtype, device=op.device), 1)
        return lambda w: apply_fun(vfn, op, w, lo, hi, K)

    def _v_bracket(self, F: float, ngrid: int = 513) -> tuple[float, float]:
        """Enclosing interval ``[v_min, v_max] ⊇ spec(V_t)`` from a scalar grid of ``v_F`` on ``[lo, hi]``.

        No eigendecomposition: ``v_F`` is sampled (in float64) across the spectrum bracket. For the
        supported power-law family ``φ(λ)=λ^a`` (``a∈{−1,0,½,1}``) ``v_F`` is monotone increasing on
        ``[τ, λ_hi]``, so the endpoint ``v_min = v_F(lo) = τ(1−e^{−Fφ(τ)})`` (on the grid) is the exact
        infimum — tight when ``Ĉ_n`` is rank-deficient. For a general non-monotone ``φ`` with structure
        between grid nodes the grid min may overestimate the true infimum; such colourings are outside
        the supported regime, and :meth:`_require_positive_bracket` additionally rejects a non-finite
        bracket (so an ill-defined ``φ`` raises rather than silently seeding a non-SPD solve).
        """
        vfn = self._vmarg_fn(F)
        lam = torch.linspace(self.lo, self.hi, ngrid, dtype=torch.float64)
        vF = vfn(lam)
        return float(vF.min()), float(vF.max())

    def v_inverse_half(
        self,
        v: Tensor,
        F: float,
        *,
        tol: float = 1e-6,
        maxiter: int = 1000,
        n_poles: int | None = None,
        method: str = "zolotarev",
    ) -> tuple[Tensor, dict]:
        """Apply ``V_t^{-1/2}`` to ``v`` matrix-free (rational + shared-Krylov multishift CG).

        Requires ``F > 0`` with ``v_min > 0`` (the steep ``F→0`` / singular ``V_0`` regime is deferred);
        raises otherwise. ``n_poles`` defaults to :func:`n_poles_for` at accuracy ``tol``. Returns
        ``(y, info)``; a whitening/encode consumer must check ``info["converged"]`` before using ``y``.
        """
        v_min, v_max = self._require_positive_bracket(F, "v_inverse_half")
        if n_poles is None:
            n_poles = n_poles_for(v_min, v_max, tol)
        sigma, w = invsqrt_poles(v_min, v_max, n_poles, method=method, dtype=v.dtype, device=v.device)
        return shifted_cg(self._vmarg_matvec(F), v, sigma, w, tol=tol, maxiter=maxiter)

    def v_inverse(
        self,
        v: Tensor,
        F: float,
        *,
        tol: float = 1e-6,
        maxiter: int = 1000,
        route: str = "cg",
        precond=None,
        n_poles: int | None = None,
        method: str = "zolotarev",
    ) -> tuple[Tensor, dict]:
        """Apply ``V_t^{-1}`` to ``v`` matrix-free.

        ``route="cg"`` (default) runs (preconditioned) CG on ``V_t``; ``route="square"`` applies the
        ``V_t^{-1/2}`` pole-sum twice. Requires ``F > 0`` with ``v_min > 0`` (steep ``F→0`` deferred).
        Returns ``(x, info)``.
        """
        self._require_positive_bracket(F, "v_inverse")
        if route == "cg":
            return cg(self._vmarg_matvec(F), v, precond=precond, tol=tol, maxiter=maxiter)
        if route == "square":
            mid, info1 = self.v_inverse_half(v, F, tol=tol, maxiter=maxiter, n_poles=n_poles, method=method)
            out, info2 = self.v_inverse_half(mid, F, tol=tol, maxiter=maxiter, n_poles=n_poles, method=method)
            c1, c2 = info1["converged"], info2["converged"]
            merged = (c1 and c2) if isinstance(c1, bool) else (c1 & c2)
            info = {"iters": info1["iters"] + info2["iters"], "converged": merged, "relres": info2["relres"]}
            return out, info
        raise ValueError(f"unknown route {route!r}; use 'cg' or 'square'")

    def _require_positive_bracket(self, F: float, who: str) -> tuple[float, float]:
        """Shared precondition: ``F > 0``, a finite ``v_min > 0``, and a valid (non-negative) colouring.

        Rejects the steep/singular regime we defer (``v_min ≤ 0`` / non-finite ``v_F``) *and* an invalid
        forward process: the colouring must be a non-negative rate, ``φ(λ) ≥ 0`` on ``[lo, hi]`` (⇔
        ``Σ_F = λ(1−e^{−Fφ}) ≥ 0``). A negative lobe is anti-damping — it yields a negative conditional /
        posterior variance even where ``V_F`` stays SPD — so it is rejected here rather than silently
        clamped to zero downstream (e.g. in ``c_out`` or ``Σ_t^{1/2}``).
        """
        if F <= 0.0:
            raise ValueError(f"{who} requires F > 0 (the steep F→0 / singular V_0 case is deferred); got F={F}")
        v_min, v_max = self._v_bracket(F)
        if not (math.isfinite(v_min) and math.isfinite(v_max)) or v_min <= 0.0:
            raise ValueError(
                f"{who}: V_t is not finite/positive-definite on [{self.lo}, {self.hi}] at F={F} "
                f"(v_min={v_min:.3e}); need F ≥ F_min > 0 and a colouring with v_F finite and positive there."
            )
        lam = torch.linspace(self.lo, self.hi, 513, dtype=torch.float64)
        phi_min = float(self.phi(lam).min())
        if not math.isfinite(phi_min) or phi_min < -1e-12:
            raise ValueError(
                f"{who}: colouring φ must be a non-negative rate on [{self.lo}, {self.hi}] "
                f"(min φ={phi_min:.3e}); a negative φ is an anti-damping / invalid forward process."
            )
        return v_min, v_max
