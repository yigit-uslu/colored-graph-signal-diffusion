"""The factored input whitener for the matrix-free route: ``c_in`` by CG on ``S``, not on ``V_F``.

``c_in = V_F^{-1/2}`` with ``V_F = v_F(S)`` and ``v_F(λ) = λ − τe^{−Fφ(λ)}``. The tree's solve
(:func:`ccd.forward.sampling.whiten`) fits ``y^{-1/2}`` on ``[v_min, v_max]`` and runs multishift CG on
``V_F``, so every CG iteration is a Chebyshev apply of ``v_F(S)``.

``v_F`` has one real zero ``λ*`` below the spectrum bracket (``v_F(0) ≤ 0 < v_F(lo)``), so

    c_in(λ) = (λ − λ*)^{-1/2} · g(λ),        g(λ) = ((λ − λ*) / v_F(λ))^{1/2},

and ``c_in x = g(S) (S − λ*I)^{-1/2} x``:

* the first factor is the same Zolotarev + shared-Krylov multishift CG, run on the shifted ``S``, so it
  costs one matvec per iteration;
* ``g`` is smooth on the bracket and costs one Chebyshev apply, at the policy's degree for ``"g"``.

Any shift ``μ < lo`` gives the same operator; ``λ*`` is the one that makes ``g`` smooth.

**Two refusals guard the construction:**

* ``v_F`` must change sign exactly once on ``[0, lo]``: a scan, then bisection. When ``e^{−Fφ(0)}``
  underflows, ``v_F(0)`` is 0 in float64 and ``λ* = 0``.
* ``g``'s resolved degree must stay below the policy's cap. A crossing the scan missed leaves an
  uncancelled root below ``lo``; that costs degree, not correctness, so the cap is where it is caught.

**Assumptions and scope.**

* **The marginal.** ``V_F = S − τe^{−Fφ(S)}`` is :class:`~ccd.forward.marginals.ColoredForward`'s
  marginal, with ``τ`` the operator's ridge and ``[lo, hi]`` a bracket of ``S``'s spectrum.
* **The clock.** ``φ`` must be finite on ``[0, lo]``, and ``v_F`` must cross zero once there. The
  refusals above enforce both at run time; a clock that fails them is not a candidate.
* **The operator.** Any operator with a ``matvec`` works. Forming ``S`` densely is a separate lever
  (:class:`~ccd.operators.gram.DenseGramOperator`).
* **What has been verified.** CPU only, on ``heat_d2048_e0`` and ``bandpass_d2048_e0`` (float32,
  registered solver settings). The per-call error against the dense ``c_in`` is at most 1.2e-5 at every
  grid ``F`` on three inputs. CUDA is untested, and every CG iteration synchronises with the host once.

:class:`FactoredCinEDM` routes :meth:`~ccd.denoiser.edm.ColoredEDM.c_in` through
:func:`factored_c_in` and logs each call's solve and ``g`` apply in the policy's degree log, under
the names ``c_in_solve`` (``K`` = CG iterations, one matvec each) and ``c_in_g`` (``K`` = ``g``'s
degree). A degree census therefore counts ``c_in``'s matvecs exactly.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ccd.degree_policy import DegreePolicy, PolicyColoredEDM
from ccd.forward.bands import v_fn
from ccd.forward.marginals import ColoredForward
from ccd.operators.chebyshev import apply_fun
from ccd.operators.rational import invsqrt_poles, n_poles_for
from ccd.operators.solvers import shifted_cg

__all__ = ["FactoredCinEDM", "FactoredCinError", "factored_c_in", "singular_root", "singular_solve",
           "smooth_factor_fn"]


class FactoredCinError(RuntimeError):
    """The factored whitener cannot be applied as asked."""


def singular_root(phi, tau: float, lo: float, F: float, *, n_scan: int = 4097, n_bisect: int = 200) -> float:
    """The zero ``λ*`` of ``v_F(λ) = λ − τe^{−Fφ(λ)}`` in ``[0, lo)``, in float64.

    Refuses unless ``v_F(lo) > 0`` and ``v_F`` changes sign exactly once on an ``n_scan``-point grid over
    ``[0, lo]`` (grid values that are exactly zero do not count as a change).
    """
    vfn = v_fn(phi, float(tau), float(F))

    def val(x: float) -> float:
        return float(vfn(torch.tensor([x], dtype=torch.float64))[0])

    if not val(float(lo)) > 0.0:
        raise FactoredCinError(f"v_F(lo) <= 0 at F={F}: no positive bracket to factor against")
    left, right = 0.0, float(lo)
    if not val(left) < 0.0:
        return 0.0
    grid = torch.linspace(left, right, int(n_scan), dtype=torch.float64)
    signs = torch.sign(vfn(grid))
    signs = signs[signs != 0]
    changes = int((torch.diff(signs) != 0).sum())
    if changes != 1:
        raise FactoredCinError(
            f"v_F changes sign {changes} times on [0, lo] at F={F}; the factor needs exactly one real "
            "zero below the spectrum")
    for _ in range(int(n_bisect)):
        mid = 0.5 * (left + right)
        if val(mid) < 0.0:
            left = mid
        else:
            right = mid
        if right - left <= 0.0:
            break
    return left


def singular_solve(op, v: Tensor, mu: float, lo: float, hi: float, *, tol: float, maxiter: int,
                   method: str) -> tuple[Tensor, dict]:
    """``(S − μI)^{-1/2} v`` by Zolotarev + multishift CG on ``S`` (one matvec per iteration)."""
    mu, lo, hi = float(mu), float(lo), float(hi)
    if not mu < lo:
        raise FactoredCinError(f"the shift must lie below the spectrum: mu={mu} >= lo={lo}")
    a, b = lo - mu, hi - mu
    n = n_poles_for(a, b, tol)
    sigma, w = invsqrt_poles(a, b, n, method=method, dtype=v.dtype, device=v.device)

    def matvec(x: Tensor) -> Tensor:
        return op.matvec(x) - mu * x if mu != 0.0 else op.matvec(x)

    y, info = shifted_cg(matvec, v, sigma, w, tol=tol, maxiter=maxiter)
    conv = info["converged"]
    return y, {"iters": int(info["iters"]),
               "converged": bool(conv) if isinstance(conv, bool) else bool(conv.all()),
               "n_poles": int(n), "kappa": float(b / a)}


def smooth_factor_fn(phi, tau: float, F: float, mu: float):
    """``g(λ) = ((λ − μ)/v_F(λ))^{1/2}``, formed in float64 and returned in the input's dtype.

    Near ``lo`` both ``λ − μ`` and ``v_F`` are differences of nearby numbers, so the scalar is formed in
    float64 whatever dtype the Chebyshev nodes arrive in.
    """
    vfn = v_fn(phi, float(tau), float(F))
    mu = float(mu)

    def g(lam: Tensor) -> Tensor:
        l64 = lam.to(torch.float64)
        return torch.sqrt((l64 - mu) / vfn(l64)).to(lam.dtype)

    return g


def factored_c_in(fwd: ColoredForward, v: Tensor, F: float, *, policy: DegreePolicy, tol: float = 1e-6,
                  maxiter: int = 1000, method: str = "zolotarev", mu: float | None = None,
                  require_converged: bool = True) -> tuple[Tensor, dict]:
    """``c_in v = g(S) (S − μI)^{-1/2} v``, with ``μ = λ*`` unless given.

    Refuses a non-converged solve (unless ``require_converged=False``) and a ``g`` degree at the
    policy's cap. Returns ``(out, info)`` with the iterations, pole count, ``μ`` and ``K_g``.
    """
    lo, hi = float(fwd.lo), float(fwd.hi)
    fwd._require_positive_bracket(F, "c_in")
    mu = singular_root(fwd.phi, fwd.op.tau, lo, F) if mu is None else float(mu)
    y, info = singular_solve(fwd.op, v, mu, lo, hi, tol=tol, maxiter=maxiter, method=method)
    if require_converged and not info["converged"]:
        raise FactoredCinError(
            f"the (S - mu I)^(-1/2) solve did not converge at F={F} in {info['iters']} iterations; "
            "refusing to return a partial whitening")
    g = smooth_factor_fn(fwd.phi, fwd.op.tau, F, mu)
    K_g = policy.resolve("g", g, lo, hi, dtype=v.dtype, device=v.device)
    if K_g >= int(policy.k_max):
        raise FactoredCinError(
            f"g's Chebyshev degree reached the policy's cap ({K_g} >= k_max={policy.k_max}) at F={F}, "
            f"mu={mu}: an uncancelled zero of v_F below lo would do this, so the factor is refused")
    out = apply_fun(g, fwd.op, y, lo, hi, K_g)
    return out, {**info, "mu": mu, "K_g": int(K_g)}


class FactoredCinEDM(PolicyColoredEDM):
    """:class:`PolicyColoredEDM` whose ``c_in`` is :func:`factored_c_in`, logged in the degree log."""

    def c_in(self, v: Tensor, F: float, **solver_kw) -> Tensor:
        out, info = factored_c_in(self.fwd, v, float(F), policy=self.policy, **solver_kw)
        if self.degree_log is not None:
            self.degree_log.append({"name": "c_in_solve", "F": float(F), "K": info["iters"],
                                    "mu": info["mu"], "n_poles": info["n_poles"]})
            self.degree_log.append({"name": "c_in_g", "F": float(F), "K": info["K_g"], "mu": info["mu"]})
        return out
