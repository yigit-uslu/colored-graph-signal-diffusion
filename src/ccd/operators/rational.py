"""Rational (pole-sum) approximation of the steep inverse square root (Phase I3).

The forward marginal-variance operator ``V_F = v_F(S)`` is smooth and bounded, but its
inverse square root is *steep* near the data end: ``v_F(λ) = λ - τ e^{-Fφ(λ)} → 0`` as
``F → 0`` (and is exactly singular at ``λ = τ`` when ``Ĉ_n`` is rank-deficient). A Chebyshev
polynomial of ``1/√v_F`` would need degree ``O(√κ)`` (``κ = v_max/v_min``) and blows up as
``F → 0``. A rational approximation

    y^{-1/2} ≈ Σ_j w_j / (y + σ_j),      σ_j > 0,  w_j > 0      on  [v_min, v_max],

needs only ``O(log κ)`` poles, and is applied matrix-free by
:func:`ccd.operators.solvers.shifted_cg` (all poles share one Krylov space). This module
supplies the *scalar* pole/weight data ``{σ_j, w_j}`` — pure ``O(N)`` arithmetic, no matvec
and no eigendecomposition.

Two rules:

* ``method="zolotarev"`` (default) — the Hale–Higham–Trefethen elliptic quadrature, which is
  essentially optimal (pole count ``~log κ``). It evaluates the *scalar* complete elliptic
  integral and Jacobi elliptic functions via :mod:`scipy.special` — imported lazily, and it
  falls back to ``"gauss"`` if SciPy is unavailable.
* ``method="gauss"`` — an elementary Gauss–Legendre-of-Stieltjes rule (NumPy only). Accurate
  for modest ``κ``; it needs more poles and degrades in the extreme-``κ`` tail, so prefer
  ``"zolotarev"`` when ``F`` is small.

Pole placement is precision-sensitive, so the scalars are computed in float64 (the downstream
operator applies may stay float32).
"""

from __future__ import annotations

import math
import warnings

import torch
from torch import Tensor


def n_poles_for(v_min: float, v_max: float, eps: float, *, method: str = "zolotarev") -> int:
    """Predicted pole count for relative accuracy ``eps`` of ``y^{-1/2}`` on ``[v_min, v_max]``.

    Zolotarev: ``⌈ln(κ)·ln(4/eps)/π²⌉`` with ``κ = v_max/v_min`` — verified slightly conservative
    (e.g. 8 poles at ``κ=1e2``, ~29 at ``κ=1e8`` for ``eps=1e-6``, both below the measured need).
    The elementary ``"gauss"`` rule is inflated by ~2.5× as a rough guide.
    """
    if v_min <= 0.0:
        raise ValueError(f"v_min must be > 0; got {v_min}")
    kappa = max(v_max / v_min, 1.0 + 1e-12)
    base = math.log(kappa) * math.log(4.0 / eps) / math.pi**2
    n = max(int(math.ceil(base)), 2)
    return n if method == "zolotarev" else max(int(math.ceil(2.5 * n)), 4)


def _zolotarev_invsqrt(v_min: float, v_max: float, N: int) -> tuple[Tensor, Tensor]:
    """Hale–Higham–Trefethen elliptic poles/weights for ``y^{-1/2}`` (float64)."""
    from scipy.special import ellipj, ellipk  # lazy; scalar special functions (matrix-free)

    mpar = 1.0 - v_min / v_max                             # elliptic parameter k² for SciPy
    Kp = float(ellipk(mpar))
    j = torch.arange(1, N + 1, dtype=torch.float64)
    u = ((2.0 * j - 1.0) / (2.0 * N) * Kp).numpy()
    sn, cn, dn, _ = ellipj(u, mpar)
    sn = torch.as_tensor(sn, dtype=torch.float64)
    cn = torch.as_tensor(cn, dtype=torch.float64)
    dn = torch.as_tensor(dn, dtype=torch.float64)
    sigma = v_min * sn**2 / cn**2
    w = (2.0 * math.sqrt(v_min) * Kp / (math.pi * N)) * dn / cn**2
    return sigma, w


def _gauss_invsqrt(v_min: float, v_max: float, N: int) -> tuple[Tensor, Tensor]:
    """Elementary Gauss–Legendre-of-Stieltjes poles/weights for ``y^{-1/2}`` (float64, NumPy only).

    ``y^{-1/2} = (2/π) ∫_0^∞ dx/(y + x²)`` with the algebraic map ``x = c(1+u)/(1-u)``,
    ``c = (v_min v_max)^{1/4}``, so ``σ = x² > 0``.
    """
    from numpy.polynomial.legendre import leggauss

    u_np, wl_np = leggauss(N)
    u = torch.as_tensor(u_np, dtype=torch.float64)
    wl = torch.as_tensor(wl_np, dtype=torch.float64)
    c = (v_min * v_max) ** 0.25
    x = c * (1.0 + u) / (1.0 - u)
    dxdu = c * 2.0 / (1.0 - u) ** 2
    sigma = x**2
    w = (2.0 / math.pi) * wl * dxdu
    return sigma, w


def invsqrt_poles(
    v_min: float,
    v_max: float,
    n_poles: int,
    *,
    method: str = "zolotarev",
    dtype: torch.dtype = torch.float64,
    device="cpu",
) -> tuple[Tensor, Tensor]:
    """Poles ``σ_j > 0`` and weights ``w_j > 0`` of ``y^{-1/2} ≈ Σ_j w_j/(y+σ_j)`` on ``[v_min, v_max]``.

    ``method="zolotarev"`` (default; needs SciPy, else falls back to ``"gauss"``) or the elementary
    ``method="gauss"``. All arithmetic is ``O(N)`` scalar — no matvec, no eigendecomposition.
    Strict positivity of both shifts and weights (the property that makes every shifted system SPD)
    is asserted before returning.
    """
    if not (0.0 < v_min <= v_max):
        raise ValueError(f"need 0 < v_min <= v_max; got v_min={v_min}, v_max={v_max}")
    if n_poles < 1:
        raise ValueError(f"n_poles must be >= 1; got {n_poles}")

    if method == "zolotarev":
        try:
            sigma, w = _zolotarev_invsqrt(v_min, v_max, n_poles)
        except ImportError:
            warnings.warn(
                "SciPy unavailable; falling back to the elementary 'gauss' pole rule "
                "(needs more poles and degrades for very small F).",
                RuntimeWarning,
                stacklevel=2,
            )
            sigma, w = _gauss_invsqrt(v_min, v_max, n_poles)
    elif method == "gauss":
        sigma, w = _gauss_invsqrt(v_min, v_max, n_poles)
    else:
        raise ValueError(f"unknown method {method!r}; use 'zolotarev' or 'gauss'")

    if not (bool((sigma > 0).all()) and bool((w > 0).all())):
        raise RuntimeError("pole rule produced a non-positive shift or weight")
    return sigma.to(dtype=dtype, device=device), w.to(dtype=dtype, device=device)
