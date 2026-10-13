"""Matrix-free Krylov solvers for the steep spectral inverses (Phase I3).

Both solvers are ``S``-agnostic: they take a ``matvec`` callable for a symmetric
positive-definite operator ``A`` (in ``ccd``, ``A = V_F = v_F(S)`` applied via
:func:`ccd.operators.apply_fun`). They touch ``A`` *only* through matvecs, inner products,
and axpys — no ``d×d`` matrix, no eigendecomposition — so the matrix-free guard stays green.

* :func:`cg` — preconditioned conjugate gradients for ``(A + shift·I) x = b`` (batched over the
  columns of ``b``); the engine behind ``V_F^{-1} b``.
* :func:`shifted_cg` — shared-Krylov multishift CG (Jegerlehner recurrence): solves
  ``(A + σ_j I) x_j = b`` for all ``σ_j > 0`` in a single Krylov space on the stiffest (seed)
  system, and returns either the stack of solutions or their weighted sum ``Σ_j w_j x_j`` — the
  rational operator apply behind ``V_F^{-1/2} b``.

Convergence is measured on the seed (base) residual, which for non-negative relative shifts
dominates every shifted residual, so a seed below ``tol`` certifies all shifted systems. Columns
and poles are frozen as soon as they converge, both to save work and to keep the scalar ζ-recurrence
away from the ``0/0`` it would hit past convergence.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

Matvec = Callable[[Tensor], Tensor]


def _coldot(x: Tensor, y: Tensor) -> Tensor:
    """Column-wise inner product of ``(d, B)`` tensors → ``(B,)``."""
    return (x * y).sum(dim=0)


def cg(
    matvec: Matvec,
    b: Tensor,
    *,
    shift: float = 0.0,
    precond: Matvec | None = None,
    tol: float = 1e-8,
    maxiter: int = 1000,
    x0: Tensor | None = None,
) -> tuple[Tensor, dict]:
    """Preconditioned CG for the SPD system ``(A + shift·I) x = b``.

    ``matvec`` applies ``A``; ``b`` is ``(d,)`` or ``(d, B)`` (each column an independent system
    sharing the operator). ``precond`` applies an SPD preconditioner ``M⁻¹``. Returns ``(x, info)``
    with ``info = {"iters", "converged", "relres"}`` (per-column ``converged``/``relres`` for a
    batched ``b``; scalars for a 1-D ``b``). ``x0`` enables warm starts. Matrix-free: no eig, no
    dense matrix.
    """
    single = b.dim() == 1
    B = b.unsqueeze(1) if single else b
    tiny = torch.finfo(B.dtype).tiny

    def Aop(x: Tensor) -> Tensor:
        return matvec(x) + shift * x if shift != 0.0 else matvec(x)

    if x0 is None:
        x = torch.zeros_like(B)
    else:
        x = (x0.unsqueeze(1) if single else x0).clone()
    r = B - Aop(x)
    z = precond(r) if precond is not None else r
    p = z.clone()
    rz = _coldot(r, z)
    bnorm = _coldot(B, B).sqrt().clamp_min(tiny)
    relres = _coldot(r, r).sqrt() / bnorm
    active = relres > tol
    iters = torch.zeros(B.shape[1], dtype=torch.int64, device=B.device)

    it = 0
    while it < maxiter and bool(active.any()):
        it += 1
        Ap = Aop(p)
        pAp = _coldot(p, Ap)
        ok = active & (pAp > 0)                                   # SPD ⇒ pAp>0; else breakdown
        alpha = torch.where(ok, rz / pAp.clamp_min(tiny), torch.zeros_like(rz))
        x = x + alpha.unsqueeze(0) * p
        r = r - alpha.unsqueeze(0) * Ap
        relres = _coldot(r, r).sqrt() / bnorm
        newly = active & (relres <= tol)
        iters = torch.where(newly, torch.full_like(iters, it), iters)
        broke = active & (pAp <= 0) & (relres > tol)
        active = (relres > tol) & (~broke)
        z = precond(r) if precond is not None else r
        rz_new = _coldot(r, z)
        beta = torch.where(active, rz_new / rz.clamp_min(tiny), torch.zeros_like(rz))
        p = torch.where(active.unsqueeze(0), z + beta.unsqueeze(0) * p, p)
        rz = torch.where(active, rz_new, rz)

    converged = relres <= tol
    iters = torch.where(converged, iters, torch.full_like(iters, it))
    x_out = x.squeeze(1) if single else x
    info = {
        "iters": int(iters.max()) if iters.numel() else 0,
        "converged": bool(converged.all()) if single else converged,
        "relres": float(relres.max()) if single else relres,
    }
    return x_out, info


def shifted_cg(
    matvec: Matvec,
    b: Tensor,
    shifts: Tensor,
    weights: Tensor | None = None,
    *,
    tol: float = 1e-8,
    maxiter: int = 1000,
    seed_shift: float | None = None,
) -> tuple[Tensor, dict]:
    """Shared-Krylov multishift CG: solve ``(A + σ_j I) x_j = b`` for all ``σ_j`` at once.

    ``shifts`` are the ``σ_j`` (all ``> 0`` in this design). With ``weights`` given, returns the
    weighted sum ``Σ_j w_j x_j`` (shape of ``b``) — the rational operator apply; otherwise returns
    the stack of solutions, shape ``(S, *b.shape)``. One Krylov space is built on the seed system
    ``A + seed_shift·I`` (default ``seed_shift = min(shifts)``, the stiffest), and every pole is
    obtained from scalar ζ/α/β recurrences with no extra matvecs. Returns ``(out, info)`` with
    ``info`` reporting the seed-residual convergence. Matrix-free: no eig, no dense matrix.
    """
    single = b.dim() == 1
    B = b.unsqueeze(1) if single else b
    d, ncol = B.shape
    dtype, device = B.dtype, B.device
    tiny = torch.finfo(dtype).tiny

    shifts = torch.as_tensor(shifts, dtype=dtype, device=device).reshape(-1)
    S = shifts.numel()
    if seed_shift is None:
        seed_shift = float(shifts.min())
    rel = shifts - seed_shift                                    # (S,) ≥ 0
    if bool((rel < -1e-12).any()):
        raise ValueError("seed_shift must be <= min(shifts) so relative shifts are non-negative")
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=dtype, device=device).reshape(-1)
        if weights.numel() != S:
            raise ValueError(f"weights has {weights.numel()} entries; expected {S}")

    def Aseed(x: Tensor) -> Tensor:
        return matvec(x) + seed_shift * x if seed_shift != 0.0 else matvec(x)

    r = B.clone()                                                # x_seed = 0 ⇒ r_0 = b
    p = B.clone()
    rr = _coldot(r, r)
    bnorm = rr.sqrt().clamp_min(tiny)
    zeta = torch.ones(S, ncol, dtype=dtype, device=device)
    zeta_prev = torch.ones(S, ncol, dtype=dtype, device=device)
    alpha_prev = torch.ones(ncol, dtype=dtype, device=device)
    beta_prev = torch.zeros(ncol, dtype=dtype, device=device)
    p_sh = B.unsqueeze(0).expand(S, d, ncol).clone()             # p^σ_0 = b
    rel_col = rel.unsqueeze(1)                                   # (S,1)

    if weights is not None:
        out = torch.zeros(d, ncol, dtype=dtype, device=device)
    else:
        out = torch.zeros(S, d, ncol, dtype=dtype, device=device)
    iters = torch.zeros(ncol, dtype=torch.int64, device=device)

    it = 0
    while it < maxiter:
        base = rr.sqrt() / bnorm                                 # (B,)
        col_active = base > tol
        if not bool(col_active.any()):
            break
        it += 1
        pole_active = (zeta * base.unsqueeze(0) > tol) & col_active.unsqueeze(0)   # (S,B)

        Ap = Aseed(p)
        pAp = _coldot(p, Ap)
        alpha = torch.where(col_active & (pAp > 0), rr / pAp.clamp_min(tiny), torch.zeros_like(rr))
        r = r - alpha.unsqueeze(0) * Ap
        rr_new = _coldot(r, r)
        beta = torch.where(col_active, rr_new / rr.clamp_min(tiny), torch.zeros_like(rr))

        a_n = alpha.unsqueeze(0)                                 # (1,B)
        num = zeta * zeta_prev * alpha_prev.unsqueeze(0)
        den = (
            a_n * beta_prev.unsqueeze(0) * (zeta_prev - zeta)
            + zeta_prev * alpha_prev.unsqueeze(0) * (1.0 + rel_col * a_n)
        )
        upd = pole_active & (den.abs() > tiny)
        zeta_new = torch.where(upd, num / torch.where(den.abs() > tiny, den, torch.ones_like(den)), zeta)
        za = torch.where(upd, zeta, torch.ones_like(zeta))       # safe denominator for frozen poles
        alpha_sh = torch.where(upd, a_n * zeta_new / za, torch.zeros_like(zeta))
        beta_sh = torch.where(upd, beta.unsqueeze(0) * (zeta_new / za) ** 2, torch.zeros_like(zeta))

        if weights is not None:
            out = out + torch.einsum("sb,sdb->db", weights.unsqueeze(1) * alpha_sh, p_sh)
        else:
            out = out + alpha_sh.unsqueeze(1) * p_sh
        p_sh = torch.where(
            upd.unsqueeze(1),
            zeta_new.unsqueeze(1) * r.unsqueeze(0) + beta_sh.unsqueeze(1) * p_sh,
            p_sh,
        )
        p = torch.where(col_active.unsqueeze(0), r + beta.unsqueeze(0) * p, p)

        newly = col_active & (rr_new.sqrt() / bnorm <= tol)
        iters = torch.where(newly, torch.full_like(iters, it), iters)
        zeta_prev, zeta = zeta, zeta_new
        alpha_prev, beta_prev = alpha, beta
        rr = torch.where(col_active, rr_new, rr)

    relres = rr.sqrt() / bnorm
    converged = relres <= tol
    iters = torch.where(converged, iters, torch.full_like(iters, it))
    if single:
        out = out.squeeze(-1)
    info = {
        "iters": int(iters.max()) if iters.numel() else 0,
        "converged": bool(converged.all()) if single else converged,
        "relres": float(relres.max()) if single else relres,
    }
    return out, info
