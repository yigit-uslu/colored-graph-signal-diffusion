r"""Spectral deflation of an edge-limited apply: exact on a Ritz subspace, cheap on its complement.

Implements Stage S4 of the edge-limited backends comparison
(``docs/agent_summaries/metr_matvec_edge_limited_backends_handoff_v3_2026-09-18.md`` and its v4
amendment). Given an SPD matrix-free operator ``S`` and its top-``m`` (here: bottom-``m``, see below)
Ritz pairs ``(Theta, U)`` from a matrix-free Lanczos (:func:`ritz_pairs`, ``scipy.sparse.linalg.eigsh``
under a ``LinearOperator``, never a dense matrix), the exact decomposition is

    f(S) v  =  U f(Theta) U^T v  +  p(S) (I - U U^T) v

for ANY ``p`` — the identity holds because ``S`` acts as ``Theta`` exactly on the Ritz subspace and as
``S`` (still exactly) on its complement; the only approximation is (a) replacing the complement's exact
``f(S)`` with a cheaper ``p`` (fewer matvecs because the complement's effective bracket has moved away
from the singularity that made ``f`` edge-limited) and (b) ``(Theta, U)`` being Ritz pairs rather than
exact eigenpairs.

**Which end to deflate.** Every target function in this comparison (``c_out``, ``c_skip``, ``q_sqrt``,
``terminal``) is edge-limited at the *lower* bracket edge (branch point at or near ``lo``, per
``backends.py``'s own docstring). Deflating the smallest eigenvalues moves the complement's effective
lower edge up and away from the singularity — the opposite of a typical "deflate the dominant modes"
setup, and worth stating plainly since it is not the default ``eigsh`` config.

**The leakage gate, corrected (v4 S1).** v3's original gate — "the deflated-subspace component of the
*result*" — is mathematically invalid: for ``v`` an exact eigenvector of the deflated block, that
component is exactly 1 and the computation is exactly correct. The quantity that actually indicates a
defect is the **complement term bleeding back into the deflated subspace**:

    y_perp = p(S) (I - U U^T) v
    leakage = || U^T y_perp || / max(|| y_perp ||, floor)

which is near-zero when ``U`` is close to invariant and grows when it is not (Ritz pairs are only
approximately invariant, and a polynomial/rational ``p`` does not exactly commute with the projector
unless ``U`` is exactly invariant).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse.linalg as sla
import torch
from torch import Tensor

Matvec = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class RitzResult:
    """Bottom- (or top-) ``m`` Ritz pairs from a matrix-free Lanczos, plus its own matvec count."""

    evals: Tensor          # (m,)
    evecs: Tensor          # (d, m)
    matvec_count: int      # matvecs spent by eigsh producing this result (the deflation setup cost)
    converged: bool


def ritz_pairs(
    matvec: Matvec, d: int, m: int, *, which: str = "SA",
    tol: float = 0.0, maxiter: int | None = None,
    dtype: torch.dtype = torch.float64, device="cpu",
) -> RitzResult:
    """``m`` Ritz pairs of the SPD operator behind ``matvec``, via ``scipy.sparse.linalg.eigsh``.

    Matrix-free: ``eigsh`` sees a :class:`scipy.sparse.linalg.LinearOperator` built from ``matvec``
    alone (bridged through NumPy float64), never a ``d x d`` array. ``which="SA"`` (smallest algebraic)
    targets the low tail, the deflation direction this comparison actually wants (see module
    docstring). ``tol=0`` is ``eigsh``'s own "machine precision" default; pass a larger ``tol`` or a
    small ``maxiter`` to construct a deliberately **under-converged** negative control.
    """
    count = 0

    def mv(x: np.ndarray) -> np.ndarray:
        nonlocal count
        count += 1
        xt = torch.as_tensor(x, dtype=dtype, device=device)
        return matvec(xt).detach().cpu().numpy().astype(np.float64)

    op = sla.LinearOperator((d, d), matvec=mv, dtype=np.float64)
    k = min(int(m), d - 1)
    if k < 1:
        raise ValueError(f"m must leave at least one non-deflated mode; got m={m}, d={d}")
    try:
        evals_np, evecs_np = sla.eigsh(op, k=k, which=which, tol=tol, maxiter=maxiter)
        converged = True
    except sla.ArpackNoConvergence as exc:
        evals_np, evecs_np = exc.eigenvalues, exc.eigenvectors
        converged = False
    if evals_np.size == 0:
        raise RuntimeError(
            f"eigsh converged 0 of {k} requested Ritz pairs within maxiter={maxiter} "
            f"(spent {count} matvecs) -- the target spectrum is too tightly clustered for plain "
            "Lanczos at this budget; this is a genuine setup-cost finding, not a bug to route around"
        )
    order = np.argsort(evals_np)
    evals = torch.as_tensor(evals_np[order].copy(), dtype=dtype, device=device)
    evecs = torch.as_tensor(evecs_np[:, order].copy(), dtype=dtype, device=device)
    return RitzResult(evals=evals, evecs=evecs, matvec_count=count, converged=converged)


@dataclass(frozen=True)
class DeflatedApplyResult:
    y: Tensor
    y_perp: Tensor
    leakage: float


def deflated_apply(
    f_scalar: Callable[[Tensor], Tensor],
    complement_apply: Callable[[Tensor], Tensor],
    v: Tensor,
    ritz: RitzResult,
    *,
    leakage_floor: float = 1e-12,
    project_complement: bool = True,
) -> DeflatedApplyResult:
    """``y = U f(Theta) U^T v + p(S)(I-UU^T)v``, plus the corrected leakage diagnostic.

    ``project_complement=False`` is the negative control named in v4 S1: feed ``complement_apply`` the
    RAW ``v`` instead of its complement projection, which should inflate the leakage measurably (the
    gate is supposed to catch exactly this).
    """
    U, Theta = ritz.evecs, ritz.evals
    Utv = U.transpose(0, 1) @ v if v.ndim == 1 else U.transpose(0, 1) @ v
    exact_part = U @ (f_scalar(Theta) * Utv) if v.ndim == 1 else U @ (f_scalar(Theta)[:, None] * Utv)
    v_perp = v - U @ Utv
    y_perp = complement_apply(v_perp if project_complement else v)
    y = exact_part + y_perp
    num = torch.linalg.norm(U.transpose(0, 1) @ y_perp)
    den = max(float(torch.linalg.norm(y_perp)), leakage_floor)
    return DeflatedApplyResult(y=y, y_perp=y_perp, leakage=float(num) / den)


def break_even_n(m_setup: float, m_cloud_deflated: float, m_cloud_baseline: float) -> float | None:
    """Smallest ``N`` for which ``m_setup + N*m_cloud_deflated <= N*m_cloud_baseline``, or ``None``
    if the deflated per-cloud cost never beats the baseline (no amortisation is possible at any N)."""
    delta = m_cloud_baseline - m_cloud_deflated
    if delta <= 0:
        return None
    return m_setup / delta
