"""Graph-native operators for package A: ``S = f(L)`` as a matvec oracle, and a bracket that holds.

🛑 **Why this is a new module and not a branch inside ``ccd.operators.linop``.** ``linop.py``,
``marginals.py`` and ``denoiser/vnn.py`` are all inside ``stage_a_manifest.json``'s sealed 49-file
closure, so editing any of them makes every Stage-A consumer refuse with ``NotSealable``. A new
module reached only from the ``gfs_`` scripts keeps the sealed benchmark sealed — the
``ccd.reverse.exp_pfode`` precedent, and the route ``docs/followup_datasets_plan.md`` §5.C requires.

Two operators and one bracket policy:

* :class:`GraphFilterOperator` — ``S = p(L)² + τI``, the EXACT in-family prior of the graph-filtered
  instances (the data is ``x = p(L) w``, so ``Cov(x) = p(L)²`` by construction). Applied by
  Chebyshev on the sparse ``L``; no ``d×d`` matrix and no eigendecomposition.
* :class:`LaplacianOperator` — ``S = L + σI``, the Laplacian arm's operator. The shift is a required
  argument, not a default: see the bracket discussion below and
  ``docs/followup_datasets_plan.md`` §0 (``λ_min(L) = 0`` ⇒ the shift is mandatory).
* :func:`lanczos_bracket` — an enclosing ``[lo, hi] ⊇ spec(S)`` derived from the Ritz **residual
  bound**, replacing the 1 % multiplicative margin used by ``SpectralModel.from_matvec``.

⭐ **Why the bracket needed replacing, measured 2026-08-29 on an RGG at d = 2048.**
``SpectralModel.from_matvec`` sets ``lo = θ_min × 0.99``. On a covariance operator the bottom of the
spectrum is the ridge τ and Lanczos resolves it immediately, so 1 % is ample. On a Laplacian the
bottom is a DENSE bulk approaching 0 and 48 Lanczos steps return ``θ_min ≈ 1.0e-3`` against a true
``λ_min = 4.3e-16`` — a bracket that does **not** enclose the spectrum:

    shift    from_matvec lo    true λ_min     encloses?
    0.0      9.98e-04         4.34e-16       NO
    1e-3     1.05e-03         1.00e-03       NO   (the 1 % margin is smaller than the Ritz error)
    0.05     4.96e-02         5.00e-02       yes  (the shift dominates the Ritz error)

A non-enclosing bracket drives every Chebyshev filter out of domain, which is the documented failure
mode that "blew VP to 1e6 at a 2 % perturbation" (`docs/fmri_generalization_state.md`, operator
stability §, design fact 2).

🛑 **What actually guarantees enclosure here, corrected 2026-08-30 after review.** An earlier version
of this module claimed the Ritz residual gave a *bound* ``λ_min ≥ θ_min − r_min``. It does not, for
two independent reasons, and both matter:

1. **The residual is an estimate, not the textbook quantity.** Paige's residual is
   ``‖S y_i − θ_i y_i‖ = β_m |s_{m,i}|`` with ``β_m`` the norm computed at the FINAL Lanczos step.
   ``ccd.spectral.lanczos`` returns ``beta`` of length ``k−1`` and discards that last ``β``, so
   ``beta[-1]`` is ``β_{m−1}``. Measured on ``heat``'s ``L_sym + σI`` at m = 64: the value used here
   is 3.277e-3 while the true ``‖Sy − θy‖`` is 3.324e-3 — the estimate is *smaller* than the real
   residual, so it cannot be leaned on as a conservative bound. (The sealed ``from_matvec`` shares
   the convention; it is not edited, and no guarantee is inherited from it.)
2. **Even with the right ``β`` the theorem is the wrong one.** ``|λ − θ_i| ≤ r_i`` says *some*
   eigenvalue lies near each Ritz value. It does not say no eigenvalue lies below ``θ_min − r_min``:
   a start vector with little weight on the extreme eigenvector yields a converged-looking bottom
   pair with a tiny residual while the true ``λ_min`` sits far below.

**So enclosure rests on what is actually known, not on the residual:**

* **bottom** — the a-priori floor ``τ``, which is exact: every operator in this module is
  ``(something PSD) + τI``. This is why ``test_the_unresolved_regime_is_real_and_defeats_the_1pct_margin``
  passes — the clamp, not the residual, is doing the work.
* **top, Laplacian** — the analytic ``2 + σ`` (``spec(L_sym) ⊆ [0,2]`` for any graph), exact and
  needing no estimate; :func:`analytic_bracket`. The returned ``hi`` is the tighter of that and the
  Ritz estimate.
* **top, covariance** — ``θ_max·(1 + TOP_MARGIN)``. This is a **safety factor, not a guarantee**:
  Lanczos converges the top of a spectrum fast and the measured error is ≤ 2.4e-15 by m = 128 on
  every instance here (`measurements/gfs_spectral_verify.json`), so ``TOP_MARGIN`` sits ~4e8× above
  the observed error. It is named and justified rather than fitted, and it is stated as what it is.

``resolve_to`` remains a useful adaptive **stopping rule** — it is no longer described as a bound.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ccd.spectral import lanczos, ritz_values

__all__ = ["GraphFilterOperator", "LaplacianOperator", "SparseGraphOp", "TOP_MARGIN",
           "analytic_bracket", "cheb_apply_torch", "householder_apply_torch", "lanczos_bracket"]

#: Multiplicative safety factor on the estimated top of a covariance spectrum. NOT a bound — see the
#: module docstring. Justified by the measured Lanczos top-convergence (≤ 2.4e-15 at m ≥ 128).
TOP_MARGIN: float = 1e-6


def _as_torch_csr(L, *, dtype=torch.float64, device="cpu") -> Tensor:
    """Accept a scipy CSR or a torch sparse tensor; return a torch sparse CSR on ``device``."""
    if isinstance(L, Tensor):
        return L.to(device=device, dtype=dtype)
    import scipy.sparse as sp
    L = sp.csr_matrix(L)
    return torch.sparse_csr_tensor(
        torch.as_tensor(L.indptr, dtype=torch.int64),
        torch.as_tensor(L.indices, dtype=torch.int64),
        torch.as_tensor(L.data, dtype=dtype),
        size=L.shape,
    ).to(device)


def householder_apply_torch(V: Tensor, X: Tensor) -> Tensor:
    """Apply ``Q = H_1 ⋯ H_m`` to columns of ``X`` — the torch mirror of ``gen_gfs.householder_apply``.

    Exists so the oracle can reproduce the generator's non-stationary component without importing
    the generator into a torch path. The two are gated against each other in
    ``tests/test_gfs_data.py``, exactly as the two Clenshaw implementations are.
    """
    Y = X.clone()
    for v in torch.flip(V, dims=(0,)):
        Y -= 2.0 * torch.outer(v, v @ Y)
    return Y


def cheb_apply_torch(L: Tensor, c: np.ndarray, X: Tensor, *, lo: float, hi: float) -> Tensor:
    """Matrix Clenshaw: apply the Chebyshev polynomial ``c`` of ``L`` to columns of ``X``.

    The generator's :func:`gen_gfs.cheb_apply` in numpy and this in torch are the same recurrence;
    ``tests/test_gfs_data.py`` gates them against each other rather than trusting the restatement.
    """
    a = 2.0 / (hi - lo)
    b = -(hi + lo) / (hi - lo)

    def shift(V: Tensor) -> Tensor:
        return a * (L @ V) + b * V

    b1 = torch.zeros_like(X)
    b2 = torch.zeros_like(X)
    for k in range(len(c) - 1, 0, -1):
        b1, b2 = 2.0 * shift(b1) - b2 + float(c[k]) * X, b1
    return shift(b1) - b2 + float(c[0]) * X


class SparseGraphOp:
    """Common matvec-oracle surface: exactly what ``lanczos``/``kpm_moments`` touch.

    ``tau`` is part of that surface — :func:`ccd.spectral.lanczos` scales its breakdown test by it —
    so every subclass must set a positive ``tau`` meaning "the smallest eigenvalue is at least this".
    """

    def __init__(self, L, *, tau: float, dtype=torch.float64, device="cpu"):
        if tau <= 0:
            raise ValueError(f"tau must be > 0 (it is the guaranteed spectral floor); got {tau}")
        self.L = _as_torch_csr(L, dtype=dtype, device=device)
        self.d = int(self.L.shape[0])
        self.tau = float(tau)
        self._matvec_count = 0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.d, self.d)

    @property
    def dtype(self) -> torch.dtype:
        return self.L.dtype

    @property
    def device(self) -> torch.device:
        return self.L.device

    @property
    def matvec_count(self) -> int:
        return self._matvec_count

    def reset_count(self) -> None:
        self._matvec_count = 0

    def _count(self, v: Tensor) -> None:
        if v.shape[0] != self.d:
            raise ValueError(f"v must have leading dim d={self.d}; got {tuple(v.shape)}")
        self._matvec_count += 1

    def matvec(self, v: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError


class LaplacianOperator(SparseGraphOp):
    """``S = L + σI`` — the Laplacian arm's operator. ``shift`` is mandatory and positive.

    For a normalised Laplacian the enclosing bracket is analytic (``spec ⊆ [0, 2]``), so
    :func:`analytic_bracket` should be preferred over any estimate; :func:`lanczos_bracket` exists
    for the combinatorial Laplacian and for the covariance arms.
    """

    def __init__(self, L, *, shift: float, normalized: bool = True, **kw):
        super().__init__(L, tau=shift, **kw)
        self.shift = float(shift)
        self.normalized = bool(normalized)

    def matvec(self, v: Tensor) -> Tensor:
        self._count(v)
        return (self.L @ v) + self.shift * v


class GraphFilterOperator(SparseGraphOp):
    """``S = p(L)² + τI`` — the exact in-family prior of a graph-filtered instance.

    ``Cov(x) = p(L)²`` exactly, because the instance is generated as ``x = p(L) w`` with white ``w``
    (`scripts/gen_gfs.py`: the realised polynomial *is* the truth, so this oracle carries no
    approximation error). Each matvec is two Chebyshev applications of ``p``.
    """

    def __init__(self, L, coeffs: np.ndarray, *, tau: float, cheb_lo: float = 0.0,
                 cheb_hi: float = 2.0, **kw):
        super().__init__(L, tau=tau, **kw)
        self.c = np.asarray(coeffs, dtype=np.float64)
        self.cheb_lo, self.cheb_hi = float(cheb_lo), float(cheb_hi)

    def matvec(self, v: Tensor) -> Tensor:
        self._count(v)
        y = cheb_apply_torch(self.L, self.c, v, lo=self.cheb_lo, hi=self.cheb_hi)
        y = cheb_apply_torch(self.L, self.c, y, lo=self.cheb_lo, hi=self.cheb_hi)
        return y + self.tau * v


def analytic_bracket(op: SparseGraphOp) -> tuple[float, float]:
    """The exact enclosing bracket where the operator's spectrum is known in closed form.

    ``spec(L_sym) ⊆ [0, 2]`` for any graph, so ``S = L_sym + σI`` has ``spec ⊆ [σ, 2+σ]``. This needs
    no Lanczos, cannot be wrong, and is what the generator uses; prefer it whenever it applies.
    """
    if not isinstance(op, LaplacianOperator) or not op.normalized:
        raise TypeError("analytic_bracket applies to a normalised-Laplacian operator only")
    return (op.shift, 2.0 + op.shift)


def lanczos_bracket(op, *, m: int = 64, generator: torch.Generator | None = None,
                    floor: float | None = None, resolve_to: float | None = None,
                    m_max: int = 1024) -> tuple[float, float, dict]:
    """An enclosing ``[lo, hi] ⊇ spec(S)``: exact where the spectrum is known, estimated where not.

    Returns ``(lo, hi, info)``.

    * ``lo`` = the a-priori floor (``op.tau`` unless ``floor`` is given). Exact: every operator here
      is ``(something PSD) + τI``. The Ritz quantity is reported but does NOT set the bottom — see
      the module docstring for why it is not a bound.
    * ``hi`` = the analytic ``2 + σ`` for a normalised :class:`LaplacianOperator`, taking whichever
      of that and the Ritz estimate is tighter; otherwise ``θ_max·(1 + TOP_MARGIN)``.

    ⭐ **``resolve_to`` adapts ``m`` against the residual ESTIMATE — a stopping rule, not a
    guarantee.** Lanczos resolves the TOP of a spectrum in a few dozen steps and the BOTTOM only as
    fast as the bottom is separated: measured on a full-rank covariance at d = 2048, the relative
    error at ``λ_min`` runs 3.2e-2 / 6.0e-3 / 2.4e-5 / 3.4e-14 at ``m`` = 64 / 128 / 256 / 512,
    while ``λ_max`` is at 2.4e-15 by ``m`` = 128. Adapting on the computable residual estimate beats
    a step count picked because it passed, and ``info["resolved"]`` records whether it was met.
    """
    m_cur = int(m)
    while True:
        alpha, beta, cols = lanczos(op, m=m_cur, reorth="full", generator=generator)
        theta, last = ritz_values(alpha, beta)
        r = (float(beta[-1]) if beta.size else 0.0) * np.abs(np.asarray(last, dtype=float))
        rel_bottom = float(r[-1] / theta[-1]) if theta[-1] > 0 else float("inf")
        if resolve_to is None or rel_bottom <= float(resolve_to) or m_cur >= int(m_max):
            break
        m_cur = min(2 * m_cur, int(m_max))

    lo = float(op.tau if floor is None else floor)
    hi_est = float(theta[0] + r[0])
    if isinstance(op, LaplacianOperator) and op.normalized:
        hi = min(hi_est, analytic_bracket(op)[1])
        hi_source = "analytic" if hi < hi_est else "ritz_estimate"
    else:
        hi = float(theta[0]) * (1.0 + TOP_MARGIN)
        hi_source = "ritz_estimate_times_margin"
    info = {"ritz_min": float(theta[-1]), "ritz_max": float(theta[0]),
            "ritz_resid_est_min": float(r[-1]), "ritz_resid_est_max": float(r[0]),
            "rel_ritz_resid_est_min": rel_bottom,
            "resolved": bool(resolve_to is None or rel_bottom <= float(resolve_to)),
            "lo_source": "a_priori_floor", "hi_source": hi_source,
            "matvec_cols": int(cols), "m": int(m_cur)}
    return (lo, hi, info)
