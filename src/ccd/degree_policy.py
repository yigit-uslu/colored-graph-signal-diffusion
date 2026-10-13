"""Per-function Chebyshev degrees for the matrix-free route: the high degree goes to ``c_out`` alone.

On the matrix-free route every function of ``S`` is a Chebyshev–Clenshaw apply costing ``K`` matvecs,
and a single ``ColoredForward.K`` sets that degree for ``c_skip``, ``c_out`` and ``c_out²`` alike. The
reverse sampler's own ``K`` sets it for the drift bands ``m_F``, ``φ`` and ``a_F``. Only one of these
needs a high degree:

* **``c_out = (Σ_0 Σ_F V_F^{-1})^{1/2}``** has a ``√(λ−τ)`` branch point on the bracket's left edge,
  so it converges algebraically, error ∝ 1/K (Round A, ``measurements/gfs_matvec_budget.json``).
* **The others** (``c_skip``, ``c_out²``, ``m_F``, ``φ``, ``a_F`` and the terminal ``S^{1/2}``) are
  analytic on ``[lo, hi]`` and converge geometrically. How fast depends on ``F`` and on how close the
  clock's own singularity at ``λ = −s`` sits to the bracket.

A uniform degree therefore spends ``c_out``'s ``K`` on every apply: ten per Heun step at ``K = 256``.

:class:`DegreePolicy` fixes ``c_out``'s degree. Every other function gets either a fixed degree or
**the tree's own adaptive rule**, :func:`~ccd.operators.chebyshev.chebyshev_degree`: the smallest
degree whose coefficient tail falls below ``tol``, capped at ``k_max``. The adaptive rule is resolved
per call, so per ``F``.

Two subclasses carry the policy, and each overrides only the choke point it needs:

* :class:`PolicyColoredEDM` applies ``c_skip``, ``c_out`` and ``c_out_sq`` each at its named degree.
  The parent's shared ``_apply_precond`` (one ``fwd.K`` for all three) raises, so no path can reach it
  silently.
* :class:`PolicyLearnedReversePFODE` overrides ``_apply``, the reverse sampler's single choke point for
  every function-of-``S`` apply. The drift formula is inherited, not restated.

Both are matrix-free-route classes and refuse a dense backend, where every apply is exact and a
degree would be a label on nothing.

**What is not governed here.**
* ``c_in`` is a Krylov solve, not a Chebyshev apply. Its inner ``v_F`` degree follows
  ``ColoredForward.K``, and ``K = None`` there selects the tree's adaptive branch.
* ``VNNBody``'s Mehler envelope already uses an adaptive degree of its own.

Every resolved degree can be logged (``degree_log``), so a census reads the degrees actually used
rather than the policy's intent.

**Calibration does not transfer between datasets.**

* **The class defaults are the tree's own adaptive rule:** ``tol`` 1e-4, coefficients in the apply's
  dtype. On ``heat_d2048_e0`` and ``bandpass_d2048_e0`` that rule left the smooth functions 4–150× less
  accurate than ``K = 256``.
* **Tightening ``tol`` in float32 does not help.** The coefficient tail saturates near 1e-7, so a tight
  ``tol`` resolves every function to ``k_max``.
* **What held on those instances** was ``degree_dtype=torch.float64``, ``tol=1e-7``, and ``c_out`` and
  ``terminal`` at 256. It is ``DEFAULT_POLICY`` in ``scripts/gfs_matvec_fast.py``. Another dataset
  calibrates its own with a tolerance ladder (that script's ``--ladder``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ccd.denoiser.edm import ColoredEDM
from ccd.forward.marginals import ColoredForward
from ccd.operators.chebyshev import apply_fun, chebyshev_degree
from ccd.operators.linop import CovarianceOperator
from ccd.reverse.learned import LearnedReversePFODE

__all__ = ["DegreePolicy", "PolicyColoredEDM", "PolicyLearnedReversePFODE", "policy_apply"]


@dataclass(frozen=True)
class DegreePolicy:
    """``c_out`` at a fixed degree; every other named function fixed or adaptive.

    Args:
        c_out: the fixed Chebyshev degree for ``c_out``.
        smooth: the degree for every other function; ``None`` resolves it per call with
            :func:`chebyshev_degree` at ``tol``/``k_max``.
        tol: coefficient-tail tolerance of the adaptive rule (the tree's default, sized for float32).
        k_max: the adaptive rule's cap.
        terminal: a fixed degree for the ``terminal`` apply (``S^{1/2}``, once per cloud); ``None``
            gives it ``smooth``.
        degree_dtype: the dtype the adaptive rule computes coefficients in; ``None`` uses the apply's.
            In float32 the coefficient tail flattens near the rounding floor (~1e-7), so a ``tol``
            near it resolves rounding noise rather than truncation.
    """

    c_out: int = 256
    smooth: int | None = None
    tol: float = 1e-4
    k_max: int = 256
    terminal: int | None = None
    degree_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if int(self.c_out) < 1:
            raise ValueError(f"c_out degree must be >= 1; got {self.c_out}")
        if self.smooth is not None and int(self.smooth) < 1:
            raise ValueError(f"smooth degree must be >= 1 or None; got {self.smooth}")
        if self.terminal is not None and int(self.terminal) < 1:
            raise ValueError(f"terminal degree must be >= 1 or None; got {self.terminal}")
        if not (0.0 < float(self.tol) < 1.0):
            raise ValueError(f"tol must lie in (0, 1); got {self.tol}")
        if int(self.k_max) < 1:
            raise ValueError(f"k_max must be >= 1; got {self.k_max}")
        if self.degree_dtype not in (None, torch.float32, torch.float64):
            raise ValueError(f"degree_dtype must be None, float32 or float64; got {self.degree_dtype}")

    def degree_for(self, name: str) -> int | None:
        """The policy's degree for ``name``: fixed for ``c_out`` (and ``terminal`` if set), else ``smooth``."""
        if name == "c_out":
            return int(self.c_out)
        if name == "terminal" and self.terminal is not None:
            return int(self.terminal)
        return self.smooth

    def resolve(self, name: str, f, lo: float, hi: float, *, dtype, device) -> int:
        """The degree actually used for ``f`` under ``name``: fixed, or ``apply_fun``'s adaptive rule.

        The adaptive rule reads the coefficient tail in ``degree_dtype`` when set, else in the apply's
        ``dtype``. Only the degree changes; the apply itself stays in the vector's dtype.
        """
        K = self.degree_for(name)
        if K is None:
            K = max(chebyshev_degree(f, lo, hi, tol=float(self.tol), k_max=int(self.k_max),
                                     dtype=self.degree_dtype or dtype, device=device), 1)
        return int(K)


def policy_apply(policy: DegreePolicy, name: str, f, op: CovarianceOperator, v: Tensor,
                 lo: float, hi: float, *, log: list | None = None, F: float | None = None) -> Tensor:
    """``f(S) v`` by Chebyshev at the degree ``policy`` resolves for ``name``, logged if asked.

    With the policy's defaults an adaptive resolution is the same computation as
    ``apply_fun(f, op, v, lo, hi, None)``: the same rule, tolerance and cap, then the same apply.
    """
    K = policy.resolve(name, f, lo, hi, dtype=v.dtype, device=v.device)
    if log is not None:
        log.append({"name": name, "F": None if F is None else float(F), "K": K})
    return apply_fun(f, op, v, lo, hi, K)


class PolicyColoredEDM(ColoredEDM):
    """:class:`ColoredEDM` on the matrix-free route with a per-function Chebyshev degree."""

    def __init__(self, fwd: ColoredForward, *, policy: DegreePolicy | None = None,
                 degree_log: list | None = None):
        if fwd.backend is not None:
            raise ValueError(
                "PolicyColoredEDM is a matrix-free-route class; this ColoredForward carries a dense "
                "backend, on which every apply is exact and a Chebyshev degree means nothing")
        super().__init__(fwd, precond_backend=None, solve_backend=None)
        self.policy = policy or DegreePolicy()
        self.degree_log = degree_log

    def _named(self, name: str, fn, v: Tensor, F: float) -> Tensor:
        return policy_apply(self.policy, name, fn, self.fwd.op, v, self.fwd.lo, self.fwd.hi,
                            log=self.degree_log, F=F)

    def _apply_precond(self, fn, v: Tensor) -> Tensor:
        raise RuntimeError(
            "PolicyColoredEDM routes each preconditioner through its NAMED degree; the parent's "
            "shared _apply_precond would apply one fwd.K to all of them. A new caller of it must be "
            "given a name here rather than falling back silently.")

    def c_skip(self, v: Tensor, F: float) -> Tensor:
        self.fwd._require_positive_bracket(F, "c_skip")
        return self._named("c_skip", self._cskip_fn(F), v, F)

    def c_out(self, v: Tensor, F: float) -> Tensor:
        self.fwd._require_positive_bracket(F, "c_out")
        return self._named("c_out", self._cout_fn(F), v, F)

    def c_out_sq(self, v: Tensor, F: float) -> Tensor:
        self.fwd._require_positive_bracket(F, "c_out_sq")
        return self._named("c_out_sq", self._cout_sq_fn(F), v, F)


class PolicyLearnedReversePFODE(LearnedReversePFODE):
    """:class:`LearnedReversePFODE` whose drift applies take the policy's non-``c_out`` degree."""

    def __init__(self, denoiser, *, policy: DegreePolicy | None = None,
                 c_in_kw: dict | None = None, degree_log: list | None = None):
        super().__init__(denoiser, K=None, c_in_kw=c_in_kw, backend=None)
        if self.fwd.backend is not None:
            raise ValueError(
                "PolicyLearnedReversePFODE is a matrix-free-route class; the denoiser's "
                "ColoredForward carries a dense backend")
        self.policy = policy or DegreePolicy()
        self.degree_log = degree_log

    def _mean_fn(self, F: float):
        f = super()._mean_fn(F)
        f._ccd_degree_name = "m_F"
        return f

    def _a_fn(self, F: float):
        f = super()._a_fn(F)
        f._ccd_degree_name = "a_F"
        return f

    def _apply(self, f, v: Tensor) -> Tensor:
        name = getattr(f, "_ccd_degree_name", "phi" if f is self.phi else "drift")
        return policy_apply(self.policy, name, f, self.op, v, self.lo, self.hi, log=self.degree_log)
