"""``S = Ĉ_n + τI`` formed ONCE as a dense ``d×d`` matrix — eigendecomposition-free, not matrix-free.

:class:`~ccd.operators.linop.CovarianceOperator` applies ``S`` through the data factor,
``v ↦ X (Xᵀ v)/n + τ v``, which costs about ``2·d·n`` per column and never forms ``S``. That is the
right access pattern when a ``d×d`` matrix does not fit. When it does (``d = 2,048`` is 16 MiB in
float32), one dense product ``S v`` costs ``d²`` per column instead: ``d/(2n)`` of the factor route,
so 0.51× at ``n = 2,000`` and 0.13× at ``n = 8,000``.

:class:`DenseGramOperator` subclasses the factor operator, so every consumer (``apply_fun``, the
Krylov solvers, ``VNNBody._build_graph``) takes it unchanged and only :meth:`matvec` differs. ``X`` is
kept, so the data-factor terminal draw still works.

⚠️ **Eigendecomposition-free, NOT matrix-free.** Nothing here diagonalises ``S``: the matrix is formed
by one ``X Xᵀ`` product. Beyond the ``d`` where a ``d×d`` matrix fits, the operator to reach for is a
sparse graph shift, which is a different object from this one.

**Portability.** This class needs the data factor ``X``. An operator given only as a graph shift (a
sparse GSO, for instance) has no ``X`` to form ``S`` from, so it keeps its own :meth:`matvec`.
:mod:`ccd.degree_policy` and :mod:`ccd.factored_cin` do not depend on this class.

The matrix is formed in ``build_dtype`` (float64 by default), symmetrised there, and then cast to
``X``'s dtype. The stored ``S`` is therefore exactly symmetric, and its entries carry one rounding
rather than the factor route's per-call reduction order.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ccd.operators.linop import CovarianceOperator


class DenseGramOperator(CovarianceOperator):
    """``S = (1/n) X Xᵀ + τI`` held densely; :meth:`matvec` is one ``d×d`` product.

    Args:
        X: ``(d, n)`` data matrix, as for :class:`CovarianceOperator`.
        tau: ridge ``τ > 0``.
        build_dtype: dtype ``S`` is formed and symmetrised in before the cast to ``X.dtype``.
    """

    def __init__(self, X: Tensor, tau: float, *, build_dtype: torch.dtype = torch.float64):
        super().__init__(X, tau)
        Xb = X.to(dtype=build_dtype)
        S = (Xb @ Xb.transpose(-2, -1)) / self.n
        S = 0.5 * (S + S.transpose(-2, -1))
        S.diagonal().add_(self.tau)
        self.S = S.to(dtype=X.dtype).contiguous()
        self.build_dtype = build_dtype

    @classmethod
    def from_operator(cls, op: CovarianceOperator, *,
                      build_dtype: torch.dtype = torch.float64) -> "DenseGramOperator":
        """The dense twin of an existing factor operator: the same ``X``, the same ``τ``."""
        return cls(op.X, op.tau, build_dtype=build_dtype)

    @property
    def flops_ratio_vs_factor(self) -> float:
        """Multiply-adds per column of one dense matvec over the factor route's: ``d / (2n)``."""
        return self.d / (2.0 * self.n)

    def matvec(self, v: Tensor) -> Tensor:
        """Apply ``S`` to ``v`` of shape ``(d,)`` or ``(d, B)`` as one dense product."""
        if v.shape[0] != self.d:
            raise ValueError(f"v must have leading dim d={self.d}; got {tuple(v.shape)}")
        self._matvec_count += 1
        return self.S @ v
