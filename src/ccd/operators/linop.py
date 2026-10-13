"""The covariance shift operator ``S = Ĉ_n + τI`` as a matvec oracle (Phase I0).

``S`` is the regularised empirical covariance of centred data ``x0 ∈ R^d`` with
``Ĉ_n = (1/n) X Xᵀ`` (``X ∈ R^{d×n}``, columns are the ``n`` samples) and ridge
``τ > 0``. Every forward colored-diffusion operator is a function of this single
matrix, so the *only* access pattern the rest of the package needs is the
matrix–vector product

    v ↦ S v = (1/n) X (Xᵀ v) + τ v      (cost O(nd), no d×d matrix formed).

``_dense`` materialises ``S`` for the test-only eigendecomposition baseline and is
guarded to small ``d`` — it must never appear on a production code path.
"""

from __future__ import annotations

import torch
from torch import Tensor


class CovarianceOperator:
    """SPD operator ``S = (1/n) X Xᵀ + τI`` accessed only through :meth:`matvec`.

    Args:
        X: ``(d, n)`` data matrix (columns are the ``n`` centred samples).
        tau: ridge ``τ > 0`` (also the exact smallest-eigenvalue lower bound of ``S``).

    The instance counts its matvecs (:attr:`matvec_count`, :meth:`reset_count`) so
    tests can assert matrix-free cost budgets.
    """

    def __init__(self, X: Tensor, tau: float):
        if X.dim() != 2:
            raise ValueError(f"X must be 2-D (d, n); got shape {tuple(X.shape)}")
        if tau <= 0:
            raise ValueError(f"tau must be > 0; got {tau}")
        self.X = X
        self.d, self.n = int(X.shape[0]), int(X.shape[1])
        self.tau = float(tau)
        self._matvec_count = 0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.d, self.d)

    @property
    def dtype(self) -> torch.dtype:
        return self.X.dtype

    @property
    def device(self) -> torch.device:
        return self.X.device

    @property
    def matvec_count(self) -> int:
        return self._matvec_count

    def reset_count(self) -> None:
        self._matvec_count = 0

    def matvec(self, v: Tensor) -> Tensor:
        """Apply ``S`` to ``v`` of shape ``(d,)`` or ``(d, B)``."""
        if v.shape[0] != self.d:
            raise ValueError(f"v must have leading dim d={self.d}; got {tuple(v.shape)}")
        self._matvec_count += 1
        return (self.X @ (self.X.transpose(-2, -1) @ v)) / self.n + self.tau * v

    def _dense(self, *, max_d: int = 1024) -> Tensor:
        """Materialise the dense ``d×d`` ``S`` — TEST/ORACLE ONLY (guarded to small d)."""
        if self.d > max_d:
            raise RuntimeError(f"_dense is test-only; d={self.d} exceeds max_d={max_d}")
        eye = torch.eye(self.d, dtype=self.dtype, device=self.device)
        return (self.X @ self.X.transpose(-2, -1)) / self.n + self.tau * eye


def make_synthetic_operator(
    n: int,
    d: int,
    *,
    tau: float = 0.05,
    decay: float = 1.0,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> CovarianceOperator:
    """Build a :class:`CovarianceOperator` with a controlled power-law spectrum.

    ``Ĉ_n ≈ Q diag(s) Qᵀ`` with ``s_i = (i+1)^(−decay)`` (normalised so ``s_0 = 1``)
    and ``Q`` a random orthogonal matrix; returns ``S = Ĉ_n + τI``. For tests and the
    efficiency study. (``decay`` larger ⇒ more ill-conditioned ``Ĉ_n``.)
    """
    A = torch.randn(d, d, generator=generator, dtype=dtype, device=device)
    Q, _ = torch.linalg.qr(A)
    idx = torch.arange(d, dtype=dtype, device=device)
    s = (idx + 1.0) ** (-float(decay))
    s = s / s[0]
    c_half = Q @ (torch.sqrt(s)[:, None] * Q.transpose(-2, -1))  # symmetric SPD factor of C
    Z = torch.randn(d, n, generator=generator, dtype=dtype, device=device)
    X = c_half @ Z  # so (1/n) X Xᵀ ≈ C = Q diag(s) Qᵀ
    return CovarianceOperator(X, tau)
